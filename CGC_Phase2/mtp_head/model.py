"""MTP Head 模型定义 (参考 DeepSeek-V3 论文).

用于 Qwen3-VL-2B 的轻量 Multi-Token Prediction head:
- 输入: base model 最后一层 hidden_state + 当前 token embedding
- 输出: next_token logits
- 参数量: ~48M (不含 shared lm_head/embedding)
- forward: ~1ms on Metal GPU

架构:
  Concat(hidden, embed) → Linear → RMSNorm → Attention → RMSNorm → MLP → RMSNorm → shared lm_head
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MTPHeadConfig:
    """MTP Head 配置 (对齐 Qwen3-VL-2B)."""
    hidden_size: int = 2048          # Qwen3-VL-2B text_config.hidden_size
    vocab_size: int = 151936         # Qwen3-VL-2B vocab_size
    num_heads: int = 16              # attention heads (head_dim=128)
    head_dim: int = 128              # hidden_size // num_heads
    intermediate_size: int = 5632    # MLP intermediate (2.75x hidden, Qwen3 标准)
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0    # Qwen3 default
    max_position_embeddings: int = 40960
    # shared weights (不训练, 从 base model 加载)
    share_embedding: bool = True
    share_lm_head: bool = True


class MTPRMSNorm(nn.Module):
    """RMSNorm (对齐 Qwen3)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class MTPAttention(nn.Module):
    """Multi-head attention with RoPE (对齐 Qwen3)."""

    def __init__(self, config: MTPHeadConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)

        # RoPE (预计算 cos/sin)
        self._rope_cached: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    def _get_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._rope_cached is not None:
            cos_cached, sin_cached = self._rope_cached
            if cos_cached.size(0) >= seq_len:
                return cos_cached[:seq_len], sin_cached[:seq_len]

        # Qwen3 RoPE
        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        self._rope_cached = (cos, sin)
        return cos, sin

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [batch, heads, seq, head_dim]
        # cos/sin: [seq, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        # rotate_half
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape

        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._get_rope(seq, x.device, x.dtype)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.o_proj(attn)


class MTPMLP(nn.Module):
    """SwiGLU MLP (对齐 Qwen3)."""

    def __init__(self, config: MTPHeadConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MTPHead(nn.Module):
    """Multi-Token Prediction Head (1 层 transformer).

    输入:
        hidden_states: [batch, seq, hidden_size] (from base model last layer)
        token_embeddings: [batch, seq, hidden_size] (current token embeddings)

    输出:
        logits: [batch, seq, vocab_size] (next token prediction)

    参数量: ~48M (不含 shared lm_head/embedding)
    """

    def __init__(self, config: MTPHeadConfig):
        super().__init__()
        self.config = config

        # Projection: concat(hidden, embed) → hidden
        self.proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        # 1 层 transformer
        self.norm1 = MTPRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = MTPAttention(config)
        self.norm2 = MTPRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MTPMLP(config)
        self.norm_out = MTPRMSNorm(config.hidden_size, config.rms_norm_eps)

        # shared lm_head (从 base model 加载, 不训练)
        self.lm_head: Optional[nn.Linear] = None
        # LoRA adapter for lm_head (当 lm_head 不可用时, 用低秩 adapter 加速训练)
        self.lm_head_lora_a: Optional[nn.Linear] = None
        self.lm_head_lora_b: Optional[nn.Linear] = None

    def set_shared_lm_head(self, lm_head_weight: torch.Tensor, trainable: bool = False, lora_rank: int = 0):
        """设置共享 lm_head 权重.

        Args:
            lm_head_weight: [vocab, hidden] 权重矩阵
            trainable: True 时 lm_head 可训练 (用于无法获取真实权重的场景, 如 gated repo)
            lora_rank: >0 时用 LoRA adapter 替代全量 lm_head 训练 (大幅减少可训练参数)
        """
        vocab_size, hidden_size = lm_head_weight.size(0), lm_head_weight.size(1)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(lm_head_weight)
        self.lm_head.weight.requires_grad = False  # base 始终冻结

        if lora_rank > 0:
            # LoRA: W' = W + B @ A, A 用高斯初始化, B 用零初始化 (初始 = W)
            self.lm_head_lora_a = nn.Linear(hidden_size, lora_rank, bias=False)
            self.lm_head_lora_b = nn.Linear(lora_rank, vocab_size, bias=False)
            nn.init.normal_(self.lm_head_lora_a.weight, std=0.02)
            nn.init.zeros_(self.lm_head_lora_b.weight)
            print(f"[MTPHead] LoRA lm_head: rank={lora_rank}, "
                  f"params={lora_rank * (hidden_size + vocab_size) / 1e6:.1f}M", flush=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden] (base model 最后一层输出)
            token_embeddings: [batch, seq, hidden] (当前 token 的 embedding)

        Returns:
            logits: [batch, seq, vocab_size]
        """
        # Concat + projection
        x = torch.cat([hidden_states, token_embeddings], dim=-1)
        x = self.proj(x)

        # Transformer layer (pre-norm)
        # Attention
        h = x + self.attn(self.norm1(x), mask)
        # MLP
        h = h + self.mlp(self.norm2(h))
        # Output norm
        h = self.norm_out(h)

        # Shared lm_head (+ optional LoRA adapter)
        if self.lm_head is None:
            raise RuntimeError("lm_head not set. Call set_shared_lm_head() first.")
        logits = self.lm_head(h)
        if self.lm_head_lora_b is not None:
            # LoRA delta: B @ A(h)
            logits = logits + self.lm_head_lora_b(self.lm_head_lora_a(h))
        return logits

    def num_parameters(self) -> int:
        """可训练参数量."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_mtp_head_for_qwen3vl_2b() -> MTPHead:
    """创建 Qwen3-VL-2B 的 MTP head."""
    config = MTPHeadConfig(
        hidden_size=2048,
        vocab_size=151936,
        num_heads=16,
        head_dim=128,
        intermediate_size=5632,
    )
    return MTPHead(config)


def create_mtp_head_for_gemma4_26b() -> MTPHead:
    """创建 Gemma4-26B-A4B 的 MTP head.

    Gemma4-26B-A4B config:
      hidden_size=2816, vocab_size=262144, head_dim=256
      num_attention_heads=16, num_key_value_heads=8 (GQA)
      intermediate_size=14336 (SwiGLU)
      rope_theta=1000000.0
      rms_norm_eps=1e-6
      EOS tokens: [1, 106]
    """
    config = MTPHeadConfig(
        hidden_size=2816,
        vocab_size=262144,
        num_heads=16,
        head_dim=256,
        intermediate_size=14336,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
    )
    return MTPHead(config)


def create_mtp_head_for_dsv4_flash() -> MTPHead:
    """创建 DeepSeek V4 Flash 的 MTP head.

    DSV4 Flash config (from /data/models/DeepSeek-V4-Flash-UD-IQ2/config.json):
      hidden_size=4096, vocab_size=129280, head_dim=512
      num_attention_heads=64, num_key_value_heads=1 (GQA extreme)
      moe_intermediate_size=2048 (per expert, 256 experts, 6/tok)
      rope_theta=10000
      rms_norm_eps=1e-6
      EOS tokens: [1]
      num_nextn_predict_layers=1 (DSV4 原生 MTP, 此 head 为自定义替代)

    MTP head intermediate_size=11264 (2.75x hidden, 标准 MLP 比率).
    注意: head_dim=512 远大于 hidden_size/num_heads=64,
    DSV4 使用非标准 attention (Q/K/V: 4096 -> 64*512=32768 -> 4096).
    """
    config = MTPHeadConfig(
        hidden_size=4096,
        vocab_size=129280,
        num_heads=64,
        head_dim=512,
        intermediate_size=11264,
        rms_norm_eps=1e-6,
        rope_theta=10000,
    )
    return MTPHead(config)


def create_mtp_head_by_model_name(model_name: str) -> MTPHead:
    """按模型注册名创建 MTP head (统一入口).

    使用 app.shared.model_registry 获取配置, 自动创建正确的 MTP head.
    支持: gemma4, dsv4, qwen3vl (及别名).
    """
    import sys
    import os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from app.shared.model_registry import get_model_config

    cfg = get_model_config(model_name)
    config = MTPHeadConfig(
        hidden_size=cfg.hidden_size,
        vocab_size=cfg.vocab_size,
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        max_position_embeddings=cfg.max_position_embeddings,
    )
    return MTPHead(config)


if __name__ == "__main__":
    import sys

    # Test all configs
    factories = [
        ("Qwen3-VL-2B", create_mtp_head_for_qwen3vl_2b),
        ("Gemma4-26B", create_mtp_head_for_gemma4_26b),
        ("DSV4-Flash", create_mtp_head_for_dsv4_flash),
    ]
    for name, factory in factories:
        mtp = factory()
        print(f"\n=== {name} MTP Head ===")
        print(f"  参数量: {mtp.num_parameters() / 1e6:.1f}M")
        print(f"  hidden_size={mtp.config.hidden_size}, vocab={mtp.config.vocab_size}")
        print(f"  head_dim={mtp.config.head_dim}, num_heads={mtp.config.num_heads}")

        batch, seq = 2, 10
        hidden = torch.randn(batch, seq, mtp.config.hidden_size)
        embed = torch.randn(batch, seq, mtp.config.hidden_size)
        lm_head_w = torch.randn(mtp.config.vocab_size, mtp.config.hidden_size)
        mtp.set_shared_lm_head(lm_head_w)
        logits = mtp(hidden, embed)
        print(f"  forward: {tuple(hidden.shape)} -> {tuple(logits.shape)}")
        print(f"  OK")

    # Test unified entry
    print("\n=== create_mtp_head_by_model_name ===")
    for name in ["gemma4", "dsv4", "qwen3vl"]:
        mtp = create_mtp_head_by_model_name(name)
        print(f"  {name}: {mtp.num_parameters() / 1e6:.1f}M params, hidden={mtp.config.hidden_size}")
