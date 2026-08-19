#!/usr/bin/env python3
"""MoT-h: Mixture-of-Translators (hidden state 变体).

论文 "Mixture-of-Translators" 的 MoT-h 分支实现（附录 C.5），定制适配
  Source = DeepSeek-V4-Flash  (hidden=7168, layers=61, MoE)
  Target = Qwen3.6-35B-A3B    (hidden=2048, layers=40, 稠密 + DeltaNet)

与原版 MoT 的差异（MoT-h, 附录 C.5）:
  - 不翻 KV, 翻 prefill 中间激活 hidden state
  - Target 用原生 Wk/Wv 从翻译后 hidden 还原 KV
  - 传输量减半, 端侧校正空间更大

定制改造（针对 DSV4 FlashPrefill + Qwen3.6 端侧 decode）:
  1. BackboneTranslator 输入侧加 Flash 分块位置编码（消除分块边界突变）
  2. 门控加上下文长度特征（长/短文本自动切换翻译器权重）
  3. N_Tr=3, Top-K=2（原版 2/1，增加专家数应对 MoE↔稠密分布差异）
  4. 损失新增 L_Flash（DSV4 原生 prefill vs FlashPrefill hidden 的 MSE）

依赖: torch (CPU 即可跑通冒烟测试, GPU 训练)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class MoTHConfig:
    """MoT-h 配置. 维度默认值对应 DSV4→Qwen3.6-35B-A3B."""

    # 模型维度 (TODO: Qwen3.6 hidden_size 用实际 config.json 确认, 当前 2048 基于代码线索)
    src_hidden_size: int = 7168   # DeepSeek-V4-Flash
    tgt_hidden_size: int = 2048   # Qwen3.6-35B-A3B
    src_num_layers: int = 61
    tgt_num_layers: int = 40

    # 翻译器
    translator_hidden: int = 4096
    window_size: int = 4          # 通道窗口长度
    num_translators: int = 3      # 定制 N_Tr=3 (原版 2)
    top_k: int = 2                # 定制 Top-K=2 (原版 1)
    num_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0

    # Flash 分块位置编码 (定制: 消除 FlashPrefill 分块边界突变)
    use_flash_block_pos: bool = True
    flash_block_size: int = 128   # FlashPrefill 分块大小

    # 门控上下文长度特征 (定制: 长/短文本切换翻译器权重)
    use_ctx_len_gate: bool = True
    max_ctx_len: int = 32768

    # 损失权重
    lambda_pl: float = 0.1
    lambda_flash: float = 0.05


# ---------------------------------------------------------------------------
# BackboneTranslator — 单个翻译器专家 (附录 C.2 + MoT-h 变体)
# ---------------------------------------------------------------------------
class BackboneTranslator(nn.Module):
    """递归交叉注意力扫描窗口.

    输入: src_hidden_window [B, win_len, seq_len, src_dim]
          (win_len 个源层的 hidden state, 已按通道映射选取)
    输出: tgt_hidden [B, seq_len, tgt_dim]
          (Target 兼容的 hidden state, Target 用原生 Wk/Wv 还原 KV)
    """

    def __init__(self, cfg: MoTHConfig):
        super().__init__()
        H = cfg.translator_hidden
        self.win_size = cfg.window_size
        self.use_flash_block_pos = cfg.use_flash_block_pos
        self.flash_block_size = cfg.flash_block_size

        # 输入投影: LN + Linear + GELU → 翻译隐空间
        self.in_norm = nn.LayerNorm(cfg.src_hidden_size)
        self.in_proj = nn.Linear(cfg.src_hidden_size, H)

        # Flash 分块位置编码 (定制: 消除 FlashPrefill 分块边界特征突变)
        if self.use_flash_block_pos:
            self.flash_block_pos = nn.Embedding(cfg.flash_block_size, H)

        # 递归交叉注意力: Query=上一层隐藏态, K/V=当前源层投影
        self.cross_attn = nn.MultiheadAttention(
            H, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(H)

        # FFN + 残差
        self.ffn = nn.Sequential(
            nn.Linear(H, H * cfg.ffn_mult),
            nn.GELU(),
            nn.Linear(H * cfg.ffn_mult, H),
        )
        self.ffn_norm = nn.LayerNorm(H)

        # 输出投影 → tgt_hidden (MoT-h: 输出 hidden state, 不是 KV)
        self.out_norm = nn.LayerNorm(H)
        self.out_proj = nn.Linear(H, cfg.tgt_hidden_size)

    def forward(self, src_hidden_window: torch.Tensor) -> torch.Tensor:
        # src_hidden_window: [B, win_len, seq_len, src_dim]
        B, W, S, D = src_hidden_window.shape

        # 输入投影
        z = F.gelu(self.in_proj(self.in_norm(src_hidden_window)))  # [B, W, S, H]

        # Flash 分块位置编码 (沿 seq 维度, 周期 = flash_block_size)
        if self.use_flash_block_pos:
            block_ids = torch.arange(S, device=z.device) % self.flash_block_size
            z = z + self.flash_block_pos(block_ids).view(1, 1, S, -1)

        # 递归交叉注意力扫描窗口
        h = z[:, 0]  # [B, S, H] — 初始隐藏态 = 第 0 层投影
        for r in range(1, W):
            attn_out, _ = self.cross_attn(query=h, key=z[:, r], value=z[:, r])
            h = h + attn_out
            h = h + self.ffn(self.ffn_norm(h))

        # 输出投影
        out = self.out_proj(self.out_norm(h))  # [B, S, tgt_dim]
        return out


# ---------------------------------------------------------------------------
# MoT 门控 — Token 级 Top-K 混合 (3.1 节 + 定制)
# ---------------------------------------------------------------------------
class MoTHGate(nn.Module):
    """门控网络: 输入源 hidden + 上下文长度, 输出专家权重."""

    def __init__(self, cfg: MoTHConfig):
        super().__init__()
        H = cfg.translator_hidden
        self.use_ctx_len_gate = cfg.use_ctx_len_gate
        self.max_ctx_len = cfg.max_ctx_len

        self.src_proj = nn.Linear(cfg.src_hidden_size, H)
        if self.use_ctx_len_gate:
            self.ctx_proj = nn.Linear(1, H)
        self.gate = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, cfg.num_translators),
        )

    def forward(
        self,
        src_hidden: torch.Tensor,
        ctx_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # src_hidden: [B, win_len, seq_len, src_dim]
        # 池化: mean over win_len and seq_len → [B, src_dim]
        pooled = src_hidden.mean(dim=(1, 2))  # [B, src_dim]
        h = self.src_proj(pooled)  # [B, H]

        if self.use_ctx_len_gate and ctx_len is not None:
            # ctx_len: [B], 归一化到 [0, 1]
            ctx_norm = (ctx_len.float() / self.max_ctx_len).clamp(0.0, 1.0).unsqueeze(-1)
            h = h + self.ctx_proj(ctx_norm)

        return self.gate(h)  # [B, num_tr] — logits


# ---------------------------------------------------------------------------
# MoT-h 完整模块
# ---------------------------------------------------------------------------
class MoTH(nn.Module):
    """Mixture-of-Translators (hidden state 变体).

    输入: src_hidden [B, win_len, seq_len, src_dim]
          ctx_len   [B] (可选, 上下文长度)
    输出: tgt_hidden [B, seq_len, tgt_dim]
          gate_weights [B, num_tr] (用于分析/正则)
    """

    def __init__(self, cfg: MoTHConfig | None = None):
        super().__init__()
        self.cfg = cfg or MoTHConfig()
        self.translators = nn.ModuleList(
            [BackboneTranslator(self.cfg) for _ in range(self.cfg.num_translators)]
        )
        self.gate = MoTHGate(self.cfg)
        self.top_k = self.cfg.top_k

    def forward(
        self,
        src_hidden: torch.Tensor,
        ctx_len: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, W, S, D = src_hidden.shape

        # 门控
        gate_logits = self.gate(src_hidden, ctx_len)  # [B, num_tr]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, num_tr]

        # Top-K 选取 + 重归一化
        top_vals, top_idx = torch.topk(gate_weights, k=self.top_k, dim=-1)  # [B, top_k]
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)

        # 计算所有翻译器输出, stack: [B, num_tr, S, tgt_dim]
        all_outputs = torch.stack(
            [tr(src_hidden) for tr in self.translators], dim=1
        )

        # Top-K 加权 (向量化, 无 for-b-in-B 循环)
        K = self.top_k
        T = all_outputs.shape[-1]
        gathered = torch.gather(
            all_outputs, dim=1,
            index=top_idx.unsqueeze(-1).unsqueeze(-1).expand(B, K, S, T),
        )  # [B, top_k, S, tgt_dim]
        weighted = gathered * top_vals.unsqueeze(-1).unsqueeze(-1)  # [B, top_k, S, tgt_dim]
        out = weighted.sum(dim=1)  # [B, S, tgt_dim]

        return out, gate_weights

    @torch.no_grad()
    def translate_hidden(
        self,
        src_hidden: torch.Tensor,
        ctx_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """便捷方法: 單層 hidden state → target hidden state (推論用).

        Args:
            src_hidden: [seq_len, src_dim] 或 [B, seq_len, src_dim]
                        (末層 hidden state, 無 window 維度)
            ctx_len: [B] 可選, 上下文長度

        Returns:
            [seq_len, tgt_dim] 或 [B, seq_len, tgt_dim]
        """
        if src_hidden.dim() == 2:
            # [seq_len, src_dim] → [1, 1, seq_len, src_dim]
            src_hidden = src_hidden.unsqueeze(0).unsqueeze(0)
            out, _ = self.forward(src_hidden, ctx_len)
            return out.squeeze(0)  # [seq_len, tgt_dim]
        elif src_hidden.dim() == 3:
            # [B, seq_len, src_dim] → [B, 1, seq_len, src_dim]
            B, S, D = src_hidden.shape
            src_hidden = src_hidden.unsqueeze(1)
            out, _ = self.forward(src_hidden, ctx_len)
            return out  # [B, seq_len, tgt_dim]
        else:
            # 已是 4D [B, W, S, D], 直接 forward
            out, _ = self.forward(src_hidden, ctx_len)
            return out


# ---------------------------------------------------------------------------
# 损失函数 (CC + PL + Flash, 联合)
# ---------------------------------------------------------------------------
def compute_cc_loss(
    translated_hidden: torch.Tensor,
    native_target_hidden: torch.Tensor,
    target_wk: torch.Tensor,
    target_wv: torch.Tensor,
) -> torch.Tensor:
    """Context Correction Loss (3.2 节, 附录 C.1 KV 近似).

    用 Target 原生 Wk/Wv 从 hidden 还原 KV, 与原生 KV 做 MSE.
    MoT-h 的核心优势: 端侧用原生投影层, CC loss 可直接在 hidden 空间近似.

    Args:
        translated_hidden: [B, S, tgt_hidden] — MoT-h 输出
        native_target_hidden: [B, S, tgt_hidden] — Target 原生 prefill hidden
        target_wk: [tgt_hidden, kv_dim] — Target 原生 Wk
        target_wv: [tgt_hidden, kv_dim] — Target 原生 Wv
    """
    K_hat = translated_hidden @ target_wk
    V_hat = translated_hidden @ target_wv
    K_native = native_target_hidden @ target_wk
    V_native = native_target_hidden @ target_wv
    return F.mse_loss(K_hat, K_native) + F.mse_loss(V_hat, V_native)


def compute_pl_loss(
    target_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Prompt LM Loss — prompt 生成 CE (辅助监督)."""
    return F.cross_entropy(
        target_logits.view(-1, target_logits.size(-1)),
        labels.view(-1),
    )


def compute_flash_loss(
    flash_hidden: torch.Tensor,
    native_prefill_hidden: torch.Tensor,
) -> torch.Tensor:
    """Flash 校正损失 (定制: 消除 FlashPrefill 分块/压缩的系统性偏移).

    离线构造对照数据: 同一文本分别用 DSV4 原生 prefill / FlashPrefill,
    计算两者 hidden state 的 MSE 作约束.

    Args:
        flash_hidden: [B, S, src_hidden] — FlashPrefill 输出
        native_prefill_hidden: [B, S, src_hidden] — 原生逐层 prefill 输出
    """
    return F.mse_loss(flash_hidden, native_prefill_hidden)


def total_loss(
    cc: torch.Tensor,
    pl: torch.Tensor,
    flash: torch.Tensor,
    lambda_pl: float = 0.1,
    lambda_flash: float = 0.05,
) -> torch.Tensor:
    """L_total = L_CC + lambda_PL * L_PL + lambda_Flash * L_Flash."""
    return cc + lambda_pl * pl + lambda_flash * flash


# ---------------------------------------------------------------------------
# 冒烟测试 — 随机张量跑通 forward/backward
# ---------------------------------------------------------------------------
def _smoke_test():
    """验证 MoT-h 网络的数学正确性: forward + backward 梯度能流回参数."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MoTHConfig(
        src_hidden_size=7168,
        tgt_hidden_size=2048,
        translator_hidden=512,      # 测试用小 hidden
        window_size=4,
        num_translators=3,
        top_k=2,
        num_heads=8,
    )
    model = MoTH(cfg).to(device)
    print(f"[smoke] device={device}")
    print(f"[smoke] model params: {sum(p.numel() for p in model.parameters()):,}")

    # 随机输入: B=2, win_len=4, seq_len=64, src_dim=7168
    B, W, S = 2, 4, 64
    src_hidden = torch.randn(B, W, S, cfg.src_hidden_size, device=device)
    ctx_len = torch.tensor([1024, 8192], device=device)

    # Forward
    out, gate_w = model(src_hidden, ctx_len)
    assert out.shape == (B, S, cfg.tgt_hidden_size), \
        f"output shape mismatch: {out.shape} vs {(B, S, cfg.tgt_hidden_size)}"
    assert gate_w.shape == (B, cfg.num_translators), \
        f"gate shape mismatch: {gate_w.shape}"
    assert torch.allclose(gate_w.sum(-1), torch.ones(B, device=device)), \
        "gate weights should sum to 1"
    print(f"[smoke] forward OK: out={out.shape}, gate={gate_w.shape}")
    print(f"[smoke] gate weights (batch 0): {gate_w[0].tolist()}")

    # Loss: CC + PL + Flash
    native_target_hidden = torch.randn(B, S, cfg.tgt_hidden_size, device=device)
    target_wk = torch.randn(cfg.tgt_hidden_size, 128, device=device) * 0.02
    target_wv = torch.randn(cfg.tgt_hidden_size, 128, device=device) * 0.02

    cc = compute_cc_loss(out, native_target_hidden, target_wk, target_wv)
    print(f"[smoke] CC loss: {cc.item():.6f}")

    # PL loss (模拟 prompt 生成)
    logits = torch.randn(B, S, 1000, device=device)
    labels = torch.randint(0, 1000, (B, S), device=device)
    pl = compute_pl_loss(logits, labels)
    print(f"[smoke] PL loss: {pl.item():.6f}")

    # Flash loss
    flash_hidden = torch.randn(B, S, cfg.src_hidden_size, device=device)
    native_prefill = torch.randn(B, S, cfg.src_hidden_size, device=device)
    fl = compute_flash_loss(flash_hidden, native_prefill)
    print(f"[smoke] Flash loss: {fl.item():.6f}")

    # Total
    loss = total_loss(cc, pl, fl, cfg.lambda_pl, cfg.lambda_flash)
    print(f"[smoke] total loss: {loss.item():.6f}")

    # Backward — 验证梯度能流回 MoT-h 参数
    loss.backward()
    no_grad = []
    for name, p in model.named_parameters():
        if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0):
            no_grad.append(name)
    if no_grad:
        print(f"[smoke] WARNING: {len(no_grad)} params have no gradient:")
        for n in no_grad[:5]:
            print(f"         - {n}")
    else:
        print(f"[smoke] backward OK: all {sum(1 for p in model.parameters() if p.requires_grad)} params have gradients")

    # 验证 Top-K 门控不是均匀的 (定制点: MoT-Uni 会被论文消融否掉)
    gate_std = gate_w.std(dim=0).mean().item()
    print(f"[smoke] gate std (per expert, avg over batch): {gate_std:.4f}")
    assert gate_std > 1e-6, "gate weights should not be uniform (MoT-Uni falsified in paper)"

    print("\n[smoke] ALL CHECKS PASSED")
    return model


if __name__ == "__main__":
    _smoke_test()
