#!/usr/bin/env python3
"""Gemma4 official MTP assistant head — Python 同構模型 (端側訓練用).

從引擎 turbo-fieldfare-github-official `LocalMTPAssistant.swift` (Metal) 逆向,
與官方 `google/gemma-4-26B-A4B-it-assistant` 的 `Gemma4AssistantForCausalLM`
架構逐算子對齊 (transformers 5.7.0 未內建該類, 故自建).

架構 (config.json):
  - backbone_hidden_size = 2816 (主模型 decode hidden 輸入)
  - hidden_size = 1024, vocab = 262144, intermediate = 8192
  - 4 層: [sliding, sliding, sliding, full]
    - sliding: head_dim=256, 16 heads, 8 KV heads, rope_theta=1e4 (default Neox)
    - full:    head_dim=512, 16 heads, 2 KV heads, rope_theta=1e6 (proportional Neox, 1/4 rotate)
  - 無 k_proj/v_proj: 共享 target 主模型的 KV (按層型別共享最後一層)
  - lm_head = tied embed_tokens
  - 單位置模型: position 固定 = 已提交 token 數 (draft 循環內不遞增)

Forward (逐算子對齊 Metal encodeStep):
  1. token_embed = embed(token); 若無 target embedding, proxy = post_projection(token_embed)
     (有 target embedding 時直接用主模型 embed_tokens 的 token embedding [2816])
  2. fused = concat([proxy, backbone_hidden])  # [2*2816]
  3. x = pre_projection(fused)                  # [1024]
  4. per layer:
     - input_layernorm(x) -> q_proj -> q_norm per-head -> RoPE -> attn(共享 KV) -> o_proj
     - post_attention_layernorm(attn_out) -> residual add (unscaled)
     - pre_feedforward_layernorm -> gate/up -> gelu_tanh -> down
     - post_feedforward_layernorm(ff) [若存在] -> residual add -> *layer_scalar (整層一次)
  5. final norm -> lm_head logits -> argmax token
  6. next backbone = post_projection(final_hidden)  # chained draft 用

權重 key (safetensors):
  model.embed_tokens.weight [262144, 1024]
  pre_projection.weight  [1024, 5632]
  post_projection.weight [2816, 1024]
  model.norm.weight      [1024]
  layers.{i}.layer_scalar                    (scalar)
  layers.{i}.input_layernorm.weight          [1024]
  layers.{i}.post_attention_layernorm.weight [1024]
  layers.{i}.self_attn.q_proj.weight         [qRows, 1024]
  layers.{i}.self_attn.q_norm.weight         [heads, head_dim]
  layers.{i}.self_attn.o_proj.weight         [1024, qRows]
  layers.{i}.mlp.gate_proj.weight [8192, 1024] / up_proj / down_proj [1024, 8192]
  layers.{i}.pre_feedforward_layernorm.weight [1024]
  layers.{i}.post_feedforward_layernorm.weight [1024]  (optional)
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import load_file


def _gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RoPE:
    """Neox 風格 RoPE (引擎 rope.metal 逐算子對齊).

    引擎 apply_neox_pair 旋轉 pair p = 元素 (p, half_dim + p) — half-split rotate,
    不是 interleaved (2i, 2i+1)!
      lower = pair; upper = half_dim + pair
      head[lower] = x0*cos - x1*sin; head[upper] = x0*sin + x1*cos
    full 層用 proportional (theta=1e6, 只旋轉前 head_dim/8 對);
    sliding 用 default (theta=1e4, 全部 half_dim 對)。
    """

    @staticmethod
    def _neox_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                     pairs: int) -> torch.Tensor:
        """[..., D]: pair p -> (p, half_dim+p), 只旋轉前 pairs 對。"""
        d = x.shape[-1]
        half = d // 2
        x1 = x[..., :pairs]             # lower half 前 pairs 個
        x2 = x[..., half:half + pairs]  # upper half 前 pairs 個
        out_low = x1 * cos - x2 * sin
        out_up = x1 * sin + x2 * cos
        return torch.cat([out_low, x[..., pairs:half], out_up, x[..., half + pairs:]], dim=-1)

    @classmethod
    def apply(cls, x: torch.Tensor, position: int, head_dim: int,
              kind: str = "sliding", num_rotated_pairs: Optional[int] = None) -> torch.Tensor:
        """x: [..., head_dim] (每頭已分開)."""
        if kind == "full":
            theta = 1_000_000.0
            pairs = num_rotated_pairs if num_rotated_pairs is not None else head_dim // 8
        else:
            theta = 10_000.0
            pairs = head_dim // 2
        # 引擎 exponent = -2*pair/head_dim; pair 索引 0..pairs-1
        dev = x.device
        freqs = 1.0 / (theta ** (torch.arange(0, 2 * pairs, 2, dtype=torch.float32) / head_dim))
        # MPS 不支援 torch.outer — 用 broadcasting; 需與 x 同 device
        cos = torch.cos(position * freqs.to(dev))  # [pairs]
        sin = torch.sin(position * freqs.to(dev))  # [pairs]
        return cls._neox_rotate(x, cos, sin, pairs)


class Gemma4Assistant(nn.Module):
    """官方 Gemma4AssistantForCausalLM 的端側同構實作 (單位置, KV 由外部傳入)。"""

    def __init__(self, config_path: str):
        super().__init__()
        with open(config_path) as f:
            raw = json.load(f)
        tc = raw["text_config"]
        self.backbone_hidden_size = raw["backbone_hidden_size"]
        self.hidden_size = tc["hidden_size"]
        self.intermediate_size = tc["intermediate_size"]
        self.vocab_size = tc["vocab_size"]
        self.num_layers = tc["num_hidden_layers"]
        self.num_heads = tc["num_attention_heads"]
        self.head_dim = tc["head_dim"]
        self.global_head_dim = tc.get("global_head_dim", tc["head_dim"])
        self.num_kv_heads = tc["num_key_value_heads"]
        self.num_global_kv_heads = tc.get("num_global_key_value_heads", 2)
        self.eps = tc.get("rms_norm_eps", 1e-6)
        self.layer_types = tc["layer_types"]
        self.sliding_window = tc.get("sliding_window", 1024)

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.pre_projection = nn.Linear(2 * self.backbone_hidden_size, self.hidden_size, bias=False)
        self.post_projection = nn.Linear(self.hidden_size, self.backbone_hidden_size, bias=False)
        self.model_norm = RMSNorm(self.hidden_size, self.eps)

        # 每層 (依 layer type 選 head_dim / kv_heads)
        layers = []
        for i in range(self.num_layers):
            is_full = self.layer_types[i] == "full_attention"
            hd = self.global_head_dim if is_full else self.head_dim
            kv = self.num_global_kv_heads if is_full else self.num_kv_heads
            layers.append(Gemma4AssistantLayer(
                hidden=self.hidden_size, intermediate=self.intermediate_size,
                heads=self.num_heads, head_dim=hd, kv_heads=kv, eps=self.eps,
            ))
        self.layers = nn.ModuleList(layers)

    def load_official_weights(self, safetensors_path: str) -> dict:
        """載入官方 safetensors, 回傳 missing/unexpected keys 清單. 權重直接覆蓋到 self。"""
        sd = load_file(safetensors_path)
        # 官方 key 前綴: model.embed_tokens / model.norm / model.layers.{i}.*
        target = self.state_dict()  # {layer_key: tensor}
        remap = {}
        # top-level
        remap["embed_tokens.weight"] = sd["model.embed_tokens.weight"]
        remap["model_norm.weight"] = sd["model.norm.weight"]
        remap["pre_projection.weight"] = sd["pre_projection.weight"]
        remap["post_projection.weight"] = sd["post_projection.weight"]
        # layers
        for i in range(self.num_layers):
            p = f"model.layers.{i}"
            layer_keys = [k for k in target if k.startswith(f"layers.{i}.")]
            for k in layer_keys:
                # k 形如 layers.0.self_attn_q_proj.weight
                suffix = k.split(".", 2)[2]  # self_attn_q_proj.weight
                src_suffix = suffix
                src_suffix = src_suffix.replace("self_attn_q_proj", "self_attn.q_proj")
                if src_suffix == "self_attn_q_norm":
                    src_suffix = "self_attn.q_norm.weight"
                src_suffix = src_suffix.replace("self_attn_o_proj", "self_attn.o_proj")
                src_suffix = src_suffix.replace("mlp_gate_proj", "mlp.gate_proj")
                src_suffix = src_suffix.replace("mlp_up_proj", "mlp.up_proj")
                src_suffix = src_suffix.replace("mlp_down_proj", "mlp.down_proj")
                src = f"{p}.{src_suffix}"
                if src in sd:
                    remap[k] = sd[src]
        missing = [k for k in target if k not in remap]
        unexpected = [k for k in sd if k not in remap.values()]
        # 直接按名覆蓋 (利用 load_state_dict 的 strict=False 忽略 unexpected,
        # 但避免 key 名不匹配問題 — 直接對 module 逐個 copy)
        state = {}
        current = self.state_dict()
        for k, v in remap.items():
            if k in current:
                state[k] = v
        self.load_state_dict(state, strict=False)
        return {"missing": missing, "unexpected": unexpected}

    def forward(
        self,
        backbone_hidden: torch.Tensor,       # [backbone_hidden_size]
        current_token: torch.Tensor,         # scalar tensor (long)
        target_token_embedding: Optional[torch.Tensor] = None,  # [backbone_hidden_size] (主模型 embed)
        kv: Optional[dict] = None,           # {"sliding": (k, v), "full": (k, v)}
        position: int = 0,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """回傳 next token (scalar tensor) 或 (token, logits, next_backbone_hidden)."""
        # 1. token embedding -> proxy
        if target_token_embedding is not None:
            proxy = target_token_embedding  # [2816]
        else:
            tok = self.embed_tokens(current_token)  # [1024]
            proxy = self.post_projection(tok)       # [2816]

        # 2-3. concat + pre_projection (引擎: concat([proxyEmbed, currentBackbone]))
        fused = torch.cat([proxy, backbone_hidden], dim=-1)  # [5632]
        x = self.pre_projection(fused)                       # [1024]

        # 4. layers
        for i, layer in enumerate(self.layers):
            kind = "full" if self.layer_types[i] == "full_attention" else "sliding"
            layer_kv = (kv or {}).get(kind)
            if layer_kv is not None:
                layer_kv = dict(layer_kv, kind=kind)
            x = layer(x, position=position, kv=layer_kv)

        # 5. final norm -> lm_head (tied)
        final = self.model_norm(x)
        logits = F.linear(final, self.embed_tokens.weight)  # [vocab]
        token = logits.argmax(dim=-1)

        # 6. next backbone for chained draft
        next_backbone = self.post_projection(final)

        if return_logits:
            return token, logits, next_backbone
        return token


class Gemma4AssistantLayer(nn.Module):
    def __init__(self, hidden: int, intermediate: int, heads: int, head_dim: int,
                 kv_heads: int, eps: float):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = head_dim
        self.kv_heads = kv_heads
        self.eps = eps
        q_rows = heads * head_dim

        self.input_layernorm = RMSNorm(hidden, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps)
        self.self_attn_q_proj = nn.Linear(hidden, q_rows, bias=False)
        # 官方 q_norm 是 [head_dim] 單一向量, 所有 heads 共享
        self.self_attn_q_norm = nn.Parameter(torch.ones(head_dim))
        self.self_attn_o_proj = nn.Linear(q_rows, hidden, bias=False)
        self.mlp_gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.mlp_up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.mlp_down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.layer_scalar = nn.Parameter(torch.tensor(1.0))
        self.post_feedforward_layernorm: Optional[RMSNorm] = None  # 偵測到才掛
        # 端側 finetune 用正則化 (train-only; eval 為 no-op)
        self.dropout = nn.Dropout(0.1)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x: torch.Tensor, position: int, kv: Optional[dict]) -> torch.Tensor:
        # Attention branch
        h = self.input_layernorm(x)
        q = self.self_attn_q_proj(h).view(self.heads, self.head_dim)  # [heads, head_dim]
        qn = q / (q.pow(2).mean(-1, keepdim=True) + self.eps).sqrt() * self.self_attn_q_norm
        # 引擎 attention 傳 scale=1.0 (Gemma4 模型本身 scale=1.0), 不做 1/sqrt(head_dim)
        # 是否 full (kv 是 full 即 full layer)
        is_full = kv is not None and kv.get("kind") == "full"
        qn = RoPE.apply(qn, position, self.head_dim,
                        kind="full" if is_full else "sliding")
        # attention with shared KV
        if kv is not None and kv.get("k") is not None:
            k, v, seq_len, window = kv["k"], kv["v"], kv.get("seq_len", 0), kv.get("window", 1024)
            # GQA: q heads = repeat_factor * kv heads, 每組 q 共享同一 kv head
            kv_heads = k.shape[1]
            repeat = self.heads // kv_heads
            q_g = qn.view(kv_heads, repeat, self.head_dim)
            # scores[kv, repeat, seq] (無 scale — 引擎 attention 的 scale=1.0)
            scores = torch.einsum("hrd,shd->hrs", q_g, k)
            scores = scores.reshape(self.heads, -1)  # [heads, seq]
            # 可見位置 [start, seq_len); 超出 validTokenCount 的位置一律 -inf
            start = max(0, seq_len - window) if not is_full else 0
            mask = torch.zeros(scores.shape, dtype=torch.bool, device=scores.device)
            mask[:, start:seq_len] = True
            scores = torch.where(mask, scores, torch.tensor(float("-inf"), dtype=scores.dtype, device=scores.device))
            probs = torch.softmax(scores, dim=-1)  # [heads, seq]
            prob_g = probs.view(kv_heads, repeat, -1)
            ctx_g = torch.einsum("hrs,shd->hrd", prob_g, v)
            ctx = ctx_g.reshape(self.heads, self.head_dim)
        else:
            # 無 KV: Metal fallback = o_proj(qNormed) 直接過 (identity attention)
            ctx = qn
        attn_out = self.dropout(self.self_attn_o_proj(ctx.reshape(-1)))

        # residual 1
        h = x + self.post_attention_layernorm(attn_out)

        # FFN
        ff_in = self.pre_feedforward_layernorm(h)
        gate = self.mlp_gate_proj(ff_in)
        up = self.mlp_up_proj(ff_in)
        act = _gelu_pytorch_tanh(gate) * up
        ff = self.dropout(self.mlp_down_proj(act))
        if self.post_feedforward_layernorm is not None:
            ff = self.post_feedforward_layernorm(ff)
        h = (h + ff) * self.layer_scalar
        return h


def load_gemma4_assistant(model_dir: str, device: str = "cpu") -> Gemma4Assistant:
    """載入官方 assistant head (config.json + model.safetensors)."""
    config_path = os.path.join(model_dir, "config.json")
    st_path = os.path.join(model_dir, "model.safetensors")
    model = Gemma4Assistant(config_path)
    info = model.load_official_weights(st_path)
    # post_feedforward_layernorm 偵測
    sd = load_file(st_path)
    for i, layer in enumerate(model.layers):
        if f"model.layers.{i}.post_feedforward_layernorm.weight" in sd:
            layer.post_feedforward_layernorm = RMSNorm(model.hidden_size, model.eps)
            with torch.no_grad():
                layer.post_feedforward_layernorm.weight.copy_(
                    sd[f"model.layers.{i}.post_feedforward_layernorm.weight"])
    model = model.to(device).eval()
    return model, info


if __name__ == "__main__":
    import sys
    m, info = load_gemma4_assistant(
        "/Users/alexchuang/Documents/flashkv0516/models/gemma-4-mtp-head")
    print("missing:", info["missing"])
    print("unexpected:", info["unexpected"])
    print("params:", sum(p.numel() for p in m.parameters()) / 1e6, "M")
    # sanity: random backbone + token
    torch.manual_seed(0)
    bh = torch.randn(m.backbone_hidden_size)
    tok = torch.tensor(3)
    with torch.no_grad():
        t = m(bh, tok)
    print("sanity token:", t.item())
