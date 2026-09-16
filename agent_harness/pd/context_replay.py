#!/usr/bin/env python3
"""Context Replay — 翻譯後 hidden state → 完整 KV cache (論文附錄 C.4).

管線:
  1. MoT-h 翻譯: hidden_src [seq, 2816] → hidden_tgt [seq, 2048]
  2. KV 還原: 通道層用 hidden_tgt @ Wk/Wv → K/V [seq, kv_dim]
  3. Context Replay: 非通道層用稀疏注意力補全 KV

MVP 簡化:
  - 所有層用同一個 hidden_tgt 還原 KV (不做通道區分)
  - 不做稀疏重放 (Swift 端全量 Wk/Wv 投影即可)
  - 後續: 通道層 + 非通道層區分 + Top-S=128 稀疏注意力

Swift 端實現參考 (resume 端點):
  for layer in 0..<40:
    K[layer] = hidden_tgt @ Wk[layer]  // [seq, kv_dim]
    V[layer] = hidden_tgt @ Wv[layer]  // [seq, kv_dim]
  // 設置 KV cache, 開始 decode
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KV 還原 (MoT-h 核心步驟)
# ---------------------------------------------------------------------------
def restore_kv_cache(
    hidden_state: torch.Tensor,  # [seq_len, hidden_dim]
    wk: torch.Tensor,            # [num_layers, hidden_dim, kv_dim]
    wv: torch.Tensor,            # [num_layers, hidden_dim, kv_dim]
) -> tuple[torch.Tensor, torch.Tensor]:
    """用原生 Wk/Wv 從 hidden state 還原完整 KV cache.

    MoT-h 設計: Target 模型用原生 Wk/Wv 投影, 不需要 MoT 翻譯 KV.
    傳輸量減半 (只傳 hidden, 不傳 K/V).

    Args:
        hidden_state: [seq_len, hidden_dim] 翻譯後的 hidden state
        wk: [num_layers, hidden_dim, kv_dim] 目標模型所有層的 Wk
        wv: [num_layers, hidden_dim, kv_dim] 目標模型所有層的 Wv

    Returns:
        K: [num_layers, seq_len, kv_dim]
        V: [num_layers, seq_len, kv_dim]
    """
    num_layers = wk.shape[0]
    # 批量投影: [num_layers, hidden_dim, kv_dim] → [num_layers, seq_len, kv_dim]
    K = torch.einsum("sh,lhk->lsk", hidden_state, wk)  # [L, S, kv_dim]
    V = torch.einsum("sh,lhk->lsk", hidden_state, wv)  # [L, S, kv_dim]
    logger.debug("KV cache restored: K=%s V=%s", K.shape, V.shape)
    return K, V


def restore_kv_cache_channel(
    hidden_state: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    channel_layers: list[int],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """只還原通道層的 KV (非通道層用 Context Replay 補全).

    Args:
        hidden_state: [seq_len, hidden_dim]
        wk: [num_layers, hidden_dim, kv_dim]
        wv: [num_layers, hidden_dim, kv_dim]
        channel_layers: 通道層索引列表

    Returns:
        {layer_idx: (K, V)} 通道層的 KV cache
    """
    kv_cache = {}
    for layer in channel_layers:
        K = hidden_state @ wk[layer]  # [seq_len, kv_dim]
        V = hidden_state @ wv[layer]  # [seq_len, kv_dim]
        kv_cache[layer] = (K, V)
    logger.debug("Channel KV restored: %d layers", len(kv_cache))
    return kv_cache


# ---------------------------------------------------------------------------
# 稀疏注意力 (Context Replay 用)
# ---------------------------------------------------------------------------
def sparse_attention(
    query: torch.Tensor,    # [num_heads, seq_len, head_dim]
    key: torch.Tensor,      # [num_heads, seq_len, head_dim]
    value: torch.Tensor,    # [num_heads, seq_len, head_dim]
    top_s: int = 128,       # Top-S 關鍵 token
    src_attn_map: torch.Tensor | None = None,  # [num_heads, seq_len, seq_len] 源引導
) -> torch.Tensor:
    """源引導稀疏注意力 (論文附錄 C.4).

    底層 L_Full=2 用完整注意力, 上層用 Top-S 稀疏.
    這裡只實現稀疏部分, 完整注意力用標準 attention.

    Args:
        query: [num_heads, seq_len, head_dim]
        key: [num_heads, seq_len, head_dim]
        value: [num_heads, seq_len, head_dim]
        top_s: Top-S 關鍵 token 數
        src_attn_map: 源模型注意力分數 (用於選 Top-S)

    Returns:
        [num_heads, seq_len, head_dim]
    """
    H, S, D = query.shape

    if src_attn_map is None:
        # 無源引導: 用 query-key 點積選 Top-S
        attn_scores = query @ key.transpose(-2, -1) / (D ** 0.5)  # [H, S, S]
    else:
        attn_scores = src_attn_map

    # 對每個 query, 選 Top-S 個 key
    top_s = min(top_s, S)
    top_vals, top_idx = torch.topk(attn_scores, k=top_s, dim=-1)  # [H, S, top_s]

    # 用選中的 key/value 計算注意力
    # gather: [H, S, top_s, D]
    gathered_k = torch.gather(
        key.unsqueeze(1).expand(H, S, S, D),
        dim=2,
        index=top_idx.unsqueeze(-1).expand(H, S, top_s, D),
    )
    gathered_v = torch.gather(
        value.unsqueeze(1).expand(H, S, S, D),
        dim=2,
        index=top_idx.unsqueeze(-1).expand(H, S, top_s, D),
    )

    # 注意力計算 (只在 Top-S 上)
    attn_weights = F.softmax(top_vals, dim=-1)  # [H, S, top_s]
    out = torch.einsum("hst,hstd->hsd", attn_weights, gathered_v)  # [H, S, D]

    return out


# ---------------------------------------------------------------------------
# Context Replay 完整管線 (模擬, 實際在 Swift/Metal 端跑)
# ---------------------------------------------------------------------------
@dataclass
class ReplayConfig:
    """Context Replay 配置."""
    top_s: int = 128           # Top-S 稀疏注意力
    l_full: int = 2            # 底層完整注意力層數
    channel_ratio: float = 0.4 # 通道比例


def context_replay_mvp(
    hidden_state: torch.Tensor,  # [seq_len, hidden_dim] 翻譯後
    wk: torch.Tensor,            # [num_layers, hidden_dim, kv_dim]
    wv: torch.Tensor,            # [num_layers, hidden_dim, kv_dim]
) -> tuple[torch.Tensor, torch.Tensor]:
    """MVP: 所有層用同一個 hidden state 還原 KV (不做通道區分).

    這是最簡單的版本, 適合先跑通管線.
    後續升級到通道層 + 非通道層 + 稀疏重放.

    Args:
        hidden_state: [seq_len, hidden_dim] MoT-h 翻譯後的 hidden state
        wk: [num_layers, hidden_dim, kv_dim]
        wv: [num_layers, hidden_dim, kv_dim]

    Returns:
        K: [num_layers, seq_len, kv_dim]
        V: [num_layers, seq_len, kv_dim]
    """
    logger.info(
        "Context Replay MVP: hidden=%s wk=%s → KV cache",
        tuple(hidden_state.shape), tuple(wk.shape),
    )
    return restore_kv_cache(hidden_state, wk, wv)


def context_replay_full(
    hidden_state: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    channel_layers: list[int],
    src_attn_maps: torch.Tensor | None = None,
    cfg: ReplayConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """完整 Context Replay: 通道層注入 + 非通道層稀疏重放.

    Args:
        hidden_state: [seq_len, hidden_dim]
        wk: [num_layers, hidden_dim, kv_dim]
        wv: [num_layers, hidden_dim, kv_dim]
        channel_layers: 通道層索引
        src_attn_maps: [num_layers, num_heads, seq_len, seq_len] 源引導注意力
        cfg: Replay 配置

    Returns:
        K: [num_layers, seq_len, kv_dim]
        V: [num_layers, seq_len, kv_dim]
    """
    if cfg is None:
        cfg = ReplayConfig()

    num_layers = wk.shape[0]
    seq_len, hidden_dim = hidden_state.shape
    kv_dim = wk.shape[-1]

    K_all = torch.zeros(num_layers, seq_len, kv_dim)
    V_all = torch.zeros(num_layers, seq_len, kv_dim)

    for layer in range(num_layers):
        if layer in channel_layers:
            # 通道層: 直接用 hidden @ Wk/Wv 還原
            K_all[layer] = hidden_state @ wk[layer]
            V_all[layer] = hidden_state @ wv[layer]
        else:
            # 非通道層: 稀疏重放 (MVP: 也用同一個 hidden, 後續改稀疏)
            K_all[layer] = hidden_state @ wk[layer]
            V_all[layer] = hidden_state @ wv[layer]
            # TODO: 實現真正的稀疏重放 (需要 target_model 的 forward)

    logger.info(
        "Context Replay Full: %d channel layers, %d replay layers",
        len(channel_layers), num_layers - len(channel_layers),
    )
    return K_all, V_all


# ---------------------------------------------------------------------------
# 主函數: 驗證 KV 還原
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Context Replay KV 還原驗證 ===\n")

    # 模擬 Qwen3.6 維度
    seq_len = 64
    hidden_dim = 2048
    num_layers = 40
    kv_dim = 512  # GQA: hidden / num_heads * num_kv_heads

    # 翻譯後 hidden state
    hidden = torch.randn(seq_len, hidden_dim)

    # 模擬 Qwen3.6 的 Wk/Wv
    wk = torch.randn(num_layers, hidden_dim, kv_dim) * 0.02
    wv = torch.randn(num_layers, hidden_dim, kv_dim) * 0.02

    # MVP KV 還原
    K, V = context_replay_mvp(hidden, wk, wv)
    print(f"輸入: hidden {tuple(hidden.shape)}")
    print(f"權重: wk {tuple(wk.shape)}")
    print(f"輸出: K {tuple(K.shape)}, V {tuple(V.shape)}")
    print(f"K 統計: mean={K.mean():.4f} std={K.std():.4f}")
    print(f"V 統計: mean={V.mean():.4f} std={V.std():.4f}")

    # 通道層還原
    channel_layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
    kv_cache = restore_kv_cache_channel(hidden, wk, wv, channel_layers)
    print(f"\n通道層 KV: {len(kv_cache)} layers (channel_layers={channel_layers})")
    for layer in list(kv_cache.keys())[:3]:
        k, v = kv_cache[layer]
        print(f"  layer {layer}: K {tuple(k.shape)} V {tuple(v.shape)}")

    # 傳輸量對比
    hidden_bytes = seq_len * hidden_dim * 4  # float32
    kv_bytes = num_layers * seq_len * kv_dim * 4 * 2  # K + V
    print(f"\n=== 傳輸量對比 ===")
    print(f"hidden state: {hidden_bytes / 1024:.1f} KB ({seq_len}×{hidden_dim} float32)")
    print(f"完整 KV cache: {kv_bytes / 1024:.1f} KB ({num_layers}×{seq_len}×{kv_dim}×2 float32)")
    print(f"節省: {(1 - hidden_bytes / kv_bytes) * 100:.1f}%")
