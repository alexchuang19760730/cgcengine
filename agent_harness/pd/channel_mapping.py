#!/usr/bin/env python3
"""深度比例通道映射 — 论文附录 C.3.

异构模型层对应规则: 按相对深度比例配对源层 i、目标层 j,
取连续滑动窗口作翻译通道集合 C.

Gemma4 (30 层) → Qwen3.6 (40 层) 通道映射:
  - 深度比例: i/30 ↔ j/40
  - 通道比例: ChannelRatio=0.3 → 取 30% 的层作通道层
  - 滑动窗口: 在候选通道中取连续窗口, 验证集选最优

定制改造 (Gemma4 → Qwen3.6):
  1. 深度比例匹配为基础
  2. 离线预跑相似度矩阵 (余弦), 筛选前 20% 层
  3. ChannelRatio 提升至 0.4 (原版 0.3, 减少校正赤字误差)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 模型层配置
# ---------------------------------------------------------------------------
@dataclass
class ModelLayerConfig:
    """模型层配置."""
    name: str
    num_layers: int
    hidden_size: int


GEMMA4_26B_A4B = ModelLayerConfig("gemma4-26b-a4b", 30, 2816)
QWEN36_35B_A3B = ModelLayerConfig("qwen36-35b-a3b", 40, 2048)


# ---------------------------------------------------------------------------
# 深度比例通道映射
# ---------------------------------------------------------------------------
def depth_ratio_mapping(
    src_num_layers: int,
    tgt_num_layers: int,
) -> List[Tuple[int, int]]:
    """深度比例配对: 每个源层找到深度比例最接近的目标层.

    Args:
        src_num_layers: 源模型层数 (e.g. 30)
        tgt_num_layers: 目标模型层数 (e.g. 40)

    Returns:
        [(src_layer, tgt_layer), ...] 长度 = src_num_layers
        每个 src_layer 映射到深度比例最接近的 tgt_layer
    """
    pairs = []
    for i in range(src_num_layers):
        i_norm = i / src_num_layers
        # 找深度比例最接近的 tgt_layer
        best_j = min(
            range(tgt_num_layers),
            key=lambda j: abs(j / tgt_num_layers - i_norm),
        )
        pairs.append((i, best_j))
    return pairs


def select_channel_layers(
    src_num_layers: int,
    tgt_num_layers: int,
    channel_ratio: float = 0.4,
) -> Tuple[List[int], List[int]]:
    """选取通道层: 按比例选取用于翻译的层.

    通道层 = 翻译窗口覆盖的层. 非通道层用 Context Replay 补全.

    Args:
        src_num_layers: 源模型层数
        tgt_num_layers: 目标模型层数
        channel_ratio: 通道比例 (0.4 = 40% 的层作通道层)

    Returns:
        src_channel_layers: 源模型通道层索引列表
        tgt_channel_layers: 目标模型通道层索引列表
    """
    all_pairs = depth_ratio_mapping(src_num_layers, tgt_num_layers)

    # 均匀采样通道层
    num_channels = max(1, int(src_num_layers * channel_ratio))
    # 在源层中均匀采样
    src_indices = np.linspace(0, src_num_layers - 1, num_channels, dtype=int)
    src_channel_layers = sorted(set(src_indices.tolist()))

    # 对应的目标层
    tgt_channel_layers = sorted(set(
        all_pairs[i][1] for i in src_channel_layers
    ))

    return src_channel_layers, tgt_channel_layers


def sliding_windows(
    channel_layers: List[int],
    window_size: int = 4,
) -> List[List[int]]:
    """在通道层上生成滑动窗口.

    Args:
        channel_layers: 通道层索引列表
        window_size: 窗口大小

    Returns:
        窗口列表, 每个窗口是通道层的连续子序列
    """
    if len(channel_layers) <= window_size:
        return [channel_layers]
    return [
        channel_layers[i : i + window_size]
        for i in range(len(channel_layers) - window_size + 1)
    ]


# ---------------------------------------------------------------------------
# 相似度矩阵 (离线预计算, 用于通道层筛选)
# ---------------------------------------------------------------------------
def compute_layer_similarity_matrix(
    src_hidden_states: torch.Tensor,  # [src_num_layers, seq_len, src_dim]
    tgt_hidden_states: torch.Tensor,  # [tgt_num_layers, seq_len, tgt_dim]
) -> torch.Tensor:
    """计算源/目标层间 hidden state 余弦相似度矩阵.

    由于 src_dim ≠ tgt_dim, 先做 mean pooling 到 [src_num_layers, src_dim]
    和 [tgt_num_layers, tgt_dim], 然后用线性投影对齐维度再算余弦.

    MVP: 用 SVD 降维到 min(src_dim, tgt_dim) 后算余弦.

    Args:
        src_hidden_states: [src_num_layers, seq_len, src_dim]
        tgt_hidden_states: [tgt_num_layers, seq_len, tgt_dim]

    Returns:
        similarity [src_num_layers, tgt_num_layers] 余弦相似度
    """
    L_src, S, D_src = src_hidden_states.shape
    L_tgt, S2, D_tgt = tgt_hidden_states.shape
    assert S == S2, f"seq_len mismatch: {S} vs {S2}"

    # Mean pool over seq_len → [L, D]
    src_pooled = src_hidden_states.mean(dim=1)  # [L_src, D_src]
    tgt_pooled = tgt_hidden_states.mean(dim=1)  # [L_tgt, D_tgt]

    # SVD 降维到 min(D_src, D_tgt) 对齐
    min_dim = min(D_src, D_tgt)
    src_proj = torch.linalg.svd(src_pooled)[0][:, :min_dim]  # [L_src, min_dim]
    tgt_proj = torch.linalg.svd(tgt_pooled)[0][:, :min_dim]  # [L_tgt, min_dim]

    # 余弦相似度
    src_norm = F.normalize(src_proj, dim=-1)  # [L_src, min_dim]
    tgt_norm = F.normalize(tgt_proj, dim=-1)  # [L_tgt, min_dim]
    sim = src_norm @ tgt_norm.T  # [L_src, L_tgt]

    return sim


def select_channels_by_similarity(
    sim_matrix: torch.Tensor,  # [src_num_layers, tgt_num_layers]
    channel_ratio: float = 0.4,
    top_percent: float = 0.2,
) -> Tuple[List[int], List[int]]:
    """基于相似度矩阵筛选通道层.

    1. 深度比例配对为基础
    2. 在配对中取相似度前 top_percent 的层
    3. 补足到 channel_ratio 要求的层数

    Args:
        sim_matrix: [src_num_layers, tgt_num_layers] 余弦相似度
        channel_ratio: 通道比例
        top_percent: 相似度前 N% 的层优先选为通道层

    Returns:
        src_channel_layers, tgt_channel_layers
    """
    L_src, L_tgt = sim_matrix.shape

    # 深度比例配对
    all_pairs = depth_ratio_mapping(L_src, L_tgt)

    # 按相似度排序
    pair_sims = [(sim_matrix[i, j].item(), i, j) for i, j in all_pairs]
    pair_sims.sort(reverse=True)

    # 取前 top_percent
    num_top = max(1, int(len(pair_sims) * top_percent))
    top_pairs = pair_sims[:num_top]

    # 补足到 channel_ratio
    num_channels = max(num_top, int(L_src * channel_ratio))
    remaining = pair_sims[num_top:]
    while len(top_pairs) < num_channels and remaining:
        top_pairs.append(remaining.pop(0))

    src_channels = sorted(set(p[1] for p in top_pairs))
    tgt_channels = sorted(set(p[2] for p in top_pairs))

    return src_channels, tgt_channels


# ---------------------------------------------------------------------------
# 便捷接口
# ---------------------------------------------------------------------------
def get_gemma4_to_qwen36_channels(
    channel_ratio: float = 0.4,
    window_size: int = 4,
) -> dict:
    """获取 Gemma4 → Qwen3.6 的通道映射配置.

    Returns:
        {
            "src_model": "gemma4-26b-a4b",
            "tgt_model": "qwen36-35b-a3b",
            "src_channel_layers": [0, 4, 8, 12, ...],
            "tgt_channel_layers": [0, 5, 11, 16, ...],
            "depth_ratio_pairs": [(0,0), (1,1), ...],
            "windows": [[0,4,8,12], [4,8,12,16], ...],
            "channel_ratio": 0.4,
        }
    """
    src_ch, tgt_ch = select_channel_layers(
        GEMMA4_26B_A4B.num_layers,
        QWEN36_35B_A3B.num_layers,
        channel_ratio,
    )
    pairs = depth_ratio_mapping(
        GEMMA4_26B_A4B.num_layers,
        QWEN36_35B_A3B.num_layers,
    )
    windows = sliding_windows(src_ch, window_size)

    return {
        "src_model": GEMMA4_26B_A4B.name,
        "tgt_model": QWEN36_35B_A3B.name,
        "src_num_layers": GEMMA4_26B_A4B.num_layers,
        "tgt_num_layers": QWEN36_35B_A3B.num_layers,
        "src_channel_layers": src_ch,
        "tgt_channel_layers": tgt_ch,
        "depth_ratio_pairs": pairs,
        "windows": windows,
        "channel_ratio": channel_ratio,
        "window_size": window_size,
    }


# ---------------------------------------------------------------------------
# 主函数: 打印通道映射
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import torch.nn.functional as F

    cfg = get_gemma4_to_qwen36_channels()
    print("=== Gemma4 (30层) → Qwen3.6 (40层) 通道映射 ===\n")
    print(f"通道比例: {cfg['channel_ratio']}")
    print(f"源通道层 ({len(cfg['src_channel_layers'])}): {cfg['src_channel_layers']}")
    print(f"目标通道层 ({len(cfg['tgt_channel_layers'])}): {cfg['tgt_channel_layers']}")
    print(f"窗口数: {len(cfg['windows'])}")
    print(f"\n深度比例配对 (前 10):")
    for i, j in cfg["depth_ratio_pairs"][:10]:
        print(f"  src[{i:2d}] (d={i/30:.3f}) → tgt[{j:2d}] (d={j/40:.3f})")
    print(f"  ...")

    # 验证: 随机 hidden state 跑相似度矩阵
    print(f"\n=== 模拟相似度矩阵 (随机数据) ===")
    sim = compute_layer_similarity_matrix(
        torch.randn(30, 64, 2816),
        torch.randn(40, 64, 2048),
    )
    print(f"sim_matrix shape: {sim.shape}")
    print(f"sim range: [{sim.min():.3f}, {sim.max():.3f}]")

    src_ch_sim, tgt_ch_sim = select_channels_by_similarity(sim)
    print(f"基于相似度的源通道层: {src_ch_sim}")
    print(f"基于相似度的目标通道层: {tgt_ch_sim}")
