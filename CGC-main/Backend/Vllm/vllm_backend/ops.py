#!/usr/bin/env python3
"""
CGC KDA Operations

提供 KDA 算子的 CUDA 實現。
"""

import torch


FLASHKDA_AVAILABLE = False
try:
    import flash_kda as _flash_kda
    FLASHKDA_AVAILABLE = True
except ImportError:
    pass


def kda_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 1.0,
    beta: float = 0.1,
    g: torch.Tensor = None,
    A_log: torch.Tensor = None,
    dt_bias: torch.Tensor = None,
    lower_bound: float = -5.0,
):
    """
    KDA forward using FlashKDA

    Args:
        q: Query [B, H, T, D]
        k: Key [B, H, T, D]
        v: Value [B, H, T, D]
        scale: Scaling factor
        beta: KDA beta parameter
        g: Gate (optional)
        A_log: A_log for FlashKDA state
        dt_bias: dt_bias for FlashKDA
        lower_bound: Lower bound for FlashKDA

    Returns:
        output: Attention output [B, H, T, D]
    """
    if not FLASHKDA_AVAILABLE:
        raise RuntimeError("FlashKDA not available")

    batch_size, num_heads, seq_len, head_dim = q.shape

    if g is None:
        g = torch.ones((batch_size, num_heads, seq_len), device=q.device, dtype=q.dtype)
    if A_log is None:
        A_log = torch.full((batch_size, num_heads), float('-inf'), device=q.device)
    if dt_bias is None:
        dt_bias = torch.zeros((batch_size, num_heads), device=q.device)

    out = torch.empty_like(q)

    _flash_kda.fwd(
        q=q, k=k, v=v, g=g,
        beta=beta, scale=scale,
        out=out,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )

    return out


def kda_backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    scale: float = 1.0,
    beta: float = 0.1,
):
    """
    KDA backward using FlashKDA

    Returns:
        grad_q, grad_k, grad_v
    """
    if not FLASHKDA_AVAILABLE:
        raise RuntimeError("FlashKDA not available")

    workspace = torch.empty(_flash_kda.get_workspace_size(q, k, v), device=q.device, dtype=torch.uint8)

    grad_q, grad_k, grad_v = _flash_kda.bwd(
        grad_out=grad_out,
        q=q, k=k, v=v,
        out=out,
        g=None,
        beta=beta,
        scale=scale,
        workspace=workspace,
    )

    return grad_q, grad_k, grad_v


def get_backend_info():
    """獲取後端信息"""
    return {
        "flashkda_available": FLASHKDA_AVAILABLE,
        "backend": "cgc_kda",
    }