#!/usr/bin/env python3
"""
vLLM CGC KDA Attention Backend

這是一個 vLLM custom attention backend，實現 KDA (Kimi Delta Attention) 替代標準 FlashAttention。

使用方法：
1. 確保 Backend/Vllm/vllm_backend 可被 Python 匯入（repo root 在 PYTHONPATH）
2. 設置環境變量或傳遞參數：
   VLLM_ATTENTION_BACKEND=cgc_kda python your_script.py
   或
   llm = LLM(model="...", attention_backend="cgc_kda")
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass

import os
import sys

FLASHKDA_AVAILABLE = False
try:
    import flash_kda as _flash_kda
    FLASHKDA_AVAILABLE = True
    print(f"[CGH KDA Backend] FlashKDA CUDA kernel loaded: {_flash_kda}")
except ImportError:
    print(f"[CGH KDA Backend] FlashKDA not available, using PyTorch SDPA fallback")


@dataclass
class CGCHKDBackendConfig:
    """CGC KDA Backend 配置"""
    enable_flashkda: bool = True
    enable_kda: bool = True
    kda_beta: float = 0.1
    kda_scale: float = 1.0
    use_gate: bool = True
    fallback_to_sdpa: bool = True


class CGCHKDBackend:
    """
    vLLM CGC KDA Attention Backend

    實現 KDA (Kimi Delta Attention) 替代 FlashAttention
    核心算法：
        S_new = S * (1 - beta * K[i] * K[j]) + beta * K[i] * V[j]
        O = Q * S_new * scale
    """

    def __init__(self, layer, head_dim, num_heads, scale=None, attn_op=None, softmax_scale=None):
        self.layer = layer
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = scale or (head_dim ** -0.5)
        self.softmax_scale = softmax_scale or self.scale

        self.config = CGCHKDBackendConfig()

        self._init_kda_state()

        print(f"[CGH KDA Backend] Initialized: head_dim={head_dim}, num_heads={num_heads}, scale={self.scale:.4f}")
        print(f"[CGH KDA Backend] FlashKDA available: {FLASHKDA_AVAILABLE}")
        print(f"[CGH KDA Backend] KDA state shape: {self.S.shape if hasattr(self, 'S') else 'N/A'}")

    def _init_kda_state(self):
        """初始化 KDA 狀態矩陣 S"""
        max_batch = 32
        max_seq_len = 4096

        self.S = torch.zeros(
            max_batch, self.num_heads, self.head_dim, self.head_dim,
            device='cuda', dtype=torch.float16
        )
        self.S_batch_ptr = 0

        self.call_count = 0

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        KDA Attention Forward

        Args:
            q: Query tensor [B, H, T, D] or [T, H, D]
            k: Key tensor [B, H, T, D] or [T, H, D]
            v: Value tensor [B, H, T, D] or [T, H, D]
            cu_seqlens: CUDA stream lengths (for varlen)
            max_seqlen: Maximum sequence length

        Returns:
            Attention output with same shape as v
        """
        self.call_count += 1

        original_shape = q.shape
        batch_size, num_heads, seq_len, head_dim = q.shape

        if cu_seqlens is not None:
            return self._forward_varlen(q, k, v, cu_seqlens, max_seqlen)

        if FLASHKDA_AVAILABLE and self.config.enable_flashkda:
            return self._forward_flashkda(q, k, v)
        elif self.config.enable_kda:
            return self._forward_kda(q, k, v)
        else:
            return self._forward_sdpa(q, k, v)

    def _forward_flashkda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Forward via FlashKDA CUDA kernel"""
        try:
            batch_size, num_heads, seq_len, head_dim = q.shape

            g = torch.ones((batch_size, num_heads, seq_len), device=q.device, dtype=q.dtype)
            A_log = torch.full((batch_size, num_heads), float('-inf'), device=q.device)
            dt_bias = torch.zeros((batch_size, num_heads), device=q.device)
            lower_bound = 0.0

            out = torch.empty_like(q)

            _flash_kda.fwd(
                q=q, k=k, v=v, g=g,
                beta=self.config.kda_beta,
                scale=self.config.kda_scale,
                out=out,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
            )

            if self.call_count <= 3:
                print(f"[CGH KDA Backend] FlashKDA forward #{self.call_count}: {q.shape} -> {out.shape}")

            return out

        except Exception as e:
            print(f"[CGH KDA Backend] FlashKDA failed: {e}, falling back to KDA")
            return self._forward_kda(q, k, v)

    def _forward_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        KDA Forward (PyTorch implementation)

        核心算法：
        S_new = S * (1 - beta * K[i] * K[j]) + beta * K[i] * V[j]
        O = Q * S_new * scale
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        beta = self.config.kda_beta
        scale = self.config.kda_scale

        out = torch.empty_like(q)

        for b in range(batch_size):
            for h in range(num_heads):
                s_offset = (b * num_heads + h) * head_dim * head_dim
                q_offset = (b * num_heads + h) * seq_len * head_dim
                kv_base = (b * num_heads + h) * seq_len * head_dim

                S_bh = self.S.reshape(-1, head_dim, head_dim)[b * num_heads + h]

                for t in range(seq_len):
                    kv_offset = kv_base + t * head_dim

                    k_t = k.reshape(-1, head_dim)[kv_offset // head_dim:kv_offset // head_dim + 1].squeeze(0)

                    k_t_expanded = k_t.unsqueeze(1)
                    k_t_k_j = torch.matmul(k_t_expanded, k_t.unsqueeze(0)).squeeze()

                    v_t = v.reshape(-1, head_dim)[kv_offset // head_dim:kv_offset // head_dim + 1].squeeze(0)
                    k_t_v_j = torch.matmul(k_t_expanded, v_t.unsqueeze(0)).squeeze()

                    S_bh = S_bh * (1.0 - beta * k_t_k_j) + beta * k_t_v_j

                self.S.reshape(-1, head_dim, head_dim)[b * num_heads + h] = S_bh

                q_bh = q.reshape(-1, head_dim)[q_offset // head_dim:q_offset // head_dim + seq_len]

                O_bh = torch.matmul(q_bh, S_bh) * scale

                out.reshape(-1, head_dim)[q_offset // head_dim:q_offset // head_dim + seq_len] = O_bh

        if self.call_count <= 3:
            print(f"[CGH KDA Backend] KDA forward #{self.call_count}: {q.shape} -> {out.shape}")

        return out

    def _forward_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Forward via PyTorch SDPA (fallback)"""
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=self.softmax_scale
        )

        if self.call_count <= 3:
            print(f"[CGH KDA Backend] SDPA fallback #{self.call_count}: {q.shape} -> {out.shape}")

        return out

    def _forward_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Variable length forward (for batched decoding)"""
        if FLASHKDA_AVAILABLE:
            return self._forward_flashkda(q, k, v)
        else:
            return self._forward_sdpa(q, k, v)


def get_backend():
    """返回 CGC KDA Backend 類工廠函數"""
    return CGCHKDBackend


def init_backend(model, layer):
    """初始化 CGC KDA Backend"""
    return CGCHKDBackend(
        layer=layer,
        head_dim=model.head_dim,
        num_heads=model.num_heads,
    )
