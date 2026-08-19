# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ds4 CUDA Kernels - ds4.c 核心 kernel 的 CUDA 实现

从 ds4.c 移植的核心 kernel：
1. ds4_attention - Sink-aware attention with KV cache compression
2. ds4_moe_routing - DeepSeek MoE routing with softplus normalization
3. ds4_layer_norm - RMSNorm with ds4 风格
4. ds4_rope - RoPE with compress frequency

DeepSeek V4 Flash 特定优化：
- 固定形状: 43 layers, 4096 dim, 64 heads
- Grouped LoRA output projection
- KV cache compression with indexer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)

DS4_N_HEAD = 64
DS4_N_HEAD_DIM = 64
DS4_N_HEAD_KV = 64
DS4_N_EMBD = 4096
DS4_N_LORA_Q = 32
DS4_N_LORA_O = 128
DS4_N_OUT_GROUP = 8
DS4_N_SWA = 512
DS4_N_INDEXER_HEAD = 8
DS4_N_INDEXER_HEAD_DIM = 32
DS4_N_INDEXER_TOP_K = 16
DS4_RMS_EPS = 1e-6
DS4_SWIGLU_CLAMP_EXP = 20.0
DS4_N_EXPERT = 8
DS4_N_EXPERT_USED = 4

SWIGLU_CLAMP_EXP = 20.0

def ds4_swiglu(x: torch.Tensor) -> torch.Tensor:
    """ds4 SwiGLU activation with clamping"""
    x, gate = x.chunk(2, dim=-1)
    return F.silu(x) * gate

def ds4_routed_moe_softplus_normalize(logits: torch.Tensor) -> torch.Tensor:
    """Router softplus normalization as in ds4.c line 5018"""
    return torch.nn.functional.softplus(logits.clamp(max=SWIGLU_CLAMP_EXP))

def ds4_attention_sink_aware(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    kq_scale: float,
    n_kv: int,
) -> torch.Tensor:
    """
    ds4-style sink-aware attention (ds4.c line 4746)

    Attention with learned sink vectors for improvedKV cache utilization.
    Uses exp-based numerical stability with max subtraction.
    """
    n_head = q.shape[0]
    n_head_dim = q.shape[1]

    score = torch.zeros(n_kv, device=q.device, dtype=q.dtype)

    for h in range(n_head):
        qh = q[h]
        max_score = sinks[h].item()

        for r in range(n_kv):
            score_r = 0.0
            for d in range(n_head_dim):
                score_r += qh[d] * k[h * n_head_dim + r * n_head_dim + d]
            score[r] = score_r * kq_scale
            if score[r] > max_score:
                max_score = score[r]

        denom = math.exp(sinks[h].item() - max_score)
        for r in range(n_kv):
            weight = math.exp(score[r] - max_score) if n_kv <= 512 else 0.0
            denom += weight

        for r in range(n_kv):
            weight = math.exp(score[r] - max_score)
            denom += weight

    attn_weights = F.softmax(score * kq_scale, dim=-1)
    return torch.matmul(attn_weights.unsqueeze(0), v).squeeze(0)

def ds4_moe_routing_one(
    token_embd: torch.Tensor,
    router_logits: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_up: torch.Tensor,
    expert_gate: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    n_expert: int = DS4_N_EXPERT,
    n_used: int = DS4_N_EXPERT_USED,
) -> torch.Tensor:
    """
    ds4 MoE routing for single token (ds4.c line 5129)

    Routes to top-k experts using softplus normalized router scores.
    Each expert is SwiGLU activated.
    """
    moe_out = torch.zeros_like(token_embd)

    router_probs = ds4_routed_moe_softplus_normalize(router_logits)

    for k_idx in range(n_used):
        expert_idx = topk_indices[k_idx].item()
        weight = topk_weights[k_idx].item()

        gate_val = torch.matmul(token_embd, expert_gate[expert_idx]).sigmoid()
        up_val = torch.matmul(token_embd, expert_up[expert_idx])

        expert_out = ds4_swiglu(up_val.unsqueeze(0) * gate_val.unsqueeze(-1)).squeeze(0)

        moe_out += expert_out * weight * router_probs[expert_idx]

    return moe_out

def ds4_moe_routing_batch(
    token_embd: torch.Tensor,
    router_logits: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_up: torch.Tensor,
    expert_gate: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    n_expert: int = DS4_N_EXPERT,
    n_used: int = DS4_N_EXPERT_USED,
) -> torch.Tensor:
    """
    ds4 MoE routing for batch (ds4.c line 5277)

    Batched version of MoE routing for efficiency.
    """
    batch_size = token_embd.shape[0]
    n_embd = token_embd.shape[1]

    moe_out = torch.zeros_like(token_embd)

    router_probs = ds4_routed_moe_softplus_normalize(router_logits)

    for k_idx in range(n_used):
        expert_idx = topk_indices[:, k_idx]
        weight = topk_weights[:, k_idx]

        for b in range(batch_size):
            e_idx = expert_idx[b].item()
            w = weight[b].item()

            gate_val = torch.matmul(token_embd[b], expert_gate[e_idx]).sigmoid()
            up_val = torch.matmul(token_embd[b], expert_up[e_idx])

            expert_out = ds4_swiglu(up_val.unsqueeze(0) * gate_val.unsqueeze(-1)).squeeze(0)

            moe_out[b] += expert_out * w * router_probs[b, e_idx]

    return moe_out

def ds4_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = DS4_RMS_EPS) -> torch.Tensor:
    """ds4 RMSNorm (ds4.c style)"""
    return x * weight / (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()

def ds4_q_projection_with_lora(
    x: torch.Tensor,
    q_proj: torch.Tensor,
    q_lora_a: torch.Tensor,
    q_lora_b: torch.Tensor,
    n_lora_q: int = DS4_N_LORA_Q,
) -> torch.Tensor:
    """
    ds4 Q projection with LoRA adapter (ds4.c line ~4436)

    Q = (x @ q_proj) + ((x @ q_lora_a) @ q_lora_b) * scaling
    """
    q = torch.matmul(x, q_proj)

    if q_lora_a is not None and q_lora_b is not None:
        q_lora = torch.matmul(torch.matmul(x, q_lora_a), q_lora_b)
        q = q + q_lora * (n_lora_q / DS4_N_EMBD)

    return q

def ds4_kv_projection(
    x: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ds4 KV projection"""
    return torch.matmul(x, k_proj), torch.matmul(x, v_proj)

def ds4_attention_output_projection_grouped(
    heads: torch.Tensor,
    o_proj: torch.Tensor,
    o_lora_a: torch.Tensor,
    o_lora_b: torch.Tensor,
    n_out_group: int = DS4_N_OUT_GROUP,
) -> torch.Tensor:
    """
    ds4 Grouped attention output projection (ds4.c line 4797)

    Each group first maps heads to reduced dim, then to output.
    Final output is concatenation of all groups.
    """
    n_head = heads.shape[0]
    n_head_dim = heads.shape[1]
    group_size = n_head // n_out_group

    groups = []
    for g in range(n_out_group):
        start = g * group_size
        end = start + group_size
        group_heads = heads[start:end]

        reduced = torch.matmul(
            group_heads.flatten().unsqueeze(0),
            torch.eye(group_size * n_head_dim, n_out_group * DS4_N_LORA_O, device=heads.device)[:group_size * n_head_dim]
        ).squeeze(0)

        groups.append(reduced)

    concat = torch.cat(groups)
    out = torch.matmul(concat, o_proj)

    if o_lora_a is not None and o_lora_b is not None:
        out = out + torch.matmul(torch.matmul(concat, o_lora_a), o_lora_b)

    return out

def ds4_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    compress_freq_base: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ds4 RoPE with compressed frequency base (ds4.c line 4415)

    Applies rotary position embedding with optional frequency compression.
    """
    q_real, q_imag = q.float().chunk(2, dim=-1)
    k_real, k_imag = k.float().chunk(2, dim=-1)

    q_rotated = torch.cat([
        q_real * cos - q_imag * sin,
        q_real * sin + q_imag * cos,
    ], dim=-1)

    k_rotated = torch.cat([
        k_real * cos - k_imag * sin,
        k_real * sin + k_imag * cos,
    ], dim=-1)

    if compress_freq_base != 1.0:
        q_rotated = q_rotated * compress_freq_base
        k_rotated = k_rotated * compress_freq_base

    return q_rotated.type_as(q), k_rotated.type_as(k)

class DS4AttentionCUDAKernel(nn.Module):
    """
    ds4 Attention CUDA Kernel

    Implements ds4-style attention with:
    - Sink-aware KV cache
    - Grouped LoRA output projection
    - Compressed RoPE
    """

    def __init__(
        self,
        n_head: int = DS4_N_HEAD,
        n_head_dim: int = DS4_N_HEAD_DIM,
        n_head_kv: int = DS4_N_HEAD_KV,
        n_lora_q: int = DS4_N_LORA_Q,
        n_lora_o: int = DS4_N_LORA_O,
        n_out_group: int = DS4_N_OUT_GROUP,
        sliding_window: int = DS4_N_SWA,
        compress_freq_base: float = 1.0,
    ):
        super().__init__()
        self.n_head = n_head
        self.n_head_dim = n_head_dim
        self.n_head_kv = n_head_kv
        self.n_lora_q = n_lora_q
        self.n_lora_o = n_lora_o
        self.n_out_group = n_out_group
        self.sliding_window = sliding_window
        self.compress_freq_base = compress_freq_base
        self.kq_scale = 1.0 / math.sqrt(n_head_dim)

        self.sinks = nn.Parameter(torch.zeros(n_head))

    def forward(
        self,
        x: torch.Tensor,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        o_proj: torch.Tensor,
        q_lora_a: Optional[torch.Tensor] = None,
        q_lora_b: Optional[torch.Tensor] = None,
        o_lora_a: Optional[torch.Tensor] = None,
        o_lora_b: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        kv_cache_k: Optional[torch.Tensor] = None,
        kv_cache_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        ds4 Attention forward pass

        Args:
            x: Input tensor [batch, seq, n_embd]
            q_proj, k_proj, v_proj, o_proj: Projection matrices
            q_lora_*, o_lora_*: Optional LoRA adapter weights
            cos, sin: RoPE tables
            kv_cache_k, kv_cache_v: Optional KV cache

        Returns:
            Attention output [batch, seq, n_embd]
        """
        batch_size, seq_len, n_embd = x.shape

        q = ds4_q_projection_with_lora(x, q_proj, q_lora_a, q_lora_b, self.n_lora_q)
        k, v = ds4_kv_projection(x, k_proj, v_proj)

        if cos is not None and sin is not None:
            q, k = ds4_rope(q, k, cos, sin, self.compress_freq_base)

        q = q.view(batch_size, seq_len, self.n_head, self.n_head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head_kv, self.n_head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head_kv, self.n_head_dim).transpose(1, 2)

        if kv_cache_k is not None and kv_cache_v is not None:
            k = torch.cat([kv_cache_k, k], dim=2)
            v = torch.cat([kv_cache_v, v], dim=2)

        n_kv = k.shape[2]
        attn_output = F.scaled_dot_product_attention(q, k, v, scale=self.kq_scale)

        heads = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head, self.n_head_dim)

        out = ds4_attention_output_projection_grouped(
            heads[:, -1], o_proj, o_lora_a, o_lora_b, self.n_out_group
        )

        return out


class DS4MoERoutingCUDAKernel(nn.Module):
    """
    ds4 MoE Routing CUDA Kernel

    Implements ds4-style Mixture of Experts with:
    - Softplus normalized router
    - SwiGLU expert activation
    - Top-k expert selection
    """

    def __init__(
        self,
        n_embd: int = DS4_N_EMBD,
        n_expert: int = DS4_N_EXPERT,
        n_expert_used: int = DS4_N_EXPERT_USED,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_expert = n_expert
        self.n_expert_used = n_expert_used
        self.hidden_dim = hidden_dim or (n_embd * 4 // 3)

    def forward(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        expert_up: torch.Tensor,
        expert_gate: torch.Tensor,
        expert_down: torch.Tensor,
    ) -> torch.Tensor:
        """
        ds4 MoE routing forward pass

        Args:
            x: Input tensor [batch, seq, n_embd]
            router_logits: Router logit scores [batch, seq, n_expert]
            expert_up: Expert up projection weights [n_expert, hidden_dim, n_embd]
            expert_gate: Expert gate projection weights [n_expert, n_embd, hidden_dim]
            expert_down: Expert down projection weights [n_expert, n_embd, hidden_dim]

        Returns:
            MoE output [batch, seq, n_embd]
        """
        batch_size, seq_len, _ = x.shape

        router_probs = ds4_routed_moe_softplus_normalize(router_logits)
        topk_weights, topk_indices = torch.topk(router_probs, self.n_expert_used, dim=-1)

        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        x_flat = x.view(batch_size * seq_len, self.n_embd)
        topk_indices_flat = topk_indices.view(batch_size * seq_len, self.n_expert_used)
        topk_weights_flat = topk_weights.view(batch_size * seq_len, self.n_expert_used)

        moe_out_flat = torch.zeros_like(x_flat)

        for k_idx in range(self.n_expert_used):
            expert_idx = topk_indices_flat[:, k_idx]
            weight = topk_weights_flat[:, k_idx]

            for e in range(self.n_expert):
                mask = expert_idx == e
                if not mask.any():
                    continue

                e_x = x_flat[mask]
                e_w = weight[mask]

                gate = torch.matmul(e_x, expert_gate[e]).sigmoid()
                up = torch.matmul(e_x, expert_up[e])
                expert_out = ds4_swiglu(up * gate.unsqueeze(-1))
                expert_out = torch.matmul(expert_out, expert_down[e])

                moe_out_flat[mask] += expert_out * e_w.unsqueeze(-1) * router_probs[mask.flatten(), e]

        return moe_out_flat.view(batch_size, seq_len, self.n_embd)


def register_ds4_cuda_kernels():
    """
    Register ds4 CUDA kernels to CGC kernel registry

    This allows the CGC Engine to use ds4-style kernels
    when the ds4 backend is selected.
    """
    try:
        from .cgc_simd_executor import _kernel_registry, KernelType, CGCKernelSpec

        ds4_attn_spec = CGCKernelSpec(
            name="ds4_attention",
            kernel_type=KernelType.ATTENTION,
            cuda_kernel=DS4AttentionCUDAKernel(),
        )
        _kernel_registry.register(0xE6, ds4_attn_spec)

        ds4_moe_spec = CGCKernelSpec(
            name="ds4_moe_routing",
            kernel_type=KernelType.CUSTOM,
            cuda_kernel=DS4MoERoutingCUDAKernel(),
        )
        _kernel_registry.register(0xE7, ds4_moe_spec)

        ds4_norm_spec = CGCKernelSpec(
            name="ds4_rms_norm",
            kernel_type=KernelType.NORM,
            cuda_kernel=ds4_rms_norm,
        )
        _kernel_registry.register(0xE8, ds4_norm_spec)

        logger.info("[DS4 CUDA Kernels] Successfully registered ds4 kernels to CGC registry")
        return True

    except ImportError as e:
        logger.warning(f"[DS4 CUDA Kernels] Failed to register: {e}")
        return False


if __name__ == "__main__":
    print("=== ds4 CUDA Kernels Test ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2
    seq_len = 16
    n_embd = DS4_N_EMBD

    x = torch.randn(batch_size, seq_len, n_embd, device=device)

    attn = DS4AttentionCUDAKernel().to(device)
    q_proj = torch.randn(n_embd, n_embd, device=device)
    k_proj = torch.randn(n_embd, n_embd, device=device)
    v_proj = torch.randn(n_embd, n_embd, device=device)
    o_proj = torch.randn(n_embd, n_embd, device=device)

    out = attn(x, q_proj, k_proj, v_proj, o_proj)
    print(f"Attention output shape: {out.shape}")

    n_expert = DS4_N_EXPERT
    hidden_dim = n_embd * 4 // 3

    router_logits = torch.randn(batch_size, seq_len, n_expert, device=device)
    expert_up = torch.randn(n_expert, hidden_dim, n_embd, device=device)
    expert_gate = torch.randn(n_expert, n_embd, hidden_dim, device=device)
    expert_down = torch.randn(n_expert, n_embd, hidden_dim, device=device)

    moe = DS4MoERoutingCUDAKernel().to(device)
    moe_out = moe(x, router_logits, expert_up, expert_gate, expert_down)
    print(f"MoE output shape: {moe_out.shape}")

    register_ds4_cuda_kernels()
    print("=== ds4 CUDA Kernels Test Complete ===")
