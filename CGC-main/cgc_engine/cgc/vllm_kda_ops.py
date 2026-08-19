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
vLLM CGC Attention Backend - 可直接复制到 vLLM

此文件可复制到: vllm/attention/backends/kda/kda_ops.py

使用方法:
    1. 复制此文件内容到 vllm/attention/backends/kda/kda_ops.py
    2. 确保 cgc_engine 已安装
    3. 启动 vLLM 时指定 attention_backend="cgc_kda"
"""

import torch
from typing import Optional, Tuple, Dict, Any

try:
    from cgc_engine.cgc import (
        CGCExecutor,
        CGC_OP_CODES,
        FlashKDALayer,
        CGCKernelRegistry,
    )
    from cgc_engine.cgc.flashkda_integration import _check_flashkda_available
    CGC_AVAILABLE = True
except ImportError:
    CGC_AVAILABLE = False
    CGC_OP_CODES = None


_cgc_executor: Optional[CGCExecutor] = None
_flashkda_layer: Optional[FlashKDALayer] = None
_kernel_registry: Optional[CGCKernelRegistry] = None


def _get_cgc_executor() -> CGCExecutor:
    """获取全局 CGC 执行器"""
    global _cgc_executor
    if _cgc_executor is None:
        _cgc_executor = CGCExecutor(enable_profiling=False)
    return _cgc_executor


def _get_kernel_registry() -> CGCKernelRegistry:
    """获取全局 Kernel 注册表"""
    global _kernel_registry
    if _kernel_registry is None:
        _kernel_registry = CGCKernelRegistry()
    return _kernel_registry


def _ensure_flashkda() -> bool:
    """确保 FlashKDA 可用"""
    if not CGC_AVAILABLE:
        return False
    return _check_flashkda_available()


def flash_kda_cgc_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 1.0,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: float = -5.0,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    FlashKDA Forward via CGC SIMD Executor

    这是 vLLM KDA 的核心 forward 函数，所有计算通过 CGC 命令执行。

    Args:
        q: Query tensor [B, T, H, K] or [T, H, K]
        k: Key tensor [B, T, H, K] or [T, H, K]
        v: Value tensor [B, T, H, V] or [T, H, V]
        scale: Attention scale
        g: Gate tensor [B, T, H, K] (optional, for KDA gate)
        beta: Beta tensor [B, T, H] (optional, for KDA beta sigmoid)
        A_log: Log gate parameter [H]
        dt_bias: Gate bias [H, K]
        lower_bound: Gate lower bound
        initial_state: Initial recurrent state
        final_state: Output buffer for final state
        cu_seqlens: Cumulative sequence lengths for varlen

    Returns:
        (output, final_state) tuple
    """
    if not _ensure_flashkda():
        raise RuntimeError(
            "FlashKDA not available. Please install:\n"
            "  1. FlashKDA: git clone https://github.com/MoonshotAI/FlashKDA.git && pip install -v .\n"
            "  2. MagiCompiler: Ensure cgc_engine is in PYTHONPATH"
        )

    exec = _get_cgc_executor()

    inputs = [q, k, v]
    if g is not None:
        inputs.append(g)
    if beta is not None:
        inputs.append(beta)

    params = {
        "scale": scale,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "lower_bound": lower_bound,
        "cu_seqlens": cu_seqlens,
    }

    command = exec.CGCCommand(
        opcode=CGC_OP_CODES.KDA_CHUNK,
        inputs=inputs,
        outputs=[],
        params=params,
        workspace=None,
    )

    outputs = exec.execute(command)
    return outputs[0], final_state


def flash_kda_cgc_forward_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: float = -5.0,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    FlashKDA Forward - Native CGC (不依赖外部库)

    当 cgc_engine 不可用时，使用原生 PyTorch 实现。
    """
    if g is None:
        g = torch.ones_like(q)
    if beta is None:
        beta = torch.ones(q.shape[0], q.shape[1], q.shape[2], device=q.device, dtype=q.dtype)

    B, T, H, K = q.shape
    V = v.shape[-1]

    out = torch.empty_like(v)

    workspace_size = 64 * K * 2 * H * 16 + 64 * 64 * 2 * H * 16 + K * 4 * H * 16
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=q.device)

    if A_log is None:
        A_log = torch.zeros(H, dtype=torch.float32, device=q.device)
    if dt_bias is None:
        dt_bias = torch.zeros(H, K, dtype=torch.float32, device=q.device)

    try:
        import flash_kda as _flash_kda
        _flash_kda.fwd(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=scale, out=out,
            workspace=workspace,
            A_log=A_log, dt_bias=dt_bias,
            lower_bound=lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
        )
    except ImportError:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

    return out, final_state


def kda_project_forward(
    k: torch.Tensor,
    proj_dim: int = 128,
    ortho_transform: bool = True,
) -> torch.Tensor:
    """
    KDA K Projection via CGC

    Args:
        k: Key tensor [B, T, H, K]
        proj_dim: Projection dimension
        ortho_transform: Use orthogonal transform

    Returns:
        Projected K tensor [B, T, H, proj_dim]
    """
    exec = _get_cgc_executor()

    params = {
        "proj_dim": proj_dim,
        "ortho_transform": ortho_transform,
    }

    command = exec.CGCCommand(
        opcode=CGC_OP_CODES.KDA_PROJECT,
        inputs=[k],
        outputs=[],
        params=params,
    )

    outputs = exec.execute(command)
    return outputs[0]


def kda_ortho_update(
    proj_kv: torch.Tensor,
    global_basis: torch.Tensor,
    decay: float = 0.99,
    gram_schmidt_iter: int = 1,
) -> torch.Tensor:
    """
    KDA Orthogonal Basis Update via CGC

    Args:
        proj_kv: Projected KV tensor
        global_basis: Global orthogonal basis to update
        decay: Decay factor for incremental update
        gram_schmidt_iter: Number of Gram-Schmidt iterations

    Returns:
        Updated global basis
    """
    exec = _get_cgc_executor()

    params = {
        "decay": decay,
        "gram_schmidt_iter": gram_schmidt_iter,
    }

    command = exec.CGCCommand(
        opcode=CGC_OP_CODES.KDA_ORTHO_UPDATE,
        inputs=[proj_kv, global_basis],
        outputs=[],
        params=params,
    )

    outputs = exec.execute(command)
    return outputs[0]


def sdpa_cgc_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 1.0,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    SDPA (Scaled Dot Product Attention) via CGC

    Args:
        q: Query [B, T, H, D]
        k: Key [B, T, H, D]
        v: Value [B, T, H, D]
        scale: Attention scale
        dropout_p: Dropout probability
        is_causal: Use causal mask
        attn_mask: Optional attention mask

    Returns:
        Attention output
    """
    exec = _get_cgc_executor()

    params = {
        "scale": scale,
        "dropout_p": dropout_p,
        "is_causal": is_causal,
    }

    inputs = [q, k, v]
    if attn_mask is not None:
        inputs.append(attn_mask)

    command = exec.CGCCommand(
        opcode=CGC_OP_CODES.ATTENTION_SDPA,
        inputs=inputs,
        outputs=[],
        params=params,
    )

    outputs = exec.execute(command)
    return outputs[0]


def rms_norm_cgc_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    RMSNorm via CGC
    """
    exec = _get_cgc_executor()

    command = exec.CGCCommand(
        opcode=CGC_OP_CODES.RMS_NORM,
        inputs=[x, weight],
        outputs=[],
        params={"eps": eps, "normalized_shape": x.shape[-1]},
    )

    outputs = exec.execute(command)
    return outputs[0]


def rope_cgc_forward(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    RoPE (Rotary Position Embedding) via CGC
    """
    exec = _get_cgc_executor()

    command = exec.CGCCommand(
        opcode=CGC_OP_CODES.ROPE_FUSED,
        inputs=[x, cos, sin],
        outputs=[],
        params={},
    )

    outputs = exec.execute(command)
    return outputs[0]


def silu_cgc_forward(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU (Sigmoid Linear Unit) via CGC
    """
    exec = _get_cgc_executor()

    command = exec.CGCCommand(
        opcode=CGC_OP_CODES.SILU,
        inputs=[x],
        outputs=[],
        params={"inplace": False},
    )

    outputs = exec.execute(command)
    return outputs[0]


def softmax_cgc_forward(
    x: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Softmax via CGC
    """
    exec = _get_cgc_executor()

    command = exec.CGCCommand(
        opcode=CGC_OP_CODES.SOFTMAX,
        inputs=[x],
        outputs=[],
        params={"dim": dim, "log_softmax": False},
    )

    outputs = exec.execute(command)
    return outputs[0]


def get_cgc_backend_info() -> Dict[str, Any]:
    """
    获取 CGC Backend 信息
    """
    return {
        "cgc_available": CGC_AVAILABLE,
        "flashkda_available": _ensure_flashkda() if CGC_AVAILABLE else False,
        "executor_initialized": _cgc_executor is not None,
        "kernel_registry_initialized": _kernel_registry is not None,
        "total_opcodes": len(CGC_OP_CODES) if CGC_OP_CODES else 0,
    }


def reset_cgc_executor():
    """重置全局 CGC 执行器"""
    global _cgc_executor, _kernel_registry, _flashkda_layer
    _cgc_executor = None
    _kernel_registry = None
    _flashkda_layer = None
