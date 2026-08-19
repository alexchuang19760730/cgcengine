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
vLLM + MagiCompiler + CGC + FlashKDA 全栈集成

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        vLLM Inference Engine                        │
    │                     (调度 + PagedAttention + Sampling)               │
    └─────────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │                   CGC SIMD Command Layer                            │
    │                    40 Commands (0x01 - 0xA4)                        │
    ├─────────────────────────────────────────────────────────────────────┤
    │  0x10-0x1F │ ATTENTION  │ SDPA, KDA, PagedAttention, Flash          │
    │  0x20-0x2F │ LINEAR     │ GEMM, Linear+Bias, BatchedGEMM            │
    │  0x30-0x3F │ NORM       │ LayerNorm, RMSNorm, GroupNorm             │
    │  0x40-0x4F │ ROPE       │ RoPE, RoPE_Fused, YaRN                    │
    │  0x50-0x5F │ ACTIVATION │ SiLU, GELU, ReLU, Sigmoid                 │
    │  0x60-0x6F │ SAMPLING   │ Softmax, TopK, TopP, Temperature          │
    │  0x70-0x7F │ MEMORY     │ KV_Load, KV_Store, KV_Update, Embedding  │
    │  0x80-0x82 │ KDA        │ KDA_Chunk, KDA_Project, KDA_OrthoUpdate  │
    │  0x90-0x9F │ DISTRIBUTED│ AllReduce, AllGather, ReduceScatter       │
    │  0xA0-0xAF │ QUANT      │ W8A16, W4A16, Dequant, GPTQ, AWQ          │
    └─────────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    CGC SIMD Executor                                │
    │  - Command Dispatcher    - Kernel Launcher                          │
    │  - Workspace Manager     - Memory Pool                              │
    └─────────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │                  CUDA Kernel Registry                                │
    ├─────────────────────────────────────────────────────────────────────┤
    │  FlashKDA      │ MoonshotAI 官方 CUDA Kernel (0x80)                │
    │  cuBLAS        │ GEMM/MatMul (0x20, 0x21, 0x22)                    │
    │  cuDNN         │ FlashAttention (0x13)                              │
    │  NCCL          │ Distributed Ops (0x90-0x92)                       │
    │  Custom        │ KDA Project/OrthoUpdate (0x81, 0x82)              │
    └─────────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         GPU (CUDA)                                   │
    │              H100 / H200 / B200 / RTX 5090                          │
    └─────────────────────────────────────────────────────────────────────┘

集成文件:
    vllm_kda_ops.py        - 可复制到 vllm/attention/backends/kda/kda_ops.py
    vllm_kda_attention.py  - 可复制到 vllm/attention/backends/kda/kda_attention.py

使用方法:
    from cgc_engine.cgc import VLLMCGCBackend

    backend = VLLMCGCBackend()
    output = backend.forward(input_ids, positions, ...)

启动 vLLM:
    from vllm import LLM
    llm = LLM(
        model="moonshotai/Kimi-Linear-48B-A3B-Instruct",
        attention_backend="cgc_kda",  # 使用 CGC KDA Backend
        device="cuda",
        dtype=torch.bfloat16,
    )
"""

from .cgc_commands import (
    CGCInstruction,
    CGCInstructionType,
    CGC_SIMD_COMMAND_SET,
)

from .cgc_opcodes import (
    CGC_OP_CODES,
    get_opcode_name,
    get_opcode_value,
    list_attention_opcodes,
    list_kda_opcodes,
    is_attention_opcode,
    is_kda_opcode,
    get_category,
)

from .cgc_simd_executor import (
    CGCExecutor,
    CGCKernelRegistry,
    CGCCommand,
    KernelType,
    register_cuda_kernel,
    execute_cgc_command,
)

from .flashkda_integration import (
    FlashKDALayer,
    create_flashkda_attention,
    get_cgc_kda_metadata,
    FLASHKDA_AVAILABLE,
)

from .vllm_integration import (
    VLLMKDAConfig,
    VLLMKDABackend,
    create_vllm_kda_backend,
)

from .vllm_cgc_backend import (
    VLLMCGCBackend,
    VLLMCGCConfig,
    VLLMModelCGC,
    create_vllm_cgc_model,
)

from .vllm_kda_ops import (
    flash_kda_cgc_forward,
    flash_kda_cgc_forward_native,
    sdpa_cgc_forward,
    rms_norm_cgc_forward,
    rope_cgc_forward,
    silu_cgc_forward,
    softmax_cgc_forward,
    get_cgc_backend_info,
)

from .vllm_kda_attention import (
    CGCKDABackend,
    CGCKDABackendConfig,
    create_cgc_kda_backend,
)

__all__ = [
    # CGC Core
    "CGCExecutor",
    "CGCKernelRegistry",
    "CGCCommand",
    "CGC_OP_CODES",
    "CGCInstruction",
    "CGCInstructionType",
    "KernelType",

    # CGC Commands
    "CGC_SIMD_COMMAND_SET",
    "get_opcode_name",
    "get_opcode_value",
    "list_attention_opcodes",
    "list_kda_opcodes",
    "is_attention_opcode",
    "is_kda_opcode",
    "get_category",

    # Kernel Execution
    "register_cuda_kernel",
    "execute_cgc_command",

    # FlashKDA
    "FlashKDALayer",
    "create_flashkda_attention",
    "get_cgc_kda_metadata",
    "FLASHKDA_AVAILABLE",

    # vLLM Integration
    "VLLMKDAConfig",
    "VLLMKDABackend",
    "create_vllm_kda_backend",

    # vLLM CGC Backend
    "VLLMCGCBackend",
    "VLLMCGCConfig",
    "VLLMModelCGC",
    "create_vllm_cgc_model",

    # vLLM KDA Ops (可复制到 vLLM)
    "flash_kda_cgc_forward",
    "flash_kda_cgc_forward_native",
    "sdpa_cgc_forward",
    "rms_norm_cgc_forward",
    "rope_cgc_forward",
    "silu_cgc_forward",
    "softmax_cgc_forward",
    "get_cgc_backend_info",

    # vLLM CGC Attention Backend
    "CGCKDABackend",
    "CGCKDABackendConfig",
    "create_cgc_kda_backend",
]


def get_full_stack_info() -> dict:
    """
    获取完整技术栈信息
    """
    return {
        "version": "1.0.0",
        "architecture": "vLLM + MagiCompiler + CGC + FlashKDA",
        "total_cgc_commands": len(CGC_OP_CODES),
        "flashkda_available": FLASHKDA_AVAILABLE,
        "components": {
            "vllm": {
                "role": "推理引擎 + 调度",
                "backends": ["cgc_kda", "flash_kda", "sdpa"],
            },
            "magicompiler": {
                "role": "整图编译 + 显存优化",
                "features": ["full_graph", "memory_lake", "fsdp_aware"],
            },
            "cgc": {
                "role": "SIMD 命令层",
                "commands": len(CGC_OP_CODES),
                "categories": 10,
            },
            "flashkda": {
                "role": "CUDA Kernel 执行",
                "vendor": "MoonshotAI",
                "repo": "github.com/MoonshotAI/FlashKDA",
            },
        },
        "supported_ops": {
            "attention": list_attention_opcodes(),
            "kda": list_kda_opcodes(),
        },
        "gpu_optimizations": [
            "FlashKDA Chunk-based KDA",
            "PagedAttention KV Cache",
            "cuBLAS GEMM Fusion",
            "NCCL Distributed AllReduce",
            "AWQ/GPTQ Quantization",
        ],
    }
