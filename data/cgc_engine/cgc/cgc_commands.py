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
CGC SIMD Command Definitions - Full Stack vLLM Coverage

This module defines the COMPLETE unified SIMD instruction set covering ALL vLLM operations.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                   CGC SIMD COMMAND SET                      │
    ├─────────────────────────────────────────────────────────────┤
    │ ATTENTION (0x10-0x1F)        │ 注意力操作                    │
    │ LINEAR/GEMM (0x20-0x2F)      │ 线性层/GEMM操作              │
    │ NORM (0x30-0x3F)             │ 归一化操作                    │
    │ ROPE (0x40-0x4F)             │ 旋转位置编码                  │
    │ ACTIVATION (0x50-0x5F)      │ 激活函数                      │
    │ SAMPLING (0x60-0x6F)        │ 采样策略                      │
    │ MEMORY (0x70-0x7F)          │ 显存管理                      │
    │ KDA (0x80-0x8F)             │ Kimi KDA 指令               │
    │ DISTRIBUTED (0x90-0x9F)     │ 分布式操作                    │
    │ QUANTIZATION (0xA0-0xAF)    │ 量化操作                      │
    └─────────────────────────────────────────────────────────────┘
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


class CGCInstructionType(Enum):
    # Original CGC types
    WEIGHT_STAY = "WEIGHT_STAY"
    LAYER_STREAM_LOAD = "LAYER_STREAM_LOAD"
    LAYER_FORWARD = "LAYER_FORWARD"
    LAYER_BACKWARD = "LAYER_BACKWARD"
    LAYER_OPTIM_UPDATE = "LAYER_OPTIM_UPDATE"
    STREAM_PIPELINE = "STREAM_PIPELINE"
    EXPERT_LOAD = "EXPERT_LOAD"
    EXPERT_OFFLOAD = "EXPERT_OFFLOAD"
    ORTHO_BASIS_UPDATE = "ORTHO_BASIS_UPDATE"
    BRANCH_PARALLEL = "BRANCH_PARALLEL"
    CHECKPOINT_SAVE = "CHECKPOINT_SAVE"

    # Attention types
    ATTENTION = "ATTENTION"
    ATTENTION_KDA = "ATTENTION_KDA"
    ATTENTION_SDPA = "ATTENTION_SDPA"
    ATTENTION_PAGED = "ATTENTION_PAGED"

    # Linear types
    LINEAR = "LINEAR"
    LINEAR_GEMM = "LINEAR_GEMM"
    LINEAR_GEMV = "LINEAR_GEMV"

    # Norm types
    LAYER_NORM = "LAYER_NORM"
    RMS_NORM = "RMS_NORM"

    # Position encoding
    ROPE = "ROPE"
    YARN_ROPE = "YARN_ROPE"

    # Activation
    SILU = "SILU"
    GELU = "GELU"
    RELU = "RELU"
    TANH = "TANH"
    SIGMOID = "SIGMOID"

    # Sampling
    SOFTMAX = "SOFTMAX"
    TOP_K = "TOP_K"
    TOP_P = "TOP_P"
    TEMPERATURE_SAMPLE = "TEMPERATURE_SAMPLE"

    # Memory
    KV_CACHE_LOAD = "KV_CACHE_LOAD"
    KV_CACHE_STORE = "KV_CACHE_STORE"
    KV_CACHE_UPDATE = "KV_CACHE_UPDATE"
    KV_CACHE_STATIC_LAYOUT = "KV_CACHE_STATIC_LAYOUT"
    KV_CACHE_COMMIT = "KV_CACHE_COMMIT"
    BATCH_COMPILE = "BATCH_COMPILE"
    BATCH_MERGE = "BATCH_MERGE"

    # Distributed
    ALL_REDUCE = "ALL_REDUCE"
    ALL_GATHER = "ALL_GATHER"
    REDUCE_SCATTER = "REDUCE_SCATTER"

    # Quantization
    QUANTIZE = "QUANTIZE"
    DEQUANTIZE = "DEQUANTIZE"
    GPTQ_KERNEL = "GPTQ_KERNEL"
    AWQ_KERNEL = "AWQ_KERNEL"
    FP8_E4M3_QUANT = "FP8_E4M3_QUANT"
    FP8_E5M2_QUANT = "FP8_E5M2_QUANT"
    FP8_DEQUANT = "FP8_DEQUANT"

    # LLAMA.CPP / GGUF types
    LLAMA_GGUF_LOAD = "LLAMA_GGUF_LOAD"
    LLAMA_GGUF_QUANTIZE = "LLAMA_GGUF_QUANTIZE"
    LLAMA_GGUF_DEQUANTIZE = "LLAMA_GGUF_DEQUANTIZE"
    LLAMA_Q4_K_MATMUL = "LLAMA_Q4_K_MATMUL"
    LLAMA_Q5_K_MATMUL = "LLAMA_Q5_K_MATMUL"
    LLAMA_Q6_K_MATMUL = "LLAMA_Q6_K_MATMUL"
    LLAMA_Q8_0_MATMUL = "LLAMA_Q8_0_MATMUL"
    LLAMA_Q2_K_MATMUL = "LLAMA_Q2_K_MATMUL"
    LLAMA_Q3_K_MATMUL = "LLAMA_Q3_K_MATMUL"
    LLAMA_Q8_K_MATMUL = "LLAMA_Q8_K_MATMUL"
    LLAMA_MOE_ROUTING = "LLAMA_MOE_ROUTING"
    LLAMA_MOE_EXPERT_FWD = "LLAMA_MOE_EXPERT_FWD"
    LLAMA_ROPE_GGUF = "LLAMA_ROPE_GGUF"
    LLAMA_RMSNORM_GGUF = "LLAMA_RMSNORM_GGUF"
    LLAMA_SILU_GGUF = "LLAMA_SILU_GGUF"
    LLAMA_GELU_GGUF = "LLAMA_GELU_GGUF"
    LLAMA_KV_CACHE_GGUF = "LLAMA_KV_CACHE_GGUF"
    LLAMA_SAMPLING_GGUF = "LLAMA_SAMPLING_GGUF"
    LLAMA_INFERENCE = "LLAMA_INFERENCE"
    LLAMA_EMBEDDING_GGUF = "LLAMA_EMBEDDING_GGUF"
    LLAMA_DETOKENIZE_GGUF = "LLAMA_DETOKENIZE_GGUF"
    LLAMA_TOKENIZE_GGUF = "LLAMA_TOKENIZE_GGUF"


@dataclass
class CGCInstruction:
    name: str
    opcode: int
    params: Dict[str, Any]
    description: str
    module: str
    category: str = "general"


@dataclass
class KDAInstruction:
    q: str
    k: str
    v: str
    g: Optional[str] = None
    beta: Optional[str] = None
    scale: float = 1.0
    A_log: Optional[str] = None
    dt_bias: Optional[str] = None
    lower_bound: float = -5.0
    initial_state: Optional[str] = None
    final_state: Optional[str] = None
    cu_seqlens: Optional[str] = None


# ============================================================================
# ATTENTION COMMANDS (0x10-0x1F)
# ============================================================================

ATTENTION_SDPA_CMD = CGCInstruction(
    name="ATTENTION_SDPA",
    opcode=0x10,
    params={
        "scale": 1.0,
        " dropout_p": 0.0,
        "is_causal": True,
        "attn_mask_format": "PADDED_MASK",
    },
    description="Scaled Dot Product Attention (FlashAttention compatible)",
    module="vLLM Attention",
    category="attention",
)

ATTENTION_KDA_CMD = CGCInstruction(
    name="ATTENTION_KDA",
    opcode=0x11,
    params={
        "dtype": "bf16",
        "chunk_size": 64,
        "use_gate": True,
        "use_qk_l2norm": True,
        "use_beta_sigmoid": True,
        "safe_gate": True,
    },
    description="Kimi Delta Attention with FlashKDA kernel",
    module="FlashKDA",
    category="attention",
)

ATTENTION_PAGED_CMD = CGCInstruction(
    name="ATTENTION_PAGED",
    opcode=0x12,
    params={
        "block_size": 16,
        "num_kv_heads": 8,
        "scale": 1.0,
        "op_name": "paged_attention_v1",
    },
    description="PagedAttention for vLLM KV cache management",
    module="vLLM PagedAttention",
    category="attention",
)

ATTENTION_FLASH_CMD = CGCInstruction(
    name="ATTENTION_FLASH",
    opcode=0x13,
    params={
        "flash_type": "flash2",
        "scale": 1.0,
        "dropout_p": 0.0,
    },
    description="Flash Attention v2/v3",
    module="FlashAttention",
    category="attention",
)

# ============================================================================
# LINEAR/GEMM COMMANDS (0x20-0x2F)
# ============================================================================

LINEAR_GEMM_CMD = CGCInstruction(
    name="LINEAR_GEMM",
    opcode=0x20,
    params={
        "trans_a": False,
        "trans_b": True,
        "beta": 0.0,
        "alpha": 1.0,
        "precision": "fp16",
    },
    description="General Matrix Multiply (cuBLAS)",
    module="cuBLAS",
    category="linear",
)

LINEAR_BIAS_CMD = CGCInstruction(
    name="LINEAR_BIAS",
    opcode=0x21,
    params={
        "bias_term": True,
        "activation": None,
    },
    description="Linear layer with optional bias",
    module="vLLM MLP",
    category="linear",
)

GEMM_BATCHED_CMD = CGCInstruction(
    name="GEMM_BATCHED",
    opcode=0x22,
    params={
        "trans_a": False,
        "trans_b": True,
        "batch_count": 0,
    },
    description="Batched GEMM for multi-head operations",
    module="cuBLAS",
    category="linear",
)

# ============================================================================
# NORM COMMANDS (0x30-0x3F)
# ============================================================================

LAYER_NORM_CMD = CGCInstruction(
    name="LAYER_NORM",
    opcode=0x30,
    params={
        "eps": 1e-5,
        "elementwise_affine": True,
    },
    description="Layer Normalization",
    module="vLLM Norm",
    category="norm",
)

RMS_NORM_CMD = CGCInstruction(
    name="RMS_NORM",
    opcode=0x31,
    params={
        "eps": 1e-6,
        "elementwise_affine": True,
    },
    description="RMS Normalization (used in LLaMA, Mistral)",
    module="vLLM Norm",
    category="norm",
)

GROUP_NORM_CMD = CGCInstruction(
    name="GROUP_NORM",
    opcode=0x32,
    params={
        "num_groups": 32,
        "eps": 1e-5,
    },
    description="Group Normalization",
    module="vLLM Norm",
    category="norm",
)

# ============================================================================
# ROPE COMMANDS (0x40-0x4F)
# ============================================================================

ROPE_CMD = CGCInstruction(
    name="ROPE",
    opcode=0x40,
    params={
        "base": 10000.0,
        "max_position": 8192,
        "rope_type": "default",
        "transformer_dim": 4096,
    },
    description="Rotary Position Embedding",
    module="vLLM RoPE",
    category="position",
)

ROPE_FUSED_CMD = CGCInstruction(
    name="ROPE_FUSED",
    opcode=0x41,
    params={
        "base": 10000.0,
        "interleaved": False,
    },
    description="Fused RoPE computation",
    module="vLLM RoPE",
    category="position",
)

YARN_ROPE_CMD = CGCInstruction(
    name="YARN_ROPE",
    opcode=0x42,
    params={
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "original_max_position": 8192,
    },
    description="YaRN RoPE for extended context",
    module="vLLM YaRN",
    category="position",
)

# ============================================================================
# ACTIVATION COMMANDS (0x50-0x5F)
# ============================================================================

SILU_CMD = CGCInstruction(
    name="SILU",
    opcode=0x50,
    params={
        "inplace": True,
    },
    description="Sigmoid Linear Unit activation (SwiGLU gate)",
    module="vLLM Activation",
    category="activation",
)

GELU_CMD = CGCInstruction(
    name="GELU",
    opcode=0x51,
    params={
        "approximate": "none",
        "inplace": False,
    },
    description="Gaussian Error Linear Unit",
    module="vLLM Activation",
    category="activation",
)

GELU_TANH_CMD = CGCInstruction(
    name="GELU_TANH",
    opcode=0x52,
    params={
        "inplace": False,
    },
    description="GELU with tanh approximation",
    module="vLLM Activation",
    category="activation",
)

RELU_CMD = CGCInstruction(
    name="RELU",
    opcode=0x53,
    params={
        "inplace": True,
    },
    description="Rectified Linear Unit",
    module="vLLM Activation",
    category="activation",
)

SIGMOID_CMD = CGCInstruction(
    name="SIGMOID",
    opcode=0x54,
    params={
        "inplace": False,
    },
    description="Sigmoid activation",
    module="vLLM Activation",
    category="activation",
)

# ============================================================================
# SAMPLING COMMANDS (0x60-0x6F)
# ============================================================================

SOFTMAX_CMD = CGCInstruction(
    name="SOFTMAX",
    opcode=0x60,
    params={
        "dim": -1,
        "log_softmax": False,
    },
    description="Softmax operation",
    module="vLLM Sampling",
    category="sampling",
)

LOG_SOFTMAX_CMD = CGCInstruction(
    name="LOG_SOFTMAX",
    opcode=0x61,
    params={
        "dim": -1,
    },
    description="Log Softmax operation",
    module="vLLM Sampling",
    category="sampling",
)

TOP_K_CMD = CGCInstruction(
    name="TOP_K",
    opcode=0x62,
    params={
        "k": 50,
        "sorted": True,
    },
    description="Top-K sampling/probs modification",
    module="vLLM Sampling",
    category="sampling",
)

TOP_P_CMD = CGCInstruction(
    name="TOP_P",
    opcode=0x63,
    params={
        "p": 0.9,
        "min_tokens_to_keep": 1,
    },
    description="Top-P (nucleus) sampling",
    module="vLLM Sampling",
    category="sampling",
)

TEMPERATURE_CMD = CGCInstruction(
    name="TEMPERATURE",
    opcode=0x64,
    params={
        "temperature": 1.0,
    },
    description="Temperature scaling",
    module="vLLM Sampling",
    category="sampling",
)

# ============================================================================
# MEMORY COMMANDS (0x70-0x7F)
# ============================================================================

KV_CACHE_LOAD_CMD = CGCInstruction(
    name="KV_CACHE_LOAD",
    opcode=0x70,
    params={
        "num_kv_heads": 8,
        "head_dim": 128,
        "dtype": "bf16",
    },
    description="Load KV cache blocks",
    module="vLLM PagedAttention",
    category="memory",
)

KV_CACHE_STORE_CMD = CGCInstruction(
    name="KV_CACHE_STORE",
    opcode=0x71,
    params={
        "num_kv_heads": 8,
        "head_dim": 128,
        "dtype": "bf16",
    },
    description="Store KV cache blocks",
    module="vLLM PagedAttention",
    category="memory",
)

KV_CACHE_UPDATE_CMD = CGCInstruction(
    name="KV_CACHE_UPDATE",
    opcode=0x72,
    params={
        "num_kv_heads": 8,
        "head_dim": 128,
    },
    description="Update KV cache with new tokens",
    module="vLLM PagedAttention",
    category="memory",
)

KV_CACHE_STATIC_LAYOUT_CMD = CGCInstruction(
    name="KV_CACHE_STATIC_LAYOUT",
    opcode=0x74,
    params={
        "num_kv_heads": 8,
        "head_dim": 128,
        "max_block_num": 1024,
    },
    description="Static layout KV cache for vLLM",
    module="vLLM PagedAttention",
    category="memory",
)

KV_CACHE_COMMIT_CMD = CGCInstruction(
    name="KV_CACHE_COMMIT",
    opcode=0x75,
    params={
        "num_kv_heads": 8,
    },
    description="Commit KV cache updates",
    module="vLLM PagedAttention",
    category="memory",
)

BATCH_COMPILE_CMD = CGCInstruction(
    name="BATCH_COMPILE",
    opcode=0x76,
    params={
        "max_batch_size": 32,
        "max_seq_len": 4096,
    },
    description="Batch compilation for dynamic shapes",
    module="vLLM Dynamic Batching",
    category="memory",
)

BATCH_MERGE_CMD = CGCInstruction(
    name="BATCH_MERGE",
    opcode=0x77,
    params={
        "max_merge_size": 4,
    },
    description="Merge batched sequences",
    module="vLLM Dynamic Batching",
    category="memory",
)

EMBEDDING_LOOKUP_CMD = CGCInstruction(
    name="EMBEDDING_LOOKUP",
    opcode=0x73,
    params={
        "pad_id": 0,
        "dtype": "fp16",
    },
    description="Token embedding lookup",
    module="vLLM Embedding",
    category="memory",
)

# ============================================================================
# KDA COMMANDS (0x80-0x8F) - Kimi KDA
# ============================================================================

WEIGHT_STAY_CMD = CGCInstruction(
    name="WEIGHT_STAY",
    opcode=0x01,
    params={
        "weight_partition": "backbone|expert|action_head",
        "lock_strategy": "page_lock",
        "gc_disable": True,
        "memory_priority": 1
    },
    description="Permanent weight residency in main memory",
    module="MegaTrain CPU Offload",
    category="memory",
)

LAYER_STREAM_LOAD_CMD = CGCInstruction(
    name="LAYER_STREAM_LOAD",
    opcode=0x02,
    params={
        "layer_id": 0,
        "block_size": 0,
        "preload": True,
        "async_copy": True
    },
    description="Single layer weight streaming load",
    module="MegaTrain Layer Streaming",
    category="memory",
)

LAYER_FORWARD_CMD = CGCInstruction(
    name="LAYER_FORWARD",
    opcode=0x03,
    params={
        "precision": "fp16",
        "fusion_ops": ["linear", "gelu", "layernorm"],
        "ortho_projection": True,
        "output_buffer": "cpu_temp"
    },
    description="Single layer forward with orthogonal basis projection",
    module="Kimi Orthogonal Basis",
    category="kda",
)

ORTHO_BASIS_UPDATE_CMD = CGCInstruction(
    name="ORTHO_BASIS_UPDATE",
    opcode=0x07,
    params={
        "algorithm": "gram_schmidt",
        "decay": 0.99,
        "fixed_size": 1024,
        "cache_on_chip": True
    },
    description="Kimi orthogonal basis update via Gram-Schmidt",
    module="Kimi Global Orthogonal Basis",
    category="kda",
)

KDA_CHUNK_CMD = CGCInstruction(
    name="KDA_CHUNK",
    opcode=0x80,
    params={
        "dtype": "bf16",
        "chunk_size": 64,
        "use_gate": True,
        "use_qk_l2norm": True,
        "use_beta_sigmoid": True,
        "safe_gate": True,
        "transpose_state_layout": True
    },
    description="FlashKDA chunk-based Kimi Delta Attention kernel",
    module="FlashKDA",
    category="kda",
)

KDA_PROJECT_CMD = CGCInstruction(
    name="KDA_PROJECT",
    opcode=0x81,
    params={
        "proj_dim": 128,
        "ortho_transform": True,
        "no_original_kv_storage": True
    },
    description="K projection for KDA orthogonal basis",
    module="Kimi KDA Projection",
    category="kda",
)

KDA_ORTHO_UPDATE_CMD = CGCInstruction(
    name="KDA_ORTHO_UPDATE",
    opcode=0x82,
    params={
        "gram_schmidt_iter": 1,
        "update_decay": 0.99,
        "fixed_basis_size": 1024,
        "increment_update": True
    },
    description="Incremental orthogonal basis update for KDA",
    module="Kimi Orthogonal Basis",
    category="kda",
)

# ============================================================================
# DISTRIBUTED COMMANDS (0x90-0x9F)
# ============================================================================

ALL_REDUCE_CMD = CGCInstruction(
    name="ALL_REDUCE",
    opcode=0x90,
    params={
        "op": "sum",
        "group_size": 8,
        "backend": "nccl",
    },
    description="AllReduce for tensor parallel",
    module="vLLM Distributed",
    category="distributed",
)

ALL_GATHER_CMD = CGCInstruction(
    name="ALL_GATHER",
    opcode=0x91,
    params={
        "group_size": 8,
        "backend": "nccl",
    },
    description="AllGather for tensor parallel output",
    module="vLLM Distributed",
    category="distributed",
)

REDUCE_SCATTER_CMD = CGCInstruction(
    name="REDUCE_SCATTER",
    opcode=0x92,
    params={
        "op": "sum",
        "group_size": 8,
    },
    description="Reduce Scatter for sequence parallel",
    module="vLLM Distributed",
    category="distributed",
)

# ============================================================================
# QUANTIZATION COMMANDS (0xA0-0xAF)
# ============================================================================

QUANTIZE_W8A16_CMD = CGCInstruction(
    name="QUANTIZE_W8A16",
    opcode=0xA0,
    params={
        "quant_type": "int8",
        "scale_type": "fp16",
    },
    description="W8A16 quantization (GPTQ-style)",
    module="vLLM Quantization",
    category="quantization",
)

QUANTIZE_W4A16_CMD = CGCInstruction(
    name="QUANTIZE_W4A16",
    opcode=0xA1,
    params={
        "quant_type": "int4",
        "scale_type": "fp16",
    },
    description="W4A16 quantization (AWQ-style)",
    module="vLLM Quantization",
    category="quantization",
)

DEQUANTIZE_CMD = CGCInstruction(
    name="DEQUANTIZE",
    opcode=0xA2,
    params={
        "orig_type": "int8",
        "target_type": "fp16",
    },
    description="Dequantize back to fp16/bf16",
    module="vLLM Quantization",
    category="quantization",
)

GPTQ_KERNEL_CMD = CGCInstruction(
    name="GPTQ_KERNEL",
    opcode=0xA3,
    params={
        "block_size": 128,
        "quant_type": "gptq",
    },
    description="GPTQ quantized matmul kernel",
    module="AutoGPTQ",
    category="quantization",
)

AWQ_KERNEL_CMD = CGCInstruction(
    name="AWQ_KERNEL",
    opcode=0xA4,
    params={
        "kernel_version": "v1",
    },
    description="AWQ quantized matmul kernel",
    module="AWQ",
    category="quantization",
)

FP8_E4M3_QUANT_CMD = CGCInstruction(
    name="FP8_E4M3_QUANT",
    opcode=0xA5,
    params={
        "scale_type": "per_tensor",
    },
    description="FP8 E4M3 quantization kernel",
    module="CGC",
    category="quantization",
)

FP8_E5M2_QUANT_CMD = CGCInstruction(
    name="FP8_E5M2_QUANT",
    opcode=0xA6,
    params={
        "scale_type": "per_tensor",
    },
    description="FP8 E5M2 quantization kernel",
    module="CGC",
    category="quantization",
)

FP8_DEQUANT_CMD = CGCInstruction(
    name="FP8_DEQUANT",
    opcode=0xA7,
    params={
        "dtype": "fp16",
    },
    description="FP8 dequantization kernel",
    module="CGC",
    category="quantization",
)


# ============================================================================
# LLAMA.CPP / GGUF COMMANDS (0xC0 ~ 0xDF)
# ============================================================================

LLAMA_GGUF_LOAD_CMD = CGCInstruction(
    name="LLAMA_GGUF_LOAD",
    opcode=0xC0,
    params={
        "model_path": "",
        "n_ctx": 2048,
        "n_batch": 512,
        "n_threads": 4,
        "use_mlock": False,
        "use_mmap": True,
    },
    description="Load GGUF formatted model file",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_GGUF_QUANTIZE_CMD = CGCInstruction(
    name="LLAMA_GGUF_QUANTIZE",
    opcode=0xC1,
    params={
        "quant_type": "Q4_K_M",
        "imatrix": None,
        "output_path": "",
    },
    description="Quantize model to GGUF format with specified type",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_GGUF_DEQUANTIZE_CMD = CGCInstruction(
    name="LLAMA_GGUF_DEQUANTIZE",
    opcode=0xC2,
    params={
        "output_type": "fp16",
    },
    description="Dequantize GGUF weights to higher precision",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_Q4_K_MATMUL_CMD = CGCInstruction(
    name="LLAMA_Q4_K_MATMUL",
    opcode=0xC3,
    params={
        "quant_type": "Q4_K",
        "alpha": 1.0,
        "beta": 0.0,
        "trans_a": False,
        "trans_b": True,
    },
    description="Q4_K quantized matrix multiplication",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_Q5_K_MATMUL_CMD = CGCInstruction(
    name="LLAMA_Q5_K_MATMUL",
    opcode=0xC4,
    params={
        "quant_type": "Q5_K",
        "alpha": 1.0,
        "beta": 0.0,
        "trans_a": False,
        "trans_b": True,
    },
    description="Q5_K quantized matrix multiplication",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_Q6_K_MATMUL_CMD = CGCInstruction(
    name="LLAMA_Q6_K_MATMUL",
    opcode=0xC5,
    params={
        "quant_type": "Q6_K",
        "alpha": 1.0,
        "beta": 0.0,
        "trans_a": False,
        "trans_b": True,
    },
    description="Q6_K quantized matrix multiplication",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_Q8_0_MATMUL_CMD = CGCInstruction(
    name="LLAMA_Q8_0_MATMUL",
    opcode=0xC6,
    params={
        "quant_type": "Q8_0",
        "alpha": 1.0,
        "beta": 0.0,
        "trans_a": False,
        "trans_b": True,
    },
    description="Q8_0 quantized matrix multiplication",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_Q2_K_MATMUL_CMD = CGCInstruction(
    name="LLAMA_Q2_K_MATMUL",
    opcode=0xC7,
    params={
        "quant_type": "Q2_K",
        "alpha": 1.0,
        "beta": 0.0,
        "trans_a": False,
        "trans_b": True,
    },
    description="Q2_K quantized matrix multiplication",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_Q3_K_MATMUL_CMD = CGCInstruction(
    name="LLAMA_Q3_K_MATMUL",
    opcode=0xC8,
    params={
        "quant_type": "Q3_K",
        "alpha": 1.0,
        "beta": 0.0,
        "trans_a": False,
        "trans_b": True,
    },
    description="Q3_K quantized matrix multiplication",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_Q8_K_MATMUL_CMD = CGCInstruction(
    name="LLAMA_Q8_K_MATMUL",
    opcode=0xC9,
    params={
        "quant_type": "Q8_K",
        "alpha": 1.0,
        "beta": 0.0,
        "trans_a": False,
        "trans_b": True,
    },
    description="Q8_K quantized matrix multiplication",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_MOE_ROUTING_CMD = CGCInstruction(
    name="LLAMA_MOE_ROUTING",
    opcode=0xCA,
    params={
        "n_experts": 8,
        "k_experts": 2,
        "score_func": "softmax",
    },
    description="MoE expert routing for GGUF MoE models",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_MOE_EXPERT_FWD_CMD = CGCInstruction(
    name="LLAMA_MOE_EXPERT_FWD",
    opcode=0xCB,
    params={
        "expert_id": 0,
        "quant_type": "Q4_K_M",
    },
    description="Single MoE expert forward pass",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_ROPE_GGUF_CMD = CGCInstruction(
    name="LLAMA_ROPE_GGUF",
    opcode=0xCC,
    params={
        "rope_freq_base": 10000.0,
        "rope_freq_scale": 1.0,
        "rope_scaling_type": None,
        "rope_scaling_factor": 1.0,
    },
    description="GGUF RoPE position encoding",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_RMSNORM_GGUF_CMD = CGCInstruction(
    name="LLAMA_RMSNORM_GGUF",
    opcode=0xCD,
    params={
        "eps": 1e-6,
    },
    description="GGUF RMS normalization",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_SILU_GGUF_CMD = CGCInstruction(
    name="LLAMA_SILU_GGUF",
    opcode=0xCE,
    params={
        "inplace": True,
    },
    description="GGUF SiLU activation",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_GELU_GGUF_CMD = CGCInstruction(
    name="LLAMA_GELU_GGUF",
    opcode=0xCF,
    params={
        "approximate": "none",
    },
    description="GGUF GELU activation",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_KV_CACHE_GGUF_CMD = CGCInstruction(
    name="LLAMA_KV_CACHE_GGUF",
    opcode=0xD0,
    params={
        "n_ctx": 2048,
        "n_batch": 512,
        "kv_seqs": [],
    },
    description="GGUF KV cache management",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_SAMPLING_GGUF_CMD = CGCInstruction(
    name="LLAMA_SAMPLING_GGUF",
    opcode=0xD1,
    params={
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
        "repeat_last_n": 64,
        "seed": -1,
    },
    description="GGUF token sampling",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_INFERENCE_CMD = CGCInstruction(
    name="LLAMA_INFERENCE",
    opcode=0xD2,
    params={
        "prompt": "",
        "max_tokens": 256,
        "temperature": 0.7,
        "stop": None,
        "stream": False,
    },
    description="Full GGUF inference with sampling",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_EMBEDDING_GGUF_CMD = CGCInstruction(
    name="LLAMA_EMBEDDING_GGUF",
    opcode=0xD3,
    params={
        "text": "",
        "pooling": "mean",
        "normalize": True,
    },
    description="GGUF text embedding extraction",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_DETOKENIZE_GGUF_CMD = CGCInstruction(
    name="LLAMA_DETOKENIZE_GGUF",
    opcode=0xD4,
    params={
        "tokens": [],
        "special_tokens": False,
    },
    description="GGUF token detokenization",
    module="llama.cpp",
    category="llama_cpp",
)

LLAMA_TOKENIZE_GGUF_CMD = CGCInstruction(
    name="LLAMA_TOKENIZE_GGUF",
    opcode=0xD5,
    params={
        "text": "",
        "add_bos": True,
        "special_tokens": False,
    },
    description="GGUF text tokenization",
    module="llama.cpp",
    category="llama_cpp",
)


# ============================================================================
# COMMAND SET REGISTRY
# ============================================================================

CGC_SIMD_COMMAND_SET: Dict[str, CGCInstruction] = {
    # Attention
    "ATTENTION_SDPA": ATTENTION_SDPA_CMD,
    "ATTENTION_KDA": ATTENTION_KDA_CMD,
    "ATTENTION_PAGED": ATTENTION_PAGED_CMD,
    "ATTENTION_FLASH": ATTENTION_FLASH_CMD,

    # Linear/GEMM
    "LINEAR_GEMM": LINEAR_GEMM_CMD,
    "LINEAR_BIAS": LINEAR_BIAS_CMD,
    "GEMM_BATCHED": GEMM_BATCHED_CMD,

    # Norm
    "LAYER_NORM": LAYER_NORM_CMD,
    "RMS_NORM": RMS_NORM_CMD,
    "GROUP_NORM": GROUP_NORM_CMD,

    # RoPE
    "ROPE": ROPE_CMD,
    "ROPE_FUSED": ROPE_FUSED_CMD,
    "YARN_ROPE": YARN_ROPE_CMD,

    # Activation
    "SILU": SILU_CMD,
    "GELU": GELU_CMD,
    "GELU_TANH": GELU_TANH_CMD,
    "RELU": RELU_CMD,
    "SIGMOID": SIGMOID_CMD,

    # Sampling
    "SOFTMAX": SOFTMAX_CMD,
    "LOG_SOFTMAX": LOG_SOFTMAX_CMD,
    "TOP_K": TOP_K_CMD,
    "TOP_P": TOP_P_CMD,
    "TEMPERATURE": TEMPERATURE_CMD,

    # Memory
    "KV_CACHE_LOAD": KV_CACHE_LOAD_CMD,
    "KV_CACHE_STORE": KV_CACHE_STORE_CMD,
    "KV_CACHE_UPDATE": KV_CACHE_UPDATE_CMD,
    "KV_CACHE_STATIC_LAYOUT": KV_CACHE_STATIC_LAYOUT_CMD,
    "KV_CACHE_COMMIT": KV_CACHE_COMMIT_CMD,
    "BATCH_COMPILE": BATCH_COMPILE_CMD,
    "BATCH_MERGE": BATCH_MERGE_CMD,
    "EMBEDDING_LOOKUP": EMBEDDING_LOOKUP_CMD,

    # KDA
    "WEIGHT_STAY": WEIGHT_STAY_CMD,
    "LAYER_STREAM_LOAD": LAYER_STREAM_LOAD_CMD,
    "LAYER_FORWARD": LAYER_FORWARD_CMD,
    "ORTHO_BASIS_UPDATE": ORTHO_BASIS_UPDATE_CMD,
    "KDA_CHUNK": KDA_CHUNK_CMD,
    "KDA_PROJECT": KDA_PROJECT_CMD,
    "KDA_ORTHO_UPDATE": KDA_ORTHO_UPDATE_CMD,

    # Distributed
    "ALL_REDUCE": ALL_REDUCE_CMD,
    "ALL_GATHER": ALL_GATHER_CMD,
    "REDUCE_SCATTER": REDUCE_SCATTER_CMD,

    # Quantization
    "QUANTIZE_W8A16": QUANTIZE_W8A16_CMD,
    "QUANTIZE_W4A16": QUANTIZE_W4A16_CMD,
    "DEQUANTIZE": DEQUANTIZE_CMD,
    "GPTQ_KERNEL": GPTQ_KERNEL_CMD,
    "AWQ_KERNEL": AWQ_KERNEL_CMD,
    "FP8_E4M3_QUANT": FP8_E4M3_QUANT_CMD,
    "FP8_E5M2_QUANT": FP8_E5M2_QUANT_CMD,
    "FP8_DEQUANT": FP8_DEQUANT_CMD,

    # Fine-tuning / LoRA / QLoRA
    "LORA_A_MATMUL": CGCInstruction(
        name="LORA_A_MATMUL",
        opcode=0xB0,
        category="finetuning",
        description="LoRA A matrix multiplication",
        params={"rank": 0, "alpha": 0},
        module="mlx_tune",
    ),
    "LORA_B_MATMUL": CGCInstruction(
        name="LORA_B_MATMUL",
        opcode=0xB1,
        category="finetuning",
        description="LoRA B matrix multiplication",
        params={"rank": 0, "alpha": 0},
        module="mlx_tune",
    ),
    "LORA_MERGE": CGCInstruction(
        name="LORA_MERGE",
        opcode=0xB2,
        category="finetuning",
        description="Merge LoRA weights into base",
        params={"alpha": 0},
        module="mlx_tune",
    ),
    "LORA_APPLY": CGCInstruction(
        name="LORA_APPLY",
        opcode=0xB3,
        category="finetuning",
        description="Apply LoRA to input",
        params={"scale": 1.0},
        module="mlx_tune",
    ),
    "QLORA_DEQUANT": CGCInstruction(
        name="QLORA_DEQUANT",
        opcode=0xB4,
        category="finetuning",
        description="QLoRA dequantization",
        params={"bits": 4, "quant_type": "fp4"},
        module="mlx_tune",
    ),
    "MLX_TUNE_FORWARD": CGCInstruction(
        name="MLX_TUNE_FORWARD",
        opcode=0xB8,
        category="finetuning",
        description="MLX forward pass",
        params={"lora_a": None, "lora_b": None},
        module="mlx_tune",
    ),
    "KDA_LORA_FUSE": CGCInstruction(
        name="KDA_LORA_FUSE",
        opcode=0xBA,
        category="finetuning",
        description="FlashKDA + LoRA fusion",
        params={"lora_a": None, "lora_b": None, "scale": 1.0},
        module="mlx_tune",
    ),

    # LLAMA.CPP / GGUF
    "LLAMA_GGUF_LOAD": LLAMA_GGUF_LOAD_CMD,
    "LLAMA_GGUF_QUANTIZE": LLAMA_GGUF_QUANTIZE_CMD,
    "LLAMA_GGUF_DEQUANTIZE": LLAMA_GGUF_DEQUANTIZE_CMD,
    "LLAMA_Q4_K_MATMUL": LLAMA_Q4_K_MATMUL_CMD,
    "LLAMA_Q5_K_MATMUL": LLAMA_Q5_K_MATMUL_CMD,
    "LLAMA_Q6_K_MATMUL": LLAMA_Q6_K_MATMUL_CMD,
    "LLAMA_Q8_0_MATMUL": LLAMA_Q8_0_MATMUL_CMD,
    "LLAMA_Q2_K_MATMUL": LLAMA_Q2_K_MATMUL_CMD,
    "LLAMA_Q3_K_MATMUL": LLAMA_Q3_K_MATMUL_CMD,
    "LLAMA_Q8_K_MATMUL": LLAMA_Q8_K_MATMUL_CMD,
    "LLAMA_MOE_ROUTING": LLAMA_MOE_ROUTING_CMD,
    "LLAMA_MOE_EXPERT_FWD": LLAMA_MOE_EXPERT_FWD_CMD,
    "LLAMA_ROPE_GGUF": LLAMA_ROPE_GGUF_CMD,
    "LLAMA_RMSNORM_GGUF": LLAMA_RMSNORM_GGUF_CMD,
    "LLAMA_SILU_GGUF": LLAMA_SILU_GGUF_CMD,
    "LLAMA_GELU_GGUF": LLAMA_GELU_GGUF_CMD,
    "LLAMA_KV_CACHE_GGUF": LLAMA_KV_CACHE_GGUF_CMD,
    "LLAMA_SAMPLING_GGUF": LLAMA_SAMPLING_GGUF_CMD,
    "LLAMA_INFERENCE": LLAMA_INFERENCE_CMD,
    "LLAMA_EMBEDDING_GGUF": LLAMA_EMBEDDING_GGUF_CMD,
    "LLAMA_DETOKENIZE_GGUF": LLAMA_DETOKENIZE_GGUF_CMD,
    "LLAMA_TOKENIZE_GGUF": LLAMA_TOKENIZE_GGUF_CMD,
}


def get_cgc_command(name: str) -> Optional[CGCInstruction]:
    """Get CGC command by name"""
    return CGC_SIMD_COMMAND_SET.get(name)


def get_cgc_command_by_opcode(opcode: int) -> Optional[CGCInstruction]:
    """Get CGC command by opcode"""
    for cmd in CGC_SIMD_COMMAND_SET.values():
        if cmd.opcode == opcode:
            return cmd
    return None


def get_all_cgc_commands() -> List[CGCInstruction]:
    """Get all CGC commands"""
    return list(CGC_SIMD_COMMAND_SET.values())


def get_commands_by_category(category: str) -> List[CGCInstruction]:
    """Get commands by category"""
    return [cmd for cmd in CGC_SIMD_COMMAND_SET.values() if cmd.category == category]


def get_commands_by_module(module: str) -> List[CGCInstruction]:
    """Get commands by module"""
    return [cmd for cmd in CGC_SIMD_COMMAND_SET.values() if cmd.module == module]


def create_kda_instruction(
    q: str,
    k: str,
    v: str,
    g: Optional[str] = None,
    beta: Optional[str] = None,
    scale: float = 1.0,
    **kwargs
) -> KDAInstruction:
    """Create a KDA instruction"""
    return KDAInstruction(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        **kwargs
    )


# ============================================================
# GDS / PD 资源指令 (0x90-0x9F) - 零拷贝 GPU ↔ SSD
# ============================================================
GDS_LOAD_KV_CMD = CGCInstruction(
    name="GDS_LOAD_KV",
    opcode=0x90,
    params={"key": "", "seq_len": 2048, "head_dim": 128, "device": "cuda"},
    description="GDS 零拷贝加载 KV Cache",
    module="gds_service",
    category="gds",
)

GDS_STORE_KV_CMD = CGCInstruction(
    name="GDS_STORE_KV",
    opcode=0x91,
    params={"key": ""},
    description="GDS 零拷贝存储 KV Cache",
    module="gds_service",
    category="gds",
)

GDS_LOAD_WEIGHT_CMD = CGCInstruction(
    name="GDS_LOAD_WEIGHT",
    opcode=0x92,
    params={"path": "", "shape": [], "dtype": "float16"},
    description="GDS 零拷贝加载权重",
    module="gds_service",
    category="gds",
)

# ============================================================
# FlashMoE / oMLX 端侧 MoE 指令 (0xD0-0xDF)
# ============================================================
FLASH_MOE_LOAD_EXPERTS_CMD = CGCInstruction(
    name="FLASH_MOE_LOAD_EXPERTS",
    opcode=0xD0,
    params={"expert_ids": [], "expert_dir": "/tmp/flash_moe_experts"},
    description="FlashMoE 按需加载专家权重",
    module="flash_moe",
    category="flash_moe",
)

FLASH_MOE_RUN_MLP_CMD = CGCInstruction(
    name="FLASH_MOE_RUN_MLP",
    opcode=0xD1,
    params={"expert_ids": [0], "top_k": 2},
    description="FlashMoE 专家 MLP 执行",
    module="flash_moe",
    category="flash_moe",
)

OMLX_FLASH_PREDICT_CMD = CGCInstruction(
    name="OMLX_FLASH_PREDICT",
    opcode=0xD2,
    params={"top_k": 2},
    description="oMLX 专家激活预测",
    module="omlx",
    category="flash_moe",
)

OMLX_FLASH_CACHE_CMD = CGCInstruction(
    name="OMLX_FLASH_CACHE",
    opcode=0xD3,
    params={"evict_policy": "lru"},
    description="oMLX 专家缓存管理",
    module="omlx",
    category="flash_moe",
)

# ============================================================
# JITLoad 即时编译指令 (0xE0-0xEF)
# ============================================================
JIT_LOAD_COMPILED_CMD = CGCInstruction(
    name="JIT_LOAD_COMPILED",
    opcode=0xE0,
    params={"kernel_path": ""},
    description="JIT 加载已编译的 kernel",
    module="cgc_jitload",
    category="jit",
)

JIT_COMPILE_KERNEL_CMD = CGCInstruction(
    name="JIT_COMPILE_KERNEL",
    opcode=0xE1,
    params={"kernel_type": "attention"},
    description="JIT 编译 CUDA/Metal kernel",
    module="cgc_jitload",
    category="jit",
)

JIT_DISPATCH_CMD = CGCInstruction(
    name="JIT_DISPATCH",
    opcode=0xE2,
    params={"auto_select": True},
    description="JIT 自动调度最优 kernel",
    module="cgc_jitload",
    category="jit",
)

# ============================================================
# SPDK 用户态 NVMe 存储指令 (0xF0-0xFF)
# ============================================================
SPDK_READ_CMD = CGCInstruction(
    name="SPDK_READ",
    opcode=0xF0,
    params={"path": "", "offset": 0, "size": 0},
    description="SPDK 零拷贝 NVMe 读取",
    module="spdk_adapter",
    category="spdk",
)

SPDK_WRITE_CMD = CGCInstruction(
    name="SPDK_WRITE",
    opcode=0xF1,
    params={"path": "", "offset": 0},
    description="SPDK 零拷贝 NVMe 写入",
    module="spdk_adapter",
    category="spdk",
)

SPDK_ALLOC_BUF_CMD = CGCInstruction(
    name="SPDK_ALLOC_BUF",
    opcode=0xF2,
    params={"size": 0},
    description="SPDK 显存缓冲分配",
    module="spdk_adapter",
    category="spdk",
)

# ============================================================
# 更新命令注册表
# ============================================================
CGC_SIMD_COMMAND_SET["GDS_LOAD_KV"] = GDS_LOAD_KV_CMD
CGC_SIMD_COMMAND_SET["GDS_STORE_KV"] = GDS_STORE_KV_CMD
CGC_SIMD_COMMAND_SET["GDS_LOAD_WEIGHT"] = GDS_LOAD_WEIGHT_CMD
CGC_SIMD_COMMAND_SET["FLASH_MOE_LOAD_EXPERTS"] = FLASH_MOE_LOAD_EXPERTS_CMD
CGC_SIMD_COMMAND_SET["FLASH_MOE_RUN_MLP"] = FLASH_MOE_RUN_MLP_CMD
CGC_SIMD_COMMAND_SET["OMLX_FLASH_PREDICT"] = OMLX_FLASH_PREDICT_CMD
CGC_SIMD_COMMAND_SET["OMLX_FLASH_CACHE"] = OMLX_FLASH_CACHE_CMD
CGC_SIMD_COMMAND_SET["JIT_LOAD_COMPILED"] = JIT_LOAD_COMPILED_CMD
CGC_SIMD_COMMAND_SET["JIT_COMPILE_KERNEL"] = JIT_COMPILE_KERNEL_CMD
CGC_SIMD_COMMAND_SET["JIT_DISPATCH"] = JIT_DISPATCH_CMD
CGC_SIMD_COMMAND_SET["SPDK_READ"] = SPDK_READ_CMD
CGC_SIMD_COMMAND_SET["SPDK_WRITE"] = SPDK_WRITE_CMD
CGC_SIMD_COMMAND_SET["SPDK_ALLOC_BUF"] = SPDK_ALLOC_BUF_CMD


def get_command_summary() -> Dict[str, Any]:
    """Get summary of all commands"""
    categories = {}
    for cmd in CGC_SIMD_COMMAND_SET.values():
        if cmd.category not in categories:
            categories[cmd.category] = []
        categories[cmd.category].append({"name": cmd.name, "opcode": f"0x{cmd.opcode:02X}"})

    return {
        "total_commands": len(CGC_SIMD_COMMAND_SET),
        "categories": categories,
    }
