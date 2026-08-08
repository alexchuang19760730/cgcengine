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
CGC Opcodes - vLLM Integration Enum

This module provides the unified opcode enum for vLLM integration.
All 40 CGC commands are mapped to hex opcodes for vLLM backend registration.
"""

from enum import IntEnum


class CGC_OP_CODES(IntEnum):
    """
    CGC SIMD Command Opcodes for vLLM Integration

    Usage:
        from cgc_engine.cgc.cgc_opcodes import CGC_OP_CODES

        # In vLLM KDA forward:
        if opcode == CGC_OP_CODES.KDA_CHUNK:
            kernel = flash_kda.fwd
    """

    # =========================================================================
    # ATTENTION (0x10-0x1F)
    # =========================================================================
    ATTENTION_SDPA = 0x10
    ATTENTION_KDA = 0x11
    ATTENTION_PAGED = 0x12
    ATTENTION_FLASH = 0x13

    # =========================================================================
    # LINEAR/GEMM (0x20-0x2F)
    # =========================================================================
    LINEAR_GEMM = 0x20
    LINEAR_BIAS = 0x21
    GEMM_BATCHED = 0x22

    # =========================================================================
    # NORM (0x30-0x3F)
    # =========================================================================
    LAYER_NORM = 0x30
    RMS_NORM = 0x31
    GROUP_NORM = 0x32

    # =========================================================================
    # ROPE (0x40-0x4F)
    # =========================================================================
    ROPE = 0x40
    ROPE_FUSED = 0x41
    YARN_ROPE = 0x42

    # =========================================================================
    # ACTIVATION (0x50-0x5F)
    # =========================================================================
    SILU = 0x50
    GELU = 0x51
    GELU_TANH = 0x52
    RELU = 0x53
    SIGMOID = 0x54

    # =========================================================================
    # SAMPLING (0x60-0x6F)
    # =========================================================================
    SOFTMAX = 0x60
    LOG_SOFTMAX = 0x61
    TOP_K = 0x62
    TOP_P = 0x63
    TEMPERATURE = 0x64

    # =========================================================================
    # MEMORY (0x70-0x7F)
    # =========================================================================
    KV_CACHE_LOAD = 0x70
    KV_CACHE_STORE = 0x71
    KV_CACHE_UPDATE = 0x72
    EMBEDDING_LOOKUP = 0x73
    KV_CACHE_STATIC_LAYOUT = 0x74
    KV_CACHE_COMMIT = 0x75
    BATCH_COMPILE = 0x76
    BATCH_MERGE = 0x77

    # =========================================================================
    # KDA (0x80-0x8F) - Kimi KDA
    # =========================================================================
    KDA_CHUNK = 0x80
    KDA_PROJECT = 0x81
    KDA_ORTHO_UPDATE = 0x82
    KDA_BACKWARD = 0x83

    # KDA Aliases for compatibility
    KDA_FORWARD = 0x80
    KDA_ATTENTION = 0x11

    # =========================================================================
    # DISTRIBUTED (0x90-0x93) - NCCL 集合通信
    # =========================================================================
    ALL_REDUCE = 0x90
    ALL_GATHER = 0x91
    REDUCE_SCATTER = 0x92

    # =========================================================================
    # GDS / PD Storage (0x94-0x9F) - GPUDirect Storage / Prefetch Distribution
    # =========================================================================
    GDS_LOAD_KV = 0x94
    GDS_SAVE_KV = 0x95
    GDS_LOAD_WEIGHT = 0x96
    GDS_SAVE_WEIGHT = 0x97
    PD_LOAD_KV = 0x98
    PD_SAVE_KV = 0x99
    PD_LOAD_WEIGHT = 0x9A
    PD_SAVE_WEIGHT = 0x9B

    # =========================================================================
    # QUANTIZATION (0xA0-0xAF)
    # =========================================================================
    QUANTIZE_W8A16 = 0xA0
    QUANTIZE_W4A16 = 0xA1
    DEQUANTIZE = 0xA2
    GPTQ_KERNEL = 0xA3
    AWQ_KERNEL = 0xA4
    FP8_E4M3_QUANT = 0xA5
    FP8_E5M2_QUANT = 0xA6
    FP8_DEQUANT = 0xA7

    # =========================================================================
    # FINE-TUNING / LoRA / QLoRA / MLX-Tune (0xB0 ~ 0xBF)
    # 16 条全新指令，不冲突原有架构
    # =========================================================================
    LORA_A_MATMUL = 0xB0
    LORA_B_MATMUL = 0xB1
    LORA_MERGE = 0xB2
    QLORA_DEQUANT = 0xB3
    LORA_SCATTER = 0xB4
    LORA_GRAD = 0xB5
    MLX_TUNE_FWD = 0xB6
    MLX_TUNE_BWD = 0xB7
    KDA_LORA_FUSE = 0xB8
    MLX_ROPE_FUSE = 0xB9
    MLX_GELU_FUSE = 0xBA
    MLX_RMS_NORM = 0xBB
    MLX_SAMPLING_TOPK = 0xBC
    MLX_KV_CACHE = 0xBD
    MLX_QUANTIZE = 0xBE
    MLX_DEQUANTIZE = 0xBF
    MLX_QGEMM = 0xD3

    # =========================================================================
    # LEGACY CGC COMMANDS
    # =========================================================================
    WEIGHT_STAY = 0x01
    LAYER_STREAM_LOAD = 0x02
    LAYER_FORWARD = 0x03
    ORTHO_BASIS_UPDATE = 0x07

    # =========================================================================
    # LLAMA.CPP (0xC0 ~ 0xDF) - GGUF Quantization & CPU/GPU Inference
    # =========================================================================
    LLAMA_GGUF_LOAD = 0xC0
    LLAMA_GGUF_QUANTIZE = 0xC1
    LLAMA_GGUF_DEQUANTIZE = 0xC2
    LLAMA_Q4_K_MATMUL = 0xC3
    LLAMA_Q5_K_MATMUL = 0xC4
    LLAMA_Q6_K_MATMUL = 0xC5
    LLAMA_Q8_0_MATMUL = 0xC6
    LLAMA_Q2_K_MATMUL = 0xC7
    LLAMA_Q3_K_MATMUL = 0xC8
    LLAMA_Q8_K_MATMUL = 0xC9
    LLAMA_MOE_ROUTING = 0xCA
    LLAMA_MOE_EXPERT_FWD = 0xCB
    LLAMA_ROPE_GGUF = 0xCC
    LLAMA_RMSNORM_GGUF = 0xCD
    LLAMA_SILU_GGUF = 0xCE
    LLAMA_GELU_GGUF = 0xCF
    LLAMA_KV_CACHE_GGUF = 0xD0
    LLAMA_SAMPLING_GGUF = 0xD1
    LLAMA_INFERENCE = 0xD2
    LLAMA_EMBEDDING_GGUF = 0xD3
    LLAMA_DETOKENIZE_GGUF = 0xD4
    LLAMA_TOKENIZE_GGUF = 0xD5

    # =========================================================================
    # FLASHMOE / OMLX (0xE0-0xEF) - 端側 MoE 引擎
    # =========================================================================
    FLASHMOE_LOAD_EXPERT = 0xE0
    FLASHMOE_MLP_FORWARD = 0xE1
    FLASHMOE_EXPERT_FWD = 0xE2
    OMLX_PREDICT_EXPERTS = 0xE3
    OMLX_CACHE_UPDATE = 0xE4
    OMLX_EVICT = 0xE5

    # =========================================================================
    # JITLOAD (0xF0-0xFF) - 即時編譯加載系統
    # =========================================================================
    JIT_LOAD_COMPILED = 0xF0
    JIT_COMPILE_KERNEL = 0xF1
    JIT_AUTO_DISPATCH = 0xF2
    JIT_CACHE_LOOKUP = 0xF3
    JIT_CACHE_INVALIDATE = 0xF4


OPCODE_TO_NAME = {
    opcode.value: opcode.name for opcode in CGC_OP_CODES
}

NAME_TO_OPCODE = {
    opcode.name: opcode.value for opcode in CGC_OP_CODES
}


def get_opcode_name(opcode: int) -> str:
    """Get opcode name from value"""
    return OPCODE_TO_NAME.get(opcode, f"UNKNOWN_0x{opcode:02X}")


def get_opcode_value(name: str) -> int:
    """Get opcode value from name"""
    return NAME_TO_OPCODE.get(name, -1)


def list_attention_opcodes() -> list:
    """List all attention-related opcodes"""
    return [
        CGC_OP_CODES.ATTENTION_SDPA,
        CGC_OP_CODES.ATTENTION_KDA,
        CGC_OP_CODES.ATTENTION_PAGED,
        CGC_OP_CODES.ATTENTION_FLASH,
        CGC_OP_CODES.KDA_CHUNK,
        CGC_OP_CODES.KDA_ATTENTION,
    ]


def list_kda_opcodes() -> list:
    """List all KDA-related opcodes"""
    return [
        CGC_OP_CODES.KDA_CHUNK,
        CGC_OP_CODES.KDA_PROJECT,
        CGC_OP_CODES.KDA_ORTHO_UPDATE,
        CGC_OP_CODES.KDA_FORWARD,
        CGC_OP_CODES.ORTHO_BASIS_UPDATE,
    ]


def is_attention_opcode(opcode: int) -> bool:
    """Check if opcode is attention-related"""
    return opcode in [
        0x10, 0x11, 0x12, 0x13, 0x80
    ]


def is_kda_opcode(opcode: int) -> bool:
    """Check if opcode is KDA-related"""
    return opcode in [0x80, 0x81, 0x82, 0x83, 0x07]


# CGC Opcode Categories for vLLM
CGC_CATEGORIES = {
    "attention": [0x10, 0x11, 0x12, 0x13, 0x80],
    "linear": [0x20, 0x21, 0x22],
    "norm": [0x30, 0x31, 0x32],
    "rope": [0x40, 0x41, 0x42],
    "activation": [0x50, 0x51, 0x52, 0x53, 0x54],
    "sampling": [0x60, 0x61, 0x62, 0x63, 0x64],
    "memory": [0x70, 0x71, 0x72, 0x73, 0x74, 0x75],
    "batch": [0x76, 0x77],
    "kda": [0x80, 0x81, 0x82, 0x83, 0x07],
    "distributed": [0x90, 0x91, 0x92],
    "gds_pd": [0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0x9B],
    "quantization": [0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7],
    "finetuning": [0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xBB, 0xBC, 0xBD, 0xBE, 0xBF],
    "llama_cpp": [0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xCB, 0xCC, 0xCD, 0xCE, 0xCF, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5],
    "flashmoe_omlx": [0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5],
    "jitload": [0xF0, 0xF1, 0xF2, 0xF3, 0xF4],
    "spdk": [0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF],
}


def is_finetuning_opcode(opcode: int) -> bool:
    """Check if opcode is fine-tuning related"""
    return opcode in range(0xB0, 0xBC)


def is_lora_opcode(opcode: int) -> bool:
    """Check if opcode is LoRA related"""
    return opcode in [0xB0, 0xB1, 0xB2, 0xB3, 0xB5, 0xB6]


def is_llama_cpp_opcode(opcode: int) -> bool:
    """Check if opcode is llama.cpp related"""
    return opcode in range(0xC0, 0xD6)


def get_llama_cpp_quant_type(opcode: int) -> str:
    """Get GGUF quantization type from opcode"""
    quant_map = {
        0xC3: "Q4_K",
        0xC4: "Q5_K",
        0xC5: "Q6_K",
        0xC6: "Q8_0",
        0xC7: "Q2_K",
        0xC8: "Q3_K",
        0xC9: "Q8_K",
    }
    return quant_map.get(opcode, "UNKNOWN")


def get_category(opcode: int) -> str:
    """Get category name for an opcode"""
    for category, opcodes in CGC_CATEGORIES.items():
        if opcode in opcodes:
            return category
    return "unknown"


def list_llama_cpp_opcodes() -> list:
    """List all llama.cpp related opcodes"""
    return [
        CGC_OP_CODES.LLAMA_GGUF_LOAD,
        CGC_OP_CODES.LLAMA_GGUF_QUANTIZE,
        CGC_OP_CODES.LLAMA_GGUF_DEQUANTIZE,
        CGC_OP_CODES.LLAMA_Q4_K_MATMUL,
        CGC_OP_CODES.LLAMA_Q5_K_MATMUL,
        CGC_OP_CODES.LLAMA_Q6_K_MATMUL,
        CGC_OP_CODES.LLAMA_Q8_0_MATMUL,
        CGC_OP_CODES.LLAMA_Q2_K_MATMUL,
        CGC_OP_CODES.LLAMA_Q3_K_MATMUL,
        CGC_OP_CODES.LLAMA_Q8_K_MATMUL,
        CGC_OP_CODES.LLAMA_MOE_ROUTING,
        CGC_OP_CODES.LLAMA_MOE_EXPERT_FWD,
        CGC_OP_CODES.LLAMA_ROPE_GGUF,
        CGC_OP_CODES.LLAMA_RMSNORM_GGUF,
        CGC_OP_CODES.LLAMA_SILU_GGUF,
        CGC_OP_CODES.LLAMA_GELU_GGUF,
        CGC_OP_CODES.LLAMA_KV_CACHE_GGUF,
        CGC_OP_CODES.LLAMA_SAMPLING_GGUF,
        CGC_OP_CODES.LLAMA_INFERENCE,
        CGC_OP_CODES.LLAMA_EMBEDDING_GGUF,
        CGC_OP_CODES.LLAMA_DETOKENIZE_GGUF,
        CGC_OP_CODES.LLAMA_TOKENIZE_GGUF,
    ]
