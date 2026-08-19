# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
Agent 操作提示枚举

定义 Agent 给 CGC SIMD 引擎的操作提示类型
"""

from enum import Enum


class AgentOpHint(Enum):
    """Agent 给 CGC SIMD 引擎的操作提示"""
    AUTO = "auto"
    FLASH_ATTENTION = "flash_attention"
    FUSE_MLP = "fuse_mlp"
    FUSE_ATTN = "fuse_attn"
    KDA_COMPRESS = "kda_compress"
    MOE_ROUTING = "moe_routing"
    TENSOR_PARALLEL = "tensor_parallel"
    VLM_CROSS_ATTENTION = "vlm_cross_attention"
