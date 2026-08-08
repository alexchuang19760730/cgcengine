# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
优化候选空间生成器 - OptimizationSpaceBuilder

功能：
- 分析模型结构，生成可供 Harness Agent 选择的优化候选空间
- 支持自动检测硬件约束（CPU/CUDA/Metal
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """硬件设备信息"""
    device_type: str  # cpu/cuda/metal/mps
    compute_capability: Optional[Tuple[int, int]] = None
    total_memory: Optional[int] = None
    available_memory: Optional[int] = None
    sm_count: Optional[int] = None
    has_tensor_cores: bool = False
    has_fp8_support: bool = False


@dataclass
class OptimizationSpace:
    """优化候选空间 - Harness Agent 决策空间"""
    model_type: str = "unknown"
    input_shape: Tuple[int, ...] = field(default_factory=tuple)
    device_info: DeviceInfo = field(default_factory=lambda: DeviceInfo(device_type="cpu"))

    # 可融合的算子候选（分组
    fusible_ops: List[List[str]] = field(default_factory=list)
    # Tiling 候选大小
    tile_sizes: Dict[str, List[int]] = field(default_factory=dict)
    # 内存布局选项
    memory_layout_options: Dict[str, List[str]] = field(default_factory=dict)
    # 调度选项
    schedule_options: Dict[str, List[Any]] = field(default_factory=dict)
    # 后端选项
    backend_options: List[str] = field(default_factory=lambda: ["auto", "cpu", "cuda", "metal"])

    metadata: Dict[str, Any] = field(default_factory=dict)


class OptimizationSpaceBuilder:
    """优化候选空间生成器"""

    FUSIBLE_CANDIDATES = [
        ["linear", "silu", "norm"],
        ["linear", "gelu", "linear"],
        ["q_proj", "k_proj", "v_proj", "rope"],
        ["all_gather", "linear"],
    ]

    TILE_SIZE_CANDIDATES = {
        "M": [64, 128, 256, 512],
        "N": [64, 128, 256, 512],
        "K": [64, 128, 256, 512],
        "ATTN_BLOCK": [64, 128, 256],
    }

    MEMORY_LAYOUT_OPTIONS = {
        "qkv": ["L1", "L2", "GPU"],
        "act": ["L1", "L2", "GPU"],
        "weight": ["L2", "GPU", "SSD"],
    }

    SCHEDULE_OPTIONS = {
        "prefetch": [1, 2, 3, 4],
        "unroll": [1, 2, 4, 8],
        "pipeline": [1, 2, 3, 4],
        "tp_overlap": [True, False],
    }

    @classmethod
    def build(
        cls,
        model: Optional[nn.Module],
        input_shape: Tuple[int, ...],
        device: Optional[str] = None,
        device_info: Optional[DeviceInfo] = None,
    ) -> OptimizationSpace:
        """
        构建优化候选空间

        Args:
            model: PyTorch 模型
            input_shape: 输入张量形状
            device: 设备类型
            device_info: 设备信息（可选）

        Returns:
            OptimizationSpace
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "metal"
            else:
                device = "cpu"

        if device_info is None:
            device_info = cls._detect_device_info(device)

        model_type = cls._detect_model_type(model)

        space = OptimizationSpace(
            model_type=model_type,
            input_shape=input_shape,
            device_info=device_info,
            fusible_ops=cls.FUSIBLE_CANDIDATES,
            tile_sizes=cls.TILE_SIZE_CANDIDATES.copy(),
            memory_layout_options=cls.MEMORY_LAYOUT_OPTIONS.copy(),
            schedule_options=cls.SCHEDULE_OPTIONS.copy(),
        )

        # 根据模型结构调整空间
        if "moe" in model_type.lower():
            space.fusible_ops.append(["all_to_expert", "silu", "expert_to_all"])
            space.tile_sizes["MOE_BLOCK"] = [64, 128]

        if "vision" in model_type.lower() or "vlm" in model_type.lower():
            space.fusible_ops.append(["conv", "gelu", "norm"])

        logger.info(
            f"[OptimizationSpaceBuilder] Built space: model={model_type}, "
            f"device={device}, input={input_shape}"
        )

        return space

    @classmethod
    def _detect_device_info(cls, device: str) -> DeviceInfo:
        """检测硬件设备信息"""
        info = DeviceInfo(device_type=device)

        if device == "cuda" and torch.cuda.is_available():
            try:
                prop = torch.cuda.get_device_properties(0)
                info.compute_capability = (prop.major, prop.minor)
                info.total_memory = prop.total_memory
                info.available_memory = torch.cuda.mem_get_info()[0]
                info.sm_count = prop.multi_processor_count
                info.has_tensor_cores = prop.major >= 7
                info.has_fp8_support = prop.major >= 8
            except Exception as e:
                logger.warning(f"Failed to get CUDA device info: {e}")

        elif device in ["metal", "mps"] and torch.backends.mps.is_available():
            try:
                info.total_memory = torch.backends.mps.recommended_max_memory()
            except (AttributeError, TypeError):
                info.total_memory = 0

        return info

    @staticmethod
    def _detect_model_type(model: Optional[nn.Module]) -> str:
        """检测模型类型"""
        if model is None:
            return "unknown"
        model_str = str(type(model)).lower()

        if "llama" in model_str:
            return "llama"
        elif "mistral" in model_str:
            return "mistral"
        elif "qwen" in model_str:
            return "qwen"
        elif "phi" in model_str:
            return "phi"
        elif "moe" in model_str:
            return "moe"
        elif "clip" in model_str or "vision" in model_str:
            return "vlm"

        # 检查内部结构
        for name, _ in model.named_modules():
            if "moe" in name.lower():
                return "moe"
            if "vision" in name.lower():
                return "vlm"

        return "unknown"
