# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
优化目标注册表 - 每个计算算子明确定义优化目标！
🔥 Harness Agent基于这个注册表来决策每个算子的优化策略
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import enum


class OptimizationTargetType(enum.Enum):
    """优化目标类型枚举"""
    TRANSFORM = "transform"           # 算子变换/替换（如：KDA替换标准Attention）
    FUSION = "fusion"                 # 算子融合
    TILING = "tiling"                 # 分块优化
    MEMORY_LAYOUT = "memory_layout"   # 内存布局优化
    SCHEDULING = "scheduling"         # 调度优化
    QUANTIZATION = "quantization"     # 量化优化
    PIPELINE = "pipeline"             # 流水线并行
    CACHING = "caching"               # 缓存策略


@dataclass
class OptimizationTarget:
    """单个优化目标定义"""
    name: str
    target_type: OptimizationTargetType
    description: str
    priority: int = 100
    enabled: bool = True
    hardware_requirements: List[str] = field(default_factory=lambda: ["cuda", "metal", "cpu"])
    memory_goal_gb: Optional[float] = None  # 期望显存降低目标
    speed_goal_x: Optional[float] = None   # 期望速度提升倍数
    accuracy_epsilon: float = 1e-3          # 精度容忍范围


OPTIMIZATION_TARGET_REGISTRY: Dict[str, OptimizationTarget] = {
    # ================================================
    # KDA - 核心替换优化 🔥
    # ================================================
    "kda_attention": OptimizationTarget(
        name="KDA Attention",
        target_type=OptimizationTargetType.TRANSFORM,
        description="Attention 子图替换：用 KDA 路径替换标准 Attention（用于 M2 contract + gate/rollback）",
        priority=999,
        enabled=True,
        hardware_requirements=["cuda", "metal", "cpu"],
        memory_goal_gb=2.0,
        speed_goal_x=1.2,
        accuracy_epsilon=1e-3,
    ),
    
    # ================================================
    # 其他算子优化
    # ================================================
    "gemm_fusion": OptimizationTarget(
        name="GEMM Fusion",
        target_type=OptimizationTargetType.FUSION,
        description="GEMM+bias+activation算子全融合",
        priority=500,
        enabled=True,
        speed_goal_x=1.3,
    ),
    
    "rms_norm_fusion": OptimizationTarget(
        name="RMSNorm Fusion",
        target_type=OptimizationTargetType.FUSION,
        description="RMSNorm算子融合，消除多余的内存读写",
        priority=450,
        enabled=True,
        speed_goal_x=1.25,
    ),
    
    "rope_fusion": OptimizationTarget(
        name="RoPE Fusion",
        target_type=OptimizationTargetType.FUSION,
        description="RoPE旋转位置编码融合",
        priority=400,
        enabled=True,
        speed_goal_x=1.2,
    ),
    
    "tiling_128x128": OptimizationTarget(
        name="Tiling 128x128",
        target_type=OptimizationTargetType.TILING,
        description="CUDA 128x128分块优化",
        priority=300,
        hardware_requirements=["cuda"],
    ),
    
    "tiling_64x64": OptimizationTarget(
        name="Tiling 64x64",
        target_type=OptimizationTargetType.TILING,
        description="Metal 64x64分块优化",
        priority=300,
        hardware_requirements=["metal"],
    ),
    
    "mtlheap_kv_cache": OptimizationTarget(
        name="MTLHeap KV Cache",
        target_type=OptimizationTargetType.CACHING,
        description="Metal MTLHeap KV缓存零拷贝管理",
        priority=350,
        hardware_requirements=["metal"],
    ),
    
    "int4_quantization": OptimizationTarget(
        name="INT4 Quantization",
        target_type=OptimizationTargetType.QUANTIZATION,
        description="4-bit权重量化",
        priority=250,
        memory_goal_gb=3.5,
        accuracy_epsilon=1e-2,
    ),
}


def get_optimization_target(op_name: str) -> Optional[OptimizationTarget]:
    """从注册表获取指定算子的优化目标"""
    return OPTIMIZATION_TARGET_REGISTRY.get(op_name)


def list_all_optimizations() -> List[OptimizationTarget]:
    """列出所有已启用的优化目标"""
    return [target for target in OPTIMIZATION_TARGET_REGISTRY.values() if target.enabled]


__all__ = [
    "OptimizationTargetType",
    "OptimizationTarget",
    "OPTIMIZATION_TARGET_REGISTRY",
    "get_optimization_target",
    "list_all_optimizations",
]
