# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
编译策略结构体 - CompileStrategy

完整编译策略定义，包括融合/Tiling/内存/调度四大核心策略
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .agent_hints import AgentOpHint


@dataclass
class CompileStrategy:
    """
    完整编译策略

    包含四大核心策略：
    - fusion_boundary: 算子融合边界
    - tiling_config: Tiling 配置 (Tile_M, Tile_N, Tile_K)
    - memory_hierarchy: 内存层级规划 (register, l1, l2, global)
    - scheduling_plan: 调度方案 (unroll, prefetch, simd_align, pipeline)
    
    KDA 专属参数：
    - kda_beta: KDA 衰减系数
    - kda_use_delta_update: 是否使用增量更新
    - kda_use_dplr: 是否使用 DPLR 优化
    """

    fusion_boundary: Optional[List[List[str]]] = None
    tiling_config: Optional[Dict[str, int]] = None
    memory_hierarchy: Optional[Dict[str, str]] = None
    scheduling_plan: Optional[Dict[str, Any]] = None
    backend: str = "auto"
    op_hints: List[AgentOpHint] = field(default_factory=list)
    attention_config: Optional[Dict[str, Any]] = None
    enable_op_fusion: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 🔥 KDA 专属参数
    kda_beta: float = 0.1
    kda_use_delta_update: bool = True
    kda_use_dplr: bool = True

    def __post_init__(self):
        if self.fusion_boundary is None:
            self.fusion_boundary = []
        if self.tiling_config is None:
            self.tiling_config = {}
        if self.memory_hierarchy is None:
            self.memory_hierarchy = {}
        if self.scheduling_plan is None:
            self.scheduling_plan = {}
        if self.attention_config is None:
            self.attention_config = {}

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "fusion_boundary": self.fusion_boundary,
            "tiling_config": self.tiling_config,
            "memory_hierarchy": self.memory_hierarchy,
            "scheduling_plan": self.scheduling_plan,
            "backend": self.backend,
            "op_hints": [hint.value for hint in self.op_hints],
            "attention_config": self.attention_config,
            "enable_op_fusion": self.enable_op_fusion,
            "metadata": self.metadata,
            "kda_beta": self.kda_beta,
            "kda_use_delta_update": self.kda_use_delta_update,
            "kda_use_dplr": self.kda_use_dplr,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CompileStrategy":
        """从字典反序列化"""
        strategy = cls(
            fusion_boundary=d.get("fusion_boundary"),
            tiling_config=d.get("tiling_config"),
            memory_hierarchy=d.get("memory_hierarchy"),
            scheduling_plan=d.get("scheduling_plan"),
            backend=d.get("backend", "auto"),
            attention_config=d.get("attention_config"),
            enable_op_fusion=d.get("enable_op_fusion", True),
            metadata=d.get("metadata", {}),
            kda_beta=d.get("kda_beta", 0.1),
            kda_use_delta_update=d.get("kda_use_delta_update", True),
            kda_use_dplr=d.get("kda_use_dplr", True),
        )
        if "op_hints" in d:
            strategy.op_hints = [AgentOpHint(h) for h in d["op_hints"]]
        return strategy

    def load_from_ground_truth(self, gt: Dict[str, Any]):
        """从 Ground Truth 加载策略"""
        if "fusion_boundary" in gt:
            self.fusion_boundary = gt["fusion_boundary"]
        if "tiling_config" in gt:
            self.tiling_config = gt["tiling_config"]
        if "memory_hierarchy" in gt:
            self.memory_hierarchy = gt["memory_hierarchy"]
        if "scheduling_plan" in gt:
            self.scheduling_plan = gt["scheduling_plan"]
        if "kda_beta" in gt:
            self.kda_beta = gt["kda_beta"]
            self.kda_use_delta_update = gt.get("kda_use_delta_update", True)
            self.kda_use_dplr = gt.get("kda_use_dplr", True)

    def __str__(self) -> str:
        return (
            f"CompileStrategy(\n"
            f"  backend={self.backend},\n"
            f"  fusion_boundary={self.fusion_boundary},\n"
            f"  tiling_config={self.tiling_config},\n"
            f"  memory_hierarchy={self.memory_hierarchy},\n"
            f"  scheduling_plan={self.scheduling_plan},\n"
            f"  op_hints={[h.value for h in self.op_hints]},\n"
            f"  attention_config={self.attention_config},\n"
            f"  kda_beta={self.kda_beta},\n"
            f"  kda_use_delta_update={self.kda_use_delta_update},\n"
            f"  kda_use_dplr={self.kda_use_dplr}\n"
            f")"
        )
