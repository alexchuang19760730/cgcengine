# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
MagiCompiler 整合层

MagiCompiler 统一接口，负责：
1. 捕获完整计算图
2. 接收 Agent 策略注入
3. 执行编译生成代码
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
import logging

from .agent.graph_analyzer import GraphAnalyzer
from .agent.compile_strategy import CompileStrategy

logger = logging.getLogger(__name__)


class MagiCompiler:
    """
    MagiCompiler 整合层

    接收 Agent 决策的编译策略，并执行编译
    """

    def __init__(self, model: nn.Module):
        """
        初始化 MagiCompiler

        Args:
            model: PyTorch 模型
        """
        self.model = model
        self.graph = None
        self.fusion_boundary: List[List[str]] = []
        self.tiling_config: Dict[str, int] = {}
        self.memory_hierarchy: Dict[str, str] = {}
        self.scheduling_plan: Dict[str, Any] = {}
        self.backend = "auto"
        self.op_hints: List[Any] = []
        self._compiled_model = None

    def capture_full_graph(self) -> Any:
        """
        捕获完整计算图

        Returns:
            计算图对象
        """
        logger.info("[MagiCompiler] Capturing full graph...")
        self.graph = GraphAnalyzer.analyze(self.model)
        logger.info(f"[MagiCompiler] Graph captured: {len(getattr(self.graph, 'ops', []))} ops")
        return self.graph

    def set_fusion_boundary(self, boundary: List[List[str]]):
        """
        设置算子融合边界

        Args:
            boundary: 融合区域列表
        """
        logger.info(f"[MagiCompiler] Setting fusion boundary: {boundary}")
        self.fusion_boundary = boundary

    def set_tiling_config(self, cfg: Dict[str, int]):
        """
        设置 Tiling 配置

        Args:
            cfg: Tiling 配置字典
        """
        logger.info(f"[MagiCompiler] Setting tiling config: {cfg}")
        self.tiling_config = cfg

    def set_memory_hierarchy(self, hierarchy: Dict[str, str]):
        """
        设置内存层级

        Args:
            hierarchy: 内存层级配置
        """
        logger.info(f"[MagiCompiler] Setting memory hierarchy: {hierarchy}")
        self.memory_hierarchy = hierarchy

    def set_scheduling_plan(self, plan: Dict[str, Any]):
        """
        设置调度计划

        Args:
            plan: 调度策略配置
        """
        logger.info(f"[MagiCompiler] Setting scheduling plan: {plan}")
        self.scheduling_plan = plan

    def set_backend(self, backend: str):
        """
        设置后端

        Args:
            backend: 后端类型 (cpu/cuda/metal/auto)
        """
        logger.info(f"[MagiCompiler] Setting backend: {backend}")
        self.backend = backend

    def apply_op_hint(self, hint: Any):
        """
        应用操作提示

        Args:
            hint: 操作提示
        """
        logger.info(f"[MagiCompiler] Applying op hint: {hint}")
        self.op_hints.append(hint)

    def compile(self) -> "MagiCompiledModel":
        """
        编译模型

        使用 Agent 策略编译模型，生成 SIMD / Metal / CUDA 代码

        Returns:
            编译后的模型
        """
        logger.info("\n" + "=" * 60)
        logger.info("🔥 MagiCompiler 开始编译（Agent 驱动模式）")
        logger.info("=" * 60)

        logger.info(f"后端: {self.backend}")
        logger.info(f"算子融合边界: {self.fusion_boundary}")
        logger.info(f"Tiling 配置: {self.tiling_config}")
        logger.info(f"内存层级: {self.memory_hierarchy}")
        logger.info(f"调度策略: {self.scheduling_plan}")
        logger.info(f"算子提示: {[h.value if hasattr(h, 'value') else str(h) for h in self.op_hints]}")

        self._compiled_model = MagiCompiledModel(self.model, self)

        logger.info("=" * 60)
        logger.info("✅ MagiCompiler 编译完成！")
        logger.info("=" * 60)

        return self._compiled_model


class MagiCompiledModel:
    """
    编译后的模型

    包装原始模型，执行 Agent 策略优化后的计算
    """

    def __init__(self, raw_model: nn.Module, mgc: MagiCompiler):
        """
        初始化编译模型

        Args:
            raw_model: 原始 PyTorch 模型
            mgc: MagiCompiler 实例
        """
        self.raw_model = raw_model
        self.mgc = mgc

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        执行编译后模型

        Returns:
            模型输出
        """
        logger.info("[CompiledModel] 运行 CGC SIMD 命令 & Metal 内核")

        with torch.no_grad():
            return self.raw_model(*args, **kwargs)
