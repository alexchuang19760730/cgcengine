# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
策略执行器 - StrategyExecutor

功能：
- 将 HarnessCompileStrategy 注入到 CGC SIMD 引擎
- 生成 CGC 命令序列
- 验证策略的正确性
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import logging
from .harness_agent import HarnessCompileStrategy, AgentOpHint

logger = logging.getLogger(__name__)

try:
    from ..cgc.cgc_strategy_injection import inject_strategy, set_backend, reset_strategy
    CGC_INJECTION_AVAILABLE = True
except ImportError:
    CGC_INJECTION_AVAILABLE = False
    logger.warning("[StrategyExecutor] CGC strategy injection not available")


class StrategyExecutor:
    """策略执行器"""

    def __init__(self, backend: Optional[str] = None, device: Optional[str] = None):
        self.executed: bool = False
        self.errors: List[str] = []
        self._c_injection_available = CGC_INJECTION_AVAILABLE
        self.backend = backend
        self.device = device

    def execute(
        self,
        strategy: HarnessCompileStrategy,
        model: nn.Module,
        executor: Any = None,  # CGCExecutor (optional)
        **kwargs,
    ) -> nn.Module:
        """
        执行策略：注入到 CGC 引擎

        Args:
            strategy: Harness 编译策略
            model: 模型
            executor: CGC 执行器（可选）

        Returns:
            优化后的模型
        """
        logger.info("[StrategyExecutor] Executing strategy...")

        # 注入策略到 C++ Engine (如果可用)
        if self._c_injection_available:
            success = inject_strategy(strategy)
            if success:
                logger.info("[StrategyExecutor] Strategy injected to C++ Engine successfully")
            else:
                logger.warning("[StrategyExecutor] Failed to inject strategy to C++ Engine")

        # 注入后端和 tiling (Python 端)
        self._inject_backend(strategy, executor)

        # 注入优化提示
        self._inject_optimization_hints(strategy, executor)

        # 注册/启用算子
        self._apply_op_hints(strategy, executor)

        # 生成 CGC 命令
        cgc_commands = self._generate_cgc_commands(strategy)
        if cgc_commands:
            logger.info(f"[StrategyExecutor] Generated {len(cgc_commands)} CGC commands")

        self.executed = True
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            return model

        max_new_tokens = int(kwargs.get("max_new_tokens", 64))
        temperature = float(kwargs.get("temperature", 0.8))
        top_k = kwargs.get("top_k")
        top_p = kwargs.get("top_p")

        if self.device is not None:
            try:
                input_ids = input_ids.to(self.device)
            except Exception:
                pass

        if hasattr(model, "generate"):
            gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_k is not None:
                gen_kwargs["top_k"] = int(top_k)
            if top_p is not None:
                gen_kwargs["top_p"] = float(top_p)
            try:
                return model.generate(input_ids=input_ids, **gen_kwargs)
            except Exception:
                return model.generate(input_ids, **gen_kwargs)

        try:
            return model(input_ids)
        except Exception:
            return input_ids

    def _inject_backend(
        self,
        strategy: HarnessCompileStrategy,
        executor: Optional[Any],
    ):
        """注入后端选择"""
        if strategy.backend != "auto":
            if self._c_injection_available:
                success = set_backend(strategy.backend)
                if success:
                    logger.info(f"[StrategyExecutor] C++ Backend set to {strategy.backend}")

            if executor and hasattr(executor, "set_backend"):
                try:
                    executor.set_backend(strategy.backend)
                    logger.info(f"[StrategyExecutor] Python backend set to {strategy.backend}")
                except Exception as e:
                    logger.warning(f"Failed to set backend: {e}")

    def _inject_optimization_hints(
        self,
        strategy: HarnessCompileStrategy,
        executor: Any,
    ):
        """注入优化参数（tiling, memory, schedules）"""
        if hasattr(executor, "set_tile_sizes") and strategy.tile_sizes:
            try:
                executor.set_tile_sizes(strategy.tile_sizes)
            except Exception as e:
                logger.warning(f"Failed to set tile sizes: {e}")

        if hasattr(executor, "set_schedules") and strategy.schedules:
            try:
                executor.set_schedules(strategy.schedules)
            except Exception as e:
                logger.warning(f"Failed to set schedules: {e}")

    def _apply_op_hints(
        self,
        strategy: HarnessCompileStrategy,
        executor: Any,
    ):
        """应用操作提示"""
        for hint in strategy.op_hints:
            if hint == AgentOpHint.FLASH_ATTENTION:
                if hasattr(executor, "enable_flash_attention"):
                    try:
                        executor.enable_flash_attention()
                        logger.info("[StrategyExecutor] Flash Attention enabled")
                    except Exception as e:
                        logger.warning(f"Failed to enable flash attention: {e}")

            elif hint == AgentOpHint.MOE_ROUTING:
                if hasattr(executor, "enable_moe_routing"):
                    try:
                        executor.enable_moe_routing()
                        logger.info("[StrategyExecutor] MoE Routing enabled")
                    except Exception as e:
                        logger.warning(f"Failed to enable MoE routing: {e}")

            elif hint == AgentOpHint.TENSOR_PARALLEL:
                if hasattr(executor, "enable_tensor_parallel"):
                    try:
                        executor.enable_tensor_parallel(strategy.tp_degree)
                        logger.info(f"[StrategyExecutor] TP enabled, degree={strategy.tp_degree}")
                    except Exception as e:
                        logger.warning(f"Failed to enable TP: {e}")

            elif hint == AgentOpHint.VLM_CROSS_ATTENTION:
                if hasattr(executor, "enable_vlm_cross_attention"):
                    try:
                        executor.enable_vlm_cross_attention()
                        logger.info("[StrategyExecutor] VLM Cross Attention enabled")
                    except Exception as e:
                        logger.warning(f"Failed to enable VLM cross attention: {e}")

    def _generate_cgc_commands(
        self,
        strategy: HarnessCompileStrategy,
    ) -> List[Dict]:
        """生成 CGC 命令序列（用于注入）"""
        commands = []

        # 1. 后端选择
        if strategy.backend != "auto":
            commands.append({
                "op": "set_backend",
                "value": strategy.backend,
            })

        # 2. Tile sizes
        for k, v in strategy.tile_sizes.items():
            commands.append({
                "op": "set_tile",
                "key": k,
                "value": v,
            })

        # 3. Fusion enable
        if strategy.enable_op_fusion:
            commands.append({
                "op": "enable_fusion",
            })

        return commands
