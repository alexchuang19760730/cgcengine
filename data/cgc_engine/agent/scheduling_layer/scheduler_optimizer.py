# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
Scheduler Optimizer - 調度層學習器

功能：
- 從 llama.cpp/vLLM 的反饋中學習調度策略
- 生成優化的 Batch 調度配置
- 生成優化的 Prefill/Decode 分離配置
- 生成優化的 Prefix Caching 配置

使用方式：
    from cgc_engine.agent.scheduling_layer.scheduler_optimizer import SchedulerOptimizer

    optimizer = SchedulerOptimizer()
    strategy = optimizer.learn(feedback)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum
import logging
import time

if TYPE_CHECKING:
    from .scheduling_feedback import SchedulingFeedback, SchedulingMetrics

logger = logging.getLogger(__name__)


class BatchStrategyType(Enum):
    """Batch 策略類型"""
    STATIC = "static"
    CONTINUOUS = "continuous"
    DYNAMIC = "dynamic"


class SchedulerTuningLevel(Enum):
    """調度器調優級別"""
    CONSERVATIVE = "conservative"  # 保守：只在明確更好時改變
    MODERATE = "moderate"  # 中等：允許適度的優化
    AGGRESSIVE = "aggressive"  # 激進：追求最大性能


@dataclass
class SchedulerStrategy:
    """
    調度策略

    包含所有調度相關的優化參數
    """

    # Batch 調度
    enable_continuous_batching: bool = False
    dynamic_batch_size: int = 32
    max_batch_size: int = 128
    min_batch_size: int = 1
    batch_timeout_ms: float = 10.0

    # Prefill/Decode 分離
    enable_pd_separation: bool = True
    prefill_chunk_size: int = 512
    prefill_max_chunks: int = 16
    hybrid_threshold: int = 4096
    prefill_only_threshold: int = 512

    # Prefix Caching
    enable_prefix_cache: bool = True
    prefix_cache_aggressive: bool = False
    prefix_match_min_length: int = 32
    prefix_cache_max_size_mb: float = 4096.0

    # Token 分配
    max_tokens_per_request: int = 8192
    max_total_tokens: int = 131072
    waiting_time_limit_ms: float = 100.0

    # Memory
    kv_cache_memory_ratio: float = 0.8
    enable_kv_quantization: bool = True
    kv_quant_bits: int = 8

    # Priority
    priority_mode: str = "fair"  # "fair", "latency", "throughput"

    # GPU
    enable_gpu_optimization: bool = True
    num_parallel_workers: int = 4

    # Metadata
    source_feedback_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    tuning_level: str = SchedulerTuningLevel.MODERATE.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_continuous_batching": self.enable_continuous_batching,
            "dynamic_batch_size": self.dynamic_batch_size,
            "max_batch_size": self.max_batch_size,
            "enable_pd_separation": self.enable_pd_separation,
            "prefill_chunk_size": self.prefill_chunk_size,
            "hybrid_threshold": self.hybrid_threshold,
            "enable_prefix_cache": self.enable_prefix_cache,
            "prefix_cache_aggressive": self.prefix_cache_aggressive,
            "kv_cache_memory_ratio": self.kv_cache_memory_ratio,
            "enable_kv_quantization": self.enable_kv_quantization,
            "kv_quant_bits": self.kv_quant_bits,
            "priority_mode": self.priority_mode,
            "source_feedback_score": self.source_feedback_score,
            "tuning_level": self.tuning_level,
        }


class SchedulerOptimizer:
    """
    調度層優化器

    從 llama.cpp/vLLM 的反饋中學習，生成優化的調度策略
    """

    def __init__(
        self,
        tuning_level: SchedulerTuningLevel = SchedulerTuningLevel.MODERATE
    ):
        self.tuning_level = tuning_level
        self.strategy_history: List[SchedulerStrategy] = []
        self.feedback_history: List["SchedulingFeedback"] = []

    def learn(
        self,
        feedback: "SchedulingFeedback",
        current_strategy: Optional[SchedulerStrategy] = None
    ) -> SchedulerStrategy:
        """
        從反饋中學習，生成優化的策略

        Args:
            feedback: 從 llama.cpp/vLLM 收集的反饋
            current_strategy: 當前的策略（可選，用於增量優化）

        Returns:
            優化後的策略
        """
        logger.info(f"[SchedulerOptimizer] Learning from {feedback.engine} feedback")

        if current_strategy is None:
            current_strategy = SchedulerStrategy()

        new_strategy = self._create_baseline_strategy()

        # 根據 tuning level 應用不同的學習策略
        if self.tuning_level == SchedulerTuningLevel.CONSERVATIVE:
            new_strategy = self._learn_conservative(feedback, current_strategy)
        elif self.tuning_level == SchedulerTuningLevel.MODERATE:
            new_strategy = self._learn_moderate(feedback, current_strategy)
        else:
            new_strategy = self._learn_aggressive(feedback, current_strategy)

        # 記錄歷史
        self.strategy_history.append(new_strategy)
        self.feedback_history.append(feedback)

        logger.info(
            f"[SchedulerOptimizer] Generated strategy with "
            f"score={new_strategy.source_feedback_score:.2f}"
        )

        return new_strategy

    def _create_baseline_strategy(self) -> SchedulerStrategy:
        """創建基線策略"""
        return SchedulerStrategy(
            enable_continuous_batching=False,
            dynamic_batch_size=32,
            max_batch_size=128,
            enable_pd_separation=True,
            prefill_chunk_size=512,
            hybrid_threshold=4096,
            enable_prefix_cache=True,
            kv_cache_memory_ratio=0.8,
            enable_kv_quantization=True,
            kv_quant_bits=8,
        )

    def _learn_conservative(
        self,
        feedback: "SchedulingFeedback",
        current: SchedulerStrategy
    ) -> SchedulerStrategy:
        """保守學習：只在明確更好時才做改變"""
        new_strategy = SchedulerStrategy(
            enable_continuous_batching=current.enable_continuous_batching,
            dynamic_batch_size=current.dynamic_batch_size,
            max_batch_size=current.max_batch_size,
            enable_pd_separation=current.enable_pd_separation,
            prefill_chunk_size=current.prefill_chunk_size,
            hybrid_threshold=current.hybrid_threshold,
            enable_prefix_cache=current.enable_prefix_cache,
            prefix_cache_aggressive=current.prefix_cache_aggressive,
            kv_cache_memory_ratio=current.kv_cache_memory_ratio,
            enable_kv_quantization=current.enable_kv_quantization,
            kv_quant_bits=current.kv_quant_bits,
            priority_mode=current.priority_mode,
            tuning_level=SchedulerTuningLevel.CONSERVATIVE.value,
        )

        metrics = feedback.metrics

        # 只有當 vLLM 明確更好時才採用
        if metrics.batch_strategy == "continuous" and metrics.avg_batch_utilization > 0.9:
            new_strategy.enable_continuous_batching = True
            new_strategy.dynamic_batch_size = max(
                current.dynamic_batch_size,
                metrics.batch_size
            )

        # Prefix Cache - 保持當前設置
        if metrics.prefix_cache_hit_rate > 0.9:
            new_strategy.enable_prefix_cache = True
            new_strategy.prefix_cache_aggressive = True

        new_strategy.source_feedback_score = feedback.get_effectiveness_score()
        return new_strategy

    def _learn_moderate(
        self,
        feedback: "SchedulingFeedback",
        current: SchedulerStrategy
    ) -> SchedulerStrategy:
        """中等學習：允許適度的優化"""
        metrics = feedback.metrics

        new_strategy = SchedulerStrategy(
            enable_continuous_batching=True,  # 默認啟用
            dynamic_batch_size=max(32, min(128, metrics.batch_size)),
            max_batch_size=current.max_batch_size,
            enable_pd_separation=True,
            prefill_chunk_size=max(256, min(1024, metrics.prefill_chunk_size)),
            hybrid_threshold=metrics.hybrid_threshold,
            enable_prefix_cache=True,
            prefix_cache_aggressive=metrics.prefix_cache_hit_rate > 0.8,
            kv_cache_memory_ratio=min(0.9, metrics.memory_pressure + 0.1),
            enable_kv_quantization=True,
            kv_quant_bits=current.kv_quant_bits,
            priority_mode=current.priority_mode,
            tuning_level=SchedulerTuningLevel.MODERATE.value,
        )

        # Batch 策略
        if metrics.batch_strategy == "continuous":
            new_strategy.enable_continuous_batching = True
            new_strategy.dynamic_batch_size = max(
                current.dynamic_batch_size,
                int(metrics.batch_size * 0.9)  # 採用 90%
            )

        # Prefill/Decode 分離
        if metrics.prefill_chunk_size > 0:
            new_strategy.prefill_chunk_size = max(
                256, int(metrics.prefill_chunk_size * 0.9)
            )

        # Hybrid threshold
        new_strategy.hybrid_threshold = metrics.hybrid_threshold

        # Prefix Cache
        new_strategy.enable_prefix_cache = True
        new_strategy.prefix_cache_aggressive = metrics.prefix_cache_hit_rate > 0.7

        # Memory
        new_strategy.kv_cache_memory_ratio = min(0.85, max(0.7, metrics.memory_pressure + 0.05))

        new_strategy.source_feedback_score = feedback.get_effectiveness_score()
        return new_strategy

    def _learn_aggressive(
        self,
        feedback: "SchedulingFeedback",
        current: SchedulerStrategy
    ) -> SchedulerStrategy:
        """激進學習：追求最大性能"""
        metrics = feedback.metrics

        new_strategy = SchedulerStrategy(
            enable_continuous_batching=True,
            dynamic_batch_size=min(256, int(metrics.batch_size * 1.2)),
            max_batch_size=256,
            enable_pd_separation=True,
            prefill_chunk_size=min(2048, metrics.prefill_chunk_size * 2),
            hybrid_threshold=metrics.hybrid_threshold,
            enable_prefix_cache=True,
            prefix_cache_aggressive=True,
            kv_cache_memory_ratio=min(0.9, metrics.memory_pressure + 0.15),
            enable_kv_quantization=True,
            kv_quant_bits=4 if metrics.kv_cache_memory_mb > 2048 else 8,
            priority_mode="throughput",
            tuning_level=SchedulerTuningLevel.AGGRESSIVE.value,
        )

        # 採用 vLLM 的最佳實踐
        if metrics.batch_strategy == "continuous":
            new_strategy.enable_continuous_batching = True
            new_strategy.dynamic_batch_size = min(256, int(metrics.batch_size * 1.2))
            new_strategy.batch_timeout_ms = metrics.avg_waiting_time_ms / 2

        # 採用更大的 batch
        new_strategy.max_batch_size = min(256, max(128, int(metrics.batch_size * 1.5)))

        # 採用更好的 prefix caching
        if metrics.prefix_cache_hit_rate > 0.5:
            new_strategy.enable_prefix_cache = True
            new_strategy.prefix_cache_aggressive = True
            new_strategy.prefix_cache_max_size_mb = min(
                8192, metrics.kv_cache_memory_mb * 1.5
            )

        # Memory 優化
        new_strategy.kv_cache_memory_ratio = min(0.9, metrics.memory_pressure + 0.2)
        new_strategy.enable_kv_quantization = True
        new_strategy.kv_quant_bits = 4  # 激進使用 INT4

        new_strategy.source_feedback_score = feedback.get_effectiveness_score()
        return new_strategy

    def learn_from_comparison(
        self,
        cgc_feedback: "SchedulingFeedback",
        reference_feedback: "SchedulingFeedback"
    ) -> SchedulerStrategy:
        """
        從 CGC 和參考引擎的對比中學習

        Args:
            cgc_feedback: CGC Engine 的反饋
            reference_feedback: llama.cpp/vLLM 的反饋

        Returns:
            優化後的策略
        """
        logger.info(
            f"[SchedulerOptimizer] Learning from comparison: "
            f"CGC ({cgc_feedback.metrics.total_latency_ms:.1f}ms) vs "
            f"{reference_feedback.engine} ({reference_feedback.metrics.total_latency_ms:.1f}ms)"
        )

        # 採用參考引擎的策略
        new_strategy = self.learn(reference_feedback)

        # 根據 CGC 的差距調整
        cgc_latency = cgc_feedback.metrics.total_latency_ms
        ref_latency = reference_feedback.metrics.total_latency_ms

        if cgc_latency > ref_latency * 1.2:
            # CGC 比參考慢 20% 以上，採用更激進的策略
            logger.info("[SchedulerOptimizer] CGC is slower, applying aggressive tuning")
            new_strategy = self._learn_aggressive(reference_feedback, new_strategy)
        elif cgc_latency > ref_latency * 1.1:
            # CGC 比參考慢 10-20%，採用中等策略
            logger.info("[SchedulerOptimizer] CGC is slightly slower, applying moderate tuning")
            new_strategy = self._learn_moderate(reference_feedback, new_strategy)

        return new_strategy

    def get_best_strategy(self) -> SchedulerStrategy:
        """
        從歷史中獲取最佳策略

        Returns:
            效果分數最高的策略
        """
        if not self.strategy_history:
            return SchedulerStrategy()

        return max(
            self.strategy_history,
            key=lambda s: s.source_feedback_score
        )

    def get_strategy_stats(self) -> Dict[str, Any]:
        """獲取策略統計"""
        if not self.strategy_history:
            return {
                "num_strategies": 0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
            }

        scores = [s.source_feedback_score for s in self.strategy_history]
        return {
            "num_strategies": len(self.strategy_history),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "latest_tuning_level": self.strategy_history[-1].tuning_level,
        }
