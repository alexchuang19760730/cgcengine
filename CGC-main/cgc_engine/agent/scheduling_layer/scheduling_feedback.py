# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
Scheduling Feedback Module - 調度層反饋收集

功能：
- 從 llama.cpp/vLLM 收集調度相關的反饋
- 分析 Batch 調度策略
- 分析 Prefill/Decode 分離效果
- 分析 Prefix Caching 命中率

使用方式：
    from cgc_engine.agent.scheduling_layer import SchedulingFeedback, SchedulingFeedbackCollector

    collector = SchedulingFeedbackCollector()
    feedback = collector.collect(engine="vllm")
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import time
import logging

import torch

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Batch 調度策略"""
    STATIC = "static"
    CONTINUOUS = "continuous"
    DYNAMIC = "dynamic"
    SPECULATIVE = "speculative"


class PDPhase(Enum):
    """Prefill/Decode 階段"""
    PREFILL = "prefill"
    DECODE = "decode"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


@dataclass
class SchedulingMetrics:
    """調度指標"""
    # Batch 調度
    batch_size: int = 0
    batch_strategy: str = "static"
    num_batches: int = 0
    avg_batch_utilization: float = 0.0

    # Prefill/Decode
    prefill_chunk_size: int = 0
    prefill_batch_size: int = 0
    decode_batch_size: int = 0
    hybrid_threshold: int = 4096
    current_phase: str = "unknown"

    # Token 分配
    total_tokens_processed: int = 0
    tokens_per_second: float = 0.0
    avg_waiting_time_ms: float = 0.0
    max_waiting_time_ms: float = 0.0

    # Prefix Caching
    prefix_cache_enabled: bool = False
    prefix_cache_hits: int = 0
    prefix_cache_misses: int = 0
    prefix_cache_hit_rate: float = 0.0
    num_prefix_reuses: int = 0

    # Memory
    kv_cache_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    memory_pressure: float = 0.0

    # Latency
    prefill_latency_ms: float = 0.0
    decode_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    time_per_output_token_ms: float = 0.0

    # GPU Utilization
    gpu_utilization: float = 0.0
    gpu_memory_utilization: float = 0.0

    # Timestamp
    timestamp: float = field(default_factory=time.time)


@dataclass
class SchedulingFeedback:
    """
    調度層反饋

    包含從 llama.cpp/vLLM 收集的所有調度相關信息
    """

    # 引擎信息
    engine: str = "unknown"
    engine_version: str = ""

    # 指標
    metrics: SchedulingMetrics = field(default_factory=SchedulingMetrics)

    # 當前配置
    config: Dict[str, Any] = field(default_factory=dict)

    # 額外信息
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_effectiveness_score(self) -> float:
        """
        計算調度有效性分數 (0-100)

        基於多個指標綜合評估
        """
        scores = []

        # Batch 利用率 (30%)
        if self.metrics.avg_batch_utilization > 0:
            batch_score = min(self.metrics.avg_batch_utilization * 100, 100)
            scores.append(("batch", batch_score, 0.3))

        # Prefix Cache 命中率 (20%)
        if self.metrics.prefix_cache_enabled:
            cache_score = self.metrics.prefix_cache_hit_rate * 100
            scores.append(("cache", cache_score, 0.2))

        # 內存效率 (20%)
        if self.metrics.available_memory_mb > 0:
            memory_score = (1 - self.metrics.memory_pressure) * 100
            scores.append(("memory", memory_score, 0.2))

        # 延遲效率 (30%)
        if self.metrics.total_latency_ms > 0:
            latency_score = max(0, 100 - self.metrics.total_latency_ms / 10)
            scores.append(("latency", latency_score, 0.3))

        # 計算加權分數
        total_weight = sum(weight for _, _, weight in scores)
        if total_weight == 0:
            return 0.0

        weighted_score = sum(score * weight for _, score, weight in scores)
        return weighted_score / total_weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "metrics": {
                "batch_size": self.metrics.batch_size,
                "batch_strategy": self.metrics.batch_strategy,
                "prefill_chunk_size": self.metrics.prefill_chunk_size,
                "decode_batch_size": self.metrics.decode_batch_size,
                "hybrid_threshold": self.metrics.hybrid_threshold,
                "current_phase": self.metrics.current_phase,
                "prefix_cache_hit_rate": self.metrics.prefix_cache_hit_rate,
                "avg_waiting_time_ms": self.metrics.avg_waiting_time_ms,
                "kv_cache_memory_mb": self.metrics.kv_cache_memory_mb,
                "total_latency_ms": self.metrics.total_latency_ms,
                "tokens_per_second": self.metrics.tokens_per_second,
            },
            "config": self.config,
            "effectiveness_score": self.get_effectiveness_score(),
        }


class SchedulingFeedbackCollector:
    """
    調度層反饋收集器

    從 llama.cpp 或 vLLM 收集調度相關的反饋信息
    """

    def __init__(self):
        self.engine_type: Optional[str] = None
        self.engine_instance: Optional[Any] = None
        self._metrics_history: List[SchedulingMetrics] = []

    def set_engine(self, engine_type: str, engine_instance: Any):
        """
        設置要收集的引擎

        Args:
            engine_type: "llama.cpp" 或 "vllm"
            engine_instance: 引擎實例
        """
        self.engine_type = engine_type
        self.engine_instance = engine_instance
        logger.info(f"[SchedulingCollector] Engine set to {engine_type}")

    def collect(self, engine: Optional[str] = None) -> SchedulingFeedback:
        """
        收集調度反饋

        Args:
            engine: 引擎類型，可選 "llama.cpp" 或 "vllm"

        Returns:
            SchedulingFeedback 對象
        """
        engine = engine or self.engine_type

        if engine == "llama.cpp":
            return self._collect_llama_cpp()
        elif engine == "vllm":
            return self._collect_vllm()
        else:
            logger.warning(f"[SchedulingCollector] Unknown engine: {engine}, returning empty feedback")
            return SchedulingFeedback()

    def _collect_llama_cpp(self) -> SchedulingFeedback:
        """從 llama.cpp 收集反饋"""
        feedback = SchedulingFeedback()
        feedback.engine = "llama.cpp"

        try:
            if self.engine_instance is not None:
                metrics = self._get_llama_cpp_metrics()
                feedback.metrics = metrics

                feedback.config = {
                    "batch_size": getattr(self.engine_instance, 'n_batch', 512),
                    "threads": getattr(self.engine_instance, 'n_threads', 8),
                    "ctx_size": getattr(self.engine_instance, 'n_ctx', 2048),
                }
        except Exception as e:
            logger.error(f"[SchedulingCollector] Failed to collect llama.cpp metrics: {e}")

        return feedback

    def _collect_vllm(self) -> SchedulingFeedback:
        """從 vLLM 收集反饋"""
        feedback = SchedulingFeedback()
        feedback.engine = "vllm"

        try:
            if self.engine_instance is not None:
                metrics = self._get_vllm_metrics()
                feedback.metrics = metrics

                feedback.config = {
                    "max_model_len": getattr(self.engine_instance, 'max_model_len', 4096),
                    "block_size": getattr(self.engine_instance, 'block_size', 16),
                }
        except Exception as e:
            logger.error(f"[SchedulingCollector] Failed to collect vLLM metrics: {e}")

        return feedback

    def _get_llama_cpp_metrics(self) -> SchedulingMetrics:
        """獲取 llama.cpp 的調度指標"""
        metrics = SchedulingMetrics()

        try:
            if self.engine_instance is None:
                return metrics

            # Batch 策略 (llama.cpp 使用 static batch)
            metrics.batch_strategy = BatchStrategy.STATIC.value
            metrics.batch_size = getattr(self.engine_instance, 'n_batch', 512)

            # llama.cpp 沒有明確的 PD 分離
            metrics.current_phase = PDPhase.UNKNOWN.value

            # 嘗試從 internal state 獲取更多信息
            if hasattr(self.engine_instance, 'ctx'):
                ctx = self.engine_instance.ctx
                if hasattr(ctx, 'kv_cache'):
                    kv_cache = ctx.kv_cache
                    metrics.kv_cache_memory_mb = kv_cache.size / (1024 * 1024) if hasattr(kv_cache, 'size') else 0

        except Exception as e:
            logger.warning(f"[SchedulingCollector] Failed to get llama.cpp metrics: {e}")

        return metrics

    def _get_vllm_metrics(self) -> SchedulingMetrics:
        """獲取 vLLM 的調度指標"""
        metrics = SchedulingMetrics()

        try:
            if self.engine_instance is None:
                return metrics

            # vLLM 使用 continuous batching
            metrics.batch_strategy = BatchStrategy.CONTINUOUS.value

            # 嘗試從 scheduler 獲取信息
            if hasattr(self.engine_instance, 'scheduler'):
                scheduler = self.engine_instance.scheduler
                if hasattr(scheduler, 'past_cache'):
                    past_cache = scheduler.past_cache
                    metrics.prefix_cache_hit_rate = past_cache.hit_rate if hasattr(past_cache, 'hit_rate') else 0.0

            # 嘗試從 cache manager 獲取信息
            if hasattr(self.engine_instance, 'cache_engine'):
                cache_engine = self.engine_instance.cache_engine
                if hasattr(cache_engine, 'num_blocks'):
                    metrics.kv_cache_memory_mb = cache_engine.num_blocks * 16 / 1024  # 假設 block_size=16

            # GPU 利用率
            if torch.cuda.is_available():
                metrics.gpu_utilization = self._get_gpu_utilization()

        except Exception as e:
            logger.warning(f"[SchedulingCollector] Failed to get vLLM metrics: {e}")

        return metrics

    def _get_gpu_utilization(self) -> float:
        """獲取 GPU 利用率"""
        try:
            if torch.cuda.is_available():
                # 簡單的內存使用率作為代理
                allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                return min(allocated * 100, 100.0)
        except Exception:
            pass
        return 0.0

    def collect_comparison(
        self,
        cgc_metrics: SchedulingMetrics,
        reference_metrics: SchedulingMetrics
    ) -> Dict[str, Any]:
        """
        收集 CGC 和參考引擎的對比

        Args:
            cgc_metrics: CGC Engine 的指標
            reference_metrics: llama.cpp/vLLM 的指標

        Returns:
            對比結果字典
        """
        return {
            "batch": {
                "cgc": cgc_metrics.batch_size,
                "reference": reference_metrics.batch_size,
                "improvement": reference_metrics.batch_size - cgc_metrics.batch_size,
            },
            "latency": {
                "cgc_ms": cgc_metrics.total_latency_ms,
                "reference_ms": reference_metrics.total_latency_ms,
                "improvement_ratio": cgc_metrics.total_latency_ms / reference_metrics.total_latency_ms
                    if reference_metrics.total_latency_ms > 0 else 0,
            },
            "prefix_cache": {
                "cgc_hit_rate": cgc_metrics.prefix_cache_hit_rate,
                "reference_hit_rate": reference_metrics.prefix_cache_hit_rate,
            },
            "memory": {
                "cgc_mb": cgc_metrics.kv_cache_memory_mb,
                "reference_mb": reference_metrics.kv_cache_memory_mb,
            },
        }


class MockSchedulingCollector(SchedulingFeedbackCollector):
    """Mock 版本的收集器，用於測試"""

    def _collect_llama_cpp(self) -> SchedulingFeedback:
        """返回模擬的 llama.cpp 反饋"""
        feedback = super()._collect_llama_cpp()
        feedback.engine = "llama.cpp"
        feedback.engine_version = "mock"

        feedback.metrics = SchedulingMetrics(
            batch_size=32,
            batch_strategy=BatchStrategy.STATIC.value,
            num_batches=100,
            avg_batch_utilization=0.85,
            prefill_chunk_size=512,
            prefill_batch_size=8,
            decode_batch_size=32,
            hybrid_threshold=4096,
            current_phase=PDPhase.DECODE.value,
            total_tokens_processed=10000,
            tokens_per_second=150.5,
            avg_waiting_time_ms=25.0,
            max_waiting_time_ms=150.0,
            prefix_cache_enabled=True,
            prefix_cache_hits=500,
            prefix_cache_misses=100,
            prefix_cache_hit_rate=0.833,
            num_prefix_reuses=50,
            kv_cache_memory_mb=2048.0,
            available_memory_mb=8192.0,
            memory_pressure=0.25,
            prefill_latency_ms=50.0,
            decode_latency_ms=15.0,
            total_latency_ms=650.0,
            time_to_first_token_ms=80.0,
            time_per_output_token_ms=12.5,
            gpu_utilization=0.78,
            gpu_memory_utilization=0.65,
        )

        feedback.config = {
            "batch_size": 32,
            "threads": 8,
            "ctx_size": 2048,
        }

        return feedback

    def _collect_vllm(self) -> SchedulingFeedback:
        """返回模擬的 vLLM 反饋"""
        feedback = super()._collect_vllm()
        feedback.engine = "vllm"
        feedback.engine_version = "mock"

        feedback.metrics = SchedulingMetrics(
            batch_size=64,
            batch_strategy=BatchStrategy.CONTINUOUS.value,
            num_batches=200,
            avg_batch_utilization=0.92,
            prefill_chunk_size=256,
            prefill_batch_size=16,
            decode_batch_size=64,
            hybrid_threshold=4096,
            current_phase=PDPhase.DECODE.value,
            total_tokens_processed=25000,
            tokens_per_second=280.0,
            avg_waiting_time_ms=10.0,
            max_waiting_time_ms=50.0,
            prefix_cache_enabled=True,
            prefix_cache_hits=1500,
            prefix_cache_misses=200,
            prefix_cache_hit_rate=0.882,
            num_prefix_reuses=150,
            kv_cache_memory_mb=1536.0,
            available_memory_mb=8192.0,
            memory_pressure=0.19,
            prefill_latency_ms=45.0,
            decode_latency_ms=10.0,
            total_latency_ms=550.0,
            time_to_first_token_ms=60.0,
            time_per_output_token_ms=8.5,
            gpu_utilization=0.88,
            gpu_memory_utilization=0.72,
        )

        feedback.config = {
            "max_model_len": 8192,
            "block_size": 16,
        }

        return feedback
