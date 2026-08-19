# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
Storage Optimizer - 存儲層學習器

功能：
- 從 llama.cpp/vLLM 的反饋中學習存儲策略
- 生成優化的 KV Cache 配置
- 生成優化的 KDA 配置
- 生成優化的 Prefetch 配置
- 生成優化的 Memory Layout 配置

使用方式：
    from cgc_engine.agent.storage_layer.storage_optimizer import StorageOptimizer

    optimizer = StorageOptimizer()
    strategy = optimizer.learn(feedback)
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum
import logging
import time

from ..base_strategy import BaseStrategy

if TYPE_CHECKING:
    from .storage_feedback import StorageFeedback, StorageMetrics

logger = logging.getLogger(__name__)


class StorageTuningLevel(Enum):
    """存儲調優級別"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class StorageStrategy(BaseStrategy):
    """
    存儲策略

    包含所有存儲相關的優化參數
    """

    # KV Cache
    kv_cache_policy: str = "lru"  # "lru", "lfu", "random", "fifo"
    kv_cache_max_size_mb: float = 4096.0
    enable_kv_quant: bool = True
    kv_quant_bits: int = 8  # 4, 8, 16
    kv_quant_type: str = "int8"  # "int4", "int8", "fp8", "bf16"
    enable_kv_compression: bool = False
    compression_block_size: int = 16

    # KDA (Kernel Direct Access)
    enable_kda: bool = False
    kda_bandwidth_target_gb: float = 500.0
    kda_prefetch_enabled: bool = True
    kda_alignment_bytes: int = 256

    # Prefetch
    enable_prefetch: bool = True
    prefetch_strategy: str = "distance_based"  # "distance_based", "confidence", "adaptive"
    prefetch_distance: int = 32
    prefetch_aggressive: bool = False
    prefetch_max_pending: int = 8

    # Memory Layout
    memory_layout: str = "flat"  # "flat", "paged", "tiled"
    page_size: int = 16
    tile_shape_m: int = 64
    tile_shape_n: int = 64
    enable_memory_pooling: bool = True
    memory_pool_size_mb: float = 2048.0

    # GDS (GPUDirect Storage)
    enable_gds: bool = False
    gds_chunk_size_mb: int = 1
    gds_prefetch_enabled: bool = False
    gds_fallback_on_error: bool = True

    # SPDK (Storage Performance Development Kit)
    enable_spdk: bool = False
    spdk_mem_pool_size_mb: int = 1024
    spdk_pci_bdf: Optional[str] = None  # e.g., "0000:01:00.0"
    spdk_io_depth: int = 32
    spdk_queue_depth: int = 64
    spdk_enable_kv_store: bool = False
    spdk_kv_store_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StorageStrategy":
        """从字典反序列化"""
        return cls(**d)

    def validate(self) -> bool:
        """验证策略配置是否合法"""
        if self.kv_cache_max_size_mb < 0:
            logger.error("KV cache max size must be non-negative")
            return False
        if self.kv_quant_bits not in [4, 8, 16]:
            logger.error(f"Invalid KV quant bits: {self.kv_quant_bits}")
            return False
        return True

    def merge(self, other: "StorageStrategy") -> None:
        """合并另一个策略的配置"""
        for key, value in asdict(other).items():
            if value is not None:
                setattr(self, key, value)

    # Eviction
    eviction_policy: str = "lru"  # "lru", "random", "size_based"
    enable_predictive_eviction: bool = False

    # Metadata
    source_feedback_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    tuning_level: str = StorageTuningLevel.MODERATE.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kv_cache": {
                "policy": self.kv_cache_policy,
                "max_size_mb": self.kv_cache_max_size_mb,
                "quant_enabled": self.enable_kv_quant,
                "quant_bits": self.kv_quant_bits,
                "quant_type": self.kv_quant_type,
            },
            "kda": {
                "enabled": self.enable_kda,
                "bandwidth_target_gb": self.kda_bandwidth_target_gb,
                "prefetch_enabled": self.kda_prefetch_enabled,
            },
            "prefetch": {
                "enabled": self.enable_prefetch,
                "strategy": self.prefetch_strategy,
                "distance": self.prefetch_distance,
                "aggressive": self.prefetch_aggressive,
            },
            "memory_layout": {
                "layout": self.memory_layout,
                "page_size": self.page_size,
                "tile_shape": f"{self.tile_shape_m}x{self.tile_shape_n}",
            },
            "gds": {
                "enabled": self.enable_gds,
                "chunk_size_mb": self.gds_chunk_size_mb,
                "prefetch_enabled": self.gds_prefetch_enabled,
                "fallback_on_error": self.gds_fallback_on_error,
            },
            "spdk": {
                "enabled": self.enable_spdk,
                "mem_pool_size_mb": self.spdk_mem_pool_size_mb,
                "pci_bdf": self.spdk_pci_bdf,
                "io_depth": self.spdk_io_depth,
                "queue_depth": self.spdk_queue_depth,
                "enable_kv_store": self.spdk_enable_kv_store,
                "kv_store_path": self.spdk_kv_store_path,
            },
            "source_feedback_score": self.source_feedback_score,
            "tuning_level": self.tuning_level,
        }


class StorageOptimizer:
    """
    存儲層優化器

    從 llama.cpp/vLLM 的反饋中學習，生成優化的存儲策略
    """

    def __init__(
        self,
        tuning_level: StorageTuningLevel = StorageTuningLevel.MODERATE
    ):
        self.tuning_level = tuning_level
        self.strategy_history: List[StorageStrategy] = []
        self.feedback_history: List["StorageFeedback"] = []

    def learn(
        self,
        feedback: "StorageFeedback",
        current_strategy: Optional[StorageStrategy] = None
    ) -> StorageStrategy:
        """
        從反饋中學習，生成優化的策略

        Args:
            feedback: 從 llama.cpp/vLLM 收集的反饋
            current_strategy: 當前的策略（可選，用於增量優化）

        Returns:
            優化後的策略
        """
        logger.info(f"[StorageOptimizer] Learning from {feedback.engine} feedback")

        if current_strategy is None:
            current_strategy = StorageStrategy()

        # 根據 tuning level 應用不同的學習策略
        if self.tuning_level == StorageTuningLevel.CONSERVATIVE:
            new_strategy = self._learn_conservative(feedback, current_strategy)
        elif self.tuning_level == StorageTuningLevel.MODERATE:
            new_strategy = self._learn_moderate(feedback, current_strategy)
        else:
            new_strategy = self._learn_aggressive(feedback, current_strategy)

        # 記錄歷史
        self.strategy_history.append(new_strategy)
        self.feedback_history.append(feedback)

        logger.info(
            f"[StorageOptimizer] Generated strategy with "
            f"score={new_strategy.source_feedback_score:.2f}"
        )

        return new_strategy

    def _create_baseline_strategy(self) -> StorageStrategy:
        """創建基線策略"""
        return StorageStrategy(
            kv_cache_policy="lru",
            kv_cache_max_size_mb=4096.0,
            enable_kv_quant=True,
            kv_quant_bits=8,
            enable_kda=False,
            enable_prefetch=True,
            prefetch_distance=32,
            memory_layout="flat",
            page_size=16,
            enable_gds=False,
        )

    def _learn_conservative(
        self,
        feedback: "StorageFeedback",
        current: StorageStrategy
    ) -> StorageStrategy:
        """保守學習：只在明確更好時才做改變"""
        metrics = feedback.metrics

        new_strategy = StorageStrategy(
            kv_cache_policy=current.kv_cache_policy,
            kv_cache_max_size_mb=current.kv_cache_max_size_mb,
            enable_kv_quant=current.enable_kv_quant,
            kv_quant_bits=current.kv_quant_bits,
            kv_quant_type=current.kv_quant_type,
            enable_kda=current.enable_kda,
            enable_prefetch=current.enable_prefetch,
            prefetch_distance=current.prefetch_distance,
            memory_layout=current.memory_layout,
            page_size=current.page_size,
            enable_gds=current.enable_gds,
            tuning_level=StorageTuningLevel.CONSERVATIVE.value,
        )

        # 只有當 vLLM/PagedAttention 明確更好時才採用
        if metrics.kv_cache.hit_rate > 0.9:
            new_strategy.memory_layout = "paged"
            new_strategy.page_size = metrics.kv_cache.block_size

        # KDA - 保守採用
        if metrics.kda.enabled and metrics.kda.bandwidth_gb_per_sec > 400:
            new_strategy.enable_kda = True
            new_strategy.kda_bandwidth_target_gb = metrics.kda.bandwidth_gb_per_sec

        # Quantization - 保持當前
        if metrics.kv_cache.quantization_enabled:
            new_strategy.enable_kv_quant = True

        new_strategy.source_feedback_score = feedback.get_efficiency_score()
        return new_strategy

    def _learn_moderate(
        self,
        feedback: "StorageFeedback",
        current: StorageStrategy
    ) -> StorageStrategy:
        """中等學習：允許適度的優化"""
        metrics = feedback.metrics

        new_strategy = StorageStrategy(
            kv_cache_policy="lru",  # 默認 LRU
            kv_cache_max_size_mb=min(8192, max(2048, metrics.kv_cache.cache_size_mb * 1.2)),
            enable_kv_quant=metrics.kv_cache.quantization_enabled or current.enable_kv_quant,
            kv_quant_bits=metrics.kv_cache.quant_bits or current.kv_quant_bits,
            kv_quant_type=metrics.kv_cache.quant_type or current.kv_quant_type,
            enable_kda=metrics.kda.enabled or current.enable_kda,
            enable_prefetch=metrics.prefetch.enabled or current.enable_prefetch,
            prefetch_distance=max(16, min(64, int(metrics.prefetch.avg_latency_ms * 10))),
            memory_layout="paged" if metrics.memory.memory_layout == "paged" else current.memory_layout,
            page_size=metrics.kv_cache.block_size if metrics.kv_cache.block_size > 0 else current.page_size,
            enable_gds=current.enable_gds,
            tuning_level=StorageTuningLevel.MODERATE.value,
        )

        # KV Cache 策略
        if metrics.kv_cache.hit_rate > 0.8:
            new_strategy.kv_cache_policy = "lru"
        elif metrics.kv_cache.hit_rate > 0.6:
            new_strategy.kv_cache_policy = "lfu"

        # KDA 優化
        if metrics.kda.enabled:
            new_strategy.enable_kda = True
            new_strategy.kda_bandwidth_target_gb = metrics.kda.bandwidth_gb_per_sec
            if metrics.kda.prefetch_enabled:
                new_strategy.kda_prefetch_enabled = True

        # Prefetch 優化
        if metrics.prefetch.enabled:
            new_strategy.enable_prefetch = True
            new_strategy.prefetch_aggressive = metrics.prefetch.accuracy > 0.8

        # Memory Layout
        if metrics.memory.memory_layout == "paged":
            new_strategy.memory_layout = "paged"
            new_strategy.page_size = metrics.kv_cache.block_size if metrics.kv_cache.block_size > 0 else 16

        new_strategy.source_feedback_score = feedback.get_efficiency_score()
        return new_strategy

    def _learn_aggressive(
        self,
        feedback: "StorageFeedback",
        current: StorageStrategy
    ) -> StorageStrategy:
        """激進學習：追求最大性能"""
        metrics = feedback.metrics

        new_strategy = StorageStrategy(
            kv_cache_policy="lru",
            kv_cache_max_size_mb=min(16384, metrics.kv_cache.cache_size_mb * 2),
            enable_kv_quant=True,
            kv_quant_bits=4 if metrics.kv_cache.cache_size_mb > 2048 else 8,
            kv_quant_type="int4" if metrics.kv_cache.cache_size_mb > 2048 else "int8",
            enable_kda=True,
            kda_bandwidth_target_gb=metrics.kda.bandwidth_gb_per_sec * 1.2 if metrics.kda.enabled else 600.0,
            kda_prefetch_enabled=True,
            enable_prefetch=True,
            prefetch_distance=max(8, min(64, int(metrics.prefetch.avg_latency_ms * 5))),
            prefetch_aggressive=True,
            memory_layout="paged",
            page_size=16,
            enable_gds=metrics.gds_enabled or current.enable_gds,
            tuning_level=StorageTuningLevel.AGGRESSIVE.value,
        )

        # KV Cache - 最大化
        new_strategy.kv_cache_max_size_mb = min(
            16384,
            metrics.kv_cache.cache_capacity_mb * 0.9 if metrics.kv_cache.cache_capacity_mb > 0 else 8192
        )

        # Quantization - 使用更高的壓縮
        if metrics.kv_cache.quantization_enabled:
            if metrics.kv_cache.compression_ratio > 2.5:
                new_strategy.kv_quant_bits = 4
                new_strategy.kv_quant_type = "int4"
            else:
                new_strategy.kv_quant_bits = 8
                new_strategy.kv_quant_type = "int8"
        else:
            # 即使沒有量化反饋，也啟用並使用 INT8
            new_strategy.enable_kv_quant = True
            new_strategy.kv_quant_bits = 8
            new_strategy.kv_quant_type = "int8"

        # KDA - 啟用並最大化帶寬
        if metrics.kda.enabled:
            new_strategy.enable_kda = True
            new_strategy.kda_bandwidth_target_gb = metrics.kda.bandwidth_gb_per_sec * 1.5
            new_strategy.kda_prefetch_enabled = True

        # Prefetch - 激進
        if metrics.prefetch.enabled:
            new_strategy.enable_prefetch = True
            new_strategy.prefetch_aggressive = True
            new_strategy.prefetch_distance = max(8, int(metrics.prefetch.avg_latency_ms * 3))

        # Memory Layout - 採用 Paged
        new_strategy.memory_layout = "paged"
        new_strategy.page_size = metrics.kv_cache.block_size if metrics.kv_cache.block_size > 0 else 16
        new_strategy.enable_memory_pooling = True
        new_strategy.memory_pool_size_mb = min(
            4096, metrics.memory.total_memory_mb * 0.2
        ) if metrics.memory.total_memory_mb > 0 else 2048

        # Eviction
        new_strategy.eviction_policy = "lru"
        new_strategy.enable_predictive_eviction = metrics.prefetch.accuracy > 0.7

        new_strategy.source_feedback_score = feedback.get_efficiency_score()
        return new_strategy

    def learn_from_comparison(
        self,
        cgc_feedback: "StorageFeedback",
        reference_feedback: "StorageFeedback"
    ) -> StorageStrategy:
        """
        從 CGC 和參考引擎的對比中學習

        Args:
            cgc_feedback: CGC Engine 的反饋
            reference_feedback: llama.cpp/vLLM 的反饋

        Returns:
            優化後的策略
        """
        logger.info(
            f"[StorageOptimizer] Learning from comparison: "
            f"CGC (score={cgc_feedback.get_efficiency_score():.2f}) vs "
            f"{reference_feedback.engine} (score={reference_feedback.get_efficiency_score():.2f})"
        )

        # 採用參考引擎的策略
        new_strategy = self.learn(reference_feedback)

        # 根據 CGC 的差距調整
        cgc_score = cgc_feedback.get_efficiency_score()
        ref_score = reference_feedback.get_efficiency_score()

        if cgc_score < ref_score * 0.8:
            # CGC 效果明顯較差，採用更激進的策略
            logger.info("[StorageOptimizer] CGC is significantly worse, applying aggressive tuning")
            new_strategy = self._learn_aggressive(reference_feedback, new_strategy)
        elif cgc_score < ref_score * 0.9:
            # CGC 效果略差，採用中等策略
            logger.info("[StorageOptimizer] CGC is slightly worse, applying moderate tuning")
            new_strategy = self._learn_moderate(reference_feedback, new_strategy)

        return new_strategy

    def get_best_strategy(self) -> StorageStrategy:
        """
        從歷史中獲取最佳策略

        Returns:
            效果分數最高的策略
        """
        if not self.strategy_history:
            return StorageStrategy()

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
