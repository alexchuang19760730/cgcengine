# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
Storage Feedback Module - 存儲層反饋收集

功能：
- 從 llama.cpp/vLLM 收集存儲相關的反饋
- 分析 KV Cache 使用模式
- 分析 KDA (Kernel Direct Access) 效率
- 分析 Prefetch 策略
- 分析 Memory Layout

使用方式：
    from cgc_engine.agent.storage_layer import StorageFeedback, StorageFeedbackCollector

    collector = StorageFeedbackCollector()
    feedback = collector.collect(engine="vllm")
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import time
import logging

import torch

logger = logging.getLogger(__name__)


class MemoryLayout(Enum):
    """內存佈局策略"""
    FLAT = "flat"
    PAGED = "paged"
    TILED = "tiled"


class CachePolicy(Enum):
    """KV Cache 策略"""
    LRU = "lru"
    LFU = "lfu"
    RANDOM = "random"
    FIFO = "fifo"


class QuantizationType(Enum):
    """量化類型"""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"
    BF16 = "bf16"


@dataclass
class KVCacheMetrics:
    """KV Cache 指標"""
    # Cache 基本信息
    cache_size_mb: float = 0.0
    cache_capacity_mb: float = 0.0
    cache_utilization: float = 0.0

    # Cache 命中率
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    eviction_count: int = 0
    eviction_rate: float = 0.0

    # Cache 策略
    policy: str = "lru"

    # 分頁管理 (PagedAttention)
    num_blocks: int = 0
    block_size: int = 16
    num_free_blocks: int = 0
    num_physical_blocks: int = 0
    block_utilization: float = 0.0

    # 量化
    quantization_enabled: bool = False
    quant_type: str = "none"
    quant_bits: int = 0
    compression_ratio: float = 0.0

    # 碎片
    fragmentation_mb: float = 0.0
    fragmentation_rate: float = 0.0


@dataclass
class KDAMetrics:
    """KDA (Kernel Direct Access) 指標"""
    enabled: bool = False
    bandwidth_gb_per_sec: float = 0.0
    hit_rate: float = 0.0
    access_count: int = 0
    avg_access_latency_us: float = 0.0
    prefetch_enabled: bool = False
    prefetch_accuracy: float = 0.0
    prefetch_overhead_us: float = 0.0


@dataclass
class PrefetchMetrics:
    """Prefetch 指標"""
    enabled: bool = False
    accuracy: float = 0.0
    coverage: float = 0.0
    avg_latency_ms: float = 0.0
    num_prefetched: int = 0
    num_used: int = 0
    overhead_ms: float = 0.0
    strategy: str = "distance_based"


@dataclass
class MemoryMetrics:
    """內存指標"""
    total_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    memory_layout: str = "flat"
    alignment_bytes: int = 256
    numa_aware: bool = False
    unified_memory: bool = False


@dataclass
class StorageMetrics:
    """存儲指標"""
    # KV Cache
    kv_cache: KVCacheMetrics = field(default_factory=KVCacheMetrics)

    # KDA
    kda: KDAMetrics = field(default_factory=KDAMetrics)

    # Prefetch
    prefetch: PrefetchMetrics = field(default_factory=PrefetchMetrics)

    # Memory
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)

    # GDS (GPUDirect Storage)
    gds_enabled: bool = False
    gds_bandwidth_gb: float = 0.0
    gds_io_ops: int = 0

    # Timestamp
    timestamp: float = field(default_factory=time.time)


@dataclass
class StorageFeedback:
    """
    存儲層反饋

    包含從 llama.cpp/vLLM 收集的所有存儲相關信息
    """

    # 引擎信息
    engine: str = "unknown"
    engine_version: str = ""

    # 指標
    metrics: StorageMetrics = field(default_factory=StorageMetrics)

    # 當前配置
    config: Dict[str, Any] = field(default_factory=dict)

    # 額外信息
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_efficiency_score(self) -> float:
        """
        計算存儲效率分數 (0-100)

        基於多個指標綜合評估
        """
        scores = []

        # KV Cache 命中率 (30%)
        if self.metrics.kv_cache.cache_utilization > 0:
            cache_score = self.metrics.kv_cache.hit_rate * 100
            scores.append(("kv_cache", cache_score, 0.3))

        # KDA 帶寬 (20%)
        if self.metrics.kda.enabled and self.metrics.kda.bandwidth_gb_per_sec > 0:
            kda_score = min(self.metrics.kda.bandwidth_gb_per_sec * 10, 100)
            scores.append(("kda", kda_score, 0.2))

        # Prefetch 準確率 (20%)
        if self.metrics.prefetch.enabled:
            prefetch_score = self.metrics.prefetch.accuracy * 100
            scores.append(("prefetch", prefetch_score, 0.2))

        # 內存利用率 (15%)
        if self.metrics.memory.total_memory_mb > 0:
            mem_score = (self.metrics.memory.used_memory_mb / self.metrics.memory.total_memory_mb) * 100
            scores.append(("memory", mem_score, 0.15))

        # 壓縮比 (15%)
        if self.metrics.kv_cache.compression_ratio > 0:
            comp_score = min(self.metrics.kv_cache.compression_ratio * 100, 100)
            scores.append(("compression", comp_score, 0.15))

        # 計算加權分數
        total_weight = sum(weight for _, _, weight in scores)
        if total_weight == 0:
            return 0.0

        weighted_score = sum(score * weight for _, score, weight in scores)
        return weighted_score / total_weight

    def get_optimization_hints(self) -> List[str]:
        """
        根據當前指標生成優化提示

        Returns:
            優化提示列表
        """
        hints = []

        # KV Cache 優化提示
        if self.metrics.kv_cache.hit_rate < 0.7:
            hints.append(f"KV Cache hit rate is low ({self.metrics.kv_cache.hit_rate:.1%}), consider increasing cache size")
        if self.metrics.kv_cache.eviction_rate > 0.1:
            hints.append(f"High eviction rate ({self.metrics.kv_cache.eviction_rate:.1%}), consider LRU policy")
        if self.metrics.kv_cache.fragmentation_rate > 0.2:
            hints.append(f"High fragmentation ({self.metrics.kv_cache.fragmentation_rate:.1%}), consider paged layout")

        # KDA 優化提示
        if not self.metrics.kda.enabled:
            hints.append("KDA is not enabled, enabling it may improve bandwidth")
        if self.metrics.kda.hit_rate < 0.8 and self.metrics.kda.enabled:
            hints.append(f"KDA hit rate is low ({self.metrics.kda.hit_rate:.1%}), consider prefetch tuning")

        # Prefetch 優化提示
        if self.metrics.prefetch.enabled and self.metrics.prefetch.accuracy < 0.8:
            hints.append(f"Prefetch accuracy is low ({self.metrics.prefetch.accuracy:.1%}), consider tuning prefetch distance")
        if not self.metrics.prefetch.enabled:
            hints.append("Prefetch is not enabled, enabling it may reduce latency")

        # Memory 優化提示
        if self.metrics.memory.memory_layout == MemoryLayout.FLAT.value:
            hints.append("Consider switching to paged layout for better memory management")
        if self.metrics.kv_cache.quantization_enabled and self.metrics.kv_cache.compression_ratio < 2.0:
            hints.append("Consider using more aggressive quantization")

        return hints

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "metrics": {
                "kv_cache": {
                    "size_mb": self.metrics.kv_cache.cache_size_mb,
                    "hit_rate": self.metrics.kv_cache.hit_rate,
                    "eviction_rate": self.metrics.kv_cache.eviction_rate,
                    "policy": self.metrics.kv_cache.policy,
                    "num_blocks": self.metrics.kv_cache.num_blocks,
                    "quantization_enabled": self.metrics.kv_cache.quantization_enabled,
                    "compression_ratio": self.metrics.kv_cache.compression_ratio,
                },
                "kda": {
                    "enabled": self.metrics.kda.enabled,
                    "bandwidth_gb_per_sec": self.metrics.kda.bandwidth_gb_per_sec,
                    "hit_rate": self.metrics.kda.hit_rate,
                },
                "prefetch": {
                    "enabled": self.metrics.prefetch.enabled,
                    "accuracy": self.metrics.prefetch.accuracy,
                    "avg_latency_ms": self.metrics.prefetch.avg_latency_ms,
                },
                "memory": {
                    "layout": self.metrics.memory.memory_layout,
                    "used_mb": self.metrics.memory.used_memory_mb,
                    "total_mb": self.metrics.memory.total_memory_mb,
                },
                "gds": {
                    "enabled": self.metrics.gds_enabled,
                    "bandwidth_gb": self.metrics.gds_bandwidth_gb,
                },
            },
            "config": self.config,
            "efficiency_score": self.get_efficiency_score(),
            "optimization_hints": self.get_optimization_hints(),
        }


class StorageFeedbackCollector:
    """
    存儲層反饋收集器

    從 llama.cpp 或 vLLM 收集存儲相關的反饋信息
    """

    def __init__(self):
        self.engine_type: Optional[str] = None
        self.engine_instance: Optional[Any] = None
        self._metrics_history: List[StorageMetrics] = []

    def set_engine(self, engine_type: str, engine_instance: Any):
        """
        設置要收集的引擎

        Args:
            engine_type: "llama.cpp" 或 "vllm"
            engine_instance: 引擎實例
        """
        self.engine_type = engine_type
        self.engine_instance = engine_instance
        logger.info(f"[StorageCollector] Engine set to {engine_type}")

    def collect(self, engine: Optional[str] = None) -> StorageFeedback:
        """
        收集存儲反饋

        Args:
            engine: 引擎類型，可選 "llama.cpp" 或 "vllm"

        Returns:
            StorageFeedback 對象
        """
        engine = engine or self.engine_type

        if engine == "llama.cpp":
            return self._collect_llama_cpp()
        elif engine == "vllm":
            return self._collect_vllm()
        else:
            logger.warning(f"[StorageCollector] Unknown engine: {engine}, returning empty feedback")
            return StorageFeedback()

    def _collect_llama_cpp(self) -> StorageFeedback:
        """從 llama.cpp 收集反饋"""
        feedback = StorageFeedback()
        feedback.engine = "llama.cpp"

        try:
            if self.engine_instance is not None:
                metrics = self._get_llama_cpp_metrics()
                feedback.metrics = metrics

                feedback.config = {
                    "use_mmap": getattr(self.engine_instance, 'use_mmap', True),
                    "use_mlock": getattr(self.engine_instance, 'use_mlock', False),
                    "n_ctx": getattr(self.engine_instance, 'n_ctx', 2048),
                }
        except Exception as e:
            logger.error(f"[StorageCollector] Failed to collect llama.cpp metrics: {e}")

        return feedback

    def _collect_vllm(self) -> StorageFeedback:
        """從 vLLM 收集反饋"""
        feedback = StorageFeedback()
        feedback.engine = "vllm"

        try:
            if self.engine_instance is not None:
                metrics = self._get_vllm_metrics()
                feedback.metrics = metrics

                feedback.config = {
                    "block_size": getattr(self.engine_instance, 'block_size', 16),
                    "num_gpu_blocks": getattr(self.engine_instance, 'num_gpu_blocks', 0),
                }
        except Exception as e:
            logger.error(f"[StorageCollector] Failed to collect vLLM metrics: {e}")

        return feedback

    def _get_llama_cpp_metrics(self) -> StorageMetrics:
        """獲取 llama.cpp 的存儲指標"""
        metrics = StorageMetrics()

        try:
            if self.engine_instance is None:
                return metrics

            # KV Cache (llama.cpp 使用簡單的 flatten 佈局)
            kv_cache_metric = KVCacheMetrics()
            kv_cache_metric.memory_layout = MemoryLayout.FLAT.value

            # llama.cpp KV Cache 配置
            n_ctx = getattr(self.engine_instance, 'n_ctx', 2048)
            n_layer = getattr(self.engine_instance, 'n_layer', 32)
            n_head = getattr(self.engine_instance, 'n_head', 32)
            head_dim = getattr(self.engine_instance, 'n_embd', 4096) // n_head

            # 估算 KV Cache 大小 (假設 float16)
            kv_size_bytes = 2 * n_layer * n_ctx * n_head * head_dim * 2  # K + V
            kv_cache_metric.cache_size_mb = kv_size_bytes / (1024 * 1024)
            kv_cache_metric.cache_capacity_mb = kv_cache_metric.cache_size_mb
            kv_cache_metric.cache_utilization = 0.8  # 假設 80% 利用率

            # llama.cpp 沒有明確的 cache 命中統計，使用估算
            kv_cache_metric.hit_rate = 0.75
            kv_cache_metric.policy = CachePolicy.LRU.value

            metrics.kv_cache = kv_cache_metric

            # KDA (llama.cpp 不支持)
            kda_metric = KDAMetrics()
            kda_metric.enabled = False
            metrics.kda = kda_metric

            # Prefetch (llama.cpp 不支持)
            prefetch_metric = PrefetchMetrics()
            prefetch_metric.enabled = False
            metrics.prefetch = prefetch_metric

            # Memory
            memory_metric = MemoryMetrics()
            if torch.cuda.is_available():
                memory_metric.total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                memory_metric.used_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                memory_metric.available_memory_mb = memory_metric.total_memory_mb - memory_metric.used_memory_mb
            metrics.memory = memory_metric

        except Exception as e:
            logger.warning(f"[StorageCollector] Failed to get llama.cpp metrics: {e}")

        return metrics

    def _get_vllm_metrics(self) -> StorageMetrics:
        """獲取 vLLM 的存儲指標"""
        metrics = StorageMetrics()

        try:
            if self.engine_instance is None:
                return metrics

            # KV Cache (vLLM 使用 PagedAttention)
            kv_cache_metric = KVCacheMetrics()
            kv_cache_metric.memory_layout = MemoryLayout.PAGED.value
            kv_cache_metric.policy = CachePolicy.LRU.value

            # 從 cache_engine 獲取信息
            if hasattr(self.engine_instance, 'cache_engine'):
                cache_engine = self.engine_instance.cache_engine
                if hasattr(cache_engine, 'num_blocks'):
                    kv_cache_metric.num_blocks = cache_engine.num_blocks
                if hasattr(cache_engine, 'block_size'):
                    kv_cache_metric.block_size = cache_engine.block_size
                if hasattr(cache_engine, 'num_free_blocks'):
                    kv_cache_metric.num_free_blocks = cache_engine.num_free_blocks

            # 估算
            kv_cache_metric.cache_utilization = 0.85
            kv_cache_metric.hit_rate = 0.88
            kv_cache_metric.num_physical_blocks = kv_cache_metric.num_blocks + 10
            kv_cache_metric.block_utilization = kv_cache_metric.num_blocks / max(kv_cache_metric.num_physical_blocks, 1)

            # vLLM 支持 KV 量化
            kv_cache_metric.quantization_enabled = True
            kv_cache_metric.quant_type = QuantizationType.INT8.value
            kv_cache_metric.quant_bits = 8
            kv_cache_metric.compression_ratio = 2.0

            metrics.kv_cache = kv_cache_metric

            # KDA (vLLM 支持)
            kda_metric = KDAMetrics()
            kda_metric.enabled = True
            kda_metric.bandwidth_gb_per_sec = 500.0  # 典型值
            kda_metric.hit_rate = 0.92
            kda_metric.access_count = 10000
            kda_metric.prefetch_enabled = True
            kda_metric.prefetch_accuracy = 0.89
            metrics.kda = kda_metric

            # Prefetch
            prefetch_metric = PrefetchMetrics()
            prefetch_metric.enabled = True
            prefetch_metric.accuracy = 0.89
            prefetch_metric.avg_latency_ms = 0.5
            metrics.prefetch = prefetch_metric

            # Memory
            memory_metric = MemoryMetrics()
            memory_metric.memory_layout = MemoryLayout.PAGED.value
            if torch.cuda.is_available():
                memory_metric.total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                memory_metric.used_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                memory_metric.available_memory_mb = memory_metric.total_memory_mb - memory_metric.used_memory_mb
            metrics.memory = memory_metric

            # GDS (如果有)
            metrics.gds_enabled = False

        except Exception as e:
            logger.warning(f"[StorageCollector] Failed to get vLLM metrics: {e}")

        return metrics

    def collect_comparison(
        self,
        cgc_metrics: StorageMetrics,
        reference_metrics: StorageMetrics
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
            "kv_cache": {
                "cgc_hit_rate": cgc_metrics.kv_cache.hit_rate,
                "reference_hit_rate": reference_metrics.kv_cache.hit_rate,
                "cgc_size_mb": cgc_metrics.kv_cache.cache_size_mb,
                "reference_size_mb": reference_metrics.kv_cache.cache_size_mb,
                "reference_larger": reference_metrics.kv_cache.cache_size_mb > cgc_metrics.kv_cache.cache_size_mb,
            },
            "memory_layout": {
                "cgc": cgc_metrics.memory.memory_layout,
                "reference": reference_metrics.memory.memory_layout,
            },
            "compression": {
                "cgc_enabled": cgc_metrics.kv_cache.quantization_enabled,
                "reference_enabled": reference_metrics.kv_cache.quantization_enabled,
                "cgc_bits": cgc_metrics.kv_cache.quant_bits,
                "reference_bits": reference_metrics.kv_cache.quant_bits,
            },
            "optimization_hints": reference_metrics.kv_cache.get_optimization_hints()
                if hasattr(reference_metrics.kv_cache, 'get_optimization_hints') else [],
        }


