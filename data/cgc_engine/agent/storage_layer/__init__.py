# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
Storage Layer - 存儲層反饋收集與優化

功能：
- StorageFeedback: 存儲層反饋結構
- StorageFeedbackCollector: 收集 llama.cpp/vLLM 存儲信息
- StorageOptimizer: 從反饋中學習存儲策略
- StorageMetrics: 存儲指標結構

使用方式：
    # 收集反饋
    from cgc_engine.agent.storage_layer import (
        StorageFeedbackCollector,
        StorageFeedback,
    )
    collector = StorageFeedbackCollector()
    feedback = collector.collect(engine="vllm")

    # 學習策略
    from cgc_engine.agent.storage_layer import StorageOptimizer
    optimizer = StorageOptimizer()
    strategy = optimizer.learn(feedback)
"""

from .storage_feedback import (
    StorageFeedback,
    StorageFeedbackCollector,
    StorageMetrics,
    KVCacheMetrics,
    KDAMetrics,
    PrefetchMetrics,
    MemoryMetrics,
    MemoryLayout,
    CachePolicy,
    QuantizationType,
)
from .storage_optimizer import (
    StorageOptimizer,
    StorageStrategy,
    StorageTuningLevel,
)

__all__ = [
    # Feedback
    "StorageFeedback",
    "StorageFeedbackCollector",
    "StorageMetrics",
    "KVCacheMetrics",
    "KDAMetrics",
    "PrefetchMetrics",
    "MemoryMetrics",
    "MemoryLayout",
    "CachePolicy",
    "QuantizationType",
    # Optimizer
    "StorageOptimizer",
    "StorageStrategy",
    "StorageTuningLevel",
]
