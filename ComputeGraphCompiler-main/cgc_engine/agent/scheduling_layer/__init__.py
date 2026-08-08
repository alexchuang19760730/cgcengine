# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
Scheduling Layer - 調度層反饋收集與優化

功能：
- SchedulingFeedback: 調度層反饋結構
- SchedulingFeedbackCollector: 收集 llama.cpp/vLLM 調度信息
- SchedulerOptimizer: 從反饋中學習調度策略
- SchedulingMetrics: 調度指標結構

使用方式：
    # 收集反饋
    from cgc_engine.agent.scheduling_layer import (
        SchedulingFeedbackCollector,
        SchedulingFeedback,
    )
    collector = SchedulingFeedbackCollector()
    feedback = collector.collect(engine="vllm")

    # 學習策略
    from cgc_engine.agent.scheduling_layer import SchedulerOptimizer
    optimizer = SchedulerOptimizer()
    strategy = optimizer.learn(feedback)
"""

from .scheduling_feedback import (
    SchedulingFeedback,
    SchedulingFeedbackCollector,
    SchedulingMetrics,
    BatchStrategy,
    PDPhase,
    MockSchedulingCollector,
)
from .scheduler_optimizer import (
    SchedulerOptimizer,
    SchedulerStrategy,
    SchedulerTuningLevel,
    BatchStrategyType,
)

__all__ = [
    # Feedback
    "SchedulingFeedback",
    "SchedulingFeedbackCollector",
    "SchedulingMetrics",
    "BatchStrategy",
    "PDPhase",
    "MockSchedulingCollector",
    # Optimizer
    "SchedulerOptimizer",
    "SchedulerStrategy",
    "SchedulerTuningLevel",
    "BatchStrategyType",
]
