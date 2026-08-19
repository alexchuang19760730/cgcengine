# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
Harness Agent 模块 - 智能编译优化代理

功能：
- 自动分析 PyTorch 计算图
- 生成优化策略（算子融合、Tiling、内存布局、调度）
- 注入策略到 CGC SIMD 引擎
- 支持三層一體優化（計算層 + 調度層 + 存儲層）
- 提供顶层策略聚合器 MagiCompilerStrategy

使用方式：
    # 顶层策略聚合器（推荐）
    from cgc_engine.agent import MagiCompilerStrategy
    
    strategy = MagiCompilerStrategy()
    strategy.optimize_for("training")  # 或 "inference", "video_gen", "moe", "mlx_finetune"
    
    # 计算层
    from cgc_engine.agent import (
        HarnessAgent,
        HarnessCompileStrategy,
        OptimizationSpaceBuilder,
        GraphAnalyzer,
        StrategyExecutor,
    )

    # 调度层
    from cgc_engine.agent.scheduling_layer import (
        SchedulingFeedbackCollector,
        SchedulingFeedback,
    )

    # 存储层
    from cgc_engine.agent.storage_layer import (
        StorageFeedbackCollector,
        StorageFeedback,
    )
"""

try:
    from .harness_agent import HarnessAgent, HarnessCompileStrategy, AgentOpHint
    from .space_builder import OptimizationSpaceBuilder, OptimizationSpace
    from .graph_analyzer import GraphAnalyzer, GraphFeatures
    from .strategy_executor import StrategyExecutor
    from .backend_dispatcher import BackendDispatcher
    from .strategy_aggregator import MagiCompilerStrategy
    from .megatrain_graph_capture import MegatrainGraphCapture, MegatrainGraphCaptureConfig
    from .mlx_tune_graph_capture import MLXTuneGraphCapture, MLXTuneGraphCaptureConfig

    # 調度層
    from .scheduling_layer import (
        SchedulingFeedback,
        SchedulingFeedbackCollector,
        SchedulingMetrics,
        BatchStrategy,
        PDPhase,
        SchedulerOptimizer,
        SchedulerStrategy,
        SchedulerTuningLevel,
    )

    # 存儲層
    from .storage_layer import (
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
        StorageOptimizer,
        StorageStrategy,
        StorageTuningLevel,
    )
except Exception:  # pragma: no cover - torch 缺失等环境下降级为 None
    # 无 torch（如仅做 GGUF 静态分析 / monkey patch）时全部置 None，
    # 保证 `import cgc_engine.agent` 不抛错。
    HarnessAgent = None
    HarnessCompileStrategy = None
    AgentOpHint = None
    OptimizationSpaceBuilder = None
    OptimizationSpace = None
    GraphAnalyzer = None
    GraphFeatures = None
    StrategyExecutor = None
    BackendDispatcher = None
    MagiCompilerStrategy = None
    MegatrainGraphCapture = None
    MegatrainGraphCaptureConfig = None
    MLXTuneGraphCapture = None
    MLXTuneGraphCaptureConfig = None
    SchedulingFeedback = None
    SchedulingFeedbackCollector = None
    SchedulingMetrics = None
    BatchStrategy = None
    PDPhase = None
    SchedulerOptimizer = None
    SchedulerStrategy = None
    SchedulerTuningLevel = None
    StorageFeedback = None
    StorageFeedbackCollector = None
    StorageMetrics = None
    KVCacheMetrics = None
    KDAMetrics = None
    PrefetchMetrics = None
    MemoryMetrics = None
    MemoryLayout = None
    CachePolicy = None
    QuantizationType = None
    StorageOptimizer = None
    StorageStrategy = None
    StorageTuningLevel = None

__all__ = [
    # 顶层策略聚合器
    "MagiCompilerStrategy",
    # 計算層
    "HarnessAgent",
    "HarnessCompileStrategy",
    "AgentOpHint",
    "OptimizationSpaceBuilder",
    "OptimizationSpace",
    "GraphAnalyzer",
    "GraphFeatures",
    "StrategyExecutor",
    "BackendDispatcher",
    # 整图捕获
    "MegatrainGraphCapture",
    "MegatrainGraphCaptureConfig",
    "MLXTuneGraphCapture",
    "MLXTuneGraphCaptureConfig",
    # 調度層
    "SchedulingFeedback",
    "SchedulingFeedbackCollector",
    "SchedulingMetrics",
    "BatchStrategy",
    "PDPhase",
    "SchedulerOptimizer",
    "SchedulerStrategy",
    "SchedulerTuningLevel",
    # 存儲層
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
    "StorageOptimizer",
    "StorageStrategy",
    "StorageTuningLevel",
]
