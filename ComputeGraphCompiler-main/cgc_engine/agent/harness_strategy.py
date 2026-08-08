from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

class MagiBackendType(Enum):
    LLAMA_CPP = "llama.cpp"
    VLLM = "vLLM"
    MEGATRAIN_2026_4 = "MegaTrain2026.4"
    MLX_TUNE = "mlx-tune"

class MagiExecuteMode(Enum):
    INFER_PREFILL = "Prefill"
    INFER_DECODE = "Decode"
    TRAIN_FWD = "Train Forward"
    TRAIN_BWD = "Train Backward"
    LAYER_EXEC = "Layer Execution"
    TRAIN_GLOBAL = "Train Global"
    TUNE_LORA = "LoRA Tune"

@dataclass
class GraphCaptureStrategy:
    enabled: bool = True
    capture_mode: MagiExecuteMode = MagiExecuteMode.INFER_DECODE
    backend_strategies: Dict[MagiBackendType, bool] = field(default_factory=lambda: {
        MagiBackendType.LLAMA_CPP: True,
        MagiBackendType.VLLM: True,
        MagiBackendType.MEGATRAIN_2026_4: True,
        MagiBackendType.MLX_TUNE: True,
    })

    def enable_for_backend(self, backend: MagiBackendType):
        self.backend_strategies[backend] = True
        logger.info(f"[GraphCaptureStrategy] 已启用 {backend.value}")

    def disable_for_backend(self, backend: MagiBackendType):
        self.backend_strategies[backend] = False
        logger.info(f"[GraphCaptureStrategy] 已禁用 {backend.value}")

@dataclass
class CompileStrategy:
    enabled: bool = True
    full_graph_compile: bool = True
    layer_wise_compile: bool = False
    backend_strategies: Dict[MagiBackendType, bool] = field(default_factory=lambda: {
        MagiBackendType.LLAMA_CPP: True,
        MagiBackendType.VLLM: True,
        MagiBackendType.MEGATRAIN_2026_4: True,
        MagiBackendType.MLX_TUNE: True,
    })

    def enable_for_backend(self, backend: MagiBackendType, mode: MagiExecuteMode):
        self.backend_strategies[backend] = True
        if mode == MagiExecuteMode.LAYER_EXEC:
            self.layer_wise_compile = True
        logger.info(f"[CompileStrategy] 已启用 {backend.value} in {mode.value} mode")

@dataclass
class OptimizationStrategy:
    enabled: bool = True
    use_kda: bool = True
    use_flashmoe: bool = True
    use_omlx: bool = True
    auto_recompute: bool = True
    jit_offload: bool = True

@dataclass
class MemoryStrategy:
    enabled: bool = True
    compiler_as_manager: bool = True
    paged_attention_aware: bool = False
    unified_memory_optimization: bool = False
    low_memory_mode: bool = False

@dataclass
class DistributedStrategy:
    enabled: bool = False
    fsdp_aware: bool = False
    communication_overlap: bool = False

@dataclass
class ProfileStrategy:
    enabled: bool = True
    measure_latency: bool = True
    measure_memory: bool = True
    measure_flops: bool = True
    compare_native_vs_optimized: bool = True

@dataclass
class HarnessStrategy:
    graph_capture: GraphCaptureStrategy = field(default_factory=GraphCaptureStrategy)
    compile: CompileStrategy = field(default_factory=CompileStrategy)
    optimization: OptimizationStrategy = field(default_factory=OptimizationStrategy)
    memory: MemoryStrategy = field(default_factory=MemoryStrategy)
    distributed: DistributedStrategy = field(default_factory=DistributedStrategy)
    profile: ProfileStrategy = field(default_factory=ProfileStrategy)

    def enable_all(self):
        self.graph_capture.enabled = True
        self.compile.enabled = True
        self.optimization.enabled = True
        self.memory.enabled = True
        self.profile.enabled = True
        logger.info("[HarnessStrategy] 全部策略已启用")

    def disable_all(self):
        self.graph_capture.enabled = False
        self.compile.enabled = False
        self.optimization.enabled = False
        self.memory.enabled = False
        self.profile.enabled = False
        logger.info("[HarnessStrategy] 全部策略已禁用")

class StrategyDispatcher:
    """
    StrategyDispatcher - Sand.ai MagiCompiler 核心策略分发器

    6 大核心策略分发规则：

    1. 图捕获策略（Graph Capture Strategy）
       - llama.cpp：GGML-Viz 劫持 ggml_backend->compute() → 整图捕获
       - vLLM：Easy-Debug hook model.forward() → 整图捕获
       - MegaTrain2026.4：单层流式执行 → 按 Layer 整层捕获
       - mlx-tune：劫持 mlx::core::compute() → 整图捕获

    2. 编译策略（Compile Strategy）
       - 推理场景（llama.cpp / vLLM / mlx-tune）：整图编译
       - 训练场景（MegaTrain2026.4）：Transformer Layer 整层编译
       - 全部后端统一：无 Graph Break，稳定输出完整计算图

    3. 优化策略（Optimization Strategy）
       - Attention / MLP 自动识别
       - KDA / FlashMoE / OMLX 自动启用
       - 启发式自动重计算（显存密集型算子自动开启）
       - JIT Offload（权重冷热分级驻留内存）

    4. 内存管理策略（Memory Strategy）
       - Compiler as Manager：全局接管显存/内存生命周期
       - llama.cpp：低内存优化
       - vLLM：PagedAttention 感知优化
       - MegaTrain：单层流式权重预取/卸载
       - mlx-tune：Apple 统一内存（Unified Memory）优化

    5. 分布式策略（Distributed Strategy）
       - vLLM / MegaTrain：FSDP-Aware 通信内联计算图
       - llama.cpp / mlx-tune：单设备无分布式，关闭

    6. 性能统计策略（Profile Strategy）
       - 所有后端统一统计：执行耗时、内存/显存占用、FLOPs、优化收益
    """

    def __init__(self):
        self.strategy = HarnessStrategy()
        self.current_backend: Optional[MagiBackendType] = None
        self.current_mode: MagiExecuteMode = MagiExecuteMode.INFER_DECODE

    def dispatch(self, backend: MagiBackendType, mode: MagiExecuteMode) -> HarnessStrategy:
        """
        根据后端类型和执行模式分发策略

        Args:
            backend: 后端类型 (LLAMA_CPP / VLLM / MEGATRAIN_2026_4 / MLX_TUNE)
            mode: 执行模式 (INFER_PREFILL / INFER_DECODE / TRAIN_* / LAYER_EXEC / TUNE_LORA)

        Returns:
            配置好的 HarnessStrategy
        """
        self.current_backend = backend
        self.current_mode = mode

        if backend == MagiBackendType.LLAMA_CPP:
            return self._dispatch_llama_cpp(mode)
        elif backend == MagiBackendType.VLLM:
            return self._dispatch_vllm(mode)
        elif backend == MagiBackendType.MEGATRAIN_2026_4:
            return self._dispatch_megatrain(mode)
        elif backend == MagiBackendType.MLX_TUNE:
            return self._dispatch_mlx_tune(mode)
        else:
            logger.warning(f"[StrategyDispatcher] 未知后端类型: {backend}")
            return self.strategy

    def _dispatch_llama_cpp(self, mode: MagiExecuteMode) -> HarnessStrategy:
        """
        llama.cpp 策略分发规则：
        → 启用：整图编译 + 低内存管理 + GGML解析
        → 关闭：分布式
        """
        logger.info(f"[StrategyDispatcher] [LLAMA_CPP] 分发策略: mode={mode.value}")

        self.strategy.graph_capture.enable_for_backend(MagiBackendType.LLAMA_CPP)
        self.strategy.graph_capture.capture_mode = mode

        self.strategy.compile.enable_for_backend(MagiBackendType.LLAMA_CPP, mode)
        self.strategy.compile.full_graph_compile = True
        self.strategy.compile.layer_wise_compile = False

        self.strategy.memory.enabled = True
        self.strategy.memory.low_memory_mode = True
        self.strategy.memory.compiler_as_manager = True

        self.strategy.distributed.enabled = False
        self.strategy.distributed.fsdp_aware = False

        self.strategy.optimization.enabled = True
        self.strategy.optimization.use_kda = True
        self.strategy.optimization.auto_recompute = True
        self.strategy.optimization.jit_offload = True

        return self.strategy

    def _dispatch_vllm(self, mode: MagiExecuteMode) -> HarnessStrategy:
        """
        vLLM 策略分发规则：
        → 启用：整图编译 + FSDP + PagedAttention感知优化
        → 启用：JIT Offload
        """
        logger.info(f"[StrategyDispatcher] [VLLM] 分发策略: mode={mode.value}")

        self.strategy.graph_capture.enable_for_backend(MagiBackendType.VLLM)
        self.strategy.graph_capture.capture_mode = mode

        self.strategy.compile.enable_for_backend(MagiBackendType.VLLM, mode)
        self.strategy.compile.full_graph_compile = True
        self.strategy.compile.layer_wise_compile = False

        self.strategy.memory.enabled = True
        self.strategy.memory.paged_attention_aware = True
        self.strategy.memory.compiler_as_manager = True

        self.strategy.distributed.enabled = True
        self.strategy.distributed.fsdp_aware = True
        self.strategy.distributed.communication_overlap = True

        self.strategy.optimization.enabled = True
        self.strategy.optimization.use_kda = True
        self.strategy.optimization.use_flashmoe = True
        self.strategy.optimization.jit_offload = True
        self.strategy.optimization.auto_recompute = True

        return self.strategy

    def _dispatch_megatrain(self, mode: MagiExecuteMode) -> HarnessStrategy:
        """
        MegaTrain 2026.4 策略分发规则：
        → 启用：Layer-wise 整层编译
        → 启用：自动重计算
        → 启用：JIT Offload 权重预取/卸载
        → 启用：FSDP-Aware 分布式
        → 启用：单层流式调度
        """
        logger.info(f"[StrategyDispatcher] [MEGATRAIN_2026_4] 分发策略: mode={mode.value}")

        self.strategy.graph_capture.enable_for_backend(MagiBackendType.MEGATRAIN_2026_4)
        self.strategy.graph_capture.capture_mode = MagiExecuteMode.LAYER_EXEC

        self.strategy.compile.enable_for_backend(MagiBackendType.MEGATRAIN_2026_4, MagiExecuteMode.LAYER_EXEC)
        self.strategy.compile.full_graph_compile = False
        self.strategy.compile.layer_wise_compile = True

        self.strategy.memory.enabled = True
        self.strategy.memory.compiler_as_manager = True

        self.strategy.distributed.enabled = True
        self.strategy.distributed.fsdp_aware = True
        self.strategy.distributed.communication_overlap = True

        self.strategy.optimization.enabled = True
        self.strategy.optimization.use_kda = True
        self.strategy.optimization.use_flashmoe = True
        self.strategy.optimization.auto_recompute = True
        self.strategy.optimization.jit_offload = True

        return self.strategy

    def _dispatch_mlx_tune(self, mode: MagiExecuteMode) -> HarnessStrategy:
        """
        mlx-tune 策略分发规则：
        → 启用：整图编译
        → 启用：统一内存优化
        → 关闭：分布式
        → 端云一体：强制使用 INFER_DECODE（端侧不做 Prefill）
        """
        logger.info(f"[StrategyDispatcher] [MLX_TUNE] 分发策略: mode={mode.value}")

        self.strategy.graph_capture.enable_for_backend(MagiBackendType.MLX_TUNE)
        self.strategy.graph_capture.capture_mode = MagiExecuteMode.INFER_DECODE

        self.strategy.compile.enable_for_backend(MagiBackendType.MLX_TUNE, MagiExecuteMode.INFER_DECODE)
        self.strategy.compile.full_graph_compile = True
        self.strategy.compile.layer_wise_compile = False

        self.strategy.memory.enabled = True
        self.strategy.memory.unified_memory_optimization = True
        self.strategy.memory.compiler_as_manager = True

        self.strategy.distributed.enabled = False
        self.strategy.distributed.fsdp_aware = False

        self.strategy.optimization.enabled = True
        self.strategy.optimization.use_omlx = True
        self.strategy.optimization.jit_offload = False

        return self.strategy

    def get_strategy_summary(self) -> Dict[str, Any]:
        """获取策略摘要"""
        return {
            "current_backend": self.current_backend.value if self.current_backend else None,
            "current_mode": self.current_mode.value,
            "graph_capture": {
                "enabled": self.strategy.graph_capture.enabled,
                "capture_mode": self.strategy.graph_capture.capture_mode.value,
                "backends": [k.value for k, v in self.strategy.graph_capture.backend_strategies.items() if v]
            },
            "compile": {
                "enabled": self.strategy.compile.enabled,
                "full_graph": self.strategy.compile.full_graph_compile,
                "layer_wise": self.strategy.compile.layer_wise_compile,
            },
            "optimization": {
                "enabled": self.strategy.optimization.enabled,
                "kda": self.strategy.optimization.use_kda,
                "flashmoe": self.strategy.optimization.use_flashmoe,
                "omlx": self.strategy.optimization.use_omlx,
                "auto_recompute": self.strategy.optimization.auto_recompute,
                "jit_offload": self.strategy.optimization.jit_offload,
            },
            "memory": {
                "enabled": self.strategy.memory.enabled,
                "compiler_as_manager": self.strategy.memory.compiler_as_manager,
                "paged_attention_aware": self.strategy.memory.paged_attention_aware,
                "unified_memory": self.strategy.memory.unified_memory_optimization,
                "low_memory": self.strategy.memory.low_memory_mode,
            },
            "distributed": {
                "enabled": self.strategy.distributed.enabled,
                "fsdp_aware": self.strategy.distributed.fsdp_aware,
                "comm_overlap": self.strategy.distributed.communication_overlap,
            },
            "profile": {
                "enabled": self.strategy.profile.enabled,
                "compare_native_vs_optimized": self.strategy.profile.compare_native_vs_optimized,
            }
        }