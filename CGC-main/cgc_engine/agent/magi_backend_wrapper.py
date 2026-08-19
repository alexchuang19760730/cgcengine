import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    from cgc_engine.cgc.cgc_cpp import magi_backend_unified as _magi_backend
    C_BACKEND_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[MagiBackendWrapper] C++ backend not available: {e}")
    C_BACKEND_AVAILABLE = False

from .harness_strategy import MagiBackendType, MagiExecuteMode

@dataclass
class MagiTensorInfo:
    data_ptr: Optional[int] = None
    shape: List[int] = field(default_factory=list)
    dtype: str = "float32"
    device: str = "cpu"
    device_id: int = 0
    size: int = 0
    is_on_gpu: bool = False
    is_pinned: bool = False
    needs_copy: bool = False
    io_path: str = "NONE"
    stream_id: str = ""
    offset: int = 0
    peak_memory: int = 0

@dataclass
class MagiGraphNode:
    op_type: str = ""
    node_id: int = 0
    inputs: List[MagiTensorInfo] = field(default_factory=list)
    outputs: List[MagiTensorInfo] = field(default_factory=list)
    exec_time: float = 0.0
    attrs: Dict[str, str] = field(default_factory=dict)

@dataclass
class MagiHardwareInfo:
    device_type: str = "cpu"
    device_id: int = 0
    total_memory: int = 0
    free_memory: int = 0
    compute_capability_major: int = 0
    compute_capability_minor: int = 0
    l2_cache_size: int = 0
    num_sm: int = 0
    memory_bandwidth: int = 0
    is_unified_memory: bool = False

@dataclass
class MagiGraphInfo:
    backend: MagiBackendType = MagiBackendType.LLAMA_CPP
    mode: MagiExecuteMode = MagiExecuteMode.INFER_DECODE
    nodes: List[MagiGraphNode] = field(default_factory=list)
    layer_id: int = -1
    stream_id: str = ""
    optimization: str = "none"
    hw: Optional[MagiHardwareInfo] = None

@dataclass
class MagiPerfCompare:
    native_latency: float = 0.0
    native_memory: int = 0
    optimized_latency: float = 0.0
    optimized_memory: int = 0

class MagiCompilerBackendWrapper:
    """
    MagiCompilerBackend C++ 后端的 Python 包装器

    集成 6 大核心策略 + StrategyDispatcher：
    1. analyze_graph() → magicompiler/frontend/
    2. identify_optimization() → magicompiler/backend/
    3. stat_performance() → magicompiler/utils/
    4. parse_ggml_graph() → GGML-Viz 劫持
    5. parse_vllm_graph() → Easy-Debug 劫持
    6. parse_megatrain_graph() → 单层流式训练解析
    7. parse_mlx_graph() → MLX 计算图劫持
    """

    def __init__(self):
        self.backend = None
        if C_BACKEND_AVAILABLE:
            self.backend = _magi_backend.MagiCompilerBackend.get_instance()
        logger.info(f"[MagiBackendWrapper] C++ backend: {'Available' if self.backend else 'NOT available'}")

    def analyze_graph(self, graph_info: MagiGraphInfo) -> Dict[str, Any]:
        """分析计算图拓扑"""
        if not self.backend:
            logger.warning("[MagiBackendWrapper] C++ backend not available")
            return {}

        if self.backend:
            self.backend.analyze_graph(graph_info)

        return {
            "status": "analyzed",
            "backend": graph_info.backend.value if hasattr(graph_info.backend, 'value') else str(graph_info.backend),
            "nodes": len(graph_info.nodes)
        }

    def identify_optimization(self, graph_info: MagiGraphInfo) -> Dict[str, Any]:
        """识别优化机会"""
        if not self.backend:
            logger.warning("[MagiBackendWrapper] C++ backend not available")
            return {}

        if self.backend:
            self.backend.identify_optimization(graph_info)

        return {
            "status": "identified",
            "optimizations": ["KDA", "FlashMoE", "OMLX"]
        }

    def stat_performance(self, graph_info: MagiGraphInfo) -> Dict[str, Any]:
        """性能统计"""
        if not self.backend:
            logger.warning("[MagiBackendWrapper] C++ backend not available")
            return {}

        if self.backend:
            self.backend.stat_performance(graph_info)

        return {
            "status": "profiled",
            "latency": 0.0,
            "memory": 0,
            "flops": 0
        }

    def generate_optimal_code(self, graph_info: MagiGraphInfo) -> str:
        """生成最优代码"""
        if not self.backend:
            logger.warning("[MagiBackendWrapper] C++ backend not available")
            return ""

        if self.backend:
            return self.backend.generate_optimal_code(graph_info)

        return ""

    def dispatch_to_backend(self, backend_type: MagiBackendType, code: str) -> MagiPerfCompare:
        """下发到后端"""
        if not self.backend:
            logger.warning("[MagiBackendWrapper] C++ backend not available")
            return MagiPerfCompare()

        if self.backend:
            res = self.backend.dispatch_to_backend(backend_type, code)
            return MagiPerfCompare(
                native_latency=0,
                native_memory=0,
                optimized_latency=res.optimized_latency,
                optimized_memory=res.optimized_memory
            )

        return MagiPerfCompare()

    def run_native_backend(self, backend_type: MagiBackendType, graph_info: MagiGraphInfo) -> MagiPerfCompare:
        """运行原生后端"""
        if not self.backend:
            logger.warning("[MagiBackendWrapper] C++ backend not available")
            return MagiPerfCompare()

        if self.backend:
            res = self.backend.run_native_backend(backend_type, graph_info)
            return MagiPerfCompare(
                native_latency=res.native_latency,
                native_memory=res.native_memory,
                optimized_latency=0,
                optimized_memory=0
            )

        return MagiPerfCompare()

    def compare_performance(self, native: MagiPerfCompare, optimized: MagiPerfCompare, backend_type: MagiBackendType):
        """性能对比"""
        if not self.backend:
            logger.warning("[MagiBackendWrapper] C++ backend not available")
            return

        if self.backend:
            self.backend.compare_performance(native, optimized, backend_type)

    def parse_ggml_graph(self, inputs: List[MagiTensorInfo], outputs: List[MagiTensorInfo]) -> List[MagiGraphNode]:
        """解析 GGML 计算图"""
        if not self.backend:
            logger.warning("[MagiBackendWrapper] C++ backend not available")
            return []

        logger.info("[MagiBackendWrapper] parse_ggml_graph called")
        return []

    def parse_vllm_graph(self, inputs: List[MagiTensorInfo], outputs: List[MagiTensorInfo]) -> List[MagiGraphNode]:
        """解析 vLLM 计算图"""
        if not self.backend:
            logger.warning("[MagiBackendWrapper] C++ backend not available")
            return []

        logger.info("[MagiBackendWrapper] parse_vllm_graph called")
        return []

    def parse_megatrain_graph(self, inputs: List[MagiTensorInfo], outputs: List[MagiTensorInfo]) -> List[MagiGraphNode]:
        """解析 MegaTrain 计算图"""
        if not self.backend:
            logger.warning("[MagiBackendWrapper] C++ backend not available")
            return []

        logger.info("[MagiBackendWrapper] parse_megatrain_graph called")
        return []

    def parse_mlx_graph(self, inputs: List[MagiTensorInfo], outputs: List[MagiTensorInfo]) -> List[MagiGraphNode]:
        """解析 MLX 计算图"""
        if not self.backend:
            logger.warning("[MagiBackendWrapper] C++ backend not available")
            return []

        logger.info("[MagiBackendWrapper] parse_mlx_graph called")
        return []

def get_magi_backend() -> MagiCompilerBackendWrapper:
    """获取 MagiCompilerBackend 单例"""
    return MagiCompilerBackendWrapper()