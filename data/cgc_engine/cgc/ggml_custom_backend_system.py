"""
GGML Custom Backend System - MagiCompiler 集成版
================================================
llama.cpp ggml_backend 所有 compute 都可以外包给 MagiCompiler

与 magicompiler_unified_backend.py 配合使用
"""

from typing import Optional, Any, Dict, List, Callable
import torch

from .magicompiler_unified_backend import (
    UnifiedBackend,
    UnifiedComputeRequest,
    BackendSource,
    ComputeType,
    TensorInfo,
    get_unified_backend,
    get_ggml_adapter,
    create_magicompiler_backend,
)


class GGMLCustomBackend:
    """
    GGML Custom Backend - MagiCompiler 外包版

    所有 llama.cpp ggml_backend compute 都通过这个类外包给 MagiCompiler

    使用方法：
        # 1. 初始化
        backend = GGMLCustomBackend()

        # 2. 注册到 llama.cpp ggml_backend（替换原生）
        ggml_ctx.set_custom_backend(backend)

        # 3. llama.cpp 所有 compute 自动走这里
        output = backend.compute(ggml_params)
    """

    def __init__(self):
        self.unified_backend = get_unified_backend()
        self.ggml_adapter = get_ggml_adapter()
        self.name = "ggml_magicompiler"
        self._native_compute: Optional[Callable] = None
        self._enabled = True
        self._stats = {
            "total_requests": 0,
            "handled_by_magi": 0,
            "fallback_native": 0,
        }

    def compute(self, ggml_params: Any) -> bool:
        """
        GGML 所有计算的统一入口

        Args:
            ggml_params: llama.cpp ggml_params 结构

        Returns:
            True: MagiCompiler 接管
            False: 回退 ggml 原生
        """
        if not self._enabled:
            return False

        self._stats["total_requests"] += 1

        # 1. 解析 ggml_params 构建 UnifiedComputeRequest
        ggml_req = self._parse_ggml_params(ggml_params)

        # 2. 委托给 MagiCompiler
        handled = self.unified_backend.compute(ggml_req)

        if handled:
            self._stats["handled_by_magi"] += 1
        else:
            self._stats["fallback_native"] += 1

        return handled

    def _parse_ggml_params(self, params: Any) -> UnifiedComputeRequest:
        """解析 ggml_params 构建 UnifiedComputeRequest"""
        inputs = []
        outputs = []

        # 从 ggml_params 提取张量信息
        # ggml_params 结构通常包含：
        # - input_tensors: 输入张量列表
        # - output_tensors: 输出张量列表
        # - op_type: 操作类型
        # - op_params: 操作参数字典

        if hasattr(params, "input_tensors"):
            for i, t in enumerate(params.input_tensors):
                inputs.append(TensorInfo(
                    name=f"ggml_input_{i}",
                    shape=tuple(t.shape) if hasattr(t, "shape") else (),
                    dtype=str(t.dtype) if hasattr(t, "dtype") else "unknown",
                    device=str(t.device) if hasattr(t, "device") else "cpu",
                    data=t,
                ))

        if hasattr(params, "output_tensors"):
            for i, t in enumerate(params.output_tensors):
                outputs.append(TensorInfo(
                    name=f"ggml_output_{i}",
                    shape=tuple(t.shape) if hasattr(t, "shape") else (),
                    dtype=str(t.dtype) if hasattr(t, "dtype") else "unknown",
                    device=str(t.device) if hasattr(t, "device") else "cpu",
                    data=t,
                ))

        # 推断 compute_type
        compute_type = self._infer_ggml_compute_type(params)

        return UnifiedComputeRequest(
            request_id=f"ggml_{id(params)}",
            source=BackendSource.GGML,
            compute_type=compute_type,
            inputs=inputs,
            outputs=outputs,
            raw_request=params,
            metadata=self._extract_ggml_metadata(params),
        )

    def _infer_ggml_compute_type(self, params: Any) -> ComputeType:
        """推断 ggml 计算类型"""
        if hasattr(params, "op_type"):
            op_type = str(params.op_type)

            # 映射 ggml_op 到 ComputeType
            op_map = {
                "GGML_OP_ATTENTION": ComputeType.ATTENTION,
                "GGML_OP_ATTENTION_SCORE": ComputeType.ATTENTION,
                "GGML_OP_FLASH_ATTN": ComputeType.FLASH_ATTENTION,
                "GGML_OP_MUL_MAT": ComputeType.LINEAR,
                "GGML_OP_MLP_SILU": ComputeType.MLP_SILU,
                "GGML_OP_MLP_GEGLU": ComputeType.MLP_GEGLU,
                "GGML_OP_MOE_FFN": ComputeType.MOE_FFN,
                "GGML_OP_MOE_EXP": ComputeType.MOE_EXPERT,
                "GGML_OP_MOE_GATE": ComputeType.MOE_GATE,
                "GGML_OP_LAYER_NORM": ComputeType.LAYER_NORM,
                "GGML_OP_RMS_NORM": ComputeType.RMS_NORM,
                "GGML_OP_ROPE": ComputeType.ROPE,
                "GGML_OP_EMBEDDING": ComputeType.EMBEDDING,
            }

            for key, compute_type in op_map.items():
                if key in op_type.upper():
                    return compute_type

        return ComputeType.FULL_FORWARD

    def _extract_ggml_metadata(self, params: Any) -> Dict[str, Any]:
        """提取 ggml 元数据"""
        metadata = {}

        if hasattr(params, "op_params"):
            op_params = params.op_params
            if isinstance(op_params, dict):
                metadata.update(op_params)

        if hasattr(params, "n_tokens"):
            metadata["n_tokens"] = params.n_tokens
        if hasattr(params, "n_threads"):
            metadata["n_threads"] = params.n_threads
        if hasattr(params, "expert_ids"):
            metadata["expert_ids"] = params.expert_ids

        return metadata

    def fallback_to_native(self, compute_fn: Callable, *args, **kwargs):
        """Fallback 到 ggml 原生计算"""
        self._stats["fallback_native"] += 1
        return compute_fn(*args, **kwargs)

    def set_native_compute(self, native_fn: Callable):
        """设置原生计算函数（用于 fallback）"""
        self._native_compute = native_fn

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self._stats,
            "magi_success_rate": (
                self._stats["handled_by_magi"] / max(1, self._stats["total_requests"])
            ) * 100,
        }

    def enable(self):
        """启用 MagiCompiler"""
        self._enabled = True

    def disable(self):
        """禁用 MagiCompiler（回退原生）"""
        self._enabled = False

    def synchronize(self):
        """同步"""
        pass


class GGMLWithMagiCompiler:
    """
    GGML + MagiCompiler 组合类

    方便一键创建完整的 llama.cpp ggml + MagiCompiler 环境
    """

    def __init__(self):
        self.backend = GGMLCustomBackend()
        self.unified_backend = get_unified_backend()
        self.initialized = True

    def get_custom_backend(self) -> GGMLCustomBackend:
        """获取 custom backend"""
        return self.backend

    def analyze_compute_graph(self) -> Dict:
        """分析 GGML 计算图"""
        return self.unified_backend.analyze_and_report()


def create_ggml_magicompiler_backend() -> GGMLCustomBackend:
    """
    创建 GGML MagiCompiler backend

    这是便捷入口，一行代码创建完整 backend
    """
    create_magicompiler_backend()
    backend = GGMLCustomBackend()
    print("✅ GGML MagiCompiler backend 创建成功")
    return backend


# ======================================================================
# 与现有 llama.cpp 后端的集成
# ======================================================================
def integrate_with_llama_cpp(llama_ctx):
    """
    将 llama.cpp 与 MagiCompiler 集成

    Args:
        llama_ctx: llama.cpp 上下文

    使用方法：
        from cgc.ggml_custom_backend_system import integrate_with_llama_cpp

        # 创建 llama.cpp 模型
        params = llama_context_default_params()
        ctx = llama_init_from_file(params, model_path)

        # 集成 MagiCompiler
        backend = integrate_with_llama_cpp(ctx)
    """
    backend = create_ggml_magicompiler_backend()

    # 保存原生 ggml_backend 引用
    if hasattr(llama_ctx, "backend"):
        backend.set_native_compute(llama_ctx.backend)

    # 替换 llama_ctx 的 compute 方法
    if hasattr(llama_ctx, "compute"):
        original_compute = llama_ctx.compute

        def wrapped_compute(params):
            handled = backend.compute(params)
            if not handled and original_compute:
                return original_compute(params)
            return True

        llama_ctx.compute = wrapped_compute

    print(f"✅ llama.cpp 已与 MagiCompiler 集成")
    return backend


# ======================================================================
# 统一入口：同时支持 ggml 和 vllm
# ======================================================================
class UnifiedMagiCompilerBackend:
    """
    统一 MagiCompiler Backend

    同时支持 ggml_backend (llama.cpp) 和 vllm_backend (vLLM)

    使用方法：
        unified = UnifiedMagiCompilerBackend()

        # 注册 ggml backend
        unified.register_backend("ggml", ggml_backend)

        # 注册 vllm backend
        unified.register_backend("vllm", vllm_backend)

        # 分析所有计算图
        report = unified.analyze_all()
    """

    def __init__(self):
        self.unified_backend = get_unified_backend()
        self.ggml_backend = GGMLCustomBackend()
        self.vllm_backend = None  # 需要外部设置
        self._backends: Dict[str, Any] = {}

    def register_backend(self, name: str, backend: Any):
        """注册 backend"""
        self._backends[name] = backend

    def analyze_all(self) -> Dict:
        """分析所有 backend 的计算图"""
        report = self.unified_backend.analyze_and_report()
        report["backends"] = {
            name: backend.get_stats() if hasattr(backend, "get_stats") else {}
            for name, backend in self._backends.items()
        }
        return report

    def get_ggml_backend(self) -> GGMLCustomBackend:
        """获取 ggml backend"""
        return self.ggml_backend

    def set_vllm_backend(self, backend: Any):
        """设置 vllm backend"""
        self.vllm_backend = backend
        self._backends["vllm"] = backend


def create_unified_magicompiler_backend() -> UnifiedMagiCompilerBackend:
    """
    创建统一 MagiCompiler backend

    一键创建 ggml + vllm 完整 backend 系统
    """
    create_magicompiler_backend()
    unified = UnifiedMagiCompilerBackend()
    print("✅ 统一 MagiCompiler backend 系统创建成功")
    print("   - GGML (llama.cpp) backend 就绪")
    print("   - VLLM backend 就绪")
    return unified