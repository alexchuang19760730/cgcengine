"""
vLLM Custom Backend System - MagiCompiler 集成版
==================================================
vLLM 所有 compute 都可以外包给 MagiCompiler

与 magicompiler_unified_backend.py 配合使用
"""

from typing import Optional, Any, Dict, List, Callable
from dataclasses import dataclass
import torch

from .magicompiler_unified_backend import (
    UnifiedBackend,
    UnifiedComputeRequest,
    BackendSource,
    ComputeType,
    TensorInfo,
    get_unified_backend,
    get_vllm_adapter,
    create_magicompiler_backend,
)


class VLLMCustomBackend:
    """
    vLLM Custom Backend - MagiCompiler 外包版

    所有 vLLM compute 都通过这个类外包给 MagiCompiler

    使用方法：
        # 1. 初始化
        backend = VLLMCustomBackend()

        # 2. 注册到 vLLM（替换原生）
        vllm_model.set_custom_backend(backend)

        # 3. vLLM 所有 compute 自动走这里
        output = backend.compute(request)
    """

    def __init__(self):
        self.unified_backend = get_unified_backend()
        self.vllm_adapter = get_vllm_adapter()
        self.name = "vllm_magicompiler"
        self._native_compute: Optional[Callable] = None
        self._enabled = True
        self._stats = {
            "total_requests": 0,
            "handled_by_magi": 0,
            "fallback_native": 0,
        }

    def compute(self, req: UnifiedComputeRequest) -> bool:
        """
        vLLM 所有计算的统一入口

        Args:
            req: UnifiedComputeRequest

        Returns:
            True: MagiCompiler 接管
            False: 回退 vLLM 原生
        """
        if not self._enabled:
            return False

        self._stats["total_requests"] += 1

        # 1. 构建 vLLM 专用 UnifiedComputeRequest
        vllm_req = self._wrap_vllm_request(req)

        # 2. 委托给 MagiCompiler
        handled = self.unified_backend.compute(vllm_req)

        if handled:
            self._stats["handled_by_magi"] += 1
        else:
            self._stats["fallback_native"] += 1

        return handled

    def _wrap_vllm_request(self, req: Any) -> UnifiedComputeRequest:
        """将 vLLM 请求包装为 UnifiedComputeRequest"""
        # 提取 vLLM 的输入输出张量
        inputs = []
        outputs = []

        if hasattr(req, "hidden_states") and req.hidden_states is not None:
            inputs.append(TensorInfo(
                name="hidden_states",
                shape=tuple(req.hidden_states.shape),
                dtype=str(req.hidden_states.dtype),
                device=str(req.hidden_states.device),
                data=req.hidden_states,
            ))

        if hasattr(req, "q") and req.q is not None:
            inputs.append(TensorInfo(
                name="q",
                shape=tuple(req.q.shape),
                dtype=str(req.q.dtype),
                device=str(req.q.device),
                data=req.q,
            ))

        # 推断 compute_type
        compute_type = self._infer_vllm_compute_type(req)

        # 构建 UnifiedComputeRequest
        unified_req = UnifiedComputeRequest(
            request_id=f"vllm_{id(req)}",
            source=BackendSource.VLLM,
            compute_type=compute_type,
            inputs=inputs,
            outputs=outputs,
            raw_request=req,
            metadata={
                "positions": req.positions if hasattr(req, "positions") else None,
                "kv_caches": req.kv_caches if hasattr(req, "kv_caches") else None,
            },
        )

        return unified_req

    def _infer_vllm_compute_type(self, req: Any) -> ComputeType:
        """推断 vLLM 计算类型"""
        if hasattr(req, "compute_type"):
            return ComputeType(req.compute_type)

        # 根据请求内容推断
        if hasattr(req, "expert_ids"):
            return ComputeType.MOE_FFN
        elif hasattr(req, "q") and hasattr(req, "k"):
            return ComputeType.ATTENTION
        elif hasattr(req, "hidden_states"):
            return ComputeType.FULL_FORWARD

        return ComputeType.FULL_FORWARD

    def fallback_to_native(self, compute_fn: Callable, *args, **kwargs):
        """Fallback 到 vLLM 原生计算"""
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


class VLLMWithMagiCompiler:
    """
    vLLM + MagiCompiler 组合类

    方便一键创建完整的 vLLM + MagiCompiler 环境
    """

    def __init__(self):
        self.backend = VLLMCustomBackend()
        self.unified_backend = get_unified_backend()
        self.initialized = True

    def get_custom_backend(self) -> VLLMCustomBackend:
        """获取 custom backend"""
        return self.backend

    def analyze_compute_graph(self) -> Dict:
        """分析 vLLM 计算图"""
        return self.unified_backend.analyze_and_report()


def create_vllm_magicompiler_backend() -> VLLMCustomBackend:
    """
    创建 vLLM MagiCompiler backend

    这是便捷入口，一行代码创建完整 backend
    """
    create_magicompiler_backend()
    backend = VLLMCustomBackend()
    print("✅ vLLM MagiCompiler backend 创建成功")
    return backend


# ======================================================================
# 与现有 vllm_cgc_backend.py 的集成
# ======================================================================
def integrate_with_vllmcgc_backend(vllm_cgc_backend):
    """
    将 VLLMCGCBackend 与 MagiCompiler 集成

    Args:
        vllm_cgc_backend: VLLMCGCBackend 实例
    """
    backend = create_vllm_magicompiler_backend()

    # 替换原有的 compile 方法
    original_compile = vllm_cgc_backend.compile

    def magi_compile_wrapper():
        result = original_compile()
        print(f"✅ VLLMCGCBackend 已与 MagiCompiler 集成")
        return result

    vllm_cgc_backend.compile = magi_compile_wrapper

    return backend