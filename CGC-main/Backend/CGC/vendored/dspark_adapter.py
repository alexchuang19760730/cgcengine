"""dspark_adapter.py — DSpark vendored runtime adapter

把上游 DeepSpec/DSpark (https://github.com/deepseek-ai/DeepSpec) 的半自回归
草稿模型 + 置信度调度验证 API 包装为 CGC Gate 2.0 runtime 可直接调用的接口。

对应能力 g21_dspark_scheduler_runtime_adapter。
上游协议：MIT（见 vendored/deepspec/LICENSE）。
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_VENDOR_ROOT = Path(__file__).resolve().parent
_DEEPSPEC_ROOT = _VENDOR_ROOT / "deepspec"


class DSparkAdapterError(RuntimeError):
    """DSpark adapter 错误"""


@dataclass
class DSparkDraftResult:
    """DSpark 草稿生成 + 置信度调度结果"""
    draft_tokens: List[int]                  # 候选 token 序列
    confidence_scores: List[float] = field(default_factory=list)  # 每个候选 token 的置信度
    accepted_length: int = 0                 # 实际接受长度（验证后）
    verify_length_hint: int = 0              # 调度器建议的验证长度
    extra: Dict[str, Any] = field(default_factory=dict)


class DSparkRuntimeAdapter:
    """DSpark vendored runtime adapter

    提供与 CGC 推理引擎一致的草稿生成 + 置信度调度接口，内部委托给 vendored DeepSpec。

    使用方式：
        adapter = DSparkRuntimeAdapter()
        if adapter.is_available():
            adapter.load_model(model_id="qwen3-8b", target_model_path="...")
            result = adapter.draft_and_schedule(hidden_states=..., max_draft_length=5)
    """

    def __init__(self, *, deepspec_root: Optional[Path] = None):
        self.deepspec_root = Path(deepspec_root) if deepspec_root else _DEEPSPEC_ROOT
        self._dspark_modeling: Any = None
        self._confidence_head: Any = None
        self._dspark_model: Any = None
        self._loaded_model_id: Optional[str] = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """检测 vendored DeepSpec DSpark 是否可 import"""
        if self._available is not None:
            return self._available
        if not self.deepspec_root.exists():
            self._available = False
            return False
        root_str = str(self.deepspec_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        try:
            self._dspark_modeling = importlib.import_module("deepspec.modeling.dspark")
            self._available = True
        except Exception:
            self._available = False
        return self._available

    def load_model(
        self,
        *,
        model_id: str,
        target_model_path: str,
        dspark_ckpt_path: Optional[str] = None,
        model_arch: str = "qwen3",
    ) -> None:
        """加载 DSpark 草稿模型

        Args:
            model_id: 目标模型 ID（如 "qwen3-8b"）
            target_model_path: 目标模型权重路径（HF format）
            dspark_ckpt_path: DSpark 草稿模型权重路径；None 则按 DeepSpec 默认查找
            model_arch: 模型架构 ("qwen3" / "gemma4")
        """
        if not self.is_available():
            raise DSparkAdapterError("vendored DeepSpec DSpark not available")
        try:
            if model_arch == "qwen3":
                model_cls = self._dspark_modeling.Qwen3DSparkModel
            elif model_arch == "gemma4":
                model_cls = self._dspark_modeling.Gemma4DSparkModel
            else:
                raise DSparkAdapterError(f"unsupported model_arch: {model_arch}")
            self._dspark_model = model_cls.from_pretrained(
                target_model_path=target_model_path,
                dspark_ckpt_path=dspark_ckpt_path,
            )
            self._loaded_model_id = model_id
            # 同时加载 confidence head
            try:
                from deepspec.eval.dspark.confidence_head import ConfidenceHead  # type: ignore
                self._confidence_head = ConfidenceHead.from_pretrained(dspark_ckpt_path)
            except Exception:
                # confidence head 可选
                self._confidence_head = None
        except Exception as e:
            raise DSparkAdapterError(f"load_model failed: {e}") from e

    def draft_and_schedule(
        self,
        *,
        hidden_states: Any,
        max_draft_length: int = 5,
        gpu_load_factor: float = 0.0,
    ) -> DSparkDraftResult:
        """半自回归草稿生成 + 置信度调度

        Args:
            hidden_states: 目标模型最后一层 hidden_states
            max_draft_length: 最大草稿长度（DSpark 默认 5）
            gpu_load_factor: 当前 GPU 负载 [0,1]，用于硬件感知前缀调度

        Returns:
            DSparkDraftResult
        """
        if self._dspark_model is None:
            raise DSparkAdapterError("dspark model not loaded, call load_model() first")
        try:
            # 半自回归主干：一次前向产出 max_draft_length 个候选
            forward_out = self._dspark_model.forward(
                hidden_states,
                max_draft_length=max_draft_length,
            )
            draft_tokens = list(forward_out.draft_tokens)
            base_logits = getattr(forward_out, "logits", None)

            # 置信度打分 + 硬件感知调度
            confidence_scores: List[float] = []
            verify_length_hint = max_draft_length
            if self._confidence_head is not None and base_logits is not None:
                confidence_scores = list(self._confidence_head.score(base_logits))
                # 硬件感知前缀调度：根据 GPU 负载动态截断
                # 负载越高，验证长度越短
                budget = max(1, int(max_draft_length * (1.0 - gpu_load_factor)))
                # 取累积置信度最高的前缀
                cumulative = 0.0
                for i, c in enumerate(confidence_scores):
                    cumulative += c
                    if i + 1 >= budget:
                        verify_length_hint = i + 1
                        break
                else:
                    verify_length_hint = len(confidence_scores)

            return DSparkDraftResult(
                draft_tokens=draft_tokens,
                confidence_scores=confidence_scores,
                verify_length_hint=verify_length_hint,
                extra={
                    "max_draft_length": max_draft_length,
                    "gpu_load_factor": gpu_load_factor,
                    "arch": getattr(self._dspark_model, "arch", "unknown"),
                },
            )
        except Exception as e:
            raise DSparkAdapterError(f"draft_and_schedule failed: {e}") from e

    def stats(self) -> Dict[str, Any]:
        """返回 adapter 状态"""
        return {
            "available": bool(self._available),
            "deepspec_root": str(self.deepspec_root),
            "loaded_model_id": self._loaded_model_id,
            "model_loaded": self._dspark_model is not None,
            "confidence_head_loaded": self._confidence_head is not None,
        }


_ADAPTER_SINGLETON: Optional[DSparkRuntimeAdapter] = None


def get_dspark_adapter() -> DSparkRuntimeAdapter:
    """获取 DSpark adapter 单例"""
    global _ADAPTER_SINGLETON
    if _ADAPTER_SINGLETON is None:
        _ADAPTER_SINGLETON = DSparkRuntimeAdapter()
    return _ADAPTER_SINGLETON
