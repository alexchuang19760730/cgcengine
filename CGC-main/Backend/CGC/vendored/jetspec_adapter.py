"""jetspec_adapter.py — JetSpec vendored runtime adapter

把上游 JetSpec (https://github.com/hao-ai-lab/JetSpec) 的并行树草稿投机解码
API 包装为 CGC Gate 2.0 runtime 可直接调用的接口。

对应能力 g21_jetspec_draft_runtime_adapter。
上游协议：MIT（见 vendored/jetspec/LICENSE）。
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# JetSpec vendored 根目录（本文件向上找到 vendored/jetspec）
_VENDOR_ROOT = Path(__file__).resolve().parent
_JETSPEC_ROOT = _VENDOR_ROOT / "jetspec"


class JetSpecAdapterError(RuntimeError):
    """JetSpec adapter 错误"""


@dataclass
class JetSpecDraftResult:
    """JetSpec 草稿生成结果"""
    draft_tokens: List[int]              # 候选 token 序列
    draft_logits: Any = None             # 候选 logits（可选，用于校验）
    tree_structure: Optional[Any] = None # 树结构（若使用 tree drafting）
    accepted_length: int = 0             # 实际接受长度（验证后）
    extra: Dict[str, Any] = field(default_factory=dict)


class JetSpecRuntimeAdapter:
    """JetSpec vendored runtime adapter

    提供与 CGC 推理引擎一致的草稿生成接口，内部委托给 vendored JetSpec。

    使用方式：
        adapter = JetSpecRuntimeAdapter()
        if adapter.is_available():
            adapter.load_draft_head(model_id="qwen3-8b")
            result = adapter.draft(hidden_states=..., num_draft_tokens=16)
    """

    def __init__(self, *, jetspec_root: Optional[Path] = None):
        self.jetspec_root = Path(jetspec_root) if jetspec_root else _JETSPEC_ROOT
        self._jetspec_mod: Any = None
        self._draft_head: Any = None
        self._loaded_model_id: Optional[str] = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """检测 vendored JetSpec 是否可 import"""
        if self._available is not None:
            return self._available
        if not self.jetspec_root.exists():
            self._available = False
            return False
        # 把 vendored jetspec 路径加到 sys.path（若未在）
        root_str = str(self.jetspec_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        try:
            self._jetspec_mod = importlib.import_module("jetspec")
            self._available = True
        except Exception:
            self._available = False
        return self._available

    def load_draft_head(self, *, model_id: str, draft_head_path: Optional[str] = None) -> None:
        """加载 JetSpec draft head 模型

        Args:
            model_id: 目标模型 ID（如 "qwen3-8b"）
            draft_head_path: draft head 权重路径；None 则用 JetSpec 默认查找逻辑
        """
        if not self.is_available():
            raise JetSpecAdapterError("vendored JetSpec not available")
        try:
            load_draft_head = self._jetspec_mod.load_draft_head
        except AttributeError as e:
            raise JetSpecAdapterError(f"jetspec.load_draft_head not found: {e}") from e

        try:
            self._draft_head = load_draft_head(model_id=model_id, ckpt_path=draft_head_path)
            self._loaded_model_id = model_id
        except Exception as e:
            raise JetSpecAdapterError(f"load_draft_head failed: {e}") from e

    def draft(
        self,
        *,
        hidden_states: Any,
        num_draft_tokens: int = 16,
        tree_budget: Optional[int] = None,
    ) -> JetSpecDraftResult:
        """生成草稿 tokens

        Args:
            hidden_states: 目标模型最后一层 hidden_states（torch.Tensor）
            num_draft_tokens: 草稿 token 数（低预算路径）
            tree_budget: 树预算（高预算路径，None 则用线性 drafting）

        Returns:
            JetSpecDraftResult
        """
        if self._draft_head is None:
            raise JetSpecAdapterError("draft_head not loaded, call load_draft_head() first")
        try:
            if tree_budget is not None:
                # 树 drafting 路径
                from jetspec.draft_head_adapter import DraftHeadTreeDrafter  # type: ignore
                drafter = DraftHeadTreeDrafter(self._draft_head, budget=tree_budget)
                draft_tokens, tree = drafter.draft(hidden_states, num_tokens=num_draft_tokens)
                return JetSpecDraftResult(
                    draft_tokens=list(draft_tokens),
                    tree_structure=tree,
                    extra={"mode": "tree", "tree_budget": tree_budget},
                )
            else:
                # 线性 drafting 路径
                from jetspec.draft_head_adapter import DraftHeadDrafter  # type: ignore
                drafter = DraftHeadDrafter(self._draft_head)
                draft_tokens = drafter.draft(hidden_states, num_tokens=num_draft_tokens)
                return JetSpecDraftResult(
                    draft_tokens=list(draft_tokens),
                    extra={"mode": "linear"},
                )
        except Exception as e:
            raise JetSpecAdapterError(f"draft failed: {e}") from e

    def stats(self) -> Dict[str, Any]:
        """返回 adapter 状态"""
        return {
            "available": bool(self._available),
            "jetspec_root": str(self.jetspec_root),
            "loaded_model_id": self._loaded_model_id,
            "draft_head_loaded": self._draft_head is not None,
        }


_ADAPTER_SINGLETON: Optional[JetSpecRuntimeAdapter] = None


def get_jetspec_adapter() -> JetSpecRuntimeAdapter:
    """获取 JetSpec adapter 单例"""
    global _ADAPTER_SINGLETON
    if _ADAPTER_SINGLETON is None:
        _ADAPTER_SINGLETON = JetSpecRuntimeAdapter()
    return _ADAPTER_SINGLETON
