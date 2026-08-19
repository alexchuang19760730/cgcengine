"""sglang_spec_plugin.py — JetSpec / DSpark 接入 SGLang verify stack

本模块把 vendored JetSpec / DSpark adapter 注册为 SGLang 的自定义 spec 算法：
  - "JETSPEC"  → JetSpec 树状草稿 + SGLang target verify
  - "DSPARK"   → DSpark 半自回归草稿 + 置信度调度 + SGLang target verify

注册后，SGLang server 启动时通过 --speculative-algorithm JETSPEC/DSPARK 即可
走 vendored runtime 的 draft → target verify 端到端路径。

对应能力 g21_jetspec_draft_runtime_adapter + g21_dspark_scheduler_runtime_adapter
的 verify stack 接入部分。

注册方式（cgc_api_server 启动 SGLang 前调用一次）：
    from Backend.CGC.vendored.sglang_spec_plugin import register_cgc_spec_algos
    register_cgc_spec_algos()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger("cgc.sglang_spec_plugin")


# ---------------------------------------------------------------------------
# 数据结构：draft → target verify 中间态
# ---------------------------------------------------------------------------

@dataclass
class DraftProposal:
    """草稿 worker 输出，供 target verify 消费"""
    draft_tokens: List[int]              # 候选 token 序列
    draft_logits: Any = None             # 候选 logits（可选）
    tree_mask: Optional[Any] = None      # 树注意力 mask（JetSpec 树状路径）
    confidence_scores: List[float] = field(default_factory=list)  # DSpark 置信度
    verify_length_hint: int = 0          # DSpark 调度器建议验证长度
    source: str = "jetspec"              # "jetspec" / "dspark"
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 轻量 draft worker（不继承 TpModelWorker，避免引入完整 SGLang 依赖）
# ---------------------------------------------------------------------------

class JetSpecDraftWorker:
    """JetSpec 草稿 worker

    包装 vendored JetSpecRuntimeAdapter，把 target worker 的 hidden_states
    转换为 JetSpec draft 输入，产出 DraftProposal 交给 target verify。
    """

    def __init__(self, *, model_id: str, draft_head_path: Optional[str] = None,
                 num_draft_tokens: int = 16, tree_budget: Optional[int] = None):
        from Backend.CGC.vendored import JetSpecRuntimeAdapter, JetSpecAdapterError  # type: ignore
        self.adapter = JetSpecRuntimeAdapter()
        self.model_id = model_id
        self.num_draft_tokens = num_draft_tokens
        self.tree_budget = tree_budget
        self.adapter_error: Optional[str] = None
        if not self.adapter.is_available():
            self.adapter_error = "vendored JetSpec not available"
            LOG.warning(f"[JetSpecDraftWorker] {self.adapter_error}")
            return
        try:
            self.adapter.load_draft_head(model_id=model_id, draft_head_path=draft_head_path)
            LOG.info(f"[JetSpecDraftWorker] loaded draft head for {model_id}")
        except Exception as e:
            self.adapter_error = f"load_draft_head failed: {e}"
            LOG.error(f"[JetSpecDraftWorker] {self.adapter_error}")

    def propose(self, *, hidden_states: Any, num_draft_tokens: Optional[int] = None) -> DraftProposal:
        """生成草稿 proposal"""
        if self.adapter_error:
            return DraftProposal(draft_tokens=[], source="jetspec", extra={"error": self.adapter_error})
        n = num_draft_tokens or self.num_draft_tokens
        try:
            r = self.adapter.draft(
                hidden_states=hidden_states,
                num_draft_tokens=n,
                tree_budget=self.tree_budget,
            )
            return DraftProposal(
                draft_tokens=r.draft_tokens,
                tree_structure=r.tree_structure,
                source="jetspec",
                extra=r.extra,
            )
        except Exception as e:
            return DraftProposal(draft_tokens=[], source="jetspec", extra={"error": str(e)})


class DSparkDraftWorker:
    """DSpark 草稿 worker

    包装 vendored DSparkRuntimeAdapter，产出 DraftProposal（含置信度）。
    """

    def __init__(self, *, model_id: str, target_model_path: str,
                 dspark_ckpt_path: Optional[str] = None,
                 model_arch: str = "qwen3",
                 max_draft_length: int = 5):
        from Backend.CGC.vendored import DSparkRuntimeAdapter  # type: ignore
        self.adapter = DSparkRuntimeAdapter()
        self.model_id = model_id
        self.max_draft_length = max_draft_length
        self.adapter_error: Optional[str] = None
        if not self.adapter.is_available():
            self.adapter_error = "vendored DSpark not available"
            LOG.warning(f"[DSparkDraftWorker] {self.adapter_error}")
            return
        try:
            self.adapter.load_model(
                model_id=model_id,
                target_model_path=target_model_path,
                dspark_ckpt_path=dspark_ckpt_path,
                model_arch=model_arch,
            )
            LOG.info(f"[DSparkDraftWorker] loaded model for {model_id} ({model_arch})")
        except Exception as e:
            self.adapter_error = f"load_model failed: {e}"
            LOG.error(f"[DSparkDraftWorker] {self.adapter_error}")

    def propose(self, *, hidden_states: Any, gpu_load_factor: float = 0.0,
                max_draft_length: Optional[int] = None) -> DraftProposal:
        """生成草稿 proposal + 置信度"""
        if self.adapter_error:
            return DraftProposal(draft_tokens=[], source="dspark", extra={"error": self.adapter_error})
        n = max_draft_length or self.max_draft_length
        try:
            r = self.adapter.draft_and_schedule(
                hidden_states=hidden_states,
                max_draft_length=n,
                gpu_load_factor=gpu_load_factor,
            )
            return DraftProposal(
                draft_tokens=r.draft_tokens,
                confidence_scores=r.confidence_scores,
                verify_length_hint=r.verify_length_hint,
                source="dspark",
                extra=r.extra,
            )
        except Exception as e:
            return DraftProposal(draft_tokens=[], source="dspark", extra={"error": str(e)})


# ---------------------------------------------------------------------------
# Target verify 桥接：把 DraftProposal 转换为 SGLang target worker 输入
# ---------------------------------------------------------------------------

def build_target_verify_input(proposal: DraftProposal, *, target_token_id: int) -> Dict[str, Any]:
    """把 DraftProposal 转换为 SGLang target worker forward_target_extend 期望的输入

    SGLang target verify 接口期望：
      - draft_tokens: [num_draft_tokens] tensor，附加在已生成 token 序列尾
      - tree_mask: 可选树注意力 mask（JetSpec 路径）
      - verify_length: 实际验证长度（DSpark 调度器建议值）

    Args:
        proposal: 草稿 worker 输出
        target_token_id: 当前 target 已生成的最后一个 token id（draft 序列起点）
    """
    # 拼接：[target_token_id, draft_tokens...]
    full_seq = [int(target_token_id)] + [int(t) for t in proposal.draft_tokens]
    return {
        "draft_tokens": full_seq,
        "tree_mask": proposal.tree_mask,
        "verify_length_hint": proposal.verify_length_hint or len(proposal.draft_tokens),
        "source": proposal.source,
        "confidence_scores": proposal.confidence_scores,
    }


def accept_tokens_after_verify(
    target_logits: Any,
    draft_proposal: DraftProposal,
    *,
    sampling_temperature: float = 1.0,
) -> Tuple[List[int], int]:
    """target verify 后做 token acceptance

    简化版贪婪 acceptance：对每个 draft token，比较 target logits argmax
    与 draft token 是否一致；连续不一致则截断。

    Args:
        target_logits: target worker forward 后的 logits，shape=[num_tokens, vocab_size]
        draft_proposal: 草稿 proposal
        sampling_temperature: 采样温度（贪婪模式下忽略）

    Returns:
        (accepted_tokens, accepted_length)
    """
    try:
        # target_logits[0] 对应 target_token_id 的 logits（不参与 acceptance）
        # target_logits[1..N] 对应 draft_tokens[0..N-1] 的 verify logits
        import torch
        if not isinstance(target_logits, torch.Tensor):
            return [], 0
        # argmax 比对
        verify_logits = target_logits[1:1 + len(draft_proposal.draft_tokens)]
        target_argmax = verify_logits.argmax(dim=-1).tolist()
        accepted: List[int] = []
        for i, (draft_t, target_t) in enumerate(zip(draft_proposal.draft_tokens, target_argmax)):
            if int(draft_t) == int(target_t):
                accepted.append(int(draft_t))
            else:
                # 第一个不匹配处截断，target_t 取代该位置
                accepted.append(int(target_t))
                break
        else:
            # 全部匹配，追加 target 最后一个 token
            if len(target_argmax) > len(draft_proposal.draft_tokens):
                accepted.append(int(target_argmax[-1]))
        return accepted, len(accepted)
    except Exception:
        return [], 0


# ---------------------------------------------------------------------------
# Plugin 注册入口
# ---------------------------------------------------------------------------

_REGISTERED = False


def register_cgc_spec_algos() -> Dict[str, Any]:
    """把 JetSpec / DSpark 注册为 SGLang 自定义 spec 算法

    注册名：
      - "JETSPEC"  → JetSpec 树状草稿
      - "DSPARK"   → DSpark 半自回归草稿

    Returns:
        注册结果摘要
    """
    global _REGISTERED
    if _REGISTERED:
        return {"registered": False, "reason": "already registered"}

    try:
        from sglang.srt.speculative.spec_registry import (
            register_algorithm, CustomSpecAlgo,
        )
    except ImportError as e:
        LOG.error(f"[register_cgc_spec_algos] spec_registry import failed: {e}")
        return {"registered": False, "error": str(e)}

    results = {}

    # ---- JetSpec ----
    try:
        class JetSpecAlgo(CustomSpecAlgo):
            """JetSpec custom spec algo"""

            def is_eagle(self) -> bool:
                # 复用 SGLang eagle 的 target verify 路径
                return True

            def supports_target_verify_for_draft(self) -> bool:
                return True

            def create_worker(self, server_args):
                return JetSpecDraftWorkerFactory(server_args)

        @register_algorithm("JETSPEC", spec_class=JetSpecAlgo)
        def _jetspec_factory(server_args):
            return JetSpecDraftWorkerFactory(server_args)

        results["JETSPEC"] = "registered"
    except Exception as e:
        results["JETSPEC"] = f"failed: {e}"

    # ---- DSpark ----
    try:
        class DSparkAlgo(CustomSpecAlgo):
            """DSpark custom spec algo"""

            def is_eagle(self) -> bool:
                return True

            def supports_target_verify_for_draft(self) -> bool:
                return True

            def create_worker(self, server_args):
                return DSparkDraftWorkerFactory(server_args)

        @register_algorithm("DSPARK", spec_class=DSparkAlgo)
        def _dspark_factory(server_args):
            return DSparkDraftWorkerFactory(server_args)

        results["DSPARK"] = "registered"
    except Exception as e:
        results["DSPARK"] = f"failed: {e}"

    _REGISTERED = True
    LOG.info(f"[register_cgc_spec_algos] results: {results}")
    return {"registered": True, "algos": results}


# ---------------------------------------------------------------------------
# Worker factory：从 server_args 构造 draft worker
# ---------------------------------------------------------------------------

class JetSpecDraftWorkerFactory:
    """从 SGLang server_args 构造 JetSpecDraftWorker"""

    def __init__(self, server_args: Any):
        self.server_args = server_args

    def __call__(self, *args, **kwargs) -> JetSpecDraftWorker:
        # 从 server_args 提取配置
        model_id = getattr(self.server_args, "model_path", "") or "default"
        draft_head_path = getattr(self.server_args, "speculative_draft_model_path", None)
        num_draft_tokens = getattr(self.server_args, "speculative_num_draft_tokens", 16)
        tree_budget = getattr(self.server_args, "speculative_eagle_topk", None)

        return JetSpecDraftWorker(
            model_id=model_id,
            draft_head_path=draft_head_path,
            num_draft_tokens=num_draft_tokens,
            tree_budget=tree_budget,
        )


class DSparkDraftWorkerFactory:
    """从 SGLang server_args 构造 DSparkDraftWorker"""

    def __init__(self, server_args: Any):
        self.server_args = server_args

    def __call__(self, *args, **kwargs) -> DSparkDraftWorker:
        model_id = getattr(self.server_args, "model_path", "") or "default"
        target_model_path = model_id
        dspark_ckpt_path = getattr(self.server_args, "speculative_draft_model_path", None)
        model_arch = getattr(self.server_args, "cgc_dspark_model_arch", "qwen3")
        max_draft_length = getattr(self.server_args, "speculative_num_draft_tokens", 5)

        return DSparkDraftWorker(
            model_id=model_id,
            target_model_path=target_model_path,
            dspark_ckpt_path=dspark_ckpt_path,
            model_arch=model_arch,
            max_draft_length=max_draft_length,
        )


# ---------------------------------------------------------------------------
# 便捷入口：单次 draft → verify 端到端调用（不依赖 SGLang scheduler）
# ---------------------------------------------------------------------------

def run_draft_verify_round(
    *,
    draft_worker: Any,
    target_worker: Any,
    hidden_states: Any,
    target_token_id: int,
    forward_batch: Any = None,
    **kwargs,
) -> Dict[str, Any]:
    """单次 draft → target verify 端到端调用

    Args:
        draft_worker: JetSpecDraftWorker / DSparkDraftWorker 实例
        target_worker: SGLang TpModelWorker 实例（target model）
        hidden_states: target 最后一层 hidden_states
        target_token_id: 当前 target 已生成最后一个 token id

    Returns:
        {accepted_tokens, accepted_length, draft_source, draft_proposal}
    """
    # 1. draft propose
    if hasattr(draft_worker, "propose"):
        if "gpu_load_factor" in getattr(draft_worker, "propose", lambda **kw: None).__code__.co_varnames:
            gpu_load = 0.0
            if forward_batch is not None:
                gpu_load = float(getattr(forward_batch, "cgc_gpu_load_factor", 0.0))
            proposal = draft_worker.propose(
                hidden_states=hidden_states, gpu_load_factor=gpu_load,
            )
        else:
            proposal = draft_worker.propose(hidden_states=hidden_states)
    else:
        return {"error": "draft_worker.propose missing", "accepted_tokens": []}

    # 2. target verify
    verify_input = build_target_verify_input(proposal, target_token_id=target_token_id)

    # 调用 target_worker.forward_target_extend（SGLang 接口）
    target_logits = None
    try:
        if hasattr(target_worker, "forward_target_extend"):
            target_logits = target_worker.forward_target_extend(
                draft_tokens=verify_input["draft_tokens"],
                tree_mask=verify_input["tree_mask"],
                forward_batch=forward_batch,
            )
        elif hasattr(target_worker, "forward"):
            # 退化：直接 forward target model
            target_logits = target_worker.forward(
                verify_input["draft_tokens"],
                forward_batch=forward_batch,
            )
    except Exception as e:
        return {
            "error": f"target verify failed: {e}",
            "draft_proposal": proposal.__dict__,
            "accepted_tokens": [],
        }

    # 3. acceptance
    accepted_tokens, accepted_length = accept_tokens_after_verify(
        target_logits, proposal,
    )

    return {
        "accepted_tokens": accepted_tokens,
        "accepted_length": accepted_length,
        "draft_source": proposal.source,
        "draft_proposal": proposal.__dict__,
        "verify_input": verify_input,
    }
