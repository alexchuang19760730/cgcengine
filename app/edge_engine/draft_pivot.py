#!/usr/bin/env python3
"""DraftPivotEngine -- 端侧 Draft 分层前向抢首包.

核心思想: MTP Draft 跑到第 pivot_layer, 就把当前 token 流式输出
         给用户, 不等云端 verify. 云端 verify 后用 SSE patch 修正.

延迟分析:
  Prefill (256 tokens, M4 Pro): ~30-50ms
  Pivot forward (6 layers): ~10-15ms
  首包延迟: 40-65ms (vs 当前 50-150ms 直连云端)
  Draft 完整生成 (8 tokens): +30-50ms
  draft 序列可用: 70-115ms (vs 当前 verify 150-300ms)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from app.edge_engine.omlx_runtime import OMLXRuntime, DraftEvent, DraftResult

logger = logging.getLogger(__name__)


@dataclass
class PivotResult:
    """Pivot 抢首包结果."""
    first_token: int = -1               # 抢出的首 token
    pivot_layer: int = -1               # pivot 层
    pivot_latency_ms: float = 0.0       # 抢首包延迟
    draft_tokens: list[int] = field(default_factory=list)  # 完整 draft 序列
    draft_latency_ms: float = 0.0       # 完整 draft 延迟
    success: bool = False
    verified: bool = False              # 云端是否已 verify
    corrected_token: Optional[int] = None  # 云端修正后的 token (如果首 token 错了)
    error: str = ""


class DraftPivotEngine:
    """端侧 Draft 分层前向 -- 抢 TTFT 首包.

    用法:
        engine = DraftPivotEngine(omlx, pivot_layer=6)
        result = await engine.stream_pivot_then_draft(prompt_ids, draft_n=8)

    集成到 edge_first_proxy:
        # cache miss 时
        result = await pivot_engine.stream_pivot_then_draft(prompt_ids)
        if result.success:
            # 立即 SSE 发送首 token
            yield sse_event("first_token", token=result.first_token)
            # 等 draft 完成 → 上云 verify
            cloud_result = await cloud_verify(result.draft_tokens)
            if cloud_result.corrected_token is not None:
                yield sse_event("correction", token=cloud_result.corrected_token)
    """

    def __init__(
        self,
        omlx: OMLXRuntime,
        pivot_layer: int = 6,
        pivot_confidence_threshold: float = 0.7,
    ):
        """初始化 Pivot 引擎.

        Args:
            omlx: oMLX Runtime 实例
            pivot_layer: 默认 pivot 层 (6 = 前 6 层后抢首包)
            pivot_confidence_threshold: 置信度低于此值不抢首包
        """
        self.omlx = omlx
        self.pivot_layer = pivot_layer
        self.confidence_threshold = pivot_confidence_threshold

        # 统计
        self._stats = {
            "total_pivots": 0,
            "successful_pivots": 0,   # 云端 verify 通过
            "corrected_pivots": 0,    # 云端修正
            "failed_pivots": 0,       # 失败
            "total_pivot_ms": 0.0,
            "total_draft_ms": 0.0,
        }

    async def stream_pivot_then_draft(
        self,
        prompt_ids: list[int],
        draft_n: int = 8,
        confidence: float = 1.0,
    ) -> AsyncIterator[DraftEvent]:
        """流式输出首包, 然后继续生成 draft 序列.

        Yields:
            DraftEvent:
              - type="first_token": pivot 抢出的首 token
              - type="draft_token": 后续 draft token
              - type="draft_sequence": 完整 draft 序列
              - type="error": 错误
        """
        # 置信度检查
        use_pivot = confidence >= self.confidence_threshold
        actual_pivot = self.pivot_layer if use_pivot else -1

        if not use_pivot:
            logger.info(
                f"[draft-pivot] Skipping pivot (confidence={confidence:.2f} < {self.confidence_threshold})"
            )

        # 调用 oMLX 异步流式生成
        async for event in self.omlx.forward_draft_async(
            model_name="",  # 由 omlx 内部管理
            prompt_ids=prompt_ids,
            draft_n=draft_n,
            pivot_layer=actual_pivot,
        ):
            if event.type == "first_token":
                self._stats["total_pivots"] += 1
                self._stats["total_pivot_ms"] += event.latency_ms
                logger.info(
                    f"[draft-pivot] First token: id={event.token_id} "
                    f"at layer {event.pivot_layer} in {event.latency_ms:.1f}ms"
                )
            elif event.type == "draft_sequence":
                self._stats["total_draft_ms"] += event.latency_ms

            yield event

    async def generate(
        self,
        prompt_ids: list[int],
        draft_n: int = 8,
        confidence: float = 1.0,
    ) -> PivotResult:
        """生成 pivot 结果 (非流式, 等全部完成).

        用于不需要流式输出的场景.
        """
        t0 = time.time()
        result = PivotResult(pivot_layer=self.pivot_layer if confidence >= self.confidence_threshold else -1)

        try:
            draft_tokens = []
            async for event in self.stream_pivot_then_draft(
                prompt_ids, draft_n, confidence
            ):
                if event.type == "first_token":
                    result.first_token = event.token_id
                    result.pivot_latency_ms = event.latency_ms
                elif event.type == "draft_token":
                    draft_tokens.append(event.token_id)
                elif event.type == "draft_sequence":
                    draft_tokens = event.sequence
                    result.draft_latency_ms = event.latency_ms
                elif event.type == "error":
                    result.error = event.error
                    return result

            result.draft_tokens = draft_tokens
            result.success = True
            result.draft_latency_ms = (time.time() - t0) * 1000

        except Exception as e:
            result.error = str(e)
            result.draft_latency_ms = (time.time() - t0) * 1000
            logger.error(f"[draft-pivot] Generate failed: {e}")

        return result

    def record_verification(
        self,
        pivot_token: int,
        cloud_token: int,
    ) -> bool:
        """记录云端 verify 结果.

        Args:
            pivot_token: pivot 抢出的 token
            cloud_token: 云端 verify 的正确 token

        Returns:
            True if pivot token was correct
        """
        self._stats["total_pivots"] += 1
        if pivot_token == cloud_token:
            self._stats["successful_pivots"] += 1
            return True
        else:
            self._stats["corrected_pivots"] += 1
            logger.info(
                f"[draft-pivot] Corrected: pivot={pivot_token} → cloud={cloud_token}"
            )
            return False

    def get_pivot_accept_rate(self) -> float:
        """获取 pivot 首包接受率."""
        total = self._stats["successful_pivots"] + self._stats["corrected_pivots"]
        if total == 0:
            return 0.0
        return self._stats["successful_pivots"] / total

    def get_stats(self) -> dict:
        """获取统计."""
        return {
            **self._stats,
            "pivot_accept_rate": round(self.get_pivot_accept_rate(), 3),
            "avg_pivot_ms": round(
                self._stats["total_pivot_ms"] / max(self._stats["total_pivots"], 1), 1
            ),
            "avg_draft_ms": round(
                self._stats["total_draft_ms"] / max(self._stats["total_pivots"], 1), 1
            ),
        }


if __name__ == "__main__":
    # 自测 (mock oMLX)
    runtime = OMLXRuntime(model_path="")
    
    class MockModelInfo:
        num_layers = 12
        hidden_size = 512
        vocab_size = 32000
        is_moe = False
        num_experts = 0
        experts_per_tok = 0
    
    runtime.load_model_metadata(MockModelInfo())

    engine = DraftPivotEngine(runtime, pivot_layer=6)

    # 非流式测试
    import asyncio
    result = asyncio.run(engine.generate([1, 2, 3, 4, 5], draft_n=4, confidence=0.9))
    print(f"Pivot result: success={result.success}")
    print(f"  first_token={result.first_token}")
    print(f"  pivot_latency={result.pivot_latency_ms:.1f}ms")
    print(f"  draft_tokens={result.draft_tokens}")
    print(f"  draft_latency={result.draft_latency_ms:.1f}ms")

    # 模拟 verify
    engine.record_verification(result.first_token, result.first_token)
    print(f"\nStats: {engine.get_stats()}")
