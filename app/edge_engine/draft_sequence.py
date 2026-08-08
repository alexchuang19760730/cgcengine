#!/usr/bin/env python3
"""DraftSequenceEngine -- 端侧多 token draft + 上云 verify.

模式 B (path 2): 端侧生成完整 draft_n tokens → 上云 verify.
与 DraftPivotEngine 的区别: 不抢首包, 等 draft 完整生成后一次性上云.

适用场景:
  - Hermes 路由 confidence < 0.7 (不抢首包)
  - 网络 RTT < 50ms (云端 verify 快, 抢首包收益小)
  - 离线模式 (无云端, 仅本地 draft)

集成 parallel preflight:
  端侧 draft 和云端 prefill 并行启动,
  draft 完成后立即上云 verify, miss penalty → 0ms.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from app.edge_engine.omlx_runtime import OMLXRuntime, DraftResult

logger = logging.getLogger(__name__)

# Lazy import llama.cpp backend
_LAZY_LLAMACPP = None

def _get_llamacpp():
    global _LAZY_LLAMACPP
    if _LAZY_LLAMACPP is None:
        from app.edge_engine.llamacpp_draft import LlamaCppDraftBackend
        _LAZY_LLAMACPP = LlamaCppDraftBackend
    return _LAZY_LLAMACPP


@dataclass
class VerifyResult:
    """云端 verify 结果."""
    accepted_tokens: list[int] = field(default_factory=list)  # 云端接受的 token
    rejected_at: int = -1          # 第几个 token 被 reject (-1 = 全部接受)
    corrected_token: int = -1      # 修正后的 token (reject 位置)
    draft_tokens: list[int] = field(default_factory=list)
    draft_latency_ms: float = 0.0
    verify_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    accept_count: int = 0
    accept_rate: float = 0.0
    success: bool = False
    error: str = ""


class DraftSequenceEngine:
    """端侧 Draft 多 token 序列生成 + 上云 verify.

    用法:
        engine = DraftSequenceEngine(omlx, cloud_url="http://...")
        result = await engine.generate_and_send(prompt_ids, draft_n=8)

    集成到 edge_first_proxy:
        # cache miss, confidence < 0.7 → 模式 B
        result = await seq_engine.generate_and_send(prompt_ids)
        if result.accept_count > 0:
            # 流式输出 accepted tokens
            for token in result.accepted_tokens:
                yield sse_event("token", token=token)
            if result.rejected_at >= 0:
                yield sse_event("token", token=result.corrected_token)
                # 继续云端 decode
    """

    def __init__(
        self,
        omlx: Optional[OMLXRuntime] = None,
        cloud_url: str = "",
        parallel_preflight: bool = True,
        verify_timeout_ms: float = 5000,
        backend: str = "omlx",
        llamacpp_model: Optional[str] = None,
        llamacpp_port: int = 8082,
    ):
        """初始化 Draft Sequence 引擎.

        Args:
            omlx: oMLX Runtime 实例 (backend="omlx" 时必需)
            cloud_url: 云端 verify URL (sglang / load balancer)
            parallel_preflight: 是否并行启动云端 preflight
            verify_timeout_ms: verify 超时 (ms)
            backend: "omlx" 或 "llamacpp"
            llamacpp_model: GGUF 模型路径 (backend="llamacpp" 时使用)
            llamacpp_port: llama-server 端口
        """
        self.omlx = omlx
        self.cloud_url = cloud_url
        self.parallel_preflight = parallel_preflight
        self.verify_timeout_ms = verify_timeout_ms
        self.backend = backend
        self._llamacpp = None

        if backend == "llamacpp":
            LlamaCppDraftBackend = _get_llamacpp()
            self._llamacpp = LlamaCppDraftBackend(
                model_path=llamacpp_model,
                port=llamacpp_port,
                auto_start=True,
            )

        # 统计
        self._stats = {
            "total_drafts": 0,
            "total_accepted": 0,
            "total_rejected": 0,
            "total_draft_ms": 0.0,
            "total_verify_ms": 0.0,
            "parallel_preflight_hits": 0,
        }

    async def generate_and_send(
        self,
        prompt_ids: list[int],
        draft_n: int = 8,
        model_name: str = "",
        temperature: float = 0.0,
        prompt_text: str = "",
    ) -> VerifyResult:
        """端侧生成 draft → 立即上云 verify (parallel).

        步骤:
        1. 并行启动: 端侧 draft + 云端 preflight
        2. 等 draft 完成 → 发上云 verify
        3. 等云端 verify 结果

        Args:
            prompt_ids: prompt token IDs (omlx backend)
            draft_n: 生成多少 draft tokens
            model_name: 云端模型名
            temperature: 0.0 = 确定性解码
            prompt_text: 文本 prompt (llamacpp backend 必需)
        """
        t0 = time.time()
        result = VerifyResult()

        try:
            # 1. 并行启动
            tasks = []

            # 端侧 draft 生成
            draft_task = asyncio.create_task(
                self._generate_draft(prompt_ids, draft_n, model_name, prompt_text)
            )
            tasks.append(("draft", draft_task))

            # 云端 preflight (并行)
            cloud_task = None
            if self.parallel_preflight and self.cloud_url:
                cloud_task = asyncio.create_task(
                    self._cloud_preflight(prompt_ids, model_name)
                )
                tasks.append(("preflight", cloud_task))

            # 2. 等 draft 完成
            draft_result = await draft_task
            result.draft_tokens = draft_result.tokens
            result.draft_latency_ms = draft_result.draft_ms
            self._stats["total_drafts"] += 1
            self._stats["total_draft_ms"] += draft_result.latency_ms

            if not draft_result.success:
                result.error = f"Draft failed: {draft_result.error}"
                result.total_latency_ms = (time.time() - t0) * 1000
                return result

            logger.info(
                f"[draft-seq] Draft {len(draft_result.tokens)} tokens in "
                f"{draft_result.draft_ms:.1f}ms"
            )

            # 3. 上云 verify
            verify_task = asyncio.create_task(
                self._cloud_verify(
                    prompt_ids=prompt_ids,
                    draft_tokens=draft_result.tokens,
                    model_name=model_name,
                    temperature=temperature,
                )
            )

            # 如果 preflight 已完成, 标记为 hit
            if cloud_task is not None:
                try:
                    preflight_result = await asyncio.wait_for(cloud_task, timeout=0.001)
                    if preflight_result:
                        self._stats["parallel_preflight_hits"] += 1
                        logger.debug("[draft-seq] Preflight completed before draft")
                except asyncio.TimeoutError:
                    # preflight 还在跑, 没关系, verify 会等
                    pass

            # 等 verify 结果
            verify_result = await asyncio.wait_for(
                verify_task,
                timeout=self.verify_timeout_ms / 1000,
            )

            result.accepted_tokens = verify_result.get("accepted_tokens", [])
            result.rejected_at = verify_result.get("rejected_at", -1)
            result.corrected_token = verify_result.get("corrected_token", -1)
            result.verify_latency_ms = verify_result.get("latency_ms", 0)
            result.accept_count = len(result.accepted_tokens)
            result.accept_rate = result.accept_count / len(draft_result.tokens) if draft_result.tokens else 0
            result.success = True

            self._stats["total_accepted"] += result.accept_count
            self._stats["total_rejected"] += len(draft_result.tokens) - result.accept_count
            self._stats["total_verify_ms"] += result.verify_latency_ms

            result.total_latency_ms = (time.time() - t0) * 1000

            logger.info(
                f"[draft-seq] Verify: {result.accept_count}/{len(draft_result.tokens)} accepted "
                f"({result.accept_rate:.1%}) in {result.verify_latency_ms:.1f}ms, "
                f"total {result.total_latency_ms:.1f}ms"
            )

        except asyncio.TimeoutError:
            result.error = f"Verify timeout ({self.verify_timeout_ms}ms)"
            result.total_latency_ms = (time.time() - t0) * 1000
            logger.warning(f"[draft-seq] Verify timeout")
        except Exception as e:
            result.error = str(e)
            result.total_latency_ms = (time.time() - t0) * 1000
            logger.error(f"[draft-seq] Generate+verify failed: {e}")

        return result

    async def _generate_draft(
        self,
        prompt_ids: list[int],
        draft_n: int,
        model_name: str,
        prompt_text: str = "",
    ) -> DraftResult:
        """端侧生成 draft.

        backend="llamacpp": 用 llama-server HTTP API 生成 (真实推理)
        backend="omlx": 用 OMLXRuntime forward (在线程池中执行)
        """
        if self.backend == "llamacpp" and self._llamacpp:
            # llama.cpp 真实推理
            llamacpp_result = await self._llamacpp.generate_draft(
                prompt=prompt_text or " ".join(str(t) for t in prompt_ids),
                n_tokens=draft_n,
                temperature=0.0,
            )
            # 转换为 OMLXRuntime DraftResult 格式
            return DraftResult(
                tokens=llamacpp_result.token_ids or list(range(len(llamacpp_result.tokens))),
                latency_ms=llamacpp_result.latency_ms,
                draft_ms=llamacpp_result.latency_ms,
                success=llamacpp_result.error is None,
                error=llamacpp_result.error or "",
            )

        # oMLX backend (原逻辑)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.omlx.forward_draft(
                model_name=model_name,
                prompt_ids=prompt_ids,
                draft_n=draft_n,
                pivot_layer=-1,  # 模式 B 不抢首包
            )
        )

    async def _cloud_preflight(
        self,
        prompt_ids: list[int],
        model_name: str,
    ) -> bool:
        """云端 preflight: 预热云端 prefill (parallel).

        生产环境: POST /v1/completions with stream=false, max_tokens=1
        当前: mock 返回 True
        """
        if not self.cloud_url:
            return False

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model_name,
                    "prompt": prompt_ids,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": False,
                }
                async with session.post(
                    f"{self.cloud_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=2.0),
                ) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.debug(f"[draft-seq] Preflight failed: {e}")
            return False

    async def _cloud_verify(
        self,
        prompt_ids: list[int],
        draft_tokens: list[int],
        model_name: str,
        temperature: float = 0.0,
    ) -> dict:
        """上云 verify draft tokens.

        生产环境: 用 sglang EAGLE/NEXTN speculative decode API
        当前: mock verify (假设全部接受)
        """
        t0 = time.time()

        if not self.cloud_url:
            # Mock: 全部接受
            await asyncio.sleep(0.001)  # 模拟网络延迟
            return {
                "accepted_tokens": draft_tokens,
                "rejected_at": -1,
                "corrected_token": -1,
                "latency_ms": (time.time() - t0) * 1000,
            }

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # sglang speculative decode API
                payload = {
                    "model": model_name,
                    "prompt": prompt_ids,
                    "draft_tokens": draft_tokens,
                    "max_tokens": len(draft_tokens) + 10,
                    "temperature": temperature,
                    "stream": False,
                }
                async with session.post(
                    f"{self.cloud_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    if resp.status != 200:
                        return {
                            "accepted_tokens": [],
                            "rejected_at": 0,
                            "corrected_token": -1,
                            "latency_ms": (time.time() - t0) * 1000,
                        }
                    data = await resp.json()
                    # 解析 verify 结果
                    # sglang 返回的 choices[0].text 包含接受的 tokens
                    # 简化: 假设前 N 个被接受
                    accepted = draft_tokens  # 全部接受
                    return {
                        "accepted_tokens": accepted,
                        "rejected_at": -1,
                        "corrected_token": -1,
                        "latency_ms": (time.time() - t0) * 1000,
                    }
        except Exception as e:
            logger.warning(f"[draft-seq] Cloud verify failed: {e}")
            return {
                "accepted_tokens": [],
                "rejected_at": 0,
                "corrected_token": -1,
                "latency_ms": (time.time() - t0) * 1000,
            }

    async def generate_local_only(
        self,
        prompt_ids: list[int],
        draft_n: int = 8,
        model_name: str = "",
        prompt_text: str = "",
    ) -> DraftResult:
        """离线模式: 仅本地 draft, 不上云 verify.

        用于 Hermes 路由 mode="local_only" 或网络不可用时.
        """
        if self.backend == "llamacpp" and self._llamacpp:
            llamacpp_result = await self._llamacpp.generate_draft(
                prompt=prompt_text or " ".join(str(t) for t in prompt_ids),
                n_tokens=draft_n,
                temperature=0.0,
            )
            result = DraftResult(
                tokens=llamacpp_result.token_ids or list(range(len(llamacpp_result.tokens))),
                latency_ms=llamacpp_result.latency_ms,
                draft_ms=llamacpp_result.latency_ms,
                success=llamacpp_result.error is None,
                error=llamacpp_result.error or "",
            )
            self._stats["total_drafts"] += 1
            self._stats["total_draft_ms"] += result.latency_ms
            self._stats["total_accepted"] += len(result.tokens)
            return result

        # oMLX backend
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.omlx.forward_draft(
                model_name=model_name,
                prompt_ids=prompt_ids,
                draft_n=draft_n,
                pivot_layer=-1,
            )
        )
        self._stats["total_drafts"] += 1
        self._stats["total_draft_ms"] += result.latency_ms
        self._stats["total_accepted"] += len(result.tokens)  # 本地模式全部"接受"
        return result

    def get_stats(self) -> dict:
        """获取统计."""
        total = self._stats["total_accepted"] + self._stats["total_rejected"]
        return {
            **self._stats,
            "accept_rate": round(
                self._stats["total_accepted"] / total if total > 0 else 0, 3
            ),
            "avg_draft_ms": round(
                self._stats["total_draft_ms"] / max(self._stats["total_drafts"], 1), 1
            ),
            "avg_verify_ms": round(
                self._stats["total_verify_ms"] / max(self._stats["total_drafts"], 1), 1
            ),
        }


if __name__ == "__main__":
    # 自测
    runtime = OMLXRuntime(model_path="")
    
    class MockModelInfo:
        num_layers = 12
        hidden_size = 512
        vocab_size = 32000
        is_moe = False
        num_experts = 0
        experts_per_tok = 0
    
    runtime.load_model_metadata(MockModelInfo())

    engine = DraftSequenceEngine(runtime, cloud_url="", parallel_preflight=True)

    # 测试生成 + verify (mock cloud)
    result = asyncio.run(engine.generate_and_send([1, 2, 3, 4, 5], draft_n=4))
    print(f"Verify result: success={result.success}")
    print(f"  draft_tokens={result.draft_tokens}")
    print(f"  accepted={result.accepted_tokens}")
    print(f"  accept_rate={result.accept_rate:.1%}")
    print(f"  draft_latency={result.draft_latency_ms:.1f}ms")
    print(f"  verify_latency={result.verify_latency_ms:.1f}ms")
    print(f"  total_latency={result.total_latency_ms:.1f}ms")

    print(f"\nStats: {engine.get_stats()}")
