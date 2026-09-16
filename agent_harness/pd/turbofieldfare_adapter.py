#!/usr/bin/env python3
"""TurboFieldfare HTTP client — Python 側調用 Mac 端的 emit/resume 端點.

用法:
  client = TurboFieldfareClient(base_url="http://192.168.1.10:8080")
  # Mac A (Gemma4 prefill)
  resp = await client.emit(EmitRequest(prompt="Hello"))
  hidden_tensor = resp.packet.to_tensor()  # [seq_len, 2816]

  # Mac B (Qwen3.6 decode)
  async for chunk in client.resume_stream(ResumeRequest(...)):
      print(chunk, end="", flush=True)
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

from .protocol import (
    EmitRequest,
    EmitResponse,
    HiddenStatePacket,
    ModelInfo,
    ResumeRequest,
    SourceModel,
    TargetModel,
)

logger = logging.getLogger(__name__)


class TurboFieldfareError(Exception):
    """TurboFieldfare 調用失敗."""


class TurboFieldfareClient:
    """異步 HTTP client, 調用 TurboFieldfare 的 /v1/cgc/* 端點.

    Args:
        base_url: TurboFieldfare 服務地址 (e.g. "http://192.168.1.10:8080")
        timeout: 請求超時 (秒). prefill 長文本時可能需要較長.
        trust_env: False = 不讀系統代理 (避免本地代理干擾)
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        trust_env: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            trust_env=trust_env,
        )

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    # ------------------------------------------------------------------
    # 健康檢查
    # ------------------------------------------------------------------
    async def health(self) -> bool:
        """檢查 TurboFieldfare 服務是否在線."""
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """列出可用模型."""
        resp = await self._client.get("/v1/models")
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    # ------------------------------------------------------------------
    # POST /v1/cgc/emit — Gemma4 prefill + emit 末層 hidden state
    # ------------------------------------------------------------------
    async def emit(self, request: EmitRequest) -> EmitResponse:
        """調用 Mac A 的 emit 端點, prefill 並返回 hidden state.

        Args:
            request: EmitRequest, 包含 prompt

        Returns:
            EmitResponse, 包含 HiddenStatePacket (末層 hidden state)

        Raises:
            TurboFieldfareError: emit 失敗
        """
        logger.info(
            "emit: request_id=%s prompt_len=%d",
            request.request_id,
            len(request.prompt),
        )
        resp = await self._client.post(
            "/v1/cgc/emit",
            json=request.to_dict(),
        )
        if resp.status_code != 200:
            raise TurboFieldfareError(
                f"emit failed: HTTP {resp.status_code} {resp.text}"
            )
        data = resp.json()
        emit_resp = EmitResponse.from_dict(data)
        if not emit_resp.success:
            raise TurboFieldfareError(f"emit returned error: {emit_resp.error}")
        logger.info(
            "emit done: seq_len=%d hidden_dim=%d latency=%.1fms",
            emit_resp.packet.seq_len,
            emit_resp.packet.hidden_dim,
            emit_resp.prefill_latency_ms,
        )
        return emit_resp

    # ------------------------------------------------------------------
    # POST /v1/cgc/resume — Qwen3.6 recv hidden + decode (SSE stream)
    # ------------------------------------------------------------------
    async def resume_stream(self, request: ResumeRequest) -> AsyncIterator[str]:
        """調用 Mac B 的 resume 端點, 流式接收 decode 的 token.

        Args:
            request: ResumeRequest, 包含翻譯後的 hidden_state_b64

        Yields:
            str: 每個 decode 的 token (或 chunk content)

        Raises:
            TurboFieldfareError: resume 失敗
        """
        logger.info(
            "resume: request_id=%s seq_len=%d max_tokens=%d",
            request.request_id,
            request.seq_len,
            request.max_tokens,
        )
        async with self._client.stream(
            "POST",
            "/v1/cgc/resume",
            json=request.to_dict(),
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise TurboFieldfareError(
                    f"resume failed: HTTP {resp.status_code} {body.decode()}"
                )
            # 解析 SSE stream (OpenAI chunk 格式)
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    logger.warning("unparseable SSE line: %s", payload)
                    continue

    # ------------------------------------------------------------------
    # 便捷方法: 直接傳 hidden tensor 做 resume
    # ------------------------------------------------------------------
    async def resume_tensor(
        self,
        hidden_state,  # torch.Tensor [seq_len, hidden_dim]
        model_info: ModelInfo,
        max_tokens: int = 512,
        temperature: float = 0.7,
        request_id: str = "",
    ) -> AsyncIterator[str]:
        """直接傳 tensor 調用 resume (自動編碼)."""
        from .protocol import encode_hidden_state

        request = ResumeRequest(
            hidden_state_b64=encode_hidden_state(hidden_state),
            seq_len=hidden_state.shape[0],
            hidden_dim=hidden_state.shape[1],
            max_tokens=max_tokens,
            temperature=temperature,
            request_id=request_id,
        )
        async for token in self.resume_stream(request):
            yield token


# ---------------------------------------------------------------------------
# 預設實例工廠 (根據 .env 配置)
# ---------------------------------------------------------------------------
def make_emit_client(
    host: str | None = None,
    port: int | None = None,
) -> TurboFieldfareClient:
    """創建連接 Mac A (Gemma4 prefill) 的 client.

    環境變量:
        TF_EMIT_HOST (預設 127.0.0.1)
        TF_EMIT_PORT (預設 8080)
    """
    import os

    host = host or os.getenv("TF_EMIT_HOST", "127.0.0.1")
    port = port or int(os.getenv("TF_EMIT_PORT", "8080"))
    return TurboFieldfareClient(f"http://{host}:{port}")


def make_resume_client(
    host: str | None = None,
    port: int | None = None,
) -> TurboFieldfareClient:
    """創建連接 Mac B (Qwen3.6 decode) 的 client.

    環境變量:
        TF_RESUME_HOST (預設 127.0.0.1)
        TF_RESUME_PORT (預設 8081)
    """
    import os

    host = host or os.getenv("TF_RESUME_HOST", "127.0.0.1")
    port = port or int(os.getenv("TF_RESUME_PORT", "8081"))
    return TurboFieldfareClient(f"http://{host}:{port}")
