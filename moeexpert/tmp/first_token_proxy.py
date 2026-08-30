#!/usr/bin/env python3
"""moeexpert first_token_proxy — 端侧小模型出 first token.

架构:
  edge_first_proxy (cache miss) → first_token_proxy → turbo-fieldfare (gemma4, Mac Metal)
                                                              ↓
                                                       POST /v1/chat/completions
                                                       stream=true, max_tokens=2, temperature=0
                                                              ↓
                                                       第一个 content chunk → first token

替代原 edge_first_proxy 的 tokenizer/cache 猜测,用真实端侧推理出 first token。
TTFT 目标: <80ms (Mac Metal prefill 短 prompt) vs tokenizer 猜测 <10ms 但不准。

用法:
  python3 first_token_proxy.py --port 30010 --edge-url http://127.0.0.1:8080

环境变量:
  MOEEXPERT_EDGE_URL: turbo-fieldfare URL (默认 http://127.0.0.1:8080)
  MOEEXPERT_FIRST_TOKEN_PORT: 本服务端口 (默认 30010)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import aiohttp
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

EDGE_URL = os.environ.get("MOEEXPERT_EDGE_URL", "http://127.0.0.1:8080")
PORT = int(os.environ.get("MOEEXPERT_FIRST_TOKEN_PORT", "30010"))

app = FastAPI(title="moeexpert first_token_proxy")

_session: Optional[aiohttp.ClientSession] = None


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=50, keepalive_timeout=30),
            timeout=aiohttp.ClientTimeout(total=30, connect=2.0),
        )
    return _session


class FirstTokenRequest(BaseModel):
    messages: list
    model: str = "gemma-4-26b-a4b-it"
    max_tokens: int = 2


@app.get("/health")
async def health():
    """检查 turbo-fieldfare 可用性."""
    try:
        session = await _get_session()
        async with session.get(f"{EDGE_URL}/health", timeout=2.0) as r:
            upstream_ok = r.status == 200
    except Exception:
        upstream_ok = False
    return {"ok": upstream_ok, "edge_url": EDGE_URL, "service": "first_token_proxy"}


@app.post("/first_token")
async def first_token(req: FirstTokenRequest):
    """调 turbo-fieldfare 出 first token (stream, 取第一个 content chunk).

    返回:
      200: {first_token, ttfb_ms, elapsed_ms, method, edge_url}
      502: 上游非 200
      503: edge 不可达
    """
    t0 = time.monotonic()
    session = await _get_session()

    payload = {
        "model": req.model,
        "messages": req.messages,
        "max_tokens": max(req.max_tokens, 2),
        "temperature": 0.0,
        "stream": True,
    }

    first_token_text = None
    ttfb_ms = 0.0

    try:
        async with session.post(
            f"{EDGE_URL}/v1/chat/completions", json=payload, timeout=30.0
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return JSONResponse(
                    {
                        "first_token": None,
                        "error": f"upstream_{resp.status}",
                        "detail": body[:200],
                        "elapsed_ms": (time.monotonic() - t0) * 1000,
                        "method": "moeexpert",
                    },
                    status_code=502,
                )

            async for raw in resp.content:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content") or ""
                    if content:
                        first_token_text = content
                        ttfb_ms = (time.monotonic() - t0) * 1000
                        break
                except Exception:
                    continue
    except aiohttp.ClientConnectorError:
        return JSONResponse(
            {
                "first_token": None,
                "error": "edge_unreachable",
                "elapsed_ms": (time.monotonic() - t0) * 1000,
                "method": "moeexpert",
            },
            status_code=503,
        )
    except Exception as e:
        return JSONResponse(
            {
                "first_token": None,
                "error": str(e),
                "elapsed_ms": (time.monotonic() - t0) * 1000,
                "method": "moeexpert",
            },
            status_code=500,
        )

    elapsed_ms = (time.monotonic() - t0) * 1000
    return {
        "first_token": first_token_text,
        "ttfb_ms": round(ttfb_ms, 1),
        "elapsed_ms": round(elapsed_ms, 1),
        "method": "moeexpert",
        "edge_url": EDGE_URL,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="moeexpert first_token_proxy")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--edge-url", type=str, default=EDGE_URL)
    args = parser.parse_args()

    EDGE_URL = args.edge_url
    print(f"[first_token_proxy] edge_url={EDGE_URL}, port={args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
