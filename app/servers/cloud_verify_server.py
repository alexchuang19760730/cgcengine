#!/usr/bin/env python3
"""Cloud Verify Server — 接收端侧 draft tokens, 用 sglang 做 single forward pass 验证.

路径 B 核心组件: 端侧 draft + 云端 verify.
任何 sglang 模型都能 verify, 不需要 EAGLE/NEXTN/DSPARK 支持.

工作原理:
  1. 接收 {prompt_ids, draft_tokens} from edge
  2. 发送 input_ids = prompt_ids + draft_tokens 到 sglang /generate
     - return_logprob=true, top_logprobs_num=1
  3. 解析 meta_info.input_top_logprobs
     - 每个位置的 top-1 token_id = 模型在该位置的 argmax 预测
  4. 逐位对比 draft token vs 模型 argmax
     - 匹配 → accept
     - 不匹配 → reject, 返回 corrected token
  5. 返回 {accepted_tokens, rejected_at, corrected_token, accepted_count}

部署: python3 cloud_verify_server.py --port 30060 --sglang-url http://127.0.0.1:30003
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

logger = logging.getLogger("cloud_verify")

app = FastAPI(title="CGC Cloud Verify Server", version="1.0")

# Global config
SGLANG_URL = "http://127.0.0.1:30003"
HTTP_TIMEOUT = 30.0

# Persistent HTTP client for connection pooling
_http_client: Optional[httpx.AsyncClient] = None

async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _http_client

# Stats
_stats = {
    "total_verifies": 0,
    "total_draft_tokens": 0,
    "total_accepted": 0,
    "total_rejected": 0,
    "total_verify_ms": 0.0,
    "total_sglang_ms": 0.0,
}


@app.post("/verify")
async def verify_draft(request: Request):
    """验证 draft tokens.

    Request body:
    {
        "prompt_ids": [int, ...],     # prompt token IDs
        "draft_tokens": [int, ...],   # draft token IDs to verify
        "model": "default",           # optional model name
        "temperature": 0.0,           # optional, 0.0 = deterministic
    }

    Response:
    {
        "accepted_tokens": [int, ...],
        "accepted_count": int,
        "rejected_at": int,           # -1 = all accepted
        "corrected_token": int,       # -1 = no correction needed
        "verify_latency_ms": float,
        "sglang_latency_ms": float,
        "success": bool,
        "error": str,
    }
    """
    t0 = time.time()
    data = await request.json()

    prompt_ids = data.get("prompt_ids", [])
    draft_tokens = data.get("draft_tokens", [])
    temperature = data.get("temperature", 0.0)

    if not draft_tokens:
        return JSONResponse({
            "accepted_tokens": [],
            "accepted_count": 0,
            "rejected_at": -1,
            "corrected_token": -1,
            "verify_latency_ms": (time.time() - t0) * 1000,
            "success": True,
            "error": "",
        })

    if not prompt_ids:
        return JSONResponse({
            "accepted_tokens": [],
            "accepted_count": 0,
            "rejected_at": 0,
            "corrected_token": -1,
            "verify_latency_ms": (time.time() - t0) * 1000,
            "success": False,
            "error": "prompt_ids is empty",
        })

    # Send prompt + draft tokens to sglang with logprobs
    all_ids = prompt_ids + draft_tokens
    prompt_len = len(prompt_ids)

    payload = {
        "input_ids": all_ids,
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": temperature,
        },
        "return_logprob": True,
        "logprob_start_len": 0,  # Get all logprobs (logprob_start_len truncation breaks index mapping)
        "top_logprobs_num": 1,
    }

    sglang_t0 = time.time()
    try:
        client = await get_http_client()
        resp = await client.post(
            f"{SGLANG_URL}/generate",
            json=payload,
        )
        sglang_ms = (time.time() - sglang_t0) * 1000

        if resp.status_code != 200:
            logger.error(f"sglang error: {resp.status_code} {resp.text[:200]}")
            return JSONResponse({
                "accepted_tokens": [],
                "accepted_count": 0,
                "rejected_at": 0,
                "corrected_token": -1,
                "verify_latency_ms": (time.time() - t0) * 1000,
                "sglang_latency_ms": sglang_ms,
                "success": False,
                "error": f"sglang returned {resp.status_code}",
            })

        result = resp.json()
        meta = result.get("meta_info", {})

        # Parse input_top_logprobs
        # Format: list of entries, each entry is null or [[logprob, token_id, text], ...]
        input_top_logprobs = meta.get("input_top_logprobs", [])

        # Also get output_ids (sglang's generated token, as fallback)
        output_ids = result.get("output_ids", [])

        # Verify each draft token
        accepted_tokens = []
        rejected_at = -1
        corrected_token = -1

        for k, draft_tok in enumerate(draft_tokens):
            idx = prompt_len + k  # Position in input_top_logprobs

            if idx >= len(input_top_logprobs):
                # Not enough logprob data, can't verify this position
                logger.warning(f"Not enough logprob data at position {idx}")
                break

            top_entry = input_top_logprobs[idx]

            if top_entry is None or len(top_entry) == 0:
                # No prediction available for this position (e.g., first token)
                # Can't verify, assume reject
                rejected_at = k
                # Use sglang's output as corrected token
                corrected_token = output_ids[0] if output_ids else -1
                break

            # top_entry[0] = [logprob, token_id, text]
            top1 = top_entry[0]
            top1_token_id = top1[1]  # token_id is at index 1

            if top1_token_id == draft_tok:
                accepted_tokens.append(draft_tok)
            else:
                # Reject at this position, return corrected token
                rejected_at = k
                corrected_token = top1_token_id
                break

        # If all accepted, also include sglang's generated token (bonus)
        if rejected_at == -1 and output_ids:
            # All draft tokens accepted, add the generated token too
            pass  # Don't add - the caller will request more drafts

        verify_ms = (time.time() - t0) * 1000

        # Update stats
        _stats["total_verifies"] += 1
        _stats["total_draft_tokens"] += len(draft_tokens)
        _stats["total_accepted"] += len(accepted_tokens)
        _stats["total_rejected"] += len(draft_tokens) - len(accepted_tokens)
        _stats["total_verify_ms"] += verify_ms
        _stats["total_sglang_ms"] += sglang_ms

        return JSONResponse({
            "accepted_tokens": accepted_tokens,
            "accepted_count": len(accepted_tokens),
            "rejected_at": rejected_at,
            "corrected_token": corrected_token,
            "verify_latency_ms": verify_ms,
            "sglang_latency_ms": sglang_ms,
            "success": True,
            "error": "",
        })

    except Exception as e:
        logger.error(f"Verify failed: {e}", exc_info=True)
        return JSONResponse({
            "accepted_tokens": [],
            "accepted_count": 0,
            "rejected_at": 0,
            "corrected_token": -1,
            "verify_latency_ms": (time.time() - t0) * 1000,
            "sglang_latency_ms": (time.time() - sglang_t0) * 1000,
            "success": False,
            "error": str(e),
        })


@app.post("/verify_text")
async def verify_draft_text(request: Request):
    """Text-based verify — 接收 prompt text + draft token IDs.

    用于端侧只有 text prompt 的场景.
    会先用 sglang /tokenize 转换 text → ids, 再走 /verify 逻辑.
    """
    data = await request.json()
    prompt_text = data.get("prompt_text", "")
    draft_tokens = data.get("draft_tokens", [])

    if not prompt_text or not draft_tokens:
        return JSONResponse({
            "accepted_tokens": [],
            "accepted_count": 0,
            "rejected_at": 0,
            "corrected_token": -1,
            "success": False,
            "error": "prompt_text or draft_tokens is empty",
        })

    # Tokenize using sglang
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            # Try /tokenize endpoint
            resp = await client.post(
                f"{SGLANG_URL}/tokenize",
                json={"text": prompt_text},
            )
            if resp.status_code == 200:
                tok_data = resp.json()
                prompt_ids = tok_data.get("input_ids", tok_data.get("token_ids", []))
            else:
                # Fallback: use /generate with max_new_tokens=0 to get token count
                # Or use the OpenAI completions API
                logger.warning(f"Tokenize failed ({resp.status_code}), using /v1/completions")
                resp = await client.post(
                    f"{SGLANG_URL}/v1/completions",
                    json={
                        "model": "default",
                        "prompt": prompt_text,
                        "max_tokens": 0,
                        "echo": True,
                    },
                )
                if resp.status_code == 200:
                    result = resp.json()
                    usage = result.get("usage", {})
                    prompt_ids_count = usage.get("prompt_tokens", 0)
                    # We don't have the actual IDs, so we need to use a different approach
                    # Use /generate with text and extract input_ids from response
                    resp = await client.post(
                        f"{SGLANG_URL}/generate",
                        json={
                            "text": prompt_text,
                            "sampling_params": {"max_new_tokens": 0},
                        },
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        # input_ids might not be in response...
                        # This is a fallback, might not work
                        prompt_ids = result.get("input_ids", [])
                    else:
                        return JSONResponse({
                            "success": False,
                            "error": f"Cannot tokenize prompt: {resp.status_code}",
                        })
                else:
                    return JSONResponse({
                        "success": False,
                        "error": f"Tokenize fallback failed: {resp.status_code}",
                    })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Tokenize error: {e}",
        })

    # Now use the token IDs to verify
    # Reuse the /verify logic
    all_ids = prompt_ids + draft_tokens
    prompt_len = len(prompt_ids)

    payload = {
        "input_ids": all_ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": 0,  # Get all logprobs for correct index mapping
        "top_logprobs_num": 1,
    }

    t0 = time.time()
    try:
        client = await get_http_client()
        resp = await client.post(f"{SGLANG_URL}/generate", json=payload)
        sglang_ms = (time.time() - t0) * 1000

        if resp.status_code != 200:
            return JSONResponse({
                "success": False,
                "error": f"sglang returned {resp.status_code}",
            })

        result = resp.json()
        meta = result.get("meta_info", {})
        input_top_logprobs = meta.get("input_top_logprobs", [])
        output_ids = result.get("output_ids", [])

        accepted_tokens = []
        rejected_at = -1
        corrected_token = -1

        for k, draft_tok in enumerate(draft_tokens):
            idx = prompt_len + k
            if idx >= len(input_top_logprobs):
                break

            top_entry = input_top_logprobs[idx]
            if top_entry is None or len(top_entry) == 0:
                rejected_at = k
                corrected_token = output_ids[0] if output_ids else -1
                break

            top1_token_id = top_entry[0][1]
            if top1_token_id == draft_tok:
                accepted_tokens.append(draft_tok)
            else:
                rejected_at = k
                corrected_token = top1_token_id
                break

        verify_ms = (time.time() - t0) * 1000
        return JSONResponse({
            "accepted_tokens": accepted_tokens,
            "accepted_count": len(accepted_tokens),
            "rejected_at": rejected_at,
            "corrected_token": corrected_token,
            "verify_latency_ms": verify_ms,
            "sglang_latency_ms": sglang_ms,
            "prompt_ids": prompt_ids,
            "success": True,
            "error": "",
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
        })


@app.get("/health")
async def health():
    """健康检查 + 统计."""
    total = _stats["total_accepted"] + _stats["total_rejected"]
    return {
        "status": "ok",
        "sglang_url": SGLANG_URL,
        "stats": {
            **_stats,
            "accept_rate": round(_stats["total_accepted"] / total, 3) if total > 0 else 0,
            "avg_verify_ms": round(_stats["total_verify_ms"] / max(_stats["total_verifies"], 1), 1),
            "avg_sglang_ms": round(_stats["total_sglang_ms"] / max(_stats["total_verifies"], 1), 1),
        },
    }


@app.post("/generate")
async def passthrough_generate(request: Request):
    """透传 sglang /generate — 用于 spec decode 完成后的正常生成."""
    data = await request.json()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(f"{SGLANG_URL}/generate", json=data)
    return JSONResponse(resp.json())


def main():
    global SGLANG_URL, HTTP_TIMEOUT

    parser = argparse.ArgumentParser(description="CGC Cloud Verify Server")
    parser.add_argument("--port", type=int, default=30060, help="Server port")
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30003",
                        help="sglang server URL")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout")
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()

    SGLANG_URL = args.sglang_url
    HTTP_TIMEOUT = args.timeout

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info(f"Starting Cloud Verify Server on port {args.port}")
    logger.info(f"sglang URL: {SGLANG_URL}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
