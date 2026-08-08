#!/usr/bin/env python3
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


app = FastAPI(title="Mock Cloud OpenAI Backend")
LOG_DIR = Path(__file__).resolve().parents[2] / "var" / "benchmark" / "mock_cloud_openai"
REQUEST_LOG = LOG_DIR / "requests.jsonl"


def _extract_text(messages: list[dict[str, Any]], role: str) -> str:
    for item in reversed(messages):
        if str(item.get("role") or "") == role:
            return str(item.get("content") or "")
    return ""


def _token_chunks(text: str) -> list[str]:
    words = [part for part in text.split(" ") if part]
    if not words:
        return [text]
    return [word + (" " if i < len(words) - 1 else "") for i, word in enumerate(words)]


def _append_request_log(entry: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with REQUEST_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = list(body.get("messages") or [])
    extra_body = dict(body.get("extra_body") or {})
    stream = bool(body.get("stream", False))
    max_tokens = int(body.get("max_tokens", 16) or 16)
    user_text = _extract_text(messages, "user")
    assistant_prefix = _extract_text(messages, "assistant")
    acceptance_case = str(
        body.get("cgc_acceptance_case")
        or extra_body.get("cgc_acceptance_case")
        or ""
    )
    route_override = str(
        body.get("cgc_route_override")
        or extra_body.get("cgc_route_override")
        or ""
    )
    route_override_reason = str(
        body.get("cgc_route_override_reason")
        or extra_body.get("cgc_route_override_reason")
        or ""
    )
    content = f"{assistant_prefix}[CONTINUED_OK max={max_tokens}]"
    behavior = "normal"
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    if "FORCE_500" in user_text:
        behavior = "force_500"
        _append_request_log(
            {
                "ts": time.time(),
                "request_id": req_id,
                "case": acceptance_case,
                "behavior": behavior,
                "status": 500,
                "stream": stream,
                "max_tokens": max_tokens,
                "route_override": route_override,
                "route_override_reason": route_override_reason,
                "assistant_prefix_chars": len(assistant_prefix),
                "user_text": user_text,
            }
        )
        return JSONResponse({"error": "forced backend failure"}, status_code=500)
    if "FORCE_TIMEOUT" in user_text:
        behavior = "force_timeout"
        await asyncio.sleep(5.0)
        content = f"{assistant_prefix}[TIMEOUT_RECOVERED max={max_tokens}]"
    _append_request_log(
        {
            "ts": time.time(),
            "request_id": req_id,
            "case": acceptance_case,
            "behavior": behavior,
            "status": 200,
            "stream": stream,
            "max_tokens": max_tokens,
            "route_override": route_override,
            "route_override_reason": route_override_reason,
            "assistant_prefix_chars": len(assistant_prefix),
            "user_text": user_text,
            "response_text": content,
        }
    )
    if not stream:
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(body.get("model") or "mock-cloud"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": len(user_text.split()), "completion_tokens": len(content.split()), "total_tokens": len(user_text.split()) + len(content.split())},
        }

    async def event_stream():
        yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': str(body.get('model') or 'mock-cloud'), 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n".encode("utf-8")
        for piece in _token_chunks(content):
            yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': str(body.get('model') or 'mock-cloud'), 'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}]})}\n\n".encode("utf-8")
            await asyncio.sleep(0.02)
        yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': str(body.get('model') or 'mock-cloud'), 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=19000, log_level="info")
