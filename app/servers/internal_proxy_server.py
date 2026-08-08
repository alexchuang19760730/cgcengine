from __future__ import annotations

import json
import os
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Dict

import requests
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_BASE_URL = str(os.environ.get("CGC_INTERNAL_PROXY_TARGET") or "http://127.0.0.1:8000").rstrip("/")
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

app = FastAPI(title="CGC Internal Proxy", description="Built-in protocol proxy for Claude/OpenAI/Ollama clients")


def _debug_report(hypothesis_id: str, msg: str, data=None) -> None:
    payload = data or {}
    try:
        env_file = (REPO_ROOT / ".dbg" / "edge-chain-connection.env").resolve()
        debug_url = "http://127.0.0.1:7777/event"
        session_id = "edge-chain-connection"
        if env_file.exists():
            content = env_file.read_text(encoding="utf-8")
            for line in content.splitlines():
                if line.startswith("DEBUG_SERVER_URL="):
                    debug_url = line.split("=", 1)[1].strip()
                elif line.startswith("DEBUG_SESSION_ID="):
                    session_id = line.split("=", 1)[1].strip()
        body = json.dumps(
            {
                "sessionId": session_id,
                "runId": "pre-fix",
                "hypothesisId": hypothesis_id,
                "location": "app/servers/internal_proxy_server.py",
                "msg": msg,
                "data": payload,
                "ts": int(time.time() * 1000),
            }
        ).encode("utf-8")
        urllib.request.urlopen(
            urllib.request.Request(
                debug_url,
                data=body,
                headers={"Content-Type": "application/json"},
            ),
            timeout=1,
        ).read()
    except Exception:
        pass


def _forward_headers(headers: Dict[str, str]) -> Dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _response_headers(upstream: requests.Response) -> Dict[str, str]:
    return {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _target_url(path: str, query_string: bytes) -> str:
    base = f"{TARGET_BASE_URL}/{path}".rstrip("/")
    query = query_string.decode("utf-8", errors="ignore")
    if query:
        return f"{base}?{query}"
    return base


@app.get("/")
@app.head("/")
async def root() -> Dict[str, str]:
    return {
        "status": "ok",
        "proxy_target": TARGET_BASE_URL,
    }


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_all(full_path: str, request: Request):
    body = await request.body()
    trace_id = str(request.headers.get("x-cgc-debug-trace-id") or "").strip() or f"proxy-{uuid.uuid4().hex[:12]}"
    target_url = _target_url(full_path, request.scope.get("query_string", b""))
    forward_headers = _forward_headers(dict(request.headers))
    forward_headers["x-cgc-debug-trace-id"] = trace_id
    # #region debug-point A:proxy-received
    _debug_report(
        "A",
        "[DEBUG] internal proxy received request",
        {
            "trace_id": trace_id,
            "method": request.method,
            "path": full_path,
            "target_url": target_url,
            "body_bytes": len(body),
            "request_content_type": request.headers.get("content-type", ""),
        },
    )
    # #endregion
    try:
        upstream = requests.request(
            request.method,
            target_url,
            headers=forward_headers,
            data=body,
            stream=True,
            timeout=(10, 600),
        )
    except Exception as exc:
        # #region debug-point A:proxy-error
        _debug_report(
            "A",
            "[DEBUG] internal proxy upstream request failed",
            {
                "trace_id": trace_id,
                "method": request.method,
                "path": full_path,
                "target_url": target_url,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        # #endregion
        raise

    response_headers = _response_headers(upstream)
    content_type = upstream.headers.get("content-type", "")
    # #region debug-point A:proxy-response
    _debug_report(
        "A",
        "[DEBUG] internal proxy received upstream response",
        {
            "trace_id": trace_id,
            "method": request.method,
            "path": full_path,
            "target_url": target_url,
            "status_code": upstream.status_code,
            "content_type": content_type,
        },
    )
    # #endregion
    if "text/event-stream" in content_type.lower():
        return StreamingResponse(
            upstream.iter_content(chunk_size=None),
            status_code=upstream.status_code,
            media_type=content_type,
            headers=response_headers,
        )

    upstream_body = upstream.content
    # #region debug-point A:proxy-body-read
    _debug_report(
        "A",
        "[DEBUG] internal proxy fully read upstream body",
        {
            "trace_id": trace_id,
            "method": request.method,
            "path": full_path,
            "target_url": target_url,
            "status_code": upstream.status_code,
            "body_bytes": len(upstream_body),
        },
    )
    # #endregion
    return Response(
        content=upstream_body,
        status_code=upstream.status_code,
        media_type=content_type or None,
        headers=response_headers,
    )
