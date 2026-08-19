#!/usr/bin/env python3
"""PD coordinator — FastAPI 服務, 串接 emit → MoT-h → resume 完整管線.

管線:
  client → /v1/cgc/generate
    → Mac A: POST /v1/cgc/emit (Gemma4 prefill, 取末層 hidden)
    → MoT-h 翻譯 (2816 → 2048)
    → Mac B: POST /v1/cgc/resume (Qwen3.6 decode, SSE stream)
    → 透傳 SSE 給 client

啟動:
  uvicorn cgc_engine.pd.coordinator:app --host 0.0.0.0 --port 9000

環境變量:
  TF_EMIT_HOST=192.168.1.10    # Mac A (Gemma4)
  TF_EMIT_PORT=8080
  TF_RESUME_HOST=192.168.1.20  # Mac B (Qwen3.6)
  TF_RESUME_PORT=8081
  MOT_H_CHECKPOINT=            # MoT-h 權重路徑 (空 = passthrough 模式)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# MoT-h 在 CGC_Phase2/mot_h/ 下, 動態加入 path
_MOT_H_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "CGC_Phase2", "mot_h")
_MOT_H_PATH = os.path.abspath(_MOT_H_PATH)
if _MOT_H_PATH not in sys.path:
    sys.path.insert(0, _MOT_H_PATH)

from mot_h import MoTHConfig, MoTH  # noqa: E402
from .protocol import (
    EmitRequest,
    EmitResponse,
    HiddenStatePacket,
    ModelInfo,
    PDMode,
    ResumeRequest,
    SourceModel,
    TargetModel,
    encode_hidden_state,
)
from .turbofieldfare_adapter import TurboFieldfareClient, make_emit_client, make_resume_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 全域狀態
# ---------------------------------------------------------------------------
class AppState:
    """應用全局狀態."""
    emit_client: Optional[TurboFieldfareClient] = None
    resume_client: Optional[TurboFieldfareClient] = None
    mot_h: Optional[MoTH] = None
    mot_h_checkpoint: Optional[str] = None
    source_model: SourceModel = SourceModel.GEMMA4_26B_A4B
    target_model: TargetModel = TargetModel.QWEN36_35B_A3B


state = AppState()


# ---------------------------------------------------------------------------
# MoT-h 載入
# ---------------------------------------------------------------------------
def load_mot_h(checkpoint_path: str | None) -> Optional[MoTH]:
    """載入 MoT-h 翻譯器.

    Args:
        checkpoint_path: 權重路徑. None 或空 = passthrough 模式 (不翻譯)

    Returns:
        MoTH 實例或 None
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning("MoT-h checkpoint not found, using passthrough mode")
        return None

    config = MoTHConfig(
        src_hidden_size=state.source_model.hidden_size,   # 2816
        tgt_hidden_size=state.target_model.hidden_size,    # 2048
    )
    model = MoTH(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    logger.info("MoT-h loaded from %s", checkpoint_path)
    return model


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    state.emit_client = make_emit_client()
    state.resume_client = make_resume_client()
    state.mot_h_checkpoint = os.getenv("MOT_H_CHECKPOINT", "")
    state.mot_h = load_mot_h(state.mot_h_checkpoint)

    # 健康檢查
    emit_ok = await state.emit_client.health()
    resume_ok = await state.resume_client.health()
    logger.info(
        "startup: emit_client=%s resume_client=%s mot_h=%s",
        "ok" if emit_ok else "FAIL",
        "ok" if resume_ok else "FAIL",
        "loaded" if state.mot_h else "passthrough",
    )

    yield

    # shutdown
    if state.emit_client:
        await state.emit_client.close()
    if state.resume_client:
        await state.resume_client.close()


app = FastAPI(title="CGC PD Coordinator", lifespan=lifespan)


# ---------------------------------------------------------------------------
# API 請求/響應模型
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    """POST /v1/cgc/generate 請求."""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = True
    mode: str = "translate"  # translate | passthrough | collect


class GenerateResponse(BaseModel):
    """POST /v1/cgc/generate 非流式響應."""
    text: str
    emit_latency_ms: float
    translate_latency_ms: float
    resume_latency_ms: float
    total_latency_ms: float


# ---------------------------------------------------------------------------
# 管線核心
# ---------------------------------------------------------------------------
async def run_pipeline(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    mode: str,
) -> AsyncIterator[dict]:
    """執行 emit → translate → resume 管線, yield SSE chunk.

    Yields:
        dict: OpenAI 格式的 SSE chunk
    """
    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()

    # ── Step 1: emit (Mac A — Gemma4 prefill) ──────────────────────
    emit_req = EmitRequest(
        prompt=prompt,
        request_id=request_id,
    )
    emit_resp: EmitResponse = await state.emit_client.emit(emit_req)
    emit_latency = (time.time() - t0) * 1000
    hidden_src = emit_resp.packet.to_tensor()  # [seq_len, 2816]
    logger.info(
        "[%s] emit: seq_len=%d hidden_dim=%d %.1fms",
        request_id, hidden_src.shape[0], hidden_src.shape[1], emit_latency,
    )

    # ── Step 2: translate (MoT-h) ──────────────────────────────────
    t1 = time.time()
    if mode == "passthrough":
        # 同模型直傳 (需 src/tgt hidden_dim 相同, 否則報錯)
        if hidden_src.shape[1] != state.target_model.hidden_size:
            raise ValueError(
                f"passthrough mode requires same hidden_dim, "
                f"got src={hidden_src.shape[1]} tgt={state.target_model.hidden_size}"
            )
        hidden_tgt = hidden_src
    elif mode == "collect":
        # 只採集, 不 decode — 返回 source hidden state 給採集腳本
        yield {
            "choices": [{"delta": {}, "finish_reason": "collect"}],
            "packet": emit_resp.packet.to_dict(),
        }
        return
    else:
        # translate mode: MoT-h 翻譯
        if state.mot_h is None:
            # 無 MoT-h 權重, fallback: 截斷/補零到 target hidden_dim
            logger.warning("MoT-h not loaded, using zero-pad fallback (NOT for production)")
            tgt_dim = state.target_model.hidden_size
            if hidden_src.shape[1] >= tgt_dim:
                hidden_tgt = hidden_src[:, :tgt_dim]
            else:
                hidden_tgt = torch.nn.functional.pad(hidden_src, (0, tgt_dim - hidden_src.shape[1]))
        else:
            with torch.no_grad():
                hidden_tgt = state.mot_h.translate_hidden(hidden_src)  # [seq_len, 2048]
    translate_latency = (time.time() - t1) * 1000
    logger.info(
        "[%s] translate: %s → %s %.1fms",
        request_id, tuple(hidden_src.shape), tuple(hidden_tgt.shape), translate_latency,
    )

    # ── Step 3: resume (Mac B — Qwen3.6 decode, SSE stream) ────────
    t2 = time.time()
    resume_req = ResumeRequest(
        hidden_state_b64=encode_hidden_state(hidden_tgt),
        seq_len=hidden_tgt.shape[0],
        hidden_dim=hidden_tgt.shape[1],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
        request_id=request_id,
    )

    first_token = True
    async for content in state.resume_client.resume_stream(resume_req):
        chunk = {
            "choices": [{"delta": {"content": content}}],
        }
        if first_token:
            chunk["cgc_meta"] = {
                "request_id": request_id,
                "emit_latency_ms": emit_latency,
                "translate_latency_ms": translate_latency,
                "seq_len": hidden_src.shape[0],
                "src_hidden_dim": hidden_src.shape[1],
                "tgt_hidden_dim": hidden_tgt.shape[1],
            }
            first_token = False
        yield chunk

    resume_latency = (time.time() - t2) * 1000
    total = (time.time() - t0) * 1000
    yield {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "cgc_meta": {
            "request_id": request_id,
            "emit_latency_ms": round(emit_latency, 1),
            "translate_latency_ms": round(translate_latency, 1),
            "resume_latency_ms": round(resume_latency, 1),
            "total_latency_ms": round(total, 1),
        },
    }
    yield "[DONE]"


def _format_sse(data: dict | str) -> str:
    """格式化為 SSE 行."""
    if isinstance(data, str):
        return f"data: {data}\n\n"
    import json
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# API 端點
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    emit_ok = await state.emit_client.health() if state.emit_client else False
    resume_ok = await state.resume_client.health() if state.resume_client else False
    return {
        "status": "ok" if (emit_ok and resume_ok) else "degraded",
        "emit_client": {"url": state.emit_client.base_url if state.emit_client else None, "ok": emit_ok},
        "resume_client": {"url": state.resume_client.base_url if state.resume_client else None, "ok": resume_ok},
        "mot_h": "loaded" if state.mot_h else "passthrough",
        "source_model": state.source_model.value,
        "target_model": state.target_model.value,
    }


@app.post("/v1/cgc/generate")
async def generate(req: GenerateRequest):
    """完整 PD 管線: emit → translate → resume.

    stream=true: 返回 SSE stream (OpenAI chat completions 格式)
    stream=false: 返回完整 JSON
    """
    if req.stream:
        return StreamingResponse(
            _generate_stream(req),
            media_type="text/event-stream",
        )
    else:
        # 非流式: 收集所有 token 後返回
        text_parts = []
        meta = {}
        async for chunk in run_pipeline(
            req.prompt, req.max_tokens, req.temperature, req.top_p, req.mode
        ):
            if isinstance(chunk, str) and chunk == "[DONE]":
                break
            if "choices" in chunk:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    text_parts.append(delta["content"])
            if "cgc_meta" in chunk:
                meta = chunk["cgc_meta"]
        return GenerateResponse(
            text="".join(text_parts),
            emit_latency_ms=meta.get("emit_latency_ms", 0),
            translate_latency_ms=meta.get("translate_latency_ms", 0),
            resume_latency_ms=meta.get("resume_latency_ms", 0),
            total_latency_ms=meta.get("total_latency_ms", 0),
        )


async def _generate_stream(req: GenerateRequest) -> AsyncIterator[str]:
    """SSE stream 生成器."""
    async for chunk in run_pipeline(
        req.prompt, req.max_tokens, req.temperature, req.top_p, req.mode
    ):
        yield _format_sse(chunk)


# ---------------------------------------------------------------------------
# 採集端點 (給 collect_parallel_data.py 用)
# ---------------------------------------------------------------------------
@app.post("/v1/cgc/collect")
async def collect(prompt: str, target_prompt: str | None = None):
    """採集平行對: 同一文本的 Gemma4 hidden + Qwen3.6 hidden.

    用於 MoT-h 訓練數據採集.
    """
    request_id = str(uuid.uuid4())[:8]

    # source hidden (Gemma4)
    emit_req = EmitRequest(prompt=prompt, request_id=request_id)
    emit_resp = await state.emit_client.emit(emit_req)
    src_packet = emit_resp.packet

    # target hidden (Qwen3.6 — 需要一個 emit 端點在 Mac B)
    # TODO: Mac B 也需實現 /v1/cgc/emit (Qwen3.6 prefill)
    # 暫時用 chat completions + 內部 hook, 或 Mac B 同時跑 emit + resume
    return {
        "request_id": request_id,
        "prompt": prompt,
        "source_packet": src_packet.to_dict(),
        "target_packet": None,  # 待實現
        "note": "Mac B 需額外實現 /v1/cgc/emit (Qwen3.6 prefill) 以採集 target hidden",
    }


# ---------------------------------------------------------------------------
# Ingest 端點 (Mac push 模式 — Mac 主動推 hidden state 到 Windows)
# ---------------------------------------------------------------------------
# 全域暫存: request_id → HiddenStatePacket
# 生產環境應改用 Redis/queue, MVP 用內存 dict
_ingest_buffer: dict[str, "HiddenStatePacket"] = {}
_ingest_events: dict[str, asyncio.Event] = {}


@app.post("/v1/cgc/ingest")
async def ingest(packet: dict):
    """接收 Mac 主動推送的 hidden state (push 模式).

    Mac 端在 prefill 完成後, 用 URLSession POST hidden state 到此端點.
    Windows 收到後存入 buffer, 等待 run_pipeline 消費.

    請求體 = HiddenStatePacket.to_dict() 的 JSON

    返回: {"ok": true, "request_id": "..."}
    """
    pkt = HiddenStatePacket.from_dict(packet)
    _ingest_buffer[pkt.request_id] = pkt
    if pkt.request_id in _ingest_events:
        _ingest_events[pkt.request_id].set()
    logger.info(
        "ingest: request_id=%s seq_len=%d hidden_dim=%d",
        pkt.request_id, pkt.seq_len, pkt.hidden_dim,
    )
    return {"ok": True, "request_id": pkt.request_id}


@app.get("/v1/cgc/ingest/{request_id}")
async def get_ingested(request_id: str):
    """查詢已接收的 hidden state (給 Mac 端輪詢用)."""
    if request_id in _ingest_buffer:
        return {"ok": True, "packet": _ingest_buffer[request_id].to_dict()}
    return {"ok": False, "error": "not found"}


async def wait_for_ingest(request_id: str, timeout: float = 30.0) -> Optional[HiddenStatePacket]:
    """等待 Mac 推送的 hidden state (供 run_pipeline 用).

    Args:
        request_id: 請求 ID
        timeout: 超時秒數

    Returns:
        HiddenStatePacket 或 None (超時)
    """
    if request_id in _ingest_buffer:
        return _ingest_buffer.pop(request_id)

    event = asyncio.Event()
    _ingest_events[request_id] = event
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return _ingest_buffer.pop(request_id, None)
    except asyncio.TimeoutError:
        logger.warning("ingest timeout: request_id=%s", request_id)
        return None
    finally:
        _ingest_events.pop(request_id, None)


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "9000")))
