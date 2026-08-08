"""Cloud resume endpoint for CGC layer-split (Phase 2).

設計 (Design)
============
Mac prefill 前 P 層 → emit hidden_P via MacEmitHandoff (TCP PUT, role=emitter)
Cloud /v1/cgc/resume 收到 HTTP 請求 → MacEmitHandoff(role=receiver).recv(rank, step)
→ 取 hidden_P → transformers Qwen3-VL 從 layer P resume forward 後 L-P 層
→ norm + lm_head → sample → decode loop (KV cache) → SSE stream 回 token

為何需要 custom endpoint (Why custom)
  sglang 不原生支持從 hidden_states resume(它從 input_ids embed 開始跑全部層)。
  需 custom endpoint: 手動跑 layers[P:] + lm_head, 跳過 embed + 前 P 層。
  transformers 可直接存取 model.model.language_model.layers[P:], 適合此場景。

架構路徑 (HF model attribute path)
  Qwen3VLMoeForConditionalGeneration
    .model → Qwen3VLMoeModel
      .language_model → Qwen3MoeForCausalLM
        .model → Qwen3MoeModel
          .embed_tokens, .layers[48], .norm
        .lm_head
  Mac 送 hidden_P = 跑完 layers[0:P] 的 residual stream (未 norm)
  Cloud 跑: layers[P:48] on hidden_P → norm → lm_head → logits → sample

環境變量 (Env)
  CGC_CLOUD_MODEL_PATH: 雲端模型路徑 (Qwen3-VL HF 格式, 必填)
  CGC_CLOUD_NUM_LAYERS: 模型總層數 (默認 48, Qwen3-VL-30B)
  CGC_CLOUD_DEVICE: 設備 (默認 cuda)
  CGC_CLOUD_DTYPE: 精度 (默認 bfloat16)
  CGC_MAC_EMIT_PORT: MacEmitHandoff receiver 監聽 port (默認 31010)
  CGC_RESUME_PORT: 本端點 HTTP port (默認 30010)

用法 (Usage)
  python3 cloud_resume_endpoint.py
  # 或
  uvicorn cloud_resume_endpoint:app --host 0.0.0.0 --port 30010

依賴 (Deps, cloud env)
  fastapi, uvicorn, torch, transformers, safetensors
"""

from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import asyncio
import threading
from typing import Optional, Generator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

# repo root for CGC_Phase2 imports
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


app = FastAPI(title="CGC Cloud Resume Endpoint")

# === Cloud model singleton (lazy load) ===
_cloud_model = None
_cloud_tokenizer = None
_cloud_model_lock = threading.Lock()

_CLOUD_MODEL_PATH = os.environ.get("CGC_CLOUD_MODEL_PATH", "")
_CLOUD_TOTAL_LAYERS = int(os.environ.get("CGC_CLOUD_NUM_LAYERS", "48"))
_CLOUD_DEVICE = os.environ.get("CGC_CLOUD_DEVICE", "cuda")
_CLOUD_DTYPE = os.environ.get("CGC_CLOUD_DTYPE", "bfloat16")


def _load_cloud_model():
    """Lazy-load Qwen3-VL via transformers (singleton, thread-safe)。"""
    global _cloud_model, _cloud_tokenizer
    if _cloud_model is not None:
        return _cloud_model, _cloud_tokenizer
    with _cloud_model_lock:
        if _cloud_model is not None:
            return _cloud_model, _cloud_tokenizer
        import torch
        from transformers import AutoTokenizer

        path = _CLOUD_MODEL_PATH
        if not path:
            raise RuntimeError("CGC_CLOUD_MODEL_PATH 未設定")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(_CLOUD_DTYPE, torch.bfloat16)

        print(
            f"[cgc-resume] 加載雲端模型 {path} "
            f"(dtype={_CLOUD_DTYPE}, device={_CLOUD_DEVICE})",
            flush=True,
        )
        # Qwen3-VL-MoE: 優先 AutoModelForCausalLM, fallback AutoModelForImageTextToText
        model = None
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map=_CLOUD_DEVICE,
            )
        except Exception as e_causal:
            print(
                f"[cgc-resume] AutoModelForCausalLM 失敗({e_causal!r}), "
                f"試 AutoModelForImageTextToText",
                flush=True,
            )
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                path,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map=_CLOUD_DEVICE,
            )
        model.eval()
        _cloud_model = model
        _cloud_tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True
        )
        nl = _get_total_layers(model)
        print(
            f"[cgc-resume] 模型加載完成, layers={nl}, "
            f"type={type(model).__name__}",
            flush=True,
        )
        return _cloud_model, _cloud_tokenizer


def _get_total_layers(model) -> int:
    """從 HF 模型取得 transformer 總層數。"""
    m = getattr(model, "model", model)
    lang = getattr(m, "language_model", None)
    if lang is not None:
        inner = getattr(lang, "model", lang)
    else:
        inner = getattr(m, "model", m)
    layers = getattr(inner, "layers", None)
    if layers is not None:
        return len(layers)
    return _CLOUD_TOTAL_LAYERS


def _get_resume_components(model, P: int):
    """取得 cloud resume 所需組件: (layers_P_to_L, norm, lm_head, embed)。

    layers_P_to_L: list of HF layer modules (layers[P:])
    norm: 最終 RMSNorm (layers 後的 norm)
    lm_head: 詞彙投影層
    embed: embed_tokens (decode 時 embed next token)

    適配兩種 HF 包裝:
      Qwen3VLMoeForConditionalGeneration: model.model.language_model.model.layers
      Qwen3MoeForCausalLM: model.model.layers
    """
    m = getattr(model, "model", model)
    lang = getattr(m, "language_model", None)
    if lang is not None:
        inner = getattr(lang, "model", lang)
        lm_head = getattr(lang, "lm_head", None)
    else:
        inner = getattr(m, "model", m)
        lm_head = (
            getattr(model, "lm_head", None)
            or getattr(m, "lm_head", None)
        )

    layers = getattr(inner, "layers", None)
    norm = getattr(inner, "norm", None)
    embed = getattr(inner, "embed_tokens", None)
    if layers is None or norm is None or lm_head is None or embed is None:
        raise RuntimeError(
            f"無法定位 layers/norm/lm_head/embed: model={type(model).__name__}, "
            f"m={type(m).__name__}, lang={type(lang).__name__ if lang else None}, "
            f"inner={type(inner).__name__}"
        )
    return layers[P:], norm, lm_head, embed


def _sample(logits, temperature: float, top_p: float) -> int:
    """從 logits [1, 1, vocab] 採樣一個 token id。"""
    import torch
    import torch.nn.functional as F

    lg = logits[:, -1, :]  # [1, vocab]
    if temperature <= 0:
        return int(lg.argmax(dim=-1).item())
    lg = lg / temperature
    if top_p < 1.0:
        probs = F.softmax(lg, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # 機率累積超過 top_p 的 token 截掉
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum()
        next_idx = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_idx.gather(-1, next_idx)
    else:
        probs = F.softmax(lg, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


def _cloud_resume_forward(
    hidden_P,
    P: int,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Generator[int, None, None]:
    """從 hidden_P resume forward 後 L-P 層 + lm_head, 生成 token (generator)。

    流程:
      1. hidden_P [1, seq_len, hidden] → CUDA
      2. 建 position_ids [0..seq_len-1] + causal mask (prefill)
      3. DynamicCache: 逐層 forward layers[P:] on hidden_P (prefill) → 建 KV cache
      4. norm → lm_head → logits[:, -1] → sample first token, yield
      5. decode loop: next_token → embed → layers[P:] (with cache) → norm → lm_head
         → sample → yield, 直到 max_tokens 或 EOS

    參數:
      hidden_P: torch.Tensor [1, seq_len, hidden_dim] (Mac 送的 CPU tensor)
      P: 從第 P 層 resume (Mac 已跑 0..P-1)
      max_tokens: 最多生成 token 數
      temperature / top_p: 採樣參數
    """
    import torch
    from transformers import DynamicCache

    model, tokenizer = _load_cloud_model()
    device = _CLOUD_DEVICE
    dtype = next(model.parameters()).dtype

    # 1. hidden_P → CUDA (Mac 送的是 CPU fp32 tensor, 轉模型 dtype)
    if not isinstance(hidden_P, torch.Tensor):
        raise TypeError(f"hidden_P 須為 torch.Tensor, got {type(hidden_P)}")
    h = hidden_P.to(device=device, dtype=dtype)
    if h.dim() == 2:
        h = h.unsqueeze(0)
    seq_len = h.shape[1]

    # 2. position_ids + causal mask (HF 4D additive mask: 0=attend, -inf=mask)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_mask = torch.full(
        (seq_len, seq_len), float("-inf"), device=device, dtype=dtype
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]

    # 3. 取 layers[P:] + norm + lm_head + embed
    resume_layers, norm, lm_head, embed = _get_resume_components(model, P)
    total_layers = _get_total_layers(model)
    print(
        f"[cgc-resume] resume forward: P={P}, layers={len(resume_layers)}/"
        f"{total_layers}, seq_len={seq_len}, hidden={h.shape[-1]}, "
        f"dtype={dtype}",
        flush=True,
    )

    # 4. prefill: 逐層 forward layers[P:] on hidden_P, 建 KV cache
    cache = DynamicCache()
    cur_h = h
    for layer in resume_layers:
        out = layer(
            cur_h,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
        )
        # HF layer.forward 返回 tuple (hidden_states, ...) 或 BaseModelOutput
        if isinstance(out, tuple):
            cur_h = out[0]
        elif hasattr(out, "last_hidden_state"):
            cur_h = out.last_hidden_state
        else:
            cur_h = out[0]

    # 5. norm + lm_head → logits → sample first token
    cur_h = norm(cur_h)
    logits = lm_head(cur_h[:, -1:, :])  # [1, 1, vocab]
    next_token = _sample(logits, temperature, top_p)
    yield next_token

    # 6. decode loop (逐 token, 用 KV cache)
    eos_token_id = tokenizer.eos_token_id if tokenizer else None
    for step in range(max_tokens - 1):
        next_id = torch.tensor(
            [[next_token]], device=device, dtype=torch.long
        )
        next_h = embed(next_id)  # [1, 1, hidden]
        next_pos = seq_len + step
        pos_ids = torch.tensor([[next_pos]], device=device)

        # decode: 單 token, attention_mask=None (HF layer 內部處理)
        for layer in resume_layers:
            out = layer(
                next_h,
                attention_mask=None,
                position_ids=pos_ids,
                past_key_value=cache,
                use_cache=True,
            )
            if isinstance(out, tuple):
                next_h = out[0]
            elif hasattr(out, "last_hidden_state"):
                next_h = out.last_hidden_state
            else:
                next_h = out[0]

        next_h = norm(next_h)
        logits = lm_head(next_h[:, -1:, :])
        next_token = _sample(logits, temperature, top_p)
        yield next_token

        if eos_token_id is not None and next_token == eos_token_id:
            break


# === MacEmitHandoff receiver (cloud side) ===
_mac_emit_receiver = None
_mac_emit_init_lock = threading.Lock()


def _get_mac_emit_receiver():
    """取得 MacEmitHandoff(role=receiver) 單例。cloud 端啟動 TCP server 監聽 Mac PUT。"""
    global _mac_emit_receiver
    if _mac_emit_receiver is not None:
        return _mac_emit_receiver
    with _mac_emit_init_lock:
        if _mac_emit_receiver is not None:
            return _mac_emit_receiver
        from CGC_Phase2.cgc_handoff_transport import HandoffTransport
        _mac_emit_receiver = HandoffTransport.make("mac_emit", role="receiver")
        return _mac_emit_receiver


def _recv_hidden_P(rank: int, step: int, timeout: float = 30.0):
    """從 MacEmitHandoff receiver pop Mac 推送的 hidden_P payload (blocking)。

    返回 dict: {finished_layer, hidden_states, step, request_id, input_ids, seq_len, ...}
    """
    transport = _get_mac_emit_receiver()
    return transport.recv(rank, step, timeout=timeout)


# === FastAPI endpoints ===

@app.post("/v1/cgc/resume")
async def cgc_resume(request: Request):
    """Cloud resume endpoint: 從 Mac 的 hidden_P resume forward 後 L-P 層。

    請求體 (OpenAI chat completions 格式 + extra_body):
      {
        "model": "...",
        "messages": [...],
        "max_tokens": 512,
        "stream": true,
        "temperature": 0.7,
        "top_p": 0.9,
        "extra_body": {
          "cgc_resume_from_layer": 8,
          "cgc_resume_request_id": "abc123",
          "cgc_resume_rank": 0,
          "cgc_resume_step": 0,
          "cgc_resume_seq_len": 1024,
          "cgc_resume_finished_layer": 8
        }
      }

    流程:
      1. 解析 cgc_resume_* 參數 (從 extra_body 或 top-level)
      2. MacEmitHandoff(role=receiver).recv(rank, step) → hidden_P payload
      3. transformers Qwen3-VL 從 layer P resume forward + lm_head + decode
      4. SSE stream 返回 token (OpenAI chunk 格式)

    失敗回退: 返回 error JSON (edge 端可 fallback 全雲)。
    """
    body = await request.json()
    extra = body.get("extra_body") or {}

    # cgc_resume_* 可能在 extra_body 或 top-level (兼容不同 HTTP client)
    def _get(key: str, default=None, cast=str):
        val = extra.get(key, body.get(key, default))
        if val is None:
            return default
        try:
            return cast(val)
        except (ValueError, TypeError):
            return default

    P = _get("cgc_resume_from_layer", 0, int)
    rank = _get("cgc_resume_rank", 0, int)
    step = _get("cgc_resume_step", 0, int)
    seq_len_expected = _get("cgc_resume_seq_len", 0, int)
    request_id = _get("cgc_resume_request_id", "", str)

    max_tokens = int(body.get("max_tokens", 512))
    temperature = float(body.get("temperature", 0.7))
    top_p = float(body.get("top_p", 0.9))
    stream = body.get("stream", True)
    model_name = body.get("model", "cgc-resume")

    print(
        f"[cgc-resume] 收到 resume 請求: P={P} rank={rank} step={step} "
        f"req={request_id} seq_len_expected={seq_len_expected} "
        f"max_tokens={max_tokens}",
        flush=True,
    )

    if P <= 0:
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": f"cgc_resume_from_layer={P} 無效 (須 >0)",
                "type": "invalid_request",
            }},
        )

    # 1. 從 MacEmitHandoff receiver 取 hidden_P (blocking, 放 thread pool)
    loop = asyncio.get_event_loop()
    try:
        payload = await loop.run_in_executor(
            None, _recv_hidden_P, rank, step, 30.0
        )
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": {
                "message": f"recv hidden_P 失敗 (rank={rank} step={step}): {e}",
                "type": "recv_error",
            }},
        )

    hidden_P = payload.get("hidden_states")
    if hidden_P is None:
        return JSONResponse(
            status_code=502,
            content={"error": {
                "message": "payload 缺 hidden_states",
                "type": "payload_error",
            }},
        )

    finished_P = payload.get("finished_layer", P)
    print(
        f"[cgc-resume] 收到 hidden_P shape={list(hidden_P.shape)} "
        f"dtype={hidden_P.dtype} finished_layer={finished_P}",
        flush=True,
    )

    # 2. 生成 token stream
    def _gen_sync():
        cid = f"chatcmpl-{hashlib.sha1(f'{time.time()}.{request_id}'.encode()).hexdigest()[:12]}"
        created = int(time.time())
        try:
            _, tokenizer = _load_cloud_model()
            for token_id in _cloud_resume_forward(
                hidden_P, P,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ):
                text = tokenizer.decode([token_id], skip_special_tokens=True)
                if not text:
                    continue
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            done = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(done)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            err = {
                "error": {
                    "message": f"resume forward 失敗: {e}",
                    "type": "resume_error",
                }
            }
            yield f"data: {json.dumps(err)}\n\n".encode()

    if stream:
        return StreamingResponse(
            _gen_sync(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        # non-stream: 收集所有 token 一次返回
        try:
            _, tokenizer = _load_cloud_model()
            token_ids = list(_cloud_resume_forward(
                hidden_P, P,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ))
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            return {
                "id": f"chatcmpl-{hashlib.sha1(f'{time.time()}'.encode()).hexdigest()[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": seq_len_expected, "completion_tokens": len(token_ids)},
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"error": {
                    "message": f"resume forward 失敗: {e}",
                    "type": "resume_error",
                }},
            )


@app.get("/health")
def health():
    """健康檢查 + 配置。"""
    return {
        "status": "ok",
        "endpoint": "/v1/cgc/resume",
        "cloud_model_path": _CLOUD_MODEL_PATH or "(unset)",
        "cloud_model_loaded": _cloud_model is not None,
        "cloud_num_layers": _CLOUD_TOTAL_LAYERS,
        "mac_emit_receiver_started": _mac_emit_receiver is not None,
        "device": _CLOUD_DEVICE,
        "dtype": _CLOUD_DTYPE,
    }


@app.on_event("startup")
async def _startup():
    """啟動時預啟 MacEmitHandoff receiver (cloud TCP server)。"""
    try:
        _get_mac_emit_receiver()
        print("[cgc-resume] MacEmitHandoff receiver 已啟動", flush=True)
    except Exception as e:
        print(f"[cgc-resume] MacEmitHandoff receiver 啟動失敗: {e}", flush=True)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("CGC_RESUME_PORT", "30010"))
    host = os.environ.get("CGC_RESUME_HOST", "0.0.0.0")
    print(
        f"[cgc-resume] 啟動 cloud resume endpoint on {host}:{port} "
        f"(model_path={_CLOUD_MODEL_PATH or '(unset)'})",
        flush=True,
    )
    uvicorn.run(app, host=host, port=port)
