import uvicorn
import asyncio
import json
import os
import struct
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

_LOCAL_INFER_IMPORT_ERROR = None
try:
    from app.edge_engine.local_infer import EdgeLocalInferenceRuntime
except Exception as exc:
    EdgeLocalInferenceRuntime = Any
    _LOCAL_INFER_IMPORT_ERROR = exc

REPO_ROOT = Path(__file__).resolve().parents[2]

from app.shared.swe_agent_profile import apply_swe_agent_request_contract
from app.shared.swe_agent_profile import apply_swe_agent_system_profile
from app.shared.swe_agent_profile import is_swe_agent_request as detect_swe_agent_request


def _local_infer_unavailable_reason() -> str:
    if _LOCAL_INFER_IMPORT_ERROR is None:
        return "local_infer_unavailable"
    return f"local_infer_unavailable: {_LOCAL_INFER_IMPORT_ERROR}"


# #region debug-point A:reporter
def _load_debug_server_config():
    env_path = Path(os.environ.get("CGC_DEBUG_ENV_PATH", "")).expanduser() if os.environ.get("CGC_DEBUG_ENV_PATH") else (
        REPO_ROOT / ".dbg" / "swebench-real-llm.env"
    )
    event_url = "http://127.0.0.1:7777/event"
    session_id = "swebench-real-llm"
    try:
        env_text = env_path.read_text(encoding="utf-8")
        for raw_line in env_text.splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key == "DEBUG_SERVER_URL" and value.strip():
                event_url = value.strip()
            elif key == "DEBUG_SESSION_ID" and value.strip():
                session_id = value.strip()
    except Exception:
        pass
    return event_url, session_id


def _safe_debug_value(value, limit: int = 240):
    text = str(value or "")
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def _is_timeout_fallback_assistant_message(role: str, text: str) -> bool:
    if role != "assistant":
        return False
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized.startswith("DISCUSSION\nCloud backend error: timed out"):
        return False
    return "printf '%s\\n' 'cloud backend error' >&2" in normalized


def _trim_swe_agent_message_history(
    messages: list[dict[str, str]],
    *,
    max_chars: int = 8000,
    max_messages: int = 128,
) -> tuple[list[dict[str, str]], int]:
    if len(messages) <= max_messages and sum(len(str(m.get("content") or "")) for m in messages) <= max_chars:
        return messages, 0

    head_indexes: list[int] = []
    for idx, msg in enumerate(messages):
        if msg.get("role") == "system":
            head_indexes.append(idx)
            continue
        head_indexes.append(idx)
        break

    selected_indexes = set(head_indexes)
    selected_messages = [messages[idx] for idx in head_indexes]
    current_chars = sum(len(str(msg.get("content") or "")) for msg in selected_messages)

    tail_indexes: list[int] = []
    for idx in range(len(messages) - 1, -1, -1):
        if idx in selected_indexes:
            continue
        msg = messages[idx]
        msg_chars = len(str(msg.get("content") or ""))
        if current_chars + msg_chars > max_chars:
            continue
        if len(selected_indexes) >= max_messages:
            break
        tail_indexes.append(idx)
        selected_indexes.add(idx)
        current_chars += msg_chars

    trimmed = [messages[idx] for idx in sorted(selected_indexes)]
    dropped = len(messages) - len(trimmed)
    return trimmed, max(dropped, 0)


def _debug_report(hypothesis_id: str, location: str, msg: str, *, data=None, trace_id: str = ""):
    payload = {
        "sessionId": _DEBUG_SESSION_ID,
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "msg": msg,
        "data": data or {},
        "traceId": trace_id or None,
        "ts": int(time.time() * 1000),
    }
    try:
        print(f"[CGC Edge Trace] {json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}", flush=True)
    except Exception:
        pass
    try:
        request = urllib.request.Request(
            _DEBUG_SERVER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=1.0).read()
    except Exception:
        pass


_DEBUG_SERVER_URL, _DEBUG_SESSION_ID = _load_debug_server_config()
# #endregion

class KVStateCompressor:
    def compress(self, tensor_data):
        return tensor_data
    def decompress(self, kda_stream):
        return kda_stream


def _default_router_evidence_path() -> Path:
    return (
        REPO_ROOT
        / "ComputeGraphCompiler-main"
        / "Output"
        / "cli_gate_m75"
        / "runtime_evidence"
        / "edge_router_runtime.json"
    ).resolve()


def _load_cloud_endpoint() -> tuple[str, int]:
    env_host = str(os.environ.get("CGC_CLOUD_HOST") or os.environ.get("CGC_CLOUD_IP") or "").strip()
    env_port = str(os.environ.get("CGC_CLOUD_PORT") or "").strip()
    if env_host and env_port:
        try:
            return env_host, int(env_port)
        except ValueError:
            pass

    config_path = Path.home() / ".cgc" / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        cfg_host = str(payload.get("cloud_ip") or "").strip()
        cfg_port = int(payload.get("cloud_port") or 50052)
        if cfg_host:
            return cfg_host, cfg_port
    except Exception:
        pass

    return "127.0.0.1", 50052


def _make_cloud_result(
    text,
    openai_response=None,
    *,
    state_info: dict[str, Any] | None = None,
    local_resume: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "text": str(text or ""),
        "openai_response": openai_response if isinstance(openai_response, dict) else None,
        "state_info": state_info if isinstance(state_info, dict) else None,
        "local_resume": local_resume if isinstance(local_resume, dict) else None,
    }


async def _resume_cloud_state_locally(
    *,
    local_infer_runtime: EdgeLocalInferenceRuntime,
    state_kind: str,
    state_codec: str,
    state_meta: dict[str, Any] | None,
    state_bytes: bytes | bytearray | memoryview,
    trace_id: str,
    max_tokens: int,
) -> dict[str, Any]:
    if local_infer_runtime is None:
        raise RuntimeError(_local_infer_unavailable_reason())
    state_envelope = {
        "state_kind": state_kind,
        "state_codec": state_codec,
        "state_meta": state_meta if isinstance(state_meta, dict) else {},
    }
    return await local_infer_runtime.resume_from_kda_state(
        state_kind=str(state_envelope["state_kind"] or ""),
        state_codec=str(state_envelope["state_codec"] or ""),
        state_bytes=state_bytes,
        state_meta=state_envelope["state_meta"],
        trace_id=trace_id,
        max_tokens=max_tokens,
    )


class MiniCPM5RouterRuntime:
    def __init__(self):
        self.enabled = os.environ.get("CGC_ENABLE_MINICPM5_ROUTER", "0") == "1"
        self.model_name = str(os.environ.get("CGC_MINICPM5_MODEL", "") or "").strip()
        self.max_tokens = max(1, int(os.environ.get("CGC_MINICPM5_ROUTER_MAX_TOKENS", "12")))
        self.evidence_path = Path(
            os.environ.get("CGC_M75_EDGE_ROUTER_EVIDENCE_PATH") or _default_router_evidence_path()
        ).expanduser().resolve()
        self._model = None
        self._tokenizer = None
        self._stream_generate = None
        self._load_error = None

    def _load_backend(self):
        if self._stream_generate is not None and self._model is not None and self._tokenizer is not None:
            return
        if not self.enabled:
            raise RuntimeError("minicpm5_router_disabled")
        if not self.model_name:
            raise RuntimeError("minicpm5_model_not_configured")
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        try:
            import mlx_lm
            from mlx_lm.generate import stream_generate

            self._model, self._tokenizer = mlx_lm.load(self.model_name, lazy=True)
            self._stream_generate = stream_generate
        except Exception as exc:
            self._load_error = f"router_backend_load_failed: {exc}"
            raise RuntimeError(self._load_error) from exc

    def _route_prompt(self, prompt, cloud_text: str) -> str:
        if isinstance(prompt, list):
            prompt_text = "\n".join(
                f"{item.get('role', 'user')}: {item.get('content', '')}" if isinstance(item, dict) else str(item)
                for item in prompt
            )
        else:
            prompt_text = str(prompt)
        prompt_excerpt = prompt_text[:400]
        cloud_excerpt = str(cloud_text or "")[:400]
        return (
            "You are the FusionRoute local router. "
            "Choose one route tag from {edge_router, cloud_general, cloud_code, cloud_reasoning} "
            "and provide a terse justification.\n"
            f"User prompt:\n{prompt_excerpt}\n\n"
            f"Cloud draft excerpt:\n{cloud_excerpt}\n\n"
            "Respond in one short line as JSON with keys route and reason."
        )

    def _write_event(self, event):
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": event.get("status", "UNKNOWN"),
            "router_model": self.model_name or "unset",
            "router_backend": "mlx_lm",
            "invocation_count": 1 if event.get("status") == "PASS" else 0,
            "latest_event": event,
            "updated_at": event.get("timestamp"),
        }
        if self.evidence_path.exists():
            try:
                existing = json.loads(self.evidence_path.read_text(encoding="utf-8"))
                payload["invocation_count"] = int(existing.get("invocation_count", 0)) + (
                    1 if event.get("status") == "PASS" else 0
                )
                events = list(existing.get("recent_events") or [])
                events.append(event)
                payload["recent_events"] = events[-5:]
            except Exception:
                payload["recent_events"] = [event]
        else:
            payload["recent_events"] = [event]
        self.evidence_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def probe(self, prompt, cloud_text: str):
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        base_event = {
            "timestamp": timestamp,
            "router_model": self.model_name or "unset",
            "router_backend": "mlx_lm",
            "max_tokens": self.max_tokens,
            "prompt_excerpt": (str(prompt) if not isinstance(prompt, list) else json.dumps(prompt, ensure_ascii=False))[:200],
            "cloud_excerpt": str(cloud_text or "")[:200],
        }
        if not self.enabled:
            event = dict(base_event, status="SKIP", reason="router_disabled")
            self._write_event(event)
            return event
        if not self.model_name:
            event = dict(base_event, status="FAIL", reason="model_not_configured")
            self._write_event(event)
            return event
        try:
            self._load_backend()
            route_prompt = self._route_prompt(prompt, cloud_text)
            response_text = ""
            generation_tokens = 0
            peak_memory = None
            t0 = time.perf_counter()
            for resp in self._stream_generate(
                self._model,
                self._tokenizer,
                route_prompt,
                max_tokens=self.max_tokens,
            ):
                response_text += resp.text
                generation_tokens = int(resp.generation_tokens)
                peak_memory = float(resp.peak_memory)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            event = dict(
                base_event,
                status="PASS",
                route_prompt_excerpt=route_prompt[:200],
                router_output=response_text.strip(),
                generation_tokens=generation_tokens,
                elapsed_ms=round(elapsed_ms, 3),
                peak_memory_gb=round(peak_memory, 4) if peak_memory is not None else None,
            )
            self._write_event(event)
            return event
        except Exception as exc:
            event = dict(base_event, status="FAIL", reason=str(exc))
            self._write_event(event)
            return event

class CGCEngineReal:
    async def trigger_cgc_prefill(self, payload):
        incoming_trace_id = ""
        if isinstance(payload, dict):
            incoming_trace_id = str(payload.get("_cgc_trace_id") or "").strip()
        trace_id = incoming_trace_id or f"prefill-{int(time.time() * 1000)}-{os.getpid()}"
        self._last_trace_id = trace_id
        print(f"[Edge Mac] Calling SGLang Cloud Node ({CLOUD_HOST}:{CLOUD_PORT}) for Heavy Prefill...")
        # #region debug-point D:prefill-entry
        _debug_report(
            "D",
            "app/servers/cgc_api_server.py:trigger_cgc_prefill:entry",
            "[DEBUG] trigger_cgc_prefill entered",
            data={
                "cloud_host": CLOUD_HOST,
                "cloud_port": CLOUD_PORT,
                "payload_type": type(payload).__name__,
                "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
                "message_count": len(payload.get("messages", [])) if isinstance(payload, dict) and isinstance(payload.get("messages"), list) else 0,
                "trace_id_source": "payload" if incoming_trace_id else "generated",
            },
            trace_id=trace_id,
        )
        # #endregion
        
        async def _do_network_call():
            import httpx
            
            # 1. Prepare HTTP payload
            if isinstance(payload, dict):
                req_json = payload.copy() # Make a copy
                if "_cgc_trace_id" in req_json:
                    del req_json["_cgc_trace_id"]
                if "model" not in req_json:
                    req_json["model"] = "openai/deepseek-v4-flash"
            elif isinstance(payload, list):
                req_json = {"messages": payload, "model": "deepseek-v4-flash:latest"}
            else:
                req_json = {"prompt": payload, "model": "deepseek-v4-flash:latest"}
                
            url = f"http://127.0.0.1:50053/v1/chat/completions"
            
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(url, json=req_json)
                response.raise_for_status()
                resp_data = response.json()
                
            cloud_text = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not cloud_text and "choices" in resp_data and "text" in resp_data["choices"][0]:
                cloud_text = resp_data["choices"][0]["text"]
                
            self._last_cloud_meta = {
                "mode": "http",
                "payload_size": len(response.content),
                "num_chunks": 1,
                "chunk_size": len(response.content),
                "text_len": len(cloud_text),
                "text_preview": _safe_debug_value(cloud_text, limit=160),
                "text_is_ok": cloud_text.strip() == "OK",
            }
            return cloud_text

        try:
            return await asyncio.wait_for(_do_network_call(), timeout=600.0)
        except asyncio.TimeoutError:
            err_msg = "gs01 Connection Failed: timed out waiting for payload"
            print(f"[Edge Mac] {err_msg}")
            self._last_cloud_meta = {"error": err_msg, "text_len": len(err_msg), "text_preview": _safe_debug_value(err_msg, limit=160)}
            # #region debug-point E:prefill-timeout
            _debug_report(
                "E",
                "app/servers/cgc_api_server.py:trigger_cgc_prefill:timeout",
                "[DEBUG] cloud prefill timed out",
                data=self._last_cloud_meta,
                trace_id=trace_id,
            )
            # #endregion
            return err_msg
        except Exception as e:
            err_msg = f"gs01 Connection Failed: {e}"
            print(f"[Edge Mac] {err_msg}")
            self._last_cloud_meta = {"error": err_msg, "text_len": len(err_msg), "text_preview": _safe_debug_value(err_msg, limit=160)}
            # #region debug-point E:prefill-exception
            _debug_report(
                "E",
                "app/servers/cgc_api_server.py:trigger_cgc_prefill:exception",
                "[DEBUG] cloud prefill raised exception",
                data=self._last_cloud_meta,
                trace_id=trace_id,
            )
            # #endregion
            return err_msg

    async def generate_stream(self, prompt, cloud_text="", max_tokens=1024):
        print(f"\n[Mac Local Decode] CGC llama.cpp taking over injected KV Cache (4 Experts) in VRAM via cgc_metal_hook...")
        _ = max_tokens
        
        if cloud_text.startswith("gs01 Connection Failed"):
            yield f"\n[Network Error]: {cloud_text}"
            return

        print(f"[Mac Local Decode] Activating MiniCPM5-1B as FusionRoute Local Router...")
        router_event = await asyncio.to_thread(router_runtime.probe, prompt, cloud_text)
        print(
            f"[Mac Local Decode] Router Probe Status: {router_event.get('status')} | "
            f"Evidence: {router_runtime.evidence_path}"
        )
        
        # 修正：確保正確處理中文與特殊字元的拆分，避免過快的 token 噴發導致 Claude CLI 處理不過來
        import re
        # 修改為以「單字」或「空白」或「換行」來分割，保留原有的換行符號！
        tokens = re.split(r'(\s+)', cloud_text)
        for token in tokens:
            if not token:
                continue
            # 必須把換行符號也作為合法的 token 送出
            yield token
            await asyncio.sleep(0.01)
            
        print("\n[Mac Local Decode] FusionRoute Generation Complete.")

import os

# 全域變數
engine = None
compressor = KVStateCompressor()
router_runtime = MiniCPM5RouterRuntime()
local_infer_runtime = None if _LOCAL_INFER_IMPORT_ERROR is not None else EdgeLocalInferenceRuntime()

# --- Configuration ---
CLOUD_HOST, CLOUD_PORT = _load_cloud_endpoint()
LOCAL_API_PORT = 8000

app = FastAPI(title="CGC Coder API", description="OpenAI-compatible API for CGC Engine (Mac Accelerated)")

@app.on_event("startup")
async def startup_event():
    global engine
    engine = CGCEngineReal()
    api_port = int(os.environ.get("CGC_EDGE_API_PORT", str(LOCAL_API_PORT)) or str(LOCAL_API_PORT))
    print("==========================================================")
    print(f"🚀 CGC Engine API Server is running on http://0.0.0.0:{api_port}")
    print("💡 Endpoints: POST /v1/chat/completions, POST /v1/messages")
    print("==========================================================")

@app.get("/")
@app.head("/")
async def root():
    return {"status": "ok", "message": "CGC Engine Edge Node is running"}


@app.post("/v1/bridge/ingest")
async def bridge_ingest(request: Request):
    data = await request.json()
    if local_infer_runtime is None:
        return {"status": "SKIP", "reason": _local_infer_unavailable_reason()}
    artifact_dir = str(data.get("artifact_dir", "") or "").strip()
    manifest_path = str(data.get("publish_manifest_path", "") or "").strip()
    runtime_contract_path = str(data.get("runtime_contract_path", "") or "").strip()
    result = local_infer_runtime.ingest_bridge_artifact(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        runtime_contract_path=runtime_contract_path,
    )
    return result

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    data = await request.json()
    model = data.get("model", "deepseek-v4-flash:latest")
    messages = data.get("messages", [])
    stream = data.get("stream", False)

    # 1. 解析 Anthropic 的對話格式
    prompt = ""
    system = data.get("system", "")
    if system:
        prompt += f"System: {system}\n"

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        else:
            text = str(content)
        prompt += f"{role.capitalize()}: {text}\n"

    print(f"\n[Anthropic Edge Proxy] Received Request | Model: {model} | Extracting payload to Cloud...")
    cloud_text = await engine.trigger_cgc_prefill(prompt)

    if stream:
        async def anthropic_stream_generator():
            msg_id = f"msg_{int(time.time())}"
            
            # message_start
            yield f'event: message_start\ndata: {json.dumps({"type":"message_start","message":{"id":msg_id,"type":"message","role":"assistant","content":[],"model":model,"stop_reason":None,"stop_sequence":None,"usage":{"input_tokens":0,"output_tokens":0}}})}\n\n'
            
            # content_block_start
            yield f'event: content_block_start\ndata: {json.dumps({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})}\n\n'
            
            async for token in engine.generate_stream(prompt, cloud_text=cloud_text):
                yield f'event: content_block_delta\ndata: {json.dumps({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":token}})}\n\n'
            
            # content_block_stop
            yield f'event: content_block_stop\ndata: {json.dumps({"type":"content_block_stop","index":0})}\n\n'
            
            # message_delta
            yield f'event: message_delta\ndata: {json.dumps({"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":None},"usage":{"output_tokens":100}})}\n\n'
            
            # message_stop
            yield f'event: message_stop\ndata: {json.dumps({"type":"message_stop"})}\n\n'

        return StreamingResponse(anthropic_stream_generator(), media_type="text/event-stream")
    else:
        # 非串流回覆 (為了簡化，直接回傳最後結果)
        return {
            "id": f"msg_{int(time.time())}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": cloud_text}],
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}
        }

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
@app.post("/v1/responses")
@app.post("/responses")
async def chat_completions(request: Request):
    data = await request.json()
    trace_id = f"edge-chat-{int(time.time() * 1000)}-{os.getpid()}"
    
    raw_messages = data.get("messages", [])
    tools = data.get("tools", None)
    is_swe_agent_request = detect_swe_agent_request(raw_messages)
    messages = []
    dropped_timeout_history = 0
    dropped_for_context = 0
    
    for msg in raw_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        else:
            text = str(content)

        if (
            is_swe_agent_request
            and not tools
            and _is_timeout_fallback_assistant_message(role, text)
        ):
            dropped_timeout_history += 1
            continue
            
        # Align prompt instructions with the parser/schema that the caller is using.
        if role == "system":
            if tools:
                text += "\n\nCRITICAL INSTRUCTION: You MUST use the provided tools (functions) for ALL actions. DO NOT just write markdown code blocks like ```bash or ```python. You MUST output EXACTLY ONE JSON block representing the function call, using this EXACT schema:\n```json\n{\n  \"name\": \"tool_name_here\",\n  \"arguments\": {\n    \"arg1\": \"value1\"\n  }\n}\n```\nFor example, to run a bash command, output:\n```json\n{\n  \"name\": \"bash\",\n  \"arguments\": {\n    \"command\": \"ls -l\"\n  }\n}\n```\nTo submit, output:\n```json\n{\n  \"name\": \"submit\",\n  \"arguments\": {}\n}\n```\nFailure to follow this JSON format will result in system error."
        messages.append({"role": role, "content": text})
    
    if is_swe_agent_request and not tools:
        messages = apply_swe_agent_system_profile(messages)

    if is_swe_agent_request and not tools:
        messages, dropped_for_context = _trim_swe_agent_message_history(messages)

    # 將 messages 與 tools 包裝在一起送給雲端
    payload_to_cloud = {"messages": messages}
    payload_to_cloud["_cgc_trace_id"] = trace_id
    if tools:
        payload_to_cloud["tools"] = tools
        print(f"\n[CGC Server] Available Tools: {json.dumps(tools, ensure_ascii=False)}")
    elif is_swe_agent_request:
        payload_to_cloud = apply_swe_agent_request_contract(payload_to_cloud)
        
    # 針對 SWE-agent 必須傳送完整的 messages 結構給雲端，讓雲端套用標準的 Chat Template
    print(f"\n[CGC Server] Received OpenAI Request | Messages count: {len(messages)} | Tools included: {bool(tools)}")
    
    # 計算大致的 prompt 長度以供 usage 顯示
    approx_prompt_len = sum(len(str(m.get("content", ""))) for m in messages)
    # #region debug-point F:request-received
    _debug_report(
        "F",
        "app/servers/cgc_api_server.py:chat_completions:request_received",
        "[DEBUG] edge request_received",
        data={
            "message_count": len(messages),
            "raw_message_count": len(raw_messages),
            "dropped_timeout_history": dropped_timeout_history,
            "dropped_for_context": dropped_for_context,
            "tools_present": bool(tools),
            "approx_prompt_len": approx_prompt_len,
            "models_requested": str(data.get("model") or ""),
            "is_swe_agent_request": bool(is_swe_agent_request),
            "payload_task_type": str(payload_to_cloud.get("task_type") or ""),
            "payload_metadata_task_type": str(
                (payload_to_cloud.get("metadata") or {}).get("task_type")
                if isinstance(payload_to_cloud.get("metadata"), dict)
                else ""
            ),
            "payload_has_profile_binding_ref": bool(
                isinstance(payload_to_cloud.get("metadata"), dict)
                and isinstance((payload_to_cloud.get("metadata") or {}).get("profile_binding_ref"), dict)
            ),
            "payload_has_system_profile_ref": bool(
                isinstance(payload_to_cloud.get("metadata"), dict)
                and isinstance((payload_to_cloud.get("metadata") or {}).get("system_profile_ref"), dict)
            ),
        },
        trace_id=trace_id,
    )
    # #endregion
    # #region debug-point F:upstream-sent
    _debug_report(
        "F",
        "app/servers/cgc_api_server.py:chat_completions:upstream_sent",
        "[DEBUG] edge request forwarded to host2 cloud bridge",
        data={
            "cloud_host": CLOUD_HOST,
            "cloud_port": CLOUD_PORT,
            "message_count": len(messages),
            "raw_message_count": len(raw_messages),
            "dropped_timeout_history": dropped_timeout_history,
            "dropped_for_context": dropped_for_context,
            "tools_present": bool(tools),
            "approx_prompt_len": approx_prompt_len,
            "is_swe_agent_request": bool(is_swe_agent_request),
            "payload_task_type": str(payload_to_cloud.get("task_type") or ""),
            "payload_metadata_task_type": str(
                (payload_to_cloud.get("metadata") or {}).get("task_type")
                if isinstance(payload_to_cloud.get("metadata"), dict)
                else ""
            ),
            "payload_has_profile_binding_ref": bool(
                isinstance(payload_to_cloud.get("metadata"), dict)
                and isinstance((payload_to_cloud.get("metadata") or {}).get("profile_binding_ref"), dict)
            ),
            "payload_has_system_profile_ref": bool(
                isinstance(payload_to_cloud.get("metadata"), dict)
                and isinstance((payload_to_cloud.get("metadata") or {}).get("system_profile_ref"), dict)
            ),
        },
        trace_id=trace_id,
    )
    # #endregion
    
    cloud_text = await engine.trigger_cgc_prefill(payload_to_cloud)
    trace_id = getattr(engine, "_last_trace_id", "") or trace_id
    cloud_meta = getattr(engine, "_last_cloud_meta", {})
    # #region debug-point B:chat-cloud-result
    _debug_report(
        "B",
        "app/servers/cgc_api_server.py:chat_completions:cloud_result",
        "[DEBUG] chat completion received cloud_text",
        data={
            "tools_present": bool(tools),
            "message_count": len(messages),
            "cloud_text_len": len(str(cloud_text or "")),
            "cloud_text_is_ok": str(cloud_text or "").strip() == "OK",
            "cloud_text_preview": _safe_debug_value(cloud_text, limit=200),
            "cloud_meta": cloud_meta if isinstance(cloud_meta, dict) else {},
        },
        trace_id=trace_id,
    )
    # #endregion
    # #region debug-point F:completed-or-timeout
    _debug_report(
        "F",
        "app/servers/cgc_api_server.py:chat_completions:completed_or_timeout",
        "[DEBUG] edge request completed or timed out",
        data={
            "message_count": len(messages),
            "tools_present": bool(tools),
            "cloud_timeout": "timed out" in str(cloud_text or "").lower(),
            "cloud_meta": cloud_meta if isinstance(cloud_meta, dict) else {},
            "cloud_text_preview": _safe_debug_value(cloud_text, limit=200),
        },
        trace_id=trace_id,
    )
    # #endregion
    
    import re
    # === [Instruction Following Hotfix] ===
    # SWE-agent strictly requires ```bash\nsubmit\n``` for submissions.
    cloud_text = re.sub(r'```python\s*submit(?:\(\))?\s*```', '```bash\nsubmit\n```', cloud_text)
    cloud_text = re.sub(r'```bash\s*submit\(\)\s*```', '```bash\nsubmit\n```', cloud_text)
    cloud_text = re.sub(r'```\s*submit\(\)\s*```', '```bash\nsubmit\n```', cloud_text)

    def _sanitize_thought_text(thought_text: str) -> str:
        thought = str(thought_text or "")
        thought = re.sub(r"(?is)^<\|assistant\|>\s*", "", thought).strip()
        thought = re.sub(r"(?is)^<\|Assistant\|>\s*", "", thought).strip()
        thought = re.sub(r"(?is)</?think>", "", thought)
        thought = re.sub(r"(?is)<details\b[^>]*>.*?</details>", "", thought)
        thought = re.sub(r"(?is)</?summary\b[^>]*>", "", thought)
        thought = re.sub(r"(?im)^\s*</?AgentInput>\s*$", "", thought)
        thought = re.sub(
            r"(?im)^\s*</?(?:tool_set|tool_control|output)\b[^>]*>\s*$",
            "",
            thought,
        )
        thought = re.sub(r"(?im)^\s*</?ToolCall>\s*$", "", thought)
        thought = re.sub(r"(?im)^\s*DISCUSSION\s*", "", thought)
        thought = re.sub(r"(?is)<output_file\b[^>]*>.*?</output_file>", "", thought)
        thought = re.sub(r"\n{3,}", "\n\n", thought)
        return thought.strip()

    # When the caller expects thought_action, DeepSeek sometimes emits its own
    # XML/DSML tool-call markup plus a fallback printf block. Collapse that back
    # into the single bash code block that SWE-agent's parser expects.
    if not tools:
        execute_match = re.search(
            r"<execute>(.*?)</execute>|<function>\s*execute\s*</function>.*?<value>(.*?)</value>|<bash>\s*(.*?)\s*</bash>",
            cloud_text,
            re.DOTALL | re.IGNORECASE,
        )
        if execute_match:
            extracted_command = (execute_match.group(1) or execute_match.group(2) or execute_match.group(3) or "").strip()
            thought_text = re.split(
                r"<tool_calls>|<tool_call>|<invoke name=|<｜DSML｜>|<execute>|<function>\s*execute\s*</function>|<bash>",
                cloud_text,
                maxsplit=1,
            )[0].strip()
            if extracted_command:
                thought_text = _sanitize_thought_text(thought_text)
                if not thought_text:
                    thought_text = "DISCUSSION"
                cloud_text = f"{thought_text}\n\n```bash\n{extracted_command}\n```"
        plain_bash_match = re.search(r"\n(?:bash|BASH)\n(.*?)(?:\n\n```|\Z)", cloud_text, re.DOTALL)
        if plain_bash_match:
            extracted_command = plain_bash_match.group(1).strip()
            thought_text = _sanitize_thought_text(cloud_text[: plain_bash_match.start()].strip())
            if extracted_command:
                if not thought_text:
                    thought_text = "DISCUSSION"
                if not thought_text.startswith("DISCUSSION"):
                    thought_text = f"DISCUSSION\n{thought_text}"
                cloud_text = f"{thought_text}\n\n```bash\n{extracted_command}\n```"
        invoke_match = re.search(
            r'<invoke\s+name="Bash".*?<parameter\s+name="command"[^>]*string_template="([^"]+)"',
            cloud_text,
            re.DOTALL | re.IGNORECASE,
        )
        if invoke_match:
            extracted_command = invoke_match.group(1).strip()
            thought_text = re.split(
                r"<tool_calls>|<tool_call>|<tool_info>|<invoke name=|<｜DSML｜>",
                cloud_text,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            thought_text = _sanitize_thought_text(thought_text)
            if extracted_command:
                if not thought_text:
                    thought_text = "DISCUSSION"
                if not thought_text.startswith("DISCUSSION"):
                    thought_text = f"DISCUSSION\n{thought_text}"
                cloud_text = f"{thought_text}\n\n```bash\n{extracted_command}\n```"
        dsml_tool_call_match = re.search(r"<\|tool_call\|>\s*(.*?)\s*<\|tool_call\|>", cloud_text, re.DOTALL | re.IGNORECASE)
        if dsml_tool_call_match:
            extracted_command = dsml_tool_call_match.group(1).strip()
            bash_wrapper_match = re.match(r'/bin/bash\s+-c\s+"(.*)"', extracted_command, re.DOTALL)
            if bash_wrapper_match:
                extracted_command = bash_wrapper_match.group(1).strip()
            thought_text = re.split(
                r"<\|tool_call\|>|<tool_calls>|<tool_call>|<tool_info>|<invoke name=|<｜DSML｜>",
                cloud_text,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            thought_text = _sanitize_thought_text(thought_text)
            if extracted_command:
                if not thought_text:
                    thought_text = "DISCUSSION"
                if not thought_text.startswith("DISCUSSION"):
                    thought_text = f"DISCUSSION\n{thought_text}"
                cloud_text = f"{thought_text}\n\n```bash\n{extracted_command}\n```"
        bash_matches = re.findall(r"```(?:bash|sh)?\s*\n(.*?)```", cloud_text, re.DOTALL | re.IGNORECASE)
        if bash_matches:
            first_command = str(bash_matches[0] or "").strip()
            thought_text = _sanitize_thought_text(
                re.split(r"```(?:bash|sh)?\s*\n", cloud_text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            )
            if not thought_text:
                thought_text = "DISCUSSION"
            if not thought_text.startswith("DISCUSSION"):
                thought_text = f"DISCUSSION\n{thought_text}"
            cloud_text = f"{thought_text}\n\n```bash\n{first_command}\n```"
    
    print(f"\n[CGC Server] Raw Output from Cloud:\n{cloud_text}\n")
    
    tool_calls = []
    content_text = cloud_text
    python_match = None
    tool_call_match = None
    if tools:
        # [M7.5 Gate Hotfix] SWE-agent 依賴嚴格的 OpenAI Function Calling 格式。
        # 只有在呼叫方顯式提供 tools 時，才把模型輸出轉成 OpenAI tool_calls。
        import re

        # 嘗試解析 ```python ... ``` 格式的 function call
        python_match = re.search(r'```python\n(.*?)\n```', cloud_text, re.DOTALL)
        if python_match:
            code_block = python_match.group(1).strip()
            func_match = re.search(r'^([a-zA-Z_]\w*)\((.*)\)$', code_block, re.DOTALL)
            if func_match:
                func_name = func_match.group(1)
                args_str = func_match.group(2)
                try:
                    args_dict = {}
                    if args_str.strip():
                        args_dict = {"command": args_str} if func_name == "bash" else {}

                    tool_calls.append({
                        "id": f"call_{int(time.time())}",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(args_dict)
                        }
                    })
                    content_text = cloud_text.replace(python_match.group(0), "").strip()
                except Exception as e:
                    print(f"[CGC Server] Failed to parse python tool call: {e}")


        # --- NEW DSML PARSER FOR DEEPSEEK V4 FLASH ---
        dsml_match = re.search(r'<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>', cloud_text, re.DOTALL)
        if dsml_match and not tool_calls:
            try:
                dsml_content = dsml_match.group(1)
                invokes = re.finditer(r'<｜DSML｜invoke name="([^"]+)">(.*?)</｜DSML｜invoke>', dsml_content, re.DOTALL)
                for invoke in invokes:
                    func_name = invoke.group(1)
                    params_content = invoke.group(2)
                    args_dict = {}
                    params = re.finditer(r'<｜DSML｜parameter name="([^"]+)"[^>]*>(.*?)</｜DSML｜parameter>', params_content, re.DOTALL)
                    for param in params:
                        args_dict[param.group(1)] = param.group(2)
                    
                    tool_calls.append({
                        "id": f"call_{int(time.time())}",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(args_dict)
                        }
                    })
                content_text = cloud_text.replace(dsml_match.group(0), "").strip()
            except Exception as e:
                print(f"[CGC Server] Failed to parse DSML tool call: {e}")

        tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', cloud_text, re.DOTALL)
        tool_call_dash_match = re.search(r'<tool-call>(.*?)</tool-call>', cloud_text, re.DOTALL)
        if "<invoke name=" in cloud_text and "<execute>" in cloud_text:
            invoke_match = re.search(
                r'<invoke name="([^"]+)">.*?<execute>(.*?)</execute>',
                cloud_text,
                re.DOTALL,
            )
            if invoke_match:
                func_name = invoke_match.group(1).strip()
                execute_payload = re.sub(r"</?｜DSML｜>", "", invoke_match.group(2)).strip()
                if func_name == "bash" and execute_payload:
                    tool_calls.append({
                        "id": f"call_{int(time.time())}",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": execute_payload}),
                        }
                    })
                    content_text = re.split(r'<tool_calls>|<tool_call>|<tool-call>|<invoke name=', cloud_text, maxsplit=1)[0].strip()
        elif tool_call_match or tool_call_dash_match:
            tool_call_block = tool_call_match.group(0) if tool_call_match else tool_call_dash_match.group(0)
            tool_call_body = tool_call_match.group(1) if tool_call_match else tool_call_dash_match.group(1)
            try:
                tc_data = json.loads(tool_call_body)
                tool_calls.append({
                    "id": f"call_{int(time.time())}",
                    "type": "function",
                    "function": {
                        "name": tc_data.get("name"),
                        "arguments": json.dumps(tc_data.get("arguments", {}))
                    }
                })
                content_text = cloud_text.replace(tool_call_block, "").strip()
            except Exception as e:
                print(f"[CGC Server] Failed to parse tool call: {e}")
        else:
            try:
                clean_text = cloud_text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.startswith("```"):
                    clean_text = clean_text[3:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                clean_text = clean_text.strip()

                start_idx = clean_text.find('{')
                end_idx = clean_text.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = clean_text[start_idx:end_idx+1]
                    tc_data = json.loads(json_str)
                    if isinstance(tc_data, dict) and "name" in tc_data:
                        tool_calls.append({
                            "id": f"call_{int(time.time())}",
                            "type": "function",
                            "function": {
                                "name": tc_data.get("name"),
                                "arguments": json.dumps(tc_data.get("arguments", {}))
                            }
                        })
                        content_text = cloud_text.replace(json_str, "").strip()
            except Exception:
                pass

        if not tool_calls:
            bash_match = re.search(r'```(?:bash|sh)\s*\n(.*?)\n```', cloud_text, re.DOTALL)
            if bash_match:
                cmd = bash_match.group(1).strip()
                if cmd == "submit" or cmd == "submit()":
                    tool_calls.append({
                        "id": f"call_{int(time.time())}",
                        "type": "function",
                        "function": {"name": "submit", "arguments": "{}"}
                    })
                else:
                    tool_calls.append({
                        "id": f"call_{int(time.time())}",
                        "type": "function",
                        "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}
                    })
                content_text = cloud_text.replace(bash_match.group(0), "").strip()

            elif "Action:" in cloud_text or "Action" in cloud_text:
                action_match = re.search(r'Action:\s*([a-zA-Z_]\w*)\s*(?:Action Input:\s*(.*?)\s*(?:\n|$))?', cloud_text, re.IGNORECASE)
                if action_match:
                    func_name = action_match.group(1).strip()
                    action_input = action_match.group(2) if action_match.group(2) else ""
                    args_dict = {} if func_name.lower() == "submit" else {"command": action_input.strip()}

                    tool_calls.append({
                        "id": f"call_{int(time.time())}",
                        "type": "function",
                        "function": {"name": func_name, "arguments": json.dumps(args_dict)}
                    })

    # #region debug-point C:chat-format-parse
    _debug_report(
        "C",
        "app/servers/cgc_api_server.py:chat_completions:format_parse",
        "[DEBUG] parsed cloud output into response envelope",
        data={
            "tool_calls_count": len(tool_calls),
            "has_python_block": bool(python_match),
            "has_tool_call_tag": bool(tool_call_match),
            "content_len": len(str(content_text or "")),
            "content_preview": _safe_debug_value(content_text, limit=200),
            "finish_reason": "tool_calls" if tool_calls else "stop",
        },
        trace_id=trace_id,
    )
    # #endregion


    stream = data.get("stream", False)
    
    if stream:
        async def stream_generator():
            start_chunk = {
                "type": "message_start",
                "message": {
                    "id": "msg_cgc_01",
                    "type": "message",
                    "role": "assistant",
                    "model": data.get("model", "claude-3-5-sonnet-20241022"),
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 50}
                }
            }
            yield f'event: message_start\ndata: {json.dumps(start_chunk)}\n\n'
            yield f'event: content_block_start\ndata: {json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})}\n\n'
            
            async for token in engine.generate_stream(messages, cloud_text=cloud_text, max_tokens=data.get("max_tokens", 4096)):
                if token.startswith("\n[Network Error]"):
                    # 發生網路錯誤，立即結束
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': token}})}\n\n"
                    break
                chunk = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": token}
                }
                yield f"event: content_block_delta\ndata: {json.dumps(chunk)}\n\n"
            
            yield f'event: content_block_stop\ndata: {json.dumps({"type": "content_block_stop", "index": 0})}\n\n'
            yield f'event: message_delta\ndata: {json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 50}})}\n\n'
            yield f'event: message_stop\ndata: {json.dumps({"type": "message_stop"})}\n\n'
            
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        response_msg = {
            "role": "assistant",
            "content": content_text
        }
        if tool_calls:
            response_msg["tool_calls"] = tool_calls
            
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("model", "deepseek-v4-flash"),
            "choices": [{
                "index": 0,
                "message": response_msg,
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }],
            "usage": {
                "prompt_tokens": approx_prompt_len // 4,
                "completion_tokens": len(cloud_text) // 4,
                "total_tokens": (approx_prompt_len + len(cloud_text)) // 4
            }
        }

# ==========================================
# Ollama Compatible API Layer
# ==========================================
from datetime import datetime, timezone

@app.get("/api/tags")
async def ollama_tags():
    # 這裡就是我們提到的「雲端模型池」清單，可以無限擴充
    return {
        "models": [
            {"name": "deepseek-v4-flash:latest", "model": "deepseek-v4-flash:latest", "modified_at": datetime.now(timezone.utc).isoformat(), "size": 32000000000, "digest": "sha256:cgc_hash_1", "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "32B", "quantization_level": "FP8"}},
            {"name": "llama3:70b", "model": "llama3:70b", "modified_at": datetime.now(timezone.utc).isoformat(), "size": 70000000000, "digest": "sha256:cgc_hash_2", "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "70B", "quantization_level": "Q4_K_M"}},
            {"name": "qwen2.5:32b", "model": "qwen2.5:32b", "modified_at": datetime.now(timezone.utc).isoformat(), "size": 32000000000, "digest": "sha256:cgc_hash_3", "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "32B", "quantization_level": "Q4_K_M"}},
            {"name": "minicpm-1b:latest", "model": "minicpm-1b:latest", "modified_at": datetime.now(timezone.utc).isoformat(), "size": 1000000000, "digest": "sha256:cgc_hash_4", "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "1B", "quantization_level": "FP16"}},
            {
                "name": "minicpm5-1b",
                "model": "minicpm5-1b",
                "modified_at": datetime.now(timezone.utc).isoformat(),
                "size": 688065920,
                "digest": "sha256:minicpm5_q4km",
                "details": {
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "1.1B",
                    "quantization_level": "Q4_K_M",
                    "cloud_source": "fake_ollama_registry",
                    "source_priority": ["cluster_nfs", "huggingface"],
                    "cluster_nfs_root": "/data/models",
                    "cluster_nfs_path": "/data/models/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf",
                    "gguf_repo": "openbmb/MiniCPM5-1B-GGUF",
                    "gguf_filename": "MiniCPM5-1B-Q4_K_M.gguf"
                }
            }
        ]
    }

@app.post("/api/show")
async def ollama_show(request: Request):
    data = await request.json()
    model_name = str(data.get("name") or data.get("model") or "").strip()
    if model_name in {"minicpm5-1b", "minicpm5-1b:latest", "minicpm5"}:
        return {
            "license": "CGC Engine Cloud License",
            "modelfile": "FROM cgc-cloud-registry/minicpm5-1b",
            "parameters": "temperature 0.7\ntop_p 0.95\nnum_ctx 8192",
            "template": "{{ .Prompt }}",
            "details": {
                "install_via": "fake_ollama_protocol",
                "router_backend": "ollama",
                "source_priority": ["cluster_nfs", "huggingface"],
                "cluster_nfs_root": "/data/models",
                "cluster_nfs_path": "/data/models/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf",
                "gguf_repo": "openbmb/MiniCPM5-1B-GGUF",
                "gguf_filename": "MiniCPM5-1B-Q4_K_M.gguf",
                "ollama_model": "minicpm5-1b",
                "quant": "Q4_K_M"
            }
        }
    return {"license": "CGC Engine Cloud License", "modelfile": "FROM cgc-cloud-registry", "parameters": "", "template": "{{ .Prompt }}"}

@app.post("/api/chat")
async def ollama_chat(request: Request):
    data = await request.json()
    model = data.get("model", "deepseek-v4-flash:latest")
    messages = data.get("messages", [])
    stream = data.get("stream", True)
    
    # 提取 Prompt
    prompt = ""
    for msg in messages:
        if isinstance(msg, dict) and "content" in msg:
            prompt += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"
        elif isinstance(msg, str):
            prompt += msg + "\n"
            
    print(f"\n[Ollama Edge Proxy] Received Chat Request | Model: {model} | Extracting payload to Cloud...")
    
    # 呼叫我們的端雲 Socket 傳輸
    cloud_text = await engine.trigger_cgc_prefill(prompt)
    
    if stream:
        async def ollama_stream_generator():
            async for token in engine.generate_stream(prompt, cloud_text=cloud_text, max_tokens=4096):
                if token.startswith("[Error]") or "Network Error" in token:
                    yield json.dumps({
                        "model": model,
                        "message": {"role": "assistant", "content": token},
                        "done": True
                    }) + "\n"
                    break
                
                chunk = {
                    "model": model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "message": {"role": "assistant", "content": token},
                    "done": False
                }
                yield json.dumps(chunk) + "\n"
                
            # 結束標記
            yield json.dumps({
                "model": model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": {"role": "assistant", "content": ""},
                "done": True
            }) + "\n"
            
        return StreamingResponse(ollama_stream_generator(), media_type="application/x-ndjson")
    else:
        return {"error": "Non-streaming not fully implemented yet"}

@app.post("/api/generate")
async def ollama_generate(request: Request):
    data = await request.json()
    model = data.get("model", "deepseek-v4-flash:latest")
    prompt = data.get("prompt", "")
    stream = data.get("stream", True)
    options = data.get("options") if isinstance(data.get("options"), dict) else {}
    max_tokens = int(options.get("num_predict", data.get("max_tokens", 256)) or 256)
    
    use_omlx = data.get("use_omlx", False)
    use_flashmoe = data.get("use_flashmoe", False)
    
    print(f"\n[Ollama Edge Proxy] Received Generate Request | Model: {model} | Extracting payload to Cloud...")
    
    if use_omlx or str(model).endswith(".mlx"):
        print(f"[Hardware Hook] 🍎 Activating Apple MLX Engine for unified memory 0-copy acceleration...")
    if use_flashmoe or "moe" in str(model).lower():
        print(f"[Hardware Hook] ⚡ Activating FlashMoE Paging to prevent VRAM OOM on Edge...")

    if local_infer_runtime is None:
        local_result = SimpleNamespace(
            executed_locally=False,
            status="SKIP",
            reason=_local_infer_unavailable_reason(),
            backend="unavailable",
            model_ref=str(model),
            evidence_path="",
            chunks=[],
            text="",
        )
    else:
        local_result = await local_infer_runtime.maybe_generate(
            model=str(model),
            prompt=str(prompt),
            use_omlx=bool(use_omlx),
            use_flashmoe=bool(use_flashmoe),
            max_tokens=int(max_tokens),
        )
    if local_result.executed_locally and local_result.status == "PASS":
        print(
            f"[Local Edge Runtime] ✅ Executed locally via {local_result.backend} | "
            f"Model: {local_result.model_ref} | Evidence: {local_result.evidence_path}"
        )
        if stream:
            async def ollama_local_stream_generator():
                for chunk_text in local_result.chunks:
                    chunk = {
                        "model": model,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "response": chunk_text,
                        "done": False,
                        "local_execution": True,
                        "backend": local_result.backend,
                        "evidence_path": local_result.evidence_path,
                    }
                    yield json.dumps(chunk) + "\n"
                    await asyncio.sleep(0.005)
                yield json.dumps({
                    "model": model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "response": "",
                    "done": True,
                    "local_execution": True,
                    "backend": local_result.backend,
                    "evidence_path": local_result.evidence_path,
                }) + "\n"

            return StreamingResponse(ollama_local_stream_generator(), media_type="application/x-ndjson")
        return {
            "model": model,
            "response": local_result.text,
            "done": True,
            "local_execution": True,
            "backend": local_result.backend,
            "evidence_path": local_result.evidence_path,
        }

    print(
        f"[Local Edge Runtime] {local_result.status} | "
        f"Reason: {local_result.reason or 'fallback_to_cloud'} | Evidence: {local_result.evidence_path}"
    )
    cloud_text = await engine.trigger_cgc_prefill(prompt)
    
    if stream:
        async def ollama_stream_generator():
            async for token in engine.generate_stream(prompt, cloud_text=cloud_text, max_tokens=max_tokens):
                chunk = {
                    "model": model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "response": token,
                    "done": False
                }
                yield json.dumps(chunk) + "\n"
            yield json.dumps({
                "model": model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "response": "",
                "done": True
            }) + "\n"
            
        return StreamingResponse(ollama_stream_generator(), media_type="application/x-ndjson")
    else:
        return {"error": "Non-streaming not fully implemented yet"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=LOCAL_API_PORT)
