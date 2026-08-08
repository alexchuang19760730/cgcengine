from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "ComputeGraphCompiler-main"
for candidate in (REPO_ROOT, ENGINE_ROOT):
    raw = str(candidate.resolve())
    if raw not in sys.path:
        sys.path.insert(0, raw)

from cgc_engine.pipeline import TinyDeepSeekV4WithCache


_MODEL_CACHE: dict[tuple[int, int, int, int, int, str], TinyDeepSeekV4WithCache] = {}
_MODEL_LOCK = threading.Lock()


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name) or "").strip()
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(1, value)


def _device_prefix(device: str) -> str:
    return str(device or "").strip().lower().split(":", 1)[0]


def _device_available(device: str) -> bool:
    prefix = _device_prefix(device)
    if prefix == "cuda":
        return bool(torch.cuda.is_available())
    if prefix == "mps":
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return prefix == "cpu"


def _device_supports_zero_copy(device: str) -> bool:
    return _device_prefix(device) in {"cuda", "mps"}


def _reference_config() -> dict[str, int]:
    hidden_size = _env_int("CGC_M75_KDA_HIDDEN_SIZE", 512)
    num_heads = _env_int("CGC_M75_KDA_NUM_HEADS", 8)
    if hidden_size % num_heads != 0:
        hidden_size = ((hidden_size + num_heads - 1) // num_heads) * num_heads
    return {
        "vocab_size": _env_int("CGC_M75_KDA_VOCAB_SIZE", 4096),
        "hidden_size": hidden_size,
        "num_layers": _env_int("CGC_M75_KDA_NUM_LAYERS", 2),
        "num_heads": num_heads,
        "seq_len": _env_int("CGC_M75_KDA_SEQ_LEN", 16),
        "seed": _env_int("CGC_M75_KDA_MODEL_SEED", 20250618),
    }


def _resolve_resume_device() -> str:
    configured = str(os.environ.get("CGC_M75_KDA_RESUME_DEVICE") or "").strip().lower()
    if configured and _device_available(configured):
        return configured
    if torch.cuda.is_available():
        return str(os.environ.get("CGC_M75_KDA_CUDA_DEVICE") or "cuda:0").strip() or "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_state_device() -> str:
    configured = str(os.environ.get("CGC_M75_KDA_STATE_DEVICE") or "").strip().lower()
    if configured and _device_available(configured):
        return configured
    return _resolve_resume_device()


def _torch_save_any(payload: Any) -> bytes:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def _torch_load_any(payload: bytes | bytearray | memoryview, *, map_location: str = "cpu") -> Any:
    buf = io.BytesIO(payload)
    try:
        return torch.load(buf, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(buf, map_location=map_location)


def _cq4_pack_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    payload = tensor.detach()
    max_abs = float(payload.abs().max().item()) if payload.numel() else 0.0
    scale = max(max_abs / 7.0, 1e-8)
    quantized = torch.clamp(torch.round(payload / scale), -8, 7).to(torch.int8)
    return {
        "shape": list(payload.shape),
        "scale": scale,
        "quantized": quantized.contiguous(),
    }


def _cq4_unpack_tensor(payload: Dict[str, Any], *, device: str) -> torch.Tensor:
    quantized = payload.get("quantized")
    if not isinstance(quantized, torch.Tensor):
        raise RuntimeError("invalid_cq4_quantized_tensor")
    scale = float(payload.get("scale") or 0.0)
    if scale <= 0.0:
        raise RuntimeError("invalid_cq4_scale")
    restored = quantized.to(device=device, dtype=torch.float16) * scale
    expected_shape = list(payload.get("shape") or [])
    if expected_shape and list(restored.shape) != expected_shape:
        restored = restored.reshape(expected_shape)
    return restored


def _encode_kda_state_payload_cq4(state_payload: Dict[str, Any]) -> tuple[bytes, Dict[str, Any]]:
    raw_reference_bytes = _torch_save_any(state_payload)
    cq4_payload = {
        "schema_version": int(state_payload.get("schema_version") or 1),
        "kind": "kda_state_v1",
        "state_source": str(state_payload.get("state_source") or ""),
        "trace_id": str(state_payload.get("trace_id") or ""),
        "config": state_payload.get("config") if isinstance(state_payload.get("config"), dict) else {},
        "prompt_len": int(state_payload.get("prompt_len") or 0),
        "prompt_ids": state_payload.get("prompt_ids"),
        "seed_token_id": int(state_payload.get("seed_token_id") or 0),
        "tensor_layout": str(state_payload.get("tensor_layout") or "S_all_stacked_v1"),
        "S_all_cq4": _cq4_pack_tensor(state_payload["S_all"]),
        "raw_state_bytes": len(raw_reference_bytes),
        "compressed_state_bytes": 0,
        "selected_payload_bytes": 0,
        "compression_ratio": 1.0,
        "compression_codec_selected": "cq4",
    }
    encoded_bytes = _torch_save_any(cq4_payload)
    cq4_payload["compressed_state_bytes"] = len(encoded_bytes)
    cq4_payload["selected_payload_bytes"] = len(encoded_bytes)
    cq4_payload["compression_ratio"] = (
        float(len(encoded_bytes)) / float(len(raw_reference_bytes))
        if raw_reference_bytes
        else 1.0
    )
    encoded_bytes = _torch_save_any(cq4_payload)
    return encoded_bytes, cq4_payload


def _get_reference_model(config: dict[str, int], *, device: str = "cpu") -> TinyDeepSeekV4WithCache:
    resolved_device = str(device or "cpu").strip() or "cpu"
    key = (
        int(config["vocab_size"]),
        int(config["hidden_size"]),
        int(config["num_layers"]),
        int(config["num_heads"]),
        int(config["seed"]),
        resolved_device,
    )
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        torch.manual_seed(int(config["seed"]))
        model = TinyDeepSeekV4WithCache(
            vocab_size=int(config["vocab_size"]),
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            num_heads=int(config["num_heads"]),
            use_kda=True,
        ).to(device=resolved_device, dtype=torch.float32)
        model.eval()
        _MODEL_CACHE[key] = model
        return model


def _extract_prompt_text(request_payload: Any) -> str:
    if isinstance(request_payload, dict):
        messages = request_payload.get("messages")
        if isinstance(messages, list):
            parts: list[str] = []
            for message in messages:
                if not isinstance(message, dict):
                    parts.append(str(message))
                    continue
                content = message.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            parts.append(str(item.get("text") or item.get("content") or ""))
                        else:
                            parts.append(str(item))
                else:
                    parts.append(str(content))
            joined = "\n".join(part for part in parts if part)
            if joined.strip():
                return joined
        prompt = request_payload.get("prompt")
        if prompt is not None:
            return str(prompt)
    if isinstance(request_payload, (dict, list)):
        return json.dumps(request_payload, ensure_ascii=False, sort_keys=True)
    return str(request_payload)


def _prompt_to_input_ids(prompt: str, *, vocab_size: int, seq_len: int) -> torch.Tensor:
    base = hashlib.sha256(str(prompt).encode("utf-8", errors="replace")).digest()
    ids: list[int] = []
    for index in range(int(seq_len)):
        digest = hashlib.sha256(base + int(index).to_bytes(4, "little", signed=False)).digest()
        ids.append(int.from_bytes(digest[:4], "little", signed=False) % int(vocab_size))
    return torch.tensor(ids, dtype=torch.long).view(1, int(seq_len))


def _layer_ffn(layer: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    if hasattr(layer, "get"):
        mlp = layer.get("mlp")
        if mlp is not None:
            return mlp(hidden_states)
        moe = layer.get("moe")
        if moe is not None:
            return moe(hidden_states)
    if "mlp" in layer:
        return layer["mlp"](hidden_states)
    return layer["moe"](hidden_states)


def _prefill_prefix_cache_kda_aot(model: TinyDeepSeekV4WithCache, input_ids: torch.Tensor) -> torch.Tensor:
    x = model.embed_tokens(input_ids)
    state_list: list[torch.Tensor] = []
    for layer in model.layers:
        attn_out, state = layer["csa"].prefill_kda_aot(x)
        x = x + attn_out
        x = x + _layer_ffn(layer, x)
        state_list.append(state)
    return torch.stack(state_list, dim=0)


def decode_one_step_kda_aot(
    model: TinyDeepSeekV4WithCache,
    token_ids: torch.Tensor,
    state_all: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = model.embed_tokens(token_ids)
    state_new_list: list[torch.Tensor] = []
    for index, layer in enumerate(model.layers):
        attn_out, state_new = layer["csa"].decode_one_kda_aot(x, state_all[index])
        x = x + attn_out
        x = x + _layer_ffn(layer, x)
        state_new_list.append(state_new)
    x = model.norm(x)
    logits = model.lm_head(x)
    return logits, torch.stack(state_new_list, dim=0)


def _load_kda_state_payload(
    *,
    state_kind: str,
    state_codec: str,
    state_bytes: bytes | bytearray | memoryview,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    if str(state_kind or "") != "kda_state_v1":
        raise RuntimeError(f"unsupported_state_kind:{state_kind}")
    payload = _torch_load_any(state_bytes, map_location=map_location)
    if not isinstance(payload, dict):
        raise RuntimeError("invalid_state_payload_root")
    if str(state_codec or "").strip() == "cq4":
        if str(payload.get("kind") or "") != "kda_state_v1":
            raise RuntimeError(f"invalid_cq4_state_payload_kind:{payload.get('kind')}")
        prompt_ids = payload.get("prompt_ids")
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.to(device=map_location)
        resolved_payload = dict(payload)
        resolved_payload["prompt_ids"] = prompt_ids
        resolved_payload["S_all"] = _cq4_unpack_tensor(
            payload.get("S_all_cq4") if isinstance(payload.get("S_all_cq4"), dict) else {},
            device=map_location,
        )
        resolved_payload["compression_codec_selected"] = "cq4"
        return resolved_payload
    if str(payload.get("kind") or "") != "kda_state_v1":
        raise RuntimeError(f"invalid_state_payload_kind:{payload.get('kind')}")
    return payload


def _summarize_kda_state_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    state_all = payload.get("S_all")
    prompt_ids = payload.get("prompt_ids")
    return {
        "schema_version": int(payload.get("schema_version") or 0),
        "kind": str(payload.get("kind") or ""),
        "state_source": str(payload.get("state_source") or ""),
        "trace_id": str(payload.get("trace_id") or ""),
        "prompt_len": int(payload.get("prompt_len") or 0),
        "seed_token_id": int(payload.get("seed_token_id") or 0),
        "config": payload.get("config") if isinstance(payload.get("config"), dict) else {},
        "tensor_layout": "S_all_stacked_v1",
        "S_all_shape": list(state_all.shape) if isinstance(state_all, torch.Tensor) else [],
        "prompt_ids_shape": list(prompt_ids.shape) if isinstance(prompt_ids, torch.Tensor) else [],
        "raw_state_bytes": int(payload.get("raw_state_bytes") or 0),
        "compressed_state_bytes": int(payload.get("compressed_state_bytes") or 0),
        "selected_payload_bytes": int(payload.get("selected_payload_bytes") or 0),
        "compression_ratio": float(payload.get("compression_ratio") or 1.0),
        "compression_codec_selected": str(payload.get("compression_codec_selected") or ""),
    }


def inspect_kda_state_bytes(
    *,
    state_kind: str,
    state_codec: str,
    state_bytes: bytes | bytearray | memoryview,
) -> Dict[str, Any]:
    payload = _load_kda_state_payload(
        state_kind=state_kind,
        state_codec=state_codec,
        state_bytes=state_bytes,
    )
    return _summarize_kda_state_payload(payload)


def build_real_kda_state_from_request(
    request_payload: Any,
    *,
    trace_id: str,
) -> Dict[str, Any]:
    prompt_text = _extract_prompt_text(request_payload)
    config = _reference_config()
    state_device = _resolve_state_device()
    model = _get_reference_model(config, device=state_device)
    input_ids = _prompt_to_input_ids(
        prompt_text,
        vocab_size=int(config["vocab_size"]),
        seq_len=int(config["seq_len"]),
    ).to(device=state_device)
    with torch.no_grad():
        state_all = _prefill_prefix_cache_kda_aot(model, input_ids)
        prefill_logits = model(input_ids)
        seed_token_id = int(prefill_logits[:, -1, :].argmax(dim=-1)[0].item())
    state_payload = {
        "schema_version": 1,
        "kind": "kda_state_v1",
        "state_source": "prefill_prefix_cache_kda_aot",
        "trace_id": str(trace_id),
        "config": config,
        "prompt_len": int(input_ids.shape[1]),
        "prompt_ids": input_ids.detach(),
        "seed_token_id": seed_token_id,
        "S_all": state_all.detach(),
        "tensor_layout": "S_all_stacked_v1",
        "raw_state_bytes": 0,
        "compressed_state_bytes": 0,
        "selected_payload_bytes": 0,
        "compression_ratio": 1.0,
        "compression_codec_selected": "",
    }
    state_bytes, encoded_payload = _encode_kda_state_payload_cq4(state_payload)
    return {
        "state_kind": "kda_state_v1",
        "state_codec": "cq4",
        "state_meta": {
            "state_source": "prefill_prefix_cache_kda_aot",
            "prompt_len": int(input_ids.shape[1]),
            "num_layers": int(state_all.shape[0]),
            "seed_token_id": seed_token_id,
            "tensor_layout": "S_all_stacked_v1",
            "state_device": state_device,
            "raw_state_bytes": int(encoded_payload.get("raw_state_bytes") or 0),
            "compressed_state_bytes": int(encoded_payload.get("compressed_state_bytes") or len(state_bytes)),
            "selected_payload_bytes": len(state_bytes),
            "compression_ratio": float(encoded_payload.get("compression_ratio") or 1.0),
            "compression_codec_selected": "cq4",
        },
        "state_bytes": state_bytes,
    }


def resume_one_token_from_kda_state(
    *,
    state_kind: str,
    state_codec: str,
    state_bytes: bytes | bytearray | memoryview,
    trace_id: str,
) -> Dict[str, Any]:
    preferred_device = _resolve_resume_device()
    actual_device = preferred_device
    try:
        payload = _load_kda_state_payload(
            state_kind=state_kind,
            state_codec=state_codec,
            state_bytes=state_bytes,
            map_location=preferred_device,
        )
    except Exception:
        actual_device = "cpu"
        payload = _load_kda_state_payload(
            state_kind=state_kind,
            state_codec=state_codec,
            state_bytes=state_bytes,
            map_location="cpu",
        )

    config = payload.get("config") if isinstance(payload.get("config"), dict) else _reference_config()
    model = _get_reference_model({key: int(value) for key, value in config.items()}, device=actual_device)
    state_all = payload.get("S_all")
    if not isinstance(state_all, torch.Tensor):
        raise RuntimeError("missing_S_all_tensor")
    seed_token_id = int(payload.get("seed_token_id") or 0)
    token_ids = torch.tensor([[seed_token_id]], dtype=torch.long, device=actual_device)
    with torch.no_grad():
        logits, state_new = decode_one_step_kda_aot(model, token_ids, state_all)
        next_token_id = int(logits[:, -1, :].argmax(dim=-1)[0].item())

    state_summary = _summarize_kda_state_payload(payload)
    loaded_tensor_device = str(state_all.device)
    resume_tensor_device = str(state_new.device)
    device_resume_consumed = (
        _device_supports_zero_copy(actual_device)
        and _device_prefix(loaded_tensor_device) == _device_prefix(actual_device)
        and _device_prefix(resume_tensor_device) == _device_prefix(actual_device)
    )
    cpu_copy_count = 0 if device_resume_consumed else 1
    uma_buffer_used = _device_prefix(actual_device) == "mps"
    state_summary["loaded_tensor_device"] = loaded_tensor_device
    return {
        "trace_id": str(trace_id),
        "resume_decode_executed": True,
        "seed_token_id": seed_token_id,
        "edge_token_id": next_token_id,
        "text": f"<kda:{next_token_id}>",
        "state_summary": state_summary,
        "next_state_shape": list(state_new.shape),
        "resume_tensor_device": resume_tensor_device,
        "zero_copy_runtime": {
            "cpu_copy_count": cpu_copy_count,
            "uma_buffer_used": uma_buffer_used,
            "device_resume_consumed": device_resume_consumed,
            "preferred_device": preferred_device,
            "actual_device": actual_device,
            "loaded_tensor_device": loaded_tensor_device,
            "zero_copy_state_codec": state_codec == "cq4",
        },
    }
