from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

LOCAL_FULL_MODEL_PATH = os.environ.get(
    "EDGE_LOCAL_MAIN_MODEL_PATH",
    os.environ.get("EDGE_LOCAL_MODEL_PATH", ""),
)
LAYER_SPLIT_MODEL_PATH = os.environ.get(
    "EDGE_LOCAL_MLX_MODEL_PATH",
    os.environ.get("EDGE_LOCAL_MODEL_PATH", ""),
)
_LOCAL_NUM_LAYERS_ENV = str(os.environ.get("EDGE_LOCAL_NUM_LAYERS", "") or "").strip()
LOCAL_NUM_LAYERS = int(_LOCAL_NUM_LAYERS_ENV or "0")
DEFAULT_LOCAL_NUM_LAYERS = 32
LOCAL_PARAMS = int(os.environ.get("EDGE_LOCAL_PARAMS", "0"))
LOCAL_KV_HEAD_DIM = int(os.environ.get("EDGE_LOCAL_KV_HEAD_DIM", "128"))
LOCAL_KV_HEADS = int(os.environ.get("EDGE_LOCAL_KV_HEADS", "8"))
LOCAL_MEM_SAFETY = float(os.environ.get("EDGE_LOCAL_MEM_SAFETY", "0.8"))
EDGE_DENSE_LAYER_STREAMING_ENABLED = os.environ.get("EDGE_DENSE_LAYER_STREAMING_ENABLED", "1") == "1"

ROUTE_LOCAL_FULL = "local_full"
ROUTE_LAYER_SPLIT_PD = "layer_split_pd"
ROUTE_CLOUD_PD = "cloud_pd"
ROUTE_CLOUD_FALLBACK = "cloud_fallback"

MAC_TFLOPS = float(os.environ.get("EDGE_MAC_TFLOPS", "30"))
CLOUD_TFLOPS = float(os.environ.get("EDGE_CLOUD_TFLOPS", "3000"))
RTT_SEC = float(os.environ.get("EDGE_RTT_SEC", "0.05"))
LAYER_ACTIVATION_BYTES = int(os.environ.get("EDGE_LAYER_ACTIVATION_BYTES", "0"))
EDGE_EXPERT_STREAMING_ENABLED = os.environ.get("EDGE_EXPERT_STREAMING_ENABLED", "1") == "1"
EDGE_EXPERT_RAM_BUDGET_BYTES = int(os.environ.get("EDGE_EXPERT_RAM_BUDGET_BYTES", str(8 * 1024**3)))
EDGE_ROUTE_MEM_CRITICAL_BYTES = int(float(os.environ.get("EDGE_ROUTE_MEM_CRITICAL_GB", "1.0")) * 1024**3)
EDGE_ROUTE_MEM_SAFE_BYTES = int(float(os.environ.get("EDGE_ROUTE_MEM_SAFE_GB", "3.0")) * 1024**3)
EDGE_ROUTE_STICKY_WINDOW_SEC = max(float(os.environ.get("EDGE_ROUTE_STICKY_WINDOW_SEC", "45") or "45"), 0.0)
CGC_LOW_MEMORY_STATE_PATH = os.environ.get("CGC_LOW_MEMORY_STATE_PATH", "")
CGC_LOCAL_FLASHMOE_MODEL = os.environ.get("CGC_LOCAL_FLASHMOE_MODEL", "")

_ROUTE_DEGRADE_CHAIN = [
    ROUTE_LOCAL_FULL,
    ROUTE_LAYER_SPLIT_PD,
    ROUTE_CLOUD_PD,
    ROUTE_CLOUD_FALLBACK,
]
_ROUTE_RANK = {mode: idx for idx, mode in enumerate(_ROUTE_DEGRADE_CHAIN)}
_route_sticky_lock = threading.Lock()
_route_sticky_state = {
    "mode": "",
    "until": 0.0,
    "reason": "",
}


def mac_supports_arch(model_name: str) -> bool:
    name = (model_name or "").lower()
    for token in ("deepseek-v4", "deepseek_v4", "dsv4", "v4-flash"):
        if token in name:
            return False
    return True


def mac_available_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 0


def model_weight_bytes() -> int:
    if LOCAL_PARAMS > 0:
        return LOCAL_PARAMS * 2
    try:
        if LOCAL_FULL_MODEL_PATH and os.path.exists(LOCAL_FULL_MODEL_PATH):
            return int(os.path.getsize(LOCAL_FULL_MODEL_PATH))
    except Exception:
        pass
    return 0


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _is_moe_candidate(model_name: str) -> bool:
    name = (model_name or "").lower()
    return any(token in name for token in (
        "moe",
        "a3b",
        "a4b",
        "deepseek",
        "dsv4",
        "glm-5.2",
        "olmoe",
        "inkling",
    ))


def _resolve_model_registry_config(model_name: str, model_path: str):
    try:
        from app.shared.model_registry import get_model_config, get_model_config_by_path
    except Exception:
        return None
    for resolver, value in (
        (get_model_config, model_name),
        (get_model_config_by_path, model_path),
    ):
        text = str(value or "").strip()
        if not text:
            continue
        try:
            return resolver(text)
        except Exception:
            continue
    return None


def _runtime_model_profile(model_name: str) -> dict[str, Any]:
    cfg = _resolve_model_registry_config(model_name, LOCAL_FULL_MODEL_PATH)
    local_num_layers = LOCAL_NUM_LAYERS
    if local_num_layers <= 0 and cfg is not None:
        local_num_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    elif cfg is not None:
        local_num_layers = max(local_num_layers, int(getattr(cfg, "num_hidden_layers", 0) or 0))
    if local_num_layers <= 0:
        local_num_layers = DEFAULT_LOCAL_NUM_LAYERS

    model_weight = model_weight_bytes()
    moe_candidate = _is_moe_candidate(model_name)
    path_hint = str(LOCAL_FULL_MODEL_PATH or "").strip().lower()
    if any(token in path_hint for token in ("e4b", "e2b", "dense")):
        moe_candidate = False
    elif any(token in path_hint for token in ("a3b", "a4b", "moe")):
        moe_candidate = True
    if cfg is not None:
        moe_candidate = bool(getattr(cfg, "is_moe", False)) or moe_candidate
        if any(token in path_hint for token in ("e4b", "e2b", "dense")):
            moe_candidate = False

    return {
        "cfg": cfg,
        "local_num_layers": max(int(local_num_layers or 0), 0),
        "model_weight_bytes": max(int(model_weight or 0), 0),
        "moe_candidate": moe_candidate,
    }


def _external_low_memory_signal() -> tuple[bool, str, str]:
    low_memory = _env_flag("CGC_LOW_MEMORY_DETECTED") or _env_flag("EDGE_LOW_MEMORY_DETECTED")
    pressure = str(os.environ.get("CGC_MEMORY_PRESSURE", "") or os.environ.get("EDGE_MEMORY_PRESSURE", "")).strip().lower()
    reason = str(os.environ.get("CGC_LOW_MEMORY_REASON", "") or os.environ.get("EDGE_LOW_MEMORY_REASON", "")).strip()
    if CGC_LOW_MEMORY_STATE_PATH and os.path.exists(CGC_LOW_MEMORY_STATE_PATH):
        try:
            with open(CGC_LOW_MEMORY_STATE_PATH, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            low_memory = bool(payload.get("low_memory_detected", low_memory))
            pressure = str(payload.get("memory_pressure", pressure) or pressure).strip().lower()
            reason = str(payload.get("reason", reason) or reason).strip()
        except Exception:
            pass
    return low_memory, pressure, reason


def _memory_pressure_label(
    *,
    avail_bytes: int,
    avail_safe_bytes: int,
    moe_required_bytes: int,
    external_low_memory: bool,
    external_pressure: str,
) -> str:
    if external_pressure in {"critical", "elevated", "normal"}:
        return external_pressure
    if external_low_memory or (avail_bytes > 0 and avail_bytes <= EDGE_ROUTE_MEM_CRITICAL_BYTES):
        return "critical"
    if moe_required_bytes > 0 and avail_safe_bytes < moe_required_bytes:
        return "elevated"
    if avail_bytes > 0 and avail_bytes <= EDGE_ROUTE_MEM_SAFE_BYTES:
        return "elevated"
    return "normal" if avail_bytes > 0 else "unknown"


def _apply_sticky_window(desired_mode: str, reason: str) -> tuple[str, bool, int, str]:
    now = time.time()
    sticky_until_epoch_ms = 0
    sticky_active = False
    final_mode = desired_mode
    effective_reason = reason
    with _route_sticky_lock:
        sticky_mode = str(_route_sticky_state.get("mode") or "")
        sticky_until = float(_route_sticky_state.get("until") or 0.0)
        sticky_reason = str(_route_sticky_state.get("reason") or "")
        sticky_valid = bool(sticky_mode) and sticky_until > now

        if sticky_valid and _ROUTE_RANK.get(desired_mode, 999) < _ROUTE_RANK.get(sticky_mode, 999):
            final_mode = sticky_mode
            sticky_active = True
            sticky_until_epoch_ms = int(sticky_until * 1000)
            effective_reason = f"sticky_hold:{sticky_reason}" if sticky_reason else "sticky_hold"
        else:
            if sticky_valid and _ROUTE_RANK.get(desired_mode, 999) >= _ROUTE_RANK.get(sticky_mode, 999):
                _route_sticky_state["mode"] = ""
                _route_sticky_state["until"] = 0.0
                _route_sticky_state["reason"] = ""
            if desired_mode != ROUTE_LOCAL_FULL and EDGE_ROUTE_STICKY_WINDOW_SEC > 0:
                until = now + EDGE_ROUTE_STICKY_WINDOW_SEC
                _route_sticky_state["mode"] = desired_mode
                _route_sticky_state["until"] = until
                _route_sticky_state["reason"] = reason
                sticky_until_epoch_ms = int(until * 1000)
            else:
                _route_sticky_state["mode"] = ""
                _route_sticky_state["until"] = 0.0
                _route_sticky_state["reason"] = ""
    return final_mode, sticky_active, sticky_until_epoch_ms, effective_reason


def estimate_kv_bytes(prompt_len: int, max_tokens: int, num_layers: int | None = None) -> int:
    seq = max(int(prompt_len or 0) + int(max_tokens or 0), 1)
    nl = num_layers if num_layers is not None else (LOCAL_NUM_LAYERS or DEFAULT_LOCAL_NUM_LAYERS)
    return seq * nl * LOCAL_KV_HEAD_DIM * LOCAL_KV_HEADS * 2 * 2


def estimate_mac_prefill_time(P: int, prompt_len: int) -> float:
    flops_per_layer_token = 2 * 4096 * 4096
    total_flops = P * max(prompt_len, 1) * flops_per_layer_token
    if MAC_TFLOPS <= 0:
        return 999.0
    return total_flops / (MAC_TFLOPS * 1e12)


def estimate_cloud_full_prefill_time(prompt_len: int) -> float:
    total_layers = int(os.environ.get("EDGE_CLOUD_NUM_LAYERS", "42"))
    flops_per_layer_token = 2 * 4096 * 4096
    total_flops = total_layers * max(prompt_len, 1) * flops_per_layer_token
    if CLOUD_TFLOPS <= 0:
        return 999.0
    return total_flops / (CLOUD_TFLOPS * 1e12)


def latency_preflight_ok(P: int, prompt_len: int) -> bool:
    mac_time = estimate_mac_prefill_time(P, prompt_len)
    cloud_time = estimate_cloud_full_prefill_time(prompt_len)
    total_split = mac_time + RTT_SEC
    return total_split < cloud_time


def build_transport_route_context(body: dict) -> dict:
    model_name = str(body.get("model", ""))
    model_profile = _runtime_model_profile(model_name)
    avail = mac_available_bytes()
    avail_safe = int(avail * LOCAL_MEM_SAFETY) if avail > 0 else 0
    external_low_memory, external_pressure, external_reason = _external_low_memory_signal()

    prompt_text = json.dumps(body.get("messages", []), ensure_ascii=False)
    prompt_len = max(len(prompt_text) // 4, 1)
    max_tokens = int(body.get("max_tokens", 32) or 32)
    arch_supported = mac_supports_arch(model_name)
    local_full_model_configured = bool(LOCAL_FULL_MODEL_PATH)
    layer_split_model_configured = bool(LAYER_SPLIT_MODEL_PATH)
    local_streaming_backend_configured = bool(CGC_LOCAL_FLASHMOE_MODEL or LOCAL_FULL_MODEL_PATH)
    moe_candidate = bool(model_profile.get("moe_candidate"))
    local_num_layers = max(int(model_profile.get("local_num_layers", 0) or 0), 0)
    weight = max(int(model_profile.get("model_weight_bytes", 0) or 0), 0)
    kv = estimate_kv_bytes(prompt_len, max_tokens, num_layers=local_num_layers if local_num_layers > 0 else None)
    needed_full = weight + kv
    full_resident_relaxed_admissible = bool(
        local_full_model_configured
        and not external_low_memory
        and avail > 0
        and needed_full > avail_safe
        and needed_full <= avail
    )
    full_resident_admissible = bool(
        local_full_model_configured
        and not external_low_memory
        and avail > 0
        and (needed_full <= avail_safe or full_resident_relaxed_admissible)
    )
    moe_streaming_required = (EDGE_EXPERT_RAM_BUDGET_BYTES + kv) if (moe_candidate and EDGE_EXPERT_STREAMING_ENABLED) else 0
    moe_streaming_headroom = avail_safe - moe_streaming_required if moe_streaming_required > 0 else avail_safe
    moe_streaming_admissible = bool(
        moe_candidate
        and EDGE_EXPERT_STREAMING_ENABLED
        and local_streaming_backend_configured
        and moe_streaming_headroom >= 0
        and not external_low_memory
    )
    memory_pressure = _memory_pressure_label(
        avail_bytes=avail,
        avail_safe_bytes=avail_safe,
        moe_required_bytes=moe_streaming_required,
        external_low_memory=external_low_memory,
        external_pressure=external_pressure,
    )
    per_layer = (weight / local_num_layers) if (local_num_layers > 0 and weight > 0) else 0.0
    act_per_layer = LAYER_ACTIVATION_BYTES if LAYER_ACTIVATION_BYTES > 0 else (max(prompt_len, 1) * 4096 * 2)
    partial_layer_capacity = int((avail_safe - act_per_layer) / per_layer) if per_layer > 0 else 0
    partial_layer_capacity = max(partial_layer_capacity, 0)
    dense_layer_streaming_admissible = bool(
        not moe_candidate
        and EDGE_DENSE_LAYER_STREAMING_ENABLED
        and local_full_model_configured
        and local_num_layers > 1
        and needed_full > avail_safe
        and partial_layer_capacity >= 1
        and not external_low_memory
    )
    mac_prefill_sec_est = estimate_mac_prefill_time(partial_layer_capacity, prompt_len) if partial_layer_capacity > 0 else 0.0
    cloud_prefill_sec_est = estimate_cloud_full_prefill_time(prompt_len)
    split_latency_sec_est = mac_prefill_sec_est + RTT_SEC if partial_layer_capacity > 0 else 0.0

    route_context = {
        "mode": "unknown",
        "mode_hint": "unknown",
        "desired_mode": "unknown",
        "reason": "",
        "arch_supported": bool(arch_supported),
        "local_model_configured": local_full_model_configured,
        "local_full_model_configured": local_full_model_configured,
        "layer_split_model_configured": layer_split_model_configured,
        "local_streaming_backend_configured": local_streaming_backend_configured,
        "prompt_len_est": prompt_len,
        "max_tokens": max_tokens,
        "mac_available_bytes": avail,
        "mac_available_safe_bytes": avail_safe,
        "model_weight_bytes": weight,
        "kv_bytes_est": kv,
        "needed_full_bytes": needed_full,
        "full_resident_admissible": full_resident_admissible,
        "full_resident_relaxed_admissible": full_resident_relaxed_admissible,
        "local_num_layers": local_num_layers,
        "per_layer_bytes": per_layer,
        "activation_bytes_per_layer": act_per_layer,
        "partial_layer_capacity": partial_layer_capacity,
        "latency_split_sec_est": split_latency_sec_est,
        "latency_cloud_sec_est": cloud_prefill_sec_est,
        "rtt_sec": RTT_SEC,
        "mac_prefill_sec_est": mac_prefill_sec_est,
        "memory_pressure": memory_pressure,
        "moe_candidate": moe_candidate,
        "moe_streaming_enabled": EDGE_EXPERT_STREAMING_ENABLED,
        "moe_streaming_admissible": moe_streaming_admissible,
        "moe_streaming_required_bytes": moe_streaming_required,
        "moe_streaming_headroom_bytes": moe_streaming_headroom,
        "dense_layer_streaming_enabled": EDGE_DENSE_LAYER_STREAMING_ENABLED,
        "dense_layer_streaming_admissible": dense_layer_streaming_admissible,
        "external_low_memory_detected": external_low_memory,
        "degrade_suggested": False,
        "degrade_target_mode": "unknown",
        "mode_switch_reason": "",
        "sticky_active": False,
        "sticky_until_epoch_ms": 0,
        "sticky_window_sec": EDGE_ROUTE_STICKY_WINDOW_SEC,
        "downgrade_chain": list(_ROUTE_DEGRADE_CHAIN),
    }

    desired_mode = "unknown"
    desired_reason = ""

    if not arch_supported:
        desired_mode = ROUTE_CLOUD_PD
        desired_reason = "arch_unsupported_v4flash"
    elif not local_full_model_configured and not local_streaming_backend_configured:
        desired_mode = ROUTE_CLOUD_PD
        desired_reason = "no_local_edge_backend"
    elif avail <= 0:
        desired_mode = ROUTE_CLOUD_FALLBACK
        desired_reason = "mem_unknown"
    else:
        forced_degrade = bool(external_low_memory or memory_pressure == "critical")
        if forced_degrade:
            route_context["degrade_suggested"] = True
            desired_reason = external_reason or "low_memory_forced_degrade"
        elif moe_candidate and EDGE_EXPERT_STREAMING_ENABLED and not moe_streaming_admissible:
            route_context["degrade_suggested"] = True
            desired_reason = "moe_streaming_not_admissible"

        if not route_context["degrade_suggested"] and (
            full_resident_admissible
            or (moe_candidate and moe_streaming_admissible)
            or dense_layer_streaming_admissible
        ):
            desired_mode = ROUTE_LOCAL_FULL
            if moe_candidate and moe_streaming_admissible:
                desired_reason = "moe_streaming_admissible"
            elif dense_layer_streaming_admissible:
                desired_reason = "dense_layer_streaming_admissible"
            elif full_resident_relaxed_admissible:
                desired_reason = "mem_full_relaxed"
            else:
                desired_reason = "mem_full"
        elif local_num_layers > 0:
            p = partial_layer_capacity
            if p >= 1 and layer_split_model_configured and latency_preflight_ok(p, prompt_len):
                desired_mode = ROUTE_LAYER_SPLIT_PD
                desired_reason = desired_reason or "memory_guard_to_layer_split_pd"
                route_context["P"] = p
                route_context["mac_time_est"] = estimate_mac_prefill_time(p, prompt_len)
            elif route_context["degrade_suggested"]:
                desired_mode = ROUTE_CLOUD_PD
                desired_reason = desired_reason or "memory_guard_to_cloud_pd"
            elif p < 1:
                desired_mode = ROUTE_CLOUD_FALLBACK
                desired_reason = "mem_insufficient_no_layer"
            elif not layer_split_model_configured:
                desired_mode = ROUTE_CLOUD_FALLBACK
                desired_reason = "no_layer_split_model"
            else:
                desired_mode = ROUTE_CLOUD_FALLBACK
                desired_reason = "latency_degraded"
        else:
            desired_mode = ROUTE_CLOUD_FALLBACK
            desired_reason = "mem_insufficient"

    route_context["desired_mode"] = desired_mode
    if route_context["degrade_suggested"]:
        route_context["degrade_target_mode"] = desired_mode
    final_mode, sticky_active, sticky_until_epoch_ms, effective_reason = _apply_sticky_window(desired_mode, desired_reason)
    route_context["mode"] = final_mode
    route_context["mode_hint"] = final_mode
    route_context["reason"] = desired_reason
    route_context["mode_switch_reason"] = effective_reason
    route_context["sticky_active"] = sticky_active
    route_context["sticky_until_epoch_ms"] = sticky_until_epoch_ms
    return route_context


def transport_runtime_snapshot() -> dict:
    profile = _runtime_model_profile("")
    return {
        "local_model_path": LOCAL_FULL_MODEL_PATH or "(none)",
        "local_full_model_path": LOCAL_FULL_MODEL_PATH or "(none)",
        "layer_split_model_path": LAYER_SPLIT_MODEL_PATH or "(none)",
        "local_num_layers": int(profile.get("local_num_layers", LOCAL_NUM_LAYERS or DEFAULT_LOCAL_NUM_LAYERS) or (LOCAL_NUM_LAYERS or DEFAULT_LOCAL_NUM_LAYERS)),
        "local_params": LOCAL_PARAMS,
        "mem_safety": LOCAL_MEM_SAFETY,
        "mac_tflops": MAC_TFLOPS,
        "cloud_tflops": CLOUD_TFLOPS,
        "rtt_sec": RTT_SEC,
        "mac_available_bytes": mac_available_bytes(),
        "routes": [
            ROUTE_LOCAL_FULL,
            ROUTE_LAYER_SPLIT_PD,
            ROUTE_CLOUD_PD,
            ROUTE_CLOUD_FALLBACK,
        ],
    }


def transport_debug_snapshot(body: dict, route_context: dict | None = None) -> dict:
    route_context = route_context or build_transport_route_context(body)
    return {
        "transport_route": route_context,
        "model": str(body.get("model", "")),
        "prompt_len_est": int(route_context.get("prompt_len_est") or 0),
        "max_tokens": int(route_context.get("max_tokens") or 0),
        "mac_available_bytes": int(route_context.get("mac_available_bytes") or 0),
        "mac_available_gb": round((int(route_context.get("mac_available_bytes") or 0) / 1024**3), 2)
        if route_context.get("mac_available_bytes")
        else 0,
        "mac_supports_arch": bool(route_context.get("arch_supported")),
        "latency_est": {
            "mac_prefill_est_sec": route_context.get("mac_time_est") or route_context.get("mac_prefill_sec_est"),
            "cloud_full_prefill_est_sec": route_context.get("latency_cloud_sec_est"),
            "rtt_sec": route_context.get("rtt_sec"),
        },
    }
