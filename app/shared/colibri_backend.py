from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional


_LANES = ("current", "next", "next_next", "far")


def _normalize_unit(unit: Any) -> dict[str, Any]:
    if isinstance(unit, dict):
        payload = dict(unit)
    else:
        payload = {"key": str(unit or "")}
    payload["key"] = str(payload.get("key") or "")
    payload["unit_kind"] = str(payload.get("unit_kind") or "expert")
    payload["model"] = str(payload.get("model") or "")
    payload["path"] = str(payload.get("path") or "")
    payload["io_backend"] = str(payload.get("io_backend") or "")
    payload["tags"] = list(payload.get("tags") or [])
    payload["resident"] = bool(payload.get("resident"))
    payload["pinned"] = bool(payload.get("pinned"))
    payload["prefetched"] = bool(payload.get("prefetched"))
    payload["available"] = bool(payload.get("available", True))
    payload["layer_id"] = int(payload.get("layer_id") or 0)
    payload["expert_id"] = int(payload.get("expert_id") or 0)
    payload["size_bytes"] = int(payload.get("size_bytes") or 0)
    payload["offset_bytes"] = int(payload.get("offset_bytes") or 0)
    payload["target_tier"] = str(payload.get("target_tier") or "")
    payload["routing_heat"] = float(payload.get("routing_heat") or 0.0)
    payload["pin_priority"] = float(payload.get("pin_priority") or 0.0)
    return payload


def normalize_runtime_unit_plan(runtime_unit_plan: Optional[dict[str, Any]]) -> dict[str, Any]:
    plan = dict(runtime_unit_plan or {})
    return {
        "control_plane": str(plan.get("control_plane") or "expert_data_plane"),
        "enabled": bool(plan.get("enabled")),
        "mode": str(plan.get("mode") or "bypass"),
        "reason": str(plan.get("reason") or ""),
        "model": str(plan.get("model") or ""),
        "family": str(plan.get("family") or ""),
        "route_mode": str(plan.get("route_mode") or ""),
        "frontier_key": str(plan.get("frontier_key") or ""),
        "current": [_normalize_unit(unit) for unit in list(plan.get("current") or [])],
        "next": [_normalize_unit(unit) for unit in list(plan.get("next") or [])],
        "next_next": [_normalize_unit(unit) for unit in list(plan.get("next_next") or [])],
        "far": [_normalize_unit(unit) for unit in list(plan.get("far") or [])],
        "summary": dict(plan.get("summary") or {}),
    }


def _default_runtime_backend(model_family: str, backend_family: str, runtime_backend: str = "") -> str:
    fam = str(model_family or "").strip().lower()
    backend = str(backend_family or "").strip().lower()
    runtime = str(runtime_backend or "").strip().lower()
    if runtime:
        return runtime
    if backend and backend != "auto":
        if backend == "mlx":
            return "turbofieldfare" if fam == "gemma4" else "omlx_mlx_lm"
        if backend == "gemma4_native":
            return "gemma4_native_metal"
        return backend
    if fam == "gemma4":
        return "turbofieldfare"
    return "colibri_generic"


def _default_adapter_name(model_family: str, runtime_backend: str, adapter_name: str = "") -> str:
    adapter = str(adapter_name or "").strip()
    if adapter:
        return adapter
    fam = str(model_family or "").strip().lower()
    runtime = str(runtime_backend or "").strip().lower()
    if fam == "gemma4" and runtime == "turbofieldfare":
        return "gemma4_a4b"
    return ""


def build_unified_runtime_ir_v0(
    *,
    request_id: str = "",
    runtime_unit_plan: Optional[dict[str, Any]] = None,
    model_id: str = "",
    model_family: str = "",
    model_format: str = "",
    architecture: str = "",
    quantization: str = "",
    runtime_mode: str = "",
    execution_intent: str = "",
    backend_family: str = "auto",
    runtime_backend: str = "",
    backend_hint: str = "",
    adapter_name: str = "",
    device_class: str = "",
    platform: str = "",
    strategy_family: str = "standard",
    speculative_mode: str = "none",
    max_tokens: int = 0,
    stream: bool = True,
    residency_policy_family: str = "",
    target_tier: str = "",
    pin_budget_bytes: int = 0,
    resident_budget_bytes: int = 0,
    prefetch_semantics: str = "",
    bootstrap_semantics: str = "",
    snapshot_level: str = "standard",
    emit_runtime_request: bool = True,
    emit_backend_snapshot: bool = True,
    required_capabilities: Optional[list[str]] = None,
    optional_capabilities: Optional[list[str]] = None,
) -> dict[str, Any]:
    plan = normalize_runtime_unit_plan(runtime_unit_plan)
    placement = {
        "runtime_unit_plan": plan,
        "current": list(plan.get("current") or []),
        "next": list(plan.get("next") or []),
        "future": list(plan.get("next_next") or []),
    }
    return normalize_unified_runtime_ir_v0({
        "ir_version": "unified_runtime_ir_v0",
        "request_id": request_id or str(plan.get("frontier_key") or ""),
        "model": {
            "model_id": model_id or str(plan.get("model") or ""),
            "model_family": model_family or str(plan.get("family") or ""),
            "model_format": model_format,
            "architecture": architecture,
            "quantization": quantization,
        },
        "runtime": {
            "mode": runtime_mode or str(plan.get("mode") or ""),
            "execution_intent": execution_intent or str(plan.get("route_mode") or plan.get("mode") or ""),
            "backend_family": backend_family or "auto",
            "runtime_backend": runtime_backend,
            "backend_hint": backend_hint,
            "adapter_name": adapter_name,
            "device_class": device_class,
            "platform": platform,
        },
        "decode_strategy": {
            "strategy_family": strategy_family or "standard",
            "speculative_mode": speculative_mode or "none",
            "max_tokens": int(max_tokens or 0),
            "stream": bool(stream),
        },
        "residency": {
            "policy_family": residency_policy_family or ("tiered_streaming" if bool(plan.get("enabled")) else "bypass"),
            "target_tier": target_tier or "ram",
            "pin_budget_bytes": int(pin_budget_bytes or 0),
            "resident_budget_bytes": int(resident_budget_bytes or 0),
            "prefetch_semantics": prefetch_semantics or ("best_effort" if bool(plan.get("enabled")) else "noop"),
            "bootstrap_semantics": bootstrap_semantics or ("decode_preprime" if bool(plan.get("enabled")) else "none"),
        },
        "placement": placement,
        "telemetry": {
            "snapshot_level": snapshot_level,
            "emit_runtime_request": bool(emit_runtime_request),
            "emit_backend_snapshot": bool(emit_backend_snapshot),
        },
        "adapter": {
            "required_capabilities": list(required_capabilities or []),
            "optional_capabilities": list(optional_capabilities or []),
        },
    })


def is_unified_runtime_ir_v0(payload: Optional[dict[str, Any]]) -> bool:
    return str(dict(payload or {}).get("ir_version") or "").strip() == "unified_runtime_ir_v0"


def normalize_unified_runtime_ir_v0(payload: Optional[dict[str, Any]]) -> dict[str, Any]:
    raw = dict(payload or {})
    model = dict(raw.get("model") or {})
    runtime = dict(raw.get("runtime") or {})
    decode = dict(raw.get("decode_strategy") or {})
    residency = dict(raw.get("residency") or {})
    placement = dict(raw.get("placement") or {})
    telemetry = dict(raw.get("telemetry") or {})
    adapter = dict(raw.get("adapter") or {})
    runtime_unit_plan = normalize_runtime_unit_plan(
        dict(placement.get("runtime_unit_plan") or {
            "control_plane": "expert_data_plane",
            "enabled": bool(placement),
            "mode": str(runtime.get("mode") or runtime.get("execution_intent") or "bypass"),
            "reason": str(runtime.get("execution_intent") or ""),
            "model": str(model.get("model_id") or ""),
            "family": str(model.get("model_family") or ""),
            "route_mode": str(runtime.get("execution_intent") or ""),
            "frontier_key": str(raw.get("request_id") or ""),
            "current": list(placement.get("current") or []),
            "next": list(placement.get("next") or []),
            "next_next": list(placement.get("future") or []),
            "far": [],
            "summary": {
                "placement_metadata_version": 1,
                "predicted_bytes_to_read_mb": 0.0,
                "predicted_cold_bytes_mb": 0.0,
            },
        })
    )
    return {
        "ir_version": "unified_runtime_ir_v0",
        "request_id": str(raw.get("request_id") or ""),
        "model": {
            "model_id": str(model.get("model_id") or ""),
            "model_family": str(model.get("model_family") or ""),
            "model_format": str(model.get("model_format") or ""),
            "architecture": str(model.get("architecture") or ""),
            "quantization": str(model.get("quantization") or ""),
        },
        "runtime": {
            "mode": str(runtime.get("mode") or ""),
            "execution_intent": str(runtime.get("execution_intent") or ""),
            "backend_family": str(runtime.get("backend_family") or "auto"),
            "runtime_backend": str(runtime.get("runtime_backend") or ""),
            "backend_hint": str(runtime.get("backend_hint") or ""),
            "adapter_name": str(runtime.get("adapter_name") or ""),
            "device_class": str(runtime.get("device_class") or ""),
            "platform": str(runtime.get("platform") or ""),
        },
        "decode_strategy": {
            "strategy_family": str(decode.get("strategy_family") or "standard"),
            "speculative_mode": str(decode.get("speculative_mode") or "none"),
            "max_tokens": int(decode.get("max_tokens") or 0),
            "stream": bool(decode.get("stream", True)),
        },
        "residency": {
            "policy_family": str(residency.get("policy_family") or ""),
            "target_tier": str(residency.get("target_tier") or ""),
            "pin_budget_bytes": int(residency.get("pin_budget_bytes") or 0),
            "resident_budget_bytes": int(residency.get("resident_budget_bytes") or 0),
            "prefetch_semantics": str(residency.get("prefetch_semantics") or ""),
            "bootstrap_semantics": str(residency.get("bootstrap_semantics") or ""),
        },
        "placement": {
            "runtime_unit_plan": runtime_unit_plan,
            "current": list(runtime_unit_plan.get("current") or []),
            "next": list(runtime_unit_plan.get("next") or []),
            "future": list(runtime_unit_plan.get("next_next") or []),
        },
        "telemetry": {
            "snapshot_level": str(telemetry.get("snapshot_level") or "standard"),
            "emit_runtime_request": bool(telemetry.get("emit_runtime_request", True)),
            "emit_backend_snapshot": bool(telemetry.get("emit_backend_snapshot", True)),
        },
        "adapter": {
            "required_capabilities": list(adapter.get("required_capabilities") or []),
            "optional_capabilities": list(adapter.get("optional_capabilities") or []),
        },
    }


def lower_unified_runtime_ir_v0(payload: Optional[dict[str, Any]]) -> dict[str, Any]:
    ir = normalize_unified_runtime_ir_v0(payload)
    model = dict(ir.get("model") or {})
    runtime = dict(ir.get("runtime") or {})
    decode = dict(ir.get("decode_strategy") or {})
    residency = dict(ir.get("residency") or {})
    plan = normalize_runtime_unit_plan(dict(ir.get("placement", {}).get("runtime_unit_plan") or {}))
    backend_family = str(runtime.get("backend_family") or "auto")
    runtime_backend = _default_runtime_backend(
        str(model.get("model_family") or ""),
        backend_family,
        str(runtime.get("runtime_backend") or ""),
    )
    adapter_name = _default_adapter_name(
        str(model.get("model_family") or ""),
        runtime_backend,
        str(runtime.get("adapter_name") or ""),
    )
    plan["mode"] = str(plan.get("mode") or runtime.get("mode") or runtime.get("execution_intent") or "bypass")
    plan["reason"] = str(plan.get("reason") or runtime.get("execution_intent") or "")
    plan["model"] = str(plan.get("model") or model.get("model_id") or "")
    plan["family"] = str(plan.get("family") or model.get("model_family") or "")
    plan["route_mode"] = str(plan.get("route_mode") or runtime.get("execution_intent") or "")
    plan["frontier_key"] = str(plan.get("frontier_key") or ir.get("request_id") or "")
    summary = dict(plan.get("summary") or {})
    summary.update({
        "placement_metadata_version": int(summary.get("placement_metadata_version") or 1),
        "unified_runtime_ir": "unified_runtime_ir_v0",
        "backend_family": backend_family,
        "runtime_backend": runtime_backend,
        "adapter_name": adapter_name,
        "decode_strategy_family": str(decode.get("strategy_family") or ""),
        "speculative_mode": str(decode.get("speculative_mode") or ""),
        "residency_policy_family": str(residency.get("policy_family") or ""),
        "residency_target_tier": str(residency.get("target_tier") or ""),
    })
    plan["summary"] = summary
    return {
        "ir": ir,
        "runtime_unit_plan": plan,
        "backend_lowering": {
            "backend_family": backend_family,
            "runtime_backend": runtime_backend,
            "adapter_name": adapter_name,
            "adapter_family": "colibri",
            "model_family": str(model.get("model_family") or ""),
            "execution_intent": str(runtime.get("execution_intent") or ""),
            "decode_strategy_family": str(decode.get("strategy_family") or ""),
            "speculative_mode": str(decode.get("speculative_mode") or ""),
            "residency_policy_family": str(residency.get("policy_family") or ""),
            "target_tier": str(residency.get("target_tier") or ""),
            "required_capabilities": list(dict(ir.get("adapter") or {}).get("required_capabilities") or []),
            "optional_capabilities": list(dict(ir.get("adapter") or {}).get("optional_capabilities") or []),
        },
    }


def _candidate_turbofieldfare_repos() -> list[str]:
    cwd = os.getcwd()
    candidates = [
        os.environ.get("CGC_TURBOFIELDFARE_REPO"),
        os.path.join(cwd, "turbo-fieldfare"),
        os.path.join(os.path.dirname(cwd), "turbo-fieldfare"),
        os.path.join(os.path.dirname(os.path.dirname(cwd)), "turbo-fieldfare"),
        "/Users/alexchuang/Documents/turbo-fieldfare",
    ]
    deduped: list[str] = []
    for candidate in candidates:
        path = str(candidate or "").strip()
        if path and path not in deduped and os.path.isdir(path):
            deduped.append(path)
    return deduped


def _default_turbofieldfare_staging_root() -> str:
    explicit = str(os.environ.get("CGC_TURBOFIELDFARE_STAGING_ROOT") or "").strip()
    if explicit:
        return explicit
    return os.path.join(os.getcwd(), "var", "external", "turbofieldfare")


def _first_existing_path(candidates: list[str], *, want_dir: bool = False) -> str:
    for candidate in candidates:
        path = str(candidate or "").strip()
        if not path:
            continue
        if want_dir:
            if os.path.isdir(path):
                return path
        else:
            if os.path.exists(path):
                return path
    return ""


def _is_complete_gturbo_dir(path: str) -> bool:
    probe = os.path.abspath(os.path.expanduser(str(path or "").strip()))
    if not probe or not os.path.isdir(probe):
        return False
    required = [
        os.path.join(probe, "manifest.json"),
        os.path.join(probe, "verified-install.json"),
        os.path.join(probe, "model_weights.bin"),
        os.path.join(probe, "packed_experts", "layout.json"),
        os.path.join(probe, "packed_experts", "layer_29.bin"),
    ]
    return all(os.path.exists(item) for item in required)


def _first_complete_gturbo_dir(candidates: list[str]) -> str:
    for candidate in candidates:
        path = str(candidate or "").strip()
        if not path:
            continue
        if _is_complete_gturbo_dir(path):
            return os.path.abspath(os.path.expanduser(path))
    return ""


def _turbofieldfare_blocked_reason(executable: str, model_dir: str) -> str:
    if executable and model_dir:
        return "ready_to_launch"
    if executable:
        return "missing_model_dir"
    if model_dir:
        return "missing_turbofieldfare_server_binary"
    return "missing_turbofieldfare_server_binary_and_model_dir"


def _collect_turbofieldfare_runtime_readiness(model_roots: Optional[dict[str, str]] = None) -> dict[str, Any]:
    repo_candidates = _candidate_turbofieldfare_repos()
    repo_root = str(repo_candidates[0] if repo_candidates else "")
    staging_root = _default_turbofieldfare_staging_root()
    server_candidates = [
        os.environ.get("CGC_TURBOFIELDFARE_SERVER_BIN"),
        os.environ.get("CGC_TURBOFIELDFARE_BIN"),
        os.environ.get("CGC_TURBOFIELDFARE_PATH"),
        os.path.join(staging_root, "bin", "TurboFieldfareServer"),
        shutil.which("TurboFieldfareServer"),
        shutil.which("turbofieldfare"),
    ]
    for repo_path in repo_candidates:
        server_candidates.append(os.path.join(repo_path, ".build", "release", "TurboFieldfareServer"))
    executable = _first_existing_path([str(candidate or "") for candidate in server_candidates])

    model_candidates = [
        os.environ.get("CGC_TURBOFIELDFARE_MODEL"),
        os.environ.get("TURBOFIELDFARE_MODEL"),
        os.environ.get("CGC_TURBOFIELDFARE_MODEL_DIR"),
        os.environ.get("CGC_TURBOFIELDFARE_RESTORED_MODEL"),
        os.path.join(staging_root, "models", "gemma4.gturbo"),
    ]
    roots = dict(model_roots or {})
    for key in ("gturbo", "turbofieldfare", "gemma4_gturbo", "gemma4"):
        if roots.get(key):
            model_candidates.append(str(roots.get(key)))
    model_candidates.append("/tmp/gemma4-restored-local.gturbo")
    for repo_path in repo_candidates:
        model_candidates.append(os.path.join(repo_path, "scratch", "gemma4.gturbo"))
    model_candidates.append(os.path.join(os.getcwd(), "scratch", "gemma4.gturbo"))
    model_dir = _first_complete_gturbo_dir([str(candidate or "") for candidate in model_candidates])

    missing_dependencies: list[str] = []
    if not executable:
        missing_dependencies.append("TurboFieldfareServer")
    if not model_dir:
        missing_dependencies.append("gemma4.gturbo")

    return {
        "repo_candidates": [str(path or "") for path in repo_candidates],
        "repo_root": repo_root,
        "staging_root": staging_root,
        "server_candidates": [str(path or "") for path in server_candidates if str(path or "").strip()],
        "model_candidates": [str(path or "") for path in model_candidates if str(path or "").strip()],
        "executable": executable,
        "model_dir": model_dir,
        "missing_dependencies": missing_dependencies,
        "blocked_reason": _turbofieldfare_blocked_reason(executable, model_dir),
        "launch_ready": bool(executable and model_dir),
    }


def _detect_turbofieldfare_binary() -> str:
    return str(_collect_turbofieldfare_runtime_readiness().get("executable") or "")


def _default_backend_request_path(request_id: str, runtime_backend: str) -> str:
    safe_request_id = str(request_id or "request").strip() or "request"
    safe_backend = str(runtime_backend or "backend").strip() or "backend"
    return os.path.join(
        os.getcwd(),
        "var",
        "colibri",
        f"{safe_backend}_{safe_request_id}_request.json",
    )


def _default_backend_receipt_path(session_id: str, runtime_backend: str) -> str:
    safe_session_id = str(session_id or "session").strip() or "session"
    safe_backend = str(runtime_backend or "backend").strip() or "backend"
    return os.path.join(
        os.getcwd(),
        "var",
        "colibri",
        "sessions",
        safe_session_id,
        f"{safe_backend}_receipt.json",
    )


def _default_backend_log_path(session_id: str, runtime_backend: str) -> str:
    safe_session_id = str(session_id or "session").strip() or "session"
    safe_backend = str(runtime_backend or "backend").strip() or "backend"
    return os.path.join(
        os.getcwd(),
        "var",
        "colibri",
        "sessions",
        safe_session_id,
        f"{safe_backend}.log",
    )


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def _detect_turbofieldfare_model_dir(model_roots: Optional[dict[str, str]] = None) -> str:
    return str(_collect_turbofieldfare_runtime_readiness(model_roots=model_roots).get("model_dir") or "")


def _http_get_json(url: str, timeout_s: float) -> tuple[int, str, dict[str, Any]]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=max(timeout_s, 0.1)) as resp:
        status_code = int(getattr(resp, "status", 200) or 200)
        raw_text = resp.read().decode("utf-8", errors="replace")
        return status_code, raw_text, _parse_json_text(raw_text)


def _build_turbofieldfare_request_contract(
    *,
    ir: dict[str, Any],
    runtime_unit_plan: dict[str, Any],
    model_roots: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    plan = normalize_runtime_unit_plan(runtime_unit_plan)
    model = dict(ir.get("model") or {})
    runtime = dict(ir.get("runtime") or {})
    decode = dict(ir.get("decode_strategy") or {})
    residency = dict(ir.get("residency") or {})
    placement = dict(ir.get("placement") or {})
    lanes: dict[str, list[dict[str, Any]]] = {}
    unresolved_source_count = 0
    for lane_name, plan_key in (("current", "current"), ("next", "next"), ("future", "next_next")):
        lane_units: list[dict[str, Any]] = []
        for unit in plan.get(plan_key, []):
            locator = _build_source_locator(unit, model_roots=model_roots)
            if str(locator.get("mode") or "") == "unresolved":
                unresolved_source_count += 1
            lane_units.append({
                "key": str(unit.get("key") or ""),
                "unit_kind": str(unit.get("unit_kind") or ""),
                "layer_id": int(unit.get("layer_id") or 0),
                "expert_id": int(unit.get("expert_id") or 0),
                "target_tier": str(unit.get("target_tier") or ""),
                "routing_heat": float(unit.get("routing_heat") or 0.0),
                "pin_priority": float(unit.get("pin_priority") or 0.0),
                "source_locator": locator,
            })
        lanes[lane_name] = lane_units
    return {
        "contract_version": "turbofieldfare.request.v0",
        "request_id": str(ir.get("request_id") or plan.get("frontier_key") or ""),
        "backend_family": str(runtime.get("backend_family") or ""),
        "runtime_backend": str(runtime.get("runtime_backend") or "turbofieldfare"),
        "adapter_name": str(runtime.get("adapter_name") or ""),
        "model": {
            "model_id": str(model.get("model_id") or plan.get("model") or ""),
            "model_family": str(model.get("model_family") or plan.get("family") or ""),
            "model_format": str(model.get("model_format") or ""),
            "architecture": str(model.get("architecture") or ""),
            "quantization": str(model.get("quantization") or ""),
        },
        "runtime": {
            "mode": str(runtime.get("mode") or plan.get("mode") or ""),
            "execution_intent": str(runtime.get("execution_intent") or plan.get("route_mode") or ""),
            "platform": str(runtime.get("platform") or ""),
            "device_class": str(runtime.get("device_class") or ""),
        },
        "decode_strategy": {
            "strategy_family": str(decode.get("strategy_family") or ""),
            "speculative_mode": str(decode.get("speculative_mode") or ""),
            "max_tokens": int(decode.get("max_tokens") or 0),
            "stream": bool(decode.get("stream", True)),
        },
        "residency": {
            "policy_family": str(residency.get("policy_family") or ""),
            "target_tier": str(residency.get("target_tier") or ""),
            "prefetch_semantics": str(residency.get("prefetch_semantics") or ""),
            "bootstrap_semantics": str(residency.get("bootstrap_semantics") or ""),
        },
        "placement": {
            "summary": dict(plan.get("summary") or {}),
            "lane_counts": {
                "current": len(lanes["current"]),
                "next": len(lanes["next"]),
                "future": len(lanes["future"]),
            },
            "lanes": lanes,
        },
        "source_resolution": {
            "unresolved_source_count": int(unresolved_source_count),
        },
        "telemetry": {
            "snapshot_level": str(dict(ir.get("telemetry") or {}).get("snapshot_level") or "standard"),
            "emit_runtime_request": bool(dict(ir.get("telemetry") or {}).get("emit_runtime_request", True)),
            "emit_backend_snapshot": bool(dict(ir.get("telemetry") or {}).get("emit_backend_snapshot", True)),
        },
    }


def _build_turbofieldfare_launch_contract(
    *,
    ir: dict[str, Any],
    request_contract: dict[str, Any],
    model_roots: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    runtime = dict(ir.get("runtime") or {})
    model = dict(ir.get("model") or {})
    decode = dict(ir.get("decode_strategy") or {})
    readiness = _collect_turbofieldfare_runtime_readiness(model_roots=model_roots)
    executable = str(readiness.get("executable") or "")
    request_id = str(ir.get("request_id") or request_contract.get("request_id") or "request")
    session_id = str(request_id or f"tf-{uuid.uuid4().hex[:12]}")
    request_path = _default_backend_request_path(request_id, "turbofieldfare")
    receipt_path = _default_backend_receipt_path(session_id, "turbofieldfare")
    log_path = _default_backend_log_path(session_id, "turbofieldfare")
    cwd = str(os.environ.get("CGC_TURBOFIELDFARE_CWD") or os.getcwd())
    model_dir = str(readiness.get("model_dir") or "")
    port = int(os.environ.get("CGC_TURBOFIELDFARE_PORT") or 0) or _find_free_local_port()
    max_context = int(os.environ.get("CGC_TURBOFIELDFARE_MAX_CONTEXT") or 16384)
    queue_limit = int(os.environ.get("CGC_TURBOFIELDFARE_QUEUE_LIMIT") or 4)
    prompt_cache_mode_raw = str(
        os.environ.get("CGC_TURBOFIELDFARE_PROMPT_CACHE_MODE") or "single_prefix"
    ).strip()
    prompt_cache_mode = prompt_cache_mode_raw.replace("_", "-")
    tokenizer_dir = ""
    if model_dir:
        sidecar_tokenizer_dir = os.path.join(model_dir, "tokenizer")
        if os.path.isfile(os.path.join(sidecar_tokenizer_dir, "tokenizer.json")):
            tokenizer_dir = sidecar_tokenizer_dir
        elif os.path.isfile(os.path.join(model_dir, "tokenizer.json")):
            tokenizer_dir = model_dir
    argv = [
        executable or "TurboFieldfareServer",
        "--model",
        model_dir or "",
        "--port",
        str(port),
        "--max-context",
        str(max_context),
        "--queue-limit",
        str(queue_limit),
        "--prompt-cache-mode",
        prompt_cache_mode,
    ]
    base_url = f"http://127.0.0.1:{port}/v1"
    health_url = f"http://127.0.0.1:{port}/health"
    model_id = str(model.get("model_id") or "gemma-4-26b-a4b-it")
    return {
        "contract_version": "turbofieldfare.launch.v0",
        "backend_family": str(runtime.get("backend_family") or ""),
        "runtime_backend": "turbofieldfare",
        "adapter_name": str(runtime.get("adapter_name") or ""),
        "model_id": model_id,
        "session_id": session_id,
        "executable": executable,
        "cwd": cwd,
        "argv": argv,
        "model_dir": model_dir,
        "repo_root": str(readiness.get("repo_root") or ""),
        "repo_candidates": list(readiness.get("repo_candidates") or []),
        "server_candidates": list(readiness.get("server_candidates") or []),
        "model_candidates": list(readiness.get("model_candidates") or []),
        "staging_root": str(readiness.get("staging_root") or ""),
        "base_url": base_url,
        "health_url": health_url,
        "log_path": log_path,
        "receipt_path": receipt_path,
        "max_context": max_context,
        "queue_limit": queue_limit,
        "prompt_cache_mode": prompt_cache_mode,
        "max_completion_tokens": int(decode.get("max_tokens") or 0),
        "env": {
            "CGC_RUNTIME_BACKEND_FAMILY": str(runtime.get("backend_family") or ""),
            "CGC_RUNTIME_BACKEND": "turbofieldfare",
            "CGC_RUNTIME_ADAPTER": str(runtime.get("adapter_name") or ""),
            "CGC_RUNTIME_REQUEST_ID": request_id,
            "CGC_TURBOFIELDFARE_REQUEST_CONTRACT": request_path,
            "CGC_TURBOFIELDFARE_RECEIPT_PATH": receipt_path,
            "CGC_TURBOFIELDFARE_LOG_PATH": log_path,
            "TURBO_FIELDFARE_TOKENIZER_DIR": tokenizer_dir,
        },
        "request_contract_path": request_path,
        "missing_dependencies": list(readiness.get("missing_dependencies") or []),
        "launch_ready": bool(readiness.get("launch_ready")),
        "reason": str(readiness.get("blocked_reason") or ""),
    }


def build_backend_execution_contracts(
    *,
    ir: Optional[dict[str, Any]],
    runtime_unit_plan: Optional[dict[str, Any]],
    backend_lowering: Optional[dict[str, Any]],
    model_roots: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    normalized_ir = normalize_unified_runtime_ir_v0(ir)
    plan = normalize_runtime_unit_plan(runtime_unit_plan)
    lowering = dict(backend_lowering or {})
    runtime_backend = str(lowering.get("runtime_backend") or "")
    if runtime_backend == "turbofieldfare":
        request_contract = _build_turbofieldfare_request_contract(
            ir=normalized_ir,
            runtime_unit_plan=plan,
            model_roots=model_roots,
        )
        launch_contract = _build_turbofieldfare_launch_contract(
            ir=normalized_ir,
            request_contract=request_contract,
            model_roots=model_roots,
        )
        return {
            "request_contract": request_contract,
            "launch_contract": launch_contract,
        }
    return {
        "request_contract": {},
        "launch_contract": {},
    }


def _infer_storage_tier(unit: dict[str, Any]) -> str:
    explicit_target = str(unit.get("target_tier") or "").strip()
    if explicit_target:
        return explicit_target
    if not unit.get("available"):
        return "unavailable"
    if unit.get("pinned"):
        return "pinned_ram"
    if unit.get("resident"):
        return "resident_ram"
    if unit.get("path"):
        return "nvme"
    return "unknown"


def _prefetch_role_for_lane(lane: str) -> str:
    if lane == "current":
        return "required_now"
    if lane == "next":
        return "layer_ahead"
    if lane == "next_next":
        return "double_buffer"
    if lane == "far":
        return "far_lookahead"
    return "unknown"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_session_id() -> str:
    return f"colibri-{uuid.uuid4().hex[:12]}"


def _truncate_text(text: str, limit: int = 2000) -> str:
    raw = str(text or "")
    if len(raw) <= limit:
        return raw
    return raw[:limit] + "...(truncated)"


def _parse_json_text(text: str) -> Optional[dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _to_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _normalize_colibri_response(
    *,
    response_payload: Optional[dict[str, Any]],
    http_status_code: Optional[int],
    engine_bridge_ready: bool,
    delivery: str,
    session_id: str,
) -> dict[str, Any]:
    payload = dict(response_payload or {})
    accepted_raw = payload.get("accepted")
    if accepted_raw is None:
        accepted_raw = payload.get("ok")
    if accepted_raw is None:
        accepted_raw = delivery in {"file_written", "http_posted"} and engine_bridge_ready
    session_state = str(
        payload.get("session_state")
        or payload.get("state")
        or ("submitted" if bool(accepted_raw) else "blocked")
    )
    worker_id = str(payload.get("worker_id") or payload.get("worker") or "").strip()
    queue_depth = _to_optional_int(
        payload.get("queue_depth", payload.get("queue_size", payload.get("queue")))
    )
    message = str(payload.get("message") or payload.get("detail") or payload.get("status") or "").strip()
    return {
        "accepted": bool(accepted_raw),
        "session_state": session_state,
        "worker_id": worker_id,
        "queue_depth": queue_depth,
        "message": message,
        "session_id": str(payload.get("session_id") or session_id or ""),
        "http_status_code": http_status_code,
        "raw": payload,
    }


def _build_colibri_session_status_url(target: str, session_id: str) -> str:
    raw_target = str(target or "").strip()
    raw_session_id = str(session_id or "").strip()
    if not raw_target or not raw_session_id:
        return ""
    parsed = urllib.parse.urlparse(raw_target)
    path = str(parsed.path or "").rstrip("/")
    if path.endswith("/colibri/session"):
        next_path = f"{path}/{raw_session_id}"
    elif "/colibri/session/" in path:
        next_path = path
    else:
        next_path = f"{path}/colibri/session/{raw_session_id}" if path else f"/colibri/session/{raw_session_id}"
    return urllib.parse.urlunparse(parsed._replace(path=next_path))


def get_colibri_http_contract_spec() -> dict[str, Any]:
    return {
        "protocol": "colibri_engine_bridge/v1",
        "endpoints": {
            "submit_session": {
                "method": "POST",
                "path": "/colibri/session",
                "content_type": "application/json",
            },
            "get_session": {
                "method": "GET",
                "path": "/colibri/session/{session_id}",
                "content_type": "application/json",
            },
            "transition_session": {
                "method": "POST",
                "path": "/colibri/session/{session_id}/transition",
                "content_type": "application/json",
            },
            "submit_receipt": {
                "method": "POST",
                "path": "/colibri/session/{session_id}/receipt",
                "content_type": "application/json",
            },
            "health": {
                "method": "GET",
                "path": "/colibri/health",
                "content_type": "application/json",
            },
            "spec": {
                "method": "GET",
                "path": "/colibri/spec",
                "content_type": "application/json",
            },
        },
        "lifecycle_states": [
            "prepared",
            "queued",
            "staging",
            "ready",
            "running",
            "closed",
            "failed",
            "blocked",
        ],
        "executor_receipt": {
            "required_fields": [
                "session_id",
                "worker_id",
                "status",
                "completed_at_ms",
            ],
            "status_values": [
                "ready",
                "failed",
                "blocked",
            ],
            "optional_fields": [
                "message",
                "artifacts",
                "metrics",
                "load_request_path",
                "cache_tier",
                "bytes_loaded",
                "resident_handles",
                "load_ms",
                "unit_results",
            ],
        },
        "request": {
            "required_top_level_fields": [
                "protocol",
                "action",
                "session",
                "transport",
                "request",
                "client",
                "response_contract",
            ],
            "action": "begin_or_update_session",
            "session_required_fields": [
                "session_id",
                "engine",
                "state",
                "request_seq",
                "frontier_key",
                "model",
                "mode",
                "created_at_ms",
                "submitted_at_ms",
            ],
            "response_contract_required_fields": [
                "accepted",
                "session_state",
                "worker_id",
                "queue_depth",
            ],
        },
        "response": {
            "required_fields": [
                "accepted",
                "session_state",
                "worker_id",
                "queue_depth",
            ],
            "optional_fields": [
                "session_id",
                "message",
            ],
        },
    }


def _gemma4_a4b_tensor_keys_for_expert(layer_id: int) -> list[str]:
    base = f"language_model.model.layers.{int(layer_id)}.experts.switch_glu"
    keys: list[str] = []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for suffix in ("weight", "scales", "biases"):
            keys.append(f"{base}.{proj}.{suffix}")
    return keys


def _gemma4_a4b_tensor_keys_for_router(layer_id: int) -> list[str]:
    base = f"language_model.model.layers.{int(layer_id)}.router"
    return [
        f"{base}.proj.weight",
        f"{base}.proj.scales",
        f"{base}.proj.biases",
        f"{base}.per_expert_scale",
        f"{base}.scale",
    ]


def _load_safetensors_weight_map(index_path: str) -> dict[str, str]:
    with open(index_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        str(key): str(value)
        for key, value in dict(payload.get("weight_map") or {}).items()
    }


def _resolve_model_root_for_unit(unit: dict[str, Any], model_roots: Optional[dict[str, str]] = None) -> str:
    roots = dict(model_roots or {})
    model_name = str(unit.get("model") or "").strip().lower()
    configured = str(roots.get(model_name) or "").strip()
    if configured:
        return configured
    return ""


def _build_source_locator(
    unit: dict[str, Any],
    *,
    model_roots: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    path = str(unit.get("path") or "").strip()
    io_backend = str(unit.get("io_backend") or "").strip()
    size_bytes = int(unit.get("size_bytes") or 0)
    offset_bytes = int(unit.get("offset_bytes") or 0)
    unit_kind = str(unit.get("unit_kind") or "")
    model_name = str(unit.get("model") or "").strip().lower()
    layer_id = int(unit.get("layer_id") or 0)
    expert_id = int(unit.get("expert_id") or 0)

    if path and io_backend == "file_range":
        return {
            "mode": "file_range",
            "path": path,
            "offset_bytes": offset_bytes,
            "size_bytes": size_bytes,
        }
    if path:
        return {
            "mode": "file_full",
            "path": path,
            "size_bytes": size_bytes,
        }

    model_root = _resolve_model_root_for_unit(unit, model_roots=model_roots)
    index_path = os.path.join(model_root, "model.safetensors.index.json") if model_root else ""
    if model_name == "gemma4" and index_path and os.path.exists(index_path):
        weight_map = _load_safetensors_weight_map(index_path)
        if unit_kind == "expert" and expert_id >= 0:
            tensor_keys = _gemma4_a4b_tensor_keys_for_expert(layer_id)
            shard_map = {
                key: os.path.join(model_root, weight_map[key])
                for key in tensor_keys
                if key in weight_map
            }
            return {
                "mode": "safetensors_tensor_slice",
                "index_path": index_path,
                "tensor_keys": tensor_keys,
                "tensor_shards": shard_map,
                "slice_axis": 0,
                "slice_index": expert_id,
                "router_tensor_keys": _gemma4_a4b_tensor_keys_for_router(layer_id),
            }
        if unit_kind == "layer":
            tensor_prefix = f"language_model.model.layers.{layer_id}."
            shard_map = {
                key: os.path.join(model_root, shard)
                for key, shard in weight_map.items()
                if key.startswith(tensor_prefix)
            }
            return {
                "mode": "safetensors_layer_prefix",
                "index_path": index_path,
                "tensor_prefix": tensor_prefix,
                "tensor_shards": shard_map,
            }
    return {"mode": "unresolved"}


def map_runtime_unit_plan_to_colibri(runtime_unit_plan: Optional[dict[str, Any]]) -> dict[str, Any]:
    plan = normalize_runtime_unit_plan(runtime_unit_plan)
    tier_counts: dict[str, int] = {}
    mapped_lanes: dict[str, list[dict[str, Any]]] = {}
    mapping_gaps: list[str] = []
    prefetch_semantics_ok = True

    for lane in _LANES:
        mapped_units: list[dict[str, Any]] = []
        for unit in plan.get(lane, []):
            tier = _infer_storage_tier(unit)
            tier_counts[tier] = int(tier_counts.get(tier, 0)) + 1
            mapped_units.append({
                "key": unit["key"],
                "unit_kind": unit["unit_kind"],
                "model": unit["model"],
                "layer_id": unit["layer_id"],
                "expert_id": unit["expert_id"],
                "tier": tier,
                "prefetch_role": _prefetch_role_for_lane(lane),
                "path": unit["path"],
                "size_bytes": unit["size_bytes"],
                "offset_bytes": unit["offset_bytes"],
                "io_backend": unit["io_backend"] or "file_full",
                "resident": unit["resident"],
                "pinned": unit["pinned"],
                "prefetched": unit["prefetched"],
                "available": unit["available"],
                "target_tier": unit["target_tier"] or tier,
                "routing_heat": float(unit["routing_heat"]),
                "pin_priority": float(unit["pin_priority"]),
                "tags": list(unit["tags"]),
            })
            if not unit["key"]:
                mapping_gaps.append(f"{lane}:missing_key")
            if lane != "current" and not unit["path"] and not unit["resident"]:
                prefetch_semantics_ok = False
                mapping_gaps.append(f"{lane}:{unit['key'] or 'unknown'}:missing_path_for_prefetch")
            if not str(unit.get("target_tier") or "").strip():
                mapping_gaps.append(f"{lane}:{unit['key'] or 'unknown'}:missing_target_tier")
            if float(unit.get("routing_heat") or 0.0) < 0.0:
                mapping_gaps.append(f"{lane}:{unit['key'] or 'unknown'}:invalid_routing_heat")
            if float(unit.get("pin_priority") or 0.0) < 0.0:
                mapping_gaps.append(f"{lane}:{unit['key'] or 'unknown'}:invalid_pin_priority")
        mapped_lanes[lane] = mapped_units

    if plan.get("enabled"):
        if "predicted_cold_bytes_mb" not in plan.get("summary", {}):
            mapping_gaps.append("summary:missing_predicted_cold_bytes_mb")
        if int(plan.get("summary", {}).get("placement_metadata_version") or 0) < 1:
            mapping_gaps.append("summary:missing_placement_metadata_version")

    unique_gaps = sorted(set(mapping_gaps))
    tier_semantics_lossless = not any(
        "target_tier" in gap or "routing_heat" in gap or "pin_priority" in gap
        for gap in unique_gaps
    )

    return {
        "control_plane": plan["control_plane"],
        "mode": plan["mode"],
        "enabled": plan["enabled"],
        "reason": plan["reason"],
        "frontier_key": plan["frontier_key"],
        "prefetch_semantics_lossless": bool(prefetch_semantics_ok),
        "tier_semantics_lossless": bool(tier_semantics_lossless),
        "mapping_lossless": bool(prefetch_semantics_ok and tier_semantics_lossless),
        "mapping_gaps": unique_gaps,
        "summary": {
            "lane_counts": {lane: len(mapped_lanes[lane]) for lane in _LANES},
            "tier_counts": tier_counts,
            "predicted_bytes_to_read_mb": float(plan.get("summary", {}).get("predicted_bytes_to_read_mb") or 0.0),
            "predicted_cold_bytes_mb": float(plan.get("summary", {}).get("predicted_cold_bytes_mb") or 0.0),
        },
        "lanes": mapped_lanes,
    }


def build_colibri_engine_request(
    runtime_unit_plan: Optional[dict[str, Any]],
    *,
    model_roots: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    mapped = map_runtime_unit_plan_to_colibri(runtime_unit_plan)
    plan = normalize_runtime_unit_plan(runtime_unit_plan)
    lanes: dict[str, list[dict[str, Any]]] = {}
    unresolved_sources = 0
    for lane in _LANES:
        lane_units: list[dict[str, Any]] = []
        for unit in plan.get(lane, []):
            locator = _build_source_locator(unit, model_roots=model_roots)
            if str(locator.get("mode") or "") == "unresolved":
                unresolved_sources += 1
            lane_units.append({
                "key": unit["key"],
                "unit_kind": unit["unit_kind"],
                "model": unit["model"],
                "layer_id": unit["layer_id"],
                "expert_id": unit["expert_id"],
                "target_tier": unit["target_tier"] or _infer_storage_tier(unit),
                "routing_heat": float(unit["routing_heat"]),
                "pin_priority": float(unit["pin_priority"]),
                "prefetch_role": _prefetch_role_for_lane(lane),
                "source_locator": locator,
                "tags": list(unit["tags"]),
            })
        lanes[lane] = lane_units
    engine_bridge_ready = bool(plan["enabled"]) and int(unresolved_sources) == 0
    return {
        "engine": "colibri",
        "request_contract_version": 1,
        "control_plane": plan["control_plane"],
        "mode": plan["mode"],
        "enabled": plan["enabled"],
        "reason": plan["reason"],
        "model": plan["model"],
        "family": plan["family"],
        "route_mode": plan["route_mode"],
        "frontier_key": plan["frontier_key"],
        "summary": {
            **dict(mapped.get("summary") or {}),
            "unresolved_source_count": int(unresolved_sources),
            "engine_bridge_ready": bool(engine_bridge_ready),
        },
        "mapping": {
            "mapping_lossless": bool(mapped.get("mapping_lossless")),
            "prefetch_semantics_lossless": bool(mapped.get("prefetch_semantics_lossless")),
            "tier_semantics_lossless": bool(mapped.get("tier_semantics_lossless")),
            "mapping_gaps": list(mapped.get("mapping_gaps") or []),
        },
        "lanes": lanes,
    }


class ColibriAdapterBackend:
    """Proof-of-concept Colibri adapter.

    This adapter validates whether our runtime_unit_plan can be translated into a
    Colibri-style tier/prefetch session contract. It does not execute model decode.
    """

    def __init__(
        self,
        *,
        model_roots: Optional[dict[str, str]] = None,
    ) -> None:
        self._request_sequence = 0
        self._model_roots = dict(model_roots or {})
        self._request_runtime_unit_plan: dict[str, Any] = {}
        self._last_unified_runtime_ir: dict[str, Any] = {}
        self._last_backend_lowering: dict[str, Any] = {}
        self._last_backend_request_contract: dict[str, Any] = {}
        self._last_backend_launch_contract: dict[str, Any] = {}
        self._last_mapping: dict[str, Any] = {}
        self._last_engine_request: dict[str, Any] = {}
        self._engine_session: dict[str, Any] = {}
        self._last_begin_request_ms = 0.0
        self._last_generate_payload: dict[str, Any] = {}
        self._last_submit_payload: dict[str, Any] = {}
        self._last_submit_result: dict[str, Any] = {}
        self._last_engine_response: dict[str, Any] = {}
        self._last_status_result: dict[str, Any] = {}

    def _ensure_engine_session(self) -> dict[str, Any]:
        plan = dict(self._request_runtime_unit_plan or {})
        frontier_key = str(plan.get("frontier_key") or "")
        model = str(plan.get("model") or "")
        mode = str(plan.get("mode") or "")
        if (
            not self._engine_session
            or str(self._engine_session.get("frontier_key") or "") != frontier_key
            or str(self._engine_session.get("model") or "") != model
            or str(self._engine_session.get("mode") or "") != mode
            or bool(self._engine_session.get("closed"))
        ):
            self._engine_session = {
                "session_id": _new_session_id(),
                "engine": "colibri",
                "state": "prepared",
                "closed": False,
                "created_at_ms": _now_ms(),
                "updated_at_ms": _now_ms(),
                "submit_count": 0,
                "request_seq": int(self._request_sequence),
                "frontier_key": frontier_key,
                "model": model,
                "mode": mode,
                "reason": str(plan.get("reason") or ""),
                "last_submit_at_ms": None,
                "last_submit_target": "",
                "last_payload_path": "",
            }
        else:
            self._engine_session["request_seq"] = int(self._request_sequence)
            self._engine_session["reason"] = str(plan.get("reason") or "")
            self._engine_session["updated_at_ms"] = _now_ms()
            if str(self._engine_session.get("state") or "") == "closed":
                self._engine_session["state"] = "prepared"
                self._engine_session["closed"] = False
        return dict(self._engine_session)

    def lower(
        self,
        payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        source = payload if payload is not None else self._last_unified_runtime_ir
        if is_unified_runtime_ir_v0(source):
            lowered = lower_unified_runtime_ir_v0(source)
            ir = dict(lowered.get("ir") or {})
            runtime_unit_plan = normalize_runtime_unit_plan(lowered.get("runtime_unit_plan"))
            backend_lowering = dict(lowered.get("backend_lowering") or {})
        else:
            ir = {}
            runtime_unit_plan = normalize_runtime_unit_plan(source)
            backend_lowering = {}
        contracts = build_backend_execution_contracts(
            ir=ir,
            runtime_unit_plan=runtime_unit_plan,
            backend_lowering=backend_lowering,
            model_roots=self._model_roots,
        )
        return {
            "ir": ir,
            "runtime_unit_plan": runtime_unit_plan,
            "backend_lowering": backend_lowering,
            "backend_request_contract": dict(contracts.get("request_contract") or {}),
            "backend_launch_contract": dict(contracts.get("launch_contract") or {}),
        }

    def begin_request(
        self,
        runtime_unit_plan: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        started = time.time()
        self._request_sequence += 1
        if is_unified_runtime_ir_v0(runtime_unit_plan):
            lowered = self.lower(runtime_unit_plan)
            self._last_unified_runtime_ir = dict(lowered.get("ir") or {})
            self._last_backend_lowering = dict(lowered.get("backend_lowering") or {})
            self._last_backend_request_contract = dict(lowered.get("backend_request_contract") or {})
            self._last_backend_launch_contract = dict(lowered.get("backend_launch_contract") or {})
            self._request_runtime_unit_plan = normalize_runtime_unit_plan(lowered.get("runtime_unit_plan"))
        else:
            self._last_unified_runtime_ir = {}
            self._last_backend_lowering = {}
            self._last_backend_request_contract = {}
            self._last_backend_launch_contract = {}
            self._request_runtime_unit_plan = normalize_runtime_unit_plan(runtime_unit_plan)
        self._last_mapping = map_runtime_unit_plan_to_colibri(self._request_runtime_unit_plan)
        self._last_engine_request = build_colibri_engine_request(
            self._request_runtime_unit_plan,
            model_roots=self._model_roots,
        )
        self._ensure_engine_session()
        self._last_begin_request_ms = (time.time() - started) * 1000
        return self.runtime_request_snapshot()

    def runtime_request_snapshot(self) -> dict[str, Any]:
        plan = dict(self._request_runtime_unit_plan or {})
        return {
            "request_seq": int(self._request_sequence),
            "control_plane": str(plan.get("control_plane") or "expert_data_plane"),
            "mode": str(plan.get("mode") or "bypass"),
            "enabled": bool(plan.get("enabled")),
            "reason": str(plan.get("reason") or ""),
            "frontier_key": str(plan.get("frontier_key") or ""),
            "mapping_lossless": bool(self._last_mapping.get("mapping_lossless")),
            "prefetch_semantics_lossless": bool(self._last_mapping.get("prefetch_semantics_lossless")),
            "tier_semantics_lossless": bool(self._last_mapping.get("tier_semantics_lossless")),
            "mapping_gaps": list(self._last_mapping.get("mapping_gaps") or []),
            "summary": dict(self._last_mapping.get("summary") or {}),
            "engine_request_unresolved_source_count": int(
                dict(self._last_engine_request.get("summary") or {}).get("unresolved_source_count") or 0
            ),
            "engine_bridge_ready": bool(
                dict(self._last_engine_request.get("summary") or {}).get("engine_bridge_ready")
            ),
            "unified_runtime_ir": dict(self._last_unified_runtime_ir or {}),
            "backend_lowering": dict(self._last_backend_lowering or {}),
            "backend_request_contract": dict(self._last_backend_request_contract or {}),
            "backend_launch_contract": dict(self._last_backend_launch_contract or {}),
            "engine_session": dict(self._engine_session or {}),
            "begin_request_ms": float(self._last_begin_request_ms or 0.0),
        }

    def submit_to_engine(
        self,
        *,
        transport: str = "file",
        target: str = "external_process",
        payload_path: str = "",
        metadata: Optional[dict[str, Any]] = None,
        timeout_ms: int = 5000,
    ) -> dict[str, Any]:
        session = self._ensure_engine_session()
        engine_summary = dict(self._last_engine_request.get("summary") or {})
        engine_bridge_ready = bool(engine_summary.get("engine_bridge_ready"))
        unresolved_source_count = int(engine_summary.get("unresolved_source_count") or 0)
        submit_started_ms = _now_ms()
        normalized_transport = str(transport or "file").strip().lower()
        payload = {
            "protocol": "colibri_engine_bridge/v1",
            "action": "begin_or_update_session",
            "session": {
                "session_id": str(session.get("session_id") or ""),
                "engine": "colibri",
                "state": "submitted" if engine_bridge_ready else "blocked",
                "request_seq": int(self._request_sequence),
                "frontier_key": str(session.get("frontier_key") or ""),
                "model": str(session.get("model") or ""),
                "mode": str(session.get("mode") or ""),
                "created_at_ms": int(session.get("created_at_ms") or submit_started_ms),
                "submitted_at_ms": int(submit_started_ms),
            },
            "transport": {
                "kind": normalized_transport,
                "target": str(target or "external_process"),
                "delivery": "stub",
                "metadata": dict(metadata or {}),
            },
            "request": dict(self._last_engine_request or {}),
            "backend_request_contract": dict(self._last_backend_request_contract or {}),
            "backend_launch_contract": dict(self._last_backend_launch_contract or {}),
            "client": {
                "adapter": "app.shared.colibri_backend.ColibriAdapterBackend",
                "pid": int(os.getpid()),
                "cwd": os.getcwd(),
            },
            "response_contract": {
                "required_fields": [
                    "accepted",
                    "session_state",
                    "worker_id",
                    "queue_depth",
                ],
                "normalization_version": 1,
            },
        }
        normalized_payload_path = str(payload_path or "").strip()
        if normalized_transport == "local_process" and not normalized_payload_path:
            normalized_payload_path = str(
                dict(self._last_backend_launch_contract or {}).get("request_contract_path")
                or _default_backend_request_path(
                    str(self._engine_session.get("session_id") or ""),
                    str(dict(self._last_backend_lowering or {}).get("runtime_backend") or "backend"),
                )
            )
        should_write_file = bool(normalized_payload_path) or normalized_transport in {"file", "local_process"}
        if should_write_file and not normalized_payload_path:
            normalized_payload_path = os.path.join(
                os.getcwd(),
                "var",
                "colibri",
                "last_engine_request.json",
            )
        if should_write_file and normalized_payload_path:
            payload_dir = os.path.dirname(normalized_payload_path)
            if payload_dir:
                os.makedirs(payload_dir, exist_ok=True)
            with open(normalized_payload_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        if normalized_transport == "local_process" and normalized_payload_path:
            backend_request_contract = dict(self._last_backend_request_contract or {})
            if backend_request_contract:
                with open(normalized_payload_path, "w", encoding="utf-8") as handle:
                    json.dump(backend_request_contract, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
        self._last_submit_payload = payload
        delivery = "not_sent"
        response_status_code: Optional[int] = None
        response_body = ""
        response_error = ""
        response_payload: Optional[dict[str, Any]] = None
        if normalized_transport == "http":
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                str(target or "").strip(),
                data=body,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Accept": "application/json",
                    "X-Colibri-Session-Id": str(session.get("session_id") or ""),
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=max(float(timeout_ms or 0) / 1000.0, 0.1)) as resp:
                    response_status_code = int(getattr(resp, "status", 200) or 200)
                    raw_body = resp.read()
                    raw_text = raw_body.decode("utf-8", errors="replace")
                    response_body = _truncate_text(raw_text)
                    response_payload = _parse_json_text(raw_text)
                    delivery = "http_posted"
            except urllib.error.HTTPError as exc:
                response_status_code = int(getattr(exc, "code", 500) or 500)
                raw_text = exc.read().decode("utf-8", errors="replace")
                response_body = _truncate_text(raw_text)
                response_payload = _parse_json_text(raw_text)
                response_error = str(exc)
                delivery = "http_error"
            except Exception as exc:
                response_error = str(exc)
                delivery = "http_error"
        elif normalized_transport == "file":
            delivery = "file_written" if normalized_payload_path else "not_sent"
        elif normalized_transport == "local_process":
            launch_contract = dict(self._last_backend_launch_contract or {})
            launch_ready = bool(launch_contract.get("launch_ready"))
            executable = str(launch_contract.get("executable") or "").strip()
            argv = list(launch_contract.get("argv") or [])
            cwd = str(launch_contract.get("cwd") or os.getcwd())
            env_vars = dict(launch_contract.get("env") or {})
            health_url = str(launch_contract.get("health_url") or "").strip()
            log_path = str(launch_contract.get("log_path") or "").strip()
            receipt_path = str(launch_contract.get("receipt_path") or "").strip()
            base_url = str(launch_contract.get("base_url") or "").strip()
            receipt_payload: dict[str, Any] = {
                "schema_version": 1,
                "runtime_backend": str(launch_contract.get("runtime_backend") or ""),
                "adapter_name": str(launch_contract.get("adapter_name") or ""),
                "session_id": str(launch_contract.get("session_id") or session.get("session_id") or ""),
                "request_contract_path": normalized_payload_path,
                "receipt_path": receipt_path,
                "log_path": log_path,
                "base_url": base_url,
                "health_url": health_url,
                "launched_at_ms": int(submit_started_ms),
                "status": "blocked",
                "message": "",
            }
            if not launch_ready:
                delivery = "local_launch_blocked"
                receipt_payload["message"] = str(launch_contract.get("reason") or "launch_not_ready")
                response_payload = {
                    "accepted": False,
                    "session_state": "blocked",
                    "worker_id": "",
                    "queue_depth": 0,
                    "message": receipt_payload["message"],
                    "launch_contract": launch_contract,
                    "receipt": receipt_payload,
                }
            else:
                os.makedirs(os.path.dirname(log_path), exist_ok=True) if log_path else None
                process_env = os.environ.copy()
                process_env.update({k: str(v) for k, v in env_vars.items()})
                with open(log_path, "ab") as log_handle:
                    proc = subprocess.Popen(
                        argv,
                        cwd=cwd,
                        env=process_env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                receipt_payload["pid"] = int(proc.pid)
                ready = False
                wait_deadline = time.time() + max(float(timeout_ms or 0) / 1000.0, 0.1)
                last_error = ""
                while time.time() < wait_deadline:
                    if proc.poll() is not None:
                        last_error = f"process_exited:{int(proc.returncode or 0)}"
                        break
                    try:
                        status_code, raw_text, _ = _http_get_json(health_url, timeout_s=0.5)
                        if status_code == 200:
                            ready = True
                            receipt_payload["ready_at_ms"] = _now_ms()
                            receipt_payload["health_status_code"] = status_code
                            receipt_payload["health_body"] = _truncate_text(raw_text, 512)
                            break
                    except Exception as exc:
                        last_error = str(exc)
                    time.sleep(0.2)
                if ready:
                    delivery = "local_process_ready"
                    receipt_payload["status"] = "ready"
                    receipt_payload["message"] = "server_ready"
                    response_payload = {
                        "accepted": bool(engine_bridge_ready),
                        "session_state": "ready" if engine_bridge_ready else "blocked",
                        "worker_id": f"pid:{int(proc.pid)}",
                        "queue_depth": 0,
                        "message": "server_ready",
                        "launch_contract": launch_contract,
                        "receipt": receipt_payload,
                    }
                else:
                    delivery = "local_process_failed"
                    try:
                        if proc.poll() is None:
                            proc.terminate()
                    except Exception:
                        pass
                    receipt_payload["status"] = "failed"
                    receipt_payload["message"] = last_error or "health_timeout"
                    response_payload = {
                        "accepted": False,
                        "session_state": "failed",
                        "worker_id": f"pid:{int(proc.pid)}",
                        "queue_depth": 0,
                        "message": receipt_payload["message"],
                        "launch_contract": launch_contract,
                        "receipt": receipt_payload,
                    }
            if receipt_path:
                os.makedirs(os.path.dirname(receipt_path), exist_ok=True)
                with open(receipt_path, "w", encoding="utf-8") as handle:
                    json.dump(receipt_payload, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
            response_payload = {
                **dict(response_payload or {}),
            }
        else:
            raise ValueError(f"Unsupported Colibri transport: {normalized_transport}")
        normalized_response = _normalize_colibri_response(
            response_payload=response_payload,
            http_status_code=response_status_code,
            engine_bridge_ready=engine_bridge_ready,
            delivery=delivery,
            session_id=str(self._engine_session.get("session_id") or ""),
        )
        self._last_engine_response = dict(normalized_response)
        self._engine_session.update({
            "state": str(normalized_response.get("session_state") or "blocked"),
            "closed": False,
            "updated_at_ms": int(submit_started_ms),
            "submit_count": int(self._engine_session.get("submit_count") or 0) + 1,
            "last_submit_at_ms": int(submit_started_ms),
            "last_submit_target": str(target or "external_process"),
            "last_submit_transport": normalized_transport,
            "last_payload_path": normalized_payload_path,
            "last_worker_id": str(normalized_response.get("worker_id") or ""),
            "last_queue_depth": normalized_response.get("queue_depth"),
        })
        self._last_submit_result = {
            "status": "submitted" if bool(normalized_response.get("accepted")) else "submit_error",
            "engine": "colibri",
            "engine_bridge_ready": bool(engine_bridge_ready),
            "unresolved_source_count": int(unresolved_source_count),
            "session_id": str(self._engine_session.get("session_id") or ""),
            "transport": normalized_transport,
            "target": str(target or "external_process"),
            "payload_path": normalized_payload_path,
            "submitted_at_ms": int(submit_started_ms),
            "delivery": delivery,
            "accepted": bool(normalized_response.get("accepted")),
            "session_state": str(normalized_response.get("session_state") or ""),
            "worker_id": str(normalized_response.get("worker_id") or ""),
            "queue_depth": normalized_response.get("queue_depth"),
            "message": str(normalized_response.get("message") or ""),
            "http_status_code": normalized_response.get("http_status_code"),
            "response_body": response_body,
            "error": response_error,
            "response_contract": dict(normalized_response),
        }
        return {
            "session": dict(self._engine_session),
            "submission": dict(self._last_submit_result),
            "payload": dict(self._last_submit_payload),
        }

    def get_session_status(
        self,
        *,
        target: str = "",
        timeout_ms: int = 3000,
    ) -> dict[str, Any]:
        session_id = str(self._engine_session.get("session_id") or "").strip()
        status_url = _build_colibri_session_status_url(
            target or str(self._engine_session.get("last_submit_target") or ""),
            session_id,
        )
        if not status_url:
            self._last_status_result = {
                "status": "status_error",
                "error": "missing_status_url",
                "session_id": session_id,
                "status_url": "",
                "response_contract": _normalize_colibri_response(
                    response_payload=None,
                    http_status_code=None,
                    engine_bridge_ready=bool(
                        dict(self._last_engine_request.get("summary") or {}).get("engine_bridge_ready")
                    ),
                    delivery="not_sent",
                    session_id=session_id,
                ),
            }
            return dict(self._last_status_result)
        http_status_code: Optional[int] = None
        response_body = ""
        response_error = ""
        response_payload: Optional[dict[str, Any]] = None
        try:
            req = urllib.request.Request(
                status_url,
                headers={"Accept": "application/json"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=max(float(timeout_ms or 0) / 1000.0, 0.1)) as resp:
                http_status_code = int(getattr(resp, "status", 200) or 200)
                raw_body = resp.read()
                raw_text = raw_body.decode("utf-8", errors="replace")
                response_body = _truncate_text(raw_text)
                response_payload = _parse_json_text(raw_text)
        except urllib.error.HTTPError as exc:
            http_status_code = int(getattr(exc, "code", 500) or 500)
            raw_text = exc.read().decode("utf-8", errors="replace")
            response_body = _truncate_text(raw_text)
            response_payload = _parse_json_text(raw_text)
            response_error = str(exc)
        except Exception as exc:
            response_error = str(exc)
        normalized = _normalize_colibri_response(
            response_payload=response_payload,
            http_status_code=http_status_code,
            engine_bridge_ready=bool(dict(self._last_engine_request.get("summary") or {}).get("engine_bridge_ready")),
            delivery="http_posted" if http_status_code else "http_error",
            session_id=session_id,
        )
        if normalized.get("worker_id"):
            self._engine_session["last_worker_id"] = str(normalized.get("worker_id") or "")
        if normalized.get("queue_depth") is not None:
            self._engine_session["last_queue_depth"] = normalized.get("queue_depth")
        if normalized.get("session_state"):
            self._engine_session["state"] = str(normalized.get("session_state") or "")
        self._engine_session["updated_at_ms"] = _now_ms()
        self._last_engine_response = dict(normalized)
        self._last_status_result = {
            "status": "ok" if not response_error and http_status_code else "status_error",
            "session_id": session_id,
            "status_url": status_url,
            "http_status_code": http_status_code,
            "response_body": response_body,
            "error": response_error,
            "response_contract": dict(normalized),
        }
        return dict(self._last_status_result)

    def wait_until_session_state(
        self,
        desired_states: list[str],
        *,
        target: str = "",
        timeout_ms: int = 10000,
        poll_interval_ms: int = 500,
    ) -> dict[str, Any]:
        deadline = time.time() + max(float(timeout_ms or 0) / 1000.0, 0.1)
        desired = {str(item or "").strip() for item in list(desired_states or []) if str(item or "").strip()}
        last: dict[str, Any] = {}
        while time.time() < deadline:
            last = self.get_session_status(target=target, timeout_ms=min(poll_interval_ms, 3000))
            state = str(dict(last.get("response_contract") or {}).get("session_state") or "")
            if state in desired:
                return {
                    "status": "matched",
                    "matched_state": state,
                    "result": last,
                }
            time.sleep(max(float(poll_interval_ms or 0) / 1000.0, 0.05))
        return {
            "status": "timeout",
            "desired_states": sorted(desired),
            "result": last,
        }

    def close_session(
        self,
        *,
        reason: str = "",
    ) -> dict[str, Any]:
        if not self._engine_session:
            return {
                "status": "no_session",
                "reason": "engine_session_not_initialized",
            }
        closed_at_ms = _now_ms()
        self._engine_session.update({
            "state": "closed",
            "closed": True,
            "updated_at_ms": int(closed_at_ms),
            "close_reason": str(reason or ""),
        })
        return {
            "status": "closed",
            "session": dict(self._engine_session),
        }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        self._last_generate_payload = {
            "status": "not_implemented",
            "reason": "colibri_engine_bridge_not_connected",
            "prompt_chars": len(str(prompt or "")),
            "requested_max_tokens": int(max_tokens or 0),
            "engine_request_ready": bool(self._last_engine_request),
            "engine_bridge_ready": bool(
                dict(self._last_engine_request.get("summary") or {}).get("engine_bridge_ready")
            ),
            "engine_session_id": str(self._engine_session.get("session_id") or ""),
            "unresolved_source_count": int(
                dict(self._last_engine_request.get("summary") or {}).get("unresolved_source_count") or 0
            ),
        }
        return {
            "text": "",
            "mode": "colibri_engine_adapter",
            "status": "not_implemented",
            "runtime_request": self.runtime_request_snapshot(),
            "engine_request": dict(self._last_engine_request),
            "engine_session": dict(self._engine_session or {}),
            "engine_response": dict(self._last_engine_response or {}),
            "adapter_generate": dict(self._last_generate_payload),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": "colibri_adapter_poc",
            "request_seq": int(self._request_sequence),
            "runtime_request": self.runtime_request_snapshot(),
            "mapping": dict(self._last_mapping or {}),
            "unified_runtime_ir": dict(self._last_unified_runtime_ir or {}),
            "backend_lowering": dict(self._last_backend_lowering or {}),
            "backend_request_contract": dict(self._last_backend_request_contract or {}),
            "backend_launch_contract": dict(self._last_backend_launch_contract or {}),
            "engine_request": dict(self._last_engine_request or {}),
            "engine_session": dict(self._engine_session or {}),
            "engine_response": dict(self._last_engine_response or {}),
            "last_status": dict(self._last_status_result or {}),
            "last_submit": dict(self._last_submit_result or {}),
            "last_submit_payload": dict(self._last_submit_payload or {}),
            "last_generate": dict(self._last_generate_payload or {}),
        }
