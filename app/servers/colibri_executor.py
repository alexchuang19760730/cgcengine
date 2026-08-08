#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from typing import Any


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_RUNTIME_TIER_NAMES = {
    0: "nvme",
    1: "ram",
    2: "vram",
}


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: str, payload: dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def _runtime_headers(api_key: str) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    normalized = str(api_key or "").strip()
    if normalized:
        headers["Authorization"] = f"Bearer {normalized}"
    return headers


def _normalize_cache_tier_name(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"disk", "ssd"}:
        return "nvme"
    if raw in {"pinned_ram", "resident_ram"}:
        return "ram"
    return raw


def _normalize_runtime_base_url(base_url: str) -> str:
    raw = str(base_url or "").strip()
    if not raw:
        return ""
    parsed = urllib.parse.urlparse(raw)
    path = str(parsed.path or "").rstrip("/")
    for suffix in (
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/messages",
        "/v1/models",
        "/v1",
    ):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    normalized = parsed._replace(path=path or "", params="", query="", fragment="")
    return urllib.parse.urlunparse(normalized).rstrip("/")


def _build_runtime_url(base_url: str, path: str) -> str:
    normalized_base = _normalize_runtime_base_url(base_url)
    normalized_path = str(path or "").strip()
    if not normalized_base:
        raise ValueError("Missing Colibri runtime base URL")
    if normalized_path.startswith("http://") or normalized_path.startswith("https://"):
        return normalized_path
    return urllib.parse.urljoin(normalized_base + "/", normalized_path.lstrip("/"))


def _fetch_json(url: str, *, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=max(timeout_s, 0.1)) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


def _fetch_optional_json(url: str, *, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    try:
        return _fetch_json(url, headers=headers, timeout_s=timeout_s)
    except Exception:
        return {}


def _fetch_runtime_snapshot(
    *,
    runtime_base_url: str,
    runtime_api_key: str,
    timeout_ms: int,
) -> dict[str, Any]:
    headers = _runtime_headers(runtime_api_key)
    timeout_s = max(float(timeout_ms or 0) / 1000.0, 0.1)
    health = _fetch_json(
        _build_runtime_url(runtime_base_url, "/health"),
        headers=headers,
        timeout_s=timeout_s,
    )
    experts = _fetch_json(
        _build_runtime_url(runtime_base_url, "/experts"),
        headers=headers,
        timeout_s=timeout_s,
    )
    profile = _fetch_optional_json(
        _build_runtime_url(runtime_base_url, "/profile"),
        headers=headers,
        timeout_s=timeout_s,
    )
    models = _fetch_optional_json(
        _build_runtime_url(runtime_base_url, "/v1/models"),
        headers=headers,
        timeout_s=timeout_s,
    )
    return {
        "runtime_base_url": _normalize_runtime_base_url(runtime_base_url),
        "health": health,
        "experts": experts,
        "profile": profile,
        "models": models,
    }


def _normalized_runtime_tiers(snapshot: dict[str, Any]) -> dict[str, Any]:
    health = dict(snapshot.get("health") or {})
    raw_tiers = dict(health.get("tiers") or {})
    normalized: dict[str, Any] = {}
    for key, value in raw_tiers.items():
        normalized_key = _normalize_cache_tier_name(key)
        if not normalized_key:
            continue
        normalized[normalized_key] = value
    return normalized


def _estimate_locator_bytes(locator: dict[str, Any]) -> int:
    mode = str(locator.get("mode") or "").strip()
    if mode == "file_range":
        requested = int(locator.get("size_bytes") or 0)
        if requested > 0:
            return requested
        path = str(locator.get("path") or "").strip()
        offset = max(int(locator.get("offset_bytes") or 0), 0)
        if path and os.path.exists(path):
            return max(int(os.path.getsize(path)) - offset, 0)
        return 0
    if mode == "file_full":
        path = str(locator.get("path") or "").strip()
        return int(os.path.getsize(path)) if path and os.path.exists(path) else 0
    if mode in {"safetensors_tensor_slice", "safetensors_layer_prefix"}:
        shard_map = dict(locator.get("tensor_shards") or {})
        unique_paths = {
            str(item or "").strip()
            for item in shard_map.values()
            if str(item or "").strip()
        }
        return sum(
            int(os.path.getsize(path))
            for path in sorted(unique_paths)
            if os.path.exists(path)
        )
    return 0


def _expert_runtime_cell(
    snapshot: dict[str, Any],
    *,
    layer_id: int,
    expert_id: int,
) -> dict[str, Any]:
    experts = dict(snapshot.get("experts") or {})
    rows = int(experts.get("rows") or 0)
    cols = int(experts.get("cols") or 0)
    emap = str(experts.get("map") or "")
    if rows <= 0 or cols <= 0:
        raise ValueError("runtime_expert_map_unavailable")
    if layer_id < 0 or layer_id >= rows or expert_id < 0 or expert_id >= cols:
        raise ValueError(f"runtime_expert_out_of_range:{layer_id}:{expert_id}:{rows}x{cols}")
    offset = (layer_id * cols + expert_id) * 2
    if offset + 2 > len(emap):
        raise ValueError(f"runtime_expert_map_truncated:{layer_id}:{expert_id}")
    raw_byte = int(emap[offset:offset + 2], 16)
    raw_tier = int(raw_byte >> 6)
    heat = int(raw_byte & 63)
    return {
        "raw_tier": raw_tier,
        "cache_tier": _RUNTIME_TIER_NAMES.get(raw_tier, "unknown"),
        "heat": heat,
        "emap_seq": int(experts.get("seq") or 0),
        "rows": rows,
        "cols": cols,
    }


def _build_runtime_handle(
    snapshot: dict[str, Any],
    *,
    layer_id: int,
    expert_id: int,
    cache_tier: str,
    emap_seq: int,
) -> str:
    runtime_base = str(snapshot.get("runtime_base_url") or "")
    return (
        f"colibri-runtime://{runtime_base}"
        f"/experts/{emap_seq}/layer/{int(layer_id)}/expert/{int(expert_id)}"
        f"?tier={urllib.parse.quote(str(cache_tier or 'unknown'))}"
    )


def _build_non_expert_runtime_handle(
    snapshot: dict[str, Any],
    *,
    unit_key: str,
    unit_kind: str,
    cache_tier: str,
    evidence_scope: str,
) -> str:
    runtime_base = str(snapshot.get("runtime_base_url") or "")
    encoded_key = urllib.parse.quote(str(unit_key or "unknown"), safe="")
    encoded_kind = urllib.parse.quote(str(unit_kind or "unknown"), safe="")
    encoded_scope = urllib.parse.quote(str(evidence_scope or "tier_snapshot"), safe="")
    return (
        f"colibri-runtime://{runtime_base}"
        f"/tiers/{urllib.parse.quote(str(cache_tier or 'unknown'), safe='')}/unit/{encoded_key}"
        f"?kind={encoded_kind}&scope={encoded_scope}"
    )


def _extract_runtime_residency_entries(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for source_name in ("health", "experts", "profile"):
        payload = dict(snapshot.get(source_name) or {})
        if payload:
            sources.append({"source": source_name, "payload": payload})
    profile_turns = list(dict(snapshot.get("profile") or {}).get("turns") or [])
    if profile_turns:
        sources.append({"source": "profile_turn", "payload": dict(profile_turns[-1] or {})})

    entries: list[dict[str, Any]] = []
    for item in sources:
        source = str(item.get("source") or "")
        payload = dict(item.get("payload") or {})
        for field in (
            "resident_units",
            "unit_residency",
            "residency",
            "resident_handles",
        ):
            raw = payload.get(field)
            if isinstance(raw, list):
                for entry in raw:
                    if isinstance(entry, dict):
                        entries.append({"source": source, **dict(entry)})
            elif isinstance(raw, dict):
                for key, value in raw.items():
                    if isinstance(value, dict):
                        entries.append({"source": source, "key": str(key), **dict(value)})
                    else:
                        entries.append({"source": source, "key": str(key), "value": value})
    return entries


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _normalize_runtime_residency_entry(entry: dict[str, Any]) -> dict[str, Any]:
    key = str(entry.get("key") or entry.get("unit_key") or entry.get("name") or "").strip()
    unit_kind = str(entry.get("unit_kind") or entry.get("kind") or "").strip()
    cache_tier = _normalize_cache_tier_name(
        entry.get("cache_tier") or entry.get("tier") or entry.get("target_tier") or ""
    )
    resident_handle = str(
        entry.get("resident_handle")
        or entry.get("handle")
        or entry.get("resident")
        or ""
    ).strip()
    return {
        "source": str(entry.get("source") or "").strip(),
        "key": key,
        "unit_kind": unit_kind,
        "layer_id": _coerce_int(entry.get("layer_id")),
        "expert_id": _coerce_int(entry.get("expert_id")),
        "cache_tier": cache_tier,
        "resident_handle": resident_handle,
        "bytes_loaded": _coerce_int(entry.get("bytes_loaded")),
        "load_ms": float(entry.get("load_ms") or 0.0),
        "runtime_confirmed": bool(entry.get("runtime_confirmed", True)),
        "raw": dict(entry),
    }


def _find_runtime_non_expert_residency(unit: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    key = str(unit.get("key") or "").strip()
    unit_kind = str(unit.get("unit_kind") or "").strip()
    layer_id = _coerce_int(unit.get("layer_id"))
    requested_tier = _normalize_cache_tier_name(unit.get("target_tier"))
    candidates: list[dict[str, Any]] = []
    for raw_entry in _extract_runtime_residency_entries(snapshot):
        entry = _normalize_runtime_residency_entry(raw_entry)
        if not entry.get("cache_tier"):
            continue
        entry_key = str(entry.get("key") or "").strip()
        entry_kind = str(entry.get("unit_kind") or "").strip()
        entry_layer = _coerce_int(entry.get("layer_id"))
        if key and entry_key and entry_key == key:
            candidates.append(entry)
            continue
        if unit_kind and entry_kind and unit_kind == entry_kind:
            if layer_id is None or entry_layer is None or layer_id == entry_layer:
                candidates.append(entry)
    if requested_tier:
        for entry in candidates:
            if _normalize_cache_tier_name(entry.get("cache_tier")) == requested_tier:
                return entry
    return candidates[0] if candidates else {}


def _best_available_runtime_tier(runtime_tiers: dict[str, Any]) -> str:
    for tier in ("vram", "ram", "nvme"):
        try:
            if float(runtime_tiers.get(tier, 0) or 0) > 0:
                return tier
        except Exception:
            continue
    return "unknown"


def _load_non_expert_unit(
    unit: dict[str, Any],
    *,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    key = str(unit.get("key") or "")
    unit_kind = str(unit.get("unit_kind") or "")
    requested_tier = _normalize_cache_tier_name(unit.get("target_tier"))
    locator = dict(unit.get("source_locator") or {})
    estimated_bytes = int(_estimate_locator_bytes(locator))
    runtime_tiers = _normalized_runtime_tiers(snapshot)
    experts = dict(snapshot.get("experts") or {})
    residency = _find_runtime_non_expert_residency(unit, snapshot)
    evidence_scope = "per_unit" if residency else "tier_snapshot"
    observed_tier = _normalize_cache_tier_name(
        residency.get("cache_tier") or requested_tier or _best_available_runtime_tier(runtime_tiers)
    )
    tier_count = runtime_tiers.get(observed_tier, 0)
    tier_gb = runtime_tiers.get(f"{observed_tier}_gb", 0.0)
    resident_handle = str(residency.get("resident_handle") or "").strip()
    if not resident_handle and observed_tier and observed_tier != "unknown":
        resident_handle = _build_non_expert_runtime_handle(
            snapshot,
            unit_key=key,
            unit_kind=unit_kind,
            cache_tier=observed_tier,
            evidence_scope=evidence_scope,
        )
    runtime_confirmed = bool(residency) or (
        bool(resident_handle)
        and (
            observed_tier == "nvme"
            or float(tier_count or 0) > 0
            or float(tier_gb or 0.0) > 0.0
        )
    )
    explicit_bytes_loaded = _coerce_int(residency.get("bytes_loaded"))
    bytes_loaded = (
        explicit_bytes_loaded
        if explicit_bytes_loaded is not None
        else estimated_bytes if runtime_confirmed and observed_tier in {"ram", "vram"} else 0
    )
    explicit_load_ms = float(residency.get("load_ms") or 0.0)
    return {
        "key": key,
        "unit_kind": unit_kind,
        "prefetch_role": str(unit.get("prefetch_role") or ""),
        "cache_tier": observed_tier,
        "bytes_loaded": int(bytes_loaded),
        "resident_handle": resident_handle,
        "artifacts": [
            {
                "kind": "colibri_runtime_non_expert_residency" if residency else "colibri_runtime_tier_snapshot",
                "runtime_base_url": str(snapshot.get("runtime_base_url") or ""),
                "cache_tier": observed_tier,
                "requested_tier": requested_tier,
                "tier_count": tier_count,
                "tier_gb": tier_gb,
                "runtime_emap_seq": int(experts.get("seq") or 0),
                "evidence_scope": evidence_scope,
                "residency_source": str(residency.get("source") or ""),
                "residency_key": str(residency.get("key") or ""),
            }
        ],
        "estimated_bytes": estimated_bytes,
        "runtime_heat": 0,
        "runtime_emap_seq": int(experts.get("seq") or 0),
        "metrics_source": "colibri_runtime_residency" if residency else "colibri_runtime_health",
        "evidence_scope": evidence_scope,
        "runtime_confirmed": runtime_confirmed,
        "load_ms": explicit_load_ms or round((time.time() - started) * 1000.0, 3),
    }


def _resolve_runtime_config(load_request: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    runtime = dict(load_request.get("runtime") or {})
    base_url = str(args.runtime_base_url or runtime.get("base_url") or "").strip()
    api_key = str(args.runtime_api_key or "").strip()
    api_key_env = str(runtime.get("api_key_env") or "").strip()
    if not api_key and api_key_env:
        api_key = str(os.environ.get(api_key_env) or "").strip()
    return {
        "base_url": _normalize_runtime_base_url(base_url),
        "api_key": api_key,
        "api_key_env": api_key_env,
        "timeout_ms": int(args.runtime_timeout_ms or 0),
    }


def _load_unit(
    unit: dict[str, Any],
    *,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    key = str(unit.get("key") or "")
    unit_kind = str(unit.get("unit_kind") or "")
    layer_id = int(unit.get("layer_id") or 0)
    expert_id = int(unit.get("expert_id") or 0)
    locator = dict(unit.get("source_locator") or {})

    if unit_kind == "expert":
        estimated_bytes = int(_estimate_locator_bytes(locator))
        artifacts: list[dict[str, Any]] = []
        resident_handle = ""
        cell = _expert_runtime_cell(
            snapshot,
            layer_id=layer_id,
            expert_id=expert_id,
        )
        cache_tier = str(cell.get("cache_tier") or "unknown")
        runtime_heat = int(cell.get("heat") or 0)
        emap_seq = int(cell.get("emap_seq") or 0)
        if cache_tier in {"ram", "vram"}:
            resident_handle = _build_runtime_handle(
                snapshot,
                layer_id=layer_id,
                expert_id=expert_id,
                cache_tier=cache_tier,
                emap_seq=emap_seq,
            )
        artifacts.append({
            "kind": "colibri_runtime_expert_cell",
            "runtime_base_url": str(snapshot.get("runtime_base_url") or ""),
            "layer_id": layer_id,
            "expert_id": expert_id,
            "cache_tier": cache_tier,
            "heat": runtime_heat,
            "emap_seq": emap_seq,
        })
        bytes_loaded = estimated_bytes if cache_tier in {"ram", "vram"} else 0
        return {
            "key": key,
            "unit_kind": unit_kind,
            "prefetch_role": str(unit.get("prefetch_role") or ""),
            "cache_tier": cache_tier,
            "bytes_loaded": int(bytes_loaded),
            "resident_handle": resident_handle,
            "artifacts": artifacts,
            "estimated_bytes": estimated_bytes,
            "runtime_heat": runtime_heat,
            "runtime_emap_seq": emap_seq,
            "metrics_source": "colibri_runtime",
            "evidence_scope": "unit",
            "runtime_confirmed": True,
            "load_ms": round((time.time() - started) * 1000.0, 3),
        }
    return _load_non_expert_unit(unit, snapshot=snapshot)


def _build_receipt(
    load_request: dict[str, Any],
    *,
    load_request_path: str,
    worker_id: str,
    runtime_snapshot: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    session_id = str(load_request.get("session_id") or "")
    units = list(load_request.get("units") or [])
    unit_results: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    resident_handles: list[str] = []
    cache_tiers: list[str] = []
    bytes_loaded = 0
    errors: list[str] = []
    for unit in units:
        try:
            result = _load_unit(
                dict(unit or {}),
                snapshot=runtime_snapshot,
            )
            unit_results.append(result)
            artifacts.extend(list(result.get("artifacts") or []))
            handle = str(result.get("resident_handle") or "").strip()
            if handle:
                resident_handles.append(handle)
            tier = str(result.get("cache_tier") or "").strip()
            if tier:
                cache_tiers.append(tier)
            bytes_loaded += int(result.get("bytes_loaded") or 0)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}:{exc}")
            unit_results.append({
                "key": str(dict(unit or {}).get("key") or ""),
                "error": f"{type(exc).__name__}:{exc}",
            })
    load_ms = round((time.time() - started) * 1000.0, 3)
    status = "ready" if not errors else "failed"
    cache_tier = cache_tiers[0] if cache_tiers and len(set(cache_tiers)) == 1 else ("mixed" if cache_tiers else "")
    message = (
        f"executor loaded {len(unit_results) - len(errors)}/{len(unit_results)} units"
        if not errors
        else f"executor failed loading {len(errors)} units"
    )
    health = dict(runtime_snapshot.get("health") or {})
    experts = dict(runtime_snapshot.get("experts") or {})
    models = dict(runtime_snapshot.get("models") or {})
    model_items = list(models.get("data") or [])
    return {
        "session_id": session_id,
        "worker_id": worker_id,
        "status": status,
        "completed_at_ms": int(time.time() * 1000),
        "message": message,
        "artifacts": artifacts,
        "metrics": {
            "units_total": len(unit_results),
            "units_loaded": len(unit_results) - len(errors),
            "artifact_count": len(artifacts),
            "load_ms": load_ms,
            "runtime_base_url": str(runtime_snapshot.get("runtime_base_url") or ""),
            "runtime_rows": int(experts.get("rows") or 0),
            "runtime_cols": int(experts.get("cols") or 0),
            "runtime_emap_seq": int(experts.get("seq") or 0),
            "runtime_tiers": dict(health.get("tiers") or {}),
            "runtime_models_count": len(model_items),
            "runtime_model_id": str(dict(model_items[0] or {}).get("id") or "") if model_items else "",
        },
        "load_request_path": load_request_path,
        "cache_tier": cache_tier,
        "bytes_loaded": int(bytes_loaded),
        "resident_handles": resident_handles,
        "load_ms": load_ms,
        "unit_results": unit_results,
    }


def _build_receipt_url(service_base_url: str, receipt_endpoint: str) -> str:
    base = str(service_base_url or "").strip().rstrip("/")
    endpoint = str(receipt_endpoint or "").strip()
    if not base or not endpoint:
        return ""
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return urllib.parse.urljoin(base + "/", endpoint.lstrip("/"))


def _post_receipt(receipt_url: str, receipt: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        receipt_url,
        data=json.dumps(receipt, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        payload = json.loads(raw)
    except Exception:
        payload = {"raw": raw}
    return payload if isinstance(payload, dict) else {"raw": raw}


def main() -> None:
    parser = argparse.ArgumentParser(description="CGC Colibri runtime-backed executor")
    parser.add_argument("--load-request", required=True, help="Path to load_request.json")
    parser.add_argument("--worker-id", default=os.environ.get("CGC_COLIBRI_EXECUTOR_ID", "colibri-executor-local"), help="Executor worker identifier")
    parser.add_argument("--service-base-url", default="", help="Base URL for POSTing the receipt back to the Colibri service")
    parser.add_argument(
        "--runtime-base-url",
        default=os.environ.get("CGC_COLIBRI_RUNTIME_BASE_URL", ""),
        help="Colibri runtime base URL exposing /health and /experts; root or /v1 forms are both accepted",
    )
    parser.add_argument(
        "--runtime-api-key",
        default=os.environ.get("CGC_COLIBRI_RUNTIME_API_KEY", ""),
        help="Optional API key for the Colibri runtime",
    )
    parser.add_argument(
        "--runtime-timeout-ms",
        type=int,
        default=int(os.environ.get("CGC_COLIBRI_RUNTIME_TIMEOUT_MS", "3000") or 3000),
        help="Timeout for runtime telemetry requests",
    )
    parser.add_argument(
        "--transport",
        default="auto",
        choices=["auto", "file", "http", "both"],
        help="How to publish the receipt",
    )
    args = parser.parse_args()

    load_request_path = os.path.abspath(str(args.load_request or "").strip())
    load_request = _read_json(load_request_path)
    receipt_contract = dict(load_request.get("receipt_contract") or {})
    receipt_path = str(receipt_contract.get("receipt_path") or "").strip()
    receipt_endpoint = str(receipt_contract.get("receipt_endpoint") or "").strip()
    runtime_config = _resolve_runtime_config(load_request, args)
    runtime_snapshot = _fetch_runtime_snapshot(
        runtime_base_url=str(runtime_config.get("base_url") or ""),
        runtime_api_key=str(runtime_config.get("api_key") or ""),
        timeout_ms=int(runtime_config.get("timeout_ms") or 0),
    )
    receipt = _build_receipt(
        load_request,
        load_request_path=load_request_path,
        worker_id=str(args.worker_id or ""),
        runtime_snapshot=runtime_snapshot,
    )

    transport = str(args.transport or "auto").strip().lower()
    should_write_file = transport in {"auto", "file", "both"}
    should_post_http = transport in {"http", "both"} or (transport == "auto" and bool(str(args.service_base_url or "").strip()))

    result: dict[str, Any] = {
        "receipt_path": receipt_path,
        "receipt_url": "",
        "wrote_file": False,
        "posted_http": False,
        "http_response": {},
        "runtime_config": runtime_config,
        "runtime_snapshot": runtime_snapshot,
        "receipt": receipt,
    }
    if should_write_file and receipt_path:
        _write_json(receipt_path, receipt)
        result["wrote_file"] = True
    if should_post_http:
        receipt_url = _build_receipt_url(str(args.service_base_url or ""), receipt_endpoint)
        if not receipt_url:
            raise ValueError("Missing receipt URL; provide --service-base-url or an absolute receipt endpoint")
        result["receipt_url"] = receipt_url
        result["http_response"] = _post_receipt(receipt_url, receipt)
        result["posted_http"] = True

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
