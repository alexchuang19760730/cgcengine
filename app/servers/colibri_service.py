#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.shared.colibri_backend import get_colibri_http_contract_spec


logger = logging.getLogger("colibri_service")

app = FastAPI(title="CGC Colibri Service", version="1.0")

WORKER_ID = os.environ.get("CGC_COLIBRI_WORKER_ID", "colibri-worker-local")
WORKER_LOOP_ENABLED = True
WORKER_POLL_MS = float(os.environ.get("CGC_COLIBRI_WORKER_POLL_MS", "100"))
WORKER_STAGING_TIMEOUT_MS = float(
    os.environ.get(
        "CGC_COLIBRI_STAGING_TIMEOUT_MS",
        os.environ.get("CGC_COLIBRI_WORKER_STAGING_MS", "3000"),
    )
)
STATE_ROOT = os.environ.get(
    "CGC_COLIBRI_STATE_ROOT",
    os.path.join(REPO_ROOT, "var", "colibri"),
)
SESSIONS_ROOT = os.path.join(STATE_ROOT, "sessions")
STATS_PATH = os.path.join(STATE_ROOT, "service_stats.json")
METRICS_ROOT = os.environ.get(
    "COLI_METRICS_ROOT",
    os.path.join(REPO_ROOT, "var", "colibri_metrics"),
)
SERVICE_METRICS_ROOT = os.path.join(METRICS_ROOT, "colibri_service")
SERVICE_METRICS_EVENTS_PATH = os.path.join(SERVICE_METRICS_ROOT, "events.jsonl")
SERVICE_METRICS_SESSIONS_ROOT = os.path.join(SERVICE_METRICS_ROOT, "sessions")
_sessions: dict[str, dict[str, Any]] = {}
_stats = {
    "accepted_requests": 0,
    "rejected_requests": 0,
    "last_submit_ms": 0.0,
}
_VALID_STATES = set(get_colibri_http_contract_spec().get("lifecycle_states") or [])
_worker_task: asyncio.Task | None = None


def _ensure_state_dirs() -> None:
    os.makedirs(SESSIONS_ROOT, exist_ok=True)
    os.makedirs(SERVICE_METRICS_SESSIONS_ROOT, exist_ok=True)


def _session_dir(session_id: str) -> str:
    return os.path.join(SESSIONS_ROOT, str(session_id or "unknown"))


def _session_state_path(session_id: str) -> str:
    return os.path.join(_session_dir(session_id), "session.json")


def _session_evidence_path(session_id: str) -> str:
    return os.path.join(_session_dir(session_id), "staging_evidence.json")


def _session_load_request_path(session_id: str) -> str:
    return os.path.join(_session_dir(session_id), "load_request.json")


def _session_load_receipt_path(session_id: str) -> str:
    return os.path.join(_session_dir(session_id), "load_receipt.json")


def _safe_write_json(path: str, payload: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def _append_jsonl(path: str, payload: dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "ab") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        )


def _session_metric_path(session_id: str) -> str:
    return os.path.join(SERVICE_METRICS_SESSIONS_ROOT, f"{str(session_id or 'unknown')}.json")


def _session_metric_snapshot(session: dict[str, Any]) -> dict[str, Any]:
    created_at_ms = int(session.get("created_at_ms") or 0)
    updated_at_ms = int(session.get("updated_at_ms") or created_at_ms or int(time.time() * 1000))
    staging_started_at_ms = int(session.get("staging_started_at_ms") or 0)
    staging_completed_at_ms = int(session.get("staging_completed_at_ms") or 0)
    receipt = dict(session.get("load_receipt") or {})
    evidence = dict(session.get("evidence") or {})
    queue_wait_ms = (
        max(staging_started_at_ms - created_at_ms, 0)
        if created_at_ms and staging_started_at_ms
        else None
    )
    staging_ms = (
        max(staging_completed_at_ms - staging_started_at_ms, 0)
        if staging_started_at_ms and staging_completed_at_ms
        else None
    )
    ready_latency_ms = (
        max(updated_at_ms - created_at_ms, 0)
        if created_at_ms and str(session.get("state") or "") in {"ready", "running", "closed"}
        else None
    )
    return {
        "schema_version": 1,
        "layer": "colibri_service",
        "scope_kind": "session",
        "event_type": "session_kpi",
        "session_id": str(session.get("session_id") or ""),
        "benchmark_run_id": str(session.get("benchmark_run_id") or ""),
        "benchmark_case": str(session.get("benchmark_case") or ""),
        "state": str(session.get("state") or ""),
        "accepted": bool(session.get("accepted", True)),
        "worker_id": str(session.get("worker_id") or WORKER_ID),
        "queue_depth": int(session.get("queue_depth") or 0),
        "request_seq": int(session.get("request_seq") or 0),
        "frontier_key": str(session.get("frontier_key") or ""),
        "model": str(session.get("model") or ""),
        "mode": str(session.get("mode") or ""),
        "created_at_ms": created_at_ms,
        "updated_at_ms": updated_at_ms,
        "last_transition_at_ms": int(session.get("last_transition_at_ms") or 0),
        "staging_started_at_ms": staging_started_at_ms,
        "staging_completed_at_ms": staging_completed_at_ms,
        "queue_wait_ms": queue_wait_ms,
        "staging_ms": staging_ms,
        "ready_latency_ms": ready_latency_ms,
        "receipt_status": str(receipt.get("status") or evidence.get("status") or ""),
        "receipt_load_ms": float(receipt.get("load_ms") or evidence.get("load_ms") or 0.0),
        "receipt_bytes_loaded": int(receipt.get("bytes_loaded") or evidence.get("bytes_loaded") or 0),
        "resident_handle_count": len(list(receipt.get("resident_handles") or evidence.get("resident_handles") or [])),
        "artifact_count": len(list(receipt.get("artifacts") or [])),
        "evidence_valid": bool(evidence.get("valid")),
        "message": str(session.get("message") or ""),
        "session_path": str(session.get("session_path") or _session_state_path(str(session.get("session_id") or ""))),
    }


def _persist_session_metric(session: dict[str, Any], *, event_type: str, extra: dict[str, Any] | None = None) -> None:
    snapshot = _session_metric_snapshot(session)
    _safe_write_json(_session_metric_path(str(session.get("session_id") or "")), snapshot)
    event = dict(snapshot)
    event["event_type"] = str(event_type or "session_event")
    event["recorded_at_ms"] = int(time.time() * 1000)
    if extra:
        event.update(dict(extra))
    _append_jsonl(SERVICE_METRICS_EVENTS_PATH, event)


def _record_service_event(event_type: str, **payload: Any) -> None:
    _append_jsonl(
        SERVICE_METRICS_EVENTS_PATH,
        {
            "schema_version": 1,
            "layer": "colibri_service",
            "scope_kind": "service",
            "event_type": str(event_type or "service_event"),
            "recorded_at_ms": int(time.time() * 1000),
            **payload,
        },
    )


def _persist_stats() -> None:
    _ensure_state_dirs()
    _safe_write_json(STATS_PATH, dict(_stats))


def _persist_session(session: dict[str, Any]) -> None:
    session_id = str(session.get("session_id") or "").strip()
    if not session_id:
        return
    _ensure_state_dirs()
    _safe_write_json(_session_state_path(session_id), session)


def _persist_all_sessions() -> None:
    for session in _sessions.values():
        _persist_session(session)


def _load_persisted_state() -> None:
    _ensure_state_dirs()
    if os.path.exists(STATS_PATH):
        try:
            with open(STATS_PATH, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                _stats.update({
                    "accepted_requests": int(payload.get("accepted_requests") or 0),
                    "rejected_requests": int(payload.get("rejected_requests") or 0),
                    "last_submit_ms": float(payload.get("last_submit_ms") or 0.0),
                })
        except Exception:
            logger.exception("Failed to load Colibri stats from %s", STATS_PATH)
    try:
        for entry in sorted(os.listdir(SESSIONS_ROOT)):
            session_path = os.path.join(SESSIONS_ROOT, entry, "session.json")
            if not os.path.exists(session_path):
                continue
            with open(session_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                continue
            session_id = str(payload.get("session_id") or "").strip()
            if session_id:
                _sessions[session_id] = payload
                _safe_write_json(_session_metric_path(session_id), _session_metric_snapshot(payload))
    except FileNotFoundError:
        pass
    except Exception:
        logger.exception("Failed to load persisted Colibri sessions from %s", SESSIONS_ROOT)
    _recompute_queue_depths()


def _safe_read_json(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _required_response(
    *,
    accepted: bool,
    session_state: str,
    worker_id: str,
    queue_depth: int,
    session_id: str,
    message: str = "",
) -> dict[str, Any]:
    return {
        "accepted": bool(accepted),
        "session_state": str(session_state or "blocked"),
        "worker_id": str(worker_id or ""),
        "queue_depth": int(queue_depth),
        "session_id": str(session_id or ""),
        "message": str(message or ""),
    }


def _state_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in _sessions.values():
        state = str(item.get("state") or "unknown")
        counts[state] = int(counts.get(state, 0)) + 1
    return counts


def _queued_count() -> int:
    return sum(1 for item in _sessions.values() if str(item.get("state") or "") == "queued")


def _recompute_queue_depths() -> None:
    queued = sorted(
        (
            item for item in _sessions.values()
            if str(item.get("state") or "") == "queued"
        ),
        key=lambda item: (
            int(item.get("created_at_ms") or 0),
            str(item.get("session_id") or ""),
        ),
    )
    for idx, item in enumerate(queued, start=1):
        item["queue_depth"] = int(idx)
    for item in _sessions.values():
        if str(item.get("state") or "") != "queued":
            item["queue_depth"] = 0


def _build_executor_load_request(session: dict[str, Any]) -> dict[str, Any]:
    request_payload = dict(session.get("request_payload") or {})
    lanes = dict(request_payload.get("lanes") or {})
    transport_payload = dict(session.get("transport_payload") or {})
    transport_metadata = dict(transport_payload.get("metadata") or {})
    units: list[dict[str, Any]] = []
    for lane in ("current", "next", "next_next", "far"):
        for unit in list(lanes.get(lane) or []):
            if isinstance(unit, dict):
                units.append({"lane": lane, **dict(unit)})
    now_ms = int(time.time() * 1000)
    load_request = {
        "protocol": str(get_colibri_http_contract_spec().get("protocol") or ""),
        "kind": "colibri_load_request",
        "created_at_ms": now_ms,
        "session_id": str(session.get("session_id") or ""),
        "worker_id": str(session.get("worker_id") or WORKER_ID),
        "frontier_key": str(session.get("frontier_key") or ""),
        "model": str(session.get("model") or ""),
        "mode": str(session.get("mode") or ""),
        "request_seq": int(session.get("request_seq") or 0),
        "units_total": len(units),
        "runtime": {
            "base_url": str(
                transport_metadata.get("runtime_base_url")
                or transport_metadata.get("telemetry_base_url")
                or transport_metadata.get("colibri_runtime_base_url")
                or ""
            ).strip(),
            "api_key_env": str(transport_metadata.get("runtime_api_key_env") or "").strip(),
            "telemetry_endpoints": {
                "health": "/health",
                "experts": "/experts",
            },
        },
        "units": [
            {
                "lane": str(unit.get("lane") or ""),
                "key": str(unit.get("key") or ""),
                "unit_kind": str(unit.get("unit_kind") or ""),
                "prefetch_role": str(unit.get("prefetch_role") or ""),
                "layer_id": int(unit.get("layer_id") or 0),
                "expert_id": int(unit.get("expert_id") or 0),
                "target_tier": str(unit.get("target_tier") or ""),
                "routing_heat": float(unit.get("routing_heat") or 0.0),
                "pin_priority": float(unit.get("pin_priority") or 0.0),
                "source_locator": dict(unit.get("source_locator") or {}),
                "tags": list(unit.get("tags") or []),
            }
            for unit in units
        ],
        "receipt_contract": {
            "required_fields": list(
                dict(get_colibri_http_contract_spec().get("executor_receipt") or {}).get("required_fields") or []
            ),
            "receipt_path": _session_load_receipt_path(str(session.get("session_id") or "")),
            "receipt_endpoint": (
                f"/colibri/session/{str(session.get('session_id') or '')}/receipt"
            ),
        },
    }
    _safe_write_json(_session_load_request_path(str(session.get("session_id") or "")), load_request)
    session["load_request"] = {
        "created_at_ms": now_ms,
        "units_total": len(units),
        "load_request_path": _session_load_request_path(str(session.get("session_id") or "")),
        "receipt_path": _session_load_receipt_path(str(session.get("session_id") or "")),
        "receipt_endpoint": f"/colibri/session/{str(session.get('session_id') or '')}/receipt",
    }
    return load_request


def _normalize_load_receipt(payload: dict[str, Any], *, session_id: str) -> dict[str, Any]:
    spec = dict(get_colibri_http_contract_spec().get("executor_receipt") or {})
    required_fields = list(spec.get("required_fields") or [])
    missing_required = [field for field in required_fields if field not in payload]
    status = str(payload.get("status") or "").strip().lower()
    session_value = str(payload.get("session_id") or "").strip()
    message = str(payload.get("message") or "").strip()
    valid_statuses = {str(item or "").strip().lower() for item in list(spec.get("status_values") or [])}
    normalized = {
        "session_id": session_value,
        "worker_id": str(payload.get("worker_id") or "").strip(),
        "status": status,
        "completed_at_ms": int(payload.get("completed_at_ms") or 0),
        "message": message,
        "artifacts": list(payload.get("artifacts") or []),
        "metrics": dict(payload.get("metrics") or {}),
        "load_request_path": str(payload.get("load_request_path") or "").strip(),
        "cache_tier": str(payload.get("cache_tier") or "").strip(),
        "bytes_loaded": int(payload.get("bytes_loaded") or 0),
        "resident_handles": list(payload.get("resident_handles") or []),
        "load_ms": float(payload.get("load_ms") or 0.0),
        "unit_results": list(payload.get("unit_results") or []),
    }
    errors: list[str] = []
    if missing_required:
        errors.append("missing_required:" + ",".join(missing_required))
    if session_value != str(session_id or ""):
        errors.append(f"session_mismatch:{session_value or 'empty'}")
    if status not in valid_statuses:
        errors.append(f"invalid_status:{status or 'empty'}")
    if normalized["completed_at_ms"] <= 0:
        errors.append("invalid_completed_at_ms")
    return {
        **normalized,
        "valid": not errors,
        "errors": errors,
    }


def _read_session_receipt(session: dict[str, Any]) -> dict[str, Any]:
    session_id = str(session.get("session_id") or "")
    receipt = _safe_read_json(_session_load_receipt_path(session_id))
    if receipt:
        return _normalize_load_receipt(receipt, session_id=session_id)
    inline = dict(session.get("load_receipt") or {})
    if inline:
        return _normalize_load_receipt(inline, session_id=session_id)
    return {
        "session_id": session_id,
        "worker_id": "",
        "status": "",
        "completed_at_ms": 0,
        "message": "",
        "artifacts": [],
        "metrics": {},
        "load_request_path": "",
        "valid": False,
        "errors": ["receipt_missing"],
    }


def _persist_receipt_summary(session: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    previous = dict(session.get("evidence") or {})
    summary = {
        "status": str(receipt.get("status") or "waiting"),
        "checked_at_ms": int(time.time() * 1000),
        "completed_at_ms": int(receipt.get("completed_at_ms") or 0),
        "worker_id": str(receipt.get("worker_id") or ""),
        "message": str(receipt.get("message") or ""),
        "artifact_count": len(list(receipt.get("artifacts") or [])),
        "cache_tier": str(receipt.get("cache_tier") or ""),
        "bytes_loaded": int(receipt.get("bytes_loaded") or 0),
        "resident_handles": list(receipt.get("resident_handles") or []),
        "load_ms": float(receipt.get("load_ms") or 0.0),
        "receipt_path": _session_load_receipt_path(str(session.get("session_id") or "")),
        "load_request_path": _session_load_request_path(str(session.get("session_id") or "")),
        "valid": bool(receipt.get("valid")),
        "errors": list(receipt.get("errors") or []),
    }
    _safe_write_json(_session_evidence_path(str(session.get("session_id") or "")), summary)
    session["evidence"] = dict(summary)
    previous_cmp = dict(previous)
    previous_cmp.pop("checked_at_ms", None)
    summary_cmp = dict(summary)
    summary_cmp.pop("checked_at_ms", None)
    if previous_cmp != summary_cmp:
        _persist_session_metric(
            session,
            event_type="receipt_summary",
            extra={
                "receipt_status": str(summary.get("status") or ""),
                "receipt_valid": bool(summary.get("valid")),
            },
        )
    return summary


def _update_session_state(
    session: dict[str, Any],
    next_state: str,
    *,
    message: str = "",
    queue_depth: int | None = None,
    accepted: bool | None = None,
    worker_id: str = "",
    mark_transition: bool = True,
) -> None:
    now_ms = int(time.time() * 1000)
    previous_state = str(session.get("state") or "")
    previous_message = str(session.get("message") or "")
    session["state"] = str(next_state or session.get("state") or "blocked")
    if accepted is not None:
        session["accepted"] = bool(accepted)
    if queue_depth is not None:
        session["queue_depth"] = int(queue_depth)
    session["worker_id"] = str(worker_id or session.get("worker_id") or WORKER_ID)
    session["message"] = str(message or session.get("message") or "")
    if next_state == "staging" and previous_state != "staging":
        session["staging_started_at_ms"] = now_ms
    if next_state != "staging" and previous_state == "staging":
        session["staging_completed_at_ms"] = now_ms
    session["updated_at_ms"] = now_ms
    if mark_transition:
        session["last_transition_at_ms"] = now_ms
    _persist_session(session)
    state_changed = previous_state != str(session.get("state") or "")
    message_changed = previous_message != str(session.get("message") or "")
    if state_changed or (mark_transition and message_changed):
        _persist_session_metric(
            session,
            event_type="session_state_change",
            extra={
                "previous_state": previous_state,
                "next_state": str(session.get("state") or ""),
            },
        )


def _advance_staging_session(session: dict[str, Any]) -> None:
    _build_executor_load_request(session)
    receipt = _read_session_receipt(session)
    evidence = _persist_receipt_summary(session, receipt)
    receipt_status = str(receipt.get("status") or "").strip().lower()
    if bool(receipt.get("valid")) and receipt_status == "ready":
        _update_session_state(
            session,
            "ready",
            message=(
                "executor load receipt accepted"
                + (f": {str(receipt.get('message') or '').strip()}" if str(receipt.get("message") or "").strip() else "")
            ),
            queue_depth=0,
            worker_id=str(receipt.get("worker_id") or WORKER_ID),
        )
        return
    if bool(receipt.get("valid")) and receipt_status in {"failed", "blocked"}:
        _update_session_state(
            session,
            "blocked",
            message=(
                "executor load receipt "
                f"{receipt_status}"
                + (f": {str(receipt.get('message') or '').strip()}" if str(receipt.get("message") or "").strip() else "")
            ),
            queue_depth=0,
            worker_id=str(receipt.get("worker_id") or WORKER_ID),
        )
        return
    staging_started_at_ms = int(
        session.get("staging_started_at_ms")
        or session.get("last_transition_at_ms")
        or session.get("updated_at_ms")
        or 0
    )
    elapsed_ms = max(int(time.time() * 1000) - staging_started_at_ms, 0)
    if elapsed_ms >= int(WORKER_STAGING_TIMEOUT_MS):
        _update_session_state(
            session,
            "blocked",
            message=(
                "load receipt timeout "
                f"after {elapsed_ms} ms"
            ),
            queue_depth=0,
            worker_id=WORKER_ID,
        )
        return
    _update_session_state(
        session,
        "staging",
        message=(
            "waiting for executor load receipt"
            + (
                f" ({','.join(list(receipt.get('errors') or [])[:2])})"
                if list(receipt.get("errors") or []) and "receipt_missing" not in list(receipt.get("errors") or [])
                else ""
            )
        ),
        queue_depth=0,
        worker_id=WORKER_ID,
        mark_transition=False,
    )


async def _worker_loop() -> None:
    logger.info(
        "Colibri worker loop enabled (poll_ms=%s, staging_timeout_ms=%s, state_root=%s)",
        WORKER_POLL_MS,
        WORKER_STAGING_TIMEOUT_MS,
        STATE_ROOT,
    )
    while True:
        try:
            _recompute_queue_depths()
            _persist_all_sessions()
            queued = sorted(
                (
                    item for item in _sessions.values()
                    if str(item.get("state") or "") == "queued"
                ),
                key=lambda item: (
                    int(item.get("created_at_ms") or 0),
                    str(item.get("session_id") or ""),
                ),
            )
            if not queued:
                staging = [
                    item for item in _sessions.values()
                    if str(item.get("state") or "") == "staging"
                ]
                for session in staging:
                    _advance_staging_session(session)
                await asyncio.sleep(max(WORKER_POLL_MS / 1000.0, 0.05))
                continue
            session = queued[0]
            _update_session_state(
                session,
                "staging",
                message="worker published load request and is waiting for executor receipt",
                queue_depth=0,
                worker_id=WORKER_ID,
            )
            _advance_staging_session(session)
            staging = [
                item for item in _sessions.values()
                if str(item.get("state") or "") == "staging"
            ]
            for item in staging:
                if item is not session:
                    _advance_staging_session(item)
            await asyncio.sleep(max(WORKER_POLL_MS / 1000.0, 0.05))
        except asyncio.CancelledError:
            logger.info("Colibri worker loop stopped")
            raise
        except Exception:
            logger.exception("Colibri worker loop iteration failed")
            await asyncio.sleep(max(WORKER_POLL_MS / 1000.0, 0.1))


def _session_status_payload(session: dict[str, Any]) -> dict[str, Any]:
    return _required_response(
        accepted=bool(session.get("accepted", True)),
        session_state=str(session.get("state") or "blocked"),
        worker_id=str(session.get("worker_id") or WORKER_ID),
        queue_depth=int(session.get("queue_depth") or 0),
        session_id=str(session.get("session_id") or ""),
        message=str(session.get("message") or ""),
    ) | {
        "request_seq": int(session.get("request_seq") or 0),
        "frontier_key": str(session.get("frontier_key") or ""),
        "model": str(session.get("model") or ""),
        "mode": str(session.get("mode") or ""),
        "created_at_ms": int(session.get("created_at_ms") or 0),
        "updated_at_ms": int(session.get("updated_at_ms") or 0),
        "last_transition_at_ms": int(session.get("last_transition_at_ms") or 0),
        "session_path": str(session.get("session_path") or _session_state_path(str(session.get("session_id") or ""))),
        "load_request": dict(session.get("load_request") or {}),
        "load_receipt": dict(session.get("load_receipt") or {}),
        "evidence": dict(session.get("evidence") or {}),
    }


def _validate_submit_payload(payload: dict[str, Any]) -> tuple[bool, str]:
    spec = get_colibri_http_contract_spec()
    request_spec = dict(spec.get("request") or {})
    required_top = list(request_spec.get("required_top_level_fields") or [])
    for field in required_top:
        if field not in payload:
            return False, f"missing_top_level_field:{field}"
    protocol = str(payload.get("protocol") or "")
    if protocol != str(spec.get("protocol") or ""):
        return False, f"unsupported_protocol:{protocol or 'empty'}"
    action = str(payload.get("action") or "")
    if action != str(request_spec.get("action") or ""):
        return False, f"unsupported_action:{action or 'empty'}"
    session = dict(payload.get("session") or {})
    for field in list(request_spec.get("session_required_fields") or []):
        if field not in session:
            return False, f"missing_session_field:{field}"
    request_payload = dict(payload.get("request") or {})
    if not bool(request_payload.get("enabled")):
        return False, "engine_request_disabled"
    if not bool(dict(request_payload.get("summary") or {}).get("engine_bridge_ready")):
        return False, "engine_bridge_not_ready"
    return True, ""


@app.get("/colibri/health")
async def colibri_health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "colibri",
        "worker_id": WORKER_ID,
        "worker_loop_enabled": bool(WORKER_LOOP_ENABLED),
        "worker_poll_ms": float(WORKER_POLL_MS),
        "worker_staging_timeout_ms": float(WORKER_STAGING_TIMEOUT_MS),
        "queued_sessions": int(_queued_count()),
        "total_sessions": int(len(_sessions)),
        "state_root": STATE_ROOT,
        "state_counts": _state_counts(),
        "stats": dict(_stats),
    }


@app.get("/colibri/spec")
async def colibri_spec() -> dict[str, Any]:
    return get_colibri_http_contract_spec()


@app.on_event("startup")
async def _startup_worker_loop() -> None:
    global _worker_task
    _load_persisted_state()
    if WORKER_LOOP_ENABLED and _worker_task is None:
        _worker_task = asyncio.create_task(_worker_loop())


@app.on_event("shutdown")
async def _shutdown_worker_loop() -> None:
    global _worker_task
    if _worker_task is not None:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        _worker_task = None


@app.get("/colibri/session/{session_id}")
async def colibri_get_session(session_id: str):
    session = _sessions.get(str(session_id or ""))
    if not session:
        raise HTTPException(status_code=404, detail="session_not_found")
    return JSONResponse(status_code=200, content=_session_status_payload(session))


@app.post("/colibri/session")
async def colibri_session(request: Request):
    t0 = time.time()
    benchmark_run_id = str(request.headers.get("X-Coli-Benchmark-Run-Id", "") or "")
    benchmark_case = str(request.headers.get("X-Coli-Benchmark-Case", "") or "")
    try:
        payload = await request.json()
    except Exception:
        _stats["rejected_requests"] += 1
        _persist_stats()
        _record_service_event(
            "session_submit",
            status="error",
            accepted=False,
            http_status=400,
            message="invalid_json",
            submit_ms=round((time.time() - t0) * 1000.0, 3),
            benchmark_run_id=benchmark_run_id,
            benchmark_case=benchmark_case,
        )
        return JSONResponse(
            status_code=400,
            content=_required_response(
                accepted=False,
                session_state="blocked",
                worker_id=WORKER_ID,
                queue_depth=sum(1 for item in _sessions.values() if str(item.get("state") or "") == "queued"),
                session_id="",
                message="invalid_json",
            ),
        )

    session = dict(payload.get("session") or {})
    session_id = str(session.get("session_id") or "")
    valid, reason = _validate_submit_payload(dict(payload or {}))
    queued_count = _queued_count()
    if not valid:
        _stats["rejected_requests"] += 1
        _stats["last_submit_ms"] = (time.time() - t0) * 1000.0
        _persist_stats()
        _record_service_event(
            "session_submit",
            status="error",
            accepted=False,
            http_status=400,
            session_id=session_id,
            message=reason,
            submit_ms=round((time.time() - t0) * 1000.0, 3),
            benchmark_run_id=benchmark_run_id,
            benchmark_case=benchmark_case,
        )
        return JSONResponse(
            status_code=400,
            content=_required_response(
                accepted=False,
                session_state="blocked",
                worker_id=WORKER_ID,
                queue_depth=int(queued_count),
                session_id=session_id,
                message=reason,
            ),
        )

    now_ms = int(time.time() * 1000)
    previous_session = dict(_sessions.get(session_id) or {})
    queue_depth = int(queued_count) + (
        0 if session_id in _sessions and str(_sessions[session_id].get("state") or "") == "queued" else 1
    )
    _sessions[session_id] = {
        "session_id": session_id,
        "accepted": True,
        "state": "queued",
        "worker_id": WORKER_ID,
        "queue_depth": queue_depth,
        "message": "queued for colibri staging",
        "benchmark_run_id": benchmark_run_id,
        "benchmark_case": benchmark_case,
        "created_at_ms": now_ms if not previous_session else int(previous_session.get("created_at_ms") or now_ms),
        "updated_at_ms": now_ms,
        "last_transition_at_ms": now_ms,
        "request_seq": int(session.get("request_seq") or 0),
        "frontier_key": str(session.get("frontier_key") or ""),
        "model": str(session.get("model") or ""),
        "mode": str(session.get("mode") or ""),
        "request_payload": dict(payload.get("request") or {}),
        "transport_payload": dict(payload.get("transport") or {}),
        "client_payload": dict(payload.get("client") or {}),
        "response_contract": dict(payload.get("response_contract") or {}),
        "session_path": _session_state_path(session_id),
        "evidence": dict(previous_session.get("evidence") or {}),
        "load_request": {
            "load_request_path": _session_load_request_path(session_id),
            "receipt_path": _session_load_receipt_path(session_id),
            "receipt_endpoint": f"/colibri/session/{session_id}/receipt",
        },
        "load_receipt": dict(previous_session.get("load_receipt") or {}),
    }
    _recompute_queue_depths()
    _stats["accepted_requests"] += 1
    _stats["last_submit_ms"] = (time.time() - t0) * 1000.0
    _persist_session(_sessions[session_id])
    _persist_session_metric(
        _sessions[session_id],
        event_type="session_submit",
        extra={
            "http_status": 202,
            "submit_ms": round((time.time() - t0) * 1000.0, 3),
        },
    )
    _persist_stats()
    return JSONResponse(
        status_code=202,
        content=_required_response(
            accepted=True,
            session_state="queued",
            worker_id=WORKER_ID,
            queue_depth=int(queue_depth),
            session_id=session_id,
            message="queued for colibri staging",
        ),
    )


@app.post("/colibri/session/{session_id}/receipt")
async def colibri_submit_receipt(session_id: str, request: Request):
    session = _sessions.get(str(session_id or ""))
    if not session:
        raise HTTPException(status_code=404, detail="session_not_found")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_json")
    normalized = _normalize_load_receipt(dict(payload or {}), session_id=str(session_id or ""))
    if not bool(normalized.get("valid")):
        raise HTTPException(status_code=400, detail=";".join(list(normalized.get("errors") or [])))
    session["load_receipt"] = {
        "session_id": str(normalized.get("session_id") or ""),
        "worker_id": str(normalized.get("worker_id") or ""),
        "status": str(normalized.get("status") or ""),
        "completed_at_ms": int(normalized.get("completed_at_ms") or 0),
        "message": str(normalized.get("message") or ""),
        "artifacts": list(normalized.get("artifacts") or []),
        "metrics": dict(normalized.get("metrics") or {}),
        "load_request_path": str(normalized.get("load_request_path") or ""),
        "cache_tier": str(normalized.get("cache_tier") or ""),
        "bytes_loaded": int(normalized.get("bytes_loaded") or 0),
        "resident_handles": list(normalized.get("resident_handles") or []),
        "load_ms": float(normalized.get("load_ms") or 0.0),
        "unit_results": list(normalized.get("unit_results") or []),
    }
    _safe_write_json(_session_load_receipt_path(str(session_id or "")), dict(session.get("load_receipt") or {}))
    _persist_receipt_summary(session, normalized)
    _persist_session(session)
    _persist_session_metric(
        session,
        event_type="receipt_submit",
        extra={
            "http_status": 200,
            "receipt_status": str(normalized.get("status") or ""),
            "receipt_worker_id": str(normalized.get("worker_id") or ""),
        },
    )
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "session_id": str(session_id or ""),
            "receipt_path": _session_load_receipt_path(str(session_id or "")),
            "status": str(normalized.get("status") or ""),
            "worker_id": str(normalized.get("worker_id") or ""),
        },
    )


@app.post("/colibri/session/{session_id}/transition")
async def colibri_transition_session(session_id: str, request: Request):
    session = _sessions.get(str(session_id or ""))
    if not session:
        raise HTTPException(status_code=404, detail="session_not_found")
    data = await request.json()
    next_state = str(data.get("session_state") or data.get("state") or "").strip()
    if next_state not in _VALID_STATES:
        raise HTTPException(status_code=400, detail=f"invalid_state:{next_state or 'empty'}")
    queue_depth = None
    if "queue_depth" in data:
        queue_depth = int(data.get("queue_depth") or 0)
    elif next_state in {"ready", "running", "closed", "failed"}:
        queue_depth = 0
    _update_session_state(
        session,
        next_state,
        message=str(data.get("message") or session.get("message") or ""),
        queue_depth=queue_depth,
        accepted=bool(data.get("accepted", session.get("accepted", True))),
        worker_id=str(data.get("worker_id") or session.get("worker_id") or WORKER_ID),
    )
    _recompute_queue_depths()
    _persist_all_sessions()
    return JSONResponse(status_code=200, content=_session_status_payload(session))


def main() -> None:
    global WORKER_ID, WORKER_LOOP_ENABLED, WORKER_POLL_MS, WORKER_STAGING_TIMEOUT_MS
    parser = argparse.ArgumentParser(description="CGC Colibri Service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=30110, help="Server port")
    parser.add_argument("--worker-id", default=WORKER_ID, help="Worker identifier")
    parser.add_argument("--disable-worker-loop", action="store_true", help="Disable automatic queued->staging->ready worker loop")
    parser.add_argument("--worker-poll-ms", type=float, default=WORKER_POLL_MS, help="Worker poll interval in milliseconds")
    parser.add_argument(
        "--staging-timeout-ms",
        "--worker-staging-ms",
        dest="staging_timeout_ms",
        type=float,
        default=WORKER_STAGING_TIMEOUT_MS,
        help="Maximum wait for executor load receipt before blocking the session",
    )
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()

    WORKER_ID = str(args.worker_id or WORKER_ID)
    WORKER_LOOP_ENABLED = not bool(args.disable_worker_loop)
    WORKER_POLL_MS = float(args.worker_poll_ms)
    WORKER_STAGING_TIMEOUT_MS = float(args.staging_timeout_ms)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level or "info").upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info("Starting Colibri service on port %s", args.port)
    logger.info("Worker ID: %s", WORKER_ID)
    logger.info("Worker loop enabled: %s", WORKER_LOOP_ENABLED)
    logger.info("State root: %s", STATE_ROOT)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
