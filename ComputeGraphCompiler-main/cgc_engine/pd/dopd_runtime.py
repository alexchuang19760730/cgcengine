"""
Minimal DOPD runtime for PD service.

This phase intentionally keeps the runtime small:
- in-memory session / handoff tracking
- optional HTTP worker bridge for ResumeDecode
- no changes to the existing tp4ep4 prefill execution path
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .dopd_schema import decode_dopd_resume_payload_v2
from .dopd_schema import extract_dopd_resume_state_bytes


def _now_us() -> int:
    return int(time.time() * 1_000_000)


def _stringify_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
    return {str(k): str(v) for k, v in (metadata or {}).items()}


def _env_value(*names: str) -> str:
    for name in names:
        raw = str(os.environ.get(name) or "").strip()
        if raw:
            return raw
    return ""


def _load_json_file(path_str: str) -> Dict[str, Any]:
    path_str = str(path_str or "").strip()
    if not path_str:
        return {}
    try:
        payload = json.loads(Path(path_str).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_contract_path(path_str: str, profile_source_path: str = "") -> str:
    raw = str(path_str or "").strip()
    if not raw:
        return ""
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    candidates = []
    if profile_source_path:
        candidates.append((Path(profile_source_path).expanduser().resolve().parent / raw).resolve())
    repo_root = Path(__file__).resolve().parents[3]
    candidates.append((repo_root / raw).resolve())
    candidates.append((repo_root / "docs" / "technical_whitepapers" / "examples" / raw).resolve())
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0] if candidates else path)


def _infer_protocol_family(state_kind: str) -> str:
    explicit = _env_value("CGC_RUNTIME_PROTOCOL_FAMILY", "CGC_MEGATRAIN_PROTOCOL_FAMILY")
    if explicit:
        return explicit
    if str(state_kind or "").strip().lower().startswith("kda_state"):
        return "trueorthokda"
    return "generic_runtime"


def _infer_pd_mode() -> str:
    explicit = _env_value("CGC_MEGATRAIN_PD_MODE", "CGC_PD_MODE")
    if explicit:
        return explicit
    return "cloud_prefill_edge_decode"


def _load_profile_bundle_context() -> Dict[str, str]:
    profile_path = _env_value(
        "CGC_SGLANG_PROFILE_SETTINGS_PATH",
        "CGC_PROFILE_SETTINGS_PATH",
        "CGC_HOST2_BENCH_PROFILE_PATH",
    )
    profile = _load_json_file(profile_path)
    if not profile:
        return {}
    bootstrap_contract_path = _resolve_contract_path(
        str(profile.get("bootstrap_contract_path") or ""),
        profile_path,
    )
    bootstrap_contract = _load_json_file(bootstrap_contract_path)
    system_profile_ref = (
        profile.get("system_profile_ref")
        if isinstance(profile.get("system_profile_ref"), dict)
        else {}
    )
    system_manifest_path = _resolve_contract_path(
        str(system_profile_ref.get("source_path") or ""),
        profile_path,
    )
    system_manifest = _load_json_file(system_manifest_path)
    system_profile = (
        system_manifest.get("system_profile")
        if isinstance(system_manifest.get("system_profile"), dict)
        else {}
    )
    model_contract_ref = (
        system_profile.get("model_contract_ref")
        if isinstance(system_profile.get("model_contract_ref"), dict)
        else {}
    )
    model_contract_path = _resolve_contract_path(
        str(model_contract_ref.get("source_path") or ""),
        system_manifest_path or profile_path,
    )
    model_contract = _load_json_file(model_contract_path)
    environment_bootstrap_ref = (
        system_profile.get("environment_bootstrap_ref")
        if isinstance(system_profile.get("environment_bootstrap_ref"), dict)
        else {}
    )
    return {
        "profile_settings_path": profile_path,
        "execution_profile_binding_key": str(profile.get("execution_profile_binding_key") or ""),
        "bootstrap_contract_binding_key": str(profile.get("bootstrap_contract_binding_key") or ""),
        "flow_parameter_contract_binding_key": str(profile.get("flow_parameter_contract_binding_key") or ""),
        "bootstrap_contract_path": bootstrap_contract_path,
        "bootstrap_contract_id": str(bootstrap_contract.get("bootstrap_contract_id") or ""),
        "system_manifest_path": system_manifest_path,
        "system_profile_id": str(system_profile.get("profile_id") or ""),
        "model_contract_path": model_contract_path,
        "model_contract_id": str(model_contract.get("contract_id") or ""),
        "profile_protocol_family": str(environment_bootstrap_ref.get("protocol_family") or ""),
        "profile_state_kind": str(environment_bootstrap_ref.get("state_kind") or ""),
        "profile_state_codec": str(environment_bootstrap_ref.get("state_codec") or ""),
    }


def _build_contract_context() -> Dict[str, str]:
    state_kind = _env_value("CGC_RUNTIME_STATE_KIND", "CGC_MEGATRAIN_STATE_KIND", "CGC_STATE_KIND") or "kda_state_v1"
    state_codec = (
        _env_value("CGC_RUNTIME_STATE_CODEC", "CGC_MEGATRAIN_STATE_CODEC", "CGC_STATE_CODEC", "CGC_CLOUD_STATE_CODEC")
        or "cq4"
    )
    context = {
        "state_kind": state_kind,
        "state_codec": state_codec,
        "protocol_family": _infer_protocol_family(state_kind),
        "pd_mode": _infer_pd_mode(),
    }
    context.update(_load_profile_bundle_context())
    return {key: value for key, value in context.items() if str(value or "").strip()}


@dataclass
class DOPDHandoffRecord:
    session_id: str
    handoff_id: str
    source_role: str
    target_role: str
    phase_role: str
    model_name: str
    cache_schema: str
    kv_variant: str
    transport_codec: str
    compression_codec: str
    zero_copy_vram: bool
    metadata: Dict[str, str] = field(default_factory=dict)
    resume_payload: bytes = b""
    resume_position: int = 0
    status: str = "prepared"
    target_worker: str = ""
    active_worker: str = ""
    resume_token: str = ""
    accepted_at_us: int = 0
    error_message: str = ""
    created_at_us: int = field(default_factory=_now_us)
    updated_at_us: int = field(default_factory=_now_us)

    def to_metadata(self) -> Dict[str, str]:
        payload = decode_dopd_resume_payload_v2(self.resume_payload)
        info = {
            "source_role": self.source_role,
            "target_role": self.target_role,
            "phase_role": self.phase_role,
            "model_name": self.model_name,
            "cache_schema": self.cache_schema,
            "kv_variant": self.kv_variant,
            "transport_codec": self.transport_codec,
            "compression_codec": self.compression_codec,
            "zero_copy_vram": "1" if self.zero_copy_vram else "0",
            "status": self.status,
            "target_worker": self.target_worker,
            "active_worker": self.active_worker,
            "resume_position": str(self.resume_position),
            "resume_payload_bytes": str(len(self.resume_payload)),
            "accepted_at_us": str(self.accepted_at_us),
            "created_at_us": str(self.created_at_us),
            "updated_at_us": str(self.updated_at_us),
            "error_message": self.error_message,
        }
        if isinstance(payload, dict):
            info["payload_kind"] = str(payload.get("payload_kind") or "")
            info["payload_integrity_valid"] = "1" if payload.get("integrity_valid") else "0"
        info.update(_stringify_metadata(self.metadata))
        return info


class DOPDWorkerAdapter:
    def __init__(self) -> None:
        self.resume_endpoint = str(os.environ.get("CGC_DOPD_RESUME_ENDPOINT") or "").strip()
        self.timeout_s = float(str(os.environ.get("CGC_DOPD_RESUME_TIMEOUT_S") or "10").strip() or "10")
        self.default_worker_id = str(os.environ.get("CGC_DOPD_WORKER_ID") or "cloud-worker").strip()
        self.contract_context = _build_contract_context()

    def _default_resume_endpoint(self) -> str:
        host = _env_value("CGC_CLOUD_HTTP_HOST")
        port = _env_value("CGC_CLOUD_HTTP_PORT")
        if not port:
            return ""
        resolved_host = host or "127.0.0.1"
        if resolved_host in {"0.0.0.0", "::"}:
            resolved_host = "127.0.0.1"
        return f"http://{resolved_host}:{port}/v1/dopd/resume"

    def submit_resume(
        self,
        *,
        record: DOPDHandoffRecord,
        resume_token: str,
        max_new_tokens: int,
        worker_id: str,
        metadata: Dict[str, str],
        resume_payload: bytes,
    ) -> Dict[str, Any]:
        target_worker = str(worker_id or record.target_worker or self.default_worker_id)
        resume_endpoint = (
            str(os.environ.get("CGC_DOPD_RESUME_ENDPOINT") or "").strip()
            or self.resume_endpoint
            or self._default_resume_endpoint()
        )
        resume_payload_meta = decode_dopd_resume_payload_v2(resume_payload)
        if not resume_endpoint:
            return {
                "success": True,
                "ack_status": "accepted_in_memory",
                "worker_id": target_worker,
                "error_message": "",
            }

        req_payload = {
            "session_id": record.session_id,
            "handoff_id": record.handoff_id,
            "resume_token": resume_token,
            "worker_id": target_worker,
            "phase_role": record.phase_role,
            "resume_position": int(record.resume_position),
            "max_new_tokens": int(max_new_tokens),
            "metadata": _stringify_metadata(metadata),
            "contract_context": dict(self.contract_context),
            "resume_payload_meta": resume_payload_meta if isinstance(resume_payload_meta, dict) else None,
            "resume_payload_b64": base64.b64encode(resume_payload).decode("ascii"),
            "state_bytes_b64": (
                base64.b64encode(extract_dopd_resume_state_bytes(resume_payload_meta)).decode("ascii")
                if isinstance(resume_payload_meta, dict)
                and str(resume_payload_meta.get("state_bytes_b64") or "").strip()
                else ""
            ),
        }
        try:
            resp = urllib.request.urlopen(
                urllib.request.Request(
                    resume_endpoint,
                    data=json.dumps(req_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                ),
                timeout=self.timeout_s,
            )
            raw = resp.read().decode("utf-8", errors="ignore")
            parsed = json.loads(raw) if raw.strip() else {}
            if not isinstance(parsed, dict):
                parsed = {}
            return {
                "success": bool(parsed.get("success", True)),
                "ack_status": str(parsed.get("ack_status") or "forwarded"),
                "worker_id": str(parsed.get("worker_id") or target_worker),
                "error_message": str(parsed.get("error_message") or ""),
            }
        except Exception as exc:
            return {
                "success": False,
                "ack_status": "worker_bridge_failed",
                "worker_id": target_worker,
                "error_message": str(exc),
            }


class DOPDSessionRuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handoffs: Dict[str, DOPDHandoffRecord] = {}
        self._session_to_handoff: Dict[str, str] = {}
        self._worker_adapter = DOPDWorkerAdapter()

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            active = sum(1 for record in self._handoffs.values() if record.status not in {"aborted", "failed"})
            return {
                "dopd_sessions": len(self._session_to_handoff),
                "dopd_handoffs": len(self._handoffs),
                "dopd_active_handoffs": active,
            }

    def prepare_handoff(
        self,
        *,
        session_id: str,
        handoff_id: str,
        source_role: str,
        target_role: str,
        phase_role: str,
        model_name: str,
        cache_schema: str,
        kv_variant: str,
        transport_codec: str,
        compression_codec: str,
        zero_copy_vram: bool,
        resume_payload: bytes,
        metadata: Dict[str, str],
    ) -> DOPDHandoffRecord:
        with self._lock:
            record = self._handoffs.get(handoff_id)
            if record is None:
                record = DOPDHandoffRecord(
                    session_id=session_id,
                    handoff_id=handoff_id,
                    source_role=source_role,
                    target_role=target_role,
                    phase_role=phase_role,
                    model_name=model_name,
                    cache_schema=cache_schema,
                    kv_variant=kv_variant,
                    transport_codec=transport_codec,
                    compression_codec=compression_codec,
                    zero_copy_vram=bool(zero_copy_vram),
                    metadata=_stringify_metadata(metadata),
                    resume_payload=bytes(resume_payload or b""),
                )
                self._handoffs[handoff_id] = record
            else:
                record.source_role = source_role
                record.target_role = target_role
                record.phase_role = phase_role
                record.model_name = model_name
                record.cache_schema = cache_schema
                record.kv_variant = kv_variant
                record.transport_codec = transport_codec
                record.compression_codec = compression_codec
                record.zero_copy_vram = bool(zero_copy_vram)
                record.metadata = _stringify_metadata(metadata)
                if resume_payload:
                    record.resume_payload = bytes(resume_payload)
                record.status = "prepared"
                record.error_message = ""
            record.updated_at_us = _now_us()
            self._session_to_handoff[session_id] = handoff_id
            return record

    def commit_handoff(
        self,
        *,
        session_id: str,
        handoff_id: str,
        target_worker: str,
        resume_position: int,
        resume_payload: bytes,
        metadata: Dict[str, str],
    ) -> DOPDHandoffRecord:
        with self._lock:
            record = self._require_record(session_id=session_id, handoff_id=handoff_id)
            record.target_worker = str(target_worker or record.target_worker)
            record.resume_position = int(resume_position)
            if resume_payload:
                record.resume_payload = bytes(resume_payload)
            if metadata:
                record.metadata.update(_stringify_metadata(metadata))
            seed = f"{record.session_id}:{record.handoff_id}:{record.resume_position}:{record.updated_at_us}".encode("utf-8")
            record.resume_token = hashlib.sha256(seed).hexdigest()[:24]
            record.status = "committed"
            record.updated_at_us = _now_us()
            return record

    def resume_decode(
        self,
        *,
        session_id: str,
        handoff_id: str,
        resume_token: str,
        worker_id: str,
        max_new_tokens: int,
        resume_payload: bytes,
        metadata: Dict[str, str],
    ) -> DOPDHandoffRecord:
        with self._lock:
            record = self._require_record(session_id=session_id, handoff_id=handoff_id)
            if str(resume_token or "") != str(record.resume_token or ""):
                raise ValueError("invalid_resume_token")
            payload = bytes(resume_payload or record.resume_payload or b"")
            adapter_result = self._worker_adapter.submit_resume(
                record=record,
                resume_token=resume_token,
                max_new_tokens=max_new_tokens,
                worker_id=worker_id,
                metadata=_stringify_metadata(metadata),
                resume_payload=payload,
            )
            record.resume_payload = payload
            record.active_worker = str(adapter_result.get("worker_id") or worker_id or record.target_worker)
            record.accepted_at_us = _now_us()
            record.updated_at_us = record.accepted_at_us
            if bool(adapter_result.get("success")):
                record.status = str(adapter_result.get("ack_status") or "resume_accepted")
                record.error_message = ""
            else:
                record.status = "failed"
                record.error_message = str(adapter_result.get("error_message") or "worker_bridge_failed")
            if metadata:
                record.metadata.update(_stringify_metadata(metadata))
            return record

    def abort_handoff(self, *, session_id: str, handoff_id: str, rollback_reason: str) -> DOPDHandoffRecord:
        with self._lock:
            record = self._require_record(session_id=session_id, handoff_id=handoff_id)
            record.status = "aborted"
            record.error_message = str(rollback_reason or "")
            record.updated_at_us = _now_us()
            return record

    def query_session(self, *, session_id: str, handoff_id: str) -> DOPDHandoffRecord:
        with self._lock:
            return self._require_record(session_id=session_id, handoff_id=handoff_id)

    def _require_record(self, *, session_id: str, handoff_id: str) -> DOPDHandoffRecord:
        resolved_handoff = str(handoff_id or self._session_to_handoff.get(session_id) or "")
        if not resolved_handoff:
            raise KeyError("handoff_not_found")
        record = self._handoffs.get(resolved_handoff)
        if record is None:
            raise KeyError("handoff_not_found")
        if session_id and str(record.session_id) != str(session_id):
            raise KeyError("session_handoff_mismatch")
        return record
