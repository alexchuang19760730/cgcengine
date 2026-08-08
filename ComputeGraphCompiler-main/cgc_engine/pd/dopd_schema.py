"""
DOPD resume payload schema helpers.

This module keeps the existing PD payload path intact while adding a
State-ABI-aligned resume payload format that can be stored inside the
existing prefix cache / transport channel.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Mapping, Optional


DOPD_RESUME_PAYLOAD_MAGIC_V2 = b"CGC_DOPD_RESUME_V2\x00"


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _to_plain_dict(payload: Any) -> Dict[str, Any]:
    if is_dataclass(payload):
        data = asdict(payload)
    elif isinstance(payload, Mapping):
        data = dict(payload)
    else:
        raise TypeError("payload must be a mapping or DOPDResumePayloadV2")
    return data


def compute_dopd_resume_checksum(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("integrity_checksum", None)
    return hashlib.sha256(_canonical_json(body)).hexdigest()


@dataclass
class DOPDResumePayloadV2:
    session_id: str
    handoff_id: str
    phase_role: str
    cache_schema: str
    kv_variant: str
    model_name: str = ""
    abi_descriptor: Dict[str, Any] = field(default_factory=dict)
    layout_meta: Dict[str, Any] = field(default_factory=dict)
    prefix_state_ref: str = ""
    kv_state_ref: str = ""
    kda_state_ref: str = ""
    resume_position: int = 0
    token_position: int = 0
    # Layer-granularity ABI (Gate 2.0 stage 1): enable by-layer edge/cloud split
    finished_layer: int = 0
    max_local_layer: int = 0
    hidden_states_ref: str = ""
    partial_kv_ref: str = ""
    layer_quant_state: Dict[str, Any] = field(default_factory=dict)
    prefill_done: bool = True
    decode_resume: bool = True
    transport_codec: str = "cq4"
    compression_codec: str = "trueorthokda"
    zero_copy_vram: bool = True
    state_bytes_b64: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)
    version: int = 2
    payload_kind: str = "dopd_resume"
    integrity_checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["integrity_checksum"] = compute_dopd_resume_checksum(data)
        return data


def normalize_dopd_resume_payload_v2(payload: Any) -> Dict[str, Any]:
    data = _to_plain_dict(payload)
    data.setdefault("version", 2)
    data.setdefault("payload_kind", "dopd_resume")
    data.setdefault("model_name", "")
    data.setdefault("abi_descriptor", {})
    data.setdefault("layout_meta", {})
    data.setdefault("prefix_state_ref", "")
    data.setdefault("kv_state_ref", "")
    data.setdefault("kda_state_ref", "")
    data.setdefault("resume_position", 0)
    data.setdefault("token_position", 0)
    data.setdefault("finished_layer", 0)
    data.setdefault("max_local_layer", 0)
    data.setdefault("hidden_states_ref", "")
    data.setdefault("partial_kv_ref", "")
    data.setdefault("layer_quant_state", {})
    data.setdefault("prefill_done", True)
    data.setdefault("decode_resume", True)
    data.setdefault("transport_codec", "cq4")
    data.setdefault("compression_codec", "trueorthokda")
    data.setdefault("zero_copy_vram", True)
    data.setdefault("state_bytes_b64", "")
    data.setdefault("metadata", {})
    required = [
        "session_id",
        "handoff_id",
        "phase_role",
        "cache_schema",
        "kv_variant",
    ]
    missing = [key for key in required if not str(data.get(key) or "").strip()]
    if missing:
        raise ValueError(f"missing required DOPD payload fields: {', '.join(missing)}")
    data["integrity_checksum"] = compute_dopd_resume_checksum(data)
    return data


def encode_dopd_resume_payload_v2(payload: Any) -> bytes:
    normalized = normalize_dopd_resume_payload_v2(payload)
    return DOPD_RESUME_PAYLOAD_MAGIC_V2 + _canonical_json(normalized)


def decode_dopd_resume_payload_v2(data: bytes) -> Optional[Dict[str, Any]]:
    if not isinstance(data, (bytes, bytearray)):
        return None
    raw = bytes(data)
    if not raw.startswith(DOPD_RESUME_PAYLOAD_MAGIC_V2):
        return None
    body = raw[len(DOPD_RESUME_PAYLOAD_MAGIC_V2) :]
    try:
        payload = json.loads(body.decode("utf-8")) if body else None
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    expected = compute_dopd_resume_checksum(payload)
    observed = str(payload.get("integrity_checksum") or "")
    payload["integrity_valid"] = bool(observed) and observed == expected
    return payload


def extract_dopd_resume_state_bytes(payload: Mapping[str, Any]) -> bytes:
    if not isinstance(payload, Mapping):
        return b""
    encoded = str(payload.get("state_bytes_b64") or "").strip()
    if not encoded:
        return b""
    return base64.b64decode(encoded.encode("ascii"), validate=True)
