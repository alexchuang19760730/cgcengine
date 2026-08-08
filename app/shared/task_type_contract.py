from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

TASK_TYPE_CONTRACT_PATH = (
    Path(__file__).resolve().parent / "contracts" / "task_type_contract.json"
).resolve()

_FALLBACK_CONTRACT = {
    "task_type_contract_version": "task_type_contract_v1",
    "task_type_header": "x-cgc-task-type",
    "default_task_type": "prefill",
    "known_task_types": [
        "prefill",
        "decode",
        "inference",
        "analysis",
        "repo_debug",
        "train",
    ],
    "task_type_aliases": {
        "prefill_only": "prefill",
        "edge_prefill": "prefill",
        "cloud_prefill": "prefill",
        "decode_only": "decode",
        "edge_decode": "decode",
        "cloud_decode": "decode",
        "infer": "inference",
        "serve": "inference",
        "serving": "inference",
        "completion": "inference",
        "chat": "inference",
        "chat_completion": "inference",
        "repo_debugging": "repo_debug",
        "repo-debug": "repo_debug",
        "debug": "repo_debug",
    },
}


def _load_contract_payload() -> dict[str, Any]:
    try:
        payload = json.loads(TASK_TYPE_CONTRACT_PATH.read_text(encoding="utf-8"))
    except Exception:
        payload = dict(_FALLBACK_CONTRACT)
    return payload if isinstance(payload, dict) else dict(_FALLBACK_CONTRACT)


_CONTRACT_PAYLOAD = _load_contract_payload()

TASK_TYPE_CONTRACT_VERSION = str(
    _CONTRACT_PAYLOAD.get("task_type_contract_version") or _FALLBACK_CONTRACT["task_type_contract_version"]
)
CGC_TASK_TYPE_HEADER = str(
    _CONTRACT_PAYLOAD.get("task_type_header") or _FALLBACK_CONTRACT["task_type_header"]
)
KNOWN_TASK_TYPES = tuple(
    str(item).strip()
    for item in (_CONTRACT_PAYLOAD.get("known_task_types") or _FALLBACK_CONTRACT["known_task_types"])
    if str(item).strip()
)
TASK_TYPE_PREFILL = str(_CONTRACT_PAYLOAD.get("default_task_type") or "prefill").strip() or "prefill"
TASK_TYPE_DECODE = "decode"
TASK_TYPE_INFERENCE = "inference"
TASK_TYPE_ANALYSIS = "analysis"
TASK_TYPE_REPO_DEBUG = "repo_debug"
TASK_TYPE_TRAIN = "train"
_TASK_TYPE_ALIASES = {
    str(key).strip(): str(value).strip()
    for key, value in (
        (_CONTRACT_PAYLOAD.get("task_type_aliases") or _FALLBACK_CONTRACT["task_type_aliases"]).items()
        if isinstance(_CONTRACT_PAYLOAD.get("task_type_aliases") or _FALLBACK_CONTRACT["task_type_aliases"], dict)
        else {}
    )
    if str(key).strip() and str(value).strip()
}


def _normalize_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def normalize_task_type(value: Any, *, default: str = "") -> str:
    token = _normalize_token(value)
    if not token:
        return _normalize_token(default)
    canonical = _TASK_TYPE_ALIASES.get(token, token)
    if canonical in KNOWN_TASK_TYPES:
        return canonical
    fallback = _normalize_token(default)
    return fallback if fallback in KNOWN_TASK_TYPES else canonical


def resolve_task_type(*values: Any, default: str = "") -> str:
    for value in values:
        canonical = normalize_task_type(value)
        if canonical:
            return canonical
    return normalize_task_type(default)


def normalize_task_type_iter(values: Iterable[Any]) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        canonical = normalize_task_type(value)
        if canonical:
            normalized.add(canonical)
    return normalized


def task_type_contract_payload() -> dict[str, Any]:
    payload = dict(_CONTRACT_PAYLOAD)
    payload["task_type_contract_path"] = str(TASK_TYPE_CONTRACT_PATH)
    return payload


def task_type_contract_ref() -> dict[str, Any]:
    return {
        "task_type_contract_version": TASK_TYPE_CONTRACT_VERSION,
        "task_type_contract_path": str(TASK_TYPE_CONTRACT_PATH),
        "task_type_header": CGC_TASK_TYPE_HEADER,
        "known_task_types": list(KNOWN_TASK_TYPES),
    }


def normalize_task_type_contract_ref(
    ref: Any,
    *,
    source_path: str = "",
) -> dict[str, Any]:
    payload = ref if isinstance(ref, dict) else {}
    contract_path = str(payload.get("task_type_contract_path") or "").strip()
    if contract_path:
        path_obj = Path(contract_path)
        if not path_obj.is_absolute() and source_path:
            contract_path = str((Path(source_path).resolve().parent / contract_path).resolve())
        else:
            contract_path = str(path_obj.resolve())
    return {
        "task_type_contract_version": str(payload.get("task_type_contract_version") or "").strip(),
        "task_type_contract_path": contract_path,
        "task_type_header": str(payload.get("task_type_header") or "").strip(),
        "known_task_types": [
            str(item).strip()
            for item in (payload.get("known_task_types") or [])
            if str(item).strip()
        ],
    }


def validate_task_type_contract_ref(
    declared_ref: Any,
    *,
    expected_ref: Optional[dict[str, Any]] = None,
    source_path: str = "",
    label: str = "task_type_contract_ref",
) -> dict[str, Any]:
    expected = normalize_task_type_contract_ref(expected_ref or task_type_contract_ref())
    declared = normalize_task_type_contract_ref(declared_ref, source_path=source_path)
    present = any(
        [
            declared.get("task_type_contract_version"),
            declared.get("task_type_contract_path"),
            declared.get("task_type_header"),
            declared.get("known_task_types"),
        ]
    )
    result = {
        "label": label,
        "present": bool(present),
        "valid": False,
        "declared_ref": declared,
        "expected_ref": expected,
        "mismatches": [],
    }
    if not present:
        result["mismatches"] = ["missing_task_type_contract_ref"]
        return result
    mismatches: list[str] = []
    if declared.get("task_type_contract_version") != expected.get("task_type_contract_version"):
        mismatches.append("task_type_contract_version")
    if declared.get("task_type_contract_path") != expected.get("task_type_contract_path"):
        mismatches.append("task_type_contract_path")
    if declared.get("task_type_header") != expected.get("task_type_header"):
        mismatches.append("task_type_header")
    if list(declared.get("known_task_types") or []) != list(expected.get("known_task_types") or []):
        mismatches.append("known_task_types")
    result["mismatches"] = mismatches
    result["valid"] = not mismatches
    return result


def build_task_type_contract_validation_report(
    stages: Iterable[tuple[str, Any, str]],
    *,
    expected_ref: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    expected = normalize_task_type_contract_ref(expected_ref or task_type_contract_ref())
    validations: dict[str, Any] = {}
    status = "PASS"
    for label, declared_ref, source_path in stages:
        validation = validate_task_type_contract_ref(
            declared_ref,
            expected_ref=expected,
            source_path=source_path,
            label=label,
        )
        validations[label] = validation
        if not validation.get("valid"):
            status = "FAIL"
    return {
        "status": status,
        "expected_ref": expected,
        "stages": validations,
    }
