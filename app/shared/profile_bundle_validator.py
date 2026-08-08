from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.shared.task_type_contract import build_task_type_contract_validation_report
from app.shared.task_type_contract import normalize_task_type_contract_ref
from app.shared.task_type_contract import task_type_contract_ref

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_json_file(path: str) -> dict[str, Any]:
    target = str(path or "").strip()
    if not target:
        return {}
    try:
        payload = json.loads(Path(target).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_contract_path(path_str: str, profile_source_path: str = "") -> str:
    path_str = str(path_str or "").strip()
    if not path_str:
        return ""
    path = Path(path_str)
    if path.is_absolute():
        return str(path.resolve())
    candidate_paths = []
    if profile_source_path:
        candidate_paths.append((Path(profile_source_path).resolve().parent / path_str).resolve())
    candidate_paths.append((REPO_ROOT / path_str).resolve())
    candidate_paths.append((REPO_ROOT / "docs" / "technical_whitepapers" / "examples" / path_str).resolve())
    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)
    return str(candidate_paths[0] if candidate_paths else path)


def validate_profile_bundle(
    *,
    profile_settings_path: str,
    system_manifest_path: str = "",
    bootstrap_contract_path: str = "",
) -> dict[str, Any]:
    resolved_profile_path = _resolve_contract_path(profile_settings_path)
    profile_settings = _load_json_file(resolved_profile_path)
    if not profile_settings:
        return {
            "status": "SKIP",
            "reason": "missing_profile_settings",
            "profile_settings_path": resolved_profile_path,
            "task_type_contract_ref": task_type_contract_ref(),
        }

    resolved_bootstrap_path = _resolve_contract_path(
        bootstrap_contract_path or str(profile_settings.get("bootstrap_contract_path") or ""),
        resolved_profile_path,
    )
    bootstrap_contract = _load_json_file(resolved_bootstrap_path)

    system_profile_ref = (
        profile_settings.get("system_profile_ref")
        if isinstance(profile_settings.get("system_profile_ref"), dict)
        else {}
    )
    resolved_system_manifest_path = _resolve_contract_path(
        system_manifest_path or str(system_profile_ref.get("source_path") or ""),
        resolved_profile_path,
    )
    system_manifest = _load_json_file(resolved_system_manifest_path)
    system_profile = (
        system_manifest.get("system_profile")
        if isinstance(system_manifest.get("system_profile"), dict)
        else {}
    )
    profile_binding_ref = (
        system_profile.get("profile_binding_ref")
        if isinstance(system_profile.get("profile_binding_ref"), dict)
        else {}
    )

    if not resolved_system_manifest_path or not resolved_bootstrap_path:
        return {
            "status": "SKIP",
            "reason": "incomplete_bundle_context",
            "profile_settings_path": resolved_profile_path,
            "system_manifest_path": resolved_system_manifest_path,
            "bootstrap_contract_path": resolved_bootstrap_path,
            "task_type_contract_ref": task_type_contract_ref(),
        }

    expected_ref = task_type_contract_ref()
    validation = build_task_type_contract_validation_report(
        [
            (
                "profile_settings.task_type_contract_ref",
                profile_settings.get("task_type_contract_ref"),
                resolved_profile_path,
            ),
            (
                "system_manifest.profile_binding_ref.task_type_contract_ref",
                profile_binding_ref.get("task_type_contract_ref"),
                resolved_system_manifest_path,
            ),
            (
                "bootstrap_contract.task_type_contract_ref",
                bootstrap_contract.get("task_type_contract_ref"),
                resolved_bootstrap_path,
            ),
            (
                "runtime_bootstrap.task_type_contract_ref",
                expected_ref,
                str(expected_ref.get("task_type_contract_path") or ""),
            ),
        ],
        expected_ref=expected_ref,
    )
    return {
        "status": str(validation.get("status") or "FAIL"),
        "profile_settings_path": resolved_profile_path,
        "system_manifest_path": resolved_system_manifest_path,
        "bootstrap_contract_path": resolved_bootstrap_path,
        "task_type_contract_ref": normalize_task_type_contract_ref(expected_ref),
        "task_type_contract_validation": validation,
    }


def validate_profile_bundle_or_raise(
    *,
    profile_settings_path: str,
    system_manifest_path: str = "",
    bootstrap_contract_path: str = "",
) -> dict[str, Any]:
    result = validate_profile_bundle(
        profile_settings_path=profile_settings_path,
        system_manifest_path=system_manifest_path,
        bootstrap_contract_path=bootstrap_contract_path,
    )
    if str(result.get("status") or "").upper() == "FAIL":
        raise ValueError(f"profile_bundle_validation_failed:{result.get('task_type_contract_validation')}")
    return result
