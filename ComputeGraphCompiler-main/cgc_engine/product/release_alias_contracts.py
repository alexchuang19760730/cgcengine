from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent

SYSTEM_EXECUTION_MANIFEST_SCHEMA_KEY = "cgc.system_execution_manifest.v0.1"
SYSTEM_PROFILE_SCHEMA_KEY = "cgc.system_profile.v0.1"
FRESH_HOST1_PROBE_SCHEMA_KEY = "cgc.fresh_host1_probe.v1"
AGENT_EXECUTION_SCHEMA_KEY = "cgc.agent_execution.v1"
DEEPEP_RELEASE_GUARD_SCHEMA_KEY = "cgc.deepep_release_guard.v1"
SWE_VERIFIED_FORMAL_SUMMARY_SCHEMA_KEY = "cgc.swe_verified_formal_summary.inline.v1"
RUNTIME_PROTOCOL_CONTRACT_SCHEMA_KEY = "cgc.runtime_protocol_contract.inline.v1"

AGENT_EXECUTION_SCHEMA_PATH = (
    REPO_ROOT / "docs" / "gate_whitepapers" / "CGC_AGENT_EXECUTION_SCHEMA_v1.0.json"
)
DEEPEP_RELEASE_GUARD_SCHEMA_PATH = (
    REPO_ROOT / "docs" / "gate_whitepapers" / "CGC_DEEPEP_RELEASE_GUARD_SCHEMA_v1.0.json"
)
FRESH_HOST1_PROBE_SCHEMA_PATH = (
    REPO_ROOT / "docs" / "gate_whitepapers" / "CGC_FRESH_HOST1_PROBE_SCHEMA_v1.0.json"
)
REGISTRY_REFERENCE_PATH = REPO_ROOT / "docs" / "registry_reference.md"
FRESH_HOST1_PROBE_PATH = WORKSPACE_ROOT / "temp" / "host1_gateway_post_restart_probe.json"
MODEL_CLI_OUTPUT_ROOT = REPO_ROOT / "Output" / "model_cli"


def _read_json_file(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_if_present(target: Dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and value.strip() == "":
        return
    target[key] = value


def _build_named_ref(
    *,
    source_path: str,
    schema_key: str,
    section: str,
    ref_kind: str = "json_payload_ref",
    source: str = "",
    profile_id: str = "",
    profile_version: str = "",
    binding_key: str = "",
    artifact_name: str = "",
) -> Dict[str, Any]:
    ref: Dict[str, Any] = {
        "ref_kind": ref_kind,
        "source_path": str(source_path),
        "schema_key": str(schema_key),
        "section": str(section),
    }
    _merge_if_present(ref, "source", source)
    _merge_if_present(ref, "profile_id", profile_id)
    _merge_if_present(ref, "profile_version", profile_version)
    _merge_if_present(ref, "binding_key", binding_key)
    _merge_if_present(ref, "artifact_name", artifact_name)
    return ref


def build_alias_schema_refs() -> Dict[str, Dict[str, Any]]:
    return {
        "system_execution_manifest": {
            "schema_key": SYSTEM_EXECUTION_MANIFEST_SCHEMA_KEY,
            "schema_path": str(
                REPO_ROOT / "docs" / "technical_whitepapers" / "CGC_System_Execution_Manifest_Schema_v0.1.json"
            ),
        },
        "system_profile": {
            "schema_key": SYSTEM_PROFILE_SCHEMA_KEY,
            "schema_path": str(
                REPO_ROOT / "docs" / "technical_whitepapers" / "CGC_System_Profile_Schema_v0.1.json"
            ),
        },
        "fresh_host1_probe": {
            "schema_key": FRESH_HOST1_PROBE_SCHEMA_KEY,
            "schema_path": str(FRESH_HOST1_PROBE_SCHEMA_PATH),
        },
        "agent_execution": {
            "schema_key": AGENT_EXECUTION_SCHEMA_KEY,
            "schema_path": str(AGENT_EXECUTION_SCHEMA_PATH),
        },
        "deepep_release_guard": {
            "schema_key": DEEPEP_RELEASE_GUARD_SCHEMA_KEY,
            "schema_path": str(DEEPEP_RELEASE_GUARD_SCHEMA_PATH),
        },
    }


def _formal_payload(manifest_payload: Dict[str, Any], evidence_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    formal_evidence = manifest_payload.get("formal_evidence") if isinstance(manifest_payload.get("formal_evidence"), dict) else {}
    entry = formal_evidence.get(evidence_name) if isinstance(formal_evidence.get(evidence_name), dict) else {}
    payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
    return entry, payload


def _build_system_profile_ref(manifest_payload: Dict[str, Any], manifest_path: Path | None) -> Dict[str, Any]:
    system_profile = manifest_payload.get("system_profile") if isinstance(manifest_payload.get("system_profile"), dict) else {}
    return _build_named_ref(
        source_path=str(manifest_path or ""),
        schema_key=SYSTEM_PROFILE_SCHEMA_KEY,
        section="system_profile",
        source="system_execution_manifest",
        profile_id=str(system_profile.get("profile_id") or ""),
        profile_version=str(system_profile.get("profile_version") or system_profile.get("schema_version") or ""),
        artifact_name="system_profile",
    )


def _latest_swe_verified_remote_summary() -> Tuple[Path | None, Dict[str, Any], Path | None]:
    try:
        session_paths = sorted(
            MODEL_CLI_OUTPUT_ROOT.glob("**/model_swe_verified_session.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        session_paths = []
    for session_path in session_paths:
        session_payload = _read_json_file(session_path)
        artifacts = session_payload.get("artifacts") if isinstance(session_payload.get("artifacts"), dict) else {}
        remote_score_summary_raw = (
            session_payload.get("remote_score_summary_path")
            or artifacts.get("remote_swebench_score_summary")
            or ""
        )
        remote_score_summary_path = Path(str(remote_score_summary_raw)).expanduser()
        if not remote_score_summary_path.is_file():
            continue
        remote_summary = _read_json_file(remote_score_summary_path)
        if remote_summary:
            return remote_score_summary_path.resolve(), remote_summary, session_path.resolve()
    try:
        summary_paths = sorted(
            MODEL_CLI_OUTPUT_ROOT.glob("**/remote_swebench_score_summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        summary_paths = []
    for summary_path in summary_paths:
        remote_summary = _read_json_file(summary_path)
        if remote_summary:
            return summary_path.resolve(), remote_summary, None
    return None, {}, None


def _extract_fresh_host1_probe_payload() -> Tuple[Path | None, Dict[str, Any], Dict[str, Any]]:
    source_path = FRESH_HOST1_PROBE_PATH if FRESH_HOST1_PROBE_PATH.is_file() else None
    wrapper = _read_json_file(source_path)
    health = wrapper.get("health") if isinstance(wrapper.get("health"), dict) else {}
    stdout = health.get("stdout")
    payload: Dict[str, Any] = {}
    if isinstance(stdout, str) and stdout.strip():
        try:
            parsed = json.loads(stdout)
        except Exception:
            parsed = {}
        if isinstance(parsed, dict):
            payload = parsed
    elif isinstance(stdout, dict):
        payload = stdout
    return source_path, wrapper, payload


def extract_fresh_host1_probe_payload() -> Tuple[Path | None, Dict[str, Any], Dict[str, Any]]:
    return _extract_fresh_host1_probe_payload()


def promote_runtime_protocol_contract_with_fresh_probe(
    runtime_protocol_contract: Dict[str, Any] | None,
) -> Dict[str, Any]:
    promoted = dict(runtime_protocol_contract or {})
    source_path, _, probe_payload = _extract_fresh_host1_probe_payload()
    if not probe_payload:
        return promoted

    moe_a2a_backend = str(probe_payload.get("moe_a2a_backend") or "").strip().lower()
    if moe_a2a_backend:
        promoted["requested_dispatch_backend"] = "deepep" if moe_a2a_backend == "deepep" else moe_a2a_backend
        promoted["declared_deepep_backend"] = moe_a2a_backend == "deepep"
        promoted["moe_a2a_backend"] = moe_a2a_backend
    if str(probe_payload.get("pd_mode") or "").strip():
        promoted["pd_mode"] = str(probe_payload.get("pd_mode") or "")
    if str(probe_payload.get("profile_settings_path") or "").strip():
        promoted["profile_settings_path"] = str(probe_payload.get("profile_settings_path") or "")
    if str(probe_payload.get("execution_profile_binding_key") or "").strip():
        promoted["execution_profile_binding_key"] = str(probe_payload.get("execution_profile_binding_key") or "")
    if str(probe_payload.get("bootstrap_contract_binding_key") or "").strip():
        promoted["bootstrap_contract_binding_key"] = str(probe_payload.get("bootstrap_contract_binding_key") or "")
    if str(probe_payload.get("system_profile_id") or "").strip():
        promoted["system_profile_id"] = str(probe_payload.get("system_profile_id") or "")
    if str(probe_payload.get("model_contract_id") or "").strip():
        promoted["model_contract_id"] = str(probe_payload.get("model_contract_id") or "")
    promoted["freshness_source"] = "fresh_host1_probe"
    promoted["freshness_source_path"] = str(source_path or "")
    return promoted


def build_fresh_host1_real_chain_payload(
    runtime_protocol_contract: Dict[str, Any] | None,
) -> Dict[str, Any]:
    source_path, wrapper, probe_payload = _extract_fresh_host1_probe_payload()
    promoted_runtime_protocol_contract = promote_runtime_protocol_contract_with_fresh_probe(
        runtime_protocol_contract
    )
    health = wrapper.get("health") if isinstance(wrapper.get("health"), dict) else {}
    task_type_validation = (
        probe_payload.get("task_type_contract_validation")
        if isinstance(probe_payload.get("task_type_contract_validation"), dict)
        else {}
    )
    backend_field = str(probe_payload.get("backend") or "")
    backend_available = bool(backend_field) and "down:" not in backend_field.lower()
    freshness_pass = bool(
        str(probe_payload.get("moe_a2a_backend") or "").strip().lower() == "deepep"
        and str(task_type_validation.get("status") or "").strip().upper() == "PASS"
    )
    return {
        "status": "PASS" if freshness_pass else "BLOCKED" if probe_payload else "MISSING",
        "mode": "fresh_host1_real_chain_promote",
        "runtime_protocol_contract": promoted_runtime_protocol_contract,
        "fresh_host1_probe_path": str(source_path or ""),
        "fresh_host1_probe_health_rc": health.get("rc"),
        "backend_available": backend_available,
        "backend_field": backend_field,
        "freshness_guard": {
            "status": "PASS" if freshness_pass else "BLOCKED" if probe_payload else "MISSING",
            "moe_a2a_backend": str(probe_payload.get("moe_a2a_backend") or ""),
            "task_type_contract_validation_status": str(task_type_validation.get("status") or ""),
            "profile_settings_path": str(probe_payload.get("profile_settings_path") or ""),
        },
    }


def _build_profile_settings_ref_from_probe(source_path: Path | None, probe_payload: Dict[str, Any]) -> Dict[str, Any]:
    return _build_named_ref(
        source_path=str(source_path or ""),
        schema_key=FRESH_HOST1_PROBE_SCHEMA_KEY,
        section="health.stdout",
        source="fresh_host1_probe",
        profile_id=str(probe_payload.get("system_profile_id") or ""),
        binding_key=str(probe_payload.get("execution_profile_binding_key") or ""),
        artifact_name="profile_settings",
    ) | {
        "profile_settings_path": str(probe_payload.get("profile_settings_path") or ""),
        "bootstrap_contract_binding_key": str(probe_payload.get("bootstrap_contract_binding_key") or ""),
    }


def _build_runtime_protocol_contract_ref(
    manifest_payload: Dict[str, Any],
    manifest_path: Path | None,
) -> Dict[str, Any]:
    entry, payload = _formal_payload(manifest_payload, "fresh_host1_real_chain")
    runtime_contract = payload.get("runtime_protocol_contract") if isinstance(payload.get("runtime_protocol_contract"), dict) else {}
    if runtime_contract:
        return _build_named_ref(
            source_path=str(entry.get("path") or manifest_path or ""),
            schema_key=RUNTIME_PROTOCOL_CONTRACT_SCHEMA_KEY,
            section="runtime_protocol_contract",
            source=str(entry.get("source") or "fresh_host1_real_chain"),
            profile_id=str(runtime_contract.get("system_profile_id") or ""),
            artifact_name="runtime_protocol_contract",
        )
    entry, payload = _formal_payload(manifest_payload, "m76_remote_runtime")
    runtime_contract = payload.get("runtime_protocol_contract") if isinstance(payload.get("runtime_protocol_contract"), dict) else {}
    if runtime_contract:
        return _build_named_ref(
            source_path=str(entry.get("path") or manifest_path or ""),
            schema_key=RUNTIME_PROTOCOL_CONTRACT_SCHEMA_KEY,
            section="runtime_protocol_contract",
            source=str(entry.get("source") or "m76_remote_runtime"),
            profile_id=str(runtime_contract.get("system_profile_id") or ""),
            artifact_name="runtime_protocol_contract",
        )
    entry, payload = _formal_payload(manifest_payload, "m75_active_runtime")
    runtime_contract = payload.get("runtime_protocol_contract") if isinstance(payload.get("runtime_protocol_contract"), dict) else {}
    return _build_named_ref(
        source_path=str(entry.get("path") or manifest_path or ""),
        schema_key=RUNTIME_PROTOCOL_CONTRACT_SCHEMA_KEY,
        section="runtime_protocol_contract",
        source=str(entry.get("source") or "m75_active_runtime"),
        profile_id=str(runtime_contract.get("system_profile_id") or ""),
        artifact_name="runtime_protocol_contract",
    )


def build_agent_execution_alias(
    manifest_payload: Dict[str, Any],
    manifest_path: Path | None,
) -> Dict[str, Any]:
    entry, payload = _formal_payload(manifest_payload, "swe_verified_formal_summary")
    remote_summary_path, remote_summary_payload, session_path = _latest_swe_verified_remote_summary()
    payload_source = "formal_summary"
    if not payload and remote_summary_payload:
        payload = dict(remote_summary_payload)
        payload_source = "remote_score_summary"
    total_tasks = int(
        payload.get("total_tasks")
        or payload.get("task_count")
        or payload.get("total_count")
        or payload.get("trajectory_count")
        or 0
    )
    submitted_tasks = int(payload.get("submitted_tasks") or payload.get("submitted_count") or 0)
    passed_tasks = int(payload.get("passed_tasks") or payload.get("passed_count") or 0)
    resolved_instances = int(payload.get("resolved_instances") or payload.get("resolved_count") or passed_tasks or 0)
    resolution_rate = float(
        payload.get("resolution_rate")
        or payload.get("pass_rate")
        or ((((payload.get("score") or {}) if isinstance(payload.get("score"), dict) else {}).get("resolve_rate_estimate")) or 0.0)
    )
    raw_status = (
        ""
        if payload_source == "remote_score_summary"
        else str(payload.get("status") or payload.get("swe_verified_status") or "").strip()
    )
    if raw_status:
        status = raw_status
    elif payload_source == "remote_score_summary":
        status = "PASSED" if passed_tasks > 0 else "SUBMITTED"
    else:
        status = "MISSING" if not payload else "PARTIAL"
    result_semantics = str(
        payload.get("result_semantics")
        or raw_status
        or (
            "remote_score_summary_promoted"
            if payload_source == "remote_score_summary"
            else ("formal_summary_missing" if not payload else "summary_present")
        )
    )
    artifact_index_path = str(payload.get("artifact_index_path") or remote_summary_path or entry.get("path") or "")
    formal_summary_source_path = str(entry.get("path") or manifest_path or "")
    formal_summary_section = "payload"
    formal_summary_source = str(entry.get("source") or "system_execution_manifest")
    if payload_source == "remote_score_summary":
        formal_summary_source_path = str(remote_summary_path or session_path or manifest_path or "")
        formal_summary_section = "root"
        formal_summary_source = "remote_swebench_score_summary"
    return {
        "schema_version": AGENT_EXECUTION_SCHEMA_KEY,
        "status": status,
        "suite_name": str(payload.get("suite_name") or payload.get("suite") or payload.get("benchmark_name") or "swe_verified_500"),
        "total_tasks": total_tasks,
        "submitted_tasks": submitted_tasks,
        "passed_tasks": passed_tasks,
        "resolved_instances": resolved_instances,
        "resolution_rate": resolution_rate,
        "result_semantics": result_semantics,
        "artifact_index_path": artifact_index_path,
        "refs": {
            "formal_summary_ref": _build_named_ref(
                source_path=formal_summary_source_path,
                schema_key=SWE_VERIFIED_FORMAL_SUMMARY_SCHEMA_KEY,
                section=formal_summary_section,
                source=formal_summary_source,
                artifact_name="swe_verified_formal_summary",
            ),
            "remote_score_summary_ref": _build_named_ref(
                source_path=str(remote_summary_path or ""),
                schema_key=SWE_VERIFIED_FORMAL_SUMMARY_SCHEMA_KEY,
                section="root",
                source="remote_swebench_score_summary",
                artifact_name="remote_swebench_score_summary",
            ),
            "session_ref": _build_named_ref(
                source_path=str(session_path or ""),
                schema_key=SYSTEM_EXECUTION_MANIFEST_SCHEMA_KEY,
                section="runtime",
                source="model_swe_verified_session",
                artifact_name="model_swe_verified_session",
            ),
            "system_profile_ref": _build_system_profile_ref(manifest_payload, manifest_path),
            "manifest_ref": _build_named_ref(
                source_path=str(manifest_path or ""),
                schema_key=SYSTEM_EXECUTION_MANIFEST_SCHEMA_KEY,
                section="formal_evidence.swe_verified_formal_summary",
                source="system_execution_manifest",
                artifact_name="system_execution_manifest",
            ),
        },
    }


def build_deepep_release_guard_alias(
    manifest_payload: Dict[str, Any],
    manifest_path: Path | None,
    *,
    gate_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    gate_payload = gate_payload if isinstance(gate_payload, dict) else {}
    _, fresh_runtime_payload = _formal_payload(manifest_payload, "fresh_host1_real_chain")
    runtime_protocol_contract = (
        fresh_runtime_payload.get("runtime_protocol_contract")
        if isinstance(fresh_runtime_payload.get("runtime_protocol_contract"), dict)
        else {}
    )
    _, m76_runtime_payload = _formal_payload(manifest_payload, "m76_remote_runtime")
    if not runtime_protocol_contract:
        runtime_protocol_contract = (
            m76_runtime_payload.get("runtime_protocol_contract")
            if isinstance(m76_runtime_payload.get("runtime_protocol_contract"), dict)
            else {}
        )
    if not runtime_protocol_contract:
        _, m75_runtime_payload = _formal_payload(manifest_payload, "m75_active_runtime")
        runtime_protocol_contract = (
            m75_runtime_payload.get("runtime_protocol_contract")
            if isinstance(m75_runtime_payload.get("runtime_protocol_contract"), dict)
            else {}
        )
    runtime_protocol_contract = promote_runtime_protocol_contract_with_fresh_probe(runtime_protocol_contract)
    source_path, _, fresh_probe_payload = _extract_fresh_host1_probe_payload()
    task_type_validation = (
        fresh_probe_payload.get("task_type_contract_validation")
        if isinstance(fresh_probe_payload.get("task_type_contract_validation"), dict)
        else {}
    )
    backend_field = str(fresh_probe_payload.get("backend") or "")
    backend_available = bool(backend_field) and "down:" not in backend_field.lower()
    freshness_status = "PASS" if (
        str(fresh_probe_payload.get("moe_a2a_backend") or "").strip().lower() == "deepep"
        and str(task_type_validation.get("status") or "").strip().upper() == "PASS"
    ) else ("MISSING" if not fresh_probe_payload else "BLOCKED")
    deepep_gate = gate_payload.get("checks", {}).get("deepep_real_chain_gate") if isinstance(gate_payload.get("checks"), dict) else {}
    deepep_gate_status = str((deepep_gate or {}).get("status") or "")
    if deepep_gate_status == "PASS" and freshness_status == "PASS" and backend_available:
        release_claim_status = "PASS"
    elif fresh_probe_payload or runtime_protocol_contract or deepep_gate_status:
        release_claim_status = "BLOCKED"
    else:
        release_claim_status = "MISSING"
    return {
        "schema_version": DEEPEP_RELEASE_GUARD_SCHEMA_KEY,
        "release_claim_status": release_claim_status,
        "freshness_guard": {
            "status": freshness_status,
            "moe_a2a_backend": str(fresh_probe_payload.get("moe_a2a_backend") or ""),
            "backend_available": backend_available,
            "backend_field": backend_field,
            "task_type_contract_validation_status": str(task_type_validation.get("status") or ""),
            "profile_settings_path": str(fresh_probe_payload.get("profile_settings_path") or ""),
        },
        "runtime_protocol_contract": {
            "requested_dispatch_backend": str(runtime_protocol_contract.get("requested_dispatch_backend") or ""),
            "distributed_runtime_backend": str(runtime_protocol_contract.get("distributed_runtime_backend") or ""),
            "service_topology_backend": str(runtime_protocol_contract.get("service_topology_backend") or ""),
            "pd_mode": str(runtime_protocol_contract.get("pd_mode") or ""),
        },
        "deepep_real_chain_gate": {
            "status": deepep_gate_status,
            "reason": str((deepep_gate or {}).get("reason") or ""),
            "requested_dispatch_backend": str((deepep_gate or {}).get("requested_dispatch_backend") or runtime_protocol_contract.get("requested_dispatch_backend") or ""),
        },
        "refs": {
            "fresh_host1_probe_ref": _build_named_ref(
                source_path=str(source_path or ""),
                schema_key=FRESH_HOST1_PROBE_SCHEMA_KEY,
                section="health.stdout",
                source="fresh_host1_probe",
                profile_id=str(fresh_probe_payload.get("system_profile_id") or ""),
                binding_key=str(fresh_probe_payload.get("bootstrap_contract_binding_key") or ""),
                artifact_name="fresh_host1_probe",
            ),
            "profile_settings_ref": _build_profile_settings_ref_from_probe(source_path, fresh_probe_payload),
            "runtime_protocol_contract_ref": _build_runtime_protocol_contract_ref(manifest_payload, manifest_path),
            "system_profile_ref": _build_system_profile_ref(manifest_payload, manifest_path),
        },
    }


def apply_release_alias_contracts(
    manifest_payload: Dict[str, Any],
    manifest_path: Path | None,
    *,
    gate_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = dict(manifest_payload or {})
    schema_refs = payload.get("schema_refs") if isinstance(payload.get("schema_refs"), dict) else {}
    schema_refs.update(build_alias_schema_refs())
    payload["schema_refs"] = schema_refs
    payload["agent_execution"] = build_agent_execution_alias(payload, manifest_path)
    payload["deepep_release_guard"] = build_deepep_release_guard_alias(
        payload,
        manifest_path,
        gate_payload=gate_payload,
    )
    return payload
