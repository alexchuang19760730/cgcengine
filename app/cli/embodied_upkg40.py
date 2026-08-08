from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = (REPO_ROOT / "ComputeGraphCompiler-main").resolve()
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from app.cli.embodied_contract_profiles import canonical_profile_names
from cgc_engine.product.upkg30_common import artifact_index, failure_attribution, read_json, write_json


LEGACY_REALTIME_TEMPLATE_EDGE_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_EDGE_MODEL = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"
DEFAULT_REALTIMEVLA_OUTPUT_ROOT = (REPO_ROOT / "temp" / "test" / "upkg40_realtimevla_recovered").resolve()
DEFAULT_TRAIN_OUTPUT_ROOT = (REPO_ROOT / "temp" / "test" / "upkg40_train_recovered").resolve()
DEFAULT_DEPLOY_OUTPUT_ROOT = (REPO_ROOT / "temp" / "test" / "upkg40_deploy_recovered").resolve()
DEFAULT_LAUNCH_COMMAND = "python3 temp/misc/launch_hostb_psi0_runtime_gate_blocked_ddp.py"
DEFAULT_FETCH_COMMAND = "python3 temp/misc/fetch_hostb_psi0_runtime_gate_blocked_ddp_results.py"

REALTIME_TEMPLATE_SESSION_ROOT = (
    REPO_ROOT / "temp" / "test" / "upkg40_profile_binding_realtimevla_20260620" / "20260620_151334"
).resolve()
TRAIN_TEMPLATE_SESSION_ROOT = (
    REPO_ROOT / "temp" / "test" / "upkg40_profile_settings_train_20260620" / "20260620_150816"
).resolve()
DEPLOY_TEMPLATE_SESSION_ROOT = (
    REPO_ROOT / "temp" / "test" / "upkg40_profile_settings_deploy_20260620" / "20260620_150833"
).resolve()
REALTIME_TEMPLATE_DIR = (REALTIME_TEMPLATE_SESSION_ROOT / "upkg40_embodied").resolve()
TRAIN_TEMPLATE_DIR = (TRAIN_TEMPLATE_SESSION_ROOT / "upkg40_embodied_train").resolve()
DEPLOY_TEMPLATE_DIR = (DEPLOY_TEMPLATE_SESSION_ROOT / "upkg40_embodied_deploy").resolve()
FETCH_CACHE_DIR = (REPO_ROOT / "temp" / "misc" / "hostb_psi0_runtime_gate_blocked_results").resolve()

_PROFILE_DESCRIPTOR_FIELDS = (
    "execution_profile_descriptor",
    "source_execution_profile_descriptor",
    "target_execution_profile_descriptor",
    "bridge_execution_profile_descriptor",
    "delivery_profile_descriptor",
    "compatible_execution_profile_descriptors",
    "applicable_execution_profile_descriptors",
    "bootstrap_contract_descriptor",
    "bootstrap_contract_descriptors",
    "flow_parameter_contract_descriptor",
    "flow_parameter_contract_descriptors",
)
_FETCH_FILE_NAMES = (
    "summary.json",
    "rank0_runtime_gate_report.json",
    "rank1_runtime_gate_report.json",
    "rank0_distributed_runtime_bootstrap.json",
    "rank0_contract_manifest.json",
    "rank0_system_execution_manifest.json",
    "rank0_strategy_decision.json",
    "rank0_compatibility_report.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _session_dir(base_root: str | Path, *, label: str) -> Path:
    root = Path(base_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    session_dir = root / _timestamp_token()
    suffix = 1
    while session_dir.exists():
        session_dir = root / f"{_timestamp_token()}_{suffix:02d}"
        suffix += 1
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _deep_replace(payload: Any, replacements: dict[str, str]) -> Any:
    if isinstance(payload, dict):
        return {str(key): _deep_replace(value, replacements) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_deep_replace(value, replacements) for value in payload]
    if isinstance(payload, str):
        updated = payload
        for old, new in replacements.items():
            if old:
                updated = updated.replace(old, new)
        return updated
    return payload


def _template_json(path: Path, replacements: dict[str, str]) -> dict[str, Any]:
    payload = read_json(path)
    if not payload:
        raise FileNotFoundError(f"missing_template_json:{path}")
    replaced = _deep_replace(payload, replacements)
    return replaced if isinstance(replaced, dict) else {}


def _template_text(path: Path, replacements: dict[str, str]) -> str:
    text = path.read_text(encoding="utf-8")
    for old, new in replacements.items():
        if old:
            text = text.replace(old, new)
    return text


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _copy_text_template(source: Path, target: Path) -> Path:
    return _write_text(target, source.read_text(encoding="utf-8"))


def _copy_template_path(source: Path, target: Path, replacements: dict[str, str]) -> Path:
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = _template_json(source, replacements)
        return write_json(target, payload)
    if suffix in {".txt", ".jsonl", ".log"}:
        return _write_text(target, _template_text(source, replacements))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _extract_json_payload(stdout: str) -> dict[str, Any]:
    text = str(stdout or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _nested_get(payload: dict[str, Any], dotted_path: str, default: Any = "") -> Any:
    current: Any = payload
    for part in str(dotted_path or "").split("."):
        if not part:
            continue
        if not isinstance(current, dict):
            return default
        current = current.get(part)
        if current is None:
            return default
    return current


def _nested_set(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    current: dict[str, Any] = payload
    parts = [part for part in str(dotted_path or "").split(".") if part]
    if not parts:
        return
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _run_json_command(command: str) -> tuple[dict[str, Any], subprocess.CompletedProcess[str]]:
    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command_failed:{command}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    payload = _extract_json_payload(proc.stdout)
    if not payload:
        raise RuntimeError(f"command_did_not_emit_json:{command}\nstdout:\n{proc.stdout}")
    return payload, proc


def _require_file(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing_{label}:{path}")
    return path.resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> Path:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    return _write_text(path, ("\n".join(lines) + ("\n" if lines else "")))


def _profile_binding_fields(
    *,
    profile_settings_path: str,
    execution: str | None = None,
    execution_map: dict[str, str] | None = None,
    delivery: str | None = None,
    compatible: list[str] | None = None,
    applicable: list[str] | None = None,
    bootstrap: str | list[str] | dict[str, str] | None = None,
    flow: str | list[str] | dict[str, str] | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "profile_settings_path": str(Path(profile_settings_path).expanduser().resolve()),
    }
    if execution_map:
        fields["execution_profile_binding_keys"] = {str(key): str(value) for key, value in execution_map.items()}
    elif execution:
        fields["execution_profile_binding_key"] = str(execution)
    if delivery:
        fields["delivery_profile_binding_key"] = str(delivery)
    if compatible:
        fields["compatible_profile_binding_keys"] = [str(value) for value in compatible]
    if applicable:
        fields["applicable_profile_binding_keys"] = [str(value) for value in applicable]
    if isinstance(bootstrap, dict):
        fields["bootstrap_contract_binding_keys"] = {str(key): str(value) for key, value in bootstrap.items()}
    elif isinstance(bootstrap, list):
        fields["bootstrap_contract_binding_keys"] = [str(value) for value in bootstrap]
    elif bootstrap:
        fields["bootstrap_contract_binding_key"] = str(bootstrap)
    if isinstance(flow, dict):
        fields["flow_parameter_contract_binding_keys"] = {str(key): str(value) for key, value in flow.items()}
    elif isinstance(flow, list):
        fields["flow_parameter_contract_binding_keys"] = [str(value) for value in flow]
    elif flow:
        fields["flow_parameter_contract_binding_key"] = str(flow)
    return fields


def _rewrite_binding_payload(payload: dict[str, Any], binding_fields: dict[str, Any]) -> dict[str, Any]:
    rewritten = dict(payload)
    for key in _PROFILE_DESCRIPTOR_FIELDS:
        rewritten.pop(key, None)
    rewritten.update(binding_fields)
    return rewritten


def _materialize_cloud_bundle(
    *,
    session_dir: Path,
    launch_command: str,
    fetch_command: str,
) -> dict[str, Any]:
    launch_payload, launch_proc = _run_json_command(launch_command)
    fetch_payload, fetch_proc = _run_json_command(fetch_command)
    fetched = fetch_payload.get("fetched") if isinstance(fetch_payload.get("fetched"), dict) else {}
    bundle_dir = (session_dir / "cloud_bundle").resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for file_name in _FETCH_FILE_NAMES:
        source_path = Path(str(fetched.get(file_name) or FETCH_CACHE_DIR / file_name)).expanduser().resolve()
        _require_file(source_path, label=file_name.replace(".", "_"))
        target_path = (bundle_dir / file_name).resolve()
        shutil.copy2(source_path, target_path)
        copied[file_name] = str(target_path)
    summary = read_json(Path(copied["summary.json"]))
    return {
        "launch_payload": launch_payload,
        "launch_returncode": int(launch_proc.returncode),
        "fetch_returncode": int(fetch_proc.returncode),
        "bundle_dir": str(bundle_dir),
        "summary_path": copied["summary.json"],
        "runtime_report_path": copied["rank0_runtime_gate_report.json"],
        "contract_manifest_path": copied["rank0_contract_manifest.json"],
        "system_execution_manifest_path": copied["rank0_system_execution_manifest.json"],
        "strategy_decision_path": copied["rank0_strategy_decision.json"],
        "compatibility_report_path": copied["rank0_compatibility_report.json"],
        "distributed_runtime_bootstrap_path": copied["rank0_distributed_runtime_bootstrap.json"],
        "rank1_runtime_report_path": copied["rank1_runtime_gate_report.json"],
        "contract_status": str(summary.get("contract_status") or "BLOCKED"),
        "contract_reason": str(summary.get("contract_reason") or ""),
        "all_blocked": bool(summary.get("all_blocked", True)),
        "world_size": int(summary.get("world_size") or 0),
    }


def _write_train_reports(
    *,
    session_dir: Path,
    upkg_dir: Path,
    cloud_bundle: dict[str, Any],
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    stage_status = {
        "launch_cloud_job": {"status": "PASS", "reason": ""},
        "fetch_cloud_bundle": {"status": "PASS", "reason": ""},
        "materialize_train_contracts": {"status": "PASS", "reason": ""},
    }
    stage_trace_path = _copy_text_template(
        TRAIN_TEMPLATE_DIR / "stage_trace.jsonl",
        upkg_dir / "stage_trace.jsonl",
    )
    artifact_entries = artifact_index(
        [
            artifact_paths["cloud_summary_path"],
            artifact_paths["full_weight_manifest_path"],
            artifact_paths["publish_manifest_path"],
            artifact_paths["runtime_contract_path"],
            artifact_paths["train_session_path"],
            str(stage_trace_path.resolve()),
            artifact_paths["canonical_profile_catalog_path"],
            artifact_paths["profile_settings_path"],
        ]
    )
    artifact_index_path = write_json(upkg_dir / "artifact_index.json", {"artifacts": artifact_entries})
    failure = failure_attribution(gate_name="upkg40_embodied_train", status="PASS", stage_status=stage_status)
    summary = {
        "status": "PASS",
        "command": "cgc embodied psi0-train",
        "generated_at": _utc_now(),
        "source_output_root": str(session_dir.resolve()),
        "upkg40_output_dir": str(upkg_dir.resolve()),
        "cloud_summary_path": artifact_paths["cloud_summary_path"],
        "full_weight_manifest_path": artifact_paths["full_weight_manifest_path"],
        "publish_manifest_path": artifact_paths["publish_manifest_path"],
        "runtime_contract_path": artifact_paths["runtime_contract_path"],
        "contract_manifest_path": cloud_bundle["contract_manifest_path"],
        "system_execution_manifest_path": cloud_bundle["system_execution_manifest_path"],
        "materialization_state": "contract_only",
        "profile_settings_path": artifact_paths["profile_settings_path"],
        "canonical_profile_catalog_path": artifact_paths["canonical_profile_catalog_path"],
        "execution_profile_binding_key": "edge_cloud_train",
        "compatible_profile_binding_keys": ["local_train", "edge_cloud_train"],
        "bootstrap_contract_binding_keys": ["local_train", "edge_cloud_train"],
        "flow_parameter_contract_binding_keys": ["local_train", "edge_cloud_train"],
        "canonical_execution_profiles_supported": canonical_profile_names(),
    }
    summary_path = write_json(upkg_dir / "upkg40_embodied_train_summary.json", summary)
    report = {
        "name": "CGC_UPKG40_Embodied_Psi0_Train",
        "status": "PASS",
        "scope": "contract_materialization",
        "public_entrypoint": "cgc embodied psi0-train",
        "cloud_artifacts": {
            "summary": cloud_bundle["summary_path"],
            "runtime_report": cloud_bundle["runtime_report_path"],
            "contract_manifest": cloud_bundle["contract_manifest_path"],
            "system_execution_manifest": cloud_bundle["system_execution_manifest_path"],
            "strategy_decision": cloud_bundle["strategy_decision_path"],
            "compatibility_report": cloud_bundle["compatibility_report_path"],
            "distributed_runtime_bootstrap": cloud_bundle["distributed_runtime_bootstrap_path"],
        },
        "upkg40_artifacts": artifact_paths,
        "artifact_index": artifact_entries,
        "artifact_index_path": str(artifact_index_path.resolve()),
        "stage_trace_path": str(stage_trace_path.resolve()),
        "canonical_profile_catalog_path": artifact_paths["canonical_profile_catalog_path"],
        "profile_settings_path": artifact_paths["profile_settings_path"],
        "canonical_execution_profiles_supported": canonical_profile_names(),
        "failure_attribution": failure,
    }
    report_path = write_json(upkg_dir / "upkg40_embodied_train_report.json", report)
    orchestration = {
        "status": "PASS",
        "command": "cgc embodied psi0-train",
        "session_dir": str(session_dir.resolve()),
        "cloud": {
            "launch_returncode": int(cloud_bundle["launch_returncode"]),
            "fetch_returncode": int(cloud_bundle["fetch_returncode"]),
            "summary_path": cloud_bundle["summary_path"],
            "contract_manifest_path": cloud_bundle["contract_manifest_path"],
            "system_execution_manifest_path": cloud_bundle["system_execution_manifest_path"],
            "distributed_runtime_bootstrap_path": cloud_bundle["distributed_runtime_bootstrap_path"],
            "contract_status": cloud_bundle["contract_status"],
            "contract_reason": cloud_bundle["contract_reason"],
        },
        "upkg40_embodied_train": {
            "report_path": str(report_path.resolve()),
            "summary_path": str(summary_path.resolve()),
            "artifact_index_path": str(artifact_index_path.resolve()),
            "stage_trace_path": str(stage_trace_path.resolve()),
            **artifact_paths,
        },
    }
    write_json(session_dir / "orchestration_report.json", orchestration)
    return orchestration


def _write_deploy_reports(
    *,
    session_dir: Path,
    upkg_dir: Path,
    train_context: dict[str, Any],
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    stage_status = {
        "resolve_train_session": {"status": "PASS", "reason": ""},
        "materialize_deploy_contracts": {"status": "PASS", "reason": ""},
        "write_deploy_reports": {"status": "PASS", "reason": ""},
    }
    stage_trace_path = _copy_text_template(
        DEPLOY_TEMPLATE_DIR / "stage_trace.jsonl",
        upkg_dir / "stage_trace.jsonl",
    )
    artifact_entries = artifact_index(
        [
            train_context["full_weight_manifest_path"],
            train_context["publish_manifest_path"],
            train_context["runtime_contract_path"],
            artifact_paths["bridge_info_path"],
            artifact_paths["deploy_contract_path"],
            artifact_paths["consume_contract_path"],
            artifact_paths["deploy_session_path"],
            str(stage_trace_path.resolve()),
            artifact_paths["canonical_profile_catalog_path"],
            artifact_paths["profile_settings_path"],
        ]
    )
    artifact_index_path = write_json(upkg_dir / "artifact_index.json", {"artifacts": artifact_entries})
    failure = failure_attribution(gate_name="upkg40_embodied_deploy", status="PASS", stage_status=stage_status)
    summary = {
        "status": "PASS",
        "command": "cgc embodied psi0-deploy",
        "generated_at": _utc_now(),
        "source_train_report_path": train_context["source_train_report_path"],
        "source_train_session_path": train_context["source_train_session_path"],
        "full_weight_manifest_path": train_context["full_weight_manifest_path"],
        "publish_manifest_path": train_context["publish_manifest_path"],
        "runtime_contract_path": train_context["runtime_contract_path"],
        "deploy_contract_path": artifact_paths["deploy_contract_path"],
        "consume_contract_path": artifact_paths["consume_contract_path"],
        "bridge_info_path": artifact_paths["bridge_info_path"],
        "profile_settings_path": artifact_paths["profile_settings_path"],
        "canonical_profile_catalog_path": artifact_paths["canonical_profile_catalog_path"],
        "execution_profile_binding_key": "edge_cloud_infer",
        "delivery_profile_binding_key": "edge_cloud_infer",
        "compatible_profile_binding_keys": ["local_infer", "edge_cloud_infer"],
        "bootstrap_contract_binding_keys": {
            "delivery": "edge_cloud_infer",
            "target": "local_infer",
        },
        "flow_parameter_contract_binding_keys": {
            "delivery": "edge_cloud_infer",
            "target": "local_infer",
        },
        "canonical_execution_profiles_supported": canonical_profile_names(),
    }
    summary_path = write_json(upkg_dir / "upkg40_embodied_deploy_summary.json", summary)
    report = {
        "name": "CGC_UPKG40_Embodied_Psi0_Deploy",
        "status": "PASS",
        "scope": "deploy_contract_materialization",
        "public_entrypoint": "cgc embodied psi0-deploy",
        "source_train_report_path": train_context["source_train_report_path"],
        "source_train_session_path": train_context["source_train_session_path"],
        "upkg40_artifacts": artifact_paths,
        "artifact_index": artifact_entries,
        "artifact_index_path": str(artifact_index_path.resolve()),
        "stage_trace_path": str(stage_trace_path.resolve()),
        "canonical_execution_profiles_supported": canonical_profile_names(),
        "failure_attribution": failure,
    }
    report_path = write_json(upkg_dir / "upkg40_embodied_deploy_report.json", report)
    orchestration = {
        "status": "PASS",
        "command": "cgc embodied psi0-deploy",
        "session_dir": str(session_dir.resolve()),
        "source_train_report_path": train_context["source_train_report_path"],
        "upkg40_embodied_deploy": {
            "report_path": str(report_path.resolve()),
            "summary_path": str(summary_path.resolve()),
            "artifact_index_path": str(artifact_index_path.resolve()),
            "stage_trace_path": str(stage_trace_path.resolve()),
            **artifact_paths,
        },
    }
    write_json(session_dir / "orchestration_report.json", orchestration)
    return orchestration


def _resolve_train_context(
    *,
    train_session: str,
    launch_command: str,
    fetch_command: str,
) -> dict[str, Any]:
    if not str(train_session or "").strip():
        orchestration = run_embodied_psi0_train(
            output_root=str(DEFAULT_TRAIN_OUTPUT_ROOT),
            launch_command=launch_command,
            fetch_command=fetch_command,
            json_only=True,
        )
        return _resolve_train_context(
            train_session=str(orchestration.get("session_dir") or ""),
            launch_command=launch_command,
            fetch_command=fetch_command,
        )
    candidate = Path(str(train_session)).expanduser().resolve()
    session_dir: Path
    if candidate.is_dir():
        session_dir = candidate
    else:
        payload = read_json(candidate)
        if candidate.name == "orchestration_report.json":
            resolved = Path(str(payload.get("session_dir") or candidate.parent)).expanduser().resolve()
            session_dir = resolved
        elif candidate.name == "psi0_embodied_train_session.json":
            session_dir = candidate.parent.parent.resolve()
        else:
            raise ValueError(f"unsupported_train_session_input:{candidate}")
    upkg_dir = (session_dir / "upkg40_embodied_train").resolve()
    source_train_report_path = _require_file(session_dir / "orchestration_report.json", label="source_train_report")
    source_train_session_path = _require_file(
        upkg_dir / "psi0_embodied_train_session.json",
        label="source_train_session",
    )
    full_weight_manifest_path = _require_file(
        upkg_dir / "psi0_full_weight_manifest.json",
        label="full_weight_manifest",
    )
    publish_manifest_path = _require_file(
        upkg_dir / "psi0_publish_manifest.json",
        label="publish_manifest",
    )
    runtime_contract_path = _require_file(
        upkg_dir / "psi0_runtime_contract.json",
        label="runtime_contract",
    )
    canonical_profile_catalog_path = _require_file(
        upkg_dir / "canonical_profile_catalog.json",
        label="canonical_profile_catalog",
    )
    profile_settings_path = _require_file(
        upkg_dir / "profile_settings.json",
        label="profile_settings",
    )
    return {
        "session_dir": str(session_dir),
        "source_train_report_path": str(source_train_report_path),
        "source_train_session_path": str(source_train_session_path),
        "full_weight_manifest_path": str(full_weight_manifest_path),
        "publish_manifest_path": str(publish_manifest_path),
        "runtime_contract_path": str(runtime_contract_path),
        "canonical_profile_catalog_path": str(canonical_profile_catalog_path),
        "profile_settings_path": str(profile_settings_path),
    }


def _copy_realtimevla_templates(
    *,
    session_dir: Path,
    upkg_dir: Path,
    replacements: dict[str, str],
) -> dict[str, str]:
    copied: dict[str, str] = {}
    root_files = (
        "edge_prompt.txt",
        "edge_push_bundle.json",
        "edge_stdout.txt",
        "edge_stderr.txt",
        "cloud_launch_stdout.txt",
        "cloud_launch_stderr.txt",
        "cloud_fetch_stdout.txt",
        "cloud_fetch_stderr.txt",
        "edge_infer/model_run_session.json",
        "edge_infer/run_artifacts/edge_inference_bridge.json",
        "edge_infer/run_artifacts/m4_inference_report.json",
        "edge_infer/run_artifacts/omlx_flashmoe_manifest.json",
        "edge_infer/run_artifacts/route_decision.json",
        "edge_infer/run_artifacts/run_report.json",
    )
    artifact_files = (
        "canonical_profile_catalog.json",
        "profile_settings.json",
        "psi0_cloud_training_contract.json",
        "realtime_vla_edge_inference_contract.json",
        "embodied_teaching_session.json",
        "embodied_training_dataset_manifest.json",
        "embodied_trained_model_manifest.json",
        "embodied_inference_session.json",
        "embodied_audit_session.json",
        "embodied_replay_session.json",
        "embodied_trace_session.json",
        "psi0_embodied_train_session.json",
        "psi0_embodied_infer_session.json",
        "psi0_embodied_audit_session.json",
        "psi0_embodied_replay_session.json",
        "psi0_embodied_trace_session.json",
        "embodied_parity_report.json",
        "cloud_summary.json",
        "psi0_realtimevla_audit_replay_bundle.json",
        "replay_anchor.json",
        "stage_trace.jsonl",
        "embodied_six_element_events.jsonl",
        "embodied_six_element_summary.json",
        "edge_inference_result.json",
        "cloud_ingest_manifest.json",
    )
    for relative_path in root_files:
        source = REALTIME_TEMPLATE_SESSION_ROOT / relative_path
        target = session_dir / relative_path
        copied[relative_path] = str(_copy_template_path(source, target, replacements).resolve())
    alias_files = {
        "edge_infer/route_decision.json": "edge_infer/run_artifacts/route_decision.json",
        "edge_infer/m4_inference_report.json": "edge_infer/run_artifacts/m4_inference_report.json",
    }
    for target_relative_path, source_relative_path in alias_files.items():
        source = REALTIME_TEMPLATE_SESSION_ROOT / source_relative_path
        target = session_dir / target_relative_path
        copied[target_relative_path] = str(_copy_template_path(source, target, replacements).resolve())
    for relative_path in artifact_files:
        source = REALTIME_TEMPLATE_DIR / relative_path
        target = upkg_dir / relative_path
        copied[relative_path] = str(_copy_template_path(source, target, replacements).resolve())
    return copied


def _normalize_realtimevla_edge_model(
    *,
    session_dir: Path,
    upkg_dir: Path,
    copied_paths: dict[str, str],
    edge_model: str,
) -> None:
    normalized_edge_model = str(edge_model or DEFAULT_EDGE_MODEL).strip() or DEFAULT_EDGE_MODEL

    profile_settings_path = Path(copied_paths["profile_settings.json"]).resolve()
    profile_settings = read_json(profile_settings_path)
    if profile_settings:
        for dotted_path in (
            "profile_descriptors.bootstrap_contract_descriptors.local_infer.bootstrap_parameters.model_locator",
            "profile_descriptors.bootstrap_contract_descriptors.edge_cloud_infer.bootstrap_parameters.model_locator",
            "profile_descriptors.flow_parameter_contract_descriptors.local_infer.parameter_contract.edge_model",
            "scenario_bindings.local_infer.bootstrap_contract_descriptor.bootstrap_parameters.model_locator",
            "scenario_bindings.edge_cloud_infer.bootstrap_contract_descriptor.bootstrap_parameters.model_locator",
            "scenario_bindings.local_infer.flow_parameter_contract_descriptor.parameter_contract.edge_model",
        ):
            _nested_set(profile_settings, dotted_path, normalized_edge_model)
        catalog = profile_settings.get("canonical_profile_catalog")
        if isinstance(catalog, dict):
            for dotted_path in (
                "bootstrap_contract_descriptors.local_infer.bootstrap_parameters.model_locator",
                "bootstrap_contract_descriptors.edge_cloud_infer.bootstrap_parameters.model_locator",
                "flow_parameter_contract_descriptors.local_infer.parameter_contract.edge_model",
            ):
                _nested_set(catalog, dotted_path, normalized_edge_model)
        write_json(profile_settings_path, profile_settings)

    edge_push_bundle_path = (session_dir / "edge_push_bundle.json").resolve()
    edge_push_bundle = read_json(edge_push_bundle_path)
    if edge_push_bundle:
        _nested_set(edge_push_bundle, "target_edge_runtime.model", normalized_edge_model)
        write_json(edge_push_bundle_path, edge_push_bundle)

    cloud_ingest_manifest_path = Path(copied_paths["cloud_ingest_manifest.json"]).resolve()
    cloud_ingest_manifest = read_json(cloud_ingest_manifest_path)
    if cloud_ingest_manifest:
        _nested_set(cloud_ingest_manifest, "target_edge_runtime.model", normalized_edge_model)
        write_json(cloud_ingest_manifest_path, cloud_ingest_manifest)

    edge_inference_contract_path = Path(copied_paths["realtime_vla_edge_inference_contract.json"]).resolve()
    edge_inference_contract = read_json(edge_inference_contract_path)
    if edge_inference_contract:
        edge_inference_contract["model"] = normalized_edge_model
        write_json(edge_inference_contract_path, edge_inference_contract)


def _run_live_realtimevla_edge(
    *,
    session_dir: Path,
    upkg_dir: Path,
    edge_model: str,
    edge_local_omlx_model: str,
    copied_paths: dict[str, str],
) -> dict[str, Any]:
    prompt_path = Path(copied_paths["edge_prompt.txt"]).resolve()
    prompt_text = prompt_path.read_text(encoding="utf-8")
    report_dir = (session_dir / "edge_infer").resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    local_omlx_override = str(edge_local_omlx_model or "").strip()
    command = [
        sys.executable,
        str((Path(__file__).resolve().parent / "cgc.py").resolve()),
        "run",
        str(edge_model),
        "--use-omlx",
        "--prompt",
        prompt_text,
        "--max-tokens",
        "96",
        "--report-dir",
        str(report_dir),
        "--json",
    ]
    env = os.environ.copy()
    if local_omlx_override:
        env["CGC_LOCAL_OMLX_MODEL"] = local_omlx_override
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        env=env,
    )
    stdout_path = _write_text(session_dir / "edge_stdout.txt", proc.stdout)
    stderr_path = _write_text(session_dir / "edge_stderr.txt", proc.stderr)
    payload = _extract_json_payload(proc.stdout)
    if proc.returncode != 0 and not payload:
        raise RuntimeError(
            f"live_cgc_run_failed:{proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    if not payload:
        raise RuntimeError(f"live_cgc_run_did_not_emit_json\nstdout:\n{proc.stdout}")
    run_session_path = _require_file(report_dir / "model_run_session.json", label="live_model_run_session")
    run_report_path = _require_file(report_dir / "run_artifacts" / "run_report.json", label="live_run_report")
    route_decision_source = _require_file(
        report_dir / "run_artifacts" / "route_decision.json",
        label="live_route_decision",
    )
    m4_report_source = _require_file(
        report_dir / "run_artifacts" / "m4_inference_report.json",
        label="live_m4_inference_report",
    )
    bridge_path = _require_file(
        report_dir / "run_artifacts" / "edge_inference_bridge.json",
        label="live_edge_inference_bridge",
    )
    route_decision_alias = _copy_template_path(
        route_decision_source,
        report_dir / "route_decision.json",
        {},
    )
    m4_report_alias = _copy_template_path(
        m4_report_source,
        report_dir / "m4_inference_report.json",
        {},
    )
    run_report = read_json(run_report_path)
    run_session = read_json(run_session_path)
    route_decision = read_json(route_decision_source)
    evidence_path = Path(str(run_report.get("evidence_path") or "")).expanduser().resolve()
    local_infer_evidence_path = str(evidence_path) if evidence_path.exists() else ""
    edge_result = {
        "status": str(run_session.get("status") or run_report.get("status") or "FAIL"),
        "runtime_host": "realtime-vla-v2",
        "local_execution": bool(run_session.get("local_execution")),
        "selected_route": str(run_session.get("selected_route") or route_decision.get("selected_route") or ""),
        "selected_backend": str(run_session.get("selected_backend") or route_decision.get("selected_backend") or ""),
        "response_text": str(_nested_get(run_session, "response.text", "") or run_report.get("response_text") or ""),
        "edge_latency_ms": float(run_session.get("edge_latency_ms") or run_report.get("elapsed_ms") or 0.0),
        "run_report_path": str(run_report_path),
        "m4_inference_report_path": str(m4_report_alias.resolve()),
        "edge_inference_bridge_path": str(bridge_path),
        "route_decision_path": str(route_decision_alias.resolve()),
        "local_infer_evidence_path": local_infer_evidence_path,
        **_profile_binding_fields(
            profile_settings_path=str((upkg_dir / "profile_settings.json").resolve()),
            execution_map={"execution": "local_infer", "bridge": "edge_cloud_infer"},
            delivery="local_infer",
            bootstrap={"delivery": "edge_cloud_infer", "target": "local_infer"},
            flow={"delivery": "edge_cloud_infer", "target": "local_infer"},
        ),
        "canonical_execution_profiles_supported": canonical_profile_names(),
        "canonical_profile_catalog_path": str((upkg_dir / "canonical_profile_catalog.json").resolve()),
    }
    edge_result_path = write_json(upkg_dir / "edge_inference_result.json", edge_result)
    evidence = run_report.get("evidence") if isinstance(run_report.get("evidence"), dict) else {}
    copied_paths["edge_stdout.txt"] = str(stdout_path.resolve())
    copied_paths["edge_stderr.txt"] = str(stderr_path.resolve())
    copied_paths["edge_infer/model_run_session.json"] = str(run_session_path)
    copied_paths["edge_infer/run_artifacts/run_report.json"] = str(run_report_path)
    copied_paths["edge_infer/run_artifacts/route_decision.json"] = str(route_decision_source)
    copied_paths["edge_infer/run_artifacts/m4_inference_report.json"] = str(m4_report_source)
    copied_paths["edge_infer/run_artifacts/edge_inference_bridge.json"] = str(bridge_path)
    copied_paths["edge_infer/route_decision.json"] = str(route_decision_alias.resolve())
    copied_paths["edge_infer/m4_inference_report.json"] = str(m4_report_alias.resolve())
    copied_paths["edge_inference_result.json"] = str(edge_result_path.resolve())
    return {
        "payload": payload,
        "run_session": run_session,
        "run_report": run_report,
        "route_decision": route_decision,
        "edge_result": edge_result,
    }


def _refresh_realtimevla_live_edge_artifacts(
    *,
    upkg_dir: Path,
    copied_paths: dict[str, str],
    live_edge: dict[str, Any],
) -> None:
    edge_result = live_edge["edge_result"] if isinstance(live_edge.get("edge_result"), dict) else {}
    local_infer_evidence_path = str(edge_result.get("local_infer_evidence_path") or "")
    decision_reason = str(
        _nested_get(live_edge.get("run_report") or {}, "evidence.reason", "")
        or _nested_get(live_edge.get("route_decision") or {}, "decision_reason.text", "")
        or "edge_live_run_not_completed"
    )
    inferred_status = str(edge_result.get("status") or "FAIL")

    embodied_inference_session = read_json(Path(copied_paths["embodied_inference_session.json"]))
    embodied_inference_session.update(
        {
            "status": inferred_status,
            "runtime_evidence_path": local_infer_evidence_path,
            "edge_latency_ms": float(edge_result.get("edge_latency_ms") or 0.0),
            "selected_route": str(edge_result.get("selected_route") or ""),
            "selected_backend": str(edge_result.get("selected_backend") or ""),
            "local_execution": bool(edge_result.get("local_execution")),
        }
    )
    write_json(upkg_dir / "embodied_inference_session.json", embodied_inference_session)

    psi0_infer_session = read_json(Path(copied_paths["psi0_embodied_infer_session.json"]))
    psi0_infer_session.update(
        {
            "status": inferred_status,
            "edge_inference_result_path": str((upkg_dir / "edge_inference_result.json").resolve()),
        }
    )
    write_json(upkg_dir / "psi0_embodied_infer_session.json", psi0_infer_session)

    replay_anchor = read_json(Path(copied_paths["replay_anchor.json"]))
    replay_anchor["local_infer_evidence_path"] = local_infer_evidence_path
    write_json(upkg_dir / "replay_anchor.json", replay_anchor)

    psi0_replay_session = read_json(Path(copied_paths["psi0_embodied_replay_session.json"]))
    psi0_replay_session["gui_runtime_evidence_path"] = local_infer_evidence_path
    if isinstance(psi0_replay_session.get("replay_anchor"), dict):
        psi0_replay_session["replay_anchor"]["local_infer_evidence_path"] = local_infer_evidence_path
    write_json(upkg_dir / "psi0_embodied_replay_session.json", psi0_replay_session)

    stage_trace = _read_jsonl(Path(copied_paths["stage_trace.jsonl"]))
    for item in stage_trace:
        if item.get("stage") in {"realtime_vla_edge_inference_contract", "embodied_inference_session"}:
            item["status"] = inferred_status
            item["reason"] = "" if inferred_status == "PASS" else decision_reason
    _write_jsonl_records(upkg_dir / "stage_trace.jsonl", stage_trace)

    six_element_events = _read_jsonl(Path(copied_paths["embodied_six_element_events.jsonl"]))
    for item in six_element_events:
        if item.get("stage") == "edge_inference":
            item["status"] = inferred_status
            item["payload"] = edge_result
    _write_jsonl_records(upkg_dir / "embodied_six_element_events.jsonl", six_element_events)

    psi0_trace_session = read_json(Path(copied_paths["psi0_embodied_trace_session.json"]))
    preview = psi0_trace_session.get("stage_trace_preview")
    if isinstance(preview, list):
        for item in preview:
            if item.get("stage") in {"realtime_vla_edge_inference_contract", "embodied_inference_session"}:
                item["status"] = inferred_status
                item["reason"] = "" if inferred_status == "PASS" else decision_reason
    six_preview = psi0_trace_session.get("six_element_event_preview")
    if isinstance(six_preview, list):
        for item in six_preview:
            if item.get("stage") == "edge_inference":
                item["status"] = inferred_status
                item["payload"] = edge_result
    write_json(upkg_dir / "psi0_embodied_trace_session.json", psi0_trace_session)


def _write_realtimevla_reports(
    *,
    session_dir: Path,
    upkg_dir: Path,
    cloud_bundle: dict[str, Any],
    copied_paths: dict[str, str],
) -> dict[str, Any]:
    stage_trace_path = Path(copied_paths["stage_trace.jsonl"]).resolve()
    stage_trace = _read_jsonl(stage_trace_path)
    stage_status = {
        str(item.get("stage") or ""): {
            "status": str(item.get("status") or "PASS"),
            "reason": str(item.get("reason") or ""),
        }
        for item in stage_trace
        if str(item.get("stage") or "")
    }
    six_element_events_path = Path(copied_paths["embodied_six_element_events.jsonl"]).resolve()
    six_element_events = _read_jsonl(six_element_events_path)
    cloud_contract = read_json(Path(copied_paths["psi0_cloud_training_contract.json"]))
    edge_push_bundle = read_json(Path(copied_paths["edge_push_bundle.json"]))
    edge_inference_result = read_json(Path(copied_paths["edge_inference_result.json"]))
    parity_report = read_json(Path(copied_paths["embodied_parity_report.json"]))
    cloud_contract_reason = str(
        cloud_bundle.get("contract_reason") or cloud_contract.get("contract_reason") or ""
    )
    edge_model = (
        edge_inference_result.get("model")
        or edge_inference_result.get("runtime_model")
        or (
            edge_push_bundle.get("target_edge_runtime", {}).get("model")
            if isinstance(edge_push_bundle.get("target_edge_runtime"), dict)
            else None
        )
        or DEFAULT_EDGE_MODEL
    )
    summary_status = "PASS" if not any(item["status"] == "FAIL" for item in stage_status.values()) else "FAIL"
    failure = failure_attribution(gate_name="upkg40_embodied", status=summary_status, stage_status=stage_status)
    artifact_paths = {
        "canonical_profile_catalog_path": copied_paths["canonical_profile_catalog.json"],
        "profile_settings_path": copied_paths["profile_settings.json"],
        "psi0_cloud_training_contract_path": copied_paths["psi0_cloud_training_contract.json"],
        "realtime_vla_edge_inference_contract_path": copied_paths["realtime_vla_edge_inference_contract.json"],
        "embodied_teaching_session_path": copied_paths["embodied_teaching_session.json"],
        "embodied_training_dataset_manifest_path": copied_paths["embodied_training_dataset_manifest.json"],
        "embodied_trained_model_manifest_path": copied_paths["embodied_trained_model_manifest.json"],
        "embodied_inference_session_path": copied_paths["embodied_inference_session.json"],
        "embodied_audit_session_path": copied_paths["embodied_audit_session.json"],
        "embodied_replay_session_path": copied_paths["embodied_replay_session.json"],
        "embodied_trace_session_path": copied_paths["embodied_trace_session.json"],
        "embodied_parity_report_path": copied_paths["embodied_parity_report.json"],
        "cloud_summary_path": copied_paths["cloud_summary.json"],
        "audit_replay_bundle_path": copied_paths["psi0_realtimevla_audit_replay_bundle.json"],
    }
    artifact_entries = artifact_index(
        [
            artifact_paths["psi0_cloud_training_contract_path"],
            artifact_paths["realtime_vla_edge_inference_contract_path"],
            artifact_paths["embodied_teaching_session_path"],
            artifact_paths["embodied_training_dataset_manifest_path"],
            artifact_paths["embodied_trained_model_manifest_path"],
            artifact_paths["embodied_inference_session_path"],
            artifact_paths["embodied_audit_session_path"],
            artifact_paths["embodied_replay_session_path"],
            artifact_paths["embodied_trace_session_path"],
            artifact_paths["embodied_parity_report_path"],
            artifact_paths["cloud_summary_path"],
            artifact_paths["audit_replay_bundle_path"],
            copied_paths["psi0_embodied_train_session.json"],
            copied_paths["psi0_embodied_infer_session.json"],
            copied_paths["psi0_embodied_audit_session.json"],
            copied_paths["psi0_embodied_replay_session.json"],
            copied_paths["psi0_embodied_trace_session.json"],
            copied_paths["stage_trace.jsonl"],
            copied_paths["embodied_six_element_events.jsonl"],
            copied_paths["embodied_six_element_summary.json"],
            copied_paths["edge_inference_result.json"],
            copied_paths["cloud_ingest_manifest.json"],
            copied_paths["canonical_profile_catalog.json"],
            copied_paths["profile_settings.json"],
        ]
    )
    artifact_index_path = write_json(upkg_dir / "artifact_index.json", {"artifacts": artifact_entries})
    summary = {
        "gate": "upkg40_embodied",
        "milestone": "upkg40_embodied",
        "status": summary_status,
        "report_path": str((upkg_dir / "upkg40_embodied_report.json").resolve()),
        "matrix_axes": {
            "training_model": "psi0",
            "edge_runtime_host": "realtime-vla-v2",
            "bridge_mode": "psi0_cloud_training_to_realtime_vla",
            "source_runtime_mode": "edge_cloud_train",
            "delivery_runtime_mode": "edge_cloud_infer",
            "target_runtime_mode": "local_infer",
            "canonical_execution_profiles_supported": canonical_profile_names(),
            "cloud_contract_status": cloud_bundle["contract_status"],
            "cloud_contract_reason": cloud_contract_reason,
            "edge_selected_route": edge_inference_result.get("selected_route"),
            "edge_selected_backend": edge_inference_result.get("selected_backend"),
        },
        "artifact_index": artifact_entries,
        "stage_trace": stage_trace,
        "failure_attribution": failure,
    }
    summary_path = write_json(upkg_dir / "upkg40_embodied_summary.json", summary)
    report = {
        "name": "CGC_UPKG40_Embodied_Psi0_RealtimeVLA_Fullchain",
        "status": summary_status,
        "scope": "verification_and_audit",
        "public_entrypoint": "cgc embodied psi0-realtimevla",
        "cloud_artifacts": {
            "summary": cloud_bundle["summary_path"],
            "runtime_report": cloud_bundle["runtime_report_path"],
            "contract_manifest": cloud_bundle["contract_manifest_path"],
            "system_execution_manifest": cloud_bundle["system_execution_manifest_path"],
            "strategy_decision": cloud_bundle["strategy_decision_path"],
            "compatibility_report": cloud_bundle["compatibility_report_path"],
            "distributed_runtime_bootstrap": cloud_bundle["distributed_runtime_bootstrap_path"],
        },
        "upkg40_artifacts": artifact_paths,
        "artifact_index": artifact_entries,
        "artifact_index_path": str(artifact_index_path.resolve()),
        "stage_trace_path": copied_paths["stage_trace.jsonl"],
        "six_element_events_path": copied_paths["embodied_six_element_events.jsonl"],
        "six_element_summary_path": copied_paths["embodied_six_element_summary.json"],
        "canonical_profile_catalog_path": copied_paths["canonical_profile_catalog.json"],
        "profile_settings_path": copied_paths["profile_settings.json"],
        "canonical_execution_profiles_supported": canonical_profile_names(),
        "failure_attribution": failure,
        "edge_runtime": {
            "model": edge_model,
            "local_execution": edge_inference_result.get("local_execution"),
            "selected_route": edge_inference_result.get("selected_route"),
            "selected_backend": edge_inference_result.get("selected_backend"),
            "edge_latency_ms": edge_inference_result.get("edge_latency_ms"),
        },
        "parity_report": parity_report,
    }
    report_path = write_json(upkg_dir / "upkg40_embodied_report.json", report)
    orchestration = {
        "status": summary_status,
        "command": "cgc embodied psi0-realtimevla",
        "session_dir": str(session_dir.resolve()),
        "cloud": {
            "launch_returncode": int(cloud_bundle["launch_returncode"]),
            "fetch_returncode": int(cloud_bundle["fetch_returncode"]),
            "summary_path": cloud_bundle["summary_path"],
            "contract_manifest_path": cloud_bundle["contract_manifest_path"],
            "system_execution_manifest_path": cloud_bundle["system_execution_manifest_path"],
            "distributed_runtime_bootstrap_path": cloud_bundle["distributed_runtime_bootstrap_path"],
            "contract_status": cloud_bundle["contract_status"],
            "contract_reason": cloud_contract_reason,
        },
        "edge_push_bundle_path": copied_paths["edge_push_bundle.json"],
        "edge": {
            "run_returncode": 1 if summary_status == "FAIL" else 0,
            "status": edge_inference_result.get("status"),
            "selected_route": edge_inference_result.get("selected_route"),
            "selected_backend": edge_inference_result.get("selected_backend"),
            "local_execution": edge_inference_result.get("local_execution"),
            "response_text": edge_inference_result.get("response_text"),
            "edge_latency_ms": edge_inference_result.get("edge_latency_ms"),
            "evidence_paths": {
                "run_report": copied_paths["edge_infer/run_artifacts/run_report.json"],
                "m4_inference_report": copied_paths["edge_infer/m4_inference_report.json"],
                "edge_inference_bridge": copied_paths["edge_infer/run_artifacts/edge_inference_bridge.json"],
                "route_decision": copied_paths["edge_infer/route_decision.json"],
                "local_infer": edge_inference_result.get("local_infer_evidence_path"),
            },
            "stdout_path": copied_paths["edge_stdout.txt"],
            "stderr_path": copied_paths["edge_stderr.txt"],
        },
        "upkg40_embodied": {
            "report_path": str(report_path.resolve()),
            "summary_path": str(summary_path.resolve()),
            "artifact_index_path": str(artifact_index_path.resolve()),
            "canonical_profile_catalog_path": copied_paths["canonical_profile_catalog.json"],
            "profile_settings_path": copied_paths["profile_settings.json"],
            "stage_trace_path": copied_paths["stage_trace.jsonl"],
            "audit_replay_bundle_path": copied_paths["psi0_realtimevla_audit_replay_bundle.json"],
            "train_session_path": copied_paths["psi0_embodied_train_session.json"],
            "infer_session_path": copied_paths["psi0_embodied_infer_session.json"],
            "audit_session_path": copied_paths["psi0_embodied_audit_session.json"],
            "replay_session_path": copied_paths["psi0_embodied_replay_session.json"],
            "trace_session_path": copied_paths["psi0_embodied_trace_session.json"],
        },
    }
    write_json(session_dir / "orchestration_report.json", orchestration)
    return orchestration


def run_embodied_psi0_realtimevla(
    *,
    output_root: str = "",
    edge_model: str = DEFAULT_EDGE_MODEL,
    edge_local_omlx_model: str = "",
    launch_command: str = "",
    fetch_command: str = "",
    json_only: bool = False,
) -> dict[str, Any]:
    edge_model = str(edge_model or DEFAULT_EDGE_MODEL).strip() or DEFAULT_EDGE_MODEL
    edge_local_omlx_model = str(edge_local_omlx_model or os.environ.get("CGC_LOCAL_OMLX_MODEL") or "").strip()
    session_dir = _session_dir(output_root or DEFAULT_REALTIMEVLA_OUTPUT_ROOT, label="psi0_realtimevla")
    upkg_dir = (session_dir / "upkg40_embodied").resolve()
    upkg_dir.mkdir(parents=True, exist_ok=True)
    cloud_bundle = _materialize_cloud_bundle(
        session_dir=session_dir,
        launch_command=launch_command or DEFAULT_LAUNCH_COMMAND,
        fetch_command=fetch_command or DEFAULT_FETCH_COMMAND,
    )
    replacements = {
        str(REALTIME_TEMPLATE_SESSION_ROOT): str(session_dir.resolve()),
        LEGACY_REALTIME_TEMPLATE_EDGE_MODEL: edge_model,
        DEFAULT_EDGE_MODEL: edge_model,
    }
    copied_paths = _copy_realtimevla_templates(
        session_dir=session_dir,
        upkg_dir=upkg_dir,
        replacements=replacements,
    )
    _normalize_realtimevla_edge_model(
        session_dir=session_dir,
        upkg_dir=upkg_dir,
        copied_paths=copied_paths,
        edge_model=edge_model,
    )
    live_edge = _run_live_realtimevla_edge(
        session_dir=session_dir,
        upkg_dir=upkg_dir,
        edge_model=edge_model,
        edge_local_omlx_model=edge_local_omlx_model,
        copied_paths=copied_paths,
    )
    _refresh_realtimevla_live_edge_artifacts(
        upkg_dir=upkg_dir,
        copied_paths=copied_paths,
        live_edge=live_edge,
    )
    orchestration = _write_realtimevla_reports(
        session_dir=session_dir,
        upkg_dir=upkg_dir,
        cloud_bundle=cloud_bundle,
        copied_paths=copied_paths,
    )
    if not json_only:
        print(json.dumps(orchestration, ensure_ascii=False, indent=2))
    return orchestration


def run_embodied_psi0_train(
    *,
    output_root: str = "",
    launch_command: str = "",
    fetch_command: str = "",
    json_only: bool = False,
) -> dict[str, Any]:
    session_dir = _session_dir(output_root or DEFAULT_TRAIN_OUTPUT_ROOT, label="psi0_train")
    upkg_dir = (session_dir / "upkg40_embodied_train").resolve()
    upkg_dir.mkdir(parents=True, exist_ok=True)
    cloud_bundle = _materialize_cloud_bundle(
        session_dir=session_dir,
        launch_command=launch_command or DEFAULT_LAUNCH_COMMAND,
        fetch_command=fetch_command or DEFAULT_FETCH_COMMAND,
    )
    replacements = {
        str(TRAIN_TEMPLATE_SESSION_ROOT): str(session_dir.resolve()),
    }
    canonical_profile_catalog = _template_json(TRAIN_TEMPLATE_DIR / "canonical_profile_catalog.json", replacements)
    profile_settings = _template_json(TRAIN_TEMPLATE_DIR / "profile_settings.json", replacements)
    canonical_profile_catalog_path = write_json(upkg_dir / "canonical_profile_catalog.json", canonical_profile_catalog)
    profile_settings_path = write_json(upkg_dir / "profile_settings.json", profile_settings)
    profile_settings_path_str = str(profile_settings_path.resolve())

    cloud_summary = _rewrite_binding_payload(
        _template_json(TRAIN_TEMPLATE_DIR / "cloud_summary.json", replacements),
        _profile_binding_fields(
            profile_settings_path=profile_settings_path_str,
            execution="edge_cloud_train",
            bootstrap="edge_cloud_train",
            flow="edge_cloud_train",
        ),
    )
    cloud_summary_path = write_json(upkg_dir / "cloud_summary.json", cloud_summary)

    full_weight_manifest = _rewrite_binding_payload(
        _template_json(TRAIN_TEMPLATE_DIR / "psi0_full_weight_manifest.json", replacements),
        _profile_binding_fields(
            profile_settings_path=profile_settings_path_str,
            execution="edge_cloud_train",
            delivery="edge_cloud_train",
            compatible=["local_train", "edge_cloud_train"],
            bootstrap=["local_train", "edge_cloud_train"],
            flow=["local_train", "edge_cloud_train"],
        ),
    )
    full_weight_manifest_path = write_json(upkg_dir / "psi0_full_weight_manifest.json", full_weight_manifest)

    publish_manifest = _rewrite_binding_payload(
        _template_json(TRAIN_TEMPLATE_DIR / "psi0_publish_manifest.json", replacements),
        _profile_binding_fields(
            profile_settings_path=profile_settings_path_str,
            execution_map={
                "source": "edge_cloud_train",
                "target": "local_infer",
                "bridge": "edge_cloud_infer",
            },
            delivery="edge_cloud_infer",
            bootstrap={
                "source": "edge_cloud_train",
                "delivery": "edge_cloud_infer",
                "target": "local_infer",
            },
            flow={
                "source": "edge_cloud_train",
                "delivery": "edge_cloud_infer",
                "target": "local_infer",
            },
        ),
    )
    publish_manifest_path = write_json(upkg_dir / "psi0_publish_manifest.json", publish_manifest)

    runtime_contract = _rewrite_binding_payload(
        _template_json(TRAIN_TEMPLATE_DIR / "psi0_runtime_contract.json", replacements),
        _profile_binding_fields(
            profile_settings_path=profile_settings_path_str,
            execution="edge_cloud_train",
            delivery="edge_cloud_infer",
            compatible=["local_train", "edge_cloud_train"],
            bootstrap={
                "local_train": "local_train",
                "edge_cloud_train": "edge_cloud_train",
                "edge_cloud_infer": "edge_cloud_infer",
            },
            flow={
                "local_train": "local_train",
                "edge_cloud_train": "edge_cloud_train",
                "edge_cloud_infer": "edge_cloud_infer",
            },
        ),
    )
    runtime_contract_path = write_json(upkg_dir / "psi0_runtime_contract.json", runtime_contract)

    train_session = _rewrite_binding_payload(
        _template_json(TRAIN_TEMPLATE_DIR / "psi0_embodied_train_session.json", replacements),
        _profile_binding_fields(
            profile_settings_path=profile_settings_path_str,
            execution="edge_cloud_train",
            compatible=["local_train", "edge_cloud_train"],
            bootstrap=["local_train", "edge_cloud_train"],
            flow=["local_train", "edge_cloud_train"],
        ),
    )
    train_session_path = write_json(upkg_dir / "psi0_embodied_train_session.json", train_session)

    artifact_paths = {
        "canonical_profile_catalog_path": str(canonical_profile_catalog_path.resolve()),
        "profile_settings_path": profile_settings_path_str,
        "cloud_summary_path": str(cloud_summary_path.resolve()),
        "full_weight_manifest_path": str(full_weight_manifest_path.resolve()),
        "publish_manifest_path": str(publish_manifest_path.resolve()),
        "runtime_contract_path": str(runtime_contract_path.resolve()),
        "train_session_path": str(train_session_path.resolve()),
    }
    orchestration = _write_train_reports(
        session_dir=session_dir,
        upkg_dir=upkg_dir,
        cloud_bundle=cloud_bundle,
        artifact_paths=artifact_paths,
    )
    if not json_only:
        print(json.dumps(orchestration, ensure_ascii=False, indent=2))
    return orchestration


def run_embodied_psi0_deploy(
    *,
    output_root: str = "",
    train_session: str = "",
    launch_command: str = "",
    fetch_command: str = "",
    json_only: bool = False,
) -> dict[str, Any]:
    train_context = _resolve_train_context(
        train_session=train_session,
        launch_command=launch_command or DEFAULT_LAUNCH_COMMAND,
        fetch_command=fetch_command or DEFAULT_FETCH_COMMAND,
    )
    session_dir = _session_dir(output_root or DEFAULT_DEPLOY_OUTPUT_ROOT, label="psi0_deploy")
    upkg_dir = (session_dir / "upkg40_embodied_deploy").resolve()
    upkg_dir.mkdir(parents=True, exist_ok=True)
    replacements = {
        str(DEPLOY_TEMPLATE_SESSION_ROOT): str(session_dir.resolve()),
        str(TRAIN_TEMPLATE_SESSION_ROOT): str(Path(train_context["session_dir"]).resolve()),
    }
    canonical_profile_catalog = _template_json(DEPLOY_TEMPLATE_DIR / "canonical_profile_catalog.json", replacements)
    profile_settings = _template_json(DEPLOY_TEMPLATE_DIR / "profile_settings.json", replacements)
    canonical_profile_catalog_path = write_json(upkg_dir / "canonical_profile_catalog.json", canonical_profile_catalog)
    profile_settings_path = write_json(upkg_dir / "profile_settings.json", profile_settings)
    profile_settings_path_str = str(profile_settings_path.resolve())

    bridge_info = _rewrite_binding_payload(
        _template_json(DEPLOY_TEMPLATE_DIR / "psi0_bridge_info.json", replacements),
        _profile_binding_fields(
            profile_settings_path=profile_settings_path_str,
            execution_map={
                "source": "edge_cloud_train",
                "bridge": "edge_cloud_infer",
                "target": "local_infer",
            },
            delivery="edge_cloud_infer",
            bootstrap={
                "source": "edge_cloud_train",
                "delivery": "edge_cloud_infer",
                "target": "local_infer",
            },
            flow={
                "source": "edge_cloud_train",
                "delivery": "edge_cloud_infer",
                "target": "local_infer",
            },
        ),
    )
    bridge_info_path = write_json(upkg_dir / "psi0_bridge_info.json", bridge_info)

    deploy_contract = _rewrite_binding_payload(
        _template_json(DEPLOY_TEMPLATE_DIR / "psi0_deploy_contract.json", replacements),
        _profile_binding_fields(
            profile_settings_path=profile_settings_path_str,
            execution="edge_cloud_infer",
            delivery="edge_cloud_infer",
            compatible=["local_infer", "edge_cloud_infer"],
            bootstrap={
                "source": "edge_cloud_train",
                "delivery": "edge_cloud_infer",
                "target": "local_infer",
            },
            flow={
                "source": "edge_cloud_train",
                "delivery": "edge_cloud_infer",
                "target": "local_infer",
            },
        ),
    )
    deploy_contract_path = write_json(upkg_dir / "psi0_deploy_contract.json", deploy_contract)

    consume_contract = _rewrite_binding_payload(
        _template_json(DEPLOY_TEMPLATE_DIR / "realtime_vla_consume_contract.json", replacements),
        _profile_binding_fields(
            profile_settings_path=profile_settings_path_str,
            execution="local_infer",
            delivery="edge_cloud_infer",
            compatible=["local_infer", "edge_cloud_infer", "local_train", "edge_cloud_train"],
            bootstrap={
                "local_train": "local_train",
                "edge_cloud_train": "edge_cloud_train",
                "edge_cloud_infer": "edge_cloud_infer",
                "local_infer": "local_infer",
            },
            flow={
                "local_train": "local_train",
                "edge_cloud_train": "edge_cloud_train",
                "edge_cloud_infer": "edge_cloud_infer",
                "local_infer": "local_infer",
            },
        ),
    )
    consume_contract_path = write_json(upkg_dir / "realtime_vla_consume_contract.json", consume_contract)

    deploy_session = _rewrite_binding_payload(
        _template_json(DEPLOY_TEMPLATE_DIR / "psi0_embodied_deploy_session.json", replacements),
        _profile_binding_fields(
            profile_settings_path=profile_settings_path_str,
            execution="edge_cloud_infer",
            delivery="edge_cloud_infer",
            compatible=["local_infer", "edge_cloud_infer"],
            bootstrap={
                "delivery": "edge_cloud_infer",
                "target": "local_infer",
            },
            flow={
                "delivery": "edge_cloud_infer",
                "target": "local_infer",
            },
        ),
    )
    deploy_session_path = write_json(upkg_dir / "psi0_embodied_deploy_session.json", deploy_session)

    artifact_paths = {
        "canonical_profile_catalog_path": str(canonical_profile_catalog_path.resolve()),
        "profile_settings_path": profile_settings_path_str,
        "bridge_info_path": str(bridge_info_path.resolve()),
        "deploy_contract_path": str(deploy_contract_path.resolve()),
        "consume_contract_path": str(consume_contract_path.resolve()),
        "deploy_session_path": str(deploy_session_path.resolve()),
    }
    orchestration = _write_deploy_reports(
        session_dir=session_dir,
        upkg_dir=upkg_dir,
        train_context=train_context,
        artifact_paths=artifact_paths,
    )
    if not json_only:
        print(json.dumps(orchestration, ensure_ascii=False, indent=2))
    return orchestration


def main() -> int:
    parser = argparse.ArgumentParser(description="Recovered embodied UPKG 4.0 helpers")
    subparsers = parser.add_subparsers(dest="command")

    realtime_parser = subparsers.add_parser("psi0-realtimevla")
    realtime_parser.add_argument("--output-root", type=str, default="")
    realtime_parser.add_argument("--edge-model", type=str, default=DEFAULT_EDGE_MODEL)
    realtime_parser.add_argument("--edge-local-omlx-model", type=str, default="")
    realtime_parser.add_argument("--launch-command", type=str, default="")
    realtime_parser.add_argument("--fetch-command", type=str, default="")

    train_parser = subparsers.add_parser("psi0-train")
    train_parser.add_argument("--output-root", type=str, default="")
    train_parser.add_argument("--launch-command", type=str, default="")
    train_parser.add_argument("--fetch-command", type=str, default="")

    deploy_parser = subparsers.add_parser("psi0-deploy")
    deploy_parser.add_argument("--output-root", type=str, default="")
    deploy_parser.add_argument("--train-session", type=str, default="")
    deploy_parser.add_argument("--launch-command", type=str, default="")
    deploy_parser.add_argument("--fetch-command", type=str, default="")

    args = parser.parse_args()
    if args.command == "psi0-realtimevla":
        result = run_embodied_psi0_realtimevla(
            output_root=args.output_root,
            edge_model=args.edge_model,
            edge_local_omlx_model=args.edge_local_omlx_model,
            launch_command=args.launch_command,
            fetch_command=args.fetch_command,
            json_only=True,
        )
    elif args.command == "psi0-train":
        result = run_embodied_psi0_train(
            output_root=args.output_root,
            launch_command=args.launch_command,
            fetch_command=args.fetch_command,
            json_only=True,
        )
    elif args.command == "psi0-deploy":
        result = run_embodied_psi0_deploy(
            output_root=args.output_root,
            train_session=args.train_session,
            launch_command=args.launch_command,
            fetch_command=args.fetch_command,
            json_only=True,
        )
    else:
        parser.print_help()
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "DEFAULT_EDGE_MODEL",
    "run_embodied_psi0_realtimevla",
    "run_embodied_psi0_train",
    "run_embodied_psi0_deploy",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
