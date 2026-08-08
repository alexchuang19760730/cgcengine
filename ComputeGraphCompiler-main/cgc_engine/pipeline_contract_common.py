from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


PIPELINE_KERNEL_ARTIFACT_KEYS = (
    "execution_context_path",
    "state_abi_path",
    "strategy_decision_path",
    "compatibility_report_path",
    "distributed_runtime_bootstrap_path",
    "contract_manifest_path",
    "system_execution_manifest_path",
)

PIPELINE_KERNEL_REQUIRED_KEYS = (
    "execution_context_path",
    "state_abi_path",
    "contract_manifest_path",
    "system_execution_manifest_path",
)


def candidate_output_roots(output_dir: Path) -> List[Path]:
    output_root = Path(output_dir).expanduser().resolve()
    candidates = [output_root, output_root.parent, output_root.parent.parent]
    unique: List[Path] = []
    seen = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path.resolve())
    return unique


def _resolve_pipeline_artifact_path(raw: Any) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    return str(Path(value).expanduser().resolve())


def pipeline_kernel_contract_artifacts_from_report(report: Dict[str, Any] | None) -> Dict[str, str]:
    payload = report if isinstance(report, dict) else {}
    artifacts = payload.get("pipeline_kernel_contract_artifacts") if isinstance(payload.get("pipeline_kernel_contract_artifacts"), dict) else {}
    resolved: Dict[str, str] = {}
    for key in PIPELINE_KERNEL_ARTIFACT_KEYS:
        resolved[key] = _resolve_pipeline_artifact_path(artifacts.get(key))
    return resolved


def pipeline_contract_descriptor_from_artifacts(artifacts: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = artifacts if isinstance(artifacts, dict) else {}
    missing_keys = [key for key in PIPELINE_KERNEL_REQUIRED_KEYS if not str(payload.get(key) or "").strip()]
    missing_paths = [
        key
        for key in PIPELINE_KERNEL_REQUIRED_KEYS
        if str(payload.get(key) or "").strip() and not Path(str(payload.get(key))).expanduser().exists()
    ]
    return {
        "source": "pipeline_kernel_contract_artifacts",
        "artifacts": dict(payload),
        "ready": not missing_keys and not missing_paths,
        "missing_keys": missing_keys,
        "missing_paths": missing_paths,
    }


def pipeline_contract_descriptor_from_report(report: Dict[str, Any] | None) -> Dict[str, Any]:
    artifacts = pipeline_kernel_contract_artifacts_from_report(report)
    return pipeline_contract_descriptor_from_artifacts(artifacts)
