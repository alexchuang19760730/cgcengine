#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cgc_engine.pipeline import MegatrainEightStepPipeline, MegatrainPipelineConfig


DEFAULT_CONFIG = (
    REPO_ROOT
    / "docs"
    / "technical_whitepapers"
    / "examples"
    / "deepseek_x4_minicpm5_fusionroute_system_manifest.example.json"
)


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid config payload: {path}")
    return payload


def _to_dtype(raw: str) -> torch.dtype:
    mapping = {
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "torch.bfloat16": torch.bfloat16,
    }
    return mapping.get(str(raw).strip(), torch.bfloat16)


def _build_component_config(system_payload: dict, component_payload: dict, export_root: Path) -> MegatrainPipelineConfig:
    required_components = [
        str(item.get("component_id") or "")
        for item in system_payload.get("components", [])
        if isinstance(item, dict) and bool(item.get("required")) and str(item.get("component_id") or "").strip()
    ]
    optional_components = [
        str(item.get("component_id") or "")
        for item in system_payload.get("components", [])
        if isinstance(item, dict) and not bool(item.get("required")) and str(item.get("component_id") or "").strip()
    ]
    export_subdir = str(component_payload.get("export_subdir") or component_payload.get("component_id") or "component")
    export_dir = export_root / export_subdir
    return MegatrainPipelineConfig(
        task_type=str(component_payload.get("task_type") or "inference"),
        backend=str(component_payload.get("backend") or "cuda"),
        environment=str(component_payload.get("environment") or "cloud_cluster"),
        task_domain=str(component_payload.get("task_domain") or "models"),
        model_name=str(component_payload.get("model_name") or ""),
        dtype=_to_dtype(str(component_payload.get("dtype") or "torch.bfloat16")),
        hf_model_path=str(component_payload.get("hf_model_path") or ""),
        load_weights=bool(component_payload.get("load_weights", True)),
        export_dir=str(export_dir),
        runtime_profile=str(component_payload.get("runtime_profile") or "cloud_cluster"),
        parallel_tp_size=int(component_payload.get("parallel_tp_size", 1) or 1),
        parallel_pp_size=int(component_payload.get("parallel_pp_size", 1) or 1),
        parallel_ep_size=int(component_payload.get("parallel_ep_size", 1) or 1),
        enable_nccl=bool(component_payload.get("enable_nccl", False)),
        component_id=str(component_payload.get("component_id") or ""),
        component_role=str(component_payload.get("component_role") or ""),
        component_required=bool(component_payload.get("required", True)),
        component_health_endpoint=str(component_payload.get("health_endpoint") or ""),
        system_id=str(system_payload.get("system_id") or ""),
        system_role=str(system_payload.get("system_role") or "multi_component_runtime"),
        system_manifest_autodiscover=True,
        system_manifest_discovery_root=str(export_root),
        system_manifest_routing_edges=json.dumps(system_payload.get("routing_edges") or [], ensure_ascii=False),
        system_manifest_required_components=json.dumps(required_components, ensure_ascii=False),
        system_manifest_optional_components=json.dumps(optional_components, ensure_ascii=False),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate example runtime contracts and system execution manifest.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to the example system manifest config JSON.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    system_payload = _load_json(config_path)
    export_root = Path(str(system_payload.get("export_root") or "")).expanduser()
    if not export_root.is_absolute():
        export_root = (REPO_ROOT / export_root).resolve()
    export_root.mkdir(parents=True, exist_ok=True)

    components = system_payload.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("config.components must be a non-empty list")

    reports: list[dict] = []
    for component_payload in components:
        if not isinstance(component_payload, dict):
            continue
        pipe = MegatrainEightStepPipeline(_build_component_config(system_payload, component_payload, export_root))
        report = pipe.materialize_contract_artifacts_only(device=torch.device("cuda:0"))
        reports.append(
            {
                "component_id": str(component_payload.get("component_id") or ""),
                "export_dir": str(component_payload.get("export_subdir") or component_payload.get("component_id") or ""),
                "contract_manifest_path": str((report.get("contract_manifest") or {}).get("artifact_path") or ""),
                "system_execution_manifest_path": str((report.get("system_execution_manifest") or {}).get("artifact_path") or ""),
            }
        )

    # Regenerate one final manifest from the orchestrator when available, so autodiscovery sees all siblings.
    orchestrator = next(
        (
            item for item in components
            if isinstance(item, dict) and str(item.get("component_role") or "").strip() == "gateway_orchestrator"
        ),
        components[0],
    )
    orchestrator_pipe = MegatrainEightStepPipeline(_build_component_config(system_payload, orchestrator, export_root))
    final_report = orchestrator_pipe.materialize_contract_artifacts_only(device=torch.device("cuda:0"))

    result = {
        "config_path": str(config_path),
        "export_root": str(export_root),
        "components": reports,
        "final_system_execution_manifest_path": str(
            (final_report.get("system_execution_manifest") or {}).get("artifact_path") or ""
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
