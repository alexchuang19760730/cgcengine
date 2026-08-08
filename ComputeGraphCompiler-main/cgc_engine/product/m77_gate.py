import json
from pathlib import Path
from typing import Any, Dict

from cgc_engine.product.upkg30_common import (
    artifact_index,
    build_gate_summary,
    derive_matrix_axes,
    evaluate_mandatory_protocol_gate,
    failure_attribution,
    incoming_upstream_contract,
    load_pipeline_report,
    prefer_upstream_pipeline_contract_descriptor,
    prefer_upstream_pipeline_kernel_contract_artifacts,
    pipeline_contract_descriptor,
    pipeline_kernel_contract_artifacts,
    resolve_runtime_protocol_projection,
    read_json,
    stage_trace_rows,
    upstream_gate_payload,
    write_json,
    write_jsonl,
)


def run_m77_gate(*, output_dir: str, cgc_report: Dict[str, Any] | None = None) -> Dict[str, Any]:
    output_root = Path(str(output_dir)).expanduser().resolve()
    m77_dir = (output_root / "m77_cloud_edge_q2rl").resolve()
    m77_dir.mkdir(parents=True, exist_ok=True)

    pipeline_report = load_pipeline_report(output_dir=output_root)
    m72_contract = incoming_upstream_contract(cgc_report if isinstance(cgc_report, dict) else {}, "m72")
    m72_gate = upstream_gate_payload(m72_contract)
    m72_dependency_source = "upstream_contracts.m72" if m72_gate else ""
    kernel_artifacts = prefer_upstream_pipeline_kernel_contract_artifacts(
        pipeline_kernel_contract_artifacts(output_dir=output_root, pipeline_report=pipeline_report),
        m72_gate,
    )
    kernel_contract = prefer_upstream_pipeline_contract_descriptor(
        pipeline_contract_descriptor(output_dir=output_root, pipeline_report=pipeline_report),
        m72_gate,
    )

    matrix_axes = derive_matrix_axes(
        milestone="m77",
        gate_name="3.7 Cloud-Edge Training And Inference Q2RL Gate",
        pipeline_report=pipeline_report,
        extra={
            "training_mode": "cloud_train_edge_infer",
            "post_training_method": "Q2RL",
            "contract_manifest_path": kernel_artifacts.get("contract_manifest_path") or "",
            "system_execution_manifest_path": kernel_artifacts.get("system_execution_manifest_path") or "",
        },
    )
    pipeline_contract_ok = bool(kernel_contract.get("ready"))
    upstream_m72_contract_ok = bool(m72_gate)
    runtime_protocol_projection = resolve_runtime_protocol_projection(
        contract_manifest_path=str(kernel_artifacts.get("contract_manifest_path") or ""),
        system_execution_manifest_path=str(kernel_artifacts.get("system_execution_manifest_path") or ""),
    )
    mandatory_protocol_gate = evaluate_mandatory_protocol_gate(
        runtime_protocol_contract=runtime_protocol_projection.get("runtime_protocol_contract"),
        zero_copy_vram_real=runtime_protocol_projection.get("zero_copy_vram_real"),
        source=str(kernel_artifacts.get("contract_manifest_path") or kernel_artifacts.get("system_execution_manifest_path") or ""),
    )
    q2rl_paths = m72_gate.get("cloud_edge_q2rl_artifacts") if isinstance(m72_gate.get("cloud_edge_q2rl_artifacts"), dict) else {}
    cloud_return_paths = m72_gate.get("cloud_return_artifacts") if isinstance(m72_gate.get("cloud_return_artifacts"), dict) else {}
    required = {
        "cloud_edge_training_inference_mode": str(q2rl_paths.get("cloud_edge_training_inference_mode_path") or ""),
        "gui_agent_edge_inference_contract": str(q2rl_paths.get("gui_agent_edge_inference_contract_path") or ""),
        "q2rl_post_training_profile": str(q2rl_paths.get("q2rl_post_training_profile_path") or ""),
        "edge_deployment_bundle_manifest": str(q2rl_paths.get("edge_deployment_bundle_manifest_path") or ""),
        "cloud_edge_q2rl_evaluation_plan": str(q2rl_paths.get("cloud_edge_q2rl_evaluation_plan_path") or ""),
        "edge_inference_result": str(cloud_return_paths.get("edge_inference_result_path") or ""),
        "replay_anchor": str(cloud_return_paths.get("replay_anchor_path") or ""),
        "reward_trace": str(cloud_return_paths.get("reward_trace_path") or ""),
        "cloud_ingest_manifest": str(cloud_return_paths.get("cloud_ingest_manifest_path") or ""),
        "cloud_summary": str(cloud_return_paths.get("cloud_summary_path") or ""),
    }
    stage_status: Dict[str, Dict[str, Any]] = {}
    stage_status["upstream_m72_contract"] = {
        "status": "PASS" if upstream_m72_contract_ok else "FAIL",
        "reason": "" if upstream_m72_contract_ok else "missing_upstream_contract.m72",
    }
    stage_status["pipeline_contract_artifacts"] = {
        "status": "PASS" if pipeline_contract_ok else "FAIL",
        "reason": "" if pipeline_contract_ok else "pipeline_kernel_contract_artifacts_not_ready",
    }
    stage_status["mandatory_protocol_gate"] = mandatory_protocol_gate
    for name, path_str in required.items():
        exists = Path(path_str).exists() if path_str else False
        stage_status[name] = {
            "status": "PASS" if exists else "FAIL",
            "reason": "" if exists else f"missing_artifact:{name}",
            "path": path_str,
        }

    q2rl_register_path = str(q2rl_paths.get("cloud_edge_q2rl_register_path") or "")
    q2rl_register_exists = Path(q2rl_register_path).exists() if q2rl_register_path else False
    stage_status["cloud_edge_q2rl_register"] = {
        "status": "PASS" if q2rl_register_exists else "FAIL",
        "reason": "" if q2rl_register_exists else "missing_artifact:cloud_edge_q2rl_register",
        "path": q2rl_register_path,
    }

    stage_rows = stage_trace_rows(gate_name="m77", stage_status=stage_status)
    stage_trace_path = write_jsonl(m77_dir / "stage_trace.jsonl", stage_rows)
    ok = upstream_m72_contract_ok and all(str(payload.get("status") or "") == "PASS" for payload in stage_status.values())

    gate = {
        "status": "PASS" if ok else "FAIL",
        "matrix_axes": matrix_axes,
        "mandatory_protocol_gate": mandatory_protocol_gate,
        "cloud_edge_q2rl_artifacts": q2rl_paths,
        "cloud_return_artifacts": cloud_return_paths,
        "cloud_single_source_path": str(cloud_return_paths.get("cloud_summary_path") or ""),
        "pipeline_kernel_contract_artifacts": kernel_artifacts,
        "pipeline_contract_descriptor": kernel_contract,
        "upkg30": {
            "3.4_unified_artifact_and_summary": {
                "status": "PASS" if ok else "FAIL",
                "single_source_mode": "cloud_aggregated",
                "primary_path": str(cloud_return_paths.get("cloud_summary_path") or ""),
            },
            "3.7_cloud_edge_training_inference_q2rl": {"status": "PASS" if ok else "FAIL"},
        },
    }
    gate["failure_attribution"] = failure_attribution(
        gate_name="m77",
        status=gate["status"],
        stage_status={**stage_status, **gate["upkg30"]},
    )
    artifact_entries = artifact_index([*required.values(), q2rl_register_path, str(stage_trace_path), *kernel_artifacts.values()])
    artifact_index_path = write_json(m77_dir / "artifact_index.json", {"artifacts": artifact_entries})
    gate["artifact_index"] = artifact_entries
    gate["artifact_index_path"] = str(artifact_index_path)
    gate["stage_trace_path"] = str(stage_trace_path)

    report_payload = {
        "name": "CGC_M7.7_Cloud_Edge_Training_Inference_Q2RL_Gate",
        "status": gate["status"],
        "scope": "verification_only",
        "public_entrypoint": "cgc gate m77",
        "matrix_axes": matrix_axes,
        "artifact_index": artifact_entries,
        "artifact_index_path": str(artifact_index_path),
        "stage_trace_path": str(stage_trace_path),
        "failure_attribution": gate["failure_attribution"],
        "pipeline_kernel_contract_artifacts": kernel_artifacts,
        "pipeline_contract_descriptor": kernel_contract,
        "root_pipeline_report_path": str(pipeline_report.get("pipeline_root_report_path") or ""),
        "root_pipeline_report_contract_source": str(pipeline_report.get("pipeline_root_report_contract_source") or ""),
        "m72_dependency_source": str(m72_dependency_source),
        "upstream_m72_report_path": str(m72_contract.get("report_path") or ""),
        "upstream_m72_summary_path": str(m72_contract.get("summary_path") or ""),
        "mandatory_protocol_gate": mandatory_protocol_gate,
        "cloud_edge_q2rl_artifacts": q2rl_paths,
        "cloud_return_artifacts": cloud_return_paths,
        "single_source_of_truth": {
            "mode": "cloud_aggregated",
            "primary_path": str(cloud_return_paths.get("cloud_summary_path") or ""),
            "cloud_ingest_manifest_path": str(cloud_return_paths.get("cloud_ingest_manifest_path") or ""),
            "contract_manifest_path": str(kernel_artifacts.get("contract_manifest_path") or ""),
            "system_execution_manifest_path": str(kernel_artifacts.get("system_execution_manifest_path") or ""),
        },
        "upkg30": gate["upkg30"],
        "gate_result": {"m77": gate},
    }
    report_path = write_json(m77_dir / "m77_report.json", report_payload)
    summary_payload = build_gate_summary(
        gate_name="m77",
        milestone="m77",
        status=gate["status"],
        matrix_axes=matrix_axes,
        report_path=report_path,
        artifact_entries=artifact_entries,
        stage_rows=stage_rows,
        failure=gate["failure_attribution"],
    )
    summary_path = write_json(m77_dir / "summary.json", summary_payload)
    report_payload["summary_path"] = str(summary_path)
    write_json(report_path, report_payload)
    return {
        "ok": ok,
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "gate_result": {"m77": gate},
    }


def run_upkg37_gate(*, output_dir: str, cgc_report: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return run_m77_gate(output_dir=output_dir, cgc_report=cgc_report)
