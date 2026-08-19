import json
import os
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


def _m78_alignment_threshold() -> float:
    raw = str(os.environ.get("CGC_M78_ALIGNMENT_THRESHOLD") or "0.5").strip()
    try:
        value = float(raw)
    except Exception:
        value = 0.5
    return max(0.0, min(1.0, value))


def run_m78_gate(*, output_dir: str, cgc_report: Dict[str, Any] | None = None) -> Dict[str, Any]:
    output_root = Path(str(output_dir)).expanduser().resolve()
    m78_dir = (output_root / "m78_teaching_pure_llm").resolve()
    m78_dir.mkdir(parents=True, exist_ok=True)

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
        milestone="m78",
        gate_name="3.8 Teaching Mode And Pure LLM Six-Element Inference Gate",
        pipeline_report=pipeline_report,
        extra={
            "teaching_mode": "gui_agent_demonstration",
            "inference_mode": "pure_llm_six_element",
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
    artifact_paths = m72_gate.get("teaching_inference_visualization_artifacts") if isinstance(m72_gate.get("teaching_inference_visualization_artifacts"), dict) else {}
    cloud_return_paths = m72_gate.get("cloud_return_artifacts") if isinstance(m72_gate.get("cloud_return_artifacts"), dict) else {}
    required = {
        "teaching_mode_contract": str(artifact_paths.get("teaching_mode_contract_path") or ""),
        "teaching_dataset_manifest": str(artifact_paths.get("teaching_dataset_manifest_path") or ""),
        "teaching_trained_model_manifest": str(artifact_paths.get("teaching_trained_model_manifest_path") or ""),
        "q2rl_training_report": str(artifact_paths.get("q2rl_training_report_path") or ""),
        "edge_inference_push_contract": str(artifact_paths.get("edge_inference_push_contract_path") or ""),
        "llm_six_element_inference_mode": str(artifact_paths.get("llm_six_element_inference_mode_path") or ""),
        "teaching_alignment_report": str(artifact_paths.get("teaching_alignment_report_path") or ""),
        "teaching_vs_inference_graph": str(artifact_paths.get("teaching_vs_inference_graph_path") or ""),
        "teaching_optimization_triplet_comparison": str(artifact_paths.get("teaching_optimization_triplet_comparison_path") or ""),
        "triplet_comparison_mmd": str(artifact_paths.get("triplet_comparison_mmd_path") or ""),
        "triplet_comparison_html": str(artifact_paths.get("triplet_comparison_html_path") or ""),
        "before_vs_after_vs_teaching_chart": str(artifact_paths.get("before_vs_after_vs_teaching_chart_path") or ""),
        "teaching_optimization_audit_replay_bundle": str(artifact_paths.get("teaching_optimization_audit_replay_bundle_path") or ""),
        "graph_error_visualization": str(artifact_paths.get("graph_error_visualization_path") or ""),
        "graph_error_visualization_mmd": str(artifact_paths.get("graph_error_visualization_mmd_path") or ""),
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

    alignment_report_path = str(artifact_paths.get("teaching_alignment_report_path") or "")
    alignment_report = read_json(Path(alignment_report_path)) if alignment_report_path else {}
    if alignment_report_path:
        m78_alignment_threshold = _m78_alignment_threshold()
        alignment_score = float(alignment_report.get("alignment_score") or 0.0)
        missing_elements = (
            alignment_report.get("missing_six_elements")
            if isinstance(alignment_report.get("missing_six_elements"), list)
            else []
        )
        alignment_pass = alignment_score >= m78_alignment_threshold and not missing_elements
        stage_status["alignment_threshold"] = {
            "status": "PASS" if alignment_pass else "FAIL",
            "reason": "" if alignment_pass else "teaching_alignment_below_threshold",
            "path": alignment_report_path,
            "alignment_score": alignment_score,
            "target_threshold": m78_alignment_threshold,
            "report_target_threshold": float(alignment_report.get("target_threshold") or 0.0),
        }

    q2rl_training_report_path = str(artifact_paths.get("q2rl_training_report_path") or "")
    q2rl_training_report = read_json(Path(q2rl_training_report_path)) if q2rl_training_report_path else {}
    if q2rl_training_report_path:
        pre_metrics = q2rl_training_report.get("pre_q2rl_metrics") if isinstance(q2rl_training_report.get("pre_q2rl_metrics"), dict) else {}
        post_metrics = q2rl_training_report.get("post_q2rl_metrics") if isinstance(q2rl_training_report.get("post_q2rl_metrics"), dict) else {}
        pre_reward = float(pre_metrics.get("reward_score") or 0.0)
        post_reward = float(post_metrics.get("reward_score") or 0.0)
        pre_alignment = float(pre_metrics.get("alignment_score") or 0.0)
        post_alignment = float(post_metrics.get("alignment_score") or 0.0)
        improved = post_reward > pre_reward and post_alignment >= pre_alignment
        stage_status["q2rl_optimization_delta"] = {
            "status": "PASS" if improved else "FAIL",
            "reason": "" if improved else "q2rl_metrics_not_improved",
            "path": q2rl_training_report_path,
            "target_model_id": str(q2rl_training_report.get("target_model_id") or ""),
        }

    stage_rows = stage_trace_rows(gate_name="m78", stage_status=stage_status)
    stage_trace_path = write_jsonl(m78_dir / "stage_trace.jsonl", stage_rows)
    ok = upstream_m72_contract_ok and all(str(payload.get("status") or "") == "PASS" for payload in stage_status.values())

    gate = {
        "status": "PASS" if ok else "FAIL",
        "matrix_axes": matrix_axes,
        "mandatory_protocol_gate": mandatory_protocol_gate,
        "m78_alignment_threshold": _m78_alignment_threshold(),
        "teaching_inference_visualization_artifacts": artifact_paths,
        "cloud_single_source_path": str(cloud_return_paths.get("cloud_summary_path") or ""),
        "target_model_id": str(q2rl_training_report.get("target_model_id") or "bytedance-research/UI-TARS-2B-SFT"),
        "target_model_source_path": str(q2rl_training_report.get("base_model_source_path") or ""),
        "target_model_source_resolution_path": str(q2rl_training_report.get("base_model_source_resolution_path") or ""),
        "pipeline_kernel_contract_artifacts": kernel_artifacts,
        "pipeline_contract_descriptor": kernel_contract,
        "upkg30": {
            "3.4_unified_artifact_and_summary": {
                "status": "PASS" if ok else "FAIL",
                "single_source_mode": "cloud_aggregated",
                "primary_path": str(cloud_return_paths.get("cloud_summary_path") or ""),
            },
            "3.8_teaching_and_pure_llm_six_element_inference": {
                "status": "PASS" if ok else "FAIL",
                "target_model_id": str(q2rl_training_report.get("target_model_id") or "bytedance-research/UI-TARS-2B-SFT"),
                "target_model_source_path": str(q2rl_training_report.get("base_model_source_path") or ""),
                "target_model_source_resolution_path": str(q2rl_training_report.get("base_model_source_resolution_path") or ""),
            },
        },
    }
    gate["failure_attribution"] = failure_attribution(
        gate_name="m78",
        status=gate["status"],
        stage_status={**stage_status, **gate["upkg30"]},
    )
    artifact_entries = artifact_index([*required.values(), str(stage_trace_path), *kernel_artifacts.values()])
    artifact_index_path = write_json(m78_dir / "artifact_index.json", {"artifacts": artifact_entries})
    gate["artifact_index"] = artifact_entries
    gate["artifact_index_path"] = str(artifact_index_path)
    gate["stage_trace_path"] = str(stage_trace_path)

    report_payload = {
        "name": "CGC_M7.8_Teaching_Mode_And_Pure_LLM_Six_Element_Inference_Gate",
        "status": gate["status"],
        "scope": "verification_only",
        "public_entrypoint": "cgc gate m78",
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
        "teaching_inference_visualization_artifacts": artifact_paths,
        "single_source_of_truth": {
            "mode": "cloud_aggregated",
            "primary_path": str(cloud_return_paths.get("cloud_summary_path") or ""),
            "contract_manifest_path": str(kernel_artifacts.get("contract_manifest_path") or ""),
            "system_execution_manifest_path": str(kernel_artifacts.get("system_execution_manifest_path") or ""),
        },
        "upkg30": gate["upkg30"],
        "gate_result": {"m78": gate},
    }
    report_path = write_json(m78_dir / "m78_report.json", report_payload)
    summary_payload = build_gate_summary(
        gate_name="m78",
        milestone="m78",
        status=gate["status"],
        matrix_axes=matrix_axes,
        report_path=report_path,
        artifact_entries=artifact_entries,
        stage_rows=stage_rows,
        failure=gate["failure_attribution"],
    )
    summary_path = write_json(m78_dir / "summary.json", summary_payload)
    report_payload["summary_path"] = str(summary_path)
    write_json(report_path, report_payload)
    return {
        "ok": ok,
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "gate_result": {"m78": gate},
    }


def run_upkg38_gate(*, output_dir: str, cgc_report: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return run_m78_gate(output_dir=output_dir, cgc_report=cgc_report)
