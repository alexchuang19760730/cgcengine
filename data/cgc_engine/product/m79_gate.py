from pathlib import Path
from typing import Any, Dict

from cgc_engine.product.upkg30_common import (
    artifact_index,
    build_gate_summary,
    derive_matrix_axes,
    failure_attribution,
    read_json,
    stage_trace_rows,
    write_json,
    write_jsonl,
)
from cgc_engine.product.upkg40_common import write_upkg40_embodied_artifacts


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def run_m79_gate(*, output_dir: str, cgc_report: Dict[str, Any] | None = None) -> Dict[str, Any]:
    output_root = Path(str(output_dir)).expanduser().resolve()
    m79_dir = (output_root / "m79_embodied_upkg40").resolve()
    m79_dir.mkdir(parents=True, exist_ok=True)

    pipeline_report = cgc_report if isinstance(cgc_report, dict) else {}
    pipeline_report_path = (output_root / "report.json").resolve()
    if not pipeline_report and pipeline_report_path.exists():
        pipeline_report = read_json(pipeline_report_path)

    m73_report_path = (output_root / "m73_physical" / "m73_report.json").resolve()
    m73_report: Dict[str, Any] = {}
    if m73_report_path.exists():
        m73_report = read_json(m73_report_path)
    m73_gate: Dict[str, Any] = {}
    if isinstance(m73_report, dict):
        m73_gate = ((m73_report.get("gate_result") or {}).get("m73") or {}) if isinstance(m73_report.get("gate_result"), dict) else {}
    if not m73_gate:
        m73_gate = ((pipeline_report.get("gate_result") or {}).get("m73") or {}) if isinstance(pipeline_report, dict) else {}

    m73_summary_path = str(m73_gate.get("summary_path") or "")
    matrix_axes = derive_matrix_axes(
        milestone="m79",
        gate_name="UPKG 4.0 Psi0 Cloud Training And Realtime-VLA Edge Inference Gate",
        pipeline_report=pipeline_report,
        extra={
            "training_model": "psi0",
            "edge_runtime_host": "realtime-vla-v2",
            "comparative_target": "official_psi0_train_plus_infer",
        },
    )

    artifact_paths = write_upkg40_embodied_artifacts(
        gate_dir=m79_dir,
        gate_name="m79",
        matrix_axes=matrix_axes,
        m73_gate=m73_gate,
        m73_report_path=str(m73_report_path),
        m73_summary_path=str(m73_summary_path),
    )

    required = {
        "psi0_cloud_training_contract": str(artifact_paths.get("psi0_cloud_training_contract_path") or ""),
        "realtime_vla_edge_inference_contract": str(artifact_paths.get("realtime_vla_edge_inference_contract_path") or ""),
        "embodied_teaching_session": str(artifact_paths.get("embodied_teaching_session_path") or ""),
        "embodied_training_dataset_manifest": str(artifact_paths.get("embodied_training_dataset_manifest_path") or ""),
        "embodied_trained_model_manifest": str(artifact_paths.get("embodied_trained_model_manifest_path") or ""),
        "embodied_inference_session": str(artifact_paths.get("embodied_inference_session_path") or ""),
        "embodied_audit_session": str(artifact_paths.get("embodied_audit_session_path") or ""),
        "embodied_replay_session": str(artifact_paths.get("embodied_replay_session_path") or ""),
        "embodied_trace_session": str(artifact_paths.get("embodied_trace_session_path") or ""),
        "psi0_official_duration_baseline": str(artifact_paths.get("psi0_official_duration_baseline_path") or ""),
        "psi0_realtime_vla_comparative": str(artifact_paths.get("psi0_realtime_vla_comparative_path") or ""),
        "psi0_realtime_vla_benchmark": str(artifact_paths.get("psi0_realtime_vla_benchmark_path") or ""),
        "embodied_parity_report": str(artifact_paths.get("embodied_parity_report_path") or ""),
        "cloud_summary": str(artifact_paths.get("cloud_summary_path") or ""),
    }

    stage_status: Dict[str, Dict[str, Any]] = {
        "m73_foundation": {
            "status": "PASS" if str(m73_gate.get("status") or "") == "PASS" else "FAIL",
            "reason": "" if str(m73_gate.get("status") or "") == "PASS" else "m73_not_pass",
            "path": str(m73_report_path),
        }
    }
    for name, path_str in required.items():
        exists = Path(path_str).exists() if path_str else False
        stage_status[name] = {
            "status": "PASS" if exists else "FAIL",
            "reason": "" if exists else f"missing_artifact:{name}",
            "path": path_str,
        }

    parity_report = read_json(Path(str(artifact_paths.get("embodied_parity_report_path") or ""))) if str(artifact_paths.get("embodied_parity_report_path") or "") else {}
    stage_status["upkg3x_capability_parity"] = {
        "status": "PASS" if str(parity_report.get("status") or "FAIL") == "PASS" else "FAIL",
        "reason": "" if str(parity_report.get("status") or "FAIL") == "PASS" else "upkg3x_capability_parity_missing",
        "path": str(artifact_paths.get("embodied_parity_report_path") or ""),
    }

    benchmark_report = read_json(Path(str(artifact_paths.get("psi0_realtime_vla_benchmark_path") or ""))) if str(artifact_paths.get("psi0_realtime_vla_benchmark_path") or "") else {}
    stage_status["duration_acceleration_gate"] = {
        "status": "PASS" if str(benchmark_report.get("status") or "FAIL") == "PASS" else "FAIL",
        "reason": "" if str(benchmark_report.get("status") or "FAIL") == "PASS" else str(benchmark_report.get("reason") or "duration_ratio_above_threshold"),
        "path": str(artifact_paths.get("psi0_realtime_vla_benchmark_path") or ""),
    }

    stage_rows = stage_trace_rows(gate_name="m79", stage_status=stage_status)
    stage_trace_path = write_jsonl(m79_dir / "stage_trace.jsonl", stage_rows)
    ok = all(str(payload.get("status") or "") == "PASS" for payload in stage_status.values())

    gate = {
        "status": "PASS" if ok else "FAIL",
        "matrix_axes": matrix_axes,
        "training_model": "psi0",
        "edge_runtime_host": "realtime-vla-v2",
        "upkg40_artifacts": artifact_paths,
        "benchmark_summary": {
            "duration_ratio": _float_or_none(benchmark_report.get("duration_ratio")),
            "speedup": _float_or_none(benchmark_report.get("speedup")),
            "threshold_ratio_max": float(benchmark_report.get("benchmark_threshold_ratio_max") or 0.2),
            "threshold_speedup_min": float(benchmark_report.get("benchmark_threshold_speedup_min") or 5.0),
            "status": str(benchmark_report.get("status") or "FAIL"),
            "reason": str(benchmark_report.get("reason") or ""),
        },
        "upkg40": {
            "4.0_embodied_runtime_comparative_benchmark": {
                "status": "PASS" if ok else "FAIL",
                "training_model": "psi0",
                "edge_runtime_host": "realtime-vla-v2",
                "cloud_single_source_path": str(artifact_paths.get("cloud_summary_path") or ""),
            }
        },
    }
    gate["failure_attribution"] = failure_attribution(
        gate_name="m79",
        status=gate["status"],
        stage_status={**stage_status, **gate["upkg40"]},
    )

    artifact_entries = artifact_index([*required.values(), str(stage_trace_path)])
    artifact_index_path = write_json(m79_dir / "artifact_index.json", {"artifacts": artifact_entries})
    gate["artifact_index"] = artifact_entries
    gate["artifact_index_path"] = str(artifact_index_path)
    gate["stage_trace_path"] = str(stage_trace_path)

    report_payload = {
        "name": "CGC_UPKG40_Psi0_Cloud_Training_And_Realtime_VLA_Edge_Inference_Gate",
        "status": gate["status"],
        "scope": "verification_only",
        "public_entrypoint": "cgc gate upkg40",
        "matrix_axes": matrix_axes,
        "artifact_index": artifact_entries,
        "artifact_index_path": str(artifact_index_path),
        "stage_trace_path": str(stage_trace_path),
        "failure_attribution": gate["failure_attribution"],
        "benchmark_summary": gate["benchmark_summary"],
        "upkg40_artifacts": artifact_paths,
        "upkg40": gate["upkg40"],
        "gate_result": {"m79": gate},
    }
    report_path = write_json(m79_dir / "m79_report.json", report_payload)
    summary_payload = build_gate_summary(
        gate_name="m79",
        milestone="m79",
        status=gate["status"],
        matrix_axes=matrix_axes,
        report_path=report_path,
        artifact_entries=artifact_entries,
        stage_rows=stage_rows,
        failure=gate["failure_attribution"],
    )
    summary_path = write_json(m79_dir / "summary.json", summary_payload)
    report_payload["summary_path"] = str(summary_path)
    write_json(report_path, report_payload)
    return {
        "ok": ok,
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "gate_result": {"m79": gate},
    }


def run_upkg40_gate(*, output_dir: str, cgc_report: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return run_m79_gate(output_dir=output_dir, cgc_report=cgc_report)
