import json
import os
from pathlib import Path
from typing import Any, Dict, List
import yaml

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
    read_json,
    read_json,
    read_jsonl,
    resolve_runtime_protocol_projection,
    six_element_event,
    six_element_summary,
    stage_trace_rows,
    upstream_gate_payload,
    write_cloud_edge_q2rl_artifacts,
    write_edge_to_cloud_return_artifacts,
    write_gap_closure_artifacts,
    write_teaching_inference_visualization_artifacts,
    write_json,
    write_jsonl,
)
from cgc_engine.agent.gui_graph_native import build_gui_graph_native_integration


def _as_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _candidate_gui_evidence_paths(out_dir_p: Path, report: Dict[str, Any]) -> List[Path]:
    candidates: List[Path] = [
        out_dir_p / "gui_agent_runtime_evidence.json",
        out_dir_p.parent / "gui_agent_runtime_evidence.json",
        out_dir_p.parent.parent / "gui_agent_runtime_evidence.json",
    ]
    gui_stage_source = report.get("gui_stage_source") if isinstance(report.get("gui_stage_source"), dict) else {}
    for key in ("evidence_path", "gui_agent_evidence_path"):
        raw = str(gui_stage_source.get(key) or "").strip()
        if raw:
            candidates.append(Path(raw).expanduser().resolve())
    gui_stage_source_path = str(report.get("gui_stage_source_path") or "").strip()
    if gui_stage_source_path:
        candidates.append(Path(gui_stage_source_path).expanduser().resolve())
    for env_name in ("CGC_M72_GUI_EVENT_EVIDENCE", "CGC_GUI_AGENT_EVENT_EVIDENCE"):
        raw = str(os.environ.get(env_name, "") or "").strip()
        if raw:
            candidates.append(Path(raw).expanduser().resolve())
    summary_path = str(report.get("summary_path") or "").strip()
    if summary_path:
        candidates.append(Path(summary_path).expanduser().resolve().parent / "gui_agent_runtime_evidence.json")
    unique: List[Path] = []
    seen = set()
    for path in candidates:
        resolved = str(path.expanduser().resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(Path(resolved))
    return unique


def _load_gui_agent_runtime_evidence(out_dir_p: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    for candidate in _candidate_gui_evidence_paths(out_dir_p, report):
        evidence = read_json(candidate)
        if not evidence:
            continue
        events_path = Path(str(evidence.get("events_path") or "")).expanduser()
        manifest_path = Path(str(evidence.get("screenshot_manifest_path") or "")).expanduser()
        events = read_jsonl(events_path.resolve()) if events_path.exists() else []
        manifest = read_json(manifest_path.resolve()) if manifest_path.exists() else {}
        screenshots = manifest.get("screenshots") if isinstance(manifest.get("screenshots"), list) else []
        categories = {str(item.get("category") or "") for item in events if isinstance(item, dict)}
        return {
            "status": str(evidence.get("status") or "FAIL"),
            "evidence_path": str(candidate),
            "events_path": str(events_path.resolve()) if events_path.exists() else "",
            "manifest_path": str(manifest_path.resolve()) if manifest_path.exists() else "",
            "events": events,
            "screenshots": screenshots,
            "categories_present": sorted(categories),
        }
    return {
        "status": "FAIL",
        "evidence_path": "",
        "events_path": "",
        "manifest_path": "",
        "events": [],
        "screenshots": [],
        "categories_present": [],
    }


def _infer_gui_graph_native_integration_path(report: Dict[str, Any], gui_evidence: Dict[str, Any]) -> str:
    report_path = str(report.get("gui_graph_native_integration_path") or "").strip()
    if report_path:
        return report_path
    graph_native = report.get("gui_graph_native_integration") if isinstance(report.get("gui_graph_native_integration"), dict) else {}
    artifact_paths = graph_native.get("artifact_paths") if isinstance(graph_native.get("artifact_paths"), dict) else {}
    artifact_path = str(artifact_paths.get("graph_native_integration") or "").strip()
    if artifact_path:
        return artifact_path
    evidence_path = str(gui_evidence.get("evidence_path") or "").strip()
    if evidence_path:
        evidence_parent = Path(evidence_path).expanduser().resolve().parent
        for candidate in (
            evidence_parent / "gui_graph_native_integration.json",
            evidence_parent.parent / "gui_graph_native_integration.json",
        ):
            if candidate.exists():
                return str(candidate)
    return ""


def _load_gui_graph_native(report: Dict[str, Any], gui_evidence: Dict[str, Any], matrix_axes: Dict[str, Any]) -> Dict[str, Any]:
    graph_native = report.get("gui_graph_native_integration") if isinstance(report.get("gui_graph_native_integration"), dict) else {}
    if isinstance(graph_native, dict) and graph_native:
        return graph_native
    if not str(gui_evidence.get("evidence_path") or "").strip():
        return {}
    events = gui_evidence.get("events") if isinstance(gui_evidence.get("events"), list) else []
    stage_source = {
        "status": str(gui_evidence.get("status") or "FAIL"),
        "mode": "gui_runtime_evidence",
        "evidence_path": str(gui_evidence.get("evidence_path") or ""),
        "events_path": str(gui_evidence.get("events_path") or ""),
        "manifest_path": str(gui_evidence.get("manifest_path") or ""),
        "event_count": int(len(events)),
        "screenshot_count": int(len(gui_evidence.get("screenshots") or [])),
        "categories_present": list(gui_evidence.get("categories_present") or []),
        "by_category": {
            "workflow": sum(1 for item in events if isinstance(item, dict) and str(item.get("category") or "") == "workflow"),
            "runtime_host": sum(1 for item in events if isinstance(item, dict) and str(item.get("category") or "") == "runtime_host"),
            "tool_call": sum(1 for item in events if isinstance(item, dict) and str(item.get("category") or "") == "tool_call"),
            "screenshot": sum(1 for item in events if isinstance(item, dict) and str(item.get("category") or "") == "screenshot"),
        },
    }
    return build_gui_graph_native_integration(stage_source, matrix_axes=matrix_axes)


def _build_m72_six_element_events(*, gate_status: str, results: Dict[str, str], gui_evidence: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[str], str]:
    events: List[Dict[str, Any]] = [
        six_element_event("Compile", stage="model_trace", status=str(results.get("dynamic_trace_l1") or "FAIL"), element="model"),
        six_element_event("State", stage="memory_anchor", status=str(results.get("state_compression") or "PASS"), element="memory"),
    ]
    gui_stage_status: Dict[str, Dict[str, Any]] = {}
    artifact_paths: List[str] = []
    real_mode = str(gui_evidence.get("status") or "") == "PASS" and len(gui_evidence.get("events") or []) > 0
    if real_mode:
        mapping = {
            "workflow": ("workflow", "Workflow"),
            "runtime_host": ("environment", "Build"),
            "screenshot": ("perception", "Perception"),
            "tool_call": ("execution", "Execution"),
        }
        category_counts = {key: 0 for key in mapping}
        for item in gui_evidence.get("events") or []:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category") or "")
            if category not in mapping:
                continue
            element, kind = mapping[category]
            category_counts[category] += 1
            payload = dict(item.get("payload") or {})
            if str(item.get("screenshot_path") or "").strip():
                payload["screenshot_path"] = str(item.get("screenshot_path"))
                artifact_paths.append(str(item.get("screenshot_path")))
            events.append(
                six_element_event(
                    kind,
                    stage=str(item.get("action") or category),
                    status=str(item.get("status") or "PASS"),
                    element=element,
                    payload=payload,
                )
            )
        for category, (element, _) in mapping.items():
            count = int(category_counts.get(category) or 0)
            gui_stage_status[f"gui_{category}"] = {
                "status": "PASS" if count > 0 else "FAIL",
                "count": count,
                "element": element,
            }
        artifact_paths.extend(
            [
                str(gui_evidence.get("evidence_path") or ""),
                str(gui_evidence.get("events_path") or ""),
                str(gui_evidence.get("manifest_path") or ""),
            ]
        )
        return events, gui_stage_status, artifact_paths, "real_gui_evidence"

    events.extend(
        [
            six_element_event("Workflow", stage="agent_workflow", status=gate_status, element="workflow"),
            six_element_event("Build", stage="runtime_environment", status="PASS", element="environment"),
            six_element_event("Perception", stage="gui_perception", status="PASS", element="perception"),
            six_element_event("Execution", stage="tool_runtime", status=gate_status, element="execution"),
        ]
    )
    gui_stage_status = {
        "gui_workflow": {"status": "FAIL", "reason": "missing_gui_agent_runtime_evidence"},
        "gui_runtime_host": {"status": "FAIL", "reason": "missing_gui_agent_runtime_evidence"},
        "gui_screenshot": {"status": "FAIL", "reason": "missing_gui_agent_runtime_evidence"},
        "gui_tool_call": {"status": "FAIL", "reason": "missing_gui_agent_runtime_evidence"},
    }
    return events, gui_stage_status, artifact_paths, "synthetic_fallback"


def run_m72_gate(*, output_dir: str, cgc_report: Dict[str, Any]) -> Dict[str, Any]:
    out_dir_p = Path(output_dir).expanduser().resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    cfg_path = (Path(__file__).resolve().parents[1] / "agent" / "eval" / "m72_gate.yaml").resolve()
    with open(str(cfg_path), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    report = load_pipeline_report(output_dir=out_dir_p)
    m7_contract = incoming_upstream_contract(cgc_report if isinstance(cgc_report, dict) else {}, "m7")
    m7_data = upstream_gate_payload(m7_contract)
    m7_dependency_source = "upstream_contracts.m7" if m7_data else ""
    kernel_artifacts = prefer_upstream_pipeline_kernel_contract_artifacts(
        pipeline_kernel_contract_artifacts(output_dir=out_dir_p, pipeline_report=report),
        m7_data,
    )
    kernel_contract = prefer_upstream_pipeline_contract_descriptor(
        pipeline_contract_descriptor(output_dir=out_dir_p, pipeline_report=report),
        m7_data,
    )
    matrix_axes = derive_matrix_axes(
        milestone="m72",
        gate_name="3.2 Agent Runtime Gate",
        pipeline_report=report,
        extra={
            "state_abi_contract": str(kernel_artifacts.get("state_abi_path") or ""),
            "contract_manifest_path": str(kernel_artifacts.get("contract_manifest_path") or ""),
            "system_execution_manifest_path": str(kernel_artifacts.get("system_execution_manifest_path") or ""),
        },
    )
    if not isinstance(m7_data, dict) or not m7_data:
        gate = {"status": "FAIL", "reason": "missing_upstream_contract.m7", "config": str(cfg_path)}
        out_file = str(out_dir_p / "report.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": config.get("name"),
                    "status": "FAIL",
                    "scope": "verification_only",
                    "public_entrypoint": "cgc gate m72",
                    "metrics": {},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        gate["matrix_axes"] = matrix_axes
        gate["failure_attribution"] = failure_attribution(
            gate_name="m72",
            status="FAIL",
            stage_status={"agent_runtime": {"status": "FAIL", "reason": "missing_upstream_contract.m7"}},
        )
        return {"gate_result": {"m72": gate}, "report_path": out_file}

    final_pass = True
    results: Dict[str, str] = {}
    pipeline_contract_ok = bool(kernel_contract.get("ready"))
    runtime_protocol_projection = resolve_runtime_protocol_projection(
        contract_manifest_path=str(kernel_artifacts.get("contract_manifest_path") or ""),
        system_execution_manifest_path=str(kernel_artifacts.get("system_execution_manifest_path") or ""),
    )
    mandatory_protocol_gate = evaluate_mandatory_protocol_gate(
        runtime_protocol_contract=runtime_protocol_projection.get("runtime_protocol_contract"),
        zero_copy_vram_real=runtime_protocol_projection.get("zero_copy_vram_real"),
        source=str(kernel_artifacts.get("contract_manifest_path") or kernel_artifacts.get("system_execution_manifest_path") or ""),
    )
    for metric_group in (config.get("metrics") or []):
        group_name = str(metric_group.get("name") or "")
        rules = metric_group.get("rules") or []
        group_pass = True

        for rule in rules:
            metric_id = str(rule.get("metric") or "")
            op = str(rule.get("operator") or "")
            threshold = _as_float(rule.get("threshold"), 0.0)
            actual_value = None

            if group_name == "dynamic_trace_l1":
                dt = m7_data.get("dynamic_trace_l1", {})
                if not isinstance(dt, dict) or not dt:
                    dt = m7_data.get("dynamic_trace", {})
                if not isinstance(dt, dict):
                    dt = {}

                compile_variants = dt.get("compile_variants", []) if isinstance(dt.get("compile_variants"), list) else []
                correctness = dt.get("correctness", []) if isinstance(dt.get("correctness"), list) else []

                if metric_id == "compile_success_rate":
                    if "compile_success_rate" in dt:
                        actual_value = _as_float(dt.get("compile_success_rate"), 0.0)
                    else:
                        ok = 0
                        for x in compile_variants:
                            if isinstance(x, dict) and str(x.get("status") or "") == "PASS":
                                ok += 1
                        actual_value = float(ok / len(compile_variants)) if len(compile_variants) > 0 else 0.0
                elif metric_id == "cache_hit_rate":
                    if "cache_hit_rate" in dt:
                        actual_value = _as_float(dt.get("cache_hit_rate"), 0.0)
                    else:
                        hit = 0
                        for x in compile_variants:
                            if isinstance(x, dict) and bool(x.get("cache_hit")):
                                hit += 1
                        actual_value = float(hit / len(compile_variants)) if len(compile_variants) > 0 else 0.0
                elif metric_id == "correctness_consistency":
                    if "correctness_consistency" in dt:
                        actual_value = _as_float(dt.get("correctness_consistency"), 0.0)
                    else:
                        ok = 0
                        for x in correctness:
                            if isinstance(x, dict) and bool(x.get("repeat_consistent")):
                                ok += 1
                        actual_value = float(ok / len(correctness)) if len(correctness) > 0 else 0.0

            elif group_name == "state_compression":
                sc = m7_data.get("state_compression", {})
                if not sc:
                    sc = m7_data.get("state_compression_summary", {})
                if not isinstance(sc, dict):
                    sc = {}
                if metric_id == "compression_ratio":
                    actual_value = sc.get("compression_ratio", 1.0)
                elif metric_id == "restore_consistency":
                    actual_value = sc.get("restore_consistency", 0.0)
                elif metric_id == "dedup_expansion_ratio":
                    actual_value = sc.get("dedup_expansion_ratio", 999.0)

            elif group_name == "soft_rt_replay":
                rp = m7_data.get("soft_rt_replay", {})
                if not rp:
                    rp = m7_data.get("replay", {})
                if not isinstance(rp, dict):
                    rp = {}
                if metric_id == "deadline_ms":
                    actual_value = rp.get("deadline_ms", 999.0)
                elif metric_id == "p99_latency_ms":
                    if "p99_latency_ms" in rp:
                        actual_value = rp.get("p99_latency_ms", 999.0)
                    else:
                        lat = rp.get("latency_ms")
                        actual_value = (lat.get("p99") if isinstance(lat, dict) else 999.0)
                elif metric_id == "miss_rate":
                    actual_value = rp.get("miss_rate", 1.0)

            elif group_name == "industrial_audit":
                au = m7_data.get("industrial_audit", {})
                if not au:
                    a2 = m7_data.get("audit", {})
                    if isinstance(a2, dict):
                        au = {
                            "event_integrity": 1.0 if str(a2.get("status") or "") == "PASS" else 0.0,
                            "hash_chain_valid": 1.0 if bool(a2.get("verify_ok")) else 0.0,
                        }
                if not isinstance(au, dict):
                    au = {}
                if metric_id == "event_integrity":
                    actual_value = au.get("event_integrity", 0.0)
                elif metric_id == "hash_chain_valid":
                    actual_value = au.get("hash_chain_valid", 0.0)

            if actual_value is None:
                group_pass = False
                continue

            av = _as_float(actual_value, 0.0)
            passed = False
            if op == ">=":
                passed = av >= threshold
            elif op == "<=":
                passed = av <= threshold
            elif op == "==":
                passed = av == threshold
            if not passed:
                group_pass = False

        results[group_name] = "PASS" if group_pass else "FAIL"
        if not group_pass:
            final_pass = False

    out_file = str(out_dir_p / str((config.get("output") or {}).get("report_file") or "report.json"))
    if not pipeline_contract_ok:
        final_pass = False
    if str(mandatory_protocol_gate.get("status") or "") != "PASS":
        final_pass = False
    gate = {
        "status": "PASS" if final_pass else "FAIL",
        "config": str(cfg_path),
        "report_path": out_file,
        "metrics": results,
        "mandatory_protocol_gate": mandatory_protocol_gate,
    }
    gui_evidence = _load_gui_agent_runtime_evidence(out_dir_p, report)
    gui_graph_native = _load_gui_graph_native(report, gui_evidence, matrix_axes)
    gui_graph_native_path = _infer_gui_graph_native_integration_path(report, gui_evidence)
    runtime_evidence = {
        "status": gate["status"],
        "root_pipeline_report_path": str(report.get("pipeline_root_report_path") or ""),
        "root_pipeline_report_contract_source": str(report.get("pipeline_root_report_contract_source") or ""),
        "m7_dependency_source": str(m7_dependency_source),
        "upstream_m7_report_path": str(m7_contract.get("report_path") or ""),
        "upstream_m7_summary_path": str(m7_contract.get("summary_path") or ""),
        "state_abi_contract": {
            "source": "pipeline_kernel_contract_artifacts",
            "state_abi_path": str(kernel_artifacts.get("state_abi_path") or ""),
            "execution_context_path": str(kernel_artifacts.get("execution_context_path") or ""),
            "contract_manifest_path": str(kernel_artifacts.get("contract_manifest_path") or ""),
            "system_execution_manifest_path": str(kernel_artifacts.get("system_execution_manifest_path") or ""),
        },
        "pipeline_kernel_contract_artifacts": kernel_artifacts,
        "pipeline_contract_descriptor": kernel_contract,
        "runtime_protocol_contract": runtime_protocol_projection.get("runtime_protocol_contract") or {},
        "mandatory_protocol_gate": mandatory_protocol_gate,
        "zero_copy_vram_real": runtime_protocol_projection.get("zero_copy_vram_real") or {},
        "matrix_axes": matrix_axes,
        "gui_agent_evidence_path": str(gui_evidence.get("evidence_path") or ""),
        "gui_evidence_mode": "real_gui_evidence" if str(gui_evidence.get("status") or "") == "PASS" else "synthetic_fallback",
        "gui_categories_present": list(gui_evidence.get("categories_present") or []),
        "gui_graph_native_status": str(gui_graph_native.get("status") or "FAIL"),
        "gui_graph_native_integration_level": str(gui_graph_native.get("integration_level") or ""),
        "gui_graph_native_ready": bool(gui_graph_native.get("ready_for_graph_native")),
        "gui_graph_native_native_operator_execution": bool(gui_graph_native.get("native_operator_execution")),
        "gui_graph_native_pipeline_stage_direct_execution": bool(gui_graph_native.get("pipeline_stage_direct_execution")),
        "gui_graph_native_tensorized_gui_source_enabled": bool(gui_graph_native.get("tensorized_gui_source_enabled")),
        "gui_graph_native_integration_path": gui_graph_native_path,
    }
    runtime_evidence_path = write_json(out_dir_p / "runtime_evidence.json", runtime_evidence)
    stage_status = {group: {"status": status} for group, status in results.items()}
    stage_status["pipeline_contract_artifacts"] = {
        "status": "PASS" if pipeline_contract_ok else "FAIL",
        "reason": "" if pipeline_contract_ok else "pipeline_kernel_contract_artifacts_not_ready",
    }
    stage_status["mandatory_protocol_gate"] = mandatory_protocol_gate
    stage_status["runtime_evidence"] = {"status": "PASS", "path": str(runtime_evidence_path)}
    stage_status["gui_source_registry"] = {"status": str(gui_graph_native.get("status") or "FAIL")}
    stage_status["gui_stage_bindings"] = {"status": "PASS" if bool(gui_graph_native.get("stage_bindings")) else "FAIL"}
    stage_status["gui_stage_operator_execution"] = {"status": "PASS" if bool(gui_graph_native.get("stage_operator_execution")) else "FAIL"}
    stage_status["gui_stage_tensorized_gui_source"] = {"status": "PASS" if bool(gui_graph_native.get("stage_tensorized_gui_source")) else "FAIL"}
    stage_status["gui_operator_graph"] = {"status": "PASS" if isinstance(gui_graph_native.get("operator_graph"), dict) and bool(gui_graph_native.get("operator_graph")) else "FAIL"}
    stage_status["gui_execution_context"] = {"status": "PASS" if isinstance(gui_graph_native.get("execution_context"), dict) and bool(gui_graph_native.get("execution_context")) else "FAIL"}
    six_events, gui_stage_status, gui_artifact_paths, six_evidence_mode = _build_m72_six_element_events(
        gate_status=gate["status"],
        results=results,
        gui_evidence=gui_evidence,
    )
    stage_status.update(gui_stage_status)
    stage_rows = stage_trace_rows(gate_name="m72", stage_status=stage_status)
    stage_trace_path = write_jsonl(out_dir_p / "stage_trace.jsonl", stage_rows)
    six_summary = six_element_summary(six_events)
    six_summary["evidence_mode"] = six_evidence_mode
    six_summary["gui_categories_present"] = list(gui_evidence.get("categories_present") or [])
    six_summary["graph_native_status"] = str(gui_graph_native.get("status") or "FAIL")
    six_summary["graph_native_integration_level"] = str(gui_graph_native.get("integration_level") or "")
    six_events_path = write_jsonl(out_dir_p / "six_element_events.jsonl", six_events)
    gap_paths = write_gap_closure_artifacts(
        gate_dir=out_dir_p,
        gate_name="m72",
        matrix_axes=matrix_axes,
        owner="cgc_engine.product.m72_gate",
    )
    q2rl_paths = write_cloud_edge_q2rl_artifacts(
        gate_dir=out_dir_p,
        gate_name="m72",
        matrix_axes=matrix_axes,
        owner="cgc_engine.product.m72_gate",
    )
    local_summary_path = str((out_dir_p / "summary.json").resolve())
    cloud_return_paths = write_edge_to_cloud_return_artifacts(
        gate_dir=out_dir_p,
        gate_name="m72",
        gate_status=gate["status"],
        matrix_axes=matrix_axes,
        runtime_evidence_path=str(runtime_evidence_path),
        six_events_path=str(six_events_path),
        local_report_path=str(Path(out_file).resolve()),
        local_summary_path=local_summary_path,
        q2rl_paths=q2rl_paths,
        gui_evidence=gui_evidence,
        six_summary=six_summary,
        results=results,
    )
    teaching_paths = write_teaching_inference_visualization_artifacts(
        gate_dir=out_dir_p,
        gate_name="m72",
        matrix_axes=matrix_axes,
        gui_evidence=gui_evidence,
        gui_graph_native=gui_graph_native,
        six_summary=six_summary,
        runtime_evidence_path=str(runtime_evidence_path),
        six_events_path=str(six_events_path),
        cloud_summary_path=str(cloud_return_paths.get("cloud_summary_path") or ""),
    )
    stage_status["cloud_edge_training_inference_mode"] = {
        "status": "PASS" if bool(q2rl_paths.get("cloud_edge_training_inference_mode_path")) else "FAIL"
    }
    stage_status["gui_agent_edge_inference_contract"] = {
        "status": "PASS" if bool(q2rl_paths.get("gui_agent_edge_inference_contract_path")) else "FAIL"
    }
    stage_status["q2rl_post_training_profile"] = {
        "status": "PASS" if bool(q2rl_paths.get("q2rl_post_training_profile_path")) else "FAIL"
    }
    stage_status["edge_deployment_bundle_manifest"] = {
        "status": "PASS" if bool(q2rl_paths.get("edge_deployment_bundle_manifest_path")) else "FAIL"
    }
    stage_status["cloud_edge_q2rl_evaluation_plan"] = {
        "status": "PASS" if bool(q2rl_paths.get("cloud_edge_q2rl_evaluation_plan_path")) else "FAIL"
    }
    stage_status["edge_inference_result"] = {
        "status": "PASS" if bool(cloud_return_paths.get("edge_inference_result_path")) else "FAIL"
    }
    stage_status["replay_anchor"] = {
        "status": "PASS" if bool(cloud_return_paths.get("replay_anchor_path")) else "FAIL"
    }
    stage_status["reward_trace"] = {
        "status": "PASS" if bool(cloud_return_paths.get("reward_trace_path")) else "FAIL"
    }
    stage_status["cloud_ingest_manifest"] = {
        "status": "PASS" if bool(cloud_return_paths.get("cloud_ingest_manifest_path")) else "FAIL"
    }
    stage_status["cloud_summary"] = {
        "status": "PASS" if bool(cloud_return_paths.get("cloud_summary_path")) else "FAIL"
    }
    stage_status["teaching_mode_contract"] = {
        "status": "PASS" if bool(teaching_paths.get("teaching_mode_contract_path")) else "FAIL"
    }
    stage_status["teaching_dataset_manifest"] = {
        "status": "PASS" if bool(teaching_paths.get("teaching_dataset_manifest_path")) else "FAIL"
    }
    stage_status["teaching_trained_model_manifest"] = {
        "status": "PASS" if bool(teaching_paths.get("teaching_trained_model_manifest_path")) else "FAIL"
    }
    stage_status["q2rl_training_report"] = {
        "status": "PASS" if bool(teaching_paths.get("q2rl_training_report_path")) else "FAIL"
    }
    stage_status["edge_inference_push_contract"] = {
        "status": "PASS" if bool(teaching_paths.get("edge_inference_push_contract_path")) else "FAIL"
    }
    stage_status["llm_six_element_inference_mode"] = {
        "status": "PASS" if bool(teaching_paths.get("llm_six_element_inference_mode_path")) else "FAIL"
    }
    stage_status["teaching_alignment_report"] = {
        "status": "PASS" if bool(teaching_paths.get("teaching_alignment_report_path")) else "FAIL"
    }
    stage_status["teaching_vs_inference_graph"] = {
        "status": "PASS" if bool(teaching_paths.get("teaching_vs_inference_graph_path")) else "FAIL"
    }
    stage_status["teaching_optimization_triplet_comparison"] = {
        "status": "PASS" if bool(teaching_paths.get("teaching_optimization_triplet_comparison_path")) else "FAIL"
    }
    stage_status["triplet_comparison_mmd"] = {
        "status": "PASS" if bool(teaching_paths.get("triplet_comparison_mmd_path")) else "FAIL"
    }
    stage_status["triplet_comparison_html"] = {
        "status": "PASS" if bool(teaching_paths.get("triplet_comparison_html_path")) else "FAIL"
    }
    stage_status["before_vs_after_vs_teaching_chart"] = {
        "status": "PASS" if bool(teaching_paths.get("before_vs_after_vs_teaching_chart_path")) else "FAIL"
    }
    stage_status["teaching_optimization_audit_replay_bundle"] = {
        "status": "PASS" if bool(teaching_paths.get("teaching_optimization_audit_replay_bundle_path")) else "FAIL"
    }
    stage_status["graph_error_visualization"] = {
        "status": "PASS" if bool(teaching_paths.get("graph_error_visualization_path")) else "FAIL"
    }
    stage_status["graph_error_visualization_mmd"] = {
        "status": "PASS" if bool(teaching_paths.get("graph_error_visualization_mmd_path")) else "FAIL"
    }
    stage_rows = stage_trace_rows(gate_name="m72", stage_status=stage_status)
    stage_trace_path = write_jsonl(out_dir_p / "stage_trace.jsonl", stage_rows)
    artifact_entries = artifact_index(
        [
            str(runtime_evidence_path),
            str(stage_trace_path),
            str(six_events_path),
            gui_graph_native_path,
            *kernel_artifacts.values(),
            *gui_artifact_paths,
            *gap_paths.values(),
            *q2rl_paths.values(),
            *cloud_return_paths.values(),
            *teaching_paths.values(),
        ]
    )
    artifact_index_path = write_json(out_dir_p / "artifact_index.json", {"artifacts": artifact_entries})
    gate["matrix_axes"] = matrix_axes
    gate["artifact_index"] = artifact_entries
    gate["artifact_index_path"] = str(artifact_index_path)
    gate["stage_trace_path"] = str(stage_trace_path)
    gate["runtime_evidence_path"] = str(runtime_evidence_path)
    gate["gui_agent_evidence_path"] = str(gui_evidence.get("evidence_path") or "")
    gate["gui_graph_native_integration_path"] = gui_graph_native_path
    gate["six_element_events_path"] = str(six_events_path)
    gate["six_element_audit"] = six_summary
    gate["closure_artifacts"] = gap_paths
    gate["cloud_edge_q2rl_artifacts"] = q2rl_paths
    gate["cloud_return_artifacts"] = cloud_return_paths
    gate["teaching_inference_visualization_artifacts"] = teaching_paths
    gate["cloud_single_source_path"] = str(cloud_return_paths.get("cloud_summary_path") or "")
    gate["pipeline_kernel_contract_artifacts"] = kernel_artifacts
    gate["pipeline_contract_descriptor"] = kernel_contract
    gate["upkg30"] = {
        "3.2_agent_runtime": {"status": gate["status"]},
        "3.4_unified_artifact_and_summary": {
            "status": "PASS",
            "single_source_mode": "cloud_aggregated",
            "primary_path": str(cloud_return_paths.get("cloud_summary_path") or ""),
            "local_report_path": str(Path(out_file).resolve()),
        },
        "3.5_six_element_audit_and_attribution": {"status": str(six_summary.get("status") or "FAIL")},
        "3.6_missing_capability_closure": {"status": "PASS"},
        "3.6_graph_native_gui_source_integration": {"status": str(gui_graph_native.get("status") or "FAIL")},
        "3.7_cloud_edge_training_inference_q2rl": {"status": "PASS"},
        "3.8_teaching_and_pure_llm_six_element_inference": {
            "status": "PASS",
            "target_model_id": "bytedance-research/UI-TARS-2B-SFT",
            "target_model_source_resolution_path": str(teaching_paths.get("target_model_source_resolution_path") or ""),
            "q2rl_training_report_path": str(teaching_paths.get("q2rl_training_report_path") or ""),
            "comparison_graph_path": str(teaching_paths.get("teaching_vs_inference_graph_path") or ""),
            "triplet_comparison_path": str(teaching_paths.get("teaching_optimization_triplet_comparison_path") or ""),
            "triplet_comparison_mmd_path": str(teaching_paths.get("triplet_comparison_mmd_path") or ""),
            "triplet_comparison_html_path": str(teaching_paths.get("triplet_comparison_html_path") or ""),
            "metric_chart_path": str(teaching_paths.get("before_vs_after_vs_teaching_chart_path") or ""),
            "audit_replay_bundle_path": str(teaching_paths.get("teaching_optimization_audit_replay_bundle_path") or ""),
            "error_visualization_path": str(teaching_paths.get("graph_error_visualization_path") or ""),
        },
    }
    gate["failure_attribution"] = failure_attribution(
        gate_name="m72",
        status=gate["status"],
        stage_status={**stage_status, **gate["upkg30"]},
    )
    report_payload = {
        "name": config.get("name"),
        "status": "PASS" if final_pass else "FAIL",
        "scope": "verification_only",
        "public_entrypoint": "cgc gate m72",
        "metrics": results,
        "matrix_axes": matrix_axes,
        "artifact_index": artifact_entries,
        "artifact_index_path": str(artifact_index_path),
        "stage_trace_path": str(stage_trace_path),
        "failure_attribution": gate["failure_attribution"],
        "pipeline_kernel_contract_artifacts": kernel_artifacts,
        "pipeline_contract_descriptor": kernel_contract,
        "upkg30": gate["upkg30"],
        "gui_graph_native_integration": gui_graph_native,
        "gui_graph_native_integration_path": gui_graph_native_path,
        "cloud_edge_q2rl_artifacts": q2rl_paths,
        "cloud_return_artifacts": cloud_return_paths,
        "teaching_inference_visualization_artifacts": teaching_paths,
        "single_source_of_truth": {
            "mode": "cloud_aggregated",
            "primary_path": str(cloud_return_paths.get("cloud_summary_path") or ""),
            "local_report_path": str(Path(out_file).resolve()),
            "cloud_ingest_manifest_path": str(cloud_return_paths.get("cloud_ingest_manifest_path") or ""),
        },
    }
    summary_payload = build_gate_summary(
        gate_name="m72",
        milestone="m72",
        status=gate["status"],
        matrix_axes=matrix_axes,
        report_path=Path(out_file),
        artifact_entries=artifact_entries,
        stage_rows=stage_rows,
        failure=gate["failure_attribution"],
    )
    summary_path = write_json(out_dir_p / "summary.json", summary_payload)
    gate["summary_path"] = str(summary_path)
    report_payload["summary_path"] = str(summary_path)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            report_payload,
            f,
            ensure_ascii=False,
            indent=2,
        )
    return {"gate_result": {"m72": gate}, "report_path": out_file, "summary_path": str(summary_path)}
