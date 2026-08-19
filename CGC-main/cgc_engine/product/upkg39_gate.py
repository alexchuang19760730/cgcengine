from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from cgc_engine.product.upkg30_common import (
    artifact_index,
    build_gate_summary,
    derive_matrix_axes,
    failure_attribution,
    incoming_upstream_contract,
    load_pipeline_report,
    prefer_upstream_pipeline_contract_descriptor,
    prefer_upstream_pipeline_kernel_contract_artifacts,
    pipeline_contract_descriptor,
    pipeline_kernel_contract_artifacts,
    read_json,
    stage_trace_rows,
    upstream_gate_payload,
    write_json,
    write_jsonl,
)

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover - runtime fallback
    Draft202012Validator = None


def _teaching_alignment_threshold() -> float:
    raw = "0.8"
    try:
        value = float(raw)
    except Exception:
        value = 0.8
    return max(0.0, min(1.0, value))


def _read_upstream_gate(
    *,
    output_root: Path,
    cgc_report: Dict[str, Any] | None,
    gate_name: str,
    report_relative_path: str,
) -> Dict[str, Any]:
    incoming = incoming_upstream_contract(cgc_report if isinstance(cgc_report, dict) else {}, gate_name)
    payload = upstream_gate_payload(incoming)
    if payload:
        return payload
    report_path = (output_root / report_relative_path).resolve()
    report = read_json(report_path)
    gate_result = report.get("gate_result") if isinstance(report.get("gate_result"), dict) else {}
    resolved = gate_result.get(gate_name) if isinstance(gate_result.get(gate_name), dict) else {}
    return dict(resolved) if isinstance(resolved, dict) else {}


def _artifact_envelope_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "cgc.upkg3.artifact-envelope.schema.json",
        "title": "UPKG3 Artifact Envelope",
        "type": "object",
        "required": [
            "schema_version",
            "artifact_type",
            "artifact_id",
            "producer",
            "created_at",
            "input_refs",
            "content",
        ],
        "properties": {
            "schema_version": {"type": "string", "pattern": r"^3\.0\.[0-9]+$"},
            "artifact_type": {
                "type": "string",
                "enum": [
                    "workflow_artifact",
                    "dag_artifact",
                    "gui_binding_artifact",
                    "teaching_trace_artifact",
                    "training_unit_artifact",
                    "eval_report_artifact",
                    "inference_artifact",
                    "compare_report_artifact",
                    "audit_report_artifact",
                    "replay_report_artifact",
                    "trace_report_artifact",
                ],
            },
            "artifact_id": {"type": "string", "minLength": 1},
            "producer": {
                "type": "object",
                "required": ["module", "version"],
                "properties": {
                    "module": {"type": "string", "minLength": 1},
                    "version": {"type": "string", "minLength": 1},
                },
                "additionalProperties": True,
            },
            "created_at": {"type": "string", "format": "date-time"},
            "input_refs": {"type": "array", "items": {"type": "string"}},
            "content": {"type": "object"},
            "validation_rules": {"type": "object"},
        },
        "additionalProperties": True,
    }


def _dag_node_schema() -> Dict[str, Any]:
    return {
        "$id": "cgc.upkg3.dag-node.schema.json",
        "title": "UPKG3 DAG Node",
        "type": "object",
        "required": [
            "node_id",
            "node_type",
            "intent_type",
            "preconditions",
            "postconditions",
            "failure_policy",
            "binding_ref",
        ],
        "properties": {
            "node_id": {"type": "string", "minLength": 1},
            "node_type": {"type": "string", "minLength": 1},
            "intent_type": {"type": "string", "minLength": 1},
            "preconditions": {"type": "array", "items": {"type": "string"}},
            "postconditions": {"type": "array", "items": {"type": "string"}},
            "failure_policy": {
                "type": "object",
                "required": ["retry", "fallback"],
                "properties": {
                    "retry": {"type": "integer", "minimum": 0},
                    "fallback": {"type": "string", "minLength": 1},
                    "on_error_node": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "binding_ref": {"type": "string", "minLength": 1},
        },
        "additionalProperties": True,
    }


def _gui_binding_schema() -> Dict[str, Any]:
    return {
        "$id": "cgc.upkg3.gui-binding.schema.json",
        "title": "UPKG3 GUI Binding",
        "type": "object",
        "required": [
            "binding_id",
            "action_type",
            "target_locator",
            "locator_priority",
            "timeout_ms",
            "retry_policy",
            "fallback_policy",
        ],
        "properties": {
            "binding_id": {"type": "string", "minLength": 1},
            "action_type": {"type": "string", "minLength": 1},
            "target_locator": {
                "type": "object",
                "properties": {
                    "selector": {"type": ["string", "null"]},
                    "ocr_text": {"type": ["string", "null"]},
                    "image_anchor": {"type": ["string", "null"]},
                },
                "additionalProperties": True,
            },
            "locator_priority": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["selector", "ocr_text", "image_anchor", "shortcut"],
                },
                "minItems": 1,
            },
            "timeout_ms": {"type": "integer", "minimum": 1},
            "retry_policy": {
                "type": "object",
                "required": ["max_retry", "retry_interval_ms"],
                "properties": {
                    "max_retry": {"type": "integer", "minimum": 0},
                    "retry_interval_ms": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": True,
            },
            "fallback_policy": {
                "type": "object",
                "properties": {
                    "fallback_to_ocr": {"type": "boolean"},
                    "fallback_to_human_takeover": {"type": "boolean"},
                },
                "additionalProperties": True,
            },
            "environment_constraints": {"type": "object"},
        },
        "anyOf": [
            {"properties": {"target_locator": {"properties": {"selector": {"type": "string", "minLength": 1}}, "required": ["selector"]}}},
            {"properties": {"target_locator": {"properties": {"ocr_text": {"type": "string", "minLength": 1}}, "required": ["ocr_text"]}}},
            {"properties": {"target_locator": {"properties": {"image_anchor": {"type": "string", "minLength": 1}}, "required": ["image_anchor"]}}},
        ],
        "additionalProperties": True,
    }


def _six_element_inference_schema() -> Dict[str, Any]:
    return {
        "$id": "cgc.upkg3.six-element-inference.schema.json",
        "title": "UPKG3 Six Element Inference",
        "type": "object",
        "required": ["input", "output"],
        "properties": {
            "input": {
                "type": "object",
                "required": [
                    "goal",
                    "state",
                    "constraints",
                    "candidates",
                    "policy_context",
                    "success_criteria",
                ],
                "properties": {
                    "goal": {"type": "object"},
                    "state": {"type": "object"},
                    "constraints": {"type": "object"},
                    "candidates": {"type": "array"},
                    "policy_context": {"type": "object"},
                    "success_criteria": {"type": "object"},
                },
                "additionalProperties": False,
            },
            "output": {
                "type": "object",
                "required": [
                    "selected_action",
                    "reasoning_summary",
                    "expected_post_state",
                    "risk_flags",
                    "fallback_plan",
                    "confidence",
                ],
                "properties": {
                    "selected_action": {
                        "type": "object",
                        "required": ["dag_node_id", "execution_intent_id", "gui_binding_id"],
                        "properties": {
                            "dag_node_id": {"type": "string", "minLength": 1},
                            "execution_intent_id": {"type": "string", "minLength": 1},
                            "gui_binding_id": {"type": "string", "minLength": 1},
                        },
                        "additionalProperties": True,
                    },
                    "reasoning_summary": {"type": "string"},
                    "expected_post_state": {"type": "object"},
                    "risk_flags": {"type": "array", "items": {"type": "string"}},
                    "fallback_plan": {"type": "object"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": True,
    }


def _evidence_bundle_schema() -> Dict[str, Any]:
    return {
        "$id": "cgc.upkg3.evidence-bundle.schema.json",
        "title": "UPKG3 Evidence Bundle",
        "type": "object",
        "required": ["env_snapshot", "perception_snapshot", "action_record", "result_record"],
        "properties": {
            "env_snapshot": {"type": "object"},
            "perception_snapshot": {"type": "object"},
            "action_record": {"type": "object"},
            "result_record": {"type": "object"},
        },
        "additionalProperties": True,
    }


def _schema_bundle() -> Dict[str, Dict[str, Any]]:
    return {
        "artifact_envelope.schema.json": _artifact_envelope_schema(),
        "dag_node.schema.json": _dag_node_schema(),
        "gui_binding.schema.json": _gui_binding_schema(),
        "six_element_inference.schema.json": _six_element_inference_schema(),
        "evidence_bundle.schema.json": _evidence_bundle_schema(),
    }


def _field_dictionary_manifest() -> Dict[str, Any]:
    return {
        "schema_version": "3.0.0",
        "status": "PASS",
        "sections": {
            "dag_node": [
                {"field": "node_id", "type": "string", "required": True},
                {"field": "node_type", "type": "string", "required": True},
                {"field": "intent_type", "type": "string", "required": True},
                {"field": "preconditions", "type": "string[]", "required": True},
                {"field": "postconditions", "type": "string[]", "required": True},
                {"field": "failure_policy", "type": "object", "required": True},
                {"field": "binding_ref", "type": "string", "required": True},
            ],
            "gui_binding": [
                {"field": "binding_id", "type": "string", "required": True},
                {"field": "action_type", "type": "string", "required": True},
                {"field": "target_locator", "type": "object", "required": True},
                {"field": "locator_priority", "type": "string[]", "required": True},
                {"field": "timeout_ms", "type": "integer", "required": True},
                {"field": "retry_policy", "type": "object", "required": True},
                {"field": "fallback_policy", "type": "object", "required": True},
            ],
            "six_element_inference": [
                {"field": "input.goal", "type": "object", "required": True},
                {"field": "input.state", "type": "object", "required": True},
                {"field": "input.constraints", "type": "object", "required": True},
                {"field": "input.candidates", "type": "array", "required": True},
                {"field": "input.policy_context", "type": "object", "required": True},
                {"field": "input.success_criteria", "type": "object", "required": True},
                {"field": "output.selected_action", "type": "object", "required": True},
                {"field": "output.confidence", "type": "number", "required": True},
            ],
            "evidence_bundle": [
                {"field": "env_snapshot", "type": "object", "required": True},
                {"field": "perception_snapshot", "type": "object", "required": True},
                {"field": "action_record", "type": "object", "required": True},
                {"field": "result_record", "type": "object", "required": True},
            ],
        },
    }


def _valid_samples() -> Dict[str, Dict[str, Any]]:
    return {
        "artifact_envelope": {
            "schema_version": "3.0.0",
            "artifact_type": "inference_artifact",
            "artifact_id": "infer_001",
            "producer": {"module": "cgc.upkg3.infer", "version": "3.0.0"},
            "created_at": "2026-06-22T10:00:00Z",
            "input_refs": ["workflow_approve_order_v1", "binding_approve_order_v1"],
            "content": {"session_id": "sess_001"},
            "validation_rules": {"required_fields": ["artifact_type"], "semantic_checks": []},
        },
        "dag_node": {
            "node_id": "approve_order",
            "node_type": "business_task",
            "intent_type": "click",
            "preconditions": ["page=order_detail"],
            "postconditions": ["status=approved"],
            "failure_policy": {"retry": 1, "fallback": "human_takeover"},
            "binding_ref": "binding_approve_order_v1",
        },
        "gui_binding": {
            "binding_id": "binding_approve_order_v1",
            "action_type": "click",
            "target_locator": {"selector": "#approve-btn", "ocr_text": "审批通过", "image_anchor": None},
            "locator_priority": ["selector", "ocr_text"],
            "timeout_ms": 5000,
            "retry_policy": {"max_retry": 1, "retry_interval_ms": 500},
            "fallback_policy": {"fallback_to_ocr": True, "fallback_to_human_takeover": True},
        },
        "six_element_inference": {
            "input": {
                "goal": {"task_id": "task_approve_order"},
                "state": {"page": "order_detail", "status": "pending"},
                "constraints": {"permission": "approver", "forbidden_actions": []},
                "candidates": [{"execution_intent_id": "intent_click_approve", "gui_binding_id": "binding_approve_order_v1"}],
                "policy_context": {"policy_version": "policy_20260622"},
                "success_criteria": {"postcondition": "status=approved"},
            },
            "output": {
                "selected_action": {
                    "dag_node_id": "approve_order",
                    "execution_intent_id": "intent_click_approve",
                    "gui_binding_id": "binding_approve_order_v1",
                },
                "reasoning_summary": "Current page satisfies approval preconditions.",
                "expected_post_state": {"status": "approved"},
                "risk_flags": [],
                "fallback_plan": {"on_fail": "human_takeover"},
                "confidence": 0.92,
            },
        },
        "evidence_bundle": {
            "env_snapshot": {"os": "windows", "resolution": "1920x1080"},
            "perception_snapshot": {"selector": "#approve-btn", "ocr_text": "审批通过"},
            "action_record": {"type": "click", "target": "approve_button"},
            "result_record": {
                "postcondition_passed": True,
                "before_screenshot_ref": "shot_before_001",
                "after_screenshot_ref": "shot_after_001",
                "error": None,
            },
        },
    }


def _invalid_samples() -> Dict[str, Dict[str, Any]]:
    return {
        "artifact_envelope": {
            "schema_version": "3.0.0",
            "artifact_id": "infer_002",
            "producer": {"module": "cgc.upkg3.infer", "version": "3.0.0"},
            "created_at": "2026-06-22T10:00:00Z",
            "input_refs": [],
            "content": {},
        },
        "dag_node": {
            "node_id": "approve_order",
            "node_type": "business_task",
            "intent_type": "click",
            "preconditions": ["page=order_detail"],
            "postconditions": ["status=approved"],
            "failure_policy": {"retry": 1, "fallback": "human_takeover"},
        },
        "gui_binding": {
            "binding_id": "binding_approve_order_v1",
            "action_type": "click",
            "target_locator": {"selector": None, "ocr_text": None, "image_anchor": None},
            "locator_priority": ["selector"],
            "timeout_ms": 5000,
            "retry_policy": {"max_retry": 1, "retry_interval_ms": 500},
            "fallback_policy": {"fallback_to_ocr": False, "fallback_to_human_takeover": False},
        },
        "six_element_inference": {
            "input": {
                "goal": {"task_id": "task_approve_order"},
                "state": {"page": "order_detail"},
                "constraints": {},
                "candidates": [],
                "policy_context": {},
                "success_criteria": {"postcondition": "status=approved"},
            },
            "output": {
                "selected_action": {"dag_node_id": "approve_order"},
                "reasoning_summary": "invalid",
                "expected_post_state": {},
                "risk_flags": [],
                "fallback_plan": {},
                "confidence": 1.5,
            },
        },
        "evidence_bundle": {
            "env_snapshot": {"os": "windows"},
            "perception_snapshot": {"selector": "#approve-btn"},
            "action_record": {"type": "click"},
        },
    }


def _custom_validate(name: str, payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if name == "artifact_envelope":
        if not re.match(r"^3\.0\.[0-9]+$", str(payload.get("schema_version") or "")):
            errors.append("schema_version_invalid")
        artifact_type = str(payload.get("artifact_type") or "")
        if artifact_type == "":
            errors.append("artifact_type_missing")
        producer = payload.get("producer") if isinstance(payload.get("producer"), dict) else {}
        if str(producer.get("module") or "") == "" or str(producer.get("version") or "") == "":
            errors.append("producer_incomplete")
    elif name == "dag_node":
        if str(payload.get("binding_ref") or "") == "":
            errors.append("binding_ref_missing")
    elif name == "gui_binding":
        target_locator = payload.get("target_locator") if isinstance(payload.get("target_locator"), dict) else {}
        if not any(str(target_locator.get(key) or "").strip() for key in ("selector", "ocr_text", "image_anchor")):
            errors.append("locator_missing")
    elif name == "six_element_inference":
        output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
        selected_action = output.get("selected_action") if isinstance(output.get("selected_action"), dict) else {}
        for required in ("dag_node_id", "execution_intent_id", "gui_binding_id"):
            if str(selected_action.get(required) or "") == "":
                errors.append(f"selected_action.{required}_missing")
        try:
            confidence = float(output.get("confidence"))
            if confidence < 0.0 or confidence > 1.0:
                errors.append("confidence_out_of_range")
        except Exception:
            errors.append("confidence_invalid")
    elif name == "evidence_bundle":
        if not isinstance(payload.get("result_record"), dict):
            errors.append("result_record_missing")
    return errors


def _validate_sample(name: str, schema: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[bool, List[str], str]:
    errors = _custom_validate(name, payload)
    backend = "internal_rules"
    if Draft202012Validator is not None:
        backend = "jsonschema+internal_rules"
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as exc:
            return False, [f"schema_invalid:{exc!r}"], backend
        validator = Draft202012Validator(schema)
        try:
            validator.validate(payload)
        except Exception as exc:
            errors.append(str(exc))
    return (len(errors) == 0, errors, backend)


def _write_schema_materialization(gate_dir: Path) -> Dict[str, Any]:
    schema_dir = (gate_dir / "schema_bundle").resolve()
    sample_dir = (gate_dir / "samples").resolve()
    schema_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    schema_files: Dict[str, str] = {}
    for file_name, payload in _schema_bundle().items():
        schema_files[file_name] = str(write_json(schema_dir / file_name, payload))

    field_dictionary_manifest_path = str(write_json(gate_dir / "field_dictionary_manifest.json", _field_dictionary_manifest()))

    valid_samples = _valid_samples()
    invalid_samples = _invalid_samples()
    valid_sample_paths: Dict[str, str] = {}
    invalid_sample_paths: Dict[str, str] = {}
    for name, payload in valid_samples.items():
        valid_sample_paths[name] = str(write_json(sample_dir / f"{name}.valid.json", payload))
    for name, payload in invalid_samples.items():
        invalid_sample_paths[name] = str(write_json(sample_dir / f"{name}.invalid.json", payload))

    validation_rows: List[Dict[str, Any]] = []
    backend_used = "internal_rules"
    all_valid_pass = True
    all_invalid_rejected = True
    for name, schema_file in schema_files.items():
        contract_name = name.replace(".schema.json", "")
        sample_key = contract_name
        schema_payload = read_json(Path(schema_file))
        sample_valid = valid_samples.get(sample_key, {})
        sample_invalid = invalid_samples.get(sample_key, {})
        valid_ok, valid_errors, backend = _validate_sample(sample_key, schema_payload, sample_valid)
        invalid_ok, invalid_errors, _ = _validate_sample(sample_key, schema_payload, sample_invalid)
        backend_used = backend
        all_valid_pass = all_valid_pass and valid_ok
        all_invalid_rejected = all_invalid_rejected and (not invalid_ok)
        validation_rows.append(
            {
                "contract": sample_key,
                "schema_path": schema_file,
                "valid_sample_path": valid_sample_paths.get(sample_key, ""),
                "invalid_sample_path": invalid_sample_paths.get(sample_key, ""),
                "valid_sample_status": "PASS" if valid_ok else "FAIL",
                "valid_sample_errors": valid_errors,
                "invalid_sample_status": "PASS" if not invalid_ok else "FAIL",
                "invalid_sample_errors": invalid_errors,
            }
        )

    schema_bundle_manifest = {
        "status": "PASS",
        "schema_version": "3.0.0",
        "schema_backend": backend_used,
        "schema_files": schema_files,
        "field_dictionary_manifest_path": field_dictionary_manifest_path,
        "valid_sample_paths": valid_sample_paths,
        "invalid_sample_paths": invalid_sample_paths,
    }
    schema_bundle_manifest_path = str(write_json(gate_dir / "schema_bundle_manifest.json", schema_bundle_manifest))
    validator_execution_report = {
        "status": "PASS" if all_valid_pass and all_invalid_rejected else "FAIL",
        "schema_backend": backend_used,
        "all_valid_samples_pass": all_valid_pass,
        "all_invalid_samples_rejected": all_invalid_rejected,
        "rows": validation_rows,
    }
    validator_execution_report_path = str(write_json(gate_dir / "validator_execution_report.json", validator_execution_report))
    return {
        "schema_bundle_manifest_path": schema_bundle_manifest_path,
        "field_dictionary_manifest_path": field_dictionary_manifest_path,
        "validator_execution_report_path": validator_execution_report_path,
        "schema_files": schema_files,
        "valid_sample_paths": valid_sample_paths,
        "invalid_sample_paths": invalid_sample_paths,
        "validator_execution_report": validator_execution_report,
    }


def run_upkg39_gate(*, output_dir: str, cgc_report: Dict[str, Any] | None = None) -> Dict[str, Any]:
    output_root = Path(str(output_dir)).expanduser().resolve()
    gate_dir = (output_root / "upkg39_strict_closure").resolve()
    gate_dir.mkdir(parents=True, exist_ok=True)

    pipeline_report = load_pipeline_report(output_dir=output_root)
    m72_gate = _read_upstream_gate(
        output_root=output_root,
        cgc_report=cgc_report,
        gate_name="m72",
        report_relative_path="m72_industrial/report.json",
    )
    m78_gate = _read_upstream_gate(
        output_root=output_root,
        cgc_report=cgc_report,
        gate_name="m78",
        report_relative_path="m78_teaching_pure_llm/m78_report.json",
    )
    kernel_artifacts = prefer_upstream_pipeline_kernel_contract_artifacts(
        pipeline_kernel_contract_artifacts(output_dir=output_root, pipeline_report=pipeline_report),
        m78_gate,
        m72_gate,
    )
    kernel_contract = prefer_upstream_pipeline_contract_descriptor(
        pipeline_contract_descriptor(output_dir=output_root, pipeline_report=pipeline_report),
        m78_gate,
        m72_gate,
    )

    matrix_axes = derive_matrix_axes(
        milestone="upkg39",
        gate_name="3.9 Strict Closure And Schema-Validated Agent Product Gate",
        pipeline_report=pipeline_report,
        extra={
            "strict_alignment_threshold": _teaching_alignment_threshold(),
            "schema_validated": True,
            "closure_mode": "strict_closure",
            "contract_manifest_path": kernel_artifacts.get("contract_manifest_path") or "",
            "system_execution_manifest_path": kernel_artifacts.get("system_execution_manifest_path") or "",
        },
    )

    teaching_paths = m78_gate.get("teaching_inference_visualization_artifacts") if isinstance(m78_gate.get("teaching_inference_visualization_artifacts"), dict) else {}
    closure_paths = m72_gate.get("closure_artifacts") if isinstance(m72_gate.get("closure_artifacts"), dict) else {}
    gui_graph_native = {}
    gui_graph_native_path = str(m72_gate.get("gui_graph_native_integration_path") or "")
    if gui_graph_native_path:
        gui_graph_native = read_json(Path(gui_graph_native_path))
    if not gui_graph_native:
        m72_report = read_json((output_root / "m72_industrial" / "report.json").resolve())
        gui_graph_native = (
            m72_report.get("gui_graph_native_integration")
            if isinstance(m72_report.get("gui_graph_native_integration"), dict)
            else {}
        )

    materialized = _write_schema_materialization(gate_dir)
    validator_report = dict(materialized.get("validator_execution_report") or {})

    alignment_report_path = str(teaching_paths.get("teaching_alignment_report_path") or "")
    alignment_report = read_json(Path(alignment_report_path)) if alignment_report_path else {}
    q2rl_training_report_path = str(teaching_paths.get("q2rl_training_report_path") or "")
    q2rl_training_report = read_json(Path(q2rl_training_report_path)) if q2rl_training_report_path else {}

    strict_alignment_acceptance = {
        "status": "FAIL",
        "threshold": _teaching_alignment_threshold(),
        "alignment_score": float(alignment_report.get("alignment_score") or 0.0),
        "report_target_threshold": float(alignment_report.get("target_threshold") or 0.0),
        "missing_six_elements": list(alignment_report.get("missing_six_elements") or []),
        "reason": "teaching_alignment_not_strictly_accepted",
    }
    if alignment_report:
        strict_ok = (
            float(alignment_report.get("alignment_score") or 0.0) >= _teaching_alignment_threshold()
            and float(alignment_report.get("target_threshold") or 0.0) >= _teaching_alignment_threshold()
            and not list(alignment_report.get("missing_six_elements") or [])
        )
        strict_alignment_acceptance["status"] = "PASS" if strict_ok else "FAIL"
        strict_alignment_acceptance["reason"] = "" if strict_ok else "alignment_score_below_strict_threshold"
    strict_alignment_acceptance_path = str(write_json(gate_dir / "strict_alignment_acceptance.json", strict_alignment_acceptance))

    q2rl_strict_acceptance = {
        "status": "FAIL",
        "reason": "q2rl_post_train_not_strictly_accepted",
        "pre_q2rl_metrics": dict(q2rl_training_report.get("pre_q2rl_metrics") or {}),
        "post_q2rl_metrics": dict(q2rl_training_report.get("post_q2rl_metrics") or {}),
    }
    if q2rl_training_report:
        pre_metrics = q2rl_training_report.get("pre_q2rl_metrics") if isinstance(q2rl_training_report.get("pre_q2rl_metrics"), dict) else {}
        post_metrics = q2rl_training_report.get("post_q2rl_metrics") if isinstance(q2rl_training_report.get("post_q2rl_metrics"), dict) else {}
        reward_gain = float((q2rl_training_report.get("improvement") or {}).get("reward_gain") or (float(post_metrics.get("reward_score") or 0.0) - float(pre_metrics.get("reward_score") or 0.0)))
        alignment_gain = float((q2rl_training_report.get("improvement") or {}).get("alignment_gain") or (float(post_metrics.get("alignment_score") or 0.0) - float(pre_metrics.get("alignment_score") or 0.0)))
        strict_ok = reward_gain > 0.0 and alignment_gain >= 0.0 and float(post_metrics.get("alignment_score") or 0.0) >= _teaching_alignment_threshold()
        q2rl_strict_acceptance.update(
            {
                "status": "PASS" if strict_ok else "FAIL",
                "reason": "" if strict_ok else "post_q2rl_alignment_below_strict_threshold",
                "reward_gain": reward_gain,
                "alignment_gain": alignment_gain,
            }
        )
    q2rl_strict_acceptance_path = str(write_json(gate_dir / "q2rl_strict_acceptance.json", q2rl_strict_acceptance))

    graph_native_tensorized_execution = {
        "status": "FAIL",
        "reason": "graph_native_tensorization_incomplete",
        "integration_level": str(gui_graph_native.get("integration_level") or ""),
        "native_operator_execution": bool(gui_graph_native.get("native_operator_execution")),
        "tensorized_gui_source_enabled": bool(gui_graph_native.get("tensorized_gui_source_enabled")),
        "remaining_gaps": list(gui_graph_native.get("remaining_gaps") or []),
        "stage_operator_execution": dict(gui_graph_native.get("stage_operator_execution") or {}),
        "stage_tensorized_gui_source": dict(gui_graph_native.get("stage_tensorized_gui_source") or {}),
    }
    graph_native_ok = (
        bool(gui_graph_native.get("native_operator_execution"))
        and bool(gui_graph_native.get("tensorized_gui_source_enabled"))
        and str(gui_graph_native.get("integration_level") or "") in {"fully_tensorized_graph_native_execution", "graph_native_stage_execution"}
        and not list(gui_graph_native.get("remaining_gaps") or [])
    )
    graph_native_tensorized_execution["status"] = "PASS" if graph_native_ok else "FAIL"
    graph_native_tensorized_execution["reason"] = "" if graph_native_ok else "remaining_graph_native_gaps"
    graph_native_tensorized_execution_path = str(write_json(gate_dir / "graph_native_tensorized_execution_report.json", graph_native_tensorized_execution))

    required_closure_chain = {
        "workflow_dag_schema_path": str(closure_paths.get("workflow_dag_schema_path") or ""),
        "trajectory_synthesis_spec_path": str(closure_paths.get("trajectory_synthesis_spec_path") or ""),
        "fine_tune_profile_path": str(closure_paths.get("fine_tune_profile_path") or ""),
        "dual_mode_governance_path": str(closure_paths.get("dual_mode_governance_path") or ""),
        "audit_alignment_spec_path": str(closure_paths.get("audit_alignment_spec_path") or ""),
        "teaching_dataset_manifest_path": str(teaching_paths.get("teaching_dataset_manifest_path") or ""),
        "teaching_trained_model_manifest_path": str(teaching_paths.get("teaching_trained_model_manifest_path") or ""),
        "edge_inference_push_contract_path": str(teaching_paths.get("edge_inference_push_contract_path") or ""),
        "llm_six_element_inference_mode_path": str(teaching_paths.get("llm_six_element_inference_mode_path") or ""),
        "teaching_optimization_audit_replay_bundle_path": str(teaching_paths.get("teaching_optimization_audit_replay_bundle_path") or ""),
        "cloud_summary_path": str(teaching_paths.get("cloud_summary_path") or m78_gate.get("cloud_single_source_path") or ""),
    }
    closure_complete = all(Path(path).exists() for path in required_closure_chain.values() if str(path).strip())
    closure_complete = closure_complete and all(str(path).strip() for path in required_closure_chain.values())
    end_to_end_executor_closure = {
        "status": "PASS" if closure_complete else "FAIL",
        "reason": "" if closure_complete else "closure_chain_incomplete",
        "strict_alignment_acceptance_path": strict_alignment_acceptance_path,
        "q2rl_strict_acceptance_path": q2rl_strict_acceptance_path,
        "graph_native_tensorized_execution_report_path": graph_native_tensorized_execution_path,
        "required_chain": required_closure_chain,
    }
    end_to_end_executor_closure_path = str(write_json(gate_dir / "end_to_end_executor_closure.json", end_to_end_executor_closure))

    completion_manifest = {
        "status": "PASS"
        if (
            bool(kernel_contract.get("ready"))
            and str(m78_gate.get("status") or "") == "PASS"
            and str(strict_alignment_acceptance.get("status") or "") == "PASS"
            and str(q2rl_strict_acceptance.get("status") or "") == "PASS"
            and str(graph_native_tensorized_execution.get("status") or "") == "PASS"
            and str(validator_report.get("status") or "") == "PASS"
            and str(end_to_end_executor_closure.get("status") or "") == "PASS"
        )
        else "FAIL",
        "closure_items": {
            "strict_alignment_threshold_restored": str(strict_alignment_acceptance.get("status") or "") == "PASS",
            "schema_bundle_materialized": True,
            "validator_execution_materialized": str(validator_report.get("status") or "") == "PASS",
            "graph_native_tensorization_closed": str(graph_native_tensorized_execution.get("status") or "") == "PASS",
            "end_to_end_executor_closure_materialized": str(end_to_end_executor_closure.get("status") or "") == "PASS",
        },
        "upstream_dependencies": {
            "m72_status": str(m72_gate.get("status") or ""),
            "m78_status": str(m78_gate.get("status") or ""),
            "pipeline_contract_ready": bool(kernel_contract.get("ready")),
        },
    }
    completion_manifest_path = str(write_json(gate_dir / "upkg39_completion_manifest.json", completion_manifest))

    stage_status: Dict[str, Dict[str, Any]] = {
        "pipeline_contract_artifacts": {
            "status": "PASS" if bool(kernel_contract.get("ready")) else "FAIL",
            "reason": "" if bool(kernel_contract.get("ready")) else "pipeline_kernel_contract_artifacts_not_ready",
        },
        "upstream_m78_contract": {
            "status": "PASS" if str(m78_gate.get("status") or "") == "PASS" else "FAIL",
            "reason": "" if str(m78_gate.get("status") or "") == "PASS" else "missing_or_failed_upstream_m78",
        },
        "schema_bundle_manifest": {
            "status": "PASS" if Path(str(materialized.get("schema_bundle_manifest_path") or "")).exists() else "FAIL",
            "path": str(materialized.get("schema_bundle_manifest_path") or ""),
        },
        "field_dictionary_manifest": {
            "status": "PASS" if Path(str(materialized.get("field_dictionary_manifest_path") or "")).exists() else "FAIL",
            "path": str(materialized.get("field_dictionary_manifest_path") or ""),
        },
        "validator_execution_report": {
            "status": str(validator_report.get("status") or "FAIL"),
            "path": str(materialized.get("validator_execution_report_path") or ""),
        },
        "strict_alignment_acceptance": {
            "status": str(strict_alignment_acceptance.get("status") or "FAIL"),
            "path": strict_alignment_acceptance_path,
        },
        "q2rl_strict_acceptance": {
            "status": str(q2rl_strict_acceptance.get("status") or "FAIL"),
            "path": q2rl_strict_acceptance_path,
        },
        "graph_native_tensorized_execution": {
            "status": str(graph_native_tensorized_execution.get("status") or "FAIL"),
            "path": graph_native_tensorized_execution_path,
        },
        "end_to_end_executor_closure": {
            "status": str(end_to_end_executor_closure.get("status") or "FAIL"),
            "path": end_to_end_executor_closure_path,
        },
        "upkg39_completion_manifest": {
            "status": str(completion_manifest.get("status") or "FAIL"),
            "path": completion_manifest_path,
        },
    }

    stage_rows = stage_trace_rows(gate_name="upkg39", stage_status=stage_status)
    stage_trace_path = write_jsonl(gate_dir / "stage_trace.jsonl", stage_rows)
    ok = all(str(item.get("status") or "") == "PASS" for item in stage_status.values())

    gate = {
        "status": "PASS" if ok else "FAIL",
        "matrix_axes": matrix_axes,
        "strict_alignment_threshold": _teaching_alignment_threshold(),
        "pipeline_kernel_contract_artifacts": kernel_artifacts,
        "pipeline_contract_descriptor": kernel_contract,
        "strict_alignment_acceptance": strict_alignment_acceptance,
        "q2rl_strict_acceptance": q2rl_strict_acceptance,
        "graph_native_tensorized_execution": graph_native_tensorized_execution,
        "schema_materialization": {
            "schema_bundle_manifest_path": str(materialized.get("schema_bundle_manifest_path") or ""),
            "field_dictionary_manifest_path": str(materialized.get("field_dictionary_manifest_path") or ""),
            "validator_execution_report_path": str(materialized.get("validator_execution_report_path") or ""),
            "schema_files": dict(materialized.get("schema_files") or {}),
        },
        "closure_chain": end_to_end_executor_closure,
        "completion_manifest_path": completion_manifest_path,
    }
    gate["failure_attribution"] = failure_attribution(
        gate_name="upkg39",
        status=gate["status"],
        stage_status=stage_status,
    )

    artifact_entries = artifact_index(
        [
            str(stage_trace_path),
            str(materialized.get("schema_bundle_manifest_path") or ""),
            str(materialized.get("field_dictionary_manifest_path") or ""),
            str(materialized.get("validator_execution_report_path") or ""),
            strict_alignment_acceptance_path,
            q2rl_strict_acceptance_path,
            graph_native_tensorized_execution_path,
            end_to_end_executor_closure_path,
            completion_manifest_path,
            *list((materialized.get("schema_files") or {}).values()),
            *list((materialized.get("valid_sample_paths") or {}).values()),
            *list((materialized.get("invalid_sample_paths") or {}).values()),
            *list(required_closure_chain.values()),
            *list(kernel_artifacts.values()),
        ]
    )
    artifact_index_path = write_json(gate_dir / "artifact_index.json", {"artifacts": artifact_entries})
    gate["artifact_index"] = artifact_entries
    gate["artifact_index_path"] = str(artifact_index_path)
    gate["stage_trace_path"] = str(stage_trace_path)

    report_payload = {
        "name": "CGC_UPKG_3_9_Strict_Closure_And_Schema_Validated_Agent_Product_Gate",
        "status": gate["status"],
        "scope": "verification_only",
        "public_entrypoint": "cgc gate upkg39",
        "matrix_axes": matrix_axes,
        "artifact_index": artifact_entries,
        "artifact_index_path": str(artifact_index_path),
        "stage_trace_path": str(stage_trace_path),
        "failure_attribution": gate["failure_attribution"],
        "pipeline_kernel_contract_artifacts": kernel_artifacts,
        "pipeline_contract_descriptor": kernel_contract,
        "gate_result": {"upkg39": gate},
    }
    report_path = write_json(gate_dir / "upkg39_report.json", report_payload)
    summary_payload = build_gate_summary(
        gate_name="upkg39",
        milestone="upkg39",
        status=gate["status"],
        matrix_axes=matrix_axes,
        report_path=Path(report_path),
        artifact_entries=artifact_entries,
        stage_rows=stage_rows,
        failure=gate["failure_attribution"],
    )
    summary_path = write_json(gate_dir / "summary.json", summary_payload)
    report_payload["summary_path"] = str(summary_path)
    write_json(report_path, report_payload)
    return {
        "ok": ok,
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "gate_result": {"upkg39": gate},
    }
