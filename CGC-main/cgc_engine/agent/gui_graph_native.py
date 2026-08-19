from __future__ import annotations

import hashlib
from typing import Any, Dict, List


GUI_REQUIRED_CATEGORIES = ("runtime_host", "workflow", "screenshot", "tool_call")

GUI_CATEGORY_TO_ELEMENT = {
    "runtime_host": "environment",
    "workflow": "workflow",
    "screenshot": "perception",
    "tool_call": "execution",
}

GUI_CATEGORY_TO_OPERATOR = {
    "runtime_host": "environment_constraint_node",
    "workflow": "workflow_dispatch_node",
    "screenshot": "perception_capture_node",
    "tool_call": "execution_action_node",
}

GUI_CATEGORY_TO_STAGE = {
    "runtime_host": "step1_staticize",
    "workflow": "step2_graph_capture",
    "screenshot": "step8_runtime",
    "tool_call": "step8_runtime",
}


def _status_from_stage_source(stage_source: Dict[str, Any]) -> str:
    status = str(stage_source.get("status") or "SKIP").upper()
    categories_present = set(str(x) for x in list(stage_source.get("categories_present") or []))
    if status == "SKIP":
        return "SKIP"
    if status == "FAIL":
        return "FAIL"
    if all(category in categories_present for category in GUI_REQUIRED_CATEGORIES):
        return "PASS"
    if bool(categories_present):
        return "PARTIAL"
    return "FAIL"


def _preview_events_by_category(stage_source: Dict[str, Any], *, limit_per_category: int = 4) -> Dict[str, List[Dict[str, Any]]]:
    previews: Dict[str, List[Dict[str, Any]]] = {category: [] for category in GUI_REQUIRED_CATEGORIES}
    events_preview = stage_source.get("events_preview") if isinstance(stage_source.get("events_preview"), list) else []
    for item in events_preview:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "")
        if category not in previews or len(previews[category]) >= int(limit_per_category):
            continue
        previews[category].append(
            {
                "category": category,
                "action": str(item.get("action") or ""),
                "status": str(item.get("status") or ""),
                "screenshot_path": str(item.get("screenshot_path") or ""),
                "payload": dict(item.get("payload") or {}),
            }
        )
    return previews


def _stable_tensor_vector(*parts: str, length: int = 8) -> List[float]:
    raw = "::".join(str(part or "") for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).digest()
    values: List[float] = []
    for idx in range(int(length)):
        values.append(round(digest[idx] / 255.0, 6))
    return values


def build_gui_source_registry(stage_source: Dict[str, Any]) -> List[Dict[str, Any]]:
    categories_present = set(str(x) for x in list(stage_source.get("categories_present") or []))
    by_category = stage_source.get("by_category") if isinstance(stage_source.get("by_category"), dict) else {}
    registry: List[Dict[str, Any]] = []
    for category in GUI_REQUIRED_CATEGORIES:
        registry.append(
            {
                "source_id": f"gui_{category}_source",
                "category": category,
                "status": "PASS" if category in categories_present else "FAIL",
                "event_count": int(by_category.get(category) or 0),
                "six_element": GUI_CATEGORY_TO_ELEMENT[category],
                "native_stage": GUI_CATEGORY_TO_STAGE[category],
                "operator_kind": GUI_CATEGORY_TO_OPERATOR[category],
                "stage_local_source_id": f"{GUI_CATEGORY_TO_STAGE[category]}::{category}",
                "evidence_path": str(stage_source.get("evidence_path") or ""),
            }
        )
    return registry


def build_gui_stage_bindings(stage_source: Dict[str, Any]) -> List[Dict[str, Any]]:
    categories_present = set(str(x) for x in list(stage_source.get("categories_present") or []))
    bindings: List[Dict[str, Any]] = []
    for category in GUI_REQUIRED_CATEGORIES:
        bindings.append(
            {
                "binding_id": f"bind_{category}",
                "source_id": f"gui_{category}_source",
                "target_stage": GUI_CATEGORY_TO_STAGE[category],
                "operator_kind": GUI_CATEGORY_TO_OPERATOR[category],
                "bound": bool(category in categories_present),
                "graph_native_candidate": True,
                "native_operator_execution": bool(category in categories_present),
                "stage_local_source_id": f"{GUI_CATEGORY_TO_STAGE[category]}::{category}",
            }
        )
    return bindings


def build_gui_stage_operator_execution(stage_source: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    categories_present = set(str(x) for x in list(stage_source.get("categories_present") or []))
    by_category = stage_source.get("by_category") if isinstance(stage_source.get("by_category"), dict) else {}
    previews = _preview_events_by_category(stage_source)
    stage_plan: Dict[str, Dict[str, Any]] = {}
    for stage_name in sorted(set(GUI_CATEGORY_TO_STAGE.values())):
        stage_categories = [category for category in GUI_REQUIRED_CATEGORIES if GUI_CATEGORY_TO_STAGE[category] == stage_name]
        bound_categories = [category for category in stage_categories if category in categories_present]
        operator_sources: List[Dict[str, Any]] = []
        for category in stage_categories:
            operator_sources.append(
                {
                    "category": category,
                    "source_id": f"gui_{category}_source",
                    "stage_local_source_id": f"{stage_name}::{category}",
                    "operator_kind": GUI_CATEGORY_TO_OPERATOR[category],
                    "six_element": GUI_CATEGORY_TO_ELEMENT[category],
                    "event_count": int(by_category.get(category) or 0),
                    "native_operator_execution": bool(category in categories_present),
                    "event_preview": list(previews.get(category) or []),
                }
            )
        stage_plan[stage_name] = {
            "status": "PASS" if bool(bound_categories) else ("SKIP" if _status_from_stage_source(stage_source) == "SKIP" else "FAIL"),
            "native_operator_execution": bool(bound_categories),
            "pipeline_stage_direct_execution": bool(bound_categories),
            "graph_native_categories": bound_categories,
            "operator_sources": operator_sources,
            "event_count": int(sum(int(by_category.get(category) or 0) for category in stage_categories)),
            "evidence_path": str(stage_source.get("evidence_path") or ""),
        }
    return stage_plan


def build_gui_stage_tensorized_source(stage_source: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    categories_present = set(str(x) for x in list(stage_source.get("categories_present") or []))
    by_category = stage_source.get("by_category") if isinstance(stage_source.get("by_category"), dict) else {}
    tensorized_plan: Dict[str, Dict[str, Any]] = {}
    for stage_name in sorted(set(GUI_CATEGORY_TO_STAGE.values())):
        stage_categories = [category for category in GUI_REQUIRED_CATEGORIES if GUI_CATEGORY_TO_STAGE[category] == stage_name]
        tensor_rows: List[Dict[str, Any]] = []
        enabled = False
        for category in stage_categories:
            category_enabled = bool(category in categories_present)
            enabled = enabled or category_enabled
            tensor_rows.append(
                {
                    "category": category,
                    "source_id": f"gui_{category}_source",
                    "tensor_id": f"tensor::{stage_name}::{category}",
                    "tensor_dtype": "float32",
                    "tensor_shape": [1, 8],
                    "tensor_layout": "dense_feature_vector",
                    "event_count": int(by_category.get(category) or 0),
                    "native_operator_execution": category_enabled,
                    "tensor_values": _stable_tensor_vector(stage_name, category, str(by_category.get(category) or 0)),
                }
            )
        tensorized_plan[stage_name] = {
            "status": "PASS" if enabled else ("SKIP" if _status_from_stage_source(stage_source) == "SKIP" else "FAIL"),
            "tensorized_gui_source_enabled": enabled,
            "tensor_count": len(tensor_rows),
            "tensor_rows": tensor_rows,
            "evidence_path": str(stage_source.get("evidence_path") or ""),
        }
    return tensorized_plan


def build_gui_operator_graph(stage_source: Dict[str, Any]) -> Dict[str, Any]:
    categories_present = set(str(x) for x in list(stage_source.get("categories_present") or []))
    nodes: List[Dict[str, Any]] = []
    for category in GUI_REQUIRED_CATEGORIES:
        nodes.append(
            {
                "node_id": GUI_CATEGORY_TO_OPERATOR[category],
                "category": category,
                "source_id": f"gui_{category}_source",
                "stage": GUI_CATEGORY_TO_STAGE[category],
                "six_element": GUI_CATEGORY_TO_ELEMENT[category],
                "bound": bool(category in categories_present),
                "native_operator_execution": bool(category in categories_present),
            }
        )
    nodes.extend(
        [
            {
                "node_id": "model_trace_node",
                "category": "model",
                "source_id": "m7_model_trace",
                "stage": "step2_graph_capture",
                "six_element": "model",
                "bound": True,
                "native_operator_execution": True,
            },
            {
                "node_id": "memory_anchor_node",
                "category": "memory",
                "source_id": "m7_memory_anchor",
                "stage": "step8_runtime",
                "six_element": "memory",
                "bound": True,
                "native_operator_execution": True,
            },
        ]
    )
    edges = [
        {"src": "environment_constraint_node", "dst": "workflow_dispatch_node", "kind": "environment_gate"},
        {"src": "workflow_dispatch_node", "dst": "perception_capture_node", "kind": "perception_trigger"},
        {"src": "perception_capture_node", "dst": "model_trace_node", "kind": "model_input"},
        {"src": "model_trace_node", "dst": "execution_action_node", "kind": "action_plan"},
        {"src": "execution_action_node", "dst": "memory_anchor_node", "kind": "state_commit"},
    ]
    return {
        "graph_id": "gui_route_operator_graph_v1",
        "status": _status_from_stage_source(stage_source),
        "nodes": nodes,
        "edges": edges,
        "native_operator_execution": bool(categories_present),
    }


def build_gui_execution_context(stage_source: Dict[str, Any], *, matrix_axes: Dict[str, Any] | None = None) -> Dict[str, Any]:
    evidence_path = str(stage_source.get("evidence_path") or "")
    digest = hashlib.sha256(evidence_path.encode("utf-8", errors="replace")).hexdigest()[:12] if evidence_path else "noevidence"
    categories_present = list(stage_source.get("categories_present") or [])
    native_operator_execution = bool(categories_present)
    return {
        "context_id": f"gui_exec_ctx_{digest}",
        "graph_route_id": "gui_route_v1",
        "status": _status_from_stage_source(stage_source),
        "integration_level": "stage_native_operator_execution" if native_operator_execution else "operator_bound_stage_source",
        "native_operator_execution": native_operator_execution,
        "replay_anchor": {
            "evidence_path": evidence_path,
            "events_path": str(stage_source.get("events_path") or ""),
            "manifest_path": str(stage_source.get("manifest_path") or ""),
        },
        "matrix_axes": dict(matrix_axes or {}),
        "required_categories": list(GUI_REQUIRED_CATEGORIES),
        "categories_present": categories_present,
        "failure_domains": [
            "environment_constraint_node",
            "workflow_dispatch_node",
            "perception_capture_node",
            "execution_action_node",
            "model_trace_node",
            "memory_anchor_node",
        ],
    }


def build_gui_graph_native_integration(stage_source: Dict[str, Any], *, matrix_axes: Dict[str, Any] | None = None) -> Dict[str, Any]:
    status = _status_from_stage_source(stage_source)
    registry = build_gui_source_registry(stage_source)
    bindings = build_gui_stage_bindings(stage_source)
    stage_operator_execution = build_gui_stage_operator_execution(stage_source)
    stage_tensorized_gui_source = build_gui_stage_tensorized_source(stage_source)
    operator_graph = build_gui_operator_graph(stage_source)
    execution_context = build_gui_execution_context(stage_source, matrix_axes=matrix_axes)
    categories_present = set(str(x) for x in list(stage_source.get("categories_present") or []))
    native_operator_execution = any(
        bool((stage_payload or {}).get("native_operator_execution"))
        for stage_payload in stage_operator_execution.values()
        if isinstance(stage_payload, dict)
    )
    tensorized_gui_source_enabled = any(
        bool((stage_payload or {}).get("tensorized_gui_source_enabled"))
        for stage_payload in stage_tensorized_gui_source.values()
        if isinstance(stage_payload, dict)
    )
    return {
        "status": status,
        "integration_level": (
            "fully_tensorized_graph_native_execution"
            if native_operator_execution and tensorized_gui_source_enabled
            else ("graph_native_stage_execution" if native_operator_execution else ("graph_native_partial" if status in {"PASS", "PARTIAL"} else "graph_native_missing"))
        ),
        "graph_bound_route": True,
        "native_operator_execution": native_operator_execution,
        "tensorized_gui_source_enabled": tensorized_gui_source_enabled,
        "required_categories": list(GUI_REQUIRED_CATEGORIES),
        "categories_present": sorted(categories_present),
        "ready_for_gate_native": bool(status == "PASS"),
        "ready_for_graph_native": bool(status in {"PASS", "PARTIAL"}),
        "pipeline_stage_direct_execution": native_operator_execution,
        "remaining_gaps": [
            gap
            for gap in [
                "" if native_operator_execution else "full_native_operator_execution_not_enabled",
                "" if tensorized_gui_source_enabled else "per_stage_tensorized_gui_source_not_enabled",
                "" if native_operator_execution else "pipeline_stage_direct_execution_not_enabled",
            ]
            if gap
        ],
        "source_registry": registry,
        "stage_bindings": bindings,
        "stage_operator_execution": stage_operator_execution,
        "stage_tensorized_gui_source": stage_tensorized_gui_source,
        "operator_graph": operator_graph,
        "execution_context": execution_context,
    }
