from __future__ import annotations

from typing import Any


CANONICAL_EXECUTION_PROFILES: dict[str, dict[str, Any]] = {
    "local_infer": {
        "runtime_mode": "local_infer",
        "lifecycle_phase": "infer",
        "orchestration_scope": "local",
        "cloud_dependency": False,
        "transport_required": False,
        "publish_manifest_required": False,
        "runtime_contract_required": False,
        "full_weight_manifest_required": False,
        "deploy_contract_required": False,
        "consume_contract_required": False,
        "state_abi_required": True,
        "bootstrap_contract_required": True,
        "flow_parameter_contract_required": True,
    },
    "local_train": {
        "runtime_mode": "local_train",
        "lifecycle_phase": "train",
        "orchestration_scope": "local",
        "cloud_dependency": False,
        "transport_required": False,
        "publish_manifest_required": False,
        "runtime_contract_required": True,
        "full_weight_manifest_required": True,
        "deploy_contract_required": False,
        "consume_contract_required": False,
        "state_abi_required": True,
        "bootstrap_contract_required": True,
        "flow_parameter_contract_required": True,
    },
    "edge_cloud_infer": {
        "runtime_mode": "edge_cloud_infer",
        "lifecycle_phase": "infer",
        "orchestration_scope": "edge_cloud",
        "cloud_dependency": True,
        "transport_required": True,
        "publish_manifest_required": True,
        "runtime_contract_required": True,
        "full_weight_manifest_required": False,
        "deploy_contract_required": True,
        "consume_contract_required": True,
        "state_abi_required": True,
        "bootstrap_contract_required": True,
        "flow_parameter_contract_required": True,
    },
    "edge_cloud_train": {
        "runtime_mode": "edge_cloud_train",
        "lifecycle_phase": "train",
        "orchestration_scope": "edge_cloud",
        "cloud_dependency": True,
        "transport_required": True,
        "publish_manifest_required": True,
        "runtime_contract_required": True,
        "full_weight_manifest_required": True,
        "deploy_contract_required": True,
        "consume_contract_required": False,
        "state_abi_required": True,
        "bootstrap_contract_required": True,
        "flow_parameter_contract_required": True,
    },
}


CANONICAL_BOOTSTRAP_TEMPLATES: dict[str, dict[str, Any]] = {
    "local_infer": {
        "bootstrap_family": "local_runtime_bootstrap",
        "bootstrap_kind": "edge_runtime_activation",
        "distributed_runtime_bootstrap_required": False,
        "bootstrap_steps": [
            "resolve_model_runtime",
            "bind_local_runtime_host",
            "load_model_artifact_or_manifest",
            "prepare_local_prompt_and_replay_anchor",
        ],
        "bootstrap_parameters": {
            "runtime_host_required": True,
            "deployment_target_required": True,
            "model_locator_required": True,
            "local_runtime_only": True,
        },
    },
    "local_train": {
        "bootstrap_family": "local_training_bootstrap",
        "bootstrap_kind": "local_training_activation",
        "distributed_runtime_bootstrap_required": False,
        "bootstrap_steps": [
            "resolve_training_workspace",
            "bind_local_training_runtime",
            "materialize_state_abi_and_runtime_contract",
            "prepare_checkpoint_output_root",
        ],
        "bootstrap_parameters": {
            "runtime_host_required": True,
            "deployment_target_required": True,
            "checkpoint_root_required": True,
            "local_runtime_only": True,
        },
    },
    "edge_cloud_infer": {
        "bootstrap_family": "edge_cloud_delivery_bootstrap",
        "bootstrap_kind": "cloud_publish_to_edge_consume",
        "distributed_runtime_bootstrap_required": False,
        "bootstrap_steps": [
            "load_publish_manifest",
            "bind_bridge_contract",
            "prepare_edge_delivery_channel",
            "activate_edge_consume_runtime",
        ],
        "bootstrap_parameters": {
            "source_side_required": True,
            "target_side_required": True,
            "delivery_channel_required": True,
            "consume_contract_required": True,
        },
    },
    "edge_cloud_train": {
        "bootstrap_family": "distributed_training_bootstrap",
        "bootstrap_kind": "cloud_training_activation",
        "distributed_runtime_bootstrap_required": True,
        "bootstrap_steps": [
            "load_distributed_runtime_bootstrap",
            "bind_training_runtime_host",
            "materialize_contract_manifest_and_system_manifest",
            "prepare_publish_handoff",
        ],
        "bootstrap_parameters": {
            "runtime_host_required": True,
            "deployment_target_required": True,
            "distributed_backend_required": True,
            "publish_handoff_required": True,
        },
    },
}


CANONICAL_FLOW_PARAMETER_TEMPLATES: dict[str, dict[str, Any]] = {
    "local_infer": {
        "flow_contract_kind": "local_inference_parameter_contract",
        "required_inputs": ["prompt", "state_abi", "consume_contract"],
        "required_outputs": ["edge_inference_result", "runtime_evidence", "replay_anchor"],
        "parameter_contract": {
            "selected_route": "m4_local",
            "selected_backend_family": "omlx_or_local_runtime",
            "audit_trace_enabled": True,
            "replay_anchor_required": True,
        },
    },
    "local_train": {
        "flow_contract_kind": "local_training_parameter_contract",
        "required_inputs": ["dataset_manifest", "state_abi", "runtime_contract"],
        "required_outputs": ["train_session", "full_weight_manifest", "stage_trace"],
        "parameter_contract": {
            "checkpoint_policy": "local_checkpoint_materialization",
            "audit_trace_enabled": True,
            "full_weight_manifest_required": True,
            "publish_handoff_enabled": False,
        },
    },
    "edge_cloud_infer": {
        "flow_contract_kind": "edge_cloud_inference_parameter_contract",
        "required_inputs": ["publish_manifest", "deploy_contract", "consume_contract", "state_abi"],
        "required_outputs": ["bridge_info", "edge_inference_result", "audit_replay_bundle"],
        "parameter_contract": {
            "delivery_channel": "cloud_to_edge_contract_bundle",
            "transport_strategy": "edge_cloud_protocol",
            "audit_trace_enabled": True,
            "consume_contract_required": True,
        },
    },
    "edge_cloud_train": {
        "flow_contract_kind": "edge_cloud_training_parameter_contract",
        "required_inputs": ["distributed_runtime_bootstrap", "contract_manifest", "system_execution_manifest", "state_abi"],
        "required_outputs": ["runtime_contract", "full_weight_manifest", "publish_manifest"],
        "parameter_contract": {
            "training_stage_scope": "stage1_stage2_stage3",
            "distributed_backend": "nccl",
            "audit_trace_enabled": True,
            "publish_manifest_required": True,
        },
    },
}


def canonical_profile_names() -> list[str]:
    return list(CANONICAL_EXECUTION_PROFILES.keys())


def canonical_execution_profile(
    profile_name: str,
    *,
    runtime_host: str = "",
    transport_strategy: str = "",
    deployment_target: str = "",
    stage_scope: str = "",
    model_scope: str = "",
    environment: str = "",
) -> dict[str, Any]:
    normalized = str(profile_name or "").strip()
    if normalized not in CANONICAL_EXECUTION_PROFILES:
        raise ValueError(f"unknown_canonical_execution_profile:{normalized}")
    base = dict(CANONICAL_EXECUTION_PROFILES[normalized])
    base.update(
        {
            "schema_version": "canonical_execution_profile_v1",
            "profile_name": normalized,
            "runtime_host": str(runtime_host or ""),
            "transport_strategy": str(transport_strategy or ""),
            "deployment_target": str(deployment_target or ""),
            "stage_scope": str(stage_scope or ""),
            "model_scope": str(model_scope or ""),
            "environment": str(environment or ""),
        }
    )
    return base


def canonical_delivery_profile(
    profile_name: str,
    *,
    source_side: str = "",
    target_side: str = "",
    runtime_host: str = "",
    deployment_target: str = "",
) -> dict[str, Any]:
    execution_profile = canonical_execution_profile(
        profile_name,
        runtime_host=runtime_host,
        deployment_target=deployment_target,
    )
    return {
        "schema_version": "canonical_delivery_profile_v1",
        "profile_name": str(execution_profile.get("profile_name") or ""),
        "runtime_mode": str(execution_profile.get("runtime_mode") or ""),
        "orchestration_scope": str(execution_profile.get("orchestration_scope") or ""),
        "source_side": str(source_side or ""),
        "target_side": str(target_side or ""),
        "runtime_host": str(runtime_host or ""),
        "deployment_target": str(deployment_target or ""),
        "transport_required": bool(execution_profile.get("transport_required")),
        "publish_manifest_required": bool(execution_profile.get("publish_manifest_required")),
        "runtime_contract_required": bool(execution_profile.get("runtime_contract_required")),
        "full_weight_manifest_required": bool(execution_profile.get("full_weight_manifest_required")),
        "deploy_contract_required": bool(execution_profile.get("deploy_contract_required")),
        "consume_contract_required": bool(execution_profile.get("consume_contract_required")),
        "state_abi_required": bool(execution_profile.get("state_abi_required")),
        "bootstrap_contract_required": bool(execution_profile.get("bootstrap_contract_required")),
        "flow_parameter_contract_required": bool(execution_profile.get("flow_parameter_contract_required")),
    }


def canonical_bootstrap_contract(
    profile_name: str,
    *,
    runtime_host: str = "",
    deployment_target: str = "",
    environment: str = "",
    distributed_runtime_bootstrap_path: str = "",
    bootstrap_source_side: str = "",
    bootstrap_target_side: str = "",
    launch_command: str = "",
    fetch_command: str = "",
    model_locator: str = "",
) -> dict[str, Any]:
    execution_profile = canonical_execution_profile(
        profile_name,
        runtime_host=runtime_host,
        deployment_target=deployment_target,
        environment=environment,
    )
    template = dict(CANONICAL_BOOTSTRAP_TEMPLATES[str(execution_profile.get("profile_name") or "")])
    return {
        "schema_version": "canonical_bootstrap_contract_v1",
        "profile_name": str(execution_profile.get("profile_name") or ""),
        "runtime_mode": str(execution_profile.get("runtime_mode") or ""),
        "lifecycle_phase": str(execution_profile.get("lifecycle_phase") or ""),
        "bootstrap_family": str(template.get("bootstrap_family") or ""),
        "bootstrap_kind": str(template.get("bootstrap_kind") or ""),
        "runtime_host": str(runtime_host or ""),
        "deployment_target": str(deployment_target or ""),
        "environment": str(environment or ""),
        "bootstrap_source_side": str(bootstrap_source_side or ""),
        "bootstrap_target_side": str(bootstrap_target_side or ""),
        "distributed_runtime_bootstrap_required": bool(template.get("distributed_runtime_bootstrap_required")),
        "distributed_runtime_bootstrap_path": str(distributed_runtime_bootstrap_path or ""),
        "bootstrap_steps": list(template.get("bootstrap_steps") or []),
        "bootstrap_parameters": {
            **(template.get("bootstrap_parameters") if isinstance(template.get("bootstrap_parameters"), dict) else {}),
            "launch_command": str(launch_command or ""),
            "fetch_command": str(fetch_command or ""),
            "model_locator": str(model_locator or ""),
        },
    }


def canonical_flow_parameter_contract(
    profile_name: str,
    *,
    runtime_host: str = "",
    deployment_target: str = "",
    environment: str = "",
    parameter_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    execution_profile = canonical_execution_profile(
        profile_name,
        runtime_host=runtime_host,
        deployment_target=deployment_target,
        environment=environment,
    )
    template = dict(CANONICAL_FLOW_PARAMETER_TEMPLATES[str(execution_profile.get("profile_name") or "")])
    parameter_contract = dict(template.get("parameter_contract") or {})
    if isinstance(parameter_overrides, dict):
        parameter_contract.update(parameter_overrides)
    return {
        "schema_version": "canonical_flow_parameter_contract_v1",
        "profile_name": str(execution_profile.get("profile_name") or ""),
        "runtime_mode": str(execution_profile.get("runtime_mode") or ""),
        "lifecycle_phase": str(execution_profile.get("lifecycle_phase") or ""),
        "runtime_host": str(runtime_host or ""),
        "deployment_target": str(deployment_target or ""),
        "environment": str(environment or ""),
        "flow_contract_kind": str(template.get("flow_contract_kind") or ""),
        "required_inputs": list(template.get("required_inputs") or []),
        "required_outputs": list(template.get("required_outputs") or []),
        "parameter_contract": parameter_contract,
    }
