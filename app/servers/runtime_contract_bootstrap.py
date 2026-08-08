from __future__ import annotations

import json

import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_REPO_ROOT = REPO_ROOT / "ComputeGraphCompiler-main"
for path in (REPO_ROOT, ENGINE_REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cgc_engine.pipeline import MegatrainEightStepPipeline, MegatrainPipelineConfig
from cgc_engine.pipeline_contract_common import (
    pipeline_contract_descriptor_from_report,
    pipeline_kernel_contract_artifacts_from_report,
)
from app.shared.task_type_contract import task_type_contract_ref


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, str(default)) or str(default)).strip())
    except Exception:
        return int(default)


def _env_str(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or default).strip()


def _basename_from_path(path: str) -> str:
    value = str(path or "").strip()
    if not value:
        return ""
    try:
        return Path(value).name or value
    except Exception:
        return value


def _infer_backend(default_backend: str) -> str:
    explicit = str(os.environ.get("CGC_MEGATRAIN_BACKEND", "") or "").strip().lower()
    if explicit:
        return explicit
    cuda_visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
    if default_backend:
        return default_backend
    if cuda_visible and cuda_visible.lower() not in {"none", "-1"}:
        return "cuda"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mlx"
    return "cpu"


def _infer_environment(default_environment: str) -> str:
    explicit = str(os.environ.get("CGC_MEGATRAIN_ENVIRONMENT", "") or "").strip()
    if explicit:
        return explicit
    if default_environment:
        return default_environment
    if str(os.environ.get("CGC_RAY_ADDRESS", "") or "").strip() or str(os.environ.get("RAY_ADDRESS", "") or "").strip():
        return "cloud_cluster"
    return "edge_local"


def _infer_model_name(default_model_name: str) -> str:
    explicit = str(os.environ.get("CGC_MEGATRAIN_MODEL_NAME", "") or "").strip()
    if explicit:
        return explicit
    cloud_model = str(os.environ.get("CGC_CLOUD_OPENAI_MODEL", "") or os.environ.get("CGC_CLOUD_MODEL_PATH", "")).strip()
    if cloud_model:
        return _basename_from_path(cloud_model)
    router_model = str(os.environ.get("CGC_EDGE_ROUTER_MODEL", "") or os.environ.get("CGC_OLLAMA_MODEL", "")).strip()
    if router_model:
        return router_model
    return default_model_name


def _infer_task_domain(default_task_domain: str, component_role: str) -> str:
    explicit = str(os.environ.get("CGC_MEGATRAIN_TASK_DOMAIN", "") or "").strip()
    if explicit:
        return explicit
    if default_task_domain:
        return default_task_domain
    if component_role in {"gateway_orchestrator", "router_runtime"}:
        return "agent"
    return "models"


def _infer_runtime_profile(default_runtime_profile: str) -> str:
    explicit = str(os.environ.get("CGC_MEGATRAIN_RUNTIME_PROFILE", "") or "").strip()
    if explicit:
        return explicit
    return default_runtime_profile or "auto"


def _infer_component_id(default_component_id: str, default_http_port: int | None, default_socket_port: int | None) -> str:
    explicit = str(os.environ.get("CGC_MEGATRAIN_COMPONENT_ID", "") or "").strip()
    if explicit:
        return explicit
    if default_component_id:
        return default_component_id
    http_port = _env_int("CGC_CLOUD_HTTP_PORT", default_http_port or 0)
    socket_port = default_socket_port or 0
    port = http_port or socket_port
    cloud_map = {
        50053: "deepseek_inst1",
        50063: "deepseek_inst2",
        50073: "deepseek_inst3",
        50083: "deepseek_inst4",
        50052: "deepseek_inst1_socket",
        50062: "deepseek_inst2_socket",
        50072: "deepseek_inst3_socket",
        50082: "deepseek_inst4_socket",
    }
    return cloud_map.get(port, default_component_id or f"component_{port or 'runtime'}")


def _infer_component_role(default_component_role: str) -> str:
    explicit = str(os.environ.get("CGC_MEGATRAIN_COMPONENT_ROLE", "") or "").strip()
    if explicit:
        return explicit
    return default_component_role or "model_runtime"


def _infer_export_dir(component_id: str, discovery_root: str) -> str:
    explicit = str(os.environ.get("CGC_MEGATRAIN_EXPORT_DIR", "") or "").strip()
    if explicit:
        return explicit
    root = str(discovery_root or "").strip()
    if root:
        return str(Path(root).expanduser() / component_id)
    return ""


def _device_for_backend(backend: str) -> torch.device:
    lowered = str(backend or "").strip().lower()
    if lowered == "cuda":
        return torch.device("cuda:0")
    if lowered == "mlx":
        return torch.device("mps")
    if lowered == "ascend":
        return torch.device("npu:0")
    return torch.device("cpu")


def _infer_state_kind() -> str:
    return (
        _env_str("CGC_RUNTIME_STATE_KIND")
        or _env_str("CGC_MEGATRAIN_STATE_KIND")
        or _env_str("CGC_STATE_KIND")
        or "kda_state_v1"
    )


def _infer_state_codec() -> str:
    return (
        _env_str("CGC_RUNTIME_STATE_CODEC")
        or _env_str("CGC_MEGATRAIN_STATE_CODEC")
        or _env_str("CGC_STATE_CODEC")
        or _env_str("CGC_CLOUD_STATE_CODEC")
        or "cq4"
    )


def _infer_protocol_family(state_kind: str) -> str:
    explicit = _env_str("CGC_RUNTIME_PROTOCOL_FAMILY") or _env_str("CGC_MEGATRAIN_PROTOCOL_FAMILY")
    if explicit:
        return explicit
    if str(state_kind or "").strip().lower().startswith("kda_state"):
        return "trueorthokda"
    return "generic_runtime"


def _infer_requested_dispatch_backend() -> str:
    explicit = (
        _env_str("CGC_MEGATRAIN_REQUESTED_DISPATCH_BACKEND")
        or _env_str("CGC_REQUESTED_DISPATCH_BACKEND")
        or _env_str("CGC_RUNTIME_DISPATCH_BACKEND")
    )
    if explicit:
        return explicit
    deepep_mode = _env_str("CGC_DEEPEP_MODE").lower()
    if deepep_mode and deepep_mode not in {"0", "false", "off", "disabled", "none"}:
        return "deepep"
    if _env_bool("CGC_M76_ENABLE_DEEPEP", False):
        return "deepep"
    return "native_sglang"


def _infer_requested_distributed_runtime(*, enable_nccl: bool, use_colossalai: bool, tp: int, pp: int, ep: int) -> str:
    explicit = (
        _env_str("CGC_MEGATRAIN_REQUESTED_DISTRIBUTED_RUNTIME")
        or _env_str("CGC_REQUESTED_DISTRIBUTED_RUNTIME")
        or _env_str("CGC_DISTRIBUTED_RUNTIME")
    )
    if explicit:
        return explicit
    if use_colossalai:
        return "colossalai"
    if enable_nccl or max(tp, pp, ep) > 1:
        return "nccl"
    return "single_process"


def _infer_requested_storage_backend(*, enable_gds: bool, enable_spdk: bool) -> str:
    explicit = (
        _env_str("CGC_MEGATRAIN_REQUESTED_STORAGE_BACKEND")
        or _env_str("CGC_REQUESTED_STORAGE_BACKEND")
        or _env_str("CGC_STORAGE_BACKEND")
    )
    if explicit:
        return explicit
    if enable_gds and enable_spdk:
        return "gds_spdk"
    if enable_gds:
        return "gds"
    if enable_spdk:
        return "spdk"
    return "posix"


def _infer_enable_pd(*, protocol_family: str, requested_dispatch_backend: str) -> bool:
    default = str(protocol_family or "").strip().lower() == "trueorthokda" or str(requested_dispatch_backend or "").strip().lower() == "deepep"
    return _env_bool("CGC_MEGATRAIN_ENABLE_PD", _env_bool("CGC_ENABLE_PD", _env_bool("CGC_PD_ENABLED", default)))


def _infer_pd_endpoint(*, enable_pd: bool) -> str:
    explicit = (
        _env_str("CGC_MEGATRAIN_PD_ENDPOINT")
        or _env_str("CGC_PD_ENDPOINT")
        or _env_str("CGC_PD_SERVICE_ENDPOINT")
        or _env_str("CGC_EDGE_CLOUD_PD_ENDPOINT")
    )
    if explicit:
        return explicit
    return "localhost:50051" if enable_pd else ""


def _infer_pd_mode(*, enable_pd: bool) -> str:
    explicit = _env_str("CGC_MEGATRAIN_PD_MODE") or _env_str("CGC_PD_MODE")
    if explicit:
        return explicit
    return "cloud_prefill_edge_decode" if enable_pd else "disabled"


def _infer_require_pd_service(*, protocol_family: str, enable_pd: bool) -> bool:
    default = bool(enable_pd) or str(protocol_family or "").strip().lower() == "trueorthokda"
    return _env_bool("CGC_MEGATRAIN_REQUIRE_PD_SERVICE", _env_bool("CGC_REQUIRE_PD_SERVICE", default))


def _read_json_file(path: str) -> dict[str, Any]:
    target = str(path or "").strip()
    if not target:
        return {}
    try:
        payload = json.loads(Path(target).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_family_name(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    normalized_chars: list[str] = []
    for ch in raw:
        if ch.isalnum():
            normalized_chars.append(ch)
        else:
            normalized_chars.append("_")
    normalized = "".join(normalized_chars)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _component_matrix_from_manifest(components: list[dict[str, Any]]) -> dict[str, list[str]]:
    matrix: dict[str, list[str]] = {}
    for component in components:
        if not isinstance(component, dict):
            continue
        role = str(component.get("component_role") or "").strip()
        component_id = str(component.get("component_id") or "").strip()
        if not role or not component_id:
            continue
        matrix.setdefault(role, []).append(component_id)
    return matrix


def _first_component_family(
    components: list[dict[str, Any]],
    *,
    roles: set[str],
) -> str:
    for component in components:
        if not isinstance(component, dict):
            continue
        role = str(component.get("component_role") or "").strip()
        if role not in roles:
            continue
        model_name = str(component.get("model_name") or "").strip()
        if model_name:
            return _normalize_family_name(_basename_from_path(model_name))
    return ""


def _derive_system_profile_payload(
    *,
    system_execution_manifest_path: str,
    component_id: str,
    component_role: str,
    model_name: str,
    environment: str,
    runtime_profile: str,
    protocol_family: str,
    state_kind: str,
    state_codec: str,
    requested_dispatch_backend: str,
    requested_distributed_runtime: str,
    requested_storage_backend: str,
    pd_mode: str,
    required: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json_file(system_execution_manifest_path)
    explicit_profile = manifest.get("system_profile") if isinstance(manifest.get("system_profile"), dict) else {}
    components = manifest.get("components") if isinstance(manifest.get("components"), list) else []
    routing_edges = manifest.get("routing_edges") if isinstance(manifest.get("routing_edges"), list) else []

    typed_components = [component for component in components if isinstance(component, dict)]
    component_matrix = _component_matrix_from_manifest(typed_components)
    required_components = [
        str(component.get("component_id") or "").strip()
        for component in typed_components
        if bool(component.get("required")) and str(component.get("component_id") or "").strip()
    ]

    llm_roles = {"llm_runtime", "model_runtime", "cloud_edge_runtime"}
    llm_component_family = str(
        ((explicit_profile.get("component_families") or {}).get("llm_component_family"))
        if isinstance(explicit_profile.get("component_families"), dict)
        else ""
    ).strip()
    if not llm_component_family:
        llm_component_family = _first_component_family(typed_components, roles=llm_roles)
    if not llm_component_family and component_role in llm_roles:
        llm_component_family = _normalize_family_name(_basename_from_path(model_name))

    llm_instance_count = 0
    explicit_families = explicit_profile.get("component_families") if isinstance(explicit_profile.get("component_families"), dict) else {}
    try:
        llm_instance_count = int(explicit_families.get("llm_instance_count") or 0)
    except Exception:
        llm_instance_count = 0
    if llm_instance_count <= 0:
        llm_instance_count = len([component for component in typed_components if str(component.get("component_role") or "").strip() in llm_roles])
    if llm_instance_count <= 0 and llm_component_family:
        llm_instance_count = 1

    router_component_family = str(explicit_families.get("router_component_family") or "").strip()
    if not router_component_family:
        router_component_family = _first_component_family(typed_components, roles={"router_runtime"})

    gateway_component_family = str(explicit_families.get("gateway_component_family") or "").strip()
    if not gateway_component_family:
        gateway_component_family = _first_component_family(typed_components, roles={"gateway_orchestrator"})

    route_policy_family = str(explicit_families.get("route_policy_family") or "").strip()
    if not route_policy_family:
        route_policy_family = gateway_component_family or router_component_family

    deployment_mode = str(explicit_profile.get("deployment_mode") or "").strip()
    if not deployment_mode:
        deployment_mode = pd_mode if pd_mode and pd_mode != "disabled" else (environment or runtime_profile or "edge_local")

    profile_id = str(explicit_profile.get("profile_id") or "").strip()
    if not profile_id:
        profile_id = _normalize_family_name(
            str(manifest.get("system_id") or "")
            or "_".join(
                part
                for part in (
                    llm_component_family,
                    f"x{llm_instance_count}" if llm_instance_count > 0 else "",
                    router_component_family,
                    gateway_component_family,
                )
                if str(part or "").strip()
            )
            or f"{component_role}_{component_id}"
        )
    profile_version = str(explicit_profile.get("profile_version") or "v0.1").strip() or "v0.1"

    ref = {
        "profile_id": profile_id,
        "profile_version": profile_version,
        "source": "system_execution_manifest" if system_execution_manifest_path else "runtime_contract_bootstrap",
        "source_path": str(system_execution_manifest_path or "").strip(),
    }
    summary: dict[str, Any] = {
        "profile_id": profile_id,
        "profile_version": profile_version,
        "deployment_mode": deployment_mode,
        "llm_component_family": llm_component_family,
        "llm_instance_count": llm_instance_count,
        "router_component_family": router_component_family,
        "gateway_component_family": gateway_component_family,
        "route_policy_family": route_policy_family,
        "component_matrix": explicit_profile.get("component_matrix")
        if isinstance(explicit_profile.get("component_matrix"), dict)
        else component_matrix,
        "required_components": explicit_profile.get("required_components")
        if isinstance(explicit_profile.get("required_components"), list)
        else required_components,
        "routing_topology_profile": str(explicit_profile.get("routing_topology_profile") or "").strip(),
        "routing_edge_count": len(routing_edges),
        "environment_bootstrap_ref": {
            "environment": environment,
            "runtime_profile": runtime_profile,
            "requested_dispatch_backend": requested_dispatch_backend,
            "requested_distributed_runtime": requested_distributed_runtime,
            "requested_storage_backend": requested_storage_backend,
            "protocol_family": protocol_family,
            "state_kind": state_kind,
            "state_codec": state_codec,
        },
        "task_type_contract_ref": task_type_contract_ref(),
        "profile_binding_ref": (
            explicit_profile.get("profile_binding_ref")
            if isinstance(explicit_profile.get("profile_binding_ref"), dict)
            else {}
        ),
    }
    if isinstance(summary.get("profile_binding_ref"), dict):
        summary["profile_binding_ref"] = {
            **dict(summary["profile_binding_ref"]),
            "task_type_contract_ref": task_type_contract_ref(),
        }
    if not summary["required_components"] and required:
        summary["required_components"] = [component_id]
    if not summary["component_matrix"] and component_role and component_id:
        summary["component_matrix"] = {component_role: [component_id]}
    return ref, summary


def materialize_runtime_contract_artifacts(
    *,
    default_component_id: str = "",
    default_component_role: str = "",
    default_system_id: str = "",
    default_system_role: str = "multi_component_runtime",
    default_task_domain: str = "",
    default_model_name: str = "",
    default_runtime_profile: str = "auto",
    default_environment: str = "",
    default_backend: str = "",
    default_http_port: int | None = None,
    default_socket_port: int | None = None,
    required: bool = True,
    health_endpoint: str = "",
) -> dict[str, Any]:
    discovery_root = str(
        os.environ.get("CGC_MEGATRAIN_SYSTEM_MANIFEST_DISCOVERY_ROOT", "")
        or os.environ.get("CGC_RUNTIME_CONTRACT_ROOT", "")
        or ""
    ).strip()
    component_id = _infer_component_id(default_component_id, default_http_port, default_socket_port)
    component_role = _infer_component_role(default_component_role)
    backend = _infer_backend(default_backend)
    environment = _infer_environment(default_environment)
    model_name = _infer_model_name(default_model_name)
    task_domain = _infer_task_domain(default_task_domain, component_role)
    runtime_profile = _infer_runtime_profile(default_runtime_profile)
    state_kind = _infer_state_kind()
    state_codec = _infer_state_codec()
    protocol_family = _infer_protocol_family(state_kind)
    export_dir = _infer_export_dir(component_id, discovery_root)
    if not export_dir:
        return {
            "status": "SKIP",
            "reason": "missing_export_dir",
            "component_id": component_id,
            "task_type_contract_ref": task_type_contract_ref(),
        }

    resolved_health = str(os.environ.get("CGC_MEGATRAIN_COMPONENT_HEALTH_ENDPOINT", "") or "").strip() or health_endpoint
    system_id = str(os.environ.get("CGC_MEGATRAIN_SYSTEM_ID", "") or "").strip() or default_system_id
    system_role = str(os.environ.get("CGC_MEGATRAIN_SYSTEM_ROLE", "") or "").strip() or default_system_role
    tp_size = _env_int("CGC_SGLANG_TP_SIZE", 1)
    pp_size = _env_int("CGC_SGLANG_PP_SIZE", 1)
    ep_size = _env_int("CGC_SGLANG_EP_SIZE", 1)
    enable_nccl = _env_bool("CGC_MEGATRAIN_ENABLE_NCCL", _env_bool("CGC_SGLANG_USE_NCCL", backend == "cuda"))
    enable_cuda_graph = _env_bool("CGC_MEGATRAIN_ENABLE_CUDA_GRAPH", False)
    use_colossalai = _env_bool("CGC_MEGATRAIN_USE_COLOSSALAI", False)
    enable_gds = _env_bool("CGC_GDS_ENABLED", False)
    enable_spdk = _env_bool("CGC_SPDK_ENABLED", False)
    requested_dispatch_backend = _infer_requested_dispatch_backend()
    requested_distributed_runtime = _infer_requested_distributed_runtime(
        enable_nccl=enable_nccl,
        use_colossalai=use_colossalai,
        tp=tp_size,
        pp=pp_size,
        ep=ep_size,
    )
    colossalai_plugin = (
        _env_str("CGC_MEGATRAIN_COLOSSALAI_PLUGIN")
        or _env_str("CGC_COLOSSALAI_PLUGIN")
        or ("HybridParallelPlugin" if use_colossalai else "")
    )
    requested_storage_backend = _infer_requested_storage_backend(
        enable_gds=enable_gds,
        enable_spdk=enable_spdk,
    )
    expected_zero_copy = _env_bool("CGC_MEGATRAIN_EXPECT_ZERO_COPY", protocol_family == "trueorthokda")
    enable_pd = _infer_enable_pd(
        protocol_family=protocol_family,
        requested_dispatch_backend=requested_dispatch_backend,
    )
    pd_endpoint = _infer_pd_endpoint(enable_pd=enable_pd)
    pd_mode = _infer_pd_mode(enable_pd=enable_pd)
    pd_prefix_cache = _env_bool("CGC_MEGATRAIN_PD_PREFIX_CACHE", _env_bool("CGC_PD_PREFIX_CACHE", True))
    require_pd_service = _infer_require_pd_service(protocol_family=protocol_family, enable_pd=enable_pd)
    runtime_protocol_contract = {
        "protocol_family": protocol_family,
        "state_kind": state_kind,
        "state_codec": state_codec,
        "task_type_contract_ref": task_type_contract_ref(),
        "expected_zero_copy": expected_zero_copy,
        "enable_nccl": enable_nccl,
        "enable_cuda_graph": enable_cuda_graph,
        "requested_dispatch_backend": requested_dispatch_backend,
        "requested_distributed_runtime": requested_distributed_runtime,
        "use_colossalai": use_colossalai,
        "colossalai_plugin": colossalai_plugin,
        "requested_storage_backend": requested_storage_backend,
        "enable_gds": enable_gds,
        "enable_spdk": enable_spdk,
        "enable_pd": enable_pd,
        "pd_endpoint": pd_endpoint,
        "pd_mode": pd_mode,
        "pd_prefix_cache": pd_prefix_cache,
        "require_pd_service": require_pd_service,
    }
    config = MegatrainPipelineConfig(
        task_type="inference",
        backend=backend,
        environment=environment,
        task_domain=task_domain,
        model_name=model_name,
        dtype=torch.bfloat16,
        hf_model_path=str(os.environ.get("CGC_CLOUD_MODEL_PATH", "") or ""),
        load_weights=bool(str(os.environ.get("CGC_CLOUD_MODEL_PATH", "") or "").strip()),
        export_dir=export_dir,
        runtime_profile=runtime_profile,
        parallel_tp_size=tp_size,
        parallel_pp_size=pp_size,
        parallel_ep_size=ep_size,
        enable_pd=enable_pd,
        pd_endpoint=pd_endpoint,
        pd_prefix_cache=pd_prefix_cache,
        enable_nccl=enable_nccl,
        enable_cuda_graph=enable_cuda_graph,
        use_colossalai=use_colossalai,
        component_id=component_id,
        component_role=component_role,
        component_required=_env_bool("CGC_MEGATRAIN_COMPONENT_REQUIRED", required),
        component_health_endpoint=resolved_health,
        system_id=system_id,
        system_role=system_role,
        system_manifest_autodiscover=_env_bool("CGC_MEGATRAIN_SYSTEM_MANIFEST_AUTODISCOVER", True),
        system_manifest_discovery_root=discovery_root,
        system_manifest_components=str(os.environ.get("CGC_MEGATRAIN_SYSTEM_MANIFEST_COMPONENTS", "") or ""),
        system_manifest_routing_edges=str(os.environ.get("CGC_MEGATRAIN_SYSTEM_MANIFEST_ROUTING_EDGES", "") or ""),
        system_manifest_required_components=str(
            os.environ.get("CGC_MEGATRAIN_SYSTEM_MANIFEST_REQUIRED_COMPONENTS", "") or ""
        ),
        system_manifest_optional_components=str(
            os.environ.get("CGC_MEGATRAIN_SYSTEM_MANIFEST_OPTIONAL_COMPONENTS", "") or ""
        ),
    )
    report = MegatrainEightStepPipeline(config).materialize_contract_artifacts_only(
        device=_device_for_backend(backend)
    )
    artifacts = pipeline_kernel_contract_artifacts_from_report(report)
    descriptor = pipeline_contract_descriptor_from_report(report)
    system_execution_manifest_path = str(
        (report.get("system_execution_manifest") or {}).get("artifact_path") or ""
    )
    system_profile_ref, system_profile_summary = _derive_system_profile_payload(
        system_execution_manifest_path=system_execution_manifest_path,
        component_id=component_id,
        component_role=component_role,
        model_name=model_name,
        environment=environment,
        runtime_profile=runtime_profile,
        protocol_family=protocol_family,
        state_kind=state_kind,
        state_codec=state_codec,
        requested_dispatch_backend=requested_dispatch_backend,
        requested_distributed_runtime=requested_distributed_runtime,
        requested_storage_backend=requested_storage_backend,
        pd_mode=pd_mode,
        required=_env_bool("CGC_MEGATRAIN_COMPONENT_REQUIRED", required),
    )
    runtime_protocol_contract["system_profile_ref"] = system_profile_ref
    runtime_protocol_contract["system_profile_summary"] = system_profile_summary
    return {
        "status": "PASS",
        "component_id": component_id,
        "component_role": component_role,
        "export_dir": export_dir,
        "system_id": system_id,
        "artifacts": artifacts,
        "pipeline_contract_descriptor": descriptor,
        "protocol_family": protocol_family,
        "state_kind": state_kind,
        "state_codec": state_codec,
        "expected_zero_copy": expected_zero_copy,
        "enable_nccl": enable_nccl,
        "enable_cuda_graph": enable_cuda_graph,
        "requested_dispatch_backend": requested_dispatch_backend,
        "requested_distributed_runtime": requested_distributed_runtime,
        "use_colossalai": use_colossalai,
        "colossalai_plugin": colossalai_plugin,
        "requested_storage_backend": requested_storage_backend,
        "enable_gds": enable_gds,
        "enable_spdk": enable_spdk,
        "enable_pd": enable_pd,
        "pd_endpoint": pd_endpoint,
        "pd_mode": pd_mode,
        "pd_prefix_cache": pd_prefix_cache,
        "require_pd_service": require_pd_service,
        "system_profile_ref": system_profile_ref,
        "system_profile_summary": system_profile_summary,
        "runtime_protocol_contract": runtime_protocol_contract,
        "task_type_contract_ref": task_type_contract_ref(),
        "system_execution_manifest_path": system_execution_manifest_path,
    }


def bootstrap_runtime_contract_artifacts(**kwargs: Any) -> dict[str, Any]:
    try:
        return materialize_runtime_contract_artifacts(**kwargs)
    except Exception as exc:
        return {
            "status": "FAIL",
            "reason": "bootstrap_exception",
            "error": f"{type(exc).__name__}: {exc}",
            "kwargs": kwargs,
            "task_type_contract_ref": task_type_contract_ref(),
        }
