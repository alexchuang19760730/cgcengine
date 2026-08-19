from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_ROUTING_TOPOLOGY: List[Dict[str, Any]] = [
    {
        "instance_id": "inst1",
        "host_label": "host2",
        "host_ip": "47.95.250.55",
        "ssh_host": "47.95.250.55",
        "ray_node_ip": "172.30.132.117",
        "visible_devices": "0,1,2,3",
        "gateway_port": 50053,
        "backend_port": 30000,
        "ray_port": 6379,
        "dist_port": 29500,
        "min_worker_port": 10002,
        "max_worker_port": 10999,
        "mem_fraction_static": 0.9,
        "cpu_offload_gb": 96,
        "tp_size": 4,
        "ep_size": 1,
        "nnodes": 1,
        "role": "deepseek_v4_flash_instance",
    },
    {
        "instance_id": "inst2",
        "host_label": "host1",
        "host_ip": "39.106.118.206",
        "ssh_host": "39.106.118.206",
        "ray_node_ip": "172.30.132.116",
        "visible_devices": "4,5,6,7",
        "gateway_port": 50063,
        "backend_port": 30010,
        "ray_port": 6389,
        "dist_port": 29510,
        "min_worker_port": 11002,
        "max_worker_port": 11999,
        "mem_fraction_static": 0.9,
        "cpu_offload_gb": 96,
        "tp_size": 4,
        "ep_size": 1,
        "nnodes": 1,
        "role": "deepseek_v4_flash_instance",
    },
    {
        "instance_id": "inst3",
        "host_label": "host2",
        "host_ip": "47.95.250.55",
        "ssh_host": "47.95.250.55",
        "ray_node_ip": "172.30.132.117",
        "visible_devices": "4,5,6,7",
        "gateway_port": 50073,
        "backend_port": 30020,
        "ray_port": 6399,
        "dist_port": 29520,
        "min_worker_port": 12002,
        "max_worker_port": 12999,
        "mem_fraction_static": 0.9,
        "cpu_offload_gb": 128,
        "tp_size": 4,
        "ep_size": 1,
        "nnodes": 1,
        "role": "deepseek_v4_flash_instance",
    },
    {
        "instance_id": "inst4",
        "host_label": "host1",
        "host_ip": "39.106.118.206",
        "ssh_host": "39.106.118.206",
        "ray_node_ip": "172.30.132.116",
        "visible_devices": "0,1,2,3",
        "gateway_port": 50083,
        "backend_port": 30030,
        "ray_port": 6398,
        "dist_port": 29530,
        "min_worker_port": 13002,
        "max_worker_port": 13999,
        "mem_fraction_static": 0.9,
        "cpu_offload_gb": 128,
        "tp_size": 4,
        "ep_size": 1,
        "nnodes": 1,
        "role": "deepseek_v4_flash_instance",
    },
]

SYNC_FILES = [
    "app/servers/cloud_socket_server.py",
    "ComputeGraphCompiler-main/Backend/CGC/ray_serve_sglang_gateway.py",
    "ComputeGraphCompiler-main/Backend/CGC/deepep_sglang_patch.py",
]


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _first_dict(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, dict) and value:
        first = next(iter(value.values()))
        if isinstance(first, dict):
            return dict(first)
    return {}


def _resolve_output_dir(base_manifest: Dict[str, Any], base_manifest_path: Path) -> Path:
    raw = str(base_manifest.get("export_dir") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return base_manifest_path.resolve().parent


def _artifact_path(output_dir: Path, name: str) -> str:
    return str((output_dir / name).resolve())


def _build_routing_topology_profile(base_manifest: Dict[str, Any]) -> Dict[str, Any]:
    runtime_contract = _first_dict(base_manifest, "runtime_protocol_contracts")
    target_model_family = str(runtime_contract.get("target_model_family") or "DeepSeek-V4-Flash")
    target_model_path = str(runtime_contract.get("target_model_path") or "/data/models/DeepSeek-V4-Flash")
    protocol_family = str(runtime_contract.get("protocol_family") or "trueorthokda")
    state_kind = str(runtime_contract.get("state_kind") or "kda_state_v1")
    state_codec = str(runtime_contract.get("state_codec") or "cq4")
    requested_distributed_runtime = str(runtime_contract.get("distributed_runtime_backend") or "single_process")
    topology_backend = str(runtime_contract.get("service_topology_backend") or "ray_cluster_dual_host")
    pd_mode = str(runtime_contract.get("pd_mode") or "cloud_prefill_edge_decode")
    gateway_ports = [int(entry["gateway_port"]) for entry in DEFAULT_ROUTING_TOPOLOGY]
    backend_ports = [int(entry["backend_port"]) for entry in DEFAULT_ROUTING_TOPOLOGY]
    return {
        "routing_mode": "fusionroute",
        "router_model": "minicpm5-1b",
        "cloud_instance_count": 4,
        "fusion_group_size": 4,
        "cloud_instance_role": "deepseek_v4_flash_pool",
        "cloud_model": target_model_family,
        "cloud_model_path": target_model_path,
        "edge_model": str(base_manifest.get("model_name") or "gui_agent_strict_closure_runtime"),
        "gateway_ports": gateway_ports,
        "backend_ports": backend_ports,
        "system_contract": {
            "contract_kind": "fusionroute_4instance_system",
            "routing_mode": "fusionroute",
            "required_instance_count": 4,
            "multi_instance_required": True,
            "acceptance_scope": "system_level_multi_instance",
        },
        "instance_contract": {
            "contract_kind": "fusionroute_subinstance_topology",
            "gpus_per_instance": 4,
            "tp_size": int(runtime_contract.get("deepep_tp_size") or 4),
            # The current launcher brings up one TP4 shard per instance; the
            # broader ep4_tp4 profile remains declared at the runtime-contract layer.
            "ep_size": 1,
            "nnodes": 1,
            "gateway_replicas": 1,
            "cpu_offload_gb_search_enabled": True,
            "role": "deepseek_v4_flash_instance",
            "deepep_parallel_profile": str(runtime_contract.get("deepep_parallel_profile") or "ep4_tp4"),
        },
        "instance_topology": [dict(entry) for entry in DEFAULT_ROUTING_TOPOLOGY],
        "bootstrap_policy": {
            "launcher_kind": "remote_runtime_ops_manifest_first",
            "clean_before_launch": True,
            "system_execution_manifest_env": "CGC_SYSTEM_EXECUTION_MANIFEST_PATH",
            "instance_id_env": "CGC_INSTANCE_ID",
            "sync_files": list(SYNC_FILES),
        },
        "service_topology_backend": topology_backend,
        "distributed_runtime_backend": requested_distributed_runtime,
        "edge_decode_enabled": pd_mode == "cloud_prefill_edge_decode",
        "cloud_prefill_enabled": True,
        "pd_mode": pd_mode,
        "protocol_family": protocol_family,
        "state_kind": state_kind,
        "state_codec": state_codec,
        "deepep_parallel_profile": str(runtime_contract.get("deepep_parallel_profile") or "ep4_tp4"),
    }


def build_upkg39_fusionroute_manifest(
    *,
    base_manifest_path: str,
    output_path: str = "",
) -> Dict[str, Any]:
    base_path = Path(base_manifest_path).expanduser().resolve()
    base_manifest = _read_json(base_path)
    output_dir = _resolve_output_dir(base_manifest, base_path)
    runtime_manifest_path = (
        Path(output_path).expanduser().resolve()
        if str(output_path or "").strip()
        else (output_dir / "system_execution_manifest.runtime.json").resolve()
    )

    routing_profile = _build_routing_topology_profile(base_manifest)
    topology_path = (output_dir / "four_instance_topology.json").resolve()
    bootstrap_runtime_path = (output_dir / "fusionroute_bootstrap_runtime.json").resolve()
    ready_report_path = (output_dir / "runtime_ready_report.json").resolve()

    artifacts = dict(base_manifest.get("artifacts") or {})
    artifacts.update(
        {
            "system_execution_manifest_runtime": str(runtime_manifest_path),
            "four_instance_topology": str(topology_path),
            "fusionroute_bootstrap_runtime": str(bootstrap_runtime_path),
            "runtime_ready_report": str(ready_report_path),
        }
    )

    runtime_manifest = dict(base_manifest)
    runtime_manifest["schema_version"] = "cgc.system_execution_manifest.v0.2"
    runtime_manifest["artifact_path"] = str(runtime_manifest_path)
    runtime_manifest["artifacts"] = artifacts
    runtime_manifest["system_profile"] = {
        "schema_version": "cgc.system_profile.v0.1",
        "mode_mapping": {
            "development_cli": "cgc",
            "user_cli": "cgc_edge",
            "remote_launcher": "remote_runtime_ops.py",
        },
        "context_profile": {
            "execution_context": dict(base_manifest.get("execution_context") or {}),
            "strategy_plan": dict(base_manifest.get("strategy_plan") or {}),
        },
        "routing_topology_profile": routing_profile,
        "formal_validation_profile": {
            "formal_suite": "upkg39_runtime_closure",
            "requires_runtime_ready_report": True,
            "required_artifacts": [
                "four_instance_topology.json",
                "fusionroute_bootstrap_runtime.json",
                "runtime_ready_report.json",
            ],
        },
    }

    topology_payload = {
        "schema_version": "cgc.fusionroute_topology.v0.1",
        "generated_at_s": time.time(),
        "source_manifest_path": str(base_path),
        "routing_topology_profile": routing_profile,
    }
    bootstrap_runtime_payload = {
        "schema_version": "cgc.fusionroute_bootstrap_runtime.v0.1",
        "generated_at_s": time.time(),
        "source_manifest_path": str(base_path),
        "runtime_manifest_path": str(runtime_manifest_path),
        "routing_mode": routing_profile.get("routing_mode"),
        "router_model": routing_profile.get("router_model"),
        "cloud_model": routing_profile.get("cloud_model"),
        "deepep_parallel_profile": routing_profile.get("deepep_parallel_profile"),
        "instance_ids": [entry["instance_id"] for entry in DEFAULT_ROUTING_TOPOLOGY],
        "hosts": sorted({str(entry["host_label"]) for entry in DEFAULT_ROUTING_TOPOLOGY}),
        "status": "pending_bootstrap",
    }
    ready_report_payload = {
        "schema_version": "cgc.runtime_ready_report.v0.1",
        "generated_at_s": time.time(),
        "source_manifest_path": str(runtime_manifest_path),
        "status": "pending_bootstrap",
        "required_instances": [entry["instance_id"] for entry in DEFAULT_ROUTING_TOPOLOGY],
        "ready_instances": [],
        "failed_instances": [],
    }

    _write_json(runtime_manifest_path, runtime_manifest)
    _write_json(topology_path, topology_payload)
    _write_json(bootstrap_runtime_path, bootstrap_runtime_payload)
    _write_json(ready_report_path, ready_report_payload)
    return {
        "ok": True,
        "base_manifest_path": str(base_path),
        "manifest_path": str(runtime_manifest_path),
        "four_instance_topology_path": str(topology_path),
        "fusionroute_bootstrap_runtime_path": str(bootstrap_runtime_path),
        "runtime_ready_report_path": str(ready_report_path),
    }
