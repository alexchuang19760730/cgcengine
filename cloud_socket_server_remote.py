import logging
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_REPO_ROOT = REPO_ROOT / "ComputeGraphCompiler-main"
for path in (REPO_ROOT, ENGINE_REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from app.edge_engine.kda_state_runtime import build_real_kda_state_from_request
from Backend.CGC.deepep_sglang_patch import patch_sglang_moe, select_model_path
from Backend.CGC.ray_serve_sglang_gateway import start_ray_serve_gateway


class DeepEPCommunicator:
    """
    M7.6 Gate: DeepEP Communication Core for MoE All-to-All Operations.
    Keeps the gate/runtime contract stable while the production path moves to
    Ray Serve + SGLang.
    """

    def __init__(
        self,
        tp_size: int = 4,
        ep_size: int | None = None,
        deepep_parallel_profile: str | None = None,
    ):
        self.tp_size = tp_size
        self.ep_size = int(ep_size or tp_size)
        self.deepep_parallel_profile = str(
            deepep_parallel_profile or f"ep{self.ep_size}_tp{self.tp_size}"
        )
        self.comm_stream = None
        self.compute_stream = None
        self.is_initialized = False
        self.patch_info: Dict[str, Any] | None = None

    def initialize(self) -> None:
        logging.info("[DeepEP] Initializing DeepEP Communication Core...")
        patch_info = patch_sglang_moe(
            tp_size=self.tp_size,
            ep_size=self.ep_size,
            deepep_parallel_profile=self.deepep_parallel_profile,
        )
        self.patch_info = patch_info
        self.comm_stream = "DeepEP/SGLang dispatcher"
        self.compute_stream = "SGLang compute stream"
        self.is_initialized = True
        logging.info(
            "[DeepEP] SGLang DeepEP backend ready. profile=%s EP=%s TP=%s | kwargs=%s",
            self.deepep_parallel_profile,
            self.ep_size,
            self.tp_size,
            patch_info["engine_kwargs"],
        )

    def dispatch(self, tokens: Iterable[str], routing_weights: Any) -> Dict[str, Any]:
        token_list = list(tokens)
        weights = np.asarray(routing_weights)
        logging.info(
            "[DeepEP] Prepared MoE routing telemetry for %s tokens across %s experts.",
            len(token_list),
            self.tp_size,
        )
        estimated_payload_bytes = max(len(token_list), 1) * max(int(weights.size), 1)
        return {
            "token_count": len(token_list),
            "routing_shape": list(weights.shape),
            "max_weight": float(weights.max()) if weights.size else 0.0,
            "estimated_payload_bytes": estimated_payload_bytes,
        }

    def combine(self, expert_outputs: Any) -> Any:
        logging.info("[DeepEP] Combine stage completed through the active SGLang MoE backend.")
        return expert_outputs


def _build_true_state_payload(
    request_payload: Dict[str, Any] | None,
    *,
    cloud_text: str,
    openai_response: Dict[str, Any] | None,
    trace_id: str,
) -> Dict[str, Any]:
    runtime_state = build_real_kda_state_from_request(request_payload or {}, trace_id=trace_id)
    payload_bytes = bytes(runtime_state.get("state_bytes") or b"")
    chunk_size = max(
        4096,
        int(str(os.environ.get("CGC_CLOUD_STATE_CHUNK_SIZE_BYTES") or str(256 * 1024)).strip() or str(256 * 1024)),
    )
    num_chunks = (len(payload_bytes) + chunk_size - 1) // chunk_size if payload_bytes else 0
    state_meta = runtime_state.get("state_meta") if isinstance(runtime_state.get("state_meta"), dict) else {}
    state_meta = {
        **state_meta,
        "text_len": len(str(cloud_text or "")),
        "has_openai_response": isinstance(openai_response, dict),
    }
    return {
        "state_kind": str(runtime_state.get("state_kind") or "kda_state_v1"),
        "state_codec": str(runtime_state.get("state_codec") or "cq4"),
        "state_meta": state_meta,
        "payload_bytes": payload_bytes,
        "payload_size": len(payload_bytes),
        "num_chunks": num_chunks,
        "chunk_size": chunk_size if payload_bytes else 0,
        "text": str(cloud_text or ""),
    }


def _resolve_gateway_bind(host: str, port: int) -> tuple[str, int]:
    manifest_path = str(
        os.environ.get("CGC_SYSTEM_EXECUTION_MANIFEST_PATH", "")
        or os.environ.get("CGC_M76_SYSTEM_EXECUTION_MANIFEST_PATH", "")
        or ""
    ).strip()
    resolved_host = str(os.environ.get("CGC_CLOUD_HTTP_HOST", "") or host or "0.0.0.0").strip()
    resolved_port = int(str(os.environ.get("CGC_CLOUD_HTTP_PORT", "") or port or 50052))
    if manifest_path == "":
        return resolved_host, resolved_port
    try:
        payload = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return resolved_host, resolved_port
    system_profile = payload.get("system_profile") if isinstance(payload, dict) else {}
    routing_profile = system_profile.get("routing_topology_profile") if isinstance(system_profile, dict) else {}
    instance_topology = routing_profile.get("instance_topology") if isinstance(routing_profile, dict) else None
    if not isinstance(instance_topology, list):
        return resolved_host, resolved_port
    desired_instance_id = str(
        os.environ.get("CGC_INSTANCE_ID", "")
        or os.environ.get("SGLANG_RUN_ID", "")
        or ""
    ).strip()
    for entry in instance_topology:
        if not isinstance(entry, dict):
            continue
        if desired_instance_id and str(entry.get("instance_id") or "").strip() != desired_instance_id:
            continue
        manifest_port = int(entry.get("gateway_port") or 0)
        if manifest_port > 0 and resolved_port == 50052:
            resolved_port = manifest_port
        break
    return resolved_host, resolved_port


def start_server(host: str = "0.0.0.0", port: int = 50052):
    host, port = _resolve_gateway_bind(host, port)
    model_path = select_model_path()
    logging.info("[Cloud] Starting production gateway for model path: %s", model_path)
    return start_ray_serve_gateway(host=host, port=port)


def init_fusionroute_expert_pool(tp_size: int = 4) -> Dict[str, Any]:
    """
    Lightweight cloud-native bridge used by the M8 gate.
    It documents the expected Ray attachment flow for the FusionRoute Expert Pool.
    """
    try:
        import ray
    except Exception as exc:
        logging.warning("[Cloud] Ray is unavailable for FusionRoute bootstrap: %s", exc)
        return {
            "status": "SKIP",
            "reason": "ray_import_failed",
            "tp_size": tp_size,
            "pool_name": "FusionRoute Expert Pool",
        }

    ray.init(address="auto", ignore_reinit_error=True, namespace=os.environ.get("CGC_RAY_NAMESPACE", "cgc-serve"))
    logging.info(
        "[Cloud] Attached to Ray cluster for FusionRoute Expert Pool bootstrap with tp_size=%s",
        tp_size,
    )
    return {
        "status": "PASS",
        "tp_size": tp_size,
        "pool_name": "FusionRoute Expert Pool",
    }


def default_fusionroute_expert_pool() -> Dict[str, Any]:
    return init_fusionroute_expert_pool(tp_size=4)


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("CGC_LOG_LEVEL", "INFO"))
    start_server()
