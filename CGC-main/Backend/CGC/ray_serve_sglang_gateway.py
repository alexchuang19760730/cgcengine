import asyncio
import base64
import codecs
import inspect
import json
import logging
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
import traceback
import fcntl
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.edge_engine.kda_state_runtime import inspect_kda_state_bytes
from app.edge_engine.local_infer import EdgeLocalInferenceRuntime
from app.shared.profile_bundle_validator import validate_profile_bundle
from app.shared.swe_agent_profile import apply_swe_agent_request_contract
from app.shared.swe_agent_profile import apply_swe_agent_system_profile
from app.shared.task_type_contract import CGC_TASK_TYPE_HEADER
from app.shared.task_type_contract import KNOWN_TASK_TYPES
from app.shared.task_type_contract import TASK_TYPE_CONTRACT_VERSION
from app.shared.task_type_contract import TASK_TYPE_PREFILL
from app.shared.task_type_contract import normalize_task_type
from app.shared.task_type_contract import normalize_task_type_contract_ref
from app.shared.task_type_contract import normalize_task_type_iter
from app.shared.task_type_contract import resolve_task_type
from app.shared.task_type_contract import task_type_contract_ref
from Backend.CGC.deepep_sglang_patch import (
    ensure_vendored_sglang_on_path,
    patch_sglang_moe,
    select_model_path,
)
from cgc_engine.pd.dopd_schema import decode_dopd_resume_payload_v2
from cgc_engine.pd.dopd_schema import extract_dopd_resume_state_bytes

try:
    from Backend.CGC.mindspore.mindir_compiler import MindIRCompiler

    MINDSPORE_AVAILABLE = True
except ImportError:
    MindIRCompiler = None
    MINDSPORE_AVAILABLE = False

try:
    from Backend.CGC.network.rdma_passthrough import RDMACommunicator

    RDMA_AVAILABLE = True
except ImportError:
    RDMACommunicator = None
    RDMA_AVAILABLE = False

try:
    from Backend.CGC.scheduler.expert_migrator import HotExpertMigrator

    MIGRATOR_AVAILABLE = True
except ImportError:
    HotExpertMigrator = None
    MIGRATOR_AVAILABLE = False

try:
    from Backend.CGC.compiler.unified_compiler import (
        UnifiedIRCompiler,
        UnifiedIRInjector,
        inject_unified_ir_for_role,
    )

    UNIFIED_COMPILER_AVAILABLE = True
except ImportError:
    UnifiedIRCompiler = None
    UnifiedIRInjector = None
    inject_unified_ir_for_role = None
    UNIFIED_COMPILER_AVAILABLE = False


LOG = logging.getLogger("cgc.ray_serve_sglang_gateway")
REPO_ROOT = Path(__file__).resolve().parents[2]
GATEWAY_BUILD_ID = "2026-06-14-nondeepep3"
_PROFILE_ENV_NAMES = (
    "CGC_SGLANG_PROFILE_SETTINGS_PATH",
    "CGC_PROFILE_SETTINGS_PATH",
    "CGC_HOST2_BENCH_PROFILE_PATH",
)


@asynccontextmanager
async def _gateway_lifespan(_app: FastAPI):
    yield


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _env_value(*names: str) -> Optional[str]:
    for name in names:
        raw = str(os.environ.get(name) or "").strip()
        if raw:
            return raw
    return None


def _load_json_file(path_str: str) -> Dict[str, Any]:
    path_str = str(path_str or "").strip()
    if not path_str:
        return {}
    try:
        payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_contract_path(path_str: str, profile_source_path: str = "") -> str:
    path_str = str(path_str or "").strip()
    if not path_str:
        return ""
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    candidate_paths = []
    if profile_source_path:
        candidate_paths.append((Path(profile_source_path).resolve().parent / path_str).resolve())
    candidate_paths.append((REPO_ROOT / path_str).resolve())
    candidate_paths.append((REPO_ROOT / "docs" / "technical_whitepapers" / "examples" / path_str).resolve())
    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)
    return str(candidate_paths[0] if candidate_paths else path)


def _int_from_any(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _float_from_any(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _csv_set_from_env(name: str) -> set[str]:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _csv_set_from_env_multi(*names: str) -> set[str]:
    values: set[str] = set()
    for name in names:
        values.update(_csv_set_from_env(name))
    return values


def _load_gateway_profile_bundle() -> Dict[str, Any]:
    profile_path = str(_env_value(*_PROFILE_ENV_NAMES) or "").strip()
    profile = _load_json_file(profile_path)
    if not profile:
        return {}

    profile.setdefault("_profile_source_path", profile_path)
    bootstrap_contract_path = _resolve_contract_path(
        str(profile.get("bootstrap_contract_path") or ""),
        profile_path,
    )
    bootstrap_contract = _load_json_file(bootstrap_contract_path)
    if bootstrap_contract:
        bootstrap_contract.setdefault("source_path", bootstrap_contract_path)

    system_profile_ref = (
        profile.get("system_profile_ref")
        if isinstance(profile.get("system_profile_ref"), dict)
        else {}
    )
    system_manifest_path = _resolve_contract_path(
        str(system_profile_ref.get("source_path") or ""),
        profile_path,
    )
    system_manifest = _load_json_file(system_manifest_path)
    if system_manifest:
        system_manifest.setdefault("source_path", system_manifest_path)

    system_profile = (
        system_manifest.get("system_profile")
        if isinstance(system_manifest.get("system_profile"), dict)
        else {}
    )
    environment_bootstrap_ref = (
        system_profile.get("environment_bootstrap_ref")
        if isinstance(system_profile.get("environment_bootstrap_ref"), dict)
        else {}
    )
    profile_binding_ref = (
        system_profile.get("profile_binding_ref")
        if isinstance(system_profile.get("profile_binding_ref"), dict)
        else {}
    )
    model_contract_ref = (
        system_profile.get("model_contract_ref")
        if isinstance(system_profile.get("model_contract_ref"), dict)
        else {}
    )
    model_contract_path = _resolve_contract_path(
        str(model_contract_ref.get("source_path") or ""),
        str(system_manifest.get("source_path") or profile_path),
    )
    model_contract = _load_json_file(model_contract_path)
    if model_contract:
        model_contract.setdefault("source_path", model_contract_path)

    runtime_shape = (
        profile.get("runtime_shape")
        if isinstance(profile.get("runtime_shape"), dict)
        else {}
    )
    bootstrap_runtime_defaults = (
        bootstrap_contract.get("runtime_defaults")
        if isinstance(bootstrap_contract.get("runtime_defaults"), dict)
        else {}
    )
    bootstrap_runtime_env = (
        bootstrap_runtime_defaults.get("env")
        if isinstance(bootstrap_runtime_defaults.get("env"), dict)
        else {}
    )
    launch_env_defaults = {
        str(key): str(value)
        for key, value in bootstrap_runtime_env.items()
    }
    for key, value in (
        profile.get("launch_env_defaults")
        if isinstance(profile.get("launch_env_defaults"), dict)
        else {}
    ).items():
        launch_env_defaults[str(key)] = str(value)

    llm_runtime_component = None
    for component in system_manifest.get("components", []):
        if not isinstance(component, dict):
            continue
        if str(component.get("component_role") or "") == "llm_runtime":
            llm_runtime_component = component
            break

    primary_model = (
        model_contract.get("primary_model")
        if isinstance(model_contract.get("primary_model"), dict)
        else {}
    )
    draft_model = (
        model_contract.get("draft_model")
        if isinstance(model_contract.get("draft_model"), dict)
        else {}
    )
    speculative_decoding = (
        model_contract.get("speculative_decoding")
        if isinstance(model_contract.get("speculative_decoding"), dict)
        else {}
    )
    safe_runtime_shape = (
        model_contract.get("safe_runtime_shape")
        if isinstance(model_contract.get("safe_runtime_shape"), dict)
        else {}
    )
    distributed_binding = (
        profile.get("distributed_binding")
        if isinstance(profile.get("distributed_binding"), dict)
        else {}
    )

    derived_env = {
        "CGC_CLOUD_MODEL_PATH": str(
            (llm_runtime_component or {}).get("hf_model_path")
            or primary_model.get("model_path")
            or ""
        ),
        "CGC_SGLANG_SPECULATIVE_DRAFT_MODEL_PATH": str(
            (llm_runtime_component or {}).get("draft_model_path")
            or draft_model.get("model_path")
            or ""
        ),
        "CGC_SGLANG_SPECULATIVE_ALGORITHM": str(
            speculative_decoding.get("algorithm") or ""
        ),
        "CGC_MOE_A2A_BACKEND": str(
            speculative_decoding.get("dispatch_backend")
            or bootstrap_contract.get("requested_dispatch_backend")
            or ""
        ),
        "CGC_DEEPEP_PARALLEL_PROFILE": str(
            profile.get("distributed_binding", {}).get("parallel_profile")
            if isinstance(profile.get("distributed_binding"), dict)
            else ""
        ).replace("_", "_"),
        "CGC_SGLANG_TP_SIZE": str(
            runtime_shape.get("tp_size")
            or safe_runtime_shape.get("tp_size")
            or (llm_runtime_component or {}).get("parallel_tp_size")
            or ""
        ),
        "CGC_SGLANG_EP_SIZE": str(
            runtime_shape.get("ep_size")
            or safe_runtime_shape.get("ep_size")
            or (llm_runtime_component or {}).get("parallel_ep_size")
            or ""
        ),
        "CGC_SGLANG_NNODES": str(
            runtime_shape.get("nnodes")
            or safe_runtime_shape.get("nnodes")
            or (llm_runtime_component or {}).get("parallel_nnodes")
            or ""
        ),
    }
    parallel_profile = str(distributed_binding.get("parallel_profile") or "").strip()
    if parallel_profile:
        derived_env["CGC_DEEPEP_PARALLEL_PROFILE"] = parallel_profile

    for key, value in derived_env.items():
        if value:
            launch_env_defaults.setdefault(key, value)

    # #region debug-point A:gateway-profile-bundle
    import json, urllib.request; _p='.dbg/deepep-realchain.env'; _u,_s='http://127.0.0.1:7777/event','deepep-realchain'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"A","location":"ray_serve_sglang_gateway.py:_load_gateway_profile_bundle","msg":"[DEBUG] gateway profile bundle resolved","data":{"profile_settings_path":str(profile_path or ""), "execution_profile_binding_key":str(profile.get("execution_profile_binding_key") or ""), "bootstrap_contract_binding_key":str(profile.get("bootstrap_contract_binding_key") or ""), "dispatch_backend_from_speculative_decoding":str(speculative_decoding.get("dispatch_backend") or ""), "dispatch_backend_from_bootstrap_contract":str(bootstrap_contract.get("requested_dispatch_backend") or ""), "derived_cgc_moe_a2a_backend":str(derived_env.get("CGC_MOE_A2A_BACKEND") or ""), "parallel_profile":str(derived_env.get("CGC_DEEPEP_PARALLEL_PROFILE") or "")}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
    # #endregion

    expected_task_type_contract_ref = task_type_contract_ref()
    bundle_validation = validate_profile_bundle(
        profile_settings_path=profile_path,
        system_manifest_path=system_manifest_path,
        bootstrap_contract_path=bootstrap_contract_path,
    )
    task_type_contract_validation = (
        bundle_validation.get("task_type_contract_validation")
        if isinstance(bundle_validation.get("task_type_contract_validation"), dict)
        else {"status": "FAIL", "reason": "validator_missing_report"}
    )

    return {
        "profile_settings_path": profile_path,
        "profile_id": str(profile.get("profile_id") or ""),
        "execution_profile_binding_key": str(
            profile.get("execution_profile_binding_key") or ""
        ),
        "bootstrap_contract_binding_key": str(
            profile.get("bootstrap_contract_binding_key") or ""
        ),
        "flow_parameter_contract_binding_key": str(
            profile.get("flow_parameter_contract_binding_key") or ""
        ),
        "bootstrap_contract_path": bootstrap_contract_path,
        "bootstrap_contract_id": str(
            bootstrap_contract.get("bootstrap_contract_id") or ""
        ),
        "system_manifest_path": system_manifest_path,
        "system_profile_id": str(system_profile.get("profile_id") or ""),
        "model_contract_path": model_contract_path,
        "model_contract_id": str(model_contract.get("contract_id") or ""),
        "environment_bootstrap_ref": environment_bootstrap_ref,
        "runtime_shape": runtime_shape,
        "safe_runtime_shape": safe_runtime_shape,
        "distributed_binding": distributed_binding,
        "launch_env_defaults": launch_env_defaults,
        "task_type_contract_ref": normalize_task_type_contract_ref(
            bundle_validation.get("task_type_contract_ref") or expected_task_type_contract_ref
        ),
        "task_type_contract_validation": task_type_contract_validation,
    }


def _apply_profile_env_defaults(profile_bundle: Dict[str, Any]) -> Dict[str, str]:
    applied: Dict[str, str] = {}
    env_defaults = (
        profile_bundle.get("launch_env_defaults")
        if isinstance(profile_bundle.get("launch_env_defaults"), dict)
        else {}
    )
    for name, value in env_defaults.items():
        key = str(name).strip()
        env_value = str(value).strip()
        if not key or not env_value or str(os.environ.get(key) or "").strip():
            continue
        os.environ[key] = env_value
        applied[key] = env_value
    return applied


def _build_profile_extra_launch_args(profile_bundle: Dict[str, Any]) -> List[str]:
    bootstrap_contract = _load_json_file(
        str(profile_bundle.get("bootstrap_contract_path") or "")
    )
    runtime_defaults = (
        bootstrap_contract.get("runtime_defaults")
        if isinstance(bootstrap_contract.get("runtime_defaults"), dict)
        else {}
    )
    extra_args = runtime_defaults.get("extra_args")
    if not isinstance(extra_args, list):
        extra_args = []
    handled_value_flags = {
        "--model-path",
        "--host",
        "--port",
        "--device",
        "--tp-size",
        "--ep-size",
        "--nnodes",
        "--dist-init-addr",
        "--context-length",
        "--mem-fraction-static",
        "--max-total-tokens",
        "--chunked-prefill-size",
        "--max-running-requests",
        "--cpu-offload-gb",
        "--moe-a2a-backend",
        "--deepep-mode",
    }
    handled_switches = {"--use-ray", "--enable-deepep-waterfill", "--disable-cuda-graph"}
    filtered: List[str] = []
    skip_next = False
    for item in extra_args:
        token = str(item)
        if skip_next:
            skip_next = False
            continue
        if token in handled_value_flags:
            skip_next = True
            continue
        if token in handled_switches:
            continue
        filtered.append(token)
    extra_launch_args_env = str(os.environ.get("CGC_SGLANG_EXTRA_ARGS") or "").strip()
    if extra_launch_args_env:
        filtered.extend(shlex.split(extra_launch_args_env))
    return filtered


def _resolve_dist_init_addr(ray_module: Any, profile_bundle: Dict[str, Any]) -> str:
    explicit = str(os.environ.get("CGC_SGLANG_DIST_INIT_ADDR", "") or "").strip()
    if explicit:
        return explicit
    distributed_binding = (
        profile_bundle.get("distributed_binding")
        if isinstance(profile_bundle.get("distributed_binding"), dict)
        else {}
    )
    master_addr_env = str(distributed_binding.get("master_addr_env") or "").strip()
    master_port_env = str(distributed_binding.get("master_port_env") or "").strip()
    if master_addr_env and master_port_env:
        master_addr = str(os.environ.get(master_addr_env, "") or "").strip()
        master_port = str(os.environ.get(master_port_env, "") or "").strip()
        if master_addr and master_port:
            return f"{master_addr}:{master_port}"
    return f"{_detect_head_ip(ray_module)}:29500"


def _resolve_speculative_algorithm() -> Optional[str]:
    if _env_flag("CGC_SGLANG_ENABLE_DFLASH", default=False) or _env_flag("CGC_DFLASH_ENABLED", default=False):
        return "DFLASH"
    algorithm = _env_value("CGC_SGLANG_SPECULATIVE_ALGORITHM", "CGC_DFLASH_SPECULATIVE_ALGORITHM")
    if algorithm and algorithm.lower() in {"none", "off", "false", "0"}:
        return None
    return algorithm


def _build_speculative_launch_args() -> List[str]:
    algorithm = _resolve_speculative_algorithm()
    if not algorithm:
        return []

    command = ["--speculative-algorithm", algorithm]
    option_env_pairs = [
        (("CGC_SGLANG_SPECULATIVE_DRAFT_MODEL_PATH", "CGC_DFLASH_DRAFT_MODEL"), "--speculative-draft-model-path"),
        (("CGC_SGLANG_SPECULATIVE_DRAFT_MODEL_REVISION",), "--speculative-draft-model-revision"),
        (("CGC_SGLANG_SPECULATIVE_DRAFT_LOAD_FORMAT",), "--speculative-draft-load-format"),
        (("CGC_SGLANG_SPECULATIVE_NUM_STEPS",), "--speculative-num-steps"),
        (("CGC_SGLANG_SPECULATIVE_EAGLE_TOPK",), "--speculative-eagle-topk"),
        (
            ("CGC_SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS", "CGC_SGLANG_SPECULATIVE_DFLASH_BLOCK_SIZE"),
            "--speculative-num-draft-tokens",
        ),
        (("CGC_SGLANG_SPECULATIVE_DFLASH_BLOCK_SIZE",), "--speculative-dflash-block-size"),
        (
            ("CGC_SGLANG_SPECULATIVE_ACCEPT_THRESHOLD_SINGLE",),
            "--speculative-accept-threshold-single",
        ),
        (
            ("CGC_SGLANG_SPECULATIVE_ACCEPT_THRESHOLD_ACC",),
            "--speculative-accept-threshold-acc",
        ),
        (("CGC_SGLANG_SPECULATIVE_TOKEN_MAP",), "--speculative-token-map"),
        (("CGC_SGLANG_SPECULATIVE_ATTENTION_MODE",), "--speculative-attention-mode"),
        (
            ("CGC_SGLANG_SPECULATIVE_DRAFT_ATTENTION_BACKEND",),
            "--speculative-draft-attention-backend",
        ),
        (
            ("CGC_SGLANG_SPECULATIVE_DRAFT_WINDOW_SIZE",),
            "--speculative-draft-window-size",
        ),
        (
            ("CGC_SGLANG_SPECULATIVE_MOE_RUNNER_BACKEND",),
            "--speculative-moe-runner-backend",
        ),
        (
            ("CGC_SGLANG_SPECULATIVE_MOE_A2A_BACKEND", "CGC_MOE_A2A_BACKEND"),
            "--speculative-moe-a2a-backend",
        ),
        (
            ("CGC_SGLANG_SPECULATIVE_DRAFT_MODEL_QUANTIZATION",),
            "--speculative-draft-model-quantization",
        ),
        (
            ("CGC_SGLANG_SPECULATIVE_ADAPTIVE_CONFIG",),
            "--speculative-adaptive-config",
        ),
    ]
    for env_names, flag in option_env_pairs:
        value = _env_value(*env_names)
        if value is not None:
            command.extend([flag, value])

    if _env_flag("CGC_SGLANG_SPECULATIVE_ADAPTIVE", default=False):
        command.append("--speculative-adaptive")
    if _env_flag("CGC_SGLANG_SPECULATIVE_SKIP_DP_MLP_SYNC", default=False):
        command.append("--speculative-skip-dp-mlp-sync")
    return command


class DeepEPCommunicator:
    """Contract-friendly DeepEP dispatcher shim backed by SGLang config."""

    def __init__(
        self,
        tp_size: int = 4,
        ep_size: int | None = None,
        deepep_parallel_profile: str | None = None,
        moe_a2a_backend: str = "none",
    ):
        self.tp_size = tp_size
        self.ep_size = int(ep_size or tp_size)
        self.deepep_parallel_profile = str(
            deepep_parallel_profile or f"ep{self.ep_size}_tp{self.tp_size}"
        )
        self.moe_a2a_backend = str(moe_a2a_backend or "none").strip().lower()
        self.comm_stream = None
        self.compute_stream = None
        self.is_initialized = False
        self.patch_info: Optional[Dict[str, Any]] = None

    def initialize(self) -> None:
        if self.moe_a2a_backend != "deepep":
            self.patch_info = {
                "patched": False,
                "engine_kwargs": {
                    "deepep_parallel_profile": self.deepep_parallel_profile,
                    "tp_size": self.tp_size,
                    "ep_size": self.ep_size,
                    "moe_a2a_backend": self.moe_a2a_backend,
                },
            }
            self.comm_stream = "native-sglang-moe"
            self.compute_stream = "SGLang compute stream"
            self.is_initialized = True
            LOG.info(
                "[Gateway] Native SGLang MoE backend selected: %s",
                self.moe_a2a_backend,
            )
            return
        patch_info = patch_sglang_moe(
            tp_size=self.tp_size,
            ep_size=self.ep_size,
            deepep_parallel_profile=self.deepep_parallel_profile,
        )
        self.patch_info = patch_info
        self.comm_stream = "DeepEP/SGLang dispatcher"
        self.compute_stream = "SGLang compute stream"
        self.is_initialized = True
        LOG.info("[DeepEP] SGLang DeepEP backend ready: %s", patch_info["engine_kwargs"])

    def dispatch(self, tokens: Iterable[str], routing_weights: Any) -> Dict[str, Any]:
        token_list = list(tokens)
        weights = np.asarray(routing_weights)
        estimated_payload_bytes = max(len(token_list), 1) * max(int(weights.size), 1)
        return {
            "token_count": len(token_list),
            "routing_shape": list(weights.shape),
            "max_weight": float(weights.max()) if weights.size else 0.0,
            "estimated_payload_bytes": estimated_payload_bytes,
        }

    def combine(self, expert_outputs: Any) -> Any:
        if self.moe_a2a_backend == "deepep":
            LOG.info("[DeepEP] Combine stage completed through the active SGLang MoE backend.")
        return expert_outputs


@dataclass
class GatewayConfig:
    model_path: str
    tp_size: int
    ep_size: int
    attn_cp_size: int
    deepep_parallel_profile: str
    nnodes: int
    backend_host: str
    backend_port: int
    gateway_host: str
    gateway_port: int
    dist_init_addr: str
    mem_fraction_static: float
    cpu_offload_gb: int
    max_running_requests: int
    chunked_prefill_size: int
    context_length: int
    max_total_tokens: int
    moe_a2a_backend: str
    disable_cuda_graph: bool
    deepep_mode: str
    enable_deepep_waterfill: bool
    ray_namespace: str
    ray_use_spread: bool
    gateway_replicas: int
    runtime_env: Dict[str, Any]
    extra_launch_args: List[str]
    profile_settings_path: str
    execution_profile_binding_key: str
    bootstrap_contract_binding_key: str
    flow_parameter_contract_binding_key: str
    bootstrap_contract_path: str
    bootstrap_contract_id: str
    system_manifest_path: str
    system_profile_id: str
    model_contract_path: str
    model_contract_id: str
    protocol_family: str
    state_kind: str
    state_codec: str
    pd_mode: str
    task_type_contract_ref: Dict[str, Any]
    task_type_contract_validation: Dict[str, Any]

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


class SGLangBackendManager:
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.thread: Optional[threading.Thread] = None
        self.start_error: Optional[BaseException] = None
        self.log_path = REPO_ROOT / "logs" / "ray_serve_sglang_backend.log"
        self.lock_path = REPO_ROOT / "logs" / "ray_serve_sglang_backend.lock"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def build_launch_command(self) -> List[str]:
        python_bin = os.environ.get(
            "CGC_SGLANG_PYTHON_BIN",
            str(REPO_ROOT / ".venv_deepep_ssp" / "bin" / "python"),
        )
        command = [
            python_bin,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.config.model_path,
            "--host",
            self.config.backend_host,
            "--port",
            str(self.config.backend_port),
            "--device",
            "cuda",
            "--tp-size",
            str(self.config.tp_size),
            "--ep-size",
            str(self.config.ep_size),
            "--attn-cp-size",
            str(self.config.attn_cp_size),
            "--nnodes",
            str(self.config.nnodes),
            "--dist-init-addr",
            self.config.dist_init_addr,
            "--context-length",
            str(self.config.context_length),
            "--mem-fraction-static",
            str(self.config.mem_fraction_static),
            "--max-total-tokens",
            str(self.config.max_total_tokens),
            "--chunked-prefill-size",
            str(self.config.chunked_prefill_size),
            "--max-running-requests",
            str(self.config.max_running_requests),
            "--use-ray",
        ]
        if self.config.cpu_offload_gb > 0:
            command.extend(
                [
                    "--cpu-offload-gb",
                    str(self.config.cpu_offload_gb),
                ]
            )
        if self.config.moe_a2a_backend != "none":
            command.extend(
                [
                    "--moe-a2a-backend",
                    self.config.moe_a2a_backend,
                ]
            )
        if self.config.disable_cuda_graph:
            command.append("--disable-cuda-graph")
        if self.config.moe_a2a_backend == "deepep":
            command.extend(
                [
                    "--deepep-mode",
                    self.config.deepep_mode,
                ]
            )
        if self.config.moe_a2a_backend == "deepep" and self.config.enable_deepep_waterfill:
            command.append("--enable-deepep-waterfill")
        if self.config.extra_launch_args:
            command.extend(self.config.extra_launch_args)
        return command

    def _probe_ready_endpoint(self) -> Optional[str]:
        probe = self.probe_status()
        if bool(probe.get("ready")):
            return str(probe.get("probe_url") or "")
        return None

    def probe_status(self) -> Dict[str, Any]:
        probe_urls = [
            f"http://{self.config.backend_host}:{self.config.backend_port}/model_info",
            f"http://{self.config.backend_host}:{self.config.backend_port}/v1/models",
            f"http://{self.config.backend_host}:{self.config.backend_port}/health",
        ]
        failures: List[Dict[str, Any]] = []
        for url in probe_urls:
            try:
                response = requests.get(url, timeout=2)
            except requests.RequestException as exc:
                failures.append(
                    {
                        "url": url,
                        "status": "request_exception",
                        "detail": str(exc),
                    }
                )
                # #region debug-point A:backend-probe-failure
                import json, urllib.request; _p='.dbg/backend-30000-rdma.env'; _u,_s='http://127.0.0.1:7777/event','backend-30000-rdma'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"A","location":"ray_serve_sglang_gateway.py:SGLangBackendSupervisor._probe_ready_endpoint","msg":"[DEBUG] backend probe request failed","data":{"url":url,"backend_host":str(self.config.backend_host),"backend_port":int(self.config.backend_port),"error_type":exc.__class__.__name__,"error":str(exc)}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
                # #endregion
                continue
            if response.ok:
                # #region debug-point A:backend-probe-success
                import json, urllib.request; _p='.dbg/backend-30000-rdma.env'; _u,_s='http://127.0.0.1:7777/event','backend-30000-rdma'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"A","location":"ray_serve_sglang_gateway.py:SGLangBackendSupervisor._probe_ready_endpoint","msg":"[DEBUG] backend probe request succeeded","data":{"url":url,"status_code":int(response.status_code),"backend_host":str(self.config.backend_host),"backend_port":int(self.config.backend_port)}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
                # #endregion
                return {
                    "ready": True,
                    "probe_url": url,
                    "status_code": int(response.status_code),
                    "failures": failures,
                }
            failures.append(
                {
                    "url": url,
                    "status": "http_error",
                    "detail": f"HTTP {int(response.status_code)}",
                }
            )
        # #region debug-point A:backend-probe-exhausted
        import json, urllib.request; _p='.dbg/backend-30000-rdma.env'; _u,_s='http://127.0.0.1:7777/event','backend-30000-rdma'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"A","location":"ray_serve_sglang_gateway.py:SGLangBackendSupervisor._probe_ready_endpoint","msg":"[DEBUG] backend probe exhausted all candidate urls","data":{"probe_urls":probe_urls,"backend_host":str(self.config.backend_host),"backend_port":int(self.config.backend_port)}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
        # #endregion
        return {
            "ready": False,
            "probe_url": "",
            "status_code": 0,
            "failures": failures,
        }

    def is_healthy(self) -> bool:
        return self._probe_ready_endpoint() is not None

    def _launch_in_process(self) -> None:
        argv = self.build_launch_command()[3:]
        try:
            from sglang.launch_server import run_server
            from sglang.srt.server_args import prepare_server_args

            server_args = prepare_server_args(argv)
            run_server(server_args)
        except BaseException as exc:  # pragma: no cover - runtime evidence path
            self.start_error = exc
            with open(self.log_path, "a", encoding="utf-8") as log_handle:
                traceback.print_exc(file=log_handle)

    def _run_preflight_command(self, command: List[str]) -> str:
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception as exc:  # pragma: no cover - best effort diagnostics only
            return f"<error: {exc}>"
        output = (completed.stdout or completed.stderr or "").strip()
        return output or "<empty>"

    def _collect_deepep_preflight(self) -> List[str]:
        notes: List[str] = []
        infiniband_devices = sorted(Path("/sys/class/infiniband").glob("*"))
        if not infiniband_devices:
            notes.append(
                "No /sys/class/infiniband devices detected. "
                "DeepEP/NVSHMEM internode transport is unlikely to initialize on this host."
            )
        else:
            notes.append(
                "Infiniband devices: "
                + ", ".join(device.name for device in infiniband_devices)
            )

        p2p_matrix = self._run_preflight_command(
            ["nvidia-smi", "topo", "-p2p", "r"]
        )
        notes.append("nvidia-smi topo -p2p r:\n" + p2p_matrix)

        rdma_links = self._run_preflight_command(["rdma", "link", "show"])
        notes.append("rdma link show:\n" + rdma_links)

        ibv_devices = self._run_preflight_command(["ibv_devices"])
        notes.append("ibv_devices:\n" + ibv_devices)
        return notes

    def start(self, *, wait_until_ready: bool = True) -> None:
        if self.is_healthy():
            LOG.info("[Gateway] Reusing healthy SGLang backend at %s:%s", self.config.backend_host, self.config.backend_port)
            return
        with open(self.lock_path, "a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            if self.is_healthy():
                LOG.info(
                    "[Gateway] Healthy backend detected after acquiring startup lock at %s:%s",
                    self.config.backend_host,
                    self.config.backend_port,
                )
                return

            command = self.build_launch_command()
            LOG.info("[Gateway] Launching SGLang backend: %s", " ".join(command))
            runtime_env_vars = dict(self.config.runtime_env.get("env_vars", {}))
            os.environ.update({k: str(v) for k, v in runtime_env_vars.items()})
            # Ray Serve replicas with zero GPU resources may clear CUDA visibility.
            # Restore the original device mapping before launching the backend subprocess.
            visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
            fallback_visible_devices = str(
                runtime_env_vars.get("CGC_SGLANG_CUDA_VISIBLE_DEVICES")
                or runtime_env_vars.get("CUDA_VISIBLE_DEVICES")
                or ""
            ).strip()
            if visible_devices == "" and fallback_visible_devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = fallback_visible_devices
            nvidia_visible_devices = str(os.environ.get("NVIDIA_VISIBLE_DEVICES") or "").strip()
            fallback_nvidia_visible_devices = str(
                runtime_env_vars.get("CGC_SGLANG_NVIDIA_VISIBLE_DEVICES")
                or runtime_env_vars.get("NVIDIA_VISIBLE_DEVICES")
                or ""
            ).strip()
            if nvidia_visible_devices == "" and fallback_nvidia_visible_devices:
                os.environ["NVIDIA_VISIBLE_DEVICES"] = fallback_nvidia_visible_devices
            cuda_device_order = str(os.environ.get("CUDA_DEVICE_ORDER") or "").strip()
            fallback_cuda_device_order = str(
                runtime_env_vars.get("CGC_SGLANG_CUDA_DEVICE_ORDER")
                or runtime_env_vars.get("CUDA_DEVICE_ORDER")
                or ""
            ).strip()
            if cuda_device_order == "" and fallback_cuda_device_order:
                os.environ["CUDA_DEVICE_ORDER"] = fallback_cuda_device_order
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
            os.environ.setdefault("CGC_SGLANG_USE_RAY_ENGINE", "1")
            if self.config.moe_a2a_backend == "deepep":
                os.environ.setdefault("NVSHMEM_DISABLE_CUDA_VMM", "1")
            with open(self.log_path, "a", encoding="utf-8") as log_handle:
                log_handle.write(f"\n=== backend start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                log_handle.write(" ".join(command) + "\n")
                log_handle.write(
                    f"=== moe transport preflight ({self.config.moe_a2a_backend}) ===\n"
                )
                if self.config.moe_a2a_backend == "deepep":
                    for note in self._collect_deepep_preflight():
                        log_handle.write(note + "\n")
                else:
                    log_handle.write(
                        "DeepEP disabled; using standard SGLang MoE transport path.\n"
                    )
            self.start_error = None
            self.thread = threading.Thread(
                target=self._launch_in_process,
                name="cgc-sglang-backend",
                daemon=True,
            )
            self.thread.start()
            if wait_until_ready:
                self.wait_until_ready()

    def wait_until_ready(self, timeout_s: int = 900) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.start_error is not None:
                raise RuntimeError(
                    f"SGLang backend exited early with error: {self.start_error}. "
                    f"See {self.log_path} for details."
                )
            ready_url = self._probe_ready_endpoint()
            if ready_url is not None:
                LOG.info("[Gateway] SGLang backend is ready at %s", ready_url)
                return
            time.sleep(5)
        raise TimeoutError(f"SGLang backend did not become healthy within {timeout_s}s")


class GatewayRuntime:
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.deepep_comm = DeepEPCommunicator(
            tp_size=config.tp_size,
            ep_size=config.ep_size,
            deepep_parallel_profile=config.deepep_parallel_profile,
            moe_a2a_backend=config.moe_a2a_backend,
        )
        self.deepep_comm.initialize()
        self.rdma_comm = None
        self.expert_migrator = HotExpertMigrator() if MIGRATOR_AVAILABLE else None
        self._lock = threading.Lock()
        self._resume_sessions: Dict[str, Dict[str, Any]] = {}
        self._auto_publish_last_ts: Dict[str, float] = {}
        self._local_infer_runtime: Optional[EdgeLocalInferenceRuntime] = None
        self.enable_edge_resume = (
            str(self.config.pd_mode or "").strip() == "cloud_prefill_edge_decode"
            and _env_flag("CGC_DOPD_EXECUTE_EDGE_RESUME", default=True)
        )
        self.enable_auto_publish_handoff = (
            str(self.config.pd_mode or "").strip() == "cloud_prefill_edge_decode"
            and _env_flag("CGC_DOPD_AUTO_PUBLISH_HANDOFF", default=True)
        )
        if os.environ.get("USE_RDMA_PASSTHROUGH", "0") == "1" and RDMA_AVAILABLE:
            self.rdma_comm = RDMACommunicator()
            self.rdma_comm.initialize()

    def build_edge_matrix(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        payload_dict = payload if isinstance(payload, dict) else {}
        raw_matrix = headers.get("x-cgc-perception-matrix", "")
        if raw_matrix:
            try:
                edge_matrix = json.loads(raw_matrix)
            except json.JSONDecodeError:
                edge_matrix = {}
        else:
            edge_matrix = {}
        edge_matrix.setdefault("bw_mbps", float(headers.get("x-cgc-bw-mbps", "1000.0")))
        edge_matrix.setdefault("hardware_type", headers.get("x-cgc-hardware-type", "Nvidia_L20N"))
        edge_matrix.setdefault("environment", headers.get("x-cgc-environment", "edge_cloud"))
        edge_matrix.setdefault("task_type", headers.get("x-cgc-task-type", "prefill"))
        edge_matrix["task_type"] = resolve_task_type(
            edge_matrix.get("task_type"),
            headers.get(CGC_TASK_TYPE_HEADER),
            (payload_dict.get("metadata") or {}).get("task_type") if isinstance(payload_dict.get("metadata"), dict) else "",
            payload_dict.get("task_type"),
            default=TASK_TYPE_PREFILL,
        )
        edge_matrix.setdefault("model_family", str(payload_dict.get("model", "deepseek-v4-flash:latest")))
        return edge_matrix

    def prepare_request(self, payload: Dict[str, Any], edge_matrix: Dict[str, Any]) -> Dict[str, Any]:
        if (
            isinstance(payload, dict)
            and not payload.get("tools")
            and self._request_expects_thought_action(payload)
            and isinstance(payload.get("messages"), list)
        ):
            payload["messages"] = apply_swe_agent_system_profile(list(payload.get("messages") or []))
            profile_bundle = _load_gateway_profile_bundle()
            payload = apply_swe_agent_request_contract(
                payload,
                profile_settings_path=str(profile_bundle.get("profile_settings_path") or ""),
                bootstrap_contract_path=str(profile_bundle.get("bootstrap_contract_path") or ""),
                system_manifest_path=str(profile_bundle.get("system_manifest_path") or ""),
            )

        prompt = payload.get("prompt", "")
        if not prompt and "messages" in payload:
            prompt = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in payload.get("messages", [])
            )
        tokens = list(str(prompt)[:100])
        routing_weights = np.random.rand(self.deepep_comm.tp_size)
        aggregated_stats = self.deepep_comm.dispatch(tokens, routing_weights)
        aggregated_size = int(aggregated_stats["estimated_payload_bytes"])

        if self.expert_migrator is not None:
            self.expert_migrator.record_routing(tokens, routing_weights, source_node_id=1)
            self.expert_migrator.evaluate_and_migrate()

        if self.rdma_comm is not None and self.rdma_comm.is_initialized:
            mr_handle = self.rdma_comm.register_memory_region(gpu_tensor_ptr="0xGPU_ADDR", size=aggregated_size)
            self.rdma_comm.send_tensor_direct(mr_handle, remote_ip="172.30.132.117", remote_qpn=1024)

        compile_target = "unknown"
        if UNIFIED_COMPILER_AVAILABLE:
            # Gate 6.0: CGC engine 统一 IR 注入 vendored SGLang compute 路径
            # 四角色（tmax/uitars/hermes/cli_universe）统一通过 IR 注入，
            # 无需各自 patch SGLang 源码。注入器强制 linear_attn backend=triton，
            # 绕过 Blackwell SM100+ 上 flashinfer 0.6.x 缺 linear_attention 模块的问题。
            role = os.environ.get("CGC_FUSIONROUTE_ROLE", "hermes")
            injection_result = inject_unified_ir_for_role(
                role=role,
                perception_matrix=edge_matrix,
            )
            compile_target = injection_result.get("compiled", {}).get("compile_target", "unknown")
            LOG.info(f"[Gateway] CGC unified IR injected for role={role}: "
                     f"{injection_result.get('injection', {})}")

        if os.environ.get("USE_MINDSPORE_BACKEND", "0") == "1" and MINDSPORE_AVAILABLE and MindIRCompiler is not None:
            compiler = MindIRCompiler(target_device="Ascend")
            compiler.compile_graph(model_graph="MoE_SubGraph")

        return {
            "edge_matrix": edge_matrix,
            "aggregated_stats": aggregated_stats,
            "compile_target": compile_target,
        }

    def expected_resume_contract(self) -> Dict[str, str]:
        return {
            "profile_settings_path": str(self.config.profile_settings_path or ""),
            "execution_profile_binding_key": str(self.config.execution_profile_binding_key or ""),
            "bootstrap_contract_binding_key": str(self.config.bootstrap_contract_binding_key or ""),
            "flow_parameter_contract_binding_key": str(self.config.flow_parameter_contract_binding_key or ""),
            "bootstrap_contract_path": str(self.config.bootstrap_contract_path or ""),
            "bootstrap_contract_id": str(self.config.bootstrap_contract_id or ""),
            "system_manifest_path": str(self.config.system_manifest_path or ""),
            "system_profile_id": str(self.config.system_profile_id or ""),
            "model_contract_path": str(self.config.model_contract_path or ""),
            "model_contract_id": str(self.config.model_contract_id or ""),
            "protocol_family": str(self.config.protocol_family or ""),
            "state_kind": str(self.config.state_kind or ""),
            "state_codec": str(self.config.state_codec or ""),
            "pd_mode": str(self.config.pd_mode or ""),
        }

    @staticmethod
    def _contract_field_matches(expected: str, observed: str) -> bool:
        expected_str = str(expected or "").strip()
        observed_str = str(observed or "").strip()
        if not expected_str or not observed_str:
            return True
        if expected_str == observed_str:
            return True
        try:
            return Path(expected_str).name == Path(observed_str).name
        except Exception:
            return False

    def _get_local_infer_runtime(self) -> EdgeLocalInferenceRuntime:
        runtime = getattr(self, "_local_infer_runtime", None)
        if runtime is None:
            runtime = EdgeLocalInferenceRuntime()
            self._local_infer_runtime = runtime
        return runtime

    @staticmethod
    def _decode_embedded_state_bytes(
        *,
        request_payload: Dict[str, Any],
        resume_payload_meta: Dict[str, Any],
    ) -> bytes:
        inline_state_b64 = str(request_payload.get("state_bytes_b64") or "").strip()
        if inline_state_b64:
            return base64.b64decode(inline_state_b64.encode("ascii"), validate=True)
        return extract_dopd_resume_state_bytes(resume_payload_meta)

    @staticmethod
    def _build_edge_resume_state_meta(
        *,
        request_payload: Dict[str, Any],
        resume_payload_meta: Dict[str, Any],
        state_summary: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        state_meta: Dict[str, Any] = {}
        for source in (
            resume_payload_meta.get("layout_meta"),
            resume_payload_meta.get("metadata"),
            request_payload.get("metadata"),
            state_summary,
        ):
            if isinstance(source, dict):
                state_meta.update(source)
        state_meta.setdefault("session_id", str(request_payload.get("session_id") or ""))
        state_meta.setdefault("handoff_id", str(request_payload.get("handoff_id") or ""))
        state_meta.setdefault("phase_role", str(request_payload.get("phase_role") or ""))
        state_meta.setdefault("model_name", str(resume_payload_meta.get("model_name") or ""))
        state_meta.setdefault("kda_state_ref", str(resume_payload_meta.get("kda_state_ref") or ""))
        return state_meta

    @staticmethod
    def _extract_response_text(response_payload: Dict[str, Any]) -> str:
        choices = response_payload.get("choices")
        if not isinstance(choices, list):
            return ""
        parts: List[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            text = str(item.get("text") or item.get("content") or "")
                        else:
                            text = str(item)
                        if text:
                            parts.append(text)
                else:
                    text = str(content or "")
                    if text:
                        parts.append(text)
                continue
            text = str(choice.get("text") or "")
            if text:
                parts.append(text)
        return "".join(parts).strip()

    @staticmethod
    def _request_expects_thought_action(request_payload: Dict[str, Any]) -> bool:
        if not isinstance(request_payload, dict):
            return False
        if request_payload.get("tools"):
            return False
        messages = request_payload.get("messages")
        if not isinstance(messages, list):
            return False
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                text = "".join(
                    str(item.get("text") or item.get("content") or "")
                    if isinstance(item, dict)
                    else str(item)
                    for item in content
                )
            else:
                text = str(content or "")
            if not text:
                continue
            text_lower = text.lower()
            if (
                "thought/action format expected by the caller" in text
                or ("DISCUSSION" in text and "```bash" in text)
                or "one bash command block per response" in text
                or "Do NOT emit JSON tool calls" in text
                or "Do NOT emit XML tags such as <tool_calls>" in text
                or (
                    "response format:" in text_lower
                    and "discussion" in text_lower
                    and "```" in text
                    and (
                        "one shell command to run" in text_lower
                        or "final fenced command block" in text_lower
                        or "exactly one command" in text_lower
                    )
                )
            ):
                return True
        return False

    @staticmethod
    def _normalize_thought_action_text(cloud_text: str) -> str:
        text = str(cloud_text or "")
        if not text:
            return text
        text = re.sub(r"```python\s*submit(?:\(\))?\s*```", "```bash\nsubmit\n```", text)
        text = re.sub(r"```bash\s*submit\(\)\s*```", "```bash\nsubmit\n```", text)
        text = re.sub(r"```\s*submit\(\)\s*```", "```bash\nsubmit\n```", text)

        def sanitize_thought_text(thought_text: str) -> str:
            thought = str(thought_text or "")
            thought = re.sub(r"(?is)^<\|assistant\|>\s*", "", thought).strip()
            thought = re.sub(r"(?is)^<\|Assistant\|>\s*", "", thought).strip()
            thought = re.sub(r"(?is)</?think>", "", thought)
            thought = re.sub(r"(?is)<details\b[^>]*>.*?</details>", "", thought)
            thought = re.sub(r"(?is)</?summary\b[^>]*>", "", thought)
            thought = re.sub(r"(?im)^\s*</?AgentInput>\s*$", "", thought)
            thought = re.sub(
                r"(?im)^\s*</?(?:tool_set|tool_control|output)\b[^>]*>\s*$",
                "",
                thought,
            )
            thought = re.sub(r"(?im)^\s*</?ToolCall>\s*$", "", thought)
            thought = re.sub(
                r"(?is)<output_file\b[^>]*>.*?</output_file>",
                "",
                thought,
            )
            thought = re.sub(r"(?im)^\s*DISCUSSION\s*", "", thought)
            thought = re.sub(r"\n{3,}", "\n\n", thought)
            return thought.strip()

        def finalize(thought_text: str, command_text: str) -> str:
            thought = sanitize_thought_text(thought_text)
            command = str(command_text or "").strip()
            if not command:
                return text
            if not thought:
                thought = "DISCUSSION"
            if not thought.startswith("DISCUSSION"):
                thought = f"DISCUSSION\n{thought}"
            return f"{thought}\n\n```bash\n{command}\n```"

        for pattern in (
            r"<execute>(.*?)</execute>",
            r"<function>\s*execute\s*</function>.*?<value>(.*?)</value>",
            r"<bash>\s*(.*?)\s*</bash>",
            r"<\|command\|>\s*(.*?)\s*</command>",
            r"<command>\s*(.*?)</command>",
        ):
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                command = match.group(1).strip()
                thought = re.split(
                    r"<tool_calls>|<tool_call>|<invoke name=|<｜DSML｜>|<execute>|<function>\s*execute\s*</function>|<bash>|<\|command\|>|<command>",
                    text,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0].strip()
                if command:
                    return finalize(thought, command)

        invoke_match = re.search(
            r'<invoke\s+name="Bash".*?<parameter\s+name="command"[^>]*string_template="([^"]+)"',
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if invoke_match:
            command = invoke_match.group(1).strip()
            thought = re.split(
                r"<tool_calls>|<tool_call>|<tool_info>|<invoke name=|<｜DSML｜>",
                text,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            if command:
                return finalize(thought, command)

        dsml_tool_call_match = re.search(r"<\|tool_call\|>\s*(.*?)\s*<\|tool_call\|>", text, re.DOTALL | re.IGNORECASE)
        if dsml_tool_call_match:
            command = dsml_tool_call_match.group(1).strip()
            bash_wrapper_match = re.match(r'/bin/bash\s+-c\s+"(.*)"', command, re.DOTALL)
            if bash_wrapper_match:
                command = bash_wrapper_match.group(1).strip()
            thought = re.split(
                r"<\|tool_call\|>|<tool_calls>|<tool_call>|<tool_info>|<invoke name=|<｜DSML｜>",
                text,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            if command:
                return finalize(thought, command)

        plain_bash_match = re.search(r"\n(?:bash|BASH)\n(.*?)(?:\n\n```|\Z)", text, re.DOTALL)
        if plain_bash_match:
            command = plain_bash_match.group(1).strip()
            thought = text[: plain_bash_match.start()].strip()
            if command:
                return finalize(thought, command)

        bash_matches = re.findall(r"```(?:bash|sh)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if bash_matches:
            command = str(bash_matches[0] or "").strip()
            thought = re.split(r"```(?:bash|sh)?\s*\n", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            if command:
                return finalize(thought, command)

        return text

    @classmethod
    def _normalize_thought_action_response(
        cls,
        *,
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not cls._request_expects_thought_action(request_payload):
            return response_payload
        choices = response_payload.get("choices")
        if not isinstance(choices, list):
            return response_payload
        changed = False
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                raw_text = ""
                content = message.get("content")
                if isinstance(content, list):
                    raw_text = "".join(
                        str(item.get("text") or item.get("content") or "")
                        if isinstance(item, dict)
                        else str(item)
                        for item in content
                    )
                else:
                    raw_text = str(content or "")
                normalized = cls._normalize_thought_action_text(raw_text)
                if normalized and normalized != raw_text:
                    message["content"] = normalized
                    changed = True
                continue
            raw_text = str(choice.get("text") or "")
            normalized = cls._normalize_thought_action_text(raw_text)
            if normalized and normalized != raw_text:
                choice["text"] = normalized
                changed = True
        if changed:
            LOG.info("[Gateway] Applied thought-action normalization to gateway response")
        return response_payload

    @staticmethod
    def _extract_prompt_class_alias(request_payload: Dict[str, Any]) -> str:
        for source in (
            request_payload,
            request_payload.get("metadata") if isinstance(request_payload.get("metadata"), dict) else None,
            request_payload.get("extra_body") if isinstance(request_payload.get("extra_body"), dict) else None,
        ):
            if isinstance(source, dict):
                value = str(source.get("prompt_class") or source.get("task_class") or "").strip()
                if value:
                    return value
        return ""

    def _extract_canonical_task_type(
        self,
        *,
        request_payload: Dict[str, Any],
        routing_context: Optional[Dict[str, Any]],
    ) -> str:
        for source in (
            routing_context if isinstance(routing_context, dict) else None,
            request_payload,
            request_payload.get("metadata") if isinstance(request_payload.get("metadata"), dict) else None,
            request_payload.get("extra_body") if isinstance(request_payload.get("extra_body"), dict) else None,
        ):
            if isinstance(source, dict):
                value = normalize_task_type(source.get("task_type"))
                if value:
                    return value
        fallback = self._extract_prompt_class_alias(request_payload)
        return resolve_task_type(fallback)

    def _should_auto_publish_handoff(
        self,
        *,
        request_payload: Dict[str, Any],
        request_kind: str,
        trace_id: str,
        routing_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, Dict[str, Any]]:
        model_name = str(request_payload.get("model") or "").strip()
        max_tokens = max(
            1,
            _int_from_any(
                request_payload.get("max_tokens", request_payload.get("max_new_tokens")),
                1,
            ),
        )
        task_type = self._extract_canonical_task_type(
            request_payload=request_payload,
            routing_context=routing_context,
        )
        prompt_class_alias = self._extract_prompt_class_alias(request_payload)
        allowed_models = _csv_set_from_env("CGC_DOPD_AUTO_PUBLISH_MODELS")
        allowed_task_types = normalize_task_type_iter(
            _csv_set_from_env_multi(
            "CGC_DOPD_AUTO_PUBLISH_TASK_TYPES",
            "CGC_DOPD_AUTO_PUBLISH_PROMPT_CLASSES",
            )
        )
        max_tokens_le = _int_from_any(os.environ.get("CGC_DOPD_AUTO_PUBLISH_MAX_TOKENS_LE"), 0)
        cooldown_s = max(0.0, _float_from_any(os.environ.get("CGC_DOPD_AUTO_PUBLISH_COOLDOWN_S"), 0.0))
        decision = {
            "trace_id": trace_id,
            "request_kind": request_kind,
            "model_name": model_name,
            "task_type": task_type,
            "prompt_class_alias": prompt_class_alias,
            "classification_key": "task_type",
            "max_tokens": max_tokens,
            "allowed_models": sorted(allowed_models),
            "allowed_task_types": sorted(allowed_task_types),
            "max_tokens_le": max_tokens_le,
            "cooldown_s": cooldown_s,
        }
        if allowed_models and model_name not in allowed_models:
            decision["skip_reason"] = "model_not_allowed"
            return False, decision
        if allowed_task_types and task_type not in allowed_task_types:
            decision["skip_reason"] = "task_type_not_allowed"
            return False, decision
        if max_tokens_le > 0 and max_tokens > max_tokens_le:
            decision["skip_reason"] = "max_tokens_exceeds_policy"
            return False, decision
        cooldown_key = f"{request_kind}|{model_name}|{task_type or '-'}"
        now_ts = time.time()
        if cooldown_s > 0:
            with self._lock:
                last_ts = float(self._auto_publish_last_ts.get(cooldown_key) or 0.0)
                if last_ts > 0 and (now_ts - last_ts) < cooldown_s:
                    decision["skip_reason"] = "cooldown_active"
                    decision["cooldown_key"] = cooldown_key
                    decision["cooldown_remaining_s"] = max(0.0, cooldown_s - (now_ts - last_ts))
                    return False, decision
                self._auto_publish_last_ts[cooldown_key] = now_ts
        decision["cooldown_key"] = cooldown_key
        return True, decision

    async def maybe_publish_prefill_handoff(
        self,
        *,
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
        request_kind: str,
        routing_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        enable_auto_publish_handoff = bool(
            getattr(
                self,
                "enable_auto_publish_handoff",
                str(self.config.pd_mode or "").strip() == "cloud_prefill_edge_decode",
            )
        )
        if not enable_auto_publish_handoff:
            return None
        if not isinstance(request_payload, dict) or not isinstance(response_payload, dict):
            return None
        cloud_text = self._extract_response_text(response_payload)
        if not cloud_text:
            return None
        trace_id = (
            str(request_payload.get("trace_id") or "")
            or str(response_payload.get("id") or "")
            or f"dopd-{int(time.time() * 1_000_000)}"
        )
        allowed, policy = self._should_auto_publish_handoff(
            request_payload=request_payload,
            request_kind=request_kind,
            trace_id=trace_id,
            routing_context=routing_context,
        )
        if not allowed:
            skipped = {
                "success": False,
                "skipped": True,
                "request_kind": request_kind,
                "trace_id": trace_id,
                **policy,
            }
            LOG.info("[Gateway] DOPD auto publish skipped: %s", skipped.get("skip_reason"))
            return skipped
        publish_metadata = {
            "request_kind": request_kind,
            "gateway_auto_publish": "1",
            "response_id": str(response_payload.get("id") or ""),
            "task_type": str(policy.get("task_type") or ""),
            "prompt_class_alias": str(policy.get("prompt_class_alias") or ""),
            "auto_publish_policy_key": str(policy.get("cooldown_key") or ""),
            "task_type_contract_version": TASK_TYPE_CONTRACT_VERSION,
            "task_type_contract_path": str(task_type_contract_ref().get("task_type_contract_path") or ""),
        }
        try:
            from app.servers.cloud_socket_server import publish_dopd_handoff_from_prefill

            result = await asyncio.to_thread(
                publish_dopd_handoff_from_prefill,
                request_payload,
                cloud_text=cloud_text,
                openai_response=response_payload,
                trace_id=trace_id,
                max_new_tokens=max(
                    1,
                    int(policy.get("max_tokens") or 1),
                ),
                metadata=publish_metadata,
            )
        except Exception as exc:
            cooldown_key = str(policy.get("cooldown_key") or "")
            if cooldown_key:
                with self._lock:
                    self._auto_publish_last_ts.pop(cooldown_key, None)
            LOG.warning("[Gateway] DOPD auto publish failed: %s", exc)
            return {
                "success": False,
                "request_kind": request_kind,
                "trace_id": trace_id,
                "error_message": str(exc),
            }
        with self._lock:
            self._resume_sessions[str((result or {}).get("handoff_id") or trace_id)] = (
                result if isinstance(result, dict) else {"success": False, "trace_id": trace_id}
            )
        return result if isinstance(result, dict) else None

    async def accept_resume_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        enable_edge_resume = bool(
            getattr(
                self,
                "enable_edge_resume",
                str(self.config.pd_mode or "").strip() == "cloud_prefill_edge_decode",
            )
        )
        contract_context = payload.get("contract_context") if isinstance(payload.get("contract_context"), dict) else {}
        expected_contract = self.expected_resume_contract()
        errors: List[str] = []

        resume_payload_b64 = str(payload.get("resume_payload_b64") or "").strip()
        if not resume_payload_b64:
            errors.append("missing_resume_payload_b64")
            raw_resume_payload = b""
        else:
            try:
                raw_resume_payload = base64.b64decode(resume_payload_b64.encode("ascii"), validate=True)
            except Exception:
                raw_resume_payload = b""
                errors.append("invalid_resume_payload_b64")

        decoded_resume_payload = decode_dopd_resume_payload_v2(raw_resume_payload) if raw_resume_payload else None
        if raw_resume_payload and not isinstance(decoded_resume_payload, dict):
            errors.append("unsupported_resume_payload_format")

        for key, expected_value in expected_contract.items():
            observed_value = str(contract_context.get(key) or "")
            if not self._contract_field_matches(expected_value, observed_value):
                errors.append(f"contract_mismatch:{key}")

        if isinstance(decoded_resume_payload, dict):
            if not bool(decoded_resume_payload.get("integrity_valid")):
                errors.append("resume_payload_integrity_invalid")
            if not self._contract_field_matches(
                str(payload.get("session_id") or ""),
                str(decoded_resume_payload.get("session_id") or ""),
            ):
                errors.append("resume_payload_session_mismatch")
            if not self._contract_field_matches(
                str(payload.get("handoff_id") or ""),
                str(decoded_resume_payload.get("handoff_id") or ""),
            ):
                errors.append("resume_payload_handoff_mismatch")
            if not self._contract_field_matches(
                str(payload.get("phase_role") or ""),
                str(decoded_resume_payload.get("phase_role") or ""),
            ):
                errors.append("resume_payload_phase_role_mismatch")
            abi_descriptor = (
                decoded_resume_payload.get("abi_descriptor")
                if isinstance(decoded_resume_payload.get("abi_descriptor"), dict)
                else {}
            )
            for abi_key in ("state_kind", "state_codec", "protocol_family"):
                if not self._contract_field_matches(
                    str(expected_contract.get(abi_key) or ""),
                    str(abi_descriptor.get(abi_key) or contract_context.get(abi_key) or ""),
                ):
                    errors.append(f"state_abi_mismatch:{abi_key}")
        else:
            abi_descriptor = {}

        state_summary = None
        local_resume = None
        state_kind = str(
            abi_descriptor.get("state_kind")
            or contract_context.get("state_kind")
            or expected_contract.get("state_kind")
            or ""
        )
        state_codec = str(
            abi_descriptor.get("state_codec")
            or contract_context.get("state_codec")
            or expected_contract.get("state_codec")
            or ""
        )
        state_bytes = b""
        if isinstance(decoded_resume_payload, dict):
            try:
                state_bytes = self._decode_embedded_state_bytes(
                    request_payload=payload,
                    resume_payload_meta=decoded_resume_payload,
                )
            except Exception as exc:
                errors.append(f"invalid_state_bytes_b64:{exc}")
        if state_bytes and state_kind and state_codec:
            try:
                state_summary = inspect_kda_state_bytes(
                    state_kind=state_kind,
                    state_codec=state_codec,
                    state_bytes=state_bytes,
                )
            except Exception as exc:
                errors.append(f"state_abi_probe_failed:{exc}")
        elif enable_edge_resume and isinstance(decoded_resume_payload, dict):
            errors.append("missing_edge_state_bytes")

        accepted = not errors
        response_payload = {
            "success": accepted,
            "ack_status": "validated_and_buffered" if accepted else "contract_validation_failed",
            "worker_id": str(payload.get("worker_id") or ""),
            "session_id": str(payload.get("session_id") or ""),
            "handoff_id": str(payload.get("handoff_id") or ""),
            "contract_context": contract_context,
            "expected_contract": expected_contract,
            "resume_payload_meta": decoded_resume_payload if isinstance(decoded_resume_payload, dict) else None,
            "state_summary": state_summary,
            "validation_errors": errors,
        }
        if accepted:
            if enable_edge_resume and state_bytes:
                trace_id = (
                    str(payload.get("resume_token") or "")
                    or str(payload.get("handoff_id") or "")
                    or str(payload.get("session_id") or "")
                )
                try:
                    local_resume = await self._get_local_infer_runtime().resume_from_kda_state(
                        state_kind=state_kind,
                        state_codec=state_codec,
                        state_bytes=state_bytes,
                        state_meta=self._build_edge_resume_state_meta(
                            request_payload=payload,
                            resume_payload_meta=decoded_resume_payload if isinstance(decoded_resume_payload, dict) else {},
                            state_summary=state_summary if isinstance(state_summary, dict) else None,
                        ),
                        trace_id=trace_id,
                        max_tokens=max(1, _int_from_any(payload.get("max_new_tokens"), 1)),
                    )
                    response_payload["ack_status"] = "validated_and_resumed_edge"
                    response_payload["local_resume"] = local_resume
                except Exception as exc:
                    response_payload["success"] = False
                    response_payload["ack_status"] = "edge_resume_failed"
                    response_payload["validation_errors"] = [*errors, f"edge_resume_failed:{exc}"]
            with self._lock:
                self._resume_sessions[str(payload.get("handoff_id") or "")] = response_payload
        return response_payload


def _prepend_pythonpath(path: str) -> None:
    if not path:
        return
    if path not in sys.path:
        sys.path.insert(0, path)
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in current_pythonpath.split(os.pathsep) if entry]
    if path not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([path, *entries])


def _detect_venv_site_packages() -> str:
    major = sys.version_info.major
    minor = sys.version_info.minor
    return str(REPO_ROOT / ".venv_deepep_ssp" / "lib" / f"python{major}.{minor}" / "site-packages")


def _detect_cluster_topology(ray_module: Any) -> Dict[str, int]:
    resources = ray_module.available_resources()
    nodes = [node for node in ray_module.nodes() if node.get("Alive")]
    total_gpus = int(resources.get("GPU", 0))
    nnodes = max(len(nodes), 1)
    gpus_per_node = max(total_gpus // nnodes, 1) if total_gpus else 1
    return {
        "total_gpus": total_gpus,
        "nnodes": nnodes,
        "gpus_per_node": gpus_per_node,
    }


def _detect_head_ip(ray_module: Any) -> str:
    for node in ray_module.nodes():
        if node.get("Alive"):
            return str(node.get("NodeManagerAddress"))
    return socket.gethostbyname(socket.gethostname())


def _build_runtime_env(_repo_root: Path, model_path: str = "") -> Dict[str, Any]:
    _prepend_pythonpath(str(_repo_root.parent))
    _prepend_pythonpath(str(_repo_root))
    _prepend_pythonpath(_detect_venv_site_packages())
    runtime_python_bin = os.environ.get(
        "CGC_SGLANG_PYTHON_BIN",
        str(_repo_root.parent / ".venv_deepep_ssp" / "bin" / "python"),
    )
    runtime_virtual_env = str(Path(runtime_python_bin).resolve().parents[1])
    vendored_sglang_path = ensure_vendored_sglang_on_path()
    _prepend_pythonpath(vendored_sglang_path)
    _prepend_pythonpath(os.path.expanduser("~/.cache/huggingface/modules"))
    moe_a2a_backend = os.environ.get("CGC_MOE_A2A_BACKEND", "none").strip().lower()
    # #region debug-point B:runtime-env-backend
    import json, urllib.request; _p='.dbg/deepep-realchain.env'; _u,_s='http://127.0.0.1:7777/event','deepep-realchain'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"B","location":"ray_serve_sglang_gateway.py:_build_runtime_env","msg":"[DEBUG] runtime env resolved moe backend","data":{"env_cgc_moe_a2a_backend":str(os.environ.get("CGC_MOE_A2A_BACKEND", "")), "effective_moe_a2a_backend":moe_a2a_backend, "runtime_python_bin":str(runtime_python_bin), "cuda_visible_devices":str(os.environ.get("CUDA_VISIBLE_DEVICES", ""))}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
    # #endregion
    resolved_model_path = str(model_path or "")
    runtime_env = {
        "py_executable": runtime_python_bin,
        "env_vars": {
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "PATH": f"{Path(runtime_python_bin).resolve().parent}:{os.environ.get('PATH', '')}",
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            "VIRTUAL_ENV": runtime_virtual_env,
            "HF_MODULES_CACHE": os.path.expanduser("~/.cache/huggingface/modules"),
            "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0"),
            "CUDA_PATH": os.environ.get("CUDA_PATH", os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0")),
            "CGC_MOE_A2A_BACKEND": moe_a2a_backend,
            "CGC_SGLANG_RAY_PY_MODULE": vendored_sglang_path,
            "CGC_SGLANG_CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "CGC_SGLANG_NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES", ""),
            "CGC_SGLANG_CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER", ""),
            "CGC_RAY_ADDRESS": os.environ.get("CGC_RAY_ADDRESS", ""),
            # Keep Ray actors on logical CUDA ordinals inside each instance slice.
            # When this stays at "1", cloud_sglang/ray/engine.py uses Ray's physical
            # GPU assignment id directly, which breaks instances launched with
            # CUDA_VISIBLE_DEVICES like "4,5,6,7" because torch then sees only 0..3.
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": os.environ.get(
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
                "0",
            ),
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        },
        "py_modules": [vendored_sglang_path],
    }
    fp8_wo_a_env = os.environ.get("SGLANG_OPT_FP8_WO_A_GEMM")
    if fp8_wo_a_env is not None and str(fp8_wo_a_env).strip() != "":
        runtime_env["env_vars"]["SGLANG_OPT_FP8_WO_A_GEMM"] = str(fp8_wo_a_env)
    elif "DeepSeek-V4-Flash-UD-IQ2" in resolved_model_path:
        # The UD-IQ2 weights currently fail in deepseek_v4.py::_setup_fp8_wo_a_scales()
        # with DeepGEMM SF-layout conversion on host1. Keep the fallback model-scoped so
        # other DeepSeek-V4 variants can continue using the FP8 WO_A GEMM fast path.
        runtime_env["env_vars"]["SGLANG_OPT_FP8_WO_A_GEMM"] = "0"
    for key in (
        "CGC_SGLANG_CPU_OFFLOAD_GB",
        "CGC_SGLANG_MODEL_LOAD_MAX_PARALLEL_PER_NODE",
        "CGC_SGLANG_MODEL_LOAD_SLOT_POLL_S",
        "CGC_SGLANG_MODEL_LOAD_SLOT_DIR",
    ):
        value = os.environ.get(key)
        if value:
            runtime_env["env_vars"][key] = value
    if moe_a2a_backend == "deepep":
        runtime_env["env_vars"].update(
            {
                "NVSHMEM_DISABLE_CUDA_VMM": os.environ.get(
                    "NVSHMEM_DISABLE_CUDA_VMM",
                    "1",
                ),
                "SGLANG_DEEPEP_ALLOW_MNNVL": os.environ.get(
                    "SGLANG_DEEPEP_ALLOW_MNNVL",
                    "0",
                ),
                "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": os.environ.get(
                    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS",
                    "8192",
                ),
            }
        )
    runtime_working_dir = os.environ.get("CGC_RAY_RUNTIME_WORKING_DIR", "").strip()
    if runtime_working_dir:
        runtime_env["working_dir"] = runtime_working_dir
    return runtime_env


def _make_gateway_config(host: str, port: int) -> GatewayConfig:
    import ray

    profile_bundle = _load_gateway_profile_bundle()
    _apply_profile_env_defaults(profile_bundle)
    cluster = _detect_cluster_topology(ray)
    runtime_shape = (
        profile_bundle.get("runtime_shape")
        if isinstance(profile_bundle.get("runtime_shape"), dict)
        else {}
    )
    safe_runtime_shape = (
        profile_bundle.get("safe_runtime_shape")
        if isinstance(profile_bundle.get("safe_runtime_shape"), dict)
        else {}
    )
    deepep_parallel_profile = str(os.environ.get("CGC_DEEPEP_PARALLEL_PROFILE", "") or "").strip().lower()
    profile_match = re.fullmatch(r"ep(\d+)_tp(\d+)", deepep_parallel_profile)
    profile_ep = int(profile_match.group(1)) if profile_match else None
    profile_tp = int(profile_match.group(2)) if profile_match else None
    declared_tp_size = int(
        os.environ.get(
            "CGC_SGLANG_TP_SIZE",
            str(
                profile_tp
                or runtime_shape.get("tp_size")
                or safe_runtime_shape.get("tp_size")
                or cluster["total_gpus"]
                or 16
            ),
        )
    )
    ep_size = int(
        os.environ.get(
            "CGC_SGLANG_EP_SIZE",
            str(
                profile_ep
                or runtime_shape.get("ep_size")
                or safe_runtime_shape.get("ep_size")
                or declared_tp_size
            ),
        )
    )
    attn_cp_size = int(
        os.environ.get(
            "CGC_SGLANG_ATTN_CP_SIZE",
            str(runtime_shape.get("attn_cp_size") or safe_runtime_shape.get("attn_cp_size") or 1),
        )
    )
    tp_size = declared_tp_size
    if deepep_parallel_profile == "":
        deepep_parallel_profile = f"ep{ep_size}_tp{tp_size}"
    nnodes = int(
        os.environ.get(
            "CGC_SGLANG_NNODES",
            str(runtime_shape.get("nnodes") or safe_runtime_shape.get("nnodes") or cluster["nnodes"]),
        )
    )
    backend_host = os.environ.get("CGC_SGLANG_BACKEND_HOST", "127.0.0.1")
    backend_port = int(os.environ.get("CGC_SGLANG_BACKEND_PORT", "30000"))
    dist_init_addr = _resolve_dist_init_addr(ray, profile_bundle)
    model_path = select_model_path()
    runtime_env = _build_runtime_env(REPO_ROOT, model_path)
    extra_launch_args = _build_profile_extra_launch_args(profile_bundle)
    extra_launch_args.extend(_build_speculative_launch_args())
    return GatewayConfig(
        model_path=model_path,
        tp_size=tp_size,
        ep_size=ep_size,
        attn_cp_size=attn_cp_size,
        deepep_parallel_profile=deepep_parallel_profile,
        nnodes=nnodes,
        backend_host=backend_host,
        backend_port=backend_port,
        gateway_host=host,
        gateway_port=port,
        dist_init_addr=dist_init_addr,
        mem_fraction_static=float(
            os.environ.get(
                "CGC_SGLANG_MEM_FRACTION_STATIC",
                str(
                    runtime_shape.get("mem_fraction_static")
                    or safe_runtime_shape.get("mem_fraction_static")
                    or 0.88
                ),
            )
        ),
        cpu_offload_gb=int(
            os.environ.get(
                "CGC_SGLANG_CPU_OFFLOAD_GB",
                str(
                    runtime_shape.get("cpu_offload_gb")
                    or safe_runtime_shape.get("cpu_offload_gb")
                    or 0
                ),
            )
        ),
        max_running_requests=int(os.environ.get("CGC_SGLANG_MAX_RUNNING_REQUESTS", "1")),
        chunked_prefill_size=int(os.environ.get("CGC_SGLANG_CHUNKED_PREFILL_SIZE", "512")),
        context_length=int(
            os.environ.get(
                "CGC_SGLANG_CONTEXT_LENGTH",
                str(
                    runtime_shape.get("context_length")
                    or safe_runtime_shape.get("context_length")
                    or 8192
                ),
            )
        ),
        max_total_tokens=int(os.environ.get("CGC_SGLANG_MAX_TOTAL_TOKENS", "512")),
        moe_a2a_backend=os.environ.get("CGC_MOE_A2A_BACKEND", "none").strip().lower(),
        disable_cuda_graph=os.environ.get(
            "CGC_SGLANG_DISABLE_CUDA_GRAPH",
            "1" if os.environ.get("CGC_MOE_A2A_BACKEND", "none").strip().lower() == "none" else "0",
        )
        == "1",
        deepep_mode=os.environ.get("CGC_DEEPEP_MODE", "normal"),
        enable_deepep_waterfill=os.environ.get("CGC_ENABLE_DEEPEP_WATERFILL", "0") == "1",
        ray_namespace=os.environ.get("CGC_RAY_NAMESPACE", "cgc-serve"),
        ray_use_spread=os.environ.get("CGC_RAY_PLACEMENT_STRATEGY", "SPREAD") == "SPREAD",
        gateway_replicas=max(1, int(os.environ.get("CGC_RAY_SERVE_GATEWAY_REPLICAS", "1"))),
        runtime_env=runtime_env,
        extra_launch_args=extra_launch_args,
        profile_settings_path=str(profile_bundle.get("profile_settings_path") or ""),
        execution_profile_binding_key=str(
            profile_bundle.get("execution_profile_binding_key") or ""
        ),
        bootstrap_contract_binding_key=str(
            profile_bundle.get("bootstrap_contract_binding_key") or ""
        ),
        flow_parameter_contract_binding_key=str(
            profile_bundle.get("flow_parameter_contract_binding_key") or ""
        ),
        bootstrap_contract_path=str(profile_bundle.get("bootstrap_contract_path") or ""),
        bootstrap_contract_id=str(profile_bundle.get("bootstrap_contract_id") or ""),
        system_manifest_path=str(profile_bundle.get("system_manifest_path") or ""),
        system_profile_id=str(profile_bundle.get("system_profile_id") or ""),
        model_contract_path=str(profile_bundle.get("model_contract_path") or ""),
        model_contract_id=str(profile_bundle.get("model_contract_id") or ""),
        protocol_family=str(
            (profile_bundle.get("environment_bootstrap_ref") or {}).get("protocol_family")
            if isinstance(profile_bundle.get("environment_bootstrap_ref"), dict)
            else ""
        )
        or str(os.environ.get("CGC_RUNTIME_PROTOCOL_FAMILY", "") or ""),
        state_kind=str(
            (profile_bundle.get("environment_bootstrap_ref") or {}).get("state_kind")
            if isinstance(profile_bundle.get("environment_bootstrap_ref"), dict)
            else ""
        )
        or str(os.environ.get("CGC_RUNTIME_STATE_KIND", "") or "kda_state_v1"),
        state_codec=str(
            (profile_bundle.get("environment_bootstrap_ref") or {}).get("state_codec")
            if isinstance(profile_bundle.get("environment_bootstrap_ref"), dict)
            else ""
        )
        or str(os.environ.get("CGC_RUNTIME_STATE_CODEC", "") or "cq4"),
        pd_mode=str(os.environ.get("CGC_PD_MODE", "") or "cloud_prefill_edge_decode"),
        task_type_contract_ref=(
            profile_bundle.get("task_type_contract_ref")
            if isinstance(profile_bundle.get("task_type_contract_ref"), dict)
            else task_type_contract_ref()
        ),
        task_type_contract_validation=(
            profile_bundle.get("task_type_contract_validation")
            if isinstance(profile_bundle.get("task_type_contract_validation"), dict)
            else {"status": "SKIP", "reason": "profile_bundle_missing"}
        ),
    )


def _extract_stream_choice_text(choice: Dict[str, Any]) -> str:
    if not isinstance(choice, dict):
        return ""
    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = str(item.get("text") or item.get("content") or "")
                else:
                    text = str(item)
                if text:
                    parts.append(text)
            return "".join(parts)
        if content is not None:
            return str(content)
    text = choice.get("text")
    if text is not None:
        return str(text)
    return ""


def _build_streaming_response_payload(
    *,
    request_kind: str,
    response_id: str,
    collected_text: str,
) -> Dict[str, Any]:
    if request_kind == "completions":
        return {
            "id": response_id,
            "object": "text_completion",
            "choices": [{"index": 0, "text": collected_text, "finish_reason": "stop"}],
        }
    return {
        "id": response_id,
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": collected_text},
                "finish_reason": "stop",
            }
        ],
    }


def _proxy_stream(
    method: str,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    *,
    on_complete: Any = None,
    request_kind: str = "",
):
    upstream = requests.request(method, url, json=payload, headers=headers, stream=True, timeout=(10, 1800))

    def iterator():
        decoder = codecs.getincrementaldecoder("utf-8")()
        text_buffer = ""
        collected_text_parts: List[str] = []
        response_id = ""
        saw_done = False

        def _process_text_fragment(fragment: str) -> None:
            nonlocal text_buffer, response_id, saw_done
            if not fragment:
                return
            text_buffer += fragment
            while "\n" in text_buffer:
                raw_line, text_buffer = text_buffer.split("\n", 1)
                line = raw_line.rstrip("\r")
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str:
                    continue
                if data_str == "[DONE]":
                    saw_done = True
                    continue
                try:
                    event = json.loads(data_str)
                except Exception:
                    continue
                if isinstance(event, dict):
                    response_id = str(event.get("id") or response_id)
                    choices = event.get("choices")
                    if isinstance(choices, list):
                        for choice in choices:
                            text = _extract_stream_choice_text(choice if isinstance(choice, dict) else {})
                            if text:
                                collected_text_parts.append(text)

        try:
            for chunk in upstream.iter_content(chunk_size=None):
                if chunk:
                    try:
                        _process_text_fragment(decoder.decode(chunk))
                    except Exception:
                        pass
                    yield chunk
        finally:
            try:
                _process_text_fragment(decoder.decode(b"", final=True))
            except Exception:
                pass
            upstream.close()
            if on_complete and (collected_text_parts or saw_done):
                try:
                    response_payload = _build_streaming_response_payload(
                        request_kind=request_kind,
                        response_id=response_id,
                        collected_text="".join(collected_text_parts),
                    )
                    callback_result = on_complete(response_payload)
                    if inspect.isawaitable(callback_result):
                        asyncio.run(callback_result)
                except Exception as exc:
                    LOG.warning("[Gateway] Streaming DOPD completion callback failed: %s", exc)

    return StreamingResponse(
        iterator(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )


def _build_gateway_deployment(serve: Any):
    app = FastAPI(
        title="CGC Ray Serve SGLang Gateway",
        lifespan=_gateway_lifespan,
    )

    @serve.deployment(
        name="cgc-sglang-openai-gateway",
        ray_actor_options={"num_cpus": 2},
    )
    @serve.ingress(app)
    class CGCRayServeGateway:
        def __init__(self, payload: Dict[str, Any]):
            self.config = GatewayConfig(**payload)
            self.backend_manager = SGLangBackendManager(self.config)
            wait_for_backend_ready = os.environ.get("CGC_WAIT_FOR_BACKEND_READY", "1") == "1"
            self.backend_manager.start(wait_until_ready=wait_for_backend_ready)
            self.runtime = GatewayRuntime(self.config)
            self.backend_base_url = (
                f"http://{self.config.backend_host}:{self.config.backend_port}"
            )

        @app.get("/health")
        async def health(self) -> Dict[str, Any]:
            backend_probe = self.backend_manager.probe_status()
            backend_probe_url = str(backend_probe.get("probe_url") or "")
            backend_ready = bool(backend_probe.get("ready"))
            backend_status = (
                f"ok via {backend_probe_url}" if backend_ready else "down"
            )
            return {
                "status": "ok",
                "gateway": "ray_serve_sglang",
                "build_id": GATEWAY_BUILD_ID,
                "backend": backend_status,
                "backend_url": self.backend_base_url,
                "backend_ready": backend_ready,
                "backend_probe_url": backend_probe_url,
                "backend_probe_failures": backend_probe.get("failures", []),
                "tp_size": self.config.tp_size,
                "ep_size": self.config.ep_size,
                "nnodes": self.config.nnodes,
                "moe_a2a_backend": self.config.moe_a2a_backend,
                "disable_cuda_graph": self.config.disable_cuda_graph,
                "gateway_replicas": self.config.gateway_replicas,
                "profile_settings_path": self.config.profile_settings_path,
                "execution_profile_binding_key": self.config.execution_profile_binding_key,
                "bootstrap_contract_binding_key": self.config.bootstrap_contract_binding_key,
                "flow_parameter_contract_binding_key": self.config.flow_parameter_contract_binding_key,
                "bootstrap_contract_id": self.config.bootstrap_contract_id,
                "system_profile_id": self.config.system_profile_id,
                "model_contract_id": self.config.model_contract_id,
                "protocol_family": self.config.protocol_family,
                "state_kind": self.config.state_kind,
                "state_codec": self.config.state_codec,
                "pd_mode": self.config.pd_mode,
                "task_type_contract_ref": self.config.task_type_contract_ref,
                "task_type_contract_validation": self.config.task_type_contract_validation,
                "dopd_auto_publish_handoff": bool(getattr(self.runtime, "enable_auto_publish_handoff", False)),
                "dopd_auto_publish_models": sorted(_csv_set_from_env("CGC_DOPD_AUTO_PUBLISH_MODELS")),
                "dopd_auto_publish_task_types": sorted(
                    _csv_set_from_env_multi(
                        "CGC_DOPD_AUTO_PUBLISH_TASK_TYPES",
                        "CGC_DOPD_AUTO_PUBLISH_PROMPT_CLASSES",
                    )
                ),
                "dopd_auto_publish_prompt_classes_legacy": sorted(
                    _csv_set_from_env("CGC_DOPD_AUTO_PUBLISH_PROMPT_CLASSES")
                ),
                "dopd_auto_publish_max_tokens_le": _int_from_any(os.environ.get("CGC_DOPD_AUTO_PUBLISH_MAX_TOKENS_LE"), 0),
                "dopd_auto_publish_cooldown_s": _float_from_any(os.environ.get("CGC_DOPD_AUTO_PUBLISH_COOLDOWN_S"), 0.0),
            }

        @app.post("/v1/dopd/resume")
        async def dopd_resume(self, request: Request):
            payload = await request.json()
            result = await self.runtime.accept_resume_request(
                payload if isinstance(payload, dict) else {}
            )
            return JSONResponse(status_code=200 if result.get("success") else 422, content=result)

        @app.get("/v1/models")
        async def list_models(self) -> Response:
            try:
                upstream = requests.get(f"{self.backend_base_url}/v1/models", timeout=30)
            except requests.RequestException as exc:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": f"SGLang backend unavailable: {exc}",
                            "type": "backend_unavailable",
                        }
                    },
                )
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )

        @app.post("/v1/chat/completions")
        async def chat_completions(self, request: Request):
            payload = await request.json()
            edge_matrix = self.runtime.build_edge_matrix(payload, dict(request.headers))
            self.runtime.prepare_request(payload, edge_matrix)
            # #region debug-point A:gateway-chat-request
            import json, urllib.request; _p='.dbg/tp4ep4-swe-timeout.env'; _u,_s='http://127.0.0.1:7777/event','tp4ep4-swe-timeout'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"A","location":"ray_serve_sglang_gateway.py:chat_completions","msg":"[DEBUG] gateway received chat completion request","data":{"backend_base_url":self.backend_base_url,"stream":bool(payload.get("stream")),"model":str(payload.get("model") or ""),"messages_count":len(payload.get("messages") or []) if isinstance(payload, dict) else 0,"max_tokens":payload.get("max_tokens") if isinstance(payload, dict) else None,"task_type":str(request.headers.get(CGC_TASK_TYPE_HEADER) or ""),"payload_task_type":str(payload.get("task_type") or "") if isinstance(payload, dict) else "","payload_metadata_task_type":str((payload.get("metadata") or {}).get("task_type") or "") if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict) else "","payload_has_profile_binding_ref":bool(isinstance(payload, dict) and isinstance(payload.get("metadata"), dict) and isinstance((payload.get("metadata") or {}).get("profile_binding_ref"), dict)),"payload_has_system_profile_ref":bool(isinstance(payload, dict) and isinstance(payload.get("metadata"), dict) and isinstance((payload.get("metadata") or {}).get("system_profile_ref"), dict)),"trace_id":str(request.headers.get("x-request-id") or request.headers.get("x-trace-id") or "")}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
            # #endregion
            headers = {
                "Content-Type": "application/json",
                "X-CGC-Perception-Matrix": json.dumps(edge_matrix),
            }
            if request.headers.get("x-request-id"):
                headers["X-Request-ID"] = str(request.headers.get("x-request-id"))
            if request.headers.get("x-cgc-trace-id"):
                headers["X-CGC-Trace-ID"] = str(request.headers.get("x-cgc-trace-id"))
            if request.headers.get(CGC_TASK_TYPE_HEADER):
                headers[CGC_TASK_TYPE_HEADER] = str(request.headers.get(CGC_TASK_TYPE_HEADER))
            elif isinstance(payload.get("metadata"), dict) and str(payload["metadata"].get("task_type") or "").strip():
                headers[CGC_TASK_TYPE_HEADER] = str(payload["metadata"].get("task_type"))
            if payload.get("stream"):
                return _proxy_stream(
                    "POST",
                    f"{self.backend_base_url}/v1/chat/completions",
                    headers,
                    payload,
                    on_complete=lambda response_payload: self.runtime.maybe_publish_prefill_handoff(
                        request_payload=payload if isinstance(payload, dict) else {},
                        response_payload=response_payload if isinstance(response_payload, dict) else {},
                        request_kind="chat_completions",
                        routing_context=edge_matrix if isinstance(edge_matrix, dict) else {},
                    ),
                    request_kind="chat_completions",
                )
            try:
                # #region debug-point A:gateway-chat-upstream-start
                import json, urllib.request; _p='.dbg/tp4ep4-swe-timeout.env'; _u,_s='http://127.0.0.1:7777/event','tp4ep4-swe-timeout'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"A","location":"ray_serve_sglang_gateway.py:chat_completions","msg":"[DEBUG] gateway upstream chat completion start","data":{"backend_base_url":self.backend_base_url,"timeout_connect_s":10,"timeout_read_s":1800,"forwarded_request_id":str(headers.get("X-Request-ID") or ""),"forwarded_task_type":str(headers.get(CGC_TASK_TYPE_HEADER) or ""),"payload_metadata_task_type":str((payload.get("metadata") or {}).get("task_type") or "") if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict) else "","payload_has_profile_binding_ref":bool(isinstance(payload, dict) and isinstance(payload.get("metadata"), dict) and isinstance((payload.get("metadata") or {}).get("profile_binding_ref"), dict)),"payload_has_system_profile_ref":bool(isinstance(payload, dict) and isinstance(payload.get("metadata"), dict) and isinstance((payload.get("metadata") or {}).get("system_profile_ref"), dict))}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
                # #endregion
                upstream = requests.post(
                    f"{self.backend_base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    stream=True,
                    timeout=(10, 1800),
                )
            except requests.RequestException as exc:
                # #region debug-point A:gateway-chat-upstream-error
                import json, urllib.request; _p='.dbg/tp4ep4-swe-timeout.env'; _u,_s='http://127.0.0.1:7777/event','tp4ep4-swe-timeout'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"A","location":"ray_serve_sglang_gateway.py:chat_completions","msg":"[DEBUG] gateway upstream chat completion failed","data":{"backend_base_url":self.backend_base_url,"error_type":exc.__class__.__name__,"error":str(exc),"stream":bool(payload.get("stream")),"model":str(payload.get("model") or "")}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
                # #endregion
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": f"SGLang backend unavailable: {exc}",
                            "type": "backend_unavailable",
                        }
                    },
                )
            # #region debug-point A:gateway-chat-upstream-headers
            import json, urllib.request; _p='.dbg/tp4ep4-swe-timeout.env'; _u,_s='http://127.0.0.1:7777/event','tp4ep4-swe-timeout'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"A","location":"ray_serve_sglang_gateway.py:chat_completions","msg":"[DEBUG] gateway upstream chat completion headers received","data":{"backend_base_url":self.backend_base_url,"status_code":int(upstream.status_code),"content_type":str(upstream.headers.get('content-type', '')),"content_length":str(upstream.headers.get('content-length', '')),"transfer_encoding":str(upstream.headers.get('transfer-encoding', ''))}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
            # #endregion
            # #region debug-point A:gateway-chat-upstream-body-start
            import json, urllib.request; _p='.dbg/tp4ep4-swe-timeout.env'; _u,_s='http://127.0.0.1:7777/event','tp4ep4-swe-timeout'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"A","location":"ray_serve_sglang_gateway.py:chat_completions","msg":"[DEBUG] gateway upstream chat completion body read start","data":{"backend_base_url":self.backend_base_url,"status_code":int(upstream.status_code)}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
            # #endregion
            response_body = upstream.content
            # #region debug-point A:gateway-chat-upstream-body-done
            import json, urllib.request; _p='.dbg/tp4ep4-swe-timeout.env'; _u,_s='http://127.0.0.1:7777/event','tp4ep4-swe-timeout'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"A","location":"ray_serve_sglang_gateway.py:chat_completions","msg":"[DEBUG] gateway upstream chat completion body read done","data":{"backend_base_url":self.backend_base_url,"response_bytes":len(response_body or b'')}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
            # #endregion
            response_payload = json.loads((response_body or b"{}").decode("utf-8"))
            upstream.close()
            response_payload = self.runtime._normalize_thought_action_response(
                request_payload=payload if isinstance(payload, dict) else {},
                response_payload=response_payload if isinstance(response_payload, dict) else {},
            )
            self.runtime.deepep_comm.combine(response_payload)
            await self.runtime.maybe_publish_prefill_handoff(
                request_payload=payload if isinstance(payload, dict) else {},
                response_payload=response_payload if isinstance(response_payload, dict) else {},
                request_kind="chat_completions",
                routing_context=edge_matrix if isinstance(edge_matrix, dict) else {},
            )
            return JSONResponse(status_code=upstream.status_code, content=response_payload)

        @app.post("/v1/completions")
        async def completions(self, request: Request):
            payload = await request.json()
            edge_matrix = self.runtime.build_edge_matrix(payload, dict(request.headers))
            self.runtime.prepare_request(payload, edge_matrix)
            # #region debug-point A:gateway-completions-request
            import json, urllib.request; _p='.dbg/tp4ep4-swe-timeout.env'; _u,_s='http://127.0.0.1:7777/event','tp4ep4-swe-timeout'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"A","location":"ray_serve_sglang_gateway.py:completions","msg":"[DEBUG] gateway received completion request","data":{"backend_base_url":self.backend_base_url,"stream":bool(payload.get("stream")),"model":str(payload.get("model") or ""),"prompt_length":len(str(payload.get("prompt") or "")) if isinstance(payload, dict) else 0,"max_tokens":payload.get("max_tokens") if isinstance(payload, dict) else None,"task_type":str(request.headers.get(CGC_TASK_TYPE_HEADER) or ""),"trace_id":str(request.headers.get("x-request-id") or request.headers.get("x-trace-id") or "")}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
            # #endregion
            headers = {
                "Content-Type": "application/json",
                "X-CGC-Perception-Matrix": json.dumps(edge_matrix),
            }
            if request.headers.get("x-request-id"):
                headers["X-Request-ID"] = str(request.headers.get("x-request-id"))
            if request.headers.get("x-cgc-trace-id"):
                headers["X-CGC-Trace-ID"] = str(request.headers.get("x-cgc-trace-id"))
            if request.headers.get(CGC_TASK_TYPE_HEADER):
                headers[CGC_TASK_TYPE_HEADER] = str(request.headers.get(CGC_TASK_TYPE_HEADER))
            if payload.get("stream"):
                return _proxy_stream(
                    "POST",
                    f"{self.backend_base_url}/v1/completions",
                    headers,
                    payload,
                    on_complete=lambda response_payload: self.runtime.maybe_publish_prefill_handoff(
                        request_payload=payload if isinstance(payload, dict) else {},
                        response_payload=response_payload if isinstance(response_payload, dict) else {},
                        request_kind="completions",
                        routing_context=edge_matrix if isinstance(edge_matrix, dict) else {},
                    ),
                    request_kind="completions",
                )
            try:
                # #region debug-point A:gateway-completions-upstream-start
                import json, urllib.request; _p='.dbg/tp4ep4-swe-timeout.env'; _u,_s='http://127.0.0.1:7777/event','tp4ep4-swe-timeout'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"A","location":"ray_serve_sglang_gateway.py:completions","msg":"[DEBUG] gateway upstream completion start","data":{"backend_base_url":self.backend_base_url,"timeout_connect_s":10,"timeout_read_s":1800,"forwarded_request_id":str(headers.get('X-Request-ID') or ''),"forwarded_task_type":str(headers.get(CGC_TASK_TYPE_HEADER) or "")}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
                # #endregion
                upstream = requests.post(
                    f"{self.backend_base_url}/v1/completions",
                    json=payload,
                    headers=headers,
                    stream=True,
                    timeout=(10, 1800),
                )
            except requests.RequestException as exc:
                # #region debug-point A:gateway-completions-upstream-error
                import json, urllib.request; _p='.dbg/tp4ep4-swe-timeout.env'; _u,_s='http://127.0.0.1:7777/event','tp4ep4-swe-timeout'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"A","location":"ray_serve_sglang_gateway.py:completions","msg":"[DEBUG] gateway upstream completion failed","data":{"backend_base_url":self.backend_base_url,"error_type":exc.__class__.__name__,"error":str(exc),"stream":bool(payload.get("stream")),"model":str(payload.get("model") or "")}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
                # #endregion
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": f"SGLang backend unavailable: {exc}",
                            "type": "backend_unavailable",
                        }
                    },
                )
            # #region debug-point A:gateway-completions-upstream-headers
            import json, urllib.request; _p='.dbg/tp4ep4-swe-timeout.env'; _u,_s='http://127.0.0.1:7777/event','tp4ep4-swe-timeout'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"A","location":"ray_serve_sglang_gateway.py:completions","msg":"[DEBUG] gateway upstream completion headers received","data":{"backend_base_url":self.backend_base_url,"status_code":int(upstream.status_code),"content_type":str(upstream.headers.get('content-type', '')),"content_length":str(upstream.headers.get('content-length', '')),"transfer_encoding":str(upstream.headers.get('transfer-encoding', ''))}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
            # #endregion
            # #region debug-point A:gateway-completions-upstream-body-start
            import json, urllib.request; _p='.dbg/tp4ep4-swe-timeout.env'; _u,_s='http://127.0.0.1:7777/event','tp4ep4-swe-timeout'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"A","location":"ray_serve_sglang_gateway.py:completions","msg":"[DEBUG] gateway upstream completion body read start","data":{"backend_base_url":self.backend_base_url,"status_code":int(upstream.status_code)}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
            # #endregion
            response_body = upstream.content
            # #region debug-point A:gateway-completions-upstream-body-done
            import json, urllib.request; _p='.dbg/tp4ep4-swe-timeout.env'; _u,_s='http://127.0.0.1:7777/event','tp4ep4-swe-timeout'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); _payload=json.dumps({"sessionId":_s,"runId":"pre-fix","hypothesisId":"A","location":"ray_serve_sglang_gateway.py:completions","msg":"[DEBUG] gateway upstream completion body read done","data":{"backend_base_url":self.backend_base_url,"response_bytes":len(response_body or b'')}}).encode(); exec("try:\n urllib.request.urlopen(urllib.request.Request(_u, data=_payload, headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
            # #endregion
            response_payload = json.loads((response_body or b"{}").decode("utf-8"))
            upstream.close()
            response_payload = self.runtime._normalize_thought_action_response(
                request_payload=payload if isinstance(payload, dict) else {},
                response_payload=response_payload if isinstance(response_payload, dict) else {},
            )
            self.runtime.deepep_comm.combine(response_payload)
            await self.runtime.maybe_publish_prefill_handoff(
                request_payload=payload if isinstance(payload, dict) else {},
                response_payload=response_payload if isinstance(response_payload, dict) else {},
                request_kind="completions",
                routing_context=edge_matrix if isinstance(edge_matrix, dict) else {},
            )
            return JSONResponse(status_code=upstream.status_code, content=response_payload)

    return CGCRayServeGateway


def start_ray_serve_gateway(host: str = "0.0.0.0", port: int = 50052) -> Dict[str, Any]:
    import ray
    from ray import serve

    os.environ["PATH"] = "/usr/local/cuda-13.0/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-13.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    profile_bundle = _load_gateway_profile_bundle()
    validation = (
        profile_bundle.get("task_type_contract_validation")
        if isinstance(profile_bundle.get("task_type_contract_validation"), dict)
        else {}
    )
    if profile_bundle and str(validation.get("status") or "").upper() == "FAIL":
        raise RuntimeError(f"task_type contract validation failed: {validation}")
    applied_profile_env = _apply_profile_env_defaults(profile_bundle)
    if applied_profile_env:
        LOG.info("[Gateway] Applied profile launch env defaults: %s", applied_profile_env)
    runtime_env = _build_runtime_env(REPO_ROOT, select_model_path())
    ray.init(
        address="auto",
        ignore_reinit_error=True,
        runtime_env=runtime_env,
        namespace=os.environ.get("CGC_RAY_NAMESPACE", "cgc-serve"),
    )

    config = _make_gateway_config(host, port)
    head_ip = _detect_head_ip(ray)
    if config.moe_a2a_backend == "deepep":
        try:
            patch_info = patch_sglang_moe(
                tp_size=config.tp_size,
                ep_size=config.ep_size,
                deepep_parallel_profile=config.deepep_parallel_profile,
            )
        except RuntimeError as exc:
            if "DeepEP is required for DeepEP integration." not in str(exc):
                raise
            LOG.warning(
                "[DeepEP] Gateway parent process does not have DeepEP runtime; "
                "deferring DeepEP patch to Ray actor/backend runtime: %s",
                exc,
            )
        else:
            LOG.info("[DeepEP] Startup patch info: %s", patch_info)
    else:
        LOG.info(
            "[Gateway] Starting with native SGLang MoE backend: %s",
            config.moe_a2a_backend,
        )

    CGCRayServeGateway = _build_gateway_deployment(serve)
    gateway_actor_options = {
        "num_cpus": 2,
        "resources": {
            f"node:{head_ip}": 0.01,
        },
        "runtime_env": runtime_env,
    }

    http_options = {"host": host, "port": port}
    serve.start(detached=False, http_options=http_options)
    serve.run(
        CGCRayServeGateway.options(
            num_replicas=config.gateway_replicas,
            ray_actor_options=gateway_actor_options,
        ).bind(config.to_payload()),
        route_prefix="/",
    )
    LOG.info(
        "[Gateway] Ray Serve + SGLang is live at http://%s:%s with backend http://%s:%s (%s replicas on head %s)",
        host,
        port,
        config.backend_host,
        config.backend_port,
        config.gateway_replicas,
        head_ip,
    )
    return {
        "status": "ok",
        "gateway_url": f"http://{host}:{port}",
        "backend_url": f"http://{config.backend_host}:{config.backend_port}",
        "config": config.to_payload(),
    }
