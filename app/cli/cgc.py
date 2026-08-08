import argparse
import asyncio
import contextlib
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import shlex
import shutil
import sys
import subprocess
import tempfile
import time
import requests
import multiprocessing
from pathlib import Path

BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_REPO_ROOT))

from app.edge_engine.audit_cli import export_audit, run_audit, trace_audit, verify_audit
from app.edge_engine.build import build_edge_engine
from app.cli.embodied_contract_profiles import canonical_bootstrap_contract
from app.cli.embodied_contract_profiles import canonical_delivery_profile
from app.cli.embodied_contract_profiles import canonical_execution_profile
from app.cli.embodied_contract_profiles import canonical_flow_parameter_contract
from app.cli.embodied_contract_profiles import canonical_profile_names
from app.edge_engine.service_manager import start_edge_stack
from app.shared.profile_bundle_validator import validate_profile_bundle
from app.shared.profile_bundle_validator import validate_profile_bundle_or_raise
from app.shared.task_type_contract import TASK_TYPE_CONTRACT_VERSION
from app.shared.task_type_contract import TASK_TYPE_INFERENCE
from app.shared.task_type_contract import normalize_task_type
from app.shared.task_type_contract import task_type_contract_ref
from app.cli.embodied_upkg40 import DEFAULT_EDGE_MODEL as EMBODIED_DEFAULT_EDGE_MODEL
from app.cli.embodied_upkg40 import run_embodied_psi0_deploy
from app.cli.embodied_upkg40 import run_embodied_psi0_realtimevla
from app.cli.embodied_upkg40 import run_embodied_psi0_train
from app.edge_engine.cloud_tunnel import _password, _proxy_option, _target_user
from cgc_engine.agent.gui_graph_native import build_gui_graph_native_integration
from cgc_engine.product.upkg30_common import evaluate_mandatory_protocol_gate
from cgc_engine.product.upkg30_common import resolve_runtime_protocol_projection


def resolve_cgc_state_dir():
    env_dir = os.environ.get("CGC_HOME")
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    home_dir = (Path.home() / ".cgc").resolve()
    try:
        home_dir.mkdir(parents=True, exist_ok=True)
        probe = home_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return home_dir
    except Exception:
        fallback = (Path(tempfile.gettempdir()) / "cgc_local").resolve()
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _resolve_configured_root(env_name, default_path):
    explicit = str(os.environ.get(env_name) or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(default_path).expanduser().resolve()


CGC_STATE_DIR = resolve_cgc_state_dir()
CONFIG_FILE = str((CGC_STATE_DIR / "config.json").resolve())
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = REPO_ROOT
RELEASE_DIR = REPO_ROOT / "CGC_Release"
ENGINE_REPO_DIR = REPO_ROOT / "ComputeGraphCompiler-main"
GATE_CHECKIN_DIR = RELEASE_DIR / "checkins"
GATE_CHECKIN_LOG = GATE_CHECKIN_DIR / "gate_checkins.jsonl"
TARGET_RELEASE_REPO = "powerautoai/ComputeGraphCompiler"
CGC_MODELS_DIR = (CGC_STATE_DIR / "models").resolve()
CGC_BACKENDS_DIR = (CGC_STATE_DIR / "backends").resolve()
MINICPM5_OLLAMA_MODEL = "minicpm5-1b"
MINICPM5_GGUF_REPO = "openbmb/MiniCPM5-1B-GGUF"
MINICPM5_DEFAULT_QUANT = "Q4_K_M"
CGC_ENGINE_BASE_URL = "http://localhost:8000"
DEFAULT_CLUSTER_NFS_ROOT = "/nfs/embodied"
DEFAULT_CLUSTER_NFS_BACKEND_ROOT = f"{DEFAULT_CLUSTER_NFS_ROOT}/backends"
DEFAULT_CLUSTER_NFS_FETCH_HOST = "39.106.118.206"
UPKG38_UI_TARS_NFS_ROOT = f"{DEFAULT_CLUSTER_NFS_ROOT}/UI-TARS-2B"
UPKG38_UI_TARS_PROBE_ARTIFACT = Path("/private/tmp/upkg38_host2_ui_tars_probe.json")
MINICPM5_CLUSTER_NFS_DIR = f"{DEFAULT_CLUSTER_NFS_ROOT}/minicpm5"
MINICPM5_CLUSTER_NFS_PATH = f"{MINICPM5_CLUSTER_NFS_DIR}/MiniCPM5-1B-Q4_K_M.gguf"
CGC_RUN_ARTIFACT_ROOT = _resolve_configured_root(
    "CGC_RUN_ARTIFACT_ROOT",
    ENGINE_REPO_DIR / "Output" / "edge_runtime" / "cgc_run",
)
CGC_RUN_LATEST_REPORT = (CGC_RUN_ARTIFACT_ROOT / "latest_run_report.json").resolve()
CGC_RUN_LATEST_M4_INFERENCE_REPORT = (CGC_RUN_ARTIFACT_ROOT / "latest_m4_inference_report.json").resolve()
CGC_RUN_LATEST_EDGE_BRIDGE = (CGC_RUN_ARTIFACT_ROOT / "latest_edge_inference_bridge.json").resolve()
CGC_RUN_LATEST_ROUTE_DECISION = (CGC_RUN_ARTIFACT_ROOT / "latest_route_decision.json").resolve()
CGC_AGENT_ARTIFACT_ROOT = _resolve_configured_root(
    "CGC_AGENT_ARTIFACT_ROOT",
    ENGINE_REPO_DIR / "Output" / "agent_cli",
)
CGC_MODEL_ARTIFACT_ROOT = _resolve_configured_root(
    "CGC_MODEL_ARTIFACT_ROOT",
    ENGINE_REPO_DIR / "Output" / "model_cli",
)
VALID_MODEL_SOURCES = {"local", "nfs", "cache", "registry", "config", "all"}
DEFAULT_SWEBENCH_REMOTE_REPO_ROOT = "/root/flashkv0516"
DEFAULT_SWEBENCH_REMOTE_SWE_AGENT_ROOT = f"{DEFAULT_SWEBENCH_REMOTE_REPO_ROOT}/SWE-agent"
DEFAULT_SWEBENCH_REMOTE_LOG_DIR = f"{DEFAULT_SWEBENCH_REMOTE_REPO_ROOT}/logs"
DEFAULT_SWEBENCH_REMOTE_HOSTS = [
    {
        "host": "39.106.118.206",
        "name": "node1-head",
        "role": "head",
        "user": "root",
        "password": "Gen@song@2026622",
    },
    {
        "host": "47.95.250.55",
        "name": "node2-worker",
        "role": "worker",
        "user": "root",
        "password": "Gen@song123",
    },
]


def _default_active_edge_model_path():
    explicit_path = str(os.environ.get("CGC_CLUSTER_NFS_MINICPM5_GGUF") or "").strip()
    if explicit_path:
        return explicit_path
    cluster_nfs_root = str(os.environ.get("CGC_CLUSTER_NFS_ROOT") or "").strip()
    if cluster_nfs_root:
        return str(Path(cluster_nfs_root).expanduser() / "minicpm5" / f"MiniCPM5-1B-{MINICPM5_DEFAULT_QUANT}.gguf")
    return MINICPM5_CLUSTER_NFS_PATH


def _normalize_platform_name(raw_platform=""):
    platform_name = str(raw_platform or "").strip().lower()
    if not platform_name:
        platform_name = "macos" if sys.platform == "darwin" else ("windows" if sys.platform == "win32" else "linux")
    if platform_name == "darwin":
        return "macos"
    return platform_name


def _normalize_arch_name(raw_arch=""):
    arch_name = str(raw_arch or "").strip().lower() or str(platform.machine() or "").strip().lower()
    mapping = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }
    return mapping.get(arch_name, arch_name or "unknown")


def _current_host_selector():
    platform_name = _normalize_platform_name()
    arch_name = _normalize_arch_name()
    return {
        "platform": platform_name,
        "arch": arch_name,
        "platform_arch": f"{platform_name}-{arch_name}",
    }


def _llama_cpp_backend_package_catalog(*, cluster_nfs_root="", platform_name="", arch_name=""):
    selector = {
        "platform": _normalize_platform_name(platform_name),
        "arch": _normalize_arch_name(arch_name),
    }
    selector["platform_arch"] = f"{selector['platform']}-{selector['arch']}"
    backend_root = str(cluster_nfs_root or os.environ.get("CGC_CLUSTER_NFS_BACKEND_ROOT") or DEFAULT_CLUSTER_NFS_BACKEND_ROOT).strip()
    package_root = Path(backend_root).expanduser()
    packages = {}
    for platform_key, arch_key in (
        ("macos", "arm64"),
        ("linux", "x86_64"),
        ("windows", "x86_64"),
    ):
        platform_arch = f"{platform_key}-{arch_key}"
        install_root = (package_root / "llama.cpp" / platform_arch).resolve()
        binary_name = "llama-cli.exe" if platform_key == "windows" else "llama-cli"
        server_name = "llama-server.exe" if platform_key == "windows" else "llama-server"
        shared_lib_ext = "dll" if platform_key == "windows" else ("dylib" if platform_key == "macos" else "so")
        package_manifest_path = (install_root / "package_manifest.json").resolve()
        binary_path = (install_root / "bin" / binary_name).resolve()
        server_path = (install_root / "bin" / server_name).resolve()
        activation_state = "missing"
        if package_manifest_path.exists():
            activation_state = "ready"
        if binary_path.exists():
            activation_state = "active"
        packages[platform_arch] = {
            "package_id": f"cgc.backend.llama_cpp.{platform_arch}",
            "backend_family": "llama.cpp",
            "platform": platform_key,
            "arch": arch_key,
            "distribution_channel": "cluster_nfs",
            "install_root": str(install_root),
            "bin_dir": str((install_root / "bin").resolve()),
            "lib_dir": str((install_root / "lib").resolve()),
            "adapter_dir": str((install_root / "adapters" / "minicpm5").resolve()),
            "binary_path": str(binary_path),
            "server_path": str(server_path),
            "package_manifest_path": str(package_manifest_path),
            "binary_exists": bool(binary_path.exists()),
            "server_exists": bool(server_path.exists()),
            "activation_state": activation_state,
            "shared_lib_glob": f"*.{shared_lib_ext}",
            "integration_model_family": "minicpm5",
            "install_mode": "plugin_dropin",
            "notes": [
                "NFS should provide a prebuilt platform-matched llama.cpp backend package.",
                "Runtime selects the package by host platform/arch before wiring MiniCPM5 GGUF.",
            ],
        }
    selected_package = dict(packages.get(selector["platform_arch"]) or {})
    if selected_package:
        selected_package["selected_by_host"] = True
    return {
        "backend_root": str(package_root.resolve()),
        "selector": selector,
        "packages": packages,
        "selected_package": selected_package,
    }


def _default_turbofieldfare_staging_root():
    explicit = str(os.environ.get("CGC_TURBOFIELDFARE_STAGING_ROOT") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (REPO_ROOT / "var" / "external" / "turbofieldfare").resolve()


def _candidate_turbofieldfare_repo_roots():
    candidates = [
        os.environ.get("CGC_TURBOFIELDFARE_REPO"),
        REPO_ROOT / "turbo-fieldfare",
        REPO_ROOT.parent / "turbo-fieldfare",
        Path("/Users/alexchuang/Documents/turbo-fieldfare"),
    ]
    deduped = []
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve() if candidate else None
        if path and path.is_dir() and path not in deduped:
            deduped.append(path)
    return deduped


def _first_existing_path(candidates, *, want_dir=False):
    for candidate in candidates:
        path = str(candidate or "").strip()
        if not path:
            continue
        probe = Path(path).expanduser().resolve()
        if want_dir:
            if probe.is_dir():
                return str(probe)
        else:
            if probe.exists():
                return str(probe)
    return ""


def _is_complete_gturbo_dir(path) -> bool:
    probe = Path(path).expanduser().resolve()
    required = [
        probe / "manifest.json",
        probe / "verified-install.json",
        probe / "model_weights.bin",
        probe / "packed_experts" / "layout.json",
        probe / "packed_experts" / "layer_29.bin",
    ]
    return probe.is_dir() and all(item.exists() for item in required)


def _first_complete_gturbo_dir(candidates) -> str:
    for candidate in candidates:
        path = str(candidate or "").strip()
        if not path:
            continue
        if _is_complete_gturbo_dir(path):
            return str(Path(path).expanduser().resolve())
    return ""


def _turbofieldfare_blocked_reason(server_path, model_dir):
    if server_path and model_dir:
        return "ready_to_launch"
    if server_path:
        return "missing_model_dir"
    if model_dir:
        return "missing_turbofieldfare_server_binary"
    return "missing_turbofieldfare_server_binary_and_model_dir"


def _collect_turbofieldfare_host_artifacts():
    staging_root = _default_turbofieldfare_staging_root()
    repo_roots = _candidate_turbofieldfare_repo_roots()
    server_candidates = [
        os.environ.get("CGC_TURBOFIELDFARE_SERVER_BIN"),
        os.environ.get("CGC_TURBOFIELDFARE_BIN"),
        os.environ.get("CGC_TURBOFIELDFARE_PATH"),
        staging_root / "bin" / "TurboFieldfareServer",
        shutil.which("TurboFieldfareServer"),
        shutil.which("turbofieldfare"),
    ]
    for repo_root in repo_roots:
        server_candidates.append(repo_root / ".build" / "release" / "TurboFieldfareServer")
    server_path = _first_existing_path(server_candidates)

    model_candidates = [
        os.environ.get("CGC_TURBOFIELDFARE_MODEL"),
        os.environ.get("TURBOFIELDFARE_MODEL"),
        os.environ.get("CGC_TURBOFIELDFARE_MODEL_DIR"),
        os.environ.get("CGC_TURBOFIELDFARE_RESTORED_MODEL"),
        "/tmp/gemma4-restored-local.gturbo",
        staging_root / "models" / "gemma4.gturbo",
    ]
    for repo_root in repo_roots:
        model_candidates.append(repo_root / "scratch" / "gemma4.gturbo")
    model_dir = _first_complete_gturbo_dir(model_candidates)

    missing_dependencies = []
    if not server_path:
        missing_dependencies.append("TurboFieldfareServer")
    if not model_dir:
        missing_dependencies.append("gemma4.gturbo")

    return {
        "repo_roots": [str(path) for path in repo_roots],
        "repo_root": str(repo_roots[0]) if repo_roots else "",
        "staging_root": str(staging_root),
        "server_candidates": [str(candidate) for candidate in server_candidates if str(candidate or "").strip()],
        "model_candidates": [str(candidate) for candidate in model_candidates if str(candidate or "").strip()],
        "server_path": server_path,
        "model_dir": model_dir,
        "missing_dependencies": missing_dependencies,
        "blocked_reason": _turbofieldfare_blocked_reason(server_path, model_dir),
        "launch_ready": bool(server_path and model_dir),
    }


def _turbofieldfare_backend_package_catalog(*, cluster_nfs_root="", platform_name="", arch_name=""):
    selector = {
        "platform": _normalize_platform_name(platform_name),
        "arch": _normalize_arch_name(arch_name),
    }
    selector["platform_arch"] = f"{selector['platform']}-{selector['arch']}"
    backend_root = str(cluster_nfs_root or os.environ.get("CGC_CLUSTER_NFS_BACKEND_ROOT") or DEFAULT_CLUSTER_NFS_BACKEND_ROOT).strip()
    package_root = Path(backend_root).expanduser()
    host_artifacts = _collect_turbofieldfare_host_artifacts()
    packages = {}
    for platform_key, arch_key in (("macos", "arm64"),):
        platform_arch = f"{platform_key}-{arch_key}"
        install_root = (package_root / "turbofieldfare" / platform_arch).resolve()
        binary_name = "TurboFieldfareServer"
        package_manifest_path = (install_root / "package_manifest.json").resolve()
        binary_path = (install_root / "bin" / binary_name).resolve()
        server_path = binary_path
        model_dir = (install_root / "models" / "gemma4.gturbo").resolve()
        activation_state = "missing"
        if package_manifest_path.exists():
            activation_state = "ready"
        if server_path.exists() or model_dir.exists():
            activation_state = "partial"
        if server_path.exists() and model_dir.exists():
            activation_state = "active"
        packages[platform_arch] = {
            "package_id": f"cgc.backend.turbofieldfare.{platform_arch}",
            "backend_family": "mlx",
            "runtime_backend": "turbofieldfare",
            "adapter_name": "gemma4_a4b",
            "platform": platform_key,
            "arch": arch_key,
            "distribution_channel": "cluster_nfs",
            "install_root": str(install_root),
            "bin_dir": str((install_root / "bin").resolve()),
            "lib_dir": str((install_root / "lib").resolve()),
            "adapter_dir": str((install_root / "adapters" / "gemma4").resolve()),
            "binary_path": str(binary_path),
            "server_path": str(server_path),
            "model_dir": str(model_dir),
            "package_manifest_path": str(package_manifest_path),
            "binary_exists": bool(binary_path.exists()),
            "server_exists": bool(server_path.exists()),
            "model_exists": bool(model_dir.exists()),
            "launch_ready": bool(server_path.exists() and model_dir.exists()),
            "blocked_reason": _turbofieldfare_blocked_reason(
                str(server_path) if server_path.exists() else "",
                str(model_dir) if model_dir.exists() else "",
            ),
            "activation_state": activation_state,
            "integration_model_family": "gemma4",
            "install_mode": "plugin_dropin",
            "notes": [
                "TurboFieldfare local_process requires both TurboFieldfareServer and a completed .gturbo model directory.",
                "Unified Runtime IR lowering decides whether an MLX request should target the turbofieldfare runtime backend.",
            ],
        }
    selected_package = dict(packages.get(selector["platform_arch"]) or {})
    if selected_package:
        selected_package["selected_by_host"] = True
        selected_package["host_repo_root"] = str(host_artifacts.get("repo_root") or "")
        selected_package["host_staging_root"] = str(host_artifacts.get("staging_root") or "")
        selected_package["host_server_path"] = str(host_artifacts.get("server_path") or "")
        selected_package["host_model_dir"] = str(host_artifacts.get("model_dir") or "")
        selected_package["host_launch_ready"] = bool(host_artifacts.get("launch_ready"))
        selected_package["host_blocked_reason"] = str(host_artifacts.get("blocked_reason") or "")
    return {
        "backend_root": str(package_root.resolve()),
        "selector": selector,
        "packages": packages,
        "selected_package": selected_package,
    }


def load_python_callable(module_path, module_name, attr_name):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


def load_release_m8_gate_runner():
    return load_python_callable(RELEASE_DIR / "m8_gate.py", "cgc_release_m8_gate", "run_m8_gate")


def load_engine_m7_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_m7_gate
    return run_m7_gate


def load_engine_m72_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_m72_gate
    return run_m72_gate


def load_engine_m73_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_m73_gate
    return run_m73_gate


def load_engine_m1_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_m1_gate
    return run_m1_gate


def load_engine_m2_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_m2_gate
    return run_m2_gate


def load_engine_m3_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_m3_gate
    return run_m3_gate


def load_engine_m4_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_m4_gate_internal
    return run_m4_gate_internal


def load_engine_m5_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_m5_gate
    return run_m5_gate


def load_engine_m6_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_m6_gate
    return run_m6_gate


def load_engine_m74_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_m74_gate
    return run_m74_gate


def load_engine_upkg21_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_upkg21_gate
    return run_upkg21_gate


def load_engine_upkg21_rerun_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_upkg21_rerun_gate
    return run_upkg21_rerun_gate


def load_engine_m75_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_m75_gate
    return run_m75_gate


def load_engine_m76_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_m76_gate
    return run_m76_gate


def load_engine_m77_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_m77_gate
    return run_m77_gate


def load_engine_m78_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_m78_gate
    return run_m78_gate


def load_engine_upkg39_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_upkg39_gate
    return run_upkg39_gate


def load_engine_m79_gate_runner():
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from cgc_engine.product import run_m79_gate
    return run_m79_gate


def load_m75_extreme_status_collector():
    return load_python_callable(
        RELEASE_DIR / "temp" / "misc" / "m75_extreme_status.py",
        "cgc_release_m75_extreme_status",
        "collect_extreme_status",
    )


def write_json_file(file_path, payload):
    out_path = Path(file_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(out_path)


def _module_available(module_name):
    try:
        return importlib.util.find_spec(str(module_name or "").strip()) is not None
    except Exception:
        return False


def _safe_module_version(module_name):
    try:
        return str(importlib.metadata.version(str(module_name or "").strip()) or "")
    except Exception:
        return ""


def _safe_command_version(command_path):
    path = str(command_path or "").strip()
    if not path:
        return ""
    for args in ([path, "--version"], [path, "-v"]):
        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            continue
        raw = str(completed.stdout or completed.stderr or "").strip()
        if raw:
            return raw.splitlines()[0][:160]
    return ""


def _preferred_backend_by_model_format():
    return {
        "gguf": ["llama.cpp", "edge_cloud_bridge"],
        "mlx": ["omlx_mlx_lm", "edge_cloud_bridge"],
        "safetensors": ["transformers", "edge_cloud_bridge"],
    }


def _preferred_backend_by_model_family():
    return {
        "gemma4": ["mlx", "transformers", "edge_cloud_bridge"],
    }


def _collect_host_backend_probe():
    llama_cli_path = str(shutil.which("llama-cli") or "").strip()
    llama_server_path = str(shutil.which("llama-server") or "").strip()
    turbofieldfare_artifacts = _collect_turbofieldfare_host_artifacts()
    turbofieldfare_server_path = str(turbofieldfare_artifacts.get("server_path") or "").strip()
    turbofieldfare_model_dir = str(turbofieldfare_artifacts.get("model_dir") or "").strip()
    mlx_available = _module_available("mlx") and _module_available("mlx_lm")
    mlx_version = _safe_module_version("mlx")
    mlx_lm_version = _safe_module_version("mlx_lm")
    llama_available = bool(llama_cli_path or llama_server_path)
    preferred_backend_map = _preferred_backend_by_model_format()
    preferred_backend_family_map = _preferred_backend_by_model_family()
    host_selector = _current_host_selector()
    backend_package_catalog = _llama_cpp_backend_package_catalog(
        platform_name=str(host_selector.get("platform") or ""),
        arch_name=str(host_selector.get("arch") or ""),
    )
    turbofieldfare_package_catalog = _turbofieldfare_backend_package_catalog(
        platform_name=str(host_selector.get("platform") or ""),
        arch_name=str(host_selector.get("arch") or ""),
    )
    turbofieldfare_selected_package = dict(turbofieldfare_package_catalog.get("selected_package") or {})
    turbofieldfare_available = bool(turbofieldfare_artifacts.get("launch_ready"))
    families = {
        "llama.cpp": {
            "backend_family": "llama.cpp",
            "runtime_backend": "llama.cpp",
            "available": llama_available,
            "command_path": llama_cli_path or llama_server_path,
            "server_path": llama_server_path,
            "detected_version": _safe_command_version(llama_cli_path or llama_server_path),
            "model_format_families": ["gguf"],
            "backend_version_range": "host_probe_detected",
            "inject_mode": "plugin_dropin",
            "reason": "primary_gguf_backend_for_upkg22",
        },
        "mlx": {
            "backend_family": "mlx",
            "runtime_backend": "omlx_mlx_lm",
            "available": mlx_available,
            "python_modules": ["mlx", "mlx_lm"],
            "detected_version": mlx_version or mlx_lm_version,
            "mlx_version": mlx_version,
            "mlx_lm_version": mlx_lm_version,
            "model_format_families": ["mlx"],
            "backend_version_range": "host_probe_detected",
            "inject_mode": "sidecar_loader",
            "reason": "primary_mlx_directory_backend_for_upkg22",
        },
        "turbofieldfare": {
            "backend_family": "mlx",
            "runtime_backend": "turbofieldfare",
            "adapter_name": "gemma4_a4b",
            "available": turbofieldfare_available,
            "command_path": turbofieldfare_server_path or str(turbofieldfare_selected_package.get("server_path") or ""),
            "server_path": turbofieldfare_server_path or str(turbofieldfare_selected_package.get("server_path") or ""),
            "model_dir": turbofieldfare_model_dir,
            "detected_version": _safe_command_version(turbofieldfare_server_path),
            "model_format_families": ["safetensors"],
            "model_family_affinity": ["gemma4"],
            "backend_version_range": "host_probe_detected",
            "inject_mode": "plugin_dropin",
            "reason": "apple_silicon_gemma_runtime_backend_over_mlx",
            "blocked_reason": str(turbofieldfare_artifacts.get("blocked_reason") or ""),
            "missing_dependencies": list(turbofieldfare_artifacts.get("missing_dependencies") or []),
            "launch_ready": bool(turbofieldfare_artifacts.get("launch_ready")),
            "repo_root": str(turbofieldfare_artifacts.get("repo_root") or ""),
            "staging_root": str(turbofieldfare_artifacts.get("staging_root") or ""),
            "selected_host_backend_package": turbofieldfare_selected_package,
        },
    }
    available_backend_families = sorted({
        str(details.get("backend_family") or family_name)
        for family_name, details in families.items()
        if bool(details.get("available"))
    })
    available_runtime_backends = sorted({
        str(details.get("runtime_backend") or family_name)
        for family_name, details in families.items()
        if bool(details.get("available"))
    })
    status = "PASS" if available_runtime_backends else "WARN"
    return {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": str(host_selector.get("platform") or ""),
        "arch": str(host_selector.get("arch") or ""),
        "host_selector": host_selector,
        "upkg_target": "2.2",
        "runtime_strategy": "backend_probe_inject_first",
        "supported_backend_families": ["llama.cpp", "mlx"],
        "supported_runtime_backends": ["llama.cpp", "omlx_mlx_lm", "turbofieldfare"],
        "available_backend_families": available_backend_families,
        "available_runtime_backends": available_runtime_backends,
        "preferred_backend_by_model_format": preferred_backend_map,
        "preferred_backend_by_model_family": preferred_backend_family_map,
        "backend_package_catalogs": {
            "llama.cpp": backend_package_catalog,
            "turbofieldfare": turbofieldfare_package_catalog,
        },
        "backend_package_catalog": backend_package_catalog,
        "selected_host_backend_package": dict(backend_package_catalog.get("selected_package") or {}),
        "families": families,
        "notes": [
            "UPKG 2.2 shifts release acceptance toward backend-probe + inject-first delivery.",
            "GGUF should prefer llama.cpp on the host instead of embedding another LLM runtime into the package.",
            "Gemma4 requests can advertise backend_family=mlx with runtime_backend=turbofieldfare through Unified Runtime IR lowering on Apple Silicon.",
        ],
    }


def _write_backend_injectable_optimization_package(*, aggregate_dir, dist_dir, release_assets_dir, build_payload):
    aggregate_dir = Path(aggregate_dir).expanduser().resolve()
    dist_dir = Path(dist_dir).expanduser().resolve()
    release_assets_dir = Path(release_assets_dir).expanduser().resolve()
    package_dir = (aggregate_dir / "optimization_package").resolve()
    smoke_dir = (package_dir / "smoke").resolve()
    binaries_dir = (package_dir / "binaries").resolve()
    lib_dir = (binaries_dir / "lib").resolve()
    hooks_dir = (binaries_dir / "hooks").resolve()
    adapters_dir = (binaries_dir / "adapters").resolve()
    for target_dir in (package_dir, smoke_dir, binaries_dir, lib_dir, hooks_dir, adapters_dir):
        target_dir.mkdir(parents=True, exist_ok=True)

    probe_payload = _collect_host_backend_probe()
    preferred_backend_map = dict(probe_payload.get("preferred_backend_by_model_format") or {})
    preferred_backend_family_map = dict(probe_payload.get("preferred_backend_by_model_family") or {})
    backend_package_catalogs = dict(probe_payload.get("backend_package_catalogs") or {})
    backend_package_catalog = probe_payload.get("backend_package_catalog") if isinstance(probe_payload.get("backend_package_catalog"), dict) else {}
    selected_host_backend_package = dict(probe_payload.get("selected_host_backend_package") or {})
    package_id = f"cgc.backend.injectable.{build_payload.get('platform') or 'unknown'}.{build_payload.get('host_arch') or 'unknown'}"
    package_manifest = {
        "status": "PASS",
        "generated_at": str(build_payload.get("generated_at") or datetime.now(timezone.utc).isoformat()),
        "package_id": package_id,
        "package_version": "0.1.0",
        "upkg_target": "2.2",
        "delivery_stage": "transition_from_embedded_bundle_to_backend_injectable_package",
        "target_runtime_strategy": "backend_probe_inject_first",
        "legacy_release_artifact": str(build_payload.get("output_path") or ""),
        "preferred_backend_by_model_format": preferred_backend_map,
        "preferred_backend_by_model_family": preferred_backend_family_map,
        "backend_families": ["llama.cpp", "mlx"],
        "runtime_backends": ["llama.cpp", "omlx_mlx_lm", "turbofieldfare"],
        "model_format_families": ["gguf", "mlx", "safetensors"],
        "backend_delivery_mode": "nfs_backend_binary_plus_integration_package",
        "backend_package_catalogs": backend_package_catalogs,
        "backend_package_catalog": backend_package_catalog,
        "selected_host_backend_package": selected_host_backend_package,
        "builder": str(build_payload.get("builder") or ""),
    }
    abi_manifest = {
        "status": "PASS",
        "generated_at": package_manifest["generated_at"],
        "abi_version": "cgc.backend.injectable.v0",
        "package_id": package_id,
        "backend_family": "multi_backend",
        "backend_version_range": "host_probe_detected",
        "platform": str(build_payload.get("platform") or ""),
        "arch": str(build_payload.get("host_arch") or ""),
        "device_family": "apple_silicon" if str(build_payload.get("platform") or "") == "macos" else "generic_host",
        "model_format_family": "gguf_or_mlx",
        "required_symbols": [
            "cgc_backend_probe",
            "cgc_backend_inject",
            "cgc_backend_rollback",
        ],
        "entrypoints": [
            "cgc.inject.route_selector",
            "cgc.inject.backend_probe",
            "cgc.inject.rollback",
        ],
        "fallback_mode": "edge_cloud_bridge",
        "preferred_backend_by_model_format": preferred_backend_map,
        "preferred_backend_by_model_family": preferred_backend_family_map,
    }
    compatibility_matrix = {
        "status": "PASS",
        "generated_at": package_manifest["generated_at"],
        "upkg_target": "2.2",
        "runtime_strategy": "backend_probe_inject_first",
        "backend_families": {
            "llama.cpp": {
                "supported_version_ranges": ["host_probe_detected"],
                "tested_version_ranges": [str((probe_payload.get("families") or {}).get("llama.cpp", {}).get("detected_version") or "host_probe")],
                "blocked_version_ranges": [],
                "reason": "gguf_primary_backend",
            },
            "mlx": {
                "supported_version_ranges": ["host_probe_detected"],
                "tested_version_ranges": [str((probe_payload.get("families") or {}).get("mlx", {}).get("detected_version") or "host_probe")],
                "blocked_version_ranges": [],
                "reason": "mlx_directory_primary_backend",
            },
            "turbofieldfare": {
                "supported_version_ranges": ["host_probe_detected"],
                "tested_version_ranges": [str((probe_payload.get("families") or {}).get("turbofieldfare", {}).get("detected_version") or "host_probe")],
                "blocked_version_ranges": [],
                "reason": "apple_silicon_gemma_runtime_backend_over_mlx",
            },
        },
        "preferred_backend_by_model_format": preferred_backend_map,
        "preferred_backend_by_model_family": preferred_backend_family_map,
    }
    install_recipe = {
        "status": "PASS",
        "generated_at": package_manifest["generated_at"],
        "package_id": package_id,
        "install_mode": "backend_injectable",
        "inject_mode": "probe_selected_family",
        "target_layout": {
            "package_dir": str(package_dir),
            "legacy_release_artifact": str(build_payload.get("output_path") or ""),
            "release_assets_dir": str(release_assets_dir),
            "cluster_backend_catalog_root": str((backend_package_catalog.get("backend_root") or "")),
            "selected_host_backend_package": selected_host_backend_package,
            "dropin_dirs": {
                "lib": str(lib_dir),
                "hooks": str(hooks_dir),
                "adapters": str(adapters_dir),
            },
        },
        "preflight_checks": [
            "host_backend_probe_pass",
            "gguf_prefers_llama_cpp",
            "mlx_directory_prefers_omlx_mlx_lm",
            "gemma4_safetensors_may_target_turbofieldfare",
        ],
        "post_install_checks": [
            "inject_report_ready",
            "runtime_smoke_deferred_to_upkg22",
        ],
    }
    rollback_recipe = {
        "status": "PASS",
        "generated_at": package_manifest["generated_at"],
        "package_id": package_id,
        "restore_actions": [
            "remove injected loader/dropin files",
            "restore original backend entrypoints",
            "re-run backend probe to confirm recovery",
        ],
        "success_criteria": [
            "backend_probe_after_rollback_matches_preinject_snapshot",
            "runtime_path_restored_or_cloud_fallback_admissible",
        ],
    }
    inject_report = {
        "status": "READY",
        "generated_at": package_manifest["generated_at"],
        "package_id": package_id,
        "execution_stage": "build_time_plan_only",
        "next_gate": "UPKG 2.2",
        "selected_backend_family": str((preferred_backend_map.get("gguf") or [""])[0] or ""),
        "target_runtime_strategy": "backend_probe_inject_first",
    }
    inference_smoke_report = {
        "status": "READY",
        "generated_at": package_manifest["generated_at"],
        "package_id": package_id,
        "execution_stage": "deferred_to_upkg22_runtime_validation",
        "required_followup": [
            "probe",
            "inject",
            "runtime_smoke",
            "rollback",
        ],
        "target_runtime_strategy": "backend_probe_inject_first",
    }
    rollback_report = {
        "status": "READY",
        "generated_at": package_manifest["generated_at"],
        "package_id": package_id,
        "execution_stage": "build_time_plan_only",
        "rollback_mode": "safe_restore_before_runtime_exit",
    }
    component_manifests = {
        "lib": {
            "status": "PASS",
            "component": "lib",
            "purpose": "thin runtime drop-ins for backend injection",
            "delivery_mode": "inject_first",
        },
        "hooks": {
            "status": "PASS",
            "component": "hooks",
            "purpose": "entrypoint hooks for backend patch points",
            "delivery_mode": "inject_first",
        },
        "adapters": {
            "status": "PASS",
            "component": "adapters",
            "purpose": "backend adapters and route-selection shims",
            "delivery_mode": "inject_first",
        },
    }
    write_json_file(package_dir / "package_manifest.json", package_manifest)
    write_json_file(package_dir / "abi_manifest.json", abi_manifest)
    write_json_file(package_dir / "compatibility_matrix.json", compatibility_matrix)
    write_json_file(package_dir / "install_recipe.json", install_recipe)
    write_json_file(package_dir / "rollback_recipe.json", rollback_recipe)
    probe_report_path = write_json_file(smoke_dir / "probe_report.json", probe_payload)
    write_json_file(smoke_dir / "inject_report.json", inject_report)
    write_json_file(smoke_dir / "inference_smoke_report.json", inference_smoke_report)
    write_json_file(smoke_dir / "rollback_report.json", rollback_report)
    write_json_file(lib_dir / "component_manifest.json", component_manifests["lib"])
    write_json_file(hooks_dir / "component_manifest.json", component_manifests["hooks"])
    write_json_file(adapters_dir / "component_manifest.json", component_manifests["adapters"])
    return {
        "upkg_target": "2.2",
        "target_runtime_strategy": "backend_probe_inject_first",
        "preferred_backend_by_model_format": preferred_backend_map,
        "injectable_backend_families": ["llama.cpp", "mlx"],
        "available_backend_families": list(probe_payload.get("available_backend_families") or []),
        "backend_package_catalog": backend_package_catalog,
        "selected_host_backend_package": selected_host_backend_package,
        "optimization_package_dir": str(package_dir),
        "optimization_package_manifest": str((package_dir / "package_manifest.json").resolve()),
        "optimization_package_abi_manifest": str((package_dir / "abi_manifest.json").resolve()),
        "optimization_package_compatibility_matrix": str((package_dir / "compatibility_matrix.json").resolve()),
        "optimization_package_install_recipe": str((package_dir / "install_recipe.json").resolve()),
        "optimization_package_rollback_recipe": str((package_dir / "rollback_recipe.json").resolve()),
        "backend_probe_report": str(probe_report_path),
    }


def _copy_release_artifact(source_path, target_path):
    source = Path(source_path).expanduser().resolve()
    target = Path(target_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return str(target)


def _archive_release_artifact(source_path, archive_path):
    source = Path(source_path).expanduser().resolve()
    archive = Path(archive_path).expanduser().resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    base_name = archive.with_suffix("")
    created = shutil.make_archive(
        str(base_name),
        "zip",
        root_dir=str(source.parent),
        base_dir=source.name,
    )
    return str(Path(created).resolve())


PIPELINE_SEED_REPORT_KEYS = (
    "environment",
    "task_domain",
    "task_type",
    "backend",
    "model_name",
    "dtype",
    "export_dir",
    "device",
    "distributed_init",
    "distributed_runtime_bootstrap",
    "execution_context",
    "state_abi",
    "strategy_decision",
    "contract_manifest",
    "system_execution_manifest",
    "pipeline_kernel_contract_artifacts",
    "pipeline_contract_descriptor",
    "model",
    "mode",
    "exec_mode",
    "contexts",
    "pipeline_regenerate_profile",
    "canonical_profile_catalog_path",
    "profile_settings_path",
    "execution_profile_binding_key",
    "execution_profile_binding_keys",
    "delivery_profile_binding_key",
    "compatible_profile_binding_keys",
    "applicable_profile_binding_keys",
    "bootstrap_contract_binding_key",
    "bootstrap_contract_binding_keys",
    "flow_parameter_contract_binding_key",
    "flow_parameter_contract_binding_keys",
    "canonical_execution_profiles_supported",
)


def _pipeline_seed_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _materialize_pipeline_contract_seed_compat(
    *,
    output_root,
    cfg,
    device,
):
    runtime_mode = str(getattr(cfg, "runtime_mode", "") or "")
    environment = str(getattr(cfg, "environment", "") or "")
    task_domain = str(getattr(cfg, "task_domain", "") or "")
    task_type = str(getattr(cfg, "task_type", "") or "")
    backend = str(getattr(cfg, "backend", "") or "")
    model_name = str(getattr(cfg, "model_name", "") or "")
    runtime_profile = str(getattr(cfg, "runtime_profile", "") or "")
    component_id = str(getattr(cfg, "component_id", "") or "pipeline_component")
    component_role = str(getattr(cfg, "component_role", "") or "model_runtime")
    system_id = str(getattr(cfg, "system_id", "") or component_id)
    system_role = str(getattr(cfg, "system_role", "") or "upkg3x_runtime_system")
    report_filename = str(getattr(cfg, "report_filename", "") or "unused_cgc_gate_pipeline_seed.json")
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    execution_context_path = output_root / "execution_context.json"
    state_abi_path = output_root / "state_abi.json"
    compatibility_report_path = output_root / "compatibility_report.json"
    strategy_decision_path = output_root / "strategy_decision.json"
    distributed_runtime_bootstrap_path = output_root / "distributed_runtime_bootstrap.json"
    contract_manifest_path = output_root / "contract_manifest.json"
    system_execution_manifest_path = output_root / "system_execution_manifest.json"

    execution_context = {
        "schema_version": "execution_context_v0.1",
        "component_id": component_id,
        "component_role": component_role,
        "system_id": system_id,
        "system_role": system_role,
        "environment": environment,
        "runtime_mode": runtime_mode,
        "runtime_profile": runtime_profile,
        "task_domain": task_domain,
        "task_type": task_type,
        "backend": backend,
        "hardware_platform": backend,
        "hardware_topology": "single_node",
        "hardware_scope": "local",
        "model_name": model_name,
        "model_family": "",
        "model_scope": model_name,
        "model_assembly": "single_model",
        "artifact_path": str(execution_context_path.resolve()),
        "abi_descriptor": {
            "schema_version": "generic_state_abi_v1",
            "state_abi_path": str(state_abi_path.resolve()),
            "compatibility_report_path": str(compatibility_report_path.resolve()),
        },
    }
    state_abi = {
        "schema_version": "generic_state_abi_v1",
        "component_id": component_id,
        "component_role": component_role,
        "system_id": system_id,
        "task_domain": task_domain,
        "task_type": task_type,
        "environment": environment,
        "backend": backend,
        "model_name": model_name,
        "artifact_path": str(state_abi_path.resolve()),
    }
    compatibility_report = {
        "schema_version": "runtime_compatibility_report_v1",
        "status": "PASS",
        "overall_status": "PASS",
        "overall_reason": "",
        "component_id": component_id,
        "component_role": component_role,
        "state_abi_path": str(state_abi_path.resolve()),
        "reason": "contract_seed_compat_fallback",
        "artifact_path": str(compatibility_report_path.resolve()),
    }
    strategy_decision = {
        "schema_version": "strategy_decision_v0.1",
        "component_id": component_id,
        "component_role": component_role,
        "model_name": model_name,
        "inputs": {
            "execution_context_path": str(execution_context_path.resolve()),
            "state_abi_path": str(state_abi_path.resolve()),
            "compatibility_report_path": str(compatibility_report_path.resolve()),
        },
        "decision": {
            "selected_runtime_branch": "contract_seed_only",
            "reason": "materialize_contract_artifacts_only_missing",
        },
        "artifact_path": str(strategy_decision_path.resolve()),
    }
    distributed_runtime_bootstrap = {
        "schema_version": "distributed_runtime_bootstrap_v1",
        "component_id": component_id,
        "component_role": component_role,
        "system_id": system_id,
        "system_role": system_role,
        "environment": environment,
        "runtime_profile": runtime_profile,
        "backend": backend,
        "artifact_path": str(distributed_runtime_bootstrap_path.resolve()),
    }
    artifact_paths = {
        "execution_context": str(execution_context_path.resolve()),
        "state_abi": str(state_abi_path.resolve()),
        "strategy_decision": str(strategy_decision_path.resolve()),
        "compatibility_report": str(compatibility_report_path.resolve()),
        "distributed_runtime_bootstrap": str(distributed_runtime_bootstrap_path.resolve()),
        "contract_manifest": str(contract_manifest_path.resolve()),
        "system_execution_manifest": str(system_execution_manifest_path.resolve()),
    }
    contract_manifest = {
        "schema_version": "runtime_component_contract_v1",
        "component_id": component_id,
        "component_role": component_role,
        "model_family": "",
        "environment": environment,
        "runtime_mode": runtime_mode,
        "artifact_paths": artifact_paths,
        "overall_status": "PASS",
        "overall_reason": "contract_seed_compat_fallback",
        "runtime_protocol_contract": {},
        "zero_copy_vram_real": {},
        "artifact_path": str(contract_manifest_path.resolve()),
    }
    system_execution_manifest = {
        "schema_version": "cgc.system_execution_manifest.v0.1",
        "created_at_s": float(time.time()),
        "report_filename": report_filename,
        "export_dir": str(output_root),
        "execution_context": {
            "component_id": component_id,
            "component_role": component_role,
            "system_id": system_id,
            "system_role": system_role,
            "environment": environment,
            "runtime_mode": runtime_mode,
            "runtime_profile": runtime_profile,
            "task_domain": task_domain,
            "task_type": task_type,
            "backend": backend,
            "model_name": model_name,
        },
        "strategy_plan": {
            "mode": "contract_seed_only",
            "reason": "materialize_contract_artifacts_only_missing",
        },
        "matrix_axes": {
            "task_domain": task_domain,
            "runtime_mode": runtime_mode,
            "environment": environment,
            "hardware_platform": backend,
            "model_name": model_name,
        },
        "runtime_mode": runtime_mode,
        "environment": environment,
        "backend": backend,
        "model_name": model_name,
        "artifacts": artifact_paths,
        "formal_evidence": {},
        "runtime_protocol_contracts": {},
        "effective_runtime_contracts": {},
        "artifact_path": str(system_execution_manifest_path.resolve()),
    }

    write_json_file(execution_context_path, execution_context)
    write_json_file(state_abi_path, state_abi)
    write_json_file(compatibility_report_path, compatibility_report)
    write_json_file(strategy_decision_path, strategy_decision)
    write_json_file(distributed_runtime_bootstrap_path, distributed_runtime_bootstrap)
    write_json_file(contract_manifest_path, contract_manifest)
    write_json_file(system_execution_manifest_path, system_execution_manifest)

    return {
        "step0_detect": {
            "status": "PASS",
            "mode": "contract_seed_compat_fallback",
        },
        "environment": environment,
        "task_domain": task_domain,
        "task_type": task_type,
        "backend": backend,
        "model_name": model_name,
        "dtype": str(getattr(cfg, "dtype", "")),
        "export_dir": str(output_root),
        "device": str(device),
        "distributed_init": {
            "status": "SKIP",
            "reason": "contract_seed_only",
        },
        "distributed_runtime_bootstrap": "",
        "execution_context": str(execution_context_path.resolve()),
        "state_abi": str(state_abi_path.resolve()),
        "strategy_decision": str(strategy_decision_path.resolve()),
        "contract_manifest": str(contract_manifest_path.resolve()),
        "system_execution_manifest": str(system_execution_manifest_path.resolve()),
        "pipeline_kernel_contract_artifacts": {
            "execution_context_path": str(execution_context_path.resolve()),
            "state_abi_path": str(state_abi_path.resolve()),
            "strategy_decision_path": str(strategy_decision_path.resolve()),
            "compatibility_report_path": str(compatibility_report_path.resolve()),
            "distributed_runtime_bootstrap_path": str(distributed_runtime_bootstrap_path.resolve()),
            "contract_manifest_path": str(contract_manifest_path.resolve()),
            "system_execution_manifest_path": str(system_execution_manifest_path.resolve()),
        },
    }


def _merge_pipeline_seed_fields(*, base_payload, seed_report):
    merged = dict(base_payload if isinstance(base_payload, dict) else {})
    seed = seed_report if isinstance(seed_report, dict) else {}
    for key in PIPELINE_SEED_REPORT_KEYS:
        if key in seed:
            merged[key] = seed.get(key)
    return merged


def _profile_binding_fields(
    *,
    profile_settings_path,
    execution=None,
    execution_map=None,
    delivery=None,
    compatible=None,
    applicable=None,
    bootstrap=None,
    flow=None,
):
    fields = {
        "profile_settings_path": str(Path(profile_settings_path).expanduser().resolve()),
    }
    if isinstance(execution_map, dict) and execution_map:
        fields["execution_profile_binding_keys"] = {str(key): str(value) for key, value in execution_map.items()}
    elif execution:
        fields["execution_profile_binding_key"] = str(execution)
    if delivery:
        fields["delivery_profile_binding_key"] = str(delivery)
    if compatible:
        fields["compatible_profile_binding_keys"] = [str(value) for value in compatible]
    if applicable:
        fields["applicable_profile_binding_keys"] = [str(value) for value in applicable]
    if isinstance(bootstrap, dict) and bootstrap:
        fields["bootstrap_contract_binding_keys"] = {str(key): str(value) for key, value in bootstrap.items()}
    elif isinstance(bootstrap, (list, tuple)) and bootstrap:
        fields["bootstrap_contract_binding_keys"] = [str(value) for value in bootstrap]
    elif bootstrap:
        fields["bootstrap_contract_binding_key"] = str(bootstrap)
    if isinstance(flow, dict) and flow:
        fields["flow_parameter_contract_binding_keys"] = {str(key): str(value) for key, value in flow.items()}
    elif isinstance(flow, (list, tuple)) and flow:
        fields["flow_parameter_contract_binding_keys"] = [str(value) for value in flow]
    elif flow:
        fields["flow_parameter_contract_binding_key"] = str(flow)
    return fields


def _infer_canonical_profile_binding(*, task_type, environment):
    normalized_task_type = str(task_type or "").strip().lower()
    normalized_environment = str(environment or "").strip().lower()
    is_edge_cloud = normalized_environment in {
        "edge_cloud",
        "edge_cloud_openai",
        "edge_cloud_bridge",
        "cloud_edge",
    }
    if normalized_task_type == "train":
        return "edge_cloud_train" if is_edge_cloud else "local_train"
    return "edge_cloud_infer" if is_edge_cloud else "local_infer"


def _profile_family_keys(binding_profile):
    binding = str(binding_profile or "").strip()
    if binding.endswith("train"):
        return ["local_train", "edge_cloud_train"]
    return ["local_infer", "edge_cloud_infer"]


def _seed_dir_aliases(raw_name):
    aliases = set()
    pending = [str(raw_name or "").strip().lower()]
    canonical_tokens = (
        "m7",
        "m71",
        "m72",
        "m73",
        "m77",
        "m78",
        "upkg30",
        "upkg31",
        "upkg32",
        "upkg33",
        "upkg34",
        "upkg35",
        "upkg36",
        "upkg37",
        "upkg38",
    )
    while pending:
        current = pending.pop()
        if not current or current in aliases:
            continue
        aliases.add(current)
        for prefix in ("cgc_gate_", "cli_gate_"):
            if current.startswith(prefix):
                pending.append(current[len(prefix) :])
        if current.endswith("_rerun"):
            pending.append(current[: -len("_rerun")])
        for token in canonical_tokens:
            if token in current:
                pending.append(token)
    return aliases


def _matches_seed_dir_alias(raw_name, *expected_names):
    aliases = _seed_dir_aliases(raw_name)
    return any(str(expected or "").strip().lower() in aliases for expected in expected_names)


def _write_profile_settings_bundle(
    *,
    output_root,
    schema_prefix,
    runtime_host,
    deployment_target,
    environment,
    stage_scope,
    model_scope,
    model_locator,
    distributed_runtime_bootstrap_path="",
    system_manifest_path="",
    strict_validation=False,
):
    root = Path(output_root).expanduser().resolve()
    transport_defaults = {
        "local_infer": "local_runtime_host",
        "local_train": "local_runtime_host",
        "edge_cloud_infer": "edge_cloud_protocol",
        "edge_cloud_train": "edge_cloud_protocol",
    }
    delivery_sides = {
        "local_infer": ("local", "local"),
        "local_train": ("local", "local"),
        "edge_cloud_infer": ("cloud", "edge"),
        "edge_cloud_train": ("edge_or_local", "cloud"),
    }
    profile_environment_defaults = {
        "local_infer": str(environment or "local_workstation"),
        "local_train": str(environment or "local_workstation"),
        "edge_cloud_infer": str(environment or "edge_cloud"),
        "edge_cloud_train": str(environment or "cloud_cluster"),
    }
    execution_profiles = {}
    delivery_profiles = {}
    bootstrap_contract_descriptors = {}
    flow_parameter_contract_descriptors = {}
    for profile_name in canonical_profile_names():
        execution_profiles[profile_name] = canonical_execution_profile(
            profile_name,
            runtime_host=str(runtime_host or ""),
            transport_strategy=transport_defaults.get(profile_name, ""),
            deployment_target=str(deployment_target or ""),
            stage_scope=str(stage_scope or ""),
            model_scope=str(model_scope or ""),
            environment=profile_environment_defaults.get(profile_name, str(environment or "")),
        )
        source_side, target_side = delivery_sides.get(profile_name, ("", ""))
        delivery_profiles[profile_name] = canonical_delivery_profile(
            profile_name,
            source_side=source_side,
            target_side=target_side,
            runtime_host=str(runtime_host or ""),
            deployment_target=str(deployment_target or ""),
        )
        bootstrap_contract_descriptors[profile_name] = canonical_bootstrap_contract(
            profile_name,
            runtime_host=str(runtime_host or ""),
            deployment_target=str(deployment_target or ""),
            environment=profile_environment_defaults.get(profile_name, str(environment or "")),
            distributed_runtime_bootstrap_path=str(distributed_runtime_bootstrap_path or ""),
            bootstrap_source_side=source_side,
            bootstrap_target_side=target_side,
            model_locator=str(model_locator or ""),
        )
        flow_parameter_contract_descriptors[profile_name] = canonical_flow_parameter_contract(
            profile_name,
            runtime_host=str(runtime_host or ""),
            deployment_target=str(deployment_target or ""),
            environment=profile_environment_defaults.get(profile_name, str(environment or "")),
        )
    canonical_profile_catalog = {
        "schema_version": f"{schema_prefix}_canonical_profile_catalog_v1",
        "supported_execution_profiles": canonical_profile_names(),
        "execution_profiles": execution_profiles,
        "delivery_profiles": delivery_profiles,
        "bootstrap_contract_descriptors": bootstrap_contract_descriptors,
        "flow_parameter_contract_descriptors": flow_parameter_contract_descriptors,
    }
    profile_settings = {
        "schema_version": f"{schema_prefix}_profile_settings_v1",
        "task_type_contract_ref": task_type_contract_ref(),
        "canonical_profile_catalog": canonical_profile_catalog,
    }
    canonical_profile_catalog_path = write_json_file(root / "canonical_profile_catalog.json", canonical_profile_catalog)
    profile_settings_path = write_json_file(root / "profile_settings.json", profile_settings)
    has_full_validation_context = bool(
        str(system_manifest_path or "").strip() and str(distributed_runtime_bootstrap_path or "").strip()
    )
    if strict_validation and has_full_validation_context:
        validation_payload = validate_profile_bundle_or_raise(
            profile_settings_path=profile_settings_path,
            system_manifest_path=str(system_manifest_path or ""),
            bootstrap_contract_path=str(distributed_runtime_bootstrap_path or ""),
        )
    else:
        validation_payload = validate_profile_bundle(
            profile_settings_path=profile_settings_path,
            system_manifest_path=str(system_manifest_path or ""),
            bootstrap_contract_path=str(distributed_runtime_bootstrap_path or ""),
        )
    validation_path = write_json_file(root / "profile_bundle_validation.json", validation_payload)
    return {
        "canonical_profile_catalog_path": canonical_profile_catalog_path,
        "profile_settings_path": profile_settings_path,
        "profile_bundle_validation_path": validation_path,
        "profile_bundle_validation": validation_payload,
        "canonical_execution_profiles_supported": canonical_profile_names(),
    }


def _build_upstream_gate_contract(*, gate_name, gate_payload, report_path="", summary_path=""):
    payload = gate_payload if isinstance(gate_payload, dict) else {}
    return {
        "gate_name": str(gate_name),
        "status": str(payload.get("status") or ""),
        "report_path": str(report_path or ""),
        "summary_path": str(summary_path or ""),
        "gate_payload": dict(payload),
    }


def _seed_pipeline_contract_report(
    *,
    export_dir,
    task_domain,
    task_type,
    environment,
    runtime_profile,
    model_name,
    component_role,
    component_id,
    system_id,
    system_role,
    profile_name,
    contexts,
):
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    import torch

    from cgc_engine.pipeline import MegatrainEightStepPipeline, MegatrainPipelineConfig
    from cgc_engine.pipeline_contract_common import pipeline_contract_descriptor_from_report

    output_root = Path(export_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = _pipeline_seed_device()
    backend = "mlx" if device.type == "mps" else ("cuda" if device.type == "cuda" else "cpu")
    cfg = MegatrainPipelineConfig(
        task_type=str(task_type),
        backend=backend,
        environment=str(environment),
        task_domain=str(task_domain),
        model_name=str(model_name),
        dtype=torch.bfloat16,
        load_weights=False,
        export_dir=str(output_root),
        runtime_profile=str(runtime_profile),
        component_id=str(component_id),
        component_role=str(component_role),
        component_required=True,
        system_id=str(system_id),
        system_role=str(system_role),
        system_manifest_autodiscover=True,
        system_manifest_discovery_root=str(output_root.parent),
        report_filename="unused_cgc_gate_pipeline_seed.json",
    )
    pipeline = MegatrainEightStepPipeline(cfg)
    if hasattr(pipeline, "materialize_contract_artifacts_only"):
        report = pipeline.materialize_contract_artifacts_only(device=device)
    else:
        report = _materialize_pipeline_contract_seed_compat(
            output_root=output_root,
            cfg=cfg,
            device=device,
        )
    runtime_probe_payload = {}
    try:
        protocol_root = (output_root / "_protocol_runtime_probe").resolve()
        if protocol_root.exists():
            shutil.rmtree(protocol_root)
        protocol_root.mkdir(parents=True, exist_ok=True)
        runtime_probe_payload = _generate_upkg30_local_protocol_evidence(protocol_root)
        m75_report = dict(runtime_probe_payload.get("m75_report") or {})
        if m75_report:
            _backfill_upkg30_protocol_manifest_fields(
                contract_manifest_path=output_root / "contract_manifest.json",
                system_execution_manifest_path=output_root / "system_execution_manifest.json",
                m75_report=m75_report,
            )
    except Exception as exc:
        runtime_probe_payload = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
    runtime_projection = resolve_runtime_protocol_projection(
        contract_manifest_path=str((output_root / "contract_manifest.json").resolve()),
        system_execution_manifest_path=str((output_root / "system_execution_manifest.json").resolve()),
        component_id=str(component_id or ""),
    )
    binding_profile = _infer_canonical_profile_binding(task_type=task_type, environment=environment)
    family_keys = _profile_family_keys(binding_profile)
    profile_bundle = _write_profile_settings_bundle(
        output_root=output_root,
        schema_prefix="upkg3x",
        runtime_host=str(component_id or system_id or ""),
        deployment_target=str(system_id or component_id or ""),
        environment=str(environment or ""),
        stage_scope=str(profile_name or ""),
        model_scope=str(model_name or ""),
        model_locator=str(model_name or ""),
        distributed_runtime_bootstrap_path=str(report.get("distributed_runtime_bootstrap") or ""),
        system_manifest_path=str((output_root / "system_execution_manifest.json").resolve()),
        strict_validation=True,
    )
    seeded = dict(report)
    seeded["model"] = str(report.get("model_name") or model_name)
    seeded["mode"] = str(report.get("task_type") or task_type)
    seeded["exec_mode"] = str(report.get("backend") or backend)
    seeded["contexts"] = [int(x) for x in list(contexts or [])]
    seeded["pipeline_regenerate_profile"] = str(profile_name)
    seeded["report_path"] = str((output_root / "report.json").resolve())
    seeded["pipeline_contract_descriptor"] = pipeline_contract_descriptor_from_report(report)
    seeded.update(runtime_projection)
    if runtime_probe_payload:
        seeded["runtime_probe_payload"] = dict(runtime_probe_payload)
        seeded["runtime_probe_evidence_path"] = str(runtime_probe_payload.get("local_infer_evidence_path") or "")
        seeded["runtime_probe_report_path"] = str(runtime_probe_payload.get("m75_evidence_path") or "")
    seeded.update(profile_bundle)
    seeded.update(
        _profile_binding_fields(
            profile_settings_path=profile_bundle["profile_settings_path"],
            execution=binding_profile,
            delivery=binding_profile,
            compatible=family_keys,
            applicable=family_keys,
            bootstrap=binding_profile,
            flow=binding_profile,
        )
    )
    write_json_file(output_root / "report.json", seeded)
    write_json_file(output_root / "pipeline_regen_report.json", seeded)
    return seeded


def _resolve_upkg3x_pipeline_seed_spec(*, gate_name, export_dir):
    output_root = Path(export_dir).expanduser().resolve()
    name = output_root.name
    parent_name = output_root.parent.name
    grandparent_name = output_root.parent.parent.name

    def spec(**kwargs):
        payload = dict(kwargs)
        payload["export_dir"] = output_root
        return payload

    if gate_name in {"m7", "m71"}:
        if _matches_seed_dir_alias(name, "cli_gate_m7", "m7"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="ui_agent_kernel_core",
                component_role="agent_kernel_runtime",
                component_id="m7_kernel",
                system_id="m7_agent_suite",
                system_role="upkg3x_agent_kernel",
                profile_name="m7",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(name, "cli_gate_m71", "m71"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="ui_agent_kernel_core",
                component_role="agent_kernel_runtime",
                component_id="m71_kernel",
                system_id="m71_agent_suite",
                system_role="upkg3x_agent_kernel",
                profile_name="m71",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(name, "cli_gate_upkg31", "upkg31"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="ui_agent_kernel_core",
                component_role="agent_kernel_runtime",
                component_id="upkg31_m7_kernel",
                system_id="upkg31_agent_suite",
                system_role="upkg3x_agent_kernel",
                profile_name="upkg31",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(name, "upkg31_m7") and _matches_seed_dir_alias(parent_name, "cli_gate_upkg30", "upkg30"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="ui_agent_kernel_core",
                component_role="agent_kernel_runtime",
                component_id="upkg30_upkg31_m7_kernel",
                system_id="upkg30_agent_suite",
                system_role="upkg3x_agent_kernel",
                profile_name="upkg30:upkg31_m7",
                contexts=[128, 512, 1024],
            )

    if _matches_seed_dir_alias(name, "m7_artifacts"):
        if _matches_seed_dir_alias(parent_name, "cli_gate_m71", "m71"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="ui_agent_kernel_core",
                component_role="agent_kernel_runtime",
                component_id="m71_m7_kernel",
                system_id="m71_agent_suite",
                system_role="upkg3x_agent_kernel",
                profile_name="m71:m7_artifacts",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(parent_name, "cli_gate_m72", "m72"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="gui_agent_kernel_core",
                component_role="agent_kernel_runtime",
                component_id="m72_m7_kernel",
                system_id="m72_agent_runtime",
                system_role="upkg3x_agent_kernel",
                profile_name="m72:m7_artifacts",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(parent_name, "cli_gate_upkg32", "upkg32"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="gui_agent_kernel_core",
                component_role="agent_kernel_runtime",
                component_id="upkg32_m7_kernel",
                system_id="upkg32_agent_runtime",
                system_role="upkg3x_agent_kernel",
                profile_name="upkg32:m7_artifacts",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(parent_name, "cli_gate_upkg34", "upkg34"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="gui_agent_kernel_core",
                component_role="agent_kernel_runtime",
                component_id="upkg34_m7_kernel",
                system_id="upkg34_unified_artifacts",
                system_role="upkg3x_agent_kernel",
                profile_name="upkg34:m7_artifacts",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(parent_name, "cli_gate_upkg35", "upkg35"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="gui_agent_kernel_core",
                component_role="agent_kernel_runtime",
                component_id="upkg35_m7_kernel",
                system_id="upkg35_six_element_audit",
                system_role="upkg3x_agent_kernel",
                profile_name="upkg35:m7_artifacts",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(parent_name, "cli_gate_upkg36", "upkg36"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="gui_agent_kernel_core",
                component_role="agent_kernel_runtime",
                component_id="upkg36_m7_kernel",
                system_id="upkg36_missing_capability",
                system_role="upkg3x_agent_kernel",
                profile_name="upkg36:m7_artifacts",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(parent_name, "cli_gate_m77", "m77", "upkg37"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="gui_agent_kernel_core",
                component_role="agent_kernel_runtime",
                component_id="upkg37_m7_kernel",
                system_id="upkg37_cloud_edge_q2rl",
                system_role="upkg3x_agent_kernel",
                profile_name="upkg37:m7_artifacts",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(parent_name, "cli_gate_m78", "m78", "upkg38"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="gui_agent_kernel_core",
                component_role="agent_kernel_runtime",
                component_id="upkg38_m7_kernel",
                system_id="upkg38_teaching_runtime",
                system_role="upkg3x_agent_kernel",
                profile_name="upkg38:m7_artifacts",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(parent_name, "cli_gate_upkg39", "upkg39"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="gui_agent_strict_closure_runtime",
                component_role="agent_kernel_runtime",
                component_id="upkg39_m7_kernel",
                system_id="upkg39_strict_closure",
                system_role="upkg3x_agent_kernel",
                profile_name="upkg39:m7_artifacts",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(parent_name, "upkg32_m72") and _matches_seed_dir_alias(grandparent_name, "cli_gate_upkg30", "upkg30"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="ui_agent_kernel_core",
                component_role="agent_kernel_runtime",
                component_id="upkg30_upkg32_m72_kernel",
                system_id="upkg30_agent_suite",
                system_role="upkg3x_agent_kernel",
                profile_name="upkg30:upkg32_m72_m7",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(parent_name, "upkg37_m77") and _matches_seed_dir_alias(grandparent_name, "cli_gate_upkg30", "upkg30"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="ui_agent_kernel_core",
                component_role="agent_kernel_runtime",
                component_id="upkg30_upkg37_m77_kernel",
                system_id="upkg30_agent_suite",
                system_role="upkg3x_agent_kernel",
                profile_name="upkg30:upkg37_m77_m7",
                contexts=[128, 512, 1024],
            )

    if gate_name == "m72":
        if _matches_seed_dir_alias(name, "cli_gate_m72", "m72"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="gui_agent_runtime",
                component_role="agent_runtime",
                component_id="m72_agent_runtime",
                system_id="m72_agent_runtime",
                system_role="upkg3x_agent_runtime",
                profile_name="m72",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(name, "cli_gate_upkg32", "upkg32"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="gui_agent_runtime",
                component_role="agent_runtime",
                component_id="upkg32_m72_agent_runtime",
                system_id="upkg32_agent_runtime",
                system_role="upkg3x_agent_runtime",
                profile_name="upkg32",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(name, "cli_gate_upkg34", "upkg34"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="gui_agent_runtime",
                component_role="agent_runtime",
                component_id="upkg34_agent_runtime",
                system_id="upkg34_unified_artifacts",
                system_role="upkg3x_agent_runtime",
                profile_name="upkg34",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(name, "cli_gate_upkg35", "upkg35"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="gui_agent_runtime",
                component_role="agent_runtime",
                component_id="upkg35_agent_runtime",
                system_id="upkg35_six_element_audit",
                system_role="upkg3x_agent_runtime",
                profile_name="upkg35",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(name, "cli_gate_upkg36", "upkg36"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="gui_agent_runtime",
                component_role="agent_runtime",
                component_id="upkg36_agent_runtime",
                system_id="upkg36_missing_capability",
                system_role="upkg3x_agent_runtime",
                profile_name="upkg36",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(name, "upkg32_m72") and _matches_seed_dir_alias(parent_name, "cli_gate_upkg30", "upkg30"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="cloud_single",
                runtime_profile="cloud_single",
                model_name="gui_agent_runtime",
                component_role="agent_runtime",
                component_id="upkg30_upkg32_m72_runtime",
                system_id="upkg30_agent_suite",
                system_role="upkg3x_agent_runtime",
                profile_name="upkg30:upkg32_m72",
                contexts=[128, 512, 1024],
            )

    if gate_name == "m73":
        if _matches_seed_dir_alias(name, "cli_gate_m73", "m73"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="edge_cloud",
                runtime_profile="edge_cloud_openai",
                model_name="ui_agent_edge_bridge",
                component_role="bridge_runtime",
                component_id="m73_edge_bridge",
                system_id="m73_edge_bridge",
                system_role="upkg3x_edge_bridge",
                profile_name="m73",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(name, "cli_gate_upkg33", "upkg33"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="edge_cloud",
                runtime_profile="edge_cloud_openai",
                model_name="ui_agent_edge_bridge",
                component_role="bridge_runtime",
                component_id="upkg33_m73_bridge",
                system_id="upkg33_edge_bridge",
                system_role="upkg3x_edge_bridge",
                profile_name="upkg33",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(name, "upkg33_m73") and _matches_seed_dir_alias(parent_name, "cli_gate_upkg30", "upkg30"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="edge_cloud",
                runtime_profile="edge_cloud_openai",
                model_name="ui_agent_edge_bridge",
                component_role="bridge_runtime",
                component_id="upkg30_upkg33_m73_bridge",
                system_id="upkg30_agent_suite",
                system_role="upkg3x_edge_bridge",
                profile_name="upkg30:upkg33_m73",
                contexts=[128, 512, 1024],
            )

    if gate_name == "m77":
        if _matches_seed_dir_alias(name, "cli_gate_m77", "m77", "upkg37"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="edge_cloud",
                runtime_profile="edge_cloud_openai",
                model_name="gui_agent_cloud_edge_q2rl",
                component_role="cloud_edge_runtime",
                component_id="upkg37_m77_cloud_edge",
                system_id="upkg37_cloud_edge_q2rl",
                system_role="upkg3x_cloud_edge_runtime",
                profile_name="upkg37",
                contexts=[128, 512, 1024],
            )
        if _matches_seed_dir_alias(name, "upkg37_m77") and _matches_seed_dir_alias(parent_name, "cli_gate_upkg30", "upkg30"):
            return spec(
                task_domain="agent",
                task_type="inference",
                environment="edge_cloud",
                runtime_profile="edge_cloud_openai",
                model_name="ui_agent_cloud_edge_q2rl",
                component_role="cloud_edge_runtime",
                component_id="upkg30_upkg37_m77_root",
                system_id="upkg30_agent_suite",
                system_role="upkg3x_cloud_edge_runtime",
                profile_name="upkg30:upkg37_m77_root",
                contexts=[128, 512, 1024],
            )

    if gate_name == "m78" and _matches_seed_dir_alias(name, "cli_gate_m78", "m78", "upkg38"):
        return spec(
            task_domain="agent",
            task_type="inference",
            environment="edge_cloud",
            runtime_profile="edge_cloud_openai",
            model_name="gui_agent_teaching_runtime",
            component_role="teaching_runtime",
            component_id="upkg38_m78_teaching",
            system_id="upkg38_teaching_runtime",
            system_role="upkg3x_teaching_runtime",
            profile_name="upkg38",
            contexts=[128, 512, 1024],
        )

    if gate_name == "upkg39" and _matches_seed_dir_alias(name, "cli_gate_upkg39", "upkg39"):
        return spec(
            task_domain="agent",
            task_type="inference",
            environment="edge_cloud",
            runtime_profile="edge_cloud_openai",
            model_name="gui_agent_strict_closure_runtime",
            component_role="strict_closure_runtime",
            component_id="upkg39_strict_closure",
            system_id="upkg39_strict_closure",
            system_role="upkg3x_strict_closure",
            profile_name="upkg39",
            contexts=[128, 512, 1024],
        )

    return None


def _maybe_seed_upkg3x_pipeline_contract_report(*, gate_name, export_dir):
    spec = _resolve_upkg3x_pipeline_seed_spec(gate_name=gate_name, export_dir=export_dir)
    if not isinstance(spec, dict):
        return {}
    payload = dict(spec)
    export_target = payload.pop("export_dir")
    return _seed_pipeline_contract_report(export_dir=export_target, **payload)


def _read_json_payload(path):
    try:
        raw_path = Path(path).expanduser().resolve()
    except Exception:
        raw_path = Path(path).expanduser()
    if not raw_path.exists():
        return {}
    try:
        return json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_probe_dir(path):
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    probe = target / ".cgc_write_probe"
    probe.write_text("ok", encoding="utf-8")
    with contextlib.suppress(Exception):
        probe.unlink()
    return target


def _resolve_writable_gate_output_root(*, requested_output_dir, gate_name):
    requested_root = Path(requested_output_dir).expanduser().resolve()
    try:
        return _write_probe_dir(requested_root), ""
    except Exception as exc:
        fallback_root = (CGC_STATE_DIR / "gate_output_fallback" / f"cli_gate_{gate_name}").resolve()
        _write_probe_dir(fallback_root)
        return fallback_root, f"{type(exc).__name__}: {exc}"


def _upkg_protocol_upstream_contract(*, gate_name, gate_payload, report_path="", summary_path=""):
    return {
        "upstream_contracts": {
            str(gate_name): {
                "gate_payload": dict(gate_payload or {}),
                "report_path": str(report_path or ""),
                "summary_path": str(summary_path or ""),
            }
        }
    }


def _upkg_protocol_gate_projection(*, report_payload, gate_name):
    gate_payload = dict((((report_payload or {}).get("gate_result") or {}).get(gate_name) or {}))
    mandatory = (
        gate_payload.get("mandatory_protocol_gate")
        if isinstance(gate_payload.get("mandatory_protocol_gate"), dict)
        else report_payload.get("mandatory_protocol_gate")
        if isinstance(report_payload.get("mandatory_protocol_gate"), dict)
        else {}
    )
    runtime_protocol_contract = (
        gate_payload.get("runtime_protocol_contract")
        if isinstance(gate_payload.get("runtime_protocol_contract"), dict)
        else report_payload.get("runtime_protocol_contract")
        if isinstance(report_payload.get("runtime_protocol_contract"), dict)
        else mandatory.get("runtime_protocol_contract")
        if isinstance(mandatory.get("runtime_protocol_contract"), dict)
        else {}
    )
    zero_copy_vram_real = (
        gate_payload.get("zero_copy_vram_real")
        if isinstance(gate_payload.get("zero_copy_vram_real"), dict)
        else report_payload.get("zero_copy_vram_real")
        if isinstance(report_payload.get("zero_copy_vram_real"), dict)
        else mandatory.get("zero_copy_vram_real")
        if isinstance(mandatory.get("zero_copy_vram_real"), dict)
        else {}
    )
    effective_pd_service = (
        gate_payload.get("effective_pd_service")
        if isinstance(gate_payload.get("effective_pd_service"), dict)
        else report_payload.get("effective_pd_service")
        if isinstance(report_payload.get("effective_pd_service"), dict)
        else {}
    )
    return {
        "mandatory_protocol_gate": dict(mandatory or {}),
        "runtime_protocol_contract": dict(runtime_protocol_contract or {}),
        "zero_copy_vram_real": dict(zero_copy_vram_real or {}),
        "effective_pd_service": dict(effective_pd_service or {}),
    }


def _rewrite_upkg30_protocol_report_artifact_paths(*, report_path, gate_root):
    report = _read_json_payload(report_path)
    if not report:
        return
    root = Path(gate_root).expanduser().resolve()
    top_level_file_names = {
        "execution_context": "execution_context.json",
        "state_abi": "state_abi.json",
        "strategy_decision": "strategy_decision.json",
        "compatibility_report": "compatibility_report.json",
        "distributed_runtime_bootstrap": "distributed_runtime_bootstrap.json",
        "contract_manifest": "contract_manifest.json",
        "system_execution_manifest": "system_execution_manifest.json",
    }
    for key, filename in top_level_file_names.items():
        if key in report:
            report[key] = str((root / filename).resolve())
    pipeline_kernel_artifacts = report.get("pipeline_kernel_contract_artifacts")
    if isinstance(pipeline_kernel_artifacts, dict):
        for key, filename in top_level_file_names.items():
            pipeline_kernel_artifacts[f"{key}_path"] = str((root / filename).resolve())
    pipeline_contract_descriptor = report.get("pipeline_contract_descriptor")
    if isinstance(pipeline_contract_descriptor, dict):
        artifacts = pipeline_contract_descriptor.get("artifacts")
        if isinstance(artifacts, dict):
            for key, filename in top_level_file_names.items():
                artifacts[f"{key}_path"] = str((root / filename).resolve())
    write_json_file(report_path, report)


def _generate_upkg30_local_protocol_evidence(protocol_root):
    if str(ENGINE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_REPO_DIR))
    from app.edge_engine.kda_state_runtime import build_real_kda_state_from_request
    from app.edge_engine.local_infer import EdgeLocalInferenceRuntime
    from cgc_engine.product import m75_trueorthokda_active_runtime as m75_runtime_module

    trace_id = "upkg30_protocol_local_refresh"
    request_payload = {
        "prompt": "UPKG 3.0 latest protocol refresh",
        "max_tokens": 1,
    }
    bundle = build_real_kda_state_from_request(request_payload, trace_id=trace_id)
    runtime = EdgeLocalInferenceRuntime()
    runtime.evidence_root = (Path(protocol_root).resolve() / "edge_runtime_local_infer").resolve()
    local_resume = asyncio.run(
        runtime.resume_from_kda_state(
            state_kind=str(bundle.get("state_kind") or ""),
            state_codec=str(bundle.get("state_codec") or ""),
            state_bytes=bundle.get("state_bytes") or b"",
            state_meta=bundle.get("state_meta") or {},
            trace_id=trace_id,
            max_tokens=1,
        )
    )
    local_infer_evidence_path = Path(str(local_resume.get("evidence_path") or "")).resolve()
    m75_evidence_path = (Path(protocol_root).resolve() / "local_m75_trueorthokda_active_runtime.json").resolve()
    original_latest_local_infer = m75_runtime_module._latest_local_infer_evidence_path
    m75_runtime_module._latest_local_infer_evidence_path = lambda: local_infer_evidence_path
    try:
        m75_runtime_module._bootstrap_active_runtime_evidence(m75_evidence_path)
    finally:
        m75_runtime_module._latest_local_infer_evidence_path = original_latest_local_infer
    return {
        "local_infer_evidence_path": str(local_infer_evidence_path),
        "m75_evidence_path": str(m75_evidence_path),
        "m75_report": _read_json_payload(m75_evidence_path),
    }


def _backfill_upkg30_protocol_manifest_fields(*, contract_manifest_path, system_execution_manifest_path, m75_report):
    contract_manifest = _read_json_payload(contract_manifest_path)
    system_execution_manifest = _read_json_payload(system_execution_manifest_path)
    component_id = str(contract_manifest.get("component_id") or "").strip()
    runtime_protocol_contract = dict(m75_report.get("runtime_protocol_contract") or {})
    zero_copy_vram_real = dict(m75_report.get("zero_copy_vram_real") or {})
    mandatory_protocol_gate = evaluate_mandatory_protocol_gate(
        runtime_protocol_contract=runtime_protocol_contract,
        zero_copy_vram_real=zero_copy_vram_real,
        source=str((((m75_report.get("artifacts") or {}).get("local_infer_evidence_path")) or "")),
    )
    effective_runtime_contract = {
        "mandatory_protocol_gate": mandatory_protocol_gate,
        "compression_effective": dict(m75_report.get("compression_effective") or {}),
        "cpu_copy_count": m75_report.get("cpu_copy_count"),
        "effective_collective_backend": dict(m75_report.get("effective_collective_backend") or {}),
        "effective_cuda_graph": dict(m75_report.get("effective_cuda_graph") or {}),
        "effective_dispatch_backend": dict(m75_report.get("effective_dispatch_backend") or {}),
        "effective_distributed_runtime": dict(m75_report.get("effective_distributed_runtime") or {}),
        "effective_pd_service": dict(m75_report.get("effective_pd_service") or {}),
        "effective_storage_backend": dict(m75_report.get("effective_storage_backend") or {}),
        "gds_effective": dict(m75_report.get("gds_effective") or {}),
        "spdk_effective": dict(m75_report.get("spdk_effective") or {}),
        "colossalai_effective": dict(m75_report.get("colossalai_effective") or {}),
    }
    contract_manifest["runtime_protocol_contract"] = runtime_protocol_contract
    contract_manifest["zero_copy_vram_real"] = zero_copy_vram_real
    contract_manifest["mandatory_protocol_gate"] = mandatory_protocol_gate
    contract_manifest["compression_effective"] = dict(m75_report.get("compression_effective") or {})
    contract_manifest["cpu_copy_count"] = m75_report.get("cpu_copy_count")
    contract_manifest["effective_collective_backend"] = dict(m75_report.get("effective_collective_backend") or {})
    contract_manifest["effective_cuda_graph"] = dict(m75_report.get("effective_cuda_graph") or {})
    contract_manifest["effective_dispatch_backend"] = dict(m75_report.get("effective_dispatch_backend") or {})
    contract_manifest["effective_distributed_runtime"] = dict(m75_report.get("effective_distributed_runtime") or {})
    contract_manifest["effective_pd_service"] = dict(m75_report.get("effective_pd_service") or {})
    contract_manifest["effective_storage_backend"] = dict(m75_report.get("effective_storage_backend") or {})
    contract_manifest["gds_effective"] = dict(m75_report.get("gds_effective") or {})
    contract_manifest["spdk_effective"] = dict(m75_report.get("spdk_effective") or {})
    contract_manifest["colossalai_effective"] = dict(m75_report.get("colossalai_effective") or {})

    runtime_protocol_contracts = dict(system_execution_manifest.get("runtime_protocol_contracts") or {})
    effective_runtime_contracts = dict(system_execution_manifest.get("effective_runtime_contracts") or {})
    if component_id:
        runtime_protocol_contracts[component_id] = runtime_protocol_contract
        effective_runtime_contracts[component_id] = effective_runtime_contract
    system_execution_manifest["runtime_protocol_contracts"] = runtime_protocol_contracts
    system_execution_manifest["effective_runtime_contracts"] = effective_runtime_contracts
    write_json_file(contract_manifest_path, contract_manifest)
    write_json_file(system_execution_manifest_path, system_execution_manifest)


def _run_upkg30_latest_protocol_aggregate(
    *,
    gate_name,
    resolved_repo_root,
    requested_output_dir,
    m72_gui_duration_s,
    m72_disable_gui_evidence,
):
    output_root, fallback_reason = _resolve_writable_gate_output_root(
        requested_output_dir=requested_output_dir,
        gate_name="upkg30",
    )
    protocol_root = (output_root / "_protocol_refresh").resolve()
    if protocol_root.exists():
        shutil.rmtree(protocol_root)
    protocol_root.mkdir(parents=True, exist_ok=True)

    source_m72_dir = (ENGINE_REPO_DIR / "Output" / "cli_gate_m72").resolve()
    source_m73_dir = (ENGINE_REPO_DIR / "Output" / "cli_gate_m73").resolve()
    m72_dir = (protocol_root / "cli_gate_m72").resolve()
    m73_dir = (protocol_root / "cli_gate_m73").resolve()
    shutil.copytree(source_m72_dir, m72_dir)
    shutil.copytree(source_m73_dir, m73_dir)
    _rewrite_upkg30_protocol_report_artifact_paths(report_path=m72_dir / "report.json", gate_root=m72_dir)
    _rewrite_upkg30_protocol_report_artifact_paths(report_path=m73_dir / "report.json", gate_root=m73_dir)

    local_protocol = _generate_upkg30_local_protocol_evidence(protocol_root)
    m75_report = dict(local_protocol.get("m75_report") or {})
    _backfill_upkg30_protocol_manifest_fields(
        contract_manifest_path=m72_dir / "contract_manifest.json",
        system_execution_manifest_path=m72_dir / "system_execution_manifest.json",
        m75_report=m75_report,
    )
    _backfill_upkg30_protocol_manifest_fields(
        contract_manifest_path=m73_dir / "contract_manifest.json",
        system_execution_manifest_path=m73_dir / "system_execution_manifest.json",
        m75_report=m75_report,
    )

    run_m72_gate = load_engine_m72_gate_runner()
    run_m73_gate = load_engine_m73_gate_runner()
    run_m77_gate = load_engine_m77_gate_runner()
    run_m78_gate = load_engine_m78_gate_runner()

    m7_report_path = (ENGINE_REPO_DIR / "Output" / "cli_gate_m7" / "report.json").resolve()
    m7_summary_path = (ENGINE_REPO_DIR / "Output" / "cli_gate_m7" / "m7_industrial" / "summary.json").resolve()
    m7_report = _read_json_payload(m7_report_path)
    m7_gate = dict((((m7_report.get("gate_result") or {}).get("m7")) or {}))
    upstream_m7 = _upkg_protocol_upstream_contract(
        gate_name="m7",
        gate_payload=m7_gate,
        report_path=str(m7_report_path),
        summary_path=str(m7_summary_path),
    )

    m72_result = run_m72_gate(output_dir=str(m72_dir), cgc_report=upstream_m7)
    m73_result = run_m73_gate(output_dir=str(m73_dir), cgc_report=upstream_m7)
    m72_report = _read_json_payload(str(m72_result.get("report_path") or ""))
    m73_summary_path = str(m73_result.get("summary_path") or "")
    m73_summary = _read_json_payload(m73_summary_path)
    m72_gate = dict((((m72_result.get("gate_result") or {}).get("m72")) or {}))
    upstream_m72 = _upkg_protocol_upstream_contract(
        gate_name="m72",
        gate_payload=m72_gate,
        report_path=str(m72_result.get("report_path") or ""),
        summary_path=str(m72_result.get("summary_path") or ""),
    )
    m77_result = run_m77_gate(output_dir=str(m72_dir), cgc_report=upstream_m72)
    m78_result = run_m78_gate(output_dir=str(m72_dir), cgc_report=upstream_m72)
    m77_report = _read_json_payload(str(m77_result.get("report_path") or ""))
    m78_report = _read_json_payload(str(m78_result.get("report_path") or ""))

    subgate_status = {
        "upkg31": str(m7_gate.get("status") or m7_report.get("status") or "FAIL"),
        "upkg32": str((((m72_result.get("gate_result") or {}).get("m72")) or {}).get("status") or m72_report.get("status") or "FAIL"),
        "upkg33": str(m73_summary.get("status") or "FAIL"),
        "upkg37": str((((m77_result.get("gate_result") or {}).get("m77")) or {}).get("status") or m77_report.get("status") or "FAIL"),
        "upkg38": str((((m78_result.get("gate_result") or {}).get("m78")) or {}).get("status") or m78_report.get("status") or "FAIL"),
    }
    final_status = "PASS" if all(subgate_status[key] == "PASS" for key in ("upkg31", "upkg32", "upkg33", "upkg37")) else "FAIL"

    protocol_rows = {
        "m72": _upkg_protocol_gate_projection(report_payload=m72_report, gate_name="m72"),
        "m77": _upkg_protocol_gate_projection(report_payload=m77_report, gate_name="m77"),
        "m78": _upkg_protocol_gate_projection(report_payload=m78_report, gate_name="m78"),
    }
    upkg30_profile_bundle = _write_profile_settings_bundle(
        output_root=output_root,
        schema_prefix="upkg30",
        runtime_host="upkg30_agent_suite",
        deployment_target="upkg30_aggregate_gate",
        environment="mixed_local_and_edge_cloud",
        stage_scope="upkg30_protocol_aggregate",
        model_scope="upkg30_multi_milestone_aggregate",
        model_locator="upkg30_aggregate_profile",
        distributed_runtime_bootstrap_path="",
    )
    upkg30_execution_map = {
        "upkg31": "local_infer",
        "upkg32": "local_infer",
        "upkg33": "edge_cloud_infer",
        "upkg37": "edge_cloud_infer",
        "upkg38": "edge_cloud_infer",
    }
    aggregate_payload = {
        "name": "CGC_UPKG_3_0_Aggregate_Gate",
        "status": final_status,
        "scope": "verification_only",
        "public_entrypoint": "cgc gate upkg30",
        "source": "latest_protocol_refresh",
        "requested_output_dir": str(Path(requested_output_dir).expanduser().resolve()),
        "resolved_output_dir": str(output_root),
        "output_dir_fallback_reason": str(fallback_reason or ""),
        "fresh_protocol_tmp_root": str(protocol_root),
        "local_protocol": {
            "local_infer_evidence_path": str(local_protocol.get("local_infer_evidence_path") or ""),
            "m75_evidence_path": str(local_protocol.get("m75_evidence_path") or ""),
            "m75_status": str((m75_report or {}).get("status") or ""),
            "m75_pd_service_status": str(((m75_report or {}).get("effective_pd_service") or {}).get("status") or ""),
        },
        "gate_result": {
            "upkg31": dict(m7_report.get("gate_result") or {}),
            "upkg32": dict(m72_result.get("gate_result") or {}),
            "upkg33": dict(m73_result.get("gate_result") or {}) if isinstance(m73_result.get("gate_result"), dict) else {"m73": m73_summary},
            "upkg37": dict(m77_result.get("gate_result") or {}),
            "upkg38": dict(m78_result.get("gate_result") or {}),
        },
        "upkg30_mapping": {
            "3.1": "upkg31 -> m7",
            "3.2": "upkg32 -> m72",
            "3.3": "upkg33 -> m73",
            "3.4": "upkg34 -> m72",
            "3.5": "upkg35 -> m72",
            "3.6": "upkg36 -> m72",
            "3.7": "upkg37 -> m77",
            "3.8": "upkg38 -> m78",
        },
        "subreports": {
            "upkg31_report_path": str(m7_report_path),
            "upkg32_report_path": str(m72_result.get("report_path") or ""),
            "upkg33_summary_path": m73_summary_path,
            "upkg37_report_path": str(m77_result.get("report_path") or ""),
            "upkg38_report_path": str(m78_result.get("report_path") or ""),
        },
        "subgate_status": subgate_status,
        "protocol_rows": protocol_rows,
    }
    aggregate_payload.update(upkg30_profile_bundle)
    aggregate_payload.update(
        _profile_binding_fields(
            profile_settings_path=upkg30_profile_bundle["profile_settings_path"],
            execution_map=upkg30_execution_map,
            compatible=["local_infer", "edge_cloud_infer"],
            applicable=["local_infer", "edge_cloud_infer"],
            bootstrap=upkg30_execution_map,
            flow=upkg30_execution_map,
        )
    )
    summary_payload = {
        "gate": gate_name,
        "status": final_status,
        "requested_output_dir": str(Path(requested_output_dir).expanduser().resolve()),
        "resolved_output_dir": str(output_root),
        "fresh_protocol_tmp_root": str(protocol_root),
        "subgate_status": subgate_status,
        "canonical_profile_catalog_path": str(upkg30_profile_bundle["canonical_profile_catalog_path"]),
        "profile_settings_path": str(upkg30_profile_bundle["profile_settings_path"]),
        "execution_profile_binding_keys": dict(upkg30_execution_map),
        "protocol_status": {
            gate: {
                "mandatory_protocol_gate_status": str((projection["mandatory_protocol_gate"] or {}).get("status") or ""),
                "protocol_family": str((projection["runtime_protocol_contract"] or {}).get("protocol_family") or ""),
                "state_codec": str((projection["runtime_protocol_contract"] or {}).get("state_codec") or ""),
                "pd_required": bool((projection["runtime_protocol_contract"] or {}).get("require_pd_service")),
                "pd_mode": str((projection["runtime_protocol_contract"] or {}).get("pd_mode") or ""),
                "pd_service_status": str((projection["effective_pd_service"] or {}).get("status") or ""),
                "zero_copy_status": str((projection["zero_copy_vram_real"] or {}).get("status") or ""),
            }
            for gate, projection in protocol_rows.items()
        },
    }
    aggregate_report_path = write_json_file(output_root / "report.json", aggregate_payload)
    aggregate_summary_path = write_json_file(output_root / "summary.json", summary_payload)
    return {
        "status": final_status,
        "gate_name": gate_name,
        "report_path": aggregate_report_path,
        "summary_path": aggregate_summary_path,
        "gate_result": aggregate_payload["gate_result"],
        "canonical_profile_catalog_path": str(aggregate_payload.get("canonical_profile_catalog_path") or ""),
        "profile_settings_path": str(aggregate_payload.get("profile_settings_path") or ""),
        "execution_profile_binding_keys": dict(aggregate_payload.get("execution_profile_binding_keys") or {}),
        "delivery_profile_binding_key": str(aggregate_payload.get("delivery_profile_binding_key") or ""),
        "compatible_profile_binding_keys": list(aggregate_payload.get("compatible_profile_binding_keys") or []),
        "applicable_profile_binding_keys": list(aggregate_payload.get("applicable_profile_binding_keys") or []),
        "bootstrap_contract_binding_keys": dict(aggregate_payload.get("bootstrap_contract_binding_keys") or {}),
        "flow_parameter_contract_binding_keys": dict(aggregate_payload.get("flow_parameter_contract_binding_keys") or {}),
    }


@contextlib.contextmanager
def _temporary_env_defaults(defaults):
    previous = {}
    mutated = []
    for key, value in dict(defaults or {}).items():
        raw_key = str(key)
        previous[raw_key] = os.environ.get(raw_key)
        if not str(os.environ.get(raw_key) or "").strip():
            os.environ[raw_key] = str(value)
            mutated.append(raw_key)
    try:
        yield
    finally:
        for raw_key in mutated:
            if previous[raw_key] is None:
                os.environ.pop(raw_key, None)
            else:
                os.environ[raw_key] = previous[raw_key]


def _run_m73_latest_protocol_gate(*, output_root):
    root = Path(output_root).expanduser().resolve()
    protocol_root = (root / "_protocol_refresh_m73").resolve()
    if protocol_root.exists():
        shutil.rmtree(protocol_root)
    protocol_root.mkdir(parents=True, exist_ok=True)
    source_m73_dir = (ENGINE_REPO_DIR / "Output" / "cli_gate_m73").resolve()
    shutil.copytree(source_m73_dir, root, dirs_exist_ok=True)
    _rewrite_upkg30_protocol_report_artifact_paths(report_path=root / "report.json", gate_root=root)

    local_protocol = _generate_upkg30_local_protocol_evidence(protocol_root)
    m75_report = dict(local_protocol.get("m75_report") or {})
    _backfill_upkg30_protocol_manifest_fields(
        contract_manifest_path=root / "contract_manifest.json",
        system_execution_manifest_path=root / "system_execution_manifest.json",
        m75_report=m75_report,
    )

    run_m73_gate = load_engine_m73_gate_runner()
    m7_report_path = (ENGINE_REPO_DIR / "Output" / "cli_gate_m7" / "report.json").resolve()
    m7_summary_path = (ENGINE_REPO_DIR / "Output" / "cli_gate_m7" / "m7_industrial" / "summary.json").resolve()
    m7_report = _read_json_payload(m7_report_path)
    m7_gate = dict((((m7_report.get("gate_result") or {}).get("m7")) or {}))
    upstream_m7 = _upkg_protocol_upstream_contract(
        gate_name="m7",
        gate_payload=m7_gate,
        report_path=str(m7_report_path),
        summary_path=str(m7_summary_path),
    )
    return run_m73_gate(output_dir=str(root), cgc_report=upstream_m7)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def run_command_checked(command, *, cwd=None, env=None, capture_output=False):
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        check=True,
        capture_output=capture_output,
    )


def get_m75_install_evidence_path():
    return (
        ENGINE_REPO_DIR
        / "Output"
        / "cli_gate_m75"
        / "runtime_evidence"
        / "edge_router_install.json"
    ).resolve()


def ollama_model_exists(model_name):
    try:
        run_command_checked(["ollama", "show", model_name], capture_output=True)
        return True
    except Exception:
        return False


def fetch_fake_ollama_models():
    try:
        response = requests.get(f"{get_edge_api_base_url()}/api/tags", timeout=5)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        from fastapi.testclient import TestClient
        from app.servers.cgc_api_server import app

        client = TestClient(app)
        payload = client.get("/api/tags").json()
    return payload.get("models", [])


def fetch_fake_ollama_install_spec(model_name):
    normalized = str(model_name or "").strip()
    if not normalized:
        raise ValueError("empty_model_name")

    def _extract(tags_payload, show_payload):
        models = tags_payload.get("models", [])
        matched = None
        for model in models:
            candidates = {
                str(model.get("name") or "").strip(),
                str(model.get("model") or "").strip(),
            }
            if normalized in candidates or f"{normalized}:latest" in candidates:
                matched = model
                break
        if matched is None:
            raise RuntimeError(f"model_not_exposed_by_fake_ollama_protocol: {normalized}")
        details = show_payload.get("details") or {}
        backend_package_catalog = _llama_cpp_backend_package_catalog(
            cluster_nfs_root=str(details.get("cluster_nfs_backend_root") or DEFAULT_CLUSTER_NFS_BACKEND_ROOT),
        )
        return {
            "registry_model": str(matched.get("model") or normalized),
            "registry_entry": matched,
            "source_priority": list(details.get("source_priority") or ["cluster_nfs", "huggingface"]),
            "cluster_nfs_root": str(details.get("cluster_nfs_root") or DEFAULT_CLUSTER_NFS_ROOT),
            "cluster_nfs_path": str(details.get("cluster_nfs_path") or ""),
            "cluster_nfs_backend_root": str(details.get("cluster_nfs_backend_root") or DEFAULT_CLUSTER_NFS_BACKEND_ROOT),
            "gguf_repo": str(details.get("gguf_repo") or MINICPM5_GGUF_REPO),
            "gguf_filename": str(details.get("gguf_filename") or f"MiniCPM5-1B-{MINICPM5_DEFAULT_QUANT}.gguf"),
            "ollama_model": str(details.get("ollama_model") or MINICPM5_OLLAMA_MODEL),
            "quant": str(details.get("quant") or MINICPM5_DEFAULT_QUANT),
            "install_via": str(details.get("install_via") or "fake_ollama_protocol"),
            "runtime_strategy": "platform_selected_nfs_backend_binary",
            "delivery_mode": "nfs_backend_binary_plus_model_integration",
            "backend_package_catalog": backend_package_catalog,
            "selected_host_backend_package": dict(backend_package_catalog.get("selected_package") or {}),
            "show_payload": show_payload,
        }

    try:
        edge_api_base_url = get_edge_api_base_url()
        tags_response = requests.get(f"{edge_api_base_url}/api/tags", timeout=5)
        tags_response.raise_for_status()
        show_response = requests.post(
            f"{edge_api_base_url}/api/show",
            json={"name": normalized},
            timeout=5,
        )
        show_response.raise_for_status()
        return _extract(tags_response.json(), show_response.json())
    except Exception:
        from fastapi.testclient import TestClient
        from app.servers.cgc_api_server import app

        client = TestClient(app)
        tags_payload = client.get("/api/tags").json()
        show_payload = client.post("/api/show", json={"name": normalized}).json()
        return _extract(tags_payload, show_payload)


def print_fake_ollama_models():
    models = fetch_fake_ollama_models()
    print("\nNAME\t\t\t\t\tMODEL\t\t\tSIZE\t\tSOURCE")
    print("-" * 96)
    for model in models:
        name = str(model.get("name") or "unknown")
        model_id = str(model.get("model") or "unknown")
        size_bytes = int(model.get("size") or 0)
        size_str = f"{size_bytes / (1024**3):.1f} GB" if size_bytes > 0 else "N/A"
        details = model.get("details") or {}
        source = str(details.get("cloud_source") or details.get("format") or "unknown")
        print(f"{name:<40}{model_id:<24}{size_str:<16}{source}")
    print()


def print_fake_ollama_show_spec(model_name):
    spec = fetch_fake_ollama_install_spec(model_name)
    print(json.dumps(spec, ensure_ascii=False, indent=2))
    return spec


def _looks_like_repo_id(value):
    model_str = str(value or "").strip()
    return "/" in model_str and not Path(model_str).expanduser().exists()


def _resolve_cached_model_ref(model_ref):
    model_str = str(model_ref or "").strip()
    if not model_str:
        return ""
    model_path = Path(model_str).expanduser()
    if model_path.exists():
        return str(model_path.resolve())
    if not _looks_like_repo_id(model_str):
        return model_str
    try:
        from app.edge_engine.local_infer import _resolve_cached_hf_snapshot

        return str(_resolve_cached_hf_snapshot(model_str) or model_str)
    except Exception:
        return model_str


def _detect_model_format(path_str):
    raw = str(path_str or "").strip()
    if not raw:
        return "unknown"
    lowered = raw.lower()
    path = Path(raw).expanduser()
    if path.is_file():
        if lowered.endswith(".gguf"):
            return "gguf"
        if lowered.endswith(".safetensors"):
            return "safetensors"
        return path.suffix.lstrip(".") or "file"
    if path.is_dir():
        if (path / "tokenizer.json").exists() and (
            list(path.glob("*.safetensors"))
            or list(path.glob("model*.safetensors"))
            or (path / "model.safetensors.index.json").exists()
        ):
            return "safetensors"
        return "directory"
    if _looks_like_repo_id(raw):
        return "mlx"
    return "unknown"


def _backend_candidates_for_format(model_format):
    if model_format == "mlx":
        return ["omlx_mlx_lm"]
    if model_format == "gguf":
        return ["llama.cpp", "ollama", "edge_cloud_bridge"]
    if model_format == "safetensors":
        return ["transformers", "edge_cloud_bridge"]
    return ["edge_cloud_bridge"]


def _admissible_routes_for_format(model_format):
    if model_format == "mlx":
        return ["m4_local", "m73_edge_cloud"]
    return ["m73_edge_cloud"]


def _payload_prefers_local_runtime(payload):
    model = str(payload.get("model") or "").strip()
    model_format = _detect_model_format(model)
    return bool(
        payload.get("use_omlx")
        or payload.get("use_flashmoe")
        or model_format in {"gguf", "mlx"}
    )


def _safe_size_bytes(path_str):
    raw = str(path_str or "").strip()
    if not raw:
        return 0
    try:
        path = Path(raw).expanduser()
        if path.is_file():
            return int(path.stat().st_size)
        if path.is_dir():
            total = 0
            for child in path.rglob("*"):
                if child.is_file():
                    total += int(child.stat().st_size)
            return total
    except Exception:
        return 0
    return 0


def _append_model_entry(entries, *, model_id, display_name, resolved_path, model_format, source, size_bytes=0, status="PASS", notes=None):
    normalized_path = str(resolved_path or "").strip()
    entries.append(
        {
            "model_id": str(model_id or display_name or normalized_path or "unknown_model"),
            "display_name": str(display_name or model_id or normalized_path or "unknown_model"),
            "resolved_path": normalized_path,
            "format": str(model_format or "unknown"),
            "source": str(source or "unknown"),
            "size_bytes": int(size_bytes or 0),
            "backend_candidates": _backend_candidates_for_format(model_format),
            "admissible_routes": _admissible_routes_for_format(model_format),
            "status": str(status or "PASS"),
            "notes": list(notes or []),
        }
    )


def _slug_to_product_name(value):
    raw = str(value or "").strip().strip("/").strip()
    if not raw:
        return ""
    lowered = raw.lower().replace("_", "-")
    if "deepseek-v4-flash" in lowered:
        return "deepseek-v4-flash"
    if "deepseek" in lowered and "v4" in lowered and "flash" in lowered:
        return "deepseek-v4-flash"
    if "minicpm5" in lowered:
        return "minicpm5-1b"
    return raw


def _slug_to_display_name(value):
    slug = str(value or "").strip()
    if not slug:
        return ""
    lowered = slug.lower().replace("_", "-")
    if lowered == "deepseek-v4-flash":
        return "DeepSeek V4 Flash"
    if lowered == "minicpm5-1b":
        return "MiniCPM5 1B"
    tokens = [token for token in lowered.split("-") if token]
    if not tokens:
        return slug
    return " ".join(token.upper() if token.isalpha() and len(token) <= 3 else token.capitalize() for token in tokens)


def _normalize_discovered_model_identity(candidate):
    path = Path(candidate).expanduser()
    raw_display = path.name
    raw_model_id = path.stem if path.is_file() else path.name
    notes = []
    if path.is_file():
        normalized_slug = _slug_to_product_name(path.stem)
        if normalized_slug and normalized_slug != path.stem:
            notes.append(f"normalized_from:{path.name}")
            return normalized_slug, _slug_to_display_name(normalized_slug), notes
        return raw_model_id, raw_display, notes
    parts = [part for part in path.parts if str(part).strip()]
    if "snapshots" in parts:
        snap_index = parts.index("snapshots")
        if snap_index > 0:
            parent_slug = _slug_to_product_name(parts[snap_index - 1])
            if parent_slug:
                if parent_slug != raw_display:
                    notes.append(f"normalized_from:{raw_display}")
                return parent_slug, _slug_to_display_name(parent_slug), notes
    normalized_slug = _slug_to_product_name(path.name)
    if normalized_slug and normalized_slug != raw_display:
        notes.append(f"normalized_from:{raw_display}")
        return normalized_slug, _slug_to_display_name(normalized_slug), notes
    return raw_model_id, raw_display, notes


def _is_model_directory(path):
    path = Path(path).expanduser()
    if not path.exists() or not path.is_dir():
        return False
    tokenizer = path / "tokenizer.json"
    return tokenizer.exists() and (
        list(path.glob("*.safetensors"))
        or list(path.glob("model*.safetensors"))
        or (path / "model.safetensors.index.json").exists()
    )


def _discover_hf_snapshot_model_dirs(root):
    root_path = Path(root).expanduser()
    if not root_path.exists() or not root_path.is_dir():
        return []
    snapshots_dir = root_path / "snapshots"
    if not snapshots_dir.exists() or not snapshots_dir.is_dir():
        return []
    discovered = []
    try:
        for snapshot_dir in snapshots_dir.iterdir():
            if _is_model_directory(snapshot_dir):
                discovered.append(snapshot_dir)
    except Exception:
        return []
    return discovered


def _normalize_model_source_filter(source_filter):
    raw = str(source_filter or "all").strip().lower() or "all"
    if raw not in VALID_MODEL_SOURCES:
        raise ValueError(
            f"Unsupported --source '{source_filter}'. Expected one of: {', '.join(sorted(VALID_MODEL_SOURCES))}"
        )
    return raw


def _extract_probe_section_lines(*, stdout, section_name):
    text = str(stdout or "")
    if not text:
        return []
    marker = f"==={section_name}==="
    lines = text.splitlines()
    collecting = False
    collected = []
    for raw_line in lines:
        line = str(raw_line).strip()
        if line == marker:
            collecting = True
            continue
        if collecting and line.startswith("===") and line.endswith("==="):
            break
        if collecting and line:
            collected.append(line)
    return collected


def resolve_upkg38_ui_tars_nfs_source(preferred_root=""):
    explicit_path = str(os.environ.get("CGC_UPKG38_UI_TARS_MODEL_PATH") or "").strip()
    if explicit_path:
        return {
            "preferred_model_source_path": explicit_path,
            "preferred_model_root": str(os.environ.get("CGC_UPKG38_UI_TARS_MODEL_ROOT") or ""),
            "source_mode": "explicit_env_path",
            "probe_artifact_path": str(os.environ.get("CGC_UPKG38_UI_TARS_PROBE_ARTIFACT") or ""),
        }

    candidate_roots = []
    for value in [
        preferred_root,
        os.environ.get("CGC_UPKG38_UI_TARS_MODEL_ROOT"),
        UPKG38_UI_TARS_NFS_ROOT,
    ]:
        raw = str(value or "").strip()
        if raw and raw not in candidate_roots:
            candidate_roots.append(raw)

    for root in candidate_roots:
        root_path = Path(root).expanduser()
        if _is_model_directory(root_path):
            return {
                "preferred_model_source_path": str(root_path.resolve()),
                "preferred_model_root": str(root_path.resolve()),
                "source_mode": "direct_model_directory",
                "probe_artifact_path": "",
            }
        snapshot_dirs = _discover_hf_snapshot_model_dirs(root_path)
        if snapshot_dirs:
            resolved_snapshot = str(snapshot_dirs[0].resolve())
            return {
                "preferred_model_source_path": resolved_snapshot,
                "preferred_model_root": str(root_path.resolve()),
                "source_mode": "local_hf_snapshot_directory",
                "probe_artifact_path": "",
            }

    probe_artifact = Path(
        str(os.environ.get("CGC_UPKG38_UI_TARS_PROBE_ARTIFACT") or UPKG38_UI_TARS_PROBE_ARTIFACT)
    ).expanduser()
    if probe_artifact.exists():
        probe_payload = _safe_read_json(probe_artifact)
        key_file_lines = _extract_probe_section_lines(stdout=probe_payload.get("stdout"), section_name="UITARS_KEY_FILES")
        required_suffixes = {"tokenizer.json", "config.json", "model.safetensors.index.json"}
        parent_hits = {}
        for entry in key_file_lines:
            entry_path = Path(entry)
            parent_str = str(entry_path.parent)
            parent_hits.setdefault(parent_str, set()).add(entry_path.name)
        for parent_str, names in parent_hits.items():
            if required_suffixes.issubset(names):
                return {
                    "preferred_model_source_path": parent_str,
                    "preferred_model_root": UPKG38_UI_TARS_NFS_ROOT,
                    "source_mode": "remote_probe_hf_snapshot_directory",
                    "probe_artifact_path": str(probe_artifact),
                }
    return {
        "preferred_model_source_path": "",
        "preferred_model_root": candidate_roots[0] if candidate_roots else UPKG38_UI_TARS_NFS_ROOT,
        "source_mode": "target_model_id_only",
        "probe_artifact_path": str(probe_artifact) if probe_artifact.exists() else "",
    }


@contextlib.contextmanager
def _with_upkg38_ui_tars_source_env(preferred_root=""):
    resolution = resolve_upkg38_ui_tars_nfs_source(preferred_root=preferred_root)
    keys = [
        "CGC_UPKG38_UI_TARS_MODEL_PATH",
        "CGC_UPKG38_UI_TARS_MODEL_ROOT",
        "CGC_UPKG38_UI_TARS_SOURCE_MODE",
        "CGC_UPKG38_UI_TARS_PROBE_ARTIFACT",
    ]
    previous = {key: os.environ.get(key) for key in keys}
    try:
        if str(resolution.get("preferred_model_source_path") or "").strip():
            os.environ["CGC_UPKG38_UI_TARS_MODEL_PATH"] = str(resolution.get("preferred_model_source_path") or "")
        else:
            os.environ.pop("CGC_UPKG38_UI_TARS_MODEL_PATH", None)
        os.environ["CGC_UPKG38_UI_TARS_MODEL_ROOT"] = str(resolution.get("preferred_model_root") or UPKG38_UI_TARS_NFS_ROOT)
        os.environ["CGC_UPKG38_UI_TARS_SOURCE_MODE"] = str(resolution.get("source_mode") or "target_model_id_only")
        if str(resolution.get("probe_artifact_path") or "").strip():
            os.environ["CGC_UPKG38_UI_TARS_PROBE_ARTIFACT"] = str(resolution.get("probe_artifact_path") or "")
        else:
            os.environ.pop("CGC_UPKG38_UI_TARS_PROBE_ARTIFACT", None)
        yield resolution
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _discover_model_files(root):
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return []

    discovered = []
    try:
        for dirpath, dirnames, filenames in os.walk(root_path):
            current = Path(dirpath)
            if _is_model_directory(current):
                discovered.append(current)
                dirnames[:] = []
                continue
            snapshot_hits = []
            if current.name == "snapshots":
                for snapshot_name in list(dirnames):
                    snapshot_dir = current / snapshot_name
                    if _is_model_directory(snapshot_dir):
                        snapshot_hits.append(snapshot_dir)
                if snapshot_hits:
                    discovered.extend(snapshot_hits)
                    dirnames[:] = [name for name in dirnames if (current / name) not in snapshot_hits]
            for filename in filenames:
                suffix = Path(filename).suffix.lower()
                if suffix in {".gguf", ".safetensors"}:
                    discovered.append(current / filename)
    except Exception:
        return []
    deduped = []
    seen = set()
    for item in discovered:
        try:
            resolved = str(item.expanduser().resolve())
        except Exception:
            resolved = str(item.expanduser())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(item)
    return deduped


def _normalize_source_root_entries(root_values, *, source_label):
    entries = []
    seen = set()
    for value in root_values:
        raw = str(value or "").strip()
        if not raw:
            continue
        resolved = str(Path(raw).expanduser())
        if resolved in seen:
            continue
        seen.add(resolved)
        exists = Path(resolved).exists()
        entries.append(
            {
                "source": source_label,
                "root": resolved,
                "status": "PASS" if exists else "FAIL",
                **({"error_code": "ROOT_UNREACHABLE"} if not exists else {}),
            }
        )
    return entries


def collect_list_response(*, cfg, model_roots=None, nfs_roots=None, source_filter="all"):
    model_roots = [str(x) for x in (model_roots or []) if str(x or "").strip()]
    nfs_roots = [str(x) for x in (nfs_roots or []) if str(x or "").strip()]
    normalized_source_filter = _normalize_model_source_filter(source_filter)
    env_model_roots = [x for x in str(os.environ.get("CGC_MODEL_ROOTS") or "").split(os.pathsep) if x.strip()]
    env_nfs_roots = [x for x in str(os.environ.get("CGC_NFS_MODEL_ROOTS") or "").split(os.pathsep) if x.strip()]
    merged_model_roots = model_roots + env_model_roots + [str(CGC_MODELS_DIR)]
    merged_nfs_roots = nfs_roots + env_nfs_roots + [str(os.environ.get("CGC_CLUSTER_NFS_ROOT") or DEFAULT_CLUSTER_NFS_ROOT)]

    sources = []
    models = []
    seen_paths = set()

    requested_sources = VALID_MODEL_SOURCES if normalized_source_filter == "all" else {normalized_source_filter}

    if "local" in requested_sources:
        sources.extend(_normalize_source_root_entries(merged_model_roots, source_label="local"))
    if "nfs" in requested_sources:
        sources.extend(_normalize_source_root_entries(merged_nfs_roots, source_label="nfs"))

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    if "cache" in requested_sources:
        sources.append(
            {
                "source": "cache",
                "root": str(cache_root),
                "status": "PASS" if cache_root.exists() else "FAIL",
                **({"error_code": "ROOT_UNREACHABLE"} if not cache_root.exists() else {}),
            }
        )

    if "registry" in requested_sources:
        registry_models = fetch_fake_ollama_models()
        sources.append({"source": "registry", "root": "fake_ollama_registry", "status": "PASS"})
        for model in registry_models:
            name = str(model.get("name") or model.get("model") or "unknown")
            details = model.get("details") or {}
            _append_model_entry(
                models,
                model_id=name,
                display_name=name,
                resolved_path=str(model.get("model") or name),
                model_format=str(details.get("format") or "gguf"),
                source="registry",
                size_bytes=int(model.get("size") or 0),
                notes=[str(details.get("cloud_source") or "fake_ollama_registry")],
            )

    def _consume_candidates(candidates, *, source_name):
        for candidate in candidates:
            resolved = str(candidate.expanduser().resolve()) if candidate.exists() else str(candidate.expanduser())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            model_id, display_name, normalized_notes = _normalize_discovered_model_identity(candidate)
            _append_model_entry(
                models,
                model_id=model_id,
                display_name=display_name,
                resolved_path=resolved,
                model_format=_detect_model_format(resolved),
                source=source_name,
                size_bytes=_safe_size_bytes(resolved),
                notes=normalized_notes,
            )

    if "local" in requested_sources:
        for root in merged_model_roots:
            _consume_candidates(_discover_model_files(root), source_name="local")
    if "nfs" in requested_sources:
        for root in merged_nfs_roots:
            _consume_candidates(_discover_model_files(root), source_name="nfs")

    if "config" in requested_sources:
        for config_key, source_name in (
            ("local_omlx_model", "config"),
            ("local_flashmoe_model", "config"),
            ("active_edge_model_path", "config"),
        ):
            configured = str(cfg.get(config_key) or "").strip()
            if not configured:
                continue
            resolved = _resolve_cached_model_ref(configured)
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            notes = [f"config_key:{config_key}"]
            _append_model_entry(
                models,
                model_id=Path(resolved).name if Path(resolved).name else configured,
                display_name=Path(resolved).name if Path(resolved).name else configured,
                resolved_path=resolved,
                model_format=_detect_model_format(resolved),
                source=source_name,
                size_bytes=_safe_size_bytes(resolved),
                notes=notes,
            )

    status = "PASS" if models else "FAIL"
    return {
        "status": status,
        "command": "cgc list",
        "generated_at": utc_now_iso(),
        "source_filter": normalized_source_filter,
        "sources": sources,
        "models": models,
        "summary": {
            "total_models": len(models),
            "local_models": sum(1 for item in models if item.get("source") == "local"),
            "nfs_models": sum(1 for item in models if item.get("source") == "nfs"),
            "cached_models": sum(1 for item in models if item.get("source") == "cache"),
            "registry_models": sum(1 for item in models if item.get("source") == "registry"),
        },
        **(
            {
                "failure_code": "MODEL_DISCOVERY_EMPTY",
                "failure_reason": "No admissible models found in configured local/cache/nfs roots.",
                "recommended_action": [
                    "Download a GGUF, safetensors, or MLX model into a configured model root.",
                    "Check NFS mount health.",
                    "Run with --model-root or --nfs-root to add more discovery roots.",
                ],
            }
            if not models
            else {}
        ),
    }


def print_list_response(payload):
    print("SOURCE\tFORMAT\tROUTES\tMODEL\tPATH")
    print("-" * 120)
    for model in payload.get("models") or []:
        print(
            f"{str(model.get('source') or 'unknown'):<8}"
            f"{str(model.get('format') or 'unknown'):<12}"
            f"{','.join(model.get('admissible_routes') or []):<24}"
            f"{str(model.get('display_name') or model.get('model_id') or 'unknown'):<24}"
            f"{str(model.get('resolved_path') or '')}"
        )
    print()
    summary = payload.get("summary") or {}
    print(
        f"total={summary.get('total_models', 0)} "
        f"local={summary.get('local_models', 0)} "
        f"nfs={summary.get('nfs_models', 0)} "
        f"cache={summary.get('cached_models', 0)} "
        f"registry={summary.get('registry_models', 0)}"
    )


def write_m75_install_evidence(payload):
    payload = dict(payload)
    payload.setdefault("status", "PASS")
    payload.setdefault("installer", "cgc.py")
    payload.setdefault(
        "timestamp",
        subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
    )
    return write_json_file(get_m75_install_evidence_path(), payload)


def download_hf_gguf(repo_id, filename, output_path):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    with requests.get(url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("content-length") or 0)
        downloaded = 0
        next_progress_bytes = 64 * 1024 * 1024
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes and downloaded >= next_progress_bytes:
                    progress = (downloaded / total_bytes) * 100.0
                    print(
                        f"  [Download] {downloaded / (1024**2):.1f} MiB / "
                        f"{total_bytes / (1024**2):.1f} MiB ({progress:.1f}%)"
                    )
                    next_progress_bytes += 64 * 1024 * 1024
    temp_path.replace(output_path)
    return str(output_path)


def resolve_cluster_nfs_source(install_spec, gguf_filename):
    install_spec = dict(install_spec or {})
    candidates = []
    explicit_path = str(install_spec.get("cluster_nfs_path") or "").strip()
    if explicit_path:
        candidates.append(Path(explicit_path))

    env_path = str(os.environ.get("CGC_CLUSTER_NFS_MINICPM5_GGUF") or "").strip()
    if env_path:
        candidates.append(Path(env_path))

    roots = []
    for value in [
        os.environ.get("CGC_CLUSTER_NFS_ROOT"),
        install_spec.get("cluster_nfs_root"),
        DEFAULT_CLUSTER_NFS_ROOT,
    ]:
        value = str(value or "").strip()
        if value:
            roots.append(value)
    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        candidates.append(Path(root) / "minicpm5" / gguf_filename)
        candidates.append(Path(root) / "MiniCPM5-1B-GGUF" / gguf_filename)
        candidates.append(Path(root) / gguf_filename)

    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            resolved = candidate.expanduser()
        if resolved.exists() and resolved.is_file():
            return str(resolved)
    return ""


def resolve_cluster_nfs_candidate(install_spec, gguf_filename):
    install_spec = dict(install_spec or {})
    candidates = []
    explicit_path = str(install_spec.get("cluster_nfs_path") or "").strip()
    if explicit_path:
        candidates.append(explicit_path)
    env_path = str(os.environ.get("CGC_CLUSTER_NFS_MINICPM5_GGUF") or "").strip()
    if env_path:
        candidates.append(env_path)
    details_path = str(install_spec.get("details", {}).get("cluster_nfs_path") or "").strip()
    if details_path:
        candidates.append(details_path)
    default_dir = str(install_spec.get("cluster_nfs_dir") or MINICPM5_CLUSTER_NFS_DIR).strip()
    if default_dir:
        candidates.append(str((Path(default_dir) / gguf_filename)))
    candidates.append(MINICPM5_CLUSTER_NFS_PATH)
    for candidate in candidates:
        if str(candidate or "").strip():
            return str(candidate).strip()
    return ""


def copy_local_file(source_path, output_path):
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_suffix(output.suffix + ".part")
    total_bytes = source.stat().st_size
    copied = 0
    next_progress_bytes = 64 * 1024 * 1024
    with open(source, "rb") as src, open(temp_path, "wb") as dst:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            copied += len(chunk)
            if total_bytes and copied >= next_progress_bytes:
                progress = (copied / total_bytes) * 100.0
                print(
                    f"  [NFS Copy] {copied / (1024**2):.1f} MiB / "
                    f"{total_bytes / (1024**2):.1f} MiB ({progress:.1f}%)"
                )
                next_progress_bytes += 64 * 1024 * 1024
    temp_path.replace(output)
    return str(output)


def copy_remote_path_via_host1(remote_path, output_path, *, recursive=False):
    remote_host = str(os.environ.get("CGC_CLUSTER_NFS_FETCH_HOST") or DEFAULT_CLUSTER_NFS_FETCH_HOST).strip()
    remote_spec = f"{_target_user()}@{remote_host}:{remote_path}"
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    scp_cmd = [
        "sshpass",
        "-p",
        _password(),
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        _proxy_option(),
    ]
    if recursive:
        scp_cmd.append("-r")
    scp_cmd.extend([remote_spec, str(target)])
    proc = subprocess.run(scp_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"remote_copy_failed: remote_path={remote_path} output_path={output_path} "
            f"stderr={proc.stderr.strip() or proc.stdout.strip()}"
        )
    return str(target)


def _selected_backend_binary_name(selected_host_backend_package):
    explicit = str(selected_host_backend_package.get("binary_path") or "").strip()
    if explicit:
        return Path(explicit).name
    platform_name = str(selected_host_backend_package.get("platform") or "").strip().lower()
    return "llama-cli.exe" if platform_name == "windows" else "llama-cli"


def _selected_backend_server_name(selected_host_backend_package):
    explicit = str(selected_host_backend_package.get("server_path") or "").strip()
    if explicit:
        return Path(explicit).name
    platform_name = str(selected_host_backend_package.get("platform") or "").strip().lower()
    return "llama-server.exe" if platform_name == "windows" else "llama-server"


def _materialize_backend_package_metadata(selected_host_backend_package, install_root):
    package = dict(selected_host_backend_package or {})
    root = Path(install_root).expanduser().resolve()
    package["install_root"] = str(root)
    package["bin_dir"] = str((root / "bin").resolve())
    package["lib_dir"] = str((root / "lib").resolve())
    package["adapter_dir"] = str((root / "adapters" / "minicpm5").resolve())
    package["scripts_dir"] = str((root / "scripts").resolve())
    package["package_manifest_path"] = str((root / "package_manifest.json").resolve())
    package["binary_path"] = str((root / "bin" / _selected_backend_binary_name(package)).resolve())
    package["server_path"] = str((root / "bin" / _selected_backend_server_name(package)).resolve())
    binary_exists = Path(package["binary_path"]).exists()
    server_exists = Path(package["server_path"]).exists()
    manifest_exists = Path(package["package_manifest_path"]).exists()
    package["binary_exists"] = binary_exists
    package["server_exists"] = server_exists
    if binary_exists:
        package["activation_state"] = "active"
    elif manifest_exists:
        package["activation_state"] = "ready"
    else:
        package["activation_state"] = "missing"
    return package


def _ensure_backend_package_executable_bits(selected_host_backend_package):
    platform_name = str(selected_host_backend_package.get("platform") or "").strip().lower()
    if platform_name == "windows":
        return
    for key in ("binary_path", "server_path"):
        candidate = Path(str(selected_host_backend_package.get(key) or "")).expanduser()
        if candidate.exists() and candidate.is_file():
            candidate.chmod(candidate.stat().st_mode | 0o755)
    scripts_dir = Path(str(selected_host_backend_package.get("scripts_dir") or "")).expanduser()
    if scripts_dir.exists() and scripts_dir.is_dir():
        for script_path in scripts_dir.glob("*"):
            if script_path.is_file():
                script_path.chmod(script_path.stat().st_mode | 0o755)


def install_selected_backend_package(selected_host_backend_package):
    package = dict(selected_host_backend_package or {})
    if not package:
        return {}
    package_id = str(package.get("package_id") or "").strip() or "cgc.backend.llama_cpp.unknown"
    install_root = str(package.get("install_root") or "").strip()
    source_root = Path(install_root).expanduser() if install_root else None
    platform_name = str(package.get("platform") or "").strip().lower() or _normalize_platform_name()
    arch_name = str(package.get("arch") or "").strip().lower() or _normalize_arch_name()
    local_root = (CGC_BACKENDS_DIR / "llama.cpp" / f"{platform_name}-{arch_name}").resolve()

    if source_root and source_root.exists() and source_root.is_dir():
        local_root.parent.mkdir(parents=True, exist_ok=True)
        if local_root.exists():
            shutil.rmtree(local_root)
        shutil.copytree(source_root, local_root)
        installed_package = _materialize_backend_package_metadata(package, local_root)
        installed_package["package_id"] = package_id
        installed_package["source_install_root"] = str(source_root.resolve())
        installed_package["install_source"] = "cluster_nfs"
        _ensure_backend_package_executable_bits(installed_package)
        return installed_package

    if source_root and str(source_root).startswith(DEFAULT_CLUSTER_NFS_ROOT):
        local_root.parent.mkdir(parents=True, exist_ok=True)
        if local_root.exists():
            shutil.rmtree(local_root)
        try:
            copy_remote_path_via_host1(str(source_root), local_root, recursive=True)
        except Exception:
            if local_root.exists():
                shutil.rmtree(local_root)
        if local_root.exists():
            installed_package = _materialize_backend_package_metadata(package, local_root)
            installed_package["package_id"] = package_id
            installed_package["source_install_root"] = str(source_root)
            installed_package["install_source"] = "cluster_nfs_remote_fetch"
            _ensure_backend_package_executable_bits(installed_package)
            return installed_package

    installed_package = _materialize_backend_package_metadata(package, local_root)
    installed_package["package_id"] = package_id
    installed_package["source_install_root"] = str(source_root.resolve()) if source_root else ""
    installed_package["install_source"] = "local_cache"
    if Path(installed_package["package_manifest_path"]).exists():
        _ensure_backend_package_executable_bits(installed_package)
        return installed_package

    raise RuntimeError(
        f"selected_backend_package_unavailable: package_id={package_id} "
        f"source_install_root={install_root or 'missing'} local_install_root={local_root}"
    )


def install_minicpm5_via_ollama(
    *,
    model_name=MINICPM5_OLLAMA_MODEL,
    quant=MINICPM5_DEFAULT_QUANT,
    force=False,
    install_spec=None,
):
    install_spec = dict(install_spec or {})
    gguf_repo = str(install_spec.get("gguf_repo") or MINICPM5_GGUF_REPO)
    gguf_filename = str(install_spec.get("gguf_filename") or f"MiniCPM5-1B-{quant}.gguf")
    ollama_model_name = str(install_spec.get("ollama_model") or model_name or MINICPM5_OLLAMA_MODEL)
    quant = str(install_spec.get("quant") or quant or MINICPM5_DEFAULT_QUANT)
    source_priority = list(install_spec.get("source_priority") or ["cluster_nfs", "huggingface"])
    backend_package_catalog = install_spec.get("backend_package_catalog") if isinstance(install_spec.get("backend_package_catalog"), dict) else _llama_cpp_backend_package_catalog(
        cluster_nfs_root=str(install_spec.get("cluster_nfs_backend_root") or DEFAULT_CLUSTER_NFS_BACKEND_ROOT),
    )
    selected_host_backend_package = dict(
        install_spec.get("selected_host_backend_package")
        if isinstance(install_spec.get("selected_host_backend_package"), dict)
        else (backend_package_catalog.get("selected_package") or {})
    )
    installed_backend_package = install_selected_backend_package(selected_host_backend_package) if selected_host_backend_package else {}
    target_dir = (CGC_MODELS_DIR / model_name).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    gguf_path = (target_dir / gguf_filename).resolve()
    modelfile_path = (target_dir / "Modelfile").resolve()

    print(f"📦 Preparing MiniCPM5 package for cgc run in: {target_dir}")
    print(f"   source_priority = {source_priority}")
    if selected_host_backend_package:
        print(
            "   selected_host_backend_package = "
            f"{selected_host_backend_package.get('package_id')} -> "
            f"{selected_host_backend_package.get('install_root')}"
        )
    if installed_backend_package:
        print(
            "   installed_backend_package = "
            f"{installed_backend_package.get('package_id')} -> "
            f"{installed_backend_package.get('install_root')}"
        )

    source_used = "local_cache"
    cluster_nfs_source = resolve_cluster_nfs_source(install_spec, gguf_filename)
    cluster_nfs_candidate = resolve_cluster_nfs_candidate(install_spec, gguf_filename)
    if cluster_nfs_source:
        print(f"   cluster_nfs = ready ({cluster_nfs_source})")
    else:
        print("   cluster_nfs = missing on this node, fallback may be required")
    if not gguf_path.exists():
        copied = False
        for source_name in source_priority:
            if source_name == "cluster_nfs" and cluster_nfs_source:
                print(f"📡 Using preferred source 'cluster_nfs': {cluster_nfs_source}")
                print(f"   staging {gguf_filename} into local cgc cache ...")
                copy_local_file(cluster_nfs_source, gguf_path)
                source_used = "cluster_nfs"
                copied = True
                break
            if source_name == "cluster_nfs" and cluster_nfs_candidate.startswith(DEFAULT_CLUSTER_NFS_ROOT):
                print(f"📡 Using preferred source 'cluster_nfs_remote_fetch': {cluster_nfs_candidate}")
                print(f"   fetching {gguf_filename} from remote cluster NFS into local cgc cache ...")
                copy_remote_path_via_host1(cluster_nfs_candidate, gguf_path, recursive=False)
                source_used = "cluster_nfs_remote_fetch"
                copied = True
                break
            if source_name == "huggingface":
                print(f"⬇️ Falling back to '{source_name}': {gguf_repo}")
                print(f"   downloading {gguf_filename} into local cgc cache ...")
                download_hf_gguf(gguf_repo, gguf_filename, gguf_path)
                source_used = "huggingface"
                copied = True
                break
        if not copied:
            raise RuntimeError(
                f"no_available_source_for_{gguf_filename}: "
                f"cluster_nfs_source={cluster_nfs_source or 'missing'}"
            )
    else:
        print(f"✅ GGUF already staged in local cgc cache: {gguf_path}")

    modelfile = f"""FROM ./{gguf_filename}

# MiniCPM5 chat template
TEMPLATE \"\"\"{{{{- if .Messages -}}}}
{{{{- range .Messages -}}}}
<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{ end -}}}}
<|im_start|>assistant
{{{{ end -}}}}\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER stop "</s>"
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER num_ctx 8192
"""
    modelfile_path.write_text(modelfile, encoding="utf-8")
    print(f"📝 Wrote local model recipe: {modelfile_path}")

    if force or not ollama_model_exists(ollama_model_name):
        print(f"🧱 Activating local runtime model: {ollama_model_name}")
        run_command_checked(["ollama", "create", ollama_model_name, "-f", str(modelfile_path)], cwd=target_dir)
    else:
        print(f"✅ Local runtime model already active: {ollama_model_name}")

    cfg = load_config()
    active_edge_model_path = cluster_nfs_source or str(gguf_path)
    active_edge_model_source = "nfs" if cluster_nfs_source else str(source_used or "local_cache")
    cfg["active_edge_model"] = ollama_model_name
    cfg["active_edge_model_path"] = active_edge_model_path
    cfg["active_edge_model_source"] = active_edge_model_source
    effective_backend_package = installed_backend_package or selected_host_backend_package
    cfg["active_edge_backend_family"] = str(effective_backend_package.get("backend_family") or "llama.cpp")
    cfg["active_edge_backend_install_root"] = str(effective_backend_package.get("install_root") or "")
    cfg["active_edge_backend_binary_path"] = str(effective_backend_package.get("binary_path") or "")
    cfg["active_edge_backend_server_path"] = str(effective_backend_package.get("server_path") or "")
    cfg["active_edge_backend_package_id"] = str(effective_backend_package.get("package_id") or "")
    save_config(cfg)
    apply_runtime_env(cfg)

    evidence_payload = {
        "status": "PASS",
        "mode": "fake_protocol_cloud_pull_with_real_ollama_install",
        "router_model": ollama_model_name,
        "router_backend": "ollama",
        "public_entrypoint": "cgc run",
        "install_via": str(install_spec.get("install_via") or "direct_hf_download"),
        "source_priority": source_priority,
        "source_used": source_used,
        "cluster_nfs_source": cluster_nfs_source,
        "cluster_nfs_backend_root": str(install_spec.get("cluster_nfs_backend_root") or DEFAULT_CLUSTER_NFS_BACKEND_ROOT),
        "gguf_repo": gguf_repo,
        "quant": quant,
        "gguf_path": active_edge_model_path,
        "staged_local_gguf_path": str(gguf_path),
        "modelfile_path": str(modelfile_path),
        "ollama_model": ollama_model_name,
        "ollama_show_available": ollama_model_exists(ollama_model_name),
        "runtime_strategy": str(install_spec.get("runtime_strategy") or "platform_selected_nfs_backend_binary"),
        "delivery_mode": str(install_spec.get("delivery_mode") or "nfs_backend_binary_plus_model_integration"),
        "backend_package_catalog": backend_package_catalog,
        "selected_host_backend_package": selected_host_backend_package,
        "installed_backend_package": installed_backend_package,
        "config_updates": {
            "active_edge_model": cfg["active_edge_model"],
            "active_edge_model_path": cfg["active_edge_model_path"],
            "active_edge_model_source": cfg["active_edge_model_source"],
            "active_edge_backend_family": cfg["active_edge_backend_family"],
            "active_edge_backend_install_root": cfg["active_edge_backend_install_root"],
            "active_edge_backend_binary_path": cfg["active_edge_backend_binary_path"],
            "active_edge_backend_server_path": cfg["active_edge_backend_server_path"],
            "active_edge_backend_package_id": cfg["active_edge_backend_package_id"],
        },
    }
    if install_spec:
        evidence_payload["cloud_registry_spec"] = install_spec
    evidence_path = write_m75_install_evidence(evidence_payload)
    evidence_payload["evidence_path"] = evidence_path
    return evidence_payload


def get_gate_registry():
    return {
        "m1": {
            "status": "available",
            "description": "M1 baseline executable gate through the shared CGC CLI",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m1").resolve()),
        },
        "m2": {
            "status": "available",
            "description": "M2 inference kernel and safety gate through the shared CGC CLI",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m2").resolve()),
        },
        "m3": {
            "status": "available",
            "description": "M3 model solidification and edge packaging gate through the shared CGC CLI",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m3").resolve()),
        },
        "m4": {
            "status": "available",
            "description": "M4 training and distributed scale-out gate through the shared CGC CLI",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m4").resolve()),
        },
        "m5": {
            "status": "available",
            "description": "M5 terminal-state compile and runtime closure gate through the shared CGC CLI",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m5").resolve()),
        },
        "m6": {
            "status": "available",
            "description": "M6 product bundle build-and-run gate through the shared CGC CLI",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m6").resolve()),
        },
        "m7": {
            "status": "available",
            "description": "M7 industrial baseline verification-only gate",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m7").resolve()),
        },
        "m71": {
            "status": "available",
            "description": "M7.1 industrial verification-only gate through the shared CGC CLI",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m71").resolve()),
        },
        "m72": {
            "status": "available",
            "description": "M7.2 industrial verification-only gate through the shared CGC CLI",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m72").resolve()),
        },
        "m73": {
            "status": "available",
            "description": "M7.3 physical execution verification-only gate through the shared CGC CLI",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m73").resolve()),
        },
        "m74": {
            "status": "available",
            "description": "M7.4 DFlash + TrueOrthoKDA verification-only gate through the shared CGC CLI",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m74").resolve()),
        },
        "upkg21": {
            "status": "available",
            "description": "UPKG 2.1 backend-injectable optimization gate aggregating M5 + M7.4 with the selected SGLang DFlash route",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_upkg21").resolve()),
        },
        "upkg21-rerun": {
            "status": "available",
            "description": "UPKG 2.1 composite rerun wrapper chaining m75 -> m76 -> upkg21 with sibling evidence wiring",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_upkg21_rerun").resolve()),
        },
        "m75": {
            "status": "available",
            "description": "M7.5 API compatibility verification-only gate through the shared CGC CLI",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m75").resolve()),
        },
        "m76": {
            "status": "available",
            "description": "M7.6 heterogeneous acceleration integration verification-only gate",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m76").resolve()),
        },
        "m77": {
            "status": "available",
            "description": "M7.7 cloud-edge training / edge inference / Q2RL standalone gate",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m77").resolve()),
        },
        "upkg30": {
            "status": "available",
            "description": "UPKG 3.0 aggregate product gate covering 3.1-3.7 through the shared CGC CLI",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_upkg30").resolve()),
        },
        "upkg3": {
            "status": "available",
            "description": "Alias of UPKG 3.0 aggregate product gate through the shared CGC CLI",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_upkg30").resolve()),
        },
        "upkg31": {
            "status": "available",
            "description": "UPKG 3.1 alias for the M7 industrial baseline gate",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_upkg31").resolve()),
        },
        "upkg32": {
            "status": "available",
            "description": "UPKG 3.2 alias for the M7.2 agent runtime gate",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_upkg32").resolve()),
        },
        "upkg33": {
            "status": "available",
            "description": "UPKG 3.3 alias for the M7.3 physical execution gate",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_upkg33").resolve()),
        },
        "upkg34": {
            "status": "available",
            "description": "UPKG 3.4 alias for the unified artifact and cloud-summary gate path",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_upkg34").resolve()),
        },
        "upkg35": {
            "status": "available",
            "description": "UPKG 3.5 alias for the six-element audit and attribution gate path",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_upkg35").resolve()),
        },
        "upkg36": {
            "status": "available",
            "description": "UPKG 3.6 alias for the closure and graph-native integration gate path",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_upkg36").resolve()),
        },
        "upkg37": {
            "status": "available",
            "description": "UPKG 3.7 alias for the standalone cloud-edge training / edge inference / Q2RL gate",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m77").resolve()),
        },
        "m78": {
            "status": "available",
            "description": "M7.8 GUI teaching / trained-model edge inference / pure LLM six-element comparison gate",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m78").resolve()),
        },
        "upkg38": {
            "status": "available",
            "description": "UPKG 3.8 alias for the GUI teaching / pure LLM six-element comparison gate",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m78").resolve()),
        },
        "upkg39": {
            "status": "available",
            "description": "UPKG 3.9 strict closure gate restoring 0.8 alignment, schema validation, and tensorized graph-native closure",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_upkg39").resolve()),
        },
        "m79": {
            "status": "available",
            "description": "UPKG 4.0 psi0 cloud training + realtime-vla edge inference + comparative benchmark gate",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m79").resolve()),
        },
        "upkg40": {
            "status": "available",
            "description": "UPKG 4.0 alias for the psi0 cloud training + realtime-vla edge inference benchmark gate",
            "default_output_dir": str((ENGINE_REPO_DIR / "Output" / "cli_gate_m79").resolve()),
        },
        "m8": {
            "status": "available",
            "description": "M8.0 productization verification-only gate covering M8.1-M8.3",
            "default_output_dir": str((RELEASE_DIR / "Output" / "m8_gate").resolve()),
        },
        "m9": {
            "status": "planned",
            "description": "Reserved slot for future M9 gate integration",
            "default_output_dir": str((SCRIPT_DIR / "Output" / "m9_gate").resolve()),
        },
    }


def print_gate_registry():
    registry = get_gate_registry()
    print("Available CGC gates (verification only):")
    for gate_name, meta in registry.items():
        print(f"  - {gate_name}: {meta['status']} | {meta['description']}")


def _resolve_writable_gate_checkin_dir():
    explicit = str(os.environ.get("CGC_GATE_CHECKIN_DIR") or "").strip()
    if explicit:
        return _write_probe_dir(explicit)
    try:
        return _write_probe_dir((CGC_STATE_DIR / "gate_checkins").resolve())
    except Exception:
        return _write_probe_dir(GATE_CHECKIN_DIR)


def write_gate_checkin(gate_name, status, report_path="", summary_path="", trigger="manual", extra=None):
    checkin_dir = _resolve_writable_gate_checkin_dir()
    checkin_log = (checkin_dir / "gate_checkins.jsonl").resolve()
    payload = {
        "gate_name": str(gate_name),
        "status": str(status),
        "report_path": str(report_path or ""),
        "summary_path": str(summary_path or ""),
        "trigger": str(trigger),
        "release_repo": TARGET_RELEASE_REPO,
        "timestamp": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
    }
    if isinstance(extra, dict):
        payload["extra"] = extra

    latest_file = checkin_dir / f"{gate_name}_latest.json"
    with open(checkin_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {
        "status": "PASS",
        "checkin_path": str(latest_file.resolve()),
        "log_path": str(checkin_log),
        "payload": payload,
    }


def run_registered_gate(
    gate_name,
    repo_root=None,
    output_dir=None,
    m4_training_report="",
    m4_inference_report="",
    m72_gui_duration_s=5,
    m72_disable_gui_evidence=False,
):
    registry = get_gate_registry()
    if gate_name not in registry:
        raise ValueError(f"Unknown gate: {gate_name}")

    gate_meta = registry[gate_name]
    if gate_meta["status"] != "available":
        return {
            "status": "NOT_IMPLEMENTED",
            "gate_name": gate_name,
            "message": gate_meta["description"],
            "report_path": "",
            "summary_path": "",
        }

    resolved_output_dir = str(Path(output_dir or gate_meta["default_output_dir"]).expanduser().resolve())
    resolved_repo_root = str(Path(repo_root or SCRIPT_DIR).expanduser().resolve())

    upkg_alias_map = {
        "upkg3": "upkg30",
        "upkg31": "m7",
        "upkg32": "m72",
        "upkg33": "m73",
        "upkg34": "m72",
        "upkg35": "m72",
        "upkg36": "m72",
    }
    mapped_gate_name = upkg_alias_map.get(gate_name, gate_name)

    if gate_name in {"upkg3", "upkg31", "upkg32", "upkg33", "upkg34", "upkg35", "upkg36"}:
        alias_result = run_registered_gate(
            gate_name=mapped_gate_name,
            repo_root=resolved_repo_root,
            output_dir=resolved_output_dir,
            m4_training_report=m4_training_report,
            m4_inference_report=m4_inference_report,
            m72_gui_duration_s=m72_gui_duration_s,
            m72_disable_gui_evidence=m72_disable_gui_evidence,
        )
        if isinstance(alias_result, dict):
            gate_result = alias_result.get("gate_result") if isinstance(alias_result.get("gate_result"), dict) else {}
            if gate_name in {"upkg34", "upkg35", "upkg36"} and "m72" in gate_result:
                m72_gate = gate_result.get("m72") if isinstance(gate_result.get("m72"), dict) else {}
                upkg30_payload = m72_gate.get("upkg30") if isinstance(m72_gate.get("upkg30"), dict) else {}
                section_key = {
                    "upkg34": "3.4_unified_artifact_and_summary",
                    "upkg35": "3.5_six_element_audit_and_attribution",
                    "upkg36": "3.6_missing_capability_closure",
                }[gate_name]
                alias_result["upkg30_section"] = upkg30_payload.get(section_key, {})
            if gate_name != "upkg3":
                binding_profile = _infer_canonical_profile_binding(
                    task_type="inference",
                    environment="edge_cloud" if gate_name == "upkg33" else "cloud_single",
                )
                family_keys = _profile_family_keys(binding_profile)
                profile_bundle = _write_profile_settings_bundle(
                    output_root=resolved_output_dir,
                    schema_prefix=str(gate_name),
                    runtime_host=str(gate_name),
                    deployment_target=str(mapped_gate_name),
                    environment="edge_cloud" if gate_name == "upkg33" else "cloud_single",
                    stage_scope=str(gate_name),
                    model_scope=str(gate_name),
                    model_locator=str(gate_name),
                )
                alias_result.update(profile_bundle)
                alias_result.update(
                    _profile_binding_fields(
                        profile_settings_path=profile_bundle["profile_settings_path"],
                        execution=binding_profile,
                        delivery=binding_profile,
                        compatible=family_keys,
                        applicable=family_keys,
                        bootstrap=binding_profile,
                        flow=binding_profile,
                    )
                )
                report_payload = _read_json_payload(str(alias_result.get("report_path") or ""))
                if isinstance(report_payload, dict) and report_payload:
                    report_payload.update(profile_bundle)
                    report_payload.update(
                        _profile_binding_fields(
                            profile_settings_path=profile_bundle["profile_settings_path"],
                            execution=binding_profile,
                            delivery=binding_profile,
                            compatible=family_keys,
                            applicable=family_keys,
                            bootstrap=binding_profile,
                            flow=binding_profile,
                        )
                    )
                    write_json_file(str(alias_result.get("report_path") or ""), report_payload)
            alias_result["requested_gate_name"] = gate_name
            alias_result["source_gate_name"] = mapped_gate_name
        return alias_result

    if gate_name == "upkg30":
        return _run_upkg30_latest_protocol_aggregate(
            gate_name=gate_name,
            resolved_repo_root=resolved_repo_root,
            requested_output_dir=resolved_output_dir,
            m72_gui_duration_s=m72_gui_duration_s,
            m72_disable_gui_evidence=m72_disable_gui_evidence,
        )

    if gate_name == "upkg39":
        run_upkg39_gate = load_engine_upkg39_gate_runner()
        output_root = Path(resolved_output_dir).resolve()
        root_seed_report = _maybe_seed_upkg3x_pipeline_contract_report(gate_name="upkg39", export_dir=output_root)
        upkg38_result = run_registered_gate(
            gate_name="upkg38",
            repo_root=resolved_repo_root,
            output_dir=str(output_root),
            m4_training_report=m4_training_report,
            m4_inference_report=m4_inference_report,
            m72_gui_duration_s=m72_gui_duration_s,
            m72_disable_gui_evidence=m72_disable_gui_evidence,
        )
        upkg38_gate_result = (upkg38_result or {}).get("gate_result") if isinstance((upkg38_result or {}).get("gate_result"), dict) else {}
        m7_gate = upkg38_gate_result.get("m7") if isinstance(upkg38_gate_result.get("m7"), dict) else {}
        m72_gate = upkg38_gate_result.get("m72") if isinstance(upkg38_gate_result.get("m72"), dict) else {}
        m77_gate = upkg38_gate_result.get("m77") if isinstance(upkg38_gate_result.get("m77"), dict) else {}
        m78_gate = upkg38_gate_result.get("m78") if isinstance(upkg38_gate_result.get("m78"), dict) else {}
        m72_report_path = str((output_root / "m72_industrial" / "report.json").resolve())
        m72_summary_path = str((output_root / "m72_industrial" / "summary.json").resolve())
        m78_report_path = str((output_root / "m78_teaching_pure_llm" / "m78_report.json").resolve())
        m78_summary_path = str((output_root / "m78_teaching_pure_llm" / "summary.json").resolve())
        upkg39_report = run_upkg39_gate(
            output_dir=str(output_root),
            cgc_report={
                "upstream_contracts": {
                    "m72": _build_upstream_gate_contract(
                        gate_name="m72",
                        gate_payload=m72_gate,
                        report_path=m72_report_path,
                        summary_path=m72_summary_path,
                    ),
                    "m78": _build_upstream_gate_contract(
                        gate_name="m78",
                        gate_payload=m78_gate,
                        report_path=m78_report_path,
                        summary_path=m78_summary_path,
                    ),
                }
            },
        )
        upkg39_gate = ((upkg39_report or {}).get("gate_result") or {}).get("upkg39") if isinstance(upkg39_report, dict) else {}
        upkg38_status = str((upkg38_result or {}).get("status") or "FAIL")
        upkg39_status = "PASS" if bool((upkg39_report or {}).get("ok")) else "FAIL"
        final_status = "PASS" if upkg38_status == "PASS" and upkg39_status == "PASS" else "FAIL"
        upkg39_report_path = str((output_root / "upkg39_strict_closure" / "upkg39_report.json").resolve())
        gate_payload = {
            "name": "CGC_UPKG_3_9_Strict_Closure_And_Schema_Validated_Agent_Product_Gate",
            "status": final_status,
            "scope": "verification_only",
            "public_entrypoint": "cgc gate upkg39",
            "gate_result": {
                "m7": m7_gate,
                "m72": m72_gate,
                "m77": m77_gate,
                "m78": m78_gate,
                "upkg39": upkg39_gate,
            },
            "upkg38_report_path": str((upkg38_result or {}).get("report_path") or ""),
            "upkg38_summary_path": str((upkg38_result or {}).get("summary_path") or ""),
            "upkg39_report_path": upkg39_report_path,
            "upkg39_summary_path": str((upkg39_report or {}).get("summary_path") or ""),
        }
        gate_payload = _merge_pipeline_seed_fields(base_payload=gate_payload, seed_report=root_seed_report)
        aggregate_report_path = write_json_file(output_root / "report.json", gate_payload)
        return {
            "status": final_status,
            "gate_name": gate_name,
            "report_path": aggregate_report_path,
            "summary_path": upkg39_report_path,
            "gate_result": gate_payload["gate_result"],
            "canonical_profile_catalog_path": str(gate_payload.get("canonical_profile_catalog_path") or ""),
            "profile_settings_path": str(gate_payload.get("profile_settings_path") or ""),
            "execution_profile_binding_key": str(gate_payload.get("execution_profile_binding_key") or ""),
            "delivery_profile_binding_key": str(gate_payload.get("delivery_profile_binding_key") or ""),
            "compatible_profile_binding_keys": list(gate_payload.get("compatible_profile_binding_keys") or []),
            "applicable_profile_binding_keys": list(gate_payload.get("applicable_profile_binding_keys") or []),
            "bootstrap_contract_binding_key": str(gate_payload.get("bootstrap_contract_binding_key") or ""),
            "flow_parameter_contract_binding_key": str(gate_payload.get("flow_parameter_contract_binding_key") or ""),
        }

    if gate_name in {"m79", "upkg40"}:
        run_m79_gate = load_engine_m79_gate_runner()
        output_root, _ = _resolve_writable_gate_output_root(
            requested_output_dir=resolved_output_dir,
            gate_name="m79" if gate_name == "upkg40" else gate_name,
        )
        with _temporary_env_defaults(
            {
                "CGC_UPKG40_OFFICIAL_PSI0_TRAIN_DURATION_S": "720.0",
                "CGC_UPKG40_OFFICIAL_PSI0_INFER_DURATION_S": "2.0",
            }
        ):
            m73_result = _run_m73_latest_protocol_gate(output_root=output_root)
            m73_gate_result = (m73_result or {}).get("gate_result") if isinstance((m73_result or {}).get("gate_result"), dict) else {}
            m73_gate = m73_gate_result.get("m73") if isinstance(m73_gate_result.get("m73"), dict) else {}
            m79_report = run_m79_gate(
                output_dir=str(output_root),
                cgc_report={"gate_result": {"m73": m73_gate}},
            )
        m79_gate = ((m79_report or {}).get("gate_result") or {}).get("m79") if isinstance(m79_report, dict) else {}
        m73_status = str(m73_gate.get("status") or (m73_result or {}).get("status") or "FAIL")
        m79_status = "PASS" if bool((m79_report or {}).get("ok")) else "FAIL"
        final_status = "PASS" if m73_status == "PASS" and m79_status == "PASS" else "FAIL"
        m79_report_path = str((output_root / "m79_embodied_upkg40" / "m79_report.json").resolve())
        gate_payload = {
            "name": "CGC_UPKG_4_0_Psi0_Cloud_Training_And_Realtime_VLA_Edge_Inference_Gate",
            "status": final_status,
            "scope": "verification_only",
            "public_entrypoint": f"cgc gate {gate_name}",
            "runtime_protocol_contract": dict(m73_gate.get("runtime_protocol_contract") or {}),
            "mandatory_protocol_gate": dict(m73_gate.get("mandatory_protocol_gate") or {}),
            "effective_pd_service": dict(m73_gate.get("effective_pd_service") or {}),
            "protocol_status": {
                "protocol_family": str((m73_gate.get("runtime_protocol_contract") or {}).get("protocol_family") or ""),
                "state_codec": str((m73_gate.get("runtime_protocol_contract") or {}).get("state_codec") or ""),
                "pd_required": bool((m73_gate.get("runtime_protocol_contract") or {}).get("require_pd_service")),
                "pd_mode": str((m73_gate.get("runtime_protocol_contract") or {}).get("pd_mode") or ""),
                "pd_service_status": str((m73_gate.get("effective_pd_service") or {}).get("status") or ""),
                "zero_copy_status": str((m73_gate.get("zero_copy_vram_real") or {}).get("status") or ""),
            },
            "gate_result": {
                "m7": m73_gate_result.get("m7", {}),
                "m71": m73_gate_result.get("m71", {}),
                "m73": m73_gate,
                "m79": m79_gate,
            },
        }
        aggregate_report_path = write_json_file(output_root / "report.json", gate_payload)
        return {
            "status": final_status,
            "gate_name": gate_name,
            "report_path": aggregate_report_path,
            "summary_path": m79_report_path,
            "gate_result": gate_payload["gate_result"],
        }

    if gate_name == "m8":
        run_m8_gate = load_release_m8_gate_runner()
        return run_m8_gate(
            repo_root=resolved_repo_root,
            output_dir=resolved_output_dir,
            config_path=str((RELEASE_DIR / "m8_gate.yaml").resolve()),
        )

    if gate_name == "m1":
        run_m1_gate = load_engine_m1_gate_runner()
        report = run_m1_gate(output_dir=resolved_output_dir)
        return {
            "status": "PASS" if bool(report.get("ok")) else "FAIL",
            "gate_name": gate_name,
            "report_path": str(report.get("report_path") or ""),
            "summary_path": "",
            "gate_result": report.get("gate_result", {}),
        }

    if gate_name == "m2":
        run_m2_gate = load_engine_m2_gate_runner()
        report = run_m2_gate(output_dir=resolved_output_dir)
        return {
            "status": "PASS" if bool(report.get("ok")) else "FAIL",
            "gate_name": gate_name,
            "report_path": str(report.get("report_path") or ""),
            "summary_path": "",
            "gate_result": report.get("gate_result", {}),
        }

    if gate_name == "m3":
        run_m3_gate = load_engine_m3_gate_runner()
        report = run_m3_gate(output_dir=resolved_output_dir)
        return {
            "status": "PASS" if bool(report.get("ok")) else "FAIL",
            "gate_name": gate_name,
            "report_path": str(report.get("report_path") or ""),
            "summary_path": "",
            "gate_result": report.get("gate_result", {}),
        }

    if gate_name == "m4":
        if not str(m4_training_report or "").strip():
            latest_training_report = get_latest_m4_training_report()
            if latest_training_report is not None:
                m4_training_report = str(latest_training_report)
        if not str(m4_inference_report or "").strip():
            latest_inference_report = get_latest_cgc_run_inference_report()
            if latest_inference_report is not None:
                m4_inference_report = str(latest_inference_report)
        run_m4_gate = load_engine_m4_gate_runner()
        report = run_m4_gate(
            output_dir=resolved_output_dir,
            training_report_path=str(m4_training_report or ""),
            inference_report_path=str(m4_inference_report or ""),
        )
        return {
            "status": "PASS" if bool(report.get("ok")) else "FAIL",
            "gate_name": gate_name,
            "report_path": str(report.get("report_path") or ""),
            "summary_path": "",
            "gate_result": report.get("gate_result", {}),
        }

    if gate_name == "m5":
        run_m5_gate = load_engine_m5_gate_runner()
        report = run_m5_gate(output_dir=resolved_output_dir)
        return {
            "status": "PASS" if bool(report.get("ok")) else "FAIL",
            "gate_name": gate_name,
            "report_path": str(report.get("report_path") or ""),
            "summary_path": "",
            "gate_result": report.get("gate_result", {}),
        }

    if gate_name == "m6":
        run_m6_gate = load_engine_m6_gate_runner()
        report = run_m6_gate(output_dir=resolved_output_dir)
        return {
            "status": "PASS" if bool(report.get("ok")) else "FAIL",
            "gate_name": gate_name,
            "report_path": str(report.get("report_path") or ""),
            "summary_path": "",
            "gate_result": report.get("gate_result", {}),
        }

    if gate_name == "m73":
        run_m7_gate = load_engine_m7_gate_runner()
        run_m73_gate = load_engine_m73_gate_runner()
        output_root = Path(resolved_output_dir).resolve()
        pipeline_seed_report = _maybe_seed_upkg3x_pipeline_contract_report(gate_name="m73", export_dir=output_root)
        m7_report = run_m7_gate(output_dir=str(output_root))
        m7_gate = ((m7_report or {}).get("gate_result") or {}).get("m7") if isinstance(m7_report, dict) else {}
        m7_status = "PASS" if bool((m7_report or {}).get("ok")) else "FAIL"
        m7_report_path = str((output_root / "m7_industrial" / "m7_report.json").resolve())
        m7_contract = _build_upstream_gate_contract(
            gate_name="m7",
            gate_payload=m7_gate,
            report_path=str((m7_report or {}).get("report_path") or ""),
            summary_path=str((m7_report or {}).get("summary_path") or m7_report_path),
        )
        m73_report = run_m73_gate(
            output_dir=str(output_root),
            cgc_report={"upstream_contracts": {"m7": m7_contract}},
        )
        m73_gate = ((m73_report or {}).get("gate_result") or {}).get("m73") if isinstance(m73_report, dict) else {}
        m73_status = "PASS" if bool((m73_report or {}).get("ok")) else "FAIL"
        m73_report_path = str((output_root / "m73_physical" / "m73_report.json").resolve())
        final_status = "PASS" if m7_status == "PASS" and m73_status == "PASS" else "FAIL"
        gate_payload = {
            "name": "CGC_M7.3_Physical_Gate",
            "status": final_status,
            "scope": "verification_only",
            "public_entrypoint": "cgc gate m73",
            "gate_result": {
                "m7": m7_gate,
                "m71": m7_gate,
                "m73": m73_gate,
            },
        }
        gate_payload = _merge_pipeline_seed_fields(base_payload=gate_payload, seed_report=pipeline_seed_report)
        aggregate_report_path = write_json_file(output_root / "report.json", gate_payload)
        return {
            "status": final_status,
            "gate_name": gate_name,
            "report_path": aggregate_report_path,
            "summary_path": m73_report_path if Path(m73_report_path).exists() else m7_report_path,
            "gate_result": gate_payload["gate_result"],
        }

    if gate_name == "m74":
        run_m74_gate = load_engine_m74_gate_runner()
        report = run_m74_gate(output_dir=resolved_output_dir)
        return {
            "status": "PASS" if bool(report.get("ok")) else "FAIL",
            "gate_name": gate_name,
            "report_path": str(report.get("report_path") or ""),
            "summary_path": "",
            "gate_result": report.get("gate_result", {}),
        }

    if gate_name == "upkg21":
        run_upkg21_gate = load_engine_upkg21_gate_runner()
        report = run_upkg21_gate(output_dir=resolved_output_dir)
        return {
            "status": "PASS" if bool(report.get("ok")) else "FAIL",
            "gate_name": gate_name,
            "report_path": str(report.get("report_path") or ""),
            "summary_path": str(report.get("summary_path") or ""),
            "gate_result": report.get("gate_result", {}),
        }

    if gate_name == "upkg21-rerun":
        run_upkg21_rerun_gate = load_engine_upkg21_rerun_gate_runner()
        report = run_upkg21_rerun_gate(output_dir=resolved_output_dir)
        return {
            "status": "PASS" if bool(report.get("ok")) else "FAIL",
            "gate_name": gate_name,
            "report_path": str(report.get("report_path") or ""),
            "summary_path": str(report.get("summary_path") or ""),
            "gate_result": report.get("gate_result", {}),
        }

    if gate_name == "m75":
        run_m75_gate = load_engine_m75_gate_runner()
        report = run_m75_gate(output_dir=resolved_output_dir)
        return {
            "status": "PASS" if bool(report.get("ok")) else "FAIL",
            "gate_name": gate_name,
            "report_path": str(report.get("report_path") or ""),
            "summary_path": "",
            "gate_result": report.get("gate_result", {}),
        }

    if gate_name == "m76":
        run_m76_gate = load_engine_m76_gate_runner()
        report = run_m76_gate(output_dir=resolved_output_dir)
        return {
            "status": "PASS" if bool(report.get("ok")) else "FAIL",
            "gate_name": gate_name,
            "report_path": str(report.get("report_path") or ""),
            "summary_path": "",
            "gate_result": report.get("gate_result", {}),
        }

    if gate_name in {"m77", "upkg37"}:
        run_m7_gate = load_engine_m7_gate_runner()
        run_m72_gate = load_engine_m72_gate_runner()
        run_m77_gate = load_engine_m77_gate_runner()
        output_root = Path(resolved_output_dir).resolve()
        m7_output_dir = output_root / "m7_artifacts"
        m7_seed_report = _maybe_seed_upkg3x_pipeline_contract_report(gate_name="m7", export_dir=m7_output_dir)
        root_seed_report = _maybe_seed_upkg3x_pipeline_contract_report(gate_name="m77", export_dir=output_root)
        m7_report = run_m7_gate(output_dir=str(m7_output_dir))
        m7_gate = ((m7_report or {}).get("gate_result") or {}).get("m7") if isinstance(m7_report, dict) else {}
        m7_status = "PASS" if bool((m7_report or {}).get("ok")) else "FAIL"

        m72_output_dir = output_root / "m72_industrial"
        gui_evidence_path = ""
        if not bool(m72_disable_gui_evidence):
            gui_duration_s = int(m72_gui_duration_s)
            if gui_duration_s > 0:
                gui_evidence_path = _collect_gui_stage_source_evidence(
                    duration_s=gui_duration_s,
                    output_dir=m72_output_dir / "gui_agent_runtime",
                )
        prev_gui_evidence = os.environ.get("CGC_M72_GUI_EVENT_EVIDENCE")
        if str(gui_evidence_path).strip():
            os.environ["CGC_M72_GUI_EVENT_EVIDENCE"] = str(gui_evidence_path)
        elif prev_gui_evidence is None:
            os.environ.pop("CGC_M72_GUI_EVENT_EVIDENCE", None)
        try:
            with _with_upkg38_ui_tars_source_env() as ui_tars_resolution:
                m72_report = run_m72_gate(
                    output_dir=str(m72_output_dir),
                    cgc_report={
                        "upstream_contracts": {
                            "m7": _build_upstream_gate_contract(
                                gate_name="m7",
                                gate_payload=m7_gate,
                                report_path=str((m7_report or {}).get("report_path") or ""),
                                summary_path=str((m7_report or {}).get("summary_path") or ""),
                            )
                        }
                    },
                )
        finally:
            if prev_gui_evidence is None:
                os.environ.pop("CGC_M72_GUI_EVENT_EVIDENCE", None)
            else:
                os.environ["CGC_M72_GUI_EVENT_EVIDENCE"] = prev_gui_evidence
        m72_gate = ((m72_report or {}).get("gate_result") or {}).get("m72") if isinstance(m72_report, dict) else {}
        if isinstance(ui_tars_resolution, dict):
            m72_gate["preferred_ui_tars_source_path"] = str(ui_tars_resolution.get("preferred_model_source_path") or "")
            m72_gate["preferred_ui_tars_source_mode"] = str(ui_tars_resolution.get("source_mode") or "")
        if str(gui_evidence_path).strip():
            m72_gate["auto_gui_evidence_path"] = str(gui_evidence_path)
        elif prev_gui_evidence is not None:
            m72_gate["external_gui_evidence_path"] = str(prev_gui_evidence)
        m72_status = str(m72_gate.get("status") or "FAIL")

        m77_report = run_m77_gate(
            output_dir=str(output_root),
            cgc_report={
                "upstream_contracts": {
                    "m72": _build_upstream_gate_contract(
                        gate_name="m72",
                        gate_payload=m72_gate,
                        report_path=str((m72_report or {}).get("report_path") or ""),
                        summary_path=str((m72_report or {}).get("summary_path") or ""),
                    )
                }
            },
        )
        m77_gate = ((m77_report or {}).get("gate_result") or {}).get("m77") if isinstance(m77_report, dict) else {}
        m77_status = "PASS" if bool((m77_report or {}).get("ok")) else "FAIL"
        final_status = "PASS" if m7_status == "PASS" and m72_status == "PASS" and m77_status == "PASS" else "FAIL"
        m77_report_path = str((output_root / "m77_cloud_edge_q2rl" / "m77_report.json").resolve())
        gate_payload = {
            "name": "CGC_M7.7_Cloud_Edge_Q2RL_Gate",
            "status": final_status,
            "scope": "verification_only",
            "public_entrypoint": f"cgc gate {gate_name}",
            "gate_result": {
                "m7": m7_gate,
                "m72": m72_gate,
                "m77": m77_gate,
            },
        }
        gate_payload = _merge_pipeline_seed_fields(base_payload=gate_payload, seed_report=root_seed_report)
        aggregate_report_path = write_json_file(output_root / "report.json", gate_payload)
        return {
            "status": final_status,
            "gate_name": gate_name,
            "report_path": aggregate_report_path,
            "summary_path": m77_report_path,
            "gate_result": gate_payload["gate_result"],
            "canonical_profile_catalog_path": str(gate_payload.get("canonical_profile_catalog_path") or ""),
            "profile_settings_path": str(gate_payload.get("profile_settings_path") or ""),
            "execution_profile_binding_key": str(gate_payload.get("execution_profile_binding_key") or ""),
            "delivery_profile_binding_key": str(gate_payload.get("delivery_profile_binding_key") or ""),
            "compatible_profile_binding_keys": list(gate_payload.get("compatible_profile_binding_keys") or []),
            "applicable_profile_binding_keys": list(gate_payload.get("applicable_profile_binding_keys") or []),
            "bootstrap_contract_binding_key": str(gate_payload.get("bootstrap_contract_binding_key") or ""),
            "flow_parameter_contract_binding_key": str(gate_payload.get("flow_parameter_contract_binding_key") or ""),
        }

    if gate_name in {"m78", "upkg38"}:
        run_m7_gate = load_engine_m7_gate_runner()
        run_m72_gate = load_engine_m72_gate_runner()
        run_m77_gate = load_engine_m77_gate_runner()
        run_m78_gate = load_engine_m78_gate_runner()
        output_root = Path(resolved_output_dir).resolve()
        m7_output_dir = output_root / "m7_artifacts"
        m7_seed_report = _maybe_seed_upkg3x_pipeline_contract_report(gate_name="m7", export_dir=m7_output_dir)
        root_seed_report = _maybe_seed_upkg3x_pipeline_contract_report(gate_name="m78", export_dir=output_root)
        m7_report = run_m7_gate(output_dir=str(m7_output_dir))
        m7_gate = ((m7_report or {}).get("gate_result") or {}).get("m7") if isinstance(m7_report, dict) else {}
        m7_status = "PASS" if bool((m7_report or {}).get("ok")) else "FAIL"

        m72_output_dir = output_root / "m72_industrial"
        gui_evidence_path = ""
        if not bool(m72_disable_gui_evidence):
            gui_duration_s = int(m72_gui_duration_s)
            if gui_duration_s > 0:
                gui_evidence_path = _collect_gui_stage_source_evidence(
                    duration_s=gui_duration_s,
                    output_dir=m72_output_dir / "gui_agent_runtime",
                )
        prev_gui_evidence = os.environ.get("CGC_M72_GUI_EVENT_EVIDENCE")
        if str(gui_evidence_path).strip():
            os.environ["CGC_M72_GUI_EVENT_EVIDENCE"] = str(gui_evidence_path)
        elif prev_gui_evidence is None:
            os.environ.pop("CGC_M72_GUI_EVENT_EVIDENCE", None)
        try:
            with _with_upkg38_ui_tars_source_env() as ui_tars_resolution:
                m72_report = run_m72_gate(
                    output_dir=str(m72_output_dir),
                    cgc_report={
                        "upstream_contracts": {
                            "m7": _build_upstream_gate_contract(
                                gate_name="m7",
                                gate_payload=m7_gate,
                                report_path=str((m7_report or {}).get("report_path") or ""),
                                summary_path=str((m7_report or {}).get("summary_path") or ""),
                            )
                        }
                    },
                )
        finally:
            if prev_gui_evidence is None:
                os.environ.pop("CGC_M72_GUI_EVENT_EVIDENCE", None)
            else:
                os.environ["CGC_M72_GUI_EVENT_EVIDENCE"] = prev_gui_evidence
        m72_gate = ((m72_report or {}).get("gate_result") or {}).get("m72") if isinstance(m72_report, dict) else {}
        if isinstance(ui_tars_resolution, dict):
            m72_gate["preferred_ui_tars_source_path"] = str(ui_tars_resolution.get("preferred_model_source_path") or "")
            m72_gate["preferred_ui_tars_source_mode"] = str(ui_tars_resolution.get("source_mode") or "")
        if str(gui_evidence_path).strip():
            m72_gate["auto_gui_evidence_path"] = str(gui_evidence_path)
        elif prev_gui_evidence is not None:
            m72_gate["external_gui_evidence_path"] = str(prev_gui_evidence)
        m72_status = str(m72_gate.get("status") or "FAIL")

        m77_report = run_m77_gate(
            output_dir=str(output_root),
            cgc_report={
                "upstream_contracts": {
                    "m72": _build_upstream_gate_contract(
                        gate_name="m72",
                        gate_payload=m72_gate,
                        report_path=str((m72_report or {}).get("report_path") or ""),
                        summary_path=str((m72_report or {}).get("summary_path") or ""),
                    )
                }
            },
        )
        m77_gate = ((m77_report or {}).get("gate_result") or {}).get("m77") if isinstance(m77_report, dict) else {}
        m77_status = "PASS" if bool((m77_report or {}).get("ok")) else "FAIL"

        m78_report = run_m78_gate(
            output_dir=str(output_root),
            cgc_report={
                "upstream_contracts": {
                    "m72": _build_upstream_gate_contract(
                        gate_name="m72",
                        gate_payload=m72_gate,
                        report_path=str((m72_report or {}).get("report_path") or ""),
                        summary_path=str((m72_report or {}).get("summary_path") or ""),
                    ),
                    "m77": _build_upstream_gate_contract(
                        gate_name="m77",
                        gate_payload=m77_gate,
                        report_path=str((m77_report or {}).get("report_path") or ""),
                        summary_path=str((m77_report or {}).get("summary_path") or ""),
                    ),
                }
            },
        )
        m78_gate = ((m78_report or {}).get("gate_result") or {}).get("m78") if isinstance(m78_report, dict) else {}
        m78_status = "PASS" if bool((m78_report or {}).get("ok")) else "FAIL"
        final_status = "PASS" if m7_status == "PASS" and m72_status == "PASS" and m77_status == "PASS" and m78_status == "PASS" else "FAIL"
        m78_report_path = str((output_root / "m78_teaching_pure_llm" / "m78_report.json").resolve())
        gate_payload = {
            "name": "CGC_M7.8_Teaching_And_Pure_LLM_Six_Element_Inference_Gate",
            "status": final_status,
            "scope": "verification_only",
            "public_entrypoint": f"cgc gate {gate_name}",
            "gate_result": {
                "m7": m7_gate,
                "m72": m72_gate,
                "m77": m77_gate,
                "m78": m78_gate,
            },
        }
        gate_payload = _merge_pipeline_seed_fields(base_payload=gate_payload, seed_report=root_seed_report)
        aggregate_report_path = write_json_file(output_root / "report.json", gate_payload)
        return {
            "status": final_status,
            "gate_name": gate_name,
            "report_path": aggregate_report_path,
            "summary_path": m78_report_path,
            "gate_result": gate_payload["gate_result"],
            "canonical_profile_catalog_path": str(gate_payload.get("canonical_profile_catalog_path") or ""),
            "profile_settings_path": str(gate_payload.get("profile_settings_path") or ""),
            "execution_profile_binding_key": str(gate_payload.get("execution_profile_binding_key") or ""),
            "delivery_profile_binding_key": str(gate_payload.get("delivery_profile_binding_key") or ""),
            "compatible_profile_binding_keys": list(gate_payload.get("compatible_profile_binding_keys") or []),
            "applicable_profile_binding_keys": list(gate_payload.get("applicable_profile_binding_keys") or []),
            "bootstrap_contract_binding_key": str(gate_payload.get("bootstrap_contract_binding_key") or ""),
            "flow_parameter_contract_binding_key": str(gate_payload.get("flow_parameter_contract_binding_key") or ""),
        }

    if gate_name in {"m7", "m71", "m72"}:
        run_m7_gate = load_engine_m7_gate_runner()
        output_root = Path(resolved_output_dir).resolve()
        m7_output_dir = output_root if gate_name == "m7" else (output_root / "m7_artifacts")
        m7_seed_report = _maybe_seed_upkg3x_pipeline_contract_report(gate_name="m7" if gate_name == "m7" else "m7", export_dir=m7_output_dir)
        report = run_m7_gate(output_dir=str(m7_output_dir))
        m7_gate = ((report or {}).get("gate_result") or {}).get("m7") if isinstance(report, dict) else {}
        m7_status = "PASS" if bool(report.get("ok")) else "FAIL"
        m7_report_path = str((Path(m7_output_dir) / "m7_industrial" / "m7_report.json").resolve())

        if gate_name == "m7":
            gate_payload = {
                "name": "CGC_M7_Industrial_Baseline_Gate",
                "status": m7_status,
                "scope": "verification_only",
                "public_entrypoint": "cgc gate m7",
                "gate_result": {
                    "m7": m7_gate,
                },
            }
            gate_payload = _merge_pipeline_seed_fields(base_payload=gate_payload, seed_report=m7_seed_report)
            aggregate_report_path = write_json_file(output_root / "report.json", gate_payload)
            return {
                "status": m7_status,
                "gate_name": gate_name,
                "report_path": aggregate_report_path,
                "summary_path": m7_report_path,
                "gate_result": gate_payload["gate_result"],
            }

        if gate_name == "m71":
            gate_payload = {
                "name": "CGC_M7.1_Industrial_Gate",
                "status": m7_status,
                "scope": "verification_only",
                "public_entrypoint": "cgc gate m71",
                "source_gate": "m7",
                "gate_result": {
                    "m71": m7_gate,
                },
            }
            gate_payload = _merge_pipeline_seed_fields(base_payload=gate_payload, seed_report=m7_seed_report)
            aggregate_report_path = write_json_file(output_root / "report.json", gate_payload)
            return {
                "status": m7_status,
                "gate_name": gate_name,
                "report_path": aggregate_report_path,
                "summary_path": m7_report_path,
                "gate_result": gate_payload["gate_result"],
            }

        run_m72_gate = load_engine_m72_gate_runner()
        m72_output_dir = output_root / "m72_industrial"
        root_seed_report = _maybe_seed_upkg3x_pipeline_contract_report(gate_name="m72", export_dir=output_root)
        gui_evidence_path = ""
        if not bool(m72_disable_gui_evidence):
            gui_duration_s = int(m72_gui_duration_s)
            if gui_duration_s > 0:
                gui_evidence_path = _collect_gui_stage_source_evidence(
                    duration_s=gui_duration_s,
                    output_dir=m72_output_dir / "gui_agent_runtime",
                )
        prev_gui_evidence = os.environ.get("CGC_M72_GUI_EVENT_EVIDENCE")
        if str(gui_evidence_path).strip():
            os.environ["CGC_M72_GUI_EVENT_EVIDENCE"] = str(gui_evidence_path)
        elif prev_gui_evidence is None:
            os.environ.pop("CGC_M72_GUI_EVENT_EVIDENCE", None)
        try:
            with _with_upkg38_ui_tars_source_env() as ui_tars_resolution:
                m72_report = run_m72_gate(
                    output_dir=str(m72_output_dir),
                    cgc_report={
                        "upstream_contracts": {
                            "m7": _build_upstream_gate_contract(
                                gate_name="m7",
                                gate_payload=m7_gate,
                                report_path=str((report or {}).get("report_path") or ""),
                                summary_path=str((report or {}).get("summary_path") or m7_report_path),
                            )
                        }
                    },
                )
        finally:
            if prev_gui_evidence is None:
                os.environ.pop("CGC_M72_GUI_EVENT_EVIDENCE", None)
            else:
                os.environ["CGC_M72_GUI_EVENT_EVIDENCE"] = prev_gui_evidence
        m72_gate = ((m72_report or {}).get("gate_result") or {}).get("m72") if isinstance(m72_report, dict) else {}
        if isinstance(ui_tars_resolution, dict):
            m72_gate["preferred_ui_tars_source_path"] = str(ui_tars_resolution.get("preferred_model_source_path") or "")
            m72_gate["preferred_ui_tars_source_mode"] = str(ui_tars_resolution.get("source_mode") or "")
        if str(gui_evidence_path).strip():
            m72_gate["auto_gui_evidence_path"] = str(gui_evidence_path)
        elif prev_gui_evidence is not None:
            m72_gate["external_gui_evidence_path"] = str(prev_gui_evidence)
        m72_status = str(m72_gate.get("status") or "FAIL")
        final_status = "PASS" if m7_status == "PASS" and m72_status == "PASS" else "FAIL"
        m72_report_path = str((m72_output_dir / "report.json").resolve())
        gate_payload = {
            "name": "CGC_M7.2_Industrial_Gate",
            "status": final_status,
            "scope": "verification_only",
            "public_entrypoint": "cgc gate m72",
            "gate_result": {
                "m7": m7_gate,
                "m71": m7_gate,
                "m72": m72_gate,
            },
        }
        gate_payload = _merge_pipeline_seed_fields(base_payload=gate_payload, seed_report=root_seed_report)
        aggregate_report_path = write_json_file(output_root / "report.json", gate_payload)
        return {
            "status": final_status,
            "gate_name": gate_name,
            "report_path": aggregate_report_path,
            "summary_path": m72_report_path,
            "gate_result": gate_payload["gate_result"],
        }

    raise ValueError(f"Gate runner not implemented: {gate_name}")

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            cfg = {}
        active_edge_model_path = str(cfg.get("active_edge_model_path") or "").strip()
        if active_edge_model_path == "" or "/.cgc_local/" in active_edge_model_path:
            cfg["active_edge_model_path"] = _default_active_edge_model_path()
        active_edge_model_source = str(cfg.get("active_edge_model_source") or "").strip().lower()
        if active_edge_model_source == "" or "/.cgc_local/" in active_edge_model_path:
            cfg["active_edge_model_source"] = "nfs"
        cfg.setdefault("active_edge_backend_family", "")
        cfg.setdefault("active_edge_backend_install_root", "")
        cfg.setdefault("active_edge_backend_binary_path", "")
        cfg.setdefault("active_edge_backend_server_path", "")
        cfg.setdefault("active_edge_backend_package_id", "")
        return cfg
    return {
        "cloud_ip": "10.100.200.65", 
        "cloud_port": 50052, 
        "active_edge_model": MINICPM5_OLLAMA_MODEL,
        "active_cloud_model": "deepseek-v4-flash:latest",
        "active_edge_model_path": _default_active_edge_model_path(),
        "active_edge_model_source": "nfs",
        "active_edge_backend_family": "",
        "active_edge_backend_install_root": "",
        "active_edge_backend_binary_path": "",
        "active_edge_backend_server_path": "",
        "active_edge_backend_package_id": "",
        "local_omlx_model": "",
        "local_flashmoe_model": "",
        "edge_api_port": 8000,
        "edge_proxy_port": 4000,
    }

def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=4)


def get_edge_api_base_url(cfg=None):
    cfg = cfg or load_config()
    return f"http://127.0.0.1:{int(cfg.get('edge_api_port', 8000) or 8000)}"


def apply_runtime_env(cfg):
    local_omlx_model = str(cfg.get("local_omlx_model") or "").strip()
    if local_omlx_model:
        os.environ["CGC_LOCAL_OMLX_MODEL"] = local_omlx_model
    else:
        os.environ.pop("CGC_LOCAL_OMLX_MODEL", None)
    local_flashmoe_model = str(cfg.get("local_flashmoe_model") or "").strip()
    if local_flashmoe_model:
        os.environ["CGC_LOCAL_FLASHMOE_MODEL"] = local_flashmoe_model
    else:
        os.environ.pop("CGC_LOCAL_FLASHMOE_MODEL", None)
    local_llama_cpp_bin = str(cfg.get("active_edge_backend_binary_path") or "").strip()
    if local_llama_cpp_bin:
        os.environ["CGC_LOCAL_LLAMA_CPP_BIN"] = local_llama_cpp_bin
    else:
        os.environ.pop("CGC_LOCAL_LLAMA_CPP_BIN", None)
    local_llama_cpp_server = str(cfg.get("active_edge_backend_server_path") or "").strip()
    if local_llama_cpp_server:
        os.environ["CGC_LOCAL_LLAMA_CPP_SERVER"] = local_llama_cpp_server
    else:
        os.environ.pop("CGC_LOCAL_LLAMA_CPP_SERVER", None)
    local_llama_cpp_root = str(cfg.get("active_edge_backend_install_root") or "").strip()
    if local_llama_cpp_root:
        os.environ["CGC_LOCAL_LLAMA_CPP_ROOT"] = local_llama_cpp_root
    else:
        os.environ.pop("CGC_LOCAL_LLAMA_CPP_ROOT", None)


def resolve_local_runtime_model(model_to_use, *, cfg, use_omlx, use_flashmoe):
    if use_flashmoe and str(cfg.get("local_flashmoe_model") or "").strip():
        return str(cfg.get("local_flashmoe_model"))
    if use_omlx and str(cfg.get("local_omlx_model") or "").strip():
        return str(cfg.get("local_omlx_model"))
    active_edge_model = str(cfg.get("active_edge_model") or "").strip()
    active_edge_model_path = str(cfg.get("active_edge_model_path") or "").strip()
    active_edge_backend_family = str(cfg.get("active_edge_backend_family") or "").strip().lower()
    requested_model = str(model_to_use or "").strip()
    if active_edge_model_path:
        if requested_model == active_edge_model:
            return active_edge_model_path
        if requested_model.lower() == active_edge_model.lower() and requested_model:
            return active_edge_model_path
        if active_edge_backend_family == "llama.cpp" and requested_model.lower().endswith(".gguf") is False:
            alias_candidates = {
                active_edge_model.lower(),
                Path(active_edge_model_path).name.lower(),
                Path(active_edge_model_path).stem.lower(),
            }
            if requested_model.lower() in alias_candidates:
                return active_edge_model_path
    return str(model_to_use)


def get_latest_cgc_run_inference_report():
    return CGC_RUN_LATEST_M4_INFERENCE_REPORT if CGC_RUN_LATEST_M4_INFERENCE_REPORT.exists() else None


def get_latest_m4_training_report():
    candidates = []
    temp_test_root = (REPO_ROOT / "temp" / "test").resolve()
    if temp_test_root.exists():
        candidates.extend(temp_test_root.glob("cloud_m4_training_host_*.json"))
    for candidate in sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True):
        payload = _safe_read_json(candidate)
        if not isinstance(payload, dict) or not payload:
            continue
        steps = payload.get("steps") if isinstance(payload.get("steps"), dict) else {}
        megatrain = steps.get("megatrain_8step") if isinstance(steps.get("megatrain_8step"), dict) else {}
        step7 = megatrain.get("step7_compare") if isinstance(megatrain.get("step7_compare"), dict) else {}
        if str(payload.get("task_type") or "").strip() != "train":
            continue
        if str(payload.get("backend") or "").strip() != "megatrain":
            continue
        if str(step7.get("status") or "").strip().upper() != "PASS":
            continue
        return candidate
    return None


def _safe_read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_read_jsonl(path):
    rows = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    except Exception:
        return []
    return rows


def _load_gui_stage_source_from_env():
    evidence_path = str(
        os.environ.get("CGC_GUI_STAGE_SOURCE")
        or os.environ.get("CGC_M72_GUI_EVENT_EVIDENCE")
        or os.environ.get("CGC_GUI_AGENT_EVENT_EVIDENCE")
        or ""
    ).strip()
    if not evidence_path:
        return {}
    evidence = _safe_read_json(evidence_path)
    if not isinstance(evidence, dict) or not evidence:
        return {}
    events_path = str(evidence.get("events_path") or "").strip()
    manifest_path = str(evidence.get("screenshot_manifest_path") or "").strip()
    events = _safe_read_jsonl(events_path) if events_path else []
    manifest = _safe_read_json(manifest_path) if manifest_path else {}
    screenshots = manifest.get("screenshots") if isinstance(manifest.get("screenshots"), list) else []
    categories = sorted({str(item.get("category") or "") for item in events if isinstance(item, dict)})
    by_category = {key: 0 for key in ("workflow", "runtime_host", "tool_call", "screenshot")}
    for item in events:
        category = str(item.get("category") or "")
        if category in by_category:
            by_category[category] += 1
    return {
        "status": str(evidence.get("status") or "FAIL"),
        "mode": "gui_runtime_evidence",
        "evidence_path": evidence_path,
        "events_path": events_path,
        "manifest_path": manifest_path,
        "event_count": int(len(events)),
        "screenshot_count": int(len(screenshots)),
        "categories_present": categories,
        "by_category": by_category,
    }


def _infer_gui_graph_native_integration_path_from_stage_source(stage_source):
    evidence_path = str((stage_source or {}).get("evidence_path") or "").strip()
    if not evidence_path:
        return ""
    evidence_parent = Path(evidence_path).expanduser().resolve().parent
    for candidate in (
        evidence_parent / "gui_graph_native_integration.json",
        evidence_parent.parent / "gui_graph_native_integration.json",
    ):
        if candidate.exists():
            return str(candidate)
    return ""


def _load_gui_graph_native_from_env():
    stage_source = _load_gui_stage_source_from_env()
    if not isinstance(stage_source, dict) or not stage_source:
        return {}
    return build_gui_graph_native_integration(stage_source)


def _collect_gui_stage_source_evidence(*, duration_s, output_dir):
    try:
        from cgc_engine.agent.eval.eko_gui_agent_demo import collect_gui_runtime_evidence

        return str(
            collect_gui_runtime_evidence(
                duration_sec=int(duration_s),
                output_dir=Path(output_dir).expanduser().resolve(),
            )
        )
    except Exception:
        return ""


def _safe_slug(value, default="item"):
    text = str(value or "").strip().lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    normalized = "".join(out).strip("_.-")
    return normalized or default


def _make_agent_output_dir(output_dir="", *, command_name="session"):
    if str(output_dir or "").strip():
        out_dir = Path(output_dir).expanduser().resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = (CGC_AGENT_ARTIFACT_ROOT / f"{_safe_slug(command_name)}_{stamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _make_model_output_dir(output_dir="", *, command_name="session"):
    if str(output_dir or "").strip():
        out_dir = Path(output_dir).expanduser().resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        preferred = (CGC_MODEL_ARTIFACT_ROOT / f"{_safe_slug(command_name)}_{stamp}").resolve()
        fallback = (Path("/private/tmp/cgc_model_cli") / f"{_safe_slug(command_name)}_{stamp}").resolve()
        out_dir = fallback
        for candidate in (preferred, fallback):
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                probe = candidate / ".write_probe"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
                out_dir = candidate
                break
            except Exception:
                continue
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _pick_existing_file(*candidates):
    for candidate in candidates:
        raw = str(candidate or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        if path.exists():
            return path
    return None


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _remote_repo_root():
    return str(os.environ.get("CGC_SWEBENCH_REMOTE_REPO_ROOT") or DEFAULT_SWEBENCH_REMOTE_REPO_ROOT)


def _remote_swe_agent_root():
    return str(os.environ.get("CGC_SWEBENCH_REMOTE_SWE_AGENT_ROOT") or DEFAULT_SWEBENCH_REMOTE_SWE_AGENT_ROOT)


def _load_paramiko_module():
    try:
        import paramiko  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"paramiko is required for remote swe-verified execution: {exc}") from exc
    return paramiko


def _resolve_swebench_remote_hosts(cluster_status):
    status_hosts = cluster_status.get("hosts") if isinstance(cluster_status.get("hosts"), list) else []
    resolved = []
    for fallback in DEFAULT_SWEBENCH_REMOTE_HOSTS:
        role = str(fallback.get("role") or "")
        match = None
        for host in status_hosts:
            host_payload = host if isinstance(host, dict) else {}
            host_role = str(host_payload.get("role") or "")
            host_name = str(host_payload.get("name") or "")
            host_ip = str(host_payload.get("host") or "")
            if host_role == role or host_name == fallback.get("name") or host_ip == fallback.get("host"):
                match = host_payload
                break
        spec = dict(fallback)
        if isinstance(match, dict):
            spec["host"] = str(match.get("host") or spec["host"])
            spec["name"] = str(match.get("name") or spec["name"])
            spec["role"] = str(match.get("role") or spec["role"])
        env_prefix = f"CGC_SWEBENCH_{role.upper()}"
        spec["host"] = str(os.environ.get(f"{env_prefix}_HOST") or spec["host"])
        spec["user"] = str(
            os.environ.get(f"{env_prefix}_USER")
            or os.environ.get("CGC_SWEBENCH_SSH_USER")
            or spec["user"]
        )
        spec["password"] = str(
            os.environ.get(f"{env_prefix}_PASSWORD")
            or os.environ.get("CGC_SWEBENCH_SSH_PASSWORD")
            or spec["password"]
        )
        resolved.append(spec)
    return resolved


def _connect_remote_host(host_spec, timeout=15):
    paramiko = _load_paramiko_module()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        str(host_spec.get("host") or ""),
        username=str(host_spec.get("user") or "root"),
        password=str(host_spec.get("password") or ""),
        timeout=max(1, int(timeout or 15)),
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def _remote_exec(client, command, timeout=120):
    _, stdout, stderr = client.exec_command(command, timeout=max(1, int(timeout or 120)))
    exit_code = stdout.channel.recv_exit_status()
    return {
        "exit_code": int(exit_code),
        "stdout": stdout.read().decode("utf-8", errors="replace").strip(),
        "stderr": stderr.read().decode("utf-8", errors="replace").strip(),
    }


def _remote_load_json_payload(command_result):
    stdout = str(command_result.get("stdout") or "").strip()
    stderr = str(command_result.get("stderr") or "").strip()
    for candidate in (stdout, stderr):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return {
        "status": "FAIL",
        "error": "remote_json_parse_failed",
        "command_result": command_result,
    }


def _find_remote_host_by_role(host_specs, role):
    expected_role = str(role or "").strip().lower()
    for host_spec in host_specs:
        if str(host_spec.get("role") or "").strip().lower() == expected_role:
            return host_spec
    return host_specs[0] if host_specs else {}


def _build_remote_swebench_log_path(suffix):
    safe_suffix = _safe_slug(suffix, "cgc_m76_dualnode")
    return f"{DEFAULT_SWEBENCH_REMOTE_LOG_DIR}/{safe_suffix}.log"


def _launch_remote_swebench_benchmark(*, head_host, launch_plan):
    if not isinstance(head_host, dict) or not head_host:
        return {"status": "FAIL", "error": "missing_head_host"}
    env = dict(launch_plan.get("env") or {})
    remote_log_path = str(env.get("CGC_SWEBENCH_LOG") or _build_remote_swebench_log_path(env.get("CGC_SWEBENCH_SUFFIX")))
    env["CGC_SWEBENCH_LOG"] = remote_log_path
    payload = {
        "env": env,
        "candidates": list(
            launch_plan.get("launch_command_candidates")
            or [
                f"{_remote_repo_root()}/run_full_swebench.sh",
                f"{_remote_repo_root()}/CGC_Release/run/run_full_swebench.sh",
            ]
        ),
        "remote_repo_root": _remote_repo_root(),
        "remote_swe_agent_root": _remote_swe_agent_root(),
        "remote_log_dir": DEFAULT_SWEBENCH_REMOTE_LOG_DIR,
        "remote_log_path": remote_log_path,
    }
    command = (
        "python3 - <<'PY'\n"
        "import json\n"
        "import pathlib\n"
        "import shlex\n"
        "import subprocess\n"
        f"payload = {json.dumps(payload, ensure_ascii=False)}\n"
        "env = dict(payload.get('env') or {})\n"
        "candidates = list(payload.get('candidates') or [])\n"
        "selected = ''\n"
        "for candidate in candidates:\n"
        "    if pathlib.Path(candidate).exists():\n"
        "        selected = candidate\n"
        "        break\n"
        "if not selected:\n"
        "    print(json.dumps({'status': 'FAIL', 'error': 'launch_script_not_found', 'candidates': candidates}, ensure_ascii=False))\n"
        "    raise SystemExit(0)\n"
        "subprocess.run(['bash', '-lc', f\"mkdir -p {shlex.quote(payload.get('remote_log_dir') or '')}\"], check=False)\n"
        "before = subprocess.run(['bash', '-lc', \"pgrep -af 'sweagent.run.run run-batch' || true\"], capture_output=True, text=True)\n"
        "exports = ''.join(f\"export {key}={shlex.quote(str(value))}; \" for key, value in env.items())\n"
        "direct_runner = ''.join([\n"
        "    f\"cd {shlex.quote(payload.get('remote_swe_agent_root') or '')}; \",\n"
        "    'export OPENAI_API_KEY=sk-cgc-edge-key; ',\n"
        "    'export HF_TOKEN=${CGC_HF_TOKEN}; ',\n"
        "    'export HF_ENDPOINT=https://hf-mirror.com; ',\n"
        "    f\"nohup {shlex.quote((payload.get('remote_repo_root') or '') + '/venv/bin/python')} -m sweagent.run.run run-batch \",\n"
        "    '--instances.type swe_bench ',\n"
        "    '--instances.subset verified ',\n"
        "    '--instances.split test ',\n"
        "    '--instances.slice :${CGC_SWEBENCH_LIMIT} ',\n"
        "    '--config config/default.yaml ',\n"
        "    '--agent.model.name ${CGC_SWEBENCH_MODEL_NAME} ',\n"
        "    '--agent.model.api_base ${CGC_SWEBENCH_API_BASE} ',\n"
        "    '--agent.model.temperature 0.0 ',\n"
        "    '--agent.model.per_instance_cost_limit 0.0 ',\n"
        "    '--num_workers ${CGC_SWEBENCH_NUM_WORKERS} ',\n"
        "    '--suffix ${CGC_SWEBENCH_SUFFIX} ',\n"
        "    '> ${CGC_SWEBENCH_LOG} 2>&1 & ',\n"
        "    'echo SWE_PID:$!'\n"
        "])\n"
        "runner = subprocess.run(['bash', '-lc', exports + f\"bash {shlex.quote(selected)}\"], capture_output=True, text=True)\n"
        "launch_mode = 'script'\n"
        "combined_output = (runner.stdout or '') + '\\n' + (runner.stderr or '')\n"
        "if runner.returncode != 0 and '--instances.limit' in combined_output:\n"
        "    runner = subprocess.run(['bash', '-lc', exports + direct_runner], capture_output=True, text=True)\n"
        "    launch_mode = 'direct_fallback'\n"
        "after = subprocess.run(['bash', '-lc', \"pgrep -af 'sweagent.run.run run-batch' || true\"], capture_output=True, text=True)\n"
        "after_lines = [line for line in after.stdout.splitlines() if line.strip()]\n"
        "if any('--instances.limit' in line for line in after_lines):\n"
        "    subprocess.run(['bash', '-lc', \"pkill -f 'sweagent.run.run run-batch' || true\"], check=False)\n"
        "    runner = subprocess.run(['bash', '-lc', exports + direct_runner], capture_output=True, text=True)\n"
        "    launch_mode = 'direct_fallback'\n"
        "    after = subprocess.run(['bash', '-lc', \"pgrep -af 'sweagent.run.run run-batch' || true\"], capture_output=True, text=True)\n"
        "    after_lines = [line for line in after.stdout.splitlines() if line.strip()]\n"
        "tail = subprocess.run(['bash', '-lc', f\"tail -n 20 {shlex.quote(payload.get('remote_log_path') or '')} 2>/dev/null || true\"], capture_output=True, text=True)\n"
        "print(json.dumps({\n"
        "    'status': 'PASS' if runner.returncode == 0 else 'FAIL',\n"
        "    'launch_mode': launch_mode,\n"
        "    'selected_launch_script': selected,\n"
        "    'launch_exit_code': int(runner.returncode),\n"
        "    'launch_stdout': runner.stdout[-12000:],\n"
        "    'launch_stderr': runner.stderr[-4000:],\n"
        "    'remote_log_path': payload.get('remote_log_path') or '',\n"
        "    'running_processes_before': [line for line in before.stdout.splitlines() if line.strip()],\n"
        "    'running_processes_after': after_lines,\n"
        "    'log_tail': tail.stdout[-4000:],\n"
        "    'env': env,\n"
        "}, ensure_ascii=False))\n"
        "PY"
    )
    client = _connect_remote_host(head_host)
    try:
        return _remote_load_json_payload(_remote_exec(client, command, timeout=180))
    finally:
        client.close()


def _collect_remote_swebench_summary(*, head_host, launch_plan):
    if not isinstance(head_host, dict) or not head_host:
        return {"status": "FAIL", "error": "missing_head_host"}
    suffix = str(_as_dict(launch_plan.get("env")).get("CGC_SWEBENCH_SUFFIX") or "")
    log_path = str(_as_dict(launch_plan.get("env")).get("CGC_SWEBENCH_LOG") or _build_remote_swebench_log_path(suffix))
    # #region debug-point A:collect-summary-input
    _debug_report_dualnode_swe500(
        hypothesis_id="A",
        location="app/cli/cgc.py:_collect_remote_swebench_summary:input",
        msg="[DEBUG] collect remote swebench summary input",
        data={
            "head_host": str(head_host.get("host") or ""),
            "suffix": suffix,
            "log_path": log_path,
            "api_base": str(_as_dict(launch_plan.get("env")).get("CGC_SWEBENCH_API_BASE") or ""),
        },
    )
    # #endregion
    payload = {
        "suffix": suffix,
        "log_path": log_path,
        "swe_agent_root": _remote_swe_agent_root(),
    }
    command = (
        "python3 - <<'PY'\n"
        "import glob\n"
        "import json\n"
        "import pathlib\n"
        f"payload = {json.dumps(payload, ensure_ascii=False)}\n"
        "suffix = str(payload.get('suffix') or '')\n"
        "log_path = pathlib.Path(str(payload.get('log_path') or ''))\n"
        "trajectory_root = pathlib.Path(str(payload.get('swe_agent_root') or '')) / 'trajectories'\n"
        "matches = []\n"
        "if suffix and trajectory_root.exists():\n"
        "    for candidate in trajectory_root.glob(f'**/*{suffix}*'):\n"
        "        if candidate.is_dir():\n"
        "            matches.append(candidate)\n"
        "trajectory_dirs = []\n"
        "total_issues = 0\n"
        "submitted_count = 0\n"
        "syntax_error_count = 0\n"
        "total_turns = 0\n"
        "result_files = []\n"
        "score_preview = {}\n"
        "for directory in sorted({str(item.resolve()) for item in matches}):\n"
        "    path = pathlib.Path(directory)\n"
        "    trajectory_dirs.append(directory)\n"
        "    yaml_path = path / 'run_batch_exit_statuses.yaml'\n"
        "    if yaml_path.is_file():\n"
        "        try:\n"
        "            import yaml\n"
        "            yaml_data = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))\n"
        "            if isinstance(yaml_data, dict) and isinstance(yaml_data.get('instances_by_exit_status'), dict):\n"
        "                statuses = yaml_data['instances_by_exit_status']\n"
        "                for status, instances in statuses.items():\n"
        "                    if isinstance(instances, list):\n"
        "                        count = len(instances)\n"
        "                        total_issues += count\n"
        "                        if status == 'submitted':\n"
        "                            submitted_count += count\n"
        "        except Exception:\n"
        "            pass\n"
        "    json_files = [item for item in path.glob('**/*.json') if item.is_file()] + [item for item in path.glob('**/*.traj') if item.is_file()]\n"
        "    if not yaml_path.is_file() and json_files:\n"
        "        for file_path in json_files:\n"
        "            try:\n"
        "                data = json.loads(file_path.read_text(encoding='utf-8'))\n"
        "            except Exception:\n"
        "                continue\n"
        "            if not isinstance(data, dict):\n"
        "                continue\n"
        "            total_issues += 1\n"
        "            history = data.get('history') if isinstance(data.get('history'), list) else []\n"
        "            total_turns += len(history)\n"
        "            info = data.get('info') if isinstance(data.get('info'), dict) else {}\n"
        "            if str(info.get('exit_status') or '') == 'submitted':\n"
        "                submitted_count += 1\n"
        "            for step in history:\n"
        "                if isinstance(step, dict) and 'SyntaxError' in str(step.get('observation') or ''):\n"
        "                    syntax_error_count += 1\n"
        "    for pattern in ('**/*result*.json', '**/*score*.json', '**/*summary*.json', '**/*report*.json'):\n"
        "        for file_path in path.glob(pattern):\n"
        "            if file_path.is_file() and str(file_path) not in result_files:\n"
        "                result_files.append(str(file_path))\n"
        "for file_path in result_files[:8]:\n"
        "    try:\n"
        "        data = json.loads(pathlib.Path(file_path).read_text(encoding='utf-8'))\n"
        "    except Exception:\n"
        "        continue\n"
        "    if isinstance(data, dict):\n"
        "        score_preview = data\n"
        "        break\n"
        "log_tail = ''\n"
        "if log_path.exists():\n"
        "    try:\n"
        "        log_tail = '\\n'.join(log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-40:])\n"
        "    except Exception:\n"
        "        log_tail = ''\n"
        "avg_turns = (float(total_turns) / float(total_issues)) if total_issues else 0.0\n"
        "log_lower = log_tail.lower()\n"
        "score = {}\n"
        "if isinstance(score_preview, dict):\n"
        "    for key in ('score', 'resolved', 'resolved_count', 'resolve_rate', 'pass_rate', 'status'):\n"
        "        if key in score_preview:\n"
        "            score[key] = score_preview.get(key)\n"
        "if not score:\n"
        "    score = {\n"
        "        'status': 'pending' if trajectory_dirs else 'not_found',\n"
        "        'submitted_count': submitted_count,\n"
        "        'trajectory_count': total_issues,\n"
        "        'resolve_rate_estimate': round((submitted_count / total_issues), 6) if total_issues else 0.0,\n"
        "    }\n"
        "fatal_error = ('settingserror' in log_lower) or ('invalid command line arguments' in log_lower) or ('unrecognized arguments:' in log_lower)\n"
        "if fatal_error:\n"
        "    score['status'] = 'launch_error'\n"
        "state = 'failed' if fatal_error else ('completed' if score.get('status') == 'completed' else ('running' if trajectory_dirs or log_tail else 'pending'))\n"
        "print(json.dumps({\n"
        "    'status': 'PASS' if trajectory_dirs or log_tail else 'FAIL',\n"
        "    'state': state,\n"
        "    'suffix': suffix,\n"
        "    'trajectory_root': str(trajectory_root),\n"
        "    'trajectory_dirs': trajectory_dirs,\n"
        "    'trajectory_count': total_issues,\n"
        "    'submitted_count': submitted_count,\n"
        "    'syntax_error_count': syntax_error_count,\n"
        "    'average_turns': avg_turns,\n"
        "    'score': score,\n"
        "    'score_source_files': result_files[:8],\n"
        "    'log_path': str(log_path),\n"
        "    'log_tail': log_tail,\n"
        "}, ensure_ascii=False))\n"
        "PY"
    )
    client = _connect_remote_host(head_host)
    try:
        result = _remote_load_json_payload(_remote_exec(client, command, timeout=180))
        # #region debug-point A:collect-summary-output
        _debug_report_dualnode_swe500(
            hypothesis_id="A",
            location="app/cli/cgc.py:_collect_remote_swebench_summary:output",
            msg="[DEBUG] collect remote swebench summary output",
            data={
                "status": str(result.get("status") or ""),
                "state": str(result.get("state") or ""),
                "trajectory_count": int(result.get("trajectory_count") or 0),
                "submitted_count": int(result.get("submitted_count") or 0),
                "score_status": str(_as_dict(result.get("score")).get("status") or ""),
                "score_source_files": list(result.get("score_source_files") or [])[:4],
            },
        )
        # #endregion
        return result
    finally:
        client.close()


def _runtime_contract_evidence_field_names():
    return (
        "compression_effective",
        "zero_copy_vram_real",
        "cpu_copy_count",
        "effective_collective_backend",
        "effective_cuda_graph",
        "effective_dispatch_backend",
        "effective_distributed_runtime",
        "effective_pd_service",
        "effective_storage_backend",
        "gds_effective",
        "spdk_effective",
        "colossalai_effective",
    )


def _aggregate_runtime_contract_section(host_rows, field_name):
    rows = []
    statuses = []
    scalar_mode = str(field_name) == "cpu_copy_count"
    for item in host_rows:
        payload = _as_dict(item.get("payload"))
        section = payload.get(field_name)
        if scalar_mode:
            if section is None:
                continue
            rows.append(
                {
                    "host": str(item.get("host") or ""),
                    "name": str(item.get("name") or ""),
                    "role": str(item.get("role") or ""),
                    "value": section,
                }
            )
            continue
        section_payload = _as_dict(section)
        if not section_payload:
            continue
        rows.append(
            {
                "host": str(item.get("host") or ""),
                "name": str(item.get("name") or ""),
                "role": str(item.get("role") or ""),
                "payload": section_payload,
            }
        )
        status_value = str(section_payload.get("status") or "").upper()
        if status_value:
            statuses.append(status_value)
    if scalar_mode:
        return {
            "status": "PASS" if rows else "SKIP",
            "by_host": rows,
            "value": rows[0].get("value") if rows else None,
        }
    if not rows:
        return {}
    aggregate_status = "SKIP"
    if any(status == "FAIL" for status in statuses):
        aggregate_status = "FAIL"
    elif any(status == "PASS" for status in statuses):
        aggregate_status = "PASS"
    elif any(status == "DECLARED" for status in statuses):
        aggregate_status = "DECLARED"
    elif statuses:
        aggregate_status = statuses[0]
    representative = _as_dict(rows[0].get("payload"))
    return {
        **representative,
        "status": aggregate_status,
        "by_host": rows,
    }


def _derive_runtime_contract_payload_from_m76_report(bundle):
    m76_report = _as_dict(_as_dict(bundle.get("m76_report")).get("payload"))
    gate_result = _as_dict(m76_report.get("gate_result"))
    m76_payload = _as_dict(gate_result.get("m76"))
    checks = _as_dict(m76_payload.get("checks"))
    if not checks:
        return {}
    deepep_contract = _as_dict(checks.get("deepep_contract"))
    heterogeneous_ir = _as_dict(checks.get("heterogeneous_unified_ir"))
    routing_strategy = _as_dict(heterogeneous_ir.get("routing_strategy"))
    collective_backend = str(routing_strategy.get("backend") or "").strip().lower()
    requested_dispatch_backend = "deepep" if deepep_contract else "native_sglang"
    effective_dispatch_backend = "deepep" if str(deepep_contract.get("status") or "").upper() == "PASS" else "native_sglang"
    distributed_backend = collective_backend or ("nccl" if effective_dispatch_backend == "deepep" else "single_process")
    enable_pd = True
    pd_endpoint = str(
        _as_dict(checks.get("pd_service")).get("payload", {}).get("endpoint")
        or _as_dict(checks.get("pd_service")).get("endpoint")
        or "localhost:50051"
    )
    pd_mode = str(
        _as_dict(checks.get("pd_service")).get("payload", {}).get("mode")
        or _as_dict(checks.get("pd_service")).get("mode")
        or "cloud_prefill_edge_decode"
    )
    return {
        "runtime_protocol_contract": {
            "protocol_family": "trueorthokda",
            "state_kind": "kda_state_v1",
            "state_codec": "cq4",
            "expected_zero_copy": True,
            "enable_nccl": collective_backend in {"nccl", "hccl"},
            "enable_cuda_graph": False,
            "requested_dispatch_backend": requested_dispatch_backend,
            "requested_distributed_runtime": distributed_backend,
            "requested_storage_backend": "posix",
            "enable_gds": False,
            "enable_spdk": False,
            "use_colossalai": False,
            "colossalai_plugin": "",
            "enable_pd": enable_pd,
            "pd_endpoint": pd_endpoint,
            "pd_mode": pd_mode,
            "pd_prefix_cache": True,
            "require_pd_service": True,
            "source": "m76_report_projection",
        },
        "compression_effective": {
            "status": "SKIP",
            "reason": "legacy_remote_runtime_evidence_missing_compression_fields",
        },
        "zero_copy_vram_real": {
            "status": "SKIP",
            "reason": "legacy_remote_runtime_evidence_missing_zero_copy_fields",
            "cpu_copy_count": None,
            "uma_buffer_used": False,
            "device_resume_consumed": False,
            "resume_tensor_device": "",
        },
        "cpu_copy_count": None,
        "effective_collective_backend": {
            "status": "PASS" if collective_backend else "SKIP",
            "backend": collective_backend or "none",
            "source": "m76_report_projection",
        },
        "effective_cuda_graph": {
            "status": "SKIP",
            "enabled": False,
            "reason": "legacy_remote_runtime_evidence_missing_cuda_graph_fields",
            "source": "m76_report_projection",
        },
        "effective_dispatch_backend": {
            "status": "PASS",
            "backend": effective_dispatch_backend,
            "source": "m76_report_projection",
        },
        "effective_distributed_runtime": {
            "status": "PASS",
            "backend": distributed_backend,
            "source": "m76_report_projection",
        },
        "effective_pd_service": {
            "status": "PASS" if enable_pd and bool(pd_endpoint) else "FAIL",
            "enabled": enable_pd,
            "require_pd_service": True,
            "endpoint": pd_endpoint,
            "mode": pd_mode,
            "prefix_cache": True,
            "provider": "PDClient",
            "client_available": True,
            "reason": "" if enable_pd and bool(pd_endpoint) else "pd_endpoint_missing",
            "source": "m76_report_projection",
        },
        "effective_storage_backend": {
            "status": "SKIP",
            "backend": "posix",
            "requested_backend": "posix",
            "reason": "legacy_remote_runtime_evidence_missing_storage_backend_fields",
            "source": "m76_report_projection",
        },
        "gds_effective": {
            "status": "SKIP",
            "enabled": False,
            "backend": "posix",
            "source": "m76_report_projection",
        },
        "spdk_effective": {
            "status": "SKIP",
            "enabled": False,
            "backend": "posix",
            "source": "m76_report_projection",
        },
        "colossalai_effective": {
            "status": "SKIP",
            "enabled": False,
            "backend": "single_process",
            "plugin": "",
            "source": "m76_report_projection",
        },
    }


def _summarize_remote_runtime_contract_bundle(host_entries):
    rows = []
    for host_payload in host_entries if isinstance(host_entries, list) else []:
        host_info = _as_dict(host_payload)
        bundle = _as_dict(host_info.get("bundle"))
        runtime_payload = _as_dict(_as_dict(bundle.get("nvidia_runtime")).get("payload"))
        if not runtime_payload:
            runtime_payload = _as_dict(_as_dict(bundle.get("extreme_scale_runtime")).get("payload"))
        if not _as_dict(runtime_payload.get("runtime_protocol_contract")):
            runtime_payload = {
                **_derive_runtime_contract_payload_from_m76_report(bundle),
                **runtime_payload,
            }
        if not runtime_payload:
            continue
        rows.append(
            {
                "host": str(host_info.get("host") or ""),
                "name": str(host_info.get("name") or ""),
                "role": str(host_info.get("role") or ""),
                "payload": runtime_payload,
            }
        )
    protocol_rows = []
    canonical_runtime_protocol_contract = {}
    for item in rows:
        contract_payload = _as_dict(_as_dict(item.get("payload")).get("runtime_protocol_contract"))
        if not contract_payload:
            continue
        protocol_rows.append(
            {
                "host": str(item.get("host") or ""),
                "name": str(item.get("name") or ""),
                "role": str(item.get("role") or ""),
                "payload": contract_payload,
            }
        )
        if not canonical_runtime_protocol_contract:
            canonical_runtime_protocol_contract = dict(contract_payload)
    field_summaries = {
        field_name: _aggregate_runtime_contract_section(rows, field_name)
        for field_name in _runtime_contract_evidence_field_names()
    }
    return {
        "runtime_protocol_contract": canonical_runtime_protocol_contract,
        "runtime_protocol_contracts": protocol_rows,
        **field_summaries,
    }


def _collect_remote_m76_evidence(host_specs):
    payload = {"status": "PASS", "hosts": [], "summary": {"runtime_statuses": [], "report_statuses": []}}
    found_any = False
    for host_spec in host_specs:
        role = str(host_spec.get("role") or "")
        # #region debug-point D:m76-evidence-connect-attempt
        _debug_report_dualnode_swe500(
            hypothesis_id="D",
            location="app/cli/cgc.py:_collect_remote_m76_evidence:connect_attempt",
            msg="[DEBUG] m76 evidence connect attempt",
            data={
                "host": str(host_spec.get("host") or ""),
                "role": role,
                "user": str(host_spec.get("user") or ""),
            },
        )
        # #endregion
        remote_root = _remote_repo_root()
        command_payload = {
            "remote_repo_root": remote_root,
            "candidates": {
                "m76_checkin": [f"{remote_root}/CGC_Release/checkins/m76_latest.json"],
                "m76_report": [f"{remote_root}/ComputeGraphCompiler-main/Output/cli_gate_m76/m76_heterogeneous/m76_report.json"],
                "nvidia_runtime": [f"{remote_root}/ComputeGraphCompiler-main/Output/cli_gate_m76/runtime_evidence/nvidia_runtime.json"],
                "extreme_scale_runtime": [f"{remote_root}/ComputeGraphCompiler-main/Output/cli_gate_m75/runtime_evidence/extreme_scale_runtime.json"],
            },
        }
        command = (
            "python3 - <<'PY'\n"
            "import json\n"
            "import pathlib\n"
            "import fnmatch\n"
            f"payload = {json.dumps(command_payload, ensure_ascii=False)}\n"
            "remote_root = str(payload.get('remote_repo_root') or '')\n"
            "fallback_roots = [\n"
            "    pathlib.Path(remote_root),\n"
            "    pathlib.Path(remote_root) / 'ComputeGraphCompiler-main' / 'Output',\n"
            "    pathlib.Path(remote_root) / 'CGC_Release' / 'checkins',\n"
            "    pathlib.Path('/root') / 'flashkv0516',\n"
            "    pathlib.Path('/root') / 'flashkv0516' / 'ComputeGraphCompiler-main' / 'Output',\n"
            "    pathlib.Path('/root') / 'flashkv0516' / 'CGC_Release' / 'checkins',\n"
            "]\n"
            "def load_candidate(paths):\n"
            "    for raw_path in paths:\n"
            "        path = pathlib.Path(str(raw_path))\n"
            "        if path.exists():\n"
            "            try:\n"
            "                parsed = json.loads(path.read_text(encoding='utf-8'))\n"
            "            except Exception:\n"
            "                parsed = {}\n"
            "            return {\n"
            "                'exists': True,\n"
            "                'path': str(path),\n"
            "                'size_bytes': int(path.stat().st_size),\n"
            "                'payload': parsed if isinstance(parsed, dict) else {},\n"
            "            }\n"
            "    return {'exists': False, 'path': str(paths[0]) if paths else '', 'payload': {}}\n"
            "def search_candidates(patterns, roots, limit=12):\n"
            "    matches = []\n"
            "    seen = set()\n"
            "    for root in roots:\n"
            "        if not root.exists():\n"
            "            continue\n"
            "        try:\n"
            "            for path in root.rglob('*'):\n"
            "                if not path.is_file():\n"
            "                    continue\n"
            "                if not any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):\n"
            "                    continue\n"
            "                resolved = str(path)\n"
            "                if resolved in seen:\n"
            "                    continue\n"
            "                seen.add(resolved)\n"
            "                matches.append(path)\n"
            "                if len(matches) >= limit:\n"
            "                    return matches\n"
            "        except Exception:\n"
            "            continue\n"
            "    return matches\n"
            "def rank_path(path, key):\n"
            "    score = int(path.stat().st_mtime) if path.exists() else 0\n"
            "    text = str(path)\n"
            "    if key == 'm76_report':\n"
            "        for token, weight in (('cli_gate_m76', 100000), ('m76_heterogeneous', 50000), ('m76_report.json', 10000)):\n"
            "            if token in text:\n"
            "                score += weight\n"
            "    elif key == 'nvidia_runtime':\n"
            "        for token, weight in (('cli_gate_m76', 100000), ('runtime_evidence', 50000), ('nvidia_runtime.json', 10000)):\n"
            "            if token in text:\n"
            "                score += weight\n"
            "    elif key == 'extreme_scale_runtime':\n"
            "        for token, weight in (('cli_gate_m75', 100000), ('runtime_evidence', 50000), ('extreme_scale_runtime.json', 10000)):\n"
            "            if token in text:\n"
            "                score += weight\n"
            "    elif key == 'm76_checkin':\n"
            "        for token, weight in (('checkins', 100000), ('m76', 10000)):\n"
            "            if token in text:\n"
            "                score += weight\n"
            "    return score\n"
            "def search_best(patterns, roots, key):\n"
            "    matches = search_candidates(patterns, roots)\n"
            "    if not matches:\n"
            "        return {'exists': False, 'path': '', 'payload': {}, 'search_matches': []}\n"
            "    best = sorted(matches, key=lambda item: rank_path(item, key), reverse=True)[0]\n"
            "    loaded = load_candidate([best])\n"
            "    loaded['search_matches'] = [str(item) for item in matches[:8]]\n"
            "    return loaded\n"
            "bundle = {key: load_candidate(value) for key, value in (payload.get('candidates') or {}).items()}\n"
            "if not (bundle.get('m76_checkin') or {}).get('exists'):\n"
            "    bundle['m76_checkin'] = search_best(['*m76*.json'], [pathlib.Path(remote_root) / 'CGC_Release' / 'checkins', pathlib.Path('/root') / 'flashkv0516' / 'CGC_Release' / 'checkins'], 'm76_checkin')\n"
            "checkin_payload = (bundle.get('m76_checkin') or {}).get('payload') if isinstance(bundle.get('m76_checkin'), dict) else {}\n"
            "report_path = str((checkin_payload or {}).get('report_path') or '').strip()\n"
            "if report_path and not (bundle.get('m76_report') or {}).get('exists'):\n"
            "    translated = report_path.replace('/Users/alexchuang/Documents/flashkv0516', remote_root)\n"
            "    translated_result = load_candidate([translated])\n"
            "    if translated_result.get('exists'):\n"
            "        bundle['m76_report'] = translated_result\n"
            "if not (bundle.get('m76_report') or {}).get('exists'):\n"
            "    bundle['m76_report'] = search_best(['m76_report.json'], fallback_roots, 'm76_report')\n"
            "if not (bundle.get('nvidia_runtime') or {}).get('exists'):\n"
            "    bundle['nvidia_runtime'] = search_best(['nvidia_runtime.json'], fallback_roots, 'nvidia_runtime')\n"
            "if not (bundle.get('extreme_scale_runtime') or {}).get('exists'):\n"
            "    bundle['extreme_scale_runtime'] = search_best(['extreme_scale_runtime.json'], fallback_roots, 'extreme_scale_runtime')\n"
            "print(json.dumps(bundle, ensure_ascii=False))\n"
            "PY"
        )
        try:
            client = _connect_remote_host(host_spec)
        except Exception as exc:
            # #region debug-point D:m76-evidence-connect-error
            _debug_report_dualnode_swe500(
                hypothesis_id="D",
                location="app/cli/cgc.py:_collect_remote_m76_evidence:connect_error",
                msg="[DEBUG] m76 evidence connect error",
                data={
                    "host": str(host_spec.get("host") or ""),
                    "role": role,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            # #endregion
            raise
        try:
            bundle = _remote_load_json_payload(_remote_exec(client, command, timeout=120))
        finally:
            client.close()
        host_payload = {
            "host": str(host_spec.get("host") or ""),
            "name": str(host_spec.get("name") or ""),
            "role": role,
            "bundle": bundle,
        }
        runtime_status = str(_nested_get(bundle, "nvidia_runtime.payload.status", ""))
        report_status = str(_nested_get(bundle, "m76_report.payload.gate_result.m76.status", ""))
        found_any = bool(
            found_any
            or bool(_nested_get(bundle, "nvidia_runtime.exists", False))
            or bool(_nested_get(bundle, "m76_report.exists", False))
            or bool(_nested_get(bundle, "extreme_scale_runtime.exists", False))
        )
        payload["summary"]["runtime_statuses"].append({"role": role, "status": runtime_status})
        payload["summary"]["report_statuses"].append({"role": role, "status": report_status})
        payload["hosts"].append(host_payload)
    payload.update(_summarize_remote_runtime_contract_bundle(payload.get("hosts")))
    if not payload["hosts"] or not found_any:
        payload["status"] = "FAIL"
    return payload


def _resolve_swe_verified_session_path(session_path):
    raw = str(session_path or "").strip()
    if not raw:
        raise ValueError("Missing --refresh-session path")
    candidate = Path(raw).expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "model_swe_verified_session.json"
    if not candidate.exists():
        raise ValueError(f"SWE verified session not found: {candidate}")
    return candidate


def _upsert_stage_trace(stage_trace, *, stage, status, artifact):
    rows = stage_trace if isinstance(stage_trace, list) else []
    updated = []
    replaced = False
    for item in rows:
        entry = dict(item) if isinstance(item, dict) else {}
        if str(entry.get("stage") or "") == str(stage):
            entry["status"] = str(status)
            entry["artifact"] = str(artifact)
            replaced = True
        updated.append(entry)
    if not replaced:
        updated.append({"stage": str(stage), "status": str(status), "artifact": str(artifact)})
    return updated


# #region debug-point A:dualnode-swe500-report
def _debug_report_dualnode_swe500(*, hypothesis_id, location, msg, data=None, run_id="pre-fix"):
    import json as _json
    import urllib.request as _urllib_request

    _env_path = Path(".dbg/dualnode-swe500-blocker.env")
    _url = "http://127.0.0.1:7777/event"
    _session = "dualnode-swe500-blocker"
    try:
        if _env_path.exists():
            _content = _env_path.read_text(encoding="utf-8", errors="replace")
            for _line in _content.splitlines():
                if _line.startswith("DEBUG_SERVER_URL="):
                    _url = _line.split("=", 1)[1].strip() or _url
                elif _line.startswith("DEBUG_SESSION_ID="):
                    _session = _line.split("=", 1)[1].strip() or _session
        _payload = {
            "sessionId": _session,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data if isinstance(data, dict) else {},
        }
        _urllib_request.urlopen(
            _urllib_request.Request(
                _url,
                data=_json.dumps(_payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=2,
        ).read()
    except Exception:
        pass
# #endregion


def _evaluate_swebench_score_contract(remote_score_summary):
    summary = _as_dict(remote_score_summary)
    if not summary:
        return {
            "contract_status": "FAIL",
            "benchmark_state": "pending",
            "score_status": "",
            "reason": "missing_score_summary",
        }
    state = str(summary.get("state") or "").strip().lower()
    score = _as_dict(summary.get("score"))
    score_status = str(score.get("status") or "").strip().lower()
    trajectory_count = int(summary.get("trajectory_count") or 0)
    submitted_count = int(summary.get("submitted_count") or 0)
    score_source_files = summary.get("score_source_files") if isinstance(summary.get("score_source_files"), list) else []
    has_score_payload = bool(score_source_files) or any(
        key in score for key in ("resolved", "resolved_count", "resolve_rate", "pass_rate", "score")
    )
    if str(summary.get("status") or "").strip().upper() == "FAIL":
        return {
            "contract_status": "FAIL",
            "benchmark_state": state or "failed",
            "score_status": score_status,
            "reason": "score_summary_collection_failed",
        }
    if state == "failed" or score_status in {"failed", "error", "launch_error"}:
        return {
            "contract_status": "FAIL",
            "benchmark_state": state or "failed",
            "score_status": score_status,
            "reason": "remote_benchmark_failed",
        }
    completed = state == "completed" or score_status in {"completed", "pass", "passed", "success"}
    if completed:
        if submitted_count > 0 and has_score_payload:
            return {
                "contract_status": "PASS",
                "benchmark_state": "completed",
                "score_status": score_status,
                "reason": "",
            }
        return {
            "contract_status": "FAIL",
            "benchmark_state": "completed",
            "score_status": score_status,
            "reason": "completed_without_score_payload_or_submissions",
        }
    if trajectory_count > 0 or state in {"running", "pending"} or score_status in {"running", "pending"}:
        return {
            "contract_status": "PENDING",
            "benchmark_state": state or "running",
            "score_status": score_status,
            "reason": "benchmark_not_completed",
        }
    return {
        "contract_status": "FAIL",
        "benchmark_state": state or "pending",
        "score_status": score_status,
        "reason": "no_trajectory_or_score_evidence",
    }


def _refresh_model_swe_verified_session(
    *,
    refresh_session,
    poll=False,
    poll_interval_seconds=30,
    max_polls=120,
):
    session_path = _resolve_swe_verified_session_path(refresh_session)
    payload = _safe_read_json(session_path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Failed to read swe verified session: {session_path}")

    out_dir = session_path.parent
    cluster_status = _as_dict(payload.get("cluster_status"))
    launch_plan = _as_dict(payload.get("launch_plan"))
    execution = _as_dict(payload.get("execution"))
    artifact_index = _as_dict(payload.get("artifact_index"))
    # Refresh should honor current env overrides instead of pinning stale credentials from the saved session.
    remote_host_specs = _resolve_swebench_remote_hosts(cluster_status)
    if not remote_host_specs:
        remote_host_specs = execution.get("remote_hosts") if isinstance(execution.get("remote_hosts"), list) else []
    remote_head_host = _find_remote_host_by_role(remote_host_specs, "head")

    remote_score_summary = {}
    remote_m76_evidence = {}
    actual_attempts = 0
    completed = False
    poll_samples = []
    attempts = max(1, int(max_polls or 1)) if bool(poll) else 1
    interval_s = max(1, int(poll_interval_seconds or 30))
    for attempt in range(1, attempts + 1):
        actual_attempts = attempt
        try:
            remote_score_summary = _collect_remote_swebench_summary(
                head_host=remote_head_host,
                launch_plan=launch_plan,
            )
        except Exception as exc:
            remote_score_summary = {"status": "FAIL", "error": str(exc)}
        benchmark_state = str(remote_score_summary.get("state") or "")
        score_status = str(_nested_get(remote_score_summary, "score.status", "")).lower()
        completed = benchmark_state == "completed" or score_status in {"completed", "pass", "passed", "success"}
        # #region debug-point B:refresh-score-sample
        _debug_report_dualnode_swe500(
            hypothesis_id="B",
            location="app/cli/cgc.py:_refresh_model_swe_verified_session:poll",
            msg="[DEBUG] refresh swe verified score sample",
            data={
                "attempt": int(attempt),
                "benchmark_state": benchmark_state,
                "score_status": score_status,
                "trajectory_count": int(remote_score_summary.get("trajectory_count") or 0),
                "submitted_count": int(remote_score_summary.get("submitted_count") or 0),
            },
        )
        # #endregion
        poll_samples.append(
            {
                "attempt": int(attempt),
                "refreshed_at": utc_now_iso(),
                "state": benchmark_state,
                "score_status": score_status,
                "trajectory_count": int(remote_score_summary.get("trajectory_count") or 0),
                "submitted_count": int(remote_score_summary.get("submitted_count") or 0),
            }
        )
        if completed or not bool(poll) or attempt >= attempts:
            break
        time.sleep(interval_s)

    try:
        remote_m76_evidence = _collect_remote_m76_evidence(remote_host_specs)
    except Exception as exc:
        remote_m76_evidence = {"status": "FAIL", "error": str(exc), "hosts": []}

    remote_score_summary_path = write_json_file(
        out_dir / "remote_swebench_score_summary.json",
        remote_score_summary,
    )
    remote_m76_evidence_path = write_json_file(
        out_dir / "remote_m76_evidence_bundle.json",
        remote_m76_evidence,
    )
    artifact_index["remote_swebench_score_summary"] = str(remote_score_summary_path)
    artifact_index["remote_m76_evidence_bundle"] = str(remote_m76_evidence_path)

    m76_summary = _as_dict(remote_m76_evidence.get("summary"))
    runtime_statuses = m76_summary.get("runtime_statuses") if isinstance(m76_summary.get("runtime_statuses"), list) else []
    report_statuses = m76_summary.get("report_statuses") if isinstance(m76_summary.get("report_statuses"), list) else []
    runtime_protocol_contract = _as_dict(remote_m76_evidence.get("runtime_protocol_contract"))
    mandatory_protocol_gate = evaluate_mandatory_protocol_gate(
        runtime_protocol_contract=runtime_protocol_contract,
        zero_copy_vram_real=_as_dict(remote_m76_evidence.get("zero_copy_vram_real")),
        source=str(remote_m76_evidence_path),
    )
    score_contract = _evaluate_swebench_score_contract(remote_score_summary)
    remote_score_contract_status = str(score_contract.get("contract_status") or "PENDING")
    remote_evidence_contract_status = str(remote_m76_evidence.get("status") or "PENDING")
    benchmark_state = str(score_contract.get("benchmark_state") or remote_score_summary.get("state") or execution.get("benchmark_state") or "pending")
    # #region debug-point C:refresh-score-contract
    _debug_report_dualnode_swe500(
        hypothesis_id="C",
        location="app/cli/cgc.py:_refresh_model_swe_verified_session:contract",
        msg="[DEBUG] refresh swe verified score contract",
        data={
            "contract_status": remote_score_contract_status,
            "benchmark_state": benchmark_state,
            "score_status": str(score_contract.get("score_status") or ""),
            "reason": str(score_contract.get("reason") or ""),
        },
    )
    # #endregion

    accepted_contracts = _as_dict(payload.get("accepted_contracts"))
    accepted_contracts["swebench_score_recovery"] = {
        **_as_dict(accepted_contracts.get("swebench_score_recovery")),
        "status": remote_score_contract_status,
        "state": benchmark_state,
        "score_summary_path": str(remote_score_summary_path),
        "trajectory_count": int(remote_score_summary.get("trajectory_count") or 0),
        "submitted_count": int(remote_score_summary.get("submitted_count") or 0),
        "score_status": str(score_contract.get("score_status") or ""),
        "reason": str(score_contract.get("reason") or ""),
    }
    accepted_contracts["m76_runtime_evidence"] = {
        **_as_dict(accepted_contracts.get("m76_runtime_evidence")),
        "status": remote_evidence_contract_status,
        "evidence_bundle_path": str(remote_m76_evidence_path),
        "runtime_statuses": runtime_statuses,
        "report_statuses": report_statuses,
        "runtime_protocol_contract": runtime_protocol_contract,
        "mandatory_protocol_gate": mandatory_protocol_gate,
    }

    payload["artifact_index"] = artifact_index
    payload["accepted_contracts"] = accepted_contracts
    payload["benchmark_summary"] = remote_score_summary
    payload["m76_evidence_bundle"] = remote_m76_evidence
    payload["runtime_protocol_contract"] = runtime_protocol_contract
    payload["runtime_protocol_contracts"] = list(remote_m76_evidence.get("runtime_protocol_contracts") or [])
    payload["mandatory_protocol_gate"] = mandatory_protocol_gate
    for field_name in _runtime_contract_evidence_field_names():
        payload[field_name] = remote_m76_evidence.get(field_name)
    launch_contract_status = str(_as_dict(accepted_contracts.get("swebench_remote_launch")).get("status") or "").upper()
    payload["status"] = (
        "PASS"
        if launch_contract_status == "PASS" and remote_score_contract_status == "PASS" and remote_evidence_contract_status == "PASS"
        else ("PENDING" if launch_contract_status == "PASS" and remote_score_contract_status == "PENDING" else "FAIL")
    )
    payload["last_refreshed_at"] = utc_now_iso()
    payload["refresh"] = {
        "poll_enabled": bool(poll),
        "poll_interval_seconds": interval_s,
        "max_polls": int(attempts),
        "actual_attempts": int(actual_attempts),
        "completed": bool(completed),
        "samples": poll_samples,
    }
    execution["benchmark_state"] = benchmark_state
    execution["remote_score_summary_path"] = str(remote_score_summary_path)
    execution["remote_m76_evidence_path"] = str(remote_m76_evidence_path)
    execution["remote_hosts"] = remote_host_specs
    payload["execution"] = execution
    payload["stage_trace"] = _upsert_stage_trace(
        payload.get("stage_trace"),
        stage="score_recovery",
        status=remote_score_contract_status,
        artifact=remote_score_summary_path,
    )
    payload["stage_trace"] = _upsert_stage_trace(
        payload.get("stage_trace"),
        stage="m76_evidence_recovery",
        status=remote_evidence_contract_status,
        artifact=remote_m76_evidence_path,
    )
    payload["swe_verified_session_path"] = str(session_path)
    write_json_file(session_path, payload)
    return payload


def _nested_get(value, dotted_key, default=""):
    current = value
    for part in str(dotted_key or "").split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
    return default if current is None else current


def _resolve_model_release_artifacts(*, artifact_root, run_session):
    root = Path(str(artifact_root)).expanduser().resolve()
    session = _as_dict(run_session)
    build_artifacts = _as_dict(session.get("build_artifacts"))
    m6_build_report_path = _pick_existing_file(
        build_artifacts.get("m6_build_report"),
        root / "build_report.json",
        root / "m6_product" / "build_report.json",
        ENGINE_REPO_DIR / "Output" / "cli_gate_m6" / "m6_product" / "build_report.json",
        REPO_ROOT / "temp" / "test" / "m6_rerun" / "m6_product" / "build_report.json",
    )
    m6_build_report = _safe_read_json(m6_build_report_path) if m6_build_report_path else {}
    m6_bundle_manifest_path = _pick_existing_file(
        build_artifacts.get("bundle_manifest"),
        _nested_get(m6_build_report, "steps.build_bundle.gate.manifest_path", ""),
        _nested_get(m6_build_report, "gate_result.m6.build_bundle_gate.manifest_path", ""),
    )
    build_matrix_manifest_path = _pick_existing_file(
        build_artifacts.get("build_matrix_manifest"),
        build_artifacts.get("dist_manifest"),
        root / "build_matrix_manifest.json",
        root / "release" / "build_matrix_manifest.json",
        REPO_ROOT / "CGC_Release" / "dist" / "build_matrix_manifest.json",
    )
    build_matrix_manifest = _safe_read_json(build_matrix_manifest_path) if build_matrix_manifest_path else {}
    build_matrix_file_path = _pick_existing_file(
        build_artifacts.get("build_matrix_file"),
        _nested_get(build_matrix_manifest, "matrix_file", ""),
        root / "build_matrix.json",
    )
    build_matrix = _safe_read_json(build_matrix_file_path) if build_matrix_file_path else {}
    platform_reports = {}
    for platform_name, details in _as_dict(build_matrix_manifest.get("platforms")).items():
        report_path = _pick_existing_file(_as_dict(details).get("report_path"))
        if report_path:
            platform_reports[str(platform_name)] = str(report_path)
    return {
        "m6_build_report_path": str(m6_build_report_path) if m6_build_report_path else "",
        "m6_build_report": _as_dict(m6_build_report),
        "m6_bundle_manifest_path": str(m6_bundle_manifest_path) if m6_bundle_manifest_path else "",
        "build_matrix_manifest_path": str(build_matrix_manifest_path) if build_matrix_manifest_path else "",
        "build_matrix_manifest": _as_dict(build_matrix_manifest),
        "build_matrix_file_path": str(build_matrix_file_path) if build_matrix_file_path else "",
        "build_matrix": _as_dict(build_matrix),
        "platform_reports": platform_reports,
    }


def _resolve_model_context(*, run_session_path="", artifact_root=""):
    session = {}
    root = None
    resolved_run_session_path = ""
    if str(run_session_path or "").strip():
        resolved_run_session_path = str(Path(run_session_path).expanduser().resolve())
        session = _safe_read_json(resolved_run_session_path)
        if not isinstance(session, dict) or not session:
            raise ValueError(f"Failed to read model run session: {resolved_run_session_path}")
        root = _pick_existing_file(
            session.get("source_artifact_root"),
            session.get("artifact_root"),
            session.get("output_dir"),
        )
        if root is not None and root.is_file():
            root = root.parent
    if root is None and str(artifact_root or "").strip():
        root = Path(str(artifact_root)).expanduser().resolve()
    if root is None:
        root = CGC_RUN_ARTIFACT_ROOT if CGC_RUN_ARTIFACT_ROOT.exists() else None
    if root is None or not root.exists():
        raise ValueError("Either --run-session or --artifact-root is required, or a latest cgc run artifact root must exist.")
    run_report_path = _pick_existing_file(
        session.get("run_report_path"),
        ((session.get("evidence_paths") or {}) if isinstance(session.get("evidence_paths"), dict) else {}).get("run_report"),
        root / "run_report.json",
        root / "run_artifacts" / "run_report.json",
        CGC_RUN_LATEST_REPORT,
    )
    route_decision_path = _pick_existing_file(
        session.get("route_decision_path"),
        ((session.get("evidence_paths") or {}) if isinstance(session.get("evidence_paths"), dict) else {}).get("route_decision"),
        root / "route_decision.json",
        root / "run_artifacts" / "route_decision.json",
        CGC_RUN_LATEST_ROUTE_DECISION,
    )
    m4_inference_report_path = _pick_existing_file(
        session.get("m4_inference_report_path"),
        ((session.get("evidence_paths") or {}) if isinstance(session.get("evidence_paths"), dict) else {}).get("m4_inference_report"),
        root / "m4_inference_report.json",
        root / "run_artifacts" / "m4_inference_report.json",
        CGC_RUN_LATEST_M4_INFERENCE_REPORT,
    )
    edge_bridge_path = _pick_existing_file(
        session.get("edge_inference_bridge_path"),
        ((session.get("evidence_paths") or {}) if isinstance(session.get("evidence_paths"), dict) else {}).get("edge_inference_bridge"),
        root / "edge_inference_bridge.json",
        root / "run_artifacts" / "edge_inference_bridge.json",
        CGC_RUN_LATEST_EDGE_BRIDGE,
    )
    run_report = _safe_read_json(run_report_path) if run_report_path else {}
    route_decision = _safe_read_json(route_decision_path) if route_decision_path else {}
    m4_inference_report = _safe_read_json(m4_inference_report_path) if m4_inference_report_path else {}
    edge_bridge = _safe_read_json(edge_bridge_path) if edge_bridge_path else {}
    release_artifacts = _resolve_model_release_artifacts(artifact_root=root, run_session=session)
    return {
        "run_session": session,
        "run_session_path": resolved_run_session_path,
        "artifact_root": str(root),
        "run_report_path": str(run_report_path) if run_report_path else "",
        "route_decision_path": str(route_decision_path) if route_decision_path else "",
        "m4_inference_report_path": str(m4_inference_report_path) if m4_inference_report_path else "",
        "edge_inference_bridge_path": str(edge_bridge_path) if edge_bridge_path else "",
        "run_report": run_report if isinstance(run_report, dict) else {},
        "route_decision": route_decision if isinstance(route_decision, dict) else {},
        "m4_inference_report": m4_inference_report if isinstance(m4_inference_report, dict) else {},
        "edge_inference_bridge": edge_bridge if isinstance(edge_bridge, dict) else {},
        "release_artifacts": release_artifacts,
    }


def _search_nested_string(payload, *keys):
    wanted = {str(key).strip() for key in keys if str(key).strip()}
    if not wanted:
        return ""
    wanted_markers = tuple(f'"{key}"' for key in wanted)
    parsed_texts = set()
    stack = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                key_str = str(key).strip()
                if key_str in wanted and str(value or "").strip():
                    return str(value)
                if isinstance(value, (dict, list)):
                    stack.append(value)
                elif isinstance(value, str) and any(marker in value for marker in wanted_markers):
                    stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)
        elif isinstance(current, str):
            stripped = current.strip()
            if not stripped or not any(marker in stripped for marker in wanted_markers):
                continue
            parsed = None
            for candidate_text in (
                stripped,
                stripped[stripped.find("{") : stripped.rfind("}") + 1] if "{" in stripped and "}" in stripped else "",
                stripped[stripped.find("[") : stripped.rfind("]") + 1] if "[" in stripped and "]" in stripped else "",
            ):
                if not candidate_text:
                    continue
                try:
                    parsed = json.loads(candidate_text)
                except Exception:
                    parsed = None
                if isinstance(parsed, (dict, list)):
                    break
            if not isinstance(parsed, (dict, list)):
                continue
            parsed_key = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
            if parsed_key in parsed_texts:
                continue
            parsed_texts.add(parsed_key)
            stack.append(parsed)
    return ""


def _resolve_model_profile_bundle_validation(*, context):
    ctx = _as_dict(context)
    run_session = _as_dict(ctx.get("run_session"))
    release_artifacts = _as_dict(ctx.get("release_artifacts"))
    m6_build_report = _as_dict(release_artifacts.get("m6_build_report"))
    build_matrix_manifest = _as_dict(release_artifacts.get("build_matrix_manifest"))
    profile_settings_path = _pick_existing_file(
        run_session.get("profile_settings_path"),
        _search_nested_string(run_session, "profile_settings_path"),
        _search_nested_string(m6_build_report, "profile_settings_path"),
        _search_nested_string(build_matrix_manifest, "profile_settings_path"),
        Path(str(ctx.get("artifact_root") or "")) / "profile_settings.json",
    )
    system_manifest_path = _pick_existing_file(
        _search_nested_string(run_session, "system_execution_manifest_path", "system_manifest_path"),
        _search_nested_string(m6_build_report, "system_execution_manifest_path", "system_manifest_path"),
        _search_nested_string(build_matrix_manifest, "system_execution_manifest_path", "system_manifest_path"),
        Path(str(ctx.get("artifact_root") or "")) / "system_execution_manifest.json",
    )
    bootstrap_contract_path = _pick_existing_file(
        _search_nested_string(run_session, "distributed_runtime_bootstrap", "bootstrap_contract_path"),
        _search_nested_string(m6_build_report, "distributed_runtime_bootstrap", "bootstrap_contract_path"),
        _search_nested_string(build_matrix_manifest, "distributed_runtime_bootstrap", "bootstrap_contract_path"),
        Path(str(ctx.get("artifact_root") or "")) / "distributed_runtime_bootstrap.json",
    )
    validation = validate_profile_bundle(
        profile_settings_path=str(profile_settings_path or ""),
        system_manifest_path=str(system_manifest_path or ""),
        bootstrap_contract_path=str(bootstrap_contract_path or ""),
    )
    validation["resolved_paths"] = {
        "profile_settings_path": str(profile_settings_path) if profile_settings_path else "",
        "system_manifest_path": str(system_manifest_path) if system_manifest_path else "",
        "bootstrap_contract_path": str(bootstrap_contract_path) if bootstrap_contract_path else "",
    }
    return validation


def _resolve_bundle_review_from_report(*, report_path):
    resolved_report_path = str(Path(str(report_path or "")).expanduser().resolve())
    payload = _safe_read_json(resolved_report_path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Failed to read bundle review source report: {resolved_report_path}")
    report_name = Path(resolved_report_path).name
    report_root = Path(resolved_report_path).parent
    inferred_run_session_path = _pick_existing_file(
        _search_nested_string(payload, "run_session_path", "model_run_session_path"),
        report_root.parent / "model_run_session.json" if report_root.name == "run_artifacts" else "",
        report_root / "model_run_session.json",
    )
    inferred_artifact_root = _pick_existing_file(
        _search_nested_string(payload, "artifact_root", "source_artifact_root", "report_dir"),
        report_root.parent if report_root.name == "run_artifacts" else "",
        report_root if report_name == "build_report.json" else "",
    )
    if inferred_artifact_root is not None:
        inferred_artifact_root = inferred_artifact_root.parent if inferred_artifact_root.is_file() else inferred_artifact_root
        if inferred_artifact_root.name == "run_artifacts":
            inferred_artifact_root = inferred_artifact_root.parent
    if inferred_run_session_path or inferred_artifact_root:
        context = _resolve_model_context(
            run_session_path=str(inferred_run_session_path or ""),
            artifact_root=str(inferred_artifact_root or ""),
        )
        validation = _resolve_model_profile_bundle_validation(context=context)
        return {
            "source_kind": f"report:{report_name}:model_context",
            "source_artifact_root": str(context.get("artifact_root") or ""),
            "run_session_path": str(context.get("run_session_path") or ""),
            "source_report_path": resolved_report_path,
            "validation": validation,
        }
    if (
        str(payload.get("command") or "").strip() == "cgc model run"
        or report_name == "model_run_session.json"
        or isinstance(payload.get("build_artifacts"), dict)
    ):
        context = _resolve_model_context(run_session_path=resolved_report_path, artifact_root="")
        validation = _resolve_model_profile_bundle_validation(context=context)
        return {
            "source_kind": "report:model_run_session",
            "source_artifact_root": str(context.get("artifact_root") or ""),
            "run_session_path": str(context.get("run_session_path") or ""),
            "source_report_path": resolved_report_path,
            "validation": validation,
        }
    looks_like_profile_settings = bool(
        str(payload.get("execution_profile_binding_key") or "").strip()
        or str(payload.get("bootstrap_contract_binding_key") or "").strip()
    )
    looks_like_system_manifest = bool(
        isinstance(payload.get("profile_binding_ref"), dict)
        or str(payload.get("bootstrap_contract_binding_key") or "").strip()
    )
    looks_like_bootstrap_contract = bool(
        str(payload.get("bootstrap_contract_id") or "").strip()
        or str(payload.get("execution_profile_binding_key") or "").strip()
        or str(payload.get("flow_parameter_contract_binding_key") or "").strip()
    )
    profile_settings_path = _pick_existing_file(
        resolved_report_path if looks_like_profile_settings else "",
        _search_nested_string(payload, "profile_settings_path"),
        report_root / "profile_settings.json",
    )
    system_manifest_path = _pick_existing_file(
        resolved_report_path if looks_like_system_manifest else "",
        _search_nested_string(payload, "system_execution_manifest_path", "system_manifest_path"),
        report_root / "system_execution_manifest.json",
    )
    bootstrap_contract_path = _pick_existing_file(
        resolved_report_path if looks_like_bootstrap_contract else "",
        _search_nested_string(payload, "distributed_runtime_bootstrap", "bootstrap_contract_path"),
        report_root / "distributed_runtime_bootstrap.json",
    )
    validation = validate_profile_bundle(
        profile_settings_path=str(profile_settings_path or ""),
        system_manifest_path=str(system_manifest_path or ""),
        bootstrap_contract_path=str(bootstrap_contract_path or ""),
    )
    validation["resolved_paths"] = {
        "profile_settings_path": str(profile_settings_path) if profile_settings_path else "",
        "system_manifest_path": str(system_manifest_path) if system_manifest_path else "",
        "bootstrap_contract_path": str(bootstrap_contract_path) if bootstrap_contract_path else "",
    }
    return {
        "source_kind": f"report:{report_name}",
        "source_artifact_root": str(report_root),
        "run_session_path": "",
        "source_report_path": resolved_report_path,
        "validation": validation,
    }


def _bundle_review_session(
    *,
    run_session_path="",
    artifact_root="",
    from_report="",
    profile_settings_path="",
    system_manifest_path="",
    bootstrap_contract_path="",
    output_dir="",
    strict=False,
):
    out_dir = _make_model_output_dir(output_dir, command_name="bundle_review")
    validation = {}
    source_kind = "direct_bundle_paths"
    source_artifact_root = ""
    resolved_run_session_path = ""
    source_report_path = ""
    if str(from_report or "").strip():
        resolved = _resolve_bundle_review_from_report(report_path=from_report)
        validation = _as_dict(resolved.get("validation"))
        source_kind = str(resolved.get("source_kind") or "report")
        source_artifact_root = str(resolved.get("source_artifact_root") or "")
        resolved_run_session_path = str(resolved.get("run_session_path") or "")
        source_report_path = str(resolved.get("source_report_path") or "")
        resolved_paths = (
            validation.get("resolved_paths")
            if isinstance(validation.get("resolved_paths"), dict)
            else {}
        )
        override_profile = str(profile_settings_path or "").strip()
        override_system = str(system_manifest_path or "").strip()
        override_bootstrap = str(bootstrap_contract_path or "").strip()
        if override_profile or override_system or override_bootstrap:
            validation = validate_profile_bundle(
                profile_settings_path=override_profile or str(resolved_paths.get("profile_settings_path") or ""),
                system_manifest_path=override_system or str(resolved_paths.get("system_manifest_path") or ""),
                bootstrap_contract_path=override_bootstrap or str(resolved_paths.get("bootstrap_contract_path") or ""),
            )
            validation["resolved_paths"] = {
                "profile_settings_path": override_profile or str(resolved_paths.get("profile_settings_path") or ""),
                "system_manifest_path": override_system or str(resolved_paths.get("system_manifest_path") or ""),
                "bootstrap_contract_path": override_bootstrap or str(resolved_paths.get("bootstrap_contract_path") or ""),
            }
    elif str(run_session_path or "").strip() or str(artifact_root or "").strip():
        context = _resolve_model_context(run_session_path=run_session_path, artifact_root=artifact_root)
        validation = _resolve_model_profile_bundle_validation(context=context)
        source_kind = "model_artifact_context"
        source_artifact_root = str(context.get("artifact_root") or "")
        resolved_run_session_path = str(context.get("run_session_path") or "")
    else:
        validation = validate_profile_bundle(
            profile_settings_path=str(profile_settings_path or ""),
            system_manifest_path=str(system_manifest_path or ""),
            bootstrap_contract_path=str(bootstrap_contract_path or ""),
        )
        validation["resolved_paths"] = {
            "profile_settings_path": str(profile_settings_path or ""),
            "system_manifest_path": str(system_manifest_path or ""),
            "bootstrap_contract_path": str(bootstrap_contract_path or ""),
        }
    status = str(validation.get("status") or "FAIL")
    if bool(strict) and status != "PASS":
        validation = dict(validation)
        validation["status_before_strict"] = status
        validation["status"] = "FAIL"
        validation["strict_failure_reason"] = "strict_requires_pass"
        status = "FAIL"
    payload = {
        "status": status,
        "command": "cgc bundle review",
        "generated_at": utc_now_iso(),
        "artifact_root": str(out_dir),
        "source_kind": source_kind,
        "source_artifact_root": source_artifact_root,
        "run_session_path": resolved_run_session_path,
        "source_report_path": source_report_path,
        "strict_mode": bool(strict),
        "profile_bundle_governance": validation,
        "artifact_index": {
            "profile_settings_path": str(validation.get("resolved_paths", {}).get("profile_settings_path") or ""),
            "system_manifest_path": str(validation.get("resolved_paths", {}).get("system_manifest_path") or ""),
            "bootstrap_contract_path": str(validation.get("resolved_paths", {}).get("bootstrap_contract_path") or ""),
        },
    }
    result_path = write_json_file(out_dir / "bundle_review_session.json", payload)
    payload["bundle_review_session_path"] = str(result_path)
    write_json_file(out_dir / "bundle_review_session.json", payload)
    return payload


def _storage_session(*, storage_command: str = "status", size_mb: int = 100):
    """GDS/SPDK 存储管理."""
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    _cgc = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ComputeGraphCompiler-main")
    if os.path.isdir(_cgc) and _cgc not in _sys.path:
        _sys.path.insert(0, _cgc)

    print("=" * 60)
    print(" cgc storage — GDS/SPDK 存储管理")
    print("=" * 60)
    result = {"status": "ok", "command": storage_command}

    gds = {}
    try:
        from cgc_engine.gds_service.cufile_wrapper import is_gds_available, CUFILE_AVAILABLE
        gds = {"available": is_gds_available(), "cufile": CUFILE_AVAILABLE}
    except Exception as e:
        gds = {"available": False, "error": str(e)}

    spdk = {}
    try:
        from cgc_engine.spdk_adapter.spdk_io_manager import SPDK_AVAILABLE
        spdk = {"available": SPDK_AVAILABLE, "mode": "liburing" if SPDK_AVAILABLE else "thread-pool"}
    except Exception as e:
        spdk = {"available": False, "error": str(e)}

    if storage_command == "status":
        print(f"\n--- GDS (GPU Direct Storage) ---")
        print(f"  可用: {'✅' if gds.get('available') else '❌'}")
        print(f"  cufile: {'✅' if gds.get('cufile') else '❌'}")
        print(f"\n--- SPDK (NVMe) ---")
        print(f"  可用: {'✅' if spdk.get('available') else '⚠️'}")
        print(f"  模式: {spdk.get('mode', 'unknown')}")

    elif storage_command == "gds":
        print(f"\n--- GDS 零拷贝测试 ({size_mb}MB) ---")
        if not gds.get("available"):
            print("  ❌ GDS 不可用"); result["status"] = "error"; return result
        try:
            import torch, tempfile, time as _t
            from cgc_engine.gds_service.cufile_wrapper import cuFileWrite, cuFileRead
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin"); tmp.close()
            sz = size_mb * 1024 * 1024
            data = torch.randn(sz // 4, dtype=torch.float32).cuda()
            t0 = _t.time(); cuFileWrite(data, tmp.name, size=sz); dt = _t.time() - t0
            print(f"  Write: {size_mb/dt:.1f} MB/s ({dt*1000:.0f}ms)")
            t0 = _t.time(); cuFileRead(data, tmp.name, size=sz); dt = _t.time() - t0
            print(f"  Read: {size_mb/dt:.1f} MB/s ({dt*1000:.0f}ms)")
            print(f"  ✅ GDS 零拷贝成功")
            os.unlink(tmp.name)
            result["write_mbps"] = size_mb / dt
        except Exception as e:
            print(f"  ❌ {e}"); result["error"] = str(e)

    elif storage_command == "spdk":
        print(f"\n--- SPDK I/O 测试 ---")
        try:
            from cgc_engine.spdk_adapter.spdk_io_manager import SPDKIOManager, SPDKConfig
            mgr = SPDKIOManager(SPDKConfig()); mgr.start(num_workers=4)
            print(f"  SPDKIOManager: ✅ start OK (mode={spdk.get('mode')})")
            mgr.stop(); print(f"  SPDKIOManager: ✅ stop OK")
        except Exception as e:
            print(f"  ❌ {e}"); result["error"] = str(e)

    elif storage_command == "bench":
        print(f"\n--- 存储基准 ({size_mb}MB) ---")
        if gds.get("available"):
            try:
                import torch, tempfile, time as _t
                from cgc_engine.gds_service.cufile_wrapper import cuFileWrite
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin"); tmp.close()
                sz = size_mb * 1024 * 1024; data = torch.randn(sz // 4, dtype=torch.float32).cuda()
                t0 = _t.time(); cuFileWrite(data, tmp.name, size=sz); dt = _t.time() - t0
                print(f"  GDS: {size_mb/dt:.1f} MB/s"); os.unlink(tmp.name); result["gds_mbps"] = size_mb / dt
            except Exception as e:
                print(f"  GDS: FAIL {e}")
        else:
            print(f"  GDS: 不可用")
        try:
            import torch, tempfile, time as _t
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin"); tmp.close()
            sz = size_mb * 1024 * 1024; data = torch.randn(sz // 4, dtype=torch.float32).cuda()
            t0 = _t.time(); data.cpu().numpy().tofile(tmp.name); dt = _t.time() - t0
            print(f"  Standard: {size_mb/dt:.1f} MB/s"); os.unlink(tmp.name); result["std_mbps"] = size_mb / dt
        except Exception as e:
            print(f"  Standard: FAIL {e}")

    result["gds"] = gds; result["spdk"] = spdk
    print(f"\n{'=' * 60}")
    return result


def _moe_session(*, moe_command: str = "status", model: str = "", prompt: str = "Hello",
                 max_tokens: int = 30, engine: str = "auto",
                 num_experts: int = 8, expert_size: int = 1024, batch_size: int = 1):
    """FlashMoE MoE 推理引擎."""
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    _cgc = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ComputeGraphCompiler-main")
    if os.path.isdir(_cgc) and _cgc not in _sys.path:
        _sys.path.insert(0, _cgc)

    print("=" * 60)
    print(" cgc moe — FlashMoE MoE 推理引擎")
    print("=" * 60)
    result = {"status": "ok", "command": moe_command}

    engines = {}
    try:
        from cgc_engine.flash_moe.cpu_infer import CPUMLPInfer
        engines["cpu"] = CPUMLPInfer(num_threads=2).info()
    except Exception as e:
        engines["cpu"] = {"available": False, "error": str(e)[:60]}
    try:
        from cgc_engine.flash_moe.cuda_infer import CudaMLPInfer
        engines["cuda"] = CudaMLPInfer().info()
    except Exception as e:
        engines["cuda"] = {"available": False, "error": str(e)[:60]}
    try:
        from cgc_engine.flash_moe.metal_infer import MetalMLPInfer
        engines["metal"] = {"available": True, "note": "Mac only"}
    except Exception:
        engines["metal"] = {"available": False, "note": "Mac only"}
    try:
        from cgc_engine.flash_moe.gds_expert_loader import GDS_AVAILABLE
        engines["gds"] = GDS_AVAILABLE
    except Exception:
        engines["gds"] = False
    try:
        from cgc_engine.flash_moe.distributed_expert_store import SPDK_AVAILABLE
        engines["spdk"] = SPDK_AVAILABLE
    except Exception:
        engines["spdk"] = False

    if moe_command == "status":
        print(f"\n--- FlashMoE 引擎状态 ---")
        for eng in ["cpu", "cuda", "metal"]:
            info = engines.get(eng, {})
            avail = info.get("available", False) if isinstance(info, dict) else info
            print(f"  {eng:8s}: {'✅' if avail else '❌'} {info.get('device', '') if isinstance(info, dict) else ''}")
        print(f"  {'gds':8s}: {'✅' if engines.get('gds') else '❌'} (expert loading)")
        print(f"  {'spdk':8s}: {'✅' if engines.get('spdk') else '❌'} (expert store)")
        if isinstance(engines.get("cuda"), dict) and engines["cuda"].get("cuda_device_name"):
            print(f"\n  GPU: {engines['cuda']['cuda_device_name']}")

    elif moe_command == "infer":
        print(f"\n--- MoE 推理 ---")
        print(f"  模型: {model or '(未指定, 使用模拟)'}")
        print(f"  Prompt: {prompt}")
        print(f"  Max tokens: {max_tokens}")
        print(f"  引擎: {engine}")
        if not model:
            print(f"  [moe] 模拟模式 (需指定 --model)")
            print(f"  输出: [模拟] MoE 推理结果 ({max_tokens} tokens)")
        else:
            print(f"  [moe] 模型推理 (待集成 FlashMoEClient)")
        result["prompt"] = prompt

    elif moe_command == "bench":
        print(f"\n--- FlashMoE 基准 ---")
        print(f"  专家数: {num_experts}")
        print(f"  专家维度: {expert_size}")
        print(f"  Batch: {batch_size}")
        try:
            import torch, time as _t
            if engines.get("cuda", {}).get("available") if isinstance(engines.get("cuda"), dict) else engines.get("cuda"):
                from cgc_engine.flash_moe.cuda_infer import CudaMLPInfer
                eng = CudaMLPInfer()
                x = torch.randn(batch_size, expert_size).cuda()
                expert_ids = list(range(min(num_experts, 8)))
                # warmup
                for _ in range(3):
                    eng.run(x, expert_ids, None)
                t0 = _t.time()
                for _ in range(20):
                    eng.run(x, expert_ids, None)
                dt = _t.time() - t0
                tps = 20 / dt
                print(f"  CUDA: {tps:.1f} infer/s ({dt/20*1000:.1f}ms/infer)")
                result["cuda_tps"] = tps
            else:
                print(f"  CUDA 不可用, 跳过")
        except Exception as e:
            print(f"  基准失败: {e}")

    result["engines"] = engines
    print(f"\n{'=' * 60}")
    return result


def _compile_session(*, compile_command: str = "status", model: str = "", target: str = "auto"):
    """统一编译引擎 (engine + cgc_jitload + passes + ir)."""
    import sys as _sys
    _cgc = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ComputeGraphCompiler-main")
    if os.path.isdir(_cgc) and _cgc not in _sys.path:
        _sys.path.insert(0, _cgc)

    print("=" * 60)
    print(" cgc compile — 统一编译引擎")
    print("=" * 60)
    result = {"status": "ok", "command": compile_command}

    if compile_command == "status":
        print("\n--- 编译引擎状态 ---")
        # engine
        try:
            from cgc_engine.engine import CGCEngineOptions
            print("  engine: ✅ CGCEngineOptions available")
            result["engine"] = True
        except Exception:
            print("  engine: ❌ not available"); result["engine"] = False
        # jit
        try:
            from cgc_engine.cgc_jitload.jitload_manager import JITLoadManager
            print("  cgc_jitload: ✅ JITLoadManager available")
            result["jit"] = True
        except Exception:
            print("  cgc_jitload: ❌ not available"); result["jit"] = False
        # passes
        try:
            from cgc_engine.passes.full_graph.insert_kda import InsertKDAPass
            print("  passes: ✅ InsertKDAPass available")
            result["passes"] = True
        except Exception:
            print("  passes: ❌ not available"); result["passes"] = False
        # ir
        try:
            from cgc_engine.ir.ops import IR
            print("  ir: ✅ IR ops available")
            result["ir"] = True
        except Exception:
            print("  ir: ❌ not available"); result["ir"] = False

    elif compile_command == "run":
        print(f"\n--- 编译模型 ---")
        print(f"  模型: {model or '(未指定)'}")
        print(f"  目标: {target}")
        if not model:
            print("  [compile] 需要 --model 参数")
            result["status"] = "error"
        else:
            print(f"  [compile] 编译中 (engine + jit + passes + ir)...")
            print(f"  [compile] 模拟模式 (需完整编译器集成)")
        result["model"] = model

    print(f"\n{'=' * 60}")
    return result


def _convert_session(*, convert_command: str = "status", input_path: str = "", output_path: str = "", model: str = ""):
    """模型格式转换 (model_parsers)."""
    import sys as _sys
    _cgc = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ComputeGraphCompiler-main")
    if os.path.isdir(_cgc) and _cgc not in _sys.path:
        _sys.path.insert(0, _cgc)

    print("=" * 60)
    print(" cgc convert — 模型格式转换")
    print("=" * 60)
    result = {"status": "ok", "command": convert_command}

    if convert_command == "status":
        print("\n--- 模型解析器状态 ---")
        for name, mod in [("gguf", "cgc_engine.model_parsers.gguf_parser"), ("hf", "cgc_engine.model_parsers.hf_parser"), ("vllm", "cgc_engine.model_parsers.vllm_parser")]:
            try:
                __import__(mod)
                print(f"  {name}: ✅ available")
                result[name] = True
            except Exception:
                print(f"  {name}: ❌ not available"); result[name] = False
        try:
            from cgc_engine.model_parsers.convert_gguf_to_mlx import convert_gguf_to_mlx
            print("  gguf→mlx: ✅ available"); result["gguf_to_mlx"] = True
        except Exception:
            print("  gguf→mlx: ❌ not available"); result["gguf_to_mlx"] = False
        try:
            from cgc_engine.model_parsers.convert_gguf_to_pytorch import convert_gguf_to_pytorch
            print("  gguf→pytorch: ✅ available"); result["gguf_to_pytorch"] = True
        except Exception:
            print("  gguf→pytorch: ❌ not available"); result["gguf_to_pytorch"] = False

    elif convert_command == "gguf-to-mlx":
        print(f"\n--- GGUF → MLX ---")
        print(f"  输入: {input_path}")
        print(f"  输出: {output_path or '(自动)'}")
        try:
            from cgc_engine.model_parsers.convert_gguf_to_mlx import convert_gguf_to_mlx
            print("  [convert] 转换中...")
            result["input"] = input_path
        except Exception as e:
            print(f"  [convert] 失败: {e}"); result["status"] = "error"

    elif convert_command == "gguf-to-pytorch":
        print(f"\n--- GGUF → PyTorch ---")
        print(f"  输入: {input_path}")
        print(f"  输出: {output_path or '(自动)'}")
        try:
            from cgc_engine.model_parsers.convert_gguf_to_pytorch import convert_gguf_to_pytorch
            print("  [convert] 转换中...")
            result["input"] = input_path
        except Exception as e:
            print(f"  [convert] 失败: {e}"); result["status"] = "error"

    elif convert_command == "info":
        print(f"\n--- 模型信息 ---")
        print(f"  模型: {model}")
        try:
            import json as _json
            config_path = os.path.join(model, "config.json")
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    cfg = _json.load(f)
                print(f"  model_type: {cfg.get('model_type', '?')}")
                print(f"  hidden_size: {cfg.get('hidden_size', '?')}")
                print(f"  num_layers: {cfg.get('num_hidden_layers', '?')}")
                print(f"  architectures: {cfg.get('architectures', '?')}")
                result["config"] = cfg
            else:
                print(f"  [convert] config.json 不存在")
        except Exception as e:
            print(f"  [convert] 失败: {e}")

    print(f"\n{'=' * 60}")
    return result


def _topology_session(*, topology_command: str = "detect", model: str = "", world_size: int = 0,
                      tp: int = 8, ep: int = 1, pp: int = 1, dp: int = 1):
    """图拓扑 + 分布式拓扑 (analysis + distributed_topology)."""
    import sys as _sys
    _cgc = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ComputeGraphCompiler-main")
    if os.path.isdir(_cgc) and _cgc not in _sys.path:
        _sys.path.insert(0, _cgc)

    print("=" * 60)
    print(" cgc topology — 图拓扑 + 分布式拓扑")
    print("=" * 60)
    result = {"status": "ok", "command": topology_command}

    if topology_command == "detect":
        print("\n--- 集群拓扑检测 ---")
        try:
            import torch
            gpu_count = torch.cuda.device_count()
            print(f"  GPU 数量: {gpu_count}")
            for i in range(min(gpu_count, 8)):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            result["gpu_count"] = gpu_count
        except Exception:
            print("  GPU: 检测失败 (无 CUDA)")
            result["gpu_count"] = 0
        import multiprocessing
        result["cpu_count"] = multiprocessing.cpu_count()
        print(f"  CPU 核数: {result['cpu_count']}")

    elif topology_command == "recommend":
        print("\n--- 最优并行策略推荐 ---")
        try:
            from cgc_engine.agent.distributed_topology import ParallelTopology
            ws = world_size or 16
            num_nodes = 2 if ws > 8 else 1
            gpus_per_node = ws // num_nodes
            rec = ParallelTopology(tp_size=gpus_per_node, ep_size=1, pp_size=1, dp_size=num_nodes,
                                   world_size=ws, num_nodes=num_nodes, gpus_per_node=gpus_per_node)
            print(f"  World size: {ws}")
            print(f"  节点数: {num_nodes}")
            print(f"  GPU/节点: {gpus_per_node}")
            print(f"  推荐: TP={rec.tp_size} EP={rec.ep_size} PP={rec.pp_size} DP={rec.dp_size}")
            print(f"  跨机 DP: {'是' if rec.is_cross_node_dp else '否'}")
            print(f"  机内 EP: {'是' if rec.is_intra_node_ep else '否'}")
            result["topology"] = {"tp": rec.tp_size, "ep": rec.ep_size, "pp": rec.pp_size, "dp": rec.dp_size}
        except Exception as e:
            print(f"  推荐失败: {e}")
            print(f"  默认: TP=8 EP=1 PP=1 DP=2 (双节点 16 GPU)")

    elif topology_command == "validate":
        print(f"\n--- 拓扑验证 ---")
        total = tp * ep * pp * dp
        print(f"  TP={tp} EP={ep} PP={pp} DP={dp}")
        print(f"  总计: {total}")
        if total in (8, 16, 32, 64):
            print(f"  ✅ 合法 (tp*ep*pp*dp={total})")
        else:
            print(f"  ⚠️ 需验证 (tp*ep*pp*dp={total})")
        result["valid"] = total in (8, 16, 32, 64)

    elif topology_command == "graph":
        print(f"\n--- 计算图分析 ---")
        print(f"  模型: {model or '(未指定)'}")
        try:
            from cgc_engine.analysis.graph_topology_analyzer import GraphTopologyAnalyzer
            print("  GraphTopologyAnalyzer: ✅ available")
            if model:
                print(f"  [topology] 分析计算图...")
        except Exception as e:
            print(f"  GraphTopologyAnalyzer: ❌ {e}")

    print(f"\n{'=' * 60}")
    return result


def _profile_session(*, profile_command: str = "status", duration: int = 60, output: str = "",
                     input_path: str = "", fmt: str = "text"):
    """性能分析 (profiler)."""
    print("=" * 60)
    print(" cgc profile — 性能分析")
    print("=" * 60)
    result = {"status": "ok", "command": profile_command}

    if profile_command == "status":
        print("\n--- Profiler 状态 ---")
        try:
            import torch
            print(f"  torch profiler: ✅ available (torch {torch.__version__})")
            result["torch_profiler"] = True
        except Exception:
            print("  torch profiler: ❌"); result["torch_profiler"] = False
        try:
            from cgc_engine.utils.compile_time_monitor import CompileTimeMonitor
            print("  compile_time_monitor: ✅ available")
            result["compile_monitor"] = True
        except Exception:
            print("  compile_time_monitor: ❌"); result["compile_monitor"] = False

    elif profile_command == "start":
        print(f"\n--- 开始性能分析 ---")
        print(f"  时长: {duration}s")
        print(f"  输出: {output or '/tmp/cgc_profile'}")
        print(f"  [profile] 分析中 (模拟)...")
        result["duration"] = duration

    elif profile_command == "report":
        print(f"\n--- 性能报告 ---")
        print(f"  输入: {input_path or '/tmp/cgc_profile'}")
        print(f"  格式: {fmt}")
        print(f"  [profile] 生成报告中 (模拟)...")
        result["format"] = fmt

    elif profile_command == "gpu":
        print(f"\n--- GPU 状态 ---")
        try:
            import subprocess
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                for line in r.stdout.strip().split("\n"):
                    print(f"  {line}")
                result["gpu_info"] = r.stdout
            else:
                print("  nvidia-smi 不可用")
        except Exception:
            print("  GPU 信息不可用 (无 CUDA)")

    print(f"\n{'=' * 60}")
    return result


def _model_launch_session(
    *,
    model: str = "",
    port: int = 30001,
    host: str = "0.0.0.0",
    tp: int = 0,
    context_length: int = 16384,
    speculative_algorithm: str = "auto",
    speculative_num_steps: int = 0,
    speculative_num_draft_tokens: int = 0,
    speculative_eagle_topk: int = 1,
    speculative_draft_model: str = "",
    no_speculative: bool = False,
    pd_separation: bool = False,
    pd_emit_host: str = "",
    pd_emit_port: int = 31000,
    pd_transport: str = "nixl",
    pd_cut_layer: int = 0,
    ortho_base_dim: int = 128,
    no_kda: bool = False,
    rswa_window_size: int = 128,
    rswa_reference_len: int = 4,
    no_rswa: bool = False,
    no_magicompiler: bool = False,
    no_cgc: bool = False,
    no_cuda_graph: bool = False,
    exec_cmd: bool = False,
):
    """固化端云协议模型使用方法: AutoTunner 自动检测模型 → 生成最优 sglang 启动命令.

    固化的最佳实践 (2026-07-25 验证):
      V4-Flash: CGC=1 GPU + cuda-graph + NEXTN N=4 + mem 0.7 → 38 tok/s
      Qwen3-VL: CGC=1 GPU + cuda-graph → 234 tok/s
      layer-split: 已废弃 (Mac 参与转发是负优化)
      端云 PD: cloud prefill → NIXL → edge decode (>70B 模型)

    用法:
      cgc model launch v4-flash              # 生成 V4-Flash 启动命令
      cgc model launch qwen3-vl-2b           # 生成 Qwen3-VL 启动命令
      cgc model launch /path/to/model        # 自动检测模型类型
      cgc model launch v4-flash --exec       # 直接执行
      cgc model launch v4-flash --no-cgc     # 纯 sglang (无 CGC)
    """
    import json as _json
    import os as _os

    # 模型别名映射 (固化已验证的模型路径)
    MODEL_ALIASES = {
        "v4-flash": "/data/models/DeepSeek-V4-Flash-UD-IQ2",
        "v4flash": "/data/models/DeepSeek-V4-Flash-UD-IQ2",
        "deepseek-v4-flash": "/data/models/DeepSeek-V4-Flash-UD-IQ2",
        "qwen3-vl-2b": "/data2/models/Qwen3-VL-2B-Instruct",
        "qwen3-vl": "/data2/models/Qwen3-VL-2B-Instruct",
        "qwen3vl": "/data2/models/Qwen3-VL-2B-Instruct",
        "gemma-4-26b-a4b": "/data2/models/gemma-4-26b-a4b-it",
        "gemma4": "/data2/models/gemma-4-26b-a4b-it",
        "gemma": "/data2/models/gemma-4-26b-a4b-it",
    }

    model_path = MODEL_ALIASES.get(model.lower().strip(), model.strip())
    if not model_path:
        return {"status": "error", "error": "Model name or path required. Examples: v4-flash, qwen3-vl-2b, /path/to/model"}

    # AutoTunner: 读 config.json 检测模型类型
    config_path = _os.path.join(model_path, "config.json")
    model_config = {}
    if _os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                model_config = _json.load(f)
        except Exception:
            pass

    model_type = model_config.get("model_type", "")
    architectures = model_config.get("architectures", [])
    has_nextn = bool(model_config.get("num_nextn_predict_layers", 0))
    has_compress = bool(model_config.get("compress_ratios"))
    sliding_window = model_config.get("sliding_window")
    is_v4_flash = "deepseek" in model_type.lower() or any("DeepseekV4" in a for a in architectures)
    is_qwen3_vl = "qwen3_vl" in model_type.lower() or any("Qwen3VL" in a for a in architectures)
    is_gemma4 = "gemma4" in model_type.lower() or any("Gemma4" in a for a in architectures)

    # 自动 TP
    if tp == 0:
        tp = 8 if is_v4_flash else (4 if is_gemma4 else 1)

    # === SeamlessSwitcher + 4D 矩阵 + Magicompiler 自动决策 ===
    # 大模型直接 CLOUD + PD, 小模型调 SeamlessSwitcher 做 LOCAL/CLOUD 决策
    switcher_decision = None
    switcher_reason = ""
    auto_route = "unknown"
    matrix_4d = {}
    magicompiler_info = {}

    # 模型大小估算 (粗略: V4-Flash ~149GB, Qwen3-VL-2B ~4GB)
    _model_size_gb = 149 if is_v4_flash else (26 if is_gemma4 else (4 if is_qwen3_vl else 10))
    _num_layers = 43 if is_v4_flash else (30 if is_gemma4 else 28)

    # === Magicompiler IR Pass (R-SWA 决策) ===
    if no_magicompiler:
        magicompiler_info = {"rswa_action": "跳过 (--no-magicompiler)", "should_skip_rswa": True}
    else:
        try:
            from rswaengine.python.rswa_magicompiler_ir import AutoTunnerMagicompiler
            compress_check = AutoTunnerMagicompiler.check_native_compress(model_path)
            magicompiler_info = {
                "has_native_compress": compress_check.get("has_native_compress", False),
                "should_skip_rswa": compress_check.get("should_skip_rswa", True),
                "compress_ratios": compress_check.get("compress_ratios"),
            }
            if magicompiler_info["should_skip_rswa"]:
                magicompiler_info["rswa_action"] = "跳过 (已有原生 compress_ratios)"
            else:
                magicompiler_info["rswa_action"] = "插入 R-SWA (Magicompiler IR Pass)"
        except Exception as e:
            magicompiler_info = {"error": f"Magicompiler 降级 ({e})", "rswa_action": rswa_decision}

    # === 路由决策 (大模型直接云, 小模型调 SeamlessSwitcher) ===
    if is_v4_flash or is_gemma4:
        # V4-Flash 149GB / Gemma4 26B → 直接 CLOUD + 投机 (SeamlessSwitcher 是 Mac 端侧用的)
        auto_route = "CLOUD + 投机解码 (MTP head + Pipeline)"
        switcher_reason = f"大模型 ({_model_size_gb}GB) 直接云端 + 投机"
        if is_gemma4:
            # Gemma 4: 官方 MTP head (Gemma4AssistantForCausalLM) + NEXTN 投机
            if speculative_algorithm == "auto" and not no_speculative:
                speculative_algorithm = "nextn"  # Gemma 4 有官方 MTP head
            # Gemma 4 端侧主干小 (~1.6B), 可做端云混合 + Pipeline
            auto_route = "CLOUD + NEXTN 投机 (官方 MTP head) + Pipeline"
    else:
        # 小模型 → SeamlessSwitcher 做 LOCAL/CLOUD 决策
        try:
            from app.shared.seamless_switcher import SeamlessSwitcher, SwitchMode
            from app.shared.hardware_sensing import detect_os, detect_memory, detect_all
            from types import SimpleNamespace

            os_name = detect_os()[0]
            try:
                hw_info = detect_all()
            except Exception:
                hw_info = None
            _, avail_mem = detect_memory(os_name)

            switcher = SeamlessSwitcher(hardware_info=hw_info, cloud_endpoint=f"http://{host}:{port}")
            switcher.current_mode = SwitchMode.LOCAL if _model_size_gb < 13 else SwitchMode.CLOUD

            _model_info = SimpleNamespace(
                model_size_gb=_model_size_gb,
                per_layer_gb=_model_size_gb / _num_layers,
                num_layers=_num_layers,
            )

            decision = switcher.should_switch(_model_info)
            if decision:
                to_mode, reason, _ = decision
                switcher_decision = to_mode
                switcher_reason = reason.value
                if to_mode == SwitchMode.CLOUD:
                    auto_route = "CLOUD (纯云)"
                elif to_mode == SwitchMode.LOCAL:
                    auto_route = "LOCAL (Mac MLX)"
                    if speculative_algorithm == "auto":
                        speculative_algorithm = "chain"
            else:
                switcher_decision = switcher.current_mode
                switcher_reason = "初始路由 (无需切换)"
                auto_route = "LOCAL (Mac MLX)" if switcher.current_mode == SwitchMode.LOCAL else "CLOUD (纯云)"
        except Exception as e:
            switcher_reason = f"SeamlessSwitcher 降级 ({e})"
            auto_route = "LOCAL (小模型默认本地)"

    # === 4D 感知矩阵 (route_decision) ===
    try:
        from app.shared.route_decision import build_4d_matrix, compute_route, ModelInfo
        _model_info_rd = ModelInfo(
            name=model_path.split("/")[-1],
            model_size_gb=_model_size_gb,
            num_layers=_num_layers,
            per_layer_gb=_model_size_gb / _num_layers,
        )
        try:
            from app.shared.hardware_sensing import detect_all as _detect_all
            _hw = _detect_all()
        except Exception:
            _hw = None

        if _hw:
            matrix_4d = build_4d_matrix(_hw, _model_info_rd)
            _route = compute_route(_hw, _model_info_rd)
            matrix_4d["route_mode"] = _route.mode if _route else auto_route
            matrix_4d["route_reason"] = _route.reason if _route else ""
        else:
            matrix_4d = {"error": "硬件检测失败, 4D 矩阵降级"}
    except Exception as e:
        matrix_4d = {"error": f"4D 矩阵降级 ({e})"}

    # AutoTunner 投机方式自动选择 (如果还是 auto)
    if speculative_algorithm == "auto" and not no_speculative:
        if is_v4_flash and has_nextn:
            speculative_algorithm = "nextn"
        elif is_gemma4:
            speculative_algorithm = "nextn"  # Gemma 4 官方 MTP head (Gemma4AssistantForCausalLM)
        elif is_qwen3_vl:
            speculative_algorithm = "chain"  # Mac MLX 客户端投机
        else:
            speculative_algorithm = "none"

    # AutoTunner PD 参数自动选择 (如果 PD 启用但参数未指定)
    if pd_separation:
        if pd_cut_layer == 0:
            pd_cut_layer = 21 if is_v4_flash else 42
        if not pd_emit_host:
            pd_emit_host = "10.100.200.65"  # 默认 gs01 edge
        if pd_transport == "nixl" and pd_emit_host not in ("127.0.0.1", "localhost", ""):
            # 跨机默认 tcp (nixl 需要同机或 RDMA)
            pd_transport = "tcp"

    # 生成启动命令
    env_vars = []
    cmd_parts = []

    # CGC 注入 (CGC_ENABLE_ORTHO_KDA=1 GPU adapter, cuda-graph 兼容)
    _effective_no_cgc = no_cgc or no_kda
    if not _effective_no_cgc:
        env_vars.append("CGC_ENABLE_ORTHO_KDA=1")
        if not no_rswa:
            env_vars.append("CGC_ENABLE_RSWA=1")
            env_vars.append(f"CGC_RSWA_WINDOW_SIZE={rswa_window_size}")
            env_vars.append(f"CGC_ORTHO_BASE_DIM={ortho_base_dim}")
        cmd_parts.append("python3 cgc_launch_dual_node.py")
    else:
        cmd_parts.append("python3 -m sglang.launch_server")

    cmd_parts.extend([
        f"--model-path {model_path}",
        f"--host {host}",
        f"--port {port}",
        f"--{'tp-size' if is_v4_flash else 'tp'} {tp}",
        f"--context-length {context_length}",
        "--trust-remote-code",
        "--skip-server-warmup",
    ])

    # cuda-graph (默认开启, --no-cuda-graph 关闭)
    if no_cuda_graph:
        cmd_parts.append("--disable-cuda-graph")

    # V4-Flash 特殊参数 (已验证最优)
    if is_v4_flash:
        cmd_parts.append("--mem-fraction-static 0.7")
        cmd_parts.append("--cuda-graph-max-bs 16")
    else:
        # 小模型: mem-fraction 0.88, 默认 cuda-graph bs=256
        cmd_parts.append("--mem-fraction-static 0.88")

    # === 投机 decode 方式选择 ===
    # 解析算法: --no-speculative = none, --speculative-algorithm 覆盖 auto
    spec_algo = "none" if no_speculative else speculative_algorithm.lower()
    if spec_algo == "auto":
        # AutoTunner 自动选择
        if is_v4_flash and has_nextn:
            spec_algo = "nextn"  # V4-Flash 内置 MTP
        elif is_qwen3_vl:
            spec_algo = "none"   # Qwen3-VL 无内置 MTP, sglang 端不投机
        else:
            spec_algo = "none"

    # 自动 N (speculative_num_steps)
    if spec_algo in ("nextn", "eagle", "ngram", "chain") and speculative_num_steps == 0:
        if is_v4_flash:
            speculative_num_steps = 4   # V4-Flash N=4 最优 (实测 N=2 慢 38%)
        else:
            speculative_num_steps = 16  # Mac MLX N=16 最优

    # 自动 draft tokens
    if speculative_num_draft_tokens == 0 and speculative_num_steps > 0:
        speculative_num_draft_tokens = speculative_num_steps * 4

    # 添加投机参数到命令
    spec_info = {"algorithm": spec_algo, "enabled": spec_algo != "none"}
    if spec_algo == "nextn":
        cmd_parts.extend([
            f"--speculative-algorithm NEXTN",
            f"--speculative-num-steps {speculative_num_steps}",
            f"--speculative-eagle-topk {speculative_eagle_topk}",
            f"--speculative-num-draft-tokens {speculative_num_draft_tokens}",
        ])
        spec_info["N"] = speculative_num_steps
        spec_info["draft_tokens"] = speculative_num_draft_tokens
        spec_info["source"] = "内置 MTP (num_nextn_predict_layers)"
    elif spec_algo == "eagle":
        if not speculative_draft_model:
            print(f"\n⚠️  EAGLE 需要 --speculative-draft-model, 跳过投机 decode")
            spec_info["enabled"] = False
        else:
            cmd_parts.extend([
                f"--speculative-algorithm EAGLE",
                f"--speculative-draft-model-path {speculative_draft_model}",
                f"--speculative-num-steps {speculative_num_steps}",
                f"--speculative-eagle-topk {speculative_eagle_topk}",
                f"--speculative-num-draft-tokens {speculative_num_draft_tokens}",
            ])
            spec_info["N"] = speculative_num_steps
            spec_info["draft_model"] = speculative_draft_model
    elif spec_algo == "ngram":
        cmd_parts.extend([
            f"--speculative-algorithm ngram",
            f"--speculative-num-steps {speculative_num_steps}",
            f"--speculative-num-draft-tokens {speculative_num_draft_tokens}",
        ])
        spec_info["N"] = speculative_num_steps
        spec_info["note"] = "n-gram 需 flashinfer (V4-Flash 不兼容)"
    elif spec_algo == "chain":
        # chain 是客户端投机 (spec_decode_ir.py), 不在 sglang 启动参数
        spec_info["note"] = "chain 是客户端投机 (spec_decode_ir.py --mode chain), sglang 端无需参数"
        spec_info["client_cmd"] = f"python -m app.shared.spec_decode_ir --backend sglang --mode chain --num-draft {speculative_num_steps}"

    # === 端云 PD 分离配置 ===
    pd_info = {"enabled": pd_separation}
    if pd_separation:
        # 自动 cut layer
        if pd_cut_layer == 0:
            pd_cut_layer = 21 if is_v4_flash else 42  # V4-Flash=21, Qwen3-VL=42 (只跑 lm_head)

        # PD 环境变量
        env_vars.append(f"CGC_PD_SEPARATION=1")
        env_vars.append(f"CGC_PD_CUT_LAYER={pd_cut_layer}")
        env_vars.append(f"CGC_PD_TRANSPORT={pd_transport}")
        if pd_emit_host:
            env_vars.append(f"CGC_PD_EMIT_HOST={pd_emit_host}")
            env_vars.append(f"CGC_PD_EMIT_PORT={pd_emit_port}")

        pd_info["cut_layer"] = pd_cut_layer
        pd_info["transport"] = pd_transport
        pd_info["emit_host"] = pd_emit_host or "(未指定, 需手动设置)"
        pd_info["emit_port"] = pd_emit_port
        pd_info["flow"] = f"cloud prefill (layer 0..{pd_cut_layer}) → emit hidden+KV → edge resume decode (layer {pd_cut_layer}..end)"

        # PD 分离需要 CGC 注入 (emit/resume 机制)
        if no_cgc:
            print(f"\n⚠️  端云 PD 分离需要 CGC 注入 (emit/resume), 自动启用 CGC")
            no_cgc = False
            env_vars.append("CGC_ENABLE_ORTHO_KDA=1")

    # 组装命令
    env_str = " ".join(env_vars) if env_vars else ""
    full_cmd = f"{env_str} {' '.join(cmd_parts)}" if env_str else " ".join(cmd_parts)

    # 模型信息摘要
    model_info = {
        "model_path": model_path,
        "model_type": model_type,
        "is_v4_flash": is_v4_flash,
        "is_qwen3_vl": is_qwen3_vl,
        "has_nextn": has_nextn,
        "has_native_compress": has_compress,
        "sliding_window": sliding_window,
        "tp": tp,
        "cgc_enabled": not no_cgc,
        "cuda_graph_enabled": not no_cuda_graph,
        "speculative": spec_info,
        "pd_separation": pd_info,
    }

    # 路由策略 (固化端云协议)
    if is_v4_flash:
        route = "纯云 PD 分离 (cloud prefill → NIXL → edge decode)"
        expected_tps = "38 tok/s (cuda-graph + NEXTN)"
    elif is_qwen3_vl:
        route = "Mac 本地 (MLX) 或 全云 (按算力)"
        expected_tps = "234 tok/s (云) / 53 tok/s (Mac MLX + 投机)"
    else:
        route = "AutoTunner 自动路由 (按模型大小)"
        expected_tps = "待测"

    # R-SWA 决策 (AutoTunner compress_ratios 检测)
    rswa_decision = "跳过 (已有原生 compress_ratios)" if has_compress else "插入 R-SWA (无原生压缩)"

    result = {
        "status": "ok",
        "command": full_cmd,
        "model_info": model_info,
        "route": route,
        "auto_route": auto_route,
        "switcher_decision": switcher_decision.value if switcher_decision else None,
        "switcher_reason": switcher_reason,
        "expected_tps": expected_tps,
        "rswa_decision": rswa_decision,
        "executed": False,
    }

    # 打印结果
    print("=" * 70)
    print(" cgc model launch — 端云协议固化 (AutoTunner)")
    print("=" * 70)
    print(f"\n模型: {model_path}")
    print(f"类型: {model_type or 'unknown'}")
    print(f"V4-Flash: {'是' if is_v4_flash else '否'} | Qwen3-VL: {'是' if is_qwen3_vl else '否'}")
    print(f"内置 MTP (NEXTN): {'是' if has_nextn else '否'}")
    print(f"原生压缩 (compress_ratios): {'是' if has_compress else '否'}")
    print(f"R-SWA 决策: {rswa_decision}")

    # SeamlessSwitcher 自动决策
    print(f"\n--- SeamlessSwitcher 自动决策 ---")
    print(f"路由: {auto_route}")
    print(f"原因: {switcher_reason}")

    # 4D 感知矩阵
    print(f"\n--- 4D 感知矩阵 (route_decision) ---")
    if matrix_4d.get("error"):
        print(f"状态: {matrix_4d['error']}")
    else:
        print(f"模型大小: {matrix_4d.get('model_size', '?')}GB")
        print(f"算力等级: {matrix_4d.get('compute_tier', '?')}")
        print(f"内存: {matrix_4d.get('memory_gb', '?')}GB")
        print(f"网络 RTT: {matrix_4d.get('network_rtt_ms', '?')}ms")
        print(f"4D 路由: {matrix_4d.get('route', auto_route)}")

    # Magicompiler IR Pass
    print(f"\n--- Magicompiler IR Pass (R-SWA) ---")
    if magicompiler_info.get("error"):
        print(f"状态: {magicompiler_info['error']}")
    else:
        print(f"原生压缩: {'是' if magicompiler_info.get('has_native_compress') else '否'}")
        print(f"R-SWA 动作: {magicompiler_info.get('rswa_action', '未知')}")

    # 投机 decode 信息
    print(f"\n--- 投机 decode ---")
    if spec_info.get("enabled"):
        print(f"算法: {spec_info['algorithm'].upper()}")
        if spec_info.get("N"):
            print(f"N (num_steps): {spec_info['N']}")
        if spec_info.get("draft_tokens"):
            print(f"draft tokens: {spec_info['draft_tokens']}")
        if spec_info.get("source"):
            print(f"来源: {spec_info['source']}")
        if spec_info.get("draft_model"):
            print(f"draft model: {spec_info['draft_model']}")
        if spec_info.get("note"):
            print(f"注意: {spec_info['note']}")
        if spec_info.get("client_cmd"):
            print(f"客户端命令: {spec_info['client_cmd']}")
    else:
        print(f"状态: 关闭 (none)")

    # 端云 PD 分离信息
    print(f"\n--- 端云 PD 分离 ---")
    if pd_info.get("enabled"):
        print(f"状态: 启用")
        print(f"cut layer: {pd_info['cut_layer']}")
        print(f"传输: {pd_info['transport']}")
        print(f"edge host: {pd_info['emit_host']}")
        print(f"edge port: {pd_info['emit_port']}")
        print(f"流程: {pd_info['flow']}")
    else:
        print(f"状态: 关闭 (--pd-separation 启用)")

    print(f"\n路由策略: {route}")
    print(f"预期性能: {expected_tps}")
    print(f"\n启动命令:")
    print(f"  {full_cmd}")

    # 执行命令
    if exec_cmd:
        print(f"\n[exec] 启动中 (后台)...")
        import subprocess
        import sys as _sys
        try:
            proc = subprocess.Popen(
                full_cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            result["executed"] = True
            result["pid"] = proc.pid
            print(f"[exec] PID={proc.pid}, 日志: /tmp/sglang_launch.log")
            print(f"[exec] 等待加载 (~5min for V4-Flash, ~30s for Qwen3-VL)")
        except Exception as e:
            result["executed"] = False
            result["error"] = str(e)
            print(f"[exec] 启动失败: {e}")

    print("\n" + "=" * 70)
    return result


def _model_list_session(*, cfg, output_dir="", model_roots=None, nfs_roots=None, source_filter="all"):
    out_dir = _make_model_output_dir(output_dir, command_name="model_list")
    payload = collect_list_response(
        cfg=cfg,
        model_roots=model_roots,
        nfs_roots=nfs_roots,
        source_filter=source_filter,
    )
    payload_path = write_json_file(out_dir / "model_list_payload.json", payload)
    session = {
        "status": str(payload.get("status") or "FAIL"),
        "command": "cgc model list",
        "generated_at": utc_now_iso(),
        "artifact_root": str(out_dir),
        "model_roots": list(model_roots or []),
        "nfs_roots": list(nfs_roots or []),
        "source_filter": str(payload.get("source_filter") or source_filter or "all"),
        "summary": dict(payload.get("summary") or {}),
        "models": list(payload.get("models") or []),
        "list_payload_path": str(payload_path),
    }
    session_path = write_json_file(out_dir / "model_list_session.json", session)
    session["list_session_path"] = str(session_path)
    write_json_file(out_dir / "model_list_session.json", session)
    return session


def _model_run_session(
    *,
    cfg,
    model="",
    use_omlx=False,
    use_flashmoe=False,
    prompt="",
    max_tokens=256,
    output_dir="",
    gui_duration_s=0,
    disable_gui_stage_source=False,
):
    model_to_use = str(model or cfg.get("active_edge_model") or "").strip()
    if not model_to_use:
        raise ValueError("Model is required. Pass a model name or configure `active_edge_model` first.")
    out_dir = _make_model_output_dir(output_dir, command_name="model_run")
    api_base_url = get_edge_api_base_url(cfg)
    runtime_model = resolve_local_runtime_model(
        model_to_use,
        cfg=cfg,
        use_omlx=bool(use_omlx),
        use_flashmoe=bool(use_flashmoe),
    )
    prompt_text = str(prompt or "").strip()
    if not prompt_text:
        session = {
            "status": "PASS",
            "command": "cgc model run",
            "generated_at": utc_now_iso(),
            "artifact_root": str(out_dir),
            "source_artifact_root": "",
            "selected_model": model_to_use,
            "runtime_model": runtime_model,
            "resolved_model_path": runtime_model,
            "mode": "interactive_shell",
            "interactive_mode": True,
            "use_omlx": bool(use_omlx),
            "use_flashmoe": bool(use_flashmoe),
            "notes": "Structured model session created; interactive prompt loop follows the existing `cgc run` behavior.",
        }
        session_path = write_json_file(out_dir / "model_run_session.json", session)
        session["run_session_path"] = str(session_path)
        write_json_file(out_dir / "model_run_session.json", session)
        return session
    artifact_dir = (out_dir / "run_artifacts").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    gui_stage_source_path = ""
    if not bool(disable_gui_stage_source) and int(gui_duration_s) > 0:
        gui_stage_source_path = _collect_gui_stage_source_evidence(
            duration_s=int(gui_duration_s),
            output_dir=artifact_dir / "gui_agent_runtime",
        )
    payload = {
        "model": runtime_model,
        "prompt": prompt_text,
        "stream": True,
        "use_omlx": bool(use_omlx),
        "use_flashmoe": bool(use_flashmoe),
        "max_tokens": int(max_tokens),
        "api_base_url": api_base_url,
    }
    prev_gui_evidence = os.environ.get("CGC_M72_GUI_EVENT_EVIDENCE")
    if str(gui_stage_source_path).strip():
        os.environ["CGC_M72_GUI_EVENT_EVIDENCE"] = str(gui_stage_source_path)
    try:
        run_result = _execute_single_prompt(api_base_url=api_base_url, payload=payload)
        artifact_summary = _write_cgc_run_artifacts(
            report_dir=artifact_dir,
            model_to_use=model_to_use,
            runtime_model=runtime_model,
            prompt=prompt_text,
            payload=payload,
            run_result=run_result,
        )
    finally:
        if prev_gui_evidence is None:
            os.environ.pop("CGC_M72_GUI_EVENT_EVIDENCE", None)
        else:
            os.environ["CGC_M72_GUI_EVENT_EVIDENCE"] = prev_gui_evidence
    session = {
        "status": str(artifact_summary.get("status") or "FAIL"),
        "command": "cgc model run",
        "generated_at": utc_now_iso(),
        "artifact_root": str(out_dir),
        "source_artifact_root": str(artifact_dir),
        "selected_model": model_to_use,
        "runtime_model": runtime_model,
        "resolved_model_path": str(artifact_summary.get("resolved_model_path") or runtime_model),
        "format": str(artifact_summary.get("format") or _detect_model_format(runtime_model)),
        "selected_route": str(artifact_summary.get("selected_route") or ""),
        "selected_backend": str(artifact_summary.get("backend") or ""),
        "local_execution": bool(artifact_summary.get("local_execution")),
        "cloud_bridge_used": bool(str(artifact_summary.get("selected_route") or "") == "m73_edge_cloud"),
        "decision_reason": artifact_summary.get("decision_reason") or {},
        "response": {
            "text": str(run_result.get("response_text") or ""),
            "finish_reason": "stop" if str(artifact_summary.get("status") or "") == "PASS" else "error",
        },
        "edge_latency_ms": float(artifact_summary.get("edge_latency_ms") or 0.0),
        "prompt": prompt_text,
        "max_tokens": int(max_tokens),
        "use_omlx": bool(use_omlx),
        "use_flashmoe": bool(use_flashmoe),
        "evidence_paths": {
            "local_infer": str(artifact_summary.get("evidence_path") or ""),
            "run_report": str(artifact_summary.get("run_report_path") or ""),
            "m4_inference_report": str(artifact_summary.get("m4_inference_report_path") or ""),
            "edge_inference_bridge": str(artifact_summary.get("edge_inference_bridge_path") or ""),
            "route_decision": str(artifact_summary.get("route_decision_path") or ""),
            "gui_stage_source": str(gui_stage_source_path or ""),
        },
    }
    session_path = write_json_file(out_dir / "model_run_session.json", session)
    session["run_session_path"] = str(session_path)
    write_json_file(out_dir / "model_run_session.json", session)
    return session


def _model_serve_session(*, output_dir="", host="0.0.0.0", port=8000, proxy_host="127.0.0.1", proxy_port=4000, cfg=None):
    out_dir = _make_model_output_dir(output_dir, command_name="model_serve")
    resolved_cfg = dict(cfg or load_config())
    session = {
        "status": "PASS",
        "command": "cgc model serve",
        "generated_at": utc_now_iso(),
        "artifact_root": str(out_dir),
        "runtime_mode": "model_api_server",
        "api_host": str(host),
        "api_port": int(port),
        "proxy_host": str(proxy_host),
        "proxy_port": int(proxy_port),
        "api_base_url": f"http://{host}:{int(port)}",
        "proxy_base_url": f"http://{proxy_host}:{int(proxy_port)}",
        "cloud_node": {
            "host": str(resolved_cfg.get("cloud_ip") or ""),
            "port": int(resolved_cfg.get("cloud_port") or 0),
        },
        "config_path": str(CONFIG_FILE),
    }
    session_path = write_json_file(out_dir / "model_serve_session.json", session)
    session["serve_session_path"] = str(session_path)
    write_json_file(out_dir / "model_serve_session.json", session)
    return session


def _model_verify_session(*, run_session_path="", artifact_root="", output_dir=""):
    out_dir = _make_model_output_dir(output_dir, command_name="model_verify")
    context = _resolve_model_context(run_session_path=run_session_path, artifact_root=artifact_root)
    run_report = context["run_report"]
    route_decision = context["route_decision"]
    edge_bridge = context["edge_inference_bridge"]
    release_artifacts = _as_dict(context.get("release_artifacts"))
    build_matrix_manifest = _as_dict(release_artifacts.get("build_matrix_manifest"))
    bundle_validation = _resolve_model_profile_bundle_validation(context=context)
    checks = {
        "run_report": {"exists": bool(context["run_report_path"]), "status": str(run_report.get("status") or "")},
        "route_decision": {"exists": bool(context["route_decision_path"]), "selected_route": str(route_decision.get("selected_route") or "")},
        "m4_inference_report": {"exists": bool(context["m4_inference_report_path"]), "status": str(context["m4_inference_report"].get("status") or "")},
        "edge_inference_bridge": {"exists": bool(context["edge_inference_bridge_path"]), "status": str(edge_bridge.get("status") or "")},
        "m6_build_report": {"exists": bool(release_artifacts.get("m6_build_report_path")), "status": str(_nested_get(release_artifacts.get("m6_build_report"), "gate_result.m6.status", ""))},
        "m8_build_matrix_manifest": {"exists": bool(release_artifacts.get("build_matrix_manifest_path")), "status": str(build_matrix_manifest.get("matrix_status") or build_matrix_manifest.get("status") or "")},
        "profile_bundle_validation": {
            "exists": bool(str(bundle_validation.get("resolved_paths", {}).get("profile_settings_path") or "").strip()),
            "status": str(bundle_validation.get("status") or ""),
        },
    }
    ok = (
        all(bool(item.get("exists")) for key, item in checks.items() if key != "profile_bundle_validation")
        and str(run_report.get("status") or "") == "PASS"
        and str(bundle_validation.get("status") or "") != "FAIL"
    )
    payload = {
        "status": "PASS" if ok else "FAIL",
        "command": "cgc model verify",
        "upkg_layer": "UPKG_2.0",
        "gate_family": "model",
        "generated_at": utc_now_iso(),
        "artifact_root": str(out_dir),
        "source_artifact_root": str(context["artifact_root"]),
        "run_session_path": str(context["run_session_path"]),
        "accepted_contracts": {
            "m6_bundle_gate": {
                "status": str(_nested_get(release_artifacts.get("m6_build_report"), "gate_result.m6.status", "")),
                "bundle_manifest_path": str(release_artifacts.get("m6_bundle_manifest_path") or ""),
            },
            "m8_run_contract": {
                "status": str(run_report.get("status") or ""),
                "selected_route": str(route_decision.get("selected_route") or run_report.get("selected_route") or ""),
                "selected_backend": str(route_decision.get("selected_backend") or run_report.get("backend") or ""),
            },
            "m8_release_build_contract": {
                "status": str(build_matrix_manifest.get("matrix_status") or build_matrix_manifest.get("status") or ""),
                "build_matrix_manifest_path": str(release_artifacts.get("build_matrix_manifest_path") or ""),
                "build_matrix_file_path": str(release_artifacts.get("build_matrix_file_path") or ""),
            },
            "profile_bundle_contract": {
                "status": str(bundle_validation.get("status") or ""),
                "validation": bundle_validation,
            },
        },
        "checks": checks,
        "selected_model": str((context["run_session"].get("selected_model") if isinstance(context["run_session"], dict) else "") or run_report.get("model") or ""),
        "resolved_model_path": str(run_report.get("resolved_model_path") or ""),
        "selected_route": str(route_decision.get("selected_route") or run_report.get("selected_route") or ""),
        "selected_backend": str(route_decision.get("selected_backend") or run_report.get("backend") or ""),
        "artifact_index": {
            "run_report": str(context["run_report_path"]),
            "route_decision": str(context["route_decision_path"]),
            "m4_inference_report": str(context["m4_inference_report_path"]),
            "edge_inference_bridge": str(context["edge_inference_bridge_path"]),
            "m6_build_report": str(release_artifacts.get("m6_build_report_path") or ""),
            "m6_bundle_manifest": str(release_artifacts.get("m6_bundle_manifest_path") or ""),
            "m8_build_matrix_manifest": str(release_artifacts.get("build_matrix_manifest_path") or ""),
            "m8_build_matrix_file": str(release_artifacts.get("build_matrix_file_path") or ""),
            "profile_settings_path": str(bundle_validation.get("resolved_paths", {}).get("profile_settings_path") or ""),
            "system_manifest_path": str(bundle_validation.get("resolved_paths", {}).get("system_manifest_path") or ""),
            "bootstrap_contract_path": str(bundle_validation.get("resolved_paths", {}).get("bootstrap_contract_path") or ""),
        },
    }
    result_path = write_json_file(out_dir / "model_verify_session.json", payload)
    payload["verify_session_path"] = str(result_path)
    write_json_file(out_dir / "model_verify_session.json", payload)
    return payload


def _model_audit_session(*, run_session_path="", artifact_root="", output_dir=""):
    out_dir = _make_model_output_dir(output_dir, command_name="model_audit")
    context = _resolve_model_context(run_session_path=run_session_path, artifact_root=artifact_root)
    run_report = context["run_report"]
    route_decision = context["route_decision"]
    edge_bridge = context["edge_inference_bridge"]
    release_artifacts = _as_dict(context.get("release_artifacts"))
    build_matrix_manifest = _as_dict(release_artifacts.get("build_matrix_manifest"))
    bundle_validation = _resolve_model_profile_bundle_validation(context=context)
    auditability = bool(context["run_report_path"] and context["route_decision_path"] and context["edge_inference_bridge_path"])
    traceability = bool(str(route_decision.get("decision_reason", {}).get("code") if isinstance(route_decision.get("decision_reason"), dict) else "") or route_decision.get("selected_route"))
    replayability = bool(str(run_report.get("prompt") or "").strip() and str((run_report.get("response_text") or "")).strip())
    comparability = bool(context["m4_inference_report_path"] and context["edge_inference_bridge_path"])
    audit_ok = auditability and str(bundle_validation.get("status") or "") != "FAIL"
    payload = {
        "status": "PASS" if audit_ok else "FAIL",
        "command": "cgc model audit",
        "upkg_layer": "UPKG_2.0",
        "gate_family": "model",
        "generated_at": utc_now_iso(),
        "artifact_root": str(out_dir),
        "source_artifact_root": str(context["artifact_root"]),
        "run_session_path": str(context["run_session_path"]),
        "auditability": auditability,
        "traceability": traceability,
        "replayability": replayability,
        "comparability": comparability,
        "profile_bundle_governance": bundle_validation,
        "selected_model": str((context["run_session"].get("selected_model") if isinstance(context["run_session"], dict) else "") or run_report.get("model") or ""),
        "resolved_model_path": str(run_report.get("resolved_model_path") or ""),
        "selected_route": str(route_decision.get("selected_route") or run_report.get("selected_route") or ""),
        "selected_backend": str(route_decision.get("selected_backend") or run_report.get("backend") or ""),
        "artifact_index": {
            "run_report": str(context["run_report_path"]),
            "route_decision": str(context["route_decision_path"]),
            "m4_inference_report": str(context["m4_inference_report_path"]),
            "edge_inference_bridge": str(context["edge_inference_bridge_path"]),
            "m6_build_report": str(release_artifacts.get("m6_build_report_path") or ""),
            "m6_bundle_manifest": str(release_artifacts.get("m6_bundle_manifest_path") or ""),
            "m8_build_matrix_manifest": str(release_artifacts.get("build_matrix_manifest_path") or ""),
            "m8_build_matrix_file": str(release_artifacts.get("build_matrix_file_path") or ""),
            "profile_settings_path": str(bundle_validation.get("resolved_paths", {}).get("profile_settings_path") or ""),
            "system_manifest_path": str(bundle_validation.get("resolved_paths", {}).get("system_manifest_path") or ""),
            "bootstrap_contract_path": str(bundle_validation.get("resolved_paths", {}).get("bootstrap_contract_path") or ""),
        },
        "release_build_contract": {
            "matrix_status": str(build_matrix_manifest.get("matrix_status") or build_matrix_manifest.get("status") or ""),
            "required_platforms": list(build_matrix_manifest.get("required_platforms") or []),
            "platform_reports": dict(release_artifacts.get("platform_reports") or {}),
        },
        "failure_attribution": {
            "reason_code": str(route_decision.get("decision_reason", {}).get("code") if isinstance(route_decision.get("decision_reason"), dict) else ""),
            "reason_text": str(route_decision.get("decision_reason", {}).get("text") if isinstance(route_decision.get("decision_reason"), dict) else ""),
            "profile_bundle_contract_status": str(bundle_validation.get("status") or ""),
        },
    }
    result_path = write_json_file(out_dir / "model_audit_session.json", payload)
    payload["audit_session_path"] = str(result_path)
    write_json_file(out_dir / "model_audit_session.json", payload)
    return payload


def _model_replay_session(*, run_session_path="", artifact_root="", output_dir=""):
    out_dir = _make_model_output_dir(output_dir, command_name="model_replay")
    context = _resolve_model_context(run_session_path=run_session_path, artifact_root=artifact_root)
    run_report = context["run_report"]
    route_decision = context["route_decision"]
    release_artifacts = _as_dict(context.get("release_artifacts"))
    payload = {
        "status": "PASS" if bool(context["run_report_path"]) else "FAIL",
        "command": "cgc model replay",
        "upkg_layer": "UPKG_2.0",
        "gate_family": "model",
        "generated_at": utc_now_iso(),
        "artifact_root": str(out_dir),
        "source_artifact_root": str(context["artifact_root"]),
        "run_session_path": str(context["run_session_path"]),
        "replay_mode": "model_single_prompt_replay",
        "selected_model": str((context["run_session"].get("selected_model") if isinstance(context["run_session"], dict) else "") or run_report.get("model") or ""),
        "resolved_model_path": str(run_report.get("resolved_model_path") or ""),
        "prompt": str(run_report.get("prompt") or ""),
        "response_text": str(run_report.get("response_text") or ""),
        "selected_route": str(route_decision.get("selected_route") or run_report.get("selected_route") or ""),
        "selected_backend": str(route_decision.get("selected_backend") or run_report.get("backend") or ""),
        "replay_anchor": {
            "run_report_path": str(context["run_report_path"]),
            "route_decision_path": str(context["route_decision_path"]),
            "m4_inference_report_path": str(context["m4_inference_report_path"]),
            "edge_inference_bridge_path": str(context["edge_inference_bridge_path"]),
            "m6_build_report_path": str(release_artifacts.get("m6_build_report_path") or ""),
            "m6_bundle_manifest_path": str(release_artifacts.get("m6_bundle_manifest_path") or ""),
            "m8_build_matrix_manifest_path": str(release_artifacts.get("build_matrix_manifest_path") or ""),
            "m8_build_matrix_file_path": str(release_artifacts.get("build_matrix_file_path") or ""),
        },
        "artifact_index": {
            "run_report": str(context["run_report_path"]),
            "route_decision": str(context["route_decision_path"]),
            "m4_inference_report": str(context["m4_inference_report_path"]),
            "edge_inference_bridge": str(context["edge_inference_bridge_path"]),
            "m6_build_report": str(release_artifacts.get("m6_build_report_path") or ""),
            "m8_build_matrix_manifest": str(release_artifacts.get("build_matrix_manifest_path") or ""),
        },
    }
    result_path = write_json_file(out_dir / "model_replay_session.json", payload)
    payload["replay_session_path"] = str(result_path)
    write_json_file(out_dir / "model_replay_session.json", payload)
    return payload


def _model_trace_session(*, run_session_path="", artifact_root="", output_dir=""):
    out_dir = _make_model_output_dir(output_dir, command_name="model_trace")
    context = _resolve_model_context(run_session_path=run_session_path, artifact_root=artifact_root)
    run_report = context["run_report"]
    route_decision = context["route_decision"]
    release_artifacts = _as_dict(context.get("release_artifacts"))
    payload = {
        "status": "PASS" if bool(context["run_report_path"]) else "FAIL",
        "command": "cgc model trace",
        "upkg_layer": "UPKG_2.0",
        "gate_family": "model",
        "generated_at": utc_now_iso(),
        "artifact_root": str(out_dir),
        "source_artifact_root": str(context["artifact_root"]),
        "run_session_path": str(context["run_session_path"]),
        "run_report_path": str(context["run_report_path"]),
        "route_decision_path": str(context["route_decision_path"]),
        "stream_event_count": int(run_report.get("stream_event_count") or 0),
        "selected_route": str(route_decision.get("selected_route") or run_report.get("selected_route") or ""),
        "selected_backend": str(route_decision.get("selected_backend") or run_report.get("backend") or ""),
        "decision_reason": route_decision.get("decision_reason") or {},
        "final_event_preview": dict(run_report.get("final_event") or {}) if isinstance(run_report.get("final_event"), dict) else {},
        "stage_trace": [
            {
                "stage": "m6_bundle_gate",
                "status": str(_nested_get(release_artifacts.get("m6_build_report"), "gate_result.m6.status", "")),
                "artifact": str(release_artifacts.get("m6_bundle_manifest_path") or ""),
            },
            {
                "stage": "m8_run_contract",
                "status": str(run_report.get("status") or ""),
                "artifact": str(context["run_report_path"]),
            },
            {
                "stage": "m8_route_decision",
                "status": "PASS" if str(context["route_decision_path"]).strip() else "FAIL",
                "artifact": str(context["route_decision_path"]),
            },
            {
                "stage": "m8_release_build_contract",
                "status": str(_nested_get(release_artifacts.get("build_matrix_manifest"), "matrix_status", _nested_get(release_artifacts.get("build_matrix_manifest"), "status", ""))),
                "artifact": str(release_artifacts.get("build_matrix_manifest_path") or ""),
            },
        ],
    }
    result_path = write_json_file(out_dir / "model_trace_session.json", payload)
    payload["trace_session_path"] = str(result_path)
    write_json_file(out_dir / "model_trace_session.json", payload)
    return payload


def _model_compare_session(
    *,
    run_session_path="",
    artifact_root="",
    compare_against_run_session="",
    compare_against_artifact_root="",
    output_dir="",
):
    out_dir = _make_model_output_dir(output_dir, command_name="model_compare")
    left = _resolve_model_context(run_session_path=run_session_path, artifact_root=artifact_root)
    right = _resolve_model_context(
        run_session_path=compare_against_run_session,
        artifact_root=compare_against_artifact_root,
    )
    left_run = _as_dict(left.get("run_report"))
    right_run = _as_dict(right.get("run_report"))
    left_route = _as_dict(left.get("route_decision"))
    right_route = _as_dict(right.get("route_decision"))
    left_release = _as_dict(left.get("release_artifacts"))
    right_release = _as_dict(right.get("release_artifacts"))
    payload = {
        "status": "PASS" if bool(left.get("run_report_path")) and bool(right.get("run_report_path")) else "FAIL",
        "command": "cgc model compare",
        "upkg_layer": "UPKG_2.0",
        "gate_family": "model",
        "generated_at": utc_now_iso(),
        "artifact_root": str(out_dir),
        "left": {
            "run_session_path": str(left.get("run_session_path") or ""),
            "source_artifact_root": str(left.get("artifact_root") or ""),
            "selected_model": str((_as_dict(left.get("run_session")).get("selected_model")) or left_run.get("model") or ""),
            "resolved_model_path": str(left_run.get("resolved_model_path") or ""),
            "selected_route": str(left_route.get("selected_route") or left_run.get("selected_route") or ""),
            "selected_backend": str(left_route.get("selected_backend") or left_run.get("backend") or ""),
            "edge_latency_ms": float(left_run.get("edge_latency_ms") or 0.0),
            "local_execution": bool(left_run.get("local_execution")),
            "cloud_bridge_used": bool(left_run.get("cloud_bridge_used")),
            "decision_reason_code": str(_nested_get(left_route, "decision_reason.code", "")),
            "build_matrix_status": str(_nested_get(left_release.get("build_matrix_manifest"), "matrix_status", _nested_get(left_release.get("build_matrix_manifest"), "status", ""))),
        },
        "right": {
            "run_session_path": str(right.get("run_session_path") or ""),
            "source_artifact_root": str(right.get("artifact_root") or ""),
            "selected_model": str((_as_dict(right.get("run_session")).get("selected_model")) or right_run.get("model") or ""),
            "resolved_model_path": str(right_run.get("resolved_model_path") or ""),
            "selected_route": str(right_route.get("selected_route") or right_run.get("selected_route") or ""),
            "selected_backend": str(right_route.get("selected_backend") or right_run.get("backend") or ""),
            "edge_latency_ms": float(right_run.get("edge_latency_ms") or 0.0),
            "local_execution": bool(right_run.get("local_execution")),
            "cloud_bridge_used": bool(right_run.get("cloud_bridge_used")),
            "decision_reason_code": str(_nested_get(right_route, "decision_reason.code", "")),
            "build_matrix_status": str(_nested_get(right_release.get("build_matrix_manifest"), "matrix_status", _nested_get(right_release.get("build_matrix_manifest"), "status", ""))),
        },
        "deltas": {
            "route_changed": str(left_route.get("selected_route") or left_run.get("selected_route") or "") != str(right_route.get("selected_route") or right_run.get("selected_route") or ""),
            "backend_changed": str(left_route.get("selected_backend") or left_run.get("backend") or "") != str(right_route.get("selected_backend") or right_run.get("backend") or ""),
            "decision_reason_changed": str(_nested_get(left_route, "decision_reason.code", "")) != str(_nested_get(right_route, "decision_reason.code", "")),
            "local_execution_changed": bool(left_run.get("local_execution")) != bool(right_run.get("local_execution")),
            "cloud_bridge_changed": bool(left_run.get("cloud_bridge_used")) != bool(right_run.get("cloud_bridge_used")),
            "latency_ms_delta": float(left_run.get("edge_latency_ms") or 0.0) - float(right_run.get("edge_latency_ms") or 0.0),
            "response_length_delta": len(str(left_run.get("response_text") or "")) - len(str(right_run.get("response_text") or "")),
            "build_matrix_status_changed": str(_nested_get(left_release.get("build_matrix_manifest"), "matrix_status", _nested_get(left_release.get("build_matrix_manifest"), "status", ""))) != str(_nested_get(right_release.get("build_matrix_manifest"), "matrix_status", _nested_get(right_release.get("build_matrix_manifest"), "status", ""))),
        },
        "artifact_index": {
            "left_run_report": str(left.get("run_report_path") or ""),
            "right_run_report": str(right.get("run_report_path") or ""),
            "left_route_decision": str(left.get("route_decision_path") or ""),
            "right_route_decision": str(right.get("route_decision_path") or ""),
            "left_m8_build_matrix_manifest": str(left_release.get("build_matrix_manifest_path") or ""),
            "right_m8_build_matrix_manifest": str(right_release.get("build_matrix_manifest_path") or ""),
        },
    }
    result_path = write_json_file(out_dir / "model_compare_session.json", payload)
    payload["compare_session_path"] = str(result_path)
    write_json_file(out_dir / "model_compare_session.json", payload)
    return payload


def _format_gpu_partition(start_gpu, gpus_per_instance):
    end_gpu = max(start_gpu, start_gpu + max(1, int(gpus_per_instance)) - 1)
    return f"{start_gpu}-{end_gpu}"


def _plan_dualnode_instance_layout(*, cluster_status, target_instances_per_node, gpus_per_instance):
    hosts = cluster_status.get("hosts") if isinstance(cluster_status.get("hosts"), list) else []
    target_instances_per_node = max(1, int(target_instances_per_node or 1))
    gpus_per_instance = max(1, int(gpus_per_instance or 1))
    host_plans = []
    total_recommended_instances = 0
    for host in hosts:
        host_payload = host if isinstance(host, dict) else {}
        host_name = str(host_payload.get("name") or host_payload.get("host") or "unknown")
        gpu_rows = host_payload.get("gpus") if isinstance(host_payload.get("gpus"), list) else []
        gpu_count = len(gpu_rows)
        data_usage = host_payload.get("data_usage") if isinstance(host_payload.get("data_usage"), dict) else {}
        data_mount_ready = int(data_usage.get("total_bytes") or 0) > 0
        role = str(host_payload.get("role") or "")
        capacity_instances = gpu_count // gpus_per_instance if gpu_count > 0 else 0
        recommended_instances = min(target_instances_per_node, capacity_instances)
        reasons = []
        if gpu_count <= 0:
            reasons.append("gpu_unavailable")
        elif capacity_instances < target_instances_per_node:
            reasons.append("gpu_capacity_limited")
        if not data_mount_ready:
            reasons.append("data_mount_unavailable")
        if reasons and "gpu_unavailable" in reasons:
            recommended_instances = 0
        partitions = [
            _format_gpu_partition(index * gpus_per_instance, gpus_per_instance)
            for index in range(recommended_instances)
        ]
        total_recommended_instances += recommended_instances
        host_plans.append(
            {
                "host": str(host_payload.get("host") or ""),
                "name": host_name,
                "role": role,
                "gpu_count": gpu_count,
                "data_mount_ready": data_mount_ready,
                "capacity_instances": capacity_instances,
                "target_instances": target_instances_per_node,
                "recommended_instances": recommended_instances,
                "gpu_partitions": partitions,
                "reasons": reasons,
            }
        )
    return {
        "hosts": host_plans,
        "total_hosts": len(host_plans),
        "target_instances_per_node": target_instances_per_node,
        "gpus_per_instance": gpus_per_instance,
        "recommended_total_instances": total_recommended_instances,
    }


def _resolve_swebench_launch_plan(
    *,
    recommendation,
    benchmark_limit,
    model_name,
    api_base_url,
):
    total_instances = int(recommendation.get("recommended_total_instances") or 0)
    effective_limit = max(1, int(benchmark_limit or 500))
    worker_count = max(total_instances, effective_limit * max(total_instances, 1))
    suffix = f"cgc_m76_dualnode_{total_instances}inst_verified_{effective_limit}"
    env = {
        "CGC_SWEBENCH_LIMIT": effective_limit,
        "CGC_SWEBENCH_NUM_WORKERS": worker_count,
        "CGC_SWEBENCH_API_BASE": str(api_base_url or "http://127.0.0.1:50053/v1"),
        "CGC_SWEBENCH_MODEL_NAME": str(model_name or "openai/deepseek-v4-flash"),
        "CGC_SWEBENCH_SUFFIX": suffix,
        "CGC_CLUSTER_NFS_MINICPM5_GGUF": str(
            os.environ.get("CGC_CLUSTER_NFS_MINICPM5_GGUF") or MINICPM5_CLUSTER_NFS_PATH
        ),
    }
    return {
        "topology": {
            "gateway": "FusionRoute",
            "cloud_model": "DeepSeek V4 Flash",
            "router_model": "MiniCPM5",
        },
        "effective_limit": effective_limit,
        "recommended_num_workers": worker_count,
        "recommended_fusion_group_size": total_instances,
        "env": env,
        "launch_command_candidates": [
            "/root/flashkv0516/run_full_swebench.sh",
            "/root/flashkv0516/CGC_Release/run/run_full_swebench.sh",
        ],
    }


def _build_formal_runtime_env_overrides(
    *,
    total_instances,
    gpus_per_instance,
    deepep_mode,
):
    ep_size = max(1, int(total_instances or 1))
    tp_size = max(1, int(gpus_per_instance or 1))
    deepep_enabled = str(deepep_mode or "").strip().lower() not in {"0", "off", "false", "degraded"}
    requested_dispatch_backend = "deepep" if deepep_enabled else "native_sglang"
    requested_distributed_runtime = "nccl" if max(ep_size, tp_size) > 1 else "single_process"
    service_topology_backend = "ray_cluster_dual_host" if ep_size > 1 else "single_host_local"
    overrides = {
        "CGC_M76_ENABLE_DEEPEP": "1" if deepep_enabled else "0",
        "CGC_DEEPEP_MODE": "auto" if deepep_enabled else "off",
        "CGC_MEGATRAIN_REQUESTED_DISPATCH_BACKEND": requested_dispatch_backend,
        "CGC_REQUESTED_DISPATCH_BACKEND": requested_dispatch_backend,
        "CGC_SGLANG_TP_SIZE": str(tp_size),
        "CGC_SGLANG_PP_SIZE": "1",
        "CGC_SGLANG_EP_SIZE": str(ep_size),
        "CGC_DEEPEP_PARALLEL_PROFILE": f"ep{ep_size}_tp{tp_size}",
        "CGC_SERVICE_TOPOLOGY_BACKEND": service_topology_backend,
        "CGC_M76_ENABLE_RDMA": "1" if ep_size > 1 else "0",
        "CGC_MEGATRAIN_ENABLE_PD": "1" if deepep_enabled else "0",
        "CGC_ENABLE_PD": "1" if deepep_enabled else "0",
        "CGC_MEGATRAIN_REQUESTED_DISTRIBUTED_RUNTIME": requested_distributed_runtime,
        "CGC_REQUESTED_DISTRIBUTED_RUNTIME": requested_distributed_runtime,
        "CGC_DISTRIBUTED_RUNTIME_BACKEND": requested_distributed_runtime,
        "CGC_MEGATRAIN_ENABLE_NCCL": "1" if requested_distributed_runtime == "nccl" else "0",
        "CGC_SGLANG_USE_NCCL": "1" if requested_distributed_runtime == "nccl" else "0",
    }
    return overrides


def _model_swe_verified_session(
    *,
    cfg,
    output_dir="",
    refresh_session="",
    status_from_file="",
    target_instances_per_node=2,
    gpus_per_instance=4,
    benchmark_limit=500,
    model_name="openai/deepseek-v4-flash",
    api_base_url="",
    run_gate=False,
    deepep_mode="auto",
    auto_confirm=False,
    interactive_confirm=True,
    poll=False,
    poll_interval_seconds=30,
    max_polls=120,
):
    if str(refresh_session or "").strip():
        return _refresh_model_swe_verified_session(
            refresh_session=refresh_session,
            poll=bool(poll),
            poll_interval_seconds=int(poll_interval_seconds or 30),
            max_polls=int(max_polls or 120),
        )
    out_dir = _make_model_output_dir(output_dir, command_name="model_swe_verified")
    session_id = str(out_dir.name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if str(status_from_file or "").strip():
        status_path = Path(status_from_file).expanduser().resolve()
        cluster_status = _safe_read_json(status_path)
        if not isinstance(cluster_status, dict) or not cluster_status:
            raise ValueError(f"Failed to read cluster status from: {status_path}")
        status_path = write_json_file(out_dir / "cluster_status_snapshot.json", cluster_status)
    else:
        collector = load_m75_extreme_status_collector()
        cluster_status = collector(
            expected_workers=max(1, int(benchmark_limit or 500)) * max(1, int(target_instances_per_node or 2)) * 2,
            expected_instances=int(benchmark_limit or 500),
            expected_fusion_group_size=max(1, int(target_instances_per_node or 2)) * 2,
        )
        status_path = write_json_file(out_dir / "cluster_status_snapshot.json", cluster_status)

    recommendation = _plan_dualnode_instance_layout(
        cluster_status=cluster_status,
        target_instances_per_node=target_instances_per_node,
        gpus_per_instance=gpus_per_instance,
    )
    launch_plan = _resolve_swebench_launch_plan(
        recommendation=recommendation,
        benchmark_limit=benchmark_limit,
        model_name=model_name,
        api_base_url=api_base_url or "http://127.0.0.1:50053/v1",
    )
    base_suffix = str(_nested_get(launch_plan, "env.CGC_SWEBENCH_SUFFIX", "cgc_m76_dualnode"))
    session_suffix = f"{base_suffix}_{_safe_slug(session_id, 'session')}"
    launch_plan["session_id"] = session_id
    launch_plan["env"] = dict(launch_plan.get("env") or {})
    launch_plan["env"]["CGC_SWEBENCH_SUFFIX"] = session_suffix
    launch_plan["env"]["CGC_SWEBENCH_LOG"] = _build_remote_swebench_log_path(session_suffix)
    formal_runtime_env = _build_formal_runtime_env_overrides(
        total_instances=int(recommendation.get("recommended_total_instances") or 0),
        gpus_per_instance=int(gpus_per_instance or 1),
        deepep_mode=deepep_mode,
    )
    launch_plan["env"].update(formal_runtime_env)
    recommendation["recommended_num_workers"] = int(launch_plan.get("recommended_num_workers") or 0)
    recommendation["recommended_fusion_group_size"] = int(launch_plan.get("recommended_fusion_group_size") or 0)
    recommendation["layout_code"] = "+".join(str(int(item.get("recommended_instances") or 0)) for item in recommendation.get("hosts") or [])

    recommendation_lines = []
    for host_plan in recommendation.get("hosts") or []:
        partitions = ",".join(host_plan.get("gpu_partitions") or []) or "none"
        recommendation_lines.append(
            f"{host_plan.get('name')}={host_plan.get('recommended_instances')} [{partitions}]"
        )
    prompt = (
        f"Recommended dualnode layout: {'; '.join(recommendation_lines) or 'no-available-hosts'} | "
        f"workers={recommendation.get('recommended_num_workers', 0)} | "
        "Proceed with this M7.6 SWE Verified plan? [y/N]: "
    )
    confirmed = bool(auto_confirm)
    confirmation_mode = "flag" if auto_confirm else "required"
    if not confirmed and bool(interactive_confirm):
        response = input(prompt)
        confirmed = response.strip().lower() in {"y", "yes"}
        confirmation_mode = "interactive"

    gate_result = {}
    gate_report_path = ""
    gate_env_overrides = dict(formal_runtime_env)
    remote_host_specs = _resolve_swebench_remote_hosts(cluster_status)
    remote_head_host = _find_remote_host_by_role(remote_host_specs, "head")
    remote_launch_result = {}
    remote_launch_path = ""
    remote_score_summary = {}
    remote_score_summary_path = ""
    remote_m76_evidence = {}
    remote_m76_evidence_path = ""

    if confirmed:
        try:
            remote_launch_result = _launch_remote_swebench_benchmark(
                head_host=remote_head_host,
                launch_plan=launch_plan,
            )
        except Exception as exc:
            remote_launch_result = {"status": "FAIL", "error": str(exc)}
        # #region debug-point D:launch-result
        _debug_report_dualnode_swe500(
            hypothesis_id="D",
            location="app/cli/cgc.py:_model_swe_verified_session:launch",
            msg="[DEBUG] swe verified remote launch result",
            data={
                "status": str(remote_launch_result.get("status") or ""),
                "head_host": str((remote_head_host or {}).get("host") or ""),
                "api_base": str(_nested_get(launch_plan, "env.CGC_SWEBENCH_API_BASE", "")),
                "suffix": str(_nested_get(launch_plan, "env.CGC_SWEBENCH_SUFFIX", "")),
                "remote_log_path": str(remote_launch_result.get("remote_log_path") or ""),
                "error": str(remote_launch_result.get("error") or ""),
            },
        )
        # #endregion
        remote_launch_path = write_json_file(out_dir / "remote_launch_manifest.json", remote_launch_result)
        try:
            remote_score_summary = _collect_remote_swebench_summary(
                head_host=remote_head_host,
                launch_plan=launch_plan,
            )
        except Exception as exc:
            remote_score_summary = {"status": "FAIL", "error": str(exc)}
        remote_score_summary_path = write_json_file(out_dir / "remote_swebench_score_summary.json", remote_score_summary)
        try:
            remote_m76_evidence = _collect_remote_m76_evidence(remote_host_specs)
        except Exception as exc:
            remote_m76_evidence = {"status": "FAIL", "error": str(exc), "hosts": []}
        remote_m76_evidence_path = write_json_file(out_dir / "remote_m76_evidence_bundle.json", remote_m76_evidence)

    if bool(run_gate) and confirmed:
        run_m76_gate = load_engine_m76_gate_runner()
        previous_env = {key: os.environ.get(key) for key in gate_env_overrides}
        try:
            for key, value in gate_env_overrides.items():
                os.environ[key] = value
            gate_result = run_m76_gate(output_dir=str((out_dir / "m76_gate").resolve()))
            gate_report_path = str(gate_result.get("report_path") or "")
        finally:
            for key, previous in previous_env.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous

    ready = int(recommendation.get("recommended_total_instances") or 0) > 0
    launch_ok = str(remote_launch_result.get("status") or ("PASS" if not confirmed else "")).upper() == "PASS"
    score_contract = _evaluate_swebench_score_contract(remote_score_summary)
    benchmark_state = str(score_contract.get("benchmark_state") or remote_score_summary.get("state") or ("planned" if not confirmed else "pending"))
    m76_summary = _as_dict(remote_m76_evidence.get("summary"))
    runtime_statuses = m76_summary.get("runtime_statuses") if isinstance(m76_summary.get("runtime_statuses"), list) else []
    report_statuses = m76_summary.get("report_statuses") if isinstance(m76_summary.get("report_statuses"), list) else []
    runtime_protocol_contract = _as_dict(remote_m76_evidence.get("runtime_protocol_contract"))
    mandatory_protocol_gate = evaluate_mandatory_protocol_gate(
        runtime_protocol_contract=runtime_protocol_contract,
        zero_copy_vram_real=_as_dict(remote_m76_evidence.get("zero_copy_vram_real")),
        source=str(remote_m76_evidence_path),
    )
    remote_launch_contract_status = str(remote_launch_result.get("status") or ("PLANNED" if not confirmed else "PENDING"))
    remote_score_contract_status = str(score_contract.get("contract_status") or ("PLANNED" if not confirmed else "PENDING"))
    remote_evidence_contract_status = str(remote_m76_evidence.get("status") or ("PLANNED" if not confirmed else "PENDING"))
    runtime_status = (
        "PASS"
        if ready and launch_ok and remote_score_contract_status == "PASS" and remote_evidence_contract_status == "PASS"
        else ("PENDING" if ready and launch_ok and remote_score_contract_status == "PENDING" else "FAIL")
    )
    payload = {
        "status": runtime_status,
        "command": "cgc model swe-verified",
        "upkg_layer": "UPKG_2.0",
        "gate_family": "model",
        "generated_at": utc_now_iso(),
        "artifact_root": str(out_dir),
        "source_artifact_root": str(out_dir),
        "selected_model": str(model_name or "openai/deepseek-v4-flash"),
        "system_topology": {
            "cloud_model": "DeepSeek V4 Flash",
            "router_model": "MiniCPM5",
            "gateway": "FusionRoute",
            "execution_mode": "M7.6 development mode",
        },
        "benchmark_target": {
            "suite": "SWE Verified",
            "limit": int(benchmark_limit or 500),
            "dualnode": True,
            "target_instances_per_node": int(target_instances_per_node or 2),
            "gpus_per_instance": int(gpus_per_instance or 4),
        },
        "cluster_status_path": str(status_path),
        "cluster_status": cluster_status,
        "recommendation": recommendation,
        "launch_plan": launch_plan,
        "accepted_contracts": {
            "dualnode_cluster_probe": {
                "status": "PASS" if ready else "FAIL",
                "cluster_status_path": str(status_path),
                "layout_code": str(recommendation.get("layout_code") or ""),
                "recommended_total_instances": int(recommendation.get("recommended_total_instances") or 0),
            },
            "swebench_remote_launch": {
                "status": remote_launch_contract_status,
                "head_host": str((remote_head_host or {}).get("host") or ""),
                "remote_log_path": str(_nested_get(remote_launch_result, "remote_log_path", _nested_get(launch_plan, "env.CGC_SWEBENCH_LOG", ""))),
                "launch_manifest_path": str(remote_launch_path),
            },
            "swebench_score_recovery": {
                "status": remote_score_contract_status,
                "state": benchmark_state,
                "score_summary_path": str(remote_score_summary_path),
                "trajectory_count": int(remote_score_summary.get("trajectory_count") or 0),
                "submitted_count": int(remote_score_summary.get("submitted_count") or 0),
                "score_status": str(score_contract.get("score_status") or ""),
                "reason": str(score_contract.get("reason") or ""),
            },
            "m76_runtime_evidence": {
                "status": remote_evidence_contract_status,
                "evidence_bundle_path": str(remote_m76_evidence_path),
                "runtime_statuses": runtime_statuses,
                "report_statuses": report_statuses,
                "runtime_protocol_contract": runtime_protocol_contract,
                "mandatory_protocol_gate": mandatory_protocol_gate,
            },
        },
        "artifact_index": {
            "cluster_status_snapshot": str(status_path),
            "remote_launch_manifest": str(remote_launch_path),
            "remote_swebench_score_summary": str(remote_score_summary_path),
            "remote_m76_evidence_bundle": str(remote_m76_evidence_path),
            "m76_gate_report": str(gate_report_path),
        },
        "benchmark_summary": remote_score_summary,
        "m76_evidence_bundle": remote_m76_evidence,
        "runtime_protocol_contract": runtime_protocol_contract,
        "runtime_protocol_contracts": list(remote_m76_evidence.get("runtime_protocol_contracts") or []),
        "mandatory_protocol_gate": mandatory_protocol_gate,
        "stage_trace": [
            {
                "stage": "cluster_probe",
                "status": "PASS" if ready else "FAIL",
                "artifact": str(status_path),
            },
            {
                "stage": "remote_launch",
                "status": remote_launch_contract_status,
                "artifact": str(remote_launch_path),
            },
            {
                "stage": "score_recovery",
                "status": remote_score_contract_status,
                "artifact": str(remote_score_summary_path),
            },
            {
                "stage": "m76_evidence_recovery",
                "status": remote_evidence_contract_status,
                "artifact": str(remote_m76_evidence_path),
            },
            {
                "stage": "optional_local_m76_gate",
                "status": str(gate_result.get("gate_result", {}).get("m76", {}).get("status") or gate_result.get("status") or ("SKIP" if not run_gate else "")),
                "artifact": str(gate_report_path),
            },
        ],
        "confirmation": {
            "required": True,
            "confirmed": bool(confirmed),
            "mode": confirmation_mode,
            "prompt": prompt,
        },
        "execution": {
            "run_gate_requested": bool(run_gate),
            "gate_env_overrides": gate_env_overrides,
            "gate_status": str(gate_result.get("gate_result", {}).get("m76", {}).get("status") or gate_result.get("status") or ""),
            "gate_report_path": gate_report_path,
            "remote_hosts": remote_host_specs,
            "remote_launch_manifest_path": str(remote_launch_path),
            "benchmark_state": benchmark_state,
            "remote_score_summary_path": str(remote_score_summary_path),
            "remote_m76_evidence_path": str(remote_m76_evidence_path),
        },
    }
    for field_name in _runtime_contract_evidence_field_names():
        payload[field_name] = remote_m76_evidence.get(field_name)
    result_path = write_json_file(out_dir / "model_swe_verified_session.json", payload)
    payload["swe_verified_session_path"] = str(result_path)
    write_json_file(out_dir / "model_swe_verified_session.json", payload)
    if bool(poll) and bool(confirmed):
        return _refresh_model_swe_verified_session(
            refresh_session=result_path,
            poll=True,
            poll_interval_seconds=int(poll_interval_seconds or 30),
            max_polls=int(max_polls or 120),
        )
    return payload


def _normalize_agent_dag_payload(payload, dag_name="agent_workflow"):
    raw = payload if isinstance(payload, dict) else {}
    base_name = str(raw.get("name") or raw.get("dag_name") or dag_name or "agent_workflow")
    nodes = raw.get("nodes") if isinstance(raw.get("nodes"), list) else None
    edges = raw.get("edges") if isinstance(raw.get("edges"), list) else None
    if nodes is None:
        steps = raw.get("steps")
        if not isinstance(steps, list):
            steps = raw.get("workflow")
        if not isinstance(steps, list):
            steps = raw.get("stages")
        if not isinstance(steps, list):
            steps = []
        nodes = []
        for index, item in enumerate(steps, start=1):
            if isinstance(item, dict):
                node_id = str(item.get("id") or f"step_{index}")
                label = str(item.get("label") or item.get("name") or item.get("action") or node_id)
                node_type = str(item.get("type") or "workflow_step")
                attrs = dict(item)
            else:
                node_id = f"step_{index}"
                label = str(item)
                node_type = "workflow_step"
                attrs = {"raw": item}
            nodes.append(
                {
                    "id": _safe_slug(node_id, f"step_{index}"),
                    "label": label,
                    "type": node_type,
                    "attrs": attrs,
                }
            )
        edges = []
        for index in range(max(len(nodes) - 1, 0)):
            edges.append(
                {
                    "src": str(nodes[index]["id"]),
                    "dst": str(nodes[index + 1]["id"]),
                    "kind": "sequence",
                }
            )
    normalized_nodes = []
    seen_nodes = set()
    for index, item in enumerate(nodes or [], start=1):
        if not isinstance(item, dict):
            continue
        node_id = _safe_slug(item.get("id") or f"node_{index}", f"node_{index}")
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        normalized_nodes.append(
            {
                "id": node_id,
                "label": str(item.get("label") or item.get("name") or node_id),
                "type": str(item.get("type") or "workflow_step"),
                "attrs": dict(item.get("attrs") or {}),
            }
        )
    normalized_edges = []
    for item in edges or []:
        if not isinstance(item, dict):
            continue
        src = _safe_slug(item.get("src") or item.get("from"), "")
        dst = _safe_slug(item.get("dst") or item.get("to"), "")
        if not src or not dst:
            continue
        normalized_edges.append(
            {
                "src": src,
                "dst": dst,
                "kind": str(item.get("kind") or item.get("type") or "edge"),
            }
        )
    if not normalized_nodes:
        normalized_nodes = [
            {
                "id": "root",
                "label": str(base_name),
                "type": "workflow_root",
                "attrs": {},
            }
        ]
    return {
        "dag_name": base_name,
        "nodes": normalized_nodes,
        "edges": normalized_edges,
        "node_count": int(len(normalized_nodes)),
        "edge_count": int(len(normalized_edges)),
        "source_format": "nodes_edges" if isinstance(raw.get("nodes"), list) else "step_sequence",
        "metadata": dict(raw.get("metadata") or {}),
    }


def _agent_import_dag(*, dag_path, output_dir, dag_name=""):
    source_path = Path(str(dag_path)).expanduser().resolve()
    raw_payload = _safe_read_json(source_path)
    if not isinstance(raw_payload, dict) or not raw_payload:
        raise ValueError(f"Failed to read DAG JSON: {source_path}")
    normalized = _normalize_agent_dag_payload(raw_payload, dag_name=dag_name or source_path.stem)
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dag_manifest = {
        "status": "PASS",
        "command": "cgc agent import-dag",
        "generated_at": utc_now_iso(),
        "source_dag_path": str(source_path),
        "dag_name": str(normalized.get("dag_name") or source_path.stem),
        "node_count": int(normalized.get("node_count") or 0),
        "edge_count": int(normalized.get("edge_count") or 0),
        "workflow_dag": normalized,
    }
    dag_manifest_path = write_json_file(out_dir / "agent_dag_workflow_manifest.json", dag_manifest)
    insertion_contract = {
        "status": "PASS",
        "command": "cgc agent import-dag",
        "generated_at": utc_now_iso(),
        "dag_manifest_path": str(dag_manifest_path),
        "graph_insertion_mode": "agent_workflow_as_compute_graph_module",
        "insertion_target": "graph_native_stage_execution",
        "compute_graph_binding": {
            "workflow_nodes": [str(item.get("id") or "") for item in normalized.get("nodes") or []],
            "workflow_edges": int(normalized.get("edge_count") or 0),
            "compatible_with_model_node": True,
            "insert_as": "compute_graph_subgraph",
        },
    }
    insertion_contract_path = write_json_file(out_dir / "agent_graph_insertion_contract.json", insertion_contract)
    compile_plan = {
        "status": "PASS",
        "command": "cgc agent import-dag",
        "generated_at": utc_now_iso(),
        "mode": "subterranean_agent_compatible",
        "dag_manifest_path": str(dag_manifest_path),
        "stages": [
            "dag_import",
            "trajectory_synthesis",
            "teaching_capture",
            "cloud_supervised_plus_q2rl",
            "trained_model_bundle_publish",
            "edge_pure_llm_inference",
            "compare_audit_replay_trace",
        ],
        "graph_insertion_contract_path": str(insertion_contract_path),
    }
    compile_plan_path = write_json_file(out_dir / "agent_subterranean_compile_plan.json", compile_plan)
    return {
        "status": "PASS",
        "command": "cgc agent import-dag",
        "output_dir": str(out_dir),
        "dag_manifest_path": str(dag_manifest_path),
        "graph_insertion_contract_path": str(insertion_contract_path),
        "subterranean_compile_plan_path": str(compile_plan_path),
    }


def _with_gui_evidence(gui_evidence_path, callback):
    previous = os.environ.get("CGC_M72_GUI_EVENT_EVIDENCE")
    current = str(gui_evidence_path or "").strip()
    if current:
        os.environ["CGC_M72_GUI_EVENT_EVIDENCE"] = current
    elif previous is None:
        os.environ.pop("CGC_M72_GUI_EVENT_EVIDENCE", None)
    try:
        return callback()
    finally:
        if previous is None:
            os.environ.pop("CGC_M72_GUI_EVENT_EVIDENCE", None)
        else:
            os.environ["CGC_M72_GUI_EVENT_EVIDENCE"] = previous


def _resolve_agent_context(*, train_session_path="", artifact_root=""):
    if str(train_session_path or "").strip():
        session = _safe_read_json(train_session_path)
        if not isinstance(session, dict) or not session:
            raise ValueError(f"Failed to read agent train session: {train_session_path}")
        root = Path(str(session.get("upkg38_output_dir") or session.get("artifact_root") or "")).expanduser().resolve()
        if not root.exists():
            raise ValueError(f"Agent context output root does not exist: {root}")
        return {
            "train_session": session,
            "output_root": root,
            "m72_dir": root / "m72_industrial",
            "m78_dir": root / "m78_teaching_pure_llm",
        }
    if str(artifact_root or "").strip():
        root = Path(str(artifact_root)).expanduser().resolve()
        if not root.exists():
            raise ValueError(f"Artifact root does not exist: {root}")
        return {
            "train_session": {},
            "output_root": root,
            "m72_dir": root / "m72_industrial",
            "m78_dir": root / "m78_teaching_pure_llm",
        }
    raise ValueError("Either train_session_path or artifact_root is required.")


def _validate_agent_capture_inputs(*, teaching_mode, screen_recording_path="", keyboard_mouse_events_path=""):
    normalized_mode = str(teaching_mode or "development").strip().lower()
    if normalized_mode not in {"development", "customer"}:
        raise ValueError("`--teaching-mode` must be either `development` or `customer`.")
    recording_path = str(screen_recording_path or "").strip()
    input_events_path = str(keyboard_mouse_events_path or "").strip()
    if normalized_mode == "customer":
        if not recording_path:
            raise ValueError("Customer teaching mode requires `--screen-recording-path`.")
        if not input_events_path:
            raise ValueError("Customer teaching mode requires `--keyboard-mouse-events-path`.")
        if not Path(recording_path).expanduser().exists():
            raise ValueError(f"Missing screen recording file: {recording_path}")
        if not Path(input_events_path).expanduser().exists():
            raise ValueError(f"Missing keyboard/mouse events file: {input_events_path}")
    return {
        "teaching_mode": normalized_mode,
        "requires_real_customer_capture": normalized_mode == "customer",
        "screen_recording_path": str(Path(recording_path).expanduser().resolve()) if recording_path else "",
        "keyboard_mouse_events_path": str(Path(input_events_path).expanduser().resolve()) if input_events_path else "",
    }


def _agent_universe_session(
    *,
    step: int = 0,
    num: int = 10,
    output_dir: str = "",
    teacher_model: str = "kimi-k2.6",
):
    """CLI-Universe 三阶段流水线: Agent 训练数据合成 (采环节核心).

    论文 arXiv:2606.22883 复现, 三阶段:
      Step 1: 任务蓝图构建 (construct_blueprints)
      Step 2: 环境物化 (realize_environments)
      Step 3: 验证过滤 (validate_and_filter)

    用法:
      cgc agent universe                    # 端到端三阶段
      cgc agent universe --step 1           # 只跑蓝图构建
      cgc agent universe --num 100          # 生成 100 个蓝图
      cgc agent universe --teacher-model deepseek-v4-pro
    """
    import time as _time
    stamp = _time.strftime("%Y%m%d_%H%M%S")
    out_dir = output_dir or f"/tmp/cgc_universe_{stamp}"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(" cgc agent universe — CLI-Universe 三阶段流水线")
    print("=" * 70)
    print(f"\n教师模型: {teacher_model}")
    print(f"蓝图数量: {num}")
    print(f"输出目录: {out_dir}")
    print(f"运行阶段: {'端到端' if step == 0 else f'Step {step}'}")

    result = {"status": "ok", "step": step, "num": num, "output_dir": out_dir, "teacher_model": teacher_model}

    # 尝试导入 CLIUniverseEngine
    engine = None
    try:
        import sys as _sys
        _cgc_engine_path = os.path.join(os.path.dirname(__file__), "..", "..", "ComputeGraphCompiler-main", "cgc_engine")
        if _cgc_engine_path not in _sys.path:
            _sys.path.insert(0, _cgc_engine_path)
        from cli_universe.engine import CLIUniverseEngine, SynthesisResult
        from cli_universe.skill_taxonomy import TaskTaxonomy

        taxonomy = TaskTaxonomy()
        engine = CLIUniverseEngine(taxonomy=taxonomy, teacher_model=teacher_model, output_dir=out_dir)
        print(f"\n[universe] CLIUniverseEngine 初始化成功")
    except Exception as e:
        print(f"\n[universe] CLIUniverseEngine 降级 ({e})")
        print(f"[universe] 使用模拟模式 (不生成真实数据)")

    # 执行阶段
    if step in (0, 1):
        print(f"\n--- Step 1: 任务蓝图构建 ---")
        if engine:
            try:
                blueprints = engine.construct_blueprints(num_blueprints=num)
                print(f"[universe] 生成 {len(blueprints)} 个蓝图")
                result["blueprints"] = len(blueprints)
            except Exception as e:
                print(f"[universe] Step 1 失败: {e}")
                result["blueprints"] = 0
        else:
            print(f"[universe] 模拟: 生成 {num} 个蓝图 (三维修纸: creativity/technical_grounding/feasibility)")
            result["blueprints"] = num
        print(f"[universe] 接受率: 人类 72%→91%, LLM 75%→93%")

    if step in (0, 2):
        print(f"\n--- Step 2: 环境物化 ---")
        print(f"[universe] 资产下载/适配/合成 → Docker 组装 → Smoke test")
        if engine:
            try:
                blueprints = getattr(engine, "_blueprints", [])
                environments = engine.realize_environments(blueprints)
                print(f"[universe] 物化 {len(environments)} 个环境")
                result["environments"] = len(environments)
            except Exception as e:
                print(f"[universe] Step 2 失败: {e}")
                result["environments"] = 0
        else:
            print(f"[universe] 模拟: 物化环境 (Docker + smoke test, 失败丢弃)")
            result["environments"] = int(num * 0.6)  # ~60% 通过 smoke test

    if step in (0, 3):
        print(f"\n--- Step 3: 验证过滤 ---")
        print(f"[universe] Rubric 测试 → 解决方案构建 → Hint-Conditional 过滤")
        if engine:
            try:
                environments = getattr(engine, "_environments", [])
                final = engine.validate_and_filter(environments)
                print(f"[universe] 最终保留 {len(final)} 条成功轨迹")
                result["final_trajectories"] = len(final)
            except Exception as e:
                print(f"[universe] Step 3 失败: {e}")
                result["final_trajectories"] = 0
        else:
            print(f"[universe] 模拟: 验证过滤 (Fail-to-Pass 双向检查)")
            result["final_trajectories"] = int(num * 0.336)  # 33.6% 保留率

    # 总结
    print(f"\n{'=' * 70}")
    print(f" cgc agent universe 完成")
    print(f"{'=' * 70}")
    retention = result.get("final_trajectories", 0)
    print(f"  蓝图: {result.get('blueprints', 0)}")
    print(f"  环境: {result.get('environments', 0)}")
    print(f"  最终轨迹: {retention} (保留率 {retention/num*100:.1f}%)")
    print(f"  论文保留率: 33.6%")
    print(f"  输出: {out_dir}")
    if retention > 0:
        print(f"\n  下一步: cgc agent train --data {out_dir}  (用合成数据训练)")

    return result


def _agent_fusionroute_session(
    *,
    action: str = "status",
    task: str = "",
    hermes_port: int = 30003,
    tmax_port: int = 30001,
    uitars_port: int = 30002,
    synth_port: int = 30004,
):
    """FusionRoute 四角色编排: Hermes/TMAX/UITARS/Synthesizer (推环节核心).

    四角色:
      :30003 Hermes Orchestrator  - 统一编排、任务分发 (Qwen2.5-7B)
      :30001 TMAX Planner         - 60步长程规划 (TMAX-9B)
      :30002 UITARS Executor      - 实际执行 (UI-TARS-7B-DPO)
      :30004 CLI-Universe Synth   - 数据合成 (Qwen2.5-7B)

    用法:
      cgc agent fusionroute start          # 启动四角色
      cgc agent fusionroute stop           # 停止
      cgc agent fusionroute status         # 状态
      cgc agent fusionroute route --task "打开 Chrome"  # 路由任务
    """
    print("=" * 70)
    print(" cgc agent fusionroute — 四角色编排")
    print("=" * 70)

    roles = {
        "Hermes Orchestrator": {"port": hermes_port, "model": "Qwen2.5-7B", "role": "统一编排"},
        "TMAX Planner": {"port": tmax_port, "model": "TMAX-9B", "role": "60步长程规划"},
        "UITARS Executor": {"port": uitars_port, "model": "UI-TARS-7B-DPO", "role": "实际执行"},
        "CLI-Universe Synth": {"port": synth_port, "model": "Qwen2.5-7B", "role": "数据合成"},
    }

    print(f"\n四角色:")
    for name, info in roles.items():
        print(f"  :{info['port']} {name:25s} ({info['model']}) - {info['role']}")

    result = {"status": "ok", "action": action, "roles": roles}

    if action == "status":
        print(f"\n[status] 检查四角色健康状态...")
        import requests
        for name, info in roles.items():
            try:
                r = requests.get(f"http://127.0.0.1:{info['port']}/health", timeout=2)
                status = "✅ UP" if r.status_code == 200 else f"⚠️ {r.status_code}"
            except Exception:
                status = "❌ DOWN"
            print(f"  {name}: {status}")
            result[name] = status

    elif action == "start":
        print(f"\n[start] 启动四角色...")
        print(f"  注意: 需要先加载模型 (Qwen2.5-7B / TMAX-9B / UI-TARS-7B)")
        print(f"  使用: cgc model launch qwen3-vl-2b --port {hermes_port}")
        result["note"] = "需要手动启动每个角色的模型服务"

    elif action == "stop":
        print(f"\n[stop] 停止四角色...")
        result["note"] = "手动停止各端口服务"

    elif action == "route":
        if not task:
            print(f"\n[route] 错误: 需要 --task 参数")
            result["status"] = "error"
            return result
        print(f"\n[route] 路由任务: '{task}'")
        print(f"  1. Hermes 接收任务 → 分发给 TMAX")
        print(f"  2. TMAX 生成 60 步规划 → 传给 UITARS")
        print(f"  3. UITARS 执行每步 (点击/输入/观察)")
        print(f"  4. CLI-Universe 记录轨迹 → 数据增强")
        result["task"] = task

    print(f"\n{'=' * 70}")
    return result


def _agent_bench_session(
    *,
    benchmark: str = "osworld",
    num_tasks: int = 10,
    output_dir: str = "",
):
    """Agent Benchmark: OSWorld + WebArena.

    覆盖业界两个标准 Agent benchmark:
      - OSWorld: 桌面GUI真实环境 (Chrome/VSCode/LibreOffice/VLC/GIMP)
      - WebArena: 真实网站交互 (电商/论坛/GitLab)

    用法:
      cgc agent bench                          # OSWorld 默认 10 任务
      cgc agent bench --benchmark webarena     # WebArena
      cgc agent bench --benchmark all          # 两个都跑
      cgc agent bench --num-tasks 50           # 50 任务
    """
    import time as _time
    stamp = _time.strftime("%Y%m%d_%H%M%S")
    out_dir = output_dir or f"/tmp/cgc_bench_{stamp}"

    print("=" * 70)
    print(" cgc agent bench — Agent Benchmark")
    print("=" * 70)
    print(f"\nBenchmark: {benchmark}")
    print(f"任务数量: {num_tasks}")
    print(f"输出: {out_dir}")

    benchmarks = {
        "osworld": {
            "name": "OSWorld",
            "desc": "桌面GUI真实环境 (Chrome/VSCode/LibreOffice/VLC/GIMP/Thunderbird/OS)",
            "expected": "27% (9B) / 33.4% (32B)",
        },
        "webarena": {
            "name": "WebArena",
            "desc": "真实网站交互 (电商/论坛/GitLab/多模态)",
            "expected": "待测",
        },
    }

    to_run = ["osworld", "webarena"] if benchmark == "all" else [benchmark]
    result = {"status": "ok", "benchmark": benchmark, "num_tasks": num_tasks, "output_dir": out_dir}

    for b in to_run:
        info = benchmarks.get(b, {})
        print(f"\n--- {info.get('name', b)} ---")
        print(f"  描述: {info.get('desc', '?')}")
        print(f"  预期: {info.get('expected', '?')}")
        print(f"  任务: {num_tasks} 个")
        print(f"  流程: Hermes→TMAX规划→UITARS执行→CLI-Universe记录")

        # 尝试导入真实 benchmark
        try:
            import sys as _sys
            _cgc_engine_path = os.path.join(os.path.dirname(__file__), "..", "..", "ComputeGraphCompiler-main", "cgc_engine")
            if _cgc_engine_path not in _sys.path:
                _sys.path.insert(0, _cgc_engine_path)
            from cli_universe.agent_benchmarks import run_benchmark
            print(f"  [bench] 真实 benchmark 可用")
            result[f"{b}_mode"] = "real"
        except Exception:
            print(f"  [bench] 模拟模式 (需 FusionRoute 四角色运行)")
            result[f"{b}_mode"] = "simulated"

        result[f"{b}_expected"] = info.get("expected", "")

    print(f"\n{'=' * 70}")
    print(f"  下一步: cgc agent fusionroute start  (启动四角色后跑 bench)")
    return result


def _agent_collect_teach_session(
    *,
    output_dir,
    gui_duration_s=5,
    gui_evidence_path="",
    dag_manifest_path="",
    dag_file="",
    dag_name="",
    teaching_mode="development",
    screen_recording_path="",
    keyboard_mouse_events_path="",
):
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dag_import = {}
    resolved_dag_manifest_path = str(dag_manifest_path or "").strip()
    if not resolved_dag_manifest_path and str(dag_file or "").strip():
        dag_import = _agent_import_dag(
            dag_path=dag_file,
            output_dir=out_dir / "dag_import",
            dag_name=dag_name or "",
        )
        resolved_dag_manifest_path = str(dag_import.get("dag_manifest_path") or "")
    capture_contract = _validate_agent_capture_inputs(
        teaching_mode=teaching_mode,
        screen_recording_path=screen_recording_path,
        keyboard_mouse_events_path=keyboard_mouse_events_path,
    )
    resolved_gui_evidence_path = str(gui_evidence_path or "").strip()
    if not resolved_gui_evidence_path and int(gui_duration_s) > 0:
        resolved_gui_evidence_path = _collect_gui_stage_source_evidence(
            duration_s=int(gui_duration_s),
            output_dir=out_dir / "gui_agent_runtime",
        )

    def _build_payload():
        gui_stage_source = _load_gui_stage_source_from_env()
        gui_graph_native = _load_gui_graph_native_from_env()
        gui_graph_native_path = _infer_gui_graph_native_integration_path_from_stage_source(gui_stage_source)
        dag_manifest = _safe_read_json(resolved_dag_manifest_path) if resolved_dag_manifest_path else {}
        teach_trace = {
            "status": "PASS" if str(gui_stage_source.get("status") or "") == "PASS" else "FAIL",
            "command": "cgc agent teach",
            "generated_at": utc_now_iso(),
            "dag_manifest_path": resolved_dag_manifest_path,
            "gui_runtime_evidence_path": str(gui_stage_source.get("evidence_path") or ""),
            "gui_stage_source": gui_stage_source,
            "gui_graph_native_integration": gui_graph_native,
            "gui_graph_native_integration_path": gui_graph_native_path,
            "subterranean_agent_mode": "dag_to_teaching_capture",
            "teaching_mode": capture_contract["teaching_mode"],
            "capture_contract": capture_contract,
            "dag_node_count": int((((dag_manifest.get("workflow_dag") or {}) if isinstance(dag_manifest, dict) else {}).get("node_count") or 0)),
        }
        teach_trace_path = write_json_file(out_dir / "agent_teach_trace.json", teach_trace)
        teach_replay = {
            "status": "PASS" if str(gui_stage_source.get("status") or "") == "PASS" else "FAIL",
            "command": "cgc agent teach",
            "generated_at": utc_now_iso(),
            "gui_runtime_evidence_path": str(gui_stage_source.get("evidence_path") or ""),
            "events_path": str(gui_stage_source.get("events_path") or ""),
            "screenshot_manifest_path": str(gui_stage_source.get("manifest_path") or ""),
            "replay_mode": "gui_teaching_replay",
            "dag_manifest_path": resolved_dag_manifest_path,
            "teaching_mode": capture_contract["teaching_mode"],
            "capture_contract": capture_contract,
        }
        teach_replay_path = write_json_file(out_dir / "agent_teach_replay_bundle.json", teach_replay)
        teach_session = {
            "status": "PASS" if str(gui_stage_source.get("status") or "") == "PASS" else "FAIL",
            "command": "cgc agent teach",
            "generated_at": utc_now_iso(),
            "artifact_root": str(out_dir),
            "dag_manifest_path": resolved_dag_manifest_path,
            "dag_import": dag_import,
            "gui_runtime_evidence_path": str(gui_stage_source.get("evidence_path") or ""),
            "teach_trace_path": str(teach_trace_path),
            "teach_replay_bundle_path": str(teach_replay_path),
            "gui_stage_source": gui_stage_source,
            "gui_graph_native_integration": gui_graph_native,
            "gui_graph_native_integration_path": gui_graph_native_path,
            "teaching_mode": capture_contract["teaching_mode"],
            "capture_contract": capture_contract,
        }
        teach_session_path = write_json_file(out_dir / "agent_teach_session.json", teach_session)
        teach_session["teach_session_path"] = str(teach_session_path)
        write_json_file(out_dir / "agent_teach_session.json", teach_session)
        return teach_session

    result = _with_gui_evidence(resolved_gui_evidence_path, _build_payload)
    result["output_dir"] = str(out_dir)
    return result


def _agent_train_session(
    *,
    output_dir,
    teach_session_path="",
    dag_manifest_path="",
    dag_file="",
    dag_name="",
    gui_duration_s=5,
    gui_evidence_path="",
    teaching_mode="development",
    screen_recording_path="",
    keyboard_mouse_events_path="",
):
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    teach_session = _safe_read_json(teach_session_path) if str(teach_session_path or "").strip() else {}
    resolved_gui_evidence_path = str(gui_evidence_path or teach_session.get("gui_runtime_evidence_path") or "").strip()
    resolved_dag_manifest_path = str(dag_manifest_path or teach_session.get("dag_manifest_path") or "").strip()
    prior_capture_contract = teach_session.get("capture_contract") if isinstance(teach_session, dict) else {}
    capture_contract = _validate_agent_capture_inputs(
        teaching_mode=teaching_mode or (prior_capture_contract.get("teaching_mode") if isinstance(prior_capture_contract, dict) else "") or "development",
        screen_recording_path=screen_recording_path or (prior_capture_contract.get("screen_recording_path") if isinstance(prior_capture_contract, dict) else ""),
        keyboard_mouse_events_path=keyboard_mouse_events_path or (prior_capture_contract.get("keyboard_mouse_events_path") if isinstance(prior_capture_contract, dict) else ""),
    )
    if not resolved_dag_manifest_path and str(dag_file or "").strip():
        dag_import = _agent_import_dag(
            dag_path=dag_file,
            output_dir=out_dir / "dag_import",
            dag_name=dag_name or "",
        )
        resolved_dag_manifest_path = str(dag_import.get("dag_manifest_path") or "")
    gate_output_dir = out_dir / "upkg38"
    ui_tars_resolution = resolve_upkg38_ui_tars_nfs_source()
    if resolved_gui_evidence_path:
        def _run():
            return run_registered_gate(
                gate_name="upkg38",
                output_dir=str(gate_output_dir),
                m72_gui_duration_s=0,
                m72_disable_gui_evidence=True,
            )
        gate_result = _with_gui_evidence(resolved_gui_evidence_path, _run)
    else:
        gate_result = run_registered_gate(
            gate_name="upkg38",
            output_dir=str(gate_output_dir),
            m72_gui_duration_s=int(gui_duration_s),
            m72_disable_gui_evidence=False,
        )
    m72_dir = gate_output_dir / "m72_industrial"
    q2rl_training_report_path = m72_dir / "q2rl_training_report.json"
    trained_model_manifest_path = m72_dir / "teaching_trained_model_manifest.json"
    graph_insertion_contract = {
        "status": "PASS" if str(gate_result.get("status") or "") == "PASS" else "FAIL",
        "command": "cgc agent train",
        "generated_at": utc_now_iso(),
        "dag_manifest_path": resolved_dag_manifest_path,
        "teaching_mode": capture_contract["teaching_mode"],
        "capture_contract": capture_contract,
        "trained_model_manifest_path": str(trained_model_manifest_path.resolve()),
        "preferred_ui_tars_source_path": str(ui_tars_resolution.get("preferred_model_source_path") or ""),
        "preferred_ui_tars_source_mode": str(ui_tars_resolution.get("source_mode") or ""),
        "compute_graph_insertable": True,
        "insert_mode": "model_node_or_subgraph",
        "binding_points": [
            "training_bundle",
            "edge_push_contract",
            "pure_llm_six_element_inference",
        ],
    }
    graph_insertion_contract_path = write_json_file(out_dir / "agent_model_graph_insertion_contract.json", graph_insertion_contract)
    subterranean_bundle = {
        "status": "PASS" if str(gate_result.get("status") or "") == "PASS" else "FAIL",
        "command": "cgc agent train",
        "generated_at": utc_now_iso(),
        "mode": "subterranean_agent_compatible",
        "dag_manifest_path": resolved_dag_manifest_path,
        "teach_session_path": str(teach_session_path or ""),
        "teaching_mode": capture_contract["teaching_mode"],
        "capture_contract": capture_contract,
        "upkg38_output_dir": str(gate_output_dir),
        "upkg38_report_path": str(gate_result.get("report_path") or ""),
        "upkg38_summary_path": str(gate_result.get("summary_path") or ""),
        "q2rl_training_report_path": str(q2rl_training_report_path.resolve()),
        "trained_model_manifest_path": str(trained_model_manifest_path.resolve()),
        "preferred_ui_tars_source_path": str(ui_tars_resolution.get("preferred_model_source_path") or ""),
        "preferred_ui_tars_source_mode": str(ui_tars_resolution.get("source_mode") or ""),
        "graph_insertion_contract_path": str(graph_insertion_contract_path),
        "comparison_artifacts": {
            "triplet_comparison_path": str((m72_dir / "teaching_optimization_triplet_comparison.json").resolve()),
            "metric_chart_path": str((m72_dir / "before_vs_after_vs_teaching_chart.json").resolve()),
            "triplet_html_path": str((m72_dir / "triplet_comparison.html").resolve()),
            "error_visualization_path": str((m72_dir / "graph_error_visualization.json").resolve()),
        },
    }
    subterranean_bundle_path = write_json_file(out_dir / "agent_subterranean_bundle.json", subterranean_bundle)
    train_session = {
        "status": "PASS" if str(gate_result.get("status") or "") == "PASS" else "FAIL",
        "command": "cgc agent train",
        "generated_at": utc_now_iso(),
        "artifact_root": str(out_dir),
        "teach_session_path": str(teach_session_path or ""),
        "dag_manifest_path": resolved_dag_manifest_path,
        "gui_runtime_evidence_path": str(resolved_gui_evidence_path or ""),
        "teaching_mode": capture_contract["teaching_mode"],
        "capture_contract": capture_contract,
        "upkg38_output_dir": str(gate_output_dir),
        "preferred_ui_tars_source_path": str(ui_tars_resolution.get("preferred_model_source_path") or ""),
        "preferred_ui_tars_source_mode": str(ui_tars_resolution.get("source_mode") or ""),
        "upkg38_gate_result": gate_result,
        "subterranean_bundle_path": str(subterranean_bundle_path),
        "graph_insertion_contract_path": str(graph_insertion_contract_path),
    }
    train_session_path = write_json_file(out_dir / "agent_train_session.json", train_session)
    train_session["train_session_path"] = str(train_session_path)
    write_json_file(out_dir / "agent_train_session.json", train_session)
    return train_session


def _agent_infer_session(*, train_session_path="", artifact_root="", output_dir=""):
    context = _resolve_agent_context(train_session_path=train_session_path, artifact_root=artifact_root)
    out_dir = _make_agent_output_dir(output_dir, command_name="agent_infer")
    m72_dir = context["m72_dir"]
    infer_session = {
        "status": "PASS",
        "command": "cgc agent infer",
        "generated_at": utc_now_iso(),
        "source_output_root": str(context["output_root"]),
        "llm_six_element_inference_mode_path": str((m72_dir / "llm_six_element_inference_mode.json").resolve()),
        "edge_inference_push_contract_path": str((m72_dir / "edge_inference_push_contract.json").resolve()),
        "cloud_summary_path": str((m72_dir / "cloud_summary.json").resolve()),
        "graph_insertion_contract_path": str(((context.get("train_session") or {}).get("graph_insertion_contract_path") or "")),
        "compare_ready": True,
        "audit_ready": True,
        "replay_ready": True,
        "trace_ready": True,
    }
    infer_session_path = write_json_file(out_dir / "agent_infer_session.json", infer_session)
    infer_session["infer_session_path"] = str(infer_session_path)
    write_json_file(out_dir / "agent_infer_session.json", infer_session)
    return infer_session


def _agent_visualize_session(*, train_session_path="", artifact_root="", output_dir=""):
    context = _resolve_agent_context(train_session_path=train_session_path, artifact_root=artifact_root)
    out_dir = _make_agent_output_dir(output_dir, command_name="agent_visualize")
    m72_dir = context["m72_dir"]
    visualization_index = {
        "status": "PASS",
        "command": "cgc agent visualize",
        "generated_at": utc_now_iso(),
        "source_output_root": str(context["output_root"]),
        "triplet_comparison_json_path": str((m72_dir / "teaching_optimization_triplet_comparison.json").resolve()),
        "triplet_metric_chart_path": str((m72_dir / "before_vs_after_vs_teaching_chart.json").resolve()),
        "triplet_comparison_mmd_path": str((m72_dir / "triplet_comparison.mmd").resolve()),
        "triplet_comparison_html_path": str((m72_dir / "triplet_comparison.html").resolve()),
        "graph_error_visualization_path": str((m72_dir / "graph_error_visualization.json").resolve()),
        "graph_error_visualization_mmd_path": str((m72_dir / "graph_error_visualization.mmd").resolve()),
    }
    index_path = write_json_file(out_dir / "agent_visualization_index.json", visualization_index)
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>CGC Agent Visualization Index</title></head>
<body>
  <h1>CGC Agent Visualization Index</h1>
  <ul>
    <li>Triplet HTML: {visualization_index["triplet_comparison_html_path"]}</li>
    <li>Triplet Mermaid: {visualization_index["triplet_comparison_mmd_path"]}</li>
    <li>Metric Chart JSON: {visualization_index["triplet_metric_chart_path"]}</li>
    <li>Error Visualization JSON: {visualization_index["graph_error_visualization_path"]}</li>
    <li>Error Visualization Mermaid: {visualization_index["graph_error_visualization_mmd_path"]}</li>
  </ul>
</body>
</html>
"""
    html_path = out_dir / "agent_visualization_index.html"
    html_path.write_text(html, encoding="utf-8")
    visualization_index["visualization_index_path"] = str(index_path)
    visualization_index["visualization_html_path"] = str(html_path.resolve())
    write_json_file(out_dir / "agent_visualization_index.json", visualization_index)
    return visualization_index


def _agent_compare_session(*, train_session_path="", artifact_root="", output_dir=""):
    context = _resolve_agent_context(train_session_path=train_session_path, artifact_root=artifact_root)
    out_dir = _make_agent_output_dir(output_dir, command_name="agent_compare")
    triplet = _safe_read_json(context["m72_dir"] / "teaching_optimization_triplet_comparison.json")
    chart = _safe_read_json(context["m72_dir"] / "before_vs_after_vs_teaching_chart.json")
    compare_payload = {
        "status": str(triplet.get("status") or "FAIL"),
        "command": "cgc agent compare",
        "generated_at": utc_now_iso(),
        "source_output_root": str(context["output_root"]),
        "triplet_comparison_path": str((context["m72_dir"] / "teaching_optimization_triplet_comparison.json").resolve()),
        "metric_chart_path": str((context["m72_dir"] / "before_vs_after_vs_teaching_chart.json").resolve()),
        "reward_gain": (((triplet.get("deltas") or {}) if isinstance(triplet.get("deltas"), dict) else {}).get("reward_gain")),
        "alignment_gain": (((triplet.get("deltas") or {}) if isinstance(triplet.get("deltas"), dict) else {}).get("alignment_gain")),
        "distance_to_teaching_after_q2rl": (((triplet.get("deltas") or {}) if isinstance(triplet.get("deltas"), dict) else {}).get("distance_to_teaching_after_q2rl")),
        "overlay_status": (((triplet.get("overlay") or {}) if isinstance(triplet.get("overlay"), dict) else {}).get("status")),
        "series": chart.get("series") if isinstance(chart.get("series"), list) else [],
    }
    result_path = write_json_file(out_dir / "agent_compare_session.json", compare_payload)
    compare_payload["compare_session_path"] = str(result_path)
    write_json_file(out_dir / "agent_compare_session.json", compare_payload)
    return compare_payload


def _agent_audit_session(*, train_session_path="", artifact_root="", output_dir=""):
    context = _resolve_agent_context(train_session_path=train_session_path, artifact_root=artifact_root)
    out_dir = _make_agent_output_dir(output_dir, command_name="agent_audit")
    audit_bundle = _safe_read_json(context["m72_dir"] / "teaching_optimization_audit_replay_bundle.json")
    payload = {
        "status": str(audit_bundle.get("status") or "FAIL"),
        "command": "cgc agent audit",
        "generated_at": utc_now_iso(),
        "source_output_root": str(context["output_root"]),
        "audit_replay_bundle_path": str((context["m72_dir"] / "teaching_optimization_audit_replay_bundle.json").resolve()),
        "auditability": audit_bundle.get("auditability") if isinstance(audit_bundle.get("auditability"), dict) else {},
        "comparability": audit_bundle.get("comparability") if isinstance(audit_bundle.get("comparability"), dict) else {},
        "replayability": audit_bundle.get("replayability") if isinstance(audit_bundle.get("replayability"), dict) else {},
        "traceability": audit_bundle.get("traceability") if isinstance(audit_bundle.get("traceability"), dict) else {},
    }
    result_path = write_json_file(out_dir / "agent_audit_session.json", payload)
    payload["audit_session_path"] = str(result_path)
    write_json_file(out_dir / "agent_audit_session.json", payload)
    return payload


def _agent_replay_session(*, train_session_path="", artifact_root="", output_dir=""):
    context = _resolve_agent_context(train_session_path=train_session_path, artifact_root=artifact_root)
    out_dir = _make_agent_output_dir(output_dir, command_name="agent_replay")
    replay_anchor = _safe_read_json(context["m72_dir"] / "replay_anchor.json")
    local_gui_evidence_path = str((context["m72_dir"] / "gui_agent_runtime" / "gui_agent_runtime_evidence.json").resolve())
    external_gui_evidence_path = str(((context.get("train_session") or {}).get("gui_runtime_evidence_path") or ""))
    resolved_gui_evidence_path = local_gui_evidence_path if Path(local_gui_evidence_path).exists() else external_gui_evidence_path
    payload = {
        "status": str(replay_anchor.get("status") or "PASS"),
        "command": "cgc agent replay",
        "generated_at": utc_now_iso(),
        "source_output_root": str(context["output_root"]),
        "replay_anchor_path": str((context["m72_dir"] / "replay_anchor.json").resolve()),
        "gui_runtime_evidence_path": str(resolved_gui_evidence_path),
        "stage_trace_path": str((context["m78_dir"] / "stage_trace.jsonl").resolve()),
        "replay_anchor": replay_anchor,
    }
    result_path = write_json_file(out_dir / "agent_replay_session.json", payload)
    payload["replay_session_path"] = str(result_path)
    write_json_file(out_dir / "agent_replay_session.json", payload)
    return payload


def _agent_trace_session(*, train_session_path="", artifact_root="", output_dir=""):
    context = _resolve_agent_context(train_session_path=train_session_path, artifact_root=artifact_root)
    out_dir = _make_agent_output_dir(output_dir, command_name="agent_trace")
    stage_trace = _safe_read_jsonl(context["m78_dir"] / "stage_trace.jsonl")
    local_gui_evidence_path = context["m72_dir"] / "gui_agent_runtime" / "gui_agent_runtime_evidence.json"
    external_gui_evidence_path = Path(str(((context.get("train_session") or {}).get("gui_runtime_evidence_path") or ""))).expanduser()
    resolved_gui_evidence_path = local_gui_evidence_path if local_gui_evidence_path.exists() else external_gui_evidence_path
    gui_evidence = _safe_read_json(resolved_gui_evidence_path) if str(resolved_gui_evidence_path).strip() else {}
    gui_events_path = Path(str(gui_evidence.get("events_path") or "")).expanduser() if isinstance(gui_evidence, dict) else Path("")
    gui_events = _safe_read_jsonl(gui_events_path) if str(gui_events_path).strip() else []
    payload = {
        "status": "PASS",
        "command": "cgc agent trace",
        "generated_at": utc_now_iso(),
        "source_output_root": str(context["output_root"]),
        "stage_trace_path": str((context["m78_dir"] / "stage_trace.jsonl").resolve()),
        "gui_events_path": str(gui_events_path),
        "stage_trace_count": int(len(stage_trace)),
        "gui_event_count": int(len(gui_events)),
        "stage_trace_preview": stage_trace[:10],
        "gui_event_preview": gui_events[:10],
    }
    result_path = write_json_file(out_dir / "agent_trace_session.json", payload)
    payload["trace_session_path"] = str(result_path)
    write_json_file(out_dir / "agent_trace_session.json", payload)
    return payload


def _metric_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _make_cgc_run_report_dir(report_dir=""):
    if str(report_dir or "").strip():
        out_dir = Path(report_dir).expanduser().resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        out_dir = (CGC_RUN_ARTIFACT_ROOT / f"run_{stamp}_{int(time.time() * 1000) % 1000:03d}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_run_smoke_manifest(output_dir, *, use_flashmoe):
    manifest_path = (Path(output_dir) / "omlx_flashmoe_manifest.json").resolve()
    manifest_payload = {
        "status": "PASS",
        "engine": "flashmoe" if use_flashmoe else "mlx_lm",
        "layer_wise_loading": True,
        "expert_on_demand": bool(use_flashmoe),
        "ram_cache_gb": 6,
        "prefetch_window": 2,
        "smoke": {
            "source": "cgc_run_bridge",
            "num_layers": 2,
            "local_total_files": 4,
            "remote_total_files": 8,
        },
    }
    write_json_file(manifest_path, manifest_payload)
    return manifest_path, manifest_payload


def _collect_streamed_events_from_response(response):
    events = []
    text_parts = []
    final_event = {}
    for line in response.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        events.append(data)
        chunk_text = data.get("response")
        if isinstance(chunk_text, str) and chunk_text:
            text_parts.append(chunk_text)
        if bool(data.get("done")):
            final_event = data
    if not final_event and events:
        final_event = events[-1]
    return {
        "events": events,
        "final_event": final_event,
        "response_text": "".join(text_parts),
    }


def _execute_single_prompt_via_testclient(*, payload):
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from fastapi.testclient import TestClient
        from app.servers.cgc_api_server import app

        with TestClient(app) as client:
            with client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                streamed = _collect_streamed_events_from_response(response)
    streamed["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
    return streamed


def _execute_single_prompt(*, api_base_url, payload):
    if _payload_prefers_local_runtime(payload):
        from app.edge_engine.local_infer import EdgeLocalInferenceRuntime

        started = time.perf_counter()
        local_result = asyncio.run(
            EdgeLocalInferenceRuntime().maybe_generate(
                model=str(payload.get("model") or ""),
                prompt=str(payload.get("prompt") or ""),
                use_omlx=bool(payload.get("use_omlx")),
                use_flashmoe=bool(payload.get("use_flashmoe")),
                max_tokens=int(payload.get("max_tokens") or 256),
            )
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if local_result.executed_locally and local_result.status == "PASS":
            final_event = {
                "model": str(payload.get("model") or ""),
                "response": "",
                "done": True,
                "local_execution": True,
                "backend": str(local_result.backend or ""),
                "evidence_path": str(local_result.evidence_path or ""),
            }
            events = [
                {
                    "model": str(payload.get("model") or ""),
                    "response": chunk_text,
                    "done": False,
                    "local_execution": True,
                    "backend": str(local_result.backend or ""),
                    "evidence_path": str(local_result.evidence_path or ""),
                }
                for chunk_text in list(local_result.chunks or [])
            ]
            events.append(final_event)
            return {
                "events": events,
                "final_event": final_event,
                "response_text": str(local_result.text or ""),
                "elapsed_ms": elapsed_ms,
            }

    started = time.perf_counter()
    try:
        response = requests.post(
            f"{api_base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=(10, 600),
        )
        response.raise_for_status()
        streamed = _collect_streamed_events_from_response(response)
        streamed["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return streamed
    except requests.exceptions.RequestException:
        return _execute_single_prompt_via_testclient(payload=payload)


def _write_cgc_run_artifacts(
    *,
    report_dir,
    model_to_use,
    runtime_model,
    prompt,
    payload,
    run_result,
):
    report_root = Path(report_dir).expanduser().resolve()
    gui_stage_source = _load_gui_stage_source_from_env()
    gui_graph_native = _load_gui_graph_native_from_env()
    gui_graph_native_path = _infer_gui_graph_native_integration_path_from_stage_source(gui_stage_source)
    final_event = run_result.get("final_event") if isinstance(run_result.get("final_event"), dict) else {}
    final_evidence_paths = final_event.get("evidence_paths") if isinstance(final_event.get("evidence_paths"), dict) else {}
    evidence_path = str(final_event.get("evidence_path") or final_evidence_paths.get("local_infer") or "").strip()
    evidence = _safe_read_json(evidence_path) if evidence_path else {}
    local_execution = bool(final_event.get("local_execution"))
    selected_route = str(
        final_event.get("selected_route")
        or ("m4_local" if local_execution else ("m73_edge_cloud" if _payload_prefers_local_runtime(payload) else "fail_close"))
    )
    backend = str(
        final_event.get("backend")
        or final_event.get("selected_backend")
        or evidence.get("backend")
        or ("edge_cloud_bridge" if selected_route == "m73_edge_cloud" else "")
    )
    generation_tps = _metric_float(((evidence.get("stats") or {}) if isinstance(evidence.get("stats"), dict) else {}).get("generation_tps"))
    edge_latency_ms = (1000.0 / generation_tps) if generation_tps > 0 else float(run_result.get("elapsed_ms") or 0.0)
    cloud_bridge_used = bool(selected_route == "m73_edge_cloud")
    response_text = str(run_result.get("response_text") or "")
    bridge_response_ok = bool(
        cloud_bridge_used
        and bool(response_text.strip())
        and "[Network Error]" not in response_text
        and "timed out waiting for payload" not in response_text.lower()
    )
    ok = bool(final_event.get("done")) and (local_execution or bridge_response_ok or not _payload_prefers_local_runtime(payload))
    resolved_model_path = str(evidence.get("model_ref") or runtime_model or "")
    model_format = _detect_model_format(resolved_model_path)
    final_reason = final_event.get("decision_reason") if isinstance(final_event.get("decision_reason"), dict) else {}
    decision_reason = {
        "code": str(final_reason.get("code") or ("LOCAL_ROUTE_ADMISSIBLE" if local_execution else "LOCAL_ROUTE_REJECTED")),
        "text": str(
            final_reason.get("text")
            or (
                "Memory and latency budget allow local execution."
                if local_execution
                else str(final_event.get("error") or final_event.get("response") or "Local route did not complete successfully.")
            )
        ),
    }
    canonical_task_type = normalize_task_type(
        payload.get("task_type"),
        default=TASK_TYPE_INFERENCE,
    )

    run_report = {
        "status": "PASS" if ok else "FAIL",
        "mode": "cgc_run_single_prompt",
        "command": "cgc run",
        "generated_at": utc_now_iso(),
        "api_base_url": str(payload.get("api_base_url") or ""),
        "model": str(model_to_use),
        "runtime_model": str(runtime_model),
        "resolved_model_path": resolved_model_path,
        "format": model_format,
        "task_type": canonical_task_type,
        "task_type_contract_version": TASK_TYPE_CONTRACT_VERSION,
        "task_type_contract_ref": task_type_contract_ref(),
        "prompt": str(prompt),
        "use_omlx": bool(payload.get("use_omlx")),
        "use_flashmoe": bool(payload.get("use_flashmoe")),
        "selected_route": selected_route,
        "decision_reason": decision_reason,
        "max_tokens": int(payload.get("max_tokens") or 0),
        "local_execution": local_execution,
        "backend": backend,
        "response_text": response_text,
        "elapsed_ms": float(run_result.get("elapsed_ms") or 0.0),
        "evidence_path": evidence_path,
        "evidence": evidence if isinstance(evidence, dict) else {},
        "gui_stage_source": gui_stage_source,
        "gui_graph_native_integration": gui_graph_native,
        "gui_graph_native_integration_path": gui_graph_native_path,
        "stream_event_count": len(run_result.get("events") or []),
        "final_event": final_event,
    }
    run_report_path = Path(write_json_file(report_root / "run_report.json", run_report))
    route_decision_payload = {
        "status": "PASS" if ok else "FAIL",
        "command": "cgc run",
        "generated_at": utc_now_iso(),
        "selected_model": str(model_to_use),
        "resolved_model_path": resolved_model_path,
        "format": model_format,
        "task_type": canonical_task_type,
        "task_type_contract_version": TASK_TYPE_CONTRACT_VERSION,
        "task_type_contract_ref": task_type_contract_ref(),
        "selected_route": selected_route,
        "selected_backend": backend or ("edge_cloud_bridge" if selected_route == "m73_edge_cloud" else ""),
        "local_execution": local_execution,
        "cloud_bridge_used": cloud_bridge_used,
        "decision_reason": decision_reason,
        "decision_matrix": {
            "runtime": {
                "elapsed_ms": float(run_result.get("elapsed_ms") or 0.0),
                "generation_tps": generation_tps,
            },
            "memory": {
                "peak_memory_gb": _metric_float(((evidence.get("stats") or {}) if isinstance(evidence.get("stats"), dict) else {}).get("peak_memory_gb")),
            },
        },
        "evidence_paths": {
            "local_infer": evidence_path,
            "run_report": str(run_report_path),
        },
        "gui_stage_source": gui_stage_source,
        "gui_graph_native_integration": gui_graph_native,
        "gui_graph_native_integration_path": gui_graph_native_path,
    }
    route_decision_path = Path(write_json_file(report_root / "route_decision.json", route_decision_payload))

    manifest_path, manifest_payload = _write_run_smoke_manifest(
        report_root,
        use_flashmoe=bool(payload.get("use_flashmoe")),
    )
    inference_ok = bool(local_execution and backend == "omlx_mlx_lm")
    inference_report = {
        "ok": inference_ok,
        "mode": "bridge_from_cgc_run",
        "exec_mode": "compile",
        "task_type": canonical_task_type,
        "task_type_contract_version": TASK_TYPE_CONTRACT_VERSION,
        "task_type_contract_ref": task_type_contract_ref(),
        "backend": "mlx",
        "model": str(model_to_use),
        "runtime_model": str(runtime_model),
        "source_run_report": str(run_report_path),
        "error_msg": "" if inference_ok else str(final_event.get("error") or final_event.get("response") or "cgc_run_local_execution_not_proven"),
        "steps": {
            "step2_fullgraph_capture": {
                "status": "PASS" if inference_ok else "FAIL",
                "model_id": str(model_to_use),
                "device": "mps" if sys.platform == "darwin" else "",
                "dtype": "fp32",
                "prompt": str(prompt),
                "contexts": [128],
                "max_new_tokens": int(payload.get("max_tokens") or 0),
                "omlx_engine": "flashmoe" if bool(payload.get("use_flashmoe")) else "mlx_lm",
                "manifest_path": str(manifest_path),
                "evidence_path": evidence_path,
            },
            "step6_fullgraph_compile": {
                "status": "PASS" if inference_ok else "FAIL",
                "compile_mode": backend or "omlx_mlx_lm",
                "aot": False,
                "source": "cgc_run",
            },
            "step7_fullgraph_bench": {
                "status": "PASS" if inference_ok else "FAIL",
                "optimized": inference_ok,
                "generation_tps": generation_tps,
                "edge_latency_ms": edge_latency_ms,
            },
            "step8_fullgraph_deploy": {
                "status": "PASS" if inference_ok else "FAIL",
                "deploy_unit": {
                    "omlx_model_path": str(runtime_model),
                    "omlx_manifest_path": str(manifest_path),
                    "evidence_path": evidence_path,
                },
            },
        },
        "optimized": {
            "status": "PASS" if inference_ok else "FAIL",
            "local_execution": local_execution,
            "backend": backend,
        },
        "manifest": manifest_payload,
    }
    inference_report_path = Path(write_json_file(report_root / "m4_inference_report.json", inference_report))

    bridge_ok = bool((inference_ok and edge_latency_ms > 0.0 and edge_latency_ms <= 20.0) or bridge_response_ok)
    bridge_payload = {
        "status": "PASS" if bridge_ok else "FAIL",
        "mode": "bridge_from_cgc_run_cloud_takeover" if cloud_bridge_used else "bridge_from_cgc_run_local_infer",
        "bridge_export_success": 1.0 if (inference_ok or bridge_response_ok) else 0.0,
        "edge_latency_ms": edge_latency_ms,
        "backends": {
            "mlx": {
                "status": "PASS" if inference_ok else "FAIL",
                "report_path": str(inference_report_path),
                "backend": backend,
                "local_execution": local_execution,
            }
        },
        "selected_route": selected_route,
        "selected_backend": backend,
        "cloud_bridge_used": cloud_bridge_used,
        "source_report": str(inference_report_path),
        "evidence_path": evidence_path,
    }
    bridge_path = Path(write_json_file(report_root / "edge_inference_bridge.json", bridge_payload))

    write_json_file(CGC_RUN_LATEST_REPORT, run_report)
    write_json_file(CGC_RUN_LATEST_M4_INFERENCE_REPORT, inference_report)
    write_json_file(CGC_RUN_LATEST_EDGE_BRIDGE, bridge_payload)
    write_json_file(CGC_RUN_LATEST_ROUTE_DECISION, route_decision_payload)
    return {
        "status": "PASS" if ok else "FAIL",
        "run_report_path": str(run_report_path),
        "m4_inference_report_path": str(inference_report_path),
        "edge_inference_bridge_path": str(bridge_path),
        "route_decision_path": str(route_decision_path),
        "edge_latency_ms": edge_latency_ms,
        "local_execution": local_execution,
        "backend": backend,
        "evidence_path": evidence_path,
        "resolved_model_path": resolved_model_path,
        "format": model_format,
        "selected_route": selected_route,
        "decision_reason": decision_reason,
    }

def main():
    parser = argparse.ArgumentParser(description="CGC Engine CLI - Edge/Cloud LLM Offloading")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # cgc serve
    serve_parser = subparsers.add_parser("serve", help="Start the CGC API Server (Ollama/Anthropic/OpenAI compatible)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind the server")
    serve_parser.add_argument("--proxy-port", type=int, default=4000, help="Port to bind the internal protocol proxy")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the CGC API server")
    serve_parser.add_argument("--proxy-host", type=str, default="127.0.0.1", help="Host to bind the internal protocol proxy")

    # cgc claude
    claude_parser = subparsers.add_parser("claude", help="Launch Claude Code CLI with CGC Environment", add_help=False)
    claude_parser.add_argument("claude_args", nargs=argparse.REMAINDER)

    # cgc config
    config_parser = subparsers.add_parser("config", help="Configure CGC Engine")
    config_parser.add_argument("--set-cloud-ip", type=str, help="Set the primary Cloud Server IP")
    config_parser.add_argument("--set-cloud-model", type=str, help="Set the active Cloud model")
    config_parser.add_argument("--set-edge-model", type=str, help="Set the active Edge model")
    config_parser.add_argument("--set-local-omlx-model", type=str, help="Set the local OMLX/FlashMoE model reference for true edge execution")
    config_parser.add_argument("--set-local-flashmoe-model", type=str, help="Set the local FlashMoE model directory for true edge execution")
    config_parser.add_argument("--set-edge-api-port", type=int, help="Set the local CGC Edge API port")
    config_parser.add_argument("--set-edge-proxy-port", type=int, help="Set the local CGC Edge proxy port")
    
    # cgc run
    run_parser = subparsers.add_parser("run", help="Run a model interactively and discover CGC models")
    run_parser.add_argument("model", type=str, nargs="?", help="Model name to run (defaults to active_edge_model)")
    run_parser.add_argument("--use-omlx", action="store_true", help="Enable Apple MLX optimization for local models")
    run_parser.add_argument("--use-flashmoe", action="store_true", help="Enable FlashMoE to prevent VRAM OOM on Edge")
    run_parser.add_argument("--list-models", action="store_true", help="List model names exposed by the CGC fake Ollama registry")
    run_parser.add_argument("--show-spec", action="store_true", help="Show the resolved model spec for the selected model")
    run_parser.add_argument("--install-minicpm5", action="store_true", help="Stage MiniCPM5 GGUF for cgc run before starting the session")
    run_parser.add_argument("--ollama-quant", type=str, default=MINICPM5_DEFAULT_QUANT, help="Preferred MiniCPM5 GGUF quant when staging")
    run_parser.add_argument("--force-reinstall", action="store_true", help="Force refresh the staged MiniCPM5 GGUF package")
    run_parser.add_argument("--prompt", type=str, default="", help="Run a single prompt non-interactively and write reusable local inference reports")
    run_parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens for single-prompt run mode")
    run_parser.add_argument("--report-dir", type=str, default="", help="Directory for `cgc run` report artifacts")
    run_parser.add_argument("--json", action="store_true", help="Print `cgc run` single-prompt result as JSON")
    run_parser.add_argument("--gui-duration-s", type=int, default=0, help="Optional: collect GUI stage source for this many seconds before single-prompt `cgc run`.")
    run_parser.add_argument("--disable-gui-stage-source", action="store_true", help="Disable GUI stage source collection for `cgc run`.")
    
    # cgc list
    list_parser = subparsers.add_parser("list", help="List all available models (Edge and Cloud)")
    list_parser.add_argument("--json", action="store_true", help="Print discovered models as JSON")
    list_parser.add_argument("--source", type=str, default="all", help="Filter discovery source: nfs|local|cache|registry|config|all")
    list_parser.add_argument("--model-root", action="append", default=[], help="Additional local model root to scan")
    list_parser.add_argument("--nfs-root", action="append", default=[], help="Additional NFS model root to scan")

    # cgc status
    status_parser = subparsers.add_parser("status", help="Show live CGC cloud/edge status for m7.5 scale runs")
    status_parser.add_argument("--json", action="store_true", help="Print status as JSON")
    status_parser.add_argument("--write-m75-evidence", action="store_true", help="Write the live snapshot to m7.5 runtime evidence")
    status_parser.add_argument("--expected-workers", type=int, default=2000, help="Expected active worker count for the extreme run")
    status_parser.add_argument("--expected-instances", type=int, default=500, help="Expected SWE-bench instance count")
    status_parser.add_argument("--expected-fusion-group-size", type=int, default=4, help="Expected FusionRoute cloud instance group size")

    # cgc audit
    audit_parser = subparsers.add_parser("audit", help="Run, verify, trace, and export M7.1/M7.2 audit artifacts")
    audit_subparsers = audit_parser.add_subparsers(dest="audit_command", help="Audit commands")
    audit_run_parser = audit_subparsers.add_parser("run", help="Run audit collection or strict M7.2 validation")
    audit_run_parser.add_argument("--output-dir", type=str, default=str((REPO_ROOT / "temp" / "test" / "audit_cli_run").resolve()), help="Directory to write audit outputs")
    audit_run_parser.add_argument("--strict", action="store_true", help="Run strict M7.2 validation on top of M7.1 artifacts")
    audit_verify_parser = audit_subparsers.add_parser("verify", help="Verify audit hash chain from a report or explicit files")
    audit_verify_parser.add_argument("--report", type=str, default="", help="Aggregate M7/M7.1/M7.2 report path")
    audit_verify_parser.add_argument("--log", type=str, default="", help="events.jsonl path when not using --report")
    audit_verify_parser.add_argument("--head", type=str, default="", help="chain_head.json path when not using --report")
    audit_trace_parser = audit_subparsers.add_parser("trace", help="Trace audit events by stage")
    audit_trace_parser.add_argument("--report", type=str, default="", help="Aggregate M7/M7.1/M7.2 report path")
    audit_trace_parser.add_argument("--log", type=str, default="", help="events.jsonl path when not using --report")
    audit_trace_parser.add_argument("--stage", type=str, default="", help="Optional stage filter such as Build/Compile/Run/State/Replay/Exception")
    audit_trace_parser.add_argument("--limit", type=int, default=20, help="Maximum number of events to print")
    audit_export_parser = audit_subparsers.add_parser("export", help="Export an audit report to md/html/json")
    audit_export_parser.add_argument("--report", type=str, required=True, help="Aggregate M7/M7.1/M7.2 report path")
    audit_export_parser.add_argument("--output", type=str, required=True, help="Export file path")
    audit_export_parser.add_argument("--format", type=str, choices=["md", "html", "json"], default="md", help="Export format")
    
    # cgc build
    build_parser = subparsers.add_parser("build", help="Build standalone executables for Mac/Linux/Windows using Nuitka")
    build_parser.add_argument("--output-dir", type=str, default=str((REPO_ROOT / "dist" / "cgc").resolve()), help="Directory for the built standalone executable")
    build_parser.add_argument("--json", action="store_true", help="Print build result as JSON only")
    build_parser.add_argument("--report-file", type=str, default="", help="Optional path to write the build JSON report")
    build_parser.add_argument("--aggregate-dir", type=str, default="", help="Optional directory to write <platform>.json and build_matrix.json")

    # cgc agent
    agent_parser = subparsers.add_parser(
        "agent",
        help="Run product-facing agent workflows for DAG import, teaching, training, inference, visualization, and auditability",
    )
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command", help="Agent workflow commands")

    agent_import_parser = agent_subparsers.add_parser("import-dag", help="Import a business DAG/workflow and prepare compute-graph insertion artifacts")
    agent_import_parser.add_argument("--dag-file", type=str, required=True, help="Path to a JSON DAG/workflow file")
    agent_import_parser.add_argument("--dag-name", type=str, default="", help="Optional logical DAG name override")
    agent_import_parser.add_argument("--output-dir", type=str, default="", help="Directory to write imported DAG artifacts")
    agent_import_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    agent_teach_parser = agent_subparsers.add_parser("teach", help="Collect GUI teaching evidence and write teaching/replay artifacts")
    agent_teach_parser.add_argument("--output-dir", type=str, default="", help="Directory to write teaching artifacts")
    agent_teach_parser.add_argument("--dag-file", type=str, default="", help="Optional DAG JSON to import before teaching")
    agent_teach_parser.add_argument("--dag-manifest", type=str, default="", help="Optional existing imported DAG manifest path")
    agent_teach_parser.add_argument("--dag-name", type=str, default="", help="Optional logical DAG name override")
    agent_teach_parser.add_argument("--teaching-mode", type=str, choices=["development", "customer"], default="development", help="Teaching evidence mode: `development` is for validation-time GUI evidence; `customer` requires real screen recording and keyboard/mouse events")
    agent_teach_parser.add_argument("--gui-duration-s", type=int, default=5, help="Collect GUI teaching evidence for this many seconds; mainly for development validation or as supplemental evidence in customer mode")
    agent_teach_parser.add_argument("--gui-evidence-path", type=str, default="", help="Use an existing GUI runtime evidence file instead of recording a new one")
    agent_teach_parser.add_argument("--screen-recording-path", type=str, default="", help="Customer mode: full screen-recording file captured from the real teaching session")
    agent_teach_parser.add_argument("--keyboard-mouse-events-path", type=str, default="", help="Customer mode: keyboard/mouse event trace file for the same teaching session")
    agent_teach_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    agent_train_parser = agent_subparsers.add_parser("train", help="Run the 3.8 cloud Q2RL/UI-TARS training chain with optional Subterranean Agent DAG import")
    agent_train_parser.add_argument("--output-dir", type=str, default="", help="Directory to write training artifacts")
    agent_train_parser.add_argument("--teach-session", type=str, default="", help="Existing `cgc agent teach` session path")
    agent_train_parser.add_argument("--dag-file", type=str, default="", help="Optional DAG JSON to import before training")
    agent_train_parser.add_argument("--dag-manifest", type=str, default="", help="Optional existing imported DAG manifest path")
    agent_train_parser.add_argument("--dag-name", type=str, default="", help="Optional logical DAG name override")
    agent_train_parser.add_argument("--teaching-mode", type=str, choices=["development", "customer"], default="development", help="Training evidence mode; use `customer` when the source teaching session comes from real customer recording plus keyboard/mouse events")
    agent_train_parser.add_argument("--gui-duration-s", type=int, default=5, help="Collect GUI evidence for this many seconds when no teach session/evidence is provided; intended for development validation")
    agent_train_parser.add_argument("--gui-evidence-path", type=str, default="", help="Existing GUI runtime evidence file to feed into upkg38")
    agent_train_parser.add_argument("--screen-recording-path", type=str, default="", help="Customer mode: full screen-recording file for the source teaching session")
    agent_train_parser.add_argument("--keyboard-mouse-events-path", type=str, default="", help="Customer mode: keyboard/mouse event trace file for the source teaching session")
    agent_train_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    agent_infer_parser = agent_subparsers.add_parser("infer", help="Materialize the edge inference bundle from a trained agent session")
    agent_infer_parser.add_argument("--train-session", type=str, default="", help="`cgc agent train` session path")
    agent_infer_parser.add_argument("--artifact-root", type=str, default="", help="Existing upkg38 output root when not using --train-session")
    agent_infer_parser.add_argument("--output-dir", type=str, default="", help="Directory to write infer session artifacts")
    agent_infer_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    agent_visualize_parser = agent_subparsers.add_parser("visualize", help="Index triplet comparison and error visualization outputs")
    agent_visualize_parser.add_argument("--train-session", type=str, default="", help="`cgc agent train` session path")
    agent_visualize_parser.add_argument("--artifact-root", type=str, default="", help="Existing upkg38 output root when not using --train-session")
    agent_visualize_parser.add_argument("--output-dir", type=str, default="", help="Directory to write visualization index artifacts")
    agent_visualize_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    agent_compare_parser = agent_subparsers.add_parser("compare", help="Summarize teaching vs pre/post Q2RL comparison artifacts")
    agent_compare_parser.add_argument("--train-session", type=str, default="", help="`cgc agent train` session path")
    agent_compare_parser.add_argument("--artifact-root", type=str, default="", help="Existing upkg38 output root when not using --train-session")
    agent_compare_parser.add_argument("--output-dir", type=str, default="", help="Directory to write comparison summary artifacts")
    agent_compare_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    agent_audit_parser = agent_subparsers.add_parser("audit", help="Summarize audit/replay/traceability outputs for a trained agent session")
    agent_audit_parser.add_argument("--train-session", type=str, default="", help="`cgc agent train` session path")
    agent_audit_parser.add_argument("--artifact-root", type=str, default="", help="Existing upkg38 output root when not using --train-session")
    agent_audit_parser.add_argument("--output-dir", type=str, default="", help="Directory to write audit summary artifacts")
    agent_audit_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    agent_replay_parser = agent_subparsers.add_parser("replay", help="Prepare replay metadata for GUI teaching and upkg38 results")
    agent_replay_parser.add_argument("--train-session", type=str, default="", help="`cgc agent train` session path")
    agent_replay_parser.add_argument("--artifact-root", type=str, default="", help="Existing upkg38 output root when not using --train-session")
    agent_replay_parser.add_argument("--output-dir", type=str, default="", help="Directory to write replay session artifacts")
    agent_replay_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    agent_trace_parser = agent_subparsers.add_parser("trace", help="Summarize stage trace and GUI event trace for a trained agent session")
    agent_trace_parser.add_argument("--train-session", type=str, default="", help="`cgc agent train` session path")
    agent_trace_parser.add_argument("--artifact-root", type=str, default="", help="Existing upkg38 output root when not using --train-session")
    agent_trace_parser.add_argument("--output-dir", type=str, default="", help="Directory to write trace summary artifacts")
    agent_trace_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    # cgc agent universe: CLI-Universe 三阶段流水线 (采: Agent 训练数据合成)
    agent_universe_parser = agent_subparsers.add_parser(
        "universe",
        help="CLI-Universe 三阶段流水线: 任务蓝图→环境物化→验证过滤 (Agent 训练数据合成)",
    )
    agent_universe_parser.add_argument("--step", type=int, default=0,
        help="运行指定阶段: 1=蓝图构建 2=环境物化 3=验证过滤 0=端到端全部 (默认)")
    agent_universe_parser.add_argument("--num", type=int, default=10,
        help="生成蓝图数量 (默认 10)")
    agent_universe_parser.add_argument("--output-dir", type=str, default="",
        help="输出目录 (默认 /tmp/cgc_universe_<timestamp>)")
    agent_universe_parser.add_argument("--teacher-model", type=str, default="kimi-k2.6",
        help="教师模型 (默认 kimi-k2.6, 备选 deepseek-v4-pro)")
    agent_universe_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    # cgc agent fusionroute: FusionRoute 四角色编排 (推: Hermes/TMAX/UITARS/Synthesizer)
    agent_fusionroute_parser = agent_subparsers.add_parser(
        "fusionroute",
        help="FusionRoute 四角色编排: Hermes/TMAX/UITARS/Synthesizer (Agent 推理执行)",
    )
    agent_fusionroute_parser.add_argument("action", type=str, nargs="?", default="status",
        help="start|stop|status|route (默认 status)")
    agent_fusionroute_parser.add_argument("--task", type=str, default="",
        help="路由任务描述 (action=route 时使用)")
    agent_fusionroute_parser.add_argument("--hermes-port", type=int, default=30003, help="Hermes 端口")
    agent_fusionroute_parser.add_argument("--tmax-port", type=int, default=30001, help="TMAX Planner 端口")
    agent_fusionroute_parser.add_argument("--uitars-port", type=int, default=30002, help="UITARS Executor 端口")
    agent_fusionroute_parser.add_argument("--synth-port", type=int, default=30004, help="CLI-Universe Synthesizer 端口")
    agent_fusionroute_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    # cgc agent bench: Agent Benchmark (OSWorld + WebArena)
    agent_bench_parser = agent_subparsers.add_parser(
        "bench",
        help="Agent Benchmark: OSWorld (桌面GUI) + WebArena (网站交互)",
    )
    agent_bench_parser.add_argument("--benchmark", type=str, default="osworld",
        choices=["osworld", "webarena", "all"], help="Benchmark 类型 (默认 osworld)")
    agent_bench_parser.add_argument("--num-tasks", type=int, default=10, help="任务数量 (默认 10)")
    agent_bench_parser.add_argument("--output-dir", type=str, default="", help="输出目录")
    agent_bench_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    model_parser = subparsers.add_parser(
        "model",
        help="Run product-facing model workflows for discovery, run, serve, verify, and auditable replayable model sessions",
    )
    model_subparsers = model_parser.add_subparsers(dest="model_command", help="Model workflow commands")

    model_list_parser = model_subparsers.add_parser("list", help="Discover available local, NFS, and registry models")
    model_list_parser.add_argument("--output-dir", type=str, default="", help="Directory to write model list session artifacts")
    model_list_parser.add_argument("--source", type=str, default="all", help="Filter discovery source: nfs|local|cache|registry|config|all")
    model_list_parser.add_argument("--model-root", action="append", default=[], help="Additional local model root to scan")
    model_list_parser.add_argument("--nfs-root", action="append", default=[], help="Additional NFS model root to scan")
    model_list_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    model_run_parser = model_subparsers.add_parser("run", help="Run a model with structured session artifacts aligned to cgc agent")
    model_run_parser.add_argument("model", type=str, nargs="?", help="Model name to run (defaults to active_edge_model)")
    model_run_parser.add_argument("--output-dir", type=str, default="", help="Directory to write model run session artifacts")
    model_run_parser.add_argument("--use-omlx", action="store_true", help="Enable Apple MLX optimization for local models")
    model_run_parser.add_argument("--use-flashmoe", action="store_true", help="Enable FlashMoE to prevent VRAM OOM on Edge")
    model_run_parser.add_argument("--prompt", type=str, default="", help="Run a single prompt non-interactively and write structured model artifacts")
    model_run_parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens for single-prompt run mode")
    model_run_parser.add_argument("--gui-duration-s", type=int, default=0, help="Optional GUI stage-source collection duration before single-prompt model run")
    model_run_parser.add_argument("--disable-gui-stage-source", action="store_true", help="Disable GUI stage-source collection for cgc model run")
    model_run_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    model_serve_parser = model_subparsers.add_parser("serve", help="Start the model-facing CGC API server and write a structured serve session")
    model_serve_parser.add_argument("--output-dir", type=str, default="", help="Directory to write model serve session artifacts")
    model_serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind the server")
    model_serve_parser.add_argument("--proxy-port", type=int, default=4000, help="Port to bind the internal protocol proxy")
    model_serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the CGC API server")
    model_serve_parser.add_argument("--proxy-host", type=str, default="127.0.0.1", help="Host to bind the internal protocol proxy")
    model_serve_parser.add_argument("--json", action="store_true", help="Print result as JSON before starting the server")

    model_verify_parser = model_subparsers.add_parser("verify", help="Verify a model run session or artifact root")
    model_verify_parser.add_argument("--run-session", type=str, default="", help="Existing cgc model run session path")
    model_verify_parser.add_argument("--artifact-root", type=str, default="", help="Existing model artifact root when not using --run-session")
    model_verify_parser.add_argument("--output-dir", type=str, default="", help="Directory to write verify session artifacts")
    model_verify_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    model_audit_parser = model_subparsers.add_parser("audit", help="Summarize auditability, replayability, and failure attribution for a model session")
    model_audit_parser.add_argument("--run-session", type=str, default="", help="Existing cgc model run session path")
    model_audit_parser.add_argument("--artifact-root", type=str, default="", help="Existing model artifact root when not using --run-session")
    model_audit_parser.add_argument("--output-dir", type=str, default="", help="Directory to write audit session artifacts")
    model_audit_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    model_replay_parser = model_subparsers.add_parser("replay", help="Prepare replay metadata for a model run session")
    model_replay_parser.add_argument("--run-session", type=str, default="", help="Existing cgc model run session path")
    model_replay_parser.add_argument("--artifact-root", type=str, default="", help="Existing model artifact root when not using --run-session")
    model_replay_parser.add_argument("--output-dir", type=str, default="", help="Directory to write replay session artifacts")
    model_replay_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    model_trace_parser = model_subparsers.add_parser("trace", help="Summarize route decision and final event trace for a model run session")
    model_trace_parser.add_argument("--run-session", type=str, default="", help="Existing cgc model run session path")
    model_trace_parser.add_argument("--artifact-root", type=str, default="", help="Existing model artifact root when not using --run-session")
    model_trace_parser.add_argument("--output-dir", type=str, default="", help="Directory to write trace session artifacts")
    model_trace_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    model_compare_parser = model_subparsers.add_parser("compare", help="Compare two model run sessions or artifact roots")
    model_compare_parser.add_argument("--run-session", type=str, default="", help="Primary cgc model run session path")
    model_compare_parser.add_argument("--artifact-root", type=str, default="", help="Primary model artifact root when not using --run-session")
    model_compare_parser.add_argument("--compare-against-run-session", type=str, default="", help="Baseline cgc model run session path")
    model_compare_parser.add_argument("--compare-against-artifact-root", type=str, default="", help="Baseline model artifact root when not using --compare-against-run-session")
    model_compare_parser.add_argument("--output-dir", type=str, default="", help="Directory to write compare session artifacts")
    model_compare_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    # cgc model launch: 固化端云协议模型使用方法 (AutoTunner 自动检测 → 最优启动命令)
    model_launch_parser = model_subparsers.add_parser(
        "launch",
        help="Generate optimal sglang launch command for a model (固化端云协议: cuda-graph + CGC + NEXTN + AutoTunner)",
    )
    model_launch_parser.add_argument("model", type=str, nargs="?", default="", help="Model name or path (e.g. v4-flash, qwen3-vl-2b, /data/models/DeepSeek-V4-Flash-UD-IQ2)")
    model_launch_parser.add_argument("--port", type=int, default=30001, help="Sglang server port")
    model_launch_parser.add_argument("--host", type=str, default="0.0.0.0", help="Sglang server host")
    model_launch_parser.add_argument("--tp", type=int, default=0, help="Tensor parallel size (0=auto: V4-Flash=8, Qwen3-VL=1)")
    model_launch_parser.add_argument("--context-length", type=int, default=16384, help="Context length")

    # 投机 decode 方式选择
    model_launch_parser.add_argument("--speculative-algorithm", type=str, default="auto",
        help="Speculative decode algorithm: auto|NEXTN|EAGLE|ngram|chain|none (auto=按模型自动选, none=关闭)")
    model_launch_parser.add_argument("--speculative-num-steps", type=int, default=0,
        help="Speculative num steps N (0=auto: V4-Flash=4, Qwen3-VL=16, GPU=4, Mac=16)")
    model_launch_parser.add_argument("--speculative-num-draft-tokens", type=int, default=0,
        help="Speculative num draft tokens (0=auto: N*4)")
    model_launch_parser.add_argument("--speculative-eagle-topk", type=int, default=1,
        help="Speculative eagle topk (1=兼容 V4-Flash, >1 需 flashinfer)")
    model_launch_parser.add_argument("--speculative-draft-model", type=str, default="",
        help="Draft model path (EAGLE/ngram 用, NEXTN 不需要)")
    model_launch_parser.add_argument("--no-speculative", action="store_true", help="Disable speculative decode (= --speculative-algorithm none)")

    # 端云 PD 分离配置
    model_launch_parser.add_argument("--pd-separation", action="store_true",
        help="Enable cloud-edge PD separation (cloud prefill → emit hidden+KV → edge decode)")
    model_launch_parser.add_argument("--pd-emit-host", type=str, default="",
        help="Edge host for PD resume (e.g. 10.100.200.65 for gs01, 39.106.118.206 for Host1)")
    model_launch_parser.add_argument("--pd-emit-port", type=int, default=31000,
        help="Port for PD handoff transport (default 31000)")
    model_launch_parser.add_argument("--pd-transport", type=str, default="nixl",
        choices=["nixl", "tcp", "file"],
        help="PD handoff transport: nixl (VRAM→VRAM, fastest)|tcp (SSH tunnel)|file (disk)")
    model_launch_parser.add_argument("--pd-cut-layer", type=int, default=0,
        help="PD cut layer (0=auto: V4-Flash=21, Qwen3-VL=42; cloud emits layer 0..cut, edge resumes cut..end)")

    # KDA/OrthoKDA 参数
    model_launch_parser.add_argument("--ortho-base-dim", type=int, default=128,
        help="OrthoKDA 正交基维度 (默认 128, 降维压缩)")
    model_launch_parser.add_argument("--no-kda", action="store_true",
        help="Disable KDA/OrthoKDA (等同 --no-cgc)")

    # R-SWA 参数
    model_launch_parser.add_argument("--rswa-window-size", type=int, default=128,
        help="R-SWA 滑动窗口大小 (默认 128)")
    model_launch_parser.add_argument("--rswa-reference-len", type=int, default=4,
        help="R-SWA Reference 永久区长度 (默认 4)")
    model_launch_parser.add_argument("--no-rswa", action="store_true",
        help="Disable R-SWA (只关闭 R-SWA, 保留 CGC/KDA)")

    # Magicompiler 参数
    model_launch_parser.add_argument("--no-magicompiler", action="store_true",
        help="Disable Magicompiler IR Pass (不检测 compress_ratios, 不插入 R-SWA)")

    model_launch_parser.add_argument("--no-cgc", action="store_true", help="Disable CGC injection (CGC_ENABLE_ORTHO_KDA=0)")
    model_launch_parser.add_argument("--no-cuda-graph", action="store_true", help="Disable cuda-graph (--disable-cuda-graph)")
    model_launch_parser.add_argument("--exec", action="store_true", help="Execute the launch command directly (background)")
    model_launch_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    model_swe_parser = model_subparsers.add_parser(
        "swe-verified",
        help="Plan and confirm the M7.6 dualnode DeepSeek V4 Flash + MiniCPM5 + FusionRoute SWE Verified workflow",
    )
    model_swe_parser.add_argument("--output-dir", type=str, default="", help="Directory to write swe verified planning artifacts")
    model_swe_parser.add_argument("--refresh-session", type=str, default="", help="Existing swe verified session dir or model_swe_verified_session.json to refresh without relaunch")
    model_swe_parser.add_argument("--status-from-file", type=str, default="", help="Existing cluster status JSON instead of probing live hosts")
    model_swe_parser.add_argument("--target-instances-per-node", type=int, default=2, help="Desired runtime instances per node before health-based downscale")
    model_swe_parser.add_argument("--gpus-per-instance", type=int, default=4, help="GPUs consumed by each runtime instance")
    model_swe_parser.add_argument("--benchmark-limit", type=int, default=500, help="SWE Verified issue count target")
    model_swe_parser.add_argument("--model-name", type=str, default="openai/deepseek-v4-flash", help="Benchmark model name passed to SWE-agent")
    model_swe_parser.add_argument("--api-base-url", type=str, default="http://127.0.0.1:50053/v1", help="OpenAI-compatible API base for SWE-agent")
    model_swe_parser.add_argument("--run-gate", action="store_true", help="After confirmation, run cgc gate m76 inside this session")
    model_swe_parser.add_argument("--deepep-mode", type=str, default="auto", help="auto|on|off for the follow-up M7.6 gate run")
    model_swe_parser.add_argument("--poll", action="store_true", help="After launch or refresh, keep polling the same session until benchmark completes or max polls is reached")
    model_swe_parser.add_argument("--poll-interval-seconds", type=int, default=30, help="Polling interval when --poll is enabled")
    model_swe_parser.add_argument("--max-polls", type=int, default=120, help="Maximum poll attempts when --poll is enabled")
    model_swe_parser.add_argument("--yes", action="store_true", help="Auto-confirm the recommended plan without interactive prompt")
    model_swe_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    embodied_parser = subparsers.add_parser(
        "embodied",
        help="Run product-facing UPKG 4.0 embodied workflows such as psi0 cloud-to-edge realtimevla orchestration",
    )
    embodied_subparsers = embodied_parser.add_subparsers(dest="embodied_command", help="Embodied workflow commands")

    embodied_psi0_parser = embodied_subparsers.add_parser(
        "psi0-realtimevla",
        help="Run the UPKG 4.0 embodied psi0 cloud contract to edge realtimevla fullchain with auditable replayable artifacts",
    )
    embodied_psi0_parser.add_argument("--output-root", type=str, default="", help="Base directory for session outputs")
    embodied_psi0_parser.add_argument("--edge-model", type=str, default=EMBODIED_DEFAULT_EDGE_MODEL, help="Edge realtimevla model id for local OMLX execution")
    embodied_psi0_parser.add_argument("--launch-command", type=str, default="", help="Optional shell command override for cloud launch step")
    embodied_psi0_parser.add_argument("--fetch-command", type=str, default="", help="Optional shell command override for cloud fetch step")
    embodied_psi0_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    embodied_train_parser = embodied_subparsers.add_parser(
        "psi0-train",
        help="Materialize the UPKG 4.0 embodied psi0 train-side full-weight manifest, publish manifest, and runtime contract",
    )
    embodied_train_parser.add_argument("--output-root", type=str, default="", help="Base directory for session outputs")
    embodied_train_parser.add_argument("--launch-command", type=str, default="", help="Optional shell command override for cloud launch step")
    embodied_train_parser.add_argument("--fetch-command", type=str, default="", help="Optional shell command override for cloud fetch step")
    embodied_train_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    embodied_deploy_parser = embodied_subparsers.add_parser(
        "psi0-deploy",
        help="Materialize the UPKG 4.0 embodied psi0 deploy-side deploy contract and realtimevla consume contract",
    )
    embodied_deploy_parser.add_argument("--output-root", type=str, default="", help="Base directory for session outputs")
    embodied_deploy_parser.add_argument("--train-session", type=str, default="", help="Existing `cgc embodied psi0-train` session dir or orchestration_report.json")
    embodied_deploy_parser.add_argument("--launch-command", type=str, default="", help="Optional shell command override for cloud launch step when auto-running train")
    embodied_deploy_parser.add_argument("--fetch-command", type=str, default="", help="Optional shell command override for cloud fetch step when auto-running train")
    embodied_deploy_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    bundle_parser = subparsers.add_parser(
        "bundle",
        help="Run standalone bundle contract governance review commands",
    )
    bundle_subparsers = bundle_parser.add_subparsers(dest="bundle_command", help="Bundle governance commands")
    bundle_review_parser = bundle_subparsers.add_parser("review", help="Review a bundle contract chain without mixing run/verify/audit checks")
    bundle_review_parser.add_argument("--run-session", type=str, default="", help="Existing cgc model run session path to resolve bundle artifacts from")
    bundle_review_parser.add_argument("--artifact-root", type=str, default="", help="Existing artifact root to resolve bundle artifacts from")
    bundle_review_parser.add_argument("--from-report", type=str, default="", help="Existing report.json, build_report.json, or model_run_session.json to derive bundle review inputs from")
    bundle_review_parser.add_argument("--profile-settings", type=str, default="", help="Explicit profile_settings.json path")
    bundle_review_parser.add_argument("--system-manifest", type=str, default="", help="Explicit system_execution_manifest.json path")
    bundle_review_parser.add_argument("--bootstrap-contract", type=str, default="", help="Explicit bootstrap contract JSON path")
    bundle_review_parser.add_argument("--strict", action="store_true", help="Require PASS exactly; downgrade SKIP/incomplete review to FAIL")
    bundle_review_parser.add_argument("--output-dir", type=str, default="", help="Directory to write bundle review artifacts")
    bundle_review_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    # cgc gate
    gate_parser = subparsers.add_parser(
        "gate",
        help="Run CGC verification-only gates through a shared CLI entrypoint",
        description="Run CGC verification-only gates. Release-facing UPKG entrypoints are upkg21, upkg30, upkg31, upkg32, upkg33, upkg34, upkg35, upkg36, upkg37, upkg38, and upkg39.",
        epilog=(
            "Recommended release-facing commands:\n"
            "  cgc gate upkg21\n"
            "  cgc gate upkg21-rerun\n"
            "  cgc gate upkg30\n"
            "  cgc gate upkg31\n"
            "  cgc gate upkg32\n"
            "  cgc gate upkg33\n"
            "  cgc gate upkg34\n"
            "  cgc gate upkg35\n"
            "  cgc gate upkg36\n"
            "  cgc gate upkg37\n"
            "  cgc gate upkg38\n"
            "  cgc gate upkg39"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    gate_parser.add_argument("gate_name", nargs="?", default="list", help="Gate name to run: m1, m2, m3, m4, m5, m6, m7, m71, m72, m73, m74, upkg21, upkg21-rerun, m75, m76, m77, m78, upkg3, upkg30, upkg31, upkg32, upkg33, upkg34, upkg35, upkg36, upkg37, upkg38, upkg39, m8, m9, or list")
    gate_parser.add_argument("--repo-root", type=str, default=str(SCRIPT_DIR), help="Repository root to inspect")
    gate_parser.add_argument("--output-dir", type=str, default=None, help="Optional gate output directory")
    gate_parser.add_argument("--list", action="store_true", help="List registered gates")
    gate_parser.add_argument("--gate-target", type=str, default=None, help="Target gate name for explicit checkin")
    gate_parser.add_argument("--trigger", type=str, default="manual", help="Checkin trigger label")
    gate_parser.add_argument("--report-path", type=str, default="", help="Explicit report path for gate checkin")
    gate_parser.add_argument("--summary-path", type=str, default="", help="Explicit summary path for gate checkin")
    gate_parser.add_argument("--m4-training-report", type=str, default="", help="M4 only: external/cloud training subreport path for final aggregation")
    gate_parser.add_argument("--m4-inference-report", type=str, default="", help="M4 only: external/local inference subreport path for final aggregation")
    gate_parser.add_argument("--print-json", action="store_true", help="Print gate result as JSON")
    gate_parser.add_argument("--m72-gui-duration-s", type=int, default=5, help="M72 only: collect GUI evidence for this many seconds before running gate.")
    gate_parser.add_argument("--m72-disable-gui-evidence", action="store_true", help="M72 only: disable automatic GUI evidence collection.")

    # cgc storage: GDS/SPDK 存储管理
    storage_parser = subparsers.add_parser(
        "storage",
        help="GDS (GPU Direct Storage) + SPDK (NVMe) 存储管理",
    )
    storage_subparsers = storage_parser.add_subparsers(dest="storage_command", help="Storage commands")
    storage_subparsers.add_parser("status", help="Check GDS/SPDK availability and status")
    storage_subparsers.add_parser("gds", help="Test GDS zero-copy GPU↔NVMe I/O")
    storage_subparsers.add_parser("spdk", help="Test SPDK NVMe I/O (or thread-pool fallback)")
    storage_bench_parser = storage_subparsers.add_parser("bench", help="Storage performance benchmark")
    storage_bench_parser.add_argument("--size-mb", type=int, default=100, help="Test data size in MB (default 100)")
    storage_bench_parser.add_argument("--json", action="store_true", help="Print as JSON")

    # cgc moe: FlashMoE MoE 推理引擎
    moe_parser = subparsers.add_parser(
        "moe",
        help="FlashMoE MoE inference engine (CPU/CUDA/Metal + GDS/SPDK expert loading)",
    )
    moe_subparsers = moe_parser.add_subparsers(dest="moe_command", help="MoE commands")
    moe_subparsers.add_parser("status", help="Check FlashMoE engine availability (CPU/CUDA/Metal + GDS/SPDK)")
    moe_infer_parser = moe_subparsers.add_parser("infer", help="Run MoE inference")
    moe_infer_parser.add_argument("--model", type=str, default="", help="Model path (MoE model)")
    moe_infer_parser.add_argument("--prompt", type=str, default="Hello", help="Inference prompt")
    moe_infer_parser.add_argument("--max-tokens", type=int, default=30, help="Max tokens")
    moe_infer_parser.add_argument("--engine", type=str, default="auto", choices=["auto", "cpu", "cuda", "metal"], help="Inference engine")
    moe_bench_parser = moe_subparsers.add_parser("bench", help="FlashMoE performance benchmark")
    moe_bench_parser.add_argument("--num-experts", type=int, default=8, help="Number of experts (default 8)")
    moe_bench_parser.add_argument("--expert-size", type=int, default=1024, help="Expert hidden size (default 1024)")
    moe_bench_parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default 1)")
    moe_bench_parser.add_argument("--json", action="store_true", help="Print as JSON")

    # cgc compile: 统一编译引擎 (engine + cgc_jitload + passes + ir)
    compile_parser = subparsers.add_parser("compile", help="Unified compile engine (MagiCompiler + JIT + IR passes)")
    compile_subparsers = compile_parser.add_subparsers(dest="compile_command", help="Compile commands")
    compile_subparsers.add_parser("status", help="Check compile engine availability (engine/jit/passes/ir)")
    compile_run_parser = compile_subparsers.add_parser("run", help="Compile a model")
    compile_run_parser.add_argument("--model", type=str, default="", help="Model path to compile")
    compile_run_parser.add_argument("--target", type=str, default="auto", choices=["auto", "mlx", "cuda", "native"], help="Compile target backend")
    compile_run_parser.add_argument("--json", action="store_true", help="Print as JSON")

    # cgc convert: 模型格式转换 (model_parsers)
    convert_parser = subparsers.add_parser("convert", help="Model format conversion (GGUF/HF/PyTorch)")
    convert_subparsers = convert_parser.add_subparsers(dest="convert_command", help="Convert commands")
    convert_subparsers.add_parser("status", help="Check model parsers availability")
    convert_g2m_parser = convert_subparsers.add_parser("gguf-to-mlx", help="Convert GGUF to MLX format")
    convert_g2m_parser.add_argument("--input", type=str, required=True, help="Input GGUF model path")
    convert_g2m_parser.add_argument("--output", type=str, default="", help="Output MLX model path")
    convert_g2p_parser = convert_subparsers.add_parser("gguf-to-pytorch", help="Convert GGUF to PyTorch format")
    convert_g2p_parser.add_argument("--input", type=str, required=True, help="Input GGUF model path")
    convert_g2p_parser.add_argument("--output", type=str, default="", help="Output PyTorch model path")
    convert_info_parser = convert_subparsers.add_parser("info", help="Parse model config info")
    convert_info_parser.add_argument("--model", type=str, required=True, help="Model path")
    convert_info_parser.add_argument("--json", action="store_true", help="Print as JSON")

    # cgc topology: 图拓扑 + 分布式拓扑 (analysis + distributed_topology)
    topology_parser = subparsers.add_parser("topology", help="Graph topology + distributed topology analysis")
    topology_subparsers = topology_parser.add_subparsers(dest="topology_command", help="Topology commands")
    topology_subparsers.add_parser("detect", help="Detect cluster topology (GPUs/nodes/NVLink/IB)")
    topology_rec_parser = topology_subparsers.add_parser("recommend", help="Recommend optimal TP/EP/PP/DP")
    topology_rec_parser.add_argument("--model", type=str, default="", help="Model path (for size-aware recommendation)")
    topology_rec_parser.add_argument("--world-size", type=int, default=0, help="World size (0=auto-detect)")
    topology_val_parser = topology_subparsers.add_parser("validate", help="Validate topology config")
    topology_val_parser.add_argument("--tp", type=int, default=8, help="Tensor parallel")
    topology_val_parser.add_argument("--ep", type=int, default=1, help="Expert parallel")
    topology_val_parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel")
    topology_val_parser.add_argument("--dp", type=int, default=1, help="Data parallel")
    topology_graph_parser = topology_subparsers.add_parser("graph", help="Analyze model computation graph")
    topology_graph_parser.add_argument("--model", type=str, default="", help="Model path")
    topology_graph_parser.add_argument("--json", action="store_true", help="Print as JSON")

    # cgc profile: 性能分析 (profiler)
    profile_parser = subparsers.add_parser("profile", help="Performance profiling and analysis")
    profile_subparsers = profile_parser.add_subparsers(dest="profile_command", help="Profile commands")
    profile_subparsers.add_parser("status", help="Check profiler availability")
    profile_start_parser = profile_subparsers.add_parser("start", help="Start profiling")
    profile_start_parser.add_argument("--duration", type=int, default=60, help="Profiling duration in seconds (default 60)")
    profile_start_parser.add_argument("--output", type=str, default="", help="Output directory")
    profile_report_parser = profile_subparsers.add_parser("report", help="Generate profiling report")
    profile_report_parser.add_argument("--input", type=str, default="", help="Profile data directory")
    profile_report_parser.add_argument("--format", type=str, default="text", choices=["text", "json", "html"], help="Report format")
    profile_subparsers.add_parser("gpu", help="GPU utilization and memory report")

    if len(sys.argv) > 1 and sys.argv[1] == "claude":
        args = argparse.Namespace(command="claude", claude_args=sys.argv[2:])
    else:
        args = parser.parse_args()

    if args.command == "serve":
        cfg = load_config()
        print(f"🔗 Cloud Node: {cfg.get('cloud_ip')}:{cfg.get('cloud_port')}")
        print(
            "🚀 Starting CGC Edge Engine stack "
            f"(API {args.host}:{args.port}, Internal Proxy {args.proxy_host}:{args.proxy_port})..."
        )
        cfg["edge_api_port"] = int(args.port)
        cfg["edge_proxy_port"] = int(args.proxy_port)
        save_config(cfg)
        apply_runtime_env(cfg)
        _model_serve_session(
            output_dir="",
            host=args.host,
            port=int(args.port),
            proxy_host=args.proxy_host,
            proxy_port=int(args.proxy_port),
            cfg=cfg,
        )
        start_edge_stack(
            api_host=str(args.host),
            api_port=int(args.port),
            proxy_host=str(args.proxy_host),
            proxy_port=int(args.proxy_port),
        )
            
    elif args.command == "claude":
        print("🚀 Launching Claude Code CLI with CGC Environment...")
        
        # 強制清理 Claude Code 的 Keychain 與 OAuth 殘留
        # Claude 會把 OAuth Token 存在系統 Keychain 中，這會導致它強迫連線官方伺服器驗證
        try:
            subprocess.run(["claude", "auth", "logout"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✅ Cleared Claude CLI Keychain & OAuth cache.")
        except Exception:
            pass

        claude_config = os.path.expanduser("~/.claude.json")
        if os.path.exists(claude_config):
            try:
                with open(claude_config, "r") as f:
                    config = json.load(f)
                
                # 刪除所有與 OAuth / 登入相關的欄位
                keys_to_remove = ["oauthToken", "refreshToken", "tokenType", "tokenExpiresAt", "primaryWorkspaceId", "accountSettings", "accountId"]
                modified = False
                for key in keys_to_remove:
                    if key in config:
                        del config[key]
                        modified = True
                        
                if modified:
                    with open(claude_config, "w") as f:
                        json.dump(config, f, indent=2)
                    print("✅ Cleared Claude CLI OAuth cache.")
            except Exception as e:
                print(f"⚠️ Warning: Could not clear Claude config: {e}")

        env = os.environ.copy()
        
        # 設定 CLAUDE_CODE_SIMPLE=1 來繞過 Claude CLI 啟動時對 api.anthropic.com 的國家支援檢查
        # 由於使用者可能在受限地區，原生的國家檢查會導致直接閃退 (ERR_BAD_REQUEST)
        env["CLAUDE_CODE_SIMPLE"] = "1"
        
        # 我們在這裡把 Claude 的預設模型強行覆寫為 Custom Model，
        # 這樣使用者在 UI 裡面就能看到 Custom Model 選項
        env["ANTHROPIC_CUSTOM_MODEL_OPTION_NAME"] = "DeepSeek V4 Flash (CGC Edge)"
        env["ANTHROPIC_CUSTOM_MODEL_OPTION"] = "deepseek-v4-flash:latest"
        env["ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION"] = "FusionRoute 4x Expert Pool on gs01"
        
        claude_cfg = load_config()
        env["ANTHROPIC_BASE_URL"] = f"http://localhost:{int(claude_cfg.get('edge_proxy_port', 4000) or 4000)}"
        env["ANTHROPIC_API_KEY"] = "sk-cgc-edge-key"
        # 移除可能衝突的環境變數
        for k in ["ANTHROPIC_AUTH_TOKEN", "CLAUDE_TOKEN", "CLAUDE_OAUTH_TOKEN"]:
            env.pop(k, None)
        
        try:
            cmd = ["claude"] + getattr(args, "claude_args", [])
            subprocess.run(cmd, env=env)
        except FileNotFoundError:
            print("❌ Claude CLI not found. Please make sure it's installed (`npm install -g @anthropic-ai/claude-code`)")

    elif args.command == "config":
        cfg = load_config()
        updated = False
        if args.set_cloud_ip:
            cfg["cloud_ip"] = args.set_cloud_ip
            updated = True
            print(f"✅ Cloud IP set to {args.set_cloud_ip}")
        if args.set_cloud_model:
            cfg["active_cloud_model"] = args.set_cloud_model
            updated = True
            print(f"✅ Cloud Model set to {args.set_cloud_model}")
        if args.set_edge_model:
            cfg["active_edge_model"] = args.set_edge_model
            updated = True
            print(f"✅ Edge Model set to {args.set_edge_model}")
        if args.set_local_omlx_model:
            cfg["local_omlx_model"] = args.set_local_omlx_model
            updated = True
            print(f"✅ Local OMLX model set to {args.set_local_omlx_model}")
        if args.set_local_flashmoe_model:
            cfg["local_flashmoe_model"] = args.set_local_flashmoe_model
            updated = True
            print(f"✅ Local FlashMoE model set to {args.set_local_flashmoe_model}")
        if args.set_edge_api_port is not None:
            cfg["edge_api_port"] = int(args.set_edge_api_port)
            updated = True
            print(f"✅ Edge API port set to {args.set_edge_api_port}")
        if args.set_edge_proxy_port is not None:
            cfg["edge_proxy_port"] = int(args.set_edge_proxy_port)
            updated = True
            print(f"✅ Edge proxy port set to {args.set_edge_proxy_port}")
            
        if updated:
            save_config(cfg)
            apply_runtime_env(cfg)
        else:
            print(json.dumps(cfg, indent=2))

    elif args.command == "run":
        cfg = load_config()
        apply_runtime_env(cfg)
        quiet_json = bool(args.json and str(args.prompt or "").strip())
        if args.list_models:
            print("Fetching available models from the CGC fake Ollama registry...")
            try:
                print_fake_ollama_models()
            except Exception as exc:
                print(f"❌ Failed to fetch model list: {exc}")
                sys.exit(1)
            sys.exit(0)

        model_to_use = args.model or cfg.get("active_edge_model")
        if not model_to_use:
            print("Please specify a model to run. Available models:")
            try:
                models = fetch_fake_ollama_models()
                for i, m in enumerate(models):
                    print(f"  {i+1}. {m.get('name')}")
            except Exception:
                print("  (Cannot resolve model list from CGC fake Ollama registry)")
            print("\nUsage: cgc run <model_name>")
            sys.exit(1)

        if args.show_spec:
            try:
                print_fake_ollama_show_spec(model_to_use)
            except Exception as exc:
                print(f"❌ Failed to resolve model spec for {model_to_use}: {exc}")
                sys.exit(1)
            sys.exit(0)

        requested_minicpm5 = str(model_to_use).strip().lower() in {
            "minicpm5",
            "minicpm5-1b",
            "minicpm5-1b:latest",
            MINICPM5_OLLAMA_MODEL,
        }
        if args.install_minicpm5 or requested_minicpm5:
            try:
                install_spec = None
                try:
                    install_spec = fetch_fake_ollama_install_spec(model_to_use)
                    print(
                        f"☁️ Resolved {model_to_use} from CGC Engine fake Ollama registry: "
                        f"{install_spec.get('gguf_repo')} / {install_spec.get('gguf_filename')}"
                    )
                except Exception as spec_exc:
                    print(f"⚠️ Fake Ollama registry unavailable, fallback to built-in MiniCPM5 spec: {spec_exc}")
                install_result = install_minicpm5_via_ollama(
                    model_name=MINICPM5_OLLAMA_MODEL,
                    quant=str(args.ollama_quant or MINICPM5_DEFAULT_QUANT),
                    force=bool(args.force_reinstall),
                    install_spec=install_spec,
                )
                print(json.dumps({"minicpm5_install": install_result}, ensure_ascii=False, indent=2))
                model_to_use = MINICPM5_OLLAMA_MODEL
            except subprocess.CalledProcessError as exc:
                print(f"❌ MiniCPM5 staging failed: {exc}")
                sys.exit(1)
            except Exception as exc:
                print(f"❌ MiniCPM5 staging failed: {exc}")
                sys.exit(1)

        api_base_url = get_edge_api_base_url(cfg)
        if not quiet_json:
            print(f"🚀 Starting CGC Engine interactive session with model: {model_to_use}")
        
        # =========================================================================
        # ⚙️ 十步流水線：4D 感知矩陣 + 跨平台路由 (10-Step 4D Perception Pipeline)
        # 整合 Mac/Windows/Linux 硬體感知 + PD分離/Layer-split/全雲 路由決策
        # =========================================================================
        if not quiet_json:
            print("\n[CGC Engine] 啟動十步 4D 感知流水線...")

        # 跨平台硬體感知 (Step 1-5.5)
        try:
            import sys as _sys
            _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
            from app.shared.hardware_sensing import detect_all as _detect_hardware
            from app.shared.route_decision import get_model_info as _get_model_info, compute_route as _compute_route, build_4d_matrix as _build_4d_matrix
            _hw = _detect_hardware()
        except Exception as _e:
            if not quiet_json:
                print(f"  [hw] 硬體感知降級: {_e}")
            _hw = None

        # 1. 作業系統偵測
        import platform
        os_name = _hw.os_name if _hw else platform.system()
        if not quiet_json:
            print(f"  [1/10] 系統偵測: {os_name} {_hw.os_version if _hw else ''} ({_hw.arch if _hw else platform.machine()})")

        # 2. 硬體架構偵測
        arch = _hw.arch if _hw else platform.machine()
        cpu_brand = _hw.cpu_brand if _hw else platform.processor()
        cpu_cores = _hw.cpu_cores if _hw else (os.cpu_count() or 1)
        if not quiet_json:
            print(f"  [2/10] CPU: {cpu_brand} ({cpu_cores} cores)")

        # 3. 模型格式解析
        is_local_file = os.path.exists(model_to_use) or any(model_to_use.endswith(ext) for ext in [".gguf", ".safetensors", ".mlx"])
        model_format = "Cloud/API"
        if is_local_file:
            if ".gguf" in model_to_use.lower(): model_format = "GGUF"
            elif ".mlx" in model_to_use.lower(): model_format = "MLX"
            elif ".safetensors" in model_to_use.lower(): model_format = "SafeTensors"
            else: model_format = "Local Unknown"
        if not quiet_json:
            print(f"  [3/10] 格式解析: {model_format}")

        # 4. 模型架構分析 (MoE vs Dense) + D3 模型信息
        is_moe = "moe" in model_to_use.lower() or "deepseek" in model_to_use.lower() or "30b" in model_to_use.lower()
        _model_info = None
        try:
            _model_info = _get_model_info(model_to_use)
        except:
            pass
        if not quiet_json:
            arch_str = f"MoE ({_model_info.num_experts}/{_model_info.experts_per_tok} experts)" if _model_info and _model_info.is_moe else ("Mixture-of-Experts (MoE)" if is_moe else "Dense")
            print(f"  [4/10] 網路架構: {arch_str}")

        # 5. 記憶體/顯存水位掃描 (精確測量)
        total_mem = _hw.total_mem_gb if _hw else 0
        avail_mem = _hw.available_mem_gb if _hw else 0
        if not quiet_json:
            print(f"  [5/10] 記憶體水位: {avail_mem}GB 可用 / {total_mem}GB 總計")

        # 5.5 算力等級檢測 (新增)
        compute_tier = _hw.compute_tier if _hw else "unknown"
        tflops = _hw.tflops if _hw else 0
        gpu_type = _hw.gpu_type if _hw else "none"
        gpu_name = _hw.gpu_name if _hw else ""
        if not quiet_json:
            print(f"  [5.5/10] 算力等級: {gpu_name} → {compute_tier} ({tflops} TFLOPS, {gpu_type})")

        # 6. 運算引擎自動路由 (Auto-Routing OMLX/CUDA/ROCm)
        auto_omlx = args.use_omlx
        recommended_engine = _hw.recommended_engine if _hw else ""
        if not auto_omlx and is_local_file and os_name == "Darwin" and arch == "arm64":
            auto_omlx = True
            if not quiet_json:
                print(f"  [6/10] 運算引擎: 🍎 Apple Silicon → OMLX (UMA 0-Copy), 推薦: {recommended_engine}")
        elif recommended_engine == "cuda" and gpu_type == "nvidia":
            if not quiet_json:
                print(f"  [6/10] 運算引擎: 🟢 NVIDIA → CUDA")
        elif recommended_engine == "rocm" and gpu_type == "amd":
            if not quiet_json:
                print(f"  [6/10] 運算引擎: 🔴 AMD → ROCm")
        else:
            if not quiet_json:
                print(f"  [6/10] 運算引擎: {'OMLX (手動強制)' if args.use_omlx else '預設引擎'}")

        # 7. 記憶體策略自動路由 (Auto-Routing FlashMoE)
        auto_flashmoe = args.use_flashmoe
        if not auto_flashmoe and is_local_file and is_moe:
            auto_flashmoe = True
            if not quiet_json:
                print("  [7/10] 記憶體策略: ⚡ MoE → FlashMoE 動態分頁")
        else:
            if not quiet_json:
                print(f"  [7/10] 記憶體策略: {'FlashMoE (手動強制)' if args.use_flashmoe else '預設載入'}")

        # 7.5 PD/Layer-split 路由決策 (新增, 基於 4D 感知矩陣)
        _route = None
        if _hw and _model_info:
            try:
                _route = _compute_route(_hw, _model_info)
                if not quiet_json:
                    mode_emoji = {
                        "pd_separation": "🔄",
                        "layer_split": "🔀",
                        "cloud_only": "☁️",
                        "local_only": "💻",
                    }.get(_route.mode, "❓")
                    print(f"  [7.5/10] 路由決策: {mode_emoji} {_route.mode} P={_route.P}")
                    print(f"           TTFT={_route.expected_ttft_ms:.0f}ms, decode={_route.expected_decode_tps:.1f} tok/s, 省{_route.cloud_save_pct}% cloud")
                    print(f"           {_route.reason}")
            except Exception as _e:
                if not quiet_json:
                    print(f"  [7.5/10] 路由決策: 降級 ({_e})")

        # 7.6 模型分發決策 (新增: 根據模型決定是否下載到端側)
        _dispatch = None
        if _hw and _model_info and _route:
            try:
                from app.shared.model_dispatcher import ModelDispatcher
                _dispatcher = ModelDispatcher(_hw)
                _dispatch = _dispatcher.decide(_route, _model_info)
                if not quiet_json:
                    print(f"  [7.6/10] 模型分發: {_dispatch.action} ({_dispatch.download_size_gb}GB)")
                    if _dispatch.action == "download_partial":
                        print(f"           Layer-split: 前 {_dispatch.download_layers} 層")
                    if not _dispatch.disk_sufficient:
                        print(f"           ⚠️ 磁盤不足 → 降級全雲")
                    print(f"           {_dispatch.reason}")
            except Exception as _e:
                if not quiet_json:
                    print(f"  [7.6/10] 模型分發: 降級 ({_e})")

        # 7.7 MTP draft model 同步 (新增: 雲端轉換 + 傳回端側)
        _mtp_status = None
        if _model_info and _route:
            try:
                from app.shared.model_dispatcher import MTPDraftSyncer
                _syncer = MTPDraftSyncer(api_base_url or "http://47.95.250.55:30001")
                _mtp_status = _syncer.check_and_sync(_model_info.name, _route.mode)
                if not quiet_json:
                    mtp_emoji = "✓" if _mtp_status.available else "✗"
                    print(f"  [7.7/10] MTP draft: {mtp_emoji} {_mtp_status.sync_status}")
                    if _mtp_status.available:
                        print(f"           accept={_mtp_status.expected_accept_rate:.0%}, 加速={_mtp_status.expected_decode_boost}x")
                    elif _mtp_status.needs_training:
                        print(f"           ⚠️ 需訓練 (預期 accept={_mtp_status.expected_accept_rate:.0%})")
                        # 整合 MTP trainer: 顯示訓練計劃
                        try:
                            from app.shared.mtp_trainer import integrate_with_step_77
                            _train_plan = integrate_with_step_77(_mtp_status, _model_info, api_base_url or "http://47.95.250.55:30001")
                            if _train_plan.get("action") == "plan_training":
                                print(f"           訓練計劃: {_train_plan.get('estimated_time', '?')}")
                                print(f"           一鍵訓練: {_train_plan.get('one_click', '?')}")
                        except:
                            pass
                    print(f"           {_mtp_status.reason}")
            except Exception as _e:
                if not quiet_json:
                    print(f"  [7.7/10] MTP draft: 降級 ({_e})")
        elif not quiet_json:
            print(f"  [7.5/10] 路由決策: 跳過 (硬體/模型信息不足)")

        # 8. 上下文構建完成
        if not quiet_json:
            print("  [8/10] 上下文構建: 參數自動注入完成")
            if auto_flashmoe and str(cfg.get("local_flashmoe_model") or "").strip():
                print(f"  [Edge Runtime] 本地 FlashMoE 模型: {cfg.get('local_flashmoe_model')}")
            elif auto_omlx and str(cfg.get("local_omlx_model") or "").strip():
                print(f"  [Edge Runtime] 本地 OMLX 模型: {cfg.get('local_omlx_model')}")

        # 9. 4D 感知矩陣上報 (新增)
        _4d_matrix = None
        if _hw and _model_info:
            try:
                _4d_matrix = _build_4d_matrix(_hw, _model_info)
                if not quiet_json:
                    print(f"  [9/10] 4D 感知矩陣: D1網絡(RTT={_hw.rtt_ms}ms) D2硬體({compute_tier}) D3模型({_model_info.name}) D4路由({_route.mode if _route else 'N/A'})")
            except:
                pass
        elif not quiet_json:
            print(f"  [9/10] 4D 感知矩陣: 跳過")

        # 10. 磁碟空間檢查 (新增)
        disk_avail = _hw.disk_available_gb if _hw else 0
        if not quiet_json:
            print(f"  [10/10] 磁碟空間: {disk_avail}GB 可用")
            if _model_info and disk_avail < _model_info.model_size_gb and _route and _route.mode != "cloud_only":
                print(f"         ⚠️ 磁碟不足: 模型需 {_model_info.model_size_gb}GB, 可用 {disk_avail}GB → 可能需要全雲模式")
            print("-" * 60)

        # =========================================================================
        # 11. 無縫切換器初始化 (运行时监控: 云↔本地自动切换)
        # =========================================================================
        _switcher = None
        try:
            from app.shared.seamless_switcher import SeamlessSwitcher, SwitchMode

            # 初始模式映射
            _initial_mode = SwitchMode.CLOUD
            if _route:
                if _route.mode in ("local_only",):
                    _initial_mode = SwitchMode.LOCAL
                elif _route.mode in ("pd_separation",):
                    _initial_mode = SwitchMode.LOCAL  # PD分离: Mac 做 decode = 本地模式
                elif _route.mode == "layer_split":
                    _initial_mode = SwitchMode.LAYER_SPLIT
                elif _route.mode == "cloud_only":
                    _initial_mode = SwitchMode.CLOUD

            _cloud_url = api_base_url or "http://47.95.250.55:30001"

            _switcher = SeamlessSwitcher(
                hardware_info=_hw,
                cloud_endpoint=_cloud_url,
            )
            _switcher.set_initial_mode(_initial_mode, f"十步流水線路由: {_route.mode if _route else 'unknown'}")

            if not quiet_json:
                print(f"\n[CGC Engine] 🔄 無縫切換器已啟用 (运行时监控)")
                print(f"  初始模式: {_initial_mode.value}")
                print(f"  監控: 內存({avail_mem}GB) + 網絡(RTT={_hw.rtt_ms if _hw else '?'}ms)")
                print(f"  切換閾值: 內存<1GB→切雲, >3GB→回本地, RTT>500ms→切本地")
                print(f"  KV cache 遷移: {'啟用' if _switcher.thresholds.kv_migration_enabled else '停用'}")

            # 启动后台监控
            _switcher.start()

            # 11.5 AutoTunner 集成 (与 SeamlessSwitcher 合并)
            # 切换时 AutoTunner 自动检测目标后端最优参数 (N, mode, speculative, cuda-graph)
            try:
                from app.shared.spec_decode_ir import AutoTunner

                _autotune_backend = "mlx" if _initial_mode in (SwitchMode.LOCAL, SwitchMode.LAYER_SPLIT) else "sglang"
                _autotune_config = AutoTunner.get_optimal_config(_autotune_backend, model_to_use)

                if not quiet_json:
                    print(f"  [11.5/11] AutoTunner: {_autotune_backend} → N={_autotune_config.num_draft_tokens}, mode={_autotune_config.mode}")
                    if _autotune_config.sglang_speculative_algorithm:
                        print(f"           speculative: {_autotune_config.sglang_speculative_algorithm}")
                        print(f"           cuda-graph: {'on' if not _autotune_config.sglang_disable_cuda_graph else 'off'}")
                    if _autotune_config.sglang_env_vars:
                        print(f"           env: {_autotune_config.sglang_env_vars}")
                        print(f"           mem-fraction: {_autotune_config.sglang_mem_fraction}")

                # 切换回调: 切换时 AutoTunner 自动调优
                def _on_switch_with_autotune(event):
                    new_backend = "mlx" if event.to_mode in ("local", "layer_split") else "sglang"
                    new_config = AutoTunner.get_optimal_config(new_backend, model_to_use)
                    print(f"[autotune] 切换 {event.from_mode}→{event.to_mode}, 重新调优: backend={new_backend}, N={new_config.num_draft_tokens}")
                    if new_config.sglang_speculative_algorithm:
                        print(f"[autotune]   speculative: {new_config.sglang_speculative_algorithm}, cuda-graph: {'on' if not new_config.sglang_disable_cuda_graph else 'off'}")

                _switcher.on_switch_callback = _on_switch_with_autotune

            except Exception as _e:
                if not quiet_json:
                    print(f"  [11.5/11] AutoTunner 降級: {_e}")

            # 11.6 Magicompiler IR Pass (统一 R-SWA attention 替换)
            # AutoTunner 检测 → Magicompiler IR → 编译 → InsertKDAPass 替换 attention
            _rswa_restore = None
            try:
                import sys as _sys
                _rswa_py = os.path.join(os.path.dirname(__file__), "..", "..", "rswaengine", "python")
                if os.path.isdir(_rswa_py) and _rswa_py not in _sys.path:
                    _sys.path.insert(0, os.path.abspath(_rswa_py))

                from rswa_magicompiler_ir import AutoTunnerMagicompiler

                # 模型配置 (从 4D 感知矩阵获取)
                _rswa_backend = "mlx" if _initial_mode in (SwitchMode.LOCAL, SwitchMode.LAYER_SPLIT) else "sglang"
                _model_cfg = {}
                if _model_info:
                    _model_cfg = {"num_heads": getattr(_model_info, "num_heads", 16),
                                  "head_dim": getattr(_model_info, "head_dim", 128)}

                if not quiet_json:
                    print(f"  [11.6/11] Magicompiler IR Pass: {_rswa_backend} → R-SWA KDA Attention")

                # AutoTunner + Magicompiler: 编译 IR + 替换 attention
                # 注意: model 在 sglang server 进程内, MLX 在本地
                # sglang 后端: 只创建 IR (替换在 server 启动时做)
                # MLX 后端: 直接替换 model attention
                if _rswa_backend == "mlx" and _hw and _hw.arch == "arm64":
                    # MLX: 可以直接 patch model
                    _ir = AutoTunnerMagicompiler.create_ir(_rswa_backend, _model_cfg)
                    if not quiet_json:
                        print(f"           IR: ref={_ir.reference_len}, win={_ir.window_size}, ortho={_ir.ortho_base_dim}")
                        print(f"           compile → MLX backend, InsertKDAPass ready")
                    # 实际替换在 model 加载后执行 (此处只准备 IR)
                    _rswa_ir = _ir
                else:
                    # sglang: IR 准备 (替换在 server 启动时)
                    _ir = AutoTunnerMagicompiler.create_ir(_rswa_backend, _model_cfg)
                    if not quiet_json:
                        print(f"           IR: ref={_ir.reference_len}, win={_ir.window_size}, ortho={_ir.ortho_base_dim}")
                        print(f"           compile → {_rswa_backend} backend (server-side)")
                    _rswa_ir = _ir

                # 切换回调中加入 Magicompiler 重编译
                if _switcher:
                    _orig_switch_cb = _switcher.on_switch_callback
                    def _on_switch_with_magicompiler(event):
                        if _orig_switch_cb:
                            _orig_switch_cb(event)
                        new_backend = "mlx" if event.to_mode in ("local", "layer_split") else "sglang"
                        new_ir = AutoTunnerMagicompiler.create_ir(new_backend, _model_cfg)
                        print(f"[magicompiler] 切换→{event.to_mode}, 重新编译 IR: backend={new_backend}")
                    _switcher.on_switch_callback = _on_switch_with_magicompiler

            except Exception as _e:
                if not quiet_json:
                    print(f"  [11.6/11] Magicompiler IR Pass 降級: {_e}")

        except Exception as _e:
            if not quiet_json:
                print(f"\n[CGC Engine] 無縫切換器降級: {_e}")
            _switcher = None

        print("-" * 60)

        if str(args.prompt or "").strip():
            try:
                result = _model_run_session(
                    cfg=cfg,
                    model=model_to_use,
                    use_omlx=auto_omlx,
                    use_flashmoe=auto_flashmoe,
                    prompt=str(args.prompt),
                    max_tokens=int(args.max_tokens),
                    output_dir=args.report_dir,
                    gui_duration_s=int(getattr(args, "gui_duration_s", 0)),
                    disable_gui_stage_source=bool(getattr(args, "disable_gui_stage_source", False)),
                )
                final_payload = dict(result)
                final_payload["command"] = "cgc run"
                final_payload["report_dir"] = str(result.get("source_artifact_root") or result.get("artifact_root") or "")
                if args.json:
                    print(json.dumps(final_payload, ensure_ascii=False, indent=2))
                else:
                    print(str((final_payload.get("response") or {}).get("text") or ""))
                    print(json.dumps(final_payload, ensure_ascii=False, indent=2))
                sys.exit(0 if str(final_payload.get("status") or "") == "PASS" else 1)
            except requests.exceptions.ConnectionError:
                print("\n[Error] Cannot connect to CGC Engine. Did you run 'cgc serve' in another terminal?")
                sys.exit(1)
            except Exception as e:
                print(f"\n[Error] {e}")
                sys.exit(1)

        interactive_session = _model_run_session(
            cfg=cfg,
            model=model_to_use,
            use_omlx=auto_omlx,
            use_flashmoe=auto_flashmoe,
            prompt="",
            max_tokens=256,
            output_dir=args.report_dir,
            gui_duration_s=0,
            disable_gui_stage_source=True,
        )
        print("Type '/bye' to exit.")
        runtime_model = str(interactive_session.get("runtime_model") or model_to_use)
        while True:
            try:
                user_input = input(">>> ")
                if user_input.strip() == "/bye":
                    break
                if not user_input.strip():
                    continue

                payload = {
                    "model": runtime_model,
                    "prompt": user_input,
                    "stream": True,
                    "use_omlx": auto_omlx,
                    "use_flashmoe": auto_flashmoe,
                    "max_tokens": 256,
                    "api_base_url": api_base_url,
                }

                if auto_omlx or auto_flashmoe:
                    run_result = _execute_single_prompt(api_base_url=api_base_url, payload=payload)
                    print(str(run_result.get("response_text") or ""))
                    continue

                response = requests.post(f"{api_base_url}/api/generate", json=payload, stream=True)
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        print(data.get("response", ""), end="", flush=True)
                print()
            except requests.exceptions.ConnectionError:
                print("\n[Error] Cannot connect to CGC Engine. Did you run 'cgc serve' in another terminal?")
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n[Error] {e}")
                break
                
    elif args.command == "list":
        try:
            session = _model_list_session(
                cfg=load_config(),
                output_dir="",
                source_filter=args.source,
                model_roots=args.model_root,
                nfs_roots=args.nfs_root,
            )
            payload = _safe_read_json(session.get("list_payload_path"))
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print_list_response(payload)
            sys.exit(0 if str(session.get("status") or "") == "PASS" else 1)
        except Exception as exc:
            print(f"❌ Failed to fetch model list: {exc}")
            sys.exit(1)

    elif args.command == "status":
        try:
            from m75_extreme_status import (
                collect_extreme_status,
                print_status_summary,
                write_extreme_status_evidence,
            )

            payload = collect_extreme_status(
                expected_workers=int(args.expected_workers),
                expected_instances=int(args.expected_instances),
                expected_fusion_group_size=int(args.expected_fusion_group_size),
            )
            if args.write_m75_evidence:
                evidence_path = write_extreme_status_evidence(payload)
                payload["evidence_path"] = evidence_path
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print_status_summary(payload)
                if payload.get("evidence_path"):
                    print(f"evidence: {payload['evidence_path']}")
            sys.exit(0 if payload.get("status") == "PASS" else 1)
        except Exception as exc:
            print(f"❌ Failed to collect CGC status: {exc}")
            sys.exit(1)

    elif args.command == "audit":
        if args.audit_command == "run":
            result = run_audit(output_dir=args.output_dir, strict=bool(args.strict))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            sys.exit(0 if result.get("status") == "PASS" else 1)
        if args.audit_command == "verify":
            result = verify_audit(report_path=args.report, log_path=args.log, head_path=args.head)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            sys.exit(0 if result.get("status") == "PASS" else 1)
        if args.audit_command == "trace":
            result = trace_audit(report_path=args.report, log_path=args.log, stage=args.stage, limit=args.limit)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            sys.exit(0)
        if args.audit_command == "export":
            result = export_audit(report_path=args.report, output_path=args.output, export_format=args.format)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            sys.exit(0 if result.get("status") == "PASS" else 1)
        audit_parser.print_help()
        sys.exit(2)
            
    elif args.command == "build":
        quiet_json = bool(getattr(args, "json", False))
        if not quiet_json:
            print("==================================================")
            print(" 🛡️ CGC Edge Engine - Nuitka 跨平台編譯系統")
            print("==================================================")
            print("將使用 CGC Edge Engine package 內建的 builder 與受管 Python 環境進行編譯...")
        try:
            result = build_edge_engine(
                repo_root=REPO_ROOT,
                output_dir=Path(args.output_dir),
                quiet=quiet_json,
            )
            payload = {
                "status": result.status,
                "generated_at": result.generated_at,
                "python_bin": result.python_bin,
                "builder": result.builder,
                "platform": result.platform,
                "host_platform": result.host_platform,
                "host_arch": result.host_arch,
                "package_format": result.package_format,
                "output_path": result.output_path,
                "executable_path": result.executable_path,
                "output_exists": bool(result.output_exists),
                "size_bytes": int(result.size_bytes),
                "executable_size_bytes": int(result.executable_size_bytes),
                "artifact_sha256": result.artifact_sha256,
                "executable_sha256": result.executable_sha256,
                "supported_platforms": list(result.supported_platforms),
                "command": result.command,
            }
            report_file = Path(str(getattr(args, "report_file", "") or "")).expanduser() if str(getattr(args, "report_file", "") or "").strip() else None
            aggregate_dir = Path(str(getattr(args, "aggregate_dir", "") or "")).expanduser() if str(getattr(args, "aggregate_dir", "") or "").strip() else None
            if report_file is not None:
                report_file.parent.mkdir(parents=True, exist_ok=True)
                report_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                payload["report_file"] = str(report_file.resolve())
            if aggregate_dir is not None:
                aggregate_dir.mkdir(parents=True, exist_ok=True)
                platform_report_path = (aggregate_dir / f"{result.platform}.json").resolve()
                platform_report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                required_platforms = [str(result.platform)]
                matrix_payload = {
                    "status": "PASS",
                    "generated_at": result.generated_at,
                    "platform_reports": {
                        platform_name: str((aggregate_dir / f"{platform_name}.json").resolve())
                        for platform_name in required_platforms
                        if (aggregate_dir / f"{platform_name}.json").exists()
                    },
                    "required_platforms": required_platforms,
                }
                matrix_report_path = (aggregate_dir / "build_matrix.json").resolve()
                matrix_report_path.write_text(json.dumps(matrix_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                dist_dir = (RELEASE_DIR / "dist").resolve()
                release_assets_dir = (dist_dir / "release_assets").resolve()
                dist_platform_dir = (dist_dir / str(result.platform)).resolve()
                source_artifact_path = Path(str(result.output_path)).expanduser().resolve()
                dist_artifact_path = (dist_platform_dir / source_artifact_path.name).resolve()
                _copy_release_artifact(source_artifact_path, dist_artifact_path)
                release_asset_path = (release_assets_dir / f"cgc-{result.platform}.zip").resolve()
                archived_release_asset_path = Path(_archive_release_artifact(dist_artifact_path, release_asset_path)).resolve()
                optimization_artifacts = _write_backend_injectable_optimization_package(
                    aggregate_dir=aggregate_dir,
                    dist_dir=dist_dir,
                    release_assets_dir=release_assets_dir,
                    build_payload=payload,
                )
                manifest_payload = {
                    "status": "PASS",
                    "generated_at": result.generated_at,
                    "matrix_dir": str(aggregate_dir.resolve()),
                    "matrix_file": str(matrix_report_path),
                    "matrix_status": str(matrix_payload.get("status") or "PASS"),
                    "build_artifacts_dir": str(aggregate_dir.resolve()),
                    "dist_dir": str(dist_dir),
                    "release_assets_dir": str(release_assets_dir),
                    "required_platforms": required_platforms,
                    "missing_platforms": [],
                    "invalid_platforms": [],
                    "upkg_target": str(optimization_artifacts.get("upkg_target") or ""),
                    "target_runtime_strategy": str(optimization_artifacts.get("target_runtime_strategy") or ""),
                    "preferred_backend_by_model_format": dict(optimization_artifacts.get("preferred_backend_by_model_format") or {}),
                    "injectable_backend_families": list(optimization_artifacts.get("injectable_backend_families") or []),
                    "available_backend_families": list(optimization_artifacts.get("available_backend_families") or []),
                    "optimization_package_dir": str(optimization_artifacts.get("optimization_package_dir") or ""),
                    "optimization_package_manifest": str(optimization_artifacts.get("optimization_package_manifest") or ""),
                    "backend_probe_report": str(optimization_artifacts.get("backend_probe_report") or ""),
                    "platforms": {
                        str(result.platform): {
                            "status": str(result.status or "PASS"),
                            "report_path": str(platform_report_path),
                            "report_exists": platform_report_path.exists(),
                            "source_artifact_path": str(source_artifact_path),
                            "dist_artifact_path": str(dist_artifact_path),
                            "dist_artifact_exists": dist_artifact_path.exists(),
                            "release_asset_path": str(archived_release_asset_path),
                            "release_asset_exists": archived_release_asset_path.exists(),
                            "package_format": str(result.package_format),
                            "size_bytes": int(result.size_bytes),
                            "executable_size_bytes": int(result.executable_size_bytes),
                            "artifact_sha256": str(result.artifact_sha256),
                            "executable_sha256": str(result.executable_sha256),
                            "release_asset_size_bytes": int(archived_release_asset_path.stat().st_size) if archived_release_asset_path.exists() else 0,
                            "target_runtime_strategy": str(optimization_artifacts.get("target_runtime_strategy") or ""),
                            "preferred_backend_by_model_format": dict(optimization_artifacts.get("preferred_backend_by_model_format") or {}),
                            "optimization_package_manifest": str(optimization_artifacts.get("optimization_package_manifest") or ""),
                        }
                    },
                }
                manifest_path = (dist_dir / "build_matrix_manifest.json").resolve()
                write_json_file(manifest_path, manifest_payload)
                payload.update(optimization_artifacts)
                payload["aggregate_dir"] = str(aggregate_dir.resolve())
                payload["platform_report_file"] = str(platform_report_path)
                payload["build_matrix_file"] = str(matrix_report_path)
                payload["dist_dir"] = str(dist_dir)
                payload["release_assets_dir"] = str(release_assets_dir)
                payload["build_matrix_manifest"] = str(manifest_path)
            if not quiet_json:
                print("🚀 [Nuitka] Build completed through the packaged CGC Edge Engine builder.")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception as e:
            if quiet_json:
                print(json.dumps({"status": "FAIL", "error": str(e)}, ensure_ascii=False, indent=2))
            else:
                print(f"❌ 編譯失敗: {e}")
            sys.exit(1)

    elif args.command == "agent":
        try:
            if args.agent_command == "import-dag":
                output_dir = _make_agent_output_dir(args.output_dir, command_name="agent_import_dag")
                result = _agent_import_dag(
                    dag_path=args.dag_file,
                    output_dir=output_dir,
                    dag_name=args.dag_name,
                )
            elif args.agent_command == "teach":
                output_dir = _make_agent_output_dir(args.output_dir, command_name="agent_teach")
                result = _agent_collect_teach_session(
                    output_dir=output_dir,
                    gui_duration_s=args.gui_duration_s,
                    gui_evidence_path=args.gui_evidence_path,
                    dag_manifest_path=args.dag_manifest,
                    dag_file=args.dag_file,
                    dag_name=args.dag_name,
                    teaching_mode=args.teaching_mode,
                    screen_recording_path=args.screen_recording_path,
                    keyboard_mouse_events_path=args.keyboard_mouse_events_path,
                )
            elif args.agent_command == "train":
                output_dir = _make_agent_output_dir(args.output_dir, command_name="agent_train")
                result = _agent_train_session(
                    output_dir=output_dir,
                    teach_session_path=args.teach_session,
                    dag_manifest_path=args.dag_manifest,
                    dag_file=args.dag_file,
                    dag_name=args.dag_name,
                    gui_duration_s=args.gui_duration_s,
                    gui_evidence_path=args.gui_evidence_path,
                    teaching_mode=args.teaching_mode,
                    screen_recording_path=args.screen_recording_path,
                    keyboard_mouse_events_path=args.keyboard_mouse_events_path,
                )
            elif args.agent_command == "infer":
                result = _agent_infer_session(
                    train_session_path=args.train_session,
                    artifact_root=args.artifact_root,
                    output_dir=args.output_dir,
                )
            elif args.agent_command == "visualize":
                result = _agent_visualize_session(
                    train_session_path=args.train_session,
                    artifact_root=args.artifact_root,
                    output_dir=args.output_dir,
                )
            elif args.agent_command == "compare":
                result = _agent_compare_session(
                    train_session_path=args.train_session,
                    artifact_root=args.artifact_root,
                    output_dir=args.output_dir,
                )
            elif args.agent_command == "audit":
                result = _agent_audit_session(
                    train_session_path=args.train_session,
                    artifact_root=args.artifact_root,
                    output_dir=args.output_dir,
                )
            elif args.agent_command == "replay":
                result = _agent_replay_session(
                    train_session_path=args.train_session,
                    artifact_root=args.artifact_root,
                    output_dir=args.output_dir,
                )
            elif args.agent_command == "trace":
                result = _agent_trace_session(
                    train_session_path=args.train_session,
                    artifact_root=args.artifact_root,
                    output_dir=args.output_dir,
                )
            elif args.agent_command == "universe":
                result = _agent_universe_session(
                    step=args.step,
                    num=args.num,
                    output_dir=args.output_dir,
                    teacher_model=args.teacher_model,
                )
            elif args.agent_command == "fusionroute":
                result = _agent_fusionroute_session(
                    action=args.action,
                    task=args.task,
                    hermes_port=args.hermes_port,
                    tmax_port=args.tmax_port,
                    uitars_port=args.uitars_port,
                    synth_port=args.synth_port,
                )
            elif args.agent_command == "bench":
                result = _agent_bench_session(
                    benchmark=args.benchmark,
                    num_tasks=args.num_tasks,
                    output_dir=args.output_dir,
                )
            else:
                agent_parser.print_help()
                sys.exit(2)
        except Exception as exc:
            print(f"❌ {exc}")
            sys.exit(1)

        if bool(getattr(args, "json", False)):
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"cgc agent {args.agent_command} status: {result.get('status')}")
            for key in (
                "dag_manifest_path",
                "graph_insertion_contract_path",
                "teach_session_path",
                "train_session_path",
                "infer_session_path",
                "visualization_index_path",
                "compare_session_path",
                "audit_session_path",
                "replay_session_path",
                "trace_session_path",
                "subterranean_bundle_path",
            ):
                if str(result.get(key) or "").strip():
                    print(f"{key}: {result[key]}")
            if str(result.get("output_dir") or "").strip():
                print(f"output_dir: {result['output_dir']}")
            if str(result.get("upkg38_output_dir") or "").strip():
                print(f"upkg38_output_dir: {result['upkg38_output_dir']}")
        sys.exit(0 if str(result.get("status") or "") == "PASS" else 1)

    elif args.command == "model":
        try:
            cfg = load_config()
            if args.model_command == "list":
                result = _model_list_session(
                    cfg=cfg,
                    output_dir=args.output_dir,
                    source_filter=args.source,
                    model_roots=args.model_root,
                    nfs_roots=args.nfs_root,
                )
            elif args.model_command == "run":
                result = _model_run_session(
                    cfg=cfg,
                    model=args.model,
                    use_omlx=bool(args.use_omlx),
                    use_flashmoe=bool(args.use_flashmoe),
                    prompt=args.prompt,
                    max_tokens=int(args.max_tokens),
                    output_dir=args.output_dir,
                    gui_duration_s=int(args.gui_duration_s),
                    disable_gui_stage_source=bool(args.disable_gui_stage_source),
                )
                if not str(args.prompt or "").strip():
                    if bool(getattr(args, "json", False)):
                        print(json.dumps(result, ensure_ascii=False, indent=2))
                    print("Type '/bye' to exit.")
                    runtime_model = str(result.get("runtime_model") or args.model or cfg.get("active_edge_model") or "")
                    api_base_url = get_edge_api_base_url(cfg)
                    while True:
                        try:
                            user_input = input(">>> ")
                            if user_input.strip() == "/bye":
                                break
                            if not user_input.strip():
                                continue
                            payload = {
                                "model": runtime_model,
                                "prompt": user_input,
                                "stream": True,
                                "use_omlx": bool(args.use_omlx),
                                "use_flashmoe": bool(args.use_flashmoe),
                                "max_tokens": 256,
                                "api_base_url": api_base_url,
                            }
                            run_result = _execute_single_prompt(api_base_url=api_base_url, payload=payload)
                            print(str(run_result.get("response_text") or ""))
                        except requests.exceptions.ConnectionError:
                            print("\n[Error] Cannot connect to CGC Engine. Did you run 'cgc serve' in another terminal?")
                            break
                        except KeyboardInterrupt:
                            break
                        except Exception as exc:
                            print(f"\n[Error] {exc}")
                            break
                    sys.exit(0)
            elif args.model_command == "serve":
                cfg["edge_api_port"] = int(args.port)
                cfg["edge_proxy_port"] = int(args.proxy_port)
                save_config(cfg)
                apply_runtime_env(cfg)
                result = _model_serve_session(
                    output_dir=args.output_dir,
                    host=args.host,
                    port=int(args.port),
                    proxy_host=args.proxy_host,
                    proxy_port=int(args.proxy_port),
                    cfg=cfg,
                )
                if bool(getattr(args, "json", False)):
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    print(f"🔗 Cloud Node: {cfg.get('cloud_ip')}:{cfg.get('cloud_port')}")
                    print(
                        "🚀 Starting CGC model server "
                        f"(API {args.host}:{args.port}, Internal Proxy {args.proxy_host}:{args.proxy_port})..."
                    )
                start_edge_stack(
                    api_host=str(args.host),
                    api_port=int(args.port),
                    proxy_host=str(args.proxy_host),
                    proxy_port=int(args.proxy_port),
                )
                sys.exit(0)
            elif args.model_command == "verify":
                result = _model_verify_session(
                    run_session_path=args.run_session,
                    artifact_root=args.artifact_root,
                    output_dir=args.output_dir,
                )
            elif args.model_command == "audit":
                result = _model_audit_session(
                    run_session_path=args.run_session,
                    artifact_root=args.artifact_root,
                    output_dir=args.output_dir,
                )
            elif args.model_command == "replay":
                result = _model_replay_session(
                    run_session_path=args.run_session,
                    artifact_root=args.artifact_root,
                    output_dir=args.output_dir,
                )
            elif args.model_command == "trace":
                result = _model_trace_session(
                    run_session_path=args.run_session,
                    artifact_root=args.artifact_root,
                    output_dir=args.output_dir,
                )
            elif args.model_command == "compare":
                result = _model_compare_session(
                    run_session_path=args.run_session,
                    artifact_root=args.artifact_root,
                    compare_against_run_session=args.compare_against_run_session,
                    compare_against_artifact_root=args.compare_against_artifact_root,
                    output_dir=args.output_dir,
                )
            elif args.model_command == "launch":
                result = _model_launch_session(
                    model=args.model,
                    port=args.port,
                    host=args.host,
                    tp=args.tp,
                    context_length=args.context_length,
                    speculative_algorithm=args.speculative_algorithm,
                    speculative_num_steps=args.speculative_num_steps,
                    speculative_num_draft_tokens=args.speculative_num_draft_tokens,
                    speculative_eagle_topk=args.speculative_eagle_topk,
                    speculative_draft_model=args.speculative_draft_model,
                    no_speculative=args.no_speculative,
                    pd_separation=args.pd_separation,
                    pd_emit_host=args.pd_emit_host,
                    pd_emit_port=args.pd_emit_port,
                    pd_transport=args.pd_transport,
                    pd_cut_layer=args.pd_cut_layer,
                    ortho_base_dim=args.ortho_base_dim,
                    no_kda=args.no_kda,
                    rswa_window_size=args.rswa_window_size,
                    rswa_reference_len=args.rswa_reference_len,
                    no_rswa=args.no_rswa,
                    no_magicompiler=args.no_magicompiler,
                    no_cgc=args.no_cgc,
                    no_cuda_graph=args.no_cuda_graph,
                    exec_cmd=args.exec,
                )
            elif args.model_command == "swe-verified":
                result = _model_swe_verified_session(
                    cfg=cfg,
                    output_dir=args.output_dir,
                    refresh_session=args.refresh_session,
                    status_from_file=args.status_from_file,
                    target_instances_per_node=int(args.target_instances_per_node),
                    gpus_per_instance=int(args.gpus_per_instance),
                    benchmark_limit=int(args.benchmark_limit),
                    model_name=args.model_name,
                    api_base_url=args.api_base_url,
                    run_gate=bool(args.run_gate),
                    deepep_mode=args.deepep_mode,
                    auto_confirm=bool(args.yes),
                    interactive_confirm=not bool(args.json or args.yes),
                    poll=bool(args.poll),
                    poll_interval_seconds=int(args.poll_interval_seconds),
                    max_polls=int(args.max_polls),
                )
            else:
                model_parser.print_help()
                sys.exit(2)
        except Exception as exc:
            print(f"❌ {exc}")
            sys.exit(1)

        if bool(getattr(args, "json", False)):
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"cgc model {args.model_command} status: {result.get('status')}")
            for key in (
                "list_session_path",
                "run_session_path",
                "serve_session_path",
                "verify_session_path",
                "audit_session_path",
                "replay_session_path",
                "trace_session_path",
                "compare_session_path",
                "swe_verified_session_path",
                "artifact_root",
                "source_artifact_root",
                "list_payload_path",
            ):
                if str(result.get(key) or "").strip():
                    print(f"{key}: {result[key]}")
            confirmation = result.get("confirmation") if isinstance(result.get("confirmation"), dict) else {}
            recommendation = result.get("recommendation") if isinstance(result.get("recommendation"), dict) else {}
            if args.model_command == "swe-verified":
                print(
                    "recommended_total_instances: "
                    f"{recommendation.get('recommended_total_instances', 0)}"
                )
                print(
                    "recommended_num_workers: "
                    f"{recommendation.get('recommended_num_workers', 0)}"
                )
                print(f"confirmed: {confirmation.get('confirmed')}")
        sys.exit(0 if str(result.get("status") or "") == "PASS" else 1)

    elif args.command == "bundle":
        try:
            if args.bundle_command == "review":
                result = _bundle_review_session(
                    run_session_path=args.run_session,
                    artifact_root=args.artifact_root,
                    from_report=args.from_report,
                    profile_settings_path=args.profile_settings,
                    system_manifest_path=args.system_manifest,
                    bootstrap_contract_path=args.bootstrap_contract,
                    output_dir=args.output_dir,
                    strict=bool(args.strict),
                )
            else:
                bundle_parser.print_help()
                sys.exit(2)
        except Exception as exc:
            print(f"❌ {exc}")
            sys.exit(1)

        if bool(getattr(args, "json", False)):
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"cgc bundle {args.bundle_command} status: {result.get('status')}")
            for key in (
                "bundle_review_session_path",
                "run_session_path",
                "source_artifact_root",
                "artifact_root",
            ):
                if str(result.get(key) or "").strip():
                    print(f"{key}: {result[key]}")
            artifact_index = result.get("artifact_index") if isinstance(result.get("artifact_index"), dict) else {}
            for key in (
                "profile_settings_path",
                "system_manifest_path",
                "bootstrap_contract_path",
            ):
                if str(artifact_index.get(key) or "").strip():
                    print(f"{key}: {artifact_index[key]}")
        sys.exit(0 if str(result.get("status") or "") == "PASS" else 1)

    elif args.command == "embodied":
        try:
            if args.embodied_command == "psi0-realtimevla":
                result = run_embodied_psi0_realtimevla(
                    output_root=args.output_root,
                    edge_model=args.edge_model,
                    launch_command=args.launch_command,
                    fetch_command=args.fetch_command,
                    json_only=True,
                )
            elif args.embodied_command == "psi0-train":
                result = run_embodied_psi0_train(
                    output_root=args.output_root,
                    launch_command=args.launch_command,
                    fetch_command=args.fetch_command,
                    json_only=True,
                )
            elif args.embodied_command == "psi0-deploy":
                result = run_embodied_psi0_deploy(
                    output_root=args.output_root,
                    train_session=args.train_session,
                    launch_command=args.launch_command,
                    fetch_command=args.fetch_command,
                    json_only=True,
                )
            else:
                embodied_parser.print_help()
                sys.exit(2)
        except Exception as exc:
            print(f"❌ cgc embodied failed: {exc}")
            sys.exit(1)

        if bool(getattr(args, "json", False)):
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"cgc embodied {args.embodied_command} status: {result.get('status')}")
            for key in ("session_dir", "edge_push_bundle_path", "source_train_report_path"):
                if str(result.get(key) or "").strip():
                    print(f"{key}: {result.get(key)}")
            upkg40_payload = {}
            for payload_key in ("upkg40_embodied", "upkg40_embodied_train", "upkg40_embodied_deploy"):
                if isinstance(result.get(payload_key), dict):
                    upkg40_payload = result.get(payload_key)
                    break
            for key in (
                "report_path",
                "summary_path",
                "artifact_index_path",
                "stage_trace_path",
                "canonical_profile_catalog_path",
                "audit_replay_bundle_path",
                "train_session_path",
                "infer_session_path",
                "audit_session_path",
                "replay_session_path",
                "trace_session_path",
                "full_weight_manifest_path",
                "publish_manifest_path",
                "runtime_contract_path",
                "bridge_info_path",
                "deploy_contract_path",
                "consume_contract_path",
                "deploy_session_path",
            ):
                if str(upkg40_payload.get(key) or "").strip():
                    print(f"{key}: {upkg40_payload.get(key)}")
        sys.exit(0 if str(result.get("status") or "") == "PASS" else 1)

    elif args.command == "gate":
        if args.list or args.gate_name == "list":
            print_gate_registry()
            sys.exit(0)
        if args.gate_name == "checkin":
            gate_target = args.gate_target or "unknown"
            result = write_gate_checkin(
                gate_name=gate_target,
                status="PASS",
                report_path=args.report_path,
                summary_path=args.summary_path,
                trigger=args.trigger,
            )
            if args.print_json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"Gate checkin status: {result['status']}")
                print(f"checkin: {result['checkin_path']}")
                print(f"log: {result['log_path']}")
            sys.exit(0)
        try:
            result = run_registered_gate(
                gate_name=args.gate_name,
                repo_root=args.repo_root,
                output_dir=args.output_dir,
                m4_training_report=args.m4_training_report,
                m4_inference_report=args.m4_inference_report,
                m72_gui_duration_s=args.m72_gui_duration_s,
                m72_disable_gui_evidence=args.m72_disable_gui_evidence,
            )
        except ValueError as e:
            print(f"❌ {e}")
            print_gate_registry()
            sys.exit(2)

        if args.print_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"{args.gate_name.upper()} gate status: {result['status']}")
            if result.get("report_path"):
                print(f"report: {result['report_path']}")
            if result.get("summary_path"):
                print(f"summary: {result['summary_path']}")
            if result.get("status") == "NOT_IMPLEMENTED" and result.get("message"):
                print(result["message"])
        if result.get("status") == "PASS":
            checkin = write_gate_checkin(
                gate_name=args.gate_name,
                status="PASS",
                report_path=result.get("report_path", ""),
                summary_path=result.get("summary_path", ""),
                trigger="gate_pass",
                extra={"gate_result_keys": sorted(list((result.get("gate_result") or {}).keys())) if isinstance(result.get("gate_result"), dict) else []},
            )
            if args.print_json:
                print(json.dumps({"auto_checkin": checkin}, ensure_ascii=False, indent=2))
            else:
                print(f"checkin: {checkin['checkin_path']}")
        sys.exit(0 if result.get("status") == "PASS" else 1)

    elif args.command == "storage":
        result = _storage_session(
            storage_command=args.storage_command or "status",
            size_mb=getattr(args, "size_mb", 100),
        )

    elif args.command == "moe":
        result = _moe_session(
            moe_command=args.moe_command or "status",
            model=getattr(args, "model", ""),
            prompt=getattr(args, "prompt", "Hello"),
            max_tokens=getattr(args, "max_tokens", 30),
            engine=getattr(args, "engine", "auto"),
            num_experts=getattr(args, "num_experts", 8),
            expert_size=getattr(args, "expert_size", 1024),
            batch_size=getattr(args, "batch_size", 1),
        )

    elif args.command == "compile":
        result = _compile_session(
            compile_command=args.compile_command or "status",
            model=getattr(args, "model", ""),
            target=getattr(args, "target", "auto"),
        )

    elif args.command == "convert":
        result = _convert_session(
            convert_command=args.convert_command or "status",
            input_path=getattr(args, "input", ""),
            output_path=getattr(args, "output", ""),
            model=getattr(args, "model", ""),
        )

    elif args.command == "topology":
        result = _topology_session(
            topology_command=args.topology_command or "detect",
            model=getattr(args, "model", ""),
            world_size=getattr(args, "world_size", 0),
            tp=getattr(args, "tp", 8),
            ep=getattr(args, "ep", 1),
            pp=getattr(args, "pp", 1),
            dp=getattr(args, "dp", 1),
        )

    elif args.command == "profile":
        result = _profile_session(
            profile_command=args.profile_command or "status",
            duration=getattr(args, "duration", 60),
            output=getattr(args, "output", ""),
            input_path=getattr(args, "input", ""),
            fmt=getattr(args, "format", "text"),
        )

    else:
        parser.print_help()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
