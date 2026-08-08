import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests


BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_REPO_ROOT))

from app.edge_engine.build import build_edge_engine
from app.edge_engine.cloud_tunnel import describe_cloud_tunnels_from_env
from app.edge_engine.service_manager import start_edge_stack


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


CGC_STATE_DIR = resolve_cgc_state_dir()
CONFIG_FILE = str((CGC_STATE_DIR / "config.json").resolve())
REPO_ROOT = Path(__file__).resolve().parents[2]
MINICPM5_OLLAMA_MODEL = "minicpm5-1b"
DEFAULT_CGC_BIN_DIR = (Path.home() / ".local" / "bin").resolve()
DEFAULT_CGC_COMMAND_NAME = "cgc"
SERVICE_LABEL = "com.cgc.edge.serve"
SERVICE_CONFIG_PATH = (CGC_STATE_DIR / "service.json").resolve()
SERVICE_PID_PATH = (CGC_STATE_DIR / "service.pid").resolve()
SERVICE_STDOUT_LOG = (CGC_STATE_DIR / "service.stdout.log").resolve()
SERVICE_STDERR_LOG = (CGC_STATE_DIR / "service.stderr.log").resolve()
DEFAULT_CLOUD_HOSTS = ["39.106.118.206", "47.95.250.55"]


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if str(cfg.get("cloud_ip") or "").strip() == "10.100.200.65" and int(cfg.get("cloud_port") or 50052) == 50052:
            cfg["cloud_ip"] = DEFAULT_CLOUD_HOSTS[0]
        cloud_hosts = cfg.get("cloud_hosts")
        if not isinstance(cloud_hosts, list) or not cloud_hosts:
            primary = str(cfg.get("cloud_ip") or "").strip() or DEFAULT_CLOUD_HOSTS[0]
            deduped_hosts: list[str] = []
            for host in [primary, *DEFAULT_CLOUD_HOSTS]:
                if host and host not in deduped_hosts:
                    deduped_hosts.append(host)
            cfg["cloud_hosts"] = deduped_hosts
        cloud_targets = cfg.get("cloud_targets")
        if not isinstance(cloud_targets, list):
            cfg["cloud_targets"] = []
        cfg["cloud_forward_tunnel_enabled"] = bool(cfg.get("cloud_forward_tunnel_enabled"))
        cfg["edge_reverse_tunnel_enabled"] = bool(cfg.get("edge_reverse_tunnel_enabled", cfg.get("cloud_forward_tunnel_enabled")))
        cfg["cloud_tunnel_ssh_host"] = str(cfg.get("cloud_tunnel_ssh_host") or "").strip()
        return cfg
    return {
        "cloud_ip": DEFAULT_CLOUD_HOSTS[0],
        "cloud_hosts": DEFAULT_CLOUD_HOSTS.copy(),
        "cloud_targets": [],
        "cloud_forward_tunnel_enabled": False,
        "edge_reverse_tunnel_enabled": False,
        "cloud_tunnel_ssh_host": "",
        "cloud_port": 50052,
        "active_edge_model": MINICPM5_OLLAMA_MODEL,
        "active_cloud_model": "deepseek-v4-flash:latest",
        "local_omlx_model": "",
        "local_flashmoe_model": "",
        "edge_api_port": 8000,
        "edge_proxy_port": 4000,
    }


def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def get_edge_api_base_url(cfg):
    return f"http://127.0.0.1:{int(cfg.get('edge_api_port', 8000) or 8000)}"


def _resolve_default_cloud_model_path() -> str:
    report_path = (REPO_ROOT / "ComputeGraphCompiler-main" / "Output" / "edge_runtime" / "cgc_run" / "latest_run_report.json").resolve()
    if not report_path.exists():
        return ""
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    for key in ("resolved_model_path", "model_ref"):
        candidate = str(payload.get(key) or "").strip()
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.exists() and path.is_dir():
            return str(path)
    return ""


def _normalized_argv() -> list[str]:
    return [arg for arg in sys.argv[1:] if not str(arg).startswith("-psn_")]


def _command_spec() -> tuple[list[str], dict[str, str]]:
    executable_path = Path(sys.executable).resolve()
    repo_python_env = {"PYTHONPATH": str(REPO_ROOT)}
    if executable_path.name.startswith("python"):
        return [str(executable_path), "-m", "app.cli.cgc_edge"], repo_python_env
    return [str(executable_path)], {}


def _write_command_wrapper(*, bin_dir: Path, command_name: str) -> Path:
    program_args, env_overrides = _command_spec()
    bin_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = (bin_dir / command_name).resolve()
    script_lines = ["#!/bin/bash", "set -e"]
    if env_overrides.get("PYTHONPATH"):
        quoted_repo_root = shlex.quote(env_overrides["PYTHONPATH"])
        script_lines.append(f'export PYTHONPATH={quoted_repo_root}${{PYTHONPATH:+:${{PYTHONPATH}}}}')
    quoted_args = " ".join(shlex.quote(arg) for arg in program_args)
    script_lines.append(f'exec {quoted_args} "$@"')
    wrapper_path.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
    wrapper_path.chmod(0o755)
    return wrapper_path


def _ensure_shell_path(bin_dir: Path) -> list[Path]:
    if not str(bin_dir).startswith(str(Path.home())):
        return []
    path_line = f'export PATH="{bin_dir}:$PATH"'
    updated_files: list[Path] = []
    for rc_path in [(Path.home() / ".zprofile").resolve(), (Path.home() / ".zshrc").resolve()]:
        existing = rc_path.read_text(encoding="utf-8") if rc_path.exists() else ""
        if path_line in existing:
            continue
        content = existing.rstrip("\n")
        addition = f"\n# Added by CGC installer\n{path_line}\n"
        rc_path.write_text((content + addition) if content else addition.lstrip("\n"), encoding="utf-8")
        updated_files.append(rc_path)
    return updated_files


def install_cgc_command(*, bin_dir: Path, command_name: str, update_shell_config: bool) -> None:
    wrapper_path = _write_command_wrapper(bin_dir=bin_dir, command_name=command_name)
    updated_files = _ensure_shell_path(bin_dir) if update_shell_config else []
    print(json.dumps({
        "status": "PASS",
        "command_path": str(wrapper_path),
        "bin_dir": str(bin_dir),
        "shell_files_updated": [str(path) for path in updated_files],
        "usage": [
            f"{command_name} serve",
            f"{command_name} list",
            f"{command_name} run <model>",
            f"{command_name} claude",
        ],
    }, ensure_ascii=False, indent=2))


def _service_program_arguments(*, api_host: str, api_port: int, proxy_host: str, proxy_port: int) -> tuple[list[str], dict[str, str]]:
    program_args, env_overrides = _command_spec()
    return [
        *program_args,
        "serve",
        "--host", str(api_host),
        "--port", str(api_port),
        "--proxy-host", str(proxy_host),
        "--proxy-port", str(proxy_port),
    ], env_overrides


def _service_definition(*, api_host: str, api_port: int, proxy_host: str, proxy_port: int) -> dict[str, object]:
    program_arguments, env_overrides = _service_program_arguments(
        api_host=api_host,
        api_port=api_port,
        proxy_host=proxy_host,
        proxy_port=proxy_port,
    )
    return {
        "label": SERVICE_LABEL,
        "program_arguments": program_arguments,
        "environment": env_overrides,
        "working_directory": str(REPO_ROOT),
        "api_host": str(api_host),
        "api_port": int(api_port),
        "proxy_host": str(proxy_host),
        "proxy_port": int(proxy_port),
    }


def install_service(*, api_host: str, api_port: int, proxy_host: str, proxy_port: int) -> Path:
    SERVICE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SERVICE_STDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    SERVICE_CONFIG_PATH.write_text(
        json.dumps(
            _service_definition(
                api_host=api_host,
                api_port=api_port,
                proxy_host=proxy_host,
                proxy_port=proxy_port,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return SERVICE_CONFIG_PATH


def _read_service_definition() -> dict[str, object]:
    if not SERVICE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Service config not found: {SERVICE_CONFIG_PATH}")
    return json.loads(SERVICE_CONFIG_PATH.read_text(encoding="utf-8"))


def _read_service_pid() -> int | None:
    if not SERVICE_PID_PATH.exists():
        return None
    try:
        return int(SERVICE_PID_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _is_pid_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def start_service() -> dict[str, object]:
    service_def = _read_service_definition()
    existing_pid = _read_service_pid()
    if _is_pid_running(existing_pid):
        return {"status": "PASS", "action": "start", "already_running": True, "pid": existing_pid}
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in dict(service_def.get("environment") or {}).items()})
    stdout_handle = SERVICE_STDOUT_LOG.open("a", encoding="utf-8")
    stderr_handle = SERVICE_STDERR_LOG.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        list(service_def.get("program_arguments") or []),
        cwd=str(service_def.get("working_directory") or REPO_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stdout_handle,
        stderr=stderr_handle,
        start_new_session=True,
        text=True,
    )
    SERVICE_PID_PATH.write_text(f"{proc.pid}\n", encoding="utf-8")
    return {"status": "PASS", "action": "start", "pid": proc.pid, "already_running": False}


def stop_service() -> dict[str, object]:
    pid = _read_service_pid()
    if not _is_pid_running(pid):
        if SERVICE_PID_PATH.exists():
            SERVICE_PID_PATH.unlink()
        return {"status": "PASS", "action": "stop", "pid": pid, "was_running": False}
    os.kill(int(pid), signal.SIGTERM)
    for _ in range(20):
        if not _is_pid_running(pid):
            break
        time.sleep(0.25)
    if _is_pid_running(pid):
        os.kill(int(pid), signal.SIGKILL)
    if SERVICE_PID_PATH.exists():
        SERVICE_PID_PATH.unlink()
    return {"status": "PASS", "action": "stop", "pid": pid, "was_running": True}


def service_status() -> dict[str, object]:
    pid = _read_service_pid()
    service_def = {}
    if SERVICE_CONFIG_PATH.exists():
        try:
            service_def = json.loads(SERVICE_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            service_def = {}
    return {
        "installed": SERVICE_CONFIG_PATH.exists(),
        "running": _is_pid_running(pid),
        "pid": pid,
        "service": SERVICE_LABEL,
        "config_path": str(SERVICE_CONFIG_PATH),
        "stdout_log": str(SERVICE_STDOUT_LOG),
        "stderr_log": str(SERVICE_STDERR_LOG),
        "api_port": service_def.get("api_port"),
        "proxy_port": service_def.get("proxy_port"),
    }


def uninstall_service() -> None:
    stop_service()
    if SERVICE_CONFIG_PATH.exists():
        SERVICE_CONFIG_PATH.unlink()


def fetch_models(*, api_base_url: str):
    response = requests.get(f"{api_base_url}/api/tags", timeout=5)
    response.raise_for_status()
    payload = response.json()
    return payload.get("models", [])


def _resolve_runtime_model(model_to_use: str, *, use_omlx: bool, use_flashmoe: bool, cfg):
    if use_flashmoe and str(cfg.get("local_flashmoe_model") or "").strip():
        return str(cfg.get("local_flashmoe_model"))
    if use_omlx and str(cfg.get("local_omlx_model") or "").strip():
        return str(cfg.get("local_omlx_model"))
    return str(model_to_use)


def _apply_runtime_env(cfg) -> None:
    cloud_targets = cfg.get("cloud_targets")
    if isinstance(cloud_targets, list):
        normalized_targets = [str(target or "").strip() for target in cloud_targets if str(target or "").strip()]
    else:
        normalized_targets = []
    cloud_hosts = cfg.get("cloud_hosts")
    if not isinstance(cloud_hosts, list) or not cloud_hosts:
        cloud_hosts = [str(cfg.get("cloud_ip") or DEFAULT_CLOUD_HOSTS[0]).strip() or DEFAULT_CLOUD_HOSTS[0]]
    normalized_hosts: list[str] = []
    for host in cloud_hosts:
        host_text = str(host or "").strip()
        if host_text and host_text not in normalized_hosts:
            normalized_hosts.append(host_text)
    if not normalized_hosts:
        normalized_hosts = DEFAULT_CLOUD_HOSTS.copy()
    os.environ["CGC_CLOUD_HOSTS"] = ",".join(normalized_hosts)
    os.environ["CGC_CLOUD_HOST"] = normalized_hosts[0]
    os.environ["CGC_CLOUD_PORT"] = str(int(cfg.get("cloud_port") or 50052))
    os.environ["CGC_EDGE_API_PORT"] = str(int(cfg.get("edge_api_port") or 8000))
    os.environ["CGC_EDGE_PROXY_PORT"] = str(int(cfg.get("edge_proxy_port") or 4000))
    if normalized_targets:
        os.environ["CGC_CLOUD_TARGETS"] = ",".join(normalized_targets)
    else:
        os.environ.pop("CGC_CLOUD_TARGETS", None)
    cloud_forward_tunnel_enabled = bool(cfg.get("cloud_forward_tunnel_enabled"))
    cloud_tunnel_ssh_host = str(cfg.get("cloud_tunnel_ssh_host") or "").strip()
    if cloud_tunnel_ssh_host:
        os.environ["CGC_CLOUD_TUNNEL_REMOTE_HOST"] = cloud_tunnel_ssh_host
    else:
        os.environ.pop("CGC_CLOUD_TUNNEL_REMOTE_HOST", None)
    if cloud_forward_tunnel_enabled:
        if not normalized_targets:
            os.environ["CGC_CLOUD_TARGETS"] = f"fan1=127.0.0.1:{int(cfg.get('cloud_port') or 50052)}"
        os.environ["CGC_ENABLE_CLOUD_FORWARD_TUNNELS"] = "1"
    else:
        os.environ.pop("CGC_ENABLE_CLOUD_FORWARD_TUNNELS", None)
    edge_reverse_tunnel_enabled = bool(cfg.get("edge_reverse_tunnel_enabled", cloud_forward_tunnel_enabled))
    if edge_reverse_tunnel_enabled:
        os.environ["CGC_ENABLE_EDGE_REVERSE_TUNNEL"] = "1"
        os.environ["CGC_EDGE_REVERSE_TUNNEL_LOCAL_PORT"] = str(int(cfg.get("edge_proxy_port") or 4000))
        os.environ["CGC_EDGE_REVERSE_TUNNEL_REMOTE_PORT"] = str(int(cfg.get("edge_reverse_tunnel_remote_port") or 18022))
    else:
        os.environ.pop("CGC_ENABLE_EDGE_REVERSE_TUNNEL", None)
        os.environ.pop("CGC_EDGE_REVERSE_TUNNEL_LOCAL_PORT", None)
        os.environ.pop("CGC_EDGE_REVERSE_TUNNEL_REMOTE_PORT", None)
    cloud_model_path = str(os.environ.get("CGC_CLOUD_MODEL_PATH") or "").strip() or _resolve_default_cloud_model_path()
    if cloud_model_path:
        os.environ["CGC_CLOUD_MODEL_PATH"] = cloud_model_path
    local_omlx_model = str(cfg.get("local_omlx_model") or "").strip()
    if local_omlx_model:
        os.environ["CGC_LOCAL_OMLX_MODEL"] = local_omlx_model

    local_flashmoe_model = str(cfg.get("local_flashmoe_model") or "").strip()
    if local_flashmoe_model:
        os.environ["CGC_LOCAL_FLASHMOE_MODEL"] = local_flashmoe_model

    edge_model = str(cfg.get("active_edge_model") or MINICPM5_OLLAMA_MODEL).strip()
    if edge_model:
        os.environ["CGC_MINICPM5_MODEL"] = edge_model
    else:
        os.environ.pop("CGC_MINICPM5_MODEL", None)

    if "minicpm5" in edge_model.lower():
        os.environ["CGC_ENABLE_MINICPM5_ROUTER"] = "1"
    else:
        os.environ["CGC_ENABLE_MINICPM5_ROUTER"] = "0"


def _prepare_claude_env(cfg) -> dict[str, str]:
    env = os.environ.copy()
    env["CLAUDE_CODE_SIMPLE"] = "1"
    env["ANTHROPIC_CUSTOM_MODEL_OPTION_NAME"] = "DeepSeek V4 Flash (CGC Edge)"
    env["ANTHROPIC_CUSTOM_MODEL_OPTION"] = str(cfg.get("active_cloud_model") or "deepseek-v4-flash:latest")
    cloud_hosts = cfg.get("cloud_hosts") if isinstance(cfg.get("cloud_hosts"), list) else None
    if not cloud_hosts:
        cloud_hosts = [str(cfg.get("cloud_ip") or DEFAULT_CLOUD_HOSTS[0]).strip() or DEFAULT_CLOUD_HOSTS[0]]
    cloud_desc = ",".join([str(host).strip() for host in cloud_hosts if str(host).strip()])
    env["ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION"] = f"FusionRoute 4x Expert Pool on {cloud_desc}"
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{int(cfg.get('edge_proxy_port', 4000) or 4000)}"
    env["ANTHROPIC_API_KEY"] = "sk-cgc-edge-key"
    env["ANTHROPIC_AUTH_TOKEN"] = "sk-cgc-edge-key"
    env["CLAUDE_MODEL"] = str(cfg.get("active_cloud_model") or "deepseek-v4-flash:latest")
    env["ANTHROPIC_MODEL"] = str(cfg.get("active_cloud_model") or "deepseek-v4-flash:latest")
    for key in [
        "ANTHROPIC_AUTH_TOKEN_OLD",
        "CLAUDE_TOKEN",
        "CLAUDE_OAUTH_TOKEN",
        "ANTHROPIC_OAUTH_TOKEN",
    ]:
        env.pop(key, None)
    return env


def _normalize_claude_args(raw_args) -> list[str]:
    args = list(raw_args or [])
    has_print = any(arg in {"-p", "--print"} for arg in args)
    has_no_persist = any(arg == "--no-session-persistence" for arg in args)
    if has_print and not has_no_persist:
        args.append("--no-session-persistence")
    return args


def run_interactive(model_to_use: str, *, api_base_url: str, use_omlx: bool, use_flashmoe: bool, cfg):
    print(f"🚀 Starting CGC Edge interactive session with model: {model_to_use}")
    if use_flashmoe and str(cfg.get("local_flashmoe_model") or "").strip():
        print(f"  [Edge Runtime] 本地 FlashMoE 模型: {cfg.get('local_flashmoe_model')}")
    elif use_omlx and str(cfg.get("local_omlx_model") or "").strip():
        print(f"  [Edge Runtime] 本地 OMLX 模型: {cfg.get('local_omlx_model')}")

    print("Type '/bye' to exit.")
    while True:
        try:
            user_input = input(">>> ")
            if user_input.strip() == "/bye":
                break
            if not user_input.strip():
                continue

            runtime_model = _resolve_runtime_model(
                model_to_use,
                use_omlx=use_omlx,
                use_flashmoe=use_flashmoe,
                cfg=cfg,
            )
            payload = {
                "model": runtime_model,
                "prompt": user_input,
                "stream": True,
                "use_omlx": bool(use_omlx),
                "use_flashmoe": bool(use_flashmoe),
                "max_tokens": 256,
            }
            response = requests.post(f"{api_base_url}/api/generate", json=payload, stream=True, timeout=(10, 600))
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    print(data.get("response", ""), end="", flush=True)
            print()
        except requests.exceptions.ConnectionError:
            print("\n[Error] Cannot connect to CGC Edge Engine. Did you run 'cgc serve'?")
            break
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"\n[Error] {exc}")
            break


def main():
    argv = _normalized_argv()
    parser = argparse.ArgumentParser(description="CGC Edge Engine product CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    serve_parser = subparsers.add_parser("serve", help="Start the CGC Edge API server stack")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--proxy-port", type=int, default=4000)
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--proxy-host", type=str, default="127.0.0.1")

    claude_parser = subparsers.add_parser("claude", help="Launch Claude Code CLI with CGC environment", add_help=False)
    claude_parser.add_argument("claude_args", nargs=argparse.REMAINDER)

    config_parser = subparsers.add_parser("config", help="Configure CGC Edge Engine")
    config_parser.add_argument("--set-cloud-ip", type=str)
    config_parser.add_argument("--set-cloud-hosts", type=str, help="Comma separated cloud hosts")
    config_parser.add_argument("--set-cloud-targets", type=str, help="Comma separated explicit cloud targets, e.g. a=host1:50053,b=host2:50053")
    config_parser.add_argument("--set-cloud-port", type=int)
    config_parser.add_argument("--set-cloud-model", type=str)
    config_parser.add_argument("--set-edge-model", type=str)
    config_parser.add_argument("--set-local-omlx-model", type=str)
    config_parser.add_argument("--set-local-flashmoe-model", type=str)
    config_parser.add_argument("--set-edge-api-port", type=int)
    config_parser.add_argument("--set-edge-proxy-port", type=int)
    config_parser.add_argument("--set-cloud-forward-tunnel", type=str, help="Enable or disable local SSH forward for localhost cloud_targets: true/false")
    config_parser.add_argument("--set-edge-reverse-tunnel", type=str, help="Enable or disable reverse SSH tunnel for exposing the local internal proxy on remote 18022: true/false")
    config_parser.add_argument("--set-cloud-tunnel-ssh-host", type=str, help="Explicit SSH host for cloud tunnels when service endpoint differs from the SSH entry host")

    run_parser = subparsers.add_parser("run", help="Run the Edge Engine interactively")
    run_parser.add_argument("model", nargs="?", default="")
    run_parser.add_argument("--use-omlx", action="store_true")
    run_parser.add_argument("--use-flashmoe", action="store_true")

    subparsers.add_parser("list", help="List available models from the Edge API")

    build_parser = subparsers.add_parser("build", help="Build standalone edge executable")
    build_parser.add_argument("--output-dir", type=str, default=str((REPO_ROOT / "dist" / "cgc").resolve()))

    install_parser = subparsers.add_parser("install", help="Install the `cgc` command wrapper")
    install_parser.add_argument("--bin-dir", type=str, default=str(DEFAULT_CGC_BIN_DIR))
    install_parser.add_argument("--command-name", type=str, default=DEFAULT_CGC_COMMAND_NAME)
    install_parser.add_argument("--no-shell-config", action="store_true", help="Do not update shell rc files")

    service_parser = subparsers.add_parser("service", help="Manage the background CGC serve service")
    service_subparsers = service_parser.add_subparsers(dest="service_command", help="Service commands")
    service_install_parser = service_subparsers.add_parser("install", help="Write the launchctl plist for background serve")
    service_install_parser.add_argument("--api-host", type=str, default="0.0.0.0")
    service_install_parser.add_argument("--api-port", type=int, default=8000)
    service_install_parser.add_argument("--proxy-host", type=str, default="127.0.0.1")
    service_install_parser.add_argument("--proxy-port", type=int, default=4000)
    service_install_parser.add_argument("--start", action="store_true", help="Start the service after installing the plist")
    service_subparsers.add_parser("start", help="Start the background serve service")
    service_subparsers.add_parser("stop", help="Stop the background serve service")
    service_subparsers.add_parser("restart", help="Restart the background serve service")
    service_subparsers.add_parser("status", help="Show service status")
    service_subparsers.add_parser("uninstall", help="Remove the background serve service")

    if argv and argv[0] == "claude":
        args = argparse.Namespace(command="claude", claude_args=argv[1:])
    else:
        args = parser.parse_args(argv)
    cfg = load_config()

    if args.command == "serve":
        cfg["edge_api_port"] = int(args.port)
        cfg["edge_proxy_port"] = int(args.proxy_port)
        _apply_runtime_env(cfg)
        cloud_targets_display = str(os.environ.get("CGC_CLOUD_TARGETS") or "").strip()
        if cloud_targets_display:
            print(f"🔗 Cloud Targets: {cloud_targets_display}")
        else:
            print(f"🔗 Cloud Nodes: {os.environ.get('CGC_CLOUD_HOSTS')}:{os.environ.get('CGC_CLOUD_PORT')}")
        tunnel_desc = describe_cloud_tunnels_from_env()
        if tunnel_desc:
            print(f"🔁 Cloud Tunnels: {tunnel_desc}")
        save_config(cfg)
        start_edge_stack(
            api_host=str(args.host),
            api_port=int(args.port),
            proxy_host=str(args.proxy_host),
            proxy_port=int(args.proxy_port),
        )
        return

    if args.command == "claude":
        env = _prepare_claude_env(cfg)
        claude_args = _normalize_claude_args(args.claude_args)
        subprocess.run(["claude", *claude_args], env=env, check=False)
        return

    if args.command == "config":
        updated = False
        if args.set_cloud_ip:
            cfg["cloud_ip"] = args.set_cloud_ip
            cfg["cloud_hosts"] = [args.set_cloud_ip]
            updated = True
        if args.set_cloud_hosts:
            hosts = [part.strip() for part in str(args.set_cloud_hosts).split(",") if part.strip()]
            if hosts:
                cfg["cloud_hosts"] = hosts
                cfg["cloud_ip"] = hosts[0]
            updated = True
        if args.set_cloud_targets is not None:
            cfg["cloud_targets"] = [part.strip() for part in str(args.set_cloud_targets).split(",") if part.strip()]
            updated = True
        if args.set_cloud_port is not None:
            cfg["cloud_port"] = int(args.set_cloud_port)
            updated = True
        if args.set_cloud_model:
            cfg["active_cloud_model"] = args.set_cloud_model
            updated = True
        if args.set_edge_model:
            cfg["active_edge_model"] = args.set_edge_model
            updated = True
        if args.set_local_omlx_model:
            cfg["local_omlx_model"] = args.set_local_omlx_model
            updated = True
        if args.set_local_flashmoe_model:
            cfg["local_flashmoe_model"] = args.set_local_flashmoe_model
            updated = True
        if args.set_edge_api_port is not None:
            cfg["edge_api_port"] = int(args.set_edge_api_port)
            updated = True
        if args.set_edge_proxy_port is not None:
            cfg["edge_proxy_port"] = int(args.set_edge_proxy_port)
            updated = True
        if args.set_cloud_forward_tunnel is not None:
            value = str(args.set_cloud_forward_tunnel).strip().lower()
            cfg["cloud_forward_tunnel_enabled"] = value in {"1", "true", "yes", "on"}
            updated = True
        if args.set_edge_reverse_tunnel is not None:
            value = str(args.set_edge_reverse_tunnel).strip().lower()
            cfg["edge_reverse_tunnel_enabled"] = value in {"1", "true", "yes", "on"}
            updated = True
        if args.set_cloud_tunnel_ssh_host is not None:
            cfg["cloud_tunnel_ssh_host"] = str(args.set_cloud_tunnel_ssh_host).strip()
            updated = True
        if updated:
            save_config(cfg)
            print(json.dumps(cfg, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return

    if args.command == "run":
        api_base_url = get_edge_api_base_url(cfg)
        model_to_use = str(args.model or cfg.get("active_edge_model") or MINICPM5_OLLAMA_MODEL)
        run_interactive(
            model_to_use,
            api_base_url=api_base_url,
            use_omlx=bool(args.use_omlx),
            use_flashmoe=bool(args.use_flashmoe),
            cfg=cfg,
        )
        return

    if args.command == "list":
        for model in fetch_models(api_base_url=get_edge_api_base_url(cfg)):
            print(str(model.get("name") or model.get("model") or "unknown"))
        return

    if args.command == "build":
        result = build_edge_engine(repo_root=REPO_ROOT, output_dir=Path(args.output_dir))
        print(json.dumps({
            "status": result.status,
            "python_bin": result.python_bin,
            "builder": result.builder,
            "output_path": result.output_path,
            "executable_path": result.executable_path,
            "command": result.command,
        }, ensure_ascii=False, indent=2))
        return

    if args.command == "install":
        install_cgc_command(
            bin_dir=Path(args.bin_dir).expanduser().resolve(),
            command_name=str(args.command_name or DEFAULT_CGC_COMMAND_NAME).strip() or DEFAULT_CGC_COMMAND_NAME,
            update_shell_config=not bool(args.no_shell_config),
        )
        return

    if args.command == "service":
        if args.service_command == "install":
            config_path = install_service(
                api_host=str(args.api_host),
                api_port=int(args.api_port),
                proxy_host=str(args.proxy_host),
                proxy_port=int(args.proxy_port),
            )
            payload = {
                "status": "PASS",
                "config_path": str(config_path),
                "stdout_log": str(SERVICE_STDOUT_LOG),
                "stderr_log": str(SERVICE_STDERR_LOG),
            }
            if args.start:
                payload["start_result"] = start_service()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        if args.service_command == "start":
            print(json.dumps(start_service(), ensure_ascii=False, indent=2))
            return
        if args.service_command == "stop":
            print(json.dumps(stop_service(), ensure_ascii=False, indent=2))
            return
        if args.service_command == "restart":
            stop_result = stop_service()
            start_result = start_service()
            print(json.dumps({"status": "PASS", "action": "restart", "stop_result": stop_result, "start_result": start_result}, ensure_ascii=False, indent=2))
            return
        if args.service_command == "status":
            print(json.dumps(service_status(), ensure_ascii=False, indent=2))
            return
        if args.service_command == "uninstall":
            uninstall_service()
            print(json.dumps({"status": "PASS", "service": SERVICE_LABEL, "action": "uninstall"}, ensure_ascii=False, indent=2))
            return
        service_parser.print_help()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
