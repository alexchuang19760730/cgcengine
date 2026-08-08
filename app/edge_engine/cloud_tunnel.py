from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PASSWORD = "Gen@song123"
DEFAULT_JUMP = "root@47.95.250.55"
DEFAULT_TARGET_USER = "root"


def _state_dir() -> Path:
    base = str(os.environ.get("CGC_HOME") or "").strip()
    if base:
        root = Path(base).expanduser().resolve()
        path = (root / "tunnels").resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    home_root = (Path.home() / ".cgc").resolve()
    try:
        home_root.mkdir(parents=True, exist_ok=True)
        probe = home_root / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        path = (home_root / "tunnels").resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        fallback = (Path(tempfile.gettempdir()) / "cgc_local" / "tunnels").resolve()
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


@dataclass
class TunnelSpec:
    name: str
    mode: str
    bind_port: int
    target_host: str
    target_port: int
    remote_host: str
    ssh_host: str
    healthcheck: str
    local_dependency_port: int | None = None


def _bool_env(name: str) -> bool:
    value = str(os.environ.get(name) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _password() -> str:
    return str(os.environ.get("CGC_CLOUD_TUNNEL_PASSWORD") or DEFAULT_PASSWORD)


def _jump() -> str:
    return str(os.environ.get("CGC_CLOUD_TUNNEL_JUMP") or DEFAULT_JUMP).strip() or DEFAULT_JUMP


def _target_user() -> str:
    return str(os.environ.get("CGC_CLOUD_TUNNEL_TARGET_USER") or DEFAULT_TARGET_USER).strip() or DEFAULT_TARGET_USER


def _proxy_option() -> str:
    password = _password()
    return (
        f"ProxyCommand=sshpass -p {password} ssh "
        "-o ServerAliveInterval=15 "
        "-o ServerAliveCountMax=6 "
        "-o TCPKeepAlive=yes "
        "-o StrictHostKeyChecking=no "
        "-o UserKnownHostsFile=/dev/null "
        f"-W %h:%p {_jump()}"
    )


def _ssh_base(ssh_host: str) -> list[str]:
    password = _password()
    return [
        "sshpass",
        "-p",
        password,
        "ssh",
        "-N",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=6",
        "-o",
        "TCPKeepAlive=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        _proxy_option(),
        f"{_target_user()}@{ssh_host}",
    ]


def _run_short(command: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def _can_connect_local(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.5)
    try:
        sock.connect(("127.0.0.1", int(port)))
        return True
    except Exception:
        return False
    finally:
        sock.close()


def _pid_path(spec: TunnelSpec) -> Path:
    return _state_dir() / f"{spec.name}.pid"


def _log_path(spec: TunnelSpec) -> Path:
    return _state_dir() / f"{spec.name}.log"


def _read_pid(spec: TunnelSpec) -> int | None:
    path = _pid_path(spec)
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _stop_process(pid: int | None) -> None:
    if pid is None:
        return
    try:
        os.kill(int(pid), signal.SIGTERM)
        time.sleep(0.5)
        if _process_alive(pid):
            os.kill(int(pid), signal.SIGKILL)
    except Exception:
        pass


def _remote_probe_http(spec: TunnelSpec) -> bool:
    code = (
        "python3 - <<'PY'\n"
        "import urllib.request\n"
        f"url = 'http://127.0.0.1:{int(spec.bind_port)}/does-not-exist'\n"
        "try:\n"
        "    urllib.request.urlopen(url, timeout=5)\n"
        "    print('OK')\n"
        "except Exception as exc:\n"
        "    msg = str(exc)\n"
        "    if '404' in msg or '405' in msg:\n"
        "        print('OK')\n"
        "    else:\n"
        "        print('BAD:' + type(exc).__name__ + ':' + msg)\n"
        "PY"
    )
    proc = _run_short(
        [
            "sshpass",
            "-p",
            _password(),
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            _proxy_option(),
            f"{_target_user()}@{spec.ssh_host}",
            code,
        ],
        timeout=25,
    )
    return proc.returncode == 0 and "OK" in (proc.stdout or "")


def _health_ok(spec: TunnelSpec) -> bool:
    if spec.healthcheck == "local_listener":
        return _can_connect_local(spec.bind_port)
    if spec.healthcheck == "remote_http":
        return _remote_probe_http(spec)
    raise RuntimeError(f"unknown_healthcheck: {spec.healthcheck}")


def _is_local_dependency_ready(spec: TunnelSpec) -> bool:
    if spec.local_dependency_port is None:
        return True
    return _can_connect_local(spec.local_dependency_port)


def _spawn_tunnel(spec: TunnelSpec) -> int:
    args = _ssh_base(spec.ssh_host)
    if spec.mode == "forward":
        args.insert(4, "-L")
    elif spec.mode == "reverse":
        args.insert(4, "-R")
    else:
        raise RuntimeError(f"unknown_mode: {spec.mode}")
    args.insert(5, f"{int(spec.bind_port)}:{spec.target_host}:{int(spec.target_port)}")
    with _log_path(spec).open("ab") as log_file:
        proc = subprocess.Popen(args, stdout=log_file, stderr=log_file)
    _pid_path(spec).write_text(f"{proc.pid}\n", encoding="utf-8")
    return int(proc.pid)


def _parse_target_spec(raw_target: str, default_port: int, remote_host: str, ssh_host: str) -> TunnelSpec | None:
    text = str(raw_target or "").strip()
    if not text:
        return None
    label = text
    endpoint = text
    if "=" in text:
        label, endpoint = [part.strip() for part in text.split("=", 1)]
    if ":" in endpoint:
        host, port_text = [part.strip() for part in endpoint.rsplit(":", 1)]
        try:
            local_port = int(port_text)
        except Exception:
            return None
    else:
        host = endpoint
        local_port = int(default_port)
    if host not in {"127.0.0.1", "localhost"}:
        return None
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in (label or f"port-{local_port}"))
    return TunnelSpec(
        name=f"cloud-forward-{safe_label}-{local_port}",
        mode="forward",
        bind_port=int(local_port),
        target_host="127.0.0.1",
        target_port=int(local_port),
        remote_host=remote_host,
        ssh_host=ssh_host,
        healthcheck="local_listener",
    )


def _resolve_forward_specs_from_env() -> list[TunnelSpec]:
    default_port = int(str(os.environ.get("CGC_CLOUD_PORT") or "50052").strip() or "50052")
    raw_targets = str(os.environ.get("CGC_CLOUD_TARGETS") or "").strip()
    if not raw_targets:
        return []
    remote_host = str(os.environ.get("CGC_CLOUD_HOST") or "").strip()
    if not remote_host:
        raw_hosts = [part.strip() for part in str(os.environ.get("CGC_CLOUD_HOSTS") or "").split(",") if part.strip()]
        remote_host = raw_hosts[0] if raw_hosts else ""
    if not remote_host or remote_host in {"127.0.0.1", "localhost"}:
        return []
    ssh_host = str(os.environ.get("CGC_CLOUD_TUNNEL_REMOTE_HOST") or "").strip() or remote_host
    specs: list[TunnelSpec] = []
    for raw_target in raw_targets.split(","):
        spec = _parse_target_spec(raw_target, default_port, remote_host, ssh_host)
        if spec is not None:
            specs.append(spec)
    return specs


def _resolve_reverse_spec_from_env() -> TunnelSpec | None:
    if not _bool_env("CGC_ENABLE_EDGE_REVERSE_TUNNEL"):
        return None
    remote_host = str(os.environ.get("CGC_CLOUD_HOST") or "").strip()
    if not remote_host:
        raw_hosts = [part.strip() for part in str(os.environ.get("CGC_CLOUD_HOSTS") or "").split(",") if part.strip()]
        remote_host = raw_hosts[0] if raw_hosts else ""
    if not remote_host or remote_host in {"127.0.0.1", "localhost"}:
        return None
    ssh_host = str(os.environ.get("CGC_CLOUD_TUNNEL_REMOTE_HOST") or "").strip() or remote_host
    local_port = int(str(os.environ.get("CGC_EDGE_REVERSE_TUNNEL_LOCAL_PORT") or os.environ.get("CGC_EDGE_PROXY_PORT") or "4000").strip() or "4000")
    remote_port = int(str(os.environ.get("CGC_EDGE_REVERSE_TUNNEL_REMOTE_PORT") or "18022").strip() or "18022")
    return TunnelSpec(
        name=f"edge-reverse-proxy-{remote_port}",
        mode="reverse",
        bind_port=remote_port,
        target_host="127.0.0.1",
        target_port=local_port,
        remote_host=remote_host,
        ssh_host=ssh_host,
        healthcheck="remote_http",
        local_dependency_port=local_port,
    )


def _resolve_tunnel_specs_from_env() -> list[TunnelSpec]:
    specs: list[TunnelSpec] = []
    if _bool_env("CGC_ENABLE_CLOUD_FORWARD_TUNNELS"):
        specs.extend(_resolve_forward_specs_from_env())
    reverse_spec = _resolve_reverse_spec_from_env()
    if reverse_spec is not None:
        specs.append(reverse_spec)
    return specs


def _describe_spec(spec: TunnelSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "mode": spec.mode,
        "bind_port": spec.bind_port,
        "target": f"{spec.target_host}:{spec.target_port}",
        "remote_host": spec.remote_host,
        "ssh_host": spec.ssh_host,
        "healthcheck": spec.healthcheck,
    }


def _ensure_tunnel(spec: TunnelSpec) -> dict[str, Any]:
    pid = _read_pid(spec)
    alive = _process_alive(pid)
    if not _is_local_dependency_ready(spec):
        return {
            **_describe_spec(spec),
            "status": "waiting_dependency",
            "pid": pid,
            "dependency_port": spec.local_dependency_port,
        }
    if spec.healthcheck == "remote_http" and not alive:
        new_pid = _spawn_tunnel(spec)
        time.sleep(1.5)
        return {
            **_describe_spec(spec),
            "status": "restarted" if _health_ok(spec) else "restart_pending",
            "pid": new_pid,
            "log_path": str(_log_path(spec)),
        }
    if _health_ok(spec):
        return {
            **_describe_spec(spec),
            "status": "healthy",
            "pid": pid,
        }
    if alive:
        _stop_process(pid)
    new_pid = _spawn_tunnel(spec)
    time.sleep(1.5)
    return {
        **_describe_spec(spec),
        "status": "restarted" if _health_ok(spec) else "restart_pending",
        "pid": new_pid,
        "log_path": str(_log_path(spec)),
    }


def ensure_cloud_tunnels_from_env() -> dict[str, Any]:
    specs = _resolve_tunnel_specs_from_env()
    if not specs:
        return {"enabled": False, "results": []}
    results: list[dict[str, Any]] = []
    for spec in specs:
        results.append(_ensure_tunnel(spec))
    return {"enabled": True, "results": results}


def ensure_cloud_forward_tunnels_from_env() -> dict[str, Any]:
    if not _bool_env("CGC_ENABLE_CLOUD_FORWARD_TUNNELS"):
        return {"enabled": False, "results": []}
    specs = _resolve_forward_specs_from_env()
    results = [_ensure_tunnel(spec) for spec in specs]
    return {"enabled": bool(specs), "results": results}


def describe_cloud_tunnels_from_env() -> str:
    specs = _resolve_tunnel_specs_from_env()
    if not specs:
        return ""
    return json.dumps([_describe_spec(spec) for spec in specs], ensure_ascii=False)


def describe_cloud_forward_from_env() -> str:
    specs = _resolve_forward_specs_from_env()
    if not specs:
        return ""
    return json.dumps(
        [
            {
                "local_port": spec.bind_port,
                "remote_host": spec.remote_host,
                "ssh_host": spec.ssh_host,
                "remote_port": spec.target_port,
            }
            for spec in specs
        ],
        ensure_ascii=False,
    )
