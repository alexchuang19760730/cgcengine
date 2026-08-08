from __future__ import annotations

import multiprocessing
import os
import time
from typing import Dict, List

from app.edge_engine.cloud_tunnel import ensure_cloud_tunnels_from_env


def _run_api_server(host: str, port: int) -> None:
    import uvicorn

    os.environ["CGC_EDGE_API_PORT"] = str(int(port))
    from app.servers.cgc_api_server import app

    uvicorn.run(app, host=host, port=port, log_level=os.environ.get("CGC_LOG_LEVEL", "info"))


def _run_internal_proxy(host: str, port: int, target_base_url: str) -> None:
    import uvicorn

    os.environ["CGC_INTERNAL_PROXY_TARGET"] = str(target_base_url)
    from app.servers.internal_proxy_server import app

    uvicorn.run(app, host=host, port=port, log_level=os.environ.get("CGC_LOG_LEVEL", "info"))


def _run_cloud_socket_gateway(host: str, port: int) -> None:
    from app.servers.cloud_socket_server import start_server

    start_server(host=host, port=port)


def _terminate_processes(processes: List[multiprocessing.Process]) -> None:
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
    for proc in processes:
        proc.join(timeout=5)


def start_edge_stack(
    *,
    api_host: str = "0.0.0.0",
    api_port: int = 8000,
    proxy_host: str = "127.0.0.1",
    proxy_port: int = 4000,
) -> Dict[str, object]:
    target_base_url = f"http://127.0.0.1:{int(api_port)}"
    preflight_tunnel_report = ensure_cloud_tunnels_from_env()
    if bool(preflight_tunnel_report.get("enabled")):
        print(f"🔁 Preflight Cloud Tunnels: {preflight_tunnel_report}")
    cloud_host = str(os.environ.get("CGC_CLOUD_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    cloud_port = int(str(os.environ.get("CGC_CLOUD_PORT") or "50052").strip() or "50052")
    processes = [
        multiprocessing.Process(
            target=_run_api_server,
            name="cgc-api-server",
            args=(api_host, int(api_port)),
        ),
        multiprocessing.Process(
            target=_run_internal_proxy,
            name="cgc-internal-proxy",
            args=(proxy_host, int(proxy_port), target_base_url),
        ),
    ]
    if cloud_host in {"127.0.0.1", "localhost"}:
        processes.append(
            multiprocessing.Process(
                target=_run_cloud_socket_gateway,
                name="cgc-cloud-socket-gateway",
                args=("0.0.0.0", int(cloud_port)),
            )
        )

    for proc in processes:
        proc.start()

    # Reverse tunnels depend on the local proxy listener, so re-run ensure after
    # the child processes have bound their local ports.
    time.sleep(1.5)
    tunnel_report = ensure_cloud_tunnels_from_env()
    if bool(tunnel_report.get("enabled")):
        print(f"🔁 Ensured Cloud Tunnels: {tunnel_report}")

    try:
        while True:
            for proc in processes:
                if not proc.is_alive():
                    exitcode = proc.exitcode
                    _terminate_processes(processes)
                    raise RuntimeError(f"{proc.name} exited unexpectedly with code {exitcode}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        _terminate_processes(processes)
        return {
            "status": "STOPPED",
            "api_url": f"http://127.0.0.1:{int(api_port)}",
            "proxy_url": f"http://127.0.0.1:{int(proxy_port)}",
        }
