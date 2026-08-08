from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_lplb_solver():
    """延迟加载 LPLB solver，避免在 cgc_engine 不可用时整个 orchestrator 失败。"""
    try:
        import sys as _sys
        cgc_engine_path = REPO_ROOT / "ComputeGraphCompiler-main"
        if str(cgc_engine_path) not in _sys.path:
            _sys.path.insert(0, str(cgc_engine_path))
        from cgc_engine.lplb_solver import solve_lplb  # noqa: WPS433
        return solve_lplb
    except Exception:
        return None


def _default_topology_path() -> Path:
    return (
        REPO_ROOT
        / "ComputeGraphCompiler-main"
        / "Output"
        / "cli_gate_upkg39"
        / "four_instance_topology.json"
    ).resolve()


def _default_placement_path() -> Path:
    return (
        REPO_ROOT
        / "ComputeGraphCompiler-main"
        / "Output"
        / "cli_gate_upkg39"
        / "fusionroute_cloud_placement_latest.json"
    ).resolve()


def _default_runtime_path() -> Path:
    return (
        REPO_ROOT
        / "ComputeGraphCompiler-main"
        / "Output"
        / "cli_gate_upkg39"
        / "fusionroute_cloud_runtime.json"
    ).resolve()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _wait_port(host: str, port: int, timeout_s: float = 8.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
            return True
        except Exception:
            time.sleep(0.2)
        finally:
            sock.close()
    return False


class SshTunnel:
    def __init__(
        self,
        *,
        local_port: int,
        remote_host: str,
        remote_port: int,
        ssh_host: str,
        ssh_user: str,
        ssh_password: str,
    ):
        self.local_port = int(local_port)
        self.remote_host = str(remote_host)
        self.remote_port = int(remote_port)
        self.ssh_host = str(ssh_host)
        self.ssh_user = str(ssh_user)
        self.ssh_password = str(ssh_password)
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        if not self.ssh_password:
            raise RuntimeError(f"missing ssh password for tunnel target {self.ssh_host}")
        cmd = [
            "sshpass",
            "-p",
            self.ssh_password,
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-N",
            "-L",
            f"127.0.0.1:{self.local_port}:{self.remote_host}:{self.remote_port}",
            f"{self.ssh_user}@{self.ssh_host}",
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if not _wait_port("127.0.0.1", self.local_port, timeout_s=8.0):
            self.stop()
            raise RuntimeError(
                f"failed to establish tunnel {self.local_port}->{self.remote_host}:{self.remote_port}"
            )

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


class GatewayTarget:
    def __init__(self, instance: dict[str, Any], current_host_label: str):
        self.instance = dict(instance)
        self.instance_id = str(instance.get("instance_id") or "")
        self.host_label = str(instance.get("host_label") or "")
        self.host_ip = str(instance.get("host_ip") or "")
        self.gateway_port = int(instance.get("gateway_port") or 0)
        self.ssh_host = str(instance.get("ssh_host") or self.host_ip)
        self.remote_host = "127.0.0.1"
        self.local_port = self.gateway_port
        self.base_url = f"http://127.0.0.1:{self.local_port}"
        self.tunnel: SshTunnel | None = None
        self._current_host_label = current_host_label

    @property
    def requires_tunnel(self) -> bool:
        return self.host_label != self._current_host_label

    def ensure_tunnel(self, password_lookup: dict[str, str]) -> None:
        if not self.requires_tunnel:
            return
        if self.tunnel is not None and self.tunnel.process is not None and self.tunnel.process.poll() is None:
            return
        local_port = self.gateway_port + 10000
        self.local_port = local_port
        self.base_url = f"http://127.0.0.1:{self.local_port}"
        password = str(password_lookup.get(self.host_label) or "")
        self.tunnel = SshTunnel(
            local_port=local_port,
            remote_host=self.remote_host,
            remote_port=self.gateway_port,
            ssh_host=self.ssh_host,
            ssh_user="root",
            ssh_password=password,
        )
        self.tunnel.start()

    def stop(self) -> None:
        if self.tunnel is not None:
            self.tunnel.stop()


class FusionRouteCloudOrchestrator:
    def __init__(self):
        self.current_host_label = str(
            os.environ.get("CGC_FUSIONROUTE_ORCH_HOST_LABEL") or "host1"
        ).strip()
        self.topology_path = Path(
            os.environ.get("CGC_FUSIONROUTE_TOPOLOGY_PATH") or _default_topology_path()
        ).expanduser().resolve()
        self.placement_path = Path(
            os.environ.get("CGC_FUSIONROUTE_PLACEMENT_REPORT_PATH")
            or _default_placement_path()
        ).expanduser().resolve()
        self.runtime_path = Path(
            os.environ.get("CGC_FUSIONROUTE_RUNTIME_EVIDENCE_PATH")
            or _default_runtime_path()
        ).expanduser().resolve()
        self.password_lookup = {
            "host1": str(os.environ.get("CGC_FUSIONROUTE_HOST1_PASSWORD") or ""),
            "host2": str(os.environ.get("CGC_FUSIONROUTE_HOST2_PASSWORD") or ""),
        }
        self.session = requests.Session()
        self.lock = Lock()
        self.next_index = 0
        self.targets = self._load_targets()
        self._ensure_tunnels()
        # LPLB 集成：基于 GPU 负载的线性规划均衡器（Gate 2.2 第 3 层）
        self._lplb_solve = _load_lplb_solver()
        self._lplb_enabled = bool(self._lplb_solve) and str(
            os.environ.get("CGC_FUSIONROUTE_LPLB_ENABLED", "1")
        ).strip() in {"1", "true", "True", "yes"}
        # 实例负载历史（用于 LPLB 输入），instance_id -> 累计 load
        self._instance_loads: dict[str, float] = {
            t.instance_id: 0.0 for t in self.targets
        }
        atexit.register(self.shutdown)

    def _load_targets(self) -> list[GatewayTarget]:
        payload = json.loads(self.topology_path.read_text(encoding="utf-8"))
        entries = (
            ((payload.get("routing_topology_profile") or {}).get("instance_topology"))
            or []
        )
        targets = [GatewayTarget(entry, self.current_host_label) for entry in entries]
        if not targets:
            raise RuntimeError("no instance_topology entries found")
        return targets

    def _ensure_tunnels(self) -> None:
        for target in self.targets:
            target.ensure_tunnel(self.password_lookup)

    def shutdown(self) -> None:
        for target in self.targets:
            target.stop()

    def _probe_target(self, target: GatewayTarget) -> tuple[bool, str]:
        try:
            resp = self.session.get(f"{target.base_url}/health", timeout=(2, 4))
            if resp.ok:
                return True, resp.text[:400]
            return False, f"http_{resp.status_code}"
        except Exception as exc:
            return False, str(exc)

    def _healthy_targets(self) -> list[tuple[GatewayTarget, str]]:
        healthy: list[tuple[GatewayTarget, str]] = []
        for target in self.targets:
            ok, detail = self._probe_target(target)
            if ok:
                healthy.append((target, detail))
        return healthy

    def select_target(self) -> tuple[GatewayTarget, list[dict[str, Any]]]:
        healthy = self._healthy_targets()
        probe_rows = [
            {
                "instance_id": target.instance_id,
                "host_label": target.host_label,
                "base_url": target.base_url,
                "healthy": True,
                "detail": detail,
            }
            for target, detail in healthy
        ]
        if not healthy:
            raise RuntimeError("no healthy targets available")

        selected_idx = self._lplb_select(healthy)
        return healthy[selected_idx][0], probe_rows

    def _lplb_select(self, healthy: list[tuple["GatewayTarget", str]]) -> int:
        """LPLB 负载均衡选择：若有负载历史则用 LPLB 求解最优分配，否则回退 round-robin。"""
        if not self._lplb_enabled or not self._lplb_solve:
            with self.lock:
                idx = self.next_index % len(healthy)
                self.next_index += 1
            return idx

        try:
            import numpy as np
            # 收集健康实例的累计负载作为 LPLB 输入
            loads = np.array(
                [self._instance_loads.get(t.instance_id, 0.0) + 1.0 for t, _ in healthy],
                dtype=np.float64,
            )
            # 容量上限 = 平均负载 * 1.5（给 LPLB 留余量）
            avg_load = float(loads.mean()) if len(loads) > 0 else 1.0
            capacities = np.full(len(healthy), max(avg_load * 1.5, 1.0), dtype=np.float64)
            result = self._lplb_solve(
                loads=loads,
                capacities=capacities,
                num_replicas=1,
                use_gpu=False,  # orchestrator 不假设有 GPU
            )
            # assignment[i] = 该负载单元应分配到的 "GPU"（这里复用为 instance index）
            # 选择方差最小化后的第一个 instance
            if result.assignment is not None and len(result.assignment) > 0:
                # 选 gpu_loads 最小的 instance
                gpu_loads = result.gpu_loads
                if gpu_loads is not None and len(gpu_loads) > 0:
                    selected_idx = int(np.argmin(gpu_loads))
                    if 0 <= selected_idx < len(healthy):
                        return selected_idx
            # fallback
            with self.lock:
                idx = self.next_index % len(healthy)
                self.next_index += 1
            return idx
        except Exception:
            with self.lock:
                idx = self.next_index % len(healthy)
                self.next_index += 1
            return idx

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def record_selection(
        self,
        *,
        trace_id: str,
        request_payload: dict[str, Any],
        target: GatewayTarget,
        probe_rows: list[dict[str, Any]],
        response_status: int,
    ) -> None:
        task_type = str(
            request_payload.get("task_type")
            or request_payload.get("x_cgc_task_type")
            or "prefill"
        )
        model_name = str(request_payload.get("model") or "")
        placement = {
            "schema_version": "fusionroute.placement_decision_report.v1",
            "report_id": f"fusionroute_cloud_placement_{trace_id}",
            "task_type": task_type.upper(),
            "gate_domain": "cloud_prefill_pool",
            "primary_role": "DeepSeek-V4-Flash-Pool",
            "secondary_roles": [],
            "selected_locality": "cloud",
            "runtime_endpoint": f"cloud://{target.instance_id}",
            "decision_reason": [
                f"selection_policy={'lplb_load_balanced' if self._lplb_enabled else 'healthy_round_robin'}",
                f"trace_id={trace_id}",
                f"model={model_name or 'unset'}",
                f"host_label={target.host_label}",
                f"gateway_port={target.gateway_port}",
                f"lplb_enabled={self._lplb_enabled}",
            ],
            "policy_source": "fusionroute_cloud_orchestrator_runtime",
            "handoff_contract": "fusionroute_4instance_system",
            "status": "PASS" if 200 <= response_status < 300 else "FAIL",
            "selected_instance_id": target.instance_id,
            "selected_host_label": target.host_label,
            "selected_gateway_url": f"{target.base_url}/v1/chat/completions",
            "healthy_candidates": probe_rows,
            "evidence_path": str(self.placement_path),
        }
        self._write_json(self.placement_path, placement)

        # 更新 instance 负载历史（供 LPLB 下次决策使用）
        # 负载 = 累计请求数 * 估算权重（成功 +1，失败 +0.5）
        with self.lock:
            prev_load = self._instance_loads.get(target.instance_id, 0.0)
            self._instance_loads[target.instance_id] = prev_load + (
                1.0 if 200 <= response_status < 300 else 0.5
            )

        runtime_event = {
            "timestamp": _now_iso(),
            "trace_id": trace_id,
            "selected_instance_id": target.instance_id,
            "selected_host_label": target.host_label,
            "selected_gateway_url": f"{target.base_url}/v1/chat/completions",
            "response_status": response_status,
            "selection_policy": "lplb_load_balanced" if self._lplb_enabled else "healthy_round_robin",
            "instance_loads_snapshot": dict(self._instance_loads),
        }
        runtime_payload: dict[str, Any]
        if self.runtime_path.exists():
            try:
                runtime_payload = json.loads(
                    self.runtime_path.read_text(encoding="utf-8")
                )
            except Exception:
                runtime_payload = {}
        else:
            runtime_payload = {}
        recent = list(runtime_payload.get("recent_events") or [])
        recent.append(runtime_event)
        runtime_payload.update(
            {
                "schema_version": "cgc.fusionroute_cloud_runtime.v0.1",
                "status": "PASS",
                "updated_at": runtime_event["timestamp"],
                "selection_policy": "healthy_round_robin",
                "topology_path": str(self.topology_path),
                "placement_report_path": str(self.placement_path),
                "invocation_count": int(runtime_payload.get("invocation_count") or 0) + 1,
                "recent_events": recent[-10:],
                "active_targets": [
                    {
                        "instance_id": target_row.instance_id,
                        "host_label": target_row.host_label,
                        "base_url": target_row.base_url,
                        "requires_tunnel": target_row.requires_tunnel,
                    }
                    for target_row in self.targets
                ],
            }
        )
        self._write_json(self.runtime_path, runtime_payload)

    def _edge_first_generate_first_token(self, messages: list) -> tuple[str | None, float]:
        """本地小模型生成首 token（文本級，用於降低 TTFT 感知）。

        返回 (首token文本, 耗時ms)。失敗返回 (None, 耗時ms)。
        注意：本地小模型詞表與雲端 DSV4 不同，這是文本級首 token，
        僅用於降低 TTFT 感知，後續 token 由雲端 DSV4 生成。
        """
        t0 = time.monotonic()
        try:
            from app.servers.edge_first_proxy import _edge_generate_first_token
            text = _edge_generate_first_token(messages, max_tokens=1)
            return text, (time.monotonic() - t0) * 1000
        except Exception as exc:
            return None, (time.monotonic() - t0) * 1000

    def _edge_first_stream(
        self, payload: dict[str, Any], target, upstream_headers: dict,
        trace_id: str, probe_rows: list,
    ):
        """Edge-first streaming：本地首 token + 雲端接續。

        流程：
          1. 本地小模型生成首 token，立即 stream 返回（TTFT < 100ms）
          2. 雲端 sglang（DSV4 + MTP）接續生成後續 token
          3. 雲端首 chunk 的 content 被跳過（已被本地首 token 替代）
        """
        messages = payload.get("messages", [])
        url = f"{target.base_url}/v1/chat/completions"
        model = payload.get("model", "default")
        max_tokens = payload.get("max_tokens", 32)

        def _generate():
            t0 = time.monotonic()
            first_token_sent = False

            # 1. 本地生成首 token
            first_text, edge_ms = self._edge_first_generate_first_token(messages)

            if first_text:
                ttft_ms = (time.monotonic() - t0) * 1000
                first_chunk = {
                    "id": f"edge_{trace_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": first_text},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(first_chunk)}\n\n".encode()
                first_token_sent = True

            # 2. 雲端接續
            cloud_payload = {**payload, "stream": True}
            if first_token_sent:
                cloud_payload["max_tokens"] = max(1, max_tokens - 1)

            try:
                upstream = self.session.post(
                    url, json=cloud_payload, headers=upstream_headers,
                    timeout=(10, 240), stream=True,
                )
                self.record_selection(
                    trace_id=trace_id, request_payload=payload,
                    target=target, probe_rows=probe_rows,
                    response_status=upstream.status_code,
                )
                if upstream.status_code >= 400:
                    error_chunk = {"error": {"message": upstream.text[:500], "type": "cloud_error"}}
                    yield f"data: {json.dumps(error_chunk)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    upstream.close()
                    return

                cloud_first = True
                for chunk in upstream.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    # 雲端首 chunk 跳過 content（已被本地首 token 替代）
                    if cloud_first and first_token_sent:
                        cloud_first = False
                        try:
                            line = chunk.decode().strip()
                            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                                obj = json.loads(line[5:].strip())
                                choices = obj.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    if delta.get("content"):
                                        delta["content"] = ""
                                        yield f"data: {json.dumps(obj)}\n\n".encode()
                                        continue
                        except Exception:
                            pass
                        continue
                    cloud_first = False
                    yield chunk
                upstream.close()
            except Exception as exc:
                error_chunk = {"error": {"message": f"cloud error: {exc}", "type": "cloud_error"}}
                yield f"data: {json.dumps(error_chunk)}\n\n".encode()

            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={
                "x-cgc-selected-instance": target.instance_id,
                "x-edge-first": "enabled",
                "Cache-Control": "no-cache",
            },
        )

    def proxy_chat(self, payload: dict[str, Any], request: Request):
        trace_id = uuid.uuid4().hex[:12]
        target, probe_rows = self.select_target()
        upstream_headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length"}
        }
        upstream_headers["x-cgc-trace-id"] = trace_id
        stream = bool(payload.get("stream"))
        url = f"{target.base_url}/v1/chat/completions"

        # Edge-first: streaming 請求先注入本地首 token，再轉發雲端
        edge_first_enabled = str(
            os.environ.get("CGC_FUSIONROUTE_EDGE_FIRST", "0")
        ).strip() in {"1", "true", "yes"}

        if stream and edge_first_enabled:
            return self._edge_first_stream(payload, target, upstream_headers, trace_id, probe_rows)

        if stream:
            upstream = self.session.post(
                url,
                json=payload,
                headers=upstream_headers,
                timeout=(10, 240),
                stream=True,
            )
            self.record_selection(
                trace_id=trace_id,
                request_payload=payload,
                target=target,
                probe_rows=probe_rows,
                response_status=upstream.status_code,
            )
            if upstream.status_code >= 400:
                raise HTTPException(
                    status_code=upstream.status_code,
                    detail=upstream.text[:2000],
                )

            def _iter():
                try:
                    for chunk in upstream.iter_content(chunk_size=8192):
                        if chunk:
                            yield chunk
                finally:
                    upstream.close()

            return StreamingResponse(
                _iter(),
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "text/event-stream"),
                headers={"x-cgc-selected-instance": target.instance_id},
            )

        upstream = self.session.post(
            url,
            json=payload,
            headers=upstream_headers,
            timeout=(10, 240),
        )
        self.record_selection(
            trace_id=trace_id,
            request_payload=payload,
            target=target,
            probe_rows=probe_rows,
            response_status=upstream.status_code,
        )
        if "application/json" in str(upstream.headers.get("content-type") or ""):
            body = upstream.json()
        else:
            body = {"text": upstream.text}
        return JSONResponse(
            content=body,
            status_code=upstream.status_code,
            headers={"x-cgc-selected-instance": target.instance_id},
        )


ORCHESTRATOR = FusionRouteCloudOrchestrator()
app = FastAPI(title="FusionRoute Cloud Orchestrator")


@app.get("/health")
def health():
    healthy = ORCHESTRATOR._healthy_targets()
    return {
        "status": "ok" if healthy else "degraded",
        "selection_policy": "healthy_round_robin",
        "healthy_targets": [target.instance_id for target, _ in healthy],
        "target_count": len(ORCHESTRATOR.targets),
        "placement_report_path": str(ORCHESTRATOR.placement_path),
        "runtime_path": str(ORCHESTRATOR.runtime_path),
    }


@app.get("/v1/models")
def models():
    target, _ = ORCHESTRATOR.select_target()
    resp = ORCHESTRATOR.session.get(f"{target.base_url}/v1/models", timeout=(5, 20))
    if "application/json" in str(resp.headers.get("content-type") or ""):
        body = resp.json()
    else:
        body = {"text": resp.text}
    return JSONResponse(
        content=body,
        status_code=resp.status_code,
        headers={"x-cgc-selected-instance": target.instance_id},
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    return ORCHESTRATOR.proxy_chat(payload, request)


def start_server(host: str = "0.0.0.0", port: int = 50052):
    uvicorn.run(app, host=host, port=int(port), log_level="info")
