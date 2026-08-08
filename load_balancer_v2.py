#!/usr/bin/env python3
"""
Enhanced Load Balancer for sglang with:
- Circuit breaker (auto-isolate failing backends)
- Latency-aware weighted routing
- Request retry on backend failure
- Auto-restart dead instances (watchdog integration)
- GPU health monitoring (memory/utilization)
- Prometheus /metrics endpoint
- Real-time /dashboard JSON
- Sliding-window stats

Usage:
  python3 load_balancer_v2.py \
    --port 30010 \
    --backends 30000,30001,30002,30003,30004,30005,30006,30007 \
    --gpu-map 0,1,2,3,4,5,6,7 \
    --model-path /data/models/gemma-4-26b-a4b-it \
    --draft-path /data/models/gemma-4-26b-a4b-it-assistant \
    --auto-restart \
    --health-interval 5 \
    --circuit-threshold 3 \
    --circuit-cooldown 30
"""
import argparse
import asyncio
import json
import time
import subprocess
import os
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lb")


# ============================================================
#  Circuit Breaker
# ============================================================
class CircuitState(Enum):
    CLOSED = "closed"        # normal, accepting traffic
    OPEN = "open"            # tripped, rejecting traffic
    HALF_OPEN = "half_open"  # testing, limited traffic


@dataclass
class BackendStats:
    port: int
    gpu_id: int = -1
    healthy: bool = True
    circuit_state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: float = 0
    last_success_time: float = 0
    circuit_opened_at: float = 0
    # Sliding window stats (last 100 requests)
    latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    ttfts: deque = field(default_factory=lambda: deque(maxlen=100))
    errors: deque = field(default_factory=lambda: deque(maxlen=50))
    # Counters
    total_requests: int = 0
    total_errors: int = 0
    total_tokens: int = 0
    # GPU stats
    gpu_mem_used: float = 0
    gpu_mem_total: float = 0
    gpu_util: float = 0
    gpu_temp: float = 0
    # Restart tracking
    restart_count: int = 0
    last_restart_time: float = 0
    is_restarting: bool = False

    @property
    def avg_latency(self):
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0

    @property
    def avg_ttft(self):
        return sum(self.ttfts) / len(self.ttfts) if self.ttfts else 0

    @property
    def error_rate(self):
        recent = len(self.latencies) + len(self.errors)
        if recent == 0:
            return 0
        return len(self.errors) / recent * 100

    @property
    def weight(self):
        """Compute routing weight based on latency and health."""
        if self.circuit_state == CircuitState.OPEN:
            return 0
        if self.is_restarting:
            return 0
        base = 100
        # Penalize high latency
        if self.avg_latency > 0:
            latency_penalty = min(self.avg_latency / 10, 80)  # max 80 penalty
            base -= latency_penalty
        # Penalize high error rate
        base -= self.error_rate * 0.5
        # Bonus for recent successes (half-open recovery)
        if self.circuit_state == CircuitState.HALF_OPEN:
            base = min(base, 10)  # limited traffic during recovery
        return max(base, 1) if self.healthy else 0


# ============================================================
#  Load Balancer
# ============================================================
class LoadBalancer:
    def __init__(self, args):
        self.port = args.port
        self.backend_ports = [int(p) for p in args.backends.split(",")]
        gpu_map = [int(g) for g in args.gpu_map.split(",")] if args.gpu_map else []
        self.model_path = args.model_path
        self.draft_path = args.draft_path
        self.python_path = args.python_path
        self.auto_restart = args.auto_restart
        self.health_interval = args.health_interval
        self.circuit_threshold = args.circuit_threshold
        self.circuit_cooldown = args.circuit_cooldown
        self.max_retries = args.max_retries
        self.log_rotation_size = args.log_rotation_size

        # Initialize backend stats
        self.backends = {}
        for i, p in enumerate(self.backend_ports):
            gpu_id = gpu_map[i] if i < len(gpu_map) else i
            self.backends[p] = BackendStats(port=p, gpu_id=gpu_id)

        # Global stats
        self.start_time = time.time()
        self.global_request_count = 0
        self.global_error_count = 0
        self.global_tokens = 0
        self.global_latencies = deque(maxlen=1000)
        self.global_ttfts = deque(maxlen=1000)

        # Weighted round-robin state
        self._rr_counter = 0

        # HTTP session (reused)
        self._session = None

    async def get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300, sock_read=120),
                connector=aiohttp.TCPConnector(limit=200, limit_per_host=50),
            )
        return self._session

    # --------------------------------------------------------
    #  Health Check + GPU Monitoring
    # --------------------------------------------------------
    async def health_check_loop(self):
        """Periodic health check + GPU monitoring for all backends."""
        log.info(f"Health checker started (interval={self.health_interval}s)")
        while True:
            tasks = [self._check_one_backend(port) for port in self.backend_ports]
            await asyncio.gather(*tasks, return_exceptions=True)
            # GPU monitoring (nvidia-smi is fast enough to call once)
            await self._update_gpu_stats()
            await asyncio.sleep(self.health_interval)

    async def _check_one_backend(self, port):
        bs = self.backends[port]
        session = await self.get_session()
        try:
            async with session.get(
                f"http://127.0.0.1:{port}/health",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                if resp.status == 200:
                    bs.healthy = True
                    bs.last_success_time = time.time()
                    # Recovery logic
                    if bs.circuit_state == CircuitState.OPEN:
                        if time.time() - bs.circuit_opened_at > self.circuit_cooldown:
                            bs.circuit_state = CircuitState.HALF_OPEN
                            bs.consecutive_successes = 0
                            log.info(f"[{port}] Circuit OPEN -> HALF_OPEN (cooldown elapsed)")
                else:
                    bs.healthy = False
                    self._record_failure(port, f"HTTP {resp.status}")
        except Exception as e:
            bs.healthy = False
            self._record_failure(port, str(e)[:100])

    def _record_failure(self, port, reason):
        bs = self.backends[port]
        bs.consecutive_failures += 1
        bs.consecutive_successes = 0
        bs.last_failure_time = time.time()
        bs.errors.append({"time": time.time(), "reason": reason})

        if bs.consecutive_failures >= self.circuit_threshold:
            if bs.circuit_state != CircuitState.OPEN:
                bs.circuit_state = CircuitState.OPEN
                bs.circuit_opened_at = time.time()
                log.warning(
                    f"[{port}] Circuit CLOSED -> OPEN "
                    f"(failures={bs.consecutive_failures}, reason={reason})"
                )
                # Trigger auto-restart if enabled
                if self.auto_restart and not bs.is_restarting:
                    asyncio.create_task(self._auto_restart_backend(port))

    def _record_success(self, port, latency_ms, ttft_ms=None, tokens=0):
        bs = self.backends[port]
        bs.consecutive_successes += 1
        bs.consecutive_failures = 0
        bs.last_success_time = time.time()
        bs.total_requests += 1
        bs.total_tokens += tokens
        bs.latencies.append(latency_ms)
        if ttft_ms is not None:
            bs.ttfts.append(ttft_ms)

        # Circuit recovery
        if bs.circuit_state == CircuitState.HALF_OPEN:
            if bs.consecutive_successes >= 3:
                bs.circuit_state = CircuitState.CLOSED
                log.info(f"[{port}] Circuit HALF_OPEN -> CLOSED (recovered)")

    async def _update_gpu_stats(self):
        """Query nvidia-smi for GPU stats."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            for line in stdout.decode().strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpu_id = int(parts[0])
                    mem_used = float(parts[1])
                    mem_total = float(parts[2])
                    util = float(parts[3])
                    temp = float(parts[4])
                    for bs in self.backends.values():
                        if bs.gpu_id == gpu_id:
                            bs.gpu_mem_used = mem_used
                            bs.gpu_mem_total = mem_total
                            bs.gpu_util = util
                            bs.gpu_temp = temp
        except Exception:
            pass

    # --------------------------------------------------------
    #  Auto-Restart (Watchdog)
    # --------------------------------------------------------
    async def _auto_restart_backend(self, port):
        """Restart a dead sglang instance."""
        bs = self.backends[port]
        if bs.is_restarting:
            return
        bs.is_restarting = True
        bs.restart_count += 1
        bs.last_restart_time = time.time()
        log.warning(f"[{port}] Auto-restarting (attempt #{bs.restart_count})...")

        # Kill any lingering process on that port
        try:
            await asyncio.create_subprocess_exec(
                "bash", "-c", f"lsof -ti:{port} | xargs -r kill -9 2>/dev/null",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.sleep(1)
        except Exception:
            pass

        # Launch sglang
        cmd = self._build_sglang_cmd(port, bs.gpu_id)
        log.info(f"[{port}] Launching: {' '.join(cmd[:6])}...")

        # Log rotation: truncate old log if too big
        log_file = f"/tmp/sglang_{port}.log"
        try:
            if os.path.exists(log_file) and os.path.getsize(log_file) > self.log_rotation_size:
                os.rename(log_file, f"{log_file}.old")
                log.info(f"[{port}] Rotated log file (was {os.path.getsize(log_file+'.old')/1e6:.1f}MB)")
        except Exception:
            pass

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=open(log_file, "w"),
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(bs.gpu_id)},
            )
            log.info(f"[{port}] sglang started (PID={proc.pid}), waiting for health...")
        except Exception as e:
            log.error(f"[{port}] Failed to start sglang: {e}")
            bs.is_restarting = False
            return

        # Wait for health (up to 3 minutes)
        session = await self.get_session()
        for attempt in range(90):  # 90 * 2s = 180s
            await asyncio.sleep(2)
            try:
                async with session.get(
                    f"http://127.0.0.1:{port}/health",
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as resp:
                    if resp.status == 200:
                        log.info(f"[{port}] Auto-restart SUCCESS (took {attempt*2}s)")
                        bs.is_restarting = False
                        bs.healthy = True
                        bs.circuit_state = CircuitState.CLOSED
                        bs.consecutive_failures = 0
                        return
            except Exception:
                pass

        log.error(f"[{port}] Auto-restart FAILED (timeout 180s)")
        bs.is_restarting = False

    def _build_sglang_cmd(self, port, gpu_id):
        return [
            self.python_path, "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--speculative-algorithm", "NEXTN",
            "--speculative-draft-model-path", self.draft_path,
            "--speculative-num-steps", "5",
            "--speculative-eagle-topk", "2",
            "--speculative-num-draft-tokens", "6",
            "--trust-remote-code",
            "--attention-backend", "triton",
            "--sampling-backend", "pytorch",
            "--quantization", "fp8",
            "--port", str(port),
            "--tp", "1",
            "--mem-fraction-static", "0.80",
            "--log-level", "info",
        ]

    # --------------------------------------------------------
    #  Backend Selection (Weighted Round-Robin)
    # --------------------------------------------------------
    def select_backend(self):
        """Select a backend using weighted round-robin."""
        candidates = [
            (port, bs) for port, bs in self.backends.items()
            if bs.weight > 0
        ]
        if not candidates:
            # All backends down — try any non-restarting one
            candidates = [
                (port, bs) for port, bs in self.backends.items()
                if not bs.is_restarting
            ]
            if not candidates:
                return None

        # Weighted selection
        total_weight = sum(bs.weight for _, bs in candidates)
        if total_weight == 0:
            port = candidates[0][0]
        else:
            r = (self._rr_counter % total_weight) + 1
            self._rr_counter += 1
            cumulative = 0
            port = candidates[0][0]
            for p, bs in candidates:
                cumulative += bs.weight
                if cumulative >= r:
                    port = p
                    break

        self.backends[port].total_requests += 1
        self.global_request_count += 1
        return port

    # --------------------------------------------------------
    #  Request Proxy with Retry
    # --------------------------------------------------------
    async def proxy_request(self, request):
        body = await request.read()
        headers = dict(request.headers)
        headers.pop("Host", None)
        headers.pop("Content-Length", None)
        path_qs = request.path_qs
        is_stream = b'"stream":true' in body or b'"stream": true' in body

        # Try up to max_retries backends
        last_error = None
        for attempt in range(self.max_retries):
            port = self.select_backend()
            if port is None:
                return web.json_response(
                    {"error": "No backends available"},
                    status=503,
                )

            backend_url = f"http://127.0.0.1:{port}{path_qs}"
            req_start = time.perf_counter()

            if is_stream:
                success, error = await self._proxy_stream(
                    request, backend_url, body, headers, port, req_start
                )
            else:
                success, error, response = await self._proxy_nonstream(
                    request, backend_url, body, headers, port, req_start
                )
                if success:
                    return response

            if not success:
                last_error = error
                self._record_failure(port, error)
                log.warning(
                    f"[{port}] Request failed (attempt {attempt+1}/{self.max_retries}): {error[:80]}"
                )
                continue
            else:
                return web.Response(status=200)

        # All retries exhausted
        self.global_error_count += 1
        return web.json_response(
            {"error": f"All retries exhausted: {last_error}"},
            status=502,
        )

    async def _proxy_stream(self, request, backend_url, body, headers, port, req_start):
        """Proxy streaming response with metrics."""
        bs = self.backends[port]
        session = await self.get_session()
        ttft = None
        token_count = 0

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

        try:
            await response.prepare(request)
            async with session.post(backend_url, data=body, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    err_msg = json.dumps({"error": f"Backend {resp.status}: {error_text[:200]}"})
                    await response.write(f"data: {err_msg}\n\n".encode())
                    await response.write_eof()
                    return False, f"Backend HTTP {resp.status}"

                async for chunk in resp.content.iter_any():
                    await response.write(chunk)
                    # Parse for metrics (best-effort, non-blocking)
                    try:
                        text = chunk.decode("utf-8", errors="ignore")
                        for line in text.split("\n"):
                            line = line.strip()
                            if line.startswith("data: ") and line != "data: [DONE]":
                                data = json.loads(line[6:])
                                if "choices" in data and data["choices"]:
                                    content = data["choices"][0].get("delta", {}).get("content", "")
                                    if content:
                                        if ttft is None:
                                            ttft = (time.perf_counter() - req_start) * 1000
                                        token_count += 1
                    except Exception:
                        pass

            await response.write_eof()
            latency = (time.perf_counter() - req_start) * 1000
            self._record_success(port, latency, ttft, token_count)
            self.global_latencies.append(latency)
            if ttft:
                self.global_ttfts.append(ttft)
            self.global_tokens += token_count
            return True, None

        except Exception as e:
            try:
                err_msg = json.dumps({"error": str(e)})
                await response.write(f"data: {err_msg}\n\n".encode())
                await response.write_eof()
            except Exception:
                pass
            return False, str(e)

    async def _proxy_nonstream(self, request, backend_url, body, headers, port, req_start):
        """Proxy non-streaming response with retry support."""
        session = await self.get_session()
        try:
            async with session.post(backend_url, data=body, headers=headers) as resp:
                content = await resp.read()
                latency = (time.perf_counter() - req_start) * 1000
                if resp.status != 200:
                    self._record_failure(port, f"HTTP {resp.status}")
                    return False, f"HTTP {resp.status}", None
                self._record_success(port, latency)
                self.global_latencies.append(latency)
                return True, None, web.Response(
                    status=resp.status,
                    body=content,
                    content_type=resp.content_type,
                )
        except Exception as e:
            return False, str(e), None

    # --------------------------------------------------------
    #  Endpoints: Health, Stats, Metrics, Dashboard
    # --------------------------------------------------------
    async def health_endpoint(self, request):
        healthy = [p for p, bs in self.backends.items() if bs.healthy and not bs.is_restarting]
        return web.json_response({
            "status": "ok" if len(healthy) > 0 else "critical",
            "healthy_backends": len(healthy),
            "total_backends": len(self.backends),
            "uptime_s": time.time() - self.start_time,
        })

    async def stats_endpoint(self, request):
        now = time.time()
        backends_info = {}
        for port, bs in self.backends.items():
            backends_info[str(port)] = {
                "gpu_id": bs.gpu_id,
                "healthy": bs.healthy,
                "circuit_state": bs.circuit_state.value,
                "is_restarting": bs.is_restarting,
                "weight": round(bs.weight, 1),
                "avg_latency_ms": round(bs.avg_latency, 1),
                "avg_ttft_ms": round(bs.avg_ttft, 1),
                "error_rate": round(bs.error_rate, 1),
                "total_requests": bs.total_requests,
                "total_errors": bs.total_errors,
                "total_tokens": bs.total_tokens,
                "restart_count": bs.restart_count,
                "gpu_mem_used_mb": round(bs.gpu_mem_used, 0),
                "gpu_util": round(bs.gpu_util, 0),
                "gpu_temp": round(bs.gpu_temp, 0),
                "consecutive_failures": bs.consecutive_failures,
                "last_success_ago_s": round(now - bs.last_success_time, 0) if bs.last_success_time else None,
            }

        # Global stats
        gl = list(self.global_latencies)
        gt = list(self.global_ttfts)
        gl_sorted = sorted(gl)
        gt_sorted = sorted(gt)

        def percentile(sorted_list, p):
            if not sorted_list:
                return 0
            idx = min(int(len(sorted_list) * p / 100), len(sorted_list) - 1)
            return sorted_list[idx]

        return web.json_response({
            "uptime_s": round(now - self.start_time, 0),
            "total_requests": self.global_request_count,
            "total_errors": self.global_error_count,
            "total_tokens": self.global_tokens,
            "global_avg_latency_ms": round(sum(gl) / len(gl), 1) if gl else 0,
            "global_avg_ttft_ms": round(sum(gt) / len(gt), 1) if gt else 0,
            "global_p50_latency_ms": round(percentile(gl_sorted, 50), 1),
            "global_p99_latency_ms": round(percentile(gl_sorted, 99), 1),
            "global_p50_ttft_ms": round(percentile(gt_sorted, 50), 1),
            "global_p99_ttft_ms": round(percentile(gt_sorted, 99), 1),
            "active_backends": sum(1 for bs in self.backends.values() if bs.weight > 0),
            "restarting_backends": sum(1 for bs in self.backends.values() if bs.is_restarting),
            "circuit_open_backends": sum(1 for bs in self.backends.values() if bs.circuit_state == CircuitState.OPEN),
            "backends": backends_info,
        })

    async def metrics_endpoint(self, request):
        """Prometheus-format metrics."""
        lines = []
        now = time.time()

        # Global metrics
        lines.append(f"# HELP lb_total_requests Total requests proxied")
        lines.append(f"# TYPE lb_total_requests counter")
        lines.append(f"lb_total_requests {self.global_request_count}")
        lines.append(f"lb_total_errors {self.global_error_count}")
        lines.append(f"lb_total_tokens {self.global_tokens}")
        lines.append(f"lb_uptime_seconds {now - self.start_time:.0f}")

        gl = list(self.global_latencies)
        if gl:
            lines.append(f"lb_avg_latency_ms {sum(gl)/len(gl):.1f}")
        gt = list(self.global_ttfts)
        if gt:
            lines.append(f"lb_avg_ttft_ms {sum(gt)/len(gt):.1f}")

        lines.append(f"\n# HELP lb_backend_healthy Backend healthy (1=yes, 0=no)")
        lines.append(f"# TYPE lb_backend_healthy gauge")
        lines.append(f"# HELP lb_backend_weight Backend routing weight")
        lines.append(f"# TYPE lb_backend_weight gauge")
        lines.append(f"# HELP lb_backend_requests Backend total requests")
        lines.append(f"# TYPE lb_backend_requests counter")
        lines.append(f"# HELP lb_backend_latency_ms Backend avg latency ms")
        lines.append(f"# TYPE lb_backend_latency_ms gauge")
        lines.append(f"# HELP lb_gpu_util GPU utilization percent")
        lines.append(f"# TYPE lb_gpu_util gauge")
        lines.append(f"# HELP lb_gpu_mem_used_mb GPU memory used MB")
        lines.append(f"# TYPE lb_gpu_mem_used_mb gauge")
        lines.append(f"# HELP lb_gpu_temp GPU temperature Celsius")
        lines.append(f"# TYPE lb_gpu_temp gauge")

        for port, bs in self.backends.items():
            labels = f'port="{port}",gpu="{bs.gpu_id}"'
            lines.append(f'lb_backend_healthy{{{labels}}} {1 if bs.healthy else 0}')
            lines.append(f'lb_backend_weight{{{labels}}} {bs.weight:.1f}')
            lines.append(f'lb_backend_requests{{{labels}}} {bs.total_requests}')
            lines.append(f'lb_backend_latency_ms{{{labels}}} {bs.avg_latency:.1f}')
            lines.append(f'lb_gpu_util{{{labels}}} {bs.gpu_util:.0f}')
            lines.append(f'lb_gpu_mem_used_mb{{{labels}}} {bs.gpu_mem_used:.0f}')
            lines.append(f'lb_gpu_temp{{{labels}}} {bs.gpu_temp:.0f}')

        return web.Response(
            text="\n".join(lines) + "\n",
            content_type="text/plain",
        )

    async def dashboard_endpoint(self, request):
        """Serve the monitoring dashboard HTML."""
        html = self._dashboard_html()
        return web.Response(text=html, content_type="text/html")

    def _dashboard_html(self):
        return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CGC Load Balancer Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0d1117; color: #c9d1d9; font-family: -apple-system, 'SF Mono', monospace; padding: 20px; }
  h1 { color: #58a6ff; font-size: 20px; margin-bottom: 16px; }
  h2 { color: #8b949e; font-size: 14px; margin: 20px 0 10px; text-transform: uppercase; letter-spacing: 1px; }
  .grid { display: grid; gap: 12px; }
  .stats-grid { grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
  .backend-grid { grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }
  .stat-card { text-align: center; }
  .stat-value { font-size: 28px; font-weight: bold; color: #58a6ff; }
  .stat-label { font-size: 11px; color: #8b949e; margin-top: 4px; text-transform: uppercase; }
  .stat-value.green { color: #3fb950; }
  .stat-value.yellow { color: #d29922; }
  .stat-value.red { color: #f85149; }
  .backend-card { position: relative; overflow: hidden; }
  .backend-card .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .backend-port { font-size: 16px; font-weight: bold; color: #58a6ff; }
  .badge { font-size: 10px; padding: 2px 8px; border-radius: 10px; font-weight: bold; }
  .badge.healthy { background: #1a4731; color: #3fb950; }
  .badge.unhealthy { background: #4a1e1e; color: #f85149; }
  .badge.open { background: #4a3a1e; color: #d29922; }
  .badge.half-open { background: #3a3a1e; color: #d29922; }
  .badge.restarting { background: #3a2a1e; color: #db6d28; }
  .metric-row { display: flex; justify-content: space-between; padding: 3px 0; font-size: 12px; }
  .metric-label { color: #8b949e; }
  .metric-value { color: #c9d1d9; font-weight: bold; }
  .gpu-bar { height: 4px; background: #21262d; border-radius: 2px; margin-top: 4px; overflow: hidden; }
  .gpu-bar-fill { height: 100%; border-radius: 2px; transition: width 0.5s; }
  .gpu-bar-fill.low { background: #3fb950; }
  .gpu-bar-fill.mid { background: #d29922; }
  .gpu-bar-fill.high { background: #f85149; }
  .chart-container { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; margin-top: 12px; }
  canvas { width: 100%; height: 120px; }
  .refresh-indicator { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #3fb950; margin-left: 8px; animation: pulse 2s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
  .timestamp { color: #8b949e; font-size: 11px; }
</style>
</head>
<body>
<h1>CGC Load Balancer Dashboard <span class="refresh-indicator"></span> <span class="timestamp" id="last-update"></span></h1>

<h2>Global Stats</h2>
<div class="grid stats-grid" id="global-stats"></div>

<h2>Backend Status</h2>
<div class="grid backend-grid" id="backend-grid"></div>

<h2>Latency (last 1000 requests)</h2>
<div class="chart-container">
  <canvas id="latency-chart"></canvas>
</div>

<h2>TTFT (last 1000 requests)</h2>
<div class="chart-container">
  <canvas id="ttft-chart"></canvas>
</div>

<script>
const API = '';
let latencyHistory = [];
let ttftHistory = [];
const MAX_HISTORY = 60;

async function fetchStats() {
  try {
    const resp = await fetch(API + '/stats');
    const data = await resp.json();
    updateGlobal(data);
    updateBackends(data);
    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();

    // Update chart history
    latencyHistory.push(data.global_avg_latency_ms || 0);
    ttftHistory.push(data.global_avg_ttft_ms || 0);
    if (latencyHistory.length > MAX_HISTORY) latencyHistory.shift();
    if (ttftHistory.length > MAX_HISTORY) ttftHistory.shift();
    drawChart('latency-chart', latencyHistory, '#58a6ff', 'Latency (ms)');
    drawChart('ttft-chart', ttftHistory, '#3fb950', 'TTFT (ms)');
  } catch (e) {
    console.error('Fetch error:', e);
  }
}

function updateGlobal(data) {
  const sr = data.total_requests > 0 ? ((1 - data.total_errors / data.total_requests) * 100).toFixed(1) : 100;
  const uptime = formatUptime(data.uptime_s);
  const cards = [
    {label: 'Uptime', value: uptime, class: ''},
    {label: 'Total Requests', value: data.total_requests.toLocaleString(), class: ''},
    {label: 'Success Rate', value: sr + '%', class: sr > 95 ? 'green' : sr > 80 ? 'yellow' : 'red'},
    {label: 'Total Tokens', value: data.total_tokens.toLocaleString(), class: ''},
    {label: 'Avg Latency', value: data.global_avg_latency_ms.toFixed(0) + 'ms', class: ''},
    {label: 'P99 Latency', value: data.global_p99_latency_ms.toFixed(0) + 'ms', class: data.global_p99_latency_ms < 1000 ? 'green' : 'red'},
    {label: 'Avg TTFT', value: data.global_avg_ttft_ms.toFixed(0) + 'ms', class: data.global_avg_ttft_ms < 100 ? 'green' : 'yellow'},
    {label: 'P99 TTFT', value: data.global_p99_ttft_ms.toFixed(0) + 'ms', class: data.global_p99_ttft_ms < 200 ? 'green' : 'red'},
    {label: 'Active Backends', value: data.active_backends + '/' + Object.keys(data.backends).length, class: 'green'},
    {label: 'Circuit Open', value: data.circuit_open_backends, class: data.circuit_open_backends == 0 ? 'green' : 'red'},
  ];
  document.getElementById('global-stats').innerHTML = cards.map(c => `
    <div class="card stat-card">
      <div class="stat-value ${c.class}">${c.value}</div>
      <div class="stat-label">${c.label}</div>
    </div>
  `).join('');
}

function updateBackends(data) {
  const entries = Object.entries(data.backends).sort((a,b) => parseInt(a[0]) - parseInt(b[0]));
  document.getElementById('backend-grid').innerHTML = entries.map(([port, b]) => {
    const stateClass = b.is_restarting ? 'restarting' : b.circuit_state;
    const stateLabel = b.is_restarting ? 'RESTARTING' : b.circuit_state.toUpperCase();
    const healthClass = b.healthy ? 'healthy' : 'unhealthy';
    const memPct = b.gpu_mem_used_mb > 0 ? Math.min(b.gpu_mem_used_mb / 48000 * 100, 100) : 0;
    const memClass = memPct < 70 ? 'low' : memPct < 90 ? 'mid' : 'high';
    const utilClass = b.gpu_util < 60 ? 'low' : b.gpu_util < 85 ? 'mid' : 'high';
    return `
      <div class="card backend-card">
        <div class="header">
          <span class="backend-port">:${port} <span style="color:#8b949e;font-size:12px">GPU ${b.gpu_id}</span></span>
          <span class="badge ${healthClass}">${b.healthy ? 'HEALTHY' : 'DOWN'}</span>
          <span class="badge ${stateClass}">${stateLabel}</span>
        </div>
        <div class="metric-row"><span class="metric-label">Requests</span><span class="metric-value">${b.total_requests.toLocaleString()}</span></div>
        <div class="metric-row"><span class="metric-label">Avg Latency</span><span class="metric-value">${b.avg_latency_ms.toFixed(0)}ms</span></div>
        <div class="metric-row"><span class="metric-label">Avg TTFT</span><span class="metric-value">${b.avg_ttft_ms.toFixed(0)}ms</span></div>
        <div class="metric-row"><span class="metric-label">Error Rate</span><span class="metric-value">${b.error_rate.toFixed(1)}%</span></div>
        <div class="metric-row"><span class="metric-label">Weight</span><span class="metric-value">${b.weight}</span></div>
        <div class="metric-row"><span class="metric-label">Restarts</span><span class="metric-value">${b.restart_count}</span></div>
        <div class="metric-row"><span class="metric-label">GPU Mem</span><span class="metric-value">${b.gpu_mem_used_mb.toFixed(0)}MB</span></div>
        <div class="gpu-bar"><div class="gpu-bar-fill ${memClass}" style="width:${memPct}%"></div></div>
        <div class="metric-row" style="margin-top:4px"><span class="metric-label">GPU Util</span><span class="metric-value">${b.gpu_util.toFixed(0)}%</span></div>
        <div class="gpu-bar"><div class="gpu-bar-fill ${utilClass}" style="width:${b.gpu_util}%"></div></div>
        <div class="metric-row" style="margin-top:4px"><span class="metric-label">GPU Temp</span><span class="metric-value">${b.gpu_temp.toFixed(0)}°C</span></div>
      </div>
    `;
  }).join('');
}

function drawChart(canvasId, data, color, label) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.offsetWidth * 2;
  const h = canvas.height = 240;
  ctx.clearRect(0, 0, w, h);

  if (data.length < 2) return;
  const max = Math.max(...data, 1) * 1.2;
  const step = w / (data.length - 1);

  // Grid
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (h / 4) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // Line
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = i * step;
    const y = h - (v / max) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Fill
  ctx.lineTo(w, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = color + '20';
  ctx.fill();

  // Label
  ctx.fillStyle = '#8b949e';
  ctx.font = '24px monospace';
  ctx.fillText(label + ': ' + (data[data.length-1] || 0).toFixed(0) + 'ms (max ' + max.toFixed(0) + ')', 10, 30);
}

function formatUptime(s) {
  if (!s) return '0s';
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  if (h > 0) return h + 'h ' + m + 'm';
  if (m > 0) return m + 'm ' + sec + 's';
  return sec + 's';
}

fetchStats();
setInterval(fetchStats, 3000);
</script>
</body>
</html>"""

    # --------------------------------------------------------
    #  Startup
    # --------------------------------------------------------
    async def run(self):
        app = web.Application(client_max_size=10 * 1024 * 1024)
        app.router.add_post("/v1/chat/completions", self.proxy_request)
        app.router.add_post("/v1/completions", self.proxy_request)
        app.router.add_post("/generate", self.proxy_request)
        app.router.add_get("/health", self.health_endpoint)
        app.router.add_get("/stats", self.stats_endpoint)
        app.router.add_get("/metrics", self.metrics_endpoint)
        app.router.add_get("/dashboard", self.dashboard_endpoint)
        app.router.add_get("/", self.dashboard_endpoint)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()

        log.info(f"Load Balancer v2 on port {self.port}")
        log.info(f"Backends: {self.backend_ports}")
        log.info(f"Auto-restart: {'ENABLED' if self.auto_restart else 'DISABLED'}")
        log.info(f"Circuit breaker: threshold={self.circuit_threshold}, cooldown={self.circuit_cooldown}s")
        log.info(f"Max retries per request: {self.max_retries}")
        log.info(f"Dashboard: http://0.0.0.0:{self.port}/dashboard")
        log.info(f"Metrics: http://0.0.0.0:{self.port}/metrics")

        # Start health checker
        asyncio.create_task(self.health_check_loop())

        # Keep running
        while True:
            await asyncio.sleep(3600)


def main():
    parser = argparse.ArgumentParser(description="Enhanced Load Balancer for sglang")
    parser.add_argument("--port", type=int, default=30010)
    parser.add_argument("--backends", type=str, required=True, help="Comma-separated backend ports")
    parser.add_argument("--gpu-map", type=str, default="", help="Comma-separated GPU IDs (parallel to --backends)")
    parser.add_argument("--model-path", type=str, default="/data/models/gemma-4-26b-a4b-it")
    parser.add_argument("--draft-path", type=str, default="/data/models/gemma-4-26b-a4b-it-assistant")
    parser.add_argument("--python-path", type=str, default="python3", help="Python executable path for sglang auto-restart")
    parser.add_argument("--auto-restart", action="store_true", help="Enable auto-restart of dead instances")
    parser.add_argument("--health-interval", type=int, default=5, help="Health check interval (seconds)")
    parser.add_argument("--circuit-threshold", type=int, default=3, help="Consecutive failures to open circuit")
    parser.add_argument("--circuit-cooldown", type=int, default=30, help="Seconds before half-open retry")
    parser.add_argument("--max-retries", type=int, default=3, help="Max backend retries per request")
    parser.add_argument("--log-rotation-size", type=int, default=500_000_000, help="Rotate sglang log at this size (bytes)")
    args = parser.parse_args()

    lb = LoadBalancer(args)
    asyncio.run(lb.run())


if __name__ == "__main__":
    main()
