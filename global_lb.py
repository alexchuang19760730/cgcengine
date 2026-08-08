#!/usr/bin/env python3
"""
Global Load Balancer — Cross-host routing for CGC cluster.
Distributes traffic across multiple per-host LBs (Host1:30010, Host2:30010).
Queries each host's /stats for active backend count → weighted routing.
Includes health check, failover, circuit breaker, dashboard.

Usage:
  python3 global_lb.py --port 30050 \
    --backends 39.106.118.206:30010,47.95.250.55:30010 \
    --names host1,host2
"""
import argparse
import asyncio
import json
import time
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
log = logging.getLogger("glb")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class HostBackend:
    name: str
    host: str
    port: int
    healthy: bool = True
    circuit_state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    circuit_opened_at: float = 0
    # Stats from per-host LB
    active_backends: int = 0
    total_backends: int = 0
    total_requests: int = 0
    avg_ttft_ms: float = 0
    avg_latency_ms: float = 0
    # Local tracking
    latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    total_proxied: int = 0
    total_errors: int = 0
    last_stats_time: float = 0

    @property
    def url(self):
        return f"http://{self.host}:{self.port}"

    @property
    def weight(self):
        if self.circuit_state == CircuitState.OPEN:
            return 0
        if not self.healthy:
            return 0
        # Weight by active backend count (more GPUs = more traffic)
        base = max(self.active_backends, 1) * 10
        # Penalty for high latency
        if self.avg_latency_ms > 0:
            base -= min(self.avg_latency_ms / 10, 50)
        if self.circuit_state == CircuitState.HALF_OPEN:
            base = min(base, 5)
        return max(base, 1)


class GlobalLoadBalancer:
    def __init__(self, args):
        self.port = args.port
        hosts = args.backends.split(",")
        names = args.names.split(",") if args.names else [f"host{i}" for i in range(len(hosts))]
        self.circuit_threshold = args.circuit_threshold
        self.circuit_cooldown = args.circuit_cooldown
        self.max_retries = args.max_retries

        self.backends = []
        for i, h in enumerate(hosts):
            host, port = h.strip().split(":")
            name = names[i] if i < len(names) else f"host{i}"
            self.backends.append(HostBackend(name=name, host=host, port=int(port)))

        self.start_time = time.time()
        self.global_count = 0
        self.global_errors = 0
        self.global_latencies = deque(maxlen=1000)
        self.global_ttfts = deque(maxlen=1000)
        self._rr = 0
        self._session = None

    async def get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300, sock_read=120),
                connector=aiohttp.TCPConnector(limit=200, limit_per_host=100),
            )
        return self._session

    async def health_check_loop(self):
        """Health check + stats polling for all host backends."""
        log.info(f"Health checker started (interval=5s)")
        while True:
            tasks = [self._check_host(b) for b in self.backends]
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(5)

    async def _check_host(self, backend):
        session = await self.get_session()
        try:
            # Health check
            async with session.get(
                f"{backend.url}/health",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                if resp.status == 200:
                    backend.healthy = True
                    if backend.circuit_state == CircuitState.OPEN:
                        if time.time() - backend.circuit_opened_at > self.circuit_cooldown:
                            backend.circuit_state = CircuitState.HALF_OPEN
                            backend.consecutive_successes = 0
                            log.info(f"[{backend.name}] Circuit OPEN -> HALF_OPEN")
                else:
                    backend.healthy = False
                    self._record_failure(backend, f"HTTP {resp.status}")

            # Stats polling (for weighted routing)
            try:
                async with session.get(
                    f"{backend.url}/stats",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        backend.active_backends = data.get("active_backends", 0)
                        backend.total_backends = data.get("total_backends", 0)
                        backend.total_requests = data.get("total_requests", 0)
                        backend.avg_ttft_ms = data.get("global_avg_ttft_ms", 0)
                        backend.avg_latency_ms = data.get("global_avg_latency_ms", 0)
                        backend.last_stats_time = time.time()
            except Exception:
                pass

        except Exception as e:
            backend.healthy = False
            self._record_failure(backend, str(e)[:100])

    def _record_failure(self, backend, reason):
        backend.consecutive_failures += 1
        backend.consecutive_successes = 0
        if backend.consecutive_failures >= self.circuit_threshold:
            if backend.circuit_state != CircuitState.OPEN:
                backend.circuit_state = CircuitState.OPEN
                backend.circuit_opened_at = time.time()
                log.warning(
                    f"[{backend.name}] Circuit OPEN (failures={backend.consecutive_failures}): {reason}"
                )

    def _record_success(self, backend, latency_ms):
        backend.consecutive_successes += 1
        backend.consecutive_failures = 0
        backend.total_proxied += 1
        backend.latencies.append(latency_ms)
        if backend.circuit_state == CircuitState.HALF_OPEN:
            if backend.consecutive_successes >= 3:
                backend.circuit_state = CircuitState.CLOSED
                log.info(f"[{backend.name}] Circuit HALF_OPEN -> CLOSED (recovered)")

    def select_backend(self):
        candidates = [b for b in self.backends if b.weight > 0]
        if not candidates:
            candidates = self.backends
            if not candidates:
                return None
        total = sum(b.weight for b in candidates)
        if total == 0:
            return candidates[0]
        r = (self._rr % total) + 1
        self._rr += 1
        cum = 0
        for b in candidates:
            cum += b.weight
            if cum >= r:
                return b
        return candidates[-1]

    async def proxy_request(self, request):
        body = await request.read()
        headers = dict(request.headers)
        headers.pop("Host", None)
        headers.pop("Content-Length", None)
        path_qs = request.path_qs
        is_stream = b'"stream":true' in body or b'"stream": true' in body

        for attempt in range(self.max_retries):
            backend = self.select_backend()
            if backend is None:
                return web.json_response({"error": "No backends available"}, status=503)

            req_start = time.perf_counter()
            if is_stream:
                ok, err = await self._proxy_stream(request, backend, body, headers, path_qs, req_start)
            else:
                ok, err, resp = await self._proxy_nonstream(request, backend, body, headers, path_qs, req_start)
                if ok:
                    return resp
            if not ok:
                self._record_failure(backend, err)
                self.global_errors += 1
                continue
            return web.Response(status=200)

        return web.json_response({"error": "All retries exhausted"}, status=502)

    async def _proxy_stream(self, request, backend, body, headers, path_qs, req_start):
        session = await self.get_session()
        url = f"{backend.url}{path_qs}"
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
        try:
            await response.prepare(request)
            async with session.post(url, data=body, headers=headers) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    err = json.dumps({"error": f"{backend.name} HTTP {resp.status}: {txt[:200]}"})
                    await response.write(f"data: {err}\n\n".encode())
                    await response.write_eof()
                    return False, f"HTTP {resp.status}"
                async for chunk in resp.content.iter_any():
                    await response.write(chunk)
            await response.write_eof()
            latency = (time.perf_counter() - req_start) * 1000
            self._record_success(backend, latency)
            self.global_latencies.append(latency)
            self.global_count += 1
            return True, None
        except Exception as e:
            try:
                err = json.dumps({"error": str(e)})
                await response.write(f"data: {err}\n\n".encode())
                await response.write_eof()
            except Exception:
                pass
            return False, str(e)

    async def _proxy_nonstream(self, request, backend, body, headers, path_qs, req_start):
        session = await self.get_session()
        url = f"{backend.url}{path_qs}"
        try:
            async with session.post(url, data=body, headers=headers) as resp:
                content = await resp.read()
                latency = (time.perf_counter() - req_start) * 1000
                if resp.status != 200:
                    self._record_failure(backend, f"HTTP {resp.status}")
                    return False, f"HTTP {resp.status}", None
                self._record_success(backend, latency)
                self.global_latencies.append(latency)
                self.global_count += 1
                return True, None, web.Response(status=resp.status, body=content, content_type=resp.content_type)
        except Exception as e:
            return False, str(e), None

    async def health_endpoint(self, request):
        healthy = sum(1 for b in self.backends if b.healthy)
        return web.json_response({
            "status": "ok" if healthy > 0 else "critical",
            "healthy_hosts": healthy,
            "total_hosts": len(self.backends),
            "total_active_backends": sum(b.active_backends for b in self.backends),
            "uptime_s": time.time() - self.start_time,
        })

    async def stats_endpoint(self, request):
        hosts = []
        for b in self.backends:
            hosts.append({
                "name": b.name,
                "host": b.host,
                "port": b.port,
                "healthy": b.healthy,
                "circuit_state": b.circuit_state.value,
                "weight": round(b.weight, 1),
                "active_backends": b.active_backends,
                "total_backends": b.total_backends,
                "proxied_requests": b.total_proxied,
                "errors": b.total_errors,
                "avg_latency_ms": round(b.avg_latency_ms, 1) if b.latencies else 0,
                "host_avg_ttft_ms": round(b.avg_ttft_ms, 1),
                "host_avg_latency_ms": round(b.avg_latency_ms, 1),
            })
        gl = list(self.global_latencies)
        return web.json_response({
            "uptime_s": round(time.time() - self.start_time, 0),
            "total_requests": self.global_count,
            "total_errors": self.global_errors,
            "avg_latency_ms": round(sum(gl) / len(gl), 1) if gl else 0,
            "total_active_backends": sum(b.active_backends for b in self.backends),
            "hosts": hosts,
        })

    async def metrics_endpoint(self, request):
        lines = [
            f"# HELP glb_total_requests Total requests proxied",
            f"# TYPE glb_total_requests counter",
            f"glb_total_requests {self.global_count}",
            f"glb_total_errors {self.global_errors}",
            f"glb_total_active_backends {sum(b.active_backends for b in self.backends)}",
        ]
        for b in self.backends:
            labels = f'host="{b.name}",addr="{b.host}:{b.port}"'
            lines.append(f'glb_host_healthy{{{labels}}} {1 if b.healthy else 0}')
            lines.append(f'glb_host_weight{{{labels}}} {b.weight:.1f}')
            lines.append(f'glb_host_active_backends{{{labels}}} {b.active_backends}')
            lines.append(f'glb_host_proxied{{{labels}}} {b.total_proxied}')
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")

    async def dashboard_endpoint(self, request):
        return web.Response(text=self._dashboard(), content_type="text/html")

    def _dashboard(self):
        return """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>CGC Global LB</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:monospace;padding:20px}
h1{color:#58a6ff;font-size:20px;margin-bottom:16px}
h2{color:#8b949e;font-size:12px;margin:20px 0 8px;text-transform:uppercase}
.grid{display:grid;gap:12px}
.stat-grid{grid-template-columns:repeat(auto-fill,minmax(160px,1fr))}
.host-grid{grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}
.stat-card{text-align:center}
.stat-value{font-size:24px;font-weight:bold;color:#58a6ff}
.stat-label{font-size:10px;color:#8b949e;margin-top:4px;text-transform:uppercase}
.green{color:#3fb950}.yellow{color:#d29922}.red{color:#f85149}
.badge{font-size:10px;padding:2px 8px;border-radius:10px;font-weight:bold}
.badge.healthy{background:#1a4731;color:#3fb950}
.badge.unhealthy{background:#4a1e1e;color:#f85149}
.badge.open{background:#4a3a1e;color:#d29922}
.mr{display:flex;justify-content:space-between;padding:3px 0;font-size:12px}
.ml{color:#8b949e}.mv{color:#c9d1d9;font-weight:bold}
.ind{display:inline-block;width:8px;height:8px;border-radius:50%;background:#3fb950;margin-left:8px;animation:p 2s infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.3}}
</style></head><body>
<h1>CGC Global Load Balancer <span class="ind"></span> <span style="color:#8b949e;font-size:11px" id="ts"></span></h1>
<h2>Cluster Overview</h2>
<div class="grid stat-grid" id="stats"></div>
<h2>Host Backends</h2>
<div class="grid host-grid" id="hosts"></div>
<script>
async function f(){try{const r=await fetch('/stats');const d=await r.json();
document.getElementById('ts').textContent=new Date().toLocaleTimeString();
const cards=[
 {l:'Uptime',v:fmt(d.uptime_s),c:''},
 {l:'Total Requests',v:d.total_requests.toLocaleString(),c:''},
 {l:'Errors',v:d.total_errors,c:d.total_errors==0?'green':'red'},
 {l:'Avg Latency',v:d.avg_latency_ms.toFixed(0)+'ms',c:''},
 {l:'Active GPUs',v:d.total_active_backends,c:'green'},
 {l:'Hosts',v:d.hosts.filter(h=>h.healthy).length+'/'+d.hosts.length,c:'green'},
];
document.getElementById('stats').innerHTML=cards.map(c=>`<div class="card stat-card"><div class="stat-value ${c.c}">${c.v}</div><div class="stat-label">${c.l}</div></div>`).join('');
document.getElementById('hosts').innerHTML=d.hosts.map(h=>`
<div class="card">
<div style="display:flex;justify-content:space-between;margin-bottom:8px">
<span style="font-size:16px;font-weight:bold;color:#58a6ff">${h.name}</span>
<span class="badge ${h.healthy?'healthy':'unhealthy'}">${h.healthy?'HEALTHY':'DOWN'}</span>
<span class="badge ${h.circuit_state=='closed'?'healthy':h.circuit_state=='open'?'open':'unhealthy'}">${h.circuit_state.toUpperCase()}</span>
</div>
<div class="mr"><span class="ml">Address</span><span class="mv">${h.host}:${h.port}</span></div>
<div class="mr"><span class="ml">Active GPUs</span><span class="mv">${h.active_backends}/${h.total_backends}</span></div>
<div class="mr"><span class="ml">Weight</span><span class="mv">${h.weight}</span></div>
<div class="mr"><span class="ml">Proxied</span><span class="mv">${h.proxied_requests.toLocaleString()}</span></div>
<div class="mr"><span class="ml">Host TTFT</span><span class="mv">${h.host_avg_ttft_ms.toFixed(0)}ms</span></div>
<div class="mr"><span class="ml">Host Latency</span><span class="mv">${h.host_avg_latency_ms.toFixed(0)}ms</span></div>
</div>`).join('');
}catch(e){console.error(e)}}
function fmt(s){if(!s)return'0s';const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=Math.floor(s%60);return h>0?h+'h '+m+'m':m>0?m+'m '+sec+'s':sec+'s'}
f();setInterval(f,3000);
</script></body></html>"""

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

        log.info(f"Global LB on port {self.port}")
        log.info(f"Host backends: {[(b.name, b.url) for b in self.backends]}")
        log.info(f"Dashboard: http://0.0.0.0:{self.port}/dashboard")

        asyncio.create_task(self.health_check_loop())
        while True:
            await asyncio.sleep(3600)


def main():
    parser = argparse.ArgumentParser(description="Global Cross-Host Load Balancer")
    parser.add_argument("--port", type=int, default=30050)
    parser.add_argument("--backends", type=str, required=True, help="Comma-separated host:port pairs")
    parser.add_argument("--names", type=str, default="", help="Comma-separated host names")
    parser.add_argument("--circuit-threshold", type=int, default=3)
    parser.add_argument("--circuit-cooldown", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()
    glb = GlobalLoadBalancer(args)
    asyncio.run(glb.run())


if __name__ == "__main__":
    main()
