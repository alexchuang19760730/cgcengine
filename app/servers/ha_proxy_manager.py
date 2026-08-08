#!/usr/bin/env python3
"""Edge Proxy HA Manager — 双实例热备 + 自动故障切换.

部署两个 edge_first_proxy 实例 (primary + backup), 此脚本监控两者健康状态,
primary 挂了自动切到 backup, primary 恢复后切回.

架构:
  Client → HA Manager (port 30020) → Primary Proxy (port 30021) / Backup Proxy (port 30022)

  30021: primary edge_first_proxy  (优先路由)
  30022: backup edge_first_proxy   (故障切换)
  30020: HA Manager (此脚本, 反向代理 + 健康检查)

用法:
  # 1. 启动两个 proxy 实例
  python3 edge_first_proxy.py --port 30021 --cloud-url http://127.0.0.1:30050 --active-model gemma4 &
  python3 edge_first_proxy.py --port 30022 --cloud-url http://127.0.0.1:30050 --active-model gemma4 &

  # 2. 启动 HA Manager
  python3 ha_proxy_manager.py --port 30020 --primary 127.0.0.1:30021 --backup 127.0.0.1:30022

  # 3. 客户端连接 30020 (HA Manager), 自动路由到健康的 proxy
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum

import aiohttp
from aiohttp import web, ClientSession, ClientTimeout, TCPConnector


class ProxyState(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"


@dataclass
class ProxyInstance:
    """Proxy 实例状态."""
    name: str
    host: str
    port: int
    state: ProxyState = ProxyState.UNHEALTHY
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check: float = 0.0
    last_success: float = 0.0
    total_requests: int = 0
    total_errors: int = 0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_active(self) -> bool:
        return self.state == ProxyState.HEALTHY


class HAProxyManager:
    """HA Manager: 健康检查 + 故障切换 + 反向代理.

    - 每 N 秒检查 primary 和 backup 的 /health
    - primary 健康 → 路由到 primary
    - primary 不健康, backup 健康 → 路由到 backup
    - 两者都不健康 → 返回 503
    - primary 恢复后自动切回 (after M consecutive successes)
    """

    def __init__(
        self,
        primary: ProxyInstance,
        backup: ProxyInstance,
        health_interval: int = 5,
        failover_threshold: int = 3,
        recovery_threshold: int = 3,
    ):
        self.primary = primary
        self.backup = backup
        self.health_interval = health_interval
        self.failover_threshold = failover_threshold
        self.recovery_threshold = recovery_threshold
        self.active_target: ProxyInstance = primary  # 当前路由目标

        # 统计
        self.stats = {
            "total_requests": 0,
            "total_errors": 0,
            "failovers": 0,
            "recoveries": 0,
            "start_time": time.time(),
        }

    async def health_check_loop(self):
        """持续健康检查循环."""
        async with ClientSession(
            connector=TCPConnector(limit=10, keepalive_timeout=30)
        ) as session:
            while True:
                await self._check_one(session, self.primary)
                await self._check_one(session, self.backup)
                self._update_active_target()
                await asyncio.sleep(self.health_interval)

    async def _check_one(self, session: ClientSession, proxy: ProxyInstance):
        """检查单个 proxy 实例."""
        try:
            async with session.get(
                f"{proxy.url}/health",
                timeout=ClientTimeout(total=5),
            ) as resp:
                proxy.last_check = time.time()
                if resp.status == 200:
                    proxy.consecutive_failures = 0
                    proxy.consecutive_successes += 1
                    proxy.last_success = time.time()

                    if proxy.state != ProxyState.HEALTHY:
                        if proxy.consecutive_successes >= self.recovery_threshold:
                            old_state = proxy.state
                            proxy.state = ProxyState.HEALTHY
                            print(
                                f"[ha] {proxy.name} recovered: {old_state.value} → healthy "
                                f"(successes={proxy.consecutive_successes})",
                                file=sys.stderr,
                            )
                        else:
                            proxy.state = ProxyState.RECOVERING
                else:
                    proxy.consecutive_successes = 0
                    proxy.consecutive_failures += 1
                    if proxy.consecutive_failures >= self.failover_threshold:
                        if proxy.state != ProxyState.UNHEALTHY:
                            print(
                                f"[ha] {proxy.name} unhealthy: HTTP {resp.status} "
                                f"(failures={proxy.consecutive_failures})",
                                file=sys.stderr,
                            )
                        proxy.state = ProxyState.UNHEALTHY
        except Exception as e:
            proxy.consecutive_successes = 0
            proxy.consecutive_failures += 1
            proxy.last_check = time.time()
            if proxy.consecutive_failures >= self.failover_threshold:
                if proxy.state != ProxyState.UNHEALTHY:
                    print(
                        f"[ha] {proxy.name} unhealthy: {e} "
                        f"(failures={proxy.consecutive_failures})",
                        file=sys.stderr,
                    )
                proxy.state = ProxyState.UNHEALTHY

    def _update_active_target(self):
        """更新当前路由目标."""
        if self.active_target.state == ProxyState.HEALTHY:
            # primary 健康, 继续用 primary (或 backup 健康 but primary 恢复了)
            if self.active_target != self.primary and self.primary.state == ProxyState.HEALTHY:
                self.active_target = self.primary
                self.stats["recoveries"] += 1
                print(f"[ha] Switched back to primary ({self.primary.url})", file=sys.stderr)
        elif self.backup.state == ProxyState.HEALTHY:
            # primary 不健康, backup 健康 → 切到 backup
            if self.active_target != self.backup:
                self.active_target = self.backup
                self.stats["failovers"] += 1
                print(f"[ha] Failover to backup ({self.backup.url})", file=sys.stderr)
        elif self.backup.state == ProxyState.RECOVERING and self.active_target.state == ProxyState.UNHEALTHY:
            # primary 完全挂了, backup 在恢复中 → 先用 backup
            self.active_target = self.backup

    def get_active_target(self) -> ProxyInstance | None:
        """获取当前活跃的 proxy (或 None 如果都不健康)."""
        if self.active_target.state in (ProxyState.HEALTHY, ProxyState.RECOVERING):
            return self.active_target
        # 尝试另一个
        other = self.backup if self.active_target == self.primary else self.primary
        if other.state in (ProxyState.HEALTHY, ProxyState.RECOVERING):
            return other
        return None

    async def proxy_request(self, request: web.Request) -> web.StreamResponse:
        """反向代理请求到活跃的 proxy."""
        self.stats["total_requests"] += 1

        target = self.get_active_target()
        if target is None:
            self.stats["total_errors"] += 1
            return web.json_response(
                {"error": "All proxies unhealthy", "stats": self.get_status()},
                status=503,
            )

        target.total_requests += 1

        # 构建目标 URL
        target_url = f"{target.url}{request.path_qs}"

        try:
            async with ClientSession(
                connector=TCPConnector(limit=50, keepalive_timeout=60)
            ) as session:
                # 读取请求体
                body = await request.read()

                # 转发请求
                headers = dict(request.headers)
                headers.pop("Host", None)

                async with session.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    data=body if body else None,
                    timeout=ClientTimeout(total=120),
                    allow_redirects=False,
                ) as resp:
                    # 流式转发响应
                    response = web.StreamResponse(
                        status=resp.status,
                        headers={k: v for k, v in resp.headers.items() if k.lower() not in ("transfer-encoding", "content-encoding")},
                    )
                    await response.prepare(request)

                    async for chunk in resp.content.iter_any():
                        await response.write(chunk)

                    await response.write_eof()
                    return response

        except Exception as e:
            target.total_errors += 1
            self.stats["total_errors"] += 1
            print(f"[ha] Proxy error to {target.name}: {e}", file=sys.stderr)

            # 标记为不健康
            target.consecutive_failures += 1
            if target.consecutive_failures >= self.failover_threshold:
                target.state = ProxyState.UNHEALTHY

            return web.json_response(
                {"error": f"Proxy error: {e}", "target": target.name},
                status=502,
            )

    def get_status(self) -> dict:
        """获取 HA Manager 状态."""
        return {
            "active_target": self.active_target.name,
            "stats": self.stats,
            "instances": [
                {
                    "name": p.name,
                    "url": p.url,
                    "state": p.state.value,
                    "consecutive_failures": p.consecutive_failures,
                    "consecutive_successes": p.consecutive_successes,
                    "total_requests": p.total_requests,
                    "total_errors": p.total_errors,
                    "last_check_ago_s": round(time.time() - p.last_check, 1) if p.last_check else 0,
                    "last_success_ago_s": round(time.time() - p.last_success, 1) if p.last_success else 0,
                }
                for p in [self.primary, self.backup]
            ],
        }


async def main():
    parser = argparse.ArgumentParser(description="HA Proxy Manager for edge_first_proxy")
    parser.add_argument("--port", type=int, default=30020, help="HA Manager port")
    parser.add_argument("--primary", type=str, default="127.0.0.1:30021", help="Primary proxy address")
    parser.add_argument("--backup", type=str, default="127.0.0.1:30022", help="Backup proxy address")
    parser.add_argument("--health-interval", type=int, default=5, help="Health check interval (seconds)")
    parser.add_argument("--failover-threshold", type=int, default=3, help="Consecutive failures before failover")
    parser.add_argument("--recovery-threshold", type=int, default=3, help="Consecutive successes before recovery")
    args = parser.parse_args()

    # 解析地址
    p_host, p_port = args.primary.rsplit(":", 1)
    b_host, b_port = args.backup.rsplit(":", 1)

    primary = ProxyInstance(name="primary", host=p_host, port=int(p_port))
    backup = ProxyInstance(name="backup", host=b_host, port=int(b_port))

    manager = HAProxyManager(
        primary=primary,
        backup=backup,
        health_interval=args.health_interval,
        failover_threshold=args.failover_threshold,
        recovery_threshold=args.recovery_threshold,
    )

    # 创建 aiohttp 应用
    app = web.Application()

    # 所有请求都走 proxy_request
    async def handle_all(request: web.Request):
        return await manager.proxy_request(request)

    async def handle_health(request: web.Request):
        return web.json_response(manager.get_status())

    async def handle_stats(request: web.Request):
        return web.json_response(manager.get_status())

    app.router.add_get("/ha/health", handle_health)
    app.router.add_get("/ha/stats", handle_stats)
    app.router.add_route("*", "/{path:.*}", handle_all)

    # 启动健康检查循环
    asyncio.create_task(manager.health_check_loop())

    print(f"[ha] HA Manager starting on port {args.port}", file=sys.stderr)
    print(f"[ha] Primary: {primary.url}", file=sys.stderr)
    print(f"[ha] Backup: {backup.url}", file=sys.stderr)
    print(f"[ha] Health check every {args.health_interval}s, "
          f"failover after {args.failover_threshold} failures, "
          f"recovery after {args.recovery_threshold} successes", file=sys.stderr)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()

    print(f"[ha] HA Manager listening on 0.0.0.0:{args.port}", file=sys.stderr)

    # 保持运行
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
