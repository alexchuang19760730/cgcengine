#!/usr/bin/env python3
"""
Multi-instance load balancer for sglang.
Round-robin + health check, supports streaming and non-streaming.
Usage: python3 load_balancer.py --port 30010 --backends 30000,30001,30002,30003,30004,30005,30006,30007
"""
import argparse
import asyncio
import json
import time
import sys
import aiohttp
from aiohttp import web

BACKENDS = []
CURRENT = 0
HEALTH_STATUS = {}  # port -> bool
REQUEST_COUNT = {}  # port -> int
LAST_HEALTH_CHECK = 0

async def health_check_all(session):
    global HEALTH_STATUS, LAST_HEALTH_CHECK
    while True:
        for port in BACKENDS:
            try:
                async with session.get(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                    HEALTH_STATUS[port] = resp.status == 200
            except Exception:
                HEALTH_STATUS[port] = False
        LAST_HEALTH_CHECK = time.time()
        await asyncio.sleep(10)

def get_next_backend():
    global CURRENT
    healthy = [p for p in BACKENDS if HEALTH_STATUS.get(p, True)]
    if not healthy:
        # Fallback: try all
        healthy = BACKENDS
    # Round-robin among healthy
    for _ in range(len(healthy)):
        CURRENT = (CURRENT + 1) % len(BACKENDS)
        port = BACKENDS[CURRENT]
        if HEALTH_STATUS.get(port, True):
            REQUEST_COUNT[port] = REQUEST_COUNT.get(port, 0) + 1
            return port
    # All unhealthy, just return next
    port = BACKENDS[CURRENT]
    REQUEST_COUNT[port] = REQUEST_COUNT.get(port, 0) + 1
    return port

async def proxy_request(request):
    port = get_next_backend()
    backend_url = f"http://127.0.0.1:{port}{request.path_qs}"
    
    # Read body
    body = await request.read()
    
    # Forward headers
    headers = dict(request.headers)
    headers.pop('Host', None)
    headers.pop('Content-Length', None)
    
    is_stream = b'"stream":true' in body or b'"stream": true' in body
    
    if is_stream:
        # Streaming: proxy SSE
        response = web.StreamResponse(
            status=200,
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )
        await response.prepare(request)
        
        try:
            timeout = aiohttp.ClientTimeout(total=300, sock_read=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(backend_url, data=body, headers=headers) as resp:
                    async for chunk in resp.content.iter_any():
                        await response.write(chunk)
        except Exception as e:
            error_data = f"data: {json.dumps({'error': str(e)})}\n\n"
            await response.write(error_data.encode())
        finally:
            await response.write_eof()
        return response
    else:
        # Non-streaming: simple proxy
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(backend_url, data=body, headers=headers) as resp:
                content = await resp.read()
                return web.Response(
                    status=resp.status,
                    body=content,
                    content_type=resp.content_type,
                )

async def health_endpoint(request):
    healthy_count = sum(1 for p in BACKENDS if HEALTH_STATUS.get(p, True))
    return web.json_response({
        "status": "ok" if healthy_count > 0 else "degraded",
        "healthy_backends": healthy_count,
        "total_backends": len(BACKENDS),
        "backends": {str(p): {"healthy": HEALTH_STATUS.get(p, True), "requests": REQUEST_COUNT.get(p, 0)} for p in BACKENDS},
        "last_check": LAST_HEALTH_CHECK,
    })

async def stats_endpoint(request):
    return web.json_response({
        "backends": {str(p): {"healthy": HEALTH_STATUS.get(p, True), "requests": REQUEST_COUNT.get(p, 0)} for p in BACKENDS},
        "total_requests": sum(REQUEST_COUNT.values()),
    })

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=30010)
    parser.add_argument("--backends", type=str, required=True, help="Comma-separated port list")
    args = parser.parse_args()
    
    global BACKENDS
    BACKENDS = [int(p.strip()) for p in args.backends.split(",")]
    
    for p in BACKENDS:
        HEALTH_STATUS[p] = True
        REQUEST_COUNT[p] = 0
    
    app = web.Application()
    app.router.add_post('/v1/chat/completions', proxy_request)
    app.router.add_post('/v1/completions', proxy_request)
    app.router.add_post('/generate', proxy_request)
    app.router.add_get('/health', health_endpoint)
    app.router.add_get('/stats', stats_endpoint)
    
    # Start health checker
    timeout = aiohttp.ClientTimeout(total=5)
    session = aiohttp.ClientSession(timeout=timeout)
    asyncio.create_task(health_check_all(session))
    
    print(f"Load balancer on port {args.port}, backends: {BACKENDS}", flush=True)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', args.port)
    await site.start()
    
    # Keep running
    while True:
        await asyncio.sleep(3600)

if __name__ == '__main__':
    asyncio.run(main())
