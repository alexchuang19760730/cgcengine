#!/usr/bin/env python3
"""
16-card cross-host concurrency stress test.
Tests edge_first_proxy -> Global LB -> Host1(8 GPUs) + Host2(8 GPUs).
"""
import asyncio
import aiohttp
import json
import time
import sys
import argparse
from collections import defaultdict

PROMPTS = [
    "Write a Python function to reverse a string",
    "Explain how quicksort works with an example",
    "Write a SQL query to find duplicate records",
    "Implement a binary search tree in Python",
    "Write a function to check if a number is prime",
    "Explain the difference between TCP and UDP",
    "Write a REST API endpoint using Flask",
    "Implement merge sort algorithm in Python",
    "Write a function to find the longest common prefix",
    "Explain Big O notation with examples",
    "Write a Python decorator for timing functions",
    "Implement a hash table from scratch",
    "Write a function to validate an email address",
    "Explain how garbage collection works in Python",
    "Write a generator function for Fibonacci numbers",
    "Implement a simple LRU cache in Python",
    "Write a function to detect a cycle in a linked list",
    "Explain the CAP theorem in distributed systems",
    "Write a Python script to parse CSV files",
    "Implement breadth-first search in Python",
]

async def send_request(session, url, prompt, req_id, results):
    payload = {
        "model": "gemma4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0,
    }
    start = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            content = await resp.read()
            elapsed = time.perf_counter() - start
            if resp.status == 200:
                data = json.loads(content)
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                ttft_approx = elapsed  # proxy for TTFT (non-stream)
                tok_s = completion_tokens / elapsed if elapsed > 0 else 0
                results[req_id] = {
                    "success": True,
                    "latency_ms": elapsed * 1000,
                    "tokens": completion_tokens,
                    "tok_s": tok_s,
                    "status": resp.status,
                }
            else:
                results[req_id] = {
                    "success": False,
                    "latency_ms": elapsed * 1000,
                    "tokens": 0,
                    "tok_s": 0,
                    "status": resp.status,
                    "error": content.decode()[:200],
                }
    except Exception as e:
        elapsed = time.perf_counter() - start
        results[req_id] = {
            "success": False,
            "latency_ms": elapsed * 1000,
            "tokens": 0,
            "tok_s": 0,
            "status": 0,
            "error": str(e)[:200],
        }


async def run_batch(url, concurrency, total_requests):
    results = {}
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(total_requests):
            prompt = PROMPTS[i % len(PROMPTS)]
            tasks.append(send_request(session, url, prompt, i, results))
        
        batch_start = time.perf_counter()
        await asyncio.gather(*tasks)
        batch_elapsed = time.perf_counter() - batch_start
    
    return results, batch_elapsed


def analyze_results(results, batch_elapsed, concurrency):
    total = len(results)
    success = sum(1 for r in results.values() if r["success"])
    failed = total - success
    
    latencies = [r["latency_ms"] for r in results.values() if r["success"]]
    tokens = [r["tokens"] for r in results.values() if r["success"]]
    tok_s_list = [r["tok_s"] for r in results.values() if r["success"]]
    
    total_tokens = sum(tokens)
    aggregate_tok_s = total_tokens / batch_elapsed if batch_elapsed > 0 else 0
    
    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p99 = latencies[int(len(latencies) * 0.99)]
        avg_lat = sum(latencies) / len(latencies)
        avg_tok_s = sum(tok_s_list) / len(tok_s_list) if tok_s_list else 0
    else:
        p50 = p99 = avg_lat = avg_tok_s = 0
    
    print(f"\n{'='*60}")
    print(f"  Concurrency: {concurrency} | Total Requests: {total}")
    print(f"{'='*60}")
    print(f"  Success:     {success}/{total} ({success/total*100:.1f}%)")
    print(f"  Failed:      {failed}")
    print(f"  Batch Time:  {batch_elapsed:.2f}s")
    print(f"  Total Tokens: {total_tokens}")
    print(f"  Aggregate:   {aggregate_tok_s:.1f} tok/s")
    print(f"  Per-req:     {avg_tok_s:.1f} tok/s")
    print(f"  Latency:     avg={avg_lat:.0f}ms  p50={p50:.0f}ms  p99={p99:.0f}ms")
    
    if failed > 0:
        errors = defaultdict(int)
        for r in results.values():
            if not r["success"]:
                errors[r.get("error", "unknown")[:80]] += 1
        print(f"  Errors:")
        for err, cnt in sorted(errors.items(), key=lambda x: -x[1])[:5]:
            print(f"    ({cnt}x) {err}")
    
    print()
    return {
        "concurrency": concurrency,
        "total": total,
        "success": success,
        "failed": failed,
        "batch_time_s": batch_elapsed,
        "total_tokens": total_tokens,
        "aggregate_tok_s": aggregate_tok_s,
        "avg_tok_s": avg_tok_s,
        "avg_latency_ms": avg_lat,
        "p50_latency_ms": p50,
        "p99_latency_ms": p99,
    }


async def check_routing(url):
    """Check Global LB stats for routing distribution."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/stats", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"\n  Routing Distribution (Global LB):")
                    for h in data.get("hosts", []):
                        print(f"    {h['name']}: proxied={h['proxied_requests']}, active={h['active_backends']}, weight={h['weight']}")
                    print(f"    Total: {data.get('total_requests', 0)} requests, {data.get('total_errors', 0)} errors")
    except Exception as e:
        print(f"  (Could not fetch Global LB stats: {e})")


async def main():
    parser = argparse.ArgumentParser(description="16-card stress test")
    parser.add_argument("--url", default="http://localhost:30020", help="edge_first_proxy URL")
    parser.add_argument("--glb-url", default="http://localhost:30050", help="Global LB stats URL")
    parser.add_argument("--levels", default="16,32,64,128,256", help="Concurrency levels")
    parser.add_argument("--per-level", type=int, default=None, help="Requests per level (default=concurrency*4)")
    args = parser.parse_args()
    
    levels = [int(x) for x in args.levels.split(",")]
    
    print(f"CGC 16-Card Cross-Host Stress Test")
    print(f"Target: {args.url}")
    print(f"Concurrency levels: {levels}")
    
    all_results = []
    for concurrency in levels:
        total = args.per_level or (concurrency * 4)
        results, batch_elapsed = await run_batch(args.url, concurrency, total)
        summary = analyze_results(results, batch_elapsed, concurrency)
        all_results.append(summary)
        await check_routing(args.glb_url)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Conc':>5} | {'Success':>8} | {'Agg tok/s':>10} | {'Avg tok/s':>10} | {'P50 ms':>8} | {'P99 ms':>8}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    for r in all_results:
        print(f"  {r['concurrency']:>5} | {r['success']:>4}/{r['total']:<3} | {r['aggregate_tok_s']:>10.1f} | {r['avg_tok_s']:>10.1f} | {r['p50_latency_ms']:>8.0f} | {r['p99_latency_ms']:>8.0f}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
