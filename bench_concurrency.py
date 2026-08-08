#!/usr/bin/env python3
"""
CGC Concurrency Benchmark
Tests pure cloud (LB) vs edge_first_proxy amplification.
Runs on Host1 to minimize network latency.
"""

import asyncio
import aiohttp
import json
import time
import statistics
import sys
import argparse
from collections import defaultdict

# Code completion prompts (realistic autocomplete scenarios)
CODE_PROMPTS = [
    {"role": "user", "content": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"},
    {"role": "user", "content": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x < pivot]\n    right = [x for x in arr[1:] if x >= pivot]\n    return"},
    {"role": "user", "content": "class LinkedList:\n    def __init__(self):\n        self.head = None\n    def append(self, value):\n        new_node = Node(value)\n        if not self.head:\n            self.head = new_node\n            return"},
    {"role": "user", "content": "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n"},
    {"role": "user", "content": "import torch\nimport torch.nn as nn\n\nclass Transformer(nn.Module):\n    def __init__(self, d_model=512, nhead=8, num_layers=6):\n        super().__init__()\n        self.embedding = nn.Embedding(10000, d_model)\n        self.layers = nn.ModuleList(["},
    {"role": "user", "content": "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return"},
    {"role": "user", "content": "async def fetch_data(session, url):\n    try:\n        async with session.get(url) as response:\n            if response.status == 200:\n                data = await response.json()\n                return"},
    {"role": "user", "content": "def train_model(model, dataloader, optimizer, criterion, epochs=10):\n    model.train()\n    for epoch in range(epochs):\n        total_loss = 0\n        for batch_idx, (data, target) in enumerate(dataloader):\n            optimizer.zero_grad()\n            output = model(data)\n            loss ="},
]

CHAT_PROMPTS = [
    {"role": "user", "content": "Explain how gradient descent works in machine learning."},
    {"role": "user", "content": "What are the advantages of microservices architecture?"},
    {"role": "user", "content": "How does HTTP/2 differ from HTTP/1.1?"},
    {"role": "user", "content": "Explain the CAP theorem in distributed systems."},
    {"role": "user", "content": "What is the difference between TCP and UDP?"},
    {"role": "user", "content": "How does garbage collection work in Python?"},
    {"role": "user", "content": "Explain the observer pattern in software design."},
    {"role": "user", "content": "What is dependency injection and why is it useful?"},
]


async def single_request(session, url, prompt, request_id, max_tokens=64):
    """Send a single request and measure TTFT + throughput."""
    # prompts are dicts like {"role":"user","content":"..."} or lists of dicts
    if isinstance(prompt, list):
        messages = prompt
    elif isinstance(prompt, dict):
        messages = [prompt]
    else:
        messages = [{"role": "user", "content": str(prompt)}]

    payload = {
        "model": "gemma-4-26b-a4b-it",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    start_time = time.perf_counter()
    ttft = None
    token_count = 0
    first_token_time = None
    error = None

    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return {
                    "id": request_id,
                    "success": False,
                    "error": error,
                    "ttft_ms": None,
                    "tokens": 0,
                    "total_ms": (time.perf_counter() - start_time) * 1000,
                    "tok_s": 0,
                }

            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if ttft is None:
                                ttft = time.perf_counter() - start_time
                                first_token_time = time.perf_counter()
                            token_count += 1
                except json.JSONDecodeError:
                    continue

    except asyncio.TimeoutError:
        error = "timeout"
    except Exception as e:
        error = str(e)[:200]

    end_time = time.perf_counter()
    total_ms = (end_time - start_time) * 1000
    ttft_ms = ttft * 1000 if ttft else None
    decode_ms = (end_time - first_token_time) * 1000 if first_token_time else 0
    tok_s = (token_count / decode_ms * 1000) if decode_ms > 0 and token_count > 0 else 0

    return {
        "id": request_id,
        "success": error is None and token_count > 0,
        "error": error,
        "ttft_ms": ttft_ms,
        "tokens": token_count,
        "total_ms": total_ms,
        "tok_s": tok_s,
    }


async def run_concurrency_test(url, concurrency, num_requests, prompts, use_cache=False, label=""):
    """Run concurrent requests at given concurrency level."""
    print(f"\n{'='*70}")
    print(f"  {label} | concurrency={concurrency} | requests={num_requests}")
    print(f"{'='*70}")

    # If use_cache, first warm up cache by sending each prompt once
    if use_cache:
        print("  [CACHE WARMUP] Sending each prompt once to build cache...")
        async with aiohttp.ClientSession() as warmup_session:
            warmup_tasks = []
            for i, p in enumerate(prompts):
                warmup_tasks.append(single_request(warmup_session, url, p, f"warmup_{i}", max_tokens=32))
            await asyncio.gather(*warmup_tasks)
        print(f"  [CACHE WARMUP] Done. {len(prompts)} prompts cached.")
        await asyncio.sleep(1)

    semaphore = asyncio.Semaphore(concurrency)
    results = []
    completed = 0

    async def bounded_request(session, prompt, req_id):
        nonlocal completed
        async with semaphore:
            result = await single_request(session, url, prompt, req_id)
            completed += 1
            if completed % 10 == 0:
                print(f"    Progress: {completed}/{num_requests}")
            return result

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            tasks.append(bounded_request(session, prompt, i))

        batch_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        batch_end = time.perf_counter()
        wall_time = batch_end - batch_start

    # Analyze
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    ttfts = [r["ttft_ms"] for r in successes if r["ttft_ms"] is not None]
    tok_s_list = [r["tok_s"] for r in successes if r["tok_s"] > 0]
    total_tokens = sum(r["tokens"] for r in successes)

    success_rate = len(successes) / len(results) * 100
    avg_ttft = statistics.mean(ttfts) if ttfts else 0
    p50_ttft = statistics.median(ttfts) if ttfts else 0
    p99_ttft = sorted(ttfts)[int(len(ttfts) * 0.99)] if len(ttfts) > 1 else (ttfts[0] if ttfts else 0)
    avg_tok_s = statistics.mean(tok_s_list) if tok_s_list else 0
    aggregate_tok_s = total_tokens / wall_time if wall_time > 0 else 0

    print(f"\n  Results:")
    print(f"    Success rate:   {success_rate:.1f}% ({len(successes)}/{len(results)})")
    print(f"    Wall time:      {wall_time:.2f}s")
    print(f"    TTFT avg:       {avg_ttft:.1f}ms")
    print(f"    TTFT p50:       {p50_ttft:.1f}ms")
    print(f"    TTFT p99:       {p99_ttft:.1f}ms")
    print(f"    Per-req tok/s:  {avg_tok_s:.1f}")
    print(f"    Total tokens:   {total_tokens}")
    print(f"    Aggregate tok/s:{aggregate_tok_s:.1f}")

    if failures:
        error_types = defaultdict(int)
        for f in failures:
            error_types[f["error"][:80] if f["error"] else "unknown"] += 1
        print(f"    Failures:       {len(failures)}")
        for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"      - {etype}: {count}")

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "success_rate": success_rate,
        "wall_time_s": wall_time,
        "ttft_avg_ms": avg_ttft,
        "ttft_p50_ms": p50_ttft,
        "ttft_p99_ms": p99_ttft,
        "per_req_tok_s": avg_tok_s,
        "aggregate_tok_s": aggregate_tok_s,
        "total_tokens": total_tokens,
        "failures": len(failures),
    }


async def main():
    parser = argparse.ArgumentParser(description="CGC Concurrency Benchmark")
    parser.add_argument("--lb-url", default="http://127.0.0.1:30010", help="Load balancer URL")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:30020", help="Proxy URL")
    parser.add_argument("--levels", default="1,4,8,16,32,64", help="Concurrency levels (comma-separated)")
    parser.add_argument("--requests", type=int, default=32, help="Total requests per level")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens per request")
    parser.add_argument("--mode", default="code", choices=["code", "chat", "both"], help="Prompt type")
    parser.add_argument("--skip-proxy", action="store_true", help="Skip proxy tests")
    parser.add_argument("--skip-lb", action="store_true", help="Skip LB tests")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    prompts = CODE_PROMPTS if args.mode == "code" else (CHAT_PROMPTS if args.mode == "chat" else CODE_PROMPTS + CHAT_PROMPTS)

    all_results = {"lb": [], "proxy": [], "proxy_cached": []}

    # Test 1: Pure cloud (direct to LB)
    if not args.skip_lb:
        print("\n" + "#"*70)
        print("#  PHASE 1: PURE CLOUD (Direct to Load Balancer)")
        print("#"*70)
        for level in levels:
            num_req = max(level * 4, args.requests)  # At least 4x concurrency
            result = await run_concurrency_test(
                args.lb_url, level, num_req, prompts,
                label="[LB - Pure Cloud]"
            )
            all_results["lb"].append(result)

    # Test 2: Through proxy (no cache warmup)
    if not args.skip_proxy:
        print("\n" + "#"*70)
        print("#  PHASE 2: EDGE PROXY (No Cache Warmup)")
        print("#"*70)
        for level in levels:
            num_req = max(level * 4, args.requests)
            result = await run_concurrency_test(
                args.proxy_url, level, num_req, prompts,
                label="[Proxy - No Cache]"
            )
            all_results["proxy"].append(result)

    # Test 3: Through proxy with cached prompts
    if not args.skip_proxy:
        print("\n" + "#"*70)
        print("#  PHASE 3: EDGE PROXY (With Cache - Simulating Repeat Requests)")
        print("#"*70)
        for level in levels:
            num_req = max(level * 4, args.requests)
            result = await run_concurrency_test(
                args.proxy_url, level, num_req, prompts,
                use_cache=True,
                label="[Proxy - Cached]"
            )
            all_results["proxy_cached"].append(result)

    # Summary table
    print("\n\n" + "="*100)
    print("  SUMMARY: Concurrency Comparison")
    print("="*100)
    print(f"  {'Level':<8} {'LB TTFT':>10} {'LB tok/s':>10} {'LB agg':>10} | "
          f"{'Proxy TTFT':>12} {'Proxy tok/s':>12} {'Proxy agg':>10} | "
          f"{'Cached TTFT':>12} {'Cached agg':>10}")
    print("-"*100)

    for i, level in enumerate(levels):
        lb = all_results["lb"][i] if i < len(all_results["lb"]) else {}
        proxy = all_results["proxy"][i] if i < len(all_results["proxy"]) else {}
        cached = all_results["proxy_cached"][i] if i < len(all_results["proxy_cached"]) else {}

        lb_ttft = f"{lb.get('ttft_avg_ms', 0):.1f}" if lb else "N/A"
        lb_tps = f"{lb.get('per_req_tok_s', 0):.1f}" if lb else "N/A"
        lb_agg = f"{lb.get('aggregate_tok_s', 0):.1f}" if lb else "N/A"

        px_ttft = f"{proxy.get('ttft_avg_ms', 0):.1f}" if proxy else "N/A"
        px_tps = f"{proxy.get('per_req_tok_s', 0):.1f}" if proxy else "N/A"
        px_agg = f"{proxy.get('aggregate_tok_s', 0):.1f}" if proxy else "N/A"

        ca_ttft = f"{cached.get('ttft_avg_ms', 0):.1f}" if cached else "N/A"
        ca_agg = f"{cached.get('aggregate_tok_s', 0):.1f}" if cached else "N/A"

        print(f"  {level:<8} {lb_ttft:>10} {lb_tps:>10} {lb_agg:>10} | "
              f"{px_ttft:>12} {px_tps:>12} {px_agg:>10} | "
              f"{ca_ttft:>12} {ca_agg:>10}")

    print("\n" + "="*100)
    print("  Amplification (Cached vs Pure Cloud):")
    for i, level in enumerate(levels):
        if i < len(all_results["lb"]) and i < len(all_results["proxy_cached"]):
            lb = all_results["lb"][i]
            ca = all_results["proxy_cached"][i]
            if lb["ttft_avg_ms"] > 0 and ca["ttft_avg_ms"] > 0:
                speedup = lb["ttft_avg_ms"] / ca["ttft_avg_ms"]
                print(f"    Concurrency {level:>3}: TTFT {lb['ttft_avg_ms']:.1f}ms → {ca['ttft_avg_ms']:.1f}ms ({speedup:.2f}x)")
    print("="*100)


if __name__ == "__main__":
    asyncio.run(main())
