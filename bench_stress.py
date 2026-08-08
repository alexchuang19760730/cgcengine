#!/usr/bin/env python3
"""CGC Concurrency Stress Test - Find the breaking point of 8x TP1+FP8 instances."""
import asyncio, aiohttp, json, time, statistics, sys, random, string
from collections import defaultdict

# Code completion prompts (cacheable - repeated across requests)
CODE_PROMPTS = [
    {"role": "user", "content": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"},
    {"role": "user", "content": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x < pivot]\n    right = [x for x in arr[1:] if x >= pivot]\n    return"},
    {"role": "user", "content": "class LinkedList:\n    def __init__(self):\n        self.head = None\n    def append(self, value):\n        new_node = Node(value)\n        if not self.head:\n            self.head = new_node\n            return"},
    {"role": "user", "content": "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:"},
    {"role": "user", "content": "import torch\nimport torch.nn as nn\n\nclass Transformer(nn.Module):\n    def __init__(self, d_model=512, nhead=8):\n        super().__init__()\n        self.embedding = nn.Embedding(10000, d_model)\n        self.layers = nn.ModuleList(["},
    {"role": "user", "content": "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return"},
    {"role": "user", "content": "async def fetch_data(session, url):\n    try:\n        async with session.get(url) as response:\n            if response.status == 200:\n                data = await response.json()\n                return"},
    {"role": "user", "content": "def train_model(model, dataloader, optimizer, criterion, epochs=10):\n    model.train()\n    for epoch in range(epochs):\n        total_loss = 0\n        for batch_idx, (data, target) in enumerate(dataloader):\n            optimizer.zero_grad()\n            output = model(data)\n            loss ="},
]

# Random unique prompts (uncacheable - each one is different)
def gen_random_prompt():
    """Generate a unique prompt that won't be in cache."""
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
    return {"role": "user", "content": f"# Session: {suffix}\n# Write a function to process data\ndef process_{suffix[:6]}(data):\n    result = []\n    for item in data:\n        if item is not None:\n            result.append("}

async def single_request(session, url, prompt, req_id, max_tokens=64):
    """Send a single streaming request and measure TTFT + throughput."""
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
    
    start = time.perf_counter()
    ttft = None
    tokens = 0
    first_tok_time = None
    error = None
    
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                txt = await resp.text()
                error = f"HTTP {resp.status}: {txt[:200]}"
            else:
                async for line in resp.content:
                    ls = line.decode("utf-8").strip()
                    if not ls or not ls.startswith("data: "):
                        continue
                    ds = ls[6:]
                    if ds == "[DONE]":
                        break
                    try:
                        d = json.loads(ds)
                        if "choices" in d and d["choices"]:
                            c = d["choices"][0].get("delta", {}).get("content", "")
                            if c:
                                if ttft is None:
                                    ttft = time.perf_counter() - start
                                    first_tok_time = time.perf_counter()
                                tokens += 1
                    except:
                        continue
    except asyncio.TimeoutError:
        error = "timeout_120s"
    except Exception as e:
        error = str(e)[:200]
    
    end = time.perf_counter()
    total_ms = (end - start) * 1000
    ttft_ms = ttft * 1000 if ttft else None
    decode_ms = (end - first_tok_time) * 1000 if first_tok_time else 0
    tok_s = (tokens / decode_ms * 1000) if decode_ms > 0 and tokens > 0 else 0
    
    return {
        "id": req_id,
        "success": error is None and tokens > 0,
        "error": error,
        "ttft_ms": ttft_ms,
        "tokens": tokens,
        "total_ms": total_ms,
        "tok_s": tok_s,
    }


async def run_test(url, concurrency, num_requests, prompt_fn, label="", warmup_prompts=None):
    """Run a concurrency test with the given parameters."""
    print(f"\n{'='*70}")
    print(f"  {label} | concurrency={concurrency} | requests={num_requests}")
    print(f"{'='*70}")
    
    # Warmup: send each prompt once to populate cache
    if warmup_prompts:
        print("  [WARMUP] Populating cache...")
        async with aiohttp.ClientSession() as ws:
            wt = [single_request(ws, url, p, f"w{i}", 32) for i, p in enumerate(warmup_prompts)]
            await asyncio.gather(*wt)
        print(f"  [WARMUP] Done. {len(warmup_prompts)} prompts cached.")
        await asyncio.sleep(0.5)
    
    sem = asyncio.Semaphore(concurrency)
    done = [0]
    
    async def bounded(session, prompt, rid):
        async with sem:
            r = await single_request(session, url, prompt, rid)
            done[0] += 1
            if done[0] % max(1, num_requests // 10) == 0:
                print(f"    Progress: {done[0]}/{num_requests}")
            return r
    
    async with aiohttp.ClientSession() as session:
        # Generate prompts
        if callable(prompt_fn):
            prompts_list = [prompt_fn() for _ in range(num_requests)]
        else:
            prompts_list = [prompt_fn[i % len(prompt_fn)] for i in range(num_requests)]
        
        tasks = [bounded(session, prompts_list[i], i) for i in range(num_requests)]
        bs = time.perf_counter()
        results = await asyncio.gather(*tasks)
        be = time.perf_counter()
        wall = be - bs
    
    succ = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]
    ttfts = [r["ttft_ms"] for r in succ if r["ttft_ms"] is not None]
    tps = [r["tok_s"] for r in succ if r["tok_s"] > 0]
    total_tok = sum(r["tokens"] for r in succ)
    sr = len(succ) / len(results) * 100
    
    avg_ttft = statistics.mean(ttfts) if ttfts else 0
    p50_ttft = statistics.median(ttfts) if ttfts else 0
    p99_ttft = sorted(ttfts)[int(len(ttfts)*0.99)] if len(ttfts) > 1 else (ttfts[0] if ttfts else 0)
    max_ttft = max(ttfts) if ttfts else 0
    
    avg_tps = statistics.mean(tps) if tps else 0
    agg = total_tok / wall if wall > 0 else 0
    
    print(f"\n  Success:    {sr:.1f}% ({len(succ)}/{len(results)})")
    print(f"  Wall time:  {wall:.2f}s")
    print(f"  TTFT avg:   {avg_ttft:.1f}ms  p50: {p50_ttft:.1f}ms  p99: {p99_ttft:.1f}ms  max: {max_ttft:.1f}ms")
    print(f"  Per-req:    {avg_tps:.1f} tok/s")
    print(f"  Aggregate:  {agg:.1f} tok/s  ({total_tok} tokens in {wall:.1f}s)")
    
    if fail:
        ets = defaultdict(int)
        for f in fail:
            ets[f["error"][:80] if f["error"] else "no_tokens"] += 1
        print(f"  Failures:   {len(fail)}")
        for e, c in sorted(ets.items(), key=lambda x: -x[1])[:5]:
            print(f"    - {e}: {c}")
    
    return {
        "concurrency": concurrency,
        "success_rate": sr,
        "wall_s": wall,
        "ttft_avg": avg_ttft,
        "ttft_p50": p50_ttft,
        "ttft_p99": p99_ttft,
        "ttft_max": max_ttft,
        "per_req_tps": avg_tps,
        "agg_tps": agg,
        "total_tok": total_tok,
        "failures": len(fail),
    }


async def main():
    lb_url = "http://127.0.0.1:30010"
    proxy_url = "http://127.0.0.1:30020"
    
    # Test levels
    levels = [64, 128, 256]
    
    results = {
        "lb_code": [],
        "proxy_cached": [],
        "proxy_random": [],
        "lb_random": [],
    }
    
    # ============================================================
    # PHASE 1: Direct LB with code prompts (pure cloud, no cache)
    # ============================================================
    print("#" * 70)
    print("#  PHASE 1: Direct LB - Code Prompts (Pure Cloud)")
    print("#" * 70)
    for lv in levels:
        nr = lv * 3  # 3 rounds per level
        r = await run_test(lb_url, lv, nr, CODE_PROMPTS, "[LB Code]")
        results["lb_code"].append(r)
        await asyncio.sleep(2)
    
    # ============================================================
    # PHASE 2: Through Proxy with cached code prompts
    # ============================================================
    print("\n#" * 35)
    print("#  PHASE 2: Proxy - Cached Code Prompts (Cache Amplified)")
    print("#" * 70)
    for lv in levels:
        nr = lv * 3
        r = await run_test(proxy_url, lv, nr, CODE_PROMPTS, "[Proxy Cached]", warmup_prompts=CODE_PROMPTS)
        results["proxy_cached"].append(r)
        await asyncio.sleep(2)
    
    # ============================================================
    # PHASE 3: Through Proxy with random unique prompts (cache miss)
    # ============================================================
    print("\n#" * 35)
    print("#  PHASE 3: Proxy - Random Prompts (Cache Miss)")
    print("#" * 70)
    for lv in levels:
        nr = lv * 3
        r = await run_test(proxy_url, lv, nr, gen_random_prompt, "[Proxy Random]")
        results["proxy_random"].append(r)
        await asyncio.sleep(2)
    
    # ============================================================
    # PHASE 4: Direct LB with random prompts (pure cloud baseline)
    # ============================================================
    print("\n#" * 35)
    print("#  PHASE 4: Direct LB - Random Prompts (Pure Cloud, No Cache)")
    print("#" * 70)
    for lv in levels:
        nr = lv * 3
        r = await run_test(lb_url, lv, nr, gen_random_prompt, "[LB Random]")
        results["lb_random"].append(r)
        await asyncio.sleep(2)
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n\n" + "=" * 120)
    print("  STRESS TEST SUMMARY - 8x TP1+FP8 Instances")
    print("=" * 120)
    print(f"  {'Conc':<6} | {'--- LB Code (Pure Cloud) ---':^30} | {'--- Proxy Cached ---':^30} | {'--- Proxy Random (Miss) ---':^30} | {'--- LB Random ---':^30}")
    print(f"  {'':6} | {'TTFT':>7} {'tok/s':>7} {'SR%':>5} {'agg':>7} | {'TTFT':>7} {'tok/s':>7} {'SR%':>5} {'agg':>7} | {'TTFT':>7} {'tok/s':>7} {'SR%':>5} {'agg':>7} | {'TTFT':>7} {'tok/s':>7} {'SR%':>5} {'agg':>7}")
    print("-" * 140)
    
    for i, lv in enumerate(levels):
        lbc = results["lb_code"][i]
        pc = results["proxy_cached"][i]
        pr = results["proxy_random"][i]
        lr = results["lb_random"][i]
        print(f"  {lv:<6} | {lbc['ttft_avg']:>7.1f} {lbc['per_req_tps']:>7.1f} {lbc['success_rate']:>5.1f} {lbc['agg_tps']:>7.1f} | {pc['ttft_avg']:>7.1f} {pc['per_req_tps']:>7.1f} {pc['success_rate']:>5.1f} {pc['agg_tps']:>7.1f} | {pr['ttft_avg']:>7.1f} {pr['per_req_tps']:>7.1f} {pr['success_rate']:>5.1f} {pr['agg_tps']:>7.1f} | {lr['ttft_avg']:>7.1f} {lr['per_req_tps']:>7.1f} {lr['success_rate']:>5.1f} {lr['agg_tps']:>7.1f}")
    
    print("\n  P99 TTFT Comparison:")
    print(f"  {'Conc':<6} | {'LB Code':>10} | {'Proxy Cache':>12} | {'Proxy Miss':>12} | {'LB Random':>10}")
    print("-" * 60)
    for i, lv in enumerate(levels):
        print(f"  {lv:<6} | {results['lb_code'][i]['ttft_p99']:>10.1f} | {results['proxy_cached'][i]['ttft_p99']:>12.1f} | {results['proxy_random'][i]['ttft_p99']:>12.1f} | {results['lb_random'][i]['ttft_p99']:>10.1f}")
    
    print("\n  Cache Amplification (LB Code TTFT / Proxy Cached TTFT):")
    for i, lv in enumerate(levels):
        lb_t = results["lb_code"][i]["ttft_avg"]
        pc_t = results["proxy_cached"][i]["ttft_avg"]
        if pc_t > 0:
            print(f"    Conc {lv:>3}: {lb_t:.1f}ms -> {pc_t:.1f}ms ({lb_t/pc_t:.1f}x faster)")
    
    print("\n  Max Concurrency Analysis:")
    for i, lv in enumerate(levels):
        lbc = results["lb_code"][i]
        pc = results["proxy_cached"][i]
        print(f"    Conc {lv:>3}: LB SR={lbc['success_rate']:.0f}% P99={lbc['ttft_p99']:.0f}ms | Proxy SR={pc['success_rate']:.0f}% P99={pc['ttft_p99']:.0f}ms")
    
    print("=" * 120)


if __name__ == "__main__":
    asyncio.run(main())
