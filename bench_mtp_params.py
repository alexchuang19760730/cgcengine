#!/usr/bin/env python3
"""MTP 投机参数基准测试。

测试不同 speculative-num-steps 和 eagle-topk 参数下的 decode 性能。
直接连接 sglang (端口 30001), 绕过 proxy。

用法:
  python3 bench_mtp_params.py
"""
import urllib.request
import time
import json
import sys
import os

SGLANG_URL = os.environ.get("SGLANG_URL", "http://39.106.118.206:30001")
COMPLETIONS_URL = f"{SGLANG_URL}/v1/chat/completions"
MODEL = "/data/models/gemma-4-26b-a4b-it"

TEST_PROMPTS = [
    "Write a Python function to sort a list using quicksort",
    "Explain how binary search works with an example",
    "Write a REST API endpoint in Flask that handles user registration",
    "Debug this Python code: def fib(n): if n<2: return n else: return fib(n-1)+fib(n-2)",
]


def bench_nonstream(prompt, max_tokens=200):
    """Non-streaming benchmark: measures total throughput (Pipeline-style)."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    t0 = time.monotonic()
    req = urllib.request.Request(COMPLETIONS_URL, data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    elapsed = time.monotonic() - t0
    tokens = data.get("usage", {}).get("completion_tokens", 0)
    tps = tokens / elapsed if elapsed > 0 else 0
    return {"tokens": tokens, "elapsed": elapsed, "tps": tps}


def bench_stream(prompt, max_tokens=100):
    """Streaming benchmark: measures TTFT and decode rate."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    t0 = time.monotonic()
    first_token = None
    total_tokens = 0
    req = urllib.request.Request(COMPLETIONS_URL, data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    resp = urllib.request.urlopen(req, timeout=120)
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            try:
                obj = json.loads(line[5:].strip())
                choices = obj.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content", "")
                    if content and first_token is None:
                        first_token = time.monotonic() - t0
                    if content:
                        total_tokens += 1
            except Exception:
                pass
    total_time = time.monotonic() - t0
    ttft = first_token * 1000 if first_token else 0
    tps = total_tokens / total_time if total_time > 0 else 0
    return {"ttft_ms": ttft, "tokens": total_tokens, "elapsed": total_time, "tps": tps}


def bench_concurrent(num_concurrent=4, max_tokens=100):
    """Concurrent benchmark: measures aggregate throughput."""
    import threading

    results = [None] * num_concurrent

    def worker(idx):
        prompt = TEST_PROMPTS[idx % len(TEST_PROMPTS)]
        results[idx] = bench_nonstream(prompt, max_tokens)

    threads = []
    t0 = time.monotonic()
    for i in range(num_concurrent):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    total_time = time.monotonic() - t0

    total_tokens = sum(r["tokens"] for r in results if r)
    aggregate_tps = total_tokens / total_time if total_time > 0 else 0
    individual_tps = [r["tps"] for r in results if r]
    return {
        "total_tokens": total_tokens,
        "total_time": total_time,
        "aggregate_tps": aggregate_tps,
        "individual_tps": individual_tps,
    }


def main():
    print("=" * 70)
    print("MTP 投机参数基准测试")
    print(f"sglang: {SGLANG_URL}")
    print(f"model: {MODEL}")
    print("=" * 70)

    # Check sglang health
    try:
        req = urllib.request.Request(f"{SGLANG_URL}/v1/models")
        resp = urllib.request.urlopen(req, timeout=10)
        models = json.loads(resp.read())
        print(f"sglang status: OK ({len(models.get('data', []))} models)")
    except Exception as e:
        print(f"sglang status: FAILED ({e})")
        return

    # Non-streaming benchmark
    print("\n--- Non-streaming (Pipeline throughput) ---")
    for prompt in TEST_PROMPTS:
        r = bench_nonstream(prompt, max_tokens=200)
        print(f"  {prompt[:40]:42s} {r['tokens']:3d} tok / {r['elapsed']:.2f}s = {r['tps']:6.1f} tok/s")

    # Average
    all_tps = []
    for _ in range(3):
        r = bench_nonstream(TEST_PROMPTS[0], max_tokens=200)
        all_tps.append(r["tps"])
    avg = sum(all_tps) / len(all_tps)
    print(f"  Average (3 runs): {avg:.1f} tok/s")

    # Streaming benchmark
    print("\n--- Streaming (TTFT + decode) ---")
    for prompt in TEST_PROMPTS:
        r = bench_stream(prompt, max_tokens=100)
        print(f"  {prompt[:40]:42s} TTFT={r['ttft_ms']:5.0f}ms  {r['tokens']:3d} tok / {r['elapsed']:.2f}s = {r['tps']:6.1f} tok/s")

    # Concurrent benchmark
    print("\n--- Concurrent (4-way Pipeline) ---")
    r = bench_concurrent(num_concurrent=4, max_tokens=100)
    print(f"  4 concurrent: {r['total_tokens']} tokens in {r['total_time']:.2f}s = {r['aggregate_tps']:.1f} tok/s aggregate")
    print(f"  Individual: {['%.1f' % t for t in r['individual_tps']]}")

    print("\n" + "=" * 70)
    print(f"Summary: non-stream={avg:.1f} tok/s, concurrent={r['aggregate_tps']:.1f} tok/s")
    print("=" * 70)


if __name__ == "__main__":
    main()
