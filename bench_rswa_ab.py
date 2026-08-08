#!/usr/bin/env python3
"""
R-SWA A/B Test: Measure memory, prefill, decode at short/medium/long contexts.

Usage:
  python3 bench_rswa_ab.py [--label baseline|rswa]
"""

import urllib.request
import json
import time
import sys
import argparse
import threading

URL = "http://39.106.118.206:30001"
MODEL = "/data/models/gemma-4-26b-a4b-it"

# Context lengths to test
CONTEXTS = {
    "short":  {"input_tokens": 128,   "max_tokens": 256,  "prompt": "Write a Python function to sort a list using quicksort. Include comments and type hints."},
    "medium": {"input_tokens": 1024,  "max_tokens": 256,  "prompt": None},  # Will be generated
    "long":   {"input_tokens": 4096,  "max_tokens": 256,  "prompt": None},  # Will be generated
    "xlong":  {"input_tokens": 8192,  "max_tokens": 256,  "prompt": None},  # Will be generated
}


def generate_long_prompt(target_tokens: int) -> str:
    """Generate a prompt of approximately target_tokens length."""
    base = "The history of computing is a fascinating journey through innovation and discovery. "
    base += "From the earliest mechanical calculators to modern quantum computers, each era has built upon the achievements of its predecessors. "
    base += "The development of programming languages has been particularly instrumental in shaping how we interact with machines. "
    # Repeat to reach target length (~4 chars per token)
    target_chars = target_tokens * 4
    result = base
    while len(result) < target_chars:
        result += base
    return result[:target_chars]


# Pre-generate medium/long prompts
for key in ["medium", "long", "xlong"]:
    if CONTEXTS[key]["prompt"] is None:
        CONTEXTS[key]["prompt"] = generate_long_prompt(CONTEXTS[key]["input_tokens"])


def get_sglang_metrics():
    """Fetch sglang internal metrics including KV cache usage."""
    try:
        req = urllib.request.Request(f"{URL}/metrics", method="GET")
        resp = urllib.request.urlopen(req, timeout=10)
        return resp.read().decode()
    except:
        return ""


def get_server_info():
    """Get server memory/cache info."""
    try:
        req = urllib.request.Request(f"{URL}/get_server_info", method="POST",
                                     data=json.dumps({}).encode(),
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read())
    except:
        return {}


def run_non_streaming(prompt, max_tokens, temperature=0.0):
    """Run non-streaming request, measure prefill and decode separately."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    resp = urllib.request.urlopen(req, timeout=300)
    data = json.loads(resp.read())
    elapsed = time.monotonic() - t0

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_time": elapsed,
        "tps": completion_tokens / elapsed if elapsed > 0 else 0,
    }


def run_streaming(prompt, max_tokens, temperature=0.0):
    """Run streaming request, measure TTFT and decode rate separately."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")

    t0 = time.monotonic()
    first_token_time = None
    total_tokens = 0
    prompt_tokens = 0
    resp = urllib.request.urlopen(req, timeout=300)

    for line in resp:
        line = line.decode().strip()
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            try:
                obj = json.loads(line[5:].strip())
                choices = obj.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                        total_tokens += 1
                usage = obj.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
            except:
                pass

    total_time = time.monotonic() - t0
    ttft = (first_token_time - t0) * 1000 if first_token_time else 0
    decode_time = (total_time - (first_token_time - t0)) if first_token_time else total_time
    decode_tokens = total_tokens - 1 if total_tokens > 0 else 0
    decode_rate = decode_tokens / decode_time if decode_time > 0 else 0

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": total_tokens,
        "ttft_ms": ttft,
        "total_time": total_time,
        "decode_rate": decode_rate,
        "prefill_time": (first_token_time - t0) if first_token_time else total_time,
        "prefill_rate": prompt_tokens / (first_token_time - t0) if first_token_time and prompt_tokens > 0 else 0,
    }


def run_concurrent(prompt, max_tokens, concurrency=4):
    """Run concurrent requests and measure aggregate throughput."""
    import concurrent.futures

    results = [None] * concurrency

    def worker(idx):
        results[idx] = run_non_streaming(prompt, max_tokens)

    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, i) for i in range(concurrency)]
        concurrent.futures.wait(futures)
    total_time = time.monotonic() - t0

    total_tokens = sum(r["completion_tokens"] for r in results if r)
    total_prompt = sum(r["prompt_tokens"] for r in results if r)
    aggregate_tps = total_tokens / total_time if total_time > 0 else 0

    return {
        "concurrency": concurrency,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_tokens,
        "total_time": total_time,
        "aggregate_tps": aggregate_tps,
        "individual": [{"tps": r["tps"], "tokens": r["completion_tokens"]} for r in results if r],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline", help="Label for this run (baseline/rswa)")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"  R-SWA A/B Test — Label: {args.label}")
    print(f"  Server: {URL}")
    print(f"  Model: {MODEL}")
    print(f"{'='*80}\n")

    # Health check
    try:
        req = urllib.request.Request(f"{URL}/v1/models", method="GET")
        resp = urllib.request.urlopen(req, timeout=10)
        models = json.loads(resp.read())
        print(f"[OK] Server healthy: {models['data'][0]['id']}")
    except Exception as e:
        print(f"[FAIL] Server not reachable: {e}")
        sys.exit(1)

    # Get server info before tests
    server_info = get_server_info()
    if server_info:
        mem_info = server_info.get("internal_states", [])
        print(f"\n[Server Info] Pre-test:")
        for state in mem_info:
            if isinstance(state, dict):
                for k, v in state.items():
                    if "mem" in k.lower() or "cache" in k.lower() or "token" in k.lower():
                        print(f"  {k}: {v}")

    results = {}

    # Test each context length
    for ctx_name, ctx_cfg in CONTEXTS.items():
        print(f"\n{'─'*60}")
        print(f"  Context: {ctx_name} (~{ctx_cfg['input_tokens']} input tokens)")
        print(f"{'─'*60}")

        # Warmup (1 request, not counted)
        print(f"  [warmup] sending 1 request...")
        try:
            run_non_streaming(ctx_cfg["prompt"], max_tokens=16)
        except:
            pass

        # Get server info after warmup
        si_warmup = get_server_info()

        # Test 1: Non-streaming (prefill + decode combined)
        print(f"  [non-stream] sending request...")
        ns_result = run_non_streaming(ctx_cfg["prompt"], ctx_cfg["max_tokens"])
        print(f"    prompt_tokens={ns_result['prompt_tokens']}, "
              f"completion_tokens={ns_result['completion_tokens']}, "
              f"time={ns_result['total_time']:.2f}s, "
              f"tps={ns_result['tps']:.1f}")

        # Test 2: Streaming (separate TTFT and decode rate)
        print(f"  [streaming] sending request...")
        try:
            s_result = run_streaming(ctx_cfg["prompt"], ctx_cfg["max_tokens"])
            print(f"    TTFT={s_result['ttft_ms']:.0f}ms, "
                  f"prefill_rate={s_result['prefill_rate']:.0f} tok/s, "
                  f"decode_rate={s_result['decode_rate']:.1f} tok/s, "
                  f"total_tokens={s_result['completion_tokens']}")
        except Exception as e:
            s_result = {"error": str(e)}
            print(f"    [ERROR] {e}")

        # Test 3: Concurrent (4-way)
        print(f"  [concurrent] sending 4 parallel requests...")
        try:
            c_result = run_concurrent(ctx_cfg["prompt"], ctx_cfg["max_tokens"], concurrency=4)
            print(f"    aggregate_tps={c_result['aggregate_tps']:.1f}, "
                  f"total_tokens={c_result['total_completion_tokens']}, "
                  f"time={c_result['total_time']:.2f}s")
        except Exception as e:
            c_result = {"error": str(e)}
            print(f"    [ERROR] {e}")

        # Get server info after tests
        si_post = get_server_info()

        results[ctx_name] = {
            "config": ctx_cfg,
            "non_streaming": ns_result,
            "streaming": s_result,
            "concurrent": c_result,
            "server_info_pre": _extract_mem(si_warmup),
            "server_info_post": _extract_mem(si_post),
        }

    # Summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY — {args.label}")
    print(f"{'='*80}")
    print(f"{'Context':<10} {'Prompt':>8} {'Output':>8} {'NonStream':>10} {'TTFT(ms)':>10} {'Prefill':>10} {'Decode':>10} {'4-way':>10}")
    print(f"{'':>10} {'tokens':>8} {'tokens':>8} {'tok/s':>10} {'':>10} {'tok/s':>10} {'tok/s':>10} {'tok/s':>10}")
    print(f"{'─'*80}")

    for ctx_name, r in results.items():
        ns = r["non_streaming"]
        s = r.get("streaming", {})
        c = r.get("concurrent", {})

        prompt_t = ns.get("prompt_tokens", 0)
        out_t = ns.get("completion_tokens", 0)
        ns_tps = ns.get("tps", 0)
        ttft = s.get("ttft_ms", 0) if isinstance(s, dict) else 0
        prefill = s.get("prefill_rate", 0) if isinstance(s, dict) else 0
        decode = s.get("decode_rate", 0) if isinstance(s, dict) else 0
        agg = c.get("aggregate_tps", 0) if isinstance(c, dict) else 0

        print(f"{ctx_name:<10} {prompt_t:>8} {out_t:>8} {ns_tps:>10.1f} {ttft:>10.0f} {prefill:>10.0f} {decode:>10.1f} {agg:>10.1f}")

    print()

    # Save JSON results
    output_file = f"/Users/alexchuang/Documents/flashkv0516/bench_rswa_{args.label}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {output_file}")


def _extract_mem(server_info):
    """Extract memory-related info from server_info."""
    if not server_info:
        return {}
    extracted = {}
    states = server_info.get("internal_states", [])
    for state in states:
        if isinstance(state, dict):
            for k, v in state.items():
                if any(x in k.lower() for x in ["mem", "cache", "token", "radix", "tree"]):
                    extracted[k] = v
    return extracted


if __name__ == "__main__":
    main()
