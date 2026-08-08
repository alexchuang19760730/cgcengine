#!/usr/bin/env python3
"""R-SWA A/B benchmark: measure prefill/decode speed at short/medium/long contexts.

Tests:
1. Short context (~50 tokens)  - R-SWA cap should NOT trigger (< 132)
2. Medium context (~500 tokens) - R-SWA cap triggers (> 132)
3. Long context (~2000 tokens)  - R-SWA cap significant (> 132)
4. Very long context (~4000 tokens) - R-SWA cap very significant

For each: measure TTFT, prefill time, decode rate, total time.
Also measures VRAM via SSH nvidia-smi before/after.
"""
import urllib.request
import json
import time
import subprocess
import sys

URL = "http://39.106.118.206:30001/v1/chat/completions"
MODEL = "/data/models/gemma-4-26b-a4b-it"
SSH_CMD = [
    "sshpass", "-p", "Gen@song@2026622",
    "ssh", "-o", "StrictHostKeyChecking=no",
    "-o", "PreferredAuthentications=password",
    "-o", "PubkeyAuthentication=no",
    "-o", "ConnectTimeout=30",
    "root@39.106.118.206"
]

def get_gpu_memory():
    """Get GPU memory usage via SSH."""
    try:
        cmd = SSH_CMD + ["nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if "," in line:
                idx, mem = line.strip().split(",")
                gpus.append({"gpu": int(idx.strip()), "mem_mb": int(mem.strip())})
        return gpus
    except Exception as e:
        return [{"error": str(e)}]

def flush_cache():
    """Flush sglang KV cache."""
    try:
        req = urllib.request.Request(
            "http://39.106.118.206:30001/flush_cache",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({}).encode()
        )
        urllib.request.urlopen(req, timeout=10)
    except:
        pass

def bench_context(label, prompt_text, max_tokens=200):
    """Benchmark a single context length."""
    # Flush cache before each test
    flush_cache()
    time.sleep(0.5)
    
    # Get VRAM before
    mem_before = get_gpu_memory()
    
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    
    req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    
    t0 = time.monotonic()
    first_token = None
    total_tokens = 0
    
    resp = urllib.request.urlopen(req, timeout=120)
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            try:
                obj = json.loads(line[5:].strip())
                choices = obj.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content", "")
                    if content:
                        if first_token is None:
                            first_token = time.monotonic() - t0
                        total_tokens += 1
            except:
                pass
    
    total_time = time.monotonic() - t0
    
    # Get VRAM after
    time.sleep(0.3)
    mem_after = get_gpu_memory()
    
    # Calculate metrics
    ttft_ms = first_token * 1000 if first_token else 0
    decode_time = total_time - first_token if first_token else 0
    decode_tokens = total_tokens - 1 if first_token else total_tokens
    decode_rate = decode_tokens / decode_time if decode_time > 0 else 0
    
    # Estimate prompt tokens (rough: 1 token ~ 4 chars)
    prompt_tokens_est = len(prompt_text) // 4
    prefill_time = first_token if first_token else 0
    prefill_rate = prompt_tokens_est / prefill_time if prefill_time > 0 else 0
    
    # VRAM delta
    mem_delta = 0
    if mem_before and mem_after and "error" not in mem_before[0] and "error" not in mem_after[0]:
        total_before = sum(g["mem_mb"] for g in mem_before)
        total_after = sum(g["mem_mb"] for g in mem_after)
        mem_delta = total_after - total_before
    
    result = {
        "label": label,
        "prompt_tokens_est": prompt_tokens_est,
        "completion_tokens": total_tokens,
        "ttft_ms": round(ttft_ms, 1),
        "prefill_time_s": round(prefill_time, 3),
        "prefill_rate_toks": round(prefill_rate, 1),
        "decode_time_s": round(decode_time, 3),
        "decode_rate_tps": round(decode_rate, 1),
        "total_time_s": round(total_time, 3),
        "vram_total_mb": sum(g.get("mem_mb", 0) for g in mem_after) if mem_after else 0,
        "vram_delta_mb": mem_delta,
    }
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Prompt tokens (est): {prompt_tokens_est}")
    print(f"  Completion tokens:   {total_tokens}")
    print(f"  TTFT:                {ttft_ms:.1f} ms")
    print(f"  Prefill time:        {prefill_time:.3f}s ({prefill_rate:.1f} tok/s)")
    print(f"  Decode time:         {decode_time:.3f}s ({decode_rate:.1f} tok/s)")
    print(f"  Total time:          {total_time:.3f}s")
    print(f"  VRAM total:          {result['vram_total_mb']} MB")
    print(f"  VRAM delta:          {mem_delta:+d} MB")
    
    return result

def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "test"
    
    print(f"\n{'#'*60}")
    print(f"# R-SWA A/B Benchmark: {label}")
    print(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    results = []
    
    # Test 1: Short context (~50 tokens) - R-SWA should NOT trigger
    short_prompt = "Write a Python function to sort a list using quicksort. Include comments."
    results.append(bench_context("Short (~50 tok)", short_prompt, max_tokens=100))
    
    # Test 2: Medium context (~500 tokens) - R-SWA triggers
    med_prompt = "Write a detailed Python tutorial covering variables, data types, control flow, functions, classes, modules, file I/O, exception handling, decorators, and generators. " * 5
    results.append(bench_context("Medium (~500 tok)", med_prompt, max_tokens=200))
    
    # Test 3: Long context (~2000 tokens) - R-SWA significant
    long_prompt = "Write a comprehensive guide to Python programming. " * 100
    results.append(bench_context("Long (~2000 tok)", long_prompt, max_tokens=200))
    
    # Test 4: Very long context (~4000 tokens) - R-SWA very significant
    vlong_prompt = ("Write a detailed tutorial on how to implement a REST API in Python using Flask. "
                     "Include code examples for CRUD operations, authentication, error handling, and testing. "
                     "Also cover database integration, middleware, blueprints, and deployment. " * 50)
    results.append(bench_context("Very Long (~4000 tok)", vlong_prompt, max_tokens=200))
    
    # Summary table
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY: {label}")
    print(f"{'='*80}")
    print(f"  {'Context':<20} {'Prompt':>8} {'TTFT':>8} {'Prefill':>10} {'Decode':>10} {'Total':>8} {'VRAM':>10}")
    print(f"  {'':20} {'toks':>8} {'ms':>8} {'tok/s':>10} {'tok/s':>10} {'s':>8} {'MB':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    for r in results:
        print(f"  {r['label']:<20} {r['prompt_tokens_est']:>8} {r['ttft_ms']:>8.1f} {r['prefill_rate_toks']:>10.1f} {r['decode_rate_tps']:>10.1f} {r['total_time_s']:>8.3f} {r['vram_total_mb']:>10}")
    
    # Save results
    out_file = f"/Users/alexchuang/Documents/flashkv0516/rswa_ab_{label}.json"
    with open(out_file, "w") as f:
        json.dump({"label": label, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}, f, indent=2)
    print(f"\n  Results saved to {out_file}")

if __name__ == "__main__":
    main()
