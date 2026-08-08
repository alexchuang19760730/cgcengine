#!/usr/bin/env python3
"""
R-SWA Memory A/B Test: Precisely measure KV cache memory growth per context length.

Measures GPU VRAM before and after sending requests at different context lengths,
with cache flushed between each test.
"""

import urllib.request
import json
import time
import subprocess
import sys
import argparse

URL = "http://39.106.118.206:30001"
MODEL = "/data/models/gemma-4-26b-a4b-it"
SSH_CMD = [
    "sshpass", "-p", "Gen@song@2026622",
    "ssh", "-o", "StrictHostKeyChecking=no",
    "-o", "PreferredAuthentications=password",
    "-o", "PubkeyAuthentication=no",
    "-o", "ConnectTimeout=15",
    "root@39.106.118.206",
]

# Context lengths to test (target input tokens)
TEST_CONFIGS = [
    {"name": "empty",    "prompt": "",                            "max_tokens": 1},
    {"name": "128tok",   "prompt": None, "target_tokens": 128,    "max_tokens": 64},
    {"name": "512tok",   "prompt": None, "target_tokens": 512,    "max_tokens": 64},
    {"name": "1024tok",  "prompt": None, "target_tokens": 1024,   "max_tokens": 64},
    {"name": "2048tok",  "prompt": None, "target_tokens": 2048,   "max_tokens": 64},
    {"name": "4096tok",  "prompt": None, "target_tokens": 4096,   "max_tokens": 64},
    {"name": "8192tok",  "prompt": None, "target_tokens": 8192,   "max_tokens": 64},
]


def generate_prompt(target_tokens):
    """Generate a prompt of approximately target_tokens."""
    base = ("The history of computing spans centuries of innovation, from the abacus to quantum computing. "
            "Each era built upon previous discoveries, creating increasingly sophisticated machines. "
            "Programming languages evolved from machine code to high-level languages, enabling complex software. "
            "The internet revolutionized communication, while mobile computing brought processing to everyone's pocket. "
            "Artificial intelligence now pushes the boundaries of what computers can accomplish. ")
    target_chars = target_tokens * 4
    result = base
    while len(result) < target_chars:
        result += base
    return result[:target_chars]


def get_gpu_memory(retries=3):
    """Get GPU memory usage via SSH nvidia-smi (with retry)."""
    cmd = SSH_CMD + ["nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"]
    for attempt in range(retries):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            gpu_mems = []
            for line in result.stdout.strip().split("\n"):
                if "," in line:
                    idx, mem = line.strip().split(",")
                    gpu_mems.append({"gpu": int(idx.strip()), "mem_used_mb": int(mem.strip())})
            if gpu_mems:
                return gpu_mems
        except subprocess.TimeoutExpired:
            print(f"    [SSH retry {attempt+1}/{retries}] timeout, waiting 10s...")
            time.sleep(10)
        except Exception as e:
            print(f"    [SSH retry {attempt+1}/{retries}] error: {e}")
            time.sleep(5)
    return []


def flush_cache():
    """Flush sglang KV cache."""
    try:
        req = urllib.request.Request(f"{URL}/flush_cache", method="POST",
                                     data=json.dumps({}).encode(),
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def send_request(prompt, max_tokens, stream=False):
    """Send a request and return timing + token info."""
    messages = [{"role": "user", "content": prompt}] if prompt else [{"role": "user", "content": "hi"}]
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")

    t0 = time.monotonic()

    if stream:
        first_token = None
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
                            if first_token is None:
                                first_token = time.monotonic()
                            total_tokens += 1
                    usage = obj.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                except:
                    pass
        total_time = time.monotonic() - t0
        ttft = (first_token - t0) * 1000 if first_token else 0
        decode_time = total_time - (first_token - t0) if first_token else 0
        decode_tokens = total_tokens - 1 if total_tokens > 0 else 0
        decode_rate = decode_tokens / decode_time if decode_time > 0 else 0
        prefill_time = (first_token - t0) if first_token else total_time
        prefill_rate = prompt_tokens / prefill_time if prefill_time > 0 and prompt_tokens > 0 else 0

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_tokens,
            "ttft_ms": ttft,
            "prefill_time_s": prefill_time,
            "prefill_rate": prefill_rate,
            "decode_time_s": decode_time,
            "decode_rate": decode_rate,
            "total_time_s": total_time,
        }
    else:
        resp = urllib.request.urlopen(req, timeout=300)
        data = json.loads(resp.read())
        elapsed = time.monotonic() - t0
        usage = data.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_time_s": elapsed,
            "tps": usage.get("completion_tokens", 0) / elapsed if elapsed > 0 else 0,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline", help="Label (baseline/rswa)")
    args = parser.parse_args()

    # Pre-generate prompts
    for cfg in TEST_CONFIGS:
        if cfg.get("target_tokens"):
            cfg["prompt"] = generate_prompt(cfg["target_tokens"])

    print(f"\n{'='*80}")
    print(f"  R-SWA Memory A/B Test — {args.label}")
    print(f"{'='*80}\n")

    results = []

    for cfg in TEST_CONFIGS:
        name = cfg["name"]
        prompt = cfg.get("prompt", "")
        max_tokens = cfg["max_tokens"]

        print(f"{'─'*60}")
        print(f"  Test: {name}")

        # Step 1: Flush cache
        flush_result = flush_cache()
        time.sleep(1)

        # Step 2: Measure GPU memory BEFORE
        mem_before = get_gpu_memory()
        mem_before_total = sum(g["mem_used_mb"] for g in mem_before[:4])  # GPU 0-3
        print(f"  [before] GPU mem (0-3): {mem_before_total} MB")

        # Step 3: Send streaming request (measure TTFT + prefill + decode)
        s_result = send_request(prompt, max_tokens, stream=True)
        print(f"  [stream]  prompt={s_result['prompt_tokens']} tok, "
              f"TTFT={s_result['ttft_ms']:.0f}ms, "
              f"prefill={s_result['prefill_rate']:.0f} tok/s, "
              f"decode={s_result['decode_rate']:.1f} tok/s, "
              f"output={s_result['completion_tokens']} tok")

        # Step 4: Measure GPU memory AFTER (while KV cache still has the data)
        mem_after = get_gpu_memory()
        mem_after_total = sum(g["mem_used_mb"] for g in mem_after[:4])
        mem_delta = mem_after_total - mem_before_total
        print(f"  [after]  GPU mem (0-3): {mem_after_total} MB (delta: {mem_delta:+d} MB)")

        # Step 5: Also run non-streaming for combined tps
        flush_cache()
        time.sleep(0.5)
        ns_result = send_request(prompt, max_tokens, stream=False)
        print(f"  [nonstr]  {ns_result['tps']:.1f} tok/s, "
              f"{ns_result['completion_tokens']} tok / {ns_result['total_time_s']:.2f}s")

        results.append({
            "name": name,
            "config": cfg,
            "streaming": s_result,
            "non_streaming": ns_result,
            "mem_before_mb": mem_before_total,
            "mem_after_mb": mem_after_total,
            "mem_delta_mb": mem_delta,
            "mem_per_gpu": [{"gpu": g["gpu"], "before": g["mem_used_mb"]} for g in mem_before[:4]],
            "mem_per_gpu_after": [{"gpu": g["gpu"], "after": g["mem_used_mb"]} for g in mem_after[:4]],
        })

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY — {args.label}")
    print(f"{'='*80}")
    print(f"{'Context':<12} {'Prompt':>8} {'TTFT':>8} {'Prefill':>10} {'Decode':>10} {'MemDelta':>10} {'PerToken':>10}")
    print(f"{'':12} {'tokens':>8} {'(ms)':>8} {'(tok/s)':>10} {'(tok/s)':>10} {'(MB)':>10} {'(KB/tok)':>10}")
    print(f"{'─'*80}")

    for r in results:
        s = r["streaming"]
        pt = s["prompt_tokens"]
        ttft = s["ttft_ms"]
        prefill = s["prefill_rate"]
        decode = s["decode_rate"]
        delta = r["mem_delta_mb"]
        per_token = (delta * 1024 / pt) if pt > 0 and delta > 0 else 0

        print(f"{r['name']:<12} {pt:>8} {ttft:>8.0f} {prefill:>10.0f} {decode:>10.1f} {delta:>+10d} {per_token:>10.0f}")

    # Save results
    output_file = f"/Users/alexchuang/Documents/flashkv0516/bench_rswa_mem_{args.label}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
