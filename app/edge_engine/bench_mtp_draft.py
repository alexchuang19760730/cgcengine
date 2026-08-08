#!/usr/bin/env python3
"""Benchmark: MTP spec decode vs plain generation on Host1.

Runs on Host1, tests:
1. Plain generation (baseline)
2. MTP spec decode (cloud_mtp_draft server)
3. Compares TPS, accept rate, latency
"""
import json
import time
import urllib.request
import sys

# Code prompts for testing
PROMPTS = [
    "def fibonacci(n):",
    "def bubble_sort(arr):",
    "class LinkedList:",
    "import numpy as np\n",
    "def binary_search(arr, target):",
    "async def fetch_data(url):",
    "def train_model(X, y):",
    "export default function App() {",
]

def http_post(url, data, timeout=120):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())

def http_get(url, timeout=10):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())

def main():
    mtp_url = "http://127.0.0.1:30070"
    sglang_url = "http://127.0.0.1:30003"
    max_tokens = 50

    # Check MTP server health
    print("=== Checking MTP server ===")
    try:
        health = http_get(f"{mtp_url}/health")
        print(f"  Status: {health['status']}")
        print(f"  MTP loaded: {health.get('mtp_loaded')}")
        print(f"  Embed shape: {health.get('embed_shape')}")
    except Exception as e:
        print(f"  MTP server not available: {e}")
        sys.exit(1)

    # First, tokenize a prompt using sglang
    print("\n=== Tokenizing test prompt ===")
    tok_result = http_post(f"{sglang_url}/generate", {
        "text": PROMPTS[0],
        "sampling_params": {"max_new_tokens": 0},
    })
    # Get token IDs from logprobs
    meta = tok_result.get("meta_info", {})
    input_lp = meta.get("input_token_logprobs", [])
    prompt_ids = [e[1] for e in input_lp if e and len(e) >= 2]
    print(f"  Prompt: {PROMPTS[0]!r}")
    print(f"  Token IDs: {prompt_ids}")
    print(f"  Token count: {len(prompt_ids)}")

    # Run benchmarks
    print(f"\n{'='*70}")
    print(f"Benchmark: MTP Spec Decode vs Plain Generation")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Prompts: {len(PROMPTS)}")
    print(f"{'='*70}\n")

    all_spec = []
    all_plain = []

    for i, prompt in enumerate(PROMPTS):
        print(f"[{i+1}/{len(PROMPTS)}] {prompt[:50]}...")

        # Tokenize
        tok = http_post(f"{sglang_url}/generate", {
            "text": prompt,
            "sampling_params": {"max_new_tokens": 0},
            "return_logprob": True,
            "logprob_start_len": 0,
        })
        meta = tok.get("meta_info", {})
        input_lp = meta.get("input_token_logprobs", [])
        pids = [e[1] for e in input_lp if e and len(e) >= 2]

        if not pids:
            print(f"  WARNING: Could not tokenize prompt, skipping")
            continue

        # Spec decode
        try:
            spec_result = http_post(f"{mtp_url}/generate", {
                "input_ids": pids,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "num_draft": 4,
            }, timeout=120)
            spec_stats = spec_result.get("stats", {})
            all_spec.append(spec_stats)
            print(f"  [SPEC]  tokens={spec_stats['tokens']}, "
                  f"accept={spec_stats['accept_rate']:.1%}, "
                  f"rounds={spec_stats['rounds']}, "
                  f"tps={spec_stats['tps']:.1f}, "
                  f"mtp={spec_stats['mtp_ms']:.0f}ms, "
                  f"sglang={spec_stats['sglang_ms']:.0f}ms, "
                  f"total={spec_stats['total_ms']:.0f}ms")
        except Exception as e:
            print(f"  [SPEC]  ERROR: {e}")

        # Plain baseline
        try:
            plain_result = http_post(f"{mtp_url}/generate_plain", {
                "input_ids": pids,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }, timeout=120)
            plain_stats = plain_result.get("stats", {})
            all_plain.append(plain_stats)
            print(f"  [PLAIN] tokens={plain_stats['tokens']}, "
                  f"tps={plain_stats['tps']:.1f}, "
                  f"total={plain_stats['total_ms']:.0f}ms")
        except Exception as e:
            print(f"  [PLAIN] ERROR: {e}")

        # Speedup
        if all_spec and all_plain and spec_stats.get('tps', 0) > 0 and plain_stats.get('tps', 0) > 0:
            speedup = spec_stats['tps'] / plain_stats['tps']
            print(f"  [SPEEDUP] {speedup:.2f}x")

        print()

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")

    if all_spec:
        avg_spec_tps = sum(s['tps'] for s in all_spec) / len(all_spec)
        total_accepted = sum(s['accepted'] for s in all_spec)
        total_draft = sum(s['accepted'] + s['rejected'] for s in all_spec)
        avg_accept = total_accepted / total_draft if total_draft > 0 else 0
        avg_mtp_ms = sum(s['mtp_ms'] for s in all_spec) / len(all_spec)
        avg_sglang_ms = sum(s['sglang_ms'] for s in all_spec) / len(all_spec)
        avg_rounds = sum(s['rounds'] for s in all_spec) / len(all_spec)

        print(f"\nMTP Spec Decode:")
        print(f"  avg_tps: {avg_spec_tps:.1f}")
        print(f"  avg_accept_rate: {avg_accept:.1%}")
        print(f"  avg_rounds: {avg_rounds:.1f}")
        print(f"  avg_mtp_time: {avg_mtp_ms:.0f}ms")
        print(f"  avg_sglang_time: {avg_sglang_ms:.0f}ms")

    if all_plain:
        avg_plain_tps = sum(s['tps'] for s in all_plain) / len(all_plain)
        print(f"\nPlain Generation:")
        print(f"  avg_tps: {avg_plain_tps:.1f}")

    if all_spec and all_plain:
        avg_spec_tps = sum(s['tps'] for s in all_spec) / len(all_spec)
        avg_plain_tps = sum(s['tps'] for s in all_plain) / len(all_plain)
        if avg_plain_tps > 0:
            print(f"\nSpeedup: {avg_spec_tps / avg_plain_tps:.2f}x")

    # MTP server stats
    try:
        health = http_get(f"{mtp_url}/health")
        print(f"\nMTP Server Stats:")
        print(json.dumps(health.get('stats', {}), indent=2))
    except:
        pass

if __name__ == "__main__":
    main()
