#!/usr/bin/env python3
"""Local E2E benchmark: edge_first_proxy + MTP draft + Hermes vs direct llama-server.

Architecture:
  Mac M4
  ├── llama-server (port 30000) — Qwen2.5-0.5B Q4_K_M GGUF (target/"cloud" model)
  ├── edge_first_proxy (port 30001) — first token prediction + proxy
  └── benchmark client — measures TTFT + decode tok/s

Usage:
  python3 bench_local_e2e.py
"""
import asyncio
import aiohttp
import time
import json
import subprocess
import sys
import os
import signal

# === Config ===
GGUF_PATH = os.path.expanduser("~/models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf")
LLAMA_SERVER_PORT = 30000
PROXY_PORT = 30001
PYTHON = os.path.expanduser("~/.workbuddy/binaries/python/envs/default/bin/python3")
LLAMA_SERVER = "/opt/homebrew/bin/llama-server"
PROXY_SCRIPT = os.path.join(os.path.dirname(__file__), "app", "servers", "edge_first_proxy.py")

# Test prompts — code completion + chat
TEST_PROMPTS = [
    ("code_def", "def calculate_sum(a, b):"),
    ("code_class", "class UserProfile:"),
    ("code_import", "import json\nimport os\nimport sys\n"),
    ("code_func", "async def fetch_data(url):"),
    ("chat_write", "Write a Python function to reverse a string"),
    ("chat_fix", "Fix this bug: IndexError: list index out of range"),
    ("chat_explain", "Explain how quicksort works"),
    ("code_json", 'const config = {\n  apiKey: "'),
]

NUM_WARMUP = 2
NUM_ROUNDS = 3
MAX_TOKENS = 30

async def measure_request(session, url, prompt, label, max_tokens=MAX_TOKENS):
    """Send streaming request, measure TTFT + decode speed."""
    payload = {
        "model": "qwen2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }

    t0 = time.perf_counter()
    first_token_time = None
    tokens = []
    spec_status = None
    predicted_token = None

    try:
        async with session.post(f"{url}/v1/chat/completions", json=payload,
                                timeout=aiohttp.ClientTimeout(total=60)) as resp:
            async for line in resp.content:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                if first_token_time is None:
                    first_token_time = time.perf_counter()

                try:
                    chunk = json.loads(data_str)
                except:
                    continue

                # Check speculation markers
                if "x-cgc-speculation" in chunk:
                    spec_status = chunk.get("x-cgc-speculation")
                    predicted_token = chunk.get("x-cgc-predicted")

                choices = chunk.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content", "")
                    if content:
                        tokens.append(content)
    except Exception as e:
        return {"label": label, "error": str(e)}

    total_time = time.perf_counter() - t0
    ttft = (first_token_time - t0) if first_token_time else total_time
    num_tokens = len(tokens)
    decode_time = total_time - ttft if num_tokens > 1 else 0
    decode_tps = (num_tokens - 1) / decode_time if decode_time > 0 and num_tokens > 1 else 0

    return {
        "label": label,
        "ttft_ms": round(ttft * 1000, 1),
        "total_ms": round(total_time * 1000, 1),
        "tokens": num_tokens,
        "decode_tps": round(decode_tps, 1),
        "spec_status": spec_status,
        "predicted": predicted_token,
        "first_token": tokens[0] if tokens else "",
    }


async def run_benchmark(url, label):
    """Run benchmark against a given URL."""
    print(f"\n{'='*60}")
    print(f"  Benchmark: {label}")
    print(f"  URL: {url}")
    print(f"{'='*60}")

    results = []

    async with aiohttp.ClientSession() as session:
        # Warmup
        for i in range(NUM_WARMUP):
            await measure_request(session, url, "Hello", f"warmup-{i}", max_tokens=5)

        # Test prompts
        for name, prompt in TEST_PROMPTS:
            round_results = []
            for r in range(NUM_ROUNDS):
                r_data = await measure_request(session, url, prompt, f"{name}_r{r}")
                round_results.append(r_data)

            # Average
            valid = [r for r in round_results if "error" not in r]
            if valid:
                avg_ttft = sum(r["ttft_ms"] for r in valid) / len(valid)
                avg_tps = sum(r["decode_tps"] for r in valid) / len(valid)
                avg_total = sum(r["total_ms"] for r in valid) / len(valid)
                spec_hits = sum(1 for r in valid if r.get("spec_status") == "hit")
                spec_misses = sum(1 for r in valid if r.get("spec_status") == "miss")
                results.append({
                    "name": name,
                    "prompt": prompt[:40],
                    "avg_ttft_ms": round(avg_ttft, 1),
                    "avg_decode_tps": round(avg_tps, 1),
                    "avg_total_ms": round(avg_total, 1),
                    "spec_hits": spec_hits,
                    "spec_misses": spec_misses,
                    "rounds": len(valid),
                    "sample_first": valid[0].get("first_token", ""),
                })
                print(f"  {name:15s} | TTFT: {avg_ttft:7.1f}ms | decode: {avg_tps:6.1f} tok/s | "
                      f"total: {avg_total:7.1f}ms | spec: {spec_hits}H/{spec_misses}M | "
                      f"first: '{valid[0].get('first_token', '')[:20]}'")
            else:
                print(f"  {name:15s} | ERROR: {round_results[0].get('error', 'unknown')}")

    return results


def start_llama_server():
    """Start llama-server with Qwen2.5-0.5B Q4_K_M."""
    cmd = [
        LLAMA_SERVER,
        "-m", GGUF_PATH,
        "--host", "127.0.0.1",
        "--port", str(LLAMA_SERVER_PORT),
        "-c", "4096",
        "-t", "4",
        "-ngl", "99",  # offload all layers to Metal GPU
        "--temp", "0.0",
    ]
    print(f"Starting llama-server on port {LLAMA_SERVER_PORT}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def start_proxy():
    """Start edge_first_proxy."""
    env = os.environ.copy()
    env["EDGE_MODEL_PATH"] = GGUF_PATH
    env["EDGE_FIRST_ENABLED"] = "1"
    env["CLOUD_URL"] = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"
    env["DSV4_TOKENIZER_PATH"] = ""
    env["EDGE_FIRST_SPECULATION_MIN_CONFIDENCE"] = "0.01"  # Force-enable speculation
    # Fix import path: app package needs repo root in PYTHONPATH
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(PROXY_SCRIPT))))
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        PYTHON, PROXY_SCRIPT,
        "--port", str(PROXY_PORT),
        "--cloud-url", f"http://127.0.0.1:{LLAMA_SERVER_PORT}",
        "--host", "127.0.0.1",
        "--active-model", "qwen3vl",  # closest to Qwen2.5 in registry
    ]

    print(f"Starting edge_first_proxy on port {PROXY_PORT}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            cwd=repo_root)
    return proc


def wait_for_server(url, timeout=30):
    """Wait for server to be ready."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{url}/health")
            urllib.request.urlopen(req, timeout=2)
            return True
        except:
            try:
                req = urllib.request.Request(f"{url}/v1/models")
                urllib.request.urlopen(req, timeout=2)
                return True
            except:
                time.sleep(0.5)
    return False


def main():
    # Start llama-server
    llama_proc = start_llama_server()
    if not wait_for_server(f"http://127.0.0.1:{LLAMA_SERVER_PORT}", timeout=30):
        print("ERROR: llama-server failed to start")
        llama_proc.kill()
        return

    print("llama-server ready!")

    # Start edge_first_proxy
    proxy_proc = start_proxy()
    time.sleep(3)  # Give proxy time to start
    if not wait_for_server(f"http://127.0.0.1:{PROXY_PORT}", timeout=20):
        print("WARNING: proxy health check failed, checking stderr...")
        try:
            out = proxy_proc.stdout.read1(4096).decode() if proxy_proc.stdout else ""
            print(f"  Proxy output: {out[:500]}")
        except:
            pass
        time.sleep(2)

    print("edge_first_proxy ready!")

    try:
        # Run benchmarks
        asyncio.run(run_full_benchmark())
    finally:
        # Cleanup
        print("\nCleaning up...")
        proxy_proc.terminate()
        llama_proc.terminate()
        try:
            proxy_proc.wait(timeout=5)
            llama_proc.wait(timeout=5)
        except:
            proxy_proc.kill()
            llama_proc.kill()


async def run_full_benchmark():
    """Run all benchmarks."""
    # 1. Direct to llama-server (baseline)
    direct_results = await run_benchmark(
        f"http://127.0.0.1:{LLAMA_SERVER_PORT}", "Direct (llama-server baseline)"
    )

    # 2. Through edge_first_proxy
    proxy_results = await run_benchmark(
        f"http://127.0.0.1:{PROXY_PORT}", "Through edge_first_proxy"
    )

    # 3. Summary comparison
    print(f"\n{'='*60}")
    print("  SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Prompt':15s} | {'Direct TTFT':>12s} | {'Proxy TTFT':>12s} | {'Direct TPS':>11s} | {'Proxy TPS':>11s} | {'Spec Hit':>9s}")
    print("-" * 80)

    for d, p in zip(direct_results, proxy_results):
        spec_rate = f"{p['spec_hits']}/{p['spec_hits']+p['spec_misses']}" if p['spec_hits']+p['spec_misses'] > 0 else "N/A"
        print(f"{d['name']:15s} | {d['avg_ttft_ms']:10.1f}ms | {p['avg_ttft_ms']:10.1f}ms | "
              f"{d['avg_decode_tps']:8.1f}tps | {p['avg_decode_tps']:8.1f}tps | {spec_rate:>9s}")

    # Averages
    if direct_results and proxy_results:
        avg_direct_ttft = sum(r["avg_ttft_ms"] for r in direct_results) / len(direct_results)
        avg_proxy_ttft = sum(r["avg_ttft_ms"] for r in proxy_results) / len(proxy_results)
        avg_direct_tps = sum(r["avg_decode_tps"] for r in direct_results) / len(direct_results)
        avg_proxy_tps = sum(r["avg_decode_tps"] for r in proxy_results) / len(proxy_results)
        total_spec_hits = sum(r["spec_hits"] for r in proxy_results)
        total_spec_misses = sum(r["spec_misses"] for r in proxy_results)
        total_spec = total_spec_hits + total_spec_misses

        print("-" * 80)
        print(f"{'AVERAGE':15s} | {avg_direct_ttft:10.1f}ms | {avg_proxy_ttft:10.1f}ms | "
              f"{avg_direct_tps:8.1f}tps | {avg_proxy_tps:8.1f}tps | "
              f"{f'{total_spec_hits}/{total_spec}' if total_spec > 0 else 'N/A':>9s}")

        # TTFT improvement
        if avg_direct_ttft > 0:
            improvement = ((avg_direct_ttft - avg_proxy_ttft) / avg_direct_ttft) * 100
            print(f"\n  TTFT improvement: {improvement:+.1f}% (proxy vs direct)")
        if avg_direct_tps > 0 and avg_proxy_tps > 0:
            tps_change = ((avg_proxy_tps - avg_direct_tps) / avg_direct_tps) * 100
            print(f"  Decode speed change: {tps_change:+.1f}% (proxy vs direct)")
        if total_spec > 0:
            print(f"  Speculation hit rate: {total_spec_hits/total_spec*100:.1f}% ({total_spec_hits}/{total_spec})")


if __name__ == "__main__":
    main()
