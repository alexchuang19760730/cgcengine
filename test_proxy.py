#!/usr/bin/env python3
"""Start edge_first_proxy on port 14001 (clean start, no conflicts)"""
import subprocess, os, sys, time, signal

REPO_ROOT = "/Users/alexchuang/Documents/flashkv0516"
PYTHON_BIN = "/Users/alexchuang/.workbuddy/binaries/python/envs/cgc_edge/bin/python3"

# Set environment
env = os.environ.copy()
env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT}/ComputeGraphCompiler-main:{env.get('PYTHONPATH', '')}"
env["EDGE_FIRST_ENABLED"] = "1"
env["CLOUD_URL"] = "http://127.0.0.1:30000"
env["EDGE_FIRST_SPECULATION_MIN_CONFIDENCE"] = "0.84"
env["EDGE_FIRST_ENABLE_WARMUP"] = "1"
env["CGC_FUSION_CONFIG"] = "gate_6_0"

# Start edge_first_proxy on port 14001
log_file = open("/tmp/cgc_edge_first_proxy_14001.log", "w")
proc = subprocess.Popen(
    [PYTHON_BIN, f"{REPO_ROOT}/app/servers/edge_first_proxy.py",
     "--port", "14001", "--host", "127.0.0.1",
     "--cloud-url", "http://127.0.0.1:30000"],
    env=env, stdout=log_file, stderr=subprocess.STDOUT
)
print(f"Edge-First Proxy started on port 14001, PID: {proc.pid}")
time.sleep(3)

# Check health
import urllib.request, json
try:
    req = urllib.request.Request("http://127.0.0.1:14001/health")
    with urllib.request.urlopen(req, timeout=5) as resp:
        health = json.loads(resp.read())
        print(f"Health: {json.dumps(health, indent=2)}")
except Exception as e:
    print(f"Health check failed: {e}")

# Test streaming
print("\n=== Testing streaming request ===")
payload = json.dumps({
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Say hello in 3 words"}],
    "max_tokens": 20,
    "stream": True
}).encode()

req = urllib.request.Request(
    "http://127.0.0.1:14001/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)

t0 = time.time()
first_token_time = None
tokens = []

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        print(f"Response status: {resp.status}")
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            tokens.append(content)
                            if first_token_time is None:
                                first_token_time = time.time()
                                ttft = (first_token_time - t0) * 1000
                                print(f"TTFT: {ttft:.0f}ms")
                                print(f'First token (edge-first): "{content}"')
                except:
                    pass
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code} {e.reason}")
    body = e.read().decode()
    print(f"Response body: {body[:500]}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

total_time = time.time() - t0
print(f"\nTotal time: {total_time*1000:.0f}ms")
print(f"Tokens received: {len(tokens)}")
if tokens:
    print(f'Full text: {" ".join(tokens)}')
if len(tokens) > 1 and first_token_time:
    decode_time = time.time() - first_token_time
    tps = (len(tokens) - 1) / decode_time if decode_time > 0 else 0
    print(f"Decode speed: {tps:.1f} tok/s (excluding first token)")
