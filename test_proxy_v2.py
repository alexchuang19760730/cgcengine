#!/usr/bin/env python3
"""Restart proxy with speculation_threshold=0.0 and run tests"""
import subprocess, os, sys, time, signal, json, urllib.request

REPO_ROOT = "/Users/alexchuang/Documents/flashkv0516"
PYTHON_BIN = "/Users/alexchuang/.workbuddy/binaries/python/envs/cgc_edge/bin/python3"

# Kill existing process on port 14001
try:
    result = subprocess.run(["lsof", "-ti", ":14001"], capture_output=True, text=True)
    pids = result.stdout.strip().split()
    for pid in pids:
        if pid:
            os.kill(int(pid), signal.SIGTERM)
            print(f"Killed PID {pid}")
    time.sleep(1)
except:
    pass

# Start with speculation_threshold=0.0
env = os.environ.copy()
env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT}/ComputeGraphCompiler-main:{env.get('PYTHONPATH', '')}"
env["EDGE_FIRST_ENABLED"] = "1"
env["CLOUD_URL"] = "http://127.0.0.1:30000"
env["EDGE_FIRST_SPECULATION_MIN_CONFIDENCE"] = "0.0"
env["EDGE_FIRST_ENABLE_WARMUP"] = "1"
env["CGC_FUSION_CONFIG"] = "gate_6_0"

log_file = open("/tmp/cgc_edge_first_proxy_14001.log", "w")
proc = subprocess.Popen(
    [PYTHON_BIN, f"{REPO_ROOT}/app/servers/edge_first_proxy.py",
     "--port", "14001", "--host", "127.0.0.1",
     "--cloud-url", "http://127.0.0.1:30000"],
    env=env, stdout=log_file, stderr=subprocess.STDOUT
)
print(f"Proxy restarted (speculation_threshold=0.0), PID: {proc.pid}")
time.sleep(3)

# Verify health
try:
    req = urllib.request.Request("http://127.0.0.1:14001/health")
    with urllib.request.urlopen(req, timeout=5) as resp:
        health = json.loads(resp.read())
        print(f"Health: confidence_threshold={health.get('edge_speculation_min_confidence')}")
except Exception as e:
    print(f"Health check failed: {e}")

# Test function
def test_stream(prompt, label):
    url = "http://127.0.0.1:14001/v1/chat/completions"
    payload = json.dumps({
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 30,
        "stream": True
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    first_token_time = None
    tokens = []
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
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
                                    print(f"[{label}] TTFT: {ttft:.0f}ms | First: '{content}'")
                    except:
                        pass
    except Exception as e:
        print(f"[{label}] Error: {e}")
        return
    
    total = time.time() - t0
    if len(tokens) > 1 and first_token_time:
        decode_t = time.time() - first_token_time
        tps = (len(tokens) - 1) / decode_t if decode_t > 0 else 0
        print(f"[{label}] Total: {total*1000:.0f}ms | Tokens: {len(tokens)} | Decode: {tps:.1f} tok/s")
    else:
        print(f"[{label}] Total: {total*1000:.0f}ms | Tokens: {len(tokens)}")
    print(f"[{label}] Text: {' '.join(tokens)[:120]}")
    print()

# Run tests
print("\n=== Test 1: cgc_g6_run style prompt ===")
test_stream("You are a software engineer working on a python codebase. Identify all concurrency bugs and race conditions.", "cgc_g6_run")

print("=== Test 2: Simple prompt ===")
test_stream("Hello, how are you?", "simple")

print("=== Test 3: Write code prompt ===")
test_stream("Write a Python function to sort a list", "write_code")
