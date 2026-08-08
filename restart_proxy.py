#!/usr/bin/env python3
"""Restart edge_first_proxy on Mac - clean restart"""
import subprocess, os, sys, time, signal

REPO_ROOT = "/Users/alexchuang/Documents/flashkv0516"
PYTHON_BIN = "/Users/alexchuang/.workbuddy/binaries/python/envs/cgc_edge/bin/python3"

# Kill existing processes on port 14000
try:
    result = subprocess.run(["lsof", "-ti", ":14000"], capture_output=True, text=True)
    pids = result.stdout.strip().split()
    for pid in pids:
        if pid:
            os.kill(int(pid), signal.SIGTERM)
            print(f"Killed PID {pid} on port 14000")
    time.sleep(1)
except Exception as e:
    print(f"No existing process to kill: {e}")

# Set environment
env = os.environ.copy()
env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT}/ComputeGraphCompiler-main:{env.get('PYTHONPATH', '')}"
env["EDGE_FIRST_ENABLED"] = "1"
env["CLOUD_URL"] = "http://127.0.0.1:30000"
env["EDGE_FIRST_SPECULATION_MIN_CONFIDENCE"] = "0.84"
env["EDGE_FIRST_ENABLE_WARMUP"] = "1"
env["EDGE_FIRST_WARMUP_TTL_SEC"] = "600"
env["CGC_FUSION_CONFIG"] = "gate_6_0"

# Start edge_first_proxy
log_file = open("/tmp/cgc_edge_first_proxy_mac.log", "w")
proc = subprocess.Popen(
    [PYTHON_BIN, f"{REPO_ROOT}/app/servers/edge_first_proxy.py",
     "--port", "14000", "--host", "127.0.0.1",
     "--cloud-url", "http://127.0.0.1:30000"],
    env=env, stdout=log_file, stderr=subprocess.STDOUT
)
print(f"Edge-First Proxy started, PID: {proc.pid}")
time.sleep(3)

# Check health
import urllib.request, json
try:
    req = urllib.request.Request("http://127.0.0.1:14000/health")
    with urllib.request.urlopen(req, timeout=5) as resp:
        health = json.loads(resp.read())
        print(f"Health check: {json.dumps(health, indent=2)}")
except Exception as e:
    print(f"Health check failed: {e}")
    # Read log
    log_file.flush()
    with open("/tmp/cgc_edge_first_proxy_mac.log") as f:
        print("Log:", f.read()[-500:])
