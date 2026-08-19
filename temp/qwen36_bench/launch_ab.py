#!/usr/bin/env python3
import subprocess, os, sys
mode = sys.argv[1] if len(sys.argv) > 1 else "repack"
env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"
env["EXPERT_MODE"] = mode
env["PYTHONPATH"] = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-repack"
logf = f"/tmp/q36_ab_{mode}.log"
with open(logf, "w") as f:
    p = subprocess.Popen(["/Users/alexchuang/Documents/flashkv0516/.venv-cgc/bin/python3",
                          "/Users/alexchuang/Documents/flashkv0516/temp/qwen36_bench/q36_bf16_ab.py"],
                         stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=env)
print(f"launched {mode} pid={p.pid} log={logf}")
