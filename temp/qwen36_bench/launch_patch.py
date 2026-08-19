#!/usr/bin/env python3
import subprocess, os
env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONPATH"] = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-repack"
logf = "/Users/alexchuang/Documents/flashkv0516/temp/qwen36_bench/patch_lmhead.log"
with open(logf, "w") as f:
    p = subprocess.Popen(
        ["/Users/alexchuang/Documents/flashkv0516/.venv-cgc/bin/python3",
         "/Users/alexchuang/Documents/flashkv0516/temp/qwen36_bench/patch_lmhead.py"],
        stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=env)
print(f"launched pid={p.pid} log={logf}")
