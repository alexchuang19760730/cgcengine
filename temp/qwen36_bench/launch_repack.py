#!/usr/bin/env python3
"""Launch qwen36 repack fully detached. Usage: launch_repack.py <bits> <out_dir> <log>"""
import subprocess, os, sys

bits = sys.argv[1] if len(sys.argv) > 1 else "3"
out = sys.argv[2] if len(sys.argv) > 2 else "/Volumes/AlexZhuang/qwen36-r3.gturbo"
log = sys.argv[3] if len(sys.argv) > 3 else "/Users/alexchuang/Documents/flashkv0516/temp/qwen36_bench/repack_r3.log"

env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"
cmd = [
    "/Users/alexchuang/Documents/flashkv0516/.venv-cgc/bin/python3",
    "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-repack/repack_qwen36.py",
    "/Volumes/AlexZhuang/qwen36-hf", out, f"--bits", bits,
]
with open(log, "w") as f:
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                         start_new_session=True, env=env)
print(f"launched bits={bits} pid={p.pid} out={out} log={log}")
