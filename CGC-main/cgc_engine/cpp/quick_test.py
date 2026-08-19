#!/usr/bin/env python3
"""
Quick test with correct llama-bench parameters.
"""

import os
import sys
import subprocess

LLAMA_BENCH = r"D:\alex\toolchains\llama-build\bin\llama-bench.exe"
MODEL = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
BIN_DIR = r"D:\alex\toolchains\llama-build\bin"
MINGW_BIN = r"D:\alex\toolchains\winlibs-gcc162\mingw64\bin"

# 构建正确的环境
env = os.environ.copy()
env["PATH"] = os.pathsep.join([
    BIN_DIR,
    MINGW_BIN,
    r"C:\Windows\System32",
]) + os.pathsep + env.get("PATH", "")

# 查看完整帮助
print("=== llama-bench 帮助 ===")
result = subprocess.run(
    [LLAMA_BENCH, "--help"],
    capture_output=True,
    text=True,
    timeout=10,
    cwd=BIN_DIR,
    env=env
)
print(result.stdout)

# 尝试一个简单的测试 (使用 -p 和 -n)
print("\n=== 简单测试 ===")
print("Running with -p 50 -n 20...")
cmd = [
    LLAMA_BENCH,
    "-m", MODEL,
    "-p", "50",     # prompt tokens
    "-n", "20",     # generation tokens
    "-t", "4",      # threads
    "-c", "512",    # context size
    "--no-mmap"
]
print(f"Command: {' '.join(cmd)}")

result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    timeout=120,
    cwd=BIN_DIR,
    env=env
)
print(f"Exit code: {result.returncode}")
print(f"Output:\n{result.stdout[:2000]}")
if result.stderr:
    print(f"Error:\n{result.stderr[:1000]}")
