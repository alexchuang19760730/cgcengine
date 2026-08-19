#!/usr/bin/env python3
"""
Run a single llama-bench test with proper output capture.
"""

import os
import sys
import subprocess

LLAMA_BENCH = r"D:\alex\toolchains\llama-build\bin\llama-bench.exe"
MODEL = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
BIN_DIR = r"D:\alex\toolchains\llama-build\bin"
MINGW_BIN = r"D:\alex\toolchains\winlibs-gcc162\mingw64\bin"

env = os.environ.copy()
env["PATH"] = os.pathsep.join([
    BIN_DIR,
    MINGW_BIN,
    r"C:\Windows\System32",
]) + os.pathsep + env.get("PATH", "")

# 测试 1: 不带 -dev 参数 (auto mode)
print("=== Test 1: Auto device mode ===")
cmd = [
    LLAMA_BENCH,
    "-m", MODEL,
    "-p", "50",
    "-n", "20",
    "-t", "4",
    "-o", "json",
]
print(f"Command: {' '.join(cmd)}")

try:
    # 使用 shell=True 来正确处理引号
    result = subprocess.run(
        " ".join(f'"{c}"' if ' ' in c else c for c in cmd),
        capture_output=True,
        text=True,
        timeout=300,
        cwd=BIN_DIR,
        env=env,
        shell=True
    )
    print(f"\nExit code: {result.returncode}")
    print(f"\n=== STDOUT ===")
    print(result.stdout[:3000])
    print(f"\n=== STDERR ===")
    print(result.stderr[:3000])
    
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 70)

# 测试 2: 带 -dev 0 (Intel UHD)
print("\n=== Test 2: Device 0 (Intel UHD) ===")
cmd = [
    LLAMA_BENCH,
    "-m", MODEL,
    "-p", "50",
    "-n", "20",
    "-t", "4",
    "-ngl", "10",
    "-dev", "0",
    "-o", "json",
]
print(f"Command: {' '.join(cmd)}")

try:
    result = subprocess.run(
        " ".join(f'"{c}"' if ' ' in c else c for c in cmd),
        capture_output=True,
        text=True,
        timeout=300,
        cwd=BIN_DIR,
        env=env,
        shell=True
    )
    print(f"\nExit code: {result.returncode}")
    print(f"\n=== STDOUT ===")
    print(result.stdout[:3000])
    print(f"\n=== STDERR ===")
    print(result.stderr[:3000])
    
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 70)

# 测试 3: 使用 cmd.exe
print("\n=== Test 3: Via cmd.exe ===")
test_cmd = f'"{LLAMA_BENCH}" -m "{MODEL}" -p 50 -n 20 -t 4 -o json'
print(f"Command: {test_cmd}")

try:
    result = subprocess.run(
        ["cmd", "/c", test_cmd],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=BIN_DIR,
        env=env
    )
    print(f"\nExit code: {result.returncode}")
    print(f"\n=== STDOUT ===")
    print(result.stdout[:3000])
    print(f"\n=== STDERR ===")
    print(result.stderr[:3000])
    
except Exception as e:
    print(f"Error: {e}")
