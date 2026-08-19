#!/usr/bin/env python3
"""
Diagnose llama-bench crash issues.
"""

import os
import sys
import subprocess

LLAMA_BENCH = r"D:\alex\toolchains\llama-build\bin\llama-bench.exe"
MODEL = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
MODEL_SMALL = None  # 尝试找一个小模型
BIN_DIR = r"D:\alex\toolchains\llama-build\bin"
MINGW_BIN = r"D:\alex\toolchains\winlibs-gcc162\mingw64\bin"

env = os.environ.copy()
env["PATH"] = os.pathsep.join([
    BIN_DIR,
    MINGW_BIN,
    r"C:\Windows\System32",
]) + os.pathsep + env.get("PATH", "")

# 查找可用模型
print("=== 查找模型文件 ===")
model_dir = r"D:\alex\flashkv0516\models"
gguf_dir = os.path.join(model_dir, "gguf")

if os.path.exists(gguf_dir):
    files = [f for f in os.listdir(gguf_dir) if f.endswith('.gguf')]
    for f in files:
        path = os.path.join(gguf_dir, f)
        size_gb = os.path.getsize(path) / (1024**3)
        print(f"  {f}: {size_gb:.2f} GB")
else:
    # 直接在 models 目录查找
    if os.path.exists(model_dir):
        files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
        for f in files:
            path = os.path.join(model_dir, f)
            size_gb = os.path.getsize(path) / (1024**3)
            print(f"  {f}: {size_gb:.2f} GB")

print("\n" + "=" * 70)

# 测试 1: 查看可用模型列表
print("\n=== 测试 1: 列出设备 ===")
result = subprocess.run(
    [LLAMA_BENCH, "--list-devices"],
    capture_output=True,
    text=True,
    timeout=10,
    cwd=BIN_DIR,
    env=env
)
print(f"Exit: {result.returncode}")
print(f"Stdout: {result.stdout}")
print(f"Stderr: {result.stderr}")

# 测试 2: CPU only (ngl=0)
print("\n=== 测试 2: CPU Only (ngl=0, -dev none) ===")
cmd = [LLAMA_BENCH, "-m", MODEL, "-p", "50", "-n", "20", "-t", "4", "-ngl", "0", "-o", "json"]
print(f"Running: {' '.join(cmd)}")

result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    timeout=120,
    cwd=BIN_DIR,
    env=env
)
print(f"Exit: {result.returncode}")
print(f"Stdout ({len(result.stdout)} chars): {result.stdout[:500]}")
print(f"Stderr ({len(result.stderr)} chars): {result.stderr[:500]}")

# 测试 3: 使用 Vulkan0 设备
print("\n=== 测试 3: Vulkan0 (Intel UHD) ===")
cmd = [LLAMA_BENCH, "-m", MODEL, "-p", "50", "-n", "20", "-t", "4", "-ngl", "5", "-dev", "Vulkan0", "-o", "json"]
print(f"Running: {' '.join(cmd)}")

result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    timeout=120,
    cwd=BIN_DIR,
    env=env
)
print(f"Exit: {result.returncode}")
print(f"Stdout ({len(result.stdout)} chars): {result.stdout[:500]}")
print(f"Stderr ({len(result.stderr)} chars): {result.stderr[:500]}")

# 测试 4: 使用 Vulkan1 设备 (NVIDIA MX250)
print("\n=== 测试 4: Vulkan1 (NVIDIA MX250) ===")
cmd = [LLAMA_BENCH, "-m", MODEL, "-p", "50", "-n", "20", "-t", "4", "-ngl", "5", "-dev", "Vulkan1", "-o", "json"]
print(f"Running: {' '.join(cmd)}")

result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    timeout=120,
    cwd=BIN_DIR,
    env=env
)
print(f"Exit: {result.returncode}")
print(f"Stdout ({len(result.stdout)} chars): {result.stdout[:500]}")
print(f"Stderr ({len(result.stderr)} chars): {result.stderr[:500]}")

# 测试 5: 尝试小模型 (如果存在)
print("\n=== 测试 5: 检查小模型 ===")
# 查找 llama.cpp 自带的测试模型
test_model_dir = r"D:\alex\flashkv0516\CGC-main\Backend\Llama.cpp\llama.cpp\models"
if os.path.exists(test_model_dir):
    for root, dirs, files in os.walk(test_model_dir):
        for f in files:
            if f.endswith('.gguf'):
                path = os.path.join(root, f)
                size_mb = os.path.getsize(path) / (1024**2)
                print(f"  Found: {path} ({size_mb:.1f} MB)")

# 测试 6: 检查崩溃日志
print("\n=== 诊断崩溃 (exit code 3221225477) ===")
print("0xC0000005 = ACCESS_VIOLATION (空指针或越界访问)")
print("可能原因:")
print("  1. Vulkan 设备内存不足无法分配大模型")
print("  2. GGUF 文件解析错误导致数据访问越界")
print("  3. Vulkan 驱动问题")
print()
print("建议:")
print("  - 先用 CPU-only 模式测试 (ngl=0)")
print("  - 或者使用更小的模型测试")
print("  - 或者使用 --no-mmap 参数禁用内存映射")
