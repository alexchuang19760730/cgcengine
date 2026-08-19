#!/usr/bin/env python3
"""
Vulkan Diagnostics Script

检查 GPU 信息, 测试 CPU fallback, 逐步增加 GPU layer 数量
"""

import subprocess
import sys
import os

LLAMA_BENCH = r"D:\alex\toolchains\llama-build\bin\llama-bench.exe"
MODEL_SMALL = r"D:\alex\flashkv0516\models\qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_LARGE = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

def run_cmd(cmd, timeout=30):
    env = os.environ.copy()
    env["PATH"] = "D:\\alex\\toolchains\\winlibs-gcc162\\mingw64\\bin;D:\\alex\\toolchains\\VulkanSDK\\Bin;" + env.get("PATH", "")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout", -1
    except Exception as e:
        return "", str(e), -1

def main():
    print("=" * 80)
    print("VULKAN DIAGNOSTICS")
    print("=" * 80)

    # Test 1: List devices
    print("\n📊 Test 1: List available devices")
    print("-" * 60)
    stdout, stderr, rc = run_cmd([LLAMA_BENCH, "--list-devices"])
    print(stdout)
    if stderr:
        print(f"Stderr: {stderr[:200]}")

    # Test 2: CPU-only (ngl=0) - should work
    print("\n📊 Test 2: CPU-only baseline (ngl=0)")
    print("-" * 60)
    stdout, stderr, rc = run_cmd([
        LLAMA_BENCH, "-m", MODEL_SMALL,
        "-p", "128", "-n", "32",
        "-r", "1", "-t", "4",
        "-fa", "1", "-ngl", "0",
        "--no-warmup", "-o", "md"
    ], timeout=120)

    if rc == 0:
        print("✅ CPU test passed!")
        for line in stdout.split('\n'):
            if '|' in line and ('pp' in line.lower() or 'tg' in line.lower()):
                print(f"  {line}")
    else:
        print(f"❌ CPU test failed (rc={rc})")
        print(f"  Stderr: {stderr[:300]}")

    # Test 3: Try with small GPU layer count (ngl=1)
    print("\n📊 Test 3: Minimal GPU (ngl=1, device=0)")
    print("-" * 60)
    stdout, stderr, rc = run_cmd([
        LLAMA_BENCH, "-m", MODEL_SMALL,
        "-p", "128", "-n", "32",
        "-r", "1", "-t", "4",
        "-fa", "1", "-ngl", "1",
        "-dev", "0",
        "--no-warmup", "-o", "md"
    ], timeout=120)

    if rc == 0:
        print("✅ GPU test (ngl=1) passed!")
        for line in stdout.split('\n'):
            if '|' in line and ('pp' in line.lower() or 'tg' in line.lower()):
                print(f"  {line}")
    else:
        print(f"❌ GPU test failed (rc={rc})")
        print(f"  Stderr: {stderr[:300]}")

    # Test 4: List Vulkan extensions
    print("\n📊 Test 4: Check Vulkan SDK")
    print("-" * 60)
    vulkaninfo = "D:\\alex\\toolchains\\VulkanSDK\\Bin\\vulkaninfo.exe"
    if os.path.exists(vulkaninfo):
        stdout, stderr, rc = run_cmd([vulkaninfo, "--summary"], timeout=30)
        print("Vulkan Info Summary:")
        # Filter relevant info
        for line in stdout.split('\n'):
            line_lower = line.lower()
            if any(kw in line_lower for kw in ['device', 'gpu', 'intel', 'nvidia', 'extension', '16bit', 'subgroup']):
                print(f"  {line.strip()}")
    else:
        print("  vulkaninfo not found")

    # Test 5: Check Vulkan runtime
    print("\n📊 Test 5: Check Vulkan runtime DLL")
    print("-" * 60)
    import glob
    search_paths = [
        "D:\\alex\\toolchains\\VulkanSDK\\Bin",
        "C:\\Windows\\System32",
    ]
    for path in search_paths:
        dll_path = os.path.join(path, "vulkan-1.dll")
        if os.path.exists(dll_path):
            size = os.path.getsize(dll_path)
            print(f"  ✅ Found: {dll_path} ({size/1024:.0f} KB)")
        else:
            print(f"  ❌ Not found: {dll_path}")

    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)
    print("""
💡 Recommendations:
  1. If CPU test fails: Check basic llama.cpp installation
  2. If GPU test fails (ngl=0 works, ngl=1 fails):
     - Intel UHD + MX250 may not support required Vulkan features
     - Try updating GPU drivers
     - Consider CPU-only inference for this hardware
  3. If Vulkan SDK missing: Install latest SDK from LunarG
""")

if __name__ == "__main__":
    main()
