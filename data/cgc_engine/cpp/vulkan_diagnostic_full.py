#!/usr/bin/env python3
"""
Vulkan Diagnostic & GPU Inference Test Script
============================================
1. Check Vulkan runtime availability
2. List available GPU devices
3. Run CPU baseline benchmark
4. Test GPU with progressive layer counts (ngl=1, 2, 4, 8, 16, 32)
5. Test with fallback if GPU fails
6. Generate performance report
"""

import subprocess
import sys
import os
import json
import time
import struct
from pathlib import Path

LLAMA_BENCH = r"D:\alex\toolchains\llama-build\bin\llama-bench.exe"
LLAMA_CLI = r"D:\alex\toolchains\llama-build\bin\llama-cli.exe"
MODEL_SMALL = r"D:\alex\flashkv0516\models\qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_MED = r"D:\alex\flashkv0516\models\gemma4_gguf\gemma-4-26B-A4B-it-UD-IQ3_S.gguf"
VULKAN_SDK_BIN = r"D:\alex\toolchains\VulkanSDK\Bin"
WINLIBS_BIN = r"D:\alex\toolchains\winlibs-gcc162\mingw64\bin"

def get_env():
    env = os.environ.copy()
    env["PATH"] = f"{WINLIBS_BIN};{VULKAN_SDK_BIN};{env.get('PATH', '')}"
    return env

def run_cmd(cmd, timeout=120, env=None):
    if env is None:
        env = get_env()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1
    except Exception as e:
        return "", str(e), -1

def check_vulkan_runtime():
    print("=" * 70)
    print("📋 STEP 1: Vulkan Runtime Check")
    print("=" * 70)

    paths = [
        (VULKAN_SDK_BIN, "SDK Bin"),
        (r"C:\Windows\System32", "System32"),
    ]

    found = []
    for path, label in paths:
        dll = os.path.join(path, "vulkan-1.dll")
        if os.path.exists(dll):
            size = os.path.getsize(dll)
            print(f"  ✅ [{label}] {dll} ({size//1024} KB)")
            found.append(dll)
        else:
            print(f"  ❌ [{label}] {dll} not found")

    return len(found) > 0

def check_vulkaninfo():
    print("\n" + "=" * 70)
    print("📋 STEP 2: vulkaninfo --summary")
    print("=" * 70)

    vulkaninfo = os.path.join(VULKAN_SDK_BIN, "vulkaninfo.exe")
    if not os.path.exists(vulkaninfo):
        print("  ❌ vulkaninfo.exe not found")
        return []

    stdout, stderr, rc = run_cmd([vulkaninfo, "--summary"], timeout=10)

    devices = []
    current_device = None
    for line in stdout.split("\n"):
        line_lower = line.lower()
        if "gpu" in line_lower or "device" in line_lower:
            line = line.strip()
            if line:
                print(f"  {line}")
                if "device" in line_lower and "type" not in line_lower:
                    current_device = line
                    devices.append(current_device)

    if stderr and "error" in stderr.lower():
        print(f"  ⚠️  Stderr: {stderr[:200]}")

    return devices

def list_devices():
    print("\n" + "=" * 70)
    print("📋 STEP 3: llama-bench --list-devices")
    print("=" * 70)

    stdout, stderr, rc = run_cmd([LLAMA_BENCH, "--list-devices"], timeout=15)
    print(stdout)
    if stderr:
        print(f"  Stderr: {stderr[:300]}")
    return stdout

def run_cpu_benchmark():
    print("\n" + "=" * 70)
    print("📋 STEP 4: CPU Baseline Benchmark (ngl=0)")
    print("=" * 70)

    if not os.path.exists(MODEL_SMALL):
        print(f"  ⚠️  Model not found: {MODEL_SMALL}")
        return None

    cmd = [
        LLAMA_BENCH, "-m", MODEL_SMALL,
        "-p", "128", "-n", "32",
        "-r", "1", "-t", "4",
        "-fa", "1", "-ngl", "0",
        "--no-warmup", "-o", "md"
    ]

    print(f"  CMD: {' '.join(cmd)}")
    stdout, stderr, rc = run_cmd(cmd, timeout=120)

    result = {"success": rc == 0, "rc": rc, "stdout": stdout, "stderr": stderr}

    if rc == 0:
        print("  ✅ CPU benchmark PASSED")
        for line in stdout.split("\n"):
            if "|" in line and ("pp" in line.lower() or "tg" in line.lower()):
                print(f"  {line.strip()}")
    else:
        print(f"  ❌ CPU benchmark FAILED (rc={rc})")
        print(f"  Stderr: {stderr[:400]}")

    return result

def run_gpu_progressive():
    print("\n" + "=" * 70)
    print("📋 STEP 5: Progressive GPU Layer Testing")
    print("=" * 70)

    if not os.path.exists(MODEL_SMALL):
        print(f"  ⚠️  Model not found: {MODEL_SMALL}")
        return []

    ngl_values = [1, 2, 4, 8, 16]
    results = []

    for ngl in ngl_values:
        print(f"\n  --- Testing ngl={ngl} ---")
        cmd = [
            LLAMA_BENCH, "-m", MODEL_SMALL,
            "-p", "128", "-n", "32",
            "-r", "1", "-t", "4",
            "-fa", "1", "-ngl", str(ngl),
            "--no-warmup", "-o", "md"
        ]

        stdout, stderr, rc = run_cmd(cmd, timeout=120)
        success = rc == 0
        results.append({"ngl": ngl, "success": success, "rc": rc, "stdout": stdout, "stderr": stderr})

        if success:
            print(f"  ✅ ngl={ngl}: PASSED")
            for line in stdout.split("\n"):
                if "|" in line and ("pp" in line.lower() or "tg" in line.lower()):
                    print(f"    {line.strip()}")
        else:
            print(f"  ❌ ngl={ngl}: FAILED (rc={rc})")
            err_short = stderr[:200].replace("\n", " ")
            print(f"    Error: {err_short}")
            if rc == 3221225477:
                print(f"    💥 Access violation (0xC0000005) - Vulkan driver issue")
            break

    return results

def run_cli_inference():
    print("\n" + "=" * 70)
    print("📋 STEP 6: Real Inference Test (llama-cli)")
    print("=" * 70)

    if not os.path.exists(MODEL_SMALL):
        print(f"  ⚠️  Model not found: {MODEL_SMALL}")
        return

    cmd = [
        LLAMA_CLI, "-m", MODEL_SMALL,
        "-p", "Hello, what are the key features of a Mixture of Experts (MoE) model?",
        "-n", "64",
        "-t", "4",
        "-ngl", "0",
        "-fa", "1",
        "--no-warmup"
    ]

    print(f"  Running inference...")
    stdout, stderr, rc = run_cmd(cmd, timeout=60)

    if rc == 0:
        print("  ✅ Inference PASSED")
        lines = stdout.strip().split("\n")
        for line in lines[:10]:
            print(f"    {line}")
        if len(lines) > 10:
            print(f"    ... ({len(lines)} lines total)")
    else:
        print(f"  ❌ Inference FAILED (rc={rc})")
        print(f"  Stderr: {stderr[:400]}")

def generate_report(vulkan_ok, devices, cpu_result, gpu_results):
    print("\n" + "=" * 70)
    print("📊 DIAGNOSTIC REPORT")
    print("=" * 70)

    report = {
        "vulkan_runtime": vulkan_ok,
        "devices_found": devices,
        "cpu_baseline": "PASS" if cpu_result and cpu_result["success"] else "FAIL",
        "gpu_tests": [],
        "recommendations": []
    }

    for r in gpu_results:
        status = "PASS" if r["success"] else "FAIL"
        report["gpu_tests"].append({"ngl": r["ngl"], "status": status, "rc": r["rc"]})

    print(f"\n  Vulkan Runtime: {'✅ OK' if vulkan_ok else '❌ MISSING'}")
    print(f"  Devices: {len(devices)} found")
    print(f"  CPU Baseline: {report['cpu_baseline']}")

    if gpu_results:
        passed = sum(1 for r in gpu_results if r["success"])
        total = len(gpu_results)
        print(f"  GPU Tests: {passed}/{total} passed")

        for r in gpu_results:
            icon = "✅" if r["success"] else "❌"
            print(f"    ngl={r['ngl']}: {icon} rc={r['rc']}")

    print("\n  Recommendations:")
    if not vulkan_ok:
        report["recommendations"].append("Install Vulkan SDK from LunarG (https://vulkan.lunarg.com)")
        print("    1. Install Vulkan SDK from LunarG")
    if cpu_result and not cpu_result["success"]:
        report["recommendations"].append("Check llama.cpp installation - CPU test failed")
        print("    2. Check llama.cpp installation")
    if gpu_results and not any(r["success"] for r in gpu_results):
        report["recommendations"].append("GPU driver may not support required Vulkan features (16-bit storage, subgroup ops). Try CPU-only mode or update drivers.")
        print("    3. GPU driver may need update - try CPU-only mode (ngl=0)")
        print("    4. For Intel UHD + MX250: consider CPU-only inference for development")

    report_path = os.path.join(os.path.dirname(LLAMA_BENCH), "vulkan_diagnostic_report.json")
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  📄 Report saved: {report_path}")
    except Exception:
        pass

    return report

def main():
    print("=" * 70)
    print("VULKAN GPU DIAGNOSTIC & INFERENCE TEST")
    print("=" * 70)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if not os.path.exists(LLAMA_BENCH):
        print(f"❌ llama-bench not found: {LLAMA_BENCH}")
        print("   Please build llama.cpp with Vulkan support first.")
        return 1

    vulkan_ok = check_vulkan_runtime()
    devices = check_vulkaninfo()
    list_devices()
    cpu_result = run_cpu_benchmark()
    gpu_results = run_gpu_progressive()
    run_cli_inference()
    generate_report(vulkan_ok, devices, cpu_result, gpu_results)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
