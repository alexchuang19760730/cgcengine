#!/usr/bin/env python3
"""
GPU Benchmark Suite for llama.cpp with Vulkan Backend.

使用修复后的环境变量运行 llama-bench 进行 GPU 基准测试。
"""

import os
import sys
import subprocess
import time
import json

# 路径配置
LLAMA_BENCH = r"D:\alex\toolchains\llama-build\bin\llama-bench.exe"
BIN_DIR = r"D:\alex\toolchains\llama-build\bin"
MINGW_BIN = r"D:\alex\toolchains\winlibs-gcc162\mingw64\bin"
MODEL = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

# 构建正确的环境
env = os.environ.copy()
env["PATH"] = os.pathsep.join([
    BIN_DIR,
    MINGW_BIN,
    r"C:\Windows\System32",
]) + os.pathsep + env.get("PATH", "")


def run_benchmark(
    model: str,
    ctx_size: int = 4096,
    device: str = "cpu",
    ngl: int = 0,
    prompt_tokens: int = 512,
    gen_tokens: int = 256,
    threads: int = 4
) -> dict:
    """运行 llama-bench 基准测试."""
    
    # 构建命令 (llama-bench 使用 --prompt-tokens 和 --generation-tokens)
    cmd = [
        LLAMA_BENCH,
        "-m", model,
        "--ctx-size", str(ctx_size),
        "--prompt-tokens", str(prompt_tokens),
        "--generation-tokens", str(gen_tokens),
        "-t", str(threads),
        "--no-mmap",
        "--output", "json",  # JSON 输出便于解析
    ]
    
    if device == "vulkan0":
        cmd.extend(["-dev", "0", "-ngl", str(ngl)])
    elif device == "vulkan1":
        cmd.extend(["-dev", "1", "-ngl", str(ngl)])
    
    print(f"  Running: {' '.join(cmd[:8])}...")
    print(f"    Device: {device}, Context: {ctx_size}, NGL: {ngl}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=BIN_DIR,
            env=env
        )
        
        # 尝试解析 JSON 输出
        output = result.stdout
        if not output:
            output = result.stderr
        
        # 查找 JSON 部分
        json_start = output.find('{')
        json_end = output.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = output[json_start:json_end]
            try:
                data = json.loads(json_str)
                return {
                    "success": True,
                    "device": device,
                    "ctx_size": ctx_size,
                    "ngl": ngl,
                    "data": data,
                }
            except json.JSONDecodeError:
                pass
        
        # 如果没有 JSON，返回原始输出
        return {
            "success": result.returncode == 0,
            "device": device,
            "ctx_size": ctx_size,
            "ngl": ngl,
            "raw_output": output[:2000],
            "returncode": result.returncode,
        }
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    print("=" * 70)
    print("LLAMA.CPP GPU BENCHMARK SUITE")
    print("=" * 70)
    
    # 检查模型
    if not os.path.exists(MODEL):
        print(f"❌ Model not found: {MODEL}")
        return 1
    
    model_size_gb = os.path.getsize(MODEL) / (1024**3)
    print(f"✅ Model: {os.path.basename(MODEL)} ({model_size_gb:.2f} GB)")
    
    # 列出 Vulkan 设备
    print("\n🔍 Listing Vulkan devices...")
    try:
        result = subprocess.run(
            [LLAMA_BENCH, "--list-devices"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=BIN_DIR,
            env=env
        )
        print(result.stdout or result.stderr)
    except Exception as e:
        print(f"  Error: {e}")
    
    all_results = []
    
    # ========== 测试 1: CPU 基线 ==========
    print("\n" + "=" * 70)
    print("TEST 1: CPU BASELINE")
    print("=" * 70)
    
    for ctx in [512, 1024, 2048]:
        result = run_benchmark(
            MODEL,
            ctx_size=ctx,
            device="cpu",
            prompt_tokens=min(ctx//2, 256),
            gen_tokens=min(ctx//2, 256),
        )
        all_results.append(result)
        if result.get("success"):
            print(f"  ✅ ctx={ctx}: completed")
        else:
            print(f"  ⚠️  ctx={ctx}: {result.get('raw_output', '')[:100]}")
        time.sleep(2)
    
    # ========== 测试 2: GPU 0 (Intel UHD) ==========
    print("\n" + "=" * 70)
    print("TEST 2: GPU 0 - INTEL UHD GRAPHICS")
    print("=" * 70)
    
    for ngl in [10, 20]:  # 尝试不同的 GPU offload 层数
        for ctx in [512, 1024]:
            result = run_benchmark(
                MODEL,
                ctx_size=ctx,
                device="vulkan0",
                ngl=ngl,
                prompt_tokens=min(ctx//2, 256),
                gen_tokens=min(ctx//2, 256),
            )
            all_results.append(result)
            if result.get("success"):
                print(f"  ✅ ctx={ctx}, ngl={ngl}: completed")
            else:
                error = result.get("error", result.get("raw_output", "")[:100])
                print(f"  ⚠️  ctx={ctx}, ngl={ngl}: {error}")
            time.sleep(3)
    
    # ========== 测试 3: GPU 1 (NVIDIA MX250) ==========
    print("\n" + "=" * 70)
    print("TEST 3: GPU 1 - NVIDIA GEFORCE MX250")
    print("=" * 70)
    
    for ngl in [10, 20]:
        for ctx in [512, 1024]:
            result = run_benchmark(
                MODEL,
                ctx_size=ctx,
                device="vulkan1",
                ngl=ngl,
                prompt_tokens=min(ctx//2, 256),
                gen_tokens=min(ctx//2, 256),
            )
            all_results.append(result)
            if result.get("success"):
                print(f"  ✅ ctx={ctx}, ngl={ngl}: completed")
            else:
                error = result.get("error", result.get("raw_output", "")[:100])
                print(f"  ⚠️  ctx={ctx}, ngl={ngl}: {error}")
            time.sleep(3)
    
    # ========== 汇总结果 ==========
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    
    # 分析 JSON 结果
    successful = [r for r in all_results if r.get("success")]
    print(f"\n成功测试: {len(successful)}/{len(all_results)}")
    
    for r in successful:
        device = r.get("device", "unknown")
        ctx = r.get("ctx_size", 0)
        ngl = r.get("ngl", 0)
        data = r.get("data", {})
        
        print(f"\n  Device: {device}, Context: {ctx}, NGL: {ngl}")
        if isinstance(data, dict):
            # 提取关键指标
            if "results" in data:
                for res in data["results"]:
                    if isinstance(res, dict):
                        prompt_eval = res.get("prompt_eval", {})
                        eval_result = res.get("eval", {})
                        
                        prompt_tps = prompt_eval.get("tokens_per_second", 0)
                        eval_tps = eval_result.get("tokens_per_second", 0)
                        
                        print(f"    Prefill: {prompt_tps:.1f} tokens/s")
                        print(f"    Decode: {eval_tps:.1f} tokens/s")
            else:
                # 直接在顶层查找
                prompt_tps = data.get("prompt_tokens_per_second", 0)
                eval_tps = data.get("tokens_per_second", 0)
                if prompt_tps or eval_tps:
                    print(f"    Prefill: {prompt_tps:.1f} tokens/s")
                    print(f"    Decode: {eval_tps:.1f} tokens/s")
                else:
                    # 显示所有键
                    for key in list(data.keys())[:10]:
                        val = data[key]
                        if isinstance(val, (int, float, str)):
                            print(f"      {key}: {val}")
    
    # 保存完整结果
    output_file = r"D:\alex\flashkv0516\benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n📁 Full results saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
