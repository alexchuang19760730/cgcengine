#!/usr/bin/env python3
"""
GPU Benchmark v4 - with correct llama-bench parameters.
"""

import os
import sys
import subprocess
import json

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


def run_test(device: str, ngl: int, prompt: int, gen: int, extra_args: list = None) -> dict:
    """运行单个测试."""
    
    cmd = [
        LLAMA_BENCH,
        "-m", MODEL,
        "-p", str(prompt),    # prompt tokens
        "-n", str(gen),       # generation tokens
        "-t", "4",
        "-ngl", str(ngl),
        "-o", "json",         # JSON output
    ]
    
    if device:
        cmd.extend(["-dev", device])
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"  CMD: {' '.join(cmd[:6])}... (dev={device}, ngl={ngl}, p={prompt}, n={gen})")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes
            cwd=BIN_DIR,
            env=env
        )
        
        output = result.stdout
        if not output.strip():
            output = result.stderr
        
        # 提取 JSON
        json_start = output.find('{')
        json_end = output.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = output[json_start:json_end]
            try:
                data = json.loads(json_str)
                return {"success": True, "data": data, "raw": output}
            except:
                return {"success": False, "raw": output, "error": "json_parse"}
        
        return {"success": result.returncode == 0, "raw": output, "code": result.returncode}
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    print("=" * 70)
    print("GPU BENCHMARK v4 - CORRECT PARAMETERS")
    print("=" * 70)
    
    print(f"Model: {os.path.basename(MODEL)} ({os.path.getsize(MODEL)/1024**3:.2f} GB)")
    
    all_results = []
    
    # ========== 测试 1: CPU 基线 ==========
    print("\n[1] CPU Baseline (ngl=0)")
    for p, n in [(512, 128), (1024, 128)]:
        r = run_test("", 0, p, n)
        r["test"] = f"cpu_p{p}_n{n}"
        all_results.append(r)
        
        if r.get("success"):
            data = r.get("data", {})
            print(f"  ✅ p={p}: {json.dumps(data, indent=2)[:500]}")
        else:
            raw = r.get("raw", "")
            print(f"  ❌ p={p}: {raw[:300]}")
    
    # ========== 测试 2: GPU 0 (Intel UHD) ==========
    print("\n[2] GPU 0 - Intel UHD Graphics")
    for ngl in [10, 20]:
        for p, n in [(512, 128), (1024, 128)]:
            r = run_test("0", ngl, p, n)
            r["test"] = f"gpu0_ngl{ngl}_p{p}_n{n}"
            all_results.append(r)
            
            if r.get("success"):
                data = r.get("data", {})
                # 提取指标
                metrics = extract_metrics(data)
                print(f"  ✅ dev=0,ngl={ngl},p={p}: {metrics}")
            else:
                raw = r.get("raw", "")
                # 查找错误信息
                for line in raw.split('\n'):
                    if 'error' in line.lower() or 'Error' in line or 'fail' in line.lower():
                        print(f"  ❌ dev=0,ngl={ngl},p={p}: {line.strip()}")
                        break
                else:
                    print(f"  ❌ dev=0,ngl={ngl},p={p}: {raw[:200]}")
    
    # ========== 测试 3: GPU 1 (NVIDIA MX250) ==========
    print("\n[3] GPU 1 - NVIDIA GeForce MX250")
    for ngl in [10, 20]:
        for p, n in [(512, 128), (1024, 128)]:
            r = run_test("1", ngl, p, n)
            r["test"] = f"gpu1_ngl{ngl}_p{p}_n{n}"
            all_results.append(r)
            
            if r.get("success"):
                data = r.get("data", {})
                metrics = extract_metrics(data)
                print(f"  ✅ dev=1,ngl={ngl},p={p}: {metrics}")
            else:
                raw = r.get("raw", "")
                for line in raw.split('\n'):
                    if 'error' in line.lower() or 'Error' in line or 'fail' in line.lower():
                        print(f"  ❌ dev=1,ngl={ngl},p={p}: {line.strip()}")
                        break
                else:
                    print(f"  ❌ dev=1,ngl={ngl},p={p}: {raw[:200]}")
    
    # ========== 汇总 ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    success_count = sum(1 for r in all_results if r.get("success"))
    print(f"成功: {success_count}/{len(all_results)}")
    
    for r in all_results:
        test_name = r.get("test", "unknown")
        if r.get("success"):
            data = r.get("data", {})
            metrics = extract_metrics(data)
            print(f"  ✅ {test_name}: {metrics}")
        else:
            error = r.get("error", "unknown")
            raw_preview = r.get("raw", "")[:100]
            print(f"  ❌ {test_name}: {error} - {raw_preview}")
    
    # 保存结果
    output_file = r"D:\alex\flashkv0516\benchmark_v4_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n📁 Results: {output_file}")
    
    return 0


def extract_metrics(data: dict) -> str:
    """从 JSON 结果中提取关键指标."""
    metrics = []
    
    # llama-bench 输出格式: 可能在不同层级
    if "results" in data:
        results = data["results"]
        if isinstance(results, list) and len(results) > 0:
            r = results[0]
            if isinstance(r, dict):
                pe = r.get("prompt_eval", {})
                eg = r.get("eval", {})
                
                if isinstance(pe, dict):
                    p_tps = pe.get("tokens_per_second", 0)
                    p_ms = pe.get("ms", 0)
                    if p_tps:
                        metrics.append(f"prefill={p_tps:.1f}tok/s")
                    if p_ms:
                        metrics.append(f"prefill_ms={p_ms:.1f}")
                
                if isinstance(eg, dict):
                    e_tps = eg.get("tokens_per_second", 0)
                    e_ms = eg.get("ms", 0)
                    if e_tps:
                        metrics.append(f"decode={e_tps:.1f}tok/s")
                    if e_ms:
                        metrics.append(f"decode_ms={e_ms:.1f}")
    
    # 其他可能的键
    for key in ["tokens_per_second", "prompt_tokens_per_second", "eval_tokens_per_second"]:
        if key in data:
            val = data[key]
            if isinstance(val, (int, float)) and val > 0:
                metrics.append(f"{key}={val:.1f}")
    
    return ", ".join(metrics) if metrics else str(list(data.keys())[:8])


if __name__ == "__main__":
    sys.exit(main())
