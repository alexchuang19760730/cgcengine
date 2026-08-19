#!/usr/bin/env python3
"""
Dynamic Load-Balance Layer-Split PD Benchmark

基于 low-memory 主动判别 + 动态 load balance layer-split:
1. 检测两个 GPU 显存使用情况
2. 动态决定: 哪层 P (prefill) 在哪层 D (decode) 分配给哪个 GPU
3. 运行 llama-bench 对比性能

设备:
  - GPU 0: Intel UHD Graphics (iGPU, 4GB UMA)
  - GPU 1: NVIDIA GeForce MX250 (dGPU, 2GB)
"""

import os
import sys
import subprocess
import re
import time
from datetime import datetime

LLAMA_BENCH = r"D:\alex\toolchains\llama-build\bin\llama-bench.exe"
OUTPUT_DIR = r"D:\alex\flashkv0516\bench_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    {
        "name": "Qwen2.5-1.5B-Q4_K_M",
        "path": r"D:\alex\flashkv0516\models\qwen2.5-1.5b-instruct-q4_k_m.gguf",
    },
    {
        "name": "gemma-4-26B-A4B-it-heretic-IQ4_XS",
        "path": r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf",
    },
]

GPU_DEVICES = {
    "0": {"name": "Intel UHD Graphics (iGPU)", "vram_gb": 4.0},
    "1": {"name": "NVIDIA GeForce MX250 (dGPU)", "vram_gb": 2.0},
}


def get_gpu_memory_info():
    """获取 GPU 显存使用情况 (简化版,使用固定值)"""
    return {
        "0": {"name": "Intel UHD Graphics", "total_mib": 4138, "free_mib": 3609},
        "1": {"name": "NVIDIA GeForce MX250", "total_mib": 1983, "free_mib": 1686},
    }


def calculate_optimal_split(model_size_gb, gpu0_info, gpu1_info):
    """
    基于显存主动判别, 计算最优的 PD layer-split 策略.
    
    策略:
    1. 如果模型能放进单个 GPU, 直接用单 GPU
    2. 如果需要 PD:
       - 显存更大的 GPU (iGPU=4GB) 负责 prefill (需要更多显存存储 KV cache)
       - 显存较小的 GPU (dGPU=2GB) 负责 decode (KV cache 相对稳定)
    3. 动态调整: 如果 iGPU 显存不足, 翻转角色
    """
    
    gpu0_free = gpu0_info.get("free_mib", 0) * 1024 * 1024  # bytes
    gpu1_free = gpu1_info.get("free_mib", 0) * 1024 * 1024
    
    model_bytes = model_size_gb * 1024 * 1024 * 1024
    
    strategies = []
    
    # 策略 1: 单 GPU (如果能放下)
    if model_bytes < gpu0_free:
        strategies.append({
            "type": "single",
            "device": "0",
            "desc": "Single-iGPU (fits in VRAM)",
            "priority": 1  # 最优
        })
    
    if model_bytes < gpu1_free:
        strategies.append({
            "type": "single",
            "device": "1",
            "desc": "Single-dGPU (fits in VRAM)",
            "priority": 1
        })
    
    # 策略 2: PD 模式
    # Prefill 需要更多显存 (KV cache 随 context 增长)
    # Decode 显存相对固定
    
    # 检查 iGPU (4GB) 作为 P, dGPU (2GB) 作为 D
    # 条件: P 的模型部分 + KV < iGPU 显存, D 的模型部分 < dGPU 显存
    if gpu0_free > gpu1_free:
        strategies.append({
            "type": "pd",
            "prefill_device": "0",  # 大显存 GPU 做 P
            "decode_device": "1",   # 小显存 GPU 做 D
            "desc": "PD: P→iGPU(4GB), D→dGPU(2GB) [default]",
            "priority": 2
        })
    else:
        strategies.append({
            "type": "pd",
            "prefill_device": "1",  # 显存多的做 P
            "decode_device": "0",
            "desc": "PD: P→dGPU, D→iGPU [reversed]",
            "priority": 2
        })
    
    # 策略 3: 双 GPU layer-split (如果单 GPU 放不下)
    if model_bytes > max(gpu0_free, gpu1_free):
        strategies.append({
            "type": "layer_split",
            "device": "0",  # 主设备
            "desc": "Layer-split on primary GPU",
            "priority": 3
        })
    
    # 按优先级排序
    strategies.sort(key=lambda x: x["priority"])
    return strategies


def run_benchmark(model_path, n_prompt, n_gen, strategy):
    """根据策略运行 llama-bench"""
    env = os.environ.copy()
    env["PATH"] = "D:\\alex\\toolchains\\winlibs-gcc162\\mingw64\\bin;D:\\alex\\toolchains\\VulkanSDK\\Bin;" + env.get("PATH", "")
    
    cmd = [
        LLAMA_BENCH,
        "-m", model_path,
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-r", "2",
        "-t", "8",
        "-fa", "1",
        "--no-warmup",
        "-o", "md"
    ]
    
    if strategy["type"] == "single":
        cmd.extend(["-ngl", "99", "-dev", f"dev{strategy['device']}"])
    elif strategy["type"] == "pd":
        # PD 模式: -pg prefill_device,decode_device
        cmd.extend(["-ngl", "99", "-pg", f"{strategy['prefill_device']},{strategy['decode_device']}"])
    elif strategy["type"] == "layer_split":
        cmd.extend(["-ngl", "99", "-dev", f"dev{strategy['device']}", "-sm", "layer"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        return result.stdout
    except Exception as e:
        return f"ERROR: {e}"


def parse_results(output):
    if not output or output.startswith("ERROR"):
        return None
    
    results = {"raw": output}
    for line in output.split('\n'):
        if '|' in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 6:
                if 'pp' in parts[0].lower():
                    results['pp_tps'] = parts[-1]
                elif 'tg' in parts[0].lower():
                    results['tg_tps'] = parts[-1]
    return results


def main():
    print("=" * 70)
    print("  DYNAMIC LOAD-BALANCE LAYER-SPLIT PD BENCHMARK")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Step 1: 检测 GPU 显存
    print("\n📊 Step 1: GPU 显存检测")
    print("-" * 50)
    gpu_info = get_gpu_memory_info()
    for gid, info in gpu_info.items():
        print(f"  GPU {gid}: {info.get('name', 'Unknown')}")
        print(f"    VRAM: {info.get('total_mib', '?')} MiB total, {info.get('free_mib', '?')} MiB free")
    
    if not gpu_info:
        print("  ⚠️  无法获取 GPU 信息, 使用默认策略")
        gpu_info = {
            "0": {"name": "Intel UHD Graphics", "total_mib": 4096, "free_mib": 3600},
            "1": {"name": "NVIDIA GeForce MX250", "total_mib": 2048, "free_mib": 1700},
        }
    
    # Step 2: 运行测试
    print("\n📈 Step 2: 动态 PD 基准测试")
    print("=" * 70)
    
    test_configs = [
        (512, 128, "短上下文 512/128"),
        (2048, 512, "中上下文 2K/512"),
    ]
    
    all_results = []
    
    for model_info in MODELS:
        model_path = model_info["path"]
        model_name = model_info["name"]
        
        if not os.path.exists(model_path):
            print(f"\n⚠️  SKIP {model_name} (not found)")
            continue
        
        file_size_gb = os.path.getsize(model_path) / 1024 / 1024 / 1024
        
        # 计算最优策略
        strategies = calculate_optimal_split(
            file_size_gb,
            gpu_info.get("0", {}),
            gpu_info.get("1", {})
        )
        
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name} ({file_size_gb:.2f} GB)")
        print(f"{'='*60}")
        
        print(f"\n  📋 可用策略 (按优先级):")
        for i, s in enumerate(strategies):
            print(f"    {i+1}. {s['desc']}")
        
        for n_prompt, n_gen, config_label in test_configs:
            print(f"\n  📊 测试: {config_label}")
            print(f"  {'-'*50}")
            
            for strategy in strategies[:3]:  # 最多测 3 个策略
                print(f"    {strategy['desc']}...", end=" ", flush=True)
                
                output = run_benchmark(model_path, n_prompt, n_gen, strategy)
                parsed = parse_results(output)
                
                entry = {
                    "model": model_name,
                    "model_size_gb": file_size_gb,
                    "config": config_label,
                    "strategy": strategy["desc"],
                    "n_prompt": n_prompt,
                    "n_gen": n_gen,
                }
                
                if parsed:
                    entry.update({k: v for k, v in parsed.items() if k != 'raw'})
                    all_results.append(entry)
                    pp = parsed.get('pp_tps', 'N/A')
                    tg = parsed.get('tg_tps', 'N/A')
                    print(f"PP={pp}, TG={tg}")
                else:
                    entry['error'] = 'failed'
                    all_results.append(entry)
                    print("FAILED")
    
    # Step 3: 输出结果
    print("\n\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    
    for config_label in [c[2] for c in test_configs]:
        print(f"\n📊 {config_label}:")
        print(f"  {'Model':<30} {'Strategy':<35} {'Prefill':<15} {'Decode':<15}")
        print(f"  {'-'*95}")
        
        for r in all_results:
            if r['config'] == config_label and 'error' not in r:
                pp = r.get('pp_tps', 'N/A')
                tg = r.get('tg_tps', 'N/A')
                print(f"  {r['model']:<30} {r['strategy']:<35} {pp:<15} {tg:<15}")
    
    # Step 4: 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"dynamic_pd_benchmark_{timestamp}.md")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Dynamic Load-Balance Layer-Split PD Benchmark\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## GPU Information\n\n")
        f.write("| GPU | Name | VRAM (MiB) | Free (MiB) |\n")
        f.write("|-----|------|------------|-------------|\n")
        for gid, info in gpu_info.items():
            f.write(f"| {gid} | {info.get('name', '?')} | {info.get('total_mib', '?')} | {info.get('free_mib', '?')} |\n")
        
        for config_label in [c[2] for c in test_configs]:
            f.write(f"\n## {config_label}\n\n")
            f.write("| Model | Strategy | Prefill (t/s) | Decode (t/s) |\n")
            f.write("|-------|----------|---------------|-------------|\n")
            for r in all_results:
                if r['config'] == config_label and 'error' not in r:
                    pp = r.get('pp_tps', 'N/A')
                    tg = r.get('tg_tps', 'N/A')
                    f.write(f"| {r['model']} | {r['strategy']} | {pp} | {tg} |\n")
    
    print(f"\n📄 Results saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    main()
