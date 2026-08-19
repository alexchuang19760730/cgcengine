#!/usr/bin/env python3
"""
GPU Benchmark Runner using Vulkan-enabled llama-bench.
Tests prefill/decode performance across context lengths and memory usage.
"""

import os
import sys
import subprocess
import re
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
        "name": "Qwen3.6-35B-A3B-IQ3XXS",
        "path": r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
    },
    {
        "name": "Gemma3-27B-IQ3_M",
        "path": r"D:\alex\flashkv0516\models\gemma3_27b_iq3m\gemma-3-27b-it-IQ3_M.gguf",
    },
    {
        "name": "Gemma4-26B-A4B-UD-IQ3S",
        "path": r"D:\alex\flashkv0516\models\gemma4_gguf\gemma-4-26B-A4B-it-UD-IQ3_S.gguf",
    },
]

GPU_DEVICES = [
    ("0", "Intel UHD Graphics (iGPU)"),
    ("1", "NVIDIA GeForce MX250 (dGPU)"),
]

TEST_CONFIGS = [
    (512, 128, "512/128"),
    (2048, 512, "2K/512"),
    (4096, 256, "4K/256"),
]


def run_bench(model_path, n_prompt, n_gen, gpu_device):
    cmd = [
        LLAMA_BENCH,
        "-m", model_path,
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-r", "3",
        "-t", "8",
        "-fa", "1",
        "-ngl", "99",
        "-dev", gpu_device,
        "--no-warmup",
        "-o", "md"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.stdout
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return str(e)


def parse_results(output):
    if not output or isinstance(output, str) and output.startswith('Error'):
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
    print("  GPU BENCHMARK (Vulkan-enabled llama.cpp)")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  GPUs: Intel UHD (iGPU) + NVIDIA MX250 (dGPU)")
    print("=" * 70)
    
    all_results = []
    
    for model_info in MODELS:
        model_path = model_info["path"]
        model_name = model_info["name"]
        
        if not os.path.exists(model_path):
            print(f"\n⚠️  SKIP {model_name} (not found)")
            continue
        
        file_size_gb = os.path.getsize(model_path) / 1024 / 1024 / 1024
        
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name} ({file_size_gb:.2f} GB)")
        print(f"{'='*60}")
        
        for gpu_id, gpu_name in GPU_DEVICES:
            print(f"\n  📟 GPU: {gpu_name}")
            print(f"  {'-'*40}")
            
            for n_prompt, n_gen, config_label in TEST_CONFIGS:
                print(f"    [{config_label}] ", end="", flush=True)
                
                output = run_bench(model_path, n_prompt, n_gen, gpu_id)
                parsed = parse_results(output)
                
                entry = {
                    "model": model_name,
                    "model_size_gb": file_size_gb,
                    "gpu_id": gpu_id,
                    "gpu_name": gpu_name,
                    "config": config_label,
                    "n_prompt": n_prompt,
                    "n_gen": n_gen,
                }
                
                if parsed:
                    entry.update({k: v for k, v in parsed.items() if k != 'raw'})
                    all_results.append(entry)
                    pp = parsed.get('pp_tps', 'N/A')
                    tg = parsed.get('tg_tps', 'N/A')
                    print(f"PP={pp} t/s, TG={tg} t/s")
                else:
                    entry['error'] = 'timeout or error'
                    all_results.append(entry)
                    print("ERROR")
    
    # Print summary
    print("\n\n" + "=" * 70)
    print("  GPU BENCHMARK SUMMARY")
    print("=" * 70)
    
    for gpu_id, gpu_name in GPU_DEVICES:
        print(f"\n📟 {gpu_name}:")
        print(f"  {'Model':<30} {'Config':<12} {'Prefill':<18} {'Decode':<18}")
        print(f"  {'-'*78}")
        
        for r in all_results:
            if r['gpu_id'] == gpu_id and 'error' not in r:
                pp = r.get('pp_tps', 'N/A')
                tg = r.get('tg_tps', 'N/A')
                print(f"  {r['model']:<30} {r['config']:<12} {pp:<18} {tg:<18}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"gpu_benchmark_{timestamp}.md")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# GPU Benchmark Results (Vulkan)\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Build:** llama.cpp with GGML_VULKAN=ON\n\n")
        
        f.write("## Hardware\n\n")
        f.write("| Device | Type | VRAM |\n")
        f.write("|--------|------|------|\n")
        f.write("| Intel UHD Graphics | iGPU | 4 GB |\n")
        f.write("| NVIDIA GeForce MX250 | dGPU | 2 GB |\n\n")
        
        for gpu_id, gpu_name in GPU_DEVICES:
            f.write(f"\n## {gpu_name}\n\n")
            f.write("| Model | Config | Prefill (t/s) | Decode (t/s) |\n")
            f.write("|-------|--------|---------------|-------------|\n")
            for r in all_results:
                if r['gpu_id'] == gpu_id and 'error' not in r:
                    pp = r.get('pp_tps', 'N/A')
                    tg = r.get('tg_tps', 'N/A')
                    f.write(f"| {r['model']} | {r['config']} | {pp} | {tg} |\n")
        
        f.write("\n## Detailed Results\n\n")
        for r in all_results:
            f.write(f"\n### {r['model']} - {r['gpu_name']} - {r['config']}\n\n")
            if 'error' in r:
                f.write(f"**Error:** {r['error']}\n")
            else:
                f.write(f"- **Prefill:** {r.get('pp_tps', 'N/A')} tokens/s\n")
                f.write(f"- **Decode:** {r.get('tg_tps', 'N/A')} tokens/s\n")
    
    print(f"\n📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
