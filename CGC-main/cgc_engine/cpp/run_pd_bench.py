#!/usr/bin/env python3
"""
PD (Prefill-Decode Disaggregation) Benchmark
Compares:
1. Single GPU mode (baseline)
2. PD mode - P-graph on Intel iGPU, D-graph on NVIDIA dGPU
"""

import os
import sys
import subprocess
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

TEST_CONFIGS = [
    (512, 128, "512/128"),
    (2048, 512, "2K/512"),
    (4096, 256, "4K/256"),
]


def run_bench(model_path, n_prompt, n_gen, extra_args=None):
    cmd = [
        LLAMA_BENCH,
        "-m", model_path,
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-r", "3",
        "-t", "8",
        "-fa", "1",
        "--no-warmup",
        "-o", "md"
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.stdout
    except Exception as e:
        return f"ERROR: {e}"


def parse_results(output):
    if not output or output.startswith('ERROR'):
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
    print("  PD (Prefill-Decode) DISAGGREGATION BENCHMARK")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    print("""
  Test configurations:
  1. Baseline - Single GPU (Intel iGPU for both P & D)
  2. Baseline - Single GPU (NVIDIA dGPU for both P & D)
  3. PD Mode  - P on Intel iGPU, D on NVIDIA dGPU
  4. PD Mode  - P on NVIDIA dGPU, D on Intel iGPU
""")
    
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
        
        for n_prompt, n_gen, config_label in TEST_CONFIGS:
            print(f"\n  📊 Test: {config_label}")
            print(f"  {'-'*50}")
            
            configs = [
                ("Single-iGPU (P+D on Intel)", ["-ngl", "99", "-dev", "0"]),
                ("Single-dGPU (P+D on NVIDIA)", ["-ngl", "99", "-dev", "1"]),
                ("PD: P→Intel, D→NVIDIA", ["-ngl", "99", "-pg", "0,1"]),
                ("PD: P→NVIDIA, D→Intel", ["-ngl", "99", "-pg", "1,0"]),
            ]
            
            for config_name, extra_args in configs:
                print(f"    {config_name}...", end=" ", flush=True)
                
                output = run_bench(model_path, n_prompt, n_gen, extra_args)
                parsed = parse_results(output)
                
                entry = {
                    "model": model_name,
                    "model_size_gb": file_size_gb,
                    "config_label": config_label,
                    "n_prompt": n_prompt,
                    "n_gen": n_gen,
                    "config_name": config_name,
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
    
    # Print summary
    print("\n\n" + "=" * 70)
    print("  PD BENCHMARK SUMMARY")
    print("=" * 70)
    
    for config_label in TEST_CONFIGS:
        n_prompt, n_gen, label = config_label
        print(f"\n📊 {label} (prompt={n_prompt}, gen={n_gen}):")
        print(f"  {'Model':<30} {'Config':<25} {'Prefill':<18} {'Decode':<18}")
        print(f"  {'-'*90}")
        
        for r in all_results:
            if r['config_label'] == label and 'error' not in r:
                pp = r.get('pp_tps', 'N/A')
                tg = r.get('tg_tps', 'N/A')
                print(f"  {r['model']:<30} {r['config_name']:<25} {pp:<18} {tg:<18}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"pd_benchmark_{timestamp}.md")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# PD (Prefill-Decode) Disaggregation Benchmark\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Test Configurations\n\n")
        f.write("1. **Single-iGPU**: Both prefill and decode on Intel UHD Graphics\n")
        f.write("2. **Single-dGPU**: Both prefill and decode on NVIDIA GeForce MX250\n")
        f.write("3. **PD: P→Intel, D→NVIDIA**: Prefill on iGPU, Decode on dGPU\n")
        f.write("4. **PD: P→NVIDIA, D→Intel**: Prefill on dGPU, Decode on iGPU\n\n")
        
        for config_label in TEST_CONFIGS:
            n_prompt, n_gen, label = config_label
            f.write(f"\n## {label}\n\n")
            f.write("| Model | Config | Prefill (t/s) | Decode (t/s) |\n")
            f.write("|-------|--------|---------------|-------------|\n")
            for r in all_results:
                if r['config_label'] == label and 'error' not in r:
                    pp = r.get('pp_tps', 'N/A')
                    tg = r.get('tg_tps', 'N/A')
                    f.write(f"| {r['model']} | {r['config_name']} | {pp} | {tg} |\n")
        
        f.write("\n## Notes\n\n")
        f.write("- Intel UHD Graphics: 4 GB VRAM, UMA (shares system memory)\n")
        f.write("- NVIDIA GeForce MX250: 2 GB VRAM, discrete GPU\n")
        f.write("- PD mode may benefit from separating compute-intensive prefill from memory-bound decode\n")
    
    print(f"\n📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
