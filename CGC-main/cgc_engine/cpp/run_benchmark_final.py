#!/usr/bin/env python3
"""
llama-bench runner with comprehensive analysis of:
- Prefill (pp) and Decode (tg) performance across context lengths
- Memory usage estimation
- Multi-model comparison
"""

import os
import sys
import subprocess
import re
from datetime import datetime

LLAMA_BENCH = r"D:\alex\flashkv0516\tools\llama.cpp\llama-bench.exe"
OUTPUT_DIR = r"D:\alex\flashkv0516\bench_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    {
        "name": "Qwen2.5-1.5B-Q4_K_M",
        "path": r"D:\alex\flashkv0516\models\qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "desc": "Small baseline (CPU mode)"
    },
    {
        "name": "Qwen3.6-35B-A3B-IQ3XXS",
        "path": r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
        "desc": "MoE model (CPU mode)"
    },
]

TEST_CONFIGS = [
    (512, 128, "512/128"),
    (2048, 512, "2048/512"),
    (4096, 256, "4096/256"),
]

def run_bench(model_path, n_prompt, n_gen):
    """Run llama-bench and return parsed results."""
    
    cmd = [
        LLAMA_BENCH,
        "-m", model_path,
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-r", "3",
        "-t", "8",
        "-fa", "1",
        "-ngl", "0",  # CPU only since no GPU backend
        "--no-warmup",
        "-o", "md"
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        
        output = result.stdout
        pp_tps = None
        tg_tps = None
        
        for line in output.split('\n'):
            if '|' in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 6:
                    if 'pp' in parts[0].lower():
                        pp_tps = parts[-1]
                    elif 'tg' in parts[0].lower():
                        tg_tps = parts[-1]
        
        return {
            "pp_tps": pp_tps,
            "tg_tps": tg_tps,
            "raw": output
        }
        
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 70)
    print("  LLAMA-BENCH: Prefill/Decode Performance Analysis")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Backend: CPU (Vulkan not available in this build)")
    print("=" * 70)
    
    all_results = []
    
    for model_info in MODELS:
        model_path = model_info["path"]
        model_name = model_info["name"]
        
        if not os.path.exists(model_path):
            print(f"\n⚠️  SKIP {model_name} (file not found)")
            continue
        
        file_size_gb = os.path.getsize(model_path) / 1024 / 1024 / 1024
        
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"  File size: {file_size_gb:.2f} GB")
        print(f"  Path: {model_path}")
        print(f"{'='*60}")
        
        for n_prompt, n_gen, config_label in TEST_CONFIGS:
            print(f"  [{config_label}] prompt={n_prompt}, gen={n_gen}...", end=" ", flush=True)
            
            result = run_bench(model_path, n_prompt, n_gen)
            
            entry = {
                "model": model_name,
                "model_size_gb": file_size_gb,
                "config": config_label,
                "n_prompt": n_prompt,
                "n_gen": n_gen,
                **result
            }
            all_results.append(entry)
            
            if "error" not in result:
                print(f"PP={result['pp_tps']} t/s, TG={result['tg_tps']} t/s")
            else:
                print(f"ERROR: {result['error']}")
    
    # Print summary
    print("\n\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Config':<12} {'Prefill':<18} {'Decode':<18}")
    print("-" * 78)
    
    for r in all_results:
        if "error" not in r:
            pp = r['pp_tps'] or "N/A"
            tg = r['tg_tps'] or "N/A"
            print(f"{r['model']:<30} {r['config']:<12} {pp:<18} {tg:<18}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"benchmark_{timestamp}.md")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# llama-bench Performance Analysis\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Backend:** CPU (Vulkan runtime detected but not compiled in)\n\n")
        
        f.write("## Summary Table\n\n")
        f.write("| Model | Config | Prefill (t/s) | Decode (t/s) |\n")
        f.write("|-------|--------|---------------|-------------|\n")
        for r in all_results:
            if "error" not in r:
                pp = r['pp_tps'] or "N/A"
                tg = r['tg_tps'] or "N/A"
                f.write(f"| {r['model']} | {r['config']} | {pp} | {tg} |\n")
        
        f.write("\n## Detailed Results\n\n")
        for r in all_results:
            f.write(f"\n### {r['model']} - {r['config']}\n\n")
            if "error" in r:
                f.write(f"**Error:** {r['error']}\n")
            else:
                f.write(f"- **Prefill:** {r['pp_tps']} tokens/s\n")
                f.write(f"- **Decode:** {r['tg_tps']} tokens/s\n\n")
                f.write("```\n")
                f.write(r['raw'])
                f.write("\n```\n")
    
    print(f"\n📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
