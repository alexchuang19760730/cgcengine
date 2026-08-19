#!/usr/bin/env python3
"""llama-bench runner for analyzing prefill/decode performance and memory usage."""

import os
import subprocess
import json
import sys
from datetime import datetime

# Configuration
LLAMA_BENCH = r"D:\alex\flashkv0516\tools\llama.cpp\llama-bench.exe"

# Available models
MODELS = {
    "qwen3_1.5b": {
        "path": r"D:\alex\flashkv0516\models\qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "desc": "Qwen2.5 1.5B Q4_K_M (Baseline)"
    },
    "qwen3_35b_a3b": {
        "path": r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
        "desc": "Qwen3.6 35B A3B IQ3_XXS (MoE)"
    },
    "gemma3_27b": {
        "path": r"D:\alex\flashkv0516\models\gemma3_27b_iq3m\gemma-3-27b-it-IQ3_M.gguf",
        "desc": "Gemma 3 27B IQ3_M"
    },
    "gemma4_ud_iq3s": {
        "path": r"D:\alex\flashkv0516\models\gemma4_gguf\gemma-4-26B-A4B-it-UD-IQ3_S.gguf",
        "desc": "Gemma 4 26B A4B UD IQ3_S (MoE)"
    }
}

# Test configurations: different context lengths
TEST_CONFIGS = [
    # (n_prompt, n_gen, desc)
    (512, 128, "短上下文 (512)"),
    (1024, 256, "中等上下文 (1K)"),
    (2048, 512, "长上下文 (2K)"),
    (4096, 512, "超长上下文 (4K)"),
    (8192, 256, "极限上下文 (8K) - prefill only"),
]

# Output directory
OUTPUT_DIR = r"D:\alex\flashkv0516\bench_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_benchmark(model_path, model_name, n_prompt, n_gen, desc):
    """Run llama-bench with specified parameters."""
    
    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"{model_name}_{n_prompt}_{timestamp}.md")
    
    cmd = [
        LLAMA_BENCH,
        "-m", model_path,
        "-p", str(n_prompt),      # prompt length
        "-n", str(n_gen),         # generation length
        "-r", "3",                # repetitions (3 runs)
        "-t", "8",                # threads
        "-fa", "1",               # flash attention
        "-ngl", "99",             # offload to GPU
        "--no-warmup",            # skip warmup for more realistic results
        "-o", "md"                # markdown output
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {desc}")
    print(f"Model: {model_name}")
    print(f"Prompt: {n_prompt}, Gen: {n_gen}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Save output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Benchmark: {model_name}\n")
            f.write(f"## Config: {desc}\n")
            f.write(f"- Prompt tokens: {n_prompt}\n")
            f.write(f"- Gen tokens: {n_gen}\n")
            f.write(f"- Timestamp: {timestamp}\n")
            f.write(f"\n## Results\n\n")
            f.write(result.stdout)
            if result.stderr:
                f.write(f"\n## Stderr\n\n")
                f.write(result.stderr)
        
        print(f"\nResults saved to: {output_file}")
        print(f"\nKey output:\n")
        # Print relevant lines
        for line in result.stdout.split('\n'):
            if any(kw in line.lower() for kw in ['prompt', 'eval', 'total', 'memory', 'peak', 'ms', 'tok']):
                print(f"  {line.strip()}")
        
        return result.stdout
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout after 10 minutes for {model_name} with prompt {n_prompt}")
        return None
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        return None


def run_full_suite():
    """Run benchmark suite across all models and configurations."""
    
    print("=" * 80)
    print("LLAMA-BENCH SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    results = {}
    
    for model_name, model_info in MODELS.items():
        model_path = model_info["path"]
        
        if not os.path.exists(model_path):
            print(f"\n⚠️  Skipping {model_name}: file not found")
            print(f"    {model_path}")
            continue
        
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model_name}")
        print(f"# Desc: {model_info['desc']}")
        print(f"# Path: {model_path}")
        print(f"{'#'*80}")
        
        model_results = []
        
        for n_prompt, n_gen, desc in TEST_CONFIGS:
            try:
                output = run_benchmark(
                    model_path, model_name, 
                    n_prompt, n_gen, desc
                )
                model_results.append({
                    "config": desc,
                    "n_prompt": n_prompt,
                    "n_gen": n_gen,
                    "output": output
                })
            except Exception as e:
                print(f"  Failed: {e}")
                model_results.append({
                    "config": desc,
                    "n_prompt": n_prompt,
                    "n_gen": n_gen,
                    "error": str(e)
                })
        
        results[model_name] = model_results
        
        # Save intermediate results
        summary_file = os.path.join(OUTPUT_DIR, f"summary_{model_name}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model": model_name,
                "desc": model_info["desc"],
                "results": model_results
            }, f, indent=2, ensure_ascii=False)
    
    # Generate final summary
    print(f"\n\n{'='*80}")
    print("BENCHMARK SUITE COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nFiles generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        filepath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(filepath)
        print(f"  {f} ({size:,} bytes)")
    
    return results


def run_single_benchmark():
    """Run a single quick benchmark with specified model."""
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python run_bench.py <model_key> [n_prompt] [n_gen]")
        print("\nAvailable models:")
        for key, info in MODELS.items():
            exists = "✓" if os.path.exists(info["path"]) else "✗"
            print(f"  {key} {exists} - {info['desc']}")
            print(f"    Path: {info['path']}")
        print("\nExamples:")
        print("  python run_bench.py qwen3_1.5b 4096 512")
        print("  python run_bench.py gemma4_ud_iq3s 2048 256")
        return
    
    model_key = sys.argv[1]
    n_prompt = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    
    if model_key not in MODELS:
        print(f"Unknown model key: {model_key}")
        return
    
    model_info = MODELS[model_key]
    model_path = model_info["path"]
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    print(f"Running single benchmark: {model_key}")
    print(f"  Prompt: {n_prompt}, Gen: {n_gen}")
    
    desc = f"Custom ({n_prompt} prompt, {n_gen} gen)"
    output = run_benchmark(model_path, model_key, n_prompt, n_gen, desc)
    
    if output:
        print("\n" + "="*80)
        print("FULL OUTPUT:")
        print("="*80)
        print(output)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        # Run full suite
        run_full_suite()
    else:
        # Run single benchmark
        run_single_benchmark()
