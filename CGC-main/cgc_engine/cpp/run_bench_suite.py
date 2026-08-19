#!/usr/bin/env python3
"""Quick llama-bench runner with multi-context comparison."""

import os
import subprocess
import re
from datetime import datetime

LLAMA_BENCH = r"D:\alex\flashkv0516\tools\llama.cpp\llama-bench.exe"
OUTPUT_DIR = r"D:\alex\flashkv0516\bench_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "qwen2.5_1.5B_Q4_K_M": r"D:\alex\flashkv0516\models\qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "qwen3.6_35B_A3B_IQ3XXS": r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
    "gemma3_27B_IQ3_M": r"D:\alex\flashkv0516\models\gemma3_27b_iq3m\gemma-3-27b-it-IQ3_M.gguf",
    "gemma4_26B_A4B_UD_IQ3S": r"D:\alex\flashkv0516\models\gemma4_gguf\gemma-4-26B-A4B-it-UD-IQ3_S.gguf",
}

# Test configurations: (prompt_len, gen_len, label)
TEST_CONFIGS = [
    (512, 128, "512/128"),
    (2048, 512, "2K/512"),
    (4096, 256, "4K/256"),
    (8192, 128, "8K/128"),
]

def run_bench(model_path, n_prompt, n_gen, model_name):
    """Run single benchmark and parse results."""
    
    cmd = [
        LLAMA_BENCH,
        "-m", model_path,
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-r", "3",
        "-t", "8",
        "-fa", "1",
        "-ngl", "99",
        "--no-warmup",
        "-o", "md"
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        
        # Parse results from markdown table
        output = result.stdout
        
        # Extract key metrics
        pp_tps = None
        tg_tps = None
        memory_info = None
        
        # Parse pp (prefill) and tg (token generation/decode) values
        for line in output.split('\n'):
            if 'pp' in line.lower() and '|' in line:
                # Extract the t/s value
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 7:
                    pp_tps = parts[-1]  # Last column is t/s
                    
            if 'tg' in line.lower() and '|' in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 7:
                    tg_tps = parts[-1]
        
        # Parse build info
        build_match = re.search(r'build:\s*(\S+)', output)
        build = build_match.group(1) if build_match else "unknown"
        
        return {
            "model": model_name,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "pp_tps": pp_tps,
            "tg_tps": tg_tps,
            "build": build,
            "raw_output": output
        }
        
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "model": model_name}
    except Exception as e:
        return {"error": str(e), "model": model_name}


def main():
    print("=" * 80)
    print("LLAMA-BENCH: Multi-Context Performance Analysis")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    all_results = []
    results_md = []
    
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"\n⚠️  SKIP {model_name} (not found)")
            print(f"    {model_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        
        for n_prompt, n_gen, label in TEST_CONFIGS:
            print(f"\n  Test: {label} (prompt={n_prompt}, gen={n_gen})...", end=" ", flush=True)
            
            result = run_bench(model_path, n_prompt, n_gen, model_name)
            all_results.append(result)
            
            if "error" not in result:
                print(f"PP={result['pp_tps']} t/s, TG={result['tg_tps']} t/s")
            else:
                print(f"ERROR: {result['error']}")
    
    # Generate summary table
    print("\n\n" + "=" * 80)
    print("SUMMARY: Performance Comparison")
    print("=" * 80)
    
    # Print results in table format
    header = f"{'Model':<30} {'Config':<12} {'Prefill t/s':<15} {'Decode t/s':<15}"
    print(header)
    print("-" * len(header))
    
    for r in all_results:
        if "error" not in r:
            config = f"{r['n_prompt']}/{r['n_gen']}"
            pp = r['pp_tps'] or "N/A"
            tg = r['tg_tps'] or "N/A"
            print(f"{r['model']:<30} {config:<12} {pp:<15} {tg:<15}")
    
    # Save full results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"benchmark_suite_{timestamp}.md")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# llama-bench Performance Analysis\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Models Tested\n\n")
        for model_name, model_path in MODELS.items():
            exists = "✓" if os.path.exists(model_path) else "✗"
            size = ""
            if os.path.exists(model_path):
                size_bytes = os.path.getsize(model_path)
                size = f" ({size_bytes/1024/1024/1024:.2f} GB)"
            f.write(f"- {exists} **{model_name}**{size}\n  - Path: `{model_path}`\n\n")
        
        f.write("\n## Results Summary\n\n")
        f.write("| Model | Config (prompt/gen) | Prefill (t/s) | Decode (t/s) |\n")
        f.write("|-------|---------------------|---------------|-------------|\n")
        
        for r in all_results:
            if "error" not in r:
                config = f"{r['n_prompt']}/{r['n_gen']}"
                pp = r['pp_tps'] or "N/A"
                tg = r['tg_tps'] or "N/A"
                f.write(f"| {r['model']} | {config} | {pp} | {tg} |\n")
        
        f.write("\n## Detailed Results\n\n")
        for r in all_results:
            f.write(f"\n### {r.get('model', 'unknown')} - {r.get('n_prompt', '?')}/{r.get('n_gen', '?')}\n\n")
            if "error" in r:
                f.write(f"**Error:** {r['error']}\n")
            else:
                f.write(f"- **Prefill (pp):** {r['pp_tps']} tokens/s\n")
                f.write(f"- **Decode (tg):** {r['tg_tps']} tokens/s\n")
                f.write(f"- **Build:** {r['build']}\n")
                f.write(f"\n```\n{r['raw_output']}\n```\n")
    
    print(f"\n📄 Full results saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    main()
