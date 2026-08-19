#!/usr/bin/env python3
"""
✅ 完整的 vLLM vs vLLM + KDA 对比！
全部用 vLLM 官方 benchmark！
"""
import os
import sys
import json
import time
import subprocess
from typing import List, Dict, Any

# 配置
MODEL_PATH = '/home/gs01/models/Qwen/Qwen2___5-7B-Instruct'
CONTEXT_LENGTHS = [256, 512, 1024, 2048]
OUTPUT_LENGTH = 128
BATCH_SIZE = 1


def run_one_benchmark(ctx_len: int, kda_enabled: bool) -> Dict[str, Any]:
    """运行单个 benchmark（可以选择是否用 KDA）"""
    env = os.environ.copy()
    if kda_enabled:
        env['VLLM_USE_CGC_KDA'] = '1'

    cmd = f"""
    python3 -m vllm.benchmarks.benchmark_throughput \
      --model {MODEL_PATH} \
      --batch-size {BATCH_SIZE} \
      --input-length {ctx_len} \
      --output-length {OUTPUT_LENGTH} \
      --trust-remote-code
    """.strip()
    label = "KDA" if kda_enabled else "BASELINE"
    print(f"\n⏳ [{label}] Running ctx={ctx_len} ...")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    output = result.stdout + result.stderr

    parsed = {
        "label": label,
        "ctx_len": ctx_len,
        "raw": output,
        "prefill_tps": 0,
        "decode_tps": 0,
        "prefill_latency_ms": 0,
        "decode_latency_ms": 0,
        "gpu_mem_mb": 0
    }

    for line in output.split('\n'):
        line_lower = line.lower()
        if 'prefill throughput' in line_lower:
            try:
                import re
                match = re.search(r'[\d.]+', line)
                if match:
                    parsed["prefill_tps"] = float(match.group())
            except Exception:
                pass
        if 'decode throughput' in line_lower:
            try:
                import re
                match = re.search(r'[\d.]+', line)
                if match:
                    parsed["decode_tps"] = float(match.group())
            except Exception:
                pass
        if 'prefill latency' in line_lower:
            try:
                import re
                match = re.search(r'[\d.]+', line)
                if match:
                    parsed["prefill_latency_ms"] = float(match.group())
            except Exception:
                pass
        if 'decode latency' in line_lower:
            try:
                import re
                match = re.search(r'[\d.]+', line)
                if match:
                    parsed["decode_latency_ms"] = float(match.group())
            except Exception:
                pass
        if 'total gpu memory' in line_lower:
            try:
                import re
                match = re.search(r'[\d.]+', line)
                if match:
                    parsed["gpu_mem_mb"] = float(match.group())
            except Exception:
                pass

    return parsed


def main():
    print("=" * 80)
    print("🏆 OFFICIAL vLLM vs vLLM+KDA FULL COMPARISON!")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"Context lengths: {CONTEXT_LENGTHS}")
    print(f"Output: {OUTPUT_LENGTH} tokens")

    all_baseline = []
    all_kda = []

    total_start = time.time()

    for ctx in CONTEXT_LENGTHS:
        baseline_res = run_one_benchmark(ctx, kda_enabled=False)
        kda_res = run_one_benchmark(ctx, kda_enabled=True)

        all_baseline.append(baseline_res)
        all_kda.append(kda_res)

    # Print comparison
    print("\n" + "=" * 120)
    print("📊 COMPARISON TABLE: vLLM vs vLLM+KDA")
    print("=" * 120)

    header = (f"{'Context':<10} | "
              f"{'vLLM Pref TPS':<15} {'vLLM Dec TPS':<15} {'vLLM Mem (MB)':<15} | "
              f"{'KDA Pref TPS':<15} {'KDA Dec TPS':<15} {'KDA Mem (MB)':<15}")
    print(header)
    print("-" * 120)

    for base, kda in zip(all_baseline, all_kda):
        line = (f"{base['ctx_len']:<10} | "
                f"{base['prefill_tps']:<15.1f} {base['decode_tps']:<15.1f} {base['gpu_mem_mb']:<15.0f} | "
                f"{kda['prefill_tps']:<15.1f} {kda['decode_tps']:<15.1f} {kda['gpu_mem_mb']:<15.0f}")
        print(line)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 120)
    print(f"✅ COMPLETE! Took {total_elapsed:.1f} seconds")

    out_file = '/home/gs01/official_vllm_comparison.json'
    with open(out_file, 'w') as f:
        json.dump({
            "baseline": all_baseline,
            "kda": all_kda
        }, f, indent=2)
    print(f"Results saved to: {out_file}")


if __name__ == '__main__':
    main()
