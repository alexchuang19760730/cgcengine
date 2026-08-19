#!/usr/bin/env python3
"""
✅ 最完整的官方 vLLM 基准测试脚本！
使用官方：
- benchmark_throughput
- benchmark_memory_usage
- 并且输出你要的完整表格！
"""
import os
import sys
import json
import time
import subprocess
from typing import List, Dict, Any

# 配置
MODEL_PATH = '/home/gs01/models/Qwen/Qwen2___5-7B-Instruct'
CONTEXT_LENGTHS = [256, 512, 1024, 2048]  # 你可以改成 [1024, 2048, 4096, 8192, 16384]
OUTPUT_LENGTH = 128
BATCH_SIZE = 1


def run_throughput_benchmark(input_len: int, output_len: int, batch_size: int) -> Dict[str, Any]:
    """
    运行官方 benchmark_throughput！
    它会返回 prefill/decode 的 latency/throughput！
    """
    cmd = f"""
    python3 -m vllm.benchmarks.benchmark_throughput \
      --model {MODEL_PATH} \
      --batch-size {batch_size} \
      --input-length {input_len} \
      --output-length {output_len} \
      --trust-remote-code
    """.strip()
    print(f"⏳ Running throughput benchmark (ctx={input_len}, output={output_len})")
    print(f"Command: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    output = result.stdout + result.stderr

    print(f"Output (first 200 chars): {output[:200]}...")

    parsed = {
        "input_len": input_len,
        "output_len": output_len,
        "batch_size": batch_size,
        "raw_output": output,
        "prefill_throughput_tps": 0,
        "decode_throughput_tps": 0,
        "prefill_latency_ms": 0,
        "decode_latency_ms_per_tok": 0,
        "total_gpu_memory_mb": 0
    }

    # 简单解析输出
    lines = output.split('\n')
    for line in lines:
        line_lower = line.lower()
        if 'prefill throughput' in line_lower:
            try:
                import re
                match = re.search(r'[\d.]+', line)
                if match:
                    parsed["prefill_throughput_tps"] = float(match.group())
            except Exception:
                pass
        elif 'decode throughput' in line_lower:
            try:
                import re
                match = re.search(r'[\d.]+', line)
                if match:
                    parsed["decode_throughput_tps"] = float(match.group())
            except Exception:
                pass
        elif 'prefill latency' in line_lower:
            try:
                import re
                match = re.search(r'[\d.]+', line)
                if match:
                    parsed["prefill_latency_ms"] = float(match.group())
            except Exception:
                pass
        elif 'decode latency' in line_lower:
            try:
                import re
                match = re.search(r'[\d.]+', line)
                if match:
                    parsed["decode_latency_ms_per_tok"] = float(match.group())
            except Exception:
                pass
        elif 'total gpu memory' in line_lower:
            try:
                import re
                match = re.search(r'[\d.]+', line)
                if match:
                    parsed["total_gpu_memory_mb"] = float(match.group())
            except Exception:
                pass

    return parsed


def run_memory_benchmark(context_len: int) -> Dict[str, Any]:
    """运行官方 benchmark_memory_usage"""
    cmd = f"""
    python3 -m vllm.benchmarks.benchmark_memory_usage \
      --model {MODEL_PATH} \
      --context-length {context_len}
    """.strip()
    print(f"\n⏳ Running memory benchmark (ctx={context_len})")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout + result.stderr

    parsed = {
        "context_len": context_len,
        "raw_output": output,
        "total_gpu_memory_mb": 0
    }

    lines = output.split('\n')
    for line in lines:
        if 'total gpu memory' in line.lower():
            try:
                import re
                match = re.search(r'[\d.]+', line)
                if match:
                    parsed["total_gpu_memory_mb"] = float(match.group())
            except Exception:
                pass
    return parsed


def print_pretty_table(results: List[Dict]):
    """打印漂亮的完整表格！"""
    print("\n" + "=" * 110)
    print("📊 FINAL FULL BENCHMARK TABLE")
    print("=" * 110)
    header = f"{'Context':<10} {'Prefill TPS':<15} {'Decode TPS':<15} {'Prefill (ms)':<15} {'Decode (ms/tok)':<15} {'GPU Mem (MB)':<15}"
    print(header)
    print("-" * 110)

    for res in results:
        line = (f"{res['input_len']:<10} "
                f"{res['prefill_throughput_tps']:<15.1f} "
                f"{res['decode_throughput_tps']:<15.1f} "
                f"{res['prefill_latency_ms']:<15.1f} "
                f"{res['decode_latency_ms_per_tok']:<15.1f} "
                f"{res['total_gpu_memory_mb']:<15.0f}")
        print(line)

    print("=" * 110)


def main():
    print("🚀 OFFICIAL vLLM BENCHMARK SUITE!")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"Context lengths: {CONTEXT_LENGTHS}")
    print(f"Output length: {OUTPUT_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")

    all_results = []
    start_time = time.time()

    for ctx_len in CONTEXT_LENGTHS:
        print("\n" + "=" * 80)
        print(f"📌 TESTING: Context Length = {ctx_len}")
        print("=" * 80)

        # 1. Throughput Benchmark (包含 prefill/decode)
        throughput_res = run_throughput_benchmark(ctx_len, OUTPUT_LENGTH, BATCH_SIZE)
        all_results.append(throughput_res)

    total_elapsed = time.time() - start_time
    print(f"\n✅ ALL DONE! Took {total_elapsed:.1f} seconds")

    # Print beautiful table
    print_pretty_table(all_results)

    # Save to file
    out_file = '/home/gs01/official_vllm_benchmark_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Results saved to: {out_file}")


if __name__ == '__main__':
    main()
