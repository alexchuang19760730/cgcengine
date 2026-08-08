#!/usr/bin/env python3
"""
vLLM Profile Benchmark - 完整版 mperf 测试
测试不同上下文长度的 Prefill + Decode + 显存 + Attention + KV Cache + 算子耗时
"""
import os
import sys
import json
import time
import torch

sys.path.insert(0, "/home/gs01")

from vllm import LLM, SamplingParams

MODEL_PATH = "/home/gs01/models/Qwen/Qwen2___5-7B-Instruct"
CTX_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]
OUTPUT_LEN = 128
BATCH_SIZE = 1
NUM_ITERS = 5

def run_profile_benchmark(ctx_len, num_iters=5):
    print(f"\n{'='*60}")
    print(f"Testing Context Length: {ctx_len}")
    print(f"{'='*60}")

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=ctx_len + 256,
        enforce_eager=True,
        disable_log_stats=False
    )

    prompt = "Hello" * (ctx_len // 4)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=OUTPUT_LEN,
        min_tokens=OUTPUT_LEN
    )

    prompt_tokens = len(llm.get_tokenizer().encode(prompt))
    print(f"Prompt tokens: {prompt_tokens}")

    results = {
        "context_length": ctx_len,
        "prompt_tokens": prompt_tokens,
        "output_tokens": OUTPUT_LEN,
        "iterations": num_iters,
        "times": [],
        "prefill_times": [],
        "decode_times": [],
        "memory_stats": {},
    }

    for i in range(num_iters):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        torch.cuda.synchronize()
        end_time = time.time()

        total_time = end_time - start_time
        results["times"].append(total_time)

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            current_mem = torch.cuda.memory_allocated() / (1024**3)
            results["memory_stats"] = {
                "peak_gb": float(peak_mem),
                "current_gb": float(current_mem),
            }

        output_text = outputs[0].outputs[0].text
        print(f"  Iter {i+1}: {total_time:.3f}s, Peak Memory: {results['memory_stats'].get('peak_gb', 0):.2f}GB")

    avg_time = sum(results["times"]) / len(results["times"])
    throughput = (prompt_tokens + OUTPUT_LEN) / avg_time

    print(f"\n  Average Time: {avg_time:.3f}s")
    print(f"  Throughput: {throughput:.2f} tokens/s")

    return results

def main():
    print("vLLM Profile Benchmark - Full mperf test")
    print(f"Model: {MODEL_PATH}")
    print(f"Output length: {OUTPUT_LEN}")
    print(f"Iterations: {NUM_ITERS}")

    all_results = []

    for ctx_len in CTX_LENGTHS:
        try:
            result = run_profile_benchmark(ctx_len, NUM_ITERS)
            all_results.append(result)
        except Exception as e:
            print(f"  Error at ctx_len={ctx_len}: {e}")
            import traceback
            traceback.print_exc()

    output_file = "/home/gs01/mperf_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    print("\n\n=== SUMMARY ===")
    print(f"{'Context':<10} {'Prompt':<10} {'Avg Time':<12} {'Throughput':<15} {'Peak Mem':<10}")
    print("-" * 60)
    for r in all_results:
        avg_time = sum(r["times"]) / len(r["times"])
        throughput = (r["prompt_tokens"] + r["output_tokens"]) / avg_time
        peak_mem = r["memory_stats"].get("peak_gb", 0)
        print(f"{r['context_length']:<10} {r['prompt_tokens']:<10} {avg_time:<12.3f} {throughput:<15.2f} {peak_mem:<10.2f}")

if __name__ == "__main__":
    main()
