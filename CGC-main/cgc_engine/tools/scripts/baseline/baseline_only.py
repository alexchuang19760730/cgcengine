#!/usr/bin/env python3
"""vLLM Baseline Benchmark for Qwen2.5-7B"""
import os
import sys
import time
import json
import torch

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3), torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0, 0.0

def run_benchmark(ctx_len=8192, num_iters=5):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="/home/gs01/models/Qwen/Qwen2___5-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=ctx_len + 256,
        enforce_eager=True,
        disable_log_stats=True
    )

    prompt = "Hello world " * (ctx_len // 12)
    prompt_tokens = len(llm.get_tokenizer().encode(prompt))

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    print(f"\n--- Context Length: {ctx_len} ---")
    print(f"Prompt tokens: {prompt_tokens}")

    results = {
        "context_length": ctx_len,
        "prompt_tokens": prompt_tokens,
        "output_tokens": 128,
        "iterations": num_iters,
        "times": [],
        "peak_memory_gb": 0.0,
        "current_memory_gb": 0.0,
    }

    for i in range(num_iters):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        torch.cuda.synchronize()
        total_time = time.time() - start_time

        current_mem, peak_mem = get_gpu_memory()

        results["times"].append(total_time)
        results["peak_memory_gb"] = max(results["peak_memory_gb"], peak_mem)
        results["current_memory_gb"] = max(results["current_memory_gb"], current_mem)

        print(f"  Iter {i+1}/{num_iters}: Time={total_time:.3f}s, Peak={peak_mem:.2f}GB")

    del llm

    avg_time = sum(results["times"]) / len(results["times"])
    total_tokens = prompt_tokens + 128
    throughput = total_tokens / avg_time

    results["avg_time"] = avg_time
    results["throughput_tokens_per_sec"] = throughput

    return results

def main():
    print("=" * 80)
    print("vLLM Baseline Benchmark: Qwen2.5-7B-Instruct")
    print("=" * 80)

    context_lengths = [1024, 2048, 4096, 8192]
    all_results = []

    for ctx_len in context_lengths:
        result = run_benchmark(ctx_len=ctx_len, num_iters=5)
        all_results.append(result)

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Context':<10} {'Prompt':<10} {'Avg Time':<12} {'Throughput':<15} {'Peak Mem':<10}")
    print("-" * 60)

    for r in all_results:
        print(f"{r['context_length']:<10} {r['prompt_tokens']:<10} {r['avg_time']:<12.3f} "
              f"{r['throughput_tokens_per_sec']:<15.2f} {r['peak_memory_gb']:<10.2f}")

    output_file = "/home/gs01/baseline_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
