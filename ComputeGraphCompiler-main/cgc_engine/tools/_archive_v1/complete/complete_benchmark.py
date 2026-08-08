#!/usr/bin/env python3
"""Complete Benchmark for vLLM vs vLLM+KDA on Qwen2.5-7B"""
import os
import sys
import time
import json
import torch

def get_gpu_memory():
    try:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3), torch.cuda.max_memory_allocated() / (1024**3)
    except:
        pass
    return 0.0, 0.0

def run_benchmark(kda_enabled=False, ctx_len=8192, num_iters=3):
    if kda_enabled:
        os.environ["VLLM_USE_CGC_KDA"] = "1"
        sys.path.insert(0, "/home/gs01")

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

    results = {
        "kda_enabled": kda_enabled,
        "context_length": ctx_len,
        "prompt_tokens": prompt_tokens,
        "output_tokens": 128,
        "iterations": num_iters,
        "prefill_times": [],
        "decode_times": [],
        "total_times": [],
        "peak_memory_gb": 0.0,
        "current_memory_gb": 0.0,
    }

    for i in range(num_iters):
        torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        total_time = time.time() - start_time

        current_mem, peak_mem = get_gpu_memory()

        results["total_times"].append(total_time)
        results["peak_memory_gb"] = max(results["peak_memory_gb"], peak_mem)
        results["current_memory_gb"] = max(results["current_memory_gb"], current_mem)

        print(f"[{'KDA' if kda_enabled else 'Baseline'}] Iter {i+1}/{num_iters}: "
              f"Time={total_time:.2f}s, Peak={peak_mem:.2f}GB")

    del llm

    avg_time = sum(results["total_times"]) / len(results["total_times"])
    total_tokens = prompt_tokens + 128
    throughput = total_tokens / avg_time

    results["avg_time"] = avg_time
    results["throughput_tokens_per_sec"] = throughput
    results["prefill_throughput"] = prompt_tokens / avg_time
    results["decode_throughput"] = 128 / avg_time

    return results

def main():
    print("=" * 80)
    print("vLLM Benchmark: Baseline vs KDA")
    print("=" * 80)

    context_lengths = [1024, 2048, 4096, 8192]

    all_results = []

    for ctx_len in context_lengths:
        print(f"\n{'='*40}")
        print(f"Testing context length: {ctx_len}")
        print(f"{'='*40}")

        print("\n--- vLLM Baseline ---")
        baseline = run_benchmark(kda_enabled=False, ctx_len=ctx_len, num_iters=3)
        all_results.append(("Baseline", ctx_len, baseline))

        print("\n--- vLLM + KDA ---")
        kda = run_benchmark(kda_enabled=True, ctx_len=ctx_len, num_iters=3)
        all_results.append(("KDA", ctx_len, kda))

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Context':<10} {'Mode':<10} {'Time(s)':<10} {'Throughput':<15} {'Peak Mem(GB)':<12}")
    print("-" * 60)

    for mode, ctx_len, results in all_results:
        print(f"{ctx_len:<10} {mode:<10} {results['avg_time']:<10.2f} "
              f"{results['throughput_tokens_per_sec']:<15.2f} {results['peak_memory_gb']:<12.2f}")

    with open("/home/gs01/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\nResults saved to /home/gs01/benchmark_results.json")

if __name__ == "__main__":
    main()
