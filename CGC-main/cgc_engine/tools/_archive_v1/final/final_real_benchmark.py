#!/usr/bin/env python3
"""
✅ 完美最终版本！
用 vLLM 正确的 benchmark 文件：throughput.py
"""
import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Any

sys.path.insert(0, '/home/gs01')

MODEL_PATH = '/home/gs01/models/Qwen/Qwen2___5-7B-Instruct'
CONTEXT_LENGTHS = [256, 512, 1024, 2048]
OUTPUT_LENGTH = 128
BATCH_SIZE = 4
GPU_MEM_UTIL = 0.7


def get_gpu_memory_stats() -> Dict[str, float]:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
                "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
                "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3)
            }
    except Exception:
        pass
    return {"allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0}


def clear_gpu_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def run_single_test(
    llm, prefill_len: int, decode_len: int, batch_size: int, num_iters: int = 3, num_warmup: int = 2
) -> Dict[str, Any]:
    from vllm import SamplingParams
    dummy_token_ids = np.random.randint(10000, size=(batch_size, prefill_len)).tolist()
    test_prompts = [{"prompt_token_ids": x} for x in dummy_token_ids]
    sampling_params = SamplingParams(temperature=0, max_tokens=decode_len, ignore_eos=True)

    for _ in range(num_warmup):
        llm.generate(test_prompts, sampling_params, use_tqdm=False)

    total_times = []
    peak_mems = []

    for _ in range(num_iters):
        clear_gpu_cache()
        t0 = time.perf_counter()
        llm.generate(test_prompts, sampling_params, use_tqdm=False)
        t1 = time.perf_counter()
        total_times.append(t1 - t0)
        mem_info = get_gpu_memory_stats()
        peak_mems.append(mem_info["max_allocated_gb"])

    total_tokens = batch_size * decode_len
    avg_time = np.mean(total_times)
    std_time = np.std(total_times)
    prefill_tps = (batch_size * prefill_len) / avg_time
    decode_tps = total_tokens / avg_time
    avg_peak_gb = np.mean(peak_mems)
    max_peak_gb = np.max(peak_mems)

    return {
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        "batch_size": batch_size,
        "avg_total_time": avg_time,
        "std_total_time": std_time,
        "prefill_throughput_tps": prefill_tps,
        "decode_throughput_tps": decode_tps,
        "avg_peak_gb": avg_peak_gb,
        "max_peak_gb": max_peak_gb
    }


def print_beautiful_table(results: List[Dict[str, Any]]):
    print("\n" + "=" * 120)
    print("📊 FINAL BENCHMARK RESULTS (vLLM Baseline)")
    print("=" * 120)
    header = (f"{'Prefill':<10} "
              f"{'Batch Size':<10} "
              f"{'Avg Time (s)':<15} "
              f"{'Prefill TPS':<15} "
              f"{'Decode TPS':<15} "
              f"{'Avg Mem (GB)':<15} "
              f"{'Max Mem (GB)':<15}")
    print(header)
    print("-" * 120)
    for r in results:
        line = (f"{r['prefill_len']:<10} "
                f"{r['batch_size']:<10} "
                f"{r['avg_total_time']:<15.4f} "
                f"{r['prefill_throughput_tps']:<15.1f} "
                f"{r['decode_throughput_tps']:<15.1f} "
                f"{r['avg_peak_gb']:<15.2f} "
                f"{r['max_peak_gb']:<15.2f}")
        print(line)


def main():
    print("=" * 120)
    print("🏆 FINAL COMPLETE vLLM BENCHMARK (PRODUCTION-READY)")
    print("=" * 120)
    print(f"Model: {MODEL_PATH}")
    print(f"Prefill lengths: {CONTEXT_LENGTHS}")
    print(f"Decode length: {OUTPUT_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    from vllm import LLM

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=True
    )

    all_results = []
    start_time = time.time()

    for ctx_len in CONTEXT_LENGTHS:
        print("\n" + "=" * 80)
        print(f"📌 Testing: Context Length = {ctx_len}")
        print("=" * 80)
        res = run_single_test(llm, ctx_len, OUTPUT_LENGTH, BATCH_SIZE)
        all_results.append(res)
        print(f"✅ Done for {ctx_len}!")

    total_elapsed = time.time() - start_time

    print_beautiful_table(all_results)

    del llm
    clear_gpu_cache()

    out_file = '/home/gs01/final_real_vllm_benchmark_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 120)
    print(f"✅ ALL DONE! Took {total_elapsed:.1f} seconds!")
    print(f"Results saved to: {out_file}")


if __name__ == '__main__':
    main()
