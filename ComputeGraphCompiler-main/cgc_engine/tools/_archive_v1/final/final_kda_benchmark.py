#!/usr/bin/env python3
"""
✨ 最终 vLLM + KDA 的完整 benchmark！
"""
import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Any

# 设置环境变量开启 KDA
os.environ["VLLM_USE_CGC_KDA"] = "1"
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


def run_single_test(llm, prefill_len: int, decode_len: int, batch_size: int) -> Dict[str, Any]:
    from vllm import SamplingParams
    dummy_token_ids = np.random.randint(10000, size=(batch_size, prefill_len)).tolist()
    test_prompts = [{"prompt_token_ids": x} for x in dummy_token_ids]
    sampling_params = SamplingParams(temperature=0, max_tokens=decode_len, ignore_eos=True)

    # Warmup
    print("Warming up ...")
    llm.generate(test_prompts, sampling_params, use_tqdm=False)

    print("Running real test ...")
    clear_gpu_cache()
    t0 = time.perf_counter()
    llm.generate(test_prompts, sampling_params, use_tqdm=False)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    mem_info = get_gpu_memory_stats()

    total_prefill_tokens = batch_size * prefill_len
    total_decode_tokens = batch_size * decode_len

    prefill_tps = total_prefill_tokens / elapsed
    decode_tps = total_decode_tokens / elapsed

    return {
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        "batch_size": batch_size,
        "total_time_s": elapsed,
        "prefill_throughput_tps": prefill_tps,
        "decode_throughput_tps": decode_tps,
        "peak_memory_gb": mem_info["max_allocated_gb"]
    }


def print_beautiful_table(results: List[Dict], label: str):
    print("\n" + "=" * 110)
    print(f"📊 FINAL BENCHMARK RESULTS ({label})")
    print("=" * 110)
    header = f"{'Context':<10} {'Batch Size':<10} {'Prefill TPS':<15} {'Decode TPS':<15} {'Mem (GB)':<15}"
    print(header)
    print("-" * 110)
    for r in results:
        line = (f"{r['prefill_len']:<10} "
                f"{r['batch_size']:<10} "
                f"{r['prefill_throughput_tps']:<15.1f} "
                f"{r['decode_throughput_tps']:<15.1f} "
                f"{r['peak_memory_gb']:<15.2f}")
        print(line)
    print("=" * 110)


def main():
    print("=" * 120)
    print("🏆 FINAL COMPLETE vLLM + KDA BENCHMARK!")
    print("=" * 120)
    print(f"Model: {MODEL_PATH}")
    print(f"Context lengths: {CONTEXT_LENGTHS}")
    print(f"Output length: {OUTPUT_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"VLLM_USE_CGC_KDA: {os.environ.get('VLLM_USE_CGC_KDA')}")
    print()

    from vllm import LLM

    print("🚀 Initializing LLM with KDA ...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=True
    )

    all_results = []
    start_total = time.time()

    for ctx_len in CONTEXT_LENGTHS:
        print("\n" + "=" * 80)
        print(f"📌 TEST: Context Length = {ctx_len} (KDA MODE)")
        print("=" * 80)
        res = run_single_test(llm, ctx_len, OUTPUT_LENGTH, BATCH_SIZE)
        all_results.append(res)
        print(f"✅ Done for {ctx_len}")

    total_elapsed = time.time() - start_total

    print_beautiful_table(all_results, "vLLM + KDA")

    # Cleanup
    del llm
    clear_gpu_cache()

    # Save
    out_file = '/home/gs01/final_kda_benchmark_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Results saved to: {out_file}")
    print(f"Total time taken: {total_elapsed:.1f} seconds")


if __name__ == '__main__':
    main()
