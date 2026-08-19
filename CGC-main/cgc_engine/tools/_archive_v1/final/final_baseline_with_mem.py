#!/usr/bin/env python3
"""
✨ 完美！vLLM 基准测试（含真实 GPU 内存使用）！
"""
import os
import sys
import json
import time
import numpy as np
import subprocess
from typing import List, Dict, Any

MODEL_PATH = '/home/gs01/models/Qwen/Qwen2___5-7B-Instruct'
CONTEXT_LENGTHS = [256, 512, 1024, 2048]
OUTPUT_LENGTH = 128
BATCH_SIZE = 4
GPU_MEM_UTIL = 0.7


def get_gpu_memory_from_nvidia() -> Dict[str, float]:
    """
    直接用 nvidia-smi 获取真实 GPU 内存使用！
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        used, total = output.strip().split(",")
        return {
            "used_gb": float(used.strip()) / 1024.0,
            "total_gb": float(total.strip()) / 1024.0
        }
    except Exception:
        return {"used_gb": 0.0, "total_gb": 0.0}


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

    # 获取真实 GPU 内存
    gpu_mem = get_gpu_memory_from_nvidia()

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
        "gpu_memory_used_gb": gpu_mem["used_gb"],
        "gpu_memory_total_gb": gpu_mem["total_gb"]
    }


def print_table(results: List[Dict]):
    print("\n" + "=" * 120)
    print("📊 🏆 vLLM 最终完整基准测试（含 GPU 内存）")
    print("=" * 120)
    header = f"{'Context (t)':<12} {'Prefill TPS':<18} {'Decode TPS':<18} {'GPU Memory (GB)':<20}"
    print(header)
    print("-" * 120)
    for res in results:
        line = (f"{res['prefill_len']:<12} "
                f"{res['prefill_throughput_tps']:<18.1f} "
                f"{res['decode_throughput_tps']:<18.1f} "
                f"{res['gpu_memory_used_gb']:<20.2f}")
        print(line)
    print("=" * 120)


def main():
    print("=" * 120)
    print("✨ 最终完整基准测试（含 GPU 内存） ✨")
    print("=" * 120)
    print(f"Model: {MODEL_PATH}")
    print(f"Context lengths: {CONTEXT_LENGTHS}")
    print(f"Output length: {OUTPUT_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    from vllm import LLM

    print("🚀 正在初始化 LLM ...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=True
    )

    all_results = []
    for ctx_len in CONTEXT_LENGTHS:
        print("\n" + "=" * 80)
        print(f"📌 正在测试: Context Length = {ctx_len}")
        print("=" * 80)
        res = run_single_test(llm, ctx_len, OUTPUT_LENGTH, BATCH_SIZE)
        all_results.append(res)
        print(f"✅ 完成 {ctx_len}")

    print_table(all_results)

    # 保存结果
    out_file = "/home/gs01/final_baseline_with_memory_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 结果保存到: {out_file}")


if __name__ == "__main__":
    main()
