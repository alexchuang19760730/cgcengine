#!/usr/bin/env python3
"""
✨ 最简单、最可靠的最终对比！先跑 Baseline，再跑 KDA
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
    llm.generate(test_prompts, sampling_params, use_tqdm=False)

    clear_gpu_cache()
    t0 = time.perf_counter()
    llm.generate(test_prompts, sampling_params, use_tqdm=False)
    t1 = time.perf_counter()
    elapsed = t1 - t0

    gpu_mem = get_gpu_memory_from_nvidia()

    total_prefill_tokens = batch_size * prefill_len
    total_decode_tokens = batch_size * decode_len

    return {
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        "batch_size": batch_size,
        "total_time_s": elapsed,
        "prefill_throughput_tps": total_prefill_tokens / elapsed,
        "decode_throughput_tps": total_decode_tokens / elapsed,
        "gpu_memory_used_gb": gpu_mem["used_gb"]
    }


def run_single_label(label: str, kda_enabled: bool) -> List[Dict]:
    print("\n" + "=" * 120)
    print(f"🏆 {label}")
    print("=" * 120)

    # Set env var
    if kda_enabled:
        os.environ["VLLM_USE_CGC_KDA"] = "1"
    else:
        os.environ.pop("VLLM_USE_CGC_KDA", None)

    from vllm import LLM

    print(f"🚀 Initializing LLM...")
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
        print(f"📌 Testing ctx: {ctx_len}")
        res = run_single_test(llm, ctx_len, OUTPUT_LENGTH, BATCH_SIZE)
        all_results.append(res)
        print(f"✅ Done: {ctx_len}")

    # Cleanup
    del llm
    clear_gpu_cache()

    return all_results


def print_final_comparison(baseline: List[Dict], kda: List[Dict]):
    print("\n" + "=" * 130)
    print("📊 🏆 🏆 🏆 最终完整对比：vLLM (Baseline) vs vLLM + KDA 🏆 🏆 🏆")
    print("=" * 130)
    header = (f"{'Context':<10} | "
              f"{'vLLM Pref TPS':<18} {'vLLM Dec TPS':<18} {'vLLM Mem (GB)':<15} | "
              f"{'KDA Pref TPS':<18} {'KDA Dec TPS':<18} {'KDA Mem (GB)':<15}")
    print(header)
    print("-" * 130)
    for base, k in zip(baseline, kda):
        line = (f"{base['prefill_len']:<10} | "
                f"{base['prefill_throughput_tps']:<18.1f} {base['decode_throughput_tps']:<18.1f} {base['gpu_memory_used_gb']:<15.2f} | "
                f"{k['prefill_throughput_tps']:<18.1f} {k['decode_throughput_tps']:<18.1f} {k['gpu_memory_used_gb']:<15.2f}")
        print(line)
    print("=" * 130)


def main():
    print("=" * 120)
    print("✨ 最终完美对比 ✨")
    print("=" * 120)

    # 先跑 Baseline
    baseline = run_single_label("vLLM (Baseline)", kda_enabled=False)

    # 再跑 KDA
    kda = run_single_label("vLLM + KDA", kda_enabled=True)

    print_final_comparison(baseline, kda)

    out_file = "/home/gs01/final_comparison_results.json"
    with open(out_file, "w") as f:
        json.dump({"baseline": baseline, "kda": kda}, f, indent=2)

    print(f"\n💾 Results saved to: {out_file}")


if __name__ == "__main__":
    main()
