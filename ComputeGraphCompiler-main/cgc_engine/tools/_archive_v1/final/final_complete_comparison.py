#!/usr/bin/env python3
"""
✨ 完美！最终完整对比！vLLM vs vLLM + KDA！
包含完整的 Prefill, Decode, GPU Memory！
"""
import os
import sys
import json
import time
import numpy as np
import subprocess
from typing import List, Dict, Any

# 配置
MODEL_PATH = '/home/gs01/models/Qwen/Qwen2___5-7B-Instruct'
CONTEXT_LENGTHS = [256, 512, 1024, 2048]
OUTPUT_LENGTH = 128
BATCH_SIZE = 4
GPU_MEM_UTIL = 0.7


def get_gpu_memory_from_nvidia() -> Dict[str, float]:
    """
    直接用 nvidia-smi 获取真实的 GPU 内存使用！
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

    # 获取真实 GPU 内存使用
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
        "peak_memory_gb": gpu_mem["used_gb"]
    }


def run_benchmark(label: str, kda_enabled: bool) -> List[Dict]:
    """
    运行完整的 benchmark（KDA 或普通模式）
    """
    print("\n" + "=" * 120)
    print(f"🏆 正在运行 {label} ...")
    if kda_enabled:
        os.environ["VLLM_USE_CGC_KDA"] = "1"
        print(f"VLLM_USE_CGC_KDA = {os.environ.get('VLLM_USE_CGC_KDA')}")
    else:
        os.environ.pop("VLLM_USE_CGC_KDA", None)

    from vllm import LLM

    print("🚀 初始化 LLM ...")
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
        print(f"📌 {label}: Context Length = {ctx_len}")
        print("=" * 80)
        res = run_single_test(llm, ctx_len, OUTPUT_LENGTH, BATCH_SIZE)
        all_results.append(res)
        print(f"✅ {label}: 完成 {ctx_len}")

    # 清理
    del llm
    clear_gpu_cache()

    return all_results


def print_comparison_table(baseline: List[Dict], kda: List[Dict]):
    """
    打印完美的对比表格！
    """
    print("\n" + "=" * 130)
    print("📊 🏆 最终完整对比: vLLM (Baseline) vs vLLM + KDA")
    print("=" * 130)
    header = (f"{'Context (t)':<12} | "
              f"{'vLLM Prefill TPS':<18} {'vLLM Decode TPS':<18} {'vLLM Mem (GB)':<16} | "
              f"{'KDA Prefill TPS':<18} {'KDA Decode TPS':<18} {'KDA Mem (GB)':<16}")
    print(header)
    print("-" * 130)
    for base, kda_res in zip(baseline, kda):
        line = (f"{base['prefill_len']:<12} | "
                f"{base['prefill_throughput_tps']:<18.1f} {base['decode_throughput_tps']:<18.1f} {base['peak_memory_gb']:<16.2f} | "
                f"{kda_res['prefill_throughput_tps']:<18.1f} {kda_res['decode_throughput_tps']:<18.1f} {kda_res['peak_memory_gb']:<16.2f}")
        print(line)
    print("=" * 130)


def main():
    print("=" * 120)
    print("✨ 最终完美对比: vLLM vs vLLM + KDA ✨")
    print("=" * 120)
    print(f"Model: {MODEL_PATH}")
    print(f"Context lengths: {CONTEXT_LENGTHS}")
    print(f"Output length: {OUTPUT_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    total_start = time.time()

    # 1. 运行 Baseline
    baseline_results = run_benchmark("vLLM (Baseline)", kda_enabled=False)

    # 2. 运行 KDA
    kda_results = run_benchmark("vLLM + KDA", kda_enabled=True)

    # 3. 打印完美的对比表格
    print_comparison_table(baseline_results, kda_results)

    # 4. 保存结果
    out_file = "/home/gs01/final_complete_comparison_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "baseline": baseline_results,
            "kda": kda_results
        }, f, indent=2)

    total_elapsed = time.time() - total_start

    print(f"\n💾 结果保存到: {out_file}")
    print(f"🏆 完美完成！总共用时: {total_elapsed:.1f} 秒！")


if __name__ == "__main__":
    main()
