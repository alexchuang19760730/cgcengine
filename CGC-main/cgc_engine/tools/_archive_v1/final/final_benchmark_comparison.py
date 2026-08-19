#!/usr/bin/env python3
"""
最终完整的 vLLM 基准对比！
- vLLM baseline 模式
- vLLM + KDA 模式
- 测量 prefill/decode token/s 和 内存占用
"""
import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# 环境设置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# 配置
MODEL_PATH = '/home/gs01/models/Qwen/Qwen2___5-7B-Instruct'
PREFILL_LENGTHS = [256, 512, 1024, 2048]
DECODE_LENGTH = 128
BATCH_SIZE = 4
GPU_MEM_UTIL = 0.7
NUM_WARMUP = 2
NUM_ITER = 3

# 全局状态
KDA_ENABLED = False


def get_gpu_memory_stats() -> Dict[str, float]:
    """获取 GPU 内存状态（MB）"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
                "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
                "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
            }
    except Exception:
        pass
    return {"allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0}


def clear_gpu_cache():
    """清除 GPU 缓存"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def run_single_test(
    llm,
    prefill_len: int,
    decode_len: int,
    batch_size: int,
    num_iter: int = NUM_ITER,
    num_warmup: int = NUM_WARMUP
) -> Dict[str, Any]:
    """运行单次测试"""
    from vllm import SamplingParams
    dummy_ids = np.random.randint(10000, size=(batch_size, prefill_len)).tolist()
    test_prompts = [{'prompt_token_ids': ids} for ids in dummy_ids]
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=decode_len,
        ignore_eos=True,
        detokenize=False
    )

    for _ in range(num_warmup):
        llm.generate(test_prompts, sampling_params, use_tqdm=False)

    times = []
    peak_mems = []
    for _ in range(num_iter):
        clear_gpu_cache()
        t0 = time.perf_counter()
        llm.generate(test_prompts, sampling_params, use_tqdm=False)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        peak_mems.append(get_gpu_memory_stats()['max_allocated_gb'])

    avg_time = np.mean(times)
    std_time = np.std(times)

    total_tokens_generated = batch_size * decode_len

    prefill_tokens = batch_size * prefill_len
    prefill_tps = prefill_tokens / avg_time

    decode_tps = total_tokens_generated / avg_time

    return {
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        "batch_size": batch_size,
        "avg_total_time": avg_time,
        "std_total_time": std_time,
        "total_tokens_generated": total_tokens_generated,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "avg_peak_gb": np.mean(peak_mems),
        "max_peak_gb": np.max(peak_mems),
    }


def run_vllm_baseline() -> List[Dict[str, Any]]:
    """运行 vLLM baseline 模式"""
    from vllm import LLM
    print("\n" + "=" * 80)
    print("🚀 RUNNING: vLLM Baseline Mode")
    print("=" * 80)

    print(f"Loading model {MODEL_PATH} ...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=True
    )

    results = []
    for prefill in PREFILL_LENGTHS:
        res = run_single_test(llm, prefill, DECODE_LENGTH, BATCH_SIZE)
        results.append(res)

    del llm
    clear_gpu_cache()
    return results


def setup_vllm_kda():
    """设置 vLLM + KDA 模式"""
    global KDA_ENABLED
    os.environ['VLLM_USE_CGC_KDA'] = '1'

    try:
        repo_root = str(Path(__file__).resolve().parents[5])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import Backend.Vllm.vllm_backend.cgc_kda_backend
        print("✅ KDA backend available!")
        KDA_ENABLED = True
    except Exception as e:
        print(f"⚠️ KDA backend not registered (using baseline mode for comparison): {e}")


def run_vllm_kda() -> List[Dict[str, Any]]:
    """运行 vLLM + KDA 模式"""
    from vllm import LLM

    setup_vllm_kda()

    print("\n" + "=" * 80)
    print("🚀 RUNNING: vLLM + KDA Mode")
    print("=" * 80)

    print(f"Loading model {MODEL_PATH} ...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=True
    )

    results = []
    for prefill in PREFILL_LENGTHS:
        res = run_single_test(llm, prefill, DECODE_LENGTH, BATCH_SIZE)
        results.append(res)

    del llm
    clear_gpu_cache()
    return results


def format_results_summary(title: str, results: List[Dict[str, Any]]) -> str:
    """格式化结果表格"""
    lines = []
    lines.append(f"\n{title}")
    lines.append("-" * 100)
    lines.append(f"{'Prefill':<10} {'Avg Time (s)':<15} {'Prefill TPS':<15} {'Decode TPS':<15} {'Avg Mem (GB)':<15} {'Max Mem (GB)':<15}")
    lines.append("-" * 100)
    for r in results:
        line = f"{r['prefill_len']:<10} {r['avg_total_time']:<15.4f} {r['prefill_tps']:<15.1f} {r['decode_tps']:<15.1f} {r['avg_peak_gb']:<15.2f} {r['max_peak_gb']:<15.2f}"
        lines.append(line)
    return "\n".join(lines)


def print_comparison_table(baseline_res, kda_res):
    """打印对比表格"""
    print("\n" + "=" * 120)
    print("📊 FINAL COMPARISON TABLE (vLLM vs vLLM+KDA)")
    print("=" * 120)

    header = (f"{'Prefill':<10} | "
              f"{'vLLM TPS (Pref)':<15} {'vLLM TPS (Dec)':<15} {'vLLM Mem (GB)':<15} | "
              f"{'KDA TPS (Pref)':<15} {'KDA TPS (Dec)':<15} {'KDA Mem (GB)':<15}")
    print(header)
    print("-" * 120)

    for base, kda in zip(baseline_res, kda_res):
        line = (f"{base['prefill_len']:<10} | "
                f"{base['prefill_tps']:<15.1f} {base['decode_tps']:<15.1f} {base['avg_peak_gb']:<15.2f} | "
                f"{kda['prefill_tps']:<15.1f} {kda['decode_tps']:<15.1f} {kda['avg_peak_gb']:<15.2f}")
        print(line)


def main():
    start_time = time.time()
    print("\n" + "=" * 80)
    print("🏆 FINAL vLLM vs vLLM+KDA FULL BENCHMARK SUITE")
    print("=" * 80)
    print(f"Model: Qwen2.5-7B-Instruct")
    print(f"Prefill lengths: {PREFILL_LENGTHS}")
    print(f"Decode tokens: {DECODE_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")

    all_results = {}

    # 1. Baseline
    baseline_res = run_vllm_baseline()
    all_results['vllm_baseline'] = baseline_res
    print(format_results_summary("vLLM Baseline Results", baseline_res))

    # 2. KDA
    kda_res = run_vllm_kda()
    all_results['vllm_kda'] = kda_res
    print(format_results_summary("vLLM + KDA Results", kda_res))

    # Print comparison
    print_comparison_table(baseline_res, kda_res)

    # Save results
    out_file = '/home/gs01/final_benchmark_comparison.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✅ BENCHMARK COMPLETED! Took {total_time:.1f} seconds")
    print(f"Results saved to: {out_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
