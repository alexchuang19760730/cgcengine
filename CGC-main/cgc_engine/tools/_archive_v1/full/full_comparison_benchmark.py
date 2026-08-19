#!/usr/bin/env python3
"""
完整的 vLLM vs vLLM+KDA 端到端推理对比脚本！
测量 prefill/decode/memory 的完整表现
"""

import os
import sys
import json
import time
import glob
import numpy as np
from typing import List, Dict, Any

# 全局配置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
sys.path.insert(0, '/home/gs01')

MODEL_PATH = '/home/gs01/models/Qwen/Qwen2___5-7B-Instruct'
PREFILL_LENGTHS = [256, 512, 1024, 2048]
DECODE_LENGTH = 128
BATCH_SIZE = 4
GPU_MEM_UTIL = 0.7


def clear_stats():
    for f in glob.glob('/tmp/kda_stats_pid*.json'):
        try:
            os.remove(f)
        except Exception:
            pass


def get_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3)
            }
    except Exception:
        pass
    return {'allocated_gb':0, 'reserved_gb':0, 'max_allocated_gb':0}


def clear_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def read_kda_stats():
    stats = []
    for f in sorted(glob.glob('/tmp/kda_stats_pid*.json')):
        try:
            with open(f, 'r') as fp:
                stats.append(json.load(fp))
        except Exception:
            pass
    return stats


def run_single_test(llm, prefill_len, decode_len, batch_size, num_iters=3, num_warmups=2):
    from vllm import SamplingParams
    dummy_token_ids = np.random.randint(10000, size=(batch_size, prefill_len)).tolist()
    dummy_prompts = [{'prompt_token_ids': t} for t in dummy_token_ids]
    sampling_params = SamplingParams(temperature=0, max_tokens=decode_len, ignore_eos=True)

    for _ in range(num_warmups):
        llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)

    total_times = []
    peak_memory_samples = []
    for _ in range(num_iters):
        clear_cache()
        t0 = time.perf_counter()
        llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)
        t1 = time.perf_counter()
        total_times.append(t1 - t0)
        gpu_info = get_gpu_memory()
        peak_memory_samples.append(gpu_info.get('max_allocated_gb', 0))
    total_times_arr = np.array(total_times)
    return {
        'prefill_len': prefill_len,
        'avg_total_time': float(np.mean(total_times_arr)),
        'total_time_percentiles': {
            'p50': float(np.percentile(total_times_arr,50))
        },
        'avg_peak_gb': float(np.mean(peak_memory_samples)),
        'max_peak_gb': float(np.max(peak_memory_samples)),
    }


def main():
    print("="*80)
    print("🚀 完整的 vLLM vs vLLM+KDA 端到端对比测试")
    print("="*80)

    all_comparison_data = []

    # 第一部分：vLLM Baseline
    print("\n" + "="*80)
    print("PHASE 1: vLLM Baseline（原生）")
    print("="*80)
    import vllm
    from vllm import LLM
    llm_baseline = LLM(model=MODEL_PATH, tensor_parallel_size=1, gpu_memory_utilization=GPU_MEM_UTIL, max_model_len=4096, enforce_eager=True, disable_log_stats=True)
    baseline_tests = []
    for prefill_len in PREFILL_LENGTHS:
        print(f"\nPrefill: {prefill_len} tokens")
        res = run_single_test(llm_baseline, prefill_len=prefill_len, decode_len=DECODE_LENGTH, batch_size=BATCH_SIZE)
        print(f"  Avg total time: {res['avg_total_time']:.4f}s, Avg peak mem: {res['avg_peak_gb']:.2f}GB")
        baseline_tests.append(res)
    del llm_baseline
    clear_cache()

    # 第二部分：vLLM + KDA
    print("\n" + "="*80)
    print("PHASE 2: vLLM + KDA（自定义后端）")
    print("="*80)
    os.environ['VLLM_USE_CGC_KDA'] = '1'
    clear_stats()
    import Backend.Vllm.vllm_backend.cgc_kda_backend
    llm_kda = LLM(model=MODEL_PATH, tensor_parallel_size=1, gpu_memory_utilization=GPU_MEM_UTIL, max_model_len=4096, enforce_eager=True, disable_log_stats=True)
    kda_tests = []
    for prefill_len in PREFILL_LENGTHS:
        print(f"\nPrefill: {prefill_len} tokens")
        res = run_single_test(llm_kda, prefill_len=prefill_len, decode_len=DECODE_LENGTH, batch_size=BATCH_SIZE)
        res['kda_stats'] = read_kda_stats()
        print(f"  Avg total time: {res['avg_total_time']:.4f}s, Avg peak mem: {res['avg_peak_gb']:.2f}GB")
        kda_tests.append(res)
    del llm_kda
    clear_cache()

    # 整理对比数据
    print("\n" + "="*80)
    print("📊 完整对比表")
    print("="*80)
    print(f"{'Prefill':<12} | {'vLLM Time':<12} | {'vLLM Mem':<12} | {'vLLM+KDA Time':<14} | {'vLLM+KDA Mem':<14}")
    print("-"*80)
    all_results = {'baseline': baseline_tests, 'kda': kda_tests}
    for bl, kd in zip(baseline_tests, kda_tests):
        t_vllm = bl['avg_total_time']
        m_vllm = bl['avg_peak_gb']
        t_kda = kd['avg_total_time']
        m_kda = kd['avg_peak_gb']
        all_comparison_data.append({
            'prefill_len': bl['prefill_len'],
            'vllm_avg_time': t_vllm,
            'vllm_avg_mem_gb': m_vllm,
            'kda_avg_time': t_kda,
            'kda_avg_mem_gb': m_kda,
        })
        print(f"{bl['prefill_len']:<12} | {t_vllm:<12.4f} | {m_vllm:<12.2f} | {t_kda:<14.4f} | {m_kda:<14.2f}")

    # 保存完整结果
    output_file = '/home/gs01/full_comparison_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'Qwen2.5-7B-Instruct',
            'decode_len': DECODE_LENGTH,
            'batch_size': BATCH_SIZE,
            'comparison_table': all_comparison_data,
            'full_baseline_results': baseline_tests,
            'full_kda_results': kda_tests
        }, f, indent=4)

    print("\n" + "="*80)
    print(f"✅ 完整结果已保存到: {output_file}")
    print("="*80)
    return 0


if __name__ == '__main__':
    main()
