#!/usr/bin/env python3
"""
Final vLLM vs KDA benchmark script
- Single process mode to avoid memory measurement issues
- Complete memory and latency stats
"""

import sys
import os
import json
import time
import numpy as np
from typing import List, Dict, Any

# Set environment for single process
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Add paths
sys.path.insert(0, '/home/gs01')

# Configuration
MODEL_PATH = '/home/gs01/models/Qwen/Qwen2___5-7B-Instruct'
PREFILL_LENGTHS = [256, 512, 1024, 2048]
DECODE_LENGTH = 128
BATCH_SIZE = 4
GPU_MEM_UTIL = 0.7


def get_gpu_info() -> Dict[str, float]:
    """Get detailed GPU info and memory stats"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device)
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated(device)
            max_allocated = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            return {
                'device_name': device_name,
                'total_gb': total_memory / (1024 ** 3),
                'allocated_gb': allocated / (1024 ** 3),
                'max_allocated_gb': max_allocated / (1024 ** 3),
                'reserved_gb': reserved / (1024 ** 3),
            }
    except Exception as e:
        print(f"GPU info error: {e}")
        pass
    return {}


def clear_gpu_cache():
    """Clear GPU cache and reset peak memory"""
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
    num_iters: int = 5,
    num_warmups: int = 2
) -> Dict[str, Any]:
    """Run single test case and return results"""
    from vllm import SamplingParams

    # Prepare prompt
    dummy_token_ids = np.random.randint(10000, size=(batch_size, prefill_len)).tolist()
    dummy_prompts = [{'prompt_token_ids': t} for t in dummy_token_ids]

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=decode_len,
        ignore_eos=True,
        detokenize=False
    )

    # Warmup runs
    print(f"  Warmup ({num_warmups} runs) ...")
    for _ in range(num_warmups):
        llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)

    # Actual benchmark runs
    print(f"  Running benchmark ({num_iters} runs) ...")
    total_times = []
    peak_memory_samples = []

    for i in range(num_iters):
        clear_gpu_cache()
        t_start = time.perf_counter()
        outputs = llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        gpu_info = get_gpu_info()
        total_times.append(elapsed)
        peak_memory_samples.append(gpu_info.get('max_allocated_gb', 0))

    total_times_arr = np.array(total_times)
    peak_memory_arr = np.array(peak_memory_samples)

    return {
        'prefill_len': prefill_len,
        'decode_len': decode_len,
        'batch_size': batch_size,
        'num_iters': num_iters,
        'num_warmups': num_warmups,
        'total_times': total_times,
        'avg_total_time': float(np.mean(total_times_arr)),
        'std_total_time': float(np.std(total_times_arr)),
        'total_time_percentiles': {
            'p10': float(np.percentile(total_times_arr, 10)),
            'p25': float(np.percentile(total_times_arr, 25)),
            'p50': float(np.percentile(total_times_arr, 50)),
            'p75': float(np.percentile(total_times_arr, 75)),
            'p90': float(np.percentile(total_times_arr, 90)),
            'p99': float(np.percentile(total_times_arr, 99)),
        },
        'peak_memory_gb_samples': peak_memory_samples,
        'avg_peak_memory_gb': float(np.mean(peak_memory_arr)),
        'max_peak_memory_gb': float(np.max(peak_memory_arr)),
        'min_peak_memory_gb': float(np.min(peak_memory_arr)),
    }


def main():
    from vllm import LLM, EngineArgs

    print("=" * 80)
    print("Final vLLM vs KDA Benchmark (Single Process)")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"Prefill lengths: {PREFILL_LENGTHS}")
    print(f"Decode length: {DECODE_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    # 1) Baseline test (vanilla vLLM)
    print("=" * 80)
    print("Phase 1: Baseline (vanilla vLLM)")
    print("=" * 80)
    llm_baseline = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=True,
        distributed_executor_backend='multiprocessing'
    )
    baseline_results = []
    for prefill_len in PREFILL_LENGTHS:
        print(f"\nTesting prefill length: {prefill_len}")
        res = run_single_test(
            llm_baseline,
            prefill_len=prefill_len,
            decode_len=DECODE_LENGTH,
            batch_size=BATCH_SIZE
        )
        print(f"  Avg total time: {res['avg_total_time']:.4f}s")
        print(f"  Avg peak memory: {res['avg_peak_memory_gb']:.4f}GB")
        baseline_results.append(res)
    del llm_baseline
    clear_gpu_cache()

    # Save baseline results
    baseline_output_file = '/home/gs01/final_benchmark_baseline.json'
    with open(baseline_output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'Qwen2.5-7B-Instruct',
            'phase': 'baseline',
            'timestamp': time.time(),
            'tests': baseline_results
        }, f, indent=4, ensure_ascii=False)
    print(f"Baseline results saved to {baseline_output_file}")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)

    # Generate summary report
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    print(f"{'Prefill Len':<12} {'Avg Time':<12} {'Avg Peak Mem':<14}")
    print("-" * 40)
    for t in baseline_results:
        print(f"{t['prefill_len']:<12} {t['avg_total_time']:<12.4f} {t['avg_peak_memory_gb']:<14.2f}")
    print("\nFull JSON saved!")


if __name__ == '__main__':
    main()
