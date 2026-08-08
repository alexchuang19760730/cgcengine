#!/usr/bin/env python3
"""
Server-side benchmark script for vLLM vs KDA
"""

import sys
import os
import json
import time
import numpy as np
from typing import List, Dict, Any

# Add paths
sys.path.insert(0, '/home/gs01')

# Model config
MODEL_PATH = '/home/gs01/models/Qwen/Qwen2___5-7B-Instruct'
TEST_PREFILL_LEN_LIST = [256, 512, 1024, 2048]
TEST_DECODE_LEN = 128
BATCH_SIZE = 4
GPU_MEM_UTIL = 0.7


def get_gpu_memory_stats() -> Dict[str, float]:
    """Get current GPU memory stats"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
            }
    except Exception:
        pass
    return {
        'allocated_gb': 0,
        'max_allocated_gb': 0,
        'reserved_gb': 0,
    }


def clear_cache():
    """Clear cache"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def run_single_benchmark(
    llm,
    prefill_len: int,
    decode_len: int,
    batch_size: int,
    num_iters: int = 5,
    warmup_iters: int = 2
) -> Dict[str, Any]:
    """Run single benchmark"""
    from vllm import SamplingParams

    # Prepare dummy prompt
    dummy_prompt_token_ids = np.random.randint(10000, size=(batch_size, prefill_len)).tolist()
    dummy_prompts = [{'prompt_token_ids': pt} for pt in dummy_prompt_token_ids]

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=decode_len,
        ignore_eos=True
    )

    # Warmup
    print(f"Warming up {warmup_iters} times...")
    for _ in range(warmup_iters):
        llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)

    # Benchmark
    latencies = []
    peak_mem_samples = []
    print(f"Running {num_iters} iterations...")
    for _ in range(num_iters):
        clear_cache()
        start = time.perf_counter()
        llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)
        latency = time.perf_counter() - start
        latencies.append(latency)
        mem_stats = get_gpu_memory_stats()
        peak_mem_samples.append(mem_stats['max_allocated_gb'])

    latencies = np.array(latencies)
    peak_mem_samples = np.array(peak_mem_samples)

    return {
        'prefill_len': prefill_len,
        'decode_len': decode_len,
        'batch_size': batch_size,
        'num_iters': num_iters,
        'latencies': latencies.tolist(),
        'avg_latency': float(np.mean(latencies)),
        'std_latency': float(np.std(latencies)),
        'percentiles': {
            'p10': float(np.percentile(latencies, 10)),
            'p25': float(np.percentile(latencies, 25)),
            'p50': float(np.percentile(latencies, 50)),
            'p75': float(np.percentile(latencies, 75)),
            'p90': float(np.percentile(latencies, 90)),
            'p99': float(np.percentile(latencies, 99)),
        },
        'avg_peak_gb': float(np.mean(peak_mem_samples)),
        'max_peak_gb': float(np.max(peak_mem_samples)),
    }


def main():
    from vllm import LLM

    print("=" * 80)
    print("Starting vLLM benchmark")
    print("=" * 80)

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=True,
    )
    print(f"Model loaded successfully!")

    # Run benchmarks
    all_results = {
        'model': MODEL_PATH.split('/')[-1],
        'timestamp': time.time(),
        'benchmarks': []
    }

    for prefill_len in TEST_PREFILL_LEN_LIST:
        print("\n" + "-" * 80)
        print(f"Running benchmark: prefill={prefill_len}, decode={TEST_DECODE_LEN}, batch={BATCH_SIZE}")
        print("-" * 80)

        result = run_single_benchmark(
            llm=llm,
            prefill_len=prefill_len,
            decode_len=TEST_DECODE_LEN,
            batch_size=BATCH_SIZE
        )

        print(f"Avg latency: {result['avg_latency']:.4f}s")
        print(f"Peak GPU mem (avg): {result['avg_peak_gb']:.2f}GB, (max): {result['max_peak_gb']:.2f}GB")
        print(f"Percentiles:")
        for k, v in result['percentiles'].items():
            print(f"  {k}: {v:.4f}s")

        all_results['benchmarks'].append(result)

    # Save results
    output_file = '/home/gs01/server_benchmark_results_baseline.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Results saved to {output_file}")
    print("=" * 80)
    print(json.dumps(all_results, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
