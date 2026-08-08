#!/bin/bash

# 完整的 vLLM vs vLLM+KDA 端到端对比测试脚本！

cd /home/gs01

echo "========================================"
echo "Step 1: Run vLLM Baseline"
echo "========================================"

# 运行 Baseline
python3 << 'EOF'
import os
import sys
import json
import time
import numpy as np
from vllm import LLM, SamplingParams

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
sys.path.insert(0, '/home/gs01')
MODEL = '/home/gs01/models/Qwen/Qwen2___5-7B-Instruct'
PREFILL = [256, 512, 1024, 2048]
DECODE = 128
BS = 4
GPU_MEM = 0.7

results = []

import torch

def get_gpu():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()/(1024**3), torch.cuda.max_memory_allocated()/(1024**3)
    return 0,0

def clear():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

llm = LLM(
    model=MODEL,
    tensor_parallel_size=1,
    gpu_memory_utilization=GPU_MEM,
    max_model_len=4096,
    enforce_eager=True,
    disable_log_stats=True
)

for pf_len in PREFILL:
    print(f"Prefill: {pf_len} tokens")
    dummy_ids = np.random.randint(10000, size=(BS, pf_len)).tolist()
    prompts = [{'prompt_token_ids': x} for x in dummy_ids]
    params = SamplingParams(temperature=0, max_tokens=DECODE, ignore_eos=True)

    # Warmup
    for _ in range(2):
        llm.generate(prompts, params, use_tqdm=False)

    # Test
    total_times = []
    peak_mems = []
    for _ in range(3):
        clear()
        t0 = time.perf_counter()
        llm.generate(prompts, params, use_tqdm=False)
        t1 = time.perf_counter()
        total_times.append(t1 - t0)
        _, peak_gb = get_gpu()
        peak_mems.append(peak_gb)

    avg_time = np.mean(total_times)
    avg_peak = np.mean(peak_mems)
    print(f"  Avg total time: {avg_time:.4f}s, Avg peak: {avg_peak:.2f}GB")
    results.append({
        'prefill_len': pf_len,
        'avg_total_time': float(avg_time),
        'avg_peak_gb': float(avg_peak),
        'total_times': [float(x) for x in total_times]
    })

del llm
clear()

with open('/home/gs01/comparison_baseline.json', 'w') as f:
    json.dump({'type': 'baseline', 'results': results}, f, indent=2)
EOF

echo "========================================"
echo "Step 2: Done!"
echo "========================================"
ls -la /home/gs01/comparison_baseline.json
echo "========================================"
cat /home/gs01/comparison_baseline.json
echo "========================================"
