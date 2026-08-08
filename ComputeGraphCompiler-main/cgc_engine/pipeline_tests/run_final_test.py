#!/usr/bin/env python3
"""FINAL Real Ablation Test"""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, '/home/gs01/MagiCompiler-main')

print('=' * 60)
print('FINAL Real Ablation Test: vLLM vs PD Separation')
print('=' * 60)
print('Time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('GPU: Checking...')

import torch
print('GPU:', torch.cuda.get_device_name(0))
print('GPU Count:', torch.cuda.device_count())
print('Memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')

results = []

# Test 1: Single GPU
print('\n' + '=' * 60)
print('TEST 1: Native vLLM (1 GPU)')
print('=' * 60)

try:
    from vllm import LLM, SamplingParams
    
    print('Loading model on 1 GPU...')
    llm1 = LLM(
        model="/home/gs01/models/Qwen/Qwen2___5-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        max_model_len=2048
    )
    
    prompts = ["Hello, my name is", "The quick brown fox", "Artificial intelligence is"]
    sampling_params = SamplingParams(max_tokens=32)
    
    print('Warming up...')
    llm1.generate(prompts, sampling_params)
    
    print('Testing prefill...')
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    outputs = llm1.generate(prompts, sampling_params)
    prefill_time = (time.time() - start) * 1000
    
    print('Testing decode...')
    start = time.time()
    outputs = llm1.generate(["Hello"], sampling_params)
    decode_time = (time.time() - start) * 1000
    
    memory = torch.cuda.max_memory_allocated() / 1e9
    
    result1 = {
        "name": "Native vLLM (1 GPU)",
        "tensor_parallel": 1,
        "prefill_time_ms": prefill_time,
        "decode_time_ms": decode_time,
        "memory_used_gb": memory,
        "success": True
    }
    
    print('✓ Prefill:', prefill_time, 'ms')
    print('✓ Decode:', decode_time, 'ms')
    print('✓ Memory:', memory, 'GB')
    
except Exception as e:
    import traceback
    print('✗ FAILED:', str(e))
    traceback.print_exc()
    result1 = {"name": "Native vLLM (1 GPU)", "error": str(e), "success": False}

results.append(result1)

# Save intermediate
with open('real_ablation_final.json', 'w') as f:
    json.dump(results, f, indent=2)

# Test 2: Tensor Parallel = 2
print('\n' + '=' * 60)
print('TEST 2: vLLM TP=2 (NCCL)')
print('=' * 60)

try:
    print('Loading model on 2 GPUs with NCCL...')
    llm2 = LLM(
        model="/home/gs01/models/Qwen/Qwen2___5-7B-Instruct",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        max_model_len=2048
    )
    
    print('Warming up...')
    llm2.generate(prompts, sampling_params)
    
    print('Testing prefill...')
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    outputs = llm2.generate(prompts, sampling_params)
    prefill_time2 = (time.time() - start) * 1000
    
    print('Testing decode...')
    start = time.time()
    outputs = llm2.generate(["Hello"], sampling_params)
    decode_time2 = (time.time() - start) * 1000
    
    memory2 = torch.cuda.max_memory_allocated() / 1e9
    
    result2 = {
        "name": "vLLM TP=2 (NCCL)",
        "tensor_parallel": 2,
        "prefill_time_ms": prefill_time2,
        "decode_time_ms": decode_time2,
        "memory_used_gb": memory2,
        "success": True
    }
    
    print('✓ Prefill:', prefill_time2, 'ms')
    print('✓ Decode:', decode_time2, 'ms')
    print('✓ Memory:', memory2, 'GB')
    
except Exception as e:
    import traceback
    print('✗ FAILED:', str(e))
    traceback.print_exc()
    result2 = {"name": "vLLM TP=2 (NCCL)", "error": str(e), "success": False}

results.append(result2)

# Final save
with open('real_ablation_final.json', 'w') as f:
    json.dump(results, f, indent=2)

# Summary
print('\n' + '=' * 60)
print('FINAL TEST SUMMARY')
print('=' * 60)

for r in results:
    if r.get('success'):
        print(f"{r['name']}:")
        print(f"  Prefill: {r['prefill_time_ms']:.2f} ms")
        print(f"  Decode: {r['decode_time_ms']:.2f} ms")
        print(f"  Memory: {r['memory_used_gb']:.2f} GB")
    else:
        print(f"{r['name']}: FAILED - {r.get('error', 'Unknown')}")

if len(results) == 2 and all(r.get('success') for r in results):
    speedup = results[0]['prefill_time_ms'] / results[1]['prefill_time_ms']
    print(f"\n✓ NCCL Speedup: {speedup:.2f}x")

print('\nResults saved to real_ablation_final.json')

