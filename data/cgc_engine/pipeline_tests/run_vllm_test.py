#!/usr/bin/env python3
import sys
import time
import json
sys.path.insert(0, '/home/gs01/MagiCompiler-main')

print('=' * 60)
print('vLLM Real Test with Qwen1.5-7B-Chat')
print('=' * 60)

import torch
print('GPU:', torch.cuda.get_device_name(0))
print('GPU Count:', torch.cuda.device_count())

results = []

try:
    from vllm import LLM, SamplingParams
    
    # Test 1: Single GPU
    print('\n[1] Testing Native vLLM (1 GPU)...')
    llm1 = LLM(
        model="/home/gs01/models/Qwen1.5-7B-Chat",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        max_model_len=2048
    )
    
    prompts = ["Hello, my name is", "The quick brown fox", "Artificial intelligence is"]
    sampling_params = SamplingParams(max_tokens=32)
    
    print('Warming up...')
    llm1.generate(prompts, sampling_params)
    
    print('Measuring prefill...')
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    outputs = llm1.generate(prompts, sampling_params)
    prefill1 = (time.time() - start) * 1000
    
    print('Measuring decode...')
    start = time.time()
    outputs = llm1.generate(["Hello"], sampling_params)
    decode1 = (time.time() - start) * 1000
    
    memory1 = torch.cuda.max_memory_allocated() / 1e9
    
    result1 = {
        "name": "Native vLLM (1 GPU)",
        "prefill_time_ms": prefill1,
        "decode_time_ms": decode1,
        "memory_used_gb": memory1,
        "success": True
    }
    
    print('OK - Prefill:', prefill1, 'ms')
    print('OK - Decode:', decode1, 'ms')
    print('OK - Memory:', memory1, 'GB')
    
    results.append(result1)
    
    # Test 2: TP=2 (NCCL)
    print('\n[2] Testing vLLM TP=2 (NCCL)...')
    llm2 = LLM(
        model="/home/gs01/models/Qwen1.5-7B-Chat",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        max_model_len=2048
    )
    
    print('Warming up...')
    llm2.generate(prompts, sampling_params)
    
    print('Measuring prefill...')
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    outputs = llm2.generate(prompts, sampling_params)
    prefill2 = (time.time() - start) * 1000
    
    print('Measuring decode...')
    start = time.time()
    outputs = llm2.generate(["Hello"], sampling_params)
    decode2 = (time.time() - start) * 1000
    
    memory2 = torch.cuda.max_memory_allocated() / 1e9
    
    result2 = {
        "name": "vLLM TP=2 (NCCL)",
        "prefill_time_ms": prefill2,
        "decode_time_ms": decode2,
        "memory_used_gb": memory2,
        "success": True
    }
    
    print('OK - Prefill:', prefill2, 'ms')
    print('OK - Decode:', decode2, 'ms')
    print('OK - Memory:', memory2, 'GB')
    
    results.append(result2)
    
except Exception as e:
    import traceback
    print('FAILED:', str(e))
    traceback.print_exc()
    results.append({"name": "Test", "error": str(e), "success": False})

# Save results
with open('vllm_real_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n' + '=' * 60)
print('TEST SUMMARY')
print('=' * 60)
for r in results:
    if r.get('success'):
        print(r['name'], ': Prefill=', r['prefill_time_ms'], 'ms, Decode=', r['decode_time_ms'], 'ms, Memory=', r['memory_used_gb'], 'GB')
    else:
        print(r['name'], ': FAILED -', r.get('error'))

if len(results) >= 2:
    speedup = results[0]['prefill_time_ms'] / results[1]['prefill_time_ms']
    print('NCCL Speedup:', speedup, 'x')

