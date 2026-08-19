#!/usr/bin/env python3
"""REAL Ablation Test: Native vLLM vs PD Separation + NCCL"""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, '/home/gs01/MagiCompiler-main')

print('=' * 60)
print('REAL Ablation Test: vLLM vs PD Separation')
print('=' * 60)
print('Time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

results = []

test_configs = [
    {"name": "Native vLLM (1 GPU)", "tensor_parallel": 1},
    {"name": "vLLM TP=2 (NCCL)", "tensor_parallel": 2},
]

def run_real_vllm_test(config):
    import torch
    import time
    
    print('')
    print('Testing:', config["name"])
    print('Config: tensor_parallel=%d' % config["tensor_parallel"])
    
    start_time = time.time()
    
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        print('Memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
        print('GPU Count:', torch.cuda.device_count())
    else:
        print('WARNING: No GPU available!')
    
    try:
        from vllm import LLM, SamplingParams
        
        print('Loading model from local path...')
        llm = LLM(model="/home/gs01/models/Qwen/Qwen2___5-7B-Instruct", 
                  tensor_parallel_size=config["tensor_parallel"],
                  gpu_memory_utilization=0.8,
                  trust_remote_code=True)
        
        print('Running prefill test...')
        prompts = ["Hello, my name is", "The quick brown fox", "Artificial intelligence is"]
        sampling_params = SamplingParams(max_tokens=32)
        
        print('Warming up...')
        llm.generate(prompts, sampling_params)
        
        print('Measuring prefill...')
        prefill_start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        prefill_time = time.time() - prefill_start
        
        print('Measuring decode...')
        decode_start = time.time()
        outputs = llm.generate(["Hello"], sampling_params)
        decode_time = time.time() - decode_start
        
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        
        result = {
            "name": config["name"],
            "prefill_time_ms": prefill_time * 1000,
            "decode_time_ms": decode_time * 1000,
            "memory_used_gb": memory_used,
            "gpu_count": torch.cuda.device_count(),
            "tensor_parallel": config["tensor_parallel"],
            "success": True
        }
        
        print('OK - Prefill:', prefill_time*1000, 'ms')
        print('OK - Decode:', decode_time*1000, 'ms')
        print('OK - Memory:', memory_used, 'GB')
        
    except Exception as e:
        import traceback
        print('FAILED - Error:', str(e))
        traceback.print_exc()
        result = {
            "name": config["name"],
            "error": str(e),
            "success": False
        }
    
    elapsed = time.time() - start_time
    print('Test time:', elapsed, 's')
    
    return result

for config in test_configs:
    result = run_real_vllm_test(config)
    results.append(result)
    
    with open('real_ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

print('')
print('=' * 60)
print('TEST SUMMARY')
print('=' * 60)
for r in results:
    if r.get('success'):
        print(r['name'], ': Prefill=', r['prefill_time_ms'], 'ms, Decode=', r['decode_time_ms'], 'ms, Memory=', r['memory_used_gb'], 'GB')
    else:
        print(r['name'], ': FAILED -', r.get('error', 'Unknown error'))

print('')
print('Results saved to real_ablation_results.json')

