#!/usr/bin/env python3
"""
测试 llama-cpp-python 的 GPU/Metal 使用情况
"""

import sys
import gc

MODEL = '/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf'

print('Testing llama-cpp-python GPU/Metal usage...')
print()

# Test 1: n_gpu_layers=0 (CPU only)
print('Test 1: n_gpu_layers=0 (CPU only)')
from llama_cpp import Llama

model_cpu = Llama(
    model_path=MODEL,
    n_ctx=8192,
    n_gpu_layers=0,  # CPU only
    use_mmap=True,
    use_mlock=False,
    verbose=True,
)

print('Running inference on CPU...')
result = model_cpu('The quick brown fox', max_tokens=10)
print(f'Result: {result}')
del model_cpu
gc.collect()
print()

# Test 2: n_gpu_layers=32 (Metal)
print('Test 2: n_gpu_layers=32 (Metal)')
model_metal = Llama(
    model_path=MODEL,
    n_ctx=8192,
    n_gpu_layers=32,  # Metal
    use_mmap=True,
    use_mlock=False,
    verbose=True,
)

print('Running inference on Metal...')
result = model_metal('The quick brown fox', max_tokens=10)
print(f'Result: {result}')
del model_metal
gc.collect()

print()
print('Done!')
