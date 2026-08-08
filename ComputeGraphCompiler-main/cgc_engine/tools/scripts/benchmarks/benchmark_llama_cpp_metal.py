#!/usr/bin/env python3
"""
CGC Engine + Metal Benchmark (直接使用 llama-cpp-python)
"""

import sys
import time
import gc
import psutil

def get_memory():
    return psutil.Process().memory_info().rss / 1024 / 1024

MODEL = '/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf'

print('=' * 70)
print('llama-cpp-python Metal Benchmark')
print('=' * 70)
print()

print('Loading model with Metal (n_gpu_layers=32)...')
mem_before = get_memory()

from llama_cpp import Llama
model = Llama(
    model_path=MODEL,
    n_ctx=8192,
    n_gpu_layers=32,
    use_mmap=True,
    use_mlock=False,
    verbose=False,
)

mem_after = get_memory()
print(f'Model loaded! Memory: {mem_after:.0f} MB (delta: {mem_after - mem_before:.0f} MB)')
print()

test_cases = [128, 512, 1024, 2048]

print('Running benchmark (warmup=1, iterations=3)...')
print()

for n_tokens in test_cases:
    prompt = ('The quick brown fox jumps over the lazy dog. ' * 20)[:n_tokens]

    # Warmup
    _ = model(prompt[:50], max_tokens=10)

    mem_start = get_memory()

    # Benchmark - 多次运行取最快
    elapsed_list = []
    for i in range(3):
        start = time.time()
        result = model(prompt, max_tokens=32)
        elapsed = time.time() - start
        elapsed_list.append(elapsed)

    elapsed = min(elapsed_list)
    mem_end = get_memory()

    gen_tokens = result['usage']['completion_tokens']
    prompt_tps = n_tokens / elapsed
    gen_tps = gen_tokens / elapsed if elapsed > 0 else 0

    print(f'--- {n_tokens} tokens ---')
    print(f'  Prefill: {elapsed*1000:.1f}ms ({prompt_tps:.1f} tokens/s)')
    print(f'  Decode: {gen_tokens} tokens in {elapsed*1000:.1f}ms ({gen_tps:.1f} tokens/s)')
    print(f'  Memory: {mem_end:.0f} MB')
    print()

del model
gc.collect()
print('Done!')