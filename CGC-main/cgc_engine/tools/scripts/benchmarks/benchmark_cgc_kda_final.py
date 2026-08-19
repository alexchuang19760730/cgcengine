#!/usr/bin/env python3
"""
CGC Engine + KDA Benchmark - 验证 C++ SIMD Engine 和 KDA 加载
"""

import sys
import time
import gc
import psutil

sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main')

def get_memory():
    return psutil.Process().memory_info().rss / 1024 / 1024

print('=' * 70)
print('CGC Engine + KDA Benchmark (C++ SIMD Engine)')
print('=' * 70)
print()

from cgc_engine import CGCEngine

MODEL = '/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf'

print('Loading CGC Engine (KDA enabled)...')
mem_before = get_memory()

engine = CGCEngine.from_gguf(gguf_path=MODEL, device='cpu')

mem_after = get_memory()
print(f'Model loaded! Memory: {mem_after:.0f} MB (delta: {mem_after - mem_before:.0f} MB)')
print(f'Mode: {engine._get_mode()}')
print()

test_cases = [128, 512, 1024, 2048]

for n_tokens in test_cases:
    prompt = ('The quick brown fox jumps over the lazy dog. ' * 20)[:n_tokens]

    # Warmup
    _ = engine.generate(prompt[:50], max_tokens=10)

    mem_start = get_memory()

    # Benchmark
    start = time.time()
    result = engine.generate(prompt, max_tokens=32)
    elapsed = time.time() - start

    mem_end = get_memory()

    content = result.get('text', '') if isinstance(result, dict) else str(result)
    gen_tokens = len(content.split()) if content else 0

    prompt_tps = n_tokens / elapsed
    gen_tps = gen_tokens / elapsed if elapsed > 0 else 0

    print(f'--- {n_tokens} tokens ---')
    print(f'  Prefill: {elapsed*1000:.1f}ms ({prompt_tps:.1f} tokens/s)')
    print(f'  Decode: {gen_tokens} tokens in {elapsed*1000:.1f}ms ({gen_tps:.1f} tokens/s)')
    print(f'  Memory: {mem_end:.0f} MB')
    print()

del engine
gc.collect()
print('Done!')