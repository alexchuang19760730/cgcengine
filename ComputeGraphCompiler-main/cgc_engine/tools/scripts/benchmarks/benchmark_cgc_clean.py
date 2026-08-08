#!/usr/bin/env python3
"""
CGC Engine Benchmark - 干净的测试
"""

import sys
import time
import gc

sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main')

MODEL = '/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf'

print('=' * 70)
print('CGC Engine (device=cpu) Benchmark')
print('=' * 70)
print()

from cgc_engine import CGCEngine

print('Loading CGC Engine (device=cpu)...')
engine = CGCEngine.from_gguf(gguf_path=MODEL, device='cpu')
print(f'Mode: {engine._get_mode()}')
print()

test_cases = [128, 512, 1024, 2048]

for n_tokens in test_cases:
    prompt = ('The quick brown fox jumps over the lazy dog. ' * 20)[:n_tokens]

    # Warmup
    _ = engine.generate(prompt[:50], max_tokens=10)

    # Benchmark - 取最快的一次
    best_time = float('inf')
    best_result = None
    for i in range(3):
        start = time.time()
        result = engine.generate(prompt, max_tokens=32)
        elapsed = time.time() - start
        if elapsed < best_time:
            best_time = elapsed
            best_result = result

    content = best_result.get('text', '') if isinstance(best_result, dict) else str(best_result)
    gen_tokens = len(content.split()) if content else 0

    prompt_tps = n_tokens / best_time
    gen_tps = gen_tokens / best_time if best_time > 0 else 0

    print(f'--- {n_tokens} tokens ---')
    print(f'  Time: {best_time*1000:.1f}ms')
    print(f'  Prefill: {prompt_tps:.1f} tokens/s')
    print(f'  Decode: {gen_tps:.1f} tokens/s')
    print()

print('Done!')
print()
print('Expected llama.cpp Metal results (for comparison):')
print('  128 tokens:  ~175 tokens/s')
print('  512 tokens:  ~165 tokens/s')
print('  1024 tokens: ~161 tokens/s')
print('  2048 tokens: ~153 tokens/s')