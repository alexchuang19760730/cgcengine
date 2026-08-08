#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/gs01')

import time
import json
from cgc_engine import CGCEngine, CGCEngineConfig

MODEL_PATH = '/home/gs01/models/Qwen/Qwen2___5-7B-Instruct'

WARMUP_TOKENS = 20
MAX_TOKENS = 100

def benchmark_vllm():
    print("\n" + "=" * 80)
    print("CGC Engine vLLM Benchmark (CUDA GPU)")
    print("=" * 80)

    config = CGCEngineConfig(
        model_name_or_path=MODEL_PATH,
        enable_vllm=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.70,
    )

    print(f"Creating CGCEngine with vLLM...")
    engine = CGCEngine(config=config)
    print(f"Engine created! Mode: {engine._get_mode()}")

    test_cases = [
        ("Short (128 tokens)", 128),
        ("Medium (512 tokens)", 512),
        ("Long (1024 tokens)", 1024),
    ]

    results = {}

    for name, target_tokens in test_cases:
        prompt = ("The quick brown fox jumps over the lazy dog. " * 10)[:target_tokens]

        print(f"\n--- {name} ---")
        print(f"Prompt: {len(prompt)} chars")

        _ = engine.generate(prompt[:50], max_tokens=WARMUP_TOKENS)

        start = time.time()
        result = engine.generate(prompt, max_tokens=MAX_TOKENS)
        elapsed = time.time() - start

        generated_text = ""
        if isinstance(result, dict):
            generated_text = result.get("text", result.get("generated_text", ""))
        elif isinstance(result, str):
            generated_text = result

        gen_tokens = len(generated_text) if generated_text else MAX_TOKENS
        tps = gen_tokens / elapsed if elapsed > 0 else 0

        print(f"  Time: {elapsed*1000:.1f}ms")
        print(f"  Tokens: {gen_tokens}")
        print(f"  TPS: {tps:.1f}")
        print(f"  Generated: {generated_text[:80]}...")

        results[name] = {
            "prompt_tokens": len(prompt),
            "gen_tokens": gen_tokens,
            "total_ms": elapsed * 1000,
            "tps": tps,
        }

    return results

def main():
    print("=" * 80)
    print("CGC Engine Benchmark on NVIDIA GeForce RTX 5090")
    print("=" * 80)

    results = benchmark_vllm()

    print("\n" + "=" * 80)
    print("Benchmark Results Summary")
    print("=" * 80)
    print(f"{'Test Case':<20} {'Prompt Tokens':<15} {'Gen Tokens':<12} {'Time (ms)':<12} {'TPS':<10}")
    print("-" * 80)
    for name, data in results.items():
        print(f"{name:<20} {data['prompt_tokens']:<15} {data['gen_tokens']:<12} {data['total_ms']:<12.1f} {data['tps']:<10.1f}")

    with open('/home/gs01/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to /home/gs01/benchmark_results.json")

if __name__ == '__main__':
    main()
