#!/usr/bin/env python3
"""
Llama.cpp 專用 Benchmark - 測量 prefill 和 decode
"""

import sys
import time
from collections import deque
from pathlib import Path

MODEL_PATH = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf"
CONTEXT_LENGTHS = [512, 1024, 2048, 4096]

def main():
    from llama_cpp import Llama
    import torch

    print("=" * 80)
    print("Llama.cpp Native Benchmark (eval_tokens)")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print()

    mps_available = torch.backends.mps.is_available()
    n_gpu_layers = 32 if mps_available else 0

    print(f"Loading llama.cpp... (n_gpu_layers={n_gpu_layers})")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=8192,
        n_gpu_layers=n_gpu_layers,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )
    print("Model loaded!")
    print()

    # Create prompts of exact token lengths
    base_text = "The quick brown fox jumps over the lazy dog. "
    base_tokens = llm.tokenize(base_text.encode())
    base_count = len(base_tokens)

    print(f"Base prompt: {base_count} tokens")
    print()

    results = {}

    for target_len in CONTEXT_LENGTHS:
        # Create prompt with exact token count
        repeats = (target_len // base_count) + 1
        prompt = base_text * repeats
        tokens = llm.tokenize(prompt.encode())[:target_len]
        n_tokens = len(tokens)

        print(f"[Context: {target_len}] ({n_tokens} tokens)")
        print("-" * 50)

        # Warmup
        _ = llm.eval(tokens[:min(32, n_tokens)])

        # Clear cache
        if mps_available:
            torch.mps.empty_cache()

        # Test 1: Prefill (eval all tokens)
        torch.mps.empty_cache() if mps_available else None
        start = time.time()
        n_eval = llm.eval(tokens)
        elapsed = time.time() - start

        if isinstance(n_eval, deque):
            n_eval = len(n_eval)
        tps = n_tokens / elapsed if elapsed > 0 else 0

        print(f"  Prefill: {elapsed*1000:.1f}ms ({n_tokens} tokens, {tps:.0f} tokens/sec)")

        # Memory
        if mps_available:
            mem_mb = torch.mps.current_allocated_memory() / 1024 / 1024
            print(f"  Memory:  {mem_mb:.0f} MB")

        print()
        results[target_len] = {
            "n_tokens": n_tokens,
            "prefill_ms": elapsed * 1000,
            "prefill_tps": tps,
            "memory_mb": mem_mb if mps_available else 0,
        }

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Context':<10} {'Tokens':<10} {'Prefill ms':<15} {'Tokens/sec':<15} {'Memory':<15}")
    print("-" * 65)
    for ctx_len, r in results.items():
        mem_str = f"{r['memory_mb']:.0f} MB" if r['memory_mb'] > 0 else "N/A"
        print(f"[{ctx_len:<8}] {r['n_tokens']:<10} {r['prefill_ms']:<15.1f} {r['prefill_tps']:<15.0f} {mem_str:<15}")

    print()
    print("Note: llama.cpp evaluates full 7B model (32 layers, ~7B params)")
    print("      CGC benchmark only tests attention kernel, not fair comparison")

if __name__ == "__main__":
    main()