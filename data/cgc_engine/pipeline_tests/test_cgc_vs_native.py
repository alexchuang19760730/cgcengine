#!/usr/bin/env python3
"""Test CGC + OrthoKDA v4 vs Native llama.cpp"""

import time
import gc
from cgc_engine.cgc.ortho_kda_v4_llama import OrthoKDAV4LlamaCppIntegration

MODEL_PATH = '/Users/alexchuang/Documents/flashkv0430/models/qwen2.5-7b-q4_k_m.gguf'

def test_native():
    """Test native llama.cpp"""
    print("\n" + "="*60)
    print("TEST 1: Native llama.cpp")
    print("="*60)

    from llama_cpp import Llama

    print("Loading model...")
    t0 = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=8,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Prefill test
    prompt = "Hello, how are you? "
    tokens = llm.tokenize(prompt.encode())
    print(f"Prompt tokens: {len(tokens)}")

    t0 = time.time()
    _ = llm.eval(tokens)
    prefill_time = time.time() - t0
    print(f"Prefill time: {prefill_time:.3f}s")

    # Decode test
    t0 = time.time()
    result = llm.create_completion(prompt, max_tokens=20)
    decode_time = time.time() - t0
    print(f"Decode time: {decode_time:.3f}s")
    print(f"Generated: {result['usage']['completion_tokens']} tokens")

    return {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "tokens": result['usage']['completion_tokens']
    }


def test_cgc_kda():
    """Test CGC + OrthoKDA v4"""
    print("\n" + "="*60)
    print("TEST 2: CGC + OrthoKDA v4")
    print("="*60)

    gc.collect()

    print("Loading model with CGC + OrthoKDA v4...")
    t0 = time.time()
    integration = OrthoKDAV4LlamaCppIntegration(
        num_heads=4,
        head_dim=128,
        ortho_base_dim=32,
        decay_rate=0.01,
        enable=True,
        model_path=MODEL_PATH,
        device='cpu',
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")

    llm = integration.llm
    if llm is None:
        print("FAILED: Model not loaded")
        return None

    # Prefill test
    prompt = "Hello, how are you? "
    tokens = llm.tokenize(prompt.encode())
    print(f"Prompt tokens: {len(tokens)}")

    t0 = time.time()
    _ = llm.eval(tokens)
    prefill_time = time.time() - t0
    print(f"Prefill time: {prefill_time:.3f}s")

    # Decode test
    t0 = time.time()
    result = llm.create_completion(prompt, max_tokens=20)
    decode_time = time.time() - t0
    print(f"Decode time: {decode_time:.3f}s")
    print(f"Generated: {result['usage']['completion_tokens']} tokens")

    return {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "tokens": result['usage']['completion_tokens']
    }


def main():
    print("="*60)
    print("CGC Engine + OrthoKDA v4 vs Native llama.cpp Benchmark")
    print("="*60)

    # Test native first
    native_result = test_native()

    # Test CGC+KDA
    cgc_result = test_cgc_kda()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if native_result and cgc_result:
        prefill_speedup = native_result['prefill_time'] / cgc_result['prefill_time']
        decode_speedup = native_result['decode_time'] / cgc_result['decode_time']

        print(f"Prefill: Native={native_result['prefill_time']:.3f}s, CGC+KDA={cgc_result['prefill_time']:.3f}s, Speedup={prefill_speedup:.2f}x")
        print(f"Decode:  Native={native_result['decode_time']:.3f}s, CGC+KDA={cgc_result['decode_time']:.3f}s, Speedup={decode_speedup:.2f}x")

    # Memory comparison
    native_kv = 4 * 2048 * 128 * 2 * 4 / 1024 / 1024  # n_kv_heads * n_ctx * head_dim * 2 * 4bytes
    ortho_kv = 4 * 32 * 128 * 2 * 4 / 1024 / 1024  # n_kv_heads * ortho_base_dim * head_dim * 2 * 4bytes
    print(f"\nKV Cache Memory:")
    print(f"  Native: {native_kv:.1f} MB (grows with context)")
    print(f"  OrthoKDA: {ortho_kv:.2f} MB (fixed O(1))")
    print(f"  Savings: {(1 - ortho_kv/native_kv)*100:.1f}%")


if __name__ == "__main__":
    main()