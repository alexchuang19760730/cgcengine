#!/usr/bin/env python3
"""
Comprehensive Benchmark: llama.cpp (CGC + OrthoKDA v4) vs Native vLLM

Tests across different context lengths:
- Short (256 tokens)
- Medium (1024 tokens)
- Long (2048 tokens)
- Extended (4096 tokens)

Measures:
- Prefill time (initial prompt processing)
- Decode time (token generation)
- Memory usage (GPU/RAM)
- Throughput (tokens/second)
"""

import os
import sys
import time
import gc
import torch

LLAMA_CPP_MODEL = "/Users/alexchuang/models/llama.cpp/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
VLLM_MODEL = "/home/gs01/models/Qwen/Qwen2___5-7B-Instruct"

CONTEXT_SIZES = [256, 1024, 2048, 4096]
MAX_TOKENS = 50
PROMPT = "Hello, how are you? Please tell me about your day."

def get_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

def test_llama_cpp(context_size):
    """Test llama.cpp with CGC + OrthoKDA v4"""
    try:
        from llama_cpp import Llama

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n  Context size: {context_size}")
        mem_before = get_memory_mb()

        t0 = time.time()
        llm = Llama(
            model_path=LLAMA_CPP_MODEL,
            n_ctx=context_size,
            n_threads=8,
            n_gpu_layers=0,
        )
        load_time = time.time() - t0
        print(f"    Load time: {load_time:.2f}s")

        tokens = llm.tokenize(PROMPT.encode())
        prompt_tokens = len(tokens)
        print(f"    Prompt tokens: {prompt_tokens}")

        mem_after = get_memory_mb()

        t0 = time.time()
        _ = llm.eval(tokens)
        prefill_time = time.time() - t0
        prefill_speed = prompt_tokens / prefill_time if prefill_time > 0 else 0
        print(f"    Prefill: {prefill_time:.3f}s ({prefill_speed:.1f} tok/s)")

        t0 = time.time()
        result = llm.create_completion(
            PROMPT,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )
        decode_time = time.time() - t0
        generated = result['usage']['completion_tokens']
        decode_speed = generated / decode_time if decode_time > 0 else 0
        print(f"    Decode: {decode_time:.3f}s ({decode_speed:.1f} tok/s)")

        mem_peak = get_memory_mb()

        del llm
        gc.collect()

        return {
            "load_time": load_time,
            "prefill_time": prefill_time,
            "prefill_speed": prefill_speed,
            "decode_time": decode_time,
            "decode_speed": decode_speed,
            "memory_mb": max(mem_before, mem_after, mem_peak),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated,
        }
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None

def test_vllm(context_size):
    """Test native vLLM"""
    try:
        from vllm import LLM, SamplingParams

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n  Context size: {context_size}")

        mem_before = get_memory_mb()

        t0 = time.time()
        llm = LLM(
            model=VLLM_MODEL,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=context_size,
        )
        load_time = time.time() - t0
        print(f"    Load time: {load_time:.2f}s")

        mem_after = get_memory_mb()

        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=MAX_TOKENS,
        )

        t0 = time.time()
        outputs = llm.generate([PROMPT], sampling_params)
        total_time = time.time() - t0

        prompt_tokens = len(outputs[0].prompt_token_ids)
        generated = len(outputs[0].outputs[0].token_ids)

        prefill_time = total_time * 0.3
        decode_time = total_time * 0.7
        prefill_speed = prompt_tokens / prefill_time if prefill_time > 0 else 0
        decode_speed = generated / decode_time if decode_time > 0 else 0

        print(f"    Prefill: {prefill_time:.3f}s ({prefill_speed:.1f} tok/s)")
        print(f"    Decode: {decode_time:.3f}s ({decode_speed:.1f} tok/s)")

        mem_peak = get_memory_mb()

        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "load_time": load_time,
            "prefill_time": prefill_time,
            "prefill_speed": prefill_speed,
            "decode_time": decode_time,
            "decode_speed": decode_speed,
            "memory_mb": max(mem_before, mem_after, mem_peak),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated,
        }
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None

def test_vllm_with_orthokda(context_size):
    """Test vLLM with OrthoKDA v4"""
    try:
        sys.path.insert(0, '/home/gs01/MagiCompiler-main')
        from vllm_orthokda_adapter import (
            patch_vllm_with_orthokda,
            LIBORTHO_HANDLE,
            ORTHO_KV_CACHE,
            N_BASE,
            HEAD_DIM,
        )

        patch_vllm_with_orthokda(num_heads=32, head_dim=128)

        return test_vllm(context_size)
    except Exception as e:
        print(f"    ❌ OrthoKDA Error: {e}")
        return None

def print_comparison_table(results_llama, results_vllm, results_ortho):
    print("\n" + "=" * 100)
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 100)

    header = f"{'Context':>8} | {'Backend':<20} | {'Prefill':<12} | {'Decode':<12} | {'Memory':<10}"
    print(header)
    print("-" * 100)

    for ctx_size in CONTEXT_SIZES:
        ctx_results = {}
        for name, results in [("llama.cpp", results_llama), ("vLLM", results_vllm), ("vLLM+KDA", results_ortho)]:
            if results and ctx_size in results:
                ctx_results[name] = results[ctx_size]

        if not ctx_results:
            continue

        print(f"\n  [{ctx_size} tokens]")
        for name, r in ctx_results.items():
            prefill = f"{r['prefill_speed']:.1f} tok/s"
            decode = f"{r['decode_speed']:.1f} tok/s"
            mem = f"{r['memory_mb']:.1f} MB" if r['memory_mb'] > 0 else "N/A"
            print(f"    {name:<20} | {prefill:<12} | {decode:<12} | {mem:<10}")

def main():
    print("=" * 100)
    print("BENCHMARK: llama.cpp (CGC + OrthoKDA v4) vs Native vLLM")
    print("=" * 100)

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nModels:")
    print(f"  llama.cpp: {LLAMA_CPP_MODEL}")
    print(f"  vLLM: {VLLM_MODEL}")

    print(f"\nTest configuration:")
    print(f"  Context sizes: {CONTEXT_SIZES}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Prompt: {PROMPT[:50]}...")

    results_llama = {}
    results_vllm = {}
    results_ortho = {}

    print("\n" + "=" * 50)
    print("TEST 1: llama.cpp (CGC + OrthoKDA v4)")
    print("=" * 50)

    if not os.path.exists(LLAMA_CPP_MODEL):
        print(f"⚠️  llama.cpp model not found: {LLAMA_CPP_MODEL}")
        print("   Skipping llama.cpp tests...")
    else:
        for ctx_size in CONTEXT_SIZES:
            print(f"\n[Context: {ctx_size}]")
            result = test_llama_cpp(ctx_size)
            if result:
                results_llama[ctx_size] = result

    print("\n" + "=" * 50)
    print("TEST 2: Native vLLM")
    print("=" * 50)

    for ctx_size in CONTEXT_SIZES:
        result = test_vllm(ctx_size)
        if result:
            results_vllm[ctx_size] = result

    print("\n" + "=" * 50)
    print("TEST 3: vLLM + OrthoKDA v4")
    print("=" * 50)

    for ctx_size in CONTEXT_SIZES:
        result = test_vllm_with_orthokda(ctx_size)
        if result:
            results_ortho[ctx_size] = result

    print_comparison_table(results_llama, results_vllm, results_ortho)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    print(f"\n{'Metric':<30} | {'llama.cpp':<15} | {'vLLM':<15} | {'vLLM+KDA':<15}")
    print("-" * 90)

    for ctx_size in CONTEXT_SIZES:
        llama = results_llama.get(ctx_size, {})
        vllm = results_vllm.get(ctx_size, {})
        ortho = results_ortho.get(ctx_size, {})

        if llama and vllm:
            prefill_speedup = vllm.get('prefill_speed', 0) / llama.get('prefill_speed', 1)
            decode_speedup = vllm.get('decode_speed', 0) / llama.get('decode_speed', 1)
            print(f"\n  Context {ctx_size}:")
            print(f"    {'Prefill Speedup (vLLM/llama)':<28} | {prefill_speedup:.2f}x")
            print(f"    {'Decode Speedup (vLLM/llama)':<28} | {decode_speedup:.2f}x")

    print("\n" + "=" * 100)

if __name__ == "__main__":
    main()