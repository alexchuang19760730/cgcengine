#!/usr/bin/env python3
"""
CGC Engine KDA vs llama.cpp Benchmark (macOS 本地测试)

对比 KDA 开启/关闭的 prefill/decode/memory 表现
"""

import sys
import time
import gc
import psutil
from pathlib import Path

sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main')

MODEL_GGUF = "/Users/alexchuang/Documents/cgc/models/qwen2.5-7b-q4_k_m.gguf"
MAX_TOKENS = 32
WARMUP_TOKENS = 10

def get_memory_usage():
    """获取当前进程内存使用 (MB)"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def benchmark_llama_cpp():
    """llama.cpp Benchmark (原生)"""
    print("=" * 80)
    print("【1】Llama.cpp Benchmark (原生 GGUF - Metal/CPU)")
    print("=" * 80)

    try:
        from llama_cpp import Llama
        import torch

        mps_available = torch.backends.mps.is_available()
        n_gpu_layers = 32 if mps_available else 0

        print(f"Loading llama.cpp... (n_gpu_layers={n_gpu_layers}, MPS={mps_available})")
        mem_before = get_memory_usage()

        llm = Llama(
            model_path=MODEL_GGUF,
            n_ctx=8192,
            n_gpu_layers=n_gpu_layers,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )

        mem_after = get_memory_usage()
        print(f"Model loaded! Memory: {mem_after:.0f} MB (delta: {mem_after - mem_before:.0f} MB)")
        print(f"Backend: {'Metal (MPS)' if mps_available else 'CPU'}")

        test_cases = [
            ("128 tokens", 128),
            ("512 tokens", 512),
            ("1024 tokens", 1024),
            ("2048 tokens", 2048),
        ]

        results = {}

        for name, n_tokens in test_cases:
            prompt = ("The quick brown fox jumps over the lazy dog. " * 20)[:n_tokens]
            mem_start = get_memory_usage()

            print(f"\n--- {name} ---")

            _ = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt[:50]}],
                max_tokens=WARMUP_TOKENS,
            )

            start = time.time()
            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
            )
            elapsed = time.time() - start

            content = response["choices"][0]["message"]["content"]
            gen_tokens = len(content.split())

            mem_end = get_memory_usage()

            prompt_tps = n_tokens / elapsed
            gen_tps = gen_tokens / elapsed if elapsed > 0 else 0

            print(f"  Prefill: {elapsed*1000:.1f}ms ({prompt_tps:.1f} tokens/s)")
            print(f"  Decode: {gen_tokens} tokens in {elapsed*1000:.1f}ms ({gen_tps:.1f} tokens/s)")
            print(f"  Memory: {mem_end:.0f} MB")

            results[name] = {
                "prefill_ms": elapsed * 1000,
                "prompt_tps": prompt_tps,
                "gen_tps": gen_tps,
                "memory_mb": mem_end,
            }

        del llm
        gc.collect()

        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_cgc_engine_kda():
    """CGC Engine Benchmark (KDA 开启)"""
    print("\n" + "=" * 80)
    print("【2】CGC Engine Benchmark (KDA 开启)")
    print("=" * 80)

    try:
        from cgc_engine import CGCEngine
        import torch

        print("Loading CGC Engine with KDA enabled...")

        engine = CGCEngine.from_gguf(
            gguf_path=MODEL_GGUF,
            device="metal" if torch.backends.mps.is_available() else "cpu",
        )

        print(f"CGC Engine loaded! Mode: {engine._get_mode()}")

        test_cases = [
            ("128 tokens", 128),
            ("512 tokens", 512),
            ("1024 tokens", 1024),
            ("2048 tokens", 2048),
        ]

        results = {}

        for name, n_tokens in test_cases:
            prompt = ("The quick brown fox jumps over the lazy dog. " * 20)[:n_tokens]
            mem_start = get_memory_usage()

            print(f"\n--- {name} ---")

            _ = engine.generate(prompt[:50], max_tokens=WARMUP_TOKENS)

            start = time.time()
            result = engine.generate(prompt, max_tokens=MAX_TOKENS)
            elapsed = time.time() - start

            content = result.get("text", result.get("generated_text", "")) if isinstance(result, dict) else str(result)
            gen_tokens = len(content.split())

            mem_end = get_memory_usage()

            prompt_tps = n_tokens / elapsed
            gen_tps = gen_tokens / elapsed if elapsed > 0 else 0

            print(f"  Prefill: {elapsed*1000:.1f}ms ({prompt_tps:.1f} tokens/s)")
            print(f"  Decode: {gen_tokens} tokens in {elapsed*1000:.1f}ms ({gen_tps:.1f} tokens/s)")
            print(f"  Memory: {mem_end:.0f} MB")

            results[name] = {
                "prefill_ms": elapsed * 1000,
                "prompt_tps": prompt_tps,
                "gen_tps": gen_tps,
                "memory_mb": mem_end,
            }

        del engine
        gc.collect()

        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 80)
    print("CGC Engine KDA Benchmark: llama.cpp vs CGC (KDA)")
    print("=" * 80)
    print()
    print(f"Model: Qwen2.5-7B GGUF")
    print(f"GGUF: {MODEL_GGUF}")
    print()

    llama_results = benchmark_llama_cpp()
    cgc_results = benchmark_cgc_engine_kda()

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()

    if llama_results:
        print("llama.cpp (原生):")
        print("-" * 80)
        print(f"{'Context':<15} {'Prefill (ms)':<15} {'Prompt TPS':<15} {'Decode TPS':<15} {'Memory (MB)':<15}")
        print("-" * 80)
        for name, r in llama_results.items():
            print(f"{name:<15} {r['prefill_ms']:<15.1f} {r['prompt_tps']:<15.1f} {r['gen_tps']:<15.1f} {r['memory_mb']:<15.0f}")

    if cgc_results:
        print("\nCGC Engine (KDA 开启):")
        print("-" * 80)
        print(f"{'Context':<15} {'Prefill (ms)':<15} {'Prompt TPS':<15} {'Decode TPS':<15} {'Memory (MB)':<15}")
        print("-" * 80)
        for name, r in cgc_results.items():
            print(f"{name:<15} {r['prefill_ms']:<15.1f} {r['prompt_tps']:<15.1f} {r['gen_tps']:<15.1f} {r['memory_mb']:<15.0f}")

    if llama_results and cgc_results:
        print("\n性能对比:")
        print("-" * 80)
        print(f"{'Context':<15} {'Prefill 提升':<20} {'Decode 提升':<20}")
        print("-" * 80)
        for name in llama_results.keys():
            r_llama = llama_results[name]
            r_cgc = cgc_results.get(name, {})
            if r_cgc:
                prefill_speedup = r_llama['prefill_ms'] / r_cgc['prefill_ms'] if r_cgc['prefill_ms'] > 0 else 0
                decode_speedup = r_llama['gen_tps'] / r_cgc['gen_tps'] if r_cgc['gen_tps'] > 0 else 0
                print(f"{name:<15} {prefill_speedup:.2f}x faster          {decode_speedup:.2f}x faster")

if __name__ == "__main__":
    main()