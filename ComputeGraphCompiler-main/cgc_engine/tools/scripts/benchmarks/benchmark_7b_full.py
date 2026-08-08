#!/usr/bin/env python3
"""
7B Model Full Benchmark: llama.cpp vs CGC Engine (GGUF)
使用 Qwen2.5-7B 模型進行效能比較

使用新的 CGCEngine 統一入口
"""

import sys
import time
import gc
from pathlib import Path

MODEL_GGUF = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf"
MODEL_HF = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS = 32
WARMUP_TOKENS = 10

def benchmark_llama_cpp():
    """llama.cpp Benchmark (直接使用 llama.cpp)"""
    print("=" * 80)
    print("【1】Llama.cpp Benchmark (Full 7B Model - GGUF)")
    print("=" * 80)

    try:
        from llama_cpp import Llama
        import torch

        mps_available = torch.backends.mps.is_available()
        n_gpu_layers = 32 if mps_available else 0

        print(f"Loading llama.cpp... (n_gpu_layers={n_gpu_layers})")
        llm = Llama(
            model_path=MODEL_GGUF,
            n_ctx=8192,
            n_gpu_layers=n_gpu_layers,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        print(f"Model loaded! (backend: {'Metal' if mps_available else 'CPU'})")

        test_cases = [
            ("Short (128 tokens)", 128),
            ("Medium (512 tokens)", 512),
            ("Long (1024 tokens)", 1024),
            ("Very Long (2048 tokens)", 2048),
        ]

        results = {}

        for name, target_tokens in test_cases:
            prompt = ("The quick brown fox jumps over the lazy dog. " * 10)[:target_tokens]
            tokens = llm.tokenize(prompt.encode())
            actual_tokens = len(tokens)

            print(f"\n--- {name} ---")
            print(f"Prompt: {actual_tokens} tokens")

            _ = llm(prompt[:50], max_tokens=WARMUP_TOKENS, stop=["</s>"])

            if mps_available:
                torch.mps.empty_cache()
            gc.collect()

            start = time.time()
            result = llm(prompt, max_tokens=MAX_TOKENS, stop=["</s>"], echo=False)
            elapsed = time.time() - start

            prompt_count = result.get('usage', {}).get('prompt_eval_count', actual_tokens)
            gen_count = result.get('usage', {}).get('eval_count', MAX_TOKENS)

            prompt_tps = prompt_count / elapsed
            gen_tps = gen_count / elapsed

            print(f"  Time: {elapsed*1000:.1f}ms")
            print(f"  Prompt: {prompt_tps:.1f} tokens/sec")
            print(f"  Generation: {gen_tps:.1f} tokens/sec")

            results[name] = {
                "prompt_tokens": actual_tokens,
                "prompt_tps": prompt_tps,
                "gen_tps": gen_tps,
                "total_ms": elapsed * 1000,
            }

        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_cgc_engine():
    """CGC Engine Benchmark (使用新的 CGCEngine 統一入口)"""
    print("\n" + "=" * 80)
    print("【2】CGC Engine Benchmark (CGCEngine + GGUF)")
    print("=" * 80)

    try:
        from cgc_engine import CGCEngine

        print(f"Creating CGCEngine with GGUF: {MODEL_GGUF}")
        engine = CGCEngine(gguf_path=MODEL_GGUF)
        print(f"CGC Engine created! device={engine._device}")
        print(f"llama_cpp mode: {engine._runtime.is_llama_cpp_mode()}")

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

            gc.collect()

            start = time.time()
            result = engine.generate(prompt, max_tokens=MAX_TOKENS)
            elapsed = time.time() - start

            generated_text = ""
            if isinstance(result, dict):
                generated_text = result.get("text", result.get("generated_text", ""))
            elif isinstance(result, str):
                generated_text = result

            gen_tokens = len(generated_text) if generated_text else MAX_TOKENS

            print(f"  Time: {elapsed*1000:.1f}ms")
            print(f"  Generated: {generated_text[:50]}...")

            results[name] = {
                "prompt_tokens": len(prompt),
                "gen_tps": gen_tokens / elapsed if elapsed > 0 else 0,
                "total_ms": elapsed * 1000,
            }

        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_mlx():
    """MLX Benchmark - Apple Silicon GPU"""
    print("\n" + "=" * 80)
    print("【3】MLX Benchmark (Apple Silicon GPU)")
    print("=" * 80)

    try:
        import mlx.core as mx
        import mlx_lm

        print(f"MLX device: {mx.default_device()}")
        print(f"Attempting to load: {MODEL_HF}")

        model, tokenizer = mlx_lm.load(MODEL_HF)
        print(f"Model loaded!")

        test_cases = [
            ("Short (128 tokens)", 128),
            ("Medium (512 tokens)", 512),
            ("Long (1024 tokens)", 1024),
        ]

        results = {}

        for name, target_tokens in test_cases:
            prompt = ("The quick brown fox jumps over the lazy dog. " * 10)[:target_tokens]

            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            n_prompt_tokens = input_ids.shape[1]

            print(f"\n--- {name} ---")
            print(f"Prompt: {n_prompt_tokens} tokens")

            _ = mlx_lm.generate(model, tokenizer, prompt[:50], max_tokens=WARMUP_TOKENS, verbose=False)

            mx.reset_peak_memory()

            start = time.time()
            result = mlx_lm.generate(
                model, tokenizer, prompt,
                max_tokens=MAX_TOKENS,
                verbose=False,
            )
            elapsed = time.time() - start

            generated_text = result if isinstance(result, str) else ""
            n_gen_tokens = len(tokenizer(generated_text, return_tensors="pt")["input_ids"]) if generated_text else MAX_TOKENS

            prompt_tps = n_prompt_tokens / elapsed
            gen_tps = n_gen_tokens / elapsed

            print(f"  Time: {elapsed*1000:.1f}ms")
            print(f"  Prompt: {prompt_tps:.1f} tokens/sec")
            print(f"  Generation: {gen_tps:.1f} tokens/sec")

            mem_info = mx.get_peak_memory()
            print(f"  Peak Memory: {mem_info / 1024 / 1024:.0f} MB")

            results[name] = {
                "prompt_tokens": n_prompt_tokens,
                "prompt_tps": prompt_tps,
                "gen_tps": gen_tps,
                "total_ms": elapsed * 1000,
            }

        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_cgc_engine_vllm():
    """CGC Engine vLLM Benchmark (CUDA GPU)"""
    print("\n" + "=" * 80)
    print("【4】CGC Engine vLLM Benchmark (CUDA GPU)")
    print("=" * 80)

    try:
        import torch
        if not torch.cuda.is_available():
            print("  CUDA not available, skipping vLLM benchmark")
            return None

        from cgc_engine import CGCEngine

        model_name = "Qwen/Qwen2.5-7B-Instruct"

        print(f"Creating CGCEngine with vLLM: {model_name}")
        engine = CGCEngine.from_vllm(
            model_name_or_path=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
        )
        print(f"CGC Engine vLLM created! mode={engine._get_mode()}")

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

            print(f"  Time: {elapsed*1000:.1f}ms")
            print(f"  Generated: {generated_text[:50]}...")

            results[name] = {
                "prompt_tokens": len(prompt),
                "gen_tps": gen_tokens / elapsed if elapsed > 0 else 0,
                "total_ms": elapsed * 1000,
            }

        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("7B Model Full Benchmark: llama.cpp vs CGC Engine (vLLM/GGUF) vs MLX")
    print("=" * 80)
    print()
    print(f"Model: Qwen2.5-7B")
    print(f"GGUF: {MODEL_GGUF}")
    print()

    llama_results = benchmark_llama_cpp()
    cgc_results = benchmark_cgc_engine()
    vllm_results = benchmark_cgc_engine_vllm()
    mlx_results = benchmark_mlx()

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()

    if llama_results:
        print("llama.cpp (GGUF - Metal):")
        print("-" * 60)
        print(f"{'Context':<25} {'Prompt TPS':<15} {'Gen TPS':<15}")
        for name, r in llama_results.items():
            print(f"{name:<25} {r['prompt_tps']:<15.1f} {r['gen_tps']:<15.1f}")

    if cgc_results:
        print("\nCGC Engine (GGUF - llama.cpp bridge):")
        print("-" * 60)
        print(f"{'Context':<25} {'Time (ms)':<15} {'Gen TPS':<15}")
        for name, r in cgc_results.items():
            print(f"{name:<25} {r['total_ms']:<15.1f} {r['gen_tps']:<15.1f}")

    if vllm_results:
        print("\nCGC Engine (vLLM - CUDA GPU):")
        print("-" * 60)
        print(f"{'Context':<25} {'Time (ms)':<15} {'Gen TPS':<15}")
        for name, r in vllm_results.items():
            print(f"{name:<25} {r['total_ms']:<15.1f} {r['gen_tps']:<15.1f}")
    else:
        print("\nvLLM: Skipped (CUDA not available or install failed)")

    if mlx_results:
        print("\nMLX (Apple Silicon GPU):")
        print("-" * 60)
        print(f"{'Context':<25} {'Prompt TPS':<15} {'Gen TPS':<15}")
        for name, r in mlx_results.items():
            print(f"{name:<25} {r['prompt_tps']:<15.1f} {r['gen_tps']:<15.1f}")
    else:
        print("\nMLX: Failed to load model (network timeout or missing files)")

    print()
    print("=" * 80)
    print("NOTE")
    print("=" * 80)
    print()
    print("CGC Engine 支援三種後端模式：")
    print("  1. llama.cpp (GGUF): 本地 CPU/Metal 推理，無需 GPU")
    print("  2. vLLM (CUDA): 高效能 GPU 推理，需要 CUDA 環境")
    print("  3. PyTorch: 原生 PyTorch 模型執行")
    print()
    print("MLX 使用不同的模型格式，需要從 HuggingFace 下載")


if __name__ == "__main__":
    main()