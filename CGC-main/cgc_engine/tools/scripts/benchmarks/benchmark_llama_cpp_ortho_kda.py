#!/usr/bin/env python3
"""
OrthoKDA v4 Benchmark - CGC Engine + KDA vs Native llama.cpp

Compares:
1. Prefill performance: Context processing time
2. Decode performance: Token generation speed
3. Memory performance: KV cache memory usage

Usage:
    python3 benchmark_llama_cpp_ortho_kda.py --model models/qwen2.5-7b-q4_k_m.gguf
"""

import argparse
import time
import sys
import os
import psutil
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║        OrthoKDA v4 Benchmark - CGC Engine + KDA vs Native llama.cpp         ║
║                                                                              ║
║  测试:                                                                       ║
║    1. Prefill性能 - 不同上下文长度的处理时间                                   ║
║    2. Decode性能 - token生成速度                                              ║
║    3. Memory性能 - KV cache内存占用                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def check_llama_cpp():
    """Check if llama.cpp is available"""
    try:
        from llama_cpp import Llama
        print("  ✅ llama.cpp is available")
        return True
    except ImportError:
        print("  ❌ llama.cpp not available")
        return False


def benchmark_native_llama_cpp(model_path, config):
    """Benchmark native llama.cpp without KDA"""
    from llama_cpp import Llama

    print("\n  [Native llama.cpp] 初始化模型...")
    mem_before = get_memory_usage_mb()

    llm = Llama(
        model_path=model_path,
        n_ctx=config.get("n_ctx", 2048),
        n_threads=os.cpu_count(),
        use_mmap=True,
        use_mlock=False,
    )

    mem_after = get_memory_usage_mb()
    model_mem = mem_after - mem_before

    print(f"  ✅ 模型加载成功!")
    print(f"     💾 额外内存: {model_mem:.1f} MB")

    return llm, model_mem


def benchmark_cgc_ortho_kda(model_path, config):
    """Benchmark llama.cpp with CGC OrthoKDA v4"""
    from cgc_engine.cgc.ortho_kda_v4_llama import (
        OrthoKDAV4LlamaCppIntegration,
        OrthoKDAV4LlamaConfig,
        OrthoKDAKVState,
    )

    print("\n  [CGC + OrthoKDA v4] 初始化...")

    n_heads = config.get("n_heads", 32)
    head_dim = config.get("head_dim", 128)
    ortho_base_dim = config.get("ortho_base_dim", 128)

    integration = OrthoKDAV4LlamaCppIntegration(
        num_heads=n_heads,
        head_dim=head_dim,
        ortho_base_dim=ortho_base_dim,
        decay_rate=0.01,
        enable=True,
        model_path=model_path,
        device="cpu",
    )

    print(f"  ✅ OrthoKDA v4 初始化成功!")
    print(f"     💾 OrthoKDA KV Memory: {ortho_base_dim * head_dim * 2 * 4 / 1024:.2f} KB (fixed)")

    return integration, integration.backend


def run_prefill_benchmark(llm, prompt, description):
    """Benchmark prefill (context processing)"""
    if llm is None:
        return {"n_tokens": 0, "time_sec": 0, "tokens_per_sec": 0}

    print(f"\n  📝 Prefill测试: {description}")

    try:
        tokens = llm.tokenize(prompt.encode())
        n_tokens = len(tokens)

        start = time.perf_counter()
        _ = llm.eval(tokens)
        end = time.perf_counter()

        elapsed = end - start
        tokens_per_sec = n_tokens / elapsed if elapsed > 0 else 0

        print(f"     Tokens: {n_tokens}, Time: {elapsed:.3f}s, Speed: {tokens_per_sec:.1f} tok/s")

        return {
            "n_tokens": n_tokens,
            "time_sec": elapsed,
            "tokens_per_sec": tokens_per_sec,
        }
    except Exception as e:
        print(f"     ❌ Prefill failed: {e}")
        return {"n_tokens": 0, "time_sec": 0, "tokens_per_sec": 0}


def run_decode_benchmark(llm, prompt, max_tokens, description):
    """Benchmark decode (token generation)"""
    if llm is None:
        return {"generated_tokens": 0, "time_sec": 0, "tokens_per_sec": 0}

    print(f"\n  🎯 Decode测试: {description}")

    try:
        start = time.perf_counter()
        result = llm.create_completion(
            prompt,
            max_tokens=max_tokens,
            stop=["</s>", "<|endoftext|>"],
        )
        end = time.perf_counter()

        elapsed = end - start
        generated = result["usage"]["completion_tokens"]
        tokens_per_sec = generated / elapsed if elapsed > 0 else 0

        print(f"     Generated: {generated} tokens, Time: {elapsed:.3f}s, Speed: {tokens_per_sec:.1f} tok/s")

        return {
            "generated_tokens": generated,
            "time_sec": elapsed,
            "tokens_per_sec": tokens_per_sec,
        }
    except Exception as e:
        print(f"     ❌ Decode failed: {e}")
        return {"generated_tokens": 0, "time_sec": 0, "tokens_per_sec": 0}


def get_model_config(model_path):
    """Detect model configuration from file or use defaults"""
    base_name = os.path.basename(model_path).lower()

    if "qwen2.5-7b" in base_name:
        return {
            "n_ctx": 8192,
            "n_heads": 32,
            "head_dim": 128,
            "n_kv_heads": 32,
            "ortho_base_dim": 128,
        }
    elif "qwen2.5-3b" in base_name:
        return {
            "n_ctx": 8192,
            "n_heads": 32,
            "head_dim": 128,
            "n_kv_heads": 32,
            "ortho_base_dim": 64,
        }
    elif "phi-3" in base_name:
        return {
            "n_ctx": 4096,
            "n_heads": 32,
            "head_dim": 128,
            "n_kv_heads": 32,
            "ortho_base_dim": 64,
        }
    else:
        return {
            "n_ctx": 2048,
            "n_heads": 32,
            "head_dim": 128,
            "n_kv_heads": 32,
            "ortho_base_dim": 128,
        }


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="OrthoKDA v4 Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--prefill-lens", type=int, nargs="+", default=[128, 512, 1024, 2048],
                        help="Context lengths for prefill test")
    parser.add_argument("--compare", action="store_true", default=True, help="Compare with native")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return 1

    model_config = get_model_config(args.model)
    print(f"\n📊 Model Configuration:")
    for k, v in model_config.items():
        print(f"   {k}: {v}")

    llama_available = check_llama_cpp()
    if not llama_available:
        print("\n❌ llama.cpp is required for this benchmark")
        return 1

    prompts = {
        "short": "Hello, how are you?",
        "medium": "Write a short story about a robot learning to paint. Include a beginning, middle, and end.",
        "long": "Explain the history of artificial intelligence, including the key milestones from the 1950s to today, the main approaches and techniques, and the future directions of the field.",
    }

    prefill_results_native = {}
    prefill_results_cgc = {}
    decode_result_native = {"tokens_per_sec": 0}
    decode_result_cgc = {"tokens_per_sec": 0}
    native_kv_mem = 0
    cgc_kv_mem = 0

    print("\n" + "=" * 70)
    print("📌 BENCHMARK 1: Native llama.cpp")
    print("=" * 70)

    try:
        native_llm, native_mem = benchmark_native_llama_cpp(args.model, model_config)
        native_kv_mem = model_config["n_kv_heads"] * model_config["n_ctx"] * model_config["head_dim"] * 2 * 4 / 1024 / 1024
        print(f"     💾 KV Cache理论内存: {native_kv_mem:.1f} MB")

        print("\n  [Prefill Tests]")
        for length in args.prefill_lens:
            prompt = prompts["medium"] * (length // len(prompts["medium"]) + 1)
            prompt = prompt[:length]
            result = run_prefill_benchmark(native_llm, prompt, f"context={length}")
            prefill_results_native[length] = result

        print("\n  [Decode Tests]")
        decode_result_native = run_decode_benchmark(
            native_llm, prompts["short"], args.max_tokens, "50 tokens"
        )

    except Exception as e:
        print(f"\n  ❌ Native llama.cpp benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    if args.compare:
        print("\n" + "=" * 70)
        print("📌 BENCHMARK 2: CGC Engine + OrthoKDA v4")
        print("=" * 70)

        try:
            gc.collect()

            cgc_integration, cgc_backend = benchmark_cgc_ortho_kda(args.model, model_config)
            cgc_llm = cgc_integration.llm

            if cgc_llm is None:
                print("  ❌ CGC integration failed to load llama.cpp model")
                print("     This is expected if llama.cpp is not properly installed")
                print("     Skipping CGC benchmark...")
                args.compare = False
            else:
                cgc_kv_mem = model_config["n_kv_heads"] * model_config["ortho_base_dim"] * model_config["head_dim"] * 2 * 4 / 1024 / 1024
                print(f"     💾 OrthoKDA KV内存: {cgc_kv_mem:.2f} MB (固定 O(1))")
                print(f"     相比原生KV Cache: {native_kv_mem:.1f} MB")
                if native_kv_mem > 0:
                    print(f"     内存节省: {(1 - cgc_kv_mem / native_kv_mem) * 100:.1f}%")

                print("\n  [Prefill Tests]")
                for length in args.prefill_lens:
                    prompt = prompts["medium"] * (length // len(prompts["medium"]) + 1)
                    prompt = prompt[:length]
                    cgc_integration.reset()
                    result = run_prefill_benchmark(cgc_llm, prompt, f"context={length}")
                    prefill_results_cgc[length] = result

                print("\n  [Decode Tests]")
                gc.collect()
                cgc_integration.reset()
                decode_result_cgc = run_decode_benchmark(
                    cgc_llm, prompts["short"], args.max_tokens, "50 tokens"
                )

        except Exception as e:
            print(f"\n  ⚠️ CGC + OrthoKDA benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            print("\n  Skipping CGC benchmark...")
            args.compare = False

    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)

    print("\n  [Prefill Speed Comparison]")
    print(f"  {'Context':<12} {'Native':<20} {'CGC+KDA':<20} {'Speedup':<10}")
    print(f"  {'-'*12} {'-'*20} {'-'*20} {'-'*10}")

    for length in args.prefill_lens:
        native_speed = prefill_results_native.get(length, {}).get("tokens_per_sec", 0)
        cgc_speed = prefill_results_cgc.get(length, {}).get("tokens_per_sec", 0)
        speedup = cgc_speed / native_speed if native_speed > 0 else 0
        print(f"  {length:<12} {native_speed:<20.1f} {cgc_speed:<20.1f} {speedup:<10.2f}x")

    print("\n  [Decode Speed Comparison]")
    native_decode_speed = decode_result_native.get("tokens_per_sec", 0)
    cgc_decode_speed = decode_result_cgc.get("tokens_per_sec", 0)
    speedup = cgc_decode_speed / native_decode_speed if native_decode_speed > 0 else 0
    print(f"  {'Native':<15}: {native_decode_speed:.1f} tok/s")
    print(f"  {'CGC+KDA':<15}: {cgc_decode_speed:.1f} tok/s")
    if native_decode_speed > 0:
        print(f"  {'Speedup':<15}: {speedup:.2f}x")

    print("\n  [Memory Comparison]")
    print(f"  {'Native KV Cache':<20}: {native_kv_mem:.1f} MB (grows with context)")
    if cgc_kv_mem > 0:
        print(f"  {'OrthoKDA KV':<20}: {cgc_kv_mem:.2f} MB (fixed O(1))")
        if native_kv_mem > 0:
            print(f"  {'Memory Savings':<20}: {(1 - cgc_kv_mem / native_kv_mem) * 100:.1f}%")

    print("\n" + "=" * 70)
    print("✅ Benchmark completed!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())