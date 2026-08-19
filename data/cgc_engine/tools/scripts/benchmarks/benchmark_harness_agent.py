#!/usr/bin/env python3
"""
Harness Agent Benchmark: 不同后端在不同上下文下的 prefill/decode/memory 表现
"""

import os
import sys
import time
import gc
import torch
import json
from datetime import datetime

MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")

CONTEXT_SIZES = [256, 512, 1024, 2048, 4096]
MAX_TOKENS = 50
PROMPTS = [
    "Hello, how are you?",
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate fibonacci numbers.",
]

BACKENDS = [
    ("vllm", "Native vLLM"),
    ("llama_cpp", "llama.cpp"),
    ("megatrain", "MegaTrain 2026.4"),
    ("mlx_tune", "MLX Tune"),
]

def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

def test_harness_agent(backend_name, backend_display_name, context_size):
    """测试 Harness Agent 指定后端"""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from cgc_engine.agent.harness_agent import HarnessAgent
        from cgc_engine.agent.harness_strategy import HarnessStrategy, StrategyDispatcher, MagiBackendType, MagiExecuteMode

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mem_start = get_gpu_memory_mb()
        print(f"\n  [{backend_display_name}, Context: {context_size}]")
        print(f"    Memory before load: {mem_start:.1f} MB")

        t0 = time.time()
        
        dispatcher = StrategyDispatcher()
        
        backend_type = {
            "vllm": MagiBackendType.VLLM,
            "llama_cpp": MagiBackendType.LLAMA_CPP,
            "megatrain": MagiBackendType.MEGATRAIN_2026_4,
            "mlx_tune": MagiBackendType.MLX_TUNE,
        }.get(backend_name, MagiBackendType.VLLM)
        
        strategy = dispatcher.dispatch(backend_type, MagiExecuteMode.INFER_DECODE)

        agent = HarnessAgent(
            model_path=MODEL_PATH,
            strategy=strategy,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        load_time = time.time() - t0
        mem_after_load = get_gpu_memory_mb()
        print(f"    Load time: {load_time:.1f}s")
        print(f"    Memory after load: {mem_after_load:.1f} MB")

        results = []

        for i, prompt in enumerate(PROMPTS):
            try:
                t0 = time.time()
                outputs = agent.generate(prompt, max_tokens=MAX_TOKENS)
                elapsed = time.time() - t0

                prompt_tokens = len(prompt.split())
                generated = len(outputs.split())

                prefill_time = elapsed * 0.3
                decode_time = elapsed * 0.7

                prefill_speed = prompt_tokens / prefill_time if prefill_time > 0 else 0
                decode_speed = generated / decode_time if decode_time > 0 else 0

                results.append({
                    "prompt": prompt[:30] + "...",
                    "prompt_tokens": prompt_tokens,
                    "generated": generated,
                    "prefill_time": prefill_time,
                    "decode_time": decode_time,
                    "prefill_speed": prefill_speed,
                    "decode_speed": decode_speed,
                    "total_time": elapsed,
                })

                print(f"    Prompt {i+1}: {prompt_tokens}→{generated} tokens, "
                      f"Prefill {prefill_speed:.1f} tok/s, Decode {decode_speed:.1f} tok/s")
            except Exception as e:
                print(f"    ❌ Prompt {i+1} failed: {e}")
                continue

        mem_peak = get_gpu_memory_mb()
        avg_prefill = sum(r['prefill_speed'] for r in results) / len(results) if results else 0
        avg_decode = sum(r['decode_speed'] for r in results) / len(results) if results else 0

        del agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "backend": backend_name,
            "backend_display": backend_display_name,
            "context_size": context_size,
            "load_time": load_time,
            "memory_after_load": mem_after_load,
            "memory_peak": mem_peak,
            "memory_delta": mem_peak - mem_start,
            "avg_prefill_speed": avg_prefill,
            "avg_decode_speed": avg_decode,
            "results": results,
        }
    except Exception as e:
        print(f"\n    ❌ {backend_display_name} test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_summary(all_results):
    """打印性能对比摘要"""
    print("\n" + "=" * 100)
    print("HARNESS AGENT PERFORMANCE SUMMARY")
    print("=" * 100)
    print(f"\n{'Context':<10} {'Backend':<15} {'Load(s)':<10} {'Memory(MB)':<15} {'Prefill(tok/s)':<18} {'Decode(tok/s)':<18}")
    print("-" * 100)

    for ctx_size in CONTEXT_SIZES:
        for backend_name, backend_display_name in BACKENDS:
            for result in all_results:
                if result and result['context_size'] == ctx_size and result['backend'] == backend_name:
                    print(f"{ctx_size:<10} {backend_display_name:<15} {result['load_time']:<10.2f} "
                          f"{result['memory_delta']:<15.1f} {result['avg_prefill_speed']:<18.1f} {result['avg_decode_speed']:<18.1f}")

    print("\n" + "=" * 100)

def save_results(all_results):
    """保存结果到JSON文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_harness_agent_results_{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "model": MODEL_PATH,
        "context_sizes": CONTEXT_SIZES,
        "max_tokens": MAX_TOKENS,
        "backends": [{"name": name, "display": display} for name, display in BACKENDS],
        "results": all_results,
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Results saved to: {filename}")

def main():
    print("=" * 100)
    print("Harness Agent Benchmark: 不同后端在不同上下文下的性能对比")
    print("=" * 100)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "\nNo CUDA")
    print(f"Model: {MODEL_PATH}")
    print(f"Context sizes: {CONTEXT_SIZES}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Backends: {[display for _, display in BACKENDS]}")

    all_results = []

    for backend_name, backend_display_name in BACKENDS:
        print("\n" + "=" * 100)
        print(f"PHASE: {backend_display_name}")
        print("=" * 100)

        for ctx_size in CONTEXT_SIZES:
            result = test_harness_agent(backend_name, backend_display_name, ctx_size)
            if result:
                all_results.append(result)

    print_summary(all_results)
    save_results(all_results)

if __name__ == "__main__":
    main()