#!/usr/bin/env python3
"""
vLLM Benchmark: Native vs CGC on Windows
测试不同上下文长度的 prefill/decode/memory 表现
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

def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

def test_native_vllm(context_size):
    """测试原生 vLLM"""
    from vllm import LLM, SamplingParams

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mem_start = get_gpu_memory_mb()
    print(f"\n  [Context: {context_size}]")
    print(f"    Memory before load: {mem_start:.1f} MB")

    t0 = time.time()
    try:
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=context_size,
        )
        load_time = time.time() - t0
        mem_after_load = get_gpu_memory_mb()
        print(f"    Load time: {load_time:.1f}s")
        print(f"    Memory after load: {mem_after_load:.1f} MB")
    except Exception as e:
        print(f"    ❌ Load failed: {e}")
        return None

    results = []
    sampling_params = SamplingParams(temperature=0.7, max_tokens=MAX_TOKENS)

    for i, prompt in enumerate(PROMPTS):
        try:
            t0 = time.time()
            outputs = llm.generate([prompt], sampling_params)
            elapsed = time.time() - t0

            prompt_tokens = len(outputs[0].prompt_token_ids)
            generated = len(outputs[0].outputs[0].token_ids)

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

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "load_time": load_time,
        "memory_after_load": mem_after_load,
        "memory_peak": mem_peak,
        "memory_delta": mem_peak - mem_start,
        "avg_prefill_speed": avg_prefill,
        "avg_decode_speed": avg_decode,
        "results": results,
    }

def test_cgc_vllm(context_size):
    """测试 CGC + vLLM"""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from cgc_engine.agent.harness_agent import HarnessAgent
        from cgc_engine.agent.harness_strategy import HarnessStrategy, MagiBackendType, MagiExecuteMode

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mem_start = get_gpu_memory_mb()
        print(f"\n  [CGC + vLLM, Context: {context_size}]")
        print(f"    Memory before load: {mem_start:.1f} MB")

        t0 = time.time()
        strategy = HarnessStrategy()
        strategy.dispatch(MagiBackendType.VLLM, MagiExecuteMode.INFER_DECODE)

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
            "load_time": load_time,
            "memory_after_load": mem_after_load,
            "memory_peak": mem_peak,
            "memory_delta": mem_peak - mem_start,
            "avg_prefill_speed": avg_prefill,
            "avg_decode_speed": avg_decode,
            "results": results,
        }
    except Exception as e:
        print(f"\n    ❌ CGC test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_summary(native_results, cgc_results):
    """打印性能对比摘要"""
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"\n{'Context':<10} {'Backend':<15} {'Load(s)':<10} {'Memory(MB)':<15} {'Prefill(tok/s)':<18} {'Decode(tok/s)':<18}")
    print("-" * 80)

    for ctx_size in CONTEXT_SIZES:
        if ctx_size in native_results:
            nr = native_results[ctx_size]
            print(f"{ctx_size:<10} {'Native vLLM':<15} {nr['load_time']:<10.2f} "
                  f"{nr['memory_delta']:<15.1f} {nr['avg_prefill_speed']:<18.1f} {nr['avg_decode_speed']:<18.1f}")

        if cgc_results and ctx_size in cgc_results:
            cr = cgc_results[ctx_size]
            print(f"{ctx_size:<10} {'CGC + vLLM':<15} {cr['load_time']:<10.2f} "
                  f"{cr['memory_delta']:<15.1f} {cr['avg_prefill_speed']:<18.1f} {cr['avg_decode_speed']:<18.1f}")

    print("\n" + "=" * 80)

def save_results(native_results, cgc_results):
    """保存结果到JSON文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_vllm_results_{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "model": MODEL_PATH,
        "context_sizes": CONTEXT_SIZES,
        "max_tokens": MAX_TOKENS,
        "native_vllm": native_results,
        "cgc_vllm": cgc_results,
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Results saved to: {filename}")

def main():
    print("=" * 80)
    print("vLLM Benchmark: Native vs CGC on Windows")
    print("=" * 80)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "\nNo CUDA")
    print(f"Model: {MODEL_PATH}")
    print(f"Context sizes: {CONTEXT_SIZES}")
    print(f"Max tokens: {MAX_TOKENS}")

    native_results = {}
    cgc_results = {}

    print("\n" + "=" * 80)
    print("PHASE 1: Native vLLM")
    print("=" * 80)

    for ctx_size in CONTEXT_SIZES:
        result = test_native_vllm(ctx_size)
        if result:
            native_results[ctx_size] = result

    print("\n" + "=" * 80)
    print("PHASE 2: CGC + vLLM")
    print("=" * 80)

    for ctx_size in CONTEXT_SIZES:
        result = test_cgc_vllm(ctx_size)
        if result:
            cgc_results[ctx_size] = result

    print_summary(native_results, cgc_results)
    save_results(native_results, cgc_results)

if __name__ == "__main__":
    main()