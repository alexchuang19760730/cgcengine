#!/usr/bin/env python3
"""
vLLM Benchmark: 真實測量記憶體使用
- Native vLLM KV Cache 記憶體
- OrthoKDA v4 KV Cache 記憶體
"""

import gc
import torch

MODEL_PATH = "/home/gs01/models/Qwen/Qwen2___5-7B-Instruct"
CONTEXT_SIZES = [256, 512, 1024, 2048, 4096]

N_BASE = 128
HEAD_DIM = 128
NUM_HEADS = 32

def get_memory_mb():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated() / 1024**2, torch.cuda.max_memory_allocated() / 1024**2
    return 0, 0

def measure_native_vllm_memory(context_size):
    """測量原生 vLLM 在特定上下文下的 KV Cache 記憶體"""
    from vllm import LLM, SamplingParams

    gc.collect()
    torch.cuda.empty_cache()

    mem_start, _ = get_memory_mb()

    print(f"\n  Loading vLLM with context={context_size}...")

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=context_size,
    )

    mem_after_load, _ = get_memory_mb()

    sampling_params = SamplingParams(temperature=0.7, max_tokens=20)

    outputs = llm.generate(["Hello, how are you?"], sampling_params)
    prompt_tokens = len(outputs[0].prompt_token_ids)

    mem_peak, mem_peak_max = get_memory_mb()

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    kv_memory = mem_peak - mem_start

    return {
        "context": context_size,
        "prompt_tokens": prompt_tokens,
        "memory_start": mem_start,
        "memory_after_load": mem_after_load,
        "memory_peak": mem_peak,
        "memory_delta": kv_memory,
    }

def main():
    print("=" * 80)
    print("vLLM KV Cache Memory Benchmark")
    print("=" * 80)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_PATH}")

    results = []

    print("\n" + "=" * 80)
    print("Measuring Native vLLM KV Cache Memory")
    print("=" * 80)

    for ctx_size in CONTEXT_SIZES:
        r = measure_native_vllm_memory(ctx_size)
        results.append(r)
        print(f"    Context {ctx_size}: KV Memory = {r['memory_delta']:.1f} MB")

    print("\n" + "=" * 80)
    print("RESULTS: Native vLLM KV Cache Memory")
    print("=" * 80)

    print(f"\n{'Context':>8} | {'Prompt Tokens':>12} | {'KV Memory (MB)':>14} | {'Memory/GPU %':>12}")
    print("-" * 55)

    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    for r in results:
        ctx = r['context']
        delta = r['memory_delta']
        pct = (delta / (gpu_total * 1024)) * 100
        print(f"{ctx:>8} | {r['prompt_tokens']:>12} | {delta:>14.1f} | {pct:>11.2f}%")

    print("\n" + "=" * 80)
    print("OrthoKDA v4 Theoretical Memory")
    print("=" * 80)

    ortho_kv = N_BASE * HEAD_DIM * 2 * 4 / 1024 / 1024

    print(f"\n  OrthoKDA v4 KV Cache: {N_BASE}x{HEAD_DIM}x2 x 4 bytes = {ortho_kv:.4f} MB (FIXED)")

    print(f"\n{'Context':>8} | {'Native KV':>12} | {'OrthoKDA KV':>12} | {'Savings':>10}")
    print("-" * 50)

    for r in results:
        ctx = r['context']
        native_kv = r['memory_delta']
        savings = (1 - ortho_kv / native_kv) * 100 if native_kv > 0 else 0
        print(f"{ctx:>8} | {native_kv:>11.1f} MB | {ortho_kv:>11.4f} MB | {savings:>9.1f}%")

if __name__ == "__main__":
    main()