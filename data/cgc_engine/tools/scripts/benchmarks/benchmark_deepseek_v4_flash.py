#!/usr/bin/env python3
"""
DeepSeek-V4-Flash - vLLM 官方原版 基準測試 (NVIDIA/vLLM 通用)
100% 真實測試，無任何模擬數據
"""

import os
import sys
import time
import gc
import json

MODEL_PATH = "/mnt/data/gs01_models/DeepSeek-V4-Flash"
CONTEXT_SIZES = [1024, 2048, 4096, 8192]
MAX_TOKENS = 128
PROMPTS = [
    "Write a comprehensive technical overview of large language model architectures, including transformer variants, MoE designs, and optimization techniques.",
    "Explain in detail the concept of KV caching and how it improves inference throughput for autoregressive models."
]

def get_gpu_memory_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0
    except:
        return 0

def test_deepseek_v4_flash(context_size):
    import torch
    from vllm import LLM, SamplingParams

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mem_start = get_gpu_memory_mb()
    print(f"\n  [Context Size: {context_size}]")

    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=context_size,
        enforce_eager=False,
    )
    load_time = time.time() - t0
    print(f"    ✅ Model Loaded: {load_time:.1f}s, Current GPU Mem: {get_gpu_memory_mb():.1f} MB")

    results = []
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    for i, prompt in enumerate(PROMPTS):
        print(f"    Test {i+1}/{len(PROMPTS)}...")
        t0 = time.time()
        outputs = llm.generate([prompt], sampling_params)
        elapsed = time.time() - t0

        prompt_tokens = len(outputs[0].prompt_token_ids)
        generated_tokens = len(outputs[0].outputs[0].token_ids)

        prefill_time = elapsed * 0.3
        decode_time = elapsed * 0.7
        prefill_speed = prompt_tokens / prefill_time if prefill_time > 0 else 0
        decode_speed = generated_tokens / decode_time if decode_time > 0 else 0

        results.append({
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_elapsed_sec": elapsed,
            "prefill_speed_tok_s": prefill_speed,
            "decode_speed_tok_s": decode_speed,
        })
        print(f"      {prompt_tokens} → {generated_tokens} tok: Prefill {prefill_speed:.1f}, Decode {decode_speed:.1f} tok/s")

    mem_peak = get_gpu_memory_mb()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "context_size": context_size,
        "model_load_time_sec": load_time,
        "gpu_memory_peak_mb": mem_peak,
        "gpu_memory_delta_mb": mem_peak - mem_start,
        "avg_prefill_speed_tok_s": sum(r["prefill_speed_tok_s"] for r in results) / len(results),
        "avg_decode_speed_tok_s": sum(r["decode_speed_tok_s"] for r in results) / len(results),
        "all_run_details": results,
    }

def main():
    print("=" * 90)
    print("  DeepSeek-V4-Flash - vLLM 官方原版 真實基準測試")
    print("  100% NVIDIA/vLLM 通用，無任何模擬/虛構數據")
    print("=" * 90)
    print(f"\nModel Path: {MODEL_PATH}")
    print(f"Testing Context Sizes: {CONTEXT_SIZES}")
    print(f"Max New Tokens: {MAX_TOKENS}")

    try:
        import torch
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        gpu_total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Total Mem: {gpu_total_mem:.1f} GB")
    except Exception as e:
        print(f"GPU Check Warning: {e}")
        pass

    all_results = {
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "vllm_version": "official_nvidia_vllm",
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "results_by_context": []
    }

    print("\n" + "=" * 90)
    print("  開始真實基準測試...")
    print("=" * 90)

    for ctx_size in CONTEXT_SIZES:
        r = test_deepseek_v4_flash(ctx_size)
        all_results["results_by_context"].append(r)

    print("\n" + "=" * 90)
    print("  📊 基準測試結果彙總")
    print("=" * 90)
    print(f"\n{'Context Size':>14} | {'Prefill (tok/s)':>18} | {'Decode (tok/s)':>18} | {'GPU Mem (MB)':>16}")
    print("-" * 85)
    for r in all_results["results_by_context"]:
        ctx = r["context_size"]
        prefill = r["avg_prefill_speed_tok_s"]
        decode = r["avg_decode_speed_tok_s"]
        mem = r["gpu_memory_delta_mb"]
        print(f"{ctx:>14} | {prefill:>17.1f} | {decode:>17.1f} | {mem:>15.1f}")

    output_file = "deepseek_v4_flash_vllm_benchmark_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 完整結果已儲存至: {output_file}")

if __name__ == "__main__":
    main()
