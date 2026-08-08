#!/usr/bin/env python3
"""
vLLM GPU Memory Benchmark - 測量真實 GPU 記憶體使用

使用 nvidia-smi 和 vLLM 內建 memory tracking
"""

import subprocess
import time
import gc
import sys
import re

def get_gpu_memory_nvidia_smi():
    """使用 nvidia-smi 獲取真實 GPU 記憶體使用"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split('\n')[0])
    except Exception as e:
        print(f"nvidia-smi error: {e}")
    return None

def get_gpu_memory_torch():
    """使用 PyTorch 獲取記憶體"""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

def test_native_vllm_mem(context_size, model_path):
    """測量原生 vLLM 記憶體"""
    import torch
    from vllm import LLM, SamplingParams

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    mem_before = get_gpu_memory_nvidia_smi()
    torch_mem_before = get_gpu_memory_torch()

    print(f"\n  [Context: {context_size}]")
    print(f"    GPU Memory (nvidia-smi) before: {mem_before:.1f} MB")

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=context_size,
    )

    torch.cuda.synchronize()
    mem_after_load = get_gpu_memory_nvidia_smi()
    torch_mem_after = get_gpu_memory_torch()

    print(f"    GPU Memory after load: {mem_after_load:.1f} MB (delta: {mem_after_load - mem_before:.1f} MB)")
    print(f"    PyTorch memory: {torch_mem_after:.1f} MB")

    sampling_params = SamplingParams(temperature=0.7, max_tokens=30)

    t0 = time.time()
    outputs = llm.generate(["Hello, how are you?"], sampling_params)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    mem_during = get_gpu_memory_nvidia_smi()

    print(f"    GPU Memory during inference: {mem_during:.1f} MB (delta from load: {mem_during - mem_after_load:.1f} MB)")
    print(f"    Inference time: {elapsed:.2f}s")

    generated = len(outputs[0].outputs[0].token_ids)
    decode_speed = generated / elapsed if elapsed > 0 else 0
    print(f"    Generated {generated} tokens, {decode_speed:.1f} tok/s")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    mem_after_del = get_gpu_memory_nvidia_smi()
    print(f"    GPU Memory after cleanup: {mem_after_del:.1f} MB")

    return {
        "context": context_size,
        "mem_before": mem_before,
        "mem_after_load": mem_after_load,
        "mem_during": mem_during,
        "mem_delta_load": mem_after_load - mem_before,
        "mem_delta_inference": mem_during - mem_after_load,
        "decode_speed": decode_speed,
    }

def main():
    MODEL_PATH = "/home/gs01/models/Qwen/Qwen2___5-7B-Instruct"
    CONTEXT_SIZES = [256, 512, 1024, 2048, 4096]

    print("=" * 80)
    print("vLLM GPU Memory Benchmark (nvidia-smi)")
    print("=" * 80)

    import torch
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_PATH}")

    print("\n" + "=" * 80)
    print("Native vLLM Memory Usage")
    print("=" * 80)

    results = []

    for ctx in CONTEXT_SIZES:
        r = test_native_vllm_mem(ctx, MODEL_PATH)
        results.append(r)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Context':>8} | {'Load Delta':>12} | {'Inference Delta':>15} | {'Decode Speed':>12}")
    print("-" * 55)

    for r in results:
        ctx = r["context"]
        load_delta = f"{r['mem_delta_load']:.1f} MB"
        inf_delta = f"{r['mem_delta_inference']:.1f} MB"
        speed = f"{r['decode_speed']:.1f} tok/s"
        print(f"{ctx:>8} | {load_delta:>12} | {inf_delta:>15} | {speed:>12}")

    print("\n" + "=" * 80)
    print("OrthoKDA v4 Expected Memory (Fixed O(1))")
    print("=" * 80)

    N_BASE = 128
    HEAD_DIM = 128
    NUM_LAYERS = 28
    ortho_kv_per_layer = N_BASE * HEAD_DIM * 2 * 4 / 1024 / 1024
    ortho_kv_total = ortho_kv_per_layer * NUM_LAYERS

    print(f"""
    OrthoKDA v4 KV Cache per layer: {ortho_kv_per_layer:.4f} MB
    OrthoKDA v4 KV Cache total ({NUM_LAYERS} layers): {ortho_kv_total:.4f} MB

    Memory Savings vs Native vLLM:
    """)

    for r in results:
        ctx = r["context"]
        native_mem = r['mem_delta_load']
        if native_mem > 0:
            saved = (1 - ortho_kv_total / native_mem) * 100
            print(f"      Context {ctx}: {saved:.1f}% memory reduction")

if __name__ == "__main__":
    main()