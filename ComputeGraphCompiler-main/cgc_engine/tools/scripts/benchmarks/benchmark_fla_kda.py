#!/usr/bin/env python3
"""
Native vLLM vs FLA KDA Benchmark

架構:
  Native vLLM:    標準 FlashAttention (O(N^2) softmax attention)
  FLA KDA:        Flash Linear Attention KDA (O(N) 線性注意力)
"""
import os
import sys
import json
import time
import subprocess
import numpy as np
import torch
from typing import List, Dict

MODEL_PATH = '/home/gs01/models/Qwen/Qwen2___5-7B-Instruct'
CONTEXT_LENGTHS = [256, 512, 1024, 2048, 4096]
OUTPUT_LENGTH = 64
BATCH_SIZE = 4
GPU_MEM_UTIL = 0.7


def get_gpu_memory() -> Dict[str, float]:
    result = {"used_gb": 0.0, "total_gb": 0.0, "free_gb": 0.0}
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            encoding="utf-8", stderr=subprocess.DEVNULL
        )
        parts = [float(x.strip()) for x in output.strip().split(",")]
        if len(parts) >= 3:
            result["used_gb"] = parts[0] / 1024.0
            result["total_gb"] = parts[1] / 1024.0
            result["free_gb"] = parts[2] / 1024.0
    except Exception:
        pass
    if torch.cuda.is_available():
        result["torch_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        result["torch_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
    return result


def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# FLA KDA Benchmark
# ============================================================================

def benchmark_fla_kda():
    """
    FLA KDA Benchmark (PyTorch implementation of KDA)
    """
    print("\n" + "=" * 100)
    print("  FLA KDA (Flash Linear Attention)")
    print("=" * 100)

    import fla
    from fla.ops.kda import fused_recurrent_kda

    clear_gpu()
    torch.cuda.reset_peak_memory_stats()

    all_results = []

    B, H, K, V = BATCH_SIZE, 28, 128, 128
    scale = 1.0 / K**0.5

    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n  --- Context Length = {ctx_len} ---")

        T = ctx_len

        q = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(B, T, H, V, device='cuda', dtype=torch.bfloat16)
        g = torch.zeros(B, T, H, K, device='cuda', dtype=torch.bfloat16)
        beta = torch.ones(B, T, H, device='cuda', dtype=torch.bfloat16)

        torch.cuda.synchronize()
        mem_start = get_gpu_memory()

        # Warmup
        for _ in range(3):
            out = fused_recurrent_kda(q, k, v, g, beta, scale=scale)

        torch.cuda.synchronize()

        # Prefill Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        out = fused_recurrent_kda(q, k, v, g, beta, scale=scale)
        end_event.record()
        torch.cuda.synchronize()

        prefill_time_ms = start_event.elapsed_time(end_event)
        prefill_time_s = prefill_time_ms / 1000.0
        prefill_tokens = B * T
        prefill_tps = prefill_tokens / prefill_time_s if prefill_time_s > 0 else 0

        mem_after_prefill = get_gpu_memory()

        # Decode Benchmark (single token steps)
        q_dec = torch.randn(B, 1, H, K, device='cuda', dtype=torch.bfloat16)
        k_dec = torch.randn(B, 1, H, K, device='cuda', dtype=torch.bfloat16)
        v_dec = torch.randn(B, 1, H, V, device='cuda', dtype=torch.bfloat16)
        g_dec = torch.zeros(B, 1, H, K, device='cuda', dtype=torch.bfloat16)
        beta_dec = torch.ones(B, 1, H, device='cuda', dtype=torch.bfloat16)

        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(OUTPUT_LENGTH):
            out_dec = fused_recurrent_kda(q_dec, k_dec, v_dec, g_dec, beta_dec, scale=scale)
        end_event.record()
        torch.cuda.synchronize()

        decode_time_ms = start_event.elapsed_time(end_event)
        decode_time_s = decode_time_ms / 1000.0
        decode_tokens = B * OUTPUT_LENGTH
        decode_tps = decode_tokens / decode_time_s if decode_time_s > 0 else 0

        mem_after_decode = get_gpu_memory()

        result = {
            "context_len": ctx_len,
            "prefill_time_s": prefill_time_s,
            "prefill_tokens": prefill_tokens,
            "prefill_tps": prefill_tps,
            "decode_time_s": decode_time_s,
            "decode_tokens": decode_tokens,
            "decode_tps": decode_tps,
            "mem_used_gb": mem_after_decode.get("used_gb", 0),
            "mem_total_gb": mem_after_decode.get("total_gb", 0),
        }

        all_results.append(result)

        print(f"    Prefill: {prefill_tps:,.1f} tok/s ({prefill_time_ms:.1f}ms, {prefill_tokens} tokens)")
        print(f"    Decode:  {decode_tps:,.1f} tok/s ({decode_time_s*1000:.1f}ms, {decode_tokens} tokens)")
        print(f"    GPU Mem: {mem_after_decode.get('used_gb', 0):.2f} GB")

        del q, k, v, q_dec, k_dec, v_dec, out, out_dec
        clear_gpu()

    return all_results


# ============================================================================
# Native vLLM Benchmark
# ============================================================================

def benchmark_native_vllm():
    print("\n" + "=" * 100)
    print("  Native vLLM (FlashAttention)")
    print("=" * 100)

    os.environ.pop("VLLM_USE_CGC_KDA", None)

    from vllm import LLM, SamplingParams

    clear_gpu()
    torch.cuda.reset_peak_memory_stats()

    load_start = time.time()
    mem_before = get_gpu_memory()

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=True,
    )

    load_time = time.time() - load_start
    mem_after = get_gpu_memory()

    print(f"\n  模型載入時間: {load_time:.2f}s")
    print(f"  GPU 記憶體 (載入後): {mem_after['used_gb']:.2f} GB")

    all_results = []

    for ctx_len in CONTEXT_LENGTHS:
        if ctx_len > 2048:
            ctx_len = 2048

        print(f"\n  --- Context Length = {ctx_len} ---")

        dummy_ids = np.random.randint(10000, size=(BATCH_SIZE, ctx_len)).tolist()
        prompts = [{"prompt_token_ids": x} for x in dummy_ids]

        sp_prefill = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        sp_decode = SamplingParams(temperature=0, max_tokens=OUTPUT_LENGTH, ignore_eos=True)

        llm.generate(prompts, sp_decode, use_tqdm=False)
        clear_gpu()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        outputs = llm.generate(prompts, sp_prefill, use_tqdm=False)
        end_event.record()
        torch.cuda.synchronize()

        prefill_time_ms = start_event.elapsed_time(end_event)
        prefill_time_s = prefill_time_ms / 1000.0
        total_prefill_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        prefill_tps = total_prefill_tokens / prefill_time_s if prefill_time_s > 0 else 0

        mem_after_prefill = get_gpu_memory()

        clear_gpu()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        outputs = llm.generate(prompts, sp_decode, use_tqdm=False)
        end_event.record()
        torch.cuda.synchronize()

        total_time_ms = start_event.elapsed_time(end_event)
        total_time_s = total_time_ms / 1000.0
        total_decode_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

        decode_time_s = total_time_s - prefill_time_s
        decode_tps = total_decode_tokens / decode_time_s if decode_time_s > 0 else 0

        mem_after_decode = get_gpu_memory()

        result = {
            "context_len": ctx_len,
            "prefill_time_s": prefill_time_s,
            "prefill_tokens": total_prefill_tokens,
            "prefill_tps": prefill_tps,
            "decode_time_s": decode_time_s,
            "decode_tokens": total_decode_tokens,
            "decode_tps": decode_tps,
            "total_time_s": total_time_s,
            "mem_used_gb": mem_after_decode["used_gb"],
            "mem_total_gb": mem_after_decode["total_gb"],
        }

        all_results.append(result)

        print(f"    Prefill: {prefill_tps:,.1f} tok/s ({prefill_time_ms:.1f}ms, {total_prefill_tokens} tokens)")
        print(f"    Decode:  {decode_tps:,.1f} tok/s ({decode_time_s*1000:.1f}ms, {total_decode_tokens} tokens)")
        print(f"    GPU Mem: {mem_after_decode['used_gb']:.2f} GB")

    del llm
    clear_gpu()

    return all_results


# ============================================================================
# 結果輸出
# ============================================================================

def print_comparison(native: List[Dict], fla_kda: List[Dict]):
    print("\n")
    print("=" * 160)
    print("  Native vLLM vs FLA KDA (Flash Linear Attention) 對比")
    print("=" * 160)

    header = (
        f"{'Ctx':<6} | "
        f"{'Native-Prefill':<14} {'Native-Decode':<14} {'N-GPU':<8} | "
        f"{'FLA-Prefill':<14} {'FLA-Decode':<14} {'F-GPU':<8} | "
        f"{'PF-Δ':<10} {'DC-Δ':<10}"
    )
    print(header)
    print("-" * 160)

    total_pf = 0
    total_dc = 0
    count = 0

    for n, f in zip(native, fla_kda):
        pf_delta = ((f["prefill_tps"] / n["prefill_tps"]) - 1) * 100 if n["prefill_tps"] > 0 else 0
        dc_delta = ((f["decode_tps"] / n["decode_tps"]) - 1) * 100 if n["decode_tps"] > 0 else 0

        line = (
            f"{n['context_len']:<6} | "
            f"{n['prefill_tps']:>12,.1f}  {n['decode_tps']:>12,.1f}  {n['mem_used_gb']:>6.2f} | "
            f"{f['prefill_tps']:>12,.1f}  {f['decode_tps']:>12,.1f}  {f['mem_used_gb']:>6.2f} | "
            f"{pf_delta:>+8.1f}%  {dc_delta:>+8.1f}%"
        )
        print(line)

        total_pf += pf_delta
        total_dc += dc_delta
        count += 1

    print("=" * 160)

    if count > 0:
        avg_pf = total_pf / count
        avg_dc = total_dc / count
        print(f"\n  平均 Prefill 差異: {avg_pf:+.1f}%")
        print(f"  平均 Decode 差異:  {avg_dc:+.1f}%")

        print(f"\n  說明:")
        print(f"    - Native vLLM: Qwen2.5-7B + FlashAttention (O(N^2))")
        print(f"    - FLA KDA: Flash Linear Attention (O(N) 線性注意力)")
        print(f"    - FLA KDA 僅測試注意力層，非端到端模型")


def main():
    print("=" * 100)
    print("  Native vLLM vs FLA KDA (Flash Linear Attention) 對比")
    print("=" * 100)
    print(f"  Model: Qwen2.5-7B-Instruct")
    print(f"  Context lengths: {CONTEXT_LENGTHS}")
    print(f"  Output length: {OUTPUT_LENGTH}")
    print(f"  Batch size: {BATCH_SIZE}")

    total_start = time.time()

    native_results = benchmark_native_vllm()

    fla_kda_results = benchmark_fla_kda()

    print_comparison(native_results, fla_kda_results)

    out_file = "/home/gs01/benchmark_native_vs_fla_kda.json"
    with open(out_file, "w") as f:
        json.dump({
            "config": {
                "native_model": "Qwen2.5-7B-Instruct",
                "fla_kda": "Flash Linear Attention KDA",
                "context_lengths": CONTEXT_LENGTHS,
                "output_length": OUTPUT_LENGTH,
                "batch_size": BATCH_SIZE,
            },
            "native_vllm": native_results,
            "fla_kda": fla_kda_results,
        }, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - total_start
    print(f"\n  結果保存到: {out_file}")
    print(f"  總耗時: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
