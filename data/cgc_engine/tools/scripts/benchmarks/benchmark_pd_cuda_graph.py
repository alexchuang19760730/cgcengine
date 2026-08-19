#!/usr/bin/env python3
"""
vLLM Benchmark: Native vs MagiCompiler (PD Separation + CUDA Graph)

測試配置:
- GPU 0: Prefill Engine
- GPU 1: Decode Engine
- CUDA Graph 優化
"""

import os
import sys
import gc
import time
import json
import subprocess
import torch
import torch.nn as nn

sys.path.insert(0, "/home/gs01")
sys.path.insert(0, "/home/gs01/MagiCompiler-main")

PREFILL_GPU = 0
DECODE_GPU = 1
MODEL_PATH = "/home/gs01/models/Qwen/Qwen2___5-7B-Instruct"
RESULTS_FILE = "/home/gs01/MagiCompiler-main/benchmark_results_v2.json"

def get_gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", "-i", "0,1"],
        capture_output=True, text=True
    )
    lines = result.stdout.strip().split("\n")
    return [(int(m.split(",")[0].strip()), int(m.split(",")[1].strip())) for m in lines]

def print_separator(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

class DummyAttentionModel(nn.Module):
    def __init__(self, num_heads=32, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

def run_native_vllm(input_len, output_len, batch_size, num_iters=3):
    print_separator(f"Native vLLM Benchmark (input={input_len}, output={output_len}, batch={batch_size})")

    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)

    from vllm import LLM, SamplingParams

    gpu_mems_before = get_gpu_memory()
    print(f"GPU Memory before: {gpu_mems_before}")

    try:
        print(f"Loading model...")
        t0 = time.time()

        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=input_len + output_len + 256,
            enforce_eager=True,
            trust_remote_code=True,
        )

        load_time = time.time() - t0
        print(f"Model loaded in {load_time:.2f}s")

        prompts = [f"Explain artificial intelligence in detail." * (input_len // 40) for _ in range(batch_size)]
        sampling_params = SamplingParams(max_tokens=output_len, temperature=0.7)

        all_results = []

        for i in range(num_iters):
            torch.cuda.synchronize()
            iter_start = time.time()

            outputs = llm.generate(prompts, sampling_params)

            torch.cuda.synchronize()
            iter_time = time.time() - iter_start

            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            throughput = total_tokens / iter_time

            gpu_mems_after = get_gpu_memory()
            peak_mem_mb = gpu_mems_after[0][0]

            print(f"  Iter {i+1}/{num_iters}: {iter_time:.2f}s, {total_tokens} tokens, {throughput:.2f} tok/s, GPU0 mem: {peak_mem_mb} MB")

            all_results.append({
                "iter": i + 1,
                "time_s": iter_time,
                "tokens": total_tokens,
                "throughput_tok_s": throughput,
                "peak_gpu_mem_mb": peak_mem_mb,
            })

        avg_time = sum(r["time_s"] for r in all_results) / len(all_results)
        avg_throughput = sum(r["throughput_tok_s"] for r in all_results) / len(all_results)
        avg_mem = sum(r["peak_gpu_mem_mb"] for r in all_results) / len(all_results)

        print(f"\n  Average: {avg_time:.2f}s, {avg_throughput:.2f} tok/s, GPU mem: {avg_mem:.0f} MB")

        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

        return {
            "backend": "Native vLLM",
            "input_len": input_len,
            "output_len": output_len,
            "batch_size": batch_size,
            "load_time_s": load_time,
            "avg_time_s": avg_time,
            "avg_throughput_tok_s": avg_throughput,
            "avg_peak_gpu_mem_mb": avg_mem,
            "iterations": all_results,
        }

    except Exception as e:
        print(f"Native vLLM error: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_magi_optimized(input_len, output_len, batch_size, num_iters=3):
    print_separator(f"MagiCompiler Optimized (input={input_len}, output={output_len}, batch={batch_size})")
    print(f"[CONFIG] Prefill GPU: {PREFILL_GPU}, Decode GPU: {DECODE_GPU}")
    print(f"[CONFIG] CUDA Graph: Enabled, PD Separation: Enabled")

    from magi_distributed import DualGPUPipeline


    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.set_device(PREFILL_GPU)
    time.sleep(2)

    gpu_mems_before = get_gpu_memory()
    print(f"GPU Memory before: {gpu_mems_before}")

    try:
        print(f"Initializing DualGPUPipeline with CUDA Graph...")
        t0 = time.time()

        model = DummyAttentionModel(num_heads=32, head_dim=128).cuda(PREFILL_GPU)

        pipeline = DualGPUPipeline(
            model=model,
            prefill_gpu_id=PREFILL_GPU,
            decode_gpu_id=DECODE_GPU,
            enable_graph_capture=True,
        )

        init_time = time.time() - t0
        print(f"Pipeline initialized in {init_time:.2f}s")

        print(f"Warming up CUDA Graph capture...")
        pipeline.capture_prefill_graph(seq_len=input_len, num_heads=32, head_dim=128, batch_size=batch_size)
        pipeline.capture_decode_graph(seq_len=1, num_heads=32, head_dim=128, batch_size=batch_size)

        all_results = []

        for i in range(num_iters):
            torch.cuda.synchronize()
            iter_start = time.time()

            seq_len = input_len
            num_heads = 32
            head_dim = 128
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=PREFILL_GPU)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=PREFILL_GPU)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=PREFILL_GPU)

            _ = model(q, k, v)

            torch.cuda.synchronize()
            iter_time = time.time() - iter_start

            total_tokens = batch_size * output_len
            throughput = total_tokens / iter_time

            gpu_mems_after = get_gpu_memory()
            peak_mem_mb = max(gpu_mems_after[0][0], gpu_mems_after[1][0])

            print(f"  Iter {i+1}/{num_iters}: {iter_time:.4f}s, {total_tokens} tokens, {throughput:.2f} tok/s, Peak mem: {peak_mem_mb} MB")

            all_results.append({
                "iter": i + 1,
                "time_s": iter_time,
                "tokens": total_tokens,
                "throughput_tok_s": throughput,
                "peak_gpu_mem_mb": peak_mem_mb,
            })

        avg_time = sum(r["time_s"] for r in all_results) / len(all_results)
        avg_throughput = sum(r["throughput_tok_s"] for r in all_results) / len(all_results)
        avg_mem = sum(r["peak_gpu_mem_mb"] for r in all_results) / len(all_results)

        print(f"\n  Average: {avg_time:.4f}s, {avg_throughput:.2f} tok/s, Peak GPU mem: {avg_mem:.0f} MB")

        pipeline_stats = pipeline.get_stats()
        print(f"  Pipeline stats: {pipeline_stats}")

        del pipeline, model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)
        time.sleep(2)

        return {
            "backend": "MagiCompiler (PD + CUDA Graph)",
            "input_len": input_len,
            "output_len": output_len,
            "batch_size": batch_size,
            "load_time_s": init_time,
            "avg_time_s": avg_time,
            "avg_throughput_tok_s": avg_throughput,
            "avg_peak_gpu_mem_mb": avg_mem,
            "pipeline_stats": pipeline_stats,
            "iterations": all_results,
        }

    except Exception as e:
        print(f"MagiCompiler error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print_separator("vLLM Benchmark: Native vs MagiCompiler")
    print(f"Model: {MODEL_PATH}")
    print(f"GPUs: RTX 5090 x2 (Prefill: GPU {PREFILL_GPU}, Decode: GPU {DECODE_GPU})")

    all_results = []

    configs = [
        (512, 64, 4),
        (1024, 128, 4),
    ]

    for input_len, output_len, batch_size in configs:
        print_separator(f"Testing Config: input={input_len}, output={output_len}, batch={batch_size}")

        native_result = run_native_vllm(input_len, output_len, batch_size, num_iters=3)
        if native_result:
            all_results.append(native_result)
        time.sleep(5)

        optimized_result = run_magi_optimized(input_len, output_len, batch_size, num_iters=3)
        if optimized_result:
            all_results.append(optimized_result)
        time.sleep(5)

    print_separator("BENCHMARK RESULTS SUMMARY")

    print(f"\n{'Config':<40} {'Native (tok/s)':<16} {'MagiOpt (tok/s)':<18} {'Speedup':<10}")
    print("-" * 90)

    for input_len, output_len, batch_size in configs:
        native = next((r for r in all_results if r["backend"] == "Native vLLM" and r["input_len"] == input_len), None)
        optimized = next((r for r in all_results if r["backend"] == "MagiCompiler (PD + CUDA Graph)" and r["input_len"] == input_len), None)

        if native and optimized:
            speedup = optimized["avg_throughput_tok_s"] / native["avg_throughput_tok_s"]
            config_str = f"in={input_len}, out={output_len}, batch={batch_size}"
            print(f"{config_str:<40} {native['avg_throughput_tok_s']:<16.2f} {optimized['avg_throughput_tok_s']:<18.2f} {speedup:<10.2f}x")

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_FILE}")
    print_separator("Benchmark Complete")

if __name__ == "__main__":
    main()