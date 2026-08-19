#!/usr/bin/env python3
"""
MagiCompiler Phase 3: 大規模性能測試
Comprehensive Performance Testing Suite
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any, Tuple

import torch
from vllm import LLM, SamplingParams
from vllm.utils import random_uuid
from cgc_engine.utils.envs import cgc_report_path


MODEL_PATH = "/home/gs01/models/Qwen/Qwen2___5-7B-Instruct"


def generate_prompt(length: int) -> str:
    """生成指定長度的提示"""
    words = ["Hello", "world", "this", "is", "a", "test", "prompt", "for", "performance", "benchmark"]
    prompt = ""
    while len(prompt.split()) < length:
        prompt += " ".join(words) + " "
    return prompt.strip()[:length * 5]  # 大約每個單詞5個字符


def run_vllm_benchmark(
    model_path: str,
    input_lens: List[int],
    output_lens: List[int],
    batch_sizes: List[int],
    num_iterations: int = 3,
    enable_cudagraph: bool = True,
) -> Dict[str, Any]:
    """
    運行 vLLM 基準測試

    Args:
        model_path: 模型路徑
        input_lens: 輸入序列長度列表
        output_lens: 輸出序列長度列表
        batch_sizes: 批次大小列表
        num_iterations: 迭代次數
        enable_cudagraph: 是否啟用 CUDA Graph

    Returns:
        測試結果
    """
    results = {}

    for batch_size in batch_sizes:
        results[batch_size] = {}

        # 創建引擎
        print(f"\n{'='*70}")
        print(f"Batch Size: {batch_size}")
        print(f"{'='*70}")

        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            max_model_len=8192,
            enforce_eager=not enable_cudagraph,
        )

        # 預熱
        print("預熱中...")
        warmup_prompts = ["Hello"] * min(batch_size, 4)
        warmup_sampling_params = SamplingParams(max_tokens=8)
        _ = llm.generate(warmup_prompts, warmup_sampling_params)

        for input_len in input_lens:
            results[batch_size][input_len] = {}

            print(f"\n  Input Length: {input_len}")

            for output_len in output_lens:
                print(f"    Output Length: {output_len}")

                # 準備提示
                prompts = [generate_prompt(input_len) for _ in range(batch_size)]
                sampling_params = SamplingParams(
                    max_tokens=output_len,
                    temperature=0.0,
                )

                # 性能測試
                total_time = 0.0
                total_tokens = 0

                for i in range(num_iterations):
                    start_time = time.time()
                    outputs = llm.generate(prompts, sampling_params)
                    elapsed = (time.time() - start_time) * 1000

                    # 統計生成的 tokens
                    tokens_generated = sum(len(out.outputs[0].token_ids) for out in outputs)

                    total_time += elapsed
                    total_tokens += tokens_generated

                    print(f"      Iteration {i+1}: {elapsed:.2f} ms, {tokens_generated} tokens")

                # 計算平均值
                avg_time_ms = total_time / num_iterations
                avg_tokens = total_tokens / num_iterations
                throughput = (avg_tokens * 1000) / avg_time_ms if avg_time_ms > 0 else 0

                results[batch_size][input_len][output_len] = {
                    "avg_time_ms": avg_time_ms,
                    "avg_tokens_generated": avg_tokens,
                    "throughput_tokens_per_sec": throughput,
                }

                print(f"      Avg Time: {avg_time_ms:.2f} ms | Throughput: {throughput:.2f} tokens/s")

        # 清理
        del llm
        torch.cuda.empty_cache()

    return results


def run_cgc_benchmark(
    model_path: str,
    input_lens: List[int],
    output_lens: List[int],
    batch_sizes: List[int],
    num_iterations: int = 3,
) -> Dict[str, Any]:
    """
    運行 MagiCompiler + CUDA Graph 基準測試

    Args:
        model_path: 模型路徑
        input_lens: 輸入序列長度列表
        output_lens: 輸出序列長度列表
        batch_sizes: 批次大小列表
        num_iterations: 迭代次數

    Returns:
        測試結果
    """
    from cgc_engine.cuda.vllm_cuda_graph_engine import VLLMCudaGraphEngine

    results = {}

    for batch_size in batch_sizes:
        results[batch_size] = {}

        print(f"\n{'='*70}")
        print(f"[CGC] Batch Size: {batch_size}")
        print(f"{'='*70}")

        # 創建引擎
        engine = VLLMCudaGraphEngine(
            model_path=model_path,
            enable_cudagraph=True,
            gpu_memory_utilization=0.6,
            max_model_len=8192,
            tensor_parallel_size=1,
        )

        # 預熱
        print("預熱中...")
        engine.warmup("Hello", max_tokens=8)

        for input_len in input_lens:
            results[batch_size][input_len] = {}

            print(f"\n  Input Length: {input_len}")

            for output_len in output_lens:
                print(f"    Output Length: {output_len}")

                prompts = [generate_prompt(input_len) for _ in range(batch_size)]
                sampling_params = SamplingParams(
                    max_tokens=output_len,
                    temperature=0.0,
                )

                # 性能測試
                benchmark_result = engine.benchmark(
                    prompts=prompts,
                    sampling_params=sampling_params,
                    num_iterations=num_iterations,
                )

                results[batch_size][input_len][output_len] = {
                    "avg_time_ms": benchmark_result["avg_total_time_ms"],
                    "avg_tokens_generated": benchmark_result["total_output_tokens"] / num_iterations,
                    "throughput_tokens_per_sec": benchmark_result["throughput_tokens_per_sec"],
                }

        # 清理
        del engine
        torch.cuda.empty_cache()

    return results


def run_attention_backend_benchmark(
    seq_lens: List[int],
    num_heads: int = 32,
    head_dim: int = 128,
    num_iterations: int = 50,
) -> Dict[str, Any]:
    """
    運行注意力後端基準測試

    Args:
        seq_lens: 序列長度列表
        num_heads: 頭數
        head_dim: 頭維度
        num_iterations: 迭代次數

    Returns:
        測試結果
    """
    from magi_attention_backend import MagiAttentionBackend, AttentionConfig

    results = {}

    print("\n" + "=" * 70)
    print("注意力後端基準測試")
    print("=" * 70)

    # 測試不同配置
    configs = [
        {"name": "Flash Attention 2", "use_flash_attn": True, "use_kda": False},
        {"name": "PyTorch (Fallback)", "use_flash_attn": False, "use_kda": False},
    ]

    for config in configs:
        print(f"\n  [Config: {config['name']}]")
        results[config["name"]] = {}

        backend = MagiAttentionBackend(
            AttentionConfig(
                num_heads=num_heads,
                head_dim=head_dim,
                use_flash_attn=config["use_flash_attn"],
                use_kda=config["use_kda"],
            )
        )

        for seq_len in seq_lens:
            print(f"    Seq Len: {seq_len}")

            # 生成隨機輸入
            query = torch.randn(1, seq_len, num_heads, head_dim).cuda()
            key = torch.randn(1, seq_len, num_heads, head_dim).cuda()
            value = torch.randn(1, seq_len, num_heads, head_dim).cuda()

            # 捕獲 Graph
            backend.capture_graph(seq_len, query, key, value)

            # Eager 模式
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = backend.forward_func(query, key, value)
            torch.cuda.synchronize()
            eager_time = (time.time() - start) * 1000 / num_iterations

            # Graph 模式
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_iterations):
                _ = backend.forward(query, key, value, use_graph=True)
            torch.cuda.synchronize()
            graph_time = (time.time() - start) * 1000 / num_iterations

            speedup = eager_time / graph_time if graph_time > 0 else 0

            results[config["name"]][seq_len] = {
                "eager_ms": eager_time,
                "graph_ms": graph_time,
                "speedup": speedup,
            }

            print(f"      Eager: {eager_time:.3f} ms | Graph: {graph_time:.3f} ms | Speedup: {speedup:.2f}x")

    return results


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="MagiCompiler Phase 3 性能測試")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="模型路徑")
    parser.add_argument("--output", type=str, default=None, help="輸出文件")
    parser.add_argument("--iterations", type=int, default=3, help="迭代次數")
    args = parser.parse_args()

    print("=" * 70)
    print("MagiCompiler Phase 3: 大規模性能測試")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    all_results = {}

    # 測試配置
    input_lens = [512, 1024, 2048]
    output_lens = [64, 128]
    batch_sizes = [1, 2, 4]

    # 測試 1: vLLM 標準基準
    print("\n" + "=" * 70)
    print("測試 1: vLLM 標準基準")
    print("=" * 70)
    all_results["vllm_standard"] = run_vllm_benchmark(
        model_path=args.model,
        input_lens=input_lens,
        output_lens=output_lens,
        batch_sizes=batch_sizes,
        num_iterations=args.iterations,
        enable_cudagraph=False,
    )

    # 測試 2: vLLM + CUDA Graph
    print("\n" + "=" * 70)
    print("測試 2: vLLM + CUDA Graph")
    print("=" * 70)
    all_results["vllm_cudagraph"] = run_vllm_benchmark(
        model_path=args.model,
        input_lens=input_lens,
        output_lens=output_lens,
        batch_sizes=batch_sizes,
        num_iterations=args.iterations,
        enable_cudagraph=True,
    )

    # 測試 3: MagiCompiler + CUDA Graph
    print("\n" + "=" * 70)
    print("測試 3: MagiCompiler + CUDA Graph")
    print("=" * 70)
    all_results["magi_cudagraph"] = run_cgc_benchmark(
        model_path=args.model,
        input_lens=input_lens,
        output_lens=output_lens,
        batch_sizes=batch_sizes,
        num_iterations=args.iterations,
    )

    # 測試 4: 注意力後端基準
    print("\n" + "=" * 70)
    print("測試 4: 注意力後端基準")
    print("=" * 70)
    all_results["attention_backend"] = run_attention_backend_benchmark(
        seq_lens=[128, 256, 512, 1024],
        num_heads=32,
        head_dim=128,
        num_iterations=50,
    )

    # 保存結果
    output_path = args.output
    if not output_path:
        output_path = cgc_report_path("phase3_results.json")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"測試完成! 結果已保存到: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
