#!/usr/bin/env python3
"""
MagiCompiler Phase 2: 完整端到端測試
End-to-End Test for Dynamic CUDA Graph Cache + vLLM Integration
"""

import os
import sys
import time
import json
from typing import List, Dict, Any

import torch
from vllm import LLM, SamplingParams
from cgc_engine.utils.envs import cgc_report_path


MODEL_PATH = "/home/gs01/models/Qwen/Qwen2___5-7B-Instruct"


def test_dynamic_graph_cache():
    """測試動態 Shape Graph 緩存"""
    print("\n" + "=" * 70)
    print("測試 1: 動態 Shape CUDA Graph 緩存")
    print("=" * 70)

    # 簡單模型
    class SimpleTransformer(torch.nn.Module):
        def __init__(self, d_model=512, nhead=8, num_layers=2):
            super().__init__()
            self.attention = torch.nn.MultiheadAttention(d_model, nhead, batch_first=True).cuda()
            self.linear1 = torch.nn.Linear(d_model, d_model * 4).cuda()
            self.linear2 = torch.nn.Linear(d_model * 4, d_model).cuda()
            self.norm1 = torch.nn.LayerNorm(d_model).cuda()
            self.norm2 = torch.nn.LayerNorm(d_model).cuda()

        def forward(self, x):
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            ff_out = self.linear2(torch.nn.functional.relu(self.linear1(x)))
            x = self.norm2(x + ff_out)
            return x

    model = SimpleTransformer()

    # 測試不同序列長度
    seq_lens = [128, 256, 512, 1024, 2048]
    results = {}

    # 創建 Graph 管理器
    from cgc_engine.cuda.dynamic_graph_cache import PrefillDecodeGraphManager
    manager = PrefillDecodeGraphManager(max_decode_cache_size=len(seq_lens) + 5)

    for seq_len in seq_lens:
        print(f"\n  [seq_len={seq_len}]")

        # 創建輸入
        input_tensor = torch.randn(1, seq_len, 512).cuda()

        # 捕獲 Graph
        capture_start = time.time()
        manager.capture_decode_for_seq_len(model, seq_len, input_tensor)
        capture_ms = (time.time() - capture_start) * 1000
        print(f"    捕獲時間: {capture_ms:.2f} ms")

        # 性能測試
        num_iters = 50

        # Graph 模式
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            _ = manager.replay_decode(seq_len, input_tensor)
        torch.cuda.synchronize()
        graph_ms = (time.time() - start) * 1000 / num_iters

        # Eager 模式
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            with torch.no_grad():
                _ = model(input_tensor)
        torch.cuda.synchronize()
        eager_ms = (time.time() - start) * 1000 / num_iters

        speedup = eager_ms / graph_ms if graph_ms > 0 else 0

        results[seq_len] = {
            "eager_ms": eager_ms,
            "graph_ms": graph_ms,
            "speedup": speedup,
        }

        print(f"    Eager: {eager_ms:.3f} ms | Graph: {graph_ms:.3f} ms | Speedup: {speedup:.2f}x")

    # 打印統計
    print("\n  [圖表] 不同序列長度的加速比:")
    for seq_len, r in results.items():
        bar = "█" * int(r["speedup"] * 10)
        print(f"    {seq_len:5d}: {bar} {r['speedup']:.2f}x")

    # 測試緩存命中率
    cache_stats = manager.decode_cache.get_stats()
    print(f"\n  [緩存統計]")
    print(f"    命中率: {cache_stats['hit_rate']:.1%}")
    print(f"    緩存大小: {cache_stats['cache_size']}")
    print(f"    Graph 命中: {cache_stats['cache_hits']}")
    print(f"    Graph 未命中: {cache_stats['cache_misses']}")

    return results


def test_vllm_basic_inference():
    """測試 vLLM 基礎推理"""
    print("\n" + "=" * 70)
    print("測試 2: vLLM 基礎推理 + CUDA Graph")
    print("=" * 70)

    from cgc_engine.cuda.vllm_cuda_graph_engine import VLLMCudaGraphEngine

    # 創建引擎
    engine = VLLMCudaGraphEngine(
        model_path=MODEL_PATH,
        enable_cudagraph=True,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        tensor_parallel_size=1,
    )

    # 預熱
    engine.warmup("Hello", max_tokens=8)

    # 測試提示
    prompts = [
        "Hello, my name is",
        "The quick brown fox jumps over the",
        "Artificial intelligence is transforming",
    ]

    # 單次推理測試
    print("\n  [單次推理測試]")
    for prompt in prompts:
        result = engine.generate([prompt], SamplingParams(max_tokens=32))
        r = result[0]
        print(f"    Prompt: {prompt}")
        print(f"    Output: {r.output_text[:50]}...")
        print(f"    Time: {r.total_time_ms:.2f} ms | Tokens: {r.num_output_tokens}")
        print()

    # 基準測試
    print("\n  [基準測試]")
    benchmark_result = engine.benchmark(
        prompts=["Hello world"],
        sampling_params=SamplingParams(max_tokens=64),
        num_iterations=3
    )

    return benchmark_result


def test_different_context_lengths():
    """測試不同上下文長度"""
    print("\n" + "=" * 70)
    print("測試 3: 不同上下文長度測試")
    print("=" * 70)

    from cgc_engine.cuda.vllm_cuda_graph_engine import VLLMCudaGraphEngine

    context_lengths = [512, 1024, 2048, 4096]
    results = {}

    for ctx_len in context_lengths:
        print(f"\n  [ctx_len={ctx_len}]")

        # 創建引擎
        engine = VLLMCudaGraphEngine(
            model_path=MODEL_PATH,
            enable_cudagraph=True,
            gpu_memory_utilization=0.6,
            max_model_len=ctx_len + 256,
            tensor_parallel_size=1,
        )

        # 預熱
        warmup_prompt = "Hello" * (ctx_len // 8)
        engine.warmup(warmup_prompt, max_tokens=8)

        # 生成提示
        prompt = "Hello" * (ctx_len // 8)

        # 推理測試
        result = engine.generate(
            [prompt],
            SamplingParams(max_tokens=32, temperature=0.0)
        )

        r = result[0]
        print(f"    推理時間: {r.total_time_ms:.2f} ms")
        print(f"    輸出 tokens: {r.num_output_tokens}")
        print(f"    峰值內存: {r.memory_allocated_gb:.2f} GB")

        results[ctx_len] = {
            "time_ms": r.total_time_ms,
            "output_tokens": r.num_output_tokens,
            "memory_gb": r.memory_allocated_gb,
        }

    # 打印對比表
    print("\n  [上下文長度對比]")
    print(f"  {'Ctx Len':>10} | {'Time (ms)':>12} | {'Tokens':>8} | {'Memory (GB)':>12}")
    print("  " + "-" * 50)
    for ctx_len, r in results.items():
        print(f"  {ctx_len:>10} | {r['time_ms']:>12.2f} | {r['output_tokens']:>8} | {r['memory_gb']:>12.2f}")

    return results


def main():
    """主測試函數"""
    print("\n" + "=" * 70)
    print("MagiCompiler Phase 2: 端到端測試")
    print("動態 CUDA Graph 緩存 + vLLM 集成")
    print("=" * 70)

    # 檢查 CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return

    print(f"\n✅ CUDA 可用")
    print(f"   設備: {torch.cuda.get_device_name(0)}")
    print(f"   PyTorch: {torch.__version__}")

    # 運行測試
    try:
        # 測試 1: 動態 Graph 緩存
        graph_results = test_dynamic_graph_cache()

        # 測試 2: vLLM 基礎推理
        vllm_results = test_vllm_basic_inference()

        # 測試 3: 不同上下文長度
        ctx_results = test_different_context_lengths()

        # 保存結果
        output = {
            "graph_cache_results": graph_results,
            "vllm_results": {
                "avg_time_ms": vllm_results.get("avg_total_time_ms", 0),
                "throughput": vllm_results.get("throughput_tokens_per_sec", 0),
            },
            "context_length_results": ctx_results,
        }

        output_path = cgc_report_path("phase2_results.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print("\n" + "=" * 70)
        print("✅ Phase 2 測試完成!")
        print(f"結果已保存到: {output_path}")
        print("=" * 70)

    except Exception as e:
        import traceback
        print(f"\n❌ 測試失敗: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
