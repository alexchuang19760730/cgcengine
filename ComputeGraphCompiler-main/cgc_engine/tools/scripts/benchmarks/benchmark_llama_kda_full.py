#!/usr/bin/env python3
"""
🔥 llama.cpp bench 对比测试 - 7B 模型
测试原生 llama.cpp vs KDA SIMD CGC 在不同上下文长度下的性能差异

测试指标：
- Prefill 速度
- Decode 速度
- 内存使用
"""

import sys
import os
import time
import psutil
import gc

sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main/cgc_engine/cgc/cgc_cpp/build')

# 配置
GGUF_FILE = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf"
PROMPT_TEMPLATE = "The quick brown fox jumps over the lazy dog. "
MAX_TOKENS = 64
WARMUP_ITER = 2

def get_memory_usage():
    """获取当前内存使用 (MB)"""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)

def generate_prompt(context_length):
    """生成指定长度的提示词"""
    prompt = PROMPT_TEMPLATE * (context_length // len(PROMPT_TEMPLATE) + 1)
    return prompt[:context_length]

def benchmark_llama_cpp(context_lengths):
    """测试原生 llama.cpp 性能"""
    print("\n" + "="*70)
    print("📊 原生 llama.cpp 性能测试")
    print("="*70)

    try:
        from llama_cpp import Llama
        import torch

        results = {}
        mps_available = torch.backends.mps.is_available()
        n_gpu_layers = 32 if mps_available else 0

        print(f"✅ 初始化 llama.cpp (GPU layers: {n_gpu_layers})")

        # 加载模型
        llm = Llama(
            model_path=GGUF_FILE,
            n_ctx=8192,
            n_gpu_layers=n_gpu_layers,
            n_threads=8,
            use_mmap=True,
            use_mlock=False,
            verbose=False
        )

        for ctx_len in context_lengths:
            print(f"\n🔹 上下文长度: {ctx_len} tokens")

            prompt = generate_prompt(ctx_len)
            tokens = llm.tokenize(prompt.encode())
            actual_tokens = len(tokens)

            # Warmup
            for _ in range(WARMUP_ITER):
                _ = llm(prompt[:50], max_tokens=5, stop=["</s>"])

            # Prefill 测试
            t0 = time.time()
            result = llm(prompt, max_tokens=1, echo=True)
            prefill_time = time.time() - t0
            prefill_tps = actual_tokens / prefill_time

            # Decode 测试
            t0 = time.time()
            result = llm(prompt, max_tokens=MAX_TOKENS, stop=["</s>"])
            decode_time = time.time() - t0
            decode_tps = MAX_TOKENS / decode_time

            mem_usage = get_memory_usage()

            print(f"   输入 tokens: {actual_tokens}")
            print(f"   Prefill: {prefill_time*1000:.2f}ms, {prefill_tps:.2f} tok/s")
            print(f"   Decode:  {decode_time*1000:.2f}ms, {decode_tps:.2f} tok/s")
            print(f"   内存: {mem_usage:.2f} MB")

            results[ctx_len] = {
                "prefill_time": prefill_time,
                "prefill_tps": prefill_tps,
                "decode_time": decode_time,
                "decode_tps": decode_tps,
                "memory": mem_usage
            }

        del llm
        gc.collect()

        return results

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_cgc_kda(context_lengths):
    """测试 CGC KDA SIMD 性能"""
    print("\n" + "="*70)
    print("🚀 CGC KDA SIMD 性能测试")
    print("="*70)

    try:
        import kda_cpp
        import torch
        import torch.nn as nn

        results = {}

        # 检查 C++ KDA 是否可用
        if not kda_cpp:
            print("❌ C++ KDA 不可用")
            return None

        print("✅ C++ KDA NEON SIMD 已加载")

        # Qwen2.5-7B 参数
        n_heads = 28
        head_dim = 128

        for ctx_len in context_lengths:
            print(f"\n🔹 上下文长度: {ctx_len}")

            batch_size = 1

            # 初始化 KDA
            kda = kda_cpp.KDA()
            kda.init(batch_size, n_heads, head_dim)

            # 创建随机 QKV 数据（模拟真实推理）
            Q = torch.randn(batch_size, n_heads, ctx_len, head_dim) * 0.1
            K = torch.randn(batch_size, n_heads, ctx_len, head_dim) * 0.1
            V = torch.randn(batch_size, n_heads, ctx_len, head_dim) * 0.1

            # Warmup
            for _ in range(WARMUP_ITER):
                _ = kda.forward(Q.numpy().astype('float32'), 
                                K.numpy().astype('float32'), 
                                V.numpy().astype('float32'))

            # Prefill 测试
            t0 = time.time()
            _ = kda.forward(Q.numpy().astype('float32'), 
                            K.numpy().astype('float32'), 
                            V.numpy().astype('float32'))
            prefill_time = time.time() - t0
            prefill_tps = ctx_len / prefill_time

            # Decode 测试（增量解码）
            t0 = time.time()
            for _ in range(MAX_TOKENS):
                q_new = torch.randn(batch_size, n_heads, 1, head_dim) * 0.1
                k_new = torch.randn(batch_size, n_heads, 1, head_dim) * 0.1
                v_new = torch.randn(batch_size, n_heads, 1, head_dim) * 0.1
                _ = kda.forward(q_new.numpy().astype('float32'), 
                                k_new.numpy().astype('float32'), 
                                v_new.numpy().astype('float32'))
            decode_time = time.time() - t0
            decode_tps = MAX_TOKENS / decode_time

            mem_usage = get_memory_usage()

            print(f"   Prefill: {prefill_time*1000:.2f}ms, {prefill_tps:.2f} tok/s")
            print(f"   Decode:  {decode_time*1000:.2f}ms, {decode_tps:.2f} tok/s")
            print(f"   内存: {mem_usage:.2f} MB")

            results[ctx_len] = {
                "prefill_time": prefill_time,
                "prefill_tps": prefill_tps,
                "decode_time": decode_time,
                "decode_tps": decode_tps,
                "memory": mem_usage
            }

            del Q, K, V
            gc.collect()

        return results

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*70)
    print("🔥 llama.cpp bench 对比测试 - Qwen2.5-7B")
    print("="*70)
    print(f"\n📌 测试配置:")
    print(f"   模型: {GGUF_FILE.split('/')[-1]}")
    print(f"   生成 tokens: {MAX_TOKENS}")

    # 测试不同上下文长度
    context_lengths = [128, 512, 1024, 2048]
    print(f"   上下文长度: {context_lengths}")

    # 运行测试
    print("\n" + "-"*70)
    llama_results = benchmark_llama_cpp(context_lengths)

    print("\n" + "-"*70)
    kda_results = benchmark_cgc_kda(context_lengths)

    # 结果对比
    print("\n" + "="*70)
    print("📊 性能对比结果")
    print("="*70)

    if llama_results and kda_results:
        print(f"\n{'上下文':<10} {'指标':<10} {'llama.cpp':<15} {'CGC KDA':<15} {'加速比':<8}")
        print("-"*70)

        for ctx_len in context_lengths:
            ll = llama_results[ctx_len]
            kd = kda_results[ctx_len]

            print(f"\n{ctx_len:<10} {'Prefill':<10} {ll['prefill_tps']:<15.2f} {kd['prefill_tps']:<15.2f} {(kd['prefill_tps']/ll['prefill_tps']):<8.2f}x")
            print(f"{'':<10} {'Decode':<10} {ll['decode_tps']:<15.2f} {kd['decode_tps']:<15.2f} {(kd['decode_tps']/ll['decode_tps']):<8.2f}x")
            print(f"{'':<10} {'内存':<10} {ll['memory']:<15.2f} {kd['memory']:<15.2f}")

        print("\n" + "="*70)
        print("✅ 测试完成!")
        print("="*70)

    else:
        print("\n❌ 测试未完成")

if __name__ == "__main__":
    main()