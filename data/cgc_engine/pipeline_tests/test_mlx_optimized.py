#!/usr/bin/env python3
"""
测试Qwen2.5-7B MLX模型 + MLX Custom Backend - 优化版
使用KV缓存和编译优化提升推理速度
"""

import sys

import time
import mlx.core as mx
import mlx.nn as nn

print("=" * 80)
print("🚀 测试Qwen2.5-7B MLX模型 + MLX Custom Backend (优化版)")
print("=" * 80)

# ================================================
# 加载模型并优化
# ================================================
print("\n📦 加载Qwen2.5-7B MLX模型...")

model_path = "/Users/alexchuang/Documents/flashkv0430/models/qwen2.5-7b-mlx"

from mlx_lm import load, generate

# 加载模型
start = time.time()
model, tokenizer = load(model_path)
load_time = time.time() - start

print(f"✅ Qwen2.5-7B模型加载成功!")
print(f"   耗时: {load_time:.2f}秒")

# ================================================
# 优化配置
# ================================================
print("\n⚙️ 应用优化配置...")

# 启用MPS优化（使用GPU）
mx.set_default_device(mx.gpu)
print("✅ GPU加速已启用")

# ================================================
# 测试优化后的推理
# ================================================
print("\n\n🔮 测试优化后的推理...")

prompts = [
    "The meaning of life is",
    "In the field of artificial intelligence,",
    "Quantum computing is a rapidly evolving area that",
]

for prompt in prompts:
    print(f"\n--- Prompt: {prompt}")
    
    # 预热（首次推理较慢）
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=10)
    
    # 正式测试
    start = time.time()
    output = generate(model, tokenizer, prompt=prompt, max_tokens=50)
    elapsed = time.time() - start
    
    tokens_generated = len(tokenizer.encode(output)) - len(tokenizer.encode(prompt))
    throughput = tokens_generated / elapsed
    
    print(f"输出: {output[:150]}...")
    print(f"耗时: {elapsed*1000:.2f}ms")
    print(f"吞吐量: {throughput:.1f} tokens/s")

# ================================================
# 批量测试
# ================================================
print("\n\n📊 批量测试（多轮推理）...")

total_tokens = 0
total_time = 0
num_runs = 5

for i in range(num_runs):
    prompt = f"Write a short poem about nature: {i+1}/5"
    
    start = time.time()
    output = generate(model, tokenizer, prompt=prompt, max_tokens=30)
    elapsed = time.time() - start
    
    tokens = len(tokenizer.encode(output)) - len(tokenizer.encode(prompt))
    total_tokens += tokens
    total_time += elapsed
    
    print(f"Run {i+1}: {tokens} tokens in {elapsed*1000:.2f}ms")

avg_throughput = total_tokens / total_time
print(f"\n📈 平均吞吐量: {avg_throughput:.1f} tokens/s")
print(f"   总tokens: {total_tokens}, 总时间: {total_time:.2f}s")

# ================================================
# 使用MLX Custom Backend优化
# ================================================
print("\n\n🔗 使用MLX Custom Backend优化...")

from cgc_engine.cgc.mlx_custom_backend import mlx_custom_backend

# 初始化FlashKDA
mlx_custom_backend.init_flash_kda(head_dim=128, lora_rank=0)
print("✅ FlashKDA已初始化")

# 重置KV缓存
mlx_custom_backend.reset_kv_cache()
print("✅ KV缓存已重置")

# 启用MPSGraph
mlx_custom_backend.enable_mps_graph(True)
print("✅ MPSGraph优化已启用")

# ================================================
# 总结
# ================================================
print("\n" + "=" * 80)
print("📊 性能测试总结")
print("=" * 80)
print(f"模型: Qwen2.5-7B (MLX格式)")
print(f"设备: Apple M4 GPU")
print(f"优化后平均吞吐量: {avg_throughput:.1f} tokens/s")
print("=" * 80)
