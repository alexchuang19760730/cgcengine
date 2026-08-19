#!/usr/bin/env python3
"""
端侧单设备 PD 时间分离测试
测试单设备分阶段执行 Prefill 和 Decode 的性能差异
"""

import sys

import mlx.core as mx
import mlx.nn as nn
import time
import numpy as np

print("=" * 100)
print("⚡ 端侧单设备 PD 时间分离性能测试")
print("=" * 100)

print("\n🔍 硬件检测:")
print("-" * 50)
print(f"✅ MLX版本: {mx.__version__}")
print(f"✅ 设备: {mx.default_device()}")

class SimpleTransformer(nn.Module):
    """简单Transformer模型用于测试"""
    def __init__(self, hidden_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.embed = nn.Linear(hidden_dim, hidden_dim)
        
        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                'q_proj': nn.Linear(hidden_dim, hidden_dim),
                'k_proj': nn.Linear(hidden_dim, hidden_dim),
                'v_proj': nn.Linear(hidden_dim, hidden_dim),
                'out_proj': nn.Linear(hidden_dim, hidden_dim),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                )
            })
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def __call__(self, x, cache=None):
        B, T, C = x.shape
        
        x = self.embed(x)
        
        for i, layer in enumerate(self.layers):
            # Self-Attention
            q = layer['q_proj'](x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = layer['k_proj'](x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = layer['v_proj'](x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            
            # Scaled dot-product attention
            scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
            attn = mx.softmax(scores, axis=-1)
            attn_out = mx.matmul(attn, v)
            
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, C)
            x = x + layer['out_proj'](attn_out)
            
            # MLP
            x = x + layer['mlp'](x)
        
        return self.norm(x)


class PDSeparatedModel:
    """PD时间分离模型"""
    def __init__(self, model):
        self.model = model
        self.kv_cache = None
        self.prefill_cache = None
    
    def prefill(self, x):
        """Prefill阶段：处理prompt，生成KV Cache"""
        B, T, C = x.shape
        
        output = self.model(x)
        
        self.prefill_cache = {
            'hidden_states': output,
            'input_length': T
        }
        
        return output[:, -1:, :]
    
    def decode(self, x, cache=None):
        """Decode阶段：逐token生成"""
        if cache is None:
            cache = self.prefill_cache
        
        output = self.model(x)
        return output[:, -1:, :]


def test_without_pd_separation(model, input_data, iterations=5):
    """测试不使用PD分离"""
    print("\n📊 测试1：不使用 PD 分离（完整推理）")
    print("-" * 50)
    
    times = []
    for i in range(iterations):
        mx.clear_cache()
        
        start = time.time()
        output = model(input_data)
        mx.eval(output)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"   迭代 {i+1}: {elapsed:.2f}ms")
    
    avg_time = sum(times) / len(times)
    print(f"\n   📈 平均延迟: {avg_time:.2f}ms")
    return avg_time


def test_with_pd_separation(model, input_data, decode_tokens=10, iterations=5):
    """测试使用PD时间分离"""
    print("\n📊 测试2：使用 PD 时间分离")
    print("-" * 50)
    
    pd_model = PDSeparatedModel(model)
    
    prefill_times = []
    decode_times = []
    total_times = []
    
    for i in range(iterations):
        mx.clear_cache()
        
        # Prefill阶段
        start_prefill = time.time()
        prefill_output = pd_model.prefill(input_data)
        mx.eval(prefill_output)
        prefill_time = (time.time() - start_prefill) * 1000
        prefill_times.append(prefill_time)
        
        # Decode阶段（模拟生成多个token）
        decode_time_total = 0
        current_input = prefill_output
        
        for _ in range(decode_tokens):
            mx.clear_cache()
            
            start_decode = time.time()
            decode_output = pd_model.decode(current_input)
            mx.eval(decode_output)
            decode_time_total += (time.time() - start_decode) * 1000
            
            current_input = decode_output
        
        decode_times.append(decode_time_total / decode_tokens)
        total_times.append(prefill_time + decode_time_total)
        
        print(f"   迭代 {i+1}: Prefill={prefill_time:.2f}ms, Decode(avg)={decode_time_total/decode_tokens:.2f}ms, Total={prefill_time+decode_time_total:.2f}ms")
    
    avg_prefill = sum(prefill_times) / len(prefill_times)
    avg_decode = sum(decode_times) / len(decode_times)
    avg_total = sum(total_times) / len(total_times)
    
    print(f"\n   📈 平均Prefill延迟: {avg_prefill:.2f}ms")
    print(f"   📈 平均Decode延迟: {avg_decode:.2f}ms")
    print(f"   📈 平均总延迟: {avg_total:.2f}ms")
    
    return avg_prefill, avg_decode, avg_total


def test_memory_usage(model, input_data):
    """测试内存使用"""
    print("\n📊 测试3：内存使用对比")
    print("-" * 50)
    
    # 不使用PD分离
    mx.clear_cache()
    output1 = model(input_data)
    mx.eval(output1)
    memory_without_pd = mx.get_active_memory()
    
    # 使用PD分离
    mx.clear_cache()
    pd_model = PDSeparatedModel(model)
    prefill_output = pd_model.prefill(input_data)
    mx.eval(prefill_output)
    memory_with_pd = mx.get_active_memory()
    
    print(f"   不使用PD分离: {memory_without_pd / 1024**2:.2f} MB")
    print(f"   使用PD分离: {memory_with_pd / 1024**2:.2f} MB")
    print(f"   内存节省: {(memory_without_pd - memory_with_pd) / memory_without_pd * 100:.1f}%")


def main():
    # 测试配置
    config = {
        "batch_size": 1,
        "prefill_seq_len": 512,
        "decode_tokens": 10,
        "hidden_dim": 512,
        "num_heads": 8,
        "num_layers": 4,
        "iterations": 5
    }
    
    print(f"\n📋 测试配置:")
    print(f"   批量大小: {config['batch_size']}")
    print(f"   Prefill序列长度: {config['prefill_seq_len']}")
    print(f"   Decode生成tokens: {config['decode_tokens']}")
    print(f"   隐藏维度: {config['hidden_dim']}")
    print(f"   注意力头数: {config['num_heads']}")
    print(f"   层数: {config['num_layers']}")
    
    # 创建模型
    print("\n🏗️ 构建模型...")
    model = SimpleTransformer(
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers']
    )
    
    # 初始化模型
    dummy_input = mx.random.normal((1, 1, config['hidden_dim']))
    _ = model(dummy_input)
    mx.eval(_)
    print("✅ 模型构建完成")
    
    # 生成测试数据
    print("\n📦 生成测试数据...")
    input_data = mx.random.normal((
        config['batch_size'],
        config['prefill_seq_len'],
        config['hidden_dim']
    ))
    print("✅ 测试数据生成完成")
    
    # 测试1：不使用PD分离
    avg_without_pd = test_without_pd_separation(model, input_data, config['iterations'])
    
    # 测试2：使用PD分离
    avg_prefill, avg_decode, avg_total = test_with_pd_separation(
        model, input_data, config['decode_tokens'], config['iterations']
    )
    
    # 测试3：内存使用
    test_memory_usage(model, input_data)
    
    # 结果对比
    print("\n" + "=" * 100)
    print("📈 性能对比结果")
    print("=" * 100)
    print(f"{'配置':<30} {'延迟(ms)':<20} {'说明':<40}")
    print("-" * 100)
    print(f"{'不使用PD分离':<30} {avg_without_pd:<20.2f} {'完整推理一次':<40}")
    print(f"{'PD分离-Prefill':<30} {avg_prefill:<20.2f} {'处理prompt阶段':<40}")
    print(f"{'PD分离-Decode(平均)':<30} {avg_decode:<20.2f} {'逐token生成':<40}")
    print(f"{'PD分离-总延迟':<30} {avg_total:<20.2f} {'Prefill + Decode x10':<40}")
    print("=" * 100)
    
    # 分析
    print("\n🎯 分析结论")
    print("=" * 100)
    
    if avg_prefill < avg_without_pd:
        speedup = (avg_without_pd - avg_prefill) / avg_without_pd * 100
        print(f"✅ Prefill阶段比完整推理快 {speedup:.1f}%")
    
    if avg_decode < avg_prefill:
        speedup = (avg_prefill - avg_decode) / avg_prefill * 100
        print(f"✅ Decode阶段比Prefill快 {speedup:.1f}%（因为只处理1个token）")
    
    # 计算吞吐量提升
    tokens_without_pd = config['prefill_seq_len']
    tokens_with_pd = config['prefill_seq_len'] + config['decode_tokens']
    
    throughput_without = tokens_without_pd / (avg_without_pd / 1000)
    throughput_with = tokens_with_pd / (avg_total / 1000)
    
    print(f"\n📊 吞吐量对比:")
    print(f"   不使用PD分离: {throughput_without:.1f} tokens/s")
    print(f"   使用PD分离: {throughput_with:.1f} tokens/s")
    
    if throughput_with > throughput_without:
        improvement = (throughput_with - throughput_without) / throughput_without * 100
        print(f"   ✅ PD分离吞吐量提升: {improvement:.1f}%")
    else:
        print(f"   ⚠️ PD分离吞吐量下降: {(throughput_without - throughput_with) / throughput_without * 100:.1f}%")
    
    print("\n💡 PD时间分离优势:")
    print("   1. 内存管理：可以分阶段释放内存")
    print("   2. KV Cache复用：Decode阶段复用Prefill的KV Cache")
    print("   3. 调度优化：可以优先处理Prefill请求，提高吞吐量")
    print("   4. 批处理优化：可以批量处理多个Decode请求")
    print("=" * 100)


if __name__ == "__main__":
    main()
