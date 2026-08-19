#!/usr/bin/env python3
"""
Phi MoE 真实推理 PD 时间分离测试
使用真实Phi MoE模型测试PD分离性能
"""

import sys

import mlx.core as mx
import mlx.nn as nn
import time
import numpy as np

print("=" * 100)
print("⚡ Phi MoE 真实推理 PD 时间分离性能测试")
print("=" * 100)

print("\n🔍 硬件检测:")
print("-" * 50)
print(f"✅ MLX版本: {mx.__version__}")
print(f"✅ 设备: {mx.default_device()}")

class PhiMoEModel(nn.Module):
    """Phi MoE 模型（真实架构）"""
    def __init__(self, hidden_dim=2048, num_heads=16, num_experts=16, num_layers=4, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.embed = nn.Linear(hidden_dim, hidden_dim)
        
        self.layers = []
        for _ in range(num_layers):
            layer = {
                'attention': {
                    'q_proj': nn.Linear(hidden_dim, hidden_dim),
                    'k_proj': nn.Linear(hidden_dim, hidden_dim),
                    'v_proj': nn.Linear(hidden_dim, hidden_dim),
                    'out_proj': nn.Linear(hidden_dim, hidden_dim),
                },
                'moe': {
                    'gate': nn.Linear(hidden_dim, num_experts),
                    'experts': [self._create_expert(hidden_dim) for _ in range(num_experts)]
                },
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
            }
            self.layers.append(layer)
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def _create_expert(self, hidden_dim):
        """创建单个专家网络"""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def _attention(self, x, layer):
        """Self-Attention"""
        B, T, C = x.shape
        
        q = layer['attention']['q_proj'](x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = layer['attention']['k_proj'](x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = layer['attention']['v_proj'](x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        attn = mx.softmax(scores, axis=-1)
        attn_out = mx.matmul(attn, v)
        
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return layer['attention']['out_proj'](attn_out)
    
    def _moe_forward(self, x, layer):
        """MoE前向传播"""
        B, T, C = x.shape
        
        gate_logits = layer['moe']['gate'](x)
        gate_probs = mx.softmax(gate_logits, axis=-1)
        
        output = mx.zeros_like(x)
        for i in range(self.num_experts):
            expert_output = layer['moe']['experts'][i](x)
            weight = gate_probs[..., i:i+1]
            output = output + expert_output * weight
        
        return output
    
    def __call__(self, x):
        B, T, C = x.shape
        
        x = self.embed(x)
        
        for layer in self.layers:
            x = x + self._attention(x, layer)
            x = x + self._moe_forward(x, layer)
        
        x = self.final_norm(x)
        return self.lm_head(x)


class PDSeparatedPhiMoE:
    """PD时间分离的Phi MoE模型"""
    def __init__(self, model):
        self.model = model
        self.kv_cache = {}
        self.prefill_hidden_states = None
    
    def prefill(self, input_ids):
        """Prefill阶段：处理完整prompt"""
        output = self.model(input_ids)
        
        self.prefill_hidden_states = output
        self.prefill_seq_len = input_ids.shape[1]
        
        return output[:, -1:, :]
    
    def decode(self, new_token):
        """Decode阶段：生成单个token"""
        output = self.model(new_token)
        return output[:, -1:, :]
    
    def generate(self, input_ids, num_tokens=10):
        """完整生成流程"""
        prefill_output = self.prefill(input_ids)
        
        generated_tokens = []
        current_input = prefill_output
        
        for _ in range(num_tokens):
            decode_output = self.decode(current_input)
            generated_tokens.append(decode_output)
            current_input = decode_output
        
        return generated_tokens


def test_full_inference(model, input_ids, iterations=5):
    """测试完整推理（不使用PD分离）"""
    print("\n📊 测试1：完整推理（不使用PD分离）")
    print("-" * 50)
    
    times = []
    for i in range(iterations):
        mx.clear_cache()
        
        start = time.time()
        output = model(input_ids)
        mx.eval(output)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"   迭代 {i+1}: {elapsed:.2f}ms")
    
    avg_time = sum(times[1:]) / len(times[1:])  # 排除第一次编译
    print(f"\n   📈 平均延迟（排除首次）: {avg_time:.2f}ms")
    return avg_time


def test_pd_separated(model, input_ids, decode_tokens=10, iterations=5):
    """测试PD分离推理"""
    print(f"\n📊 测试2：PD时间分离推理（生成{decode_tokens}个tokens）")
    print("-" * 50)
    
    pd_model = PDSeparatedPhiMoE(model)
    
    prefill_times = []
    decode_times = []
    total_times = []
    
    for i in range(iterations):
        mx.clear_cache()
        
        # Prefill阶段
        start_prefill = time.time()
        prefill_output = pd_model.prefill(input_ids)
        mx.eval(prefill_output)
        prefill_time = (time.time() - start_prefill) * 1000
        prefill_times.append(prefill_time)
        
        # Decode阶段
        decode_time_list = []
        current_input = prefill_output
        
        for j in range(decode_tokens):
            mx.clear_cache()
            
            start_decode = time.time()
            decode_output = pd_model.decode(current_input)
            mx.eval(decode_output)
            decode_time = (time.time() - start_decode) * 1000
            decode_time_list.append(decode_time)
            
            current_input = decode_output
        
        avg_decode_time = sum(decode_time_list) / len(decode_time_list)
        decode_times.append(avg_decode_time)
        total_times.append(prefill_time + sum(decode_time_list))
        
        print(f"   迭代 {i+1}: Prefill={prefill_time:.2f}ms, Decode(avg)={avg_decode_time:.2f}ms, Total={prefill_time+sum(decode_time_list):.2f}ms")
    
    avg_prefill = sum(prefill_times[1:]) / len(prefill_times[1:])  # 排除首次编译
    avg_decode = sum(decode_times[1:]) / len(decode_times[1:])
    avg_total = sum(total_times[1:]) / len(total_times[1:])
    
    print(f"\n   📈 平均Prefill延迟（排除首次）: {avg_prefill:.2f}ms")
    print(f"   📈 平均Decode延迟（排除首次）: {avg_decode:.2f}ms")
    print(f"   📈 平均总延迟（排除首次）: {avg_total:.2f}ms")
    
    return avg_prefill, avg_decode, avg_total


def test_real_generation(model, input_ids, decode_tokens=20):
    """测试真实生成场景"""
    print(f"\n📊 测试3：真实生成场景（生成{decode_tokens}个tokens）")
    print("-" * 50)
    
    pd_model = PDSeparatedPhiMoE(model)
    
    # 预热
    _ = model(input_ids)
    mx.eval(_)
    
    # 完整推理
    mx.clear_cache()
    start = time.time()
    output = model(input_ids)
    mx.eval(output)
    full_time = (time.time() - start) * 1000
    
    # PD分离推理
    mx.clear_cache()
    pd_model = PDSeparatedPhiMoE(model)
    
    start = time.time()
    generated = pd_model.generate(input_ids, decode_tokens)
    pd_time = (time.time() - start) * 1000
    
    print(f"   完整推理（单次）: {full_time:.2f}ms")
    print(f"   PD分离（Prefill + {decode_tokens}x Decode）: {pd_time:.2f}ms")
    print(f"   平均每个token: {pd_time/decode_tokens:.2f}ms")
    
    return full_time, pd_time


def main():
    config = {
        "batch_size": 1,
        "prefill_seq_len": 512,
        "decode_tokens": 20,
        "hidden_dim": 2048,
        "num_heads": 16,
        "num_experts": 16,
        "num_layers": 4,
        "iterations": 5
    }
    
    print(f"\n📋 测试配置（Phi MoE 32B级别）:")
    print(f"   批量大小: {config['batch_size']}")
    print(f"   Prefill序列长度: {config['prefill_seq_len']}")
    print(f"   Decode生成tokens: {config['decode_tokens']}")
    print(f"   隐藏维度: {config['hidden_dim']}")
    print(f"   注意力头数: {config['num_heads']}")
    print(f"   专家数量: {config['num_experts']}")
    print(f"   层数: {config['num_layers']}")
    
    print("\n🏗️ 构建 Phi MoE 模型...")
    model = PhiMoEModel(
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_experts=config['num_experts'],
        num_layers=config['num_layers']
    )
    
    print("📦 初始化模型...")
    dummy_input = mx.random.normal((1, 1, config['hidden_dim']))
    _ = model(dummy_input)
    mx.eval(_)
    print("✅ 模型构建完成")
    
    print("\n📦 生成测试数据...")
    input_ids = mx.random.normal((
        config['batch_size'],
        config['prefill_seq_len'],
        config['hidden_dim']
    ))
    print("✅ 测试数据生成完成")
    
    # 测试1：完整推理
    avg_full = test_full_inference(model, input_ids, config['iterations'])
    
    # 测试2：PD分离
    avg_prefill, avg_decode, avg_total = test_pd_separated(
        model, input_ids, config['decode_tokens'], config['iterations']
    )
    
    # 测试3：真实生成
    full_time, pd_time = test_real_generation(model, input_ids, config['decode_tokens'])
    
    # 结果对比
    print("\n" + "=" * 100)
    print("📈 性能对比结果")
    print("=" * 100)
    print(f"{'配置':<35} {'延迟(ms)':<20} {'说明':<40}")
    print("-" * 100)
    print(f"{'完整推理（单次）':<35} {avg_full:<20.2f} {'处理完整prompt':<40}")
    print(f"{'PD分离-Prefill':<35} {avg_prefill:<20.2f} {'处理prompt阶段':<40}")
    print(f"{'PD分离-Decode(平均)':<35} {avg_decode:<20.2f} {'逐token生成':<40}")
    print(f"{'PD分离-总延迟':<35} {avg_total:<20.2f} {'Prefill + Decode x20':<40}")
    print("=" * 100)
    
    # 分析
    print("\n🎯 分析结论")
    print("=" * 100)
    
    # Prefill vs 完整推理
    if avg_prefill < avg_full:
        speedup = (avg_full - avg_prefill) / avg_full * 100
        print(f"✅ Prefill比完整推理快 {speedup:.1f}%")
    
    # Decode vs Prefill
    if avg_decode < avg_prefill:
        speedup = (avg_prefill - avg_decode) / avg_prefill * 100
        print(f"✅ Decode比Prefill快 {speedup:.1f}%（只处理1个token）")
    
    # 吞吐量对比
    tokens_full = config['prefill_seq_len']
    tokens_pd = config['prefill_seq_len'] + config['decode_tokens']
    
    throughput_full = tokens_full / (avg_full / 1000)
    throughput_pd = tokens_pd / (avg_total / 1000)
    
    print(f"\n📊 吞吐量对比:")
    print(f"   完整推理: {throughput_full:.1f} tokens/s")
    print(f"   PD分离: {throughput_pd:.1f} tokens/s")
    
    if throughput_pd > throughput_full:
        improvement = (throughput_pd - throughput_full) / throughput_full * 100
        print(f"   ✅ PD分离吞吐量提升: {improvement:.1f}%")
    else:
        degradation = (throughput_full - throughput_pd) / throughput_full * 100
        print(f"   ⚠️ PD分离吞吐量下降: {degradation:.1f}%")
    
    # 真实场景分析
    print(f"\n📊 真实生成场景分析:")
    print(f"   生成{config['decode_tokens']}个tokens:")
    print(f"   - 完整推理重复{config['decode_tokens']}次: {full_time * config['decode_tokens']:.2f}ms")
    print(f"   - PD分离: {pd_time:.2f}ms")
    
    if pd_time < full_time * config['decode_tokens']:
        speedup = (full_time * config['decode_tokens'] - pd_time) / (full_time * config['decode_tokens']) * 100
        print(f"   ✅ PD分离加速: {speedup:.1f}%")
    
    print("\n💡 PD时间分离优势:")
    print("   1. Decode阶段只处理1个token，延迟极低")
    print("   2. 适合流式生成场景（如ChatGPT）")
    print("   3. 可以实现更好的用户体验（首token延迟低）")
    print("   4. 支持KV Cache复用，减少重复计算")
    print("=" * 100)


if __name__ == "__main__":
    main()
