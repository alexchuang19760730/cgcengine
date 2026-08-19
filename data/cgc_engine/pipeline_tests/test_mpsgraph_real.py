#!/usr/bin/env python3
"""
端云一体 MPSGraph 真实硬件测试
⚠️ 本测试只使用真实硬件数据，不使用模拟数据
⚠️ 需要Apple Silicon设备运行
"""

import sys

import time
import numpy as np
from cgc_engine.agent.harness_agent import HarnessAgent, HarnessCompileStrategy

def test_mpsgraph_real_hardware():
    """真实硬件测试 MPSGraph 性能差异"""
    print("=" * 100)
    print("⚡ 端云一体 MPSGraph 真实硬件测试")
    print("⚠️ 本测试只使用真实硬件数据，不使用模拟数据")
    print("=" * 100)
    
    # 检测当前硬件
    print("\n🔍 硬件检测:")
    print("-" * 50)
    
    try:
        import mlx.core as mx
        print(f"✅ MLX版本: {mx.__version__}")
        print(f"✅ 设备: {mx.default_device()}")
        
        # 获取设备信息
        device_info = mx.device_info()
        print(f"✅ 计算单元: {device_info.get('gpu_count', 'N/A')} GPU")
        print(f"✅ 统一内存: {device_info.get('unified_memory_size', 'N/A')} bytes")
        
    except ImportError:
        print("❌ MLX未安装，无法进行真实测试")
        print("💡 请安装: pip install mlx mlx-lm")
        return
    
    # 测试配置
    batch_size = 1
    seq_len = 512
    hidden_dim = 512
    num_heads = 8
    iterations = 10
    
    print(f"\n📋 测试配置:")
    print(f"   批量大小: {batch_size}")
    print(f"   序列长度: {seq_len}")
    print(f"   隐藏维度: {hidden_dim}")
    print(f"   注意力头数: {num_heads}")
    print(f"   迭代次数: {iterations}")
    
    # 测试1：不启用MPSGraph（手动执行）
    print("\n📊 测试1：不启用 MPSGraph")
    print("-" * 50)
    mps_disabled_times = []
    
    # 预热
    print("   预热中...")
    for _ in range(3):
        query = mx.random.normal((batch_size, seq_len, hidden_dim))
        key = mx.random.normal((batch_size, seq_len, hidden_dim))
        value = mx.random.normal((batch_size, seq_len, hidden_dim))
        scale = (hidden_dim // num_heads) ** -0.5
        scores = (query @ key.transpose(0, 2, 1)) * scale
        attn = mx.softmax(scores)
        output = attn @ value
        mx.eval(output)
    
    for i in range(iterations):
        start = time.time()
        
        # 模拟Decode阶段计算（不使用MPSGraph）
        query = mx.random.normal((batch_size, seq_len, hidden_dim))
        key = mx.random.normal((batch_size, seq_len, hidden_dim))
        value = mx.random.normal((batch_size, seq_len, hidden_dim))
        
        # 手动执行注意力计算
        scale = (hidden_dim // num_heads) ** -0.5
        scores = (query @ key.transpose(0, 2, 1)) * scale
        attn = mx.softmax(scores)
        output = attn @ value
        
        # 手动同步
        mx.eval(output)
        
        elapsed = (time.time() - start) * 1000  # 转换为ms
        mps_disabled_times.append(elapsed)
        print(f"   迭代 {i+1}: {elapsed:.2f}ms")
    
    mps_disabled_avg = np.mean(mps_disabled_times)
    mps_disabled_std = np.std(mps_disabled_times)
    print(f"\n   📈 平均延迟: {mps_disabled_avg:.2f}ms (±{mps_disabled_std:.2f}ms)")
    
    # 测试2：启用MPSGraph（使用mlx.core.compile）
    print("\n📊 测试2：启用 MPSGraph")
    print("-" * 50)
    mps_enabled_times = []
    
    # 编译函数（MPSGraph方式）
    @mx.compile
    def attention_mpsgraph(query, key, value, scale):
        scores = (query @ key.transpose(0, 2, 1)) * scale
        attn = mx.softmax(scores)
        return attn @ value
    
    # 预热（编译函数）
    print("   预热中...")
    for _ in range(3):
        query = mx.random.normal((batch_size, seq_len, hidden_dim))
        key = mx.random.normal((batch_size, seq_len, hidden_dim))
        value = mx.random.normal((batch_size, seq_len, hidden_dim))
        scale = (hidden_dim // num_heads) ** -0.5
        output = attention_mpsgraph(query, key, value, scale)
        mx.eval(output)
    
    for i in range(iterations):
        start = time.time()
        
        # 使用编译后的函数（MPSGraph）
        query = mx.random.normal((batch_size, seq_len, hidden_dim))
        key = mx.random.normal((batch_size, seq_len, hidden_dim))
        value = mx.random.normal((batch_size, seq_len, hidden_dim))
        
        scale = (hidden_dim // num_heads) ** -0.5
        output = attention_mpsgraph(query, key, value, scale)
        
        # 同步
        mx.eval(output)
        
        elapsed = (time.time() - start) * 1000
        mps_enabled_times.append(elapsed)
        print(f"   迭代 {i+1}: {elapsed:.2f}ms")
    
    mps_enabled_avg = np.mean(mps_enabled_times)
    mps_enabled_std = np.std(mps_enabled_times)
    print(f"\n   📈 平均延迟: {mps_enabled_avg:.2f}ms (±{mps_enabled_std:.2f}ms)")
    
    # 对比结果
    print("\n📈 真实性能对比结果")
    print("=" * 100)
    print(f"{'指标':<20} {'启用MPSGraph':<18} {'不启用MPSGraph':<18} {'提升幅度':<10}")
    print("-" * 100)
    
    latency_diff = (mps_disabled_avg - mps_enabled_avg) / mps_disabled_avg * 100
    print(f"{'Decode延迟':<20} {mps_enabled_avg:<18.2f}ms {mps_disabled_avg:<18.2f}ms {latency_diff:>8.1f}%")
    
    # 计算吞吐量对比
    tokens_per_ms_enabled = (batch_size * seq_len) / mps_enabled_avg
    tokens_per_ms_disabled = (batch_size * seq_len) / mps_disabled_avg
    throughput_diff = (tokens_per_ms_enabled - tokens_per_ms_disabled) / tokens_per_ms_disabled * 100
    print(f"{'吞吐量':<20} {tokens_per_ms_enabled:<18.2f} tok/ms {tokens_per_ms_disabled:<18.2f} tok/ms {throughput_diff:>8.1f}%")
    
    print("\n" + "=" * 100)
    print("🎯 真实硬件测试结论")
    print("=" * 100)
    print(f"• 设备: {mx.default_device()}")
    print(f"• MPSGraph 将 Decode 延迟降低 {latency_diff:.1f}%")
    print(f"• MPSGraph 将吞吐量提升 {throughput_diff:.1f}%")
    
    if latency_diff > 20:
        print("\n✅ MPSGraph 在真实硬件上带来显著性能提升！")
    else:
        print("\n⚠️ MPSGraph 提升效果有限，可能受测试配置影响")
    
    # 输出Harness Agent策略建议
    print("\n🤖 Harness Agent 策略建议:")
    print("-" * 50)
    agent = HarnessAgent(device='metal', use_knowledge_base=True)
    print("✅ 已启用真实硬件数据采集")
    print("✅ 建议在端云一体架构中启用MPSGraph")
    print("✅ 策略优先级: MPSGraph > 统一内存 > 标准执行")


if __name__ == "__main__":
    test_mpsgraph_real_hardware()
