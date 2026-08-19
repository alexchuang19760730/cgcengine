#!/usr/bin/env python3
"""
性能基准测试：对比传统KV与固定KV方案

测试场景:
1. KV存储性能
2. 注意力计算性能
3. 端云同步性能
"""

import time
import numpy as np
import argparse


def benchmark_kv_storage():
    """KV存储性能基准测试"""
    print("=" * 60)
    print("KV存储性能测试")
    print("=" * 60)
    
    num_heads = 32
    head_dim = 128
    
    # 测试不同序列长度
    seq_lengths = [1024, 4096, 16384, 65536, 131072]
    ortho_dim = 128
    
    results = []
    
    for seq_len in seq_lengths:
        # 传统KV大小
        traditional_kv_size = seq_len * num_heads * head_dim * 2 * 4  # bytes
        
        # 固定KV大小（与序列长度无关）
        fixed_kv_size = num_heads * ortho_dim * head_dim * 2 * 4  # bytes
        
        # 模拟存储时间（假设带宽10GB/s）
        bandwidth_gbs = 10
        traditional_time_ms = (traditional_kv_size / (1024**3)) * bandwidth_gbs * 1000
        fixed_time_ms = (fixed_kv_size / (1024**3)) * bandwidth_gbs * 1000
        
        # 生成模拟数据
        K_traditional = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        V_traditional = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
        
        K_fixed = np.random.randn(num_heads, ortho_dim, head_dim).astype(np.float32)
        V_fixed = np.random.randn(num_heads, ortho_dim, head_dim).astype(np.float32)
        
        # 实际存储时间测量
        start = time.time()
        # 模拟存储操作
        _ = K_traditional.tobytes() + V_traditional.tobytes()
        traditional_store_time_ms = (time.time() - start) * 1000
        
        start = time.time()
        _ = K_fixed.tobytes() + V_fixed.tobytes()
        fixed_store_time_ms = (time.time() - start) * 1000
        
        results.append({
            'seq_len': seq_len,
            'traditional_size_mb': traditional_kv_size / (1024**2),
            'fixed_size_mb': fixed_kv_size / (1024**2),
            'traditional_time_ms': traditional_store_time_ms,
            'fixed_time_ms': fixed_store_time_ms,
        })
        
        print(f"序列长度: {seq_len:,}")
        print(f"  传统KV: {traditional_kv_size / (1024**2):.2f} MB")
        print(f"  固定KV: {fixed_kv_size / (1024**2):.2f} MB")
        print(f"  存储时间: {traditional_store_time_ms:.2f} ms vs {fixed_store_time_ms:.2f} ms")
        print(f"  加速比: {traditional_store_time_ms / max(fixed_store_time_ms, 1e-6):.2f}x")
        print()
    
    return results


def benchmark_attention():
    """注意力计算性能基准测试"""
    print("=" * 60)
    print("注意力计算性能测试")
    print("=" * 60)
    
    num_heads = 32
    head_dim = 128
    ortho_dim = 128
    
    seq_lengths = [4096, 16384, 65536]
    results = []
    
    for seq_len in seq_lengths:
        # 生成数据
        Q = np.random.randn(num_heads, 1, head_dim).astype(np.float32)
        K = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)
        V = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)
        
        # 固定KV数据
        fixed_K = np.random.randn(num_heads, ortho_dim, head_dim).astype(np.float32)
        fixed_V = np.random.randn(num_heads, ortho_dim, head_dim).astype(np.float32)
        
        # 传统注意力
        start = time.time()
        for _ in range(100):
            attn_weights = Q @ K.transpose(0, 2, 1) / np.sqrt(head_dim)
            attn_weights = np.exp(attn_weights) / np.sum(np.exp(attn_weights), axis=-1, keepdims=True)
            output = attn_weights @ V
        traditional_time_ms = (time.time() - start) * 10
        
        # 固定KV注意力
        start = time.time()
        for _ in range(100):
            attn_weights = Q @ fixed_K / np.sqrt(head_dim)
            attn_weights = np.exp(attn_weights) / np.sum(np.exp(attn_weights), axis=-1, keepdims=True)
            output = attn_weights @ fixed_V
        fixed_time_ms = (time.time() - start) * 10
        
        results.append({
            'seq_len': seq_len,
            'traditional_time_ms': traditional_time_ms,
            'fixed_time_ms': fixed_time_ms,
        })
        
        print(f"序列长度: {seq_len:,}")
        print(f"  传统注意力: {traditional_time_ms:.2f} ms")
        print(f"  固定KV注意力: {fixed_time_ms:.2f} ms")
        print(f"  加速比: {traditional_time_ms / max(fixed_time_ms, 1e-6):.2f}x")
        print()
    
    return results


def benchmark_edge_cloud_sync():
    """端云同步性能基准测试"""
    print("=" * 60)
    print("端云同步性能测试")
    print("=" * 60)
    
    num_heads = 32
    head_dim = 128
    ortho_dim = 128
    
    # 不同网络带宽
    bandwidths = [1, 10, 100]  # Gbps
    
    # 传统KV大小（128K序列）
    seq_len = 131072
    traditional_kv_size_bytes = seq_len * num_heads * head_dim * 2 * 4
    
    # 固定KV大小
    fixed_kv_size_bytes = num_heads * ortho_dim * head_dim * 2 * 4
    
    print(f"序列长度: {seq_len:,}")
    print(f"传统KV大小: {traditional_kv_size_bytes / (1024**3):.2f} GB")
    print(f"固定KV大小: {fixed_kv_size_bytes / (1024**2):.2f} MB")
    print()
    
    for bandwidth in bandwidths:
        # 计算同步时间（考虑协议开销）
        overhead_factor = 1.2  # 协议开销
        
        traditional_time_ms = (traditional_kv_size_bytes * 8 / (bandwidth * 10**9)) * 1000 * overhead_factor
        fixed_time_ms = (fixed_kv_size_bytes * 8 / (bandwidth * 10**9)) * 1000 * overhead_factor
        
        print(f"网络带宽: {bandwidth} Gbps")
        print(f"  传统KV同步: {traditional_time_ms:.2f} ms")
        print(f"  固定KV同步: {fixed_time_ms:.2f} ms")
        print(f"  加速比: {traditional_time_ms / max(fixed_time_ms, 1e-6):.2f}x")
        print()


def main():
    parser = argparse.ArgumentParser(description='性能基准测试')
    parser.add_argument('--all', action='store_true', help='运行所有测试')
    parser.add_argument('--kv', action='store_true', help='运行KV存储测试')
    parser.add_argument('--attention', action='store_true', help='运行注意力测试')
    parser.add_argument('--sync', action='store_true', help='运行端云同步测试')
    
    args = parser.parse_args()
    
    if args.all or args.kv:
        benchmark_kv_storage()
    
    if args.all or args.attention:
        benchmark_attention()
    
    if args.all or args.sync:
        benchmark_edge_cloud_sync()
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    main()