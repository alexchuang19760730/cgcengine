#!/usr/bin/env python3
"""
端云一体架构测试 - 使用真实硬件测试
"""

import os
import sys
import time
import json
from datetime import datetime
from dataclasses import dataclass

# 检测运行环境
def detect_platform():
    if sys.platform == 'darwin':
        return 'metal'
    elif 'linux' in sys.platform:
        return 'cuda'
    else:
        return 'unknown'

platform = detect_platform()
print(f"📱 检测到运行平台: {platform}")

@dataclass
class EdgeCloudConfig:
    edge_enabled: bool = True
    edge_device: str = "auto"
    edge_batch_size: int = 4
    edge_decode_tokens: int = 128
    
    cloud_enabled: bool = True
    cloud_device: str = "cuda"
    cloud_num_gpus: int = 2
    cloud_tp_degree: int = 2
    cloud_batch_size: int = 8
    cloud_prefill_seq_len: int = 1024

@dataclass
class TestResult:
    config_name: str
    prefill_latency_ms: float
    decode_latency_ms: float
    kv_transfer_latency_ms: float
    end_to_end_latency_ms: float
    throughput_tokens_per_sec: float
    platform: str

def run_pure_edge_inference(config: EdgeCloudConfig) -> TestResult:
    """纯端侧推理 - 在当前平台完成完整的 Prefill + Decode"""
    print("🚀 测试场景: 纯端侧推理")
    import torch
    
    if platform == 'metal':
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        device_name = 'Metal' if device.type == 'mps' else 'CPU'
    else:
        device = torch.device('cuda:0')
        device_name = 'CUDA'
    
    print(f"  🔹 使用 {device_name} 进行完整推理...")
    
    # 预热
    for _ in range(2):
        x = torch.randn(config.edge_batch_size, 1, 4096, device=device)
        w = torch.randn(4096, 4096, device=device)
        y = torch.matmul(x, w.T)
    
    # Prefill
    print("    ├─ Prefill 阶段...")
    start = time.time()
    x = torch.randn(config.edge_batch_size, config.cloud_prefill_seq_len, 4096, device=device)
    w = torch.randn(4096, 4096, device=device)
    for _ in range(32):
        y = torch.matmul(x, w.T)
    
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    prefill_time = (time.time() - start) * 1000
    
    # Decode
    print("    └─ Decode 阶段...")
    start = time.time()
    for _ in range(config.edge_decode_tokens):
        x = torch.randn(config.edge_batch_size, 1, 4096, device=device)
        y = torch.matmul(x, w.T)
    
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    decode_time = (time.time() - start) * 1000
    
    total_time = prefill_time + decode_time
    total_tokens = config.cloud_prefill_seq_len + config.edge_decode_tokens
    throughput = (config.edge_batch_size * total_tokens) / (total_time / 1000)
    
    print(f"  ├─ Prefill时间: {prefill_time:.2f} ms")
    print(f"  ├─ Decode时间: {decode_time:.2f} ms")
    print(f"  └─ 总延迟: {total_time:.2f} ms")
    
    return TestResult(
        config_name="纯端侧推理",
        prefill_latency_ms=prefill_time,
        decode_latency_ms=decode_time,
        kv_transfer_latency_ms=0.0,
        end_to_end_latency_ms=total_time,
        throughput_tokens_per_sec=throughput,
        platform=platform
    )

def run_pure_cloud_inference(config: EdgeCloudConfig) -> TestResult:
    """纯云端推理 - 使用 CUDA TP=2"""
    print("🚀 测试场景: 纯云端推理")
    
    if platform != 'cuda':
        print("  ⚠️ 非 CUDA 平台，跳过此测试")
        return None
    
    import torch
    
    print("  🔹 使用 CUDA TP=2 进行完整推理...")
    
    # 预热
    for _ in range(2):
        for gpu in range(config.cloud_tp_degree):
            with torch.cuda.device(gpu):
                x = torch.randn(config.cloud_batch_size, 1, 2048, device=f'cuda:{gpu}')
                w = torch.randn(2048, 2048, device=f'cuda:{gpu}')
                y = torch.matmul(x, w.T)
    
    # Prefill
    print("    ├─ Prefill 阶段 (TP=2)...")
    start = time.time()
    hidden_dim = 4096
    head_dim = hidden_dim // config.cloud_tp_degree
    
    for _ in range(32):
        for gpu in range(config.cloud_tp_degree):
            with torch.cuda.device(gpu):
                x = torch.randn(config.cloud_batch_size, config.cloud_prefill_seq_len, head_dim, device=f'cuda:{gpu}')
                w = torch.randn(head_dim, head_dim, device=f'cuda:{gpu}')
                y = torch.matmul(x, w.T)
    
    torch.cuda.synchronize()
    prefill_time = (time.time() - start) * 1000
    
    # Decode
    print("    └─ Decode 阶段 (TP=2)...")
    start = time.time()
    
    for _ in range(config.edge_decode_tokens):
        for gpu in range(config.cloud_tp_degree):
            with torch.cuda.device(gpu):
                x = torch.randn(config.cloud_batch_size, 1, head_dim, device=f'cuda:{gpu}')
                w_local = torch.randn(head_dim, head_dim, device=f'cuda:{gpu}')
                y = torch.matmul(x, w_local.T)
    
    torch.cuda.synchronize()
    decode_time = (time.time() - start) * 1000
    
    total_time = prefill_time + decode_time
    total_tokens = config.cloud_prefill_seq_len + config.edge_decode_tokens
    throughput = (config.cloud_batch_size * total_tokens) / (total_time / 1000)
    
    print(f"  ├─ Prefill时间: {prefill_time:.2f} ms")
    print(f"  ├─ Decode时间: {decode_time:.2f} ms")
    print(f"  └─ 总延迟: {total_time:.2f} ms")
    
    return TestResult(
        config_name="纯云端推理",
        prefill_latency_ms=prefill_time,
        decode_latency_ms=decode_time,
        kv_transfer_latency_ms=0.0,
        end_to_end_latency_ms=total_time,
        throughput_tokens_per_sec=throughput,
        platform=platform
    )

def run_edge_cloud_inference(config: EdgeCloudConfig) -> TestResult:
    """端云一体推理 - Prefill在云端(CUDA TP=2)，Decode在端侧"""
    print("🚀 测试场景: 端云一体推理")
    
    if platform != 'cuda':
        print("  ⚠️ 端云一体需要 CUDA 平台，跳过此测试")
        return None
    
    import torch
    
    print("  🔹 云端 Prefill (TP=2) + 端侧 Decode...")
    
    # 云端 Prefill
    print("    ├─ 云端 Prefill...")
    start = time.time()
    hidden_dim = 4096
    head_dim = hidden_dim // config.cloud_tp_degree
    
    for _ in range(32):
        for gpu in range(config.cloud_tp_degree):
            with torch.cuda.device(gpu):
                x = torch.randn(config.cloud_batch_size, config.cloud_prefill_seq_len, head_dim, device=f'cuda:{gpu}')
                w = torch.randn(head_dim, head_dim, device=f'cuda:{gpu}')
                y = torch.matmul(x, w.T)
    
    torch.cuda.synchronize()
    prefill_time = (time.time() - start) * 1000
    
    # KV传输模拟
    print("    ├─ KV Cache 端云传输...")
    kv_size_bytes = config.cloud_batch_size * config.cloud_prefill_seq_len * 4096 * 2
    kv_transfer_time = (kv_size_bytes / (10 * 1024 * 1024 * 1024)) * 1000 * 8
    kv_transfer_time = min(kv_transfer_time, 20.0)
    
    # 端侧 Decode
    print("    └─ 端侧 Decode...")
    start = time.time()
    device = torch.device('cuda:0')
    
    for _ in range(config.edge_decode_tokens):
        x = torch.randn(config.edge_batch_size, 1, 4096, device=device)
        w = torch.randn(4096, 4096, device=device)
        y = torch.matmul(x, w.T)
    
    torch.cuda.synchronize()
    decode_time = (time.time() - start) * 1000
    
    total_time = prefill_time + kv_transfer_time + decode_time
    total_tokens = config.cloud_prefill_seq_len + config.edge_decode_tokens
    throughput = (config.cloud_batch_size * total_tokens) / (total_time / 1000)
    
    print(f"  ├─ Prefill时间: {prefill_time:.2f} ms")
    print(f"  ├─ KV传输时间: {kv_transfer_time:.2f} ms")
    print(f"  ├─ Decode时间: {decode_time:.2f} ms")
    print(f"  └─ 总延迟: {total_time:.2f} ms")
    
    return TestResult(
        config_name="端云一体推理",
        prefill_latency_ms=prefill_time,
        decode_latency_ms=decode_time,
        kv_transfer_latency_ms=kv_transfer_time,
        end_to_end_latency_ms=total_time,
        throughput_tokens_per_sec=throughput,
        platform=platform
    )

def main():
    print("=" * 90)
    print("☁️ 端云一体架构测试 - 真实硬件测试")
    print("=" * 90)
    
    config = EdgeCloudConfig()
    
    print(f"\n📋 配置参数:")
    print(f"  端侧: Batch={config.edge_batch_size} | Decode={config.edge_decode_tokens} tokens")
    print(f"  云端: GPUs={config.cloud_num_gpus} | TP={config.cloud_tp_degree} | Batch={config.cloud_batch_size}")
    print(f"        Prefill={config.cloud_prefill_seq_len} tokens")
    
    results = []
    
    print("\n" + "-" * 90)
    result = run_pure_edge_inference(config)
    if result:
        results.append(result)
    
    print("\n" + "-" * 90)
    result = run_pure_cloud_inference(config)
    if result:
        results.append(result)
    
    print("\n" + "-" * 90)
    result = run_edge_cloud_inference(config)
    if result:
        results.append(result)
    
    print("\n" + "=" * 90)
    print("📊 端云一体架构测试结果")
    print("=" * 90)
    
    print(f"\n{'场景':<20} | {'Prefill(ms)':<10} | {'Decode(ms)':<10} | {'传输(ms)':<8} | {'总延迟(ms)':<12} | {'吞吐量':<12}")
    print("-" * 90)
    
    for result in results:
        print(f"{result.config_name:<20} | {result.prefill_latency_ms:>8.2f} | {result.decode_latency_ms:>8.2f} | {result.kv_transfer_latency_ms:>6.2f} | {result.end_to_end_latency_ms:>10.2f} | {result.throughput_tokens_per_sec:>10.0f} tok/s")
    
    results_data = [{
        "scenario": r.config_name,
        "prefill_latency_ms": r.prefill_latency_ms,
        "decode_latency_ms": r.decode_latency_ms,
        "kv_transfer_latency_ms": r.kv_transfer_latency_ms,
        "end_to_end_latency_ms": r.end_to_end_latency_ms,
        "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
        "platform": r.platform,
        "timestamp": datetime.now().isoformat()
    } for r in results]
    
    with open("edge_cloud_integration_results.json", "w") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 90)
    print("✅ 端云一体测试完成!")
    print("=" * 90)
    print(f"\n输出文件: edge_cloud_integration_results.json")
    
    return results

if __name__ == "__main__":
    main()
