#!/usr/bin/env python3
"""
R-SWA vs Baseline 对比测试脚本

测试指标:
1. 显存占用
2. 推理延迟
3. 吞吐量
4. 上下文长度扩展性
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgc_engine.rswa_integration import CGCUnlimitedRSWAAttention, RSWAPrefillPoolEngine
from cgc_engine.gds_service.cufile_wrapper import is_gds_available


class BaselineAttention(nn.Module):
    """
    基线注意力机制（普通 Full Attention）
    
    作为 R-SWA 的对比基准
    """
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        print(f"[Baseline] 初始化完成 (dim={dim}, heads={num_heads})")
    
    def forward(self, x: torch.Tensor, past_kv: Tuple = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Args:
            x: (B, T, C)
            past_kv: 可选的历史 KV
        
        Returns:
            out: (B, T, C)
            new_kv: 新的 KV
        """
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 拼接历史 KV
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Full Attention（无滑动窗口限制）
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(attn)
        
        return out, (k, v)


def measure_memory() -> Dict[str, float]:
    """测量当前显存使用"""
    if not torch.cuda.is_available():
        return {
            "used": 0.0,
            "free": 0.0,
            "total": 0.0,
        }
    
    return {
        "used": torch.cuda.memory_allocated(0) / (1024 ** 3),  # GB
        "free": torch.cuda.memory_reserved(0) / (1024 ** 3),  # GB
        "total": torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),  # GB
    }


def measure_latency(func, *args, **kwargs) -> Tuple[float, any]:
    """测量函数执行延迟"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start) * 1000, result  # ms


def test_memory_scalability(max_seq_len: int = 16384, step: int = 1024):
    """
    测试显存随序列长度的扩展性
    
    对比 Baseline vs R-SWA
    """
    print("=" * 70)
    print("测试 1: 显存扩展性")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ 未检测到 CUDA，跳过显存测试")
        return
    
    device = torch.device("cuda:0")
    dim = 4096
    num_heads = 32
    batch_size = 1
    
    results = []
    
    # Baseline 测试
    print("\n--- Baseline (Full Attention) ---")
    baseline = BaselineAttention(dim, num_heads).to(device)
    
    for seq_len in range(step, max_seq_len + 1, step):
        torch.cuda.empty_cache()
        
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.bfloat16)
        
        try:
            _, _ = baseline(x)
            memory = measure_memory()
            print(f"  seq_len={seq_len:6d} | used={memory['used']:.2f} GB")
            results.append({
                "seq_len": seq_len,
                "type": "baseline",
                "memory_used_gb": memory["used"],
                "success": True,
            })
        except RuntimeError as e:
            print(f"  seq_len={seq_len:6d} | ❌ OOM")
            results.append({
                "seq_len": seq_len,
                "type": "baseline",
                "memory_used_gb": float("inf"),
                "success": False,
            })
            break
    
    # R-SWA 测试
    print("\n--- R-SWA (Unlimited) ---")
    torch.cuda.empty_cache()
    rswa = CGCUnlimitedRSWAAttention(dim, num_heads, window_size=128).to(device)
    
    # 添加参考块（模拟超长上下文）
    for ref_len in [4096, 8192]:
        ref_k = torch.randn(1, num_heads, ref_len, dim // num_heads, 
                           device=device, dtype=torch.bfloat16)
        ref_v = torch.randn(1, num_heads, ref_len, dim // num_heads, 
                           device=device, dtype=torch.bfloat16)
        token_ids = torch.arange(ref_len, device=device)
        rswa.add_reference_chunk(token_ids, ref_k, ref_v)
    
    for seq_len in range(step, max_seq_len + 1, step):
        torch.cuda.empty_cache()
        
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.bfloat16)
        
        try:
            _, _, _ = rswa(x, use_reference=True)
            memory = measure_memory()
            pool_info = rswa.get_pool_info()
            print(f"  seq_len={seq_len:6d} | used={memory['used']:.2f} GB | hot_chunks={pool_info['hot_chunks']}")
            results.append({
                "seq_len": seq_len,
                "type": "rswa",
                "memory_used_gb": memory["used"],
                "hot_chunks": pool_info["hot_chunks"],
                "success": True,
            })
        except RuntimeError as e:
            print(f"  seq_len={seq_len:6d} | ❌ OOM")
            results.append({
                "seq_len": seq_len,
                "type": "rswa",
                "memory_used_gb": float("inf"),
                "success": False,
            })
            break
    
    # 清理
    del baseline, rswa
    torch.cuda.empty_cache()
    
    return results


def test_latency_throughput(num_runs: int = 100, seq_len: int = 2048):
    """
    测试延迟和吞吐量
    
    对比 Baseline vs R-SWA
    """
    print("\n" + "=" * 70)
    print("测试 2: 延迟与吞吐量")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ 未检测到 CUDA，跳过延迟测试")
        return
    
    device = torch.device("cuda:0")
    dim = 4096
    num_heads = 32
    batch_size = 1
    
    # 初始化
    baseline = BaselineAttention(dim, num_heads).to(device)
    rswa = CGCUnlimitedRSWAAttention(dim, num_heads, window_size=128).to(device)
    
    # R-SWA 添加参考块
    ref_len = 8192
    ref_k = torch.randn(1, num_heads, ref_len, dim // num_heads, 
                       device=device, dtype=torch.bfloat16)
    ref_v = torch.randn(1, num_heads, ref_len, dim // num_heads, 
                       device=device, dtype=torch.bfloat16)
    token_ids = torch.arange(ref_len, device=device)
    rswa.add_reference_chunk(token_ids, ref_k, ref_v)
    
    # 准备输入
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.bfloat16)
    
    # Warm-up
    print("\nWarm-up...")
    for _ in range(5):
        baseline(x)
        rswa(x, use_reference=True)
    
    torch.cuda.synchronize()
    
    # Baseline 测试
    print("\n--- Baseline (Full Attention) ---")
    baseline_latencies = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        baseline(x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        baseline_latencies.append((end - start) * 1000)  # ms
    
    # R-SWA 测试
    print("\n--- R-SWA (with Reference) ---")
    rswa_latencies = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        rswa(x, use_reference=True)
        torch.cuda.synchronize()
        end = time.perf_counter()
        rswa_latencies.append((end - start) * 1000)  # ms
    
    # 计算统计
    results = {
        "baseline": {
            "mean_latency_ms": sum(baseline_latencies) / len(baseline_latencies),
            "min_latency_ms": min(baseline_latencies),
            "max_latency_ms": max(baseline_latencies),
            "throughput_tokens_per_sec": (seq_len * num_runs) / (sum(baseline_latencies) / 1000),
        },
        "rswa": {
            "mean_latency_ms": sum(rswa_latencies) / len(rswa_latencies),
            "min_latency_ms": min(rswa_latencies),
            "max_latency_ms": max(rswa_latencies),
            "throughput_tokens_per_sec": (seq_len * num_runs) / (sum(rswa_latencies) / 1000),
            "reference_chunks": 1,
            "reference_tokens": ref_len,
        },
        "num_runs": num_runs,
        "seq_len": seq_len,
        "gds_enabled": is_gds_available(),
    }
    
    # 输出结果
    print("\n📊 测试结果:")
    print(f"{'指标':<20} {'Baseline':<15} {'R-SWA':<15}")
    print(f"{'平均延迟 (ms)':<20} {results['baseline']['mean_latency_ms']:<15.2f} {results['rswa']['mean_latency_ms']:<15.2f}")
    print(f"{'最小延迟 (ms)':<20} {results['baseline']['min_latency_ms']:<15.2f} {results['rswa']['min_latency_ms']:<15.2f}")
    print(f"{'最大延迟 (ms)':<20} {results['baseline']['max_latency_ms']:<15.2f} {results['rswa']['max_latency_ms']:<15.2f}")
    print(f"{'吞吐量 (tokens/s)':<20} {results['baseline']['throughput_tokens_per_sec']:<15.1f} {results['rswa']['throughput_tokens_per_sec']:<15.1f}")
    
    # 清理
    del baseline, rswa
    torch.cuda.empty_cache()
    
    return results


def test_inference_quality():
    """
    测试推理质量（模拟）
    
    验证 R-SWA 是否能正确使用参考上下文
    """
    print("\n" + "=" * 70)
    print("测试 3: 推理质量验证")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ 未检测到 CUDA，跳过质量测试")
        return
    
    # 初始化引擎
    engine = RSWAPrefillPoolEngine(
        dim=4096,
        num_heads=32,
        window_size=128,
        max_hot_chunks=4,
        chunk_size=4096,
    )
    
    # 添加参考知识
    reference_texts = [
        "法律知识: 《中华人民共和国刑法》第三百零二条规定：盗窃、侮辱、故意毁坏尸体、尸骨、骨灰的，处三年以下有期徒刑、拘役或者管制。",
        "技术知识: GPU Direct Storage (GDS) 是 NVIDIA 公司开发的一项技术，允许 GPU 直接从存储设备读取数据，无需经过 CPU 内存中转，从而大幅提高数据传输效率。",
        "历史知识: 唐朝（618年—907年）是中国历史上继隋朝之后的大一统中原王朝，共历二十一帝，享国二百八十九年，是中国历史上公认的强盛时代之一。",
    ]
    
    print("\n添加参考知识...")
    chunk_ids = engine.prefill_reference(reference_texts)
    print(f"已添加 {len(chunk_ids)} 个参考块")
    
    # 测试查询
    test_cases = [
        "什么是 GDS 技术？",
        "盗窃尸体的法律后果是什么？",
        "唐朝存在了多少年？",
    ]
    
    print("\n测试查询:")
    for query in test_cases:
        print(f"\nQ: {query}")
        response = engine.infer(query, max_tokens=50)
        print(f"A: {response[:100]}...")
    
    # 获取引擎状态
    info = engine.info()
    print(f"\n引擎状态: {json.dumps(info, indent=2, ensure_ascii=False)}")
    
    # 清理
    engine.attention.clear_pool()
    
    return {
        "test_cases": len(test_cases),
        "reference_chunks": len(chunk_ids),
        "gds_enabled": info["gds_enabled"],
    }


def main():
    print("=" * 70)
    print("R-SWA vs Baseline 对比测试套件")
    print("=" * 70)
    print(f"日期: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"GDS 可用: {is_gds_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    all_results = {
        "metadata": {
            "date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "cuda_available": torch.cuda.is_available(),
            "gds_available": is_gds_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
        "tests": {},
    }
    
    # 运行所有测试
    try:
        # 测试 1: 显存扩展性
        mem_results = test_memory_scalability(max_seq_len=16384, step=1024)
        all_results["tests"]["memory_scalability"] = mem_results
        
        # 测试 2: 延迟与吞吐量
        latency_results = test_latency_throughput(num_runs=100, seq_len=2048)
        all_results["tests"]["latency_throughput"] = latency_results
        
        # 测试 3: 推理质量
        quality_results = test_inference_quality()
        all_results["tests"]["inference_quality"] = quality_results
        
        # 保存结果
        output_path = "rswa_vs_baseline_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 所有测试完成!")
        print(f"结果已保存到: {output_path}")
        
        # 生成总结报告
        generate_report(all_results)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def generate_report(results: Dict):
    """生成测试报告摘要"""
    print("\n" + "=" * 70)
    print("📋 测试报告摘要")
    print("=" * 70)
    
    # 显存测试总结
    mem_tests = results["tests"].get("memory_scalability", [])
    if mem_tests:
        baseline_max_seq = max(
            t["seq_len"] for t in mem_tests 
            if t["type"] == "baseline" and t["success"]
        )
        rswa_max_seq = max(
            t["seq_len"] for t in mem_tests 
            if t["type"] == "rswa" and t["success"]
        )
        
        print(f"\n📈 显存扩展性:")
        print(f"  Baseline 最大序列长度: {baseline_max_seq:,} tokens")
        print(f"  R-SWA 最大序列长度: {rswa_max_seq:,} tokens")
        print(f"  提升倍数: {rswa_max_seq / baseline_max_seq:.1f}x")
    
    # 延迟测试总结
    latency = results["tests"].get("latency_throughput", {})
    if latency:
        base_lat = latency["baseline"]["mean_latency_ms"]
        rswa_lat = latency["rswa"]["mean_latency_ms"]
        base_throughput = latency["baseline"]["throughput_tokens_per_sec"]
        rswa_throughput = latency["rswa"]["throughput_tokens_per_sec"]
        
        print(f"\n⚡ 延迟与吞吐量:")
        print(f"  Baseline 延迟: {base_lat:.2f} ms")
        print(f"  R-SWA 延迟: {rswa_lat:.2f} ms")
        print(f"  延迟变化: {'+' if rswa_lat > base_lat else '-'}{abs(rswa_lat - base_lat):.2f} ms ({(rswa_lat/base_lat-1)*100:.1f}%)")
        print(f"  Baseline 吞吐量: {base_throughput:.1f} tokens/s")
        print(f"  R-SWA 吞吐量: {rswa_throughput:.1f} tokens/s")
        print(f"  吞吐量变化: {'+' if rswa_throughput > base_throughput else '-'}{abs(rswa_throughput/base_throughput-1)*100:.1f}%")
    
    # 质量测试总结
    quality = results["tests"].get("inference_quality", {})
    if quality:
        print(f"\n🧠 推理质量:")
        print(f"  测试用例数: {quality['test_cases']}")
        print(f"  参考块数: {quality['reference_chunks']}")
        print(f"  GDS 启用: {'✅' if quality['gds_enabled'] else '❌'}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    sys.exit(main())
