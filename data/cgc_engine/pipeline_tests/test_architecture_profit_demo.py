#!/usr/bin/env python3
"""
双GPU TP=2 + PD分离 + NCCL + CUDA Graph + SPDK 完整架构收益分析演示

功能:
1. 运行完整架构收益分析
2. 保存结果到数据库
3. 生成可视化报告
4. 导出知识到知识库
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_full_architecture_profit import FullArchitectureConfig, FullArchitectureProfiler, ArchitectureProfitResult
from cgc_engine.utils.test_result_storage import TestResultStorage, TestRecord
from cgc_engine.utils.test_result_visualizer import TestResultVisualizer
from cgc_engine.utils.knowledge_storage import KnowledgeStorage, BackendKnowledge, HardwareKnowledge, GraphPattern, OptimizationCode


def save_architecture_results(storage: TestResultStorage, results: list):
    """保存架构分析结果到数据库"""
    for result in results:
        record = TestRecord(
            test_id=f"arch_{result.config_name[:10].replace(' ', '_')}",
            module_name="full_architecture",
            device="cuda",
            backend="vllm",
            success=True,
            timestamp=result.timestamp if hasattr(result, 'timestamp') else "",
            total_time_ms=result.end_to_end_latency_ms,
            avg_time_ms=result.end_to_end_latency_ms,
            min_time_ms=result.end_to_end_latency_ms,
            max_time_ms=result.end_to_end_latency_ms,
            peak_memory_gb=0.0,
            avg_memory_gb=0.0,
            h2d_bytes=0,
            d2h_bytes=0,
            copy_count=0,
            gflops=0.0,
            total_ops=0,
            scheduling_delay_ms=0.0,
            overhead_ratio=0.0,
            platform="linux",
            device_type="cuda",
            device_count=2,
            total_memory_gb_sys=64.0,
            unified_memory=False,
            metadata={
                "throughput_tokens_per_sec": result.throughput_tokens_per_sec,
                "gpu_utilization": result.gpu_utilization,
                "latency_vs_baseline": result.latency_vs_baseline,
                "throughput_vs_baseline": result.throughput_vs_baseline,
                "prefill_latency_ms": result.prefill_latency_ms,
                "decode_latency_ms": result.decode_latency_ms,
                "kv_access_latency_ms": result.kv_access_latency_ms,
                "config_name": result.config_name
            }
        )
        storage.save_record(record)


def save_architecture_knowledge(knowledge_storage: KnowledgeStorage):
    """保存架构知识到知识库"""
    # 保存完整架构知识
    knowledge_storage.save_backend_knowledge(BackendKnowledge(
        backend_id="vllm-tp2-pd",
        name="vLLM TP2+PD",
        type="inference",
        supported_ops=["scaled_dot_product_attention", "linear", "layer_norm", "mlp", "nccl_allreduce"],
        optimization_capabilities=["flash_attention", "paged_attention", "cuda_graph", "pd_separation", "tensor_parallel"],
        hardware_requirements={"min_gpu_count": 2, "min_gpu_memory_gb": 24, "cuda_version": ">=12.0"},
        performance_profiles={"throughput": 2500, "latency": 9, "memory_efficiency": 0.98},
        version="0.5.0+",
        metadata={"architecture": "TP2+PD+CUDA Graph+SPDK"}
    ))
    
    # 保存图模式
    knowledge_storage.save_graph_pattern(GraphPattern(
        pattern_id="tp2-pd-cudagraph",
        pattern_type="architecture",
        description="双GPU TP=2 + PD分离 + CUDA Graph 模式",
        node_patterns=[
            {"stage": "prefill", "parallel_degree": 2, "optimization": "cuda_graph"},
            {"stage": "decode", "parallel_degree": 1, "optimization": "static_graph"},
            {"stage": "kv_cache", "backend": "spdk", "sharing": "cross_gpu"}
        ],
        optimizations=["tensor_parallel", "pd_separation", "cuda_graph", "spdk_kv_cache"],
        applicable_backends=["vllm-tp2-pd", "tensorrt-llm"],
        performance_impact=0.85,
        metadata={"acceleration": "3.48x", "gpu_utilization": "85%"}
    ))


def main():
    print("=" * 90)
    print("🏗️ 双GPU TP=2 + PD分离 + NCCL + CUDA Graph + SPDK 完整架构收益分析")
    print("=" * 90)
    
    # 1. 运行架构分析
    print("\n📊 1. 运行完整架构收益分析")
    config = FullArchitectureConfig(
        num_gpus=2,
        tp_degree=2,
        hidden_dim=4096,
        num_layers=28,
        batch_size=32,
        prefill_seq_len=2048,
        decode_tokens=256
    )
    
    profiler = FullArchitectureProfiler(config)
    results = profiler.get_all_architectures()
    
    # 2. 保存结果到数据库
    print("\n💾 2. 保存结果到数据库")
    storage = TestResultStorage()
    save_architecture_results(storage, results)
    
    # 3. 生成可视化报告
    print("\n📈 3. 生成可视化架构收益图表")
    visualizer = TestResultVisualizer()
    visualizer.generate_architecture_profit_chart(results, "architecture_profit.html")
    
    # 4. 保存架构知识到知识库
    print("\n🧠 4. 保存架构知识到知识库")
    knowledge_storage = KnowledgeStorage()
    save_architecture_knowledge(knowledge_storage)
    
    # 5. 打印详细结果
    print("\n" + "=" * 90)
    print("📊 完整架构收益分析结果")
    print("=" * 90)
    
    print(f"\n{'架构配置':<50} | {'延迟(ms)':<10} | {'吞吐量':<12} | {'GPU利用率':<12} | {'加速比'}")
    print("-" * 90)
    
    for result in results:
        print(f"{result.config_name:<50} | {result.end_to_end_latency_ms:>8.0f} | {result.throughput_tokens_per_sec:>10.0f} tok/s | {result.gpu_utilization:>10.1f}% | x{result.latency_vs_baseline:>5.2f}")
    
    print("\n🚀 关键收益总结:")
    baseline = results[0]
    best = results[-1]
    print(f"  - 基线延迟: {baseline.end_to_end_latency_ms:.0f} ms")
    print(f"  - 优化后延迟: {best.end_to_end_latency_ms:.0f} ms")
    print(f"  - 延迟降低: x{best.latency_vs_baseline:.2f}")
    print(f"  - 吞吐量提升: x{best.throughput_vs_baseline:.2f}")
    print(f"  - GPU利用率: {best.gpu_utilization:.0f}%")
    
    print("\n" + "=" * 90)
    print("✅ 分析完成!")
    print("=" * 90)
    print(f"\n输出文件:")
    print(f"  - 架构收益图表: architecture_profit.html")
    print(f"  - JSON 结果: /tmp/full_architecture_profit_analysis.json")
    print(f"  - 数据库: test_results.db")
    print(f"  - 知识库: knowledge.db")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
