# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
图拓扑分析测试脚本 - 参考 Harness Agent 测试模板

功能:
✅ 节点分析
✅ 依赖分析  
✅ 拓扑排序
✅ 关键路径识别
✅ 并行分析
✅ 图统计信息

使用方式:
    python test_graph_topology_analysis.py
"""

import sys
import os
from typing import Dict, Any, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test.harness_module_test_template import ModuleTestTemplate, TestResult


def test_graph_topology_analysis():
    """
    图拓扑分析测试函数
    
    测试内容:
    1. 创建简单模型并获取 FX 图
    2. 使用 GraphTopologyAnalyzer 分析图
    3. 验证所有分析功能
    """
    import torch
    import torch.nn as nn
    
    # 创建测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 128)
            self.linear2 = nn.Linear(128, 256)
            self.norm = nn.LayerNorm(128)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.norm(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    model = TestModel()
    traced_graph = torch.fx.symbolic_trace(model).graph
    
    # 导入分析器
    from cgc_engine.analysis import GraphTopologyAnalyzer
    
    analyzer = GraphTopologyAnalyzer()
    result = analyzer.analyze(traced_graph)
    
    # 打印分析结果
    analyzer.print_summary(result)
    
    # 返回分析结果作为测试指标
    return {
        "num_nodes": result.statistics.num_nodes,
        "num_edges": result.statistics.num_edges,
        "depth": result.statistics.depth,
        "width": result.statistics.width,
        "num_compute_nodes": result.statistics.num_compute_nodes,
        "num_memory_nodes": result.statistics.num_memory_nodes,
        "critical_path_length": len(result.critical_path.nodes),
        "critical_path_time_ms": result.critical_path.total_time_ms,
        "num_parallel_groups": len(result.parallel_groups),
        "graph_type": result.graph_type,
    }


def test_transformer_topology():
    """
    Transformer 图拓扑分析测试
    
    测试内容:
    1. 创建简单 Transformer 模型
    2. 分析 Attention 结构
    """
    import torch
    import torch.nn as nn
    
    class SimpleAttention(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
        
        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            
            # Scaled Dot-Product Attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn, v)
            
            output = self.out_proj(output)
            return output
    
    model = SimpleAttention(512)
    traced_graph = torch.fx.symbolic_trace(model).graph
    
    from cgc_engine.analysis import GraphTopologyAnalyzer
    analyzer = GraphTopologyAnalyzer()
    result = analyzer.analyze(traced_graph)
    
    analyzer.print_summary(result)
    
    return {
        "num_nodes": result.statistics.num_nodes,
        "graph_type": result.graph_type,
        "critical_path_percentage": result.critical_path.percentage_of_total,
    }


def main():
    """
    运行所有图拓扑分析测试
    
    使用 Harness Agent 测试模板格式
    """
    print("=" * 70)
    print("图拓扑分析测试")
    print("=" * 70)
    
    # 测试 1: 简单模型分析
    print("\n[测试1] 简单模型图拓扑分析")
    print("-" * 50)
    
    tester1 = ModuleTestTemplate(
        module_name="graph_topology_simple",
        device="auto",
        backend="auto"
    )
    
    result1 = tester1.run_test(test_graph_topology_analysis)
    print(f"测试状态: {'通过' if result1.success else '失败'}")
    if result1.success:
        print("性能指标:")
        print(f"  总耗时: {result1.metrics.total_time_ms:.2f} ms")
        print(f"  平均耗时: {result1.metrics.avg_time_ms:.2f} ms")
        print(f"  峰值内存: {result1.metrics.peak_memory_gb:.2f} GB")
    
    # 测试 2: Transformer 模型分析
    print("\n[测试2] Transformer 图拓扑分析")
    print("-" * 50)
    
    tester2 = ModuleTestTemplate(
        module_name="graph_topology_transformer",
        device="auto",
        backend="auto"
    )
    
    result2 = tester2.run_test(test_transformer_topology)
    print(f"测试状态: {'通过' if result2.success else '失败'}")
    
    # 生成报告
    print("\n[报告] 生成综合报告")
    print("-" * 50)
    
    report = tester1.generate_report([result1, result2])
    tester1.print_report_summary(report)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
