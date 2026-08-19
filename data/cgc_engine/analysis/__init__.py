# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
分析模块 - 提供完整的图拓扑分析功能

功能:
✅ 节点分析 (节点类型识别、属性提取)
✅ 依赖分析 (前驱/后继关系)
✅ 拓扑排序 (执行顺序确定)
✅ 关键路径识别 (性能瓶颈分析)
✅ 并行分析 (可并行节点组识别)
✅ 图统计信息 (深度/广度/节点数)
✅ 可视化支持

使用方式:
    from cgc_engine.analysis import GraphTopologyAnalyzer
    
    analyzer = GraphTopologyAnalyzer()
    result = analyzer.analyze(fx_graph)
    analyzer.print_summary(result)
"""

from .graph_topology_analyzer import (
    GraphTopologyAnalyzer,
    GraphAnalysisResult,
    NodeInfo,
    GraphStatistics,
    CriticalPathInfo,
    ParallelGroup,
)

__all__ = [
    "GraphTopologyAnalyzer",
    "GraphAnalysisResult",
    "NodeInfo",
    "GraphStatistics",
    "CriticalPathInfo",
    "ParallelGroup",
]
