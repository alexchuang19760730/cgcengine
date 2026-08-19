# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
依赖重排模块 - DependencyReorder

功能：
- 重排执行顺序，最大化缓存局部性
- 基于图着色的调度算法
- 考虑数据重用和缓存层次
"""

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
import logging
import heapq

from .dependency_analyzer import DependencyAnalyzer, DependencyGraph

logger = logging.getLogger(__name__)


@dataclass
class SchedulingNode:
    """调度节点"""
    node: Node
    ready: bool = True
    dependencies: int = 0
    cache_score: float = 0.0
    memory_usage: int = 0


class DependencyReorder:
    """依赖重排器"""
    
    def __init__(self, graph_module: GraphModule):
        self.graph_module = graph_module
        self.graph = graph_module.graph
        self.scheduling_nodes: Dict[Node, SchedulingNode] = {}
    
    def reorder(self, cache_size_bytes: int = 4 * 1024 * 1024 * 1024) -> GraphModule:
        """执行依赖重排"""
        logger.info(f"[DependencyReorder] Starting dependency reordering (cache_size={cache_size_bytes/1e9:.2f}GB)")
        
        # 1. 分析依赖图
        analyzer = DependencyAnalyzer(self.graph_module)
        dependency_graph = analyzer.analyze()
        
        # 2. 构建调度节点
        self._build_scheduling_nodes(dependency_graph)
        
        # 3. 计算缓存分数
        self._compute_cache_scores(dependency_graph, cache_size_bytes)
        
        # 4. 执行重排
        new_order = self._schedule_with_cache_awareness(cache_size_bytes)
        
        # 5. 应用新顺序
        self._apply_new_order(new_order)
        
        logger.info("[DependencyReorder] Reordering complete!")
        
        return self.graph_module
    
    def _build_scheduling_nodes(self, dependency_graph: DependencyGraph):
        """构建调度节点"""
        for analysis in dependency_graph.nodes:
            # 计算依赖数量
            deps = 0
            for producer, consumer in dependency_graph.edges:
                if consumer == dependency_graph.node_index[analysis.node]:
                    deps += 1
            
            self.scheduling_nodes[analysis.node] = SchedulingNode(
                node=analysis.node,
                ready=deps == 0,
                dependencies=deps,
            )
    
    def _compute_cache_scores(self, dependency_graph: DependencyGraph, cache_size_bytes: int):
        """计算缓存分数"""
        # 分析数据重用模式
        data_usage = self._analyze_data_reuse(dependency_graph)
        
        for node in self.graph.nodes:
            sched_node = self.scheduling_nodes.get(node)
            if sched_node is None:
                continue
            
            # 计算缓存分数：基于数据重用和内存访问模式
            score = self._calculate_cache_score(node, data_usage, cache_size_bytes)
            sched_node.cache_score = score
    
    def _analyze_data_reuse(self, dependency_graph: DependencyGraph) -> Dict[Node, int]:
        """分析数据重用模式"""
        usage_count: Dict[Node, int] = {}
        
        for node in self.graph.nodes:
            usage_count[node] = 0
        
        for producer, consumer in dependency_graph.edges:
            producer_node = dependency_graph.nodes[producer].node
            usage_count[producer_node] += 1
        
        return usage_count
    
    def _calculate_cache_score(self, node: Node, data_usage: Dict[Node, int], cache_size_bytes: int) -> float:
        """计算缓存分数"""
        score = 0.0
        
        # 基于输出大小的分数（越小越容易缓存）
        try:
            output_val = node.meta.get("val", torch.randn(1024))
            output_size = output_val.numel() * 4  # bytes
            
            if output_size < cache_size_bytes * 0.1:
                score += 0.5
            elif output_size < cache_size_bytes * 0.5:
                score += 0.3
        except:
            pass
        
        # 基于数据重用的分数（被使用次数越多越值得缓存）
        usage = data_usage.get(node, 0)
        if usage >= 3:
            score += 0.3
        elif usage == 2:
            score += 0.1
        
        # 基于操作类型的分数
        op_name = str(node.target)
        if "matmul" in op_name.lower() or "linear" in op_name.lower():
            score += 0.2
        
        return score
    
    def _schedule_with_cache_awareness(self, cache_size_bytes: int) -> List[Node]:
        """基于缓存感知的调度"""
        # 使用贪心算法：优先调度缓存分数高的节点
        result = []
        ready_nodes = []
        in_cache: Set[Node] = set()
        current_cache_usage = 0
        
        # 初始化就绪节点堆
        for sched_node in self.scheduling_nodes.values():
            if sched_node.ready:
                heapq.heappush(ready_nodes, (-sched_node.cache_score, id(sched_node), sched_node))
        
        while ready_nodes:
            # 选择缓存分数最高的节点
            neg_score, _, sched_node = heapq.heappop(ready_nodes)
            
            # 检查缓存容量
            node_size = self._estimate_node_memory(sched_node.node)
            
            # 如果缓存已满，先释放一些
            while current_cache_usage + node_size > cache_size_bytes and in_cache:
                # 释放最旧的或使用最少的
                oldest = in_cache.pop()
                oldest_size = self._estimate_node_memory(oldest)
                current_cache_usage -= oldest_size
            
            # 调度节点
            result.append(sched_node.node)
            current_cache_usage += node_size
            in_cache.add(sched_node.node)
            
            # 更新依赖关系
            self._update_dependencies(sched_node.node, ready_nodes)
        
        return result
    
    def _estimate_node_memory(self, node: Node) -> int:
        """估算节点输出的内存使用"""
        try:
            output_val = node.meta.get("val", torch.randn(1024))
            return output_val.numel() * 4  # float32
        except:
            return 1024 * 1024  # 默认 1MB
    
    def _update_dependencies(self, completed_node: Node, ready_nodes: list):
        """更新依赖关系"""
        for sched_node in self.scheduling_nodes.values():
            # 检查此节点是否依赖于已完成的节点
            for arg in sched_node.node.args:
                if arg == completed_node:
                    sched_node.dependencies -= 1
                    if sched_node.dependencies == 0:
                        sched_node.ready = True
                        heapq.heappush(ready_nodes, (-sched_node.cache_score, id(sched_node), sched_node))
    
    def _apply_new_order(self, new_order: List[Node]):
        """应用新的执行顺序"""
        # 创建新图
        new_graph = Graph()
        
        # 复制所有占位符节点
        placeholder_nodes = []
        for node in self.graph.nodes:
            if node.op == "placeholder":
                placeholder_nodes.append(node)
        
        # 按新顺序添加节点
        node_map: Dict[Node, Node] = {}
        
        # 先添加占位符
        for node in placeholder_nodes:
            new_node = new_graph.create_node(
                op=node.op,
                target=node.target,
                args=node.args,
                kwargs=node.kwargs,
            )
            node_map[node] = new_node
            new_node.meta = node.meta
        
        # 添加其他节点（按新顺序）
        for node in new_order:
            if node.op == "placeholder":
                continue
            
            # 更新参数引用
            new_args = []
            for arg in node.args:
                if isinstance(arg, Node):
                    new_args.append(node_map.get(arg, arg))
                else:
                    new_args.append(arg)
            
            new_kwargs = {}
            if node.kwargs:
                for key, value in node.kwargs.items():
                    if isinstance(value, Node):
                        new_kwargs[key] = node_map.get(value, value)
                    else:
                        new_kwargs[key] = value
            
            new_node = new_graph.create_node(
                op=node.op,
                target=node.target,
                args=tuple(new_args),
                kwargs=new_kwargs,
            )
            node_map[node] = new_node
            new_node.meta = node.meta
        
        # 添加输出节点
        for node in self.graph.nodes:
            if node.op == "output":
                new_args = [node_map.get(arg, arg) for arg in node.args]
                new_graph.output(tuple(new_args))
        
        # 替换旧图
        self.graph_module.graph = new_graph
        self.graph_module.recompile()


def reorder_dependencies(graph_module: GraphModule, **kwargs) -> GraphModule:
    """便捷函数：重排依赖"""
    reorder = DependencyReorder(graph_module)
    return reorder.reorder(**kwargs)