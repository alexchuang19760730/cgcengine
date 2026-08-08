# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
依赖图分析模块 - DependencyAnalyzer

功能：
- 分析 PyTorch FX 计算图的依赖关系
- 标记可融合算子
- 标记可并行执行的算子
- 标记可丢弃的中间张量
- 生成优化建议
"""

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TensorUsage(Enum):
    """张量使用类型"""
    SINGLE_USE = "single_use"        # 只被使用一次
    MULTI_USE = "multi_use"          # 被多次使用
    OUTPUT = "output"                # 输出张量
    DISCARDABLE = "discardable"      # 可丢弃（中间结果）


class OperatorCategory(Enum):
    """算子类别"""
    ELEMENTWISE = "elementwise"      # 逐元素操作
    REDUCTION = "reduction"          # 归约操作
    MATRIX = "matrix"                # 矩阵操作
    CONTROL_FLOW = "control_flow"    # 控制流操作
    MEMORY = "memory"                # 内存操作
    ATTENTION = "attention"          # 注意力操作
    ACTIVATION = "activation"        # 激活函数


class FusionPattern(Enum):
    """融合模式"""
    ELEMENTWISE_CHAIN = "elementwise_chain"  # 逐元素链式融合
    LINEAR_ACT = "linear_act"                # Linear+Activation
    ATTENTION_BLOCK = "attention_block"      # 注意力块
    LAYER_NORM_ACT = "layer_norm_act"        # LayerNorm+Activation
    MLP_BLOCK = "mlp_block"                  # MLP块


@dataclass
class NodeAnalysis:
    """节点分析结果"""
    node: Node
    op_category: OperatorCategory
    can_fuse: bool = False
    fuse_partner: Optional[Node] = None
    fuse_pattern: Optional[FusionPattern] = None
    can_parallel: bool = False
    parallel_group: Optional[int] = None
    produces_discardable: bool = False
    memory_usage_bytes: int = 0
    compute_intensity: float = 0.0  # FLOPs / bytes


@dataclass
class DependencyGraph:
    """依赖图"""
    nodes: List[NodeAnalysis] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)  # (producer, consumer)
    node_index: Dict[Node, int] = field(default_factory=dict)
    
    def add_node(self, analysis: NodeAnalysis) -> int:
        idx = len(self.nodes)
        self.nodes.append(analysis)
        self.node_index[analysis.node] = idx
        return idx
    
    def add_edge(self, producer: Node, consumer: Node):
        if producer in self.node_index and consumer in self.node_index:
            self.edges.append((self.node_index[producer], self.node_index[consumer]))


class DependencyAnalyzer:
    """依赖图分析器"""
    
    def __init__(self, graph_module: GraphModule):
        self.graph_module = graph_module
        self.graph = graph_module.graph
        self.dependency_graph = DependencyGraph()
        self.tensor_usage: Dict[Node, TensorUsage] = {}
        
    def analyze(self) -> DependencyGraph:
        """执行完整的依赖图分析"""
        logger.info("[DependencyAnalyzer] Starting dependency graph analysis...")
        
        # 1. 分析每个节点
        for node in self.graph.nodes:
            analysis = self._analyze_node(node)
            self.dependency_graph.add_node(analysis)
        
        # 2. 构建依赖边
        self._build_dependency_edges()
        
        # 3. 分析张量使用情况
        self._analyze_tensor_usage()
        
        # 4. 标记可融合节点
        self._mark_fusible_nodes()
        
        # 5. 标记可并行节点
        self._mark_parallel_nodes()
        
        # 6. 标记可丢弃张量
        self._mark_discardable_tensors()
        
        logger.info("[DependencyAnalyzer] Analysis complete!")
        
        return self.dependency_graph
    
    def _analyze_node(self, node: Node) -> NodeAnalysis:
        """分析单个节点"""
        op_category = self._classify_operator(node)
        can_fuse = self._check_fusible(node, op_category)
        compute_intensity = self._estimate_compute_intensity(node)
        
        return NodeAnalysis(
            node=node,
            op_category=op_category,
            can_fuse=can_fuse,
            compute_intensity=compute_intensity,
        )
    
    def _classify_operator(self, node: Node) -> OperatorCategory:
        """分类算子类型"""
        op_name = node.target.__name__ if callable(node.target) else str(node.target)
        
        # 逐元素操作
        elementwise_ops = {"add", "mul", "sub", "div", "pow", "neg", "abs", "sqrt", "rsqrt"}
        if op_name in elementwise_ops or "elementwise" in op_name.lower():
            return OperatorCategory.ELEMENTWISE
        
        # 激活函数
        activation_ops = {"relu", "gelu", "silu", "swish", "tanh", "sigmoid", "softmax"}
        if op_name.lower() in activation_ops:
            return OperatorCategory.ACTIVATION
        
        # 矩阵操作
        if op_name in {"matmul", "linear", "conv2d", "conv_transpose2d"}:
            return OperatorCategory.MATRIX
        
        # 归约操作
        reduction_ops = {"sum", "mean", "max", "min", "var", "std"}
        if op_name in reduction_ops:
            return OperatorCategory.REDUCTION
        
        # 注意力操作
        if "attention" in op_name.lower() or "sdpa" in op_name.lower():
            return OperatorCategory.ATTENTION
        
        # 内存操作
        if op_name in {"view", "reshape", "transpose", "permute", "contiguous"}:
            return OperatorCategory.MEMORY
        
        # 控制流
        if op_name in {"if", "loop", "cond"} or node.op == "call_function":
            try:
                import operator
                if node.target in {operator.add, operator.mul, operator.sub, operator.truediv}:
                    return OperatorCategory.ELEMENTWISE
            except:
                pass
        
        return OperatorCategory.ELEMENTWISE
    
    def _check_fusible(self, node: Node, category: OperatorCategory) -> bool:
        """检查节点是否可融合"""
        # 逐元素操作和激活函数通常可以融合
        if category in {OperatorCategory.ELEMENTWISE, OperatorCategory.ACTIVATION}:
            return True
        
        # 矩阵操作后跟激活函数时可以融合
        if category == OperatorCategory.MATRIX:
            return True
        
        return False
    
    def _estimate_compute_intensity(self, node: Node) -> float:
        """估算计算强度 (FLOPs / bytes)"""
        try:
            # 获取输出形状
            output_shape = node.meta.get("val", torch.randn(1024)).shape
            if not output_shape:
                return 1.0
            
            # 估算操作类型的 FLOPs
            op_name = node.target.__name__ if callable(node.target) else str(node.target)
            
            if op_name in {"matmul", "linear"}:
                # matmul: M*N*K FLOPs
                if len(output_shape) >= 2:
                    m, n = output_shape[-2:]
                    k = 1024  # 假设
                    flops = m * n * k
                    bytes = m * n * 4  # float32
                    return flops / bytes if bytes > 0 else 1.0
            
            elif op_name in {"add", "mul", "relu", "gelu"}:
                # 逐元素操作: N FLOPs
                n = torch.prod(torch.tensor(output_shape))
                flops = n
                bytes = n * 4
                return flops / bytes if bytes > 0 else 1.0
            
        except Exception as e:
            logger.debug(f"Failed to estimate compute intensity for {node}: {e}")
        
        return 1.0
    
    def _build_dependency_edges(self):
        """构建依赖边"""
        for node in self.graph.nodes:
            # 遍历节点的所有输入
            for arg in node.args:
                if isinstance(arg, Node):
                    self.dependency_graph.add_edge(arg, node)
    
    def _analyze_tensor_usage(self):
        """分析张量使用情况"""
        usage_count: Dict[Node, int] = {}
        
        # 统计每个节点输出的使用次数
        for node in self.graph.nodes:
            usage_count[node] = 0
        
        # 遍历所有节点的参数
        for node in self.graph.nodes:
            for arg in node.args:
                if isinstance(arg, Node):
                    usage_count[arg] += 1
        
        # 确定使用类型
        for node, count in usage_count.items():
            if count == 0:
                # 没有被使用，可能是输出
                self.tensor_usage[node] = TensorUsage.OUTPUT
            elif count == 1:
                self.tensor_usage[node] = TensorUsage.SINGLE_USE
            else:
                self.tensor_usage[node] = TensorUsage.MULTI_USE
    
    def _mark_fusible_nodes(self):
        """标记可融合节点"""
        nodes = self.dependency_graph.nodes
        
        for i, analysis in enumerate(nodes):
            if not analysis.can_fuse:
                continue
            
            # 寻找可融合的伙伴
            for j in range(i + 1, len(nodes)):
                other = nodes[j]
                if other.can_fuse and self._can_fuse_together(analysis, other):
                    analysis.fuse_partner = other.node
                    analysis.fuse_pattern = self._detect_fusion_pattern(analysis, other)
                    break
    
    def _can_fuse_together(self, a: NodeAnalysis, b: NodeAnalysis) -> bool:
        """检查两个节点是否可以融合"""
        # 检查是否相邻（a 的输出是 b 的输入）
        if a.node in self.dependency_graph.node_index:
            a_idx = self.dependency_graph.node_index[a.node]
            for producer, consumer in self.dependency_graph.edges:
                if producer == a_idx:
                    consumer_node = self.dependency_graph.nodes[consumer].node
                    if consumer_node == b.node:
                        return True
        
        return False
    
    def _detect_fusion_pattern(self, a: NodeAnalysis, b: NodeAnalysis) -> FusionPattern:
        """检测融合模式"""
        if a.op_category == OperatorCategory.MATRIX and b.op_category == OperatorCategory.ACTIVATION:
            return FusionPattern.LINEAR_ACT
        
        if a.op_category == OperatorCategory.ELEMENTWISE and b.op_category == OperatorCategory.ELEMENTWISE:
            return FusionPattern.ELEMENTWISE_CHAIN
        
        if a.op_category == OperatorCategory.ATTENTION:
            return FusionPattern.ATTENTION_BLOCK
        
        return FusionPattern.ELEMENTWISE_CHAIN
    
    def _mark_parallel_nodes(self):
        """标记可并行节点"""
        # 寻找没有依赖关系的节点组
        visited = set()
        
        for i, analysis in enumerate(self.dependency_graph.nodes):
            if i in visited:
                continue
            
            # BFS 找到所有可达节点
            reachable = set()
            queue = [i]
            
            while queue:
                idx = queue.pop(0)
                if idx in reachable:
                    continue
                reachable.add(idx)
                visited.add(idx)
                
                # 添加所有后继节点
                for producer, consumer in self.dependency_graph.edges:
                    if producer == idx and consumer not in reachable:
                        queue.append(consumer)
            
            # 标记同一组内的节点可以并行
            if len(reachable) > 1:
                for idx in reachable:
                    self.dependency_graph.nodes[idx].can_parallel = True
                    self.dependency_graph.nodes[idx].parallel_group = len(visited) // len(reachable)
    
    def _mark_discardable_tensors(self):
        """标记可丢弃张量"""
        for analysis in self.dependency_graph.nodes:
            usage = self.tensor_usage.get(analysis.node)
            if usage == TensorUsage.SINGLE_USE and analysis.op_category != OperatorCategory.MATRIX:
                analysis.produces_discardable = True
    
    def get_fusion_groups(self) -> List[List[Node]]:
        """获取融合组"""
        groups = []
        visited = set()
        
        for analysis in self.dependency_graph.nodes:
            if analysis.node in visited:
                continue
            
            if analysis.can_fuse and analysis.fuse_partner:
                group = [analysis.node, analysis.fuse_partner]
                visited.add(analysis.node)
                visited.add(analysis.fuse_partner)
                groups.append(group)
            else:
                visited.add(analysis.node)
        
        return groups
    
    def get_parallel_groups(self) -> List[List[Node]]:
        """获取并行组"""
        groups: Dict[int, List[Node]] = {}
        
        for analysis in self.dependency_graph.nodes:
            if analysis.can_parallel and analysis.parallel_group is not None:
                if analysis.parallel_group not in groups:
                    groups[analysis.parallel_group] = []
                groups[analysis.parallel_group].append(analysis.node)
        
        return list(groups.values())
    
    def get_discardable_tensors(self) -> List[Node]:
        """获取可丢弃张量"""
        return [
            analysis.node for analysis in self.dependency_graph.nodes
            if analysis.produces_discardable
        ]


def analyze_dependencies(graph_module: GraphModule) -> DependencyGraph:
    """便捷函数：分析依赖图"""
    analyzer = DependencyAnalyzer(graph_module)
    return analyzer.analyze()