# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
自动深层融合模块 - AutoFusion

功能：
- 无人工限制的自动算子融合
- 支持多种融合模式
- 递归融合直到无法继续
- 生成融合后的优化图
"""

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node, Graph
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
import logging
from enum import Enum

from .dependency_analyzer import DependencyAnalyzer, DependencyGraph, NodeAnalysis

logger = logging.getLogger(__name__)


class FusionType(Enum):
    """融合类型"""
    ELEMENTWISE_FUSION = "elementwise_fusion"      # 逐元素融合
    LINEAR_ACT_FUSION = "linear_act_fusion"        # Linear+Activation
    LAYER_NORM_FUSION = "layer_norm_fusion"        # LayerNorm融合
    ATTENTION_FUSION = "attention_fusion"          # 注意力融合
    MLP_FUSION = "mlp_fusion"                      # MLP融合
    CHAIN_FUSION = "chain_fusion"                  # 链式融合


@dataclass
class FusionPlan:
    """融合计划"""
    fusion_type: FusionType
    nodes: List[Node]
    fused_node: Optional[Node] = None
    flops_saved: int = 0
    memory_saved_bytes: int = 0


class AutoFusion:
    """自动深层融合器"""
    
    def __init__(self, graph_module: GraphModule):
        self.graph_module = graph_module
        self.graph = graph_module.graph
        self.fusion_plans: List[FusionPlan] = []
        self.fused_graph: Optional[Graph] = None
    
    def fuse(self, max_fusion_depth: int = 10) -> GraphModule:
        """执行自动深层融合"""
        logger.info(f"[AutoFusion] Starting automatic fusion (max_depth={max_fusion_depth})")
        
        # 1. 分析依赖图
        analyzer = DependencyAnalyzer(self.graph_module)
        dependency_graph = analyzer.analyze()
        
        # 2. 生成融合计划
        self._generate_fusion_plans(dependency_graph)
        
        # 3. 应用融合（递归直到无法继续）
        self._apply_fusion_recursively(max_fusion_depth)
        
        # 4. 优化融合后的图
        self._optimize_fused_graph()
        
        logger.info(f"[AutoFusion] Fusion complete! Generated {len(self.fusion_plans)} fusion plans")
        
        return self.graph_module
    
    def _generate_fusion_plans(self, dependency_graph: DependencyGraph):
        """生成融合计划"""
        visited = set()
        
        for analysis in dependency_graph.nodes:
            if analysis.node in visited:
                continue
            
            # 尝试融合
            if analysis.can_fuse:
                plan = self._create_fusion_plan(dependency_graph, analysis)
                if plan:
                    self.fusion_plans.append(plan)
                    for node in plan.nodes:
                        visited.add(node)
    
    def _create_fusion_plan(self, dependency_graph: DependencyGraph, start: NodeAnalysis) -> Optional[FusionPlan]:
        """创建融合计划"""
        nodes_to_fuse = [start.node]
        current = start
        
        # 递归寻找可融合的后续节点
        while True:
            next_node = self._find_next_fusible_node(dependency_graph, current)
            if next_node is None:
                break
            
            # 检查是否会创建无效的融合
            if not self._validate_fusion(nodes_to_fuse + [next_node.node]):
                break
            
            nodes_to_fuse.append(next_node.node)
            current = next_node
            
            # 限制融合深度
            if len(nodes_to_fuse) >= 8:
                break
        
        if len(nodes_to_fuse) < 2:
            return None
        
        # 确定融合类型
        fusion_type = self._determine_fusion_type(nodes_to_fuse)
        
        return FusionPlan(
            fusion_type=fusion_type,
            nodes=nodes_to_fuse,
        )
    
    def _find_next_fusible_node(self, dependency_graph: DependencyGraph, current: NodeAnalysis) -> Optional[NodeAnalysis]:
        """寻找下一个可融合的节点"""
        if current.node not in dependency_graph.node_index:
            return None
        
        current_idx = dependency_graph.node_index[current.node]
        
        # 找到直接后继节点
        for producer, consumer in dependency_graph.edges:
            if producer == current_idx:
                consumer_analysis = dependency_graph.nodes[consumer]
                if consumer_analysis.can_fuse and consumer_analysis.node != current.node:
                    return consumer_analysis
        
        return None
    
    def _validate_fusion(self, nodes: List[Node]) -> bool:
        """验证融合是否有效"""
        # 检查是否有冲突的操作
        for node in nodes:
            op_name = str(node.target)
            
            # 不允许融合控制流操作
            if "if" in op_name or "loop" in op_name or "cond" in op_name:
                return False
            
            # 不允许融合内存操作
            if "view" in op_name or "reshape" in op_name or "transpose" in op_name:
                return False
        
        return True
    
    def _determine_fusion_type(self, nodes: List[Node]) -> FusionType:
        """确定融合类型"""
        op_names = [str(n.target) for n in nodes]
        
        # 检查是否包含注意力操作
        if any("attention" in name.lower() or "sdpa" in name.lower() for name in op_names):
            return FusionType.ATTENTION_FUSION
        
        # 检查是否包含线性层
        if any("linear" in name.lower() or "matmul" in name.lower() for name in op_names):
            # 检查是否包含激活函数
            if any("relu" in name.lower() or "gelu" in name.lower() or "silu" in name.lower() for name in op_names):
                return FusionType.LINEAR_ACT_FUSION
        
        # 检查是否包含 LayerNorm
        if any("layer_norm" in name.lower() or "layernorm" in name.lower() for name in op_names):
            return FusionType.LAYER_NORM_FUSION
        
        # 检查是否全是逐元素操作
        elementwise_ops = {"add", "mul", "sub", "div", "relu", "gelu", "silu", "tanh", "sigmoid"}
        if all(any(op in name.lower() for op in elementwise_ops) for name in op_names):
            if len(nodes) > 2:
                return FusionType.CHAIN_FUSION
            return FusionType.ELEMENTWISE_FUSION
        
        return FusionType.CHAIN_FUSION
    
    def _apply_fusion_recursively(self, max_depth: int):
        """递归应用融合"""
        depth = 0
        changed = True
        
        while changed and depth < max_depth:
            changed = False
            
            for plan in self.fusion_plans:
                if plan.fused_node is None:
                    if self._apply_single_fusion(plan):
                        changed = True
            
            depth += 1
    
    def _apply_single_fusion(self, plan: FusionPlan) -> bool:
        """应用单个融合"""
        try:
            # 创建融合节点
            fused_node = self._create_fused_node(plan)
            
            if fused_node:
                plan.fused_node = fused_node
                
                # 替换原始节点
                self._replace_nodes_with_fused(plan)
                
                return True
        except Exception as e:
            logger.debug(f"Failed to apply fusion: {e}")
        
        return False
    
    def _create_fused_node(self, plan: FusionPlan) -> Optional[Node]:
        """创建融合节点"""
        # 创建融合后的目标函数引用
        fused_target = self._create_fused_function(plan)
        
        # 获取第一个节点的参数作为融合节点的参数
        first_node = plan.nodes[0]
        args = first_node.args
        kwargs = first_node.kwargs or {}
        
        # 创建新节点
        with self.graph.inserting_before(first_node):
            new_node = self.graph.create_node(
                op="call_function",
                target=fused_target,
                args=args,
                kwargs=kwargs,
            )
            
            # 复制元数据
            new_node.meta = first_node.meta.copy()
            
            return new_node
    
    def _create_fused_function(self, plan: FusionPlan):
        """创建融合函数"""
        nodes = plan.nodes
        
        def fused_function(*args, **kwargs):
            # 执行第一个节点
            result = nodes[0].target(*args, **kwargs)
            
            # 依次执行后续节点
            for node in nodes[1:]:
                result = node.target(result)
            
            return result
        
        fused_function.__name__ = f"fused_{plan.fusion_type.value}"
        return fused_function
    
    def _replace_nodes_with_fused(self, plan: FusionPlan):
        """用融合节点替换原始节点"""
        if plan.fused_node is None:
            return
        
        # 将所有引用最后一个节点的地方改为引用融合节点
        last_node = plan.nodes[-1]
        
        for node in list(self.graph.nodes):
            # 更新参数
            new_args = list(node.args)
            for i, arg in enumerate(new_args):
                if arg == last_node:
                    new_args[i] = plan.fused_node
            node.args = tuple(new_args)
            
            # 更新关键字参数
            if node.kwargs:
                for key, value in node.kwargs.items():
                    if value == last_node:
                        node.kwargs[key] = plan.fused_node
        
        # 删除被融合的节点
        for node in plan.nodes:
            self.graph.erase_node(node)
    
    def _optimize_fused_graph(self):
        """优化融合后的图"""
        # 执行一些图优化
        self.graph.eliminate_dead_code()
        self.graph.lint()
        
        # 重新编译图模块
        self.graph_module.recompile()
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """获取融合统计信息"""
        stats = {
            "total_fusion_plans": len(self.fusion_plans),
            "fusion_types": {},
            "total_flops_saved": sum(p.flops_saved for p in self.fusion_plans),
            "total_memory_saved_bytes": sum(p.memory_saved_bytes for p in self.fusion_plans),
        }
        
        for plan in self.fusion_plans:
            fusion_type = plan.fusion_type.value
            stats["fusion_types"][fusion_type] = stats["fusion_types"].get(fusion_type, 0) + 1
        
        return stats


def auto_fuse(graph_module: GraphModule, **kwargs) -> GraphModule:
    """便捷函数：自动融合"""
    fuser = AutoFusion(graph_module)
    return fuser.fuse(**kwargs)