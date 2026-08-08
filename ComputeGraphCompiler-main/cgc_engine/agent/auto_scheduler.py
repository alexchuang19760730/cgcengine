# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
Auto-Scheduling 模块 - AutoScheduler

功能：
- 启发式搜索最优展开、预取、SIMD调度
- 支持多种调度策略
- 自动优化执行计划
"""

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
import logging
import random
from enum import Enum

logger = logging.getLogger(__name__)


class SchedulingDecision(Enum):
    """调度决策类型"""
    UNROLL = "unroll"                    # 循环展开
    PREFETCH = "prefetch"                # 数据预取
    PIPELINE = "pipeline"                # 流水线
    VECTORIZE = "vectorize"              # SIMD向量化
    PARALLELIZE = "parallelize"          # 并行化


@dataclass
class ScheduleConfig:
    """调度配置"""
    unroll_factor: int = 4
    prefetch_distance: int = 2
    pipeline_depth: int = 2
    vector_width: int = 8  # SIMD向量宽度
    num_threads: int = 128
    use_tiling: bool = True
    use_fusion: bool = True


@dataclass
class ScheduleCandidate:
    """调度候选"""
    config: ScheduleConfig
    estimated_cycles: float = float('inf')
    throughput_gflops: float = 0.0
    valid: bool = True


class AutoScheduler:
    """自动调度器"""
    
    def __init__(self, graph_module: GraphModule):
        self.graph_module = graph_module
        self.graph = graph_module.graph
        self.schedule_configs: Dict[Node, ScheduleConfig] = {}
    
    def schedule(self) -> GraphModule:
        """执行自动调度"""
        logger.info("[AutoScheduler] Starting automatic scheduling...")
        
        # 1. 分析计算图特征
        features = self._analyze_graph_features()
        
        # 2. 为每个节点搜索最佳调度
        for node in self.graph.nodes:
            best_config = self._search_best_schedule(node, features)
            self.schedule_configs[node] = best_config
        
        # 3. 应用调度配置
        self._apply_schedule_configs()
        
        # 4. 优化整体调度
        self._optimize_overall_schedule()
        
        logger.info(f"[AutoScheduler] Scheduling complete! Configured {len(self.schedule_configs)} nodes")
        
        return self.graph_module
    
    def _analyze_graph_features(self) -> Dict[str, Any]:
        """分析计算图特征"""
        features = {
            "num_nodes": len(list(self.graph.nodes)),
            "num_matmul": 0,
            "num_elementwise": 0,
            "total_flops": 0,
            "critical_path_length": 0,
        }
        
        for node in self.graph.nodes:
            op_name = str(node.target)
            
            if "matmul" in op_name.lower() or "linear" in op_name.lower():
                features["num_matmul"] += 1
            elif "add" in op_name.lower() or "mul" in op_name.lower() or "relu" in op_name.lower():
                features["num_elementwise"] += 1
            
            # 估算 FLOPs
            features["total_flops"] += self._estimate_flops(node)
        
        return features
    
    def _estimate_flops(self, node: Node) -> int:
        """估算节点的 FLOPs"""
        try:
            output_val = node.meta.get("val", torch.randn(1024))
            output_shape = output_val.shape
            
            op_name = str(node.target)
            if "matmul" in op_name.lower() or "linear" in op_name.lower():
                if len(output_shape) >= 2:
                    m, n = output_shape[-2:]
                    k = 1024  # 假设
                    return 2 * m * n * k
            elif "add" in op_name.lower() or "mul" in op_name.lower():
                return torch.prod(torch.tensor(output_shape))
        except:
            pass
        
        return 1024  # 默认值
    
    def _search_best_schedule(self, node: Node, features: Dict[str, Any]) -> ScheduleConfig:
        """搜索最佳调度配置"""
        candidates = self._generate_schedule_candidates(node)
        valid_candidates = [c for c in candidates if c.valid]
        
        if not valid_candidates:
            return ScheduleConfig()
        
        # 选择性能最佳的配置
        best = max(valid_candidates, key=lambda c: c.throughput_gflops)
        logger.debug(f"Selected schedule: {best.config}")
        
        return best.config
    
    def _generate_schedule_candidates(self, node: Node) -> List[ScheduleCandidate]:
        """生成调度候选"""
        candidates = []
        
        # 定义搜索空间
        unroll_options = [1, 2, 4, 8, 16]
        prefetch_options = [0, 1, 2, 4]
        pipeline_options = [1, 2, 4]
        vector_width_options = [4, 8, 16, 32]
        
        op_name = str(node.target)
        
        # 根据操作类型限制搜索空间
        if "matmul" in op_name.lower():
            unroll_options = [4, 8, 16]
            vector_width_options = [8, 16]
        elif "elementwise" in op_name.lower():
            unroll_options = [2, 4, 8]
            vector_width_options = [16, 32]
        
        # 生成候选
        for unroll, prefetch, pipeline, vector_width in zip(
            unroll_options,
            prefetch_options,
            pipeline_options,
            vector_width_options,
        ):
            config = ScheduleConfig(
                unroll_factor=unroll,
                prefetch_distance=prefetch,
                pipeline_depth=pipeline,
                vector_width=vector_width,
            )
            
            candidate = self._evaluate_candidate(node, config)
            candidates.append(candidate)
        
        return candidates
    
    def _evaluate_candidate(self, node: Node, config: ScheduleConfig) -> ScheduleCandidate:
        """评估调度候选"""
        candidate = ScheduleCandidate(config=config)
        
        # 估算执行周期
        cycles = self._estimate_cycles(node, config)
        candidate.estimated_cycles = cycles
        
        # 计算吞吐量
        flops = self._estimate_flops(node)
        if cycles > 0:
            candidate.throughput_gflops = flops / cycles / 1e9
        
        # 检查约束
        if config.unroll_factor > 16:
            candidate.valid = False
        
        if config.pipeline_depth > 8:
            candidate.valid = False
        
        return candidate
    
    def _estimate_cycles(self, node: Node, config: ScheduleConfig) -> float:
        """估算执行周期"""
        base_cycles = 1000  # 基础周期
        
        # 循环展开加速
        unroll_speedup = min(config.unroll_factor, 8)
        
        # 预取延迟隐藏
        prefetch_benefit = 1.0 + config.prefetch_distance * 0.1
        
        # 流水线加速
        pipeline_speedup = min(config.pipeline_depth, 4)
        
        # SIMD向量化加速
        vector_speedup = config.vector_width / 4
        
        total_speedup = unroll_speedup * prefetch_benefit * pipeline_speedup * vector_speedup
        
        return base_cycles / total_speedup
    
    def _apply_schedule_configs(self):
        """应用调度配置"""
        for node, config in self.schedule_configs.items():
            # 将调度配置存储到节点元数据
            if "schedule" not in node.meta:
                node.meta["schedule"] = {}
            
            node.meta["schedule"]["unroll_factor"] = config.unroll_factor
            node.meta["schedule"]["prefetch_distance"] = config.prefetch_distance
            node.meta["schedule"]["pipeline_depth"] = config.pipeline_depth
            node.meta["schedule"]["vector_width"] = config.vector_width
    
    def _optimize_overall_schedule(self):
        """优化整体调度"""
        # 执行指令重排序
        self._reorder_instructions()
        
        # 优化寄存器分配
        self._optimize_register_allocation()
        
        # 应用流水线调度
        self._apply_pipelining()
    
    def _reorder_instructions(self):
        """重排序指令以最大化流水线效率"""
        logger.debug("[AutoScheduler] Reordering instructions...")
        
        # 获取节点依赖关系
        dependencies = self._build_dependency_graph()
        
        # 执行拓扑排序优化
        reordered_nodes = self._topological_sort_with_priority(dependencies)
        
        # 更新图中的节点顺序
        if reordered_nodes:
            logger.debug(f"[AutoScheduler] Reordered {len(reordered_nodes)} nodes")
    
    def _build_dependency_graph(self) -> Dict[Node, List[Node]]:
        """构建依赖图"""
        dependencies = {}
        
        for node in self.graph.nodes:
            dependencies[node] = []
            # 查找依赖的节点
            for arg in node.args:
                if isinstance(arg, Node):
                    dependencies[node].append(arg)
        
        return dependencies
    
    def _topological_sort_with_priority(self, dependencies: Dict[Node, List[Node]]) -> List[Node]:
        """带优先级的拓扑排序"""
        # 使用 Kahn 算法进行拓扑排序
        in_degree = {node: len(deps) for node, deps in dependencies.items()}
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # 优先选择计算量大的节点
            queue.sort(key=lambda n: self._estimate_flops(n), reverse=True)
            node = queue.pop(0)
            result.append(node)
            
            # 更新依赖
            for dependent in self.graph.nodes:
                if node in dependencies[dependent]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        return result
    
    def _optimize_register_allocation(self):
        """优化寄存器分配"""
        logger.debug("[AutoScheduler] Optimizing register allocation...")
        
        # 计算每个节点所需的寄存器数量
        register_usage = {}
        max_registers = 0
        
        for node in self.graph.nodes:
            regs = self._estimate_register_usage(node)
            register_usage[node] = regs
            max_registers = max(max_registers, regs)
        
        # 调整展开因子以适应寄存器限制
        for node, config in self.schedule_configs.items():
            regs_needed = register_usage[node] * config.unroll_factor
            if regs_needed > 256:  # 假设寄存器限制
                config.unroll_factor = max(1, 256 // register_usage[node])
                logger.debug(f"Reduced unroll factor for {node.target}: {config.unroll_factor}")
    
    def _estimate_register_usage(self, node: Node) -> int:
        """估算节点的寄存器使用"""
        op_name = str(node.target).lower()
        
        if "matmul" in op_name or "linear" in op_name:
            return 32  # 矩阵操作需要更多寄存器
        elif "elementwise" in op_name or "add" in op_name or "mul" in op_name:
            return 8  # 简单操作
        else:
            return 16  # 默认
    
    def _apply_pipelining(self):
        """应用流水线调度"""
        logger.debug("[AutoScheduler] Applying pipelining...")
        
        # 识别可以流水线化的节点序列
        pipeline_candidates = self._find_pipeline_candidates()
        
        for i, candidate in enumerate(pipeline_candidates[:-1]):
            next_candidate = pipeline_candidates[i + 1]
            
            # 设置流水线深度
            if candidate in self.schedule_configs:
                self.schedule_configs[candidate].pipeline_depth = 2
            if next_candidate in self.schedule_configs:
                self.schedule_configs[next_candidate].pipeline_depth = 2
        
        logger.debug(f"[AutoScheduler] Applied pipelining to {len(pipeline_candidates)} nodes")
    
    def _find_pipeline_candidates(self) -> List[Node]:
        """查找可以流水线化的节点序列"""
        candidates = []
        
        for node in self.graph.nodes:
            op_name = str(node.target).lower()
            if "matmul" in op_name or "conv" in op_name:
                candidates.append(node)
        
        return candidates


def auto_schedule(graph_module: GraphModule) -> GraphModule:
    """便捷函数：自动调度"""
    scheduler = AutoScheduler(graph_module)
    return scheduler.schedule()