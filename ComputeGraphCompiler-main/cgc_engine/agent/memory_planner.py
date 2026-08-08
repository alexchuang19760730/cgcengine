# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
层级内存规划模块 - MemoryPlanner

功能：
- 自动安排寄存器/L1/L2/全局内存驻留
- 基于数据访问模式的内存分配优化
- 支持多级缓存层次
"""

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryLevel(Enum):
    """内存层级"""
    REGISTER = "register"
    L1_CACHE = "l1_cache"
    L2_CACHE = "l2_cache"
    SHARED_MEMORY = "shared_memory"
    GLOBAL_MEMORY = "global_memory"


@dataclass
class MemoryHierarchy:
    """内存层次结构"""
    register_size_bytes: int = 65536 * 4  # 每个SM的寄存器大小
    l1_cache_size_bytes: int = 128 * 1024  # 128KB
    l2_cache_size_bytes: int = 4 * 1024 * 1024  # 4MB
    shared_memory_size_bytes: int = 48 * 1024  # 48KB
    global_memory_size_bytes: int = 40 * 1024 * 1024 * 1024  # 40GB


@dataclass
class MemoryAssignment:
    """内存分配"""
    node: Node
    level: MemoryLevel
    priority: int = 0
    allocation_size_bytes: int = 0


class MemoryPlanner:
    """层级内存规划器"""
    
    def __init__(self, graph_module: GraphModule, hierarchy: Optional[MemoryHierarchy] = None):
        self.graph_module = graph_module
        self.graph = graph_module.graph
        self.hierarchy = hierarchy or MemoryHierarchy()
        self.assignments: List[MemoryAssignment] = []
        
        # 当前各层级的内存使用
        self.current_usage: Dict[MemoryLevel, int] = {
            MemoryLevel.REGISTER: 0,
            MemoryLevel.L1_CACHE: 0,
            MemoryLevel.L2_CACHE: 0,
            MemoryLevel.SHARED_MEMORY: 0,
            MemoryLevel.GLOBAL_MEMORY: 0,
        }
    
    def plan(self) -> GraphModule:
        """执行内存规划"""
        logger.info("[MemoryPlanner] Starting memory planning...")
        
        # 1. 分析数据访问模式
        access_patterns = self._analyze_access_patterns()
        
        # 2. 计算每个节点的内存分配优先级
        priorities = self._compute_priorities(access_patterns)
        
        # 3. 执行内存分配
        self._perform_allocation(priorities)
        
        # 4. 优化内存布局
        self._optimize_layout()
        
        logger.info(f"[MemoryPlanner] Planning complete! Assigned {len(self.assignments)} nodes")
        
        return self.graph_module
    
    def _analyze_access_patterns(self) -> Dict[Node, Dict[str, Any]]:
        """分析数据访问模式"""
        patterns = {}
        
        for node in self.graph.nodes:
            pattern = {
                "access_count": 0,
                "reuse_distance": 0,
                "lifetime": 0,
                "output_size_bytes": 0,
            }
            
            # 计算访问次数
            for consumer in self.graph.nodes:
                if node in consumer.args:
                    pattern["access_count"] += 1
            
            # 估算输出大小
            try:
                output_val = node.meta.get("val", torch.randn(1024))
                pattern["output_size_bytes"] = output_val.numel() * 4  # float32
            except:
                pattern["output_size_bytes"] = 1024 * 1024  # 默认 1MB
            
            patterns[node] = pattern
        
        return patterns
    
    def _compute_priorities(self, access_patterns: Dict[Node, Dict[str, Any]]) -> List[Tuple[int, Node]]:
        """计算优先级"""
        priorities = []
        
        for node, pattern in access_patterns.items():
            # 基于访问次数和重用距离计算优先级
            priority = 0
            
            # 访问次数越多优先级越高
            if pattern["access_count"] >= 5:
                priority += 50
            elif pattern["access_count"] >= 3:
                priority += 30
            elif pattern["access_count"] >= 2:
                priority += 15
            
            # 数据越小越容易放入高速缓存
            if pattern["output_size_bytes"] < 1024:
                priority += 30
            elif pattern["output_size_bytes"] < 16 * 1024:
                priority += 20
            elif pattern["output_size_bytes"] < 256 * 1024:
                priority += 10
            
            # 操作类型优先级
            op_name = str(node.target)
            if "matmul" in op_name.lower() or "linear" in op_name.lower():
                priority += 20
            elif "attention" in op_name.lower():
                priority += 15
            
            priorities.append((priority, node))
        
        # 按优先级排序
        priorities.sort(reverse=True, key=lambda x: x[0])
        
        return priorities
    
    def _perform_allocation(self, priorities: List[Tuple[int, Node]]):
        """执行内存分配"""
        for priority, node in priorities:
            # 获取节点输出大小
            output_size = self._get_output_size(node)
            
            # 尝试按层级分配
            for level in [MemoryLevel.REGISTER, MemoryLevel.SHARED_MEMORY, 
                         MemoryLevel.L1_CACHE, MemoryLevel.L2_CACHE, MemoryLevel.GLOBAL_MEMORY]:
                if self._can_allocate(level, output_size):
                    self._allocate(level, node, output_size, priority)
                    break
    
    def _get_output_size(self, node: Node) -> int:
        """获取节点输出大小"""
        try:
            output_val = node.meta.get("val", torch.randn(1024))
            return output_val.numel() * 4  # float32
        except:
            return 1024 * 1024  # 默认 1MB
    
    def _can_allocate(self, level: MemoryLevel, size: int) -> bool:
        """检查是否可以在指定层级分配"""
        max_sizes = {
            MemoryLevel.REGISTER: self.hierarchy.register_size_bytes // 32,  # 假设32个warp
            MemoryLevel.SHARED_MEMORY: self.hierarchy.shared_memory_size_bytes,
            MemoryLevel.L1_CACHE: self.hierarchy.l1_cache_size_bytes,
            MemoryLevel.L2_CACHE: self.hierarchy.l2_cache_size_bytes,
            MemoryLevel.GLOBAL_MEMORY: self.hierarchy.global_memory_size_bytes,
        }
        
        return self.current_usage[level] + size <= max_sizes[level]
    
    def _allocate(self, level: MemoryLevel, node: Node, size: int, priority: int):
        """执行分配"""
        self.current_usage[level] += size
        
        assignment = MemoryAssignment(
            node=node,
            level=level,
            priority=priority,
            allocation_size_bytes=size,
        )
        self.assignments.append(assignment)
        
        # 记录到节点元数据
        if "memory_level" not in node.meta:
            node.meta["memory_level"] = level.value
    
    def _optimize_layout(self):
        """优化内存布局"""
        # 执行内存合并优化
        self._merge_small_allocations()
        
        # 执行预取优化
        self._schedule_prefetch()
        
        # 执行数据重用优化
        self._optimize_data_reuse()
    
    def _merge_small_allocations(self):
        """合并小的内存分配"""
        logger.debug("[MemoryPlanner] Merging small allocations...")
        
        # 按内存层级分组
        assignments_by_level = {}
        for level in MemoryLevel:
            assignments_by_level[level] = []
        
        for assignment in self.assignments:
            assignments_by_level[assignment.level].append(assignment)
        
        # 合并小于阈值的分配
        merge_threshold = 1024  # 1KB
        merged_count = 0
        
        for level, assignments in assignments_by_level.items():
            # 按节点顺序排序
            sorted_assignments = sorted(assignments, key=lambda a: self._get_node_index(a.node))
            
            i = 0
            while i < len(sorted_assignments) - 1:
                current = sorted_assignments[i]
                next_assignment = sorted_assignments[i + 1]
                
                if current.allocation_size_bytes < merge_threshold and \
                   next_assignment.allocation_size_bytes < merge_threshold:
                    # 合并两个分配
                    merged_size = current.allocation_size_bytes + next_assignment.allocation_size_bytes
                    
                    # 更新当前分配
                    current.allocation_size_bytes = merged_size
                    
                    # 更新内存使用
                    self.current_usage[level] -= next_assignment.allocation_size_bytes
                    
                    # 移除下一个分配
                    self.assignments.remove(next_assignment)
                    sorted_assignments.pop(i + 1)
                    
                    merged_count += 1
                else:
                    i += 1
        
        logger.debug(f"[MemoryPlanner] Merged {merged_count} small allocations")
    
    def _get_node_index(self, node: Node) -> int:
        """获取节点在图中的索引"""
        for i, n in enumerate(self.graph.nodes):
            if n == node:
                return i
        return 0
    
    def _schedule_prefetch(self):
        """调度预取"""
        logger.debug("[MemoryPlanner] Scheduling prefetch...")
        
        prefetch_count = 0
        
        for assignment in self.assignments:
            if assignment.level == MemoryLevel.GLOBAL_MEMORY:
                # 计算预取距离
                prefetch_distance = self._compute_prefetch_distance(assignment)
                
                # 添加预取指令
                self._add_prefetch(assignment.node, prefetch_distance)
                prefetch_count += 1
        
        logger.debug(f"[MemoryPlanner] Added prefetch for {prefetch_count} nodes")
    
    def _compute_prefetch_distance(self, assignment: MemoryAssignment) -> int:
        """计算预取距离"""
        # 根据数据大小和带宽计算预取距离
        size_kb = assignment.allocation_size_bytes / 1024
        
        if size_kb < 64:
            return 2
        elif size_kb < 256:
            return 4
        else:
            return 8
    
    def _add_prefetch(self, node: Node, distance: int):
        """添加预取指令"""
        if "prefetch" not in node.meta:
            node.meta["prefetch"] = {}
        
        node.meta["prefetch"]["enabled"] = True
        node.meta["prefetch"]["distance"] = distance
    
    def _optimize_data_reuse(self):
        """优化数据重用"""
        logger.debug("[MemoryPlanner] Optimizing data reuse...")
        
        # 分析数据访问模式
        access_patterns = self._analyze_access_patterns()
        
        # 识别可以重用的数据
        reused_count = 0
        
        for node, pattern in access_patterns.items():
            if pattern["access_count"] >= 3:
                # 尝试将数据提升到更高层级的缓存
                current_level = self._get_assignment_level(node)
                if current_level and current_level != MemoryLevel.REGISTER:
                    # 尝试提升
                    higher_levels = [MemoryLevel.REGISTER, MemoryLevel.SHARED_MEMORY, 
                                    MemoryLevel.L1_CACHE, MemoryLevel.L2_CACHE]
                    
                    current_idx = higher_levels.index(current_level)
                    if current_idx > 0:
                        target_level = higher_levels[current_idx - 1]
                        if self._can_promote(node, target_level):
                            self._promote_to_level(node, target_level)
                            reused_count += 1
        
        logger.debug(f"[MemoryPlanner] Promoted {reused_count} nodes for better reuse")
    
    def _get_assignment_level(self, node: Node) -> Optional[MemoryLevel]:
        """获取节点的当前分配层级"""
        for assignment in self.assignments:
            if assignment.node == node:
                return assignment.level
        return None
    
    def _can_promote(self, node: Node, target_level: MemoryLevel) -> bool:
        """检查是否可以提升到目标层级"""
        assignment = next((a for a in self.assignments if a.node == node), None)
        if not assignment:
            return False
        
        size = assignment.allocation_size_bytes
        return self._can_allocate(target_level, size)
    
    def _promote_to_level(self, node: Node, target_level: MemoryLevel):
        """将节点提升到目标层级"""
        # 找到当前分配
        assignment = next((a for a in self.assignments if a.node == node), None)
        if not assignment:
            return
        
        # 释放当前层级的内存
        self.current_usage[assignment.level] -= assignment.allocation_size_bytes
        
        # 在新层级分配
        self._allocate(target_level, node, assignment.allocation_size_bytes, assignment.priority)
        
        # 更新分配记录
        assignment.level = target_level
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取内存使用摘要"""
        summary = {
            "total_assignments": len(self.assignments),
            "usage_by_level": {},
            "level_distribution": {},
        }
        
        for level in MemoryLevel:
            summary["usage_by_level"][level.value] = self.current_usage[level]
            summary["level_distribution"][level.value] = 0
        
        for assignment in self.assignments:
            summary["level_distribution"][assignment.level.value] += 1
        
        return summary


def plan_memory(graph_module: GraphModule, hierarchy: Optional[MemoryHierarchy] = None) -> GraphModule:
    """便捷函数：规划内存"""
    planner = MemoryPlanner(graph_module, hierarchy)
    return planner.plan()