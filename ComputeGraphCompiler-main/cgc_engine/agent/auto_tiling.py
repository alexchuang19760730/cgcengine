# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
Auto-Tiling 模块 - AutoTiling

功能：
- 按硬件约束搜索最佳分块
- 支持多种硬件架构
- 自动优化分块策略
"""

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
import logging
import itertools

logger = logging.getLogger(__name__)


@dataclass
class HardwareConstraints:
    """硬件约束（自动检测版）"""
    max_shared_memory_bytes: int = 48 * 1024  # 默认 48KB
    max_registers: int = 65536
    warp_size: int = 32
    sm_count: int = 108
    max_threads_per_block: int = 1024
    max_threads_per_sm: int = 2048
    compute_capability: Tuple[int, int] = (8, 0)
    has_tensor_cores: bool = True
    has_fp8_support: bool = True
    
    @classmethod
    def auto_detect(cls) -> 'HardwareConstraints':
        """从实际硬件自动检测约束"""
        constraints = cls()
        
        if torch.cuda.is_available():
            try:
                prop = torch.cuda.get_device_properties(torch.cuda.current_device())
                
                # CUDA 设备属性
                constraints.sm_count = prop.multi_processor_count
                constraints.compute_capability = (prop.major, prop.minor)
                constraints.has_tensor_cores = prop.major >= 7
                constraints.has_fp8_support = prop.major >= 8
                
                # 根据架构设置默认值
                if prop.major >= 8:  # Ampere 及以上
                    constraints.max_shared_memory_bytes = 163840  # 160KB (可配置)
                    constraints.max_registers = 65536
                    constraints.max_threads_per_sm = 2048
                elif prop.major == 7:  # Volta
                    constraints.max_shared_memory_bytes = 98304  # 96KB
                    constraints.max_registers = 65536
                    constraints.max_threads_per_sm = 2048
                elif prop.major == 6:  # Pascal
                    constraints.max_shared_memory_bytes = 65536  # 64KB
                    constraints.max_registers = 65536
                    constraints.max_threads_per_sm = 2048
                
                logger.info(f"[HardwareConstraints] Auto-detected CUDA device: {prop.name}")
                logger.info(f"  SM Count: {constraints.sm_count}")
                logger.info(f"  Compute Capability: {constraints.compute_capability}")
                logger.info(f"  Shared Memory: {constraints.max_shared_memory_bytes} bytes")
            
            except Exception as e:
                logger.warning(f"Failed to auto-detect CUDA constraints: {e}")
        
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon Metal 后端
            constraints.warp_size = 32  # SIMD group size
            constraints.max_threads_per_block = 1024
            constraints.max_shared_memory_bytes = 32768  # 32KB shared memory per threadgroup
            constraints.sm_count = torch.backends.mps.device_count() or 8  # 估计值
            
            logger.info("[HardwareConstraints] Auto-detected Metal device")
        
        else:
            # CPU 回退
            constraints.max_shared_memory_bytes = 0
            constraints.max_threads_per_block = 256
            constraints.warp_size = 1
            constraints.sm_count = max(1, torch.get_num_threads())
            
            logger.info("[HardwareConstraints] Using CPU defaults")
        
        return constraints
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_shared_memory_bytes": self.max_shared_memory_bytes,
            "max_registers": self.max_registers,
            "warp_size": self.warp_size,
            "sm_count": self.sm_count,
            "max_threads_per_block": self.max_threads_per_block,
            "max_threads_per_sm": self.max_threads_per_sm,
            "compute_capability": self.compute_capability,
            "has_tensor_cores": self.has_tensor_cores,
            "has_fp8_support": self.has_fp8_support,
        }


@dataclass
class TileConfig:
    """分块配置"""
    M: int = 128
    N: int = 128
    K: int = 128
    block_M: int = 32
    block_N: int = 32
    block_K: int = 8
    num_stages: int = 2
    unroll_factor: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "block_M": self.block_M,
            "block_N": self.block_N,
            "block_K": self.block_K,
            "num_stages": self.num_stages,
            "unroll_factor": self.unroll_factor,
        }


@dataclass
class TileCandidate:
    """分块候选"""
    config: TileConfig
    estimated_time_ms: float = float('inf')
    shared_memory_usage: int = 0
    register_usage: int = 0
    valid: bool = True


class AutoTiling:
    """自动分块优化器（硬件感知版）"""
    
    def __init__(self, graph_module: GraphModule, constraints: Optional[HardwareConstraints] = None):
        self.graph_module = graph_module
        self.graph = graph_module.graph
        # 默认使用自动检测的硬件约束
        self.constraints = constraints or HardwareConstraints.auto_detect()
        self.tile_configs: Dict[Node, TileConfig] = {}
        logger.info(f"[AutoTiling] Using hardware constraints: {self.constraints.to_dict()}")
    
    def optimize(self) -> GraphModule:
        """执行自动分块优化"""
        logger.info("[AutoTiling] Starting automatic tiling optimization...")
        
        # 1. 识别需要分块的节点
        matmul_nodes = self._identify_matmul_nodes()
        
        # 2. 为每个节点搜索最佳分块
        for node in matmul_nodes:
            best_config = self._search_best_tile_config(node)
            self.tile_configs[node] = best_config
        
        # 3. 应用分块配置
        self._apply_tile_configs()
        
        logger.info(f"[AutoTiling] Tiling complete! Optimized {len(self.tile_configs)} nodes")
        
        return self.graph_module
    
    def _identify_matmul_nodes(self) -> List[Node]:
        """识别需要分块的节点"""
        matmul_nodes = []
        
        for node in self.graph.nodes:
            op_name = str(node.target)
            if "matmul" in op_name.lower() or "linear" in op_name.lower():
                matmul_nodes.append(node)
        
        return matmul_nodes
    
    def _search_best_tile_config(self, node: Node) -> TileConfig:
        """搜索最佳分块配置"""
        candidates = self._generate_candidates(node)
        valid_candidates = [c for c in candidates if c.valid]
        
        if not valid_candidates:
            return TileConfig()
        
        # 选择估计时间最短的配置
        best = min(valid_candidates, key=lambda c: c.estimated_time_ms)
        logger.debug(f"Selected tile config: {best.config.to_dict()}")
        
        return best.config
    
    def _generate_candidates(self, node: Node) -> List[TileCandidate]:
        """生成分块候选"""
        candidates = []
        
        # 获取输入形状
        input_shapes = self._get_input_shapes(node)
        if not input_shapes:
            return [TileCandidate(config=TileConfig())]
        
        M, K, N = self._infer_matmul_dimensions(input_shapes)
        
        # 定义搜索空间
        block_M_options = [16, 32, 64, 128]
        block_N_options = [16, 32, 64, 128]
        block_K_options = [8, 16, 32, 64]
        num_stages_options = [1, 2, 4]
        unroll_factor_options = [2, 4, 8]
        
        # 生成所有组合
        for block_M, block_N, block_K, stages, unroll in itertools.product(
            block_M_options,
            block_N_options,
            block_K_options,
            num_stages_options,
            unroll_factor_options,
        ):
            config = TileConfig(
                M=M,
                N=N,
                K=K,
                block_M=block_M,
                block_N=block_N,
                block_K=block_K,
                num_stages=stages,
                unroll_factor=unroll,
            )
            
            candidate = self._evaluate_candidate(config, M, N, K)
            candidates.append(candidate)
        
        return candidates
    
    def _get_input_shapes(self, node: Node) -> List[torch.Size]:
        """获取输入形状"""
        shapes = []
        
        for arg in node.args:
            if isinstance(arg, Node):
                try:
                    val = arg.meta.get("val")
                    if val is not None:
                        shapes.append(val.shape)
                except:
                    pass
        
        return shapes
    
    def _infer_matmul_dimensions(self, shapes: List[torch.Size]) -> Tuple[int, int, int]:
        """推断矩阵乘法维度"""
        if len(shapes) >= 2:
            a_shape = shapes[0]
            b_shape = shapes[1]
            
            if len(a_shape) >= 2 and len(b_shape) >= 2:
                M = a_shape[-2]
                K = a_shape[-1]
                N = b_shape[-1]
                return M, K, N
        
        return 1024, 1024, 1024  # 默认值
    
    def _evaluate_candidate(self, config: TileConfig, M: int, N: int, K: int) -> TileCandidate:
        """评估分块候选"""
        candidate = TileCandidate(config=config)
        
        # 计算共享内存使用
        shared_usage = self._calculate_shared_memory_usage(config)
        candidate.shared_memory_usage = shared_usage
        
        # 计算寄存器使用
        register_usage = self._calculate_register_usage(config)
        candidate.register_usage = register_usage
        
        # 检查约束
        if shared_usage > self.constraints.max_shared_memory_bytes:
            candidate.valid = False
        
        if register_usage > self.constraints.max_registers:
            candidate.valid = False
        
        if not candidate.valid:
            return candidate
        
        # 估算执行时间
        candidate.estimated_time_ms = self._estimate_execution_time(config, M, N, K)
        
        return candidate
    
    def _calculate_shared_memory_usage(self, config: TileConfig) -> int:
        """计算共享内存使用"""
        # 两个矩阵块 + 结果块
        elem_size = 4  # float32
        return (config.block_M * config.block_K + 
                config.block_K * config.block_N + 
                config.block_M * config.block_N) * elem_size * config.num_stages
    
    def _calculate_register_usage(self, config: TileConfig) -> int:
        """计算寄存器使用"""
        # 粗略估算
        threads_per_block = config.block_M * config.block_N // 32
        return threads_per_block * 64  # 假设每个线程约64个寄存器
    
    def _estimate_execution_time(self, config: TileConfig, M: int, N: int, K: int) -> float:
        """估算执行时间"""
        # 计算总 FLOPs
        flops = 2 * M * N * K
        
        # 计算内存带宽需求
        bytes_loaded = (M * K + K * N) * 4  # float32
        bytes_stored = M * N * 4
        
        # 计算算术强度
        arithmetic_intensity = flops / (bytes_loaded + bytes_stored)
        
        # 假设硬件峰值性能和带宽
        peak_flops = 1e15  # 1 TFLOPS
        peak_bandwidth = 1e12  # 1 TB/s
        
        # Roofline 模型估算
        flop_bound_time = flops / peak_flops * 1000  # ms
        bandwidth_bound_time = (bytes_loaded + bytes_stored) / peak_bandwidth * 1000  # ms
        
        # 考虑分块效率
        efficiency = self._estimate_tiling_efficiency(config)
        
        return max(flop_bound_time, bandwidth_bound_time) / efficiency
    
    def _estimate_tiling_efficiency(self, config: TileConfig) -> float:
        """估算分块效率"""
        # 基于分块大小的启发式效率估计
        if config.block_M >= 64 and config.block_N >= 64:
            return 0.9
        elif config.block_M >= 32 and config.block_N >= 32:
            return 0.75
        else:
            return 0.5
    
    def _apply_tile_configs(self):
        """应用分块配置"""
        # 将分块配置存储到节点元数据中
        for node, config in self.tile_configs.items():
            if "tile_config" not in node.meta:
                node.meta["tile_config"] = {}
            node.meta["tile_config"].update(config.to_dict())


def auto_tile(graph_module: GraphModule, constraints: Optional[HardwareConstraints] = None) -> GraphModule:
    """便捷函数：自动分块"""
    tiler = AutoTiling(graph_module, constraints)
    return tiler.optimize()