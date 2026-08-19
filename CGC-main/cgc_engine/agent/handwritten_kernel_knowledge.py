# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
手写核知识库 - HandWrittenKernelKnowledge

核心大脑：学习 llama.cpp / CUDA / Metal 手写核的最佳实践
用于基于硬件和计算图特征自动决策融合 / Tiling / 内存 / 调度策略
"""

from typing import List, Dict, Any


class HandWrittenKernelKnowledge:
    """
    手写核知识库

    存储和提供不同硬件平台的最优算子融合、Tiling、内存和调度策略
    """

    def __init__(self):
        self.kernel_db = {
            "llama.cpp": self._load_llama_cpp_kernels(),
            "cuda": self._load_cuda_kernels(),
            "metal": self._load_metal_kernels(),
            "cpu": self._load_cpu_kernels(),
        }

    def _load_llama_cpp_kernels(self) -> Dict[str, Any]:
        """加载 llama.cpp 手写核知识"""
        return {
            "fusion_style": "block_fuse",
            "tile": (64, 64, 32),
            "prefetch": True,
            "memory_hint": {
                "qkv": "l1",
                "attn_output": "l1",
                "mlp_up": "l2",
                "final": "global"
            }
        }

    def _load_cuda_kernels(self) -> Dict[str, Any]:
        """加载 CUDA 手写核知识"""
        return {
            "fusion_style": "full_fuse",
            "tile": (128, 128, 32),
            "prefetch": True,
            "memory_hint": {
                "qkv": "register",
                "attn_output": "l1",
                "mlp_up": "l2",
                "final": "global"
            }
        }

    def _load_metal_kernels(self) -> Dict[str, Any]:
        """加载 Metal 手写核知识"""
        return {
            "fusion_style": "simd_fuse",
            "tile": (32, 32, 32),
            "prefetch": False,
            "memory_hint": {
                "qkv": "register",
                "attn_output": "l1",
                "mlp_up": "l2",
                "final": "global"
            }
        }

    def _load_cpu_kernels(self) -> Dict[str, Any]:
        """加载 CPU 手写核知识"""
        return {
            "fusion_style": "basic_fuse",
            "tile": (16, 16, 16),
            "prefetch": False,
            "memory_hint": {
                "qkv": "l1",
                "attn_output": "l2",
                "mlp_up": "l3",
                "final": "global"
            }
        }

    def best_fusion(self, graph, device: str) -> List[List[str]]:
        """
        最佳算子融合边界

        Args:
            graph: 计算图
            device: 目标设备

        Returns:
            融合区域列表
        """
        style = self.kernel_db.get(device, self.kernel_db["cpu"])["fusion_style"]

        if style == "simd_fuse":
            return [
                ["q_proj", "k_proj", "v_proj", "rope", "attn_split"],
                ["mlp_up", "mlp_act", "mlp_down"]
            ]
        elif style == "full_fuse":
            return [
                ["q_proj", "k_proj", "v_proj", "rope", "attn"],
                ["mlp_up", "mlp_act", "mlp_down"]
            ]
        elif style == "block_fuse":
            return [
                ["qkv_proj", "attn_rope", "attn"],
                ["mlp_up", "mlp_act", "mlp_down"]
            ]
        else:
            return [
                ["qkv_proj", "attn"],
                ["mlp"]
            ]

    def best_tile_size(
        self,
        graph,
        device: str,
        hw: Any
    ) -> Dict[str, int]:
        """
        最佳 Tile 大小

        Args:
            graph: 计算图
            device: 目标设备
            hw: HardwareConstraints 硬件约束

        Returns:
            Tiling 配置字典
        """
        base_tile = self.kernel_db.get(device, self.kernel_db["cpu"])["tile"]

        tile_m = min(base_tile[0], hw.max_tile_m)
        tile_n = min(base_tile[1], hw.max_tile_n)
        tile_k = min(base_tile[2], hw.max_tile_k)

        return {
            "Tile_M": tile_m,
            "Tile_N": tile_n,
            "Tile_K": tile_k,
            "Attn_Block": tile_m
        }

    def best_memory_liveout(self, graph) -> Dict[str, str]:
        """
        最佳内存层级规划

        Args:
            graph: 计算图

        Returns:
            各算子的内存层级配置
        """
        return {
            "q_proj": "register",
            "k_proj": "register",
            "v_proj": "register",
            "attn_output": "l1",
            "mlp_up": "l2",
            "final": "global"
        }

    def best_schedule(self, graph) -> Dict[str, Any]:
        """
        最佳调度方案

        Args:
            graph: 计算图

        Returns:
            调度策略配置
        """
        return {
            "unroll": 4,
            "prefetch": True,
            "simd_align": True,
            "pipeline": True
        }
