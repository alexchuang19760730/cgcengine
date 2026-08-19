# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
Optimization Identifier - Step4: 智能优化识别器
基于图拓扑+硬件约束自动匹配最优优化组合
支持识别: Attention/MLP/KDA/FlashMoE
"""

from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import logging

from .graph_analyzer import GraphFeatures
from .space_builder import DeviceInfo
from .optimization_target_registry import (
    OPTIMIZATION_TARGET_REGISTRY,
    OptimizationTarget
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationMatch:
    """优化匹配结果"""
    opt_name: str
    confidence: float  # 0.0 - 1.0
    reason: str


class OptimizationIdentifier:
    """智能优化识别器"""
    
    @classmethod
    def identify(
        cls,
        graph_features: GraphFeatures,
        device_info: DeviceInfo,
        backend: str = "auto",
    ) -> List[str]:
        """主入口: 识别所有可用优化"""
        logger.info("[OptimizationIdentifier] 🎯 开始智能优化识别...")
        
        matches: List[OptimizationMatch] = []
        
        # 1. 硬件过滤
        hardware_valid = cls._filter_by_hardware(
            graph_features, device_info
        )
        matches.extend(hardware_valid)
        
        # 2. 图拓扑特征匹配
        topology_matches = cls._match_by_topology(
            graph_features, device_info
        )
        matches.extend(topology_matches)
        
        # 3. 后端专属优化
        backend_matches = cls._match_by_backend(
            backend, graph_features, device_info
        )
        matches.extend(backend_matches)
        
        # 4. 按置信度+优先级排序
        final_list = cls._rank_and_deduplicate(matches)
        
        logger.info(f"[OptimizationIdentifier] ✅ 共识别 {len(final_list)} 个优化")
        return final_list
    
    @classmethod
    def _filter_by_hardware(
        cls,
        graph_features: GraphFeatures,
        device_info: DeviceInfo,
    ) -> List[OptimizationMatch]:
        """按硬件过滤优化"""
        matches = []
        
        for opt_name, opt in OPTIMIZATION_TARGET_REGISTRY.items():
            if device_info.device_type not in opt.hardware_requirements:
                continue
            matches.append(OptimizationMatch(
                opt_name=opt_name,
                confidence=0.5,
                reason=f"硬件兼容: {device_info.device_type}"
            ))
        
        return matches
    
    @classmethod
    def _match_by_topology(
        cls,
        features: GraphFeatures,
        device_info: DeviceInfo,
    ) -> List[OptimizationMatch]:
        """基于图拓扑特征匹配优化"""
        matches = []
        
        if features.has_attention:
            matches.append(OptimizationMatch(
                opt_name="kda_attention",
                confidence=0.95,
                reason="检测到Attention层，启用KDA Attention替换"
            ))
        
        if features.has_moe:
            matches.append(OptimizationMatch(
                opt_name="flash_moe",
                confidence=0.90,
                reason="检测到MoE架构，启用FlashMoE"
            ))
        
        if features.has_vlm:
            matches.append(OptimizationMatch(
                opt_name="vlm_fusion",
                confidence=0.85,
                reason="检测到VLM多模态模型，启用多模态融合优化"
            ))
        
        # MLP相关优化
        matches.append(OptimizationMatch(
            opt_name="gemm_fusion",
            confidence=0.80,
            reason="通用GEMM融合"
        ))
        matches.append(OptimizationMatch(
            opt_name="rms_norm_fusion",
            confidence=0.80,
            reason="RMSNorm融合"
        ))
        matches.append(OptimizationMatch(
            opt_name="rope_fusion",
            confidence=0.80,
            reason="RoPE融合"
        ))
        
        return matches
    
    @classmethod
    def _match_by_backend(
        cls,
        backend: str,
        features: GraphFeatures,
        device_info: DeviceInfo,
    ) -> List[OptimizationMatch]:
        """后端专属优化"""
        matches = []
        
        # mlx-tune + Metal
        if backend in ["mlx-tune", "mlx_tune"] and device_info.device_type in ["metal", "mps"]:
            matches.append(OptimizationMatch(
                opt_name="tiling_64x64",
                confidence=0.98,
                reason="Apple Silicon专属64x64分块"
            ))
            matches.append(OptimizationMatch(
                opt_name="mtlheap_kv_cache",
                confidence=0.95,
                reason="Metal MTLHeap零拷贝KV缓存"
            ))
        
        # Megatrain + CUDA
        if backend in ["megatrain"] and device_info.device_type == "cuda":
            matches.append(OptimizationMatch(
                opt_name="tiling_128x128",
                confidence=0.95,
                reason="NVIDIA CUDA专属128x128分块"
            ))
        
        # oMLX/FlashMoE + CUDA
        if backend in ["omlx-flashmoe", "omlx_flashmoe", "flashmoe"] and device_info.device_type == "cuda":
            matches.append(OptimizationMatch(
                opt_name="flash_moe",
                confidence=0.98,
                reason="oMLX FlashMoE专家级优化（8专家Top2，GDS/SPDK SSD按需加载）"
            ))
        
        # GDS/SPDK + CUDA
        if backend in ["gds-spdk", "gds_spdk"] and device_info.device_type == "cuda":
            matches.append(OptimizationMatch(
                opt_name="gds_spdk",
                confidence=0.99,
                reason="NVIDIA GPUDirect Storage + SPDK异步IO，零拷贝专家权重/KV Cache SSD按需加载"
            ))
        
        return matches
    
    @classmethod
    def _rank_and_deduplicate(
        cls,
        matches: List[OptimizationMatch],
    ) -> List[str]:
        """排序和去重"""
        seen: Set[str] = set()
        unique = []
        
        # 先加，再排序
        for m in matches:
            if m.opt_name not in seen:
                seen.add(m.opt_name)
                unique.append(m)
        
        # 综合得分 = confidence * (1 + priority/100)
        def score(m: OptimizationMatch) -> float:
            opt = OPTIMIZATION_TARGET_REGISTRY.get(m.opt_name)
            p = opt.priority if opt else 50
            return m.confidence * (1 + p / 100)
        
        unique.sort(key=score, reverse=True)
        return [m.opt_name for m in unique]
