# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
计算图特征分析器 - GraphAnalyzer

功能：
- 分析 PyTorch 模型的计算图结构
- 检测 Flash Attention、MoE、VLM 等高级特征
- 识别可优化的模式
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class GraphFeatures:
    """计算图特征"""
    has_attention: bool = False
    has_flash_attention: bool = False
    has_moe: bool = False
    has_vlm: bool = False
    has_tensor_parallel: bool = False
    has_pipeline_parallel: bool = False
    
    num_layers: int = 0
    hidden_dim: int = 0
    num_heads: int = 0
    seq_len_limit: int = 0
    
    small_gemm: bool = False
    large_gemm: bool = False
    
    attention_patterns: List[str] = field(default_factory=list)
    moe_patterns: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphAnalyzer:
    """计算图分析器"""

    @classmethod
    def analyze(cls, model: nn.Module, inputs: Optional[Tuple] = None) -> GraphFeatures:
        """
        分析计算图特征

        Args:
            model: PyTorch 模型
            inputs: 输入张量（可选）

        Returns:
            GraphFeatures
        """
        features = GraphFeatures()

        # 分析模块结构
        cls._analyze_modules(model, features)

        # 尝试通过 inputs 做完整图分析
        if inputs is not None:
            try:
                cls._analyze_graph_trace(model, inputs, features)
            except Exception as e:
                logger.warning(f"Graph tracing failed: {e}")

        logger.info(
            f"[GraphAnalyzer] Analysis complete: "
            f"attention={features.has_attention}, "
            f"flash_attn={features.has_flash_attention}, "
            f"moe={features.has_moe}, "
            f"vlm={features.has_vlm}"
        )

        return features

    @staticmethod
    def _analyze_modules(model: nn.Module, features: GraphFeatures):
        """通过遍历子模块分析图"""
        module_dict = dict(model.named_modules())
        for name, module in module_dict.items():
            name_lower = name.lower()

            # 检测 Attention
            if any(k in name_lower for k in ["attention", "attn", "sdpa"]):
                features.has_attention = True
                if any(k in name_lower for k in ["flash", "paged"]):
                    features.has_flash_attention = True
                    features.attention_patterns.append(name)

            # 检测 MoE
            if any(k in name_lower for k in ["moe", "expert", "mixture"]):
                features.has_moe = True
                features.moe_patterns.append(name)

            # 检测 Vision/VLM
            if any(k in name_lower for k in ["vision", "clip", "image", "vit"]):
                features.has_vlm = True

            # 检测并行通信
            if any(k in name_lower for k in ["all_reduce", "all_gather", "reduce_scatter"]):
                features.has_tensor_parallel = True

            # 检测线性层
            if isinstance(module, nn.Linear):
                if features.hidden_dim == 0:
                    features.hidden_dim = module.in_features
                if module.in_features <= 1024 and module.out_features <= 1024:
                    features.small_gemm = True
                if module.in_features >= 4096 or module.out_features >= 4096:
                    features.large_gemm = True

            # 计数层数
            if any(k in name_lower for k in ["layer", "block", "transformer.h"]) and "." in name:
                try:
                    layer_idx = int(name.split(".")[-1])
                    if layer_idx >= features.num_layers:
                        features.num_layers = layer_idx + 1
                except:
                    pass

            # 查找 num_heads
            if "num_heads" in name_lower or "n_head" in name_lower:
                if hasattr(module, "num_heads"):
                    features.num_heads = module.num_heads
                elif hasattr(module, "n_head"):
                    features.num_heads = module.n_head

        # 尝试从主模型对象找属性
        for attr_name in dir(model):
            if "num_layers" in attr_name:
                try:
                    features.num_layers = max(features.num_layers, getattr(model, attr_name))
                except:
                    pass
            if "num_heads" in attr_name or "n_head" in attr_name:
                try:
                    features.num_heads = max(features.num_heads, getattr(model, attr_name))
                except:
                    pass
            if "hidden_dim" in attr_name or "hidden_size" in attr_name:
                try:
                    features.hidden_dim = max(features.hidden_dim, getattr(model, attr_name))
                except:
                    pass

    @staticmethod
    def _analyze_graph_trace(model: nn.Module, inputs: Tuple, features: GraphFeatures):
        """通过 torch.jit.trace 分析实际计算图"""
        try:
            model.eval()
            with torch.no_grad():
                # 简单 trace
                scripted = torch.jit.trace(model, inputs)
                graph = scripted.graph
                graph_str = str(graph)

                # 从图中找关键字
                if "scaled_dot_product" in graph_str or "flash" in graph_str.lower():
                    features.has_flash_attention = True

                if "all_reduce" in graph_str or "all_gather" in graph_str:
                    features.has_tensor_parallel = True

                features.metadata["trace_available"] = True

        except Exception as e:
            logger.debug(f"Detailed trace not available: {e}")
            features.metadata["trace_available"] = False
