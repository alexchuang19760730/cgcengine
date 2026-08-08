# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
MLX-Tune 整图捕获模块

功能：
- 使用 torch.compile 捕获 LoRA/QLoRA 微调模型的完整计算图
- 支持 Apple Silicon Metal 后端图捕获
- 支持量化微调图捕获
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MLXTuneGraphCaptureConfig:
    """MLX-Tune 图捕获配置"""
    
    enable_full_graph: bool = True
    enable_metal_backend: bool = True
    enable_dynamic_shapes: bool = False
    
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    enable_qlora: bool = False
    qlora_bits: int = 4
    qlora_group_size: int = 64
    
    max_batch_size: int = 16
    max_seq_len: int = 2048
    hidden_dim: int = 4096
    
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_full_graph": self.enable_full_graph,
            "enable_metal_backend": self.enable_metal_backend,
            "enable_dynamic_shapes": self.enable_dynamic_shapes,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "enable_qlora": self.enable_qlora,
            "qlora_bits": self.qlora_bits,
            "qlora_group_size": self.qlora_group_size,
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "hidden_dim": self.hidden_dim,
            "target_modules": self.target_modules,
        }


class LoRALinear(nn.Module):
    """LoRA 线性层"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        result = x @ (self.dropout(self.lora_A) @ self.lora_B).T * self.scaling
        return result


class MLXTuneGraphCapture:
    """MLX-Tune 整图捕获器"""
    
    def __init__(self, config: Optional[MLXTuneGraphCaptureConfig] = None):
        self.config = config or MLXTuneGraphCaptureConfig()
        self.captured_model = None
        self.graph_module = None
        self.dummy_inputs = None
        self.lora_modules = []
        
    def capture(
        self,
        model: nn.Module,
        input_shape: Optional[Tuple[int, ...]] = None,
        use_metal: bool = True
    ) -> Tuple[nn.Module, torch.fx.GraphModule]:
        """
        捕获 LoRA/QLoRA 微调模型的完整计算图
        
        Args:
            model: PyTorch 模型
            input_shape: 输入张量形状 (batch_size, seq_len, hidden_dim)
            use_metal: 是否使用 Metal 后端
            
        Returns:
            (compiled_model, graph_module)
        """
        logger.info(f"[MLXTuneGraphCapture] Starting graph capture...")
        
        # 准备输入
        if input_shape is None:
            input_shape = (
                self.config.max_batch_size,
                self.config.max_seq_len,
                self.config.hidden_dim
            )
        
        self.dummy_inputs = self._create_dummy_inputs(input_shape)
        
        # 应用 LoRA 适配器
        model = self._apply_lora(model)
        
        # 应用 QLoRA 量化
        if self.config.enable_qlora:
            model = self._apply_qlora(model)
        
        # 应用 Metal 后端
        if use_metal and self.config.enable_metal_backend:
            model = model.to("mps")
            self.dummy_inputs = {k: v.to("mps") for k, v in self.dummy_inputs.items()}
        
        # 使用 torch.compile 捕获完整图
        compiled_model = self._compile_model(model, self.dummy_inputs)
        
        # 提取图模块
        self.graph_module = self._extract_graph_module(compiled_model)
        self.captured_model = compiled_model
        
        logger.info(f"[MLXTuneGraphCapture] Graph capture complete!")
        
        return compiled_model, self.graph_module
    
    def _create_dummy_inputs(self, input_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        """创建虚拟输入"""
        batch_size, seq_len, hidden_dim = input_shape
        
        dummy_inputs = {
            "input_ids": torch.randint(0, 32000, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "labels": torch.randint(0, 32000, (batch_size, seq_len)),
        }
        
        return dummy_inputs
    
    def _apply_lora(self, model: nn.Module) -> nn.Module:
        """应用 LoRA 适配器到目标模块"""
        for name, module in model.named_modules():
            if any(target in name for target in self.config.target_modules):
                if isinstance(module, nn.Linear):
                    lora = LoRALinear(
                        module.in_features,
                        module.out_features,
                        rank=self.config.lora_rank,
                        alpha=self.config.lora_alpha,
                        dropout=self.config.lora_dropout,
                    )
                    
                    # 冻结原始权重
                    for param in module.parameters():
                        param.requires_grad = False
                    
                    # 添加 LoRA 层
                    self.lora_modules.append((name, lora))
                    logger.info(f"[MLXTuneGraphCapture] Applied LoRA to {name}")
        
        return model
    
    def _apply_qlora(self, model: nn.Module) -> nn.Module:
        """应用 QLoRA 量化"""
        if not self.config.enable_qlora:
            return model
        
        try:
            from bitsandbytes import (
                replace_linear_with_lora_layer,
                prepare_model_for_kbit_training,
            )
            
            # 量化模型
            model = replace_linear_with_lora_layer(
                model,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bits=self.config.qlora_bits,
                group_size=self.config.qlora_group_size,
            )
            
            # 准备 kbit 训练
            model = prepare_model_for_kbit_training(model)
            
            logger.info(f"[MLXTuneGraphCapture] Applied QLoRA ({self.config.qlora_bits}-bit)")
            
        except ImportError:
            logger.warning("[MLXTuneGraphCapture] bitsandbytes not available, skipping QLoRA")
        
        return model
    
    def _compile_model(
        self,
        model: nn.Module,
        dummy_inputs: Dict[str, torch.Tensor]
    ) -> nn.Module:
        """使用 torch.compile 编译模型"""
        
        compile_kwargs = {
            "mode": "reduce-overhead",
            "fullgraph": self.config.enable_full_graph,
        }
        
        if self.config.enable_dynamic_shapes:
            compile_kwargs["dynamic"] = True
        
        compiled_model = torch.compile(model, **compile_kwargs)
        
        # 预热以触发图捕获
        with torch.no_grad():
            _ = compiled_model(**dummy_inputs)
        
        return compiled_model
    
    def _extract_graph_module(self, model: nn.Module) -> torch.fx.GraphModule:
        """提取图模块"""
        if hasattr(model, "_orig_mod"):
            return model._orig_mod
        elif hasattr(model, "_forward_module"):
            return model._forward_module
        else:
            return model
    
    def get_lora_params(self) -> List[nn.Parameter]:
        """获取 LoRA 可训练参数"""
        params = []
        for name, lora in self.lora_modules:
            params.extend([lora.lora_A, lora.lora_B])
        return params
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """获取图统计信息"""
        if self.graph_module is None:
            return {}

        graph = getattr(self.graph_module, "graph", None)
        if graph is None or not hasattr(graph, "nodes"):
            return {}

        nodes = list(graph.nodes)
        
        stats = {
            "num_nodes": len(nodes),
            "num_parameters": sum(p.numel() for p in self.graph_module.parameters()),
            "num_trainable_parameters": sum(
                p.numel() for p in self.get_lora_params()
            ),
            "num_lora_modules": len(self.lora_modules),
            "lora_rank": self.config.lora_rank,
            "node_types": {},
        }
        
        for node in nodes:
            op_type = node.op
            stats["node_types"][op_type] = stats["node_types"].get(op_type, 0) + 1
        
        return stats
    
    def export_graph(self, path: str) -> None:
        """导出计算图"""
        if self.graph_module is None:
            raise ValueError("No graph captured yet")
        
        import torch.fx
        
        # 导出为 Python 代码
        code = torch.fx.graph.python_code(self.graph_module.graph)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        
        logger.info(f"[MLXTuneGraphCapture] Graph exported to {path}")


def capture_mlx_tune_graph(
    model: nn.Module,
    config: Optional[MLXTuneGraphCaptureConfig] = None,
    **kwargs
) -> Tuple[nn.Module, torch.fx.GraphModule]:
    """
    便捷函数：捕获 MLX-Tune 微调图
    
    Args:
        model: PyTorch 模型
        config: 图捕获配置
        **kwargs: 额外配置参数
        
    Returns:
        (compiled_model, graph_module)
    """
    capturer = MLXTuneGraphCapture(config)
    return capturer.capture(model, **kwargs)
