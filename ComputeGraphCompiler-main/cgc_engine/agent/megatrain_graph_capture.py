# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
Megatrain 整图捕获模块

功能：
- 使用 torch.compile 捕获训练模型的完整计算图
- 支持 FSDP 分布式训练图捕获
- 支持混合精度训练图捕获
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MegatrainGraphCaptureConfig:
    """Megatrain 图捕获配置"""
    
    enable_full_graph: bool = True
    enable_cudagraphs: bool = True
    enable_dynamic_shapes: bool = False
    
    training_mode: str = "fsdp"  # fsdp, ddp, standalone
    mixed_precision: str = "bf16"  # bf16, fp16, fp32
    
    fsdp_sharding_strategy: str = "full_shard"  # full_shard, shard_grad_op, no_shard
    fsdp_cpu_offload: bool = False
    
    max_batch_size: int = 32
    max_seq_len: int = 4096
    hidden_dim: int = 4096
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_full_graph": self.enable_full_graph,
            "enable_cudagraphs": self.enable_cudagraphs,
            "enable_dynamic_shapes": self.enable_dynamic_shapes,
            "training_mode": self.training_mode,
            "mixed_precision": self.mixed_precision,
            "fsdp_sharding_strategy": self.fsdp_sharding_strategy,
            "fsdp_cpu_offload": self.fsdp_cpu_offload,
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "hidden_dim": self.hidden_dim,
        }


class MegatrainGraphCapture:
    """Megatrain 整图捕获器"""
    
    def __init__(self, config: Optional[MegatrainGraphCaptureConfig] = None):
        self.config = config or MegatrainGraphCaptureConfig()
        self.captured_model = None
        self.graph_module = None
        self.dummy_inputs = None
        self.device: Optional[torch.device] = None
        
    def capture(
        self,
        model: nn.Module,
        input_shape: Optional[Tuple[int, ...]] = None,
        use_fsdp: bool = True,
        device: Optional[torch.device] = None,
    ) -> Tuple[nn.Module, torch.fx.GraphModule]:
        """
        捕获训练模型的完整计算图
        
        Args:
            model: PyTorch 模型
            input_shape: 输入张量形状 (batch_size, seq_len, hidden_dim)
            use_fsdp: 是否使用 FSDP 分布式
            
        Returns:
            (compiled_model, graph_module)
        """
        logger.info(f"[MegatrainGraphCapture] Starting graph capture...")
        
        # 准备输入
        if input_shape is None:
            input_shape = (
                self.config.max_batch_size,
                self.config.max_seq_len,
                self.config.hidden_dim
            )
        
        self.device = device or self._infer_device(model)
        self.dummy_inputs = self._create_dummy_inputs(input_shape, device=self.device)
        
        # 应用 FSDP 包装
        if use_fsdp and self.config.training_mode == "fsdp" and (self.device is not None and self.device.type == "cuda"):
            model = self._wrap_with_fsdp(model)
        
        # 应用混合精度
        model = self._apply_mixed_precision(model)
        
        # 使用 torch.compile 捕获完整图
        compiled_model = self._compile_model(model, self.dummy_inputs)
        
        # 提取图模块
        self.graph_module = self._extract_graph_module(compiled_model)
        self.captured_model = compiled_model
        
        logger.info(f"[MegatrainGraphCapture] Graph capture complete!")
        
        return compiled_model, self.graph_module
    
    def _create_dummy_inputs(self, input_shape: Tuple[int, ...], *, device: torch.device) -> Dict[str, torch.Tensor]:
        """创建虚拟输入"""
        batch_size, seq_len, hidden_dim = input_shape
        self.device = device
        
        dummy_inputs = {
            "input_ids": torch.randint(0, 32000, (batch_size, seq_len), device=self._device_str(), dtype=torch.long),
            "attention_mask": torch.ones(batch_size, seq_len, device=self._device_str()),
            "labels": torch.randint(0, 32000, (batch_size, seq_len), device=self._device_str(), dtype=torch.long),
        }
        
        return dummy_inputs

    def _infer_device(self, model: nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

    def _device_str(self) -> str:
        if self.device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return str(self.device)
    
    def _wrap_with_fsdp(self, model: nn.Module) -> nn.Module:
        """使用 FSDP 包装模型"""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy
            
            sharding_map = {
                "full_shard": ShardingStrategy.FULL_SHARD,
                "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
                "no_shard": ShardingStrategy.NO_SHARD,
            }
            
            sharding_strategy = sharding_map.get(
                self.config.fsdp_sharding_strategy,
                ShardingStrategy.FULL_SHARD
            )
            
            model = FSDP(
                model,
                sharding_strategy=sharding_strategy,
                cpu_offload=self.config.fsdp_cpu_offload,
            )
            
            logger.info(f"[MegatrainGraphCapture] Model wrapped with FSDP")
            
        except ImportError:
            logger.warning("[MegatrainGraphCapture] FSDP not available, using DDP")
            if dist.is_initialized():
                model = nn.parallel.DistributedDataParallel(model)
        
        return model
    
    def _apply_mixed_precision(self, model: nn.Module) -> nn.Module:
        """应用混合精度训练"""
        if self.config.mixed_precision == "bf16":
            model = model.to(torch.bfloat16)
        elif self.config.mixed_precision == "fp16":
            model = model.to(torch.float16)
        
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
                p.numel() for p in self.graph_module.parameters() if p.requires_grad
            ),
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
        
        logger.info(f"[MegatrainGraphCapture] Graph exported to {path}")


def capture_megatrain_graph(
    model: nn.Module,
    config: Optional[MegatrainGraphCaptureConfig] = None,
    **kwargs
) -> Tuple[nn.Module, torch.fx.GraphModule]:
    """
    便捷函数：捕获 Megatrain 训练图
    
    Args:
        model: PyTorch 模型
        config: 图捕获配置
        **kwargs: 额外配置参数
        
    Returns:
        (compiled_model, graph_module)
    """
    capturer = MegatrainGraphCapture(config)
    return capturer.capture(model, **kwargs)
