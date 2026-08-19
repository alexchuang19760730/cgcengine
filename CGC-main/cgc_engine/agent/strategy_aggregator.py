# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
策略聚合器 - MagiCompilerStrategy

统一管理所有策略层，提供单一配置入口和访问接口。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import logging

from .harness_agent import HarnessCompileStrategy, AgentOpHint
from .storage_layer.storage_optimizer import StorageStrategy
from .scheduling_layer.scheduler_optimizer import SchedulerStrategy
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class MagiCompilerStrategy:
    """
    MagiCompiler 顶层策略聚合器
    
    统一管理所有策略层，提供单一配置入口和访问接口。
    """

    # ========== 计算层策略 ==========
    compute: HarnessCompileStrategy = field(default_factory=HarnessCompileStrategy)

    # ========== 存储层策略 ==========
    storage: StorageStrategy = field(default_factory=StorageStrategy)

    # ========== 调度层策略 ==========
    scheduler: SchedulerStrategy = field(default_factory=SchedulerStrategy)

    # ========== 全局配置 ==========
    device: str = "auto"
    mode: str = "inference"  # inference, training, fine_tune
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """序列化为统一字典"""
        return {
            "compute": self.compute.to_dict(),
            "storage": self.storage.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "device": self.device,
            "mode": self.mode,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MagiCompilerStrategy":
        """从字典反序列化"""
        return cls(
            compute=HarnessCompileStrategy.from_dict(d.get("compute", {})),
            storage=StorageStrategy.from_dict(d.get("storage", {})),
            scheduler=SchedulerStrategy.from_dict(d.get("scheduler", {})),
            device=d.get("device", "auto"),
            mode=d.get("mode", "inference"),
            verbose=d.get("verbose", False),
        )

    def validate(self) -> bool:
        """验证所有策略配置是否合法"""
        valid = True
        if not self.compute.validate():
            logger.error("Compute strategy validation failed")
            valid = False
        if not self.storage.validate():
            logger.error("Storage strategy validation failed")
            valid = False
        if not self.scheduler.validate():
            logger.error("Scheduler strategy validation failed")
            valid = False
        return valid

    def optimize_for(self, scenario: str):
        """
        根据场景自动优化策略

        Args:
            scenario: "training", "inference", "video_gen", "moe", "mlx_finetune"
        """
        if scenario == "training":
            self._optimize_for_training()
        elif scenario == "inference":
            self._optimize_for_inference()
        elif scenario == "video_gen":
            self._optimize_for_video_gen()
        elif scenario == "moe":
            self._optimize_for_moe()
        elif scenario == "mlx_finetune":
            self._optimize_for_mlx_finetune()
        else:
            logger.warning(f"Unknown scenario: {scenario}")

    def _optimize_for_training(self):
        """优化策略用于训练"""
        self.mode = "training"
        
        # 计算层
        self.compute.recompute_config["enabled"] = True
        self.compute.megatrain_config["enabled"] = True
        self.compute.megatrain_config["training_mode"] = "fsdp"
        self.compute.megatrain_config["mixed_precision"] = "bf16"
        self.compute.tp_degree = 8
        self.compute.op_hints.append(AgentOpHint.MEGATRAIN)
        
        # 存储层
        self.storage.enable_kda = True
        self.storage.enable_prefetch = True
        
        # 调度层
        self.scheduler.enable_continuous_batching = False
        self.scheduler.enable_pd_separation = False

    def _optimize_for_inference(self):
        """优化策略用于推理"""
        self.mode = "inference"
        
        # 计算层
        self.compute.recompute_config["enabled"] = False
        self.compute.quantization_mode = "int8"
        self.compute.megatrain_config["enabled"] = False
        
        # 存储层
        self.storage.enable_kv_quant = True
        self.storage.kv_quant_bits = 8
        
        # 调度层
        self.scheduler.enable_continuous_batching = True
        self.scheduler.dynamic_batch_size = 64

    def _optimize_for_video_gen(self):
        """优化策略用于视频生成"""
        self.mode = "inference"
        
        # 计算层
        self.compute.enable_op_fusion = True
        self.compute.recompute_config["enabled"] = True
        
        # 存储层
        self.storage.enable_gds = True
        self.storage.memory_pool_size_mb = 4096.0
        
        # 调度层
        self.scheduler.prefill_chunk_size = 1024
        self.scheduler.enable_prefix_cache = True

    def _optimize_for_moe(self):
        """优化策略用于 MoE 模型"""
        self.mode = "inference"
        
        # 计算层
        self.compute.moe_config = {
            "num_experts": 8,
            "top_k": 2,
            "routing_impl": "cgc_routing",
        }
        self.compute.op_hints.append(AgentOpHint.MOE_ROUTING)
        
        # 存储层
        self.storage.kv_cache_max_size_mb = 8192.0
        
        # 调度层
        self.scheduler.max_batch_size = 256

    def _optimize_for_mlx_finetune(self):
        """优化策略用于 MLX LoRA 微调"""
        self.mode = "fine_tune"
        self.device = "metal"
        
        # 计算层
        self.compute.mlx_tune_config["enabled"] = True
        self.compute.mlx_tune_config["lora_rank"] = 8
        self.compute.mlx_tune_config["lora_alpha"] = 16.0
        self.compute.mlx_tune_config["target_modules"] = ["q_proj", "v_proj"]
        self.compute.op_hints.append(AgentOpHint.MLX_TUNE)
        
        # 存储层
        self.storage.memory_layout = "flat"
        self.storage.enable_memory_pooling = True
        
        # 调度层
        self.scheduler.enable_continuous_batching = False
        self.scheduler.dynamic_batch_size = 8

    def enable_feature(self, feature: str, enabled: bool = True):
        """
        启用/禁用特定功能

        Args:
            feature: "flash_attention", "moe", "recompute", "megatrain", "mlx_tune"
            enabled: 是否启用
        """
        if feature == "flash_attention":
            if enabled:
                self.compute.op_hints.append(AgentOpHint.FLASH_ATTENTION)
            else:
                self.compute.op_hints = [h for h in self.compute.op_hints 
                                        if h != AgentOpHint.FLASH_ATTENTION]
        elif feature == "moe":
            self.compute.op_hints.append(AgentOpHint.MOE_ROUTING)
        elif feature == "recompute":
            self.compute.recompute_config["enabled"] = enabled
        elif feature == "megatrain":
            self.compute.megatrain_config["enabled"] = enabled
            if enabled:
                self.compute.op_hints.append(AgentOpHint.MEGATRAIN)
            else:
                self.compute.op_hints = [h for h in self.compute.op_hints 
                                        if h != AgentOpHint.MEGATRAIN]
        elif feature == "mlx_tune":
            self.compute.mlx_tune_config["enabled"] = enabled
            if enabled:
                self.compute.op_hints.append(AgentOpHint.MLX_TUNE)
            else:
                self.compute.op_hints = [h for h in self.compute.op_hints 
                                        if h != AgentOpHint.MLX_TUNE]
        else:
            logger.warning(f"Unknown feature: {feature}")

    def merge(self, other: "MagiCompilerStrategy") -> None:
        """合并另一个策略聚合器的配置"""
        self.compute.merge(other.compute)
        self.storage.merge(other.storage)
        self.scheduler.merge(other.scheduler)
        if other.device:
            self.device = other.device
        if other.mode:
            self.mode = other.mode
        self.verbose = other.verbose