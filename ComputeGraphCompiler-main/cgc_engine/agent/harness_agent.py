# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
Harness Agent - 策略决策核心

功能：
- 基于特征的策略决策（启发式 + 学习）
- 输出编译策略到 CGC SIMD 引擎
- 策略验证与应用
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
import enum
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class AgentOpHint(enum.Enum):
    """Agent 给 CGC SIMD 引擎的操作提示"""
    AUTO = "auto"
    FLASH_ATTENTION = "flash_attention"
    MOE_ROUTING = "moe_routing"
    TENSOR_PARALLEL = "tensor_parallel"
    VLM_CROSS_ATTENTION = "vlm_cross_attention"
    RECOMPUTE = "recompute"  # 启发式重计算
    MEGATRAIN = "megatrain"  # 训练模式
    MLX_TUNE = "mlx_tune"    # MLX LoRA 微调


@dataclass
class HarnessCompileStrategy(BaseStrategy):
    """Harness Agent 决策后的编译策略"""
    enable_op_fusion: bool = True
    fusion_regions: List[List[str]] = field(default_factory=list)
    tile_sizes: Dict[str, int] = field(default_factory=dict)
    memory_layouts: Dict[str, str] = field(default_factory=dict)
    schedules: Dict[str, Any] = field(default_factory=dict)
    backend: str = "auto"
    backend_overrides: Dict[str, str] = field(default_factory=dict)
    quantization_mode: str = "auto"
    tp_degree: int = 1
    pp_degree: int = 1
    moe_config: Optional[Dict[str, Any]] = None
    attention_config: Optional[Dict[str, Any]] = None
    vlm_config: Optional[Dict[str, Any]] = None
    
    # === Metal专属优化策略 ===
    enable_tiling_64x64: bool = True
    enable_mtlheap_kv_cache: bool = True
    
    # === 量化优化策略 ===
    enable_int4_quantization: bool = True
    
    # 启发式自动重计算配置
    recompute_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "enabled": True,
        "mode": "heuristic",  # heuristic, always, never
        "preserve_ops": ["matmul", "attention", "layer_norm"],  # 保留计算密集型算子
        "recompute_ops": ["activation", "dropout", "add"],  # 重算显存密集算子
        "threshold_mb": 1024,  # 显存阈值触发重算
        "min_compute_ratio": 0.8,  # 计算/显存比阈值
        "full_graph": False,  # 是否整图重算
    })
    
    # Megatrain 训练策略配置
    megatrain_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "enabled": False,
        "training_mode": "fsdp",  # fsdp, ddp, data_parallel
        "mixed_precision": "bf16",  # fp32, fp16, bf16
        "gradient_accumulation_steps": 1,
        "enable_gradient_checkpointing": True,
        "fsdp_sharding_strategy": "full_shard",  # full_shard, shard_grad_op, no_shard
        "fsdp_use_orig_params": True,
        "enable_activation_checkpointing": True,
        "checkpoint_granularity": "full",  # full, selective, nothing
    })
    
    # MLX-Tune LoRA 微调策略配置
    mlx_tune_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "enabled": False,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "enable_qlora": False,
        "qlora_bits": 4,  # 4, 8
        "quant_type": "nf4",  # nf4, fp4, int4
        "compute_dtype": "bf16",  # fp32, fp16, bf16
        "adapter_path": None,
    })
    
    # 整图捕获配置
    graph_capture_config: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "enable_full_graph": True,
        "enable_cudagraphs": True,
        "enable_dynamic_shapes": False,
        "capture_mode": "auto",  # auto, megatrain, mlx_tune, inference
        "export_graph": False,
        "export_path": None,
    })
    
    op_hints: List[AgentOpHint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "enable_op_fusion": self.enable_op_fusion,
            "fusion_regions": self.fusion_regions,
            "tile_sizes": self.tile_sizes,
            "memory_layouts": self.memory_layouts,
            "schedules": self.schedules,
            "backend": self.backend,
            "backend_overrides": self.backend_overrides,
            "quantization_mode": self.quantization_mode,
            "tp_degree": self.tp_degree,
            "pp_degree": self.pp_degree,
            "moe_config": self.moe_config,
            "attention_config": self.attention_config,
            "vlm_config": self.vlm_config,
            "enable_tiling_64x64": self.enable_tiling_64x64,
            "enable_mtlheap_kv_cache": self.enable_mtlheap_kv_cache,
            "enable_int4_quantization": self.enable_int4_quantization,
            "recompute_config": self.recompute_config,
            "megatrain_config": self.megatrain_config,
            "mlx_tune_config": self.mlx_tune_config,
            "graph_capture_config": self.graph_capture_config,
            "op_hints": [hint.value for hint in self.op_hints],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HarnessCompileStrategy":
        """从字典反序列化"""
        strategy = cls()
        for k, v in d.items():
            if k == "op_hints":
                try:
                    strategy.op_hints = [AgentOpHint(h) for h in v]
                except Exception:
                    strategy.op_hints = []
                continue
            if hasattr(strategy, k):
                setattr(strategy, k, v)
        return strategy

    def validate(self) -> bool:
        """验证策略配置是否合法"""
        if self.tp_degree < 1:
            logger.error("TP degree must be >= 1")
            return False
        if self.pp_degree < 1:
            logger.error("PP degree must be >= 1")
            return False
        if self.recompute_config and self.recompute_config.get("enabled"):
            if self.recompute_config.get("threshold_mb", 0) < 0:
                logger.error("Recompute threshold must be non-negative")
                return False
        if self.megatrain_config and self.megatrain_config.get("enabled"):
            if self.megatrain_config.get("gradient_accumulation_steps", 1) < 1:
                logger.error("Gradient accumulation steps must be >= 1")
                return False
        return True

    def merge(self, other: "HarnessCompileStrategy") -> None:
        """合并另一个策略的配置"""
        for key, value in asdict(other).items():
            if value is not None:
                setattr(self, key, value)


class HarnessAgent:
    """Harness Agent - 编译策略智能决策"""

    def __init__(
        self,
        device: Optional[str] = None,
        enable_llama_cpp_reference: bool = True,
        enable_vllm_reference: bool = True,
        enable_heuristic: bool = True,
        enable_learning: bool = False,  # 学习模式暂未实现
    ):
        """
        初始化 Harness Agent

        Args:
            device: 设备
            enable_llama_cpp_reference: 是否参考 llama.cpp 实现
            enable_vllm_reference: 是否参考 vLLM 实现
            enable_heuristic: 是否启用启发式规则
            enable_learning: 是否启用学习模式
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.enable_llama_cpp_reference = enable_llama_cpp_reference
        self.enable_vllm_reference = enable_vllm_reference
        self.enable_heuristic = enable_heuristic
        self.enable_learning = enable_learning
        self.compilation_history: List[Dict[str, Any]] = []

        logger.info(
            f"[HarnessAgent] Initialized: device={device}, "
            f"llama_cpp_ref={enable_llama_cpp_reference}, "
            f"vllm_ref={enable_vllm_reference}"
        )

    def decide(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        graph_features: Optional[Any] = None,
        optimization_space: Optional[Any] = None,
        user_hints: Optional[Dict[str, Any]] = None,
    ) -> HarnessCompileStrategy:
        """
        决策：输出编译策略

        Args:
            model: PyTorch 模型
            input_shape: 输入形状
            graph_features: GraphFeatures（可选）
            optimization_space: OptimizationSpace（可选）
            user_hints: 用户显式提示（可选）

        Returns:
            HarnessCompileStrategy
        """
        from .graph_analyzer import GraphAnalyzer, GraphFeatures
        from .space_builder import OptimizationSpaceBuilder, OptimizationSpace

        if graph_features is None:
            # 分析图特征
            logger.info("[HarnessAgent] Analyzing graph...")
            graph_features = GraphAnalyzer.analyze(model)

        if optimization_space is None:
            # 构建优化空间
            logger.info("[HarnessAgent] Building optimization space...")
            optimization_space = OptimizationSpaceBuilder.build(model, input_shape, self.device)

        logger.info("[HarnessAgent] Deciding strategy...")
        strategy = HarnessCompileStrategy()

        # 合并用户 hints
        if user_hints:
            if "tensor_parallel_size" in user_hints:
                try:
                    strategy.tp_degree = int(user_hints["tensor_parallel_size"])
                except Exception:
                    pass
            if "enable_tiling_64x64" in user_hints:
                strategy.enable_tiling_64x64 = bool(user_hints["enable_tiling_64x64"])
            if "enable_mtlheap_kv_cache" in user_hints:
                strategy.enable_mtlheap_kv_cache = bool(user_hints["enable_mtlheap_kv_cache"])
            if "enable_int4_quantization" in user_hints:
                strategy.enable_int4_quantization = bool(user_hints["enable_int4_quantization"])

            if user_hints.get("enable_recompute"):
                strategy.recompute_config = {
                    "enabled": True,
                    "mode": user_hints.get("recompute_mode", "heuristic"),
                    "preserve_ops": ["matmul", "attention", "layer_norm"],
                    "recompute_ops": ["activation", "dropout", "add"],
                    "threshold_mb": int(user_hints.get("recompute_threshold", 1024)),
                    "min_compute_ratio": float(user_hints.get("recompute_min_ratio", 0.8)),
                    "full_graph": False,
                }

            if user_hints.get("enable_megatrain"):
                strategy.megatrain_config = strategy.megatrain_config or {}
                strategy.megatrain_config.update(
                    {
                        "enabled": True,
                        "training_mode": user_hints.get("megatrain_mode", "fsdp"),
                        "mixed_precision": user_hints.get("mixed_precision", "bf16"),
                    }
                )

            if user_hints.get("enable_mlx_tune"):
                strategy.mlx_tune_config = strategy.mlx_tune_config or {}
                strategy.mlx_tune_config.update(
                    {
                        "enabled": True,
                        "lora_rank": int(user_hints.get("lora_rank", 8)),
                        "lora_alpha": float(user_hints.get("lora_alpha", 16.0)),
                        "enable_qlora": bool(user_hints.get("enable_qlora", False)),
                        "qlora_bits": int(user_hints.get("qlora_bits", 4)),
                    }
                )

            if "enable_cuda_graphs" in user_hints or "enable_cuda_graph" in user_hints:
                graph_capture_config = strategy.graph_capture_config or {}
                if "enable_cuda_graphs" in user_hints:
                    graph_capture_config["enable_cudagraphs"] = bool(user_hints["enable_cuda_graphs"])
                if "enable_cuda_graph" in user_hints:
                    graph_capture_config["enable_cudagraphs"] = bool(user_hints["enable_cuda_graph"])
                strategy.graph_capture_config = graph_capture_config

            if user_hints.get("enable_flash_attn"):
                if not getattr(graph_features, "has_flash_attention", False):
                    try:
                        graph_features.has_flash_attention = True
                    except Exception:
                        pass
            if user_hints.get("enable_moe"):
                if not getattr(graph_features, "has_moe", False):
                    try:
                        graph_features.has_moe = True
                    except Exception:
                        pass

        # 启发式决策
        if self.enable_heuristic:
            self._apply_heuristics(strategy, graph_features, optimization_space, user_hints)

        # 参考 llama.cpp 优化
        if self.enable_llama_cpp_reference:
            self._apply_llama_cpp_reference(strategy, graph_features, optimization_space)

        # 参考 vLLM 优化
        if self.enable_vllm_reference:
            self._apply_vllm_reference(strategy, graph_features, optimization_space)

        # 记录决策
        self._record_decision(strategy, graph_features, optimization_space)

        logger.info(
            f"[HarnessAgent] Decision complete: "
            f"fusion={strategy.enable_op_fusion}, "
            f"backend={strategy.backend}, "
            f"moe={bool(strategy.moe_config)}, "
            f"flash_attn={bool(strategy.attention_config)}"
        )

        return strategy

    def _apply_heuristics(
        self,
        strategy: HarnessCompileStrategy,
        features: Any,
        space: Any,
        user_hints: Optional[Dict],
    ):
        """应用启发式规则"""
        import platform
        import torch

        # 默认后端（auto）
        strategy.backend = "auto"
        device_str = str(self.device).lower()

        if device_str.startswith("cuda"):
            strategy.backend = "cuda"
        elif device_str in ["metal", "mps"]:
            strategy.backend = "metal"
        elif platform.system().lower() == "darwin":
            if os.environ.get("CGC_REQUIRE_MLX") == "1":
                strategy.backend = "metal"
            else:
                strategy.backend = "metal"
        elif device_str == "cpu":
            raise RuntimeError("STRICT MODE: CPU backend is strictly PROHIBITED.")
        elif platform.system().lower() == "linux":
            if torch.cuda.is_available():
                strategy.backend = "cuda"
            else:
                raise RuntimeError("STRICT MODE: CUDA not available on Linux. CPU backend is strictly PROHIBITED.")

        # 根据设备选 tiling size
        if strategy.backend == "cuda":
            strategy.tile_sizes["M"] = 128
            strategy.tile_sizes["N"] = 128
            strategy.tile_sizes["K"] = 128
        else:
            strategy.tile_sizes["M"] = 64
            strategy.tile_sizes["N"] = 64
            strategy.tile_sizes["K"] = 64
        
        # Metal专属优化：64x64分块 + MTLHeap KV Cache
        if strategy.backend == "metal":
            strategy.enable_tiling_64x64 = True
            strategy.enable_mtlheap_kv_cache = True
            strategy.memory_layouts["kv_cache"] = "mtlheap"
            strategy.memory_layouts["weights"] = "tiled_64x64"
        
        # INT4 量化
        strategy.quantization_mode = "int4" if strategy.enable_int4_quantization else "none"

        # 默认预取/调度
        strategy.schedules["prefetch"] = 2
        strategy.schedules["unroll"] = 2
        strategy.schedules["pipeline"] = 2

        # 检测到 Flash Attention
        if features.has_flash_attention:
            strategy.enable_op_fusion = True
            strategy.attention_config = {
                "flash_attention": True,
                "causal": True,
                "attn_impl": "cgc_flash",
            }
            strategy.op_hints.append(AgentOpHint.FLASH_ATTENTION)
            strategy.fusion_regions.append(["q_proj", "k_proj", "v_proj", "rope", "sdpa"])

        # 检测到 MoE
        if features.has_moe:
            strategy.moe_config = {
                "num_experts": 8,
                "top_k": 2,
                "routing_impl": "cgc_routing",
            }
            strategy.op_hints.append(AgentOpHint.MOE_ROUTING)
            strategy.fusion_regions.append(["moe_gate", "expert_forward", "moe_aggregate"])

        # 检测到 TP
        if features.has_tensor_parallel:
            strategy.tp_degree = 2
            strategy.op_hints.append(AgentOpHint.TENSOR_PARALLEL)

        # 检测到 VLM
        if features.has_vlm:
            strategy.vlm_config = {
                "vision_encoder": True,
                "cross_attention": True,
            }
            strategy.op_hints.append(AgentOpHint.VLM_CROSS_ATTENTION)

    def _apply_llama_cpp_reference(
        self,
        strategy: HarnessCompileStrategy,
        features: Any,
        space: Any,
    ):
        """参考 llama.cpp 的优化"""
        # 这里可以从 llama.cpp 学习最佳实践
        pass

    def _apply_vllm_reference(
        self,
        strategy: HarnessCompileStrategy,
        features: Any,
        space: Any,
    ):
        """参考 vLLM 的优化"""
        if "cuda" in strategy.backend:
            if features.has_moe:
                strategy.attention_config = strategy.attention_config or {}
                strategy.attention_config["paged_attention"] = True
        pass

    def _record_decision(
        self,
        strategy: HarnessCompileStrategy,
        features: Any,
        space: Any,
    ):
        """记录决策历史"""
        record = {
            "timestamp": None,
            "model_type": space.model_type,
            "device": self.device,
            "strategy": strategy.to_dict(),
            "features": {
                "has_attention": features.has_attention,
                "has_moe": features.has_moe,
                "has_vlm": features.has_vlm,
            },
        }
        self.compilation_history.append(record)

    def capture_graph(
        self,
        model: nn.Module,
        strategy: HarnessCompileStrategy,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[nn.Module, torch.fx.GraphModule]:
        """
        捕获完整计算图
        
        Args:
            model: PyTorch 模型
            strategy: 编译策略
            input_shape: 输入形状
            
        Returns:
            (compiled_model, graph_module)
        """
        graph_capture_config = strategy.graph_capture_config or {}
        capture_mode = graph_capture_config.get("capture_mode", "auto")
        
        if capture_mode == "auto":
            if strategy.megatrain_config and strategy.megatrain_config.get("enabled"):
                capture_mode = "megatrain"
            elif strategy.mlx_tune_config and strategy.mlx_tune_config.get("enabled"):
                capture_mode = "mlx_tune"
            else:
                capture_mode = "inference"
        
        logger.info(f"[HarnessAgent] Capturing graph in {capture_mode} mode...")
        
        if capture_mode == "megatrain":
            from .megatrain_graph_capture import MegatrainGraphCapture, MegatrainGraphCaptureConfig
            
            config = MegatrainGraphCaptureConfig(
                enable_full_graph=graph_capture_config.get("enable_full_graph", True),
                enable_cudagraphs=graph_capture_config.get("enable_cudagraphs", True),
                enable_dynamic_shapes=graph_capture_config.get("enable_dynamic_shapes", False),
                training_mode=strategy.megatrain_config.get("training_mode", "fsdp"),
                mixed_precision=strategy.megatrain_config.get("mixed_precision", "bf16"),
            )
            
            capturer = MegatrainGraphCapture(config)
            compiled_model, graph_module = capturer.capture(model, input_shape)
            
        elif capture_mode == "mlx_tune":
            from .mlx_tune_graph_capture import MLXTuneGraphCapture, MLXTuneGraphCaptureConfig
            
            config = MLXTuneGraphCaptureConfig(
                enable_full_graph=graph_capture_config.get("enable_full_graph", True),
                enable_metal_backend=self.device in ["metal", "mps"],
                enable_dynamic_shapes=graph_capture_config.get("enable_dynamic_shapes", False),
                lora_rank=strategy.mlx_tune_config.get("lora_rank", 8),
                lora_alpha=strategy.mlx_tune_config.get("lora_alpha", 16),
                lora_dropout=strategy.mlx_tune_config.get("lora_dropout", 0.05),
                enable_qlora=strategy.mlx_tune_config.get("enable_qlora", False),
                qlora_bits=strategy.mlx_tune_config.get("qlora_bits", 4),
            )
            
            capturer = MLXTuneGraphCapture(config)
            compiled_model, graph_module = capturer.capture(model, input_shape)
            
        else:
            from .graph_analyzer import GraphAnalyzer
            
            if input_shape is None:
                input_shape = (1, 2048, 4096)

            if self.device in ["metal", "mps"]:
                torch_device = torch.device("mps")
            elif self.device == "cuda":
                torch_device = torch.device("cuda")
            else:
                torch_device = torch.device("cpu")

            model = model.to(torch_device)
            dummy_input = torch.randn(*input_shape, device=torch_device)
            graph_features = GraphAnalyzer.analyze(model, dummy_input)

            compiled_model = torch.compile(
                model,
                mode="reduce-overhead",
                fullgraph=graph_capture_config.get("enable_full_graph", True),
            )
            
            with torch.no_grad():
                _ = compiled_model(dummy_input)
            
            graph_module = self._extract_graph_module(compiled_model)
        
        # 导出图（如果需要）
        if graph_capture_config.get("export_graph", False):
            export_path = graph_capture_config.get("export_path")
            if export_path:
                if capture_mode == "megatrain":
                    capturer.export_graph(export_path)
                elif capture_mode == "mlx_tune":
                    capturer.export_graph(export_path)
                else:
                    self._export_graph(graph_module, export_path)
        
        logger.info(f"[HarnessAgent] Graph capture complete!")
        
        return compiled_model, graph_module
    
    def _extract_graph_module(self, model: nn.Module) -> torch.fx.GraphModule:
        """提取图模块"""
        import torch.fx

        if isinstance(model, torch.fx.GraphModule):
            return model
        if hasattr(model, "_orig_mod"):
            try:
                return torch.fx.symbolic_trace(model._orig_mod)
            except Exception:
                return model._orig_mod
        if hasattr(model, "_forward_module"):
            try:
                return torch.fx.symbolic_trace(model._forward_module)
            except Exception:
                return model._forward_module
        try:
            return torch.fx.symbolic_trace(model)
        except Exception:
            return model
    
    def _export_graph(self, graph_module: torch.fx.GraphModule, path: str) -> None:
        """导出计算图"""
        import torch.fx
        
        code = torch.fx.graph.python_code(graph_module.graph)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        
        logger.info(f"[HarnessAgent] Graph exported to {path}")
