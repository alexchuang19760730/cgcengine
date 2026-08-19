# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
MLX-Tune 代码生成模块 - MLXTuneCGC

功能：
- 为 Apple Silicon Metal 后端生成优化代码
- 支持 LoRA/QLoRA 微调查找优化
- 支持统一内存优化
- 支持端云一体策略
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import logging

from .compile_strategy import CompileStrategy

logger = logging.getLogger(__name__)


@dataclass
class MLXTuneCGCConfig:
    """MLX-Tune CGC 配置"""
    
    # LoRA 配置
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # QLoRA 配置
    enable_qlora: bool = False
    qlora_bits: int = 4
    qlora_group_size: int = 64
    
    # 训练配置
    max_batch_size: int = 16
    max_seq_len: int = 2048
    hidden_dim: int = 4096
    num_heads: int = 32
    head_dim: int = 128
    
    # Metal 后端配置
    enable_metal_backend: bool = True
    use_unified_memory: bool = True
    use_graph_execution: bool = True
    
    # 优化配置
    use_flash_attention: bool = True
    use_fused_layernorm: bool = True
    use_fused_dropout: bool = True
    
    # 端云一体配置
    edge_cloud_mode: bool = True
    prefill_on_cloud: bool = True
    decode_on_edge: bool = True
    
    # 目标模块
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "enable_qlora": self.enable_qlora,
            "qlora_bits": self.qlora_bits,
            "qlora_group_size": self.qlora_group_size,
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "enable_metal_backend": self.enable_metal_backend,
            "use_unified_memory": self.use_unified_memory,
            "use_graph_execution": self.use_graph_execution,
            "use_flash_attention": self.use_flash_attention,
            "use_fused_layernorm": self.use_fused_layernorm,
            "use_fused_dropout": self.use_fused_dropout,
            "edge_cloud_mode": self.edge_cloud_mode,
            "prefill_on_cloud": self.prefill_on_cloud,
            "decode_on_edge": self.decode_on_edge,
            "target_modules": self.target_modules,
        }

@dataclass
class MetalKernelSpec:
    op_type: str
    entry: str
    source: str
    threadgroup_size: Tuple[int, int, int] = (32, 1, 1)


class MLXTuneCGC:
    """MLX-Tune 代码生成器"""
    
    def __init__(self, config: Optional[MLXTuneCGCConfig] = None):
        self.config = config or MLXTuneCGCConfig()
        self.compile_strategy = CompileStrategy(backend="mlx-tune")
    
    def generate_compile_strategy(self) -> CompileStrategy:
        """生成编译策略"""
        logger.info("[MLXTuneCGC] Generating compile strategy...")
        
        # 设置融合边界
        self.compile_strategy.fusion_boundary = [
            ["q_proj", "k_proj", "v_proj", "flash_attn", "o_proj"],
            ["gate_proj", "silu", "up_proj", "down_proj"],
            ["layer_norm", "add"],
            ["lora_A", "lora_B", "dropout"],
        ]
        
        # 设置 Tiling 配置 (Metal 优化)
        self.compile_strategy.tiling_config = {
            "Tile_M": 256,
            "Tile_N": 256,
            "Tile_K": 128,
            "block_size": 32,
            "thread_groups": 8,
            "wave_size": 32,
        }
        
        # 设置内存层级 (统一内存优化)
        self.compile_strategy.memory_hierarchy = {
            "q": "register",
            "k": "shared",
            "v": "shared",
            "attn_out": "register",
            "lora_A": "constant",
            "lora_B": "constant",
            "grad": "unified",
            "optimizer_state": "unified",
        }
        
        # 设置调度方案
        self.compile_strategy.scheduling_plan = {
            "use_metal_graph": self.config.use_graph_execution,
            "unified_memory_optimization": self.config.use_unified_memory,
            "lora_batch_scheduling": True,
            "enable_overlap": True,
            "prefetch_layers": 1,
            "edge_cloud_coordination": self.config.edge_cloud_mode,
        }
        
        # 设置 Attention 配置
        self.compile_strategy.attention_config = {
            "use_flash": self.config.use_flash_attention,
            "causal": True,
            "head_dim": self.config.head_dim,
            "num_heads": self.config.num_heads,
            "scale": True,
            "lora_enabled": True,
        }
        
        # 设置元数据
        self.compile_strategy.metadata = {
            "backend": "mlx-tune",
            "lora_enabled": True,
            "qlora_enabled": self.config.enable_qlora,
            "edge_cloud_mode": self.config.edge_cloud_mode,
            "unified_memory": self.config.use_unified_memory,
        }
        
        return self.compile_strategy
    
    def generate_metal_code(self, op_type: str) -> Optional[str]:
        """
        生成特定算子的 Metal Shader 代码
        
        Args:
            op_type: 算子类型 (attention, mlp, layernorm, lora, etc.)
        
        Returns:
            Metal Shader 代码字符串
        """
        logger.info(f"[MLXTuneCGC] Generating Metal code for {op_type}...")
        
        if op_type == "attention":
            return self._generate_flash_attention_metal()
        elif op_type == "mlp":
            return self._generate_fused_mlp_metal()
        elif op_type == "layernorm":
            return self._generate_fused_layernorm_metal()
        elif op_type == "lora":
            return self._generate_lora_metal()
        else:
            logger.warning(f"[MLXTuneCGC] Unknown op type: {op_type}")
            return None

    def generate_metal_spec(self, op_type: str) -> Optional[MetalKernelSpec]:
        source = self.generate_metal_code(op_type)
        if not source:
            return None
        entry_map = {
            "attention": "mlx_flash_attention",
            "mlp": "mlx_fused_mlp",
            "layernorm": "mlx_fused_layernorm_add",
            "lora": "mlx_lora_forward",
        }
        entry = entry_map.get(op_type, "")
        return MetalKernelSpec(op_type=op_type, entry=entry, source=source)

    def generate_metal_plan(self, op_types: List[str]) -> List[MetalKernelSpec]:
        specs: List[MetalKernelSpec] = []
        for op_type in op_types:
            spec = self.generate_metal_spec(op_type)
            if spec is not None:
                specs.append(spec)
        return specs
    
    def _generate_flash_attention_metal(self) -> str:
        """生成 Flash Attention Metal Shader"""
        tiling = self.compile_strategy.tiling_config
        return f"""
#include <metal_stdlib>
using namespace metal;

kernel void mlx_flash_attention(
    device half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    uint batch_size [[buffer(4)]],
    uint seq_len [[buffer(5)]],
    uint num_heads [[buffer(6)]],
    uint head_dim [[buffer(7)]],
    float scale [[buffer(8)]],
    uint3 gridDim [[threadgroup_grid_size]],
    uint3 blockDim [[threadgroup_size]]
) {{
    // Flash Attention for Apple Silicon
    // Tile: {tiling.get('Tile_M', 256)}x{tiling.get('Tile_N', 256)}x{tiling.get('Tile_K', 128)}
    // Head dim: {self.config.head_dim}
    // Unified Memory: {self.config.use_unified_memory}
    
    threadgroup half smem[2 * {tiling.get('Tile_M', 256)} * {tiling.get('Tile_N', 256)}];
    
    uint batch_idx = gridDim.x * blockDim.x + threadIdx.x;
    uint head_idx = gridDim.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    
    // ... Flash Attention implementation for Metal ...
}}

kernel void mlx_flash_attention_backward(
    device half* O [[buffer(0)]],
    device half* Q [[buffer(1)]],
    device half* K [[buffer(2)]],
    device half* V [[buffer(3)]],
    device half* dQ [[buffer(4)]],
    device half* dK [[buffer(5)]],
    device half* dV [[buffer(6)]],
    uint batch_size [[buffer(7)]],
    uint seq_len [[buffer(8)]],
    uint num_heads [[buffer(9)]],
    uint head_dim [[buffer(10)]],
    uint3 gridDim [[threadgroup_grid_size]],
    uint3 blockDim [[threadgroup_size]]
) {{
    // Flash Attention backward for Metal
    // ... Backward implementation ...
}}
"""
    
    def _generate_fused_mlp_metal(self) -> str:
        """生成融合 MLP Metal Shader"""
        return f"""
#include <metal_stdlib>
using namespace metal;

kernel void mlx_fused_mlp(
    device half* input [[buffer(0)]],
    device half* gate_weight [[buffer(1)]],
    device half* up_weight [[buffer(2)]],
    device half* down_weight [[buffer(3)]],
    device half* output [[buffer(4)]],
    uint batch_size [[buffer(5)]],
    uint seq_len [[buffer(6)]],
    uint hidden_dim [[buffer(7)]],
    uint intermediate_dim [[buffer(8)]],
    uint3 gridDim [[threadgroup_grid_size]],
    uint3 blockDim [[threadgroup_size]]
) {{
    // Fused MLP: gate_proj -> SiLU -> up_proj -> down_proj
    // Hidden dim: {self.config.hidden_dim}
    // ... Implementation ...
}}
"""
    
    def _generate_fused_layernorm_metal(self) -> str:
        """生成融合 LayerNorm Metal Shader"""
        return f"""
#include <metal_stdlib>
using namespace metal;

kernel void mlx_fused_layernorm_add(
    device half* input [[buffer(0)]],
    device half* residual [[buffer(1)]],
    device half* output [[buffer(2)]],
    device half* weight [[buffer(3)]],
    device half* bias [[buffer(4)]],
    uint batch_size [[buffer(5)]],
    uint seq_len [[buffer(6)]],
    uint hidden_dim [[buffer(7)]],
    float eps [[buffer(8)]],
    uint3 gridDim [[threadgroup_grid_size]],
    uint3 blockDim [[threadgroup_size]]
) {{
    // Fused LayerNorm + Add for Metal
    // ... Implementation ...
}}
"""
    
    def _generate_lora_metal(self) -> str:
        """生成 LoRA Metal Shader"""
        return f"""
#include <metal_stdlib>
using namespace metal;

kernel void mlx_lora_forward(
    device half* input [[buffer(0)]],
    constant half* lora_A [[buffer(1)]],
    constant half* lora_B [[buffer(2)]],
    device half* output [[buffer(3)]],
    uint batch_size [[buffer(4)]],
    uint seq_len [[buffer(5)]],
    uint in_features [[buffer(6)]],
    uint out_features [[buffer(7)]],
    uint rank [[buffer(8)]],
    float scaling [[buffer(9)]],
    uint3 gridDim [[threadgroup_grid_size]],
    uint3 blockDim [[threadgroup_size]]
) {{
    // LoRA forward pass
    // Rank: {self.config.lora_rank}, Alpha: {self.config.lora_alpha}
    // Scaling: {(self.config.lora_alpha / self.config.lora_rank)}
    
    uint idx = gridDim.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * seq_len * out_features) return;
    
    // ... LoRA implementation ...
}}

kernel void mlx_lora_backward(
    device half* output_grad [[buffer(0)]],
    device half* input [[buffer(1)]],
    constant half* lora_A [[buffer(2)]],
    constant half* lora_B [[buffer(3)]],
    device half* input_grad [[buffer(4)]],
    device half* lora_A_grad [[buffer(5)]],
    device half* lora_B_grad [[buffer(6)]],
    uint batch_size [[buffer(7)]],
    uint seq_len [[buffer(8)]],
    uint in_features [[buffer(9)]],
    uint out_features [[buffer(10)]],
    uint rank [[buffer(11)]],
    float scaling [[buffer(12)]],
    uint3 gridDim [[threadgroup_grid_size]],
    uint3 blockDim [[threadgroup_size]]
) {{
    // LoRA backward pass
    // ... Backward implementation ...
}}
"""
    
    def generate_tune_loop_code(self) -> str:
        """生成完整的 LoRA 微调循环代码"""
        return f"""
// MLX-Tune Training Loop - Generated by CGC
// Config: {self.config.to_dict()}

void mlx_tune_step(
    Tensor& input_ids,
    Tensor& labels,
    Model& model,
    Optimizer& optimizer,
    int step
) {{
    model.train();
    
    // Enable Metal Graph execution
    #if {1 if self.config.use_graph_execution else 0}
    static id<MTLComputePipelineState> pipeline = nil;
    static id<MTLCommandBuffer> commandBuffer = nil;
    
    if (!pipeline) {{
        // Create pipeline once
    }}
    #endif
    
    // Forward pass with LoRA
    Tensor loss = model(input_ids, labels);
    
    // Backward pass
    loss.backward();
    
    // Step optimizer (only update LoRA weights)
    optimizer.step();
    optimizer.zero_grad();
    
    // Unified memory optimization hint
    #if {1 if self.config.use_unified_memory else 0}
    [commandBuffer addMemoryBarrier];
    #endif
}}
"""
    
    def compare_with_native(self, model: nn.Module, dummy_inputs: Any) -> Dict[str, Any]:
        """
        性能对比：MagiCompiler 优化 vs 原生 MLX-Tune
        
        Args:
            model: PyTorch 模型 (带有 LoRA 层)
            dummy_inputs: 输入张量
            
        Returns:
            性能对比结果
        """
        logger.info("[MLXTuneCGC] Running performance comparison...")

        import time

        model.train()
        device_type = self._infer_device_type(dummy_inputs, model)

        def sync():
            self._sync_device(device_type)

        warmup_steps = 3
        measure_steps = 10

        native_times = []
        for i in range(warmup_steps + measure_steps):
            start = time.perf_counter()
            loss = self._run_train_step(model, dummy_inputs)
            if isinstance(loss, torch.Tensor):
                loss.backward()
            sync()
            if i >= warmup_steps:
                native_times.append(time.perf_counter() - start)
            self._zero_grads(model)

        native_avg_time = sum(native_times) / len(native_times)
        native_throughput = self.config.max_batch_size / native_avg_time if native_avg_time > 0 else 0.0

        compiled_model = None
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
            loss = self._run_train_step(compiled_model, dummy_inputs)
            if isinstance(loss, torch.Tensor):
                loss.backward()
            sync()
            self._zero_grads(model)
        except Exception as e:
            logger.warning(f"[MLXTuneCGC] Failed to build compiled model: {e}")
            compiled_model = None

        optimized_times = []
        opt_model = compiled_model if compiled_model is not None else model
        for i in range(warmup_steps + measure_steps):
            start = time.perf_counter()
            loss = self._run_train_step(opt_model, dummy_inputs)
            if isinstance(loss, torch.Tensor):
                loss.backward()
            sync()
            if i >= warmup_steps:
                optimized_times.append(time.perf_counter() - start)
            self._zero_grads(model)

        optimized_avg_time = sum(optimized_times) / len(optimized_times)
        optimized_throughput = self.config.max_batch_size / optimized_avg_time if optimized_avg_time > 0 else 0.0

        speedup = native_avg_time / optimized_avg_time if optimized_avg_time > 0 else 0.0

        return {
            "native": {
                "avg_time_ms": native_avg_time * 1000,
                "throughput_samples_per_sec": native_throughput,
                "flops": self._estimate_flops(),
            },
            "optimized": {
                "avg_time_ms": optimized_avg_time * 1000,
                "throughput_samples_per_sec": optimized_throughput,
                "flops": self._estimate_flops(),
            },
            "speedup": speedup,
            "config": self.config.to_dict(),
            "device_type": device_type,
            "lora_config": {
                "rank": self.config.lora_rank,
                "alpha": self.config.lora_alpha,
                "qlora_enabled": self.config.enable_qlora,
            },
        }

    def _infer_device_type(self, dummy_inputs: Any, model: nn.Module) -> str:
        if isinstance(dummy_inputs, dict):
            for v in dummy_inputs.values():
                if isinstance(v, torch.Tensor):
                    return v.device.type
        if isinstance(dummy_inputs, (tuple, list)):
            for v in dummy_inputs:
                if isinstance(v, torch.Tensor):
                    return v.device.type
        try:
            return next(model.parameters()).device.type
        except StopIteration:
            return "cpu"

    def _sync_device(self, device_type: str) -> None:
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            return
        if device_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    def _zero_grads(self, model: nn.Module) -> None:
        try:
            model.zero_grad(set_to_none=True)
        except TypeError:
            model.zero_grad()

    def _run_train_step(self, model: nn.Module, dummy_inputs: Any) -> torch.Tensor:
        outputs = self._call_model(model, dummy_inputs)
        return self._extract_loss(outputs)

    def _call_model(self, model: nn.Module, dummy_inputs: Any) -> Any:
        if isinstance(dummy_inputs, dict):
            try:
                return model(**dummy_inputs)
            except TypeError:
                return model(*tuple(dummy_inputs.values()))
        if isinstance(dummy_inputs, (tuple, list)):
            return model(*dummy_inputs)
        return model(dummy_inputs)

    def _extract_loss(self, outputs: Any) -> torch.Tensor:
        if isinstance(outputs, dict) and "loss" in outputs and isinstance(outputs["loss"], torch.Tensor):
            return outputs["loss"]
        if isinstance(outputs, (tuple, list)) and outputs and isinstance(outputs[0], torch.Tensor):
            return outputs[0]
        if isinstance(outputs, torch.Tensor):
            return outputs.sum()
        return torch.tensor(0.0)
    
    def _estimate_flops(self) -> float:
        """估算 LoRA 微调 FLOPs"""
        batch_size = self.config.max_batch_size
        seq_len = self.config.max_seq_len
        hidden_dim = self.config.hidden_dim
        rank = self.config.lora_rank
        
        # LoRA FLOPs (主要计算)
        lora_flops = 2 * batch_size * seq_len * hidden_dim * rank
        
        # 原有 Attention FLOPs (冻结权重，不计算梯度)
        attn_flops = 2 * batch_size * seq_len * hidden_dim * seq_len
        
        # MLP FLOPs (冻结权重)
        mlp_flops = 2 * batch_size * seq_len * hidden_dim * hidden_dim
        
        return lora_flops + attn_flops + mlp_flops
