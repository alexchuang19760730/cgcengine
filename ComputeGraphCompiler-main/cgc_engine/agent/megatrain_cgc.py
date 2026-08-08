# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
Megatrain 代码生成模块 - MegaTrainCGC

功能：
- 为 NVIDIA CUDA 训练后端生成优化代码
- 支持 FSDP 分布式训练优化
- 支持混合精度训练优化
- 支持梯度累积优化
"""

import json
import os
import urllib.request

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import logging

from .compile_strategy import CompileStrategy

logger = logging.getLogger(__name__)


def _debug_report(hypothesis_id: str, location: str, msg: str, data: Optional[Dict[str, Any]] = None) -> None:
    # #region debug-point A:report
    env_path = os.path.join(os.getcwd(), ".dbg", "qwen3vl-compile-benchmark.env")
    url = "http://127.0.0.1:7777/event"
    session_id = "qwen3vl-compile-benchmark"
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read()
        for line in content.splitlines():
            if line.startswith("DEBUG_SERVER_URL="):
                url = line.split("=", 1)[1].strip() or url
            elif line.startswith("DEBUG_SESSION_ID="):
                session_id = line.split("=", 1)[1].strip() or session_id
    except Exception:
        pass
    payload = {
        "sessionId": session_id,
        "runId": "pre",
        "hypothesisId": hypothesis_id,
        "location": location,
        "msg": msg,
        "data": data or {},
    }
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=1.0,
        ).read()
    except Exception:
        pass
    # #endregion


def _debug_describe_inputs(dummy_inputs: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"type": type(dummy_inputs).__name__}
    if isinstance(dummy_inputs, dict):
        fields: Dict[str, Any] = {}
        for key, value in dummy_inputs.items():
            if isinstance(value, torch.Tensor):
                fields[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
            else:
                fields[key] = {"type": type(value).__name__}
        summary["fields"] = fields
    return summary


@dataclass
class MegatrainCGCConfig:
    """Megatrain CGC 配置"""
    
    # 训练模式
    training_mode: str = "fsdp"  # fsdp, ddp, standalone
    mixed_precision: str = "bf16"  # bf16, fp16, fp32
    
    # FSDP 配置
    fsdp_sharding_strategy: str = "full_shard"
    fsdp_activation_checkpointing: bool = True
    fsdp_cpu_offload: bool = False
    
    # 梯度累积
    gradient_accumulation_steps: int = 4
    use_gradient_checkpointing: bool = True
    
    # CUDA 优化
    use_cudagraphs: bool = True
    use_flash_attention: bool = True
    use_fused_layernorm: bool = True
    use_fused_dropout: bool = True
    
    # 性能配置
    max_batch_size: int = 32
    max_seq_len: int = 4096
    hidden_dim: int = 4096
    num_heads: int = 32
    head_dim: int = 128
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "training_mode": self.training_mode,
            "mixed_precision": self.mixed_precision,
            "fsdp_sharding_strategy": self.fsdp_sharding_strategy,
            "fsdp_activation_checkpointing": self.fsdp_activation_checkpointing,
            "fsdp_cpu_offload": self.fsdp_cpu_offload,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_cudagraphs": self.use_cudagraphs,
            "use_flash_attention": self.use_flash_attention,
            "use_fused_layernorm": self.use_fused_layernorm,
            "use_fused_dropout": self.use_fused_dropout,
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
        }


class MegatrainCGC:
    """Megatrain 代码生成器"""
    
    def __init__(self, config: Optional[MegatrainCGCConfig] = None):
        self.config = config or MegatrainCGCConfig()
        self.compile_strategy = CompileStrategy(backend="megatrain")
    
    def generate_compile_strategy(self) -> CompileStrategy:
        """生成编译策略"""
        logger.info("[MegatrainCGC] Generating compile strategy...")
        
        # 设置融合边界
        self.compile_strategy.fusion_boundary = [
            ["q_proj", "k_proj", "v_proj", "flash_attn", "o_proj"],
            ["gate_proj", "silu", "up_proj", "down_proj"],
            ["layer_norm", "add"],
        ]
        
        # 设置 Tiling 配置
        self.compile_strategy.tiling_config = {
            "Tile_M": 128,
            "Tile_N": 128,
            "Tile_K": 64,
            "block_size": 32,
            "num_warps": 4,
            "num_stages": 2,
        }
        
        # 设置内存层级
        self.compile_strategy.memory_hierarchy = {
            "q": "register",
            "k": "shared",
            "v": "shared",
            "attn_out": "register",
            "mlp_out": "register",
            "grad": "global",
            "optimizer_state": "global",
        }
        
        # 设置调度方案
        self.compile_strategy.scheduling_plan = {
            "use_cuda_graph": self.config.use_cudagraphs,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "pipeline_parallelism": False,
            "sequence_parallelism": True,
            "enable_overlap": True,
            "prefetch_layers": 2,
        }
        
        # 设置 Attention 配置
        self.compile_strategy.attention_config = {
            "use_flash": self.config.use_flash_attention,
            "causal": True,
            "head_dim": self.config.head_dim,
            "num_heads": self.config.num_heads,
            "scale": True,
        }
        
        # 设置元数据
        self.compile_strategy.metadata = {
            "backend": "megatrain",
            "training_mode": self.config.training_mode,
            "mixed_precision": self.config.mixed_precision,
            "fsdp_enabled": self.config.training_mode == "fsdp",
        }
        
        return self.compile_strategy
    
    def generate_kernel_code(self, op_type: str) -> Optional[str]:
        """
        生成特定算子的 CUDA kernel 代码
        
        Args:
            op_type: 算子类型 (attention, mlp, layernorm, etc.)
        
        Returns:
            CUDA kernel 代码字符串
        """
        logger.info(f"[MegatrainCGC] Generating kernel code for {op_type}...")
        
        if op_type == "attention":
            return self._generate_flash_attention_kernel()
        elif op_type == "mlp":
            return self._generate_fused_mlp_kernel()
        elif op_type == "layernorm":
            return self._generate_fused_layernorm_kernel()
        elif op_type == "fsdp_allreduce":
            return self._generate_fsdp_allreduce_kernel()
        else:
            logger.warning(f"[MegatrainCGC] Unknown op type: {op_type}")
            return None
    
    def _generate_flash_attention_kernel(self) -> str:
        """生成 Flash Attention CUDA Kernel"""
        tiling = self.compile_strategy.tiling_config
        return f"""
__global__ void megatrain_flash_attention_bf16(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    float scale
) {{
    // Flash Attention v4 kernel for training
    // Tile: {tiling.get('Tile_M', 128)}x{tiling.get('Tile_N', 128)}x{tiling.get('Tile_K', 64)}
    // Head dim: {self.config.head_dim}
    // Mixed precision: {self.config.mixed_precision}
    
    // Shared memory allocation
    __shared__ __half smem[2 * {tiling.get('Tile_M', 128)} * {tiling.get('Tile_N', 128)}];
    
    // Load Q, K, V tiles
    // ... Flash Attention implementation ...
}}

__global__ void megatrain_flash_attention_backward_bf16(
    const __half* __restrict__ O,
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ dQ,
    __half* __restrict__ dK,
    __half* __restrict__ dV,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {{
    // Flash Attention backward kernel
    // ... Backward implementation ...
}}
"""
    
    def _generate_fused_mlp_kernel(self) -> str:
        """生成融合 MLP CUDA Kernel"""
        return f"""
__global__ void megatrain_fused_mlp_bf16(
    const __half* __restrict__ input,
    const __half* __restrict__ gate_weight,
    const __half* __restrict__ up_weight,
    const __half* __restrict__ down_weight,
    __half* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int intermediate_dim
) {{
    // Fused MLP: gate_proj -> SiLU -> up_proj -> down_proj
    // Hidden dim: {self.config.hidden_dim}
    // Mixed precision: {self.config.mixed_precision}
    
    // ... Fused MLP implementation ...
}}

__global__ void megatrain_fused_mlp_backward_bf16(
    const __half* __restrict__ output_grad,
    const __half* __restrict__ input,
    const __half* __restrict__ gate_weight,
    const __half* __restrict__ up_weight,
    const __half* __restrict__ down_weight,
    __half* __restrict__ input_grad,
    __half* __restrict__ gate_weight_grad,
    __half* __restrict__ up_weight_grad,
    __half* __restrict__ down_weight_grad,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int intermediate_dim
) {{
    // Fused MLP backward
    // ... Backward implementation ...
}}
"""
    
    def _generate_fused_layernorm_kernel(self) -> str:
        """生成融合 LayerNorm CUDA Kernel"""
        return f"""
__global__ void megatrain_fused_layernorm_add_bf16(
    const __half* __restrict__ input,
    const __half* __restrict__ residual,
    __half* __restrict__ output,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    int batch_size,
    int seq_len,
    int hidden_dim,
    float eps
) {{
    // Fused LayerNorm + Add
    // Hidden dim: {self.config.hidden_dim}
    // ... Implementation ...
}}
"""
    
    def _generate_fsdp_allreduce_kernel(self) -> str:
        """生成 FSDP AllReduce 优化 Kernel"""
        return f"""
__global__ void megatrain_fsdp_allreduce_bf16(
    __half* __restrict__ tensor,
    int size,
    int rank,
    int world_size
) {{
    // FSDP AllReduce with overlap
    // Sharding strategy: {self.config.fsdp_sharding_strategy}
    // ... Implementation ...
}}

__global__ void megatrain_fsdp_grad_scatter_bf16(
    const __half* __restrict__ full_grad,
    __half* __restrict__ sharded_grad,
    int size,
    int rank,
    int world_size
) {{
    // FSDP gradient scatter
    // ... Implementation ...
}}
"""
    
    def generate_training_loop_code(self) -> str:
        """生成完整的训练循环代码"""
        return f"""
// Megatrain Training Loop - Generated by CGC
// Config: {self.config.to_dict()}

void megatrain_train_step(
    Tensor& input_ids,
    Tensor& labels,
    Model& model,
    Optimizer& optimizer,
    int step,
    int gradient_accumulation_steps={self.config.gradient_accumulation_steps}
) {{
    // Forward pass with gradient checkpointing
    model.train();
    
    // Enable CUDA Graph capture
    #if {1 if self.config.use_cudagraphs else 0}
    cudaStream_t stream = cudaStreamDefault;
    static cudaGraph_t graph = nullptr;
    static cudaGraphExec_t graph_exec = nullptr;
    
    if (step == 0) {{
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    }}
    #endif
    
    // Forward pass
    Tensor loss = model(input_ids, labels);
    
    // Backward pass with gradient accumulation
    loss = loss / gradient_accumulation_steps;
    loss.backward();
    
    // Step optimizer
    if ((step + 1) % gradient_accumulation_steps == 0) {{
        optimizer.step();
        optimizer.zero_grad();
    }}
    
    #if {1 if self.config.use_cudagraphs else 0}
    if (step == 0) {{
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
    }} else {{
        cudaGraphLaunch(graph_exec, stream);
    }}
    cudaStreamSynchronize(stream);
    #endif
}}
"""
    
    def compare_with_native(
        self,
        model: nn.Module,
        dummy_inputs: Any,
        *,
        optimized_model: Optional[nn.Module] = None,
        prepared_inputs_fn: Optional[Callable[[Any], Any]] = None,
        optimized_inputs: Optional[Any] = None,
        optimizer_factory: Optional[Callable[[nn.Module], Any]] = None,
        scheduler_factory: Optional[Callable[[Any], Any]] = None,
        max_grad_norm: Optional[float] = None,
        post_step_fn: Optional[Callable[[nn.Module, Any, Any, Any], None]] = None,
    ) -> Dict[str, Any]:
        """
        性能对比：MagiCompiler 优化 vs 原生 Megatrain
        
        Args:
            model: PyTorch 模型
            dummy_inputs: 输入张量
            
        Returns:
            性能对比结果
        """
        logger.info("[MegatrainCGC] Running performance comparison...")
        # #region debug-point B:compare-entry
        _debug_report(
            "B",
            "megatrain_cgc.py:compare_with_native",
            "[DEBUG] compare_with_native entry",
            {
                "optimized_model_provided": bool(optimized_model is not None),
                "input_summary": _debug_describe_inputs(dummy_inputs),
                "config": self.config.to_dict(),
            },
        )
        # #endregion
        
        import time

        model.train()
        device_type = self._infer_device_type(dummy_inputs, model)

        def sync() -> None:
            self._sync_device(device_type)

        def make_optimizer(target_model: nn.Module) -> Any:
            if callable(optimizer_factory):
                return optimizer_factory(target_model)
            return None

        def make_scheduler(optimizer: Any) -> Any:
            if optimizer is None:
                return None
            if callable(scheduler_factory):
                return scheduler_factory(optimizer)
            return None

        native_optimizer = make_optimizer(model)
        native_scheduler = make_scheduler(native_optimizer)

        warmup_steps = 3
        measure_steps = 10

        native_times: list[float] = []
        for i in range(warmup_steps + measure_steps):
            start = time.perf_counter()
            current_inputs = prepared_inputs_fn(dummy_inputs) if callable(prepared_inputs_fn) else dummy_inputs
            self._run_benchmark_step(
                model,
                current_inputs,
                optimizer=native_optimizer,
                scheduler=native_scheduler,
                max_grad_norm=max_grad_norm,
                post_step_fn=post_step_fn,
                sync_fn=sync,
            )
            if i >= warmup_steps:
                native_times.append(time.perf_counter() - start)

        native_avg_time = sum(native_times) / len(native_times) if native_times else 0.0
        native_throughput = self.config.max_batch_size / native_avg_time if native_avg_time > 0 else 0.0

        compiled_model = optimized_model
        compile_source = "provided" if optimized_model is not None else "fresh"
        compile_error: str = ""
        if compiled_model is None:
            try:
                compile_inputs = optimized_inputs
                if compile_inputs is None:
                    compile_inputs = prepared_inputs_fn(dummy_inputs) if callable(prepared_inputs_fn) else dummy_inputs
                compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                compile_optimizer = make_optimizer(compiled_model)
                compile_scheduler = make_scheduler(compile_optimizer)
                self._run_benchmark_step(
                    compiled_model,
                    compile_inputs,
                    optimizer=compile_optimizer,
                    scheduler=compile_scheduler,
                    max_grad_norm=max_grad_norm,
                    post_step_fn=post_step_fn,
                    sync_fn=sync,
                )
            except Exception as e:
                logger.warning(f"[MegatrainCGC] Failed to build compiled model: {e}")
                compile_error = repr(e)
                compiled_model = None
                compile_source = "fallback_native"
                # #region debug-point A:compile-fallback
                _debug_report(
                    "A",
                    "megatrain_cgc.py:compare_with_native",
                    "[DEBUG] compare compile fallback",
                    {"compile_error": compile_error, "compile_source": compile_source},
                )
                # #endregion

        optimized_times: list[float] = []
        opt_model = compiled_model if compiled_model is not None else model
        optimized_optimizer = make_optimizer(opt_model)
        optimized_scheduler = make_scheduler(optimized_optimizer)
        for i in range(warmup_steps + measure_steps):
            start = time.perf_counter()
            current_inputs = prepared_inputs_fn(dummy_inputs) if callable(prepared_inputs_fn) else (optimized_inputs if optimized_inputs is not None else dummy_inputs)
            self._run_benchmark_step(
                opt_model,
                current_inputs,
                optimizer=optimized_optimizer,
                scheduler=optimized_scheduler,
                max_grad_norm=max_grad_norm,
                post_step_fn=post_step_fn,
                sync_fn=sync,
            )
            if i >= warmup_steps:
                optimized_times.append(time.perf_counter() - start)

        optimized_avg_time = sum(optimized_times) / len(optimized_times) if optimized_times else 0.0
        optimized_throughput = self.config.max_batch_size / optimized_avg_time if optimized_avg_time > 0 else 0.0

        speedup = native_avg_time / optimized_avg_time if optimized_avg_time > 0 else 0.0
        raw_speedup_min = (
            os.environ.get("CGC_MEGATRAIN_SPEEDUP_MIN")
            or os.environ.get("CGC_M4_SPEEDUP_MIN")
            or "1.5"
        )
        try:
            speedup_min = float(raw_speedup_min)
        except Exception:
            speedup_min = 1.5
        meets_speedup_gate = bool(speedup >= speedup_min)
        # #region debug-point C:compare-exit
        _debug_report(
            "C",
            "megatrain_cgc.py:compare_with_native",
            "[DEBUG] compare_with_native exit",
            {
                "compile_source": compile_source,
                "compile_error": compile_error,
                "native_avg_time_ms": native_avg_time * 1000,
                "optimized_avg_time_ms": optimized_avg_time * 1000,
                "speedup": speedup,
                "device_type": device_type,
                "prepared_inputs_in_timing": bool(callable(prepared_inputs_fn)),
                "optimizer_in_timing": bool(native_optimizer is not None),
                "scheduler_in_timing": bool(native_scheduler is not None),
                "grad_clip_in_timing": max_grad_norm is not None,
            },
        )
        # #endregion
        
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
            "compile_source": compile_source,
            "compile_error": compile_error,
            "speedup": speedup,
            "speedup_min": speedup_min,
            "meets_speedup_gate": meets_speedup_gate,
            "performance_gate_status": "PASS" if meets_speedup_gate else "FAIL",
            "config": self.config.to_dict(),
            "device_type": device_type,
            "prepared_inputs_in_timing": bool(callable(prepared_inputs_fn)),
            "optimizer_in_timing": bool(native_optimizer is not None),
            "scheduler_in_timing": bool(native_scheduler is not None),
            "grad_clip_in_timing": max_grad_norm is not None,
            "post_step_in_timing": bool(callable(post_step_fn)),
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

    def _zero_optimizer(self, optimizer: Any) -> None:
        try:
            optimizer.zero_grad(set_to_none=True)
        except TypeError:
            optimizer.zero_grad()

    def _run_benchmark_step(
        self,
        model: nn.Module,
        dummy_inputs: Any,
        *,
        optimizer: Optional[Any],
        scheduler: Optional[Any],
        max_grad_norm: Optional[float],
        post_step_fn: Optional[Callable[[nn.Module, Any, Any, Any], None]],
        sync_fn: Callable[[], None],
    ) -> torch.Tensor:
        if optimizer is not None:
            self._zero_optimizer(optimizer)
        else:
            self._zero_grads(model)
        outputs = self._call_model(model, dummy_inputs)
        loss = self._extract_loss(outputs)
        if isinstance(loss, torch.Tensor):
            loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if optimizer is not None:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if callable(post_step_fn):
            post_step_fn(model, optimizer, outputs, dummy_inputs)
        sync_fn()
        return loss

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
        """估算训练 FLOPs"""
        batch_size = self.config.max_batch_size
        seq_len = self.config.max_seq_len
        hidden_dim = self.config.hidden_dim
        
        # Attention FLOPs
        attn_flops = 8 * batch_size * seq_len * hidden_dim * seq_len
        
        # MLP FLOPs
        mlp_flops = 8 * batch_size * seq_len * hidden_dim * hidden_dim
        
        # LayerNorm FLOPs
        ln_flops = 6 * batch_size * seq_len * hidden_dim
        
        return attn_flops + mlp_flops + ln_flops
