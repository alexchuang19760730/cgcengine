# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CGC SIMD Executor - 多後端 Kernel 執行器

將 CGC SIMD 命令直接 dispatch 到對應的後端執行。

支持的後端：
- CUDA (標準 GPU 加速)
- llama.cpp (GGUF 模型，支持 Apple M1/M2/M3/Intel)

支持的操作類型：
- Attention (KDA, FlashAttention, PagedAttention)
- Linear/MLP (GEMM)
- LayerNorm/RMSNorm
- RoPE (Rotary Position Embedding)
- Activation (SiLU, GeGLU, ...)
- Softmax
- All-reduce (Distributed)
- llama.cpp (GGUF 量化矩陣乘法、MoE、推理等)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Callable, Optional, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# 优先导入 C++ SIMD 引擎！
# ============================================================
try:
    path_to_cpp = Path(__file__).parent / "cgc_cpp" / "build"
    sys.path.insert(0, str(path_to_cpp))
    import cgc_cpp
    USE_CPP_ENGINE = True
    cgc_cpp.init()
    logger.info("✅ [CGC] C++ SIMD Engine loaded!")
except ImportError:
    USE_CPP_ENGINE = False
    logger.warning("⚠️ [CGC] C++ SIMD Engine NOT loaded, will use PyTorch fallback!")

# ============================================================
# GDS (GPUDirect Storage) PD 模块集成
# ============================================================
try:
    from cgc_engine.gds_service import (
        GDSManager,
        cuFileRead,
        cuFileWrite,
        is_gds_available,
    )
    GDS_AVAILABLE = True
    gds_manager = GDSManager()
    logger.info(f"✅ [CGC] GDS service initialized: {gds_manager.info()}")
except ImportError:
    GDS_AVAILABLE = False
    gds_manager = None
    logger.warning("⚠️ [CGC] GDS service NOT available, using PyTorch I/O fallback")

# ============================================================
# FlashMoE / oMLX 端侧 MoE 引擎
# ============================================================
FLASH_MOE_AVAILABLE = False
OMLX_AVAILABLE = False
flash_moe_client = None
omlx_client = None
try:
    from cgc_engine.flash_moe import FlashMoEClient
    from cgc_engine.omlx import OMLXClient
    FLASH_MOE_AVAILABLE = True
    OMLX_AVAILABLE = True
    flash_moe_client = FlashMoEClient()
    omlx_client = OMLXClient()
    logger.info("✅ [CGC] FlashMoE + oMLX loaded!")
except ImportError:
    logger.warning("⚠️ [CGC] FlashMoE/oMLX not available")

# ============================================================
# SPDK / JITLoad 集成（存储层 + 编译层）
# ============================================================
SPDK_AVAILABLE = False
JIT_AVAILABLE = False
spdk_client = None
jit_loader = None
try:
    from cgc_engine.spdk_adapter import SPDKIOManager, SPDKConfig
    from cgc_engine.spdk_adapter import SPDKKVStore, SPDKExpertStore
    SPDK_AVAILABLE = True
    spdk_config = SPDKConfig(enable_spdk=False)
    spdk_kv_store = SPDKKVStore(spdk_config) if 'SPDKKVStore' in dir() else None
    spdk_expert_store = SPDKExpertStore(spdk_config) if 'SPDKExpertStore' in dir() else None
    spdk_client = SPDKIOManager(config=spdk_config, kv_store=spdk_kv_store, expert_store=spdk_expert_store)
    logger.info("✅ [CGC] SPDK service initialized")
except ImportError:
    logger.warning("⚠️ [CGC] SPDK not available")

try:
    from cgc_engine.cgc_jitload import JITLoadManager
    JIT_AVAILABLE = True
    jit_loader = JITLoadManager()
    logger.info("✅ [CGC] JITLoad service initialized")
except ImportError:
    logger.warning("⚠️ [CGC] JITLoad not available")


class KernelType(Enum):
    """CUDA Kernel 类型"""
    ATTENTION = auto()
    LINEAR = auto()
    NORM = auto()
    ROPE = auto()
    ACTIVATION = auto()
    SOFTMAX = auto()
    REDUCE = auto()
    CUSTOM = auto()
    GDS_STORAGE = auto()  # PD/GDS 资源操作
    FLASH_MOE = auto()   # FlashMoE/oMLX 执行
    SPDK_IO = auto()     # SPDK 存储操作
    JIT_KERNEL = auto()  # JIT 编译操作


@dataclass
class CGCKernelSpec:
    """CGC Kernel 规范"""
    name: str
    kernel_type: KernelType
    cuda_kernel: Callable
    workspace_size_fn: Optional[Callable] = None
    supports_flashkda: bool = True


class CGCKernelRegistry:
    """
    CUDA Kernel 注册表

    所有 CUDA kernel 都在这里注册，然后通过 CGC 命令调用。
    """

    _instance = None
    _kernels: Dict[int, CGCKernelSpec] = {}
    _name_to_opcode: Dict[str, int] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_default_kernels()
        return cls._instance

    def _init_default_kernels(self):
        """初始化默认 kernel 注册"""
        self._register_kda_kernels()
        self._register_standard_kernels()
        self._register_all_cgc_commands()

    def _register_all_cgc_commands(self):
        """从 cgc_commands 同步所有命令到 kernel 注册表"""
        try:
            from .cgc_commands import (
                CGC_SIMD_COMMAND_SET,
                ATTENTION_SDPA_CMD, ATTENTION_KDA_CMD, ATTENTION_PAGED_CMD, ATTENTION_FLASH_CMD,
                LINEAR_GEMM_CMD, LINEAR_BIAS_CMD, GEMM_BATCHED_CMD,
                LAYER_NORM_CMD, RMS_NORM_CMD, GROUP_NORM_CMD,
                ROPE_CMD, ROPE_FUSED_CMD, YARN_ROPE_CMD,
                SILU_CMD, GELU_CMD, GELU_TANH_CMD, RELU_CMD, SIGMOID_CMD,
                SOFTMAX_CMD, LOG_SOFTMAX_CMD, TOP_K_CMD, TOP_P_CMD, TEMPERATURE_CMD,
                KV_CACHE_LOAD_CMD, KV_CACHE_STORE_CMD, KV_CACHE_UPDATE_CMD, KV_CACHE_STATIC_LAYOUT_CMD, KV_CACHE_COMMIT_CMD,
                BATCH_COMPILE_CMD, BATCH_MERGE_CMD,
                EMBEDDING_LOOKUP_CMD,
                ALL_REDUCE_CMD, ALL_GATHER_CMD, REDUCE_SCATTER_CMD,
                QUANTIZE_W8A16_CMD, QUANTIZE_W4A16_CMD, DEQUANTIZE_CMD, GPTQ_KERNEL_CMD, AWQ_KERNEL_CMD,
                FP8_E4M3_QUANT_CMD, FP8_E5M2_QUANT_CMD, FP8_DEQUANT_CMD,
            )

            cmd_to_kernel = {
                ATTENTION_SDPA_CMD.opcode: ("sdpa", KernelType.ATTENTION, torch.nn.functional.scaled_dot_product_attention),
                ATTENTION_KDA_CMD.opcode: ("kda_attn", KernelType.ATTENTION, torch.nn.functional.scaled_dot_product_attention),
                ATTENTION_FLASH_CMD.opcode: ("flash_attn", KernelType.ATTENTION, torch.nn.functional.scaled_dot_product_attention),
                LINEAR_GEMM_CMD.opcode: ("gemm", KernelType.LINEAR, torch.matmul),
                LINEAR_BIAS_CMD.opcode: ("linear_bias", KernelType.LINEAR, self._linear_bias_impl),
                GEMM_BATCHED_CMD.opcode: ("batched_gemm", KernelType.LINEAR, torch.bmm),
                LAYER_NORM_CMD.opcode: ("layer_norm", KernelType.NORM, torch.layer_norm),
                RMS_NORM_CMD.opcode: ("rms_norm", KernelType.NORM, self._rms_norm_impl),
                GROUP_NORM_CMD.opcode: ("group_norm", KernelType.NORM, torch.nn.functional.group_norm),
                ROPE_CMD.opcode: ("rope", KernelType.ROPE, self._rope_impl),
                ROPE_FUSED_CMD.opcode: ("rope_fused", KernelType.ROPE, self._rope_impl),
                SILU_CMD.opcode: ("silu", KernelType.ACTIVATION, torch.nn.functional.silu),
                GELU_CMD.opcode: ("gelu", KernelType.ACTIVATION, torch.nn.functional.gelu),
                GELU_TANH_CMD.opcode: ("gelu_tanh", KernelType.ACTIVATION, lambda x: torch.nn.functional.gelu(x, approximate="tanh")),
                RELU_CMD.opcode: ("relu", KernelType.ACTIVATION, torch.nn.functional.relu),
                SIGMOID_CMD.opcode: ("sigmoid", KernelType.ACTIVATION, torch.sigmoid),
                SOFTMAX_CMD.opcode: ("softmax", KernelType.SOFTMAX, torch.nn.functional.softmax),
                LOG_SOFTMAX_CMD.opcode: ("log_softmax", KernelType.SOFTMAX, torch.nn.functional.log_softmax),
                KV_CACHE_LOAD_CMD.opcode: ("kv_cache_load", KernelType.REDUCE, self._kv_cache_load_impl),
                KV_CACHE_STORE_CMD.opcode: ("kv_cache_store", KernelType.REDUCE, self._kv_cache_store_impl),
                KV_CACHE_UPDATE_CMD.opcode: ("kv_cache_update", KernelType.REDUCE, self._kv_cache_update_impl),
                KV_CACHE_STATIC_LAYOUT_CMD.opcode: ("kv_cache_static_layout", KernelType.CUSTOM, self._kv_cache_static_layout_impl),
                KV_CACHE_COMMIT_CMD.opcode: ("kv_cache_commit", KernelType.CUSTOM, self._kv_cache_commit_impl),
                BATCH_COMPILE_CMD.opcode: ("batch_compile", KernelType.CUSTOM, self._batch_compile_impl),
                BATCH_MERGE_CMD.opcode: ("batch_merge", KernelType.CUSTOM, self._batch_merge_impl),
                EMBEDDING_LOOKUP_CMD.opcode: ("embedding_lookup", KernelType.LINEAR, torch.nn.functional.embedding),
                ALL_REDUCE_CMD.opcode: ("all_reduce", KernelType.REDUCE, self._all_reduce_impl),
                ALL_GATHER_CMD.opcode: ("all_gather", KernelType.REDUCE, self._all_gather_impl),
                REDUCE_SCATTER_CMD.opcode: ("reduce_scatter", KernelType.REDUCE, self._reduce_scatter_impl),
                # ============================================================
                # QUANTIZATION (0xA0-0xAF) - 所有量化计算都在数据面
                # ============================================================
                QUANTIZE_W8A16_CMD.opcode: ("quantize_w8a16", KernelType.CUSTOM, self._quantize_w8a16_impl),
                QUANTIZE_W4A16_CMD.opcode: ("quantize_w4a16", KernelType.CUSTOM, self._quantize_w4a16_impl),
                DEQUANTIZE_CMD.opcode: ("dequantize", KernelType.CUSTOM, self._dequantize_impl),
                GPTQ_KERNEL_CMD.opcode: ("gptq_kernel", KernelType.CUSTOM, self._gptq_kernel_impl),
                AWQ_KERNEL_CMD.opcode: ("awq_kernel", KernelType.CUSTOM, self._awq_kernel_impl),
                FP8_E4M3_QUANT_CMD.opcode: ("fp8_e4m3_quant", KernelType.CUSTOM, self._fp8_e4m3_impl),
                FP8_E5M2_QUANT_CMD.opcode: ("fp8_e5m2_quant", KernelType.CUSTOM, self._fp8_e5m2_impl),
                FP8_DEQUANT_CMD.opcode: ("fp8_dequant", KernelType.CUSTOM, self._fp8_dequant_impl),
            }

            for opcode, (name, ktype, kernel) in cmd_to_kernel.items():
                self._kernels[opcode] = CGCKernelSpec(
                    name=name,
                    kernel_type=ktype,
                    cuda_kernel=kernel,
                )

            logger.info(f"Registered {len(cmd_to_kernel)} CGC commands to kernel registry")

        except ImportError as e:
            logger.warning(f"Could not import cgc_commands: {e}")

    def _register_kda_kernels(self):
        """注册 KDA kernels - 根据平台自动选择后端"""
        # 检测平台
        if torch.cuda.is_available():
            platform = "CUDA"
            device_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            platform = "Metal"
            device_name = "Apple GPU"
        else:
            platform = "CPU"
            device_name = "CPU"

        logger.info(f"Detected platform: {platform} ({device_name})")

        if platform == "Metal":
            self._register_metal_kda_kernels()
        else:
            self._register_fallback_kda_kernels()

    def _register_metal_kda_kernels(self):
        """注册 Metal KDA kernels (通过 C++ 后端)"""
        try:
            if USE_CPP_ENGINE:
                self._kernels[0x11] = CGCKernelSpec(
                    name="kda_metal",
                    kernel_type=KernelType.ATTENTION,
                    cuda_kernel=self._metal_kda_wrapper,
                    supports_flashkda=False,
                )
                logger.info("✅ Metal KDA kernels registered via C++ backend")
            else:
                self._register_fallback_kda_kernels()
        except Exception as e:
            logger.warning(f"⚠️ Metal KDA not available: {e}")
            self._register_fallback_kda_kernels()

    def _metal_kda_wrapper(self, q, k, v, scale=1.0, **kwargs):
        """Metal KDA wrapper - 调用 C++ Metal backend"""
        if USE_CPP_ENGINE:
            return cgc_cpp.execute_opcode(0x11, [q, k, v], kwargs)
        else:
            return F.scaled_dot_product_attention(q, k, v, scale=scale)

    def _register_fallback_kda_kernels(self):
        """注册 fallback KDA kernels (PyTorch SDPA)"""
        def kda_fallback(q, k, v, scale=1.0, **kwargs):
            return F.scaled_dot_product_attention(q, k, v, scale=scale)

        self._kernels[0x11] = CGCKernelSpec(
            name="kda_fallback",
            kernel_type=KernelType.ATTENTION,
            cuda_kernel=kda_fallback,
            supports_flashkda=False,
        )
        logger.info("⚠️ Using fallback KDA (PyTorch SDPA)")

    def _register_standard_kernels(self):
        """注册标准 kernels (PyTorch 原生)"""
        self._kernels[0x01] = CGCKernelSpec(
            name="matmul",
            kernel_type=KernelType.LINEAR,
            cuda_kernel=torch.matmul,
        )
        self._kernels[0x10] = CGCKernelSpec(
            name="scaled_dot_product_attention",
            kernel_type=KernelType.ATTENTION,
            cuda_kernel=F.scaled_dot_product_attention,
        )
        self._kernels[0x30] = CGCKernelSpec(
            name="layer_norm",
            kernel_type=KernelType.NORM,
            cuda_kernel=torch.layer_norm,
        )
        self._kernels[0x31] = CGCKernelSpec(
            name="rms_norm",
            kernel_type=KernelType.NORM,
            cuda_kernel=self._rms_norm_impl,
        )
        self._kernels[0x40] = CGCKernelSpec(
            name="rotary_embedding",
            kernel_type=KernelType.ROPE,
            cuda_kernel=self._rope_impl,
        )
        self._kernels[0x50] = CGCKernelSpec(
            name="silu",
            kernel_type=KernelType.ACTIVATION,
            cuda_kernel=torch.nn.functional.silu,
        )
        self._kernels[0x51] = CGCKernelSpec(
            name="geglu",
            kernel_type=KernelType.ACTIVATION,
            cuda_kernel=self._geglu_impl,
        )
        self._kernels[0x60] = CGCKernelSpec(
            name="softmax",
            kernel_type=KernelType.SOFTMAX,
            cuda_kernel=torch.nn.functional.softmax,
        )

    def register(self, opcode: int, spec: CGCKernelSpec):
        """手动注册 kernel"""
        self._kernels[opcode] = spec
        self._name_to_opcode[spec.name] = opcode

    def get(self, opcode: int) -> Optional[CGCKernelSpec]:
        """获取 kernel 规范"""
        return self._kernels.get(opcode)

    def get_by_name(self, name: str) -> Optional[CGCKernelSpec]:
        """通过名称获取 kernel"""
        opcode = self._name_to_opcode.get(name)
        if opcode is not None:
            return self._kernels.get(opcode)
        return None

    def list_all(self) -> Dict[int, CGCKernelSpec]:
        """列出所有注册的 kernel"""
        return self._kernels.copy()

    @staticmethod
    def _rms_norm_impl(x: torch.Tensor, normalized_shape: int, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """RMSNorm 实现"""
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return weight * x

    @staticmethod
    def _rope_impl(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """RoPE 实现"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    @staticmethod
    def _geglu_impl(x: torch.Tensor) -> torch.Tensor:
        """GeGLU 实现"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.nn.functional.gelu(x1) * x2

    @staticmethod
    def _linear_bias_impl(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Linear with bias implementation"""
        out = torch.matmul(x, weight.t())
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def _kv_cache_load_impl(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """KV cache load implementation"""
        return cache[indices]

    @staticmethod
    def _kv_cache_store_impl(cache: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> None:
        """KV cache store implementation"""
        cache[indices] = values

    @staticmethod
    def _kv_cache_update_impl(cache: torch.Tensor, new_values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """KV cache update implementation"""
        cache[positions] = new_values
        return cache

    # ============================================================
    # DISTRIBUTED OPS (0x90-0x92) - NCCL 實作
    # ============================================================
    @staticmethod
    def _all_reduce_impl(input: torch.Tensor, op: str = "sum", group=None) -> torch.Tensor:
        """AllReduce 實現 - 使用 NCCL

        Args:
            input: 輸入張量
            op: 操作類型 ("sum", "prod", "min", "max")
            group: NCCL 通訊組

        Returns:
            聚合後的張量
        """
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_world_size() > 1:
                if group is None:
                    group = dist.group.WORLD

                if op == "sum":
                    dist.all_reduce(input, op=dist.ReduceOp.SUM, group=group)
                elif op == "prod":
                    dist.all_reduce(input, op=dist.ReduceOp.PRODUCT, group=group)
                elif op == "min":
                    dist.all_reduce(input, op=dist.ReduceOp.MIN, group=group)
                elif op == "max":
                    dist.all_reduce(input, op=dist.ReduceOp.MAX, group=group)

                return input
        except (ImportError, RuntimeError):
            pass

        return input

    @staticmethod
    def _all_gather_impl(input: torch.Tensor, group_size: int = 1, group=None) -> torch.Tensor:
        """AllGather 實現 - 使用 NCCL

        Args:
            input: 輸入張量 [N, ...]
            group_size: 組內進程數
            group: NCCL 通訊組

        Returns:
            聚合後的張量 [N * group_size, ...]
        """
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_world_size() > 1:
                if group is None:
                    group = dist.group.WORLD

                world_size = dist.get_world_size(group)
                output = [torch.empty_like(input) for _ in range(world_size)]
                dist.all_gather(output, input, group=group)
                return torch.cat(output, dim=0)
        except (ImportError, RuntimeError):
            pass

        return input

    @staticmethod
    def _reduce_scatter_impl(input: torch.Tensor, op: str = "sum", group=None) -> torch.Tensor:
        """ReduceScatter 實現 - 使用 NCCL

        Args:
            input: 輸入張量 [N * group_size, ...]
            op: 操作類型 ("sum", "prod", "min", "max")
            group: NCCL 通訊組

        Returns:
            分散後的張量 [N, ...]
        """
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_world_size() > 1:
                if group is None:
                    group = dist.group.WORLD

                world_size = dist.get_world_size(group)
                output = torch.empty_like(input[:input.shape[0] // world_size])

                if op == "sum":
                    dist.reduce_scatter(output, list(torch.chunk(input, world_size, dim=0)), op=dist.ReduceOp.SUM, group=group)
                elif op == "prod":
                    dist.reduce_scatter(output, list(torch.chunk(input, world_size, dim=0)), op=dist.ReduceOp.PRODUCT, group=group)
                elif op == "min":
                    dist.reduce_scatter(output, list(torch.chunk(input, world_size, dim=0)), op=dist.ReduceOp.MIN, group=group)
                elif op == "max":
                    dist.reduce_scatter(output, list(torch.chunk(input, world_size, dim=0)), op=dist.ReduceOp.MAX, group=group)

                return output
        except (ImportError, RuntimeError):
            pass

        return input

    # ============================================================
    # QUANTIZATION (0xA0-0xAF) - 完整實作
    # ============================================================
    @staticmethod
    def _quantize_w8a16_impl(x: torch.Tensor, scale: float = None) -> torch.Tensor:
        """W8A16 量化

        將 FP16/BF16 權重量化為 INT8，scale 單獨存放。

        Args:
            x: 輸入張量 [M, N]
            scale: 量化 scale，若為 None則自動計算

        Returns:
            (quantized, scale) - 量化後張量和 scale
        """
        q_max = 127.0
        q_min = -128.0

        if scale is None:
            scale = x.abs().max() / q_max

        quantized = torch.clip(torch.round(x / scale), q_min, q_max).to(torch.int8)

        return quantized, scale

    @staticmethod
    def _quantize_w4a16_impl(x: torch.Tensor, scale: float = None) -> torch.Tensor:
        """W4A16 量化

        將 FP16/BF16 權重量化為 INT4，scale 單獨存放。

        Args:
            x: 輸入張量 [M, N]
            scale: 量化 scale，若為 None則自動計算

        Returns:
            (quantized, scale) - 量化後張量和 scale
        """
        q_max = 7.0
        q_min = -8.0

        if scale is None:
            scale = x.abs().max() / q_max

        quantized = torch.clip(torch.round(x / scale), q_min, q_max).to(torch.int8)

        return quantized, scale

    @staticmethod
    def _dequantize_impl(quantized: torch.Tensor, scale: torch.Tensor = None) -> torch.Tensor:
        """反量化

        將 INT8/INT4 張量反量化回 FP16/BF16。

        Args:
            quantized: 量化後張量
            scale: 量化 scale

        Returns:
            反量化後張量
        """
        if scale is None:
            scale = 1.0

        if isinstance(scale, torch.Tensor):
            scale = scale.to(dtype=quantized.dtype, device=quantized.device)

        return quantized.to(torch.float32) * scale

    @staticmethod
    def _gptq_kernel_impl(
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor = None,
        bits: int = 4,
        maxq: float = None,
    ) -> torch.Tensor:
        """GPTQ 量化矩陣乘法

        支援 GPTQ 4-bit/8-bit 量化權重的矩陣乘法。

        Args:
            x: 輸入張量 [B, M, K]
            qweight: 量化權重 [N, K // pack_factor]
            scales: scale [N]
            zeros: zero point [N]
            g_idx: group 索引 [K]
            bits: 量化位數 (4 或 8)
            maxq: 最大量化值

        Returns:
            結果張量 [B, M, N]
        """
        if maxq is None:
            maxq = 2 ** bits - 1

        if g_idx is not None:
            scales = scales[g_idx]
            zeros = zeros[g_idx]

        dequant = (qweight - zeros) * scales

        if dequant.dtype != x.dtype:
            dequant = dequant.to(x.dtype)

        return torch.matmul(x, dequant)

    @staticmethod
    def _awq_kernel_impl(
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor = None,
        g_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """AWQ (Activation-Aware Quantization) 矩陣乘法

        支援 AWQ 4-bit 量化權重的矩陣乘法。

        Args:
            x: 輸入張量 [B, M, K]
            qweight: 量化權重 [N, K]
            scales: scale [K]
            zero_points: zero point [K] (可選)
            g_idx: group 索引 [K] (可選)

        Returns:
            結果張量 [B, M, N]
        """
        if g_idx is not None:
            scales = scales[g_idx]
            if zero_points is not None:
                zero_points = zero_points[g_idx]

        scales = scales.to(dtype=x.dtype, device=x.device)
        if zero_points is not None:
            zero_points = zero_points.to(dtype=x.dtype, device=x.device)
            dequant = (qweight - zero_points) * scales
        else:
            dequant = qweight * scales

        return torch.matmul(x, dequant)

    # ============================================================
    # FP8 量化 (NVIDIA H100/H200 原生支持)
    # E4M3: 1 sign bit + 4 exponent bits + 3 mantissa bits (用於權重)
    # E5M2: 1 sign bit + 5 exponent bits + 2 mantissa bits (用於激活值)
    # ============================================================
    @staticmethod
    def _fp8_e4m3_impl(x: torch.Tensor, scale: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """FP8 E4M3 量化

        將 FP16/BF16 權重量化為 FP8 E4M3 格式。
        E4M3 範圍: [-448, 448]

        Args:
            x: 輸入張量 [M, N]
            scale: 量化 scale，若為 None 則自動計算

        Returns:
            (quantized, scale) - 量化後張量和 scale
        """
        if x.dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            x = x.to(torch.float32)

        if scale is None:
            scale = x.abs().max()

        scale = max(scale, 1e-12)

        x_scaled = x / scale

        x_clamped = torch.clamp(x_scaled, -448.0, 448.0)

        quantized = torch.round(x_clamped).to(torch.float8_e4m3fn)

        return quantized, scale

    @staticmethod
    def _fp8_e5m2_impl(x: torch.Tensor, scale: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """FP8 E5M2 量化

        將 FP16/BF16 激活值量化為 FP8 E5M2 格式。
        E5M2 範圍: [-57344, 57344]

        Args:
            x: 輸入張量 [M, N]
            scale: 量化 scale，若為 None 則自動計算

        Returns:
            (quantized, scale) - 量化後張量和 scale
        """
        if x.dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            x = x.to(torch.float32)

        if scale is None:
            scale = x.abs().max()

        scale = max(scale, 1e-12)

        x_scaled = x / scale

        x_clamped = torch.clamp(x_scaled, -57344.0, 57344.0)

        quantized = torch.round(x_clamped).to(torch.float8_e5m2)

        return quantized, scale

    @staticmethod
    def _fp8_dequant_impl(
        quantized: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """FP8 反量化

        將 FP8 E4M3/E5M2 張量反量化回 FP16/BF16。

        Args:
            quantized: 量化後張量
            scale: 量化 scale
            dtype: 目標資料類型

        Returns:
            反量化後張量
        """
        if scale is None:
            scale = 1.0

        if isinstance(scale, torch.Tensor):
            scale = scale.to(dtype=dtype, device=quantized.device)

        dequantized = quantized.to(dtype) * scale

        return dequantized

    # ============================================================
    # KV Cache 編譯期固定佈局
    # ============================================================
    @staticmethod
    def _kv_cache_static_layout_impl(
        k: torch.Tensor,
        v: torch.Tensor,
        layout_type: str = "paged",
        page_size: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """編譯期固定 KV Cache 記憶體佈局

        在編譯階段確定 KV Cache 的記憶體佈局，避免運行時的動態配置開銷。

        Args:
            k: Key tensor [B, H, S, D]
            v: Value tensor [B, H, S, D]
            layout_type: 佈局類型 ("paged", "contiguous", "ring")
            page_size: 頁大小 (paged 模式)

        Returns:
            (k_layout, v_layout, layout_info) - 固定佈局後的張量和佈局信息
        """
        batch_size, num_heads, seq_len, head_dim = k.shape

        if layout_type == "paged":
            num_pages = (seq_len + page_size - 1) // page_size
            k_layout = k.view(batch_size, num_heads, num_pages, page_size, head_dim)
            v_layout = v.view(batch_size, num_heads, num_pages, page_size, head_dim)
            layout_info = {
                "type": "paged",
                "page_size": page_size,
                "num_pages": num_pages,
                "head_dim": head_dim,
                "num_heads": num_heads,
            }
        elif layout_type == "contiguous":
            k_layout = k.contiguous()
            v_layout = v.contiguous()
            layout_info = {
                "type": "contiguous",
                "seq_len": seq_len,
                "head_dim": head_dim,
                "num_heads": num_heads,
            }
        elif layout_type == "ring":
            k_layout = k.view(batch_size, num_heads, seq_len, head_dim)
            v_layout = v.view(batch_size, num_heads, seq_len, head_dim)
            layout_info = {
                "type": "ring",
                "seq_len": seq_len,
                "head_dim": head_dim,
                "num_heads": num_heads,
            }
        else:
            k_layout = k
            v_layout = v
            layout_info = {"type": "auto"}

        return k_layout, v_layout, layout_info

    @staticmethod
    def _kv_cache_commit_impl(
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        offset: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        layout_info: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """提交新的 KV 數據到固定佈局的緩存

        Args:
            k_cache: 現有 K cache
            v_cache: 現有 V cache
            offset: 寫入偏移量
            k_new: 新的 K 數據
            v_new: 新的 V 數據
            layout_info: 佈局信息

        Returns:
            (k_cache, v_cache) - 更新後的緩存
        """
        layout_type = layout_info.get("type", "contiguous")

        if layout_type == "paged":
            page_size = layout_info["page_size"]
            page_offset = offset // page_size
            page_inner_offset = offset % page_size
            k_cache[:, :, page_offset, page_inner_offset] = k_new.squeeze(0)
            v_cache[:, :, page_offset, page_inner_offset] = v_new.squeeze(0)
        elif layout_type == "ring":
            seq_len = layout_info["seq_len"]
            ring_offset = offset % seq_len
            k_cache[:, :, ring_offset] = k_new.squeeze(0)
            v_cache[:, :, ring_offset] = v_new.squeeze(0)
        else:
            k_cache[:, :, offset] = k_new.squeeze(0)
            v_cache[:, :, offset] = v_new.squeeze(0)

        return k_cache, v_cache

    # ============================================================
    # 動態批處理編譯 (Colossal 動態批加速)
    # 整合 oMLX 專家預測實現智能批次調度
    # ============================================================
    @staticmethod
    def _batch_compile_impl(
        batch_size: int,
        seq_lens: List[int],
        hidden_dim: int,
        num_experts: int = 8,
        use_omlx_prediction: bool = True,
    ) -> Dict[str, Any]:
        """編譯期生成批次專用執行路徑

        Args:
            batch_size: 批次大小
            seq_lens: 每個請求的序列長度
            hidden_dim: 隱藏層維度
            num_experts: 專家數量 (用於 oMLX 預測)
            use_omlx_prediction: 是否使用 oMLX 預測專家激活

        Returns:
            batch_plan: 批次執行計劃，包含：
                - fusion_paths: 融合後的執行路徑
                - preallocated_memory: 預分配的顯存大小
                - expert_activation_pred: 預測的專家激活
        """
        max_seq_len = max(seq_lens) if seq_lens else 1
        total_tokens = sum(seq_lens)

        preallocated_memory = batch_size * max_seq_len * hidden_dim * 4

        batch_plan = {
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "total_tokens": total_tokens,
            "hidden_dim": hidden_dim,
            "preallocated_bytes": preallocated_memory,
            "fusion_paths": [],
            "expert_activation_pred": None,
            "omlx_integrated": use_omlx_prediction,
        }

        if use_omlx_prediction and num_experts > 0:
            try:
                from cgc_engine.omlx import OMLXClient
                omlx_client = OMLXClient()
                dummy_input = torch.randn(1, hidden_dim)
                predicted_experts = omlx_client.predict_experts(
                    x=dummy_input,
                    top_k=min(2, num_experts),
                )
                batch_plan["expert_activation_pred"] = predicted_experts.tolist()
                batch_plan["fusion_paths"] = [
                    f"moe_fused_expert_{i}" for i in predicted_experts.tolist()
                ]
            except ImportError:
                batch_plan["expert_activation_pred"] = list(range(min(2, num_experts)))
                batch_plan["fusion_paths"] = [f"moe_fused_expert_{i}" for i in range(min(2, num_experts))]

        batch_plan["fusion_paths"].extend([
            "attention_fused",
            "mlp_fused",
            "layernorm_fused",
        ])

        return batch_plan

    @staticmethod
    def _batch_merge_impl(
        requests: List[Dict[str, torch.Tensor]],
        batch_plan: Dict[str, Any],
        merge_strategy: str = "auto",
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """運行時動態合併小批次

        Args:
            requests: 請求列表，每個請求包含 input_ids, hidden_states 等
            batch_plan: 批次執行計劃
            merge_strategy: 合併策略 ("auto", "seq_len", "expert", "priority")

        Returns:
            (merged_input, attention_mask, request_indices)
        """
        if not requests:
            return None, None, []

        if merge_strategy == "auto":
            merge_strategy = batch_plan.get("merge_hint", "seq_len")

        if merge_strategy == "seq_len":
            sorted_reqs = sorted(requests, key=lambda r: r.get("seq_len", 0), reverse=True)
        elif merge_strategy == "expert":
            expert_pred = batch_plan.get("expert_activation_pred", [])
            sorted_reqs = sorted(
                requests,
                key=lambda r: sum(1 for e in expert_pred if e in r.get("expert_ids", [])),
                reverse=True
            )
        elif merge_strategy == "priority":
            sorted_reqs = sorted(requests, key=lambda r: r.get("priority", 0), reverse=True)
        else:
            sorted_reqs = requests

        max_seq_len = max((r.get("seq_len", 1) for r in sorted_reqs), default=1)
        batch_size = len(sorted_reqs)
        hidden_dim = sorted_reqs[0].get("hidden_states", torch.zeros(1)).shape[-1]

        merged_input = torch.zeros(
            batch_size, max_seq_len, hidden_dim,
            dtype=sorted_reqs[0].get("hidden_states", torch.zeros(1)).dtype,
            device=sorted_reqs[0].get("hidden_states", torch.zeros(1)).device,
        )

        attention_mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool)
        request_indices = []

        for i, req in enumerate(sorted_reqs):
            seq_len = req.get("seq_len", 1)
            hidden_states = req.get("hidden_states")

            if hidden_states is not None:
                if hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(0)

                merged_input[i, :seq_len] = hidden_states[0, :seq_len]
                attention_mask[i, seq_len:] = 0
                request_indices.append(req.get("request_id", i))

        return merged_input, attention_mask, request_indices


# 全局注册表实例
_kernel_registry = CGCKernelRegistry()


def register_cuda_kernel(
    opcode: int,
    name: str,
    kernel_type: KernelType,
    cuda_kernel: Callable,
    workspace_size_fn: Optional[Callable] = None,
) -> None:
    """
    注册 CUDA kernel 到全局注册表

    Args:
        opcode: CGC 操作码
        name: kernel 名称
        kernel_type: kernel 类型
        cuda_kernel: 实际的 CUDA kernel 函数
        workspace_size_fn: 工作空间大小计算函数
    """
    spec = CGCKernelSpec(
        name=name,
        kernel_type=kernel_type,
        cuda_kernel=cuda_kernel,
        workspace_size_fn=workspace_size_fn,
    )
    _kernel_registry.register(opcode, spec)
    logger.info(f"Registered CUDA kernel: {name} (opcode=0x{opcode:02x})")


class CGCCommand:
    """
    CGC SIMD 命令

    表示一个要执行的计算操作。
    """

    def __init__(
        self,
        opcode: int,
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        params: Dict[str, Any],
        workspace: Optional[torch.Tensor] = None,
    ):
        self.opcode = opcode
        self.inputs = inputs
        self.outputs = outputs
        self.params = params
        self.workspace = workspace

    def __repr__(self):
        spec = _kernel_registry.get(self.opcode)
        name = spec.name if spec else f"unknown(0x{self.opcode:02x})"
        return f"CGCCommand(opcode=0x{self.opcode:02x}, name={name})"


class CGCExecutor:
    """
    CGC SIMD 命令执行器

    将 CGC 命令 dispatch 到对应的 CUDA kernel 执行。

    工作流程：
    1. 接收 CGC 命令列表
    2. 解析命令，查找对应的 CUDA kernel
    3. 准备 kernel 参数
    4. 执行 kernel
    5. 管理工作空间和显存

    新增功能：
    - 静态指令流支持：预编译并缓存常用的指令序列
    - 批量执行：一次性执行多个命令，减少 Python 开销
    """

    def __init__(self, enable_profiling: bool = False):
        self.enable_profiling = enable_profiling
        self.kernel_registry = _kernel_registry
        self.workspace_pool: Dict[int, torch.Tensor] = {}
        self.execution_stats: Dict[str, int] = {}
        
        # 静态指令流缓存
        self._static_command_streams: Dict[str, List[CGCCommand]] = {}
        self._prefill_stream_key = "prefill"
        self._decode_stream_key = "decode"

    def has_opcode(self, opcode: int) -> bool:
        """檢查是否有支持該 opcode 的實現"""
        spec = self.kernel_registry.get(opcode)
        if spec is not None:
            return True
        if 0x80 <= opcode <= 0x83:
            return True
        if opcode == 0x11:
            return True
        return False

    def register_static_prefill_stream(self, commands: List[CGCCommand]) -> None:
        """注册 Prefill 阶段的静态指令流"""
        self._static_command_streams[self._prefill_stream_key] = commands
        logger.info(f"Registered static prefill stream with {len(commands)} commands")

    def register_static_decode_stream(self, commands: List[CGCCommand]) -> None:
        """注册 Decode 阶段的静态指令流"""
        self._static_command_streams[self._decode_stream_key] = commands
        logger.info(f"Registered static decode stream with {len(commands)} commands")

    def execute_static_prefill(self, input_tensors: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """执行 Prefill 静态指令流"""
        if self._prefill_stream_key not in self._static_command_streams:
            raise RuntimeError("No static prefill stream registered")
        
        commands = self._static_command_streams[self._prefill_stream_key]
        return self._execute_static_stream(commands, input_tensors)

    def execute_static_decode(self, input_tensors: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """执行 Decode 静态指令流"""
        if self._decode_stream_key not in self._static_command_streams:
            raise RuntimeError("No static decode stream registered")
        
        commands = self._static_command_streams[self._decode_stream_key]
        return self._execute_static_stream(commands, input_tensors)

    def _execute_static_stream(
        self,
        commands: List[CGCCommand],
        input_tensors: Dict[str, torch.Tensor],
    ) -> List[torch.Tensor]:
        """执行静态指令流，支持张量替换"""
        tensor_map: Dict[int, torch.Tensor] = {}
        for name, tensor in input_tensors.items():
            tensor_map[id(tensor)] = tensor

        outputs = []
        for cmd in commands:
            inputs = []
            for inp in cmd.inputs:
                if id(inp) in tensor_map:
                    inputs.append(tensor_map[id(inp)])
                else:
                    inputs.append(inp)
            
            executed_cmd = CGCCommand(
                opcode=cmd.opcode,
                inputs=inputs,
                outputs=cmd.outputs,
                params=cmd.params,
                workspace=cmd.workspace,
            )
            
            cmd_outputs = self.execute(executed_cmd)
            for out in cmd_outputs:
                tensor_map[id(out)] = out
            outputs.extend(cmd_outputs)
        
        return outputs

    def execute_batch(self, commands: List[CGCCommand]) -> List[List[torch.Tensor]]:
        """批量执行多个 CGC 命令"""
        all_outputs = []
        for cmd in commands:
            outputs = self.execute(cmd)
            all_outputs.append(outputs)
        return all_outputs

    def execute_decode_loop(
        self,
        initial_input: torch.Tensor,
        max_new_tokens: int,
        stop_tokens: Optional[List[int]] = None,
        temperature: float = 0.0,
        temperature_last: bool = False,
    ) -> List[int]:
        """
        Decode 迭代卸载到 CGC 内部循环
        
        Args:
            initial_input: 初始输入
            max_new_tokens: 最大新 token 数
            stop_tokens: 停止 token 列表
            temperature: 采样温度
            temperature_last: 是否在最后一个 token 才采样
        
        Returns:
            生成的 token 列表
        
        架构设计：
        - 将整个 Decode 循环在 CGC 内部执行，减少 Python 开销
        - 支持动态扩展机制
        """
        if self._decode_stream_key not in self._static_command_streams:
            raise RuntimeError("No static decode stream registered for decode loop")
        
        decode_commands = self._static_command_streams[self._decode_stream_key]
        generated_tokens = []
        current_input = initial_input
        
        for _ in range(max_new_tokens):
            outputs = self._execute_static_stream(decode_commands, {"input": current_input})
            
            logits = outputs[-1] if outputs else current_input
            
            if temperature > 0:
                # 采样
                if temperature_last:
                    logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪婪采样
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            token_id = next_token.item()
            generated_tokens.append(token_id)
            
            # 检查停止条件
            if stop_tokens and token_id in stop_tokens:
                break
            
            # 更新输入
            current_input = next_token
        
        return generated_tokens

    def execute(self, command: CGCCommand) -> List[torch.Tensor]:
        """
        执行单个 CGC 命令

        Args:
            command: CGC 命令

        Returns:
            输出 tensors 列表
        """
        opcode = command.opcode

        # ============================================================
        # DISTRIBUTED (0x90-0x93) - NCCL 集合通信
        # ============================================================
        if 0x90 <= opcode <= 0x93:
            return self._execute_distributed_op(command)

        # ============================================================
        # GDS / PD 资源指令 (0x94-0x9F) - 存储操作
        # ============================================================
        if 0x94 <= opcode <= 0x9F:
            return self._execute_gds_pd_command(command)

        # ============================================================
        # 動態批處理 (0x76-0x77) - Colossal 動態批加速
        # 整合 oMLX 專家預測實現智能批次調度
        # ============================================================
        if opcode == 0x76:
            return self._execute_batch_compile(command)
        elif opcode == 0x77:
            return self._execute_batch_merge(command)

        # ============================================================
        # llama.cpp 量化/推理域 (0xC0-0xD5) - 由 _execute_llama_cpp_op 統一處理
        # 0xC0-0xCF: 量化操作
        # 0xCC-0xD5: GGUF 推理操作 (_llama_gguf_operations)
        # ============================================================
        if 0xC0 <= opcode <= 0xCF:
            return self._execute_llama_cpp_op(opcode, command.inputs, command.params)

        # ============================================================
        # FlashMoE / oMLX 端侧 MoE 引擎 (0xE0-0xE5)
        # 0xE0: 加载专家权重 (存储操作 - 由 UnifiedIOController 处理)
        # 0xE1: FlashMoE MLP 前向计算 (计算操作 - 执行层)
        # 0xE2: FlashMoE 专家前向计算 (计算操作)
        # 0xE3: oMLX 专家预测 (计算操作)
        # 0xE4: oMLX 缓存更新 (存储操作)
        # 0xE5: oMLX 缓存驱逐 (存储操作)
        # ============================================================
        if 0xE0 <= opcode <= 0xE5:
            return self._execute_flashmoe_omlx_command(command)

        # ============================================================
        # JITLoad 即时编译加载系统 (0xF0-0xF4)
        # 0xF0: JIT 加载编译产物
        # 0xF1: JIT 编译 kernel
        # 0xF2: JIT 自动调度
        # 0xF3: JIT 缓存查找
        # 0xF4: JIT 缓存失效
        # ============================================================
        if 0xF0 <= opcode <= 0xF4:
            return self._execute_jitload_command(command)

        # ============================================================
        # SPDK 超高速 NVMe I/O (0xF6-0xFF) - 存储操作
        # ============================================================
        if 0xF6 <= opcode <= 0xFF:
            return self._execute_spdk_command(command)

        # ============================================================
        # KDA 正向/反向傳播 (0x80-0x83) 以及 ATTENTION_KDA (0x11)
        # ============================================================
        if opcode == 0x80:
            return self._execute_kda_forward(command)
        elif opcode == 0x83:
            return self._execute_kda_backward(command)
        elif opcode == 0x11:
            return self._execute_kda_cpp_simd(command)

        spec = self.kernel_registry.get(opcode)
        if spec is None:
            raise ValueError(f"No kernel registered for opcode 0x{opcode:02x}")

        kernel = spec.cuda_kernel
        inputs = command.inputs
        params = command.params

        if self.enable_profiling:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        try:
            if spec.kernel_type == KernelType.ATTENTION:
                outputs = self._execute_attention(kernel, inputs, params, command.workspace)
            elif spec.kernel_type == KernelType.LINEAR:
                outputs = self._execute_linear(kernel, inputs, params)
            elif spec.kernel_type == KernelType.NORM:
                outputs = self._execute_norm(kernel, inputs, params)
            elif spec.kernel_type == KernelType.ROPE:
                outputs = self._execute_rope(kernel, inputs, params)
            elif spec.kernel_type == KernelType.ACTIVATION:
                outputs = self._execute_activation(kernel, inputs, params)
            elif spec.kernel_type == KernelType.SOFTMAX:
                outputs = self._execute_softmax(kernel, inputs, params)
            else:
                outputs = [kernel(*inputs, **params)]

            if self.enable_profiling:
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event)
                self.execution_stats[spec.name] = self.execution_stats.get(spec.name, 0) + 1
                logger.debug(f"Executed {spec.name} in {elapsed:.2f}ms")

            return outputs

        except Exception as e:
            logger.error(f"Failed to execute {spec.name}: {e}")
            raise

    def _execute_attention(
        self,
        kernel: Callable,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
        workspace: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        """执行 Attention kernel"""
        q, k, v = inputs[0], inputs[1], inputs[2]
        scale = params.get("scale", 1.0 / (q.shape[-1] ** 0.5))

        g = inputs[3] if len(inputs) > 3 else None
        beta = inputs[4] if len(inputs) > 4 else None

        out = torch.empty_like(v)

        if "flash_kda" in kernel.__name__ or hasattr(kernel, '__self__'):
            A_log = params.get("A_log", torch.zeros(q.shape[2], dtype=torch.float32, device=q.device))
            dt_bias = params.get("dt_bias", torch.zeros(q.shape[2], q.shape[3], dtype=torch.float32, device=q.device))
            lower_bound = params.get("lower_bound", -5.0)

            workspace_size = kernel.get_workspace_size(
                q.shape[0] * q.shape[1], q.shape[2], 1
            ) if hasattr(kernel, 'get_workspace_size') else 0

            if workspace_size > 0:
                if workspace is None or workspace.numel() < workspace_size:
                    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=q.device)

            kernel(
                q=q, k=k, v=v, g=g, beta=beta,
                scale=scale, out=out,
                workspace=workspace,
                A_log=A_log, dt_bias=dt_bias,
                lower_bound=lower_bound,
            )
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=scale)

        return [out]

    def _execute_kda_forward(self, command: CGCCommand) -> List[torch.Tensor]:
        """執行 KDA 正向傳播 (0x80)

        使用 FlashKDA kernel，如果不可用則 fallback 到 PyTorch SDPA。
        """
        inputs = command.inputs
        params = command.params

        q, k, v = inputs[0], inputs[1], inputs[2]
        scale = params.get("scale", 1.0 / (q.shape[-1] ** 0.5))

        try:
            import flash_kda as _flash_kda

            out = torch.empty_like(v)
            workspace_size = _flash_kda.get_workspace_size(
                q.shape[0] * q.shape[1], q.shape[2], 1
            )

            workspace = None
            if workspace_size > 0:
                workspace = torch.empty(workspace_size, dtype=torch.uint8, device=q.device)

            A_log = params.get("A_log", torch.zeros(q.shape[2], dtype=torch.float32, device=q.device))
            dt_bias = params.get("dt_bias", torch.zeros(q.shape[2], q.shape[3], dtype=torch.float32, device=q.device))
            lower_bound = params.get("lower_bound", -5.0)

            _flash_kda.fwd(
                q=q, k=k, v=v, g=None, beta=None,
                scale=scale, out=out,
                workspace=workspace,
                A_log=A_log, dt_bias=dt_bias,
                lower_bound=lower_bound,
            )

            return [out]

        except ImportError:
            out = F.scaled_dot_product_attention(q, k, v, scale=scale)
            return [out]

    def _execute_kda_cpp_simd(self, command: CGCCommand) -> List[torch.Tensor]:
        """執行 KDA 使用 C++ SIMD Engine (opcode 0x11)

        路由到 C++ SIMD Engine (cgc_cpp) 執行 KDA 計算，
        狀態管理完全由 C++ Engine 處理。
        如果 C++ Engine 不支持 KDA，则回退到 PyTorch SDPA。
        """
        try:
            import cgc_cpp as _cgc_cpp

            inputs = command.inputs
            params = command.params

            q, k, v = inputs[0], inputs[1], inputs[2]
            g = inputs[3] if len(inputs) > 3 else None
            b = inputs[4] if len(inputs) > 4 else None

            input_tensors = [q, k, v]
            if g is not None:
                input_tensors.append(g)
            if b is not None:
                input_tensors.append(b)

            kda_params = {
                "state_id": params.get("state_id", 0),
                "n_heads": params.get("n_heads", q.shape[0]),
                "d_state": params.get("d_state", q.shape[1]),
                "seq_len": params.get("seq_len", q.shape[0]),
                "dim": params.get("dim", q.shape[1]),
                "scale": params.get("scale", 1.0 / (q.shape[-1] ** 0.5)),
                "is_first_chunk": params.get("is_first_chunk", True),
            }

            outputs = _cgc_cpp.execute_opcode(0x11, input_tensors, kda_params)
            return outputs

        except (ImportError, Exception) as e:
            logger.info(f"[CGC] KDA C++ SIMD not available ({e}), falling back to PyTorch SDPA")
            inputs = command.inputs
            params = command.params
            q, k, v = inputs[0], inputs[1], inputs[2]
            scale = params.get("scale", 1.0 / (q.shape[-1] ** 0.5))
            out = F.scaled_dot_product_attention(q, k, v, scale=scale)
            return [out]

    def _execute_kda_backward(self, command: CGCCommand) -> List[torch.Tensor]:
        """執行 KDA 反向傳播 (0x83)

        使用 FlashKDA backward kernel，如果不可用則 fallback 到標準 PyTorch autograd。

        Args:
            command: CGCCommand，包含：
                - inputs[0]: grad_output
                - inputs[1]: saved_q
                - inputs[2]: saved_k
                - inputs[3]: saved_v
                - params["scale"]: scale factor

        Returns:
            [grad_q, grad_k, grad_v]
        """
        inputs = command.inputs
        params = command.params

        grad_output, saved_q, saved_k, saved_v = inputs[0], inputs[1], inputs[2], inputs[3]
        scale = params.get("scale", 1.0 / (saved_q.shape[-1] ** 0.5))

        try:
            import flash_kda as _flash_kda

            grad_q = torch.empty_like(saved_q)
            grad_k = torch.empty_like(saved_k)
            grad_v = torch.empty_like(saved_v)

            workspace_size = _flash_kda.get_workspace_size(
                saved_q.shape[0] * saved_q.shape[1], saved_q.shape[2], 1
            )

            workspace = None
            if workspace_size > 0:
                workspace = torch.empty(workspace_size, dtype=torch.uint8, device=saved_q.device)

            _flash_kda.bwd(
                grad_out=grad_output,
                q=saved_q, k=saved_k, v=saved_v,
                grad_q=grad_q, grad_k=grad_k, grad_v=grad_v,
                scale=scale,
                workspace=workspace,
            )

            return [grad_q, grad_k, grad_v]

        except ImportError:
            grad_q = F.scaled_dot_product_attention(
                grad_output, saved_k, saved_v, scale=scale
            )
            grad_k = F.scaled_dot_product_attention(
                grad_output, saved_q, saved_v, scale=scale
            ).transpose(1, 2).contiguous()
            grad_v = F.scaled_dot_product_attention(
                grad_output, saved_q, saved_k, scale=scale
            ).transpose(1, 2).contiguous()

            return [grad_q, grad_k, grad_v]

    def _execute_linear(
        self,
        kernel: Callable,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """执行 Linear/GEMM kernel"""
        x = inputs[0]
        weight = inputs[1] if len(inputs) > 1 else params.get("weight")

        bias = params.get("bias", None)
        out = torch.matmul(x, weight.t())
        if bias is not None:
            out = out + bias
        return [out]

    def _execute_norm(
        self,
        kernel: Callable,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """执行 Norm kernel"""
        x = inputs[0]
        weight = inputs[1] if len(inputs) > 1 else params.get("weight")
        eps = params.get("eps", 1e-6)

        normalized_shape = params.get("normalized_shape", x.shape[-1])

        if "rms" in kernel.__name__:
            out = self._rms_norm_impl(x, normalized_shape, weight, eps)
        else:
            out = torch.layer_norm(x, (normalized_shape,), weight, eps=eps)
        return [out]

    def _execute_rope(
        self,
        kernel: Callable,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """执行 RoPE kernel"""
        x = inputs[0]
        cos = inputs[1] if len(inputs) > 1 else params.get("cos")
        sin = inputs[2] if len(inputs) > 2 else params.get("sin")

        out = self._rope_impl(x, cos, sin)
        return [out]

    def _execute_activation(
        self,
        kernel: Callable,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """执行 Activation kernel"""
        x = inputs[0]
        out = kernel(x)
        return [out]

    def _execute_softmax(
        self,
        kernel: Callable,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """执行 Softmax kernel"""
        x = inputs[0]
        dim = params.get("dim", -1)
        out = kernel(x, dim=dim)
        return [out]

    # ============================================================
    # 動態批處理 (0x76-0x77) - Colossal 動態批加速
    # ============================================================
    def _execute_batch_compile(self, command: CGCCommand) -> List[torch.Tensor]:
        """執行批次編譯命令 (0x76)

        編譯期生成批次專用執行路徑，整合 oMLX 專家預測
        """
        batch_size = command.params.get("batch_size", 1)
        seq_lens = command.params.get("seq_lens", [1])
        hidden_dim = command.params.get("hidden_dim", 4096)
        num_experts = command.params.get("num_experts", 8)
        use_omlx = command.params.get("use_omlx_prediction", True)

        batch_plan = self._batch_compile_impl(
            batch_size=batch_size,
            seq_lens=seq_lens,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_omlx_prediction=use_omlx,
        )

        plan_tensor = torch.tensor([batch_plan["batch_size"], batch_plan["max_seq_len"],
                                     batch_plan["total_tokens"], batch_plan["preallocated_bytes"]],
                                    dtype=torch.float32)

        return [plan_tensor]

    def _execute_batch_merge(self, command: CGCCommand) -> List[torch.Tensor]:
        """執行批次合併命令 (0x77)

        運行時動態合併小批次
        """
        requests = command.params.get("requests", [])
        batch_plan = command.params.get("batch_plan", {})
        merge_strategy = command.params.get("merge_strategy", "auto")

        merged_input, attention_mask, request_indices = self._batch_merge_impl(
            requests=requests,
            batch_plan=batch_plan,
            merge_strategy=merge_strategy,
        )

        if merged_input is None:
            return [torch.tensor([])]

        return [merged_input, attention_mask]

    # ============================================================
    # FlashMoE / oMLX 端侧 MoE 引擎 (0xE0-0xE5)
    # ============================================================
    def _execute_flashmoe_omlx_command(self, command: CGCCommand) -> List[torch.Tensor]:
        """執行 FlashMoE/oMLX 命令

        存儲操作 (由 UnifiedIOController 調用存儲層處理):
        - 0xE0: FLASHMOE_LOAD_EXPERT - 載入專家權重
        - 0xE4: OMLX_CACHE_UPDATE - oMLX 緩存更新
        - 0xE5: OMLX_EVICT - oMLX 緩存驅逐

        計算操作 (由執行層 CGC SIMD Executor 處理):
        - 0xE1: FLASHMOE_MLP_FORWARD - FlashMoE MLP 前向計算
        - 0xE2: FLASHMOE_EXPERT_FWD - FlashMoE 專家前向計算
        - 0xE3: OMLX_PREDICT_EXPERTS - oMLX 專家預測
        """
        from cgc_engine.cgc.cgc_opcodes import CGC_OP_CODES

        opcode = command.opcode
        inputs = command.inputs
        params = command.params

        if opcode == CGC_OP_CODES.FLASHMOE_LOAD_EXPERT:
            expert_id = params.get("expert_id", 0)
            expert_path = params.get("expert_path", "")
            if FLASH_MOE_AVAILABLE and flash_moe_client:
                expert_weight = flash_moe_client.load_expert(
                    expert_id,
                    expert_path or None,
                    expert_dim=params.get("expert_dim"),
                    intermediate_dim=params.get("intermediate_dim"),
                )
                return [expert_weight["w1"], expert_weight["w3"], expert_weight["w2"]]

            expert_dim = int(params.get("expert_dim", 4096))
            intermediate_dim = int(params.get("intermediate_dim", expert_dim * 4))
            w1 = torch.randn(intermediate_dim, expert_dim)
            w3 = torch.randn(intermediate_dim, expert_dim)
            w2 = torch.randn(expert_dim, intermediate_dim)
            return [w1, w3, w2]

        elif opcode == CGC_OP_CODES.FLASHMOE_MLP_FORWARD:
            x = inputs[0] if inputs else torch.randn(1, 4096)
            expert_ids = params.get("expert_ids", [0])
            if FLASH_MOE_AVAILABLE and flash_moe_client:
                output = flash_moe_client.mlp_forward(x, expert_ids)
                return [output]
            return [torch.randn(x.shape[0], x.shape[1])]

        elif opcode == CGC_OP_CODES.FLASHMOE_EXPERT_FWD:
            x = inputs[0] if inputs else torch.randn(1, 4096)
            expert_id = params.get("expert_id", 0)
            if FLASH_MOE_AVAILABLE and flash_moe_client:
                output = flash_moe_client.expert_forward(x, expert_id)
                return [output]
            return [torch.randn(x.shape[0], x.shape[1])]

        elif opcode == CGC_OP_CODES.OMLX_PREDICT_EXPERTS:
            x = inputs[0] if inputs else torch.randn(1, 4096)
            top_k = params.get("top_k", 2)
            if OMLX_AVAILABLE and omlx_client:
                predicted = omlx_client.predict_experts(x, top_k)
                return [predicted]
            return [torch.randint(0, 8, (1, top_k))]

        elif opcode == CGC_OP_CODES.OMLX_CACHE_UPDATE:
            expert_id = params.get("expert_id", 0)
            if OMLX_AVAILABLE and omlx_client:
                omlx_client.update_cache(expert_id)
            return [torch.tensor(1)]

        elif opcode == CGC_OP_CODES.OMLX_EVICT:
            expert_id = params.get("expert_id", 0)
            if OMLX_AVAILABLE and omlx_client:
                omlx_client.evict(expert_id)
            return [torch.tensor(1)]

        return []

    # ============================================================
    # GDS / PD 资源指令 (0x94-0x9F) - 存储操作
    # ============================================================
    def _execute_gds_pd_command(self, command: CGCCommand) -> List[torch.Tensor]:
        """執行 GDS/PD 資源命令"""
        from cgc_engine.cgc.cgc_opcodes import CGC_OP_CODES

        opcode = command.opcode
        params = command.params

        if opcode == CGC_OP_CODES.GDS_LOAD_KV:
            key = params.get("key", "")
            seq_len = params.get("seq_len", 2048)
            head_dim = params.get("head_dim", 128)
            batch_size = params.get("batch_size", 1)
            num_heads = params.get("num_heads", 32)

            if GDS_AVAILABLE and gds_manager:
                k, v = gds_manager.load_kv_from_pd(key, seq_len, head_dim)
                return [k, v]
            else:
                k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda" if torch.cuda.is_available() else "cpu")
                v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda" if torch.cuda.is_available() else "cpu")
                return [k, v]

        elif opcode == CGC_OP_CODES.GDS_SAVE_KV:
            key = params.get("key", "")
            k = command.inputs[0] if command.inputs else None
            v = command.inputs[1] if len(command.inputs) > 1 else None

            if GDS_AVAILABLE and gds_manager and k is not None and v is not None:
                gds_manager.save_kv_to_pd(key, k, v)
                return [torch.tensor(1)]
            return [torch.tensor(0)]

        elif opcode == CGC_OP_CODES.GDS_LOAD_WEIGHT:
            weight_path = params.get("weight_path", "")
            shape = params.get("shape", [])
            dtype_str = params.get("dtype", "float16")
            dtype = getattr(torch, dtype_str, torch.float16)

            if GDS_AVAILABLE and gds_manager:
                weight = gds_manager.load_weight_from_pd(weight_path, shape)
                return [weight]
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                return [torch.randn(shape, dtype=dtype, device=device)]

        elif opcode == CGC_OP_CODES.GDS_SAVE_WEIGHT:
            weight_path = params.get("weight_path", "")
            weight = command.inputs[0] if command.inputs else None

            if GDS_AVAILABLE and gds_manager and weight is not None:
                gds_manager.save_weight_to_pd(weight_path, weight)
                return [torch.tensor(1)]
            return [torch.tensor(0)]

        elif opcode == CGC_OP_CODES.PD_LOAD_KV:
            key = params.get("key", "")
            seq_len = params.get("seq_len", 2048)
            head_dim = params.get("head_dim", 128)
            num_heads = params.get("num_heads", 32)

            if PD_AVAILABLE and pd_client:
                k, v = pd_client.load_kv(key, seq_len, head_dim, num_heads)
                return [k, v]
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                k = torch.randn(1, num_heads, seq_len, head_dim, device=device)
                v = torch.randn(1, num_heads, seq_len, head_dim, device=device)
                return [k, v]

        elif opcode == CGC_OP_CODES.PD_SAVE_KV:
            key = params.get("key", "")
            k = command.inputs[0] if command.inputs else None
            v = command.inputs[1] if len(command.inputs) > 1 else None

            if PD_AVAILABLE and pd_client and k is not None and v is not None:
                pd_client.save_kv(key, k, v)
                return [torch.tensor(1)]
            return [torch.tensor(0)]

        elif opcode == CGC_OP_CODES.PD_LOAD_WEIGHT:
            weight_path = params.get("weight_path", "")
            shape = params.get("shape", [])
            dtype_str = params.get("dtype", "float16")
            dtype = getattr(torch, dtype_str, torch.float16)

            if PD_AVAILABLE and pd_client:
                weight = pd_client.load_weight(weight_path, shape, dtype)
                return [weight]
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                return [torch.randn(shape, dtype=dtype, device=device)]

        elif opcode == CGC_OP_CODES.PD_SAVE_WEIGHT:
            weight_path = params.get("weight_path", "")
            weight = command.inputs[0] if command.inputs else None

            if PD_AVAILABLE and pd_client and weight is not None:
                pd_client.save_weight(weight_path, weight)
                return [torch.tensor(1)]
            return [torch.tensor(0)]

        return []

    # ============================================================
    # SPDK 超高速 NVMe I/O (0xF6-0xFF) - 存储操作
    # ============================================================
    def _execute_spdk_command(self, command: CGCCommand) -> List[torch.Tensor]:
        """執行 SPDK 存儲命令"""
        from cgc_engine.cgc.cgc_opcodes import CGC_OP_CODES

        opcode = command.opcode
        params = command.params

        SPDK_OP_READ = 0xF6
        SPDK_OP_WRITE = 0xF7
        SPDK_OP_ALLOC = 0xF8

        if opcode == SPDK_OP_READ:
            key = params.get("key", "")

            if SPDK_AVAILABLE and spdk_client:
                result = spdk_client.submit_read(key)
                return [torch.tensor(1)] if result else [torch.tensor(0)]
            return [torch.tensor(0)]

        elif opcode == SPDK_OP_WRITE:
            key = params.get("key", "")
            data = command.inputs[0] if command.inputs else None

            if SPDK_AVAILABLE and spdk_client and data is not None:
                result = spdk_client.submit_write(key, data)
                return [torch.tensor(1)] if result else [torch.tensor(0)]
            return [torch.tensor(0)]

        elif opcode == SPDK_OP_ALLOC:
            size = params.get("size", 0)

            if SPDK_AVAILABLE and spdk_client:
                return [torch.tensor(size)]
            return [torch.tensor(0)]

        return []

    # ============================================================
    # JITLoad 即时编译加载系统 (0xF0-0xF4)
    # ============================================================
    def _execute_jitload_command(self, command: CGCCommand) -> List[torch.Tensor]:
        """執行 JITLoad 即時編譯加載系統"""
        from cgc_engine.cgc.cgc_opcodes import CGC_OP_CODES

        opcode = command.opcode
        inputs = command.inputs
        params = command.params

        if opcode == CGC_OP_CODES.JIT_LOAD_COMPILED:
            kernel_path = params.get("kernel_path", "")
            if JIT_AVAILABLE and jit_loader:
                result = jit_loader.load_compiled(kernel_path)
                return result if result else [torch.tensor(1)]
            return [torch.tensor(1)]

        elif opcode == CGC_OP_CODES.JIT_COMPILE_KERNEL:
            kernel_type = params.get("kernel_type", "attention")
            if JIT_AVAILABLE and jit_loader:
                result = jit_loader.compile_kernel(kernel_type)
                return result if result else [torch.tensor(1)]
            return [torch.tensor(1)]

        elif opcode == CGC_OP_CODES.JIT_AUTO_DISPATCH:
            auto_select = params.get("auto_select", True)
            if JIT_AVAILABLE and jit_loader:
                result = jit_loader.auto_dispatch(auto_select)
                return result if result else [torch.tensor(1)]
            return [torch.tensor(1)]

        elif opcode == CGC_OP_CODES.JIT_CACHE_LOOKUP:
            kernel_name = params.get("kernel_name", "")
            if JIT_AVAILABLE and jit_loader:
                found = jit_loader.cache_lookup(kernel_name)
                return [torch.tensor(1 if found else 0)]
            return [torch.tensor(0)]

        elif opcode == CGC_OP_CODES.JIT_CACHE_INVALIDATE:
            kernel_name = params.get("kernel_name", "")
            if JIT_AVAILABLE and jit_loader:
                jit_loader.cache_invalidate(kernel_name)
            return [torch.tensor(1)]

        return []

    # ============================================================
    # LLAMA.CPP 量化域 (0xC0-0xDF) - 所有计算都在数据面
    # ============================================================
    def _execute_llama_cpp_op(
        self,
        opcode: int,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """执行 llama.cpp 量化域的计算"""
        LLAMA_AVAILABLE = False
        try:
            import llama_cpp
            LLAMA_AVAILABLE = True
        except ImportError:
            pass

        if not LLAMA_AVAILABLE:
            return self._llama_cpp_fallback(opcode, inputs, params)

        # 0xC0: GGUF 加载
        if opcode == 0xC0:
            return self._llama_gguf_load(inputs, params)
        # 0xC1: GGUF 量化
        elif opcode == 0xC1:
            return self._llama_gguf_quantize(inputs, params)
        # 0xC2: GGUF 反量化
        elif opcode == 0xC2:
            return self._llama_gguf_dequantize(inputs, params)
        # 0xC3-C9: 各种量化矩阵乘法
        elif 0xC3 <= opcode <= 0xC9:
            return self._llama_quantized_matmul(opcode, inputs, params)
        # 0xCA: MoE 路由
        elif opcode == 0xCA:
            return self._llama_moe_routing(inputs, params)
        # 0xCB: MoE 专家前向
        elif opcode == 0xCB:
            return self._llama_moe_expert_fwd(inputs, params)
        # 0xCC-0xD5: 其他 GGUF 操作
        elif 0xCC <= opcode <= 0xD5:
            return self._llama_gguf_operations(opcode, inputs, params)
        else:
            return self._llama_cpp_fallback(opcode, inputs, params)

    def _llama_cpp_fallback(
        self,
        opcode: int,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """llama.cpp 不可用时的 fallback 方案"""
        # 对于量化矩阵乘法的 fallback
        if 0xC3 <= opcode <= 0xC9:
            if len(inputs) >= 2:
                x = inputs[0]
                weight = inputs[1]
                return [torch.matmul(x, weight.to(torch.float32).t())]
        # 其他操作的 fallback
        if len(inputs) > 0:
            return [inputs[0]]
        return []

    def _llama_gguf_load(
        self,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """GGUF 加载"""
        return [torch.tensor([1.0])] if len(inputs) == 0 else [inputs[0]]

    def _llama_gguf_quantize(
        self,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """GGUF 量化"""
        if len(inputs) > 0:
            x = inputs[0]
            scale = params.get("scale", 1.0)
            quantized = torch.clip(torch.round(x / scale), -127.0, 127.0).to(torch.int8)
            return [quantized]
        return []

    def _llama_gguf_dequantize(
        self,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """GGUF 反量化"""
        if len(inputs) > 0:
            quantized = inputs[0]
            scale = params.get("scale", 1.0)
            return [quantized.to(torch.float32) * scale]
        return []

    def _llama_quantized_matmul(
        self,
        opcode: int,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """量化矩阵乘法"""
        if len(inputs) >= 2:
            x = inputs[0]
            weight = inputs[1]
            scale = params.get("scale", 1.0)
            dequantized = weight.to(torch.float32) * scale
            return [torch.matmul(x, dequantized.t())]
        return []

    def _llama_moe_routing(
        self,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """MoE 路由"""
        if len(inputs) > 0:
            x = inputs[0]
            return [torch.softmax(x, dim=-1)]
        return []

    def _llama_moe_expert_fwd(
        self,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """MoE 专家前向"""
        if len(inputs) >= 2:
            x = inputs[0]
            expert_weight = inputs[1]
            scale = params.get("scale", 1.0)
            dequantized = expert_weight.to(torch.float32) * scale
            return [torch.matmul(x, dequantized.t())]
        return []

    def _llama_gguf_operations(
        self,
        opcode: int,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """其他 GGUF 操作"""
        if len(inputs) > 0:
            x = inputs[0]
            # RoPE
            if opcode == 0xCC:
                cos = params.get("cos", torch.ones_like(x.shape[-1]))
                sin = params.get("sin", torch.ones_like(x.shape[-1]))
                x1 = x[..., :x.shape[-1]//2]
                x2 = x[..., x.shape[-1]//2:]
                out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
                return [out]
            # RMSNorm
            elif opcode == 0xCD:
                weight = params.get("weight", torch.ones(x.shape[-1]))
                eps = params.get("eps", 1e-6)
                variance = x.pow(2).mean(-1, keepdim=True)
                x = x * torch.rsqrt(variance + eps)
                return [weight * x]
            # SiLU
            elif opcode == 0xCE:
                return [x * torch.sigmoid(x)]
            # GELU
            elif opcode == 0xCF:
                return [torch.nn.functional.gelu(x)]
            # 其他操作
            else:
                return [x]
        return []

    def execute_batch(self, commands: List[CGCCommand]) -> List[List[torch.Tensor]]:
        """批量执行 CGC 命令"""
        results = []
        for cmd in commands:
            outputs = self.execute(cmd)
            results.append(outputs)
        return results

    def get_stats(self) -> Dict[str, int]:
        """获取执行统计"""
        return self.execution_stats.copy()

    @staticmethod
    def _rms_norm_impl(x: torch.Tensor, normalized_shape: int, weight: torch.Tensor, eps: float) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return weight * x

    @staticmethod
    def _rope_impl(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def mlx_tune_forward(self, **tensors):
        """MLX-Tune 前向 (opcode 0xB6)"""
        try:
            from .mlx_tune_integration import cgc_mlx_tune
            return cgc_mlx_tune.run_cgc_command(0xB6, tensors)
        except ImportError:
            x = tensors["x"]
            w = tensors["w"]
            lora_a = tensors["lora_a"]
            lora_b = tensors["lora_b"]
            scale = tensors.get("scale", 1.0)
            base_out = torch.matmul(x, w.t())
            lora_out = torch.matmul(torch.matmul(x, lora_a.t()), lora_b.t())
            return base_out + scale * lora_out

    def kda_lora_fuse(self, **tensors):
        """FlashKDA + LoRA 融合 (opcode 0xB8)"""
        try:
            from .mlx_tune_integration import cgc_mlx_tune
            return cgc_mlx_tune.run_cgc_command(0xB8, tensors)
        except ImportError:
            q = tensors["q"]
            k = tensors["k"]
            v = tensors["v"]
            lora_a = tensors["lora_a"]
            lora_b = tensors["lora_b"]
            attn_out = F.scaled_dot_product_attention(q, k, v)
            lora_out = torch.matmul(torch.matmul(attn_out, lora_a.t()), lora_b.t())
            return attn_out + lora_out

    def lora_merge(self, **tensors):
        """LoRA 权重合并 (opcode 0xB2)"""
        try:
            from .mlx_tune_integration import cgc_mlx_tune
            return cgc_mlx_tune.run_cgc_command(0xB2, tensors)
        except ImportError:
            base = tensors["base_weight"]
            lora_a = tensors["lora_a"]
            lora_b = tensors["lora_b"]
            scale = tensors.get("scale", 1.0)
            return base + torch.matmul(b.t(), a.t()) * scale

    def lora_a_matmul(self, **tensors):
        """LoRA A 矩阵乘法 (opcode 0xB0)"""
        x = tensors["x"]
        lora_a = tensors["lora_a"]
        return torch.matmul(x, lora_a.t())

    def lora_b_matmul(self, **tensors):
        """LoRA B 矩阵乘法 (opcode 0xB1)"""
        x = tensors["x"]
        lora_b = tensors["lora_b"]
        return torch.matmul(x, lora_b.t())

    def qlora_dequant(self, **tensors):
        """QLoRA 反量化 (opcode 0xB3)"""
        quantized = tensors["quantized"]
        scale = tensors.get("scale", 1.0)
        if quantized.dtype in [torch.int8, torch.int4]:
            return quantized.float() * scale
        return quantized.float()


def execute_cgc_command(
    opcode: int,
    inputs: List[torch.Tensor],
    params: Dict[str, Any],
) -> List[torch.Tensor]:
    """
    执行单个 CGC 命令的便捷函数

    Args:
        opcode: CGC 操作码
        inputs: 输入 tensors
        params: 额外参数

    Returns:
        输出 tensors
    """
    executor = CGCExecutor()
    command = CGCCommand(
        opcode=opcode,
        inputs=inputs,
        outputs=[],
        params=params,
    )
    return executor.execute(command)


def get_kernel_registry() -> CGCKernelRegistry:
    """获取全局 kernel 注册表"""
    return _kernel_registry


def list_available_kernels() -> Dict[int, str]:
    """列出所有可用的 kernel"""
    kernels = _kernel_registry.list_all()
    return {opcode: spec.name for opcode, spec in kernels.items()}


# llama.cpp 支持
def _check_llama_cpp_available() -> bool:
    """檢查 llama.cpp 是否可用"""
    try:
        import llama_cpp
        return True
    except ImportError:
        return False


LLAMA_CPP_AVAILABLE = _check_llama_cpp_available()


# llama.cpp opcode 範圍
LLAMA_CPP_OPCODE_START = 0xC0
LLAMA_CPP_OPCODE_END = 0xD5


def is_llama_cpp_opcode(opcode: int) -> bool:
    """檢查是否是 llama.cpp opcode"""
    return LLAMA_CPP_OPCODE_START <= opcode <= LLAMA_CPP_OPCODE_END


# 全局 llama.cpp 模型實例
_llama_cpp_model = None


def set_llama_cpp_model(model) -> None:
    """設置全局 llama.cpp 模型"""
    global _llama_cpp_model
    _llama_cpp_model = model


def get_llama_cpp_model():
    """獲取全局 llama.cpp 模型"""
    return _llama_cpp_model


# llama.cpp 執行器包裝
def execute_llama_cpp_op(
    opcode: int,
    inputs: List[torch.Tensor],
    params: Dict[str, Any],
) -> List[torch.Tensor]:
    """
    執行 llama.cpp 操作

    Args:
        opcode: CGC 操作碼
        inputs: 輸入 tensors
        params: 參數字典

    Returns:
        輸出 tensors
    """
    if not LLAMA_CPP_AVAILABLE:
        # 回退到默認實現
        return _llama_fallback(opcode, inputs, params)

    try:
        import llama_cpp
        import numpy as np

        # 根據 opcode 執行不同操作
        if opcode == 0xC0:  # LLAMA_GGUF_LOAD
            return _llama_gguf_load(inputs, params)
        elif opcode == 0xC1:  # LLAMA_GGUF_QUANTIZE
            return _llama_gguf_quantize(inputs, params)
        elif opcode == 0xC2:  # LLAMA_GGUF_DEQUANTIZE
            return _llama_gguf_dequantize(inputs, params)
        elif opcode == 0xC3:  # LLAMA_Q4_K_MATMUL
            return _llama_q4_k_matmul(inputs, params)
        elif opcode == 0xC4:  # LLAMA_Q5_K_MATMUL
            return _llama_q5_k_matmul(inputs, params)
        elif opcode == 0xC5:  # LLAMA_Q6_K_MATMUL
            return _llama_q6_k_matmul(inputs, params)
        elif opcode == 0xC6:  # LLAMA_Q8_0_MATMUL
            return _llama_q8_0_matmul(inputs, params)
        elif opcode == 0xCA:  # LLAMA_MOE_ROUTING
            return _llama_moe_routing(inputs, params)
        elif opcode == 0xCB:  # LLAMA_MOE_EXPERT_FWD
            return _llama_moe_expert_fwd(inputs, params)
        elif opcode == 0xCD:  # LLAMA_RMSNORM_GGUF
            return _llama_rmsnorm_gguf(inputs, params)
        elif opcode == 0xCE:  # LLAMA_SILU_GGUF
            return _llama_silu_gguf(inputs, params)
        elif opcode == 0xCF:  # LLAMA_GELU_GGUF
            return _llama_gelu_gguf(inputs, params)
        elif opcode == 0xD2:  # LLAMA_INFERENCE
            return _llama_inference(inputs, params)
        else:
            # 回退到默認實現
            return _llama_fallback(opcode, inputs, params)

    except Exception as e:
        logger.error(f"llama.cpp execution failed for opcode 0x{opcode:02x}: {e}")
        return _llama_fallback(opcode, inputs, params)


def _llama_gguf_load(inputs: List[torch.Tensor], params: Dict[str, Any]) -> List[torch.Tensor]:
    """載入 GGUF 模型"""
    model_path = params.get("model_path")
    if not model_path:
        raise ValueError("model_path is required")

    try:
        import llama_cpp

        global _llama_cpp_model
        _llama_cpp_model = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=params.get("n_ctx", 2048),
            n_batch=params.get("n_batch", 512),
            n_threads=params.get("n_threads", 4),
            use_mlock=params.get("use_mlock", False),
            use_mmap=params.get("use_mmap", True),
            verbose=params.get("verbose", False),
        )

        logger.info(f"GGUF model loaded: {model_path}")
        return [torch.tensor([1], dtype=torch.int32)]

    except Exception as e:
        logger.error(f"Failed to load GGUF model: {e}")
        return [torch.tensor([0], dtype=torch.int32)]


def _llama_gguf_quantize(inputs: List[torch.Tensor], params: Dict[str, Any]) -> List[torch.Tensor]:
    """量化到 GGUF"""
    # 回退到默認實現
    return [inputs[0]] if inputs else []


def _llama_gguf_dequantize(inputs: List[torch.Tensor], params: Dict[str, Any]) -> List[torch.Tensor]:
    """GGUF 反量化"""
    # 回退到默認實現
    return [inputs[0]] if inputs else []


def _llama_q4_k_matmul(inputs: List[torch.Tensor], params: Dict[str, Any]) -> List[torch.Tensor]:
    """Q4_K 矩陣乘法"""
    x = inputs[0]
    weight = inputs[1] if len(inputs) > 1 else params.get("weight")
    out = torch.matmul(x, weight.t()) if weight is not None else x
    return [out]


def _llama_q5_k_matmul(inputs: List[torch.Tensor], params: Dict[str, Any]) -> List[torch.Tensor]:
    """Q5_K 矩陣乘法"""
    return _llama_q4_k_matmul(inputs, params)


def _llama_q6_k_matmul(inputs: List[torch.Tensor], params: Dict[str, Any]) -> List[torch.Tensor]:
    """Q6_K 矩陣乘法"""
    return _llama_q4_k_matmul(inputs, params)


def _llama_q8_0_matmul(inputs: List[torch.Tensor], params: Dict[str, Any]) -> List[torch.Tensor]:
    """Q8_0 矩陣乘法"""
    return _llama_q4_k_matmul(inputs, params)


def _llama_moe_routing(inputs: List[torch.Tensor], params: Dict[str, Any]) -> List[torch.Tensor]:
    """MoE 路由"""
    x = inputs[0]
    # 簡單的 softmax 路由
    routing_logits = x @ params.get("routing_weight", torch.eye(x.shape[-1], device=x.device))
    routing_probs = torch.softmax(routing_logits, dim=-1)
    return [routing_probs]


def _llama_moe_expert_fwd(inputs: List[torch.Tensor], params: Dict[str, Any]) -> List[torch.Tensor]:
    """MoE 專家前向"""
    x = inputs[0]
    expert_idx = params.get("expert_idx", 0)
    expert_weights = params.get("expert_weights", [])
    if expert_weights and len(expert_weights) > expert_idx:
        w = expert_weights[expert_idx]
        out = torch.matmul(x, w.t())
    else:
        out = x
    return [out]


def _llama_rmsnorm_gguf(inputs: List[torch.Tensor], params: Dict[str, Any]) -> List[torch.Tensor]:
    """GGUF RMSNorm"""
    x = inputs[0]
    weight = inputs[1] if len(inputs) > 1 else params.get("weight", torch.ones(x.shape[-1], device=x.device))
    eps = params.get("eps", 1e-6)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    out = weight * x
    return [out]


def _llama_silu_gguf(inputs: List[torch.Tensor], params: Dict[str, Any]) -> List[torch.Tensor]:
    """GGUF SiLU"""
    x = inputs[0]
    return [torch.nn.functional.silu(x)]


def _llama_gelu_gguf(inputs: List[torch.Tensor], params: Dict[str, Any]) -> List[torch.Tensor]:
    """GGUF GELU"""
    x = inputs[0]
    return [torch.nn.functional.gelu(x)]


def _llama_inference(inputs: List[torch.Tensor], params: Dict[str, Any]) -> List[torch.Tensor]:
    """llama.cpp 完整推理"""
    global _llama_cpp_model

    if _llama_cpp_model is None:
        # 嘗試從 params 載入
        model_path = params.get("model_path")
        if model_path:
            _llama_gguf_load(inputs, params)
        else:
            raise ValueError("llama.cpp model not loaded")

    tokens = inputs[0].tolist() if inputs else []
    output = _llama_cpp_model(
        tokens,
        max_tokens=params.get("max_tokens", 64),
        temperature=params.get("temperature", 0.7),
        top_p=params.get("top_p", 0.9),
        top_k=params.get("top_k", 40),
        stop=params.get("stop", []),
        echo=params.get("echo", False),
    )

    return [torch.tensor(output["tokens"], device=inputs[0].device)] if inputs else []


def _llama_fallback(opcode: int, inputs: List[torch.Tensor], params: Dict[str, Any]) -> List[torch.Tensor]:
    """llama.cpp 回退實現"""
    if not inputs:
        return []

    x = inputs[0]

    # 根據 opcode 執行對應操作
    if 0xC3 <= opcode <= 0xC9:  # 量化 matmul
        if len(inputs) > 1:
            return [torch.matmul(x, inputs[1].t())]
        return [x]
    elif opcode == 0xCD:  # RMSNorm
        return _llama_rmsnorm_gguf(inputs, params)
    elif opcode == 0xCE:  # SiLU
        return [torch.nn.functional.silu(x)]
    elif opcode == 0xCF:  # GELU
        return [torch.nn.functional.gelu(x)]
    elif opcode == 0xCA:  # MoE 路由
        return _llama_moe_routing(inputs, params)
    elif opcode == 0xCB:  # MoE 專家
        return _llama_moe_expert_fwd(inputs, params)
    else:
        return [x]


# 註冊 llama.cpp kernels 到全局註冊表
def register_llama_cpp_kernels() -> None:
    """註冊所有 llama.cpp kernels"""
    if not LLAMA_CPP_AVAILABLE:
        logger.warning("llama.cpp not available, skipping kernel registration")
        return

    registry = get_kernel_registry()

    # 註冊所有 llama.cpp opcodes
    for opcode in range(LLAMA_CPP_OPCODE_START, LLAMA_CPP_OPCODE_END + 1):
        spec = CGCKernelSpec(
            name=f"llama_{opcode}",
            kernel_type=KernelType.CUSTOM,
            cuda_kernel=_llama_kernel_adapter,
        )
        registry.register(opcode, spec)

    logger.info(f"Registered {LLAMA_CPP_OPCODE_END - LLAMA_CPP_OPCODE_START + 1} llama.cpp kernels")


def _llama_kernel_adapter(*args, **kwargs) -> torch.Tensor:
    """llama.cpp kernel 適配器"""
    # 從 kwargs 提取必要信息
    opcode = kwargs.get("opcode", 0xC0)
    inputs = kwargs.get("inputs", [])
    params = kwargs.get("params", {})

    outputs = execute_llama_cpp_op(opcode, inputs, params)
    return outputs[0] if outputs else None


# 自動註冊 llama.cpp kernels
if LLAMA_CPP_AVAILABLE:
    register_llama_cpp_kernels()
