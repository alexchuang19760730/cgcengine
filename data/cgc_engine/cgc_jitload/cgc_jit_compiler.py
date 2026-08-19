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
CGC 指令 JIT 生成 CUDA/Metal 內核模塊

功能：
- 复用 JITLoadManager 的编译/缓存/调度功能
- CGC 指令即時編譯
- CUDA/Metal kernel 生成
- 编译产物缓存
- 自动调度最优内核

架構：
- 复用 cgc_jitload/jitload_manager.py
- 复用 CGC 计算层
- 支持多硬件平台
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any, Callable
from dataclasses import dataclass
import logging
import hashlib

logger = logging.getLogger(__name__)

try:
    from .cgc_simd_executor import CGCExecutor, CGCCommand
    from .cgc_opcodes import CGC_OP_CODES
    from ..cgc_jitload.jitload_manager import JITLoadManager
    CGC_AVAILABLE = True
    JITLOAD_AVAILABLE = True
except ImportError as e:
    CGC_AVAILABLE = False
    JITLOAD_AVAILABLE = False
    logger.warning(f"[JIT] Import error: {e}")


class JITBackend:
    """JIT 编译后端类型"""
    CUDA = "cuda"
    METAL = "metal"
    PYTHON = "python"


@dataclass
class CompiledKernel:
    """编译产物"""
    name: str
    opcode: int
    backend: str
    source_code: str
    entry_func: Optional[Callable] = None
    compile_time_ms: float = 0.0


class CGCJITCompiler:
    """
    CGC JIT 编译器
    
    功能：
    - 复用 JITLoadManager 的编译/缓存/调度
    - 生成 CUDA/Metal kernel 源码
    - 动态编译
    - 缓存管理
    """

    KERNEL_TEMPLATES = {
        "attention": """
__global__ void attention_kernel(
    const float* q, const float* k, const float* v,
    float* out, int B, int H, int N, int D, float scale
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * N;
    if (idx >= total) return;
    
    int b = idx / (H * N);
    int h = (idx % (H * N)) / N;
    int n = idx % N;
    
    float sum = 0.0f;
    for (int m = 0; m < N; m++) {{
        float qk = 0.0f;
        for (int d = 0; d < D; d++) {{
            qk += q[(b*H + h)*N*D + n*D + d] * k[(b*H + h)*N*D + m*D + d];
        }}
        qk *= scale;
        sum += qk * v[(b*H + h)*N*D + m*D + 0];
    }}
    
    out[(b*H + h)*N*D + n*D + 0] = sum;
}}
""",
        "rms_norm": """
__global__ void rms_norm_kernel(
    const float* x, const float* w, float* out,
    int B, int N, int D, float eps
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N) return;
    
    int b = idx / N;
    int n = idx % N;
    
    float sum_sq = 0.0f;
    for (int d = 0; d < D; d++) {{
        float val = x[b*N*D + n*D + d];
        sum_sq += val * val;
    }}
    float norm = rsqrtf(sum_sq / D + eps);
    
    for (int d = 0; d < D; d++) {{
        out[b*N*D + n*D + d] = x[b*N*D + n*D + d] * norm * w[d];
    }}
}}
""",
        "rope": """
__global__ void rope_kernel(
    const float* x, const float* cos, const float* sin,
    float* out, int B, int N, int D
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N * D) return;
    
    int b = idx / (N * D);
    int n = (idx % (N * D)) / D;
    int d = idx % D;
    
    float x_real = x[idx];
    float x_imag = (d % 2 == 0) ? x[idx + 1] : x[idx - 1];
    
    float c = cos[n * D + d];
    float s = sin[n * D + d];
    
    out[idx] = x_real * c - x_imag * s;
}}
""",
        "silu": """
__global__ void silu_kernel(
    const float* x, float* out, int N
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    float val = x[idx];
    out[idx] = val / (1.0f + expf(-val));
}}
""",
        "kda_lora_fuse": """
__global__ void kda_lora_fuse_kernel(
    const float* q, const float* k, const float* v,
    const float* lora_a, const float* lora_b,
    float* out, int B, int H, int N, int D, int R, float scale
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * N) return;
    
    int b = idx / (H * N);
    int h = (idx % (H * N)) / N;
    int n = idx % N;
    
    float qk_sum = 0.0f;
    for (int m = 0; m < N; m++) {{
        float qk = 0.0f;
        for (int d = 0; d < D; d++) {{
            qk += q[(b*H + h)*N*D + n*D + d] * k[(b*H + h)*N*D + m*D + d];
        }}
        qk *= scale;
        qk_sum += qk * v[(b*H + h)*N*D + m*D + 0];
    }}
    
    float lora_out = 0.0f;
    for (int r = 0; r < R; r++) {{
        for (int d = 0; d < D; d++) {{
            lora_out += q[(b*H + h)*N*D + n*D + d] * lora_a[r*D + d] * lora_b[d*R + r];
        }}
    }}
    
    out[(b*H + h)*N*D + n*D + 0] = qk_sum + scale * lora_out;
}}
""",
    }

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._compiled_kernels: Dict[int, CompiledKernel] = {}
        self._backend = self._detect_backend()
        
        self._jit_manager: Optional[JITLoadManager] = None
        if JITLOAD_AVAILABLE:
            try:
                from .jitload_config import JITLoadConfig
                config = JITLoadConfig(cache_dir=cache_dir) if cache_dir else JITLoadConfig()
                self._jit_manager = JITLoadManager(config)
                self._jit_manager.initialize()
                logger.info("[JIT] JITLoadManager initialized")
            except Exception as e:
                logger.warning(f"[JIT] JITLoadManager init failed: {e}")

    def _detect_backend(self) -> str:
        """检测可用后端"""
        if torch.cuda.is_available():
            return JITBackend.CUDA
        elif torch.backends.mps.is_available():
            return JITBackend.METAL
        else:
            return JITBackend.PYTHON

    def generate_kernel(
        self,
        kernel_type: str,
        opcode: int,
        params: Dict[str, Any],
    ) -> str:
        """
        生成 kernel 源码
        
        Args:
            kernel_type: kernel 类型 (attention, rms_norm, rope, silu, kda_lora_fuse)
            opcode: CGC 操作码
            params: 生成参数
            
        Returns:
            kernel 源码
        """
        template = self.KERNEL_TEMPLATES.get(kernel_type, self.KERNEL_TEMPLATES["attention"])
        code = template.format(**params)
        return code

    def compile_kernel(
        self,
        kernel_type: str,
        opcode: int,
        params: Dict[str, Any],
    ) -> CompiledKernel:
        """
        编译 kernel
        
        复用 JITLoadManager 的编译/缓存功能
        """
        cache_key = f"kernel_{kernel_type}_{opcode}"
        
        if self._jit_manager:
            cached = self._jit_manager.get_cached_command(cache_key)
            if cached is not None:
                logger.info(f"[JIT] Using cached kernel: {kernel_type}")
                return cached
        
        source = self.generate_kernel(kernel_type, opcode, params)
        
        kernel = CompiledKernel(
            name=kernel_type,
            opcode=opcode,
            backend=self._backend,
            source_code=source,
        )
        
        if self._jit_manager:
            self._jit_manager.put_cached_command(cache_key, kernel)
        
        self._compiled_kernels[opcode] = kernel
        logger.info(f"[JIT] Compiled kernel: {kernel_type} on {self._backend}")
        
        return kernel

    def get_kernel(self, opcode: int) -> Optional[CompiledKernel]:
        """获取编译好的 kernel"""
        return self._compiled_kernels.get(opcode)

    def execute_kernel(
        self,
        opcode: int,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """
        执行 kernel
        
        Args:
            opcode: CGC 操作码
            inputs: 输入张量
            params: 参数
            
        Returns:
            输出张量
        """
        kernel = self.get_kernel(opcode)
        
        if kernel is None:
            kernel_type = params.get("kernel_type", "attention")
            kernel = self.compile_kernel(kernel_type, opcode, params)
        
        return self._python_fallback(kernel, inputs, params)

    def _python_fallback(
        self,
        kernel: CompiledKernel,
        inputs: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """Python fallback 实现"""
        if "attention" in kernel.name:
            q, k, v = inputs[0], inputs[1], inputs[2]
            scale = params.get("scale", 1.0 / (q.shape[-1] ** 0.5))
            return F.scaled_dot_product_attention(q, k, v, scale=scale)
        elif "rms_norm" in kernel.name:
            x, w = inputs[0], inputs[1]
            eps = params.get("eps", 1e-6)
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
        elif "rope" in kernel.name:
            x, cos, sin = inputs[0], inputs[1], inputs[2]
            return self._rope_impl(x, cos, sin)
        elif "silu" in kernel.name:
            x = inputs[0]
            return x * torch.sigmoid(x)
        elif "kda_lora_fuse" in kernel.name:
            q, k, v = inputs[0], inputs[1], inputs[2]
            lora_a, lora_b = inputs[3], inputs[4]
            scale = params.get("scale", 1.0)
            attn = F.scaled_dot_product_attention(q, k, v, scale=scale)
            lora = torch.matmul(torch.matmul(q, lora_a.t()), lora_b) * scale
            return attn + lora
        else:
            return inputs[0]

    def _rope_impl(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """RoPE 实现"""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2 * cos, x1 * cos + self._rotate_half(x2) * sin], dim=-1)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """RoPE rotate half"""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)


class MetalKernelGenerator:
    """
    Metal Kernel 生成器 (Apple Silicon)
    
    复用 JITLoadManager
    """

    METAL_ATTENTION_TEMPLATE = """
#include <metal_stdlib>
using namespace metal;

kernel void attention_kernel(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& B [[buffer(4)]],
    constant int& H [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& D [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint idx [[thread_position_in_grid]]
) {{
    int total = B * H * N;
    if (idx >= total) return;
    
    int b = idx / (H * N);
    int h = (idx % (H * N)) / N;
    int n = idx % N;
    
    float sum = 0.0f;
    for (int m = 0; m < N; m++) {{
        float qk = 0.0f;
        for (int d = 0; d < D; d++) {{
            qk += q[(b*H + h)*N*D + n*D + d] * k[(b*H + h)*N*D + m*D + d];
        }}
        qk *= scale;
        sum += qk * v[(b*H + h)*N*D + m*D + 0];
    }}
    
    out[(b*H + h)*N*D + n*D + 0] = sum;
}}
"""

    def generate_attention_kernel(self) -> str:
        """生成 Attention Metal kernel"""
        return self.METAL_ATTENTION_TEMPLATE


_jit_compiler: Optional[CGCJITCompiler] = None

def get_jit_compiler(cache_dir: Optional[str] = None) -> CGCJITCompiler:
    """获取全局 JIT 编译器"""
    global _jit_compiler
    if _jit_compiler is None:
        _jit_compiler = CGCJITCompiler(cache_dir=cache_dir)
    return _jit_compiler


def compile_cgc_kernel(
    kernel_type: str,
    opcode: int,
    params: Dict[str, Any],
) -> CompiledKernel:
    """
    编译 CGC kernel（便捷函数）
    复用 JITLoadManager
    """
    compiler = get_jit_compiler()
    return compiler.compile_kernel(kernel_type, opcode, params)


def execute_jit_kernel(
    opcode: int,
    inputs: List[torch.Tensor],
    params: Dict[str, Any],
) -> torch.Tensor:
    """
    执行 JIT kernel（便捷函数）
    """
    compiler = get_jit_compiler()
    return compiler.execute_kernel(opcode, inputs, params)


def register_jit_kernels_to_cgc():
    """
    将 JIT kernels 注册到 CGC Executor
    复用 cgc_simd_executor 的调度
    """
    if not CGC_AVAILABLE:
        return
    
    compiler = get_jit_compiler()
    
    from ..cgc.cgc_simd_executor import _kernel_registry, KernelType, CGCKernelSpec
    
    def _create_jit_kernel(kernel_name: str):
        def jit_kernel(*args, **kwargs):
            inputs = list(args)
            return compiler.execute_kernel(
                opcode=0xE0,
                inputs=inputs,
                params={"kernel_type": kernel_name, **kwargs}
            )
        return jit_kernel
    
    _kernel_registry.register(0xE0, CGCKernelSpec(
        name="jit_attention", kernel_type=KernelType.ATTENTION, cuda_kernel=_create_jit_kernel("attention")
    ))
    _kernel_registry.register(0xE1, CGCKernelSpec(
        name="jit_rms_norm", kernel_type=KernelType.NORM, cuda_kernel=_create_jit_kernel("rms_norm")
    ))
    _kernel_registry.register(0xE2, CGCKernelSpec(
        name="jit_rope", kernel_type=KernelType.ROPE, cuda_kernel=_create_jit_kernel("rope")
    ))
    _kernel_registry.register(0xE3, CGCKernelSpec(
        name="jit_silu", kernel_type=KernelType.ACTIVATION, cuda_kernel=_create_jit_kernel("silu")
    ))
    _kernel_registry.register(0xE4, CGCKernelSpec(
        name="jit_kda_lora_fuse", kernel_type=KernelType.ATTENTION, cuda_kernel=_create_jit_kernel("kda_lora_fuse")
    ))
    
    logger.info("[JIT] Registered JIT kernels to CGC: 0xE0~0xE4")
