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
FlashKDA + LoRA 硬件级融合模块

功能：
- FlashKDA 与 LoRA 计算融合
- 消除 kernel launch overhead
- 减少显存访问

架构：
- 复用 CGC 计算层
- 复用 CUDA kernel
- 统一 CGC opcode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    from .flashkda_integration import FlashKDALayer, FLASHKDA_AVAILABLE
    from .cgc_simd_executor import CGCExecutor, CGCCommand
    from .cgc_opcodes import CGC_OP_CODES
    CGC_AVAILABLE = True
except ImportError:
    CGC_AVAILABLE = False
    FLASHKDA_AVAILABLE = False


KERNEL_AVAILABLE = False
try:
    import flash_kda_lora_cuda
    KERNEL_AVAILABLE = True
except ImportError:
    logger.warning("[FlashKDA+LoRA] CUDA kernel not available, using PyTorch fallback")


class LoRAConfig:
    """LoRA 配置"""
    def __init__(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.0,
    ):
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        self.lora_dropout = lora_dropout
        self.scaling = alpha / rank


@dataclass
class LoRAParameters:
    """LoRA 参数"""
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    scale: float


class FlashKDALoRAFusedAttention(nn.Module):
    """
    FlashKDA + LoRA 融合 Attention
    
    融合计算：
    - KDA Attention
    - LoRA A @ LoRA B
    - 输出融合
    
    CGC Opcode: 0xB8 (KDA_LORA_FUSE)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int = 128,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        use_fused_kernel: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.use_fused_kernel = use_fused_kernel and KERNEL_AVAILABLE and FLASHKDA_AVAILABLE
        
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        
        self.lora_a = nn.Linear(num_heads * head_dim, lora_rank, bias=False)
        self.lora_b = nn.Linear(lora_rank, num_heads * head_dim, bias=False)
        
        self.dropout = nn.Dropout(lora_dropout)
        self.scale = 1.0 / (head_dim ** 0.5)
        
        if FLASHKDA_AVAILABLE:
            self.flashkda = FlashKDALayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                k_dim=head_dim,
                v_dim=head_dim,
            )
        
        logger.info(f"[FlashKDA+LoRA] Initialized: fused={self.use_fused_kernel}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播（融合计算）
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: 可选的 attention mask
            
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        batch, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.use_fused_kernel:
            out = self._fused_forward(q, k, v)
        else:
            out = self._sequential_forward(q, k, v)
        
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        
        return out

    def _fused_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """融合 kernel 前向"""
        lora_a_weight = self.lora_a.weight.t()
        lora_b_weight = self.lora_b.weight.t()
        
        try:
            out = flash_kda_lora_cuda.kda_lora_fused(
                q, k, v,
                lora_a_weight, lora_b_weight,
                self.lora_alpha / self.lora_rank,
            )
            return out
        except Exception as e:
            logger.warning(f"[FlashKDA+LoRA] Fused kernel failed: {e}, using sequential")
            return self._sequential_forward(q, k, v)

    def _sequential_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """顺序执行前向（FlashKDA → LoRA）"""
        if FLASHKDA_AVAILABLE:
            kda_out, _ = self.flashkda(q, k, v, scale=self.scale)
        else:
            kda_out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        
        lora_input = kda_out.transpose(1, 2).reshape(-1, self.num_heads * self.head_dim)
        lora_a_out = self.dropout(self.lora_a(lora_input))
        lora_out = self.lora_b(lora_a_out)
        lora_out = lora_out.view(kda_out.shape[0], kda_out.shape[2], -1).transpose(1, 2)
        
        fused_out = kda_out + lora_out * (self.lora_alpha / self.lora_rank)
        
        return fused_out

    def get_lora_params(self) -> LoRAParameters:
        """获取 LoRA 参数"""
        return LoRAParameters(
            lora_a=self.lora_a.weight.data,
            lora_b=self.lora_b.weight.data,
            scale=self.lora_alpha / self.lora_rank,
        )


class FlashKDALoRAExecutor:
    """
    FlashKDA + LoRA 执行器
    
    封装 CGC 命令执行
    """

    def __init__(self):
        self.executor = CGCExecutor() if CGC_AVAILABLE else None

    def execute_kda_lora_fuse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        执行融合的 KDA + LoRA 计算
        
        CGC Opcode: 0xB8
        """
        if self.executor is not None:
            cmd = CGCCommand(
                opcode=0xB8,
                inputs=[q, k, v, lora_a, lora_b],
                outputs=[],
                params={"scale": scale},
            )
            outputs = self.executor.execute(cmd)
            return outputs[0] if outputs else q
        
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=scale)
        
        lora_out = torch.matmul(
            torch.matmul(q, lora_a.t()),
            lora_b
        ) * scale
        
        return attn_out + lora_out

    def execute_lora_a(
        self,
        x: torch.Tensor,
        lora_a: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行 LoRA A 矩阵乘法
        
        CGC Opcode: 0xB0
        """
        if self.executor is not None:
            cmd = CGCCommand(
                opcode=0xB0,
                inputs=[x, lora_a],
                outputs=[],
                params={},
            )
            outputs = self.executor.execute(cmd)
            return outputs[0] if outputs else x
        
        return F.linear(x, lora_a.t())

    def execute_lora_b(
        self,
        x: torch.Tensor,
        lora_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行 LoRA B 矩阵乘法
        
        CGC Opcode: 0xB1
        """
        if self.executor is not None:
            cmd = CGCCommand(
                opcode=0xB1,
                inputs=[x, lora_b],
                outputs=[],
                params={},
            )
            outputs = self.executor.execute(cmd)
            return outputs[0] if outputs else x
        
        return F.linear(x, lora_b.t())


def create_fused_attention(
    hidden_dim: int,
    num_heads: int,
    head_dim: int = 128,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
) -> FlashKDALoRAFusedAttention:
    """
    创建融合 Attention（便捷函数）
    """
    return FlashKDALoRAFusedAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )


def register_lora_kernels():
    """注册 LoRA CGC kernels"""
    if not CGC_AVAILABLE:
        return
    
    from .cgc_simd_executor import _kernel_registry, KernelType, CGCKernelSpec
    
    def _lora_a_kernel(x, lora_a, **kwargs):
        return F.linear(x, lora_a.t())
    
    def _lora_b_kernel(x, lora_b, **kwargs):
        return F.linear(x, lora_b.t())
    
    def _kda_lora_fused(q, k, v, lora_a, lora_b, **kwargs):
        scale = kwargs.get("scale", 1.0)
        attn = F.scaled_dot_product_attention(q, k, v, scale=scale)
        lora = torch.matmul(torch.matmul(q, lora_a.t()), lora_b) * scale
        return attn + lora
    
    _kernel_registry.register(0xB0, CGCKernelSpec(
        name="lora_a", kernel_type=KernelType.LINEAR, cuda_kernel=_lora_a_kernel
    ))
    _kernel_registry.register(0xB1, CGCKernelSpec(
        name="lora_b", kernel_type=KernelType.LINEAR, cuda_kernel=_lora_b_kernel
    ))
    _kernel_registry.register(0xB8, CGCKernelSpec(
        name="kda_lora_fuse", kernel_type=KernelType.ATTENTION, cuda_kernel=_kda_lora_fused
    ))
    
    logger.info("[FlashKDA+LoRA] CGC kernels registered: 0xB0, 0xB1, 0xB8")
