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
vLLM + FlashKDA + MagiCompiler Integration

This module provides the integration between:
- vLLM (调度 + PagedAttention)
- MagiCompiler (整图编译 + 显存优化)
- FlashKDA (Moonshot 原始 CUDA kernel)

Architecture:
    vLLM（调度 + PagedAttention）
       ↓
    MagiCompiler（整图编译 + 显存优化）
       ↓
    FlashKDA（Moonshot 原始 CUDA kernel）

Usage:
    from cgc_engine.cgc.vllm_integration import create_vllm_kda_backend

    # Option 1: Use with MagiCompiler
    backend = create_vllm_kda_backend(enable_magicompiler=True)

    # Option 2: Direct FlashKDA backend
    backend = create_vllm_kda_backend(enable_flashkda=True)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

VLLM_AVAILABLE = False
try:
    from vllm.attention.backends.kda import KDAAttentionBackend
    from vllm.attention import Attention
    VLLM_AVAILABLE = True
except ImportError:
    logger.warning("vLLM not available. Install with: pip install vllm")
    KDAAttentionBackend = None
    Attention = None


@dataclass
class VLLMKDAConfig:
    """Configuration for vLLM KDA backend with MagiCompiler/FlashKDA."""
    enable_flashkda: bool = True
    enable_magicompiler: bool = True
    enable_ortho_basis_update: bool = True
    kda_scale: float = 1.0
    use_gate: bool = True
    use_qk_l2norm: bool = True
    use_beta_sigmoid: bool = True
    chunk_size: int = 64
    k_dim: int = 128
    v_dim: int = 128
    lower_bound: float = -5.0
    num_heads: int = 0
    head_dim: int = 128

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_flashkda": self.enable_flashkda,
            "enable_magicompiler": self.enable_magicompiler,
            "enable_ortho_basis_update": self.enable_ortho_basis_update,
            "kda_scale": self.kda_scale,
            "use_gate": self.use_gate,
            "use_qk_l2norm": self.use_qk_l2norm,
            "use_beta_sigmoid": self.use_beta_sigmoid,
            "chunk_size": self.chunk_size,
            "k_dim": self.k_dim,
            "v_dim": self.v_dim,
            "lower_bound": self.lower_bound,
        }


class VLLMKDABackend:
    """
    vLLM KDA Attention Backend with FlashKDA/MagiCompiler integration.

    This backend replaces the default vLLM attention backend with
    FlashKDA CUDA kernels for improved performance on long sequences.

    Key Benefits:
    - 比原生 vLLM + FA2 更快、更省显存
    - 长序列（32k+）优势巨大
    - 可无缝切换到 Megatrain 训练（同一套 CGC + KDA）
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        kda_config: Optional[VLLMKDAConfig] = None,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kda_config = kda_config or VLLMKDAConfig(
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.kda_config.num_heads = num_heads
        self.kda_config.head_dim = head_dim

        self._init_flashkda_layer()
        self._init_magicompiler_pass()

    def _init_flashkda_layer(self):
        """Initialize FlashKDA layer."""
        if not self.kda_config.enable_flashkda:
            logger.info("FlashKDA disabled, using standard attention")
            return

        try:
            from ..cgc.flashkda_integration import FlashKDALayer, _check_flashkda_available

            if not _check_flashkda_available():
                logger.warning("FlashKDA not available, falling back to standard attention")
                self.flashkda_layer = None
                return

            self.flashkda_layer = FlashKDALayer(
                hidden_dim=self.num_heads * self.head_dim,
                num_heads=self.num_heads,
                k_dim=self.kda_config.k_dim,
                v_dim=self.kda_config.v_dim,
                use_gate=self.kda_config.use_gate,
                use_qk_l2norm=self.kda_config.use_qk_l2norm,
                use_beta_sigmoid=self.kda_config.use_beta_sigmoid,
                scale=self.kda_config.kda_scale,
                lower_bound=self.kda_config.lower_bound,
            )
            logger.info(f"FlashKDA layer initialized: {self.num_heads} heads, dim={self.head_dim}")

        except ImportError as e:
            logger.warning(f"Failed to import FlashKDALayer: {e}")
            self.flashkda_layer = None

    def _init_magicompiler_pass(self):
        """Initialize MagiCompiler KDA pass."""
        if not self.kda_config.enable_magicompiler:
            return

        try:
            from ..cgc.kda_pass import InsertKDAPass

            self.kda_pass = InsertKDAPass(
                enable_ortho_basis_update=self.kda_config.enable_ortho_basis_update,
                enable_flashkda_fusion=self.kda_config.enable_flashkda,
                kda_scale=self.kda_config.kda_scale,
                use_gate=self.kda_config.use_gate,
                use_qk_l2norm=self.kda_config.use_qk_l2norm,
                use_beta_sigmoid=self.kda_config.use_beta_sigmoid,
            )
            logger.info("MagiCompiler KDA pass initialized")

        except ImportError as e:
            logger.warning(f"Failed to import MagiCompiler KDA pass: {e}")
            self.kda_pass = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass using FlashKDA.

        Args:
            q: Query tensor [B, T, H, K] or [T, H, K]
            k: Key tensor [B, T, H, K] or [T, H, K]
            v: Value tensor [B, T, H, V] or [T, H, V]
            cu_seqlens: Cumulative sequence lengths for varlen mode
            max_seqlen: Maximum sequence length

        Returns:
            Attention output with same shape as v
        """
        if self.flashkda_layer is None:
            return self._fallback_attention(q, k, v)

        if cu_seqlens is not None:
            return self._forward_varlen(q, k, v, cu_seqlens)
        else:
            return self._forward_batched(q, k, v)

    def _forward_batched(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Batched forward pass."""
        if q.dim() == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        B, T, H, K = q.shape
        if hasattr(self, 'flashkda_layer') and self.flashkda_layer is not None:
            out, _ = self.flashkda_layer(
                x=q.view(B, T, H * K),
                initial_state=None,
            )
            return out
        else:
            return self._fallback_attention(q, k, v)

    def _forward_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Variable-length forward pass."""
        if q.dim() == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        if hasattr(self, 'flashkda_layer') and self.flashkda_layer is not None:
            out, _ = self.flashkda_layer(
                x=q.view(1, q.shape[1], q.shape[2] * q.shape[3]),
                initial_state=None,
                cu_seqlens=cu_seqlens,
            )
            return out
        else:
            return self._fallback_attention(q, k, v)

    def _fallback_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback to standard scaled dot-product attention."""
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)


def create_vllm_kda_backend(
    num_heads: int,
    head_dim: int,
    enable_flashkda: bool = True,
    enable_magicompiler: bool = True,
    **kwargs
) -> VLLMKDABackend:
    """
    Factory function to create a vLLM-compatible KDA backend.

    Args:
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        enable_flashkda: Enable FlashKDA CUDA kernel
        enable_magicompiler: Enable MagiCompiler optimization
        **kwargs: Additional config passed to VLLMKDAConfig

    Returns:
        VLLMKDABackend instance

    Example:
        backend = create_vllm_kda_backend(
            num_heads=16,
            head_dim=128,
            enable_flashkda=True,
            enable_magicompiler=True,
        )
    """
    config = VLLMKDAConfig(
        num_heads=num_heads,
        head_dim=head_dim,
        enable_flashkda=enable_flashkda,
        enable_magicompiler=enable_magicompiler,
        **kwargs
    )
    return VLLMKDABackend(num_heads=num_heads, head_dim=head_dim, kda_config=config)


class MagiCompilerVLLMWrapper(nn.Module):
    """
    MagiCompiler + vLLM wrapper for easy integration.

    This wrapper provides a simple interface for using MagiCompiler
    optimized KDA attention with vLLM models.

    Usage:
        from cgc_engine import magi_compile
        from cgc_engine.cgc.vllm_integration import MagiCompilerVLLMWrapper

        wrapper = MagiCompilerVLLMWrapper(
            hidden_dim=4096,
            num_heads=32,
            enable_flashkda=True,
        )

        @magi_compile
        class OptimizedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.wrapper = wrapper
                self.attn = wrapper.create_attention()

            def forward(self, x):
                return self.wrapper(self.attn, x)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        k_dim: int = 128,
        v_dim: int = 128,
        enable_flashkda: bool = True,
        enable_magicompiler: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.kda_config = VLLMKDAConfig(
            num_heads=num_heads,
            head_dim=hidden_dim // num_heads,
            enable_flashkda=enable_flashkda,
            enable_magicompiler=enable_magicompiler,
        )

        self.backend = VLLMKDABackend(
            num_heads=num_heads,
            head_dim=hidden_dim // num_heads,
            kda_config=self.kda_config,
        )

    def create_attention(self) -> VLLMKDABackend:
        """Create attention backend instance."""
        return self.backend

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through wrapped attention.

        Args:
            x: Input tensor [B, T, D]
            mask: Attention mask (optional)

        Returns:
            Output tensor [B, T, D]
        """
        return self.backend.forward(x, x, x)


def get_vllm_kda_attention_layer(
    num_heads: int,
    head_dim: int,
    kda_config: Optional[VLLMKDAConfig] = None,
) -> VLLMKDABackend:
    """
    Get a vLLM-compatible KDA attention layer.

    This function is the main entry point for integrating
    MagiCompiler + FlashKDA with vLLM.

    Args:
        num_heads: Number of attention heads
        head_dim: Dimension per head
        kda_config: Optional KDA configuration

    Returns:
        Attention layer ready for vLLM use

    Example in vLLM model:
        class KDAOptimizedModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.attn = get_vllm_kda_attention_layer(
                    num_heads=config.num_heads,
                    head_dim=config.head_dim,
                )

            def forward(self, x):
                return self.attn(x, x, x)
    """
    config = kda_config or VLLMKDAConfig(
        num_heads=num_heads,
        head_dim=head_dim,
    )
    return VLLMKDABackend(num_heads=num_heads, head_dim=head_dim, kda_config=config)


# Integration guide for vLLM
INTEGRATION_GUIDE = """
vLLM + FlashKDA + MagiCompiler Integration Guide
=================================================

1. Installation
---------------
# Install vLLM
pip install vllm

# Install FlashKDA
cd MoonshotAI-FlashKDA-784a210
pip install -v .

# MagiCompiler is already installed (this project)


2. Model Integration
--------------------
# In your vLLM model file (e.g., your_model.py):

from cgc_engine.cgc.vllm_integration import get_vllm_kda_attention_layer

class YourOptimizedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = get_vllm_kda_attention_layer(
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
        )

    def forward(self, x, y, z):
        return self.attn(x, y, z)


3. Launch vLLM with KDA Backend
--------------------------------
from vllm import LLM

llm = LLM(
    model="your-model-path",
    attention_backend="kda",       # Enable KDA backend
    enable_flash_attn=False,        # Disable FlashAttention
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,
)


4. Architecture Overview
------------------------
    vLLM（调度 + PagedAttention）
       ↓
    MagiCompiler（整图编译 + 显存优化）
       ↓
    FlashKDA（Moonshot 原始 CUDA kernel）

Benefits:
- 比原生 vLLM + FA2 更快、更省显存
- 长序列（32k+）优势巨大
- 可无缝切换到 Megatrain 训练
"""


if __name__ == "__main__":
    print(INTEGRATION_GUIDE)
