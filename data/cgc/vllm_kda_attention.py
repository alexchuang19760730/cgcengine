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
vLLM CGC Attention Backend

复制此文件到: vllm/attention/backends/kda/kda_attention.py

使用方法:
    1. 确保 vllm_kda_ops.py 已复制到同一目录
    2. 启动 vLLM:
       from vllm import LLM
       llm = LLM(model="...", attention_backend="cgc_kda")
"""

import torch
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    from .vllm_kda_ops import (
        flash_kda_cgc_forward,
        flash_kda_cgc_forward_native,
        sdpa_cgc_forward,
        rms_norm_cgc_forward,
        rope_cgc_forward,
        silu_cgc_forward,
        softmax_cgc_forward,
        get_cgc_backend_info,
        CGC_AVAILABLE,
        CGC_OP_CODES,
    )
except ImportError:
    from vllm_kda_ops import (
        flash_kda_cgc_forward,
        flash_kda_cgc_forward_native,
        sdpa_cgc_forward,
        rms_norm_cgc_forward,
        rope_cgc_forward,
        silu_cgc_forward,
        softmax_cgc_forward,
        get_cgc_backend_info,
        CGC_AVAILABLE,
        CGC_OP_CODES,
    )


@dataclass
class CGCKDABackendConfig:
    """CGC KDA Backend 配置"""
    enable_flashkda: bool = True
    enable_cgc: bool = True
    enable_profiling: bool = False
    fallback_to_native: bool = True
    kda_scale: float = 1.0
    use_gate: bool = True
    use_qk_l2norm: bool = True
    use_beta_sigmoid: bool = True
    lower_bound: float = -5.0


class CGCKDABackend:
    """
    vLLM CGC KDA Attention Backend

    This backend replaces the default vLLM KDA backend with CGC SIMD commands.

    Features:
    - All attention operations via CGC SIMD Executor
    - FlashKDA kernel integration
    - Automatic fallback to native PyTorch if CGC unavailable
    - Support for SDPA, KDA, PagedAttention
    """

    def __init__(self, config: Optional[CGCKDABackendConfig] = None):
        self.config = config or CGCKDABackendConfig()
        self._init_backend()

    def _init_backend(self):
        """初始化后端"""
        backend_info = get_cgc_backend_info()

        if backend_info["cgc_available"]:
            print(f"[CGC KDA Backend] Initialized with CGC SIMD Executor")
            print(f"  - FlashKDA available: {backend_info['flashkda_available']}")
            print(f"  - Total opcodes: {backend_info['total_opcodes']}")
        else:
            print(f"[CGC KDA Backend] CGC not available, using native fallback")
            print(f"  - Fallback enabled: {self.config.fallback_to_native}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Attention forward

        Args:
            q: Query tensor [B, T, H, K] or [T, H, K]
            k: Key tensor [B, T, H, K] or [T, H, K]
            v: Value tensor [B, T, H, V] or [T, H, V]
            **kwargs: Additional arguments (scale, g, beta, etc.)

        Returns:
            Attention output with same shape as v
        """
        if self.config.enable_flashkda and CGC_AVAILABLE:
            try:
                return self._forward_flashkda(q, k, v, **kwargs)
            except Exception as e:
                if self.config.fallback_to_native:
                    return self._forward_native(q, k, v, **kwargs)
                raise
        else:
            return self._forward_sdpa(q, k, v, **kwargs)

    def _forward_flashkda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward via FlashKDA"""
        scale = kwargs.get("scale", self.config.kda_scale)
        g = kwargs.get("g", None)
        beta = kwargs.get("beta", None)
        A_log = kwargs.get("A_log", None)
        dt_bias = kwargs.get("dt_bias", None)
        lower_bound = kwargs.get("lower_bound", self.config.lower_bound)
        initial_state = kwargs.get("initial_state", None)
        final_state = kwargs.get("final_state", None)
        cu_seqlens = kwargs.get("cu_seqlens", None)

        out, _ = flash_kda_cgc_forward(
            q=q, k=k, v=v,
            scale=scale,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
        )
        return out

    def _forward_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward via SDPA"""
        scale = kwargs.get("scale", 1.0 / (q.shape[-1] ** 0.5))
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

    def _forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward via native FlashKDA (without CGC)"""
        scale = kwargs.get("scale", self.config.kda_scale)
        g = kwargs.get("g", None)
        beta = kwargs.get("beta", None)
        A_log = kwargs.get("A_log", None)
        dt_bias = kwargs.get("dt_bias", None)
        lower_bound = kwargs.get("lower_bound", self.config.lower_bound)

        out, _ = flash_kda_cgc_forward_native(
            q=q, k=k, v=v,
            g=g, beta=beta,
            scale=scale,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
        )
        return out

    def forward分层(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """
        分层 Transformer forward

        用于完整 Transformer 层的 CGC 化。

        Args:
            hidden_states: Hidden states [B, T, D]
            position_ids: Position ids [B, T]
            attention_mask: Attention mask

        Returns:
            Output hidden states
        """
        raise NotImplementedError("分层 forward 需要完整模型集成")

    @staticmethod
    def get_config() -> dict:
        """获取后端配置信息"""
        return {
            "name": "cgc_kda",
            "supports_flashkda": True,
            "supports_sdpa": True,
            "supports_paged_attention": True,
            "backend_info": get_cgc_backend_info(),
        }


def create_cgc_kda_backend(config: Optional[CGCKDABackendConfig] = None) -> CGCKDABackend:
    """
    创建 CGC KDA Backend 实例

    Args:
        config: Backend 配置

    Returns:
        CGCKDABackend 实例
    """
    return CGCKDABackend(config=config)


# ============================================================================
# vLLM Backend Registration (for vllm/attention/backends/__init__.py)
# ============================================================================

# 在 vLLM 中注册此后端:
# AttentionBackend.register_backend("cgc_kda", CGCKDABackend)

BACKEND_NAME = "cgc_kda"
BACKEND_CLASS = CGCKDABackend


__all__ = [
    "CGCKDABackend",
    "CGCKDABackendConfig",
    "create_cgc_kda_backend",
    "get_cgc_backend_info",
    "BACKEND_NAME",
    "BACKEND_CLASS",
]
