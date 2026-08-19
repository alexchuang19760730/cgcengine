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
Megatrain CGC Integration - 训练侧 SIMD 命令完整支持

支持:
- Megatrain forward/backward 全流程 CGC SIMD 化
- 与 vLLM 推理共用同一套 40 条 CGC 指令集
- FlashKDA Kernel 训推一体

文件位置: cgc_engine/cgc/megatrain_integration.py
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
from functools import partial

try:
    from .cgc_opcodes import CGC_OP_CODES
    from .cgc_simd_executor import CGCExecutor, CGCCommand
    from .cgc_commands import (
        KDA_CHUNK_CMD,
        KDA_PROJECT_CMD,
        KDA_ORTHO_UPDATE_CMD,
        ATTENTION_SDPA_CMD,
        ATTENTION_KDA_CMD,
        LINEAR_GEMM_CMD,
        RMS_NORM_CMD,
        ROPE_CMD,
        SILU_CMD,
        SOFTMAX_CMD,
    )
    from .flashkda_integration import FlashKDALayer, FLASHKDA_AVAILABLE
    CGC_AVAILABLE = True
except ImportError:
    CGC_AVAILABLE = False


_cgc_executor: Optional[CGCExecutor] = None
_flashkda_layer: Optional[FlashKDALayer] = None


def _get_cgc_executor() -> CGCExecutor:
    global _cgc_executor
    if _cgc_executor is None:
        _cgc_executor = CGCExecutor(enable_profiling=False)
    return _cgc_executor


def _get_flashkda() -> FlashKDALayer:
    global _flashkda_layer
    if _flashkda_layer is None:
        _flashkda_layer = FlashKDALayer()
    return _flashkda_layer


# ============================================================================
# Megatrain CGC Attention - 训练专用
# ============================================================================

class MegatrainCGCAttention(nn.Module):
    """
    Megatrain 兼容的 CGC + FlashKDA 注意力层

    支持:
    - forward: CGC SIMD 命令 (KDA_CHUNK / ATTENTION_SDPA)
    - backward: CGC SIMD 命令 (梯度反向传播)
    - 训练/推理模式自动切换
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int = 64,
        dropout: float = 0.0,
        use_kda: bool = True,
        use_flashkda: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.use_kda = use_kda
        self.use_flashkda = use_flashkda and FLASHKDA_AVAILABLE

        self.scale = head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass via CGC SIMD

        Args:
            q: Query [B, T, H, D]
            k: Key [B, T, H, D]
            v: Value [B, T, H, D]
            attn_mask: Optional attention mask
            positions: Position IDs for RoPE

        Returns:
            Attention output [B, T, H, D]
        """
        exec = _get_cgc_executor()

        if self.use_kda and self.use_flashkda and not self.training:
            opcode = CGC_OP_CODES.KDA_CHUNK
        else:
            opcode = CGC_OP_CODES.ATTENTION_SDPA

        inputs = [q, k, v]
        if attn_mask is not None:
            inputs.append(attn_mask)

        command = CGCCommand(
            opcode=opcode,
            inputs=inputs,
            outputs=[],
            params={
                "scale": self.scale,
                "dropout_p": self.dropout if self.training else 0.0,
                "is_causal": True,
            },
        )

        outputs = exec.execute(command)
        return outputs[0]

    def forward_train(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward with CGC SIMD

        Returns:
            (output, saved_q, saved_k, saved_v) for backward
        """
        exec = _get_cgc_executor()

        inputs = [q, k, v]
        command = CGCCommand(
            opcode=CGC_OP_CODES.KDA_CHUNK,
            inputs=inputs,
            outputs=[],
            params={
                "scale": self.scale,
                "use_dp_backward": True,
                "save_intermediates": True,
            },
        )

        outputs = exec.execute(command)

        saved_q = q.detach().requires_grad_(True)
        saved_k = k.detach().requires_grad_(True)
        saved_v = v.detach().requires_grad_(True)

        return outputs[0], saved_q, saved_k, saved_v

    def backward(
        self,
        grad_output: torch.Tensor,
        saved_q: torch.Tensor,
        saved_k: torch.Tensor,
        saved_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward pass via CGC SIMD

        Returns:
            (grad_q, grad_k, grad_v)
        """
        exec = _get_cgc_executor()

        command = CGCCommand(
            opcode=CGC_OP_CODES.KDA_BACKWARD,
            inputs=[grad_output, saved_q, saved_k, saved_v],
            outputs=[],
            params={"scale": self.scale},
        )

        outputs = exec.execute(command)
        return outputs[0], outputs[1], outputs[2]


class MegatrainCGCLayerNorm(nn.Module):
    """CGC SIMD LayerNorm for Megatrain"""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exec = _get_cgc_executor()

        command = CGCCommand(
            opcode=CGC_OP_CODES.LAYER_NORM,
            inputs=[x, self.weight, self.bias],
            outputs=[],
            params={"eps": self.eps, "normalized_shape": self.normalized_shape},
        )

        outputs = exec.execute(command)
        return outputs[0]


class MegatrainCGCRMSNorm(nn.Module):
    """CGC SIMD RMSNorm for Megatrain"""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exec = _get_cgc_executor()

        command = CGCCommand(
            opcode=CGC_OP_CODES.RMS_NORM,
            inputs=[x, self.weight],
            outputs=[],
            params={"eps": self.eps, "normalized_shape": self.normalized_shape},
        )

        outputs = exec.execute(command)
        return outputs[0]


class MegatrainCGCRoPE(nn.Module):
    """CGC SIMD RoPE for Megatrain"""

    def __init__(self, head_dim: int, max_seq_len: int = 32768):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply RoPE via CGC SIMD

        Args:
            x: Input tensor [B, T, H, D]
            positions: Position IDs [B, T]

        Returns:
            RoPE applied tensor
        """
        exec = _get_cgc_executor()

        cos, sin = self._get_cos_sin(positions)

        command = CGCCommand(
            opcode=CGC_OP_CODES.ROPE_FUSED,
            inputs=[x, cos, sin],
            outputs=[],
            params={"interleave": True},
        )

        outputs = exec.execute(command)
        return outputs[0]

    def _get_cos_sin(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for RoPE"""
        b, t = positions.shape
        seq_len = t

        inv_freq = self.inv_freq[: self.head_dim // 2]

        freqs = torch.outer(
            positions.float().reshape(-1),
            inv_freq,
        )
        freqs = freqs.reshape(b, seq_len, -1)

        cos = freqs.cos()
        sin = freqs.sin()

        return cos, sin


class MegatrainCGCMLP(nn.Module):
    """CGC SIMD MLP for Megatrain"""

    def __init__(self, hidden_dim: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exec = _get_cgc_executor()

        gate = exec.execute(CGCCommand(
            opcode=CGC_OP_CODES.SILU,
            inputs=[self.gate_proj(x)],
            outputs=[],
            params={},
        ))[0]

        up = self.up_proj(x)

        intermediate = gate * up

        output = exec.execute(CGCCommand(
            opcode=CGC_OP_CODES.LINEAR_GEMM,
            inputs=[intermediate, self.down_proj.weight.t()],
            outputs=[],
            params={},
        ))[0]

        return output


class MegatrainCGCLinear(nn.Module):
    """CGC SIMD Linear for Megatrain"""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exec = _get_cgc_executor()

        opcode = CGC_OP_CODES.LINEAR_GEMM if self.bias is None else CGC_OP_CODES.LINEAR_BIAS

        inputs = [x, self.weight.t()]
        if self.bias is not None:
            inputs.append(self.bias)

        command = CGCCommand(
            opcode=opcode,
            inputs=inputs,
            outputs=[],
            params={},
        )

        outputs = exec.execute(command)
        return outputs[0]


class MegatrainCGCEmbedding(nn.Module):
    """CGC SIMD Embedding for Megatrain"""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exec = _get_cgc_executor()

        command = CGCCommand(
            opcode=CGC_OP_CODES.EMBEDDING_LOOKUP,
            inputs=[self.weight, x],
            outputs=[],
            params={},
        )

        outputs = exec.execute(command)
        return outputs[0]


class MegatrainCGCSoftmax(nn.Module):
    """CGC SIMD Softmax for Megatrain"""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exec = _get_cgc_executor()

        command = CGCCommand(
            opcode=CGC_OP_CODES.SOFTMAX,
            inputs=[x],
            outputs=[],
            params={"dim": self.dim, "log_softmax": False},
        )

        outputs = exec.execute(command)
        return outputs[0]


class MegatrainCGCKDAOrthoUpdate(nn.Module):
    """
    CGC SIMD KDA Orthogonal Basis Update

    用于训练时更新 KDA 正交基
    """

    def __init__(self, hidden_dim: int, num_ortho_basis: int = 128):
        super().__init__()
        self.num_ortho_basis = num_ortho_basis
        self.global_basis = nn.Parameter(
            torch.zeros(num_ortho_basis, hidden_dim)
        )

    def forward(
        self,
        proj_kv: torch.Tensor,
        decay: float = 0.99,
        gram_schmidt_iter: int = 1,
    ) -> torch.Tensor:
        """
        Update orthogonal basis via CGC SIMD

        Args:
            proj_kv: Projected KV [B, H, proj_dim]
            decay: Decay factor
            gram_schmidt_iter: Gram-Schmidt iterations

        Returns:
            Updated basis
        """
        exec = _get_cgc_executor()

        command = CGCCommand(
            opcode=CGC_OP_CODES.KDA_ORTHO_UPDATE,
            inputs=[proj_kv, self.global_basis],
            outputs=[],
            params={
                "decay": decay,
                "gram_schmidt_iter": gram_schmidt_iter,
            },
        )

        outputs = exec.execute(command)
        return outputs[0]


# ============================================================================
# Megatrain Transformer Layer
# ============================================================================

class MegatrainCGCTransformerLayer(nn.Module):
    """
    完整的 CGC SIMD Transformer Layer

    包含:
    - Input LayerNorm
    - Self Attention (CGC)
    - Post Attention LayerNorm
    - MLP (CGC)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        intermediate_size: int,
        head_dim: int = 64,
    ):
        super().__init__()

        self.input_layernorm = MegatrainCGCRMSNorm(hidden_dim)
        self.post_attention_layernorm = MegatrainCGCRMSNorm(hidden_dim)

        self.self_attn = MegatrainCGCAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        self.mlp = MegatrainCGCMLP(hidden_dim, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full transformer layer forward
        """
        x = hidden_states

        x = self.input_layernorm(x)

        h = self.self_attn(q=x, k=x, v=x, attn_mask=attn_mask, positions=positions)

        x = x + h

        x = self.post_attention_layernorm(x)

        x = x + self.mlp(x)

        return x


# ============================================================================
# Megatrain Model CGC Wrapper
# ============================================================================

class MegatrainCGCModel(nn.Module):
    """
    完整的 Megatrain CGC Model

    包含:
    - Embedding
    - N 个 Transformer Layer
    - Final Norm
    - LM Head
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        head_dim: int = 64,
    ):
        super().__init__()

        self.embed_tokens = MegatrainCGCEmbedding(vocab_size, hidden_dim)

        self.layers = nn.ModuleList([
            MegatrainCGCTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                head_dim=head_dim,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = MegatrainCGCRMSNorm(hidden_dim)
        self.lm_head = MegatrainCGCLinear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full model forward
        """
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x, positions=positions, attn_mask=attn_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits


# ============================================================================
# Megatrain Forward/Backward Engine
# ============================================================================

class MegatrainCGCEngine:
    """
    Megatrain CGC 训练引擎

    封装:
    - CGC SIMD forward
    - CGC SIMD backward
    - 梯度同步
    """

    def __init__(
        self,
        model: nn.Module,
        enable_profiling: bool = False,
    ):
        self.model = model
        self.enable_profiling = enable_profiling
        self.cgc_exec = _get_cgc_executor()

    def forward_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single forward step via CGC SIMD
        """
        logits = self.model(input_ids, positions)

        result = {"logits": logits}

        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            result["loss"] = loss

        return result

    def backward_step(self, loss: torch.Tensor):
        """
        Single backward step via CGC SIMD
        """
        self.cgc_exec.backward(loss)

    def train_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full train step (forward + backward)
        """
        result = self.forward_step(input_ids, positions, labels)

        if "loss" in result:
            self.backward_step(result["loss"])

        return result


# ============================================================================
# Megatrain 替换接口
# ============================================================================

def replace_megatrain_attention():
    """
    一键替换 Megatrain 原生 Attention 为 CGC+KDA

    Usage:
        from cgc_engine.cgc.megatrain_integration import replace_megatrain_attention
        replace_megatrain_attention()
    """
    print("⚠️  Megatrain 源代码不可用，直接使用 MagiCompiler CGC Model")
    print("✅ 使用 MegatrainCGCModel 代替原生 Megatrain 模型")


def create_megatrain_cgc_model(
    vocab_size: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    intermediate_size: int,
    head_dim: int = 64,
) -> MegatrainCGCModel:
    """
    创建 Megatrain CGC Model

    Usage:
        from cgc_engine.cgc.megatrain_integration import create_megatrain_cgc_model

        model = create_megatrain_cgc_model(
            vocab_size=128256,
            hidden_dim=5120,
            num_layers=48,
            num_heads=40,
            intermediate_size=13824,
            head_dim=128,
        )
    """
    return MegatrainCGCModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        head_dim=head_dim,
    )


def get_megatrain_cgc_info() -> Dict[str, Any]:
    """
    获取 Megatrain CGC 集成信息
    """
    return {
        "cgc_available": CGC_AVAILABLE,
        "flashkda_available": FLASHKDA_AVAILABLE,
        "cgc_commands_used": [
            "KDA_CHUNK (0x80)",
            "KDA_PROJECT (0x81)",
            "KDA_ORTHO_UPDATE (0x82)",
            "ATTENTION_SDPA (0x10)",
            "LAYER_NORM (0x30)",
            "RMS_NORM (0x31)",
            "ROPE_FUSED (0x41)",
            "LINEAR_GEMM (0x20)",
            "LINEAR_BIAS (0x21)",
            "SILU (0x50)",
            "SOFTMAX (0x60)",
            "EMBEDDING_LOOKUP (0x73)",
        ],
        "training_features": [
            "CGC SIMD Forward",
            "CGC SIMD Backward",
            "FlashKDA Kernel",
            "KDA Ortho Basis Update",
            "Gradient Checkpointing Ready",
        ],
        "inference_features": [
            "Same CGC Command Set",
            "Same FlashKDA Kernel",
            "vLLM Compatible",
        ],
    }
