"""
Loop MoE full model: Prelude (static) -> RecurrentBlock (loop) -> Coda (static).

Three-stage architecture inherited from OpenMythos RDT:
1. Prelude: static transformer layers (reused from Qwen3.6 shallow layers)
   - Responsible for initial feature extraction
2. RecurrentBlock: loop with Gated DeltaNet + Qwen MoE + ACT
   - Core recurrent reasoning, dynamic iteration depth
3. Coda: static transformer layers (reused from Qwen3.6 deep layers)
   - Final feature refinement before LM head

This is the Phase 0 skeleton: Prelude/Coda use simple dense transformer layers
(no MoE) for verification. Later phases will graft Qwen3.6 weights and use
full Qwen decoder layers for Prelude/Coda.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import LoopMoEConfig
from ..states.dual_state import RecurrentDualState
from .recurrent_block import RMSNorm, LoopMoERecurrentBlock


class DenseDecoderLayer(nn.Module):
    """
    Simple dense transformer decoder layer for Prelude/Coda blocks.

    Uses standard multi-head self-attention + SwiGLU FFN.
    In later phases, this will be replaced with full Qwen decoder layers
    (with Gated Attention + MoE) for weight grafting.
    """

    def __init__(self, config: LoopMoEConfig, layer_id: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id

        # Simple multi-head self-attention
        self.num_heads = config.gated_attn_num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # SwiGLU FFN
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)

        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Self-attention
        residual = x
        x_norm = self.input_layernorm(x)

        q = self.q_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_out = self.o_proj(attn_out)

        x = residual + attn_out

        # FFN
        residual = x
        x_norm = self.post_attention_layernorm(x)
        ffn_out = self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x_norm)) * self.up_proj(x_norm)
        )
        x = residual + ffn_out

        return x


class LoopMoEModel(nn.Module):
    """
    Full Loop MoE model.

    Architecture:
      Embedding -> Prelude (N dense layers) -> RecurrentBlock (loop) -> Coda (N dense layers) -> LM Head

    The recurrent block contains Gated DeltaNet + Qwen MoE with ACT.
    """

    def __init__(self, config: LoopMoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Prelude block (static dense layers)
        self.prelude_layers = nn.ModuleList(
            [DenseDecoderLayer(config, layer_id=i) for i in range(config.num_prelude_layers)]
        )

        # Recurrent block (core loop)
        self.recurrent_block = LoopMoERecurrentBlock(config, layer_id=config.num_prelude_layers)

        # Coda block (static dense layers)
        self.coda_layers = nn.ModuleList(
            [
                DenseDecoderLayer(config, layer_id=config.num_prelude_layers + 1 + i)
                for i in range(config.num_coda_layers)
            ]
        )

        # Final norm + LM head
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Simple weight initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        initial_state: Optional[RecurrentDualState] = None,
        max_recurrent_steps: Optional[int] = None,
    ) -> dict:
        """
        Full forward pass.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] attention mask
            labels: [batch, seq_len] labels for loss computation
            initial_state: optional initial recurrent state
            max_recurrent_steps: override max recurrent steps

        Returns:
            dict with:
                loss: language modeling loss (if labels provided)
                logits: [batch, seq_len, vocab_size]
                recurrent_state: final recurrent dual state
                metrics: iteration metrics
        """
        batch_size, seq_len = input_ids.shape

        # Create causal attention mask if not provided
        if attention_mask is None:
            # Causal mask: [1, 1, seq_len, seq_len]
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
                diagonal=1,
            )
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # ===== Embedding =====
        hidden_states = self.embed_tokens(input_ids)

        # ===== Prelude (static layers) =====
        for layer in self.prelude_layers:
            hidden_states = layer(hidden_states, attention_mask)

        # ===== Recurrent block (loop with ACT) =====
        hidden_states, recurrent_state, recurrent_metrics = self.recurrent_block(
            hidden_states,
            initial_state=initial_state,
            attention_mask=attention_mask,
            training=self.training,
            max_steps=max_recurrent_steps,
        )

        # ===== Coda (static layers) =====
        for layer in self.coda_layers:
            hidden_states = layer(hidden_states, attention_mask)

        # ===== Final norm + LM head =====
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # ===== Loss =====
        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "recurrent_state": recurrent_state,
            "metrics": recurrent_metrics,
        }

    def count_parameters(self) -> dict:
        """Count model parameters by component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        components = {
            "embedding": sum(p.numel() for p in self.embed_tokens.parameters()),
            "prelude": sum(p.numel() for p in self.prelude_layers.parameters()),
            "recurrent_block": sum(p.numel() for p in self.recurrent_block.parameters()),
            "coda": sum(p.numel() for p in self.coda_layers.parameters()),
            "lm_head": sum(p.numel() for p in self.lm_head.parameters()),
        }

        return {
            "total": total,
            "trainable": trainable,
            "components": components,
        }
