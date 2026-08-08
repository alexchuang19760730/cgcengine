"""CGC MTP Head as EAGLE draft model for sglang.

Implements our trained MTP head (DeepSeek-V3 MTP style) as a sglang-compatible
EAGLE draft model. This allows deploying any trained .pt MTP head checkpoint
as an in-process EAGLE draft model, eliminating HTTP overhead.

Architecture (matches CGC_Phase2/mtp_head/model.py):
  proj(concat(hidden, embed)) -> norm1 -> attn -> +residual -> norm2 -> mlp -> +residual -> norm_out

  NOTE: concat order is [hidden(target), embed(token)] — matches training code:
    distill_train.py: torch.cat([h_3d, e_3d], dim=-1)
    model.py: torch.cat([hidden_states, token_embeddings], dim=-1)

Compatible with sglang's EAGLE speculative decoding:
  - Uses Qwen2DecoderLayer for optimized attention backend
  - Shares embed_tokens and lm_head from target model (via base class set_embed_and_head)
  - Receives target hidden states via forward_batch.spec_info.hidden_states

Config.json example:
  {
    "architectures": ["CgcMtpForCausalLMEagle"],
    "hidden_size": 2048,
    "vocab_size": 151936,
    "num_hidden_layers": 1,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 128,
    "intermediate_size": 5632,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 262144,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16"
  }
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM
from sglang.srt.utils import add_prefix

Qwen2Config = None


def _unwrap(x):
    """Unwrap (output, bias) tuple from sglang parallel linear layers."""
    return x[0] if isinstance(x, tuple) else x


class CgcMtpModel(nn.Module):
    """Inner model: our MTP head forward pass.

    Flow (manual attention to match training seq_len=1 behavior):
      1. embed_tokens(input_ids) -> token_embed
      2. fc(concat(target_hidden, token_embed)) -> projected  [order matches training]
      3. norm1 -> qkv_proj -> take V -> o_proj (manual attn, matches training seq_len=1)
      4. +residual -> norm2 -> mlp -> +residual
      5. norm_out -> output hidden_states

    Why manual attention:
      Training used MTPAttention with F.scaled_dot_product_attention on seq_len=1.
      When seq_len=1, softmax(Q.K^T/sqrt(d)) = 1.0, so attn_out = o_proj(v_proj(norm1(x))).
      sglang EAGLE uses RadixAttention with KV cache attending to ALL previous tokens,
      which is fundamentally different. We replicate training behavior by computing
      attention manually (only current token, no KV cache).
    """

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # Embedding (shared from target model at runtime via set_embed_and_head)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        # Projection: concat(target_hidden, token_embed) -> hidden  [order matches training]
        self.fc = torch.nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        # Single decoder layer (standard Qwen2 for optimized attention)
        # NOTE: We do NOT skip input_layernorm (unlike standard EAGLE).
        # Our MTP head uses pre-norm: norm1 -> attn -> residual -> norm2 -> mlp -> residual
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(
                    config,
                    layer_id=0,
                    quant_config=quant_config,
                    prefix=add_prefix("layers.0", prefix),
                )
            ]
        )

        # Final output norm (our MTP head has norm_out after the decoder layer)
        self.norm_out = RMSNorm(
            config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        # Concat target hidden states with token embedding
        # CRITICAL: order must match training! Training used [hidden_states, token_embed]
        # (distill_train.py line 200: torch.cat([h_3d, e_3d], dim=-1))
        # model.py line 181: torch.cat([hidden_states, token_embeddings], dim=-1)
        target_hidden = forward_batch.spec_info.hidden_states

        hidden_states = self.fc(
            torch.cat((target_hidden, hidden_states), dim=-1)
        )

        # Manual forward matching training behavior exactly.
        # Training: x = x + attn(norm1(x)), where attn on seq_len=1 gives o_proj(v_proj(norm1(x)))
        # because softmax of single element = 1.0, so attention output = V.
        # We skip RadixAttention (which uses KV cache) and compute manually.
        layer = self.layers[0]

        # norm1 (input_layernorm) — one-arg form returns single tensor
        normed = _unwrap(layer.input_layernorm(hidden_states))

        # Manual attention: qkv_proj -> take V -> o_proj
        # With TP=1 and num_kv_heads=num_heads, qkv output is [Q, K, V] concatenated
        # NOTE: sglang's QKVParallelLinear returns (output, bias) tuple
        qkv = _unwrap(layer.self_attn.qkv_proj(normed))
        # QKV stacking: [Q (hidden), K (kv_hidden), V (kv_hidden)]
        # With num_kv_heads=num_heads=16, all chunks are equal size
        # For seq_len=1, attention output = V (softmax of single element = 1.0)
        v = qkv.chunk(3, dim=-1)[2]
        # o_proj (RowParallelLinear) also returns (output, bias) tuple
        attn_out = _unwrap(layer.self_attn.o_proj(v))

        # Attention residual
        hidden_states = hidden_states + attn_out

        # norm2 (post_attention_layernorm) — one-arg form returns single tensor
        normed2 = _unwrap(layer.post_attention_layernorm(hidden_states))

        # MLP + residual
        hidden_states = hidden_states + _unwrap(layer.mlp(normed2))

        # Final norm (our MTP head's norm_out)
        hidden_states = _unwrap(self.norm_out(hidden_states))
        return hidden_states


class CgcMtpForCausalLMEagle(Qwen2ForCausalLM):
    """CGC MTP Head as EAGLE draft model.

    Extends Qwen2ForCausalLM to inherit logits processing, weight loading,
    and embed/head sharing (set_embed_and_head from base class).
    The inner model (CgcMtpModel) implements our MTP head architecture.
    """

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.model = CgcMtpModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

    # NOTE: Do NOT override set_embed_and_head / set_embed.
    # The base class Qwen2ForCausalLM correctly replaces only .weight attributes
    # while keeping the VocabParallelEmbedding / ParallelLMHead modules callable.

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights from safetensors.

        The safetensors file (produced by convert_pt_to_eagle.py) already uses
        sglang internal names with 'model.' prefix:
          model.fc.weight
          model.layers.0.input_layernorm.weight
          model.layers.0.self_attn.q/k/v/o_proj.weight
          model.layers.0.post_attention_layernorm.weight
          model.layers.0.mlp.gate/up/down_proj.weight
          model.norm_out.weight

        We skip lm_head (shared from target model at runtime).
        Everything else goes to the base class load_weights which handles
        stacked param mapping (qkv_proj, gate_up_proj) and params_dict matching.
        """
        for name, loaded_weight in weights:
            if "lm_head" in name:
                continue
            super().load_weights([(name, loaded_weight)])


EntryClass = [CgcMtpForCausalLMEagle]
