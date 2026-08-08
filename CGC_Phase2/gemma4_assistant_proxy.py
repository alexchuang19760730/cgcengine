from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    normed = x_fp32 * torch.rsqrt(variance + eps)
    return normed.to(x.dtype) * weight


@dataclass
class AssistantLayerWeights:
    input_layernorm: torch.Tensor
    q_proj: torch.Tensor
    q_norm: torch.Tensor
    o_proj: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor
    pre_feedforward_layernorm: torch.Tensor
    post_feedforward_layernorm: Optional[torch.Tensor]
    layer_scalar: float
    head_dim: int


class Gemma4AssistantProxy:
    """Minimal trained drafter proxy derived from official Gemma4 assistant weights.

    The official assistant expects shared KV states from the target model. Our current
    llama.cpp verify loop cannot expose those states, so this proxy keeps the trained
    projections/MLP stack and uses a q-only attention surrogate. That is still a real
    trained assistant path and removes n-gram fallback from the hot path.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        state = load_file(model_path, device=device)
        self.device = torch.device(device)
        self.dtype = torch.float32

        self.embed_weight = state["model.embed_tokens.weight"].to(self.dtype)
        self.pre_projection = state["pre_projection.weight"].to(self.dtype)
        self.post_projection = state["post_projection.weight"].to(self.dtype)
        self.final_norm = state["model.norm.weight"].to(self.dtype)
        self.layers = self._load_layers(state)

    @staticmethod
    def _load_layers(state: dict[str, torch.Tensor]) -> list[AssistantLayerWeights]:
        layers: list[AssistantLayerWeights] = []
        index = 0
        while f"model.layers.{index}.input_layernorm.weight" in state:
            prefix = f"model.layers.{index}"
            q_proj = state[f"{prefix}.self_attn.q_proj.weight"].to(torch.float32)
            q_norm = state[f"{prefix}.self_attn.q_norm.weight"].to(torch.float32)
            head_dim = int(q_norm.shape[0])
            layers.append(
                AssistantLayerWeights(
                    input_layernorm=state[f"{prefix}.input_layernorm.weight"].to(torch.float32),
                    q_proj=q_proj,
                    q_norm=q_norm,
                    o_proj=state[f"{prefix}.self_attn.o_proj.weight"].to(torch.float32),
                    gate_proj=state[f"{prefix}.mlp.gate_proj.weight"].to(torch.float32),
                    up_proj=state[f"{prefix}.mlp.up_proj.weight"].to(torch.float32),
                    down_proj=state[f"{prefix}.mlp.down_proj.weight"].to(torch.float32),
                    pre_feedforward_layernorm=state[f"{prefix}.pre_feedforward_layernorm.weight"].to(torch.float32),
                    post_feedforward_layernorm=state.get(f"{prefix}.post_feedforward_layernorm.weight", None).to(torch.float32)
                    if state.get(f"{prefix}.post_feedforward_layernorm.weight", None) is not None
                    else None,
                    layer_scalar=float(state[f"{prefix}.layer_scalar"].item()),
                    head_dim=head_dim,
                )
            )
            index += 1
        return layers

    def _proxy_backbone_embed(self, token_id: int) -> torch.Tensor:
        small_embed = self.embed_weight[token_id].unsqueeze(0).unsqueeze(0)
        return F.linear(small_embed, self.post_projection)

    def _layer_forward(self, x: torch.Tensor, layer: AssistantLayerWeights) -> torch.Tensor:
        normed = _rms_norm(x, layer.input_layernorm)
        q = F.linear(normed, layer.q_proj)
        q = q.view(q.shape[0], q.shape[1], -1, layer.head_dim)
        q = _rms_norm(q, layer.q_norm)
        q = q.reshape(q.shape[0], q.shape[1], -1)
        attn_out = F.linear(q, layer.o_proj)
        x = x + attn_out * layer.layer_scalar

        ff_in = _rms_norm(x, layer.pre_feedforward_layernorm)
        ff = F.linear(F.silu(F.linear(ff_in, layer.gate_proj)) * F.linear(ff_in, layer.up_proj), layer.down_proj)
        x = x + ff * layer.layer_scalar
        if layer.post_feedforward_layernorm is not None:
            x = _rms_norm(x, layer.post_feedforward_layernorm)
        return x

    def step(self, hidden_state: torch.Tensor, token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0).unsqueeze(0)
        elif hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(0)
        hidden_state = hidden_state.to(self.dtype)

        proxy_embed = self._proxy_backbone_embed(token_id).to(self.dtype)
        x = torch.cat([hidden_state, proxy_embed], dim=-1)
        x = F.linear(x, self.pre_projection)
        for layer in self.layers:
            x = self._layer_forward(x, layer)
        small_hidden = _rms_norm(x, self.final_norm)
        logits = F.linear(small_hidden, self.embed_weight)
        next_hidden = F.linear(small_hidden, self.post_projection)
        return next_hidden[:, 0, :], logits[:, 0, :]
