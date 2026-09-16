"""
LoRA adapter for Loop MoE continual pre-training.

Applies low-rank adaptation to:
- Gated DeltaNet projections (q_proj, k_proj, v_proj, o_proj)
- MoE expert gate/up/down projections
- Recurrent block input/output projections
- Prelude/Coda attention projections

Freezes base model weights, only trains LoRA A/B matrices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn


@dataclass
class LoopMoELoRAConfig:
    """Configuration for LoRA adaptation on Loop MoE."""

    # LoRA rank and alpha
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Which modules to apply LoRA to
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # DeltaNet attention
        "gate_proj", "up_proj", "down_proj",       # MoE experts
        "input_proj", "output_proj",                # Recurrent block
    ])

    # Whether to apply LoRA to shared experts
    lora_shared_experts: bool = True

    # Whether to apply LoRA to router (usually not - router is small)
    lora_router: bool = False

    # Whether to train embedding / lm_head (usually frozen for LoRA)
    train_embeddings: bool = False
    train_lm_head: bool = False

    # Scaling
    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.lora_rank


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation: y = Wx + (B @ A)x * scaling."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Freeze base weights
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        # LoRA A (rank x in) and B (out x rank)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B starts at zero so initial output = base output

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into base linear and return a new Linear."""
        merged_weight = self.base_linear.weight.data + (self.lora_B @ self.lora_A * self.scaling).data
        merged = nn.Linear(
            self.base_linear.in_features,
            self.base_linear.out_features,
            bias=self.base_linear.bias is not None,
        )
        merged.weight.data = merged_weight
        if self.base_linear.bias is not None:
            merged.bias.data = self.base_linear.bias.data.clone()
        return merged


def _find_linear_modules(model: nn.Module, target_names: Set[str]) -> List[str]:
    """Find all module paths whose leaf name matches target_names."""
    matches = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(t) for t in target_names):
            matches.append(name)
    return matches


def apply_lora(
    model: nn.Module,
    config: LoopMoELoRAConfig,
) -> nn.Module:
    """
    Apply LoRA adaptation to target modules in the Loop MoE model.

    Freezes all base weights, only LoRA A/B matrices are trainable.
    Returns the model (modified in-place).
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Optionally unfreeze embeddings / lm_head
    if config.train_embeddings:
        if hasattr(model, "embed_tokens"):
            for param in model.embed_tokens.parameters():
                param.requires_grad = True
    if config.train_lm_head:
        if hasattr(model, "lm_head"):
            for param in model.lm_head.parameters():
                param.requires_grad = True

    # Find target linear modules
    target_set = set(config.target_modules)
    if not config.lora_router:
        target_set.discard("router")
        target_set.discard("gate")  # router gate is separate from expert gate_proj

    matches = _find_linear_modules(model, target_set)

    # Replace each with LoRALinear
    for path in matches:
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        base_linear = getattr(parent, parts[-1])
        lora_linear = LoRALinear(
            base_linear,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        setattr(parent, parts[-1], lora_linear)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Applied to {len(matches)} modules")
    print(f"[LoRA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA adapters back into base weights.
    Returns model with LoRALinear replaced by plain Linear.
    """
    to_merge = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            to_merge.append((name, module))

    for path, lora_module in to_merge:
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        merged = lora_module.merge()
        setattr(parent, parts[-1], merged)

    print(f"[LoRA] Merged {len(to_merge)} adapters into base weights")
    return model


def save_lora_state(model: nn.Module, path: str) -> None:
    """Save only LoRA trainable parameters (A/B matrices)."""
    lora_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_A" in name or "lora_B" in name):
            lora_state[name] = param.data.cpu()
    torch.save(lora_state, path)
    print(f"[LoRA] Saved {len(lora_state)} tensors to {path}")


def load_lora_state(model: nn.Module, path: str) -> nn.Module:
    """Load LoRA parameters from checkpoint."""
    lora_state = torch.load(path, map_location="cpu", weights_only=True)
    model_dict = dict(model.named_parameters())
    loaded = 0
    for name, tensor in lora_state.items():
        if name in model_dict:
            model_dict[name].data.copy_(tensor)
            loaded += 1
    print(f"[LoRA] Loaded {loaded}/{len(lora_state)} tensors from {path}")
    return model
