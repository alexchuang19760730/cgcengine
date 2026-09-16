"""
LoopMoEConfig: Configuration class for Loop MoE model.

Integrates Qwen3.5/3.6 (Gated DeltaNet + MoE) and OpenMythos (RDT recurrent)
configuration parameters into a single dataclass.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import json


@dataclass
class LoopMoEConfig:
    """
    Complete configuration for Loop MoE.

    Combines:
    - Base model config (Qwen3.5/3.6 compatible)
    - Gated DeltaNet config (linear attention)
    - MoE config (256 experts, top-8 + 1 shared)
    - RDT recurrent config (OpenMythos compatible)
    - LTI constraint config (extended for dual state)
    - Weight grafting config
    """

    # ===== Base Model Config =====
    vocab_size: int = 152064
    hidden_size: int = 2048
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"

    # ===== Gated DeltaNet Config (Qwen3.5/3.6) =====
    # Linear attention head configuration
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: Optional[str] = "silu"  # gate activation for RMSNormGated

    # Whether to also use traditional Gated Attention in the recurrent block
    # (Qwen3.5 uses hybrid: some layers DeltaNet, some layers traditional attn)
    use_gated_attention_in_loop: bool = False
    gated_attn_num_heads: int = 16
    gated_attn_num_kv_heads: int = 4  # GQA
    gated_attn_head_dim: int = 128

    # ===== MoE Config (Qwen3.6: 256 experts, top-8 + 1 shared) =====
    num_experts: int = 256
    num_experts_per_tok: int = 8  # top-k
    num_shared_experts: int = 1
    moe_intermediate_size: int = 1408  # per-expert FFN intermediate dimension
    moe_router_loss_coef: float = 0.01
    moe_aux_loss_coef: float = 0.001
    moe_z_loss_coef: float = 0.001
    moe_expert_dropout: float = 0.0
    norm_topk_prob: bool = True  # normalize top-k router probabilities

    # ===== RDT Recurrent Config (OpenMythos) =====
    num_prelude_layers: int = 2
    num_coda_layers: int = 1
    num_hidden_layers: int = 28  # total Qwen layers (for grafting reference)
    max_recurrent_steps: int = 8
    min_recurrent_steps: int = 1
    act_epsilon: float = 1e-3  # ACT convergence threshold
    act_temperature: float = 1.0  # halting probability temperature

    # ===== Extended LTI Constraint Config =====
    use_lti_constraint: bool = True
    lti_rho_target_recurrent: float = 0.99
    lti_rho_target_delta: float = 0.95
    lti_weight_recurrent: float = 1.0
    lti_weight_delta: float = 0.5
    lti_warmup_steps: int = 1000
    lti_delta_state_norm_target: float = 1.0
    lti_delta_state_max_norm: float = 5.0
    delta_decay_max: float = 0.95  # upper bound for DeltaNet decay rate

    # ===== Weight Grafting Config =====
    graft_source_model: str = "Qwen/Qwen3.5-35B-A3B"
    graft_prelude_layer_indices: Optional[List[int]] = None
    graft_coda_layer_indices: Optional[List[int]] = None
    graft_deltanet_source_layer: int = 4
    graft_moe_source_layer: int = 4

    # ===== Training / Inference =====
    use_cache: bool = True
    output_router_logits: bool = True
    initializer_range: float = 0.02

    def __post_init__(self):
        """Validate and derive config values."""
        if self.graft_prelude_layer_indices is None:
            self.graft_prelude_layer_indices = list(range(self.num_prelude_layers))
        if self.graft_coda_layer_indices is None:
            self.graft_coda_layer_indices = [
                self.num_hidden_layers - 1 - i
                for i in range(self.num_coda_layers)
            ]

        # Derived dimensions
        self.linear_key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_value_dim = self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def delta_state_shape(self) -> Tuple[int, int, int]:
        """Shape of DeltaNet hidden state: (num_heads, head_v_dim, head_v_dim)."""
        return (self.linear_num_value_heads, self.linear_value_head_dim, self.linear_value_head_dim)

    @classmethod
    def from_json(cls, path: str) -> "LoopMoEConfig":
        """Load config from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding derived fields set in __post_init__."""
        derived_fields = {"linear_key_dim", "linear_value_dim"}
        return {k: v for k, v in self.__dict__.items() if k not in derived_fields}

    def validate(self) -> List[str]:
        """
        Validate configuration consistency.
        Returns list of warning/error messages (empty = valid).
        """
        issues = []

        if self.num_experts_per_tok >= self.num_experts:
            issues.append(
                f"num_experts_per_tok ({self.num_experts_per_tok}) >= "
                f"num_experts ({self.num_experts})"
            )

        if self.max_recurrent_steps < self.min_recurrent_steps:
            issues.append(
                f"max_recurrent_steps ({self.max_recurrent_steps}) < "
                f"min_recurrent_steps ({self.min_recurrent_steps})"
            )

        if self.lti_rho_target_delta > self.lti_rho_target_recurrent:
            issues.append(
                "lti_rho_target_delta > lti_rho_target_recurrent: "
                "Delta state should have stricter (smaller) spectral radius target "
                "due to nested recursion"
            )

        if self.delta_decay_max > 1.0:
            issues.append(f"delta_decay_max ({self.delta_decay_max}) > 1.0 will cause state explosion")

        if self.num_prelude_layers + self.num_coda_layers >= self.num_hidden_layers:
            issues.append(
                f"prelude ({self.num_prelude_layers}) + coda ({self.num_coda_layers}) "
                f">= total layers ({self.num_hidden_layers}), no room for recurrent block"
            )

        return issues
