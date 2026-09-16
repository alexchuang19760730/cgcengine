"""
Dual state management for Loop MoE recurrent block.

Manages two parallel state tracks during recurrent iterations:
1. KV Cache state (for traditional Gated Attention, if enabled)
2. DeltaNet hidden state (fixed-dimension linear attention state)

Plus the recurrent flow state (residual stream carried across iterations)
and ACT (Adaptive Computation Time) tracking.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

import torch


@dataclass
class RecurrentDualState:
    """
    Container for all recurrent block states.

    Attributes:
        kv_state: KV Cache dict with 'key' and 'value' tensors,
                  or None if Gated Attention is not used in loop.
                  Shape: [batch, kv_heads, seq_len, head_dim]
        delta_state: DeltaNet hidden state, fixed dimension.
                     Shape: [batch, num_v_heads, head_v_dim, head_v_dim]
        recurrent_flow: Residual stream carried across iterations.
                        Shape: [batch, seq_len, hidden_size]
        act_cumulative_prob: Cumulative ACT halting probability.
        step_count: Current iteration step number.
        outputs_history: List of outputs from each iteration (for ACT weighted average).
        halting_probs_history: List of halting probabilities from each iteration.
    """

    kv_state: Optional[Dict[str, torch.Tensor]] = None
    delta_state: Optional[torch.Tensor] = None
    recurrent_flow: Optional[torch.Tensor] = None
    act_cumulative_prob: float = 0.0
    step_count: int = 0
    outputs_history: list = field(default_factory=list)
    halting_probs_history: list = field(default_factory=list)

    def clone(self) -> "RecurrentDualState":
        """Deep clone state (for gradient isolation / checkpointing)."""
        return RecurrentDualState(
            kv_state={
                k: v.clone() for k, v in self.kv_state.items()
            } if self.kv_state is not None else None,
            delta_state=self.delta_state.clone() if self.delta_state is not None else None,
            recurrent_flow=self.recurrent_flow.clone() if self.recurrent_flow is not None else None,
            act_cumulative_prob=self.act_cumulative_prob,
            step_count=self.step_count,
            outputs_history=list(self.outputs_history),
            halting_probs_history=list(self.halting_probs_history),
        )

    def should_stop(self, epsilon: float = 1e-3) -> bool:
        """Check if ACT stopping criterion is met."""
        return self.act_cumulative_prob >= (1.0 - epsilon)

    def get_act_weighted_output(self) -> Optional[torch.Tensor]:
        """
        Compute ACT-weighted average of all iteration outputs.

        In ACT, the final output is a weighted average where weights are
        the halting probabilities (with the final iteration getting the
        remaining probability mass).

        Halting probabilities are stored as tensors to preserve gradient flow.
        """
        if not self.outputs_history:
            return None

        if len(self.outputs_history) == 1:
            return self.outputs_history[0]

        # Compute weights from halting probabilities (tensors for autograd)
        weights = []
        cumulative = None
        for i, p in enumerate(self.halting_probs_history):
            # p is [batch, 1] tensor; use mean over batch for scalar weight
            p_mean = p.mean() if isinstance(p, torch.Tensor) else torch.tensor(p)
            if i == len(self.halting_probs_history) - 1:
                # Last iteration gets remaining probability
                w = (1.0 - cumulative) if cumulative is not None else torch.tensor(1.0)
            else:
                w = p_mean
                cumulative = p_mean if cumulative is None else cumulative + p_mean
            weights.append(w)

        # Weighted average
        output = torch.zeros_like(self.outputs_history[0])
        for w, out in zip(weights, self.outputs_history):
            output = output + w * out

        return output

    def get_metrics(self) -> dict:
        """Return state metrics for monitoring."""
        metrics = {
            "step_count": self.step_count,
            "act_cumulative_prob": self.act_cumulative_prob,
            "num_iterations": len(self.outputs_history),
        }

        if self.delta_state is not None:
            metrics["delta_state_norm"] = torch.norm(
                self.delta_state, dim=(-2, -1)
            ).mean().item()
            metrics["delta_state_max"] = self.delta_state.abs().max().item()

        if self.kv_state is not None:
            metrics["kv_seq_len"] = self.kv_state["key"].shape[2]

        if self.recurrent_flow is not None:
            metrics["recurrent_flow_norm"] = torch.norm(
                self.recurrent_flow, dim=-1
            ).mean().item()

        return metrics


def init_dual_state(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_value_heads: int,
    value_head_dim: int,
    use_kv_state: bool = False,
    kv_num_heads: int = 4,
    kv_head_dim: int = 128,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> RecurrentDualState:
    """
    Initialize a fresh RecurrentDualState with zeros.

    Args:
        batch_size: batch size
        seq_len: sequence length
        hidden_size: model hidden dimension
        num_value_heads: DeltaNet number of value heads
        value_head_dim: DeltaNet value head dimension
        use_kv_state: whether to initialize KV Cache (for Gated Attention)
        kv_num_heads: number of KV heads (if use_kv_state)
        kv_head_dim: KV head dimension (if use_kv_state)
        device: torch device
        dtype: torch dtype

    Returns:
        Initialized RecurrentDualState
    """
    state = RecurrentDualState()

    # DeltaNet state: fixed dimension [batch, num_v_heads, v_dim, v_dim]
    state.delta_state = torch.zeros(
        batch_size,
        num_value_heads,
        value_head_dim,
        value_head_dim,
        device=device,
        dtype=dtype,
    )

    # KV Cache state (optional, for hybrid attention)
    if use_kv_state:
        state.kv_state = {
            "key": torch.zeros(
                batch_size, kv_num_heads, 0, kv_head_dim,
                device=device, dtype=dtype,
            ),
            "value": torch.zeros(
                batch_size, kv_num_heads, 0, kv_head_dim,
                device=device, dtype=dtype,
            ),
        }

    # Recurrent flow initialized to None (set on first iteration)
    state.recurrent_flow = None

    state.step_count = 0
    state.act_cumulative_prob = 0.0

    return state


def clip_delta_state(
    delta_state: torch.Tensor,
    max_norm: float = 5.0,
) -> torch.Tensor:
    """
    Clip DeltaNet hidden state norm to prevent explosion (LTI stability).

    Args:
        delta_state: [batch, num_heads, v_dim, v_dim]
        max_norm: maximum allowed norm per state matrix

    Returns:
        Clipped delta_state
    """
    delta_norm = torch.norm(delta_state, dim=(-2, -1), keepdim=True)
    clip_factor = torch.clamp(max_norm / (delta_norm + 1e-8), max=1.0)
    return delta_state * clip_factor
