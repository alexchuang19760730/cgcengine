"""
Extended LTI (Linear Time-Invariant) spectral radius constraint for Loop MoE.

Extends OpenMythos's original LTI constraint (which only stabilizes the
recurrent residual flow) to also cover the DeltaNet hidden state.

This is necessary because Gated DeltaNet has its own recursive state update
(state_t = decay * state_{t-1} + K^T * V), creating nested recursion with
the outer RDT loop. Without joint constraint, the nested recursion can cause
hidden state explosion or drift.

Reference: OpenMythos LTI constraint (spectral radius of state transition)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExtendedLTILoss(nn.Module):
    """
    Extended LTI spectral radius constraint loss.

    Simultaneously constrains:
    1. Recurrent residual flow: ||x_{t+1}|| / ||x_t|| <= rho_target_recurrent
    2. DeltaNet hidden state: max decay rate <= rho_target_delta,
       and/or state norm <= delta_state_norm_target

    The loss is zero when constraints are satisfied, quadratic penalty when violated.
    Warmup period: no constraint applied for first `warmup_steps` training steps.
    """

    def __init__(
        self,
        rho_target_recurrent: float = 0.99,
        rho_target_delta: float = 0.95,
        weight_recurrent: float = 1.0,
        weight_delta: float = 0.5,
        warmup_steps: int = 1000,
        delta_state_norm_target: float = 1.0,
    ):
        super().__init__()
        self.rho_target_recurrent = rho_target_recurrent
        self.rho_target_delta = rho_target_delta
        self.weight_recurrent = weight_recurrent
        self.weight_delta = weight_delta
        self.warmup_steps = warmup_steps
        self.delta_state_norm_target = delta_state_norm_target

    def forward(
        self,
        x_t: torch.Tensor,
        x_next: torch.Tensor,
        delta_state: Optional[torch.Tensor] = None,
        decay_rates: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute extended LTI loss.

        Args:
            x_t: current recurrent flow [batch, seq, hidden]
            x_next: next recurrent flow [batch, seq, hidden]
            delta_state: DeltaNet hidden state [batch, heads, v_dim, v_dim]
            decay_rates: DeltaNet decay rates [batch, seq, num_heads] (optional)
            step: current training step (for warmup)

        Returns:
            total_loss: scalar LTI loss
            metrics: dict of monitoring values
        """
        metrics = {"step": step}

        # Warmup: no constraint
        if step < self.warmup_steps:
            metrics["warmup"] = True
            return torch.tensor(0.0, device=x_t.device), metrics

        metrics["warmup"] = False

        # ===== 1. Recurrent flow spectral radius constraint =====
        lti_recurrent, rho_approx = self._recurrent_flow_loss(x_t, x_next)
        metrics["lti_loss_recurrent"] = lti_recurrent.item()
        metrics["rho_recurrent_approx"] = rho_approx

        # ===== 2. DeltaNet state constraint =====
        lti_delta = torch.tensor(0.0, device=x_t.device)
        if delta_state is not None:
            lti_delta, delta_metrics = self._delta_state_loss(
                delta_state, decay_rates
            )
            metrics.update(delta_metrics)

        metrics["lti_loss_delta"] = lti_delta.item()

        # ===== 3. Weighted sum =====
        total_loss = (
            self.weight_recurrent * lti_recurrent
            + self.weight_delta * lti_delta
        )
        metrics["lti_loss_total"] = total_loss.item()

        return total_loss, metrics

    def _recurrent_flow_loss(
        self, x_t: torch.Tensor, x_next: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute recurrent flow LTI loss.

        Approximates spectral radius of state transition as:
          rho ≈ ||x_next|| / ||x_t||

        More precise implementation would compute the Jacobian's spectral radius
        via power iteration, but norm ratio is a good proxy for stability.

        Returns:
            loss: scalar penalty
            rho_approx: approximate spectral radius value
        """
        x_t_norm = torch.norm(x_t, dim=-1).mean()
        x_next_norm = torch.norm(x_next, dim=-1).mean()

        rho_approx = (x_next_norm / (x_t_norm + 1e-8)).item()

        # Quadratic penalty for exceeding target
        loss = F.relu(
            x_next_norm / (x_t_norm + 1e-8) - self.rho_target_recurrent
        ).pow(2)

        return loss, rho_approx

    def _delta_state_loss(
        self,
        delta_state: torch.Tensor,
        decay_rates: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute DeltaNet state LTI loss.

        Two complementary constraints:
        1. Decay rate upper bound: max(decay) <= rho_target_delta
           (decay is the state retention factor, must be < 1 for stability)
        2. State norm bound: ||delta_state|| <= delta_state_norm_target
           (backup constraint if decay rates unavailable)

        Returns:
            loss: scalar penalty
            metrics: monitoring dict
        """
        metrics = {}
        loss = torch.tensor(0.0, device=delta_state.device)

        # Constraint 1: decay rate upper bound
        if decay_rates is not None:
            beta_max = decay_rates.max()
            decay_loss = F.relu(beta_max - self.rho_target_delta).pow(2)
            loss = loss + decay_loss
            metrics["delta_decay_max"] = beta_max.item()
            metrics["lti_loss_delta_decay"] = decay_loss.item()

        # Constraint 2: state norm (always applied as backup)
        delta_norm = torch.norm(delta_state, dim=(-2, -1)).mean()
        norm_loss = F.relu(
            delta_norm - self.delta_state_norm_target
        ).pow(2)
        loss = loss + norm_loss
        metrics["delta_state_norm"] = delta_norm.item()
        metrics["lti_loss_delta_norm"] = norm_loss.item()

        return loss, metrics
