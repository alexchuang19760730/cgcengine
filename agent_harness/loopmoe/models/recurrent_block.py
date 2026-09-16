"""
Loop MoE RecurrentBlock: The core RDT (Recurrent Deep Transformer) block.

Wraps Gated DeltaNet (attention) + Qwen MoE (FFN) in a recurrent loop with
ACT (Adaptive Computation Time) for dynamic iteration depth.

This replaces OpenMythos's native RecurrentBlock (which used MLA/GQA +
DeepSeekMoE) with Qwen3.6's Gated DeltaNet + Qwen MoE, while preserving
the RDT recurrent skeleton, ACT mechanism, and dual-state management.

Architecture per iteration:
  x_t -> RMSNorm -> GatedDeltaNet -> residual -> RMSNorm -> QwenMoE -> residual -> x_{t+1}
  with ACT halting probability computed from x_{t+1}
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LoopMoEConfig
from ..states.dual_state import RecurrentDualState, clip_delta_state
from .gated_deltanet import GatedDeltaNet
from .qwen_moe import QwenMoE


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class LoopMoERecurrentBlock(nn.Module):
    """
    Recurrent block for Loop MoE.

    Contains:
    - Gated DeltaNet (attention sub-layer, replaces MLA/GQA)
    - Qwen MoE (FFN sub-layer, replaces DeepSeekMoE)
    - RMSNorm for both sub-layers
    - ACT halting projection
    - Dual state management (KV + Delta)

    The block is designed to be called iteratively in a loop, maintaining
    state across iterations.
    """

    def __init__(self, config: LoopMoEConfig, layer_id: int = 0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        # ===== Attention: Gated DeltaNet =====
        self.self_attn = GatedDeltaNet(
            hidden_size=config.hidden_size,
            num_key_heads=config.linear_num_key_heads,
            num_value_heads=config.linear_num_value_heads,
            key_head_dim=config.linear_key_head_dim,
            value_head_dim=config.linear_value_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
            rms_norm_eps=config.rms_norm_eps,
            output_gate_type=config.output_gate_type or "silu",
            delta_decay_max=config.delta_decay_max,
            layer_id=layer_id,
        )

        # ===== FFN: Qwen MoE =====
        self.mlp = QwenMoE(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            num_shared_experts=config.num_shared_experts,
            norm_topk_prob=config.norm_topk_prob,
            router_loss_coef=config.moe_router_loss_coef,
            aux_loss_coef=config.moe_aux_loss_coef,
            z_loss_coef=config.moe_z_loss_coef,
            expert_dropout=config.moe_expert_dropout,
        )

        # ===== Layer norms =====
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ===== ACT halting projection =====
        self.halting_proj = nn.Linear(config.hidden_size, 1, bias=True)
        nn.init.zeros_(self.halting_proj.weight)
        nn.init.constant_(self.halting_proj.bias, 0.0)

    def forward_single_step(
        self,
        x: torch.Tensor,
        state: RecurrentDualState,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, RecurrentDualState, Optional[torch.Tensor], torch.Tensor]:
        """
        Execute a single recurrent iteration step.

        Args:
            x: input hidden states [batch, seq_len, hidden_size]
            state: current recurrent dual state
            attention_mask: optional attention mask
            training: whether in training mode

        Returns:
            x_next: output hidden states [batch, seq_len, hidden_size]
            new_state: updated recurrent dual state
            router_loss: MoE router loss (or None)
            halting_prob: ACT halting probability [batch, 1]
        """
        batch_size, seq_len, _ = x.shape

        # ===== Attention sub-layer: Gated DeltaNet =====
        residual = x
        x_norm = self.input_layernorm(x)

        attn_out, new_delta_state = self.self_attn(
            x_norm,
            state=state.delta_state,
            attention_mask=attention_mask,
        )
        x = residual + attn_out

        # ===== FFN sub-layer: Qwen MoE =====
        residual = x
        x_norm = self.post_attention_layernorm(x)
        moe_out, router_loss = self.mlp(x_norm, training=training)
        x = residual + moe_out

        # ===== Delta state stability: clip if needed =====
        if self.config.use_lti_constraint:
            new_delta_state = clip_delta_state(
                new_delta_state,
                max_norm=self.config.lti_delta_state_max_norm,
            )

        # ===== ACT halting probability =====
        # Use mean pooling over sequence for halting signal
        halting_input = x.mean(dim=1)  # [batch, hidden]
        halting_logit = self.halting_proj(halting_input)  # [batch, 1]
        halting_prob = torch.sigmoid(
            halting_logit / self.config.act_temperature
        )  # [batch, 1]

        # ===== Update state =====
        new_state = RecurrentDualState(
            kv_state=state.kv_state,  # KV unchanged (DeltaNet-only attention)
            delta_state=new_delta_state,
            recurrent_flow=x,
            act_cumulative_prob=state.act_cumulative_prob,
            step_count=state.step_count + 1,
            outputs_history=list(state.outputs_history),
            halting_probs_history=list(state.halting_probs_history),
        )

        return x, new_state, router_loss, halting_prob

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[RecurrentDualState] = None,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True,
        max_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, RecurrentDualState, dict]:
        """
        Execute the full recurrent loop with ACT.

        Args:
            x: input hidden states [batch, seq_len, hidden_size]
            initial_state: initial recurrent state (or None to initialize)
            attention_mask: optional attention mask
            training: whether in training mode
            max_steps: override max recurrent steps

        Returns:
            output: ACT-weighted average output [batch, seq_len, hidden_size]
            final_state: final recurrent dual state
            metrics: dict of iteration metrics
        """
        batch_size, seq_len, _ = x.shape
        max_steps = max_steps or self.config.max_recurrent_steps
        min_steps = self.config.min_recurrent_steps

        # Initialize state
        if initial_state is None:
            from ..states.dual_state import init_dual_state
            state = init_dual_state(
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=self.hidden_size,
                num_value_heads=self.config.linear_num_value_heads,
                value_head_dim=self.config.linear_value_head_dim,
                use_kv_state=self.config.use_gated_attention_in_loop,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            state = initial_state

        # Set initial recurrent flow
        state.recurrent_flow = x

        # ===== Recurrent loop =====
        current_x = x
        total_router_loss = torch.tensor(0.0, device=x.device)
        all_halting_probs = []

        for step in range(max_steps):
            # Single step forward
            current_x, state, router_loss, halting_prob = self.forward_single_step(
                current_x, state, attention_mask, training
            )

            # Accumulate router loss
            if router_loss is not None:
                total_router_loss = total_router_loss + router_loss

            # Record output and halting prob (store tensor for gradient flow)
            state.outputs_history.append(current_x)
            state.halting_probs_history.append(halting_prob)
            all_halting_probs.append(halting_prob)

            # Update ACT cumulative probability (use batch mean)
            state.act_cumulative_prob += halting_prob.mean().item()

            # Check stopping condition (after min_steps)
            if step >= min_steps - 1 and state.should_stop(self.config.act_epsilon):
                break

        # ===== ACT-weighted average output =====
        output = state.get_act_weighted_output()
        if output is None:
            output = current_x

        # ===== Metrics =====
        mean_hp = 0.0
        if state.halting_probs_history:
            hp_tensors = [p.mean() if isinstance(p, torch.Tensor) else torch.tensor(p)
                          for p in state.halting_probs_history]
            mean_hp = (sum(hp_tensors) / len(hp_tensors)).item()

        metrics = {
            "num_iterations": len(state.outputs_history),
            "mean_halting_prob": mean_hp,
            "final_cumulative_prob": state.act_cumulative_prob,
            "total_router_loss": total_router_loss.item(),
            "delta_state_norm": (
                torch.norm(state.delta_state, dim=(-2, -1)).mean().item()
                if state.delta_state is not None else 0.0
            ),
        }

        return output, state, metrics
