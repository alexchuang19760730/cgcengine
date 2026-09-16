"""
Qwen MoE: Pure PyTorch reference implementation of Qwen3.5/3.6 MoE
(256 experts, top-8 routing + 1 shared expert).

Architecturally equivalent to the SGLang version in:
  D:\\flashkv0516\\backend\\cloud_sglang\\python\\sglang\\srt\\models\\qwen2_moe.py
  (Qwen2MoeSparseMoeBlock, used by Qwen3.5 MoE)

This reference version uses standard PyTorch ops for Phase 0 verification.
Production versions use fused MoE kernels (Triton/CUDA) for efficiency.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QwenExpertMLP(nn.Module):
    """
    Single expert FFN: SwiGLU (gate_proj + up_proj + down_proj).

    output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenMoE(nn.Module):
    """
    Qwen3.5/3.6 Mixture-of-Experts block.

    Configuration:
    - num_experts: 256 routed experts
    - num_experts_per_tok: top-8 (k=8)
    - num_shared_experts: 1 (always active, no routing)

    Router:
    - Computes routing scores for all 256 experts
    - Selects top-8 with softmax-normalized weights
    - Shared expert always added (no router weight)

    Losses:
    - router_loss: load balancing (auxiliary loss)
    - z_loss: router logit stabilization
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = 1408,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        num_shared_experts: int = 1,
        norm_topk_prob: bool = True,
        router_loss_coef: float = 0.01,
        aux_loss_coef: float = 0.001,
        z_loss_coef: float = 0.001,
        expert_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.norm_topk_prob = norm_topk_prob
        self.router_loss_coef = router_loss_coef
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef
        self.expert_dropout = expert_dropout

        # ===== Router (gate) =====
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # ===== Routed experts (256 experts) =====
        self.experts = nn.ModuleList(
            [QwenExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

        # ===== Shared experts (always active) =====
        self.shared_experts = nn.ModuleList(
            [QwenExpertMLP(hidden_size, intermediate_size) for _ in range(num_shared_experts)]
        )

    def _compute_router_losses(
        self,
        router_logits: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute router auxiliary losses.

        Args:
            router_logits: [num_tokens, num_experts] (pre-softmax)
            topk_indices: [num_tokens, k] selected expert indices

        Returns:
            aux_loss: load balancing loss
            z_loss: router logit stabilization loss
        """
        num_tokens = router_logits.shape[0]

        # ===== Load balancing loss (auxiliary) =====
        # Encourage uniform expert selection across tokens
        # P(expert j selected) ≈ fraction of tokens selecting j
        # Loss = num_experts * sum_j(P_j * f_j)
        router_probs = F.softmax(router_logits, dim=-1)  # [num_tokens, num_experts]

        # Expert selection mask: [num_tokens, num_experts]
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).sum(dim=1)
        expert_mask = expert_mask.float()  # [num_tokens, num_experts]

        # Mean probability per expert (across tokens)
        mean_prob = router_probs.mean(dim=0)  # [num_experts]
        # Mean selection frequency per expert
        mean_freq = expert_mask.mean(dim=0)  # [num_experts]

        # Load balancing loss
        aux_loss = self.num_experts * torch.sum(mean_prob * mean_freq)

        # ===== Z-loss (stabilize router logits) =====
        # Penalizes large logit magnitudes, prevents router overconfidence
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

        return aux_loss, z_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        MoE forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            training: whether in training mode (enables dropout + losses)

        Returns:
            output: [batch, seq_len, hidden_size]
            router_loss: scalar tensor (aux + z losses), None in eval mode
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        # Flatten to [num_tokens, hidden_size] for efficient expert routing
        x = hidden_states.view(-1, hidden_size)
        num_tokens = x.shape[0]

        # ===== Router: compute scores & select top-k =====
        router_logits = self.gate(x)  # [num_tokens, num_experts]

        # Expert dropout during training
        if training and self.expert_dropout > 0:
            dropout_mask = torch.rand(self.num_experts, device=x.device) > self.expert_dropout
            router_logits = router_logits.masked_fill(~dropout_mask, float("-inf"))

        # Top-k selection
        topk_weights, topk_indices = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)

        # Normalize top-k weights (softmax over selected experts only)
        if self.norm_topk_prob:
            topk_weights = F.softmax(topk_weights, dim=-1)
        else:
            topk_weights = topk_weights.softmax(dim=-1)  # same thing

        # ===== Compute router losses (training only) =====
        router_loss = None
        if training:
            aux_loss, z_loss = self._compute_router_losses(router_logits, topk_indices)
            router_loss = (
                self.aux_loss_coef * aux_loss
                + self.z_loss_coef * z_loss
            ) * self.router_loss_coef

        # ===== Expert computation =====
        # Efficient approach: for each expert, gather tokens assigned to it,
        # compute expert output, then scatter back with weights.
        final_output = torch.zeros_like(x)

        # Flatten topk for token-expert assignment
        # topk_indices: [num_tokens, k] -> [num_tokens * k]
        flat_indices = topk_indices.view(-1)
        flat_weights = topk_weights.view(-1)  # [num_tokens * k]

        # Token indices for each (token, expert) pair
        token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(-1, self.num_experts_per_tok).reshape(-1)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (flat_indices == expert_idx)
            if not expert_mask.any():
                continue

            # Gather input tokens for this expert
            expert_token_indices = token_indices[expert_mask]
            expert_input = x[expert_token_indices]  # [num_assigned, hidden_size]
            expert_weight = flat_weights[expert_mask].unsqueeze(-1)  # [num_assigned, 1]

            # Compute expert output
            expert_output = self.experts[expert_idx](expert_input)

            # Scatter back with routing weights
            final_output.index_add_(0, expert_token_indices, expert_output * expert_weight)

        # ===== Shared experts (always active) =====
        shared_output = torch.zeros_like(x)
        for shared_expert in self.shared_experts:
            shared_output = shared_output + shared_expert(x)
        shared_output = shared_output / self.num_shared_experts  # average if >1

        final_output = final_output + shared_output

        # Reshape back to [batch, seq_len, hidden_size]
        final_output = final_output.view(batch_size, seq_len, hidden_size)

        return final_output, router_loss

    def get_expert_utilization(self, router_logits: torch.Tensor) -> dict:
        """
        Compute expert utilization metrics for monitoring.

        Args:
            router_logits: [num_tokens, num_experts]

        Returns:
            dict with utilization metrics
        """
        topk_indices = torch.topk(router_logits, self.num_experts_per_tok, dim=-1).indices
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).sum(dim=1)
        expert_counts = expert_mask.sum(dim=0)  # [num_experts]

        total_assignments = expert_counts.sum()
        utilization = expert_counts / (total_assignments + 1e-8)

        return {
            "expert_counts": expert_counts,
            "expert_utilization": utilization,
            "num_active_experts": (expert_counts > 0).sum().item(),
            "top1_expert_share": (expert_counts.max() / (total_assignments + 1e-8)).item(),
            "load_entropy": -(utilization * (utilization + 1e-8).log()).sum().item(),
            "max_possible_entropy": torch.log(torch.tensor(float(self.num_experts))).item(),
        }
