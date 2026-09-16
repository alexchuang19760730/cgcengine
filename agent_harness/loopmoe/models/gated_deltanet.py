"""
Gated DeltaNet: Pure PyTorch reference implementation of Qwen3.5/3.6
Gated DeltaNet (hybrid linear attention with decay gating + convolution +
delta rule state update).

This is a clean, dependency-light reference implementation architecturally
equivalent to the SGLang optimized version in:
  D:\\flashkv0516\\backend\\cloud_sglang\\python\\sglang\\srt\\models\\qwen3_5.py

The SGLang version uses custom Triton/CUDA kernels (RadixLinearAttention)
for production inference. This reference version uses standard PyTorch ops
for Phase 0 verification and can be swapped for the optimized version later.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormGated(nn.Module):
    """
    Gated RMSNorm: RMSNorm applied to attention output, gated by a separate
    z projection (like Qwen3.5's output gating).

    output = gate(z) * RMSNorm(x)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        activation: str = "silu",
        norm_before_gate: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.norm_before_gate = norm_before_gate

        if activation == "silu":
            self.act = F.silu
        elif activation == "gelu":
            self.act = F.gelu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            self.act = F.silu

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: attention output [*, hidden_size]
            z: gate projection [*, hidden_size]
        """
        if self.norm_before_gate:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            x = x * self.weight
            return self.act(z) * x
        else:
            x = self.act(z) * x
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * self.weight


class GatedDeltaNet(nn.Module):
    """
    Qwen3.5/3.6 Gated DeltaNet (hybrid linear attention).

    Architecture:
      input x
        -> in_proj_qkvz: split into Q, K, V, Z(gate)
        -> in_proj_ba: split into B(dt/decay), A
        -> conv1d on K, V (depthwise causal convolution)
        -> dt = softplus(B + dt_bias), A = -exp(A_log + A)
        -> decay = exp(-dt)  (per-head, data-dependent)
        -> state update (delta rule): state = decay * state + K^T * V
        -> output = Q * state
        -> RMSNormGated(output, Z)
        -> out_proj

    State shape: [batch, num_v_heads, head_v_dim, head_v_dim] (fixed, O(1) memory w.r.t. seq_len)
    """

    def __init__(
        self,
        hidden_size: int,
        num_key_heads: int = 16,
        num_value_heads: int = 32,
        key_head_dim: int = 128,
        value_head_dim: int = 128,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
        output_gate_type: str = "silu",
        delta_decay_max: float = 0.95,
        layer_id: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.key_dim = num_key_heads * key_head_dim
        self.value_dim = num_value_heads * value_head_dim
        self.conv_kernel_size = conv_kernel_size
        self.delta_decay_max = delta_decay_max
        self.layer_id = layer_id

        # ===== Input projections =====
        # Q, K, V, Z merged projection (Qwen uses packed checkpoint format)
        self.in_proj_qkvz = nn.Linear(
            hidden_size,
            self.key_dim + self.key_dim + self.value_dim + self.value_dim,
            bias=False,
        )
        # B (dt), A merged projection
        self.in_proj_ba = nn.Linear(
            hidden_size,
            num_value_heads + num_value_heads,
            bias=False,
        )

        # ===== Conv1d on K, V (causal depthwise) =====
        # Applied to concatenated [K, K2, V] for local enhancement
        conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=conv_kernel_size,
            groups=conv_dim,  # depthwise
            bias=True,
            padding=conv_kernel_size - 1,  # causal padding
        )

        # ===== State parameters (learnable per-head) =====
        self.dt_bias = nn.Parameter(torch.ones(num_value_heads))
        self.A_log = nn.Parameter(torch.empty(num_value_heads, dtype=torch.float32))
        nn.init.normal_(self.A_log, mean=0.0, std=0.02)

        # ===== Output gating & projection =====
        self.norm = RMSNormGated(
            self.value_dim,
            eps=rms_norm_eps,
            activation=output_gate_type,
            norm_before_gate=True,
        )
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _compute_decay(self, b: torch.Tensor) -> torch.Tensor:
        """
        Compute per-head decay rate from B projection + dt_bias + A_log.

        decay = exp(-softplus(b + dt_bias) * exp(A_log))
        Scaled to [0, delta_decay_max] for LTI stability.

        Args:
            b: [batch, seq_len, num_value_heads]
        Returns:
            decay: [batch, seq_len, num_value_heads, 1, 1] (broadcastable for state)
        """
        # dt (data-dependent time step)
        dt = F.softplus(b + self.dt_bias)  # [batch, seq, num_v_heads]

        # A (learnable per-head, always negative)
        A = -torch.exp(self.A_log.float())  # [num_v_heads]

        # decay = exp(dt * A), A < 0 so decay in (0, 1)
        decay = torch.exp(dt.float() * A)  # [batch, seq, num_v_heads]

        # Enforce upper bound for LTI stability
        decay = decay * self.delta_decay_max

        # Reshape for broadcasting with state [batch, heads, v_dim, v_dim]
        decay = decay.unsqueeze(-1).unsqueeze(-1)  # [batch, seq, heads, 1, 1]
        return decay

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Gated DeltaNet.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            state: [batch, num_value_heads, value_head_dim, value_head_dim]
                   or None (initialize to zeros)
            attention_mask: optional (not used in linear attention, kept for API compat)

        Returns:
            output: [batch, seq_len, hidden_size]
            new_state: [batch, num_value_heads, value_head_dim, value_head_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Initialize state if not provided
        if state is None:
            state = torch.zeros(
                batch_size,
                self.num_value_heads,
                self.value_head_dim,
                self.value_head_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        # ===== Input projections =====
        qkvz = self.in_proj_qkvz(hidden_states)  # [batch, seq, key_dim*2 + value_dim*2]
        ba = self.in_proj_ba(hidden_states)       # [batch, seq, num_v_heads*2]

        # Split Q, K, V, Z
        q, k, v, z = qkvz.split(
            [self.key_dim, self.key_dim, self.value_dim, self.value_dim],
            dim=-1,
        )
        # Split B (dt), A
        b, _a = ba.split([self.num_value_heads, self.num_value_heads], dim=-1)

        # ===== Conv1d on K, V (causal) =====
        # Concatenate K, K(duplicate for conv), V
        kv_conv = torch.cat([k, k, v], dim=-1)  # [batch, seq, key_dim*2 + value_dim]
        kv_conv = kv_conv.transpose(1, 2)  # [batch, channels, seq]
        kv_conv = self.conv1d(kv_conv)[:, :, :seq_len]  # causal: trim padding
        kv_conv = kv_conv.transpose(1, 2)  # [batch, seq, channels]

        k_conv, _, v_conv = kv_conv.split(
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        # ===== Reshape for multi-head =====
        # Q: [batch, seq, num_k_heads, key_head_dim]
        q = q.view(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        # K: [batch, seq, num_k_heads, key_head_dim]
        k = k_conv.view(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        # V: [batch, seq, num_v_heads, value_head_dim]
        v = v_conv.view(batch_size, seq_len, self.num_value_heads, self.value_head_dim)
        # Z: [batch, seq, num_v_heads, value_head_dim]
        z = z.view(batch_size, seq_len, self.num_value_heads, self.value_head_dim)

        # ===== Compute decay rates =====
        decay = self._compute_decay(b)  # [batch, seq, num_v_heads, 1, 1]

        # ===== Linear attention with delta rule state update =====
        # Handle grouped key/value heads: num_k_heads may differ from num_v_heads
        # Qwen3.5 uses num_k_heads < num_v_heads (grouped query for linear attn)
        num_kv_groups = self.num_value_heads // self.num_key_heads

        # Expand K to match V heads: [batch, seq, num_v_heads, key_head_dim]
        k_expanded = k.unsqueeze(3).expand(
            batch_size, seq_len, self.num_key_heads, num_kv_groups, self.key_head_dim
        ).reshape(batch_size, seq_len, self.num_value_heads, self.key_head_dim)

        # Expand Q similarly
        q_expanded = q.unsqueeze(3).expand(
            batch_size, seq_len, self.num_key_heads, num_kv_groups, self.key_head_dim
        ).reshape(batch_size, seq_len, self.num_value_heads, self.key_head_dim)

        # Sequential state update (delta rule)
        # state_t = decay_t * state_{t-1} + K_t^T * V_t
        # output_t = Q_t * state_t
        outputs = []
        current_state = state  # [batch, num_v_heads, v_dim, v_dim]

        for t in range(seq_len):
            # K_t: [batch, num_v_heads, key_dim] -> [batch, num_v_heads, key_dim, 1]
            k_t = k_expanded[:, t, :, :].unsqueeze(-1)
            # V_t: [batch, num_v_heads, v_dim] -> [batch, num_v_heads, 1, v_dim]
            v_t = v[:, t, :, :].unsqueeze(2)
            # Q_t: [batch, num_v_heads, key_dim]
            q_t = q_expanded[:, t, :, :]
            # decay_t: [batch, num_v_heads, 1, 1]
            d_t = decay[:, t, :, :, :]

            # Delta rule state update: state = decay * state + K^T * V
            # K^T * V: [batch, heads, key_dim, 1] * [batch, heads, 1, v_dim]
            #         = [batch, heads, key_dim, v_dim]
            kv_update = torch.matmul(k_t, v_t)
            current_state = d_t * current_state + kv_update

            # Output: Q * state = [batch, heads, key_dim] * [batch, heads, key_dim, v_dim]
            #                    = [batch, heads, v_dim]
            out_t = torch.matmul(q_t.unsqueeze(2), current_state).squeeze(2)
            outputs.append(out_t)

        # Stack outputs: [batch, seq, num_v_heads, v_dim]
        core_attn_out = torch.stack(outputs, dim=1)

        # ===== Gated RMSNorm =====
        # Flatten heads for norm
        core_attn_out_flat = core_attn_out.reshape(batch_size, seq_len, self.value_dim)
        z_flat = z.reshape(batch_size, seq_len, self.value_dim)
        gated_out = self.norm(core_attn_out_flat, z_flat)

        # ===== Output projection =====
        output = self.out_proj(gated_out)

        return output, current_state

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Initialize DeltaNet hidden state to zeros."""
        return torch.zeros(
            batch_size,
            self.num_value_heads,
            self.value_head_dim,
            self.value_head_dim,
            device=device,
            dtype=dtype,
        )
