"""
Unit tests for RecurrentDualState and dual state management.
"""

import pytest
import torch

from loopmoe.states.dual_state import (
    RecurrentDualState,
    init_dual_state,
    clip_delta_state,
)


class TestRecurrentDualState:
    """Tests for RecurrentDualState container."""

    def test_creation(self):
        """Test default state creation."""
        state = RecurrentDualState()
        assert state.kv_state is None
        assert state.delta_state is None
        assert state.recurrent_flow is None
        assert state.step_count == 0
        assert state.act_cumulative_prob == 0.0

    def test_clone(self):
        """Test deep clone."""
        state = RecurrentDualState(
            delta_state=torch.randn(2, 4, 8, 8),
            step_count=3,
            act_cumulative_prob=0.5,
        )
        cloned = state.clone()

        assert cloned.step_count == 3
        assert cloned.act_cumulative_prob == 0.5
        assert torch.allclose(cloned.delta_state, state.delta_state)

        # Modify clone should not affect original
        cloned.delta_state[0, 0, 0, 0] = 999.0
        assert state.delta_state[0, 0, 0, 0] != 999.0

    def test_should_stop(self):
        """Test ACT stopping criterion."""
        state = RecurrentDualState(act_cumulative_prob=0.5)
        assert not state.should_stop()

        state.act_cumulative_prob = 0.999
        assert state.should_stop(epsilon=1e-3)

        state.act_cumulative_prob = 1.0
        assert state.should_stop()

    def test_act_weighted_output_single(self):
        """Test ACT weighted output with single iteration."""
        out = torch.randn(2, 4, 8)
        state = RecurrentDualState(
            outputs_history=[out],
            halting_probs_history=[0.5],
        )
        result = state.get_act_weighted_output()
        assert torch.allclose(result, out)

    def test_act_weighted_output_multiple(self):
        """Test ACT weighted output with multiple iterations."""
        out1 = torch.ones(1, 2, 4)
        out2 = torch.full((1, 2, 4), 3.0)
        state = RecurrentDualState(
            outputs_history=[out1, out2],
            halting_probs_history=[0.4, 0.6],
        )
        result = state.get_act_weighted_output()
        # First gets 0.4, second gets remaining 0.6
        expected = 0.4 * out1 + 0.6 * out2
        assert torch.allclose(result, expected, atol=1e-5)

    def test_get_metrics(self):
        """Test metrics extraction."""
        state = RecurrentDualState(
            delta_state=torch.randn(2, 4, 8, 8),
            step_count=5,
            act_cumulative_prob=0.8,
        )
        metrics = state.get_metrics()
        assert metrics["step_count"] == 5
        assert metrics["act_cumulative_prob"] == 0.8
        assert "delta_state_norm" in metrics
        assert "delta_state_max" in metrics


class TestInitDualState:
    """Tests for init_dual_state function."""

    def test_init_delta_only(self):
        """Test initialization with Delta state only."""
        state = init_dual_state(
            batch_size=2,
            seq_len=8,
            hidden_size=64,
            num_value_heads=4,
            value_head_dim=16,
            use_kv_state=False,
        )
        assert state.delta_state.shape == (2, 4, 16, 16)
        assert torch.all(state.delta_state == 0)
        assert state.kv_state is None
        assert state.step_count == 0

    def test_init_with_kv(self):
        """Test initialization with KV state."""
        state = init_dual_state(
            batch_size=2,
            seq_len=8,
            hidden_size=64,
            num_value_heads=4,
            value_head_dim=16,
            use_kv_state=True,
            kv_num_heads=2,
            kv_head_dim=32,
        )
        assert state.kv_state is not None
        assert state.kv_state["key"].shape == (2, 2, 0, 32)
        assert state.kv_state["value"].shape == (2, 2, 0, 32)


class TestClipDeltaState:
    """Tests for clip_delta_state function."""

    def test_no_clipping_needed(self):
        """Test that small states are not clipped."""
        state = torch.randn(1, 2, 4, 4) * 0.1  # Very small
        clipped = clip_delta_state(state, max_norm=5.0)
        assert torch.allclose(clipped, state)

    def test_clipping_large_state(self):
        """Test that large states are clipped."""
        state = torch.randn(1, 2, 4, 4) * 100.0  # Very large
        clipped = clip_delta_state(state, max_norm=1.0)

        clipped_norm = torch.norm(clipped, dim=(-2, -1))
        assert (clipped_norm <= 1.0 + 1e-5).all()

    def test_clipping_preserves_direction(self):
        """Test that clipping preserves state direction."""
        state = torch.randn(1, 1, 4, 4)
        # Make it large in a specific direction
        state = state * 10.0
        clipped = clip_delta_state(state, max_norm=1.0)

        # Direction should be preserved (cosine similarity ~ 1)
        state_flat = state.flatten()
        clipped_flat = clipped.flatten()
        cos_sim = torch.dot(state_flat, clipped_flat) / (
            state_flat.norm() * clipped_flat.norm() + 1e-8
        )
        assert cos_sim > 0.99
