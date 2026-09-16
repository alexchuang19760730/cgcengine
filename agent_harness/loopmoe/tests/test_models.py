"""
Unit tests for GatedDeltaNet and QwenMoE modules.
"""

import pytest
import torch

from loopmoe.config import LoopMoEConfig
from loopmoe.models.gated_deltanet import GatedDeltaNet
from loopmoe.models.qwen_moe import QwenMoE


def make_small_config():
    """Create a small config for testing."""
    return LoopMoEConfig(
        hidden_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=2,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=32,
    )


class TestGatedDeltaNet:
    """Tests for GatedDeltaNet module."""

    def test_creation(self):
        """Test that GatedDeltaNet can be created."""
        config = make_small_config()
        model = GatedDeltaNet(
            hidden_size=config.hidden_size,
            num_key_heads=config.linear_num_key_heads,
            num_value_heads=config.linear_num_value_heads,
            key_head_dim=config.linear_key_head_dim,
            value_head_dim=config.linear_value_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
        )
        assert model is not None

    def test_forward_shape(self):
        """Test forward pass output shape."""
        config = make_small_config()
        model = GatedDeltaNet(
            hidden_size=config.hidden_size,
            num_key_heads=config.linear_num_key_heads,
            num_value_heads=config.linear_num_value_heads,
            key_head_dim=config.linear_key_head_dim,
            value_head_dim=config.linear_value_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
        )
        model.eval()

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, config.hidden_size)

        with torch.no_grad():
            output, state = model(x)

        assert output.shape == (batch_size, seq_len, config.hidden_size)
        assert state.shape == (
            batch_size,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
            config.linear_value_head_dim,
        )

    def test_state_persistence(self):
        """Test that state is correctly maintained across calls."""
        config = make_small_config()
        model = GatedDeltaNet(
            hidden_size=config.hidden_size,
            num_key_heads=config.linear_num_key_heads,
            num_value_heads=config.linear_num_value_heads,
            key_head_dim=config.linear_key_head_dim,
            value_head_dim=config.linear_value_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
        )
        model.eval()

        batch_size, seq_len = 1, 4
        x1 = torch.randn(batch_size, seq_len, config.hidden_size)
        x2 = torch.randn(batch_size, seq_len, config.hidden_size)

        with torch.no_grad():
            # First call: state initialized to zeros
            out1, state1 = model(x1)
            # Second call: use state from first call
            out2, state2 = model(x2, state=state1)

        # State should have changed after second call
        assert not torch.allclose(state1, state2)
        # Output shape should be correct
        assert out2.shape == (batch_size, seq_len, config.hidden_size)

    def test_init_state(self):
        """Test state initialization."""
        config = make_small_config()
        model = GatedDeltaNet(
            hidden_size=config.hidden_size,
            num_key_heads=config.linear_num_key_heads,
            num_value_heads=config.linear_num_value_heads,
            key_head_dim=config.linear_key_head_dim,
            value_head_dim=config.linear_value_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
        )
        state = model.init_state(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
        assert state.shape == (2, config.linear_num_value_heads, 16, 16)
        assert torch.all(state == 0)

    def test_gradient_flow(self):
        """Test that gradients flow through the module."""
        config = make_small_config()
        model = GatedDeltaNet(
            hidden_size=config.hidden_size,
            num_key_heads=config.linear_num_key_heads,
            num_value_heads=config.linear_num_value_heads,
            key_head_dim=config.linear_key_head_dim,
            value_head_dim=config.linear_value_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
        )
        model.train()

        x = torch.randn(1, 4, config.hidden_size, requires_grad=True)
        output, _ = model(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist for key parameters
        assert model.in_proj_qkvz.weight.grad is not None
        assert model.out_proj.weight.grad is not None


class TestQwenMoE:
    """Tests for QwenMoE module."""

    def test_creation(self):
        """Test that QwenMoE can be created."""
        moe = QwenMoE(
            hidden_size=64,
            intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
        )
        assert len(moe.experts) == 4
        assert len(moe.shared_experts) == 1

    def test_forward_shape(self):
        """Test forward pass output shape."""
        moe = QwenMoE(
            hidden_size=64,
            intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
        )
        moe.eval()

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 64)

        with torch.no_grad():
            output, router_loss = moe(x, training=False)

        assert output.shape == (batch_size, seq_len, 64)
        assert router_loss is None  # No loss in eval mode

    def test_forward_with_loss(self):
        """Test forward pass in training mode returns router loss."""
        moe = QwenMoE(
            hidden_size=64,
            intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
        )
        moe.train()

        x = torch.randn(2, 4, 64)
        output, router_loss = moe(x, training=True)

        assert output.shape == (2, 4, 64)
        assert router_loss is not None
        assert router_loss.item() >= 0

    def test_expert_utilization(self):
        """Test expert utilization metrics."""
        moe = QwenMoE(
            hidden_size=64,
            intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
        )
        moe.eval()

        x = torch.randn(4, 8, 64)
        router_logits = moe.gate(x.view(-1, 64))
        metrics = moe.get_expert_utilization(router_logits)

        assert "num_active_experts" in metrics
        assert "load_entropy" in metrics
        assert "top1_expert_share" in metrics
        assert metrics["num_active_experts"] <= 4

    def test_gradient_flow(self):
        """Test that gradients flow through MoE."""
        moe = QwenMoE(
            hidden_size=64,
            intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
        )
        moe.train()

        x = torch.randn(1, 4, 64, requires_grad=True)
        output, router_loss = moe(x, training=True)
        loss = output.sum() + (router_loss if router_loss is not None else 0)
        loss.backward()

        assert moe.gate.weight.grad is not None
        assert moe.experts[0].gate_proj.weight.grad is not None
