"""
Integration tests for LoopMoERecurrentBlock and LoopMoEModel.
"""

import pytest
import torch

from loopmoe.config import LoopMoEConfig
from loopmoe.models.recurrent_block import LoopMoERecurrentBlock
from loopmoe.models.loop_moe_model import LoopMoEModel


def make_integration_config():
    """Create a tiny config for integration testing."""
    return LoopMoEConfig(
        vocab_size=100,
        hidden_size=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        output_gate_type="silu",
        num_experts=2,
        num_experts_per_tok=1,
        num_shared_experts=1,
        moe_intermediate_size=16,
        num_prelude_layers=1,
        num_coda_layers=1,
        num_hidden_layers=4,
        max_recurrent_steps=2,
        min_recurrent_steps=1,
        gated_attn_num_heads=2,
        use_lti_constraint=True,
        lti_warmup_steps=0,  # No warmup for testing
    )


class TestLoopMoERecurrentBlock:
    """Integration tests for LoopMoERecurrentBlock."""

    def test_creation(self):
        """Test recurrent block creation."""
        config = make_integration_config()
        block = LoopMoERecurrentBlock(config, layer_id=0)
        assert block is not None
        assert block.self_attn is not None
        assert block.mlp is not None

    def test_single_step_forward(self):
        """Test single step forward pass."""
        config = make_integration_config()
        block = LoopMoERecurrentBlock(config, layer_id=0)
        block.eval()

        from loopmoe.states.dual_state import init_dual_state
        state = init_dual_state(
            batch_size=2,
            seq_len=4,
            hidden_size=config.hidden_size,
            num_value_heads=config.linear_num_value_heads,
            value_head_dim=config.linear_value_head_dim,
            device=torch.device("cpu"),
        )

        x = torch.randn(2, 4, config.hidden_size)
        with torch.no_grad():
            x_next, new_state, router_loss, halting_prob = block.forward_single_step(
                x, state, training=False
            )

        assert x_next.shape == x.shape
        assert new_state.delta_state is not None
        assert halting_prob.shape == (2, 1)
        assert 0 <= halting_prob.mean().item() <= 1

    def test_full_recurrent_forward(self):
        """Test full recurrent loop with ACT."""
        config = make_integration_config()
        block = LoopMoERecurrentBlock(config, layer_id=0)
        block.eval()

        x = torch.randn(2, 4, config.hidden_size)
        with torch.no_grad():
            output, state, metrics = block(x, training=False)

        assert output.shape == x.shape
        assert metrics["num_iterations"] >= 1
        assert metrics["num_iterations"] <= config.max_recurrent_steps
        assert "delta_state_norm" in metrics
        assert state.delta_state is not None

    def test_recurrent_gradient_flow(self):
        """Test gradient flow through recurrent block."""
        config = make_integration_config()
        block = LoopMoERecurrentBlock(config, layer_id=0)
        block.train()

        x = torch.randn(1, 4, config.hidden_size, requires_grad=True)
        output, state, metrics = block(x, training=True)
        loss = output.sum()
        loss.backward()

        assert block.self_attn.in_proj_qkvz.weight.grad is not None
        assert block.mlp.gate.weight.grad is not None
        assert block.halting_proj.weight.grad is not None


class TestLoopMoEModel:
    """Integration tests for full LoopMoEModel."""

    def test_creation(self):
        """Test full model creation."""
        config = make_integration_config()
        model = LoopMoEModel(config)
        assert model is not None
        assert len(model.prelude_layers) == 1
        assert len(model.coda_layers) == 1
        assert model.recurrent_block is not None

    def test_forward_shape(self):
        """Test full model forward pass output shape."""
        config = make_integration_config()
        model = LoopMoEModel(config)
        model.eval()

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
        assert outputs["loss"] is None  # No labels
        assert outputs["recurrent_state"] is not None
        assert "num_iterations" in outputs["metrics"]

    def test_forward_with_loss(self):
        """Test forward pass with labels computes loss."""
        config = make_integration_config()
        model = LoopMoEModel(config)
        model.train()

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels)
        assert outputs["loss"] is not None
        assert outputs["loss"].item() > 0
        assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)

    def test_gradient_flow_full_model(self):
        """Test gradient flow through full model."""
        config = make_integration_config()
        model = LoopMoEModel(config)
        model.train()

        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels)
        outputs["loss"].backward()

        # Check gradients in key components
        assert model.embed_tokens.weight.grad is not None
        assert model.recurrent_block.self_attn.in_proj_qkvz.weight.grad is not None
        assert model.recurrent_block.mlp.gate.weight.grad is not None
        assert model.lm_head.weight.grad is not None

    def test_count_parameters(self):
        """Test parameter counting."""
        config = make_integration_config()
        model = LoopMoEModel(config)
        param_info = model.count_parameters()

        assert "total" in param_info
        assert "trainable" in param_info
        assert "components" in param_info
        assert param_info["total"] > 0
        assert "recurrent_block" in param_info["components"]

    def test_deterministic_eval(self):
        """Test that eval mode produces deterministic outputs."""
        config = make_integration_config()
        model = LoopMoEModel(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 4))

        with torch.no_grad():
            out1 = model(input_ids)["logits"]
            out2 = model(input_ids)["logits"]

        assert torch.allclose(out1, out2, atol=1e-5)
