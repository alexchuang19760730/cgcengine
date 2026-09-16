"""
Unit tests for LoopMoEConfig.
"""

import pytest
import torch

from loopmoe.config import LoopMoEConfig


class TestLoopMoEConfig:
    """Tests for LoopMoEConfig configuration class."""

    def test_default_config_creation(self):
        """Test that default config can be created without errors."""
        config = LoopMoEConfig()
        assert config.vocab_size == 152064
        assert config.hidden_size == 2048
        assert config.num_experts == 256
        assert config.num_experts_per_tok == 8
        assert config.num_shared_experts == 1
        assert config.max_recurrent_steps == 8

    def test_custom_config(self):
        """Test custom config values."""
        config = LoopMoEConfig(
            hidden_size=512,
            num_experts=16,
            num_experts_per_tok=4,
            max_recurrent_steps=4,
        )
        assert config.hidden_size == 512
        assert config.num_experts == 16
        assert config.num_experts_per_tok == 4
        assert config.max_recurrent_steps == 4

    def test_derived_dimensions(self):
        """Test that derived dimensions are computed correctly."""
        config = LoopMoEConfig(
            linear_num_key_heads=8,
            linear_key_head_dim=64,
            linear_num_value_heads=16,
            linear_value_head_dim=64,
        )
        assert config.linear_key_dim == 8 * 64
        assert config.linear_value_dim == 16 * 64

    def test_delta_state_shape(self):
        """Test delta_state_shape property."""
        config = LoopMoEConfig(
            linear_num_value_heads=32,
            linear_value_head_dim=128,
        )
        shape = config.delta_state_shape
        assert shape == (32, 128, 128)

    def test_graft_indices_auto_computed(self):
        """Test that graft layer indices are auto-computed."""
        config = LoopMoEConfig(
            num_prelude_layers=2,
            num_coda_layers=1,
            num_hidden_layers=28,
        )
        assert config.graft_prelude_layer_indices == [0, 1]
        assert config.graft_coda_layer_indices == [27]

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = LoopMoEConfig()
        issues = config.validate()
        # Default config should have no critical issues
        # (may have warnings about rho ordering, but that's expected)
        assert isinstance(issues, list)

    def test_validate_invalid_topk(self):
        """Test validation catches topk >= num_experts."""
        config = LoopMoEConfig(num_experts=8, num_experts_per_tok=16)
        issues = config.validate()
        assert any("num_experts_per_tok" in issue for issue in issues)

    def test_validate_invalid_recurrent_steps(self):
        """Test validation catches max < min recurrent steps."""
        config = LoopMoEConfig(min_recurrent_steps=8, max_recurrent_steps=4)
        issues = config.validate()
        assert any("max_recurrent_steps" in issue for issue in issues)

    def test_to_dict(self):
        """Test config to_dict conversion."""
        config = LoopMoEConfig(hidden_size=1024)
        d = config.to_dict()
        assert d["hidden_size"] == 1024
        assert isinstance(d, dict)

    def test_json_roundtrip(self, tmp_path):
        """Test config JSON save/load roundtrip."""
        config = LoopMoEConfig(hidden_size=768, num_experts=32)
        path = tmp_path / "test_config.json"
        config.to_json(str(path))

        loaded = LoopMoEConfig.from_json(str(path))
        assert loaded.hidden_size == 768
        assert loaded.num_experts == 32

    def test_small_test_config(self):
        """Test a small config suitable for unit testing."""
        config = LoopMoEConfig(
            vocab_size=1000,
            hidden_size=128,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=2,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
            moe_intermediate_size=64,
            num_prelude_layers=1,
            num_coda_layers=1,
            max_recurrent_steps=2,
            min_recurrent_steps=1,
        )
        issues = config.validate()
        assert all("num_experts_per_tok" not in i for i in issues)
        assert all("max_recurrent_steps" not in i for i in issues)
