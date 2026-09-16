"""
Loop MoE + agent_harness integration smoke test.

Verifies:
1. Loop MoE model can be imported and instantiated
2. LoRA can be applied
3. Qwen36 inference client can be created
4. LoopMoEAgent adapter can be instantiated
5. Weight grafting config can be created
6. Training config can be created
"""

import os
import sys

# Add agent_harness to path (3 dirs up from this test file)
# test file: agent_harness/loopmoe/tests/test_integration.py
_agent_harness = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _agent_harness)

# Remove any pre-imported loopmoe (from installed package) to force local import
for _mod in list(sys.modules.keys()):
    if _mod.startswith("loopmoe") or _mod.startswith("qwen36"):
        del sys.modules[_mod]


def test_loopmoe_import():
    """Test that all loopmoe modules can be imported."""
    from loopmoe.config import LoopMoEConfig
    from loopmoe.models.gated_deltanet import GatedDeltaNet
    from loopmoe.models.qwen_moe import QwenMoE
    from loopmoe.models.recurrent_block import LoopMoERecurrentBlock
    from loopmoe.models.loop_moe_model import LoopMoEModel
    from loopmoe.states.dual_state import RecurrentDualState
    from loopmoe.lti.extended_lti import ExtendedLTILoss
    from loopmoe.training.lora import LoopMoELoRAConfig, apply_lora
    from loopmoe.training.weight_graft import WeightGraftConfig
    from loopmoe.training.train_loopmoe import LoopMoETrainer, TrainingConfig
    print("  [PASS] All loopmoe modules imported")


def test_loopmoe_config():
    """Test LoopMoEConfig creation and validation."""
    from loopmoe.config import LoopMoEConfig

    config = LoopMoEConfig(
        hidden_size=256,
        num_experts=4,
        num_experts_per_tok=2,
        max_recurrent_steps=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
    )
    issues = config.validate()
    assert len(issues) == 0, f"Config validation failed: {issues}"
    assert config.hidden_size == 256
    assert config.num_experts == 4
    print(f"  [PASS] LoopMoEConfig created (hidden={config.hidden_size}, experts={config.num_experts})")


def test_loopmoe_model_forward():
    """Test LoopMoEModel forward pass with small config."""
    import torch
    from loopmoe.config import LoopMoEConfig
    from loopmoe.models.loop_moe_model import LoopMoEModel

    config = LoopMoEConfig(
        hidden_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        max_recurrent_steps=2,
        min_recurrent_steps=1,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        moe_intermediate_size=64,
        num_prelude_layers=1,
        num_coda_layers=1,
        vocab_size=1000,
        max_position_embeddings=512,
    )

    model = LoopMoEModel(config)
    input_ids = torch.randint(0, 1000, (1, 16))
    outputs = model(input_ids=input_ids)

    assert "logits" in outputs
    assert outputs["logits"].shape == (1, 16, 1000)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [PASS] LoopMoEModel forward pass (params={n_params:,}, logits shape={outputs['logits'].shape})")


def test_lora_application():
    """Test LoRA application to Loop MoE model."""
    import torch
    from loopmoe.config import LoopMoEConfig
    from loopmoe.models.loop_moe_model import LoopMoEModel
    from loopmoe.training.lora import LoopMoELoRAConfig, apply_lora

    config = LoopMoEConfig(
        hidden_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        max_recurrent_steps=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        moe_intermediate_size=64,
        num_prelude_layers=1,
        num_coda_layers=1,
        vocab_size=1000,
    )

    model = LoopMoEModel(config)
    total_params = sum(p.numel() for p in model.parameters())

    lora_config = LoopMoELoRAConfig(lora_rank=8, lora_alpha=16)
    model = apply_lora(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params < total_params, "LoRA should reduce trainable params"
    assert trainable_params > 0, "LoRA should have some trainable params"

    # Forward pass should still work
    input_ids = torch.randint(0, 1000, (1, 8))
    outputs = model(input_ids=input_ids)
    assert outputs["logits"].shape == (1, 8, 1000)

    print(f"  [PASS] LoRA applied (trainable={trainable_params:,}/{total_params:,} = {100*trainable_params/total_params:.2f}%)")


def test_qwen36_inference_config():
    """Test Qwen36 inference client creation (without actual API call)."""
    from qwen36 import Qwen36Config, Qwen36Inference

    config = Qwen36Config(
        backend="cgc",
        base_url="http://127.0.0.1:1234/v1",
        model_name="qwen3.6-35b-a3b",
    )
    infer = Qwen36Inference(config)
    assert infer.config.backend == "cgc"
    assert infer.config.model_name == "qwen3.6-35b-a3b"
    print(f"  [PASS] Qwen36Inference created (backend={config.backend}, model={config.model_name})")


def test_training_config():
    """Test training config creation."""
    from loopmoe.training.train_loopmoe import TrainingConfig

    config = TrainingConfig(
        max_steps=100,
        batch_size=1,
        learning_rate=1e-4,
        output_dir="/tmp/loopmoe_test",
    )
    assert config.max_steps == 100
    assert config.learning_rate == 1e-4
    print(f"  [PASS] TrainingConfig created (max_steps={config.max_steps}, lr={config.learning_rate})")


def test_weight_graft_config():
    """Test weight grafting config creation."""
    from loopmoe.training.weight_graft import WeightGraftConfig

    config = WeightGraftConfig(
        source_model_path="/tmp/qwen36",
        source_model_type="hf",
        deltanet_source_layer=4,
        moe_source_layer=4,
    )
    assert config.deltanet_source_layer == 4
    assert config.moe_source_layer == 4
    print(f"  [PASS] WeightGraftConfig created (source={config.source_model_path})")


def main():
    print("=" * 60)
    print("Loop MoE + agent_harness Integration Smoke Test")
    print("=" * 60)

    tests = [
        ("Import all modules", test_loopmoe_import),
        ("Config creation", test_loopmoe_config),
        ("Model forward pass", test_loopmoe_model_forward),
        ("LoRA application", test_lora_application),
        ("Qwen36 inference config", test_qwen36_inference_config),
        ("Training config", test_training_config),
        ("Weight graft config", test_weight_graft_config),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[{name}]")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
