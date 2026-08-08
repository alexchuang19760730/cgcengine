#!/usr/bin/env python3
"""Convert trained .pt MTP head checkpoint to sglang EAGLE HF format.

Takes our .pt checkpoint (DeepSeek-V3 MTP style) and converts it to a
HuggingFace-compatible directory that sglang can load as an EAGLE draft model.

Output directory structure:
  output_dir/
    config.json           # Qwen2-compatible config with CgcMtpForCausalLMEagle architecture
    model.safetensors     # Converted weights
    modeling_cgc_mtp_eagle.py  # Copy of the model definition (for trust_remote_code)

Usage:
  python3 convert_pt_to_eagle.py \
    --checkpoint /data/mtp_output/qwen3vl/mtp_head_qwen3vl_decode.pt \
    --output-dir /data/eagle_drafts/qwen3vl \
    --model-name qwen3vl

  # For DSV4:
  python3 convert_pt_to_eagle.py \
    --checkpoint /data/mtp_output/dsv4/mtp_head_dsv4_decode.pt \
    --output-dir /data/eagle_drafts/dsv4 \
    --model-name dsv4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file


# Weight name mapping: our .pt checkpoint -> sglang internal names
WEIGHT_NAME_MAP = {
    "proj.weight": "model.fc.weight",
    "norm1.weight": "model.layers.0.input_layernorm.weight",
    "norm2.weight": "model.layers.0.post_attention_layernorm.weight",
    "norm_out.weight": "model.norm_out.weight",
    # Attention
    "attn.q_proj.weight": "model.layers.0.self_attn.q_proj.weight",
    "attn.k_proj.weight": "model.layers.0.self_attn.k_proj.weight",
    "attn.v_proj.weight": "model.layers.0.self_attn.v_proj.weight",
    "attn.o_proj.weight": "model.layers.0.self_attn.o_proj.weight",
    # MLP
    "mlp.gate_proj.weight": "model.layers.0.mlp.gate_proj.weight",
    "mlp.up_proj.weight": "model.layers.0.mlp.up_proj.weight",
    "mlp.down_proj.weight": "model.layers.0.mlp.down_proj.weight",
}

# Model-specific configs (must match model_registry.py and training config)
MODEL_CONFIGS = {
    "qwen3vl": {
        "hidden_size": 2048,
        "vocab_size": 151936,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "intermediate_size": 5632,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 40960,
        "tie_word_embeddings": False,
    },
    "dsv4": {
        "hidden_size": 4096,
        "vocab_size": 129280,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 512,
        "intermediate_size": 11264,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "max_position_embeddings": 40960,
        "tie_word_embeddings": False,
    },
    "gemma4": {
        "hidden_size": 2816,
        "vocab_size": 262144,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "intermediate_size": 14336,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 40960,
        "tie_word_embeddings": False,
    },
}


def create_config_json(model_name: str) -> dict:
    """Create config.json for the EAGLE draft model."""
    cfg = MODEL_CONFIGS[model_name].copy()

    config = {
        "architectures": ["CgcMtpForCausalLMEagle"],
        "model_type": "qwen2",
        "num_hidden_layers": 1,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.45.0",
        "use_cache": True,
        "auto_map": {
            "AutoModelForCausalLM": "modeling_cgc_mtp_eagle.CgcMtpForCausalLMEagle",
        },
        # Qwen2 compatibility fields
        "hidden_act": "silu",
        "bos_token_id": cfg["vocab_size"] - 1,  # Qwen3 default
        "eos_token_id": cfg["vocab_size"] - 2,
        **cfg,
    }

    return config


def convert_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    model_name: str,
) -> None:
    """Convert .pt checkpoint to sglang EAGLE HF format."""

    # 1. Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Also check if config is embedded in checkpoint
    ckpt_config = ckpt.get("config", {})
    if isinstance(ckpt_config, str):
        import ast
        ckpt_config = ast.literal_eval(ckpt_config)

    print(f"  Checkpoint keys: {list(state_dict.keys())}")
    print(f"  Checkpoint config: {ckpt_config}")

    # 2. Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 3. Remap weights
    mapped_weights = {}
    skipped = []

    for name, tensor in state_dict.items():
        # Skip lm_head and embed (shared from target)
        if "lm_head" in name or "embed" in name:
            skipped.append(name)
            continue

        # Apply mapping
        if name in WEIGHT_NAME_MAP:
            mapped_name = WEIGHT_NAME_MAP[name]
        elif name.startswith("model."):
            mapped_name = name
        else:
            print(f"  WARNING: unmapped weight: {name}")
            mapped_name = name

        # Convert to bfloat16 for GPU efficiency
        mapped_weights[mapped_name] = tensor.to(torch.bfloat16).contiguous()

    print(f"\nMapped weights ({len(mapped_weights)}):")
    for name, tensor in sorted(mapped_weights.items()):
        print(f"  {name}: {tuple(tensor.shape)} {tensor.dtype}")

    if skipped:
        print(f"\nSkipped (shared from target): {skipped}")

    # 4. Save as safetensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(mapped_weights, safetensors_path)
    print(f"\nSaved: {safetensors_path}")

    # 5. Create config.json
    config = create_config_json(model_name)

    # Override with checkpoint config if available
    if ckpt_config:
        for key in ["hidden_size", "vocab_size", "num_heads", "head_dim", "intermediate_size"]:
            if key in ckpt_config:
                if key == "num_heads":
                    config["num_attention_heads"] = ckpt_config[key]
                    config["num_key_value_heads"] = ckpt_config[key]
                else:
                    config[key] = ckpt_config[key]

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {config_path}")

    # 6. Copy modeling file (for trust_remote_code)
    modeling_src = os.path.join(
        os.path.dirname(__file__), "cgc_mtp_eagle.py"
    )
    if os.path.exists(modeling_src):
        modeling_dst = os.path.join(output_dir, "modeling_cgc_mtp_eagle.py")
        shutil.copy2(modeling_src, modeling_dst)
        print(f"Saved: {modeling_dst}")
    else:
        print(f"WARNING: modeling file not found at {modeling_src}")

    # 7. Print sglang launch command
    print(f"\n{'='*60}")
    print(f"EAGLE draft model ready: {output_dir}")
    print(f"{'='*60}")
    print(f"\nsglang launch command:")
    print(f"  python3 -m sglang.launch_server \\")
    print(f"    --model-path /data/models/<target_model> \\")
    print(f"    --speculative-algorithm EAGLE \\")
    print(f"    --speculative-draft-model-path {output_dir} \\")
    print(f"    --speculative-num-steps 3 \\")
    print(f"    --speculative-eagle-topk 1 \\")
    print(f"    --speculative-num-draft-tokens 4 \\")
    print(f"    --trust-remote-code \\")
    print(f"    --host 0.0.0.0 --port 30003")


def main():
    parser = argparse.ArgumentParser(
        description="Convert .pt MTP head checkpoint to sglang EAGLE HF format"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to .pt checkpoint (e.g., mtp_head_qwen3vl_decode.pt)"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for EAGLE draft model"
    )
    parser.add_argument(
        "--model-name", required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model name for config (qwen3vl, dsv4, gemma4)"
    )
    args = parser.parse_args()

    convert_checkpoint(args.checkpoint, args.output_dir, args.model_name)


if __name__ == "__main__":
    main()
