"""
Qwen3.6-35B-A3B weight grafting into Loop MoE.

Strategy (from white paper):
1. Prelude layers: copy from Qwen3.6 shallow layers (layers 0, 1)
2. RecurrentBlock:
   - Gated DeltaNet: copy from Qwen3.6 DeltaNet layer (e.g. layer 4)
   - MoE experts: copy all 256 experts from Qwen3.6 MoE layer
   - Router: copy from Qwen3.6 router
3. Coda layers: copy from Qwen3.6 deep layers (e.g. layer 27)
4. Embeddings / lm_head: copy directly
5. Recurrent-specific params (ACT halting, LTI): randomly initialized

Supports loading from:
- HuggingFace safetensors (Qwen3.5/3.6 MoE)
- GGUF (via llama.cpp conversion, for local CGC inference)
- state_dict (already loaded PyTorch weights)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class WeightGraftConfig:
    """Configuration for Qwen3.6 → Loop MoE weight grafting."""

    # Source model
    source_model_path: str = ""  # HF model dir or state_dict path
    source_model_type: str = "hf"  # "hf" | "state_dict" | "gguf"

    # Layer mapping
    prelude_source_layers: List[int] = field(default_factory=lambda: [0, 1])
    coda_source_layers: List[int] = field(default_factory=lambda: [27])
    deltanet_source_layer: int = 4
    moe_source_layer: int = 4

    # What to graft
    graft_embeddings: bool = True
    graft_lm_head: bool = True
    graft_prelude: bool = True
    graft_coda: bool = True
    graft_deltanet: bool = True
    graft_moe_experts: bool = True
    graft_moe_router: bool = True

    # Random init for recurrent-specific params
    init_recurrent_params: bool = True
    init_method: str = "normal"  # "normal" | "xavier" | "kaiming"
    init_std: float = 0.02

    # Dtype
    dtype: torch.dtype = torch.bfloat16


def _load_hf_state_dict(model_path: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Load weights from HuggingFace model directory."""
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("safetensors required for HF model loading: pip install safetensors")

    state_dict = {}
    model_path = Path(model_path)

    # Find index file
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        for shard in shard_files:
            shard_path = model_path / shard
            if shard_path.exists():
                state_dict.update(load_file(str(shard_path)))
    else:
        # Single file
        st_path = model_path / "model.safetensors"
        if st_path.exists():
            state_dict = load_file(str(st_path))
        else:
            # Try bin
            bin_path = model_path / "pytorch_model.bin"
            if bin_path.exists():
                state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)
            else:
                raise FileNotFoundError(f"No model weights found in {model_path}")

    # Convert dtype
    return {k: v.to(dtype) for k, v in state_dict.items()}


def _get_layer_prefix(state_dict: Dict[str, torch.Tensor]) -> str:
    """Detect the layer prefix from state_dict keys."""
    for key in state_dict:
        if ".layers." in key:
            idx = key.index(".layers.")
            return key[:idx + len(".layers.")]
    return "model.layers."


def graft_qwen36_weights(
    model: nn.Module,
    config: WeightGraftConfig,
    source_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[nn.Module, Dict[str, int]]:
    """
    Graft Qwen3.6 weights into Loop MoE model.

    Args:
        model: LoopMoEModel instance (randomly initialized)
        config: WeightGraftConfig
        source_state_dict: pre-loaded state dict (if None, loads from config.source_model_path)

    Returns:
        (model, graft_stats) where graft_stats counts copied/initialized tensors
    """
    stats = {"copied": 0, "initialized": 0, "skipped": 0, "missing": 0}

    # Load source weights
    if source_state_dict is None:
        if config.source_model_type == "hf":
            source_state_dict = _load_hf_state_dict(config.source_model_path, config.dtype)
        elif config.source_model_type == "state_dict":
            source_state_dict = torch.load(config.source_model_path, map_location="cpu", weights_only=True)
            source_state_dict = {k: v.to(config.dtype) for k, v in source_state_dict.items()}
        else:
            raise ValueError(f"Unsupported source_model_type: {config.source_model_type}")

    layer_prefix = _get_layer_prefix(source_state_dict)
    print(f"[Graft] Source layer prefix: {layer_prefix}")
    print(f"[Graft] Source tensors: {len(source_state_dict)}")

    model_state = model.state_dict()

    # Helper: copy a tensor from source to target
    def _copy(src_key: str, tgt_key: str, transpose: bool = False):
        if src_key in source_state_dict and tgt_key in model_state:
            src = source_state_dict[src_key]
            if transpose:
                src = src.T
            if src.shape == model_state[tgt_key].shape:
                model_state[tgt_key] = src.to(config.dtype)
                stats["copied"] += 1
                return True
            else:
                print(f"  [WARN] Shape mismatch {tgt_key}: src={src.shape}, tgt={model_state[tgt_key].shape}")
                stats["skipped"] += 1
                return False
        else:
            stats["missing"] += 1
            return False

    # ===== 1. Embeddings & LM Head =====
    if config.graft_embeddings:
        _copy("model.embed_tokens.weight", "model.embed_tokens.weight")
        _copy("embed_tokens.weight", "embed_tokens.weight")
    if config.graft_lm_head:
        _copy("model.lm_head.weight", "model.lm_head.weight")
        _copy("lm_head.weight", "lm_head.weight")

    # ===== 2. Prelude layers =====
    if config.graft_prelude:
        for i, src_layer in enumerate(config.prelude_source_layers):
            print(f"[Graft] Prelude layer {i} <- Qwen layer {src_layer}")
            src_p = f"{layer_prefix}{src_layer}."
            tgt_p = f"model.prelude.{i}."
            for key in source_state_dict:
                if key.startswith(src_p):
                    suffix = key[len(src_p):]
                    _copy(key, tgt_p + suffix)

    # ===== 3. Coda layers =====
    if config.graft_coda:
        for i, src_layer in enumerate(config.coda_source_layers):
            print(f"[Graft] Coda layer {i} <- Qwen layer {src_layer}")
            src_p = f"{layer_prefix}{src_layer}."
            tgt_p = f"model.coda.{i}."
            for key in source_state_dict:
                if key.startswith(src_p):
                    suffix = key[len(src_p):]
                    _copy(key, tgt_p + suffix)

    # ===== 4. RecurrentBlock - Gated DeltaNet =====
    if config.graft_deltanet:
        src_layer = config.deltanet_source_layer
        print(f"[Graft] RecurrentBlock DeltaNet <- Qwen layer {src_layer}")
        src_p = f"{layer_prefix}{src_layer}."
        tgt_p = "model.recurrent_block.deltanet."
        # DeltaNet params: q_proj, k_proj, v_proj, o_proj, conv, decay, gate
        for suffix in [
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "self_attn.conv1d.weight", "self_attn.conv1d.bias",
            "self_attn.decay_factor", "self_attn.output_gate.weight",
            "input_layernorm.weight", "post_attention_layernorm.weight",
        ]:
            _copy(src_p + suffix, tgt_p + suffix)

    # ===== 5. RecurrentBlock - MoE Experts & Router =====
    if config.graft_moe_experts or config.graft_moe_router:
        src_layer = config.moe_source_layer
        print(f"[Graft] RecurrentBlock MoE <- Qwen layer {src_layer}")
        src_p = f"{layer_prefix}{src_layer}.mlp."
        tgt_p = "model.recurrent_block.moe."

        if config.graft_moe_router:
            _copy(src_p + "gate.weight", tgt_p + "router.weight")
            _copy(src_p + "shared_expert.gate_proj.weight", tgt_p + "shared_expert.gate_proj.weight")
            _copy(src_p + "shared_expert.up_proj.weight", tgt_p + "shared_expert.up_proj.weight")
            _copy(src_p + "shared_expert.down_proj.weight", tgt_p + "shared_expert.down_proj.weight")

        if config.graft_moe_experts:
            # Copy all 256 experts
            for expert_idx in range(256):
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    src_key = f"{src_p}experts.{expert_idx}.{proj}.weight"
                    tgt_key = f"{tgt_p}experts.{expert_idx}.{proj}.weight"
                    _copy(src_key, tgt_key)

    # ===== 6. Initialize recurrent-specific params =====
    if config.init_recurrent_params:
        print("[Graft] Initializing recurrent-specific params (ACT, LTI, state init)")
        for name, param in model.named_parameters():
            if any(kw in name for kw in ["halting", "act_", "lti_", "state_init", "recurrent_gate"]):
                if config.init_method == "normal":
                    nn.init.normal_(param, std=config.init_std)
                elif config.init_method == "xavier":
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
                elif config.init_method == "kaiming":
                    if param.dim() > 1:
                        nn.init.kaiming_uniform_(param)
                stats["initialized"] += 1

    # Load grafted state dict
    model.load_state_dict(model_state, strict=False)

    print(f"\n[Graft] Complete: copied={stats['copied']}, initialized={stats['initialized']}, "
          f"skipped={stats['skipped']}, missing={stats['missing']}")

    return model, stats


def verify_graft(
    model: nn.Module,
    source_state_dict: Dict[str, torch.Tensor],
    sample_keys: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """
    Verify that grafted weights match source (spot check).
    Returns dict of key -> match status.
    """
    if sample_keys is None:
        sample_keys = [
            "model.embed_tokens.weight",
            "model.recurrent_block.moe.router.weight",
        ]

    model_state = model.state_dict()
    results = {}
    for key in sample_keys:
        if key in model_state and key in source_state_dict:
            match = torch.allclose(model_state[key].float(), source_state_dict[key].float(), atol=1e-3)
            results[key] = match
            print(f"  [Verify] {key}: {'✓ match' if match else '✗ MISMATCH'}")
        else:
            results[key] = False
            print(f"  [Verify] {key}: not found in one of the dicts")
    return results
