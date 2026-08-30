#!/usr/bin/env python3
"""
Convert maxlaurence distilled MTP head (MLX affine-quant safetensors)
to a small BF16 GGUF containing only blk.40.* tensors.

Output layout follows llama.cpp qwen35moe convention:
  - attn / shexp / fc:  [in, out]
  - switch_mlp experts:  [n_embd, n_ff, n_experts]

Step 1 of "Path A" (graft maxlaurence distilled MTP head into Nail IQ3_XXS trunk):
  - This file:            MLX safetensors  ->  BF16 GGUF  (this script)
  - Next:                 llama-quantize --pure Q4_K  (or selective --tensor-type)
  - Then:                 graft_gguf.py copies blk.40 tensors into Nail GGUF

See: moeexpert/MTP轉正規劃書_v1.0.md §8.85+
"""
import os, sys, json, struct, time
from pathlib import Path

# Add fork's gguf-py to path
GGUF_PY_PATH = "/Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/gguf-py"
sys.path.insert(0, GGUF_PY_PATH)

import numpy as np
import mlx.core as mx
from gguf import GGUFWriter

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SAFETENSORS_PATH = "/tmp/maxlaurence_mtp/model-mtp.safetensors"
OUTPUT_GGUF       = "/tmp/maxlaurence_mtp/blk40_maxlaurence_BF16.gguf"

# Tensor mapping: maxlaurence key -> GGUF tensor name
# Notes:
#   * maxlaunce MLX shape for switch_mlp.* is [n_experts, n_ff, n_embd] (row-major,
#     n_embd fastest). GGUF/llama.cpp qwen35moe expects [n_embd, n_ff, n_experts]
#     (n_experts slowest).  np.transpose(arr, [2,1,0]) does this swap.
#   * maxlaunce MLX shape for attn/shexp/fc projections is [out, in].
#     GGUF expects [in, out].  np.transpose(arr, [1,0]) does this swap.
#   * maxlaunce stores all norms in BF16; we promote to F32 to match Nail GGUF.
#   * ffn_gate_inp / ffn_gate_inp_shexp: keep BF16 (matches Nail GGUF).
# -----------------------------------------------------------------------------
TENSOR_MAP = [
    # (maxlaurence_key,                                gguf_name,                                 transpose,  target_type)
    ("language_model.mtp.fc.weight",                   "blk.40.nextn.eh_proj.weight",              True,       "BF16"),  # fc: [out=4096, in=2048] -> [in, out]
    ("language_model.mtp.pre_fc_norm_embedding.weight","blk.40.nextn.enorm.weight",                False,      "F32"),
    ("language_model.mtp.pre_fc_norm_hidden.weight",   "blk.40.nextn.hnorm.weight",               False,      "F32"),
    ("language_model.mtp.norm.weight",                 "blk.40.nextn.shared_head_norm.weight",     False,      "F32"),

    ("language_model.mtp.layers.0.input_layernorm.weight",          "blk.40.attn_norm.weight",    False, "F32"),
    ("language_model.mtp.layers.0.post_attention_layernorm.weight", "blk.40.post_attention_norm.weight", False, "F32"),
    ("language_model.mtp.layers.0.self_attn.q_norm.weight", "blk.40.attn_q_norm.weight", False, "F32"),
    ("language_model.mtp.layers.0.self_attn.k_norm.weight", "blk.40.attn_k_norm.weight", False, "F32"),

    # attn projections: MLX [out, in] -> GGUF [in, out]
    ("language_model.mtp.layers.0.self_attn.q_proj.weight", "blk.40.attn_q.weight",      True, "BF16"),
    ("language_model.mtp.layers.0.self_attn.k_proj.weight", "blk.40.attn_k.weight",      True, "BF16"),
    ("language_model.mtp.layers.0.self_attn.v_proj.weight", "blk.40.attn_v.weight",      True, "BF16"),
    ("language_model.mtp.layers.0.self_attn.o_proj.weight", "blk.40.attn_output.weight", True, "BF16"),

    # gating (BF16 to match Nail GGUF)
    ("language_model.mtp.layers.0.mlp.gate.weight",            "blk.40.ffn_gate_inp.weight",      False, "BF16"),
    ("language_model.mtp.layers.0.mlp.shared_expert_gate.weight", "blk.40.ffn_gate_inp_shexp.weight", False, "BF16"),

    # shared expert: MLX [out, in] -> GGUF [in, out]
    ("language_model.mtp.layers.0.mlp.shared_expert.gate_proj.weight", "blk.40.ffn_gate_shexp.weight",  True, "BF16"),
    ("language_model.mtp.layers.0.mlp.shared_expert.up_proj.weight",  "blk.40.ffn_up_shexp.weight",    True, "BF16"),
    ("language_model.mtp.layers.0.mlp.shared_expert.down_proj.weight","blk.40.ffn_down_shexp.weight",   True, "BF16"),

    # switch_mlp experts: MLX [n_experts, n_ff, n_embd] -> GGUF [n_embd, n_ff, n_experts]
    ("language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight", "blk.40.ffn_gate_exps.weight", "trans3d", "BF16"),
    ("language_model.mtp.layers.0.mlp.switch_mlp.up_proj.weight",   "blk.40.ffn_up_exps.weight",   "trans3d", "BF16"),
    ("language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight", "blk.40.ffn_down_exps.weight", "trans3d", "BF16"),
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def mlx_to_np(key, all_weights):
    """Load one tensor via mlx (auto-dequant) and return numpy array."""
    arr = all_weights[key]
    # mlx returns mlx.core.array; .astype works, but to convert to numpy we use np.asarray
    # Force BF16 first to ensure uniform dtype then convert to numpy via float32 intermediate
    np_arr = np.array(arr)
    return np_arr


def to_gguf_layout(np_arr, transpose_mode):
    """Apply the layout transform required by llama.cpp convention."""
    if transpose_mode is False or transpose_mode == "F32-norm":
        return np_arr
    if transpose_mode is True:
        # 2D [out, in] -> [in, out]
        assert np_arr.ndim == 2, f"2D transpose expects 2D, got {np_arr.shape}"
        return np.ascontiguousarray(np_arr.T)
    if transpose_mode == "trans3d":
        # 3D [n_experts, n_ff, n_embd] -> [n_embd, n_ff, n_experts]
        assert np_arr.ndim == 3, f"3D transpose expects 3D, got {np_arr.shape}"
        # np.transpose returns a view; ensure C-contiguous for gguf writer
        return np.ascontiguousarray(np.transpose(np_arr, (2, 1, 0)))
    raise ValueError(f"unknown transpose_mode: {transpose_mode}")


def promote_dtype(np_arr, target_type):
    """Convert numpy dtype to match GGUF target_type semantics."""
    if target_type == "F32":
        return np_arr.astype(np.float32, copy=False)
    if target_type == "BF16":
        # GGUF raw_dtype will be BF16 (ggml type 1). numpy doesn't have native bf16,
        # so we store as float32 and tell gguf writer to interpret as BF16 via raw_dtype.
        # Easier: keep float32 here; quantize step will see the values as F32 and produce Q4_K/Q6_K.
        # For BF16-target tensors (gate_inp) we still write as F32 and let gguf writer decide.
        return np_arr.astype(np.float32, copy=False)
    raise ValueError(f"unknown target_type: {target_type}")


# -----------------------------------------------------------------------------
# Main conversion
# -----------------------------------------------------------------------------

def main():
    out_path = Path(OUTPUT_GGUF)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading safetensors: {SAFETENSORS_PATH}")
    weights, _ = mx.load(SAFETENSORS_PATH, return_metadata=True, format="safetensors")
    print(f"  mlx loaded {len(weights)} tensors")

    # Build GGUF writer - architecture is dummy; we'll add minimal metadata
    writer = GGUFWriter(str(out_path), arch="llama")
    # Minimal required metadata
    writer.add_block_count(1)  # only 1 "layer" in this small head GGUF
    writer.add_uint32("general.file_type", 1)  # mostly f16/bf16 in this temp file
    writer.add_uint32("llama.context_length", 4096)
    writer.add_uint32("llama.embedding_length", 2048)
    writer.add_uint32("llama.feed_forward_length", 512)
    writer.add_uint32("llama.attention.head_count", 32)
    writer.add_uint32("llama.attention.head_count_kv", 8)
    writer.add_uint32("llama.expert_count", 256)
    writer.add_uint32("llama.expert_used_count", 8)
    writer.add_uint32("llama.attention.layer_norm_rms_epsilon", 1e-6)
    # Mark this as a "head-only" patch file: no token embeddings, no output
    writer.add_uint32("llama.tensor_data_layout", 0)

    print(f"\n=== Converting {len(TENSOR_MAP)} tensors ===")
    for mlx_key, gguf_name, transpose_mode, target_type in TENSOR_MAP:
        t0 = time.time()
        if mlx_key not in weights:
            print(f"  [MISSING] {mlx_key}")
            continue
        np_arr = mlx_to_np(mlx_key, weights)
        np_arr = to_gguf_layout(np_arr, transpose_mode)
        np_arr = promote_dtype(np_arr, target_type)
        # add_tensor expects C-contiguous array
        np_arr = np.ascontiguousarray(np_arr)
        # Write as F32; llama-quantize will handle BF16/F32 -> Q4_K/Q6_K/etc.
        writer.add_tensor(gguf_name, np_arr)
        print(f"  [OK] {gguf_name:50s}  shape={list(np_arr.shape)}  dtype={np_arr.dtype}  t={time.time()-t0:.2f}s")

    print(f"\n=== Finalizing GGUF: {out_path} ===")
    writer.finalize()
    print(f"  size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\nNext step:")
    print(f"  llama-quantize --pure q4_k {out_path} /tmp/maxlaurence_mtp/blk40_maxlaurence_Q4K.gguf q4_k")


if __name__ == "__main__":
    main()
