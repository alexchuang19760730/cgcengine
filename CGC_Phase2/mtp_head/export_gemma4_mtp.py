#!/usr/bin/env python3
"""把 finetune 後的 MTP head state_dict 導回官方格式 bf16 safetensors。

官方 key 集 (model.* 前綴) + 未被訓練的 embed_tokens 保留原值。
用法:
  python3 export_gemma4_mtp.py <ckpt.pt> <out/model.safetensors>
"""
from __future__ import annotations

import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from gemma4_assistant import load_gemma4_assistant

HEAD_DIR = "/Users/alexchuang/Documents/flashkv0516/models/gemma-4-mtp-head"
ORIG = os.path.join(HEAD_DIR, "model.safetensors")

# mirror key -> official key
KEYMAP = {
    "embed_tokens.weight": "model.embed_tokens.weight",
    "model_norm.weight": "model.norm.weight",
    "pre_projection.weight": "pre_projection.weight",
    "post_projection.weight": "post_projection.weight",
    "input_layernorm.weight": "input_layernorm.weight",
    "post_attention_layernorm.weight": "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight": "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight": "post_feedforward_layernorm.weight",
    "self_attn_q_proj.weight": "self_attn.q_proj.weight",
    "self_attn_q_norm": "self_attn.q_norm.weight",
    "self_attn_o_proj.weight": "self_attn.o_proj.weight",
    "mlp_gate_proj.weight": "mlp.gate_proj.weight",
    "mlp_up_proj.weight": "mlp.up_proj.weight",
    "mlp_down_proj.weight": "mlp.down_proj.weight",
    "layer_scalar": "layer_scalar",
}


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/alexchuang/Documents/flashkv0516/temp/g4_mtp_head_ft.pt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/Users/alexchuang/Documents/flashkv0516/temp/g4_mtp_head_ft.safetensors"

    # 原權重 (embed 保留原值; 也驗證 shape)
    orig = load_file(ORIG)  # safetensors (非 pickle)
    orig_keys = set(orig.keys())

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    out = {}
    # embed 用原值
    out["model.embed_tokens.weight"] = orig["model.embed_tokens.weight"]
    for k, v in sd.items():
        parts = k.split(".")
        if parts[0] == "layers":
            # layers.N.<suffix>
            layer = int(parts[1])
            suffix = ".".join(parts[2:])
            if suffix not in KEYMAP:
                print(f"skip {k} (no official mapping)")
                continue
            ok = f"model.layers.{layer}.{KEYMAP[suffix]}"
        elif k in KEYMAP:
            ok = KEYMAP[k]
        else:
            print(f"skip {k}")
            continue
        t = v.float().to(torch.bfloat16) if v.dtype != torch.bfloat16 else v
        if ok in orig_keys:
            # scalar param (layer_scalar) 官方是 (1,)
            if t.ndim == 0 and tuple(orig[ok].shape) == (1,):
                t = t.reshape(1)
            assert tuple(t.shape) == tuple(orig[ok].shape), \
                f"shape mismatch {ok}: {tuple(t.shape)} vs {tuple(orig[ok].shape)}"
        out[ok] = t.contiguous()
    save_file(out, out_path)
    print(f"saved {len(out)} tensors -> {out_path}")
    missing = sorted(orig_keys - set(out.keys()))
    print("official keys NOT written (should be empty):", missing)


if __name__ == "__main__":
    main()
