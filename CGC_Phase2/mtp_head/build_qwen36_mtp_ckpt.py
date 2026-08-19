#!/usr/bin/env python3
"""Build the Qwen3.6 official fused-MTP head checkpoint in CGC framework format.

Loads the 19 official `mtp.*` tensors (shard 26), maps them into
Qwen3_6MTPHead state-dict keys, converts norm weights from the Qwen3.6
`1 + w` convention to MTPRMSNorm's `w` convention, and saves
{"model_state_dict": sd} for validate_mtp_accept.py --checkpoint.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

HF = Path("/Volumes/AlexZhuang/qwen36-hf")
# project-local temp (macOS StorageManagement periodically purges /tmp)
OUT = Path("/Users/alexchuang/Documents/flashkv0516/temp/qwen36_mtp_data")
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, "/Users/alexchuang/Documents/flashkv0516")
sys.path.insert(0, "/Users/alexchuang/Documents/flashkv0516/CGC_Phase2/mtp_head")
from model import create_mtp_head_for_qwen36


def load(name, cache):
    idx = json.load(open(HF / "model.safetensors.index.json"))
    shard = idx["weight_map"][name]
    if shard not in cache:
        cache[shard] = safe_open(HF / shard, framework="pt")
    return cache[shard].get_tensor(name)


# official tensor -> module attr path
M = {
    "mtp.fc.weight": "proj.weight",
    "mtp.pre_fc_norm_embedding.weight": "proj.norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight": "proj.norm_hidden.weight",
    "mtp.norm.weight": "norm_out.weight",
    "mtp.layers.0.input_layernorm.weight": "norm1.weight",
    "mtp.layers.0.post_attention_layernorm.weight": "norm2.weight",
    "mtp.layers.0.self_attn.q_proj.weight": "attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight": "attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight": "attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight": "attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight": "attn.q_norm",
    "mtp.layers.0.self_attn.k_norm.weight": "attn.k_norm",
    "mtp.layers.0.mlp.gate.weight": "mlp.gate.weight",
    "mtp.layers.0.mlp.experts.gate_up_proj": "mlp.experts_gate_up",
    "mtp.layers.0.mlp.experts.down_proj": "mlp.experts_down",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": "mlp.shared_gate.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": "mlp.shared_up.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": "mlp.shared_down.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight": "mlp.shared_gate_scalar.weight",
}
# keys using the 1+w norm convention (convert to MTPRMSNorm w convention).
# 实测定案 (2026-08-09): 仅 q/k norm 使用 1+w (与主模型 GatedAttn 一致);
# layer norm (pre_fc_norm_{hidden,embedding}/input_layernorm/post_attention_layernorm/norm)
# 的存储权重就是完整权重 (raw 直接用), 不要 +1 — plus1_all 把 code 接受率
# 从 90.8% 拉到 86.7% (prose 55.4 -> 50.8)。
NORM_KEYS = {"attn.q_norm", "attn.k_norm"}

mtp = create_mtp_head_for_qwen36()
cache = {}
sd = {}
with torch.no_grad():
    for official_key, module_key in M.items():
        t = load(official_key, cache).float()
        if module_key in NORM_KEYS:
            t = 1.0 + t
        sd[module_key] = t
        print(f"  {official_key:55s} -> {module_key:35s} {tuple(t.shape)}")

mtp.load_state_dict(sd, strict=True)
print(f"\n[ok] strict load. params: {mtp.num_parameters()/1e6:.1f}M")

torch.save({"model_state_dict": sd}, OUT / "mtp_head_qwen36_official.pt")
print(f"[ok] saved {OUT / 'mtp_head_qwen36_official.pt'}", flush=True)
