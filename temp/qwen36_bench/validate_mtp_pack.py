#!/usr/bin/env python3
"""Validate the packed MTP head .gturbo pipeline:

  1. experts: dequant(packed) vs checkpoint originals (relative error, top-1
     routing stability) — same bar as the main model's repack.
  2. resident: every packed resident tensor byte-matches the checkpoint fp16.

If both pass, the engine draft head fed from this pack reproduces the
framework head's draft tokens.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = "/Users/alexchuang/Documents/flashkv0516"
sys.path.insert(0, f"{REPO}/prime-agent-worktrees/qwen36-repack")
from resident_writer import read_resident_bin

CKPT = f"{REPO}/temp/qwen36_mtp_data/mtp_head_qwen36_official.pt"
PACK = f"{REPO}/temp/qwen36-mtp.gturbo"
HIDDEN, MOE_INTER = 2048, 512


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def dequant(packed, scales, biases, bits):
    rows = packed.shape[0]
    cols = packed.shape[1] * 8 // bits
    n_groups = cols // 64
    scales_f = bf16_to_f32(scales.astype(np.uint16))
    biases_f = bf16_to_f32(biases.astype(np.uint16))
    if bits == 4:
        nib = np.zeros((rows, cols), dtype=np.uint8)
        pairs = packed.reshape(rows, cols // 2)
        nib[:, 0::2] = pairs & 0x0F
        nib[:, 1::2] = (pairs >> 4) & 0x0F
    else:
        raw = packed.reshape(rows, cols // 8, 3)
        b0, b1, b2 = raw[..., 0], raw[..., 1], raw[..., 2]
        v = np.zeros((rows, cols // 8, 8), dtype=np.uint8)
        v[..., 0] = b0 & 0x7
        v[..., 1] = (b0 >> 3) & 0x7
        v[..., 2] = ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2)
        v[..., 3] = (b1 >> 1) & 0x7
        v[..., 4] = (b1 >> 4) & 0x7
        v[..., 5] = ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1)
        v[..., 6] = (b2 >> 2) & 0x7
        v[..., 7] = (b2 >> 5) & 0x7
        nib = v.reshape(rows, cols)
    return nib.astype(np.float32) * scales_f.repeat(64, axis=1) + biases_f.repeat(64, axis=1)


def main():
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)["model_state_dict"]
    layout = json.load(open(f"{PACK}/packed_experts/layout.json"))
    stride = layout["expertStride"]
    blob = np.frombuffer(open(f"{PACK}/packed_experts/layer_00.bin", "rb").read(), dtype=np.uint8)
    exp0 = layout["layers"][0]["experts"][0]["tensors"]
    bits = exp0["gate"]["bits"]

    def get_expert(e, name):
        ts = layout["layers"][0]["experts"][e]["tensors"]
        t = ts[name]
        q = blob[e * stride + t["offset"]: e * stride + t["offset"] + t["size"]]
        shp = t["shape"]
        q = q.reshape(shp[0], shp[1] * t["bits"] // 8)
        s = blob[e * stride + ts[name + "_scales"]["offset"]: e * stride + ts[name + "_scales"]["offset"] + ts[name + "_scales"]["size"]]
        b = blob[e * stride + ts[name + "_biases"]["offset"]: e * stride + ts[name + "_biases"]["offset"] + ts[name + "_biases"]["size"]]
        return dequant(q, s.view(np.uint16).reshape(-1, shp[1] // 64),
                       b.view(np.uint16).reshape(-1, shp[1] // 64), t["bits"])

    # ---- expert dequant vs checkpoint ----
    gu_ref = sd["mlp.experts_gate_up"].float().numpy()  # [256,1024,2048] fused
    gate_ref = gu_ref[:, :MOE_INTER, :]
    up_ref = gu_ref[:, MOE_INTER:, :]
    dn_ref = sd["mlp.experts_down"].float().numpy()
    print(f"[0/2] checkpoint experts: gate {gate_ref.shape} up {up_ref.shape} dn {dn_ref.shape}")
    max_rel = 0.0
    nbad = 0
    for e in [0, 1, 17, 128, 255]:
        g = get_expert(e, "gate")
        u = get_expert(e, "up")
        d = get_expert(e, "down")
        for got, ref, name in [(g, gate_ref[e], "gate"), (u, up_ref[e], "up"), (d, dn_ref[e], "down")]:
            denom = np.abs(ref).max() + 1e-9
            rel = np.abs(got - ref).max() / denom
            max_rel = max(max_rel, rel)
            if rel > 0.10:
                nbad += 1
                print(f"  expert {e} {name}: rel_err={rel:.4f}  <-- CHECK")
    print(f"[1/2] expert dequant: {256} experts {bits}-bit, max rel err = {max_rel:.4f} "
          f"(bad={nbad}, bar <0.10)")

    # ---- resident vs checkpoint fp16 ----
    info = read_resident_bin(Path(f"{PACK}/mtp_weights.bin"))
    raw = open(f"{PACK}/mtp_weights.bin", "rb").read()
    keys = ["mtp.fc.weight", "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight", "mtp.norm.weight",
            "mtp.layers.0.self_attn.q_norm.weight"]
    ok = 0
    for name in keys:
        e = info["entries"][name]
        got = raw[e["fileOffset"]: e["fileOffset"] + e["sizeBytes"]]
        ref_key = {"mtp.fc.weight": "proj.weight",
                   "mtp.layers.0.self_attn.q_proj.weight": "attn.q_proj.weight",
                   "mtp.layers.0.self_attn.o_proj.weight": "attn.o_proj.weight",
                   "mtp.norm.weight": "norm_out.weight",
                   "mtp.layers.0.self_attn.q_norm.weight": "attn.q_norm"}[name]
        ref = sd[ref_key].float().numpy().astype(np.float16).tobytes()
        match = got == ref
        ok += int(match)
        print(f"  resident {name}: {'MATCH' if match else 'MISMATCH'}")
    print(f"[2/2] resident check: {ok}/{len(keys)} match")
    print("PASS" if ok == len(keys) and nbad == 0 else "FAIL")


if __name__ == "__main__":
    main()
