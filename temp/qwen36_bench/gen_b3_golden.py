#!/usr/bin/env python3
"""生成 delta_rule B=3 golden: framework Qwen3_5MoeGatedDeltaNet 层 0,
3-token 输入的 q/k/v/g/beta (post-GQA, fp16) + torch_recurrent_gated_delta_rule
的 per-token 输出 o 与 per-step 状态 h (fp32), 供 Metal deltanet_delta_rule_b
B=3 对拍。"""
import sys, json
sys.path.insert(0, "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-repack")
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M
from resident_writer import read_resident_bin

TEST_DIR = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/models/qwen36-test"
OUT = f"{TEST_DIR}/metal_out"
HF = "/Volumes/AlexZhuang/qwen36-hf"

torch.set_default_dtype(torch.bfloat16)

info = read_resident_bin(Path(f"{TEST_DIR}/model_weights.bin"))

def load_entry(name, want_fp16=True):
    e = info["entries"][name]
    with open(f"{TEST_DIR}/model_weights.bin", "rb") as f:
        f.seek(e["fileOffset"])
        blob = f.read(e["sizeBytes"])
    shp = [s for s in e["shape"] if s != 0]
    if e["dtype"] == 1:
        arr = np.frombuffer(blob, dtype=np.float16).reshape(shp)
    elif e["dtype"] == 2:
        arr = np.frombuffer(blob, dtype=np.float32).reshape(shp)
    elif e["dtype"] == 0 and e.get("scaleSize", 0) > 0:
        # q4 packed: dequant
        packed = np.frombuffer(blob, dtype=np.uint8)
        fo = e["fileOffset"]
        ss, sb = e["scaleOffset"] - fo, e["scaleSize"]
        bs_, bb = e["biasOffset"] - fo, e["biasSize"]
        scales = np.frombuffer(blob[ss:ss + sb], dtype=np.uint16)
        biases = np.frombuffer(blob[bs_:bs_ + bb], dtype=np.uint16)
        cols, rows = shp[1], shp[0]
        p = packed[:rows * cols // 2].reshape(rows, cols // 2).astype(np.uint16)
        q = np.empty((rows, cols), dtype=np.uint8)
        q[:, 0::2] = p & 0x0F
        q[:, 1::2] = p >> 4
        sc = np.repeat((scales.astype(np.uint32) << 16).view(np.float32), 64).reshape(rows, cols)
        bs2 = np.repeat((biases.astype(np.uint32) << 16).view(np.float32), 64).reshape(rows, cols)
        arr = (q.astype(np.float32) * sc + bs2)
        return torch.from_numpy(arr).to(torch.bfloat16)
    else:
        raise SystemExit(f"unsupported dtype {e['dtype']} for {name}")
    t = torch.from_numpy(arr)
    return t.to(torch.bfloat16) if want_fp16 else t.to(torch.bfloat16)

cfg = AutoConfig.from_pretrained(HF)
text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
layer = M.Qwen3_5MoeGatedDeltaNet(text_cfg, 0)
layer.eval()

prefix = "model.language_model.layers.0.linear_attn."
for pname, p in layer.named_parameters():
    if pname.endswith(("in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight",
                       "conv1d.weight", "norm.weight", "A_log", "dt_bias")):
        continue
sd = {}
for n in ["in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight",
          "in_proj_a.weight", "in_proj_b.weight",
          "conv1d.weight", "norm.weight", "A_log", "dt_bias"]:
    try:
        sd[n] = load_entry(prefix + n)
    except KeyError:
        print("skip", n)
# conv1d bias + dt_bias may be fp16 in resident
loaded = layer.load_state_dict(sd, strict=False)
print("missing:", loaded.missing_keys)

# 3-token input (from the existing test inputs_layer0.bin first 3 tokens)
inp = np.fromfile(f"{TEST_DIR}/metal_out/inputs_layer0.bin", dtype=np.float16)[: 3 * 2048].reshape(1, 3, 2048)
x = torch.from_numpy(inp).to(torch.bfloat16)

with torch.no_grad():
    # replicate forward up to recurrent call (no cache, multi-token)
    hs = x
    mixed_qkv = layer.in_proj_qkv(hs).transpose(1, 2)
    z = layer.in_proj_z(hs).reshape(1, 3, -1, layer.head_v_dim)
    b = layer.in_proj_b(hs)
    a = layer.in_proj_a(hs)
    # causal conv (no cache → zero left context)
    mixed_qkv = F.silu(layer.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
    B, S = 1, 3
    query = query.reshape(B, S, -1, layer.head_k_dim)
    key = key.reshape(B, S, -1, layer.head_k_dim)
    value = value.reshape(B, S, -1, layer.head_v_dim)
    beta = b.sigmoid()
    g = -layer.A_log.float().exp() * F.softplus(a.float() + layer.dt_bias)
    if layer.num_v_heads // layer.num_k_heads > 1:
        query = query.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)
        key = key.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)

    q = query[0].float()   # [3, 32, 128]
    k = key[0].float()     # [3, 32, 128]
    v = value[0].float()   # [3, 32, 128]
    g3 = g[0].float()      # [3, 32]
    beta3 = beta[0].float()  # [3, 32]

    # l2norm + scale to match kernel prepare (q *= 1/sqrt(128))
    scale = 1.0 / np.sqrt(128.0)
    qn = F.normalize(q, p=2, dim=-1, eps=1e-6) * scale
    kn = F.normalize(k, p=2, dim=-1, eps=1e-6)

    # recurrent reference: framework torch_recurrent_gated_delta_rule
    # (l2norm 已在上面应用; 传回未 scale 的 q 让函数内部 scale)
    o_all, h_final = M.torch_recurrent_gated_delta_rule(
        qn.unsqueeze(0) / scale, kn.unsqueeze(0), v.unsqueeze(0),
        g3.unsqueeze(0), beta3.unsqueeze(0), None, True)
    o_all = o_all[0].float()   # [3, 32, 128]
    h_final = h_final[0].float()  # [32, 128, 128]
    # per-step state snapshots: re-run stepwise to capture h after each token
    # (framework 只回传 final state; 用正确 broadcasting 的 per-token loop)
    h = torch.zeros(32, 128, 128)
    hs_all = []
    for t in range(3):
        ht = h * g3[t, :, None, None].exp()
        kv_mem = (ht * kn[t].view(32, 128, 1)).sum(dim=1)      # [32, 128] 正确: Σ_i h[i,j]·k[i]
        delta = (v[t] - kv_mem) * beta3[t, :, None]
        h = ht + kn[t].view(32, 128, 1) * delta.view(32, 1, 128)
        hs_all.append(h.clone())
    o_loop = torch.stack([(hs_all[t] * qn[t].view(32, 128, 1)).sum(dim=1) for t in range(3)])
    # 确认 stepwise loop 与 framework 一致 (o 用 h 计算, 与框架 o 对齐)
    print("loop vs framework o max|d| =", (o_loop - o_all).abs().max().item())
    print("loop vs framework h max|d| =", (hs_all[-1] - h_final).abs().max().item())
    o_all = o_loop  # 使用 stepwise 输出 (与 h snapshots 一致)

np.save(f"{OUT}/b3_q.npy", qn.numpy().astype(np.float32))
np.save(f"{OUT}/b3_k.npy", kn.numpy().astype(np.float32))
np.save(f"{OUT}/b3_v.npy", v.numpy().astype(np.float32))
np.save(f"{OUT}/b3_g.npy", g3.numpy().astype(np.float32))
np.save(f"{OUT}/b3_beta.npy", beta3.numpy().astype(np.float32))
np.save(f"{OUT}/b3_o.npy", o_all.numpy().astype(np.float32))
np.save(f"{OUT}/b3_h_final.npy", h_final.numpy().astype(np.float32))
np.save(f"{OUT}/b3_h_snap.npy", torch.stack(hs_all).numpy().astype(np.float32))

# ---- §13.121 Phase A 全链 golden: norm + out_proj 尾段 (对拍 batch 化链 y) ----
# o_all [3,32,128] → reshape [3*32,128]; z [1,3,32,128] → reshape [3*32,128]
# 语义 (Qwen3_5MoeRMSNormGated): rms = o/sqrt(mean(o^2)+eps) × weight × silu(z)
o2 = o_all.reshape(-1, 128).float()
z2 = z.reshape(-1, 128).float()
norm_w = load_entry(prefix + "norm.weight")   # [128] bf16
variance = o2.pow(2).mean(-1, keepdim=True)
rms = o2 * torch.rsqrt(variance + 1e-6)
gated = (rms * norm_w.float()) * torch.nn.functional.silu(z2)
proj_w = load_entry(prefix + "out_proj.weight")  # [2048, 4096] bf16 (q4 dequantized)
y_full = (gated.reshape(1, 3, 4096) @ proj_w.float().T)  # [1,3,2048]
y_full = y_full[0].float()                       # [3, 2048]
np.save(f"{OUT}/b3_x.npy", inp.reshape(3, 2048).astype(np.float32))
np.save(f"{OUT}/b3_y.npy", y_full.numpy().astype(np.float32))
print("B=3 full-chain golden (x/y) written to", OUT)
print("y[0][:4] =", y_full[0][:4].tolist())
print("B=3 golden written to", OUT)
print("o[0][0][:4] =", o_all[0][0][:4].tolist())
