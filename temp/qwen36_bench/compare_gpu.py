#!/usr/bin/env python3
"""Feed the EXACT token ids the GPU engine used, run full CPU forward, and
dump per-layer post-attention-norm stats + final logits top-5 for comparison."""
import os, sys, json, time, gc
sys.path.insert(0, "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-repack")
from pathlib import Path
from resident_writer import read_resident_bin as _read_resident
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.activations import ACT2FN
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M

GTURBO = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-r4.gturbo"
HF = "/Volumes/AlexZhuang/qwen36-hf"
BITS = 4
IDS = [248045, 846, 198, 760, 6511, 314, 9338, 369, 248046, 198, 248045, 74455, 198]

import torch
import faulthandler; faulthandler.enable()
torch.set_num_threads(4)
def log(m): print(m, flush=True)

def load_dense_state_dict(gturbo):
    info = _read_resident(Path(gturbo) / "model_weights.bin")
    sd = {}
    with open(os.path.join(gturbo, "model_weights.bin"), "rb") as f:
        for name, e in info["entries"].items():
            f.seek(e["fileOffset"])
            blob = f.read(e["sizeBytes"])
            shp = [s for s in e["shape"] if s != 0]
            key = name.replace("model.language_model.", "model.", 1)
            if e["dtype"] == 1:
                arr = np.frombuffer(blob, dtype=np.float16).reshape(shp).astype(np.float32)
                sd[key] = torch.from_numpy(arr).to(torch.bfloat16)
            elif e["dtype"] == 2:
                arr = np.frombuffer(blob, dtype=np.float32).reshape(shp)
                sd[key] = torch.from_numpy(arr).to(torch.bfloat16)
            else:
                sd[key] = torch.from_numpy(np.frombuffer(blob, dtype=np.uint32).reshape(shp))
    return sd

def bf16_to_float(bits):
    return np.frombuffer(np.uint32(int(bits) << 16).tobytes(), dtype=np.float32)[0].item()

def dequant_rows(packed, scales, biases, bits, cols):
    rows = packed.shape[0]
    sc = np.repeat(np.vectorize(bf16_to_float, otypes=[np.float32])(scales.reshape(-1).astype(np.uint16)), 64).reshape(rows, cols)
    bs = np.repeat(np.vectorize(bf16_to_float, otypes=[np.float32])(biases.reshape(-1).astype(np.uint16)), 64).reshape(rows, cols)
    q = np.zeros((rows, cols), dtype=np.int64)
    for k in range(cols):
        byte = packed[:, k // 2]
        q[:, k] = (byte & 0x0F) if k % 2 == 0 else (byte >> 4)
    return (q.astype(np.float32) * sc + bs)

layout = json.load(open(f"{GTURBO}/packed_experts/layout.json"))
num_layers = layout["numLayers"]
n_experts = layout["expertsPerLayer"]

_layer_counter = [0]
class LocalStreamingQwenExperts(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_dim = config.moe_intermediate_size
        self.hidden_dim = config.hidden_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.layer_idx = _layer_counter[0]
        _layer_counter[0] += 1
    def _expert_tensors(self, e):
        lay = layout["layers"][self.layer_idx]
        ent = lay["experts"][e]
        t = ent["tensors"]
        with open(os.path.join(GTURBO, "packed_experts", lay["file"]), "rb") as f:
            def read(role):
                info = t[role]
                f.seek(ent["offset"] + info["offset"])
                raw = f.read(info["size"])
                if info["bits"] == 16:
                    return np.frombuffer(raw, dtype=np.uint16).reshape(info["shape"])
                return raw
            gate_w = read("gate"); gate_s = read("gate_scales"); gate_b = read("gate_biases")
            up_w = read("up"); up_s = read("up_scales"); up_b = read("up_biases")
            down_w = read("down"); down_s = read("down_scales"); down_b = read("down_biases")
        cols = self.hidden_dim
        gate = dequant_rows(np.frombuffer(gate_w, dtype=np.uint8).reshape(-1, cols * BITS // 8), gate_s, gate_b, BITS, cols).astype(np.float32)
        up = dequant_rows(np.frombuffer(up_w, dtype=np.uint8).reshape(-1, cols * BITS // 8), up_s, up_b, BITS, cols).astype(np.float32)
        ic = self.intermediate_dim
        down = dequant_rows(np.frombuffer(down_w, dtype=np.uint8).reshape(-1, ic * BITS // 8), down_s, down_b, BITS, ic).astype(np.float32)
        return (gate, up, down)
    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()
        for row in expert_hit:
            e = row[0].item()
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            w = top_k_weights[token_idx, top_k_pos, None]
            gate, up, down = self._expert_tensors(e)
            g = torch.from_numpy(gate).to(hidden_states.device)
            u = torch.from_numpy(up).to(hidden_states.device)
            d = torch.from_numpy(down).to(hidden_states.device)
            gate_o = F.linear(hidden_states[token_idx], g)
            up_o = F.linear(hidden_states[token_idx], u)
            h = self.act_fn(gate_o) * up_o
            out = F.linear(h, d) * w
            final.index_add_(0, token_idx, out.to(final.dtype))
        return final

M.Qwen3_5MoeExperts = LocalStreamingQwenExperts
def _patched_init_weights(self, module):
    if isinstance(module, LocalStreamingQwenExperts):
        return
    return _orig_init_weights(self, module)
_orig_init_weights = M.Qwen3_5MoePreTrainedModel._init_weights
M.Qwen3_5MoePreTrainedModel._init_weights = _patched_init_weights

cfg = AutoConfig.from_pretrained(HF)
text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
model = M.Qwen3_5MoeForCausalLM(text_cfg)
model.eval()

t1 = time.time()
sd = load_dense_state_dict(GTURBO)
missing = model.load_state_dict(sd, strict=False)
log(f"dense loaded ({time.time()-t1:.0f}s); missing={len(missing.missing_keys)}")
del sd; gc.collect()

ids = torch.tensor([IDS], dtype=torch.long)

# hooks: capture post-attention layernorm output per layer
norm_stats = {}
def make_hook(i):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        n = h[:, -1]  # last token row
        norm_stats[i] = (float(n.std()), np.round(n[:8].detach().float().numpy(), 3).tolist())
    return hook
for i, layer in enumerate(model.model.layers):
    layer.post_attention_layernorm.register_forward_hook(make_hook(i))

t0 = time.time()
with torch.no_grad():
    out = model(ids)
log(f"forward ({time.time()-t0:.0f}s)")

lg = out.logits[0, -1]
top5 = torch.topk(lg, 5)
log("CPU logits top5: " + " ".join(f"{idx.item()}:{val.item():.2f}" for idx, val in zip(top5.indices, top5.values)))
for i in range(num_layers):
    s, f8 = norm_stats.get(i, (0.0, []))
    log(f"CPU L{i}.normed std={s:.4f} first8={f8}")
