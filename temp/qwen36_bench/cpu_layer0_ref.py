#!/usr/bin/env python3
"""Layer-0-only CPU reference: embed + input_layernorm + linear_attn + residual
+ post_attention_layernorm for the EXACT 13 GPU token ids. No MoE needed for the
post-attn hook. Cheap on memory so it survives desktop memory pressure."""
import os, sys, time
sys.path.insert(0, "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-repack")
from pathlib import Path
from resident_writer import read_resident_bin as _read_resident
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M

GTURBO = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-r4.gturbo"
GPU_IDS = [248045, 846, 198, 760, 6511, 314, 9338, 369, 248046, 198, 248045, 74455, 198]
OUT = "/tmp/cpu_layer0_ref.log"

def log(m):
    print(m, flush=True)
    with open(OUT, "a") as f:
        f.write(m + "\n")

log("=== layer0 ref start ===")
cfg = AutoConfig.from_pretrained("/Volumes/AlexZhuang/qwen36-hf", trust_remote_code=True)
text = cfg.text_config

# Build ONLY the layer-0 submodules with real weights (load dense state dict)
info = _read_resident(Path(GTURBO) / "model_weights.bin")
sd = {}
with open(os.path.join(GTURBO, "model_weights.bin"), "rb") as f:
    for name, e in info["entries"].items():
        f.seek(e["fileOffset"])
        blob = f.read(e["sizeBytes"])
        shp = [s for s in e["shape"] if s != 0]
        key = name.replace("model.language_model.", "model.", 1)
        if e["dtype"] == 1:
            arr = np.frombuffer(blob, dtype=np.float16).reshape(shp).astype(np.float32)
            sd[key] = torch.from_numpy(arr).to(torch.bfloat16)
        elif e["dtype"] == 2:
            sd[key] = torch.from_numpy(np.frombuffer(blob, dtype=np.float32).reshape(shp)).to(torch.bfloat16)
        else:
            sd[key] = torch.from_numpy(np.frombuffer(blob, dtype=np.uint32).reshape(shp))
log(f"dense loaded ({len(sd)} keys)")

# Sub-modules we need (layer 0):
emb = torch.nn.Embedding(text.vocab_size, text.hidden_size).to(torch.bfloat16)
emb.weight.data.copy_(sd["model.embed_tokens.weight"])

lin = M.Qwen3_5MoeGatedDeltaNet(text, 0).to(torch.bfloat16)
# Load its state dict (strip prefix)
lin_sd = {k.replace("model.layers.0.linear_attn.", "", 1): v
          for k, v in sd.items() if k.startswith("model.layers.0.linear_attn.")}
lin.load_state_dict(lin_sd, strict=True)
log("linear_attn loaded")

def rmsnorm(x, w, eps):
    x = x.float()
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + w.float())).type_as(x)

w_in = sd["model.layers.0.input_layernorm.weight"]
w_post = sd["model.layers.0.post_attention_layernorm.weight"]

ids = torch.tensor([GPU_IDS], dtype=torch.long)
h = emb(ids)                      # [1,13,2048] bf16
h_in = rmsnorm(h, w_in, text.rms_norm_eps)
lin = lin.to(torch.float32)
h_in32 = h_in.float()

# hook the linear_attn internals (must be BEFORE forward)
inter = {}
def mk_hook(name):
    def hook(mod, inp, out):
        inter[name] = out[0] if isinstance(out, tuple) else out
    return hook
lin.in_proj_qkv.register_forward_hook(mk_hook('qkv'))
lin.conv1d.register_forward_hook(mk_hook('conv'))
lin.in_proj_z.register_forward_hook(mk_hook('z'))
lin.in_proj_b.register_forward_hook(mk_hook('b'))
lin.in_proj_a.register_forward_hook(mk_hook('a'))
lin.norm.register_forward_hook(mk_hook('norm'))
lin.out_proj.register_forward_hook(mk_hook('outproj'))

attn_out = lin(hidden_states=h_in32, cache_params=None, attention_mask=None)
attn_out = attn_out.to(torch.bfloat16)
h = h + attn_out
h_post = rmsnorm(h, w_post, text.rms_norm_eps)

def flat8(t):
    t = t.detach().float()
    if t.dim() == 3:
        t = t[0, 0]
    elif t.dim() == 2:
        t = t[0]
    return float(t.std()), np.round(t[:8].numpy(), 3).tolist()

with torch.no_grad():
    for pos in (0, 12):
        n = h_post[0, pos].float().detach()
        log(f"CPU L0.normed pos={pos} std={float(n.std()):.4f} first8={np.round(n[:8].numpy(), 3).tolist()}")
        a = attn_out[0, pos].float().detach()
        log(f"CPU L0.attnOut pos={pos} std={float(a.std()):.4f} first8={np.round(a[:8].numpy(), 3).tolist()}")
for name in ('qkv', 'z', 'b', 'a', 'norm', 'outproj'):
    if name in inter:
        for pos in (0, 12):
            s, f8 = flat8(inter[name][:, pos])
            log(f"CPU 0.{name} pos={pos} std={s:.4f} first8={f8}")
if 'conv' in inter:
    conv_silu = torch.nn.functional.silu(inter['conv'][:, :, :13])
    for pos in (0, 12):
        s, f8 = flat8(conv_silu[:, pos])
        log(f"CPU 0.qkvPost pos={pos} std={s:.4f} first8={f8}")
log("=== layer0 ref done ===")
