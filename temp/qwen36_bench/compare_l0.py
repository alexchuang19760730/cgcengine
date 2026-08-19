#!/usr/bin/env python3
"""Compute token-0 L0 input_layernorm output (CPU, bf16) and compare with GPU dump."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-repack")
from pathlib import Path
from resident_writer import read_resident_bin as _read_resident
from transformers import AutoTokenizer

GTURBO = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-r4.gturbo"
def load_embed_norm():
    info = _read_resident(Path(GTURBO) / "model_weights.bin")
    ents = info['entries']
    out = {}
    with open(f"{GTURBO}/model_weights.bin", "rb") as f:
        for name in ["model.language_model.embed_tokens.weight",
                     "model.language_model.layers.0.input_layernorm.weight",
                     "model.language_model.layers.0.post_attention_layernorm.weight"]:
            e = ents[name]
            f.seek(e['fileOffset'])
            blob = f.read(e['sizeBytes'])
            shp = [s for s in e['shape'] if s != 0]
            arr = np.frombuffer(blob, dtype=np.float16).reshape(shp).astype(np.float32)
            out[name] = torch.from_numpy(arr).to(torch.bfloat16)
    return out

def rmsnorm(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

tok = AutoTokenizer.from_pretrained(f"{GTURBO}/tokenizer", trust_remote_code=True)
prompt_ids = tok.encode("The capital of France is", add_special_tokens=False)
print("prompt tokens:", prompt_ids, len(prompt_ids))
w = load_embed_norm()
emb = w["model.language_model.embed_tokens.weight"]
iln = w["model.language_model.layers.0.input_layernorm.weight"]
t = prompt_ids[0]
h = emb[t].float()
print("L0 embed first8:", np.round(h[:8].numpy(), 3).tolist())
n = rmsnorm(h, iln.float(), 1e-6)
print("L0 normed std:", float(n.std()))
print("L0 normed first8:", np.round(n[:8].numpy(), 3).tolist())
