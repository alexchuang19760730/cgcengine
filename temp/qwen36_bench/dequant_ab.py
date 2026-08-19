import json, numpy as np, torch
from pathlib import Path

def load_layer(lay, lay_idx, expert):
    lay_meta = lay['layers'][lay_idx]
    e = lay_meta['experts'][expert]
    off = e['offset']
    # tensors inside expert
    t = e['tensors']
    return off, t

def read_expert(layer_path, lay, lay_idx, expert):
    off, t = load_layer(lay, lay_idx, expert)
    data = np.fromfile(layer_path, dtype=np.uint8, count=lay['expertStride'], offset=off)
    gq = data[t['gate']['offset']:t['gate']['offset']+t['gate']['size']]          # [512,1024] u8 int4
    gs = data[t['gate_scales']['offset']:t['gate_scales']['offset']+t['gate_scales']['size']].view(np.float32)
    uq = data[t['up']['offset']:t['up']['offset']+t['up']['size']]
    us = data[t['up_scales']['offset']:t['up_scales']['offset']+t['up_scales']['size']].view(np.float32)
    dq = data[t['down']['offset']:t['down']['offset']+t['down']['size']]
    ds = data[t['down_scales']['offset']:t['down_scales']['offset']+t['down_scales']['size']].view(np.float32)
    return gq, gs, uq, us, dq, ds

def dequant4(q, col_odd):
    # int4: even byte = low nibble, odd = high nibble; values 0..15, sign-extend 8
    v = (q >> (4*col_odd)) & 0xF
    return (v ^ 8) - 8  # sign extend 4-bit to int8

def expert_ffn(gq, gs, uq, us, dq, ds, x, group=64):
    H, INTER, NG = 2048, 512, 32
    # gate/up: rows=INTER
    accG = np.zeros(INTER, dtype=np.float64); accU = np.zeros(INTER, dtype=np.float64)
    for r in range(INTER):
        row = gq[r*1024:(r+1)*1024]; usrow = us[r*NG:(r+1)*NG]; gsrow = gs[r*NG:(r+1)*NG]
        s = 0.0; u = 0.0
        for c in range(0, H, 2):
            b = row[c//2]
            g = c//group
            xc = x[c]; xc1 = x[c+1]
            s += float(dequant4(b,0)) * gsrow[g] * xc + float(dequant4(b,1)) * gsrow[g] * xc1
            u += float(dequant4(b,0)) * usrow[g] * xc + float(dequant4(b,1)) * usrow[g] * xc1
        accG[r] = s; accU[r] = u
    acts = np.log1p(np.exp(accG)) * accU
    # down: out rows H
    out = np.zeros(H, dtype=np.float64)
    for r in range(H):
        row = dq[r*256:(r+1)*256]; dsrow = ds[r*8:(r+1)*8]
        s = 0.0
        for i in range(0, INTER, 2):
            b = row[i//2]
            g = i//group
            s += float(dequant4(b,0)) * dsrow[g] * acts[i] + float(dequant4(b,1)) * dsrow[g] * acts[i+1]
        out[r] = s
    return accG, accU, out

lay = json.load(open('prime-agent-worktrees/qwen36-r4.gturbo/packed_experts/layout.json'))
rng = np.random.default_rng(0)
x = rng.standard_normal(2048).astype(np.float64)
for L in (36, 37):
    for E in (224, 141, 0):
        gq,gs,uq,us,dq,ds = read_expert(f'prime-agent-worktrees/qwen36-r4.gturbo/packed_experts/layer_{L:02d}.bin', lay, L, E)
        accG, accU, out = expert_ffn(gq,gs,uq,us,dq,ds,x)
        print(f'L{L} E{E}: gate acc std={accG.std():.4f} up acc std={accU.std():.4f} out std={out.std():.4f} out max={np.abs(out).max():.4f}')
