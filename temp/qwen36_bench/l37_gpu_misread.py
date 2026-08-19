import json, numpy as np
lay = json.load(open('prime-agent-worktrees/qwen36-r4.gturbo/packed_experts/layout.json'))
H, INTER, GROUP, NG = 2048, 512, 64, 32
def bf16_to_f32(bits16):
    return (bits16.astype(np.uint32) << 16).view(np.float32)
def read_expert(L, E):
    e = lay['layers'][L]['experts'][E]; off = e['offset']; t = e['tensors']
    data = np.fromfile(f'prime-agent-worktrees/qwen36-r4.gturbo/packed_experts/layer_{L:02d}.bin', dtype=np.uint8, count=lay['expertStride'], offset=off)
    return data, t
def gpu_scale_read(data, t, name_sc, name_bs, rows, ngroups):
    # kernel reads (float*) from scales offset: each float = 2 consecutive BF16 (scale, bias pairs)
    sc = data[t[name_sc]['offset']:t[name_sc]['offset']+t[name_sc]['size']+t[name_bs]['size']].view(np.float32)
    return sc  # [rows*ngroups] floats
def q8(data, t, name, rows, cols):
    q = np.zeros((rows, cols), dtype=np.int32)
    packed = data[t[name]['offset']:t[name]['offset']+t[name]['size']].reshape(rows, cols//2)
    for k in range(cols):
        byte = packed[:, k//2]
        q[:, k] = (byte & 0x0F) if k % 2 == 0 else (byte >> 4)
    return q.astype(np.float32)
rng = np.random.default_rng(0)
x = rng.standard_normal(H).astype(np.float32)
ids37 = [224,141,39,124,240,195,8,85]
outs = []
for E in ids37:
    data, t = read_expert(37, E)
    gs = gpu_scale_read(data, t, 'gate_scales','gate_biases', 512, 32)
    us = gpu_scale_read(data, t, 'up_scales','up_biases', 512, 32)
    ds = gpu_scale_read(data, t, 'down_scales','down_biases', 2048, 8)
    gate = q8(data, t, 'gate', 512, H); up = q8(data, t, 'up', 512, H); down = q8(data, t, 'down', H, INTER)
    # kernel: accG = sum_c q*gs[row*NG+g]*x  (NO bias)
    accG = np.zeros(512, dtype=np.float32); accU = np.zeros(512, dtype=np.float32)
    for r in range(512):
        gsrow = gs[r*NG:(r+1)*NG]; usrow = us[r*NG:(r+1)*NG]
        gidx = np.arange(H)//GROUP
        accG[r] = np.sum(gate[r]*gsrow[gidx]*x); accU[r] = np.sum(up[r]*usrow[gidx]*x)
    acts = np.log1p(np.exp(accG)) * accU
    out = np.zeros(H, dtype=np.float32)
    for r in range(H):
        dsrow = ds[r*8:(r+1)*8]
        gidx = np.arange(INTER)//GROUP
        out[r] = np.sum(down[r]*dsrow[gidx]*acts)
    outs.append(out)
    print(f'  E{E}: gate_acc std={accG.std():.3f} up_acc std={accU.std():.3f} out std={out.std():.4f}')
O = np.stack(outs)
print(f'L37 GPU-misread expertOut: std={O.std():.4f} absmax={np.abs(O).max():.4f}')
