import json, numpy as np
lay = json.load(open('prime-agent-worktrees/qwen36-r4.gturbo/packed_experts/layout.json'))
def bf16_to_f32(bits16):
    return (bits16.astype(np.uint32) << 16).view(np.float32)
def read_expert(L, E):
    e = lay['layers'][L]['experts'][E]; off = e['offset']; t = e['tensors']
    data = np.fromfile(f'prime-agent-worktrees/qwen36-r4.gturbo/packed_experts/layer_{L:02d}.bin', dtype=np.uint8, count=lay['expertStride'], offset=off)
    def r(name):
        b = data[t[name]['offset']:t[name]['offset']+t[name]['size']]
        return b
    return r
def dequant(q8, sc16, bs16, cols, group):
    rows = q8.shape[0]
    sc = bf16_to_f32(sc16.view(np.uint16)); bs = bf16_to_f32(bs16.view(np.uint16))
    sc = np.repeat(sc, group).reshape(rows, cols); bs = np.repeat(bs, group).reshape(rows, cols)
    q = np.zeros((rows, cols), dtype=np.int32)
    for k in range(cols):
        byte = q8[:, k//2]
        q[:, k] = (byte & 0x0F) if k % 2 == 0 else (byte >> 4)
    return q.astype(np.float32)*sc + bs
H, INTER, GROUP = 2048, 512, 64
rng = np.random.default_rng(0)
x = rng.standard_normal(H).astype(np.float32)
ids37 = [224,141,39,124,240,195,8,85]
ids36 = [224,141,39,124,240,195,8,85]  # same ids for comparison
for L, ids in ((36, ids36), (37, ids37)):
    outs = []
    for E in ids:
        r = read_expert(L, E)
        gq = r('gate').reshape(-1, H//2); gs = r('gate_scales'); gb = r('gate_biases')
        uq = r('up').reshape(-1, H//2); us = r('up_scales'); ub = r('up_biases')
        dq = r('down').reshape(-1, INTER//2); ds = r('down_scales'); db = r('down_biases')
        gate = dequant(gq, gs, gb, H, GROUP)
        up   = dequant(uq, us, ub, H, GROUP)
        down = dequant(dq, ds, db, INTER, GROUP)
        acts = np.log1p(np.exp(gate @ x)) * (up @ x)
        out = down @ acts
        outs.append(out)
    O = np.stack(outs)
    print(f'L{L}: expertOut std={O.std():.4f} absmax={np.abs(O).max():.4f} per-expert std={[round(float(o.std()),2) for o in outs]}')
