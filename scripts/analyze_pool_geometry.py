#!/usr/bin/env python3
"""Why F16 MTP weights starved the expert pool.

Usage:  python3 scripts/analyze_pool_geometry.py [model.gguf] [budget_bytes]

Reproduces the arithmetic behind the whitepaper's section 5: the L4 pool's
per-slot geometry, and how the MTP layer's storage type changes the TRUNK's slot
count. Both arms printed below match the slot counts in the real server logs
(118 for the shipped artifact, 33 for the F16-head artifact that measured 0%
MTP accept) -- which is what validates the formula.

compute_l4_pool_capacity (llama-model-loader.cpp:1068) does:

    per_slot_layer[il] = sum of (row_size(type, ne0) * ne1) over that layer's *_exps tensors
    per_slot          = MAX over ALL layers            <-- includes blk.40 (the MTP layer)
    capacity          = clamp(budget / (n_layers * per_slot), 8, 256)

So the trunk's slot count is charged the *worst* layer's per-slot bytes. If the MTP layer
carries F16 experts while the trunk is IQ4_XS/Q2_K, blk.40 alone inflates per_slot and the
trunk's capacity collapses -- the pool is then too small for the trunk's top-k unions, the
FFN gets nil buffers, and every drafted token is rejected.

This script reads the real metadata and prints both arms. It never writes to the model.
"""
import os
import sys

import gguf
from gguf.constants import GGML_QUANT_SIZES

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _REPO, 'models/gguf/Edge0-35B-Q4_0-MTP-edge0head.gguf')
# run_server.sh:277 -- BUDGET_DEFAULT = 8 GiB expert pool
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 8589934592

# Types the F16 MTP head carried before requantizing (npz is fp16/fp32).
F16, F32 = gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32


def row_size(t, n):                # ggml_row_size
    blk, ts = GGML_QUANT_SIZES[t]
    return (n // blk) * ts


def collect(reader):
    per_layer = {}
    tensors = {}
    for t in reader.tensors:
        n = t.name
        if '_exps' not in n or 'blk.' not in n:
            continue
        il = int(n.split('blk.')[1].split('.')[0])
        # gguf-py's ReaderTensor.shape IS the raw stored dims = ggml order (ne[0] is the row),
        # i.e. exactly what llama-model-loader.cpp feeds ggml_row_size. Do NOT reverse it.
        ggml_ne = [int(x) for x in t.shape]
        b = row_size(t.tensor_type, ggml_ne[0]) * ggml_ne[1]
        per_layer[il] = per_layer.get(il, 0) + b
        tensors[n] = (t.tensor_type, ggml_ne, b)
    return per_layer, tensors


def capacity(per_layer, n_layers):
    per_slot = max(per_layer.values())
    denom = n_layers * per_slot
    base = max(8, min(256, BUDGET // denom if denom else 256))
    return base, per_slot


def main():
    global BUDGET
    r = gguf.GGUFReader(MODEL)
    per_layer, tensors = collect(r)
    n_layers = max(per_layer) + 1
    arch = r.fields['general.architecture'].contents()
    if isinstance(arch, bytes):
        arch = arch.decode('utf-8', 'replace')
    print(f'model   : {os.path.basename(MODEL)}')
    print(f'arch    : {arch}')
    print(f'n_layers: {n_layers}  (blk.40 present -> 41 in the denominator, as in the loader)')

    trunk_max = max((v, k) for k, v in per_layer.items() if k != 40)
    print(f'\nper-slot bytes (sum of that layer\'s *_exps row*ne1, one expert):')
    print(f'  blk.40 (MTP)      : {per_layer[40]:>12,} B  ({per_layer[40]/2**20:.3f} MiB)')
    print(f'  worst trunk layer : {trunk_max[0]:>12,} B  ({trunk_max[0]/2**20:.3f} MiB)  (blk.{trunk_max[1]})')
    print(f'  ratio             : {per_layer[40]/trunk_max[0]:.3f}x')

    print(f'\nblk.40 *_exps storage types in this file:')
    for n, (t, ne, b) in sorted(tensors.items()):
        if n.startswith('blk.40'):
            print(f'  {n:<28} {t.name:<8} ne={ne}  {b:>11,} B/expert')

    # Solve for the budget that reproduces the observed 118 slots, then replay both arms.
    if BUDGET is None:
        BUDGET = 118 * n_layers * trunk_max[0]   # anchor on the quantized arm's measurement
    print(f'\nbudget (expert_cache_bytes): {BUDGET:,} B ({BUDGET/2**30:.2f} GiB)')

    cap_q, ps_q = capacity(per_layer, n_layers)
    print(f'\nARM A -- quantized blk.40 (as shipped now):')
    print(f'  per_slot = max over layers = {ps_q:,} B')
    print(f'  capacity = budget/(41*per_slot) = {cap_q} slots/layer   <-- observed 118')

    # ARM B: only blk.40's experts changed to F16 (the artifact before requant).
    per_layer_f16 = dict(per_layer)
    for n, (t, ne, _) in tensors.items():
        if n.startswith('blk.40') and t not in (F32,):
            per_layer_f16[40] = per_layer_f16[40] - row_size(t, ne[0]) * ne[1] \
                                + row_size(F16, ne[0]) * ne[1]
    cap_f, ps_f = capacity(per_layer_f16, n_layers)
    print(f'\nARM B -- blk.40 experts at F16 (the artifact that measured 0% accept):')
    print(f'  blk.40 per-slot   = {per_layer_f16[40]:,} B ({per_layer_f16[40]/2**20:.3f} MiB)')
    print(f'  per_slot = max    = {ps_f:,} B')
    print(f'  capacity          = {cap_f} slots/layer   <-- observed 33')
    print(f'  trunk slots lost  : {cap_q} -> {cap_f}  ({100*(cap_q-cap_f)/cap_q:.0f}% fewer)')

    # ARM C: what the fix in llama-model-loader.cpp computes -- per_slot is a max over DECODER
    # layers only (NextN/MTP layers get their slots from LAYER_CAPS instead).
    per_slot_c = trunk_max[0]
    cap_c = max(8, min(256, BUDGET // (n_layers * per_slot_c)))
    print(f'\nARM C -- with the fix (MTP excluded from per_slot), per_slot = {per_slot_c:,} B:')
    print(f'  capacity = {cap_c} slots/layer')
    print(f'  on the shipped artifact the fix is a no-op: {cap_c} == {cap_q}'
          f'  --> {"IDENTICAL" if cap_c == cap_q else "CHANGED"}')


if __name__ == '__main__':
    main()
