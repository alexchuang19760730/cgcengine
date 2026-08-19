#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/gguf-py')
from gguf import GGUFReader

def tname(t):
    return t.name if isinstance(t.name, str) else t.name.decode()

fraQtl = GGUFReader('/Users/alexchuang/Documents/flashkv0516/models/gguf/fraQtl/qwen36-35b-a3b-hi-fi-mtp-runtime.gguf')
nail = GGUFReader('/Users/alexchuang/Documents/flashkv0516/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf')

def build_dict(r):
    d = {}
    for t in r.tensors:
        d[tname(t)] = (list(t.shape), t.tensor_type)
    return d

f_dict = build_dict(fraQtl)
n_dict = build_dict(nail)

f_blk40 = {k: v for k, v in f_dict.items() if k.startswith('blk.40.')}
n_blk40 = {k: v for k, v in n_dict.items() if k.startswith('blk.40.')}

print(f'fraQtl blk.40: {len(f_blk40)} tensors')
print(f'Nail    blk.40: {len(n_blk40)} tensors')
print()
hdr = f'{"name":60s} {"shape":30s} {"fQ-type":8s} {"nail-type":8s} {"match":6s}'
print(hdr)
print('-' * len(hdr))
for name in sorted(set(list(f_blk40.keys()) + list(n_blk40.keys()))):
    f_s, f_t = f_blk40.get(name, (None, None))
    n_s, n_t = n_blk40.get(name, (None, None))
    shape_str = str(f_s or n_s)
    match = 'OK' if (f_s == n_s) else 'MISMATCH'
    print(f'{name:60s} {shape_str:30s} {str(f_t):8s} {str(n_t):8s} {match:6s}')
