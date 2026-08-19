#!/usr/bin/env python3
"""Inspect graft output metadata to diagnose the rope.dimension_sections error."""
import sys
sys.path.insert(0, '/opt/homebrew/lib/python3.13/site-packages')
from gguf import GGUFReader

OUT = '/Users/alexchuang/Documents/flashkv0516/models/gguf/Qwen3.6-35B-A3B-IQ3XXS-trunk_Q4K-blk40.gguf'
g = GGUFReader(OUT)
print(f'total fields: {len(g.fields)}')
print()
print('=== qwen35moe.* / rope / dimension fields in graft output ===')
for k, f in g.fields.items():
    ks = k.decode() if isinstance(k, bytes) else k
    if ks.startswith('qwen35moe.') or 'rope' in ks or 'dimension' in ks:
        print(f'[{ks}] types={list(f.types)} data_len={len(f.data)}')
        for i, pi in enumerate(f.data):
            arr = f.parts[pi]
            v = arr.item() if arr.size == 1 else arr.tolist()
            print(f'  data[{i}]: {v}')
