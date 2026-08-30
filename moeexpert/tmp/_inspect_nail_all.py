#!/usr/bin/env python3
"""Dump all Nail model fields with raw byte names."""
import sys
sys.path.insert(0, '/opt/homebrew/lib/python3.13/site-packages')
from gguf import GGUFReader

NAIL = '/Users/alexchuang/Documents/flashkv0516/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf'
r = GGUFReader(NAIL)
print(f'total fields: {len(r.fields)}')
print()
for k, f in r.fields.items():
    if isinstance(k, bytes):
        ks = k.decode('utf-8', errors='replace')
        # Show repr to catch hidden chars
        kr = repr(k)
    else:
        ks = k
        kr = k
    if 'rope' in ks.lower() or 'dimension' in ks.lower() or 'qwen35moe' in ks.lower():
        print(f'key: {ks!r}  bytes={kr}')
        print(f'  types={list(f.types)} data_len={len(f.data)}')
        for i, pi in enumerate(f.data):
            arr = f.parts[pi]
            v = arr.item() if arr.size == 1 else arr.tolist()
            print(f'  data[{i}]: {v}')
