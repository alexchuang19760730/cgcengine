#!/usr/bin/env python3
"""Detailed inspection of Nail's rope.dimension_sections field structure."""
import sys
sys.path.insert(0, '/opt/homebrew/lib/python3.13/site-packages')
from gguf import GGUFReader

NAIL = '/Users/alexchuang/Documents/flashkv0516/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf'
r = GGUFReader(NAIL)

key = b'qwen35moe.rope.dimension_sections'
f = r.fields[key]
print(f'key: {key.decode()}')
print(f'types: {list(f.types)}  len={len(f.types)}')
print(f'data: {list(f.data)}  len={len(f.data)}')
print(f'parts count: {len(f.parts)}')
print()
for i, (t, pi) in enumerate(zip(f.types, f.data)):
    arr = f.parts[pi]
    v = arr.item() if arr.size == 1 else arr.tolist()
    print(f'  zip[{i}]  type={t}  parts_idx={pi}  shape={arr.shape} dtype={arr.dtype} value={v}')

print()
print('--- raw parts list ---')
for i, arr in enumerate(f.parts):
    v = arr.item() if arr.size == 1 else arr.tolist()
    print(f'  parts[{i}]: shape={arr.shape} dtype={arr.dtype} value={v}')

print()
print('--- iterate field.data[1:] ---')
for i, pi in enumerate(f.data[1:]):
    arr = f.parts[pi]
    v = arr.item() if arr.size == 1 else arr.tolist()
    print(f'  data[1:][{i}] parts_idx={pi} value={v}')
