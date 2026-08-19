#!/usr/bin/env python3
"""Quick test: simulate the ARRAY metadata fix using Nail's rope.dimension_sections."""
import sys
sys.path.insert(0, '/opt/homebrew/lib/python3.13/site-packages')
from gguf import GGUFReader, GGUFValueType

NAIL = '/Users/alexchuang/Documents/flashkv0516/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf'
r = GGUFReader(NAIL)

# Find rope.dimension_sections by iterating (r.fields[key] may KeyError due to bytes/str)
for k, field in r.fields.items():
    ks = k.decode() if isinstance(k, bytes) else k
    if ks != 'qwen35moe.rope.dimension_sections':
        continue
    print(f'Found: {ks}')
    print(f'  types: {list(field.types)}')
    print(f'  data: {list(field.data)} (len={len(field.data)})')

    # OLD logic (buggy): skip data[0]
    old_elems = []
    for pi in field.data[1:]:
        arr = field.parts[pi]
        old_elems.append(arr.item() if arr.size == 1 else arr.tolist())
    print(f'  OLD (data[1:]): {old_elems}  len={len(old_elems)}')

    # NEW logic (fixed): iterate all data
    new_elems = []
    for pi in field.data:
        arr = field.parts[pi]
        new_elems.append(arr.item() if arr.size == 1 else arr.tolist())
    print(f'  NEW (all data): {new_elems}  len={len(new_elems)}')

    # Expected
    print(f'  Expected: [11, 11, 10, 0]  len=4')
    break
