#!/usr/bin/env python3
"""Test that we correctly get general.architecture and read tensors."""

import sys
sys.path.insert(0, "D:/alex/flashkv0516/app/edge_engine")
from llama_monkey_patch import parse_gguf_header

model_path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
header = parse_gguf_header(model_path)

print(f"GGUF version: {header['version']}")
print(f"KV count (from header): {header['n_kv']}")
print(f"Actual KV items read: {len(header['kv'])}")

print(f"\nKV keys read:")
for k in sorted(header['kv'].keys()):
    print(f"  {k}")

print(f"\nTensor count (from header): {header['n_tensors']}")
print(f"Actual tensors read: {len(header['tensors'])}")

if header['tensors']:
    print(f"\nFirst 10 tensors:")
    for t in header['tensors'][:10]:
        print(f"  {t['name']}: dims={t['dims']}")