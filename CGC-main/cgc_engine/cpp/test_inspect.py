#!/usr/bin/env python3
"""Inspect qwen35moe model structure."""

import sys
sys.path.insert(0, "D:/alex/flashkv0516/app/edge_engine")
from llama_monkey_patch import parse_gguf_header

header = parse_gguf_header(r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf")

print("=== First 30 tensors ===")
for t in header["tensors"][:30]:
    print(f"  {t['name']}: dims={t['dims']}")

print("\n=== Tensors with 'blk' ===")
blk_tensors = [t for t in header["tensors"] if "blk" in t["name"]]
for t in blk_tensors[:25]:
    print(f"  {t['name']}: dims={t['dims']}")

# Count unique patterns
patterns = {}
for t in header["tensors"]:
    parts = t["name"].split(".")
    if len(parts) >= 3:
        prefix = ".".join(parts[:3])
        patterns[prefix] = patterns.get(prefix, 0) + 1
    elif len(parts) >= 2:
        prefix = ".".join(parts[:2])
        patterns[prefix] = patterns.get(prefix, 0) + 1

print("\n=== Name patterns (top 20) ===")
for p, count in sorted(patterns.items(), key=lambda x: -x[1])[:20]:
    print(f"  {p}: {count}")

print("\n=== KV keys ===")
for k, v in sorted(header["kv"].items()):
    if isinstance(v, str) and len(v) > 100:
        v = v[:100] + "..."
    print(f"  {k} = {v}")