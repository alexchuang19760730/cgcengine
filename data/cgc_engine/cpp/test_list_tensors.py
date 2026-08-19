#!/usr/bin/env python3
"""List all tensors with 'exps' or 'expert' in name."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    # Skip to tensor info
    f.seek(10943832)
    
    n_tensors = 733
    
    tensors = []
    for idx in range(n_tensors):
        nlen_raw = f.read(8)
        if len(nlen_raw) < 8:
            break
        nlen = struct.unpack("<Q", nlen_raw)[0]
        if nlen == 0 or nlen > 256:
            break
        name = f.read(nlen).decode("utf-8", errors="replace")
        
        n_dims_raw = f.read(4)
        n_dims = struct.unpack("<I", n_dims_raw)[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
        
        ggml_type = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]
        
        tensors.append({"name": name, "dims": dims, "type": ggml_type, "offset": offset})
    
    print(f"Total tensors: {len(tensors)}")
    
    # Find tensors with exps or expert
    print(f"\n=== Expert/exps tensors ===")
    expert_related = [t for t in tensors if "exps" in t["name"].lower() or "expert" in t["name"].lower()]
    print(f"Found {len(expert_related)} tensors with 'exps' or 'expert'")
    for t in expert_related[:20]:
        print(f"  {t['name']}: dims={t['dims']} type={t['type']}")
    
    # Unique patterns
    patterns = set()
    for t in tensors:
        parts = t["name"].split(".")
        if len(parts) >= 3:
            pattern = ".".join(parts[2:])  # role pattern
            patterns.add(pattern)
    
    print(f"\n=== Unique tensor role patterns ===")
    for p in sorted(patterns):
        count = sum(1 for t in tensors if ".".join(t["name"].split(".")[2:]) == p)
        print(f"  [{count:3d}x] {p}")
    
    # Specifically for expert layers
    print(f"\n=== Layer-by-layer expert analysis ===")
    for layer in range(min(5, 40)):
        layer_tensors = [t for t in tensors if f"blk.{layer}." in t["name"]]
        expert_tensors = [t for t in layer_tensors if "exps" in t["name"]]
        print(f"blk.{layer}: {len(layer_tensors)} tensors, {len(expert_tensors)} expert-related")
        for t in expert_tensors:
            print(f"  {t['name']}: dims={t['dims']}")