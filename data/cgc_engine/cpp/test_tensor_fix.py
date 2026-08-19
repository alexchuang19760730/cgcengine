#!/usr/bin/env python3
"""Quick test of fixed tensor parsing for qwen3.6 MoE."""

import sys
sys.path.insert(0, r"D:\alex\flashkv0516\app\edge_engine")
from llama_monkey_patch import parse_gguf_header

header = parse_gguf_header(r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf")
print("Version:", header["version"])
print("KV items:", len(header["kv"]))
print("Tensor items:", len(header["tensors"]))
print("Data start:", header["data_start"])

for t in header["tensors"][:5]:
    print(f"  {t['name']}: dims={t['dims']} type={t['type']} offset={t['offset']}")

expert_count = sum(1 for t in header["tensors"] if "expert" in t["name"].lower())
print(f"Expert tensors: {expert_count}")

# Check first and last expert tensors
expert_tensors = [t for t in header["tensors"] if "expert" in t["name"].lower()]
if expert_tensors:
    print(f"\nFirst expert: {expert_tensors[0]['name']}")
    print(f"Last expert: {expert_tensors[-1]['name']}")
    
    # Check expert structure: blk.X.expert.Y.role
    roles = set()
    layers = set()
    eids = set()
    for t in expert_tensors:
        parts = t["name"].split(".")
        if len(parts) >= 5:
            try:
                layers.add(int(parts[1]))
                eids.add(int(parts[3]))
                roles.add(parts[4])
            except (ValueError, IndexError):
                pass
    
    print(f"\nExpert layers: {sorted(layers)}")
    print(f"Expert IDs (first 10): {sorted(eids)[:10]}")
    print(f"Expert roles: {sorted(roles)}")
    print(f"Total unique experts: {len(eids)}")

if len(header["tensors"]) > 0:
    print("\n[SUCCESS] Tensor parsing works!")
else:
    print("\n[FAIL] No tensors parsed")