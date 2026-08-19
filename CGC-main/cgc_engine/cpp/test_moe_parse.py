#!/usr/bin/env python3
"""Parse qwen35moe model with fixed parser."""

import sys
sys.path.insert(0, "D:/alex/flashkv0516/app/edge_engine")
from llama_monkey_patch import parse_gguf_header, ExpertStreamerLite

model_path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
print(f"Parsing: {model_path}")

header = parse_gguf_header(model_path)
print(f"GGUF version: {header['version']}")
print(f"Tensor count: {header['n_tensors']}")
print(f"KV count: {header['n_kv']}")

kv = header['kv']
arch = kv.get("general.architecture", "unknown")
print(f"Architecture: {arch}")

# Count expert tensors
expert_tensors = [t for t in header['tensors'] if 'expert' in t['name'].lower()]
print(f"Expert tensor count: {len(expert_tensors)}")
if expert_tensors:
    print("Sample expert tensors:")
    for t in expert_tensors[:10]:
        print(f"  {t['name']}: dims={t['dims']} type={t['type']}")

# Find any tensor with 'blk' and numeric suffix
blk_tensors = [t for t in header['tensors'] if 'blk' in t['name']]
print(f"\nAll tensors with 'blk': {len(blk_tensors)}")
for t in blk_tensors[:15]:
    print(f"  {t['name']}: dims={t['dims']}")

# Try streamer
streamer = ExpertStreamerLite(model_path)
stats = streamer.cache_stats()
print(f"\nExpertStreamerLite:")
print(f"  Architecture: {stats['architecture']}")
print(f"  Hidden: {stats['hidden']}")
print(f"  Inter: {stats['inter']}")
print(f"  Num experts: {stats['num_experts']}")
print(f"  Has experts: {stats['has_experts']}")
print(f"  Num layers: {stats['num_layers']}")
if stats['has_experts']:
    eids = streamer.list_experts()
    print(f"  Expert IDs: {eids[:8]}{'...' if len(eids) > 8 else ''}")
    print(f"  Total experts: {len(eids)}")

print("\n[OK] Verification complete!")