#!/usr/bin/env python3
"""Test the updated llama_monkey_patch parse_gguf_header function."""
import sys
sys.path.insert(0, r"D:\alex\flashkv0516\app\edge_engine")

# Import the updated function
from llama_monkey_patch import parse_gguf_header

# Test with qwen3.6 A3B model
model_path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

print("Testing parse_gguf_header with official gguf library:")
print("=" * 60)

result = parse_gguf_header(model_path)

print(f"\nHeader:")
print(f"  version: {result['version']}")
print(f"  n_tensors: {result['n_tensors']}")
print(f"  n_kv: {result['n_kv']}")
print(f"  data_start: {result['data_start']} (0x{result['data_start']:X})")

print(f"\nKV metadata:")
kv = result['kv']
for key in sorted(kv.keys()):
    val = kv[key]
    val_str = str(val)
    if len(val_str) > 100:
        val_str = val_str[:100] + "..."
    print(f"  {key}: {val_str} (type={type(val).__name__})")

print(f"\nArchitecture: {kv.get('general.architecture', 'N/A')}")

# Check key architecture params
arch = kv.get('general.architecture', '')
print(f"\nArchitecture-specific parameters:")
for prefix in [arch, 'gemma4', 'gemma', 'qwen2', 'llama']:
    for key in ['hidden_size', 'embedding_length', 'intermediate_size', 'feed_forward_length',
                'moe_intermediate_size', 'expert_count', 'num_experts', 'block_count', 'num_layers']:
        full_key = f"{prefix}.{key}"
        if full_key in kv:
            print(f"  {full_key}: {kv[full_key]}")

# Check first few tensors
print(f"\nFirst 10 tensors:")
for i, t in enumerate(result['tensors'][:10]):
    print(f"  [{i}] '{t['name']}' dims={t['dims']} type={t['type']} offset={t['offset']} size={t['size_bytes']}")

# Check expert tensors
expert_count = 0
for t in result['tensors']:
    if 'expert' in t['name'] or 'exps' in t['name']:
        expert_count += 1
        if expert_count <= 10:
            print(f"  EXPERT: '{t['name']}' dims={t['dims']} type={t['type']}")

print(f"\nTotal expert-related tensors: {expert_count}")

# Verify data_start matches first tensor offset
if result['tensors']:
    first_offset = result['tensors'][0]['offset']
    print(f"\nFirst tensor offset: {first_offset}")
    print(f"Data start: {result['data_start']}")
    print(f"Match: {first_offset == result['data_start']}")
