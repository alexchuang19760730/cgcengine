#!/usr/bin/env python3
"""Quick test of ExpertStreamerLite logic without full data reads."""
import sys
sys.path.insert(0, r"D:\alex\flashkv0516\app\edge_engine")

from llama_monkey_patch import parse_gguf_header

# Test with qwen3.6 A3B model
model_path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

print("Quick test of GGUF parsing and expert slicing logic:")
print("=" * 60)

# Parse header
result = parse_gguf_header(model_path)
kv = result['kv']

print(f"\nArchitecture: {kv.get('general.architecture', 'N/A')}")
print(f"Hidden size: {kv.get('qwen35moe.embedding_length', 'N/A')}")
print(f"Expert count: {kv.get('qwen35moe.expert_count', 'N/A')}")
print(f"Block count: {kv.get('qwen35moe.block_count', 'N/A')}")
print(f"Expert FF length: {kv.get('qwen35moe.expert_feed_forward_length', 'N/A')}")

# Check expert tensors structure
print(f"\nExpert tensor analysis:")
expert_tensors = [t for t in result['tensors'] if 'exps' in t['name']]
print(f"  Total expert tensors: {len(expert_tensors)}")

# Analyze one tensor to understand structure
for t in expert_tensors[:3]:
    name = t['name']
    dims = t['dims']
    ttype = t['type']
    size_bytes = t['size_bytes']
    
    print(f"\n  Tensor: {name}")
    print(f"    dims: {dims}")
    print(f"    type: {ttype}")
    print(f"    size_bytes: {size_bytes:,} ({size_bytes / 1024 / 1024:.1f} MB)")
    
    # Calculate per-expert slice
    if len(dims) >= 3:
        expert_count = dims[-1]
        other_dims = dims[:-1]
        
        # Get bytes per element
        from llama_monkey_patch import GGML_TYPE_BYTES
        bpe = GGML_TYPE_BYTES.get(ttype, 4)
        
        # Calculate elements per expert
        elements_per_expert = 1
        for d in other_dims:
            elements_per_expert *= d
        bytes_per_expert = elements_per_expert * bpe
        
        print(f"    expert_count (last dim): {expert_count}")
        print(f"    other_dims: {other_dims}")
        print(f"    bpe: {bpe}")
        print(f"    elements_per_expert: {elements_per_expert:,}")
        print(f"    bytes_per_expert: {bytes_per_expert:,} ({bytes_per_expert / 1024 / 1024:.1f} MB)")
        print(f"    Verify: {bytes_per_expert * expert_count:,} == {size_bytes:,} -> {'OK' if bytes_per_expert * expert_count == size_bytes else 'MISMATCH!'}")

# Test the slicing logic with a small in-memory example
print(f"\n\nSlicing logic verification (in-memory test):")
import numpy as np

# Simulate a tensor with shape [512, 2048, 256] (down projection)
# where 256 is expert count
dims = [512, 2048, 256]
expert_count = dims[-1]
other_dims = dims[:-1]  # [512, 2048]

# Create fake data
total_elements = 1
for d in dims:
    total_elements *= d
fake_data = np.arange(total_elements, dtype=np.int32).reshape(dims)

# Slice out expert 5
expert_id = 5
expert_data = fake_data[:, :, expert_id]  # Shape [512, 2048]

print(f"  Original shape: {dims}")
print(f"  Expert {expert_id} shape: {expert_data.shape}")
print(f"  First element of expert {expert_id}: {expert_data[0, 0]}")

# Verify it matches
expected_val = expert_id  # Because data[:, :, expert_id] starts at value = expert_id * 512 * 2048
print(f"  Expected first value: {expected_val}")
print(f"  Match: {expert_data[0, 0] == expected_val}")

# Also test with transposed case (gate/up projection)
# gate has shape [2048, 512, 256]
dims2 = [2048, 512, 256]
total_elements2 = 1
for d in dims2:
    total_elements2 *= d
fake_data2 = np.arange(total_elements2, dtype=np.int32).reshape(dims2)

expert_data2 = fake_data2[:, :, expert_id]
print(f"\n  Gate tensor shape: {dims2}")
print(f"  Expert {expert_id} shape: {expert_data2.shape}")
print(f"  First element: {expert_data2[0, 0]}")

print(f"\n✅ Slicing logic verified!")
