#!/usr/bin/env python3
"""
Verification test: Compare our expert weight extraction with gguf.GGUFReader.get_tensor().
This validates that our slicing logic is correct and offsets are absolute.
"""
import sys
sys.path.insert(0, r"D:\alex\flashkv0516\app\edge_engine")

import gguf
import numpy as np
from llama_monkey_patch import ExpertStreamerLite

# Test with qwen3.6 A3B model
model_path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

print("Verification test: Comparing expert weight extraction")
print("=" * 80)

# Create our streamer
streamer = ExpertStreamerLite(model_path)

# Create gguf reader for reference comparison
reader = gguf.GGUFReader(model_path)

# Test 1: Verify offset values are absolute (not relative to data_start)
print("\n1. Verifying offset values are absolute file offsets:")
header_data_start = reader.data_offset
print(f"   reader.data_offset = {header_data_start} (0x{header_data_start:X})")

# Find first tensor by name
first_tensor = None
for t in reader.tensors:
    if t.name == "output.weight":
        first_tensor = t
        break

if first_tensor:
    print(f"   first tensor 'output.weight':")
    print(f"     reader.data_offset = {first_tensor.data_offset}")
    print(f"     Our stored offset   = {streamer.header['tensors'][0]['offset']}")
    print(f"     Match: {first_tensor.data_offset == streamer.header['tensors'][0]['offset']}")

# Test 2: Read expert 0 gate weight and compare with gguf reader
print("\n2. Verifying expert weight extraction (Layer 0, Expert 0, gate):")
layer_id, expert_id, role = 0, 0, "gate"

# Get our extraction
our_weight = streamer.get_expert_weight(layer_id, expert_id, role)
print(f"   Our extraction:")
if our_weight is not None:
    print(f"     type: {type(our_weight)}")
    print(f"     shape: {our_weight.shape}")
    print(f"     dtype: {our_weight.dtype}")
    print(f"     first 10 values: {our_weight.flatten()[:10]}")

# Get reference from gguf reader
# Find the full packed tensor name
tensor_name = f"blk.{layer_id}.ffn_gate_exps.weight"
ref_tensor = None
for t in reader.tensors:
    if t.name == tensor_name:
        ref_tensor = t
        break

if ref_tensor is None:
    print(f"   ERROR: Could not find tensor '{tensor_name}'")
    sys.exit(1)

# Extract expert 0 slice from reference
ref_data = ref_tensor.data  # Full tensor data as memmap
dims = list(ref_tensor.shape)  # [2048, 512, 256]
expert_count = dims[-1]
other_dims = dims[:-1]  # [2048, 512]

# Calculate bytes per expert
bpe = {0: 4, 1: 2, 30: 2, 22: 3, 18: 4}.get(ref_tensor.tensor_type.value, 4)
elements_per_expert = 1
for d in other_dims:
    elements_per_expert *= d
bytes_per_expert = elements_per_expert * bpe

# Extract expert 0 from reference
ref_expert_flat = ref_data.flatten()  # Flatten to 1D
expert_start = expert_id * bytes_per_expert
expert_end = expert_start + bytes_per_expert
ref_expert_bytes = ref_expert_flat[expert_start:expert_end]

print(f"\n   Reference (from gguf.GGUFReader):")
print(f"     tensor: {tensor_name}")
print(f"     shape: {dims}")
print(f"     type: {ref_tensor.tensor_type}")
print(f"     bytes_per_expert: {bytes_per_expert}")
print(f"     expert 0 slice first 10 bytes: {ref_expert_bytes[:10]}")

# Compare
print(f"\n3. Comparison:")
if our_weight is not None:
    our_bytes = our_weight.tobytes() if hasattr(our_weight, 'tobytes') else bytes(our_weight)
    ref_bytes = bytes(ref_expert_bytes)
    match = our_bytes == ref_bytes
    print(f"   Our bytes length: {len(our_bytes)}")
    print(f"   Ref bytes length: {len(ref_bytes)}")
    print(f"   EXACT MATCH: {match}")
    
    if not match:
        # Find first difference
        for i in range(min(len(our_bytes), len(ref_bytes))):
            if our_bytes[i] != ref_bytes[i]:
                print(f"   First diff at byte {i}: ours={our_bytes[i]}, ref={ref_bytes[i]}")
                break
        else:
            if len(our_bytes) != len(ref_bytes):
                print(f"   Length mismatch: ours={len(our_bytes)}, ref={len(ref_bytes)}")
    else:
        print(f"   ✅ Expert weight extraction VERIFIED!")

# Test 3: Verify load_expert returns correct data
print(f"\n4. Verifying load_expert (Layer 0, Expert 5):")
result = streamer.load_expert(5, 0)
if result:
    print(f"   Expert {result['expert_id']}, Layer {result['layer_id']}:")
    for role, data in result.get("roles", {}).items():
        print(f"     {role}: dims={data.get('dims')} size={data.get('data_size')} type={data.get('ggml_type')}")
        
        # Compare with reference
        if role == "gate":
            ref_tensor_name = f"blk.0.ffn_gate_exps.weight"
            ref = None
            for t in reader.tensors:
                if t.name == ref_tensor_name:
                    ref = t
                    break
            
            if ref:
                ref_data = ref.data.flatten()
                dims = list(ref.shape)
                bpe = {0: 4, 1: 2, 30: 2, 22: 3, 18: 4}.get(ref.tensor_type.value, 4)
                elements_per_expert = int(np.prod(dims[:-1]))
                bytes_per_expert = elements_per_expert * bpe
                
                expert_start = 5 * bytes_per_expert
                our_data = data["data"]
                ref_slice = bytes(ref_data[expert_start:expert_start + bytes_per_expert])
                
                if isinstance(our_data, bytes):
                    match = our_data == ref_slice
                else:
                    match = bytes(our_data) == ref_slice
                print(f"     ✅ Data match: {match}")

# Test 4: Verify different experts give different data
print(f"\n5. Verifying different experts give different data (sanity check):")
for eid in [0, 1, 255]:
    w = streamer.get_expert_weight(0, eid, "gate")
    if w is not None:
        first_bytes = w.flatten()[:5].tolist()
        print(f"   Expert {eid}: first 5 bytes = {first_bytes}")

print(f"\n✅ All verification tests completed!")
