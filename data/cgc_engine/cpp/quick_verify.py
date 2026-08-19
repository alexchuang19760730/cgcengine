#!/usr/bin/env python3
"""Quick verification test - minimal data reading."""
import sys
sys.path.insert(0, r"D:\alex\flashkv0516\app\edge_engine")

import gguf
import numpy as np
from llama_monkey_patch import ExpertStreamerLite

model_path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

print("Quick verification test:")
print("=" * 60)

# Test 1: Verify offset is absolute
reader = gguf.GGUFReader(model_path)
streamer = ExpertStreamerLite(model_path)

tensor_name = "blk.0.ffn_gate_exps.weight"
ref_tensor = None
for t in reader.tensors:
    if t.name == tensor_name:
        ref_tensor = t
        break

print(f"\n1. Offset verification:")
print(f"   Reference offset: {ref_tensor.data_offset}")
our_offset = None
for t in streamer.header['tensors']:
    if t['name'] == tensor_name:
        our_offset = t['offset']
        break
print(f"   Our offset: {our_offset}")
print(f"   Match: {ref_tensor.data_offset == our_offset}")

# Test 2: Read small portion of expert 0 gate (just first 100 bytes)
print(f"\n2. Data extraction test (first 100 bytes of expert 0 gate):")

# Reference: read first 100 bytes from expert 0
ref_data = ref_tensor.data  # memmap
dims = list(ref_tensor.shape)
bpe = {0: 4, 1: 2, 30: 2, 22: 3, 18: 4}.get(ref_tensor.tensor_type.value, 4)
elements_per_expert = int(np.prod(dims[:-1]))
bytes_per_expert = elements_per_expert * bpe

# Expert 0 starts at byte 0 within the packed tensor
ref_expert_0 = ref_data.flatten()[:100]  # First 100 bytes of expert 0
print(f"   Reference first 100 bytes: {list(ref_expert_0)}")

# Our extraction: read expert 0 gate and check first 100 bytes
our_weight = streamer.get_expert_weight(0, 0, "gate")
if our_weight is not None:
    our_first_100 = our_weight.flatten()[:100]
    print(f"   Our first 100 bytes: {list(our_first_100)}")
    match = bytes(our_first_100) == bytes(ref_expert_0)
    print(f"   MATCH: {match}")
else:
    print(f"   Our extraction returned None!")

# Test 3: Read expert 5 and verify it's different from expert 0
print(f"\n3. Verify different experts give different data:")
our_weight_5 = streamer.get_expert_weight(0, 5, "gate")
if our_weight_5 is not None:
    our_first_100_5 = our_weight_5.flatten()[:100]
    different = bytes(our_first_100) != bytes(our_first_100_5)
    print(f"   Expert 0 first 100: {list(our_first_100[:5])}...")
    print(f"   Expert 5 first 100: {list(our_first_100_5[:5])}...")
    print(f"   Different: {different}")
else:
    print(f"   Expert 5 extraction returned None!")

print(f"\n✅ Verification complete!")
