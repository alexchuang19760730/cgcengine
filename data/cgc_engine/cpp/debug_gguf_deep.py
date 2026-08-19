#!/usr/bin/env python3
"""Debug tensor parsing using gguf library internals."""

import os
import struct
import numpy as np

filepath = r"D:\alex\flashkv0516\models\gemma4_gguf\gemma-4-26B-A4B-it-UD-IQ3_S.gguf"

from gguf import GGUFReader
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import quant_shape_to_byte_shape

# Try to construct reader step by step
with open(filepath, "rb") as f:
    raw_data = f.read()

# Parse header
magic = struct.unpack("<I", raw_data[:4])[0]
version = struct.unpack("<I", raw_data[4:8])[0]
n_tensors = struct.unpack("<Q", raw_data[8:16])[0]
n_kv = struct.unpack("<Q", raw_data[16:24])[0]

print(f"Version: {version}, Tensors: {n_tensors}, KV: {n_kv}")

# Use gguf's internal parsing
from gguf.gguf_reader import GGUFReader

# The issue is likely that gguf library's _build_tensors fails on this file
# Let's try to parse just the tensor info section manually and check
# if the data offsets are correct

# First, parse KV correctly using GGUFv3 format
def parse_kv_v3(data, n_kv):
    pos = 24
    kv = {}
    
    dtype_sizes = {
        0: 1,   # UINT8
        1: 1,   # INT8
        2: 2,   # UINT16
        3: 2,   # INT16
        4: 4,   # UINT32
        5: 4,   # INT32
        6: 4,   # FLOAT32
        7: 4,   # BOOL (4 bytes, padded)
        10: 8,  # UINT64
        11: 8,  # INT64
        12: 8,  # FLOAT64
    }
    
    def read_string(d, p):
        str_len = struct.unpack("<I", d[p:p+4])[0]
        s = d[p+4:p+4+str_len].decode('utf-8', errors='replace')
        return s, p + 4 + str_len
    
    for i in range(n_kv):
        # Read key
        key, pos = read_string(data, pos)
        
        # Read dtype
        dtype = struct.unpack("<I", data[pos:pos+4])[0]
        pos += 4
        
        if dtype == 7:  # BOOL
            val = struct.unpack("<I", data[pos:pos+4])[0]
            kv[key] = bool(val)
            pos += 4
        elif dtype == 8:  # STRING
            val, pos = read_string(data, pos)
            kv[key] = val
        elif dtype == 9:  # ARRAY
            arr_type = struct.unpack("<I", data[pos:pos+4])[0]
            pos += 4
            arr_count = struct.unpack("<Q", data[pos:pos+8])[0]
            pos += 8
            
            if arr_type == 8:  # Array of strings
                arr = []
                for _ in range(arr_count):
                    s, pos = read_string(data, pos)
                    arr.append(s)
                kv[key] = arr
            else:
                elem_size = dtype_sizes.get(arr_type, 4)
                total_bytes = arr_count * elem_size
                # Skip unknown array
                pos += total_bytes
        else:
            elem_size = dtype_sizes.get(dtype, 4)
            if elem_size > 0:
                # Simple scalar
                if elem_size == 1:
                    val = struct.unpack("<B" if dtype == 0 else "<b", data[pos:pos+1])[0]
                elif elem_size == 2:
                    val = struct.unpack("<H" if dtype == 2 else "<h", data[pos:pos+2])[0]
                elif elem_size == 4:
                    if dtype == 4:  # UINT32
                        val = struct.unpack("<I", data[pos:pos+4])[0]
                    elif dtype == 5:  # INT32
                        val = struct.unpack("<i", data[pos:pos+4])[0]
                    elif dtype == 6:  # FLOAT32
                        val = struct.unpack("<f", data[pos:pos+4])[0]
                    else:
                        val = 0
                elif elem_size == 8:
                    if dtype == 10:  # UINT64
                        val = struct.unpack("<Q", data[pos:pos+8])[0]
                    elif dtype == 11:  # INT64
                        val = struct.unpack("<q", data[pos:pos+8])[0]
                    elif dtype == 12:  # FLOAT64
                        val = struct.unpack("<d", data[pos:pos+8])[0]
                    else:
                        val = 0
                kv[key] = val
                pos += elem_size
    
    return kv, pos

kv, kv_end = parse_kv_v3(raw_data, n_kv)
print(f"KV end offset: {kv_end}")

# Print key architecture params
for key in ["gemma4.embedding_length", "gemma4.expert_feed_forward_length", 
            "gemma4.expert_count", "gemma4.expert_used_count",
            "general.architecture", "general.quantization_version"]:
    if key in kv:
        print(f"  {key}: {kv[key]}")

# Parse tensor info
def parse_tensor_info(data, n_tensors, start_pos):
    pos = start_pos
    tensors = []
    
    for i in range(n_tensors):
        # Read tensor name
        name_len = struct.unpack("<I", data[pos:pos+4])[0]
        pos += 4
        name = data[pos:pos+name_len].decode('utf-8', errors='replace')
        pos += name_len
        
        # Read dimensions
        n_dims = struct.unpack("<I", data[pos:pos+4])[0]
        pos += 4
        dims = []
        for d in range(n_dims):
            dims.append(struct.unpack("<Q", data[pos:pos+8])[0])
            pos += 8
        
        # Read ggml type
        ggml_type = struct.unpack("<I", data[pos:pos+4])[0]
        pos += 4
        
        # Read offset
        offset = struct.unpack("<Q", data[pos:pos+8])[0]
        pos += 8
        
        tensors.append({
            "name": name, "dims": dims, "type": ggml_type, "offset": offset
        })
    
    return tensors, pos

tensors, tensor_end = parse_tensor_info(raw_data, n_tensors, kv_end)
print(f"Tensor info section end: {tensor_end}")

# Calculate data start (32-byte aligned)
data_start = tensor_end
if data_start % 32 != 0:
    data_start += 32 - (data_start % 32)
print(f"Data start (aligned): {data_start}")

# Now verify each tensor
print(f"\nVerifying {len(tensors)} tensors...")
errors = []

for t in tensors:
    ggml_type = t["type"]
    dims = t["dims"]
    offset = t["offset"]
    
    # Calculate nbytes
    n_elements = 1
    for d in dims:
        n_elements *= d
    
    if ggml_type in (0, 1, 30):  # F32, F16, BF16 (unquantized)
        bpe = {0: 4, 1: 2, 30: 2}[ggml_type]
        nbytes = n_elements * bpe
    else:
        # Quantized type
        if ggml_type not in GGML_QUANT_SIZES:
            errors.append(f"  Unknown type {ggml_type}: {t['name']}")
            continue
        
        block_size, type_size = GGML_QUANT_SIZES[ggml_type]
        last_dim = dims[-1]
        
        if last_dim % block_size != 0:
            errors.append(f"  Block alignment error: {t['name']}, last_dim={last_dim}, block_size={block_size}")
            continue
        
        n_blocks = last_dim // block_size
        nbytes = (n_elements // last_dim) * n_blocks * type_size
    
    abs_offset = data_start + offset
    if abs_offset + nbytes > len(raw_data):
        errors.append(f"  Overflow: {t['name']}, needs {nbytes} bytes at offset {abs_offset}, file has {len(raw_data) - abs_offset}")

if errors:
    print(f"❌ ERRORS ({len(errors)}):")
    for e in errors[:20]:
        print(e)
else:
    print(f"✅ All tensor ranges valid")

# Now find the specific tensor from the error
# Error: cannot reshape array of size 487255660 into shape (262144,2310)
# This means gguf library tried to get 487255660 bytes and reshape into (262144, 2310)
# 262144 * 2310 = 605,552,640 elements
# But 487255660 bytes / 2 bytes_per_element = 243,627,830 elements (if F16)
# So gguf got 487MB of data but tried to interpret it as shape (262144, 2310)

# Let's find all tensors with these dims
print(f"\nSearching for tensors with dims containing 262144 or 2310...")
for t in tensors:
    if 262144 in t["dims"] or 2310 in t["dims"]:
        ggml_type = t["type"]
        type_name = GGMLQuantizationType(ggml_type).name if ggml_type in GGMLQuantizationType.__members__ else f"UNKNOWN({ggml_type})"
        print(f"  {t['name']}: dims={t['dims']}, type={type_name} (id={ggml_type})")
        
        if ggml_type in GGML_QUANT_SIZES:
            block_size, type_size = GGML_QUANT_SIZES[ggml_type]
            print(f"    block_size={block_size}, type_size={type_size}")
        else:
            print(f"    NOT IN GGML_QUANT_SIZES!")
