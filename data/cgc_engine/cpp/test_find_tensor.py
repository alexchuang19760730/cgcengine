#!/usr/bin/env python3
"""Find correct tensor info start position."""

import struct
import os

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
file_size = os.path.getsize(filepath)

# From previous test, we know KV ends at 10943779
# Let's search for tensor names like "blk.0.attn_norm.weight" or "token_embd.weight"

# Known tensor names for qwen35moe:
# First tensor is typically "token_embd.weight" or "blk.0.attn_norm.weight"

with open(filepath, "rb") as f:
    # Search from 10943779 for common tensor names
    search_start = 10943779
    search_end = min(search_start + 1000000, file_size)
    
    print(f"Searching for tensor names from {search_start} to {search_end}...")
    
    # Read a large chunk and search for known patterns
    chunk_size = 1000000
    f.seek(search_start)
    data = f.read(chunk_size)
    
    # Search for "token_embd" or "blk." patterns
    for pattern in [b"token_embd", b"blk.0.", b"blk.0.attn_norm", b"weight"]:
        pos = data.find(pattern)
        if pos >= 0:
            print(f"Found '{pattern.decode()}' at offset {search_start + pos}")
            # Check if this looks like a tensor name (preceded by 8-byte length)
            if pos >= 8:
                nlen_candidate = struct.unpack("<Q", data[pos-8:pos])[0]
                print(f"  Preceding nlen candidate: {nlen_candidate}")
                if 1 <= nlen_candidate <= 256:
                    # This might be a valid tensor!
                    tensor_name = data[pos:pos+nlen_candidate].decode("utf-8", errors="replace")
                    print(f"  Possible tensor name: '{tensor_name}'")
                    
                    # Check what follows
                    after_name = pos + nlen_candidate
                    n_dims = struct.unpack("<I", data[after_name:after_name+4])[0]
                    print(f"  n_dims: {n_dims}")
                    
                    if 1 <= n_dims <= 5:
                        dims = []
                        for d in range(n_dims):
                            dim = struct.unpack("<Q", data[after_name+4+d*8:after_name+4+(d+1)*8])[0]
                            dims.append(dim)
                        print(f"  dims: {dims}")
                        
                        ggml_type = struct.unpack("<I", data[after_name+4+n_dims*8:after_name+4+n_dims*8+4])[0]
                        offset = struct.unpack("<Q", data[after_name+4+n_dims*8+4:after_name+4+n_dims*8+4+8])[0]
                        print(f"  ggml_type: {ggml_type}")
                        print(f"  offset: {offset}")
                        
                        if offset > 0:
                            print(f"\n  *** VALID TENSOR FOUND at offset {search_start + pos - 8}! ***")
    
    # Also try: search backwards from known data areas
    print(f"\n=== Trying known tensor offsets ===")
    
    # For qwen3.6 35B A3B, we know:
    # token_embd.weight: [256000, 2048] or similar
    # Let's try to find the data start by looking for valid offsets
    
    # GGUF stores tensor data sequentially
    # Let's try to find a valid sequence of tensors by scanning
    pos = search_start
    found = None
    for _ in range(1000):
        if pos >= search_end:
            break
        
        f.seek(pos)
        try:
            nlen_raw = f.read(8)
            if len(nlen_raw) < 8:
                break
            nlen = struct.unpack("<Q", nlen_raw)[0]
            
            if 1 <= nlen <= 256:
                name = f.read(nlen).decode("utf-8", errors="replace")
                if name.isprintable() or "." in name:
                    n_dims_raw = f.read(4)
                    if len(n_dims_raw) < 4:
                        break
                    n_dims = struct.unpack("<I", n_dims_raw)[0]
                    
                    if 1 <= n_dims <= 6:
                        dims_raw = f.read(n_dims * 8)
                        if len(dims_raw) < n_dims * 8:
                            break
                        dims = [struct.unpack("<Q", dims_raw[j*8:(j+1)*8])[0] for j in range(n_dims)]
                        
                        ggml_type_raw = f.read(4)
                        if len(ggml_type_raw) < 4:
                            break
                        ggml_type = struct.unpack("<I", ggml_type_raw)[0]
                        
                        offset_raw = f.read(8)
                        if len(offset_raw) < 8:
                            break
                        tensor_offset = struct.unpack("<Q", offset_raw)[0]
                        
                        if tensor_offset > 0 and tensor_offset < file_size:
                            found = pos
                            print(f"Found valid tensor at {pos}: '{name}' dims={dims} type={ggml_type} offset={tensor_offset}")
                            break
        except:
            pass
        
        pos += 1  # Try next byte
    
    if found:
        print(f"\n*** CORRECT TENSOR INFO START: {found} ***")
        print(f"  32-byte aligned: {(found + 31) & ~31}")
    else:
        print("Could not find valid tensor")