#!/usr/bin/env python3
"""Parse qwen35moe tensors by finding correct data start."""

import struct
import os

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
file_size = os.path.getsize(filepath)
print(f"File size: {file_size} bytes ({file_size/1024**3:.2f} GB)")

with open(filepath, "rb") as f:
    # Parse header
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    print(f"Version: {version}, n_tensors: {n_tensors}, n_kv: {n_kv}")
    
    # Parse all KVs, but skip large values
    kv_count = 0
    pos = f.tell()
    
    for i in range(n_kv):
        if f.tell() >= file_size:
            break
            
        klen_raw = f.read(8)
        if len(klen_raw) < 8:
            break
        klen = struct.unpack("<Q", klen_raw)[0]
        
        if klen > 65536:
            # Very large key - this should not happen
            print(f"KV[{i}]: klen={klen} too large, breaking")
            break
        
        key = f.read(klen).decode("utf-8", errors="replace")
        
        dtype_raw = f.read(4)
        if len(dtype_raw) < 4:
            break
        dtype = struct.unpack("<I", dtype_raw)[0]
        
        # Handle value
        if dtype in (4, 5, 6, 10):
            # 4-byte numeric
            val_raw = f.read(4)
            if len(val_raw) < 4:
                break
        elif dtype == 7:
            val_raw = f.read(8)
            if len(val_raw) < 8:
                break
        elif dtype == 8:  # STRING
            slen_raw = f.read(8)
            if len(slen_raw) < 8:
                break
            slen = struct.unpack("<Q", slen_raw)[0]
            if slen > 1048576:
                # Large string - try seek
                try:
                    f.seek(slen, 1)
                except:
                    # If seek fails, we need to skip differently
                    # Read in chunks
                    remaining = slen
                    while remaining > 0:
                        chunk = min(remaining, 100*1024*1024)  # 100MB chunks
                        _ = f.read(chunk)
                        remaining -= chunk
                val = f"<skipped:{slen} bytes>"
            else:
                val_raw = f.read(slen)
                if len(val_raw) < slen:
                    break
                val = val_raw.decode("utf-8", errors="replace") if slen > 0 else ""
        elif dtype == 9:  # ARRAY
            elem_type_raw = f.read(4)
            if len(elem_type_raw) < 4:
                break
            elem_type = struct.unpack("<I", elem_type_raw)[0]
            array_len_raw = f.read(8)
            if len(array_len_raw) < 8:
                break
            array_len = struct.unpack("<Q", array_len_raw)[0]
            
            elem_sizes = {4: 4, 5: 4, 6: 4, 7: 8, 10: 4, 11: 8, 12: 8}
            
            if elem_type == 8:  # STRING array
                for _ in range(array_len):
                    slen_raw = f.read(8)
                    if len(slen_raw) < 8:
                        break
                    slen = struct.unpack("<Q", slen_raw)[0]
                    try:
                        f.seek(slen, 1)
                    except:
                        chunk = 100*1024*1024
                        remaining = slen
                        while remaining > 0:
                            _ = f.read(min(chunk, remaining))
                            remaining -= min(chunk, remaining)
            else:
                total = elem_sizes.get(elem_type, 0) * array_len
                try:
                    f.seek(total, 1)
                except:
                    _ = f.read(total)
            val = f"<array[{array_len}]>"
        elif dtype == 11:
            val_raw = f.read(8)
            if len(val_raw) < 8:
                break
        elif dtype == 12:
            val_raw = f.read(8)
            if len(val_raw) < 8:
                break
        else:
            print(f"KV[{i}]: Unknown dtype={dtype} for key '{key}'")
            break
        
        kv_count += 1
        if i >= 47:
            print(f"KV[{i}]: '{key}' dtype={dtype} val={val if dtype != 8 or isinstance(val, str) else val[:50]+'...'}")
    
    pos_after_kv = f.tell()
    print(f"\nAfter {kv_count} KVs: pos={pos_after_kv}")
    
    # Now try to parse tensors
    # Data should start at pos_after_kv, 32-byte aligned
    data_start = (pos_after_kv + 31) & ~31
    print(f"Expected data start (32-byte aligned): {data_start}")
    print(f"Diff from KV end: {data_start - pos_after_kv} bytes")
    
    # Parse tensors
    tensor_count = 0
    expert_tensor_count = 0
    first_10_tensors = []
    blk_tensors = []
    
    for idx in range(n_tensors):
        nlen_raw = f.read(8)
        if len(nlen_raw) < 8:
            break
        nlen = struct.unpack("<Q", nlen_raw)[0]
        if nlen > 65536:
            print(f"Tensor[{idx}]: nlen={nlen} too large at pos {f.tell()-8}")
            break
        
        name = f.read(nlen).decode("utf-8", errors="replace")
        
        n_dims_raw = f.read(4)
        if len(n_dims_raw) < 4:
            break
        n_dims = struct.unpack("<I", n_dims_raw)[0]
        
        dims = []
        for _ in range(n_dims):
            dim_raw = f.read(8)
            if len(dim_raw) < 8:
                break
            dims.append(struct.unpack("<Q", dim_raw)[0])
        
        ggml_type_raw = f.read(4)
        if len(ggml_type_raw) < 4:
            break
        ggml_type = struct.unpack("<I", ggml_type_raw)[0]
        
        offset_raw = f.read(8)
        if len(offset_raw) < 8:
            break
        offset = struct.unpack("<Q", offset_raw)[0]
        
        tensor_count += 1
        
        if idx < 10:
            first_10_tensors.append((name, dims, ggml_type, offset))
        
        if 'expert' in name.lower():
            expert_tensor_count += 1
        
        if 'blk' in name.lower():
            blk_tensors.append((name, dims, ggml_type, offset))
    
    print(f"\n=== Tensor Parsing Result ===")
    print(f"Total tensors parsed: {tensor_count}")
    print(f"Expert tensors: {expert_tensor_count}")
    print(f"Block tensors: {len(blk_tensors)}")
    
    if first_10_tensors:
        print(f"\nFirst 10 tensors:")
        for i, (name, dims, ggml_type, offset) in enumerate(first_10_tensors):
            print(f"  [{i}] {name}: dims={dims} type={ggml_type} offset={offset}")
    
    if expert_tensor_count > 0:
        print(f"\nExpert tensor samples:")
        for name, dims, ggml_type, offset in blk_tensors:
            if 'expert' in name.lower():
                print(f"  {name}: dims={dims}")
                break
    
    if tensor_count > 0:
        print(f"\n[SUCCESS] GGUF parsing verified!")
    else:
        print(f"\n[FAIL] No tensors parsed - need to fix parser")