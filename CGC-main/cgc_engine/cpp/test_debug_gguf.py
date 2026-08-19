#!/usr/bin/env python3
"""Debug GGUF parsing for qwen35moe."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    
    print(f"Magic: 0x{magic:08X}")
    print(f"Version: {version}")
    print(f"n_tensors: {n_tensors}")
    print(f"n_kv: {n_kv}")
    
    # Skip KV parsing (we know it works)
    kv_count = 0
    for _ in range(n_kv):
        klen_raw = f.read(8)
        if len(klen_raw) < 8:
            break
        klen = struct.unpack("<Q", klen_raw)[0]
        if klen == 0 or klen > 65536:
            print(f"  KV[{_}]: klen={klen} (invalid), stopping")
            break
        key = f.read(klen)
        dtype_raw = f.read(4)
        dtype = struct.unpack("<I", dtype_raw)[0]
        # Skip value based on type
        if dtype == 4 or dtype == 5 or dtype == 6:
            f.read(4)
        elif dtype == 7:
            f.read(8)
        elif dtype == 8:
            slen = struct.unpack("<Q", f.read(8))[0]
            if slen > 1048576:
                f.seek(slen)
            else:
                f.read(slen)
        elif dtype == 9:
            f.read(1)
        kv_count += 1
    
    print(f"KV parsed: {kv_count}")
    print(f"File position after KV: {f.tell()}")
    
    # Now try to parse tensors
    print(f"\n=== Parsing {n_tensors} tensors ===")
    
    success = 0
    failures = 0
    for i in range(n_tensors):
        pos_before = f.tell()
        
        # Read name length
        nlen_raw = f.read(8)
        if len(nlen_raw) < 8:
            print(f"  Tensor[{i}]: Failed to read nlen at pos {pos_before}")
            failures += 1
            break
        
        nlen = struct.unpack("<Q", nlen_raw)[0]
        
        if nlen == 0 or nlen > 65536:
            print(f"  Tensor[{i}]: Invalid nlen={nlen} at pos {pos_before}, stopping")
            failures += 1
            break
        
        # Read name
        name_raw = f.read(nlen)
        if len(name_raw) < nlen:
            print(f"  Tensor[{i}]: Failed to read name at pos {pos_before}")
            failures += 1
            break
        name = name_raw.decode("utf-8", errors="replace")
        
        # Read n_dims
        ndims_raw = f.read(4)
        if len(ndims_raw) < 4:
            print(f"  Tensor[{i}]: Failed to read n_dims")
            failures += 1
            break
        n_dims = struct.unpack("<I", ndims_raw)[0]
        
        # Read dims
        dims = []
        for _ in range(n_dims):
            dim_raw = f.read(8)
            if len(dim_raw) < 8:
                print(f"  Tensor[{i}]: Failed to read dims")
                failures += 1
                break
            dims.append(struct.unpack("<Q", dim_raw)[0])
        
        # Read ggml_type
        type_raw = f.read(4)
        if len(type_raw) < 4:
            print(f"  Tensor[{i}]: Failed to read type")
            failures += 1
            break
        ggml_type = struct.unpack("<I", type_raw)[0]
        
        # Read offset
        offset_raw = f.read(8)
        if len(offset_raw) < 8:
            print(f"  Tensor[{i}]: Failed to read offset")
            failures += 1
            break
        offset = struct.unpack("<Q", offset_raw)[0]
        
        if i < 10:
            print(f"  Tensor[{i}]: name={name}, dims={dims}, type={ggml_type}, offset={offset}")
        
        success += 1
    
    print(f"\nSuccessfully parsed: {success} tensors, failures: {failures}")
    print(f"Final file position: {f.tell()}")
    print(f"File size: {filepath}")
    
    import os
    file_size = os.path.getsize(filepath)
    print(f"Actual file size: {file_size} bytes ({file_size/1024**3:.2f} GB)")
    
    # Calculate data_start
    data_start = (f.tell() + 31) & ~31
    print(f"\nCalculated data_start: {data_start}")
    print(f"data_start hex: 0x{data_start:X}")