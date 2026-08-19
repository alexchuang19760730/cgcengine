#!/usr/bin/env python3
"""Re-parse KV[46] with correct dtype."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    f.seek(10935305)
    
    # Read KV[46]
    klen = struct.unpack("<Q", f.read(8))[0]
    key = f.read(klen).decode("utf-8")
    dtype = struct.unpack("<I", f.read(4))[0]
    print(f"KV[46]: key='{key}' dtype={dtype}")
    
    # dtype=4 means uint32 (4 bytes)
    val_bytes = f.read(4)
    val = struct.unpack("<I", val_bytes)[0]
    print(f"  As uint32: {val} (bytes: {val_bytes.hex()})")
    
    # Now KV[47] should start
    pos = f.tell()
    print(f"\nKV[47] at pos {pos}:")
    klen2_raw = f.read(8)
    klen2 = struct.unpack("<Q", klen2_raw)[0]
    print(f"  klen = {klen2} (hex: {klen2_raw.hex()})")
    
    if klen2 <= 65536:
        key2 = f.read(klen2).decode("utf-8", errors="replace")
        dtype2 = struct.unpack("<I", f.read(4))[0]
        print(f"  key = '{key2}' dtype={dtype2}")
        
        # Read value based on dtype
        if dtype2 == 4:
            val2 = struct.unpack("<I", f.read(4))[0]
        elif dtype2 == 7:
            val2 = struct.unpack("<Q", f.read(8))[0]
        elif dtype2 == 8:
            slen = struct.unpack("<Q", f.read(8))[0]
            val2 = f.read(slen).decode("utf-8", errors="replace")[:50]
        elif dtype2 == 9:
            elem_type = struct.unpack("<I", f.read(4))[0]
            array_len = struct.unpack("<Q", f.read(8))[0]
            val2 = f"<array[{array_len}] elem_type={elem_type}>"
        else:
            val2 = f"<dtype={dtype2}>"
        print(f"  value = {val2}")
    
    # Continue to tensor area
    print(f"\nChecking tensor area...")
    f.seek(pos + klen2 + 4)  # After KV[47] key + dtype
    # Need to skip the value too
    f.seek(0, 2)  # Go to end
    file_size = f.tell()
    print(f"File size: {file_size}")