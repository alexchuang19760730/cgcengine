#!/usr/bin/env python3
"""Precise manual tracing of tokenizer KV parsing."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    # Skip header
    f.seek(24)
    
    for i in range(54):
        pos = f.tell()
        klen_raw = f.read(8)
        if len(klen_raw) < 8:
            break
        klen = struct.unpack("<Q", klen_raw)[0]
        
        if klen > 65536:
            print(f"KV[{i}]: pos={pos} klen={klen} LARGE")
            break
        
        key = f.read(klen).decode("utf-8", errors="replace")
        print(f"KV[{i}]: pos={pos} key='{key}'", end="")
        
        dtype_raw = f.read(4)
        if len(dtype_raw) < 4:
            break
        dtype = struct.unpack("<I", dtype_raw)[0]
        print(f" dtype={dtype}", end="")
        
        if dtype == 9:  # ARRAY
            elem_type_raw = f.read(4)
            elem_type = struct.unpack("<I", elem_type_raw)[0]
            array_len_raw = f.read(8)
            array_len = struct.unpack("<Q", array_len_raw)[0]
            print(f" elem_type={elem_type} array_len={array_len}", end="")
            
            if elem_type == 8:  # STRING array
                total_bytes = 0
                for j in range(array_len):
                    slen_raw = f.read(8)
                    if len(slen_raw) < 8:
                        print(f" [EOF at string {j}]")
                        break
                    slen = struct.unpack("<Q", slen_raw)[0]
                    total_bytes += 8 + slen
                    if slen > 50 * 1024 * 1024:
                        remaining = slen
                        chunk = 100 * 1024 * 1024
                        while remaining > 0:
                            _ = f.read(min(chunk, remaining))
                            remaining -= min(chunk, remaining)
                    else:
                        _ = f.seek(slen, 1)
                print(f" total_bytes={total_bytes}")
            else:
                elem_sizes = {4: 4, 5: 4, 6: 4, 7: 8, 10: 4, 11: 8, 12: 8}
                total = elem_sizes.get(elem_type, 0) * array_len
                _ = f.seek(total, 1)
                print(f" total_numeric={total}")
        elif dtype == 8:  # STRING
            slen_raw = f.read(8)
            slen = struct.unpack("<Q", slen_raw)[0]
            if slen > 50 * 1024 * 1024:
                remaining = slen
                chunk = 100 * 1024 * 1024
                while remaining > 0:
                    _ = f.read(min(chunk, remaining))
                    remaining -= min(chunk, remaining)
            else:
                _ = f.seek(slen, 1)
            print(f" slen={slen}")
        elif dtype in (4, 5, 6, 10):
            val = struct.unpack("<I", f.read(4))[0] if dtype in (4, 5, 10) else struct.unpack("<f", f.read(4))[0]
            print(f" val={val}")
        elif dtype == 7:
            val = struct.unpack("<Q", f.read(8))[0]
            print(f" val={val}")
        elif dtype == 11:
            val = struct.unpack("<d", f.read(8))[0]
            print(f" val={val}")
        elif dtype == 12:
            val = struct.unpack("<q", f.read(8))[0]
            print(f" val={val}")
        else:
            print(f" UNKNOWN")
            break
    
    final_pos = f.tell()
    print(f"\nFinal pos: {final_pos}")
    
    # Align to 32 bytes
    aligned = (final_pos + 31) & ~31
    print(f"Aligned pos: {aligned}")
    
    # Read first tensor
    f.seek(aligned)
    nlen_raw = f.read(8)
    nlen = struct.unpack("<Q", nlen_raw)[0]
    print(f"\nFirst tensor nlen: {nlen} (hex: {nlen_raw.hex()})")
    if nlen > 65536:
        print("=> FAILED: nlen too large, alignment is wrong")
    else:
        name = f.read(nlen).decode("utf-8", errors="replace")
        print(f"First tensor name: '{name}'")