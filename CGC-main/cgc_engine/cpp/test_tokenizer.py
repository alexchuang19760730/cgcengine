#!/usr/bin/env python3
"""Debug tokenizer KV parsing."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    f.seek(24)  # Skip header
    
    for i in range(54):
        pos = f.tell()
        klen_raw = f.read(8)
        if len(klen_raw) < 8:
            print(f"KV[{i}]: EOF at pos {pos}")
            break
        klen = struct.unpack("<Q", klen_raw)[0]
        
        if klen > 65536:
            print(f"KV[{i}]: pos={pos} klen={klen} LARGE, hex={klen_raw.hex()}")
            # Show context
            try:
                f.seek(max(0, pos-16))
                ctx = f.read(48)
                print(f"  Context: {ctx.hex()}")
            except:
                pass
            break
        
        key = f.read(klen).decode("utf-8", errors="replace")
        
        dtype_raw = f.read(4)
        if len(dtype_raw) < 4:
            break
        dtype = struct.unpack("<I", dtype_raw)[0]
        
        if dtype in (4, 5, 6, 10):
            val_raw = f.read(4)
        elif dtype == 7:
            val_raw = f.read(8)
        elif dtype == 8:  # STRING
            slen_raw = f.read(8)
            slen = struct.unpack("<Q", slen_raw)[0]
            if slen > 1048576:
                print(f"KV[{i}]: '{key}' LARGE STRING slen={slen}, skipping...")
                # Use chunked read
                remaining = slen
                chunk = 100*1024*1024
                while remaining > 0:
                    read_size = min(chunk, remaining)
                    _ = f.read(read_size)
                    remaining -= read_size
                val = f"<skipped:{slen} bytes>"
            else:
                val_raw = f.read(slen)
                val = val_raw.decode("utf-8", errors="replace") if slen > 0 else ""
        elif dtype == 9:  # ARRAY
            elem_type_raw = f.read(4)
            elem_type = struct.unpack("<I", elem_type_raw)[0]
            array_len_raw = f.read(8)
            array_len = struct.unpack("<Q", array_len_raw)[0]
            
            elem_sizes = {4: 4, 5: 4, 6: 4, 7: 8, 10: 4, 11: 8, 12: 8}
            
            if elem_type == 8:  # STRING array
                total_strings = array_len
                bytes_skipped = 0
                for j in range(array_len):
                    slen_raw = f.read(8)
                    slen = struct.unpack("<Q", slen_raw)[0]
                    if slen > 1048576:
                        # Large string in array - skip
                        remaining = slen
                        chunk = 100*1024*1024
                        while remaining > 0:
                            read_size = min(chunk, remaining)
                            _ = f.read(read_size)
                            remaining -= read_size
                    else:
                        _ = f.read(slen)
                    bytes_skipped += 8 + slen
                val = f"<array[{array_len}] of string ({bytes_skipped} bytes)>"
            else:
                total = elem_sizes.get(elem_type, 0) * array_len
                _ = f.read(total)
                val = f"<array[{array_len}]>"
        elif dtype == 11:
            val_raw = f.read(8)
        elif dtype == 12:
            val_raw = f.read(8)
        else:
            print(f"KV[{i}]: '{key}' UNKNOWN dtype={dtype}")
            break
        
        if i >= 40:
            vshow = str(val)[:80] if isinstance(val, str) else val
            print(f"KV[{i}]: pos={pos} '{key}' dtype={dtype} = {vshow}")
    
    print(f"\nFinal pos: {f.tell()}")