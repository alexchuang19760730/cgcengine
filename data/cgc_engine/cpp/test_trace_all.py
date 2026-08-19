#!/usr/bin/env python3
"""Trace ALL KVs from start, checking for misalignment."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    f.seek(24)  # Skip header
    
    for i in range(54):
        pos = f.tell()
        
        # Save position in case we need to recover
        saved_pos = pos
        
        klen_raw = f.read(8)
        if len(klen_raw) < 8:
            print(f"[{i}] EOF at pos {pos}")
            break
        klen = struct.unpack("<Q", klen_raw)[0]
        
        if klen > 65536:
            print(f"[{i}] pos={pos} klen={klen} TOO LARGE!")
            # Try to find next valid KV
            for skip in range(1, 1000):
                f.seek(saved_pos + skip)
                test_bytes = f.read(8)
                test_klen = struct.unpack("<Q", test_bytes)[0]
                if 3 <= test_klen <= 65536:
                    # Check if this looks like a key
                    key_test = f.read(test_klen)
                    try:
                        key_str = key_test.decode("utf-8")
                        if key_str.isprintable() or "." in key_str:
                            print(f"  => Recovered with {skip}B skip, key='{key_str}'")
                            f.seek(saved_pos + skip)
                            break
                    except:
                        pass
            else:
                print(f"  => Could not recover, stopping")
                break
            # Don't parse the rest properly, just skip
            continue
        
        key = f.read(klen).decode("utf-8", errors="replace")
        
        dtype_raw = f.read(4)
        if len(dtype_raw) < 4:
            break
        dtype = struct.unpack("<I", dtype_raw)[0]
        
        # Check dtype validity
        if dtype not in (4, 5, 6, 7, 8, 9, 10, 11, 12):
            print(f"[{i}] pos={pos} key='{key}' INVALID dtype={dtype}")
            break
        
        # Now parse value
        val = None
        if dtype in (4, 5, 10):
            val = struct.unpack("<I", f.read(4))[0]
        elif dtype == 6:
            val = struct.unpack("<f", f.read(4))[0]
        elif dtype == 7:
            val = struct.unpack("<Q", f.read(8))[0]
        elif dtype == 8:
            slen_raw = f.read(8)
            slen = struct.unpack("<Q", slen_raw)[0]
            if slen > 1048576:
                # Large string, skip
                remaining = slen
                while remaining > 0:
                    chunk = min(remaining, 100*1024*1024)
                    _ = f.read(chunk)
                    remaining -= chunk
                val = f"<LARGE:{slen}B>"
            else:
                val = f.read(slen).decode("utf-8", errors="replace") if slen > 0 else ""
        elif dtype == 9:
            # Array or bool
            probe = f.read(12)
            elem_type = struct.unpack("<I", probe[0:4])[0]
            array_len = struct.unpack("<Q", probe[4:12])[0]
            
            # Validate: if array_len is reasonable and elem_type is known
            if array_len < 10000000 and elem_type in (4,5,6,7,8,10,11,12):
                # It's an array
                elem_sizes = {4:4,5:4,6:4,7:8,10:4,11:8,12:8}
                if elem_type == 8:
                    # String array
                    for _ in range(min(array_len, 10000000)):
                        slen = struct.unpack("<Q", f.read(8))[0]
                        if slen > 1048576:
                            remaining = slen
                            while remaining > 0:
                                chunk = min(remaining, 100*1024*1024)
                                _ = f.read(chunk)
                                remaining -= chunk
                        else:
                            _ = f.seek(slen, 1)
                    val = f"<array[{array_len}] of strings>"
                else:
                    total = elem_sizes.get(elem_type, 4) * array_len
                    _ = f.read(total)
                    val = f"<array[{array_len}] of type {elem_type}>"
            else:
                # It's a bool (1 byte)
                f.seek(saved_pos + 8 + klen + 4)  # Back to value position
                val_byte = f.read(1)
                val = bool(val_byte[0])
        
        # Validate next KV
        next_pos = f.tell()
        next_bytes = f.read(8)
        next_klen = struct.unpack("<Q", next_bytes)[0]
        
        if 3 <= next_klen <= 65536:
            # Valid next KV
            status = "OK"
            # Don't forget to go back
            f.seek(next_pos)
        elif next_klen > 65536:
            # Check if it's actually a tokenizer large array we just skipped
            status = "LARGE"
            # Don't go back, continue
        else:
            # next_klen < 3: suspicious
            status = f"SUSPECT (next_klen={next_klen})"
            f.seek(next_pos)
        
        if i >= 40 or status != "OK":
            print(f"[{i}] pos={pos} key='{key}' dtype={dtype} val={val} | next_klen={next_klen} {status}")
            if status != "OK" and i >= 45:
                print(f"  *** PROBLEM DETECTED ***")
                # Show bytes
                f.seek(next_pos - 8)
                context = f.read(32)
                print(f"  Context: {context.hex()}")
                break
    
    print(f"\nFinal position: {f.tell()}")