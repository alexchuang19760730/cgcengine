#!/usr/bin/env python3
"""Trace raw bytes for first 10 KV items."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    # Skip header
    f.seek(24)  # 4+4+8+8
    
    for i in range(10):
        pos = f.tell()
        
        # Read klen
        klen_raw = f.read(8)
        klen = struct.unpack("<Q", klen_raw)[0]
        
        if klen > 1048576:
            print(f"KV[{i}]: pos={pos} klen={klen} TOO LARGE, hex={klen_raw.hex()}")
            try:
                f.seek(klen)
            except:
                _ = f.read(klen)
            continue
        
        # Read key
        if klen > 0:
            key = f.read(klen).decode("utf-8", errors="replace")
        else:
            key = ""
        
        # Show the next 4 bytes (dtype) and 16 bytes context
        dtype_raw = f.read(4)
        dtype = struct.unpack("<I", dtype_raw)[0]
        
        # Read the value (next 16 bytes for context)
        val_ctx = f.read(16) if klen <= 65536 else b''
        
        print(f"KV[{i}]: pos={pos}")
        print(f"  klen={klen}, key='{key}'")
        print(f"  dtype bytes: {dtype_raw.hex()} -> dtype={dtype}")
        print(f"  value context (16 bytes): {val_ctx.hex()}")
        
        # Now try to determine actual type from context
        # For known keys, we expect:
        if key == 'general.architecture' and dtype == 8:
            # string: read slen then value
            slen = struct.unpack("<Q", val_ctx[:8])[0]
            v = val_ctx[8:8+slen].decode("utf-8", errors="replace")
            print(f"  -> STRING slen={slen} value='{v}'")
            f.seek(16 - 8 - slen, 1)  # reposition after value_ctx
            # Actually let's just skip properly
            f.seek(pos + 8 + klen + 4)  # back to after dtype
            slen = struct.unpack("<Q", f.read(8))[0]
            v = f.read(slen).decode("utf-8", errors="replace") if slen > 0 else ""
            print(f"  [parsed] '{key}' = '{v}'")
        elif key == 'general.sampling.top_k':
            # Expect uint32(20)
            # val_ctx[0:4] should be 0x14000000 (20 in LE)
            expected = struct.unpack("<I", val_ctx[:4])[0]
            print(f"  -> First 4 bytes as uint32: {expected}")
            f.seek(pos + 8 + klen + 4)  # rewind
            actual_dtype = struct.unpack("<I", f.read(4))[0]
            print(f"  Actual dtype: {actual_dtype}")
            if actual_dtype == 5:
                v = struct.unpack("<I", f.read(4))[0]
                print(f"  -> uint32 = {v}")
            elif actual_dtype == 4:
                # Hmm, the file said dtype=4 but earlier mapping was off
                print(f"  -> dtype=4, unknown")
            # skip
            f.seek(pos + 8 + klen + 4 + 4 + 4)  # klen(8) + key + dtype(4) + 4 bytes value
        else:
            # Skip based on guessed dtype
            f.seek(pos + 8 + klen + 4)  # rewind to after dtype
            if dtype == 4 or dtype == 5 or dtype == 10:
                f.read(4)
            elif dtype == 6 or dtype == 11 or dtype == 7 or dtype == 8 or dtype == 12:
                # For uint64/float64/string, read 8 more
                extra = f.read(8)
                if dtype == 8:  # string
                    slen = struct.unpack("<Q", extra)[0]
                    if slen <= 1048576:
                        f.read(slen)
                    else:
                        try: f.seek(slen, 1)
                        except: _ = f.read(slen)
                elif dtype == 7:  # uint64
                    pass
                elif dtype == 11:  # float64
                    pass
            elif dtype == 9:  # bool
                f.read(1)
    
    print(f"\nFinal pos: {f.tell()}")