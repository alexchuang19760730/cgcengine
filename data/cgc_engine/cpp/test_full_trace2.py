#!/usr/bin/env python3
"""Full trace with correct dtype=4 as uint32."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    f.seek(24)
    
    for i in range(54):
        pos = f.tell()
        klen_raw = f.read(8)
        if len(klen_raw) < 8:
            break
        klen = struct.unpack("<Q", klen_raw)[0]
        
        if klen > 65536:
            print(f"KV[{i}]: pos={pos} klen={klen} LARGE, stopping")
            break
        
        key = f.read(klen).decode("utf-8", errors="replace")
        
        dtype = struct.unpack("<I", f.read(4))[0]
        
        if dtype in (4, 5):  # uint32
            v = struct.unpack("<I", f.read(4))[0]
            print(f"KV[{i}]: '{key}' = uint32({v})")
        elif dtype == 6:  # float32
            v = struct.unpack("<f", f.read(4))[0]
            print(f"KV[{i}]: '{key}' = float32({v})")
        elif dtype == 7:  # uint64
            v = struct.unpack("<Q", f.read(8))[0]
            print(f"KV[{i}]: '{key}' = uint64({v})")
        elif dtype == 8:  # string
            slen = struct.unpack("<Q", f.read(8))[0]
            if slen > 1048576:
                print(f"KV[{i}]: '{key}' = LARGE_STRING(slen={slen}), stopping")
                try: f.seek(slen)
                except: pass
                break
            else:
                v = f.read(slen).decode("utf-8", errors="replace") if slen > 0 else ""
                print(f"KV[{i}]: '{key}' = string('{v}')")
        elif dtype == 9:  # bool
            v_raw = f.read(1)
            if len(v_raw) < 1:
                break
            v = struct.unpack("<?", v_raw)[0]
            print(f"KV[{i}]: '{key}' = bool({v})")
        elif dtype == 10:  # int32
            v = struct.unpack("<i", f.read(4))[0]
            print(f"KV[{i}]: '{key}' = int32({v})")
        elif dtype == 11:  # float64
            v = struct.unpack("<d", f.read(8))[0]
            print(f"KV[{i}]: '{key}' = float64({v})")
        elif dtype == 12:  # int64
            v = struct.unpack("<q", f.read(8))[0]
            print(f"KV[{i}]: '{key}' = int64({v})")
        else:
            print(f"KV[{i}]: '{key}' = UNKNOWN dtype={dtype}")
            break
    
    print(f"\nAfter KV: pos={f.tell()}")
    
    # Try first tensor
    nlen = struct.unpack("<Q", f.read(8))[0]
    print(f"First tensor nlen: {nlen}")
    if nlen <= 65536:
        name = f.read(nlen).decode("utf-8", errors="replace")
        print(f"First tensor: {name}")
    else:
        print("MISALIGNED!")
        # Show context at pos
        f.seek(pos)
        data = f.read(256)
        print(f"Context: {data[:128].hex()}")