#!/usr/bin/env python3
"""Full trace with GGUF v2 dtype mapping."""

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
        
        if klen > 1048576:
            print(f"KV[{i}]: pos={pos} klen={klen} LARGE, skipping key")
            try:
                f.seek(klen)
            except:
                _ = f.read(klen)
            dtype = struct.unpack("<I", f.read(4))[0]
            if dtype in (5, 6, 10):
                f.read(4)
            elif dtype in (7, 11, 12):
                f.read(8)
            elif dtype == 8:
                slen = struct.unpack("<Q", f.read(8))[0]
                try: f.seek(slen)
                except: _ = f.read(slen)
            elif dtype == 9:
                f.read(1)
            elif dtype == 4:
                f.read(2)
            continue
        
        if klen == 0:
            key = ""
        else:
            key = f.read(klen).decode("utf-8", errors="replace")
        
        dtype_raw = f.read(4)
        if len(dtype_raw) < 4:
            break
        dtype = struct.unpack("<I", dtype_raw)[0]
        
        if dtype == 4:
            v = struct.unpack("<H", f.read(2))[0]
            print(f"KV[{i}]: '{key}' = uint16({v})")
        elif dtype == 5:
            v = struct.unpack("<I", f.read(4))[0]
            print(f"KV[{i}]: '{key}' = uint32({v})")
        elif dtype == 6:
            v = struct.unpack("<f", f.read(4))[0]
            print(f"KV[{i}]: '{key}' = float32({v})")
        elif dtype == 7:
            v = struct.unpack("<Q", f.read(8))[0]
            print(f"KV[{i}]: '{key}' = uint64({v})")
        elif dtype == 8:
            slen = struct.unpack("<Q", f.read(8))[0]
            if slen > 1048576:
                print(f"KV[{i}]: '{key}' = LARGE_STRING(slen={slen})")
                try: f.seek(slen)
                except: _ = f.read(slen)
            else:
                v = f.read(slen).decode("utf-8", errors="replace") if slen > 0 else ""
                print(f"KV[{i}]: '{key}' = string('{v}')")
        elif dtype == 9:
            v_raw = f.read(1)
            v = struct.unpack("<?", v_raw)[0]
            print(f"KV[{i}]: '{key}' = bool({v})")
        elif dtype == 10:
            v = struct.unpack("<i", f.read(4))[0]
            print(f"KV[{i}]: '{key}' = int32({v})")
        elif dtype == 11:
            v = struct.unpack("<d", f.read(8))[0]
            print(f"KV[{i}]: '{key}' = float64({v})")
        elif dtype == 12:
            v = struct.unpack("<q", f.read(8))[0]
            print(f"KV[{i}]: '{key}' = int64({v})")
        else:
            print(f"KV[{i}]: '{key}' = UNKNOWN dtype={dtype}, pos={pos}")
            print(f"  dtype bytes: {dtype_raw.hex()}")
            break
    
    print(f"\nAfter KV: pos={f.tell()}")
    
    # Try first tensor
    nlen = struct.unpack("<Q", f.read(8))[0]
    print(f"First tensor nlen: {nlen}")
    if nlen <= 65536:
        name = f.read(nlen).decode("utf-8", errors="replace")
        print(f"First tensor: {name}")
        # Read full tensor info
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
        ggml_type = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]
        print(f"  dims={dims}, type={ggml_type}, offset={offset}")
    else:
        print("MISALIGNED!")