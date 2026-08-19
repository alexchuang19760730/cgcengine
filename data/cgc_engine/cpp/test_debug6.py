#!/usr/bin/env python3
"""Trace with corrected v2 dtype mapping."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    
    print(f"Version: {version}, n_tensors: {n_tensors}, n_kv: {n_kv}")
    
    for i in range(n_kv):
        pos = f.tell()
        klen_raw = f.read(8)
        if len(klen_raw) < 8:
            print(f"KV[{i}]: EOF at klen, pos={pos}")
            break
        klen = struct.unpack("<Q", klen_raw)[0]
        
        if klen > 1048576:
            print(f"KV[{i}]: pos={pos} klen={klen} TOO LARGE")
            try:
                f.seek(klen)
            except:
                _ = f.read(klen)
            continue
        
        key = f.read(klen).decode("utf-8", errors="replace") if klen > 0 else ""
        
        dtype_raw = f.read(4)
        dtype = struct.unpack("<I", dtype_raw)[0]
        
        # v2 dtype mapping
        if dtype == 4:   # uint32
            v = struct.unpack("<I", f.read(4))[0]
            print(f"KV[{i}]: '{key}' = uint32({v})")
        elif dtype == 5:  # float32
            v = struct.unpack("<f", f.read(4))[0]
            print(f"KV[{i}]: '{key}' = float32({v})")
        elif dtype == 6:  # uint16
            v = struct.unpack("<H", f.read(2))[0]
            print(f"KV[{i}]: '{key}' = uint16({v})")
        elif dtype == 7:  # uint64
            v = struct.unpack("<Q", f.read(8))[0]
            print(f"KV[{i}]: '{key}' = uint64({v})")
        elif dtype == 8:  # string
            slen = struct.unpack("<Q", f.read(8))[0]
            if slen > 1048576:
                print(f"KV[{i}]: '{key}' = LARGE_STRING(slen={slen})")
                try:
                    f.seek(slen)
                except:
                    _ = f.read(slen)
            else:
                v = f.read(slen).decode("utf-8", errors="replace") if slen > 0 else ""
                print(f"KV[{i}]: '{key}' = string('{v}')")
        elif dtype == 9:  # bool
            v = struct.unpack("<?", f.read(1))[0]
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
            print(f"KV[{i}]: '{key}' = UNKNOWN dtype={dtype}, pos={pos}")
            print(f"  Raw dtype bytes: {dtype_raw.hex()}")
            break
    
    print(f"\nAfter KV: pos={f.tell()}")
    
    # Read first tensor
    nlen = struct.unpack("<Q", f.read(8))[0]
    print(f"First tensor nlen: {nlen}")
    if nlen <= 65536:
        name = f.read(nlen).decode("utf-8", errors="replace")
        print(f"First tensor: {name}")
    else:
        print("MISALIGNED!")
        # Show context
        f.seek(pos)
        data = f.read(256)
        print(f"Context hex: {data[:128].hex()}")