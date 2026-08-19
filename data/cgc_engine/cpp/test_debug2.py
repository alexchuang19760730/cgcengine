#!/usr/bin/env python3
"""Debug the exact KV item that fails."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    
    print(f"Version: {version}, n_tensors: {n_tensors}, n_kv: {n_kv}")
    print(f"Header end (no KV parsed): {f.tell()}")
    
    for _ in range(n_kv):
        klen_raw = f.read(8)
        if len(klen_raw) < 8:
            break
        klen = struct.unpack("<Q", klen_raw)[0]
        
        if klen == 0 or klen > 1048576:
            # Likely we hit the tokenizer section
            print(f"KV[{_}]: klen={klen} at pos {f.tell()-8} - LARGE KEY, stopping KV parse")
            print(f"  Next 32 bytes raw: {f.read(min(32, klen))}")
            break
        
        key = f.read(klen).decode("utf-8", errors="replace")
        
        dtype_raw = f.read(4)
        if len(dtype_raw) < 4:
            break
        dtype = struct.unpack("<I", dtype_raw)[0]
        
        if dtype == 4:
            val = struct.unpack("<I", f.read(4))[0]
        elif dtype == 5:
            val = struct.unpack("<i", f.read(4))[0]
        elif dtype == 6:
            val = struct.unpack("<f", f.read(4))[0]
        elif dtype == 7:
            val = struct.unpack("<d", f.read(8))[0]
        elif dtype == 8:
            slen_raw = f.read(8)
            if len(slen_raw) < 8:
                break
            slen = struct.unpack("<Q", slen_raw)[0]
            if slen > 1048576:
                print(f"KV[{_}]: key={key}, dtype=8, slen={slen} - LARGE VALUE")
                f.seek(slen, 1)
                val = f"<skipped:{slen}>"
            else:
                val = f.read(slen).decode("utf-8", errors="replace") if slen > 0 else ""
        elif dtype == 9:
            val = struct.unpack("<?", f.read(1))[0]
        else:
            val = None
        
        if _ < 20:
            vshow = str(val)[:80]
            print(f"KV[{_}]: {key} = {vshow}")

    print(f"\nFile position after KV: {f.tell()}")
    
    # Calculate expected tensor start
    # After all KVs, tensors start
    print(f"\nAttempting tensor parse from pos {f.tell()}")
    
    # Read first tensor
    nlen_raw = f.read(8)
    print(f"First tensor nlen raw: {nlen_raw.hex()}")
    nlen = struct.unpack("<Q", nlen_raw)[0]
    print(f"First tensor nlen: {nlen}")
    
    if nlen <= 65536:
        name = f.read(nlen).decode("utf-8", errors="replace")
        print(f"First tensor name: {name}")