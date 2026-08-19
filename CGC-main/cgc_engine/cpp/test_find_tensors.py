#!/usr/bin/env python3
"""Manually locate tensor start by pattern matching."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    # First read n_tensors and n_kv from header
    f.seek(4)  # skip magic
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    print(f"version={version}, n_tensors={n_tensors}, n_kv={n_kv}")
    
    # Now, from test_full_trace2.py, we know:
    # After 17 KVs, we are at pos=899 (KV[17] starts here)
    # KV[16] is at pos=893, key='general.tags', dtype=9 (bool), val=True
    # pos after KV[16] = 893 + 8 + len("general.tags") + 4 + 1 = let's calculate
    # "general.tags" = 14 characters
    # 8 + 14 = 22 bytes for klen+key, pos from 893 to 915?
    # Actually let's look for first tensor name pattern
    # Tensor names likely start with: "qwen35moe.blocks", "qwen35moe.blk", "blk."
    # Let's search file for common patterns
    # Let's first skip all KVs by parsing all of them properly, skipping large values
    f.seek(24)
    
    for i in range(n_kv):
        pos = f.tell()
        klen_raw = f.read(8)
        if len(klen_raw) < 8:
            break
        klen = struct.unpack("<Q", klen_raw)[0]
        
        if klen > 65536:
            # Skip key and parse rest of KV then continue parsing other KVs
            try: f.seek(klen, 1)
            except: _ = f.read(klen)
            
            dtype_raw = f.read(4)
            if len(dtype_raw) < 4:
                break
            dtype = struct.unpack("<I", dtype_raw)[0]
            
            if dtype in (4,5,6,10):
                f.seek(4, 1)
            elif dtype in (7,11,12):
                f.seek(8, 1)
            elif dtype == 8:
                slen_raw = f.read(8)
                if len(slen_raw) < 8:
                    break
                slen = struct.unpack("<Q", slen_raw)[0]
                try: f.seek(slen, 1)
                except: _ = f.read(slen)
            elif dtype ==9:
                f.seek(1,1)
            
            print(f"[SKIP] KV[{i}]: klen={klen}, dtype={dtype}, skipping to continue")
            continue
        
        key = f.read(klen).decode("utf-8", errors="replace")
        dtype = struct.unpack("<I", f.read(4))[0]
        
        # Parse value
        if dtype in (4,5,6,10):
            f.read(4)
        elif dtype in (7,11,12):
            f.read(8)
        elif dtype == 8:
            slen_raw = f.read(8)
            slen = struct.unpack("<Q", slen_raw)[0]
            if slen <= 1048576:
                f.read(slen)
            else:
                try: f.seek(slen, 1)
                except: pass
        elif dtype == 9:
            f.read(1)
        
        if i < 25:
            print(f"[OK] KV[{i}]: '{key}' dtype={dtype}")
        
    print(f"All KVs parsed, current pos={f.tell()}")
    
    # Now parse first tensor
    nlen_raw = f.read(8)
    if len(nlen_raw) < 8:
        print("EOF at tensor parse start")
    else:
        nlen = struct.unpack("<Q", nlen_raw)[0]
        print(f"First tensor nlen: {nlen}")
        if nlen <= 65536:
            name = f.read(nlen).decode("utf-8", errors="replace")
            print(f"First tensor name: '{name}'")
            # Read full tensor info
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ggml_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            print(f"  dims={dims}, type={ggml_type}, offset={offset}")
        else:
            print("First tensor nlen too big - MISALIGNED!")
            print(f"Showing next 256 bytes: {f.read(256).hex()}")