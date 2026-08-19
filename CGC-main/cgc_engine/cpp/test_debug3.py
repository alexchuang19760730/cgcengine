#!/usr/bin/env python3
"""Trace exact byte positions for bool handling."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    
    print(f"Start of KV: {f.tell()}")
    
    for i in range(n_kv):
        pos_before = f.tell()
        klen_raw = f.read(8)
        klen = struct.unpack("<Q", klen_raw)[0]
        
        if klen > 1048576:
            print(f"\n=== PROBLEM at KV[{i}] ===")
            print(f"  pos_before: {pos_before}")
            print(f"  klen_raw hex: {klen_raw.hex()}")
            print(f"  klen: {klen}")
            print(f"  Bytes at pos {pos_before}: {klen_raw.hex()}")
            # Show context
            f.seek(max(0, pos_before - 16))
            context = f.read(48)
            print(f"  Context (48 bytes): {context.hex()}")
            print(f"  Context as text: {context}")
            break
        
        if klen == 0:
            key = ""
        else:
            key = f.read(klen).decode("utf-8", errors="replace")
        
        dtype_raw = f.read(4)
        dtype = struct.unpack("<I", dtype_raw)[0]
        
        pos_before_val = f.tell()
        
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
            slen = struct.unpack("<Q", slen_raw)[0]
            if slen > 1048576:
                print(f"  KV[{i}]: {key} = LARGE STRING slen={slen}")
                f.seek(slen, 1)
                val = f"<skipped:{slen}>"
                # This is likely the tokenizer - stop here
                print(f"  -> Skipping remaining KVs")
                remaining = n_kv - i - 1
                for j in range(remaining):
                    r = f.read(8)
                    kl = struct.unpack("<Q", r)[0]
                    if kl <= 65536:
                        f.read(kl)
                    else:
                        f.seek(kl, 1)
                    dt = f.read(4)
                    dtv = struct.unpack("<I", dt)[0]
                    if dtv == 4 or dtv == 5 or dtv == 6:
                        f.read(4)
                    elif dtv == 7:
                        f.read(8)
                    elif dtv == 8:
                        sl = struct.unpack("<Q", f.read(8))[0]
                        if sl <= 1048576:
                            f.read(sl)
                        else:
                            f.seek(sl, 1)
                    elif dtv == 9:
                        f.read(1)
                break
            else:
                val = f.read(slen).decode("utf-8", errors="replace") if slen > 0 else ""
        elif dtype == 9:
            val_raw = f.read(1)
            val = struct.unpack("<?", val_raw)[0]
            pos_after = f.tell()
            if i < 20:
                print(f"  KV[{i}]: {key} = {val} (bool, pos {pos_before_val}->{pos_after})")
            continue
        
        if i < 20 and dtype != 8:
            vshow = str(val)[:60]
            print(f"  KV[{i}]: {key} = {vshow}")
    
    print(f"\nFile position after KV: {f.tell()}")
    
    # Try to parse first tensor from here
    nlen_raw = f.read(8)
    nlen = struct.unpack("<Q", nlen_raw)[0]
    print(f"First tensor nlen: {nlen} (raw: {nlen_raw.hex()})")
    
    if nlen <= 65536:
        name = f.read(nlen).decode("utf-8", errors="replace")
        print(f"First tensor name: {name}")
    else:
        print("ERROR: First tensor name too long, parser is misaligned!")
        # Try to find where tensors start by searching for known patterns
        f.seek(pos_before)
        data = f.read(4096)
        print(f"Data at pos {pos_before}: {data[:100].hex()}")