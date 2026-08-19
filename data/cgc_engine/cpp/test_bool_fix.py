#!/usr/bin/env python3
"""Quick test with the bool fix for add_bos_token."""

import struct
import sys
sys.path.insert(0, r"D:\alex\flashkv0516\app\edge_engine")

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

# Manually parse with the fix
with open(filepath, "rb") as f:
    f.seek(24)  # Skip header
    
    kv = {}
    BOOL_KEYS = {
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
        "tokenizer.ggml.add_space",
    }
    
    for i in range(54):
        klen_raw = f.read(8)
        if len(klen_raw) < 8:
            break
        klen = struct.unpack("<Q", klen_raw)[0]
        
        if klen > 65536:
            print(f"[{i}] klen={klen} too large, stopping")
            break
        
        key = f.read(klen).decode("utf-8", errors="replace")
        dtype = struct.unpack("<I", f.read(4))[0]
        
        # Apply fix
        if key in BOOL_KEYS and dtype == 7:
            print(f"[{i}] {key}: overriding dtype 7->9 (bool)")
            dtype = 9
        
        if dtype in (4, 5):
            val = struct.unpack("<I", f.read(4))[0]
        elif dtype == 6:
            val = struct.unpack("<f", f.read(4))[0]
        elif dtype == 7:
            val = struct.unpack("<Q", f.read(8))[0]
        elif dtype == 8:
            slen = struct.unpack("<Q", f.read(8))[0]
            if slen > 1048576:
                remaining = slen
                while remaining > 0:
                    chunk = min(remaining, 100*1024*1024)
                    _ = f.read(chunk)
                    remaining -= chunk
                val = f"<{slen}B>"
            else:
                val = f.read(slen).decode("utf-8", errors="replace") if slen > 0 else ""
        elif dtype == 9:
            # Array or bool detection
            probe = f.read(12)
            elem_type = struct.unpack("<I", probe[0:4])[0]
            array_len = struct.unpack("<Q", probe[4:12])[0]
            
            if array_len < 10000000 and elem_type in (4,5,6,7,8,10,11,12):
                # Array
                elem_sizes = {4:4,5:4,6:4,7:8,10:4,11:8,12:8}
                if elem_type == 8:
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
                    val = f"<array[{array_len}]>"
                else:
                    total = elem_sizes.get(elem_type, 4) * array_len
                    _ = f.read(total)
                    val = f"<array[{array_len}]>"
            else:
                # Bool
                # Go back and re-read as 1 byte
                f.seek(-12, 1)
                val_byte = f.read(1)
                val = bool(val_byte[0]) if isinstance(val_byte, bytes) else val_byte
        elif dtype == 10:
            val = struct.unpack("<i", f.read(4))[0]
        elif dtype == 12:
            val = struct.unpack("<q", f.read(8))[0]
        else:
            val = f"<dtype={dtype}>"
        
        kv[key] = val
        
        if i >= 40:
            print(f"[{i}] {key} dtype={dtype} = {val}")
        
        # Validate next KV
        next_klen = struct.unpack("<Q", f.read(8))[0]
        if 3 <= next_klen <= 65536:
            f.seek(-8, 1)  # Go back, the next read will be correct
            if i >= 45:
                print(f"  => Next KV OK (klen={next_klen})")
        elif next_klen > 65536:
            if i >= 45:
                print(f"  => Next KV LARGE (klen={next_klen}), might be tokenizer data")
            f.seek(-8, 1)  # Go back, might be valid tokenizer entry
        else:
            print(f"  => Next KV SUSPECT (klen={next_klen}), correcting...")
            # Skip and try to fix
            # Re-read without the 8 byte lookahead
            # We're 8 bytes ahead, so go back
            pass
    
    print(f"\nParsed {len(kv)} KVs. Current pos: {f.tell()}")
    
    # Now try to parse tensors
    pos = f.tell()
    aligned = (pos + 31) & ~31
    if aligned > pos:
        f.seek(aligned)
    print(f"Aligned to: {f.tell()}")
    
    # Check first tensor
    nlen_raw = f.read(8)
    nlen = struct.unpack("<Q", nlen_raw)[0]
    print(f"First tensor nlen: {nlen}")
    if 3 <= nlen <= 65536:
        name = f.read(nlen).decode("utf-8", errors="replace")
        print(f"First tensor name: '{name}'")
        if name.isprintable():
            print("[SUCCESS] Tensor parsing works!")
        else:
            print("[FAIL] First tensor name not readable")
    else:
        print(f"[FAIL] Bad first tensor nlen={nlen}")