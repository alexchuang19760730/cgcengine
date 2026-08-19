#!/usr/bin/env python3
"""
Fix GGUF tensor parsing for qwen35moe by finding the correct data start position.
From manual tracing, we know:
- KV[0]-KV[45] parse correctly
- KV[46] add_bos_token has dtype=7 (uint64) but the actual data has 1 byte padding
- After that, KV[47]-KV[53] are tokenizer-related
- Then tensor info starts, 32-byte aligned
"""

import struct
import os

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
file_size = os.path.getsize(filepath)

with open(filepath, "rb") as f:
    # Read header
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    
    print(f"Version: {version}, n_tensors: {n_tensors}, n_kv: {n_kv}")
    
    # Parse KV[0] to KV[45] correctly
    kv_parsed = 0
    for i in range(46):  # First 46 KVs (0-45)
        klen = struct.unpack("<Q", f.read(8))[0]
        key = f.read(klen).decode("utf-8", errors="replace")
        dtype = struct.unpack("<I", f.read(4))[0]
        
        if dtype in (4, 5):
            val = struct.unpack("<I", f.read(4))[0]
        elif dtype == 6:
            val = struct.unpack("<f", f.read(4))[0]
        elif dtype == 7:
            val = struct.unpack("<Q", f.read(8))[0]
        elif dtype == 8:
            slen = struct.unpack("<Q", f.read(8))[0]
            _ = f.seek(slen, 1) if slen <= 1048576 else None
            if slen > 1048576:
                remaining = slen
                while remaining > 0:
                    chunk = min(remaining, 100*1024*1024)
                    _ = f.read(chunk)
                    remaining -= chunk
            val = f"<{slen}B>"
        elif dtype == 9:
            elem_type = struct.unpack("<I", f.read(4))[0]
            array_len = struct.unpack("<Q", f.read(8))[0]
            if elem_type == 8:
                for _ in range(array_len):
                    slen = struct.unpack("<Q", f.read(8))[0]
                    if slen > 1048576:
                        remaining = slen
                        while remaining > 0:
                            chunk = min(remaining, 100*1024*1024)
                            _ = f.read(chunk)
                            remaining -= chunk
                    else:
                        _ = f.seek(slen, 1)
            else:
                elem_sizes = {4: 4, 5: 4, 6: 4, 7: 8, 10: 4, 11: 8, 12: 8}
                total = elem_sizes.get(elem_type, 4) * array_len
                _ = f.read(total)
            val = f"<array[{array_len}]>"
        elif dtype == 10:
            val = struct.unpack("<i", f.read(4))[0]
        elif dtype == 12:
            val = struct.unpack("<q", f.read(8))[0]
        else:
            val = f"<dtype={dtype}>"
        
        kv_parsed += 1
        if i < 5 or key.startswith("qwen"):
            print(f"KV[{i}]: {key} = {val}")
    
    print(f"\nParsed {kv_parsed} KVs. Current pos: {f.tell()}")
    
    # Now handle the problematic KV[46] = tokenizer.ggml.add_bos_token
    # dtype=7 (uint64), value=5888, BUT followed by 1 extra byte (bool padding)
    klen = struct.unpack("<Q", f.read(8))[0]
    key = f.read(klen).decode("utf-8")
    dtype = struct.unpack("<I", f.read(4))[0]
    print(f"\nKV[46]: {key} dtype={dtype}")
    
    # The uint64 value
    val_bytes = f.read(8)
    val = struct.unpack("<Q", val_bytes)[0]
    print(f"  uint64 value: {val}")
    
    # After uint64, there's a 1-byte bool-like padding
    # Let's check what comes next
    next_bytes = f.read(16)
    print(f"  Next 16 bytes: {next_bytes.hex()}")
    
    # Check: if these bytes form a valid KV (klen >= 3 and <= 65536)
    next_klen = struct.unpack("<Q", next_bytes[0:8])[0]
    print(f"  Candidate next klen: {next_klen}")
    
    if next_klen > 65536 or next_klen < 3:
        # Misaligned! The first byte (00) is actually the bool value
        # Re-read: skip 1 extra byte
        f.seek(-16, 1)  # Go back
        _ = f.read(1)   # Skip the bool padding byte
        # Now read KV[47]
        klen47 = struct.unpack("<Q", f.read(8))[0]
        key47 = f.read(klen47).decode("utf-8", errors="replace")
        dtype47 = struct.unpack("<I", f.read(4))[0]
        print(f"\nKV[47] (after fix): {key47} dtype={dtype47}")
    
    # Now skip the remaining tokenizer KVs (47-53)
    # We know KV[47] is tokenizer.chat_template (dtype=8, string)
    # Let's skip it and the rest
    
    for i in range(47, n_kv):
        klen = struct.unpack("<Q", f.read(8))[0]
        if klen > 65536:
            print(f"KV[{i}]: klen={klen} too large, stopping")
            break
        key = f.read(klen).decode("utf-8", errors="replace")
        dtype = struct.unpack("<I", f.read(4))[0]
        
        if dtype == 4:
            _ = f.read(4)
        elif dtype == 6:
            _ = f.read(4)
        elif dtype == 7:
            _ = f.read(8)
        elif dtype == 8:
            slen = struct.unpack("<Q", f.read(8))[0]
            if slen > 1048576:
                remaining = slen
                while remaining > 0:
                    chunk = min(remaining, 100*1024*1024)
                    _ = f.read(chunk)
                    remaining -= chunk
            else:
                _ = f.seek(slen, 1)
        elif dtype == 9:
            # Need to detect bool vs array
            saved = f.tell()
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
                else:
                    total = elem_sizes.get(elem_type, 4) * array_len
                    _ = f.read(total)
            else:
                # Bool
                f.seek(saved)
                _ = f.read(1)
        elif dtype == 10:
            _ = f.read(4)
        elif dtype == 12:
            _ = f.read(8)
        else:
            print(f"KV[{i}]: unknown dtype={dtype} for key '{key}'")
            break
    
    pos_after_kv = f.tell()
    print(f"\nAfter all KVs: pos={pos_after_kv}")
    
    # Align to 32 bytes
    aligned = (pos_after_kv + 31) & ~31
    if aligned > pos_after_kv:
        f.seek(aligned)
    print(f"Aligned to: {f.tell()}")
    
    # Now try to parse tensors!
    tensors = []
    for idx in range(n_tensors):
        nlen_raw = f.read(8)
        if len(nlen_raw) < 8:
            print(f"Stopped at tensor {idx}: EOF")
            break
        nlen = struct.unpack("<Q", nlen_raw)[0]
        if nlen == 0 or nlen > 65536:
            print(f"Stopped at tensor {idx}: bad nlen={nlen} (at pos {f.tell()-8})")
            break
        
        name = f.read(nlen).decode("utf-8", errors="replace")
        
        n_dims_raw = f.read(4)
        if len(n_dims_raw) < 4:
            break
        n_dims = struct.unpack("<I", n_dims_raw)[0]
        if n_dims > 10:
            print(f"Stopped at tensor {idx}: bad n_dims={n_dims} for '{name}'")
            break
        
        dims = []
        for _ in range(n_dims):
            dim_raw = f.read(8)
            dims.append(struct.unpack("<Q", dim_raw)[0])
        
        ggml_type_raw = f.read(4)
        ggml_type = struct.unpack("<I", ggml_type_raw)[0]
        
        offset_raw = f.read(8)
        tensor_offset = struct.unpack("<Q", offset_raw)[0]
        
        tensors.append({
            "name": name,
            "dims": dims,
            "type": ggml_type,
            "offset": tensor_offset,
        })
        
        if idx < 10:
            print(f"  [{idx}] {name}: dims={dims} type={ggml_type} offset={tensor_offset}")
    
    print(f"\nTotal tensors parsed: {len(tensors)}")
    
    # Expert tensor analysis
    expert_tensors = [t for t in tensors if "expert" in t["name"].lower()]
    print(f"Expert tensors: {len(expert_tensors)}")
    
    if expert_tensors:
        # Check structure
        layers = set()
        eids = set()
        roles = set()
        for t in expert_tensors:
            parts = t["name"].split(".")
            if len(parts) >= 5:
                try:
                    layers.add(int(parts[1]))
                    eids.add(int(parts[3]))
                    roles.add(parts[4])
                except:
                    pass
        
        print(f"Layers with experts: {sorted(layers)[:10]}...")
        print(f"Unique experts: {len(eids)}")
        print(f"Roles: {sorted(roles)}")
        
        # Sample expert
        for t in expert_tensors[:5]:
            print(f"  {t['name']}: dims={t['dims']} type={t['type']}")
    
    if len(tensors) > 0:
        data_start = (f.tell() + 31) & ~31
        print(f"\nData start: {data_start}")
        print("[SUCCESS] Tensor parsing works!")
    else:
        print("[FAIL] No tensors parsed")