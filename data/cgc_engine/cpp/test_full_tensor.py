#!/usr/bin/env python3
"""Full tensor parsing from correct position for qwen35moe."""

import struct
import os

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
file_size = os.path.getsize(filepath)

with open(filepath, "rb") as f:
    # Skip header
    f.seek(24)
    
    # Parse all 54 KVs (same as test_bool_fix.py)
    kv = {}
    BOOL_KEYS = {
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
        "tokenizer.ggml.add_space",
    }
    
    for i in range(54):
        klen = struct.unpack("<Q", f.read(8))[0]
        if klen > 65536:
            break
        key = f.read(klen).decode("utf-8", errors="replace")
        dtype = struct.unpack("<I", f.read(4))[0]
        
        if key in BOOL_KEYS and dtype == 7:
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
            probe = f.read(12)
            elem_type = struct.unpack("<I", probe[0:4])[0]
            array_len = struct.unpack("<Q", probe[4:12])[0]
            if array_len < 10000000 and elem_type in (4,5,6,7,8,10,11,12):
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
    
    print(f"KV parsing done. Pos after KVs: {f.tell()}")
    
    # Now find correct tensor start
    # From search: tensor info starts at 10943832
    tensor_info_start = 10943832
    f.seek(tensor_info_start)
    
    n_tensors = 733  # From header
    
    tensors = []
    expert_tensors = []
    
    for idx in range(n_tensors):
        nlen_raw = f.read(8)
        if len(nlen_raw) < 8:
            break
        nlen = struct.unpack("<Q", nlen_raw)[0]
        if nlen == 0 or nlen > 256:
            print(f"ERROR: Bad nlen={nlen} at tensor {idx}, pos {f.tell()-8}")
            break
        
        name = f.read(nlen).decode("utf-8", errors="replace")
        
        n_dims_raw = f.read(4)
        if len(n_dims_raw) < 4:
            break
        n_dims = struct.unpack("<I", n_dims_raw)[0]
        if n_dims > 6:
            print(f"ERROR: Bad n_dims={n_dims} for '{name}'")
            break
        
        dims = []
        for _ in range(n_dims):
            dim_raw = f.read(8)
            if len(dim_raw) < 8:
                break
            dims.append(struct.unpack("<Q", dim_raw)[0])
        
        ggml_type_raw = f.read(4)
        if len(ggml_type_raw) < 4:
            break
        ggml_type = struct.unpack("<I", ggml_type_raw)[0]
        
        offset_raw = f.read(8)
        if len(offset_raw) < 8:
            break
        tensor_offset = struct.unpack("<Q", offset_raw)[0]
        
        # Calculate size in bytes
        GGML_TYPE_BYTES = {0:4, 1:2, 2:2, 3:2, 4:2, 5:2, 6:2, 7:1, 8:8, 9:8, 10:4, 11:2, 12:2, 13:2, 14:4, 15:4, 16:4, 17:4, 18:3, 19:3, 20:3, 21:3, 22:3, 23:4, 24:4, 25:4, 26:4, 27:4, 28:3, 29:4, 30:2}
        bpe = GGML_TYPE_BYTES.get(ggml_type, 4)
        n_elements = 1
        for d in dims:
            n_elements *= d
        size_bytes = int(n_elements * bpe)
        
        tensor = {
            "name": name,
            "dims": dims,
            "type": ggml_type,
            "offset": tensor_offset,
            "size_bytes": size_bytes,
        }
        tensors.append(tensor)
        
        if "expert" in name.lower():
            expert_tensors.append(tensor)
        
        if idx < 10:
            print(f"[{idx}] {name}: dims={dims} type={ggml_type} offset={tensor_offset} size={size_bytes}")
    
    print(f"\n=== RESULTS ===")
    print(f"Total tensors parsed: {len(tensors)}")
    print(f"Expert tensors: {len(expert_tensors)}")
    
    if expert_tensors:
        # Analyze expert structure
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
        
        print(f"Layers with experts: {len(layers)} layers, e.g. {sorted(layers)[:5]}")
        print(f"Unique expert IDs: {len(eids)}")
        print(f"Expert roles: {sorted(roles)}")
        
        # Sample experts
        print(f"\nSample expert tensors:")
        for t in expert_tensors[:10]:
            print(f"  {t['name']}: dims={t['dims']} type={t['type']} offset={t['offset']} size={t['size_bytes']}")
    
    # Data start
    data_start = (f.tell() + 31) & ~31
    print(f"\nData start (32-byte aligned): {data_start}")
    
    if len(tensors) > 0:
        print("\n[SUCCESS] Full GGUF tensor parsing verified!")
    else:
        print("\n[FAIL] No tensors parsed")