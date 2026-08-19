#!/usr/bin/env python3
"""Quick fix: manually jump to correct tensor info position for qwen3.6 moe."""

import struct
import sys
sys.path.insert(0, r"D:\alex\flashkv0516\app\edge_engine")
from llama_monkey_patch import parse_gguf_header

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

# From manual trace: after all KVs (correctly parsed), we jump directly
# Let's read header then manually jump to correct position

with open(filepath, "rb") as f:
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    
    print(f"Header: version={version}, n_tensors={n_tensors}, n_kv={n_kv}")
    
    # From manual trace, we know after KV[46], we need to skip 1 byte
    # then KV[47] = tokenizer.chat_template, then KV[48], etc.
    # Let's skip all KVs and jump to tensor info!
    
    # First, parse KV[0]-KV[46] correctly
    kv = {}
    
    for i in range(47):
        klen = struct.unpack("<Q", f.read(8))[0]
        key = f.read(klen).decode("utf-8", errors="replace")
        dtype = struct.unpack("<I", f.read(4))[0]
        
        if dtype == 4:
            val = struct.unpack("<I", f.read(4))[0]
        elif dtype == 6:
            val = struct.unpack("<f", f.read(4))[0]
        elif dtype == 7:
            val = struct.unpack("<Q", f.read(8))[0]
        elif dtype == 8:
            slen = struct.unpack("<Q", f.read(8))[0]
            if slen > 1048576:
                # Skip large string
                remaining = slen
                while remaining > 0:
                    chunk = min(remaining, 100 * 1024 * 1024)
                    _ = f.read(chunk)
                    remaining -= chunk
                val = f"<{slen} bytes>"
            else:
                val = f.read(slen).decode("utf-8", errors="replace") if slen > 0 else ""
        elif dtype == 9:
            elem_type = struct.unpack("<I", f.read(4))[0]
            array_len = struct.unpack("<Q", f.read(8))[0]
            if elem_type == 8:
                # Skip string array
                for _ in range(array_len):
                    slen = struct.unpack("<Q", f.read(8))[0]
                    if slen > 100 * 1024 * 1024:
                        remaining = slen
                        while remaining > 0:
                            chunk = min(remaining, 100 * 1024 * 1024)
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
        
        kv[key] = val
        if i < 10:
            print(f"KV[{i}]: {key} = {val}")
    
    print(f"\nParsed first {len(kv)} KVs")
    
    # Now, manually parse KV[46] add_bos_token as bool (1 byte)
    # Wait we already parsed 47 KVs, so at KV[46] now
    # From manual trace, next is KV[47]: tokenizer.chat_template (dtype=8)
    # And after that, some more KVs, then tensor info!
    
    # Instead, let's jump to where we know tensor info should be!
    # From manual trace: after KV[46], we saw:
    # - KV[46] = add_bos_token (dtype=7, read 8 bytes)
    # - Then there's an extra 00 byte, then KV[47] klen=23
    
    # But let's instead skip all remaining KVs and find tensor info
    
    # First, let's check what's at pos 10935349 (from manual trace):
    f.seek(10935305 + 49)
    print(f"\nChecking at pos {f.tell()}:")
    print(f.read(32).hex())
    
    # Let's try different offsets!
    # We know tensor info starts with:
    # for each tensor: klen(8) + name(klen) + n_dims(4) + dims(n_dims*8) + ggml_type(4) + offset(8)
    
    # Let's search backwards from current position for valid tensor entries
    for offset in [10935305 + 49, 10935305 + 48, 10935305 + 41, 10935305 + 40, 10935305 + 50, 10935305 + 100, 10935305 + 200]:
        try:
            f.seek(offset)
            nlen = struct.unpack("<Q", f.read(8))[0]
            if 0 < nlen <= 100:
                name = f.read(nlen).decode("utf-8", errors="replace")
                if len(name) == nlen and name.isprintable():
                    n_dims = struct.unpack("<I", f.read(4))[0]
                    if 0 < n_dims <= 5:
                        dims = []
                        ok = True
                        for _ in range(n_dims):
                            dim = struct.unpack("<Q", f.read(8))[0]
                            dims.append(dim)
                            if dim > 10000000:
                                ok = False
                        if ok:
                            ggml_type = struct.unpack("<I", f.read(4))[0]
                            tensor_offset = struct.unpack("<Q", f.read(8))[0]
                            print(f"\nFound valid tensor at offset {offset}!")
                            print(f"  name: {name}")
                            print(f"  dims: {dims}")
                            print(f"  ggml_type: {ggml_type}")
                            print(f"  tensor_offset: {tensor_offset}")
                            # This is the right spot!
                            f.seek(offset)
                            break
        except Exception:
            continue
    else:
        # No luck - let's use what we know from file size
        print("\nFallback: estimating tensor info start")
        file_size = 13211155424
        # Estimate that tensor data starts ~10-20% from end
        f.seek(file_size // 2)
        f.seek(10935305 + 1000)
    
    # Now let's manually parse n_tensors tensors!
    print(f"\nParsing tensors at pos {f.tell()}")
    tensors = []
    expert_count = 0
    
    for idx in range(n_tensors):
        nlen_raw = f.read(8)
        if len(nlen_raw) < 8:
            break
        nlen = struct.unpack("<Q", nlen_raw)[0]
        if nlen == 0 or nlen > 1000:
            print(f"ERROR: bad nlen={nlen} at tensor {idx}")
            break
        
        name = f.read(nlen).decode("utf-8", errors="replace")
        
        n_dims_raw = f.read(4)
        if len(n_dims_raw) < 4:
            break
        n_dims = struct.unpack("<I", n_dims_raw)[0]
        
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
        
        tensors.append({
            "name": name,
            "dims": dims,
            "type": ggml_type,
            "offset": tensor_offset,
            "size_bytes": 1,
        })
        
        if idx < 5:
            print(f"Tensor[{idx}]: {name} dims={dims}")
        if "expert" in name.lower():
            expert_count += 1
        if len(tensors) >= n_tensors:
            break
    
    print(f"\nSuccessfully parsed {len(tensors)} tensors!")
    print(f"Expert tensors: {expert_count}")
    
    # Align to 32-byte for data start
    data_start = (f.tell() + 31) & ~31
    
    # Now let's build what ExpertStreamerLite needs!
    print(f"\nVerifying expert structure:")
    for t in tensors:
        parts = t["name"].split(".")
        if len(parts) >= 5 and parts[0] == "blk" and parts[2] == "expert":
            print(f"  {t['name']}")
            break
    
    # Check we have the right key prefix
    prefix = None
    for t in tensors:
        if "block_count" in kv or "blk." in t["name"]:
            if "qwen35moe" in kv.get("general.architecture", "qwen35moe"):
                prefix = "qwen35moe"
            break
    print(f"\nArchitecture: {kv.get('general.architecture')}")
    print(f"Block count: {kv.get(f'{prefix}.block_count', kv.get('general.block_count', 0))}")
    print(f"Embedding length: {kv.get(f'{prefix}.embedding_length', 0)}")
    print(f"Expert count: {kv.get(f'{prefix}.expert_count', 0)}")
