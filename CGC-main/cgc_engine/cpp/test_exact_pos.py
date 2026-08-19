#!/usr/bin/env python3
"""Calculate exact byte positions."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    # Skip header
    f.seek(24)
    
    # Manually parse known KVs with exact byte counting
    expected_kvs = [
        ("general.architecture", 8, "qwen35moe"),  # dtype=8 string
        ("general.type", 8, "model"),
        ("general.sampling.top_k", 5, 20),  # dtype=5 uint32
        ("general.sampling.top_p", 6, 0.95),  # dtype=6 float32
        ("general.sampling.temp", 6, 1.0),
        ("general.name", 8, "Qwen3.6-35B-A3B"),
        ("general.basename", 8, "Qwen3.6-35B-A3B"),
        ("general.quantized_by", 8, "Unsloth"),
        ("general.size_label", 8, "35B-A3B"),
        ("general.license", 8, "apache-2.0"),
        ("general.license.link", 8, "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE"),
        ("general.repo_url", 8, "https://huggingface.co/unsloth"),
        ("general.base_model.count", 4, 1),  # dtype=4 uint32
        ("general.base_model.0.name", 8, "Qwen3.6 35B A3B"),
        ("general.base_model.0.organization", 8, "Qwen"),
        ("general.base_model.0.repo_url", 8, "https://huggingface.co/Qwen/Qwen3.6-35B-A3B"),
        ("general.tags", 9, True),  # dtype=9 bool
    ]
    
    for i, (key, dtype, expected) in enumerate(expected_kvs):
        pos = f.tell()
        klen = struct.unpack("<Q", f.read(8))[0]
        actual_key = f.read(klen).decode()
        actual_dtype = struct.unpack("<I", f.read(4))[0]
        
        if dtype == 8:
            slen = struct.unpack("<Q", f.read(8))[0]
            val = f.read(slen).decode() if slen > 0 else ""
        elif dtype in (4, 5):
            val = struct.unpack("<I", f.read(4))[0]
        elif dtype == 6:
            val = struct.unpack("<f", f.read(4))[0]
        elif dtype == 9:
            val = struct.unpack("<?", f.read(1))[0]
        elif dtype == 7:
            val = struct.unpack("<Q", f.read(8))[0]
        elif dtype == 10:
            val = struct.unpack("<i", f.read(4))[0]
        elif dtype == 11:
            val = struct.unpack("<d", f.read(8))[0]
        
        ok = actual_key == key and actual_dtype == dtype
        print(f"KV[{i}]: pos={pos} key='{actual_key}' dtype={actual_dtype} val={val} {'OK' if ok else 'FAIL'}")
    
    pos_after_tags = f.tell()
    print(f"\nAfter general.tags (KV[16]): pos={pos_after_tags}")
    
    # Now show next 128 bytes
    raw = f.read(128)
    print(f"Next 128 bytes: {raw.hex()}")
    
    # Parse next KV item manually
    f.seek(pos_after_tags)
    klen = struct.unpack("<Q", f.read(8))[0]
    print(f"\nNext klen: {klen}")
    if klen <= 256:
        key = f.read(klen).decode()
        print(f"Next key: '{key}'")
        dtype = struct.unpack("<I", f.read(4))[0]
        print(f"Next dtype: {dtype}")
        if dtype == 8:
            slen = struct.unpack("<Q", f.read(8))[0]
            print(f"Next string slen: {slen}")
            if slen <= 256:
                val = f.read(slen).decode()
                print(f"Next value: '{val}'")
        elif dtype in (4, 5):
            val = struct.unpack("<I", f.read(4))[0]
            print(f"Next value (uint32): {val}")
        elif dtype == 6:
            val = struct.unpack("<f", f.read(4))[0]
            print(f"Next value (float32): {val}")
        elif dtype == 9:
            val = struct.unpack("<?", f.read(1))[0]
            print(f"Next value (bool): {val}")
        elif dtype == 7:
            val = struct.unpack("<Q", f.read(8))[0]
            print(f"Next value (uint64): {val}")
    else:
        print(f"klen too large ({klen}), examining raw bytes...")
        # The raw bytes from pos_after_tags
        raw = f.read(min(256, klen))
        print(f"First 256 bytes of key: {raw.hex()}")
        print(f"As text: {raw[:100]}")