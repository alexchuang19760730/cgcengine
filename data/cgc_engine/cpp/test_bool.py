#!/usr/bin/env python3
"""Verify exact bytes around general.tags."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    # Go to known position before general.tags
    # From test_exact_pos.py, KV[15] (general.base_model.0.repo_url) ends at pos 874
    # So KV[16] starts at pos 874
    
    # Let's read from pos 870 to see raw bytes
    f.seek(870)
    raw = f.read(50)
    print(f"Raw bytes from pos 870: {raw.hex()}")
    print(f"As text: {raw}")
    
    # Now manually parse
    f.seek(874)
    
    klen = struct.unpack("<Q", f.read(8))[0]
    print(f"\nklen = {klen}")
    
    key = f.read(klen).decode()
    print(f"key = '{key}'")
    
    dtype = struct.unpack("<I", f.read(4))[0]
    print(f"dtype = {dtype}")
    
    # Read 8 bytes for bool
    val_raw = f.read(8)
    val = struct.unpack("<Q", val_raw)[0]
    print(f"bool value (uint64) = {val}")
    print(f"val raw bytes: {val_raw.hex()}")
    
    print(f"\nPosition after bool: {f.tell()}")
    
    # Now read next klen
    next_klen = struct.unpack("<Q", f.read(8))[0]
    print(f"Next klen: {next_klen}")
    if next_klen <= 256:
        next_key = f.read(next_klen).decode()
        print(f"Next key: '{next_key}'")
        next_dtype = struct.unpack("<I", f.read(4))[0]
        print(f"Next dtype: {next_dtype}")
    else:
        print("Next klen too large - MISALIGNED!")