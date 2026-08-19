#!/usr/bin/env python3
"""Verify KV[46] raw bytes."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    # Skip header (24 bytes) + all KVs up to [46]
    f.seek(10935305)  # Position of KV[46]
    
    # Read KV[46] raw
    raw = f.read(60)
    print("KV[46] raw bytes (hex):")
    print(raw.hex())
    print()
    
    # Parse manually
    klen = struct.unpack("<Q", raw[0:8])[0]
    print(f"klen = {klen}")
    print(f"klen bytes as text: {raw[0:8]}")
    
    key_end = 8 + klen
    key = raw[8:key_end]
    print(f"key = {key}")
    
    dtype = struct.unpack("<I", raw[key_end:key_end+4])[0]
    print(f"dtype = {dtype}")
    
    # After dtype
    pos = key_end + 4
    print(f"Bytes after dtype (raw[40:60]): {raw[pos:pos+20].hex()}")
    print(f"As uint64: {struct.unpack('<Q', raw[pos:pos+8])[0]}")
    
    # Now check what comes next
    next_pos = key_end + 4 + 8  # After uint64 value
    print(f"\nNext KV starts at raw[{next_pos}:]:")
    next_klen = struct.unpack("<Q", raw[next_pos:next_pos+8])[0]
    print(f"next klen = {next_klen}")
    print(f"next klen hex: {raw[next_pos:next_pos+8].hex()}")
    print(f"next key would be: {raw[next_pos+8:next_pos+8+min(next_klen, 50)]}")