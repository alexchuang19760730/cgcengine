#!/usr/bin/env python3
"""Verify bos_token_id and add_bos_token dtypes by raw byte inspection."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    # Check KV[45] bos_token_id at pos 10935262
    f.seek(10935262)
    raw_45 = f.read(60)
    print("KV[45] bos_token_id raw bytes:")
    for i in range(0, 60, 16):
        print(f"  {i:4d}: {raw_45[i:i+16].hex()}")
    
    # Parse manually
    klen = struct.unpack("<Q", raw_45[0:8])[0]
    key = raw_45[8:8+klen].decode("utf-8")
    dtype = struct.unpack("<I", raw_45[8+klen:8+klen+4])[0]
    print(f"\nKV[45]: klen={klen}, key='{key}', dtype={dtype}")
    
    # Check dtype 4 (uint32, 4B)
    val_4 = struct.unpack("<I", raw_45[8+klen+4:8+klen+4+4])[0]
    print(f"  If uint32 (4B): val={val_4}")
    next_after_4 = raw_45[8+klen+4+4:8+klen+4+4+8]
    next_klen_4 = struct.unpack("<Q", next_after_4)[0]
    print(f"    Next klen: {next_klen_4}")
    
    # Check dtype 7 (uint64, 8B)
    val_8 = struct.unpack("<Q", raw_45[8+klen+4:8+klen+4+8])[0]
    print(f"  If uint64 (8B): val={val_8}")
    next_after_8 = raw_45[8+klen+4+8:8+klen+4+8+8]
    next_klen_8 = struct.unpack("<Q", next_after_8)[0]
    print(f"    Next klen: {next_klen_8}")
    
    # Check dtype 9 (bool, 1B)
    val_1 = raw_45[8+klen+4]
    print(f"  If bool (1B): val={val_1}")
    next_after_1 = raw_45[8+klen+4+1:8+klen+4+1+8]
    next_klen_1 = struct.unpack("<Q", next_after_1)[0]
    print(f"    Next klen: {next_klen_1}")
    
    print(f"\n=== Now check KV[46] add_bos_token ===")
    # We know KV[45] ends at: 10935262 + 8 + 27 + 4 + 4 = 10935305 (if uint32)
    # Or: 10935262 + 8 + 27 + 4 + 8 = 10935309 (if uint64)
    # Or: 10935262 + 8 + 27 + 4 + 1 = 10935302 (if bool)
    
    for pos_name, pos_val in [("after uint32", 10935305), ("after uint64", 10935309), ("after bool", 10935302)]:
        f.seek(pos_val)
        klen46_raw = f.read(8)
        klen46 = struct.unpack("<Q", klen46_raw)[0]
        print(f"  KV[46] at {pos_name} ({pos_val}): klen={klen46}")
        if 3 <= klen46 <= 65536:
            key46 = f.read(klen46).decode("utf-8", errors="replace")
            dtype46 = struct.unpack("<I", f.read(4))[0]
            print(f"    key='{key46}', dtype={dtype46}")
        else:
            print(f"    INVALID klen={klen46}")