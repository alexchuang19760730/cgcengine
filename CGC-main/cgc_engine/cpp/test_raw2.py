#!/usr/bin/env python3
"""Detailed byte-level analysis of KV[46] area."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    f.seek(10935305)
    
    # Dump 80 bytes
    raw = f.read(80)
    print("All bytes from pos 10935305:")
    for i in range(0, len(raw), 16):
        hex_str = raw[i:i+16].hex()
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in raw[i:i+16])
        print(f"  {i:4d}: {hex_str}  {ascii_str}")
    
    print("\n" + "="*60)
    
    # GGUF spec: each KV = klen(8) + key(klen) + dtype(4) + value
    # For add_bos_token, typical dtype in different GGUF versions:
    # v2: dtype=9 (bool, 1 byte)
    # v3: dtype=4 (uint32, 4 bytes)
    
    print("\nAttempt 1: dtype=4 (uint32, 4B value)")
    pos = 0
    klen = struct.unpack("<Q", raw[pos:pos+8])[0]
    pos += 8
    key = raw[pos:pos+klen].decode("utf-8")
    pos += klen
    dtype = struct.unpack("<I", raw[pos:pos+4])[0]
    pos += 4
    val = struct.unpack("<I", raw[pos:pos+4])[0]
    pos += 4
    print(f"  key='{key}' dtype={dtype} val={val}")
    print(f"  Next KV at offset {pos}: klen = {struct.unpack('<Q', raw[pos:pos+8])[0]}")
    
    print("\nAttempt 2: dtype=7 (uint64, 8B value)")
    pos = 0
    klen = struct.unpack("<Q", raw[pos:pos+8])[0]
    pos += 8
    key = raw[pos:pos+klen].decode("utf-8")
    pos += klen
    dtype = struct.unpack("<I", raw[pos:pos+4])[0]
    pos += 4
    val = struct.unpack("<Q", raw[pos:pos+8])[0]
    pos += 8
    print(f"  key='{key}' dtype={dtype} val={val}")
    print(f"  Next KV at offset {pos}: klen = {struct.unpack('<Q', raw[pos:pos+8])[0]}")
    
    print("\nAttempt 3: dtype=9 (bool, 1B value)")
    pos = 0
    klen = struct.unpack("<Q", raw[pos:pos+8])[0]
    pos += 8
    key = raw[pos:pos+klen].decode("utf-8")
    pos += klen
    dtype = struct.unpack("<I", raw[pos:pos+4])[0]
    pos += 4
    val = raw[pos]
    pos += 1
    print(f"  key='{key}' dtype={dtype} val={val}")
    print(f"  Next KV at offset {pos}: klen = {struct.unpack('<Q', raw[pos:pos+8])[0]}")
    
    # Let's try from a known-good position
    print("\n" + "="*60)
    print("Searching for 'tokenizer.chat_template' or 'general.' in raw bytes...")
    for offset in range(0, len(raw) - 20):
        snippet = raw[offset:offset+20]
        text = snippet.decode("utf-8", errors="replace")
        if "tokenizer" in text or "chat_template" in text.lower():
            print(f"  Offset {offset}: {snippet.hex()} => '{text}'")
    
    # Also check: maybe add_bos_token is actually a bool/array?
    print("\n" + "="*60)
    print("Maybe 'add_bos_token' is actually an array?")
    pos = 0
    klen = struct.unpack("<Q", raw[pos:pos+8])[0]
    pos += 8
    key = raw[pos:pos+klen].decode("utf-8")
    pos += klen
    dtype = struct.unpack("<I", raw[pos:pos+4])[0]
    print(f"  key='{key}' dtype={dtype}")
    if dtype == 9:  # ARRAY
        elem_type = struct.unpack("<I", raw[pos+4:pos+8])[0]
        array_len = struct.unpack("<Q", raw[pos+8:pos+16])[0]
        print(f"  elem_type={elem_type} array_len={array_len}")
        if elem_type == 4:
            total = array_len * 4
            print(f"  Total bytes for array: {total}")
            # Skip header (4+8=12 bytes) + data
            pos += 12 + total
            print(f"  Next KV at offset {pos}: klen = {struct.unpack('<Q', raw[pos:pos+8])[0]}")