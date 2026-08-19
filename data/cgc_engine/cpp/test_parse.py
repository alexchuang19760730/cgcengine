#!/usr/bin/env python3
"""
Minimal GGUF parser test - just get past item 10 to find the bug.
"""
import struct
import os

def big_skip(f, n):
    CHUNK = 4 * 1024 * 1024
    while n > 0:
        d = f.read(min(CHUNK, n))
        if not d:
            break
        n -= len(d)

path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
f = open(path, "rb")

# Read header
magic = struct.unpack("<I", f.read(4))[0]
version = struct.unpack("<I", f.read(4))[0]
n_tensors = struct.unpack("<Q", f.read(8))[0]
n_kv = struct.unpack("<Q", f.read(8))[0]

print(f"Version: {version}, n_kv: {n_kv}")

for idx in range(54):
    item_start = f.tell()
    
    klen_bytes = f.read(8)
    if len(klen_bytes) < 8:
        print(f"  [{idx}] EOF at klen, pos={item_start}")
        break
    klen = struct.unpack("<Q", klen_bytes)[0]
    
    if klen == 0 or klen > 1000000:
        print(f"  [{idx}] INVALID klen={klen}, raw={klen_bytes.hex()}, pos={item_start}")
        break
    
    key_bytes = f.read(klen)
    if len(key_bytes) < klen:
        print(f"  [{idx}] EOF at key, pos={item_start}")
        break
    key = key_bytes.decode("utf-8", errors="replace")
    
    dtype_bytes = f.read(4)
    if len(dtype_bytes) < 4:
        break
    dtype = struct.unpack("<I", dtype_bytes)[0]
    
    val_start = f.tell()
    
    if dtype == 7:  # bool: 1 byte
        v = f.read(1)
        if len(v) != 1:
            print(f"  [{idx}] bool read fail, pos={val_start}")
            break
        
    elif dtype == 8:  # string
        slen_bytes = f.read(8)
        if len(slen_bytes) < 8:
            break
        slen = struct.unpack("<Q", slen_bytes)[0]
        
        # CORRECT approach: read all bytes
        s_bytes = f.read(slen)
        if len(s_bytes) < slen:
            print(f"  [{idx}] string read fail: expected {slen}, got {len(s_bytes)}, pos={val_start + 8}")
            break
        s = s_bytes.decode("utf-8", errors="replace")
        
    elif dtype == 9:  # array
        et_bytes = f.read(4)
        al_bytes = f.read(8)
        if len(et_bytes) < 4 or len(al_bytes) < 8:
            break
        elem_type = struct.unpack("<I", et_bytes)[0]
        array_len = struct.unpack("<Q", al_bytes)[0]
        
        if elem_type == 8:  # string array
            for i in range(array_len):
                sl_bytes = f.read(8)
                if len(sl_bytes) < 8:
                    break
                sl = struct.unpack("<Q", sl_bytes)[0]
                s_bytes = f.read(sl)
                if len(s_bytes) < sl:
                    print(f"  [{idx}] string {i} read fail: expected {sl}, got {len(s_bytes)}")
                    break
        elif elem_type in (4, 5, 6, 10):
            elem_size = 8 if elem_type == 10 else 4
            total = elem_size * array_len
            f.seek(total, 1)  # Use seek for large arrays
        elif elem_type == 7:
            f.seek(array_len, 1)  # 1-byte bool
        else:
            f.seek(4 * array_len, 1)
            
    else:
        sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 10:8, 11:8, 12:8}
        s = sizes.get(dtype, 4)
        v = f.read(s)
        if len(v) != s:
            break
    
    item_end = f.tell()
    item_size = item_end - item_start
    
    if item_size > 1000 or idx < 20 or "tokenizer" in key or "add_bos" in key:
        print(f"  [{idx:3d}] @{item_start:12d} -> @{item_end:12d} ({item_size:8d}B) dtype={dtype:2d} key='{key[:70]}'")

print(f"\nAfter {idx+1} items, pos={f.tell()}")

f.close()
