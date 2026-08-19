#!/usr/bin/env python3
"""Read top_p value bytes directly."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    # Skip to known position
    f.seek(24)
    
    # KV[0]: general.architecture (dtype=8, string)
    klen = struct.unpack("<Q", f.read(8))[0]
    key = f.read(klen).decode()
    dtype = struct.unpack("<I", f.read(4))[0]
    print(f"KV[0]: '{key}' dtype={dtype}")
    slen = struct.unpack("<Q", f.read(8))[0]
    val = f.read(slen).decode()
    print(f"  = '{val}'")
    
    # KV[1]: general.type (dtype=8, string)
    klen = struct.unpack("<Q", f.read(8))[0]
    key = f.read(klen).decode()
    dtype = struct.unpack("<I", f.read(4))[0]
    print(f"KV[1]: '{key}' dtype={dtype}")
    slen = struct.unpack("<Q", f.read(8))[0]
    val = f.read(slen).decode()
    print(f"  = '{val}'")
    
    # KV[2]: general.sampling.top_k (dtype=5, uint32)
    klen = struct.unpack("<Q", f.read(8))[0]
    key = f.read(klen).decode()
    dtype = struct.unpack("<I", f.read(4))[0]
    print(f"KV[2]: '{key}' dtype={dtype}")
    val_raw = f.read(4)
    val_u32 = struct.unpack("<I", val_raw)[0]
    val_f32 = struct.unpack("<f", val_raw)[0]
    print(f"  raw={val_raw.hex()} u32={val_u32} f32={val_f32}")
    
    # KV[3]: general.sampling.top_p
    klen = struct.unpack("<Q", f.read(8))[0]
    key = f.read(klen).decode()
    dtype = struct.unpack("<I", f.read(4))[0]
    print(f"KV[3]: '{key}' dtype={dtype}")
    # Read next 8 bytes raw
    val_raw = f.read(8)
    print(f"  raw (8 bytes): {val_raw.hex()}")
    # Try various interpretations
    u16 = struct.unpack("<H", val_raw[:2])[0]
    u32 = struct.unpack("<I", val_raw[:4])[0]
    f32 = struct.unpack("<f", val_raw[:4])[0]
    u64 = struct.unpack("<Q", val_raw)[0]
    f64 = struct.unpack("<d", val_raw)[0]
    print(f"  As u16={u16}, u32={u32}, f32={f32}, u64={u64}, f64={f64}")
    
    # float32(0.95) = 0x3F733333
    target = struct.pack("<f", 0.95)
    print(f"  Target f32(0.95) = {target.hex()}")
    
    # KV[4]: general.sampling.temp
    # Position ourselves after top_p's value
    # If dtype=6 means float32(4 bytes):
    # After top_p's dtype(4) + 4 bytes value = we are 8 bytes into the 8 we read
    # Let's just check: from current position, the next KV
    
    # Actually let me re-read with correct positioning
    f.seek(24)
    for i in range(4):
        klen = struct.unpack("<Q", f.read(8))[0]
        key = f.read(klen).decode()
        dtype = struct.unpack("<I", f.read(4))[0]
        
        if dtype == 8:  # string
            slen = struct.unpack("<Q", f.read(8))[0]
            f.seek(slen, 1)
            print(f"KV[{i}]: '{key}' dtype=8(string)")
        elif dtype == 5:  # uint32
            v = struct.unpack("<I", f.read(4))[0]
            print(f"KV[{i}]: '{key}' dtype=5(uint32)={v}")
        elif dtype == 6:  # float32 (4 bytes)
            v = struct.unpack("<f", f.read(4))[0]
            print(f"KV[{i}]: '{key}' dtype=6(float32)={v}")
        elif dtype == 4:  # uint16 (2 bytes)
            v = struct.unpack("<H", f.read(2))[0]
            print(f"KV[{i}]: '{key}' dtype=4(uint16)={v}")
        elif dtype == 9:  # bool
            v = struct.unpack("<?", f.read(1))[0]
            print(f"KV[{i}]: '{key}' dtype=9(bool)={v}")
        elif dtype == 7:  # uint64
            v = struct.unpack("<Q", f.read(8))[0]
            print(f"KV[{i}]: '{key}' dtype=7(uint64)={v}")
        elif dtype == 11:  # float64
            v = struct.unpack("<d", f.read(8))[0]
            print(f"KV[{i}]: '{key}' dtype=11(float64)={v}")
        elif dtype == 10:  # int32
            v = struct.unpack("<i", f.read(4))[0]
            print(f"KV[{i}]: '{key}' dtype=10(int32)={v}")
    
    print(f"\nAfter 4 KVs: pos={f.tell()}")
    
    # Try KV[4] (temp)
    klen = struct.unpack("<Q", f.read(8))[0]
    key = f.read(klen).decode()
    dtype = struct.unpack("<I", f.read(4))[0]
    print(f"KV[4]: '{key}' dtype={dtype}")
    val_raw = f.read(8)
    print(f"  raw: {val_raw.hex()}")
    f32 = struct.unpack("<f", val_raw[:4])[0]
    print(f"  as float32: {f32}")