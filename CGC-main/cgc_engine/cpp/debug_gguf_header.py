#!/usr/bin/env python3
"""Simple GGUF header dump."""

import struct

filepath = r"D:\alex\flashkv0516\models\gemma4_gguf\gemma-4-26B-A4B-it-UD-IQ3_S.gguf"

with open(filepath, "rb") as f:
    data = f.read(2048)  # Read first 2KB

# Header
magic = struct.unpack("<I", data[:4])[0]
version = struct.unpack("<I", data[4:8])[0]
n_tensors = struct.unpack("<Q", data[8:16])[0]
n_kv = struct.unpack("<Q", data[16:24])[0]

print(f"Magic: 0x{magic:x} ('{data[:4].decode('ascii', errors='replace')}')")
print(f"Version: {version}")
print(f"Tensors: {n_tensors}")
print(f"KV items: {n_kv}")

# Print raw bytes from offset 24 to 200
print(f"\nRaw bytes 24-200:")
for i in range(24, min(200, len(data)), 16):
    chunk = data[i:i+16]
    hex_str = ' '.join(f'{b:02x}' for b in chunk)
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
    print(f"  {i:4d}: {hex_str}  {ascii_str}")

# Manually parse first few KV items
print(f"\nFirst KV item parsing:")
pos = 24
for item in range(min(5, n_kv)):
    # Read string length
    str_len_raw = data[pos:pos+4]
    str_len = struct.unpack("<I", str_len_raw)[0]
    print(f"  pos={pos}: str_len_raw={str_len_raw.hex()}, str_len={str_len}")
    
    if str_len > 1000:
        print(f"    ERROR: str_len too large, likely wrong format")
        print(f"    Next 20 bytes: {data[pos:pos+20].hex()}")
        break
    
    key = data[pos+4:pos+4+str_len].decode('utf-8', errors='replace')
    print(f"    key: '{key}'")
    pos += 4 + str_len
    
    # Read dtype
    dtype = struct.unpack("<I", data[pos:pos+4])[0]
    print(f"    dtype: {dtype} (at pos {pos})")
    pos += 4
    
    # Read value based on dtype
    if dtype == 7:  # BOOL
        val = struct.unpack("<I", data[pos:pos+4])[0]
        print(f"    BOOL value: {val}")
        pos += 4
    elif dtype == 8:  # STRING
        str_len2 = struct.unpack("<I", data[pos:pos+4])[0]
        print(f"    string len: {str_len2}")
        if str_len2 > 1000:
            print(f"      ERROR: string len too large")
            break
        val = data[pos+4:pos+4+str_len2].decode('utf-8', errors='replace')
        print(f"    string value: '{val}'")
        pos += 4 + str_len2
    elif dtype == 9:  # ARRAY
        arr_type = struct.unpack("<I", data[pos:pos+4])[0]
        arr_count = struct.unpack("<Q", data[pos+4:pos+12])[0]
        print(f"    array: type={arr_type}, count={arr_count}")
        pos += 12
        
        if arr_type == 8:  # Array of strings
            arr = []
            for _ in range(min(arr_count, 5)):
                s_len = struct.unpack("<I", data[pos:pos+4])[0]
                if s_len > 10000:
                    print(f"      ERROR: string len {s_len} too large")
                    break
                s = data[pos+4:pos+4+s_len].decode('utf-8', errors='replace')
                arr.append(s)
                pos += 4 + s_len
            print(f"    first strings: {arr[:3]}")
        else:
            # Skip based on arr_type
            type_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 4, 10: 8, 11: 8, 12: 8}
            elem_size = type_sizes.get(arr_type, 4)
            pos += arr_count * elem_size
    elif dtype == 4:  # UINT32
        val = struct.unpack("<I", data[pos:pos+4])[0]
        print(f"    UINT32 value: {val}")
        pos += 4
    elif dtype == 5:  # INT32
        val = struct.unpack("<i", data[pos:pos+4])[0]
        print(f"    INT32 value: {val}")
        pos += 4
    elif dtype == 6:  # FLOAT32
        val = struct.unpack("<f", data[pos:pos+4])[0]
        print(f"    FLOAT32 value: {val}")
        pos += 4
    else:
        print(f"    Unknown dtype, stopping")
        break
    
    print(f"  New pos: {pos}")
