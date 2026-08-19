#!/usr/bin/env python3
"""Trace first 17 KV items byte-by-byte."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    
    print(f"Version: {version}, n_tensors: {n_tensors}, n_kv: {n_kv}")
    print(f"Start pos: {f.tell()}")
    
    for i in range(n_kv):
        pos = f.tell()
        
        # Read klen
        klen_raw = f.read(8)
        klen = struct.unpack("<Q", klen_raw)[0]
        
        if klen > 1048576:
            print(f"\nKV[{i}] at {pos}: klen={klen} TOO LARGE, hex={klen_raw.hex()}")
            print(f"  Context: {f.read(min(64,klen)).hex() if klen <= 64 else '...'}")
            break
        
        # Read key
        if klen == 0:
            key = ""
        else:
            key_bytes = f.read(klen)
            key = key_bytes.decode("utf-8", errors="replace")
        
        # Read dtype
        dtype_raw = f.read(4)
        dtype = struct.unpack("<I", dtype_raw)[0]
        
        print(f"KV[{i}]: pos={pos} key='{key}' dtype={dtype} ", end="")
        
        # Now decode value based on dtype
        # GGUF 3.0 spec:
        # 0=null, 1=bool, 2=uint8, 3=int8, 4=uint16, 5=int16
        # 6=uint32, 7=int32, 8=uint64, 9=int64
        # 10=float32, 11=float64, 12=string, 13=array, 14=object
        
        if dtype == 0:
            print("= NULL")
        elif dtype == 1:
            v = struct.unpack("<?", f.read(1))[0]
            print(f"= bool({v})")
        elif dtype == 2:
            v = struct.unpack("<B", f.read(1))[0]
            print(f"= uint8({v})")
        elif dtype == 3:
            v = struct.unpack("<b", f.read(1))[0]
            print(f"= int8({v})")
        elif dtype == 4:
            v = struct.unpack("<H", f.read(2))[0]
            print(f"= uint16({v})")
        elif dtype == 5:
            v = struct.unpack("<h", f.read(2))[0]
            print(f"= int16({v})")
        elif dtype == 6:
            v = struct.unpack("<I", f.read(4))[0]
            print(f"= uint32({v})")
        elif dtype == 7:
            v = struct.unpack("<i", f.read(4))[0]
            print(f"= int32({v})")
        elif dtype == 8:
            v = struct.unpack("<Q", f.read(8))[0]
            print(f"= uint64({v})")
        elif dtype == 9:
            v = struct.unpack("<q", f.read(8))[0]
            print(f"= int64({v})")
        elif dtype == 10:
            v = struct.unpack("<f", f.read(4))[0]
            print(f"= float32({v})")
        elif dtype == 11:
            v = struct.unpack("<d", f.read(8))[0]
            print(f"= float64({v})")
        elif dtype == 12:  # STRING
            slen = struct.unpack("<Q", f.read(8))[0]
            if slen > 1048576:
                print(f"= LARGE_STRING(slen={slen})")
                try:
                    f.seek(slen)
                except:
                    _ = f.read(slen)
            else:
                v = f.read(slen).decode("utf-8", errors="replace") if slen > 0 else ""
                print(f"= string('{v}')")
        elif dtype == 13:  # ARRAY
            elem_type = struct.unpack("<I", f.read(4))[0]
            array_len = struct.unpack("<Q", f.read(8))[0]
            
            elem_sizes = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 4, 8: 8, 9: 8, 10: 4, 11: 8}
            
            if elem_type == 12:  # STRING array
                print(f"= ARRAY[{array_len}] of STRING")
                # Show first 3
                for j in range(min(array_len, 3)):
                    sl = struct.unpack("<Q", f.read(8))[0]
                    sv = f.read(sl).decode("utf-8", errors="replace") if sl > 0 else ""
                    print(f"    [{j}]: '{sv}'")
                if array_len > 3:
                    # Skip rest
                    for _ in range(array_len - 3):
                        sl = struct.unpack("<Q", f.read(8))[0]
                        try:
                            f.seek(sl)
                        except:
                            _ = f.read(sl)
                    print(f"    ... ({array_len - 3} more)")
            else:
                es = elem_sizes.get(elem_type, 0)
                print(f"= ARRAY[{array_len}] of type {elem_type} (size={es})")
                total = es * array_len
                try:
                    f.seek(total, 1)
                except:
                    _ = f.read(total)
        elif dtype == 14:  # OBJECT
            print("= OBJECT (stopping)")
            break
        else:
            print(f"= UNKNOWN dtype={dtype}")
            break
    
    print(f"\nAfter KV loop, pos={f.tell()}")
    
    # Try first tensor
    nlen = struct.unpack("<Q", f.read(8))[0]
    print(f"First tensor nlen={nlen}")
    if nlen <= 65536:
        name = f.read(nlen).decode("utf-8", errors="replace")
        print(f"First tensor name: {name}")
    else:
        print("MISALIGNED! Checking raw bytes...")
        f.seek(pos)
        raw = f.read(64)
        print(f"Raw at {pos}: {raw.hex()}")