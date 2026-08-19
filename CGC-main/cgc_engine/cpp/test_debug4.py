#!/usr/bin/env python3
"""Debug with corrected dtype mapping."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    
    print(f"Version: {version}, n_tensors: {n_tensors}, n_kv: {n_kv}")
    print(f"Start pos: {f.tell()}")
    
    dtype_names = {
        0: "NULL", 1: "BOOL", 2: "UINT8", 3: "INT8",
        4: "UINT16", 5: "INT16", 6: "UINT32", 7: "INT32",
        8: "UINT64", 9: "INT64", 10: "FLOAT32", 11: "FLOAT64",
        12: "STRING", 13: "ARRAY", 14: "OBJECT"
    }
    
    for i in range(min(n_kv, 20)):
        pos_before = f.tell()
        
        klen_raw = f.read(8)
        if len(klen_raw) < 8:
            print(f"KV[{i}]: EOF at klen read")
            break
        klen = struct.unpack("<Q", klen_raw)[0]
        
        if klen > 1048576:
            print(f"KV[{i}]: klen={klen} LARGE, skipping")
            try:
                f.seek(klen)
            except:
                _ = f.read(klen)
            key = f"<skipped>"
        else:
            key = f.read(klen).decode("utf-8", errors="replace")
        
        dtype_raw = f.read(4)
        if len(dtype_raw) < 4:
            print(f"KV[{i}]: EOF at dtype read")
            break
        dtype = struct.unpack("<I", dtype_raw)[0]
        dtype_name = dtype_names.get(dtype, f"UNKNOWN({dtype})")
        
        print(f"KV[{i}]: pos={pos_before} key='{key}' dtype={dtype}({dtype_name})", end="")
        
        if dtype == 0:
            print(" = NULL")
        elif dtype == 1:  # BOOL
            v = struct.unpack("<?", f.read(1))[0]
            print(f" = {v}")
        elif dtype == 2:  # UINT8
            v = struct.unpack("<B", f.read(1))[0]
            print(f" = {v}")
        elif dtype == 3:  # INT8
            v = struct.unpack("<b", f.read(1))[0]
            print(f" = {v}")
        elif dtype == 4:  # UINT16
            v = struct.unpack("<H", f.read(2))[0]
            print(f" = {v}")
        elif dtype == 5:  # INT16
            v = struct.unpack("<h", f.read(2))[0]
            print(f" = {v}")
        elif dtype == 6:  # UINT32
            v = struct.unpack("<I", f.read(4))[0]
            print(f" = {v}")
        elif dtype == 7:  # INT32
            v = struct.unpack("<i", f.read(4))[0]
            print(f" = {v}")
        elif dtype == 8:  # UINT64
            v = struct.unpack("<Q", f.read(8))[0]
            print(f" = {v}")
        elif dtype == 9:  # INT64
            v = struct.unpack("<q", f.read(8))[0]
            print(f" = {v}")
        elif dtype == 10:  # FLOAT32
            v = struct.unpack("<f", f.read(4))[0]
            print(f" = {v}")
        elif dtype == 11:  # FLOAT64
            v = struct.unpack("<d", f.read(8))[0]
            print(f" = {v}")
        elif dtype == 12:  # STRING
            slen_raw = f.read(8)
            slen = struct.unpack("<Q", slen_raw)[0]
            if slen > 1048576:
                try:
                    f.seek(slen)
                except:
                    _ = f.read(slen)
                print(f" = <large string, slen={slen}>")
            else:
                v = f.read(slen).decode("utf-8", errors="replace") if slen > 0 else ""
                print(f" = '{v}'")
        elif dtype == 13:  # ARRAY
            elem_type = struct.unpack("<I", f.read(4))[0]
            array_len = struct.unpack("<Q", f.read(8))[0]
            elem_dtype_names = {0: "NULL", 1: "BOOL", 2: "UINT8", 3: "INT8",
                4: "UINT16", 5: "INT16", 6: "UINT32", 7: "INT32",
                8: "UINT64", 9: "INT64", 10: "FLOAT32", 11: "FLOAT64",
                12: "STRING"}
            ename = elem_dtype_names.get(elem_type, f"UNKNOWN({elem_type})")
            print(f" = ARRAY[{array_len}] of {ename}")
            
            elem_sizes = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 4, 8: 8, 9: 8, 10: 4, 11: 8}
            if elem_type == 12:  # STRING array
                for j in range(min(array_len, 5)):
                    sl = struct.unpack("<Q", f.read(8))[0]
                    if sl > 1048576:
                        print(f"    [{j}]: <large string slen={sl}>, skipping rest")
                        for _ in range(j+1, array_len):
                            sl2 = struct.unpack("<Q", f.read(8))[0]
                            try:
                                f.seek(sl2)
                            except:
                                _ = f.read(sl2)
                        break
                    else:
                        sv = f.read(sl).decode("utf-8", errors="replace") if sl > 0 else ""
                        print(f"    [{j}]: '{sv}'")
                if array_len > 5:
                    remaining = array_len - 5
                    print(f"    ... and {remaining} more strings")
                    for _ in range(remaining):
                        sl = struct.unpack("<Q", f.read(8))[0]
                        try:
                            f.seek(sl)
                        except:
                            _ = f.read(sl)
            else:
                es = elem_sizes.get(elem_type, 0)
                total = es * array_len
                print(f"    Skipping {total} bytes")
                try:
                    f.seek(total, 1)
                except:
                    _ = f.read(total)
        elif dtype == 14:  # OBJECT
            print(" = <object>")
            break
        else:
            print(f" = <unknown>")
            break
    
    print(f"\nFile position after KV: {f.tell()}")
    
    # Check for more KVs beyond the first 20
    # Search for a pattern that looks like a tensor
    print(f"\nChecking for tensor data...")
    next_nlen_raw = f.read(8)
    next_nlen = struct.unpack("<Q", next_nlen_raw)[0]
    print(f"Next nlen: {next_nlen}")
    if next_nlen <= 65536:
        name = f.read(next_nlen).decode("utf-8", errors="replace")
        print(f"Next tensor name: {name}")
    else:
        print("Misaligned! Let me check bytes...")
        f.seek(pos_before)
        data = f.read(256)
        print(f"Data at pos {pos_before}: {data.hex()}")