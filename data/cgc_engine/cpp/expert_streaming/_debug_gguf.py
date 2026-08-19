import struct

gguf_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

with open(gguf_path, "rb") as f:
    # Read and verify magic
    magic = struct.unpack("<I", f.read(4))[0]
    print(f"Magic: 0x{magic:08X}")
    
    version = struct.unpack("<I", f.read(4))[0]
    print(f"Version: {version}")
    
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    print(f"Tensor count: {n_tensors}")
    print(f"KV count: {n_kv}")
    
    print(f"\n=== KV Metadata ===")
    
    def read_u8(f):
        return struct.unpack("<B", f.read(1))[0]
    
    def read_u32(f):
        return struct.unpack("<I", f.read(4))[0]
    
    def read_u64(f):
        return struct.unpack("<Q", f.read(8))[0]
    
    def read_f32(f):
        return struct.unpack("<f", f.read(4))[0]
    
    def read_f64(f):
        return struct.unpack("<d", f.read(8))[0]
    
    def read_bool(f):
        b = f.read(1)
        f.read(3)  # padding
        return struct.unpack("<B", b)[0] != 0
    
    def read_string(f):
        n = read_u64(f)
        s = f.read(n).decode("utf-8", errors="replace")
        return s
    
    type_names = {
        0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16",
        4: "UINT32", 5: "INT32", 6: "FLOAT32", 7: "BOOL",
        8: "STRING", 9: "ARRAY", 10: "UINT64", 11: "INT64", 12: "FLOAT64"
    }
    
    for kv_idx in range(min(n_kv, 30)):
        key = read_string(f)
        dtype = read_u32(f)
        type_name = type_names.get(dtype, f"UNKNOWN({dtype})")
        
        pos_before = f.tell() - 4 - len(key) - 8  # approximate
        
        val_str = ""
        try:
            if dtype in (0, 1):  # UINT8, INT8
                val = read_u8(f) if dtype == 0 else struct.unpack("<b", f.read(1))[0]
                val_str = str(val)
            elif dtype in (2, 3):  # UINT16, INT16
                val = struct.unpack("<H", f.read(2))[0] if dtype == 2 else struct.unpack("<h", f.read(2))[0]
                val_str = str(val)
            elif dtype in (4, 5):  # UINT32, INT32
                val = read_u32(f) if dtype == 4 else struct.unpack("<i", f.read(4))[0]
                val_str = str(val)
            elif dtype == 6:  # FLOAT32
                val = read_f32(f)
                val_str = f"{val:.6f}"
            elif dtype == 7:  # BOOL
                val = read_bool(f)
                val_str = str(val)
            elif dtype == 8:  # STRING
                val = read_string(f)
                val_str = f'"{val[:50]}' + ('...' if len(val) > 50 else '') + '"'
            elif dtype == 9:  # ARRAY
                n = read_u64(f)
                elem_type = read_u32(f)
                elem_type_name = type_names.get(elem_type, f"UNKNOWN({elem_type})")
                val_str = f"ARRAY[{n}] of {elem_type_name}"
                
                if n > 10000 and elem_type == 8:  # Large string array
                    # Just skip
                    for _ in range(n):
                        slen = read_u64(f)
                        f.seek(slen, 1)  # SEEK_CUR
                    val_str += f" (skipped, {n} strings)"
                elif elem_type == 8:  # Small string array
                    strings = []
                    for _ in range(n):
                        s = read_string(f)
                        strings.append(s[:30])
                    val_str += f" = {strings[:3]}..." if n > 3 else f" = {strings}"
                elif elem_type in (4, 5, 7):  # UINT32, INT32, BOOL
                    f.seek(n * 4, 1)
                    val_str += f" (skipped {n} elements)"
                elif elem_type in (10, 11, 12):  # UINT64, INT64, FLOAT64
                    f.seek(n * 8, 1)
                    val_str += f" (skipped {n} elements)"
                elif elem_type == 6:  # FLOAT32
                    f.seek(n * 4, 1)
                    val_str += f" (skipped {n} elements)"
                else:
                    f.seek(n * 4, 1)
                    val_str += f" (skipped {n} elements)"
            elif dtype == 10:  # UINT64
                val = read_u64(f)
                val_str = str(val)
            elif dtype == 11:  # INT64
                val = struct.unpack("<q", f.read(8))[0]
                val_str = str(val)
            elif dtype == 12:  # FLOAT64
                val = read_f64(f)
                val_str = f"{val:.6f}"
            else:
                val_str = f"UNKNOWN TYPE"
                break
            
            print(f"  [{kv_idx:3d}] {key}: {type_name} = {val_str}")
        except Exception as e:
            print(f"  [{kv_idx:3d}] {key}: {type_name} ERROR at pos {f.tell()}: {e}")
            break