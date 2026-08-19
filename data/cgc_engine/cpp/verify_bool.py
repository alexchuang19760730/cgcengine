#!/usr/bin/env python3
"""
Verify correct bool storage format in GGUF.
Try different bool sizes to find the correct one.
"""
import struct
import os

def verify_bool_format(filepath):
    file_size = os.path.getsize(filepath)
    f = open(filepath, "rb")
    
    # Read header
    f.read(24)  # Skip header
    
    # We know item 46 is tokenizer.ggml.add_bos_token (dtype=7)
    # Let's get to that position and analyze different bool sizes
    
    # First parse items 0-45
    for idx in range(46):
        klen_bytes = f.read(8)
        klen = struct.unpack("<Q", klen_bytes)[0]
        key = f.read(klen).decode("utf-8", errors="replace")
        dtype = struct.unpack("<I", f.read(4))[0]
        
        if dtype == 8:
            slen = struct.unpack("<Q", f.read(8))[0]
            f.seek(slen, 1)
        elif dtype == 9:
            elem_type = struct.unpack("<I", f.read(4))[0]
            array_len = struct.unpack("<Q", f.read(8))[0]
            if elem_type == 8:
                for _ in range(array_len):
                    sl = struct.unpack("<Q", f.read(8))[0]
                    f.seek(sl, 1)
            elif elem_type in (4, 5, 6, 10):
                f.seek((8 if elem_type == 10 else 4) * array_len, 1)
            elif elem_type == 7:
                f.seek(4 * array_len, 1)
            else:
                f.seek(4 * array_len, 1)
        elif dtype == 7:
            # Save position before bool value
            bool_start = f.tell()
            
            # Try different interpretations
            for bool_size in [1, 4]:
                f.seek(bool_start)
                bool_val_bytes = f.read(bool_size)
                
                if bool_size == 1:
                    bool_val = bool_val_bytes[0] if len(bool_val_bytes) == 1 else -1
                    next_item_start = bool_start + 1
                else:  # 4
                    bool_val = struct.unpack("<i", bool_val_bytes)[0] if len(bool_val_bytes) == 4 else -1
                    next_item_start = bool_start + 4
                
                # Check if next item makes sense
                f.seek(next_item_start)
                next_klen_bytes = f.read(8)
                if len(next_klen_bytes) >= 8:
                    next_klen = struct.unpack("<Q", next_klen_bytes)[0]
                    valid = 3 <= next_klen <= 10000000
                    
                    if valid:
                        # Try to read potential key
                        f.seek(next_item_start + 8)
                        potential_key = f.read(min(next_klen, 60))
                        key_str = potential_key.decode("utf-8", errors="replace")
                        print(f"  Bool size={bool_size}: val={bool_val}, next klen={next_klen}, next key='{key_str}'")
                    else:
                        print(f"  Bool size={bool_size}: val={bool_val}, next klen={next_klen} (INVALID)")
            
            # Restore position and continue with 4-byte interpretation for now
            f.seek(bool_start + 4)
        else:
            sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 10:8, 11:8, 12:8}
            s = sizes.get(dtype, 4)
            f.seek(s, 1)
    
    pos_46 = f.tell()
    print(f"\nAfter item 46 (with 4-byte bool), position: {pos_46}")
    
    # Now manually analyze what's at pos_46
    f.seek(pos_46)
    raw = f.read(64)
    print(f"Bytes at pos {pos_46}: {raw.hex()}")
    print(f"ASCII: {''.join(chr(b) if 32 <= b < 127 else '.' for b in raw)}")
    
    # Now let's try: what if bool is 1 byte?
    # Item 46 value starts at pos_46 - 44 (44 = 8 klen + 28 key + 4 dtype + 4 val)
    # Actually let me just re-parse with 1-byte bool from the start
    print(f"\n--- Re-parsing with 1-byte bool ---")
    f.seek(24)
    
    for idx in range(n_kv := 54):
        item_start = f.tell()
        klen_bytes = f.read(8)
        if len(klen_bytes) < 8:
            break
        klen = struct.unpack("<Q", klen_bytes)[0]
        if klen == 0 or klen > 1000000:
            print(f"  [{idx}] @{item_start} INVALID klen={klen}")
            break
        key = f.read(klen).decode("utf-8", errors="replace")
        dtype = struct.unpack("<I", f.read(4))[0]
        
        if dtype == 7:
            # 1-byte bool
            b = f.read(1)
            if len(b) == 1:
                print(f"  [{idx}] @{item_start} -> @{f.tell()} bool={b[0]} key='{key}'")
            else:
                break
        elif dtype == 8:
            slen = struct.unpack("<Q", f.read(8))[0]
            f.seek(slen, 1)
        elif dtype == 9:
            elem_type = struct.unpack("<I", f.read(4))[0]
            array_len = struct.unpack("<Q", f.read(8))[0]
            if elem_type == 8:
                for _ in range(array_len):
                    sl = struct.unpack("<Q", f.read(8))[0]
                    f.seek(sl, 1)
            elif elem_type in (4, 5, 6, 10):
                f.seek((8 if elem_type == 10 else 4) * array_len, 1)
            elif elem_type == 7:
                f.seek(4 * array_len, 1)
            else:
                f.seek(4 * array_len, 1)
        else:
            sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 10:8, 11:8, 12:8}
            s = sizes.get(dtype, 4)
            f.seek(s, 1)
    
    pos_after_1byte_bool = f.tell()
    print(f"\nAfter parsing with 1-byte bool, position: {pos_after_1byte_bool}")
    
    # Now search for tensor info from this position
    print(f"\n--- Searching for tensor info from position {pos_after_1byte_bool} ---")
    
    for search_start in range(pos_after_1byte_bool, min(pos_after_1byte_bool + 20000, file_size - 100), 32):
        f.seek(search_start)
        nlen_bytes = f.read(8)
        if len(nlen_bytes) < 8:
            continue
        nlen = struct.unpack("<Q", nlen_bytes)[0]
        if nlen < 3 or nlen > 256:
            continue
        name = f.read(nlen).decode("utf-8", errors="replace")
        if not any(kw in name for kw in ["blk.", "token_embd", "output", "norm.", "rope.", "tokenizer"]):
            continue
        
        nd = struct.unpack("<I", f.read(4))[0]
        if nd == 0 or nd > 5:
            continue
        dims = []
        for _ in range(nd):
            d = f.read(8)
            dims.append(struct.unpack("<Q", d)[0])
        if len(dims) != nd:
            continue
        
        gt = struct.unpack("<I", f.read(4))[0]
        if gt > 50:
            continue
        
        off = struct.unpack("<Q", f.read(8))[0]
        if off > file_size:
            continue
        
        print(f"  FOUND at {search_start} (0x{search_start:X}): '{name}' dims={dims} type={gt} offset=0x{off:X}")
        break
    else:
        print(f"  NOT FOUND within 20KB search from pos {pos_after_1byte_bool}")
    
    f.close()

if __name__ == "__main__":
    path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
    if os.path.exists(path):
        verify_bool_format(path)
    else:
        print(f"File not found: {path}")
