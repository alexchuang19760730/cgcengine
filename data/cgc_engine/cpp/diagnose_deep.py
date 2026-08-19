#!/usr/bin/env python3
"""
Deep analysis of bool storage and string array size calculation.
The issue: after parsing large string arrays, the file pointer is wrong.
"""
import struct
import os

def analyze_bool_and_array(filepath):
    file_size = os.path.getsize(filepath)
    f = open(filepath, "rb")
    
    # Read header
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    
    print(f"File: {os.path.basename(filepath)}")
    print(f"Size: {file_size:,} bytes, Version: {version}, n_kv: {n_kv}")
    
    # Focus on the problematic area: tokens/merges arrays and bool
    # We'll manually parse from item 40 onwards
    
    # First, skip to item 40 (tokenizer.ggml.tokens)
    # Let me read items one by one but be more careful about sizing
    
    pos = 24  # After header
    f.seek(pos)
    
    # Parse items 0-39 to get to the problematic area
    for idx in range(40):
        klen_bytes = f.read(8)
        klen = struct.unpack("<Q", klen_bytes)[0]
        key = f.read(klen).decode("utf-8", errors="replace")
        dtype = struct.unpack("<I", f.read(4))[0]
        
        if dtype == 8:  # string
            slen = struct.unpack("<Q", f.read(8))[0]
            f.seek(slen, 1)
        elif dtype == 9:  # array
            elem_type = struct.unpack("<I", f.read(4))[0]
            array_len = struct.unpack("<Q", f.read(8))[0]
            if elem_type == 8:  # string array
                total = 0
                for _ in range(array_len):
                    sl = struct.unpack("<Q", f.read(8))[0]
                    total += 8 + sl
                    f.seek(sl, 1)
            elif elem_type in (4, 5, 6, 10):
                f.seek(4 * array_len if elem_type != 10 else 8 * array_len, 1)
            elif elem_type == 7:
                f.seek(array_len, 1)
            else:
                f.seek(4 * array_len, 1)
        elif dtype == 7:  # bool
            f.seek(4, 1)  # 4-byte bool
        else:
            sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 10:8, 11:8, 12:8}
            s = sizes.get(dtype, 4)
            f.seek(s, 1)
    
    print(f"\nAfter 40 items, position: {f.tell()} (0x{f.tell():X})")
    
    # Now analyze items 40-54 more carefully
    for idx in range(40, min(n_kv, 55)):
        start = f.tell()
        
        klen_bytes = f.read(8)
        if len(klen_bytes) < 8:
            print(f"  [{idx}] @{start} READ ERROR at klen")
            break
        klen = struct.unpack("<Q", klen_bytes)[0]
        
        if klen == 0 or klen > 1000000:
            print(f"  [{idx}] @{start} INVALID klen={klen}, raw={klen_bytes.hex()}")
            # Dump context
            f.seek(max(0, start - 32))
            ctx = f.read(64)
            print(f"    Context: {ctx.hex()}")
            break
        
        key = f.read(klen).decode("utf-8", errors="replace")
        dtype_bytes = f.read(4)
        dtype = struct.unpack("<I", dtype_bytes)[0]
        
        before_val = f.tell()
        
        print(f"  [{idx}] @{start:12d} key_len={klen:6d} dtype={dtype:2d} key='{key}'")
        print(f"    Value starts at: {before_val}")
        
        if dtype == 7:  # bool
            # Read raw bytes
            raw4 = f.read(4)
            print(f"    Raw 4 bytes: {raw4.hex()}")
            v1 = raw4[0] if len(raw4) >= 1 else -1
            v2 = raw4[1] if len(raw4) >= 2 else -1
            v3 = raw4[2] if len(raw4) >= 3 else -1
            v4 = raw4[3] if len(raw4) >= 4 else -1
            print(f"    Bytes breakdown: byte0={v1}, byte1={v2}, byte2={v3}, byte3={v4}")
            
            # Check: is this 4-byte aligned?
            # The question: is bool stored as 4 bytes total, or 1+3 padding?
            # Let's check if byte1-3 are padding zeros
            is_padded = (v2 == 0 and v3 == 0)
            print(f"    Byte1-3 are zeros (padding): {is_padded}")
            
            if not is_padded:
                # Maybe it's stored differently - check if it's part of the next item
                # Let's try alternative interpretation
                print(f"    Byte1={v1} may be the actual bool value, next bytes may be start of next item")
                
                # Try: bool is just 1 byte, then the next item starts
                f.seek(before_val + 1)
                next_klen_bytes = f.read(8)
                next_klen = struct.unpack("<Q", next_klen_bytes)[0]
                print(f"    If 1-byte bool: next klen would be {next_klen} (0x{next_klen:X})")
                
                # Check if this looks valid
                if 3 <= next_klen <= 10000000:
                    # Read the potential key
                    potential_key = f.read(min(next_klen, 60))
                    print(f"    Potential next key: '{potential_key.decode('utf-8', errors='replace')}'")
                else:
                    # Try: maybe it's 2-byte or 8-byte bool?
                    f.seek(before_val + 2)
                    nk2 = struct.unpack("<Q", f.read(8))[0]
                    print(f"    If 2-byte bool: next klen = {nk2}")
                    
                    f.seek(before_val + 8)
                    nk8 = struct.unpack("<Q", f.read(8))[0]
                    print(f"    If 8-byte bool: next klen = {nk8}")
            
            # Stay at +4 for now
            f.seek(before_val + 4)
            
        elif dtype == 8:  # string
            slen_bytes = f.read(8)
            slen = struct.unpack("<Q", slen_bytes)[0]
            f.seek(slen, 1)
            print(f"    string[{slen}]")
            
        elif dtype == 9:  # array
            et_bytes = f.read(4)
            al_bytes = f.read(8)
            elem_type = struct.unpack("<I", et_bytes)[0]
            array_len = struct.unpack("<Q", al_bytes)[0]
            
            print(f"    array[{array_len}] elem_type={elem_type}")
            
            if elem_type == 8:  # string array
                total_string_bytes = 0
                for i in range(array_len):
                    sl_bytes = f.read(8)
                    if len(sl_bytes) < 8:
                        print(f"    ERROR reading string {i} length")
                        break
                    sl = struct.unpack("<Q", sl_bytes)[0]
                    total_string_bytes += sl
                    f.seek(sl, 1)
                
                array_size = array_len * 8 + total_string_bytes
                print(f"    String array: {array_len} strings, total string bytes: {total_string_bytes:,}, total array: {array_size:,} bytes")
                
            elif elem_type in (4, 5, 6, 10):  # uint32/int32/float32/uint64
                elem_size = 8 if elem_type == 10 else 4
                total = elem_size * array_len
                f.seek(total, 1)
                print(f"    Numeric array: {total} bytes")
                
            elif elem_type == 7:  # bool array
                # Each bool: 1 byte value + 3 padding? Or just 1 byte?
                # Let's read a few to check
                total = 0
                for i in range(min(array_len, 5)):
                    b = f.read(1)
                    if len(b) < 1:
                        break
                    # Read 3 more padding
                    p = f.read(3)
                    total += 4
                # Skip remaining
                f.seek((array_len - 5) * 4, 1)
                print(f"    Bool array: {array_len} items, estimated {array_len * 4} bytes")
                
            else:
                f.seek(4 * array_len, 1)
    
    pos_after_kv = f.tell()
    print(f"\n--- After KV parsing ---")
    print(f"  Position: {pos_after_kv} (0x{pos_after_kv:X})")
    print(f"  32-byte aligned: {(pos_after_kv + 31) & ~31} (0x{(pos_after_kv + 31) & ~31:X})")
    
    # Now search for tensor info starting from pos_after_kv
    print(f"\n--- Searching for tensor info ---")
    found = False
    for delta in range(0, 512, 32):
        pos = pos_after_kv + delta
        if pos >= file_size - 100:
            break
        
        f.seek(pos)
        
        nlen_bytes = f.read(8)
        if len(nlen_bytes) < 8:
            continue
        nlen = struct.unpack("<Q", nlen_bytes)[0]
        if nlen < 3 or nlen > 256:
            continue
        
        name_bytes = f.read(nlen)
        if len(name_bytes) < nlen:
            continue
        name = name_bytes.decode("utf-8", errors="replace")
        
        valid = any(kw in name for kw in ["blk.", "token_embd", "output", "norm.", "rope.", "tokenizer"])
        if not valid:
            continue
        
        # Try to parse full tensor info
        nd_bytes = f.read(4)
        if len(nd_bytes) < 4:
            continue
        n_dims = struct.unpack("<I", nd_bytes)[0]
        if n_dims == 0 or n_dims > 5:
            continue
        
        dims = []
        for _ in range(n_dims):
            d = f.read(8)
            if len(d) < 8:
                break
            dims.append(struct.unpack("<Q", d)[0])
        if len(dims) != n_dims:
            continue
        
        gt_bytes = f.read(4)
        if len(gt_bytes) < 4:
            continue
        ggml_type = struct.unpack("<I", gt_bytes)[0]
        if ggml_type > 50:
            continue
        
        off_bytes = f.read(8)
        if len(off_bytes) < 8:
            continue
        offset = struct.unpack("<Q", off_bytes)[0]
        if offset > file_size:
            continue
        
        print(f"  FOUND at pos {pos} (delta +{delta}):")
        print(f"    First tensor: '{name}' dims={dims} type={ggml_type} offset=0x{offset:X}")
        
        # Count more
        count = 1
        for _ in range(n_tensors):
            nl = f.read(8)
            if len(nl) < 8:
                break
            nlen2 = struct.unpack("<Q", nl)[0]
            if nlen2 < 3 or nlen2 > 256:
                break
            nm = f.read(nlen2)
            if len(nm) < nlen2:
                break
            count += 1
        
        print(f"    Can parse {count}/{n_tensors} tensors")
        found = True
        break
    
    if not found:
        print("  FAILED within first 512 bytes")
        
        # Try searching backwards a bit
        print(f"  Trying backwards...")
        for delta in range(-256, 0, 32):
            pos = pos_after_kv + delta
            if pos < 0:
                continue
            
            f.seek(pos)
            
            nlen_bytes = f.read(8)
            if len(nlen_bytes) < 8:
                continue
            nlen = struct.unpack("<Q", nlen_bytes)[0]
            if nlen < 3 or nlen > 256:
                continue
            
            name_bytes = f.read(nlen)
            if len(name_bytes) < nlen:
                continue
            name = name_bytes.decode("utf-8", errors="replace")
            
            valid = any(kw in name for kw in ["blk.", "token_embd", "output", "norm.", "rope.", "tokenizer"])
            if not valid:
                continue
            
            nd_bytes = f.read(4)
            n_dims = struct.unpack("<I", nd_bytes)[0]
            if n_dims == 0 or n_dims > 5:
                continue
            
            dims = []
            for _ in range(n_dims):
                d = f.read(8)
                dims.append(struct.unpack("<Q", d)[0])
            if len(dims) != n_dims:
                continue
            
            gt_bytes = f.read(4)
            ggml_type = struct.unpack("<I", gt_bytes)[0]
            if ggml_type > 50:
                continue
            
            off_bytes = f.read(8)
            offset = struct.unpack("<Q", off_bytes)[0]
            if offset > file_size:
                continue
            
            print(f"  FOUND at pos {pos} (delta {delta}):")
            print(f"    First tensor: '{name}' dims={dims} type={ggml_type} offset=0x{offset:X}")
            found = True
            break
    
    if not found:
        print("  STILL FAILED - dumping raw bytes around KV end...")
        f.seek(pos_after_kv - 32)
        raw = f.read(128)
        for i in range(0, len(raw), 16):
            hex_str = ' '.join(f'{b:02x}' for b in raw[i:i+16])
            ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in raw[i:i+16])
            print(f"    {pos_after_kv - 32 + i:10d}: {hex_str:48s} {ascii_str}")
    
    f.close()

if __name__ == "__main__":
    path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
    if os.path.exists(path):
        analyze_bool_and_array(path)
    else:
        print(f"File not found: {path}")
