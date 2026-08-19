#!/usr/bin/env python3
"""
Precisely calculate GGUF array sizes and verify file pointer positioning.
Focus on finding exact tensor info start position.
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

def parse_kv_with_precise_sizes(filepath):
    file_size = os.path.getsize(filepath)
    f = open(filepath, "rb")
    
    # Read header
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    
    print(f"File: {os.path.basename(filepath)}")
    print(f"Size: {file_size:,} bytes, Version: {version}, n_kv: {n_kv}, n_tensors: {n_tensors}")
    
    pos = f.tell()
    print(f"Header ends at: {pos}")
    
    # Parse all KV items with precise size tracking
    for idx in range(n_kv):
        item_start = f.tell()
        
        # Read klen
        klen_bytes = f.read(8)
        if len(klen_bytes) < 8:
            print(f"  [{idx}] @{item_start} EOF at klen")
            break
        klen = struct.unpack("<Q", klen_bytes)[0]
        
        if klen == 0 or klen > 1000000:
            print(f"  [{idx}] @{item_start} INVALID klen={klen}, raw={klen_bytes.hex()}")
            # Try to recover by searching for next valid klen
            recovered = False
            for search_delta in range(0, 1024, 4):
                f.seek(item_start + search_delta)
                test_bytes = f.read(8)
                if len(test_bytes) >= 8:
                    test_klen = struct.unpack("<Q", test_bytes)[0]
                    if 3 <= test_klen <= 256:  # Valid key length
                        print(f"    Recovered at delta +{search_delta}, klen={test_klen}")
                        recovered = True
                        break
            if not recovered:
                print(f"    Could not recover, stopping")
                break
            item_start = f.tell()
            klen = test_klen
        
        # Read key
        key_bytes = f.read(klen)
        if len(key_bytes) < klen:
            print(f"  [{idx}] @{item_start} EOF at key")
            break
        key = key_bytes.decode("utf-8", errors="replace")
        
        # Read dtype
        dtype_bytes = f.read(4)
        if len(dtype_bytes) < 4:
            print(f"  [{idx}] @{item_start} EOF at dtype")
            break
        dtype = struct.unpack("<I", dtype_bytes)[0]
        
        val_start = f.tell()
        
        # Calculate value size based on dtype
        if dtype == 7:  # bool: 4 bytes total (1 byte value + 3 bytes padding)
            val_size = 4
            f.seek(val_size, 1)
            
        elif dtype == 8:  # string: 8 bytes length + string bytes
            slen_bytes = f.read(8)
            if len(slen_bytes) < 8:
                break
            slen = struct.unpack("<Q", slen_bytes)[0]
            val_size = 8 + slen
            f.seek(slen, 1)
            
        elif dtype == 9:  # array
            et_bytes = f.read(4)
            al_bytes = f.read(8)
            if len(et_bytes) < 4 or len(al_bytes) < 8:
                break
            elem_type = struct.unpack("<I", et_bytes)[0]
            array_len = struct.unpack("<Q", al_bytes)[0]
            
            if elem_type == 8:  # string array
                # Each string: 8 bytes length + string content
                total_str_bytes = 0
                for i in range(array_len):
                    sl_bytes = f.read(8)
                    if len(sl_bytes) < 8:
                        print(f"    ERROR reading string {i} length at pos {f.tell()}")
                        break
                    sl = struct.unpack("<Q", sl_bytes)[0]
                    total_str_bytes += sl
                    f.seek(sl, 1)
                val_size = 4 + 8 + array_len * 8 + total_str_bytes
                
            else:
                # Numeric/bool array
                elem_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 4, 10: 8, 11: 8, 12: 8}
                elem_size = elem_sizes.get(elem_type, 4)
                total_numeric = elem_size * array_len
                f.seek(total_numeric, 1)
                val_size = 4 + 8 + total_numeric
        
        elif dtype == 10:  # uint64
            val_size = 8
            f.seek(val_size, 1)
        elif dtype in (11, 12):  # float64, int64
            val_size = 8
            f.seek(val_size, 1)
        else:  # uint8, int8, uint16, int16, uint32, int32, float32
            val_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4}
            val_size = val_sizes.get(dtype, 4)
            f.seek(val_size, 1)
        
        item_end = f.tell()
        item_total = item_end - item_start
        
        # Print large items
        if item_total > 1000 or "tokenizer" in key or "add_bos" in key or "chat" in key:
            print(f"  [{idx:3d}] @{item_start:12d} -> @{item_end:12d} ({item_total:8d} bytes) dtype={dtype:2d} key='{key[:70]}'")
    
    pos_after_kv = f.tell()
    print(f"\n--- After KV parsing ---")
    print(f"  Position: {pos_after_kv} (0x{pos_after_kv:X})")
    print(f"  32-byte aligned: {(pos_after_kv + 31) & ~31} (0x{(pos_after_kv + 31) & ~31:X})")
    print(f"  Remaining bytes: {file_size - pos_after_kv:,}")
    
    # Now search for tensor info
    print(f"\n--- Searching for tensor info section ---")
    
    # The tensor info section should start at a 32-byte aligned position
    # Let's search more broadly
    best_pos = None
    best_count = 0
    best_name = None
    
    for aligned_pos in range((pos_after_kv + 31) & ~31, min(pos_after_kv + 10000, file_size - 100), 32):
        f.seek(aligned_pos)
        
        # Try to read first tensor entry
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
        
        # Must start with known prefixes
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
            d_bytes = f.read(8)
            if len(d_bytes) < 8:
                break
            dims.append(struct.unpack("<Q", d_bytes)[0])
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
        
        # Valid first tensor found, try to count more
        count = 1
        for _ in range(n_tensors - 1):
            nl_bytes = f.read(8)
            if len(nl_bytes) < 8:
                break
            nl = struct.unpack("<Q", nl_bytes)[0]
            if nl < 3 or nl > 256:
                break
            nm = f.read(nl)
            if len(nm) < nl:
                break
            
            # Skip dimensions
            nd2 = struct.unpack("<I", f.read(4))[0]
            for _ in range(nd2):
                f.read(8)
            f.read(4)  # ggml_type
            f.read(8)  # offset
            count += 1
        
        print(f"  FOUND at {aligned_pos} (0x{aligned_pos:X}): '{name}' dims={dims} type={ggml_type} offset=0x{offset:X}, count={count}/{n_tensors}")
        
        if count > best_count:
            best_count = count
            best_pos = aligned_pos
            best_name = name
        
        if count >= n_tensors * 0.8:  # Found most tensors, good enough
            break
    
    if best_pos:
        print(f"\n  Best: {best_pos} (0x{best_pos:X}), {best_count} tensors, first='{best_name}'")
        
        # Now calculate the actual data start position
        # After last tensor info, data starts at 32-byte boundary
        f.seek(best_pos)
        for _ in range(best_count):
            nl = struct.unpack("<Q", f.read(8))[0]
            f.seek(nl, 1)  # name
            nd = struct.unpack("<I", f.read(4))[0]
            f.seek(8 * nd, 1)  # dimensions
            f.seek(4 + 8, 1)  # ggml_type + offset
        
        info_end = f.tell()
        data_start = (info_end + 31) & ~31
        print(f"  Tensor info ends at: {info_end} (0x{info_end:X})")
        print(f"  Data starts at: {data_start} (0x{data_start:X})")
    else:
        print(f"\n  FAILED to find tensor info within 10KB search range")
        print(f"  Dumping raw bytes around KV end...")
        f.seek(pos_after_kv - 16)
        for i in range(256):
            b = f.read(1)
            if b:
                pass
        f.seek(pos_after_kv - 16)
        raw = f.read(256)
        for i in range(0, len(raw), 16):
            hex_str = ' '.join(f'{b:02x}' for b in raw[i:i+16])
            ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in raw[i:i+16])
            print(f"    {pos_after_kv - 16 + i:10d}: {hex_str:48s} {ascii_str}")
    
    f.close()

if __name__ == "__main__":
    path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
    if os.path.exists(path):
        parse_kv_with_precise_sizes(path)
    else:
        print(f"File not found: {path}")
