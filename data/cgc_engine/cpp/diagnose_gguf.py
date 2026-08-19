#!/usr/bin/env python3
"""
Diagnose GGUF parsing: trace KV parsing with exact position tracking
to find where the offset goes wrong.
"""
import struct
import os
import sys

def big_skip(f, n):
    CHUNK = 4 * 1024 * 1024
    while n > 0:
        d = f.read(min(CHUNK, n))
        if not d:
            break
        n -= len(d)

def trace_kv(filepath, max_kv=None):
    file_size = os.path.getsize(filepath)
    f = open(filepath, "rb")
    
    # Read header
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    
    print(f"File: {os.path.basename(filepath)}")
    print(f"Size: {file_size:,} bytes")
    print(f"Version: {version}, n_tensors: {n_tensors}, n_kv: {n_kv}")
    
    if max_kv is None:
        max_kv = n_kv
    
    print(f"\n--- Tracing {max_kv}/{n_kv} KV entries ---")
    
    prev_end = 24  # After header
    
    for idx in range(max_kv):
        start = f.tell()
        
        # Read klen
        klen_bytes = f.read(8)
        if len(klen_bytes) < 8:
            print(f"  [{idx:3d}] @{start:12d} READ ERROR: EOF at klen")
            break
        klen = struct.unpack("<Q", klen_bytes)[0]
        
        if klen == 0 or klen > 1000000:
            print(f"  [{idx:3d}] @{start:12d} INVALID klen={klen}")
            print(f"    Prev item ended at: {prev_end}")
            print(f"    Raw bytes at pos: {klen_bytes.hex()}")
            # Try to find the right pos
            f.seek(max(0, start - 100))
            raw = f.read(200)
            print(f"    Context (-100..+100): {raw[:200]}")
            break
        
        # Read key
        key_bytes = f.read(klen)
        if len(key_bytes) < klen:
            print(f"  [{idx:3d}] @{start:12d} READ ERROR: EOF at key (klen={klen})")
            break
        key = key_bytes.decode("utf-8", errors="replace")
        
        # Read dtype
        dtype_bytes = f.read(4)
        if len(dtype_bytes) < 4:
            print(f"  [{idx:3d}] @{start:12d} READ ERROR: EOF at dtype")
            break
        dtype = struct.unpack("<I", dtype_bytes)[0]
        
        # Save pos before reading value
        before_val = f.tell()
        
        # Read value based on dtype
        val_desc = ""
        if dtype == 7:  # bool - stored as int8 (1 byte) + 3 byte padding
            v = f.read(1)
            if len(v) == 1:
                pad = f.read(3)
                val_desc = f"bool={v[0]}"
            else:
                val_desc = "ERR"
        elif dtype == 8:  # string
            slen_bytes = f.read(8)
            if len(slen_bytes) >= 8:
                slen = struct.unpack("<Q", slen_bytes)[0]
                val_desc = f"str[{slen}]"
                big_skip(f, slen)
            else:
                val_desc = "ERR"
        elif dtype == 9:  # array
            et_bytes = f.read(4)
            al_bytes = f.read(8)
            if len(et_bytes) >= 4 and len(al_bytes) >= 8:
                elem_type = struct.unpack("<I", et_bytes)[0]
                array_len = struct.unpack("<Q", al_bytes)[0]
                val_desc = f"array[{array_len}] elem_type={elem_type}"
                
                if elem_type == 8:  # string array
                    total_strings_size = 0
                    for _ in range(array_len):
                        sl_bytes = f.read(8)
                        if len(sl_bytes) < 8:
                            break
                        sl = struct.unpack("<Q", sl_bytes)[0]
                        total_strings_size += sl
                        big_skip(f, sl)
                    val_desc += f" total_str_size={total_strings_size:,}"
                elif elem_type in (4, 5, 6):
                    big_skip(f, 4 * array_len)
                elif elem_type == 7:
                    big_skip(f, array_len)
                elif elem_type == 10 or elem_type == 11 or elem_type == 12:
                    big_skip(f, 8 * array_len)
                else:
                    big_skip(f, 4 * array_len)
            else:
                val_desc = "ERR"
        elif dtype in (4, 5, 6, 10):  # uint32/int32/float32/uint64
            v = f.read(4 if dtype != 10 else 8)
            val_desc = f"val_bytes={len(v)}"
        elif dtype in (11, 12):  # float64/int64
            v = f.read(8)
            val_desc = f"val_bytes={len(v)}"
        else:
            v = f.read(4)
            val_desc = f"unknown dtype={dtype}"
        
        end = f.tell()
        
        # Check alignment: after each item, next klen should be valid
        next_check = ""
        if idx < n_kv - 1:
            saved_pos = end
            f.seek(end)
            next_bytes = f.read(8)
            if len(next_bytes) >= 8:
                next_klen = struct.unpack("<Q", next_bytes)[0]
                if next_klen < 3 or next_klen > 10000000:
                    next_check = f" [NEXT klen={next_klen} SUSPICIOUS!]"
            f.seek(saved_pos)
        
        diff = end - prev_end
        if diff != 8 + klen + 4:  # Expected minimum: klen(8) + key(klen) + dtype(4)
            # This is normal for non-trivial values
            pass
        
        # Print item
        print(f"  [{idx:3d}] @{start:12d} key_len={klen:6d} dtype={dtype:2d} key='{key[:60]}' -> @{end:12d} ({end-start:8d} bytes) val={val_desc}{next_check}")
        
        prev_end = end
    
    pos_after_kv = f.tell()
    print(f"\n--- After KV parsing ---")
    print(f"  Position: {pos_after_kv:,} (0x{pos_after_kv:X})")
    print(f"  32-byte aligned: {(pos_after_kv + 31) & ~31:,} (0x{(pos_after_kv + 31) & ~31:X})")
    print(f"  Remaining bytes: {file_size - pos_after_kv:,}")
    
    # Now try to find tensor info
    # First try: right after KV
    print(f"\n--- Trying to find tensor info ---")
    
    # Try positions: right after KV, then 32-byte aligned, then with offsets
    candidates = [pos_after_kv, (pos_after_kv + 31) & ~31]
    for delta in [-256, -128, -64, -32, -16, -8, 8, 16, 32, 64, 128, 256]:
        candidates.append((pos_after_kv + delta) & ~31)
        candidates.append(pos_after_kv + delta)
    
    candidates = list(set(c for c in candidates if 0 < c < file_size - 100))
    
    best = None
    best_count = 0
    
    for pos in sorted(candidates):
        f.seek(pos)
        
        # Try to read first tensor
        nlen_bytes = f.read(8)
        if len(nlen_bytes) < 8:
            continue
        nlen = struct.unpack("<Q", nlen_bytes)[0]
        if nlen == 0 or nlen > 256:
            continue
        
        name_bytes = f.read(nlen)
        if len(name_bytes) < nlen:
            continue
        name = name_bytes.decode("utf-8", errors="replace")
        
        # Validate name
        has_valid_prefix = any(kw in name for kw in ["blk.", "token_embd", "output", "norm", "rope", "tokenizer"])
        if not has_valid_prefix:
            continue
        
        # Try to read more
        # n_dims
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
        
        # ggml_type
        gt_bytes = f.read(4)
        if len(gt_bytes) < 4:
            continue
        ggml_type = struct.unpack("<I", gt_bytes)[0]
        if ggml_type > 50:
            continue
        
        # offset
        off_bytes = f.read(8)
        if len(off_bytes) < 8:
            continue
        offset = struct.unpack("<Q", off_bytes)[0]
        if offset > file_size:
            continue
        
        print(f"  FOUND at pos {pos:,} (0x{pos:X})!")
        print(f"    First tensor: '{name}' dims={dims} type={ggml_type} offset=0x{offset:X}")
        
        # Try to parse more
        count = 1
        for _ in range(min(n_tensors, 10)):
            nlen_bytes2 = f.read(8)
            if len(nlen_bytes2) < 8:
                break
            nlen2 = struct.unpack("<Q", nlen_bytes2)[0]
            if nlen2 == 0 or nlen2 > 256:
                break
            name2 = f.read(nlen2)
            if len(name2) < nlen2:
                break
            count += 1
        
        print(f"    Can parse {count} tensors from this position")
        best = pos
        best_count = count
        
        if count >= 10:
            break
    
    if best:
        print(f"\n  Best position: {best:,} (0x{best:X}) with {best_count} tensors")
    else:
        print(f"\n  FAILED to find tensor info!")
        
        # Dump raw bytes around the expected position
        print(f"\n  Dumping raw bytes around aligned position...")
        aligned = (pos_after_kv + 31) & ~31
        f.seek(max(0, aligned - 64))
        raw = f.read(128)
        for i in range(0, len(raw), 16):
            hex_str = ' '.join(f'{b:02x}' for b in raw[i:i+16])
            ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in raw[i:i+16])
            print(f"    {aligned - 64 + i:10d}: {hex_str:48s} {ascii_str}")
    
    f.close()
    return best, best_count


if __name__ == "__main__":
    models = [
        (r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf", 60),
        (r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf", 60),
        (r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf", 40),
    ]
    
    for path, max_kv in models:
        if os.path.exists(path):
            print(f"\n{'='*80}")
            trace_kv(path, max_kv=max_kv)
        else:
            print(f"\nSKIP: {path} not found")
