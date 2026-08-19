#!/usr/bin/env python3
"""
Simplest possible GGUF parser. Read header directly, then parse KV item by item
with full position tracking.
"""
import struct
import os
import sys

def parse_file(filepath):
    fsize = os.path.getsize(filepath)
    f = open(filepath, "rb")
    
    # Read first 4 bytes = magic
    magic_bytes = f.read(4)
    magic = struct.unpack("<I", magic_bytes)[0]
    
    if magic == 0x46554747:
        print(f"Magic: GGUF (0x{magic:08X}) ✓")
    else:
        print(f"Magic: 0x{magic:08X} ✗ (expected 0x46554747)")
        print(f"  First 32 bytes: {' '.join(f'{b:02x}' for b in magic_bytes + f.read(28))}")
        f.close()
        return
    
    # Read header fields
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    print(f"Version: {version}")
    print(f"n_tensors: {n_tensors}")
    print(f"n_kv: {n_kv}")
    print(f"File size: {fsize}")
    print(f"Header ends at: {f.tell()} (0x{f.tell():X})")
    
    # Verify file is big enough
    header_size = f.tell()
    if fsize <= header_size:
        print(f"ERROR: File too small for header!")
        f.close()
        return
    
    # Parse each KV entry
    print(f"\n--- KV Entries ---")
    parsed_kv = {}
    errors = []
    
    for idx in range(n_kv):
        entry_start = f.tell()
        
        # Read klen
        klen_bytes = f.read(8)
        if len(klen_bytes) < 8:
            errors.append(f"KV[{idx}]: EOF at klen ({fsize - f.tell()} bytes remaining)")
            break
        klen = struct.unpack("<Q", klen_bytes)[0]
        
        # Read key
        if klen <= 65536:
            key_bytes = f.read(klen)
            if len(key_bytes) < klen:
                errors.append(f"KV[{idx}]: EOF at key (klen={klen})")
                break
            key = key_bytes.decode("utf-8", errors="replace")
        else:
            # Huge key - skip it
            _skip(f, klen)
            dtype_bytes = f.read(4)
            if len(dtype_bytes) < 4:
                break
            dtype = struct.unpack("<I", dtype_bytes)[0]
            _skip_value(f, dtype)
            entry_end = f.tell()
            print(f"  [{idx:3d}] @{entry_start:10d} BIG klen={klen} dtype={dtype} -> @{entry_end:10d} ({entry_end-entry_start} bytes)")
            continue
        
        # Read dtype
        dtype_bytes = f.read(4)
        if len(dtype_bytes) < 4:
            errors.append(f"KV[{idx}] '{key}': EOF at dtype")
            break
        dtype = struct.unpack("<I", dtype_bytes)[0]
        
        # Read value
        val = _read_value(f, dtype, key, idx)
        
        entry_end = f.tell()
        parsed_kv[key] = val
        
        vs = str(val)
        if len(vs) > 60:
            vs = vs[:60] + "..."
        print(f"  [{idx:3d}] @{entry_start:10d} '{key}' dtype={dtype} val={vs} -> @{entry_end:10d} ({entry_end-entry_start} bytes)")
        
        # Verify alignment
        if idx < n_kv - 1:
            next_bytes = f.read(min(8, fsize - f.tell()))
            if len(next_bytes) >= 8:
                next_klen = struct.unpack("<Q", next_bytes[:8])[0]
                if next_klen < 3 or next_klen > 65536:
                    # Could be a problem - but let's just note it
                    pass
                f.seek(entry_end)
            else:
                f.seek(entry_end)
    
    pos_after_kv = f.tell()
    aligned = (pos_after_kv + 31) & ~31
    
    print(f"\n--- Summary ---")
    print(f"  Parsed KV: {len(parsed_kv)}/{n_kv}")
    print(f"  Pos after KV: {pos_after_kv} (0x{pos_after_kv:X})")
    print(f"  Aligned: {aligned} (0x{aligned:X})")
    print(f"  Remaining bytes: {fsize - pos_after_kv}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors:
            print(f"    {e}")
    
    # Try to parse tensors
    print(f"\n--- Tensor Parsing ---")
    f.seek(aligned)
    tensors = []
    
    for idx in range(n_tensors):
        t_start = f.tell()
        
        nlen_bytes = f.read(8)
        if len(nlen_bytes) < 8:
            print(f"  Tensor[{idx}]: EOF at nlen ({fsize - t_start} remaining)")
            break
        nlen = struct.unpack("<Q", nlen_bytes)[0]
        if nlen == 0 or nlen > 256:
            print(f"  Tensor[{idx}]: Invalid nlen={nlen} at pos {t_start}")
            break
        
        name_bytes = f.read(nlen)
        if len(name_bytes) < nlen:
            break
        name = name_bytes.decode("utf-8", errors="replace")
        
        nd_bytes = f.read(4)
        if len(nd_bytes) < 4:
            break
        n_dims = struct.unpack("<I", nd_bytes)[0]
        if n_dims > 6:
            print(f"  Tensor[{idx}]: Invalid n_dims={n_dims} for '{name}'")
            break
        
        dims = []
        for _ in range(n_dims):
            d_bytes = f.read(8)
            if len(d_bytes) < 8:
                break
            dims.append(struct.unpack("<Q", d_bytes)[0])
        
        gt_bytes = f.read(4)
        if len(gt_bytes) < 4:
            break
        ggml_type = struct.unpack("<I", gt_bytes)[0]
        
        off_bytes = f.read(8)
        if len(off_bytes) < 8:
            break
        offset = struct.unpack("<Q", off_bytes)[0]
        
        tensors.append({
            "name": name, "dims": dims, "type": ggml_type,
            "offset": offset,
        })
        
        if idx < 5 or idx >= n_tensors - 3:
            print(f"  [{idx:3d}] '{name}' dims={dims} type={ggml_type} offset=0x{offset:X}")
    
    print(f"\n  Total tensors parsed: {len(tensors)}/{n_tensors}")
    
    # Tensor type distribution
    if tensors:
        type_counts = {}
        for t in tensors:
            tn = t['type']
            type_counts[tn] = type_counts.get(tn, 0) + 1
        print(f"  Tensor type distribution: {type_counts}")
    
    f.close()


def _skip(f, n):
    CHUNK = 4 * 1024 * 1024
    while n > 0:
        d = f.read(min(CHUNK, n))
        if not d:
            return
        n -= len(d)


def _skip_value(f, dtype):
    if dtype in (4, 5, 6, 10):
        _skip(f, 4)
    elif dtype == 7:
        _skip(f, 8)
    elif dtype == 8:
        slen = struct.unpack("<Q", f.read(8))[0]
        _skip(f, slen)
    elif dtype == 9:
        et = struct.unpack("<I", f.read(4))[0]
        al = struct.unpack("<Q", f.read(8))[0]
        if et == 8:
            total = 0
            for _ in range(al):
                sl = struct.unpack("<Q", f.read(8))[0]
                total += sl
            _skip(f, total)
        else:
            sizes = {4:4,5:4,6:4,7:8,10:4,11:8,12:8}
            _skip(f, sizes.get(et, 4) * al)
    elif dtype == 11:
        _skip(f, 8)
    elif dtype == 12:
        _skip(f, 8)
    else:
        _skip(f, 4)


def _read_value(f, dtype, key, idx):
    if dtype in (4, 5, 10):
        v = f.read(4)
        return struct.unpack("<I", v)[0] if len(v) == 4 else None
    elif dtype == 6:
        v = f.read(4)
        return struct.unpack("<f", v)[0] if len(v) == 4 else None
    elif dtype == 7:
        v = f.read(8)
        return struct.unpack("<Q", v)[0] if len(v) == 8 else None
    elif dtype == 8:
        slen = struct.unpack("<Q", f.read(8))[0]
        _skip(f, slen)
        return f"<str-len={slen}>"
    elif dtype == 9:
        # Read 12-byte probe
        probe = f.read(12)
        if len(probe) < 12:
            return "<truncated>"
        elem_type = struct.unpack("<I", probe[0:4])[0]
        array_len = struct.unpack("<Q", probe[4:12])[0]
        
        if array_len < 10000000 and elem_type in (4, 5, 6, 7, 8, 10, 11, 12):
            if elem_type == 8:
                total = 0
                for _ in range(array_len):
                    sl_raw = f.read(8)
                    if len(sl_raw) < 8:
                        break
                    sl = struct.unpack("<Q", sl_raw)[0]
                    total += sl
                _skip(f, total)
                return f"<str-array[{array_len}]>"
            else:
                sizes = {4:4,5:4,6:4,7:8,10:4,11:8,12:8}
                total = sizes.get(elem_type, 4) * array_len
                _skip(f, total)
                return f"<array[{array_len}]>"
        else:
            # Treat as bool - back up to after klen+dtype and read 1 byte
            # Actually we read 12 bytes. For bool, value is 1 byte + padding to 4 bytes
            # We need to rewind 11 bytes and handle properly
            # Let's rewind and handle from the dtype value position
            return "<bool-unhandled>"
    elif dtype == 11:
        v = f.read(8)
        return struct.unpack("<d", v)[0] if len(v) == 8 else None
    elif dtype == 12:
        v = f.read(8)
        return struct.unpack("<q", v)[0] if len(v) == 8 else None
    else:
        v = f.read(4)
        return f"<unknown dtype={dtype}>"


if __name__ == "__main__":
    models = [
        r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf",
        r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
        r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
    ]
    for path in models:
        if os.path.exists(path):
            print(f"\n{'#'*70}")
            print(f"# {os.path.basename(path)}")
            print(f"{'#'*70}")
            parse_file(path)
