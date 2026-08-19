#!/usr/bin/env python3
"""
Full GGUF parser. The issue is our KV parsing is stopping early because
dtype=9 (array) with elem_type=8 (string) is consuming the wrong number of bytes.
Let's debug the exact failure point by stopping at the first bad KV entry.
"""

import struct
import os

MODELS = {
    "Q4_K_M": r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf",
    "IQ2_XXS": r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
}

CHUNK = 4 * 1024 * 1024

def skip(f, n):
    while n > 0:
        d = f.read(min(CHUNK, n))
        if not d:
            break
        n -= len(d)

def parse_all_kv(f, n_kv):
    """Parse all KV entries, return (kv_dict, last_kv_end, errors)."""
    kv = {}
    errors = []
    
    for idx in range(n_kv):
        entry_start = f.tell()
        klen_raw = f.read(8)
        if len(klen_raw) < 8:
            errors.append(f"KV[{idx}]: EOF at klen (pos={entry_start})")
            break
        klen = struct.unpack("<Q", klen_raw)[0]
        
        if klen <= 65536:
            key_raw = f.read(klen)
            if len(key_raw) < klen:
                errors.append(f"KV[{idx}]: EOF at key (pos={entry_start}, klen={klen})")
                break
            key = key_raw.decode("utf-8", errors="replace")
        else:
            skip(f, klen)
            dtype = struct.unpack("<I", f.read(4))[0]
            kv[f"<big-klen={klen}"] = f"<skipped dtype={dtype}>"
            if dtype in (4, 5, 6, 10):
                skip(f, 4)
            elif dtype == 7:
                skip(f, 8)
            elif dtype == 8:
                slen_raw = f.read(8)
                if len(slen_raw) < 8:
                    break
                slen = struct.unpack("<Q", slen_raw)[0]
                skip(f, slen)
            elif dtype == 9:
                et = struct.unpack("<I", f.read(4))[0]
                al = struct.unpack("<Q", f.read(8))[0]
                if et == 8:
                    total = 0
                    for _ in range(al):
                        sl = struct.unpack("<Q", f.read(8))[0]
                        total += sl
                    skip(f, total)
                else:
                    sizes = {4:4,5:4,6:4,7:8,10:4,11:8,12:8}
                    skip(f, sizes.get(et, 4) * al)
            elif dtype == 11:
                skip(f, 8)
            elif dtype == 12:
                skip(f, 8)
            else:
                skip(f, 4)
            continue
        
        # Now read value dtype
        dtype_raw = f.read(4)
        if len(dtype_raw) < 4:
            errors.append(f"KV[{idx}]: EOF at dtype for key '{key}' (pos={entry_start})")
            break
        dtype = struct.unpack("<I", dtype_raw)[0]
        
        # Now parse value and track position carefully
        val_start = f.tell()
        
        if dtype in (4, 5, 10):
            v = f.read(4)
            if len(v) < 4:
                errors.append(f"KV[{idx}] '{key}': truncated uint32 (pos={val_start})")
                break
            val = struct.unpack("<I", v)[0]
        elif dtype == 6:
            v = f.read(4)
            if len(v) < 4:
                break
            val = struct.unpack("<f", v)[0]
        elif dtype == 7:
            v = f.read(8)
            if len(v) < 8:
                break
            val = struct.unpack("<Q", v)[0]
        elif dtype == 8:
            slen_raw = f.read(8)
            if len(slen_raw) < 8:
                break
            slen = struct.unpack("<Q", slen_raw)[0]
            skip(f, slen)
            val = f"<str-len={slen}>"
        elif dtype == 9:
            # Array or bool
            probe = f.read(12)
            if len(probe) < 12:
                break
            elem_type = struct.unpack("<I", probe[0:4])[0]
            array_len = struct.unpack("<Q", probe[4:12])[0]
            
            # Check if this looks like a valid array
            if array_len < 10000000 and elem_type in (4, 5, 6, 7, 8, 10, 11, 12):
                if elem_type == 8:
                    total = 0
                    for _ in range(array_len):
                        sl_raw = f.read(8)
                        if len(sl_raw) < 8:
                            errors.append(f"KV[{idx}] '{key}': truncated string array len at entry {_}")
                            break
                        sl = struct.unpack("<Q", sl_raw)[0]
                        total += sl
                    skip(f, total)
                    val = f"<str-array[{array_len}]>"
                else:
                    sizes = {4:4,5:4,6:4,7:8,10:4,11:8,12:8}
                    total = sizes.get(elem_type, 4) * array_len
                    skip(f, total)
                    val = f"<array[{array_len}]>"
            else:
                # Likely bool (dtype=9 in GGUF v2) - just 1 byte
                # We already read 12 bytes, need to back up
                # Actually, we read 12 bytes that might include bool + next KV start
                # For now, handle the common case
                # If array_len is unreasonable, treat as bool
                # The 12 bytes we read: first 4 = elem_type (could be 1 for bool), 
                # next 8 = bool value + next KV's klen
                # We need to unread 11 bytes (keeping only the bool byte)
                f.seek(val_start + 1)  # rewind to after the bool byte
                val = True
        elif dtype == 11:
            v = f.read(8)
            if len(v) < 8:
                break
            val = struct.unpack("<d", v)[0]
        elif dtype == 12:
            v = f.read(8)
            if len(v) < 8:
                break
            val = struct.unpack("<q", v)[0]
        else:
            v = f.read(4)
            val = f"<unknown dtype={dtype}>"
        
        kv[key] = val
    
    return kv, f.tell(), errors


def main():
    for label, path in MODELS.items():
        if not os.path.exists(path):
            print(f"[SKIP] {label}")
            continue
        
        fsize = os.path.getsize(path)
        print(f"\n{'='*60}")
        print(f"MODEL: {label} ({fsize/1e9:.2f} GB)")
        print(f"{'='*60}")
        
        with open(path, "rb") as f:
            # Header
            magic = f.read(4)
            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            print(f"Version={version}, n_tensors={n_tensors}, n_kv={n_kv}")
            
            # Parse all KV
            kv, pos_after_kv, errors = parse_all_kv(f, n_kv)
            
            print(f"\nKV parsing:")
            print(f"  Parsed keys: {len(kv)}")
            print(f"  Pos after KV: {pos_after_kv} (0x{pos_after_kv:X})")
            if errors:
                print(f"  ERRORS ({len(errors)}):")
                for e in errors[:10]:
                    print(f"    {e}")
            else:
                print(f"  No errors!")
            
            # Print all keys for verification
            print(f"\n  KV keys:")
            for k, v in sorted(kv.items()):
                vs = str(v)
                if len(vs) > 100:
                    vs = vs[:100] + "..."
                print(f"    {k} = {vs}")
            
            # Now try to parse tensors
            aligned = (pos_after_kv + 31) & ~31
            f.seek(aligned)
            
            print(f"\n  Trying tensor parse from aligned ({aligned} = 0x{aligned:X})...")
            tensors = []
            
            for i in range(n_tensors):
                nlen_raw = f.read(8)
                if len(nlen_raw) < 8:
                    print(f"  Stopped at tensor {i}: EOF")
                    break
                nlen = struct.unpack("<Q", nlen_raw)[0]
                if nlen == 0 or nlen > 256:
                    print(f"  Stopped at tensor {i}: invalid nlen={nlen}")
                    # Show hex context
                    f.seek(aligned + i * 50)
                    ctx = f.read(64)
                    print(f"    Hex: {' '.join(f'{b:02x}' for b in ctx[:32])}")
                    break
                
                name_raw = f.read(nlen)
                if len(name_raw) < nlen:
                    break
                name = name_raw.decode("utf-8", errors="replace")
                
                if not name.replace("_", "").replace(".", "").replace("-", "").isalnum():
                    print(f"  Stopped at tensor {i}: invalid name '{name}'")
                    break
                
                nd_raw = f.read(4)
                if len(nd_raw) < 4:
                    break
                n_dims = struct.unpack("<I", nd_raw)[0]
                if n_dims == 0 or n_dims > 6:
                    print(f"  Stopped at tensor {i}: invalid n_dims={n_dims}")
                    break
                
                dims = []
                for _ in range(n_dims):
                    d_raw = f.read(8)
                    if len(d_raw) < 8:
                        break
                    dims.append(struct.unpack("<Q", d_raw)[0])
                if len(dims) < n_dims:
                    break
                
                gt_raw = f.read(4)
                if len(gt_raw) < 4:
                    break
                ggml_type = struct.unpack("<I", gt_raw)[0]
                
                off_raw = f.read(8)
                if len(off_raw) < 8:
                    break
                offset = struct.unpack("<Q", off_raw)[0]
                
                GGML_TYPE_BYTES = {0:4, 1:2, 30:2, 8:8, 14:4, 21:3, 22:3, 16:4, 17:4, 19:2, 20:2}
                bpe = GGML_TYPE_BYTES.get(ggml_type, 4)
                ne = 1
                for d in dims:
                    ne *= d
                
                tensors.append({
                    "name": name, "dims": dims, "type": ggml_type,
                    "offset": offset, "size_bytes": int(ne * bpe),
                })
            
            print(f"\n  Tensors parsed: {len(tensors)}/{n_tensors}")
            
            if tensors:
                # Show first and last
                print(f"\n  First 5 tensors:")
                for t in tensors[:5]:
                    print(f"    {t['name']:50s} type={t['type']:2d} dims={t['dims']} offset=0x{t['offset']:X}")
                
                print(f"\n  Last 5 tensors:")
                for t in tensors[-5:]:
                    print(f"    {t['name']:50s} type={t['type']:2d} dims={t['dims']} offset=0x{t['offset']:X}")
                
                # Tensor type distribution
                type_counts = {}
                for t in tensors:
                    tn = t['type']
                    type_counts[tn] = type_counts.get(tn, 0) + 1
                print(f"\n  Tensor types: {type_counts}")

if __name__ == "__main__":
    main()
