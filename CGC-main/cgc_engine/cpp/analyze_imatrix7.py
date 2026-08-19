#!/usr/bin/env python3
"""
Debug: dump EVERY KV entry's position and size to find where parsing goes wrong.
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
            return False
        n -= len(d)
    return True

def main():
    for label, path in MODELS.items():
        if not os.path.exists(path):
            continue
        fsize = os.path.getsize(path)
        print(f"\n{'='*60}")
        print(f"MODEL: {label} ({fsize/1e9:.2f} GB)")
        print(f"{'='*60}")
        
        with open(path, "rb") as f:
            f.seek(0)
            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            print(f"Version={version}, n_tensors={n_tensors}, n_kv={n_kv}")
            print(f"File size: {fsize}")
            
            for idx in range(n_kv):
                entry_start = f.tell()
                
                # Read klen
                klen_raw = f.read(8)
                if len(klen_raw) < 8:
                    print(f"  KV[{idx}]: EOF reading klen at pos={entry_start}")
                    break
                klen = struct.unpack("<Q", klen_raw)[0]
                
                # Read key
                if klen <= 65536:
                    key = f.read(klen).decode("utf-8", errors="replace")
                else:
                    # Big key - skip it
                    skip(f, klen)
                    # Now read dtype and skip value
                    dtype = struct.unpack("<I", f.read(4))[0]
                    value_bytes = {4:4, 5:4, 6:4, 7:8, 8:0, 9:0, 10:4, 11:8, 12:8}
                    if dtype in (4, 5, 6, 10, 11, 12):
                        skip(f, value_bytes[dtype])
                    elif dtype == 8:
                        slen = struct.unpack("<Q", f.read(8))[0]
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
                    entry_end = f.tell()
                    print(f"  KV[{idx}]: BIG key klen={klen} dtype={dtype} | pos {entry_start} -> {entry_end} (delta={entry_end-entry_start})")
                    continue
                
                # Read dtype
                dtype_raw = f.read(4)
                if len(dtype_raw) < 4:
                    print(f"  KV[{idx}] '{key}': EOF reading dtype at pos={f.tell()}")
                    break
                dtype = struct.unpack("<I", dtype_raw)[0]
                
                # Save position before value
                before_val = f.tell()
                
                # Read/skip value
                if dtype in (4, 5, 10):
                    v = f.read(4)
                    val = struct.unpack("<I", v)[0] if len(v) == 4 else None
                elif dtype == 6:
                    v = f.read(4)
                    val = struct.unpack("<f", v)[0] if len(v) == 4 else None
                elif dtype == 7:
                    v = f.read(8)
                    val = struct.unpack("<Q", v)[0] if len(v) == 8 else None
                elif dtype == 8:
                    slen = struct.unpack("<Q", f.read(8))[0]
                    skip(f, slen)
                    val = f"<str len={slen}>"
                elif dtype == 9:
                    # Read the 12-byte probe
                    probe = f.read(12)
                    if len(probe) < 12:
                        break
                    elem_type = struct.unpack("<I", probe[0:4])[0]
                    array_len = struct.unpack("<Q", probe[4:12])[0]
                    
                    if array_len < 10000000 and elem_type in (4, 5, 6, 7, 8, 10, 11, 12):
                        if elem_type == 8:
                            total = 0
                            for _ in range(array_len):
                                sl_raw = f.read(8)
                                if len(sl_raw) < 8:
                                    print(f"  KV[{idx}] '{key}': EOF in string array at entry {_}")
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
                        # Treat as bool: rewind past the 12-byte probe, read 1 byte
                        f.seek(before_val + 1)
                        val = True
                elif dtype == 11:
                    v = f.read(8)
                    val = struct.unpack("<d", v)[0] if len(v) == 8 else None
                elif dtype == 12:
                    v = f.read(8)
                    val = struct.unpack("<q", v)[0] if len(v) == 8 else None
                else:
                    v = f.read(4)
                    val = f"<unknown dtype={dtype}>"
                
                entry_end = f.tell()
                delta = entry_end - entry_start
                
                # Pretty print value
                vs = str(val)
                if len(vs) > 80:
                    vs = vs[:80] + "..."
                
                print(f"  KV[{idx}]: pos={entry_start} key='{key}' dtype={dtype} val={vs} | {delta} bytes")
                
                # Validate next entry
                if idx < n_kv - 1:
                    next_test = f.read(min(8, 65536))
                    if len(next_test) >= 8:
                        next_klen = struct.unpack("<Q", next_test[:8])[0]
                        if next_klen > 65536 or next_klen < 3:
                            print(f"    ^^^ NEXT klen={next_klen} looks WRONG! (expected 3-65536)")
                            f.seek(entry_end)  # restore
                        else:
                            f.seek(entry_end)
                    else:
                        f.seek(entry_end)
            
            final_pos = f.tell()
            aligned = (final_pos + 31) & ~31
            print(f"\n  Final pos: {final_pos} (0x{final_pos:X})")
            print(f"  Aligned pos: {aligned} (0x{aligned:X})")
            print(f"  Remaining: {fsize - final_pos} bytes")
            
            if fsize - final_pos < 100:
                print(f"  *** PROBLEM: File ends right after KV! ***")

if __name__ == "__main__":
    main()
