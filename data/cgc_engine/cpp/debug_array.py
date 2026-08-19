#!/usr/bin/env python3
"""Debug the array reading issue."""
import struct
import os

def _big_skip(f, n):
    CHUNK = 4 * 1024 * 1024
    while n > 0:
        d = f.read(min(CHUNK, n))
        if not d:
            break
        n -= len(d)

def _read_array_value(f, elem_type, array_len):
    if elem_type == 8:
        if array_len <= 1000:
            for i in range(array_len):
                sl_bytes = f.read(8)
                if len(sl_bytes) < 8:
                    print(f"    ERROR: EOF at string {i} length")
                    return False
                sl = struct.unpack("<Q", sl_bytes)[0]
                _big_skip(f, sl)
        else:
            total_str_bytes = 0
            _read_remaining = array_len
            while _read_remaining > 0:
                chunk = min(_read_remaining, 1000000)
                prefixes = f.read(chunk * 8)
                if len(prefixes) < chunk * 8:
                    print(f"    ERROR: EOF reading prefixes ({len(prefixes)} < {chunk * 8})")
                    return False
                for i in range(chunk):
                    sl = struct.unpack("<Q", prefixes[i * 8:(i + 1) * 8])[0]
                    total_str_bytes += sl
                _read_remaining -= chunk
            print(f"    String array: {array_len} strings, total content: {total_str_bytes:,} bytes")
            _big_skip(f, total_str_bytes)
        return True
    else:
        elem_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        elem_size = elem_sizes.get(elem_type, 4)
        total = elem_size * array_len
        if total > 2 * 1024 * 1024 * 1024:
            _big_skip(f, total)
        else:
            f.seek(total, 1)
        return True

def debug_parse(filepath):
    file_size = os.path.getsize(filepath)
    f = open(filepath, "rb")
    
    print(f"File: {os.path.basename(filepath)} (size: {file_size:,})")
    
    # Header
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    
    print(f"Version: {version}, n_kv: {n_kv}, n_tensors: {n_tensors}")
    
    for idx in range(n_kv):
        start = f.tell()
        klen_bytes = f.read(8)
        if len(klen_bytes) < 8:
            print(f"  [{idx}] ERROR: EOF at klen (pos={start})")
            break
        klen = struct.unpack("<Q", klen_bytes)[0]
        if klen == 0 or klen > 1000000:
            print(f"  [{idx}] ERROR: Invalid klen={klen} (pos={start})")
            break
        
        key_bytes = f.read(klen)
        if len(key_bytes) < klen:
            print(f"  [{idx}] ERROR: EOF at key (pos={start})")
            break
        key = key_bytes.decode("utf-8", errors="replace")
        
        dtype_bytes = f.read(4)
        if len(dtype_bytes) < 4:
            break
        dtype = struct.unpack("<I", dtype_bytes)[0]
        
        print(f"  [{idx:3d}] @{start:12d} key_len={klen:6d} dtype={dtype:2d} key='{key[:50]}'")
        
        val_start = f.tell()
        
        ok = True
        if dtype == 7:
            v = f.read(1)
            if len(v) != 1:
                ok = False
            else:
                print(f"    bool={v[0]}")
        elif dtype == 8:
            slen_bytes = f.read(8)
            if len(slen_bytes) < 8:
                ok = False
            else:
                slen = struct.unpack("<Q", slen_bytes)[0]
                s = f.read(slen)
                if len(s) < slen:
                    ok = False
                else:
                    print(f"    string[{slen}]")
        elif dtype == 9:
            et = f.read(4)
            al = f.read(8)
            if len(et) < 4 or len(al) < 8:
                ok = False
            else:
                elem_type = struct.unpack("<I", et)[0]
                array_len = struct.unpack("<Q", al)[0]
                print(f"    array[{array_len}] elem_type={elem_type}")
                ok = _read_array_value(f, elem_type, array_len)
        elif dtype == 10:
            v = f.read(8)
            ok = len(v) == 8
        elif dtype == 11:
            v = f.read(8)
            ok = len(v) == 8
        elif dtype == 12:
            v = f.read(8)
            ok = len(v) == 8
        elif dtype in (0, 1):
            v = f.read(1)
            ok = len(v) == 1
        elif dtype in (2, 3):
            v = f.read(2)
            ok = len(v) == 2
        elif dtype in (4, 5, 6):
            v = f.read(4)
            ok = len(v) == 4
        else:
            f.seek(4, 1)
        
        if not ok:
            print(f"  [{idx}] PARSE ERROR at pos {val_start}!")
            break
        
        end = f.tell()
        print(f"    -> @{end:12d} ({end - start:8d} bytes)")
    
    pos_after = f.tell()
    print(f"\nAfter KV: pos={pos_after} (0x{pos_after:X})")
    
    # Now check if we can find tensor info
    print(f"\nSearching for tensor info...")
    for delta in range(0, 4096, 32):
        pos = pos_after + delta
        if pos >= file_size - 100:
            break
        
        f.seek(pos)
        nl_bytes = f.read(8)
        if len(nl_bytes) < 8:
            continue
        nl = struct.unpack("<Q", nl_bytes)[0]
        if nl < 3 or nl > 256:
            continue
        
        name = f.read(nl)
        if len(name) < nl:
            continue
        name_str = name.decode("utf-8", errors="replace")
        
        if not any(kw in name_str for kw in ["blk.", "token_embd", "output", "norm.", "rope.", "tokenizer"]):
            continue
        
        nd_bytes = f.read(4)
        if len(nd_bytes) < 4:
            continue
        nd = struct.unpack("<I", nd_bytes)[0]
        if nd == 0 or nd > 5:
            continue
        
        dims = []
        for _ in range(nd):
            d = f.read(8)
            if len(d) < 8:
                break
            dims.append(struct.unpack("<Q", d)[0])
        if len(dims) != nd:
            continue
        
        gt_bytes = f.read(4)
        if len(gt_bytes) < 4:
            continue
        gt = struct.unpack("<I", gt_bytes)[0]
        if gt > 50:
            continue
        
        off_bytes = f.read(8)
        if len(off_bytes) < 8:
            continue
        off = struct.unpack("<Q", off_bytes)[0]
        if off > file_size:
            continue
        
        print(f"  FOUND at {pos} (delta +{delta}): '{name_str}' dims={dims} type={gt} offset=0x{off:X}")
        
        # Count more
        count = 1
        for _ in range(n_tensors - 1):
            nl2 = f.read(8)
            if len(nl2) < 8:
                break
            nl2 = struct.unpack("<Q", nl2)[0]
            if nl2 < 3 or nl2 > 256:
                break
            nm = f.read(nl2)
            if len(nm) < nl2:
                break
            count += 1
        
        print(f"  Can parse {count}/{n_tensors} tensors")
        return
    
    print("  FAILED to find tensor info")

if __name__ == "__main__":
    path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
    if os.path.exists(path):
        debug_parse(path)
