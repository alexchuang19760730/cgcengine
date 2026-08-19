#!/usr/bin/env python3
"""Debug KV parsing step by step."""
import sys
sys.path.insert(0, r"D:\alex\flashkv0516")

import struct
import os

def _big_skip(f, n):
    CHUNK = 4 * 1024 * 1024
    while n > 0:
        d = f.read(min(CHUNK, n))
        if not d:
            break
        n -= len(d)

def read_bool(f):
    v = f.read(1)
    return v[0] if len(v) == 1 else None

def read_string(f):
    slen_bytes = f.read(8)
    if len(slen_bytes) < 8:
        return None
    slen = struct.unpack("<Q", slen_bytes)[0]
    s_bytes = f.read(slen)
    if len(s_bytes) < slen:
        return None
    return s_bytes.decode("utf-8", errors="replace")

def read_array(f, elem_type, array_len):
    if elem_type == 8:
        total = 0
        for _ in range(array_len):
            sl_bytes = f.read(8)
            if len(sl_bytes) < 8:
                return False
            sl = struct.unpack("<Q", sl_bytes)[0]
            total += sl
        _big_skip(f, total)
        return True
    else:
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        elem_size = sizes.get(elem_type, 4)
        total = elem_size * array_len
        if total > 2 * 1024 * 1024 * 1024:
            _big_skip(f, total)
        else:
            f.seek(total, 1)
        return True

def debug_kv(filepath):
    file_size = os.path.getsize(filepath)
    f = open(filepath, "rb")
    
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    
    print(f"File: {os.path.basename(filepath)}")
    print(f"Version: {version}, n_kv: {n_kv}, n_tensors: {n_tensors}")
    
    kv = {}
    for idx in range(n_kv):
        start = f.tell()
        
        klen_bytes = f.read(8)
        if len(klen_bytes) < 8:
            print(f"  [{idx}] ERROR: EOF at klen (pos={start})")
            break
        klen = struct.unpack("<Q", klen_bytes)[0]
        
        if klen == 0 or klen > 1000000:
            print(f"  [{idx}] ERROR: Invalid klen={klen} (pos={start}), raw={klen_bytes.hex()}")
            f.seek(start)
            # Search for next valid klen
            for search_delta in range(0, 1024):
                f.seek(start + search_delta)
                test = f.read(8)
                if len(test) >= 8:
                    tk = struct.unpack("<Q", test)[0]
                    if 3 <= tk <= 256:
                        print(f"    Recovered at delta +{search_delta}, klen={tk}")
                        start = f.tell() - 8
                        klen = tk
                        break
            else:
                print(f"    Could not recover")
                break
        
        key_bytes = f.read(klen)
        if len(key_bytes) < klen:
            print(f"  [{idx}] ERROR: EOF at key (pos={start})")
            break
        key = key_bytes.decode("utf-8", errors="replace")
        
        dtype_bytes = f.read(4)
        if len(dtype_bytes) < 4:
            print(f"  [{idx}] ERROR: EOF at dtype (pos={start})")
            break
        dtype = struct.unpack("<I", dtype_bytes)[0]
        
        val_start = f.tell()
        ok = True
        
        if dtype == 7:
            v = read_bool(f)
            if v is None:
                ok = False
            else:
                val_desc = f"bool={v}"
        elif dtype == 8:
            s = read_string(f)
            if s is None:
                ok = False
            else:
                val_desc = f"str[{len(s)}]"
                kv[key] = s
        elif dtype == 9:
            et_bytes = f.read(4)
            al_bytes = f.read(8)
            if len(et_bytes) < 4 or len(al_bytes) < 8:
                ok = False
            else:
                elem_type = struct.unpack("<I", et_bytes)[0]
                array_len = struct.unpack("<Q", al_bytes)[0]
                ok = read_array(f, elem_type, array_len)
                val_desc = f"array[{array_len}] elem_type={elem_type}"
        elif dtype == 10:
            v = f.read(8)
            if len(v) != 8:
                ok = False
            else:
                val_desc = f"uint64={struct.unpack('<Q', v)[0]}"
        elif dtype == 11:
            v = f.read(8)
            if len(v) != 8:
                ok = False
            else:
                val_desc = f"float64={struct.unpack('<d', v)[0]:.6g}"
        elif dtype == 12:
            v = f.read(8)
            if len(v) != 8:
                ok = False
            else:
                val_desc = f"int64={struct.unpack('<q', v)[0]}"
        elif dtype in (0, 1):
            v = f.read(1)
            if len(v) != 1:
                ok = False
            else:
                val_desc = f"uint8={v[0]}" if dtype == 0 else f"int8={v[0]}"
        elif dtype in (2, 3):
            v = f.read(2)
            if len(v) != 2:
                ok = False
            else:
                val_desc = f"uint16={struct.unpack('<H', v)[0]}" if dtype == 2 else f"int16={struct.unpack('<h', v)[0]}"
        elif dtype in (4, 5, 6):
            v = f.read(4)
            if len(v) != 4:
                ok = False
            else:
                if dtype == 4:
                    val_desc = f"uint32={struct.unpack('<I', v)[0]}"
                elif dtype == 5:
                    val_desc = f"int32={struct.unpack('<i', v)[0]}"
                else:
                    val_desc = f"float32={struct.unpack('<f', v)[0]:.6g}"
        else:
            f.seek(4, 1)
            val_desc = f"unknown dtype={dtype}"
        
        if not ok:
            print(f"  [{idx}] PARSE FAILURE! pos={val_start}, dtype={dtype}, key='{key}'")
            break
        
        end = f.tell()
        item_size = end - start
        
        if item_size > 10000 or "tokenizer" in key or "add_bos" in key or idx < 20:
            print(f"  [{idx:3d}] @{start:12d} -> @{end:12d} ({item_size:10d}B) dtype={dtype:2d} key='{key[:60]}' {val_desc}")
        
        kv[key] = val_desc
    
    pos_after = f.tell()
    print(f"\nKV parsing stopped at item {idx}, pos={pos_after} (0x{pos_after:X})")
    print(f"KV entries parsed: {len(kv)}/{n_kv}")
    
    # Try to find tensor info
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
        
        name_bytes = f.read(nl)
        if len(name_bytes) < nl:
            continue
        name = name_bytes.decode("utf-8", errors="replace")
        
        if not any(kw in name for kw in ["blk.", "token_embd", "output", "norm.", "rope.", "tokenizer"]):
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
        
        print(f"  FOUND at {pos} (delta +{delta}): '{name}' dims={dims} type={gt} offset=0x{off:X}")
        
        # Count tensors
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
        
        # Calculate data start
        f.seek(pos)
        for _ in range(count):
            nlen = struct.unpack("<Q", f.read(8))[0]
            f.seek(nlen, 1)
            nd = struct.unpack("<I", f.read(4))[0]
            f.seek(8 * nd + 4 + 8, 1)  # dims + ggml_type + offset
        info_end = f.tell()
        data_start = (info_end + 31) & ~31
        print(f"  Data start: 0x{data_start:X}")
        return
    
    print("  FAILED to find tensor info")

if __name__ == "__main__":
    path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
    debug_kv(path)
