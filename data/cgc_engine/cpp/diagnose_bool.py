#!/usr/bin/env python3
"""
Precisely diagnose bool storage in GGUF files.
Find the exact bytes around bool items and determine correct storage format.
"""
import struct
import os

def diagnose_bool_storage(filepath):
    file_size = os.path.getsize(filepath)
    f = open(filepath, "rb")
    
    # Read header
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    
    print(f"File: {os.path.basename(filepath)}")
    print(f"Size: {file_size:,} bytes, Version: {version}, n_kv: {n_kv}")
    
    # Parse all KV items
    items = []
    for idx in range(n_kv):
        start = f.tell()
        klen_bytes = f.read(8)
        if len(klen_bytes) < 8:
            break
        klen = struct.unpack("<Q", klen_bytes)[0]
        if klen == 0 or klen > 1000000:
            break
        key_bytes = f.read(klen)
        if len(key_bytes) < klen:
            break
        key = key_bytes.decode("utf-8", errors="replace")
        dtype_bytes = f.read(4)
        if len(dtype_bytes) < 4:
            break
        dtype = struct.unpack("<I", dtype_bytes)[0]
        
        before_val = f.tell()
        
        if dtype == 7:  # bool
            val_bytes = f.read(4)  # Read 4 bytes raw
            items.append({
                "idx": idx, "start": start, "klen": klen, "key": key,
                "dtype": dtype, "before_val": before_val,
                "raw_bytes": val_bytes, "val_len": 4
            })
            # Also try reading as 1+3
            f.seek(before_val)
            v1 = f.read(1)
            p3 = f.read(3)
            items[-1]["bool_1_3"] = (v1, p3)
            # And try reading as 4-byte int
            f.seek(before_val)
            v4 = f.read(4)
            items[-1]["bool_4"] = v4
            # Move to end
            f.seek(before_val + 4)
            
            # Now check what's next
            next_klen_pos = f.tell()
            next_klen_bytes = f.read(8)
            if len(next_klen_bytes) >= 8:
                next_klen = struct.unpack("<Q", next_klen_bytes)[0]
                items[-1]["next_klen_after_4"] = next_klen
        elif dtype == 8:  # string
            slen_bytes = f.read(8)
            slen = struct.unpack("<Q", slen_bytes)[0]
            f.seek(slen, 1)
            items.append({"idx": idx, "start": start, "klen": klen, "key": key, "dtype": dtype})
        elif dtype == 9:  # array
            et = f.read(4)
            al = f.read(8)
            elem_type = struct.unpack("<I", et)[0]
            array_len = struct.unpack("<Q", al)[0]
            if elem_type == 8:  # string array
                total = 0
                for _ in range(array_len):
                    sl = struct.unpack("<Q", f.read(8))[0]
                    total += 8 + sl
                    f.seek(sl, 1)
                items.append({"idx": idx, "start": start, "klen": klen, "key": key, "dtype": dtype, "array_len": array_len, "total_size": total})
            else:
                sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
                es = sizes.get(elem_type, 4)
                total = es * array_len
                f.seek(total, 1)
                items.append({"idx": idx, "start": start, "klen": klen, "key": key, "dtype": dtype, "array_len": array_len, "total_size": total})
        else:
            sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
            s = sizes.get(dtype, 4)
            f.seek(s, 1)
            items.append({"idx": idx, "start": start, "klen": klen, "key": key, "dtype": dtype})
    
    # Now analyze bool items
    print(f"\n--- Found {len(items)} KV items ---")
    bool_items = [it for it in items if it["dtype"] == 7]
    
    for bi in bool_items:
        print(f"\n  Bool item #{bi['idx']}: '{bi['key']}'")
        print(f"    Position: {bi['start']}")
        print(f"    Value starts at: {bi['before_val']}")
        print(f"    Raw 4 bytes: {bi['raw_bytes'].hex()}")
        
        v1, p3 = bi["bool_1_3"]
        print(f"    Interpreted as 1+3: value={v1[0] if len(v1)==1 else 'ERR'}, padding={p3.hex()}")
        
        v4 = bi["bool_4"]
        print(f"    Interpreted as 4-byte int: {struct.unpack('<i', v4)[0]}")
        
        # Check what follows
        if "next_klen_after_4" in bi:
            nk = bi["next_klen_after_4"]
            valid = 3 <= nk <= 10000000
            print(f"    Next klen after 4 bytes: {nk} {'✓ VALID' if valid else '✗ INVALID'}")
        
        # Also try 8-byte read
        f.seek(bi["before_val"] + 8)
        nk8_bytes = f.read(8)
        if len(nk8_bytes) >= 8:
            nk8 = struct.unpack("<Q", nk8_bytes)[0]
            valid8 = 3 <= nk8 <= 10000000
            print(f"    Next klen after 8 bytes: {nk8} {'✓ VALID' if valid8 else '✗ INVALID'}")
    
    # Check the item just before the first bool
    if bool_items:
        first_bool_idx = bool_items[0]["idx"]
        if first_bool_idx > 0:
            prev = items[first_bool_idx - 1]
            print(f"\n  Previous item (#{prev['idx']}): '{prev['key']}' dtype={prev['dtype']} ends at approximately {prev['before_val']}")
    
    f.close()

def scan_for_tensor_info(filepath, kv_end):
    """Scan forward from kv_end to find tensor info section."""
    file_size = os.path.getsize(filepath)
    f = open(filepath, "rb")
    
    # Try scanning forward from kv_end
    print(f"\n--- Scanning for tensor info from KV end ({kv_end}) ---")
    
    # The tensor info section should be 32-byte aligned after KV
    # But the issue is we don't know exactly where KV ends due to bool parsing
    
    # Let's try different offsets
    for offset_delta in range(0, 256, 32):
        pos = kv_end + offset_delta
        if pos >= file_size - 100:
            break
        
        f.seek(pos)
        
        # Try to read first tensor
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
        
        # Validate
        valid_prefixes = ["blk.", "token_embd", "output", "norm.", "rope.", "tokenizer"]
        if not any(kw in name for kw in valid_prefixes):
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
        
        print(f"  FOUND at pos {pos} (delta +{offset_delta}):")
        print(f"    First tensor: '{name}' dims={dims} type={ggml_type} offset=0x{offset:X}")
        
        # Try to parse more tensors
        count = 1
        for _ in range(100):
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
        
        print(f"    Can parse {count} tensors from this position")
        f.close()
        return pos, count, name
    
    print("  FAILED to find tensor info within 256 bytes of KV end")
    f.close()
    return None, 0, None

if __name__ == "__main__":
    models = [
        r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
    ]
    
    for path in models:
        if not os.path.exists(path):
            print(f"SKIP: {path} not found")
            continue
        
        diagnose_bool_storage(path)
