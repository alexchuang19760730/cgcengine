#!/usr/bin/env python3
"""Search for tensor info in GGUF files."""
import struct
import os

def skip_val(f, dtype):
    if dtype in (4, 5, 6, 10):
        f.seek(4, 1)
    elif dtype == 7:
        f.seek(8, 1)
    elif dtype == 8:
        slen = struct.unpack("<Q", f.read(8))[0]
        _big_skip(f, slen)
    elif dtype == 9:
        et_raw = f.read(4)
        if len(et_raw) < 4: return
        et = struct.unpack("<I", et_raw)[0]
        al_raw = f.read(8)
        if len(al_raw) < 8: return
        al = struct.unpack("<Q", al_raw)[0]
        if et == 8:
            total = 0
            for _ in range(al):
                sl_raw = f.read(8)
                if len(sl_raw) < 8: break
                sl = struct.unpack("<Q", sl_raw)[0]
                total += sl
            _big_skip(f, total)
        else:
            sizes = {4:4,5:4,6:4,7:8,10:4,11:8,12:8}
            f.seek(sizes.get(et, 4) * al, 1)
    elif dtype == 11:
        f.seek(8, 1)
    elif dtype == 12:
        f.seek(8, 1)
    else:
        f.seek(4, 1)

def _big_skip(f, n):
    """Skip potentially large values in chunks."""
    CHUNK = 4 * 1024 * 1024
    while n > 0:
        d = f.read(min(CHUNK, n))
        if not d: break
        n -= len(d)

# Analyze Q4_K_M
path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf"
fsize = os.path.getsize(path)

with open(path, "rb") as f:
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    print(f"Version={version}, n_tensors={n_tensors}, n_kv={n_kv}, fsize={fsize}")
    
    # Skip KV entries correctly
    for idx in range(n_kv):
        klen_raw = f.read(8)
        if len(klen_raw) < 8: break
        klen = struct.unpack("<Q", klen_raw)[0]
        if klen <= 65536:
            key_raw = f.read(klen)
        else:
            f.seek(klen, 1)
            dtype = struct.unpack("<I", f.read(4))[0]
            skip_val(f, dtype)
            continue
        dtype = struct.unpack("<I", f.read(4))[0]
        skip_val(f, dtype)
    
    pos_after_kv = f.tell()
    aligned = (pos_after_kv + 31) & ~31
    
    print(f"Pos after KV: {pos_after_kv} (0x{pos_after_kv:X})")
    print(f"Remaining bytes: {fsize - pos_after_kv}")
    
    # Search for tensor names in first 1MB after KV
    print(f"\n--- Searching first 1MB after KV ---")
    f.seek(pos_after_kv)
    chunk = f.read(min(1_000_000, fsize - pos_after_kv))
    
    for pattern in [b"token_embd", b"blk.0.attn_norm", b"output_norm", b"blk.0.ffn_gate", b"blk.0.ffn_norm"]:
        idx = chunk.find(pattern)
        if idx >= 0:
            print(f"  Found '{pattern.decode()}' at +{idx} (abs pos {pos_after_kv+idx})")
        else:
            print(f"  '{pattern.decode()}' NOT FOUND")
    
    # Search last 1MB
    print(f"\n--- Searching last 1MB ---")
    f.seek(max(0, fsize - 1_000_000))
    tail = f.read(min(1_000_000, fsize))
    
    for pattern in [b"token_embd", b"blk.0.attn_norm", b"output_norm", b"blk.0.ffn_gate", b"blk.0.ffn_norm"]:
        idx = tail.find(pattern)
        if idx >= 0:
            print(f"  Found '{pattern.decode()}' at +{idx} from tail (abs pos {max(0, fsize - 1_000_000) + idx})")
        else:
            print(f"  '{pattern.decode()}' NOT FOUND in last 1MB")
    
    # Show what's at aligned position
    print(f"\n--- First 128 bytes at aligned ({aligned}) ---")
    f.seek(aligned)
    data = f.read(128)
    hex_str = ' '.join(f'{b:02x}' for b in data)
    print(f"  {hex_str}")
    
    # Check if what we have is just raw tensor data
    # For Q4_K_M, the tensors start at aligned position and extend to EOF
    # There should be NO tensor info section between KV and data
    # The info is embedded differently
    
    # Let's see if there's a "tensor data" section header
    print(f"\n--- Checking if file has separate tensor info section ---")
    # In GGUF, after KV comes:
    # 1. padding to 32 bytes
    # 2. For each tensor: name (klen+data) + dims + type + offset
    # 3. Then padding to 32 bytes
    # 4. Then actual tensor data
    
    # Let's try parsing the aligned position as tensor info
    print(f"\n--- Attempting tensor parse from aligned ---")
    f.seek(aligned)
    
    # Read first potential tensor entry
    nlen_raw = f.read(8)
    if len(nlen_raw) == 8:
        nlen = struct.unpack("<Q", nlen_raw)[0]
        print(f"  First nlen: {nlen}")
        
        # Check if this looks like a valid tensor name
        if 1 <= nlen <= 256:
            name_bytes = f.read(nlen)
            if len(name_bytes) == nlen:
                name = name_bytes.decode("utf-8", errors="replace")
                print(f"  First tensor name: '{name}'")
                
                nd_raw = f.read(4)
                if len(nd_raw) == 4:
                    n_dims = struct.unpack("<I", nd_raw)[0]
                    print(f"  n_dims: {n_dims}")
                    
                    dims = []
                    for _ in range(min(n_dims, 6)):
                        d_raw = f.read(8)
                        if len(d_raw) < 8: break
                        dims.append(struct.unpack("<Q", d_raw)[0])
                    print(f"  dims: {dims}")
                    
                    gt_raw = f.read(4)
                    if len(gt_raw) == 4:
                        ggml_type = struct.unpack("<I", gt_raw)[0]
                        print(f"  ggml_type: {ggml_type}")
                        
                        off_raw = f.read(8)
                        if len(off_raw) == 8:
                            offset = struct.unpack("<Q", off_raw)[0]
                            print(f"  offset: {offset} (0x{offset:X})")
    
    print(f"\n--- If this fails, the file likely has NO separate tensor info section ---")
    print(f"--- Instead, tensor info may be stored as a KV entry or in a different format ---")
