#!/usr/bin/env python3
"""Direct hex dump of GGUF files at key positions to understand structure."""
import struct
import os

MODELS = {
    "Q4_K_M": r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf",
    "IQ2_XXS": r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
    "IQ3_XXS": r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
}

CHUNK = 4 * 1024 * 1024

def skip(f, n):
    while n > 0:
        d = f.read(min(CHUNK, n))
        if not d:
            break
        n -= len(d)

def main():
    for label, path in MODELS.items():
        if not os.path.exists(path):
            print(f"[SKIP] {label}")
            continue
        
        print(f"\n{'='*60}")
        print(f"MODEL: {label}")
        print(f"{'='*60}")
        
        with open(path, "rb") as f:
            # Header
            magic = f.read(4)
            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            print(f"Version={version}, n_tensors={n_tensors}, n_kv={n_kv}")
            
            # Parse all KV - but just skip them and track positions
            kv_count = 0
            last_kv_end = 0
            
            for kv_idx in range(n_kv):
                klen_raw = f.read(8)
                if len(klen_raw) < 8:
                    break
                klen = struct.unpack("<Q", klen_raw)[0]
                
                if klen <= 65536:
                    key = f.read(klen).decode("utf-8", errors="replace")
                else:
                    skip(f, klen)
                    dtype_raw = f.read(4)
                    if len(dtype_raw) < 4:
                        break
                    dtype = struct.unpack("<I", dtype_raw)[0]
                    print(f"  KV[{kv_idx}] BIG klen={klen} dtype={dtype}")
                    if dtype in (4, 5, 10):
                        skip(f, 4)
                    elif dtype == 6:
                        skip(f, 4)
                    elif dtype == 7:
                        skip(f, 8)
                    elif dtype == 8:
                        slen = struct.unpack("<Q", f.read(8))[0]
                        skip(f, slen)
                    elif dtype == 9:
                        elem_type = struct.unpack("<I", f.read(4))[0]
                        array_len = struct.unpack("<Q", f.read(8))[0]
                        if elem_type == 8:
                            total = 0
                            for _ in range(array_len):
                                sl = struct.unpack("<Q", f.read(8))[0]
                                total += sl
                            skip(f, total)
                        else:
                            sizes = {4:4, 5:4, 6:4, 7:8, 10:4, 11:8, 12:8}
                            skip(f, sizes.get(elem_type, 4) * array_len)
                    elif dtype == 11:
                        skip(f, 8)
                    elif dtype == 12:
                        skip(f, 8)
                    kv_count += 1
                    last_kv_end = f.tell()
                    continue
                # key was read above if klen <= 65536
                
                dtype_raw = f.read(4)
                if len(dtype_raw) < 4:
                    break
                dtype = struct.unpack("<I", dtype_raw)[0]
                
                # Skip value
                if dtype in (4, 5, 10):
                    skip(f, 4)
                elif dtype == 6:
                    skip(f, 4)
                elif dtype == 7:
                    skip(f, 8)
                elif dtype == 8:
                    slen = struct.unpack("<Q", f.read(8))[0]
                    skip(f, slen)
                elif dtype == 9:
                    elem_type = struct.unpack("<I", f.read(4))[0]
                    array_len = struct.unpack("<Q", f.read(8))[0]
                    if elem_type == 8:
                        total = 0
                        for _ in range(array_len):
                            sl = struct.unpack("<Q", f.read(8))[0]
                            total += sl
                        skip(f, total)
                    else:
                        sizes = {4:4, 5:4, 6:4, 7:8, 10:4, 11:8, 12:8}
                        skip(f, sizes.get(elem_type, 4) * array_len)
                elif dtype == 11:
                    skip(f, 8)
                elif dtype == 12:
                    skip(f, 8)
                else:
                    skip(f, 4)
                
                kv_count += 1
                last_kv_end = f.tell()
            
            print(f"  KV parsed: {kv_count}/{n_kv}")
            print(f"  Pos after KV: {last_kv_end} (0x{last_kv_end:X})")
            
            # Aligned position
            aligned = (last_kv_end + 31) & ~31
            print(f"  Aligned pos: {aligned} (0x{aligned:X})")
            
            # Hex dump around aligned position
            for offset in range(-128, 129, 16):
                pos = aligned + offset
                f.seek(pos)
                data = f.read(64)
                hex_str = ' '.join(f'{b:02x}' for b in data[:32])
                ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[:32])
                marker = " <<<" if offset == 0 else ""
                print(f"  {pos:12d} (0x{pos:08X}): {hex_str}{marker}")
                if offset == 0:
                    print(f"    ASCII: {ascii_str}")
            
            # Search for tensor names like "token_embd"
            print(f"\n  Searching for tensor names near aligned position...")
            f.seek(aligned)
            search_size = min(2_000_000, os.path.getsize(path) - aligned)
            chunk = f.read(search_size)
            
            for pattern in [b"token_embd", b"blk.0.attn_norm", b"output_norm", b"blk.0.ffn_gate"]:
                idx = chunk.find(pattern)
                if idx >= 0:
                    # Check 8 bytes before (nlen)
                    start = max(0, idx - 16)
                    end = min(len(chunk), idx + 64)
                    context = chunk[start:end]
                    print(f"    Found '{pattern.decode()}' at offset {idx} from aligned (file pos {aligned+idx})")
                    print(f"      Context (hex): {' '.join(f'{b:02x}' for b in context[:48])}")
                    
                    if idx >= 8:
                        nlen_candidate = struct.unpack("<Q", chunk[idx-8:idx])[0]
                        print(f"      nlen candidate (8 bytes before): {nlen_candidate}")
                    
                    # Try to parse tensor from idx-8
                    tpos = aligned + idx - 8
                    f.seek(tpos)
                    test = f.read(8)
                    if len(test) == 8:
                        nlen = struct.unpack("<Q", test)[0]
                        print(f"      nlen at {tpos}: {nlen}")
                        if 1 <= nlen <= 256:
                            name_data = f.read(nlen)
                            name = name_data.decode("utf-8", errors="replace")
                            print(f"      name: {name}")
                            nd = f.read(4)
                            if len(nd) == 4:
                                n_dims = struct.unpack("<I", nd)[0]
                                dims = []
                                for _ in range(n_dims):
                                    d = f.read(8)
                                    if len(d) == 8:
                                        dims.append(struct.unpack("<Q", d)[0])
                                gt = f.read(4)
                                if len(gt) == 4:
                                    ggml_type = struct.unpack("<I", gt)[0]
                                    off = f.read(8)
                                    if len(off) == 8:
                                        offset = struct.unpack("<Q", off)[0]
                                        print(f"      n_dims={n_dims}, dims={dims}, type={ggml_type}, offset=0x{offset:X}")
                                print()
                else:
                    print(f"    '{pattern.decode()}' not found in first {search_size} bytes from aligned")
        
        print()

if __name__ == "__main__":
    main()
