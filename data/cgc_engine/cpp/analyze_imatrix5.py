#!/usr/bin/env python3
"""Deep search for tensor data in GGUF files.
The aligned position shows FF32 data (expert weights). 
We need to find the tensor INFO section (names + offsets), which is
typically at the end of the KV section but BEFORE actual tensor data.
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

def main():
    for label, path in MODELS.items():
        if not os.path.exists(path):
            continue
        
        fsize = os.path.getsize(path)
        print(f"\n{'='*60}")
        print(f"MODEL: {label} ({fsize/1e9:.2f} GB)")
        print(f"{'='*60}")
        
        with open(path, "rb") as f:
            # Read header
            f.seek(0)
            magic = f.read(4)
            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            print(f"Version={version}, n_tensors={n_tensors}, n_kv={n_kv}")
            
            # Skip all KV entries
            kv_count = 0
            for _ in range(n_kv):
                klen_raw = f.read(8)
                if len(klen_raw) < 8:
                    break
                klen = struct.unpack("<Q", klen_raw)[0]
                if klen <= 65536:
                    skip(f, klen)
                else:
                    skip(f, klen)
                    dtype = struct.unpack("<I", f.read(4))[0]
                    if dtype in (4, 5, 6, 10, 11, 12):
                        skip(f, 4 if dtype in (4,5,6,10) else 8)
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
                            sizes = {4:4,5:4,6:4,7:8,10:4,11:8,12:8}
                            skip(f, sizes.get(elem_type, 4) * array_len)
                    kv_count += 1
                    continue
                
                # Normal entry: read dtype + value
                dtype = struct.unpack("<I", f.read(4))[0]
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
                        sizes = {4:4,5:4,6:4,7:8,10:4,11:8,12:8}
                        skip(f, sizes.get(elem_type, 4) * array_len)
                elif dtype == 11:
                    skip(f, 8)
                elif dtype == 12:
                    skip(f, 8)
                else:
                    skip(f, 4)
                kv_count += 1
            
            pos_after_kv = f.tell()
            print(f"KV parsed: {kv_count}/{n_kv}")
            print(f"Pos after KV: {pos_after_kv} (0x{pos_after_kv:X})")
            
            # The Q4_K_M file is small (0.27 GB), so let's read from aligned
            # and search for tensor name patterns
            aligned = (pos_after_kv + 31) & ~31
            
            if fsize < 1 * 1024**3:  # For small files
                print(f"\n  Reading and searching...")
                f.seek(aligned)
                data = f.read(min(2_000_000, fsize - aligned))
                
                for pattern in [b"token_embd", b"blk.0.attn_norm", b"output_norm"]:
                    idx = data.find(pattern)
                    if idx >= 0:
                        print(f"  Found '{pattern.decode()}' at +{idx} from aligned (file 0x{aligned+idx:X})")
                        # Show context
                        start = max(0, idx - 16)
                        end = min(len(data), idx + 64)
                        ctx = data[start:end]
                        hex_str = ' '.join(f'{b:02x}' for b in ctx[:48])
                        print(f"    Context: {hex_str}")
                        
                        # Check for valid nlen before
                        if idx >= 8:
                            nlen = struct.unpack("<Q", data[idx-8:idx])[0]
                            print(f"    nlen before: {nlen}")
                    else:
                        print(f"  '{pattern.decode()}' NOT FOUND in first {len(data)} bytes from aligned")
            else:
                # For large files, search a larger region
                print(f"\n  Large file ({fsize/1e9:.2f} GB), searching wider area...")
                f.seek(aligned)
                # Search the first 50MB after aligned for tensor names
                search_bytes = min(50 * 1024 * 1024, fsize - aligned)
                data = f.read(search_bytes)
                
                for pattern in [b"token_embd", b"blk.0.attn_norm", b"output_norm", b"blk.0.ffn_gate_exps"]:
                    idx = data.find(pattern)
                    if idx >= 0:
                        print(f"  Found '{pattern.decode()}' at +{idx} from aligned (file 0x{aligned+idx:X})")
                        start = max(0, idx - 16)
                        end = min(len(data), idx + 64)
                        ctx = data[start:end]
                        hex_str = ' '.join(f'{b:02x}' for b in ctx[:48])
                        print(f"    Context: {hex_str}")
                        
                        if idx >= 8:
                            nlen = struct.unpack("<Q", data[idx-8:idx])[0]
                            print(f"    nlen before: {nlen}")
                        
                        # Try to parse a tensor from this position
                        tpos = aligned + idx - 8
                        print(f"\n    Trying to parse tensor at {tpos} (0x{tpos:X}):")
                        f.seek(tpos)
                        try:
                            nlen = struct.unpack("<Q", f.read(8))[0]
                            name = f.read(nlen).decode("utf-8", errors="replace")
                            print(f"    name: {name}")
                            nd = f.read(4)
                            n_dims = struct.unpack("<I", nd)[0]
                            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                            gt = f.read(4)
                            ggml_type = struct.unpack("<I", gt)[0]
                            off = f.read(8)
                            offset = struct.unpack("<Q", off)[0]
                            print(f"    n_dims={n_dims}, dims={dims}, type={ggml_type}, offset=0x{offset:X}")
                        except Exception as e:
                            print(f"    Error: {e}")
                    else:
                        print(f"  '{pattern.decode()}' NOT FOUND in {search_bytes/(1024*1024):.0f}MB from aligned")

if __name__ == "__main__":
    main()
