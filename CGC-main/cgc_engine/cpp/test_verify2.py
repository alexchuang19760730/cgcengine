#!/usr/bin/env python3
"""Test parse_gguf_header with debug."""

import sys
sys.path.insert(0, "D:/alex/flashkv0516/app/edge_engine")
from llama_monkey_patch import parse_gguf_header

model_path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
header = parse_gguf_header(model_path)

print(f"KV items read: {len(header['kv'])}")
print(f"Tensor items read: {len(header['tensors'])}")

if header['tensors']:
    print(f"\nFirst 5 tensors:")
    for t in header['tensors'][:5]:
        print(f"  {t['name']}: dims={t['dims']} type={t['type']} offset={t['offset']}")
    
    # Check for expert tensors
    expert_tensors = [t for t in header['tensors'] if 'expert' in t['name'].lower()]
    print(f"\nExpert tensors: {len(expert_tensors)}")
    if expert_tensors:
        for t in expert_tensors[:10]:
            print(f"  {t['name']}: dims={t['dims']}")
else:
    print("\nNo tensors parsed! Checking what went wrong...")
    import struct
    with open(model_path, "rb") as f:
        f.seek(24)  # skip header
        
        # Parse all KVs
        for i in range(54):
            pos = f.tell()
            klen_raw = f.read(8)
            if len(klen_raw) < 8:
                print(f"KV[{i}]: EOF at pos {pos}")
                break
            klen = struct.unpack("<Q", klen_raw)[0]
            
            if klen > 65536:
                print(f"KV[{i}]: pos={pos} klen={klen} LARGE, skipping")
                try: f.seek(klen)
                except: _ = f.read(klen)
                dtype = struct.unpack("<I", f.read(4))[0]
                if dtype in (4,5,6,10):
                    f.seek(4, 1)
                elif dtype in (7,9,11,12):
                    f.seek(8, 1)
                elif dtype == 8:
                    slen = struct.unpack("<Q", f.read(8))[0]
                    try: f.seek(slen, 1)
                    except: pass
                print(f"  -> Skipped, pos now: {f.tell()}")
                continue
            
            key = f.read(klen).decode("utf-8", errors="replace")
            dtype = struct.unpack("<I", f.read(4))[0]
            
            if dtype in (4, 5):
                f.read(4)
            elif dtype == 6:
                f.read(4)
            elif dtype == 7:
                f.read(8)
            elif dtype == 8:
                slen = struct.unpack("<Q", f.read(8))[0]
                if slen <= 1048576:
                    f.read(slen)
                else:
                    try: f.seek(slen, 1)
                    except: pass
            elif dtype == 9:
                f.read(8)
            elif dtype == 10:
                f.read(4)
            elif dtype == 11:
                f.read(8)
            elif dtype == 12:
                f.read(8)
            else:
                print(f"KV[{i}]: UNKNOWN dtype={dtype} for key '{key}'")
                break
            
            if i >= 16:  # Show items after tags
                print(f"KV[{i}]: pos={pos} '{key}' dtype={dtype}")
        
        print(f"\nAfter all KVs: pos={f.tell()}")
        
        # Try to read first tensor
        nlen_raw = f.read(8)
        if len(nlen_raw) < 8:
            print("EOF at tensor start")
        else:
            nlen = struct.unpack("<Q", nlen_raw)[0]
            print(f"First tensor nlen: {nlen}")
            if nlen <= 65536:
                name = f.read(nlen).decode("utf-8", errors="replace")
                print(f"First tensor: {name}")
            else:
                print("MISALIGNED!")
                raw = f.read(64)
                print(f"Raw at pos {f.tell()-8}: {raw.hex()}")