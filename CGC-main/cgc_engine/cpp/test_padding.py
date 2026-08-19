#!/usr/bin/env python3
"""Exact byte-level analysis of the area after add_bos_token."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    # We know KV[46] starts at pos 10935305
    # Read the FULL KV[46] + following bytes
    f.seek(10935305)
    
    # KV[46]: klen(8) + key(28) + dtype(4) + value(?)
    raw = f.read(200)
    
    print("Full 200 bytes from pos 10935305:")
    for i in range(0, min(200, len(raw)), 16):
        hex_str = raw[i:i+16].hex()
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in raw[i:i+16])
        print(f"  {i:4d}: {hex_str}")
        print(f"         {ascii_str}")
    
    # Parse KV[46]
    klen = struct.unpack("<Q", raw[0:8])[0]
    key = raw[8:8+klen].decode("utf-8")
    dtype = struct.unpack("<I", raw[8+klen:8+klen+4])[0]
    val_start = 8 + klen + 4
    
    print(f"\nKV[46]: klen={klen}, key='{key}', dtype={dtype}")
    print(f"  Value starts at offset {val_start}")
    print(f"  Value bytes (16): {raw[val_start:val_start+16].hex()}")
    
    # Test different value sizes
    for val_size, val_name in [(1, "bool"), (4, "uint32"), (8, "uint64")]:
        val_end = val_start + val_size
        next_klen_raw = raw[val_end:val_end+8]
        next_klen = struct.unpack("<Q", next_klen_raw)[0]
        print(f"  If {val_name} ({val_size}B): next klen={next_klen}, raw={next_klen_raw.hex()}")
        if 3 <= next_klen <= 65536:
            next_key = raw[val_end+8:val_end+8+next_klen].decode("utf-8", errors="replace")
            print(f"    => next key would be: '{next_key}'")
    
    # Also check: what if the padding is part of the value?
    # What if dtype=7 but only uses 1 byte (like a bool)?
    # Then the extra 7 bytes + 1 = 8 bytes padding total?
    print(f"\n  If dtype=7 but only 1 byte used (bool):")
    val = raw[val_start]
    print(f"    Value byte: {val}")
    next_start = val_start + 1
    next_klen = struct.unpack("<Q", raw[next_start:next_start+8])[0]
    print(f"    Next klen at {next_start}: {next_klen}")
    if 3 <= next_klen <= 65536:
        next_key = raw[next_start+8:next_start+8+next_klen].decode("utf-8", errors="replace")
        next_dtype = struct.unpack("<I", raw[next_start+8+next_klen:next_start+8+next_klen+4])[0]
        print(f"    Next key: '{next_key}' dtype={next_dtype}")
    
    # Check padding between KVs
    # GGUF spec says entries are not padded, but some implementations add padding
    # Let's check if there's a pattern
    print(f"\n=== Checking for padding patterns ===")
    
    # Read KV[0] to KV[1] transition
    f.seek(24)  # After header
    for i in range(5):
        start = f.tell()
        klen = struct.unpack("<Q", f.read(8))[0]
        key = f.read(klen).decode("utf-8", errors="replace")
        dtype = struct.unpack("<I", f.read(4))[0]
        if dtype == 7:
            val_bytes = f.read(8)
        elif dtype in (4, 5, 10):
            val_bytes = f.read(4)
        elif dtype == 6:
            val_bytes = f.read(4)
        elif dtype == 8:
            slen = struct.unpack("<Q", f.read(8))[0]
            val_bytes = f.read(min(slen, 100))  # Just peek
        elif dtype == 9:
            # Skip array
            elem_type = struct.unpack("<I", f.read(4))[0]
            array_len = struct.unpack("<Q", f.read(8))[0]
            val_bytes = f"<array[{array_len}]>".encode()
        else:
            val_bytes = b"?"
        
        after_val = f.tell()
        total_entry = after_val - start
        # Check alignment
        aligned = (start + total_entry + 31) & ~31
        print(f"KV[{i}]: '{key}' dtype={dtype} entry_size={total_entry} aligned_to_32={aligned - start}")
        
        # Now check if next entry starts immediately or with padding
        next_start_raw = f.read(8)
        next_start_pos = after_val
        next_klen_candidate = struct.unpack("<Q", next_start_raw)[0]
        if 3 <= next_klen_candidate <= 65536:
            print(f"  => Next entry starts at {next_start_pos}, no padding needed")
            # Go back to read this entry properly
            f.seek(next_start_pos)
        else:
            print(f"  => Next entry NOT valid at {next_start_pos}, checking for padding...")
            # Check what byte follows
            for pad in range(1, 33):
                f.seek(next_start_pos + pad)
                candidate = struct.unpack("<Q", f.read(8))[0]
                if 3 <= candidate <= 65536:
                    next_key = f.read(candidate).decode("utf-8", errors="replace")
                    print(f"  => Found next entry at {next_start_pos + pad} with {pad}B padding! key='{next_key}'")
                    f.seek(next_start_pos + pad)  # Reposition
                    break
            else:
                print(f"  => Could not find next entry with up to 32B padding")
                f.seek(next_start_pos)  # Go back to try again later