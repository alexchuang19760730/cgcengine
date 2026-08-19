#!/usr/bin/env python3
"""
Targeted debug: dump raw bytes around the general.tags entry in IQ2_XXS.
We need to understand why the string array parsing consumes 10GB.
"""
import struct

path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"

with open(path, "rb") as f:
    # Read header
    magic = struct.unpack("<I", f.read(4))[0]
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    print(f"Version={version}, n_tensors={n_tensors}, n_kv={n_kv}")
    
    # Skip KV entries 0-15 (we know they work)
    for idx in range(16):
        klen = struct.unpack("<Q", f.read(8))[0]
        if klen <= 65536:
            f.read(klen)
        else:
            # Big key - skip full entry
            import os
            f.seek(0, 1)  # just a nop
            # Actually need to skip properly
            pass
        dtype = struct.unpack("<I", f.read(4))[0]
        # Skip value
        if dtype in (4, 5, 6, 10):
            f.read(4)
        elif dtype == 7:
            f.read(8)
        elif dtype == 8:
            slen = struct.unpack("<Q", f.read(8))[0]
            f.seek(slen, 1)
        elif dtype == 9:
            et = struct.unpack("<I", f.read(4))[0]
            al = struct.unpack("<Q", f.read(8))[0]
            if et == 8:
                total = 0
                for _ in range(al):
                    sl = struct.unpack("<Q", f.read(8))[0]
                    total += sl
                f.seek(total, 1)
            else:
                sizes = {4:4,5:4,6:4,7:8,10:4,11:8,12:8}
                f.seek(sizes.get(et, 4) * al, 1)
        elif dtype == 11:
            f.read(8)
        elif dtype == 12:
            f.read(8)
        else:
            f.read(4)
    
    # Now at KV[16] = general.tags
    pos_before_tags = f.tell()
    print(f"\nPosition before general.tags: {pos_before_tags} (0x{pos_before_tags:X})")
    
    # Read and show raw bytes for general.tags
    print(f"\n--- Raw bytes for general.tags ---")
    f.seek(pos_before_tags)
    
    # Read klen
    klen_bytes = f.read(8)
    klen = struct.unpack("<Q", klen_bytes)[0]
    print(f"klen = {klen} (0x{klen:X})")
    
    # Read key
    key_bytes = f.read(klen)
    key = key_bytes.decode("utf-8")
    print(f"key = '{key}'")
    
    # Read dtype
    dtype_bytes = f.read(4)
    dtype = struct.unpack("<I", dtype_bytes)[0]
    print(f"dtype = {dtype}")
    
    # Now dump raw bytes of the VALUE section
    value_start = f.tell()
    print(f"\nValue starts at: {value_start} (0x{value_start:X})")
    
    # Show first 100 bytes of value
    val_dump = f.read(100)
    hex_str = ' '.join(f'{b:02x}' for b in val_dump)
    print(f"First 100 bytes of value:")
    print(f"  {hex_str}")
    
    # Interpret: elem_type (4 bytes) + array_len (8 bytes)
    et = struct.unpack("<I", val_dump[0:4])[0]
    al = struct.unpack("<Q", val_dump[4:12])[0]
    print(f"\nelem_type = {et}")
    print(f"array_len = {al}")
    
    if et == 8:
        print(f"\nThis is a STRING array with {al} entries.")
        print(f"Reading string lengths...")
        
        total = 0
        for i in range(min(al, 10)):
            sl_pos = value_start + 12 + i * 8 + total
            # Read from actual file position
            f.seek(value_start + 12 + total)
            sl_bytes = f.read(8)
            if len(sl_bytes) < 8:
                print(f"  [{i}] EOF reading string length")
                break
            sl = struct.unpack("<Q", sl_bytes)[0]
            print(f"  [{i}] string[{i}] length = {sl} (0x{sl:X})")
            total += sl
        
        if al > 10:
            print(f"  ... ({al - 10} more strings)")
        
        print(f"\n  Calculated total bytes for all strings: {total}")
        print(f"  This would put us at position: {value_start + 12 + total}")
        
        # Now let's manually check what comes after general.tags
        # by reading the next few KV entries correctly
        print(f"\n--- Manually parsing next KV entries ---")
        
        # After general.tags value, the next thing should be a new KV entry
        # Let's try to find the correct position
        # The issue is: after general.tags with 3 strings, we should be at:
        # value_start + 12 (header) + sum_of_string_lengths
        
        # Let me try reading from value_start + 12 and see if we can find
        # valid string content
        print(f"\n  Trying to read strings directly...")
        f.seek(value_start + 12)
        
        for i in range(min(al, 5)):
            sl_bytes = f.read(8)
            if len(sl_bytes) < 8:
                print(f"  [{i}] EOF")
                break
            sl = struct.unpack("<Q", sl_bytes)[0]
            print(f"  [{i}] sl={sl}")
            
            # Show what those bytes look like
            if sl < 1000 and sl > 0:
                str_bytes = f.read(min(sl, 100))
                print(f"       content: {str_bytes.decode('utf-8', errors='replace')}")
                if sl > 100:
                    f.seek(sl - 100, 1)  # skip the rest
            else:
                print(f"       (skipping - too large or invalid)")
                # DON'T skip - this is likely wrong
                # Instead let's look for a valid KV entry marker
                break
        
        # Let's look for the next valid KV entry by searching for known patterns
        print(f"\n--- Searching for next KV entry ---")
        
        # Go to end of what should be general.tags
        # Try several positions
        for test_offset in [12 + 70, 12 + 80, 12 + 90, 12 + 100, 12 + 110, 12 + 120, 12 + 150, 12 + 200]:
            test_pos = value_start + test_offset
            f.seek(test_pos)
            # Read potential klen
            kc = f.read(8)
            if len(kc) == 8:
                kc_val = struct.unpack("<Q", kc)[0]
                if 5 <= kc_val <= 100:
                    # Read potential key
                    key_candidate = f.read(min(kc_val, 100))
                    if len(key_candidate) == kc_val:
                        try:
                            key_str = key_candidate.decode("utf-8")
                            if key_str.isalpha() or '.' in key_str:
                                print(f"  @{test_pos} (0x{test_pos:X}): klen={kc_val}, key='{key_str}'")
                                # Check what follows
                                dtype_byte = f.read(4)
                                if len(dtype_byte) == 4:
                                    d = struct.unpack("<I", dtype_byte)[0]
                                    print(f"    dtype={d}")
                        except:
                            pass

if __name__ == "__main__":
    pass  # already running above
