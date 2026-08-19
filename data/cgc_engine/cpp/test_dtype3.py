#!/usr/bin/env python3
"""Examine bytes around base_model.count."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    # First let's just get the exact position
    f.seek(24)
    
    # Re-read all KVs manually
    for i in range(13):
        pos = f.tell()
        klen = struct.unpack("<Q", f.read(8))[0]
        key = f.read(klen).decode()
        dtype = struct.unpack("<I", f.read(4))[0]
        
        if dtype == 8:
            slen = struct.unpack("<Q", f.read(8))[0]
            f.seek(slen, 1)
            v = "string"
        elif dtype == 5:
            v = struct.unpack("<I", f.read(4))[0]
        elif dtype == 6:
            v = struct.unpack("<f", f.read(4))[0]
        elif dtype == 4:
            # Read raw 4 bytes to check
            raw = f.read(4)
            v_u32 = struct.unpack("<I", raw)[0]
            v_u16 = struct.unpack("<H", raw[:2])[0]
            print(f"  dtype=4: raw={raw.hex()} u32={v_u32} u16={v_u16}")
            v = f"u32={v_u32}/u16={v_u16}"
        elif dtype == 7:
            v = struct.unpack("<Q", f.read(8))[0]
        elif dtype == 9:
            v = struct.unpack("<?", f.read(1))[0]
        elif dtype == 10:
            v = struct.unpack("<i", f.read(4))[0]
        elif dtype == 11:
            v = struct.unpack("<d", f.read(8))[0]
        elif dtype == 12:
            v = struct.unpack("<q", f.read(8))[0]
        
        print(f"KV[{i}]: pos={pos} '{key}' dtype={dtype} => {v}")
    
    pos_after = f.tell()
    print(f"\nAfter 13 KVs: pos={pos_after}")
    
    # Show next 64 bytes
    raw = f.read(64)
    print(f"Next 64 bytes: {raw.hex()}")
    print(f"As text: {raw}")
    
    # Check: what if dtype=4 is actually uint32 (4 bytes)?
    # Then base_model.count would read 4 bytes and we'd be at correct position
    # Let me rewind and try with uint32
    f.seek(pos_after - 64)  # back to before base_model.count's value
    
    # We are at pos where base_model.count's dtype starts
    # Let me re-read with correct assumption
    for test_name, test_func in [
        ("uint32(4bytes)", lambda: struct.unpack("<I", f.read(4))[0]),
        ("uint16(2bytes)", lambda: struct.unpack("<H", f.read(2))[0]),
    ]:
        f.seek(pos_after - 4)  # back to dtype of base_model.count
        klen = struct.unpack("<Q", f.read(8))[0]
        key = f.read(klen).decode()
        dtype = struct.unpack("<I", f.read(4))[0]
        print(f"\nTest: {test_name}")
        print(f"  key='{key}', dtype={dtype}")
        
        f.seek(pos_after)  # reset to after dtype
        val = test_func()
        pos_after_val = f.tell()
        
        # Next KV should start with klen
        next_klen_raw = f.read(8)
        next_klen = struct.unpack("<Q", next_klen_raw)[0]
        print(f"  val={val}, next pos={pos_after_val}")
        print(f"  next klen raw: {next_klen_raw.hex()}")
        print(f"  next klen: {next_klen}")
        
        if next_klen < 1000:
            # Likely a valid key length
            next_key = f.read(next_klen).decode("utf-8", errors="replace") if next_klen <= 256 else "?"
            print(f"  next key: '{next_key}'")
            next_dtype = struct.unpack("<I", f.read(4))[0]
            print(f"  next dtype: {next_dtype}")
        else:
            print(f"  INVALID next klen, this interpretation is WRONG")