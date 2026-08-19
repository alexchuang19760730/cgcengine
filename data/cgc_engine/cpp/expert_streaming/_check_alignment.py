import struct

gguf_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

with open(gguf_path, 'rb') as f:
    # general.tags starts at offset 804
    f.seek(804)
    
    # Read key
    key_len = struct.unpack('<Q', f.read(8))[0]
    key = f.read(key_len).decode('utf-8')
    print(f"Key: {key} (len={key_len})")
    
    # Read type (ARRAY = 9)
    typ = struct.unpack('<I', f.read(4))[0]
    print(f"Type: {typ} (ARRAY)")
    
    # Read array header
    n = struct.unpack('<Q', f.read(8))[0]
    elem_type = struct.unpack('<I', f.read(4))[0]
    print(f"Array: n={n}, elem_type={elem_type} (STRING=8)")
    
    # Now read each string element
    for i in range(n):
        pos = f.tell()
        slen = struct.unpack('<Q', f.read(8))[0]
        s = f.read(slen).decode('utf-8')
        after_read = f.tell()
        
        # Calculate alignment
        aligned = (after_read + 3) & ~3  # round up to 4 bytes
        pad = aligned - after_read
        
        print(f"  [{i}] pos={pos}: '{s}' (len={slen}), after_read={after_read}, pad={pad}, aligned={aligned}")
        
        # Skip padding
        f.seek(pad, 1)
    
    final_pos = f.tell()
    print(f"\nFinal position: {final_pos}")
    print(f"Expected (gemma4.block_count): 947")
    print(f"Difference: {final_pos - 947}")