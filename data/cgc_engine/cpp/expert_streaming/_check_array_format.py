from gguf import GGUFReader
import struct

gguf_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

reader = GGUFReader(gguf_path)

# Let's check the raw structure of general.tags
field = reader.fields["general.tags"]
print(f"Offset: {field.offset}")
print(f"Types: {field.types}")

# Read raw bytes
with open(gguf_path, 'rb') as f:
    f.seek(field.offset)
    
    # Read key
    key_len = struct.unpack('<Q', f.read(8))[0]
    key = f.read(key_len).decode('utf-8')
    print(f"Key: {key}")
    
    # Read value type
    val_type = struct.unpack('<I', f.read(4))[0]
    print(f"Value type: {val_type} (9=ARRAY)")
    
    # Now we're at the value
    pos = f.tell()
    print(f"Value starts at: {pos}")
    
    # Read array: first count then type
    count = struct.unpack('<Q', f.read(8))[0]
    elem_type = struct.unpack('<I', f.read(4))[0]
    print(f"Array (count-first): count={count}, elem_type={elem_type}")
    
    # Let's also try type-first
    f.seek(pos)
    elem_type2 = struct.unpack('<I', f.read(4))[0]
    count2 = struct.unpack('<Q', f.read(8))[0]
    print(f"Array (type-first): elem_type={elem_type2}, count={count2}")