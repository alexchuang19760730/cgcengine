from gguf import GGUFReader

gguf_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

reader = GGUFReader(gguf_path)

print("All KV fields with their offsets:")
print("-" * 60)
for key in sorted(reader.fields.keys(), key=lambda k: reader.fields[k].offset):
    field = reader.fields[key]
    offset = field.offset
    typ = field.types
    print(f"  {offset:6d}: {key} (type={typ})")