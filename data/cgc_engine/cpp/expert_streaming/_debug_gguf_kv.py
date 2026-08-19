import sys
sys.path.insert(0, r"C:\Users\alexchuang\AppData\Local\Programs\Python\Python312\Lib\site-packages")

from gguf import GGUFReader

gguf_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

reader = GGUFReader(gguf_path)

print("=== GGUF KV Metadata ===")
print(f"Total KV pairs: {len(reader.fields)}")
print()

for i, (key, field) in enumerate(reader.fields.items()):
    if i < 20:
        try:
            val = field.contents()
            if isinstance(val, str) and len(val) > 50:
                val = val[:50] + "..."
            elif isinstance(val, list) and len(val) > 10:
                val = val[:3] + [f"... ({len(val)} items)"]
            print(f"  [{i:3d}] {key}: {type(val).__name__} = {val}")
        except Exception as e:
            print(f"  [{i:3d}] {key}: ERROR - {e}")