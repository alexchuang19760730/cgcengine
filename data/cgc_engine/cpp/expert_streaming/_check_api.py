import sys
from gguf import GGUFReader

gguf_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

reader = GGUFReader(gguf_path)

print(dir(reader))
print()
print(f"fields keys: {list(reader.fields.keys())[:5]}...")

import gguf
print(dir(gguf.GGUFReader))
