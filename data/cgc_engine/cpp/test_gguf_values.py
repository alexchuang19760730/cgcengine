#!/usr/bin/env python3
"""Test how to extract actual values from gguf reader fields."""
import gguf
import numpy as np

path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
reader = gguf.GGUFReader(path)

# Check how to extract actual values
print("Testing value extraction from GGUFReader fields:")
print("=" * 60)

# Test 1: architecture (string)
field = reader.fields["general.architecture"]
print(f"\n1. general.architecture:")
print(f"   field.parts = {field.parts}")
print(f"   field.parts type = {type(field.parts)}")
# Try to get the string value
string_data = field.parts
if hasattr(string_data, 'tobytes'):
    raw_bytes = string_data.tobytes()
    print(f"   raw bytes = {raw_bytes}")
    print(f"   as string = {raw_bytes.decode('utf-8', errors='replace')}")

# Test 2: uint32 value
field = reader.fields["GGUF.version"]
print(f"\n2. GGUF.version:")
print(f"   field.parts = {field.parts}")
print(f"   field.parts dtype = {field.parts.dtype if hasattr(field.parts, 'dtype') else 'N/A'}")
# Try converting
val = int(field.parts[0]) if len(field.parts) > 0 else None
print(f"   as int = {val}")

# Test 3: hidden_size
# Find keys that contain integer values
print(f"\n3. Looking for architecture-specific keys...")
for name in list(reader.fields.keys())[:20]:
    if "hidden_size" in name or "intermediate" in name or "expert" in name:
        field = reader.fields[name]
        print(f"   {name}: parts={field.parts}, types={field.types}")

# Test 4: Use get_field method
print(f"\n4. Testing get_field:")
try:
    arch = reader.get_field("general.architecture")
    print(f"   get_field('general.architecture') = {arch}")
except Exception as e:
    print(f"   Error: {e}")

# Test 5: Check what data looks like for different dtypes
print(f"\n5. Field types for first 10 KV items:")
for i, (name, field) in enumerate(reader.fields.items()):
    if i >= 10:
        break
    print(f"   {name}: types={field.types}")

# Test 6: Check how gguf converts parts to actual values
print(f"\n6. Testing gguf.scalar conversion...")
from gguf import GGUFReader
# Check if there's a method to get actual values
print(f"   methods on reader: {[m for m in dir(reader) if not m.startswith('_')]}")
print(f"   methods on field: {[m for m in dir(field) if not m.startswith('_')]}")
