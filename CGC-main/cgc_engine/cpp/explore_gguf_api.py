#!/usr/bin/env python3
"""Explore gguf package API."""
import gguf
import os

path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
reader = gguf.GGUFReader(path)

# List all attributes
print("GGUFReader attributes:")
for attr in dir(reader):
    if not attr.startswith('_'):
        print(f"  {attr}")

print(f"\ntype(reader): {type(reader)}")

# Try to access common properties
print(f"\nAttempting to access properties...")
try:
    print(f"  version: {reader.version}")
except AttributeError as e:
    print(f"  version: {e}")

try:
    print(f"  tensor_count: {reader.tensor_count}")
except AttributeError as e:
    print(f"  tensor_count: {e}")

try:
    print(f"  kv_count: {reader.kv_count}")
except AttributeError as e:
    print(f"  kv_count: {e}")

# Try to iterate
print(f"\nListing all tensors:")
count = 0
for t in reader.tensors:
    if count < 5:
        print(f"  [{count}] {t}")
        if hasattr(t, 'name'):
            print(f"       name={t.name}")
        if hasattr(t, 'shape'):
            print(f"       shape={t.shape}")
        if hasattr(t, 'type'):
            print(f"       type={t.type}")
    count += 1
print(f"Total tensors: {count}")

# List KV items
print(f"\nListing KV items:")
kv_count = 0
for name, item in reader.fields.items():
    if kv_count < 10:
        print(f"  {name}: {item}")
    kv_count += 1
print(f"Total KV items: {kv_count}")
