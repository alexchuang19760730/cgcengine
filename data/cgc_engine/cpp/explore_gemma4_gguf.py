#!/usr/bin/env python3
"""
Explore Gemma 4 GGUF structure
"""
import gguf
import os

model_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

print("Loading GGUF file...")
reader = gguf.GGUFReader(model_path)

# 列出所有 KV 字段
print("\n=== KV FIELDS ===")
for name, field in reader.fields.items():
    if name.startswith("GGUF."):
        continue
    try:
        val = field.contents()
        if isinstance(val, (bytes, bytearray)):
            val = f"<bytes len={len(val)}>"
        elif isinstance(val, list) and len(val) > 10:
            val = f"<list len={len(val)}>"
        print(f"  {name}: {val}")
    except Exception as e:
        print(f"  {name}: <error: {e}>")

# 列出张量名称 (仅前 30 个)
print("\n=== TENSOR NAMES (first 50) ===")
for i, t in enumerate(reader.tensors[:50]):
    print(f"  {i}: {t.name} (dims={list(t.shape)}, type={t.tensor_type.name}, size={int(t.data_offset)})")

# 搜索专家张量
print("\n=== EXPERT-RELATED TENSORS ===")
expert_count = 0
for t in reader.tensors:
    name = t.name
    if "expert" in name.lower():
        print(f"  {name} (dims={list(t.shape)}, type={t.tensor_type.name})")
        expert_count += 1
        if expert_count > 50:
            print(f"  ... and more")
            break

# 搜索 block 张量
print("\n=== BLOCK TENSORS ===")
block_count = 0
block_layers = set()
for t in reader.tensors:
    name = t.name
    if name.startswith("blk."):
        parts = name.split(".")
        if len(parts) >= 2:
            try:
                layer = int(parts[1])
                block_layers.add(layer)
            except ValueError:
                pass
        block_count += 1
        if block_count <= 10:
            print(f"  {name} (dims={list(t.shape)}, type={t.tensor_type.name})")

print(f"\nTotal block tensors: {block_count}")
print(f"Block layers: {sorted(block_layers)}")

# 统计各类型张量
print("\n=== TENSOR TYPE DISTRIBUTION ===")
type_counts = {}
for t in reader.tensors:
    type_name = t.tensor_type.name
    type_counts[type_name] = type_counts.get(type_name, 0) + 1
for type_name, count in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {type_name}: {count}")
