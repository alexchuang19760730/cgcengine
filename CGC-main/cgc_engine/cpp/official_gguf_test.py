#!/usr/bin/env python3
"""
Use official gguf Python package (by llama.cpp team) to parse GGUF files.
This is the reference implementation that correctly handles all dtypes.
"""
import gguf
import os

def parse_with_official_gguf(filepath):
    """Parse GGUF file using official gguf library."""
    print(f"Loading: {os.path.basename(filepath)}")
    print(f"File size: {os.path.getsize(filepath):,} bytes")
    
    # Use gguf.GGUFReader to parse
    reader = gguf.GGUFReader(filepath)
    
    print(f"\nHeader:")
    print(f"  Version: {reader.version}")
    print(f"  Tensor count: {reader.tensor_count}")
    print(f"  KV count: {reader.kv_count}")
    
    print(f"\nKV items:")
    kv_count = 0
    for item in reader.fields.values():
        print(f"  {item.name}: dtype={item.part_type.name}, count={item.count}")
        kv_count += 1
        if kv_count >= 20:
            break
    
    print(f"\nAll KV keys:")
    for name in reader.fields.keys():
        print(f"  {name}")
    
    print(f"\nFirst 10 tensors:")
    for i, t in enumerate(reader.tensors):
        if i >= 10:
            break
        print(f"  [{i}] '{t.name}' dims={t.shape} type={t.type.name} offset={t.offset}")
    
    print(f"\nData start offset: {reader.data_offset}")
    
    # Check for expert tensors
    expert_tensors = []
    for t in reader.tensors:
        if "expert" in t.name or "exps" in t.name:
            expert_tensors.append(t)
    
    print(f"\nExpert tensors: {len(expert_tensors)}")
    for t in expert_tensors[:10]:
        print(f"  '{t.name}' dims={t.shape} type={t.type.name}")
    
    return reader

if __name__ == "__main__":
    models = [
        r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
    ]
    
    for path in models:
        if os.path.exists(path):
            reader = parse_with_official_gguf(path)
        else:
            print(f"Not found: {path}")
