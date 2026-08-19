#!/usr/bin/env python3
"""Test how to extract actual values from gguf reader fields - CORRECT WAY."""
import gguf
import numpy as np

path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
reader = gguf.GGUFReader(path)

def extract_field_value(field):
    """Extract the actual value from a GGUF field."""
    # field.parts is a list of numpy arrays representing the parsed value
    # For scalars (uint32, uint64, int32, float32, etc.), parts[0] contains the value
    # For strings, parts is structured differently
    
    if not field.parts:
        return None
    
    # Check if it's a string type (STRING = 8)
    if field.types and field.types[0].value == 8:
        # String parts: [len_array, string_bytes_array, ...]
        # The string bytes are typically the last element
        for part in field.parts:
            if hasattr(part, 'dtype') and part.dtype == np.uint8:
                try:
                    return part.tobytes().decode('utf-8', errors='replace')
                except:
                    pass
        return None
    
    # For numeric scalars, the first part contains the value
    parts = field.parts
    if len(parts) > 0:
        val_array = parts[0]
        if hasattr(val_array, 'item'):
            try:
                return val_array.item()
            except:
                pass
        if hasattr(val_array, 'tolist'):
            try:
                lst = val_array.tolist()
                if isinstance(lst, (int, float)):
                    return lst
                return lst
            except:
                pass
    
    return parts[0] if parts else None


# Test key fields
test_keys = [
    "GGUF.version",
    "GGUF.tensor_count", 
    "GGUF.kv_count",
    "general.architecture",
    "general.type",
    "general.name",
    "qwen35moe.hidden_size",
    "qwen35moe.intermediate_size",
    "qwen35moe.expert_count",
    "qwen35moe.block_count",
    "qwen35moe.attention.head_count",
]

print("Extracting values from GGUF fields:")
print("=" * 60)
for key in test_keys:
    if key in reader.fields:
        field = reader.fields[key]
        val = extract_field_value(field)
        print(f"{key}: {val} (type={type(val).__name__})")
    else:
        print(f"{key}: NOT FOUND")

# List all available keys
print(f"\n\nAll KV keys (excluding GGUF.*):")
for name in reader.fields.keys():
    if not name.startswith("GGUF."):
        field = reader.fields[name]
        val = extract_field_value(field)
        val_str = str(val)
        if len(val_str) > 80:
            val_str = val_str[:80] + "..."
        print(f"  {name}: {val_str}")
