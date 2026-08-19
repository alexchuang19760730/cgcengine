#!/usr/bin/env python3
"""Test CORRECT way to extract values from gguf reader fields."""
import gguf
import numpy as np

path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
reader = gguf.GGUFReader(path)

def extract_field_value(field, key_name):
    """Extract the actual VALUE (not key) from a GGUF field.
    
    GGUF field.parts layout:
    - For scalar types (uint32, uint64, int32, float32, bool):
        parts = [value_array]
      BUT the field already groups key+value, so for uint32 key+uint32 value:
        parts = [key_name_len, key_name_bytes, value_array]
    
    - For string types (STRING):
        parts = [key_len, key_bytes, value_len, value_bytes]
      where value_bytes is the actual string content
    """
    parts = field.parts
    types = field.types  # List of GGUFValueType for each part
    
    if not parts:
        return None
    
    # Check if this is a string value type
    # field.types typically has types for the VALUE parts only
    # Looking at the actual structure:
    # For "general.architecture" (key) -> STRING value:
    #   parts = [memmap([20]), memmap(key_bytes), memmap([8]), memmap([9]), memmap(value_bytes)]
    #   The STRING type info tells us where the value bytes are
    
    # Let's use a simpler approach - count the key parts and value parts
    # The field has key + value, so we need to skip key parts
    
    # Actually, let me just look at the raw structure more carefully
    return parts


# Let me inspect the actual structure of each field type
print("Inspecting field structures in detail:")
print("=" * 80)

for key_name in [
    "GGUF.version",
    "GGUF.tensor_count",
    "general.architecture",
    "qwen35moe.hidden_size" if "qwen35moe.hidden_size" in reader.fields else "qwen35moe.embedding_length",
    "qwen35moe.intermediate_size" if "qwen35moe.intermediate_size" in reader.fields else "qwen35moe.expert_feed_forward_length",
]:
    if key_name in reader.fields:
        field = reader.fields[key_name]
        print(f"\n{key_name}:")
        print(f"  types: {field.types}")
        print(f"  parts count: {len(field.parts)}")
        for i, part in enumerate(field.parts):
            if hasattr(part, 'shape') and hasattr(part, 'dtype'):
                print(f"  parts[{i}]: shape={part.shape}, dtype={part.dtype}, min={part.min() if part.size > 0 else 'N/A'}, max={part.max() if part.size > 0 else 'N/A'}")
                if part.dtype == np.uint8 and part.size < 100:
                    print(f"    as text: {part.tobytes().decode('utf-8', errors='replace')}")
                elif part.size <= 10:
                    print(f"    values: {part}")
            else:
                print(f"  parts[{i}]: {type(part)}")

# Now let's use gguf's own conversion
print("\n\n" + "=" * 80)
print("Trying gguf.scalar_to_np and other methods:")

# Check what methods are available
for attr in dir(reader):
    if not attr.startswith('_'):
        print(f"  reader.{attr}")

print("\n  ReaderField methods:")
for attr in dir(reader.fields["GGUF.version"]):
    if not attr.startswith('_'):
        print(f"    field.{attr}")

# Let me try the get_field method or direct value extraction
print("\n\nManual extraction:")

# For uint32 fields:
field = reader.fields["GGUF.version"]
# parts structure: [value]
val = field.parts[0].flatten()[0]
print(f"GGUF.version = {val} (raw: {field.parts})")

# For string fields:
# Looking at "general.architecture":
# parts = [memmap([20], uint64), memmap([key_bytes], uint8), memmap([8], uint32), memmap([9], uint64), memmap([value_bytes], uint8)]
# The key is "general.architecture" (20 chars including null terminator maybe?)
# Wait, let me count: "general.architecture" = 20 chars! Yes!
# Then the value is "qwen35moe" = 10 chars... but parts[3] has shape [9]
# Maybe the string includes a null terminator?

field = reader.fields["general.architecture"]
print(f"\ngeneral.architecture parts breakdown:")
for i, part in enumerate(field.parts):
    raw = part.tobytes() if hasattr(part, 'tobytes') else bytes(part)
    print(f"  [{i}] len={len(raw)}, raw={raw[:50]}")

# So for string field, the pattern is:
# parts[0] = string length (uint64) for KEY
# parts[1] = key bytes
# parts[2] = string length (uint32 or uint64) for VALUE  
# parts[3] = value length (uint64)... wait this is confusing

# Let me check the gguf source code for GGUFReader
