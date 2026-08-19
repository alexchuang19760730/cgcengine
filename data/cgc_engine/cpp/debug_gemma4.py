#!/usr/bin/env python3
"""Debug Gemma 4 model parsing."""
import gguf
import os

model_path = r"D:\alex\flashkv0516\models\gemma4_gguf\gemma-4-26B-A4B-it-UD-IQ3_S.gguf"

print(f"File size: {os.path.getsize(model_path):,} bytes")
print(f"File: {os.path.basename(model_path)}")

# Try to parse just the header without building tensors
import struct

with open(model_path, 'rb') as f:
    magic = struct.unpack('<I', f.read(4))[0]
    print(f"\nMagic: 0x{magic:08X}")
    
    version = struct.unpack('<I', f.read(4))[0]
    print(f"Version: {version}")
    
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    print(f"Tensor count: {n_tensors}")
    
    n_kv = struct.unpack('<Q', f.read(8))[0]
    print(f"KV count: {n_kv}")

# Try to manually find the problematic tensor
print(f"\n--- Trying alternative parsing ---")

# The issue is with a tensor that has shape (262144, 2310) 
# Let's check what tensor that is
try:
    reader = gguf.GGUFReader(model_path)
except ValueError as e:
    print(f"Error: {e}")
    
    # Try to find the problematic tensor
    error_msg = str(e)
    print(f"\nLooking for tensor with shape mismatch...")
    
    # The error says: cannot reshape array of size 487255660 into shape (262144,2310)
    # 262144 * 2310 = 605,552,640 elements
    # But data is only 487,255,660 bytes
    # This suggests the type's bytes per element calculation is wrong
    
    expected_elements = 262144 * 2310
    actual_bytes = 487255660
    print(f"  Expected elements: {expected_elements:,}")
    print(f"  Actual bytes: {actual_bytes:,}")
    print(f"  Bytes per element: {actual_bytes / expected_elements:.4f}")
    
    # This might be a type issue - let's check what type has this bpe
    # If type is correct, bpe should be a multiple of 0.5 or 1
    # 487255660 / (262144 * 2310) = 0.804...
    
    # Check if this could be a different type
    # For example, if the type was supposed to be IQ4_XS (bpe=1), 
    # then 262144 * 2310 * 1 = 605,552,640 bytes
    # But actual data is 487,255,660 bytes
    
    # This might be because the stored dims are wrong
    # Let's try to find what the correct dimensions should be
    # 487255660 / 2310 = 210,933 (approximately)
    # Or 487255660 / 262144 = 1,859 (approximately)

print(f"\n--- Checking file structure ---")
# Look at the first few tensor entries manually
with open(model_path, 'rb') as f:
    f.seek(4 + 4 + 8 + 8)  # Skip header
    
    # Read KV items to find tensor info start
    # This is complex, let's just check specific offsets
    
print("\nSuggestion: The model file might be corrupted or have a non-standard format.")
print("Try downloading a fresh copy or use a different quantization.")
