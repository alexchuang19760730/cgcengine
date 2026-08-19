#!/usr/bin/env python3
"""Verify string array reading - check if batch reading is correct."""
import struct
import os

def test_array_reading(filepath):
    f = open(filepath, "rb")
    
    # Skip to item 40 (tokenizer.ggml.tokens)
    # We know it starts at position 2027
    f.seek(2027)
    
    klen = struct.unpack("<Q", f.read(8))[0]
    key = f.read(klen).decode()
    dtype = struct.unpack("<I", f.read(4))[0]
    
    print(f"Key: {key}, dtype: {dtype}")
    
    # Read array header
    elem_type = struct.unpack("<I", f.read(4))[0]
    array_len = struct.unpack("<Q", f.read(8))[0]
    
    print(f"Array: elem_type={elem_type}, array_len={array_len}")
    
    # Read first 10 string lengths individually
    print(f"\nReading first 10 strings individually:")
    lengths_individual = []
    for i in range(10):
        sl = struct.unpack("<Q", f.read(8))[0]
        lengths_individual.append(sl)
        print(f"  [{i}] len={sl}")
        f.seek(sl, 1)  # Skip content
    
    # Now go back and read using batch method
    print(f"\nResetting and reading with batch method...")
    f.seek(2027 + 8 + klen + 4 + 4 + 8)  # Back to first string length
    
    # Read a chunk of 10 string lengths using batch method
    chunk = 10
    prefixes = f.read(chunk * 8)
    print(f"  Read {len(prefixes)} bytes")
    
    lengths_batch = []
    for i in range(chunk):
        sl = struct.unpack("<Q", prefixes[i * 8:(i + 1) * 8])[0]
        lengths_batch.append(sl)
        print(f"  [{i}] len={sl}")
    
    # Compare
    print(f"\nComparison:")
    for i in range(10):
        match = "✓" if lengths_individual[i] == lengths_batch[i] else "✗"
        print(f"  [{i}] individual={lengths_individual[i]:>20} batch={lengths_batch[i]:>20} {match}")
    
    # Now check if the issue is with larger chunk sizes
    # Let's try reading 248320 strings in batches and sum
    print(f"\nReading all {array_len} strings in batches...")
    total = 0
    remaining = array_len
    batch_num = 0
    while remaining > 0:
        batch_size = min(remaining, 100000)
        prefixes = f.read(batch_size * 8)
        if len(prefixes) < batch_size * 8:
            print(f"  ERROR: Short read at batch {batch_num}")
            break
        for i in range(batch_size):
            sl = struct.unpack("<Q", prefixes[i * 8:(i + 1) * 8])[0]
            total += sl
        remaining -= batch_size
        batch_num += 1
        if batch_num <= 5 or remaining == 0:
            print(f"  Batch {batch_num}: read {batch_size} strings, running total={total:,}")
    
    print(f"\nTotal string content size: {total:,} bytes ({total/1024/1024:.2f} MB)")
    
    f.close()

if __name__ == "__main__":
    path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
    test_array_reading(path)
