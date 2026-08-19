import sys
import os
import struct
import ctypes

sys.path.insert(0, r"D:\alex\flashkv0516\CGC-main\cgc_engine\cpp\expert_streaming")

from gguf import GGUFReader

gguf_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

print("Loading GGUF with official library...")
reader = GGUFReader(gguf_path)

expert_count = reader.fields['gemma4.expert_count'].contents()
expert_feed_forward_length = reader.fields['gemma4.expert_feed_forward_length'].contents()
expert_used_count = reader.fields['gemma4.expert_used_count'].contents()
hidden_size = reader.fields['gemma4.embedding_length'].contents()
block_count = reader.fields['gemma4.block_count'].contents()

print(f"gemma4.expert_count: {expert_count}")
print(f"gemma4.expert_feed_forward_length: {expert_feed_forward_length}")
print(f"gemma4.expert_used_count: {expert_used_count}")
print(f"gemma4.embedding_length: {hidden_size}")
print(f"gemma4.block_count: {block_count}")

print("\n--- Searching for expert tensors ---")
expert_info = []

tensor_list = reader.fields.get('GGUF.tensor_count', None)
print(f"Tensor count field: {tensor_list}")

import gguf
print(f"\nGGUFReader methods: {[m for m in dir(gguf.GGUFReader) if not m.startswith('_')]}")

try:
    tnames = reader.tensor_names
    print(f"tensor_names: {tnames[:5]}...")
except:
    try:
        tlist = reader.tensors
        print(f"tensors type: {type(tlist)}")
        if hasattr(tlist, 'keys'):
            tnames = list(tlist.keys())
            print(f"tensor names from dict: {tnames[:5]}...")
        elif isinstance(tlist, list):
            tnames = [t.name if hasattr(t, 'name') else str(t) for t in tlist[:10]]
            print(f"tensor names from list: {tnames}")
    except Exception as e:
        print(f"Error: {e}")
