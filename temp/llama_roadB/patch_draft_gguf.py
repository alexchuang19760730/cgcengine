#!/usr/bin/env python3
"""Patch abhinand draft GGUF: rename dflash.target_layer_ids -> dflash.target_layers
so llama.cpp master's dflash arch can load it."""
import sys, numpy as np
sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/gguf-py')
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGUFValueType

SRC = '/Users/alexchuang/Documents/flashkv0516/models/gguf/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf'
DST = '/Users/alexchuang/Documents/flashkv0516/models/gguf/Qwen3.6-35B-A3B-DFlash-Q4_K_M_fixed.gguf'

r = GGUFReader(SRC)
w = GGUFWriter(DST, 'dflash')

def field_val(field):
    """Extract a python value from a GGUFReader field."""
    if field.types[0] == GGUFValueType.STRING:
        return str(field.data[0])
    if field.types[0] == GGUFValueType.ARRAY:
        arr = []
        for x in field.data:
            x = x.decode() if isinstance(x, (bytes, np.bytes_)) else x
            arr.append(x)
        return arr
    # scalar numeric
    return field.data[()]

def vtype_of(field):
    return GGUFValueType(field.types[0])

n_renamed = 0
for name, field in r.fields.items():
    out_name = name
    if name == 'dflash.target_layer_ids':
        out_name = 'dflash.target_layers'
        n_renamed += 1
    val = field_val(field)
    t = vtype_of(field)
    sub = None
    if t == GGUFValueType.ARRAY:
        sub = GGUFValueType(field.types[1])
    w.add_key_value(out_name, val, t, sub)
print(f"metadata copied, renamed {n_renamed} key(s)")

for tensor in r.tensors:
    name = tensor.name
    dtype = tensor.tensor_type
    data = np.concatenate([tensor.parts[p] for p in tensor.data])
    w.add_tensor(name, data, raw_dtype=dtype)

print(f"tensors copied: {len(r.tensors)}")
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("written:", DST)
