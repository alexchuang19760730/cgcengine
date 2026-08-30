#!/usr/bin/env python3
"""Minimal GGUF reader: extract only blk.40 tensor offsets/sizes from fraQtl
without building full ReaderField objects (avoids OOM from tokenizer strings).

Usage:
  python3.13 extract_blk40_minimal.py fraQtl.gguf blk.40. > blk40_offsets.json
"""
import struct, json, sys

# GGUFValueType: UINT8=0,INT8=1,UINT16=2,INT16=3,UINT32=4,INT32=5,
# FLOAT32=6,BOOL=7,STRING=8,ARRAY=9,UINT64=10,INT64=11,FLOAT64=12
SCALAR_SIZES = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}


def skip_value(f, vtype, endian):
    if vtype in SCALAR_SIZES:
        f.seek(SCALAR_SIZES[vtype], 1)
    elif vtype == 8:  # STRING
        slen = struct.unpack(endian+'Q', f.read(8))[0]
        f.seek(slen, 1)
    elif vtype == 9:  # ARRAY
        elem_type = struct.unpack(endian+'I', f.read(4))[0]
        n = struct.unpack(endian+'Q', f.read(8))[0]
        for _ in range(n):
            skip_value(f, elem_type, endian)
    else:
        raise ValueError(f'unknown vtype {vtype}')


def read_gguf_tensor_info(path):
    f = open(path, 'rb')
    magic = struct.unpack('<I', f.read(4))[0]
    assert magic == 0x46554747, f'bad magic {magic:#x}'
    version = struct.unpack('<I', f.read(4))[0]
    # endianness: if low 16 bits are 0, it's swapped
    if version & 0xFFFF == 0:
        endian = '>'
        version = struct.unpack('>I', struct.pack('<I', version))[0]
    else:
        endian = '<'
    assert version in (2, 3), f'bad version {version}'

    n_tensors = struct.unpack(endian+'Q', f.read(8))[0]
    n_kv = struct.unpack(endian+'Q', f.read(8))[0]

    # alignment: default 32, may be overridden by general.alignment KV
    align = 32

    for _ in range(n_kv):
        key_len = struct.unpack(endian+'Q', f.read(8))[0]
        key = f.read(key_len).decode('utf-8', errors='replace')
        vtype = struct.unpack(endian+'I', f.read(4))[0]
        # Capture general.alignment
        if key == 'general.alignment' and vtype == 4:  # UINT32
            align = struct.unpack(endian+'I', f.read(4))[0]
        else:
            skip_value(f, vtype, endian)

    # Read tensor info
    tensors = []
    for _ in range(n_tensors):
        name_len = struct.unpack(endian+'Q', f.read(8))[0]
        name = f.read(name_len).decode('utf-8', errors='replace')
        n_dims = struct.unpack(endian+'I', f.read(4))[0]
        dims = [struct.unpack(endian+'Q', f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack(endian+'I', f.read(4))[0]
        offset = struct.unpack(endian+'Q', f.read(8))[0]
        tensors.append({'name': name, 'dims': dims, 'ttype': ttype, 'offset': offset})

    data_start = f.tell()
    # Align data_start to alignment boundary
    data_start = (data_start + align - 1) // align * align
    f.close()
    return tensors, data_start, endian, align


if __name__ == '__main__':
    path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else 'blk.40.'
    tensors, data_start, endian, align = read_gguf_tensor_info(path)

    # Import quant sizes from gguf lib
    import sys as _sys
    _sys.path.insert(0, '/opt/homebrew/lib/python3.13/site-packages')
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

    result = []
    for t in tensors:
        if not t['name'].startswith(prefix):
            continue
        bs, ts = GGML_QUANT_SIZES.get(GGMLQuantizationType(t['ttype']), (1, 1))
        n = 1
        for d in t['dims']:
            n *= d
        n_bytes = (n // bs) * ts
        abs_offset = data_start + t['offset']
        result.append({
            'name': t['name'],
            'dims': t['dims'],
            'ttype': t['ttype'],
            'abs_offset': abs_offset,
            'n_bytes': n_bytes,
        })
    print(json.dumps({'data_start': data_start, 'align': align, 'n_blk40': len(result), 'tensors': result}, indent=2))
