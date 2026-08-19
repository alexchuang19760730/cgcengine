#!/usr/bin/env python3
"""Determine the exact dtype mapping by analyzing known values."""

import struct

filepath = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

with open(filepath, "rb") as f:
    f.seek(24)
    
    # Known expected types for qwen35moe GGUF v3:
    expected = {
        "general.architecture": "string",
        "general.type": "string",
        "general.sampling.top_k": "uint32",
        "general.sampling.top_p": "float32",
        "general.sampling.temp": "float32",
        "general.name": "string",
        "general.basename": "string",
        "general.quantized_by": "string",
        "general.size_label": "string",
        "general.license": "string",
        "general.license.link": "string",
        "general.repo_url": "string",
        "general.base_model.count": "uint32",
        "general.base_model.0.name": "string",
        "general.base_model.0.organization": "string",
        "general.base_model.0.repo_url": "string",
        "general.tags": "string_array",
        "qwen35moe.block_count": "uint32",
        "qwen35moe.context_length": "uint32",
        "qwen35moe.embedding_length": "uint32",
        "qwen35moe.attention.head_count": "uint32",
        "qwen35moe.attention.head_count_kv": "uint32",
        "qwen35moe.rope.dimension_count": "uint32",
        "qwen35moe.rope.freq_base": "float64",
        "qwen35moe.attention.layer_norm_rms_epsilon": "float32",
        "qwen35moe.expert_count": "uint32",
        "qwen35moe.expert_used_count": "uint32",
        "qwen35moe.attention.key_length": "uint32",
        "qwen35moe.attention.value_length": "uint32",
        "qwen35moe.expert_feed_forward_length": "uint32",
        "qwen35moe.expert_shared_feed_forward_length": "uint32",
        "tokenizer.ggml.model": "string",
        "tokenizer.ggml.pre": "string",
        "tokenizer.ggml.tokens": "string_array",
    }
    
    dtype_to_type = {}  # dtype value -> type name
    
    for i in range(54):
        pos = f.tell()
        klen_raw = f.read(8)
        if len(klen_raw) < 8:
            break
        klen = struct.unpack("<Q", klen_raw)[0]
        
        if klen > 1048576:
            print(f"KV[{i}]: pos={pos} klen={klen} LARGE, skipping")
            try:
                f.seek(klen)
            except:
                _ = f.read(klen)
            # Read dtype and value too
            dtype = struct.unpack("<I", f.read(4))[0]
            if dtype == 8:  # string
                slen = struct.unpack("<Q", f.read(8))[0]
                try: f.seek(slen)
                except: _ = f.read(slen)
            elif dtype in (4, 5, 10):
                f.read(4)
            elif dtype in (7, 11):
                f.read(8)
            elif dtype == 9:
                f.read(1)
            elif dtype == 6:
                f.read(2)
            continue
        
        if klen == 0:
            key = ""
        else:
            key = f.read(klen).decode("utf-8", errors="replace")
        
        dtype = struct.unpack("<I", f.read(4))[0]
        
        expected_type = expected.get(key, "unknown")
        
        # Now read value
        if dtype == 8:  # STRING (from first 2 items)
            slen = struct.unpack("<Q", f.read(8))[0]
            if slen > 1048576:
                try: f.seek(slen)
                except: _ = f.read(slen)
                val = f"<large string>"
            else:
                val = f.read(slen).decode("utf-8", errors="replace") if slen > 0 else ""
        elif dtype == 5:  # Maybe uint32 or float32
            val_bytes = f.read(4)
            val_u32 = struct.unpack("<I", val_bytes)[0]
            val_f32 = struct.unpack("<f", val_bytes)[0]
            val = f"u32={val_u32} f32={val_f32}"
        elif dtype == 7:  # uint64 or int64
            val_bytes = f.read(8)
            val_u64 = struct.unpack("<Q", val_bytes)[0]
            val_f64 = struct.unpack("<d", val_bytes)[0]
            val = f"u64={val_u64} f64={val_f64}"
        elif dtype == 4:  # Maybe float32
            val_bytes = f.read(4)
            val_f32 = struct.unpack("<f", val_bytes)[0]
            val_u32 = struct.unpack("<I", val_bytes)[0]
            val = f"u32={val_u32} f32={val_f32}"
        elif dtype == 9:  # bool or string_array
            val_bytes = f.read(1)
            val_b = struct.unpack("<?", val_bytes)[0]
            val = f"bool={val_b}"
        elif dtype == 6:
            val_bytes = f.read(2)
            val_u16 = struct.unpack("<H", val_bytes)[0]
            val = f"u16={val_u16}"
        elif dtype == 10:
            val_bytes = f.read(4)
            val_i32 = struct.unpack("<i", val_bytes)[0]
            val_f32 = struct.unpack("<f", val_bytes)[0]
            val = f"i32={val_i32} f32={val_f32}"
        elif dtype == 11:
            val_bytes = f.read(8)
            val_f64 = struct.unpack("<d", val_bytes)[0]
            val_u64 = struct.unpack("<Q", val_bytes)[0]
            val = f"f64={val_f64} u64={val_u64}"
        else:
            val = f"<unknown dtype={dtype}>"
            break
        
        print(f"KV[{i}]: '{key}' dtype={dtype} expected={expected_type} => {val}")
        
        if dtype not in dtype_to_type:
            dtype_to_type[dtype] = set()
        dtype_to_type[dtype].add(expected_type)
    
    print(f"\n=== Dtype mapping summary ===")
    for dtype, types in sorted(dtype_to_type.items()):
        print(f"  dtype={dtype}: {types}")