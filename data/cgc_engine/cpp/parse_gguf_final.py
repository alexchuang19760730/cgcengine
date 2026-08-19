#!/usr/bin/env python3
"""
Final correct GGUF parser with:
- bool: 1-byte storage
- string: direct f.read() for full content
- array: f.seek() for large arrays to avoid I/O overhead
- Correct tensor info and data start calculation
"""
import struct
import os
import sys

GGML_TYPE_MAP = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 4: "q5_0", 5: "q5_1",
    6: "q8_0", 7: "tq1_0", 8: "tq2_0", 9: "iq4_nl", 10: "iq3_s",
    11: "iq2_xxs", 12: "iq2_xs", 13: "iq3_xxs", 14: "q4_k",
    15: "q5_k", 16: "q6_k", 17: "q5_k_m", 18: "q5_k_s", 19: "q4_k_m",
    20: "q4_k_s", 21: "q3_k_l", 22: "q3_k_m", 23: "q3_k_s",
    24: "q2_k", 25: "iq3_m", 26: "iq3_s", 27: "iq2_m", 28: "iq1_m",
    29: "iq4_xs", 30: "iq3_xs", 31: "iq2_xs",
}

BLOCK_SIZE = {
    0: 4, 1: 2, 2: 256, 3: 256, 4: 256, 5: 256, 6: 32,
    7: 2, 8: 3, 9: 6, 10: 6, 11: 8, 12: 6, 13: 8,
    14: 256, 15: 256, 16: 256, 17: 256, 18: 256, 19: 256,
    20: 256, 21: 256, 22: 256, 23: 256, 24: 256,
    25: 256, 26: 256, 27: 256, 28: 256, 29: 256, 30: 256, 31: 256,
}

BLOCK_BITS = {
    0: 32, 1: 16, 2: 4, 3: 4, 4: 5, 5: 5, 6: 8,
    7: 2, 8: 3, 9: 4, 10: 3, 11: 2, 12: 2, 13: 3,
    14: 4, 15: 5, 16: 6, 17: 5, 18: 5, 19: 4, 20: 4,
    21: 3, 22: 3, 23: 3, 24: 2, 25: 3, 26: 3, 27: 2, 28: 1,
    29: 4, 30: 3, 31: 2,
}


def parse_gguf(filepath):
    file_size = os.path.getsize(filepath)
    f = open(filepath, "rb")
    
    result = {
        "path": filepath,
        "size": file_size,
        "header": {},
        "kv": {},
        "kv_list": [],
        "tensors": [],
        "tensor_types": {},
        "expert_info": {},
    }
    
    # Header
    magic = struct.unpack("<I", f.read(4))[0]
    if magic != 0x46554747:
        print(f"Invalid magic: 0x{magic:X}")
        f.close()
        return None
    
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    
    result["header"] = {
        "version": version,
        "n_tensors": n_tensors,
        "n_kv": n_kv,
    }
    
    # Parse KV items
    for idx in range(n_kv):
        item_start = f.tell()
        
        klen_bytes = f.read(8)
        if len(klen_bytes) < 8:
            break
        klen = struct.unpack("<Q", klen_bytes)[0]
        
        if klen == 0 or klen > 1000000:
            break
        
        key_bytes = f.read(klen)
        if len(key_bytes) < klen:
            break
        key = key_bytes.decode("utf-8", errors="replace")
        
        dtype_bytes = f.read(4)
        if len(dtype_bytes) < 4:
            break
        dtype = struct.unpack("<I", dtype_bytes)[0]
        
        val_start = f.tell()
        
        if dtype == 7:  # bool: 1 byte
            v = f.read(1)
            if len(v) != 1:
                break
            val = v[0]
            val_desc = f"bool={v[0]}"
            
        elif dtype == 8:  # string
            slen_bytes = f.read(8)
            if len(slen_bytes) < 8:
                break
            slen = struct.unpack("<Q", slen_bytes)[0]
            s_bytes = f.read(slen)
            if len(s_bytes) < slen:
                break
            s = s_bytes.decode("utf-8", errors="replace")
            val = s
            val_desc = f"str[{slen}]"
            
        elif dtype == 9:  # array
            et_bytes = f.read(4)
            al_bytes = f.read(8)
            if len(et_bytes) < 4 or len(al_bytes) < 8:
                break
            elem_type = struct.unpack("<I", et_bytes)[0]
            array_len = struct.unpack("<Q", al_bytes)[0]
            
            if elem_type == 8:  # string array
                total_str_bytes = 0
                strings = []
                for i in range(array_len):
                    sl_bytes = f.read(8)
                    if len(sl_bytes) < 8:
                        break
                    sl = struct.unpack("<Q", sl_bytes)[0]
                    total_str_bytes += sl
                    s_bytes = f.read(sl)
                    if len(s_bytes) < sl:
                        break
                    if i < 3:
                        strings.append(s_bytes.decode("utf-8", errors="replace")[:60])
                
                array_total = array_len * 8 + total_str_bytes
                val_desc = f"str-array[{array_len}] total={array_total:,}"
                if strings:
                    val_desc += f" first3={strings}"
                    
            elif elem_type in (4, 5, 6, 10):
                elem_size = 8 if elem_type == 10 else 4
                total = elem_size * array_len
                f.seek(total, 1)
                val_desc = f"array[{array_len}] elem_type={elem_type} size={total:,}"
                
            elif elem_type == 7:
                total = 1 * array_len
                f.seek(total, 1)
                val_desc = f"bool-array[{array_len}] size={total:,}"
                
            else:
                total = 4 * array_len
                f.seek(total, 1)
                val_desc = f"array[{array_len}] elem_type={elem_type}"
            
            val = {"array_len": array_len, "elem_type": elem_type}
            
        elif dtype == 10:  # uint64
            v = f.read(8)
            if len(v) != 8:
                break
            val = struct.unpack("<Q", v)[0]
            val_desc = f"uint64={val}"
            
        elif dtype == 11:  # float64
            v = f.read(8)
            if len(v) != 8:
                break
            val = struct.unpack("<d", v)[0]
            val_desc = f"float64={val:.6g}"
            
        elif dtype == 12:  # int64
            v = f.read(8)
            if len(v) != 8:
                break
            val = struct.unpack("<q", v)[0]
            val_desc = f"int64={val}"
            
        else:
            sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4}
            s = sizes.get(dtype, 4)
            v = f.read(s)
            if len(v) != s:
                break
            
            if dtype == 0:
                val = v[0]
                val_desc = f"uint8={val}"
            elif dtype == 1:
                val = v[0]
                val_desc = f"int8={val}"
            elif dtype == 2:
                val = struct.unpack("<H", v)[0]
                val_desc = f"uint16={val}"
            elif dtype == 3:
                val = struct.unpack("<h", v)[0]
                val_desc = f"int16={val}"
            elif dtype == 4:
                val = struct.unpack("<I", v)[0]
                val_desc = f"uint32={val}"
            elif dtype == 5:
                val = struct.unpack("<i", v)[0]
                val_desc = f"int32={val}"
            elif dtype == 6:
                val = struct.unpack("<f", v)[0]
                val_desc = f"float32={val:.6g}"
            else:
                val = v
                val_desc = f"unknown_dtype[{dtype}]"
        
        result["kv"][key] = {"dtype": dtype, "value": val, "desc": val_desc}
        result["kv_list"].append({"key": key, "dtype": dtype, "value": val})
    
    kv_end = f.tell()
    result["kv_end"] = kv_end
    
    # Parse tensor info from KV end position
    f.seek(kv_end)
    
    for t_idx in range(n_tensors):
        nl_bytes = f.read(8)
        if len(nl_bytes) < 8:
            print(f"  ERROR: EOF at tensor {t_idx} name length")
            break
        nl = struct.unpack("<Q", nl_bytes)[0]
        if nl < 3 or nl > 256:
            print(f"  ERROR: Invalid name length {nl} at tensor {t_idx}")
            break
        
        tname_bytes = f.read(nl)
        if len(tname_bytes) < nl:
            break
        tname = tname_bytes.decode("utf-8", errors="replace")
        
        nd_bytes = f.read(4)
        if len(nd_bytes) < 4:
            break
        nd = struct.unpack("<I", nd_bytes)[0]
        if nd == 0 or nd > 5:
            break
        
        tdims = []
        for _ in range(nd):
            d_bytes = f.read(8)
            if len(d_bytes) < 8:
                break
            tdims.append(struct.unpack("<Q", d_bytes)[0])
        if len(tdims) != nd:
            break
        
        gt_bytes = f.read(4)
        if len(gt_bytes) < 4:
            break
        ggml_type = struct.unpack("<I", gt_bytes)[0]
        if ggml_type > 50:
            break
        
        off_bytes = f.read(8)
        if len(off_bytes) < 8:
            break
        offset = struct.unpack("<Q", off_bytes)[0]
        if offset > file_size:
            break
        
        tensor = {
            "name": tname,
            "dims": tdims,
            "type": ggml_type,
            "offset": offset,
        }
        result["tensors"].append(tensor)
        
        # Classify type
        result["tensor_types"][ggml_type] = result["tensor_types"].get(ggml_type, 0) + 1
        
        # Classify expert tensors
        if "expert" in tname or "exps" in tname:
            if "gate" in tname:
                role = "gate"
            elif "up" in tname:
                role = "up"
            elif "down" in tname:
                role = "down"
            elif "w" in tname:
                role = "w"
            else:
                role = "unknown"
            
            if role not in result["expert_info"]:
                result["expert_info"][role] = {"count": 0, "types": {}, "total_elements": 0}
            result["expert_info"][role]["count"] += 1
            result["expert_info"][role]["types"][ggml_type] = result["expert_info"][role]["types"].get(ggml_type, 0) + 1
            result["expert_info"][role]["total_elements"] += 1
    
    result["tensors_parsed"] = len(result["tensors"])
    
    # Calculate data start position
    if result["tensors"]:
        f.seek(kv_end)
        for t in result["tensors"]:
            f.seek(8 + len(t["name"]), 1)  # nlen + name
            nd = len(t["dims"])
            f.seek(4 + 8 * nd, 1)  # n_dims + dims
            f.seek(4 + 8, 1)  # ggml_type + offset
        
        info_end = f.tell()
        data_start = (info_end + 31) & ~31
        result["info_end"] = info_end
        result["data_start"] = data_start
    
    f.close()
    return result


def print_result(result):
    if result is None:
        print("Invalid GGUF file")
        return
    
    print(f"\n{'='*60}")
    print(f"GGUF Analysis: {os.path.basename(result['path'])}")
    print(f"{'='*60}")
    
    print(f"\n  File size: {result['size']:,} bytes ({result['size']/1024**3:.2f} GB)")
    print(f"  GGUF version: {result['header']['version']}")
    print(f"  Tensors: {result['tensors_parsed']}/{result['header']['n_tensors']}")
    print(f"  KV entries: {len(result['kv'])}/{result['header']['n_kv']}")
    
    arch = result["kv"].get("general.architecture", {}).get("value", "N/A")
    name_val = result["kv"].get("general.name", {}).get("value", "N/A")
    quant_by = result["kv"].get("general.quantized_by", {}).get("value", "N/A")
    print(f"  Architecture: {arch}")
    print(f"  Model name: {name_val}")
    print(f"  Quantized by: {quant_by}")
    
    if result.get("data_start"):
        print(f"\n  Data starts at: {result['data_start']:,} (0x{result['data_start']:X})")
    
    if result["tensors"]:
        print(f"\n  First 5 tensors:")
        for t in result["tensors"][:5]:
            type_name = GGML_TYPE_MAP.get(t["type"], f"type_{t['type']}")
            print(f"    '{t['name']}' dims={t['dims']} type={type_name} offset=0x{t['offset']:X}")
    
    print(f"\n  Tensor type distribution:")
    for tn in sorted(result["tensor_types"].keys()):
        count = result["tensor_types"][tn]
        type_name = GGML_TYPE_MAP.get(tn, f"type_{tn}")
        block_size = BLOCK_SIZE.get(tn, "?")
        block_bits = BLOCK_BITS.get(tn, "?")
        print(f"    {type_name:12s} (type={tn:2d}, block={str(block_size):>4s}, {str(block_bits)+'bit':>4s}): {count:4d}")
    
    if result["expert_info"]:
        print(f"\n  Expert tensor summary:")
        for role, info in sorted(result["expert_info"].items()):
            types_str = ", ".join(f"{GGML_TYPE_MAP.get(t, t)}={c}" for t, c in info["types"].items())
            print(f"    {role:10s}: {info['count']:4d} tensors, types: {types_str}")
    
    # Print some key KV values
    print(f"\n  Key KV metadata:")
    important_keys = [
        "general.architecture", "general.name", "general.size_label",
        "general.quantized_by", "qwen35moe.block_count",
        "qwen35moe.embedding_length", "qwen35moe.expert_count",
        "qwen35moe.attention.head_count", "qwen35moe.attention.head_count_kv",
    ]
    for key in important_keys:
        if key in result["kv"]:
            val = result["kv"][key]
            print(f"    {key}: {val.get('value', val.get('desc', 'N/A'))}")


if __name__ == "__main__":
    models = [
        r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
        r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
        r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf",
    ]
    
    for path in models:
        if not os.path.exists(path):
            print(f"SKIP: {path} not found")
            continue
        result = parse_gguf(path)
        print_result(result)
