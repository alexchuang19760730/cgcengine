#!/usr/bin/env python3
"""
GGUF imatrix impact analyzer: compares KV metadata and tensor distribution
between non-imatrix (Q4_K_M) and imatrix-quantized (IQ2_XXS/IQ3_XXS) models.
"""
import struct
import os
import sys

DTYPE_MAP = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32",
    5: "int32", 6: "float32", 7: "bool", 8: "string", 9: "array",
    10: "uint64", 11: "float64", 12: "int64",
}

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


def big_skip(f, n):
    """Skip n bytes by reading chunks to avoid 32-bit seek limit."""
    CHUNK = 4 * 1024 * 1024
    while n > 0:
        d = f.read(min(CHUNK, n))
        if not d:
            break
        n -= len(d)


def read_kv_item(f):
    """Read a single KV item. Returns (key, dtype, value) or None."""
    pos = f.tell()
    
    klen_bytes = f.read(8)
    if len(klen_bytes) < 8:
        return None
    klen = struct.unpack("<Q", klen_bytes)[0]
    
    if klen == 0 or klen > 1000000:
        return None
    
    key_bytes = f.read(klen)
    if len(key_bytes) < klen:
        return None
    key = key_bytes.decode("utf-8", errors="replace")
    
    dtype_bytes = f.read(4)
    if len(dtype_bytes) < 4:
        return None
    dtype = struct.unpack("<I", dtype_bytes)[0]
    
    val = read_value(f, dtype, key)
    return (key, dtype, val)


def read_value(f, dtype, key_name):
    """Read value based on dtype. Returns a summary string."""
    if dtype == 0:
        v = f.read(1)
        return f"uint8[1]={v[0]}" if len(v) == 1 else "ERR"
    elif dtype == 1:
        v = f.read(1)
        return f"int8[1]={v[0]}" if len(v) == 1 else "ERR"
    elif dtype == 2:
        v = f.read(2)
        return f"uint16={struct.unpack('<H', v)[0]}" if len(v) == 2 else "ERR"
    elif dtype == 3:
        v = f.read(2)
        return f"int16={struct.unpack('<h', v)[0]}" if len(v) == 2 else "ERR"
    elif dtype in (4,):
        v = f.read(4)
        return f"uint32={struct.unpack('<I', v)[0]}" if len(v) == 4 else "ERR"
    elif dtype == 5:
        v = f.read(4)
        return f"int32={struct.unpack('<i', v)[0]}" if len(v) == 4 else "ERR"
    elif dtype == 6:
        v = f.read(4)
        return f"float32={struct.unpack('<f', v)[0]:.6g}" if len(v) == 4 else "ERR"
    elif dtype == 7:
        # bool: stored as int8 (1 byte) + 3 byte padding for 4-byte alignment
        v = f.read(1)
        pad = f.read(3)
        return f"bool={v[0]}" if len(v) == 1 else "ERR"
    elif dtype == 8:
        slen_bytes = f.read(8)
        if len(slen_bytes) < 8:
            return "ERR"
        slen = struct.unpack("<Q", slen_bytes)[0]
        # Read first 80 chars
        chunk = f.read(min(80, slen))
        big_skip(f, slen - len(chunk))
        s = chunk.decode("utf-8", errors="replace")
        if len(s) > 80:
            s = s[:80] + "..."
        return f"str[{slen}]='{s}'"
    elif dtype == 9:
        # array: element_type(4) + array_len(8)
        et_bytes = f.read(4)
        al_bytes = f.read(8)
        if len(et_bytes) < 4 or len(al_bytes) < 8:
            return "ERR"
        elem_type = struct.unpack("<I", et_bytes)[0]
        array_len = struct.unpack("<Q", al_bytes)[0]
        
        if elem_type == 8:  # string array
            str_data = []
            for i in range(array_len):
                sl_bytes = f.read(8)
                if len(sl_bytes) < 8:
                    break
                sl = struct.unpack("<Q", sl_bytes)[0]
                if i < 3:
                    chunk = f.read(min(60, sl))
                    s = chunk.decode("utf-8", errors="replace")
                    str_data.append(s[:60])
                    big_skip(f, sl - len(chunk))
                else:
                    big_skip(f, sl)
            return f"str-array[{array_len}] first3={str_data}"
        else:
            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            elem_size = sizes.get(elem_type, 4)
            total = elem_size * array_len
            big_skip(f, total)
            return f"array[{array_len}] elem_type={DTYPE_MAP.get(elem_type, elem_type)}"
    elif dtype == 10:
        v = f.read(8)
        return f"uint64={struct.unpack('<Q', v)[0]}" if len(v) == 8 else "ERR"
    elif dtype == 11:
        v = f.read(8)
        return f"float64={struct.unpack('<d', v)[0]:.6g}" if len(v) == 8 else "ERR"
    elif dtype == 12:
        v = f.read(8)
        return f"int64={struct.unpack('<q', v)[0]}" if len(v) == 8 else "ERR"
    else:
        v = f.read(4)
        return f"unknown_dtype[{dtype}]"


def try_parse_tensors_here(f, max_tensors):
    """Try to parse tensor info starting from current position."""
    tensors = []
    
    for idx in range(max_tensors):
        t_start = f.tell()
        
        nlen_bytes = f.read(8)
        if len(nlen_bytes) < 8:
            return tensors
        nlen = struct.unpack("<Q", nlen_bytes)[0]
        
        if nlen == 0 or nlen > 256:
            return tensors
        
        name_bytes = f.read(nlen)
        if len(name_bytes) < nlen:
            return tensors
        name = name_bytes.decode("utf-8", errors="replace")
        
        # Validate name
        if not any(kw in name for kw in ["blk.", "token_embd", "output", "norm", "rope", "tokenizer", "vision"]):
            if idx >= 3:
                return tensors
        
        nd_bytes = f.read(4)
        if len(nd_bytes) < 4:
            return tensors
        n_dims = struct.unpack("<I", nd_bytes)[0]
        if n_dims == 0 or n_dims > 5:
            return tensors
        
        dims = []
        for _ in range(n_dims):
            d_bytes = f.read(8)
            if len(d_bytes) < 8:
                return tensors
            dims.append(struct.unpack("<Q", d_bytes)[0])
        
        gt_bytes = f.read(4)
        if len(gt_bytes) < 4:
            return tensors
        ggml_type = struct.unpack("<I", gt_bytes)[0]
        if ggml_type > 50:
            return tensors
        
        off_bytes = f.read(8)
        if len(off_bytes) < 8:
            return tensors
        offset = struct.unpack("<Q", off_bytes)[0]
        
        if offset > 100000000000:  # > 100GB
            return tensors
        
        tensors.append({
            "name": name,
            "dims": dims,
            "type": ggml_type,
            "offset": offset,
        })
    
    return tensors


def find_tensor_info_start(f, kv_end, file_size, n_tensors):
    """Search for the correct tensor info start position."""
    candidates = []
    
    # Try 32-byte aligned
    aligned = (kv_end + 31) & ~31
    candidates.append(aligned)
    
    # Try offsets around the aligned position
    for delta in range(-512, 513, 8):
        candidates.append(aligned + delta)
    
    best_tensors = []
    best_pos = aligned
    best_score = 0
    
    for pos in candidates:
        if pos >= file_size - 100:
            continue
        f.seek(pos)
        tensors = try_parse_tensors_here(f, min(n_tensors, 15))
        score = len(tensors)
        if score > best_score:
            best_score = score
            best_tensors = tensors
            best_pos = pos
            if score >= 15:
                break
    
    return best_pos, best_tensors


def parse_gguf(filepath):
    """Parse a GGUF file and return metadata."""
    file_size = os.path.getsize(filepath)
    f = open(filepath, "rb")
    
    result = {
        "path": filepath,
        "size": file_size,
        "header": {},
        "kv": {},
        "tensors": [],
        "tensor_types": {},
        "expert_info": {},
    }
    
    # Header
    magic = struct.unpack("<I", f.read(4))[0]
    if magic != 0x46554747:
        f.close()
        return None
    
    result["header"]["version"] = struct.unpack("<I", f.read(4))[0]
    result["header"]["n_tensors"] = struct.unpack("<Q", f.read(8))[0]
    result["header"]["n_kv"] = struct.unpack("<Q", f.read(8))[0]
    
    # Parse KV entries
    parsed_count = 0
    for idx in range(result["header"]["n_kv"]):
        item = read_kv_item(f)
        if item is None:
            break
        key, dtype, val = item
        result["kv"][key] = {"dtype": dtype, "value": val}
        parsed_count += 1
    
    result["kv_parsed"] = parsed_count
    
    # Find tensor info start
    kv_end = f.tell()
    tensor_start, first_tensors = find_tensor_info_start(f, kv_end, file_size, result["header"]["n_tensors"])
    result["tensor_start"] = tensor_start
    result["first_tensors"] = first_tensors[:5]
    
    # Parse all tensors from the found position
    if first_tensors:
        f.seek(tensor_start)
        tensors = try_parse_tensors_here(f, result["header"]["n_tensors"])
        result["tensors"] = tensors
        
        # Data start = after last tensor info
        if tensors:
            last = tensors[-1]
            # Recalculate position after last tensor
            f.seek(tensor_start)
            for t in tensors:
                f.seek(8 + len(t["name"]), 1)  # skip nlen + name
                f.seek(4 + 8 * len(t["dims"]), 1)  # skip n_dims + dims
                f.seek(4 + 8, 1)  # skip ggml_type + offset
            result["data_start"] = (f.tell() + 31) & ~31
        
        # Classify tensors
        for t in tensors:
            tn = t["type"]
            result["tensor_types"][tn] = result["tensor_types"].get(tn, 0) + 1
            
            name = t["name"]
            
            # Detect expert tensors
            if "expert" in name or "exps" in name:
                if "gate" in name:
                    role = "gate"
                elif "up" in name:
                    role = "up"
                elif "down" in name:
                    role = "down"
                elif "w" in name:
                    role = "w"
                else:
                    role = "unknown"
                
                if role not in result["expert_info"]:
                    result["expert_info"][role] = {"count": 0, "types": {}, "total_elements": 0}
                result["expert_info"][role]["count"] += 1
                result["expert_info"][role]["types"][tn] = result["expert_info"][role]["types"].get(tn, 0) + 1
                result["expert_info"][role]["total_elements"] += 1
    
    f.close()
    return result


def compare_models(models):
    """Compare multiple GGUF models."""
    all_results = []
    
    for path in models:
        if not os.path.exists(path):
            print(f"SKIP: {path} not found")
            continue
        
        print(f"\n{'#'*80}")
        print(f"# Analyzing: {os.path.basename(path)}")
        print(f"{'#'*80}")
        
        result = parse_gguf(path)
        if result is None:
            print("  INVALID GGUF file")
            continue
        
        # Basic info
        print(f"\n  File size: {result['size']:,} bytes ({result['size']/1024**3:.2f} GB)")
        print(f"  GGUF version: {result['header']['version']}")
        print(f"  Tensors: {result['header']['n_tensors']}")
        print(f"  KV entries: {result['header']['n_kv']} (parsed: {result['kv_parsed']})")
        
        # Architecture
        arch = result["kv"].get("general.architecture", {}).get("value", "N/A")
        name_val = result["kv"].get("general.name", {}).get("value", "N/A")
        quant_by = result["kv"].get("general.quantized_by", {}).get("value", "N/A")
        print(f"  Architecture: {arch}")
        print(f"  Model name: {name_val}")
        print(f"  Quantized by: {quant_by}")
        
        # imatrix specific metadata
        imatrix_keys = [k for k in result["kv"].keys() if "base_model" in k or "imatrix" in k.lower()]
        if imatrix_keys:
            print(f"\n  imatrix-related keys:")
            for k in sorted(imatrix_keys):
                v = result["kv"][k]
                print(f"    {k}: {v['value']}")
        
        # Tensor type distribution
        if result["tensor_types"]:
            print(f"\n  Tensor type distribution:")
            for tn in sorted(result["tensor_types"].keys()):
                count = result["tensor_types"][tn]
                type_name = GGML_TYPE_MAP.get(tn, f"type_{tn}")
                block_size = BLOCK_SIZE.get(tn, "?")
                block_bits = BLOCK_BITS.get(tn, "?")
                print(f"    {type_name:12s} (type={tn:2d}, block={str(block_size):>4s}, {str(block_bits)+'bit':>4s}): {count:4d} tensors")
        
        # Expert info
        if result["expert_info"]:
            print(f"\n  Expert tensor summary:")
            for role, info in sorted(result["expert_info"].items()):
                types_str = ", ".join(f"{GGML_TYPE_MAP.get(t, t)}={c}" for t, c in info["types"].items())
                print(f"    {role:10s}: {info['count']:4d} tensors, types: {types_str}")
        
        # Data start
        if result.get("data_start"):
            print(f"\n  Data starts at: {result['data_start']:,} (0x{result['data_start']:X})")
            print(f"  First 5 tensors:")
            for t in result.get("first_tensors", [])[:5]:
                type_name = GGML_TYPE_MAP.get(t["type"], f"type_{t['type']}")
                print(f"    '{t['name']}' dims={t['dims']} type={type_name} offset=0x{t['offset']:X}")
        
        all_results.append(result)
    
    # Cross-model comparison
    if len(all_results) >= 2:
        print(f"\n{'='*80}")
        print(f"# Cross-Model Comparison")
        print(f"{'='*80}")
        
        # Compare KV keys
        all_kv_keys = set()
        for r in all_results:
            all_kv_keys.update(r["kv"].keys())
        
        print(f"\n  KV key comparison:")
        for key in sorted(all_kv_keys):
            present_in = []
            for r in all_results:
                if key in r["kv"]:
                    present_in.append(os.path.basename(r["path"]))
            
            if len(present_in) == 1:
                r = [x for x in all_results if key in x["kv"]][0]
                print(f"    [ONLY IN {os.path.basename(r['path'])[:30]}]")
                print(f"      {key}: {r['kv'][key]['value']}")
            elif len(present_in) == len(all_results):
                values = set()
                for r in all_results:
                    values.add(r["kv"][key]["value"][:80])
                if len(values) == 1:
                    print(f"    [SAME] {key}: {list(values)[0]}")
                else:
                    print(f"    [DIFFERENT] {key}:")
                    for r in all_results:
                        print(f"      {os.path.basename(r['path'])[:35]}: {r['kv'][key]['value'][:100]}")
        
        # Tensor type distribution comparison
        print(f"\n  Tensor type distribution comparison:")
        all_types = set()
        for r in all_results:
            all_types.update(r["tensor_types"].keys())
        
        short_names = [os.path.basename(r['path'])[:25] for r in all_results]
        header = f"    {'type':12s}"
        for sn in short_names:
            header += f" {sn:>25s}"
        print(header)
        print("    " + "-" * (12 + 25 * len(all_results)))
        
        for tn in sorted(all_types):
            type_name = GGML_TYPE_MAP.get(tn, f"type_{tn}")
            row = f"    {type_name:12s}"
            for r in all_results:
                count = r["tensor_types"].get(tn, 0)
                row += f" {count:25d}"
            print(row)
        
        # Expert comparison
        print(f"\n  Expert tensor type comparison:")
        all_roles = set()
        for r in all_results:
            all_roles.update(r["expert_info"].keys())
        
        for role in sorted(all_roles):
            print(f"    [{role}]")
            all_role_types = set()
            for r in all_results:
                if role in r["expert_info"]:
                    all_role_types.update(r["expert_info"][role]["types"].keys())
            
            for tn in sorted(all_role_types):
                type_name = GGML_TYPE_MAP.get(tn, f"type_{tn}")
                row = f"      {type_name:12s}"
                for r in all_results:
                    if role in r["expert_info"]:
                        count = r["expert_info"][role]["types"].get(tn, 0)
                    else:
                        count = 0
                    row += f" {count:25d}"
                print(row)
    
    return all_results


if __name__ == "__main__":
    models = [
        r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf",
        r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
        r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
    ]
    
    results = compare_models(models)
