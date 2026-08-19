#!/usr/bin/env python3
"""
Final GGUF parser with correct bool (1-byte) and string array handling.
Verifies all tensors can be correctly parsed.
"""
import struct
import os

def big_skip(f, n):
    CHUNK = 4 * 1024 * 1024
    while n > 0:
        d = f.read(min(CHUNK, n))
        if not d:
            break
        n -= len(d)

def parse_gguf_correct(filepath):
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
    
    print(f"File: {os.path.basename(filepath)}")
    print(f"Version: {version}, n_tensors: {n_tensors}, n_kv: {n_kv}")
    
    # Parse KV items with CORRECT bool handling (1-byte)
    parsed_kv = 0
    for idx in range(n_kv):
        item_start = f.tell()
        
        klen_bytes = f.read(8)
        if len(klen_bytes) < 8:
            print(f"  [{idx}] EOF at klen")
            break
        klen = struct.unpack("<Q", klen_bytes)[0]
        
        if klen == 0 or klen > 1000000:
            print(f"  [{idx}] INVALID klen={klen}")
            break
        
        key_bytes = f.read(klen)
        if len(key_bytes) < klen:
            print(f"  [{idx}] EOF at key")
            break
        key = key_bytes.decode("utf-8", errors="replace")
        
        dtype_bytes = f.read(4)
        if len(dtype_bytes) < 4:
            break
        dtype = struct.unpack("<I", dtype_bytes)[0]
        
        # Read value based on dtype
        val_desc = ""
        
        if dtype == 7:  # bool: 1 byte value, NO padding
            v = f.read(1)
            if len(v) == 1:
                val_desc = f"bool={v[0]}"
            else:
                val_desc = "ERR"
                
        elif dtype == 8:  # string
            slen_bytes = f.read(8)
            if len(slen_bytes) < 8:
                break
            slen = struct.unpack("<Q", slen_bytes)[0]
            chunk = f.read(min(80, slen))
            big_skip(f, slen - len(chunk))
            s = chunk.decode("utf-8", errors="replace")
            val_desc = f"str[{slen}]='{s[:80]}'"
            
        elif dtype == 9:  # array
            et_bytes = f.read(4)
            al_bytes = f.read(8)
            if len(et_bytes) < 4 or len(al_bytes) < 8:
                break
            elem_type = struct.unpack("<I", et_bytes)[0]
            array_len = struct.unpack("<Q", al_bytes)[0]
            
            if elem_type == 8:  # string array
                total_str_bytes = 0
                for i in range(array_len):
                    sl_bytes = f.read(8)
                    if len(sl_bytes) < 8:
                        break
                    sl = struct.unpack("<Q", sl_bytes)[0]
                    total_str_bytes += sl
                    if i < 3:
                        chunk = f.read(min(60, sl))
                        s = chunk.decode("utf-8", errors="replace")
                        val_desc += f", '{s[:60]}'" if val_desc == "" else ""
                    big_skip(f, sl)
                
                array_total = array_len * 8 + total_str_bytes
                val_desc = f"str-array[{array_len}] total={array_total:,} first3=[{val_desc}]"
                
            elif elem_type in (4, 5, 6, 10):  # uint32/int32/float32/uint64
                elem_size = 8 if elem_type == 10 else 4
                total = elem_size * array_len
                big_skip(f, total)
                val_desc = f"array[{array_len}] elem_type={elem_type} size={total:,}"
                
            elif elem_type == 7:  # bool array
                total = 1 * array_len
                big_skip(f, total)
                val_desc = f"bool-array[{array_len}] size={total:,}"
                
            else:
                total = 4 * array_len
                big_skip(f, total)
                val_desc = f"array[{array_len}] elem_type={elem_type}"
        
        elif dtype == 10:  # uint64
            v = f.read(8)
            val_desc = f"uint64={struct.unpack('<Q', v)[0]}" if len(v) == 8 else "ERR"
        elif dtype == 11:  # float64
            v = f.read(8)
            val_desc = f"float64={struct.unpack('<d', v)[0]:.6g}" if len(v) == 8 else "ERR"
        elif dtype == 12:  # int64
            v = f.read(8)
            val_desc = f"int64={struct.unpack('<q', v)[0]}" if len(v) == 8 else "ERR"
        else:
            sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4}
            s = sizes.get(dtype, 4)
            v = f.read(s)
            if len(v) == s:
                if dtype == 0:
                    val_desc = f"uint8={v[0]}"
                elif dtype == 1:
                    val_desc = f"int8={v[0]}"
                elif dtype == 2:
                    val_desc = f"uint16={struct.unpack('<H', v)[0]}"
                elif dtype == 3:
                    val_desc = f"int16={struct.unpack('<h', v)[0]}"
                elif dtype == 4:
                    val_desc = f"uint32={struct.unpack('<I', v)[0]}"
                elif dtype == 5:
                    val_desc = f"int32={struct.unpack('<i', v)[0]}"
                elif dtype == 6:
                    val_desc = f"float32={struct.unpack('<f', v)[0]:.6g}"
                else:
                    val_desc = f"val={v.hex()}"
            else:
                val_desc = "ERR"
        
        result["kv"][key] = {"dtype": dtype, "value": val_desc}
        parsed_kv += 1
        
        if parsed_kv % 10 == 0 or "tokenizer" in key or "add_bos" in key:
            print(f"  [{idx:3d}] key='{key[:60]}' dtype={dtype} -> {val_desc[:80]}")
    
    result["kv_parsed"] = parsed_kv
    print(f"\nParsed {parsed_kv}/{n_kv} KV items")
    
    # Now get position after KV
    kv_end = f.tell()
    print(f"KV ends at: {kv_end} (0x{kv_end:X})")
    
    # Verify: try reading first tensor info right here
    print(f"\n--- Verifying tensor info at KV end ---")
    f.seek(kv_end)
    
    # Try first tensor
    nlen_bytes = f.read(8)
    nlen = struct.unpack("<Q", nlen_bytes)[0]
    print(f"  First nlen at KV end: {nlen}")
    
    if 3 <= nlen <= 256:
        name = f.read(nlen).decode("utf-8", errors="replace")
        print(f"  First potential tensor name: '{name}'")
        
        if any(kw in name for kw in ["blk.", "token_embd", "output", "norm.", "rope.", "tokenizer"]):
            print(f"  VALID tensor name!")
            
            nd = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            gt = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            
            print(f"  Dims: {dims}, type: {gt}, offset: 0x{off:X}")
            
            # Parse all tensors
            f.seek(kv_end)
            for t_idx in range(n_tensors):
                nl_bytes = f.read(8)
                if len(nl_bytes) < 8:
                    print(f"  ERROR: EOF at tensor {t_idx}")
                    break
                nl = struct.unpack("<Q", nl_bytes)[0]
                if nl < 3 or nl > 256:
                    print(f"  ERROR: Invalid name length {nl} at tensor {t_idx}")
                    break
                
                tname = f.read(nl).decode("utf-8", errors="replace")
                if len(tname) < nl:
                    print(f"  ERROR: EOF at tensor name {t_idx}")
                    break
                
                nd2_bytes = f.read(4)
                if len(nd2_bytes) < 4:
                    break
                nd2 = struct.unpack("<I", nd2_bytes)[0]
                if nd2 == 0 or nd2 > 5:
                    print(f"  ERROR: Invalid n_dims {nd2} at tensor {t_idx}")
                    break
                
                tdims = []
                for _ in range(nd2):
                    d_bytes = f.read(8)
                    if len(d_bytes) < 8:
                        break
                    tdims.append(struct.unpack("<Q", d_bytes)[0])
                if len(tdims) != nd2:
                    break
                
                gt_bytes = f.read(4)
                if len(gt_bytes) < 4:
                    break
                ggml_type = struct.unpack("<I", gt_bytes)[0]
                
                off_bytes = f.read(8)
                if len(off_bytes) < 8:
                    break
                offset = struct.unpack("<Q", off_bytes)[0]
                
                result["tensors"].append({
                    "name": tname,
                    "dims": tdims,
                    "type": ggml_type,
                    "offset": offset,
                })
                
                # Classify
                result["tensor_types"][ggml_type] = result["tensor_types"].get(ggml_type, 0) + 1
                
                tn = tname
                if "expert" in tn or "exps" in tn:
                    if "gate" in tn:
                        role = "gate"
                    elif "up" in tn:
                        role = "up"
                    elif "down" in tn:
                        role = "down"
                    elif "w" in tn:
                        role = "w"
                    else:
                        role = "unknown"
                    if role not in result["expert_info"]:
                        result["expert_info"][role] = {"count": 0, "types": {}}
                    result["expert_info"][role]["count"] += 1
                    result["expert_info"][role]["types"][ggml_type] = result["expert_info"][role]["types"].get(ggml_type, 0) + 1
                
                if t_idx < 5 or "expert" in tname:
                    print(f"  Tensor {t_idx}: '{tname}' dims={tdims} type={ggml_type} offset=0x{offset:X}")
        else:
            print(f"  INVALID tensor name (no known prefix)")
    else:
        print(f"  INVALID name length: {nlen}")
        
        # Search for correct position
        print(f"\n  Searching for tensor info...")
        for search in range(kv_end, min(kv_end + 20000, file_size - 100), 32):
            f.seek(search)
            nl_bytes = f.read(8)
            nl = struct.unpack("<Q", nl_bytes)[0]
            if nl < 3 or nl > 256:
                continue
            nm = f.read(nl).decode("utf-8", errors="replace")
            if not any(kw in nm for kw in ["blk.", "token_embd", "output", "norm.", "rope.", "tokenizer"]):
                continue
            
            nd = struct.unpack("<I", f.read(4))[0]
            if nd == 0 or nd > 5:
                continue
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            gt = struct.unpack("<I", f.read(4))[0]
            if gt > 50:
                continue
            off = struct.unpack("<Q", f.read(8))[0]
            if off > file_size:
                continue
            
            print(f"  FOUND at {search} (0x{search:X}): '{nm}' dims={dims} type={gt} offset=0x{off:X}")
            break
    
    # Calculate data start
    if result["tensors"]:
        f.seek(kv_end)
        for t in result["tensors"]:
            f.seek(8 + len(t["name"]), 1)  # nlen + name
            nd = len(t["dims"])
            f.seek(4 + 8 * nd, 1)  # n_dims + dims
            f.seek(4 + 8, 1)  # ggml_type + offset
        
        info_end = f.tell()
        data_start = (info_end + 31) & ~31
        result["data_start"] = data_start
        result["info_end"] = info_end
        print(f"\nTensor info ends at: {info_end} (0x{info_end:X})")
        print(f"Data starts at: {data_start} (0x{data_start:X})")
        print(f"Parsed {len(result['tensors'])}/{n_tensors} tensors")
    
    f.close()
    return result

if __name__ == "__main__":
    path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
    if os.path.exists(path):
        result = parse_gguf_correct(path)
        
        if result:
            print(f"\n{'='*60}")
            print(f"Summary:")
            print(f"  KV entries: {result['kv_parsed']}/{result['header']['n_kv']}")
            print(f"  Tensors: {len(result['tensors'])}/{result['header']['n_tensors']}")
            print(f"  Data start: 0x{result.get('data_start', 0):X}")
            
            arch = result["kv"].get("general.architecture", {}).get("value", "N/A")
            print(f"  Architecture: {arch}")
            
            print(f"\n  Tensor types:")
            GGML_TYPE_MAP = {0:"f32",1:"f16",2:"q4_0",3:"q4_1",4:"q5_0",5:"q5_1",6:"q8_0",
                           14:"q4_k",15:"q5_k",16:"q6_k",17:"q5_k_m",18:"q5_k_s",19:"q4_k_m",
                           20:"q4_k_s",21:"q3_k_l",22:"q3_k_m",23:"q3_k_s",24:"q2_k",
                           25:"iq3_m",26:"iq3_s",27:"iq2_m",28:"iq1_m",
                           29:"iq4_xs",30:"iq3_xs",31:"iq2_xs"}
            for tn, cnt in sorted(result["tensor_types"].items()):
                tname = GGML_TYPE_MAP.get(tn, f"type_{tn}")
                print(f"    {tname:12s} (type={tn:2d}): {cnt:4d}")
            
            if result["expert_info"]:
                print(f"\n  Expert info:")
                for role, info in sorted(result["expert_info"].items()):
                    print(f"    {role}: {info['count']} tensors, types: {info['types']}")
    else:
        print(f"File not found: {path}")
