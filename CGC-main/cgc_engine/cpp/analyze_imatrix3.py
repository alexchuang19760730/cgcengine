#!/usr/bin/env python3
"""Analyze GGUF models properly using chunked reads for large blobs."""

import struct
import os

MODELS = {
    "Q4_K_M": r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf",
    "IQ2_XXS": r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
    "IQ3_XXS": r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
}

GGML_TYPE_BYTES = {
    0: 4, 1: 2, 30: 2, 8: 8, 14: 4,
    21: 3, 22: 3, 16: 4, 17: 4,
    19: 2, 20: 2,
}

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 30: "BF16", 8: "Q8_0", 14: "Q4_K",
    21: "IQ3_S", 22: "IQ3_M", 16: "IQ4_NL", 17: "IQ4_XS",
    2: "Q4_0", 3: "Q4_1", 4: "Q5_0", 5: "Q5_1",
    6: "Q5_K", 7: "Q5_K_S", 9: "Q8_0", 10: "INT8",
    11: "INT16", 12: "INT32", 15: "Q6_K",
    18: "IQ3_M", 19: "IQ2_XXS", 20: "IQ2_XS",
}

CHUNK = 4 * 1024 * 1024  # 4MB


def skip_stream(f, n):
    """Skip n bytes in a stream, handling large values."""
    while n > 0:
        to_read = min(CHUNK, n)
        data = f.read(to_read)
        if not data:
            break
        n -= len(data)


def parse_gguf(filepath):
    with open(filepath, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        
        kv = {}
        kv_keys = []
        skipped_kv = 0
        
        for _ in range(n_kv):
            klen_raw = f.read(8)
            if len(klen_raw) < 8:
                break
            klen = struct.unpack("<Q", klen_raw)[0]
            
            if klen <= 65536:
                key_raw = f.read(klen)
                if len(key_raw) < klen:
                    break
                key = key_raw.decode("utf-8", errors="replace")
            else:
                skip_stream(f, klen)
                dtype = struct.unpack("<I", f.read(4))[0]
                kv[f"<big-klen={klen}>"] = f"<skipped dtype={dtype}>"
                kv_keys.append(f"<big-klen={klen}>")
                skipped_kv += 1
                continue
            
            dtype_raw = f.read(4)
            if len(dtype_raw) < 4:
                break
            dtype = struct.unpack("<I", dtype_raw)[0]
            
            val = None
            if dtype in (4, 5, 10):
                v = f.read(4)
                val = struct.unpack("<I", v)[0] if len(v) == 4 else None
            elif dtype == 6:
                v = f.read(4)
                val = struct.unpack("<f", v)[0] if len(v) == 4 else None
            elif dtype == 7:
                v = f.read(8)
                val = struct.unpack("<Q", v)[0] if len(v) == 8 else None
            elif dtype == 8:
                slen_raw = f.read(8)
                if len(slen_raw) < 8:
                    break
                slen = struct.unpack("<Q", slen_raw)[0]
                skip_stream(f, slen)
                val = f"<str-len={slen}>"
            elif dtype == 9:
                elem_type_raw = f.read(4)
                if len(elem_type_raw) < 4:
                    break
                elem_type = struct.unpack("<I", elem_type_raw)[0]
                array_len_raw = f.read(8)
                if len(array_len_raw) < 8:
                    break
                array_len = struct.unpack("<Q", array_len_raw)[0]
                
                if elem_type == 8:
                    # String array: each element has 8-byte length + content
                    # Calculate total bytes to skip
                    total_bytes = 0
                    for _ in range(array_len):
                        sl_raw = f.read(8)
                        if len(sl_raw) < 8:
                            break
                        sl = struct.unpack("<Q", sl_raw)[0]
                        total_bytes += sl
                    skip_stream(f, total_bytes)
                    val = f"<str-array[{array_len}]>"
                else:
                    sizes = {4:4, 5:4, 6:4, 7:8, 10:4, 11:8, 12:8}
                    total = sizes.get(elem_type, 4) * array_len
                    skip_stream(f, total)
                    val = f"<array[{array_len}]>"
            elif dtype == 11:
                v = f.read(8)
                val = struct.unpack("<d", v)[0] if len(v) == 8 else None
            elif dtype == 12:
                v = f.read(8)
                val = struct.unpack("<q", v)[0] if len(v) == 8 else None
            else:
                v = f.read(4)
                val = f"<unknown dtype={dtype}>"
            
            kv[key] = val
            kv_keys.append(key)

        pos_after_kv = f.tell()
        aligned_pos = (pos_after_kv + 31) & ~31
        
        # Parse tensors
        f.seek(aligned_pos)
        tensors = parse_tensors(f, n_tensors)
        
        if len(tensors) < 10:
            # Search nearby
            for offset in range(-512, 513, 4):
                test_pos = aligned_pos + offset
                f.seek(test_pos)
                test = parse_tensors(f, min(n_tensors, 5))
                if len(test) >= 3:
                    f.seek(test_pos)
                    tensors = parse_tensors(f, n_tensors)
                    print(f"  [FIX] Found {len(tensors)} tensors at offset {offset:+d}")
                    break
        
        return {
            "version": version, "n_tensors": n_tensors, "n_kv": n_kv,
            "pos_after_kv": pos_after_kv, "aligned_pos": aligned_pos,
            "kv": kv, "kv_keys": kv_keys, "tensors": tensors,
            "skipped_kv": skipped_kv,
        }


def parse_tensors(f, max_tensors):
    tensors = []
    for _ in range(max_tensors):
        nlen_raw = f.read(8)
        if len(nlen_raw) < 8:
            break
        nlen = struct.unpack("<Q", nlen_raw)[0]
        if nlen == 0 or nlen > 256:
            break
        
        name = f.read(nlen).decode("utf-8", errors="replace")
        if not name.replace("_", "").replace(".", "").replace("-", "").isalnum():
            break
        
        nd_raw = f.read(4)
        if len(nd_raw) < 4:
            break
        n_dims = struct.unpack("<I", nd_raw)[0]
        if n_dims == 0 or n_dims > 6:
            break
        
        dims = []
        for _ in range(n_dims):
            d_raw = f.read(8)
            if len(d_raw) < 8:
                break
            dims.append(struct.unpack("<Q", d_raw)[0])
        if len(dims) < n_dims:
            break
        
        gt_raw = f.read(4)
        if len(gt_raw) < 4:
            break
        ggml_type = struct.unpack("<I", gt_raw)[0]
        
        off_raw = f.read(8)
        if len(off_raw) < 8:
            break
        offset = struct.unpack("<Q", off_raw)[0]
        
        bpe = GGML_TYPE_BYTES.get(ggml_type, 4)
        ne = 1
        for d in dims:
            ne *= d
        
        tensors.append({
            "name": name, "dims": dims, "type": ggml_type,
            "offset": offset, "size_bytes": int(ne * bpe),
        })
    
    return tensors


def main():
    results = {}
    
    for label, path in MODELS.items():
        if not os.path.exists(path):
            print(f"[SKIP] {label}: not found")
            continue
        
        size_gb = os.path.getsize(path) / (1024**3)
        print(f"\n{'='*70}")
        print(f"MODEL: {label} ({size_gb:.2f} GB)")
        print(f"{'='*70}")
        
        try:
            data = parse_gguf(path)
        except Exception as e:
            print(f"  PARSE ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        results[label] = data
        
        print(f"  Version: {data['version']}")
        print(f"  KV items: {data['n_kv']} (skipped: {data['skipped_kv']})")
        print(f"  Tensor items: {data['n_tensors']} (parsed: {len(data['tensors'])})")
        print(f"  Pos after KV: 0x{data['pos_after_kv']:X}")
        print(f"  Aligned tensor start: 0x{data['aligned_pos']:X}")
        
        # --- All KV keys ---
        print(f"\n  --- ALL KV Keys ---")
        for k in data['kv_keys']:
            v = data['kv'][k]
            vs = str(v)
            if len(vs) > 150:
                vs = vs[:150] + "..."
            print(f"    {k} = {vs}")
        
        # --- Tensor Distribution ---
        if data['tensors']:
            print(f"\n  --- Tensor Type Distribution ---")
            type_counts = {}
            type_bytes = {}
            for t in data['tensors']:
                tn = GGML_TYPE_NAMES.get(t['type'], f"T_{t['type']}")
                type_counts[tn] = type_counts.get(tn, 0) + 1
                type_bytes[tn] = type_bytes.get(tn, 0) + t['size_bytes']
            
            for tn in sorted(type_counts.keys(), key=lambda x: -type_bytes.get(x, 0)):
                print(f"    {tn:12s}: {type_counts[tn]:5d} tensors, {type_bytes[tn]/(1024**2):10.1f} MB")
            
            # --- Expert analysis ---
            print(f"\n  --- Expert / MoE Analysis ---")
            expert_by_layer = {}
            for t in data['tensors']:
                name = t['name']
                parts = name.split('.')
                if len(parts) >= 4 and parts[0] == 'blk' and 'exps' in name:
                    try:
                        lid = int(parts[1])
                        if lid not in expert_by_layer:
                            expert_by_layer[lid] = {'gate': 0, 'up': 0, 'down': 0, 'bytes': 0, 'types': set()}
                        if 'ffn_gate_exps' in name:
                            expert_by_layer[lid]['gate'] += 1
                        elif 'ffn_up_exps' in name:
                            expert_by_layer[lid]['up'] += 1
                        elif 'ffn_down_exps' in name:
                            expert_by_layer[lid]['down'] += 1
                        expert_by_layer[lid]['bytes'] += t['size_bytes']
                        expert_by_layer[lid]['types'].add(t['type'])
                    except (ValueError, IndexError):
                        pass
            
            if expert_by_layer:
                num_layers = len(expert_by_layer)
                print(f"  Expert layers: {num_layers}")
                lids = sorted(expert_by_layer.keys())
                for lid in lids[:2]:
                    info = expert_by_layer[lid]
                    types_str = ','.join(GGML_TYPE_NAMES.get(x, f"T{x}") for x in sorted(info['types']))
                    print(f"    Layer {lid}: gate={info['gate']}, up={info['up']}, down={info['down']}, {info['bytes']/(1024**2):.1f} MB, types=[{types_str}]")
                if len(lids) > 3:
                    print(f"    ... ({len(lids)-3} more)")
                for lid in lids[-1:]:
                    info = expert_by_layer[lid]
                    types_str = ','.join(GGML_TYPE_NAMES.get(x, f"T{x}") for x in sorted(info['types']))
                    print(f"    Layer {lid}: gate={info['gate']}, up={info['up']}, down={info['down']}, {info['bytes']/(1024**2):.1f} MB, types=[{types_str}]")
            
            # Check gate/up/down tensor types vs rest
            gate_types = set()
            up_types = set()
            down_types = set()
            non_expert_types = set()
            for t in data['tensors']:
                name = t['name']
                if 'ffn_gate_exps' in name:
                    gate_types.add(t['type'])
                elif 'ffn_up_exps' in name:
                    up_types.add(t['type'])
                elif 'ffn_down_exps' in name:
                    down_types.add(t['type'])
                elif 'exps' not in name:
                    non_expert_types.add(t['type'])
            
            print(f"\n  --- Quantization Types by Role ---")
            print(f"    Gate experts : {[GGML_TYPE_NAMES.get(x, f'T{x}') for x in sorted(gate_types)]}")
            print(f"    Up experts   : {[GGML_TYPE_NAMES.get(x, f'T{x}') for x in sorted(up_types)]}")
            print(f"    Down experts : {[GGML_TYPE_NAMES.get(x, f'T{x}') for x in sorted(down_types)]}")
            print(f"    Non-expert   : {[GGML_TYPE_NAMES.get(x, f'T{x}') for x in sorted(non_expert_types)]}")
            
            # First 5 tensors
            print(f"\n  --- First 5 Tensors ---")
            for t in data['tensors'][:5]:
                print(f"    {t['name']:50s} {GGML_TYPE_NAMES.get(t['type'], t['type']):8s} dims={t['dims']}")
            
            # imatrix-related tensors
            imat_tensors = [t for t in data['tensors'] if 'imat' in t['name'].lower() or 'importance' in t['name'].lower()]
            if imat_tensors:
                print(f"\n  *** Imatrix Tensors Found: ***")
                for t in imat_tensors:
                    print(f"    {t['name']} type={GGML_TYPE_NAMES.get(t['type'], t['type'])} dims={t['dims']}")

    # --- Cross-model comparison ---
    if len(results) >= 2:
        print(f"\n{'='*70}")
        print(f"CROSS-MODEL: Imatrix Impact Analysis")
        print(f"{'='*70}")
        
        # 1. KV key differences
        print(f"\n  --- Imatrix-Related KV Keys ---")
        for key in sorted(set(k for h in results.values() for k in h['kv'])):
            if any(w in key.lower() for w in ['imat', 'quant', 'importance', 'calib', 'dataset', 'base_model']):
                print(f"\n    {key}:")
                for label, h in results.items():
                    v = h['kv'].get(key, '<MISSING>')
                    vs = str(v)
                    if len(vs) > 200:
                        vs = vs[:200] + "..."
                    print(f"      {label:10s}: {vs}")
        
        # 2. Tensor type comparison
        print(f"\n  --- Tensor Type Distribution Comparison ---")
        all_types = set()
        for h in results.values():
            for t in h['tensors']:
                all_types.add(t['type'])
        
        for tt in sorted(all_types):
            tn = GGML_TYPE_NAMES.get(tt, f"T_{tt}")
            parts = []
            for label, h in results.items():
                tensors_of_type = [t for t in h['tensors'] if t['type'] == tt]
                bytes_of_type = sum(t['size_bytes'] for t in tensors_of_type)
                parts.append(f"{label}: {len(tensors_of_type):4d} tensors, {bytes_of_type/(1024**2):8.1f}MB")
            print(f"    {tn:12s}: {' | '.join(parts)}")
        
        # 3. Expert role type comparison
        print(f"\n  --- Expert Role Quantization Comparison ---")
        for label, h in results.items():
            gate_types = set()
            up_types = set()
            down_types = set()
            for t in h['tensors']:
                name = t['name']
                if 'ffn_gate_exps' in name:
                    gate_types.add(t['type'])
                elif 'ffn_up_exps' in name:
                    up_types.add(t['type'])
                elif 'ffn_down_exps' in name:
                    down_types.add(t['type'])
            
            gt_s = [GGML_TYPE_NAMES.get(x, f"T{x}") for x in sorted(gate_types)]
            ut_s = [GGML_TYPE_NAMES.get(x, f"T{x}") for x in sorted(up_types)]
            dt_s = [GGML_TYPE_NAMES.get(x, f"T{x}") for x in sorted(down_types)]
            print(f"    {label}: gate={gt_s}, up={ut_s}, down={dt_s}")

    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
