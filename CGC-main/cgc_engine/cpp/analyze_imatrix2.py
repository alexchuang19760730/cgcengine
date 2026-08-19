#!/usr/bin/env python3
"""Analyze imatrix by reading GGUF correctly - skip tokenizer tokens blob properly."""

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


def parse_gguf(filepath):
    with open(filepath, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        
        kv = {}
        kv_keys = []
        
        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]
            key = f.read(klen).decode("utf-8", errors="replace") if klen <= 65536 else f"<klen={klen}>"
            
            dtype = struct.unpack("<I", f.read(4))[0]
            
            if dtype in (4, 5, 10):
                val = struct.unpack("<I", f.read(4))[0]
            elif dtype == 6:
                val = struct.unpack("<f", f.read(4))[0]
            elif dtype == 7:
                val = struct.unpack("<Q", f.read(8))[0]
            elif dtype == 8:
                slen = struct.unpack("<Q", f.read(8))[0]
                f.seek(slen, 1)
                val = f"<str-len={slen}>"
            elif dtype == 9:
                elem_type = struct.unpack("<I", f.read(4))[0]
                array_len = struct.unpack("<Q", f.read(8))[0]
                if elem_type == 8:
                    # String array - skip properly
                    total_str_bytes = 0
                    for _ in range(array_len):
                        sl = struct.unpack("<Q", f.read(8))[0]
                        total_str_bytes += 8 + sl
                    # Now seek back and skip the total
                    f.seek(-(8 * array_len), 1)  # unread the lengths
                    f.seek(total_str_bytes, 1)  # skip everything
                    val = f"<str-array[{array_len}]>"
                else:
                    sizes = {4:4, 5:4, 6:4, 7:8, 10:4, 11:8, 12:8}
                    total = sizes.get(elem_type, 4) * array_len
                    f.seek(total, 1)
                    val = f"<array[{array_len}]>"
            elif dtype == 11:
                val = struct.unpack("<d", f.read(8))[0]
            elif dtype == 12:
                val = struct.unpack("<q", f.read(8))[0]
            else:
                val = f"<unknown dtype={dtype}>"
            
            kv[key] = val
            kv_keys.append(key)

        pos_after_kv = f.tell()
        
        # Align to 32 bytes
        aligned_pos = (pos_after_kv + 31) & ~31
        
        # Try parsing tensors from aligned position
        f.seek(aligned_pos)
        tensors = parse_tensors_from(f, n_tensors)
        
        # If failed, search nearby for tensor names
        if len(tensors) < 10:
            print(f"  [WARN] Aligned parse gave {len(tensors)} tensors, searching...")
            for offset in range(-256, 257, 8):
                test_pos = aligned_pos + offset
                f.seek(test_pos)
                test_tensors = parse_tensors_from(f, min(n_tensors, 10))
                if len(test_tensors) >= 5:
                    f.seek(test_pos)
                    tensors = parse_tensors_from(f, n_tensors)
                    print(f"  [OK] Found {len(tensors)} tensors at offset {offset:+d}")
                    break
        
        return {
            "version": version, "n_tensors": n_tensors, "n_kv": n_kv,
            "pos_after_kv": pos_after_kv, "aligned_pos": aligned_pos,
            "kv": kv, "kv_keys": kv_keys, "tensors": tensors,
        }


def parse_tensors_from(f, max_tensors):
    tensors = []
    for _ in range(max_tensors):
        nlen_raw = f.read(8)
        if len(nlen_raw) < 8:
            break
        nlen = struct.unpack("<Q", nlen_raw)[0]
        if nlen == 0 or nlen > 256:
            break
        name = f.read(nlen).decode("utf-8", errors="replace")
        if not name.isprintable() or len(name) < 2:
            break
        
        nd_raw = f.read(4)
        if len(nd_raw) < 4:
            break
        n_dims = struct.unpack("<I", nd_raw)[0]
        if n_dims > 6:
            break
        
        dims_raw = f.read(8 * n_dims)
        if len(dims_raw) < 8 * n_dims:
            break
        dims = [struct.unpack("<Q", dims_raw[i*8:(i+1)*8])[0] for i in range(n_dims)]
        
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
        print(f"  KV items: {data['n_kv']}")
        print(f"  Tensor items: {data['n_tensors']} (parsed: {len(data['tensors'])})")
        print(f"  Pos after KV: 0x{data['pos_after_kv']:X}")
        print(f"  Aligned tensor start: 0x{data['aligned_pos']:X}")
        
        # --- All KV keys ---
        print(f"\n  --- ALL KV Keys ---")
        for k in data['kv_keys']:
            v = data['kv'][k]
            vs = str(v)
            if len(vs) > 120:
                vs = vs[:120] + "..."
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
                            expert_by_layer[lid] = {'gate': 0, 'up': 0, 'down': 0, 'bytes': 0}
                        if 'ffn_gate_exps' in name:
                            expert_by_layer[lid]['gate'] += 1
                        elif 'ffn_up_exps' in name:
                            expert_by_layer[lid]['up'] += 1
                        elif 'ffn_down_exps' in name:
                            expert_by_layer[lid]['down'] += 1
                        expert_by_layer[lid]['bytes'] += t['size_bytes']
                    except (ValueError, IndexError):
                        pass
            
            if expert_by_layer:
                num_layers = len(expert_by_layer)
                print(f"  Expert layers: {num_layers}")
                lids = sorted(expert_by_layer.keys())
                for lid in lids[:2]:
                    info = expert_by_layer[lid]
                    print(f"    Layer {lid}: gate={info['gate']}, up={info['up']}, down={info['down']}, {info['bytes']/(1024**2):.1f} MB")
                if len(lids) > 3:
                    print(f"    ... ({len(lids)-3} more)")
                for lid in lids[-1:]:
                    info = expert_by_layer[lid]
                    print(f"    Layer {lid}: gate={info['gate']}, up={info['up']}, down={info['down']}, {info['bytes']/(1024**2):.1f} MB")
            
            # First 5 and last 5 tensors
            print(f"\n  --- First 5 Tensors ---")
            for t in data['tensors'][:5]:
                print(f"    {t['name']:50s} {GGML_TYPE_NAMES.get(t['type'], t['type']):8s} dims={t['dims']} offset=0x{t['offset']:X}")
            print(f"\n  --- Last 5 Tensors ---")
            for t in data['tensors'][-5:]:
                print(f"    {t['name']:50s} {GGML_TYPE_NAMES.get(t['type'], t['type']):8s} dims={t['dims']} offset=0x{t['offset']:X}")

    # --- Cross-model comparison ---
    if len(results) >= 2:
        print(f"\n{'='*70}")
        print(f"CROSS-MODEL COMPARISON: Imatrix Impact")
        print(f"{'='*70}")
        
        # Compare KV keys
        for key in sorted(set(k for h in results.values() for k in h['kv'])):
            if any(w in key.lower() for w in ['imat', 'quant', 'importance', 'calib', 'dataset', 'offset', 'scale', 'base_model']):
                print(f"\n  {key}:")
                for label, h in results.items():
                    v = h['kv'].get(key, '<MISSING>')
                    vs = str(v)
                    if len(vs) > 200:
                        vs = vs[:200] + "..."
                    print(f"    {label:10s}: {vs}")
        
        # Compare tensor types
        print(f"\n  --- Tensor Type Comparison ---")
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

    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
