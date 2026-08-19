#!/usr/bin/env python3
"""Deep analysis: what imatrix does to GGUF files.

Compares KV metadata and tensor distribution between:
- Original Q4_K_M (non-imatrix quantized)
- IQ2_XXS / IQ3_XXS (imatrix-quantized)
Both from the same Qwen3.6-35B-A3B-DFlash family.
"""

import struct
import sys
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


def safe_seek(f, pos):
    try:
        f.seek(pos)
        return True
    except Exception:
        return False


def safe_read(f, n):
    try:
        data = f.read(n)
        return data if len(data) == n else None
    except Exception:
        return None


def parse_gguf_header(filepath):
    with open(filepath, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != 0x46554747:
            raise ValueError(f"Invalid magic: 0x{magic:08X}")

        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        kv = {}
        kv_raw = []
        skipped = 0
        
        for _ in range(n_kv):
            entry_start = f.tell()
            klen_raw = safe_read(f, 8)
            if klen_raw is None:
                break
            klen = struct.unpack("<Q", klen_raw)[0]
            
            if klen <= 65536:
                key = safe_read(f, klen)
                if key is None:
                    break
                key = key.decode("utf-8", errors="replace")
            else:
                key = f"<big-klen={klen}>"
                if not safe_seek(f, klen):
                    _ = safe_read(f, klen) or b""
                dtype_raw = safe_read(f, 4)
                if dtype_raw is None:
                    break
                dtype = struct.unpack("<I", dtype_raw)[0]
                val = f"<skipped: big klen>"
                kv[key] = val
                kv_raw.append((key, dtype, val))
                skipped += 1
                continue
            
            dtype_raw = safe_read(f, 4)
            if dtype_raw is None:
                break
            dtype = struct.unpack("<I", dtype_raw)[0]
            
            val = None
            if dtype in (4, 5, 10):
                v = safe_read(f, 4)
                val = struct.unpack("<I", v)[0] if v else None
            elif dtype == 6:
                v = safe_read(f, 4)
                val = struct.unpack("<f", v)[0] if v else None
            elif dtype == 7:
                v = safe_read(f, 8)
                val = struct.unpack("<Q", v)[0] if v else None
            elif dtype == 8:
                slen_raw = safe_read(f, 8)
                if slen_raw is None:
                    break
                slen = struct.unpack("<Q", slen_raw)[0]
                if slen > 1048576:
                    if not safe_seek(f, slen):
                        _ = safe_read(f, slen) or b""
                    val = f"<big-string-len={slen}>"
                else:
                    v = safe_read(f, slen)
                    val = v.decode("utf-8", errors="replace") if v else ""
            elif dtype == 9:
                saved_pos = f.tell()
                probe = safe_read(f, 12)
                if probe is None or len(probe) < 12:
                    break
                elem_type = struct.unpack("<I", probe[0:4])[0]
                array_len = struct.unpack("<Q", probe[4:12])[0]
                if array_len < 10000000 and elem_type in (4, 5, 6, 7, 8, 10, 11, 12):
                    if elem_type == 8:
                        strings = []
                        for _ in range(min(array_len, 50)):
                            sl_raw = safe_read(f, 8)
                            if sl_raw is None:
                                break
                            sl = struct.unpack("<Q", sl_raw)[0]
                            if sl > 50 * 1024 * 1024:
                                if not safe_seek(f, sl):
                                    _ = safe_read(f, sl) or b""
                            else:
                                v = safe_read(f, sl)
                                if v:
                                    strings.append(v.decode("utf-8", errors="replace"))
                        val = strings if array_len <= 50 else f"<str-array[{array_len}] first 5: {strings[:5]}>"
                    else:
                        sizes = {4:4, 5:4, 6:4, 7:8, 10:4, 11:8, 12:8}
                        total = sizes.get(elem_type, 4) * array_len
                        if not safe_seek(f, saved_pos + 12 + total):
                            _ = safe_read(f, total) or b""
                        val = f"<array[{array_len}]>"
                else:
                    # Treat as bool
                    one = safe_read(f, 1)
                    val = True
            elif dtype == 11:
                v = safe_read(f, 8)
                val = struct.unpack("<d", v)[0] if v else None
            elif dtype == 12:
                v = safe_read(f, 8)
                val = struct.unpack("<q", v)[0] if v else None
            else:
                v = safe_read(f, 4)
                val = f"<unknown dtype={dtype}>"
            
            kv[key] = val
            kv_raw.append((key, dtype, val))

        pos_after_kv = f.tell()
        aligned_pos = (pos_after_kv + 31) & ~31
        
        f.seek(aligned_pos)
        tensors = []
        for _ in range(n_tensors):
            nlen_raw = safe_read(f, 8)
            if nlen_raw is None or len(nlen_raw) < 8:
                break
            nlen = struct.unpack("<Q", nlen_raw)[0]
            if nlen == 0 or nlen > 256:
                break
            name = safe_read(f, nlen)
            if name is None:
                break
            name = name.decode("utf-8", errors="replace")
            nd_raw = safe_read(f, 4)
            if nd_raw is None:
                break
            n_dims = struct.unpack("<I", nd_raw)[0]
            dims = []
            for _ in range(n_dims):
                d_raw = safe_read(f, 8)
                if d_raw is None:
                    break
                dims.append(struct.unpack("<Q", d_raw)[0])
            gt_raw = safe_read(f, 4)
            if gt_raw is None:
                break
            ggml_type = struct.unpack("<I", gt_raw)[0]
            off_raw = safe_read(f, 8)
            if off_raw is None:
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
        
        return {
            "version": version, "n_tensors": n_tensors, "n_kv": n_kv,
            "pos_after_kv": pos_after_kv, "aligned_pos": aligned_pos,
            "kv": kv, "kv_raw": kv_raw, "tensors": tensors,
            "skipped_kv": skipped,
        }


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
            header = parse_gguf_header(path)
        except Exception as e:
            print(f"  PARSE ERROR: {e}")
            continue
        
        results[label] = header
        
        print(f"  Version: {header['version']}")
        print(f"  KV items: {header['n_kv']} (skipped big: {header['skipped_kv']})")
        print(f"  Tensor items: {header['n_tensors']} (parsed: {len(header['tensors'])})")
        print(f"  Pos after KV: 0x{header['pos_after_kv']:X}")
        print(f"  Aligned tensor start: 0x{header['aligned_pos']:X}")
        
        # --- Imatrix KV keys ---
        imatrix_keys = [(k, d, v) for (k, d, v) in header['kv_raw'] 
                        if any(w in k.lower() for w in ['imat', 'quant', 'importance', 'calib', 'dataset'])]
        
        print(f"\n  --- Imatrix / Quantization KV Keys ({len(imatrix_keys)}) ---")
        for k, d, v in imatrix_keys:
            vs = str(v)
            if len(vs) > 200:
                vs = vs[:200] + "..."
            print(f"    [{d:2d}] {k} = {vs}")
        
        # --- ALL KV keys ---
        print(f"\n  --- ALL KV Keys ---")
        for k, d, v in header['kv_raw']:
            vs = str(v)
            if len(vs) > 150:
                vs = vs[:150] + "..."
            print(f"    [{d:2d}] {k} = {vs}")
        
        # --- Tensor Distribution ---
        print(f"\n  --- Tensor Type Distribution ---")
        type_counts = {}
        type_bytes = {}
        for t in header['tensors']:
            tn = GGML_TYPE_NAMES.get(t['type'], f"T_{t['type']}")
            type_counts[tn] = type_counts.get(tn, 0) + 1
            type_bytes[tn] = type_bytes.get(tn, 0) + t['size_bytes']
        
        for tn in sorted(type_counts.keys(), key=lambda x: -type_bytes.get(x, 0)):
            print(f"    {tn:12s}: {type_counts[tn]:5d} tensors, {type_bytes[tn]/(1024**2):10.1f} MB")
        
        # --- Expert tensor analysis ---
        print(f"\n  --- Expert / MoE Analysis ---")
        expert_by_layer = {}
        for t in header['tensors']:
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
        
        num_layers = len(expert_by_layer)
        if num_layers > 0:
            print(f"  Expert layers: {num_layers}")
            lids = sorted(expert_by_layer.keys())
            for lid in lids[:2]:
                info = expert_by_layer[lid]
                print(f"    Layer {lid}: gate={info['gate']}, up={info['up']}, down={info['down']}, {info['bytes']/(1024**2):.1f} MB")
            if len(lids) > 3:
                print(f"    ... ({len(lids)-3} more layers)")
            for lid in lids[-1:]:
                info = expert_by_layer[lid]
                print(f"    Layer {lid}: gate={info['gate']}, up={info['up']}, down={info['down']}, {info['bytes']/(1024**2):.1f} MB")
        
        # Check if any imatrix-related tensors
        imat_tensors = [t for t in header['tensors'] if 'imat' in t['name'].lower() or 'importance' in t['name'].lower()]
        if imat_tensors:
            print(f"\n  *** Imatrix Tensors Found ({len(imat_tensors)}): ***")
            for t in imat_tensors[:10]:
                print(f"    {t['name']} type={GGML_TYPE_NAMES.get(t['type'], t['type'])} dims={t['dims']} size={t['size_bytes']/1024:.1f}KB")
        else:
            print(f"\n  No tensors with 'imatrix' or 'importance' in name.")
            print(f"  (Imatrix data is stored in KV metadata only, not as separate tensors)")
    
    # --- Cross-model imatrix comparison ---
    if len(results) >= 2:
        print(f"\n{'='*70}")
        print(f"IMATRIX IMPACT: Cross-Model Comparison")
        print(f"{'='*70}")
        
        for key in sorted(set(k for h in results.values() for k in h['kv'])):
            if any(w in key.lower() for w in ['imat', 'quant', 'importance', 'calib', 'dataset', 'offset', 'scale']):
                print(f"\n  {key}:")
                for label, h in results.items():
                    v = h['kv'].get(key, '<MISSING>')
                    vs = str(v)
                    if len(vs) > 200:
                        vs = vs[:200] + "..."
                    print(f"    {label:10s}: {vs}")

    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
