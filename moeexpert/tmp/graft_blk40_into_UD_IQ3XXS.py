"""
Path A: Graft fraQtl Q4_K blk.40 tensors INTO UD-IQ3_XXS trunk (which has NO blk.40).

Unlike graft_blk40_fraQtl_into_Nail.py (which REPLACES Nail's blk.40),
this script APPENDS blk.40 tensors to UD-IQ3_XXS (733 tensors -> 733+~30).

UD-IQ3_XXS has block_count=40 and NO blk.40.* tensors.
Output will have block_count=41 and blk.40.* from fraQtl (Q4_K).

Why: UD-IQ3_XXS blk.0-39 ffn_down_exps = IQ3_XXS (fast, 102MB each).
Nail trunk blk.0-39 ffn_down_exps = IQ3_S (slow, 115MB each).
Using UD-IQ3_XXS trunk keeps IQ3_XXS for all forward layers.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFValueType
from gguf.quants import quant_shape_to_byte_shape

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TRUNK_GGUF  = "/Users/alexchuang/Documents/flashkv0516/models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
FRAQTL_GGUF = "/Volumes/AlexZhuang/flashkv0516/models/gguf/fraQtl/qwen36-35b-a3b-hi-fi-mtp-runtime.gguf"
OUTPUT_GGUF = "/Users/alexchuang/Documents/flashkv0516/models/gguf/Qwen3.6-35B-A3B-UD-IQ3XXS-trunk_Q4K-blk40.gguf"
BLK40_PREFIX = "blk.40."

# Metadata keys we override manually (do NOT copy from trunk)
OVERRIDE_META_KEYS = {
    b"qwen35moe.block_count",  # 40 -> 41 (trunk has 40, we add blk.40)
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def copy_metadata(reader, writer):
    """Copy all KV metadata from reader to writer, skipping OVERRIDE_META_KEYS."""
    copied = 0
    skipped = 0
    for fname, field in reader.fields.items():
        if isinstance(fname, bytes):
            key_str = fname.decode("utf-8", errors="replace")
        else:
            key_str = fname
        if key_str.startswith("GGUF."):
            continue
        if key_str == "general.alignment":
            continue
        if fname in OVERRIDE_META_KEYS or key_str in OVERRIDE_META_KEYS:
            skipped += 1
            continue
        if not field.types:
            continue
        first_val_type = field.types[0]
        if first_val_type == GGUFValueType.ARRAY:
            elem_type = field.types[1] if len(field.types) > 1 else None
            elements = []
            for pi in field.data:
                arr = field.parts[pi]
                if elem_type == GGUFValueType.STRING:
                    elements.append(arr.tobytes().decode("utf-8", errors="replace"))
                elif arr.ndim == 0 or arr.size == 1:
                    elements.append(arr.item())
                else:
                    elements.append(arr.tolist())
            try:
                if elem_type == GGUFValueType.STRING:
                    writer.add_array(key_str, elements)
                elif elem_type == GGUFValueType.INT32:
                    writer.add_array(key_str, [int(x) for x in elements])
                elif elem_type == GGUFValueType.UINT32:
                    writer.add_array(key_str, [int(x) for x in elements])
                elif elem_type == GGUFValueType.FLOAT32:
                    writer.add_array(key_str, [float(x) for x in elements])
                else:
                    skipped += 1
                    continue
                copied += 1
            except Exception as e:
                print(f"  [skip-meta] {key_str} (array {elem_type}): {e}")
                skipped += 1
            continue
        val_type = field.types[0]
        part_idx = field.data[0]
        arr = field.parts[part_idx]
        try:
            if val_type == GGUFValueType.STRING:
                v_str = arr.tobytes().decode("utf-8", errors="replace") if arr.dtype == np.uint8 else str(arr.item())
                writer.add_string(key_str, v_str)
            else:
                v = arr.item() if arr.size == 1 else arr.tolist()
                if val_type == GGUFValueType.UINT8:
                    writer.add_uint8(key_str, int(v))
                elif val_type == GGUFValueType.INT8:
                    writer.add_int8(key_str, int(v))
                elif val_type == GGUFValueType.UINT16:
                    writer.add_uint16(key_str, int(v))
                elif val_type == GGUFValueType.INT16:
                    writer.add_int16(key_str, int(v))
                elif val_type == GGUFValueType.UINT32:
                    writer.add_uint32(key_str, int(v))
                elif val_type == GGUFValueType.INT32:
                    writer.add_int32(key_str, int(v))
                elif val_type == GGUFValueType.UINT64:
                    writer.add_uint64(key_str, int(v))
                elif val_type == GGUFValueType.INT64:
                    writer.add_int64(key_str, int(v))
                elif val_type == GGUFValueType.FLOAT32:
                    writer.add_float32(key_str, float(v))
                elif val_type == GGUFValueType.FLOAT64:
                    writer.add_float64(key_str, float(v))
                elif val_type == GGUFValueType.BOOL:
                    writer.add_bool(key_str, bool(v))
                else:
                    skipped += 1
                    continue
            copied += 1
        except Exception as e:
            print(f"  [skip-meta] {key_str}: {e}")
            skipped += 1
    return copied, skipped


def read_tensor_bytes(reader, reader_tensor, file_obj):
    """Read raw bytes of a tensor from the underlying file."""
    file_obj.seek(reader_tensor.data_offset)
    n = reader_tensor.n_bytes
    data = file_obj.read(n)
    if len(data) != n:
        raise IOError(f"short read: wanted {n}, got {len(data)}")
    return np.frombuffer(data, dtype=np.uint8)


def compute_byte_shape_numpy(ggml_dims, gguf_type):
    logical_numpy = [int(d) for d in reversed(ggml_dims)]
    if len(logical_numpy) <= 1:
        return list(quant_shape_to_byte_shape(logical_numpy, gguf_type))
    return list(quant_shape_to_byte_shape(logical_numpy, gguf_type))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    for path in (TRUNK_GGUF, FRAQTL_GGUF):
        if not os.path.exists(path):
            print(f"ERROR: missing {path}")
            sys.exit(1)
    out_path = Path(OUTPUT_GGUF)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    # -------------------------------------------------------------------------
    # Phase 1: Read fraQtl blk.40 tensor bytes into memory
    # -------------------------------------------------------------------------
    print(f"=== Phase 1: load fraQtl blk.40 into memory ===")
    print(f"  head : {FRAQTL_GGUF}")
    from extract_blk40_minimal import read_gguf_tensor_info
    head_tensors, head_data_start, _, _ = read_gguf_tensor_info(FRAQTL_GGUF)
    print(f"  head  tensors: {len(head_tensors)}")

    head_blk40 = [t for t in head_tensors if t["name"].startswith(BLK40_PREFIX)]
    print(f"  head  blk.40 tensors: {len(head_blk40)}")

    f_head = open(FRAQTL_GGUF, "rb")
    blk40_bytes = {}
    for t in head_blk40:
        name = t["name"]
        abs_off = head_data_start + t["offset"]
        bs, ts = GGML_QUANT_SIZES.get(GGMLQuantizationType(t["ttype"]), (1, 1))
        n_elem = 1
        for d in t["dims"]:
            n_elem *= d
        n_bytes = (n_elem // bs) * ts
        f_head.seek(abs_off)
        data = f_head.read(n_bytes)
        if len(data) != n_bytes:
            print(f"  [ERROR] short read for {name}: got {len(data)} expected {n_bytes}")
            sys.exit(5)
        blk40_bytes[name] = (np.frombuffer(data, dtype=np.uint8), t["ttype"], t["dims"])
    f_head.close()
    total_blk40_bytes = sum(v[0].nbytes for v in blk40_bytes.values())
    print(f"  loaded blk.40 into memory: {total_blk40_bytes/1024/1024:.1f} MB")

    # -------------------------------------------------------------------------
    # Phase 2: Open UD-IQ3_XXS (trunk) and build output
    # -------------------------------------------------------------------------
    print(f"\n=== Phase 2: build output ===")
    print(f"  trunk: {TRUNK_GGUF}")
    r_trunk = GGUFReader(TRUNK_GGUF)
    print(f"  trunk tensors: {len(r_trunk.tensors)}")
    trunk_blk40 = {t.name for t in r_trunk.tensors if isinstance(t.name, str) and t.name.startswith(BLK40_PREFIX)}
    print(f"  trunk blk.40 tensors: {len(trunk_blk40)} (should be 0)")
    if trunk_blk40:
        print(f"  [ERROR] trunk already has blk.40 tensors; use replace-style graft instead")
        sys.exit(2)

    # Verify all blk.40 tensors from fraQtl have shapes we can append
    # (no shape match needed against trunk since trunk has none; just need fraQtl data)
    print(f"  all {len(blk40_bytes)} blk.40 tensors loaded from fraQtl")

    # -------------------------------------------------------------------------
    # Build output GGUF
    # -------------------------------------------------------------------------
    print(f"\n=== building output: {OUTPUT_GGUF} ===")
    import tempfile
    _orig_spooled = tempfile.SpooledTemporaryFile
    EXT_TMPDIR = "/Volumes/AlexZhuang/flashkv0516/_tmp"
    os.makedirs(EXT_TMPDIR, exist_ok=True)

    class _SpooledExt(_orig_spooled):
        def __init__(self, *args, **kwargs):
            kwargs["dir"] = EXT_TMPDIR
            kwargs["max_size"] = 0
            super().__init__(*args, **kwargs)

    tempfile.SpooledTemporaryFile = _SpooledExt
    writer = GGUFWriter(str(out_path), arch="llama", use_temp_file=True)

    print(f"  copying metadata from trunk (skipping block_count)...")
    n_meta, n_skip = copy_metadata(r_trunk, writer)
    print(f"  copied {n_meta} metadata fields (skipped {n_skip})")

    # Override block_count: 40 -> 41 (trunk has 40, we append blk.40)
    writer.add_uint32("qwen35moe.block_count", 41)
    print(f"  override qwen35moe.block_count = 41")

    # Add nextn_predict_layers = 1 (trunk UD-IQ3_XXS lacks this key; without it
    # llama.cpp treats blk.40 as a trunk layer and demands ssm_conv1d etc.)
    writer.add_uint32("qwen35moe.nextn_predict_layers", 1)
    print(f"  add qwen35moe.nextn_predict_layers = 1")

    f_trunk = open(TRUNK_GGUF, "rb")
    qtype_by_value = {v.value: v for v in GGMLQuantizationType}

    # tensor_plan: all trunk tensors (no blk.40) + append blk.40 from fraQtl
    tensor_plan = []
    for t in r_trunk.tensors:
        if not isinstance(t.name, str):
            continue
        if t.name.startswith(BLK40_PREFIX):
            # trunk has no blk.40, but guard anyway
            continue
        gguf_t = qtype_by_value.get(t.tensor_type)
        raw_shape = compute_byte_shape_numpy(t.shape, gguf_t)
        tensor_plan.append((t.name, raw_shape, t.tensor_type, "trunk", t, t.n_bytes))

    n_trunk = len(tensor_plan)
    # Append blk.40 tensors in fraQtl order
    for name, (data_arr, ttype_int, shape) in blk40_bytes.items():
        gguf_t = qtype_by_value.get(ttype_int)
        raw_shape = compute_byte_shape_numpy(shape, gguf_t)
        tensor_plan.append((name, raw_shape, ttype_int, "mem", data_arr, len(data_arr)))
    n_appended = len(tensor_plan) - n_trunk

    print(f"\n  tensor plan: {len(tensor_plan)} tensors")
    print(f"    from trunk (blk.0-39 + embed): {n_trunk}")
    print(f"    appended (blk.40 from fraQtl) : {n_appended}")

    print(f"\n  registering tensors + writing data...")
    total_bytes = 0
    t0_all = time.time()
    for i, (name, raw_shape, ttype_int, src, src_val, n_bytes) in enumerate(tensor_plan):
        gguf_t = qtype_by_value.get(ttype_int)
        if gguf_t is None:
            print(f"  [ERROR] unknown tensor_type {ttype_int} for {name}")
            sys.exit(4)
        if src == "trunk":
            data = read_tensor_bytes(r_trunk, src_val, f_trunk)
        else:
            data = src_val
        writer.add_tensor(name, data, raw_shape=raw_shape, raw_dtype=gguf_t)
        total_bytes += len(data)
        if name.startswith("blk.40") or name in ("output.weight", "token_embd.weight") or i < 5 or i % 100 == 0:
            elapsed = time.time() - t0_all
            print(f"    [{i+1:4d}/{len(tensor_plan)}] {name:50s}  src={src}  type={gguf_t.name:8s}  bytes={len(data):>12d}  total={total_bytes/1024/1024:.1f}MB  elapsed={elapsed:.1f}s")

    print(f"\n  total bytes written: {total_bytes/1024/1024/1024:.2f} GB")
    print(f"  total elapsed: {time.time()-t0_all:.1f}s")

    print(f"\n  writing GGUF (header + KV + tensor info + tensor data)...")
    t0_write = time.time()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    print(f"  GGUF write complete in {time.time()-t0_write:.1f}s")
    f_trunk.close()

    sz = out_path.stat().st_size
    print(f"\n  output: {out_path}")
    print(f"  size  : {sz/1024/1024/1024:.2f} GB")
    print(f"\nDone. Next:")
    print(f"  llama-simple -m {OUTPUT_GGUF} -ngl 99 -expert-cache 4294967296 -no-mmap -t 8 -s 7 -n 128 -p '<prompt>'")


if __name__ == "__main__":
    main()
