#!/usr/bin/env python3
"""
Path B step 2: Graft fraQtl Q4_K blk.40 tensors into Nail IQ3_XXS trunk.

  Input A (trunk source):  models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf
                           (architecture=qwen35moe, IQ3_XXS trunk, IQ2/Q2/Q3 blk.40)
  Input B (head source):   models/gguf/fraQtl/qwen36-35b-a3b-hi-fi-mtp-runtime.gguf
                           (Q4_K_M trunk, Q4_K blk.40 distilled/robust)
  Output:                  models/gguf/Qwen3.6-35B-A3B-IQ3XXS-trunk_Q4K-blk40.gguf
                           (IQ3_XXS trunk + Q4_K/Q5_K/Q6_K/Q8_0 blk.40 from fraQtl)

Strategy:
  - Open both source GGUFs as readers.
  - Open a new GGUF writer.
  - Copy metadata from Nail (trunk defines the model: 41 layers, 256 experts, etc.).
  - Walk Nail tensors in order:
      * If tensor name does NOT start with "blk.40." -> copy verbatim (raw bytes).
      * If tensor name starts with "blk.40."       -> pull same-named tensor from
                                                       fraQtl reader, copy raw bytes.
  - gguf-py's add_tensor(np.ndarray<uint8>, raw_shape, raw_dtype) is the right
    API: it accepts raw quantized bytes as a uint8 numpy array + the logical
    tensor shape + the GGML quantization type, and writes them to output
    verbatim (no dequant/requant).

Notes:
  - We do NOT dequant/requant. Tensors stay in their source quantization format.
    This works because both files share architecture=qwen35moe and identical
    blk.40 tensor names + shapes (verified via mtp_blk40_inventory.json).
"""
import os, sys, time
from pathlib import Path

GGUF_PY_PATH = "/Users/alexchuang/Documents/flashkv0516/temp/llama_roadB/llama.cpp-master/gguf-py"
sys.path.insert(0, GGUF_PY_PATH)
# Also add this script's dir so we can import extract_blk40_minimal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gguf import GGUFReader, GGUFWriter, GGUFValueType
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
from gguf.quants import quant_shape_to_byte_shape
import numpy as np

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
NAIL_GGUF   = "/Users/alexchuang/Documents/flashkv0516/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf"
FRAQTL_GGUF = "/Volumes/AlexZhuang/flashkv0516/models/gguf/fraQtl/qwen36-35b-a3b-hi-fi-mtp-runtime.gguf"
OUTPUT_GGUF = "/Users/alexchuang/Documents/flashkv0516/models/gguf/Qwen3.6-35B-A3B-IQ3XXS-trunk_Q4K-blk40.gguf"
BLK40_PREFIX = "blk.40."

# Metadata keys we let the writer regenerate rather than copy from source
SKIP_META_KEYS = {
    b'general.alignment',
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def copy_metadata(reader, writer):
    """Copy all KV metadata from reader to writer.

    Each entry in reader.fields maps fname (str or bytes) -> ReaderField with:
      - .types: list of GGUFValueType ints (one per value; arrays have multiple)
      - .data: list of int indices into field.parts (one per value)
      - .parts: list of np.ndarray buffers (header bytes + value bytes + type bytes)

    For STRING scalar: parts layout is [offset, uint8 bytes, type] and data points to bytes index.
    For ARRAY: first entry is ARRAY marker, subsequent entries are element values.
    For other scalars (UINT32 etc): parts[part_idx] is a 1-element np.ndarray.

    We handle scalar STRING (uint8 array of bytes -> str) and scalars of common types.
    Arrays: we copy INT32/UINT32/FLOAT32 element lists. STRING arrays are skipped
    (no clean writer API for them; tokenizer fields handled separately if needed).
    """
    copied = 0
    skipped = 0
    for fname, field in reader.fields.items():
        # Normalize name to str
        if isinstance(fname, bytes):
            key_str = fname.decode('utf-8', errors='replace')
        else:
            key_str = fname
        # Skip GGUF.* internal header fields (let writer regenerate)
        if key_str.startswith('GGUF.'):
            continue
        if key_str in ('general.alignment',):
            continue
        # Skip tokenizer data fields (will be added separately if needed)
        if key_str in ('tokenizer.ggml.tokens', 'tokenizer.ggml.scores',
                       'tokenizer.ggml.merges', 'tokenizer.ggml.token_type',
                       'tokenizer.ggml.bos_token_id', 'tokenizer.ggml.eos_token_id',
                       'tokenizer.ggml.add_bos_token', 'tokenizer.ggml.add_eos_token',
                       'tokenizer.ggml.unknown_token_id', 'tokenizer.ggml.padding_token_id',
                       'tokenizer.ggml.model', 'tokenizer.ggml.pre',
                       'tokenizer.chat_template'):
            # We'll let these through as they're regular fields
            pass

        # GGUFReader stores ARRAY fields as:
        #   types = [ARRAY, elem_type]   (len 2: type marker + elem type)
        #   data = [elem0_idx, elem1_idx, ...]  (one part_idx per element, NO marker)
        # So iterate ALL of field.data (do NOT skip data[0]).
        # Verified: Nail's rope.dimension_sections has data_len=4 = [11, 11, 10, 0].
        if not field.types:
            continue

        first_val_type = field.types[0]
        if first_val_type == GGUFValueType.ARRAY:
            elem_type = field.types[1] if len(field.types) > 1 else None
            elements = []
            for pi in field.data:  # ALL entries are element part-indices
                arr = field.parts[pi]
                if elem_type == GGUFValueType.STRING:
                    # arr is uint8 array of bytes for one string
                    elements.append(arr.tobytes().decode('utf-8', errors='replace'))
                elif arr.ndim == 0 or arr.size == 1:
                    elements.append(arr.item())
                else:
                    # shouldn't happen for scalar array elements
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

        # Scalar field: exactly one (val_type, part_idx)
        val_type = field.types[0]
        part_idx = field.data[0]
        arr = field.parts[part_idx]
        try:
            if val_type == GGUFValueType.STRING:
                # arr is uint8 array of UTF-8 bytes for the string value
                v_str = arr.tobytes().decode('utf-8', errors='replace') if arr.dtype == np.uint8 else str(arr.item())
                writer.add_string(key_str, v_str)
            else:
                # Numeric scalar: arr is 1-element ndarray
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


# -----------------------------------------------------------------------------
# Main graft
# -----------------------------------------------------------------------------

def main():
    for path in (NAIL_GGUF, FRAQTL_GGUF):
        if not os.path.exists(path):
            print(f"ERROR: missing {path}")
            sys.exit(1)
    out_path = Path(OUTPUT_GGUF)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    # -------------------------------------------------------------------------
    # Phase 1: Read fraQtl blk.40 tensor bytes into memory, then free fraQtl
    # (saves 20GB disk so we have room for the ~14GB output)
    # -------------------------------------------------------------------------
    print(f"=== Phase 1: load fraQtl blk.40 into memory ===")
    print(f"  head : {FRAQTL_GGUF}")
    # Use minimal reader to avoid OOM from tokenizer fields on 21.9GB file
    from extract_blk40_minimal import read_gguf_tensor_info
    head_tensors, head_data_start, _, _ = read_gguf_tensor_info(FRAQTL_GGUF)
    print(f"  head  tensors: {len(head_tensors)}")

    head_blk40 = [t for t in head_tensors if t['name'].startswith(BLK40_PREFIX)]
    print(f"  head  blk.40 tensors: {len(head_blk40)}")

    f_head = open(FRAQTL_GGUF, "rb")
    blk40_bytes = {}  # name -> (np.ndarray(uint8), tensor_type, shape)
    for t in head_blk40:
        name = t['name']
        abs_off = head_data_start + t['offset']
        # Compute n_bytes from dims + ttype
        bs, ts = GGML_QUANT_SIZES.get(GGMLQuantizationType(t['ttype']), (1, 1))
        n_elem = 1
        for d in t['dims']:
            n_elem *= d
        n_bytes = (n_elem // bs) * ts
        f_head.seek(abs_off)
        data = f_head.read(n_bytes)
        if len(data) != n_bytes:
            print(f"  [ERROR] short read for {name}: got {len(data)} expected {n_bytes}")
            sys.exit(5)
        blk40_bytes[name] = (np.frombuffer(data, dtype=np.uint8), t['ttype'], t['dims'])
    f_head.close()
    total_blk40_bytes = sum(v[0].nbytes for v in blk40_bytes.values())
    print(f"  loaded blk.40 into memory: {total_blk40_bytes/1024/1024:.1f} MB")

    # Now safe to delete fraQtl GGUF to free disk space
    print(f"  keeping fraQtl GGUF on external disk (no delete)")

    # -------------------------------------------------------------------------
    # Phase 2: Open Nail (trunk) and build output
    # -------------------------------------------------------------------------
    print(f"\n=== Phase 2: build output ===")
    print(f"  trunk: {NAIL_GGUF}")
    r_trunk = GGUFReader(NAIL_GGUF)
    print(f"  trunk tensors: {len(r_trunk.tensors)}")

    trunk_blk40 = {t.name: t for t in r_trunk.tensors if isinstance(t.name, str) and t.name.startswith(BLK40_PREFIX)}
    print(f"  trunk blk.40 tensors: {len(trunk_blk40)}")

    # Verify all trunk blk.40 tensors have a replacement loaded
    missing = [n for n in trunk_blk40 if n not in blk40_bytes]
    if missing:
        print(f"  [ERROR] missing blk.40 replacements for:")
        for n in missing:
            print(f"    {n}")
        sys.exit(2)
    for n, t in trunk_blk40.items():
        _, _, h_shape = blk40_bytes[n]
        if list(t.shape) != h_shape:
            print(f"  [ERROR] shape mismatch {n}: trunk={list(t.shape)} head={h_shape}")
            sys.exit(3)
    print(f"  all {len(trunk_blk40)} blk.40 tensors present + shape-matched")

    # -----------------------------------------------------------------------------
    # Build output GGUF
    # -----------------------------------------------------------------------------
    print(f"\n=== building output: {OUTPUT_GGUF} ===")
    # use_temp_file=True: stream tensors to a temp file instead of accumulating in RAM
    # (13GB of tensors in RAM would OOM M4 Max 16GB)
    # Patch SpooledTemporaryFile to use external disk (internal only has ~5GB free)
    import tempfile
    _orig_spooled = tempfile.SpooledTemporaryFile
    EXT_TMPDIR = '/Volumes/AlexZhuang/flashkv0516/_tmp'
    os.makedirs(EXT_TMPDIR, exist_ok=True)

    class _SpooledExt(_orig_spooled):
        def __init__(self, *args, **kwargs):
            kwargs['dir'] = EXT_TMPDIR
            # Force rollover to disk immediately by setting max_size=0
            kwargs['max_size'] = 0
            super().__init__(*args, **kwargs)

    tempfile.SpooledTemporaryFile = _SpooledExt
    writer = GGUFWriter(str(out_path), arch="llama", use_temp_file=True)

    print(f"  copying metadata from trunk...")
    n_meta, n_skip = copy_metadata(r_trunk, writer)
    print(f"  copied {n_meta} metadata fields (skipped {n_skip})")

    # Open trunk file for streaming non-blk.40 tensor data
    f_trunk = open(NAIL_GGUF, "rb")

    # qtype lookup
    qtype_by_value = {v.value: v for v in GGMLQuantizationType}

    def compute_byte_shape_numpy(ggml_dims, gguf_type):
        # GGUFReader.ReaderTensor.shape and the minimal reader's t['dims']
        # both expose the dims as STORED in the file (GGML column-major
        # convention: first dim is the LAST numpy axis). GGUFWriter writes
        # dims into the file by REVERSING the user-supplied raw_shape
        # (gguf_writer.py: ti.shape[n_dims - 1 - j]). Therefore raw_shape
        # MUST be in numpy (row-major) convention or the output dims get
        # double-reversed (bug: token_embd.weight came out as
        # [248320, 2048] instead of [2048, 248320]).
        #
        # raw_dtype is always set below and the data is np.uint8, so
        # add_tensor_info calls quant_shape_from_byte_shape which expects
        # the BYTE shape (last dim = bytes per row). quant_shape_to_byte_shape
        # works for ALL GGML types because GGML_QUANT_SIZES has block_size=1
        # for non-quantized types (F32/F16/BF16/I8...).
        logical_numpy = [int(d) for d in reversed(ggml_dims)]
        if len(logical_numpy) <= 1:
            # 1-D: byte shape is just [n_bytes] (= n_elements * type_size).
            # Reconstruct via quant_shape_to_byte_shape on the single axis.
            return list(quant_shape_to_byte_shape(logical_numpy, gguf_type))
        return list(quant_shape_to_byte_shape(logical_numpy, gguf_type))

    # Plan: list of (name, raw_shape, ttype_int, source, src_val, n_bytes)
    tensor_plan = []
    for t in r_trunk.tensors:
        if not isinstance(t.name, str):
            continue
        if t.name.startswith(BLK40_PREFIX):
            data_arr, ttype_int, shape = blk40_bytes[t.name]
            gguf_t = qtype_by_value.get(ttype_int)
            raw_shape = compute_byte_shape_numpy(shape, gguf_t)
            tensor_plan.append((t.name, raw_shape, ttype_int, "mem", data_arr, len(data_arr)))
        else:
            gguf_t = qtype_by_value.get(t.tensor_type)
            raw_shape = compute_byte_shape_numpy(t.shape, gguf_t)
            tensor_plan.append((t.name, raw_shape, t.tensor_type, "trunk", t, t.n_bytes))

    n_from_trunk = sum(1 for _,_,_,s,_,_ in tensor_plan if s == "trunk")
    n_from_head  = sum(1 for _,_,_,s,_,_ in tensor_plan if s == "mem")
    print(f"\n  tensor plan: {len(tensor_plan)} tensors")
    print(f"    from trunk: {n_from_trunk}")
    print(f"    from mem  : {n_from_head}")

    print(f"\n  registering tensors + writing data...")
    total_bytes = 0
    t0_all = time.time()
    for i, (name, raw_shape, ttype_int, src, src_val, n_bytes) in enumerate(tensor_plan):
        gguf_t = qtype_by_value.get(ttype_int)
        if gguf_t is None:
            print(f"  [ERROR] unknown tensor_type {ttype_int} for {name}")
            sys.exit(4)
        if src == "trunk":
            # src_val is the ReaderTensor from Nail
            data = read_tensor_bytes(r_trunk, src_val, f_trunk)
        else:
            # src_val is np.ndarray(uint8) already in memory
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
    print(f"  llama-server -m {OUTPUT_GGUF} -ngl 99 -expert-cache 4GiB --spec-type draft-mtp --spec-draft-n-max 2")


if __name__ == "__main__":
    main()
