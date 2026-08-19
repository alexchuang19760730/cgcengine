#!/usr/bin/env python3
"""Append lm_head.weight (independent, not tied) to an existing r4 model_weights.bin.
Original repack used key 'model.language_model.lm_head.weight' but weight_map stores
'lm_head.weight' -> silently skipped -> random lm_head -> top1-acc=0%.

Streaming approach: rebuild header/index (few KB) + copy existing data blob by chunks
+ append lm_head data. Memory stays small regardless of 4.9GB resident size.
"""
import json, struct, shutil, sys
from pathlib import Path

GTURBO = Path("/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-r4.gturbo")
HF = "/Volumes/AlexZhuang/qwen36-hf"
BIN = GTURBO / "model_weights.bin"
NEW = GTURBO / "model_weights.bin.new"

sys.path.insert(0, "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/qwen36-repack")
from resident_writer import read_resident_bin, HEADER_BYTES, ENTRY_BYTES, PAGE

# 1) load lm_head (fp16 encode)
from safetensors.torch import load_file
import numpy as np
idx = json.load(open(f"{HF}/model.safetensors.index.json"))
wm = idx["weight_map"]
key = "lm_head.weight"
shard = wm[key]
print(f"lm_head.weight in {shard}", flush=True)
sd = load_file(f"{HF}/{shard}")
t = sd[key].float().numpy().astype(np.float16)
lm_data = t.tobytes()
print(f"lm_head: shape={t.shape} bytes={len(lm_data)}", flush=True)

# 2) read existing index (no data)
info = read_resident_bin(BIN)
entries = list(info["entries"].items())
print(f"before: entries={len(entries)} index_size={info['indexSize']} resident={info['residentSize']}", flush=True)

old_index_size = info["indexSize"]
old_resident = info["residentSize"]

# 3) build new index: same 612 entries + lm_head, then name blob
tensors_meta = []
for name, e in entries:
    shp = [int(s) for s in e["shape"] if s != 0]
    tensors_meta.append((name, shp, e["dtype"], e["fileOffset"], e["sizeBytes"]))
tensors_meta.append((key, [int(x) for x in t.shape], 1, None, len(lm_data)))

entry_table_size = ENTRY_BYTES * len(tensors_meta)
name_base = HEADER_BYTES + entry_table_size
name_blob = b""
name_offsets = {}
for name, *_ in tensors_meta:
    name_offsets[name] = name_base + len(name_blob)
    name_blob += name.encode("utf-8")

raw_index_size = name_base + len(name_blob)
new_index_size = ((raw_index_size + PAGE - 1) // PAGE) * PAGE
pad = new_index_size - raw_index_size
print(f"index: {old_index_size} -> {new_index_size} (pad {pad})", flush=True)

# new data offsets: existing data stays at old positions; lm_head appended after old data
def new_file_offset(i):
    name, shp, dtype, old_off, size = tensors_meta[i]
    if old_off is not None:
        # keep absolute offset unchanged IF index size unchanged; else shift by delta
        return old_off + (new_index_size - old_index_size)
    return new_index_size + old_resident

# 4) assemble header + entry table + names
buf = bytearray()
buf += struct.pack("<QQQ", new_index_size, old_resident + len(lm_data), len(tensors_meta))
for i, (name, shp, dtype, old_off, size) in enumerate(tensors_meta):
    noff = name_offsets[name]
    nlen = len(name.encode("utf-8"))
    fo = new_file_offset(i)
    shp4 = (list(shp) + [0, 0, 0, 0])[:4]
    e = struct.pack("<IHBB", noff, nlen, dtype, 0)
    e += struct.pack("<QQ", fo, size)
    e += struct.pack("<4I", *shp4)
    e += struct.pack("<QQQQ", 0, 0, 0, 0)
    assert len(e) == ENTRY_BYTES
    buf += e
buf += name_blob
buf += b"\x00" * pad
assert len(buf) == new_index_size, f"{len(buf)} != {new_index_size}"

# 5) stream: write header+index, copy old data (old_index_size..end), append lm_head
with open(NEW, "wb") as out:
    out.write(buf)
    with open(BIN, "rb") as src:
        src.seek(old_index_size)
        total = 0
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
    out.write(lm_data)
print(f"written {NEW} total_bytes={new_index_size + old_resident + len(lm_data)}", flush=True)

# 6) swap
shutil.move(str(NEW), str(BIN))
print("swapped", flush=True)

# 7) verify
info2 = read_resident_bin(BIN)
print(f"after: entries={len(info2['entries'])} lm_head present={key in info2['entries']}", flush=True)
if key in info2["entries"]:
    e = info2["entries"][key]
    raw = BIN.read_bytes()[e["fileOffset"]:e["fileOffset"] + e["sizeBytes"]]
    arr = np.frombuffer(raw, dtype=np.float16).reshape([int(s) for s in e["shape"] if s != 0])
    print(f"  lm_head row0[:4]: {arr[0,:4]} match={np.allclose(arr, t, atol=0)}", flush=True)
print("=== PATCH DONE ===", flush=True)
