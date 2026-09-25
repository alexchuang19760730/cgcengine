#!/usr/bin/env python3
"""Are one expert's three weight segments contiguous in the GGUF file?

This is the geometry half of the `cgc-io-request-shape` judgement for the "merge the N
preads of one expert into one" lever. It is cheap (header only, ~0 I/O) and it can kill
the lever outright, so it runs BEFORE any timing work.

Why it can kill the lever: merging is paid for in bytes. If the three segments of one
expert sit far apart, one merged pread has to swallow the whole span, and the byte bill
is `span / useful` -- at 260x no plausible syscall saving pays for it.

    python3 scripts/check/expert_file_layout.py            # default model
    python3 scripts/check/expert_file_layout.py --layer 0 --expert 0

⚠ GGUF v3 value types: ARRAY=9 and UINT64=10. `Backup/phase_decomp/parse_expert_geometry.py`
writes UINT64 as 9 (collides with ARRAY); it happens to work because no UINT64 KV appears
in this file, but do not copy that mapping.
"""
from __future__ import annotations

import argparse
import struct
import sys

MAGIC = b"GGUF"
DEFAULT_MODEL = ("models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf")

# GGUF v3 value types (ggml/src/gguf.cpp)
T_UINT32, T_INT32, T_FLOAT32 = 4, 5, 6
T_BOOL, T_STRING, T_ARRAY = 7, 8, 9
T_UINT64, T_INT64, T_FLOAT64 = 10, 11, 12


class R:
    def __init__(self, f):
        self.f = f

    def scalar(self, vtype):
        f = self.f
        if vtype == T_UINT32:
            return struct.unpack("<I", f.read(4))[0]
        if vtype in (T_INT32,):
            return struct.unpack("<i", f.read(4))[0]
        if vtype == T_FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        if vtype == T_UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        if vtype in (T_INT64,):
            return struct.unpack("<q", f.read(8))[0]
        if vtype == T_FLOAT64:
            return struct.unpack("<d", f.read(8))[0]
        if vtype == T_BOOL:
            return f.read(1) != b"\x00"
        if vtype == T_STRING:
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")
        raise ValueError(f"unhandled scalar type {vtype}")

    def value(self, vtype):
        if vtype == T_ARRAY:
            et, n = struct.unpack("<IQ", self.f.read(12))
            return [self.value(et) for _ in range(n)]
        return self.scalar(vtype)


def read_header(path: str) -> tuple[dict, int]:
    """(tensor dict name -> (offset, nbytes, dims), alignment) from the GGUF header."""
    with open(path, "rb") as f:
        r = R(f)
        if f.read(4) != MAGIC:
            raise SystemExit(f"not a GGUF file: {path}")
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        alignment = 32
        meta = {}
        for _ in range(n_kv):
            key = r.scalar(T_STRING)
            vtype = struct.unpack("<I", f.read(4))[0]
            meta[key] = r.value(vtype)
        if "general.alignment" in meta:
            alignment = int(meta["general.alignment"])

        tensors = {}
        for _ in range(n_tensors):
            name = r.scalar(T_STRING)
            ndims = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack("<%dQ" % ndims, f.read(8 * ndims))
            ttype = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            nbytes = 1
            for d in dims:
                nbytes *= d
            tensors[name] = {"offset": off, "dims": dims, "type": ttype, "nelem": nbytes}
        return tensors, alignment, version


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def analyse(path: str, layer: int, expert: int) -> dict:
    tensors, alignment, version = read_header(path)
    # Expert weight tensors are quantised, so `nelem` is NOT bytes; use the on-disk span
    # between consecutive experts of the same tensor instead (that IS bytes).
    kinds = ["ffn_down_exps", "ffn_gate_exps", "ffn_up_exps"]
    found = {}
    for k in kinds:
        name = f"blk.{layer}.{k}.weight"
        if name not in tensors:
            continue
        t = tensors[name]
        dims = t["dims"]
        n_experts = dims[-1] if len(dims) == 3 else dims[0]
        # byte size of one expert within this tensor = total span / n_experts
        # (total span taken from the next tensor's offset when available)
        found[k] = {"offset": t["offset"], "dims": dims, "n_experts": n_experts,
                    "nelem": t["nelem"]}

    if not found:
        raise SystemExit(f"no expert tensors for layer {layer} in {path}")

    # Derive per-tensor byte size by looking at the file: the next tensor after this one.
    names_sorted = sorted(tensors.items(), key=lambda kv: kv[1]["offset"])
    spans = {}
    for i, (nm, t) in enumerate(names_sorted):
        if i + 1 < len(names_sorted):
            spans[nm] = names_sorted[i + 1][1]["offset"] - t["offset"]
    for k in found:
        nm = f"blk.{layer}.{k}.weight"
        total = spans.get(nm)
        if total is None:
            # last tensor in file: fall back to file size
            import os
            total = os.path.getsize(path) - found[k]["offset"]
        found[k]["total_bytes"] = total
        found[k]["expert_bytes"] = total // found[k]["n_experts"]

    segs = []
    for k in kinds:
        if k not in found:
            continue
        f = found[k]
        segs.append({
            "kind": k,
            "offset": f["offset"] + expert * f["expert_bytes"],
            "bytes": f["expert_bytes"],
        })
    segs.sort(key=lambda s: s["offset"])
    useful = sum(s["bytes"] for s in segs)
    span = (segs[-1]["offset"] + segs[-1]["bytes"]) - segs[0]["offset"] if segs else 0
    return {
        "version": version, "alignment": alignment, "layer": layer, "expert": expert,
        "segments": segs, "useful_bytes": useful, "span_bytes": span,
        "merge_cost_x": (span / useful) if useful else None,
    }


def self_test() -> int:
    ok = []

    def chk(cond, label):
        ok.append(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")

    print("selftest expert_file_layout")
    chk(align_up(0, 32) == 0, "align_up 0 stays 0")
    chk(align_up(1, 32) == 32, "align_up 1 -> 32")
    chk(align_up(32, 32) == 32, "align_up 32 stays 32")
    chk(align_up(33, 32) == 64, "align_up 33 -> 64")
    # The merge arithmetic that decides the lever.
    segs = [{"offset": 0, "bytes": 100}, {"offset": 1000, "bytes": 100}]
    useful = sum(s["bytes"] for s in segs)
    span = (segs[-1]["offset"] + segs[-1]["bytes"]) - segs[0]["offset"]
    chk(useful == 200 and span == 1100, "span/useful arithmetic")
    chk(abs(span / useful - 5.5) < 1e-9, "merge cost = 5.5x for a 900 B gap")
    # Contiguous case must give exactly 1.0 -- that is the only case worth merging.
    segs2 = [{"offset": 0, "bytes": 100}, {"offset": 100, "bytes": 100}]
    u2 = sum(s["bytes"] for s in segs2)
    s2 = (segs2[-1]["offset"] + segs2[-1]["bytes"]) - segs2[0]["offset"]
    chk(s2 / u2 == 1.0, "contiguous segments -> merge cost 1.0x (no extra bytes)")
    print(f"selftest {sum(ok)}/{len(ok)}")
    return 0 if all(ok) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    a = analyse(args.model, args.layer, args.expert)
    print(f"GGUF v{a['version']}  alignment={a['alignment']}")
    print(f"layer {a['layer']}, expert {a['expert']} -- segments in file order:")
    prev_end = None
    for s in a["segments"]:
        gap = "" if prev_end is None else f"   gap = {s['offset'] - prev_end:>12,} B"
        print(f"  {s['kind']:14s} offset={s['offset']:>12,}  bytes={s['bytes']:>10,}{gap}")
        prev_end = s["offset"] + s["bytes"]
    print()
    print(f"useful bytes (3 segments) = {a['useful_bytes']:,}")
    print(f"span if merged into 1     = {a['span_bytes']:,}")
    if a["merge_cost_x"] is not None:
        print(f"merge cost                = {a['merge_cost_x']:.1f}x extra bytes")
        if a["merge_cost_x"] > 1.01:
            print("VERDICT: segments are NOT contiguous -- merging costs bytes, "
                  "only worth it if a syscall is worth >"
                  f"{(1 - 1 / a['merge_cost_x']) * 100:.1f}% of a read.")
        else:
            print("VERDICT: contiguous -- merging is free in bytes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
