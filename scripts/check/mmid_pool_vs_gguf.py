#!/usr/bin/env python3
"""Verify that the bytes mul_mat_id reads out of the expert-cache pool are the bytes
that actually live in the GGUF file.

The probe (CGC_MMID_MV_DBG=1) prints, for every routed MoE mul_mat_id node:
    src0 ne/ne02, nb02, the ids, and an FNV-1a of the first 4096 bytes of src0 at
    id*nb02 for the first four ids.
CGC_EXACT_PATH_DBG=1 prints the expert -> slot map for il <= 2.

This script joins the two and recomputes the same FNV-1a straight from the GGUF file
at <tensor_offset> + <expert>*<nb02>. Any mismatch means the pool does NOT hold the
expert that remap claims it holds -- which is exactly the failure that survives every
"data layer" check (pread succeeded, pool bytes non-zero, GPU readback matches CPU).

Usage:
    /opt/homebrew/bin/python3 scripts/check/mmid_pool_vs_gguf.py [log] [model.gguf]
"""
import re
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MMID_RE = re.compile(
    r"CGC-MMID name=(?P<name>\S+) type=(?P<type>\S+) \| "
    r"src0 ne=\[(?P<ne00>\d+),(?P<ne01>\d+),(?P<ne02>\d+)\] nb01=(?P<nb01>\d+) nb02=(?P<nb02>\d+) \| "
    r"ids ne=\[(?P<ne20>\d+),(?P<ne21>\d+)\] nbi1=(?P<nbi1>\d+) \| "
    r"dst ne=\[(?P<d0>\d+),(?P<d1>\d+)\] \| "
    r"ids=\[(?P<ids>[^\]]*)\] id_min=(?P<idmin>-?\d+) id_max=(?P<idmax>-?\d+) id_oob_vs_ne02=(?P<oob>\d+) \| "
    r"fp\((?P<fpb>\d+)B/row\) (?P<fps>.*?) \| src0_data=(?P<data>\S+) src0_offs=(?P<offs>\d+)"
)
FP_RE = re.compile(r"id(-?\d+):([0-9a-f]{16})")
# CGC-EXACT-POST: il=0
#   uni[0]=161 slot=3 (was -99)
POST_RE = re.compile(r"CGC-EXACT-POST: il=(\d+)\s*\n((?:\s+uni\[\d+\]=\d+ slot=-?\d+.*\n)+)")
UNI_RE = re.compile(r"uni\[(\d+)\]=(\d+) slot=(-?\d+)")

FNV_OFF = 1469598103934665603
FNV_PRIME = 1099511628211
MASK = (1 << 64) - 1


def fnv1a64(buf: bytes) -> int:
    h = FNV_OFF
    for b in buf:
        h ^= b
        h = (h * FNV_PRIME) & MASK
    return h


KIND_OF = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}


def main() -> int:
    log = sys.argv[1] if len(sys.argv) > 1 else None
    model = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        ROOT, "models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf")
    if log is None:
        d = os.path.join(ROOT, "Backup/mmid_probe")
        cands = sorted((f for f in os.listdir(d) if f.startswith("gather_") and f.endswith(".log")),
                       key=lambda f: os.path.getmtime(os.path.join(d, f)))
        if not cands:
            print("error: no Backup/mmid_probe/gather_*.log found", file=sys.stderr)
            return 1
        log = os.path.join(d, cands[-1])
    print(f"log   : {log}")
    print(f"model : {model}")

    text = open(log, encoding="utf-8", errors="replace").read()

    # expert -> slot, per layer (first occurrence wins: the first eval pass)
    slot_of = {}
    for m in POST_RE.finditer(text):
        il = int(m.group(1))
        if il in slot_of:
            continue
        tbl = {}
        for u in UNI_RE.finditer(m.group(2)):
            tbl[int(u.group(2))] = int(u.group(3))   # expert -> slot
        slot_of[il] = tbl
    print(f"parsed slot tables for layers: {sorted(slot_of)}")
    if not slot_of:
        print("note: no CGC-EXACT-POST expert->slot map in this log; raw-id (slab) records still check out")

    from gguf.gguf_reader import GGUFReader
    reader = GGUFReader(model)
    offs = {t.name: (t.data_offset, int(t.n_bytes)) for t in reader.tensors}
    fh = open(model, "rb")

    mmids = list(MMID_RE.finditer(text))
    print(f"parsed {len(mmids)} CGC-MMID records\n")
    if not mmids:
        print("!! no CGC-MMID records -- probe did not fire (env not set, or path not taken)")
        return 1

    n_bad = n_checked = 0
    seen = set()
    print(f"{'node':16s} {'il':>3s} {'nb02':>8s} {'id':>4s} {'expert':>6s} {'pool_fp':>18s} {'gguf_fp':>18s}  verdict")
    print("-" * 100)
    for m in mmids:
        name = m.group("name")
        mm = re.match(r"ffn_moe_(gate|up|down)-(\d+)$", name)
        if not mm:
            continue
        kind, il = mm.group(1), int(mm.group(2))
        if il not in slot_of:
            slot_of[il] = {}   # slab path: ids are raw, no slot table needed
        nb02 = int(m.group("nb02"))
        fpb = int(m.group("fpb"))
        tname = f"blk.{il}.{KIND_OF[kind]}.weight"
        if tname not in offs:
            continue
        t_off, t_nbytes = offs[tname]
        n_expert_file = t_nbytes // nb02 if nb02 else 0

        # Two possible id conventions:
        #   ne02 == n_expert  -> ids are RAW expert ids (M2 whole-layer slab path, no remap leaf)
        #   ne02 == n_slots   -> ids are slot indices (pool path), need the expert->slot table
        raw_ids = (m.group("ne02") and int(m.group("ne02")) >= n_expert_file)
        inv = {s: e for e, s in slot_of[il].items()}   # slot -> expert
        for idstr, fp in FP_RE.findall(m.group("fps")):
            slot = int(idstr)
            if slot < 0:
                continue
            key = (name, slot)
            if key in seen:
                continue
            seen.add(key)
            exp = slot if raw_ids else inv.get(slot)
            if exp is None:
                print(f"{name:16s} {il:3d} {nb02:8d} {slot:4d} {'?':>6s} {fp:>18s} {'-':>18s}  slot not in union map")
                continue
            fh.seek(t_off + exp * nb02)
            buf = fh.read(fpb)
            gfp = "%016x" % fnv1a64(buf)
            n_checked += 1
            ok = (gfp == fp)
            if not ok:
                n_bad += 1
            print(f"{name:16s} {il:3d} {nb02:8d} {slot:4d} {exp:6d} {fp:>18s} {gfp:>18s}  "
                  f"{'OK' if ok else '*** MISMATCH ***'}"
                  + (f"  (n_expert_file={n_expert_file})" if not ok else ""))

    print("-" * 100)
    print(f"checked {n_checked} expert rows, mismatches: {n_bad}")
    if n_bad:
        print("\nVERDICT: the pool does NOT contain the expert remap points at.")
        print("         -> bug is in the fill / slot assignment, not in the kernel or the ids.")
    elif n_checked:
        print("\nVERDICT: every probed pool row is byte-identical to the GGUF.")
        print("         -> pool contents are correct; look at ids/geometry or the kernel instead.")
    return 0 if n_bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
