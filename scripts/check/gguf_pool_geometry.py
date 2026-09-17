#!/usr/bin/env python3
"""Expert-pool geometry of a GGUF: per-slot bytes, the BINDING layer, and the slot capacity.

WHY THIS EXISTS
---------------
The pool is not a raw byte budget -- it is a COMPUTED number of slots:

    per_slot_layer[il] = sum over that layer's *_exps tensors of ( row_size(type, ne0) * ne1 )
    per_slot          = MAX over ALL layers                 <-- includes the MTP layer
    capacity          = clamp( budget / (n_layers * per_slot), 8, 256 )

(`compute_l4_pool_capacity`; the formula and its two worked examples -- 118 slots for the shipped
artifact, 33 for the F16-head one -- are written out in `scripts/analyze_pool_geometry.py`, which
reproduces slot counts recorded in real server logs. `run_server.sh:411` records the same identity
for this family of models: `capacity = budget / (41 layers * per_slot 1.465MB) -> 8GiB = 143`.)
Note what `row_size(type, ne0) * ne1` is: the bytes of ONE EXPERT of one tensor. So `per_slot` is a
PER-EXPERT figure, not a per-layer one. Both are printed below because confusing them is easy and
the confusion is silent.

Two consequences are why this deserves a tool rather than a one-off:

  * `per_slot` is a MAX, so **one fat layer costs every layer its slots**. Which layer binds is
    therefore the most valuable number here, and it is not the layer with the most experts or the
    lowest precision -- just the fattest one. (2026-09-17: blk.39 of 41 is 30% fatter than the other
    37 trunk layers and single-handedly holds the slot count ~24% below what they would allow.)
  * because the same number is charged to all layers, "change the quantisation" and "fix the binding
    layer" are two views of one knob.

WHY IT READS ONLY THE HEADER
----------------------------
The shipped model is ~13 GB and the question ("how many slots would I get at 6 GiB?") has to be
answerable while a measurement is running, so this parses the header and nothing else: no tensor
data, no numpy, no ggml. Tensor sizes come from the quant-size table below.

That table MIRRORS ggml's trait table. `scripts/gguf_retensor.py types` enumerates the same table
from ggml itself, so if the two ever disagree, that one is authoritative. An unknown type id is a
hard error rather than a size of 0: a silently-zero tensor would move `per_slot`, and `per_slot` is
the whole answer.

Usage:
    python3 scripts/check/gguf_pool_geometry.py [--gguf PATH] [--budgets 6GiB,8GiB,10GiB]
                                               [--json OUT] [--normalize-binding] [--layer-caps]

`--normalize-binding` is the MODEL route: it reports what the capacity would be if the thick layers
were requantised down to the typical one. `--layer-caps` is the CONFIG route: it prints the
`CGC_SERVER_LAYER_CAPS` string that gives every layer its own slot count from the SAME bytes, with
no model change and no rebuild. They are alternatives, and the config route is the lossless one.

With no --gguf, the single `models/gguf/*denseIQ4X*.gguf` is used (the model run_server.sh ships);
if there is not exactly one, the candidates are listed and nothing is guessed.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import struct
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(REPO, "models", "gguf")

# --- ggml's type table -------------------------------------------------------------------------
# (block, bytes) per ggml_type. ids 4 and 5 are the removed Q4_2/Q4_3 and must stay absent so that a
# file carrying them fails loudly instead of being sized as something else.
QTYPE = {
    0:  ("F32",      1,   4),
    1:  ("F16",      1,   2),
    2:  ("Q4_0",     32,  18),
    3:  ("Q4_1",     32,  20),
    6:  ("Q5_0",     32,  22),
    7:  ("Q5_1",     32,  24),
    8:  ("Q8_0",     32,  34),
    9:  ("Q8_1",     32,  36),
    10: ("Q2_K",     256, 84),
    11: ("Q3_K",     256, 110),
    12: ("Q4_K",     256, 144),
    13: ("Q5_K",     256, 176),
    14: ("Q6_K",     256, 210),
    15: ("Q8_K",     256, 292),
    16: ("IQ2_XXS",  256, 66),
    17: ("IQ2_XS",   256, 74),
    18: ("IQ3_XXS",  256, 98),
    19: ("IQ1_S",    256, 50),
    20: ("IQ4_NL",   32,  18),
    21: ("IQ3_S",    256, 110),
    22: ("IQ2_S",    256, 82),
    23: ("IQ4_XS",   256, 136),
    24: ("I8",       1,   1),
    25: ("I16",      1,   2),
    26: ("I32",      1,   4),
    27: ("I64",      1,   8),
    28: ("F64",      1,   8),
    29: ("IQ1_M",    256, 56),
    30: ("BF16",     1,   2),
    31: ("Q4_0_4_4", 32,  18),
    32: ("Q4_0_4_8", 32,  18),
    33: ("Q4_0_8_8", 32,  18),
    34: ("TQ1_0",    256, 54),
    35: ("TQ2_0",    256, 66),
}

KV_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class BadGGUF(Exception):
    pass


# --- header parsing ----------------------------------------------------------------------------

def _rd(f, n):
    b = f.read(n)
    if len(b) != n:
        raise BadGGUF("truncated header")
    return b


def parse_header(path):
    """-> (version, {name: (type_id, shape, offset)}, file_type, meta).

    GGUF stores dims fastest-first, i.e. exactly ggml's ne order, so `shape[2]` is the expert count.
    A v2 file uses u32 for the counts; v3 uses u64. `meta` carries only `arch` and `n_nextn`
    (NextN / MTP layer count) -- the two keys that decide which layers are trunk.
    """
    with open(path, "rb") as f:
        if _rd(f, 4) != b"GGUF":
            raise BadGGUF("not a GGUF file (bad magic)")
        (version,) = struct.unpack("<I", _rd(f, 4))
        if version == 2:
            n_tensors, n_kv = struct.unpack("<II", _rd(f, 8))
            csz = 4
        elif version == 3:
            n_tensors, n_kv = struct.unpack("<QQ", _rd(f, 16))
            csz = 8
        else:
            raise BadGGUF(f"unsupported GGUF version {version}")

        def rd_str():
            (n,) = struct.unpack("<Q", _rd(f, 8))
            return _rd(f, n).decode("utf-8", "replace")

        def skip_val(t):
            if t == 8:
                rd_str()
            elif t == 9:
                (et,) = struct.unpack("<I", _rd(f, 4))
                (n,) = struct.unpack("<Q", _rd(f, 8))
                for _ in range(n):
                    skip_val(et)
            elif t in KV_SIZES:
                _rd(f, KV_SIZES[t])
            else:
                raise BadGGUF(f"unknown metadata value type {t}")

        # Only two metadata keys are kept: the architecture (to build the nextn key) and the
        # nextn count. The loader needs the same pair to decide which layers are trunk -- and
        # `blk.<max>` being an MTP layer is exactly what makes "cap for layer 40" a different
        # question from "cap for the trunk".
        file_type = None
        arch = None
        n_nextn = 0
        for _ in range(n_kv):
            key = rd_str()
            (t,) = struct.unpack("<I", _rd(f, 4))
            # Every branch that CONSUMES the value must be in this chain. An extra `if` after the
            # chain would read the value twice and desynchronise the rest of the header (which here
            # would not fail loudly -- it would parse garbage tensor names and report a wrong
            # geometry, the one answer this tool exists to produce).
            if key == "general.file_type" and t == 4:
                (file_type,) = struct.unpack("<i", _rd(f, 4))
            elif key == "general.architecture" and t == 8:
                arch = rd_str()
            elif key.endswith(".nextn_predict_layers") and t == 4:
                (n_nextn,) = struct.unpack("<I", _rd(f, 4))
            else:
                skip_val(t)

        tensors = {}
        for _ in range(n_tensors):
            name = rd_str()
            (nd,) = struct.unpack("<I", _rd(f, 4))
            dims = struct.unpack("<" + ("I" * nd if csz == 4 else "Q" * nd), _rd(f, csz * nd))
            (tid,) = struct.unpack("<I", _rd(f, 4))
            (off,) = struct.unpack("<Q", _rd(f, 8))
            tensors[name] = (tid, dims, off)
    return version, tensors, file_type, {"arch": arch, "n_nextn": n_nextn}


def n_bytes(tid, dims):
    if tid not in QTYPE:
        raise BadGGUF(f"type id {tid} is not in the table -- run `scripts/gguf_retensor.py types` "
                      f"to enumerate ggml's own table and add it (sizing it as 0 would silently "
                      f"move per_slot, which is the whole answer)")
    _, blk, sz = QTYPE[tid]
    n = 1
    for d in dims:
        n *= d
    if n % blk:
        raise BadGGUF(f"element count {n} is not a multiple of the block size {blk}")
    return n // blk * sz


def tname(tid):
    return QTYPE[tid][0] if tid in QTYPE else f"?{tid}"


# --- geometry ----------------------------------------------------------------------------------

EXPERT_PROJ = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


def expert_layers(tensors):
    """-> {il: {proj: (tid, dims, bytes)}} for every layer that has routed-expert stacks."""
    out = collections.defaultdict(dict)
    for name, (tid, dims, _off) in tensors.items():
        if not name.startswith("blk.") or not name.endswith(".weight"):
            continue
        body = name[len("blk."):-len(".weight")]
        il_s, _, proj = body.partition(".")
        if proj not in EXPERT_PROJ:
            continue
        try:
            il = int(il_s)
        except ValueError:
            continue
        out[il][proj] = (tid, dims, n_bytes(tid, dims))
    return out


def geometry(per_layer):
    """-> list of (il, per_layer_bytes, n_expert, sig) sorted by layer, and the MAX."""
    rows = []
    for il in sorted(per_layer):
        d = per_layer[il]
        total = sum(v[2] for v in d.values())
        n_expert = 0
        for _tid, dims, _b in d.values():
            if len(dims) >= 3:
                n_expert = dims[2] if n_expert in (0, dims[2]) else n_expert
        sig = " ".join(f"{p.split('_')[1]}={tname(d[p][0])}" for p in EXPERT_PROJ if p in d)
        rows.append((il, total, n_expert, sig))
    return rows


def capacity(budget, n_layers, per_slot):
    return max(8, min(256, int(budget // (n_layers * per_slot))))


def parse_budget(s):
    t = s.strip().lower()
    mult = 1
    for suf, m in (("gib", 1 << 30), ("gb", 10 ** 9), ("mib", 1 << 20), ("mb", 10 ** 6), ("b", 1)):
        if t.endswith(suf):
            t, mult = t[: -len(suf)], m
            break
    return int(float(t) * mult)


def find_default_gguf():
    c = sorted(glob.glob(os.path.join(MODEL_DIR, "*denseIQ4X*.gguf")))
    if len(c) == 1:
        return c[0]
    return None


# ---------------------------------------------------------------------------------------------
# Per-layer slot allocation: the CONFIG route (no model bytes touched, no rebuild).
#
# Why the uniform formula wastes the budget: `per_slot` is a MAX over layers, and the loader
# sizes EVERY layer's pool region to the SAME slot count (`t_meta.ne[2] = cgc_layer_cap(il, cap)`,
# llama-model-loader.cpp:1502). So one thick layer is billed to all 41 and the thin layers cannot
# spend the slack -- with the default 10 GiB budget the pool actually occupies ~7.85 GiB.
#
# The machinery to vary it already exists in HEAD and is already exercised on every run:
# `LLAMA_EXPERT_CACHE_LAYER_CAPS` ("start-end:cap;...") is read by BOTH sides -- the loader's
# ne[2] shrink (cgc_layer_cap) and the cache's n_slots_l vectors (llama-expert-cache.cpp:3136) --
# and run_server.sh already sets it to "40-40:256" on every launch. So this is one env string.
#
# The rule is neither "equal slots" (that is today) nor "equal bytes with no floor". A layer's cap
# must NEVER drop below today's uniform value, because the pool path requires the top-k union of
# the selected experts to fit in that layer's slots, and the failure mode of too few slots is a
# SILENT read of nothing (measured on Edge0: 33 slots logged 585 "buffer is nil" errors, 0/48).
# So we solve
#
#     maximise L subject to  sum_il max(base, min(ceiling, L // ps_il)) * ps_il  <=  avail
#
# with avail = budget - mtp_slots * mtp_ps. That is equal BYTES per trunk layer, clipped at the
# floor `base`, which makes the change strictly additive: thin layers go up, thick layers stay put.
#
# ★ TWO avails, and the difference is not cosmetic (measured 2026-09-17):
#   avail = budget slack   -> the pool grows into memory that was previously unused. On the
#                             memory-tight profile that slack is LOAD-BEARING: prefill250 holds
#                             ub 6144 and the server was SIGKILLed (Killed: 9) at load with the
#                             caps on, because what must fit in 16 GB is `pool + compute buffer`.
#   avail = today's pool    -> REDISTRIBUTE at constant memory. Same bytes, but they are now
#   (--keep-pool-bytes)        spread by per-layer cost instead of by the worst layer, so the thin
#                             layers still gain ~27% and the footprint does not move at all.
# The second is the one to try first whenever the machine is memory-tight.
# ---------------------------------------------------------------------------------------------
def allocate_caps(budget, per_slot_layer, base, mtp_layers, mtp_cap, ceiling=256, cap_bytes=None):
    """-> (caps_by_layer, total_bytes, overflow). See the note above for the objective."""
    mtp = set(mtp_layers)
    caps = {il: base for il in sorted(per_slot_layer)}
    for il in mtp:
        if il in caps:
            caps[il] = mtp_cap
    avail = budget - sum(mtp_cap * per_slot_layer[il] for il in mtp if il in per_slot_layer)
    if cap_bytes is not None:
        # constant-memory mode: the trunk may only spend what the TRUNK spends today
        avail = cap_bytes - sum(mtp_cap * per_slot_layer[il] for il in mtp if il in per_slot_layer)
    trunk = [il for il in sorted(per_slot_layer) if il not in mtp]
    if not trunk:
        return caps, sum(caps[il] * per_slot_layer[il] for il in caps), 0

    def used(L):
        return sum(max(base, min(ceiling, L // per_slot_layer[il])) * per_slot_layer[il]
                   for il in trunk)

    overflow = 1 if used(0) > avail else 0   # even the floor does not fit: keep today's shape
    hi = ceiling * max(per_slot_layer[il] for il in trunk)
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if used(mid) <= avail:
            lo = mid
        else:
            hi = mid - 1
    for il in trunk:
        caps[il] = max(base, min(ceiling, lo // per_slot_layer[il]))
    return caps, sum(caps[il] * per_slot_layer[il] for il in caps), overflow


def emit_layer_caps(caps):
    """-> 'a-b:c;d-d:e' covering EVERY layer explicitly.

    Explicit coverage is the point: the parser keeps `def` for uncovered layers, so a string that
    leaves the thick layer to `def` happens to be right today and is silently wrong the moment the
    budget changes (def moves with the budget; the layer's role does not).
    """
    runs, start, prev = [], None, None
    for il in sorted(caps):
        if start is not None and il == prev + 1 and caps[il] == caps[prev]:
            prev = il
            continue
        if start is not None:
            runs.append((start, prev, caps[start]))
        start = prev = il
    if start is not None:
        runs.append((start, prev, caps[start]))
    # int() on the cap is not cosmetic: the consumer parses with sscanf("%u-%u:%u"), so a float
    # cap prints as "232.0" and is accepted only because %u stops at the '.'. Emitting an integer
    # keeps the string valid for any reader, including a stricter one.
    return ";".join(f"{a}-{b}:{int(c)}" if a != b else f"{a}-{a}:{int(c)}" for a, b, c in runs)


def parse_layer_caps(spec, default_cap, n_layers):
    """Python mirror of `cgc_layer_cap()` in llama-expert-cache.h -- used ONLY to self-check the
    string this tool emits, so the emitter and the C parser cannot drift apart silently.

    Semantics copied deliberately: segments are split on ';' or ','; a segment that does not parse
    as `<u>-<u>:<u>` is skipped; LATER matching segments override EARLIER ones; an uncovered layer
    keeps `default_cap`.
    """
    cur = [default_cap] * n_layers
    s = spec or ""
    p = 0
    while p < len(s):
        i = p

        def num():
            nonlocal i
            j = i
            while i < len(s) and s[i].isdigit():
                i += 1
            return int(s[j:i]) if i > j else None

        a = num()
        b = c = None
        if a is not None and i < len(s) and s[i] == '-':
            i += 1
            b = num()
            if b is not None and i < len(s) and s[i] == ':':
                i += 1
                c = num()
        if a is not None and b is not None and c is not None:
            for l in range(n_layers):
                if a <= l <= b:
                    cur[l] = c
        while p < len(s) and s[p] not in ";,":
            p += 1
        if p < len(s):
            p += 1
    return cur


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="gguf_pool_geometry.py",
        description="Compute per_slot bytes, the binding layer, and slot capacity for a GGUF.")
    ap.add_argument("--gguf", help="model file (default: the single models/gguf/*denseIQ4X*.gguf)")
    ap.add_argument("--budgets", default="6GiB,8GiB,10GiB",
                    help="comma separated; 6GiB / 6442450944 both work")
    ap.add_argument("--json", help="also write the result here")
    ap.add_argument("--normalize-binding", action="store_true",
                    help="ALSO print the capacity if the binding layer matched the median layer "
                         "(arithmetic on the same header, i.e. a scenario, not a measurement)")
    ap.add_argument("--layer-caps", action="store_true",
                    help="ALSO print the per-layer CGC_SERVER_LAYER_CAPS value for each budget "
                         "(the CONFIG route: same bytes, no model change, no rebuild)")
    ap.add_argument("--mtp-cap", type=int, default=256,
                    help="slots for the NextN/MTP layer(s); default 256 == run_server.sh's default")
    ap.add_argument("--keep-pool-bytes", action="store_true",
                    help="REDISTRIBUTE at constant pool bytes (== --slack 0). NOTE: for this model "
                         "that is a NO-OP -- today's uniform cap is already within 0.7%% of equal "
                         "bytes, so there is nothing to redistribute; the gain only exists in the "
                         "unused slack")
    ap.add_argument("--slack", default=None,
                    help="extra pool bytes the machine can afford, e.g. 512MiB. This is the real "
                         "knob: the whole M6 gain comes from spending slack, and the memory-tight "
                         "profile SIGKILLed at load when the full budget slack was spent")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.layer_caps and args.quiet:
        # --layer-caps is a report that lives inside the verbose block, so -q would turn it into a
        # silent no-op -- and "asked for it, got nothing" reads as "there is nothing to report".
        # Refuse loudly instead of printing nothing.
        ap.error("--layer-caps prints a report, so -q would silently suppress it; drop -q")

    path = args.gguf or find_default_gguf()
    if path is None:
        cands = sorted(glob.glob(os.path.join(MODEL_DIR, "*.gguf")))
        print("cannot pick a default model -- pass --gguf. candidates:", file=sys.stderr)
        for c in cands:
            print("  ", os.path.relpath(c, REPO), file=sys.stderr)
        return 2
    if not os.path.exists(path):
        print(f"no such file: {path}", file=sys.stderr)
        return 2

    try:
        version, tensors, file_type, meta = parse_header(path)
        per_layer = expert_layers(tensors)
        if not per_layer:
            print("no routed-expert stacks (*ffn_{gate,up,down}_exps.weight) in this file",
                  file=sys.stderr)
            return 3
        rows = geometry(per_layer)
    except BadGGUF as e:
        print(f"!! {path}: {e}", file=sys.stderr)
        return 3

    budgets = [parse_budget(b) for b in args.budgets.split(",") if b.strip()]
    n_layers = len(rows)
    per_exp = [(il, total / n if n else float("nan"), total, n, sig) for il, total, n, sig in rows]

    # Which layers are NextN / MTP. This is NOT cosmetic: the loader computes
    #   per_slot = MAX over TRUNK layers only   (llama-model-loader.cpp:1171-1174 skips il >= n_decoder)
    # precisely so the trunk's pool cannot depend on how the *head* happens to be stored, while
    # `denom` still counts every layer. Running the MAX over all layers happens to agree on the
    # shipped Nail model (its blk.40 is thinner than blk.39) and disagrees the moment a model ships
    # a THICKER head -- measured on Ornith, whose blk.40 is q8_0: per_slot 3,342,336 B (tool) vs
    # 1,769,472 B (loader), i.e. 78 slots reported where the engine would compute 148.
    n_total = max(per_layer) + 1
    n_nextn = int(meta.get("n_nextn") or 0)
    n_decoder = max(0, n_total - n_nextn)
    mtp_layers = [il for il in sorted(per_layer) if il >= n_decoder]
    trunk_exp = [r for r in per_exp if r[0] < n_decoder] or per_exp

    binding = max(trunk_exp, key=lambda r: r[1])
    per_slot = int(binding[1])
    rest = sorted((r for r in trunk_exp if r[0] != binding[0]), key=lambda r: -r[1])
    typical = sorted(r[1] for r in trunk_exp)[len(trunk_exp) // 2]
    head_over_trunk = [r for r in per_exp if r[0] >= n_decoder and r[1] > per_slot]

    def ils_label(ils):
        """[0..4, 7, 8] -> 'blk.0-4,blk.7-8'. Printing 37 layer numbers was the first version's
        bug: it is one group, and the group is the point."""
        runs, start, prev = [], ils[0], ils[0]
        for i in list(ils[1:]) + [None]:
            if i is not None and i == prev + 1:
                prev = i
                continue
            runs.append((start, prev))
            if i is not None:
                start = prev = i
        return ",".join(("blk.%d" % a) if a == b else ("blk.%d-%d" % (a, b)) for a, b in runs)

    # collapse identical layers for the printed table (consecutive runs are compressed)
    groups = collections.OrderedDict()
    for il, total, n, sig in sorted(rows):
        groups.setdefault((total, n, sig), []).append(il)

    out = {
        "gguf": os.path.relpath(path, REPO),
        "bytes": os.path.getsize(path),
        "gguf_version": version,
        "file_type": file_type,
        "n_tensors": len(tensors),
        "n_layers": n_layers,
        "per_slot_bytes": per_slot,
        "binding_layer": binding[0],
        "binding_layer_bytes": binding[2],
        "per_layer": [{"il": il, "bytes": t, "n_expert": n, "types": sig} for il, t, n, sig in rows],
        "capacity": [{"budget": b, "slots": capacity(b, n_layers, per_slot)} for b in budgets],
    }

    if not args.quiet:
        gib = 1 << 30
        print(f"model   : {out['gguf']}  ({out['bytes'] / gib:.2f} GiB, GGUF v{version}, "
              f"file_type={file_type} {tname(file_type) if file_type in QTYPE else '?'})")
        print(f"tensors : {len(tensors)} total, {sum(len(d) for d in per_layer.values())} "
              f"routed-expert, {n_layers} layers, n_expert={binding[3]}")
        tc = collections.Counter(tname(tid) for d in per_layer.values() for tid, _d, _b in d.values())
        print("types   : " + ", ".join(f"{k} x{v}" for k, v in sorted(tc.items(), key=lambda kv: -kv[1])))
        print()
        print(f"{'layer(s)':<26} {'n':>3} {'per-layer':>11} {'per-expert':>11}  types")
        for (total, n, sig), ils in groups.items():
            mark = " *" if binding[0] in ils else ""
            print(f"{ils_label(ils):<26} {len(ils):>3} {total/(1<<20):>8.2f} MiB "
                  f"{total/n/(1<<20):>8.4f} MiB  {sig}{mark}")
        print("  (* = the binding layer. per-layer = all n_expert of that layer; the formula charges "
              "per-expert.)")
        print()
        print(f"per_slot = MAX over TRUNK layers = blk.{binding[0]} -> {per_slot:,} B "
              f"({binding[1]/ (1<<20):.4f} MiB/expert)")
        if rest:
            print(f"  next   = blk.{rest[0][0]} -> {int(rest[0][1]):,} B "
                  f"({rest[0][1]/ (1<<20):.4f} MiB/expert)")
        if n_nextn:
            print(f"  NextN/MTP layer(s) {ils_label(mtp_layers)} are EXCLUDED from this MAX "
                  f"(n_nextn={n_nextn} from the GGUF); `denom` still counts all {n_layers} layers.")
            for r in head_over_trunk:
                print(f"  !! blk.{r[0]} (NextN) is THICKER than the trunk's MAX: {int(r[1]):,} B. "
                      f"Running the MAX over all layers would report "
                      f"{capacity(budgets[0], n_layers, int(r[1])) if budgets else '?'} slots where "
                      f"the engine computes {capacity(budgets[0], n_layers, per_slot) if budgets else '?'}.")
        else:
            print("  (no NextN/MTP layer in this file: n_nextn=0, every layer is trunk)")
        print()
        print(f"{'budget':<10} {'per_slot':>12} {'n_layers':>9} {'capacity':>9}   "
              f"= clamp(budget/(n_layers*per_slot), 8, 256)")
        for b in budgets:
            print(f"{b/gib:>6.2f} GiB {per_slot/1e6:>9.4f} MB {n_layers:>9} "
                  f"{capacity(b, n_layers, per_slot):>8} slots")
        print()
        print("if this disagrees with what the server logs for the same budget, the formula's "
              "assumption is what is wrong -- not the measurement.")

        if args.layer_caps:
            # per-EXPERT bytes, not per-layer: the capacity formula charges cap * per_slot where
            # per_slot is one expert's blob. `rows` carries the whole layer (all n_expert), so using
            # it here multiplies the answer by 256 -- and the result still LOOKS like a number
            # (a pool "2008 GiB" against a 10 GiB budget is the only tell).
            # int(), because per_exp divides: a float cap would be printed as "232.0" and fed to a
            # `%u` sscanf, which accepts it while quietly depending on where the '.' lands.
            ps_by_layer = {il: int(pe) for il, pe, _t, _n, _sig in per_exp}
            sig_by_layer = {il: sig for il, _t, _n, sig in rows}

            print()
            print("== per-layer caps (CONFIG route: no model bytes touched, no rebuild) ==")
            print(f"   arch={meta.get('arch')}  n_nextn={n_nextn} -> trunk blk.0..{n_decoder-1}, "
                  f"NextN/MTP {ils_label(mtp_layers) if mtp_layers else '(none)'}")
            if not mtp_layers:
                print("   no NextN layer in this file: every layer is trunk.")
            print("   rule: equal BYTES per trunk layer, clipped so no layer goes BELOW today's "
                  "uniform cap")
            print("         (the pool path requires the top-k union to fit in the layer's slots, and "
                  "too few slots reads nothing silently)")
            print("   mode: " + ("REDISTRIBUTE at constant pool bytes -- the footprint does not move"
                                 if args.keep_pool_bytes else
                                 ("SPEND %.3f GiB of slack above today's pool"
                                  % (parse_budget(args.slack) / (1 << 30)) if args.slack else
                                  "SPEND the whole budget slack -- the footprint grows (OOM risk on "
                                  "the memory-tight profile; use --slack to bound it)")))

            caps_json = {}
            for b in budgets:
                base = capacity(b, n_layers, per_slot)
                today = sum((args.mtp_cap if il in mtp_layers else base) * ps_by_layer[il]
                            for il in ps_by_layer)
                mtp_share = sum(args.mtp_cap * ps_by_layer[il] for il in mtp_layers)
                today_trunk = today - mtp_share
                if args.keep_pool_bytes:
                    cap_bytes = today
                elif args.slack is not None:
                    cap_bytes = today_trunk + parse_budget(args.slack) + mtp_share
                else:
                    cap_bytes = None
                caps, total, overflow = allocate_caps(
                    b, ps_by_layer, base, mtp_layers, args.mtp_cap, cap_bytes=cap_bytes)
                spec = emit_layer_caps(caps)

                # self-check: the emitter and cgc_layer_cap() must agree, layer by layer. Without
                # this the failure is silent -- a well-formed string that gives one layer the
                # wrong width is exactly a wrong pool geometry, and the pool region would be
                # narrower than the cache's slot writes (OOB).
                got = parse_layer_caps(spec, base, n_total)
                bad = [il for il in sorted(caps) if got[il] != caps[il]]
                by_cap = collections.defaultdict(list)
                for il in sorted(caps):
                    by_cap[caps[il]].append(il)

                print()
                print(f"  budget {b/gib:>6.2f} GiB   uniform today = {base:>3} slots/layer -> pool "
                      f"{today/gib:>5.2f} GiB ({100*today/b:>4.1f}% of budget)"
                      + ("   [!! the floor alone does not fit; keeping today's shape]" if overflow else ""))
                print(f"    {'layer(s)':<24} {'cap':>4} {'vs today':>9} {'pool':>9}  types")
                for cap, ils in sorted(by_cap.items(), key=lambda kv: (-kv[0], kv[1][0])):
                    pb = sum(cap * ps_by_layer[il] for il in ils)
                    print(f"    {ils_label(ils):<24} {cap:>4} {(cap/base - 1)*100:>+8.1f}% "
                          f"{pb/gib:>6.2f} GiB  {sig_by_layer[ils[0]]}")
                print(f"    {'TOTAL':<24} {'':>4} {'':>9} {total/gib:>6.2f} GiB "
                      f"({100*total/b:>4.1f}% of budget)")
                # The engine prints its own census of exactly this: "LAYER_CAPS per-layer caps:
                # total N slots (avg A/layer, min M/layer)". Quoting the expected line turns the
                # verification into one grep, and a mismatch is then unambiguous.
                #
                # AND the census is over the layers the cache actually pools. With the MTP (NextN)
                # arm OFF it pools the trunk only, so the same caps string yields two different
                # totals. 2026-09-17: the first gate run predicted 9443 and the engine printed
                # 9187 -- a 256-slot gap (exactly mtp_cap) that reads like a bug and is just the
                # NextN layer missing from the sum. Print both arms, so the operator knows which
                # number to expect BEFORE the run instead of explaining the gap afterwards.
                mtp_set = set(mtp_layers)
                trunk = [il for il in caps if il not in mtp_set]
                mtp_bytes = sum(args.mtp_cap * ps_by_layer[il] for il in mtp_set)
                nsl = sum(caps[il] for il in caps)
                nsl_trunk = sum(caps[il] for il in trunk)
                print(f"    expect in the log: \"LAYER_CAPS per-layer caps: total {nsl} slots "
                      f"(avg {nsl/len(caps):.1f}/layer, min {min(caps.values())}/layer)\""
                      + (f"   <- MTP=1, all {len(caps)} layers" if mtp_set else ""))
                if mtp_set:
                    print(f"    expect in the log: \"LAYER_CAPS per-layer caps: total {nsl_trunk} slots "
                          f"(avg {nsl_trunk/len(trunk):.1f}/layer, min "
                          f"{min(caps[il] for il in trunk)}/layer)\"   <- MTP=0, trunk only "
                          f"({len(trunk)} layers; the decode arms run MTP=0)")
                print(f"    expect resident ~ {total/gib:.2f} GiB (today ~ {today/gib:.2f} GiB)"
                      + (f"; with MTP=0 ~ {(total-mtp_bytes)/gib:.2f} GiB"
                         if mtp_set else ""))
                if bad:
                    print(f"    !! SELF-CHECK FAILED: {len(bad)} layer(s) differ from the emitted "
                          f"string -- first {bad[:4]}. Do NOT use this string.")
                else:
                    print(f"    self-check OK: re-parsed the string with cgc_layer_cap's semantics "
                          f"and all {len(caps)} layers match")
                print(f"    CGC_SERVER_LAYER_CAPS=\"{spec}\"")
                caps_json[str(b)] = {"uniform_today": base, "caps": {str(k): v for k, v in caps.items()},
                                     "spec": spec, "total_bytes": total, "today_bytes": today,
                                     "self_check_ok": not bad}

            out["layer_caps"] = caps_json
            if any(not v["self_check_ok"] for v in caps_json.values()):
                print()
                print("!! emitting a string that does not round-trip: refusing to be quiet about it.")
                return 4

        if args.normalize_binding:
            new_slot = int(rest[0][1]) if rest else per_slot
            print()
            print(f"scenario (ARITHMETIC, not a measurement): bring blk.{binding[0]} down to the "
                  f"typical layer ({typical/(1<<20):.4f} MiB/expert)")
            if rest:
                print(f"  per_slot would then be blk.{rest[0][0]}'s {new_slot:,} B "
                      f"({new_slot/(1<<20):.4f} MiB/expert)")
            else:
                print(f"  per_slot would stay {new_slot:,} B")
            for b in budgets:
                prev = capacity(b, n_layers, per_slot)
                new = capacity(b, n_layers, new_slot)
                print(f"  {b/gib:>6.2f} GiB  {prev:>4} -> {new:>4} slots  "
                      f"({(new / prev - 1) * 100:+.1f}%)")

            # A second, strictly larger step: bring EVERY layer that is above the typical layer down
            # to it. Reported separately because the two numbers bracket the real decision (fix one
            # layer, or fix the shape) and quoting only the first understates the ceiling.
            fat = [r for r in per_exp if r[1] > typical + 1]
            all_slot = int(typical)
            print(f"  ... and if every layer above the typical one ({len(fat)} of {n_layers}) is "
                  f"brought down to it:")
            for b in budgets:
                prev = capacity(b, n_layers, per_slot)
                new = capacity(b, n_layers, all_slot)
                print(f"  {b/gib:>6.2f} GiB  {prev:>4} -> {new:>4} slots  "
                      f"({(new / prev - 1) * 100:+.1f}%)")

            out["normalize_binding"] = {
                "typical_per_expert_bytes": int(typical),
                "new_binding_layer": rest[0][0] if rest else binding[0],
                "new_per_slot_bytes": new_slot,
                "capacity": [{"budget": b, "slots": capacity(b, n_layers, new_slot)} for b in budgets],
                "fat_layers": [r[0] for r in fat],
                "all_typical_per_slot_bytes": all_slot,
                "capacity_all_typical": [{"budget": b, "slots": capacity(b, n_layers, all_slot)}
                                         for b in budgets],
            }

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        if not args.quiet:
            print(f"\njson -> {os.path.relpath(args.json, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
