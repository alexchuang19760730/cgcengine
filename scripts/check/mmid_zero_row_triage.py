#!/usr/bin/env python3
"""Adjudicate CGC zero-row / zero-region alarms against the model file itself.

Why this exists
---------------
Two instruments raise alarms of this shape, both by scanning a *fixed-width prefix* of an
expert row for non-zero bytes:

    ggml-metal-ops.cpp          CGC-MMID-ASSERT  ... zero_row=N (first_id=X, ...)
    llama-expert-cache.cpp      pool integrity: owner-set slots=... zero-regions=N

Neither can tell WHY a region is zero, and the two reasons are opposite in consequence:

    the FILE row is zero too  ->  MODEL-ZERO. A multiply by zeros is correct here. The
                                  engine is doing the right thing; the alarm is a property
                                  of the checkpoint, not a defect.
    the file row is NON-zero  ->  ENGINE-ZERO. The fill never landed (or landed in the
                                  wrong slot) and the expert's contribution was dropped.
                                  This IS a real defect.

THE FLAW THIS SCRIPT USED TO HAVE (fixed 2026-09-16)
----------------------------------------------------
It answered that question with a 4096-byte prefix read of the file row -- the SAME width as
the alarm it was judging. So for any row whose first 4 KiB is zero it could only ever return
MODEL-ZERO, and every inspection confirmed the alarm's own premise instead of testing it.
Its "14 MODEL-ZERO / 0 ENGINE-ZERO" verdict was therefore structurally guaranteed, not
measured. Reading the ENTIRE row stride shows the truth for this GGUF:

    10 of 31488 gate/up/down expert rows begin with 4592-13120 bytes of zeros and are
    17.8-47.0% non-zero overall. Every one of the other 31478 rows has a leading zero run
    of exactly 0.

So those rows are NOT dead; they are zero-PREFIXED. A pool region that is zero in its first
4 KiB is what the file says it should be. The verdict for them is PROBE-TOO-NARROW -- the
instrument is too blind to see the data behind the prefix -- and the fix belongs in the
probe, not in the fill. `scripts/check/gguf_dead_expert_census.py` produces the same census
straight from the file, with no log needed.

Conventions
-----------
`ne02 >= n_file_expert` means the ids are RAW expert ids (whole-layer slab path); anything
smaller means SLOT indices and needs the expert->slot map inverted. The slot->expert map for
the pool paths comes from `CGC-EXACT-POST` (needs CGC_EXACT_PATH_DBG=1) or, for the FIRST
zero region, from the pool-integrity line itself, which names `owner_expert`.

Usage:
    /opt/homebrew/bin/python3 scripts/check/mmid_zero_row_triage.py LOG [LOG ...]
    /opt/homebrew/bin/python3 scripts/check/mmid_zero_row_triage.py Backup/cgc_logs/*.log

Exit codes: 0 = every alarm explained by the file (MODEL-ZERO / PROBE-TOO-NARROW / none),
            1 = at least one ENGINE-ZERO (a real dropped-expert defect),
            2 = nothing parseable / no GGUF,
            3 = no ENGINE-ZERO, but at least one PROBE-TOO-NARROW (fix the probe width).
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL = os.path.join(
    ROOT, "models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf")

PROBE_BYTES = 4096

ASSERT_RE = re.compile(
    r"CGC-MMID-ASSERT name=(?P<name>\S+) ne02=(?P<ne02>\d+) n_ids=(?P<n_ids>\d+) \| "
    r"id_oob=(?P<oob>\d+) \(first=(?P<oob_first>-?\d+), total=(?P<oob_total>\d+)\) \| "
    r"zero_row=(?P<n_zero>\d+) \(first_id=(?P<first_id>-?\d+), total=(?P<zero_total>\d+)\) "
    # The 2026-09-16 format adds the whole-row-confirmed counter right here.
    r"(?:\| zero_full=(?P<n_full>\d+) \(first_id=(?P<full_first>-?\d+), "
    r"total=(?P<full_total>\d+)\) )?"
    r"\| src0 ne=\[(?P<ne00>\d+),(?P<ne01>\d+),(?P<ne02v>\d+)\] nb02=(?P<nb02>\d+) "
    r"ids ne=\[(?P<ne20>\d+),(?P<ne21>\d+)\]"
)

POOL_RE = re.compile(
    r"pool integrity: owner-set slots=(?P<slots>\d+) zero-regions=(?P<zreg>\d+)"
    r"(?: zero-prefix-only=(?P<zpref>\d+))?")
POOL_FIRST_RE = re.compile(
    r"first zero region: layer=(?P<l>\d+) slot=(?P<s>-?\d+) kind=(?P<k>\d+) owner_expert=(?P<e>-?\d+)")
POOL_FIRST_PREF_RE = re.compile(
    r"zero-PREFIX only .*?: first layer=(?P<l>\d+) slot=(?P<s>-?\d+) kind=(?P<k>\d+) owner_expert=(?P<e>-?\d+)")

TAG_RE = re.compile(r"^ffn_moe_(?P<kind>gate|up|down)-(?P<il>\d+)$")
POST_RE = re.compile(r"CGC-EXACT-POST: il=(\d+)\s*\n((?:\s+uni\[\d+\]=\d+ slot=-?\d+.*\n)+)")
UNI_RE = re.compile(r"uni\[(\d+)\]=(\d+) slot=(-?\d+)")

KIND_OF = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}
KIND_FROM_IDX = {0: "gate", 1: "up", 2: "down"}


def slot_tables(text):
    """expert -> slot per layer, first occurrence wins (the first eval pass)."""
    out = {}
    for m in POST_RE.finditer(text):
        il = int(m.group(1))
        if il in out:
            continue
        out[il] = {int(u.group(2)): int(u.group(3)) for u in UNI_RE.finditer(m.group(2))}
    return out


def scan_row(fh, off, stride, probe_cap=PROBE_BYTES):
    """Non-zero bytes in the first min(stride, probe_cap) B, the leading zero run, and the
    non-zero byte count over the WHOLE stride. The third number is the one that matters."""
    probe = min(stride, probe_cap)
    fh.seek(off)
    buf = fh.read(stride)
    got = len(buf)
    nz_probe = sum(1 for b in buf[:probe] if b)
    nz_full = sum(1 for b in buf if b)
    lead = next((i for i, b in enumerate(buf) if b), got)
    return nz_probe, lead, nz_full, got


def main() -> int:
    argv = sys.argv[1:]
    model = DEFAULT_MODEL
    args = []
    for a in argv:
        if a.startswith("--model="):
            model = a.split("=", 1)[1]
        elif not a.startswith("--"):
            args.append(a)
    if not args:
        print(__doc__.strip().splitlines()[-6].strip(), file=sys.stderr)
        return 2
    if not os.path.exists(model):
        print(f"error: model not found: {model}", file=sys.stderr)
        return 2
    sys.path.insert(0, os.path.join(ROOT, "scripts/check"))
    try:
        from gguf.gguf_reader import GGUFReader
    except ImportError:
        print("error: the `gguf` python package is required (pip install gguf)", file=sys.stderr)
        return 2

    reader = GGUFReader(model)
    tensors = {t.name: t for t in reader.tensors}
    fh = open(model, "rb")

    print(f"model : {model}")
    tally = {"MODEL-ZERO": 0, "PROBE-TOO-NARROW": 0, "ENGINE-ZERO": 0,
             "UNRESOLVED": 0, "alarms": 0}
    engine_hits, probe_hits, cross = [], [], []
    print(f"\n{'log':<26s} {'alarm':<10s} {'op':<15s} {'id':>5s} {'expert':>6s} "
          f"{'nz@probe':>8s} {'nz@full':>8s} {'lead':>6s}  verdict")
    print("-" * 100)

    for path in args:
        text = open(path, encoding="utf-8", errors="replace").read()
        tables = slot_tables(text)
        # The pool-integrity line names one (layer, slot, owner_expert) triple. Fold it in as an
        # extra slot->expert source, so a mul_mat_id alarm on the same slot resolves without
        # CGC_EXACT_PATH_DBG=1 having been set.
        for pm in POOL_RE.finditer(text):
            tail = text[pm.end():pm.end() + 1200]
            for src in (POOL_FIRST_RE.search(tail), POOL_FIRST_PREF_RE.search(tail)):
                if src is not None:
                    tables.setdefault(int(src.group("l")), {}).setdefault(
                        int(src.group("e")), int(src.group("s")))
        hits = 0

        # ---- alarms from the always-on mul_mat_id assertion (probe-wide by construction) ----
        for m in ASSERT_RE.finditer(text):
            if int(m.group("n_zero")) == 0:
                continue
            tag = TAG_RE.search(m.group("name"))
            if tag is None:
                continue
            hits += 1
            tally["alarms"] += 1
            kind, il = tag.group("kind"), int(tag.group("il"))
            first_id = int(m.group("first_id"))
            ne02, nb02 = int(m.group("ne02")), int(m.group("nb02"))
            verdict, expert, nz_probe, lead, nz_full = adjudicate(
                fh, tensors, tables, il, KIND_OF[kind], ne02, nb02, first_id)
            record(alarm="zero_row", op=m.group("name"), first_id=first_id,
                   expert=expert, verdict=verdict, nums=(nz_probe, lead, nz_full),
                   tally=tally, engine_hits=engine_hits, probe_hits=probe_hits,
                   path=path, tag=f" il={il}")
            # Cross-check against the engine's own whole-row confirmation when the log carries
            # it (2026-09-16 format). A disagreement means the two are reading different rows.
            if m.group("n_full") is not None and expert is not None:
                expect = 1 if nz_full == 0 else 0
                if int(m.group("n_full")) != expect:
                    cross.append((os.path.basename(path), m.group("name"),
                                  int(m.group("n_full")), expect))

        # ---- alarms from the teardown pool-integrity scan ----
        for pm in POOL_RE.finditer(text):
            if int(pm.group("zreg")) == 0 and int(pm.group("zpref") or 0) == 0:
                continue
            hits += 1
            tally["alarms"] += 1
            n_report = int(pm.group("zreg"))
            has_new = pm.group("zpref") is not None
            tail = text[pm.end():pm.end() + 1200]
            fz = POOL_FIRST_RE.search(tail)
            fz_pref = POOL_FIRST_PREF_RE.search(tail)
            src = fz if (fz is not None and n_report > 0) else fz_pref
            if src is None:
                tally["UNRESOLVED"] += 1
                print(f"{os.path.basename(path):<26s} {'zero-region':<10s} {'(pool)':<15s} "
                      f"{'-':>5s} {'-':>6s} {'-':>8s} {'-':>8s} {'-':>6s}  UNRESOLVED"
                      f"  (no region line names the owner expert)")
                continue
            kind = KIND_FROM_IDX.get(int(src.group("k")), "gate")
            il, first_id = int(src.group("l")), int(src.group("s"))
            owner = int(src.group("e"))
            t = tensors.get(f"blk.{il}.{KIND_OF[kind]}.weight")
            if t is None:
                tally["UNRESOLVED"] += 1
                print(f"{os.path.basename(path):<26s} {'zero-region':<10s} "
                      f"{KIND_OF[kind]:<15s} {first_id:>5d} {owner:>6d} {'-':>8s} {'-':>8s} "
                      f"{'-':>6s}  UNRESOLVED")
                continue
            stride = int(t.n_bytes) // max(1, int(t.shape[2]))
            nz_probe, lead, nz_full, _ = scan_row(
                fh, int(t.data_offset) + owner * stride, stride)
            # A NEW-format `zero-regions` count is whole-stride-confirmed. An OLD-format one
            # (or a `zero-prefix-only` count) is probe-wide, so it may be a prefix hit.
            if has_new and src is fz:
                verdict = "MODEL-ZERO" if nz_full == 0 else "ENGINE-ZERO"
            else:
                verdict = ("MODEL-ZERO" if nz_full == 0 else
                           ("ENGINE-ZERO" if nz_probe != 0 else "PROBE-TOO-NARROW"))
            record(alarm="zero-region", op=KIND_OF[kind], first_id=first_id, expert=owner,
                   verdict=verdict, nums=(nz_probe, lead, nz_full), tally=tally,
                   engine_hits=engine_hits, probe_hits=probe_hits, path=path,
                   tag=f" il={il}")

        if hits == 0:
            print(f"{os.path.basename(path):<26s} {'(no zero alarm)':<10s}")

    print("-" * 100)
    print(json.dumps(tally, indent=1, ensure_ascii=False))
    if cross:
        print(f"\nCROSS-CHECK MISMATCH ({len(cross)}): the engine's own zero_full counter disagrees")
        print("  with this script's whole-row read. The two are not looking at the same row:")
        for p, name, got, expect in cross:
            print(f"  {p} {name} engine_zero_full={got} expected={expect}")
    if engine_hits:
        print("\nVERDICT: ENGINE-ZERO -- the file row is non-zero but the engine's region was zero.")
        print("         This IS a real dropped-expert defect, not a model property.")
        for h in engine_hits:
            print(f"  {h[0]} {h[1]} expert={h[2]} file_nonzero_bytes={h[3]}")
        return 1
    if tally["alarms"] == 0:
        print("\nVERDICT: no zero alarm in the given logs -- nothing to adjudicate.")
        return 0
    if probe_hits:
        print(f"\nVERDICT: {tally['PROBE-TOO-NARROW']} PROBE-TOO-NARROW -- no defect. The file row is")
        print("         zero over the probe window only; it is non-zero further in. The instrument")
        print("         cannot see past the prefix. Fix the probe (two-tier: probe then full stride)")
        print("         rather than the fill:")
        for h in probe_hits:
            print(f"  {h[0]} {h[1]} expert={h[2]} leading_zero_run={h[3]}B "
                  f"file_nonzero_bytes_over_row={h[4]}")
        print("         Census without a log: scripts/check/gguf_dead_expert_census.py")
        return 3
    print(f"\nVERDICT: {tally['MODEL-ZERO']} MODEL-ZERO, 0 ENGINE-ZERO "
          f"({tally['UNRESOLVED']} UNRESOLVED of {tally['alarms']} alarms).")
    return 0


def adjudicate(fh, tensors, tables, il, kind_of, ne02, nb02, first_id):
    """Map an alarm id to an expert and read that expert's row straight from the GGUF.

    Returns (verdict, expert, nz_probe, lead, nz_full)."""
    tname = f"blk.{il}.{kind_of}.weight"
    t = tensors.get(tname)
    if t is None or nb02 <= 0:
        return "UNRESOLVED", None, None, None, None
    n_file_expert = (int(t.n_bytes) // nb02) if nb02 else 0
    if ne02 >= n_file_expert:
        expert = first_id                       # raw expert id (slab path)
    else:
        inv = {s: e for e, s in tables.get(il, {}).items()}
        expert = inv.get(first_id)              # slot id -> expert (pool path)
    if expert is None:
        return "UNRESOLVED", None, None, None, None
    stride = nb02 if nb02 else (int(t.n_bytes) // max(1, int(t.shape[2])))
    nz_probe, lead, nz_full, _got = scan_row(
        fh, int(t.data_offset) + expert * stride, stride)
    if nz_full == 0:
        return "MODEL-ZERO", expert, nz_probe, lead, nz_full
    if nz_probe != 0:
        return "ENGINE-ZERO", expert, nz_probe, lead, nz_full
    return "PROBE-TOO-NARROW", expert, nz_probe, lead, nz_full


def record(*, alarm, op, first_id, expert, verdict, nums, tally, engine_hits,
           probe_hits, path, tag=""):
    tally[verdict] += 1
    nz_probe, lead, nz_full = nums
    base = os.path.basename(path)
    if verdict == "ENGINE-ZERO":
        engine_hits.append((base, op + tag, expert, nz_full))
    elif verdict == "PROBE-TOO-NARROW":
        probe_hits.append((base, op + tag, expert, lead, nz_full))
    fmt = lambda v: "-" if v is None else str(v)
    print(f"{base:<26s} {alarm:<10s} {op:<15s} {first_id:>5d} {fmt(expert):>6s} "
          f"{fmt(nz_probe):>8s} {fmt(nz_full):>8s} {fmt(lead):>6s}  {verdict}")


if __name__ == "__main__":
    sys.exit(main())
