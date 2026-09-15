#!/usr/bin/env python3
"""Adjudicate CGC-MMID-ASSERT zero_row alarms against the model file itself.

Why this exists
---------------
`ggml-metal-ops.cpp` asserts that a routed expert row read by mul_mat_id is not
all-zero, on the theory that "quantized expert weights are never all-zero in their
first 4KB". That theory is FALSE for this GGUF: `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-
denseIQ4X.gguf` really does contain dead experts whose 4KB probes are zero -- measured
`blk.1.ffn_gate_exps.weight` / `blk.1.ffn_up_exps.weight` expert 214 => 0 non-zero
bytes out of 4096, while experts 0/148/151/161 give 4048-4055.

So a `zero_row=1` line on its own is NOT evidence of a defect. The engine cannot tell
the two cases apart: it sees the slab, never the file. The distinction is recoverable
offline, and that is all this script does -- for every alarm it reads the GGUF at the
exact expert row and says which of the two it is:

    MODEL-ZERO   the file row is zero too. Expected. Not a defect, do not chase it.
    ENGINE-ZERO  the file row is NON-zero but the slab row is zero. This is the real
                 failure: the fill did not land (or landed elsewhere) and the expert's
                 contribution was silently dropped.
    UNRESOLVED   the id could not be mapped to an expert (pool-path slot with no
                 CGC-EXACT-POST tablet in the log, or a tensor name that is absent).

The two conventions are told apart exactly as mmid_pool_vs_gguf.py does it:
`ne02 >= n_expert` means the ids are RAW expert ids (M2 whole-layer slab path);
anything smaller means they are SLOT indices and need the expert->slot map inverted.

Usage:
    /opt/homebrew/bin/python3 scripts/check/mmid_zero_row_triage.py LOG [LOG ...]
    /opt/homebrew/bin/python3 scripts/check/mmid_zero_row_triage.py Backup/cgc_logs/*.log
Exit codes: 0 = every alarm is MODEL-ZERO (or there were none),
            1 = at least one ENGINE-ZERO (a real dropped-expert defect),
            2 = nothing parseable / no GGUF.
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
    r"zero_row=(?P<n_zero>\d+) \(first_id=(?P<first_id>-?\d+), total=(?P<zero_total>\d+)\) \| "
    r"src0 ne=\[(?P<ne00>\d+),(?P<ne01>\d+),(?P<ne02v>\d+)\] nb02=(?P<nb02>\d+) "
    r"ids ne=\[(?P<ne20>\d+),(?P<ne21>\d+)\]"
)
TAG_RE = re.compile(r"^ffn_moe_(?P<kind>gate|up|down)-(?P<il>\d+)$")
POST_RE = re.compile(r"CGC-EXACT-POST: il=(\d+)\s*\n((?:\s+uni\[\d+\]=\d+ slot=-?\d+.*\n)+)")
UNI_RE = re.compile(r"uni\[(\d+)\]=(\d+) slot=(-?\d+)")

KIND_OF = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}


def slot_tables(text):
    """expert -> slot per layer, first occurrence wins (the first eval pass)."""
    out = {}
    for m in POST_RE.finditer(text):
        il = int(m.group(1))
        if il in out:
            continue
        out[il] = {int(u.group(2)): int(u.group(3)) for u in UNI_RE.finditer(m.group(2))}
    return out


def nonzero_prefix(fh, off, nbytes):
    fh.seek(off)
    buf = fh.read(nbytes)
    return sum(1 for b in buf if b), len(buf)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    model = DEFAULT_MODEL
    for a in sys.argv[1:]:
        if a.startswith("--model="):
            model = a.split("=", 1)[1]
    if not args:
        print(__doc__.strip().splitlines()[-4].strip(), file=sys.stderr)
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
    tally = {"MODEL-ZERO": 0, "ENGINE-ZERO": 0, "UNRESOLVED": 0, "rows": 0}
    engine_hits = []
    print(f"\n{'log':<28s} {'op':<16s} {'ne02':>5s} {'id':>5s} {'expert':>6s} "
          f"{'file_nz/4K':>10s}  verdict")
    print("-" * 96)
    for path in args:
        text = open(path, encoding="utf-8", errors="replace").read()
        tables = slot_tables(text)
        hits = 0
        for m in ASSERT_RE.finditer(text):
            if int(m.group("n_zero")) == 0:
                continue
            tag = TAG_RE.search(m.group("name"))
            if tag is None:
                continue
            kind, il = tag.group("kind"), int(tag.group("il"))
            tname = f"blk.{il}.{KIND_OF[kind]}.weight"
            t = tensors.get(tname)
            first_id = int(m.group("first_id"))
            ne02 = int(m.group("ne02"))
            nb02 = int(m.group("nb02"))
            hits += 1
            tally["rows"] += 1
            if t is None:
                verdict, expert = "UNRESOLVED", None
                tally["UNRESOLVED"] += 1
            else:
                n_file_expert = (int(t.n_bytes) // nb02) if nb02 else 0
                if ne02 >= n_file_expert:
                    expert = first_id                       # raw expert id (slab path)
                else:
                    inv = {s: e for e, s in tables.get(il, {}).items()}
                    expert = inv.get(first_id)              # slot id -> expert (pool path)
                if expert is None:
                    verdict = "UNRESOLVED"
                    tally["UNRESOLVED"] += 1
                else:
                    nz, got = nonzero_prefix(fh, int(t.data_offset) + expert * nb02, PROBE_BYTES)
                    verdict = "MODEL-ZERO" if nz == 0 else "ENGINE-ZERO"
                    if verdict == "ENGINE-ZERO":
                        engine_hits.append((os.path.basename(path), m.group("name"),
                                            expert, nz, got))
                        tally["ENGINE-ZERO"] += 1
                    else:
                        tally["MODEL-ZERO"] += 1
            shown_nz = "-" if expert is None else str(nz)
            print(f"{os.path.basename(path):<28s} {m.group('name'):<16s} {ne02:>5d} "
                  f"{first_id:>5d} {('-' if expert is None else expert):>6} {shown_nz:>10s}  {verdict}")
        if hits == 0:
            print(f"{os.path.basename(path):<28s} {'(no zero_row alarm)':<16s}")
    print("-" * 96)
    print(json.dumps(tally, indent=1, ensure_ascii=False))
    if engine_hits:
        print("\nVERDICT: ENGINE-ZERO alarms found -- the fill dropped an expert whose file row is")
        print("         non-zero. This IS a real dropped-expert defect, not a model property.")
        for h in engine_hits:
            print(f"  {h[0]} {h[1]} expert={h[2]} file_nonzero_prefix={h[3]}/{h[4]}")
        return 1
    if tally["rows"] == 0:
        print("\nVERDICT: no zero_row alarm in the given logs -- nothing to adjudicate.")
        return 0
    print(f"\nVERDICT: {tally['MODEL-ZERO']} MODEL-ZERO, {tally['ENGINE-ZERO']} ENGINE-ZERO, "
          f"{tally['UNRESOLVED']} UNRESOLVED (of {tally['rows']} alarms).")
    if tally["UNRESOLVED"]:
        print("         UNRESOLVED = a pool-path (ne02 < n_expert) alarm whose slot->expert map is")
        print("         absent from the log. Re-run with CGC_EXACT_PATH_DBG=1 to get the map.")
    print("         MODEL-ZERO is the model file's own data, not a fill/read defect: this GGUF")
    print("         has dead experts (measured 10 of 35484 rows, all in layers 0-1 gate/up), so")
    print("         the assertion's premise ('never all-zero') does not hold for it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
