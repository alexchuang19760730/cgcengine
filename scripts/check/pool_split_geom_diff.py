#!/usr/bin/env python3
"""Is the LAYER_CAPS numeric carrier the per-layer PATH (pool vs wide), or something inside one path?

The question this answers
-------------------------
Measured on one binary, same 884-step long probe (2026-09-19):
  * canonicalising the k=8 reduction does NOT close it (canonA vs canonE2: M1 1/968; the
    same-config null control was 1003/1003 clean)
  * tensor SHAPE is not the carrier: pool 8 -> 4 GiB halves every layer's expert axis
    (145.8 -> 75.5 slots/layer) and the output is bit-identical (884/884)
  * layer 40 is not the carrier in either direction, and the effect is confined to the TRUNK's
    explicit per-layer caps
What is left is the path. `cgc_apply_expert_geometry` picks between the pool region and the
wide/overflow region per layer per step, with an expert-axis width (`ne2`) per case:

    CGC-POOL-SPLIT-GEOM: il=0 kind=1 ntok=1 mode=pool ne2=143 data=0x.. buf=0x..
    CGC-POOL-SPLIT-GEOM-CHANGE: il=3 kind=1 pool/143 -> wide/256

Until now that line printed for `il <= 2` only, and three layers cannot distinguish "these layers
changed path" from "every layer did". This tool takes two arms' logs and answers it per layer.

Reading rule (the failure this tool is built to refuse)
-------------------------------------------------------
**An absent instrument is not an equal one.** If an arm's log has no GEOM lines, the two arms are
NOT "the same path" -- that is the exact shape of the failure this project keeps re-learning
("the knob did not apply" and "the knob changed nothing" are the same observation without an
arm-identity line). So the verdict is ABSENT, not EQUAL, whenever either side is empty, and the
tool exits non-zero for it.

Usage
-----
    python3 scripts/check/pool_split_geom_diff.py A.log B.log [--arm-a NAME --arm-b NAME]
    python3 scripts/check/pool_split_geom_diff.py --selftest
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

GEOM = re.compile(
    r"CGC-POOL-SPLIT-GEOM:\s+il=(\d+)\s+kind=(\d+)\s+ntok=(\d+)\s+mode=(\S+)\s+ne2=(\d+)")
CHANGE = re.compile(
    r"CGC-POOL-SPLIT-GEOM-CHANGE:\s+il=(\d+)\s+kind=(\d+)\s+(\S+)\s*->\s*(\S+)")


def parse(path: str | Path) -> dict[tuple[int, int], set[tuple[str, int]]]:
    """(il, kind) -> set of (mode, ne2) tuples observed. CHANGE lines add their target too, so a
    path that moved and moved back is still visible as two tuples rather than one."""
    out: dict[tuple[int, int], set[tuple[str, int]]] = {}
    text = Path(path).read_text(errors="replace")
    for m in GEOM.finditer(text):
        il, kind, _ntok, mode, ne2 = m.groups()
        out.setdefault((int(il), int(kind)), set()).add((mode, int(ne2)))
    for m in CHANGE.finditer(text):
        il, kind, _old, new = m.groups()
        if "/" in new:
            mode, ne2 = new.rsplit("/", 1)
            out.setdefault((int(il), int(kind)), set()).add((mode, int(ne2)))
    return out


def compare(a: dict, b: dict) -> dict:
    keys = sorted(set(a) | set(b))
    rows = []
    for k in keys:
        sa, sb = a.get(k), b.get(k)
        rows.append({"il": k[0], "kind": k[1],
                     "a": sorted(sa) if sa else None,
                     "b": sorted(sb) if sb else None,
                     "same": (sa is not None and sb is not None and sa == sb)})
    return {"rows": rows,
            "n_keys": len(keys),
            "n_same": sum(1 for r in rows if r["same"]),
            "a_empty": not a,
            "b_empty": not b}


def render(rep: dict, name_a: str, name_b: str) -> tuple[str, int]:
    L = []
    if rep["a_empty"] or rep["b_empty"]:
        which = name_a if rep["a_empty"] else name_b
        L.append(f"VERDICT: INSTRUMENT ABSENT in {which} -- this is NOT evidence of an equal path.")
        L.append("  No CGC-POOL-SPLIT-GEOM line means the arm did not report its geometry at all")
        L.append("  (knob not applied, or not reaching the server), which looks exactly like")
        L.append("  'the paths are the same'. Fix the instrumentation before reading anything else.")
        return "\n".join(L), 2
    diff = [r for r in rep["rows"] if not r["same"]]
    L.append(f"layers/kinds compared: {rep['n_keys']}   identical: {rep['n_same']}   "
             f"differing: {len(diff)}")
    if not diff:
        L.append("")
        L.append("VERDICT: PATH IS NOT THE CARRIER.")
        L.append("  Every (layer, kind) took the same mode at the same expert-axis width in both")
        L.append("  arms, so the numeric difference is NOT path selection. It has to be inside one")
        L.append("  path (what the path reads and in which order), and then LAYER_CAPS can only be")
        L.append("  a different configuration, never an equivalent one.")
        return "\n".join(L), 0
    L.append("")
    L.append("VERDICT: PATH SELECTION IS LIVE -- these (layer, kind) differ:")
    L.append(f"  {'layer':>5} {'kind':>4}  {name_a:<28} {name_b}")
    for r in diff[:40]:
        L.append(f"  {r['il']:>5} {r['kind']:>4}  {str(r['a']):<28} {r['b']}")
    if len(diff) > 40:
        L.append(f"  ... and {len(diff) - 40} more")
    L.append("")
    L.append("  A differing PATH (pool vs wide) is a different computation, not a reordering, so a")
    L.append("  fix must make the path caps-independent -- canonicalising the reduction cannot reach")
    L.append("  something that a different kernel computed.")
    return "\n".join(L), 1


def selftest() -> int:
    fails = []

    def expect(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            fails.append(name)

    A = ("CGC-POOL-SPLIT-GEOM: il=0 kind=1 ntok=1 mode=pool ne2=143 data=0x1 buf=0x2\n"
         "CGC-POOL-SPLIT-GEOM: il=5 kind=1 ntok=1 mode=pool ne2=143 data=0x1 buf=0x2\n")
    B_same = ("CGC-POOL-SPLIT-GEOM: il=0 kind=1 ntok=2 mode=pool ne2=143 data=0x1 buf=0x2\n"
              "CGC-POOL-SPLIT-GEOM: il=5 kind=1 ntok=2 mode=pool ne2=143 data=0x1 buf=0x2\n")
    B_width = ("CGC-POOL-SPLIT-GEOM: il=0 kind=1 ntok=2 mode=pool ne2=163 data=0x1 buf=0x2\n"
               "CGC-POOL-SPLIT-GEOM: il=5 kind=1 ntok=2 mode=pool ne2=143 data=0x1 buf=0x2\n")
    B_mode = ("CGC-POOL-SPLIT-GEOM: il=0 kind=1 ntok=2 mode=wide ne2=256 data=0x1 buf=0x2\n"
              "CGC-POOL-SPLIT-GEOM: il=5 kind=1 ntok=2 mode=pool ne2=143 data=0x1 buf=0x2\n")

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)

        def w(n, s):
            (p / n).write_text(s)
            return p / n

        fa, fb = w("a.log", A), w("b.log", B_same)
        rep = compare(parse(fa), parse(fb))
        expect("identical geometry -> same", [r["same"] for r in rep["rows"]], [True, True])
        expect("ntok is not part of the identity", rep["n_same"], 2)

        rep = compare(parse(fa), parse(w("bw.log", B_width)))
        expect("a width difference is caught", [r["same"] for r in rep["rows"]], [False, True])

        rep = compare(parse(fa), parse(w("bm.log", B_mode)))
        expect("a mode difference is caught", [r["same"] for r in rep["rows"]], [False, True])
        txt, rc = render(rep, "armA", "armB")
        expect("and it is named as the carrier", "PATH SELECTION IS LIVE" in txt, True)
        expect("with a non-zero exit", rc, 1)

        # the failure this tool exists to refuse
        rep = compare(parse(fa), parse(w("empty.log", "no geometry here\n")))
        txt, rc = render(rep, "armA", "armB")
        expect("an absent instrument is NOT read as equal", "INSTRUMENT ABSENT" in txt, True)
        expect("and is not a pass", rc, 2)
        expect("same-path verdict only when both sides reported",
               "PATH IS NOT THE CARRIER" in render(compare(parse(fa), parse(fb)), "a", "b")[0], True)

        rep = compare(parse(fa), parse(w("chg.log", A + "CGC-POOL-SPLIT-GEOM-CHANGE: il=0 kind=1 "
                                        "pool/143 -> wide/256\n")))
        expect("a CHANGE target is picked up as a second tuple",
               rep["rows"][0]["b"], [("pool", 143), ("wide", 256)])

    print()
    if fails:
        print(f"SELFTEST FAIL ({len(fails)}): {fails}")
        return 1
    print("SELFTEST PASS -- and an absent instrument can never read as an equal one")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--arm-a", default="A")
    ap.add_argument("--arm-b", default="B")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if len(args.logs) != 2:
        ap.error("need exactly two log paths (or --selftest)")
    a, b = (parse(x) for x in args.logs)
    txt, rc = render(compare(a, b), args.arm_a, args.arm_b)
    print(txt)
    return rc


if __name__ == "__main__":
    sys.exit(main())
