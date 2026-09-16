#!/usr/bin/env python3
"""Bucket CGC-DECPROF decode graphs by their ORDINAL in the process.

WHY. On 2026-09-16 the two decode instruments disagreed by 2.6x at n=24 and 1.31x at n=128 while
carrying the same engine env (`Backup/run_instrument_compare.sh`). A median over the whole process
could not separate "this process's step is slower" from "this process spent its early steps filling
a cold expert pool". Ordinal bucketing can: if the cost is a warming curve it must fall with the
ordinal, and the last bucket is the steady state the two instruments should be compared on.

The parser is NOT re-implemented here. `prefill_certifiability.harvest_decprof` is the single
implementation, and it classifies a graph by what the graph SAYS (`ntok=`) rather than by position
-- which is the difference between "the first graph is a prefill" and "a prefill is a graph that
says ntok>1" (lesson eng-src-0011). Bucketing on position would mix a 677 ms prefill chunk into a
decode median; that is not hypothetical, it happened while this file was being written.

Usage:
    python3 Backup/instr_step_profile.py <log> [<log> ...] [--buckets 4] [--tsv out.tsv]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT / "scripts" / "check" / "prefill_certifiability.py"

spec = importlib.util.spec_from_file_location("pc", str(PC))
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def decode_graphs(path: Path) -> list[dict]:
    """Every decode graph of one process, in ordinal order."""
    d = pc.harvest_decprof(path.read_text(errors="replace"))
    series = d.get("dp_series") or []
    # `ntok` is None for graphs printed before the ntok= field existed; those are excluded rather
    # than guessed, because the whole point of the field is that a graph states its own shape.
    return [g for g in series if g.get("ntok") == 1]


def profile(path: Path, n_buckets: int) -> dict:
    gs = decode_graphs(path)
    if not gs:
        return {"log": str(path), "decode_graphs": 0, "buckets": []}
    k = max(1, len(gs) // n_buckets)
    buckets = []
    for i in range(n_buckets):
        chunk = gs[i * k:] if i == n_buckets - 1 else gs[i * k:(i + 1) * k]
        if not chunk:
            continue
        tot = [g["total_ms"] for g in chunk]
        buckets.append({
            "lo": i * k + 1, "hi": i * k + len(chunk), "n": len(chunk),
            "total_median": round(st.median(tot), 2),
            "total_mean": round(st.mean(tot), 2),
            "wait_median": round(st.median([g["wait_ms"] for g in chunk]), 2),
            "cb_median": round(st.median([g["cb_ms"] for g in chunk]), 2),
            "submit_median": round(st.median([g["submit_ms"] for g in chunk]), 2),
        })
    return {"log": os.path.basename(str(path)), "decode_graphs": len(gs), "buckets": buckets}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--buckets", type=int, default=4)
    ap.add_argument("--tsv")
    args = ap.parse_args()

    rows = []
    for p in args.logs:
        r = profile(Path(p), args.buckets)
        rows.append(r)
        print(f"\n=== {r['log']}   decode graphs: {r['decode_graphs']}")
        if not r["buckets"]:
            print("    (no ntok=1 graph; was CGC_DECODE_PROFILE=1 set?)")
            continue
        print(f"    {'ordinal':>12s} {'n':>5s} {'total_med':>10s} {'mean':>9s} "
              f"{'wait_med':>9s} {'cb_med':>8s} {'submit_med':>10s}")
        for b in r["buckets"]:
            print(f"    {b['lo']:>5d}-{b['hi']:<6d} {b['n']:>5d} {b['total_median']:>10.2f} "
                  f"{b['total_mean']:>9.2f} {b['wait_median']:>9.2f} {b['cb_median']:>8.2f} "
                  f"{b['submit_median']:>10.2f}")

    if args.tsv:
        out = ["\t".join(["log", "decode_graphs", "bucket", "ord_lo", "ord_hi", "n",
                          "total_median_ms", "total_mean_ms", "wait_median_ms",
                          "cb_median_ms", "submit_median_ms"])]
        for r in rows:
            for i, b in enumerate(r["buckets"], 1):
                out.append("\t".join([r["log"], str(r["decode_graphs"]), str(i),
                                      str(b["lo"]), str(b["hi"]), str(b["n"]),
                                      f"{b['total_median']:.2f}", f"{b['total_mean']:.2f}",
                                      f"{b['wait_median']:.2f}", f"{b['cb_median']:.2f}",
                                      f"{b['submit_median']:.2f}"]))
        Path(args.tsv).write_text("\n".join(out) + "\n")
        print(f"\ntsv -> {args.tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
