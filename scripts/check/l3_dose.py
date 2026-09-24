#!/usr/bin/env python3
"""CGC-L3DOSE reader -- the L3 dose of a decode step, and whether the async clause is armed.

One line per decode step, emitted by llama-context.cpp (M0, 2026-09-20):

    CGC-L3DOSE: ctx=verify ntok=4  pool_wait=32.400 fill_wait=0.000 ms
                await_missed=0 armed=0  la_entered=40 la_pred_missing=40
                la_pred_empty=0 la_pred_ok=0 la_issued=0

    pool_wait   time in `pool_done_cv.wait(pool_outstanding == 0)` -- the demand batch's own block,
                i.e. the term L3 (issue-and-return) would move off the critical path.
    fill_wait   bg_cv waits + serial fill_pool_direct preads. It does NOT contain pool_wait, which
                is why reading fill_wait alone reads 0.000 whatever the overlap does.
    la_*        layer-ahead trigger census (entered / no prediction / empty union / candidates /
                prefetch attempts). F2R §7: without it, "predicate false" and "never reached" were
                the same reading.

Values are PER STEP (summed over the step's layers). M-F5's per-call figures are ~1/40 of these;
dividing one by the other is the per-job-vs-per-step confusion this repo already paid for (F3).

Two rules, enforced rather than narrated:
  * NO LINES IS `INVALID`, NOT `PASS`: an absent line means the flag never reached the process, or
    this step did not take that path -- an absence, not a clean read.
  * `await_missed > 0` with `armed=1` is `FAIL` (a torn read is possible). While `armed=0` there is
    no async split, so a 0 is reported as "not checked", never as "clean".

    python3 scripts/check/l3_dose.py --log Backup/cgc_logs/llama_server_*.log
    python3 scripts/check/l3_dose.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import statistics as st
import sys

RE = re.compile(
    r"CGC-L3DOSE:\s+ctx=(?P<ctx>\S+)\s+ntok=(?P<ntok>\d+)\s+"
    r"pool_wait=(?P<pw>[\d.]+)\s+(?:pool_submit=[\d.]+\s+)?fill_wait=(?P<fw>[\d.]+)\s+ms\s+"
    r"await_missed=(?P<am>\d+)\s+armed=(?P<armed>\d+)\s+"
    r"la_entered=(?P<lae>\d+)\s+la_pred_missing=(?P<lap>\d+)\s+"
    r"la_pred_empty=(?P<laem>\d+)\s+la_pred_ok=(?P<laok>\d+)\s+la_issued=(?P<lai>\d+)"
)


INT_FIELDS = ("ntok", "am", "armed", "lae", "lap", "laem", "laok", "lai")


def parse(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                m = RE.search(line)
                if not m:
                    continue
                rows.append({"ctx": m.group("ctx"),
                             "pw": float(m.group("pw")), "fw": float(m.group("fw")),
                             **{k: int(m.group(k)) for k in INT_FIELDS}})
    return rows


def summarise(rows: list[dict]) -> dict:
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["ctx"], r["ntok"]), []).append(r)
    return {f"{ctx}/ntok={ntok}": {
        "n": len(rs),
        "pool_wait_ms": round(float(st.median([r["pw"] for r in rs])), 3),
        "fill_wait_ms": round(float(st.median([r["fw"] for r in rs])), 3),
        "pool_wait_on": sum(1 for r in rs if r["pw"] > 0),
        "await_missed": sum(r["am"] for r in rs),
        "armed": sorted({r["armed"] for r in rs}),
        **{k: sum(r[k] for r in rs) for k in ("lae", "lap", "laem", "laok", "lai")},
    } for (ctx, ntok), rs in sorted(groups.items())}


def verdict(summary: dict, n_lines: int) -> tuple[str, str]:
    if n_lines == 0:
        return "INVALID", ("no CGC-L3DOSE lines: the flag never reached the process, or these steps "
                           "did not take the path that emits it. An absence, not a PASS.")
    armed = {a for g in summary.values() for a in g["armed"]}
    missed = sum(g["await_missed"] for g in summary.values())
    if armed - {0} and missed > 0:
        return "FAIL", (f"armed={armed} with await_missed={missed} > 0 -- an awaited layer's bytes "
                        "may not have landed. Quote no speed number from this run.")
    if all(g["pool_wait_on"] == 0 for g in summary.values()):
        return "READABLE-BUT-ZERO", ("line present, pool_wait 0 on every step: the pool path is "
                                     "probably not the one serving them (check the geometry line). "
                                     "Not the same as 'L3 has no target'.")
    what = ("armed=0, so await_missed=0 means NOT CHECKED (no async split exists)"
            if armed == {0} else f"armed={sorted(armed)}")
    return "READABLE", f"dose readable on {n_lines} steps; {what}."


def render(summary: dict, n_lines: int) -> str:
    out = [f"CGC-L3DOSE lines: {n_lines} (one per decode step; values are per-step sums)"]
    for name, g in summary.items():
        out.append(f"  {name:<14} n={g['n']:<4} pool_wait={g['pool_wait_ms']:.3f} "
                   f"fill_wait={g['fill_wait_ms']:.3f} ms  (pool_wait>0 on "
                   f"{g['pool_wait_on']}/{g['n']})")
        out.append(f"                 await_missed={g['await_missed']} armed={g['armed']}  la: "
                   f"entered={g['lae']} pred_missing={g['lap']} pred_empty={g['laem']} "
                   f"pred_ok={g['laok']} issued={g['lai']}")
    kind, why = verdict(summary, n_lines)
    out.append(f"VERDICT: {kind} -- {why}")
    return "\n".join(out)


# ── selftest: the shape that parses, and the three that must not read as PASS ──
GOOD = ("CGC-L3DOSE: ctx=verify ntok=4  pool_wait=52.400 fill_wait=3.000 ms  await_missed=0 "
        "armed=0  la_entered=40 la_pred_missing=12 la_pred_empty=0 la_pred_ok=28 la_issued=0\n"
        "CGC-L3DOSE: ctx=draft ntok=1  pool_wait=11.200 fill_wait=0.900 ms  await_missed=0 "
        "armed=0  la_entered=0 la_pred_missing=0 la_pred_empty=0 la_pred_ok=0 la_issued=0\n")
ARMED_MISSED = ("CGC-L3DOSE: ctx=verify ntok=4  pool_wait=52.400 fill_wait=3.000 ms  "
                "await_missed=2 armed=1  la_entered=40 la_pred_missing=0 la_pred_empty=0 "
                "la_pred_ok=40 la_issued=12\n")
ZERO_DOSE = ("CGC-L3DOSE: ctx=verify ntok=4  pool_wait=0.000 fill_wait=0.000 ms  await_missed=0 "
             "armed=0  la_entered=0 la_pred_missing=0 la_pred_empty=0 la_pred_ok=0 la_issued=0\n")


def selftest() -> int:
    import tempfile
    checks = 0
    bad = 0

    def read(text: str) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write(text)
            path = fh.name
        try:
            rows = parse([path])
        finally:
            os.unlink(path)
        return rows, summarise(rows), len(rows)

    for name, text, want in (("good", GOOD, "READABLE"), ("armed+missed", ARMED_MISSED, "FAIL"),
                             ("zero dose", ZERO_DOSE, "READABLE-BUT-ZERO"),
                             ("absent", "nothing here\n", "INVALID")):
        _, s, n = read(text)
        got = verdict(s, n)[0]
        checks += 1
        bad += 0 if got == want else 1
        print(f"  [{'ok ' if got == want else 'BAD'}] {name:<12} want={want:<17} got={got}")

    _, s, _ = read(GOOD)
    g = s["verify/ntok=4"]
    for field, want in (("pool_wait_ms", 52.4), ("fill_wait_ms", 3.0),
                        ("laok", 28), ("pool_wait_on", 1)):
        checks += 1
        ok = abs(g[field] - want) < 1e-6
        bad += 0 if ok else 1
        print(f"  [{'ok ' if ok else 'BAD'}] {field:<12} want={want} got={g[field]}")
    print(f"l3_dose selftest: {checks - bad}/{checks} passed")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", default=[], help="server log(s); globs allowed")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    paths = [p for pat in args.log for p in sorted(glob.glob(pat)) or [pat]]
    paths = [p for p in paths if os.path.exists(p)]  # llama_server_latest.log dangles between runs
    if not paths:
        ap.error("give --log (the engine prints this to stderr; run_server.sh writes it into "
                 "Backup/cgc_logs/llama_server_*.log)")
    rows = parse(paths)
    summary = summarise(rows)
    print(render(summary, len(rows)))
    return 0 if verdict(summary, len(rows))[0].startswith("READABLE") else 1


if __name__ == "__main__":
    sys.exit(main())
