#!/usr/bin/env python3
"""Is the expert-cache demand fill on the decode critical path?

The question this answers: `CGC_EB_TIMER=1` reports how long `ensure_batch` (which includes
the blocking demand fill) takes per decode step. That number is only worth engineering
against if the time is actually exposed on the critical path -- if the GPU were busy
throughout, removing the fill would buy nothing.

The test is a difference, not a model: run the SAME delivery cell twice on the SAME build,
once with fill on and once with `CGC_EB_NOFILL=1` (fill becomes a no-op; slots are still
allocated, bytes are simply not read, so the output is garbage -- this arm is timing-only).
If tg t/s rises by roughly fill_ms/step converted to t/s, the fill is on the critical path.

    python3 scripts/check/fill_onpath_ab.py \
        --fill  Backup/nofill_ab/fill_A.json:Backup/nofill_ab/wd_fill_A \
        --nofill Backup/nofill_ab/nofill_B.json:Backup/nofill_ab/wd_nofill_B

Each argument is `<bench_json>:<workdir>`; the workdir is where llama_bench_matrix.py puts the
sub-process stderr log that carries the `CGC-EBTIMER:` lines.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys

EB_RE = re.compile(
    r"CGC-EBTIMER: step_usec=(\d+) calls=(\d+) miss=(\d+) n_sum=(\d+)")


def load_ebtimer(workdir: str) -> list[tuple[int, int, int, int]]:
    """Collect `CGC-EBTIMER` rows from every stderr log under `workdir`."""
    rows: list[tuple[int, int, int, int]] = []
    for path in sorted(glob.glob(os.path.join(workdir, "**", "*.stderr.log"),
                                 recursive=True)):
        with open(path, errors="replace") as fh:
            for line in fh:
                m = EB_RE.search(line)
                if m:
                    rows.append(tuple(int(x) for x in m.groups()))
    return rows


def segment_profile(rows: list[tuple[int, int, int, int]],
                    nseg: int = 4) -> list[float]:
    """Per-segment median step time in ms, oldest segment first.

    Every rep re-loads the 13 GB model, so a reps>1 run has a cold start at the head of
    EACH rep, not just at the head of the log. That is why neither "the last N lines" nor
    "the last 25%" is safe: with reps=3 the tail crosses into the final rep's cold start and
    reports a number several times too large. Segmenting and taking the median OF the segment
    medians lets those cold heads be outvoted.
    """
    if not rows:
        return []
    k = max(1, len(rows) // nseg)
    prof = []
    for s in range(nseg):
        seg = rows[s * k:(s + 1) * k] if s < nseg - 1 else rows[s * k:]
        us = sorted(r[0] for r in seg)
        if us:
            prof.append(us[len(us) // 2] / 1000.0)
    return prof


def steady_ms(rows: list[tuple[int, int, int, int]], nseg: int = 4) -> float | None:
    """Median of the per-segment medians -- robust to a cold head AND a cold tail."""
    prof = segment_profile(rows, nseg)
    if not prof:
        return None
    p = sorted(prof)
    return p[len(p) // 2]


def decode_ts(bench_json: str) -> tuple[float | None, float | None]:
    """(avg t/s, stddev t/s) of the tg (decode) row -- or (None, None) when there is none.

    Both branches require `n_gen` to be truthy. The fallback must NOT accept an arbitrary
    `avg_ts` row: on a full-side cell (`-p 2048 -n 128`) an arm that died during decode still
    carries the PREFILL row (`n_gen=0`, ~300 t/s), and accepting it makes the pair report a
    several-thousand-percent "win" out of a failure. That is exactly what happened on
    2026-09-25 with `nf_nofill` (+2465%); the honest answer was "missing tg t/s".
    """
    with open(bench_json) as fh:
        data = json.load(fh)
    best = None
    for entry in (data if isinstance(data, list) else [data]):
        for row in entry.get("rows") or []:
            # llama-bench emits one row per (pp, tg); decode is the token-generation one.
            if row.get("n_gen") and (row.get("n_prompt") or 0) == 0:
                best = row
            elif best is None and row.get("avg_ts") and row.get("n_gen"):
                best = row
    if best is None:
        return (None, None)
    return (best.get("avg_ts"), best.get("stddev_ts"))


def _chk(ok: list[bool], cond: bool, label: str) -> None:
    ok.append(cond)
    print(f"  {'ok  ' if cond else 'FAIL'} {label}")


def self_test() -> int:
    ok: list[bool] = []
    print("selftest fill_onpath_ab")
    _chk(ok, EB_RE.search("CGC-EBTIMER: step_usec=21000 calls=40 miss=30 n_sum=752")
         .groups() == ("21000", "40", "30", "752"), "EBTIMER regex")
    _chk(ok, steady_ms([(1000, 40, 1, 752)] * 8) == 1.0, "steady_ms single value")
    rows = [(100_000, 40, 1, 1)] * 10 + [(10_000, 40, 1, 1)] * 30
    _chk(ok, steady_ms(rows) == 10.0, "steady_ms ignores cold head")
    _chk(ok, steady_ms([]) is None, "steady_ms empty -> None")
    # A cold tail must NOT be reported as steady either: this is the trap that made the
    # madvise control arm look 4x worse (tail crossing into the NEXT rep's cold start).
    _chk(ok, steady_ms([(10_000, 40, 1, 1)] * 40 + [(90_000, 40, 1, 1)] * 10) == 10.0,
         "steady_ms ignores a cold tail too")
    # ...and the same for a cold patch in the middle (a mid-run rep boundary).
    _chk(ok, steady_ms([(10_000, 40, 1, 1)] * 15 + [(90_000, 40, 1, 1)] * 5
                       + [(10_000, 40, 1, 1)] * 30) == 10.0,
         "steady_ms ignores a mid-run cold patch")
    # 2026-09-25 trap: a full-side arm (-p 2048 -n 128) that DIED during decode still carries
    # the prefill row (n_gen=0, ~300 t/s). Accepting it reported a +2465% "win" out of a
    # failure. `decode_ts` must return (None, None) so the pair reports "missing tg t/s".
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump([{"rows": [{"n_prompt": 2048, "n_gen": 0,
                              "avg_ts": 302.5, "stddev_ts": 10.5}]}], fh)
        tmp = fh.name
    try:
        _chk(ok, decode_ts(tmp) == (None, None), "decode_ts rejects a prefill-only arm")
    finally:
        os.unlink(tmp)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump([{"rows": [{"n_prompt": 2048, "n_gen": 0, "avg_ts": 302.5},
                             {"n_prompt": 0, "n_gen": 64, "avg_ts": 11.79,
                              "stddev_ts": 0.22}]}], fh)
        tmp = fh.name
    try:
        _chk(ok, decode_ts(tmp)[0] == 11.79, "decode_ts prefers the tg row when both exist")
    finally:
        os.unlink(tmp)
    print(f"selftest {sum(ok)}/{len(ok)}")
    return 0 if all(ok) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", help="<bench.json>:<workdir>")
    ap.add_argument("--nofill", help="<bench.json>:<workdir>")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not (args.fill and args.nofill):
        ap.error("--fill and --nofill are required unless --self-test")

    out = {}
    for name, spec in (("fill", args.fill), ("nofill", args.nofill)):
        bj, wd = spec.split(":", 1)
        ts, sd = decode_ts(bj)
        rows = load_ebtimer(wd)
        out[name] = {"ts": ts, "sd": sd, "ms": steady_ms(rows), "n_rows": len(rows)}
        print(f"{name:7s} tg={ts} t/s (sd {sd})  fill={out[name]['ms']} ms/step  "
              f"({len(rows)} EBTIMER rows)")

    a, b = out["fill"], out["nofill"]
    print()
    if not (a["ts"] and b["ts"]):
        print("VERDICT: unresolved -- missing tg t/s on one arm")
        return 2
    gain = (b["ts"] - a["ts"]) / a["ts"]
    print(f"nofill - fill = {b['ts'] - a['ts']:+.2f} t/s ({gain * 100:+.1f}%)")
    print(f"fill measured by EBTIMER: {a['ms']} ms/step (fill arm), "
          f"{b['ms']} ms/step (nofill arm -- should be ~0)")
    # What removing fill_ms from a step would be worth, if the step is the measured t/s.
    if a["ms"]:
        step_ms = 1000.0 / a["ts"]
        print(f"implied step: {step_ms:.1f} ms  ->  fill share {a['ms'] / step_ms * 100:.1f}%")
    if gain > 0.15:
        print("VERDICT: fill IS on the critical path (>=15% faster without it)")
    elif gain < -0.15:
        print("VERDICT: nofill is SLOWER -- unexpected, treat as noise/confound")
    else:
        print("VERDICT: fill is NOT a major critical-path term (difference within noise)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
