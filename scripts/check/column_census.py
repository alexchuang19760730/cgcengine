#!/usr/bin/env python3
"""Column census: which DECPROF column does the fill's 20.8 ms/step actually live in?

WHY THIS EXISTS
---------------
Two instruments disagree by ~10x about the cost of the expert-cache fill on the *same* engine:

  * `CGC_EB_TIMER=1` -> `CGC-EBTIMER: step_usec=<us> calls=<40> miss=<n> n_sum=<n>` once per step
    (llama-expert-cache.cpp:1155). It wraps the WHOLE `llama_expert_cache_ensure_batch` call --
    assignment loop, the synchronous demand fill, and the bg_cv wait. Cross-validated in
    docs/FILL_COST_MEASURED_2026-09-25.md, which reports **20.8 ms/step** at steady state.
  * `CGC_DECODE_PROFILE=1 CGC_DECODE_PROFILE_ALL=1` gives per-layer `wait` (polling for the segment
    to finish), `cb` (the top-k hook, which is where `ensure_batch` is CALLED from) and `submit`.
    On the same cell, decode steps sum to something much smaller in `cb`.

Those two cannot both describe the same step. Until this is settled, the "S1 + synchronous wait =
48.2 + 20.8 ms" arithmetic for shape 1 is unquotable in either direction -- and so is the claim
that the fill is too small to matter.

WHAT THE CENSUS IS (and is not)
-------------------------------
It is a *containment* test, not a pairing test. The two producers do not share a step index
(DECPROF prints on a cadence; EBTIMER prints every step), so the census compares per-step
DISTRIBUTIONS within one stratum:
  * stratum: `n_sum` from EBTIMER is the demands-per-step (40 layers x ~19 for decode), so
    `n_sum < 40 * 64` marks decode steps and larger values mark prefill chunks. Only decode is
    compared, because that is the regime every claim above is about.
  * the containment claim: `ensure_batch` runs INSIDE the hook, so EBTIMER's per-step sum must be
    <= the same step's `cb` sum. The ratio `step_usec / sum(cb)` is therefore the reading.

DECISION RULE -- FROZEN BEFORE THE RUN (2026-09-25, before the census arm was launched)
---------------------------------------------------------------------------------------
  FILL_IN_CB        : 0.5 <= ratio <= 2.0  -> the fill is inside `cb`, so its removable share is
                      `cb`'s share of the step, and `wait` (round-trip) is the real term.
  FILL_UNACCOUNTED  : ratio > 2.0          -> EBTIMER prices work that no DECPROF column holds;
                      the fill cost is real but invisible in this instrument set (instrument gap).
  FILL_UNDER_CB     : ratio < 0.5          -> EBTIMER sees less than the hook; the hook is doing
                      work that is not ensure_batch (then `cb` is not the fill's carrier either).
  INVALID           : no EBTIMER lines, no decode-stratum DECPROF block, or no step in either.

Selftest: `python3 scripts/check/column_census.py --selftest` (5 fixtures, including one where the
ratio must come out FILL_UNACCOUNTED and one where a missing producer must refuse).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

HDR_RE = re.compile(
    r"CGC-DECPROF: step=(?P<step>-?\d+) segs=(?P<segs>\d+) layers=(?P<layers>\d+) "
    r"total=(?P<total>[-\d.]+) ms \| wait=(?P<wait>[-\d.]+) \([-\d.]+%\) "
    r"cb=(?P<cb>[-\d.]+) \([-\d.]+%\) submit=(?P<sub>[-\d.]+) \([-\d.]+%\) "
    r"ntok=(?P<ntok>-?\d+)"
    # present only when CGC_GPU_TIMING=1: the device-side split of the same step
    r"(?: \| layer gpu_sum=(?P<gpu_sum>[-\d.]+) union_sum=(?P<union_sum>[-\d.]+) "
    r"gap_sum=(?P<gap_sum>[-\d.]+) ms)?")
ALL_RE = re.compile(r"CGC-DECPROF all: L(?P<layer>\d+) wait=(?P<wait>[-\d.]+) "
                    r"cb=(?P<cb>[-\d.]+) submit=(?P<submit>[-\d.]+) ms")
EB_RE = re.compile(r"CGC-EBTIMER: step_usec=(?P<usec>\d+) calls=(?P<calls>\d+) "
                   r"miss=(?P<miss>\d+) n_sum=(?P<nsum>\d+)")
FINAL_RE = re.compile(r"final stats: runtime requests=(?P<req>\d+) hits=(?P<hits>\d+) "
                      r"misses=(?P<misses>\d+) \(hit rate (?P<rate>[-\d.]+)%\)")

RATIO_LO, RATIO_HI = 0.5, 2.0     # frozen bands
DECODE_NSUM_MAX = 40 * 64         # decode steps stay far below a prefill chunk's demand count


def parse_decprof(text: str) -> tuple[list[dict], int]:
    blocks: list[dict] = []
    cur: dict | None = None
    bad = 0
    for line in text.splitlines():
        if (m := HDR_RE.search(line)):
            cur = {k: float(v) if k not in ("step", "segs", "layers", "ntok") and v is not None
                   else (None if v is None else int(v))
                   for k, v in m.groupdict().items()}
            cur["rows"] = {}
            blocks.append(cur)
            continue
        if "CGC-DECPROF all:" in line:
            if cur is None:
                bad += 1
                continue
            if (m := ALL_RE.search(line)):
                cur["rows"][int(m.group("layer"))] = {k: float(m.group(k))
                                                      for k in ("wait", "cb", "submit")}
            else:
                bad += 1
    return [b for b in blocks if b["rows"]], bad


def parse_ebtimer(text: str) -> tuple[list[dict], int]:
    steps: list[dict] = []
    bad = 0
    for line in text.splitlines():
        if "CGC-EBTIMER" not in line:
            continue
        if (m := EB_RE.search(line)):
            steps.append({k: int(v) for k, v in m.groupdict().items()})
        else:
            bad += 1
    return steps, bad


def census(stderr: str) -> dict:
    blocks, bad_dp = parse_decprof(stderr)
    eb, bad_eb = parse_ebtimer(stderr)
    dec = [b for b in blocks if b["ntok"] == 1]
    eb_dec = [s for s in eb if s["nsum"] < DECODE_NSUM_MAX]
    eb_pre = [s for s in eb if s["nsum"] >= DECODE_NSUM_MAX]
    fin = FINAL_RE.search(stderr)
    out: dict = {
        "n_blocks": len(blocks),
        "n_decode_blocks": len(dec),
        "n_ebtimer_steps": len(eb),
        "n_ebtimer_decode": len(eb_dec),
        "n_ebtimer_prefill": len(eb_pre),
        "n_unparsed_decprof_lines": bad_dp,
        "n_unparsed_ebtimer_lines": bad_eb,
        "pool": ({k: (int(v) if k in ("req", "hits", "misses") else float(v))
                  for k, v in fin.groupdict().items()} if fin else None),
    }
    if not eb:
        out["verdict"] = {"verdict": "INVALID", "why": "no CGC-EBTIMER line (CGC_EB_TIMER not armed "
                                                      "or not forwarded by the launcher)"}
        return out
    if not dec:
        out["verdict"] = {"verdict": "INVALID", "why": "no decode-stratum DECPROF block"}
        return out

    def per_step(key: str, block: dict) -> float:
        return sum(r[key] for r in block["rows"].values())

    sums = {"wait": [per_step("wait", b) for b in dec],
            "cb": [per_step("cb", b) for b in dec],
            "submit": [per_step("submit", b) for b in dec]}
    totals = [b["total"] for b in dec]
    med = {k: st.median(v) for k, v in sums.items()}
    total_med = st.median(totals)
    eb_med = st.median([s["usec"] / 1000.0 for s in eb_dec]) if eb_dec else None
    ratio = (eb_med / med["cb"]) if (eb_med is not None and med["cb"] > 0) else None

    if ratio is None:
        verdict, why = "INVALID", "no decode-stratum EBTIMER step or zero cb"
    elif RATIO_LO <= ratio <= RATIO_HI:
        verdict = "FILL_IN_CB"
        why = (f"EBTIMER step median {eb_med:.2f} ms is within {RATIO_LO}-{RATIO_HI}x of the "
               f"decode-step cb sum {med['cb']:.2f} ms -> the fill is inside cb")
    elif ratio > RATIO_HI:
        verdict = "FILL_UNACCOUNTED"
        why = (f"EBTIMER {eb_med:.2f} ms is >{RATIO_HI}x the cb sum {med['cb']:.2f} ms -> EBTIMER "
               f"prices work no DECPROF column holds")
    else:
        verdict = "FILL_UNDER_CB"
        why = (f"EBTIMER {eb_med:.2f} ms is <{RATIO_LO}x the cb sum {med['cb']:.2f} ms -> the hook "
               f"does work that is not ensure_batch")
    out["verdict"] = {"verdict": verdict, "why": why, "ratio_ebtimer_over_cb":
                      None if ratio is None else round(ratio, 3)}
    # the device-side split, when CGC_GPU_TIMING=1 was on: does `wait` contain real GPU work or
    # a gap (GPU idle between commands)?  This is what decides whether overlap has anything to take.
    dev = {k: [b[k] for b in dec if b.get(k) is not None]
           for k in ("gpu_sum", "union_sum", "gap_sum")}
    if all(v for v in dev.values()):
        split = {k: st.median(v) for k, v in dev.items()}
        out["device_split_ms"] = {k: round(v, 3) for k, v in split.items()}
        # The device numbers are sums over the step's 41 segments, so they are NOT shares of `wait`
        # (gpu_sum legitimately exceeds it when segments overlap). The comparable pairing is inside
        # the device span: how much of it is gap (device idle between commands) vs union (working).
        span = split["union_sum"] + split["gap_sum"]
        out["gap_share_of_device_span"] = round(split["gap_sum"] / span, 4) if span else None
    out["decode_step_ms"] = {k: round(v, 3) for k, v in med.items()}
    out["decode_step_total_ms"] = round(total_med, 3)
    out["decode_step_shares"] = {k: round(v / total_med, 4) for k, v in med.items()} \
        if total_med else {}
    out["ebtimer_decode_ms"] = None if eb_med is None else round(eb_med, 3)
    out["ebtimer_prefill_ms_median"] = (round(st.median([s["usec"] / 1000.0 for s in eb_pre]), 2)
                                        if eb_pre else None)
    out["ebtimer_misses_per_step_median"] = (st.median([s["miss"] for s in eb_dec])
                                             if eb_dec else None)
    out["decprof_step_total_check"] = rounded_check(med, total_med)
    return out


def rounded_check(med: dict, total_med: float) -> dict:
    """Do the three columns add up to the header's own total? A mismatch means the header (which the
    shares are computed against) and the per-layer rows describe different things."""
    s = sum(med.values())
    return {"columns_sum_ms": round(s, 3), "header_total_ms": round(total_med, 3),
            "delta_ms": round(s - total_med, 3),
            "consistent": abs(s - total_med) <= max(1.0, 0.05 * total_med)}


# ── fixtures ────────────────────────────────────────────────────────────────────────

def _dec(steps: int, wait: float, cb: float, sub: float, ntok: int = 1) -> str:
    out = []
    for i in range(steps):
        out.append(f"CGC-DECPROF: step={i + 1} segs=41 layers=40 total={wait + cb + sub:.2f} ms | "
                   f"wait={wait:.2f} (0%) cb={cb:.2f} (0%) submit={sub:.2f} (0%) ntok={ntok} | "
                   f"layer gpu_sum=1.00 union_sum=1.00 gap_sum=1.00 ms")
        for l in range(40):
            out.append(f"CGC-DECPROF all: L{l} wait={wait / 40:.3f} cb={cb / 40:.3f} "
                       f"submit={sub / 40:.3f} ms gpu=0.5 union=0.5 gap=0.5 sg=1 n=19 st=0.0 en=0.5")
    return "\n".join(out) + "\n"


def _eb(steps: int, usec: int, nsum: int = 752, miss: int = 20) -> str:
    return "".join(f"CGC-EBTIMER: step_usec={usec} calls=40 miss={miss} n_sum={nsum}\n"
                   for _ in range(steps))


def selftest() -> int:
    fails = 0

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal fails
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + extra) if extra and not cond else ''}")
        if not cond:
            fails += 1

    # (a) the fill is inside cb: EBTIMER 6 ms vs cb sum 6 ms
    a = census(_dec(8, 60.0, 6.0, 4.0) + _eb(8, 6000))
    check("fill-inside-cb fixture -> FILL_IN_CB", a["verdict"]["verdict"] == "FILL_IN_CB",
          str(a["verdict"]))
    check("  columns add up to the header total", a["decprof_step_total_check"]["consistent"],
          str(a["decprof_step_total_check"]))
    # shares are rounded to 4 decimals for the artifact, so the sum is 1.0 within that rounding
    check("  shares are computed against the header total",
          abs(sum(a["decode_step_shares"].values()) - 1.0) < 1e-3, str(a["decode_step_shares"]))

    # (a2) the device split is read only when the header carries it
    a2 = census(_dec(8, 60.0, 6.0, 4.0).replace(
        "gap_sum=1.00 ms", "gap_sum=40.00 ms") + _eb(8, 6000))
    check("device split parsed from the header",
          a2.get("device_split_ms", {}).get("gap_sum") == 40.0, str(a2.get("device_split_ms")))
    no_dev = census(_dec(8, 60.0, 6.0, 4.0).replace(
        " | layer gpu_sum=1.00 union_sum=1.00 gap_sum=1.00 ms", "") + _eb(8, 6000))
    check("  absent device split is absent, not zero", "device_split_ms" not in no_dev,
          str(no_dev.get("device_split_ms")))

    # (b) EBTIMER prices 20 ms that cb cannot hold -> FILL_UNACCOUNTED
    b = census(_dec(8, 60.0, 6.0, 4.0) + _eb(8, 20000))
    check("20.8-vs-cb fixture -> FILL_UNACCOUNTED", b["verdict"]["verdict"] == "FILL_UNACCOUNTED",
          str(b["verdict"]))

    # (c) no EBTIMER at all -> INVALID (the launcher-dropped case looks exactly like this)
    c = census(_dec(8, 60.0, 6.0, 4.0))
    check("no EBTIMER -> INVALID", c["verdict"]["verdict"] == "INVALID", str(c["verdict"]))

    # (d) prefill chunks must be excluded from the decode stratum
    d = census(_dec(8, 60.0, 6.0, 4.0) + _eb(3, 200000, nsum=81920) + _eb(6, 6000))
    check("prefill chunks are a separate stratum",
          d["n_ebtimer_prefill"] == 3 and d["n_ebtimer_decode"] == 6,
          f"{d['n_ebtimer_prefill']}/{d['n_ebtimer_decode']}")
    check("  and the decode-only reading survives them",
          d["verdict"]["verdict"] == "FILL_IN_CB", str(d["verdict"]))

    # (e) a malformed EBTIMER line is counted, and a prefill-only run refuses
    e = census(_dec(8, 60.0, 6.0, 4.0) + _eb(4, 6000) + "CGC-EBTIMER: step_usec=oops calls=40\n")
    check("malformed EBTIMER line counted", e["n_unparsed_ebtimer_lines"] == 1,
          str(e["n_unparsed_ebtimer_lines"]))
    f = census(_dec(4, 60.0, 6.0, 4.0, ntok=512) + _eb(4, 6000, nsum=20480))
    check("prefill-only run -> INVALID", f["verdict"]["verdict"] == "INVALID", str(f["verdict"]))

    print(f"\nselftest {'OK' if fails == 0 else f'FAILED ({fails})'}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stderr", help="one arm's llama-bench stderr log")
    ap.add_argument("--json", help="write the census here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.stderr:
        ap.error("--stderr is required (or --selftest)")
    res = census(Path(args.stderr).read_text(errors="replace"))
    print(f"blocks={res['n_blocks']} (decode {res['n_decode_blocks']})  "
          f"EBTIMER steps={res['n_ebtimer_steps']} (decode {res['n_ebtimer_decode']}, "
          f"prefill {res['n_ebtimer_prefill']})  unparsed: dp={res['n_unparsed_decprof_lines']} "
          f"eb={res['n_unparsed_ebtimer_lines']}")
    if "decode_step_ms" in res:
        t = res["decode_step_total_ms"]
        print(f"\ndecode step (median, header total {t} ms):")
        for k in ("wait", "cb", "submit"):
            print(f"  {k:<7} {res['decode_step_ms'][k]:8.2f} ms  {res['decode_step_shares'][k]:6.1%}")
        print(f"  columns sum check: {res['decprof_step_total_check']}")
        print(f"  EBTIMER step_usec (decode): {res['ebtimer_decode_ms']} ms  "
              f"(misses/step median {res['ebtimer_misses_per_step_median']})")
        if "device_split_ms" in res:
            d = res["device_split_ms"]
            print(f"  device side (sums over the step's segments, not shares of wait): "
                  f"gpu_sum {d['gpu_sum']} union_sum {d['union_sum']} gap_sum {d['gap_sum']} ms"
                  f"  gap = {res['gap_share_of_device_span']:.0%} of the device span")
    v = res["verdict"]
    print(f"\nVERDICT: {v['verdict']}\n  {v['why']}")
    print(f"\npre-registered bands: ratio(fill/cb) in [{RATIO_LO}, {RATIO_HI}] => FILL_IN_CB")
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"json -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
