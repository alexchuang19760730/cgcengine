#!/usr/bin/env python3
"""Generic two-arm decode pair comparison on the delivery cell.

`fill_onpath_ab.py` hard-codes the arm names (fill / nofill) because it answers exactly one
question. This is the reusable version: **any** two arms, with labels, reporting tg t/s plus
the *mechanistic* side evidence -- EBTIMER fill ms/step, the `cache` counters from the bench
json, and `-- thermal` -- so a win can be checked against its own mechanism instead of being
believed on throughput alone.

    python3 scripts/check/pair_ab.py \
        --a Backup/layer_ahead/a1_ctrl.json:Backup/layer_ahead/wd_a1_ctrl \
        --b Backup/layer_ahead/a2_la.json:Backup/layer_ahead/wd_a2_la \
        --alabel ctrl --blabel "la=1"

Each arm is `<bench_json>:<workdir>`; the workdir is where `llama_bench_matrix.py` drops the
sub-process stderr log carrying the `CGC-EBTIMER:` lines.

Two things this adds over reading the numbers by hand:

- **thermal is printed per arm.** A pair measured through anything other than NOMINAL cannot
  be cited at all (`cgc-prefill-thermal-delivery`), and this makes that impossible to miss.
- **the mechanism check.** If an arm claims to cut fill but its EBTIMER number does not move,
  or its `io_effective_mib_s` does not budge, the win came from somewhere else -- worth
  knowing before writing any C++.

⚠ `pread_usec` / `per_miss_us` in the bench json are **accumulated across workers** and can
exceed wall time; only `io_effective_mib_s` and the per-job mean are safe to compare.
"""
from __future__ import annotations

import argparse
import json

import fill_onpath_ab as F


def load_meta(bench_json: str) -> dict:
    """(cache block, thermal labels) from the bench json written by llama_bench_matrix."""
    with open(bench_json) as fh:
        data = json.load(fh)
    entry = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})
    return {
        "cache": entry.get("cache") or {},
        "thermal": entry.get("thermal") or {},
    }


def thermal_label(th: dict) -> str:
    """`launch/worst` thermal labels -- anything but NOMINAL makes the arm uncitable."""
    launch = (th.get("launch") or {}).get("label", "?")
    worst = (th.get("worst") or {}).get("label", "?")
    return f"{launch}/{worst}"


def _fmt(v: object, unit: str = "", nd: int = 1) -> str:
    """Render a metric for the report row; anything unexpected degrades instead of raising.

    The degradation is deliberate: this runs at the end of a ~10 minute GPU pair, and a
    TypeError here would throw away both arms.
    """
    if v is None:
        return "--"
    if isinstance(v, bool):
        return f"{v}{unit}"
    if isinstance(v, int):
        return f"{v}{unit}"
    if isinstance(v, float):
        return f"{v:.{nd}f}{unit}"
    return f"{v}{unit}"


def report(name: str, spec: str) -> dict:
    bj, wd = spec.split(":", 1)
    ts, sd = F.decode_ts(bj)
    rows = F.load_ebtimer(wd)
    ms = F.steady_ms(rows)
    meta = load_meta(bj)
    c = meta["cache"]
    tl = thermal_label(meta["thermal"])
    warn = "" if tl == "NOMINAL/NOMINAL" else "  ⚠ NOT CITABLE"
    print(f"{name:>8s}  tg={_fmt(ts, ' t/s', 2)} (sd {_fmt(sd, '', 2)})"
          f"  fill={_fmt(ms, ' ms/step')}"
          f"  hit={_fmt(c.get('hit_rate_pct'), '%')}"
          f" miss={_fmt(c.get('misses'))}"
          f" evict={_fmt(c.get('evictions'))}"
          f" io={_fmt(c.get('io_effective_mib_s'), ' MiB/s')}"
          f"  [{tl}] [{len(rows)} rows]{warn}")
    return {"ts": ts, "sd": sd, "ms": ms, "cache": c, "thermal": tl}


def self_test() -> int:
    ok: list[bool] = []

    def chk(cond: bool, label: str) -> None:
        ok.append(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")

    print("selftest pair_ab")
    chk(_fmt(None) == "--", "None renders as --")
    chk(_fmt(12.567, " t/s", 2) == "12.57 t/s", "float formatting honours precision")
    chk(_fmt(43128) == "43128", "int renders without decimals")
    chk(thermal_label({"launch": {"label": "NOMINAL"}, "worst": {"label": "NOMINAL"}})
        == "NOMINAL/NOMINAL", "thermal label both NOMINAL")
    chk(thermal_label({"launch": {"label": "NOMINAL"}, "worst": {"label": "HEAVY"}})
        == "NOMINAL/HEAVY", "thermal label reports worst too")
    chk(thermal_label({}) == "?/?", "thermal label survives missing block")
    chk(_fmt({}, "x") == "{}x",
        "non-numeric degrades instead of crashing away a whole pair")
    chk(_fmt(True, "") == "True", "bool renders before int (True is not 1 here)")
    print(f"selftest {sum(ok)}/{len(ok)}")
    return 0 if all(ok) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", help="<bench.json>:<workdir>")
    ap.add_argument("--b", help="<bench.json>:<workdir>")
    ap.add_argument("--alabel", default="A")
    ap.add_argument("--blabel", default="B")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="relative tg t/s gain to call it a win (default 0.15)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not (args.a and args.b):
        ap.error("--a and --b are required unless --self-test")

    ra = report(args.alabel, args.a)
    rb = report(args.blabel, args.b)
    print()

    if "NOMINAL/NOMINAL" not in (ra["thermal"], rb["thermal"]):
        print("⚠ one arm is not NOMINAL/NOMINAL -- do not cite the magnitudes below, "
              "only their direction.")

    if not (ra["ts"] and rb["ts"]):
        print("VERDICT: unresolved -- missing tg t/s on one arm")
        return 2

    gain = (rb["ts"] - ra["ts"]) / ra["ts"]
    print(f"{args.blabel} - {args.alabel} = {rb['ts'] - ra['ts']:+.2f} t/s ({gain * 100:+.1f}%)")
    print(f"implied step: {1000.0 / ra['ts']:.1f} ms -> {1000.0 / rb['ts']:.1f} ms"
          f"  (threshold {args.threshold * 100:.0f}%)")

    # Mechanism check: does the fill accounting agree with the throughput story?
    if ra["ms"] and rb["ms"]:
        dms = rb["ms"] - ra["ms"]
        print(f"fill ms/step: {ra['ms']:.1f} -> {rb['ms']:.1f}  ({dms:+.1f})")
        if gain >= args.threshold and dms >= 0:
            print("  ⚠ mechanism DISAGREES: faster, but fill did not drop. "
                  "Find the confound before trusting this pair.")
        elif gain <= -args.threshold and dms <= 0:
            print("  ⚠ mechanism DISAGREES: slower, yet fill dropped. "
                  "Something other than fill dominates.")
        elif abs(gain) < args.threshold:
            print("  mechanism is moot: throughput difference is inside the noise floor.")

    io_a = ra["cache"].get("io_effective_mib_s")
    io_b = rb["cache"].get("io_effective_mib_s")
    if io_a and io_b:
        print(f"effective read: {io_a:.1f} -> {io_b:.1f} MiB/s "
              f"({(io_b - io_a) / io_a * 100:+.0f}%)")

    if gain >= args.threshold:
        print(f"VERDICT: WIN for {args.blabel} (>= {args.threshold * 100:.0f}% faster)")
    elif gain <= -args.threshold:
        print(f"VERDICT: LOSS for {args.blabel} (>= {args.threshold * 100:.0f}% slower -- "
              "expected? if not, suspect a confound)")
    else:
        print(f"VERDICT: no separation (< {args.threshold * 100:.0f}%). "
              "Single-arm noise floor is ~±27%, so call this unresolved rather than zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
