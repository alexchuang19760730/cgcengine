#!/usr/bin/env python3
"""Measure one profile on BOTH axes -- prefill and decode -- and never report one without the other.

WHY THIS EXISTS. Every throughput claim in this repo is a pair: a profile is only "the fast one"
if it is fast at BOTH prefill and decode, because a served request pays both. But the existing
tools let you measure either, so the record drifted into single-axis numbers that could not be
combined: `prefill250` was quoted for prefill and `prod25` for decode, and nothing ever showed
one profile holding both.

The second reason is a gate, and it is the load-bearing part. `prod_matrix.py` *records* the
thermal level at launch but does not *wait* for it: it runs cells back to back, so arms 2..N are
measured on a machine that arm 1 just heated. On 2026-09-18 that produced a four-cell run whose
first arm launched NOMINAL and whose other three launched HEAVY -- i.e. three of four numbers
were unusable, and the only thing that caught it was a human reading the labels afterwards.
Here the wait is BEFORE the launch, and a cell whose launch label is not NOMINAL is reported
but explicitly refused as a bar-meeting number.

WHAT IT MEASURES. One llama-bench launch per (profile, cell) -- never two shapes in one process.
Default cells are the two production axes:
    prefill-house  -p 2048 -n 16  -d 0    (the shape behind the recorded 276.25 / 300.43)
    decode         -p 0    -n 128 -d 512  (the warm-platform decode standard)

USAGE
    python3 scripts/check/profile_duo.py --profiles prefill250,prod25
    python3 scripts/check/profile_duo.py --profiles prefill250 --reps 3 \
        --json Backup/prod_matrix/duo.json --md docs/duo.md

Exit code 0 always: the point is the table, and a refused bar is a finding, not a crash.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import thermal_pressure as tp  # noqa: E402
import prod_matrix as pm  # noqa: E402

# The two axes. Anything that reports a profile must report both -- see the module docstring.
PREFILL_CELL = "prefill-house"
DECODE_CELL = "decode"
DEFAULT_CELLS = f"{PREFILL_CELL},{DECODE_CELL}"

BAR_PREFILL = 250.0
BAR_DECODE = 12.0


def wait_nominal(timeout: float, poll: float, quiet: bool = False) -> dict:
    """Block until the OS reports level 0. Returns `{ok, waited_s, level, label}`.

    The timeout is a refusal, not a green light: if the machine has not cooled in `timeout`
    seconds the caller must not launch, because launching is what produces the number that
    later gets quoted as if it were NOMINAL. Note that cooldown is NOT a constant -- the
    recorded range is 35 s to well over 120 s -- so a fixed sleep cannot substitute.
    """
    t0 = time.time()
    while True:
        lv = tp.level()
        if lv == 0:
            return {"ok": True, "waited_s": round(time.time() - t0, 1),
                    "level": lv, "label": tp.label(lv)}
        if time.time() - t0 > timeout:
            return {"ok": False, "waited_s": round(time.time() - t0, 1),
                    "level": lv, "label": tp.label(lv)}
        if not quiet:
            print(f"    thermal {tp.label(lv)} -- waiting for NOMINAL "
                  f"({time.time() - t0:.0f}s / {timeout:.0f}s)", flush=True)
        time.sleep(poll)


def one(profile: str, cell: str, args, waited: dict) -> dict:
    out = pm.run_cell(profile, cell, args)
    out["waited"] = waited
    th = out.get("thermal") or {}
    out["launch_label"] = (th.get("launch") or {}).get("label", "UNREADABLE")
    out["worst_label"] = (th.get("worst") or {}).get("label", "UNREADABLE")
    # Launching from NOMINAL is NOT sufficient, and this cost a whole wrong conclusion: on
    # 2026-09-18 four interleaved arms all launched NOMINAL yet gave 10.80 / 8.58 / 8.92 / 10.34,
    # and the ordering tracked the worst level reached DURING the arm, not the engine under test
    # (NOMINAL 109/109 -> 10.80; MODERATE -> 10.34; HEAVY x17 -> 8.92; HEAVY x33 -> 8.58). A long
    # decode arm heats the machine by itself. So the number is only quotable when the arm never
    # left NOMINAL, and that is a separate claim from "it launched clean".
    out["nominal_ok"] = out["launch_label"] == "NOMINAL"
    out["clean"] = out["nominal_ok"] and out["worst_label"] == "NOMINAL"
    return out


def val(res: dict) -> float | None:
    pl = list((res.get("platform") or {}).values())
    v = pl[0] if pl else {}
    return v.get("platform_ts")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="prefill AND decode, per profile, each arm launched only from NOMINAL")
    ap.add_argument("--profiles", default="prefill250,prod25",
                    help="comma-separated profile names (parsed from run_server.sh)")
    ap.add_argument("--cells", default=DEFAULT_CELLS,
                    help=f"comma-separated cells (default {DEFAULT_CELLS})")
    ap.add_argument("--reps", type=int, default=3, help="llama-bench repetitions (rep 1 dropped)")
    ap.add_argument("--logdir", default="Backup/prod_matrix")
    ap.add_argument("--extra-env", default="", help="ENV=VAL;ENV=VAL passed through to the server env")
    ap.add_argument("--json", default=None, help="write the raw results here")
    ap.add_argument("--md", default=None, help="write a markdown table here")
    ap.add_argument("--cooldown-timeout", type=float, default=420.0,
                    help="give up waiting for NOMINAL after this many seconds (a refusal)")
    ap.add_argument("--poll", type=float, default=5.0)
    # 30.0 mirrors prod_matrix.py's own default, so a duo run refuses exactly what a matrix
    # run would refuse. Do not "improve" this here: a stricter floor would make the two tools
    # disagree about whether a launch was legal.
    ap.add_argument("--min-usable-pct", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bin-dir", default=None,
                    help="run llama-bench from THIS directory instead of the built one. How a "
                         "bisect is done without rebuilding: `git archive <old-commit> "
                         "src/llama.cpp/build/bin` unpacks the binaries that commit shipped. "
                         "Rebuilding is not just slow here -- it WRITES the artifacts other "
                         "sessions are currently mapping, which corrupts their runs.")
    args = ap.parse_args()

    if args.bin_dir:
        cand = Path(args.bin_dir) / "llama-bench"
        if not cand.exists():
            print(f"error: no llama-bench in {args.bin_dir}", file=sys.stderr)
            return 2
        # The matrix resolves the binary from a module-level constant; repoint it at the checkout.
        # `llama_bench_matrix` must be imported HERE, not reached through `pm`: prod_matrix only did
        # `from llama_bench_matrix import ...`, which does not bind the submodule as an attribute.
        # Importing it directly yields the same object in sys.modules, so prod_matrix sees the change.
        import llama_bench_matrix as lbm
        lbm.LLAMA_BENCH = cand
        print(f"note: bisect mode -- llama-bench from {cand}", flush=True)

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    known = pm.profiles_from_run_server()
    bad = [p for p in profiles if p not in known]
    if bad:
        print(f"error: unknown profile(s) {bad}; known: {known}", file=sys.stderr)
        return 2
    if PREFILL_CELL not in cells or DECODE_CELL not in cells:
        print(f"note: cells {cells} do not include both {PREFILL_CELL} and {DECODE_CELL} -- "
              f"the result is not a duo and cannot settle 'highest at both'", file=sys.stderr)

    results = []
    for profile in profiles:
        for cell in cells:
            print(f"=== {profile} / {cell}", flush=True)
            waited = wait_nominal(args.cooldown_timeout, args.poll)
            if not waited["ok"]:
                print(f"  REFUSED: machine still {waited['label']} after "
                      f"{waited['waited_s']}s -- not launching", flush=True)
                results.append({"profile": profile, "cell": cell, "refused": True,
                                "waited": waited, "nominal_ok": False,
                                "launch_label": waited["label"]})
                continue
            r = one(profile, cell, args, waited)
            v = val(r)
            print(f"  launch={r['launch_label']:<9} waited={waited['waited_s']}s  "
                  f"platform={v if v is not None else '-'}", flush=True)
            if r.get("incompatible"):
                print(f"  incompatible: {r.get('reason')}", flush=True)
            results.append(r)

    # ---- the table: prefill and decode on ONE row, always both ----
    print("\n" + "=" * 88)
    print(f"{'profile':<14}{'prefill t/s':>13}{'decode t/s':>13}"
          f"{'pf worst':>10}{'dc worst':>10}{'bar 250/12':>13}")
    print("-" * 88)
    table = []
    for profile in profiles:
        row = {"profile": profile}
        for cell in cells:
            r = next((x for x in results if x["profile"] == profile and x["cell"] == cell), None)
            row[cell] = val(r) if r else None
            row[cell + "_launch"] = r["launch_label"] if r else "-"
            row[cell + "_worst"] = r.get("worst_label", "-") if r else "-"
            row[cell + "_clean"] = r.get("clean", False) if r else False
        pf, dc = row.get(PREFILL_CELL), row.get(DECODE_CELL)
        both_clean = row.get(PREFILL_CELL + "_clean") and row.get(DECODE_CELL + "_clean")
        meets = (pf is not None and dc is not None and both_clean
                 and pf >= BAR_PREFILL and dc >= BAR_DECODE)
        row["meets_bar"] = meets
        row["both_clean"] = both_clean
        table.append(row)
        print(f"{profile:<14}"
              f"{(f'{pf:.2f}' if pf is not None else '-'):>13}"
              f"{(f'{dc:.2f}' if dc is not None else '-'):>13}"
              f"{(row[PREFILL_CELL + '_worst'] if pf is not None else '-'):>10}"
              f"{(row[DECODE_CELL + '_worst'] if dc is not None else '-'):>10}"
              f"{('PASS' if meets else 'FAIL'):>13}")
    print("-" * 88)
    print("bar = prefill >= 250 AND decode >= 12, and BOTH arms stayed NOMINAL for their whole run.")
    print("A 'FAIL' can mean too slow OR measured on a arm that heated up -- different facts. The "
          "'worst' column is the one that decides: launching NOMINAL is not enough.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print(f"\nwrote {args.json}")
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        lines = ["| profile | prefill t/s | decode t/s | launch | bar 250/12 |",
                 "|---|---|---|---|---|"]
        for row in table:
            pf, dc = row.get(PREFILL_CELL), row.get(DECODE_CELL)
            lines.append(f"| {row['profile']} "
                         f"| {f'{pf:.2f}' if pf is not None else '-'} "
                         f"| {f'{dc:.2f}' if dc is not None else '-'} "
                         f"| {row.get(PREFILL_CELL + '_launch', '-')} "
                         f"| {'PASS' if row['meets_bar'] else 'FAIL'} |")
        Path(args.md).write_text("\n".join(lines) + "\n")
        print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
