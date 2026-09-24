#!/usr/bin/env python3
"""What would 25 t/s actually require? A two-axis table generated from a real decode step.

WHY GENERATED. The repo has been burned by hand-typed tables (a cross-tab once shipped with a
duplicated key and a missing one, and a "hard floor" was quoted from a field that counts overlapping
buffers twice). So the step axis here is READ off the log, and the arithmetic is printed with the
inputs next to it.

THE TWO AXES, and why they are independent:
  overlap  -- how much of the host-serial `gap` is removed. Bounded below by `union_sum`: the GPU
              spans are chained (attention(il+1) consumes MoE(il)), so no amount of host overlap can
              make a step shorter than the sum of the spans. `gpu_busy_sum` is NOT that bound -- its
              own definition is "overlap counted twice" (ggml-backend.cpp:2049).
  mean_len -- tokens emitted per verify step. Bounded above by k+1 = 4 for k=3 drafts.

step <= 40 * mean_len is the whole of 25 t/s; everything else is bookkeeping about which lever moves
which axis, and which of the two binds first.

TWO MODES, because the two regimes have different step decompositions and must not be mixed:
  (default)  --log PATH   a server-regime log: the step axis is "how much host gap is removed".
  --from-pair DIR         a k=1/k=3 pair: the step axis is the MEASURED round budget
                          (`docs/VERIFY_MARGINAL_2026-09-22.md`), which is the axis that matters whenever
                          MTP is on, because there the marginal token is 70-102% round remainder.

Usage: target25_grid.py [--log PATH] [--from-pair DIR] [--selftest]
"""
from __future__ import annotations

import argparse
import collections
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_marginal as vm  # noqa: E402  (the pair's instruments; one parser, not a second one)
DEFAULT_LOG = "Backup/cgc_logs/llama_server_20260920_023021.log"
K3_CAP = 4.0                     # k=3 drafts => at most 4 tokens per verify step
TARGET = 25.0
# mean_len readings in circulation, each with the arm it came from. They are NOT interchangeable
# (the G6 arm's own doc records 2.40 vs 3.375 both being quoted), so the table keeps them apart.
MEAN_LENS = (2.71, 2.97, 3.375, 3.73, K3_CAP)


def decode_rows(path: Path, ntok: str = "4", segs: str = "41") -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if "CGC-DECPROF:" not in line or "layers=" not in line:
            continue
        d = dict(re.findall(r"([a-z_]+)=([\d.]+)", line))
        if d.get("ntok") == ntok and d.get("segs") == segs:
            rows.append({k: float(v) for k, v in d.items()})
    return rows


def axes(rows: list[dict]) -> dict:
    med = lambda k: st.median([r[k] for r in rows])
    return {"n": len(rows), "total": med("total"), "wait": med("wait"), "cb": med("cb"),
            "submit": med("submit"), "union": med("union_sum"), "gap": med("gap_sum"),
            "busy": med("gpu_sum")}


def report(a: dict) -> int:
    if a["n"] == 0:
        print("INVALID: no ntok/segs rows in the log -- nothing measured, nothing to divide")
        return 1
    floor = a["union"]
    steps = [("today (read)", a["total"]),
             ("cb hidden", a["total"] - a["cb"]),
             ("cb+submit hidden", a["total"] - a["cb"] - a["submit"]),
             ("union floor (hard limit)", floor)]

    print(f"log rows n={a['n']}  ntok=4 segs=41 (medians)")
    print(f"  total {a['total']:.2f} = wait {a['wait']:.2f} + cb {a['cb']:.2f} + submit {a['submit']:.2f}")
    print(f"  union_sum {a['union']:.2f}  (hard floor)   gap_sum {a['gap']:.2f}   "
          f"gpu_busy_sum {a['busy']:.2f} [NOT a floor: overlap counted twice]")
    print(f"  cross-checks: union+gap {a['union'] + a['gap']:.2f} vs total "
          f"{a['total']:.2f} ({100 * (a['union'] + a['gap'] - a['total']) / a['total']:+.1f}%); "
          f"wait+cb+submit {a['wait'] + a['cb'] + a['submit']:.2f} "
          f"({100 * (a['wait'] + a['cb'] + a['submit'] - a['total']) / a['total']:+.1f}%)")
    print()

    print("t/s = mean_len / step   (>= requires step <= 40 * mean_len)")
    head = "  step ms |" + "".join(f" ml {m:<5.3f} |" for m in MEAN_LENS)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name, s in steps:
        cells = []
        for m in MEAN_LENS:
            tps = m / (s / 1000.0)
            cells.append(f" {tps:6.2f}{'PASS' if tps >= TARGET else '    '} |")
        print(f"  {s:7.2f} |" + "".join(cells) + f"  <- {name}")
    print()

    print("required ml at each step, and the overlap fraction it implies")
    span = a["total"] - floor
    for name, s in steps:
        need = TARGET * s / 1000.0
        ov = 100.0 * (a["total"] - s) / span
        verdict = "above the k=3 cap (4.0)" if need > K3_CAP else "within k=3"
        print(f"  {s:7.2f} ms  needs ml {need:5.3f}  ({verdict})   overlap {ov:5.1f}% of "
              f"the {span:.2f} ms gap")
    print()

    lo = TARGET * floor / 1000.0                 # ml needed at the floor
    hi_step = K3_CAP * 1000.0 / TARGET           # step allowed at the k=3 ceiling
    print(f"feasible region for 25 t/s: step in [{floor:.2f}, {hi_step:.1f}] ms and "
          f"ml in [{lo:.3f}, {K3_CAP}]")
    print(f"  at ml = {K3_CAP} (accept at the k=3 ceiling) the overlap must reach "
          f"{100 * (a['total'] - hi_step) / span:.0f}% of the gap ({a['total']:.0f} -> {hi_step:.0f} ms)")
    print(f"  at the union floor the ml must reach {lo:.3f} "
          f"({100 * lo / K3_CAP:.0f}% of the cap)")
    return 0


def expected_len(alpha: float, k: int) -> float:
    """Emitted tokens per round for a geometric per-draft accept: 1 + a + ... + a^k."""
    return sum(alpha ** i for i in range(k + 1))


def required_alpha(want: float, k: int) -> float | None:
    """Smallest per-draft accept whose geometric E reaches `want`. None if even alpha=1 falls short --
    that None is the whole point of the table: it is the cells no accept rate can reach."""
    if want > expected_len(1.0, k) + 1e-9:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if expected_len(mid, k) < want:
            lo = mid
        else:
            hi = mid
    return hi


def round_ladder(run1: dict, run3: dict, k: int, dt: int = 2) -> tuple:
    """The T=k+1 round, with one round term removed per row, from the measured budget.

    Returns (rows, reason); rows are (label, step_ms, lever). The last row is the eval-sync chain, i.e.
    the GPU floor seen from the CPU -- a limit, not a target.
    """
    lt, why = vm.layer_table(run1, run3, dt, vm.PER_MISS_MS)
    if lt is None:
        return None, why
    wait = sum(r["wait_3"] for r in lt)
    host = sum(r["cb_3"] + r["submit_3"] for r in lt)
    draft, begin = run3.get("draft_ms") or 0.0, run3.get("begin_ms") or 0.0
    rnd = run3["round_ms"]
    remainder = rnd - (wait + host + draft + begin)
    return [("today (measured)", rnd, "--"),
            ("- round remainder", rnd - remainder,
             "remainder: post-layer per-round work (70-102% of the marginal)"),
            ("- remainder - draft head", rnd - remainder - draft,
             "+ a cheaper drafter (no draft forward)"),
            ("- remainder - draft - per-layer host", rnd - remainder - draft - host,
             "+ dispatcher: per-layer gather+dispatch (HARD FLOOR, not a target)")], None


def report_pair(logdir: Path, k: int) -> int:
    jsonl = logdir / "arms.jsonl"
    if not jsonl.exists():
        print(f"INVALID: no {jsonl}")
        return 1
    arms = vm.arms_from_jsonl(jsonl)
    if not arms.get("k1") or not arms.get(f"k{k}"):
        print(f"INVALID: {jsonl} needs both a k1 and a k{k} arm (present: {sorted(arms)})")
        return 1
    reps = sorted(set(arms["k1"]) & set(arms[f"k{k}"]))
    if not reps:
        print("INVALID: the two arms share no rep")
        return 1
    print(f"25 t/s necessity from the MEASURED round budget  ({logdir}, k={k}, {len(reps)} rep(s))")
    print("  step <= 40 * mean_len;  mean_len <= k+1 = %d by construction\n" % (k + 1))
    for rep in reps:
        rows, why = round_ladder(arms["k1"][rep], arms[f"k{k}"][rep], k)
        if rows is None:
            print(f"  rep {rep + 1}: refused -- {why}")
            continue
        real = arms[f"k{k}"][rep].get("mean_len")
        print(f"  rep {rep + 1}: T={k + 1} arm emits {real:.2f} tok/round measured" if real else
              f"  rep {rep + 1}:")
        print(f"    {'step ms':>9} {'needs mean_len':>15} {'needs alpha':>12} {'lever to get here':<64}")
        for label, step, lever in rows:
            want = TARGET * step / 1000.0
            a = required_alpha(want, k)
            verdict = f"{a:.3f}" if a is not None else "impossible"
            print(f"    {step:>9.1f} {want:>15.3f} {verdict:>12}   {label:<38} {lever}")
        print()
    print("  Read the alpha column against the SAME-ARM measurement, not against a hoped-for accept:")
    print("  every row above the floor needs mean_len > k+1, i.e. no accept rate reaches it.")
    return 0


def selftest() -> int:
    """Arithmetic, and the two ways this table could lie: a permissive zero and a wrong floor."""
    bad = 0
    def check(name, cond):
        nonlocal bad
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        bad += 0 if cond else 1

    check("25 t/s is exactly ml/step", abs(3.375 / 0.135 - 25.0) < 1e-9)
    check("ml=4.0 clears 25 only at step<=160", 4.0 / 0.160 >= 25.0 and 4.0 / 0.161 < 25.0)
    check("today's step does not clear 25 even at the cap", 4.0 / 0.24798 < 25.0)
    check("the floor is union, not busy_sum", 149.24 < 181.56)
    check("the feasible step band is [149.24, 160.0]", abs(4.0 * 1000.0 / 25.0 - 160.0) < 1e-9)
    check("a missing log is INVALID, not 0 t/s", report({"n": 0}) == 1)

    # the round-budget mode
    check("alpha=1 emits exactly k+1", abs(expected_len(1.0, 3) - 4.0) < 1e-12)
    check("alpha=0 emits exactly 1", abs(expected_len(0.0, 3) - 1.0) < 1e-12)
    check("alpha=0.5 emits 1.875", abs(expected_len(0.5, 3) - 1.875) < 1e-12)
    check("required alpha is monotone in want",
          required_alpha(2.0, 3) < required_alpha(3.0, 3) < required_alpha(3.9, 3))
    check("want above the cap has no alpha", required_alpha(4.001, 3) is None)
    check("want == cap needs alpha 1", abs((required_alpha(4.0, 3) or 0) - 1.0) < 1e-6)
    check("a 160 ms step at the k=3 cap is exactly 25 t/s",
          abs(4.0 * 1000.0 / 160.0 - TARGET) < 1e-9)
    # the ladder must be strictly decreasing, or it is not a ladder
    run3 = {"round_ms": 352.4, "draft_ms": 29.5, "begin_ms": 0.0}
    fake = ([{"wait_3": 100.0, "cb_3": 20.0, "submit_3": 10.0},
             {"wait_3": 58.5, "cb_3": 22.0, "submit_3": 4.0}], None)
    real_layer_table = vm.layer_table
    vm.layer_table = lambda *a, **kw: fake  # the parser is tested where it lives, not here
    try:
        rows, why = round_ladder({}, run3, 3)
    finally:
        vm.layer_table = real_layer_table
    check("ladder built", rows is not None and why is None)
    steps = [s for _, s, _ in (rows or [])]
    check("ladder is strictly decreasing",
          all(steps[i] > steps[i + 1] for i in range(len(steps) - 1)))
    check("ladder's last row is the eval-sync chain", abs(steps[-1] - 158.5) < 1e-9)
    check("ladder's first row is the measured round", abs(steps[0] - 352.4) < 1e-9)
    check("every rung above the floor needs more than the cap",
          all(TARGET * s / 1000.0 > 4.0 for s in steps[:-1]))
    check("a mismatched pair is INVALID, not an empty table",
          report_pair(Path("/nonexistent"), 3) == 1)
    print(f"target25 selftest: {14 - bad}/14 passed")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--from-pair", default="", help="k-sweep logdir with arms.jsonl (k1 + k3 arms)")
    ap.add_argument("--k", type=int, default=3, help="draft width of the wide arm")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.from_pair:
        return report_pair(Path(args.from_pair), args.k)
    path = Path(args.log)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        print(f"no such log: {path}")
        return 1
    return report(axes(decode_rows(path)))


if __name__ == "__main__":
    sys.exit(main())
