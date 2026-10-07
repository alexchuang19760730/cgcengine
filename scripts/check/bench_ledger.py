#!/usr/bin/env python3
"""bench_ledger.py -- read the acceptance ledger instead of scanning 97 files.

WHY THIS TOOL EXISTS
---------------------
On 2026-10-07 the question "what can this build actually run under the acceptance caliber?"
took a ten-minute scan of Backup/spec_tree/, because:

  * `Backup/commit_bench/` was EMPTY while 97 readings sat scattered across Backup/spec_tree/,
  * only 3 of those 97 were quotable (34 `both`, 23 `swap`, 14 `thermal`, 23 field-absent),
  * and the pass/fail decision lived in the gate, not next to the number.

`commit_bench.py` now appends every run to Backup/commit_bench/ledger.jsonl. This reads it back.
The point is not the pretty output -- it is that the quotability judgement was made ONCE, by the
tool that had the attribution data, instead of re-litigated per reader. That is the whole
"缺口從來不是沒有檔案，是沉默" lesson applied to a number rather than to a gate.

WHAT IT WILL NOT DO
-------------------
It does not average, does not pick a best, and does not treat a `quotable: false` row as an error
in the run. A swap-tainted window is a real measurement of a hot machine; it just cannot be quoted
as the machine's capability. Both readings are kept, and the two columns are reported separately.

USAGE
    python3 scripts/check/bench_ledger.py                 # quotable range + last row
    python3 scripts/check/bench_ledger.py --all           # every row, quotable or not
    python3 scripts/check/bench_ledger.py --json           # machine-readable
    python3 scripts/check/bench_ledger.py --selftest       # grading rules, on fixtures
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LEDGER = os.path.join(ROOT, "Backup", "commit_bench", "ledger.jsonl")


def load(path=LEDGER):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                # A corrupt line is reported, not skipped: silently dropping a reading is the exact
                # failure this repo keeps paying for ("no record" vs "never happened" are different).
                print(f"[bench_ledger] line {ln} is not valid JSON ({e}); excluded", file=sys.stderr)
    return rows


def _f(v):
    return float(v) if isinstance(v, (int, float)) else None


def summarize(rows):
    q = [r for r in rows if r.get("quotable") is True]
    nq = [r for r in rows if r.get("quotable") is not True]
    out = {"total": len(rows), "quotable": len(q), "not_quotable": len(nq)}
    for key, idx in (("prefill", "prefill_ts"), ("decode", "decode_ts")):
        vals = [_f(r.get(idx)) for r in q]
        vals = [v for v in vals if v is not None]
        out[key + "_quotable_range"] = [min(vals), max(vals)] if vals else None
    last = rows[-1] if rows else None
    out["last"] = {
        "ts": last.get("ts"), "prefill_ts": last.get("prefill_ts"),
        "decode_ts": last.get("decode_ts"), "mtp": last.get("mtp"),
        "quotable": last.get("quotable"), "why": last.get("quotable_why"),
        "gate_pass": last.get("gate_pass"),
    } if last else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="print every row, including unquotable ones")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    rows = load()
    if not rows:
        print("no ledger rows. Run: python3 scripts/check/commit_bench.py", file=sys.stderr)
        print("(Backup/commit_bench/ledger.jsonl does not exist yet -- an empty ledger is not a "
              "measurement of zero, it is the absence of one)", file=sys.stderr)
        return 2

    s = summarize(rows)
    if args.json:
        print(json.dumps({"summary": s, "rows": rows}, ensure_ascii=False, indent=2))
        return 0

    if args.all:
        print(f"{'ts':<20} {'pp512':>8} {'tg128':>8} {'mtp':<9} {'quotable':<9} gate  why")
        for r in rows:
            print(f"{str(r.get('ts')):<20} {_f(r.get('prefill_ts')) or 0:>8.2f} "
                  f"{_f(r.get('decode_ts')) or 0:>8.2f} {str(r.get('mtp')):<9} "
                  f"{'yes' if r.get('quotable') else 'NO':<9} "
                  f"{'pass' if r.get('gate_pass') else 'fail'}  {str(r.get('quotable_why'))[:44]}")
        print()

    print(f"rows: {s['total']}   quotable: {s['quotable']}   not quotable: {s['not_quotable']}")
    pp, dg = s["prefill_quotable_range"], s["decode_quotable_range"]
    if pp:
        print(f"  prefill (quotable): {pp[0]:.2f} – {pp[1]:.2f} t/s")
    else:
        print("  prefill: no quotable reading yet")
    if dg:
        print(f"  decode  (quotable): {dg[0]:.2f} – {dg[1]:.2f} t/s")
    else:
        print("  decode:  no quotable reading yet")
    if s["last"]:
        L = s["last"]
        print(f"  latest ({L['ts']}): pp={_f(L['prefill_ts']):.2f} tg={_f(L['decode_ts']):.2f} "
              f"{L['mtp']}  quotable={'yes' if L['quotable'] else 'NO'} ({L['why']})")
    if s["not_quotable"]:
        print(f"\n  {s['not_quotable']} row(s) are real measurements of a hot/swap-tainted window "
              f"and are NOT quotable as this machine's capability.")
    return 0


def _selftest():
    """Fixtures, not repo rows: the grading rule is what needs proving, and the repo's own history
    is that a gate which never goes red is not a gate. Each case is a plausible ledger line."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "commit_bench", os.path.join(HERE, "commit_bench.py"))
    cb = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(cb)
    except Exception as e:
        print(f"  [FAIL] cannot import commit_bench: {e}")
        return 1
    cases = [
        ("attribution absent (23 of 97 real rows)", None, False),
        ("attribution is not a dict", "none", False),
        ("verdict none, NOMINAL + negative swap",
         {"verdict": "none", "why": "swap_growth=-16.0 MiB, thermal=NOMINAL"}, True),
        ("verdict both (34 of 97 real rows)", {"verdict": "both", "why": "thermal HEAVY + swap 711"}, False),
        ("verdict swap", {"verdict": "swap"}, False),
        ("verdict thermal", {"verdict": "thermal"}, False),
        # [CGC 2026-10-07] New verdict from memory_pressure.attribution: budget-gate found the
        # static demand exceeds RAM, so the arm could only complete through compression + swap
        # even though every delta read zero. It must NOT be quotable -- that exact shape is what
        # made the 12:05 arm look clean (swap_growth=0.0, thermal=NOMINAL) while 260-300 MiB of
        # swap was still held by the 11:15 run. `quotable` recognises only "none", so this case
        # asserts the rejection is automatic rather than depending on a special-case branch.
        ("verdict oversubscribed (every delta zero, but budget-gate said no)",
         {"verdict": "oversubscribed", "oversubscribed": True,
          "why": "static demand exceeds physical RAM (budget-gate OVERSUBSCRIBED)"}, False),
        ("verdict empty string", {"verdict": ""}, False),
    ]
    ok = 0
    for name, attr, want in cases:
        got, _why = cb.quotable(attr)
        flag = "ok" if got is want else "FAIL"
        if got is want:
            ok += 1
        print(f"  [{flag}] {name}\n         want quotable={want} got={got}")
    # The other half: summarize must ignore unquotable rows when computing the range. If it did not,
    # a swap-tainted 160 pp would become the quoted capability -- which is the mistake being fixed.
    rows = [
        {"quotable": True, "prefill_ts": 159.91, "decode_ts": 11.36, "ts": "t1", "mtp": "mtp=off",
         "quotable_why": "NOMINAL", "gate_pass": True},
        {"quotable": False, "prefill_ts": 999.0, "decode_ts": 44.0, "ts": "t2", "mtp": "mtp=on",
         "quotable_why": "both", "gate_pass": True},
        {"quotable": True, "prefill_ts": 124.01, "decode_ts": 11.69, "ts": "t3", "mtp": "mtp=off",
         "quotable_why": "MODERATE", "gate_pass": True},
    ]
    s = summarize(rows)
    good = (s["prefill_quotable_range"] == [124.01, 159.91]
            and s["decode_quotable_range"] == [11.36, 11.69]
            and s["not_quotable"] == 1 and s["total"] == 3)
    print(f"  [{'ok' if good else 'FAIL'}] summarize excludes unquotable rows from the range")
    print(f"         pp={s['prefill_quotable_range']} tg={s['decode_quotable_range']} "
          f"(a 999/44 row must not appear)")
    if good:
        ok += 1
    total = len(cases) + 1
    print("-" * 74)
    print(f"selftest: {ok}/{total}" + ("" if ok == total else "   <-- FAIL"))
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
