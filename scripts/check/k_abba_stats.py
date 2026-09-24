#!/usr/bin/env python3
"""Statistics for a rotated (ABBA) arm comparison: the effect, and the reproducibility that
bounds it.

Why this exists as a tool and not as a paragraph of arithmetic: the two quantities that decide
whether an A/B on this box says anything are (a) the paired effect and (b) the same-arm spread,
and both are easy to get wrong by hand. The specific wrongness this pins: in a rotated run the
arms swap positions between rounds, so a "first minus second" subtraction flips the sign in half
the rounds and turns a 4/5 effect into a 2/5 one. That happened here (measured: -0.49 printed for
a round whose A-B is +0.49), which is the same class of defect as reading an absent value as a
zero. The subtraction follows the arm.

Second convention it enforces: the weakest paired round is reported, not just the mean. On this
box one round can land in a state the others did not, and a mean over five rounds hides that.

    k_abba_stats.py --driver-log /tmp/ka/driver.log --arms-dir /tmp/ka
    k_abba_stats.py --selftest

The driver log supplies the rotation (which arm sat in which position, plus the per-arm
treatment such as the thermal-gate wait); the arm JSONs supply the readings and provenance.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics as st
from pathlib import Path


def paired_diffs(pairs: list[tuple[float, float]]) -> dict:
    """(A, B) pairs -> the effect A - B, with the weak round kept visible.

    `weakest` is the paired difference closest to zero with its sign: it is the round that a
    reviewer has to explain before the mean may be quoted as an effect.
    """
    d = [a - b for a, b in pairs]
    sd = st.stdev(d) if len(d) > 1 else float("nan")
    se = sd / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return {
        "n": len(d),
        "mean": st.mean(d) if d else float("nan"),
        "median": st.median(d) if d else float("nan"),
        "sd": sd,
        "se": se,
        "t": (st.mean(d) / se) if d and se else float("nan"),
        "positives": sum(1 for x in d if x > 0),
        "weakest": min(d, key=abs) if d else float("nan"),
        "diff": d,
    }


def spread(vals: list[float]) -> dict:
    sd = st.stdev(vals) if len(vals) > 1 else float("nan")
    return {"n": len(vals), "mean": st.mean(vals), "sd": sd,
            "cv_pct": 100 * sd / st.mean(vals) if st.mean(vals) else float("nan"),
            "se_of_mean": sd / math.sqrt(len(vals)) if len(vals) > 1 else float("nan"),
            "min": min(vals), "max": max(vals), "ratio": max(vals) / min(vals) if min(vals) else float("nan")}


def pair_by_arm(rows: dict, ks: list[int]) -> list[tuple[float, float]]:
    """Rows of a rotated run -> (A, B) pairs, resolving each round's position to its arm.

    This is the unit that can silently invert an effect, so it is the one the selftest drives: a
    run that rotates its arms (ABBA) puts A second in half the rounds, and taking "first minus
    second" there returns the negated difference for those rounds -- a 4/5 effect becomes 2/5.
    """
    pairs = []
    for rnd in sorted({k.split(".")[0] for k in rows}):
        a, b = rows.get(f"{rnd}.a"), rows.get(f"{rnd}.b")
        if not (a and b and a.get("decode") and b.get("decode")):
            continue
        if a["k"] == b["k"]:            # a null cell (A vs A) is not an effect pair
            continue
        hi = next(r for r in (a, b) if r["k"] == ks[-1])
        lo = next(r for r in (a, b) if r["k"] == ks[0])
        pairs.append((lo["decode"], hi["decode"]))
    return pairs


def parse_driver_log(path: Path) -> dict:
    """Per-position readings and treatments from a rotating driver's log.

    Two shapes are accepted, because two drivers exist: the ABBA one writes
        R2.a k=3 waited=141s level=0 ... => accept ..., decode 10.72 t/s
    and the back-to-back one writes a single gated arm whose wait is in the tool's own line
        thermal gate: Nominal after 60.1s  /  => accept ..., decode 12.25 t/s
    The second shape is recorded as position 'z' so it still participates as a pair.
    """
    txt = path.read_text()
    out: dict[str, dict] = {}
    for m in re.finditer(r"R(\d)\.([ab]) k=(\d) waited=(\S+)s level=(\S+)", txt):
        key = f"R{m.group(1)}.{m.group(2)}"
        dec = re.search(rf"R{m.group(1)}\.{m.group(2)} k={m.group(3)}.*?"
                        r"=> accept [\d.]+% \([\d/]+\)(?:,\s*[^,]*)?"
                        r"(?:,\s*[^,]*)?,\s*decode ([\d.]+) t/s", txt, re.S)
        if not dec:
            continue
        out[key] = {"k": int(m.group(3)),
                    "waited_s": int(m.group(4)) if m.group(4).isdigit() else None,
                    "level_at_launch": m.group(5), "decode": float(dec.group(1))}
    return out


BACKTOBACK_FIXTURE = """
[02:09:15] --- gated arm k=2
[02:09:23]     thermal gate: Nominal after 0.0s
[02:10:10]     => accept 73.16% (169/231), decode 12.20 t/s
[02:10:20] --- gated arm k=3
[02:11:28]     thermal gate: Nominal after 60.1s
[02:12:15]     => accept 58.25% (180/309), decode 12.25 t/s
"""


def backtoback_arms(text: str) -> list[dict]:
    """Arms from a driver that runs them head-to-tail without rotating (the gate-verification
    shape): a `--- gated arm k=N` marker followed by that arm's reading.

    Kept as an ordered list rather than as positions, because in this shape the position carries
    no rotation information -- the arms are simply sequential, and the caller pairs them by order.
    """
    out = []
    for m in re.finditer(r"--- \S* ?arm k=(\d)(.*?)(?=--- \S* ?arm k=|\Z)", text, re.S):
        body, k = m.group(2), int(m.group(1))
        dec = re.search(r"decode (\d+(?:\.\d+)?) t/s", body)
        wait = re.search(r"thermal gate: Nominal after ([\d.]+)s", body)
        late = re.search(r"thermal gate: level (\d+) after", body)
        if dec:
            out.append({"k": k, "decode": float(dec.group(1)),
                        "waited_s": float(wait.group(1)) if wait else None,
                        "level_at_launch": 0 if wait else (int(late.group(1)) if late else None)})
    return out


def parse_backtoback_log(path: Path) -> list[dict]:
    try:
        return backtoback_arms(path.read_text())
    except OSError:
        return []


def sequential_pairs(arms: list[dict], ks: list[int]) -> list[tuple[float, float]]:
    """Ordered arms -> pairs, taking each adjacent lo/hi couple once."""
    pairs, pend = [], None
    for a in arms:
        if a["k"] == ks[0]:
            pend = a
        elif a["k"] == ks[-1] and pend is not None:
            pairs.append((pend["decode"], a["decode"]))
            pend = None
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--driver-log", action="append", default=None,
                    help="driver log (repeatable): a rotated run and/or a back-to-back run")
    ap.add_argument("--arms-dir", default="", help="directory of arm JSONs to fold in")
    ap.add_argument("--pair", default="", help="arm names to pair, e.g. the two k values' labels")
    ap.add_argument("--selftest", action="store_true",
                    help="drive the parser and the pairing rule, incl. the rotated-sign case")
    args = ap.parse_args()

    if args.selftest:
        bad = 0
        # A rotated run whose two arms do not move: every round must read +1 regardless of which
        # position the faster arm sat in. A position-based subtraction gives [+1, -1] here.
        rotated = {
            "R1.a": {"k": 2, "decode": 3.0}, "R1.b": {"k": 3, "decode": 2.0},
            "R2.a": {"k": 3, "decode": 2.0}, "R2.b": {"k": 2, "decode": 3.0},
        }
        got = pair_by_arm(rotated, [2, 3])
        if got != [(3.0, 2.0), (3.0, 2.0)]:
            print(f"FAIL rotation: {got!r}, want (A, B) both +1")
            bad += 1
        p = paired_diffs(got)
        if abs(p["mean"] - 1.0) > 1e-9 or p["positives"] != 2:
            print(f"FAIL rotation stats: {p['mean']!r}, positives {p['positives']}")
            bad += 1
        # Same-k rounds are a null cell, not an effect: they must be dropped, not differenced.
        null = {"R1.a": {"k": 2, "decode": 3.0}, "R1.b": {"k": 2, "decode": 2.5}}
        if pair_by_arm(null, [2, 3]) != []:
            print("FAIL null cell: a same-arm round was differenced")
            bad += 1
        # The real round, whose A sat second: +0.49, not -0.49.
        p = paired_diffs(pair_by_arm(
            {"R1.a": {"k": 3, "decode": 10.72}, "R1.b": {"k": 2, "decode": 11.21}}, [2, 3]))
        if abs(p["mean"] - 0.49) > 1e-9:
            print(f"FAIL real round sign: {p['mean']!r}")
            bad += 1
        # A zero round survives as the weakest rather than being dropped.
        p = paired_diffs([(12.20, 12.25), (12.83, 9.09)])
        if p["positives"] != 1 or abs(p["weakest"] + 0.05) > 1e-9:
            print(f"FAIL weak round: positives={p['positives']} weakest={p['weakest']!r}")
            bad += 1
        # One pair is not an effect: no sd may be reported for it.
        if not math.isnan(paired_diffs([(12.2, 12.25)])["sd"]):
            print("FAIL single pair: sd should be nan")
            bad += 1
        # The back-to-back parser: two arms, in order, each with its own gate wait. This is a
        # fixture rather than a run, because the parse is what a fixture can hold still (its
        # first version called re.search without a subject and died on the first real log).
        seq = backtoback_arms(BACKTOBACK_FIXTURE)
        if [a["k"] for a in seq] != [2, 3] or [a["decode"] for a in seq] != [12.20, 12.25]:
            print(f"FAIL back-to-back parse: {seq!r}")
            bad += 1
        elif [a["waited_s"] for a in seq] != [0.0, 60.1]:
            print(f"FAIL back-to-back waits: {[a['waited_s'] for a in seq]!r}")
            bad += 1
        elif sequential_pairs(seq, [2, 3]) != [(12.20, 12.25)]:
            print(f"FAIL sequential pairing: {sequential_pairs(seq, [2, 3])!r}")
            bad += 1
        print(f"selftest: {9 - bad}/9 passed")
        return 1 if bad else 0

    logs = [Path(x) for x in (args.driver_log or ["/tmp/ka/driver.log"])]
    rows: dict = {}
    sequential: list[dict] = []
    for lg in logs:
        rows.update(parse_driver_log(lg))
        sequential += parse_backtoback_log(lg)
    for p in sorted(glob.glob(str(Path(args.arms_dir or ".") / "*.json"))):
        stem = Path(p).stem
        m = re.match(r"r(\d)_([ab])_k(\d)$", stem)
        if not m:
            continue
        arm = json.loads(Path(p).read_text())[0]
        key = f"R{m.group(1)}.{m.group(2)}"
        rows.setdefault(key, {"k": int(m.group(3)), "waited_s": None, "level_at_launch": None})
        rows[key]["decode"] = arm.get("decode_tps_mean")
        rows[key]["accept"] = arm.get("accept")

    print(f"{'pos':<7} {'k':>2} {'waited':>7} {'lvl':>4} {'decode':>8} {'accept':>8}")
    for key, r in sorted(rows.items()):
        acc = r.get("accept")
        print(f"{key:<7} {r['k']:>2} {str(r.get('waited_s')):>7} {str(r.get('level_at_launch')):>4} "
              f"{r['decode']:>8.2f} {f'{acc*100:7.2f}%' if acc else '     n/a'}")

    by_k: dict[int, list[float]] = {}
    for r in rows.values():
        if r.get("decode"):
            by_k.setdefault(r["k"], []).append(r["decode"])
    print()
    for k in sorted(by_k):
        s = spread(by_k[k])
        print(f"k={k}: n={s['n']} mean={s['mean']:.2f} sd={s['sd']:.2f} cv={s['cv_pct']:.1f}% "
              f"min={s['min']:.2f} max={s['max']:.2f} ({s['ratio']:.2f}x)  "
              f"SE(mean)={s['se_of_mean']:.2f} ({100*s['se_of_mean']/s['mean']:.1f}%)")

    ks = sorted(by_k)
    if len(ks) != 2:
        if sequential:
            ks = sorted({a["k"] for a in sequential})
        else:
            return 0
    for i, a in enumerate(sequential, 1):
        print(f"{'seq%d' % i:<7} {a['k']:>2} {str(a.get('waited_s')):>7} "
              f"{str(a.get('level_at_launch')):>4} {a['decode']:>8.2f}")
    pairs = pair_by_arm(rows, ks) + sequential_pairs(sequential, ks)
    p = paired_diffs(pairs)
    print(f"\npaired k={ks[0]} - k={ks[1]}: n={p['n']} mean={p['mean']:+.2f} "
          f"median={p['median']:+.2f} sd={p['sd']:.2f} t={p['t']:.2f} (df={p['n']-1}) "
          f"positives={p['positives']}/{p['n']} weakest={p['weakest']:+.2f}")
    print(f"  per round: {[round(x, 2) for x in p['diff']]}")
    weak = [r for r in pairs if abs(r[0] - r[1]) < 1.0]
    if weak:
        print(f"  below 1 t/s: {[(round(a,2), round(b,2)) for a, b in weak]} -- the mean may not be "
              f"quoted as an effect while one round sits here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
