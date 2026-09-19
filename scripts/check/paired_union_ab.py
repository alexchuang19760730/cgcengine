#!/usr/bin/env python3
"""Paired AB/BA on `union_sum` (the GPU-side span), from server logs that already exist.

WHY THIS EXISTS. G4's target is a -12.8% change in union_sum. `decode_sweep.py` does not
report union, and the reading that was actually used to judge the down-combine lever was
t/s -- an effect of 1.76 t/s on an arm of ~9, against a documented single-arm noise floor of
+/-1.9 t/s. union is the discriminating instrument, so it needs a tool that reads it with a
design that can be trusted. This is that tool.

DESIGN RULES IT ENFORCES, each paid for already:

  * PAIR BY STEP NUMBER over a step set shared by ALL the logs being compared. Slicing each
    arm's own common-step list and zipping pairs POSITIONS, which is only valid if the two
    arms happen to share their step sequence -- they do not (the four Ornith logs share 45 of
    55 steps). Worse: the shared subset can be BIASED (fwd's own median 113.3 ms vs 89.4 ms
    over the shared 45), so the absolute base is reported from the shared set, never pooled.
  * SAME ntok on both sides of every pair, and only full-graph steps (`layers >= 40`); a local
    profile (`segs=2 layers=1`) passes an ntok filter and is not a step.
  * NEVER quote a ratio of medians as the effect. Report the median of the PAIRED differences,
    its standard error (1.253*sigma/sqrt(n) for the median), and the residual drift of the
    AB/BA. The ratio of medians is what produced a reported "+25%" out of a flag whose two
    halves disagreed in SIGN.
  * Say whether the design can resolve the target at all. If the effect is under 3*SE, the
    honest output is NOT MEASURABLE, not "no effect".
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import statistics as st

H = re.compile(
    r"CGC-DECPROF: step=(\d+) segs=(\d+) layers=(\d+) total=([0-9.]+) ms \| "
    r"wait=([0-9.]+) \(([0-9.]+)%\) cb=([0-9.]+) \(([0-9.]+)%\) submit=([0-9.]+) \(([0-9.]+)%\) ntok=(\d+)"
    r"(?: \| layer gpu_sum=([0-9.]+) union_sum=([0-9.]+) gap_sum=([0-9.]+) ms)?")

TARGET_FRAC = 0.128     # G4: union <= 38*mean_len is -12.8% at the measured mean_len 2.40


def decprof_steps(path):
    """Count CGC-DECPROF full-graph headers, with or without the GPU-side tail.

    Needed to tell two failures apart that look identical downstream ("0 steps with union"):
    CGC_DECODE_PROFILE off (no DECPROF at all) and CGC_GPU_TIMING off (DECPROF present, but
    no `| layer gpu_sum=.. union_sum=.. gap_sum=..`). MEASURED 2026-09-20: of the 13 server
    logs written in the previous hour, ONE carried union and twelve carried DECPROF without
    it -- so a run that forgets the second knob silently yields nothing to pair, and would
    otherwise be read as "no effect".
    """
    n = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            if "CGC-DECPROF: step=" not in line:
                continue
            m = H.search(line.strip())
            if m and int(m.group(3)) >= 40:
                n += 1
    return n


def series(path):
    out = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            if "union_sum=" not in line or "CGC-DECPROF: step=" not in line:
                continue
            m = H.search(line.strip())
            if not m or int(m.group(3)) < 40:
                continue
            out[int(m.group(1))] = dict(ntok=int(m.group(11)), total=float(m.group(4)),
                                        union=float(m.group(12)), gap=float(m.group(14)))
    return out


def se_median(d):
    return 1.253 * st.pstdev(d) / (len(d) ** 0.5)


class Refuse(Exception):
    """Raised instead of returning a number when the inputs are not comparable."""


def align_series(S):
    """Shared step set across every series, at one common ntok, full-graph steps only.

    Split out from `aligned()` so the refusal paths can be exercised with synthetic series --
    a refusal that can only be reached with a fixture is a refusal nobody has tested.
    """
    empty = [k for k, v in S.items() if not v]
    if empty:
        raise Refuse("no union_sum= decoded in: " + ", ".join(sorted(empty)))
    shared = set.intersection(*[set(v) for v in S.values()])
    if not shared:
        raise Refuse("these runs share NO step numbers -- they are not comparable runs")
    ntok = st.mode([x["ntok"] for v in S.values() for x in v.values()])
    steps = sorted(s for s in shared if all(S[k][s]["ntok"] == ntok for k in S))
    if not steps:
        raise Refuse(f"shared steps exist ({len(shared)}) but none at a common ntok={ntok}")
    return steps, ntok, len(shared)


def aligned(logs):
    S = {k: series(v) for k, v in logs.items()}
    steps, ntok, nshared = align_series(S)
    return S, steps, ntok, nshared


def report(logs, verbose=True):
    S, steps, ntok, nshared = aligned(logs)
    base = st.median([S["a1"][s]["union"] for s in steps]) if "a1" in S else None
    if verbose:
        for k, v in S.items():
            own = st.median([x["union"] for x in v.values()])
            print(f"  {k:<10} {len(v):>4} 步  union 自身中位 {own:>7.2f} ms"
                  f"{'   (共享子集內)' if k in ('a1','b1') else ''}")
    print(f"  共同步 {nshared}，其中兩側同為 ntok={ntok} 的 {len(steps)} 步")
    return S, steps, ntok, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd", nargs=2, metavar=("ON", "CTL"), help="fwd half: on-log ctl-log")
    ap.add_argument("--rev", nargs=2, metavar=("ON", "CTL"), help="rev half (optional)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pick", type=float, metavar="SINCE_EPOCH",
                    help="list the server logs written after SINCE_EPOCH in mtime order and "
                         "print the --fwd/--rev arguments for them; refuses on an odd count or "
                         "on fewer than 4. Use this to collect a paired run's logs, because "
                         "paired_ab.py itself records no log paths.")
    a = ap.parse_args()

    if a.self_test:
        q = {
            "a1": "Backup/cgc_logs/llama_server_20260918_043439.log",   # on_fwd
            "b1": "Backup/cgc_logs/llama_server_20260918_043255.log",   # ctl_fwd
            "a2": "Backup/cgc_logs/llama_server_20260918_043746.log",   # on_rev
            "b2": "Backup/cgc_logs/llama_server_20260918_043625.log",   # ctl_rev
        }
        if not all(os.path.exists(v) for v in q.values()):
            raise SystemExit("selftest fixture logs missing")
        S = {k: series(v) for k, v in q.items()}
        # 1) a log against ITSELF must give exactly zero, on every statistic
        same = {k: q[k] for k in ("a1", "b1")}
        _, steps, _, base = report(same, verbose=False)
        d = [S["a1"][s]["union"] - S["a1"][s]["union"] for s in steps]
        assert len(steps) > 0 and all(x == 0 for x in d), "self-pair is not identically zero"
        assert se_median([1.0, 1.0, 1.0]) == 0.0
        print(f"  [1] 同一份 log 對自己: n={len(steps)} Δ≡0  base={base:.2f} ms  OK")
        # 2) swapping the halves must flip the sign of every delta
        f = [S["a1"][s]["union"] - S["b1"][s]["union"] for s in steps]
        r = [S["b1"][s]["union"] - S["a1"][s]["union"] for s in steps]
        assert all(abs(x + y) < 1e-9 for x, y in zip(f, r)), "sign flip broken"
        print(f"  [2] 交換兩半 ⇒ Δ 逐點變號: OK（中位 {st.median(f):+.2f} vs {st.median(r):+.2f} ms）")
        # 3) the three refusal paths, on synthetic series (no fixture can be trusted to be empty)
        def must_refuse(tag, S2):
            try:
                align_series(S2)
            except Refuse as e:
                print(f"  [3{tag}] 拒絕: {e}")
                return
            raise AssertionError(f"path {tag} did not refuse")
        must_refuse("a", {"x": {}, "y": series(q["a1"])})
        must_refuse("b", {"x": {1: dict(ntok=1, total=1.0, union=1.0, gap=0.0)},
                          "y": {2: dict(ntok=1, total=1.0, union=1.0, gap=0.0)}})
        must_refuse("c", {"x": {1: dict(ntok=1, total=1.0, union=1.0, gap=0.0)},
                          "y": {1: dict(ntok=4, total=1.0, union=1.0, gap=0.0)}})
        # 4) ntok filter: the local profiles must be excluded
        assert all(S["a1"][s]["ntok"] == st.mode([x["ntok"] for x in S["a1"].values()])
                   for s in steps), "mixed ntok leaked into the pair set"
        print("  [4] 配對步全為同一 ntok: OK")
        # 5) required-n arithmetic
        need = TARGET_FRAC * base
        sd = st.pstdev([(x - y) / 2 for x, y in zip(f, r)])
        print(f"  [5] 目標 {need:.2f} ms，σ(效應)={sd:.1f} ms ⇒ 需 n≈{(1.253*sd/(need/3))**2:.0f}")
        print("SELF-TEST OK")
        return

    if a.pick:
        import time as _t
        found = [(os.path.getmtime(f), f) for f in glob.glob("Backup/cgc_logs/llama_server_*.log")
                 if os.path.getmtime(f) >= a.pick and not os.path.islink(f)]
        # islink: llama_server_latest.log is a SYMLINK to the newest and would double-count it
        found.sort()
        print(f"自 {_t.strftime('%H:%M:%S', _t.localtime(a.pick))} 後寫入的 server log: {len(found)} 份")
        for mt, f in found:
            n, d = len(series(f)), decprof_steps(f)
            why = ("OK" if n else
                   "缺 CGC_GPU_TIMING=1（有 DECPROF 但沒有 gpu/union/gap 尾欄）" if d else
                   "缺 CGC_DECODE_PROFILE=1（完全沒有 DECPROF）")
            print(f"  {_t.strftime('%H:%M:%S', _t.localtime(mt))}  {os.path.basename(f):<44}"
                  f" union 步 {n:>3}/{d:<3} {why}")
        if not any(len(series(f)) for _, f in found):
            raise SystemExit("這一批 log 一份都讀不到 union —— 先確認 run 有開 "
                             "CGC_DECODE_PROFILE=1 **與** CGC_GPU_TIMING=1，否則窗口白跑")
        if len(found) % 2 or len(found) < 4:
            raise SystemExit(f"需要偶數且 >= 4 份（拿到 {len(found)}）—— 拒絕猜測排程")
        print("\n配對排程按 mtime：奇數索引 = A、偶數 = B。若 paired_ab.py 是『先 A 後 B』交錯，\n"
              "則 fwd = (1st,2nd) 與 rev = (3rd,4th)，但**要照它自己印的排程**，不要猜：")
        it = iter(found)
        for k, (m1, f1) in enumerate(it):
            m2, f2 = next(it)
            print(f"  pair {k+1}: --fwd {f1} {f2}" if k % 2 == 0 else
                  f"  pair {k+1}: --rev {f1} {f2}   （同上，A/B 需照排程對調）")
        return

    if not a.fwd:
        raise SystemExit("need --fwd ON CTL [--rev ON CTL], or --self-test")
    logs = {"a1": a.fwd[0], "b1": a.fwd[1]}
    if a.rev:
        logs["a2"], logs["b2"] = a.rev[0], a.rev[1]

    print("logs:")
    S, steps, ntok, base = report(logs)
    d1 = [S["a1"][s]["union"] - S["b1"][s]["union"] for s in steps]
    print(f"\n  fwd  Δunion = {st.median(d1):+8.2f} ms ({st.median(d1)/base:+.1%} of {base:.1f})"
          f"  on>ctl {sum(1 for x in d1 if x>0)}/{len(d1)}  SE={se_median(d1):.2f}")
    if "a2" in S:
        d2 = [S["a2"][s]["union"] - S["b2"][s]["union"] for s in steps]
        eff = [(x - y) / 2 for x, y in zip(d1, d2)]
        drift = [(x + y) / 2 for x, y in zip(d1, d2)]
        print(f"  rev  Δunion = {st.median(d2):+8.2f} ms ({st.median(d2)/base:+.1%})"
              f"  on>ctl {sum(1 for x in d2 if x>0)}/{len(d2)}  SE={se_median(d2):.2f}")
        print(f"\n  AB/BA 效應 (Δf-Δr)/2 = {st.median(eff):+8.2f} ms ({st.median(eff)/base:+.2%})"
              f"   SE={se_median(eff):.2f}")
        print(f"        漂移 (Δf+Δr)/2 = {st.median(drift):+8.2f} ms ({st.median(drift)/base:+.2%})")
        est, sd = eff, st.pstdev(eff)
        print(f"        半對半 = {st.median(d1)/base:+.1%} / {st.median(d2)/base:+.1%}"
              f"  ⇒ 先後兩次（未交錯）會把 {abs(st.median(d1)-st.median(d2))/2/base:.1%} 報成效應")
    else:
        est, sd = d1, st.pstdev(d1)
        print("\n  ⚠ 只有 fwd 一半 ⇒ 沒有漂移可扣，這個數字不能當效應量（見 doc §勘誤二）")

    need = TARGET_FRAC * base
    thr = 3 * se_median(est)
    print(f"\n  G4 目標: union 的 -{TARGET_FRAC:.1%} = {need:.2f} ms @ {base:.1f} ms")
    print(f"  本設計 3SE = {thr:.2f} ms = {thr/base:.1%} ⇒ "
          f"{'可分辨' if need > thr else '**分辨不了**（效應只有門檻的 %.1f×）' % (need/thr)}")
    print(f"  欲達 3SE 需 n ≈ {(1.253*sd/(need/3))**2:.0f} 個配對步（σ(逐步Δ)={sd:.1f} ms，"
          f"現有 n={len(steps)}）")


if __name__ == "__main__":
    main()
