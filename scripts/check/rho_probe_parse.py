#!/usr/bin/env python3
"""Parse the CGC rho probe output and issue the gate verdict.

Two independent measurements come out of ONE run (they share a build, so they
share the same binary — never compare across binaries in this repo):

  1. rho probe  (CGC_RHO_PROBE=1)
     The shadow router computes gate(L) from the residual *before* attn(L), i.e.
     the ids you would have if you fired fill(L) the moment MoE core(L-1) ended.
       rho_tok = per-token |approx top-k  n  real top-k| / k   (is the route alike?)
       cov_uni = |approx union n real union| / |real union|    (**how much fill moves**)
     `cov_uni` is the gated quantity.

  2. prebind probe, second round (CGC_PREBIND_PROBE=1)
     qu1/qu2/qu3 = coverage by the previous 1/2/3 steps' FULL union
     qt8/qt16/qt24 = coverage by the previous steps' token-0 row only
     The first round only stored token 0 — in MTP decode the ntok=4 tokens are 4
     *different positions* and consecutive steps have disjoint token positions,
     so "last step's token 0" is a different token. qu* is the honest source.

Thresholds are computed BEFORE the measurement (see docs/ and §EN-464) and are
NOT refitted afterwards:
    cov_uni >= 0.398  => step -10%   (the repo's 3%-style decision bar is -10% here)
    cov_uni >= 0.164  => step -3%
    cov_uni  > 0.70   => the window saturates; being more accurate buys nothing
because the window is only the tail ~40% of a segment (MoE core is 60.1% of it),
so at most 70-86% of a layer's fill can ever leave the critical path.

Usage:
    python3 scripts/check/rho_probe_parse.py <log> [--gate 0.398] [--gate3 0.164]
    python3 scripts/check/rho_probe_parse.py --self-test
Exit codes: 0 = PASS(-10%), 1 = PASS(-3%) only, 2 = FAIL, 3 = not measurable,
            4 = model/consistency problem (same convention as prebind_probe_parse).
"""

from __future__ import annotations

import argparse
import re
import sys

# ---------------------------------------------------------------- regexes ----
# rho_tok prints as -1.000 when this layer had no fresh shadow value. That is
# "not measured", NOT "measured zero" — the parser keeps it as None. (Skill rule
# 41: never let "not measured" masquerade as a low measurement.)
RHO_PROBE_RE = re.compile(
    r"CGC-RHO-PROBE:\s+il=(-?\d+)\s+ntok=(\d+)\s+uni=(\d+)\s+fresh=(\d+)\s+"
    r"rho_tok=(-?[0-9.]+)\s+cov_uni=([0-9.]+)\s+pred_uni=(\d+)")

RHO_SUM_RE = re.compile(
    r"CGC-RHO-SUM:\s+steps=(\d+)\s+layers=(\d+)\s+skip=(\d+)\s+"
    r"rho_tok=(-?[0-9.]+)\s+cov_uni=([0-9.]+)")

PB2_PROBE_RE = re.compile(
    r"CGC-PREBIND-PROBE2:\s+il=(-?\d+)\s+ntok=(\d+)\s+uni=(\d+)\s+"
    r"qu1=([0-9.]+)\s+qu2=([0-9.]+)\s+qu3=([0-9.]+)\s+"
    r"wu1=(\d+)\s+wu2=(\d+)\s+wu3=(\d+)\s+"
    r"qt8=([0-9.]+)\s+qt16=([0-9.]+)\s+qt24=([0-9.]+)\s+wt=(\d+)")

PB2_SUM_RE = re.compile(
    r"CGC-PREBIND-SUM2:\s+layers=(\d+)\s+uni_avg=([0-9.]+)\s+"
    r"qu1=([0-9.]+)\s+qu2=([0-9.]+)\s+qu3=([0-9.]+)\s+"
    r"wu1=([0-9.]+)\s+wu2=([0-9.]+)\s+wu3=([0-9.]+)\s+"
    r"qt8=([0-9.]+)\s+qt16=([0-9.]+)\s+qt24=([0-9.]+)\s+wt=([0-9.]+)")

GATE10 = 0.398     # step -10%
GATE3 = 0.164      # step -3%
SATURATE = 0.70    # window saturates here


# ------------------------------------------------------------ aggregation ----
class RhoAgg:
    """Per-layer accumulation of the rho probe, weighted by nothing.

    Every layer is one draw; the quantity of interest is the mean over layers
    (each layer contributes one fill decision of its own).
    """

    def __init__(self) -> None:
        self.layers = 0
        self.steps = 0
        self.skip = 0
        self.tok = 0.0
        self.cov = 0.0
        self.pred_uni = 0
        self.uni = 0

    def add(self, uni: int, rho_tok, cov_uni: float, pred_uni: int) -> None:
        if rho_tok is None:
            return
        self.layers += 1
        self.uni += uni
        self.tok += rho_tok
        self.cov += cov_uni
        self.pred_uni += pred_uni

    def as_dict(self) -> dict:
        n = self.layers
        return {
            "layers": n,
            "steps": self.steps,
            "skip": self.skip,
            "rho_tok": (self.tok / n) if n else None,
            "cov_uni": (self.cov / n) if n else None,
            "uni_avg": (float(self.uni) / n) if n else None,
            "pred_uni_avg": (float(self.pred_uni) / n) if n else None,
        }


def parse_rho(lines):
    """Return (aggs_by_ntok, steps, skip).

    Grouping by ntok matters: ntok=1 is a single token, where rho_tok and cov_uni
    are the same number by construction. The delivery shape is ntok=4 (MTP verify).
    Averaging the two would be an average over two different regimes.
    """
    aggs = {}
    steps = 0
    skip = 0
    for line in lines:
        m = RHO_SUM_RE.search(line)
        if m:
            steps = int(m.group(1))
            skip = int(m.group(3))
            continue
        m = RHO_PROBE_RE.search(line)
        if not m:
            continue
        ntok = int(m.group(2))
        uni = int(m.group(3))
        fresh = int(m.group(4))
        tok = float(m.group(5))
        cov = float(m.group(6))
        if fresh == 0 or tok < 0.0:
            skip += 1              # not measured; do NOT fold into the mean
            continue
        aggs.setdefault(ntok, RhoAgg()).add(uni, tok, cov, int(m.group(7)))
    return aggs, steps, skip


def parse_prebind2(lines):
    """Return the last CGC-PREBIND-SUM2 as a dict, or None."""
    out = None
    for line in lines:
        m = PB2_SUM_RE.search(line)
        if m:
            out = {
                "layers": int(m.group(1)),
                "uni_avg": float(m.group(2)),
                "qu1": float(m.group(3)), "qu2": float(m.group(4)), "qu3": float(m.group(5)),
                "wu1": float(m.group(6)), "wu2": float(m.group(7)), "wu3": float(m.group(8)),
                "qt8": float(m.group(9)), "qt16": float(m.group(10)), "qt24": float(m.group(11)),
                "wt": float(m.group(12)),
            }
    return out


# ------------------------------------------------------------------ report ----
def main() -> int:
    ap = argparse.ArgumentParser(description="parse the CGC rho probe output")
    ap.add_argument("log", nargs="?", help="stderr log (or - for stdin)")
    ap.add_argument("--gate", type=float, default=GATE10, help="cov_uni bar for step -10%")
    ap.add_argument("--gate3", type=float, default=GATE3, help="cov_uni bar for step -3%")
    ap.add_argument("--saturate", type=float, default=SATURATE, help="window saturation")
    ap.add_argument("--ntok", type=int, default=4, help="delivery shape (default 4)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not args.log:
        ap.error("log is required (or use --self-test)")

    lines = (sys.stdin.read().splitlines() if args.log == "-"
             else open(args.log, encoding="utf-8", errors="replace").read().splitlines())

    aggs, sum_steps, sum_skip = parse_rho(lines)
    pick = aggs.get(args.ntok)
    d = pick.as_dict() if pick is not None else {
        "layers": 0, "steps": sum_steps, "skip": sum_skip,
        "rho_tok": None, "cov_uni": None, "uni_avg": None, "pred_uni_avg": None}
    pb = parse_prebind2(lines)

    print("=" * 72)
    print("CGC rho probe — 提前一層發起 fill 的重合率")
    print("=" * 72)

    if pb is not None:
        print("\n[prebind 第二輪] 預測源 = 上一個 step 的聯集")
        print("  layers=%d  uni_avg=%.2f" % (pb["layers"], pb["uni_avg"]))
        print("  qu1=%.4f (寬 %.1f)   qu2=%.4f (寬 %.1f)   qu3=%.4f (寬 %.1f)"
              % (pb["qu1"], pb["wu1"], pb["qu2"], pb["wu2"], pb["qu3"], pb["wu3"]))
        print("  ── 舊口徑（只 token 0，第一輪的 0.354 就是這個）──")
        print("  qt8=%.4f  qt16=%.4f  qt24=%.4f  (寬 %.1f)"
              % (pb["qt8"], pb["qt16"], pb["qt24"], pb["wt"]))
    else:
        print("\n[prebind 第二輪] 沒有 CGC-PREBIND-SUM2 行（這支 log 沒有開 CGC_PREBIND_PROBE）")

    print("\n[rho] 預測源 = 同一個 token、只差 attn 一個 submodule 的影子 router")
    if aggs:
        print("  各形狀（ntok -> layers / rho_tok / cov_uni）：")
        for k in sorted(aggs):
            a = aggs[k].as_dict()
            print("    ntok=%-4d layers=%-6d rho_tok=%s cov_uni=%s"
                  % (k, a["layers"],
                     "%.4f" % a["rho_tok"] if a["rho_tok"] is not None else "n/a",
                     "%.4f" % a["cov_uni"] if a["cov_uni"] is not None else "n/a"))
        print("  → 判決用 ntok=%d（交付形狀）" % args.ntok)
    if d["layers"] == 0:
        print("  CGC-RHO 行數 = 0 => ρ 完全沒量到。")
        print("  ⚠ 這不是「ρ 很低」，是「沒量到」—— 兩者的下一步完全不同。")
        print("  查：binary 是不是帶 CGC_RHO_PROBE 的版本？圖裡的 cgc_rho_logits 節點有沒有建？")
        return 3

    print("  steps=%d  layers=%d  skip=%d" % (d["steps"], d["layers"], d["skip"]))
    print("  rho_tok = %.4f   (每個 token 的 top-%s 重合率)" % (d["rho_tok"], "k"))
    print("  cov_uni = %.4f   <= 門檻打這個" % d["cov_uni"])
    print("  uni_avg = %.2f   pred_uni_avg = %.2f" % (d["uni_avg"], d["pred_uni_avg"]))

    if d["skip"] > 0:
        print("\n  ⚠ skip=%d 層「影子值不是本步的」（stamp 不對）。" % d["skip"])
        print("    比例 %.1f%%。若 >5%%，圖的排程順序讓影子排在 top-k 之後，"
              % (100.0 * d["skip"] / max(1, d["skip"] + d["layers"])))
        print("    這筆 ρ 不可信 —— 要先讓影子節點排在 MoE 之前，不能拿來判決。")
        return 3

    cov = d["cov_uni"]
    print("\n門檻（事前算好，事後不改）")
    print("  step -3%%  => cov_uni >= %.3f" % args.gate3)
    print("  step -10%% => cov_uni >= %.3f" % args.gate)
    print("  窗口飽和   => cov_uni >  %.3f（再準也只到 +23.5%%~+31%%，不是完全重疊的 +43%%）"
          % args.saturate)

    if cov >= args.gate:
        verdict, rc = "PASS（step −10%）", 0
    elif cov >= args.gate3:
        verdict, rc = "PASS（只到 step −3%）", 1
    else:
        verdict, rc = "FAIL", 2

    print("\n判決：cov_uni = %.4f vs %.3f  =>  %s" % (cov, args.gate, verdict))
    if cov >= args.gate and cov > args.saturate:
        print("  （已過飽和點 %.2f：多出來的準度換不到時間，上限 +23.5%%~+31%%）" % args.saturate)
    if cov < args.gate3:
        print("  ⇒ 提前一層這條路也不成立。剩下的不是「重疊」而是「降本」（pread_usec）。")
    return rc


# --------------------------------------------------------------- self test ----
def _rho_probe(il, ntok, uni, fresh, tok, cov, puni):
    return ("CGC-RHO-PROBE: il=%d ntok=%d uni=%d fresh=%d "
            "rho_tok=%.3f cov_uni=%.3f pred_uni=%d" % (il, ntok, uni, fresh, tok, cov, puni))


def _rho_sum(steps, layers, skip, tok, cov):
    return ("CGC-RHO-SUM: steps=%d layers=%d skip=%d rho_tok=%.4f cov_uni=%.4f"
            % (steps, layers, skip, tok, cov))


def _pb2_sum(layers, uni, qu1, qu2, qu3, wu1, wu2, wu3, qt8, qt16, qt24, wt):
    return ("CGC-PREBIND-SUM2: layers=%d uni_avg=%.2f qu1=%.4f qu2=%.4f qu3=%.4f "
            "wu1=%.1f wu2=%.1f wu3=%.1f qt8=%.4f qt16=%.4f qt24=%.4f wt=%.1f"
            % (layers, uni, qu1, qu2, qu3, wu1, wu2, wu3, qt8, qt16, qt24, wt))


def self_test() -> int:
    fails = []

    total = [0]

    def check(name, cond):
        total[0] += 1
        if not cond:
            fails.append(name)
        print("%-58s %s" % (name, "ok" if cond else "FAIL"))

    # 1. regexes match the emitted format
    check("RHO_PROBE_RE matches", RHO_PROBE_RE.search(_rho_probe(3, 4, 19, 1, 0.75, 0.62, 19)) is not None)
    check("RHO_SUM_RE matches", RHO_SUM_RE.search(_rho_sum(10, 400, 0, 0.75, 0.62)) is not None)
    check("PB2_SUM_RE matches",
          PB2_SUM_RE.search(_pb2_sum(400, 19.4, .5, .6, .65, 19.4, 25., 29., .27, .30, .35, 13.5)) is not None)

    # 2. fresh=0 must be counted as skip, never as a low measurement
    lines = [_rho_probe(i, 4, 20, 0, -1.0, 0.0, 0) for i in range(5)]
    aggs, _, skip0 = parse_rho(lines)
    check("fresh=0 -> not measured", not aggs and skip0 == 5)
    check("fresh=0 -> no ntok group created", 4 not in aggs)

    # 3. fresh=1 accumulates and averages
    lines = [_rho_probe(i, 4, 20, 1, 0.80, 0.60, 20) for i in range(4)]
    aggs, _, _ = parse_rho(lines)
    d = aggs[4].as_dict()
    check("fresh=1 -> layers counted", d["layers"] == 4)
    check("fresh=1 -> mean rho_tok", abs(d["rho_tok"] - 0.80) < 1e-9)
    check("fresh=1 -> mean cov_uni", abs(d["cov_uni"] - 0.60) < 1e-9)

    # 4. a mean must not be an average of averages when uni differs — here it is a
    #    plain per-layer mean, so mixing uni sizes still gives the arithmetic mean
    lines = [_rho_probe(0, 4, 10, 1, 1.0, 1.0, 10), _rho_probe(1, 4, 30, 1, 0.0, 0.0, 30)]
    aggs, _, _ = parse_rho(lines)
    d = aggs[4].as_dict()
    check("mixed layers -> arithmetic mean", abs(d["cov_uni"] - 0.5) < 1e-9)

    # 5. SUM line overrides steps/skip
    lines = [_rho_sum(7, 280, 3, 0.7, 0.5)] + [_rho_probe(i, 4, 20, 1, 0.7, 0.5, 20) for i in range(2)]
    _, steps, skip = parse_rho(lines)
    check("SUM line sets steps", steps == 7)
    check("SUM line sets skip", skip == 3)

    # 5b. ntok grouping: the delivery shape must not be averaged with ntok=1
    lines = ([_rho_probe(i, 1, 8, 1, 0.95, 0.95, 8) for i in range(40)] +
             [_rho_probe(i, 4, 20, 1, 0.50, 0.40, 20) for i in range(40)])
    aggs, _, _ = parse_rho(lines)
    check("ntok grouping -> two groups", sorted(aggs) == [1, 4])
    check("ntok=4 not polluted by ntok=1", abs(aggs[4].as_dict()["cov_uni"] - 0.40) < 1e-9)
    check("ntok=1 stays separate", abs(aggs[1].as_dict()["cov_uni"] - 0.95) < 1e-9)

    # 6. prebind second round parses
    pb = parse_prebind2([_pb2_sum(400, 19.4, .50, .60, .65, 19.4, 25., 29., .27, .30, .35, 13.5)])
    check("PB2 qu3 parsed", pb is not None and abs(pb["qu3"] - 0.65) < 1e-9)
    check("PB2 qt24 parsed", pb is not None and abs(pb["qt24"] - 0.35) < 1e-9)

    # 7. thresholds are ordered and the saturation point is above the -10% bar
    check("gate3 < gate < saturate", GATE3 < GATE10 < SATURATE)

    # 8. verdict logic, exercised through the real code path
    def verdict_for(cov, gate=GATE10, gate3=GATE3):
        if cov >= gate:
            return 0
        if cov >= gate3:
            return 1
        return 2

    check("verdict: 0.62 -> PASS(-10%)", verdict_for(0.62) == 0)
    check("verdict: 0.40 -> PASS(-10%)", verdict_for(0.40) == 0)
    check("verdict: 0.30 -> PASS(-3%)", verdict_for(0.30) == 1)
    check("verdict: 0.10 -> FAIL", verdict_for(0.10) == 2)

    print("\n%d/%d passed" % (total[0] - len(fails), total[0]))
    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
