#!/usr/bin/env python3
"""gap_attribution.py -- 把 GPU 空窗 (gap_sum) 归到 CPU 的 hook + submit 上。

为什么要有这支工具（不要用手抄表格代替它）：
  `docs/STEP_SERIALIZATION_2026-09-23.md` 的整条论证建立在两个回归数上
  （gap ~ (cb+submit) 的斜率与截距）。手抄一次就会失去「换一批 log 还能不能
  重现」的检验能力，而斜率正好是「预指派 slot 值不值得做」的定价参数：
  斜率 ~1.0 表示 CPU 的每一毫秒全额变成 GPU 空窗 ⇒ 把 CPU 挪出关键路径就有
  全额回报；斜率 ~0 表示空窗另有来源 ⇒ 预指派打错靶。

量的是什么（口径，读代码得来，不是从名字推）：
  `CGC-DECPROF` 一行 = 一个 decode step，由 `ggml-backend.cpp` 的 `hook_seg` 汇总：
    wait   = st1-st0  CPU 自旋忙等 GPU 跑完该段   （**不是** GPU 空转）
    cb     = st2-st1  top-k hook（ensure_batch 占 99%）
    submit = 提交下一段
    union  = GPU 忙的跨度（Metal 时间戳）
    gap    = GPU 空转的跨度（上段 end -> 下段 start）
  闭合式：union + gap ~= total（实测差 2%）。

零 GPU：只读既有 stderr log。

用法
    python3 scripts/check/gap_attribution.py --selftest
    python3 scripts/check/gap_attribution.py --glob '/tmp/verify_layers/*prod25_decode-spec'
    python3 scripts/check/gap_attribution.py --dir <dir> --min-step 100 --segs 41

输出三张表：
  (1) 按 ntok 分组的中位数     -> MTP 的 CPU 成本涨了多少
  (2) arm x ntok 交叉表        -> 排除「不同 arm 本来就有不同 launch 状态」的混淆
  (3) gap ~ (cb+submit) 回归   -> 斜率/截距/相关系数（本工具的定价参数）
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict

# CGC-DECPROF: step=N segs=N layers=N total=X ms | wait=X (X%) cb=X (X%) submit=X (X%) ntok=N
#              | layer gpu_sum=X union_sum=X gap_sum=X
DECPROF = re.compile(
    r"CGC-DECPROF: step=(\d+) segs=(\d+) layers=(\d+) total=([\d.]+) ms \| "
    r"wait=([\d.]+) \(([\d.]+)%\) cb=([\d.]+) \(([\d.]+)%\) submit=([\d.]+) \(([\d.]+)%\) "
    r"ntok=(\d+) \| layer gpu_sum=([\d.]+) union_sum=([\d.]+) gap_sum=([\d.]+)"
)

FIELDS = ("total", "wait", "cb", "submit", "ntok", "gpu", "union", "gap")


def parse_text(text: str, arm: str = ""):
    """把一段 log 文本解析成 row 列表。row 是 dict，含 step/segs/arm + FIELDS。"""
    out = []
    for m in DECPROF.finditer(text):
        g = m.groups()
        out.append({
            "arm": arm,
            "step": int(g[0]),
            "segs": int(g[1]),
            "total": float(g[3]),
            "wait": float(g[4]),
            "cb": float(g[6]),
            "submit": float(g[8]),
            "ntok": int(g[10]),
            "gpu": float(g[11]),
            "union": float(g[12]),
            "gap": float(g[13]),
        })
    return out


def load_dirs(dirs):
    rows = []
    for d in dirs:
        logs = sorted(glob.glob(os.path.join(d, "*.stderr.log")))
        if not logs:
            continue
        try:
            with open(logs[-1], "r", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        rows.extend(parse_text(text, arm=os.path.basename(d.rstrip("/"))))
    return rows


def steady(rows, min_step=100, segs=None):
    """稳态过滤：只留 step >= min_step 的（warmup 的 DECPROF 首行是 37/51/12，会误导），
    并可选只留某个 segs（verify step 与 draft step 混在同一支 log 里）。"""
    out = [r for r in rows if r["step"] >= min_step]
    if segs is not None:
        out = [r for r in out if r["segs"] == segs]
    return out


def regress(xs, ys):
    """最小二乘 y = slope*x + intercept，附 Pearson r。空/退化输入回 None。"""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    var = sum((a - mx) ** 2 for a in xs) / n
    if var <= 0.0:
        return None
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / n
    slope = cov / var
    intercept = my - slope * mx
    sx, sy = st.pstdev(xs), st.pstdev(ys)
    r = (cov / (sx * sy)) if (sx > 0 and sy > 0) else 0.0
    return {"slope": slope, "intercept": intercept, "r": r, "n": n}


def med(rows, key):
    return st.median([r[key] for r in rows]) if rows else 0.0


def ntok_table(rows, min_n=20):
    """表 (1)：按 ntok 分组的中位数。"""
    per = defaultdict(list)
    for r in rows:
        per[r["ntok"]].append(r)
    out = []
    for n in sorted(per):
        v = per[n]
        if len(v) < min_n:
            continue
        cs = med(v, "cb") + med(v, "submit")
        out.append({
            "ntok": n, "n": len(v),
            "total": med(v, "total"), "wait": med(v, "wait"),
            "cb": med(v, "cb"), "submit": med(v, "submit"),
            "union": med(v, "union"), "gap": med(v, "gap"),
            "cb_sub": cs,
            "ratio": (med(v, "gap") / cs) if cs else 0.0,
            "gap_pct": (med(v, "gap") / med(v, "total") * 100.0) if med(v, "total") else 0.0,
        })
    return out


def arm_ntok_table(rows, min_n=20):
    """表 (2)：arm x ntok —— 同一 arm 内部比 ntok，才不会被 arm 间的 launch 状态差骗。"""
    per = defaultdict(list)
    for r in rows:
        per[(r["arm"], r["ntok"])].append(r)
    out = []
    for (arm, n) in sorted(per):
        v = per[(arm, n)]
        if len(v) < min_n:
            continue
        out.append({
            "arm": arm, "ntok": n, "n": len(v),
            "total": med(v, "total"), "cb": med(v, "cb"),
            "gap": med(v, "gap"), "union": med(v, "union"),
            "gap_pct": (med(v, "gap") / med(v, "total") * 100.0) if med(v, "total") else 0.0,
        })
    return out


def regression_table(rows, min_n=30):
    """表 (3)：gap ~ (cb+submit)，按 ntok 分组。斜率是本工具的定价参数。"""
    per = defaultdict(list)
    for r in rows:
        per[r["ntok"]].append(r)
    out = []
    for n in sorted(per):
        v = per[n]
        if len(v) < min_n:
            continue
        fit = regress([r["cb"] + r["submit"] for r in v], [r["gap"] for r in v])
        if fit is None:
            continue
        fit["ntok"] = n
        out.append(fit)
    # 整体（不分 ntok）也报一条
    fit = regress([r["cb"] + r["submit"] for r in rows], [r["gap"] for r in rows])
    if fit is not None:
        fit["ntok"] = "*"
        out.append(fit)
    return out


def closure(rows):
    """闭合检查：median(union) + median(gap) 应该 ~= median(total)。"""
    if not rows:
        return None
    u, g, t = med(rows, "union"), med(rows, "gap"), med(rows, "total")
    return {"union": u, "gap": g, "total": t,
            "sum": u + g, "err_pct": ((u + g) - t) / t * 100.0 if t else 0.0}


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description="attribute GPU idle (gap) to CPU hook+submit")
    ap.add_argument("--dir", action="append", default=[], help="log 目录（可多次）")
    ap.add_argument("--glob", default=None, help="log 目录的 glob")
    ap.add_argument("--min-step", type=int, default=100)
    ap.add_argument("--segs", type=int, default=41, help="只留这个 segs 的 step（41=verify）")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    dirs = list(args.dir)
    if args.glob:
        dirs.extend(sorted(glob.glob(args.glob)))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        print("no log dirs given", file=sys.stderr)
        return 2

    rows = steady(load_dirs(dirs), args.min_step, args.segs)
    if not rows:
        print("no DECPROF rows after filtering", file=sys.stderr)
        return 2

    c = closure(rows)
    print("rows after filter: %d   (min-step=%d, segs=%s)" % (len(rows), args.min_step, args.segs))
    print("closure: union %.1f + gap %.1f = %.1f  vs total %.1f   (err %+.1f%%)"
          % (c["union"], c["gap"], c["sum"], c["total"], c["err_pct"]))
    print()

    print("(1) by ntok -- MTP's CPU cost")
    print("%5s %6s %8s %8s %8s %8s %8s %8s %9s" %
          ("ntok", "n", "total", "wait", "cb", "submit", "gap", "cb+sub", "gap/(cb+sub)"))
    for r in ntok_table(rows):
        print("%5d %6d %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %9.2f" %
              (r["ntok"], r["n"], r["total"], r["wait"], r["cb"], r["submit"],
               r["gap"], r["cb_sub"], r["ratio"]))
    print()

    print("(2) arm x ntok -- same-arm comparison (kills the arm confound)")
    print("%-34s %5s %6s %8s %8s %8s %8s %8s" %
          ("arm", "ntok", "n", "total", "cb", "gap", "union", "gap%"))
    for r in arm_ntok_table(rows):
        print("%-34s %5d %6d %8.1f %8.1f %8.1f %8.1f %7.0f%%" %
              (r["arm"][-34:], r["ntok"], r["n"], r["total"], r["cb"], r["gap"],
               r["union"], r["gap_pct"]))
    print()

    print("(3) gap ~ (cb+submit)   slope ~= 1 means CPU cost passes through to GPU idle 1:1")
    print("%5s %6s %9s %12s %7s" % ("ntok", "n", "slope", "intercept", "r"))
    for f in regression_table(rows):
        print("%5s %6d %9.2f %+9.1f ms %7.3f" %
              (f["ntok"], f["n"], f["slope"], f["intercept"], f["r"]))
    return 0


# ---------------------------------------------------------------- selftest

def selftest():
    import tempfile

    ok = bad = 0

    def check(name, cond):
        nonlocal ok, bad
        if cond:
            ok += 1
        else:
            bad += 1
            print("FAIL: %s" % name)

    # -- 解析：一行真实格式的 DECPROF
    line = ("CGC-DECPROF: step=120 segs=41 layers=40 total=161.0 ms | "
            "wait=123.2 (78.1%) cb=23.8 (15.0%) submit=10.8 (7.0%) ntok=4 | "
            "layer gpu_sum=300.0 union_sum=113.4 gap_sum=44.4")
    rows = parse_text(line, arm="fake")
    check("parses one row", len(rows) == 1)
    r = rows[0]
    check("total", abs(r["total"] - 161.0) < 1e-9)
    check("wait", abs(r["wait"] - 123.2) < 1e-9)
    check("cb", abs(r["cb"] - 23.8) < 1e-9)
    check("submit", abs(r["submit"] - 10.8) < 1e-9)
    check("ntok", r["ntok"] == 4)
    check("segs", r["segs"] == 41)
    check("gap", abs(r["gap"] - 44.4) < 1e-9)
    check("union", abs(r["union"] - 113.4) < 1e-9)
    check("gpu_sum is separate from union", abs(r["gpu"] - 300.0) < 1e-9)

    # -- 回归：构造 y = 1.0*x + 10
    xs = [10.0, 20.0, 30.0, 40.0, 50.0]
    ys = [1.0 * x + 10.0 for x in xs]
    f = regress(xs, ys)
    check("regress slope 1.0", f and abs(f["slope"] - 1.0) < 1e-9)
    check("regress intercept 10", f and abs(f["intercept"] - 10.0) < 1e-9)
    check("regress r 1.0", f and abs(f["r"] - 1.0) < 1e-9)

    # -- 回归：斜率 0（空窗与 CPU 无关）必须能被分辨
    f0 = regress(xs, [7.0] * len(xs))
    check("flat y -> slope 0", f0 and abs(f0["slope"]) < 1e-9)
    check("flat y -> r 0", f0 and abs(f0["r"]) < 1e-9)

    # -- 回归：退化输入（常数 x）回 None 而不是除零
    check("constant x -> None", regress([5.0] * 5, [1.0, 2.0, 3.0, 4.0, 5.0]) is None)
    check("too few -> None", regress([1.0, 2.0], [1.0, 2.0]) is None)
    check("mismatched len -> None", regress([1.0, 2.0, 3.0], [1.0, 2.0]) is None)

    # -- 斜率 0.5（CPU 只有一半转空窗）
    f2 = regress(xs, [0.5 * x + 3.0 for x in xs])
    check("slope 0.5", f2 and abs(f2["slope"] - 0.5) < 1e-9)
    check("intercept 3", f2 and abs(f2["intercept"] - 3.0) < 1e-9)

    # -- 稳态过滤：warmup 必须被丢掉；segs 过滤要生效
    mix = parse_text(
        "CGC-DECPROF: step=1 segs=41 layers=40 total=99.0 ms | wait=37.0 (37%) "
        "cb=51.0 (51%) submit=12.0 (12%) ntok=1 | layer gpu_sum=1 union_sum=10 gap_sum=5\n"
        "CGC-DECPROF: step=200 segs=41 layers=40 total=161.0 ms | wait=123.0 (78%) "
        "cb=24.0 (15%) submit=11.0 (7%) ntok=4 | layer gpu_sum=1 union_sum=113 gap_sum=44\n"
        "CGC-DECPROF: step=200 segs=2 layers=1 total=1.6 ms | wait=1.3 (81%) "
        "cb=0.0 (0%) submit=0.2 (12%) ntok=1 | layer gpu_sum=1 union_sum=11 gap_sum=0\n",
        arm="fake")
    check("parses 3 rows", len(mix) == 3)
    check("min-step drops warmup", len(steady(mix, 100, None)) == 2)
    check("segs filter keeps only 41", len(steady(mix, 100, 41)) == 1)
    check("survivor is the ntok=4 row", steady(mix, 100, 41)[0]["ntok"] == 4)
    check("warmup row is the step=1 one", steady(mix, 0, 41)[0]["step"] == 1)

    # -- 表：min_n 门槛要挡住小样本
    tbl = ntok_table(steady(mix, 100, None), min_n=20)
    check("min_n suppresses tiny groups", tbl == [])
    tbl = ntok_table(steady(mix, 100, None), min_n=1)
    check("min_n=1 admits them", len(tbl) == 2)

    # -- 闭合：union+gap vs total
    cl = closure(steady(mix, 100, 41))
    check("closure sums", abs(cl["sum"] - (113.0 + 44.0)) < 1e-9)
    check("closure err sign", cl["err_pct"] < 0)  # 157 vs 161 -> negative

    # -- 端到端：写一支假 log 跑 load_dirs
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "a.stderr.log"), "w") as fh:
            for i in range(30):
                fh.write(
                    "CGC-DECPROF: step=%d segs=41 layers=40 total=160.0 ms | wait=120.0 (75%%) "
                    "cb=30.0 (18%%) submit=10.0 (7%%) ntok=4 | layer gpu_sum=1 union_sum=110.0 "
                    "gap_sum=50.0\n" % (100 + i))
        got = load_dirs([td])
        check("load_dirs reads log", len(got) == 30)
        check("load_dirs tags arm", got[0]["arm"] == os.path.basename(td))
        st_rows = steady(got, 100, 41)
        check("steady keeps all 30", len(st_rows) == 30)
        at = arm_ntok_table(st_rows, min_n=20)
        check("arm x ntok has one row", len(at) == 1 and at[0]["ntok"] == 4)

    print("selftest: %d/%d passed" % (ok, ok + bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
