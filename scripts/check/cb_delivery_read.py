#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cb_delivery_read.py -- 把 `cb` 從一支交付 cell 的 log 裡讀出來，並**標明它屬於哪個窗口**。

為什麼要有這支（2026-09-23 17:5x）
--------------------------------
「cb 到底是 42 還是 74」被當成兩個 regime 的分歧爭了兩天，而它決定了
「把 expert fill 藏進 GPU 陰影」值 +14% 還是 +24~31%（`window_joint_ev.py`）。

本輪在**交付 cell 本體**（prod25-stream / -n 128 -d 512 -b 512 -c 4096 --warm-skip 64
--spec-type draft-mtp）上開 `CGC_DECODE_PROFILE=1` 實跑，同一支 log 得到：

    全部步（含池冷啟動）  cb mean 72.07 ms   佔 step 28.5%
    穩態（對齊 warm-skip）cb mean 36.81 ms   佔 step 18.3%

⇒ **74.18 是「含預熱」的數，42.04 是「穩態」的數。它們是同一支 run 的兩個窗口，
  不是兩個 regime。** 交付錨點 12.57 t/s 用 `--warm-skip 64`（時鐘從預熱之後才開始）
  ⇒ 交付 regime 的 `cb` 是**穩態那一個 ≈ 42**，不是 74。

這支把那件事做成可重跑、可自測的動作，讓任何人不必再相信上面這段話。

它順手修掉的三個坑（每一個都曾讓數字讀錯）
----------------------------------------
1. **prefill 圖也算一個 step row**：`-d 512` 的 prefill 是一整個 `graph_compute`，
   印成 `ntok=512, cb=5791 ms`。它落在第一個 rep 的頭上，一筆就把 rep1 的 mean
   從 61.8 拉到 134.4。⇒ 用 `--max-ntok` 排除（預設 16）。
2. **MTP draft 步不是 verify 步**：dp_layers 只有兩個值 —— **1（draft，MTP 模組一層）
   與 40（verify，主模型）**。混在一起 median 會描述一個不存在的母體。
   ⇒ 用 `--layers` 只取 verify 步（預設 40，0 = 全部）。
3. **定價要用 mean 不是 median**：總時間 = Σcb，所以能換多少 t/s 取決於 **mean**，
   而 `joint_reconcile.py` 報的是 median。本支兩個都報，並額外報
   **時間加權佔比 Σcb/Σtotal**（這才是「藏掉 cb 能省多少 step」的那個數）。

用法
    python3 scripts/check/cb_delivery_read.py --self-test
    python3 scripts/check/cb_delivery_read.py <stderr.log> [--reps 3] [--warm-steps 27]
    python3 scripts/check/cb_delivery_read.py <stderr.log> --layers 0 --max-ntok 99999

EXIT  0 ok | 2 沒有可用的 step row
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys

# 與 joint_reconcile.py 相同的 regex（ggml-backend.cpp:2720 的列格式）。
STEP_RX = re.compile(
    r"CGC-DECPROF: step=(\d+) segs=(\d+) layers=(\d+) total=([\d.]+) ms \| "
    r"wait=([\d.]+) \([\d.]+%\) cb=([\d.]+) \([\d.]+%\) submit=([\d.]+) \([\d.]+%\) ntok=(\d+)"
    r"(?: \| layer gpu_sum=([\d.]+) union_sum=([\d.]+) gap_sum=([\d.]+) ms)?")


def parse_log(path: str, layers: int = 40, max_ntok: int = 16) -> list[dict]:
    """取出 verify 步的 step row。`layers`=0 表示不分層；`max_ntok` 排除 prefill 圖。"""
    out: list[dict] = []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = STEP_RX.search(line)
            if not m:
                continue
            ntok = int(m.group(8))
            n_layers = int(m.group(3))
            if ntok > max_ntok:
                continue
            if layers and n_layers != layers:
                continue
            out.append({
                "step": int(m.group(1)), "segs": int(m.group(2)), "layers": n_layers,
                "total": float(m.group(4)), "wait": float(m.group(5)),
                "cb": float(m.group(6)), "submit": float(m.group(7)), "ntok": ntok,
            })
    return out


def split_reps(rows: list[dict], reps: int) -> list[list[dict]]:
    """切成 reps 段。優先用 step 號回跳找邊界；回跳不出來就等分。

    llama-bench 的 `-r 3` 不重置 dp_step（實測一支 log 內 step 1..867 連續），
    所以「等分」在多數情況下就是對的，而回跳分支是給會重置的那種 log 用的。
    """
    if reps <= 1:
        return [rows]
    bnd = [i for i in range(1, len(rows)) if rows[i]["step"] < rows[i - 1]["step"]]
    if len(bnd) == reps - 1:
        edges = [0] + bnd + [len(rows)]
        return [rows[edges[k]:edges[k + 1]] for k in range(reps)]
    n = len(rows) // reps
    return [rows[k * n:(k + 1) * n] for k in range(reps)]


def summarise(rows: list[dict]) -> dict:
    """一個窗口的統計。時間加權佔比才是定價要的那個數。"""
    if not rows:
        return {}
    s_total = sum(r["total"] for r in rows)
    s_cb = sum(r["cb"] for r in rows)
    s_w = sum(r["wait"] for r in rows)
    s_s = sum(r["submit"] for r in rows)
    return {
        "n": len(rows),
        "cb_mean": st.mean(r["cb"] for r in rows),
        "cb_median": st.median(r["cb"] for r in rows),
        "total_mean": st.mean(r["total"] for r in rows),
        "wait_mean": st.mean(r["wait"] for r in rows),
        "submit_mean": st.mean(r["submit"] for r in rows),
        # 定價用的量：藏掉 cb 之後 step 少掉的百分比
        "cb_pct_timewt": 100.0 * s_cb / s_total if s_total else float("nan"),
        "cb_pct_medratio": st.median(100.0 * r["cb"] / r["total"] for r in rows if r["total"]),
        "sum_cb": s_cb, "sum_total": s_total,
        # self-check：total 的定義就是 wait+cb+submit（ggml-backend.cpp:2691）
        "additive_resid": s_total - (s_w + s_cb + s_s),
        "mean_ntok": st.mean(r["ntok"] for r in rows),
    }


def windowed(rows: list[dict], reps: int, warm_steps: int) -> dict:
    """回傳「全部」與「穩態」兩個窗口，讓 74 與 42 各自對號入座。"""
    chunks = split_reps(rows, reps)
    steady = [r for c in chunks for r in c[warm_steps:]]
    return {"all": summarise(rows), "steady": summarise(steady),
            "reps": [summarise(c) for c in chunks],
            "rep_steady": [summarise(c[warm_steps:]) for c in chunks],
            "warm_steps": warm_steps, "n_reps": reps}


def _fmt(tag: str, s: dict) -> str:
    if not s:
        return "%-24s (no rows)" % tag
    return ("%-24s n=%3d | cb mean %7.2f  med %7.2f | total mean %7.2f | "
            "cb%% %5.1f%% (Σ) %5.1f%% (med)") % (
        tag, s["n"], s["cb_mean"], s["cb_median"], s["total_mean"],
        s["cb_pct_timewt"], s["cb_pct_medratio"])


def report(res: dict) -> None:
    print(_fmt("★ 全部（含池預熱）", res["all"]))
    print(_fmt("★ 穩態（去每 rep 前 %d 步）" % res["warm_steps"], res["steady"]))
    print()
    for k, (a, b) in enumerate(zip(res["reps"], res["rep_steady"])):
        print(_fmt("  rep%d 全部" % (k + 1), a))
        print(_fmt("  rep%d 穩態" % (k + 1), b))
    print()
    a, s = res["all"], res["steady"]
    if a and s:
        print("  預熱貢獻：cb mean %.2f -> %.2f ms（×%.2f）；佔比 %.1f%% -> %.1f%%" % (
            a["cb_mean"], s["cb_mean"], a["cb_mean"] / s["cb_mean"] if s["cb_mean"] else float("nan"),
            a["cb_pct_timewt"], s["cb_pct_timewt"]))
        print("  ⇒ 交付 regime（`--warm-skip` 之後）該引用的是穩態那一列。")


# ------------------------------------------------------------------ selftest
def cmd_selftest() -> int:
    ok = 0
    fails: list[str] = []

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        if cond:
            ok += 1
        else:
            fails.append(name)

    # 1-4. parser：prefill 圖被排除、draft 步（layers=1）被排除、欄位正確
    def row(step, layers, total, wait, cb, sub, ntok):
        return ("CGC-DECPROF: step=%d segs=8 layers=%d total=%.2f ms | wait=%.2f (%.0f%%) "
                "cb=%.2f (%.0f%%) submit=%.2f (%.0f%%) ntok=%d" % (
                    step, layers, total, wait, 0.0, cb, 0.0, sub, 0.0, ntok))

    import tempfile, os
    lines = [
        row(1, 40, 9287.94, 2167.92, 5791.08, 1328.93, 512),   # prefill 圖：必須被排除
        row(2, 1, 50.0, 40.0, 0.02, 10.0, 1),                  # MTP draft：必須被排除
        row(3, 40, 200.0, 150.0, 40.0, 10.0, 4),
        row(4, 40, 200.0, 140.0, 50.0, 10.0, 4),
        row(5, 40, 200.0, 160.0, 30.0, 10.0, 3),
    ]
    fd, p = tempfile.mkstemp(suffix=".log")
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    try:
        rows = parse_log(p, layers=40, max_ntok=16)
        check("排除 prefill + draft，只留 verify", len(rows) == 3)
        check("cb 欄位正確", [r["cb"] for r in rows] == [40.0, 50.0, 30.0])
        check("ntok 欄位正確", [r["ntok"] for r in rows] == [4, 4, 3])
        check("total 欄位正確", [r["total"] for r in rows] == [200.0, 200.0, 200.0])

        # 5-7. summarise：mean / median / 時間加權佔比
        s = summarise(rows)
        check("cb_mean = 40", abs(s["cb_mean"] - 40.0) < 1e-9)
        check("cb_median = 40", abs(s["cb_median"] - 40.0) < 1e-9)
        check("cb 時間加權佔比 = 20%", abs(s["cb_pct_timewt"] - 20.0) < 1e-9)
        check("加性殘差 = 0（total 的定義）", abs(s["additive_resid"]) < 1e-6)

        # 8. 時間加權佔比 != mean-of-ratio：構造一個能分辨的例子
        two = [{"cb": 10.0, "total": 100.0, "wait": 80.0, "submit": 10.0, "ntok": 4},
               {"cb": 10.0, "total": 100.0, "wait": 80.0, "submit": 10.0, "ntok": 4}]
        s2 = summarise(two)
        check("兩個統計量在均勻時相等", abs(s2["cb_pct_timewt"] - s2["cb_pct_medratio"]) < 1e-9)

        # 9. 視窗化：預熱段拉高 all、steady 較低
        warm = [{"cb": 500.0, "total": 600.0, "wait": 80.0, "submit": 20.0, "ntok": 4,
                 "step": i, "layers": 40, "segs": 8} for i in range(5)]
        cold = [{"cb": 40.0, "total": 200.0, "wait": 150.0, "submit": 10.0, "ntok": 4,
                 "step": 5 + i, "layers": 40, "segs": 8} for i in range(45)]
        res = windowed(warm + cold, reps=1, warm_steps=5)
        # 倍數門檻只取「明顯拉高」：5 步 500 ms 混 45 步 40 ms ⇒ all=86 vs steady=40（×2.15）。
        check("all 的 cb_mean 明顯高於 steady", res["all"]["cb_mean"] > res["steady"]["cb_mean"] * 1.5)
        check("steady 的 cb_mean = 40", abs(res["steady"]["cb_mean"] - 40.0) < 1e-9)
        check("steady 的 cb 佔比 = 20%", abs(res["steady"]["cb_pct_timewt"] - 20.0) < 1e-9)

        # 10. 不變式：穩態佔比必須低於全部佔比（預熱只會把 cb 抬高）
        check("穩態佔比 <= 全部佔比", res["steady"]["cb_pct_timewt"] <= res["all"]["cb_pct_timewt"] + 1e-9)

        # 11. reps 切分：3 rep 等分
        many = [{"cb": 40.0 + i % 7, "total": 200.0, "wait": 150.0, "submit": 10.0,
                 "ntok": 4, "step": i, "layers": 40, "segs": 8} for i in range(90)]
        r3 = windowed(many, reps=3, warm_steps=10)
        check("3 rep 切分", len(r3["reps"]) == 3 and all(x["n"] == 30 for x in r3["reps"]))
        check("每 rep 穩態 n=20", all(x["n"] == 20 for x in r3["rep_steady"]))

        # 12. 空輸入不炸
        check("空輸入回傳空 dict", summarise([]) == {})
    finally:
        os.unlink(p)

    total = ok + len(fails)
    for f in fails:
        print("FAIL: %s" % f)
    print("selftest: %d/%d passed" % (ok, total))
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="從一支交付 log 讀 cb，並標明它屬於哪個窗口")
    ap.add_argument("log", nargs="?", default="")
    ap.add_argument("--layers", type=int, default=40,
                    help="只取 layers==N 的 verify 步（0 = 全部；MTP draft 步是 layers=1）")
    ap.add_argument("--max-ntok", type=int, default=16,
                    help="排除 ntok 大於此值的圖（prefill 是一整個 graph，實測 ntok=512）")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warm-steps", type=int, default=27,
                    help="每個 rep 丟掉前面幾步（--warm-skip 64 token ÷ ~2.4 token/步 ≈ 27）")
    ap.add_argument("--json", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return cmd_selftest()
    if not args.log:
        print("need a log (or --self-test)", file=sys.stderr)
        return 2
    rows = parse_log(args.log, layers=args.layers, max_ntok=args.max_ntok)
    if not rows:
        print("no usable step rows in %s (layers=%d, max_ntok=%d)"
              % (args.log, args.layers, args.max_ntok), file=sys.stderr)
        return 2
    res = windowed(rows, reps=args.reps, warm_steps=args.warm_steps)
    print("log: %s" % args.log)
    print("verify steps: %d  (layers=%s, max_ntok=%d, mean_ntok=%.2f)\n"
          % (len(rows), args.layers or "any", args.max_ntok, res["all"]["mean_ntok"]))
    report(res)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
