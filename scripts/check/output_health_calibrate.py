#!/usr/bin/env python3
"""[CGC 2026-10-06] 輸出健康度的**標定台**：把門檻的來歷做成可重跑的證據。

跑法（預設吃 2026-10-06 的 19 臂真文字 log）：
    python3 scripts/check/output_health_calibrate.py [glob …]

它做三件事：
  1. 逐臂印 run_max / selfsim / rep8 與判詞（門檻可對帳）。
  2. 檢查 detok **人眼**標籤與機器判詞是否一致（那 6 臂有人眼見證，是 ground truth）。
  3. 檢查家族分離：W1 real 全 OK、W2 real 全 DEGENERATE（≤ 2026-10-06 的 19 臂）。

⚠ 這支是**標定證據**，不是判詞；它不發車、不吃 GPU。門檻在 output_health.py 頂端。
"""
from __future__ import annotations

import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import output_health as oh   # noqa: E402

DEFAULT_GLOB = os.path.expanduser("~/Documents/flashkv-devserver/Backup/spec_tree/t1w2_realp_*.stderr.log")

# 人眼 ground truth（讀 detok 出來的文字；見 Backup/spec_tree/t1w2_realp_detok.json
#   與 T1W2_REALPROMPT_2026-10-06.md §1 的 detok 見證）：
#   w1_d512_real  = 連貫中文（"我始终坚持以三严三实为准则…"）                ⇒ OK
#   w1_d512_rand  = 阿拉伯字母 + 90 個連字（"تصدياااا…"）                   ⇒ DEGENERATE
#   w2_d512_real  = "在中在中在中…"（~50 次）                                ⇒ DEGENERATE
#   w2_d512_rand  = "ننننن…"（~116 個）                                      ⇒ DEGENERATE
#   w1_pg512_real = 連貫中文，但整段重複一次（句子級迴響，非 token 迴圈）      ⇒ OK（本指標不閘這一類）
#   w2_pg512_real = "，，，，…" 與 "醫學，，" 迴圈                            ⇒ DEGENERATE
WITNESS = {"w1_d512_real": "OK", "w1_d512_rand": "DEGENERATE",
           "w2_d512_real": "DEGENERATE", "w2_d512_rand": "DEGENERATE",
           "w1_pg512_real": "OK", "w2_pg512_real": "DEGENERATE"}


def main(argv):
    pats = argv or [DEFAULT_GLOB]
    paths = []
    for p in pats:
        paths += sorted(glob.glob(p))
    if not paths:
        print("⛔ 找不到任何 log —— 標定台需要資料", file=sys.stderr)
        return 2

    rows = []
    for p in paths:
        tag = os.path.basename(p)
        for pre in ("t1w2_realp_",):
            if tag.startswith(pre):
                tag = tag[len(pre):]
        tag = tag.replace(".stderr.log", "")
        h = oh.health_from_stderr(open(p, errors="replace").read())
        rows.append((tag, h))

    print("=== 逐臂（門檻：run_max ≥ %d 或 selfsim ≥ %.2f ⇒ DEGENERATE）==="
          % (oh.RUN_MAX_MAX, oh.SELFSIM_MAX))
    print(f"{'arm':<20} {'state':>11} {'n':>4} {'run_max':>8} {'selfsim':>8} {'n*':>3} {'rep8':>6}")
    for tag, h in rows:
        m = h["metrics"]
        if m is None:
            print(f"{tag:<20} {h['state']:>11} {'—':>4} {'—':>8} {'—':>8} {'—':>3} {'—':>6}")
        else:
            print(f"{tag:<20} {h['state']:>11} {m['n']:>4} {m['run_max']:>8} "
                  f"{m['selfsim']:>8.3f} {m['selfsim_n']:>3} {m['rep_k']:>6.3f}")

    ok = True

    # (1) 人眼 ground truth 一致
    print("\n=== 1. 與 detok 人眼見證一致（6 臂）===")
    for tag, want in sorted(WITNESS.items()):
        got = next((h["state"] for t, h in rows if t == tag), "MISSING")
        good = (got == want)
        ok = ok and good
        print(f"  {'✅' if good else '⛔'} {tag:<20} 人眼 {want:<11} 機器 {got}")

    # (2) 家族分離（真文字家族；這才是打平 gate 用的填充）
    print("\n=== 2. 家族分離（real prompt）===")
    for fam, want in (("w1", "OK"), ("w2", "DEGENERATE")):
        sub = [(t, h["state"]) for t, h in rows
               if t.startswith(fam + "_") and "real" in t.split("_")]
        n_ok = sum(1 for _, s in sub if s == want)
        good = (len(sub) > 0 and n_ok == len(sub))
        ok = ok and good
        print(f"  {'✅' if good else '⛔'} {fam.upper()} real：{n_ok}/{len(sub)} 判為 {want}"
              f"  {['%s=%s' % (t, s) for t, s in sub]}")

    # (3) 門檻邊界（健康側與退化側各離門檻多遠）
    print("\n=== 3. 門檻邊界（real 家族；健康側要低、退化側要高）===")
    healthy = [h["metrics"]["selfsim"] for t, h in rows if "real" in t.split("_") and h["state"] == "OK"
               and h["metrics"]]
    degen = [h["metrics"]["selfsim"] for t, h in rows if "real" in t.split("_") and h["state"] == "DEGENERATE"
             and h["metrics"]]
    hr = [h["metrics"]["run_max"] for t, h in rows if "real" in t.split("_") and h["state"] == "OK"
          and h["metrics"]]
    dr = [h["metrics"]["run_max"] for t, h in rows if "real" in t.split("_") and h["state"] == "DEGENERATE"
          and h["metrics"]]
    if healthy and degen:
        print(f"  selfsim：健康 max {max(healthy):.3f} → 門檻 {oh.SELFSIM_MAX:.2f} "
              f"（{oh.SELFSIM_MAX/max(healthy):.1f}×）← → 退化 min {min(degen):.3f}"
              f"（{min(degen)/oh.SELFSIM_MAX:.1f}×）")
        print(f"  run_max：健康 max {max(hr)} → 門檻 {oh.RUN_MAX_MAX} "
              f"（{oh.RUN_MAX_MAX/max(hr):.1f}×）← → 退化 min {min(dr)}"
              f"（{min(dr)/oh.RUN_MAX_MAX:.1f}×）")

    print("\nCALIBRATION:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
