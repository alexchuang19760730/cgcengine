#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""window_joint_ev.py -- 「把 CPU 的 expert fill 藏進 GPU 陰影」能換多少 t/s（單 regime 定價）。

為什麼要有這支（2026-09-23 16:1x）
--------------------------------
`docs/PREBIND_STAGE0_RESULT_2026-09-23.md` 與 `docs/RHO_STAGE0_RESULT_2026-09-23.md`
手算了三個上界，其中最大的是「ρ＋prebind 疊加 ⇒ **16.76 t/s（+33.3%）
**」。那個數字看起來像是「兩個機制相加」，但它其實是**在同一個格點上同時取了三個軸的樂觀值**，
而其中一個軸（cb = 74.18 ms）是**從另一個 regime 的另一支 log 借來的** —— 本專案已經為同一類
錯誤寫過兩支工具（`joint_reconcile.py`、`cb_headroom_probe.py`）明令禁止。

這支把整個模型變成一條可跑、可自測的公式，讓下面這句話可以被驗而不是被相信：

  **「疊加」相對「只做 ρ」的邊際價值是 +0.1 ~ +0.3 t/s，不是 +4.2 t/s。**

模型（明確聲明是線性代理，和 `prebind_ev.py` 同一族）
------------------------------------------------
  step = 247.98 ms（交付錨點）            tok/step = 12.57 * 0.24798 = 3.117
  每層 IO 需求        F      = cb / 40                      （ms/layer）
  單機制可隱藏        moved  = min(lead, cov * F)           （視窗 vs 覆蓋，取小）
  雙機制（ρ 有限提前，prebind 提前一整步 ⇒ lead = ∞）：
      moved = min(lead_rho, cov_rho * F) + (cov_joint - cov_rho) * F
  step_new = 247.98 - (moved * 40 - insert)
  t/s      = tok_per_step / (step_new / 1000)

★ 三條是由數值工具（selftest 第 9/10/12 項）機械檢查的不變式，不是口號：

  1. **推導式建議 != 證等方法**：在 cb=42.04（本次同 log 口徑）下 F = 1.051 ms/層，
     比最悲觀的視窗估計 1.30 ms 還小 ⇒ **視窗永遠綁不住**，遮多少只由覆蓋率決定。
     「下一個實驗要量視窗」只有在 cb 偏大的那一口徑下才成立。
  2. **沒有提前量時 ρ 是淨虧**：lead = 0 ⇒ moved = 0 ⇒ step 反而 +4.76 ms（多插 GPU 節點）。
  3. **結構上限 = 把 cb 整段藏起來**：即使 cov=1、lead=∞、insert=0，最好也只是
     step − cb ⇒ t/s 上限完全由 cb 決定（cb=42.04 ⇒ +20.4%；cb=74.18 ⇒ +38.8%）。

用法:
    python3 window_joint_ev.py --self-test
    python3 window_joint_ev.py --grid          # cb x 視窗 的定價表 + Δ(疊加 − 只做 ρ)
    python3 window_joint_ev.py --cb 42.04 --cov 0.854 --lead 1.30
    python3 window_joint_ev.py --cb 74.18 --cov 0.854 --joint 0.90 --lead 1.59
"""

from __future__ import annotations

import argparse
import math
import sys

# ---------------------------------------------------------------- 實測錨點
STEP_BASE_MS = 247.98    # 交付 decode step（prod_profile.py / llama-bench）
ANCHOR_TPS = 12.57       # 交付 decode
N_LAYERS = 40            # 可路由層
INSERT_RHO_MS = 4.76     # ρ 每一輪多插的 gate + topk GPU 成本（估）
INSERT_PREBIND_MS = 0.0  # prebind 不插任何 GPU 節點

# 同一個 `cb`，本專案在不同 regime 下量到過的全部數值。**每一個都帶它的來源。**
# 引用規則（見 joint_reconcile.py）：同一個句子的所有分量必須出自同一支 log。
#
# [2026-09-23 18:0x 定讞] **42 vs 74 不是兩個 regime，是同一支 run 的兩個窗口。**
# 在交付 cell 本體上開 `CGC_DECODE_PROFILE=1` 實跑兩趟（`cb_delivery_read.py`）：
#     穩態（對齊 --warm-skip 64）ntok=4  cb median 42.09 / 42.24   mean 50.33 / 51.50
#     含池預熱                            cb mean   47.84 / 49.62（rep1 全部 61.84 / 70.59）
# ⇒ **42.04 = 穩態 median**（交付錨點 12.57 用 --warm-skip，時鐘從預熱之後才開始 ⇒ 就是它）；
#   **74.18 = 含預熱／高 swap 那一支的數**，交付 regime 用不上。
# ⇒ 定價改用 **50.40**（穩態 mean，Σcb 才是吞吐相關的那個量），42.04 留下來當保守下界。
MEASURED_CB = [
    (11.26, "214450 log，startup free 84%"),
    (21.23, "E2b 暖，mean_len 2.40（另一條線）"),
    (42.04, "★ 交付 cell 穩態 median（ntok=4），2026-09-23 兩趟實測 42.09/42.24"),
    (50.40, "★ 交付 cell 穩態 mean（ntok=4），2026-09-23 兩趟實測 50.33/51.50"),
    (74.18, "023021 log，high swap + 含池預熱（交付 regime 用不到）"),
]
CB_DELIVERY = 50.40      # 定價用這一個（Σcb 決定吞吐）
CB_DELIVERY_LO = 42.04   # 保守下界 = median

COV_RHO = 0.8542       # ρ 實測 cov_uni（ntok=4）
COV_PREBIND_Q3 = 0.7257  # prebind qu3 實測（ntok=4）
JOINT_ASSUMED = 0.90   # ⚠ 從未實測：兩個預測源的聯集覆蓋率


# ---------------------------------------------------------------- 核心
def tokens_per_step() -> float:
    """錨點反推出來的第一步所產生的 token 數（12.57 * 0.24798 = 3.117）。"""
    return ANCHOR_TPS * STEP_BASE_MS / 1000.0


def tps_of(saving_ms: float) -> float:
    """藏掉 saving_ms 的 CPU 工作之後的 decode t/s。saving 為負 = 變慢。"""
    new_step = STEP_BASE_MS - saving_ms
    if new_step <= 0.0:
        return float("inf")
    return tokens_per_step() / (new_step / 1000.0)


def moved_per_layer(cb_ms: float, cov: float, lead_ms=None) -> float:
    """每層能被藏進 GPU 陰影的毫秒數 = min(提前量, 覆蓋到的 IO 量)。

    lead_ms = None 代表提前量不受限（prebind：提前一整步 ≈ 248 ms）。
    """
    per_layer_io = cb_ms / N_LAYERS
    covered = cov * per_layer_io
    if lead_ms is None:
        return covered
    return min(covered, lead_ms)


def evaluate(cb_ms: float, cov: float, lead_ms=None, insert_ms: float = 0.0,
             joint: float = None, cov_first: float = None) -> dict:
    """單機制（joint=None）或雙機制（joint 給定）的完整定價。

    雙機制：第一個機制（ρ）只有有限提前量 `lead_ms`；第二個機制（prebind，提前一整步）
    只貢獻「聯集相對第一機制的增量」覆蓋。
    """
    per_layer_io = cb_ms / N_LAYERS
    if joint is None:
        moved = moved_per_layer(cb_ms, cov, lead_ms)
    else:
        cov_first = COV_RHO if cov_first is None else cov_first
        if joint < cov_first - 1e-12:
            raise ValueError("joint coverage must be >= the first mechanism's")
        inc = max(0.0, joint - cov_first)
        moved = min(moved_per_layer(cb_ms, cov_first, lead_ms), per_layer_io)
        moved = min(moved + inc * per_layer_io, per_layer_io)
    saving = moved * N_LAYERS - insert_ms
    tps = tps_of(saving)
    return {
        "cb_ms": cb_ms, "cov": cov, "lead_ms": lead_ms, "insert_ms": insert_ms,
        "joint": joint, "moved_ms_per_layer": moved,
        "saving_ms": saving, "step_ms": STEP_BASE_MS - saving, "tps": tps,
        "d_tps_pct": 100.0 * (tps / ANCHOR_TPS - 1.0),
    }


def window_binds(cb_ms: float, cov: float, lead_ms: float) -> bool:
    """這個格點上，綁住結果的是「視窗」還是「覆蓋率」？"""
    return cov * (cb_ms / N_LAYERS) >= lead_ms


# ---------------------------------------------------------------- 報表
def do_grid() -> int:
    tok = tokens_per_step()
    print("錨點：step {:.2f} ms @ {:.2f} t/s ⇒ tok/step = {:.4f}；每層 {} 個可路由層".format(
        STEP_BASE_MS, ANCHOR_TPS, tok, N_LAYERS))
    print("查詢組合：ρ cov={:.4f}（實測）、prebind qu3 cov={:.4f}（實測）、joint={:.2f}（⚠未實測）".format(
        COV_RHO, COV_PREBIND_Q3, JOINT_ASSUMED))
    print()
    print("= A. 結構上限：把 cb 整段藏起來（cov=1, lead=∞, insert=0）=")
    for cb, src in MEASURED_CB:
        r = evaluate(cb, 1.0, None, 0.0)
        print("  cb={:6.2f} ms ({:<42}) -> {:6.2f} t/s ({:+6.1f}%)".format(
            cb, src, r["tps"], r["d_tps_pct"]))
    print()
    for lead in (1.30, 1.59):
        print("= B. lead = {:.2f} ms ==".format(lead))
        print(("  {:>6} | {:>18} | {:>18} | {:>18} | {:>9} | {}").format(
            "cb", "rho only ({:.2f})".format(lead), "rho only (1.59 固定)",
            "stacked joint=0.90", "Δ t/s", "綁住的是"))
        for cb, _src in MEASURED_CB:
            r30 = evaluate(cb, COV_RHO, lead, INSERT_RHO_MS)
            r59 = evaluate(cb, COV_RHO, 1.59, INSERT_RHO_MS)
            rj = evaluate(cb, COV_RHO, lead, INSERT_RHO_MS, joint=JOINT_ASSUMED)
            binder = "視窗" if window_binds(cb, COV_RHO, lead) else "覆蓋率"
            print("  {:6.2f} | {:>18} | {:>18} | {:>18} | {:>+9.2f} | {}".format(
                cb,
                "{:6.2f} ({:+5.1f}%)".format(r30["tps"], r30["d_tps_pct"]),
                "{:6.2f} ({:+5.1f}%)".format(r59["tps"], r59["d_tps_pct"]),
                "{:6.2f} ({:+5.1f}%)".format(rj["tps"], rj["d_tps_pct"]),
                rj["tps"] - r30["tps"], binder))
        print()
    print("= C. prebind qu3（提前一整步 ⇒ lead = ∞，不受視窗限制）=")
    for cb, _src in MEASURED_CB:
        r = evaluate(cb, COV_PREBIND_Q3, None, INSERT_PREBIND_MS)
        print("  cb={:6.2f} ms -> {:6.2f} t/s ({:+6.1f}%)".format(
            cb, r["tps"], r["d_tps_pct"]))
    print()
    print("= D. ⚠ 若影子分支拿不到任何提前量（lead = 0）=")
    for cb, _src in MEASURED_CB:
        r = evaluate(cb, COV_RHO, 0.0, INSERT_RHO_MS)
        print("  cb={:6.2f} ms -> rho {:6.2f} t/s ({:+6.1f}%)   <= 淨虧".format(
            cb, r["tps"], r["d_tps_pct"]))
    return 0


# ---------------------------------------------------------------- selftest
def _approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def self_test() -> int:
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # 1. tok/step 反推自洽：12.57 t/s @ 247.98 ms
    check("tokens per step", _approx(tokens_per_step(), 3.1169, 5e-3))
    check("tps round-trip", _approx(tps_of(0.0), ANCHOR_TPS, 1e-9))

    # 2-8. 重現兩個既有文件裡發表過的每一個數字（容差 0.03 t/s，對上出版表的取整差異）
    pub = [
        ("rho cb=42.04 lead=1.30", dict(cb_ms=42.04, cov=COV_RHO, lead_ms=1.30,
                                        insert_ms=INSERT_RHO_MS), 14.38),
        ("rho cb=74.18 lead=1.30", dict(cb_ms=74.18, cov=COV_RHO, lead_ms=1.30,
                                        insert_ms=INSERT_RHO_MS), 15.53),
        ("rho cb=74.18 lead=1.59", dict(cb_ms=74.18, cov=COV_RHO, lead_ms=1.59,
                                        insert_ms=INSERT_RHO_MS), 16.46),
        ("prebind cb=42.04", dict(cb_ms=42.04, cov=COV_PREBIND_Q3, lead_ms=None,
                                  insert_ms=INSERT_PREBIND_MS), 14.33),
        ("prebind cb=74.18", dict(cb_ms=74.18, cov=COV_PREBIND_Q3, lead_ms=None,
                                  insert_ms=INSERT_PREBIND_MS), 16.06),
        ("stacked cb=42.04", dict(cb_ms=42.04, cov=COV_RHO, lead_ms=1.30,
                                  insert_ms=INSERT_RHO_MS, joint=JOINT_ASSUMED), 14.50),
        ("stacked cb=74.18 lead=1.59", dict(cb_ms=74.18, cov=COV_RHO, lead_ms=1.59,
                                            insert_ms=INSERT_RHO_MS, joint=JOINT_ASSUMED), 16.76),
    ]
    for name, kw, want in pub:
        got = evaluate(**kw)["tps"]
        check("reproduce {}".format(name), _approx(got, want, 0.03))

    # 9. ★★ 本支存在的命題：「疊加」相對「只做 ρ」的邊際價值，在所有實測 cb x 視窗下
    #     都只有 +0.1 ~ +0.3 t/s —— 不是那個 +4.19 t/s 的差。
    for cb, _src in MEASURED_CB:
        for lead in (1.30, 1.59, None):
            r_alone = evaluate(cb, COV_RHO, lead, INSERT_RHO_MS)
            r_joint = evaluate(cb, COV_RHO, lead, INSERT_RHO_MS, joint=JOINT_ASSUMED)
            d = r_joint["tps"] - r_alone["tps"]
            check("stacking delta <= 0.31 (cb={}, lead={})".format(cb, lead),
                  0.0 <= d <= 0.31)

    # 10. ★ 交付口徑的 cb 下，連最悲觀的視窗都比每層 IO 需求大 ⇒ 視窗綁不住。
    #     實測 lead（EARLY 臂）是 1.31~2.27 ms；兩個 cb 都必須過這一條。
    for cb in (CB_DELIVERY_LO, CB_DELIVERY):
        for lead in (1.30, 1.31, 1.59, 2.27):
            check("window cannot bind (cb={}, lead={})".format(cb, lead),
                  not window_binds(cb, COV_RHO, lead))
    #     反之 cb=74.18 下視窗綁得住（這才是「下一站是視窗」成立的前提）
    check("window binds at cb=74.18 lead=1.30", window_binds(74.18, COV_RHO, 1.30))

    # 10b. ★ 視窗既然綁不住，lead 在實測區間內換任何值都必須得到同一個 t/s。
    #      這條把「視窗不是主導項」從一句話變成一個會紅的檢查。
    for cb in (CB_DELIVERY_LO, CB_DELIVERY):
        tps_at_lead = [evaluate(cb, COV_RHO, lead, INSERT_RHO_MS)["tps"]
                       for lead in (1.31, 1.59, 2.27, 3.00)]
        check("lead-insensitive at cb={}".format(cb),
              max(tps_at_lead) - min(tps_at_lead) < 1e-9)

    # 10c. ★ 交付 cb 的定價結果落在 +14%~+19%，而不是 +24%~+31%。
    #      （+24~31% 是 cb=74.18 那一格才有的數；本輪實測已把它排除。）
    for cb, lo, hi in ((CB_DELIVERY_LO, 14.0, 15.5), (CB_DELIVERY, 17.5, 19.5)):
        r = evaluate(cb, COV_RHO, 1.31, INSERT_RHO_MS)
        check("delivery gain in [{},{}]% at cb={}".format(lo, hi, cb),
              lo <= r["d_tps_pct"] <= hi)

    seq = [moved_per_layer(74.18, COV_RHO, None if l is None else l)
           for l in (0.0, 0.5, 1.0, 1.5, 2.0, None)]
    check("lead monotone", all(b >= a - 1e-12 for a, b in zip(seq, seq[1:])))
    check("lead saturates at cov*F",
          _approx(seq[-1], COV_RHO * 74.18 / N_LAYERS, 1e-12))

    # 12. ★ 拿不到提前量時 ρ 是淨虧（多插 GPU 節點卻沒遮到任何東西）
    r0 = evaluate(74.18, COV_RHO, 0.0, INSERT_RHO_MS)
    check("zero lead -> rho is a net loss", r0["tps"] < ANCHOR_TPS)
    check("zero lead -> pays the insertion", _approx(
        r0["step_ms"] - STEP_BASE_MS, INSERT_RHO_MS, 1e-9))

    # 13. 結構上限 = step − cb（cov=1, lead=∞, insert=0）
    for cb, _src in MEASURED_CB:
        r = evaluate(cb, 1.0, None, 0.0)
        check("structural cap = step-cb (cb={})".format(cb),
              _approx(r["step_ms"], STEP_BASE_MS - cb, 1e-9))

    # 14. cb 越大越好，且不吃負值
    tps_by_cb = [evaluate(cb, COV_RHO, None, INSERT_RHO_MS)["tps"] for cb, _ in MEASURED_CB]
    check("cb monotone", all(b >= a - 1e-12 for a, b in zip(tps_by_cb, tps_by_cb[1:])))

    # 15. 雙機制不能超過「藏掉整段 cb」
    for cb, _src in MEASURED_CB:
        rj = evaluate(cb, COV_RHO, None, INSERT_RHO_MS, joint=1.0)
        check("joint <= full cb hide (cb={})".format(cb),
              rj["step_ms"] >= STEP_BASE_MS - cb - 1e-9)

    # 16. 錯值拋出
    try:
        evaluate(42.04, COV_RHO, 1.30, INSERT_RHO_MS, joint=0.5)
        check("joint < first raises", False)
    except ValueError:
        check("joint < first raises", True)

    # 17. 數字有限且可直接謝絕除零場景
    check("all finite", all(math.isfinite(v) for v in tps_by_cb))

    print("selftest: {}/{} passed".format(ok, ok + len(fail)))
    for f in fail:
        print("  FAIL: {}".format(f))
    return 0 if not fail else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true", help="跑內建自測")
    p.add_argument("--grid", action="store_true", help="印 cb x 視窗 定價表")
    p.add_argument("--cb", type=float, help="CB-based bound (ms)")
    p.add_argument("--cov", type=float, help="第一機制的覆蓋率")
    p.add_argument("--lead", type=float, default=None, help="提前量上限（ms/層，不給 = 不受限）")
    p.add_argument("--insert", type=float, default=INSERT_RHO_MS, help="插入成本 ms/步")
    p.add_argument("--joint", type=float, default=None, help="雙機制聯集覆蓋率")
    a = p.parse_args(argv)

    if a.self_test:
        return self_test()
    if a.grid:
        return do_grid()
    if a.cb is not None and a.cov is not None:
        r = evaluate(a.cb, a.cov, a.lead, a.insert, joint=a.joint)
        print("cb={cb_ms:.2f} cov={cov:.4f} lead={lead_ms} insert={insert_ms:.2f} "
              "joint={joint}".format(**r))
        print("  moved {:7.4f} ms/層 | saving {:7.2f} ms/步 | step {:7.2f} ms "
              "| {:6.2f} t/s ({:+.1f}%)".format(
                  r["moved_ms_per_layer"], r["saving_ms"], r["step_ms"],
                  r["tps"], r["d_tps_pct"]))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
