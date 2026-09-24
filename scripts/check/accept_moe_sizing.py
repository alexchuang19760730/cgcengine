#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
accept_moe_sizing.py — AcceptMoE (arXiv 2608.02989) 移植到本專案的可行性定價器。

## 它回答什麼

AcceptMoE 的核心主張：在 offloading 下，**不要**預測自然路由再預取，
而是把「專家資格」限制在**已駐留**的集合上（verifier-side、commitment-weighted、
self-sizing）。它宣稱 offloading 下 2.06×、流量 −73.6~77.1%、精度只掉 0.27pp。

移植與否取決於兩個數，本工具把它們從既有量測資產裡算出來（0 GPU）：

  1. **駐留覆蓋率 cur** = 一次 (token, expert) 選擇命中駐留 slot 的比例。
     ⇒ 若把資格限制成「只准駐留」，被丟掉的訪問比例 = 1 − cur。這是精度代價的代理。
  2. **容量上限 kN** = 靜態駐留「頻次最高的 N 個專家」能覆蓋的訪問比例。
     ⇒ cur 與 k143 的差 = **替換策略的浪費**（容量已經夠，但沒被用滿）。

AcceptMoE 在「駐留集合已經覆蓋絕大部分路由」的前提下才划算；本工具就是要驗這個前提。

## 口徑：已於 2026-09-24 讀碼定案（0 GPU）

`llama-expert-cache.cpp` `masscov_record()` :2041-2070 與 dump :3084-3105：

    total / cur / k96 / k128 / k143 / k192   → **MASS**（路由權重 w 的累計比例）
    selcold / sel                            → **COUNT**（逐次選擇，含 4 token 選同一專家的重複）

⇒ **同一行檔裡混了兩個口徑**，極易混用。（檔頭註釋自己也矛盾：:2062 寫
"count ratio, not mass"、:3025 寫 "MASS, not counts"。）

第三個口徑：llama-bench 的 `hit%` = `n_hits / n_requests`，`n_requests` 在
`ensure_batch` 是 `+= n`（:1064）、`ensure_slot` 是 `++`（:877）
⇒ **逐「專家請求」**，與 `selcold` 的逐次選擇分母不同。

實測：base 三者（mass 65.85 / count 65.84 / hit% 67.9~68.0）差 **<2.2pp ⇒ 對 base 一致**；
pin 則 masscov 89.3~89.4% vs hit% 73~75%，**差 ~15pp**，方向符合「冷專家多半只被 1 token 選中」
（理論比 19.36/32=0.605，實測 10.64/25=0.43，同階）。

⚠ **`kN` 是 in-sample oracle 上界**：它在同一批資料裡取累計質量前 N 名（:3092-3105），
而這批資料與 PIN_PROFILE 的 profile 都出自 `--prompt 0`（`rho_fill_ab.sh:34`、
`pin_abba.sh:9`）⇒ `k96=95.65%`／`k143=98.88%` **不是線上可達值**，只是上界。
本工具的 `capacity_ok` 只回答「容量夠不夠」，**不代表修得好**。

## 用法

    python3 accept_moe_sizing.py --self-test
    python3 accept_moe_sizing.py --masscov scripts/check/pin_profiles/masscov_base_2026-09-23.txt
    python3 accept_moe_sizing.py --compare A.txt B.txt
    python3 accept_moe_sizing.py --overlap routeA.txt routeB.txt --n 143
"""

import argparse
import os
import re
import statistics
import sys

# ---------------------------------------------------------------- 錨點常數
# 交付尺子（prod25-stream）：見 MEMORY_PERF.md「交付 decode 權威讀數」。
ANCHOR = {
    "ts": 12.57,            # 交付 decode t/s
    "ms_per_round": 247.98, # 每回合牆鐘
    "tok_per_round": 3.117, # MTP on，每回合產出 token 數
    "ms_per_step": 161.0,   # DECPROF per-step 總時
    "ms_cb": 23.8,          # DECPROF per-step 中 expert cache 填池
    "steps_per_round": 1.55,
}
# AcceptMoE 論文宣稱（arXiv 2608.02989 摘要，2026-09-24 覆核）
ACCEPTMOE = {
    "speedup_offload": 2.06,      # 物理 expert offloading 下 vs EAGLE-3 自然路由
    "speedup_allincore": 1.290,   # 全部專家在 GPU 記憶體時
    "traffic_cut_lo": 0.736,
    "traffic_cut_hi": 0.771,
    "acc_drop_pp": 0.27,          # 12 model-task pairs 的平均精度損失（pp）
}

_MASSCOV_RE = re.compile(r"(\w+)=([0-9.]+)")


# ---------------------------------------------------------------- 解析
def parse_masscov(path):
    """解析 masscov 檔 → list[dict]。每行一層：
       layer L total=.. cur=.. selcold=.. sel=.. k96=.. k128=.. k143=.. k192=.."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            d = {}
            for k, v in _MASSCOV_RE.findall(ln):
                d[k] = float(v)
            if "cur" not in d:
                continue
            m = re.search(r"layer\s+(\d+)", ln)
            d["_layer"] = int(m.group(1)) if m else len(rows)
            rows.append(d)
    if not rows:
        raise ValueError("no masscov rows parsed from %s" % path)
    return rows


def parse_route_dump(path):
    """解析 ROUTE_DUMP 的 route_topN 檔 → list[list[int]]（逐層、依頻次降冪的專家 id）。"""
    layers = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            layers.append([int(x) for x in ln.split()])
    if not layers:
        raise ValueError("no route rows parsed from %s" % path)
    return layers


# ---------------------------------------------------------------- 聚合
def aggregate(rows, keys=("cur", "selcold", "k96", "k128", "k143", "k192", "total")):
    """逐 key 回傳 dict(mean/min/max)，缺 key 則跳過。"""
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if not vals:
            continue
        out[k] = {"mean": statistics.mean(vals), "min": min(vals), "max": max(vals)}
    return out


def residency_gap(agg):
    """駐留覆蓋率 vs 靜態容量的落差。

    回傳 dict，含：
      cur           目前駐留集合覆蓋率（口徑見檔頭）
      k96/k128/k143 靜態駐留前 N 高頻專家的覆蓋率
      gap_pp        k143 − cur，單位 **百分點**（替換策略的浪費）
      capacity_ok   是否「容量已足」：k96 >= 0.95 表示 96 slot 就夠 95% 以上
    """
    cur = agg["cur"]["mean"]
    k143 = agg["k143"]["mean"]
    k96 = agg.get("k96", {}).get("mean")
    return {
        "cur": cur,
        "cold": 1.0 - cur,
        "k96": k96,
        "k128": agg.get("k128", {}).get("mean"),
        "k143": k143,
        "gap_pp": (k143 - cur) * 100.0,
        "capacity_ok": bool(k96 is not None and k96 >= 0.95),
    }


# ---------------------------------------------------------------- 速度模型
def speed_model(ms_cb_saved):
    """從 per-step 省下的 cb 毫秒數 → 新的 t/s 與相對錨點的增益%。

    橋：per-step 161.0 ms × steps/round 1.55 = 249.6 ms ≈ 錨點 247.98（差 0.6%）。
    ⇒ per-step 省 X ms ⇒ per-round 省 X × 1.55 ms。
    """
    a = ANCHOR
    new_step = a["ms_per_step"] - ms_cb_saved
    if new_step <= 0:
        raise ValueError("ms_cb_saved 超過 per-step 總時，模型失效")
    new_round = a["ms_per_round"] - ms_cb_saved * a["steps_per_round"]
    if new_round <= 0:
        raise ValueError("ms_cb_saved 過大，per-round 變負")
    new_ts = a["tok_per_round"] / (new_round / 1000.0)
    return {
        "ms_cb_saved": ms_cb_saved,
        "new_ms_per_step": new_step,
        "new_ms_per_round": new_round,
        "new_ts": new_ts,
        "gain_pct": (new_ts / a["ts"] - 1.0) * 100.0,
    }


def acceptmoe_ceiling(traffic_cut):
    """按「流量削減比例」定價：cb 中屬於填池的部分按比例消失。

    注意：本專案 cb 只佔 per-step 的 14.8%，而 AcceptMoE 拿到 2.06×
    是因為它的 baseline 是 transfer-dominated。⇒ 同樣削減 73.6% 流量，
    兩邊的收益完全不同量級。這個函式就是把這個差算出來。
    """
    saved = ANCHOR["ms_cb"] * traffic_cut
    r = speed_model(saved)
    r["traffic_cut"] = traffic_cut
    return r


# ---------------------------------------------------------------- route dump 重疊
def route_overlap(dump_a, dump_b, n):
    """兩個 route dump 的逐層 top-n 集合重疊率。

    用途：判斷「路由分佈是否跨 prompt 穩定」。若重疊率高 ⇒ 靜態／混合放置可行，
    駐留率可以從 66% 往上拉；若重疊率低 ⇒ 路由是 prompt-dependent，
    靜態放置與「以駐留為資格」兩條路都站不住。
    """
    A, B = parse_route_dump(dump_a), parse_route_dump(dump_b)
    if len(A) != len(B):
        raise ValueError("layer count differs: %d vs %d" % (len(A), len(B)))
    per_layer = []
    for a, b in zip(A, B):
        sa, sb = set(a[:n]), set(b[:n])
        inter = len(sa & sb)
        union = len(sa | sb) or 1
        per_layer.append({
            "jaccard": inter / union,
            "recall": inter / (len(sb) or 1),   # B 的 top-n 有多少落在 A 的 top-n
        })
    return {
        "n": n,
        "layers": len(per_layer),
        "jaccard_mean": statistics.mean(x["jaccard"] for x in per_layer),
        "jaccard_min": min(x["jaccard"] for x in per_layer),
        "recall_mean": statistics.mean(x["recall"] for x in per_layer),
        "recall_min": min(x["recall"] for x in per_layer),
        "per_layer": per_layer,
    }


# ---------------------------------------------------------------- 報表
def fmt_agg(tag, agg):
    lines = ["== %s ==" % tag]
    for k in ("cur", "selcold", "k96", "k128", "k143", "k192"):
        if k in agg:
            v = agg[k]
            lines.append("  %-8s mean=%.4f  min=%.4f  max=%.4f" % (k, v["mean"], v["min"], v["max"]))
    return "\n".join(lines)


def report(path):
    rows = parse_masscov(path)
    agg = aggregate(rows)
    gap = residency_gap(agg)
    out = [fmt_agg(path, agg), "", "駐留落差："]
    out.append("  cur = %.4f  (被丟掉的訪問 = %.4f)" % (gap["cur"], gap["cold"]))
    out.append("  k96 = %.4f   k128 = %.4f   k143 = %.4f" % (gap["k96"], gap["k128"], gap["k143"]))
    out.append("  gap = %.1f pp   （容量足以覆蓋 %.1f%%，實拿 %.1f%%）"
               % (gap["gap_pp"], gap["k143"] * 100, gap["cur"] * 100))
    out.append("  容量是否早已足夠：%s" % ("是（k96 ≥ 0.95）" if gap["capacity_ok"] else "否"))
    out.append("")
    out.append("若「資格 = 只准駐留」的代價與收益：")
    out.append("  精度代理（丟掉的訪問比例） = %.1f%%   對照 AcceptMoE 宣稱 0.27pp"
               % (gap["cold"] * 100))
    full = speed_model(ANCHOR["ms_cb"])
    out.append("  速度上限（cb 全部消失）     = %.2f t/s（%+0.1f%%）" % (full["new_ts"], full["gain_pct"]))
    for cut in (ACCEPTMOE["traffic_cut_lo"], ACCEPTMOE["traffic_cut_hi"]):
        r = acceptmoe_ceiling(cut)
        out.append("  流量削減 %.1f%%            → %.2f t/s（%+0.1f%%）"
                   % (cut * 100, r["new_ts"], r["gain_pct"]))
    return "\n".join(out)


# ---------------------------------------------------------------- selftest
def _write(tmp, name, text):
    p = os.path.join(tmp, name)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return p


def self_test():
    import tempfile
    ok = fail = 0

    def chk(name, cond, extra=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print("  PASS  %s" % name)
        else:
            fail += 1
            print("  FAIL  %s  %s" % (name, extra))

    print("[accept_moe_sizing] self-test")

    # 1) 解析：格式要能吃下真實那一行
    sample = ("layer 0 total=20.3717 cur=0.6068 selcold=0.3929 sel=5128 "
              "k143=0.9517 k96=0.8560 k128=0.9293 k192=0.9907 k256=1.0000\n"
              "layer 1 total=20.4172 cur=0.6157 selcold=0.3836 sel=5128 "
              "k143=0.9409 k96=0.8525 k128=0.9191 k192=0.9858 k256=1.0000\n")
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "m.txt", sample)
        rows = parse_masscov(p)
        chk("parse 2 layers", len(rows) == 2, "got %d" % len(rows))
        chk("layer index", rows[1]["_layer"] == 1, rows[1].get("_layer"))
        chk("cur value", abs(rows[0]["cur"] - 0.6068) < 1e-9)
        agg = aggregate(rows)
        chk("aggregate mean cur", abs(agg["cur"]["mean"] - (0.6068 + 0.6157) / 2) < 1e-9)
        # 2) cur + selcold 必須互補（同一個分母的兩個比例）
        for r in rows:
            chk("cur+selcold≈1", abs(r["cur"] + r["selcold"] - 1.0) < 0.002,
                "%.4f" % (r["cur"] + r["selcold"]))
        gap = residency_gap(agg)
        chk("gap_pp sign", gap["gap_pp"] > 0, "%.2f" % gap["gap_pp"])
        chk("capacity_ok False at k96=0.85", gap["capacity_ok"] is False)
        # 3) 空檔要拋錯而不是 silently 回傳 0
        try:
            parse_masscov(_write(td, "empty.txt", "\n"))
            chk("empty raises", False)
        except ValueError:
            chk("empty raises", True)

    # 4) 速度模型：橋必須對回錨點（省 0 ⇒ 回到 12.57）
    z = speed_model(0.0)
    chk("zero-save returns anchor", abs(z["new_ts"] - ANCHOR["ts"]) < 0.05,
        "%.3f vs %.2f" % (z["new_ts"], ANCHOR["ts"]))
    chk("bridge closes", abs(ANCHOR["ms_per_step"] * ANCHOR["steps_per_round"]
                             - ANCHOR["ms_per_round"]) / ANCHOR["ms_per_round"] < 0.01,
        "%.1f vs %.1f" % (ANCHOR["ms_per_step"] * ANCHOR["steps_per_round"], ANCHOR["ms_per_round"]))
    full = speed_model(ANCHOR["ms_cb"])
    chk("cb 全消失 < +20%", 10.0 < full["gain_pct"] < 20.0, "%.1f%%" % full["gain_pct"])
    chk("cb 全消失 > +10%", full["gain_pct"] > 10.0)
    cut = acceptmoe_ceiling(ACCEPTMOE["traffic_cut_hi"])
    chk("77.1% 流量削減 < 全消失", cut["new_ts"] < full["new_ts"])
    chk("遠低於 2.06×", cut["gain_pct"] < 30.0, "%.1f%%" % cut["gain_pct"])
    try:
        speed_model(ANCHOR["ms_per_step"] * 2)
        chk("over-save raises", False)
    except ValueError:
        chk("over-save raises", True)

    # 5) route dump 重疊
    with tempfile.TemporaryDirectory() as td:
        a = _write(td, "a.txt", "0 1 2 3 4 5\n10 11 12 13 14 15\n")
        # 兩層都刻意留 3 個交集：交集 3、聯集 9 ⇒ jaccard 1/3、recall 3/6
        b = _write(td, "b.txt", "0 1 2 9 8 7\n10 11 12 99 98 97\n")
        ov = route_overlap(a, b, 6)
        chk("overlap layers", ov["layers"] == 2)
        chk("jaccard 3/9", abs(ov["jaccard_mean"] - 3.0 / 9.0) < 1e-9,
            "%.4f" % ov["jaccard_mean"])
        chk("recall 3/6", abs(ov["recall_mean"] - 0.5) < 1e-9, "%.4f" % ov["recall_mean"])
        # 逐層都要一致（防止只有平均值對、單層壞掉）
        chk("jaccard per-layer uniform",
            all(abs(x["jaccard"] - 3.0 / 9.0) < 1e-9 for x in ov["per_layer"]))
        same = route_overlap(a, a, 6)
        chk("self overlap = 1", abs(same["jaccard_mean"] - 1.0) < 1e-9)
        try:
            route_overlap(a, _write(td, "c.txt", "0 1 2\n"), 6)
            chk("layer mismatch raises", False)
        except ValueError:
            chk("layer mismatch raises", True)

    print("[accept_moe_sizing] self-test: %d/%d passed" % (ok, ok + fail))
    return 0 if fail == 0 else 1


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="AcceptMoE 移植可行性定價器")
    ap.add_argument("--masscov", action="append", default=[],
                    help="masscov 檔，可重複（列聚合與駐留落差）")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None,
                    help="並排比較兩個 masscov（例如 base vs pin）")
    ap.add_argument("--overlap", nargs=2, metavar=("DUMPA", "DUMPB"), default=None,
                    help="兩個 route dump 的 top-n 重疊率")
    ap.add_argument("--n", type=int, default=143, help="overlap 的 top-n（預設 143）")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    did = False
    paths = list(args.masscov)
    if args.compare:
        paths = list(args.compare)
    for p in paths:
        print(report(p))
        print()
        did = True

    if args.compare:
        ra, rb = parse_masscov(args.compare[0]), parse_masscov(args.compare[1])
        ca, cb = residency_gap(aggregate(ra)), residency_gap(aggregate(rb))
        print("== 並排 ==")
        print("  cur      %.4f → %.4f   (%+.1f pp)" % (ca["cur"], cb["cur"],
                                                       (cb["cur"] - ca["cur"]) * 100))
        print("  cold     %.4f → %.4f" % (ca["cold"], cb["cold"]))
        print("  k143     %.4f → %.4f   （靜態容量上限，兩輪應幾乎相同）" % (ca["k143"], cb["k143"]))
        print("  gap      %.1f pp → %.1f pp" % (ca["gap_pp"], cb["gap_pp"]))
        did = True

    if args.overlap:
        ov = route_overlap(args.overlap[0], args.overlap[1], args.n)
        print("== route dump 重疊 (top-%d) ==" % ov["n"])
        print("  jaccard mean=%.4f min=%.4f" % (ov["jaccard_mean"], ov["jaccard_min"]))
        print("  recall  mean=%.4f min=%.4f" % (ov["recall_mean"], ov["recall_min"]))
        did = True

    if not did:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
