#!/usr/bin/env python3
"""miss_mask_check.py -- 驗證「GPU 端 miss mask」與「host 端 BATCHDBG」逐位對得上。

背景（2026-09-24，第 2 步）
--------------------------
單段提交（CGC_SEG_BATCH）把整張圖一次丟出去，路由在 GPU 上算，所以 host **不會**知道這一步
選了哪些 expert -- 這正是 prebind／預測路線被判死的原因（h=0.03）。第 2 步的解法是在圖裡加
一個 gather：

    vmask = get_rows(valid_table, ids_flat)      valid_table[e] = (slot_table[e] >= 0) ? 1 : 0

它跟既有的 slot gather 用同一個 index vector，所以「哪些 (layer, expert) 是佔位」在裝置上就
有了答案，而不需要 host 讀回 ids（= 不需要多一次同步）。

這個腳本回答的是「那個答案對不對」。基準是 host 自己的 `BATCHDBG` 行
（`llama-expert-cache.cpp`，`ensure_batch` 在 `table[e] < 0` 時印的 miss 列表）。兩邊讀的是
同一個陣列（`llama_expert_cache_slot_table()`），所以正確的實作應該**逐位相同**；任何差異都是
真訊號（prefetch 中途 publish、snapshot 時機、index vector 讀錯），不是噪音。

為什麼用「逐層序列」比對，而不是「逐步比對」
--------------------------------------------
第一版用「layer id 不再遞增 = 新 step」來切 step，實測是錯的：miss 數為 0 的 step 不會產生任何
行，於是兩個相鄰的 step 會被併成一個（若後者的首個 layer id 大於前者最後一個）。這會製造出
假的 LAYER_DIFF（實測 57/103 step 報錯，但逐層抽樣看每一層其實都對得上）。

逐層序列沒有這個問題：把同一層在所有 step 的 miss 集合排成序列，兩邊各自省略「該 step 該層
沒 miss」的情況，所以只要兩邊對每一步的判斷一致，序列就會等長且逐項相同。
兩邊長度不同時（BATCHDBG 含 prefill step、MISSMASK 只建在 decode 圖上）**從尾端對齊**。

用法
----
    miss_mask_check.py --self-test
    miss_mask_check.py --log <stderr.log> [--min-il 1] [--json out.json]
    miss_mask_check.py --cost-baseline-ms 82.90 --cost-mask-ms 83.05   # 成本門檻判決

判決規則（跑之前寫死，不在看過數據之後才決定）
---------------------------------------------
PASS 需要全部滿足：
  a. 可比的層數 >= MIN_LAYERS (30)     -- `CGC_S1_MIN_IL` 預設 1，所以 layer 0 不在 GPU 表路徑上
  b. 對齊的元素總數 >= MIN_PAIRS (200)
  c. 每一個對齊元素 (expert set) 完全相同 -- mismatch == 0
  d. 每元素 miss 數的平均絕對差 <= TOL_MEAN_ABS (0.0，即要求完全相等)

mismatch 分兩類，因為修法完全不同：
  * SET_DIFF     同一層同一個 step，兩邊 miss 的 expert 集合不同 -> index vector 或 valid_table 錯
  * LEN_DIFF     同一層兩邊序列長度不同（已從尾端對齊，只回報多出來的筆數）

成本門檻：mask 的代價 <= 0.2 ms/step（`--cost-*`）。
"""

import argparse
import json
import re
import sys
from statistics import mean

# --- 判決門檻（寫死，勿在讀到數據後改） -------------------------------------------------
MIN_LAYERS = 30             # 可比的層數下限（`CGC_S1_MIN_IL=1` => 39 層）
MIN_PAIRS = 200             # 對齊元素總數下限
TOL_MEAN_ABS = 0.0          # 每元素 miss 數的平均絕對差：0 = 要求完全逐位相同
COST_BUDGET_MS = 0.2        # mask 的代價預算：<= 0.2 ms/step

# --- 解析 -------------------------------------------------------------------------------
RE_BATCHDBG = re.compile(r"^BATCHDBG layer=(\d+) misses=(\d+) slots:(.*)$")
RE_BATCH_E = re.compile(r"\se(\d+)->s(-?\d+)")
RE_MISSMASK = re.compile(r"^MISSMASK il=(\d+) step=(\d+) nsel=(\d+) misses=(\d+) exps:(.*)$")
RE_MASKSTEP = re.compile(r"^CGC-MISSMASK-STEP: step=(\d+) misses=(\d+) layers=(\d+)$")
RE_DECPROF = re.compile(r"CGC-DECPROF: step=(\d+).*?ntok=(\d+)")


def parse_log(text):
    """回傳 dict: batchdbg=[(il, frozenset)], missmask=[(il, frozenset)], masksteps, decprof."""
    out = {"batchdbg": [], "missmask": [], "masksteps": [], "decprof": []}
    for line in text.splitlines():
        line = line.strip()
        m = RE_BATCHDBG.match(line)
        if m:
            exps = frozenset(int(e) for e, _ in RE_BATCH_E.findall(m.group(3)))
            out["batchdbg"].append((int(m.group(1)), exps))
            continue
        m = RE_MISSMASK.match(line)
        if m:
            exps = frozenset(int(x) for x in m.group(5).split())
            out["missmask"].append((int(m.group(1)), exps))
            continue
        m = RE_MASKSTEP.match(line)
        if m:
            out["masksteps"].append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
            continue
        m = RE_DECPROF.search(line)
        if m:
            out["decprof"].append((int(m.group(1)), int(m.group(2))))
    return out


def per_layer(rows):
    d = {}
    for il, exps in rows:
        d.setdefault(il, []).append(exps)
    return d


def analyse(parsed, min_il=1):
    a = per_layer(parsed["batchdbg"])
    b = per_layer(parsed["missmask"])
    layers = sorted((set(a) & set(b)) - {l for l in set(a) | set(b) if l < min_il})

    kinds = {}
    samples = []
    absdiffs = []
    pairs = 0
    len_diffs = []
    per_layer_n = {}

    for il in layers:
        sa, sb = a[il], b[il]
        k = min(len(sa), len(sb))
        per_layer_n[il] = {"batchdbg": len(sa), "missmask": len(sb), "aligned": k}
        if len(sa) != len(sb):
            kinds["LEN_DIFF"] = kinds.get("LEN_DIFF", 0) + 1
            len_diffs.append({"il": il, "batchdbg": len(sa), "missmask": len(sb)})
        ta = sa[len(sa) - k:] if k else []
        tb = sb[len(sb) - k:] if k else []
        for i, (ea, eb) in enumerate(zip(ta, tb)):
            pairs += 1
            absdiffs.append(abs(len(ea) - len(eb)))
            if ea != eb:
                kinds["SET_DIFF"] = kinds.get("SET_DIFF", 0) + 1
                if len(samples) < 5:
                    samples.append({"il": il, "tail_index": i,
                                    "only_batchdbg": sorted(ea - eb)[:8],
                                    "only_missmask": sorted(eb - ea)[:8]})

    n_mismatch = sum(v for kk, v in kinds.items() if kk == "SET_DIFF")
    mean_absdiff = mean(absdiffs) if absdiffs else None

    reasons = []
    if len(layers) < MIN_LAYERS:
        reasons.append("comparable layers %d < MIN_LAYERS %d" % (len(layers), MIN_LAYERS))
    if pairs < MIN_PAIRS:
        reasons.append("aligned pairs %d < MIN_PAIRS %d" % (pairs, MIN_PAIRS))
    if n_mismatch > 0:
        reasons.append("%d/%d aligned elements differ (%s)" % (n_mismatch, pairs, kinds))
    if mean_absdiff is not None and mean_absdiff > TOL_MEAN_ABS:
        reasons.append("mean |miss count diff| %.4f > tol %.1f" % (mean_absdiff, TOL_MEAN_ABS))

    return {
        "n_layers_compared": len(layers),
        "layers_only_in_batchdbg": sorted(set(a) - set(b)),
        "layers_only_in_missmask": sorted(set(b) - set(a)),
        "aligned_pairs": pairs,
        "n_set_diff": n_mismatch,
        "n_len_diff": kinds.get("LEN_DIFF", 0),
        "mean_abs_missdiff": mean_absdiff,
        "len_diff_samples": len_diffs[:5],
        "set_diff_samples": samples,
        "verdict": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        # 參考量
        "batchdbg_rows": len(parsed["batchdbg"]),
        "missmask_rows": len(parsed["missmask"]),
        "missmask_steps": len(parsed["masksteps"]),
        "batchdbg_misses_per_row": round(mean([len(e) for _, e in parsed["batchdbg"]]), 3) if parsed["batchdbg"] else None,
        "missmask_misses_per_row": round(mean([len(e) for _, e in parsed["missmask"]]), 3) if parsed["missmask"] else None,
        "per_layer_n": per_layer_n,
    }


def cost_verdict(base_ms, mask_ms):
    """成本門檻：mask 加了多少 ms/step。"""
    if base_ms is None or mask_ms is None:
        return {"verdict": "SKIP", "delta_ms": None, "reason": "缺少一個臂"}
    d = mask_ms - base_ms
    return {
        "delta_ms": round(d, 3),
        "budget_ms": COST_BUDGET_MS,
        "rel_pct": round(100.0 * d / base_ms, 3) if base_ms else None,
        "verdict": "PASS" if d <= COST_BUDGET_MS else "FAIL",
    }


# --- selftest ----------------------------------------------------------------------------
def _mk_batchdbg(seqs):
    """seqs: {il: [frozenset, ...]} -> 依 step 順序交錯輸出（模擬 hook 的逐層順序）。"""
    n = max((len(v) for v in seqs.values()), default=0)
    out = []
    for i in range(n):
        for il in sorted(seqs):
            if i < len(seqs[il]):
                exps = seqs[il][i]
                out.append("BATCHDBG layer=%d misses=%d slots:%s" %
                           (il, len(exps), "".join(" e%d->s%d" % (e, e % 143) for e in sorted(exps))))
    return out


def _mk_missmask(seqs, skip_layers=()):
    n = max((len(v) for v in seqs.values()), default=0)
    out = []
    for s in range(1, n + 1):
        tot = 0
        for il in sorted(seqs):
            if il in skip_layers or (s - 1) >= len(seqs[il]):
                continue
            exps = seqs[il][s - 1]
            out.append("MISSMASK il=%d step=%d nsel=%d misses=%d exps:%s" %
                       (il, s, 8, len(exps), "".join(" %d" % e for e in sorted(exps))))
            tot += len(exps)
        out.append("CGC-MISSMASK-STEP: step=%d misses=%d layers=%d" % (s, tot, 39))
    return out


def _seqs(n_steps=40, layers=range(1, 40)):
    return {il: [frozenset({(il + 3 * s) % 256, (il * 7 + s) % 256}) for s in range(n_steps)]
            for il in layers}


def self_test():
    ok = 0
    bad = []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            bad.append(name)

    # 1. 完全逐位相同 -> PASS
    s = _seqs()
    r = analyse(parse_log("\n".join(_mk_batchdbg(s) + _mk_missmask(s))))
    chk("identical -> PASS", r["verdict"] == "PASS")
    chk("identical: 0 set_diff", r["n_set_diff"] == 0)
    chk("identical: 39 layers", r["n_layers_compared"] == 39)
    chk("identical: 39*40 pairs", r["aligned_pairs"] == 39 * 40)

    # 2. 其中一層一個 step 的 expert set 差一個 -> FAIL + SET_DIFF
    s2 = {k: list(v) for k, v in _seqs().items()}
    s2[5][10] = frozenset({200, 201})
    r2 = analyse(parse_log("\n".join(_mk_batchdbg(_seqs()) + _mk_missmask(s2))))
    chk("one set differs -> FAIL", r2["verdict"] == "FAIL")
    chk("classified SET_DIFF", r2["n_set_diff"] == 1)

    # 3. BATCHDBG 多 3 個 prefill step（序列較長）-> 尾端對齊後仍 PASS，但回報 LEN_DIFF
    s3 = {k: [frozenset({1, 2})] * 3 + list(v) for k, v in _seqs().items()}
    r3 = analyse(parse_log("\n".join(_mk_batchdbg(s3) + _mk_missmask(_seqs()))))
    chk("leading prefill rows: still PASS", r3["verdict"] == "PASS")
    chk("leading prefill rows: LEN_DIFF reported", r3["n_len_diff"] == 39)

    # 4. 其中一層完全沒有 MISSMASK -> 該層不可比，層數不足 -> FAIL
    s4 = _seqs()
    txt4 = "\n".join(_mk_batchdbg(s4) + _mk_missmask(s4, skip_layers=(7,)))
    r4 = analyse(parse_log(txt4))
    chk("missing layer: 38 compared", r4["n_layers_compared"] == 38)
    chk("missing layer: reported", r4["layers_only_in_batchdbg"] == [7])

    # 5. 空 log -> FAIL 且不崩
    r5 = analyse(parse_log(""))
    chk("empty -> FAIL", r5["verdict"] == "FAIL")
    chk("empty: 0 pairs", r5["aligned_pairs"] == 0)
    chk("empty: mean None", r5["mean_abs_missdiff"] is None)

    # 6. min_il 過濾：layer 0 只出現在 BATCHDBG 時不該被算進可比層
    txt6 = "\n".join(["BATCHDBG layer=0 misses=1 slots: e5->s5"] * 40 + _mk_missmask(_seqs()))
    r6 = analyse(parse_log(txt6))
    chk("layer0 only in batchdbg: excluded", r6["n_layers_compared"] == 0)
    chk("layer0: listed as only_in_batchdbg", r6["layers_only_in_batchdbg"] == [0])

    # 7. 解析欄位正確
    p = parse_log("\n".join(_mk_batchdbg({1: [frozenset({3, 7})]}) + _mk_missmask({1: [frozenset({3, 7})]})))
    chk("parse batchdbg", p["batchdbg"] == [(1, frozenset({3, 7}))])
    chk("parse missmask", p["missmask"] == [(1, frozenset({3, 7}))])
    chk("parse masksteps", p["masksteps"] == [(1, 2, 39)])

    # 8. 成本門檻
    chk("cost 0.15 -> PASS", cost_verdict(82.90, 83.05)["verdict"] == "PASS")
    chk("cost 0.30 -> FAIL", cost_verdict(82.90, 83.20)["verdict"] == "FAIL")
    chk("cost negative -> PASS", cost_verdict(82.90, 82.70)["verdict"] == "PASS")
    chk("cost missing -> SKIP", cost_verdict(None, 83.0)["verdict"] == "SKIP")

    print("selftest: %d/%d passed" % (ok, ok + len(bad)))
    for b in bad:
        print("  FAILED: %s" % b)
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", help="llama-bench 的 stderr log")
    ap.add_argument("--json", help="把判決寫成 json")
    ap.add_argument("--min-il", type=int, default=1,
                    help="GPU slot-table 路徑的最小層（CGC_S1_MIN_IL，預設 1）")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--cost-baseline-ms", type=float, help="無 mask 的 ms/step")
    ap.add_argument("--cost-mask-ms", type=float, help="有 mask 的 ms/step")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.log and a.cost_baseline_ms is None:
        ap.error("需要 --log（或 --self-test）")

    res = {}
    if a.log:
        res = analyse(parse_log(open(a.log, errors="replace").read()), min_il=a.min_il)
    if a.cost_baseline_ms is not None or a.cost_mask_ms is not None:
        res["cost"] = cost_verdict(a.cost_baseline_ms, a.cost_mask_ms)

    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    if a.json and a.log:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    return 0 if res.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
