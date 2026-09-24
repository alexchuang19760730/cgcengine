#!/usr/bin/env python3
"""swap-miss 修復「三條判據」的驗收器（docs/SWAP_MISS_LINK_2026-09-24.md §6）。

把「dswap≈0 不累積 ＋ hit_pct 不降 ＋ capacity 佔比不升且 pread_usec 降」從口號變成
可機檢的判決。輸入是量測 log（stderr），輸出逐條 PASS/FAIL 與理由。

欄位來源（讀碼定案 2026-09-24，全部**無 env 門控**、跑在 destructor 裡）：
  A. `CGC-SHAPE v=1 phase=final ...`            llama-shape-knob.cpp:230-250
     欄位：req= hits= misses= hit_pct= compulsory= capacity= evict= read_mib=
           pread_us= fill_wait_us=            ← 這裡是 `evict=`（縮寫）
  B. `llama_expert_cache: final stats: ...`      llama-expert-cache.cpp:2553
     欄位：requests= ... (hit rate X%) ... pread_usec= fill_batch_usec= fill_wait_us=
     ⚠ 依賴乾淨關機：SIGINT 時 destructor 可能跑不到（同檔 :515-518 自己的註解）。
  C. `llama_expert_cache: miss attribution: ...` llama-expert-cache.cpp:2764（**不是 2765**）
     欄位：compulsory= capacity= (x% / y% of N)  evictions=   ← 這裡是 `evictions=`（全名）
  D. `CGC-RIG-SNAPSHOT ... pread_usec=`           llama-expert-cache.cpp:520
     ⚠ **有 env 門控**：要設 `CGC_PREFILL_PROTECT_FILE`（:535-537 沒設就直接 return），
       而且要檔案內容變動才印 ⇒ 這是 09-23 全部 log 零命中的原因。

★ 口徑陷阱（本工具因此不直接比「capacity 佔比」）：
  `ever_loaded[layer][expert]` 只在 init 清一次（:3738），跑途中永不重置 ⇒
  compulsory 的上界是 `n_layer × n_expert`（本模型 40×256 = 10240），跑到後面分子飽和、
  分母（總 miss）繼續長 ⇒ **capacity 佔比會隨 run 長度機械性上升**。
  所以「佔比不升」只有在各 run 的 `req` 同量級時可比；本工具用
  `capacity per 1k req`（= capacity*1000/req）當主判據，並在 req 差異 >10% 時警告。
  ⚠ `pread_us`／`pread_usec` 是**跨 worker thread 的加總**（llama-expert-cache.h:600），
  不是 wall clock ⇒ 比大小要在 WORKERS 相同的條件下（prod-new WORKERS=8）。

用法：
  python3 scripts/check/miss_attr_gate.py --row base=run1.log:1200:1300 --row fix=run2.log:1250:1260
                                          [--row fix2=run3.log:1240:1255]
  # 沒有 swap 數就省略：--row base=run1.log
exit code：0 = 三條全 PASS；1 = 有 FAIL
"""
from __future__ import annotations
import argparse, re, sys

RE_SHAPE = re.compile(
    r"CGC-SHAPE v=1 phase=final .*?"
    r"req=(?P<req>\d+) hits=(?P<hits>\d+) misses=(?P<misses>\d+) "
    r"hit_pct=(?P<hit_pct>[\d.]+) compulsory=(?P<compulsory>\d+) "
    r"capacity=(?P<capacity>\d+) evict=(?P<evict>\d+)")
RE_SHAPE_TAIL = re.compile(r"read_mib=(?P<read_mib>[\d.]+) pread_us=(?P<pread_us>\d+) "
                           r"fill_wait_us=(?P<fill_wait_us>\d+)")
RE_ATTR = re.compile(r"miss attribution: compulsory=(?P<compulsory>\d+) "
                     r"capacity=(?P<capacity>\d+) \((?P<cpct>[\d.]+)% / (?P<capct>[\d.]+)% "
                     r"of (?P<total>\d+)\)\s+evictions=(?P<evictions>\d+)")
RE_FINAL = re.compile(r"final stats: runtime requests=(?P<req>\d+) hits=(?P<hits>\d+) "
                      r"misses=(?P<misses>\d+) \(hit rate (?P<hit_pct>[\d.]+)%\)")
RE_FINAL_PREAD = re.compile(r"pread_usec=(?P<pread_usec>\d+)")


def parse_log(text: str) -> dict:
    """從一份 log 抓出判據需要的欄位；CGC-SHAPE 優先，缺什麼就用 B/C 補。"""
    out: dict = {}
    m = RE_SHAPE.search(text)
    if m:
        out.update({k: float(v) if k == "hit_pct" else int(v)
                    for k, v in m.groupdict().items()})
        mt = RE_SHAPE_TAIL.search(text)
        if mt:
            out["pread"] = int(mt.group("pread_us"))
            out["read_mib"] = float(mt.group("read_mib"))
            out["fill_wait_us"] = int(mt.group("fill_wait_us"))
    ma = RE_ATTR.search(text)
    if ma:
        d = ma.groupdict()
        out.setdefault("compulsory", int(d["compulsory"]))
        out.setdefault("capacity", int(d["capacity"]))
        out["evictions"] = int(d["evictions"])
        out["capacity_pct"] = float(d["capct"] if "capct" in d else d["capacity"] and d["capct"])
    mf = RE_FINAL.search(text)
    if mf:
        out.setdefault("req", int(mf.group("req")))
        out.setdefault("hits", int(mf.group("hits")))
        out.setdefault("hit_pct", float(mf.group("hit_pct")))
    mp = RE_FINAL_PREAD.search(text)
    if mp:
        out.setdefault("pread", int(mp.group("pread_usec")))
    return out


def evaluate(rows: list[dict], tol_dswap: float, tol_hit: float, tol_cap: float,
             req_warn: float = 10.0) -> tuple[list[tuple[str, bool, str]], bool]:
    """rows: [{label, fields, swap_before, swap_after}]。回傳 (條目, 全過?)。"""
    items: list[tuple[str, bool, str]] = []
    base = rows[0]

    # ── 判據 1：dswap ≈ 0，且多趟不累積 ──
    have_swap = [r for r in rows if r.get("swap_before") is not None]
    if not have_swap:
        items.append(("1 dswap≈0 且多趟不累積", False,
                      "無 swap 數據（--row label=LOG:SWAP_BEFORE:SWAP_AFTER）⇒ 此條**未驗**，不算過"))
    else:
        worst = max(abs(r["swap_after"] - r["swap_before"]) for r in have_swap)
        ok1a = worst <= tol_dswap
        drift = max(r["swap_after"] for r in have_swap) - min(r["swap_before"] for r in have_swap)
        ok1b = drift <= tol_dswap
        items.append(("1a 單趟 dswap≈0", ok1a,
                      "最大 |dswap| = %d MiB（門檻 %d）" % (worst, tol_dswap)))
        items.append(("1b 多趟不累積", ok1b,
                      "首趟前 → 末趟後 累積 %+d MiB（門檻 %d）" % (drift, tol_dswap)))

    # ── req 同量級？不同量級時「佔比」不可比（compulsory 分子會飽和）──
    def req_of(r):
        return r["fields"].get("req")
    reqs = [req_of(r) for r in rows if req_of(r)]
    if len(reqs) == len(rows) and len(reqs) > 1 and min(reqs) > 0:
        spread = 100.0 * (max(reqs) - min(reqs)) / max(reqs)
        if spread > req_warn:
            items.append(("0 req 同量級（可比性前提）", False,
                          "req 差異 %.1f%% > %.1f%% ⇒ capacity **佔比**不可比；"
                          "下面改用 per-1k-req" % (spread, req_warn)))

    # ── 判據 2：hit_pct 不降 ──
    b_hit = base["fields"].get("hit_pct")
    if b_hit is None:
        items.append(("2 hit_pct 不降", False, "baseline 抓不到 hit_pct ⇒ 未驗"))
    else:
        for r in rows[1:]:
            h = r["fields"].get("hit_pct")
            if h is None:
                items.append(("2 hit_pct 不降 [%s]" % r["label"], False, "抓不到 hit_pct ⇒ 未驗"))
            else:
                items.append(("2 hit_pct 不降 [%s]" % r["label"], h >= b_hit - tol_hit,
                              "%.2f%% vs baseline %.2f%%（容許 -%.2fpp）" % (h, b_hit, tol_hit)))

    # ── 判據 3a：capacity per 1k req 不升 ──
    def cap_per_1k(r):
        f = r["fields"]
        req = f.get("req")
        if not req or "capacity" not in f:
            return None
        return 1000.0 * f["capacity"] / req
    b_cap = cap_per_1k(base)
    if b_cap is None:
        items.append(("3a capacity/1k-req 不升", False, "baseline 缺 req 或 capacity ⇒ 未驗"))
    else:
        for r in rows[1:]:
            c = cap_per_1k(r)
            if c is None:
                items.append(("3a capacity/1k-req 不升 [%s]" % r["label"], False, "缺欄位 ⇒ 未驗"))
            else:
                items.append(("3a capacity/1k-req 不升 [%s]" % r["label"], c <= b_cap + tol_cap,
                              "%.3f vs baseline %.3f（門檻 +%.3f）" % (c, b_cap, tol_cap)))

    # ── 判據 3b：pread per req 下降（要真的降，不是「不升」）──
    def pread_per_req(r):
        f = r["fields"]
        req = f.get("req")
        if not req or "pread" not in f:
            return None
        return f["pread"] / req
    b_pr = pread_per_req(base)
    if b_pr is None:
        items.append(("3b pread/req 下降", False,
                      "baseline 缺 pread（CGC-SHAPE 的 pread_us= 或 final stats 的 pread_usec=）"
                      " ⇒ 未驗；注意 CGC-RIG-SNAPSHOT 有 env 門控"))
    else:
        for r in rows[1:]:
            p = pread_per_req(r)
            if p is None:
                items.append(("3b pread/req 下降 [%s]" % r["label"], False, "缺欄位 ⇒ 未驗"))
            else:
                items.append(("3b pread/req 下降 [%s]" % r["label"], p < b_pr,
                              "%.1f µs/req vs baseline %.1f ⇒ %+.1f%%"
                              % (p, b_pr, 100.0 * (p - b_pr) / b_pr)))

    all_ok = all(ok for _, ok, _ in items) and any(True for _ in items)
    return items, all_ok


def parse_row(spec: str) -> dict:
    """label=LOG[:SWAP_BEFORE:SWAP_AFTER]"""
    if "=" not in spec:
        raise ValueError("--row 需要 label=LOG[:SWAP_BEFORE:SWAP_AFTER]，拿到 %r" % spec)
    label, rest = spec.split("=", 1)
    parts = rest.split(":")
    path = parts[0]
    swap_b = swap_a = None
    if len(parts) == 3:
        swap_b, swap_a = float(parts[1]), float(parts[2])
    elif len(parts) != 1:
        raise ValueError("--row %r：swap 要給兩個數（before:after）或都不給" % spec)
    with open(path, encoding="utf-8", errors="replace") as fh:
        fields = parse_log(fh.read())
    return {"label": label, "fields": fields, "swap_before": swap_b, "swap_after": swap_a}


def self_test() -> int:
    SHAPE = ("CGC-SHAPE v=1 phase=final M=8 M_stat=ok width=8 union=113 fits=1 "
             "pool_cap_slots=5720 slots_layer=143 n_layer=40 "
             "req=%d hits=%d misses=%d hit_pct=%.2f compulsory=%d capacity=%d evict=%d "
             "zero_mapped=0 verify_refused=0 inv_viol=0 "
             "read_mib=%.1f pread_us=%d fill_wait_us=%d final_counters=1 "
             "gdn_ablated=0 gdn_saw_fused=1 gdn_saw_manual=0\n")
    pass_n = fail_n = 0

    def check(name, got, want):
        nonlocal pass_n, fail_n
        if got == want:
            pass_n += 1
            print("  ok   %s" % name)
        else:
            fail_n += 1
            print("  FAIL %s (got %r want %r)" % (name, got, want))

    # baseline：req 10000 / hit 91.4% / capacity 900 / pread 5,000,000 µs
    base = SHAPE % (10000, 9140, 860, 91.40, 8000, 900, 40, 900.0, 5000000, 100)
    # 修好：req 相近 / hit 不降 / capacity 降 / pread 降
    fixed = SHAPE % (10000, 9200, 800, 92.00, 8000, 700, 30, 900.0, 3500000, 100)
    # 變壞：hit 降 / capacity 升 / pread 升
    worse = SHAPE % (10000, 8800, 1200, 88.00, 8000, 1400, 90, 900.0, 7000000, 100)
    # 只有 B/C 行（沒有 CGC-SHAPE）：fallback 要能補出 req/hit_pct/capacity/pread
    legacy = ("llama_expert_cache: final stats: runtime requests=10000 hits=9140 misses=860 "
              "(hit rate 91.4%)  prewarm req=0 hit=0 miss=0  resident=10262.00 MiB "
              "file_reads=25470 pread_usec=5000000 fill_batch_usec=1 fill_wait_us=100 "
              "prefetch=0/0\n"
              "llama_expert_cache: miss attribution: compulsory=8000 capacity=900 "
              "(89.9% / 10.1% of 8900)  evictions=40  layers_distinct_over_slots=0  "
              "worst=layer 3 distinct=200 slots=143\n")

    f_base = parse_log(base)
    check("CGC-SHAPE req", f_base.get("req"), 10000)
    check("CGC-SHAPE hit_pct", f_base.get("hit_pct"), 91.40)
    check("CGC-SHAPE capacity", f_base.get("capacity"), 900)
    check("CGC-SHAPE pread_us 進到 pread", f_base.get("pread"), 5000000)
    f_leg = parse_log(legacy)
    check("fallback req", f_leg.get("req"), 10000)
    check("fallback hit_pct", f_leg.get("hit_pct"), 91.4)
    check("fallback capacity", f_leg.get("capacity"), 900)
    check("fallback evictions", f_leg.get("evictions"), 40)
    check("fallback pread_usec 進到 pread", f_leg.get("pread"), 5000000)

    rows_ok = [{"label": "base", "fields": parse_log(base), "swap_before": 1200, "swap_after": 1250},
               {"label": "fix", "fields": parse_log(fixed), "swap_before": 1240, "swap_after": 1255}]
    items, ok = evaluate(rows_ok, tol_dswap=64, tol_hit=0.5, tol_cap=0.0)
    check("全過 => ok", ok, True)

    rows_bad = [{"label": "base", "fields": parse_log(base), "swap_before": 1200, "swap_after": 1250},
                {"label": "worse", "fields": parse_log(worse), "swap_before": 1240, "swap_after": 1255}]
    _, ok_bad = evaluate(rows_bad, tol_dswap=64, tol_hit=0.5, tol_cap=0.0)
    check("變壞 => 不過", ok_bad, False)

    # swap 累積（單趟 dswap 都小，但一路往上）⇒ 判據 1b 要抓到
    rows_acc = [{"label": "r1", "fields": parse_log(base), "swap_before": 1200, "swap_after": 1240},
                {"label": "r2", "fields": parse_log(base), "swap_before": 1240, "swap_after": 1300},
                {"label": "r3", "fields": parse_log(base), "swap_before": 1300, "swap_after": 1400}]
    items_acc, ok_acc = evaluate(rows_acc, tol_dswap=64, tol_hit=0.5, tol_cap=0.0)
    check("swap 累積 => 不過（1b 抓到）", ok_acc, False)
    check("1b 條目存在", any(n.startswith("1b") for n, _, _ in items_acc), True)

    # 沒給 swap ⇒ 判據 1 標「未驗」且整體不過（不能靠省略欄位混過）
    rows_noswap = [{"label": "base", "fields": parse_log(base), "swap_before": None, "swap_after": None},
                   {"label": "fix", "fields": parse_log(fixed), "swap_before": None, "swap_after": None}]
    _, ok_ns = evaluate(rows_noswap, tol_dswap=64, tol_hit=0.5, tol_cap=0.0)
    check("缺 swap => 不過", ok_ns, False)

    # req 差太多 ⇒ 觸發可比性警告（容量佔比的陷阱）
    big = SHAPE % (40000, 36600, 3400, 91.50, 8000, 3400, 200, 900.0, 20000000, 100)
    rows_req = [{"label": "base", "fields": parse_log(base), "swap_before": 1200, "swap_after": 1250},
                {"label": "big", "fields": parse_log(big), "swap_before": 1240, "swap_after": 1255}]
    items_req, ok_req = evaluate(rows_req, tol_dswap=64, tol_hit=0.5, tol_cap=0.0)
    check("req 差 4x => 可比性條目 FAIL",
          any(n.startswith("0 ") and not o for n, o, _ in items_req), True)

    print("[miss-attr-gate] self-test %d passed / %d failed" % (pass_n, fail_n))
    return 0 if fail_n == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", action="append", default=[], metavar="label=LOG[:SWAP_BEFORE:SWAP_AFTER]",
                    help="第一個 row 是 baseline；可重複（多趟看累積）")
    ap.add_argument("--tol-dswap", type=float, default=64.0, help="MiB，單趟與累積都適用")
    ap.add_argument("--tol-hit-pct", type=float, default=0.5, help="hit_pct 容許下降的 pp 數")
    ap.add_argument("--tol-cap-per-req", type=float, default=0.0,
                    help="capacity/1k-req 容許上升量（0 = 一點都不許升）")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if len(args.row) < 2:
        print("需要至少兩個 --row（第一個是 baseline）", file=sys.stderr)
        return 2

    rows = [parse_row(s) for s in args.row]
    for r in rows:
        f = r["fields"]
        print("[%s] req=%s hit_pct=%s compulsory=%s capacity=%s pread=%s "
              "(per-1k-req cap=%s, µs/req pread=%s)"
              % (r["label"], f.get("req"), f.get("hit_pct"), f.get("compulsory"),
                 f.get("capacity"), f.get("pread"),
                 ("%.3f" % (1000.0 * f["capacity"] / f["req"])) if f.get("req") and "capacity" in f else "-",
                 ("%.1f" % (f["pread"] / f["req"])) if f.get("req") and "pread" in f else "-"))
    print()
    items, ok = evaluate(rows, args.tol_dswap, args.tol_hit_pct, args.tol_cap_per_req)
    for name, good, why in items:
        print("%-42s %s  %s" % (name, "PASS" if good else "FAIL", why))
    print()
    print("判決：%s（三條同時成立才算修好；「未驗」一律當 FAIL）" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
