#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0C-1：MTP 時間維 union 預取的「離線 oracle 上界」探針（不訓練、不動引擎、不需 GPU）。

只讀 G1 的真實路由 dump（CGC-A1-RAW stderr log，每 decode step 每層的真實 top-k 專家），
回答一個縮池問題：

    如果 MTP 能預測未來 N 個 token（N 個 draft），並在每層把「當前+未來 N 步」的
    專家 union 預取進 cache，在 cache 容量只有 cap 槽/層（8G=143、6G≈107、4G≈72）時，
    最理想（oracle 替換）能把專家命中率做到多少？

這是 Ember「N draft × 逐層 batch-router + union 聚合 + verify 前集合裁剪」的
【預測完美上界】：它假設 draft token 全對（ρ_expert=1）。真實 MTP 有誤差（token α=0.654，
專家集合級更寬鬆但非 1），故：
  * 本腳本的命中率 = 預測完全正確時的天花板；
  * 真實收益 = 天花板 × 真實專家 recall（由 0C-2 在 verify forward 加 dump 後量，需 GPU）；
  * 若連 oracle 上界都顯示縮池後命中率追不回，則訓練 MTP 也救不了縮池（可及早否證）。

三個模型：
  1. union 擴大曲線：|S(p) ∪ … ∪ S(p+N)| 隨 N 擴大（要覆蓋未來 N 步需多少專家/層）。
  2. 容量 cap 下的命中率（MTP union 縮池天花板）：
       N=0 用 LRU（無預測，當前近似基線）；
       N 有限用「預取保護 LRU」（union 預取提前載入、替換時保護未來 N 步視窗，不短視誤殺，
       對 N 單調）；N=∞ 用全序列 Belady-MIN（完美 oracle 上界）。
  3. 單點 overlap 對照：|S(p) ∩ S(p+n)|/k（不做 union、只預測單一未來步的衰減）。

口徑：每層獨立快取（MoE 專家按層）；每個 decode step 每層請求 k=8 個專家（一個批量）；
批量 Belady：先批量判 hit/miss，再依「未來 N 步內最遠使用」淘汰。
純標準庫（struct 不需要；re/argparse/json/collections）。

用法：
  python3 mtp_union_recall.py \
      --a1-log Backup/les_g1_hitrate_20261017/a1raw_code.stderr.log \
      --warm-skip 64 --label code --json out_code.json
  （三套 prompt 各跑一次；腳本只處理一份 log。）
"""
import argparse
import json
import re
import sys
from collections import OrderedDict, defaultdict

A1_RE = re.compile(
    r"CGC-A1-RAW:\s*pass=(\d+)\s+il=(\d+)\s+j=(\d+)\s+k=(\d+)\s+ids=([0-9,\s]*)\s+cold=([01,\s]*)"
)

# 8G 池實測 n_slots=143/層（a1raw log: CGC Soft Pool init n_slots=143）。
# 6G/4G 按池位元組等比換算（143×0.75≈107、143×0.5≈72）；真實縮池 LAYER_CAPS 以重測為準。
CAP_8G = 143
CAP_6G = 107
CAP_4G = 72


def parse_int_list(s):
    return [int(x) for x in s.replace(" ", "").split(",") if x != ""]


def parse_a1(path, warm_skip, only_j0=True):
    """by_il[il] = list[set] 依 pass 排序的真實專家集合（僅含 pass>=warm_skip 的連續段）。"""
    raw = defaultdict(dict)  # il -> {pass: set}
    k_seen = defaultdict(int)
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = A1_RE.search(line)
            if not m:
                continue
            pas, il, j, k, ids_s, _cold = m.groups()
            pas, il, j, k = int(pas), int(il), int(j), int(k)
            if only_j0 and j != 0:
                continue
            ids = parse_int_list(ids_s)
            if not ids:
                continue
            if pas < warm_skip:
                continue
            if pas not in raw[il]:
                raw[il][pas] = set(ids)
                k_seen[il] = k
    by_il = {}
    for il, pmap in raw.items():
        ps = sorted(pmap)
        by_il[il] = [pmap[p] for p in ps]
    return by_il, k_seen


def union_growth(seq, ns):
    """對每個起點 p，|∪_{t=0..N} S(p+t)|；回傳各 N 的平均 union 大小（除以 k 給倍數）。"""
    P = len(seq)
    out = {}
    for N in ns:
        sizes = []
        for p in range(P - N):
            u = set()
            for t in range(N + 1):
                u |= seq[p + t]
            sizes.append(len(u))
        out[N] = (sum(sizes) / len(sizes)) if sizes else float("nan")
    return out


def point_overlap(seq, k, ns):
    """|S(p) ∩ S(p+n)|/k 平均（單點預測衰減，不 union）。"""
    P = len(seq)
    out = {}
    for n in ns:
        vals = []
        for p in range(P - n):
            vals.append(len(seq[p] & seq[p + n]) / float(k))
        out[n] = (sum(vals) / len(vals)) if vals else float("nan")
    return out


def build_next_use(seq):
    """next_use[p][e] = 在位置 p 之後，專家 e 第一次出現的位置（不含 p）；無則 None。"""
    P = len(seq)
    nxt = [None] * P
    future = {}
    for p in range(P - 1, -1, -1):
        nxt[p] = dict(future)
        for e in seq[p]:
            future[e] = p
    return nxt


def sim_lru(seq, cap):
    """N=0 無未來預測的 LRU 基線；回傳 (hits, total)。"""
    cache = OrderedDict()
    hits = 0
    total = 0
    for s in seq:
        for e in s:
            total += 1
            if e in cache:
                cache.move_to_end(e)
                hits += 1
            else:
                if len(cache) >= cap:
                    cache.popitem(last=False)
                cache[e] = True
    return hits, total


def sim_belady_min(seq, cap, nxt):
    """
    全序列 Belady(MIN)：無限未來視野的最優離線替換（oracle 上界）。
    批量請求 s；淘汰快取中、不在當前步、且全序列未來最遠才使用的專家。
    """
    P = len(seq)
    cache = set()
    hits = 0
    total = 0
    for p in range(P):
        s = seq[p]
        miss = [e for e in s if e not in cache]
        hits += (len(s) - len(miss))
        total += len(s)
        if not miss:
            continue
        need_evict = len(miss) - (cap - len(cache))
        if need_evict > 0:
            cand = cache - s
            nu = nxt[p]
            def key(e):
                q = nu.get(e)
                return (1, 0) if q is None else (0, q)  # 永不復用最優先；否則越晚用越先踢
            for e in sorted(cand, key=key, reverse=True)[:need_evict]:
                cache.discard(e)
        for e in miss:
            if len(cache) < cap:
                cache.add(e)
    return hits, total


def sim_protected_lru(seq, cap, horizon):
    """
    有限視野「預取保護 LRU」（Ember 真實語義：union 預取只提前載入、並在替換時保護未來視窗，
    不對視野外做短視淘汰）。
      每步為 miss 騰位時，淘汰候選 = cache 減當前步；
      先踢「不在未來 horizon 步 union W(p)」內的（按 LRU 舊→新），不夠再踢在 W 內的。
    horizon=0 時 W 只含當前步（與普通 LRU 等價，因當前步本就不會被踢）。
    相對短視 Belady，它不會因「視窗外看不到」而誤殺即將復用的專家，命中率對 N 單調不降。
    """
    P = len(seq)
    cache = OrderedDict()  # LRU：舊→新
    hits = 0
    total = 0
    for p in range(P):
        s = seq[p]
        s_set = set(s)
        # 命中（刷新 LRU）
        for e in s:
            if e in cache:
                cache.move_to_end(e)
                hits += 1
        total += len(s)
        miss = [e for e in s if e not in cache]
        if not miss:
            continue
        # 未來 horizon 步 union（含當前）
        W = set()
        for t in range(p, min(P, p + horizon + 1)):
            W |= seq[t]
        need_evict = len(miss) - (cap - len(cache))
        if need_evict > 0:
            cand = [e for e in cache.keys() if e not in s_set]
            cand.sort(key=lambda e: 1 if e in W else 0)  # stable：組內維持 LRU 舊→新
            for e in cand[:need_evict]:
                cache.pop(e, None)
        for e in miss:
            if len(cache) < cap:
                cache[e] = True
    return hits, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1-log", required=True)
    ap.add_argument("--warm-skip", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--label", default="")
    ap.add_argument("--ns", default="0,1,2,3,4,7,11,15",
                    help="union 視野 N 掃描（MTP draft 數），逗號分隔")
    ap.add_argument("--caps", default="%d,%d,%d,80,96,120,128,160" % (CAP_4G, CAP_6G, CAP_8G),
                    help="每層 cache 槽位 cap 掃描")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    ns = sorted(set(parse_int_list(args.ns)))
    caps = sorted(set(parse_int_list(args.caps)))

    by_il, k_seen = parse_a1(args.a1_log, args.warm_skip)
    if not by_il:
        sys.stderr.write("[fatal] 無 A1 資料，確認 --a1-log 與 --warm-skip\n")
        sys.exit(2)
    layers = sorted(by_il)
    k = args.k or max(k_seen.values())

    # 預 compute next_use（每層）
    nxt_by_il = {il: build_next_use(by_il[il]) for il in layers}

    # ---- 1. 穩態工作集 / union 擴大 / 單點 overlap（對所有層聚合後平均）----
    growth_acc = defaultdict(list)
    point_acc = defaultdict(list)
    workset = {}
    for il in layers:
        seq = by_il[il]
        workset[il] = len(set().union(*seq)) if seq else 0
        g = union_growth(seq, ns)
        for N, v in g.items():
            growth_acc[N].append(v)
        po = point_overlap(seq, k, ns)
        for n, v in po.items():
            point_acc[n].append(v)
    growth = {N: round(sum(v) / len(v), 2) for N, v in sorted(growth_acc.items())}
    growth_x = {N: round(sum(v) / len(v) / k, 2) for N, v in sorted(growth_acc.items())}
    point = {n: round(sum(v) / len(v), 3) for n, v in sorted(point_acc.items())}
    ws_vals = [workset[il] for il in layers]
    ws_mean = sum(ws_vals) / len(ws_vals)
    ws_min, ws_max = min(ws_vals), max(ws_vals)

    # ---- 2. 命中率網格：列=視野 N，行=cap（跨層彙總 hits/total）----
    # N=0 走 LRU；N>=1 走有限 Belady；N=∞ 全序列 Belady。
    horizons = [0] + [n for n in ns if n > 0]
    # 統計矩陣：rate[N][cap] = 總命中率
    rate = {}
    totals = {}
    # 先 LRU（N=0）
    for cap in caps:
        H = T = 0
        for il in layers:
            h, t = sim_lru(by_il[il], cap)
            H += h
            T += t
        rate.setdefault(0, {})[cap] = H / float(T)
        totals[cap] = T
    # 有限視野：預取保護 LRU（Ember 現實語義，N 單調）
    for N in ns:
        if N == 0:
            continue
        for cap in caps:
            H = T = 0
            for il in layers:
                h, t = sim_protected_lru(by_il[il], cap, N)
                H += h
                T += t
            rate.setdefault(N, {})[cap] = H / float(T)
    # 全序列 oracle：Belady-MIN
    for cap in caps:
        H = T = 0
        for il in layers:
            h, t = sim_belady_min(by_il[il], cap, nxt_by_il[il])
            H += h
            T += t
        rate.setdefault("inf", {})[cap] = H / float(T)

    def rc(cap):
        """cap 對應的池標籤（8G/6G/4G）。"""
        return {CAP_8G: "≈8G", CAP_6G: "≈6G", CAP_4G: "≈4G"}.get(cap, "")

    # ---- 報表 ----
    lbl = f"[{args.label}] " if args.label else ""
    print(f"===== 0C MTP 時間維 union 預取 oracle 上界 {lbl}=====")
    print(f"layers={len(layers)} (il {min(layers)}..{max(layers)})  warm-skip>={args.warm_skip}  "
          f"k={k}  每層穩態 steps≈{len(by_il[layers[0]])}")
    print()
    print(f"穩態工作集 distinct 專家/層（577 步總動員）：mean={ws_mean:.1f}  min={ws_min}  max={ws_max}  "
          f"(256 上限；8G 池 143 槽/層)")
    print()
    print("union 擴大（覆蓋當前+未來 N 步，每層平均）：")
    print("  N(draft數) = " + " ".join(f"{N:>4}" for N in ns))
    print("  union 專家 = " + " ".join(f"{growth[N]:>4.0f}" for N in ns))
    print("  /k 倍數    = " + " ".join(f"{growth_x[N]:>4.1f}" for N in ns))
    print()
    print("單點 overlap 對照（不 union，只預測第 n 步）：|S(p)∩S(p+n)|/k")
    print("  n = " + " ".join(f"{n:>4}" for n in ns if n > 0))
    print("  r = " + " ".join(f"{point[n]:>4.2f}" for n in ns if n > 0))
    print()
    print("命中率網格（列=預測視野 N，行=每層 cap 槽；N=0 LRU 現狀，N 有限=預取保護 LRU，N=∞ 完美 oracle）：")
    head = "  N\\cap | " + " ".join(f"{c:>5}{rc(c):<3}" for c in caps)
    print(head)
    order_ns = [0] + [n for n in ns if n > 0] + ["inf"]
    for N in order_ns:
        row = rate.get(N, {})
        tag = "LRU" if N == 0 else ("oracle" if N == "inf" else f"N={N:<3}")
        print(f"  {tag:>6} | " + " ".join(f"{row.get(c, float('nan')):>8.3f}" for c in caps))
    print()

    # ---- 縮池決策摘要 ----
    base8 = rate[0][CAP_8G]          # 8G 無預測（LRU）現狀近似
    if CAP_6G in caps:
        def best_for(cap):
            return rate.get("inf", {}).get(cap)
        print("縮池關鍵（LRU 現狀 vs 完美 union 預測）：")
        for cap, tag in [(CAP_8G, "8G"), (CAP_6G, "6G"), (CAP_4G, "4G")]:
            if cap in caps:
                lru = rate[0][cap]
                orc = rate.get("inf", {}).get(cap)
                print(f"  {tag} cap={cap}: LRU={lru:.3f}  完美預測上限={orc:.3f}  "
                      f"(vs 8G/LRU {base8:.3f}，差 {orc - base8:+.3f})")
        # 各視野 N 在 6G 下的命中
        if CAP_6G in caps:
            print("  6G 池隨視野 N 的命中：" + " ".join(
                f"N{N}={rate[N][CAP_6G]:.3f}" for N in ns if N > 0) +
                f"  oracle={rate['inf'][CAP_6G]:.3f}")

    result = {
        "label": args.label,
        "warm_skip": args.warm_skip,
        "k": k,
        "layers": [min(layers), max(layers), len(layers)],
        "steady_workset_per_layer": {"mean": ws_mean, "min": ws_min, "max": ws_max},
        "union_growth_experts": {str(N): growth[N] for N in ns},
        "union_growth_x_k": {str(N): growth_x[N] for N in ns},
        "point_overlap": {str(n): point[n] for n in ns if n > 0},
        "hit_rate_grid": {
            ("LRU" if N == 0 else ("oracle" if N == "inf" else str(N))): {
                str(cap): round(rate[N][cap], 4) for cap in caps
            } for N in order_ns
        },
        "cap_map": {CAP_8G: "8G", CAP_6G: "6G", CAP_4G: "4G"},
        "note": "命中率為預測完美(ρ_expert=1)的 oracle 天花板；真實 MTP 收益需乘真實專家 recall(0C-2 量)。",
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nfull JSON -> {args.json}")


if __name__ == "__main__":
    main()
