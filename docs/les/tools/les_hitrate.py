#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LES G1 離線命中率分析（cross-layer router lookahead）。

對齊兩份只讀探針產物（皆須在同一次 decode 跑、同一支 llama-bench/server 開啟）：
  1. LES 預測 sidecar（CGC_LES_PROBE=1）：
       每筆 LesHdr{magic,pass,il,dim,get_us}（5×int32, little-endian）+ dim×float32。
       命名 cgc_les_h-<il>：在【物理層 il】用本層 attn_post_norm 對【下一層】
       ffn_gate_inp 打分，故目標層 target = il+1。
  2. 真實路由 stderr（CGC_A1_HIST_PROBE=1）：
       CGC-A1-RAW: pass=<int> il=<int> j=<int> k=<int> ids=<id,...> cold=<bit,...>

指標：
  - top8 recall      : 預測 top8 與真實 top8 的交集 / 8（LES 命中率主口徑）
  - recall@d (d=1..16): 預測 top-d 覆蓋真實 top8 的比例（給 G3 depth 決策）
  - cold recall      : 真實【cold】專家被預測覆蓋的比例（resident 命中無預取價值，
                       cold 覆蓋才是縮池/省 SSD 的有效人口）
  - 分層：SSM(recr) vs full-attention；分步：early vs steady（warm-skip 後）
  - 對照基線：跨 token 同層 prev-token overlap（同一 il，pass p vs p-1 的真實 top8）

判定（charter G1）：三套 prompt 各 ≥128 穩態 step；
  top8 recall ≥80% PASS / <60% FAIL / 介於其間 FURTHER。

純標準庫（struct/re/argparse/json），無 numpy 依賴。
"""
import argparse
import json
import re
import struct
import sys
from collections import defaultdict

LES_MAGIC = 0x3153454C  # 'LES1' little-endian
A1_RE = re.compile(
    r"CGC-A1-RAW:\s*pass=(\d+)\s+il=(\d+)\s+j=(\d+)\s+k=(\d+)\s+ids=([0-9,\s]*)\s+cold=([01,\s]*)"
)


def parse_int_list(s):
    return [int(x) for x in s.replace(" ", "").split(",") if x != ""]


def parse_a1(path, only_j0=True):
    """true[(pass, il)] = (ids:list[int], cold:list[int])；並按 il 收集 pass 序列。"""
    true = {}
    by_il = defaultdict(dict)  # il -> {pass: set(ids)}
    n_rows = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = A1_RE.search(line)
            if not m:
                continue
            pas, il, j, k, ids_s, cold_s = m.groups()
            pas, il, j = int(pas), int(il), int(j)
            if only_j0 and j != 0:
                continue
            ids = parse_int_list(ids_s)
            cold = parse_int_list(cold_s)
            if not ids:
                continue
            key = (pas, il)
            if key not in true:  # j=0 每格一筆；保留第一筆
                true[key] = (ids, cold)
                by_il[il][pas] = set(ids)
                n_rows += 1
    return true, by_il, n_rows


def parse_les(path):
    """pred[(pass, target_il)] = list[float] logits（target_il = 物理 il+1）。"""
    pred = {}
    n_rec = 0
    dims = set()
    with open(path, "rb") as f:
        data = f.read()
    hdr_sz = struct.calcsize("<iiiii")
    off = 0
    while off + hdr_sz <= len(data):
        magic, pas, il, dim, get_us = struct.unpack_from("<iiiii", data, off)
        off += hdr_sz
        if magic != LES_MAGIC:
            # 不同步則中止（sidecar 只含同構記錄，理論上不會發生）
            sys.stderr.write(f"[warn] bad magic 0x{magic & 0xffffffff:08x} at rec {n_rec}, stop\n")
            break
        if off + dim * 4 > len(data):
            sys.stderr.write("[warn] truncated logits record, stop\n")
            break
        logits = list(struct.unpack_from("<%df" % dim, data, off))
        off += dim * 4
        target = il + 1  # 物理層 il 預測下一層
        pred[(pas, target)] = logits
        dims.add(dim)
        n_rec += 1
    return pred, n_rec, dims


def topd_set(logits, d):
    """分數最高的 d 個專家 id（分數相異時按值降序，穩定）。"""
    idx = sorted(range(len(logits)), key=lambda e: logits[e], reverse=True)
    return idx[:d]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1-log", required=True, help="stderr log 含 CGC-A1-RAW 行")
    ap.add_argument("--les-sidecar", default="/tmp/cgc_les_logits.bin")
    ap.add_argument("--warm-skip", type=int, default=64, help="丟棄前 N 個 decode pass（預設 64）")
    ap.add_argument("--k", type=int, default=8, help="真實路由 top-k（模型 n_expert_used，預設 8）")
    ap.add_argument("--max-depth", type=int, default=16, help="recall@d 掃描上限")
    ap.add_argument(
        "--prefetch-from", type=int, default=4,
        help="可預取層口徑起始 target 層（預設 4：豁免前 3 層 target1-3，"
             "由 prefill 自然預熱、不做 lookahead）。G1 判據以此口徑為準，"
             "另上報全層均值作對照",
    )
    ap.add_argument("--recr", default="", help="逗號分隔的 SSM(recr) 層列表，如 1,2,4,5,...；不填則不分層")
    ap.add_argument("--json", default="", help="可選：把完整結果寫成 JSON")
    ap.add_argument("--label", default="", help="這套 prompt 的標籤（報表用）")
    args = ap.parse_args()

    recr = set(parse_int_list(args.recr)) if args.recr else None

    true, by_il, n_a1 = parse_a1(args.a1_log)
    pred, n_pred, dims = parse_les(args.les_sidecar)

    if n_pred == 0:
        sys.stderr.write("[fatal] LES sidecar 無記錄：確認 CGC_LES_PROBE=1 且 CGC_LES_OUT 路徑正確\n")
        sys.exit(2)

    # ---- 對齊 LES pred vs true（warm-skip 後）----
    per_cell = []  # (pass, target_il, pred_set, true_set, cold_set)
    depth_recall = defaultdict(list)   # d -> [recall@d]
    for (pas, tgt), logits in pred.items():
        if pas < args.warm_skip:
            continue
        if (pas, tgt) not in true:
            continue
        ids, cold = true[(pas, tgt)]
        true_set = set(ids)
        cold_set = {e for e, c in zip(ids, cold) if c == 1} if len(cold) == len(ids) else set()
        order = sorted(range(len(logits)), key=lambda e: logits[e], reverse=True)
        for d in range(1, max(args.max_depth, args.k) + 1):
            pred_d = set(order[:d])
            depth_recall[d].append(len(pred_d & true_set) / float(args.k))
        pred_k = set(order[:args.k])
        hit = len(pred_k & true_set) / float(args.k)
        cold_hit = (len(pred_k & cold_set) / float(len(cold_set))) if cold_set else None
        per_cell.append((pas, tgt, hit, cold_hit, len(cold_set)))

    if not per_cell:
        sys.stderr.write("[fatal] warm-skip 後無可對齊 (pass,layer) 格；確認兩探針同一次跑、pass 計數一致\n")
        sys.exit(3)

    def group_stats(cells, layer_filter=None):
        hits = [h for (_, il, h, _, _) in cells if layer_filter is None or layer_filter(il)]
        colds = [c for (_, il, _, c, _) in cells if c is not None and (layer_filter is None or layer_filter(il))]
        return {
            "n_cells": len(hits),
            "top8_recall_mean": mean(hits),
            "top8_recall_min": min(hits) if hits else None,
            "top8_recall_p25": sorted(hits)[len(hits) // 4] if hits else None,
            "top8_recall_median": sorted(hits)[len(hits) // 2] if hits else None,
            "cold_recall_mean": mean(colds) if colds else None,
            "cold_cells": len(colds),
        }

    steady = [c for c in per_cell if c[0] >= args.warm_skip]
    n_pass = len({c[0] for c in per_cell})

    overall = group_stats(steady)
    recr_stats = group_stats(steady, lambda il: il in recr) if recr is not None else None
    attn_stats = group_stats(steady, lambda il: il not in recr) if recr is not None else None

    # per-layer top8 recall
    per_layer = {}
    for il in sorted({c[1] for c in steady}):
        st = group_stats(steady, lambda x, il=il: x == il)
        per_layer[il] = round(st["top8_recall_mean"], 4)

    depth_curve = {d: round(mean(depth_recall[d]), 4) for d in sorted(depth_recall)}

    # ---- 可預取層口徑（豁免前 pf-1 層，由 prefill 自然預熱、不做 lookahead）----
    # G1 裁定（2026-10-17）：判據以可預取層（target>=pf，預設 4）為準；全層均值僅對照。
    pf = args.prefetch_from
    pf_cells = [c for c in steady if c[1] >= pf]
    pref = group_stats(pf_cells)
    pf_per_layer = {il: v for il, v in per_layer.items() if int(il) >= pf}
    pf_min_layer = min(pf_per_layer.items(), key=lambda kv: kv[1]) if pf_per_layer else (None, 1.0)

    # ---- 跨 token prev-token 同層基線（只需 A1）----
    prev_overlap = []
    for il, passes in by_il.items():
        ps = sorted(p for p in passes if p >= args.warm_skip)
        for a, b in zip(ps[:-1], ps[1:]):
            prev_overlap.append(len(passes[a] & passes[b]) / float(args.k))
    prev_baseline = mean(prev_overlap)

    # ---- 判定（G1 裁定 2026-10-17：以可預取層 target>=pf 為準，全層僅對照）----
    med_all = overall["top8_recall_mean"]
    med = pref["top8_recall_mean"]
    enough_steps = n_pass >= 128
    no_low_pref_layer = pf_min_layer[1] >= 0.50
    verdict = "FURTHER"
    if enough_steps and med >= 0.80 and no_low_pref_layer:
        verdict = "PASS"
    elif med < 0.60:
        verdict = "FAIL"
    elif not enough_steps:
        verdict = "INSUFFICIENT_STEPS"

    # ---- 報表 ----
    lbl = f"[{args.label}] " if args.label else ""
    print(f"===== LES G1 跨層命中率 {lbl}=====")
    print(f"records: A1-RAW rows={n_a1}  LES sidecar={n_pred}  sidecar dims={sorted(dims)}")
    print(f"steady decode steps (warm-skip>={args.warm_skip}): {n_pass}  (charter 需 ≥128)")
    print(f"aligned cells (pass,target_layer): {overall['n_cells']}")
    print()
    print(f"[全層 target1+] top8 recall : mean={overall['top8_recall_mean']:.4f}  "
          f"median={overall['top8_recall_median']:.4f}  p25={overall['top8_recall_p25']:.4f}  "
          f"min={overall['top8_recall_min']:.4f}  (僅對照)")
    print(f"[可預取層 target>={pf}] top8 recall : mean={pref['top8_recall_mean']:.4f}  "
          f"median={pref['top8_recall_median']:.4f}  p25={pref['top8_recall_p25']:.4f}  "
          f"min={pref['top8_recall_min']:.4f}  (G1 判據口徑；前 {pf-1} 層由 prefill warm、豁免)")
    if pf_per_layer:
        print(f"  可預取層最低單層 = target {pf_min_layer[0]}: {pf_min_layer[1]:.4f} "
              f"(門檻 ≥0.50：{'過' if no_low_pref_layer else '破'})")
    if overall["cold_recall_mean"] is not None:
        print(f"LES 對 cold 專家覆蓋率 : mean={overall['cold_recall_mean']:.4f}  "
              f"(cold 格 {overall['cold_cells']}；resident 命中不計預取價值)")
    print(f"對照：跨 token 同層 prev-token overlap = {prev_baseline:.4f}  (n={len(prev_overlap)})")
    if recr_stats and attn_stats:
        print(f"  ├─ SSM(recr) 層 top8 recall = {recr_stats['top8_recall_mean']:.4f} (n={recr_stats['n_cells']})")
        print(f"  └─ full-attn 層 top8 recall = {attn_stats['top8_recall_mean']:.4f} (n={attn_stats['n_cells']})")
    else:
        print("  (未提供 --recr 層表，略過 SSM/attention 分層)")
    print()
    print("recall@d 掃描（預取 depth 越大覆蓋越高，給 G3 depth 決策）：")
    line = "  d=" + " ".join(f"{d:>2}" for d in sorted(depth_curve))
    val = "  r =" + " ".join(f"{depth_curve[d]:.2f}" for d in sorted(depth_curve))
    print(line)
    print(val)
    print()
    print(f"判定（可預取層 target>={pf}）: {verdict}  "
          f"(mean={med:.4f} ≥0.80、無單層<0.50={no_low_pref_layer} / <0.60 FAIL, "
          f"steps≥128: {enough_steps}；全層對照 mean={med_all:.4f})")

    result = {
        "label": args.label,
        "n_a1_rows": n_a1,
        "n_les_records": n_pred,
        "sidecar_dims": sorted(dims),
        "warm_skip": args.warm_skip,
        "prefetch_from_target": pf,
        "steady_steps": n_pass,
        "n_cells": overall["n_cells"],
        "overall": overall,
        "prefetchable": pref,
        "prefetchable_min_layer": [pf_min_layer[0], pf_min_layer[1]],
        "verdict_basis": "prefetchable target>=%d (前 %d 層由 prefill warm 豁免，2026-10-17 裁定)" % (pf, pf - 1),
        "recr": recr_stats,
        "attention": attn_stats,
        "per_layer_top8": per_layer,
        "depth_recall": depth_curve,
        "prev_token_baseline": prev_baseline,
        "verdict": verdict,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nfull JSON -> {args.json}")


if __name__ == "__main__":
    main()
