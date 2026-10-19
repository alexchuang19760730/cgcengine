#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0A：從 server/llama-bench 日誌抽 MTP token 接受率 α_tok（純標準庫，不跑引擎）。

目的：IQ4_XS（等權 imatrix，task#22）量化後，與 IQ3_XXS 同場跑 MTP on（warm-skip64、seed1、
prod-new harness，CGC_SERVER_MTP_N_MAX=3），用同一抽取口徑把 α_tok 抽出來橫評，判
「α=0.654 偏低是否量化位元（R1）造成」。

認列的權威欄位（口徑來自 server-context.cpp:673-674、MTP3_25TPS_ARITHMETIC_2026-09-18.md）：
  draft acceptance = X (a accepted / g generated)   -> α_tok = a/g
  mean len = Y                                      -> 每步 token 數（含基底）= 1 + a/verif_steps
  /metrics: spec_decode_num_drafts_total            -> 驗證步數

llama-bench JSON 路徑（若有）：draft_n / draft_n_accepted 或 n_draft_* 鍵，逐筆抽。

用法：
  python3 extract_mtp_acceptance.py <server.log> [more.log ...]
  python3 extract_mtp_acceptance.py --json out.json *.log
"""
import argparse
import json
import re
import statistics
import sys

# draft acceptance = 0.654 (123 accepted / 188 generated), mean len = 1.65
RE_ACC = re.compile(
    r"draft acceptance\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\)"
    r"(?:\s*,\s*mean len\s*=\s*([0-9.]+))?"
)
# 有些版本只印 mean len
RE_MEANLEN = re.compile(r"mean len\s*=\s*([0-9.]+)")
# llama-bench 風格 JSON 行（整行是 JSON）
RE_JSONKEYS = re.compile(r"")


def scrape_text(path):
    rows = []
    with open(path, "r", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            m = RE_ACC.search(line)
            if m:
                alpha, a, g, meanlen = m.groups()
                rows.append({
                    "line": ln,
                    "alpha": float(alpha),
                    "accepted": int(a),
                    "generated": int(g),
                    "mean_len": float(meanlen) if meanlen else None,
                })
                continue
            # 嘗試整行 JSON（llama-bench --json 每筆）
            s = line.strip()
            if s.startswith("{") and ("accept" in s or "draft" in s):
                try:
                    obj = json.loads(s)
                except Exception:
                    obj = None
                if obj:
                    a = obj.get("draft_n_accepted", obj.get("n_draft_accepted"))
                    g = obj.get("draft_n", obj.get("n_draft_total"))
                    if a is not None and g:
                        rows.append({
                            "line": ln, "alpha": float(a) / float(g),
                            "accepted": int(a), "generated": int(g),
                            "mean_len": obj.get("mean_len"),
                        })
    return rows


def summarize(rows):
    if not rows:
        return None
    a = sum(r["accepted"] for r in rows)
    g = sum(r["generated"] for r in rows)
    alphas = [r["alpha"] for r in rows]
    mls = [r["mean_len"] for r in rows if r["mean_len"] is not None]
    return {
        "samples": len(rows),
        "alpha_pooled": round(a / g, 4) if g else None,  # 加總 a/加總 g（權威）
        "alpha_median": round(statistics.median(alphas), 4),
        "alpha_mean": round(statistics.mean(alphas), 4),
        "alpha_p10": round(sorted(alphas)[max(0, int(len(alphas) * 0.1))], 4),
        "alpha_p90": round(sorted(alphas)[min(len(alphas) - 1, int(len(alphas) * 0.9))], 4),
        "mean_len_median": round(statistics.median(mls), 3) if mls else None,
        "accepted": a, "generated": g,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    allout = {}
    for path in args.logs:
        rows = scrape_text(path)
        summ = summarize(rows)
        allout[path] = {"summary": summ, "rows": rows}
        if summ:
            print(f"== {path} ==")
            print(f"  samples={summ['samples']}  α_pooled={summ['alpha_pooled']}  "
                  f"α_median={summ['alpha_median']}  [{summ['alpha_p10']}..{summ['alpha_p90']}]  "
                  f"mean_len={summ['mean_len_median']}  ({summ['accepted']}/{summ['generated']})")
        else:
            print(f"== {path} ==  （無 draft acceptance 欄位，確認該 log 是 MTP on）")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(allout, f, ensure_ascii=False, indent=2)
        print(f"\n-> {args.json}")
    if not any(v["summary"] for v in allout.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
