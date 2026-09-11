#!/usr/bin/env python3
"""
cgc_logits_oracle_compare.py - 比較兩份 CGC logits oracle JSONL 檔, 判定 A/B 結果。

== 兩個獨立指標 (2026-09-11) ==

這支工具以前把「bit identical」與「argmax 相同」混成單一 PASS/FAIL, 結果同一組數字
可以被兩邊各自引用成相反的結論。現在它們是**兩個獨立指標, 永遠不合併**:

  M1 numeric identity    row_fnv1a64 逐位元組相同  -> 「logits 有沒有飄」
  M2 decision agreement  argmax_token 相同         -> 「選到的 token 一不一樣」
  M3 topk set agreement  top-N id 集合相同         -> 「候選集合一不一樣」(輔助)

為什麼不能合併:

  * M1 全同 M2 全同  -> 兩個配置真的是同一個計算。
  * M1 全同 M2 不同  -> 理論上不該發生; 出現就是 tie-break / 排序不穩定, 是 bug。
  * M1 不同 M2 全同  -> **最容易被誤讀的一格**: 只有 M1 能說「有差」, 只有 M2 會說
                        「沒差」。真相是「有數值飄移, 但還沒改變決策」。greedy 解碼
                        是混沌的, 這種狀態隨時可能在下一個 token 翻面。
  * M1 不同 M2 不同  -> 真的分歧。

所以報告一定要**兩個一起報**, 而且 2x2 交叉表要一起印出來。單看其中一個都足以
得出錯誤結論。

== 用法 ==

    python3 cgc_logits_oracle_compare.py --a oracle_8g.jsonl --b oracle_4g.jsonl \
        --report /tmp/cgc_oracle_diff.json

== exit code ==

  0 = --fail-on 的條件全部成立 (預設 both: M1 與 M2 都必須全同)
  1 = 有差異
  2 = 錯誤 (檔案壞掉 / 沒有可比對的 key)

  舊行為 (row hash 全同就算 PASS, 即使 argmax 不同) 可用 --fail-on numeric 重現。
"""
import argparse
import json
import sys

METRIC_DEFS = {
    "numeric_identity": "M1: same row_fnv1a64 for every step (= bit-identical logits)",
    "decision_agreement": "M2: same argmax_token for every step (= same chosen token)",
    "topk_set_agreement": "M3: same top-N token id set (= same candidate set)",
}


def _load_oracle(path):
    """Read JSONL; return {(step, token_idx, ctx_type): obj}."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"ERROR: {path}:{lineno} invalid JSON: {e}", file=sys.stderr)
                sys.exit(2)
            # include ctx_type in the alignment key so MTP/DEF (step, token_idx) entries
            # from different contexts never collide.
            key = (int(obj.get("step", 0)), int(obj.get("token_idx", 0)), obj.get("ctx_type", "DEF"))
            out[key] = obj
    return out


def _top_token_ids(obj):
    return [int(x["t"]) for x in obj.get("top", [])]


def _compare(va, vb):
    """Return list of (field, value_a, value_b) for every field that differs."""
    diffs = []
    for field in ("logits_fnv1a64", "row_fnv1a64", "argmax_token", "sum", "mean"):
        if va.get(field) != vb.get(field):
            diffs.append((field, va.get(field), vb.get(field)))
    ta, tb = _top_token_ids(va), _top_token_ids(vb)
    if ta != tb:
        diffs.append(("top_token_ids", ta, tb))
    return diffs


def _metric(equal, n):
    return {"equal": equal, "n": n, "rate": round(equal / n, 4) if n else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="oracle A JSONL (baseline)")
    ap.add_argument("--b", required=True, help="oracle B JSONL (candidate)")
    ap.add_argument("--report", default=None, help="optional diff report JSON output path")
    ap.add_argument("--fail-on", choices=("both", "numeric", "decision"), default="both",
                    help="which metric(s) may fail the run. 'numeric' reproduces the old "
                         "lenient rule where row-hash identity alone was a PASS.")
    args = ap.parse_args()

    a = _load_oracle(args.a)
    b = _load_oracle(args.b)

    common = sorted(set(a.keys()) & set(b.keys()))
    only_a = sorted(set(a.keys()) - set(b.keys()))
    only_b = sorted(set(b.keys()) - set(a.keys()))

    n_common = len(common)
    n_row_hash_eq = n_full_hash_eq = n_argmax_eq = n_top_eq = 0
    # 2x2: numeric identity x decision agreement
    tab = {"num_eq_dec_eq": 0, "num_eq_dec_ne": 0, "num_ne_dec_eq": 0, "num_ne_dec_ne": 0}
    diff_examples = []
    for key in common:
        va, vb = a[key], b[key]
        num_eq = va.get("row_fnv1a64") == vb.get("row_fnv1a64")
        dec_eq = va.get("argmax_token") == vb.get("argmax_token")
        n_row_hash_eq += int(num_eq)
        n_full_hash_eq += int(va.get("logits_fnv1a64") == vb.get("logits_fnv1a64"))
        n_argmax_eq += int(dec_eq)
        n_top_eq += int(_top_token_ids(va) == _top_token_ids(vb))
        tab[("num_eq_" if num_eq else "num_ne_") + ("dec_eq" if dec_eq else "dec_ne")] += 1
        diffs = _compare(va, vb)
        if diffs and len(diff_examples) < 5:
            diff_examples.append({"key": list(key), "numeric_equal": num_eq,
                                  "decision_equal": dec_eq, "diffs": diffs})

    metrics = {
        "numeric_identity": {**_metric(n_row_hash_eq, n_common), "definition": METRIC_DEFS["numeric_identity"]},
        "decision_agreement": {**_metric(n_argmax_eq, n_common), "definition": METRIC_DEFS["decision_agreement"]},
        "topk_set_agreement": {**_metric(n_top_eq, n_common), "definition": METRIC_DEFS["topk_set_agreement"]},
    }

    summary = {
        "a_path": args.a,
        "b_path": args.b,
        "n_a": len(a), "n_b": len(b), "n_common": n_common,
        "n_only_a": len(only_a), "n_only_b": len(only_b),
        # legacy keys (kept so older reports/plots keep working)
        "row_hash_equal": n_row_hash_eq,
        "full_hash_equal": n_full_hash_eq,
        "argmax_equal": n_argmax_eq,
        "top_equal": n_top_eq,
        # new: the independent metrics + the cross-tab that must be read together
        "metrics": metrics,
        "cross_tab": tab,
        "cross_tab_note": ("num_eq_dec_eq = no drift, same choice | num_eq_dec_ne = impossible "
                           "without a tie-break bug | num_ne_dec_eq = drift but same choice "
                           "(M1 says 'differs', M2 says 'identical') | num_ne_dec_ne = real divergence"),
        "fail_on": args.fail_on,
        "diff_examples": diff_examples,
    }
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")

    def pct(e, n):
        return f"{100.0 * e / n:.1f}%" if n else "n/a"

    print("=" * 74)
    print(f"oracle A: {args.a}  ({len(a)} entries)")
    print(f"oracle B: {args.b}  ({len(b)} entries)")
    print(f"common: {n_common}    only_A: {len(only_a)}    only_B: {len(only_b)}")
    if n_common == 0:
        print("FAIL: no common (step, token_idx, ctx_type) keys to compare")
        return 2
    print("-" * 74)
    print("INDEPENDENT METRICS  (never merge these into one verdict)")
    print(f"  M1 numeric identity    (bit-identical logits)  : "
          f"{n_row_hash_eq}/{n_common}  ({pct(n_row_hash_eq, n_common)})")
    print(f"  M2 decision agreement  (same argmax token)     : "
          f"{n_argmax_eq}/{n_common}  ({pct(n_argmax_eq, n_common)})")
    print(f"  M3 top-k set agreement (same top-N id set)     : "
          f"{n_top_eq}/{n_common}  ({pct(n_top_eq, n_common)})")
    print(f"  (aux) full_fnv1a64 identical                   : "
          f"{n_full_hash_eq}/{n_common}  ({pct(n_full_hash_eq, n_common)})")
    print("-" * 74)
    print("CROSS-TAB  (n=%d) -- read BOTH axes or you will draw the wrong conclusion" % n_common)
    print(f"  M1 same  & M2 same : {tab['num_eq_dec_eq']:>5}   no drift, same choice")
    print(f"  M1 same  & M2 diff : {tab['num_eq_dec_ne']:>5}   should be 0; non-zero = tie-break bug")
    print(f"  M1 diff  & M2 same : {tab['num_ne_dec_eq']:>5}   drift only; M1 says 'differs', M2 says 'same'")
    print(f"  M1 diff  & M2 diff : {tab['num_ne_dec_ne']:>5}   real divergence")
    if only_a:
        print(f"only_in_A (first 5): {only_a[:5]}")
    if only_b:
        print(f"only_in_B (first 5): {only_b[:5]}")
    if diff_examples:
        print("-" * 74)
        print("diff examples (first 5):")
        for ex in diff_examples:
            print(f"  key={ex['key']}  numeric_equal={ex['numeric_equal']} "
                  f"decision_equal={ex['decision_equal']}")
            for fld, va_, vb_ in ex["diffs"]:
                print(f"    {fld}: A={va_!r}  B={vb_!r}")
    print("=" * 74)

    m1_full = n_row_hash_eq == n_common
    m2_full = n_argmax_eq == n_common
    print(f"VERDICT M1 (numeric identity)   : {'PASS' if m1_full else 'FAIL'} "
          f"({n_row_hash_eq}/{n_common})")
    print(f"VERDICT M2 (decision agreement) : {'PASS' if m2_full else 'FAIL'} "
          f"({n_argmax_eq}/{n_common})")
    if m1_full and not m2_full:
        print("NOTE: logits are bit-identical yet the chosen token differs -> tie-break / "
              "ordering instability, not MoE layout drift. Treat as a bug.")
    if not m1_full and m2_full:
        print("NOTE: logits drifted but every choice matched *on this sample*. That is NOT "
              "'no difference' -- greedy decoding is chaotic and this can flip later. "
              "Report M1 and M2 separately; do not cite M2 alone.")

    if args.fail_on == "numeric":
        return 0 if m1_full else 1
    if args.fail_on == "decision":
        return 0 if m2_full else 1
    return 0 if (m1_full and m2_full) else 1


if __name__ == "__main__":
    sys.exit(main())
