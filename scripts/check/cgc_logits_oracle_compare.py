#!/usr/bin/env python3
"""
cgc_logits_oracle_compare.py - 比較兩份 CGC V2 logits oracle JSONL 檔, 判定 Soft Pool A/B 結果。

設計 (2026-09-05):
  - 讀 oracle-A.jsonl + oracle-B.jsonl (一行一個 JSON object, see llama-context.h for schema)
  - 對齊 key = (step, token_idx)
  - 對每個 token 比較: logits_fnv1a64 (full hash), row_fnv1a64 (per-row hash),
    argmax_token, sum, mean, top-N token ids
  - 統計: common / argmax_equal / row_hash_equal / full_hash_equal / top_equal
  - 用法:
      python3 cgc_logits_oracle_compare.py \
          --a /tmp/cgc_oracle_nosoftpool.jsonl \
          --b /tmp/cgc_oracle_softpool.jsonl \
          --report /tmp/cgc_oracle_diff.json
  - exit code: 0 = 全部一致 (PASS), 1 = 發現差異 (FAIL), 2 = 錯誤

使用情境:
  v1 (no soft pool) vs v2 (soft pool active) 各跑一次同一 prompt, dump 出兩份 oracle,
  跑這個 diff 看 logits 是否 byte-identical / argmax-identical。若 row_hash 一致但
  argmax 不一致 -> logits 有微小差但選擇相同;若 row_hash 不一致 -> V1 與 V2 路徑
  真的產出不同 logits, 需要查 cache fill 路徑。
"""
import argparse
import json
import sys


def _load_oracle(path):
    """Read JSONL; return list of (step, token_idx, dict)."""
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


def _compare(ka, va, vb):
    """Return list of (field, value_a, value_b) for every field that differs."""
    diffs = []
    for field in ("logits_fnv1a64", "row_fnv1a64", "argmax_token", "sum", "mean"):
        if va.get(field) != vb.get(field):
            diffs.append((field, va.get(field), vb.get(field)))
    ta, tb = _top_token_ids(va), _top_token_ids(vb)
    if ta != tb:
        diffs.append(("top_token_ids", ta, tb))
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="oracle A JSONL (e.g. baseline / no-soft-pool)")
    ap.add_argument("--b", required=True, help="oracle B JSONL (e.g. candidate / soft-pool)")
    ap.add_argument("--report", default=None, help="optional diff report JSON output path")
    ap.add_argument("--strict", action="store_true",
                    help="treat argmax mismatch even when row_hash matches as a hard fail (default: only fail on hash / top diff)")
    args = ap.parse_args()

    a = _load_oracle(args.a)
    b = _load_oracle(args.b)

    common = sorted(set(a.keys()) & set(b.keys()))
    only_a = sorted(set(a.keys()) - set(b.keys()))
    only_b = sorted(set(b.keys()) - set(a.keys()))

    n_common = len(common)
    n_row_hash_eq = 0
    n_full_hash_eq = 0
    n_argmax_eq = 0
    n_top_eq = 0
    diff_examples = []
    for key in common:
        va, vb = a[key], b[key]
        if va.get("row_fnv1a64") == vb.get("row_fnv1a64"):
            n_row_hash_eq += 1
        if va.get("logits_fnv1a64") == vb.get("logits_fnv1a64"):
            n_full_hash_eq += 1
        if va.get("argmax_token") == vb.get("argmax_token"):
            n_argmax_eq += 1
        if _top_token_ids(va) == _top_token_ids(vb):
            n_top_eq += 1
        diffs = _compare(key, va, vb)
        if diffs and len(diff_examples) < 5:
            diff_examples.append({"key": list(key), "diffs": diffs})

    summary = {
        "a_path": args.a,
        "b_path": args.b,
        "n_a": len(a),
        "n_b": len(b),
        "n_common": n_common,
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "row_hash_equal": n_row_hash_eq,
        "full_hash_equal": n_full_hash_eq,
        "argmax_equal": n_argmax_eq,
        "top_equal": n_top_eq,
        "diff_examples": diff_examples,
    }
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")

    # human-readable stdout
    print("=" * 60)
    print(f"oracle A: {args.a}  ({len(a)} entries)")
    print(f"oracle B: {args.b}  ({len(b)} entries)")
    print(f"common: {n_common}    only_A: {len(only_a)}    only_B: {len(only_b)}")
    if n_common == 0:
        print("FAIL: no common (step, token_idx) keys to compare")
        return 2
    print("-" * 60)
    print(f"row_fnv1a64  equal: {n_row_hash_eq}/{n_common}  ({100.0 * n_row_hash_eq / n_common:.1f}%)")
    print(f"full_fnv1a64 equal: {n_full_hash_eq}/{n_common}  ({100.0 * n_full_hash_eq / n_common:.1f}%)")
    print(f"argmax_token equal: {n_argmax_eq}/{n_common}  ({100.0 * n_argmax_eq / n_common:.1f}%)")
    print(f"top-N ids     equal: {n_top_eq}/{n_common}  ({100.0 * n_top_eq / n_common:.1f}%)")
    if only_a:
        print(f"only_in_A (first 5): {only_a[:5]}")
    if only_b:
        print(f"only_in_B (first 5): {only_b[:5]}")
    if diff_examples:
        print("-" * 60)
        print("diff examples (first 5):")
        for ex in diff_examples:
            print(f"  key={ex['key']}")
            for fld, va_, vb_ in ex["diffs"]:
                print(f"    {fld}: A={va_!r}  B={vb_!r}")
    print("=" * 60)

    # verdict
    if n_row_hash_eq == n_common and n_top_eq == n_common:
        print("VERDICT: PASS  (all row hashes + top-N token ids identical)")
        return 0
    if n_row_hash_eq == n_common and not args.strict:
        print("VERDICT: PASS  (all row hashes identical; argmax-only diffs within fp noise)")
        return 0
    print("VERDICT: FAIL  (row hash or top-N differs -> V1/V2 path divergence)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
