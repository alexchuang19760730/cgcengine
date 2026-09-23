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

== 對齊鍵 (--align, 2026-09-23) ==

預設 `step` 用 (step, token_idx, ctx_type) 對齊, 其中 `step` 是引擎的全域 dump 序號。這在
「同一個請求形狀、只差一個數值旋鈕」時是對的, 但它假設兩份 dump 的評估序列**一樣長、一樣順序**。

跨幾何的比較不成立: 一個「重用前綴」(還原到 checkpoint, 只算尾巴) 的請求與一個「完整 prefill」
的請求, 評估次數與順序都不同, 全域 step 會整段錯位 —— 於是每個鍵都對到別的計算, 印出一堆看起來
很像真實分歧的差異。

`--align pos` 改用列自己的身分 (ctx_type, n_tokens, pmax, token_idx): pmax 是該次評估後記憶體的
最大位置, 所以同一個位置在兩臂上會拿到同一個鍵, 與它前面發生過什麼無關。

一個鍵可以有多筆列, 而且**值不一定相同** (實測: 一個請求內約 50 個 MTP 鍵各有兩筆、hashes
不同 —— 同一個位置在同一個請求裡被算了不只一次), 所以一併比較的是 multiset, 不是單值:

    鍵相同 = 兩邊的 row_fnv1a64 排序後清單相同 (逐位元); 長度不同 = structural, 另外計數

這樣不必假設「第幾筆對第幾筆」, 也不會靜默丟掉任何一筆。前提是範圍要對: 位置在多個請求間會
重來, 所以要先用 request 界限把 dump 切開 (pxf_restore_probe.py 寫出 `<dump>.reqN.jsonl`
並把切出幾段當成斷言), 否則兩個請求會被混在同一個鍵下。

== 用法 ==

    python3 cgc_logits_oracle_compare.py --a oracle_8g.jsonl --b oracle_4g.jsonl \
        --report /tmp/cgc_oracle_diff.json
    python3 cgc_logits_oracle_compare.py --a full.jsonl --b restored.jsonl --align pos
    python3 cgc_logits_oracle_compare.py --selftest

== exit code ==

  0 = --fail-on 的條件全部成立 (預設 both: M1 與 M2 都必須全同)
  1 = 有差異
  2 = 錯誤 (檔案壞掉 / 沒有可比對的 key)

  舊行為 (row hash 全同就算 PASS, 即使 argmax 不同) 可用 --fail-on numeric 重現。
"""
import argparse
import importlib.util
import json
import os
import sys

ALIGN_KEYS = {"step": "(step, token_idx, ctx_type)",
              "pos":  "(ctx_type, n_tokens, pmax, token_idx)"}

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


def _load_oracle_pos(path):
    """Read JSONL keyed on the ROW's own identity: (ctx_type, n_tokens, pmax, token_idx).

    Returns ({key: [obj, ...]}, notes). Records that share a key are kept together -- they are a
    multiset, not a collision to collapse (measured: ~50 MTP keys per request hold two records with
    DIFFERENT hashes). Callers must scope that key by request; positions repeat across requests.
    """
    out, n_rows = {}, 0
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
            n_rows += 1
            key = (obj.get("ctx_type", "DEF"), int(obj.get("n_tokens", 0)),
                   int(obj.get("pmax", 0)), int(obj.get("token_idx", 0)))
            out.setdefault(key, []).append(obj)
    return out, {"rows": n_rows, "keys": len(out),
                 "repeated_keys": sum(1 for v in out.values() if len(v) > 1),
                 "rows_in_repeated_keys": sum(len(v) for v in out.values() if len(v) > 1),
                 "rule": "every record kept; a key compares as a multiset of row hashes"}


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


def _write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def selftest():
    """Every claim of --align pos, exercised through the real CLI (exit codes included).

    A test that re-implements the comparison cannot fail; this one shells out to this file.
    """
    import subprocess
    import tempfile

    keys = [("DEF", 1, 207, 0), ("MTP", 1, 208, 0), ("DEF", 4, 211, 0), ("DEF", 4, 211, 1)]

    def rows(step0, hashes, extra=()):
        out = []
        for i, ((ctx, nt, pmax, tidx), h) in enumerate(zip(keys, hashes)):
            out.append({"step": step0 + i, "token_idx": tidx, "n_tokens": nt, "ctx_type": ctx,
                        "pmax": pmax, "row_fnv1a64": h, "argmax_token": 7, "top": [{"t": 1, "v": 1.0}]})
        return out + list(extra)

    def run(a, b, align):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--a", a, "--b", b,
                            "--align", align], capture_output=True, text=True)
        return r.returncode, r.stdout

    bad = 0
    with tempfile.TemporaryDirectory() as td:
        pa, pb = os.path.join(td, "a.jsonl"), os.path.join(td, "b.jsonl")
        H = ["h0", "h1", "h2", "h3"]
        _write_rows(pa, rows(0, H))
        _write_rows(pb, rows(100, H))          # same positions, different global steps

        rc, out = run(pa, pb, "pos")
        ok = rc == 0 and "(4/4 keys bit-identical)" in out
        print(f"  [{'ok' if ok else 'FAIL'}] pos-align ignores the global step counter "
              f"(rc={rc})")
        bad += not ok

        rc, out = run(pa, pb, "step")
        ok = rc == 2 and "no common" in out
        print(f"  [{'ok' if ok else 'FAIL'}] step-align cannot compare shifted dumps (rc={rc}) "
              f"-- so the mode is not cosmetic")
        bad += not ok

        _write_rows(pb, rows(100, ["h0", "h1", "h2", "MUT"]))
        rc, out = run(pa, pb, "pos")
        ok = rc == 1 and "3/4" in out
        print(f"  [{'ok' if ok else 'FAIL'}] a single mutated row is caught and named (rc={rc})")
        bad += not ok

        # A key may hold several records (measured: ~50 MTP keys per request hold two records with
        # DIFFERENT hashes). They are compared as a multiset, so two cases must not be confused:
        # same multiset on both sides = identical; different record COUNT = structural, not a
        # numeric difference.
        dup_same = {"step": 9, "token_idx": 0, "n_tokens": 1, "ctx_type": "DEF", "pmax": 207,
                    "row_fnv1a64": "h0", "argmax_token": 7, "top": [{"t": 1, "v": 1.0}]}
        dup_other = {**dup_same, "step": 10, "row_fnv1a64": "hOTHER"}
        _write_rows(pa, rows(0, H, extra=[dup_same]))
        _write_rows(pb, rows(100, H, extra=[dup_same]))
        rc, out = run(pa, pb, "pos")
        ok = rc == 0 and "keys_with_more_than_one_record=1" in out and "4/4" in out
        print(f"  [{'ok' if ok else 'FAIL'}] a repeated key is compared as a multiset, and only "
              f"because it is repeated is it reported (rc={rc})")
        bad += not ok

        _write_rows(pb, rows(100, H, extra=[dup_other]))
        rc, out = run(pa, pb, "pos")
        ok = rc == 1 and "differing=1" in out
        print(f"  [{'ok' if ok else 'FAIL'}] same multiset SIZE but different hashes is a numeric "
              f"difference, not a structural one (rc={rc})")
        bad += not ok

        _write_rows(pa, rows(0, H, extra=[dup_same, dup_other]))   # 3 records vs 1
        _write_rows(pb, rows(100, H, extra=[dup_same]))
        rc, out = run(pa, pb, "pos")
        ok = rc == 1 and "record_count_mismatch=1" in out
        print(f"  [{'ok' if ok else 'FAIL'}] a key with a different number of records is reported "
              f"as structural, not silently dropped (rc={rc})")
        bad += not ok

    print(f"cgc_logits_oracle_compare selftest: {6 - bad}/6 passed")
    return 0 if bad == 0 else 1


def main():

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="oracle A JSONL (baseline)")
    ap.add_argument("--b", required=True, help="oracle B JSONL (candidate)")
    ap.add_argument("--report", default=None, help="optional diff report JSON output path")
    ap.add_argument("--fail-on", choices=("both", "numeric", "decision"), default="both",
                    help="which metric(s) may fail the run. 'numeric' reproduces the old "
                         "lenient rule where row-hash identity alone was a PASS.")
    ap.add_argument("--align", choices=("step", "pos"), default="step",
                    help="'step' (default): (step, token_idx, ctx_type) -- only valid when both "
                         "dumps have the same evaluation sequence (one request, one knob). "
                         "'pos': (ctx_type, n_tokens, pmax, token_idx) -- for dumps whose geometry "
                         "differs (e.g. a reused prefix vs a full prefill).")
    ap.add_argument("--selftest", action="store_true")
    # Before parse_args: --a/--b are required, and a selftest has no oracle to point them at.
    if "--selftest" in sys.argv:
        return selftest()
    args = ap.parse_args()

    if args.align == "pos":
        a, na = _load_oracle_pos(args.a)
        b, nb = _load_oracle_pos(args.b)
        for tag, n in (("A", na), ("B", nb)):
            print(f"align=pos {tag}: rows={n['rows']} keys={n['keys']} "
                  f"keys_with_more_than_one_record={n['repeated_keys']} "
                  f"rows_in_those={n['rows_in_repeated_keys']} ({n['rule']})")
        n_common = len(set(a) & set(b))
        n_multiset_eq = 0
        n_len_diff = 0
        pos_diffs = []
        for key in sorted(set(a) & set(b)):
            ha = sorted(r.get("row_fnv1a64") for r in a[key])
            hb = sorted(r.get("row_fnv1a64") for r in b[key])
            if len(ha) != len(hb):
                n_len_diff += 1
                if len(pos_diffs) < 5:
                    pos_diffs.append({"key": list(key), "n_a": len(ha), "n_b": len(hb),
                                      "structural": True})
                continue
            if ha == hb:
                n_multiset_eq += 1
            elif len(pos_diffs) < 5:
                pos_diffs.append({"key": list(key), "a_only": sorted(set(ha) - set(hb)),
                                  "b_only": sorted(set(hb) - set(ha)), "structural": False})
        print(f"align=pos multiset: keys_common={n_common} bit_identical={n_multiset_eq} "
              f"differing={n_common - n_multiset_eq - n_len_diff} "
              f"record_count_mismatch={n_len_diff} "
              f"(keys only in A: {len(set(a) - set(b))}, only in B: {len(set(b) - set(a))})")
        if pos_diffs:
            print("align=pos examples (first 5):")
            for ex in pos_diffs:
                print(f"  {ex}")
        print("=" * 74)
        ok = (n_common > 0 and n_multiset_eq == n_common)
        print(f"VERDICT (pos multiset) : {'PASS' if ok else 'FAIL'} "
              f"({n_multiset_eq}/{n_common} keys bit-identical)")
        if args.report:
            json.dump({"a_path": args.a, "b_path": args.b, "align": "pos",
                       "n_common": n_common, "bit_identical": n_multiset_eq,
                       "differing": n_common - n_multiset_eq - n_len_diff,
                       "record_count_mismatch": n_len_diff,
                       "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)),
                       "notes": {"a": na, "b": nb}, "examples": pos_diffs},
                      open(args.report, "w"), indent=2, ensure_ascii=False)
        return 0 if ok else 1

    a = _load_oracle(args.a)
    b = _load_oracle(args.b)
    na = nb = None

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
        "align": args.align,
        "align_notes": {k: v for k, v in (("a", na), ("b", nb)) if v is not None},
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
            # Identify the binary at write time. This product compares two dumps but, until now,
            # said nothing about WHICH build produced them -- 51 of the 186 capability products the
            # timeline reads were unplaceable for exactly this reason, and a comparison whose
            # binary is unknown cannot be quoted against any other reading. See
            # scripts/check/engine_identity.py (no retrofitting: only new runs are stamped).
            _spec = importlib.util.spec_from_file_location(
                "engine_identity", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "engine_identity.py"))
            _ei = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_ei)
            json.dump(_ei.stamp(summary), f, indent=2, ensure_ascii=False)
            f.write("\n")

    def pct(e, n):
        return f"{100.0 * e / n:.1f}%" if n else "n/a"

    print("=" * 74)
    print(f"oracle A: {args.a}  ({len(a)} entries)")
    print(f"oracle B: {args.b}  ({len(b)} entries)")
    print(f"common: {n_common}    only_A: {len(only_a)}    only_B: {len(only_b)}")
    if n_common == 0:
        print(f"FAIL: no common {ALIGN_KEYS[args.align]} keys to compare")
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
