#!/usr/bin/env python3
"""commit_bench.py -- the pre-commit production cell: prefill 2048 + decode, llama-bench 口徑.

[2026-09-23] 每次 commit 前必跑：證明新的 build 沒有把 production shape 弄壞。
形狀固定為「完整側」：2048-token prefill（-p 2048），接著 128-token decode（-n 128），
context 預置 512（-d 512）。profile 固定 prod-new（MTP off + prefill250 支柱）。
[2026-09-25] decode 加 --warm-skip 64，與測試卡 §2 權威口徑一致（docs/PROD_NEW_TEST_CARD_2026-09-24.md）：
    commit_bench 量的是穩定態（池已暖）；冷啟動代價另測 —— pool 4G 下無 warm-skip 是 -33%
    （docs/POOL_SWEET_SPOT_2026-09-25.md §2/§5）。生產冷啟動由 run_server.sh 的預熱請求解決。

判定（可配）：
  prefill t/s >= --prefill-min （預設 120；prod-new 目標 250+，但機器狀態波動大，閘門保守）
  decode  t/s >= --decode-min （預設 10；prod-new 目標 13-14）
任一未達 → rc=1（commit 被擋）；加 --record-only 只記錄不擋。

測量口徑與 ABBA_MEASUREMENT_PROTOCOL 一致：llama-bench 標準、每 rep 完整 JSON、
build 指紋由 llama_bench_matrix 自帶。冷卻由調用方負責（跑在 ABBA 協議下）。

用法：
    python3 scripts/check/commit_bench.py                     # 跑 prod-new p2048+d512
    python3 scripts/check/commit_bench.py --dry-run           # 只印命令
    python3 scripts/check/commit_bench.py --record-only       # 記錄不擋
    python3 scripts/check/commit_bench.py --prefill-min 200 --decode-min 12
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

PROFILE = "prod-new"          # MTP off + prefill250 支柱（§EN-473：decode 13-14 / prefill 250+）
PROMPT = "2048"               # 完整側：2048-token prefill
GEN = "128"                   # 接著 128-token decode
DEPTHS = "512"                # context 預置 512（decode 在長一點的 context 上量）
REPS = 3


def run_matrix(profile, prompt, gen, depths, reps, workdir, json_out, dry_run):
    cmd = [sys.executable, os.path.join(HERE, "llama_bench_matrix.py"),
           "--arms", profile,
           "--prompt", prompt, "--gen", gen, "--depths", depths,
           "--reps", str(reps),
           "--ctx-size", str(0),
           "--warm-skip", "64",
           "--workdir", workdir,
           "--json", json_out]
    if dry_run:
        cmd.append("--dry-run")
    print(" ".join(cmd), flush=True)
    if dry_run:
        return None
    os.makedirs(workdir, exist_ok=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"[commit_bench] llama_bench_matrix rc={r.returncode}", file=sys.stderr)
        return None
    return json.load(open(json_out)) if os.path.exists(json_out) else None


def extract(results):
    """From the matrix json (list of run_arm dicts), find the pp(2048) row and the tg(d512) row.

    Fixed 2026-09-23: llama_bench_matrix emits run_arm dicts with a `rows` list of llama-bench
    rows (flat n_prompt/n_gen/n_depth/avg_ts), NOT the params.test structure this previously
    expected -- so prefill/decode were never matched and the summary crashed after a PASSING
    run (`Unknown format code 'f' for object of type 'str'` when a wrong row got picked).
    """
    prefill = decode = None
    if not results:
        return prefill, decode
    for r in results:
        for row in r.get("rows", []):
            np_ = int(row.get("n_prompt", 0))
            ng_ = int(row.get("n_gen", 0))
            if np_ == int(PROMPT) and ng_ == 0:
                prefill = row
            # [2026-09-25] tg row 匹配：warm-skip N 時 llama-bench 輸出 n_gen = GEN - N（計時 token），
            # 固定 ng_==GEN 會匹配不到 → decode=None → summary 崩/假 MISSING。改 np_==0 && ng_>0。
            if np_ == 0 and ng_ > 0:
                decode = row
    # [2026-09-25] avg_ts / samples_ts 可能是 str（llama-bench json 未轉型）→ 統一轉 float，
    # 否則 summary 的 :.2f 會崩（history: 兩個 PASSING run 都掛在 print）。
    for row in (prefill, decode):
        if row is None:
            continue
        if isinstance(row.get("avg_ts"), str):
            try:
                row["avg_ts"] = float(row["avg_ts"])
            except (TypeError, ValueError):
                pass
        st = row.get("samples_ts")
        if isinstance(st, list):
            row["samples_ts"] = [float(x) for x in st if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).replace('-', '', 1).isdigit() or True]
    return prefill, decode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--gen", default=GEN)
    ap.add_argument("--depths", default=DEPTHS)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--prefill-min", type=float, default=120.0)
    ap.add_argument("--decode-min", type=float, default=10.0)
    ap.add_argument("--record-only", action="store_true",
                    help="record the numbers but do not fail the commit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workdir", default="/tmp/commit_bench")
    ap.add_argument("--json", default="/tmp/commit_bench_result.json")
    args = ap.parse_args()

    results = run_matrix(args.profile, args.prompt, args.gen, args.depths, args.reps,
                         args.workdir, args.json, args.dry_run)
    if args.dry_run:
        return 0
    if results is None:
        print("[commit_bench] FAILED to run (see llama_bench_matrix output)", file=sys.stderr)
        return 1

    prefill, decode = extract(results)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"\n=== commit_bench {stamp} ===")
    print(f"profile={args.profile}  shape: p{args.prompt} n{args.gen} d{args.depths}  reps={args.reps}")
    if prefill:
        print(f"prefill: {prefill.get('avg_ts', 0):7.2f} t/s   samples="
              f"{[f'{x:.2f}' for x in prefill.get('samples_ts', [])]}")
    if decode:
        print(f"decode:  {decode.get('avg_ts', 0):7.2f} t/s   samples="
              f"{[f'{x:.2f}' for x in decode.get('samples_ts', [])]}")

    ok = True
    if prefill is None or float(prefill.get("avg_ts", 0)) < args.prefill_min:
        pv_ = "MISSING" if prefill is None else f"{float(prefill.get('avg_ts', 0)):.2f}"
        print(f"[commit_bench] prefill {pv_} < {args.prefill_min}  {'(record-only)' if args.record_only else 'FAIL'}")
        ok = False
    if decode is None or float(decode.get("avg_ts", 0)) < args.decode_min:
        dv_ = "MISSING" if decode is None else f"{float(decode.get('avg_ts', 0)):.2f}"
        print(f"[commit_bench] decode {dv_} < {args.decode_min}  {'(record-only)' if args.record_only else 'FAIL'}")
        ok = False

    # [2026-09-23] commit 標題必須帶量測成績：把這行貼進 commit message（subject 或 body 首行）。
    # 格式固定，供 commit_gates / 事後 grep 追溯（"數字跟東西對不上" 的歷史禁止重演）。
    # [2026-09-23 修復] decode 成績必須自帶 MTP 口徑：prod-new 預設 MTP off，
    # 但 off/on 不可混讀（off 12.19 vs on 11.35 是同級差距）；從結果 JSON 的
    # spec_type 判（None=off，draft-mtp=on），缺省時 fail-closed 標 off。
    pv = prefill.get("avg_ts", 0) if prefill else 0.0
    dv = decode.get("avg_ts", 0) if decode else 0.0
    try:
        spec_type = results[0].get("spec_type") if results else None
    except Exception:
        spec_type = None
    mtp_tag = "mtp=on" if spec_type else "mtp=off"
    bench_line = (f"prefill={pv:.2f}t/s decode={dv:.2f}t/s({mtp_tag}) "
                  f"(commit_bench {args.profile} p{args.prompt} n{args.gen} d{args.depths} r{args.reps})")
    print(f"\n[commit-gate] {bench_line}")
    print(f"[commit-gate] 標題範例: perf(server): <你的改動>（{bench_line}）")

    if ok or args.record_only:
        print("[commit_bench] PASS" if ok else "[commit_bench] FAIL (record-only, not blocking)")
        return 0
    print("[commit_bench] FAIL -- commit blocked")
    return 1


if __name__ == "__main__":
    sys.exit(main())
