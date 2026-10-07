#!/usr/bin/env python3
"""commit_bench.py -- the pre-commit production cell: llama-bench 出廠形狀（pp512 / tg128）.

[2026-09-23] 每次 commit 前必跑：證明新的 build 沒有把 production shape 弄壞。
[2026-10-03] **operator 裁定：形狀改為 llama-bench 出廠值**（移除所有具名格之後，測試卡 §2.5
    的預設 cell 就是這一組）：`-p 512 -n 128 -b 2048 -ub 512 -d 0 -r 5`、warmup 開、warm_skip 0。
    出處：`src/llama.cpp/tools/llama-bench/llama-bench.cpp:392-439` 的 `cmd_params_defaults`。
    舊口徑（`-p 2048 -d 512 -r 3 --warm-skip 64`）＝**已退役形狀**，與新讀數不可互比。
    ⚠ `--warm-skip` 曾是「量穩定態（池已暖）」的手段；llama-bench 沒有這個概念，改成靠它自己的
    warmup run（預設開）。冷啟動代價的舊量測見 docs/POOL_SWEET_SPOT_2026-09-25.md §2/§5。
    profile 仍固定 prod-new（記憶體側不動：池 8 GiB／KV q8_0／ngl 99／load_mode none）。

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

# [CGC 2026-10-07] The acceptance ledger. `Backup/commit_bench/` was EMPTY while 97 readings sat
# in Backup/spec_tree/, so "what can this build actually run, under the acceptance caliber?" took
# a 97-file scan plus a hand judgement about `attribution` every single time (2026-10-07: only
# 3 of 97 were quotable). `attribution` was never missing from the artefacts -- llama_bench_matrix
# has always written it -- it was simply never read out and never recorded centrally.
# The repo rule this serves: 缺口從來不是「沒有檔案」，是沉默. A gate that blocks a commit but
# leaves no quotable record is a gate whose verdict can only be remembered, and remembered
# verdicts are what this repo has been fighting for three months.
LEDGER_DIR = os.path.join(ROOT, "Backup", "commit_bench")
LEDGER = os.path.join(LEDGER_DIR, "ledger.jsonl")

PROFILE = "prod-new"          # MTP off + prefill250 支柱（§EN-473：decode 13-14 / prefill 250+）
# [CGC 2026-10-03] 形狀＝**llama-bench 出廠**（測試卡 §2.5 預設 cell，operator 裁定）。
# 出處：`src/llama.cpp/tools/llama-bench/llama-bench.cpp:392-439` 的 `cmd_params_defaults`：
# `-p 512 -n 128 -b 2048 -ub 512 -d 0 -r 5`、warmup 開、warm_skip 0（llama-bench 沒有 warm-skip）。
# ⚠ 這裡與 §2.5 不一致 ⇒ `cell_contract` fail-closed 拒跑（單一口徑），所以兩邊要一起改。
PROMPT = "512"
GEN = "128"
DEPTHS = "0"
REPS = 5
WARM_SKIP = 0


def run_matrix(profile, prompt, gen, depths, reps, workdir, json_out, dry_run):
    cmd = [sys.executable, os.path.join(HERE, "llama_bench_matrix.py"),
           "--arms", profile,
           "--prompt", prompt, "--gen", gen, "--depths", depths,
           "--reps", str(reps),
           "--ctx-size", str(0),
           # [CGC 2026-10-03] warm-skip 0：llama-bench **沒有** warm-skip 這個概念，而本卡的預設格
           # 已貼齊它的出廠形狀 ⇒ 這一項留 64 會讓 tg 那一列量的是「第 65–128 顆」，不是 tg128。
           "--warm-skip", str(WARM_SKIP),
           "--workdir", workdir,
           "--json", json_out]
    if dry_run:
        cmd.append("--dry-run")
    print(" ".join(cmd), flush=True)
    if dry_run:
        return None
    os.makedirs(workdir, exist_ok=True)
    # 與 harness 共用同一把 GPU 原子鎖：commit 測量也不能和其他實驗並發
    sys.path.insert(0, str(HERE))
    import harness
    try:
        with harness.gpu_window_lock(held_by="commit_bench " + profile, timeout_s=1800):
            r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "CGC_INTERNAL_CALL": "1"})
    except TimeoutError as e:
        print(f"[commit_bench] {e}", file=sys.stderr)
        return None
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


def record_ledger(entry):
    """Append one acceptance reading to Backup/commit_bench/ledger.jsonl.

    WHY FAIL-CLOSED ON `attribution`. The 2026-10-07 scan of 97 readings found only 3 quotable:
    34 were `both` (thermal AND swap over limit), 23 `swap`, 14 `thermal`, and 23 had the field
    MISSING entirely. A gate that only records the number would have banked all 97 as equally
    real. So the ledger records the verdict alongside the reading, and `quotable` is computed
    here rather than left to whoever reads it later -- `attribution` absent is NOT clean, it is
    unverified, and it is graded as unquotable for the same reason drift_adjudicate fail-closes
    on a missing sidecar field.

    Append-only on purpose: overwriting would destroy the history that makes a regression visible,
    and this repo's own convention is that a lost reading is indistinguishable from a reading that
    never happened.
    """
    os.makedirs(LEDGER_DIR, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return LEDGER


def quotable(attribution):
    """`attribution.verdict == "none"` and nothing else. Missing field => NOT quotable."""
    if not isinstance(attribution, dict):
        return False, "attribution field absent => unverified, not clean"
    v = attribution.get("verdict")
    if v == "none":
        return True, attribution.get("why", "")
    return False, attribution.get("why") or f"verdict={v}"


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

    # [CGC 2026-10-07] Record BEFORE the pass/fail return, so a FAILING run is banked too -- a
    # regression is exactly the reading you need later and it is exactly the one a gate that only
    # writes on success throws away. `attribution` decides quotability, NOT the thresholds: a run
    # can clear both thresholds on a swap-tainted window and still be unusable as evidence.
    try:
        _arm = (results[0] if isinstance(results, list) and results else results) or {}
    except Exception:
        _arm = {}
    _attrib = _arm.get("attribution")
    _q, _why = quotable(_attrib)
    _cell = _arm.get("cell") or {}
    _entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stamp": stamp,
        "profile": args.profile,
        "cell": {"prompt": _cell.get("prompt", args.prompt), "gen": _cell.get("gen", args.gen),
                 "depths": _cell.get("depths", args.depths), "reps": _cell.get("reps", args.reps),
                 "warm_skip": _cell.get("warm_skip"), "named_cell": _cell.get("named_cell"),
                 "contract_ok": (_arm.get("contract") or {}).get("ok")},
        "prefill_ts": pv, "decode_ts": dv, "mtp": mtp_tag,
        "prefill_samples": (prefill or {}).get("samples_ts"),
        "decode_samples": (decode or {}).get("samples_ts"),
        "thresholds": {"prefill_min": args.prefill_min, "decode_min": args.decode_min},
        "gate_pass": bool(ok),
        "attribution": _attrib,
        "quotable": _q,
        "quotable_why": _why,
        "record_only": bool(args.record_only),
        "bench_line": bench_line,
    }
    _path = record_ledger(_entry)
    print(f"[commit_bench] ledger: {os.path.relpath(_path, ROOT)}  "
          f"(quotable={'yes' if _q else 'NO'} — {_why})")
    if not _q:
        print("[commit_bench] ⛔ 這筆不可引用：數字是真的，但視窗不合格（熱態／swap）。"
              "要引用請先降溫或等 swap 回落。")

    if ok or args.record_only:
        print("[commit_bench] PASS" if ok else "[commit_bench] FAIL (record-only, not blocking)")
        return 0
    print("[commit_bench] FAIL -- commit blocked")
    return 1


if __name__ == "__main__":
    sys.exit(main())
