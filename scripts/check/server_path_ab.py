#!/usr/bin/env python3
"""Server-path A/B: same harness, same box, two llama-server builds.

Why this exists
---------------
The p2/bit-identical branch quotes decode 31.5 tok/s from a *server* run
(short prompt, hot expert cache, MTP accept 100%). Our delivery number (12.57) is a
*llama-bench* number at depth 512. Those are not the same measurement, so no amount of
arguing about either one settles "is the p2 kernel 2.5x faster".

This script removes the harness and the box from the comparison: it launches each
build's own `scripts/run_server.sh`, drives it with `decode_bench.py` (byte-identical to
the copy the p2 branch imported), and interleaves the arms. What is left is the binary.

Two things it deliberately does NOT do
--------------------------------------
* It does not install, purge or close anything. If the memory gate refuses, the fix is an
  operator action (see window_sentinel.PROCEDURE) and this script stops.
* It does not call the two numbers comparable. A short-context hot-cache number stays a
  short-context hot-cache number; only the *ratio* between arms is portable.

Usage
    python3 scripts/check/server_path_ab.py --dry-run
    python3 scripts/check/server_path_ab.py --arms ours --tag a --rounds 5 --warmup 10
    python3 scripts/check/server_path_ab.py --arms ours,p2 --pairs 1 --cool 150

Cells
    short : the p2 cell. ~30-token prompt, n_predict 160. Small KV + hot pool.
    long  : ~512-token prompt, n_predict 160. Approximates the llama-bench -d 512 cell.
            The server reports the true `prompt_n`, so no tokenisation guess is needed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "scripts" / "check"
sys.path.insert(0, str(CHECK))

import thermal_pressure as tp  # noqa: E402
import window_sentinel as ws  # noqa: E402

P2_ROOT = Path("/Users/alexchuang/Documents/flashkv-p2bitident")

ARMS = {
    "ours": {"script": ROOT / "scripts" / "run_server.sh", "model_root": None},
    "p2":   {"script": P2_ROOT / "scripts" / "run_server.sh",
             "model_root": ROOT / "models" / "gguf"},   # that worktree ships no model
}

SHORT_PROMPT = "請用繁體中文簡短說明巴黎為什麼是法國的首都。"
# ~512 tokens of prompt: the -d 512 cell, expressed the only way a chat endpoint allows.
LONG_PROMPT = ("下面是一段需要你記住的長文本，請在讀完之後回答最後的問題。" * 46) + "請問這段文本的主題是什麼？"

PORT = 8080
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT


def profile_env():
    """The delivery profile's env, imported (not retyped) so A/B stays one source of truth."""
    import llama_bench_matrix as lbm
    env = dict(os.environ)
    for k, v in (lbm.resolve("prefill250", {}).get("env") or {}).items():
        if v not in (None, ""):
            env[k] = str(v)
    return env


def server_pids():
    out = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True)
    return [p for p in out.stdout.split() if p.isdigit()]


def stop_server():
    pids = server_pids()
    if not pids:
        return 0
    for p in pids:
        subprocess.run(["kill", "-INT", p], capture_output=True)
    for _ in range(120):
        if not server_pids():
            break
        time.sleep(0.5)
    left = server_pids()
    for p in left:
        subprocess.run(["kill", "-9", p], capture_output=True)
    time.sleep(3)          # Metal buffers are not returned the instant the pid dies
    return len(pids)


def launch(arm, env, log_dir, timeout=900):
    spec = ARMS[arm]
    e = dict(env)
    if spec["model_root"]:
        e["CGC_SERVER_MODEL_ROOT"] = str(spec["model_root"])
    e["CGC_SERVER_PORT"] = str(PORT)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / ("launch_%s.log" % arm)
    started = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run(["bash", str(spec["script"]), "--detach"],
                            env=e, stdout=fh, stderr=subprocess.STDOUT,
                            timeout=timeout).returncode
    return {"rc": rc, "load_s": round(time.time() - started, 1), "log": str(log)}


def measure(tag, rounds, warmup, n_predict, prompt, out_json):
    cmd = [sys.executable, str(CHECK / "decode_bench.py"),
           "--url", URL, "--rounds", str(rounds), "--warmup", str(warmup),
           "--n-predict", str(n_predict), "--prompt", prompt,
           "--tag", tag, "--json", str(out_json)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    # Do NOT parse stdout: decode_bench prints the result json and then an "appended -> ..."
    # line, so the tail is not a single json document (2026-09-22: JSONDecodeError "Extra data").
    # It also writes the same record to --json; read the last entry from there instead.
    data = {}
    try:
        prev = json.load(open(out_json))
        if isinstance(prev, list) and prev:
            data = prev[-1]
    except Exception as exc:  # noqa: BLE001
        print("  (could not read %s: %s)" % (out_json, exc), file=sys.stderr)
    return {"rc": r.returncode, "result": data, "stdout_tail": r.stdout[-1500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sp")
    ap.add_argument("--arms", default="ours")
    ap.add_argument("--pairs", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=10,
                    help="p2's own recipe: 10 warmup runs to heat the expert pool")
    ap.add_argument("--n-predict", type=int, default=160)
    ap.add_argument("--cell", default="short", choices=("short", "long"))
    ap.add_argument("--cool", type=int, default=150)
    ap.add_argument("--override", action="store_true",
                    help="proceed even when the memory gate refuses (still reported)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            ap.error("unknown arm %s (have %s)" % (a, ",".join(ARMS)))
    prompt = SHORT_PROMPT if args.cell == "short" else LONG_PROMPT

    out_dir = ROOT / "Backup" / "phase_decomp" / "L3" / ("spab_%s" % args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / ("summary_%s.json" % args.cell)

    print("cell=%s  arms=%s  pairs=%d  rounds=%d  warmup=%d  n_predict=%d"
          % (args.cell, ",".join(arms), args.pairs, args.rounds, args.warmup, args.n_predict))
    print("prompt tokens (chars): %d" % len(prompt))

    if args.dry_run:
        print("dry-run: would launch %s" % [str(ARMS[a]["script"]) for a in arms])
        return 0

    env = profile_env()
    results = []
    for pair in range(args.pairs):
        order = arms if pair % 2 == 0 else list(reversed(arms))   # ABBA
        for arm in order:
            ok, why, det = ws.mem_gate()
            th = tp.stamp()
            print("\n--- pair %d arm %s | gate %s | thermal %s"
                  % (pair, arm, "PASS" if ok else "REFUSE", th["label"]))
            if not ok:
                print(ws.mem_report(det))
                if not args.override:
                    print("refusing (use --override to run anyway and label the numbers)")
                    json.dump(results, open(summary_json, "w"), indent=1, ensure_ascii=False)
                    return 2
                print("OVERRIDE: running anyway -- absolute numbers are not quotable")
            rec = {"pair": pair, "arm": arm, "cell": args.cell,
                   "gate_ok": ok, "gate_why": why, "thermal_launch": th,
                   "mem_launch": det.get("mem"), "swap_launch": det.get("swap")}

            L = launch(arm, env, out_dir)
            rec["launch"] = L
            if L["rc"] != 0:
                print("  launch FAILED rc=%d (%s s) see %s" % (L["rc"], L["load_s"], L["log"]))
                rec["failed"] = "launch"
                results.append(rec)
                json.dump(results, open(summary_json, "w"), indent=1, ensure_ascii=False)
                continue
            print("  server up in %.1f s" % L["load_s"])

            try:
                M = measure("%s-%s-p%d" % (args.tag, arm, pair), args.rounds, args.warmup,
                            args.n_predict, prompt, out_dir / "rounds.json")
            except Exception as exc:  # noqa: BLE001  -- the server must still be torn down
                print("  measure FAILED: %s" % exc, file=sys.stderr)
                rec["failed"] = "measure: %s" % exc
                M = {"rc": -1, "result": {}}
            rec["measure"] = {"rc": M["rc"], "result": M["result"]}
            r = M["result"]
            if r:
                print("  decode median %.2f t/s  (min %.2f max %.2f, n=%s)  prefill %.1f t/s"
                      % (r.get("decode_tps_median", 0), r.get("decode_tps_min", 0),
                         r.get("decode_tps_max", 0), r.get("n_rounds"),
                         r.get("prefill_tps_median", 0)))
                print("  thermal_worst=%s  answer_stable=%s  predicted_n=%s"
                      % ((r.get("thermal_worst") or {}).get("label"), r.get("answer_stable"),
                         (r.get("rounds") or [{}])[0].get("predicted_n")))

            n = stop_server()
            if server_pids():                      # SIGINT was not enough; say so loudly
                print("  WARNING: %s llama-server pid(s) still alive after stop_server()"
                      % len(server_pids()))
            rec["stopped_pids"] = n
            results.append(rec)
            json.dump(results, open(summary_json, "w"), indent=1, ensure_ascii=False)
            if args.cool and (pair < args.pairs - 1 or arm != order[-1]):
                print("  cooling %d s" % args.cool)
                time.sleep(args.cool)

    print("\n=== summary (%s) ===" % summary_json)
    by = {}
    for r in results:
        if r.get("failed"):
            continue
        by.setdefault(r["arm"], []).append(r["measure"]["result"].get("decode_tps_median", 0))
    for arm, v in by.items():
        print("  %-6s medians %s  -> %.2f t/s" % (arm, v, sorted(v)[len(v) // 2]))
    if "ours" in by and "p2" in by:
        a = sorted(by["ours"])[len(by["ours"]) // 2]
        b = sorted(by["p2"])[len(by["p2"]) // 2]
        if a:
            print("  ratio p2/ours = %.2fx" % (b / a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
