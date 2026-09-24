#!/usr/bin/env python3
"""Interleaved A/B driver: alternate arms A,B,A,B,... so that each pair is measured back to
back inside one thermal / page-cache window.

Why this exists instead of `decode_sweep.py --arms A,B`: the sweep runs each arm once and keys
its rows by tag (a second row for the same tag is skipped), so it structurally cannot express
"3 interleaved rounds". It also reports an unpaired median-of-medians, which mixes the between-arm
difference with whatever drift happened between the two launches. Here:

  - the row key is `<arm>#r<rep>`, so 3 reps are recorded instead of 1;
  - the headline number is the **paired** per-rep ratio B_r / A_r (median of 3), not the ratio of
    two medians. Launch-to-launch drift cancels inside each pair;
  - every row carries the build fingerprint (md5 of llama-server + libggml-metal), so a result can
    never be silently attributed to a different binary;
  - `answer_md5_set` is carried through: a ceiling probe like CGC_SUBMIT_AHEAD is EXPECTED to
    change the md5. If it does not, the flag never reached the process and the speedup is a lie.

Usage:
    python3 scripts/check/ab_interleave.py \
        --arms p25-gputime,p25-submit-ahead --reps 3 --profile prod25 \
        --rounds 3 --warmup 1 --n-predict 120 \
        --json Backup/phase_decomp/ab_submit_ahead.json
"""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

_spec = importlib.util.spec_from_file_location("decode_sweep", os.path.join(HERE, "decode_sweep.py"))
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)


def build_fingerprint():
    """Delegates to `decode_sweep.build_fingerprint` -- deliberately, because a second copy of
    the rule is a second place for it to drift, and this one already has: the copy that used to
    live here hashed the same hand-picked three files (llama-server + libggml-metal + libllama)
    and therefore shared with the sweep both the blindness to libggml-base and the exact
    comparability claim that blindness produced. An interleave's whole purpose is "rows in this
    file are comparable", so it must not own a divergent definition of comparable.
    See lessons.jsonl eng-mh-0007."""
    return ds.build_fingerprint()


def others_measuring():
    """True if any foreign llama-bench / llama-server process is running.

    [CGC 2026-09-23 §EN-473] Two sessions measuring at once is exactly how the
    'clean window' turned out to be someone else's run -- the swap watermark and the
    timing both change while we are between reps. Pollution is a hard gate (rc=2),
    not a covariate: it is not something to fit out of the data."""
    out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
    for name in ("llama-bench", "llama-server"):
        for line in out.splitlines():
            if name in line and "grep" not in line and "ab_interleave" not in line:
                return True
    return False


def ensure_idle(json_path, min_idle_s):
    """Enforce a deep-cooldown gate between launches.

    [CGC 2026-09-23 §EN-473] 'NOMINAL' is not a point on this fanless M4 Air: the same
    anchor binary read 12.37 t/s when launched after a 10-minute GPU idle and 9.9-10.5
    after only 3 minutes. Thermal is continuous below the NOMINAL label, so the gate is
    wall-clock idle since the last measurement, not the label. The json's mtime is the
    last write of a completed row, i.e. the end of the previous arm."""
    if not min_idle_s:
        return
    if os.path.exists(json_path):
        elapsed = time.time() - os.path.getmtime(json_path)
        if elapsed < min_idle_s:
            wait = min_idle_s - elapsed
            print(f"[cooldown] last arm ended {elapsed:.0f}s ago; waiting {wait:.0f}s "
                  f"(min-idle {min_idle_s}s)", flush=True)
            time.sleep(wait)
    else:
        print(f"[cooldown] no previous run on file; assuming cold start", flush=True)


def run_arm(arm, key, profile, rounds, warmup, n_predict):
    ds.killed()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ctrl = os.path.join(ds.LOG_DIR, f"ab_{key}_{stamp}.ctrl.log")
    if not ds.start(ds.ARMS[arm], ctrl, profile):
        print(f"[FAIL] {key}: server never became ready; see {ctrl}", flush=True)
        return None
    srv_log = ds.server_log_of(ctrl)
    print(f"  ready; srv={srv_log}", flush=True)

    bench_json = f"/tmp/ab_interleave_bench_{key}.json"
    if os.path.exists(bench_json):
        os.remove(bench_json)
    subprocess.run([sys.executable, "scripts/check/decode_bench.py",
                    "--rounds", str(rounds), "--warmup", str(warmup),
                    "--n-predict", str(n_predict), "--tag", key, "--json", bench_json],
                   cwd=ROOT)
    bench = json.load(open(bench_json))[-1] if os.path.exists(bench_json) else {}

    # SIGINT (not -9) so the teardown counters (final stats / miss attribution / read shape) land.
    subprocess.run(["pkill", "-INT", "-f", ds.SERVER_MATCH],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        time.sleep(2)
        if srv_log and os.path.exists(srv_log) and \
                "final stats" in open(srv_log, errors="replace").read():
            break
    subprocess.run(["pkill", "-9", "-f", ds.SERVER_MATCH],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # [CGC 2026-09-23 §EN-473] pkill -9 returns BEFORE the process is gone; the next arm's
    # others_measuring() then sees our own dying server as "foreign" and aborts the interleave
    # (hit twice today: p25-gputime#r1 and p25-nail-mtpoff#r1 both blocked their mtp partner).
    # Wait until the match set is empty before returning.
    for _ in range(30):
        if subprocess.run(["pgrep", "-f", ds.SERVER_MATCH],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
            break
        time.sleep(2)
    else:
        print(f"[teardown] server still alive 60s after SIGKILL; continuing anyway", flush=True)

    row = {"key": key, "arm": arm, "env": ds.ARMS[arm], "log": srv_log}
    row.update({k: v for k, v in bench.items() if k != "tag"})
    row["loopiness"] = ds.loopiness(bench.get("sample", "") or "")
    row.update(ds.harvest(srv_log))
    return row


def report(rows):
    fp = build_fingerprint()
    fps = {json.dumps(r.get("build", {}), sort_keys=True) for r in rows}
    print("\nbuild fingerprint now:", fp)
    print(f"distinct fingerprints across rows: {len(fps)}"
          + ("  <-- OK (single binary for every rep)" if len(fps) == 1 else
             "  <-- WARNING: a rebuild happened mid-sweep, pairs are not comparable"))

    arms = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])

    print(f"\n{'key':22s} {'decode':>7s} {'min':>7s} {'prefill':>8s} {'hit%':>6s} "
          f"{'miss':>7s} {'loop':>5s} {'stable':>6s}  md5set")
    for r in rows:
        print(f"{r['key']:22s} {r.get('decode_tps_median', 0):7.2f} "
              f"{r.get('decode_tps_min', 0):7.2f} {r.get('prefill_tps_median', 0):8.2f} "
              f"{r.get('hit_rate_pct', 0):6.1f} {r.get('misses', 0):7d} "
              f"{r.get('loopiness', 0):5.2f} {str(r.get('answer_stable')):>6s}  "
              f"{','.join(r.get('answer_md5_set', []) or [])}")

    print()
    stats = {}
    for a in arms:
        v = [r["decode_tps_median"] for r in rows
             if r["arm"] == a and r.get("decode_tps_median")]
        if v:
            stats[a] = v
            print(f"{a:22s} median {statistics.median(v):7.2f} t/s   "
                  f"min {min(v):7.2f}  max {max(v):7.2f}  n={len(v)}")

    if len(arms) == 2 and all(a in stats for a in arms):
        base, treat = arms
        by_rep = {}
        for r in rows:
            if r.get("decode_tps_median"):
                by_rep.setdefault(r["key"].split("#r")[-1], {})[r["arm"]] = r["decode_tps_median"]
        ratios = []
        print(f"\npaired per-rep ratio ({treat} / {base}):")
        for rep in sorted(by_rep):
            pair = by_rep[rep]
            if base in pair and treat in pair and pair[base] > 0:
                rt = pair[treat] / pair[base]
                ratios.append(rt)
                print(f"  rep {rep}: {pair[base]:7.2f} -> {pair[treat]:7.2f}   "
                      f"x{rt:.3f}   (delta {pair[treat] - pair[base]:+.2f} t/s)")
        if ratios:
            m = statistics.median(ratios)
            print(f"\n  paired median x{m:.3f}  ({', '.join(f'{x:.3f}' for x in ratios)})")
            print(f"  unpaired ratio of medians x{statistics.median(stats[treat]) / statistics.median(stats[base]):.3f}")
            if len(ratios) >= 3:
                print(f"  min x{min(ratios):.3f}  max x{max(ratios):.3f}  "
                      f"all>1: {all(x > 1 for x in ratios)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="exactly two arms, base first: A,B")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--profile", default="prod25")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--n-predict", type=int, default=120)
    ap.add_argument("--json", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--min-idle-s", type=int, default=300,
                    help="deep-cooldown gate: seconds of GPU idle since the last arm "
                         "before the next launch (default 300). 0 disables. [§EN-473]")
    ap.add_argument("--no-window-check", action="store_true",
                    help="skip the foreign-process window check (only for scripted "
                         "single-session runs). [§EN-473]")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ds.ARMS:
            print(f"unknown arm {a!r}; known: {','.join(ds.ARMS)}", file=sys.stderr)
            return 2

    rows = []
    if os.path.exists(args.json):
        try:
            rows = json.load(open(args.json))
        except Exception:  # noqa: BLE001
            rows = []
    have = {r["key"] for r in rows}

    if not args.report_only:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        fp = build_fingerprint()
        print(f"build fingerprint: {fp}\narms={arms} reps={args.reps} profile={args.profile} "
              f"rounds={args.rounds} warmup={args.warmup} n_predict={args.n_predict}", flush=True)
        for rep in range(1, args.reps + 1):
            for arm in arms:
                key = f"{arm}#r{rep}"
                if key in have and not args.force:
                    print(f"[skip] {key} already recorded", flush=True)
                    continue
                if not args.report_only and not args.no_window_check and others_measuring():
                    print(f"[WINDOW] foreign llama process running; refusing to launch {key} "
                          f"(use --no-window-check only for scripted single-session runs)",
                          file=sys.stderr, flush=True)
                    return 2
                ensure_idle(args.json, args.min_idle_s)
                print(f"\n===== [{key}] arm={arm} env={ds.ARMS[arm]} =====", flush=True)
                row = run_arm(arm, key, args.profile, args.rounds, args.warmup, args.n_predict)
                if row is None:
                    continue
                row["build"] = fp
                row["rep"] = rep
                rows.append(row)
                json.dump(rows, open(args.json, "w"), ensure_ascii=False, indent=2)
                print(f"  -> decode {row.get('decode_tps_median')} t/s  "
                      f"hit {row.get('hit_rate_pct')}%  miss {row.get('misses')}  "
                      f"loop {row.get('loopiness')}  md5 {row.get('answer_md5_set')}", flush=True)

    report(rows)
    json.dump(rows, open(args.json, "w"), ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
