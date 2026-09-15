#!/usr/bin/env python3
"""Decode arm sweep: restart the server per arm, run decode_bench, harvest cache final stats.

Why a driver and not a shell loop: every arm needs the SAME request shape, the SAME pool
accounting read back, and the server has to be SIGINT'd (not SIGKILL'd) so the expert-cache
teardown prints `final stats` / `miss attribution` / `read shape`. Those three lines are the
only way to attribute decode time to (miss count) vs (per-miss IO latency) — and they are
exactly what is missing from every decode number quoted in the docs so far.

Arms are declared below. Each is a dict of extra env vars layered on the base profile.
The driver is resumable: rows accumulate into --json keyed by tag, so a partial sweep is
still useful and re-running skips nothing already recorded (unless --force).

Usage:
    python3 scripts/check/decode_sweep.py --arms baseline,mtp-off,pool-4g --rounds 5
    python3 scripts/check/decode_sweep.py --report /tmp/decode_sweep.json
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LOG_DIR = os.path.join(ROOT, "Backup", "cgc_logs")
SERVER_MATCH = "build/bin/llama-server"

# tag -> extra env. Base is CGC_SERVER_PROFILE=prefill250 for every arm so the only
# difference between arms is the variable under test.
ARMS = {
    "baseline":      {},
    "mtp-off":       {"CGC_SERVER_MTP": "0"},
    "pool-4g":       {"CGC_SERVER_EXPERT_CACHE_BYTES": "4294967296"},
    "pool-6g":       {"CGC_SERVER_EXPERT_CACHE_BYTES": "6442450944"},
    "pool-10g":      {"CGC_SERVER_EXPERT_CACHE_BYTES": "10737418240"},
    # budget 0 = expert cache off entirely: every expert is read where the loader put it, so
    # decode has no pool, no remap, no fill. This is the "zero-IO-work" upper bound and the
    # single most informative arm -- if decode does NOT go up here, the bottleneck is compute,
    # not the cache, and no amount of pool/fill tuning can reach 25 t/s.
    "nocache":       {"CGC_SERVER_EXPERT_CACHE_BYTES": "0"},
    # P1 prefill-protect. The build default is OFF and run_server.sh never set it, so every
    # server-profile decode number recorded so far is the "after a prefill" case. The code
    # comment at llama-context.cpp:4987 records 22.2 t/s steady-state with it on.
    "protect-on":    {"CGC_PREFILL_PROTECT": "1"},
    "protect-noprefetch": {"CGC_PREFILL_PROTECT": "1", "CGC_SERVER_NO_PREFETCH": "0"},
    # ---- tested on the prod25 base profile (see --profile) ----
    # `--no-mmap` makes the dense tensors anonymous pages that the OS may swap; the 09-12 note
    # in run_server.sh measured swap used 5.8GB / free 61MB as a top speed cause. This machine
    # is at 3.1GB of a 4GB swap file, so this is a live hypothesis, not a theory.
    "p25-mmap":      {"CGC_SERVER_LOAD_MODE": "mmap"},
    "p25-noasync":   {"CGC_SERVER_OA_ASYNC": "0"},
    "p25-nospac":    {"CGC_SPAC": "0"},
    "p25-mtpoff":    {"CGC_SERVER_MTP": "0"},
    "p25-mmap-mtpoff": {"CGC_SERVER_LOAD_MODE": "mmap", "CGC_SERVER_MTP": "0"},
    "p25-noident":   {"CGC_MM_BITIDENT": "0"},
    "prefetch-on":   {"CGC_SERVER_NO_PREFETCH": "0"},
    "spac-on":       {"CGC_SPAC": "1"},
    "verify-off":    {"CGC_SERVER_VERIFY_DECODE": "0"},
    "draft-off":     {"CGC_SERVER_DRAFT_DECODE": "0"},
    # ---- 2026-09-15: MTP acceptance is the whole ballgame ----
    # 09-05 (27.71 t/s) amortised 622 tokens over 156 decode steps: draft acceptance 0.9974,
    # mean len 3.99, 144 ms/step. Today prod25-mtpoff spends only 96 ms/step but yields ONE
    # token per step (10.43 t/s), and prod25 with MTP on is SLOWER still (7.6-8.0 t/s) while
    # emitting a degenerate repeat. So per-step cost did NOT regress -- MTP did.
    # The three `bit-identical` pillars were flipped from default-0 to default-1 earlier today
    # and prod25 forces all three; MTP_NO_WARMUP in particular touches the draft context's
    # init, so it is the prime suspect. Test them one at a time, plus DRY (which penalises
    # repetition on the TARGET sampler only -- a systematic target/draft mismatch).
    "p25-mtp-on":    {},                       # same-session bare prod25 reference
    "p25-pillars-off": {"CGC_SERVER_MTP_NO_WARMUP": "0",
                        "CGC_SERVER_NO_SEQ_RM_PROBE": "0",
                        "CGC_MM_BITIDENT": "0"},
    "p25-nowarmup":  {"CGC_SERVER_MTP_NO_WARMUP": "0"},
    "p25-probe-on":  {"CGC_SERVER_NO_SEQ_RM_PROBE": "0"},
    "p25-nodry":     {"CGC_SERVER_DRY_MULTIPLIER": "0"},
    # ---- the money arm ----
    # Measured on 09-15: MTP-on runs are 88.4% CAPACITY misses (68298 of 77266) over a 143-slot
    # pool, 553.8 s of pread inside a 92.6 s wall, plus a `prewarm req=22902 hit=0 miss=22902`
    # pass that is 100% wasted work and evicts 22902 slots on the way. So CAPACITY misses --
    # not compulsory traffic, not per-read latency (1941us vs 1739us on 09-05, +12%) -- are what
    # kills decode.
    #
    # CGC_PREFILL_PROTECT exists for exactly this and has NEVER been reachable from a server
    # profile: cgc_prefill_protect_on() is a bare getenv()!=nullptr test, and run_server.sh only
    # added it to the allowlist today, so every server decode number on record is the
    # "after a prefill" case. The mechanism is direct -- a decode HIT stamps
    # slot_decode_reserved[l][slot]=1 (llama-expert-cache.cpp:748-752), and pick_slot will not
    # evict a protected slot -- so it converts the hot decode working set into pinned residency.
    # The in-code measurement is 22.2 t/s steady-state with it on vs 8.1-8.9 after a prefill.
    "p25-protect":   {"CGC_PREFILL_PROTECT": "1"},
    "p25-protect-mtpoff": {"CGC_PREFILL_PROTECT": "1", "CGC_SERVER_MTP": "0"},
    # 09-05's operating point, rebuilt: 4 GiB pool (71 slots) WITH the L0/L1 tier split active.
    # run_server.sh pins SOFT_POOL_L0/L1 to 0, so the partition cannot be enabled from a profile
    # even though 09-05 ran with L0=32 L1=32. This arm exists to test whether the partition --
    # not the capacity -- is what collapsed the miss rate.
    "p25-softpool":  {"CGC_SERVER_EXPERT_CACHE_BYTES": "4294967296",
                      "CGC_SOFT_POOL_L0": "32", "CGC_SOFT_POOL_L1": "32"},
    # Speed-max combination. The 2026-09-15 sweep isolated two independent wins:
    #   pillars-off  -> reads 285279 -> 136290 (-52%), 6.48 -> 7.83 t/s, and loopiness 0.0
    #                   (the ONLY MTP-on arm whose answer was not a repeated sentence)
    #   nodry        -> draft acceptance 0.735 -> 0.812, 6.48 -> 7.48 t/s
    # They touch different terms (IO volume vs acceptance), so they should compose. NOTE this arm
    # deliberately gives up the three bit-identical pillars, so it is a SPEED datapoint only --
    # it cannot be a candidate production config until the bit-identical gate passes.
    "p25-nodry-pillars": {"CGC_SERVER_DRY_MULTIPLIER": "0",
                          "CGC_SERVER_MTP_NO_WARMUP": "0",
                          "CGC_SERVER_NO_SEQ_RM_PROBE": "0",
                          "CGC_MM_BITIDENT": "0"},
    # Fill-pool concurrency. LLAMA_EXPERT_CACHE_WORKERS was the literal 8 in run_server.sh and
    # had never been swept, yet it is the only knob controlling how many expert preads are in
    # flight. The measurements say latency -- not bandwidth -- is the binding term, so this is
    # the cheapest untested lever on the whole path.
    "p25-workers16": {"CGC_SERVER_WORKERS": "16"},
    "p25-workers32": {"CGC_SERVER_WORKERS": "32"},
}


def killed():
    subprocess.run(["pkill", "-INT", "-f", SERVER_MATCH],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6)
    subprocess.run(["pkill", "-9", "-f", SERVER_MATCH],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)


def wait_ready(ctrl_path, timeout=240):
    """run_server.sh prints `[detach] server ready` once the child is health-checked, and its
    own `[log] <path>` line names the file the SERVER writes (which is where `listening on
    http`, `print_timing` and the expert-cache teardown actually land). Polling the shim's
    stdout for `listening on http` never matches -- the shim only relays its own messages."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(ctrl_path):
            try:
                with open(ctrl_path, "r", errors="replace") as f:
                    txt = f.read()
                if "[detach] server ready" in txt:
                    return True
                if "error:" in txt or "FAILED" in txt:
                    return False
            except OSError:
                pass
        time.sleep(2)
    return False


LOG_LINE_RE = re.compile(r"\[log\]\s+([^\s（(]+)")


def server_log_of(ctrl_path):
    """The server's own log path, as announced by run_server.sh."""
    try:
        txt = open(ctrl_path, "r", errors="replace").read()
    except OSError:
        return ""
    m = LOG_LINE_RE.search(txt)
    return m.group(1) if m else ""


def start(env_extra, ctrl_path, profile="prefill250"):
    env = dict(os.environ)
    env["CGC_SERVER_PROFILE"] = profile
    env.update(env_extra)
    open(ctrl_path, "w").close()
    with open(ctrl_path, "a") as fh:
        subprocess.Popen(["./scripts/run_server.sh", "--detach"],
                         cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return wait_ready(ctrl_path)


# NOTE ON WHITESPACE: the C++ teardown pads fields with TWO spaces
# (`... misses=20080 (hit rate 92.4%)  prewarm req=0 ...`, `... slots=39  worst=layer 2 ...`).
# An earlier revision hard-coded single spaces, so FINAL_RE/ATTR_RE silently matched nothing and
# every sweep row reported `hit None% miss None` while SHAPE_RE (which used \s+) worked. Use \s+
# between every field -- never a literal space -- so the harvest cannot silently degrade again.
FINAL_RE = re.compile(r"final stats: runtime requests=(\d+)\s+hits=(\d+)\s+misses=(\d+)\s+"
                      r"\(hit rate ([\d.]+)%\)\s+prewarm req=(\d+)\s+hit=(\d+)\s+miss=(\d+)\s+"
                      r"resident=([\d.]+) MiB\s+file_reads=(\d+)\s+pread_usec=(\d+)\s+"
                      r"fill_batch_usec=(\d+)")
ATTR_RE = re.compile(r"miss attribution: compulsory=(\d+)\s+capacity=(\d+)\s+"
                     r"\(([\d.]+)% / ([\d.]+)%\s+of\s+(\d+)\)\s+evictions=(\d+)\s+"
                     r"layers_distinct_over_slots=(\d+)\s+worst=layer (\d+)\s+distinct=(\d+)\s+slots=(\d+)")
SHAPE_RE = re.compile(r"read shape: jobs=(\d+)\s+bytes=(\d+).*us/job=(\d+)\s+effective_rate=([\d.]+) MiB/s")
# `slot print_timing: id 0 | task 88 | draft acceptance = 0.99740 (383 accepted / 384 generated),
# mean len = 3.99` -- the LAST one in the log is the steady-state value. This is the number that
# decides whether the MTP speculative path nets a speedup: at mean len ~4 each decode step
# amortises 4 tokens; at mean len ~1 it is pure overhead.
DRAFT_RE = re.compile(r"draft acceptance = ([\d.]+)\s+\(\s*(\d+) accepted\s*/\s*(\d+) generated\),\s*"
                      r"mean len =\s*([\d.]+)")
# Server-measured throughput, straight from slot print_timing (`tg`). Independent of the
# client-side timing decode_bench derives, so the two cross-check each other.
TG_RE = re.compile(r"n_decoded =\s*(\d+),\s*tg =\s*([\d.]+) t/s")


def loopiness(text):
    """Degeneration detector for the sampled answer.

    A collapsed draft/verify path does not crash -- it emits a coherent-looking sentence over
    and over (`p25-nowarmup`: `Here, the user request is genuinely ambiguous, ask a sharp
    question.` repeated to the token limit). md5 stability across rounds CANNOT see this (the
    loop is perfectly deterministic), so every arm must report it: `answer_stable=True` next to
    a loopiness near 1.0 is a broken arm, not a stable one.

    Measured as the fraction of REPEATED WORD 4-GRAMS, not lines and not fixed-size chunks.
    Two earlier revisions failed on real data: comparing stripped lines scored every MTP-on loop
    0.00 (the repeats were separated by `. `, or differed in one word -- `don'guess.` vs
    `don't guess.` -- so each line was unique), and half-overlapping 32-char chunks scored
    `p25-probe-on` 0.00 even though its sample is visibly one sentence twice, because the repeat
    period (~66 chars) did not line up with the 32/16 window. Word 4-grams are period-agnostic
    and ignore punctuation and whitespace, which is what makes them survive the sample being
    truncated to ~200 chars. 0 = no phrase reused, 1 = every 4-gram is a repeat."""
    words = " ".join(text.split()).split()
    k = 4
    if len(words) < 2 * k:
        return 0.0
    grams = [tuple(words[i:i + k]) for i in range(len(words) - k + 1)]
    return round(1.0 - len(set(grams)) / len(grams), 3)


def harvest(log_path):
    try:
        txt = open(log_path, "r", errors="replace").read()
    except OSError:
        return {}
    out = {}
    if (m := FINAL_RE.search(txt)):
        req, hit, miss, rate, p_req, p_hit, p_miss, res, reads, pu, fbu = m.groups()
        out.update({
            "requests": int(req), "hits": int(hit), "misses": int(miss),
            "hit_rate_pct": float(rate),
            "prewarm_req": int(p_req), "prewarm_miss": int(p_miss),
            "resident_mib": float(res), "file_reads": int(reads),
            "pread_usec": int(pu), "fill_batch_usec": int(fbu),
        })
        rt = int(miss) + int(p_miss)
        if rt > 0:
            out["per_miss_us"] = round(int(pu) / rt, 1)
    if (m := ATTR_RE.search(txt)):
        comp, cap, cp, cap_p, tot, ev, over, wl, wd, wns = m.groups()
        out.update({
            "miss_compulsory": int(comp), "miss_capacity": int(cap),
            "capacity_pct": float(cap_p),
            "evictions": int(ev), "layers_over_slots": int(over),
            "worst_layer": int(wl), "worst_distinct": int(wd), "worst_slots": int(wns),
        })
    if (m := SHAPE_RE.search(txt)):
        jobs, b, usj, rate = m.groups()
        out.update({"io_jobs": int(jobs), "io_bytes": int(b),
                    "io_us_per_job": int(usj), "io_effective_mib_s": float(rate)})
    if (ms := DRAFT_RE.findall(txt)):
        acc, a, g, ml = ms[-1]
        out.update({"draft_accept": float(acc), "draft_accepted": int(a),
                    "draft_generated": int(g), "draft_mean_len": float(ml)})
    if (ms := TG_RE.findall(txt)):
        n, tg = ms[-1]
        out.update({"n_decoded_server": int(n), "tg_server": float(tg)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="baseline")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--n-predict", type=int, default=160)
    ap.add_argument("--json", default="/tmp/decode_sweep.json")
    ap.add_argument("--profile", default="prefill250")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    if args.report:
        rows = json.load(open(args.report))
        print(f"{'tag':16s} {'decode':>7s} {'dec_min':>8s} {'prefill':>8s} {'hit%':>6s} "
              f"{'miss':>7s} {'us/miss':>8s} {'MiB/s':>7s} {'reads':>7s} {'acc':>6s} "
              f"{'mlen':>5s} {'loop':>5s} {'stable':>6s}")
        for r in rows:
            ml = r.get('draft_mean_len')
            acc = r.get('draft_accept')
            acc_s = f"{acc:.3f}" if acc is not None else "-"
            ml_s = f"{ml:.2f}" if ml is not None else "-"
            print(f"{r['tag']:16s} {r.get('decode_tps_median', 0):7.2f} "
                  f"{r.get('decode_tps_min', 0):8.2f} {r.get('prefill_tps_median', 0):8.2f} "
                  f"{r.get('hit_rate_pct', 0):6.1f} {r.get('misses', 0):7d} "
                  f"{r.get('per_miss_us', 0):8.1f} {r.get('io_effective_mib_s', 0):7.1f} "
                  f"{r.get('file_reads', 0):7d} {acc_s:>6s} {ml_s:>5s} "
                  f"{r.get('loopiness', 0):5.2f} {str(r.get('answer_stable')):>6s}")
        return

    rows = []
    if os.path.exists(args.json):
        try:
            rows = json.load(open(args.json))
        except Exception:  # noqa: BLE001
            rows = []
    have = {r["tag"] for r in rows}

    for tag in args.arms.split(","):
        tag = tag.strip()
        if not tag:
            continue
        if tag not in ARMS:
            print(f"unknown arm {tag!r}; known: {','.join(ARMS)}", file=sys.stderr)
            continue
        if tag in have and not args.force:
            print(f"[skip] {tag} already recorded")
            continue

        print(f"\n===== arm {tag}  env={ARMS[tag]} =====", flush=True)
        killed()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        ctrl_path = os.path.join(LOG_DIR, f"arm_{tag}_{stamp}.ctrl.log")
        if not start(ARMS[tag], ctrl_path, args.profile):
            print(f"[FAIL] {tag}: server never became ready; see {ctrl_path}", flush=True)
            continue
        srv_log = server_log_of(ctrl_path)
        print(f"  ready; ctrl={ctrl_path}\n         srv={srv_log}", flush=True)

        bench_json = f"/tmp/decode_sweep_bench_{tag}.json"
        if os.path.exists(bench_json):
            os.remove(bench_json)
        subprocess.run(["/opt/homebrew/bin/python3", "scripts/check/decode_bench.py",
                        "--rounds", str(args.rounds), "--warmup", str(args.warmup),
                        "--n-predict", str(args.n_predict), "--tag", tag,
                        "--json", bench_json],
                       cwd=ROOT)
        bench = json.load(open(bench_json))[-1] if os.path.exists(bench_json) else {}

        # SIGINT (not -9) so the teardown stats are printed, then wait for them to land.
        subprocess.run(["pkill", "-INT", "-f", SERVER_MATCH],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(30):
            time.sleep(2)
            if srv_log and os.path.exists(srv_log) and \
                    "final stats" in open(srv_log, errors="replace").read():
                break
        subprocess.run(["pkill", "-9", "-f", SERVER_MATCH],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        row = {"tag": tag, "env": ARMS[tag], "log": srv_log}
        row.update({k: v for k, v in bench.items() if k != "tag"})
        row["loopiness"] = loopiness(bench.get("sample", "") or "")
        row.update(harvest(srv_log))
        rows.append(row)
        json.dump(rows, open(args.json, "w"), ensure_ascii=False, indent=2)
        print(f"  -> decode {row.get('decode_tps_median')} t/s  "
              f"hit {row.get('hit_rate_pct')}%  miss {row.get('misses')}  "
              f"us/miss {row.get('per_miss_us')}  io {row.get('io_effective_mib_s')} MiB/s  "
              f"accept {row.get('draft_accept')} / mean_len {row.get('draft_mean_len')}  "
              f"loop {row.get('loopiness')}",
              flush=True)

    print(f"\nsaved -> {args.json}")
    subprocess.run(["/opt/homebrew/bin/python3", __file__, "--report", args.json], cwd=ROOT)


if __name__ == "__main__":
    main()
