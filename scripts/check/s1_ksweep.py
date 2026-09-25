#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""s1_ksweep.py -- what does verify token 2..4 cost on the one-shot-submit path?

WHY THIS ARM EXISTS
-------------------
The delivery cell (`prod-new`, MTP on) moves 3.117 tokens per step. The single-submit (S1) shape was
measured at worse than that: 1.72x the segmented arm, but on a pair whose I/O structures differed
(`file_reads` 82,293 vs 0) and with neither arm carrying a verify batch (no `--spec-type` in either
command line). So the one number that decides whether 25 t/s is reachable has never been taken:

    on the one-shot path, how much does verify token #2..#4 add to the step?

At the S1 pair's own 49.2 ms/token, 25 t/s needs only `mean_len` 1.23 (accept ~8%). If the marginal
verify token is cheap (the repo measured 7-8 ms/token when batching does amortise), the step for k=4
is ~73 ms and the shape delivers >40 t/s. If it is not amortised at all, the step is ~196 ms and the
shape delivers ~16. Same shape, 16..42 t/s, decided by this sweep.

THE ARM
-------
    S1 (CGC_SEG_BATCH + CGC_B_SCHEME + CGC_SLOT_TABLE_GPU) + MTP on + a FROZEN pool.
    "Frozen" is not a setting: with the SEG_BATCH path the per-layer hook never fires, so no demand
    fill ever runs and nothing is ever evicted. The smoke run proves the state is real:
        final stats: requests=2173 hits=2173 misses=0  file_reads=0  fill_wait_us=0  hit 100.0%
    (requests == slots x layers exactly: the per-expert demand path is never exercised.)

WHAT THIS ARM'S NUMBERS ARE, AND ARE NOT
----------------------------------------
 * ms/step is the object: MoE gather/matmul is dense over the gathered rows, so it does not depend on
   the expert bytes' VALUES. Timing is valid even though the values are not.
 * mean_len is NOT the delivery accept: experts outside the frozen set are never filled, so part of
   the routing contributes a zeroed slot. Report it as this arm's accept, not the product's.
 * The pool is 3072 MiB, chosen by `budget_preflight.py` as the largest legal one (model 13030 +
   3072 <= 16384). At 8 GiB the box is oversubscribed by 4838 MiB, which is the state in which this
   project has previously burned 50-minute runs on all-zero arms. Side effect: routable=53 ->
   decode width 6 instead of 8, which does not bind for k<=4 (T<=5).
 * Every launch is labelled with the STATE it was taken in -- thermal level and swap, from the two
   instruments this tree already owns (`thermal_pressure`, `memory_pressure`), held OVER the child so
   the reading is the state it launched into and the middle it spent hot. A launch with no readable
   reading is UNKNOWN, and the artifact is then not quotable: state is part of the provenance.
 * The same launch measures PREFILL and decode (`-p 2048 -n 128`): the sweep reports both, because
   the slab that serves prefill and the gathered path that serves decode share the pool.

Usage:
    python3 scripts/check/s1_ksweep.py --dry-run
    python3 scripts/check/s1_ksweep.py --reps 2
"""
import argparse
import json
import os
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import server_window as sw          # noqa: E402  the shared probe + window taxonomy
import io_symmetry as ios           # noqa: E402  arms must also be I/O-symmetric to each other
import thermal_pressure as tp       # noqa: E402  the OS thermal level, the tree's own instrument
import memory_pressure as mp        # noqa: E402  swap/wired, and the attribution rule between them

BENCH = ROOT / "src/llama.cpp/build/bin/llama-bench"
MODEL = ROOT / "models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf"
POOL_BYTES = 3221225472             # 3072 MiB; the largest that passes budget_preflight (gate-clean)

# The operator's settled cell, taken verbatim from the S1 pair so this sweep sits on a recorded shape.
CELL = ["-ngl", "99", "--load-mode", "none", "-t", "8", "-b", "5632", "-ub", "5632",
        "-p", "2048", "-n", "128", "-d", "512", "-r", "3", "--warm-skip", "64", "--ctx-size", "0",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-o", "json"]

# Presence-based switches: "off" must be ABSENT, not "0" (the family that historically tested presence).
S1_ENV = {"CGC_SEG_BATCH": "1", "CGC_B_SCHEME": "1", "CGC_SLOT_TABLE_GPU": "1"}
# Required or the whole expert cache is disabled and the 13 GB model goes to Metal (cap 11.45 GB) -> OOM.
BASE_ENV = {"LLAMA_EXPERT_CACHE_ALLOW_NGL": "1", "CGC_EXPERT_SKIP_READRAW": "1",
            "CGC_MTP_PERF": "1", "CGC_OA_ASYNC": "1", "CGC_SPAC": "1",
            "CGC_PREFILL_STREAM": "1", "CGC_GATHER_SLAB_CAP": "256", "CGC_MM_BITIDENT": "1"}

# k = --spec-draft-n-max. `nospec` is the T=1 base: it reproduces the S1 pair's B arm on this cell.
ARMS = [("nospec", None), ("k1", 1), ("k2", 2), ("k3", 3), ("k4", 4)]

RE_PERF = re.compile(r"CGC-MTP-PERF type=(\S+) calls_begin=(\d+) calls_draft=(\d+) calls_accept=(\d+) "
                     r"gen_tokens=(\d+) acc_tokens=(\d+) .*acc_rate=([\d.]+) gen_tok_per_round=([\d.]+) "
                     r"acc_tok_per_round=([\d.]+) emit_tok_per_round=([\d.]+) ms_per_round=([\d.]+)")
RE_SPLIT = re.compile(r"CGC-PHASE-SPLIT: cap=(\d+) routable=(\d+) slots / top_k=(\d+) -> "
                      r"decode graph width=(\d+) tokens")
RE_FINAL = re.compile(r"final stats: runtime requests=(\d+) hits=(\d+) misses=(\d+) \(hit rate "
                      r"([\d.]+)%\)(.*?)file_reads=(\d+) pread_usec=(\d+) fill_batch_usec=(\d+) "
                      r"fill_wait_us=(\d+)")


def env_for(arm, k):
    env = dict(os.environ)
    for key in (*S1_ENV, "CGC_MTP_PERF"):
        env.pop(key, None)
    env.update(S1_ENV if arm != "nospec" else {})
    env.update(BASE_ENV)
    return env


def cmd_for(k):
    c = [str(BENCH), "-m", str(MODEL), "-expert-cache", str(POOL_BYTES), *CELL]
    if k is not None:
        c += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(k)]
    return c


def parse_log(text):
    """Everything the two instruments printed, taken from the LAST occurrence (cumulative)."""
    m = RE_PERF.findall(text)
    perf = None
    if m:
        g = m[-1]
        perf = {"type": g[0], "calls_begin": int(g[1]), "calls_draft": int(g[2]),
                "calls_accept": int(g[3]), "gen_tokens": int(g[4]), "acc_tokens": int(g[5]),
                "acc_rate": float(g[6]), "gen_tok_per_round": float(g[7]),
                "acc_tok_per_round": float(g[8]), "emit_tok_per_round": float(g[9]),
                "ms_draft_per_round": float(g[10])}
    s = RE_SPLIT.findall(text)
    split = {"routable": int(s[-1][1]), "width": int(s[-1][3])} if s else None
    f = RE_FINAL.findall(text)
    pool = None
    if f:
        g = f[-1]
        pool = {"requests": int(g[0]), "hits": int(g[1]), "misses": int(g[2]), "hit_pct": float(g[3]),
                "file_reads": int(g[5]), "pread_usec": int(g[6]), "fill_wait_us": int(g[8])}
    return perf, split, pool


def md5(p):
    return subprocess.run(["md5", "-q", str(p)], capture_output=True, text=True).stdout.strip()[:16]


def build_fingerprint():
    """Every dylib: the S1 switch halves live in DIFFERENT libraries, so one file is a false negative."""
    out = {}
    for lib in sorted((ROOT / "src/llama.cpp/build/bin").glob("*.dylib")):
        out[lib.name] = md5(lib)
    for exe in ("llama-bench", "llama-server"):
        p = ROOT / "src/llama.cpp/build/bin" / exe
        if p.exists():
            out[exe] = md5(p)
    return out


def wait_for_window(need_mb, minutes, label=""):
    """Poll until the box is ours, bounded. Used at start AND before every launch: a neighbour that
    starts mid-sweep would otherwise be measured into whichever arms follow it, and this box's other
    session does exactly that (successive llama-bench launches with changing pids)."""
    w = window_ok(need_mb)
    if w["admits"]:
        return w
    if not minutes:
        return w
    t_end = time.time() + 60 * minutes
    while not w["admits"] and time.time() < t_end:
        print(f"WAITING{label}: {w['refused_by']} (reclaimable={w['reclaimable_mb']}MB, "
              f"foreign={w['foreign_llama']}, {int(t_end - time.time())}s left)", flush=True)
        time.sleep(60)
        w = window_ok(need_mb)
    return w


def run_one(arm, k, outdir, idx):
    log = outdir / f"{idx:02d}_{arm}.stderr.log"
    jpath = outdir / f"{idx:02d}_{arm}.json"
    cmd = cmd_for(k)
    t0 = time.time()
    # Both state instruments are held OVER the child, not read at its bookends. Their first sample is
    # taken in start(), i.e. before the spawn, so `launch` really is the state the arm started in;
    # and `worst` covers a middle that an arm starting and ending Nominal can still spend at HEAVY.
    # Reading `vm.swapusage` either side of the run instead is what made the previous artifact record
    # the same number for before and after -- both reads landed after the arm had already finished.
    th, mem = tp.Sampler(), mp.Sampler()
    th.start()
    mem.start()
    proc = subprocess.run(cmd, env=env_for(arm, k), capture_output=True, text=True)
    wall = time.time() - t0
    th_r, mem_r = th.stop(), mem.stop()
    log.write_text(proc.stderr)
    tg = pp = None
    try:
        rows = json.loads(proc.stdout)
        for r in rows:
            if r.get("n_gen"):
                tg = r
            elif r.get("n_prompt"):
                pp = r
    except Exception:
        pass
    jpath.write_text(proc.stdout)
    perf, split, pool = parse_log(proc.stderr)
    rec = {"idx": idx, "arm": arm, "k": k, "T": (k + 1) if k is not None else 1, "rc": proc.returncode,
           "wall_s": round(wall, 1), "tg_tps": tg and tg["avg_ts"], "tg_sd": tg and tg["stddev_ts"],
           "tg_n": tg and tg.get("n_kept"), "pp_tps": pp and pp["avg_ts"],
           "pp_sd": pp and pp["stddev_ts"], "pp_n": pp and pp.get("n_kept"),
           "state": {"thermal": th_r, "memory": mem_r,
                     "attribution": mp.attribution(th_r, mem_r)},
           "perf": perf, "split": split, "pool": pool, "log": str(log.relative_to(ROOT))}
    if perf and tg:
        # one round == one step. `emit_tok_per_round` is what maps to throughput (acc + bonus token).
        rec["mean_len"] = perf["emit_tok_per_round"]
        rec["ms_per_step"] = 1000.0 * perf["emit_tok_per_round"] / tg["avg_ts"]
        rec["ms_verify_plus_host"] = rec["ms_per_step"] - perf["ms_draft_per_round"]
    return rec


def fit_line(xs, ys):
    """Least squares y = a + b*x. Two points would fit exactly; five test the line."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    a = my - b * mx
    resid = max(abs(y - (a + b * x)) for x, y in zip(xs, ys))
    return {"intercept_ms": a, "marginal_ms_per_verify_token": b, "max_resid_ms": resid}


def thermal_of(r, slot="worst"):
    """The label, or UNKNOWN: an arm with no readable reading is not an arm measured at Nominal."""
    t = ((r.get("state") or {}).get("thermal") or {})
    s = t.get(slot) if isinstance(t.get(slot), dict) else None
    return (s or {}).get("label", "UNKNOWN")


def swap_of(r, slot="launch"):
    m = ((r.get("state") or {}).get("memory") or {})
    s = m.get(slot) if isinstance(m.get(slot), dict) else None
    return (s or {}).get("swap_used_mb")


def state_block(recs):
    """One line per axis, over the whole run -- the form a reader of the table needs.

    The per-launch series are in `records` (both samplers keep every sample); this is the summary
    that answers "can this table be quoted, and on what box". `unlabelled_launches` is deliberately a
    list and not a count: it names the launches whose state is unknown, so they can be re-run.
    """
    lv = [thermal_of(r, "launch") for r in recs]
    wr = [thermal_of(r, "worst") for r in recs]
    sw = [swap_of(r, "launch") for r in recs]
    en = [swap_of(r, "end") for r in recs]
    at = [((r.get("state") or {}).get("attribution") or {}).get("verdict", "UNKNOWN") for r in recs]
    rng = lambda xs: (min(xs), max(xs)) if xs and all(x is not None for x in xs) else None
    return {"n_launches": len(recs),
            "thermal_at_launch": {k: lv.count(k) for k in sorted(set(lv))},
            "thermal_worst": {k: wr.count(k) for k in sorted(set(wr))},
            "swap_launch_mb": rng(sw), "swap_end_mb": rng(en),
            "attribution": {k: at.count(k) for k in sorted(set(at))},
            "unlabelled_launches": [r["idx"] for r in recs
                                   if thermal_of(r, "launch") == "UNKNOWN" or swap_of(r) is None]}


def window_ok(need_mb):
    """The repo's own two-probe admission test, not a re-implementation of it.

    `decision` returns both answers (harness reclaimable-memory, and the launcher's free-percentage)
    because on this machine they disagree; taking either one alone is how a run gets measured on a
    neighbour. `scan` is the fallback when neither module is importable.
    """
    d = sw.decision(need_mb=need_mb)
    return {"admits": d["admits"], "reclaimable_mb": d["reclaimable_mb"],
            "foreign_llama": d["foreign_llama"], "port_held": d["port_held"],
            "binding": d["binding"], "agree": d["agree"], "refused_by": d["refused_by"]}


# --------------------------------------------------------------------------- selftest
# Fixtures are verbatim lines from the smoke launch's own stderr (/tmp/s1k_smoke.log, 2026-09-24).
FIX_PERF_1 = ("CGC-MTP-PERF type=draft-mtp calls_begin=1 calls_draft=4 calls_accept=4 gen_tokens=8 "
              "acc_tokens=6 t_begin_ms=0.0 t_draft_ms=189.6 t_accept_ms=0.0 acc_rate=0.7500 "
              "gen_tok_per_round=2.000 acc_tok_per_round=1.500 emit_tok_per_round=2.500 ms_per_round=47.408")
FIX_PERF_2 = ("CGC-MTP-PERF type=draft-mtp calls_begin=2 calls_draft=12 calls_accept=12 gen_tokens=24 "
              "acc_tokens=22 t_begin_ms=0.0 t_draft_ms=302.2 t_accept_ms=0.0 acc_rate=0.9167 "
              "gen_tok_per_round=2.000 acc_tok_per_round=1.833 emit_tok_per_round=2.833 ms_per_round=25.184")
FIX_SPLIT = ("CGC-PHASE-SPLIT: cap=8 routable=53 slots / top_k=8 -> decode graph width=6 tokens "
             "(bound=6, T_prefill=512, non-binding); prefill slab armed (CGC_PREFILL_STREAM=1)")
FIX_FINAL = ("llama_expert_cache: final stats: runtime requests=2173 hits=2173 misses=0 (hit rate "
             "100.0%)  prewarm req=0 hit=0 miss=0  resident=2354.36 MiB file_reads=0 pread_usec=0 "
             "fill_batch_usec=0 fill_wait_us=0 prefetch=0/0")


def self_test() -> int:
    bad = 0

    def chk(name, cond):
        nonlocal bad
        if not cond:
            bad += 1
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")

    text = "\n".join([FIX_PERF_1, FIX_SPLIT, FIX_PERF_2, FIX_FINAL])
    perf, split, pool = parse_log(text)
    chk("the LAST perf line wins (it is the cumulative one)", perf and perf["calls_draft"] == 12)
    chk("mean_len = emit_tok_per_round, not gen/acc", perf["emit_tok_per_round"] == 2.833)
    chk("wide decode width read", split == {"routable": 53, "width": 6})
    chk("pool read: frozen (0 reads, 0 misses, 100%)",
        pool and pool["file_reads"] == 0 and pool["misses"] == 0 and pool["hit_pct"] == 100.0)

    ms = 1000.0 * perf["emit_tok_per_round"] / 30.069743        # the smoke run's own tg t/s
    chk(f"ms/step identity (got {ms:.2f})", abs(ms - 94.22) < 0.05)

    chk("T mapping: nospec->1, k4->5", [1 if k is None else k + 1 for _, k in ARMS]
        == [1, 2, 3, 4, 5])

    f = fit_line([1, 2, 3], [100.0, 110.0, 120.0])
    chk("perfect line: slope 10, intercept 90, residual 0",
        abs(f["marginal_ms_per_verify_token"] - 10) < 1e-9 and abs(f["intercept_ms"] - 90) < 1e-9
        and f["max_resid_ms"] < 1e-9)
    # A convex sweep is the physical case worth detecting: a verify token that costs MORE the wider
    # the batch gets would show here, and a 2-point fit could never reveal it.
    f2 = fit_line([1, 2, 3], [100.0, 110.0, 140.0])
    chk(f"a NON-linear sweep is detectable, not silently fitted away (resid {f2['max_resid_ms']:.2f} ms)",
        f2["max_resid_ms"] > 1.0)
    chk("fewer than 2 points -> no line claimed", fit_line([1], [100.0]) is None)
    chk("no data -> no line", fit_line([], []) is None)

    # The state axes. The failure this guards is the one the previous artifact shipped: a launch with
    # no thermal reading counted as if it had been measured at Nominal, and swap read the same value
    # on both sides of the run. Absence must be UNKNOWN and must make the table unquotable.
    clean = {"idx": 1, "state": {"thermal": {"launch": {"label": "NOMINAL"},
                                            "worst": {"label": "NOMINAL"}},
                              "memory": {"launch": {"swap_used_mb": 512.0},
                                         "end": {"swap_used_mb": 512.0}},
                              "attribution": {"verdict": "none"}}}
    unlabelled = {"idx": 2, "state": {"thermal": {}, "memory": {}, "attribution": {}}}
    legacy = {"idx": 3}                       # the shape the previous artifact actually had
    s = state_block([clean, unlabelled, legacy])
    chk("a labelled launch counts its real level", s["thermal_at_launch"] == {"NOMINAL": 1,
                                                                           "UNKNOWN": 2})
    chk("the unlabelled launches are NAMED, not counted", s["unlabelled_launches"] == [2, 3])
    chk("swap range is None when any launch is unreadable, never a plausible pair",
        s["swap_launch_mb"] is None)
    chk("no state at all -> UNKNOWN, and never NOMINAL",
        thermal_of(legacy) == "UNKNOWN" and swap_of(legacy) is None)
    chk("a clean launch round-trips its own label", thermal_of(clean) == "NOMINAL"
        and swap_of(clean) == 512.0)
    chk("one unlabelled launch is enough to refuse the whole run",
        bool(state_block([clean, legacy])["unlabelled_launches"])
        and not state_block([clean])["unlabelled_launches"])

    # Prefill must be surfaced with decode: the same launch measures both, and a sweep that prints
    # only tg would hide the half of the target that the slab serves.
    med = {"k2": {"T": 3, "n": 2, "ms_per_step": 129.2, "mean_len": 2.83, "tg_tps": 21.9,
                  "pp_tps": 280.6, "acc_rate": 0.915}}
    chk("prefill is carried in the per-arm summary, not just the raw records",
        med["k2"].get("pp_tps") == 280.6)
    chk("a missing prefill prints as absence, not 0 t/s",
        "--" if not med["k2"].get("pp_tps") else "value")
    print(f"\ns1_ksweep selftest: {'PASS' if bad == 0 else f'{bad} FAILED'}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--launch-wait-minutes", type=int, default=20,
                    help="per-launch re-admission budget: how long to wait before an individual "
                         "launch if a neighbour has taken the box (0 = abort the sweep instead)")
    ap.add_argument("--wait-minutes", type=int, default=0,
                    help="poll for a clean window this long, then launch (0 = refuse immediately). "
                         "A clean window is waited for, not scheduled -- this box is shared.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return self_test()

    need = POOL_BYTES // 1048576 + 4096
    win = window_ok(need)
    print(f"window: reclaimable={win['reclaimable_mb']}MB (need>={need}MB) foreign={win['foreign_llama']} "
          f"port_held={win['port_held']} binding={win['binding']} agree={win['agree']} "
          f"-> {'OK' if win['admits'] else 'REFUSED ' + str(win['refused_by'])}")
    print(f"swap in use: {mp.swap_used_mb()} MB, thermal {tp.stamp()['label']} -- recorded per "
          f"launch by both instruments, and part of this artifact's quotable test.")
    print(f"build: {json.dumps(build_fingerprint())}")
    if a.dry_run:
        for arm, k in ARMS:
            print(f"  {arm:7s} T={k+1 if k is not None else 1}  {' '.join(cmd_for(k))}")
        return 0
    if not win["admits"]:
        if not a.wait_minutes:
            print("refusing: the box is not ours. Nothing launched.")
            return 2
        win = wait_for_window(need, a.wait_minutes)
        if not win["admits"]:
            print("gave up waiting: no clean window. Nothing launched.")
            return 2
        print(f"window opened: reclaimable={win['reclaimable_mb']}MB foreign={win['foreign_llama']}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(a.outdir) if a.outdir else ROOT / "Backup/s1_ksweep" / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"outdir {outdir.relative_to(ROOT)}")

    recs, idx, stop = [], 0, False
    for rep in range(a.reps):
        # rotate the arm order per rep: launch order dominates here (measured), so a fixed order reads
        # that drift as a k-effect.
        order = ARMS[rep % len(ARMS):] + ARMS[:rep % len(ARMS)]
        for arm, k in order:
            # Re-admit before EVERY launch. A window that opens for one arm and closes for the next
            # would otherwise be recorded as a k-effect.
            w = wait_for_window(need, a.launch_wait_minutes, label=f" (before {arm})")
            if not w["admits"]:
                print(f"STOPPING before {arm}: no clean window ({w['refused_by']}); "
                      f"{idx} launch(es) recorded, artifact written")
                stop = True
                break
            idx += 1
            r = run_one(arm, k, outdir, idx)
            r["rep"] = rep + 1
            # Per-launch window record: an arm that ran while a neighbour held the box must be
            # identifiable from the artifact, not from a note in a shell's scrollback.
            r["window"] = window_ok(need)
            recs.append(r)
            fmt = lambda v, w=0: "--" if v is None else f"{v:.{w}f}"  # absence must not print as 0
            print(f"  [r{rep+1}] {arm:7s} T={r['T']} pp={fmt(r['pp_tps'],1)} tg={fmt(r['tg_tps'],2)} "
                  f"ms/step={fmt(r.get('ms_per_step'),1)} mean_len={fmt(r.get('mean_len'),3)} "
                  f"thermal={thermal_of(r)} swap={fmt(swap_of(r),0)}MB "
                  f"rc={r['rc']} wall={r['wall_s']}s")
        if stop:
            break

    # per-arm medians over reps
    med = {}
    for arm, _ in ARMS:
        mine = [r for r in recs if r["arm"] == arm and r.get("ms_per_step")]
        if mine:
            med[arm] = {"T": mine[0]["T"], "n": len(mine),
                        "ms_per_step": st.median(r["ms_per_step"] for r in mine),
                        "mean_len": st.median(r["mean_len"] for r in mine),
                        "tg_tps": st.median(r["tg_tps"] for r in mine if r["tg_tps"]),
                        # Prefill rides in the same launch (-p 2048), so the two halves of the target
                        # are read on one box state rather than two.
                        "pp_tps": st.median(r["pp_tps"] for r in mine if r["pp_tps"]),
                        "acc_rate": st.median(r["perf"]["acc_rate"] for r in mine)}
    fit = fit_line([v["T"] for v in med.values()], [v["ms_per_step"] for v in med.values()]) if len(med) >= 2 else None

    # The arms must also be I/O-symmetric to each other, by the rule added 2026-09-24.
    sym = []
    arms_c = [(a_, {"file_reads": st.median([r["pool"]["file_reads"] for r in recs
                                             if r["arm"] == a_ and r.get("pool")] or [0])})
              for a_, _ in ARMS]
    for i in range(len(arms_c) - 1):
        sym.append(ios.compare(arms_c[i][1], arms_c[i + 1][1], arms_c[i][0], arms_c[i + 1][0]))

    stt = state_block(recs)
    out = {"product": "S1 k-sweep (ms/step and mean_len per k; prefill+decode per launch)",
           "stamp": stamp, "cell": CELL, "pool_bytes": POOL_BYTES,
           "arms": [a_ for a_, _ in ARMS],
           "build": build_fingerprint(), "window": sw.provenance(),
           "window_gate": win, "medians": med, "fit": fit, "state": stt,
           "io_symmetry_between_arms": sym, "records": recs}
    # An arm launched into a box a neighbour was holding measures the neighbour. Say so in the
    # artifact rather than in prose: the fit is only quotable from unpolluted launches.
    polluted = [f"{r['arm']}#r{r['rep']}" for r in recs
                if not (r.get("window") or {}).get("admits")]
    out["polluted_launches"] = polluted
    # State is part of the provenance, so an unlabelled launch makes the fit unquotable too -- the
    # alternative is a table of numbers with no way to ask what box they were measured on.
    out["unlabelled_launches"] = stt["unlabelled_launches"]
    out["quotable"] = not polluted and not stt["unlabelled_launches"]
    j = outdir / "s1_ksweep.json"
    j.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"\nwrote {j.relative_to(ROOT)}")
    if polluted:
        print(f"WARNING: {len(polluted)} launch(es) ran into a busy box: {polluted} -> the fit is "
              f"NOT quotable; re-run those arms in a clean window")
    print("\nstate every arm ran in (thermal at launch | worst during, swap at launch -> end):")
    print(f"  thermal {stt['thermal_at_launch']} | worst {stt['thermal_worst']}")
    print(f"  swap {stt['swap_launch_mb']} -> {stt['swap_end_mb']} MB   "
          f"attribution {stt['attribution']}")
    if stt["unlabelled_launches"]:
        print(f"  WARNING: no readable state for launch idx {stt['unlabelled_launches']} "
              f"-> UNKNOWN (not Nominal), and the fit is not quotable")
    print(f"\n{'arm':8s}{'T':>3}{'n':>3}{'prefill':>9}{'ms/step':>10}{'ms/token':>10}"
          f"{'mean_len':>10}{'decode':>8}{'accept':>8}")
    for a_, _ in ARMS:
        v = med.get(a_)
        if v:
            pp_ = "--" if not v.get("pp_tps") else f"{v['pp_tps']:.1f}"
            print(f"{a_:8s}{v['T']:>3}{v['n']:>3}{pp_:>9}{v['ms_per_step']:>10.1f}"
                  f"{v['ms_per_step']/v['mean_len']:>10.2f}{v['mean_len']:>10.3f}"
                  f"{v['tg_tps']:>8.2f}{v['acc_rate']:>8.3f}")
    if fit:
        print(f"\nline: ms/step = {fit['intercept_ms']:.1f} + {fit['marginal_ms_per_verify_token']:.2f} x T"
              f"   (max residual {fit['max_resid_ms']:.1f} ms over {len(med)} points)")
        print(f"=> marginal verify token = {fit['marginal_ms_per_verify_token']:.2f} ms; "
              f"fixed cost per round = {fit['intercept_ms']:.1f} ms")
        base = med.get("nospec")
        if base:
            print(f"=> at k=4 the shape should deliver mean_len "
                  f"{base['mean_len'] * base['T'] / (fit['intercept_ms'] + 5 * fit['marginal_ms_per_verify_token']) * 1000:.1f}"
                  f" t/s if its accept matched its own nospec arm")
    bad = [s for s in sym if s.get("verdict") != "comparable"]
    print(f"\nI/O symmetry across arms: {'all comparable' if not bad else json.dumps(bad, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
