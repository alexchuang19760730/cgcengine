#!/usr/bin/env python3
"""F2 / F3 / F4 / F5 — apply the pre-registered rules to the measured arms, and say WHY.

Rules are in `Backup/phase_decomp/F2_F5_OVERLAP_AND_AUDIT_20260920.conditions.md`, written before
the launches. This tool repeats them verbatim and never re-tunes them.

The one thing this tool exists for beyond arithmetic: **F2's null has a cause that its own metric
cannot see.** Arm B (layer-ahead prefetch ON) produced the same `cb` as arm A, and the tempting
reading is "overlap does not help". The logs say something narrower and much more useful -- the
prefetch was never queued at all, so arm B is not a test of overlap. Three independent witnesses:

  * `prefetch=N/M` -- the success counter is bumped only on prefetch_slot's success path
  * the `prefetch drop breakdown` line -- absent means no `-1` return either
  * `PFDBG` line shapes -- every `-1` path prints under PREFETCH_DBG, and ZERO prefetch_slot lines
    appeared while thousands of `PFDBG batch` lines prove the env var was live

Same class as the earlier defect where `CGC_SERVER_MTP=0` silently dropped a seven-variable env
block: an arm that is labelled "treatment ON" but never executes the treatment.

Inputs are paths, not "the newest file": a newest-file guess would attribute a neighbouring run's
log to this arm, which is exactly the mistake this line of work keeps catching.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cb_miss_regression as cmr  # noqa: E402  (same directory, same pairing rule)

# ── fixed in advance ─────────────────────────────────────────────────────────────────────────────
# F2 (pre-registered): PASS cb_B <= 0.5*cb_A; FAIL cb_B >= 0.75*cb_A; else PARTIAL.
F2_PASS = 0.50
F2_FAIL = 0.75
# F3 (pre-registered): CLOSED if the read rate is at/above the copy speed implied by the job size,
# RE-OPENED if it sits in the device band. The threshold is compared against the AGGREGATE rate --
# see the correction below, which is the whole point of F3's section.
COPY_SPEED_MIB_S = 1000.0
DEVICE_BAND_MIB_S = 823.0            # documented internal-SSD peak (~785 MiB/s; 823 is the MB/s figure)
# The teardown's `effective_rate` divides bytes by the SUM of per-job pread times across worker
# threads (llama-expert-cache.cpp: the header comment states "pread_usec is an aggregate over worker
# threads and therefore cannot be compared to a step at all"). With W workers running concurrently,
# wall time ~= sum/W, so the aggregate throughput is effective_rate * W. The line prints the
# per-job rate under a throughput-sounding label; reading it literally is how F3 almost re-opened.
DEFAULT_WORKERS = 8


def arm_rows(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    return d if isinstance(d, list) else [d]


def cb_of(arm: dict) -> float | None:
    sp = arm.get("step_profile") or {}
    v = sp.get("median_cb_ms")
    return float(v) if isinstance(v, (int, float)) else None


def step_of(arm: dict) -> float | None:
    sp = arm.get("step_profile") or {}
    v = sp.get("median_total_ms")
    return float(v) if isinstance(v, (int, float)) else None


# ── the treatment-application check ──────────────────────────────────────────────────────────────

def _resolve_log(server_log: str | None) -> Path | None:
    if not server_log:
        return None
    p = Path(server_log)
    if p.exists():
        return p
    cand = glob.glob(f"Backup/cgc_logs/*{p.name}")
    return Path(cand[0]) if cand else None


def prefetch_witness(server_log: str | None) -> dict:
    """Was the treatment actually executed? Read the arm's OWN log, never a newest-file guess.

    Evidence strength is reported, not collapsed. `PFDBG batch` lines prove PREFETCH_DBG was live
    in that process, which is what upgrades "no prefetch succeeded" into "prefetch_slot was never
    entered" -- every `-1` return prints and a success bumps `prefetch=`. Without the instrument
    the same counters still mean "nothing was queued", which is weaker and is labelled so.
    """
    out = {"log": None, "present": False, "prefetch": None, "drop_line": False,
           "ps_resident": 0, "ps_no_evictable": 0, "ps_guard_reject": 0,
           "batch_lines": 0, "instrumented": False, "calls_evidenced": None,
           "verdict": "UNKNOWN"}
    p = _resolve_log(server_log)
    if p is None:
        return out
    txt = p.read_text(errors="replace")
    out["log"] = str(p)
    out["present"] = True
    m = re.findall(r"prefetch=(\d+)/(\d+)", txt)
    if m:
        out["prefetch"] = (int(m[-1][0]), int(m[-1][1]))
    out["drop_line"] = "prefetch drop breakdown" in txt
    out["ps_resident"] = txt.count("PFDBG drop: resident")
    out["ps_no_evictable"] = txt.count("PFDBG drop: no-evictable")
    out["ps_guard_reject"] = txt.count("PFDBG guard-reject")
    out["batch_lines"] = txt.count("PFDBG batch")
    out["instrumented"] = out["batch_lines"] > 0
    ps_traces = out["ps_resident"] + out["ps_no_evictable"] + out["ps_guard_reject"]
    if not m and not ps_traces:
        out["verdict"] = "NO-COUNTERS"
        out["calls_evidenced"] = None
    elif ps_traces > 0:
        out["verdict"] = "CALLED"
        out["calls_evidenced"] = True
    elif out["instrumented"]:
        # instrument live, zero traces anywhere -> proven never entered
        out["verdict"] = "NEVER-CALLED"
        out["calls_evidenced"] = 0
    else:
        # no instrument: "nothing queued" is what the counters support, nothing stronger
        out["verdict"] = "NOTHING-QUEUED"
        out["calls_evidenced"] = 0
    return out


def read_shape(server_log: str | None, workers: int) -> dict:
    out = {"found": False, "mib_per_job": None, "us_per_job": None,
           "per_job_mib_s": None, "aggregate_mib_s": None, "workers": workers}
    p = _resolve_log(server_log)
    if p is None:
        return out
    m = re.search(r"read shape: jobs=(\d+) bytes=(\d+) \(([\d.]+) MiB/job as one contiguous run\)\s+"
                  r"us/job=(\d+)\s+effective_rate=(\d+) MiB/s", p.read_text(errors="replace"))
    if not m:
        return out
    out.update(found=True, jobs=int(m.group(1)), bytes=int(m.group(2)),
               mib_per_job=float(m.group(3)), us_per_job=int(m.group(4)),
               per_job_mib_s=float(m.group(5)))
    out["aggregate_mib_s"] = out["per_job_mib_s"] * workers
    return out


# ── verdicts ─────────────────────────────────────────────────────────────────────────────────────

def f2_verdict(cb_a: list[float], cb_b: list[float]) -> tuple[str, float, str]:
    if not cb_a or not cb_b:
        return "NO-DATA", float("nan"), "an arm has no median_cb_ms"
    a, b = st.median(cb_a), st.median(cb_b)
    r = b / a
    if r <= F2_PASS:
        v = "PASS"
    elif r >= F2_FAIL:
        v = "FAIL"
    else:
        v = "PARTIAL"
    return v, r, f"median cb_A={a:.2f} ms (n={len(cb_a)})  median cb_B={b:.2f} ms (n={len(cb_b)})"


def f3_verdict(rs: dict) -> tuple[str, str]:
    if not rs.get("found"):
        return "NO-EVIDENCE", "no `read shape` line: the run did not tear down gracefully"
    agg = rs["aggregate_mib_s"]
    if agg >= COPY_SPEED_MIB_S:
        return "CLOSED", (f"aggregate {agg:.0f} MiB/s >= {COPY_SPEED_MIB_S:.0f} MiB/s copy speed "
                          f"(per-job {rs['per_job_mib_s']:.0f} x {rs['workers']} workers)")
    if agg <= DEVICE_BAND_MIB_S:
        return "RE-OPENED", (f"aggregate {agg:.0f} MiB/s is inside the device band "
                             f"(<= {DEVICE_BAND_MIB_S:.0f}) -- bytes really are coming off the device")
    return "BETWEEN", f"aggregate {agg:.0f} MiB/s is above the device band but below copy speed"


def f5_verdict(l_a: list[float], m1_a: list[float], l_b: list[float], m1_b: list[float]):
    if not (l_a and l_b):
        return "NO-DATA", "an arm has no per-layer rows"
    La, Lb = st.mean(l_a), st.mean(l_b)
    Ma, Mb = st.median(m1_a), st.median(m1_b)
    dL = (Lb - La) / La * 100.0 if La else float("nan")
    if abs(dL) < 5.0:
        return "REFUTED", (f"L unchanged: mean {La:.2f} -> {Lb:.2f} ({dL:+.1f}%), "
                           f"median(cb|m=1) {Ma:.2f} -> {Mb:.2f} ms")
    return "SUPPORTED", (f"L moved {La:.2f} -> {Lb:.2f} ({dL:+.1f}%), "
                         f"median(cb|m=1) {Ma:.2f} -> {Mb:.2f} ms")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms-dir", default="Backup/phase_decomp/f2_arms")
    ap.add_argument("--probe-dir", default="Backup/phase_decomp/f2_trigger_probe")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="expert-cache pread worker threads (launcher default 8)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    # Arm order is A,B,B,A per the pre-registered protocol; the label is taken from the filename so
    # a missing arm is visible as a missing file rather than silently shrinking the sample.
    order = ["armA_1", "armB_2", "armB_3", "armA_4"]
    arms: dict[str, dict] = {}
    for name in order:
        p = Path(a.arms_dir) / f"{name}.json"
        if not p.exists():
            print(f"MISSING ARM {p} -- refusing to report a verdict on an incomplete grid")
            return 2
        arms[name] = arm_rows(p)

    print("=" * 100)
    print("F2 -- does the layer-ahead prefetch move `cb`?   (pre-registered: PASS <=0.5x, FAIL >=0.75x)")
    print("=" * 100)
    cb_a = [c for n in ("armA_1", "armA_4") for arm in arms[n] if (c := cb_of(arm)) is not None]
    cb_b = [c for n in ("armB_2", "armB_3") for arm in arms[n] if (c := cb_of(arm)) is not None]
    v, ratio, detail = f2_verdict(cb_a, cb_b)
    print(f"VERDICT: {v}   cb_B/cb_A = {ratio:.3f}")
    print(f"  {detail}")
    for n in order:
        for arm in arms[n]:
            print(f"    {n:9s} cb={cb_of(arm):7.2f} ms  step={step_of(arm):7.2f} ms  "
                  f"mean_len={arm.get('engine_mean_len')}  accept={arm.get('engine_accept_rate')}")

    print()
    print("-- treatment-application check: was the prefetch executed AT ALL? --")
    la_arms = {}
    for n in order:
        arm = arms[n][0]
        env = arm.get("server_env_cgc") or {}
        want = env.get("CGC_LAYER_AHEAD_PREFETCH", "absent")
        w = prefetch_witness(arm.get("server_log"))
        la_arms[n] = w
        print(f"  {n:9s} LAYER_AHEAD={want:7s} prefetch={w['prefetch']} "
              f"drop_line={w['drop_line']} prefetch_slot(traces)="
              f"{w['ps_resident'] + w['ps_no_evictable'] + w['ps_guard_reject']} "
              f"DBG={'live' if w['instrumented'] else 'not set'}  -> {w['verdict']}")
    never = [n for n in ("armB_2", "armB_3")
             if la_arms[n]["verdict"] in ("NEVER-CALLED", "NOTHING-QUEUED")]
    if v == "FAIL" and len(never) == 2:
        print()
        print("  *** THE ARM DID NOT TEST OVERLAP. ***")
        print("  Both B arms carry CGC_LAYER_AHEAD_PREFETCH=1 in their live env, yet neither queued a")
        print("  single prefetch, so there was nothing that could have overlapped. F2's FAIL is a")
        print("  treatment-application null, NOT evidence that hiding cb is impossible. The only")
        print("  difference between the arms that the engine could have seen was one getenv().")
        print("  Mechanism follows in the F2-probe section below.")

    print()
    print("=" * 100)
    print("F2-probe -- WHY the trigger never fired (three arms, two pool sizes)")
    print("=" * 100)
    probe_logs: dict[str, str | None] = {}
    for lab in ("p1_prevtoken_8g", "p2_layerahead_4g", "p3_both_4g"):
        p = Path(a.probe_dir) / f"{lab}.json"
        log = None
        # The json carries the log the run itself used. Only if it is absent do we fall back to the
        # run's stdout banner -- and never to "the newest file", which is how a neighbouring run's
        # log gets attributed to this arm.
        if p.exists():
            for arm in arm_rows(p):
                if arm.get("server_log"):
                    log = arm["server_log"]
                    break
        if log is None:
            so = Path(a.probe_dir) / f"{lab}.stdout"
            if so.exists():
                m = re.findall(r"server log:\s*(\S+)", so.read_text(errors="replace"))
                if m:
                    log = m[-1]
        probe_logs[lab] = log
        w = prefetch_witness(log)
        print(f"  {lab:17s} pool={'8G' if '8g' in lab else '4G'}  prefetch={w['prefetch']}  "
              f"prefetch_slot traces={w['ps_resident'] + w['ps_no_evictable'] + w['ps_guard_reject']}  "
              f"DBG={'live' if w['instrumented'] else 'not set'}  {w['verdict']}"
              + ("" if p.exists() else "   (no json)"))
    # the decisive line: the prev-token consumer shares prev_token_expert_ids with layer-ahead
    print()
    print("  The prev-token consumer reads the SAME `prev_token_expert_ids`/`prev_token_valid` pair")
    print("  that layer-ahead reads. Its own line separates 'no layers valid' from 'all resident':")
    for lab in ("p1_prevtoken_8g", "p3_both_4g"):
        lp = _resolve_log(probe_logs.get(lab))
        if lp is None:
            continue
        lines = re.findall(r"CGC-PREV-PF: trigger ASYNC queued=(\d+)/(\d+) cold "
                           r"\(valid_layers=(\d+)/(\d+), resident=(\d+), total_queued=(\d+)\)",
                           lp.read_text(errors="replace"))
        if not lines:
            continue
        vl = st.median([int(x[2]) for x in lines])
        res = st.median([int(x[4]) for x in lines])
        cold = sum(int(x[1]) for x in lines)
        pred = sum(int(x[2]) * 8 for x in lines)      # valid_layers x top_k = predictions offered
        pct = (100.0 * (pred - cold) / pred) if pred else float("nan")
        print(f"    {lab:17s} trigger events={len(lines):3d}  valid_layers median={vl:.0f}/41 "
              f"(min {min(int(x[2]) for x in lines)})  resident median={res}  "
              f"cold offered={cold}  predictions={pred}  => {pct:.2f}% already resident")
    print()
    print("  READ: valid_layers=40/41 => the shared prediction structure IS populated, so this is not")
    print("  a collection bug. resident=320 of 320 => every predicted expert was ALREADY in the pool.")
    print("  The prediction is 'the experts the previous token used', and the pool still holds them")
    print("  by construction, so there is nothing for either consumer to prefetch.")

    print()
    print("=" * 100)
    print("F3 -- fatter reads: is there transfer headroom left?")
    print("=" * 100)
    rs = read_shape(arms["armA_1"][0].get("server_log"), a.workers)
    v3, why3 = f3_verdict(rs)
    if rs.get("found"):
        print(f"  jobs={rs['jobs']}  {rs['mib_per_job']:.2f} MiB/job  us/job={rs['us_per_job']}  "
              f"per-job rate={rs['per_job_mib_s']:.0f} MiB/s")
        print(f"  workers={a.workers} (launcher default; corroborated by the ceil(3m/8) round-trip fit)")
        print(f"  AGGREGATE = {rs['per_job_mib_s']:.0f} x {a.workers} = {rs['aggregate_mib_s']:.0f} MiB/s")
    print(f"VERDICT: {v3}   {why3}")

    print()
    print("=" * 100)
    print("F4 / F5")
    print("=" * 100)
    L_a, m1_a, L_b, m1_b = [], [], [], []
    for n in order:
        log = arms[n][0].get("server_log")
        if not log:
            continue
        lp = Path(log)
        if not lp.exists():
            cand = glob.glob(f"Backup/cgc_logs/*{lp.name}")
            if not cand:
                continue
            lp = Path(cand[0])
        rep = cmr.analyse(cmr.parse_log(lp))
        f5 = rep.get("f5") or {}
        f4ok, f4msg = cmr.f4_check(rep)
        print(f"  {n:9s} F4 {'PASS' if f4ok else 'FAIL'}: {f4msg}")
        # Use the WORK population: `median_cb_ms` is taken over work rows, so L must be too. Reading
        # L off the pooled rows (which include the near-empty shadow rows) inflates the "fully warm"
        # share from 3-4% to 58% -- measured, and the reason this reads the matching population.
        pop = (f5.get("populations") or {}).get("work") or (f5.get("populations") or {}).get("all") or {}
        print(f"  {'':9s} F5 [work] n={pop.get('n_steps')} L_mean={pop.get('L_mean'):.1f} "
              f"median={pop.get('L_median')} max={pop.get('L_max')} "
              f"warm={100 * (pop.get('warm_share') or 0):.0f}%  "
              f"median(cb|m=1)={f5.get('cb_at_m1_median')}")
        if pop.get("L_mean") is not None:
            (L_a if n.startswith("armA") else L_b).append(pop["L_mean"])
            (m1_a if n.startswith("armA") else m1_b).append(f5["cb_at_m1_median"])
    v5, why5 = f5_verdict(L_a, m1_a, L_b, m1_b)
    print(f"VERDICT: {v5}   {why5}")

    print()
    print("NOT CLAIMED: any absolute t/s for an instrumented arm (both carry CGC_HOOK_SPLIT and the")
    print("box was under swap pressure); any cross-arm mean_len (the renderer's own rule).")
    return 0


# ── selftest: the rules must be able to fire in every direction ─────────────────────────────────

def selftest() -> int:
    bad = 0

    def ck(name, cond, detail=""):
        nonlocal bad
        if cond:
            print(f"  ok   {name}")
        else:
            bad += 1
            print(f"  FAIL {name} {detail}")

    ck("F2 PASS at exactly the 0.50 boundary", f2_verdict([10.0] * 4, [5.0] * 4)[0] == "PASS")
    ck("F2 FAIL at exactly the 0.75 boundary", f2_verdict([10.0] * 4, [7.5] * 4)[0] == "FAIL")
    ck("F2 PARTIAL between the boundaries", f2_verdict([10.0] * 4, [6.0] * 4)[0] == "PARTIAL")
    ck("F2 NO-DATA on an empty arm", f2_verdict([], [1.0])[0] == "NO-DATA")

    ck("F3 CLOSED at copy speed",
       f3_verdict({"found": True, "per_job_mib_s": 200.0, "workers": 8,
                   "aggregate_mib_s": 1600.0})[0] == "CLOSED")
    # the exact miss this tool exists to prevent: the raw 226 MiB/s would re-open F3
    ck("F3 RE-OPENED if the worker factor is NOT applied",
       f3_verdict({"found": True, "per_job_mib_s": 226.0, "workers": 1,
                   "aggregate_mib_s": 226.0})[0] == "RE-OPENED")
    ck("...and CLOSED once it is (226 x 8 = 1808)",
       f3_verdict({"found": True, "per_job_mib_s": 226.0, "workers": 8,
                   "aggregate_mib_s": 1808.0})[0] == "CLOSED")
    ck("F3 NO-EVIDENCE without a read shape line", f3_verdict({"found": False})[0] == "NO-EVIDENCE")

    ck("F5 REFUTED when L is flat",
       f5_verdict([12.0, 11.0], [1.1, 1.2], [12.1, 11.9], [1.15, 1.18])[0] == "REFUTED")
    ck("F5 SUPPORTED when L drops",
       f5_verdict([12.0, 11.0], [1.1, 1.2], [4.0, 4.1], [1.1, 1.2])[0] == "SUPPORTED")

    # witness classifier, against the real known-negative: a log with PFDBG live and no queueing
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write("PFDBG batch l=0 n=15 busy_before=142\n"
                 "PFDBG batch l=1 n=16 busy_before=142\n"
                 "llama_expert_cache: final stats: runtime requests=8 hits=99 misses=1 "
                 "(hit rate 99.0%)  prewarm req=0 hit=0 miss=0  resident=0.00 MiB file_reads=1 "
                 "pread_usec=1 fill_batch_usec=1 fill_wait_us=1 prefetch=0/0\n")
        p = Path(fh.name)
    w = prefetch_witness(str(p))
    ck("witness says NEVER-CALLED when PFDBG is live but nothing queued",
       w["verdict"] == "NEVER-CALLED", str(w))
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh2:
        fh2.write("PFDBG batch l=0 n=1 busy_before=1\n"
                  "PFDBG drop: resident l=3 e=9\n"
                  "llama_expert_cache: final stats: runtime requests=1 hits=1 misses=0 "
                  "(hit rate 100.0%)  prewarm req=0 hit=0 miss=0  resident=0.00 MiB file_reads=0 "
                  "pread_usec=0 fill_batch_usec=0 fill_wait_us=0 prefetch=0/0\n")
        p2 = Path(fh2.name)
    w2 = prefetch_witness(str(p2))
    ck("witness says CALLED once a prefetch_slot trace appears",
       w2["verdict"] == "CALLED", str(w2))
    print()
    print("selftest: " + ("PASS" if bad == 0 else f"FAIL ({bad})"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
