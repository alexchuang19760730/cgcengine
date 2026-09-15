#!/usr/bin/env python3
"""Locate the thermal recovery time constant tau by sweeping `--idle-before`.

THE QUESTION
------------
`docs/PREFILL250_THERMAL_TRANSIENT_20260916.html` established that prefill throughput is bimodal
on this fanless `Mac16,12`: ~286-299 t/s when the machine is cold, 155-184 t/s once it is hot, with
byte-identical workload counters in both states (so the difference is GPU execution rate, not any
data movement). It also bracketed the recovery constant to `67 s < tau <= 247 s` -- but that bracket
was *derived* from preserved logs (session end -> next session's first launch), not measured, so it
came with two samples per side and no shape.

This script measures it directly. `prefill_certifiability.py --idle-before N` sleeps N seconds
before its first launch, which makes "how long has the machine been quiet" the independent variable
instead of a side effect of how the operator happened to run the commands.

DESIGN
------
Sessions run in ONE sequence with increasing idle, which is what makes each session's premise real:
the previous session ends hot, then the machine idles for `idle_k` seconds, then the next session's
run 1 measures the recovered state. Each session takes `--runs 2`:

    run 1 = the state that `idle_k` seconds of quiet bought   <- the measurement
    run 2 = the reference immediately after run 1             <- the hot control, same process shape

Every session uses the `prefill250-decprof` arm, so run 1 additionally yields the per-layer
wait/cb/submit split of the first prefill graph. That is the difference between "it was slower" and
"every layer was slower by the same factor" -- see eng-src-0011.

COST
----
Each idle value costs `idle_k + runs * ~55 s`. The default set is ~26 minutes of wall time.

Usage:
    /opt/homebrew/bin/python3 scripts/check/prefill_idle_sweep.py
    /opt/homebrew/bin/python3 scripts/check/prefill_idle_sweep.py --idles 0,90,240 --runs 2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "scripts" / "check" / "prefill_certifiability.py"
PY = sys.executable


def one(idle: int, args, runs: int | None = None) -> dict:
    """Run one certifiability session with `idle` seconds of quiet before its first launch."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    logdir = ROOT / args.logdir / f"idle{idle:04d}_{stamp}"
    jpath = logdir / "sweep.json"
    logdir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(CERT), "--arm", args.arm, "--runs", str(runs or args.runs),
           "--idle-before", str(idle), "--logdir", str(logdir), "--json", str(jpath),
           "--prompt", str(args.prompt)]
    print(f"\n{'#'*100}\n# idle-before = {idle}s   -> {logdir.relative_to(ROOT)}\n{'#'*100}",
          flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    wall = time.time() - t0
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stdout.write(proc.stderr[-2000:])

    out: dict = {"idle_s": idle, "wall_s": round(wall, 1), "rc": proc.returncode,
                 "logdir": str(logdir.relative_to(ROOT))}
    try:
        res = json.loads(jpath.read_text())
    except Exception as e:                                   # noqa: BLE001 -- report, don't crash
        out["error"] = f"summary unreadable: {e}"
        return out
    for r in (res.get("runs") or []):
        if r.get("skipped") or not r.get("pp"):
            continue
        k = r["run"]
        pp = r["pp"][0]
        seg = r.get("seg") or {}
        out[f"run{k}_tps"] = round(pp["avg_ts"], 2)
        out[f"run{k}_sd"] = round(pp.get("stddev_ts") or 0.0, 2)
        out[f"run{k}_wall_s"] = r.get("wall_s")
        out[f"run{k}_free_gib"] = round((r.get("pre") or {}).get("free_gib", 0.0), 2)
        out[f"run{k}_swap_mib"] = round((r.get("pre") or {}).get("swap_used_mib", 0.0), 0)
        out[f"run{k}_nonresident_pct"] = seg.get("nonresident_pct")
        # the per-layer shape of that run's first prefill graph
        out[f"run{k}_dpq1_wait_ms"] = seg.get("dpq1_wait_ms")
        out[f"run{k}_dpq1_cb_ms"] = seg.get("dpq1_cb_ms")
        out[f"run{k}_dpq1_submit_ms"] = seg.get("dpq1_submit_ms")
        out[f"run{k}_dpq1_lay_w_median_ms"] = seg.get("dpq1_lay_w_median_ms")
        out[f"run{k}_dpq1_lay_w_iqr_over_median"] = seg.get("dpq1_lay_w_iqr_over_median")
        out[f"run{k}_dpq1_lay_w_frac_within_25pct"] = seg.get("dpq1_lay_w_frac_within_25pct")
        out[f"run{k}_dpq1_lay_w_argmax_layer"] = seg.get("dpq1_lay_w_argmax_layer")
        out[f"run{k}_dpq1_layers"] = seg.get("dpq1_layers_seen")
        out[f"run{k}_gpu_union_pct"] = seg.get("gpu_union_pct")
        out[f"run{k}_gpu_skipped"] = seg.get("gpu_skipped")
    # The recovery ratio the sweep is actually looking for: how much cheaper (per layer) was the
    # first prefill graph of run 1 than that of run 2 in the same session. < 1 means run 1 was the
    # *faster* one, i.e. `idle_s` seconds of quiet did buy something.
    m1 = out.get("run1_dpq1_lay_w_median_ms")
    m2 = out.get("run2_dpq1_lay_w_median_ms")
    if m1 and m2:
        out["session_lay_w_ratio_run1_over_run2"] = round(m1 / m2, 3)
    return out


def _pc():
    """Import the certifiability harness so the CURRENT harvester can be replayed over old logs."""
    sys.path.insert(0, str(CERT.parent))
    import prefill_certifiability as pc
    return pc


def _layer_stats(log: Path) -> dict | None:
    """Per-layer shape of one launch, read by the harvester as it exists NOW.

    Averaged over every prefill graph in the log, for the reason lesson `eng-mh-0027` records: the
    first graph of a process carries a once-per-launch load/first-touch transient, and a transient
    lands on whichever layers happened to be running, which is not a property of those layers.
    """
    pc = _pc()
    h = pc.harvest_decprof(log.read_text(errors="replace"))
    pre = [g for g in h.get("dp_series", []) if (g.get("ntok") or 0) > 1]
    if not pre:
        return None
    acc: dict = {}
    for g in pre:
        for l in g["lay"]:
            acc.setdefault(l[0], []).append(l[1])
    lays = sorted(acc)
    vals = [sum(acc[l]) / len(acc[l]) for l in lays]
    vs = sorted(vals)
    n = len(vs)
    med = vs[n // 2] if n % 2 else 0.5 * (vs[n // 2 - 1] + vs[n // 2])
    return {
        "graphs": len(pre), "ntok": pre[0]["ntok"], "med": med,
        "iqr": (vs[(3 * n) // 4] - vs[n // 4]) / med,
        "within": sum(1 for v in vals if abs(v - med) <= 0.25 * med) / n,
        "nlay": len(lays), "argmax": lays[vals.index(max(vals))],
    }


def reharvest(jpath: Path, preheat: str | None, extra: str | None) -> int:
    """Re-derive the sweep's table from the saved logs with the harvester as it exists NOW.

    Why this exists at all: `prefill_idle_sweep.py` calls the harness as a SUBPROCESS per session,
    so each session was analysed by the code present at that moment -- but this script's own
    aggregation was fixed at process start. Editing the harvester mid-sweep therefore produces a
    run where the per-session summaries and the summary table are different vintages, and neither
    one complains. The logs are kept, so the fix is cheap: re-read all of them with one version of
    the code and treat THAT as the number. See lesson `eng-mh-0029`.
    """
    rows = json.loads(jpath.read_text())
    print(f"re-harvesting {len(rows)} session(s) from {jpath} with the CURRENT harvester")
    print(f"  harvester: {CERT.relative_to(ROOT)}  mtime={time.strftime('%H:%M:%S', time.localtime(CERT.stat().st_mtime))}")
    hdr = (f"{'idle s':>7s} {'run':>4s} {'t/s':>8s} {'graphs':>7s} {'ntok':>6s} {'med ms':>8s} "
           f"{'iqr/med':>8s} {'within25':>9s} {'nlay':>5s} {'argmax':>7s}   log")
    print("\n" + hdr)
    print("-" * len(hdr))
    shown = 0
    seq: list = []
    if preheat:
        seq.append(("pre", 1, None, preheat))
    for r in rows:
        for k in (1, 2):
            seq.append((r.get("idle_s"), k, r.get(f"run{k}_tps"), r.get("logdir") or ""))
    for idle, k, tps, base in seq:
        pat = base if base.endswith(".log") else f"{base}/*_run0{k}/llama_bench_*decprof.stderr.log"
        cand = sorted(Path().glob(pat))
        if not cand:
            print(f"{str(idle):>7s} {k:>4d} {str(tps):>8s}  (no decprof log for {pat})")
            continue
        st = _layer_stats(cand[0])
        if st is None:
            print(f"{str(idle):>7s} {k:>4d} {str(tps):>8s}  (no prefill graph in {cand[0].name})")
            continue
        t = f"{tps:>8.2f}" if isinstance(tps, (int, float)) else f"{'-':>8s}"
        print(f"{str(idle):>7s} {k:>4d} {t} {st['graphs']:>7d} {st['ntok']:>6d} {st['med']:>8.1f} "
              f"{st['iqr']:>8.3f} {st['within']:>9.0%} {st['nlay']:>5d} {st['argmax']:>7d}   {cand[0].name}")
        shown += 1
    print(f"\n  {shown} launch(es) re-read. Every row here comes from ONE version of the harvester,")
    print("  which is the only way the table can be internally consistent.")
    if extra:
        print("\n  extra launches (not in the sweep json):")
        for p in sorted(Path().glob(extra)):
            st = _layer_stats(p)
            if st:
                print(f"    {p.parent.name}/{p.name}: med={st['med']:.1f} iqr/med={st['iqr']:.3f} "
                      f"within={st['within']:.0%} nlay={st['nlay']} argmax=L{st['argmax']} "
                      f"graphs={st['graphs']} ntok={st['ntok']}")
    return 0


def provenance() -> None:
    """Print the identity of the code that is about to be used.

    The trap this guards against has already happened once: this script keeps the code it imported
    at start, so a mid-run edit to the harvester leaves the table describing one vintage and the
    per-session summaries another. Printing the mtime up front makes that visible in the log rather
    than only in the artifacts afterwards.
    """
    for p in (Path(__file__), CERT):
        st = p.stat()
        print(f"  code: {p.relative_to(ROOT)}  mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}  "
              f"bytes={st.st_size}", flush=True)


def report(rows: list[dict]) -> None:
    print(f"\n{'='*118}\n  IDLE SWEEP -- prefill pp{rows[0].get('prompt', '')} "
          f"(arm {rows[0].get('arm', '')})\n{'='*118}")
    hdr = (f"{'idle s':>7s} {'wall s':>7s} | {'run1 t/s':>9s} {'run2 t/s':>9s} {'gain':>7s} | "
           f"{'r1 lay med':>11s} {'r2 lay med':>11s} {'r1/r2':>7s} {'iqr/med':>8s} "
           f"{'within25':>9s} | {'union%':>7s} {'skip':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        gain = "-"
        if r.get("run1_tps") and r.get("run2_tps"):
            gain = f"{100.0 * (r['run1_tps'] / r['run2_tps'] - 1.0):+.1f}%"

        def g(k, f="{:>9.2f}"):
            v = r.get(k)
            return f.format(v) if isinstance(v, (int, float)) else f"{'-':>9s}"

        frac = r.get("run1_dpq1_lay_w_frac_within_25pct")
        print(f"{r.get('idle_s', 0):>7d} {r.get('wall_s', 0):>7.0f} | "
              f"{g('run1_tps')} {g('run2_tps')} {gain:>7s} | "
              f"{g('run1_dpq1_lay_w_median_ms', '{:>11.1f}')} "
              f"{g('run2_dpq1_lay_w_median_ms', '{:>11.1f}')} "
              f"{g('session_lay_w_ratio_run1_over_run2', '{:>7.3f}')} "
              f"{g('run1_dpq1_lay_w_iqr_over_median', '{:>8.3f}')} "
              f"{(f'{frac:.0%}' if isinstance(frac, float) else '-'):>9s} | "
              f"{g('run1_gpu_union_pct', '{:>7.1f}')} "
              f"{(str(r.get('run1_gpu_skipped')) if r.get('run1_gpu_skipped') is not None else '-'):>5s}")
    print("\n  read: tau is the smallest idle whose run-1 throughput lands back in the cold band")
    print("        (>=~250 t/s) with run 2 well below it. A run-1 that already matches run 2 means")
    print("        that idle did not recover the machine.")
    print("  r1/r2  = median per-layer wait of run 1 divided by run 2's, same session, same shape.")
    print("           < 1 is the recovery signature (run 1's prefill graph was the cheaper one).")
    print("  iqr/med + within25 describe the layer SERIES: a uniformly scaled graph keeps a tight")
    print("           spread, a graph carried by a few layers does not. Judge on these, not on")
    print("           peak/mean -- the first layers read low in every state (pipeline ramp).")
    print("  NOTE  idle=0 is the already-hot control; it is the first session only if it is the"
          " first listed value.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idles", default="0,60,90,120,150,180,240",
                    help="comma list of --idle-before values, run in this order (default tests the "
                         "derived bracket 67 s < tau <= 247 s and the shape around it)")
    ap.add_argument("--runs", type=int, default=2,
                    help="launches per session: run 1 is the measurement, run 2 the hot control")
    ap.add_argument("--arm", default="prefill250-decprof")
    ap.add_argument("--prompt", default="2048")
    ap.add_argument("--logdir", default="Backup/llama_bench/idle_sweep")
    ap.add_argument("--json", default="Backup/llama_bench/prefill_idle_sweep_20260916.json")
    ap.add_argument("--preheat", default="1",
                    help="run one discarded launch first. Without it the FIRST session inherits "
                         "whatever quiet period preceded the script, so an `idle=0` first datapoint "
                         "would silently be a cold-state measurement mislabelled as the hot control. "
                         "The sweep's premise is that each session starts from a machine made hot by "
                         "the previous one; preheating is what makes that true for session 1 too. "
                         "Set 0 to skip (only when a hot machine is already guaranteed).")
    ap.add_argument("--reharvest", metavar="SWEEP_JSON",
                    help="do not run anything: re-derive the table from an existing sweep json's "
                         "saved logs using the harvester as it exists NOW. Use this after editing "
                         "prefill_certifiability.py, otherwise the table and the per-session "
                         "summaries are different vintages (lesson eng-mh-0029).")
    ap.add_argument("--reharvest-preheat",
                    help="with --reharvest: glob for the discarded preheat launch's decprof log")
    ap.add_argument("--reharvest-extra",
                    help="with --reharvest: an extra glob of launch logs to re-read as well")
    args = ap.parse_args()

    if args.reharvest:
        rp = Path(args.reharvest)
        return reharvest(rp if rp.is_absolute() else ROOT / rp,
                         args.reharvest_preheat, args.reharvest_extra)

    provenance()
    idles = [int(x) for x in args.idles.split(",") if x.strip()]
    if args.preheat not in ("0", ""):
        print(f"\n{'='*100}\n  PREHEAT -- one discarded launch so session 1 starts from a hot "
              f"machine\n{'='*100}", flush=True)
        ph = one(0, args, runs=1)
        print(f"  [preheat] discarded: run1={ph.get('run1_tps')} t/s, "
              f"wall={ph.get('wall_s')} s", flush=True)
    rows = []
    for n, idle in enumerate(idles, 1):
        print(f"\n[sweep] session {n}/{len(idles)}", flush=True)
        r = one(idle, args)
        r["prompt"] = args.prompt
        r["arm"] = args.arm
        rows.append(r)
        jout = ROOT / args.json if not Path(args.json).is_absolute() else Path(args.json)
        jout.parent.mkdir(parents=True, exist_ok=True)
        jout.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"[sweep] wrote {jout.relative_to(ROOT)} ({len(rows)} session(s) so far)", flush=True)

    report(rows)
    jout = ROOT / args.json if not Path(args.json).is_absolute() else Path(args.json)
    print(f"\njson -> {jout.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
