#!/usr/bin/env python3
"""Print the per-launch GPU decomposition of every preserved cert_runs launch.

Answers exactly one question: of the wall time the engine spends waiting for a prefill graph's
command buffers (`wait`), how much is the GPU actually busy (`gpu_union`)? The engine states its
own decision rule at ggml-backend.cpp:2115-2122 -- union/wait >= 70% means the wait IS GPU
execution, <= 40% means launch/completion latency.

`--gpu-util` switches to the *second, independent* instrument: the 1 Hz `ioreg -r -c IOAccelerator`
Device Utilization % trace. That trace is a bare `HH:MM:SS <pct>` series with no run labels, so the
alignment to runs is by clock: each contiguous high-utilisation block is matched to the
`cert_runs/*/summary.json` whose mtime (= that run's completion time) falls inside the block.

Why the second instrument matters: the numbers above come from Metal's own
`GPUStartTime/GPUEndTime`. Device Utilization % comes from a different set of counters, so
agreement between the two is what makes "the GPU is saturated, the slow run is not idle" a
two-source claim rather than a one-source assertion.

Usage:
    /opt/homebrew/bin/python3 scripts/check/prefill_gputime_report.py [glob]
    /opt/homebrew/bin/python3 scripts/check/prefill_gputime_report.py --gpu-util [path]
"""
from __future__ import annotations

import glob
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "scripts" / "check" / "prefill_certifiability.py"
DEFAULT_GPU_UTIL = "Backup/cgc_logs/gpu_util_prefill_paired_20260916.log"

spec = importlib.util.spec_from_file_location("pc", str(CERT))
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def _hhmmss(t: str) -> int:
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def parse_gpu_util(path: Path) -> list[tuple[str, int]]:
    """`ioreg` trace -> [(HH:MM:SS, util_pct)]. Lines that are not `<time> <int>` are skipped."""
    rows = []
    for ln in path.read_text(errors="replace").splitlines():
        p = ln.split()
        if len(p) == 2:
            try:
                rows.append((p[0], int(p[1])))
            except ValueError:
                pass
    return rows


def load_blocks(rows: list[tuple[str, int]], thr: int = 60) -> list[list]:
    """Contiguous stretches at or above `thr`. The rifts below it are the between-run reloads."""
    blocks: list[list] = []
    cur = None
    for t, v in rows:
        if v >= thr:
            if cur is None:
                cur = [t, t, [v]]
            else:
                cur[1] = t
                cur[2].append(v)
        elif cur is not None:
            blocks.append(cur)
            cur = None
    if cur is not None:
        blocks.append(cur)
    return blocks


def run_mtimes() -> list[tuple[str, str, float]]:
    """[(cert_runs dir name, HH:MM:SS of summary.json, prefill t/s)] for every preserved run."""
    out = []
    for sp in sorted(glob.glob(str(ROOT / "Backup/llama_bench/cert_runs/*/summary.json"))):
        ts = None
        try:
            for arm in json.load(open(sp)):
                for r in arm.get("rows", []):
                    if r.get("n_prompt", 0) > 0:
                        ts = r["avg_ts"]
        except Exception:
            continue
        if ts is None:
            continue
        import datetime
        mt = datetime.datetime.fromtimestamp(Path(sp).stat().st_mtime).strftime("%H:%M:%S")
        out.append((Path(sp).parent.name, mt, ts))
    return out


def report_gpu_util(path: Path) -> int:
    rows = parse_gpu_util(path)
    if not rows:
        print(f"{path}: no `<HH:MM:SS> <pct>` samples parsed")
        return 2
    runs = run_mtimes()
    if not runs:
        print("no preserved cert_runs summary.json to label the trace with")
        return 2

    # Attribute samples to runs by clock. Each run's summary.json mtime is when that run FINISHED,
    # so run k owns the window (mtime of run k-1, mtime of run k]. This is more robust than
    # "which block contains the mtime": a single sub-threshold sample between the prefill and the
    # short decode tail splits the trace into two blocks, and the mtime then lands in the small
    # fragment -- which is how a naive containment rule mis-attributes a run to its own tail.
    s_first = _hhmmss(rows[0][0])
    s_last = _hhmmss(rows[-1][0])
    ordered = sorted(runs, key=lambda r: _hhmmss(r[1]))
    print(f"{path.name}: {len(rows)} samples, {rows[0][0]} -> {rows[-1][0]}\n")

    def longest_load(seg: list[tuple[str, int]]) -> list:
        best, cur = [], []
        for t, v in seg:
            if v >= 60:
                cur.append((t, v))
                if len(cur) > len(best):
                    best = list(cur)
            else:
                cur = []
        return best

    print(f"{'run':>24s} {'done':>9s} {'pp t/s':>8s} | {'window':>17s} {'n':>4s} | "
          f"{'prefill load block':>22s} {'n':>4s} {'mean':>6s} {'peak':>5s} {'min':>5s}")
    prev = s_first
    for name, mt, ts in ordered:
        s_mt = _hhmmss(mt)
        if s_mt < s_first or s_mt > s_last:
            continue
        window = [(t, v) for t, v in rows if prev < _hhmmss(t) <= s_mt]
        prev = s_mt
        load = longest_load(window)
        w_txt = f"{window[0][0] if window else ''}->{window[-1][0] if window else ''}"
        if load:
            vs = [v for _, v in load]
            print(f"{name:>24s} {mt:>9s} {ts:>8.2f} | {w_txt:>17s} {len(window):>4d} | "
                  f"{load[0][0]+' -> '+load[-1][0]:>22s} {len(vs):>4d} {sum(vs)/len(vs):>6.1f} "
                  f"{max(vs):>5d} {min(vs):>5d}")
        else:
            print(f"{name:>24s} {mt:>9s} {ts:>8.2f} | {w_txt:>17s} {len(window):>4d} | "
                  f"{'(no sample >= 60%)':>22s}")

    rifts = [v for _, v in rows if v < 60]
    if rifts:
        print(f"\n  between-run rifts (util < 60%): {len(rifts)} samples, "
              f"range {min(rifts)}-{max(rifts)}%")
    print("  read: if every run's prefill load block shows a high mean AND a 99% peak, the GPU is "
          "saturated in the fast run and the slow run alike,\n        so the difference is the "
          "GPU's work RATE -- not an idle GPU. Utilisation is an occupancy fraction, NOT a "
          "frequency.")
    return 0


def report_decprof_pair(fast_log: Path, slow_log: Path, tps: tuple | None = None,
                        mode: str = "mean") -> int:
    """Layer-by-layer difference between a FAST prefill graph and a SLOW one.

    This is the question `docs/PREFILL250_THERMAL_TRANSIENT_20260916.html` §8 item 1 asks and could
    not answer with the original instrument: **did one layer get slower, or did every layer get
    slower by the same factor?** The answer discriminates the two mechanisms the report left open --
    a layer-localised cost is an implementation problem, a uniform scaling is a clock/power ceiling.

    The comparison is deliberately made on the per-layer `wait` SERIES and not on its spread alone,
    because the spread is dominated by a pipeline ramp: with `submit_ahead`, layer i waits only on
    the GPU backlog that has accumulated by layer i, so the first few layers read low in *every*
    state (in the 285 t/s reference the minimum was L0 at 112 ms against a 155 ms plateau). A pair
    whose ratios are flat across the common layers is uniform *even when* each series individually
    has a large max/mean.

    `mode` selects WHICH prefill graph to read, and the default is `mean` (per-layer average over
    every prefill graph in the log) rather than `first`. That default is load-bearing: `-r 3` leaves
    several prefill graphs in one log, the first of which carries model-load and first-touch effects.
    Reading only the first graph attributes a once-per-process transient to whichever layers happen
    to be executing during it -- and the per-layer profile of a transient is not the per-layer
    profile of the steady state. Averaging over all graphs dilutes that; the per-graph table printed
    above the pair makes the transient visible instead of silent.
    """
    fa = pc.harvest_decprof(fast_log.read_text(errors="replace"))
    sl = pc.harvest_decprof(slow_log.read_text(errors="replace"))
    if not fa or not sl:
        print("one of the logs has no CGC-DECPROF output -- was the arm `prefill250-decprof` used?")
        return 2

    def pick_prefill(d, tag: str):
        pre = [g for g in d.get("dp_series", []) if (g["ntok"] or 0) > 1]
        if not pre:
            return None, pre
        if pre:
            print(f"{tag}: {len(pre)} prefill graph(s), per-graph shape:")
            for i, g in enumerate(pre):
                lay = {l[0]: l[1] for l in g["lay"]}
                worst = max(lay.items(), key=lambda kv: kv[1]) if lay else (0, 0.0)
                print(f"    g{i} ntok={g['ntok']:>5d} layers={len(lay):>3d} "
                      f"total={g['total_ms']:>9.1f} ms wait={g['wait_ms']:>9.1f} ms "
                      f"worst=L{worst[0]}({worst[1]:.1f} ms)")
        if mode == "first":
            return pre[0], pre
        if mode == "last":
            return pre[-1], pre
        acc: dict = {}
        for g in pre:
            for l in g["lay"]:
                acc.setdefault(l[0], []).append(l[1])
        n = len(pre)
        avg = lambda k: sum(g[k] for g in pre) / n          # noqa: E731
        return {"ntok": pre[0]["ntok"],
                "lay": [[li, sum(v) / len(v)] for li, v in sorted(acc.items())],
                "total_ms": avg("total_ms"), "wait_ms": avg("wait_ms"),
                "cb_ms": avg("cb_ms"), "submit_ms": avg("submit_ms")}, pre

    gf, pre_f = pick_prefill(fa, "fast")
    gs, pre_s = pick_prefill(sl, "slow")
    if gf is None or gs is None:
        print("no prefill graph (ntok > 1) in one of the logs")
        return 2
    if mode == "mean":
        print(f"(mode=mean: per-layer wait averaged over {len(pre_f)} x fast and "
              f"{len(pre_s)} x slow prefill graphs)")

    wf = {l[0]: l[1] for l in gf["lay"]}
    ws_ = {l[0]: l[1] for l in gs["lay"]}
    common = sorted(set(wf) & set(ws_))
    if not common:
        print("the two logs share no layer indices")
        return 2

    print(f"fast: {fast_log.name}\n  ntok={gf['ntok']} layers={len(wf)} "
          f"total={gf['total_ms']:.1f} ms wait={gf['wait_ms']:.1f} cb={gf['cb_ms']:.1f} "
          f"submit={gf['submit_ms']:.1f}")
    print(f"slow: {slow_log.name}\n  ntok={gs['ntok']} layers={len(ws_)} "
          f"total={gs['total_ms']:.1f} ms wait={gs['wait_ms']:.1f} cb={gs['cb_ms']:.1f} "
          f"submit={gs['submit_ms']:.1f}")
    print(f"\ncommon layers: {len(common)} of {len(wf)} (the fast log's coverage is the limit)\n")
    print(f"{'layer':>6s} {'fast ms':>9s} {'slow ms':>9s} {'slow/fast':>10s}")
    ratios = []
    for l in common:
        r = ws_[l] / wf[l] if wf[l] else 0.0
        ratios.append(r)
        print(f"{l:>6d} {wf[l]:>9.1f} {ws_[l]:>9.1f} {r:>10.2f}")
    n = len(ratios)
    mean = sum(ratios) / n
    var = sum((x - mean) ** 2 for x in ratios) / n
    sd = var ** 0.5
    # L0 is the pipeline-ramp layer: it waits on almost no accumulated GPU work, so its ratio is
    # pulled toward 1 no matter how much the GPU slowed. Excluding it is the honest read, and it is
    # excluded explicitly rather than by tweaking a threshold.
    wo = [ws_[l] / wf[l] for l in common if l != 0 and wf[l]]
    wo_mean = sum(wo) / len(wo) if wo else 0.0
    wo_cv = (sum((x - wo_mean) ** 2 for x in wo) / len(wo)) ** 0.5 / wo_mean * 100.0 if wo else 0.0

    print(f"\n  ratio slow/fast over all {n} common layers : mean={mean:.2f}x  "
          f"cv={100.0 * sd / mean:.1f}%  min={min(ratios):.2f}  max={max(ratios):.2f}")
    print(f"  excluding L0 (pipeline ramp) over {len(wo)} layers : mean={wo_mean:.2f}x  cv={wo_cv:.1f}%")
    if tps:
        print(f"  throughput slow/fast                       : {tps[0] / tps[1]:.2f}x "
              f"({tps[0]:.2f} -> {tps[1]:.2f} t/s)")
        print(f"  CHECK: the per-layer wait ratio must track the throughput ratio if the graph scaled "
              f"as a whole -- {wo_mean:.2f}x vs {tps[0] / tps[1]:.2f}x")

    # A mean that matches throughput does NOT by itself mean the ratio is flat: a few layers
    # blowing up while the rest barely move can average out to the same number. Separate the two
    # halves so the reader sees BOTH the global rate effect and any localised excess, and weight
    # the excess by how much of the total gap it actually accounts for -- a 4x layer that carries
    # 2% of the gap is a different finding from a 4x layer that carries 40% of it.
    med = sorted(wo)[len(wo) // 2] if wo else 0.0
    hot = [(l, ws_[l] / wf[l]) for l in common if l != 0 and wf[l] and med and ws_[l] / wf[l] >= 1.5 * med]
    gap_ms = (gs["wait_ms"] - gf["wait_ms"]) or (sum(ws_.values()) - sum(wf.values()))
    exc_ms = sum((ws_[l] - med * wf[l]) for l, _ in hot) if med else 0.0
    if hot:
        top = ", ".join(f"L{l}({r:.2f}x)" for l, r in sorted(hot, key=lambda t: -t[1])[:8])
        share = (100.0 * exc_ms / gap_ms) if gap_ms else 0.0
        print(f"  LOCALISED excess: median ratio {med:.2f}x, but {len(hot)} layer(s) >= 1.5x median "
              f"-> {top}")
        print(f"                    those layers carry ~{exc_ms:.0f} ms of the {gap_ms:.0f} ms gap "
              f"({share:.0f}%); the rest is a broad rate effect")
    else:
        print(f"  no layer exceeds 1.5x the median ratio ({med:.2f}x) -> no localised excess")

    verdict = ("UNIFORM: every layer scaled by the same factor => the cost is the GPU's work RATE "
               "(clock / power ceiling), not a layer" if wo_cv <= 15.0 else
               "NOT UNIFORM: the ratio varies by layer => look for a layer-localised cost")
    if hot and wo_cv <= 15.0:
        verdict = ("UNIFORM but check the excess list above -- cv is low yet specific layers "
                   "exceed 1.5x the median")
    print(f"\n  VERDICT: {verdict}")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--decprof-pair":
        rest = argv[1:]
        tps = None
        mode = "mean"
        want = []
        i = 0
        while i < len(rest):
            if rest[i] == "--tps":
                tps = (float(rest[i + 1]), float(rest[i + 2]))
                i += 3
                continue
            if rest[i] == "--graph":
                mode = rest[i + 1]
                i += 2
                continue
            want.append(rest[i])
            i += 1
        if len(want) != 2 or mode not in ("first", "last", "mean"):
            print("usage: --decprof-pair <fast log> <slow log> [--tps <fast> <slow>] "
                  "[--graph first|last|mean]")
            return 2
        ps = []
        for w in want:
            p = Path(w)
            ps.append(p if p.is_absolute() else ROOT / p)
        return report_decprof_pair(ps[0], ps[1], tps, mode)
    if argv and argv[0] == "--gpu-util":
        p = Path(argv[1]) if len(argv) > 1 else ROOT / DEFAULT_GPU_UTIL
        if not p.is_absolute():
            p = ROOT / p
        return report_gpu_util(p)

    pats = argv or ["Backup/llama_bench/cert_runs/20260916_0156*/",
                    "Backup/llama_bench/cert_runs/20260916_0205*/"]
    dirs: list[str] = []
    for p in pats:
        dirs.extend(sorted(glob.glob(str(ROOT / p) if not p.startswith("/") else p)))
    if not dirs:
        print(f"no launch directories matched {pats}")
        return 2

    print(f"{'run':>6s} {'pp t/s':>8s} | {'prefill graph (the big one)':^52s} | "
          f"{'all graphs':^26s}")
    print(f"{'':>6s} {'':>8s} | {'wait_s':>7s} {'union_s':>8s} {'u/w%':>5s} {'gap_ms':>7s} "
          f"{'busy/w%':>8s} {'bufs':>5s} {'skip':>4s} | {'wait_s':>7s} {'union_s':>8s} {'u/w%':>5s}")

    rows = []
    for d in dirs:
        lp = Path(d) / "llama_bench_prefill250.stderr.log"
        sp = Path(d) / "summary.json"
        if not lp.exists():
            continue
        raw = lp.read_text(errors="replace")
        seg = pc.harvest_segments(raw)
        seg.update(pc.harvest_gputime(raw))
        ts = None
        try:
            for arm in json.load(open(sp)):
                for r in arm.get("rows", []):
                    if r.get("n_prompt", 0) > 0:
                        ts = r["avg_ts"]
        except Exception:
            pass
        if ts is None or "ppg_wait_ms" not in seg:
            continue
        rows.append((Path(d).name, ts, seg))
        s = seg
        print(f"{Path(d).name[-5:]:>6s} {ts:>8.2f} | {s['ppg_wait_ms']/1000:>7.2f} "
              f"{s['ppg_union_ms']/1000:>8.2f} {s['ppg_union_pct']:>5.0f} "
              f"{s['ppg_gap_ms']:>7.0f} "
              f"{100.0*s['ppg_wait_ms']/max(s['ppg_wait_ms'],1):>8.0f} "
              f"{s['ppg_bufs']:>5d} {s['gpu_skipped']:>4d} | "
              f"{s.get('gpu_wait_ms',0)/1000:>7.2f} {s.get('gpu_union_ms',0)/1000:>8.2f} "
              f"{s.get('gpu_union_pct',0):>5.0f}")

    if len(rows) >= 2:
        rows.sort(key=lambda r: r[1])
        slow, fast = rows[0], rows[-1]
        print(f"\n  slowest {slow[0][-5:]}: {slow[1]:.2f} t/s   fastest {fast[0][-5:]}: "
              f"{fast[1]:.2f} t/s   ratio {fast[1]/slow[1]:.2f}x")
        sf, ss = fast[2], slow[2]
        for k, lbl in (("ppg_wait_ms", "prefill graph wait"),
                       ("ppg_union_ms", "prefill graph GPU busy"),
                       ("seg_wait_s", "seg-wait total"),
                       ("seg_cb_s", "seg-encode total")):
            if k in sf and k in ss and ss[k]:
                print(f"    {lbl:24s} fast={sf[k]:10.1f}  slow={ss[k]:10.1f}  "
                      f"slow/fast={ss[k]/sf[k]:.2f}x")
        print(f"    GPU busy / wait          fast={sf['ppg_union_pct']:.0f}%  "
              f"slow={ss['ppg_union_pct']:.0f}%   (>=70% => the wait IS GPU execution)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
