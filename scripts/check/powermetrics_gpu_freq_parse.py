#!/usr/bin/env python3
"""powermetrics_gpu_freq_parse.py — turn a `powermetrics --samplers gpu_power` capture into an answer.

WHY THIS EXISTS
---------------
`scripts/check/powermetrics_gpu_freq.sh --parse` used to `grep -o` four labels and `paste` the
values into four long lines. That answers nothing on its own: a 4-minute capture at `-i 500` is
~480 samples per series, and the question the capture exists to settle -- *was the prefill
slowdown a clock or a power cap* -- is a comparison (cold run vs hot run), not a series.
Dumping the series makes the reader do the segmentation and the averaging by eye.

It also threw away the single most informative field. The real text output is

    **** GPU usage ****
    GPU HW active frequency: 1277 MHz
    GPU HW active residency: 99.35% (396 MHz: .06% 528 MHz: 0% 720 MHz: 0% 924 MHz: 0%
                                     1128 MHz: 0% 1278 MHz: 99%)
    GPU SW requested state: (P1 : 0% ... P6 : 100%)
    GPU idle residency: 0.65%
    GPU Power: 8503 mW

The parenthetical is the DVFS residency distribution -- how much of the active time was spent at
EACH clock step. `GPU HW active frequency` collapses that to one number, so two runs can report
the same "active frequency" while occupying different mixtures of steps. A clock cap shows up as
mass moving off the top step, and only the distribution shows it.

THERMAL (2026-09-16, the second half of the question)
----------------------------------------------------
The cause of a clock cap is either a power ceiling or a thermal one, and those have different
fixes -- so the capture is taken with `--samplers gpu_power,thermal` and `Current pressure level`
is reported per block. On the first real capture it did not just correlate with the clock, it
*drove* it: at 09:12:21 the level flipped Moderate -> Heavy and the clock began falling in the
same sample (1300 -> 1236 -> 1180 -> ... -> 967 MHz) while active residency held at 100.00% the
whole way down. A pressure level that escalates across the cold/hot pair is the difference
between "the clock is capped" and "the clock is capped BY HEAT".

WHY THE BLOCK THRESHOLD IS A HIGH PERCENTILE
--------------------------------------------
Segmentation needs a threshold that says "the GPU is doing the workload, not idling". The first
version used `0.15 * p95` of power, which is a percentile of the WHOLE capture: as soon as the
idle tail grows, p95 slides out of the active mode and into the idle distribution, the threshold
collapses to its 200 mW floor, and idle blips of 240-300 mW are then counted as launches. That is
not hypothetical -- the 2026-09-16 capture carried a 32-minute idle tail (the harness exited at
09:14 but `powermetrics` was never killed, see `powermetrics_gpu_freq.sh`) and came out as 25-31
blocks instead of 3.

Measured, by appending resampled idle samples to the real 09:08-09:15 prefix:

    rule                prefix   +10 min   +32 min   +90 min
    0.15 * p95 (old)         3        6        22        31
    0.15 * p99               3        3         3         6
    0.15 * p99.5             3        3         3         3
    Otsu on log10(power)     3        3       104        13

p99.5 stays inside the active mode down to a ~0.5% duty cycle, which is the regime an idling
capture is in. The block boundaries it produces are also stable (identical +/-1 sample at every
tail length), and they cut the post-run 338 MHz model-unload tail off the launch, which is what
you want. Residual risk: p99.5 tracks the top of the ACTIVE mode, so a capture whose maximum is a
one-off outlier far above the sustained workload could push the threshold too high. The threshold
and the duty cycle are therefore always printed, below, rather than left implicit.

Robustness is deliberate (2026-09-16): the label set was confirmed against real captures from
three independent sources, but this parser still degrades LOUDLY. If the labels ever change, it
prints the GPU-looking lines it did see instead of four empty series -- an instrument that
silently prints nothing is indistinguishable from one that measured nothing.

A capture that failed leaves a file too, and it is NOT a short capture: on 2026-09-16 a 40-byte
file containing exactly `(eval):1: operation not permitted: sudo` was reported only as "the
capture looks empty or was truncated", which is true and useless. The file's own content is the
diagnosis, so it is now printed. Several logs may be given at once (a shell glob is the natural
way to call this after a few captures); every file gets a verdict line, including the ones being
skipped, because a silently-dropped argument is how a failed capture gets mistaken for a good one.

Usage:
  powermetrics_gpu_freq_parse.py <capture.log> [more.log ...]
  powermetrics_gpu_freq_parse.py --json out.json <capture.log> [more.log ...]
  powermetrics_gpu_freq_parse.py --selftest

Exit: 0 parsed, 2 nothing parsed, 3 bad usage.
"""
import argparse
import json
import re
import sys

# Labels, matched tolerantly: exact strings as of macOS 13-15 on Apple Silicon, but anchored
# case-insensitively on the stable part so a reordering of words does not empty the series.
RE_FREQ = re.compile(r"GPU\s+HW\s+active\s+frequency\s*:\s*([0-9.]+)\s*MHz", re.I)
RE_ACT = re.compile(r"GPU\s+HW\s+active\s+residency\s*:\s*([0-9.]+)\s*%", re.I)
RE_IDLE = re.compile(r"GPU\s+idle\s+residency\s*:\s*([0-9.]+)\s*%", re.I)
RE_PWR = re.compile(r"GPU\s+Power\s*:\s*([0-9.]+)\s*mW", re.I)
RE_STEPS = re.compile(r"([0-9]{3,4})\s*MHz\s*:\s*([0-9.]+)\s*%")
RE_ELAPSED = re.compile(r"\(([0-9.]+)\s*ms\s+elapsed\)", re.I)
RE_TS = re.compile(r"Sampled system activity\s*\((.*?)\)\s*\(", re.I)
RE_PRES = re.compile(r"Current pressure level\s*:\s*(\w+)", re.I)
RE_GPULINE = re.compile(r"GPU", re.I)

# Severity of the thermal pressure levels macOS reports, so "did it escalate across the pair" is
# a comparison and not a string match. Two vocabularies exist in the wild -- Nominal/Fair/Serious/
# Critical on newer builds, Nominal/Moderate/Heavy on the builds seen here -- and they are ranked
# together rather than assumed to be exclusive. An unknown word ranks 0 and is still printed.
PRES_RANK = {"Nominal": 0, "Moderate": 1, "Fair": 1, "Heavy": 2, "Serious": 2, "Critical": 3}


def parse(path):
    """One record per `**** GPU usage ****` sample, in file order."""
    recs = []
    cur = None
    elapsed_ms = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RE_ELAPSED.search(line)
            if m:
                if cur:
                    recs.append(cur)
                    cur = None
                try:
                    elapsed_ms = float(m.group(1))
                except ValueError:
                    pass
                ts = RE_TS.search(line)
                cur = {"ts": ts.group(1) if ts else None,
                       "freq": None, "act": None, "idle": None, "power": None,
                       "pres": None, "steps": {}}
                continue
            if cur is None:
                continue
            # thermal comes before the GPU block in the text output, but the file order is not
            # depended on: any label inside this sample's window is attributed to this sample.
            m = RE_PRES.search(line)
            if m:
                cur["pres"] = m.group(1)
                continue
            m = RE_FREQ.search(line)
            if m:
                cur["freq"] = float(m.group(1))
                continue
            m = RE_ACT.search(line)
            if m:
                cur["act"] = float(m.group(1))
                cur["steps"] = {int(a): float(b) for a, b in RE_STEPS.findall(line)}
                continue
            m = RE_IDLE.search(line)
            if m:
                cur["idle"] = float(m.group(1))
                continue
            m = RE_PWR.search(line)
            if m:
                cur["power"] = float(m.group(1))
                continue
    if cur:
        recs.append(cur)
    return recs, elapsed_ms


def pct(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def block_threshold(p):
    """Where "idle" ends and "the workload" begins.

    A HIGH percentile on purpose -- the rationale and the measurement are in the module docstring
    under WHY THE BLOCK THRESHOLD IS A HIGH PERCENTILE. In one line: p95 is a percentile of the
    WHOLE capture, so a long idle tail drags it into the idle distribution and the threshold
    collapses to its floor; p99.5 stays in the active mode down to a ~0.5% duty cycle, which is
    the regime an idling capture is in.
    """
    if not p:
        return 200.0
    return max(200.0, 0.15 * pct(p, 0.995))


def blocks_of(recs, min_len=3, gap_merge=2, thr=None):
    """Contiguous stretches where the GPU is actually working.

    Segmentation is on POWER, not on frequency or residency: those are exactly the quantities
    under test, so using them to define "a run" would beg the question. Power separates idle
    (~tens of mW) from a prefill (tens of W) by three orders of magnitude.

    `thr` exists so the selftest can drive the historical threshold through the same merging code
    and demonstrate that the rule change is what fixes the tail problem, rather than asserting it.
    """
    p = [r["power"] for r in recs if r["power"] is not None]
    if len(p) < 5:
        return []
    if thr is None:
        thr = block_threshold(p)
    active = [i for i, r in enumerate(recs)
              if r["power"] is not None and r["power"] >= thr]
    if not active:
        return []
    groups, run = [], [active[0]]
    for i in active[1:]:
        if i - run[-1] <= gap_merge + 1:
            run.append(i)
        else:
            groups.append(run)
            run = [i]
    groups.append(run)
    return [(g[0], g[-1]) for g in groups if len(g) >= min_len]


def summarise(recs, span):
    lo, hi = span
    cut = recs[lo:hi + 1]
    freqs = [r["freq"] for r in cut if r["freq"] is not None]
    acts = [r["act"] for r in cut if r["act"] is not None]
    pwrs = [r["power"] for r in cut if r["power"] is not None]
    # Per-step residency: average the parenthetical across the block's samples, then normalise,
    # because a sample with no parenthetical must not act as a zero vote for every step.
    steps = {}
    n_with = 0
    for r in cut:
        if r["steps"]:
            n_with += 1
            for k, v in r["steps"].items():
                steps[k] = steps.get(k, 0.0) + v
    if n_with:
        tot = sum(steps.values())
        if tot > 0:
            steps = {k: v / tot * 100.0 for k, v in steps.items()}
    eff = mean([sum(k * v for k, v in steps.items()) / 100.0]) if steps else mean(freqs)
    # The step the GPU actually sat on, weighted by residency. `freq_top` counts exact
    # instantaneous values and is therefore near-useless once the clock is hunting (a 928 MHz
    # plateau prints as 956/942/929/919/... so no single value repeats); the residency-weighted
    # mode is the number that corresponds to "the clock it settled on".
    step_top, step_top_pct = (max(steps.items(), key=lambda kv: kv[1]) if steps else (0.0, 0.0))
    pres = {}
    for r in cut:
        if r["pres"]:
            pres[r["pres"]] = pres.get(r["pres"], 0) + 1
    pres_dom = max(pres.items(), key=lambda kv: kv[1])[0] if pres else None
    return {
        "n": len(cut),
        "freq_mean": mean(freqs), "freq_min": min(freqs) if freqs else 0.0,
        "freq_max": max(freqs) if freqs else 0.0,
        "freq_top": max(freqs, key=lambda f: freqs.count(f)) if freqs else 0.0,
        "act_mean": mean(acts), "idle_mean": mean([r["idle"] for r in cut
                                                   if r["idle"] is not None]),
        "power_mean": mean(pwrs), "power_max": max(pwrs) if pwrs else 0.0,
        "steps": {k: round(v, 3) for k, v in sorted(steps.items())},
        "eff_freq": round(eff, 1),
        "step_top": step_top, "step_top_pct": round(step_top_pct, 2),
        "steps_n": n_with,
        "pres": pres, "pres_dom": pres_dom,
    }


def pres_rank(rec):
    """Highest thermal severity seen in a block summary (unknown words rank 0)."""
    return max((PRES_RANK.get(k, 0) for k in (rec.get("pres") or {})), default=-1)


def pres_str(rec):
    p = rec.get("pres") or {}
    if not p:
        return "not captured"
    return " ".join(f"{k}x{v}" for k, v in sorted(p.items(), key=lambda kv: -kv[1]))


def verdict(cold, hot):
    """Return (label, reasoning lines). Thresholds are printed, never just applied."""
    if not cold or not hot:
        return "INCONCLUSIVE", ["need at least one cold and one hot block"]
    df = (hot["freq_mean"] - cold["freq_mean"]) / cold["freq_mean"] * 100.0 if cold["freq_mean"] else 0.0
    dr = hot["act_mean"] - cold["act_mean"]
    dp = (hot["power_mean"] - cold["power_mean"]) / cold["power_mean"] * 100.0 if cold["power_mean"] else 0.0
    why = [
        f"clock       : {cold['freq_mean']:.0f} -> {hot['freq_mean']:.0f} MHz  ({df:+.1f}%)",
        f"residency   : {cold['act_mean']:.2f} -> {hot['act_mean']:.2f} %  ({dr:+.2f} pp)",
        f"power       : {cold['power_mean']:.0f} -> {hot['power_mean']:.0f} mW ({dp:+.1f}%)",
    ]
    # Thermal is reported whenever the capture carried it, and the escalation flag is what turns
    # "the clock is capped" into "the clock is capped by heat" -- the two have different fixes.
    cr, hr = pres_rank(cold), pres_rank(hot)
    thermal_known = cr >= 0 or hr >= 0
    thermal_esc = hr > cr
    if thermal_known:
        why.append(f"thermal     : {pres_str(cold)} -> {pres_str(hot)}"
                   + ("   <- PRESSURE ESCALATED across the pair" if thermal_esc else
                      "   (no escalation)"))
    if dr <= -10.0:
        return ("GPU KEPT OFF THE WORK", why + [
            "active residency fell by >=10 pp: the GPU is idle for part of the window even though",
            "the same work is being submitted. Look at submission/scheduling, not at clocks."])
    if abs(df) >= 3.0 and hot["freq_mean"] < cold["freq_mean"]:
        extra = ["the clock falls while residency holds -> the SAME work is executing SLOWER"]
        if thermal_esc:
            extra.append("and thermal pressure escalated across the same window, so the ceiling that"
                         " holds the clock")
            extra.append("down is a THERMAL one -- the operating point is being chosen by the"
                         " thermal governor,")
            extra.append("not by the workload and not by a fixed power budget.")
        if dp >= -8.0:
            extra.append("power holds while the clock falls -> a fixed ceiling is holding the clock down"
                         if not thermal_esc else
                         "power holds while the clock falls -> consistent with a capped operating point")
        else:
            extra.append(f"and power falls {dp:+.1f}% with it -> the clock is being held down and the")
            extra.append("lower power is the consequence, not the constraint")
        label = ("CLOCK (same work, lower clock)" if not thermal_esc else
                 "CLOCK (same work, lower clock -- thermally driven)")
        return (label, why + extra)
    if abs(df) < 3.0 and dp >= -3.0:
        return ("POWER CEILING", why + [
            "clock and power are both flat across cold/hot -> the ceiling is not being reached in a",
            "way that moves the clock. If throughput still differs, this capture does NOT explain it."])
    if abs(df) < 3.0 and dp < -3.0:
        return ("INCONCLUSIVE", why + [
            "clock is flat but power moved -> the difference is elsewhere (work per sample, or the",
            "block boundaries do not correspond to the runs)."])
    return ("INCONCLUSIVE", why)


def first_lines(path, n=6):
    """The non-blank lines as they appear. An empty capture's own content IS the diagnosis."""
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                out.append(line.rstrip())
                if len(out) >= n:
                    break
    return out


def report(paths, as_json=None):
    if isinstance(paths, str):
        paths = [paths] if paths else []
    if not paths:
        print("no capture given", file=sys.stderr)
        return 3

    parsed = []
    for p in paths:
        recs, elapsed_ms = parse(p)
        parsed.append({"path": p, "recs": recs, "elapsed_ms": elapsed_ms,
                       "freq_n": sum(1 for r in recs if r["freq"] is not None)})
    usable = [x for x in parsed if x["freq_n"] > 0]

    if len(paths) == 1:
        print(f"=== powermetrics summary: {paths[0]} ===")
    else:
        # A glob is the natural way to call this once several captures exist, so every file gets a
        # verdict line, including the ones being skipped: a silently-dropped argument is how a
        # failed capture gets mistaken for a successful one.
        print(f"=== powermetrics summary: {len(paths)} captures, {len(usable)} with data ===")
        print(f"  {'capture':<58} {'samples':>8}  note")
        for x in parsed:
            note = "ok" if x["freq_n"] else "SKIPPED -- no GPU samples"
            print(f"  {x['path']:<58} {len(x['recs']):>8}  {note}")

    recs = [r for x in usable for r in x["recs"]]
    elapsed_ms = usable[0]["elapsed_ms"] if usable else None
    freq_n = sum(1 for r in recs if r["freq"] is not None)
    print(f"samples                : {len(recs)}  (with a frequency reading: {freq_n})")
    if elapsed_ms:
        print(f"sample interval        : {elapsed_ms:.0f} ms  "
              f"(capture spans ~{len(recs) * elapsed_ms / 60000.0:.1f} min)")
    if freq_n == 0:
        print()
        print("NOTHING PARSED -- no sample carried a `GPU HW active frequency` label.")
        for x in parsed:
            print(f"  {x['path']}:")
            gpu, shown = [], 0
            with open(x["path"], "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if RE_GPULINE.search(line) and line.strip():
                        gpu.append(line.rstrip())
                        shown += 1
                        if shown >= 25:
                            gpu.append("      ... (truncated)")
                            break
            if gpu:
                print("    GPU-looking lines present:")
                for l in gpu:
                    print("      " + l)
            else:
                print("    GPU-looking lines present: none")
            # THE POINT. A capture whose powermetrics never started leaves an error line, not a
            # short capture, and "0 samples" cannot tell those apart. 2026-09-16: a 40-byte file
            # holding exactly `(eval):1: operation not permitted: sudo` was reported only as
            # "looks empty or was truncated" -- true, and useless.
            head = first_lines(x["path"], 6)
            print("    first non-blank lines actually in the file:")
            if head:
                for l in head:
                    print("      " + l)
            else:
                print("      (the file is completely empty)")
        print()
        print("A file with no samples AND no GPU lines is not a capture at all -- read its error")
        print("line above. The capture needs root: `sudo -v`, then run")
        print("`scripts/check/powermetrics_gpu_freq.sh` with NO arguments to produce one.")
        return 2

    blocks = blocks_of(recs)
    p_all = [r["power"] for r in recs if r["power"] is not None]
    thr = block_threshold(p_all)
    n_active = sum(1 for x in p_all if x >= thr)
    print()
    # Printed, never just applied: if the threshold is wrong the block count below is wrong, and
    # a reader has to be able to see that without re-deriving it.
    print(f"block threshold        : {thr:.0f} mW  (0.15 x p99.5 of power)")
    print(f"duty cycle above it    : {n_active}/{len(p_all)} samples = "
          f"{100.0 * n_active / len(p_all) if p_all else 0.0:.1f}%"
          f"   (a low duty cycle is expected for an idling capture)")
    if len(blocks) < 2:
        print()
        print(f"NOTE: {len(blocks)} activity block(s) found; a cold/hot contrast needs 2.")
        print(f"      Threshold was {thr:.0f} mW. If that looks too high for this workload, the")
        print(f"      capture's maximum may be an outlier above the sustained level -- see")
        print(f"      WHY THE BLOCK THRESHOLD IS A HIGH PERCENTILE in this file's docstring.")
        print("      If RUNS>1 was used, the gap between launches was too short to separate on")
        print("      power -- the whole capture is then ONE thermal state and cannot answer")
        print("      the clock-vs-cap question.")
    summaries = []
    for i, (lo, hi) in enumerate(blocks):
        label = "cold" if i == 0 else ("hot" if i == 1 else f"hot{i}")
        s = summarise(recs, (lo, hi))
        s.update({"index": i, "label": label, "first_sample": lo, "last_sample": hi,
                  "ts_first": recs[lo].get("ts"), "ts_last": recs[hi].get("ts")})
        summaries.append(s)

    print()
    print("--- activity blocks (segmented on GPU Power, not on the clock under test) ---")
    print(f"  {'#':>2} {'label':<5} {'samples':>7} {'freq MHz':>8} {'resid %':>8} "
          f"{'power mW':>9} {'eff MHz':>8} {'plateau MHz':>15}  {'thermal':<22} {'start':<8}")
    for s in summaries:
        plateau = f"{s['step_top']:.0f} ({s['step_top_pct']:.0f}%)" if s["steps"] else "-"
        print(f"  {s['index']:>2} {s['label']:<5} {s['n']:>7} {s['freq_mean']:>8.0f} "
              f"{s['act_mean']:>8.2f} {s['power_mean']:>9.0f} {s['eff_freq']:>8.0f} "
              f"{plateau:>15}  {pres_str(s):<22} {(s['ts_first'] or '')[11:19]:<8}")
    print("  eff MHz     = residency-weighted clock while active; the fair cold-vs-hot clock number")
    print("  plateau MHz = the DVFS step the run actually settled on, and its share of active time")
    print("  thermal     = `Current pressure level` counts, from the thermal sampler")

    cold = summaries[0]
    hots = [s for s in summaries if s["label"].startswith("hot")]
    hot = None
    if hots:
        # Thermal has to be merged across the hot blocks, not averaged: the verdict asks whether
        # the hot state is at a HIGHER pressure than cold, and a mean of labels is meaningless.
        pres = {}
        for s in hots:
            for k, v in (s.get("pres") or {}).items():
                pres[k] = pres.get(k, 0) + v
        hot = {
            "freq_mean": mean([s["freq_mean"] for s in hots]),
            "act_mean": mean([s["act_mean"] for s in hots]),
            "power_mean": mean([s["power_mean"] for s in hots]),
            "eff_freq": mean([s["eff_freq"] for s in hots]),
            "pres": pres,
        }

    if any(s.get("steps") for s in summaries):
        print()
        print("--- DVFS step residency (%), per block ---")
        print("    this is the field `GPU HW active frequency` collapses away:")
        # One column per block, not a cold/hot pair: with three launches (RUNS=3) a 2-column table
        # averages the two hot blocks together and averages a 928 MHz plateau with a 618 MHz one
        # -- i.e. it hides exactly the staircase the table exists to show.
        allk = sorted(set().union(*[set(s.get("steps", {})) for s in summaries]), reverse=True)
        used = [k for k in allk
                if max(s.get("steps", {}).get(k, 0.0) for s in summaries) > 0.0]
        hdr = "".join(f"{s['label']:>9}" for s in summaries)
        print(f"  {'MHz':>6}{hdr} {'spread pp':>10}")
        for k in used:
            row = [s.get("steps", {}).get(k, 0.0) for s in summaries]
            spread = max(row) - min(row) if row else 0.0
            flag = "   <-- moved" if len(row) >= 2 and spread >= 5.0 else ""
            cells = "".join(f"{v:>9.2f}" for v in row)
            print(f"  {k:>6}{cells} {spread:>10.2f}{flag}")
        print("  effective clock while active:  " + "   ".join(
            f"{s['label']} {s['eff_freq']:.0f} MHz" for s in summaries))
        print("  (effective = residency-weighted mean over the parenthetical; it is the number to")
        print("   compare when the states occupy DIFFERENT steps, which the single `GPU HW active")
        print("   frequency` value cannot show. Steps that are 0.00 everywhere are omitted.)")

    # Thermal gets its own section rather than one more column. It is the field that turns "the
    # clock is capped" into "the clock is capped BY HEAT", and on the 2026-09-16 capture it
    # flipped (Moderate -> Heavy) in the very same sample as the clock began to fall.
    if any(s.get("pres") for s in summaries):
        print()
        print("--- thermal pressure per block (from the thermal sampler) ---")
        for s in summaries:
            print(f"  {s['label']:<5} {s['n']:>4} samples   {pres_str(s)}")
        cr = pres_rank(summaries[0])
        hr = max((pres_rank(s) for s in summaries[1:]), default=-1)
        if hr > cr:
            print("  -> pressure ESCALATED from the cold block to the hot ones. That is a thermal")
            print("     ceiling rather than a fixed power budget: a fixed cap holds power flat while")
            print("     the clock falls, whereas here power falls WITH the clock.")
        elif hr == cr:
            print("  -> pressure did not escalate across the pair; the thermal governor had the")
            print("     same headroom in both states and is not what moved the clock.")
    else:
        print()
        print("NOTE: no `Current pressure level` lines in this capture -- it was taken without the")
        print("      thermal sampler, so a clock cap cannot be attributed to heat or to a power")
        print("      budget. Re-take it with `scripts/check/powermetrics_gpu_freq.sh`, which passes")
        print("      `--samplers gpu_power,thermal`.")

    if hot:
        label, why = verdict(cold, hot)
        print()
        print("--- verdict ---")
        print(f"  {label}")
        for w in why:
            print(f"    {w}")
        print()
        print("  (the FIRST block is the cold one by construction: the harness idles before run 1)")
        print("  (segmentation is on power, so a mis-split shows up as an odd block count above)")

    if as_json:
        with open(as_json, "w", encoding="utf-8") as fh:
            json.dump({"path": paths[0] if len(paths) == 1 else None,
                       "paths": paths,
                       "skipped": [x["path"] for x in parsed if x["freq_n"] == 0],
                       "samples": len(recs), "interval_ms": elapsed_ms,
                       "threshold_mw": round(thr, 1),
                       "duty_pct": round(100.0 * n_active / len(p_all), 2) if p_all else 0.0,
                       "blocks": summaries,
                       "cold": cold, "hot": hot,
                       "verdict": verdict(cold, hot)[0] if hot else "INCONCLUSIVE"},
                      fh, ensure_ascii=False, indent=2)
        print(f"\njson -> {as_json}")
    return 0


def selftest():
    """Synthetic captures in the real text format, one per verdict, so the rules are exercised."""
    steps = [396, 528, 720, 924, 1128, 1278, 1398]

    def mk(top, idle_pct, power, leak=1.0, pres="Nominal"):
        d = {s: 0.0 for s in steps}
        d[steps[top]] = 100.0 - leak
        if top > 0:
            d[steps[top - 1]] = leak
        body = " ".join(f"{s} MHz: {v:g}%" for s, v in d.items())
        # Same record shape as the real capture: the thermal block PRECEDES the GPU block.
        return ("*** Sampled system activity (x) (500.00ms elapsed) ***\n\n"
                "**** Thermal pressure ****\n\n"
                f"Current pressure level: {pres}\n\n"
                "**** GPU usage ****\n\n"
                f"GPU HW active frequency: {steps[top]} MHz\n"
                f"GPU HW active residency: {100.0 - idle_pct:.2f}% ({body})\n"
                f"GPU idle residency: {idle_pct:.2f}%\n"
                f"GPU Power: {power} mW\n\n")

    def cap(cold_f, cold_p, hot_f, hot_p, hot_idle=0.3, cold_idle=0.2,
            cold_pres="Nominal", hot_pres="Nominal"):
        s = "Machine model: Mac16,10\n\n" + mk(0, 98.5, 40) * 20
        s += mk(cold_f, cold_idle, cold_p, pres=cold_pres) * 40
        s += mk(0, 97.0, 80) * 4
        s += mk(hot_f, hot_idle, hot_p, pres=hot_pres) * 40
        return s

    import tempfile, os, io, contextlib

    def parse_text(text):
        fd, p = tempfile.mkstemp(suffix=".log")
        os.close(fd)
        with open(p, "w") as fh:
            fh.write(text)
        try:
            recs, _ = parse(p)
        finally:
            os.unlink(p)
        return recs

    def verdict_of(recs):
        bl = blocks_of(recs)
        if len(bl) < 2:
            return "INCONCLUSIVE", [], bl
        c = summarise(recs, bl[0])
        hs = [summarise(recs, b) for b in bl[1:]]
        pres = {}
        for x in hs:
            for k, v in (x.get("pres") or {}).items():
                pres[k] = pres.get(k, 0) + v
        h = {"freq_mean": mean([x["freq_mean"] for x in hs]),
             "act_mean": mean([x["act_mean"] for x in hs]),
             "power_mean": mean([x["power_mean"] for x in hs]),
             "pres": pres}
        label, why = verdict(c, h)
        return label, why, bl

    cases = [
        ("clock with power ceiling",  cap(6, 40500, 5, 40000), "CLOCK", None),
        ("clock with power released", cap(6, 40500, 5, 33000), "CLOCK", None),
        ("flat clock and power",      cap(6, 40000, 6, 40000), "POWER CEILING", None),
        ("residency collapse",        cap(6, 40000, 6, 40000, hot_idle=30.0),
         "GPU KEPT OFF THE WORK", None),
        # The addition that names the mechanism: same clock drop, but pressure escalates across
        # the pair, which is what separates "a ceiling" from "a ceiling caused by heat".
        ("thermal escalation",        cap(6, 40500, 5, 33000, cold_pres="Nominal", hot_pres="Heavy"),
         "CLOCK", "PRESSURE ESCALATED"),
        ("thermal flat",              cap(6, 40500, 5, 33000, cold_pres="Moderate", hot_pres="Moderate"),
         "CLOCK", "no escalation"),
    ]
    ok = 0
    for name, text, want, must in cases:
        got, why, _ = verdict_of(parse_text(text))
        hit = got.startswith(want) and (must is None or any(must in w for w in why))
        ok += hit
        extra = "" if must is None else f"  +{must!r}"
        print(f"  {'ok  ' if hit else 'FAIL'}  {name:<26} -> {got:<26} want {want}{extra}")

    # Regression for the block threshold. A long idle tail must not change the segmentation.
    # The tail is modelled on the real thing, which matters: this machine's idle is not flat but
    # spiky (p50 51 mW, p95 217 mW, max 398 mW -- measured on the 2026-09-16 capture), and it is
    # the SPIKES that a collapsed threshold mistakes for launches. A flat tail would prove
    # nothing, because it is contiguous with the last launch and merges into it. So the tail is
    # 4-sample bursts of 300 mW separated by 6 samples of true idle: bursts are >= min_len apart,
    # gaps are > gap_merge, i.e. every burst becomes its own block if the threshold lets it.
    # The pre-2026-09-16 rule (0.15 * p95) is driven through the SAME merging code, so this shows
    # that the rule change is the fix instead of asserting that it is.
    base = cap(6, 40500, 5, 33000)
    tail = (mk(0, 98.5, 300) * 4 + mk(0, 99.0, 50) * 6) * 400
    long_text = base + tail
    r_short, r_long = parse_text(base), parse_text(long_text)
    p_long = [r["power"] for r in r_long if r["power"] is not None]
    n_short = len(blocks_of(r_short))
    n_new = len(blocks_of(r_long))
    n_old = len(blocks_of(r_long, thr=max(200.0, 0.15 * pct(p_long, 0.95))))
    hit = n_new == n_short and n_old != n_short
    ok += hit
    print(f"  {'ok  ' if hit else 'FAIL'}  {'idle tail keeps 2 blocks':<26} -> "
          f"new={n_new} (want {n_short}), old rule={n_old} (wants to differ)")

    # A capture that failed is still a file, and a glob will hand it to this parser alongside real
    # captures. The failure must be reported per file and must NOT suppress the good file.
    fd, bad = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    with open(bad, "w") as fh:
        fh.write("(eval):1: operation not permitted: sudo\n")
    fd, good = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    with open(good, "w") as fh:
        fh.write(cap(6, 40500, 5, 33000))
    multi = [
        ("a dead capture first, a good one after", [bad, good], 0, True),
        ("only dead captures",                     [bad],        2, False),
    ]
    for name, paths, want_rc, want_body in multi:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = report(paths)
        out = buf.getvalue()
        hit = rc == want_rc and ("CLOCK" in out) == want_body
        if hit and not want_body:
            # the dead file's own content must be shown, not just "looks empty"
            hit = "operation not permitted: sudo" in out
        ok += hit
        print(f"  {'ok  ' if hit else 'FAIL'}  {name:<26} -> rc={rc:<3} "
              f"{'parsed' if want_body else 'loud-fail w/ file content'}")
    total = len(cases) + 1 + len(multi)
    print(f"\n  {ok}/{total} cases match")
    return 0 if ok == total else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # "*", not "+": --selftest takes no capture, and a required positional would reject it before
    # main() ever gets to look at the flag.
    ap.add_argument("capture", nargs="*", help="powermetrics text capture(s)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.capture:
        ap.print_usage(sys.stderr)
        return 3
    return report(a.capture, a.json)


if __name__ == "__main__":
    sys.exit(main())
