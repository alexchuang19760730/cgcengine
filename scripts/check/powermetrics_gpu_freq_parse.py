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
RE_GPULINE = re.compile(r"GPU", re.I)


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
                       "freq": None, "act": None, "idle": None, "power": None, "steps": {}}
                continue
            if cur is None:
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


def blocks_of(recs, min_len=3, gap_merge=2):
    """Contiguous stretches where the GPU is actually working.

    Segmentation is on POWER, not on frequency or residency: those are exactly the quantities
    under test, so using them to define "a run" would beg the question. Power separates idle
    (~tens of mW) from a prefill (tens of W) by three orders of magnitude.
    """
    p = [r["power"] for r in recs if r["power"] is not None]
    if len(p) < 5:
        return []
    thr = max(200.0, 0.15 * pct(p, 0.95))
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
        "steps_n": n_with,
    }


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
    if dr <= -10.0:
        return ("GPU KEPT OFF THE WORK", why + [
            "active residency fell by >=10 pp: the GPU is idle for part of the window even though",
            "the same work is being submitted. Look at submission/scheduling, not at clocks."])
    if abs(df) >= 3.0 and hot["freq_mean"] < cold["freq_mean"]:
        extra = ["the clock falls while residency holds -> the SAME work is executing SLOWER"]
        if dp >= -8.0:
            extra.append("and power holds too -> a fixed power/thermal ceiling is holding the clock down")
        else:
            extra.append(f"and power falls {dp:+.1f}% with it -> the clock is being held down and the")
            extra.append("lower power is the consequence, not the constraint")
        return ("CLOCK (same work, lower clock)", why + extra)
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
    print()
    if len(blocks) < 2:
        print(f"NOTE: {len(blocks)} activity block(s) found; a cold/hot contrast needs 2.")
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

    print("--- activity blocks (segmented on GPU Power, not on the clock under test) ---")
    print(f"  {'#':>2} {'label':<5} {'samples':>7} {'freq MHz':>8} {'resid %':>8} "
          f"{'power mW':>9}  {'first timestamp':<32}")
    for s in summaries:
        print(f"  {s['index']:>2} {s['label']:<5} {s['n']:>7} {s['freq_mean']:>8.0f} "
              f"{s['act_mean']:>8.2f} {s['power_mean']:>9.0f}  {(s['ts_first'] or '')[:32]:<32}")

    cold = summaries[0]
    hots = [s for s in summaries if s["label"].startswith("hot")]
    hot = None
    if hots:
        hot = {
            "freq_mean": mean([s["freq_mean"] for s in hots]),
            "act_mean": mean([s["act_mean"] for s in hots]),
            "power_mean": mean([s["power_mean"] for s in hots]),
            "eff_freq": mean([s["eff_freq"] for s in hots]),
        }

    if cold.get("steps") or (hot and hot.get("steps")):
        print()
        print("--- DVFS step residency (%), cold vs hot ---")
        print("    this is the field `GPU HW active frequency` collapses away:")
        allk = sorted(set(cold.get("steps", {})) | set((hot or {}).get("steps", {})), reverse=True)
        print(f"  {'MHz':>6} {'cold %':>9} {'hot %':>9} {'delta pp':>10}")
        for k in allk:
            c = cold.get("steps", {}).get(k, 0.0)
            h = (hot or {}).get("steps", {}).get(k, 0.0)
            flag = "   <-- moved" if abs(h - c) >= 5.0 else ""
            print(f"  {k:>6} {c:>9.2f} {h:>9.2f} {h - c:>+10.2f}{flag}")
        if hot:
            print(f"  effective clock while active: cold {cold['eff_freq']:.0f} MHz"
                  f"  hot {hot['eff_freq']:.0f} MHz")
            print("  (effective = residency-weighted mean over the parenthetical; it is the number")
            print("   to compare when the two states occupy DIFFERENT steps, which the single")
            print("   `GPU HW active frequency` value cannot show)")

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
                       "blocks": summaries,
                       "cold": cold, "hot": hot,
                       "verdict": verdict(cold, hot)[0] if hot else "INCONCLUSIVE"},
                      fh, ensure_ascii=False, indent=2)
        print(f"\njson -> {as_json}")
    return 0


def selftest():
    """Synthetic captures in the real text format, one per verdict, so the rules are exercised."""
    steps = [396, 528, 720, 924, 1128, 1278, 1398]

    def mk(top, idle_pct, power, leak=1.0):
        d = {s: 0.0 for s in steps}
        d[steps[top]] = 100.0 - leak
        if top > 0:
            d[steps[top - 1]] = leak
        body = " ".join(f"{s} MHz: {v:g}%" for s, v in d.items())
        return ("*** Sampled system activity (x) (500.00ms elapsed) ***\n\n**** GPU usage ****\n\n"
                f"GPU HW active frequency: {steps[top]} MHz\n"
                f"GPU HW active residency: {100.0 - idle_pct:.2f}% ({body})\n"
                f"GPU idle residency: {idle_pct:.2f}%\n"
                f"GPU Power: {power} mW\n\n")

    def cap(cold_f, cold_p, hot_f, hot_p, hot_idle=0.3, cold_idle=0.2):
        s = "Machine model: Mac16,10\n\n" + mk(0, 98.5, 40) * 20
        s += mk(cold_f, cold_idle, cold_p) * 40
        s += mk(0, 97.0, 80) * 4
        s += mk(hot_f, hot_idle, hot_p) * 40
        return s

    import tempfile, os, io, contextlib
    cases = [
        ("clock with power ceiling", cap(6, 40500, 5, 40000), "CLOCK"),
        ("clock with power released", cap(6, 40500, 5, 33000), "CLOCK"),
        ("flat clock and power",      cap(6, 40000, 6, 40000), "POWER CEILING"),
        ("residency collapse",        cap(6, 40000, 6, 40000, hot_idle=30.0), "GPU KEPT OFF THE WORK"),
    ]
    ok = 0
    for name, text, want in cases:
        fd, p = tempfile.mkstemp(suffix=".log")
        os.close(fd)
        with open(p, "w") as fh:
            fh.write(text)
        recs, _ = parse(p)
        bl = blocks_of(recs)
        got = "INCONCLUSIVE"
        if len(bl) >= 2:
            c = summarise(recs, bl[0])
            hs = [summarise(recs, b) for b in bl[1:]]
            h = {"freq_mean": mean([x["freq_mean"] for x in hs]),
                 "act_mean": mean([x["act_mean"] for x in hs]),
                 "power_mean": mean([x["power_mean"] for x in hs])}
            got = verdict(c, h)[0]
        os.unlink(p)
        hit = got.startswith(want)
        ok += hit
        print(f"  {'ok  ' if hit else 'FAIL'}  {name:<26} -> {got:<26} want {want}")

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
        n = len(cases) + multi.index((name, paths, want_rc, want_body))
        print(f"  {'ok  ' if hit else 'FAIL'}  {name:<26} -> rc={rc:<3} "
              f"{'parsed' if want_body else 'loud-fail w/ file content'}")
    total = len(cases) + len(multi)
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
