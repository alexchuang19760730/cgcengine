#!/usr/bin/env python3
"""Sample the OS memory pressure (swap / free / wired pages) in band, next to a measurement,
and attribute a slow run to thermal vs swap.

WHY THIS EXISTS. A slow llama-bench arm is ambiguous: the same arm produced 11.90 t/s at
NOMINAL thermal with a cold launch, then 9.91 t/s under HEAVY thermal -- and separately a run
whose swap grew by GBs measured slower with no thermal story at all. Without a reading taken
at the same moment, "this arm is slower" cannot be attributed, so every A/B lives under a
permanent alibi: *thermal drift, or swap pressure, or a real regression* -- three causes with
one observable.

THE TWO INSTRUMENTS. `thermal_pressure.py` already samples the OS thermal level in the same
shape (background thread, launch reading before spawn, launch/worst/hist result). This module
samples the memory side with the same contract, so a harness can hold BOTH samplers over one
arm and get an attribution instead of a guess.

WHAT IT READS, AND THE TRAP IT IS BUILT AROUND:
  * `sysctl -n vm.swapusage`  -> `total = X.XXM used = Y.YYM free = Z.ZZM` -- macOS reports
    swap *used*, which accumulates and does not shrink on its own. A run that GROWS swap
    (end - start > threshold) is pushing anonymous pages out; that is the memory-pressure
    signature that matters, not the absolute level (a machine can sit at 3.2 GB used forever
    without any run being at fault).
  * `vm_stat` -> pages free / inactive / wired down. `Pages free` on macOS is chronically
    small (the system counts most free memory as inactive/compressible), so FREE ALONE IS NOT
    A PRESSURE SIGNAL -- it is wired growth + swap growth together that are. The module keeps
    both numbers but the attribution rule leans on swap growth and wired growth, not free.
  * `ps` count of llama processes -- how many engines are alive during the run (a neighbour
    server is a different attribution: "someone else holds the GPU/memory").

THE ATTRIBUTION RULE (documented, not magic):
    thermal worst >= HEAVY                       -> "thermal"
    swap growth > 512 MiB  OR  (swap used > 6144 MiB AND growth > 0)
                                                -> "swap"
    wired growth > 1024 MiB AND swap growth > 0 -> "swap"   (the L4 pattern: wired crowds out anonymous)
    llama_procs > 1                              -> "contention"  (a neighbour engine is the alibi)
    both of the first two                        -> "both"
    none of the above                            -> "none"   (environment other: CPU/IO, not measured here)

DELIBERATELY NOT A GATE. Like thermal_pressure.py, this records and attributes; it does not
refuse to run.

CALIBRATED 2026-09-28 (`curve_gate`). The first-pass numbers above were never checked against
history, and `.workbuddy/memory/swap_log.tsv` cannot do it: it holds 5 rows of point-in-time stock
with no arm boundaries. The calibration set is the repo's own arm records (~2.3k JSON files
scanned, 127 arms carry a run-internal curve), grouped by cell, decode t/s vs peak run-internal
swap growth:

    cell prod-new (n=111)   growth 0-512 MiB   n=23   tg median 11.96 t/s
                            512-2048          n=45   tg median 11.54
                            2048-8192         n=41   tg median 10.64
                            >8192             n= 2   tg median  8.02
    cell delivery (n=15)    0-512             n= 5   tg median 11.38
                            512-2048          n= 5   tg median 10.44
                            2048-8192         n= 5   tg median  9.14

So `SWAP_GROWTH_MB = 512` sits at the KNEE of a monotone curve -- and it was used as a GATE for
one day. CALIBRATED AGAIN 2026-09-28 (`state_gate`), on a bigger slice of the same corpus (129
arms carry a curve AND a decode sample; 122 of them are in the one cell with n), because a
measurement on the cleanest box of that night was refused by it while the arm was valid in every
other respect (draft chain alive, 62/62 samples NOMINAL, single-rep launch):

  * growth is almost a function of the LAUNCH state: rho(growth, launch `pages_free`) = **-0.870**
    (free 66-1993 MiB -> growth median 3372; free 7323-10859 -> growth 342). A tight box has to
    evict; a roomy one does not. With a 13.6 GB model on a 16 GB machine that eviction is
    structural -- it is part of the arm's own load, not evidence of a neighbour's interference.
  * the SAME arm on the SAME box produced growth -32 / +11 / +202 / +312 / +983 / +1971 MiB in
    one day. A negative "growth" is decisive: the number is not a property of the arm.
  * the in-arm signal is real but only PARTLY about the arm: after a two-segment expectation
    `growth ~= max(226, 3914 - 0.4404*launch_free)` (breakpoint fitted on the 122-arm cell), the
    residual still ranks decode t/s (rho = -0.327). Residual <= 0 -> ~11.7 t/s; 250-1000 -> 10.60;
    > 1000 -> 10.12. A single straight line does NOT work: it predicts ~112 MiB at free 9191 where
    the corpus's roomiest quartile actually sits at 342, and it refused a launch whose growth was
    LOWER than the previous launch's purely because its box was roomier (measured 2026-09-28).
  * launch free ALONE is weak (rho(t/s, free) = +0.275) and in-arm `min_free` is useless: p10..p90
    is 14..50 MiB in every bucket, including the cleanest arms, because the box runs there as soon
    as a model plus expert cache is resident. That is why `FREE_FLOOR_MB` (the IN-ARM floor) stays
    None -- and why the free-side gate below is on the LAUNCH reading instead.
  * the wired bar was worse than uncalibrated, it pointed the wrong way: rho(t/s, wired growth) =
    **+0.230** and the >1024 MiB group is FASTER (median 11.70 vs 10.80).

THE GATE THAT CAME OUT OF THAT (two halves, both calibrated, both recorded in the artifact):

    state_gate   PRE-LAUNCH, enforced: launch `pages_free_mb` >= `STATE_FREE_FLOOR_MB` (5000 MiB).
                 Refuses BEFORE the arm is spawned -- a tight box costs a cleanup, not 44 s.
                 Sweep on the 122-arm cell: floor 5000 keeps 65/122 (median 11.84 t/s, p25 10.91);
                 the refused group is median 11.03, p25 9.82, p10 7.86.
                 Override: CGC_STATE_FREE_FLOOR_MB.
    curve_gate   POST-HOC, enforced: residual `growth - max(226, 3914 - 0.4404*free)` <= 500 MiB.
                 500 and not 250 because 250 is smaller than the box's own motion (free swings
                 660 MiB in 10 idle seconds) and the residual differences two single readings;
                 at 500 the kept median is 11.70 t/s and the refused median 10.51.
                 The ABSOLUTE `SWAP_GROWTH_MB` bar is now OPT-IN (`CGC_SWAP_BUDGET_MB=<N>` turns it
                 back on). As a default it demanded launch free >~ 8.5 GB -- on this box that means
                 an almost empty machine -- and because growth is 87% launch state, it was mostly
                 re-encoding "was the box roomy" while also charging the arm for its OWN footprint
                 (expert-cache size, batch), which differs between configs on an identical box.
                 The wired bar is demoted to evidence (see above).

WHAT IS NOT CLAIMED: none of these numbers is a strong instrument (|rho| 0.27-0.41, one cell,
mixed configs inside it). The gate is a readiness check with teeth, not a t/s predictor -- and the
honest reason to prefer the launch half is that it is checkable BEFORE spending the measurement
and does not depend on what is being measured.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Attribution thresholds -- first pass, calibrate against swap_log.tsv before gating.
SWAP_GROWTH_MB = 512.0      # run-internal swap growth beyond this = memory pressure
SWAP_HIGH_MB = 6144.0       # absolute swap used beyond this + any growth = pressure
WIRED_GROWTH_MB = 1024.0    # wired growth beyond this + swap growth = the L4 crowding pattern

UNREADABLE = "UNREADABLE"


def _sysctl_swap() -> dict:
    """`{total_mb, used_mb, free_mb}` from `sysctl -n vm.swapusage`, or None on failure.

    Returns None (never a made-up number) when the output does not match the documented
    shape -- the same refusal family as thermal_pressure's unsupported-key handling.
    """
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                             capture_output=True, text=True, timeout=5)
    except Exception:  # noqa: BLE001 - a missing tool must not kill a measurement
        return None
    m = out.stdout.strip()
    # e.g. "total = 4096.00M  used = 3205.62M  free = 890.38M  (encrypted)"
    try:
        def _num(s):
            # s is a single token like "4096.00M" (split("=") would be for the whole line)
            return float(s.strip().rstrip("M"))

        parts = m.split()
        total = _num(parts[parts.index("total") + 2])
        used = _num(parts[parts.index("used") + 2])
        free = _num(parts[parts.index("free") + 2])
        return {"total_mb": total, "used_mb": used, "free_mb": free}
    except Exception:  # noqa: BLE001
        return None


def _vm_stat() -> dict:
    """`{free_pages, inactive_pages, wired_pages}` from `vm_stat`, or None on failure."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
    except Exception:  # noqa: BLE001
        return None
    res: dict = {}
    for line in out.stdout.splitlines():
        # "Pages free:                              12345."
        if line.startswith("Pages free:"):
            res["free_pages"] = int(line.split(":")[1].strip().rstrip("."))
        elif line.startswith("Pages speculative:"):
            res["speculative_pages"] = int(line.split(":")[1].strip().rstrip("."))
        elif line.startswith("Pages purgeable:"):
            res["purgeable_pages"] = int(line.split(":")[1].strip().rstrip("."))
        elif line.startswith("Pages inactive:"):
            res["inactive_pages"] = int(line.split(":")[1].strip().rstrip("."))
        elif line.startswith("Pages wired down:"):
            res["wired_pages"] = int(line.split(":")[1].strip().rstrip("."))
    if not res:
        return None
    res["page_kb"] = 16  # macOS 4 KiB pages -> 16 KiB per page is wrong; see below
    return res


# Correction: macOS page size is 16 KiB on Apple Silicon (and 4 KiB on Intel). We ask the
# kernel rather than hard-code it.
def _page_size_kb() -> float:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.pagesize"],
                             capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip()) / 1024.0
    except Exception:  # noqa: BLE001
        return 16.0  # Apple Silicon default; documented fallback, not a silent guess


_PAGE_KB = _page_size_kb()


def llama_procs() -> int:
    """How many ENGINE processes are alive right now (neighbours = contention).

    Matches the *program* (argv[0]'s basename), not any line that contains the word. The previous
    rule -- `"llama" in line` over `ps aux` -- counted a shell whose command text merely names the
    tool, an editor with such a file open, a `pgrep -fl llama...` (and `ps aux` truncates its
    command column, so whether a mention was even visible depended on how long the asking command
    was). That mattered because `attribution()` turns `max_procs > 1` into a `contention` verdict:
    a text match could therefore alibi a bad reading with a neighbour engine that does not exist.
    Measured 2026-09-26: the counter read 1 on a box with zero engine processes, and, with two
    real processes running, 0.
    """
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True,
                             timeout=5)
        n = 0
        for line in out.stdout.splitlines():
            pid, _, cmd = line.strip().partition(" ")
            if not pid.isdigit() or not cmd:
                continue
            try:                       # a program is the first token; its basename is the name
                prog = os.path.basename(cmd.split()[0])
            except IndexError:
                continue
            if "llama" in prog.lower():
                n += 1
        return n
    except Exception:  # noqa: BLE001
        return -1


def swap_used_mb() -> float | None:
    s = _sysctl_swap()
    return s["used_mb"] if s else None


# The measurement contract's start line: above this the run starts saturated and the decay is
# already in the artifact before the first token. Same number as §3.3.
SWAP_START_LIMIT_MB = 2048.0


def top_rss(n: int = 5) -> list[dict]:
    """The `n` biggest RSS processes right now -- what the user can actually close.

    Exists because "swap is at 5 GB" is not actionable and "your swap is not the engine's" is
    not either: the residency is held by *named* other applications, and this is the list that
    names them. Returns `[]` (never a fabricated row) when `ps` gives nothing.
    """
    try:
        out = subprocess.run(["ps", "-Ao", "rss=,comm="], capture_output=True, text=True,
                             timeout=10)
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            rss_kb = int(parts[0])
        except ValueError:
            continue
        rows.append({"rss_mb": round(rss_kb / 1024.0, 1), "proc": parts[1].strip()})
    rows.sort(key=lambda r: -r["rss_mb"])
    return rows[:n]


def swap_advice(limit_mb: float = SWAP_START_LIMIT_MB, wait_mb: float = 500.0) -> dict:
    """LABEL on the box's swap stock BEFORE a launch, with the named consumers.

    2026-09-26: this judges a STOCK, and the launch gate is a FLOW (`compressor_pressure`). The two
    do not agree on purpose: an idle box carrying 7993 MiB of stock reads 0.00 MiB/s of compressor
    flow (clean), while a busy one reads 245-540 MiB/s. So the verdict here is advisory wording about
    pressure budget -- it must not be read as "this box cannot be measured", which is why callers
    render it as a label and `harness.py` gates on the compressor instead.

    `ok` (≤ wait); `warn` (over `wait`, within `limit`, engine not running); `advise` (over `limit`
    with nothing of ours holding those pages) -- the message names the processes to close. Fail-closed
    is deliberately NOT used here: the launcher is not the owner of the user's desktop, it is the
    messenger.
    """
    used = swap_used_mb()
    procs = llama_procs()
    if used is None:
        return {"verdict": "unknown", "swap_used_mb": None, "llama_procs": procs,
                "limit_mb": limit_mb, "top": [],
                "why": "sysctl vm.swapusage 讀不到（不當成 0）"}
    if used <= wait_mb:
        return {"verdict": "ok", "swap_used_mb": used, "llama_procs": procs,
                "limit_mb": limit_mb, "top": [], "why": f"swap {used:.0f} MiB ≈ 乾淨"}
    if used <= limit_mb and procs <= 0:
        return {"verdict": "warn", "swap_used_mb": used, "llama_procs": procs,
                "limit_mb": limit_mb, "top": top_rss(5),
                "why": f"swap 存量 {used:.0f} MiB 已高於乾淨線 {wait_mb:.0f}（標籤；閘門是壓縮機流量）"}
    return {"verdict": "advise", "swap_used_mb": used, "llama_procs": procs,
            "limit_mb": limit_mb, "top": top_rss(5),
            "why": (f"swap 存量 {used:.0f} MiB > 舊起跑線 {limit_mb:.0f} MiB（**標籤，不是閘**；"
                    f"起跑閘是壓縮機安靜度，見 compressor_pressure.py），機器上只有 {procs} 個 "
                    f"llama 行程 ⇒ 這些頁不是引擎的（P0/P1/P2 管不到別人的駐留）。存量大本身不降速"
                    f"（2026-09-26 實測：存量 7862 MiB、流量 0.00 MiB/s 的盒子量到 tg 11.46）；"
                    f"它只是壓縮機的壓力預算小。關掉下表幾項可讓壓縮機少被觸發")}


def stamp() -> dict:
    """One JSON-safe reading: `{t, swap_used_mb, swap_total_mb, pages_free_mb,
    pages_wired_mb, llama_procs}`. Missing fields are None (never folded into a number)."""
    s = _sysctl_swap()
    v = _vm_stat()
    def _mb(key):
        return (v[key] * _PAGE_KB / 1024.0) if (v and v.get(key) is not None) else None

    free = _mb("free_pages")
    parts = [_mb(k) for k in ("free_pages", "inactive_pages", "speculative_pages", "purgeable_pages")]
    avail = sum(p for p in parts if p is not None) if any(p is not None for p in parts) else None
    return {
        "t": time.strftime("%H:%M:%S"),
        "swap_used_mb": s["used_mb"] if s else None,
        "swap_total_mb": s["total_mb"] if s else None,
        "pages_free_mb": free,
        "pages_wired_mb": _mb("wired_pages"),
        # `pages_free` alone INVERTS on a freshly booted box: measured 2026-09-28 after a reboot,
        # free was 57-72 MiB while the machine was at its cleanest ever (swap 0/0, 5.8-6.0 GiB
        # reclaimable) -- macOS fills the free list with cache, so the strict free list is a
        # cache statistic, not an availability one. `available` (free+inactive+speculative+
        # purgeable) is what the OS can hand over without swapping out someone else's anonymous
        # pages, and it is what the launch gate judges.
        "pages_inactive_mb": _mb("inactive_pages"),
        "pages_speculative_mb": _mb("speculative_pages"),
        "pages_purgeable_mb": _mb("purgeable_pages"),
        "pages_available_mb": avail,
        "llama_procs": llama_procs(),
    }


def worst(stamps) -> dict:
    """The memory-side worst over the series: max swap used, min free pages, max wired.

    Unreadable readings are not silently ignored: if every reading failed, the result is
    None-valued -- the same "we did not measure" honesty as thermal_pressure.
    """
    swaps = [s.get("swap_used_mb") for s in stamps if isinstance(s, dict)
             and s.get("swap_used_mb") is not None]
    frees = [s.get("pages_free_mb") for s in stamps if isinstance(s, dict)
             and s.get("pages_free_mb") is not None]
    wired = [s.get("pages_wired_mb") for s in stamps if isinstance(s, dict)
             and s.get("pages_wired_mb") is not None]
    return {
        "max_swap_mb": max(swaps) if swaps else None,
        "min_free_mb": min(frees) if frees else None,
        "max_wired_mb": max(wired) if wired else None,
    }


def attribution(thermal_result: dict, mem_result: dict, oversubscribed: bool = False) -> dict:
    """Classify the run: oversubscribed / thermal / swap / contention / both / none.

    `thermal_result` is a thermal_pressure.Sampler.result dict; `mem_result` is this
    module's Sampler.result. The rule is documented in the module docstring; every branch
    names its evidence so a reader can dispute the call instead of the number.

    `oversubscribed` is budget-gate's STATIC verdict (static demand > physical RAM). It is passed
    IN, not re-derived here: budget_gate owns that arithmetic, and a second copy of the same
    subtraction would drift from it.

    WHY IT OVERRIDES `none` (2026-10-07). Every other branch reasons about CHANGE -- swap growth,
    wired growth, thermal level. A machine that is ALREADY being held up by swap therefore reports
    growth = 0 and lands on `none`; "nothing got worse" gets read as "nothing was wrong". Measured
    twice in one afternoon on this box: the 11:15 arm reported `swap_growth=+3678 MiB` (verdict
    swap) and the 12:05 arm reported `swap_growth=0.0 MiB` (verdict none) -- minutes apart, with
    260-300 MiB of swap still HELD by the earlier run. Under a static oversubscription of 4838 MiB
    the launch can only complete through compression + swap, so it is not a clean baseline no
    matter what the post-hoc arithmetic says.

    It can only ever REMOVE a `none`, never create one -- the same direction the `_box_gate` swap
    STOCK demotion took (see the note below this function), and for the same reason: stock cannot
    separate "old swap + quiet compressor" from "small swap + busy compressor", but a static
    oversubscription is unambiguous -- there is no reading of the deltas that undoes it.
    """
    th_worst = ((thermal_result or {}).get("worst") or {}).get("label", UNREADABLE)
    launch = ((mem_result or {}).get("launch") or {}).get("swap_used_mb")
    end = ((mem_result or {}).get("end") or {}).get("swap_used_mb")
    wr_worst = (worst((mem_result or {}).get("samples", [])) or {})
    max_swap = wr_worst.get("max_swap_mb")
    launch_wired = ((mem_result or {}).get("launch") or {}).get("pages_wired_mb")
    end_wired = ((mem_result or {}).get("end") or {}).get("pages_wired_mb")
    procs = [s.get("llama_procs") for s in (mem_result or {}).get("samples", [])
             if isinstance(s, dict) and s.get("llama_procs") not in (None, -1)]
    max_procs = max(procs) if procs else 0

    swap_growth = (end - launch) if (launch is not None and end is not None) else None
    wired_growth = (end_wired - launch_wired) if (launch_wired is not None and end_wired is not None) else None

    hot = th_worst in ("HEAVY", "TRAPPING", "SLEEPING")
    swap_pressure = (swap_growth is not None and swap_growth > SWAP_GROWTH_MB) or \
                    (max_swap is not None and max_swap > SWAP_HIGH_MB and (swap_growth or 0) > 0)
    wired_crowd = (wired_growth is not None and wired_growth > WIRED_GROWTH_MB and
                   (swap_growth or 0) > 0)
    contested = max_procs > 1

    # Ordered so the RICHER diagnoses win: hot AND oversubscribed is `both`, because it carries
    # strictly more information than `oversubscribed` alone. `oversubscribed` is reserved for the
    # case where the static budget is the ONLY thing disqualifying the run.
    if hot and (swap_pressure or wired_crowd):
        verdict, why = "both", f"thermal={th_worst}, swap_growth={swap_growth} MiB"
    elif hot:
        verdict, why = "thermal", f"thermal worst {th_worst}"
    elif swap_pressure or wired_crowd:
        verdict, why = "swap", f"swap_growth={swap_growth} MiB, max_swap={max_swap} MiB"
    elif contested:
        verdict, why = "contention", f"llama_procs={max_procs}"
    elif oversubscribed:
        verdict, why = "oversubscribed", (
            "static demand exceeds physical RAM (budget-gate OVERSUBSCRIBED): this launch can only "
            f"complete through compression + swap, so it is not a clean baseline; "
            f"swap_growth={swap_growth} MiB, max_swap={max_swap} MiB, thermal={th_worst}")
    else:
        verdict, why = "none", f"swap_growth={swap_growth} MiB, thermal={th_worst}"

    return {"verdict": verdict, "why": why, "thermal_worst": th_worst,
            "oversubscribed": bool(oversubscribed),
            "swap_growth_mb": swap_growth, "max_swap_mb": max_swap,
            "wired_growth_mb": wired_growth, "max_llama_procs": max_procs}


# ── run-internal budget: the GATE (as opposed to the launch-stock LABEL) ──────────────────────
#
# WHY THE STOCK IS ONLY A LABEL AND THE GROWTH IS A GATE: `_box_gate` demoted the swap STOCK to a
# label (2026-09-26) because stock cannot separate "8 GB of old swap + a quiet compressor" from
# "small swap + a busy compressor", and only the latter eats t/s. The measurement that demotion
# left open is the other half of the same question -- what THIS arm adds while it runs. That is
# unambiguous here, because the launch reading is taken before the child exists (`Sampler.start`)
# and the end reading after it is gone, and it is now calibrated against history (see the module
# docstring): growth above 512 MiB tracks a ~10-20% lower decode t/s, monotone across both cells.
# The IN-ARM free floor stays off: p10..p90 of min_free is 14..50 MiB in every growth bucket,
# cleanest arms included, so it cannot separate anything (see the docstring).
FREE_FLOOR_MB = None

# --- the LAUNCH-state floor (the gate that replaced the absolute growth bar) -----------------
# ENFORCED on `pages_available_mb` (free + inactive + speculative + purgeable): the amount the OS
# can hand over without swapping out someone else's anonymous pages.
#
# WHY NOT `pages_free`, WHICH IS WHAT THE CORPUS WAS CALIBRATED ON: measured 2026-09-28, right
# after a reboot, `pages_free` was 57-72 MiB on the CLEANEST state this box ever has (swap 0/0,
# 5.8-6.0 GiB available) -- macOS fills the free list with cache, so the strict free list is a
# cache statistic. A 5000 MiB floor on it refused a freshly booted box. So the corpus-calibrated
# `pages_free` band stays RECORDED AND PRINTED as a warning, and the enforced floor is on a
# quantity whose failure mode is the one we care about.
#
# The 4000 MiB figure is a PHYSICAL floor, not a t/s calibration and is recorded as such: below it
# the model's own 13.6 GB demand starts evicting other processes' anonymous pages. Override:
# CGC_STATE_AVAILABLE_FLOOR_MB.
STATE_AVAILABLE_FLOOR_MB = 4000.0
STATE_FREE_FLOOR_MB = 5000.0   # corpus band (n=122) -- WARNING ONLY, see above

# --- the residual form of the in-arm bar -----------------------------------------------------
# growth is 87% determined by the launch reading, so the absolute value ranks boxes, not arms.
# What is left after removing the fit IS about the arm -- but the fit has to be the right SHAPE.
#
# A single straight line does NOT fit: it predicts ~112 MiB at launch free 9191, while the corpus's
# roomiest quartile has a median growth of 342 -- because the arm's own cold resident set (model
# pages, expert cache, Metal buffers) puts a FLOOR under the eviction it causes no matter how roomy
# the box is. Measured 2026-09-28: a linear-fit residual refused one of two back-to-back launches
# whose absolute growth (550 MiB) was LOWER than the other's (670 MiB) purely because its box was
# roomier -- i.e. the linear version re-created the old defect at the other end of the range.
#
# So the expectation is two-segment (hockey stick), breakpoint fitted by grid search over SSE on the
# 122-arm cell: `expected = max(FLOOR, A + B*free)`.
GROWTH_FIT_A = 3914.0        # free < 8100 -> growth ≈ A + B*free
GROWTH_FIT_B = -0.4404
GROWTH_FIT_FLOOR = 226.0     # free >= 8100 -> growth ≈ 226 (constant; line crosses at free 8375)
# The residual bar is NOT enforced by default, and that is a measurement result, not a retreat.
# Its own budget (250, then 500 MiB) turned out to be smaller than the quantity's motion: the SAME
# arm on the SAME box in ONE hour produced run-internal growth of 550 / 670 / 1971 / 2801 / 2984 MiB,
# and the expectation's input (the launch free reading) itself moved 1.2 GB between the pre-flight
# steady reading and the in-arm sampler reading (0.44 GB of expected growth per GB of free). A bar
# that refuses by draw re-creates exactly the defect this file was rewritten to remove: an arm that
# passed liveness, thermal and contract gets thrown away for a number nobody can reproduce.
# So: recorded and printed as a WARNING always; enforced only when `CGC_GROWTH_RESIDUAL_MB=<N>` says
# so (calibrated values: 500 -> refused median 10.51 t/s vs kept 11.70; 1000 -> refused 10.12).
GROWTH_RESIDUAL_MB = 500.0


def measured_worst(thermal_windows) -> str | None:
    """**量測窗**的熱 worst（＝`thermal_windows[].worst.label` 的最壞者）；讀不到回 None。

    [CGC 2026-10-03 線A] 為什麼需要它：`attribution()['thermal_worst']` 取的是**整臂**取樣的
    worst（含載入／冷卻／收尾），而 `is_clean()` 拿它當「這一輪乾淨嗎」的熱條件 ⇒ 一個
    **量測窗內 37/37 NOMINAL** 的臂，只要窗外的收尾讀到一次 MODERATE，就會被判不乾淨。
    當日三次同型（`mtpon_pair_warm_20261003`：OFF 窗內 NOMINAL 37/37、整臂 MODERATE；
    `mtpon_pair_20261003`：OFF 窗內 NOMINAL 3／MODERATE 29、整臂 HEAVY）。
    判準的**嚴格度不變**（仍要求 NOMINAL），只是把「哪一段的熱」問清楚：
    被引用的是**計時段**的數字，所以熱條件看計時段。
    """
    labels = []
    for w in thermal_windows or []:
        if not isinstance(w, dict):
            continue
        wr = w.get("worst")
        lab = (wr or {}).get("label") if isinstance(wr, dict) else None
        if lab:
            labels.append(str(lab))
    if not labels:
        return None
    order = {"NOMINAL": 0, "MODERATE": 1, "HEAVY": 2, "TRAPPING": 3, "SLEEPING": 4}
    return max(labels, key=lambda x: order.get(x, 99))


def is_clean(attribution_result: dict, thermal_windows=None) -> tuple:
    """(clean, why) —— 「這一輪的歸因乾淨到可以把吞吐放上成績面嗎」的**唯一定義**。

    WHY THIS IS A FUNCTION AND NOT ONE `if` PER CALL SITE: 2026-09-28 晚上，一次 swap 歸因的
    9.94 t/s 被寫進 mindmap 節點的 `res`，還讓 `target_gap` 算出「距 decode 20 目標 49.7%」
    （`docs/PREMISE_B_CHURN_DELIVERY_2026-09-28.md` §8）。產它的是 `experiment_sync.merge_run`，
    抓到它的是 `mindmap_void_check.py`。兩個檔各寫一遍「什麼叫乾淨」就會漂移，
    而漂移的那一天就是這種病回來的時候 —— 所以定義放這裡，寫入端與檢查端都呼叫它。

    乾淨 = `attribute()` 判 `none`（不是 HEAVY/TRAPPING/SLEEPING、沒有 swap／wired 成長、
    沒有別的 llama 行程）**且** thermal 是 NOMINAL。
    缺席一律不乾淨：讀不到 attribution、verdict 空字串、verdict 是未知字串 ⇒ False。
    （`clean` **不是**本檔的標籤；標籤集是 `none`／`swap`／`both`／`thermal`／`contention`。）

    [CGC 2026-10-03 線A · 熱口徑修正] `thermal_windows` 給了就用**量測窗**的 worst
    （`measured_worst()`），沒給才退回 `attribution.thermal_worst`（整臂 worst）。
    理由見 `measured_worst()`；**嚴格度不變**（仍要求 NOMINAL），只是把「哪一段的熱算數」
    對齊到**被引用的那一段**。
    ⚠ 這是**判準修正**，不是放寬：窗內真的熱 ⇒ 結論不變；只有「窗內乾淨、窗外熱」的臂
    會由不乾淨變成乾淨。**沒有 `thermal_windows` 的舊產物行為一字不變。**
    """
    if not isinstance(attribution_result, dict):
        return False, "沒有 attribution（缺席不是乾淨）"
    v = str(attribution_result.get("verdict") or "").strip()
    if not v:
        return False, "attribution.verdict 是空的"
    if v != "none":
        why = f"attribution.verdict={v}"
        if attribution_result.get("why"):
            why += f"（{attribution_result['why']}）"
        return False, why
    th = attribution_result.get("thermal_worst")
    win = measured_worst(thermal_windows)
    if win is not None:
        th, src = win, "量測窗"
    else:
        src = "整臂(舊口徑)"
    if th is not None and str(th) != "NOMINAL":
        return False, f"attribution.verdict=none 但 {src} thermal={th}"
    return True, "attribution.verdict=none" + (f"（{src} thermal {th}）" if th else "")


def budget() -> dict:
    """The budgets, with any env override recorded INSIDE the returned dict.

    Overrides exist for honest uses: make the launch bar STRICTER for a probe run
    (`CGC_STATE_FREE_FLOOR_MB=8000`), tighten the residual bar, turn the OLD absolute growth bar
    back on (`CGC_SWAP_BUDGET_MB=512` -> `growth_enforced`), or run a deliberately dirty stress
    arm. Either way the artifact must say which bar was used -- a number that cannot state its bar
    is the defect this file exists to avoid.
    """
    out = {"growth_mb": SWAP_GROWTH_MB, "free_floor_mb": FREE_FLOOR_MB,
           "wired_growth_mb": WIRED_GROWTH_MB,
           "state_free_floor_mb": STATE_FREE_FLOOR_MB,
           "state_available_floor_mb": STATE_AVAILABLE_FLOOR_MB,
           "growth_residual_mb": GROWTH_RESIDUAL_MB,
           # 出處要跟著係數走：這組 fit 是在**一個 cell**（prod-new，n=122）上配的，而 k-pair
           # session 跑的是 delivery cell（有曲線的樣本只有 1~3 支）。殘差對其他 cell 是外推，
           # 所以產物必須說出外推自哪裡，而不是讓人以為它就是那個 cell 的校準。
           "growth_fit": {"a": GROWTH_FIT_A, "b": GROWTH_FIT_B, "floor": GROWTH_FIT_FLOOR,
                          "form": "max(floor, a + b*free)",
                          "fit_from": "prod-new n=122（單一 cell；其他 cell 為外推）"},
           # The absolute growth bar is opt-in: setting its env is what arms it. Default OFF, and
           # the artifact says so, because as a default it refused arms for starting on a clean box.
           # Both in-arm bars are opt-in; the launch-state floor is the default gate.
           # Metal 工作集那一根桿子（見 metal_gate）：上限預設去問盒子／引擎，保留預設 1024 MiB。
           "metal_working_set_mb": None, "metal_reserve_mb": METAL_RESERVE_MB,
           # 絕對成長桿是 opt-in：預設 OFF，且產物要這樣寫。起跑狀態地板才是預設的閘。
           "growth_enforced": False, "residual_enforced": False, "overridden": {}}
    for key, ev in (("growth_mb", "CGC_SWAP_BUDGET_MB"),
                    ("metal_working_set_mb", "CGC_METAL_WORKING_SET_MB"),
                    ("metal_reserve_mb", "CGC_METAL_RESERVE_MB"),
                    ("free_floor_mb", "CGC_FREE_FLOOR_MB"),
                    ("wired_growth_mb", "CGC_WIRED_BUDGET_MB"),
                    ("state_free_floor_mb", "CGC_STATE_FREE_FLOOR_MB"),
                    ("state_available_floor_mb", "CGC_STATE_AVAILABLE_FLOOR_MB"),
                    ("growth_residual_mb", "CGC_GROWTH_RESIDUAL_MB")):
        raw = os.environ.get(ev)
        if raw in (None, ""):
            continue
        try:
            out[key] = float(raw)
            out["overridden"][key] = f"{ev}={raw}"
            if key == "growth_mb":
                out["growth_enforced"] = True
            if key == "growth_residual_mb":
                out["residual_enforced"] = True
        except ValueError:
            out["overridden"][key] = f"{ev}={raw!r} (unparsable -- IGNORED, default kept)"
    return out


def steady(seconds: float = 1.4, n: int = 3) -> dict:
    """A median-of-n stamp -- because `pages_free` on this box swings GBs on its own.

    MEASURED 2026-09-28, on a quiet box with nothing launched in between: the same reading went
    free 6152 MiB -> 4334 MiB and swap 6152 MiB -> 4692 MiB in about a minute (file cache grows and
    is reclaimed; the compressor breathes). A gate on ONE sample therefore decides by luck of the
    draw -- and the two readings above sit on opposite sides of the 5000 MiB floor, i.e. the same
    box would be judged both ways.

    So the launch reading the gate judges is the MEDIAN of `n` samples spread over `seconds`, and
    the spread it saw is carried into the gate's result (`spread_free_mb`) instead of being hidden:
    a reading that can state its own swing is the whole point of this module.
    """
    n = max(1, int(n))
    if n == 1:
        return stamp()
    out = [stamp()]
    for _ in range(n - 1):
        time.sleep(max(0.0, seconds) / (n - 1))
        out.append(stamp())

    def _field(k):
        vals = sorted(x[k] for x in out if isinstance(x, dict) and x.get(k) is not None)
        if not vals:
            return None
        m = len(vals) // 2
        return vals[m] if len(vals) % 2 else (vals[m - 1] + vals[m]) / 2.0

    mid = out[len(out) // 2]
    res = {k: _field(k) for k in ("swap_used_mb", "swap_total_mb", "pages_free_mb",
                                  "pages_wired_mb", "pages_inactive_mb",
                                  "pages_speculative_mb", "pages_purgeable_mb",
                                  "pages_available_mb")}
    res["t"] = mid.get("t")
    res["llama_procs"] = mid.get("llama_procs")
    res["n_readings"] = len(out)
    for col, key in (("pages_free_mb", "spread_free_mb"),
                     ("pages_available_mb", "spread_available_mb")):
        vals = [x.get(col) for x in out if x.get(col) is not None]
        res[key] = (max(vals) - min(vals)) if vals else None
    return res


def _launch_reading(d) -> dict:
    """The launch-time reading out of either a raw `stamp()` or a `Sampler.result`.

    Both shapes reach this module (`llama_bench_matrix` has the result, `rep_split` has a stamp),
    and a gate that only understood one of them would be off in whichever caller passed the other
    -- silently, which is the failure family this repo keeps meeting.
    """
    if not isinstance(d, dict):
        return {}
    if isinstance(d.get("launch"), dict):
        return d["launch"]
    s = d.get("samples")
    if isinstance(s, list) and s and isinstance(s[0], dict):
        return s[0]
    return d


def state_gate(reading, *, b: dict | None = None, waived: str | None = None) -> dict:
    """Judge the box's state BEFORE the arm starts -- the pre-runnable half of the memory gate.

    Returns `{ok, reasons, waived, phase, free_mb, floor_mb, swap_used_mb, swap_total_mb,
    swap_headroom_mb, criteria, budget}`.

    FAIL-CLOSED ON UNREADABLE: a missing `pages_free_mb` makes the state NOT acceptable. "Could
    not measure" rendered as "fine" is the defect class this repo has now met four times (null log
    sink, `lv` key, unwired ACCEPT lines, refused-but-unread log dir), and the number it would let
    through here is a t/s taken on an unknown box.

    The SWAP STOCK is recorded, not gated: rho(t/s, launch swap_used) = +0.054 on the calibration
    set -- it separates nothing, and `swap_advice` already says why (a machine can sit at GBs of
    old swap with a quiet compressor and measure at full speed).
    """
    b = {**budget(), **(b or {})}
    d = _launch_reading(reading)
    free = d.get("pages_free_mb")
    avail = d.get("pages_available_mb")
    swap = d.get("swap_used_mb")
    total = d.get("swap_total_mb")
    floor = b["state_available_floor_mb"]
    warn_floor = b["state_free_floor_mb"]

    reasons: list[str] = []
    warnings: list[str] = []
    if avail is None:
        # 舊記錄（在 available 被採之前）只有 free：那就用 free 判，但它就是在開機那種情況下會翻轉的那個量 ——
        # 所以思考寫進結果裡，而不是假裝它沒發生。
        if free is None:
            reasons.append(
                "UNREADABLE：起跑的 pages_free/pages_available 都讀不到（讀值 %r）--"
                " 缺讀值不是通過。起跑狀態是這個闘門唯一的前提。" % (d.get("t"),))
        elif warn_floor is not None and free < warn_floor:
            reasons.append(
                "起跑 pages_free %.0f MiB < 地板 %.0f MiB（舊讀值路徑：這筆記錄沒有 available，只能用它判）"
                % (free, warn_floor))
    elif floor is not None and avail < floor:
        reasons.append(
            "起跑 available %.0f MiB < %.0f MiB（free %.0f + inactive/spec/purgeable）-- 盒子太緷：這是"
            "核心不得不把別人的 anonymous 頁推出去才能放下模型的位置，而"
            "13.6 GB 模型進 16 GB 盒子時那些被推出去的頁是結構性的。先讓盒子安静再量。"
            % (avail, floor, free if free is not None else -1))
    if free is not None and warn_floor is not None and free < warn_floor and avail is not None \
            and floor is not None and avail >= floor:
        warnings.append(
            "pages_free %.0f MiB < %.0f MiB（corpus 校準的呼吸帶），但 available %.0f MiB 夠——"
            "開機後 free list 被快取壅滿時它會讀到幾十 MiB，所以這一條是警告不是拒"
            % (free, warn_floor, avail))

    return {"ok": (not reasons) or bool(waived), "reasons": reasons, "warnings": warnings,
            "waived": waived, "phase": "pre-launch",
            "available_mb": avail, "available_floor_mb": floor,
            "free_mb": free, "free_warn_floor_mb": warn_floor,
            "floor_mb": floor,
            "swap_used_mb": swap, "swap_total_mb": total,
            "swap_headroom_mb": ((total - swap) if (total is not None and swap is not None)
                                 else None),
            "t": d.get("t"),
            # 讀值本身的擺盪也要留下來（median-of-n 的証據；單一取樣沒有這個欄位就是誠實地說「沒量」）
            "free_spread_mb": d.get("spread_free_mb"),
            "available_spread_mb": d.get("spread_available_mb"),
            "n_readings": d.get("n_readings"),
            "criteria": {
                "launch_available_floor_mb": floor,
                "launch_free_warn_mb": warn_floor,
                "free_term": "warning-only（開機後 free 會讀到幾十 MiB，它是快取統計不是可用量）",
                "swap_criterion": "off（校準：rho(t/s, 起跑 swap) = +0.054 -- 不具區辨力）"},
            "budget": b}


# ── Metal 工作集預算：池子 + Metal 駐留 vs recommendedMaxWorkingSetSize ────────────────
#
# WHY THIS IS A GATE OF ITS OWN（2026-09-29）：交付 cell 的臂在 `ggml_metal_synchronize` 上
# fail-stop（`command buffer 0 failed with status 5` ＝ `kIOGPUCommandBufferCallbackErrorOutOfMemory`），
# 而當時既有的閘門一道都沒攔：cell 口徑過、起跑 available 過、歸因不是它管的、產物也還沒寫。
# Metal 的上限是**全機**的（不是本行程的），而這一輪要同時放進去的東西是「池子（宣告的
# `--expert-cache`）」＋「Metal 駐留（`wired`，量得到的）」；兩者相加超過上限時 OOM 是必然的，
# 與盒子當時多安靜無關 ⇒ 它需要自己的桿子，而且**起跑前**就要能判。
#
# 「讀不到上限」是常態、不是異常：這台盒子 `sysctl -n iogpu.wired_limit_mb` 回 **0**（未設），
# 真正的數字由引擎在啟動時印（`ggml_metal_device_init: recommendedMaxWorkingSetSize = 11453.25 MB`）。
# 所以讀不到時回 `armed=False` 並說出「問過哪裡」——不假裝通過，也不假裝超額。
#
# 兩個半邊，跟這支檔案的其他閘門一樣：
#   * `phase="pre-launch"`：池子 ＋ 起跑 wired ⇒ 拒跑（花不到 GPU 時間）
#   * `phase="peak"`：池子 ＋ 整趟峰值 wired ⇒ 那一輪不可用（並說出是哪一項）
# 而如果引擎自己留了 OOM 的一行，那比預算算術硬：直接判那一輪沒有輸出。
METAL_RESERVE_MB = 1024.0
METAL_LIMIT_SYSCTL = "iogpu.wired_limit_mb"
METAL_CEILING_CACHE = os.path.join(ROOT, "Backup", "metal_working_set.json")


def _sysctl_wired_limit_mb():
    """(mb, why)。**0 不是一個上限，是「沒有設」** —— 所以 0 回 None（同 `worst()` 的立場）。"""
    try:
        r = subprocess.run(["sysctl", "-n", METAL_LIMIT_SYSCTL],
                           capture_output=True, text=True, timeout=5)
    except Exception as e:  # noqa: BLE001 - 沒有這個 key 不該殺掉一次量測
        return None, "讀不到（%s）" % type(e).__name__
    txt = (r.stdout or "").strip()
    try:
        mb = float(txt)
    except ValueError:
        return None, "讀不到（%r）" % txt
    if mb <= 0:
        return None, "%s=%s ⇒ 未設（這不是上限，引擎會自己用 recommendedMaxWorkingSetSize）" % (
            METAL_LIMIT_SYSCTL, txt)
    return mb, "sysctl %s" % METAL_LIMIT_SYSCTL


def metal_ceiling_from_log(text):
    """引擎自己印的那一行：`ggml_metal_device_init: recommendedMaxWorkingSetSize  = 11453.25 MB`。"""
    m = re.search(r"recommendedMaxWorkingSetSize\s*=\s*([\d.]+)\s*MB", text or "")
    if not m:
        return None, "本次 stderr 沒有 recommendedMaxWorkingSetSize"
    try:
        return float(m.group(1)), "引擎 stderr 本次的 recommendedMaxWorkingSetSize"
    except ValueError:
        return None, "那行的數字讀不出（%r）" % m.group(1)


def metal_ceiling(log_text=None, cache_path=None, b: dict | None = None):
    """(mb, source)。優先序與理由：

    1. `CGC_METAL_WORKING_SET_MB`：人明確指定的真實（用於換機器或除錯）
    2. **本次** stderr：同一顆引擎剛說的話，是這個行程正在跑的那個上限
    3. `sysctl iogpu.wired_limit_mb`（>0 才算）
    4. 快取檔（上一次某場跑完留下來的）：讓**起跑前**那道閘真的有數字可用
       —— 這是這台盒子上唯一能在 spawn 之前拿到它的方法（sysctl 是 0）
    5. 都沒有 ⇒ `(None, why)` ⇒ 閘門 `armed=False`
    """
    b = {**budget(), **(b or {})}
    if b.get("metal_working_set_mb") is not None:
        return float(b["metal_working_set_mb"]), "CGC_METAL_WORKING_SET_MB（人指定）"
    mb, src = metal_ceiling_from_log(log_text)
    if mb is not None:
        return mb, src
    mb, sys_why = _sysctl_wired_limit_mb()
    if mb is not None:
        return mb, "sysctl %s（%.0f MB）" % (METAL_LIMIT_SYSCTL, mb)
    path = cache_path or METAL_CEILING_CACHE
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        v = d.get("mb")
        if isinstance(v, (int, float)) and v > 0:
            return float(v), "快取 %s（%s 記於 %s）" % (
                os.path.relpath(path, ROOT), d.get("source") or "?", d.get("when") or "?")
    except Exception:  # noqa: BLE001  沒有快取就沒有，不是錯
        pass
    return None, "沒有上限可用（%s；沒有快取檔；本次還沒有 stderr）" % sys_why


def remember_metal_ceiling(mb, source, cache_path=None, when=None, peak_wired_mb=None):
    """把引擎說的上限（以及可選的**上一趟峰值 wired**）記下來，讓下一次**起跑前**那道閘有數字。

    為什麼連峰值一起記：起跑前量不到引擎自己的 footprint（`_pre` 取在 spawn **之前**，那時盒子
    上還沒有它就 12 GB 的那一塊）。所以「池子 ＋ 起跑 wired」通過**不代表**整趟通過——
    上一趟的峰值是唯一能在起跑前說出這件事的數字（2026-09-28 量到 12.4 GB，另一條線的文件有）。

    回寫入的路徑或 None。失敗不拋 —— 一個快取寫不進去不該讓一次量測掛掉。
    """
    if not isinstance(mb, (int, float)) or mb <= 0:
        return None
    path = cache_path or METAL_CEILING_CACHE
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rec = {"mb": float(mb), "source": source,
               "when": when or time.strftime("%Y-%m-%d %H:%M:%S")}
        if isinstance(peak_wired_mb, (int, float)) and peak_wired_mb > 0:
            rec["peak_wired_mb"] = float(peak_wired_mb)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        return path
    except Exception:  # noqa: BLE001
        return None


def metal_learned_peak(cache_path=None):
    """(mb, note)：上一次某趟量到的**峰值 wired**（引擎自己的 Metal 駐留），否則 (None, why)。"""
    path = cache_path or METAL_CEILING_CACHE
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        v = d.get("peak_wired_mb")
        if isinstance(v, (int, float)) and v > 0:
            return float(v), "快取 %s（記於 %s）" % (os.path.relpath(path, ROOT), d.get("when") or "?")
    except Exception:  # noqa: BLE001
        pass
    return None, "沒有上一趟的峰值可用"


def metal_fail_line(text):
    """(line, why)：引擎自己說的 Metal OOM／fail-stop，沒有就 (None, "")。

    比預算算術硬：有這一行的那一輪**已經沒有輸出可用**，不是「跑得慢」。
    """
    if not text:
        return None, ""
    for line in text.splitlines():
        s = line.strip()
        if ("kIOGPUCommandBufferCallbackErrorOutOfMemory" in s or "Insufficient Memory" in s
                or "CGC-METAL-FAIL" in s):
            return s, ("引擎回報 Metal 命令緩衝區 OOM／fail-stop（status 5）"
                       "⇒ 這一輪沒有可用的輸出（不是「跑得慢」，是「跑不完」）")
    return None, ""


def metal_gate(reading, pool_mb=None, *, pool_source="", b: dict | None = None,
               ceiling_mb=None, ceiling_source=None, phase="pre-launch", log_text=None,
               cache_path=None, waived=None) -> dict:
    """Pool ＋ Metal 駐留 ＋ 保留 vs 上限。拒跑（pre-launch）或判該輪不可用（peak），並點名超額的是哪一項。

    `reading`：`phase="pre-launch"` 吃 `steady()` 或 `Sampler.result`（都認，見 `_launch_reading`）；
    `phase="peak"` 吃 `Sampler.result`（用整趟的 `worst().max_wired_mb`）。

    三個項目都有出處：池子＝臂宣告的（`--expert-cache`），wired＝量到的，保留＝留給別的程式的
    （Metal 上限是全機的，不是本行程的）。`armed=False` 時**不是通過**：它是「這一輪沒有這根桿子」。
    """
    b = {**budget(), **(b or {})}
    store = dict(reading) if isinstance(reading, dict) else {}
    if phase == "peak":
        w = worst(store.get("samples") or [])
        wired, wired_src = w.get("max_wired_mb"), "整趟峰值 pages_wired（memory.worst.max_wired_mb）"
    else:
        d = _launch_reading(store)
        wired, wired_src = d.get("pages_wired_mb"), "起跑 pages_wired（child 出現之前）"
    if ceiling_mb is None:
        ceiling_mb, ceiling_source = metal_ceiling(log_text=log_text, cache_path=cache_path, b=b)
    reserve = float(b["metal_reserve_mb"])
    fail_line, fail_why = metal_fail_line(log_text)

    items = []
    if pool_mb is not None:
        items.append({"name": "pool_mb", "label": "池子", "mb": float(pool_mb),
                      "source": pool_source or "宣告的池子大小"})
    if wired is not None:
        items.append({"name": "wired_mb", "label": "Metal 駐留", "mb": float(wired),
                      "source": wired_src})
    items.append({"name": "reserve_mb", "label": "保留", "mb": reserve,
                  "source": "留給別的程式（Metal 上限是全機的）＝ CGC_METAL_RESERVE_MB"})
    label_of = {i["name"]: i["label"] for i in items}

    armed = ceiling_mb is not None and pool_mb is not None and wired is not None
    total = (sum(i["mb"] for i in items) if (pool_mb is not None and wired is not None) else None)
    over = (total - ceiling_mb) if (armed and total is not None) else None
    headroom = (-over) if over is not None else None
    biggest = None
    if armed:
        variable = [i for i in items if i["name"] in ("pool_mb", "wired_mb")]
        biggest = max(variable, key=lambda i: i["mb"])["name"] if variable else None

    reasons, warnings = [], []
    if armed and over is not None and over > 0:
        lever = {
            "pool_mb": "（池子是宣告的 ⇒ 它是唯一能直接調的一項；但 2026-09-25 的曲線說 8→4 GiB "
                       "慢 36%，所以先確認超額不是由 wired 那一項造成的）",
            "wired_mb": "（wired 峰值 ≈ 模型權重的 Metal 駐留：另一條線 2026-09-28 量到 12.4 GB ⇒ "
                        "縮 pool 解不掉這一項，要從模型側或「同時間不要跑別的」下手）",
        }.get(biggest, "")
        reasons.append(
            "Metal 工作集超額：%s ＝ %.0f MiB ＞ 上限 %.0f MiB（%s）⇒ 超 %.0f MiB；"
            "最大項＝「%s %.0f MiB」%s"
            % (" ＋ ".join("%s %.0f" % (i["label"], i["mb"]) for i in items), total,
               ceiling_mb, ceiling_source, over, label_of.get(biggest, biggest),
               next(i["mb"] for i in items if i["name"] == biggest) if biggest else 0.0, lever))
    elif not armed:
        missing = [n for n, v in (("上限", ceiling_mb), ("池子", pool_mb), ("Metal 駐留", wired))
                   if v is None]
        warnings.append("Metal 預算閘未武裝：讀不到 %s ⇒ 這一輪**沒有**這根桿子（不等於通過）；"
                        "上限可由 CGC_METAL_WORKING_SET_MB 指定，引擎跑過一次的 stderr 也會留下它"
                        % "、".join(missing))
    # 起跑合格 ≠ 整趟合格：引擎自己的那一塊（峰值 Metal 駐留）在 spawn 之前不存在，
    # 所以起跑前那道閘永遠低估它。上一趟的峰值可以把它說出來（只有警告，不拒跑 ——
    # 「必然換頁」不是「不准跑」，那條線由歸因閘畫）。
    learned_peak, learned_src = (None, "")
    if phase != "peak":
        learned_peak, learned_src = metal_learned_peak(cache_path)
        if armed and learned_peak is not None and pool_mb + learned_peak + reserve > ceiling_mb:
            warnings.append(
                "起跑合格、但整趟會超：池子 %.0f ＋ 上一趟峰值 Metal 駐留 %.0f ＋ 保留 %.0f"
                " ＝ %.0f MiB ＞ 上限 %.0f MiB（%s）⇒ 這一輪的池子必然有一部分在 swap 上。"
                "起跑那道閘看不到這一塊，因為它取在 spawn 之前。"
                % (pool_mb, learned_peak, reserve, pool_mb + learned_peak + reserve,
                   ceiling_mb, learned_src))
    if fail_line:
        reasons.append("%s：%s" % (fail_why, fail_line[:180]))

    return {"ok": (not reasons) or bool(waived), "armed": armed, "waived": waived,
            "phase": phase, "items": items, "total_mb": total,
            "ceiling_mb": ceiling_mb, "ceiling_source": ceiling_source,
            "reserve_mb": reserve, "headroom_mb": headroom, "over_by_mb": over,
            "biggest": biggest, "reasons": reasons, "warnings": warnings,
            "metal_fail": ({"line": fail_line, "why": fail_why} if fail_line else None),
            "log_had_metal_fail": fail_line is not None,
            "learned_peak_mb": learned_peak, "learned_peak_source": learned_src,
            "criteria": {
                "rule": "pool（宣告）＋ Metal 駐留（量到的）＋ 保留 <= recommendedMaxWorkingSetSize",
                "wired_item": wired_src,
                "reserve_mb": reserve,
                "cannot_judge": "上限或任一項讀不到 ⇒ armed=False（不判死，也不假裝通過）",
                "direct_evidence": "引擎若有 Metal OOM／fail-stop 一行 ⇒ 直接判該輪無輸出（與預算算術獨立）",
            },
            "budget": b}


def curve(mem_result: dict, max_points: int = 12) -> list:
    """Downsampled `[[t, swap_mb, free_mb], ...]` -- the readable shape of the same curve.

    The RAW series stays where it was measured (`mem_result["samples"]`, one point per
    `interval_s`); this is for the record's summary and for printing. First and last are always
    kept (they are the launch/end readings the growth is defined on) and so is the PEAK -- a curve
    that smooths away its own peak would hide the thing being gated.
    """
    s = [x for x in (mem_result or {}).get("samples", []) if isinstance(x, dict)]
    if not s:
        return []
    pts = [[x.get("t"), x.get("swap_used_mb"), x.get("pages_free_mb")] for x in s]
    if len(pts) <= max_points:
        return pts
    n = len(pts)
    peak = max(range(n), key=lambda i: pts[i][1] if pts[i][1] is not None else -1)
    # Evenly spaced FIRST (guarantees <= max_points including both ends), then the peak is moved
    # IN by replacing its nearest non-endpoint neighbour. Appending it instead would make the
    # "12 points" summary 13 points, which is how a cap quietly stops being a cap.
    idx = sorted({round(i * (n - 1) / (max_points - 1)) for i in range(max_points)})
    if peak not in idx:
        cand = [j for j in idx if j not in (0, n - 1)]
        if cand:
            idx[idx.index(min(cand, key=lambda j: abs(j - peak)))] = peak
        else:
            idx.append(peak)
    return [pts[i] for i in sorted(set(idx))]


def sparkline(points, width: int = 48) -> str:
    """A text sparkline of the swap column, so a curve is visible in one log line."""
    vals = [p[1] for p in points if p[1] is not None]
    if len(vals) < 2:
        return "(no curve)"
    lo, hi = min(vals), max(vals)
    ramp = "▁▂▃▄▅▆▇█"
    if hi - lo < 1.0:
        return "·" * min(len(vals), width)
    step = max(1, len(vals) // width)
    return "".join(ramp[min(len(ramp) - 1,
                           int((v - lo) / (hi - lo) * (len(ramp) - 1) + 0.5))]
                   for v in vals[::step])


def curve_gate(mem_result: dict, *, b: dict | None = None, waived: str | None = None,
               launch_reading: dict | None = None) -> dict:
    """Judge ONE arm's run-internal memory behaviour: usable, or not usable.

    Returns `{ok, reasons, waived, growth_mb, peak_growth_mb, peak_swap_mb, peak_at, min_free_mb,
    wired_growth_mb, residual_mb, growth_predicted_mb, n, budget, state_gate, growth_criterion,
    wired_criterion, free_criterion, curve, sparkline}`.

    WHAT IS ENFORCED, AND WHY NOT THE REST (calibrated 2026-09-28, module docstring):

      * the STATE half (`state_gate`, on the launch reading) is the gate. It runs before the arm
        in the callers; here it is re-derived from the same reading so the artifact carries it too.
      * the RESIDUAL half is enforced: `growth - (A + B*launch_free) <= 250 MiB`. The absolute
        growth is NOT, by default: it is -0.870 correlated with launch free, so as a bar it mostly
        asked "was the box roomy" and charged the arm for its own footprint (a config with a
        bigger expert cache evicts more on an identical box). `CGC_SWAP_BUDGET_MB=<N>` arms it.
      * wired growth is recorded and NOT enforced: rho(t/s, wired growth) = **+0.230**, i.e. the
        group this bar used to refuse is faster (median 11.70 vs 10.80).
      * the in-arm min_free floor (`free_floor_mb`) also stays off -- 14..50 MiB in every bucket.

    FAIL-CLOSED ON UNREADABLE: a missing launch free reading makes the arm NOT usable. A gate that
    treats "could not measure" as "fine" is the same defect class as the log sink that swallowed
    the engine's reason -- and the number it would let through here is a t/s reading taken under
    unknown pressure.

    `waived` is a reason string: the arm stays usable but the reasons are NOT erased, because the
    waiver is a decision that belongs in the record next to the evidence.
    """
    # Merge over the defaults rather than replacing them: partial dicts arrive from tests and from
    # older callers, and a missing key here would be a KeyError in the gate -- i.e. a crash where a
    # refusal belongs, and only on the path nobody exercises.
    b = {**budget(), **(b or {})}
    m = mem_result or {}
    s = [x for x in m.get("samples", []) if isinstance(x, dict)]
    launch = m.get("launch") or (s[0] if s else {})
    end = m.get("end") or (s[-1] if s else {})
    swaps = [x.get("swap_used_mb") for x in s if x.get("swap_used_mb") is not None]
    frees = [x.get("pages_free_mb") for x in s if x.get("pages_free_mb") is not None]
    l_swap, e_swap = launch.get("swap_used_mb"), end.get("swap_used_mb")
    l_wired, e_wired = launch.get("pages_wired_mb"), end.get("pages_wired_mb")
    peak_swap = max(swaps) if swaps else None
    min_free = min(frees) if frees else None
    growth = (e_swap - l_swap) if (l_swap is not None and e_swap is not None) else None
    peak_growth = (peak_swap - l_swap) if (l_swap is not None and peak_swap is not None) else None
    wired_growth = (e_wired - l_wired) if (l_wired is not None and e_wired is not None) else None
    peak_at = next((x.get("t") for x in s if x.get("swap_used_mb") == peak_swap), None)

    # 起跑狀態（與期望值的輸入）用呼叫端交過來的 steady 讀值（如果有的話）：閘門與期望值必須
    # 同一個瞬間、同一個量，否则兩者之間的差異會變成一個沒人能重現的殘差（實測：pre-flight 的
    # steady 讀 6520 MiB，而臂內 Sampler 的單一樣本讀 5352 MiB）。
    st = state_gate(launch_reading or m, b=b, waived=waived)
    reasons: list[str] = list(st["reasons"])
    warnings: list[str] = []
    free = st["free_mb"]
    predicted = (max(b["growth_fit"]["floor"],
                     b["growth_fit"]["a"] + b["growth_fit"]["b"] * free)
                 if free is not None else None)
    residual = ((peak_growth - predicted)
                if (peak_growth is not None and predicted is not None) else None)

    if b["growth_enforced"]:
        if peak_growth is None:
            reasons.append("UNREADABLE：swap 讀值有缺（launch=%r end=%r peak=%r min_free=%r）--"
                           " 勾了 growth 閘卻量不到成長，不可用。用 memory.samples 看採樣在不在。"
                           % (l_swap, e_swap, peak_swap, min_free))
        elif peak_growth > b["growth_mb"]:
            reasons.append(
                "【選用 growth 閘】run-internal swap 成長 %+.0f MiB > 預算 %.0f（end-launch %+.0f MiB，"
                "峰值在 %s，%d 個取樣）"
                % (peak_growth, b["growth_mb"], growth, peak_at, len(s)))
    if residual is not None and residual > b["growth_residual_mb"]:
        _txt = ("成長殘差 %+.0f MiB > %.0f（峰值成長 %+.0f，起跑 free %.0f MiB 的期望值 %+.0f）-- 比同起跑"
                "狀態的臂多推了這麼多 swap。校準（n=122）：殘差 <=0 的臂 decode t/s 中位數 ~11.7、"
                "250-1000 為 10.60、>1000 為 10.12"
                % (residual, b["growth_residual_mb"], peak_growth, free, predicted))
        if b["residual_enforced"]:
            reasons.append("【選用殘差閘】" + _txt)
        else:
            warnings.append(_txt)
    if b["free_floor_mb"] is not None and min_free is not None and min_free < b["free_floor_mb"]:
        reasons.append("跑中 pages_free 觸底 %.0f MiB < 地板 %.0f MiB" % (min_free, b["free_floor_mb"]))

    growth_criterion = (
        "off（校準：與起跑 free rho -0.870 ⇒ 主要在量盒子而不是臂；同一支臂跨場次 -32..+1971 MiB，"
        "負值可出現。要舊的絕對桿子請設 CGC_SWAP_BUDGET_MB=<N>）"
        if not b["growth_enforced"] else
        "on（CGC_SWAP_BUDGET_MB 被明確設定 ⇒ 峰值成長 ≤ %.0f MiB）" % b["growth_mb"])
    residual_criterion = (
        ("warning-only（校準：同一支臂一小時內的成長歷 550/670/1971/2801/2984 MiB，而閘門自己的"
         "預算只有 %.0f MiB ⇒ 拒或不拒有一半是抽籤。仍記錄並印出；要它拒就設 CGC_GROWTH_RESIDUAL_MB）")
        % b["growth_residual_mb"]
        if not b["residual_enforced"] else
        "on（CGC_GROWTH_RESIDUAL_MB 被明確設定 ⇒ 殘差 ≤ %.0f MiB）" % b["growth_residual_mb"])
    wired_criterion = ("off（校準：rho(t/s, wired 成長) = +0.230 ⇒ 這根桿子擋的那組反而比較快："
                       ">1024 MiB 中位 11.70 t/s，≤1024 中位 10.80）")
    free_criterion = ("off（跑中 min_free 不具區辨力：每個桶 p10..p90 都是 14..50 MiB。"
                      "起跑 free 的閘在 state_gate）"
                      if b["free_floor_mb"] is None
                      else f"on，跑中地板 {b['free_floor_mb']:.0f} MiB")
    return {"ok": (not reasons) or bool(waived),
            "reasons": reasons, "warnings": warnings, "waived": waived,
            "growth_mb": growth, "peak_growth_mb": peak_growth, "peak_swap_mb": peak_swap,
            "peak_at": peak_at, "min_free_mb": min_free, "wired_growth_mb": wired_growth,
            "residual_mb": residual, "growth_predicted_mb": predicted,
            "n": len(s), "budget": b, "state_gate": st,
            "growth_criterion": growth_criterion, "residual_criterion": residual_criterion,
            "wired_criterion": wired_criterion,
            "free_criterion": free_criterion,
            "curve": curve(m), "sparkline": sparkline(curve(m))}


class Sampler:
    """Sample memory pressure on a background thread while a child owns the GPU.

    Same contract as thermal_pressure.Sampler: the FIRST sample is taken in `start()`
    (before the child exists -- the launch reading), a thread samples every `interval`,
    and `stop()` takes one final reading after the child is gone so the run's swap GROWTH
    (end - launch) is attributable to the run, not to what happened before it.
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._samples: list = []
        self._stop = threading.Event()
        self._thread = None
        self.result: dict = {}

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._samples.append(stamp())
            self._stop.wait(self.interval)

    def start(self) -> "Sampler":
        self._samples.append(stamp())  # <-- the launch reading; before the child exists
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        end = stamp()  # and the reading after the child is gone
        self._samples.append(end)
        self.result = {
            "interval_s": self.interval,
            "n": len(self._samples),
            "launch": self._samples[0] if self._samples else None,
            "end": end,
            "worst": worst(self._samples),
            "samples": list(self._samples),
        }
        return self.result

    def __enter__(self) -> "Sampler":
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False


def _selftest() -> int:
    bad = 0
    n = 0

    def check(name, ok, detail=""):
        nonlocal bad, n
        n += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
        if not ok:
            bad += 1

    live = stamp()
    # `>= 0` and not `> 0`: swap 0.0 is not「讀不到」,它是一台剛重開機的盒子最乾淨的狀態（2026-09-28 實測）
    check("stamp() reads the real sysctl shape on this machine (or is honestly unreadable)",
          live["swap_used_mb"] is None or live["swap_used_mb"] >= 0,
          f"swap={live['swap_used_mb']} free={live['pages_free_mb']} wired={live['pages_wired_mb']} procs={live['llama_procs']}")
    check("stamp() carries the availability fields the launch gate judges on",
          live.get("pages_available_mb") is not None and live["pages_available_mb"] >= 0,
          f"available={live.get('pages_available_mb')}")
    check("stamp() keeps totals separate from used", "swap_total_mb" in live)
    check("worst() of only-unreadable is None, not 0",
          worst([{"swap_used_mb": None}])["max_swap_mb"] is None)
    check("worst() takes max swap / min free / max wired",
          worst([{"swap_used_mb": 1.0, "pages_free_mb": 900.0, "pages_wired_mb": 100.0},
                 {"swap_used_mb": 3.0, "pages_free_mb": 100.0, "pages_wired_mb": 500.0}])
          == {"max_swap_mb": 3.0, "min_free_mb": 100.0, "max_wired_mb": 500.0})

    # --- attribution: each branch must name its evidence, and the rule must be checkable ---
    def _mem(launch_swap, end_swap, launch_wired=1000.0, end_wired=1000.0, procs=(1,)):
        return {"launch": {"swap_used_mb": launch_swap, "pages_wired_mb": launch_wired},
                "end": {"swap_used_mb": end_swap, "pages_wired_mb": end_wired},
                "samples": [{"swap_used_mb": launch_swap, "pages_free_mb": 100.0,
                             "pages_wired_mb": launch_wired, "llama_procs": p} for p in procs]}

    def _th(label):
        return {"worst": {"label": label}}

    check("thermal HEAVY -> thermal",
          attribution(_th("HEAVY"), _mem(1000.0, 1200.0))["verdict"] == "thermal")
    check("big swap growth -> swap",
          attribution(_th("NOMINAL"), _mem(3000.0, 3900.0))["verdict"] == "swap",
          attribution(_th("NOMINAL"), _mem(3000.0, 3900.0))["why"])
    check("HEAVY + swap growth -> both",
          attribution(_th("HEAVY"), _mem(3000.0, 3900.0))["verdict"] == "both")
    check("wired crowding + swap growth -> swap",
          attribution(_th("NOMINAL"), _mem(3000.0, 3200.0, 1000.0, 2500.0))["verdict"] == "swap")
    check("two llama processes -> contention",
          attribution(_th("NOMINAL"), _mem(1000.0, 1000.0, procs=(1, 2)))["verdict"] == "contention")
    check("clean run -> none",
          attribution(_th("NOMINAL"), _mem(1000.0, 1100.0))["verdict"] == "none")

    # --- is_clean: 唯一判準（寫入端與檢查端共用）---
    check("is_clean：唯一的乾淨標籤 none 才是乾淨",
          is_clean({"verdict": "none", "thermal_worst": "NOMINAL"})[0] is True)
    check("is_clean：swap／both／thermal／contention 都不乾淨",
          all(is_clean({"verdict": v})[0] is False
              for v in ("swap", "both", "thermal", "contention", "clean")))
    check("is_clean：verdict=none 但 thermal 不是 NOMINAL 仍不乾淨",
          is_clean({"verdict": "none", "thermal_worst": "HEAVY"})[0] is False)
    # [CGC 2026-10-03 線A] 熱口徑：有量測窗就看量測窗（整臂 worst 含載入／冷卻／收尾，
    # 不是被引用的那一段 ⇒ 舊口徑會把「窗內 37/37 NOMINAL」的臂誤判成不乾淨）。
    _mw = lambda *ls: [{"kind": "tg", "worst": {"label": l}} for l in ls]
    check("is_clean：窗內 NOMINAL／整臂 MODERATE ⇒ 乾淨（舊口徑會誤判）",
          is_clean({"verdict": "none", "thermal_worst": "MODERATE"},
                   thermal_windows=_mw("NOMINAL", "NOMINAL"))[0] is True)
    check("is_clean：窗內 MODERATE／整臂 NOMINAL ⇒ 不乾淨（窗內才作數）",
          is_clean({"verdict": "none", "thermal_worst": "NOMINAL"},
                   thermal_windows=_mw("NOMINAL", "MODERATE"))[0] is False)
    check("is_clean：沒有 thermal_windows ⇒ 退回整臂 worst（舊產物行為不變）",
          is_clean({"verdict": "none", "thermal_worst": "MODERATE"})[0] is False)
    check("measured_worst：取最壞者；空／None ⇒ None",
          measured_worst(_mw("NOMINAL", "HEAVY", "MODERATE")) == "HEAVY"
          and measured_worst([]) is None and measured_worst(None) is None)
    check("is_clean：理由要寫出用的是哪一段熱（可爭議）",
          "量測窗" in is_clean({"verdict": "none", "thermal_worst": "MODERATE"},
                              thermal_windows=_mw("NOMINAL"))[1])
    check("is_clean：缺席／空字串不是乾淨（fail-closed）",
          is_clean({})[0] is False and is_clean(None)[0] is False
          and is_clean({"verdict": ""})[0] is False)
    check("is_clean：reason 帶著為什麼（引用時要一起寫）",
          "swap" in is_clean({"verdict": "swap", "why": "swap_growth=1579.0 MiB"})[1])

    # --- Sampler: launch-before-spawn and end-after are the whole point --------------------
    pre = stamp()
    with Sampler(interval=0.05) as s:
        time.sleep(0.3)
    r = s.result
    check("Sampler takes its launch reading BEFORE the work",
          r["launch"]["t"] == pre["t"], f"{r['launch']['t']} vs {pre['t']}")
    check("Sampler samples during the run, not only at the ends", r["n"] >= 4, f"n={r['n']}")
    check("Sampler records an end reading after the work", "end" in r and r["end"] is not None)
    check("Sampler's worst() accounts for the series",
          r["worst"]["max_swap_mb"] is None or
          r["worst"]["max_swap_mb"] == max(x["swap_used_mb"] for x in r["samples"]
                                           if x["swap_used_mb"] is not None))

    # --- top_rss / swap_advice: the reminder must be actionable, and the verdict must be ----
    # --- derived from a real reading rather than from the knob settings ------------------
    top = top_rss(5)
    check("top_rss() parses the real ps shape (or honestly returns nothing)",
          all(isinstance(x["rss_mb"], float) and x["proc"] for x in top), repr(top[:2]))
    check("top_rss() is sorted descending",
          all(top[i]["rss_mb"] >= top[i + 1]["rss_mb"] for i in range(len(top) - 1)))
    check("top_rss() returns at most n rows", len(top_rss(3)) <= 3)
    a = swap_advice(limit_mb=1e9, wait_mb=950.0)
    check("swap_advice() at a clean box says ok and names no one",
          a["verdict"] in ("ok", "warn") and (a["verdict"] == "warn" or not a["top"]), a["why"])
    check("swap_advice() over a tiny limit advises（swap 刚好 0 則是 ok，那是最乾淨的狀態）",
          swap_advice(limit_mb=1.0, wait_mb=0.0)["verdict"] in ("advise", "unknown", "ok"),
          swap_advice(limit_mb=1.0, wait_mb=0.0)["why"])
    check("swap_advice() never reports a number it did not read",
          (swap_advice()["swap_used_mb"] is None) == (swap_used_mb() is None))

    # --- curve_gate: 預算是閘門，而且「讀不到」不是通過 -------------------------------------
    def _curve(swaps, frees, wired=None):
        k = len(swaps)
        s = [{"t": f"00:{i:02d}:00", "swap_used_mb": swaps[i], "pages_free_mb": frees[i],
              "pages_wired_mb": (wired[i] if wired is not None else 1000.0)} for i in range(k)]
        return {"launch": s[0], "end": s[-1], "samples": s}

    g = curve_gate(_curve([1000, 1100, 1200], [6000, 5900, 5800]))
    check("gate: 起跑夠乾淨＋小成長 -> 可用", g["ok"] and not g["reasons"])
    g = curve_gate(_curve([1000, 4000, 3900], [9000, 300, 400]))
    check("gate: 看峰值而不只看 end（回落到預算下也不放行）",
          g["peak_growth_mb"] == 3000.0 and g["growth_mb"] == 2900.0 and g["peak_at"] == "00:01:00")
    check("gate: 殘差超標預設只警告、不拒（校準：它自己比預算還吵）",
          g["ok"] and not g["reasons"] and any("殘差" in w for w in g["warnings"]),
          f"殘差 {g['residual_mb']:+.0f}（預測 {g['growth_predicted_mb']:+.0f}）")
    g = curve_gate(_curve([1000, 4000, 3900], [9000, 300, 400]),
                   b={"growth_residual_mb": 500.0, "residual_enforced": True})
    check("gate: 設了 CGC_GROWTH_RESIDUAL_MB 才拒（且指名是選用閘）",
          not g["ok"] and any("選用殘差閘" in r for r in g["reasons"]))
    g = curve_gate(_curve([1000, 1200], [5300, 5200]),
                   launch_reading={"pages_free_mb": 6520.0, "swap_used_mb": 1000.0})
    check("gate: 期望值用呼叫端交來的 steady 讀值（不是臂內單一樣本）",
          g["state_gate"]["free_mb"] == 6520.0
          and abs(g["growth_predicted_mb"] - max(226.0, 3914.0 - 0.4404 * 6520.0)) < 1.0,
          f"free={g['state_gate']['free_mb']} 期望={g['growth_predicted_mb']:.0f}")
    # 絕對成長桿子現在是選用：預設關，且結果裡要寫出「為什麼關」
    g0 = curve_gate(_curve([1000, 4000, 3900], [9000, 8000, 8000]))
    check("gate: 絕對成長預設不當閘（只當證據）",
          g0["growth_mb"] == 2900.0 and "off" in g0["growth_criterion"], g0["growth_criterion"][:60])
    g1 = curve_gate(_curve([1000, 4000, 3900], [9000, 8000, 8000]),
                    b={"growth_enforced": True, "growth_mb": 512.0, "growth_residual_mb": 1e9})
    check("gate: 設了 CGC_SWAP_BUDGET_MB 才把絕對桿子打開（且指名它）",
          not g1["ok"] and any("選用 growth 閘" in r for r in g1["reasons"]))
    check("gate: 起跑太緊 -> 不可用（就是那道新閘）",
          not curve_gate(_curve([1000, 1000, 1000], [3000, 2900, 2800]))["ok"])
    check("gate: 跑中 min_free 地板預設關閉（校準：不具區辨力）",
          curve_gate(_curve([1000, 1000, 1000], [6000, 100, 6000]))["ok"])
    check("gate: 跑中 free 判準的狀態寫在結果裡（不是默默不設）",
          "off" in curve_gate(_curve([1000, 1100], [9000, 9000]))["free_criterion"])
    g = curve_gate(_curve([1000, 1000], [6000, 1500]), b={"free_floor_mb": 2000.0})
    check("gate: 跑中 free 地板若開啟，觸底 -> 不可用",
          not g["ok"] and any("觸底" in r for r in g["reasons"]))
    g = curve_gate(_curve([1000, 1200], [6000, 6000], wired=[1000, 5000]))
    check("gate: wired 成長不再是閘（校準：相關 +0.230，擋到的那組反而快）",
          g["ok"] and "off" in g["wired_criterion"] and not any("wired" in r for r in g["reasons"]))
    g = curve_gate({"launch": {"swap_used_mb": None}, "end": {}, "samples": []})
    check("gate: 讀值缺席 -> 不可用（不是通過）",
          not g["ok"] and any("UNREADABLE" in r for r in g["reasons"]))
    g = curve_gate(_curve([3000, 3900], [4000, 20]), waived="故意壓 swap 的壓力臂")
    check("gate: waiver 只降 ok、不抹掉理由",
          g["ok"] and bool(g["waived"]) and bool(g["reasons"]))

    # --- state_gate: 跑前那一半，而且它必須是 fail-closed 的 -------------------------------
    sg = state_gate({"t": "12:00:00", "pages_free_mb": 6000.0, "swap_used_mb": 1000.0,
                     "swap_total_mb": 4096.0})
    check("state: 起跑夠乾淨 -> ok，且把 swap 存量記下來",
          sg["ok"] and sg["free_mb"] == 6000.0 and sg["swap_headroom_mb"] == 3096.0)
    check("state: swap 存量不是判準（寫在 criteria 裡，不是默默不管）",
          "off" in sg["criteria"]["swap_criterion"])
    sg = state_gate({"pages_free_mb": 1200.0})
    check("state: 起跑太緊 -> 不可用且指名地板與校準",
          not sg["ok"] and any("地板" in r for r in sg["reasons"]), sg["reasons"][0][:70])
    sg = state_gate({"swap_used_mb": 500.0})
    check("state: 讀不到 free -> 不可用（不是通過）",
          not sg["ok"] and any("UNREADABLE" in r for r in sg["reasons"]))
    check("state: 接受 Sampler.result 的形狀（拿 launch 那一格）",
          state_gate({"launch": {"pages_free_mb": 8000.0}, "samples": [{"pages_free_mb": 10.0}]})["ok"]
          and state_gate({"launch": {"pages_free_mb": 8000.0}}) ["free_mb"] == 8000.0)
    sg = state_gate({"pages_free_mb": 100.0}, waived="使用者決定在緊盒子上跑")
    check("state: waiver 降 ok、理由留著",
          sg["ok"] and sg["waived"] and bool(sg["reasons"]))

    # 重開機那個案例（頁頁實測）：free 幾十 MiB、swap 0/0、available 5.9 GiB ⇒ 必須放行
    boot = {"t": "16:38:54", "pages_free_mb": 64.0, "pages_inactive_mb": 5780.0,
            "pages_speculative_mb": 10.0, "pages_purgeable_mb": 23.0,
            "pages_available_mb": 5877.0, "swap_used_mb": 0.0, "swap_total_mb": 0.0}
    sg = state_gate(boot)
    check("state: 剛重開機（free 64 MiB、swap 0/0、available 5.9 GiB）-> 放行",
          sg["ok"] and not sg["reasons"], sg["reasons"][:1])
    check("state: 但仍然警告 corpus 那條 free 帶（不静默）",
          any("pages_free" in w for w in sg["warnings"]))
    check("state: 判的是 available 而非 free",
          sg["available_mb"] == 5877.0 and sg["available_floor_mb"] == STATE_AVAILABLE_FLOOR_MB)
    sg = state_gate({**boot, "pages_available_mb": 2500.0})
    check("state: available 不足 -> 不可用（就算 free 看起來無害）",
          not sg["ok"] and any("available" in r for r in sg["reasons"]))
    sg = state_gate({"pages_free_mb": 9000.0})
    check("state: 舊記錄（只有 free）走 free 路徑，且 criteria 說明白",
          sg["ok"] and sg["criteria"]["free_term"].startswith("warning-only"))
    _env_save = {k: os.environ.get(k) for k in ("CGC_STATE_FREE_FLOOR_MB", "CGC_SWAP_BUDGET_MB")}
    try:
        os.environ["CGC_STATE_FREE_FLOOR_MB"] = "9000"
        os.environ["CGC_SWAP_BUDGET_MB"] = "777"
        be = budget()
        check("budget(): 起跑地板與絕對桿子都可由 env 覆寫，且覆寫被記下來",
              be["state_free_floor_mb"] == 9000.0 and be["growth_enforced"] is True
              and be["growth_mb"] == 777.0 and "CGC_SWAP_BUDGET_MB" in be["overridden"]["growth_mb"])
        check("state: 覆寫真的生效（9000 地板拒掉 6000）",
              not state_gate({"pages_free_mb": 6000.0})["ok"])
        os.environ.pop("CGC_SWAP_BUDGET_MB", None)
        check("budget(): 沒設 env 時絕對桿子是關的（growth_enforced False）",
              budget()["growth_enforced"] is False)
    finally:
        for k, v in _env_save.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    check("gate: 部分 budget dict 不會炸（補上預設再判）",
          curve_gate(_curve([1000, 1100], [6000, 6000]),
                     b={"state_free_floor_mb": 1000.0})["ok"] is True)

    # --- steady(): 單一樣本會抽籤，所以要取中位數、並把擺盪帶進結果 ------------------------
    _real_stamp = stamp
    _seq = [{"t": f"00:{i:02d}:00", "swap_used_mb": 1000.0 + i * 100, "swap_total_mb": 8192.0,
             "pages_free_mb": f, "pages_wired_mb": 1000.0, "llama_procs": 0}
            for i, f in enumerate([6100.0, 8100.0, 4100.0])]
    _it = iter(_seq)
    try:
        globals()["stamp"] = lambda: next(_it)
        st3 = steady(seconds=0.0, n=3)
        check("steady(): 三個讀值取中位數（不被最高/最低單樣本牽走）",
              st3["pages_free_mb"] == 6100.0 and st3["swap_used_mb"] == 1100.0,
              f"free={st3['pages_free_mb']} swap={st3['swap_used_mb']}")
        check("steady(): 把讀值的擺盪記下來（4100..8100 ⇒ 4000）",
              st3["spread_free_mb"] == 4000.0 and st3["n_readings"] == 3)
        _it1 = iter([_seq[0]])          # 自己的序列：上一個迭代器已經用完（StopIteration 不是斷言）
        globals()["stamp"] = lambda: next(_it1)
        check("steady(): n=1 就是單一 stamp（不假裝有中位數）",
              steady(seconds=0.0, n=1).get("n_readings") is None)
    finally:
        globals()["stamp"] = _real_stamp
    sg = state_gate({"pages_free_mb": 6000.0, "swap_used_mb": 1000.0, "spread_free_mb": 1800.0,
                     "n_readings": 3})
    check("state: median-of-n 的來源與擺盪會進產物",
          sg["free_spread_mb"] == 1800.0 and sg["n_readings"] == 3)
    # 峰值必須明確且不落在均勻取樣的索引上，否則這條測試測不到「保留峰值」那件事
    # （第一版就錯在資料上：`1000+i for i in range(60)` 只到 1059，而断言寫 1060）。
    _pk = [1000] * 20 + [5000] + [1000] * 41        # 63 點，峰值在 idx=20
    cg = curve_gate(_curve(_pk, [4000] * len(_pk)))
    check("gate: 曲線降採樣仍保留首、尾與峰值",
          len(cg["curve"]) <= 12 and cg["curve"][0][1] == 1000 and cg["curve"][-1][1] == 1000
          and any(p[1] == 5000 for p in cg["curve"]), f"n={len(cg['curve'])}")
    check("gate: sparkline 對空/單點不炸",
          sparkline([]) == "(no curve)" and sparkline([[0, 5, 5]]) == "(no curve)")
    # --- Metal 工作集預算（2026-09-29：交付 cell 那一場 OOM 沒有閘門攔得住）-----------------
    _LOG = ("ggml_metal_device_init: recommendedMaxWorkingSetSize  = 11453.25 MB\n"
            "ggml_metal_synchronize: error: command buffer 0 failed with status 5\n"
            "error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)\n")
    _mb_, _src = metal_ceiling_from_log(_LOG)
    check("metal_ceiling：從引擎 stderr 讀得到上限", _mb_ == 11453.25, "%s（%s）" % (_mb_, _src))
    check("metal_ceiling_from_log：沒有那行 ⇒ None（不猜）", metal_ceiling_from_log("x")[0] is None)
    check("sysctl 的 0 是「未設」不是上限",
          _sysctl_wired_limit_mb()[0] is None or _sysctl_wired_limit_mb()[0] > 0,
          str(_sysctl_wired_limit_mb()))
    # 健康的一趟（起跑 wired 1937、池 8192、保留 1024）：不拒跑
    _ok = metal_gate({"launch": {"pages_wired_mb": 1937.0}}, 8192.0, phase="pre-launch",
                     pool_source="--expert-cache", ceiling_mb=11453.25,
                     ceiling_source="測試注入")
    check("metal_gate：池 8192 ＋ 起跑 wired 1937 ＋ 保留 1024 < 11453 ⇒ 放行",
          _ok["ok"] and _ok["armed"], "headroom=%.0f" % _ok["headroom_mb"])
    # 09-28 那一趟的幾何（峰值 wired 12423）＋ 8 GiB 池子：必然超額
    _bad = metal_gate({"samples": [{"pages_wired_mb": 1800.0}, {"pages_wired_mb": 12423.0}]},
                      8192.0, phase="peak", pool_source="--expert-cache",
                      ceiling_mb=11453.25, ceiling_source="測試注入")
    check("metal_gate：峰值 wired 12423 ＋ 池 8192 ⇒ 超額且拒",
          (not _bad["ok"]) and _bad["over_by_mb"] and _bad["over_by_mb"] > 9000,
          "over=%.0f" % _bad["over_by_mb"])
    check("metal_gate：超額時要**點名**最大項（不然不知道調哪一個）",
          _bad["biggest"] == "wired_mb" and "Metal 駐留" in _bad["reasons"][0]
          and "12423" in _bad["reasons"][0],
          _bad["reasons"][0][:90])
    check("metal_gate：每個項目都帶出處",
          all(i.get("source") for i in _bad["items"]) and len(_bad["items"]) == 3)
    # 引擎自己留了 OOM 一行：那一輪沒有輸出，與預算算術無關
    _fail = metal_gate({"launch": {"pages_wired_mb": 1937.0}}, 8192.0, ceiling_mb=11453.25,
                       ceiling_source="測試注入", log_text=_LOG)
    check("metal_gate：stderr 有 kIOGPUCommandBufferCallbackErrorOutOfMemory ⇒ 直接判該輪無輸出",
          (not _fail["ok"]) and _fail["log_had_metal_fail"],
          (_fail["reasons"] or [""])[-1][:70])
    # 讀不到上限 ⇒ armed=False（**不是**通過，不該靜默）
    _un = metal_gate({"launch": {"pages_wired_mb": 1937.0}}, 8192.0, ceiling_mb=None,
                     log_text="no ceiling here", cache_path="/nonexistent/cache.json")
    check("metal_gate：讀不到上限 ⇒ armed=False 且有警告（不假裝通過）",
          _un["armed"] is False and _un["warnings"] and not _un["reasons"])
    _un2 = metal_gate({"launch": {"pages_wired_mb": 1937.0}}, None, ceiling_mb=11453.25,
                      ceiling_source="測試注入")
    check("metal_gate：沒宣告池子 ⇒ 也判不了（armed=False）",
          _un2["armed"] is False and any("池子" in w for w in _un2["warnings"]))
    # 起跑合格 ≠ 整趟合格：上一趟的峰值要能把它說出來（警告，不拒）
    import tempfile as _tf
    _d = _tf.mkdtemp(prefix="mp_metal_")
    _c = os.path.join(_d, "c.json")
    remember_metal_ceiling(11453.25, "測試", cache_path=_c, peak_wired_mb=12105.0)
    _lp, _ = metal_learned_peak(_c)
    check("remember/learned_peak：上限與上一趟峰值都記得下來", _lp == 12105.0)
    _learn = metal_gate({"launch": {"pages_wired_mb": 1972.0}}, 8192.0, phase="pre-launch",
                        cache_path=_c)
    check("metal_gate：起跑合格但上一趟峰值會超 ⇒ 放行＋警告說明白（不假裝沒事）",
          _learn["ok"] and any("起跑合格、但整趟會超" in w for w in _learn["warnings"]),
          (_learn["warnings"] or [""])[0][:60])
    import shutil as _sh
    _sh.rmtree(_d, ignore_errors=True)
    _waived = metal_gate({"samples": [{"pages_wired_mb": 12423.0}]}, 8192.0, phase="peak",
                         ceiling_mb=11453.25, ceiling_source="測試注入", waived="急件")
    check("metal_gate：--waive 只是不拒，理由仍留在產物裡",
          _waived["ok"] and _waived["reasons"] and _waived["waived"] == "急件")

    bd = budget()
    check("budget(): 預設值有出處（跑中 free 判準仍預設關閉）",
          bd["growth_mb"] == SWAP_GROWTH_MB and bd["free_floor_mb"] is None)
    check("budget(): 起跑地板與殘差上限的預設值有出處",
          bd["state_free_floor_mb"] == STATE_FREE_FLOOR_MB
          and bd["growth_residual_mb"] == GROWTH_RESIDUAL_MB
          and bd["growth_fit"]["a"] == GROWTH_FIT_A
          and bd["growth_fit"]["floor"] == GROWTH_FIT_FLOOR
          and bd["growth_fit"]["form"] == "max(floor, a + b*free)")
    # 兩段模型的關鍵：空機那一端期望值不得隨 free 無限下降（線性版就是這樣誤拒了 launch 2）
    _gf = budget()["growth_fit"]
    _pred_roomy = max(_gf["floor"], _gf["a"] + _gf["b"] * 9191.0)
    _pred_linear = _gf["a"] + _gf["b"] * 9191.0
    check("gate: 高 free 端的期望值是地板、不是負值（線性版在這裡低估到負）",
          _pred_roomy == _gf["floor"] and _pred_linear < 0,
          f"兩段 {_pred_roomy:.0f} vs 線性 {_pred_linear:.0f}")
    g = curve_gate(_curve([1000, 1400], [10000, 10000]))
    check("gate: 空機上多推 400 MiB（<地板+500）→ 放行", g["ok"], f"殘差 {g['residual_mb']:+.0f}")

    print()
    print(f"  {n - bad}/{n} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    if "--advice" in sys.argv:
        # CLI form for the shell launcher: `--advice [LIMIT_MB]`
        _i = sys.argv.index("--advice")
        _lim = float(sys.argv[_i + 1]) if len(sys.argv) > _i + 1 else SWAP_START_LIMIT_MB
        _a = swap_advice(limit_mb=_lim)
        print(f"[swap] verdict={_a['verdict']} {_a['why']}")
        for _r in _a["top"]:
            print(f"       {_r['rss_mb']:8.1f} MiB  {_r['proc']}")
        sys.exit(0)
    s = stamp()
    print(f"swap used={s['swap_used_mb']} MiB  free={s['pages_free_mb']:.1f} MiB  "
          f"wired={s['pages_wired_mb']:.1f} MiB  llama_procs={s['llama_procs']}")
