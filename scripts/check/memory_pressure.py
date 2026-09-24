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
refuse to run. The threshold numbers are first-pass and must be tuned against history (the
swap_log.tsv series is the calibration set) before any harness starts gating on them.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time

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
    """How many llama-named processes are alive right now (neighbours = contention)."""
    try:
        out = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5)
        n = sum(1 for l in out.stdout.splitlines() if "llama" in l.lower()
                and "grep" not in l.lower())
        return n
    except Exception:  # noqa: BLE001
        return -1


def swap_used_mb() -> float | None:
    s = _sysctl_swap()
    return s["used_mb"] if s else None


def stamp() -> dict:
    """One JSON-safe reading: `{t, swap_used_mb, swap_total_mb, pages_free_mb,
    pages_wired_mb, llama_procs}`. Missing fields are None (never folded into a number)."""
    s = _sysctl_swap()
    v = _vm_stat()
    return {
        "t": time.strftime("%H:%M:%S"),
        "swap_used_mb": s["used_mb"] if s else None,
        "swap_total_mb": s["total_mb"] if s else None,
        "pages_free_mb": (v["free_pages"] * _PAGE_KB / 1024.0) if v else None,
        "pages_wired_mb": (v["wired_pages"] * _PAGE_KB / 1024.0) if v else None,
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


def attribution(thermal_result: dict, mem_result: dict) -> dict:
    """Classify the run: thermal / swap / contention / both / none.

    `thermal_result` is a thermal_pressure.Sampler.result dict; `mem_result` is this
    module's Sampler.result. The rule is documented in the module docstring; every branch
    names its evidence so a reader can dispute the call instead of the number.
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

    if hot and (swap_pressure or wired_crowd):
        verdict, why = "both", f"thermal={th_worst}, swap_growth={swap_growth} MiB"
    elif hot:
        verdict, why = "thermal", f"thermal worst {th_worst}"
    elif swap_pressure or wired_crowd:
        verdict, why = "swap", f"swap_growth={swap_growth} MiB, max_swap={max_swap} MiB"
    elif contested:
        verdict, why = "contention", f"llama_procs={max_procs}"
    else:
        verdict, why = "none", f"swap_growth={swap_growth} MiB, thermal={th_worst}"

    return {"verdict": verdict, "why": why, "thermal_worst": th_worst,
            "swap_growth_mb": swap_growth, "max_swap_mb": max_swap,
            "wired_growth_mb": wired_growth, "max_llama_procs": max_procs}


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
    check("stamp() reads the real sysctl shape on this machine (or is honestly unreadable)",
          live["swap_used_mb"] is None or live["swap_used_mb"] > 0,
          f"swap={live['swap_used_mb']} free={live['pages_free_mb']} wired={live['pages_wired_mb']} procs={live['llama_procs']}")
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

    print()
    print(f"  {n - bad}/{n} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    s = stamp()
    print(f"swap used={s['swap_used_mb']} MiB  free={s['pages_free_mb']:.1f} MiB  "
          f"wired={s['pages_wired_mb']:.1f} MiB  llama_procs={s['llama_procs']}")
