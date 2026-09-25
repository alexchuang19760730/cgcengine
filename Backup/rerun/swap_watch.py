#!/usr/bin/env python3
"""While a driver runs arms, sample the ENGINE's own swapped bytes (vmmap SWAPPED) per arm.

Why alongside the A/B rather than after it: the two load modes differ in exactly one thing that
matters to memory -- `none` makes the model anonymous and unpagable, `mmap` makes it file-backed and
clean. Whether that shows up as *the engine's own bytes sitting in swap* is a different question from
whether the box's swap total moves, and only a per-process reading answers it. Attaching during the
run is the only way to get it (the process is gone afterwards).

Writes one JSON object per sample to `--out`; each carries the wall clock so samples can be joined
to the driver's `### arm <tag> ... <time>` lines offline.

    python3 Backup/rerun/swap_watch.py --minutes 45 --out /tmp/swap_watch.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

ROOT = "/Users/alexchuang/Documents/flashkv-devserver"
sys.path.insert(0, os.path.join(ROOT, "scripts/check"))
import memory_pressure as mp          # noqa: E402

spec = importlib.util.spec_from_file_location("sop", os.path.join(ROOT, "Backup/rerun/swap_owner_probe.py"))
sop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sop)


def engine_pids() -> list[int]:
    out = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if pid.isdigit() and "llama-server" in cmd and "grep" not in cmd:
            pids.append(int(pid))
    return pids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--out", default="/tmp/swap_watch.jsonl")
    args = ap.parse_args()

    end = time.time() + args.minutes * 60
    n = 0
    with open(args.out, "a") as fh:
        while time.time() < end:
            for pid in engine_pids():
                p = sop.sample(pid)
                rec = {"ts": time.time(), "wall": time.strftime("%H:%M:%S"), "pid": pid,
                       "engine_swapped_kb": p.get("writable_swapped_kb"),
                       "engine_resident_kb": p.get("writable_resident_kb"),
                       "box_swap_mb": p.get("swap_used_mb"),
                       "pageins": None, "swapouts": None,
                       "top_regions": [r["region"][:40] for r in (p.get("regions") or [])[:3]]}
                vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
                for line in vm.splitlines():
                    if "Swapouts" in line or "Pageins" in line:
                        k = line.split(":")[0].strip()
                        v = line.split(":")[1].strip().rstrip(".")
                        if k == "Swapouts":
                            rec["swapouts"] = int(v)
                        elif k == "Pageins":
                            rec["pageins"] = int(v)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n += 1
            time.sleep(args.interval)
    print(f"swap_watch: {n} samples -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
