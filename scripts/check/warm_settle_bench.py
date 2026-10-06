#!/usr/bin/env python3
"""Warm-settle benchmark wrapper — get a stable-swap (attribution=none) window.

WHY THIS EXISTS
---------------
The production measurement path (`harness.py bench` -> `llama_bench_matrix.py`) spawns a
STANDALONE `llama-bench` per arm. Each arm reloads the 13.65 GB model from scratch, and the
attribution window (memory/launch -> memory/end) wraps the whole load. On this 16 GB box the
footprint (model + 8 GB pool) exceeds RAM, so a COLD load pushes 3.9-4.8 GB of swap growth
inside the window -> `attribution=swap` (dirty). The only way to get `attribution=none` is for
the box to already be WARM at launch: the model must be resident in the OS page cache so the
measured arm's reload causes minimal swap growth (C12 did exactly this: +376 MB, not +4 GB).

A standalone warmup process frees its anonymous memory on exit, BUT the model is mmap'd from
disk, so its pages persist in the OS file/page cache across process boundaries. A second
`llama-bench` that mmaps the SAME file then faults the model in from cache -> tiny swap growth.

So the method is: (1) run a throwaway warmup bench to fault the model into cache and let swap
grow to its steady level, (2) poll `vm.swapusage` until swap stops growing (settled), (3) run
the REAL measured bench, which now reloads from cache and keeps swap flat -> clean window.

The measured run uses the EXACT same `harness.py bench` caliber path, so its artifact is
unchanged in口径 — only the box state at launch differs.

USAGE
-----
  # Style A (preferred): pass the bench subcommand args only
  python3 scripts/check/warm_settle_bench.py -- --arms prod-new --cell default --json /tmp/measured.json

  # Style B: pass the full harness command (it will be used as-is, only --json is (re)set)
  python3 scripts/check/warm_settle_bench.py -- python3 scripts/check/harness.py bench \
      --arms prod-new --cell default --json /tmp/measured.json

Options (before `--`):
  --settle-wait N     max seconds to poll for swap stability (default 120)
  --stable-delta N    MB change between samples treated as "stable" (default 50)
  --poll N            consecutive stable samples required to declare settled (default 3)
  --poll-int N        seconds between swap samples (default 3)
  --no-warmup         skip the warmup load (just settle + measure) — use if box already warm
  --warmup-json PATH  where to dump the throwaway warmup artifact (default /tmp/warm_settle_warmup.json)
  --dry-run           print the plan and verify the swap probe, then exit (no GPU work)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HARNESS = ROOT / "scripts" / "check" / "harness.py"


def swap_used_mb() -> float | None:
    """Mirror decode_window_harness.swap_used_mb: read vm.swapusage via sysctl."""
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else None


def set_json_arg(args: list[str], path: str) -> list[str]:
    """Return a copy of `args` with `--json PATH` set (replace existing or append)."""
    out = list(args)
    for i, a in enumerate(out):
        if a == "--json" and i + 1 < len(out):
            out[i + 1] = path
            return out
        if a.startswith("--json="):
            out[i] = f"--json={path}"
            return out
    out += ["--json", path]
    return out


def build_cmd(bench_args: list[str], json_path: str) -> list[str]:
    """Build the full harness command.

    Two call styles are accepted:
      Style A (preferred): bench subcommand args only, e.g. `--arms prod-new --cell default`
                           -> [python3, harness.py, bench, *args]
      Style B: a full command was passed, e.g. `python3 scripts/check/harness.py bench --arms ...`
                           -> used as-is, only --json is (re)set.
    """
    joined = " ".join(bench_args[:3])
    if "harness.py" in joined:
        # Style B: full command supplied; just set the json path in place.
        return set_json_arg(bench_args, json_path)
    # Style A: bench subcommand args; prepend the harness.
    return [sys.executable, str(HARNESS), "bench", *set_json_arg(bench_args, json_path)]


def run_harness_bench(bench_args: list[str], json_path: str, label: str) -> int:
    cmd = build_cmd(bench_args, json_path)
    print(f"\n>>> [{label}] { ' '.join(cmd) }", flush=True)
    rc = subprocess.run(cmd).returncode
    print(f"<<< [{label}] rc={rc} (artifact -> {json_path})", flush=True)
    return rc


def report_attribution(json_path: str) -> None:
    """Best-effort: print the artifact's attribution verdict if it is a JSON list/dict."""
    p = Path(json_path)
    if not p.exists():
        return
    try:
        import json
        data = json.loads(p.read_text(errors="replace"))
    except Exception:
        return
    # artifacts may be a list of arm objects or a single dict
    items = data if isinstance(data, list) else [data]
    for it in items:
        if not isinstance(it, dict):
            continue
        # dig for attribution verdict defensively
        verdict = None
        growth = None
        # top-level
        if isinstance(it.get("attribution"), dict):
            verdict = it["attribution"].get("verdict")
            growth = it["attribution"].get("swap_growth_mb")
        # sometimes nested under [0]
        arr = it.get("arms") or it.get("rows")
        if arr and isinstance(arr, list):
            for a in arr:
                if isinstance(a, dict) and isinstance(a.get("attribution"), dict):
                    verdict = a["attribution"].get("verdict")
                    growth = a["attribution"].get("swap_growth_mb")
                    break
        if verdict:
            print(f"    attribution.verdict = {verdict}"
                  + (f"  (swap_growth={growth} MB)" if growth is not None else ""), flush=True)


def wait_settle(wait_s: int, delta_mb: float, n_stable: int, poll_s: int) -> tuple[bool, list[float]]:
    print(f"\n--- waiting for swap to settle (max {wait_s}s, stable if |Δ|≤{delta_mb}MB "
          f"for {n_stable} samples) ---", flush=True)
    samples: list[float] = []
    stable_run = 0
    t0 = time.time()
    while time.time() - t0 < wait_s:
        s = swap_used_mb()
        if s is None:
            print("    swap probe unavailable — skipping settle wait", flush=True)
            return True, samples
        samples.append(s)
        print(f"    t={time.time()-t0:5.1f}s  swap_used={s:.1f} MB", flush=True)
        if len(samples) >= 2 and abs(samples[-1] - samples[-2]) <= delta_mb:
            stable_run += 1
            if stable_run >= n_stable:
                print(f"    >>> swap settled at {s:.1f} MB after {len(samples)} samples", flush=True)
                return True, samples
        else:
            stable_run = 0
        time.sleep(poll_s)
    print(f"    !!! settle timeout — last swap={samples[-1] if samples else '?'} MB "
          f"(proceeding anyway; measured run may still be dirty)", flush=True)
    return False, samples


def main() -> int:
    argv = sys.argv[1:]
    # split on bare "--"
    if "--" in argv:
        idx = argv.index("--")
        pre, bench_args = argv[:idx], argv[idx + 1:]
    else:
        # no "--": treat everything as harness args but still parse our flags greedily
        pre, bench_args = [], argv

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--settle-wait", type=int, default=120)
    ap.add_argument("--stable-delta", type=float, default=50.0)
    ap.add_argument("--poll", type=int, default=3)
    ap.add_argument("--poll-int", type=float, default=3.0)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--warmup-json", default="/tmp/warm_settle_warmup.json")
    ap.add_argument("--dry-run", action="store_true")
    ns, _ = ap.parse_known_args(pre)

    if not bench_args or bench_args[0] != "bench":
        # be forgiving: if user passed harness args without "bench", assume bench
        if bench_args and bench_args[0].endswith("harness.py"):
            pass
    if not bench_args:
        print("usage: warm_settle_bench.py [opts] -- python3 scripts/check/harness.py bench <args>",
              file=sys.stderr)
        return 2

    # strip our own flags if they leaked into bench_args (defensive)
    bench_args = [a for a in bench_args if not a.startswith("--settle-wait")
                  and a not in ("--no-warmup", "--dry-run")
                  and not a.startswith("--stable-delta") and not a.startswith("--poll")
                  and not a.startswith("--warmup-json") and not a.startswith("--poll-int")]

    # resolve measured artifact path (respect user's --json, else default)
    measured_json = "/tmp/warm_settle_measured.json"
    for i, a in enumerate(bench_args):
        if a == "--json" and i + 1 < len(bench_args):
            measured_json = bench_args[i + 1]
        elif a.startswith("--json="):
            measured_json = a.split("=", 1)[1]

    print("=== warm_settle_bench ===")
    print(f"  measured artifact : {measured_json}")
    print(f"  warmup artifact   : {'SKIPPED' if ns.no_warmup else ns.warmup_json}")
    print(f"  settle            : wait≤{ns.settle_wait}s, |Δ|≤{ns.stable_delta}MB x{ns.poll}")

    probe = swap_used_mb()
    print(f"  swap probe        : {'OK ' + str(probe) + ' MB' if probe is not None else 'UNAVAILABLE'}")

    if ns.dry_run:
        print("  [dry-run] plan verified, exiting without GPU work.", flush=True)
        return 0

    # 1) warmup load — fault model into page cache, let swap grow to steady
    if not ns.no_warmup:
        run_harness_bench(bench_args, ns.warmup_json, "WARMUP")
    else:
        print("\n--- --no-warmup: skipping warmup load ---", flush=True)

    # 2) settle — wait for swap to stop growing
    wait_settle(ns.settle_wait, ns.stable_delta, ns.poll, ns.poll_int)

    # 3) measured run — reloads from cache, swap stays flat -> clean window
    rc = run_harness_bench(bench_args, measured_json, "MEASURED")
    print("\n--- measured attribution ---", flush=True)
    report_attribution(measured_json)
    return rc


if __name__ == "__main__":
    sys.exit(main())
