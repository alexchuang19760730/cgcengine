#!/usr/bin/env python3
"""Phase-split prefill A/B: is the armed slab faster, or is the clamp merely slower?

THE QUESTION (decided 2026-09-17, option D)
------------------------------------------
The phase split only exists when the whole-layer slab is armed, and only the `prefill250` profile
arms it. If arming is worth having as a default, the reason has to be a measurement. This is it.

WHY THIS DRIVES THE SERVER AND NOT llama-bench  (v2 -- the v1 design was unmeasurable)
-------------------------------------------------------------------------------------
v1 ran `llama_bench_matrix.py` with three arms. Two independent defects, both measured:

  * macOS has no `setsid`, so the detached launch never happened. The evidence was one line:
    `nohup: setsid: No such file or directory`. v2 detaches with Python's own double-fork, which is
    what `run_server.sh --detach` itself does, so it works on this platform.
  * `llama-bench` does NOT chunk. An unarmed arm has the `n_batch` clamp applied, so a 2048-token
    `-p` asserts inside the context:

        llama-context.cpp:2455: GGML_ASSERT(n_tokens_all <= cparams.n_batch) failed

    (seen in phase-pool{17}_rep{3,4}.stdout.txt of the v1 run). The slab arm passes only because it
    is the arm where the clamp is lifted. So llama-bench can measure AT MOST one of the three arms
    at p=2048 -- the comparison did not exist.

`llama-server` chunks a long prompt into `n_batch`/`n_ubatch` pieces internally, which is exactly the
production behaviour being asked about, so the whole grid is measurable through `/completion`. The
server log then also carries the mechanism, per launch: the slab arm shows `CGC-PREFILL-STREAM ...
ntok=<prompt>`, the pool arms show many `CGC-HOOK` lines instead -- so each timing arrives with the
evidence of which phase served it (v1 could not do this at all; llama-bench writes no such log).

THE ARMS
--------
All on `prefill250`, so pool / ctx / batch / MTP / template are identical and only the phase decision
can differ:

    phase-slab    slab armed, clamp lifted      -> 1 x width-2048 PREFILL graph
    phase-pool8   pool path, clamp = cap 8      -> ~256 x width-8 pool graphs
    phase-pool17  pool path, clamp = bound 17   -> ~121 x width-17 pool graphs

`phase-slab` vs `phase-pool8` is NOT single-variable: for a chunk wider than the decode width the
unarmed arm also takes the clamp, so the honest claim is "slab + no clamp" vs "pool + clamp".
`pool8` vs `pool17` differ only in the clamp width (the graph-count axis), and at the SHORT prompt
`pool17` is a single pool-path graph -- the pool baseline at matched graph count, which is what
separates "the slab is faster" from "the clamp made it slower".

PROTOCOL
--------
* Interleaved, not arm-major: launch order dominates on this box (three consecutive launches of one
  fixed command gave 246.86 / 134.83 / 110.70 t/s, free-at-launch anti-monotone, carried swap
  monotone -- docs/M1_WORKITEM2_PHASE_SPLIT_STATUS_2026-09-17.md section D).
* 4 launches per arm, discard launch 1 (cold page cache), pair by rep index, median.
* free GiB and swap-in-use recorded per launch, because carried swap is the live suspect.
* Refuses to start when another process holds the model, or free memory is below --min-free-gib. The
  v1 attempt on a contended box was SIGKILLed mid-prefill (`rc=-9` at `ntok=2048 il=12`); those
  numbers would have described jetsam, not the engine.

CAVEAT THAT MUST TRAVEL WITH ANY NUMBER FROM THIS SCRIPT: one launch of this model on this box has
been measured from 110 to 247 t/s for a fixed command, so per-arm absolutes are only trustworthy as
PAIRED comparisons within one interleaved session. The ratio between interleaved arms is the result;
the absolute t/s is context.

USAGE
    python3 scripts/check/phase_split_ab.py --dry-run        # print the 3 resolved arms, no launch
    python3 scripts/check/phase_split_ab.py --reps 4         # the measurement
    python3 scripts/check/phase_split_ab.py --aggregate-only # re-read an earlier --out directory
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PORT = 8080
BASE = f"http://127.0.0.1:{PORT}"

ARMS: dict[str, dict[str, str]] = {
    "phase-slab":   {},
    "phase-pool8":  {"CGC_PREFILL_STREAM": "0"},
    "phase-pool17": {"CGC_PREFILL_STREAM": "0", "CGC_POOL_MAX_TOKENS": "64"},
}
# ~2048 tokens (the M1 exit criterion's chunk) and ~17 tokens (a single pool-path graph at the
# clamp-17 arm). Widths are READ BACK from the response's own timings, never assumed.
LONG_PROMPT = ("The history of the city of Paris spans more than two thousand years. " * 200)[:11000]
SHORT_PROMPT = "Please give me three colours and two shapes, comma separated:"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mem_state() -> dict:
    vs = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page, free, inactive = 16384, 0, 0
    for line in vs.splitlines():
        m = re.match(r"Pages free:\s+(\d+)", line)
        if m:
            free = int(m.group(1))
        m = re.match(r"Pages inactive:\s+(\d+)", line)
        if m:
            inactive = int(m.group(1))
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([0-9.]+)M", sw)
    # `usable` is the metric this repo already reports for launch state (prefill_certifiability:
    # "usable=59.3%"), i.e. free + inactive. Raw free alone refuses windows that are actually fine
    # (measured 17:56: free=0.82 GiB with inactive=6.38 GiB); usable alone would accept a box that is
    # actively swapping. So both are gated, with the floor on raw free doing the real work.
    return {"free_gib": round(free * page / 1e9, 2),
            "inactive_gib": round(inactive * page / 1e9, 2),
            "usable_gib": round((free + inactive) * page / 1e9, 2),
            "swap_mib": float(m.group(1)) if m else None}


def other_llama_procs() -> list[str]:
    """Anything from this repo's build/bin holding the model.

    Both binaries, not just the server: the process that starved an earlier attempt was another
    line's `llama-bench` (7.5 GB resident) while a server-only guard reported the box as free.
    """
    found = []
    for pat in ("build/bin/llama-server", "build/bin/llama-bench"):
        p = subprocess.run(["pgrep", "-fl", pat], capture_output=True, text=True)
        found += [l for l in (p.stdout or "").splitlines() if l.strip()]
    return found


def latest_log() -> str:
    d = os.path.join(ROOT, "Backup/cgc_logs")
    files = [os.path.join(d, f) for f in os.listdir(d)
             if f.startswith("llama_server_2026") and f.endswith(".log")]
    return max(files, key=os.path.getmtime) if files else ""


def detach(cmd: list[str], env: dict, launch_log: str) -> None:
    """os.fork twice + os.setsid. NOT `setsid(1)`: macOS does not ship it."""
    pid = os.fork()
    if pid != 0:
        os.waitpid(pid, 0)
        return
    os.setsid()
    pid2 = os.fork()
    if pid2 != 0:
        os._exit(0)
    with open(launch_log, "wb") as f:
        subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    os._exit(0)


def wait_healthy(timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=5) as r:
                if json.load(r).get("status") == "ok":
                    return True
        except Exception:  # noqa: BLE001 - not up yet
            pass
        time.sleep(5)
    return False


def stop_server() -> None:
    """TERM, never KILL: the engine's own reminder is that kill -9 leaks Metal buffers."""
    subprocess.run(["pkill", "-TERM", "-f", "build/bin/llama-server"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        if subprocess.run(["pgrep", "-f", "build/bin/llama-server"],
                          stdout=subprocess.DEVNULL).returncode != 0:
            return
        time.sleep(2)


def probe(text: str, n_predict: int = 8, timeout: float = 1200) -> dict:
    body = json.dumps({"prompt": text, "n_predict": n_predict, "temperature": 0.0,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(f"{BASE}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception as e:  # noqa: BLE001 - report, never mask
        return {"error": f"{type(e).__name__}: {e}", "wall_s": round(time.time() - t0, 1)}
    tt = r.get("timings", {})
    return {"wall_s": round(time.time() - t0, 1),
            "prompt_n": tt.get("prompt_n"),
            "prefill_tps": round(tt.get("prompt_per_second") or 0.0, 2),
            "decode_tps": round(tt.get("predicted_per_second") or 0.0, 2)}


def mechanism_evidence(text: str, long_n: int | None) -> dict:
    """What the SERVER LOG says about which phase served the long prompt.

    This is v2's addition: a timing that arrives without this cannot be attributed. `slab` means the
    long prompt was served by the whole-layer PREFILL graph; `pool_chunks` means it was served by the
    DECODE graph (pool path) in n_batch-sized pieces, which is the clamp's behaviour.
    """
    banners = [l for l in text.splitlines()
               if "decode graph width" in l and "routable=0 slots" not in l]
    slab_ntok = sorted({int(x) for x in
                        re.findall(r"CGC-PREFILL-STREAM: il=\d+ kind=\d+ ntok=(\d+)", text)})
    return {
        "geometry_banner": banners[-1].strip() if banners else None,
        "slab_ntok": slab_ntok,
        "slab_served_long": (long_n in slab_ntok) if long_n else None,
        "hook_lines": len(re.findall(r"CGC-HOOK: ctx=", text)),
        "verify_strict": (re.findall(r"verify-strict: [^\n]*", text) or [None])[-1],
        "nil_count": text.count("buffer is nil"),
        "nan_count": len(re.findall(r"NAN|NaN|nan", text)),
        "assert_count": len(re.findall(r"GGML_ASSERT|Abort trap|ggml_abort", text)),
    }


def run_arm(arm: str, rep: int, out: str, launch_timeout: float) -> None:
    done = os.path.join(out, f"{arm}_rep{rep}.json")
    if os.path.exists(done):
        log(f"skip (present) {arm} rep{rep}")
        return
    before = mem_state()
    env = dict(os.environ)
    env["CGC_SERVER_PROFILE"] = "prefill250"
    env.update(ARMS[arm])
    launch_log = os.path.join(out, f"{arm}_rep{rep}.launch.log")
    log(f"{arm} rep{rep} launching (free={before['free_gib']}GiB swap={before['swap_mib']}MiB)")
    detach(["./scripts/run_server.sh", "--detach"], env, launch_log)
    rec: dict = {"arm": arm, "rep": rep, "before": before, "extra_env": ARMS[arm]}
    try:
        if not wait_healthy(launch_timeout):
            rec["error"] = "server never became healthy"
            log(f"{arm} rep{rep} NOT HEALTHY (see {os.path.basename(launch_log)})")
            return
        probe(SHORT_PROMPT, n_predict=1)                 # warm: first request pays cold faults
        rec["long"] = probe(LONG_PROMPT, n_predict=8)
        rec["short"] = probe(SHORT_PROMPT, n_predict=8)
        path = latest_log()
        rec["log"] = os.path.basename(path)
        rec["mechanism"] = mechanism_evidence(open(path, errors="replace").read(),
                                              rec["long"].get("prompt_n"))
        log(f"{arm} rep{rep} long={rec['long'].get('prefill_tps')} t/s "
            f"(n={rec['long'].get('prompt_n')}) short={rec['short'].get('prefill_tps')} t/s "
            f"served_by={'slab' if rec['mechanism']['slab_served_long'] else 'pool'}")
    finally:
        rec["after"] = mem_state()
        with open(done, "w") as f:
            json.dump(rec, f, indent=1)
        stop_server()


def aggregate(reps: int, out: str) -> dict:
    rows = {}
    for arm in ARMS:
        rows[arm] = {}
        for shape in ("long", "short"):
            vals, notes = [], []
            for rep in range(1, reps + 1):
                p = os.path.join(out, f"{arm}_rep{rep}.json")
                if not os.path.exists(p):
                    notes.append(f"rep{rep}:missing")
                    continue
                r = json.load(open(p))
                v = (r.get(shape) or {}).get("prefill_tps")
                if v:
                    vals.append(v)
                else:
                    notes.append(f"rep{rep}:{(r.get(shape) or {}).get('error', 'no-tps')}")
            rows[arm][shape] = {"per_launch": vals,
                                "median_discard_rep1": round(statistics.median(vals[1:]), 2)
                                if len(vals[1:]) > 1 else (vals[1] if len(vals) > 1 else None),
                                "notes": notes}
    print()
    print("=== prefill t/s per launch (launch 1 is the cold one) ===")
    for arm in ARMS:
        print(f"  {arm:<14} long {rows[arm]['long']['per_launch']}   "
              f"short {rows[arm]['short']['per_launch']}")
    print()
    print("=== medians over launches 2..N ===")
    for arm in ARMS:
        print(f"  {arm:<14} long={rows[arm]['long']['median_discard_rep1']:<8} "
              f"short={rows[arm]['short']['median_discard_rep1']}")
    print()
    print("=== how each arm was served (from the server log, per launch) ===")
    for arm in ARMS:
        for rep in range(1, reps + 1):
            p = os.path.join(out, f"{arm}_rep{rep}.json")
            if os.path.exists(p):
                r = json.load(open(p))
                m = r.get("mechanism") or {}
                print(f"  {arm:<14} rep{rep} slab_served_long={m.get('slab_served_long')} "
                      f"slab_ntok={m.get('slab_ntok')} hooks={m.get('hook_lines')} "
                      f"nil={m.get('nil_count')} nan={m.get('nan_count')} assert={m.get('assert_count')}")
    print()
    print("=== read this as ===")
    print("  pool8 vs pool17 (long) : if ~equal, the gap is GRAPH COUNT (a clamp penalty)")
    print("  slab vs pool17 (short) : matched graph count, so this is the slab itself")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"rows": rows, "arms": ARMS}, f, indent=1)
    log(f"summary -> {os.path.join(out, 'summary.json')}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out", default="/tmp/phase_split_ab")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved arms and exit")
    ap.add_argument("--min-free-gib", type=float, default=2.0,
                    help="raw free pages floor; refuses a box that is actively swapping")
    ap.add_argument("--min-usable-gib", type=float, default=8.0,
                    help="free+inactive floor, the metric this repo reports as launch state")
    ap.add_argument("--allow-busy-box", action="store_true")
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    args = ap.parse_args()

    if args.dry_run:
        for arm, extra in ARMS.items():
            env = dict(os.environ, CGC_SERVER_PROFILE="prefill250", **extra)
            p = subprocess.run(["./scripts/run_server.sh"], cwd=ROOT, env=dict(env, CGC_DUMP_ENV="1"),
                               capture_output=True, text=True)
            keep = [l for l in p.stdout.splitlines()
                    if l.startswith(("[arm]", "ENV CGC_PREFILL_STREAM", "ENV CGC_POOL_MAX_TOKENS"))]
            print(f"--- {arm}  extra_env={extra}")
            for l in keep:
                print(f"    {l}")
        return 0

    os.makedirs(args.out, exist_ok=True)
    if not args.aggregate_only:
        st, busy = mem_state(), other_llama_procs()
        log(f"preflight: free={st['free_gib']}GiB inactive={st['inactive_gib']}GiB "
            f"swap={st['swap_mib']}MiB, other llama processes={len(busy)}")
        if not args.allow_busy_box:
            if busy:
                log(f"REFUSING: {len(busy)} other llama process(es) hold the model "
                    f"(e.g. {busy[0].split(' ', 1)[0]}). No room for two on this box; the earlier "
                    "attempt here was SIGKILLed mid-prefill. Free it, or pass --allow-busy-box.")
                return 2
            if st["free_gib"] < args.min_free_gib:
                log(f"REFUSING: free={st['free_gib']}GiB < --min-free-gib={args.min_free_gib}.")
                return 2
            if st["usable_gib"] < args.min_usable_gib:
                log(f"REFUSING: usable={st['usable_gib']}GiB (free+inactive) < "
                    f"--min-usable-gib={args.min_usable_gib}. A 13 GB model plus an 8 GiB pool does "
                    "not fit in what is left; the resulting numbers would describe swap, not the "
                    "engine.")
                return 2
        for rep in range(1, args.reps + 1):
            for arm in ARMS:                      # interleaved
                run_arm(arm, rep, args.out, args.ready_timeout)
    aggregate(args.reps, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
