#!/usr/bin/env python3
"""A window is not certified by "thermal == NOMINAL and memory is free".

WHY THIS EXISTS. 2026-09-20, 18:0x: three arms of a decode A/B came back at 2.17 / 1.36 / 1.35 t/s
against 10.64 t/s for the SAME cell at 16:10 -- while `notifyutil -g ...thermalpressurelevel` said
NOMINAL for the whole run, `mem_state()` said 54% usable, and no other session was running. Both
conditions the existing gates check were satisfied and the box was ~5x slow. That window then
silently corrupted two rounds of measurement before anyone noticed (one of them mine).

The lesson is that "is the box healthy" is a PHYSICAL question, not a bookkeeping one. It has to be
answered by MEASURING THROUGHPUT against a known-shape reference, on the same box, immediately
before the run that matters.

WHAT IT MEASURES. The prefill cell of the frozen production profile (`-p 2048 -n 0`, `-b 5632`).
Prefill is chosen because it is the axis that is dominated by sustained matmul throughput, so it
tracks clock/power state tightly, and because it is cheap (one pass) and far less noisy than the
decode cell (whose single-arm spread is +/-27%). A healthy box reads ~276-292 t/s; see
`window_sentinel_ref.json` for the band and where each sample came from.

USAGE
    python3 scripts/check/window_sentinel.py                     # measure + verdict (exit 0/3)
    python3 scripts/check/window_sentinel.py --json /tmp/s.json
    python3 scripts/check/window_sentinel.py --dry-run           # print the command, launch nothing
    python3 scripts/check/window_sentinel.py --record --this-box-is-healthy
                                                                 # refresh the reference band

EXIT CODES  0 healthy (or dry-run) | 2 refused (another session / no window) | 3 DEGRADED
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys

ROOT = "/Users/alexchuang/Documents/flashkv-devserver"
HERE = os.path.join(ROOT, "scripts/check")
sys.path.insert(0, HERE)
os.chdir(ROOT)

import llama_bench_matrix as lbm                    # noqa: E402
from prefill_certifiability import mem_state        # noqa: E402
import thermal_pressure as tp                       # noqa: E402

REF_PATH = os.path.join(HERE, "window_sentinel_ref.json")
PROFILE = "prefill250"
SHAPE = ["-p", "2048", "-n", "0"]
BIN = os.path.join(ROOT, "src/llama.cpp/build/bin/llama-bench")
MODEL = os.path.join(ROOT, "models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf")
OTHER = "[l]lama|[p]rod_profile|[m]123_oracle|[s]erver_window|[d]ecode_sweep|[m]tp_accept"
# A healthy box is not "somewhere above a bar" -- it is within a band. 0.85 of the recorded median
# is the line: the recorded healthy samples span 275.65-300.43 (1.09x), so a reading 15% under the
# median cannot be a healthy sample of that same population, and the degraded window measured
# on 2026-09-20 was ~0.2x of it (i.e. nowhere near the boundary -- the line is not a knife edge).
DEFAULT_FRAC = 0.85


def full_command():
    env = {k: str(v) for k, v in lbm.resolve(PROFILE, {})["env"].items() if v not in (None, "")}
    cmd = [BIN, "-m", MODEL, "-ngl", "99", "-t", "8",
           "-expert-cache", env.get("CGC_EXPERT_CACHE_BYTES", "8589934592"),
           "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--load-mode", "none", "-o", "json",
           "-b", "5632", "-ub", "5632"] + SHAPE
    return cmd, env


PY_SCRIPTS = ("prod_profile", "m123_oracle", "server_window", "decode_sweep",
              "mtp_accept", "llama_bench_matrix", "profile_duo")


def _ps(pid, key):
    return subprocess.run(["ps", "-o", key + "=", "-p", pid],
                          capture_output=True, text=True).stdout.strip()


def others():
    """pids actually RUNNING a measured binary -- not shells whose argv merely quotes one.

    2026-09-20 20:1x: another line's post-run `bash -c '... pgrep -f llama-server ...' sleep 180`
    wrapper kept every gate red for three minutes after its server had already exited, because its
    argv contains the pattern as TEXT. A shell is not a measurement. The binaries (`llama-*`) and
    the python drivers are; anything else is a mention and must not block.
    Being wrong in the other direction is much cheaper: a false positive only means "did not run".
    """
    # `ps` is DENIED inside the agent sandbox (PermissionError: [Errno 1] Operation not
    # permitted) while `pgrep` is allowed -- decode_step_profile.py:128 hit the same wall.
    # The first version called `ps -o comm=` per pid, which is fine while pgrep matches
    # nothing (the loop body never runs) and crashes the gate the moment it does. A gate
    # that raises instead of answering is worse than no gate: it aborts the caller's run
    # and looks like "the machine is busy" when it is the tool that broke.
    # `pgrep -fl` already prints "<pid> <full command>", so comm is the basename of its
    # first token and the python check can read the whole line -- no `ps` needed.
    out = subprocess.run(["pgrep", "-fl", OTHER], capture_output=True, text=True).stdout
    blocking = []
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        pid, cmd = parts[0], parts[1]
        tok = cmd.split()
        comm = os.path.basename(tok[0]) if tok else ""
        if comm.startswith("llama"):
            blocking.append(pid)
        elif comm.startswith("python"):
            if any(s in cmd for s in PY_SCRIPTS):
                blocking.append(pid)
    return blocking


SWAP_MAX_FRAC = 0.85      # swap total MOVES on this box (5120/7168/8192 seen) -> never a fixed MiB
USABLE_MIN_GIB = 8.0      # free + inactive + speculative
HOG_REPORT_GIB = 1.0      # at or above this: reported
HOG_REJECT_GIB = 4.0      # at or above this: refused


def _swapusage():
    """(used_mib, total_mib) -- `sysctl -n vm.swapusage` prints K/M/G, so never assume M."""
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout

    def val(key):
        m = re.search(key + r"\s*=\s*([\d.]+)\s*([KMG])?", out)
        if not m:
            return 0.0
        v = float(m.group(1))
        return v * {"K": 1 / 1024.0, "M": 1.0, "G": 1024.0}[(m.group(2) or "M").upper()]

    return val("used"), val("total")


def mem_state():
    """(usable_gib, swap_used_mib, swap_total_mib). Page size is 16384 on this box, not 4096."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(r"page size of (\d+) bytes", out)
    pg = int(m.group(1)) if m else 4096
    pages = {}
    for line in out.splitlines():
        m2 = re.match(r"^(.+?):\s+(\d+)\.$", line.strip())
        if m2:
            pages[m2.group(1)] = int(m2.group(2))
    usable = sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative"))
    used, total = _swapusage()
    return usable * pg / (1024 ** 3), used, total


def desktop_hogs(min_gib=HOG_REPORT_GIB):
    """[(gib, pid, name)] heaviest first. WorkBuddy Helper (~2.8 GiB) is us -- do not kill it."""
    rows = []
    try:
        out = subprocess.run(["ps", "-axo", "pid=,rss=,comm="],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return rows                      # gate must answer, not raise
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            kb = int(parts[1])
        except ValueError:
            continue
        gib = kb * 1024.0 / (1024 ** 3)
        if gib >= min_gib:
            rows.append((gib, int(parts[0]), os.path.basename(parts[2])))
    rows.sort(reverse=True)
    return rows


def mem_gate(usable_min_gib=USABLE_MIN_GIB, swap_max_frac=SWAP_MAX_FRAC):
    """(ok, why, detail). Refuse on low usable, near-full swap, or one app >= 4 GiB.

    Why the swap rule is a FRACTION: 2026-09-22 the old fixed 6144 MiB threshold stopped
    refusing anything when macOS shrank swap total to 5120 MiB. And `purge` does NOT reduce
    swap_used (measured: free 0.07 -> 7.3 GB while swap stayed 6490-6770 MiB), so the gate
    must never ask for "swap == 0"; "swap not nearly full" is the achievable condition.
    """
    usable, used, total = mem_state()
    frac = (used / total) if total else 0.0
    hogs = desktop_hogs()
    det = {"usable_gib": round(usable, 2), "swap_used_mib": int(used),
           "swap_total_mib": int(total), "swap_frac": round(frac, 3),
           "hogs": [{"gib": round(g, 2), "pid": p, "name": n} for g, p, n in hogs[:8]]}
    why = []
    if usable < usable_min_gib:
        why.append("usable %.2f GiB < %.1f" % (usable, usable_min_gib))
    if frac > swap_max_frac:
        why.append("swap %.0f%% > %.0f%% (%d/%d MiB)" % (frac * 100, swap_max_frac * 100, used, total))
    if hogs and hogs[0][0] >= HOG_REJECT_GIB:
        why.append("%s %.2f GiB >= %.1f" % (hogs[0][2], hogs[0][0], HOG_REJECT_GIB))
    return (not why), ("; ".join(why) if why else "ok"), det


def mem_report():
    usable, used, total = mem_state()
    print("usable %.2f GiB | swap %.0f/%.0f MiB (%.0f%%)" % (
        usable, used, total, (used / total * 100) if total else 0.0))
    for g, p, n in desktop_hogs():
        print("  %6.2f GiB  pid %-7d %s" % (g, p, n))


def purge():
    """purge needs root and the agent shell has no tty -> go through the macOS auth dialog."""
    r = subprocess.run(["osascript", "-e",
                        'do shell script "/usr/sbin/purge" with administrator privileges'],
                       capture_output=True, text=True)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def measure():
    cmd, env = full_command()
    e = dict(os.environ)
    e.update(env)
    p = subprocess.run(cmd, env=e, capture_output=True, text=True)
    ts = None
    try:
        arr = json.loads(p.stdout[p.stdout.index("["):p.stdout.rindex("]") + 1])
        for row in arr:
            if int(row.get("n_gen", 0)) == 0:
                ts = float(row.get("avg_ts"))
    except Exception:
        pass
    return ts, p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--frac", type=float, default=DEFAULT_FRAC)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--this-box-is-healthy", action="store_true",
                    help="required with --record: the reference must come from a box you have "
                         "independently certified, or the sentinel will certify the degradation")
    ap.add_argument("--mem-gate", action="store_true",
                    help="memory/swap/desktop gate only; exit 0 ok, 2 refused")
    ap.add_argument("--mem-report", action="store_true", help="print usable/swap and the >= 1 GiB apps")
    ap.add_argument("--purge", action="store_true",
                    help="run /usr/sbin/purge through the macOS auth dialog (root; no tty needed)")
    ap.add_argument("--swap-max-frac", type=float, default=SWAP_MAX_FRAC)
    ap.add_argument("--usable-min-gib", type=float, default=USABLE_MIN_GIB)
    ap.add_argument("--min-usable-gib", type=float, default=None, metavar="GIB",
                    help="lower the memory precondition for THIS reading and say so. The default "
                         "(%.1f GiB) is right when the question is 'is this window free', but it "
                         "is bookkeeping, and this tool's whole reason to exist is that "
                         "bookkeeping cannot answer the PHYSICAL question. On a 16 GB box with a "
                         "desktop the steady state reads ~6.9 GiB usable, so the physical readout "
                         "never runs at all -- measured 2026-09-23: six decode arms spanned 1.79x "
                         "(5.84-10.46 t/s) with bit-identical outputs while this gate stayed red and "
                         "the launcher admitted every one. The reading is then reported with "
                         "mem_gate=overridden so it cannot be mistaken for an unqualified pass."
                         % USABLE_MIN_GIB)
    args = ap.parse_args()
    if args.min_usable_gib is not None:
        args.usable_min_gib = args.min_usable_gib

    if args.mem_report:
        mem_report()
        return 0
    if args.purge:
        rc, out, err = purge()
        print("purge rc=%d %s%s" % (rc, out, (" " + err) if err else ""))
        mem_report()
        return rc
    if args.mem_gate:
        ok, why, det = mem_gate(args.usable_min_gib, args.swap_max_frac)
        print(json.dumps({"ok": ok, "why": why, **det}, indent=1))
        if not ok:
            print("\nrefused. close the apps named above (>= 1 GiB), then re-run "
                  "`(sudo purge)` / `--purge`, then re-run this gate.")
        return 0 if ok else 2

    cmd, _ = full_command()
    print("sentinel shape: -p 2048 -n 0 -b 5632 (prefill cell of profile %s)" % PROFILE)
    print("argv:", " ".join(cmd))
    if args.dry_run:
        return 0

    ps = others()
    if ps:
        print("refused: another session is running (pids %s)" % ",".join(ps))
        return 2

    t = tp.stamp()
    usable_gib, swap_used, swap_total = mem_state()   # mem_state() is a 3-tuple, not a dict
    print("thermal %s | usable %.2f GiB | swap %d/%d MiB"
          % (t.get("label"), usable_gib, swap_used, swap_total))
    if usable_gib < args.usable_min_gib:
        print("refused: usable %.2f GiB < %.2f GiB -- the sentinel would certify a degraded box"
              % (usable_gib, args.usable_min_gib))
        return 2
    mem_gate = "pass" if args.min_usable_gib is None else "overridden"
    ts, rc = measure()
    print("measured %.2f t/s (rc=%s, mem_gate=%s)" % (ts or -1.0, rc, mem_gate))

    if args.record:
        if not args.this_box_is_healthy:
            print("refused: --record needs --this-box-is-healthy. A reference recorded on a degraded "
                  "box would certify the degradation instead of catching it.")
            return 2
        if not ts:
            print("refused: no reading to record")
            return 2
        ref = {"shape": " ".join(SHAPE), "batch": 5632, "profile": PROFILE,
               "median": ts, "samples": [ts], "frac": args.frac,
               "note": "recorded %s on a box the operator certified healthy" % t.get("t")}
        with open(REF_PATH, "w") as f:
            json.dump(ref, f, ensure_ascii=False, indent=2)
        print("reference -> %s (median %.2f)" % (REF_PATH, ts))
        return 0

    if not os.path.exists(REF_PATH):
        print("DEGRADED? no reference band at %s -- run --record on a certified-healthy box" % REF_PATH)
        return 3
    ref = json.load(open(REF_PATH))
    floor = ref["median"] * args.frac
    ok = ts is not None and ts >= floor
    verdict = "HEALTHY" if ok else "DEGRADED"
    print("\nreference median %.2f (%s)  floor %.2f (%.0f%%)  ->  %s"
          % (ref["median"], ref.get("note", "recorded"), floor, args.frac * 100, verdict))
    if not ok:
        print("  a window like this is NOT a measurement window: the existing gates (thermal key, "
              "usable memory) both pass on it -- measured 2026-09-20 18:1x")
    if args.json:
        # `m` was never assigned here: `--json` raised NameError at the last step, so the one tool
        # that measures a window could not write the artifact that cites it. Same class as
        # `harness.py show`'s missing `sw.decision()` -- a tool that raises instead of answering.
        mem = {"usable_gib": round(usable_gib, 2), "swap_used_mib": swap_used,
               "swap_total_mib": swap_total, "mem_gate": mem_gate,
               "min_usable_gib": args.usable_min_gib}
        json.dump({"t_s": ts, "ref_median": ref["median"], "floor": floor, "verdict": verdict,
                   "thermal": t, "mem": mem, "mem_gate": mem_gate}, open(args.json, "w"),
                  ensure_ascii=False, indent=2)
        print("json ->", args.json)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
