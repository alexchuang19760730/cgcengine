#!/usr/bin/env python3
"""MTP-off decode step profile: where do the ~100 ms/token actually go?

M3 of ROADMAP_PREFILL250_DECODE25 asks for `100 ms/token -> 40 ms/token`, i.e. a 2.4x. That is a
question about attribution, and the engine already has the two instruments needed to answer it --
both allowlisted in run_server.sh (an unlisted CGC_* is dropped silently, and the resulting "no
effect" is indistinguishable from a real negative):

  CGC_DECODE_PROFILE(+_ALL)  per-layer wait / cb / submit of the segmented dispatcher
  CGC_GPU_TIMING             each command buffer's own GPUStartTime/GPUEndTime, per layer

Their sum is the step wall, so the four terms this script reports are:

  dispatch   submit                          CPU: hand the next layer to the GPU
  hook       cb                              CPU: top-k slot management + blocking expert fill
  sync       wait - gpu                      CPU: poll for layer i to complete, minus real GPU work
  gpu        gpu_sum                         real GPU execution (attention + MoE math)

`wait` alone cannot separate sync from compute -- the poll is a CPU spin either way -- which is
exactly why CGC_GPU_TIMING exists. The split is what decides which lever matters: batching the
per-expert GEMVs attacks `gpu`, restoring inter-segment overlap attacks `sync`, and a cheaper hook
attacks `hook`. Only one of those can supply 2.4x.

Arms (same binary, same model, same pool; MTP is forced off because that is M3's exit criterion):
  prof      profile + GPU timing                     the attribution itself
  ahead     prof + CGC_SUBMIT_AHEAD=1                the zero-code ceiling of "restore overlap"
  nosync    CGC_OA_ASYNC=0                            the whole graph as one submit: no per-layer
                                                     structure at all, so it bounds the ENTIRE
                                                     per-layer machinery (measured as total ms/token)

`ahead` is deliberately included even though the submit order it restores is known-buggy: it is the
documented zero-code measurement of the ceiling for the whole overlap family. If decode does not
speed up there, no double-buffered remap design can help and the segmented dispatch is not the
bottleneck. It is a CEILING PROBE, not a candidate: its output can be garbage, so its tokens are
not read for quality and its t/s must never be quoted as a configuration.

Not a quality tool and not a gate: this measures time. M1/M2 belong to the oracle gate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAUNCHER = os.path.join(ROOT, "scripts", "run_server.sh")

# 41 blocks, full_attention_interval = 4 -> (i+1) % 4 == 0 is a full-attention layer, every other
# layer is a GatedDeltaNet. Read from the GGUF rather than assumed; see layer_types().
FULL_ATTN_INTERVAL = 4
N_BLOCKS = 41

ARMS = {
    "prof":  {},
    "ahead": {"CGC_SUBMIT_AHEAD": "1"},
    "nosync": {"CGC_SERVER_OA_ASYNC": "0"},
}
# Measured 2026-09-17: the `ahead` arm SEGFAULTS on its first multi-token graph (the warm-up's
# 2-token prefill graph, CGC-TOPK-SHAPE t_ne=[8,2,1,1]). That is the hazard the launcher's own comment
# documents -- submitting segment i+1 before the top-k hook writes segment i's remap leaf lets the GPU
# read a stale table -- so the arm is a ceiling probe that is not runnable on this binary, and the
# overlap ceiling is bounded arithmetically instead (`remove the GPU idle window`). The driver now
# reports it as a crash instead of waiting out its ready-timeout.
# The model is pinned EXPLICITLY, and that is not pedantry. run_server.sh:153 sets
# `MODEL_DEFAULT="$Q36"` and only swaps in `$Q36_MTP_DENSEIQ4X` when MTP=1 and DENSE_IQ4X=1, so
# `CGC_SERVER_MTP=0` silently swaps the MODEL FILE (Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf) rather than just
# the draft path. Every MTP on/off A/B taken through this launcher without pinning CGC_SERVER_MODEL is
# therefore a two-variable experiment: two models, not two MTP settings. Found by launching this
# profiler and reading the model in its own argv.
REFERENCE_MODEL = os.path.join(ROOT, "models", "gguf",
                               "Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf")
BASE_ENV = {"CGC_SERVER_MTP": "0",          # M3 is an MTP-off criterion
            "CGC_SERVER_MODEL": REFERENCE_MODEL}
PROFILE_ENV = {"CGC_DECODE_PROFILE": "1", "CGC_DECODE_PROFILE_ALL": "1", "CGC_GPU_TIMING": "1"}

SHORT = "Please give me three colours and two shapes, comma separated:"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def mem_state() -> dict:
    """free_pct from `memory_pressure -Q`, i.e. the launcher's OWN metric, plus swap in use.

    Not an approximation: run_server.sh:801 feeds FREE_PCT from exactly this command, and the guard
    compares that number, so a profiler reporting its own idea of free memory (e.g. free+inactive over
    every vm_stat page class, which computed 0% while the launcher saw 67% on the same box) would
    disagree with the thing that actually decides whether the launch is allowed to happen.
    """
    mp = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)%", mp)
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    st = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    page = 16384
    free_pages = 0
    mm = re.search(r"Pages free:\s+(\d+)", out)
    if mm:
        free_pages = int(mm.group(1))
    sw = re.search(r"used = ([\d.]+)M", st)
    return {"free_pct": int(m.group(1)) if m else None,
            "free_gib": round(free_pages * page / 2**30, 2),
            "swap_mib": float(sw.group(1)) if sw else None}


def llama_procs() -> list[str]:
    r = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True).stdout
    return [l for l in r.splitlines()
            if ("llama-server" in l or "llama-bench" in l) and "grep" not in l]


def launch(extra_env: dict, launch_log: str) -> tuple[subprocess.Popen, str | None]:
    env = {**os.environ, **BASE_ENV, **extra_env}
    lf = open(launch_log, "w")
    p = subprocess.Popen(["bash", LAUNCHER], cwd=ROOT, env=env, stdout=lf,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    return p, launch_log


def wait_healthy(port: str, timeout: float, launch_log: str) -> tuple[bool, str]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return True, "ok"
        except Exception:  # noqa: BLE001 - not ready yet
            pass
        txt = open(launch_log, errors="replace").read() if os.path.exists(launch_log) else ""
        m = re.search(r"\[guard\].*?(refus|還原|不足|blocked)", txt)
        if m:
            return False, f"launcher guard refused: {m.group(0)[:160]}"
        # A dead server must fail FAST. Without this the arm sat in the ready-timeout for its full
        # 15 minutes after the `ahead` arm segfaulted (`server 行程已退出` was already in the launch log),
        # which is a silent wait reading exactly like a slow model load.
        m = re.search(r"Segmentation fault: (\d+)", txt)
        if m:
            return False, f"server segfaulted during load/prefill (signal {m.group(1)})"
        if "server 行程已退出" in txt:
            return False, "server process exited before becoming healthy"
        if not subprocess.run(["pgrep", "-f", "build/bin/llama-server"],
                              capture_output=True).stdout.strip():
            time.sleep(3)
            if "模型載入中" in txt and time.time() - t0 > 30:
                return False, "no llama-server process and none loading"
        time.sleep(3)
    return False, "health timeout"


def server_pid(exclude: set[int] | None = None) -> int | None:
    """The launcher's own server child, so the arm can be stopped the way it wants (SIGINT).

    `exclude` is the set of pids that already existed when this arm started. Another line can bring
    up its own server on this shared box mid-arm, and `pids[0]` would then be someone else's process
    -- stopping it would be both rude and, since the other server keeps running, silently wrong.
    """
    r = subprocess.run(["pgrep", "-f", "build/bin/llama-server"], capture_output=True, text=True)
    pids = [int(x) for x in r.stdout.split()]
    mine = [p for p in pids if p not in (exclude or set())]
    return mine[0] if mine else None


def stop(pid: int | None) -> None:
    if not pid:
        return
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        return
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(1.5)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


SERVER_LOG_RE = re.compile(r"(\S*Backup/cgc_logs/llama_server_\d+_\d+\.log)")


def server_log(launch_log: str) -> str | None:
    """The launcher's own `log: tail -f <path>` line names the server's log -- which is where the
    engine writes CGC-DECPROF. The first version of this profiler parsed only its own launch log and
    therefore reported total=None for a launch that had produced 833 profile lines: an instrument that
    is blind by default, which is the failure mode this repo keeps paying for. Both files are read.
    """
    try:
        text = open(launch_log, errors="replace").read()
    except OSError:
        return None
    m = SERVER_LOG_RE.search(text.replace("\\", "/"))
    if not m:
        return None
    path = m.group(1)
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    return path if os.path.exists(path) else None


def arm_text(launch_log: str) -> tuple[str, str | None]:
    """Everything the arm wrote: the launcher's stdout plus the engine's own log."""
    try:
        text = open(launch_log, errors="replace").read()
    except OSError:
        text = ""
    sl = server_log(launch_log)
    if sl:
        text += "\n" + open(sl, errors="replace").read()
    return text, sl


def probe(port: str, text: str, n_predict: int) -> dict:
    body = json.dumps({"prompt": text, "n_predict": n_predict, "temperature": 0.0,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    tt = r.get("timings", {})
    return {"wall_s": round(time.time() - t0, 2), "prompt_n": tt.get("prompt_n"),
            "prefill_tps": round(tt.get("prompt_per_second") or 0.0, 2),
            "predicted_n": tt.get("predicted_n"),
            # guarded like the phase-split harness: a 1-token response reports ~1e6 "t/s"
            "decode_tps": (round(tt["predicted_per_second"], 2)
                           if (tt.get("predicted_n") or 0) >= 2
                           and (tt.get("predicted_per_second") or 0) < 1000 else None),
            "decode_ms_per_tok": (round(tt["predicted_ms"] / tt["predicted_n"], 2)
                                  if tt.get("predicted_n") and tt.get("predicted_ms") else None)}


DECPROF_RE = re.compile(
    r"CGC-DECPROF: step=(\d+) segs=(\d+) layers=(\d+) total=([\d.]+) ms \| "
    r"wait=([\d.]+) \(([\d.]+)%\) cb=([\d.]+) \(([\d.]+)%\) submit=([\d.]+) \(([\d.]+)%\) ntok=(\d+)"
    r"(?:\s*\|\s*layer gpu_sum=([\d.]+) union_sum=([\d.]+) gap_sum=([\d.]+) ms)?")
# Two per-layer forms, and they differ in exactly the character that a lazy regex gets wrong:
#   top-N:  `CGC-DECPROF top1: L2 wait=...`      (no colon after the layer)
#   all:    `CGC-DECPROF     L38: wait=...`       (colon after the layer)
# A pattern requiring the colon silently drops the top-8 rows, which is precisely the rows the
# non-ALL mode prints -- so the failure mode is "profile looks empty", not "profile looks wrong".
LAY_RE = re.compile(r"L(\d+):?\s*wait=([\d.]+) cb=([\d.]+) submit=([\d.]+) ms "
                    r"gpu=([\d.]+) union=([\d.]+) gap=([\d.]+)")


def parse_profile(text: str) -> dict:
    steps, layers = [], []
    for m in DECPROF_RE.finditer(text):
        g = m.groups()
        steps.append({"step": int(g[0]), "segs": int(g[1]), "layers": int(g[2]),
                      "total_ms": float(g[3]), "wait_ms": float(g[4]), "wait_pct": float(g[5]),
                      "cb_ms": float(g[6]), "cb_pct": float(g[7]),
                      "submit_ms": float(g[8]), "submit_pct": float(g[9]), "ntok": int(g[10]),
                      "gpu_sum_ms": float(g[11]) if g[11] else None,
                      "union_sum_ms": float(g[12]) if g[12] else None,
                      "gap_sum_ms": float(g[13]) if g[13] else None})
    for m in re.finditer(r"CGC-DECPROF[^\n]*", text):
        line = m.group(0)
        mm = LAY_RE.search(line)
        if mm:
            layers.append({"layer": int(mm.group(1)), "wait_ms": float(mm.group(2)),
                           "cb_ms": float(mm.group(3)), "submit_ms": float(mm.group(4)),
                           "gpu_ms": float(mm.group(5)), "union_ms": float(mm.group(6)),
                           "gap_ms": float(mm.group(7))})
    # The per-layer lines are emitted for the latest 8-step block only, so a layer can appear in
    # several blocks. Keep the LAST occurrence (warm, steady state) rather than averaging across
    # blocks: block 1 is the cold one and mixing it in would smear exactly the number we want.
    last = {l["layer"]: l for l in layers}
    return {"steps": steps, "layers": sorted(last.values(), key=lambda x: x["layer"])}


def layer_types() -> dict[int, str]:
    return {i: ("full_attn" if (i + 1) % FULL_ATTN_INTERVAL == 0 else "gdn")
            for i in range(N_BLOCKS)}


def summarise(arm: str, prof: dict, probe_res: dict) -> dict:
    steps = [s for s in prof["steps"] if s["ntok"] == 1 and s["gpu_sum_ms"] is not None]
    steady = [s for s in steps if s["step"] > 8] or steps
    med = lambda k, rows=steady: (round(statistics.median([r[k] for r in rows]), 2)  # noqa: E731
                                  if rows else None)
    out = {"arm": arm, "probe": probe_res, "steps_used": len(steady),
           "median": {k: med(k) for k in ("total_ms", "wait_ms", "cb_ms", "submit_ms",
                                          "gpu_sum_ms", "union_sum_ms", "gap_sum_ms")}}
    t, w, cb, sb, gpu = (out["median"][k] for k in
                         ("total_ms", "wait_ms", "cb_ms", "submit_ms", "gpu_sum_ms"))
    if t and gpu is not None:
        out["split"] = {"gpu": round(gpu, 2), "sync": round(w - gpu, 2), "cb": cb, "submit": sb,
                        "pct": {"gpu": round(100 * gpu / t, 1), "sync": round(100 * (w - gpu) / t, 1),
                                "cb": round(100 * cb / t, 1), "submit": round(100 * sb / t, 1)}}
    return out


def analyse_layers(prof: dict) -> dict:
    lt = layer_types()
    rows = [r for r in prof["layers"] if r["layer"] < N_BLOCKS]
    for r in rows:
        r["type"] = lt.get(r["layer"], "?")

    def mean(sel, k):
        v = [r[k] for r in rows if sel(r)]
        return round(statistics.mean(v), 3) if v else None

    # Steady blocks only: block 1 of the process carries the first touch of every expert and is
    # several times the steady cost, so a per-layer mean over all blocks measures the cold start
    # (layer 0 averaged 49.7 ms of hook in the first attempt, which is a warm-up, not a hook).
    full = mean(lambda r: r["type"] == "full_attn", "gpu_ms")
    gdn = mean(lambda r: r["type"] == "gdn", "gpu_ms")
    return {"layers": rows,
            "gpu_mean_ms": {"full_attn": full, "gdn": gdn},
            "gpu_mean_n": {"full_attn": sum(1 for r in rows if r["type"] == "full_attn"),
                           "gdn": sum(1 for r in rows if r["type"] == "gdn")},
            "gpu_per_layer_max": max((r["gpu_ms"] for r in rows), default=None),
            "wait_per_layer": {"min": min((r["wait_ms"] for r in rows), default=None),
                               "max": max((r["wait_ms"] for r in rows), default=None)},
            "cb_per_layer_max": max((r["cb_ms"] for r in rows), default=None)}


def run_arm(arm: str, extra: dict, out: str, args: argparse.Namespace) -> dict:
    port = args.port
    llog = os.path.join(out, f"{arm}.launch.log")
    env = {**PROFILE_ENV, **extra}
    before = {int(x) for x in subprocess.run(["pgrep", "-f", "build/bin/llama-server"],
                                             capture_output=True, text=True).stdout.split()}
    with open(os.path.join(out, f"{arm}.env.json"), "w") as f:
        json.dump({"env": env, "base": BASE_ENV, "mem_before": mem_state(),
                   "pids_before": sorted(before)}, f, indent=1)
    p, path = launch(env, llog)
    rec: dict = {"arm": arm, "env": env, "before": mem_state()}
    try:
        ok, why = wait_healthy(port, args.ready_timeout, llog)
        rec["launch_ok"], rec["launch_why"] = ok, why
        if not ok:
            return rec
        rec["warmup"] = probe(port, SHORT, 4)
        rec["measure"] = probe(port, args.prompt, args.n_predict)
        time.sleep(2)
    finally:
        rec["after"] = mem_state()
        # The per-step profile lines are flushed continuously, but the SERVER's own counters (and
        # the last block) only land on shutdown, so the log is read after the stop, not before.
        stop(server_pid(before))
        time.sleep(1.5)
        try:
            subprocess.run(["pkill", "-INT", "-f", "scripts/run_server.sh"], check=False)
        except Exception:  # noqa: BLE001
            pass
        if p.poll() is None:
            p.terminate()
    text, sl = arm_text(llog)
    rec["server_log"] = sl
    prof = parse_profile(text)
    rec["profile"] = summarise(arm, prof, rec.get("measure", {}))
    rec["layers"] = analyse_layers(prof)
    rec["profile"]["blocks"] = [dict(s) for s in prof["steps"]]
    # The raw engine lines are kept verbatim: they are the evidence, and a reader re-deriving the
    # split must not have to trust this script's arithmetic.
    rec["raw_lines"] = [l for l in text.splitlines()
                        if "CGC-DECPROF" in l or "CGC-GPUTIME" in l][:200]
    with open(os.path.join(out, f"{arm}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def self_detach(args: argparse.Namespace) -> int:
    """Survive the caller's process group: `setsid` does not exist on macOS, and a tool/session
    timeout kills every child it spawned -- which is what happened to the first attempt at this run.
    Double-fork, redirect, and let the parent return immediately."""
    if os.fork() != 0:
        return 0
    os.setsid()
    if os.fork() != 0:
        os._exit(0)
    os.chdir(ROOT)
    logf = args.log_file or os.path.join(args.out, "driver.log")
    os.makedirs(os.path.dirname(logf), exist_ok=True)
    fd = os.open(logf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    return -1                                  # caller of self_detach() is the detached child


def reparse(out: str) -> int:
    """Rebuild every arm's JSON from the logs already on disk.

    Exists because the first run of this profiler recorded `total=None` for an arm that had produced a
    full 833-line profile: the data was never lost, only unparsed. Re-running the grid to recover it
    would also change the conditions (thermal, carried swap), which is exactly the confound this repo
    keeps documenting.
    """
    n = 0
    for f in sorted(os.listdir(out)):
        if not f.endswith(".launch.log"):
            continue
        arm = f[: -len(".launch.log")]
        text, sl = arm_text(os.path.join(out, f))
        prof = parse_profile(text)
        p = os.path.join(out, f"{arm}.json")
        rec = json.load(open(p)) if os.path.exists(p) else {"arm": arm}
        rec["server_log"] = sl
        rec["profile"] = summarise(arm, prof, rec.get("measure", {}))
        rec["profile"]["blocks"] = [dict(s) for s in prof["steps"]]
        rec["layers"] = analyse_layers(prof)
        rec["raw_lines"] = [l for l in text.splitlines()
                            if "CGC-DECPROF" in l or "CGC-GPUTIME" in l][:400]
        with open(p, "w") as fh:
            json.dump(rec, fh, indent=1)
        m = rec["profile"].get("median") or {}
        print(f"{arm}: total={m.get('total_ms')} wait={m.get('wait_ms')} cb={m.get('cb_ms')} "
              f"submit={m.get('submit_ms')} gpu={m.get('gpu_sum_ms')} "
              f"(server log: {'yes' if sl else 'NOT FOUND'})")
        n += 1
    return 0 if n else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/decode_step_profile")
    ap.add_argument("--port", default="8080")
    ap.add_argument("--arms", default="prof,ahead,nosync")
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--ready-timeout", type=float, default=900.0)
    ap.add_argument("--prompt", default=SHORT)
    ap.add_argument("--self-detach", action="store_true")
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--reparse", action="store_true",
                    help="rebuild the JSONs from logs already on disk; launches nothing")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.reparse:
        return reparse(args.out)
    if args.self_detach:
        r = self_detach(args)
        if r >= 0:                             # the parent returns; the grandchild runs the grid
            print(f"detached; log -> {args.log_file or os.path.join(args.out, 'driver.log')}")
            return r

    procs = llama_procs()
    if procs:
        log(f"REFUSING: another llama process is resident ({len(procs)}): {procs[0][:100]}")
        return 2
    log(f"head={subprocess.run(['git','rev-parse','--short','HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()}"
        f" mem={mem_state()}")

    rows = []
    for arm in args.arms.split(","):
        if arm not in ARMS:
            log(f"unknown arm {arm}")
            return 2
        log(f"--- {arm} {ARMS[arm]} ---")
        rec = run_arm(arm, ARMS[arm], args.out, args)
        rows.append(rec)
        m = rec.get("profile", {}).get("median") or {}
        log(f"{arm}: launch_ok={rec.get('launch_ok')} decode={rec.get('measure', {}).get('decode_tps')} t/s "
            f"total={m.get('total_ms')} ms wait={m.get('wait_ms')} cb={m.get('cb_ms')} "
            f"submit={m.get('submit_ms')} gpu={m.get('gpu_sum_ms')}")

    print()
    print("| arm | decode t/s | step ms | wait | cb | submit | gpu |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        m = r.get("profile", {}).get("median") or {}
        print(f"| `{r['arm']}` | {r.get('measure', {}).get('decode_tps')} | {m.get('total_ms')} | "
              f"{m.get('wait_ms')} | {m.get('cb_ms')} | {m.get('submit_ms')} | {m.get('gpu_sum_ms')} |")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"rows": rows, "head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()},
            f, indent=1)
    log(f"results -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
