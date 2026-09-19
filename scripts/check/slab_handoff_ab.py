#!/usr/bin/env python3
"""Slab→pool handoff A/B: can prefill 250 and a warm decode hold at the same time?

Why this shape
--------------
The 2026-09-17 rotated run (docs/M1_WORKITEM2_PHASE_SPLIT_STATUS_2026-09-17.md) established that the
slab prefill harms decode (slab/pool per-rep medians 0.18-0.54 on three of three reps) and named the
mechanism from the counters: the slab arm's decode-side misses were all compulsory with capacity=0,
i.e. the pool was FULL but not of the experts decode demanded -- because the slab path repoints the
FFN weights at a per-layer slab and never writes the pool.

`prewarm_hot` is the mechanism for exactly that set, but `hot_prewarm_done` is a PROCESS-level flag,
so in a server it is consumed by whichever request decodes first and every later prefill→decode
transition gets no publish. CGC_SLAB_HANDOFF=<cap> adds a per-transition, capped, evicting publish
(llama_expert_cache_prewarm_hot_capped).

So the question this driver answers, per arm:
  1. what did the prefill cost (t/s) -- the handoff must not eat the 250,
  2. what did decode cost (t/s) on request 1 AND request 2 -- request 2 is the one a process-level
     one-shot cannot serve, which is the whole point,
  3. what did the handoff itself cost (the CGC-SLAB-HANDOFF line), and what were the decode-side
     misses (teardown: compulsory vs capacity).

Usage: python3 scripts/check/slab_handoff_ab.py [--arms off,on,off2] [--cap 32] [--port 8123]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "run_server.sh"
MODEL = ROOT / "models" / "gguf" / "Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf"
BASE_ENV = {"CGC_SERVER_MTP": "0", "CGC_SERVER_MODEL": str(MODEL),
            "CGC_SERVER_PROFILE": "prefill250"}       # slab armed: STREAM=1 + SLAB_CAP=256 + ub 6144
PROMPT_UNIT = ("Expert caching trades disk reads for memory residency. In a mixture-of-experts model "
               "only a few experts are routed per token, so the working set is a small fraction of "
               "the weights, and the policy that decides which experts stay resident is what the "
               "decode rate actually depends on. ")
TASK = re.compile(r"task (\d+) \| prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens"
                  r".*?([\d.]+) tokens per second", re.S)
TASK_DEC = re.compile(r"task (\d+) \|        eval time =\s*([\d.]+) ms /\s*(\d+) tokens"
                      r".*?([\d.]+) tokens per second", re.S)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def mem():
    st = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    co = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    f = re.search(r"free percentage:\s*(\d+)%", co)
    s = re.search(r"used = ([\d.]+)M", st)
    return {"free_pct": int(f.group(1)) if f else None,
            "swap_mib": float(s.group(1)) if s else None}


def kill_all():
    for pat in ("decode_step_profile", "llama.server", "slab_handoff_ab"):
        subprocess.run(["pkill", "-f", pat], capture_output=True)
    time.sleep(3)


def launch(env_extra, port, launch_log):
    env = {**os.environ, **BASE_ENV, **env_extra, "CGC_SERVER_PORT": str(port)}
    # The shared box probe (docs/SERVER_WINDOW_LEDGER_2026-09-19.md): refuses a launch into a busy
    # box. Self-contained import; gated once per process across this script's arms.
    import importlib.util as _ilu, os as _os
    _spec = _ilu.spec_from_file_location(
        "server_window", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                       "server_window.py"))
    _sw = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_sw)
    _sw.require_first(port=int(port),
                      need_mb=float(_os.environ.get("CGC_WINDOW_NEED_MB", "8000")),
                      where="slab_handoff_ab")
    lf = open(launch_log, "w")
    p = subprocess.Popen(["bash", str(LAUNCHER)], cwd=str(ROOT), env=env, stdout=lf,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    return p


def wait_health(port, timeout, launch_log):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        txt = Path(launch_log).read_text(errors="replace") if Path(launch_log).exists() else ""
        if re.search(r"\[guard\].*?(拒跑|blocked|不足|refus)", txt):
            return False
        if "server 行程已退出" in txt or "Segmentation fault" in txt:
            return False
        time.sleep(3)
    return False


def ask(port, prompt, n_predict):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "n_predict": n_predict, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    txt = (out.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
    return round(time.time() - t0, 1), len(txt)


def server_log(launch_log):
    p = Path(launch_log)
    if not p.exists():
        return None
    m = re.findall(r"llama_server_(\d{8}_\d{6})\.log", p.read_text(errors="replace"))
    if not m:
        return None
    return ROOT / "Backup" / "cgc_logs" / f"llama_server_{sorted(set(m))[-1]}.log"


def arm_report(name, launch_log, wall):
    lg = server_log(launch_log)
    out = {"arm": name, "wall_s": wall, "log": str(lg) if lg else None}
    if lg is None or not lg.exists():
        out["error"] = "no server log"
        return out
    txt = lg.read_text(errors="replace")
    pre = {int(t): (float(pms), int(nt), float(tps)) for t, pms, nt, tps in TASK.findall(txt)}
    dec = {int(t): (float(ems), int(nt), float(tps)) for t, ems, nt, tps in TASK_DEC.findall(txt)}
    out["requests"] = [{"task": t,
                        "prompt_tokens": pre[t][1],
                        "prefill_tps": round(pre[t][2], 1),
                        "prefill_s": round(pre[t][0] / 1000.0, 1),
                        "decode_tokens": dec.get(t, (0, 0, 0))[1],
                        "decode_tps": round(dec.get(t, (0, 0, 0))[2], 2),
                        "decode_ms_per_tok": round(dec.get(t, (0, 0, 0))[0] / max(1, dec.get(t, (0, 0, 0))[1]), 1)}
                       for t in sorted(set(pre) & set(dec))]
    ho = re.findall(r"CGC-SLAB-HANDOFF: cap=(\d+) warmed=(\d+) experts in ([\d.]+) ms", txt)
    out["handoff"] = [{"cap": int(a), "warmed": int(b), "ms": float(c)} for a, b, c in ho]
    mi = re.search(r"miss attribution: compulsory=(\d+) capacity=(\d+)", txt)
    if mi:
        out["misses"] = {"compulsory": int(mi.group(1)), "capacity": int(mi.group(2))}
    hs = re.search(r"hits=(\d+)/(\d+) \(([\d.]+)%\)", txt)
    if hs:
        out["hits"] = {"hits": int(hs.group(1)), "requests": int(hs.group(2)), "rate": float(hs.group(3))}
    pr = re.search(r"resident=([\d.]+) MiB", txt)
    if pr:
        out["resident_mib"] = float(pr.group(1))
    sl = re.findall(r"CGC-PREFILL-STREAM: il=(\d+) kind=\d+ ntok=(\d+)", txt)
    if sl:
        out["slab_chunks"] = len(sl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="off,on,off2")
    ap.add_argument("--cap", type=int, default=32)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--prompt-repeat", type=int, default=90)
    ap.add_argument("--n-predict", type=int, default=24)
    ap.add_argument("--out", default="/tmp/slab_handoff")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    prompt = PROMPT_UNIT * a.prompt_repeat
    log(f"prompt chars={len(prompt)} (~{len(prompt)//4} tokens), n_predict={a.n_predict}")
    results = []
    for arm in a.arms.split(","):
        kill_all()
        env_extra = {"CGC_SLAB_HANDOFF": str(a.cap)} if arm == "on" else {}
        ll = f"{a.out}/{arm}.launch.log"
        log(f"--- arm {arm} env={env_extra} mem={mem()} ---")
        launch(env_extra, a.port, ll)
        ok = wait_health(a.port, 900, ll)
        log(f"    launch_ok={ok}")
        if not ok:
            results.append({"arm": arm, "error": "launch refused or timed out"})
            kill_all()
            continue
        for i in (1, 2):
            wall, nchar = ask(a.port, prompt, a.n_predict)
            log(f"    request {i}: wall={wall}s chars={nchar}")
        subprocess.run(["pkill", "-TERM", "-f", "llama.server"], capture_output=True)
        time.sleep(8)
        rep = arm_report(arm, ll, wall)
        results.append(rep)
        for r in rep.get("requests", []):
            log(f"    task {r['task']}: prefill {r['prompt_tokens']} tok @ {r['prefill_tps']} t/s"
                f", decode {r['decode_tokens']} tok @ {r['decode_tps']} t/s"
                f" ({r['decode_ms_per_tok']} ms/tok)")
        log(f"    handoff={rep.get('handoff')} misses={rep.get('misses')} hits={rep.get('hits')}"
            f" resident={rep.get('resident_mib')}")
    Path(f"{a.out}/result.json").write_text(json.dumps({"cap": a.cap, "arms": results}, indent=1))
    print(f"\nresult -> {a.out}/result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
