#!/usr/bin/env python3
"""E0-E4 driver for the `s1-asyncgather` mindmap node (see docs/mindmap/briefs/s1-asyncgather.md §9).

Runs the acceptance suite the node declares, in the order the node argues for it, and writes
`results.json` INCREMENTALLY (a step that dies must not erase the earlier steps' reading):

  E4  delivery-cell pairing      : segmented arm vs S1 arm, same cell/pool, r3  -> bench json
  E2  per-layer window arithmetic : segmented arm (has the hook => per-layer misses/cb)
                                    + S1 arm (has the GPU timing)  -> per-layer stderr
  E0  warm-pool correctness       : oracle gate   seg(ref) -> S1+8GiB -> S1+tiny pool(contrast)

E1 and E3 are offline (no GPU) and are not run here -- they live in the node's own doc trail.
Everything is launched through the production entries (`harness bench`, `m123_oracle_gate`) so
the cell contract, the base gate and the swap-arm gate all apply.
"""
import json
import os
import re
import subprocess
import sys
import time

ROOT = "/Users/alexchuang/Documents/flashkv-devserver"
OUT = os.path.join(ROOT, "Backup/eseries")
RES = os.path.join(OUT, "results.json")
S1 = "CGC_SEG_BATCH=1;CGC_B_SCHEME=1;CGC_SLOT_TABLE_GPU=1"
S1_ARM = "prod-new:" + S1
# Every launch is preceded by this cooldown: the first E4 ran its two arms back to back, so the
# second started on a warmer box and the ratio could only be diagnostic (see the node).
COOL_S = float(os.environ.get("ESERIES_COOL_S", "420"))


def swap_mib():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(re.search(r"used = ([\d.]+)M", out).group(1))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load():
    try:
        with open(RES, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save(key, payload):
    d = load()
    d[key] = payload
    d.setdefault("_order", [])
    if key not in d["_order"]:
        d["_order"].append(key)
    os.makedirs(OUT, exist_ok=True)
    with open(RES, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=2)
    log(f"saved {key}")


def run(cmd, cwd=ROOT, timeout=2400):
    log("$ " + " ".join(cmd))
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    log(f"rc={p.returncode} wall={time.time() - t0:.0f}s")
    return p


def bench_step(key, arms, workdir, extra_args=()):
    d = os.path.join(OUT, key)
    os.makedirs(d, exist_ok=True)
    cmd = [sys.executable, os.path.join(ROOT, "scripts/check/harness.py"), "bench",
           "--prompt", "2048", "--gen", "128", "--depths", "512", "--reps", "3",
           "--ctx-size", "0", "--warm-skip", "64",
           "--workdir", d, "--json", os.path.join(d, "summary.json")]
    for a in arms:
        cmd += ["--arm", a]
    cmd += list(extra_args)
    p = run(cmd)
    with open(os.path.join(d, "driver.log"), "w", encoding="utf-8") as fh:
        fh.write(p.stdout + "\n=== stderr ===\n" + p.stderr)
    save(key, bench_payload(p.returncode, d, p.stdout))
    return p


def bench_payload(rc, d, stdout):
    try:
        with open(os.path.join(d, "summary.json"), encoding="utf-8") as fh:
            summary = json.load(fh)
    except Exception as e:
        log(f"no summary: {e}")
        summary = {}
    return {"rc": rc, "dir": d, "stdout_tail": stdout[-3000:],
            "arms": [{"tag": a.get("tag"), "rows": a.get("rows"),
                      "thermal": a.get("thermal", {}).get("worst"),
                      "memory": {k: v for k, v in (a.get("memory") or {}).items()
                                 if k in ("launch", "end", "worst")},
                      "attribution": a.get("attribution"),
                      "cell": a.get("cell"), "contract": a.get("contract")}
                     for a in summary] if isinstance(summary, list) else summary}


def bench_arms_cooled(key, arms, workdir):
    """One launch per arm, each after the cooldown, order recorded -- for pairings.

    `harness bench --arm A --arm B` launches B straight after A, so the two arms of E4 started
    from different box states. Splitting them gives each arm its own cooldown while keeping
    `summary.json` in the same shape (a list of arms) so the artifacts stay comparable.
    """
    os.makedirs(workdir, exist_ok=True)
    merged, runs = [], []
    for i, a in enumerate(arms):
        log(f"[cool] {COOL_S:.0f}s before {a}")
        time.sleep(COOL_S)
        sw0 = swap_mib()
        cmd = [sys.executable, os.path.join(ROOT, "scripts/check/harness.py"), "bench",
               "--prompt", "2048", "--gen", "128", "--depths", "512", "--reps", "3",
               "--ctx-size", "0", "--warm-skip", "64", "--arm", a,
               "--workdir", workdir, "--json", os.path.join(workdir, f"arm_{i}.json")]
        p = run(cmd)
        sw1 = swap_mib()
        jp = os.path.join(workdir, f"arm_{i}.json")
        if os.path.exists(jp):
            with open(jp, encoding="utf-8") as fh:
                rows = json.load(fh)
            merged += rows if isinstance(rows, list) else [rows]
        runs.append({"arm": a, "rc": p.returncode, "json": jp, "swap_before_mib": sw0,
                     "swap_after_mib": sw1, "swap_growth_mib": round(sw1 - sw0, 1)})
    with open(os.path.join(workdir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    with open(os.path.join(workdir, "runs.json"), "w", encoding="utf-8") as fh:
        json.dump({"launch_order": arms, "cool_s": COOL_S, "runs": runs}, fh, indent=2)
    save(key, bench_payload(max(r["rc"] for r in runs), workdir, ""))


def probe_of(stdout):
    """Persist the gate's probe readings -- they are E0's decisive evidence.

    The byte answer (`文摘文摘…` vs `42`) lived only in the console because the S1 arms exit
    before the gate writes a summary, so the one reading that proves the defect was in no
    artifact. The prompt hash travels with it so two runs' probes can be compared.
    """
    out = {}
    for line in stdout.splitlines():
        m = re.match(r"\s*(probe|answer|prompt_md5)\s*:\s*(.+)$", line)
        if m:
            out["probe_" + m.group(1)] = m.group(2).strip()
    return out


def per_layer_from(stderr_path):
    """Pull the CGC-DECPROF / CGC-GPUNODE / CGC-NSM per-layer lines out of a bench stderr log."""
    rows = {"decprof": [], "gpunode": [], "spac": [], "slab": [], "miss": []}
    with open(stderr_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "CGC-DECPROF" in line:
                rows["decprof"].append(line.rstrip())
            elif "CGC-GPUNODE" in line or "CGC-GPU-NODE" in line:
                rows["gpunode"].append(line.rstrip())
            elif "CGC-SPAC-MEM" in line:
                rows["spac"].append(line.rstrip())
            elif "slab fills" in line or "CGC-SLAB" in line:
                rows["slab"].append(line.rstrip())
            elif re.search(r"miss(es)?\b", line) and "CGC" in line:
                rows["miss"].append(line.rstrip())
    return {k: (v[:60] + (["...(%d more)" % (len(v) - 60)] if len(v) > 60 else []))
            for k, v in rows.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    steps = sys.argv[1:] or ["E4", "E2", "E0"]
    log(f"steps={steps}")

    if "E4" in steps:
        bench_arms_cooled("E4", ["prod-new", S1_ARM], os.path.join(OUT, "E4"))

    if "E2" in steps:
        # The segmented arm carries the per-layer hook readings (misses / cb split);
        # the S1 arm carries the device-side timing. The pair is what the window arithmetic needs.
        bench_step("E2_seg", ["prod-new:" + ";".join([
            S1, "CGC_DECODE_PROFILE=1", "CGC_DECODE_PROFILE_ALL=1", "CGC_GPU_TIMING=1",
            "CGC_HOOK_SPLIT=1",
            "LLAMA_EXPERT_CACHE_MISS_DUMP=" + os.path.join(OUT, "E2_misses.txt")])],
            os.path.join(OUT, "E2seg"))
        bench_step("E2_s1", [S1_ARM + ";" + ";".join([
            "CGC_DECODE_PROFILE=1", "CGC_DECODE_PROFILE_ALL=1", "CGC_GPU_TIMING=1"])],
            os.path.join(OUT, "E2s1"))
        for k in ("E2_seg", "E2_s1"):
            d = os.path.join(OUT, k)
            logs = [f for f in os.listdir(d) if f.endswith(".stderr.log")] if os.path.isdir(d) else []
            if logs:
                save(k + "_layers", {f: per_layer_from(os.path.join(d, f)) for f in logs})

    if "E0" in steps:
        gate = os.path.join(ROOT, "scripts/check/m123_oracle_gate.py")
        d = os.path.join(OUT, "E0")
        os.makedirs(d, exist_ok=True)
        seg_ref = os.path.join(d, "seg_ref.jsonl")
        p0 = run([sys.executable, gate, "--profile", "prod-new", "--tag", "E0-seg",
                  "--dump", os.path.join(d, "seg.jsonl"), "--write-ref", seg_ref,
                  "--ref-note", "E0 seg arm (production shape, no S1)"])
        save("E0_seg", dict({"rc": p0.returncode, "tail": (p0.stdout + p0.stderr)[-2500:]},
                             **probe_of(p0.stdout)))
        p1 = run([sys.executable, gate, "--profile", "prod-new", "--tag", "E0-s1",
                  "--ref", seg_ref, "--dump", os.path.join(d, "s1.jsonl"),
                  "--allow-incomparable",
                  "--env", "CGC_SEG_BATCH=1", "--env", "CGC_B_SCHEME=1",
                  "--env", "CGC_SLOT_TABLE_GPU=1"])
        save("E0_s1", dict({"rc": p1.returncode, "tail": (p1.stdout + p1.stderr)[-2500:]},
                            **probe_of(p1.stdout)))
        # contrast: the gate must be able to SEE a difference, else "pass" means nothing
        p2 = run([sys.executable, gate, "--profile", "prod-new", "--tag", "E0-s1-tiny",
                  "--ref", seg_ref, "--dump", os.path.join(d, "s1_tiny.jsonl"),
                  "--allow-incomparable",
                  "--env", "CGC_SEG_BATCH=1", "--env", "CGC_B_SCHEME=1",
                  "--env", "CGC_SLOT_TABLE_GPU=1",
                  "--env", "CGC_SERVER_EXPERT_CACHE_BYTES=1073741824"])
        save("E0_s1_tiny", dict({"rc": p2.returncode, "tail": (p2.stdout + p2.stderr)[-2500:]},
                                 **probe_of(p2.stdout)))

    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
