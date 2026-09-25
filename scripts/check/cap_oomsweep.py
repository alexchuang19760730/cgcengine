#!/usr/bin/env python3
"""cap_oomsweep — 找比 143 高但不 Jetsam-OOM 的 L4 pool capacity。

直接跑 llama-bench binary（env 直传、绕过 run_server allowlist），逐点二分：
子进程 returncode==0 => 活；returncode==-9 (SIGKILL) => Jetsam OOM。
每点记录 init 实际 cap、峰值 RSS、峰值 swap。
"""
import argparse, json, os, re, signal, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO, "src/llama.cpp/build/bin/llama-bench")
MODEL = os.path.join(REPO, "models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf")
STRIDE_MIB = 42.81

BASE_ENV = dict(
    CGC_L4_CAP_TOTALSTRIDE="1", CGC_SPAC="1", CGC_SERVER_DENSE_IQ4X="1",
    CGC_PREFILL_STREAM="1", CGC_MM_BITIDENT="1", CGC_OA_ASYNC="1",
)

def swap_mib():
    out = subprocess.run(["sysctl","vm.swapusage"],capture_output=True,text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else 0.0

def run_point(cap, gen, depth):
    budget_b = int(cap * STRIDE_MIB * 1024 * 1024)
    cmd = [BENCH, "-m", MODEL, "-ngl", "99", "--load-mode", "none", "-t", "8",
           "-expert-cache", str(budget_b), "--cache-type-k","q8_0","--cache-type-v","q8_0",
           "-b","5632","-ub","5632","-p","0","-n",str(gen),"-d",str(depth),"-r","1",
           "--warm-skip","0","-o","json"]
    env = dict(os.environ); env.update(BASE_ENV)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    peak_rss, peak_swap, real_cap = 0, swap_mib(), None
    while p.poll() is None:
        try:
            ps = subprocess.run(["ps","-o","rss=","-p",str(p.pid)],capture_output=True,text=True)
            rss = int(ps.stdout.strip())//1024 if ps.stdout.strip() else 0
            peak_rss = max(peak_rss, rss)
        except Exception: pass
        peak_swap = max(peak_swap, swap_mib())
        time.sleep(1.5)
    out, err = p.communicate()
    m = re.search(r"L4 metal pool capacity=(\d+) slots/layer", err.decode(errors="ignore"))
    if m: real_cap = int(m.group(1))
    oom = p.returncode == -signal.SIGKILL
    return dict(target=cap, real_cap=real_cap, oom=oom, rc=p.returncode,
                peak_rss_mib=peak_rss, peak_swap_mib=round(peak_swap,1),
                wall_s=round(time.time()-t0,1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="189,175,165,155")
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--depth", type=int, default=256)
    ap.add_argument("--json", default="/tmp/cap_oomsweep.json")
    a = ap.parse_args()
    points = [int(x) for x in a.points.split(",")]
    results, best = [], None
    for c in points:
        print(f"--- cap {c} ---", flush=True)
        r = run_point(c, a.gen, a.depth)
        results.append(r)
        print("   ", {k:r[k] for k in ("real_cap","oom","rc","peak_rss_mib","peak_swap_mib","wall_s")}, flush=True)
        if not r["oom"] and r["rc"]==0:
            best = r["real_cap"]; break   # 降序、第一個活點即上界下的最高已測點
    json.dump(dict(best=best, results=results), open(a.json,"w"), indent=1)
    print("BEST =", best)

if __name__ == "__main__":
    main()
