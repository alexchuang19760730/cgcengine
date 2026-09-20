#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
io_granularity_curve.py -- 量「一次 pread 讀多大」與「有效頻寬」的關係曲線。

WHY THIS EXISTS
    2026-09-21 的 `-np N` 實測（docs/PARALLEL_AND_WORKERS_AB_2026-09-21.md §3）出現一組
    只用 bytes 解釋不了的反常：N=1 → N=4 時 **每 step 讀的 bytes 少了 44×**（145 MB → 3.3 MB），
    步時卻從 167.6 ms 漲到 ~4600 ms。同時平均 job 大小從 **133 KiB 掉到 ~257 B**。

    反推出來的數字是：N=1 的有效頻寬 = 145 MB / 167.6 ms ≈ **0.87 GB/s**，
    而同一顆 SSD 的循序讀峰值約 7 GB/s ⇒ **只跑到 12%**。
    若這是真的，那 decode 步時裡最大的一項不是「讀多少」，是「每次讀多小」——
    而這是可以改的（合併讀），與「降低 miss」是完全不同的槓桿。

    本工具就是為了把「job 大小 → 有效頻寬」這條曲線量出來，讓「合併讀能拿多少」是算的
    而不是猜的。它**不起 server、不碰 GPU**，所以不需要窗口；只吃 IO。

WHAT IT MEASURES
    對一個真實大檔（預設 models/gguf/ 下最大的 .gguf）做隨機 offset 的 pread：
        size × threads × reps → (GB/s, µs/request, 每 request 的 bytes)
    `F_NOCACHE` 強制繞過 page cache，否則第二次讀會量到 RAM 而不是 SSD。

USAGE
    python3 scripts/check/io_granularity_curve.py --selftest
    python3 scripts/check/io_granularity_curve.py --run --threads 1 4
    python3 scripts/check/io_granularity_curve.py --run --sizes 4096 131072 1048576

OUTPUT 記到 Backup/phase_decomp/io_granularity_<ts>.json，附 `provenance` 標記是否受窗口保護
    （本工具不起 server ⇒ 不需要，但會記下 machine 上當下有沒有別人在跑量測，
      因為那會污染 IO 曲線）。

⚠️ 口徑邊界
    - pread（本工具）≠ 池的 fill 路徑（可能走 mmap / 自己切 chunk / 有 mutex）。
      本曲線給的是**裝置＋syscall 的上界**，不是池的現況。
    - 所以它只能回答「合併讀**理論上**有多少空間」，不能回答「池今天讀多大」。
      要回答後者要看池的 teardown 計數器（已有）。
"""

import argparse
import fcntl
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE = 48  # fcntl(fd, F_NOCACHE, 1) on macOS

DEFAULT_SIZES = [256, 4096, 65536, 131072, 524288, 1048576, 4194304]
OUT_DIR = "Backup/phase_decomp"


def biggest_gguf(root="."):
    best, best_sz = None, -1
    for dirpath, _dirnames, filenames in os.walk(os.path.join(root, "models")):
        for fn in filenames:
            if fn.endswith(".gguf"):
                p = os.path.join(dirpath, fn)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                if sz > best_sz:
                    best, best_sz = p, sz
    return best


def one_point(path, size, threads, n_req, seed=7):
    """Run `n_req` random preads of `size` bytes across `threads` threads. F_NOCACHE, O_RDONLY."""
    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    fsz = os.path.getsize(path)
    span = max(fsz - size - 1, 1)
    try:
        random.seed(seed)
        offsets = [random.randrange(0, span) for _ in range(n_req)]
        per = (n_req + threads - 1) // threads

        def work(k):
            lo, hi = k * per, min((k + 1) * per, n_req)
            got = 0
            for i in range(lo, hi):
                got += len(os.pread(fd, size, offsets[i]))
            return got

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as ex:
            got = sum(ex.map(work, range(threads)))
        dt = time.perf_counter() - t0
    finally:
        os.close(fd)
    return dict(size=size, threads=threads, n_req=n_req, bytes=got, seconds=dt,
                gbps=(got / 2 ** 30) / dt if dt > 0 else 0.0,
                us_per_req=(dt * 1e6 / n_req) if n_req else 0.0)


def choose_n(size, target_seconds=0.6):
    """Pick a request count so each point runs long enough to be above timer noise."""
    # rough: assume ~3 GB/s until proven otherwise, but floor/ceiling the count
    guess = max(3e9 * target_seconds / max(size, 1), 1)
    return int(min(max(guess, 32), 200000))


def others_running():
    """Who else is doing measurements right now. IO curves are polluted by them."""
    try:
        out = subprocess.run(["pgrep", "-fl", "llama-server|llama-bench|decode_sweep|prod_profile"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        return None
    return [ln for ln in out.splitlines() if ln.strip()] or []


def fit_two_term(points):
    """Fit  t = size/BW + tau  over the measured (size, us_per_req) points (threads=1 only).

    Returns (BW_GBps, tau_us, residual_note). This is the model that says
    'a job costs a fixed latency plus a transfer time at peak bandwidth'.
    If the measured curve does NOT look like that (e.g. tau grows with size, or the
    small-size end is far above the fit), residual_note says so -- that is the result,
    not an error.
    """
    pts = [(p["size"], p["us_per_req"]) for p in points
           if p.get("threads", 1) == 1 and p["us_per_req"] > 0 and p["size"] > 0]
    if len(pts) < 3:
        return (None, None, "need >=3 single-thread points, got %d" % len(pts))
    pts.sort()
    # least squares on y = a*x + b  (a = us/byte, b = tau_us)
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    den = n * sxx - sx * sx
    if den == 0:
        return (None, None, "degenerate")
    a = (n * sxy - sx * sy) / den          # us per byte
    b = (sy - a * sx) / n                  # fixed overhead, us
    bw = (1e6 / a) / 1e9 if a > 0 else None   # bytes/us -> GB/s
    # how badly the smallest point misses the fit
    x0, y0 = pts[0]
    pred0 = a * x0 + b
    note = "smallest point %dB: measured %.0f us vs fit %.0f us (%.2fx)" % (
        x0, y0, pred0, y0 / pred0 if pred0 else float("nan"))
    return (bw, b, note)


def cmd_run(a):
    path = a.file or biggest_gguf()
    if not path or not os.path.exists(path):
        print("no model file found; pass --file", file=sys.stderr)
        return 2
    sizes = a.sizes or DEFAULT_SIZES
    others = others_running()
    if others and not a.force:
        print("IO is busy -- an IO curve measured next to a live run is not a curve:\n  %s\n"
              "pass --force to measure anyway (the record will say it was contested)."
              % "\n  ".join(others[:6]), file=sys.stderr)
        return 3
    fsz = os.path.getsize(path)
    print("file  %s  (%.2f GiB)" % (path, fsz / 2 ** 30))
    print("contested: %s" % ("YES" if others else "no"))
    print()
    print("%10s %8s %10s %12s %12s" % ("size", "threads", "n_req", "GB/s", "us/request"))
    points = []
    for th in (a.threads or [1]):
        for sz in sizes:
            n = choose_n(sz, a.target_seconds)
            r = one_point(path, sz, th, n, seed=a.seed)
            r["file"] = path
            points.append(r)
            print("%10d %8d %10d %12.3f %12.1f" % (sz, th, n, r["gbps"], r["us_per_req"]))
    bw, tau, note = fit_two_term(points)
    print()
    if bw is None:
        print("two-term fit: %s" % note)
    else:
        print("two-term fit  t = size/BW + tau :  BW = %.2f GB/s   tau = %.1f us" % (bw, tau))
        print("  %s" % note)
    rec = dict(when=time.strftime("%Y-%m-%dT%H:%M:%S"), file=path, file_bytes=fsz,
               sizes=sizes, threads=a.threads or [1], target_seconds=a.target_seconds,
               contested=bool(others), points=points,
               fit=dict(bw_gbps=bw, tau_us=tau, note=note))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "io_granularity_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print("\nwrote %s" % out)
    return 0


def selftest():
    ok = []

    def expect(name, got, want):
        good = got == want
        ok.append(good)
        print("%-58s %s" % (name, "PASS" if good else "FAIL (got %r want %r)" % (got, want)))

    # choose_n: bigger requests -> fewer of them, always in range
    n_small, n_big = choose_n(4096), choose_n(4 << 20)
    expect("choose_n decreases with size", n_small > n_big, True)
    expect("choose_n floors at 32", choose_n(1 << 30) >= 32, True)
    expect("choose_n caps at 200000", choose_n(1) <= 200000, True)

    # fit_two_term on a synthetic curve that IS two-term: 100us + size/5GB/s
    synth = [dict(threads=1, size=s, us_per_req=100.0 + s / 5000.0)
             for s in (4096, 131072, 1048576, 4194304)]
    bw, tau, _ = fit_two_term(synth)
    expect("recovers BW from a synthetic two-term curve", round(bw), 5)
    expect("recovers tau from a synthetic two-term curve", round(tau), 100)

    # and it must NOT invent a fit from too few points
    bw2, tau2, note2 = fit_two_term(synth[:2])
    expect("refuses to fit 2 points", (bw2, tau2), (None, None))
    expect("says why", "need" in note2, True)

    # threads!=1 points must be excluded from the single-thread fit
    mixed = synth + [dict(threads=4, size=4096, us_per_req=1.0)]
    bw3, tau3, _ = fit_two_term(mixed)
    expect("4-thread points do not pollute the 1-thread fit", round(tau3), 100)

    # one_point: reads the bytes it claims to, and reports a sane rate
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(os.urandom(1 << 20))
        tmp = tf.name
    try:
        r = one_point(tmp, 65536, 1, 64)
        expect("one_point reads n_req * size bytes", r["bytes"], 64 * 65536)
        expect("one_point reports a positive rate", r["gbps"] > 0, True)
        expect("one_point reports us/request", r["us_per_req"] > 0, True)
        r2 = one_point(tmp, 65536, 4, 64)
        expect("threaded variant also reads everything", r2["bytes"], 64 * 65536)
    finally:
        os.unlink(tmp)

    print("\n%d/%d" % (sum(ok), len(ok)))
    return 0 if all(ok) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("run", help="measure the size -> bandwidth curve")
    p.add_argument("--file", help="large file to read (default: biggest models/**/*.gguf)")
    p.add_argument("--sizes", type=int, nargs="+")
    p.add_argument("--threads", type=int, nargs="+", default=[1])
    p.add_argument("--target-seconds", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--force", action="store_true", help="measure even if the box is busy")
    sub.add_parser("selftest", help="self-check this instrument")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "run":
        return cmd_run(a)
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
