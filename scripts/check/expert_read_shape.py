#!/usr/bin/env python3
"""expert_read_shape.py -- the producer end of the expert-cache channel, measured on the file.

WHY THIS EXISTS
---------------
The engine prints its own read shape:

    llama_expert_cache: read shape: jobs=N bytes=B (0.04 MiB/job)  us/job=…  effective_rate=… MiB/s

That line cannot be quoted as a physical read size. `fill_job` (llama-expert-cache.cpp:33-45)
increments `n_reads` but NOT `n_read_bytes`; bytes are added only by the merged-iov branch
(:153) and by the pool worker (:3240/:3254). So its numerator comes from the pool path while
its denominator also counts the bg/prefetch single-segment reads -- and the more merging fails,
the smaller the printed MiB/job gets. The same line's `us/job` is bounded by the worker count:
a run that prints 23,755 us/job over 82,119 jobs implies 1,951 s of thread time inside a 76.6 s
wall, while 8 workers cap that at 613 s.

This tool measures the thing directly instead: same file, same offsets (taken from the GGUF's own
tensor table), the same byte counts the cache asks for, at three read units and two queue depths.

    python3 scripts/check/expert_read_shape.py --model models/gguf/<model>.gguf
    python3 scripts/check/expert_read_shape.py --selftest

WHAT IT ANSWERS
---------------
Whether a given read unit saturates the device, and therefore whether "the cache is slow" is a
bandwidth claim, a latency claim, or a *shape* claim. On this box (M4 Air, 16 GB) the current
unit (one (layer,expert,kind) blob, 0.34-0.56 MB) already reaches ~1.26 GB/s at depth 8, while
8 adjacent expert ids (3.44 MiB contiguous, which the file layout offers for free because the
expert dim is ne[2]) reach ~3.35 GB/s. So the unit is a second-order question; the first-order
one is whether the reads are allowed to run back-to-back at all (see
docs/EXPERT_CHANNEL_SHAPE_2026-09-23.md).

F_NOCACHE IS SET ON PURPOSE
---------------------------
A warm read would flatter the numbers, and would also change what the next engine launch sees.
The probe therefore neither reads nor pollutes the page cache. It issues ~40 MB of reads and
issues nothing else; it starts no server and leaves nothing running.
"""

import argparse
import fcntl
import os
import statistics
import sys
import threading
import time

F_NOCACHE = 48  # Darwin: fcntl(fd, F_NOCACHE, 1)
GGUF_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "src", "llama.cpp", "gguf-py")


def expert_geometry(model, layer=20, kind="down"):
    """(name, per_expert_bytes, data_offset, tensor_bytes) for one expert tensor of `kind`."""
    sys.path.insert(0, os.path.normpath(GGUF_PY))
    from gguf import GGUFReader  # noqa: PLC0415

    reader = GGUFReader(model)
    for t in reader.tensors:
        if f"blk.{layer}.ffn_{kind}_exps" in t.name:
            n_expert = int(t.shape[2]) if len(t.shape) > 2 else 1
            total = int(t.data.nbytes)
            return t.name, total // n_expert, int(t.data_offset), total
    raise SystemExit(f"no blk.{layer}.ffn_{kind}_exps tensor in {model}")


def measure(fd, off, size, depth, reps, span):
    """Mean seconds per read. `span` bounds the offsets used (so reads stay inside the tensor)."""
    def at(k):
        # a different offset per stream, still inside the tensor
        return off + (k * size) % max(size, span - size)

    if depth == 1:
        t0 = time.perf_counter()
        for _ in range(reps):
            os.pread(fd, size, at(0))
        return (time.perf_counter() - t0) / reps
    per = max(1, reps // depth)
    threads = [threading.Thread(target=lambda k=k: [os.pread(fd, size, at(k)) for _ in range(per)])
               for k in range(depth)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return (time.perf_counter() - t0) / (per * depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="GGUF to measure (the same file the cache reads)")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--kind", default="down", choices=("gate", "up", "down"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        # The invariant that makes the read unit what it is: the expert dim is a whole multiple
        # of a per-expert blob, so `entry.bytes` (loader:1823) is uniform across experts.
        path = os.environ.get("CGC_SHAPE_MODEL")
        if not path:
            print("selftest: CGC_SHAPE_MODEL unset -- nothing to check against "
                  "(the probe has no model-independent invariant to assert)")
            return 0
        name, per, _, total = expert_geometry(path, args.layer, args.kind)
        ok = (total % per == 0) and per > 0
        print(f"selftest: {name} per-expert={per:,}B total={total:,}B divisible={ok}")
        return 0 if ok else 1

    if not args.model:
        raise SystemExit("--model is required (or --selftest)")

    name, per, off, total = expert_geometry(args.model, args.layer, args.kind)
    print(f"{name}: B/expert={per:,}  tensor={total/2**20:.1f} MiB  data_off={off:,}")
    print(f"  whole (layer,kind) region = {total/2**20:.0f} MiB contiguous "
          f"(expert dim is ne[2], so adjacent ids are adjacent in the file)")
    print()

    fd = os.open(args.model, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    print(f"{'shape':44} {'per read':>10} {'MiB/s':>9} {'GB/s':>6}")
    for label, size, depth, reps in (
        ("1 expert   depth 1", per, 1, 200),
        ("1 expert   depth 8  (the 8 pool workers)", per, 8, 200),
        ("8 experts  depth 1  (fully adjacent union)", 8 * per, 1, 40),
        ("8 experts  depth 8", 8 * per, 8, 40),
        ("64 experts depth 8  (27 MiB per read)", 64 * per, 8, 8),
    ):
        if size > total:
            continue
        dt = measure(fd, off, size, depth, reps, total)
        mibs = (size / 2**20) / dt
        print(f"{label:44} {dt*1e6:9.0f}us {mibs:9.1f} {mibs/1024:6.2f}")
    os.close(fd)
    print()
    print("read independently: bandwidth is the read unit's, latency is the channel's.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
