#!/usr/bin/env python3
"""Concurrent-request throughput inside ONE server process.

The question this answers (task 5, docs/NEXT_ACTIONS_2026-09-23.md): two servers cannot run at
once on this box (the launcher refuses any second llama process, and one arm is already
oversubscribed). So the "0.79 x 2 = 1.58" premise is dead in the shape it was written. The
remaining way to buy the same thing -- total tokens per wall second rather than per-stream
latency -- is N concurrent requests inside one process, which needs `-np N` (the launcher's
CGC_SERVER_CONCURRENCY). That cell had never been measured.

The probe is deliberately server-free (like decode_bench.py): it only speaks HTTP, so it needs
no window gate and can be pointed at any live server. The driver that launches the two `-np`
arms is Backup/phase_decomp/t5_intra_np.py.

Definitions, because two of them are easy to confuse and only one answers the question:

  per-request t/s   the server's own `predicted_n / predicted_ms` for ONE request. This is what
                    every number in this repo's docs means by "t/s", and it is the only one
                    comparable with a decode_bench.py reading.
  aggregate t/s     total tokens delivered by the whole batch / wall time of the batch. THIS is
                    "total efficiency". It counts the N requests as one workload. It is NOT
                    comparable with a per-request t/s: on this box a ~35 s request spends ~21 s
                    of that in prompt processing (prod25 does not use a big chunk), so the same
                    run reads 7.4 t/s per-request and 3.0 t/s aggregate. Both are reported; a
                    reader who mixes them gets a 2.5x error, which is why `non_gen_s` below is
                    always printed next to the aggregate.
  sum_server t/s    sum of per-request t/s. An upper reference for the aggregate: it excludes the
                    HTTP, prefill and slot-queue edges, so aggregate <= sum_server always, and the
                    gap is the part of the wall the batch does not spend generating.

Both arms send the SAME number of requests and the same prompt shape as the 4 GiB prod25
baseline, so `serial` is comparable with the 8.30 t/s that baseline recorded.

Usage:
    python3 scripts/check/concurrency_probe.py --np 2 --reps 3 --json /tmp/conc_np2.json
    python3 scripts/check/concurrency_probe.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_bench as db          # noqa: E402  one request shape, one thermal stamp
import thermal_pressure as tp      # noqa: E402


def one_request(url, n_predict, prompt, timeout=1800):
    """One request, with its wall clock and the thermal level on both sides.

    `ask`/`extract` are imported rather than re-implemented: the baseline these numbers will be
    compared against was taken with exactly that body and that temperature.
    """
    th_before = tp.stamp()
    t0 = time.time()
    d, _ = db.ask(url, prompt, n_predict, timeout=timeout)
    t1 = time.time()
    m = db.extract(d)
    m["wall_s"] = round(t1 - t0, 3)
    m["t_start"] = t0
    m["t_end"] = t1
    m["thermal_before"] = th_before
    m["thermal_after"] = tp.stamp()
    m["md5"] = hashlib.md5(m.get("text", "").encode()).hexdigest()[:8]
    return m


def arm(url, n_predict, prompt, concurrency, timeout=1800):
    """Run `concurrency` requests, sequentially or simultaneously, and reduce the batch.

    Returns the batch's rows plus the three throughput definitions above. The batch is scored
    only if every request produced tokens; a partial batch would silently change the denominator
    and the aggregate would still look plausible.
    """
    rows = [None] * concurrency
    if concurrency == 1:
        rows[0] = one_request(url, n_predict, prompt, timeout)
    else:
        barrier = threading.Barrier(concurrency)

        def worker(i):
            # Barrier, not just "start N threads": thread startup jitter of a few ms would be
            # attributed to the server as a shorter overlap window and inflate the aggregate.
            barrier.wait()
            rows[i] = one_request(url, n_predict, prompt, timeout)

        ts = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

    ok = [r for r in rows if r and r.get("predicted_n")]
    n_tokens = sum(r["predicted_n"] for r in ok)
    sum_server = sum(r["decode_tps"] for r in ok)

    if len(ok) != concurrency:
        # Do not print a number for a batch that did not complete. "N-1 of N returned" has
        # a different denominator than N and the printout would not say so.
        return {"arm": "pair" if concurrency > 1 else "serial", "n_requests": concurrency,
                "complete": False, "usable": len(ok), "rows": rows}

    if concurrency == 1:
        wall = ok[0]["wall_s"]
    else:
        wall = max(r["t_end"] for r in ok) - min(r["t_start"] for r in ok)
    # How much of the batch wall was NOT generation. Keeping prompt_n/prompt_ms here is the
    # difference between "concurrency is slow" and "this request shape is mostly prefill": the
    # 2026-09-23 np2 arm read 0.90x aggregate while each request's own decode was 2.4-8.1 t/s, and
    # only these fields say which phase absorbed the time.
    # An absent timing is not a zero: if any request did not report predicted_ms the split is
    # unreadable, and reporting non_gen_pct=100.0 would be an invented number.
    have_gen = all(r.get("predicted_ms") for r in ok)
    gen_s = max(r["predicted_ms"] for r in ok) / 1000.0 if have_gen else None
    pre = [r["prefill_tps"] for r in ok if r.get("prefill_tps")]
    return {
        "arm": "pair" if concurrency > 1 else "serial",
        "n_requests": concurrency,
        "complete": True,
        "n_tokens": n_tokens,
        "wall_s": round(wall, 3),
        "agg_wall_tps": round(n_tokens / wall, 3) if wall > 0 else 0.0,
        "sum_server_tps": round(sum_server, 3),
        "gen_s": round(gen_s, 2) if gen_s is not None else None,
        "non_gen_s": round(wall - gen_s, 2) if gen_s is not None else None,
        "non_gen_pct": (round(100 * (wall - gen_s) / wall, 1)
                        if gen_s is not None and wall > 0 else None),
        "per_req_tps_median": round(st.median([r["decode_tps"] for r in ok]), 3),
        "per_req_tps_min": round(min(r["decode_tps"] for r in ok), 3),
        "per_req_prompt_tps_median": round(st.median(pre), 2) if pre else None,
        "thermal_before": [
            {"level": r["thermal_before"]["level"], "label": r["thermal_before"]["label"]} for r in ok],
        "thermal_after_worst": tp.worst([r["thermal_after"] for r in ok]),
        "rows": [{k: r.get(k) for k in ("decode_tps", "prefill_tps", "prompt_n", "prompt_ms",
                                        "predicted_n", "predicted_ms", "wall_s", "md5")}
                 for r in ok],
    }


def verdict(batches, baseline_tps=None):
    """The decision the task asks for, stated only when the contrast exists.

    Two readers: whether the pair arm beat the serial arm on the SAME server (a within-run
    contrast, so launch drift cancels), and -- if a baseline is supplied -- whether the serial
    arm reproduces the 4 GiB prod25 baseline (the null cell: `-np 2` must not change the
    single-stream path).
    """
    ser = [b for b in batches if b["arm"] == "serial" and b.get("complete")]
    par = [b for b in batches if b["arm"] == "pair" and b.get("complete")]
    out = {"n_serial": len(ser), "n_pair": len(par), "verdict": "NO CONTRAST"}
    if ser and par:
        s = st.median([b["agg_wall_tps"] for b in ser])
        p = st.median([b["agg_wall_tps"] for b in par])
        out.update(serial_agg_tps=round(s, 3), pair_agg_tps=round(p, 3),
                   pair_over_serial=round(p / s, 3) if s else None,
                   # Aggregating N requests saves the N-1 serial latencies. If the engine ran
                   # them one after another the pair arm would land at 1.00x, not 2.00x -- so
                   # the two candidate answers are "~1.0 (slots are not parallel)" and
                   # ">1.0 (batched)". Anything at or below 1.0 means -np bought nothing.
                   verdict=("PARALLEL BUYS THROUGHPUT" if p > 1.05 * s else
                            "NO GAIN (slots serialise)" if p < 0.95 * s else
                            "AT THE BOUNDARY"),
                   serial_per_req_tps=round(st.median([b["per_req_tps_median"] for b in ser]), 3),
                   pair_per_req_tps=round(st.median([b["per_req_tps_median"] for b in par]), 3),
                   non_gen_pct=(round(st.median([b["non_gen_pct"] for b in par]), 1)
                                if all(b.get("non_gen_pct") is not None for b in par) else None))
    if baseline_tps and ser:
        # Against the BASELINE the comparable quantity is the per-request decode rate: the
        # baseline (decode_bench.py on the same shape) is the server's own predicted_n/predicted_ms,
        # not a wall aggregate. Comparing walls to it would report a -65% that is entirely the
        # prompt-processing share, not a regression.
        r = st.median([b["per_req_tps_median"] for b in ser])
        out["null_cell"] = {"serial_per_req_tps": round(r, 3), "baseline_per_req_tps": baseline_tps,
                            "delta_pct": round(100 * (r / baseline_tps - 1), 1)}
    return out


def selftest():
    """The reduction and the verdict, on synthetic batches whose answer is known by hand.

    A tool that measures "did N requests overlap" has one job: not to call a serialised pair
    parallel. That is exactly what the last case here pins.
    """
    cases, bad = [], 0

    def chk(name, got, want, tol=0.01):
        nonlocal bad
        ok = (got == want) if not isinstance(want, float) else abs(got - want) <= tol
        cases.append((name, ok, got, want))
        if not ok:
            bad += 1

    # two requests, 10 tokens each, fully overlapping over 2 s -> 10 agg t/s, sum 10
    rows = [{"decode_tps": 5.0, "predicted_n": 10, "predicted_ms": 2000.0, "wall_s": 2.05,
             "t_start": 100.0, "t_end": 102.05, "md5": "a", "thermal_before": tp.stamp(),
             "thermal_after": tp.stamp()},
            {"decode_tps": 5.0, "predicted_n": 10, "predicted_ms": 2000.0, "wall_s": 2.10,
             "t_start": 100.0, "t_end": 102.10, "md5": "b", "thermal_before": tp.stamp(),
             "thermal_after": tp.stamp()}]
    wall = max(r["t_end"] for r in rows) - min(r["t_start"] for r in rows)
    chk("overlap window = max(end)-min(start)", round(wall, 3), 2.10, 0.001)
    chk("aggregate = 20 tokens / 2.10 s", round(20 / wall, 3), 9.524, 0.01)

    # the same two requests run back to back -> the aggregate must equal per-stream
    def batch(a, agg, per_req, ng=None):
        return {"arm": a, "complete": True, "agg_wall_tps": agg,
                "per_req_tps_median": per_req, "non_gen_pct": ng}

    ser = batch("serial", 4.82, 4.82, 0.0)
    par = batch("pair", 4.82, 4.82, 0.0)
    chk("serialised pair is not a win", verdict([ser, par])["verdict"], "AT THE BOUNDARY")
    par2 = batch("pair", 8.10, 4.82, 0.0)
    chk("overlapping pair is a win", verdict([ser, par2])["verdict"], "PARALLEL BUYS THROUGHPUT")
    par3 = batch("pair", 3.00, 2.00, 0.0)
    chk("pair slower than serial is reported as such", verdict([ser, par3])["verdict"],
        "NO GAIN (slots serialise)")
    chk("no pair arm -> no verdict", verdict([ser])["verdict"], "NO CONTRAST")

    # an incomplete batch must not be scored at all
    inc = {"arm": "pair", "complete": False, "n_requests": 2, "usable": 1}
    chk("incomplete batch has no aggregate", "agg_wall_tps" in inc, False)

    # the null cell reads the PER-REQUEST rate, never the wall aggregate. Using the aggregate
    # against a decode_bench baseline is a live 2.5x error: on 2026-09-23 the same np1-b run
    # read 2.91 aggregate and 7.04 per-request against a baseline of 8.30.
    ser1 = batch("serial", 2.91, 7.04, 60.0)
    par1 = batch("pair", 3.04, 7.04, 60.0)
    nc = verdict([ser1, par1], baseline_tps=8.30)["null_cell"]
    chk("null cell uses per-request t/s", nc["serial_per_req_tps"], 7.04, 0.001)
    chk("null cell delta is -15.2% not -65%", nc["delta_pct"], -15.2, 0.1)
    chk("non_gen_pct surfaced with the verdict", verdict([ser1, par1])["non_gen_pct"], 60.0, 0.01)

    # an unreadable split must not be summarised as a number: a pair arm with no timings at all
    # cannot produce non_gen_pct, and the verdict has to say so rather than print 0.0%.
    ser2 = batch("serial", 2.9, 7.0, None)
    par2 = batch("pair", 3.0, 7.0, None)
    chk("unknown split -> None, not 0.0", verdict([ser2, par2])["non_gen_pct"], None)

    for name, ok, got, want in cases:
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: got={got} want={want}")
    print(f"selftest: {len(cases) - bad}/{len(cases)} passed")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--np", type=int, default=2,
                    help="label only: the server's -np, recorded in the product")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=160, help="decode_bench's own default")
    ap.add_argument("--prompt", default=db.SHORT_PROMPT)
    ap.add_argument("--baseline-tps", type=float, default=None,
                    help="the 4 GiB prod25 baseline; enables the null cell (single-stream check)")
    ap.add_argument("--json", default="")
    ap.add_argument("--reanalyze", default="",
                    help="re-derive the verdict of an existing product from its own batches")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.reanalyze:
        # A derived field computed under a definition that later changed must not be quotable from
        # the product. This recomputes `verdict` from the batches that are already in the file --
        # no measurement is touched -- so the 2026-09-23 arms taken before the null-cell fix stop
        # carrying a -65% that was an instrument artifact.
        d = json.load(open(args.reanalyze))
        scored = [b for b in d["batches"] if b.get("complete") and b.get("rep", 0) > 0]
        d["verdict"] = verdict(scored, args.baseline_tps)
        d["verdict_reanalyzed"] = True
        json.dump(d, open(args.reanalyze, "w"), ensure_ascii=False, indent=2)
        print(json.dumps(d["verdict"], ensure_ascii=False, indent=2))
        return 0

    batches = []
    for r in range(args.reps):
        # Alternate which arm goes first: on this box the ordinal position in a launch series is
        # itself a covariate (measured repeatedly), so a fixed order would put a constant bias on
        # one arm. Dropping rep 0 entirely is the other half of that defence.
        order = ["serial", "pair"] if r % 2 == 0 else ["pair", "serial"]
        for a in order:
            sys.stderr.write(f"  rep {r} arm {a} ...\n")
            sys.stderr.flush()
            b = arm(args.url, args.n_predict, args.prompt,
                    1 if a == "serial" else args.concurrency)
            b["rep"] = r
            batches.append(b)
            if b.get("complete"):
                ng = b.get("non_gen_pct")
                print(f"  [rep {r} {a}] agg {b['agg_wall_tps']:6.2f} t/s  "
                      f"per-req {b['per_req_tps_median']:6.2f}  "
                      f"sum_server {b['sum_server_tps']:6.2f}  wall {b['wall_s']:.1f}s  "
                      f"non-gen {('%5.1f%%' % ng) if ng is not None else '  n/a'}  "
                      f"thermal {b['thermal_before'][0]['label']}->{b['thermal_after_worst']['label']}",
                      flush=True)
            else:
                print(f"  [rep {r} {a}] INCOMPLETE: {b.get('usable')}/{b.get('n_requests')} requests "
                      f"returned tokens -- not scored", flush=True)

    scored = [b for b in batches if b.get("complete") and b.get("rep", 0) > 0]
    v = verdict(scored, args.baseline_tps)
    result = {"engine_probe": "concurrency_probe", "np_label": args.np,
              "concurrency": args.concurrency, "reps": args.reps, "n_predict": args.n_predict,
              "url": args.url, "scored_reps": len({b["rep"] for b in scored}),
              "batches": batches, "verdict": v}
    print()
    print(json.dumps(v, ensure_ascii=False, indent=2))
    if args.json:
        json.dump(result, open(args.json, "w"), ensure_ascii=False, indent=2)
        print("saved ->", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
