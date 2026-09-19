#!/usr/bin/env python3
"""Per-layer `cb` (command-buffer cost) from a CGC-DECPROF log, segmented BY REQUEST.

WHY THIS EXISTS -- the two mistakes it exists to stop

  1. WRONG COLUMN. A per-layer line reads
         CGC-DECPROF all: L<k> wait=<w> cb=<c> submit=<s> ms gpu=.. union=.. gap=.. sg=.. n=..
     The FIRST number is `wait`, the SECOND is `cb`. A version of this analysis took group(2)
     and printed 2.0-3.4 ms as "cb" for every layer -- an order of magnitude too big, and it made
     every arm look like a regression at once. `cb` is group(3) of `wait= cb= submit=`.
     (No self-test can catch a wrong-column error for you; the check is that the numbers agree
     with what the memory notes quote for the same log.)

  2. POOLED ACROSS REQUESTS. Taking a median over the whole run hides per-request structure.
     Measured on Run E2 (2026-09-19): its third request collapsed (decode 2.816 t/s, every layer's
     cb moving from 0.14-0.27 ms to 0.9-1.3 ms), and the cross-request median reported "every layer
     got worse" for a configuration whose first two requests had ALL FORTY layers at the floor.
     A single degraded request is enough to invert the conclusion, so segment first, always.

WHAT A REQUEST IS HERE
  Per-layer rows belong to the most recent `CGC-DECPROF: step=.. ntok=..` header. A header with
  ntok >= --prefill-ntok starts a new request (decode steps carry ntok <= 4 with MTP; prefill
  steps carry ntok in the hundreds). `--drop-requests N` lets you skip warmup requests.

USAGE
  python3 scripts/check/decode_layer_cb.py Backup/cgc_logs/llama_server_*.log
  python3 scripts/check/decode_layer_cb.py LOG --drop-requests 1 --json /tmp/layer_cb.json
  python3 scripts/check/decode_layer_cb.py --selftest
"""

import argparse
import json
import re
import sys
from statistics import median

RE_HDR = re.compile(r"^CGC-DECPROF: step=(?P<step>\d+) segs=(?P<segs>\d+) layers=(?P<layers>\d+)\b"
                    r".*?\btotal=(?P<total>[\d.]+) ms.*?\bwait=(?P<wait>[\d.]+)\b.*?"
                    r"\bcb=(?P<cb>[\d.]+)\b.*?\bsubmit=(?P<submit>[\d.]+)\b.*?\bntok=(?P<ntok>\d+)")
RE_LAYER = re.compile(r"^CGC-DECPROF all: L(\d+) wait=([\d.]+) cb=([\d.]+) submit=([\d.]+)")

DEFAULT_PREFILL_NTOK = 100
DEFAULT_DECODE_NTOK = 4
DEFAULT_LAYERS = 40


def parse(text, prefill_ntok=DEFAULT_PREFILL_NTOK, decode_ntok=DEFAULT_DECODE_NTOK,
          layers=DEFAULT_LAYERS):
    """-> {"per_request": {req: {"layer_cb": {L: [..]}, "steps": [(total, cb, wait), ..]}},
           "n_headers": int, "n_layer_rows": int}"""
    req = 0
    cur_ntok = None
    in_prefill = False
    out = {"per_request": {}, "n_headers": 0, "n_layer_rows": 0}
    for line in text.splitlines():
        if line.startswith("CGC-DECPROF "):
            if line.startswith("CGC-DECPROF all: "):
                m = RE_LAYER.match(line)
                if m and cur_ntok is not None and cur_ntok <= decode_ntok:
                    slot = out["per_request"].setdefault(req, {"layer_cb": {}, "steps": []})
                    slot["layer_cb"].setdefault(int(m.group(1)), []).append(float(m.group(3)))
                    out["n_layer_rows"] += 1
            continue
        if not line.startswith("CGC-DECPROF:"):
            continue
        m = RE_HDR.match(line)
        if not m:
            continue
        # A PARTIAL profile is not a step. Alongside the 40-layer step profile the engine emits
        # one-off headers such as `step=91 segs=2 layers=1 total=2.00 ms ... ntok=4`; their ntok
        # passes the decode filter, so counting them adds a bogus 2.00 ms "decode step" each and
        # inflates the step count (measured: 149 steps for a 64-step request, step cb 0.06 ms
        # instead of the ~1-3 ms the per-layer rows show). Keep full-width profiles only.
        if int(m.group("layers")) != layers:
            continue
        out["n_headers"] += 1
        n = int(m.group("ntok"))
        cur_ntok = n
        if n >= prefill_ntok:
            in_prefill = True
            continue
        if n <= decode_ntok:
            # One request = one prefill (arriving as several chunks) followed by its decode steps.
            # The boundary is therefore the FIRST decode step after a prefill -- keying on every
            # prefill step turned 3 real requests into 24, because each prefill has ~8 chunks.
            if in_prefill or req == 0:
                req += 1
                in_prefill = False
            slot = out["per_request"].setdefault(req, {"layer_cb": {}, "steps": []})
            slot["steps"].append((float(m.group("total")), float(m.group("cb")),
                                  float(m.group("wait"))))
    return out


def summarise(parsed, drop_requests=0, n_layers=40):
    rows = []
    for req in sorted(parsed["per_request"]):
        if req <= drop_requests:
            continue
        slot = parsed["per_request"][req]
        med = {L: median(v) for L, v in slot["layer_cb"].items() if v}
        steps = slot["steps"]
        tot = [t for t, _, _ in steps]
        cb = [c for _, c, _ in steps]
        wait = [w for _, _, w in steps]
        rows.append({
            "request": req,
            "n_steps": len(steps),
            "layer_cb_median": {str(L): med[L] for L in sorted(med)},
            "cb_sum": sum(med.values()),
            "loud_layers": [L for L in sorted(med) if med[L] >= 0.5],
            "step_total_median": median(tot) if tot else None,
            "step_cb_median": median(cb) if cb else None,
            "step_wait_median": median(wait) if wait else None,
            # steps tuples are (total, cb, wait): the numerator here is cb, the SECOND element.
            "cb_share_median": (median([c / t for t, c, _ in steps if t > 0])
                                if steps else None),
        })
    return rows


def print_report(path, rows, n_layers):
    print("=" * 110)
    print("PER-LAYER cb BY REQUEST  (%s)" % path)
    print("=" * 110)
    for r in rows:
        print()
        print("request %d   decode steps %d   cb sum %.2f ms   loud(>=0.5) %s"
              % (r["request"], r["n_steps"], r["cb_sum"],
                 ",".join("L%d" % L for L in r["loud_layers"]) or "none"))
        print("  step-level: total %.2f ms   cb %.2f ms   wait %.2f ms   cb share %.1f%%"
              % (r["step_total_median"] or 0, r["step_cb_median"] or 0,
                 r["step_wait_median"] or 0, 100 * (r["cb_share_median"] or 0)))
        med = {int(k): v for k, v in r["layer_cb_median"].items()}
        print("  L0-%d:" % (n_layers - 1))
        for start in range(0, n_layers, 20):
            cells = ["%5.2f" % med.get(L, float("nan")) for L in range(start, min(start + 20, n_layers))]
            print("    %-5s %s" % ("L%d-" % start, " ".join(cells)))
    if len(rows) > 1:
        print()
        print("  NOTE: compare requests on cb_sum / loud set, NOT on a pooled median -- one")
        print("  degraded request moves a pooled median for every layer at once.")


# --------------------------------------------------------------------------- selftest
def _synth(requests):
    """Build a fake log. requests = [(n_steps, {layer: cb}, total, wait)]."""
    out = []
    for n_steps, cbs, total, wait in requests:
        out.append("CGC-DECPROF: step=1 segs=41 layers=40 total=%.2f ms | wait=%.2f (40%%) "
                   "cb=%.2f (50%%) submit=1.00 (1%%) ntok=512 | layer gpu_sum=1.0 union_sum=1.0 "
                   "gap_sum=1.0 ms" % (total, wait, total - wait))
        for _ in range(n_steps):
            out.append("CGC-DECPROF: step=2 segs=41 layers=40 total=%.2f ms | wait=%.2f (40%%) "
                       "cb=%.2f (50%%) submit=1.00 (1%%) ntok=2 | layer gpu_sum=1.0 union_sum=1.0 "
                       "gap_sum=1.0 ms" % (total, wait, total - wait))
            for L in range(40):
                cb = cbs.get(L, 0.2)
                out.append("CGC-DECPROF all: L%d wait=9.99 cb=%.2f submit=0.10 ms gpu=1.0 union=1.0 "
                           "gap=1.0 sg=1 n=1" % (L, cb))
    return "\n".join(out) + "\n"


def cmd_selftest(_args):
    fails = []

    def ok(name, cond, extra=""):
        print(("  PASS  " if cond else "  FAIL  ") + name + (("   " + extra) if extra else ""))
        if not cond:
            fails.append(name)

    # 1. cb is the second numeric after wait=, not the first.
    t = _synth([(3, {0: 0.13, 4: 0.92}, 120.0, 40.0)])
    p = parse(t)
    ok("cb column is group(3), not wait (wait was 9.99 here)",
       abs(p["per_request"][1]["layer_cb"][0][0] - 0.13) < 1e-9)

    # 2. request segmentation: a prefill header opens a new request.
    #    Every layer is set explicitly per request -- `_synth` only overrides the layers you name,
    #    so leaving 39 of them at the default makes the three requests partly identical and the
    #    "polluted median" case below unable to fail. (That is how the first version of this test
    #    passed a wrong assertion by accident.)
    clean = {L: 0.13 for L in range(40)}
    normal = {L: 0.20 for L in range(40)}
    broken = {L: 1.20 for L in range(40)}
    t = _synth([(3, clean, 120.0, 40.0), (3, normal, 120.0, 40.0), (3, broken, 120.0, 40.0)])
    p = parse(t)
    ok("three requests are detected", len(p["per_request"]) == 3,
       "got %d" % len(p["per_request"]))

    # 3. THE POINT: one collapsed request must not be able to speak for the others.
    rows = {r["request"]: r for r in summarise(p)}
    per_req_clean = median(rows[1]["layer_cb_median"].values())
    pooled = median([0.13] * 120 + [0.20] * 120 + [1.20] * 120)
    ok("request 1's own median is the clean value", abs(per_req_clean - 0.13) < 1e-9,
       "got %.2f" % per_req_clean)
    ok("a pooled median WOULD have been polluted", pooled > per_req_clean,
       "pooled %.2f vs clean %.2f" % (pooled, per_req_clean))
    ok("the clean row is not inflated by request 3", abs(rows[1]["cb_sum"] - 40 * 0.13) < 1e-6,
       "got %.2f" % rows[1]["cb_sum"])
    ok("the collapse is visible in its own row", rows[3]["cb_sum"] > 5 * rows[1]["cb_sum"],
       "%.1f vs %.1f" % (rows[3]["cb_sum"], rows[1]["cb_sum"]))

    # 4. --drop-requests skips warmup without touching the rest.
    rows2 = summarise(p, drop_requests=1)
    ok("drop-requests removes exactly one row", len(rows2) == 2 and rows2[0]["request"] == 2)

    # 5. prefill steps are not counted as decode steps.
    t = ("CGC-DECPROF: step=1 segs=41 layers=40 total=500.00 ms | wait=100.00 (20%) "
         "cb=400.00 (80%) submit=1.00 (0%) ntok=512 | layer gpu_sum=1 union_sum=1 gap_sum=1 ms\n"
         "CGC-DECPROF all: L0 wait=1.00 cb=99.00 submit=0.10 ms gpu=1 union=1 gap=1 sg=1 n=1\n")
    p = parse(t)
    ok("a prefill-only log yields no decode steps",
       all(not v["steps"] and not v["layer_cb"] for v in p["per_request"].values()))

    # 6. cb share is computed from the header's own total/cb, per step.
    t = _synth([(2, {0: 0.1}, 100.0, 80.0)])
    rows = summarise(parse(t))
    ok("cb share = cb/total from the header", abs(rows[0]["cb_share_median"] - 0.20) < 1e-6,
       "got %.3f" % rows[0]["cb_share_median"])

    print()
    print("%d failed" % len(fails) if fails else "all selftest cases behaved")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--prefill-ntok", type=int, default=DEFAULT_PREFILL_NTOK)
    ap.add_argument("--decode-ntok", type=int, default=DEFAULT_DECODE_NTOK)
    ap.add_argument("--drop-requests", type=int, default=0,
                    help="skip the first N requests (warmup)")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--json", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return cmd_selftest(args)
    if not args.logs:
        raise SystemExit("give one or more server logs (they contain CGC-DECPROF lines)")

    out = {}
    for path in args.logs:
        text = open(path, errors="replace").read()
        parsed = parse(text, args.prefill_ntok, args.decode_ntok, args.layers)
        if parsed["n_headers"] == 0:
            print("%s: no CGC-DECPROF headers (was CGC_DECODE_PROFILE=1 set?) -- skipped" % path,
                  file=sys.stderr)
            continue
        rows = summarise(parsed, args.drop_requests, args.layers)
        print_report(path, rows, args.layers)
        out[path] = {"requests": rows, "n_headers": parsed["n_headers"],
                     "n_layer_rows": parsed["n_layer_rows"]}

    if args.json:
        open(args.json, "w").write(json.dumps(out, ensure_ascii=False, indent=1))
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
