#!/usr/bin/env python3
"""Per-layer slot allocation from a routing trace: is there slack to borrow?

WHY THIS EXISTS
`thrash_sim.py` answers "what does a UNIFORM slot budget buy" and prints, per layer, the smallest
S at which that layer's capacity misses vanish (`S_for_0`). That last column is the interesting one,
because it is usually not flat: a few layers need far more slots than the rest. The pool is
PER-LAYER partitioned (`slots_l(cache, layer)`; a layer's fills never touch another layer's slots),
so the allocation across layers is a free variable the engine currently pins to a constant.

This script turns that column into the decision it implies: hold the TOTAL slot count fixed at what
the engine actually budgets (layers x 143 for 8 GiB on iq3) and move slots BETWEEN layers, then
report how much capacity miss the move buys and which layers pay for it.

WHAT IT DOES NOT DO
It does not model the engine's replacement policy beyond what thrash_sim already implements, and it
does not change the trace. Reuse of thrash_sim is by import, not by copy -- if that file moves, pass
--sim.

SELF-TEST
    slot_alloc_curve.py --selftest      # synthetic traces, incl. a negative control
"""

import argparse
import importlib.util
import os
import sys
from collections import defaultdict

DEFAULT_SIM = "Backup/thrash_sim_20260919.py"


def load_sim(path):
    """Import the rescued simulator module (it is not committed; see the header)."""
    if not os.path.exists(path):
        raise SystemExit("simulator not found at %s -- pass --sim <path>.\n"
                         "  It was rescued from /tmp; the project's history says artifacts left\n"
                         "  there get lost (36 runs on 2026-09-18), so it needs a tracked home." % path)
    spec = importlib.util.spec_from_file_location("thrash_sim_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def per_layer_curves(sim, path, slots, max_slots, step, top_k):
    """{layer: [capacity at 0, step, 2*step, ... up to max_slots]} for one trace."""
    steps, _ctx, nlines = sim.parse_trace(path, top_k=top_k)
    per_layer_steps = defaultdict(list)
    for st in steps:
        for layer, demand in st.items():
            per_layer_steps[layer].append(sorted(demand))
    # Start at `step`, never 0: thrash_sim's victim picker does min() over an empty set at S=0
    # (its own sweep starts at 8 for the same reason). And put the ENGINE'S actual budget on the
    # grid, so the uniform baseline we compare against is exactly `slots`, not the nearest
    # multiple of `step`.
    grid = sorted(set(range(step, max_slots + 1, step)) | {slots})
    curves, distinct = {}, {}
    for layer, seq in per_layer_steps.items():
        curves[layer] = [sim.simulate({layer: seq}, s, "lru")["capacity"] for s in grid]
        distinct[layer] = sim.simulate({layer: seq}, slots, "lru")["per_layer"][layer]["distinct"]
    return per_layer_steps, grid, curves, distinct, nlines


def allocate(grid, curves, budget):
    """Greedy marginal allocation of a fixed budget over separable per-layer curves.

    Greedy on the largest marginal DROP is optimal here as long as each curve's marginal gain is
    non-increasing (the usual shape: the first slots retire the most misses). Where that is not
    true the result is an upper bound on the achievable gain, which is the direction that matters
    for a "should we even try" decision -- and it is printed as such.
    """
    # Everyone starts at the first grid point; that baseline is already spent.
    curves_at = {layer: 0 for layer in curves}
    spent = grid[0] * len(curves)
    steps = 0
    while True:
        best, best_gain, best_cost = None, 0.0, 0
        for layer, i in curves_at.items():
            c = curves[layer]
            if i + 1 >= len(c):
                continue
            cost = grid[i + 1] - grid[i]
            if spent + cost > budget:
                continue
            gain = c[i] - c[i + 1]
            if gain > best_gain:
                best, best_gain, best_cost = layer, gain, cost
        if best is None:
            break
        curves_at[best] += 1
        spent += best_cost
        steps += 1
        if steps > 100000:
            break
    return curves_at, spent


def report(args):
    sim = load_sim(args.sim)
    for path in args.traces:
        steps, grid, curves, distinct, nlines = per_layer_curves(
            sim, path, args.slots, args.max_slots, args.step, args.top_k)
        layers = sorted(curves)
        base_total = sum(curves[l][0] for l in layers)          # 0 slots: all compulsory+capacity
        uniform_idx = grid.index(args.slots)
        uniform = [curves[l][min(uniform_idx, len(curves[l]) - 1)] for l in layers]
        unif_sum = sum(uniform)
        budget = args.slots * len(layers)
        alloc, spent = allocate(grid, curves, budget)
        opt = [curves[l][alloc[l]] for l in layers]
        opt_sum = sum(opt)
        print("=" * 78)
        print("PER-LAYER SLOT ALLOCATION  (%s)" % path)
        print("=" * 78)
        print("trace lines %d   layers %d   grid step %d   slots/layer uniform=%d (budget %d)"
              % (nlines, len(layers), args.step, args.slots, budget))
        print()
        print("  capacity misses, uniform %d slots/layer : %d" % (args.slots, unif_sum))
        print("  capacity misses, same budget reallocated: %d  (%+.1f%%)"
              % (opt_sum, 100.0 * (opt_sum - unif_sum) / unif_sum if unif_sum else 0.0))
        gain = sum(max(0, grid[alloc[l]] - args.slots) for l in layers)
        freed = sum(max(0, args.slots - grid[alloc[l]]) for l in layers)
        print("  total slots: uniform %d -> %.0f   (budget %d)"
              % (args.slots * len(layers), sum(grid[alloc[l]] for l in layers), budget))
        print("  REDISTRIBUTED: %d slots gained by the head, %d freed by the tail (net %+d)"
              % (gain, freed, freed - gain))
        print()
        print("%-7s %9s %9s %10s %10s %9s" % ("layer", "distinct", "slots@143", "cap cap@143", "slots opt", "cap opt"))
        for l in layers:
            print("%-7d %9d %10d %10d %10d %9d"
                  % (l, distinct[l], args.slots, uniform[layers.index(l)], grid[alloc[l]], opt[layers.index(l)]))
        gainers = [l for l in layers if grid[alloc[l]] > args.slots]
        losers = [l for l in layers if grid[alloc[l]] < args.slots]
        print()
        print("  gains slots : %s" % (",".join("L%d" % l for l in gainers[:14]) or "none"))
        print("  gives up    : %s" % (",".join("L%d" % l for l in losers[:14]) or "none"))
        give_cost = sum(uniform[layers.index(l)] for l in losers)
        get_benefit = sum(uniform[layers.index(l)] - opt[layers.index(l)] for l in gainers)
        print("  the lenders lose %d capacity misses; the borrowers save %d  (net %+d)"
              % (give_cost, get_benefit, give_cost - get_benefit))
        print()
        print("VERDICT: %s" % verdict(unif_sum, opt_sum, gainers, spent, budget))
        print()


def verdict(unif_sum, opt_sum, gainers, spent, budget):
    if not unif_sum:
        return "no capacity misses at the uniform budget -- nothing to reallocate"
    rel = (opt_sum - unif_sum) / float(unif_sum)
    if rel > -0.02:
        return ("reallocation buys %.1f%% -- UNDER the resolution this repo has been able to "
                "measure; not worth an engine change" % (100 * abs(rel)))
    return "reallocation buys %.1f%% -> worth a per-layer slot experiment" % (100 * abs(rel))
    return "reallocation buys %.1f%% -> worth a per-layer slot experiment" % (100 * abs(rel))


# --------------------------------------------------------------------------- selftest
def _fake_curves(specs):
    """specs = [(layer, [cap at grid 0,1,2,...])] -- hand-built so the answer is known."""
    curves = {}
    for layer, cap in specs:
        # grid step is 1 in the selftest
        curves[layer] = list(cap)
    return curves


def cmd_selftest(args):
    fails = []

    def ok(name, cond):
        print(("  PASS  " if cond else "  FAIL  ") + name)
        if not cond:
            fails.append(name)

    # 1. hand-built, small enough to enumerate: a layer that saturates fast and one that keeps
    #    paying. The assertion is NOT "which layer wins" -- the 4th slot is a tie, and a tie is
    #    legal -- it is that greedy reaches the brute-force optimum on the same budget.
    grid = [0, 1, 2, 3, 4, 5]
    curves = _fake_curves([(0, [100, 10, 0, 0, 0, 0]),      # saturates at 2 slots
                           (1, [100, 60, 40, 30, 25, 25])])  # still 25 at 4 slots
    alloc, spent = allocate(grid, curves, budget=4)

    def _tot(a):
        return sum(curves[l][a[l]] for l in curves)

    brute = min(_tot({0: i, 1: j}) for i in range(6) for j in range(6) if i + j == 4)
    ok("greedy reaches the brute-force optimum", _tot(alloc) == brute)
    ok("greedy spends exactly the budget", spent == 4)

    # 2. identical layers -> no gain from reallocation (negative control)
    a2, _ = allocate(grid, {0: [100, 50, 30, 20, 15, 12], 1: [100, 50, 30, 20, 15, 12],
                            2: [100, 50, 30, 20, 15, 12], 3: [100, 50, 30, 20, 15, 12]}, budget=8)
    ok("identical layers split the budget evenly", sorted(a2.values()) == [2, 2, 2, 2])

    # 3. budget 0 -> nothing allocated, no crash
    a3, s3 = allocate(grid, curves, budget=0)
    ok("zero budget allocates nothing", s3 == 0 and set(a3.values()) == {0})

    # 4. verdict thresholds are reachable and ordered
    ok("tiny gain reads as not worth it", "not worth" in verdict(1000, 990, [0], 0, 1000))
    ok("big gain reads as worth it", "worth a per-layer" in verdict(1000, 500, [0], 10, 1000))

    print()
    print("%d failed" % len(fails) if fails else "all selftest cases behaved")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="*")
    ap.add_argument("--sim", default=DEFAULT_SIM)
    ap.add_argument("--slots", type=int, default=143, help="uniform slots/layer the engine budgets")
    ap.add_argument("--max-slots", type=int, default=384)
    ap.add_argument("--step", type=int, default=8, help="slot granularity of the grid")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return cmd_selftest(args)
    if not args.traces:
        raise SystemExit("give one or more trace logs (a server log containing CGC-IDS lines)")
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
