#!/usr/bin/env python3
"""Draft budget: what verify width k is optimal once the batch is known NOT to amortise.

What changed
------------
The old model was `S = E / (1 + m*k_eff)`, where m was treated as one opaque number and the
verify batch was assumed to have SOME amortisation. `docs/SHAPE_ROOFLINE_2026-09-22.md` §7
measured the opposite at the kernel level: for n=1..4 at our exact MoE shapes the per-token
cost is FLAT (iq2_s 60.22 -> 57.48 us/token, +4.6%; iq3_s 98.86 -> 102.56, -3.7%). Each token
carries its own top-k ids, so it re-reads its own experts; batching buys rows, not bytes.

So the cost law is affine in the token count T, with the same per-token slope as plain decode:

    cost(T) = C_fixed + c_tok * T

and the two parameters are pinned by two measured points on this carrier:
  * T = 1  (MTP off):   S0 = 93.3 ms   (docs/S0_MTP_ONOFF_AB_2026-09-21.md, median of the OFF arm)
  * T = 4  (MTP on k=3): 247.98 ms     (docs/TARGET25_TWO_AXIS_2026-09-20.md, CGC-DECPROF)

Two points, two parameters, zero degrees of freedom: **this fit describes the data, it does not
test it.** What supports the affine FORM rather than the numbers is the independent kernel
measurement (linear in n, no intercept) and the fact that k=0 reproduces S0 exactly. The test is
a third point -- a T = 2 or T = 3 arm -- and until it exists, k>3 extrapolation is unfounded.

Usage
-----
    python3 scripts/check/draft_budget.py                 # default carrier numbers
    python3 scripts/check/draft_budget.py --selftest
    python3 scripts/check/draft_budget.py --s0 93.3 --step-t4 247.98 --kernel-ms 8.355
"""
import argparse
import sys

# measured on the delivered carrier (docs/SHAPE_ROOFLINE_2026-09-22.md §4)
KERNEL_MS_PER_TOKEN = 8.355


def fit(s0_ms, step_at_T, T):
    """-> (C_fixed_ms, c_tok_ms). Refuses an incoherent pair rather than fitting nonsense."""
    if step_at_T <= s0_ms:
        sys.exit(f"FATAL: step(T={T}) = {step_at_T} ms is not above S0 = {s0_ms} ms; "
                 f"there is no per-token term to fit")
    c_tok = (step_at_T - s0_ms) / (T - 1)
    return s0_ms - c_tok, c_tok


def cost_ms(C_fixed, c_tok, k):
    """One verify round: the target forward over T = k+1 tokens."""
    return C_fixed + c_tok * (k + 1)


def e_tokens(k, alpha):
    """Expected tokens emitted per round. Geometric per-draft accept: draft i is accepted with
    probability alpha given 1..i-1 were, so E = 1 + sum_{i=1..k} alpha^i."""
    if alpha >= 1.0:
        return 1.0 + k
    return 1.0 + (alpha - alpha ** (k + 1)) / (1.0 - alpha)


def tps(C_fixed, c_tok, k, alpha):
    return 1000.0 * e_tokens(k, alpha) / cost_ms(C_fixed, c_tok, k)


def break_even_alpha(C_fixed, c_tok, s0_ms, k):
    """Smallest per-draft accept at which round k beats a plain step. None if unreachable."""
    need = cost_ms(C_fixed, c_tok, k) / s0_ms          # E must exceed this
    if need > k + 1:
        return None                                    # even alpha=1 cannot break even
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if e_tokens(k, mid) < need:
            lo = mid
        else:
            hi = mid
    return hi


def best_k(C_fixed, c_tok, alpha, kmax=16):
    return max(range(0, kmax + 1), key=lambda k: tps(C_fixed, c_tok, k, alpha))


def alpha_from_ratio(C_fixed, c_tok, s0_ms, ratio, k):
    """Invert a same-harness measured ON/OFF t/s ratio at a known k to recover the per-draft
    accept that must have produced it. This is what closes the loop between the cost fit and
    the acceptance measurement: the ratio was measured in ONE harness, so it carries no
    cross-harness error -- the accept it implies is therefore firmer than any quoted accept.
    """
    want = ratio * cost_ms(C_fixed, c_tok, k) / s0_ms        # E that the ratio implies
    lo, hi = 0.0, 1.0
    if e_tokens(k, 1.0) < want:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if e_tokens(k, mid) < want:
            lo = mid
        else:
            hi = mid
    return hi


def _selftest():
    bad = total = 0

    def chk(name, ok, detail=""):
        nonlocal bad, total
        total += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{(' - ' + detail) if detail else ''}")
        if not ok:
            bad += 1

    C, c = fit(93.3, 247.98, 4)
    chk("fit: c_tok lands in the 46-59 ms band the engine measured",
        46.0 <= c <= 59.0, f"{c:.2f}")
    chk("fit: k=0 reproduces S0 exactly",
        abs(cost_ms(C, c, 0) - 93.3) < 1e-9, f"{cost_ms(C, c, 0):.4f}")
    chk("fit: refuses an incoherent pair", _refuses(lambda: fit(93.3, 80.0, 4)))
    # the existing two-axis grid cell: T=4 with mean_len 4.00 reads 16.13 t/s
    chk("model vs shipped grid: alpha=1, k=3 == 16.13 t/s",
        abs(tps(C, c, 3, 1.0) - 16.13) < 0.02, f"{tps(C, c, 3, 1.0):.2f}")
    be = [break_even_alpha(C, c, 93.3, k) for k in (1, 2, 3, 4)]
    chk("break-even accept rises monotonically with k",
        all(be[i] is not None and be[i] < be[i + 1] for i in range(len(be) - 1)),
        " ".join(f"{x:.3f}" for x in be))
    chk("alpha=1 ceiling is the per-token cost, not the fixed term",
        abs(1000.0 / c - 19.39) < 0.05, f"{1000.0 / c:.2f}")
    chk("kernel is a minority of c_tok (the measured 15-19% claim)",
        0.10 < KERNEL_MS_PER_TOKEN / c < 0.25, f"{KERNEL_MS_PER_TOKEN / c * 100:.1f}%")
    print(f"selftest: {total - bad}/{total} passed")
    return 1 if bad else 0


def _refuses(fn):
    try:
        fn()
        return False
    except SystemExit:
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", type=float, default=93.3, help="MTP-off step, ms")
    ap.add_argument("--step-t4", type=float, default=247.98, help="MTP-on k=3 step, ms")
    ap.add_argument("--kernel-ms", type=float, default=KERNEL_MS_PER_TOKEN,
                    help="measured MoE GEMV ms per token (all 41 layers)")
    ap.add_argument("--ratio", type=float, default=0.889,
                    help="same-harness measured MTP on/off t/s ratio (S0_MTP_ONOFF_AB median)")
    ap.add_argument("--ratio-k", type=int, default=3,
                    help="the k that ratio was measured at")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())

    C, c = fit(args.s0, args.step_t4, 4)
    s0_tps = 1000.0 / args.s0
    print(f"# draft budget  (S0={args.s0} ms -> {s0_tps:.2f} t/s; step(T=4)={args.step_t4} ms)")
    print(f"cost(T) = {C:.2f} + {c:.2f}*T  ms      [2-point fit: describes, does not test]")
    print(f"  per-token term    : {c:.2f} ms   (kernel {args.kernel_ms:.2f} = "
          f"{args.kernel_ms / c * 100:.1f}%; everything else {c - args.kernel_ms:.2f} = "
          f"{(c - args.kernel_ms) / c * 100:.1f}%)")
    print(f"  per-round fixed   : {C:.2f} ms")
    print(f"  NO batch discount : c_tok in a verify round == c_tok in plain decode")

    print(f"\n## break-even: the accept a round must reach to beat one plain step")
    print(f"{'k':>2} {'T':>2} {'cost ms':>9} {'E needed':>9} {'alpha needed':>13}")
    for k in range(1, 8):
        need = cost_ms(C, c, k) / args.s0
        a = break_even_alpha(C, c, args.s0, k)
        print(f"{k:>2} {k + 1:>2} {cost_ms(C, c, k):>9.2f} {need:>9.3f} "
              + (f"{a:>13.3f}" if a is not None else f"{'unreachable':>13}"))
    print("  -> wider k demands HIGHER accept, monotonically. That is the consequence of "
          "linear cost.")

    print(f"\n## optimum k by per-draft accept (t/s; MTP-off baseline {s0_tps:.2f})")
    alphas = [0.465, 0.555, 0.600, 0.673, 0.735, 0.800, 0.900, 1.000]
    print("alpha   " + "".join(f"k={k:<8}" for k in range(1, 7)) + "  best")
    for a in alphas:
        row = "".join(f"{tps(C, c, k, a):<10.2f}" for k in range(1, 7))
        bk = best_k(C, c, a)
        print(f"{a:<7.3f} {row} k={bk} ({tps(C, c, bk, a):.2f})")

    print(f"\n## the same-harness ratio, inverted (no cross-harness error)")
    a = alpha_from_ratio(C, c, args.s0, args.ratio, args.ratio_k)
    if a is None:
        print(f"  ratio {args.ratio} at k={args.ratio_k} is unreachable at any accept")
    else:
        bk = best_k(C, c, a)
        print(f"  measured ON/OFF = {args.ratio} at k={args.ratio_k} "
              f"=> per-draft accept alpha = {a:.3f}")
        print(f"  break-even alpha at k={args.ratio_k} is "
              f"{break_even_alpha(C, c, args.s0, args.ratio_k):.3f} "
              f"=> that k is {'PROFITABLE' if args.ratio > 1 else 'a NET LOSS'} today")
        print(f"  at alpha={a:.3f} the best k is {bk}: {tps(C, c, bk, a):.2f} t/s "
              f"= {tps(C, c, bk, a) / s0_tps:.3f}x the MTP-off baseline")
        for k in range(1, 8):
            r = tps(C, c, k, a) / s0_tps
            print(f"    k={k}: {tps(C, c, k, a):6.2f} t/s  {r:6.3f}x"
                  + ("   <- best" if k == bk else ""))

    print(f"\n## what the accept axis alone can buy")
    print(f"  alpha=1 (perfect drafter), k->inf: {1000.0 / c:.2f} t/s "
          f"= {1000.0 / c / s0_tps:.2f}x")
    print(f"  => 25 t/s needs c_tok <= {1000.0 / 25:.2f} ms, i.e. a "
          f"{(1 - 40.0 / c) * 100:.0f}% cut in the per-token cost.")
    print(f"     Removing the WHOLE kernel term gives {C:.2f} + {c - args.kernel_ms:.2f}*T "
          f"=> alpha=1 ceiling {1000.0 / (c - args.kernel_ms):.2f} t/s. Kernel is necessary, "
          f"not sufficient.")
    print(f"     The {c - args.kernel_ms:.2f} ms/token of non-kernel cost is the actual target.")


if __name__ == "__main__":
    main()
