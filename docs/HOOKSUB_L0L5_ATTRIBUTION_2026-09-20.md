# L0-L5's hook, split into sub-items: the cost is not the bookkeeping, and it is not the round trip

**Date** 2026-09-20 · carrier `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X` (head digest
`219b27c4f9eebc75…`), MTP on, pool 8.0 GiB, temp 0.4, `--n-predict 96`, chat door, `CGC_HOOK_SPLIT=1`
\+ `CGC_GPU_TIMING=1` + `CGC_DECODE_PROFILE_ALL=1` · engine `libggml-base fea88a48e636389d` ·
gate on the same build: **M1 9/9, M2 9/9, M3 9/9**.

## 0. One line

**On every layer, `assign`+`collect`+`publish` are 13 us/call or less (0.3% of the ensure block), so
the slot/LRU/publish bookkeeping is not the cost; on the delivery path the heavy sub-item is the
verify-route syncfill cold fill (`pre_cold`: 1.83 ms per verify call, and *all* of it is verify --
the draft route fills 0), and on the exact path L0-L5 are 62-77% `wait` where the rest of the model
is 76% `submit`.**

## 1. First: the reading the instrument could not give, and why

The first capture on this instrument printed, for L0:

```
CGC-HOOKSUB-L: layer=0 n=42 pre=816 pre_cold=3435 ensure=5595 ...
```

`pre_cold` (3435 us) larger than the `pre` (816 us) that is supposed to *contain* it. That is not a
rounding artefact: it is the hook having **two exits that bank different counters**.

* the decode fast path (`(verify_fast || draft_fast) && cgc_fast_eligible`) banks the cold fill and
  then `return;`s *above* the `t1..t4` accounting block (`llama-context.cpp:6599` banked, exit at
  `:6808`, accounting at `:6846`);
* so `pre_cold` counted a **superset** of the calls that `pre`, `ensure` and `n` counted, and its
  mean was divided by the smaller denominator.

The number that exposes it: **the first capture counted 800 hook calls; the fix revealed 3840** —
80% of the hook's calls were invisible to every column except `pre_cold`. A tool that prints a
nesting claim ("`pre_cold` lives INSIDE `pre`") without testing it is the same defect class as the
rest of this line's history; the claim is now check **(e)** and the banking is now **one** lambda
called from both exits, not two copies of the same six lines (the old shape was correct at each site
and wrong in the pair).

## 2. The reading (log `llama_server_20260920_231649.log`, decode 10.53 t/s, accept 68.72%)

```
paths: fast=1755 of 2400 calls (73%), exact=645
by context: fast verify=1627 draft=128 | cold fill 2985 ms verify (1.83 ms/call) vs 0 ms draft

group       n_hook       pre  pre_cold    ensure   drain    tail     total      wait  wait/tot
L0-L5          336      1707      1611      1795     0.2     1.6      3503      1119       0.32
rest          2064      1212      1184      1106     0.1     0.3      2318       264       0.11

within ensure_batch (L0-L5, per call): assign=7 us (0%), collect=5 us (0%), fill=4729 us (100%),
                                       publish=1 us (0%)  |  within fill: submit=1780 (38%), wait=2948 (62%)
within ensure_batch (rest,   per call): assign=5 us (0%), collect=3 us (0%), fill=4398 us (100%),
                                       publish=0 us (0%)  |  within fill: submit=3347 (76%), wait=1051 (24%)
```

Per-layer rows (us, two populations in one row -- `cold/fast` is per **verify** call):

```
       layer     n  fast exact    total     pre  pre_cold cold/fast  ensure  assign collect   fill publish  submit    wait  wait%
    L0-L5 L0    56    20    36     5832    2206      1751      4903    3621     8.1     6.2   5617     0.7    1291    4325    77%
    L0-L5 L1    56    34    22     3807    1748      1714      2823    2056     6.9     4.1   5222     0.9    2146    3075    59%
    L0-L5 L2    56    36    20     3652    1835      1815      2824    1815     6.5     4.3   5069     0.6    2204    2865    57%
    L0-L5 L3    56    37    19     3238    1897      1874      2836    1340     7.4     5.0   3936     0.5    1951    1985    50%
    L0-L5 L4    56    38    18     ...    (same shape: cold/fast 2.5-2.8 ms)
```

Internal consistency, all checked by the tool on this log: (a) `fill == submit+wait` consistent;
(b) per-layer weighted total 2484 us vs aggregate line 2484 us (1.00x); (c) `sub_n <= n`;
(d) `ensure` total == `assign+collect+fill+publish` total, worst L5 at 1.02x; (e) `pre_cold <= pre`;
**(f) `sub_n == n - fast_n` on every layer** -- two independently incremented counters agree, which is
the strongest available proof that the path attribution is right.

## 3. Which sub-item is heavy on L0-L5 -- and which are ruled out

1. **Ruled out with numbers: the bookkeeping.** `assign`+`collect`+`publish` = 15 us/call on L0-L5 and
   8 us/call on the rest, i.e. **0.3% of the ensure block**. Any fix aimed at the slot allocator, the
   LRU touch, or the publish path can pay at most that, on any of the 41 layers. This is a negative
   result the line can stop paying for.
2. **The delivery path's heavy item is the verify-route cold fill.** `pre_cold` is 94% of `pre` on
   L0-L5 (1611/1707) and 98% on the rest (1184/1212), so `pre` *is* the cold fill. It is
   **fast-path-only** by construction (banked inside the fast guard; the only exit between it and
   `t1` is the fast return) and it is **verify-only in fact**: 2985 ms on 1627 verify calls, 0 us on
   128 draft calls. Per verify call: 1.83 ms across the model, **4.90 ms on L0**, 2.5-2.8 ms on L1-L4.
3. **On the exact path the two groups differ in *mix*, not in total.** Both are ~4.4-4.7 ms/call of
   `fill`, but L0-L5 spend 62-77% of it in `wait` (the pool round-trip) where the rest of the model
   spends 76% in `submit` (sort + job build + `F_RDADVISE`). So a submit-side lever (the ADVISE_ASYNC
   family) targets the *rest*; it does not touch L0-L5.

## 4. Falsifiable single-layer targets

| # | layer | target | must drop by | if it does not |
|---|---|---|---|---|
| T1 | **L0** | the verify-route cold fill (`pre_cold` = 4.90 ms/verify call, 30% of L0's whole hook budget: 98 ms of 327 ms over the run) | **>= 2.45 ms per verify call** (half), same instrument, same run | 'the cold fill is the lever on L0' is falsified; the next target is `submit` (1291 us/call on L0's exact path) |
| T2 | L0 (exact path) | the pool round-trip (`wait` = 2780 us/hook-call reconstructed, 77% of L0's fill) | **>= 1390 us/hook-call** | falsifies 'L0's exact path waits on the round trip'; target moves to `pre` |
| T3 | all 41 | bookkeeping (`assign`+`collect`+`publish`) | ceiling is **15 us/call** | -- the ceiling *is* the verdict: do not spend a window on it |

Population labels are part of the target: T1 is per **verify** call, T2 is per **exact-path** hook
call scaled onto the hook denominator, and neither is a per-step figure (the run has 20-38 fast calls
per layer, so not every step takes the fast path for every layer).

## 5. What is not claimed

* **No per-step extrapolation.** `fast` calls per layer are 20-38 over the run, so "X ms per step"
  would need the per-step call count, which this line does not carry.
* **No absolute t/s claim here.** Absolutes were taken below the harness floor (reclaimable 5.6-7.1 GB
  against the 8000 MB harness floor; the launcher admitted at 79% free). All quoted ratios are
  same-run; the two captures' workloads differ (58.89%/12.57 vs 68.72%/10.53 t/s) and are not
  compared across.
* **The cold fill is not decomposed into expert count vs per-expert latency.** This line has µs only.
* **`pre_cold` is not claimed to be removable.** It exists because a cold expert read as the ZERO slot
  silently loses that expert's contribution; the target is to pay it off the critical path, not to
  stop paying it.

## 6. Artifacts

* engine: `src/llama.cpp/src/llama-context.cpp` -- one `cgc_hs_bank` lambda at both exits; fields
  `fast_n`, `fast_v`, `fast_d`, `cold_v`, `cold_d` appended (old readers unaffected; a log without
  them reports UNKNOWN, never 0).
* analyzer: `scripts/check/hooksub_layers.py` (`--table`; 16/16 selftest, including 'a log without
  the field says UNKNOWN', '`pre_cold > pre` is caught', 'a `fast_n` that does not pair with `sub_n`
  is caught', 'no division without both terms').
* logs: `225808` (fix 1), `231649` (fix 2, quoted); `223754` (pre-fix, the impossible row).
* one-off window watcher: `/tmp/hooksub/watch6.sh` (not tracked; the durable drivers live in
  `Backup/phase_decomp/`).
