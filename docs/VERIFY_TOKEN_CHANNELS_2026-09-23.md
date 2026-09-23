# The marginal verify token, split by channel: it is not the dispatcher

**Date** 2026-09-23 · carrier `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X` · cell `prod25 / decode-spec`
(llama-bench `-p 0 -n 128 -d 512 -b 8 --spec-type draft-mtp --spec-draft-n-max 3`) · tool
`scripts/check/verify_marginal.py --steps` / `--arm-dir`.

## 0. One line

**One more verify token costs ~51 ms inside the step, and its channels are device span ~56-61%, host hook
~35-40%, dispatch ~2% — so batching the host path is a dead end; and the *delivered* cost of widening the
round is only ~+9 ms/token, because the win of k=3 over k=1 is amortisation of the round, not the token.**

Three readings were added the same day. `§8` the device span's per-op attribution exists in **no product
on disk** (45 logs carry the op table; 25 are refused by its sampling mix, 17 by a one-to-three-sample
width cell, 0 pass). `§8b` the capture that was meant to fix that **was then run** (6 launches, NSM
unsampled, widths 2/3/4) and the design matrix is now **solved** instead of bucketed -- and its answer is
a refusal with numbers: rank 20 of 44, R2 0.67-0.69, fitted marginal -0.98 against a measured +12.66
ms/token, so no per-op slice of the verify token's device time is recoverable from it. `§9` the round
budget is arithmetic (`rest = F + m*T`) and `§9b` turns it into the k decision as a function of accept:
**at the deployed a = 0.5825 the model says k = 1** (measured arms agree, 98.7 vs 108.9 ms/token), wide
verify only pays above a ~ 0.80, and **25 t/s is unreachable at any accept rate** -- the best arm stops
at 13.5-14.1 t/s even at a = 1.0. `§8c`/`§8d` then closed the per-op question as far as this graph can
take it: composition medians *do* identify per-kind levels once the encoder shards a segment (28 -> 136
compositions, rank 20 -> **36 of 44**, and the model-free direct check agrees to **4.5%** where it
previously contradicted the fit), but the *marginal* still cannot be attributed -- **42% of it is
per-command-buffer overhead** and the rest needs shape-keyed kinds, not counts.

## 1. Why the round-level axes were not enough

The pre-existing axes (`docs/MTP_K_SWEEP_2026-09-22.md`) needed CGC-VERIFY-OP for the compute term (which
suppresses the segmented dispatcher — the trap that tool documents) and the borrowed `PER_MISS_MS` law.
The DECPROF **step line** answers the same question with neither: its own three terms are measured
(`total = wait + cb + submit`, closure exact to 0.00 in every arm below), and with `CGC_GPU_TIMING=1` it
also carries the device-clock tail (`gpu_sum / union_sum / gap_sum`). Channels, as the engine defines them
(`ggml-backend.cpp`): `wait` = the CPU window enclosing the segment's GPU work, `cb` = the expert-cache
top-k hook (slot management + blocking fill), `submit` = dispatching that segment, `union` = device span
of the step's segments, `gap` = device idle before a segment.

## 2. The primary estimator: the slope inside one launch

Verify width is **not assigned** (see §4 for what picks it), so the within-arm slope is the drift-free
reading: every step in the family comes from the same launch, the same pool and the same round structure.
Three k=3 arms, 225 verify steps each (widths 2/3/4), 64 prefill chunks excluded and counted
(`excluded widths > 4: [(8, 64)]`):

| rep | total | wait | device span (union) | host hook (cb) | dispatch (submit) | shares union/cb/submit |
|---|---:|---:|---:|---:|---:|---|
| 1 | +51.02 | +32.07 | **+31.13** | **+17.84** | +1.10 | 61% / 35% / 2% |
| 2 | +54.97 | +32.35 | **+31.05** | **+21.27** | +1.35 | 56% / 39% / 2% |
| 3 | +53.88 | +32.15 | **+30.73** | **+20.36** | +1.36 | 57% / 38% / 3% |

ms per token; the device-span column is stable to 1.3% across the three launches while the host hook is
not (it is miss-driven, §5). Predeclared rule (≥0.60 on one channel ⇒ that channel owns it): rep 1
`UNION-LED (61%)`, reps 2-3 `NO SINGLE LEVER` — i.e. the honest reading is **device span leads, the hook is
a real second slice, dispatch is noise**. `gap` (device idle) moves +19.7-23.2 ms/token inside the same
window; it is a share of the device timeline, not of the step, so it is reported and not bucketed.

## 3. The second estimator, and why the two disagree by 5.7x

The same question asked as an arm contrast — k=1 (T=2) vs k=3 (T=4), interleaved, 3 reps, per-step medians
at each arm's **configured** width, differenced over dT=2 (the same `arms.jsonl` the k-sweep wrote):

| channel | k1 T=2 | k3 T=4 | marginal/token |
|---|---:|---:|---:|
| total | 161.62 | 179.57 | **+8.97** |
| wait | 124.83 | 130.89 | +3.03 |
| device span | 113.85 | 122.83 | +4.49 |
| host hook | 24.45 | 34.24 | +4.90 |
| dispatch | 10.74 | 11.81 | +0.54 |

The tool prints the disagreement (5.7x) rather than choosing. The reason is structural, not noise: **the k=1
arm pays more hook time per step at the same width** (24.45 ms at T=2 versus 9.66 ms for the k=3 arm's own
T=2 steps) because it does ~2x the rounds per emitted token. So:

* the **slope** answers "what does the (k+1)-th token cost inside this arm's round" — the batch-verify
  economics question;
* the **contrast** answers "what does raising k from 1 to 3 do to the delivered cost per token" — and it
  is small *because the round's fixed work is spread over three tokens instead of one*.

Both are real; they are different quantities, and only the first one is "the marginal verify token".

## 4. What actually picks the verify width (measured, not inferred)

`CGC_PHASE_DBG=1` labels each graph before its step line. Fresh cell, `prod25 / decode-spec`, 297 verify
steps, **297/297 labelled**:

| family | width | phase | n |
|---|---:|---|---:|
| trunk | 2 | VERIFY | 99 |
| trunk | 3 | VERIFY | 28 |
| trunk | 4 | VERIFY | 170 |
| trunk | 8 | UNKNOWN (prefill chunk) | 64 |
| trunk | 1 | UNKNOWN (first step after prefill) | 1 |
| MTP head | 2/3/4 | (one layer: no il=0/1 pair) | 297 |

So every narrow trunk step in this shape **is** a verify graph — the width mix (4,2,4,3,4,2,...) is the
spec loop's own, and the slope in §2 is a verify-token slope. Turning the phase filter on (`--phase VERIFY`)
uses exactly these rows. The `ntok=8` chunks are why the family's upper bound is 4 and not the pool clamp 8:
including them moved the slope from +51.0 to +45.0 (they are 64 of 289 steps and much more expensive).

## 5. The host slice is a regime quantity, not a token quantity

Three independent readings agree that `cb` tracks the miss regime and not the width: 9.66 ms (k3, T=2),
24.45 ms (k1, T=2), 34.24 ms (k3, T=4) — the same width, 2.5x apart across arms. This is why the slope's
35-40% host slice cannot be converted into "the token needs 18 ms of hook": it is the *observed* marginal
in this arm's mix, and the F1 census (25.6% compulsory / 74.4% capacity misses) is what says how much of it
is policyable.

## 6. The conclusion, and the direction it kills

* **Dispatch is not a lever.** +0.45 to +1.36 ms/token (1-3%), and in the delivery regime `submit` is
  11.8 ms of a 179.6 ms step (6.6%) of which the marginal is ~1 ms — the rest is already overlapped
  (measured before: fill hidden, prefetch completed and nobody waits). Batching the host path to zero buys
  ~2% of the marginal token.
* **The device span leads** (56-61% of the marginal token, ~31 ms/token, stable to 1.3% across launches):
  the verify graph's GPU time for 2-4 tokens. That is the `MUL_MAT_ID`/MoE verify path — the shape work
  (`SHAPE_WORLD_MODEL` P0: NSG on the id path, the 256-expert verify shape that `mmid_shapes.py` measures),
  not the host.
* **The hook is the second lever** (35-40%, ~18-21 ms/token) and it is miss-driven; its size depends on the
  arm's rounds-per-token, which is exactly what the coverage/routing-locality work can move.
* **The biggest measured effect is neither**: widening the round from k=1 to k=3 changes the delivered cost
  per token by only +8.97 ms, because a round's fixed work is amortised over 3 tokens. Any "batch verify"
  project should be aimed at the *device span per token* and at the *round's fixed work*, not at the
  host-side batching of tokens.

## 7. An instrument defect found on the way (not this line's file)

The device channels are refused by a new unit gate: a span may not exceed the CPU window that encloses it
by more than the segment overlap the dispatcher allows. Measured:

Family restricted to verify widths 2-4 (the same rule as §2), `union/wait` median and the fraction of
steps above 1.10x:

| arm | `libggml-base` | verify steps | median | implausible |
|---|---|---:|---:|---:|
| 09-22 16:53 k1/k3 pair (6 arms) | `4fc9832ab2f8bc95` | 225-311 each | 0.93-0.95 | 0.3-2.4% |
| 09-22 15:54 L3 dense arm | (09-22 build) | 231 | 0.93 | 0.9% |
| 09-23 13:41 fresh cell | `7099cb53fba54531` | 297 | **793981** | **93% (from step 1)** |

So the tail is sound on the 09-22 builds and off by 10^5-10^7 on the 09-23 one; the host channels of the
same lines are unaffected in both. The file is clean in the worktree and its last commits are the G4
instrumentation series (`fc0b8b406`, `61ad76bae`), so the reading is "a 09-23 build of that file reports
the GPU clock in the wrong unit", not "the phase knob did it" and not "every DECPROF arm is unsafe". It is
invisible to every consumer that reads only `wait/cb/submit`. The tool now fails closed above 5% (device
channels refused, host channels reported, the median ratio printed) — which is why §2's device column
comes from the 09-22 pair, while §2's host columns are reproduced on the 09-23 cell (total +44.04,
wait +25.84, cb +17.76, dispatch +0.45 — same structure, dispatch ~1%).

A second rule defect surfaced on the L3 arm and is fixed: the trunk used to be chosen as the *modal*
`layers` value, and that arm has 4400 one-layer MTP-head steps against 231 trunk steps, so the mode is the
head. The trunk is now the LARGEST layer count, with a 4x separation rule and a refusal when it does not
hold. The same arm also shows why the family's upper bound is 4: its 64 prefill chunks are 8 wide, and
including them turned a 0.9% violation rate into 24% (and a +47.8 ms/token device slope into +31.1).

## 7b. What the device span is NOT (so the next instrument is chosen, not guessed)

The per-step `CGC-GPUOPK` kind x op table looks like the obvious attribution and is not one: it prints a
*ranked* subset per step (most kinds are absent from most steps), so regressing it over widths leaves
**86% of the span unattributed** — the printed rows sum to +6.9 ms/token against a span slope of +47.8 (or
+31.1 in the clean pair). A partition needs the node-level family (`CGC_GPU_NODES` /
`CGC_GPU_NODES_MATRIX`, already parsed by `gdn_split.py`'s design matrix) or a full-table variant, and any
future "the span is X" claim must show its rows summing to the span.

## 8. Where the span lives, per op: not answerable from any log on disk (and the flag that fixes it)

`§7b` said a span claim has to show its rows summing to the span. This is the attempt, with the
instrument built into the tool (`--node-attr LOG`) so the next log is judged the same way instead of
being read hopefully. The table is `CGC-GPUOPS`: one row per ggml op per step, with `wcntw` = the
buffer's duration shared over the nodes whose op encodes (`ggml-backend.cpp`: *"Justified by the
encoder, not by a model"*), printed on a **1-in-8 sample** of the DECPROF cadence and wiped per step.

Two gates, both measured, both reported with their numbers:

| gate | rule | the L3 dense arm | a 0919 server log |
|---|---|---|---|
| G1 sampling | sample width mix within 10pp of the family's | pop {2:23,3:12,4:65}% vs sample {2:17,3:0,4:83}% = **17.2pp -> refused** | 1.0pp -> ok |
| G2 cells | every width used needs >= 4 samples | width 3 is 12% of the family and absent | width 2 is **one step** -> refused |

The G2 clause that catches the right-hand column was added after the fact: with a single width-2
observation the regression is that one step, and the table-total slope came out **-31.4 ms/token**
against the family's own `gpu_sum` slope of **+50.2** on the same steps. Both clauses are in the
selftest (a decimated sample, a one-sample cell, and a misparsed table each have to be refused).

Census over everything on disk that carries the table:

```
45 logs with CGC-GPUOPS tables
  25  refused by G1 (the decimation is entangled with the width pattern)
  17  refused by G2 (a cell used by the regression has 1-3 samples)
   3  no table inside the trunk family at all
   0  pass both gates
```

So the per-op slope of the span **does not exist in any product on disk**, and the reason is
structural rather than careless: the logs with real width spread are the ones the 1-in-8 decimation
biases (spec-loop widths are periodic, so a fixed stride entangles with them), and the logs whose
sample is faithful are the ones whose family is 97% one width. Reported anyway, because it needs no
regression -- the per-width **level** composition (medians of `wcntw` in ms over the sampled steps),
together with the two denominators that qualify it:

| width (n) | MUL_MAT | ADD | MUL | GET_ROWS | UNARY | MUL_MAT_ID | UNATTRIBUTED |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 (19) | 37.63 | 29.61 | 21.79 | 11.61 | 9.81 | 8.62 | 26.9% below `gpu_sum` + 12.8% residual |
| 2 (4) | 28.31 | 23.48 | 15.79 | 9.58 | 8.91 | 6.75 | same |

The "unattributed" column is the honest part: the table's own `total` is **26.9% below the step's
`gpu_sum`** (median) because it sums only buffers that held >=1 node, and `total - sum(wcntw)` is a
further **12.8%**. Nothing is redistributed over the rows, so these medians are not a partition of
the span and must not be added up as one.

**The capture that answers it is `CGC_GPU_NODES_MATRIX=1`** (one line per command buffer with its
duration and its node census, unsampled): its model `dur_i = sum_k cnt_ik * t_k` is identified
**within one width**, because buffers differ in node composition, so it does not need the width
spread that the sampling destroyed. `gdn_split.py` already parses it into a design matrix.

### 8b. That capture was run (18:14-18:20). The design matrix is now SOLVED, and it identifies no op

```
python3 scripts/check/mtp_k_sweep.py --cells k1,k2,k3 --reps 2 --logdir /tmp/nsm_ksweep \
  --extra-env "CGC_GPU_NODES=1;CGC_GPU_NODES_MATRIX=1;CGC_DECODE_PROFILE=1;CGC_DECODE_PROFILE_ALL=1;CGC_GPU_TIMING=1;CGC_PHASE_DBG=1"
```

6 launches, ~1.1 min each, engine `llama-bench=639a88bf1411dd68`, widths 2/3/4 with >= 4 steps at
every cell used by the marginal. The read is `--nsm`, now one solve over several logs (`dur_i = sum_k
cnt_ik * t_k` + one launch offset per log) rather than the co-occurrence partition it started as --
that partition put 20.30 of 20.36 ms/token into a single `(mixed)` bucket and left every real group at
0.00, i.e. a table that reads like an attribution and identifies nothing (measured on the first
capture, 17:01, same day).

| read | steps | buffers | R2 | count rank | fitted slope | measured slope (NSM) | host wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| k1+k3 pooled | 349 | 110 972 | 0.6914 | 20 of 44 | **-0.98** | +12.66 | +19.06 ms/token |
| k3 alone (one launch) | 187 | 59 178 | 0.6663 | 20 of 44 | **+0.34** | +31.19 | +43.67 ms/token |

So the answer to this section's question is a refusal with numbers, not a table: **no per-op slope of
the verify token's device time is recoverable from this design matrix.** Its two reasons are measured:

* **rank 20 of 44.** This graph emits ~20 kinds in fixed ratios, so 24 kinds' columns are linear
  combinations of others' in *every* buffer -- no fit recovers them, and the solver's pivot test says
  so (`cond_ok: False`). 26 kinds also sit in fewer than 3 distinct co-occurrence signatures, where
  the fit returns a number it does not own (R2 stays 1.0); `patterns` is what catches them, and they
  are held out of the marginal instead of quoted.
* **R2 0.67-0.69.** The linear model explains two thirds of the buffer durations; the missing third is
  not small enough for the remainder to be presented as a row. The fitted marginal (-0.98) and the
  measured one (+12.66) are 13.6 ms/token apart on the same steps -- the gap §9 calls composition.

### 8c. Why it is not identification: 28 compositions, rank 20, and a falsifier

The estimator was then rebuilt around what the graph actually emits, and this is the part that
settles the question:

```
distinct buffer compositions in the whole capture            28   (56 pooled over k1+k3)
within-composition IQR/median of dur_ns                      64.8%  (p10 35.5%, p90 129.6%)
fit R2 on composition MEDIANS (the new estimator)            0.9197   k3 alone
fit R2 of the same costs on the raw buffer rows              0.6611   -> the gap is jitter
count-matrix rank                                            20 of 44 kinds (unchanged)
```

So the earlier `R2 = 0.69` was not model error -- it was the per-buffer jitter of a shared GPU, and
`--nsm` now reports both numbers side by side so the two can never be confused again. Medians fix the
levels (0.92). They do **not** fix identification, and the capture contains a falsifier that no R2 can
see: a buffer whose nodes are all ONE kind measures that kind with no model at all. Measured
(k3 arm, direct = median minus the fit's own offset):

| kind | observations | fitted | direct (no model) |
|---|---:|---:|---:|
| `ffn_moe_argsort` | 1870 | **-407.1 us** | **+13.5 us** |
| `cache` | 69 | 94.6 us | 716.9 us |

The fit assigns a **negative** duration to a kind it measures at +13.5 us, while its R2 stays 0.92.
That is the whole lesson in one row: a level fit can be excellent and its coefficients can still be
arbitrary, because with rank 20 of 44 the solution distributes whatever the identified directions leave
to the kinds it cannot separate. Consequently the marginal is refused too -- fitted **+0.29** against
measured **+31.19** ms/token on the k3 arm (pooled: +2.16 vs +12.66).

### 8d. The shard the encoder CAN do, and what it fixed (18:36)

The encoder cannot emit one node per buffer (`n_cb=127` deadlocks Metal), but it can shard a segment
into ~17 smaller ones. Same protocol, adding `CGC_CB_N_MAIN=1;CGC_SERVER_N_CB=16` (`/tmp/nsm_shard`,
2 launches, 149 family steps, 96 206 buffers):

| read | unsharded (18:14) | sharded (18:36) |
|---|---:|---:|
| distinct buffer compositions | 28 | **136** |
| count-matrix rank | 20 of 44 | **36 of 44** |
| kinds held out (< 3 signatures) | 26 | **10** |
| nodes per buffer (`nk`) | 1, 3-5, 45, 64 | **1-7** |
| composition-median R2 | 0.9197 | 0.9234 |
| `ffn_moe_topk`: fitted vs direct | -- | **675.0 vs 645.9 us (n=1600, 4.5%)** |

The falsifier from §8c is **gone** on the largest directly measurable kind: before, `ffn_moe_argsort`
was fitted at -407 us against a direct +13.5; now the biggest kind agrees with its model-free
measurement to 4.5% on 1600 observations. That is identification progress, and it is the shard doing it
-- not a better solver, not more data.

**The marginal still does not close, and the reason is now specific rather than structural.** Same
matrix, widths 2 -> 4: rows + unident + offset = the fitted slope **+4.70** (arithmetic exact),
against a measured **+17.80 ms/token**. Two named terms:

* `(per-buffer) +7.50` -- the fit's own offset column times the growth in buffer COUNT, i.e. **42% of
the measured marginal is per-command-buffer overhead**, not node work. Sharding buys identification and
pays for it in exactly this term.
* `+13.10` measured minus fitted -- the remainder, and it has a named cause: the model prices a kind per
  *count*, while a MUL_MAT's cost depends on its **shape**, which changes with the token count at
  constant count. A count-only design matrix cannot carry that, so an op-level attribution of the
  *marginal* needs shape-keyed kinds (op x shape), an emitter change (the kind vocabulary is names,
  `ggml-backend.cpp:2100-2109`).

So: **levels, yes**, with a direct-measurement check that now passes; **the marginal, no** -- and for a
third, smaller reason than §8c's two: a per-buffer floor plus shape-blind kinds.

**The two prerequisites this leaves, both measured:**

1. **The encoder must shard the segment so different buffers hold different node subsets.** Today it
   emits only 28 compositions, so 24 kinds never vary independently. `CGC_CB_N_MAIN=1` +
   `CGC_SERVER_N_CB<=16` is the knob that exists (run_server.sh:1735-1750); `n_cb=127` deadlocks
   Metal's in-flight limit, so one-node slices are not reachable that way, and a per-kind grouping
   would be an encoder change.
2. **The step-level aggregation must be fixed before any closure against the span.** The *per-layer*
   `CGC-DECPROF all: Lxx gpu=/union=` lines are sane ns-derived values, but the step's `gpu_sum` is
   `dp_lay_gpu[]` summed while those arrays receive **nanoseconds** from `sg_busy`
   (`ggml-backend.cpp:2580-2582`) and the printer treats them as microseconds -- measured 2.57e8 ms in
   one step against 41 per-layer lines at ~9 ms each. `--nsm`'s device gate therefore refuses the
   union slope (median union/wait = 1.2e6 on 92% of steps) and closes against NSM's own durations.

Two rule defects were found and fixed by this run (both are fixtures in `--selftest` now):

* **the offset column is per BUFFER row**, and the buffer count grows with the width -- so a 1 us
  per-buffer level shift is worth **+0.0025 ms/token** of the marginal (measured on the fixture).
  Left in the residual it would have been a silent attribution; it now prints as its own
  `(per-buffer)` row, and rows + unident + offset = the fitted slope *exactly*.
* **a launch offset must not move `t_k`** -- it does not (fixture: t stays 10 000 ns with a 1 us
  offset), which is what makes pooling k1 and k3 into one matrix legitimate at all.

The device-clock tail is unusable on this build (median union/wait = 1.2e6, 92% of steps > 1.10x), so
the closure denominator is NSM's own duration sum, which is what the rows partition -- and that sum is
**74.1%** of the family's host wall (the rest are buffers with no nodes, which NSM does not print).

## 9. The round's budget: rest = F + m*T, and what amortising F actually buys

`§3` read the delivered contrast (+8.97 ms/token from k=1 to k=3) as amortisation of the round's
fixed work. This is that sentence as arithmetic, on the same product (`--k-econ`, which reads the
sweep's own `arms.jsonl` or a `k_sweep.json`; `draft_ms` and `rest = round - draft` are both
measured, so the model has no free term beyond F and m):

| arm | T | E = emitted/round | round ms | draft ms | rest = F + m*T | delivered ms/token | t/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| k1 | 2 | 2.000 | 255.0 | 13.2 | 241.8 | 127.5 | 7.84 |
| k3 | 4 | 3.031 | 369.1 | 29.5 | 339.6 | 121.8 | 8.21 |

* **fit**: `m = +48.87 ms per verify token`, `F = 144.1 ms per round` -- 2 arms, **dof 0**, i.e. this
  describes the pair and cannot test it. It is quotable next to the independent within-arm slope
  (`+51.0`, three widths x three launches, no cross-arm assumption): **4.2% apart**.
* **amortisation**, quantified: `F` is **56% of the k1 delivered cost** (72.0 of 127.5 ms/token,
  because 144.1 ms is paid once per E = 2.000 tokens). That is the entire reason k is a lever -- and
  the same sentence as "+8.97 delivered against +51 within the round".
* **break-even**: k3 beats k1 while `E > 2.894` (measured 3.031 -> margin **+4.7%**).
* **the degree of freedom gets spent** on the 09-21 three-arm sweep (`k_sweep.json`, k=1/2/3):
  `m = 61.3`, `F = 56.9`, **dof 1** -- and the two intervals read **+51.1** (T2->T3) and **+71.5**
  (T3->T4). The line is **convex**, so a single m is a description of two endpoints, not a law. In
  that regime k3's break-even reads `E > 3.407` against a measured 3.197 (**-6.2%**: it loses), while
  its perfect-accept ceiling is the best of the three (82.5 ms/token = 12.1 t/s).
* **ceiling at perfect accept** (E = T, the number that bounds any accept-only programme): 09-22 k1
  127.5 (7.8 t/s), k3 92.3 (**10.8**); 09-21 k1 96.9 (10.3), k2 84.0 (11.9), k3 82.5 (**12.1**).

**The necessary condition for 25 t/s**: `F + m*T <= E*target - draft`. At the measured E,

```
09-22 k3: budget 91.7 ms  vs measured rest 339.6 ms   -> gap 247.8
09-22 k1: budget 66.8 ms  vs measured rest 241.8 ms   -> gap 175.0
```

and even at perfect accept (E = 4, the same round) k3 delivers 92.3 ms/token = 10.8 t/s. So the gap
is not acceptance and not the fixed work: it is `m*T` -- the round's own verify work -- of which 56-61%
is the device span. Which is exactly the quantity §8 tried to attribute and could not (8b).

### 9b. The k decision as a function of the accept rate

Two of the three terms are measured (the fit `rest = F + m*T`, and the draft chain `D(k)` through the
arms' own `draft_ms`); the third has to be modelled, and it is the accept rate:

```
E(k, a)         = 1 + a + ... + a^k              a verify token survives only if its predecessors did
delivered(k, a) = (D(k) + F + m*(k+1)) / E(k, a)
```

The arms' own `E` is deliberately NOT used as `a`: on llama-bench it sits at its ceiling (the k1 arm
emits E = 2.000, i.e. a = 1.0) because the synthetic cell repeats tokens. `--k-econ` prints the curve,
the argmin runs, and what accept buys.

**The two anchors, measured** -- inside a round an extra verify token costs **m = +52.6 ms**, while
the **delivered** contrast k3 vs k1 is only **+10.2 ms/token**; the difference is F spread over
E = 2.00 -> 2.90 tokens. That gap is the whole economics of k.

| product | fit | k*=1 until | k*=3 from | ceiling at perfect accept |
|---|---|---:|---:|---:|
| `/tmp/nsm_ksweep` (18:14) | F = 105.8, m = +52.62, D = 7.3 + 6.55k | a <= 0.661 | a >= 0.798 | k=8 -> 14.1 t/s |
| `spec_onoff/k_sweep.json` (09-22) | F = 56.9, m = +61.3, D = 4.1 + 6.88k | a <= 0.803 | a >= 0.892 | k=8 -> 13.5 t/s |

* **At the deployed accept (0.5825) the model says k = 1** -- and the measured arms agree (k1 98.7 vs
  k3 108.9 ms/token on the 18:14 sweep; the earlier pair showed the same order). Wide verify only
  starts paying above **a ~ 0.80**, i.e. above where the accept programme has been living.
* **25 t/s is unreachable at ANY accept rate** on both products: with these F and m the best arm stops
  at **13.5-14.1 t/s** even at a = 1.0. The budget gap (117-230 ms/token against the 25 t/s budget) is
  not acceptance, not the fixed work, and not the dispatcher: it is `m*T`.
* **The curve inherits the fit's own weakness**: dof 1, and the intervals are non-monotone
  (+132.1 then -26.9 per token on the 18:14 arms) -- so the line is a description of three measured
  arms, not a law, and the k thresholds move by ~0.10-0.14 of accept between the two products.

## 10. Reproduction

```
# the pair (09-22 build, sane tail): slopes + arm contrast
python3 scripts/check/verify_marginal.py --logdir /tmp/verify_layers --arm-t1 k1 --arm-t4 k3 --steps
# one arm only (no pair needed): the split + the unit gate
python3 scripts/check/verify_marginal.py --arm-dir Backup/prod_matrix/<run>/  [--phase VERIFY]
# §8 the span per op (refuses on the two gates; prints the composition either way)
python3 scripts/check/verify_marginal.py --node-attr Backup/phase_decomp/L3/dense_bw/<arm>.stderr.log
# §9 the round budget AND the accept curve, from either product this line writes
python3 scripts/check/verify_marginal.py --k-econ /tmp/verify_layers/arms.jsonl
python3 scripts/check/verify_marginal.py --k-econ /tmp/nsm_ksweep/arms.jsonl
python3 scripts/check/verify_marginal.py --k-econ Backup/phase_decomp/spec_onoff/k_sweep.json
# §8b the design matrix, SOLVED; comma-separated logs become one matrix (k1 log + k3 log)
python3 scripts/check/verify_marginal.py --nsm Backup/phase_decomp/node_attr/nsm_k1_r0_20260923.log,\
Backup/phase_decomp/node_attr/nsm_k3_r0_20260923.log
# §8d the same read on a SHARDED encoder (CGC_CB_N_MAIN=1 + CGC_SERVER_N_CB=16), which is what turns
#   28 compositions / rank 20 into 136 / 36 and makes the direct check agree
python3 scripts/check/verify_marginal.py --nsm Backup/phase_decomp/node_attr/nsm_shard_k1_20260923.log,\
Backup/phase_decomp/node_attr/nsm_shard_k3_20260923.log
# the phase identity (needs CGC_PHASE_DBG=1 in the arm)
python3 scripts/check/prod_matrix.py --profiles prod25 --cells decode-spec --reps 3 \
  --extra-env "CGC_SERVER_MTP_N_MAX=3;CGC_DECODE_PROFILE=1;CGC_DECODE_PROFILE_ALL=1;CGC_GPU_TIMING=1;CGC_PHASE_DBG=1;CGC_MTP_PERF=1"
```

Window and provenance: both artefacts record `window_before/window_after`, `launch_thermal` (k1 reps
NOMINAL/MODERATE/MODERATE; k3 reps HEAVY/HEAVY/HEAVY — the k3 arms ran hot, which is why the device column
is quoted across three reps rather than from one), the engine digest per dylib, and the pool counters
(hit 85.7-88.7%, misses 10519-14510 per arm).
