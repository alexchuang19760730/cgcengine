# The k cost curve, measured — and where the per-token cost actually lives

**Date** 2026-09-22 · carrier Nail IQ3_XXS-denseIQ4X, profile `prod25`, MTP on, 8 GiB pool ·
engine `efeade7e2`, llama-bench `639a88bf1411dd68`, `libggml-metal` `8b61dfa206882d93` ·
tool `scripts/check/mtp_k_sweep.py` (39 self-test checks) · raw records `/tmp/mtp_k_sweep/arms.jsonl`
(15 records, provenance per record).

## 0. One line

The per-draft-token cost is **12% the MTP head and 88% the target verify path** — measured with
dof=2 instead of asserted from a zero-dof two-point fit — and the one-var sweep that produced it
also shows this instrument **cannot rank k** for production, because its accept rate is 1.000
where the server's is 0.654.

## 1. What was measured, and through which path

`docs/DRAFT_BUDGET_K_2026-09-22.md` fitted `cost(T) = C + c_tok*T` on two points from two
different instruments (llama-bench OFF and server CGC-DECPROF k=3): two parameters, zero degrees
of freedom. It also predicted that the best k is 1, not 3.

This run supplies the missing points with **one** knob on **one** harness in **one** window:

```
CGC_SERVER_MTP_N_MAX={k}  ->  run_server.sh:436 SPEC_DRAFT_N_MAX  ->  :1243 --spec-draft-n-max
```

Verified before spending the window: the dump prints `ARG 1` for k=1 and `ARG 3` by default, and
`ENV CGC_MTP_PERF=1` survives into the bench env (`llama_bench_matrix:231,382`). So k is a config
knob, not a code change — nothing in the engine was touched.

Arms, 4 k-points + an OFF reference, 3 reps with the arm order rotated per rep:

| arm | cell | batch | spec |
|---|---|---|---|
| k1..k4 | `decode-spec` (`p0 n128 d512 r3`) | `-b 8` | `--spec-type draft-mtp --spec-draft-n-max k` |
| off | `decode` | `-b 512` | none — **not a fit point** |

The split comes from the engine's own counter line, not from a model:

```
CGC-MTP-PERF type=draft-mtp calls_begin=3 calls_draft=192 … emit_tok_per_round=2.000 ms_per_round=10.309
```

with `round_ms = 1000 * emit_tok_per_round / tps` and `residual = round_ms - t_begin/calls_draft -
t_draft/calls_draft`. Cross-checked: the k=1 arm's `calls_draft=192` = 3 reps × 64 rounds for 128
tokens at E=2 ⇒ the counters are cumulative over the whole bench process, so the last line is the
run total (as `common/speculative.cpp:2986` states).

## 2. The cost law

```
per-rep fit  round_ms = C + c_tok*T      (T = k+1, spec arms only, same batch/env)
  rep 1: C=  28.43  c_tok=  66.47  n=4 dof=2  r2=0.980  maxres=17.3 ms
  rep 2: C=  92.43  c_tok=  69.56  n=4 dof=2  r2=0.902  maxres=39.5 ms
  rep 3: C=  19.39  c_tok=  98.54  n=4 dof=2  r2=0.998  maxres= 8.0 ms
  MEDIAN: C=28.43 ms   c_tok=69.56 ms per extra token
  POOLED (drift-contaminated, for contrast): C=67.38  c_tok=74.89  r2=0.929
```

`C` is badly determined (19–92 ms) because the four x-values only span T=2..5 and the intercept
is an extrapolation to T=0; `c_tok` is stable at 66–99 ms. **Both differ from the prediction:**

| | predicted (DRAFT_BUDGET_K) | measured (median of per-rep fits) |
|---|---:|---:|
| `c_tok` | 51.56 ms | **69.56 ms** (+35%) |
| `C` | 41.74 ms | 28.43 ms |
| OFF step | 93.3 ms (server instrument) | **103.8 ms** here (+11%, cross-config) |

## 3. The result that matters: where the per-token cost lives

`DRAFT_BUDGET_K` §3 left this as an open fork — *"is the per-draft-token cost in the draft forward
or in the verify per-token path?"* — and the plan was an n-gram ablation (`--spec-type
ngram-map-k`) to decide it. The engine's own counter already carries the split, so no ablation was
needed. Slopes of each component against k:

```
  draft head forward                      :  +8.35 ms per extra k   (r2 0.98, dof 2)
  residual (target verify + host)         : +61.21 ms per extra k   (dof 2)
```

**88% of the marginal token is the target verify path; 12% is the MTP head.** And the head's 8.35
ms is at its own kernel floor — the `mul_mat_id` family measured 8.36 ms/token of GPU time
(`docs/SHAPE_ROOFLINE_2026-09-22.md`). So on this axis the head is not the problem and cannot be
made much cheaper; **the money is in the verify path**, which is exactly the `c_tok` term the
roadmap's F1/M3 work has been chasing from the other end.

This also retires a prediction: DRAFT_BUDGET_K expected the kernel to be ~16% of `c_tok`. It is
~12% of it — the claim survives, and slightly strengthens, on an independent instrument.

## 4. A confound the cost model did not have: MTP runs the pool colder

Per-arm expert-cache teardown counters (same engine, same pool size):

| arm | hit rate | misses |
|---|---:|---:|
| off | 96.3–96.7% | 4,220–4,827 |
| k1 | 87.3–87.7% | 12,230–12,873 |
| k2 | 86.3–86.7% | 13,236–13,848 |
| k3 | 85.6–87.4% | 12,411–14,584 |
| k4 | 85.0–85.6% | 14,672–15,437 |

**The MTP arms take ~3× the misses of the OFF arm**, and misses grow with k. So MTP's price is not
only "one extra token costs `c_tok`" — it is also that draft+verify touch far more distinct experts
and evict the resident set. Any future `c_tok` model that does not carry a pool-churn term is
missing a measured effect of the same order as the term it does carry.

## 5. What this instrument cannot decide (the pre-registered predictions, read honestly)

**P1 — "best k is 1" — NOT TESTED here.** By median t/s the ranking on this instrument is
k3 (0.994× OFF) > k1 (1.020×) > k2 (0.930×) > k4 (0.815×), i.e. **MTP at best breaks even** on
llama-bench's workload. But the ranking is a function of the accept rate, and:

**P2 — measured α from the k=1 arm = 1.000** (model-free: `α = E-1`, and E = 2.000 exactly in all
three reps — every single draft accepted), against **0.654** from the server regime that P1 was
predicted for. llama-bench has no prompt, so its α is a different quantity. The prediction was
about the production regime; **it is therefore untested, not falsified** — and the reason is now a
number rather than an excuse.

**P5 — cross-config sanity check.** The fit's intercept implies nothing useful at T=1 (dof-poor),
and the OFF arm runs `-b 512` with no `--spec-type` at all, so its 103.8 ms is a reference datum,
not a fit point.

## 6. Limitations, stated rather than discovered later

- **Inter-rep drift up to 1.5×** swamps the k ranking (k1: 12.38 / 8.82 / 9.40 t/s). Every number
  above is therefore computed *inside* a rep; the pooled fit is printed beside it precisely so the
  two are not confused.
- **Swap was carried at ~5.0 GiB throughout** (4,950–5,109 MB, never below 4.9 GiB). Absolute t/s
  here must not be placed beside the 12.57 family; the same-rep ratios can travel.
- **Within-arm drift is not removed by dropping rep 1** on this box: k3 rep1's *first* sample was
  its fastest (15.32 t/s) and later samples fell.
- **E is an all-rep average while the quoted t/s is a reps-2..3 median.** The mismatch is small
  (the counters span the process) but it is a real difference in denominators.
- **`t_begin ≈ 0.0` in every arm.** Reported as measured; whether the refresh is genuinely free or
  the counter is not accumulated is not something this run can distinguish.
- **No M1/M2/M3 oracle gate was run for this sweep** — deliberate, and the reason is the shape of
  the change: one env var, no code, so nothing that decides numerics moved. The digest is recorded
  per arm anyway, and it equals the post-rebuild clean-HEAD build.
- **k4's rep-1 E (2.825) sits well below its rep-3 value (3.425)**: acceptance genuinely varies per
  rep, and averaging E over reps flattens that.

## 7. Artifacts

- `scripts/check/mtp_k_sweep.py` — 39 self-test checks (parser, derivation, fit dof, rotation,
  pool stats, absence-is-not-zero on every axis). Refuses rather than fabricating on: <2 samples,
  missing `CGC-MTP-PERF`, `calls_draft=0`, `E<=1`, a missing field, `n<2` for a fit. Reports `r2`
  as `None` when dof=0 instead of the 1.00 a 2-point fit always produces.
- `/tmp/mtp_k_sweep/arms.jsonl` — 15 records: samples, E, round/draft/residual, pool counters,
  window before/after (usable%, swap), thermal labels, engine digest.

**Not committed. No engine source changed. No server launched. Zero processes left** (`pgrep`
`llama` = 0; verified at the end of the sweep).

## 8. What this buys

The cheapest next step is no longer another k sweep: it is that **`c_tok` is a verify-path
number**, so the four-way split the roadmap wanted (draft / verify / gather / sync) is now a
one-sided problem — the draft side is 8.35 ms and already at its kernel floor, and 61 ms/token sits
in the target's verify round. Two consequences worth acting on:

1. **The n-gram ablation is now a cross-check, not a discovery step.** It uses the same counter, so
   if the counter is wrong both are wrong — that is the boundary, and it is the only reason to
   still run it.
2. **Pool churn belongs in the cost model.** MTP's misses are ~3× OFF's. Pairing `c_tok` with a
   residency term is what would make the 3.4× in-situ residency penalty and the `c_tok` term
   comparable instead of additive-looking.
