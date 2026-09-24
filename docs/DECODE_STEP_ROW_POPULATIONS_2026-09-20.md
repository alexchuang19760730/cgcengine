# Decode step rows come in pairs — and two reported numbers were artifacts of pooling them

Date: 2026-09-20. Evidence: `Backup/cgc_logs/llama_server_20260920_023021.log` (MTP on) and
`..._023202.log` (MTP off), read through `scripts/check/mtp_accept_ab.py` after this change.

## 1. The defect

A `CGC-DECPROF` log with `CGC_GPU_TIMING=1` writes **two step rows per verify round**: one row that
carries the round's real work, and one row that carries almost none. On the 09-20 MTP-on log, all 298
step rows split into exactly two populations with **no overlap**:

| population | discriminator | n | `total` span |
|---|---|---|---|
| work | `gap_sum > 0` | 128 | 111.36 – 1309.63 ms |
| shadow | `gap_sum == 0` | 170 | 0.68 – 12.98 ms |

`parse_step_profile` pooled them. At the steady width `ntok=4` that produced **62.17 ms** — a value
that sits *between* the two modes and describes neither, because the mode width's rows are 107 work +
107 shadow. The real round is the work row.

## 2. What this retracts

**Retracted: "steady state is already 54–62 t/s."** That came from the pooled median (62.17 ms) divided
by `mean_len` 3.375. Corrected:

| | pooled (old) | work rows (this) |
|---|---|---|
| `ntok=4` round | 62.17 ms | **247.98 ms** |
| implied steady | 54.29 t/s | **13.61 t/s** |
| `cb` | 0.53 ms | **74.18 ms** |
| `wait` / `submit` | 59.02 / 2.91 | 159.77 / 10.09 |

The 247.98 ms agrees with the figure G7's own evidence block carried (`step 247.98 ms, n=107`), so
that number was right and this parser was wrong.

**Retracted: "the delivery/steady gap is 4.8×."** It was the artifact divided into a delivered rate.
Corrected, from the same product:

| arm | steady (work rows) | delivered, token-weighted | delivered/steady |
|---|---|---|---|
| MTP on | 13.61 t/s | 9.18 t/s | **1.48×** |
| MTP off | 7.74 t/s | 7.36 t/s | **1.05×** |

## 3. A second defect, in the delivery column

`decode_tps_mean` is `sum(per-request rates) / n_requests` — an **unweighted mean of rates**, which is
not a throughput. On the same run it reads **11.27 t/s** while the 8 requests delivered **321 tokens in
34.97 s = 9.18 t/s**, because a 6-token request scoring 19.19 t/s weighs as much as a 96-token request
scoring 10.68. Ratio **1.23×** (MTP off: 1.29×).

`decode_tps_weighted` is the throughput (`Σ tokens / Σ decode seconds`, decode seconds from the
engine's own `predicted_n / predicted_per_second`), and `decode_time_weighting` records both plus the
ratio, so the historical field cannot be read as a throughput by someone who was not told.

## 4. What the corrected round says, and what it does not

MTP-on steady round, `ntok=4`, `mean_len` 3.375, n=107 work rows:

```
total   247.98 ms   = wait 159.77 (64%) + cb 74.18 (30%) + submit 10.09 (4%)
layer totals (overlapping spans, so NOT additive with `total`):
        gpu_sum 181.56   union_sum 149.24   gap_sum 95.17
tokens  3.375  ->  73.5 ms/token  ->  13.61 t/s
target  25 t/s needs step <= 135.00 ms at this mean_len, or mean_len >= 6.199 at this step
```

`cb` is the largest **named, serial** term, and it is host-side work on the critical path — not GPU,
not I/O. It also **scales with tokens**: 19.61 ms at `ntok=1` (MTP off) vs 74.18 ms at `ntok=4`, a
ratio of 3.78 for an `ntok` ratio of 4. So widening the verify batch does not amortise it.

**Do not read the layer sums as a budget.** `union_sum`/`gap_sum` exceed `total` in this regime
(union/total 1.13–1.15) because the 41 layer spans overlap; `gap_sum` 95.17 is a sum across layers, not
95 ms of dead time added to the 248. The claim this file stands behind is only the additive one:
`wait + cb + submit = total`, with `cb` on the serial path.

## 5. Consequence for the plan

- The 09-20 premise "attack the delivery/steady gap (1.48×) without touching numerics" survives, but
  it is **1.48× on MTP-on and absent on MTP-off** (1.05×) — so it is not a general per-request
  overhead; it is specific to the faster configuration, where a fixed per-request cost is amortised
  over fewer ms/token.
- The 30 % `cb` term is the reason `gap` exists at all on the work rows (`gap_L ≈ cb_{L−1} + 0.3 ms`,
  G1's own mechanism, and 74.18 + 40×0.32 ≈ 87 ms vs 95.17 measured). Any plan that treats `cb` as
  "5 % of the step" is reading the shadow rows.

## 6. Guards added (scripts/check/mtp_accept_ab.py, selftest 74/74)

- `parse_step_profile` splits work/shadow mechanically on `gap_sum == 0` **only when the log actually
  carries a gap** — a NO-TIMESTAMPS row also reports `gap_sum=0.00` while carrying real work, and the
  existing selftest caught that over-wide rule on the first run. Without GPU timing the split is
  disabled and `split_note` says the median may pool two populations.
- Reported: `n_work`, `n_shadow`, `shadow_median_total_ms`, `gpu_timing_present`, `split_note`, and
  `artifact_pooled_median_ms` (what the old code would have said at this width — 62.17 here — so
  reports written before this change can be told apart from ones written after).
- `--render` re-reads any product whose stored `step_profile` predates the split, prints both values,
  and recomputes `arithmetic`; a product that cannot be re-read is flagged, not silently reprinted.
- MTP-off logs are unaffected: 60 work rows, 0 shadow rows. `decode_step_profile.py` (line B's tool)
  reads only `ntok==1` rows and is not exposed to this.

## 6b. Independent confirmation of the split (the closure test)

M3's own document states the check that has to hold if the rows are read correctly:
`union_sum + gap_sum ≈ step total` (the step is "GPU working + GPU waiting"). On the 09-20 MTP-on log:

| rows used | union + gap | step total | closure |
|---|---|---|---|
| **work only** | 149.24 + 95.17 = **244.41** | 247.98 | **1.4% off** |
| pooled (old) | 71.73 + 5.48 = 77.21 | 62.17 | 24% off |

So the split is confirmed by a relation it was not fitted to, and the old pooled read did not close.
The 09-17 M3 table (`M3_M4_STATUS_2026-09-17.md` §2, MTP off, 8 steps) also closes (151.6 vs 148.4,
±2%) and is therefore **not** affected: MTP-off logs carry no shadow rows at all (60 work / 0 shadow
on 09-20's own off-arm log), so its `cb 56.54` is a real measurement in a different regime rather
than the same defect.

## 7. Not established here

- **Which graph the shadow row is.** It has the same `ntok` as its work partner and ~1/100 of the
  work. Guessing it is the draft head, the bonus token, or a metadata-only eval would be exactly the
  kind of unfalsified mechanism this line keeps having to retract. The parser deliberately does not
  assume; it only refuses to average the two.
- **Whether one work row is exactly one round** at every width. Request-level check does corroborate
  it (request 2: 30 work rows ↔ 96 tokens at `mean_len` 3.39 ≈ 28 rounds, and their summed time 8.63 s
  vs its 8.99 s delivered decode), but that is one request, not a proof for the population.
