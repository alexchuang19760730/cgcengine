# A paired protocol for 3%-class comparisons: launch index as a covariate

**Date** 2026-09-23 · tool `scripts/check/bracketed_ab.py` (selftest 22/22) · applied to the 10
certified-window arms of `docs/CERTIFIED_WINDOW_K_AB_2026-09-23.md` · engine `b70ff34a8bc69441`

## 0. One line

Launch index is a real, measurable covariate and it can be **removed** rather than averaged: adding
`order` to a log-space fit turns the k2/k3 reading from *unseparated* into **−15.1% (t = −2.51, 27% of
the arm-only residual variance removed)** — and adding swap and gap alongside it gives **−15.3%
(t = −2.33, 41% removed)** — while the simulator shows that on this box's measured scatter a 3%-class effect needs
**≈110 alternating launches with the regression** (≈430 with a bracket), and that the naive pairing
protocol I used earlier reports **+8 to +21 percentage points of pure drift** as if it were an effect.

## 1. What the covariate is (measured, not assumed)

From the 10 certified-window arms (all launched at thermal Nominal, one engine digest):

| arm | n | mean | sd | spread | ρ(order, tps) | n=5 floor |
|---|---:|---:|---:|---:|---:|---:|
| k=2 | 5 | 12.13 | 0.58 (4.8%) | 11.21–12.83 | **−0.15** | 0.90 (not met) |
| k=3 | 5 | 10.43 | 1.56 (14.9%) | 8.59–12.25 | **+0.90** | 0.90 (at the floor) |

The striking part is that **the scatter is arm-specific**: 14.9% against 4.8% at bit-identical work, and
the regression's covariate term removes 27–41% of the residual variance. The *rank correlation* is a
weaker statement than the earlier draft of this section made it: at n=5 a ρ needs to reach 0.90 to be
worth anything, so k=3 sits exactly on that floor and k=2 is not distinguishable from zero — and k=2's
value is not even stable, because two of its five readings are the same number (12.20, 12.20), so
ranking that tie in input order instead of by average rank moves ρ from −0.15 to −0.10. (That is why
`bracketed_ab.py` computes ρ itself with average ranks and prints the floor next to it.)

Two consequences the protocol has to live with:

1. A *single* common slope (one `order` column) is a partial model: it removes 27% of the residual
   variance on this data, and 41% once swap and gap-since-previous-launch are added (thermal is a
   label — the tool excludes it with a message rather than coding it as 0). The rest is in an
   **arm × order interaction** that this design cannot estimate — which is also why the honest
   statement is "the covariate is removable to first order", not "the drift is understood".
2. It is *not* thermal-at-launch: every one of these arms was Nominal (that is enforced), and it is
   not swap (the lowest-swap arm was among the slowest). The carrier of the k=3 instability is still
   unidentified, and no covariate can correct what cannot be measured.

Recorded per launch, and all of them at once: `arm`, `order`, `tps`, `gap_s`, `swap_mb`, `thermal`,
and the pool `partition` (measured earlier to differ across launches of one configuration at 33 vs 36
segments). A label covariate is **excluded with a message, never coded as 0**.

## 2. The protocol

One window, one engine digest, one thermal regime, and an **alternating** sequence:

```
R T R T R T R T R ... R        R = reference arm, T = test arm, odd length
```

* Alternation is what makes the arm effect **orthogonal to the sequence index**. In the ABBA
  rotation used earlier (A B B A B A A B …) they are *not* orthogonal, and a test arm never has the
  reference on both sides — the tool reports `no contrast` on that data, i.e. **the design I ran had
  no brackets at all**.
* Every launch is gated (thermal Nominal; engine digest; pool geometry recorded).
* Same wall-clock budget per arm, and the gap between launches fixed or recorded.
* Minimal useful length: **6 blocks (13 launches)**; beyond that the resolution improves as 1/√n.

Four estimators, all reading the same rows, in fractions of the reference level (a t/s difference is
meaningless across a window whose level moves 2×):

| estimator | what it is | what it survives |
|---|---|---|
| `paired` | consecutive (R, T) log-ratios | **nothing** — kept only to show the drift it carries |
| `bracketed` | `T / sqrt(R_before · R_after)` per block, averaged in log space | a geometric drift, **exactly**; any smooth monotone drift to second order |
| `bracket-arith` | same, with the arithmetic mid | a drift linear in the index, exactly — its disagreement with the line above **is the protocol's own model error** |
| `adjusted` | OLS of `log tps` on `[arm, order, covariates…]` | a linear-in-index drift in log space, and uses every launch (most efficient) |

**Pre-registered decision rule** (all five, or the effect is not quoted):

1. the two bracket forms agree within **1 pp**;
2. `adjusted` agrees with `bracketed` within their CIs (they are biased by *different* misspecifications,
   so agreement is evidence);
3. the CI95 of the pooled estimate excludes zero;
4. every arm launched at Nominal, one engine digest, and the pool partition recorded;
5. the per-arm **minimum** is reported next to the median as the one-sided-contention upper bound.

## 3. Why the previous protocol could not have answered this

Simulated, 6 blocks (13 launches), true effect +3%, 400 repeats, per-arm noise 5%, seed 7 (`--seed` is
deterministic, so these are the numbers the commands in §8 print):

| drift per launch | estimator | apparent effect | bias |
|---|---|---:|---:|
| none | paired | +2.84% | −0.16 pp |
| none | adjusted | +2.84% | −0.16 pp |
| **8% geometric** | **paired** | **+11.07%** | **+8.07 pp** |
| 8% geometric | bracketed | +2.82% | −0.18 pp |
| **20% geometric** | **paired** | **+23.41%** | **+20.41 pp** |
| 20% geometric | bracketed | +2.82% | −0.18 pp |
| 20% geometric | bracket-arith | +1.00% | **−2.00 pp** |
| 20% **linear** | adjusted | +5.20% | +2.20 pp |

(The seed was pinned *after* this table was first written; the pre-seed version of these rows differed
in the third digit and, being unreproducible, was not quotable — a measurement product whose own
command does not reproduce it is the defect this repo keeps catching.)

Two things to read off it: the paired estimator is not a weak instrument, it is a **wrong** one whose
error is entirely the drift; and the bracket's second-order model error grows to the size of the
effect (2 pp) once the drift reaches 20%/launch — which is why the two bracket forms must both be
reported and compared.

## 4. Power: what this box can resolve, and at what cost

Calibrated to the **measured** scatter (10% per arm, i.e. between k=2's 4.8% and k=3's 14.9%), drift
8%/launch, one-sided contention on 20% of arms:

| blocks | launches | estimator | smallest true effect this window resolves at 80% | launches needed for a 3% effect |
|---:|---:|---|---:|---:|
| 6 | 13 | paired | 17.4% | **403** |
| 6 | 13 | adjusted | 17.2% | **106** |
| 6 | 13 | bracketed | 14.7% | **433** |
| 12 | 25 | adjusted | 12.6% | 111 |
| 12 | 25 | bracketed | 10.9% | 475 |

* **3% is reachable** — with the regression on an alternating sequence, ≈**110 launches** (≈2–3 hours
  of back-to-back launches, each ~70 s), or ≈430 with the bracket-only protocol.
* Resolution improves only as 1/√n past the point where the model error dominates: at 20%/launch the
  bracket's own bias is 2 pp, so **no amount of n buys below ~2 pp** for that estimator. The
  regression does not have that floor, but it is the one biased under a purely linear drift.
* Therefore: run the alternation, use the regression as the primary estimate, the brackets as the
  shape cross-check, and stop when the two agree — that is the resolution limit, not n.

## 5. Applied to the real 10 arms (the certified window's data, re-read with the covariate)

```
rows: 10 arms (k2 x5, k3 x5), order 1..10
  k2            n=5 mean 12.13 sd 0.58 (4.8%) spread 11.21-12.83  rho(order,tps) -0.15  (below the n=5 floor 0.90)
  k3            n=5 mean 10.43 sd 1.56 (14.9%) spread 8.59-12.25  rho(order,tps) +0.90
  paired        n=5   effect  -14.77%  per-block sd 0.1743  CI95 +-15.28pp
  bracketed     no contrast (the pattern does not bracket a test arm with references)
  bracket-arith no contrast (the pattern does not bracket a test arm with references)
  adjusted      n=10  effect  -15.08%  t -2.51  drift +0.01823/launch  order removed 27%
```

With `--covariates swap_mb,gap_s,thermal` the last line becomes `-15.30%  t -2.33  drift
+0.00797/launch  order removed 41%` plus the note `(non-numeric covariate(s) excluded rather than coded
as 0: thermal)`.

That `bracketed → no contrast` line is the finding about the *previous* design, not a missing number:
the ABBA rotation A B B A B A A B never puts the reference on both sides of a test arm, so the protocol
that was supposed to be drift-immune was never actually run — the five paired differences were the whole
of it.

What changed against `docs/CERTIFIED_WINDOW_K_AB_2026-09-23.md`: that report called k2 vs k3
**NOT SEPARATED** on the pre-registered "weakest round" rule, and that rule still says the same (one of
the five rounds is a tie). Pooling with the covariate gives **−15.1%, t = −2.51 (df 6, p ≈ 0.046)**.
Both statements are about the same ten launches; the difference is the model, and the honest summary is
**"a stable ~15% direction, now significant when pooled and adjusted, still short of a 3%-class
claim"**. The paired estimate lands on the same number (−14.8%), which looks like corroboration and is
weaker than it looks: §3 shows that pairing inherits the whole drift, and here the drift over the ten
launches (~+1.8%/launch in the order-only fit) is simply too small to separate them — so the agreement
says the *drift* is weak in this dataset, not that the two methods are interchangeable.

This is a **re-analysis, not a certification**: the pre-registered rule in §2 needs brackets and
there were none, so rules 1 and 2 could not be evaluated at all. What the re-analysis buys is a number
that can be quoted *as* a re-analysis (10 launches, one covariate, one model) and a measurement of how
much a covariate is worth on this data — not a licence to call k=2 vs k=3 settled. And note the
direction of the finding: the covariate makes a difference that was invisible to the pre-registered
"weakest round" rule *visible*, i.e. the adjustment can also promote a reading. That cuts both ways,
which is why the rule's brackets are required before any 3%-class claim, in either direction.

## 6. What to do when the target is below the floor

Three escalation steps, in order of return:

1. **Cut the per-arm scatter.** The floor here is k=3's 14.9%; at k=2's own 4.8% the regression would
   need **~42 launches** for a 3% effect instead of ~110 (and the bracket ~170 instead of ~430 — so
   even at the good arm's scatter the bracket stays the expensive estimator). The lever is not
   statistical: whatever makes the wide-verify arm swing 1.43× at identical work has to be identified
   (its own per-process step curve is measurable).
2. **Move the comparison inside one process** where the sequence covariate does not exist: the same
   server, requests alternated, internal counters (DECPROF step time, per-layer cb) as the metric.
   This is available for any runtime-togglable knob and for the *marginal cost of a draft token*;
   it is unavailable for a launch-level flag such as `CGC_SERVER_MTP_N_MAX`.
3. **Only then** spend the launches — ≈110 with the regression at today's mixed scatter (≈430 if
   bracket-only), or ≈42 once step 1 has brought every arm to the good arm's scatter — in one
   uninterrupted window.

## 7. Honest ledger

- The simulator's drift and contention distributions are *stylised* (geometric/linear drift, one-sided
  exponential contention); they are calibrated in magnitude to the measurements in §1, not fitted to
  them.
- The real-data refit is 10 rows against up to 6 parameters: `t = −2.33` with 4 dof is fragile, and the
  covariate columns (swap, gap) are collinear with order by construction in a sequence.
- `order` is a proxy. The thing that actually varies is unknown; a proxy can only remove the part of
  the variance it correlates with, which is what the 27–41% measures.
- The tool's `paired` estimator is deliberately retained as the wrong baseline; it must never be the
  one quoted.
- Nothing was launched for this document: it is the existing 10 arms re-analysed plus 400×5 simulated
  windows. No server, no residual processes, nothing committed.

## 8. Reproduce

```bash
python3 scripts/check/bracketed_ab.py --selftest              # 22/22

# §3: the two estimators under three drift shapes (per-arm noise 5%, seed 7 by default)
python3 scripts/check/bracketed_ab.py --simulate --blocks 6 --effect 0.03 --reps 400 \
        --drift geometric --drift-pct 8  --noise 0.05 --contention 0.2
python3 scripts/check/bracketed_ab.py --simulate --blocks 6 --effect 0.03 --reps 400 \
        --drift geometric --drift-pct 20 --noise 0.05 --contention 0.2
python3 scripts/check/bracketed_ab.py --simulate --blocks 6 --effect 0.03 --reps 400 \
        --drift linear    --drift-pct 20 --noise 0.05 --contention 0.2

# §4: what this box's measured scatter (10%/arm) can resolve, at 6 and 12 blocks
python3 scripts/check/bracketed_ab.py --simulate --blocks 6  --effect 0.03 --reps 400 \
        --drift geometric --drift-pct 8 --noise 0.10 --contention 0.2
python3 scripts/check/bracketed_ab.py --simulate --blocks 12 --effect 0.03 --reps 400 \
        --drift geometric --drift-pct 8 --noise 0.10 --contention 0.2

# §1 and §5: the covariate table and the four estimators on the real 10 arms
# (`arm,order,tps[,gap_s,swap_mb,thermal,partition]`)
python3 scripts/check/bracketed_ab.py --analyze \
        Backup/k_abba_2026-09-23/certified_window_10arms.csv --ref k2 --test k3
python3 scripts/check/bracketed_ab.py --analyze \
        Backup/k_abba_2026-09-23/certified_window_10arms.csv --ref k2 --test k3 \
        --covariates swap_mb,gap_s,thermal
```

The arm CSV is a copy of the one used here (that one lived in `/tmp`); `Backup/` is gitignored, so the
reproduction depends on `docs/CERTIFIED_WINDOW_K_AB_2026-09-23.md`'s arms being re-run, or on this file
being kept alongside them.
