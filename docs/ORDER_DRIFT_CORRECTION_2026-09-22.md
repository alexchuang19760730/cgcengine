# Order / drift correction for bracketed A/B cells — 2026-09-22

Trigger: "the noise problem could be the earlier measurement" — with the suggestion of an AB/BA
crossover combined as `sqrt(AB x BA)` instead of averaging as `(AB + BA)/2`.

The suggestion is right in kind and needed two changes to become usable here.

## 1. The plain language argument

A throughput is multiplied by the machine's state, not reduced by it: a hot box reads `rho * t/s`.
So `log y = log theta_c + log rho(t)` — the model becomes additive **only in logs**. Any mean or
interpolation taken on the raw values is therefore the wrong estimator, and the size of the mistake
grows with the drift.

Measured on tonight's own reference series `11.72 / 10.83 / 5.22`:

| row | refs used | arithmetic ref | geometric ref | gap | Δ with arith | Δ with geom |
|---|---|---|---|---|---|---|
| `cell_CGC_SHAPE_M4` | 11.72, 10.83 | 11.28 | 11.27 | 0.08% | −34.60% | −34.55% |
| `cell_CGC_SHAPE_M17` | 10.83, 5.22 | 8.02 | 7.52 | **6.75%** | +17.47% | **+25.40%** |

On the second row the choice of mean moves the headline by 7.9 percentage points — more than twice
the repo's 3% threshold.

**But that row is still not data.** Its two references disagree by 51.8%, above the harness's own
`MAX_DRIFT_PCT = 10%`, so the existing rule already rejected it as UNANCHORED. The correction shows
how large the artefact was; it does not promote the row to a measurement. This is the part that
matters most for how the numbers get quoted.

## 2. What was implemented

`scripts/check/crossover_estimate.py` — pure functions, nothing launches anything, so every claim is
checkable by planting a known effect.

- `ref_at(v_before, t_before, v_after, t_after, t_cell, mode)` — the reference **at the instant the
  cell ran**, interpolated in log space. Weights come from measurement times because a cell takes
  several times longer than a reference does, so the cell's instant is nowhere near the midpoint.
- `block_rc_cr(...)` — the mirrored block `ref, cell, cell, ref`: each config appears once early and
  once late, so a multiplicative **period** factor hits each exactly once and cancels in the product
  `sqrt(r_lead * r_trail)`.
- `mean_gap_pct(a, b)` — reported next to every row. Diagnostic, deliberately **not** a second gate:
  the existing drift rule already voids the rows where this number gets large.

## 3. What it removes and what it does not

Say it exactly, because overclaiming is the failure this repo keeps punishing.

- **Removes:** smooth drift, and a period effect (machine differing between the two halves of a
  block).
- **Does not remove:** a penalty attached to the *configuration* rather than to the position. That
  is indistinguishable from `theta_cell` genuinely being lower — no crossover can separate them. The
  self-test asserts this explicitly: a planted `cell_penalty=0.95` stays in the answer.
- **Does not remove:** carry-over left behind by whichever arm ran immediately before (pool,
  page cache, SLC). The two arms are therefore reported separately; if they disagree by more than
  10% the row is flagged rather than averaged into a number nobody can interpret.

Planted truth `theta_cell/theta_ref = 1.15` under a 45% drift (the size tonight's refs actually
moved) plus a 6% second-in-pair penalty:

| geometry | geometric | arithmetic (old) |
|---|---|---|
| cell 3x longer than a ref, symmetric block | 0.00% | 1.39% |
| cell 10x longer than a ref | 0.00% | **8.03%** |
| refs of unequal length around the block | 0.00% | 2.82% |
| **trailing ref ran 5x longer than the leading one** | **0.47%** | **0.07%** |
| mild 10% drift / no drift | 0.00% | 0.12% / 0.00% |

The fourth row is included on purpose: **the log-space estimator is not universally better.** When
one reference spans a large fraction of the drift, "the instant it ran" stops being a point, and the
arithmetic mid-point happens to land closer. Both were within 0.5% there, so this is a caveat and
not a counterexample, but picking one flattering geometry to prove the point is exactly what this
repo exists to prevent — hence the table rather than a single number.

`--selftest --broken` runs the old estimator through the same ground truth and **must fail**; it
currently misses by up to 8.03% (rc=1). A correction nobody can turn off is a correction nobody has
shown does anything.

## 4. Wired into the search harness

`scripts/check/shape_knob_search.py` now records `t_start`/`t_end` per arm and reads every cell
against a log-space, time-weighted reference (`--combine geometric`, the default; `arithmetic`
reproduces the old numbers when a previously published figure has to be compared). Each row carries
`ref_ts` (chosen mode), `ref_ts_alt` (what the arithmetic mean would have given) and `mean_gap_pct`,
and any row whose mean-choice gap exceeds 3% is called out at the end of the run.

Time-only change, no new space: bracketing still uses the same refs it already launched.

Self-tests: `crossover_estimate.py --selftest` 14/14, `shape_knob_search.py --selftest` 24/24.
