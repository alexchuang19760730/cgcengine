# F2 / F3 / F4 / F5 — results, and the reason F2's null is not about overlap

Date 2026-09-20. Rules were pre-registered in
`Backup/phase_decomp/F2_F5_OVERLAP_AND_AUDIT_20260920.conditions.md` before any launch.
Reproduce with `python3 scripts/check/f2_overlap_verdict.py` (self-test: 12 checks) and
`python3 scripts/check/cb_miss_regression.py --log <log>` (self-test: 21 checks).

| | verdict | one line |
|---|---|---|
| **F2** | **FAIL** (cb_B/cb_A = **0.971**) | …and the arm **never executed the treatment**, so it did not test overlap |
| **F3** | **CLOSED** | 220 MiB/s is a *per-job* rate; ×8 workers = **1760 MiB/s**, above copy speed |
| **F4** | **PASS** (4/4 arms) | warm-union bound holds: `median(cb|m=0)` = 0.010–0.020 ms vs bound 0.05 |
| **F5** | **REFUTED** | L unchanged 27.95 → 27.69 (−0.9%); the treatment removed no layers |

Plus one correction to my own F1 commentary, and one instrument-label defect that came within a
hair of flipping F3's verdict in the wrong direction.

---

## 1. F2 — the null, and what it actually says

Per the pre-registered rule (`PASS ≤ 0.50×`, `FAIL ≥ 0.75×`), protocol A,B,B,A:

| arm | cb (ms) | step (ms) | mean_len | accept |
|---|---|---|---|---|
| A1 | 65.66 | 227.13 | 3.34 | 0.781 |
| B2 (`LAYER_AHEAD=1`) | 67.06 | 229.69 | 3.36 | 0.788 |
| B3 (`LAYER_AHEAD=1`) | 66.13 | 209.20 | 3.31 | 0.769 |
| A4 | 71.54 | 220.51 | 3.55 | 0.850 |

`median cb_A = 68.60`, `median cb_B = 66.59` → ratio **0.971** → **FAIL**.

**But the treatment never ran.** Both B arms carry `CGC_LAYER_AHEAD_PREFETCH=1` in their live env
witness, and neither queued a single prefetch:

| witness | armB_dbg (8 GiB) | p1 (8 GiB) | p2 (4 GiB) |
|---|---|---|---|
| `prefetch=` successes/drops | `0/0` | `0/0` | `0/0` |
| `prefetch drop breakdown` line | absent | absent | absent |
| `PFDBG` lines **from `prefetch_slot`** | **0** | **0** | **0** |
| `PFDBG batch` lines (proves DBG was live) | 4357 | 3276 | 4443 |

Every `-1` return from `prefetch_slot` prints under `LLAMA_EXPERT_CACHE_PREFETCH_DBG`, and a success
bumps `prefetch=`. With the instrument demonstrably live and **zero traces of any shape**, the
function was never entered.

**⇒ F2's FAIL is a treatment-application null.** The only difference the engine could see between
arms was one `getenv()`. This is the same class as the earlier finding that `CGC_SERVER_MTP=0`
silently drops a seven-variable env block: an arm labelled "treatment ON" that never executes the
treatment. **The question "can `cb` be hidden behind GPU work?" is still untested.**

## 2. F2-probe — why the trigger is dead, and why it cannot be revived by a smaller pool

The trigger (`llama-context.cpp:6727`) fires only when
`prev_token_valid[next_l] && st_next[e] < 0`. Three arms isolate which conjunct fails. The
prev-token consumer (`:5655`) reads the **same** `prev_token_expert_ids` / `prev_token_valid` pair
and reports the two cases separately:

```
p1 (8 GiB, prev-token)  99 events  valid_layers median 40/41  resident 320/320  cold 0     31360 predictions -> 100.00% resident
p3 (4 GiB, both)       127 events  valid_layers median 40/41  resident 320/320  cold 52    40320 predictions ->  99.87% resident
```

* `valid_layers = 40/41` ⇒ the shared prediction structure **is fully populated**. This is **not** a
  collection bug, and it refutes the hypothesis that the trigger's precondition is dead.
* `resident = 320` of 320 ⇒ **every predicted expert was already in the pool.**

The predictor's target is *the experts the previous token used*. Those were demand-filled one token
ago, and the pool still holds them — so neither consumer has anything to queue. This is a property
of the geometry, not of the two flags:

| pool | slots/layer | per-token demand | over-provision |
|---|---|---|---|
| 8 GiB | 143 | 8 experts/layer | **17.9×** |
| 4 GiB | 71 | 8 experts/layer | **8.9×** |

(per-expert ≈ **1.40 MiB**, derived from both geometries independently: 8 GiB/143 and 4 GiB/71.)

**⇒ A predictor keyed on "what I just used" needs `slots/layer < 8`, i.e. a pool below ~0.43 GiB**
— one token's whole expert set (8 × 41 × 1.40 MiB = 443 MiB). That is below anything the loader can
start. So layer-ahead and prev-token prefetch are **structurally dead across the entire supported
pool range**, at both 8 GiB and 4 GiB, for the measured reason rather than by inference.

This also explains the 0.17% of predictions that *were* cold in p3 (52 of 40 320): they are the ones
quota-evicted since the previous token, i.e. exactly the experts the predictor is worst at naming.

## 3. F3 — CLOSED, but only after correcting an instrument label

Graceful teardown (the fix made last round) produced the evidence F3 needed:

```
read shape: jobs=27585 bytes=10594033664 (0.37 MiB/job as one contiguous run)  us/job=1748  effective_rate=220 MiB/s
```

Read literally, 220 MiB/s is inside the device band and **re-opens** F3 — which was my
pre-registered disposition. That reading is wrong:

* the engine's own comment states `pread_usec` is *"an aggregate over worker threads and therefore
  cannot be compared to a step at all"*;
* each job is timed **inside** a worker, so `effective_rate = bytes / Σ(per-job µs)`, and wall time
  ≈ Σ/W. `effective_rate` is a **per-job** rate under a throughput-sounding label.
* `LLAMA_EXPERT_CACHE_WORKERS` defaults to **8** (`run_server.sh:1345`), corroborated
  independently by F1's round-trip model fitting `ceil(3m/8)`.

**Aggregate = 220 × 8 = 1760 MiB/s ≥ 1000 MiB/s copy speed ⇒ CLOSED.** Two independent instruments
now agree: F1's marginal cost (0.695 ms per ~1.11 MiB = **1.6 GiB/s**) and F3's bytes/µs × workers
(**1.76 GiB/s**). Both exceed the documented device peak (~823 MB/s), so the bytes are not coming off
the device and **a fatter read has no headroom to win**.

`f2_overlap_verdict.py` carries this as a self-test on both directions: 226 MiB/s × 1 worker must
read RE-OPENED, × 8 must read CLOSED.

## 4. F4 — PASS on all four arms

| arm | `median(cb|m=0)` | n | bound |
|---|---|---|---|
| A1 | 0.010 ms | 1394 | ≤ 0.05 |
| B2 | 0.020 ms | 1271 | ≤ 0.05 |
| B3 | 0.010 ms | 1439 | ≤ 0.05 |
| A4 | 0.010 ms | 1535 | ≤ 0.05 |

The warm-union bound holds, and it is now a **gate** in `cb_miss_regression.py` (with a
must-fail test using a 0.9 ms "no-miss" layer) rather than a remembered number: any future change
that lowers `cb` without moving it toward this bound has not removed the mechanism.

## 5. F5 — REFUTED, and a correction to my own F1 commentary

Pre-registered: supported if the B arm reduces `L` and leaves `median(cb|m=1)` ~unchanged; refuted
if `L` is unchanged. **Read on work rows** — the same population `median_cb_ms` uses:

| arm | n | L mean | L median | fully warm | `median(cb|m=1)` |
|---|---|---|---|---|---|
| A1 | 102 | 27.6 | 29.0 | 4% | 1.17 |
| B2 | 95 | 27.9 | 30.0 | 4% | 1.25 |
| B3 | 104 | 27.5 | 29.0 | 4% | 1.14 |
| A4 | 118 | 28.3 | 30.0 | 3% | 1.15 |

`L` 27.95 → 27.69 (−0.9%), premium 1.16 → 1.19 ms ⇒ **REFUTED**.

**Correction to §1 of the F1 follow-up.** I read the `L` distribution off *all* step rows and
reported "58% of steps are fully warm". That was the **shadow-row artifact** again: with
`CGC_GPU_TIMING=1` every turn emits a work row and a near-empty shadow row, and pooling them
inflates the warm share from **3–4% to 58%**. On the work population, **~29 of 40 layers pay a
first-miss cost in a typical work step**. That strengthens rather than weakens F1's per-layer
barrier reading, and it is the same defect as `docs/DECODE_STEP_ROW_POPULATIONS_2026-09-20.md`
transplanted into a new tool. The split is now implemented **in the parser** so every consumer has
it, with a self-test that fails if the two populations are conflated and another that fails if an
uninstrumented log is silently called "shadow".

## 6. What this leaves on the table

* **`cb` ≈ 66–72 ms of a ~220 ms step.** F4's bound says the warm cost is ~0.4 ms/step, so the
  mechanism is real and large. Nothing in this round moved it.
* **No available predictor can supply the prefetch.** The only prefetch-shaped lever measured here
  predicts the resident set, and needs a sub-0.43 GiB pool to ever fire. A prediction that can be
  cold must be about *a different token's* routing — the current token's own layer-(il+1) set is
  produced by the layer's own argsort and cannot be had earlier; a multi-token-ahead or
  hidden-state-based predictor is a different project, not a flag.
* **F1's two terms remain the honest targets**: the per-layer barrier (0.46 ms × ~29 layers ≈ 13 ms)
  and the per-miss cost (0.695 ms × ~74 ≈ 51 ms). The first is independent of the pool and of
  quantisation; the second is what a pool/quantisation change moves.
* **Untested and still open**: the MTP-on `H`/`C` step split (the `--spec-type ngram-map-k` ablation
  self-rejected because `mean_draft` was 1.016, not 3 — a T≈2 step with a silent drafter, so it
  never measured what it claimed).

## 7. Honest limits

* No absolute t/s is claimed for either F2 arm: both are instrumented (`CGC_HOOK_SPLIT`) and the box
  carried swap pressure. Only the within-arm ratio is quoted.
* The `p2` arm has no `CGC-PREV-PF` line (that instrument is gated on the prev-token flag), so its
  "all resident" is inferred from the identical counter signature plus the geometry — not read out
  directly. Both 8 GiB and 4 GiB were read out directly for the *same* shared structure in p1/p3.
* `cb` is the work-row median; the pre-existing pooled 62 ms figure was an artifact and is not used.
* The compiler comment claiming `pread_usec` is wall-comparable sits one function away from the one
  stating it is not; the teardown line's `effective_rate` label is what invites the misreading. The
  one-line engine fix (print the worker count, or rename to `per_job_rate`) is **not applied** — it
  needs a shared-library rebuild, and the correction is done in `f2_overlap_verdict.py` instead.

## 8. Artifacts

* `scripts/check/f2_overlap_verdict.py` — applies the pre-registered rules; separates
  `NEVER-CALLED` (instrument live) from `NOTHING-QUEUED` (no instrument) rather than collapsing them;
  12-check self-test.
* `scripts/check/cb_miss_regression.py` — now parses the work/shadow split itself, reports `L` per
  population, and renders both; 21-check self-test.
* `Backup/phase_decomp/f2_arms/` (4 arms + `armB_dbg`), `Backup/phase_decomp/f2_trigger_probe/`
  (`p1`, `p2`, `p3` + `f2_trigger_probe.sh`), `Backup/phase_decomp/F2_F5_OVERLAP_AND_AUDIT_20260920.conditions.md`.
* Server logs: `llama_server_20260920_1150{59,208,318,426}.log` (the 4 arms),
  `115810` (armB_dbg), `1202{02,310,537}` (p1/p2/p3).

Uncommitted. No llama processes left running.
