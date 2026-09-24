# The marginal verify token is not in the 41 layers

> ⚠ **更正 2026-09-23（見 `docs/ROUND_REMAINDER_IDENTIFIED_2026-09-23.md`）**
> §5 的「round remainder」**不是一個存在的工作項**：它是 `round_ms −（一個 step 的層時間）− draft`
> 的殘差，而**一個 round 有 1.5–2.3 個 `segs=41` verify step**（`CGC-DECPROF` 實測）⇒ 減掉的比
> 實際花的少，剩下的部分隨 T 成長只是因為 step 數隨 T 成長。**單一 step 的中位時間不隨 T 變。**
> 工具 `scripts/check/round_ledger.py`（selftest 9/9）把 round 重算成 `n_step × step + draft`，
> 閉合到 +7~22%。**§5 的表格與 §6 的「≥100 ms per round at T=4」預測不要再引用。**
> §2–§4（三分法與 per-layer 結論）不受影響。

**Date** 2026-09-22 · carrier Nail `IQ3_XXS-denseIQ4X`, MTP on, k=1 (T=2) vs k=3 (T=4), pool 8 GiB,
profile `prod25`, `-p 0 -n 128 -d 512`, llama-bench · engine `libggml-metal` md5 `8b61dfa206882d93`,
`llama-bench` md5 `639a88bf1411dd68` · instrument `scripts/check/verify_marginal.py` (30 selftests).

## 0. One line

The k-sweep left 61 ms/token as a black box ("residual: target verify + host"). It is **not** in the
per-layer decode work: the 40-layer `wait`/`gather`/`dispatch` terms move **-9.4 to +20.3 ms/token**
(median +2.7) while the round moves **+39 to +95**. The owner is a **round-level remainder of +29 to +66 ms/token
(70–102% of the marginal)** that neither the per-layer nor the miss instrument covers.

## 1. Why this is a measurement and not an inference

Three arms of evidence, two pairs of runs, all same-shape, same binary, interleaved reps:

| axis | instrument | pair | what it can and cannot see |
|---|---|---|---|
| round total | `CGC-MTP-PERF` (`calls_draft`, `t_draft_ms`) | both | the denominator; also splits the draft head out |
| misses | `LLAMA_EXPERT_CACHE_BATCH_DBG` | both | per-layer miss counts (prefill excluded by the `CGC-POST:` boundary) |
| verify matmul | `CGC_VERIFY_OP_TIMING` | `/tmp/verify_marginal` | **perturbs the run** — see §4 |
| per-layer wait/gather/dispatch | `CGC_DECODE_PROFILE_ALL` | `/tmp/verify_layers` | medians of a sampled population; T-insensitive |

## 2. The three-way split (VERIFY-OP pair), per token

```
  rep  round T=2  round T=4  marginal/tok     miss  compute     host
    1      396.2      684.4       +144.08   +14.98   +20.22  +108.88
    2      333.9      673.8       +169.94   +18.55    +9.29  +142.10
    3      399.0      579.7        +90.33   +13.40    +1.02   +75.91
  MEDIAN shares: miss 10.9%  compute 5.5%  host 83.6%   -> HOST-BOUND
```

`miss` uses F1's borrowed law (0.695 ms/miss); `compute` is `CGC-VERIFY-OP`; `host` is the residual.
The absolute numbers in this pair are inflated (§4), so read the **shares**, not the totals. They say:
the matmul is 5.5%, the miss axis 10.9%, and ~84% is host-side per-token work.

## 3. The per-layer decomposition (DECPROF pair) — and its own audit

`CGC-DECPROF all: Lk wait= cb= submit= ms` is the only per-layer split of a decode step; read as
`wait` = eval-sync, `cb` = gather, `submit` = dispatch. Summed over the 40 layers, as **means**
(a median step does not carry a marginal cost — the parser says so itself, and the median version of
this table sums to 7% where the mean version sums to 7–21%, i.e. both are small):

```
  rep  eval-sync  gather  dispatch   total  | round-level marginal | coverage
   1     -0.50    +2.04   +1.14     +2.67   |       +39.45         |    7%
   2     -9.63    -0.85   +1.12     -9.36   |       +64.22         |  -15%
   3    +11.85    +6.78   +1.66    +20.29   |       +95.32         |   21%
```

Note rep 2: the layers move *backwards* while the round grows 64 ms/token. A term whose sign flips
between reps is not the carrier of the effect.

Top layers by |marginal| are **not** a story: L0 is *negative* (−0.66), the largest single-layer
value is L39 at +0.33, and the per-layer residual never exceeds ±1.1 ms/token. **No layer owns this.**

## 4. The instrument trap that must travel with these numbers

`CGC_VERIFY_OP_TIMING=1` and the DECPROF/GAPSPLIT family are **mutually exclusive**. VERIFY_OP_TIMING
makes the expert-cache callback answer `ask=true` on the verify `MUL_MAT_ID` nodes, so those nodes are
evaluated one at a time and the segmented async dispatcher never runs. Measured in the same run:
`CGC-VERIFY-OP` 4096 rows + `BATCHDBG` 5647 rows, and `CGC-SEG` / `CGC-DECPROF` / `CGC-GAPSPLIT-L` /
`CGC-GPUTIME` **all zero**. Visible in the totals: that pair's rounds are 396/684 ms where the
uninstrumented pair's are 273/352. So the §2 shares are qualitative and the §3 shares come from the
other pair. A run cannot have both; the tool refuses rather than printing an empty table.

## 5. Where the marginal actually is: the round budget

Terms per round, all measured (`MTP-PERF` + `DECPROF`), remainder by subtraction:

```
  rep 1:  term                          T=2      T=4   marginal/tok   share
          per-layer terms (40)        209.7    215.0         +2.67      7%
          draft head                   14.7     29.5         +7.43     19%
          begin                         0.0      0.0         +0.00      0%
          remainder (unmeasured)       49.2    107.9        +29.35     74%
          ROUND TOTAL                 273.5    352.4        +39.45    100%

  rep 2:  per-layer 241.7 -> 223.0 (-9.36)  draft +8.23  remainder 0.1 -> 130.8 (+65.35, 102%)
  rep 3:  per-layer 161.5 -> 202.1 (+20.29) draft +8.75  remainder 5.5 -> 138.0 (+66.27,  70%)
```

Two things to read out of this:

1. **The per-layer terms are T-insensitive.** 209.7 → 215.0 for +2 verified tokens. Whatever the extra
   tokens cost, the 41-layer loop barely notices.
2. **The remainder is the T-dependent term, and at T=2 it is ≈0.** rep 2's remainder is 0.1 ms at T=2
   and 130.8 ms at T=4; rep 3's is 5.5 → 138.0. The remainder is not a rounding artefact — it appears
   *with* the wide verify batch and grows with it.

## 6. The falsifiable target (and the one that does not exist)

**There is no single-layer target.** The request was for one; the measurement says the *median* layer
moves 0.08–0.48 ms/token, and the layers that look large flip sign between reps: L0 is -0.66 / — /
+1.15 and L25 is the largest in rep 2 at -16.73 ms/token (26% of that rep's round marginal) while
being unremarkable in the other two. On three reps that is drift, not attribution; a single-layer
optimisation chosen from this table would be fitting noise.

The target that does exist is a **round term that scales with the verified token count and lives
outside the 41-layer segmented loop** — the per-round work after the layers: final `lm_head`, logits
and the sampler/argmax over all T positions, the accept path, and inter-graph host bookkeeping.

Falsifiable prediction, from rep 2 and rep 3 (12× and 25× separation between arms): instrument the
section between the last layer's publish and the next round's first layer and it should measure
**≥100 ms per round at T=4 and ≤10 ms at T=2**. If it measures flat in T, this decomposition is wrong
and the remainder is instead a per-round constant plus drift.

## 7. Boundaries — do not quote these numbers outside them

- **swap 4.5–6.8 GB** for the whole sweep, `usable` 39.9–63.9%, thermal `HEAVY` on some arms. Absolute
  ms are not comparable with the 12.57 t/s family; the **paired per-rep ratios** are.
- **rep drift is 1.5×** (k1: 7.31/7.84/11.21 t/s). Every marginal is differenced **within a rep**;
  pooled numbers are not used.
- `PER_MISS_MS = 0.695` is F1's law, applied as an external constant with its range (0.59–0.85) named.
- The per-layer rows are **medians of a sampled population**, so the per-layer sums explain 61–77% of
  the round *level* but only 7–21% of the *marginal*. The tool prints both.
- `llama-bench` has no prompt ⇒ its accept rate is 1.000 in the k=1 arm, not the server's 0.654.
  This is a cost-law measurement, not a production-regime ranking.
- No M1/M2/M3 gate was run for this pair: **nothing was changed in the engine** (env-only), and the
  digest is recorded per arm in `arms.jsonl`.

## 8. Artifacts

- `scripts/check/verify_marginal.py` — 30 selftests; refuses (exit 1) on missing boundary, missing
  miss axis, equal round counts, or a DECPROF-free arm. `--logdir` + `--arm-t1/--arm-t4`.
- `scripts/check/gap_vs_miss.py` — now keeps `wait` and the `*_mean` variants alongside `cb`, so the
  three axes come from one parser instead of three.
- Raw: `/tmp/verify_layers/arms.jsonl` (DECPROF pair), `/tmp/verify_marginal/arms.jsonl` (VERIFY-OP
  pair), each record carrying engine digest + window + pool statistics.
- **Not committed**; zero engines rebuilt for this analysis; no server left running.
