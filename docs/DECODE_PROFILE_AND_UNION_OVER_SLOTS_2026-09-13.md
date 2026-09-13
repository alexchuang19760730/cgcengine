# M0 — decode step profile, and the `union > usable slots` region is numerically wrong

Date: 2026-09-14 · Branch: `demo/sweet-spot-windows-fix` · Box: MacBook Air M4 16 GB

M0 of `ROADMAP_PREFILL250_DECODE25_2026-09-13.md` had two jobs: make the decode step's wall
time attributable, and find out whether the one pool region the gates never covered —
`union > usable slots` — actually changes the numerics. Both are answered below, and the
second answer is yes.

---

## 1. New instrumentation: `CGC_DECODE_PROFILE`

`llama_context` / the pool path already had phase timing (`CGC_PHASE_TIMING`) and an
every-160-segment average (`CGC-SEG`). Neither can answer "which layer dominates", and the
`CGC-SEG` window (~4 layers at a time) blurs exactly the layers the churn data says matter.

The segmented dispatcher in `ggml-backend.cpp` (`CGC_OA_ASYNC=1`) runs a loop that, **per
layer**, serializes:

```
wait for GPU layer i to complete  ->  fire the top-k hook (slot mgmt + blocking fill)  ->  submit layer i+1
```

so the step wall is `sum(wait + cb + submit)` over layers. The new counters attribute that
three-way split **per layer** and print one line per 8 steps, plus the top-8 layers by
`wait + cb` (`CGC_DECODE_PROFILE_ALL=1` prints all of them).

Both knobs are in `run_server.sh`'s env allowlist — without that they are silently dropped
and an A/B reports "no effect" when it is really "no knob", which has bitten this repo before.

**Reachability caveat:** this path only exists under `CGC_OA_ASYNC=1`. That is **not** the
launcher default (`CGC_SERVER_OA_ASYNC=0`, set that way on 2026-09-09 for a quality reason).
So the profile below describes the OA_ASYNC dispatch, not the default one; without it the whole
41-layer graph is a single async submit and none of the three components is separable.

## 2. Result — where a decode step's wall time goes

10 GiB pool (`n_slots=179`), MTP off, 39 MoE layers attributed, 28 reported blocks:

| component | mean per step | share |
|---|---|---|
| **wait** (GPU layer i completes) | **87.2 ms** | **72 %** |
| **cb** (top-k hook: slot mgmt + blocking fill) | **30.8 ms** | **25 %** |
| **submit** (dispatch layer i+1) | **3.2 ms** | 3 % |
| **total** | **121.1 ms** | 8.3 t/s |

The per-layer spread is **narrow** — wait 1.58–2.91 ms (1.8×), cb 0.47–1.10 ms (2.3×). There is
**no hot-layer outlier**, unlike edge0's `L37–39` 8× cluster. That is the diagnostic value of the
table: a flat profile means a **per-layer fixed cost**, not a few bad layers. The highest-wait
layer is L2 (2.91 ms) and the highest-cb layers are L38 (1.31 ms) and L30 (1.17 ms); L2 being
worst-wait is consistent with the churn measurement (layers 1–2 carry `distinct ≈ 250/256`).

Self-consistency check: the header line equals the sum of its own per-layer rows to 0.1 ms in
every block (e.g. header `w=101.84 c=23.05 s=3.99` vs rows `101.86 / 23.15 / 4.00`).

### 2.1 What this says about the 25 t/s target

`wait` is 87 ms of the 121 ms step, i.e. **~2.24 ms per layer of GPU wait**. For scale, one
MoE layer at T=1 for this model is ~64 MFLOP, which an M4 GPU does in tens of microseconds.
**So the 87 ms is not compute — it is per-layer launch/queue latency**, and the fact that it is
flat across 39 layers is what proves it. The `submit` side is tiny (3 %), so the gap is not
CPU-side submission cost either.

That reframes M3: the 2.4× needed for 25 t/s is a **~2.2 ms/layer fixed latency** problem, not a
FLOPs problem. It also means a prefetcher cannot help — the pool path's I/O already only uses
one of eight background workers, and the hook is 25 % of the step.

## 3. The `union > usable slots` region is numerically WRONG

The `union-fit` gate certifies `union ≤ usable slots` for the reference prompts, which means the
opposite case has never been gated. Forced it with a small pool instead of a bigger cap, so the
graph shape (cap 8) — and therefore the computation — stays fixed and only the pool moves:

| | 8 GiB (`n_slots=143`, usable 142) | **2 GiB (`n_slots=35`, usable 34)** |
|---|---|---|
| `buffer is nil` | **0** | **234** |
| **M1** (`row_fnv1a64` bit-identical) | *reference* | **2 / 42** |
| first divergence | — | **DEF step 2, token_idx 0** |
| argmax at that step | token **760** | token **248045** |
| `sum` at that step | −335486.28 | −519561.70 (**55 % apart**) |

Same model file, same prompt, same sampler (`temp 0`), `CGC_POOL_MAX_TOKENS=8` pinned on both.

**The divergence is not rounding.** A 55 % change in `sum` and a different argmax token 2 steps
in is a wrong computation, and it coincides exactly with the 234 `buffer is nil` dispatches. This
is the region the `usable_slots` gate kept the reference prompts *out* of — the pool path is
refused (correctly, that was the fix for the earlier `120 × buffer is nil`), but what runs
*instead* is not equivalent.

Two honest limits on this result:

1. The 8 GiB arm's *text* output was degenerate (`"\n" × 18`). The oracle compares per-step
   logits, and divergence at step 2 means the two runs' internal states differ immediately, so
   the M1 result stands on its own — but this prompt pair is not a clean golden reference and
   the run should be repeated with a prompt the reference answers properly before the number is
   quoted as a headline.
2. `union` was not logged in these runs (the fast-path `union=` counter appears only for MTP
   fast-path calls), so the causal link "union > 34 → gather path → nil → wrong" is inferred
   from the `n_slots=35` geometry plus the 234 nil dispatches, not from a direct union dump at
   the divergent layer. **Making `union` and the chosen path log per layer under
   `CGC_DECODE_PROFILE` is the next cheap step** and would close that gap.

## 4. What M0 changes for the roadmap

- `union-fit` must become `union-routable` (union ≤ slots **or** the streaming path ran), and
  the M1/M2 gate must include a `union > slots` case. **Today that case fails, so the gate
  cannot be tightened before M1 (pool/graph decoupling) lands** — the decoupling is what makes
  the wide-union case legal instead of broken.
- M3's target is now quantified: ~2.2 ms/layer of **fixed dispatch latency**, flat across layers.
- M2's prefill I/O budget is unchanged (0.70 GB/s at chunk 2048), and the prefill M1 invariance
  argument (always materialize 256) is unaffected by this finding.

## 5. Artifacts and how to reproduce

```
# profile (needs the segmented dispatcher)
CGC_SERVER_OA_ASYNC=1 CGC_DECODE_PROFILE=1 CGC_DECODE_PROFILE_ALL=1 \
CGC_SERVER_MTP=0 N30CACHE_BUDGET=10737418240 ./scripts/run_server.sh

# union > slots oracle case: same prompt, same cap, only the budget moves
#   8 GiB  -> N30CACHE_BUDGET=8589934592    (n_slots=143, 0 nil)
#   2 GiB  -> N30CACHE_BUDGET=2147483648    (n_slots=35, 234 nil, M1 2/42)
# both with CGC_LOGITS_ORACLE_DUMP=... CGC_LOGITS_ORACLE_TOPN=8 CGC_POOL_MAX_TOKENS=8
```

Code changed (uncommitted): `src/llama.cpp/ggml/src/ggml-backend.cpp` (per-layer accumulators +
report, +103 lines, inert without the env), `scripts/run_server.sh` (allowlist the two knobs).

Raw logs: `Backup/cgc_logs/llama_server_20260913_235408.log` (profile),
`..._235737.log` (8 GiB ref, 0 nil), `..._20260914_000023.log` (2 GiB, 234 nil).
Dumps: `/tmp/m0_ref8.jsonl`, `/tmp/m0_pool2.jsonl`. Server stopped; no leftover processes.

**Note on the profile run's model:** the launcher's `runtime_profile=auto` loaded
`Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`, **not** the `Nail-…-denseIQ4X` file the earlier baselines
used. The percentages are robust (they are ratios within one run) but the absolute ms and the
t/s should not be compared across those runs.
