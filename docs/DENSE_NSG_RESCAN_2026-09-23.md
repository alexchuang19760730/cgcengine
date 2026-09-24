# Dense NSG rescan: the knob now reaches all three types, and it buys nothing the box can see

**Date** 2026-09-23 · carrier `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X` · shape-isolated probe
(`cgc_shape_probe`, `--op mul_mat`) · engine `libggml-metal` md5 `650dd9d1d9fb89a4` (independent
build) and the tree's binary (verified equal in behaviour) · 2 sweeps, ~700 probe invocations.

## 0. One line

The rescan is now **valid** — `CGC_MMV_NSG` is verified to reach the dense GEMV kernels for IQ4_XS,
Q6_K and Q8_0 (the compiled pipeline name carries the value for every arm), which the previous
dense sweep had no way to check. The result on the shapes that hold **98%** of the dense bytes is
**no lever separable from this box** (the tightest channel, the `lm_head`, is flat within 2.1
points across six arms on a ±5.4% floor), and the whole dense family's headroom is bounded at
**3.09 ms/step = 2.16%** of a 143 ms step. Whatever path reaches 25 t/s, it is not this one.

## 1. What the rescan needed — and what turned out to be true about reach

`ggml_metal_nsg_env()` is a function of the *type*, and until P1-3d the dense getter had no case for
IQ4_XS/Q6_K/Q8_0, so a sweep of those types could not have moved for any value: it would have
measured the plumbing. Both binaries now carry it:

| binary | md5 | dense reach | id reach (P1-3e) |
|---|---|---|---|
| independent build `/tmp/cgc-nsgid-build` | `650dd9d1d9fb89a4` | iq4_xs `nsg=2`, q6_k `nsg=2`, q8_0 `nsg=4` all move | iq3_s `nsg=2` moves |
| tree `src/llama.cpp/build/bin` | `128b6048870fcc55…` | same, verified | same, verified |

Reach is checked as the **pipeline name the probe compiled** (`kernel_mul_mv_iq4_xs_f32_nsg=16…`),
never as the env var that was set, and every arm in the sweep below reports `reached: True`.
The sweep runs against the independent build so the tree's shared dylib is never the thing under test.

## 2. The census: what the dense family actually is

From the GGUF header (2D tensors only; `token_embd` is a lookup, not a matmul):

| type | tensors | where |
|---|---|---|
| `iq4_xs` | 250 (2D) | every dense weight of layers 0–38: `attn_qkv`/`attn_q` (2048×8192), `ssm_out`/`attn_output` (4096×2048), `attn_gate` (2048×4096), `ffn_{gate,up}_shexp` + `attn_k/v` (2048×512), `ffn_down_shexp` (512×2048) |
| `q6_k` | 2 | `output.weight` (**lm_head**, 2048×248320) and `token_embd` (lookup) |
| `q8_0` | 8 | **one layer only**: `blk.39.attn_{q,k,v,output}`, `blk.39.ffn_{gate,up,down}_shexp`, plus `blk.40.nextn.eh_proj` |

q8_0 is therefore not a model-wide type — it is the last trunk layer plus the MTP block's projection,
worth 49 MiB/step = **0.34 ms/step at peak, total**. That is why it is priced but not swept: its four
shapes are 1–18 MiB/step, below what the batch model can resolve (see §5).

Per-step price at the quoted peak (108.8 GiB/s), decode, MTP off:

| shape | ×/step | MiB/step | ms/step at peak |
|---|---:|---:|---:|
| iq4_xs 2048×8192 | 40 | 340.0 | 3.05 |
| iq4_xs 4096×2048 | 40 | 170.0 | 1.53 |
| iq4_xs 2048×4096 | 30 | 127.5 | 1.14 |
| iq4_xs 2048×512 | 100 | 53.1 | 0.48 |
| iq4_xs 512×2048 | 40 | 21.2 | 0.19 |
| q6_k 2048×248320 (lm_head) | 1 | 397.9 | 3.57 |
| q8_0 (all four) | 7 | 38.4 | 0.34 |
| **family floor** | | **1148.1** | **10.30** |

Measured family cost is 13.39 ms/step (`DENSE_GEMV_INSTRUMENT_2026-09-22.md`), so the entire
recoverable headroom of every dense GEMV in the model is **3.09 ms = 2.16% of a step** — before
accounting for the fact that a tile knob changes occupancy, not the bytes.

The f32 router (`blk.*.ffn_gate_inp`, 2048×256, ×40 = 84 MiB/step = 0.72 ms) has **no runtime tile
dimension at all** in this kernel family, so of the dense bytes it is the one part this knob cannot
address even in principle.

## 3. Pass 1 — 3 reps, 700 ms/arm, six shapes: every channel refused

| shape | unset arm's own span across 3 reps | verdict |
|---|---:|---|
| iq4_xs 2048×8192 | 9.8% | INVALID (channel moved) |
| iq4_xs 4096×2048 | 17.2% | INVALID |
| iq4_xs 2048×4096 | 14.5% | INVALID |
| iq4_xs 2048×512 | 56.5% | INVALID |
| iq4_xs 512×2048 | 30.0% | INVALID |
| q6_k 2048×248320 | — | `unjudged`: its arms failed (b=64 asks for a 25 GB device pool) |

The refusal is the tool's own rule (unset spread > 5% ⇒ no reading), and it is the sixth dense/moe
channel on this box to be refused. The box was BUSY throughout (FreeBuff renderer 38% CPU,
WindowServer 20%, reclaimable 4.8 GiB) and thermally cycled NOMINAL → HEAVY — the latter partly from
this sweep's own load. Pass 1 is evidence about the box, not about the knob.

## 4. Pass 2 — 7 short reps, per-arm **minima** (contention is one-sided), and the lm_head's first measurement

The `lm_head` was never priced before; it is the single biggest dense item (398 MiB/step, 3.57 ms at
peak) and it is a pure bandwidth case: one GEMV, n=248320, T=1, 25 GB pool at b=64 fixed by using
b=1..4.

| arm | min %peak | median %peak |
|---|---:|---:|
| nsg=unset (default 2) | 80.1 | 84.4 |
| nsg=2 | 77.9 | 85.6 |
| nsg=4 | 81.7 | 86.5 |
| nsg=8 | 82.1 | 85.3 |
| nsg=16 | 80.4 | 87.4 |
| nsg=32 | **84.4** | 87.2 |

All five values reached the kernel. The **medians span 2.1 points (2.5%)** — smaller than the unset
arm's own 10.1% span across reps — and the best-case minimum (nsg=32, +5.4%) is **exactly the unset
floor's own uncertainty (±5.4%)**, so it is not separable. The useful, quotable reading is the level
itself: **the lm_head runs at 84–87% of peak** (≈4.1–4.5 ms/step against a 3.57 ms floor).

The two big iq4_xs shapes came back with floors of ±15.9% and ±73.7% (the second because its unset
arm read 42.3% in one rep and 78.0% median) ⇒ **no reading**, not "no lever". An independent
single-invocation check at nsg=8 on iq4_xs 4096×2048 (61.43 µs/op, 67.6 GB/s = 62% of peak) agrees
with the medians, which is the one cross-check available.

## 5. Two instrument facts this round established, both of which had been silently assumed

1. **The two-point difference inherits the full noise of both points.** `marginal = (g(b2) − g(b1)) /
   (b2 − b1)` cancels the per-graph fixed cost *only if F is identical in both invocations*; a 300 µs
   F-error lands as 9.4 µs on a 76 µs marginal (12%) at b2−b1=32. That is why pass 1 read 143% and
   148% of peak on one shape: g(64) measured under contention below g(32). A tile knob cannot be
   resolved by taking a difference of two separately-measured points on a box that moves by 20%.
2. **Shapes below ~20 MiB/step are below this model's resolution.** A 3.2 MiB/step shape returned
   −150.7% of peak. Small shapes need a much wider b-spread, not more reps.

The probe's own accounting was checked against the census model and agrees within 7% (61.43 µs/op vs
4.25 MiB/op ⇒ 72.5 GB/s modelled, 67.6 GB/s reported), so the disagreement is in the *diffing*, not
in the bytes.

## 6. What this means for the target

25 t/s needs ~30% off the step (218 ms → ≤106 ms at `mean len` 2.64). The dense GEMV family's
*entire* headroom is 2.16%, and this knob's separable contribution is zero to date. Closing this line
is worth stating plainly: **`CGC_MMV_NSG` on dense is finished as a lever** — it was worth the rescan
because reach had never been verified, and now it has.

The remaining 3.09 ms (if it is real) has to come from something that changes the kernel's structure
(larger tiles/r0 with a wider b-spread, or a different reduction), not from a function constant — and
even then it cannot exceed 2.16%.

## 7. Instrument changes (all in `scripts/check/shape_probe/sweep_nsg.py`, selftest 22/22)

| change | why |
|---|---|
| reach gate (pipeline name per arm) | a flat reading on an unwired value is evidence about the plumbing; this is the whole reason the rescan was owed |
| rotation per rep + null cell in every rep | a single `unset,…,unset` bracket measures the drift only in time |
| refusal when the channel moves (>5% unset span) | a median across a 30%-wide floor is not a small effect, it is no reading |
| best-case diagnostic (per-arm minima), labelled a bound | contention is one-sided; it is the only statement left when the channel is refused, and it must never be read as a win |
| provenance in the product (engine md5, thermal, window) | the repo's rule for measurement products; this line has five runs that cannot be compared |
| `--b1/--b2` raised per shape where the pool allows | fixes the 25 GB `lm_head` pool that made pass 1's arms fail |

Three defects in my own upgrade were caught by the sweep itself and are fixed: the reach gate compared
an `int` pipeline value against a `str` CLI value (so **every** arm read "never reached the kernel");
the refusal path lacked `median_pct`, a `KeyError` mid-sweep; and `--shapes` mis-parsed type names
containing underscores.

## 8. Honest ledger

- **One channel is readable (lm_head), two are refused.** "No lever on lm_head" is a reading; "no
  lever on the big iq4_xs shapes" is *not* — those channels' floors were ±16% and ±74%.
- The window was never clean: `window_at_start.class` came back unread and the box was BUSY by CPU
  and reclaimable memory (4.8 GiB) throughout. Thermal was NOMINAL at the start of pass 2 and HEAVY
  during it, partly from this sweep.
- Pricing uses the census model (k·n·bpw/8), which agrees with the probe's own byte count within 7%;
  the quoted "peak" 108.8 GiB/s is a cited value for this M4, not a measurement of it.
- The 13.39 ms measured family cost is from the other line's instrument, not re-measured here.
- Nothing committed; the shared `src/llama.cpp` metal source was left exactly as the other line had it
  (the only engine change in the worktree is theirs), and no server was started.

## 9. Reproduce

```bash
python3 scripts/check/shape_probe/sweep_nsg.py --selftest
# the readable channel (7 short reps, pool that fits the lm_head)
python3 scripts/check/shape_probe/sweep_nsg.py --libdir /tmp/cgc-nsgid-build/bin \
    --reps 7 --target-ms 400 --b1 1 --b2 4 --shapes "q6_k:2048:248320:1" --json /tmp/lmhead.json
# the bound table from any sweep product
python3 Backup/phase_decomp/L3/shape_probe/nsg_dense_bound.py \
    Backup/phase_decomp/L3/shape_probe/nsg_dense_lmhead.json
```
