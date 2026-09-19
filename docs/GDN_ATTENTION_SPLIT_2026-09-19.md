# Where the GDN layers' extra per-layer GPU time lives — and why it is not the conv, and not the scan

Date: 2026-09-19 18:44 / 18:51. Box: 16 GB M4 Air, `CGC_SERVER_MTP=0`, profile `off`, decode only,
greedy (`CGC_FORCE_TEMP0=1`), port 8199, memory-guarded launch.
Instrument: `CGC_GPU_NODES=1 CGC_GPU_NODES_MATRIX=1 CGC_GPU_OPS=1 CGC_GPU_NODES_TRACE=1`
`CGC_DECODE_PROFILE=1 CGC_DECODE_PROFILE_ALL=1`.

## 0. The numbers belong to a named build

| artifact | md5[:16] |
|---|---|
| `llama-server` | `054fb22f04a01c5c` |
| `libllama.0.dylib` | `aab787412e572550` |
| `libggml-metal.0.dylib` | `1be366306c604669` |
| `libllama-server-impl.dylib` | `a87bfd2e80a086b4` |

The last three are the M1/M2/M3 = 9/9 PASS anchor (`libllama` `aab787412e572550`,
`libggml-metal` `1be366306c604669`), so this is the *same* binary the bit-identity gate describes.
Raw logs: `Backup/cgc_logs/llama_server_20260919_184439.log` (96-token run),
`Backup/cgc_logs/llama_server_20260919_185132.log` (400-token run).
Probe: `scripts/check/gdn_split.py` (`run` / `reparse` / `analyze` / `selftest`, 40 self-test
branches).

## 1. The answer

`docs/DECODE_STEP_BUDGET_2026-09-19.md` §16 measures the 30 GDN layers against the 10 full-attention
layers *within the same step* and reports a per-layer GPU-union difference. This run measures it again
(12 decode steps, medians):

| | ms/layer GPU union | wait | gap |
|---|---|---|---|
| 30 GDN layers | **2.143** | 2.408 | 0.670 |
| 10 full-attention layers | **1.373** | 1.523 | 0.753 |
| delta to explain | **+0.770** | | |

> **Read §9 with this table.** The `+0.770 ms/layer` column is the unstable form of the claim: over 25
> steps it has CV 68% and tracks the step's own regime. The quotable form is the ratio
> **GDN/attn = 1.55x**, which is invariant across decode width, launch and position (§9.2), and which
> disappears on the prefill-shaped steps (§9.3).

Inside the GDN layer, by op (node counts are the layer population — §3):

| bucket | ms/GDN-layer | bound | standing of the bound |
|---|---|---|---|
| conv (`SSM_CONV`, nd=30 = 1/layer) | **0.032** | 1.343 | **diluted 42×, not quotable** |
| scan **+ state update** (`GATED_DELTA_NET`, nd=30 = 1/layer) | **0.131** | 0.302 | usable |
| conv + scan + state | **0.163** | 0.334 | |
| GDN gates + projections (name table, work-weighted) | 0.286 | — | — |

> **conv and the fused recurrence together are 21% of the measured +0.770 ms/layer** (43% if every
> defensible bound is spent). For comparison, the same two buckets were 0.133 against a +0.618 delta
> in the 96-token run = **21.5%** — two runs in different regimes agree on the ratio.

**scan and state update cannot be separated in this build**: at `n_tokens == 1` the builder takes
`build_delta_net_fused` (`delta-net-base.cpp:441`) and the whole recurrence — including the state
write — is ONE node, `ggml_gated_delta_net(...)`, named `gdn_state`. Splitting it needs a code
change, not a better probe.

## 2. Why a within-step comparison is the only admissible shape

Every cross-launch comparison on this box has already been shown to lack resolution (lane A's
anchors spanned 34.5% and 97.5%; the drift lane saw the same config move −60% inside one arm). Both
numbers above come from the **same steps of one warm request**, so the regime cannot enter the
difference — and that is why the answer is quotable at all, even though the 400-token run's absolute
`ms/token` (159.2) is 36% worse than the 96-token run's (117.5): different regime, same ratio.

## 3. The op table is decisive because its node counts ARE the layer populations

The name-based count test cannot separate "1 per GDN layer × 30" from "3 per attention layer × 10",
and on the first real run it mislabelled 23 kinds as CONFLICT when almost none of them could decide
anything (now reported as *inconclusive*). The op table does not have that problem, because a model
with 40 layers of which 30 are GDN and 10 are attention has to show those populations:

| op | nd | per layer | reading |
|---|---|---|---|
| `GATED_DELTA_NET` | 30 | 1 | 30 GDN layers |
| `SSM_CONV` | 30 | 1 | 30 GDN layers |
| `ROPE` | 20 | 2 | 10 attention layers |
| `SET_ROWS` (KV write) | 20 | 2 | 10 attention layers |
| `SOFT_MAX` | 40 | 4 | 10 attention layers |
| `MUL_MAT_ID` / `GLU` | 117 / 78 | 3 / 2 | **39** layers (one layer has no MoE) |

**30 + 10 = 40 closes with the layer count**, which is the consistency argument the divisibility test
could not make. The 39-layer MoE population is a fact this probe found rather than assumed, and it is
why `nd % 40 == 0` is the wrong test to lean on.

## 4. What this falsifies

1. **"The GDN layers cost more because of their attention."** They do not. Their own attention-side
   total (0.557 ms/layer by the name table) is *lower* than the attention layers' (0.730 ms/layer).
   Whatever the +0.770 is, it is not GDN attention work.
2. **"Fix the conv / the recurrence and the GDN gap closes."** It cannot: both are 0.163 ms/layer
   together, and they would have to be ~4.7× larger before they could account for the delta.
3. The falsification test, stated so that a different instrument can contradict it: **if the fused
   node ever measures above its own upper bound, 0.302 ms/GDN-layer, the percentages above are
   wrong.** They are not a fit result — they are the instrument's work-weighted column, with the
   bound the instrument itself reports.

## 5. What it does not settle

- **Where the +0.770 actually is.** The live candidates are (a) shared kinds whose per-layer cost
  differs between the two layer types (`moe` 0.747 ms/layer is shared and is the largest single
  bucket), and (b) the delta's own stability. This probe does not separate them. Note that the
  per-layer `union` is the GPU *span* of that layer's segment, which includes waiting for other
  in-flight work, so a per-layer-type difference in span is not automatically a difference in work.
- **Per (layer, kind) resolution.** The node→layer mapping in index space is not dumped for a
  decode-shaped graph (`CGC_GRPH_DBG` only fires on the first 6 graph computes, which are prefill), so
  the split is layer-normalised, not per-layer.
- **Splitting the fused node.** Needs a builder change (see §1).
- **The least-squares column.** It is rank deficient AND R² = 0.461 on 29 038 buffers × 44 kinds, so
  the `ls` column is left blank by the report: a fit that does not describe the data must not be
  quoted per kind. The name table's `wcntw` is the reading used above; `lb`/`ub` bracket it.
- **What the GPU is idle on** (35.3% at MTP-off per §16) is a different instrument; this one measures
  busy time only.

## 6. The target arithmetic, in the same units

25 t/s needs F to fall from 1.81 to ≈0.54 ms/layer, i.e. **−1.27 ms/layer**. Zeroing the *entire* GDN
attention side buys **0.163 ms/layer = 13%** of that. So: **the GDN layer is not the lever for
25 t/s**, and the remaining ~1.11 ms/layer sits in the shared per-layer cost (MoE math, the per-layer
eval barrier, dispatch). The single largest measured shared bucket is `moe` at 0.747 ms/layer.

## 7. Bugs found in the probe (all in the probe; the engine was not touched)

| defect | what it produced |
|---|---|
| grouped steps on the GPU table's header, which prints 1 in 8 steps | eight steps of NSM merged into one record → every per-step count ~8× high, a number that still looks plausible |
| JSON round trip stringifies layer keys | all 40 layers counted as GDN, zero as attention — a one-directional error that inflates one side and empties the other |
| a throwaway `_blank()` for pre-header NSM | the whole first step of every chunk silently dropped |
| `ms/L` divided by nodes-per-layer instead of the layer count | inflated every op with multiplicity > 1 by exactly that multiplicity (ROPE 5×) |
| `abs(None)` on the self-check | crashed the analysis when no sampled step was a decode step, instead of reporting the check as absent |
| a weak test labelled CONFLICT | 23 kinds flagged as contradicting the builder source when they were merely undecidable |
| quoting the MAX of the two op bounds | printed "174% of the delta" — a number that cannot be true |

Each has a self-test branch that fails if it comes back; the JSON-key one is tested in both
directions (normalised → 30/10 split; unnormalised → refuse to report a split).

## 8. Reproduce

```bash
python3 scripts/check/gdn_split.py selftest      # 50 branches, no GPU, no server
PROBE_N_PREDICT=400 python3 scripts/check/gdn_split.py run
python3 scripts/check/gdn_split.py reparse       # re-parse the saved server log, no re-launch
python3 scripts/check/gdn_split.py analyze
python3 scripts/check/gdn_split.py widthscan <server_log> [...]   # §9, no GPU
```

`reparse` exists because the raw record is the server log: when a parser bug is found, the fix must
not cost a launch (`CGC_GPU_NODES_MATRIX` is not a cheap instrument — 30 300 NSM lines for one
400-token request).

## 9. Correction: the gap is a RATIO, not an amount of milliseconds

§1 and the `+0.770 ms/layer` figure quantify the GDN penalty in a form that only exists at the
regime it was measured in. Asking the question a second way — *does the per-step gap move with the
box?* — settles which form is quotable (the tool now does this itself; see `stability`).

### 9.1 Two forms of the same 25 steps

The 25 `ntok=1` decode steps of run A, same binary (`aab787412e572550` / `1be366604669`):

| form | value | CV | corr with that step's own `union_sum` |
|---|---:|---:|---:|
| additive delta | **+0.767 ms/layer** | **68%** | **+0.51** |
| ratio GDN/attn | **1.565x** | 22% | +0.18 |

The additive delta mostly reports the regime: it tracks the step's own span, which ranged **66..162 ms**
within that one run. A number that moves with the regime cannot be spent as a budget line.

### 9.2 The ratio survives every axis we can vary

| axis | additive delta | ratio GDN/attn |
|---|---|---|
| decode width `ntok` = 1 / 2 / 3 / 4 | 0.49 → 0.71 → 0.79 → **1.15** ms | **1.54 / 1.54 / 1.55 / 1.52** (p25-p75 within 0.04) |
| per launch, across swap 5.1 → 6.9 GB | 0.49 → 1.74 ms | 1.51, 1.52, 1.52, 1.53, 1.54 (CV 4-16%) |
| position inside the repeating (3 GDN + 1 attn) group | — | 1.55 / 1.56 / 1.56 |

Widths 1-4 are **968 steps** pooled from nine separate launches (`widthscan` prints exactly this
table). Position is the alternative hypothesis that the penalty is a scheduling artefact rather
than a layer-type property; it does not survive.

Per-layer medians make the same point without any grouping: **every** GDN layer lands in
1.83-2.33 ms and **every** attention layer in 1.14-1.30 ms. L39 is the single exception at 2.22 ms —
it is the last layer, so it has no successor whose submission could overlap it.

### 9.3 The penalty is absent at prefill widths — but NOT because the GDN path changed

`widthscan` over the same logs also reads the prefill-shaped steps (`ntok` = 13 and 17, prompt
chunks): their ratio is **0.76 and 1.02** — the GDN penalty *disappears*.

**Earlier in this same session I wrote that this proves the T=1 autoregressive path is the cause.
It does not, and the check is arithmetic.** Fitting `span ~ T^e` separately on each side:

| width pair | GDN `e` | attention `e` |
|---|---:|---:|
| 1 -> 4 | 0.67 | 0.73 |
| 2 -> 4 | 0.75 | 0.84 |
| **4 -> 13** | **1.04** | **1.58** |
| 4 -> 17 | 1.05 | 1.28 |
| 13 -> 17 | 1.09 | -0.08 |

The GDN side is linear (1.03-1.09). It is the **attention side that moves** — superlinear in the
4 -> 13 range — so the ratio crossing 1.0 is explained by the attention layer's own scaling and
says nothing about which GDN kernel ran. The per-token view says the same thing:

| width | GDN ms/layer/token | attention ms/layer/token |
|---|---:|---:|
| 1 | 1.39 | 0.90 |
| 4 | 0.89 | 0.62 |
| 13 | 0.93 | **1.23** |

What survives from the original claim: the penalty measured at decode widths is real and
reproducible. What does not survive: attributing it to `fused_gdn_ar` vs `fused_gdn_ch`. Reading
`delta-net-base.cpp` shows the two tags build the **same single** `ggml_gated_delta_net` node (only
the registration tag differs, `n_tokens == 1` -> AR else CH), so this data cannot separate them.
The honest state is that the ratio's *cause* is still open; §9.4's "fixable thing is overlap on the
T=1 chain" was a hypothesis, not a finding.

### 9.4 What this does to the conclusions of §1 and §6

- §1's **0.163 vs 0.770 = 21%** stands as a *work* statement (the class-exclusive ops are 21% of the
extra span at that regime) — but the extra span itself is **not** an amount of work to harvest.
- §6's target arithmetic is unchanged in its conclusion: F must fall 1.27 ms/layer and zeroing the
entire GDN attention side buys 0.163. What changes is the *shape* of the remaining lever: since the
penalty is a **scale-free 1.55x** that no GDN-exclusive op accounts for (~3/4 of it), the fixable
thing is overlap on the T=1 chain, not deleted milliseconds.
- Stated as a falsifiable target: bring the ratio from **1.55 toward 1.15** — that is ~0.5 ms/layer
at the measured level, ~15 ms of a ~71 ms step (21% of what 25 t/s needs). Killing the whole GDN
penalty is still not sufficient for 25 t/s by itself; it is the largest single identified item.
- The form-gate is now a self-test branch in both directions: a drift-shaped sample is refused as an
amount, a flat one is declared quotable. The first version of that gate treated `CV == 0.0` as
"absent" and silently dropped the quotable branch — a zero read as a missing value, the exact bug
class this line of work keeps finding.

### 9.5 What is still not settled

- Which node in the T=1 GDN chain owns the span: the fused recurrence is **one** node, so the
  instrument cannot show whether the span is the scan itself, the state write-back, or the wait for
  the *previous* layer's state update. Splitting it needs a builder change (name the sub-steps), not
  a different reading.
- Whether the width-4 ratio (1.52) and the width-1 ratio (1.54) differ by anything: the p25-p75
  bands overlap, and each width is a different launch, so this is a resolution limit of the box, not
  a measured change.
- The per-layer `union` is a **span**, so it includes waiting on other in-flight work. That is a
  property of the quantity, and it is why the ratio — not the level — is the interpretable number.
