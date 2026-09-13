# CGC_PREFILL_PROTECT A/B — variance, effect size, and the default decision

Date: 2026-09-13 · Branch: `demo/sweet-spot-windows-fix` · Config: IQ3_XXS-denseIQ4X, 8 GiB pool,
`mtp` profile, `--no-mmap`, cap=6, box = MacBook Air M4 16 GB.

## Why this experiment was run

The earlier claim was that `CGC_PREFILL_PROTECT=1` makes after-prefill decode 1.11–1.33× faster
(8.9 → 10–15 t/s). Two separate measurements of the *same* source gave 12.7–14.9 t/s and
10.0–10.4 t/s — a 1.5× disagreement between runs. The task was to pin the variance down and
then decide whether the flag should become the default.

## Finding 1 — the variance was machine state, not noise

The wall-clock "after-prefill decode" metric on this box is dominated by how deep into swap the
machine is, and the workload itself drives that:

| Observation | Value |
|---|---|
| Same arm, clean start, good run | 116 ms/tok (8.6 t/s) |
| Same arm, dirty start | 158–541 ms/tok (1.8–6.3 t/s) |
| Swap at a clean start vs mid-run | 1 488 MB → 6 195 MB in 2 warmup requests |
| Across 5 interleaved blocks, block 3 | swap 15 720 MB, decode 259–541 ms/tok |

Reaching that state takes ~2 requests; recovering from it takes ~140 s of idle after the server
dies. Two operational traps made this invisible:

1. **Gating on free% is not enough.** `memory_pressure` reports 80 %+ free while
   `vm.swapusage` still says 12 GB. Gating on free% alone let session 1 start arms in a
   thrashing state — that was the entire source of the 1.5× disagreement.
2. **Teardown telemetry never printed.** The stop path is `kill -INT` (see
   `scripts/run_server.sh`); sending `SIGTERM` to the process group bypasses the script's trap,
   so `~llama_expert_cache()` never runs and `final stats` / `prefill-protect` lines are
   silently discarded. Every earlier arm threw away its own counters.

Even from a clean start with **flat** swap (1 520 → 1 344 MB), the metric still drifts
monotonically within one run: 109 → 166 ms/tok over 18 requests (first-half median 121.9,
second-half median 136.9). So repetition and interleaving alone cannot fix this metric; the
drift has to be removed by the estimator or the metric has to be replaced.

## Finding 2 — replace the metric: counts are machine-independent

`file_reads` / `misses` are properties of routing + pool policy and do not move when the machine
thrashes. The OFF arm's per-request read count is essentially deterministic:
**15 951 … 15 954 (spread 3 of 15 952 = 0.02 %)**.

Protocol actually used:

- **one** server process, so both arms share one machine state;
- runtime toggle via `CGC_PREFILL_PROTECT_FILE` (new; the flag used to be a function-local
  static, i.e. startup-only, which is why an A/B previously needed a server reload per arm);
- **ABBA** request order per unit — `[A,B,B,A]` with A/B alternating between units — whose
  estimator `mean(A) − mean(B)` cancels any *linear* drift in sequence position;
- per-request counter snapshots (`CGC-RIG-SNAPSHOT`, printed on every toggle content change) so
  per-arm counts come from differencing consecutive snapshots;
- graceful `SIGINT` shutdown so the teardown totals print.

## Finding 3 — engagement is real and rigorously verified

| arm | `defer_skip` per request |
|---|---|
| OFF (9 requests) | 0, 0, 0, 0, 0, 0, 0, 0, 0 |
| ON (9 requests) | 121 668 … 154 017 |

Teardown: `defer_skip=1 404 089`, `defer_yield(overflow)=320` → **0.023 %** of skips had to
evict a decode-stamped slot anyway, i.e. the pool was not too small for the rule to work.

A snapshot is printed only when the toggle file's content changes; the alignment was confirmed
empirically (snapshot *k+1* = start of request *k*) because only under that alignment does every
ON delta have `defer_skip > 0` and every OFF delta have exactly 0.

## Finding 4 — effect size

ABBA paired, 4 complete units, ON − OFF (negative = PROTECT better):

| metric | per unit | mean | units favouring ON | sign-test p |
|---|---|---|---|---|
| **reads/req** | −640.5, −549.0, −789.0, −604.5 | **−645.8 (−4.05 %)** | **4/4** | 0.062 |
| **misses/req** | −170.0, −198.0, −193.0, −203.0 | **−191.0 (−4.24 %)** | **4/4** | 0.062 |
| ms/tok | +3.0, −1.4, −3.5, −11.4 | −3.3 (median −2.4) | 3/4 | 0.312 |

Note how the naive pooled comparison would have inverted the conclusion: pooled medians are
OFF 123.6 vs ON 132.0 ms/tok, i.e. "ON is 8.4 ms slower" — pure drift, and the ABBA estimate
says the opposite sign. This is the concrete value of the design.

## Finding 5 — numerically inert (5/5 gates, both cells)

| dump | md5 |
|---|---|
| `oracle_iq3_pool8gb_p1base.jsonl` (pre-rig binary, default) | `c59a4bb1ad59919f0aa7e1b41b4f021d` |
| `oracle_iq3_pool8gb_rigbase.jsonl` (current binary, default) | `c59a4bb1ad59919f0aa7e1b41b4f021d` |
| `oracle_iq3_pool8gb_rigprotect.jsonl` (current binary, `=1`) | `c59a4bb1ad59919f0aa7e1b41b4f021d` |

unique = 1 → 53 steps, `row_fnv1a64` identical. Both cells: `M1 53/53`, `M2 53/53`,
gate-template PASS, gate-buffer-nil PASS, gate-union-fit PASS (cap=6 × topk=8 = 48 ≤ 142),
preflight PASS (`15+27=42`, no echo, no leak, `finish=stop`). Separately, all 18 requests in the
A/B produced identical output text (sha `958ad5dbdd`, len 49).

So the mechanism only changes **which slot** an expert lands in; it does not move one logit bit,
and the rig instrumentation added here is itself bit-identical to the pre-rig binary.

## Decision: keep `CGC_PREFILL_PROTECT` **OFF** as the default

For:

- consistent 4 % reduction in expert reads/misses, 4/4 paired units;
- zero numerical risk (bit-identical, 5/5 gates), ~5.9 KB extra state, 0.023 % overflow.

Against:

- **no detectable gain in the user-visible metric** (median −2.4 ms/tok, p = 0.31);
- the test used a **single repeated prompt** whose output was a **prompt echo**, i.e. the
  routing trace repeats exactly and the pool converges to one stable set — the most favourable
  possible case for protecting it. With changing prompts the protected set goes stale;
- 4 % of reads is not on the critical path (the real cost is per-op submit+sync and paging).

The flag stays as a validated, numerically safe knob. It is not evidence-backed as a default.

## Open question (the one that could flip the decision)

Does the read reduction survive **changing prompts**? The rig can answer it directly: run the
same ABBA protocol with 3 rotating prompts so the decode working set changes between requests.
If ON stays 4/4 negative there, the case for defaulting it on becomes much stronger; if the
effect collapses or reverses, that confirms the repeated-prompt caveat above.

## Artifacts

- rig: `CGC_PREFILL_PROTECT_FILE`, `CGC-RIG-SNAPSHOT` (both inert unless the env var is set)
- driver: `/tmp/cgc_p1_abba.py`, analyser: `/tmp/cgc_p1_analyze.py`, raw: `/tmp/cgc_p1_abba.jsonl`
- server log with the 20 snapshots + teardown: `Backup/cgc_logs/llama_server_20260913_010558.log`
- gate cells: `Backup/knifeedge_matrix/*_rigbase.json`, `*_rigprotect.json`
