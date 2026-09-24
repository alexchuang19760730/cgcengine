# G6 — the accept rule, measured at last (2026-09-20)

## §0 One line

The accept rule (`CGC_MTP_REJECTION`) **does** raise `mean_len`: all 5 paired reps, same sign,
median **+0.375** tokens/step — but the weakest rep is **+0.090**, below G6's +3.5% (+0.118), so
G6 is **not delivered** by this knob; and it buys that at **−0.45 t/s** (median), i.e. the metric
moves while the objective does not.

## §1 The instrument had five defects, and each one alone would have produced a wrong answer

Everything below is in `scripts/check/mtp_accept_ab.py` / `server_window.py`, with a selftest that
fails if the defect returns (25/25 and the shared probe's suite are green).

| # | Defect | The wrong answer it would have produced | Fix |
|---|---|---|---|
| 1 | `complete()` hard-coded `temperature: 0` while the delivery recipe runs **0.4** (`run_server.sh:1277`), and the request value wins over the server default | **Every accept this harness ever produced was a GREEDY accept.** At greedy p is one-hot, `min(1,p/q)` degenerates to exact match, so the accept RULE is a null by construction — an A/B would report "the rule does nothing" from an experiment that could not have shown anything. The historical products carry **no `temp` field at all**, which is how to recognise them | temp sent explicitly, recorded in the product; `--rule-ab` **refuses** at `temp <= 0` (rc 2) instead of printing a flat column |
| 2 | The harness probed raw `/completion`; G6's `from` came from a **chat** turn (`llama_server_20260918_163448.log` task 9, output opening `</think>`) | An accept measured asking a different question through a different template than the number it is compared against | `--chat` (the suite's own request shape, `chat_body()`) and `--profiles`, which reads the prompts from `replay_bench_reference_v2.json` |
| 3 | The engine's `draft acceptance … mean len …` rows live in the **server's** log; the parser read the **launcher's** log | Run 1 reported `mean_len: null` for all six arms ⇒ `INVALID` verdict, and the reason given was "the MTP path printed nothing" — a false statement about the engine | `resolve_server_log()` reads the launcher's own `[log]` / `log: tail -f` lines; `--reread` re-derives an existing product from its logs |
| 3b | The first resolver matched **neither** real line shape, and its selftest passed because the fixture was invented (`log: <path>`, one line) rather than copied | The tool looked correct and resolved nothing in production | both real lines pinned verbatim in the fixture; a candidate must exist on disk or it is rejected |
| 4 | `rule_verdict` judged by the **median** of the paired deltas — and the median of two values *is the larger one* | `{+0.005, +0.275}` would read as **MOVED** on one lucky rep | the effect is the **weakest consistent rep**; a new selftest pins that exact two-rep shape |
| 5 | A window stolen between the guard's poll and the launch raised `BusyBox` → traceback, and the chain aborted | A stolen window is indistinguishable from a broken instrument | refusal returns **rc 4**; `server_window.py wait --retries` re-waits and re-verifies (never proceeds), and `rule_ab` flushes the product **after every pair** so a late steal cannot erase the pairs already measured |

A sixth item was verified by hand because the running process predated it: the knob reaches the
engine. `ps -Eww -p 43121` showed `CGC_MTP_REJECTION=1` in the **server's own environment** during
the ON arm of run 2, and `ps -Eww -p 42168` showed it **absent** (25 `CGC_*` vars, none this one)
during the preceding OFF arm. `server_env()` now records that witness in every arm, because
`run_server.sh` forwards the variable without echoing it, so no log could have shown it.

## §2 The paired A/B (run 2)

    python3 scripts/check/mtp_accept_ab.py --rule-ab --arms nail --chat --profiles qa-zh \
        --reps 5 --n-predict 96 --temp 0.4 --out Backup/cgc_logs/mtp_rule_ab_g6_r2.json

Carrier Nail, head identity `219b27c4f9eebc75` (gate PASS, 21 tensors, 700.8 MiB). Arms alternate
order per rep (`off,on` / `on,off` / …), same pool (8 GiB), same door, same prompts, same temp; the
only difference is the variable. Window class **clean** (10 samples, 0 busy). Product:
`Backup/cgc_logs/mtp_rule_ab_g6_r2.json` (`partial: false`, `pairs_done: 5`).

| rep | mean_len off | mean_len on | Δ | accept off → on | decode t/s off → on | Δ t/s |
|---|---|---|---|---|---|---|
| 0 | 3.375 | 3.540 | **+0.165** | 69.4% → 70.5% | 12.36 → 9.68 | −2.68 |
| 1 | 3.340 | 3.825 | **+0.485** | 67.1% → 74.2% | 12.18 → 11.86 | −0.32 |
| 2 | 3.375 | 3.465 | **+0.090** | 67.9% → 81.0% | 10.52 → 11.26 | +0.74 |
| 3 | 3.365 | 3.740 | **+0.375** | 75.1% → 71.0% | 11.98 → 11.53 | −0.45 |
| 4 | 3.515 | 3.900 | **+0.385** | 80.6% → 83.7% | 12.87 → 11.40 | −1.47 |

    verdict: BELOW THRESHOLD -- same sign in all 5 reps, but the weakest of them is 0.090
    tokens/step < the 0.118 G6 needs (3.5% of 3.38); median +0.375
    paired decode: median -0.45 t/s, mean -0.83 t/s, range [-2.68, +0.74]

**Two readings, and they point the same way for the roadmap but not for G6.**

1. All five deltas are **positive**: the rule reliably raises acceptance (mean_len median +0.375 =
   +11%, accept e.g. 67% → 74%). This is not noise in the sign — 5/5.
2. The rule **costs more wall clock than it buys**: tokens/s falls in 4 of 5 reps, median −0.45
   (~−4%), mean −0.83 (~−7%). The mechanism is visible in the design: with the rule on, the draft
   step copies its distribution (`dist = ... ? &slot.spec_draft_dist : nullptr`) and the accept step
   runs `cgc_rejection_accept` per draft token. The extra tokens do not pay for the extra work at
   k=3 on this box.

⇒ G6's `mean_len` is a means, not the end. A change that raises `mean_len` and lowers t/s **fails the
objective while passing the gate**, so the roadmap's M4/G6 cells must be read together with a paired
t/s reading. On today's numbers the rule is net negative for throughput.

### Run 1, and why its numbers are not quoted

Run 1 (3 reps, 00:52–00:58) is `Backup/cgc_logs/mtp_rule_ab_g6.json` + `.reread.json`. Its reading
was empty (defect 3) and was recovered by `--reread` from the logs: deltas −0.300, +0.275, +0.005 ⇒
`NOT SEPARATED` (sign changes). **Two confounds**, both recorded before they could be forgotten
(`Backup/cgc_logs/mtp_rule_ab_g6.conditions.md`): a `purge` landed inside rep 0's *off* arm, and the
run's own provenance says `clean` because the in-process samples were taken before it. Read together,
run 1 is a 3-rep, contaminated, lower-power version of run 2 — that is why run 2 exists and why the
number quoted is run 2's.

### The purge is a negative result worth keeping

Asked for root `purge` to open a window, measured immediately after (while a server was loading):
reclaimable **7574 → 541 MB**, compressor **597 950 → 1 075 093** pages stored, swap **5.59 → 6.62 GB**.
On this box `purge` does not create headroom; it forces compression and grows swap. Consistent with
the project's earlier finding that purge only drops clean file cache and cannot touch the compressor,
where live processes' anonymous pages sit, and with the recorded 09-19 case where a `sudo purge`
left T1 aborting every time (`.workbuddy/memory/2026-09-19.md` §EN-187). **Operational rule: do not
purge to open a measurement window here; release anonymous pages (close sessions) instead.**

## §3 What accept actually is — and why the portal's `from` cannot be compared

Every accept row in `llama_server_20260918_163448.log` (the source of the portal's `from`):

```
task 9   : 0.46541 (74/159), mean len 2.40      <- prompt A (14 tokens -> 128 out)
task 73  : 0.46541 (74/159), mean len 2.40      <- prompt A again, IDENTICAL integers
task 128 : 0.46541 (74/159), mean len 2.40      <- prompt A again
task 14x : 0.46541 (74/159), mean len 2.40      <- prompt A again
task 0   : 0.42105 ( 8/ 19), mean len 2.14      <- prompt B (4500-token prefill, 16 out)
one more : 0.42105 ( 8/ 19), mean len 2.14      <- prompt B
run mean : 0.4506 / 2.313 over 6 rows
```

So `0.46541 / 2.40` is **one request, replayed four times, byte-identical** (74/159 every time — a
fixed seed, or effectively greedy), plus one other prompt twice. It is a per-prompt value, not a
regime average, and it cannot be compared with any other prompt mix.

Today's runs, on the same carrier at the same temperature through the same door, give
**0.67–0.84 accept / mean_len 3.34–3.90** on `qa-zh`. That is not a contradiction: it is a different
question. The suite instruction for G6's field is therefore that the value must be quoted as a
5-tuple, **carrier · prompt set · temperature · door · k** — `mean_len = 1 + k·a` is only a mean_len
at a stated k (k=3 here, `--spec-draft-n-max`), and at k=3 the ceiling is `mean_len = 4.00`, which
several reps already reach per-task.

A full-suite accept reading is queued
(`Backup/cgc_logs/mtp_suite_accept_g6.json`, all 7 profiles, one arm) precisely because `qa-zh` alone
is not the delivery mix; `coding`/`longform-zh`/`writing` are where acceptance has to be low.

## §4 What this means for G6 and for G4

- **G6 stays open, owner line S.** The accept-rule lever is measured and insufficient: +0.375 median
  but +0.090 weakest, and negative on t/s. What is left is (a) the (base, head) pair and sampling
  (`accept` itself), and (b) `k`, which the arithmetics of `mean_len = 1 + k·a` bound at
  `1 + 3·a`: at the *measured* a≈0.70 that is 3.1, at a≈0.83 it is 3.5 — i.e. **G6's E2b threshold
  (≥3.479) is reachable at k=3 in the qa-zh regime already**, and the suite mix is the open question.
- **G4 gains no support from this run.** G4 exists to cover "G6 cannot deliver". This run does not
  move `mean_len` enough to close G6 and does not change G1's ceiling (24.15 t/s at 99.39 ms/step),
  so per §EN-266 the per-op union decomposition is still **not warranted** — the same conclusion, now
  with one more measured reason.

## §5b The two lines joined: `tokens/s = mean_len / (step_ms/1000)`

The 25 t/s target is ONE equation whose two sides were owned by two lines, and neither side is
interpretable alone: an accept rate has no units until divided by a step, and a step time says
nothing until multiplied by the tokens that step emits. That is why "24.15 t/s at 99.39 ms" (line B)
and "12 t/s today" (line A) could coexist for days without either being wrong.

### What was integrated

| Piece | Owner | How it is now shared |
|---|---|---|
| engine `draft acceptance … mean len …` rows | this line | `parse_engine_accept()` (per task, warmup excluded) |
| DECPROF step + per-layer rows | **line B** | **imported**: `attn_moe_split.parse_step_rows/parse_layer_tables` and its `STEPRE`/`LAYRE`, not re-spelled. A second copy of a log format is how two instruments end up disagreeing about one file |
| the quotient | neither | `join_arithmetic()` + a joint table printed by `--decprof` runs and by `--reread` |

    python3 scripts/check/mtp_accept_ab.py --arms nail,nail_nomtp --chat --profiles qa-zh \
        --decprof --n-predict 96 --temp 0.4 --out Backup/cgc_logs/joint_step_accept_g6.json

`--decprof` sets `CGC_DECODE_PROFILE=1` (+`_ALL=1`) through the same allowlist every other knob
uses, and the product then carries, per arm: `mean_len`, `median_total_ms`, `median_wait/cb/submit`,
`median_union_sum/gap_sum`, per-layer medians (trunk and the L40 head), and `arithmetic` with
`implied_tps`, `step_needed_ms`, `mean_len_needed`.

### The guardrail this needed, learned from four real logs

| log | steady ntok | steps | median step |
|---|---|---|---|
| `llama_server_20260919_224333.log` (the racy `ahead` probe) | 4 | 70 | **48.20 ms** |
| `llama_server_20260919_224011.log` | 4 | 348 | **61.46 ms** |
| `llama_server_20260920_010541.log` (line B, today) | 4 | 54 | **92.13 ms** |
| `llama_server_20260920_010445.log` (cold first step) | 2 | 1 | **759.29 ms** |

**15× at the same `ntok=4`.** A step median is exactly as unquotable without its workload as an
accept is without its prompt set — the defect that produced "0.46541 is today's accept". So
`compare_step_profiles()` carries the workload identity and **refuses** to difference two
decompositions that differ (`comparable: false`, with the axis named). It is the same discipline as
the oracle gate's `comparable`, for the same reason.

**Correction (02:45): the step count is not one of those axes.** It was, and that was wrong. `n_steady`
is how many steady steps a run *happened to emit*, and with a stochastic accept that is an outcome,
not a property: two runs of one recipe legitimately produced **214** and **264** steady steps while
being identical in every axis that defines the workload (prompt mix, count, `mtp`, `temp`, `pool`).
Counting it as identity made the rule refuse the very pair needed to certify reproducibility — while
still refusing the pair it was written for, i.e. one axis doing a true positive and a false positive
indistinguishably.

It is now a **warmth** signal, reported rather than refused, and the two questions are separate:

| question | field | consequence |
|---|---|---|
| same workload? | `comparable` | a mismatch makes a difference meaningless → refuse |
| equally warm? | `warmth.matched` | a mismatch makes a *gap* contaminated → say so; an *equality* is still quotable |

`judge_repeat()` is what the split buys: it certifies two runs of one recipe directly, and returns
`reproduced: null` (cannot decide) rather than `false` when the pair is a different recipe, has a
missing half, or is warmth-unmatched. Applied to the two real runs of the joint harness — 62.17 vs
56.54 ms — it reads **reproduced, 10.0% spread**, which is the first reproducibility claim on this
line that a tool made rather than an author. The warmth gate is `n_steady` within 1.5× (band
calibrated on the two pairs that exist: 214/264 = 0.81 matched; 54/214 = 0.25 not), and `cb` is
reported, never gated — it is sub-millisecond at the warm end, where a 3× ratio (0.53 vs 1.71 ms) is
noise between two runs that did reproduce.

> **[SUPERSEDED THE SAME DAY — 62.17/56.54 ms are artifacts, and so are the `n_steady`/`cb` figures
> above them]** A `CGC-DECPROF` step log writes **two rows per verify round**, one carrying the round
> and one carrying ~1/100 of it (`gap_sum == 0` on the second). The parser pooled them, so every step
> number in this document that came from `parse_step_profile` is a mixture: the same pair re-read on
> work rows only is **247.98 vs 227.58 ms** (13.61 and 15.29 t/s steady), `n_steady` **107 and 132**,
> `cb` **74.18 and 62.85 ms**. The 09-20 joint product's own delivery field is affected too: it is an
> unweighted mean of per-request rates (11.27 t/s) where the run's throughput is 9.18.
> See `docs/DECODE_STEP_ROW_POPULATIONS_2026-09-20.md`. **The reproducibility VERDICT survives the
> correction** (`judge_repeat` → reproduced, 8.96% spread) — but note *why* it could: both runs were
> polluted by the same defect, exactly the shape that let a cross-pool invariance check pass while
> every arm routed to the wrong experts. A ratio test is not evidence that its inputs are right.

### Where today's step actually goes (line B's run, `ntok=4`, 54 steady steps)

    total 92.13 ms | wait 65.56 (71%)  cb 16.75 (18%)  submit 3.04 (3%)
                   | GPU: union_sum 76.69 (83% of total)  gap_sum 22.83 (25%)

That is **the same shape as E2b** (union 104.60 = 75.2%, gap 34.76 = 25.0%), i.e. the MoE **union**
is the dominant term in both, and it is the term a kernel can attack: a 30% union cut at
`mean_len 3.40` puts the step at ~108 ms ⇒ **~31 t/s**; a 2× cut ⇒ **~39 t/s**. The `gap` term is
host-side (`gap_L = 1.00·cb_{L-1} + 0.29–0.35 ms`) and a kernel cannot touch it, so the two levers
are complementary, not alternatives.

### What is *not* yet a reading

Composing the two halves across runs — this line's `mean_len 3.40` with line B's `92.13 ms` — gives
`36.9 t/s`. That arithmetic is now one function call away, and it is **not a measurement**: it mixes
two workloads, which is precisely what the guardrail above forbids. The in-run version is queued
(`Backup/cgc_logs/joint_step_accept_g6.json`, MTP-on and MTP-off arms with DECPROF, window-guarded).
Until it lands, the defensible statements are the two that do not cross a workload boundary: the
*shape* (union ≈ 75–83% of the step) and the *level* (today's `ntok=4` step median is 92.13 ms in
that arm, versus 169.21 ms in the arm G1 was measured on — a difference that must be explained by
workload or regime before it is called a speedup).

## §5 Limits, and what would falsify this

- qa-zh only, 8 prompts, `--n-predict 96`, one carrier, one binary, one box (16 GB, ~5.6 GB carried
  swap). The t/s deltas are HTTP `predicted_per_second` per arm and are the noisiest column here;
  the sign pattern (−4 of 5) is consistent but a 5-rep median is not a tight bound.
- The `0.4` temperature is a *request* value; the server's own default is also 0.4
  (`--temp 0 … --temp 0.4`, where llama.cpp warns only the last is used). Both agree, and the request
  wins, so the arm's regime is unambiguous — but it is worth stating since that warning is present in
  the 09-18 log too.
- **Falsifiers.** If a rerun at ≥8 reps shows a weakest delta ≥ +0.118 *and* a paired t/s median
  ≥ 0, the rule becomes a legitimate G6 candidate and this document is wrong. If the suite-level arm
  lands at mean_len ≥ 3.479, then G6 is met in the delivery mix *without* the rule, and the rule's
  t/s cost makes it a thing to leave off.
- Everything uncommitted: `scripts/check/mtp_accept_ab.py`, `scripts/check/server_window.py`,
  `agent_harness/portal/targets.json` (G6 field), this file, and the `Backup/` products. No `src/`
  change, so no binary/source pairing obligation arises.
