# plain_match: the control arm, the k=0 dead end, and the result on a comparable build

**2026-09-19, HEAD `98b44c8c6`, working tree dirty (engine hunks uncommitted), machine otherwise idle.**
Artifacts: `Backup/phase_decomp/plain_match_v2.json`, `plain_match_v3.json`, `plain_match_bisect.json`,
`Backup/cgc_logs/llama_server_20260919_04*.log`; the pre-fix run is `/tmp/plain_match/result.json`
plus `Backup/cgc_logs/llama_server_20260919_025*.log`.

## 1. What the review said, and what was done

P0-1 — the MTP=0 arm was not "the same engine without speculation". `run_server.sh` exported
`CGC_MM_BITIDENT=1` (and `CGC_NO_PREFETCH=1`) only inside `if [ "$SERVER_MTP" = "1" ]`, so the two
arms ran **different kernels** for the decode GEMV (M=1 is inside `CGC_MM_BITIDENT`'s M<=8 range,
`ggml-metal-ops.cpp:2470`), and an output difference had two candidate causes. Correct diagnosis; the
value could not be re-added from outside because the launcher read it inside that block.

Fix (`scripts/run_server.sh`): those two knobs are hoisted out of the MTP block into an
unconditional block, with **default behaviour preserved** — MTP=1 still exports both, MTP=0 still
exports neither *unless the caller asks*, which is the new capability the A/B needs. Flipping the
MTP=0 default would silently re-baseline every recorded MTP-off number in this repo, so it was not
done. Evidence, `CGC_DUMP_ENV=1` (no launch):

| invocation | `CGC_NO_PREFETCH` | `CGC_MM_BITIDENT` | `LAYER_CAPS` | `VERIFY/DRAFT_DECODE` |
|---|---|---|---|---|
| default (MTP=1) | 1 | 1 | 40-40:256 | 1 / 1 |
| `CGC_SERVER_MTP=0` | — | — | — | — |
| `CGC_SERVER_MTP=0` + `CGC_MM_BITIDENT=1 CGC_SERVER_NO_PREFETCH=1 CGC_SERVER_LAYER_CAPS=40-40:256` | 1 | 1 | 40-40:256 | — |

P0-2 — `pkill -TERM -f "build/bin/llama-server"` in the driver's `finally` was pid-blind and would
kill other sessions' servers. Removed (already in the previous revision of the driver, `proc.terminate()`
+ `wait()` + `kill()` is the complete teardown).

## 2. The control arm the review proposed does not run: `--spec-draft-n-max 0` aborts the server

The review suggested `CGC_SERVER_MTP=1 + CGC_SERVER_MTP_N_MAX=0` (same launcher branch, no hoist
needed). Measured 2026-09-19 03:49: the server dies on the first generation step:

```
GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max) failed   llama-context.cpp:2961
Abort trap: 6    Backup/cgc_logs/llama_server_20260919_034940.log
```

Mechanism, both halves in `common/speculative.cpp`:

* the MTP draft impl emits **one** draft token even at `n_max = 0` — `result.push_back(id)` at :1795,
  and the clamp `if (params.n_max <= (int) result.size())` only at :1820, i.e. after it;
* the server reserves `1 + n_max = 1` output rows (`per_seq`, `common_speculative_get_output_limits`
  :2597).

So the verify batch is 2 tokens against a 1-row budget. This is an engine defect in its own right
(`n_max=0` reads as "no drafts" and is not), and it is **not fixed here**: it needs a rebuild of
`libllama-common`, which this box shares with other sessions. `plain_match_ab.py` now refuses that
configuration up front with the mechanism in the message instead of measuring a server that dies
mid-request.

## 3. The result on the current build

Arms equalised (`EQUALISE` = `CGC_MM_BITIDENT=1`, `CGC_SERVER_NO_PREFETCH=1`,
`CGC_SERVER_LAYER_CAPS=40-40:256`), prod25 profile, 8 GiB pool, `-p 42 -n 128`, greedy
(`CGC_FORCE_TEMP0=1`), order `1,0,1,0` then `on,off,on-exact,on-exact,off,on`:

| pair | compared | divergence | first at | within-arm deterministic |
|---|---|---|---|---|
| MTP on vs MTP off | 262 chars (text; this build returns no `ids`) | **57 / 262 = 21.8 %** | 205 | yes, both arms, all reps |
| MTP on (fast path on) vs MTP on (fast path off) | 262 | **0** | — | yes |

* geometry is equal by measurement, not by assertion: every arm's own server log prints
  `L4 pool capacity=143` and its own launcher banner `layer_caps=40-40:256`; the carrier is the same
  bytes (`realpath` + size 13,663,116,512 + head/tail hash `6d34d8cc3c38fce1` for every arm — the
  earlier `same_carrier=False` was a driver defect, it compared the symlink *path*);
* accept is identical in both MTP-on arms (76/149), so the extra arm is the same speculation run.

**Attribution: the ZERO-slot fast path is exonerated.** Turning `CGC_VERIFY_DECODE`/`CGC_DRAFT_DECODE`
off changes nothing at all, so the divergence from the MTP-off arm is not the documented pool
approximation — it lives in the batch/verify path itself (batch vs single-token).

**And it is not any of the equalisers.** Five MTP-off configurations × 2 reps, rotation-symmetric
order, all on this build: `off-plain` (nothing exported — exactly the pre-fix arm), `off-bitid`,
`off-nopf`, `off-caps`, `off` (all three) — **all ten runs produced the identical 278-char text**
(0/278 pairwise differences, deterministic within every arm).

## 4. The part that must not be hidden: the previous TRUE is not reproducible, and the arm that moved is the MTP-off arm

`/tmp/plain_match/result.json` (02:57, same driver arguments, same prompt, same model file, MTP=0 arm
with **no** extra env — i.e. byte-for-byte the `off-plain` arm above) reported
`PLAIN_MATCH TRUE`, 262 chars, and its MTP-on arm also 262 chars — the two agreed. Today:

| | 02:57 build | 04:0x build |
|---|---|---|
| MTP **on** arm text | 262 chars | 262 chars (**identical**) |
| MTP **off** arm text | 262 chars | **278 chars** |
| verdict | TRUE | **FALSE** |

Everything I can compare is identical between the two settings: env (dumped), argv, prompt
(`prompt_n=42`), `predicted_n=128`, pool bytes, carrier bytes, and the engine's own configuration
lines (`cap=8 routable=143 … width=8 … prefill slab NOT armed`, same `CGC-ADD-ORDER`). The one thing
that changed is the **binary**: `libllama.0.0.279.dylib` and `llama-server` were rebuilt at
**03:39:45/46**, i.e. after the first run and before the second, and the artifact that was linked at
02:54 is no longer on disk (only 0.0.275/0.0.277 from 09-16 and 0.0.279 remain), so the flip cannot
be bisected from artifacts.

What that does and does not license:

* it **does** invalidate the earlier TRUE as evidence of anything: it was measured with (a) different
  kernels on the two arms and (b) a different build from the one on disk now;
* it **does not** by itself establish that the verify path is wrong. The arm that changed across the
  rebuild is the *reference* (MTP off), so "the verify path diverges" and "plain decode changed" are
  both consistent with the table. The engine hunks currently uncommitted in `llama-context.cpp` /
  `llama-expert-cache.{h,cpp}` were read hunk by hunk: every one is behind a flag that is off by
  default (`CGC_LAYER_AHEAD_PREFETCH`, `CGC_HOOK_SPLIT`, `CGC_SLAB_HANDOFF`, `CGC_EP`), and
  `prewarm_hot` → `prewarm_hot_capped(cache, 0)` keeps the one-shot path. So the most likely carrier
  is an uncommitted engine edit that *was* in the 02:54 build and is not in the tree now (this box
  runs parallel sessions), which is exactly why the review's "commit, then measure" rule exists.
* to make it attributable, the driver now records `engine_digest` (md5 of `libllama`,
  `libggml-metal`, `libggml-base`, `llama-server`) **with every arm**, so a flip like this one is a
  fact about build identity instead of a mystery. This run's digests: `libllama aab787412e572550`
  (mtime 03:39:45), `libggml-metal 1be366306c604669`, `libggml-base 229f03a3996355da`,
  `llama-server 054fb22f04a01c5c`.

## 5. Driver defects found and fixed while doing this

1. `same_carrier` compared the whole carrier record including `path`; `run_server.sh` serves MTP=0 via
   the `$Q36` symlink, so two arms serving identical bytes reported `False`. Now compares
   `realpath` + `size` + `head_tail_hash`.
2. The `on-exact` verdict wording was inverted (it now reads "fast path exonerated" when the two
   MTP-on arms agree, which is the direction the comparison actually runs).
3. `layer_caps`/`pool_capacity` are recorded per arm from that arm's own artifacts. Note the launcher
   banner prints `layer_caps=40-40:256 (default)` even for MTP=0, where nothing is exported — the
   banner is wrong even though the export is right; the pool-capacity line from the server log is the
   one to trust.
4. The k=0 route is refused with its mechanism (see §2).

## 6. Next, in order of decision value

1. **Freeze the build, then re-run §3.** Commit or export the working tree, record `engine_digest`,
   and re-run `on,off` + `on,on-exact`. Until that is done, every "TRUE/FALSE" on this box is a
   statement about an uncommitted build that parallel sessions can change under it.
2. **Locate the flipped hunk, one side at a time.** On a frozen tree: re-run the MTP-off arm against
   the digests above and bisect the engine diff by building with `llama-context.cpp` /
   `llama-expert-cache.cpp` at `HEAD` vs at the working tree. The MTP-off arm alone is enough to see
   it (262 vs 278 chars), so each probe is one start.
3. **Is the divergence in the logits or in the sampling?** `on == on-exact` means it is not the pool
   fast path; the remaining candidates are the verify batch's own math and the sampler being applied
   to a K+1-row batch instead of one row at a time. The `cgc_logits_oracle_compare.py` /
   knifeedge machinery already dumps per-position logits — the first divergence is at a known place
   (char 205 of the rendered text), so the dump has a target.
4. **M4 consequence, if 3 lands on "logits".** Accept 0.465 would then be partly a *target-side*
   disagreement: the draft can be perfect and still be rejected because the verify batch's argmax is
   not plain decode's. That caps what "improve the draft" can buy and is the reason this belongs
   ahead of further accept tuning.

## 7. Files (all uncommitted, none staged)

* `scripts/run_server.sh` — hoisted `CGC_MM_BITIDENT` / `CGC_NO_PREFETCH`, MTP=1 defaults unchanged.
* `scripts/check/plain_match_ab.py` — `EQUALISE` on both arms, `on-exact` attribution arm, four
  bisect arms, `engine_digest` per run, k=0 refusal, carrier/geometry provenance, verdict direction.
* `Backup/phase_decomp/plain_match_{v2,v3,bisect}.json` — raw records.
