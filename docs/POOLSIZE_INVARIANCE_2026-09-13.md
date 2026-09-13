# Pool-size invariance gate — rebuild and hardening (2026-09-13)

> **NOTE (added later the same day): this study ran pinned to `--pool-cap 6`.**
> `CGC_POOL_MAX_TOKENS` turned out to be a **quality** knob: cap 6 scores **0/48** on the
> 48-question suite, cap 8 scores **9/48**, at identical speed. So the invariance below was
> established at a shape where the model answers nothing. It has since been **redone at cap 8**
> — same verdict, and now over **117** step keys instead of 103 (see
> `CAP_AB_2026-09-13.md` §3b). The default reference has been replaced accordingly:
> `ref_iq3_pool8gb.jsonl` is now the cap-8 dump (`d03574ec…`), the cap-6 one is archived as
> `ref_iq3_pool8gb_STALE_cap6.jsonl` and is **refused** by `_cap_guard` against any cap-8 dump.
> Everything below about the *harness defects* (§3–§7) still stands; the numbers in §1 are the
> cap-6 ones.

**Question.** Does the expert-cache pool size (4 / 6 / 8 GiB) change the model's numerics?
Rebuild the gate against a *current* reference oracle and report **M1 (bit-identical)** and
**M2 (argmax agreement)** as two separate metrics, never merged into one verdict.

**Answer.** No. At a pinned cap, 4 / 6 / 8 GiB produce **byte-identical logits**: every dump is
103/103 on M1, M2 and M3, and all five files hash to the same md5.

---

## 1. Result

Model `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf` (13,663,116,512 B), ctx 8192,
ngl 99, `CGC_POOL_MAX_TOKENS=6` pinned, nothink-ChatML template, `--gates-only`.

| gate | 4 GiB | 6 GiB | 8 GiB |
|---|---|---|---|
| **oracle M1** (bit-identical logits) | **103/103** | **103/103** | *ref* |
| **oracle M2** (argmax agreement) | **103/103** | **103/103** | *ref* |
| oracle M3 (top-k set agreement) | 103/103 | 103/103 | *ref* |
| template (closed scaffold) | PASS | PASS | PASS |
| buffer-nil | PASS | PASS | PASS |
| union-fit | PASS (48 ≤ 70) | PASS (48 ≤ 106) | PASS (48 ≤ 142) |
| preflight | PASS | PASS | PASS |

`n_slots` = 71 / 107 / 143; `union_max` = 48 in every cell, so the pool path is unreachable-by-
construction at all three sizes (that is what union-fit certifies, independent of which experts
this prompt happens to route to).

**Byte-level proof** — five dumps, one md5:

```
c59a4bb1ad59919f0aa7e1b41b4f021d  ref_iq3_pool8gb.jsonl            (reference)
c59a4bb1ad59919f0aa7e1b41b4f021d  oracle_iq3_pool4gb_inv4gb.jsonl
c59a4bb1ad59919f0aa7e1b41b4f021d  oracle_iq3_pool6gb_inv6gb.jsonl
c59a4bb1ad59919f0aa7e1b41b4f021d  oracle_iq3_pool8gb_inv8gb.jsonl
c59a4bb1ad59919f0aa7e1b41b4f021d  oracle_iq3_pool4gb_refpin.jsonl    (re-verification run)
```

All 103 distinct `(step, token_idx, ctx_type)` keys, `only_A = 0`, `only_B = 0`, and
`num_eq_dec_eq = 103, num_eq_dec_ne = 0, num_ne_dec_eq = 0, num_ne_dec_ne = 0`.

---

## 2. Why the old reference could never pass — it was superseded, not corrupted

The previous `ref_iq3_pool8gb.jsonl` scored M1 0/53 against *both* the old and the new binary.
That looked like a regression; it was not.

| event | time |
|---|---|
| old reference captured (`ref_iq3_pool8gb.jsonl`, 88 keys, md5 `7a608f16…`) | **2026-09-11 21:58** |
| commit `d222a1d6e` — *"fix(expert-cache): make decode pool-independent and stop the no-think scaffold echo"* | **2026-09-11 22:54** |

The reference predates the commit that made decode pool-independent by **56 minutes**. Comparing
the two: 88 vs 103 keys, only 63 keys in common, and **all 63 overlapping keys differ** in
`row_fnv1a64`. That is a different computation, not a drifting one — so the reference was made
obsolete by a deliberate correctness fix. It is archived as
`ref_iq3_pool8gb_STALE_pre-d222a1d6e.jsonl` and the fresh dump is installed as the reference.

---

## 3. Harness defect A — the gate compared only a prefix of the dump

`oracle_gate` runs first by design (its comment: the dump's early ubatches must line up with the
reference run's). But the template and preflight probes that follow keep **appending to the same
JSONL**, so the comparison measured a prefix — 53 of the eventual 103 keys. That silently
*understates* the gate rather than failing it, which is the worse kind of error: a smaller N is
easy to read past.

**Fix.** `oracle_recompare()` re-runs the comparison on the finished file at cell end and records
it as `gates.oracle_final`. The complete file is a strict superset of what was compared and no
extra request is sent, so this is free and strictly stronger. The original mid-cell verdict is
kept alongside it, never overwritten.

Verified live: mid-cell `M1 53/53` → **`oracle_final M1 103/103 M2 103/103 M3 103/103 (n=103)`**.

## 4. Harness defect B — the cap is a quality knob, and a mismatch is silent

Found while verifying the rebuild. The re-verification run omitted `--pool-cap`, so:

- `cap_for` fell through to `probe_pool_cap`, which starts at `--pool-cap-max` (8) and works
  down. At 4 GiB, usable = 70 and cap 8 → union 64 ≤ 70 is already safe, so it accepted **cap 8**.
- The reference was taken at **cap 6**. Comparing cap8 vs cap6 scored **M1 3/98, M2 12/98** —
  a total collapse that reads as "the pool broke numerics".

It is a shape difference. `probe_pool_cap`'s own docstring records the size of the effect: *"8GB
with pmax=8 vs 8GB with pmax=6 agreed on argmax only 15.4% of steps — the cap IS a quality knob."*
The cap clamps `n_batch`, hence the ubatch shape and the prefill chunking.

Two properties made this a trap rather than a blip: the probe's answer depends on **cache state**
(`--force` bypasses the cache and re-probes), and it is **ambiguous** — at 4 GiB both 6 and 8 are
safe, so nothing in the output says which one the reference used.

**Fix.** The cap is now written to a sidecar beside every dump (`oracle_<label>.jsonl.cap`), and
`_cap_guard()` refuses to compare across a difference:

```
CAP MISMATCH: reference was produced at cap=6, this dump at cap=8. The cap clamps n_batch, so
these are different computations -- any M1/M2 number would be a shape artifact, not a pool
difference. Re-run with --pool-cap 6.
```

Guard tested both ways (cap6-vs-cap6 → passes through; cap6-vs-cap8 → fires). Sidecars were
backfilled for the reference and the three invariance dumps at cap=6.

**Operational rule.** Always pass `--pool-cap <N>` explicitly when producing *or* using an oracle
reference. `--pool-cap auto` is not reproducible across runs or across cache states.

---

## 5. Reproduction

```bash
cd /Users/alexchuang/Documents/flashkv-devserver
# produce a reference (pin the cap!)
python3 scripts/check/knifeedge_matrix.py --models iq3 --pools 8 --pool-cap 6 \
  --oracle-ref Backup/knifeedge_matrix/ref_iq3_pool8gb.jsonl --gates-only \
  --kill-existing --tag ref8g
# compare 4 / 6 GiB against it (same pinned cap)
python3 scripts/check/knifeedge_matrix.py --models iq3 --pools 4,6 --pool-cap 6 \
  --oracle-ref Backup/knifeedge_matrix/ref_iq3_pool8gb.jsonl --gates-only \
  --kill-existing --tag inv
```

---

## 6. Harness defect C — a stale reference fails *loudly* and points at the wrong subsystem

This is the failure that cost the most time today, so it gets its own guard.

A reference captured at **09-11 21:58** was compared against an engine fixed at **09-11 22:54**.
The verdict was **M1 0/53** — a total collapse. It looked exactly like a real numeric regression
in the pool path, so it was chased as one (including an old-binary vs new-binary A/B) before the
timeline was checked. A stale reference cannot be recognised from the numbers: it fails hard, the
failure is reproducible, and it names the wrong subsystem.

**Guard.** Every dump now gets a provenance sidecar (`oracle_<label>.jsonl.cap`, JSON) recording
the cap, a timestamp, and a **source stamp** of the code that decides the numerics:

```json
{ "cap": "6", "created": "2026-09-13T02:10:45",
  "stamp": { "head": "c14785bda", "dirty": true,
             "last_commit": "d222a1d6e 1789138468 fix(expert-cache): make decode pool-independent…",
             "source_digest": "e50eb794…b2cb",
             "sources": { "src/llama.cpp/src/llama-expert-cache.cpp": "5f631e07b52ab35a",
                          "src/llama.cpp/src/llama-expert-cache.h":   "8060f663fd103e06",
                          "src/llama.cpp/src/llama-context.cpp":     "c38debfc94d0189c",
                          "src/llama.cpp/src/llama-graph.cpp":       "7c00e5a44b5cddc6" } } }
```

`_stamp_guard()` checks it **before** the comparison and refuses with a named error:

```
STALE REFERENCE (content_changed): … ref last_commit = d222a1d6e … | current last_commit = … |
changed sources: llama-expert-cache.cpp. Any M1/M2 number here describes the code drift, not the
pool size. Rebuild the reference from the current tree, then re-run …
```

`--allow-stale-oracle` waives it (warns on stderr and proceeds) for the case where the drift is
known to be numerics-neutral.

### Why the stamp needs BOTH a content digest and a commit

Neither half is sufficient, and today's live state is the proof:

| | value |
|---|---|
| HEAD's newest commit over the numerics paths | `d222a1d6e` — **09-11 22:54** |
| those files' actual last edit | **09-13 00:31 / 00:44 / 01:35** |
| binary rebuilt | **09-13 01:36:03** |
| reference dumped | **09-13 01:51:27** |

A **commit-only** rule would call this tree unchanged (nothing has been committed since 09-11)
while it is in fact three uncommitted edits ahead. A **content-only** rule can reject a stale
reference but cannot say which commit moved or who changed it. So both are recorded, and the
two failure modes are reported as distinct kinds:

- `content_changed` — the engine itself differs → refuse, rebuild the reference.
- `commit_changed` — contents identical, only the newest commit over those paths moved → refuse
  by default (the user-facing rule) but the message says the numerics are unchanged, so
  `--allow-stale-oracle` is the reasonable response.

`NUMERIC_SOURCES` is deliberately narrow (`llama-expert-cache.{cpp,h}`, `llama-context.cpp`,
`llama-graph.cpp`). Harness and probe scripts are excluded: editing them cannot move a logit.

### Provenance of the current reference (why backfilling was legitimate)

The existing reference predates this mechanism, so it was stamped by measurement, not by
guessing: its three relevant sources were last edited **01:35:55**, the binary was built
**01:36:03** (8 s later, i.e. from exactly that content), and the dump was written **01:51:27**.
The current stamp therefore *is* the stamp of that reference.

### Guard tests (all five, plus a live cell)

| test | expected | result |
|---|---|---|
| fresh reference vs candidate | allow | `None` |
| legacy plain-cap sidecar (`6`, bare `{"cap": 8}`) | still parse | `'6'`, `'8'` |
| reference digest tampered | refuse, name the file | `content_changed`, `['llama-expert-cache.cpp']` |
| same, with `--allow-stale-oracle` | warn + allow | waived, stderr warning |
| only `last_commit` moved | refuse as `commit_changed` | `commit_changed` |
| **live cell** (iq3 / 4 GiB / cap 6) | must not be refused | `oracle_final M1 103/103 M2 103/103 M3 103/103`, all 5 gates PASS |

The live cell is the important one: a guard that always refuses is worse than no guard.

### Operational rule

Pin `--pool-cap` **and** rebuild the reference whenever `NUMERIC_SOURCES` changes. Both the cap
and the stamp are now *proven* rather than assumed, and a reference that cannot prove itself is
reported as `REFERENCE PROVENANCE UNKNOWN` instead of being silently trusted.

## 7. Harness defect D — the sidecar recorded the code but not the WEIGHTS

The source stamp covers one half of "same engine". The other half is *which model*. A reference
taken from one GGUF could be compared against another, and every gate would report a clean
comparison of two different models.

That is not hypothetical: this matrix's `iq3` column was silently resolving to the IQ4_XS file,
so two columns measured a single model. Nothing in the record showed it — the discovery came
from reading timestamps and sizes by hand, not from the harness.

The sidecar now carries a `model` block:

```json
"model": { "path": "models/gguf/Nail-…IQ3_XXS-denseIQ4X.gguf",
           "realpath": "/Users/…/Nail-…IQ3_XXS-denseIQ4X.gguf",
           "size": 13663116512, "mtime": 1787931174, "inode": 153048921,
           "digest": "c55499c1f035c02a…", "digest_mode": "sampled(head+tail)" }
```

`_model_guard()` refuses a comparison whose weights differ:

```
MODEL MISMATCH: ref_iq3_pool8gb.jsonl was produced from /…/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
(18215220576 B, sampled(head+tail)=0f1e2d3c4b5a), but this run measures
/…/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf (13663116512 B,
sampled(head+tail)=c55499c1f035). Any M1/M2 number would compare two different models.
Regenerate the reference for this GGUF, or pass --allow-model-mismatch if that is deliberate.
```

**Why the digest is sampled, not full.** `realpath + size + inode + mtime` are free and already
catch the failure that actually happened (IQ4_XS is 18.2 GB vs IQ3's 13.66 GB). A sampled digest
(head 4 MiB + tail 4 MiB) additionally catches a *same-size* swap, because a re-quantized file
differs in its GGUF header/KV metadata block and in its tail. A full sha256 is available via
`CGC_ORACLE_MODEL_DIGEST=full` but is **not** the default: reading 13.7 GB evicts the page cache,
and page-cache state (7721 MiB/s hot vs 1225 MiB/s cold) is exactly what these measurements turn
on. Cost of the default: 14–22 ms.

### Guard tests

| test | expected | result |
|---|---|---|
| `model_stamp()` on the real GGUF | stable digest, cheap | 22 ms; second call identical |
| reference with no model block | refuse (unknown) | `MODEL IDENTITY UNKNOWN` |
| after backfill | allow | `None` |
| ref = IQ4_XS identity vs IQ3 on disk | refuse, name both | `MODEL MISMATCH` + both sizes |
| same path+size, tampered digest | refuse | `model_mismatch` |
| `--allow-model-mismatch` | warn + allow | waived (stderr warning) |
| all 7 existing sidecars after backfill | allow | `None` for every one |

All 7 sidecars were backfilled with the IQ3 identity (each was produced from that GGUF).

## 8. Honest limits

- **One prompt.** The oracle dump is driven by a single deterministic probe; 103 steps is the
  whole trace. Pool-*size* invariance is what is proven here — not invariance under a different
  prompt distribution.
- **One model.** iq3 only. The iq4 column was not re-run in this pass; the earlier finding that
  the two matrix columns can resolve to the same GGUF file is the reason `--models` is verified
  against `realpath` + size before launch (see the `[model-id]` line: it printed
  `exists=True size=13663116512`).
- **The RIG-clean state matters.** All cells ran with swap low; a dirty machine (swap 15–18 GB
  from repeated 13 GB model loads) makes wall-clock numbers unusable. Counts and logits are
  unaffected, which is why the invariance claim is a *numeric* claim.
- `preflight` answered `15+27=42` with no echo and `finish=stop`, and the template tail shows a
  closed scaffold (`…</think>

答：`). That is the probe only — it is not a 48-question quality score.

## 9. Files touched

- `scripts/check/knifeedge_matrix.py` — `oracle_recompare()`, `source_stamp()`, `_oracle_meta()`,
  `_cap_sidecar()`, `_cap_guard()`, `_stamp_guard()`, `model_stamp()`, `_model_guard()`,
  `_oracle_guard()`, `_write_oracle_meta()`, `NUMERIC_SOURCES`, the `--allow-stale-oracle` and
  `--allow-model-mismatch` flags, and `oracle_final` in all three result records.
- `Backup/knifeedge_matrix/ref_iq3_pool8gb.jsonl` — replaced with the current-binary reference;
  old file archived as `ref_iq3_pool8gb_STALE_pre-d222a1d6e.jsonl`.
- Sidecars (JSON: cap + timestamp + source stamp + weights identity):
  `ref_iq3_pool8gb.jsonl.cap`,
  `oracle_iq3_pool{8gb_ref8g,4gb_inv4gb,6gb_inv6gb,8gb_inv8gb,4gb_refpin,4gb_stamp1}.jsonl.cap`.

The three guards are independent, and each one exists because it produced a large and spurious
verdict on this box:

| guard | refuses when | records |
|---|---|---|
| `_cap_guard` | candidate ran at a different cap | cap |
| `_model_guard` | different weights | realpath, size, inode, mtime, sampled digest |
| `_stamp_guard` | engine source changed / commit moved | per-file digests + last commit |

### What this does NOT catch

- A change to files **outside** `NUMERIC_SOURCES` that still moves the numerics — e.g. a Metal
  backend or ggml op edit.
- A rebuild of the binary from *identical* sources (the stamp does not hash the dylib; it hashes
  the sources that produce it).
- A **same-size, same-head, same-tail** GGUF swap. Set `CGC_ORACLE_MODEL_DIGEST=full` for the
  case where that matters more than page-cache state.
- The three guards have since been exercised by **four live cells** (the cap-8 redo: one
  reference + 4/6/8 GiB, all four PASS with `n=117`). At the time of writing this section they
  were only unit-tested, and the end-to-end run had been deferred to avoid running a second
  8 GiB-pool server while another measurement owned the machine.
- §5's reproduction recipe says to pin the cap; as of the cap A/B the value to pin is **8**.
  An `auto` cap is not reproducible (the probe is cache-dependent and both 6 and 8 can be
  legal at 4 GiB).

Nothing committed. No servers left running.
