# Every result now says what produced it — and `--summary` refuses cross-geometry comparisons

Date: 2026-09-14 · repo: `flashkv-devserver` · branch: `demo/sweet-spot-windows-fix`

## The problem this closes

A pool sweep produces rows that look identical in shape but may describe **different
computations**. Nothing in the artifact said which. Two concrete instances, both already paid for:

| Instance | What it looked like | What it was |
|---|---|---|
| `cap_probe_iq3_pool8gb.json` (2026-09-11) | a record saying 106 slots/layer | a geometry the engine no longer had; the sweep's cap came from it |
| oracle reference captured 21:58, engine fixed 22:54 | M1 0/53 | code drift, read as a pool regression for hours |
| an old 4 GiB row next to a new 8 GiB row in the table | a pool comparison | two engines, two loaders, two binary builds |

The rule from here on: **a measurement without provenance is not comparable with anything, and the
table says so instead of implying otherwise.**

## What a result record carries now

Written into *both* artifacts of a cell — `knifeedge_*.json` (the suite's own output) and
`combo_*.json` (the harness record with the gates) — by `record_provenance()`:

```json
"provenance": {
  "when": "2026-09-14T11:02:49",
  "pool":    {"gb": 8, "bytes": 8589934592, "cap": 8, "min_layer_slots": 143,
              "usable_slots": 142, "cap_max": 17, "floor": 5, "verdict": "USABLE",
              "launched_min_layer_slots": 143, "launched_n_slots": 143},
  "engine":  {"source_digest": "e9427e44…", "sources": {...},
              "geometry_digest": "434daf1b…", "geometry_sources": {...},
              "head": "e8b85393d", "last_commit": "...", "dirty": true,
              "binary": {"llama-server": {"size": 49984, "head": "924e2810…"}, ...}},
  "weights": {"realpath": "models/gguf/Nail-…-IQ3_XXS-denseIQ4X.gguf",
              "size": 13663116512, "digest": "c55499c1…", ...},
  "launch":  {"mtp": true, "spec_n_max": 3, "layer_caps": "default", "extra_env": []},
  "suite":   {"profiles": "qa-zh", "per_profile": 1, "greedy_repeats": 2,
              "repeats": 2, "temperature": 0.4, "max_tokens": 64},
  "harness": {"script_digest": "713d022a…"}
}
```

`pool` carries **both** numbers on purpose: the *predicted* slot geometry
(`pool_feasibility.py`'s arithmetic) and the *launched* one (what the engine reported at load).
A disagreement is now visible per row instead of needing a separate investigation.

`engine.binary` is content-based (size + hash of the first 64 KiB), never mtime: a rebuild that
changes nothing must not invalidate a comparison. It is narrowed to `llama-server` + `libllama*` +
`libggml*`, because the other `*-impl.dylib` files are separate tools (bench/quantize/perplexity)
that cannot change what the server computes.

## The comparability key (what must MATCH)

```python
record_geometry_key(rec) -> {
  "engine":        source digest of llama-{expert-cache,context,graph}.cpp
  "pool_geometry": source digest of llama-model-loader.cpp + expert-cache.{cpp,h}
  "binary":        (name, size, head-hash) per artifact
  "weights":       (realpath, size, digest)
  "cap":           CGC_POOL_MAX_TOKENS
  "mtp", "spec_n_max", "layer_caps", "extra_env"
  "suite":         (profiles, per_profile, greedy_repeats, repeats, temperature, max_tokens)
}
```

`suite` is in the key because **the shape of the measurement is part of the experiment**: a
1-question smoke run and a 48-question suite both land in `iq3_pool8gb`. (This was found by doing
it: the end-to-end verification below wrote a 1-question run under the plain label, which would
have sat next to multi-question rows as if the pass rates were comparable. The smoke artifacts are
now named `*_smoke1q` explicitly.)

Deliberately **excluded**:

* **the pool size** — it is the variable under test;
* **runtime numbers** (rss, free%, tok/s, pass rates) — noise must not split a group;
* **`harness.script_digest`** — it is recorded for forensics, but a harness edit (a print, a new
  gate) must not retroactively declare every past measurement incomparable;
* **the newest commit over those paths** — `source_digest` already covers content, and a commit
  that did not change the bytes is inert (the same distinction `_stamp_guard` makes).

> **Hazard, stated plainly:** the key is only as stable as its own definition. **Changing which
> fields are in it invalidates every existing record** — correctly, because the old records cannot
> be compared on the new terms, but it does mean one re-measure. This happened once already, while
> narrowing the binary stamp during development: the next run printed
> `REDO: … measured under a different geometry (built binary)`.

## `--summary` groups before it prints

`summary_table()` now reads **both** `combo_*.json` and `knifeedge_*.json` (merged by label, so a
suite result no longer hides from the summary), then groups rows per model by the key above.

| State | Meaning | What the table does |
|---|---|---|
| `OK` | one geometry group | rows are laid out together; `pool size is the only variable. OK` |
| `UNATTRIBUTED` | one group, no provenance at all | rows printed, and `comparability UNVERIFIED` — no ranking may be read off them |
| `REFUSED` | two or more groups | rows printed **in separate blocks**, the differences named, and **no cross-group claim made** |

Real output today (one stamped cell next to 35 pre-stamp cells):

```
iq3: REFUSED -- 2 geometry group(s); rows from different groups are NOT comparable and are
     printed separately below, with no cross-group claim made:
    - group(s) ['unattributed'] carry NO provenance, so none of their fields can be compared
      with anything (records written before the stamp existed) -- re-measure them if they are
      meant to sit beside the 1 stamped group(s)
[iq3 group unattributed]  35 row(s)
  (no provenance: recorded before the stamp existed -> comparability unverified)
...
[iq3 group f80ae7fc]  1 row(s)
  engine=e9427e44 geometry=434daf1b head=e8b85393d (dirty) weights=Nail-…-IQ3_XXS-denseIQ4X.gguf@13663116512 cap=8 mtp=True spec_n_max=3 layer_caps=default extra_env=-
```

A group with no provenance differs from everything *by construction*; reporting nine field diffs
for it (starting with a 38-element binary tuple) would bury the one fact that matters, so it is
reported as that fact instead.

The same verdict is printed at the end of a live sweep (`print_matrix`), so a mixed table can never
be read as a comparison just because it was printed by the run that produced it.

## Adjacent fix: resume could never skip a full run

`run_combo`'s RESUME check read `knifeedge_<label>.json` for `up`/`gates` — but that file is
`flip_rate.py`'s suite summary and contains neither, so the check always fell through to
`REDO` and **no completed quality run could ever be skipped**. It now reads the harness's own
`combo_<label>.json` and requires that it be *reusable*, with the reason printed when it is not:

```
[iq3_pool8gb] SKIP (complete, attributable result exists; --force to redo)
[iq3_pool8gb] REDO: the cached result is not reusable -- it is a --gates-only run, not a quality
              measurement; re-running
```

Reasons it refuses: never came up · no gate evidence · `--gates-only` · skipped
(`low_memory`/`foreign_server`/`infeasible_*`) · **no provenance** · **provenance differs** (with
the moved fields named).

## Verification

* `scripts/check/feasibility_gate_selftest.py` — **71 assertions, ALL PASS**, offline (no GGUF, no
  GPU, no launch). Section 8 pins the grouping rules: pool size excluded from the key; a different
  cap / weights / loader digest / built binary / **suite shape** each force `REFUSED`; runtime
  numbers do **not** split a group; stamped + unstamped is `REFUSED` and names the missing
  provenance (as one fact, not as nine field diffs).
* End-to-end, on the real machine: a cell was measured (`--tag`-less, 1 question; stamped into both
  artifacts, predicted slots = launched slots = 143), the **same command re-run printed SKIP and
  launched nothing** (`servers: 0`), and `--summary` refused the mixed table as shown above.
* Artifacts: `Backup/knifeedge_matrix/knifeedge_iq3_pool8gb_smoke1q.json` +
  `combo_iq3_pool8gb_smoke1q.json` (labels rewritten to match their filenames) are the only stamped
  records in the tree so far.

## Limits, honestly

* The stamp is only as good as the fields in it. A change to the harness that alters *how a cell is
  measured* (not what it prints) would not move any stamped field and would not be caught.
* `weights.digest` is a 4 MiB head + 4 MiB tail sample by default (`CGC_ORACLE_MODEL_DIGEST=full`
  for a full sha256), so a same-size change confined to the middle of a GGUF would slip through.
  The same limitation already applies to the oracle guards.
* Existing records have no provenance and will stay `UNATTRIBUTED` until re-measured. That is the
  point — but it does mean the current summary refuses comparisons for every model that has any
  pre-stamp row.
* `launched_*` values come from the launch log's own parsing; if a future build renames those log
  fields they go `None` silently (the predicted values remain).
