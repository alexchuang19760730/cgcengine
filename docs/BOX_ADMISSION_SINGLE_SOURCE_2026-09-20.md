# Box admission: two predicates, one decision — and which of them fails first on a small box

**Date** 2026-09-20 · host this 16 GiB Mac (M4, 16 KiB pages) · carrier Nail IQ3_XXS-denseIQ4X,
MTP on k=3, pool 8 GiB · tools `scripts/check/server_window.py`, `scripts/check/box_probe_compare.py`

## 0. One line

The harness probe and the launcher's guard were answering different questions and were allowed to
disagree; **the decision is now their conjunction** (`server_window.decision()["admits"]`), the
launcher's threshold table is read from the launcher at runtime, and the two definitions' verdicts
plus `refused_by` are recorded so a refusal can still be attributed. Measured today, on this box,
they disagreed in **10 of 10 paired samples, always the same way** — harness refuses, launcher
admits — which is exactly the pair that spends a window.

## 1. What the two definitions actually are

| | metric | source |
|---|---|---|
| harness | `reclaimable = free + purgeable + inactive` MB, absolute | `decode_window_harness.vm_free_mb`, floor `NEED_MB = 8000` |
| launcher | `memory_pressure -Q` "free percentage" ≥ per-class pct, **fraction of physical RAM** | `run_server.sh:856` (probe) + `:582` (class table) |

They are not two readings of one quantity: one is an absolute byte count, the other a fraction of
RAM. That difference is what this document is about.

## 2. The disagreement, measured (not asserted)

`python3 scripts/check/box_probe_compare.py --n 4 --interval 3` — four paired samples, one per row:

```
   free   inact   purge    comp    swap |  reclaim   need  harness |  FREE%  req% oth  launcher | DECISION
   2384    4511     208     771    3245 |     7099   8000  refuses |     83    40   0    admits |     busy
   2345    4511     221     771    3245 |     7081   8000  refuses |     83    40   0    admits |     busy
   2379    4511     205     771    3245 |     7095   8000  refuses |     83    40   0    admits |     busy
   2371    4511     212     771    3245 |     7094   8000  refuses |     83    40   0    admits |     busy

samples=4  quiet=0  disagreements=4  (all binding=harness)  refused_by=['harness']
reclaimable range 7081-7099 MB against a 8000 MB floor; swap moved 3245 -> 3245 MB
```

An earlier six-sample run the same day gave the same verdict with reclaimable 7294–7705 MB and
`FREE%` 83–84. The reverse state — launcher refusing while the harness is quiet — also occurred
today at a different moment (`memory_pressure` 12% with ~7.5 GB reclaimable); it is covered by a
selftest rather than by a sample, and it is the one the conjunction now refuses.

## 3. The decision, and what a refusal can be attributed to

- `admits = harness_conjunction AND launcher_conjunction` (the launcher side only when its table was
  readable; an unreadable table cannot be evaluated and is recorded as `launcher_table: unreadable`,
  not as agreement).
- `harness_admits`, `launcher_admits`, `harness_terms`, `launcher_terms` — per-definition verdicts.
- `refused_by` — every predicate that refused, e.g. `["harness"]`, `["launcher"]`, or both;
  `binding` is its first entry, so "the harness refused although the launcher would have started it"
  stays distinguishable from the reverse.
- `quiet()` now emits a reason when the launcher alone refuses
  (`launcher_refuses(full-mtp failed free_ok)`), because a `False` with an empty reason is a guard
  nobody can act on.

`agree` is still recorded and is deliberately **not** claimed when the launcher's probe could not be
read: the launcher defaults that value to 0 and refuses, but a default is not a measurement.

## 4. Which definition fails first as the box shrinks

`python3 scripts/check/server_window.py scaling` — computed from the launcher's own requirement and
`NEED_MB`, so it cannot drift into a hand-typed table:

```
need_mb=8000 class=full-mtp crossover=19.53 GiB
    RAM  floor%  req%  binds      workload_fits  verdict
      8    97.7    40  harness    False          harness: floor 97.7% > req 40%
     12    65.1    40  harness    False          harness: floor 65.1% > req 40%
     16    48.8    40  harness    False          harness: floor 48.8% > req 40%
     20    39.1    40  launcher   False          launcher: req 40% >= floor 39.1%
     24    32.6    40  launcher   True           launcher: req 40% >= floor 32.6%
     32    24.4    40  launcher   True           launcher: req 40% >= floor 24.4%
```

The two definitions cross at **19.53 GiB RAM** (`need/req`), which is why the harness is the binding
one on this 16 GiB box and why that fact does not generalise upward.

Answering the question directly:

1. **The absolute floor fails first.** Its percentage of RAM grows as the box shrinks (48.8% at
   16 GiB → 97.7% at 8 GiB → >100% below ~7.8 GiB, where it becomes unsatisfiable by construction),
   while the fractional gate's demand in bytes shrinks with the box.
2. **The fractional gate is the dangerous one, not the safe one.** It never fails first, it gets
   *easier* to pass as RAM shrinks, and the work does not shrink with it: `run_server.sh:854`'s own
   arithmetic is a 13.2 GB model (`--no-mmap`) plus an 8 GiB pool ≈ **21 GiB** resident. Every
   configuration in the table below 20 GiB therefore satisfies a 40%-free gate while being unable to
   hold the work without swapping — the `workload_fits` column is that second, independent check, and
   it is the column that decides whether an admission is meaningful at all.
3. **Corollary for a smaller target.** On an 8–12 GiB box the correct predicate is not the harness
   floor either (97.7% of RAM is a demand no running machine meets, and the lazy `purge` that once
   got this box from 6157 → 8465 MB would not save it). A predicate that means something there has to
   be scaled to the *work*, e.g. `need_mb = min(NEED_MB, k × workload)` with the pool sized down in
   the same breath — a fraction of a RAM the work no longer fits in is measuring the wrong thing.

## 5. Two launcher facts read off `run_server.sh` (reported, not changed here)

1. **The physical-RAM term is inert.** `cgc_memory_guard_req()` (line 582) returns `0` as its first
   field for all five classes, so `PHYS_MEM_GB < MEM_REQ_PHYS_GB` is false everywhere and the class
   table's model gate is effectively disabled. Only the fraction and the other-servers count decide.
2. **A shell precedence bug in the guard's own condition** (lines 869 and 883, the same condition
   duplicated): `if [ A ] || [ B ] || [ C ] && [ "$DUMP_ONLY" = "0" ]; then` — in POSIX `sh`, `&&`
   and `||` have equal precedence and associate left to right, so this parses as
   `A || B || (C && DUMP_ONLY=0)`. Consequently `DUMP_ONLY=1` bypasses only the other-servers term;
   a low `FREE_PCT` still blocks. That is why "dump-only" is not usable as a dry run, which is the
   one thing that would have let this tool ask the launcher for its verdict instead of re-implementing
   it. Fixing it changes launcher behaviour, so it is left to the owner of that file; the correct
   form is a grouped condition `{ A || B || C; } && [ "$DUMP_ONLY" = "0" ]`.

## 6. What would falsify this

- A paired sample where the launcher refuses while the harness admits **and** the launch would have
  succeeded — that would say the fraction carries information the byte floor lacks. The conjunction
  would then need a third predicate, not a weaker one.
- A box where `reclaimable` and `memory_pressure`'s free percentage agree in direction across many
  samples — that would collapse the two into one metric, and one of the two probes could go.
- A workload whose resident footprint tracks RAM (it does not here: the GGUF and the pool are fixed
  files), which is the only case where a fraction of RAM would be the right form.

## 7. Files

- `scripts/check/server_window.py` — decision, `refused_by`, `scaling`, selftest (all assertions
  above, including the synthetic reverse disagreement and the crossover arithmetic).
- `scripts/check/box_probe_compare.py` — the paired sampler; prints both definitions **and** the
  decision; it is what produced §2.
- Uncommitted at the time of writing; no server left running.
