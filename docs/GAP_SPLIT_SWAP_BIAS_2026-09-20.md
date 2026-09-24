# Does over-commit inflate the hook term? A below-floor gap split, with swap on every reading

**Date** 2026-09-20 22:05 · same build as the 21:49 baseline (`libggml-base.0.19.0.dylib` md5
`fea88a48e636389d`, verified against the gate summary's per-file digest) · arm: Nail carrier, MTP on
k=3, pool 8 GiB, temp 0.4, chat door, `--n-predict 96` · runner
`Backup/phase_decomp/gap_split_swap.py`, products `/tmp/gap_split_swap/{samples,readings,report}.json`

## 0. One line

**The predicted inflation is not visible, and the verdict does not move.** In a heavily
over-committed regime (reclaimable **1.2–1.5 GB**, swap **3.9–4.1 GB** for 127 of 128 readings) the
hook's share of the host accounting is **0.238** and `idle/gap = 0.838`, against the clean-window
baseline's **0.264** and **0.90** — both still above the 0.80 HOST WORK threshold, and the shift is in
the *opposite* direction from the hypothesis.

## 1. Why this run exists

`gap_split_capture.py` refuses to lower the documented 8000 MB floor, with the argument that under
memory pressure the `hook` term absorbs page-fault stalls and therefore biases the ratio toward "host
work" — the answer that would justify building L3. That was an argument, never a number. This runner
takes the same reading in the over-committed state, and stamps the box state onto every aggregate
line so the bias becomes measurable.

Design notes that matter for reading the result:

- the floor was set to **5500 MB deliberately** (`--need-mb 5500`), which `mtp_accept_ab.py` records as
  `below_documented_floor: true` in its own provenance — the run is labelled, not disguised;
- **the box state is measured, not asserted**: a sampler records reclaimable/swap/free/inactive every
  0.5 s together with the server log's byte size, so each engine line is joined to the box state at the
  byte offset it appeared at (`stamp()`);
- the line format is not redefined: the regex is imported from `gap_split.py`;
- **the metric is the hook's SHARE, not raw ms.** Under pressure every term grows, so a rising hook in
  ms is not evidence that the hook absorbed anything. `hook / (poll+hook+tail+submit)` separates "the
  machine slowed" (share flat) from "the hook absorbed the stall" (share rises).

## 2. The reading

```
box during the run (per reading):  reclaimable 1147-10016 MB, median 1321 MB
                                   swap        3917-4068 MB,  median 3932 MB
readings > 3000 MB reclaimable:    1 of 128          <- the run is ONE regime

                              over-committed (22:05)   baseline (21:49)   delta
hook_share (hook/host_total)          0.238                 0.264         -0.026
idle/gap                              0.838                 0.90          -0.06
hook         (median ms)             39.11                 51.29         -12.18
idle_host    (median ms)             45.87                 57.88         -12.01
poll         (median ms)            115.96                130.16         -14.20
gap          (median ms)             54.74                 64.35          -9.61
host_total   (median ms)            165.50                192.88         -27.38
box: reclaimable / swap          1321 / 3932           7090 / 3245
```

Every reading carries its own box state; the first five and the last five, as they came out:

```
reclaimable= 1218 MB  swap=4068 MB  hook=289.55  gap=302.41   (warm-up: hook is largest here)
reclaimable= 1429 MB  swap=4060 MB  hook=547.32  gap=541.42
reclaimable= 1487 MB  swap=4060 MB  hook=445.87  gap=452.25
...
reclaimable= 1221 MB  swap=3917 MB  hook= 33.99  gap= 45.01
reclaimable= 1221 MB  swap=3917 MB  hook= 40.94  gap= 48.77
reclaimable=10016 MB  swap=3917 MB  hook= 19.89  gap= 31.67   (final reading, post-teardown)
```

## 3. What this can and cannot attribute

1. **The within-run regression is not usable here.** The box collapsed to 1.2–1.5 GB within the first
   readings and stayed there: the low/high reclaimable quartiles are **1182 vs 1429 MB** — a 250 MB
   contrast, not the 5.8 GB the question implies. `hook_share` medians in those quartiles are 0.23769
   and 0.23761 (difference 8e-5). The Pearson values (hook_share vs swap +0.12, vs reclaimable −0.06)
   are reported as descriptive only; they are computed over a span too narrow to test anything.
2. **Resolution, stated so "no change" means something:** `hook_share` MAD = **0.0555**, so with
   N=128 readings the smallest shift this run could detect is ≈ **0.012** (hook in ms: MAD 16.2 ms →
   3.6 ms). The cross-run difference (−0.026) is above that floor — it is real, and it is *smaller*
   than the content-driven spread between arms of this same workload.
3. **The cross-run delta is confounded by content, so it is not a box statement.** Same command, same
   prompts, but temp 0.4 gave a different generation: accept **52.7%** (baseline 73.9%), mean_len
   **2.57** (3.17), and `cb` 38.7 vs 50.8 ms — a different expert union stream. `host_total` differs
   by 14%. So the −0.026 is *not* attributable to memory pressure; what it does establish is that the
   over-committed box did not push the share **up**.
4. **The baseline's box state was sampled after its capture, not during it** (7090 MB / 3245 MB at
   21:52–21:55). It is the same instrument, at a different moment. Its own reading (hook_share 0.264,
   idle/gap 0.90) is what carries the comparison, and it remains the quotable one.

## 4. Verdict

- **The bias argument is retired as an argument.** At this box's resolution, an over-committed run
  does not move the hook's share toward "host work"; the measured move is −0.026 in the other
  direction, and the verdict (HOST WORK, ratio 0.84 vs the 0.80 threshold) is unchanged.
- **It does not license lowering the floor for production readings**: the 0.84 reading is *near* the
  0.80 threshold, one regime away from flipping the verdict, and the record above shows why a
  below-floor run is not comparable to a clean one (content drifts with it).
- **The hypothesis is now falsifiable and failed once, not unfalsifiable.** To test it properly the
  box state has to be *controlled* (deliberate ballast with the same request stream), because a
  natural run collapses into a single regime within the first readings — that is the design change
  this measurement asks for, not a bigger N.

## 5. Files

- `Backup/phase_decomp/gap_split_swap.py` — the run, the per-reading stamp, `--reanalyze` (the metric
  can be revised without spending another window).
- `/tmp/gap_split_swap/{samples,readings,report}.json`, `capture.log`, `accept.json`;
  server log `Backup/cgc_logs/llama_server_20260920_220542.log`;
  baseline `Backup/cgc_logs/llama_server_20260920_214904.log`.
