# The certified window fixed k=2, not k=3 — so the k2/k3 effect is still unseparated

**Date** 2026-09-23 · carrier `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X`, MTP on, pool 8 GiB,
`prod25`, temp 0 greedy, completion door · binary `b70ff34a8bc69441` · 13 launches on port 9933/9934,
no other llama process, zero residual processes afterwards.

## 0. One line

Enforcing **Nominal-at-launch** on every arm cut the k=2 arm's same-config spread from 18.5% to **4.8%**
and removed the 5.84 t/s regime entirely — but the k=3 arm still swings **1.43×** on bit-identical work,
so `k2` vs `k3` remains **NOT SEPARATED** (5 pairs, 4 positive, mean +1.71 t/s, t=2.09, weakest −0.05).
The earlier "not separated" was **not** merely thermal drift. The gate is necessary and not sufficient.

## 1. Protocol

| run | arms | shape |
|---|---|---|
| A | 8 | **ABBA rotation**, 4 rounds: k=2 first in rounds 1/3, last in 2/4 |
| B | 2 | gate verification: k=2 then k=3 back-to-back |

Every arm waits for the OS to report **Nominal** before launching (`--require-thermal-0`), and the wait
is recorded per arm. Each arm = one warmup request (not counted) + 3 counted requests (`n_predict 96`).
`k=2` is set with `CGC_SERVER_MTP_N_MAX=2`, `k=3` is the profile default; nothing else differs.

**The work is identical in every arm** — the engine's own counters, not a tool field:

| arm | draft counted in every launch | engine `mean len` in every launch |
|---|---|---|
| k=2 | 73.16% (**169/231**) | **2.64** |
| k=3 | 58.25% (**180/309**) | **2.97** |

So every t/s difference below is the same tokens produced by the same work, in a different amount of
time. Artifacts: `Backup/k_abba_2026-09-23/{abba,gated}/` (driver logs + the 10 arm JSONs).

## 2. The effect, and why it still is not one

| round | k=2 | k=3 | k2 − k3 | wait (k2 / k3) |
|---|---:|---:|---:|---|
| R1 | 12.23 | 8.59 | +3.64 | 0 s / 50 s |
| R2 | 11.21 | 10.72 | +0.49 | 151 s / 141 s |
| R3 | 12.83 | 9.09 | +3.74 | 61 s / 41 s |
| R4 | 12.20 | 11.48 | +0.72 | 60 s / 181 s |
| **B** | **12.20** | **12.25** | **−0.05** | 0 s / 60 s |

```
k=2: n=5 mean=12.13 sd=0.58 cv=4.8%  11.21–12.83 (1.14x)  SE(mean)=0.26 (2.1%)
k=3: n=5 mean=10.43 sd=1.56 cv=14.9%  8.59–12.25 (1.43x)  SE(mean)=0.70 (6.7%)
paired k2−k3: mean=+1.71 median=+0.72 sd=1.83 t=2.09 (df=4) positives=4/5 weakest=−0.05
```

`scripts/check/k_abba_stats.py --driver-log …/abba/driver.log --driver-log …/gated/driver.log`
reproduces that block from the artifacts. The fifth pair is the one that decides the sentence: the
gated run measured k3 **at the top of its whole range** and k2 at its mean, giving a tie. A mean of
+1.71 t/s may not be quoted as an effect while a round sits at −0.05.

## 3. What the certified window did buy

- **The low regime is gone.** Six same-config arms before the gate spanned 5.84–10.46 t/s (1.79×,
  cv 18.5%, `docs/INSTRUMENT_RESOLUTION_2026-09-23.md`). With the gate, the minimum over 13 arms is
  **8.59** and the k=2 arm is reproducible to **±2.1%** of its mean.
- **k=2 is now precise enough to test small candidates** (2σ on 5 arms ≈ ±0.5 t/s = 4%); **k=3 is not**
  (±1.4 t/s = 13%). Any future candidate that only touches the wide-verify arm inherits that floor.
- The level signal stays a **regime label, not a resolution fix**: it is read at launch, has hysteresis,
  and `MODERATE` covered both 6.26 and 9.26 t/s in the cross-axis pairs.

## 4. What it did not fix: k=3 swings 1.43× with identical work

Two candidate explanations were tested with data already on disk, and both fail:

1. **The gate wait.** On the first four k3 arms the reading rose with it (41 s→9.09, 50 s→8.59,
   141 s→10.72, 181 s→11.48; ρ=+0.80). The fifth kills it: **60.1 s → 12.25**, the fastest k3 reading,
   while 41 s and 50 s gave the two slowest. (`k=2` never showed the relation: ρ=−0.40.)
2. **The pool-event counts.** All four rotated k3 arms have **byte-identical event patterns**
   (175 `CGC-SYNCFILL` / 33 `CGC-SEG` / 80 `CGC-HOOK`) and still span 8.59–11.48. So the swing is in
   **time per event** — clocks, memory bandwidth, or host contention — not in what the pool path did.

That narrows the residual to the rate, not the script. Nothing in this round identifies *which* rate.

## 5. A side finding that does explain the k=2 spread — and is its own hazard

The pool path does **not** partition reproducibly across launches of the same configuration:

| pattern (`SYNCFILL`/`SEG`) | arms | readings |
|---|---|---|
| 175 / **33** | 4 of 4 k=3 arms + the slowest k=2 arm | k3: 8.59, 10.72, 9.09, 11.48 |
| 184 / **36** | the three fastest k=2 arms | 12.23, 12.83, 12.20 (mean 12.42) |

A three-segment difference (33 → 36) with identical requests and bit-identical outputs, and inside
k=2 the 36-segment arms read 12.42 on average against the 33-segment arm's 11.21 (**+11%**, same k,
so the comparison is not a k effect). That is the "pool geometry is an input, not just a size" family:
two launches of one configuration can take a **different partition**. Caveat, recorded rather than
filled in: the fourth k=2 arm (11.21, the slow one) resolved its counters through the
`llama_server_latest.log` **symlink**, so its pattern is unrecoverable — the evidence is 3/3 against
1 unknown, not 4/4.

## 6. 25 t/s at HEAD, unchanged

`mean len` and decode are both the engine's own numbers, same arms:

| | mean len | decode | ms/token | ms/step |
|---|---:|---:|---:|---:|
| k=2 | 2.64 | 12.13 | 82.4 | **218** |
| k=3 | 2.97 | 10.43 | 95.9 | **285** |

25 t/s at `mean len` 2.64 needs **S ≤ 106 ms/step** ⇒ **2.06×** from today's k=2 and 2.69× from k=3.
Note the k=3 arm buys +12.5% tokens per step for **+31%** step time: at this accept rate the wider
verify is a loss on the mean, and this round puts the direction (not the significance) on k=2 for the
first time.

## 7. Tooling (what makes this repeatable)

| change | why |
|---|---|
| `mtp_accept_ab.py --require-thermal-0 TIMEOUT_S` | per-arm block until Nominal, giving up (and measuring anyway) at the timeout; the arm product records `thermal_gate {waited_s, reached_nominal}` so the treatment is visible |
| `mtp_accept_ab.py server_log_at()` | the arm product now names the **server** log its counters went to. Not `logpath` — that capture has zero CGC lines; and the resolver skips `llama_server_latest.log`, whose symlink is how one arm's counters silently became another run's in this very round |
| `mtp_accept_ab.py --selftest` | 7/7: log resolution (incl. "the symlink is never the answer") and the gate's give-up path |
| `scripts/check/k_abba_stats.py` (new) | rotation-aware pairing, per-arm spread next to the paired effect, and the weakest round printed. `--selftest` 9/9 |

The sign defect it was written for is worth naming because it is this project's recurring class: a
"first minus second" subtraction across a **rotated** run printed **−0.49** for a round whose k2−k3 is
**+0.49**, halving an apparent effect and inventing a negative. The subtraction follows the arm.

## 8. Honest ledger

- **5 pairs is at the edge.** t=2.09 (df=4) ⇒ p≈0.10; by sign test 4/5 ⇒ p≈0.19. Neither certifies, and
  the pre-registered "weakest round must clear the resolution" rule says *below threshold*.
- **Regime**: this is temp 0 greedy with a short prompt. The 0.4-temp delivery family is not comparable;
  only the k2/k3 **comparison** holds inside this regime.
- One arm's pool counters are unknown (§5) — recorded as unknown, not inferred.
- The gate's wait is a *treatment* (it changes what the arm measured), which is why it is recorded per arm.
- `mean len` comes from the engine's own line; the tool's `mean_acc_len` is per-request accepts and is
  **not** tokens per step.
- 13 launches ≈ 15 minutes of box time. Nothing committed; `window_gate` ungated-launcher count unchanged.

## 9. Reproduce

```bash
# the gate, in one arm
python3 scripts/check/mtp_accept_ab.py --selftest
python3 scripts/check/mtp_accept_ab.py --arms nail --port 9934 --require-thermal-0 300 \
        --extra-env CGC_SERVER_PROFILE=prod25 --out /tmp/one_arm.json

# the statistics, from the archived artifacts
python3 scripts/check/k_abba_stats.py --selftest
python3 scripts/check/k_abba_stats.py \
    --driver-log Backup/k_abba_2026-09-23/abba/driver.log \
    --driver-log Backup/k_abba_2026-09-23/gated/driver.log \
    --arms-dir  Backup/k_abba_2026-09-23/abba
```

## 10. What this means for the next lever

The certified window converted "the box is unusable for small effects" into "the box is usable for
small effects **on the k=2 arm**", and it converted the k question from *unknown* into *unseparated at
5 pairs, direction on k=2*. The blocking item is no longer the box's thermal state but **k=3's own
1.43× swing at identical work** — which is a property of the wide-verify arm's *rate*, and is therefore
measurable inside a single process (step cost at ntok=1..4), where launch-to-launch drift cannot enter.
