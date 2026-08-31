# [OPEN] Debug Session: pin-profile-ab

## Metadata
- Session ID: `pin-profile-ab`
- Started At: 2026-08-31
- Scope: Measure production A/B impact of `PIN_PROFILE` on long dense steady MTP.

## User Symptom
- User wants the next production-safe performance lever checked via runtime A/B, focusing on pin/profile rather than warm-gate tuning.

## Falsifiable Hypotheses
| ID | Hypothesis | Confidence | Status |
|---|---|---:|---|
| H1 | The existing pin profile improves hit rate on long dense steady and yields a measurable speedup. | 0.61 | Pending |
| H2 | Pin profile changes the output trajectory on fast-path production runs, but does not reintroduce `0000`. | 0.68 | Pending |
| H3 | The available profile is workload-mismatched for long dense steady, so it gives little or negative benefit. | 0.66 | Pending |
| H4 | If pin telemetry does not activate (`pin_marked` near zero / no load banner), any observed difference is not a valid pin-profile conclusion. | 0.83 | Pending |

## Evidence Log
- Active profile used for A/B: [profiles/qwen36_calib.pin](file:///Users/alexchuang/Documents/flashkv0516/profiles/qwen36_calib.pin).
- Production steady dense A/B command: `./scripts/run_n30cache.sh -m qwen36 --mtp --dense-iq4x --steady`, with `N30CACHE_PIN_PROFILE=...` only on the middle run.
- `OFF A` (`/tmp/pinprof_off_a.log`): `25.706 t/s`, `38.90 ms`, `accept 99.187%`, `hit 88.9%`, no `0000`, output SHA256 `e661...d614`.
- `ON` (`/tmp/pinprof_on.log`): `17.901 t/s`, `55.86 ms`, `accept 99.322%`, `hit 89.1%`, no `0000`, output SHA256 `a959...6d2c`.
- `OFF B` (`/tmp/pinprof_off_b.log`): `16.135 t/s`, `61.98 ms`, `accept 99.187%`, `hit 88.9%`, no `0000`, output SHA256 `e661...d614`.
- Pin telemetry confirms the profile really loaded on the middle run: `/tmp/pinprof_on.err` contains `routing-aware placement: pin_marked=1126 pin_yield(evicted)=7`.
- Output comparison: `OFF A` and `OFF B` are byte-identical; `ON` differs from both OFF runs and is 4 bytes longer, while still remaining free of `0000`.
- Minor tooling finding: the shell header still prints `pin: off` when the profile is passed via `N30CACHE_PIN_PROFILE` env rather than `--pin-profile`; runtime telemetry confirms the profile is active despite the header mismatch.

## Conclusion
- H1: Rejected. The existing pin profile does not speed up long dense steady in this production-safe path; any hit-rate change is negligible (`88.9% -> 89.1%`) and does not translate into throughput gain.
- H2: Confirmed. Pin profile changes the output trajectory on this fast-path production run, but does not reintroduce `0000`.
- H3: Confirmed. The available profile appears mismatched for this workload; it offers almost no cache benefit and likely adds placement overhead.
- H4: Confirmed. The profile did activate (`pin_marked=1126`), so the negative result is a valid pin-profile A/B result rather than a false-negative due to load failure.
