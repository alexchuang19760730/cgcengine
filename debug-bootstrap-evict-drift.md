# Debug Session: bootstrap-evict-drift
- **Status**: [OPEN]
- **Issue**: `warm_wcache_confbootstrap_v2_max4` in `samples=2` shows `decode_evict_s ~5.6s`, but in `samples=5` drifts to `~12.1s`. Need to determine whether drift comes from layer selection instability, candidate expert instability, prompt-local payoff variance, or downstream slot1 churn after bootstrap.
- **Debug Server**: collector-restart pending
- **Log File**: .dbg/trae-debug-log-bootstrap-evict-drift.ndjson

## Reproduction Steps
1. Start warm server with current best config (`prefetch=0`, `budget=2`, `slot1_adopt_gap=1`, `anchor_floor=12`, `slot1_floor=12`, `max_layers=4`).
2. Run `scripts/benchmark/gemma4_decode_ttft_probe.py` with `--samples 5`.
3. Compare per-sample bootstrap layer/candidate logs against per-sample `decode_expert_load_evict_s`, `wcache_opportunistic_*`, and `wcache_anchor_hit`.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | Selected bootstrap layers drift across samples and drive decode evict drift | High | Low | Pending |
| B | Selected layers are stable, but anchor/slot1 expert candidates drift across samples | High | Low | Pending |
| C | Layer/candidate selection is stable, but prompt-local path causes slot1 payoff variance | Medium | Medium | Pending |
| D | Bootstrap selection is fine, but post-bootstrap `w cache` churn amplifies into decode evict drift | High | Low | Pending |
| E | Current layer value still ranks the wrong layers for reuse stability | Medium | Medium | Pending |

## Log Evidence
- Debug log: [`.dbg/trae-debug-log-bootstrap-evict-drift.ndjson`](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson)
- Short decode samples (`traceId=2,3`) selected different layer sets:
  - `2`: layers `13,14,16,17` with low decode evict ([L16-L20](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L16-L20), [L36](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L36))
  - `3`: layers `12,13,17,24` with similarly low decode evict ([L50-L62](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L50-L62), [L72](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L72))
- Longer decode samples (`traceId=4,5,6`) also selected different layer sets:
  - `4`: `8,14,24,29` ([L82-L103](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L82-L103))
  - `5`: `4,13,24,29` ([L114-L139](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L114-L139))
  - `6`: `4,13,16,29` ([L150-L175](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L150-L175))
- In all long samples, `slot1_adopt` and `slot1_evict` remained almost equal:
  - `trace 4`: `72 / 72` ([L108](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L108))
  - `trace 5`: `70 / 70` ([L144](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L144))
  - `trace 6`: `69 / 69` ([L180](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L180))
- This indicates slot1 churn remains structurally high even when selected layers differ only moderately.

## Verification Conclusion
- **A partially confirmed**: selected bootstrap layers do drift between samples, especially once prompt/decode path changes from short to longer runs.
- **B partially confirmed**: candidate experts also drift with the selected layers (e.g. layer 29 anchor `70` in trace 5 vs `103` in trace 6).
- **C likely confirmed**: slot1 payoff depends on prompt-local path; long samples with different selected sets still converge to high churn and worse decode evict.
- **D strongly confirmed**: downstream slot1 churn is the most stable signal. `slot1_adopt ~= slot1_evict` across long samples correlates with elevated decode evict.
- **E still open**: current layer value likely ranks layers by score, but not yet by reuse stability. Need a second round focused on “selected layer stability vs slot1 churn risk” before changing logic.

## Second-Layer Evidence
- Added request-scoped per-layer accounting in [`colibri/c/gemma4.c`](file:///Users/alexchuang/Documents/flashkv0516/colibri/c/gemma4.c) for:
  - `selected`
  - `layer_value`
  - `anchor/slot1 eid + score`
  - `anchor_hit`
  - `slot1_hit`
  - `anchor_adopt`
  - `slot1_adopt`
  - `slot1_evict`
  - `slot1_churn = slot1_evict / slot1_adopt`
- Rebuilt `gemma4-metal` and reproduced with an instrumented warm server on `:18082`.

### Request-Level Findings
- Longer requests still show request-level slot1 churn saturation:
  - trace `5`: `slot1_adopt=78`, `slot1_evict=78`, `decode_evict_s=0.092680` ([L231](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L231))
  - trace `6`: `slot1_adopt=99`, `slot1_evict=99`, `decode_evict_s=0.099824` ([L297](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L297))
  - trace `7`: `slot1_adopt=83`, `slot1_evict=83`, `decode_evict_s=0.102388` ([L363](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L363))

### High-Score but High-Churn Layers
- `layer 13`:
  - selected in all 6 reproduced requests
  - mean selected `layer_value ~= 20.769`
  - cumulative selected-request `slot1 = 12/12`, i.e. `slot1_churn = 1.0`
  - representative lines: [L47](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L47), [L113](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L113), [L179](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L179), [L245](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L245), [L311](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L311), [L377](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L377)
- `layer 16`:
  - selected in 5/6 requests
  - mean selected `layer_value ~= 20.484`
  - cumulative selected-request `slot1 = 13/13`, i.e. `slot1_churn = 1.0`
  - representative lines: [L50](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L50), [L116](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L116), [L182](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L182), [L248](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L248), [L314](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L314)
- `layer 24`:
  - selected in all 6 requests
  - mean selected `layer_value ~= 21.045`
  - when slot1 actually fires, cumulative selected-request `slot1 = 6/6`, i.e. `slot1_churn = 1.0`
  - two short traces show selected but `slot1_adopt=0`, meaning it is not universally bad, but once slot1 is used it still churns out immediately
  - representative lines: [L58](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L58), [L124](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L124), [L190](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L190), [L256](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L256), [L322](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L322), [L388](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.ndjson#L388)

### Interpretation
- The bad pattern is no longer “layer selection drifts, so maybe ranking is noisy”.
- The stronger result is:
  - some layers (`13`, `16`, and conditionally `24`) are repeatedly selected because they score high,
  - but their `slot1` path is not sticky at all once exercised,
  - so they are exactly the “high-score but high-churn” layers the user asked to identify.
- This shifts hypothesis `E` from open-ended ranking doubt toward a concrete next move:
  - the layer value needs an explicit churn-risk penalty, or
  - these layers need a layer-level bootstrap deny/discount path for slot1.

## Collector Restart Confirm
- **Goal**: restore the debug collector for the existing `bootstrap-evict-drift` instrumentation and rerun the unchanged `samples=5` confirm so we can pin three signals in one pass:
  1. which layers get non-zero `hist_adjust`
  2. whether selected sets now bias toward replacement layers rather than only avoiding bad layers
  3. whether those replacement layers actually reduce `slot1_evict / slot1_adopt` and `decode_evict_s`

### Runtime Hypotheses
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | The collector is simply down, so runtime events are emitted but never persisted | High | Low | Pending |
| B | The collector restarts, but `.dbg/bootstrap-evict-drift.env` still points to a stale port/session | Medium | Low | Pending |
| C | Once logs are restored, `hist_adjust` remains zero across `samples=5`, meaning the quality gate threshold still does not trigger | Medium | Low | Pending |
| D | Once logs are restored, selected sets do shift toward replacement layers, but churn metrics remain bad, meaning the replacement layers are still not sticky enough | High | Low | Pending |

### Collector Restart Findings
- Restarted Debug Server for the existing `bootstrap-evict-drift` session on `:7777` and reran unchanged `samples=5`.
- Fresh rerun log: [`.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson`](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson)
- Fresh rerun report: [`replacement-quality-confirm-s5-rerun.json`](file:///Users/alexchuang/Documents/flashkv0516/var/replacement_quality_confirm/reports/replacement-quality-confirm-s5-rerun.json)

#### What the rerun proved
1. **`hist_adjust` did trigger, but only as negative penalties**
   - non-zero penalties appeared repeatedly on layers `15` and `17`, always `hist_adjust=-3.0`
   - representative lines: [L52](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson#L52), [L54](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson#L54), [L382](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson#L382), [L384](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson#L384), [L646](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson#L646)
2. **Selected sets did shift toward replacement layers**
   - early traces: `4,8,12,14` and then `4,8,12,29`
   - later traces converge to `4,12,26,29`, with only the final trace briefly reintroducing `17`
   - representative bootstrap selections: [L6](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson#L6), [L14](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson#L14), [L280](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson#L280), [L292](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson#L292), [L600](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson#L600), [L625](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson#L625)
3. **Decode churn still did not come down structurally**
   - request-level `slot1_adopt == slot1_evict` still holds across all rerun traces:
     - trace `16`: `75 / 75`, `decode_evict_s=0.096208` ([L366](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson#L366))
     - trace `17`: `114 / 114`, `decode_evict_s=0.110753` ([L432](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson#L432))
     - trace `18`: `75 / 75`, `decode_evict_s=0.098009` ([L498](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson#L498))
     - trace `19`: `36 / 36`, `decode_evict_s=0.074532` ([L564](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson#L564))
     - trace `20`: `71 / 71`, `decode_evict_s=0.110666` ([L630](file:///Users/alexchuang/Documents/flashkv0516/.dbg/trae-debug-log-bootstrap-evict-drift.qualityconfirm-s5-rerun.ndjson#L630))

#### Updated Interpretation
- **A confirmed**: the previous missing evidence was a dead collector, not missing runtime events.
- **B rejected**: after restart, `.env` still correctly pointed to `http://127.0.0.1:7777/event`.
- **C rejected in its strong form**: `hist_adjust` is not globally zero; it really does fire by `samples=5`, but only as negative penalties on churn-prone replacement layers (`15`, `17`).
- **D confirmed**: selection already biases toward new replacement layers (`4/12/26/29`), but decode churn still saturates. The replacement layers are different, yet still not sticky enough.
