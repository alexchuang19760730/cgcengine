# Debug Session: huihui-bypass-overhead [FIXED]

## Symptom
- `Huihui-MoE-0.8B-2E` 在 fit-in-memory 條件下：
  - `streaming off` 已可穩定跑在 `local_full`
  - `streaming on_bypass` 也已翻正到 `local_full`
- 但目前 `on_bypass` 相比 `off` 仍有可見性能差距：
  - `off`: `TTFT 117.5ms`, `decode TPS 131.7`
  - `on_bypass`: `TTFT 126.6ms`, `decode TPS 113.9`

## Session Goal
- 釐清 `Huihui on_bypass` 的額外成本是否穩定存在。
- 若 overhead 可重現，定位它落在 request lifecycle 的哪一段。

## Falsifiable Hypotheses
1. 目前 `Huihui off/on_bypass` 的差值主要是短樣本波動，重跑後會顯著收斂。
2. `on_bypass` 雖已是 `fit_memory_bypass`，但 request lifecycle 仍多做了 `begin_request / complete_request / snapshot` 類控制面工作。
3. `degrade_suggested=true` 雖被 runtime override 翻正為 `local_full`，但仍留下額外 bookkeeping 成本。
4. 額外成本不在 decode 內核，而是在 TTFT 前後或 request 收尾階段，被 bench 口徑算進整體 decode TPS。
5. `Huihui` 的 true-MoE metadata 讓 `on_bypass` 仍走到某些 dense baseline 不會碰到的控制面分支。

## Current Evidence
- `route-test` 已翻正：
  - `final_route_mode=local_full`
  - `mode_switch_reason=local_llama_full_resident_loaded`
  - `expert_reason=fit_memory_bypass`
- 目前差值：
  - `delta_ttft_ms_avg = +9.1`
  - `delta_decode_tps_avg = -17.8`
- rerun evidence:
  - `off initial`: `TTFT 117.5ms`, `decode TPS 131.7`
  - `off rerun`: `TTFT 121.1ms`, `decode TPS 113.1`
  - `on_bypass`: `TTFT 126.6ms`, `decode TPS 113.9`
  - rerun 後 `off` 與 `on_bypass` 的差值收斂為：
    - `delta_ttft_ms_avg = +5.5`
    - `delta_decode_tps_avg = +0.8`

## Verification Conclusion
- Root cause:
  - `Huihui on_bypass` 先前看見的 `-17.8 tok/s` 差距不是穩定 overhead，
    而是短樣本下的 performance variance。
- Supporting evidence:
  - 在同參數、同機器狀態下重跑 `off`，其 `decode TPS` 從 `131.7` 回落到 `113.1`，
    已與 `on_bypass` 的 `113.9` 幾乎重合。
  - `route-test` 同時證明 `on_bypass` 已被翻正到：
    - `final_route_mode=local_full`
    - `mode_switch_reason=local_llama_full_resident_loaded`
    - `expert_reason=fit_memory_bypass`
- Practical conclusion:
  - 目前 `Huihui` 的 fit-memory bypass 路徑已達到「控制面翻正、性能近似」；
    後續若要再追 overhead，應先用更長 rounds 或固定機器負載，再談更細的 lifecycle timing。

## Next Step
- 如需更高置信度，可把 `Huihui off/on_bypass` 都提升到更長 rounds 重新量測。
- 若之後仍觀察到穩定差距，再對 `begin_request / snapshot / complete_request` 做 timing instrumentation。
