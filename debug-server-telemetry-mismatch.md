# Debug Session: server-telemetry-mismatch
- **Status**: [OPEN]
- **Issue**: TurboFieldfareServer source already contains `prefill_ms`, `decode_ms`, and routed expert cache diagnostics/logging, but the running binary does not emit them, blocking steady-state validation for `cold_start_guard` and per-layer adaptive budget behavior.
- **Debug Server**: not started
- **Log File**: n/a

## Reproduction Steps
1. Build `TurboFieldfareServer` from `/Users/alexchuang/Documents/turbo-fieldfare`.
2. Start the built server with the staged Gemma4 model.
3. Send a chat completion request and inspect stderr log plus HTTP response body.
4. Observe that `prefill_ms`, `decode_ms`, `expert-cache-summary`, and `diagnostics.routed_expert_cache` are missing.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | `.build/release/TurboFieldfareServer` is a stale artifact built before telemetry source changes | High | Low | Pending |
| B | The process was started from a different binary than the inspected source tree | High | Low | Pending |
| C | SwiftPM reused an old `TurboFieldfareServerCore` build and did not relink the executable with new telemetry code | Medium | Medium | Pending |
| D | The binary contains the code, but runtime uses a non-telemetry path that leaves diagnostics empty | Medium | Medium | Pending |

## Log Evidence
- `fresh scratch build` 的 release binary 已包含 `expert-cache-summary` / `decode_protected_cap=` / `cold_start_guard=` 字串，證明 telemetry source 已正確編入新 artifact。
- 舊 `.build/release/TurboFieldfareServer` 的時間戳早於 `ServerLog.swift` / `HTTPServer.swift` 改動，符合 stale artifact 假設。
- fresh rebuild server 小請求已同時回傳 `diagnostics.routed_expert_cache`，並在 stderr 輸出 `prefill_ms=... decode_ms=... expert-cache-summary`。
- 重新跑 long steady-state 前發現 staging `gemma4.gturbo` 僅剩 `manifest.json` / `tokenizer` / `layout.json`，缺少 `model_weights.bin` 與全部 `packed_experts/layer_*.bin`；已從本地 HF 模型重新 repack 並 verify 成功。

## Verification Conclusion
- **A Confirmed**: 舊 `.build/release/TurboFieldfareServer` 是 stale artifact。
- **B Rejected**: 問題不是讀錯 source tree；fresh rebuild 已證明新 source 可正常生效。
- **C Confirmed**: 透過 isolated scratch path clean rebuild 後，`TurboFieldfareServerCore` 與 executable 已重新正確鏈結。
- **D Rejected**: runtime 並非走到無 telemetry 的分支；真正原因是 stale binary 加上後續 `.gturbo` 資產缺失。
- 後續正式 benchmark 已完成，報表輸出：
  - `var/colibri_metrics/reports/turbofieldfare_multiturn_long_steady_20260804_185752_cold_start_guard.json`
  - `var/colibri_metrics/reports/turbofieldfare_multiturn_long_steady_20260804_185752_cold_start_guard.md`
- 後續針對 `turn1` 的 `15/21` budget 呼吸點做過一輪 multiplier 型 `cold_start_guard` release tightening 實驗，但 runtime `turn1-only s=5` 幾乎無變化；進一步檢查顯示單輪 `turn1` 完成後這些層的 `total_loads` 已達 `1558-1600+`，說明目前 guard release 受單輪內部 expert/load volume 主導，單純提高 multiplier 不是有效控制桿。下一刀應改成 request/decode-step 級別的 warmup counter，而不是繼續堆 per-expert volume 門檻。
