# Debug Session: local-full-decode [FIXED]

## Symptom
- `Gemma4 E4B Q2_K` 在 `local_full` 路徑下，`streaming off` 與 `streaming on_bypass` 都會於實際 decode 階段返回 `llama_decode returned -3`。
- 控制面已翻正：兩邊都能穩定路由到 `local_full`，`mode_switch_reason=local_llama_full_resident_loaded`。
- 目前 benchmark 無法量得有效 `cold TTFT / warm TTFT / decode TPS`，因為 execution 在首個 decode 階段即失敗。

## Session Goal
- 先把 `local_full` 恢復到可量狀態。
- 再重新比較 `off` 與 `on_bypass` 的 TTFT / TPS 是否收斂。

## Falsifiable Hypotheses
1. `n_gpu_layers=-1` 對當前 `E4B Q2_K + Metal` 僅能成功 load，實際 decode 會觸發 backend/OOM 類錯誤。
2. 問題來自 local llama runtime 參數組合（如 `n_batch` / `n_ubatch` / `flash_attn`），而非 route 或 planner。
3. 問題來自 prompt/template/tokenizer 生成的實際輸入形狀，導致 decode 才走到失敗分支。
4. `off` 與 `on_bypass` 共用同一條 local decode 熱路徑，所以兩者同時 `-3`，證明 blocker 位於 planner 之外。
5. 將 `n_gpu_layers` 從 `-1` 降到部分 offload 後，local decode 可恢復成功並重新量測。

## Hypotheses & Verification
| ID | Hypothesis | Status | Evidence |
|----|------------|--------|----------|
| A | `n_gpu_layers=-1` 本身不可用 | Rejected | 在 `256/128` 下仍为 `-1`，已可正常生成。 |
| B | 問題來自 local llama runtime 參數組合 | Confirmed | `512/256` 會 `llama_decode=-3`，`256/128` 恢復正常。 |
| C | prompt/template 導致 decode 失敗 | Rejected | 最小 prompt 約 29 tokens，仍可穩定復現/修復。 |
| D | blocker 位於 planner 之外 | Confirmed | `off` / `on_bypass` 都共享 local_full decode 路徑。 |
| E | 降低 GPU resident 才能恢復 | Rejected | 不需降 `n_gpu_layers`，僅收斂 batch profile 即恢復。 |

## Current Evidence
- pre-fix:
  - `streaming off` raw stream: `local llama.cpp error: llama_decode returned -3`
  - `streaming on_bypass` raw stream: `local llama.cpp error: llama_decode returned -3`
  - 兩邊 benchmark 均為 `routing_status=PASS`，但 `completed_requests=0`
- instrumentation evidence:
  - `A`: local llama load success，`n_gpu_layers=-1`
  - `B`: 最小 prompt 約 `29 tokens`
  - `C`: 在 `512/256` 下沒有首 token；在 `256/128` 下可見首 token
  - `D`: `512/256` 下直接 `llama_decode returned -3`
- post-fix:
  - stability guard 將 `E4B + n_gpu_layers=-1` 的 runtime batch 自動收斂為 `256/128`
  - `streaming off` baseline: `cold TTFT 69.6ms`, `decode TPS 17.5`, `routing_status=PASS`, `production_readiness=GO`
  - `streaming on_bypass` baseline: `cold TTFT 64.3ms`, `decode TPS 18.5`, `routing_status=PASS`, `production_readiness=GO`

## Verification Conclusion
- Root cause:
  - `E4B Q2_K` 的 local_full failure 不是 route 問題，也不是 `n_gpu_layers=-1` 本身不可用，
    而是 `n_batch=512 / n_ubatch=256` 這個 full-GPU runtime profile 在 decode 階段不穩定。
- Minimal fix:
  - 在 `edge_first_proxy.py` 的 `_local_llama_runtime_options()` 對 `E4B + n_gpu_layers=-1` 啟用 stability guard，
    將 oversized batch profile 自動收斂到 `256/128`。
- Pre-fix vs Post-fix:
  - pre-fix: `local_full` 首個 decode 即 `llama_decode=-3`
  - post-fix: `local_full` 可穩定完成 8/8 rounds，A/B 兩邊 route 與 gate 語義一致，且 TTFT / decode 指標接近

## Next Step
- 先以目前 `E4B + local_full` 穩定設定作為後續 decode TPS 調優基線。
- 之後再把同樣的驗證口徑擴大到更多 fit-in-memory 模型。
