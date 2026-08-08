# Debug Session: completion-max-new-tokens

- Status: OPEN
- Goal: 查明為何 `/health` 與 `/v1/chat/completions` 在 tokenizer 端保留正確的 `max_new_tokens`，但 scheduler 入隊時變成 `0`

## Findings

1. `init_req_max_new_tokens()` 確實曾因 `max_total_num_tokens=256` 將 chat 請求的 `max_new_tokens=8` clip 成 `0`
2. host1 重啟腳本硬編碼 `CGC_SGLANG_MAX_TOTAL_TOKENS=256` 是實際配置來源
3. 將 host1 運行配置提升到 `CGC_SGLANG_MAX_TOTAL_TOKENS=768` 後，scheduler 已保留 `after_max_new_tokens=8`
4. 最新邊界變成：請求可正常入隊，但沒有完成單次 completion，最終在 client timeout 後被 abort

## Current Hypotheses

1. `get_new_batch_prefill()` admission 邏輯在 waiting queue 非空時仍返回 `None`
2. scheduler 已選出 batch，但 `run_batch()` 未被調用
3. `run_batch()` 已執行，但 forward 後沒有正常產出結果給 detokenizer / tokenizer
4. 仍有其他長尾請求卡在 running batch，導致新請求只入隊不被消費

## Evidence Plan

1. 在 `_get_new_batch_prefill_raw()` 記錄 early-return / can-run selection
2. 在 `run_batch()` 前後記錄 batch 是否真正啟動
3. 對照最新 health/chat RID 驗證 admission -> batch -> forward 鏈路
4. 若 batch 已啟動，再下鑽 `process_batch_result()` 與 detokenizer 交付
