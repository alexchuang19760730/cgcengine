# Debug Session: scheduler-no-run

- Status: OPEN
- Goal: 查明 host1 上 scheduler 為何「請求能入隊但不跑」，並打通單次 completion

## Scope

- 只看 `dequeue / batch selection / forward execution` 鏈
- 不再回頭重查 sampling 參數是否被覆寫
- 優先用最小插樁與現場 runtime artifact 定位

## Falsifiable Hypotheses

1. `--disable-hybrid-swa-memory` 之後，DSV4 的 KV cache 初始化路徑仍無條件引用 `swa_max_total_num_tokens`，導致 scheduler actor 在進 event loop 前崩潰
2. 即使修掉 startup crash，prefill admission 仍可能在 non-hybrid path 被其他 token budget 拒絕，導致 `prefill_selected_can_run_list` 依然拿不到 batch
3. `running_batch.batch_is_full` 可能在空 batch 狀態下被過早設為 `True`，使 dequeue 雖持續輪詢但永遠跳過 batch 建立
4. non-hybrid SWA 關閉後，forward worker / pool config 的尺寸映射仍可能不一致，造成 batch 選出後在 `run_batch()` 前後異常

## Evidence Checklist

- 確認 host1 啟動命令已帶 `--disable-hybrid-swa-memory`
- 確認 scheduler actor 是否完成初始化
- 確認第一個 health/chat request 是否出現 `prefill_selected_can_run_list`
- 確認是否出現 `run_batch_begin` / `run_batch_end`
- 確認是否能拿到單次 completion 成功返回

## Current Narrowing

- 既有 runtime 證據顯示 tokenizer 已打到 `dispatch_single_to_scheduler` 與 `send_to_scheduler_begin/done`
- 新鮮 health/chat RID 仍未在 scheduler trace 中看到 `recv_from_tokenizer_raw`
- `http_worker_ipc=None` 更像是 response routing 線索，未必能解釋 ingress 缺失
- 本輪先只驗證單 tokenizer 與 scheduler rank0 是否綁到同一條 `scheduler_input_ipc_name`
