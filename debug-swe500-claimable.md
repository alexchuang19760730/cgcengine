# [OPEN] Debug Session: swe500-claimable

## Symptoms
- `swe_verified_500` 目前在 `validate --all` / `validate --capability swe_verified_500` 中顯示：
  - `formal_chain_status=PASS`
  - `official_eval_status=SUBMITTED`
  - `claimable=false`
  - `swe_verified_passed_tasks=0`
- 已經把 `agent_execution.status` 從 `MISSING` 前移到 `SUBMITTED`，但仍未形成可 claimable 的正式結果。

## Expected
- 找到 `submitted_tasks/passed_tasks` 的正式來源，或推進一次真 `submission / score ingest`。
- 讓 `claimable=false` 前移到下一個真實節點，例如：
  - `submitted_tasks > 0`
  - `passed_tasks > 0`
  - `official_eval_status` 從 `SUBMITTED` 前移到更具體狀態

## Hypotheses
1. `remote_swebench_score_summary.json` 只提供 trajectory/score 摘要，並不包含正式 `submitted_tasks/passed_tasks`，真正的官方評測結果在另一份 artifact，當前 alias 沒接到。
2. 真正的 score/submission artifact 已存在於 `model_swe_verified_session.json` 指向的相鄰文件中，但目前 `agent_execution` / `validate` 沒有消費。
3. 真 submission 尚未完成，當前 remote run 只生成了 trajectory，因此 `submitted_tasks=0` 是現場真值，不是聚合缺口。
4. official eval 其實已有更細狀態，但被 `remote_swebench_score_summary.status=PASS` 這種通用字段遮蔽，缺少面向 `claimable` 的專用翻譯規則。

## Plan
1. 盤點 `model_swe_verified_session.json` 鄰近 artifact，確認是否存在正式 `submission/score` 檔。
2. 若存在，先在 alias 聚合入口加 instrumentation，證明目前到底讀了哪個 source、漏了哪個 source。
3. 若不存在，轉向 runtime 路徑，追 `submission / score ingest` 為何沒落盤。

## Evidence
- `model_swe_verified_session.json` 當前 `accepted_contracts.swebench_score_recovery.status=FAIL`、`reason=score_summary_collection_failed`。
- `benchmark_summary.status=FAIL` 且 `benchmark_summary.error=Authentication failed.`，說明 refresh-session 目前卡在更上游的 SSH 認證，而不是先卡在 score parser。
- `execution.remote_hosts` 中 `host1 (39.106.118.206)` 的密碼已被寫成 `Gen@song123`，與使用者提供的正確密碼 `Gen@song@2026622` 不一致。
- `app/cli/cgc.py` 的 `DEFAULT_SWEBENCH_REMOTE_HOSTS` 也把 `host1` 預設密碼寫成 `Gen@song123`；`_resolve_swebench_remote_hosts()` 目前會優先使用 default/env，而 refresh-session 會把解析結果直接覆寫回 session 的 `execution.remote_hosts`。

## Hypothesis Checkpoint
- H1 `remote_swebench_score_summary.json` 只是摘要、真正官方 score 在別處：暫未證實，先保留。
- H2 相鄰 artifact 已存在更正式 score 檔但 alias 未消費：目前未找到直接證據，先保留。
- H3 真 submission 尚未完成，因此 `submitted_tasks=0` 是現場真值：暫時無法成立，因為 refresh-session 在 SSH 認證階段就已失敗。
- H4 official eval 已有更細狀態但被通用欄位遮蔽：暫未證實，先保留。
- H5 host 憑證來源被 default/session 污染，導致 head host 無法連線而使 score recovery 假性歸零：目前最強假設，已有 pre-fix runtime evidence 支持。

## Next Step
1. 最小修正 `refresh-session` 的 host 憑證來源，讓 host1 優先採用 Host 真源而不是 session 緩存錯值。
2. 重跑一次 `model swe-verified --refresh-session`。
3. 比對 `pre-fix` 與 `post-fix` 的 `benchmark_summary`、`submitted_count`、`score_status` 與 debug log，確認第一錯是否前移到真正的 score ingest / submission 節點。

## Post-fix Evidence
- 已將 `app/cli/cgc.py` 中 `DEFAULT_SWEBENCH_REMOTE_HOSTS` 的 `host1 (39.106.118.206)` 預設密碼修正為 `Gen@song@2026622`，`host2` 維持 `Gen@song123`。
- 重新執行 `python3 app/cli/cgc.py model swe-verified --refresh-session ...` 後，命令成功退出，session 頂層狀態從先前 `FAIL` 前移為 `PENDING`。
- refresh 後的 `execution.remote_hosts` 已回寫為：
  - `host1.password = Gen@song@2026622`
  - `host2.password = Gen@song123`
- `accepted_contracts.swebench_score_recovery` 已從：
  - `status=FAIL`
  - `reason=score_summary_collection_failed`
  前移為：
  - `status=PENDING`
  - `state=running`
  - `trajectory_count=1`
  - `submitted_count=0`
  - `score_status=pending`
  - `reason=benchmark_not_completed`
- `accepted_contracts.m76_runtime_evidence.status` 已從 `FAIL` 前移為 `PASS`，表示先前的 SSH 認證失敗已消失。
- 最新 debug log 顯示：
  - 已成功收集 `remote_swebench_score_summary`，`status=PASS`、`state=running`、`trajectory_count=1`
  - 不再出現 `AuthenticationException`
  - 新的第一個真實 blocker 已前移到 benchmark/runtime 本身尚未完成，而非憑證錯誤

## Current Conclusion
- H3「`submitted_tasks=0` 是現場真值」目前仍不能完全確認，但至少現在看到的是有效 runtime 現場，而不是被 SSH 失敗污染後的假性歸零。
- H5「host 憑證來源被污染」已證實且已修復。
- 下一刀不再是 credential source；應直接追為何 `trajectory_count=1` 但 `submitted_count=0`，以及遠端 `500/500` log 對應的正式 score/result artifact 為何未被產出或未被回收。
