[OPEN] Debug Session: dualnode-swe500-blocker

## Symptoms
- `FusionRoute + DeepSeek-V4-Flash + MiniCPM5 + SWE Verified 500` 在 gate/document 層曾被視為已收口，但 runtime evidence 顯示舊 session 並未形成正式 completed score。
- 已刷新 session 後，`model_swe_verified_session.json` 顯示頂層 `status=PENDING`。
- `swebench_score_recovery` 目前為 `status=PENDING`、`state=running`、`trajectory_count=1`、`submitted_count=0`。
- `Gate 2.1` 最新狀態為 `10/11 PASS`，唯一 blocker 為 `verified500_speedup_closure`。

## Expected
- dual-node `swe-verified 500` 應產生可驗收的 completed score。
- `submitted_count` 應大於 0，且存在可被 recovery/驗收消費的 score payload 或逐題結果。
- `Gate 2.1` 的 `verified500_speedup_closure` 應由真實 runtime evidence 驗證通過，而非文件宣稱。

## Hypotheses
1. `host1` head 端成功啟動 `run-batch`，但 gateway/backend completion path 卡住，導致只產生極少量 trajectory 且沒有 `submitted`。
2. launch env 或 `CGC_SWEBENCH_API_BASE` 仍存在舊配置/殘留，`SWE-agent` 沒有命中目前正確可用的推理入口。
3. dual-node runtime 基本存活，但在 worker/backend 路徑上存在 early exit/error，讓 500 題批次在完成前就被大量 `exit_error` 吃掉。
4. score/逐題產物其實存在，但 `score recovery` 的搜尋規則或路徑假設有誤，導致 recovery 沒有撿到真正結果。

## Plan
- 先讀現有 session / manifest / debug note，盤清目前已知 runtime 證據鏈。
- 再在不改業務邏輯的前提下補最小插樁到 `model swe-verified` 的 remote launch / score recovery 關鍵節點。
- 用 refresh 或重跑收集 pre-fix runtime 證據，判定上述假設。
- 等證據收斂後，再做最小修補並用 post-fix evidence 比較。

## Current Status
- Step 1-4: hypotheses established, no business logic modified in this debug session.

## Evidence Update
- `host1` / `host2` 直連 SSH 已用新憑證恢復，手動 Paramiko `echo ok` 均可通。
- 舊 session 的 `CGC_SWEBENCH_API_BASE` 取證確認為 `http://127.0.0.1:8000/v1`，且只留下 `trajectory_count=1 / submitted_count=0`，不足以形成 completed score。
- 第一輪 fix 誤將預設 API base 對齊到 `8001/v1`；runtime 證據顯示：
  - `8001/v1/chat/completions` 回 `501 Unsupported method ('POST')`
  - `1 題 probe` 會留下 exit_error 與 autosubmission，但不是正確 completion 入口
- 進一步遠端探針顯示：
  - `8000/v1/models` -> `404`
  - `30000/v1/models` -> `200`
  - `50053/v1/models` -> `200`
  - `30000/v1/chat/completions` / `50053/v1/chat/completions` 在短超時內未返回，屬於真正的 completion latency/runtime 問題，而非入口不存在
- 第二輪 fix 已把 `model swe-verified` 預設 API base 改到 `http://127.0.0.1:50053/v1`。
- `post-fix` 的 `1 題 probe` 已成功：
  - `remote_launch = PASS`
  - 不再出現 `501 Unsupported method ('POST')`
  - log 已進到 agent `STEP 1`，說明入口修正有效，問題前移到 LM query / completion latency
- 另一條獨立 blocker 仍存在：`_collect_remote_m76_evidence()` 在使用同一組憑證時仍拋出 `AuthenticationException('Authentication failed.')`，這與手動 Paramiko 直連可用形成不一致，需繼續釐清函式內的連線行為。
- `m76 evidence` 後續已確認為 per-host 密碼不一致：
  - `host1` 使用 `Gen@song@2026622`
  - `host2/worker` 使用 `Gen@song123`
  - `refresh-session` 也已修成優先重新 resolve host specs，不再被舊 session 憑證釘住。
- `host2` 的 `30000` backend 現場顯示本機 completion 可正常工作：
  - `GET /v1/models` -> `200`
  - `POST /v1/chat/completions` -> `200`
  - `e2e_latency` 約 `0.35s`
- `host1` 的 `50053` 由 `ray::ProxyActor` 監聽，`api_server.log` 中也可見大量 `POST /v1/chat/completions 200 OK`。
- 但直接在 `host1` 對 `50053/v1/chat/completions` 做 isolated replay 時：
  - `short prompt` 也在 `180s` timeout
  - `long prompt` 同樣在 `180s` timeout
  - 因此「只有長 prompt 才觸發 connection error」這個假設已被排除。
- `SWE-agent` session 內的 `recent_errors` 顯示：
  - `litellm.InternalServerError: ... Connection error`
  - `(slept for 765.60s)`
  - 對照 `SWE-agent/sweagent/agent/models.py` 可確認預設 `RetryConfig` 為 `retries=20, min_wait=10, max_wait=120`。
  - `litellm.completion()` 呼叫未見明確 request timeout，且 `completion_kwargs` 預設為空。
- 已在 remote `SWE-agent/sweagent/agent/models.py` 加入 local gateway guardrail：
  - `http://127.0.0.1` / `http://localhost` 的 `api_base` 若未顯式設定 timeout，預設補 `45s`
  - local gateway 路徑的 retry 預設收斂為 `retries=3, min_wait=1, max_wait=5`
  - 同時保留 env override：`CGC_SWE_LOCAL_LM_TIMEOUT_S`、`CGC_SWE_LOCAL_LM_RETRIES`、`CGC_SWE_LOCAL_LM_RETRY_MIN_WAIT`、`CGC_SWE_LOCAL_LM_RETRY_MAX_WAIT`
- `probe3-postfix` fresh rerun 結果：
  - 不再出現 `765.60s` 級別的 `Connection error`
  - remote log 顯示 `Retrying LM query: attempt 1 (slept for 1.00s) due to Timeout`
  - 最終失敗收斂為 `litellm.Timeout: APITimeoutError - Request timed out`
  - agent 已保存 trajectory，`trajectory_count` 從 `0` 推進到 `1`
  - 但 `submitted_count` 仍為 `0`

## Hypothesis Status
1. `host1` head 端 launch 不成功：已排除。`remote_launch_manifest` 顯示 fresh probe 可成功啟動 `run-batch`。
2. API base/入口仍指到錯服務：已證實且已部分修正。`8001` 為錯入口，`50053` 才是較一致的 gateway 入口。
3. dual-node runtime 完全未進入 agent：已排除。`post-fix` log 已進到 `STEP 1`。
4. 只有長 prompt 會觸發 connection error：已排除。`short` 與 `long` isolated replay 均在 `180s` timeout。
5. timeout/retry 契約是否為主要放大器：已證實。guardrail 套用後，錯誤從 `765.60s + Connection error` 收斂成可預期的 `APITimeoutError`，且 retry backoff 顯著縮短。
6. score recovery 只是沒撿到結果：仍未證實。當前 agent 已能保存 trajectory，但尚未形成 `submitted_count > 0` 的正式 score。

## Next
- 釘住 `host1:50053` 單次 LM query 為何在 `45s` 內仍然 timeout，並判斷是否要：
  - 增加 gateway/serve trace，
  - 或暫時改走更直接的 backend 路徑做 A/B。
- 在單次 LM query timeout 根因收斂後，再推 `submitted_count` 從 `0` 前進到正式 completed score。
