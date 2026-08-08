[OPEN] Debug Session: host1-gateway-timeout

## Symptoms
- `SWE-agent` 在 `host1` 走 `http://127.0.0.1:50053/v1` 時，`probe3-postfix` 可進入 `STEP 1`，但單次 LM query 仍以 `litellm.Timeout: APITimeoutError - Request timed out` 結束。
- `trajectory_count` 已從 `0` 推進到 `1`，但 `submitted_count` 仍為 `0`。
- `host2:30000` backend 本機 completion 正常，`host1:50053` 的 isolated replay 對 short/long prompt 都可在 `180s` 內 timeout。

## Expected
- `host1:50053` 應能穩定回應單次 LM query，至少讓 `SWE-agent` 產生有效 submission，使 `submitted_count > 0`。

## Hypotheses
1. `host1:50053` 的 Ray Serve proxy 已收到請求，但卡在 proxy -> replica 轉發前。
2. 請求已進入 replica，但 replica -> backend / scheduler path 卡住，導致 client 端在 timeout 前收不到回應。
3. `50053` 這層額外包裝才是 timeout 根因；若改走更直接的 backend 路徑，單次 LM query 可成功。
4. `SWE-agent/litellm` 對 local gateway 的 request shape 或連線策略，會觸發與 curl/manual probe 不同的 timeout 路徑。

## Plan
- 補抓 `host1` gateway / serve trace，確認請求是否落到 proxy、replica、backend。
- 做 `50053` vs 更直接 backend 路徑的 A/B，確認 timeout 是否由 gateway 層引入。
- 以最小修補原則收斂根因，再比較 `recent_errors / trajectory_count / submitted_count`。

## Current Status
- Session initialized. No new business logic changes in this debug session.

## Evidence Update
- `host1` 的 `api_server.log` 內存在大量 `POST /v1/chat/completions 200 OK`，說明 gateway/serve 並非全域不可用。
- `host1` 的歷史 Ray Serve log 出現 `ProxyActor` queue-length deadline warning，顯示 proxy -> replica 這層曾有明確壓力訊號。
- A/B curl probe:
  - `host1 -> 50053`: `curl (28) Operation timed out after 15002 milliseconds with 0 bytes received`
  - `host1 -> 47.95.250.55:30000`: `curl (28) Connection timed out after 15002 milliseconds`
- 當前解讀：
  - `worker:30000` 不是可直接替代的 head-side backend 路徑，因為從 `host1` 直連本身就不通。
  - `50053` 雖在 listen，但對這次 probe 在 15s 內沒有返回任何 bytes，仍需進一步鎖定是 proxy queue、replica dispatch，還是 backend response 前卡住。
