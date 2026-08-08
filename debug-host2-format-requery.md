# Debug Session: host2-format-requery

Status: OPEN

## Goal
- 收實 `host2` 上 `astropy__astropy-13453` 的 `FormatError -> requery` 根因。
- 僅在 harness/config 層尋找最小 `thought_action` contract alignment patch。

## Hypotheses
1. `SWE-agent 1.1.0` 的 `thought_action` parser 期待嚴格的 `DISCUSSION` + `COMMAND` 雙段格式，但模型回應只給了 `DISCUSSION`。
2. bridge 會對上游文本做包裝或抽取，導致原本可解析的回應在進 agent 前失去 `COMMAND` 區段。
3. stop 序列或回應截斷讓模型只輸出前半段，因此 parser 看見 `DISCUSSION` 但看不到 `COMMAND`。
4. 當前 harness/config 沒把最適合 `deepseek-v4-flash` 的 parse contract/prompt 模板帶進去，造成格式契約失配。

## Evidence Plan
- 先核對 `SWE-agent 1.1.0` 對 `thought_action` 的精確格式要求。
- 再比對 bridge `extracted/final_text_preview` 與 sample/trajectory 中實際收到的文本形狀。
- 最後只在 harness/config 層做最小對齊，並立即重跑樣本驗證。

## Evidence
- `parsing.py` 已收實 `ThoughtActionParser` 期待「discussion + 最後一個 fenced code block 命令」，而不是 function-calling。
- `config/default.yaml` 預設其實是 `function_calling`，但現場 harness 又強制 `--agent.tools.parse_function.type thought_action`，形成 parser/prompt 不對齊。
- 舊樣本中 `bridge final/extracted` 都只有 `DISCUSSION`，沒有 `COMMAND`，且 `bridge_rewrite_detected = false`，排除 bridge 為主因。
- 舊樣本明確出現 `FormatError -> Requerying model`，主因判定為 `thought_action_format_missing_command`。

## Fix
- 將 harness 預設 config 從 `config/default.yaml` 對齊到 `config/default_backticks.yaml`。
- 在 runtime config 生成時，顯式補入 `thought_action` 的 `RESPONSE FORMAT` 提示。
- 單題 sample helper 顯式帶入：
  - `CGC_SWEBENCH_CONFIG=config/default_backticks.yaml`
  - `CGC_SWEBENCH_ALIGN_THOUGHT_ACTION=1`

## Post-Fix Observation
- 新樣本 `host2_image_resolver_smoke_20260701T050614` 中，`FormatError` 與 `Requerying model` 暫未再出現。
- `traj path correlation probe` 已證實 `.traj` 其實已落到 instance 目錄，先前只是 probe 抓得太早。
- 第一個非空事件已被收實：
  - role = `assistant`
  - action = `cd /testbed && grep -r "class Html" --include="*.py" | head -5`
- `action-to-observation gate probe` 進一步證實：
  - 第一條 `grep` action 的確有被執行。
  - 第一條命令對大小寫敏感，回傳了「成功但無輸出」。
  - 第二條 `grep "class HTML"` 已回到 agent，觀測結果被寫成下一個 `user` history 項，而不是同一個 step 的 `observation` 欄位。
- 因此目前已排除 `action -> observation` 邊界作為主 blocker；當前 frontier 後移到更後段的多輪推進/上游 timeout 行為。

## Multi-Turn Progress
- 前 3 輪 assistant action / user observation 都已成立：
  - Turn 1: `grep "class Html" | head -5` -> 成功但無輸出
  - Turn 2: `grep "class HTML" | head -10` -> 成功回傳 `astropy/io/ascii/html.py:class HTML...`
  - Turn 3: `grep "class Html" | head -10` -> 再次成功但無輸出
- 第 3 輪最後一次成功 assistant 回應時間是 `2026-07-01 05:10:59`。
- 之後不是立刻進入 shell/tool 失敗，而是下一次 LM 查詢在 `2026-07-01 05:41:01` 開始出現 `APITimeoutError`，之後持續 retry。
- 因此 timeout 邊界位於「第 3 輪之後的下一次模型查詢」，而不是第 2 或第 3 輪內部。

## Current Decision
- 當前更應優先處理 `bridge/provider` 長延遲與 timeout，而不是再縮 agent 早期探索路徑。
- 原因是早期探索前 3 輪已可正常推進；真正被打斷的是後續 LM round-trip。

## Bridge/Provider Latency Probe
- 已新增 `probe_host2_bridge_provider_latency_gate.py`，專門對齊：
  - client 側最後一次成功 LM 回應 `2026-07-01 05:10:59,558`
  - 第一次 timeout retry `2026-07-01 05:41:01,438`
  - sample `STEP 4` 的最後一個 `MODEL INPUT`
  - `run_cgc_cloud_openai.log` 內的 `request_received / upstream_sent / first_token / completed_or_timeout / bridge_completed`
- `client_timeout_window` 已收實：
  - 第 3 輪最後一次成功 response id = `6941087dfd4247dfa6cd929acbdda8f8`
  - timeout 並非只到第 5 次，後續還有第 6 次 retry（`2026-07-01 08:12:02,046`）
  - 對齊出的 bridge trace window 是 `1782853739558 -> 1782855781438`
- `sample_timeout_window` 已收實：
  - 第 4 次 `MODEL INPUT` 緊接在 `grep "class Html" | head -10` 成功但無輸出之後
  - timeout 是從 `STEP 4` 的模型查詢直接開始，而不是 shell/tool 邊界
- `bridge_phase_rollup` 顯示：
  - `run_cgc_cloud_openai.log` 最近仍有 `request_received -> upstream_sent -> completed_or_timeout -> bridge_completed` 的 180 秒級錯誤樣本
  - 這些 trace 共有明確的 `timeout_s = 180`
  - 但最近一批 trace 的最後活動時間停在 `1782853730186`，早於當前 client timeout window 起點 `1782853739558`
- `bridge_window_focus` / `derived_assessment` 已收實：
  - 在 `05:10:59 -> 05:41:01` 這個 client timeout window 內，`run_cgc_cloud_openai.log` 沒有任何對應 trace
  - 目前分類是 `client-timeout-before-bridge-trace`
  - 這表示第 4 次 LM query 沒有穩定進到這份 bridge log 所代表的路徑，不能直接把當前 blocker 歸咎於 `run_cgc_cloud_openai.py`
- 先前 liveness 證據與本次 probe 一起看，當前更可信的 ingress 路徑是：
  - `localhost:8001` -> `ray::ProxyActor`
  - `cgc-gateway-port-8001` ServeReplica
  - `127.0.0.1:30000` sglang backend

## Refined Decision
- 當前不應優先調 `run_cgc_cloud_openai.py` 的 timeout/retry，因為第 4 次 LM query 沒有進到它的 trace 視窗。
- 下一步應直接補 `host2 8001 ray/serve ingress latency probe`，只盯：
  - `ray::ProxyActor / cgc-gateway-port-8001` 是否在第 4 次 LM query 期間收到了請求
  - 這次請求是在 `8001 -> 30000` 之間卡住、在 ServeReplica 內排隊，還是根本沒有進到 Ray path
  - 再決定是修 Ray/Serve ingress timeout，還是修 `30000` backend 生成延遲

## Ray/Serve Ingress Probe
- 已新增 `probe_host2_ray_serve_ingress_latency_gate.py`，專門對齊：
  - client timeout window：`2026-07-01 05:08:59,558 -> 2026-07-01 05:43:01,438`
  - Ray Serve proxy log：`/data/ray/host2-cgc/session_latest/logs/serve/proxy_172.30.132.117.log`
  - `cgc-gateway-port-8001` replica worker log
  - `ray_serve_sglang_backend.log`
- `meta` 已收實目前 ingress 拓撲：
  - `0.0.0.0:8001` 由 `ray::ProxyActor` 監聽
  - `ray::ServeReplica:cgc-gateway-port-8001:cgc-sglang-openai-gateway` 存活
  - `127.0.0.1:30000` 由 `sglang.launch_server` 監聽
- `ray_proxy_window` 已收實：
  - 第 4 次 LM query 並非沒進 `8001`
  - proxy log 在關鍵視窗內可見：
    - `05:10:59,556` `POST /v1/chat/completions 200 10333.5ms`
    - `05:20:59,884` `POST /v1/chat/completions 499 600077.1ms`
    - `05:31:00,411` `POST /v1/chat/completions 499 600058.2ms`
    - `05:41:01,423` `POST /v1/chat/completions 499 600100.6ms`
- 補上 replica pid 對位之後，`replica_pid_window` 已收實：
  - `cgc-gateway-port-8001` 的 replica pid = `716728`
  - 對應 worker log 也能看到同一批 request：
    - `05:10:59,555` `POST /v1/chat/completions 200 10331.0ms`
    - `05:20:59,884` `POST /v1/chat/completions CANCELLED 600075.5ms`
    - `05:31:00,411` `POST /v1/chat/completions CANCELLED 600055.7ms`
    - `05:41:01,423` `POST /v1/chat/completions CANCELLED 600098.3ms`
- 同一視窗內，`backend_30000_window` 仍沒有對應 hits。
- 因此目前最可信的分類已從 `request-reached-proxy-but-not-replica` 收斂為：
  - `request-likely-queued-or-stalled-inside-serve-replica`

## Current Decision
- 現在不應優先修 `30000 backend` 生成延遲，因為在當前 client timeout window 內還沒有收實 backend 對應請求。
- 也不應回頭再縮 agent 早期探索，因為前 3 輪已正常推進。
- 下一步更應優先處理 `Ray/Serve ingress timeout/cancellation`：
  - 查 `cgc-gateway-port-8001` 為何在 600 秒後把 chat request 標成 `CANCELLED`
  - 查這個 cancellation 是 proxy/client disconnect、ServeReplica 內部 timeout，還是 gateway 自己等待上游過久後放棄

## Cancellation Cause Probe
- 已新增 `probe_host2_ray_serve_cancellation_cause.py`，專門對齊：
  - client `Retrying LM query: attempt N`
  - proxy access log 的 request id / 499 / elapsed_ms
  - replica worker access log 的同 request id / `CANCELLED`
- 當前關鍵視窗內的 3 個 cancelled request 已收實：
  - `3f763430-b15c-421b-b103-59f9151d87c8`
    - start est = `2026-07-01 05:10:59,806`
    - end = `2026-07-01 05:20:59,884`
    - proxy = `499`
    - worker = `CANCELLED`
  - `ad516a0b-e83a-4167-849e-70ee87c79f19`
    - start est = `2026-07-01 05:21:00,352`
    - end = `2026-07-01 05:31:00,411`
    - proxy = `499`
    - worker = `CANCELLED`
  - `089db19b-3700-454b-9bd4-e5e4c767e801`
    - start est = `2026-07-01 05:31:01,322`
    - end = `2026-07-01 05:41:01,423`
    - proxy = `499`
    - worker = `CANCELLED`
- 最關鍵的對位是最後一筆：
  - `089db19b-...` 的 proxy/worker 結束時間是 `2026-07-01 05:41:01,423`
  - client `attempt 1` 的 timeout 記錄是 `2026-07-01 05:41:01,438`
  - 兩者只差 `15ms`
- probe 的 `alignment_assessment` 已收斂到：
  - `likely-client-disconnect-after-client-timeout`
- 目前還沒有在同 request id 的周邊收實「ServeReplica 自己先 timeout 再反向取消 client」這種更強證據。

## Updated Decision
- 當前更應優先處理 `client -> Ray/Serve` 的 timeout budget / disconnect handling 對齊，而不是先修 `30000 backend`。
- 更精確地說，下一步應優先檢查：
  - `sweagent/litellm/openai client` 端的 timeout budget 是否過短
  - `Ray Serve proxy/replica` 在 client 先斷線後的 cancellation 傳播是否只是被動記錄
- 在補齊這一格之前，不應把主因宣稱為 `ServeReplica` 內部 timeout 或 `30000 backend` 生成太慢。

## Client Timeout Budget Probe
- 已新增 `probe_host2_client_timeout_budget_alignment.py`，專門對齊：
  - `run_full_swebench.sh` 與 runtime yaml 是否有顯式 timeout 設定
  - `SWE-agent -> litellm -> openai` 這條 client stack 的預設 timeout / retry
  - `05:41:01` 這次 timeout 與 `proxy 499` request 的實際 duration
- `run_full_swebench.sh` / `run_full_swebench.runtime.yaml` 已收實：
  - 沒有顯式 `timeout`、`request_timeout`、`max_retries` 設定
  - 目前只有 `api_base=http://localhost:8001/v1` 等基本模型參數
- 直接讀 `sweagent/agent/models.py` 已收實：
  - `litellm.completion(...)` 呼叫只帶：
    - `model / messages / temperature / top_p / api_version / api_key / fallbacks`
    - `**completion_kwargs`
    - `**extra_args`
  - 現場 `completion_kwargs` 沒有顯式 timeout
  - 也就是這次 host2 樣本沒有在 harness / sweagent 層主動把 timeout 設成 600 秒
- venv 套件預設已收實：
  - `openai==2.44.0`
  - `httpx==0.28.1`
  - `openai._constants.DEFAULT_TIMEOUT = Timeout(connect=5.0, read=600, write=600, pool=600)`
  - `openai._constants.DEFAULT_MAX_RETRIES = 2`
  - `litellm.request_timeout = 6000.0`
- SWE-agent 模型配置與 retry 已收實：
  - `GenericAPIModelConfig` 預設 `retry = RetryConfig(retries=20, min_wait=10, max_wait=120)`
  - `LiteLLMModel.query()` 會對 `_query()` 套 tenacity retry
  - 因此單次 LM request 的 600 秒 timeout 之外，還會再疊加多輪 retry/backoff
- 實際觀測對帳已收實：
  - `first_timeout = 2026-07-01 05:41:01,438`
  - 最接近的 proxy request：
    - `rid = 089db19b-3700-454b-9bd4-e5e4c767e801`
    - `status = 499`
    - `elapsed_ms = 600100.6`
    - `end_ts = 2026-07-01 05:41:01,423`
    - 與 client timeout 只差 `15ms`
- probe 的最終分類已收斂為：
  - `client-timeout-budget-aligned-with-600s-cancel-window`

## Refined Decision
- 現在沒有證據支持「client timeout 與 Ray Serve 600s cancellation window 存在明顯 budget mismatch」。
- 更可信的結論是：
  - 單次 request timeout 主要吃到 `openai` 預設 `600s`
  - 長時間卡死的放大器則是 `SWE-agent` 的 `RetryConfig(retries=20, min_wait=10, max_wait=120)`
- 因此下一步不應優先把 root cause 定義成 `client timeout budget` 本身。
- 更合適的下一步是補更深一層的 upstream request tracing，確認：
  - 這筆 600 秒 request 在 `cgc-gateway-port-8001` 內是否真的有向上游送出
  - 若有送出，是卡在 gateway -> backend 轉發、backend 排隊，還是 backend 無對應 access log
- 若只是為了縮短 debug 迭代時間，後續可以再做一個最小 harness-only patch，暫時降低 retry 次數或縮短 timeout；但這應視為觀測優化，不是根因修復。

## Upstream Request Trace
- 已新增 `probe_host2_upstream_request_trace.py`，以 `05:41:01` 這筆 timeout request 為主鍵，對齊：
  - proxy access log
  - `cgc-gateway-port-8001` replica worker log
  - backend log `ray_serve_sglang_backend.log`
  - 同 request id 在 Ray logs 內的所有命中
- 目標 request 已再次收實為：
  - `rid = 089db19b-3700-454b-9bd4-e5e4c767e801`
  - `start_ts_est = 2026-07-01 05:31:01,322`
  - `end_ts = 2026-07-01 05:41:01,423`
  - `status = 499`
  - `elapsed_ms = 600100.6`
  - 與 client `attempt 1 @ 05:41:01,438` 只差 `15ms`
- `gateway_rid_context` / `backend_rid_and_ray_logs` 已收實：
  - proxy log 有：
    - `Client for request 089db19b-... disconnected, cancelling request.`
    - `POST /v1/chat/completions 499 600100.6ms`
  - replica worker log 有：
    - `POST /v1/chat/completions CANCELLED 600098.3ms`
  - 但 backend 端沒有同 request id 命中
- `backend_window_trace` 已收實：
  - 在 `05:10:59 -> 05:41:01` 這個關鍵視窗內，`ray_serve_sglang_backend.log` 的 `hit_count = 0`
  - 不是只有缺 request id，而是連同時間窗的 chat/generate/request/error 類事件都沒有可對上的 backend log 訊號
- probe 的最終分類已收斂為：
  - `phase-gap-between-gateway-worker-and-backend-log`

## Current Decision
- 目前最可信的 phase 缺口位置不是 client，也不是 proxy ingress，而是：
  - request 已到 proxy
  - 已到 `cgc-gateway-port-8001` replica worker
  - 但 backend 觀測面完全沒有對應痕跡
- 因此下一步更應優先處理：
  - `gateway -> backend dispatch` 的可觀測性與取證
  - 先確認 gateway 內部是否真的有對上游發出請求，再決定是修轉發/排隊，還是做 harness-only timeout/retry 觀測優化
- 在這一格未補齊之前，不應把 blocker 簡化成：
  - 單純 backend 太慢
  - 或單純 retry 太多

## Gateway Dispatch Confirmation
- 已新增 `probe_host2_gateway_dispatch_confirmation.py`，專門對齊：
  - 現場 `cgc-gateway-port-8001` worker 的 chat handler 實碼
  - 目標 request `089db19b-3700-454b-9bd4-e5e4c767e801`
  - worker 歷史 `GatewayTrace`
  - backend `30000` 的當前可達性與 backend log 可觀測面
- 當前最重要的 deployed code facts 已收實：
  - 遠端 `ray_serve_sglang_gateway.py` 的 chat route 已包含：
    - `trace_id`
    - `8001_request_received`
    - `8001_upstream_sent`
    - `30000_first_token`
    - `8001_completed_or_timeout`
  - chat handler 仍是直接：
    - `requests.post(f"{self.backend_base_url}/v1/chat/completions", ..., timeout=(10, 1800))`
  - 也保留：
    - `except requests.RequestException`
    - `backend_unavailable` error path
- worker log 已收實：
  - 歷史上同一個 worker 確實多次留下：
    - `8001_request_received`
    - `8001_upstream_sent`
    - `30000_first_token`
    - `8001_completed_or_timeout`
  - 而且 `8001_completed_or_timeout` 明確記錄：
    - `HTTPConnectionPool(host='127.0.0.1', port=30000): Read timed out. (read timeout=1800)`
- 這表示：
  - `cgc-gateway-port-8001` worker 的 upstream dispatch path 並不是假的
  - 它歷史上確實會把 request 往 `127.0.0.1:30000/v1/chat/completions` 送出
  - 某些 request 也確實拿到過 `30000_first_token`
- 對當前 target request `089db19b-...`：
  - 已收實：
    - `499 @ 2026-07-01 05:41:01,423`
    - `elapsed_ms = 600100.6`
    - 與 client `attempt 1` 只差 `15ms`
  - 但因 `GatewayTrace` 用的是 `trace_id` 而不是 request id，當前還不能把 target request 的 `8001_upstream_sent` 逐條精準對位到同一個 rid
  - `target_start_window_phases` 只抓到前一筆 cancelled task 的邊界，沒有直接抓到 target 自己的 phase line
- backend 端另外收實：
  - `127.0.0.1:30000/v1/models` 當前可用
  - `127.0.0.1:30000/health` 當前 8 秒內超時
  - `ray_serve_sglang_backend.log` 仍沒有與這批 request 對齊的可用觀測
- probe 的最終分類已收斂為：
  - `dispatch-path-exists-but-backend-observability-gap`

## Updated Decision
- 現在更可信的結論是：
  - 不是 dispatch path 缺失
  - 而是 gateway 已具備 upstream dispatch 與 timeout trace，但 backend 觀測面不足，導致 target request 在 `worker -> backend` 之間無法逐筆閉環
- 因此下一步不應先做 harness-only timeout/retry patch。
- 更應優先補：
  - `gateway -> 30000 backend` 的 trace correlation / observability
  - 特別是把 `GatewayTrace.trace_id` 與 backend request/rid 對起來，或直接在 backend 入口補最小 request trace

## Gateway-Backend Correlation
- 已新增 `probe_host2_gateway_backend_trace_correlation.py`，專門對齊：
  - `GatewayTrace.trace_id`
  - backend `TokenizerTrace.trace_id -> rid`
  - backend `SchedulerTrace/PrefillAdderTrace`
  - `30000 /v1/models`、`/health`、`/health_generate` 的 phase 差異
- 對 target request `089db19b-3700-454b-9bd4-e5e4c767e801`：
  - `start_ts_est = 2026-07-01 05:31:01,322`
  - `gateway_trace_start_window.hits = []`
  - `backend_trace_correlation.candidate_trace_ids = []`
  - `backend_trace_correlation.tokenizer_hits = []`
  - `backend_trace_correlation.scheduler_hits = []`
  - `backend_trace_correlation.prefill_hits = []`
- 也就是：
  - 在 target request 啟動窗口內，沒有收實任何可把 `GatewayTrace.trace_id` 對到 backend `TokenizerTrace/SchedulerTrace` 的現場證據
  - 這和前一輪「dispatch path 存在」不矛盾；它說明的是：
    - 當前 target request 缺少逐筆 correlation，backend 觀測面仍不足
- `30000` 的 phase 差異已再次收實：
  - `/v1/models`：`200`，`~1ms`
  - `/health`：`12s timeout`
  - `/health_generate`：`12s timeout`
  - 同時，probe 後最近時間窗內沒有新增 `HealthTrace/TokenizerTrace`
- 這表示：
  - `models` 只走 metadata path，幾乎不碰 generate pipeline
  - `health` / `health_generate` 會觸發 generate path，但當前 host2 在這一層既不返回，也沒有留下足夠 trace
- 以 `host1` 作為對照基準，現場差異更明確：
  - `host1_health_rid_trace.json` 已證明正常情況下，health request 會快速落到：
    - `SchedulerTrace.handle_generate_request_received`
    - `request_enqueued_waiting_queue`
    - 後續 scheduler/prefill 痕跡
  - `host2` 當前則是：
    - `/health` / `/health_generate` timeout
    - 無對應 `HealthTrace/TokenizerTrace`
- probe 的最終分類目前仍標成：
  - `needs-more-trace`
- 但這個 `needs-more-trace` 的含義已經很具體：
  - 不是還不知道去哪裡找
  - 而是下一步必須在 backend generate path 入口補最小 request trace，否則 `trace_id -> rid` 不可能在 host2 上被逐筆閉環

## Current Decision
- 在當前證據下，不應先做最小 harness-only timeout/retry 觀測優化 patch。
- 更優先的下一步應該是：
  - 補 backend 最小 request trace
  - 直接照亮 `30000` 的 `/health_generate` 與 chat generate path 是否進到 `TokenizerTrace.generate_start`
- `host1` 可以繼續作為對照基準，但主線不需要切去 host1；host2 的下一格已經收斂為 backend 入口可觀測性。

## Backend Minimal Request Trace
- 已新增 `probe_host2_backend_minimal_request_trace.py`，對 `127.0.0.1:30000` 主動打兩筆受控請求：
  - `/health_generate` with `x-cgc-trace-id = host2-backend-health-<ts>`
  - `/v1/chat/completions` with `x-cgc-trace-id = host2-backend-chat-<ts>`
- 新結果與前一輪不同，而且更關鍵：
  - `30000 /v1/models`：`200`
  - `30000 /health_generate`：`200`，約 `1.0s`
  - `30000 /health`：`200`，約 `1.0s`
  - 最小 chat request 也成功返回單 token completion
- 也就是：
  - `30000` backend 並不是持續不可用
  - generate path 在當前時刻至少可以直接被打通
- 但同時，對這兩筆受控請求的 trace grep 已收實：
  - 已新增 `grep_host2_backend_probe_trace.py`
  - 對最新 `host2-backend-health-1782875913` / `host2-backend-chat-1782875913` 在：
    - `worker-*.err`
    - `worker-*.out`
    - `python-core-worker*.log`
    做精準 grep
  - 結果 `stdout = ""`
  - 沒有任何：
    - `TokenizerTrace`
    - `SchedulerTrace`
    - `HealthTrace`
    - `handle_generate_request_received`
    - `generate_start`
    - `request_enqueued_waiting_queue`
- 這一步很重要，因為它把問題重新定義成：
  - 不是 backend generate path 一定掛住
  - 而是 backend 入口 trace 在 host2 現場沒有被穩定寫出來，至少不在當前看的 worker/core-worker log 集合內

## Refined Decision
- 現在更不應先做 harness-only timeout/retry 觀測優化 patch。
- 更應優先補的是：
  - backend 入口 observability
  - 尤其是 `30000` 在收到 `/health_generate` / chat request 後，立即寫一條最小 request trace
- 原因是：
  - 現在 HTTP 層已證明 request 可以成功
  - 但 `TokenizerTrace.generate_start -> SchedulerTrace.handle_generate_request_received` 仍完全不可見
  - 如果不先補 backend 入口 trace，就無法再把後續 timeout 精確落到 tokenizer、scheduler，還是更上游的某一格

## Backend Ingress Observability
- 已新增：
  - `patch_remote_host2_backend_ingress_observability.py`
  - `restart_host2_backend_30000.py`
  - `probe_host2_backend_import_path.py`
  - `probe_host2_backend_real_module_path.py`
  - `probe_host2_backend_ingress_followup.py`
- 新收實的生效路徑很重要：
  - host2 `30000` backend 並不是跑 `/root/flashkv0516/venv/bin/python`
  - 而是：
    - `/root/host2_torch_venv/bin/python -m sglang.launch_server ... --port 30000`
  - 實際載入的模組檔是：
    - `/usr/local/lib/python3.12/dist-packages/sglang/srt/entrypoints/http_server.py`
    - `/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/tokenizer_manager.py`
- 因此 ingress instrumentation 已直接 patch 到 host2 真正使用的 dist-packages `http_server.py`，並通過：
  - `py_compile_rc = 0`
- instrumentation 內容只做最小入口取證，不改推理邏輯：
  - `health_ingress`
  - `chat_ingress`
  - 寫到：
    - `/root/flashkv0516/logs/backend_ingress_probe.ndjson`

## Restart Attempt
- 為了讓新 instrumentation 生效，我對 `30000` backend 做了最小 restart。
- 這一步沒有成功把新版本穩定載入，且收實了更前一格 blocker：
  - restart 後新進程一度起來，但隨後被打掉
  - `backend_ingress_probe.ndjson` 仍是 `0 bytes`
  - follow-up probe 顯示當前 `30000` 存活的是新的 pid：
    - `2385111`
  - 但 restart log 關鍵錯誤已收實：
    - `torch.distributed.DistNetworkError`
    - `port: 29923`
    - `EADDRINUSE`
    - `address already in use`
- 這表示：
  - 當前不是 ingress trace 代碼本身有錯
  - 而是 host2 backend 在 reload 過程中，被既有 `dist-init-addr=172.30.132.117:29923` 的佔用衝突卡住
  - 因而新的 instrumentation 雖然已 patch 到正確檔位，但尚未被穩定載入到可用 backend 進程

## Updated Decision
- 目前更不應先做 harness-only timeout/retry patch。
- 也不應再把主因說成 backend HTTP 不可用。
- 更精確的當前 blocker 是：
  - host2 backend ingress observability patch 已就位
  - 但 backend scoped restart 被 `dist-init-addr :29923` 佔用衝突阻斷
- 所以下一步更應優先做：
  - host2 backend scoped restart cleanup
  - 先把 `29923` 相關殘留進程或舊 scheduler/worker 清乾淨，再讓帶 instrumentation 的 `30000` backend 穩定起來

## Scoped Restart Cleanup
- 已新增 `probe_host2_backend_scoped_restart_cleanup.py`，把這一格收窄成：
  - 誰佔了 `dist-init-addr 172.30.132.117:29923`
  - scoped cleanup 後 `30000` backend 能否穩定 reload
  - reload 後 `backend_ingress_probe.ndjson` 與 `TokenizerTrace/SchedulerTrace` 能否閉環
- 這一輪先收實了 `29923` 的真正 holder：
  - `BACKEND_PID_BEFORE = 2385111`
  - `HOLDER_PIDS_BEFORE = 2385160`
  - `lsof/ss` 對齊到：
    - `sglang::scheduler_TP0_EP0` 以 `pid=2385160` 監聽 `*:29923`
  - 同一批殘留還包括：
    - `sglang::scheduler_TP1_EP1`
    - `sglang::scheduler_TP2_EP2`
    - `sglang::scheduler_TP3_EP3`
    - `sglang::detokenizer`
- 這表示：
  - `29923` 的 holder 不是別的外部服務
  - 而是上一輪 `30000` backend 自己殘留下來的 scheduler 子進程
- scoped cleanup 後：
  - `lsof_29923_after_cleanup` 已空
  - `ss_29923_after_cleanup` 已空
  - 新 backend process 也確實重新啟動：
    - `NEW_PID = 2408505`
- 但 reload 之後仍然沒有穩定起來：
  - `models_probe` / `health_generate_probe` / `chat_probe` 都是：
    - `CURL_CODE:000`
    - `Failed to connect to 127.0.0.1:30000`
  - `backend_ingress_probe.ndjson` 仍是 `0 bytes`
  - 所以 instrumentation 雖已 patch 到正確檔位，但仍未真正進入可服務的 backend process

## New Root Cause Signal
- 這一輪最重要的新證據不是 `EADDRINUSE`，而是 reload 後更深一層的啟動崩潰：
  - restart log 出現多次：
    - `!!!!!!! Segfault encountered !!!!!!!`
    - `plugin/net.cc`
    - `ncclNetPluginInit`
    - `ncclCommInitRank`
- 也就是：
  - 清掉 `29923` 之後，舊的 `address already in use` 確實被打掉了
  - 但新的 blocker 往更深一層後移到：
    - NCCL / net plugin 初始化崩潰
    - 導致 `30000` backend 根本沒成功 listen 起來
- 因此這一輪的決策也同步更新：
  - 當前不是繼續做 timeout/retry patch
  - 也不是 ingress ndjson 還不夠
  - 而是 `host2` backend reload path 本身在 NCCL/net plugin 初始化階段崩潰

## NCCL Net Plugin Crash Probe
- 已新增：
  - `probe_host2_backend_nccl_net_plugin_crash.py`
  - `extract_host2_backend_nccl_probe_summary.py`
- 這一輪把問題收得更窄，只盯：
  - crash stack 第一個關鍵 frame 到底落在 `ncclNetPluginInit`、`ncclCommInitRank` 還是更前面的 plugin 裝載
  - 同一個 `host2_torch_venv` 下，最小 controlled repro 是否也能重現
- 新收實的結論已足夠明確：
  - `restart_log_has_ncclNetPluginInit = true`
  - `restart_log_has_ncclCommInitRank = true`
  - `restart_log_has_plugin_net_cc = true`
  - `restart_log_has_segfault = true`
  - stack 先後順序收斂為：
    - `plugin/net.cc:216 in ncclNetPluginInit`
    - `plugin/net.cc:362 in ncclNetInit`
    - `init.cc ... ncclCommInitRankFunc / ncclCommInitRankDev / ncclCommInitRank`
    - `pynccl_wrapper.py:404 in ncclCommInitRank`
- 也就是：
  - 問題不是先從 harness 或 Python control path 炸開
  - 而是更接近 NCCL IB/net plugin 初始化本身

## Controlled Repro
- 我在 host2 真正使用的：
  - `/root/host2_torch_venv/bin/python`
  下跑了最小單卡 `torch.distributed` controlled repro，結果如下：
  - `default_rc = -11`
  - `ib_disable_rc = 0`
  - `socket_rc = 0`
- 更具體地說：
  - default case：
    - 已完成 `init_process_group_done`
    - 之後直接 `timeout: the monitored command dumped core`
  - `NCCL_IB_DISABLE=1`：
    - 可成功走到 `all_reduce_done`
    - 並正常 `done`
  - `NCCL_IB_DISABLE=1 + NCCL_NET=Socket + NCCL_SOCKET_IFNAME=eth0`：
    - 同樣可成功走到 `all_reduce_done`
    - 並正常 `done`
- 這一步非常關鍵，因為它把主因從「泛化的 backend 啟動參數問題」進一步收斂成：
  - host2 預設 NCCL IB/net plugin 路徑會 core dump
  - 不是殘留 scheduler/port cleanup 沒做乾淨
  - 也不是任意 NCCL 初始化都會失敗

## Refined Root Cause
- 綜合：
  - 先前既有 matrix：host2 預設 NCCL collective 會 `-11`，而 `NCCL_IB_DISABLE=1` 類配置可 `PASS`
  - 本輪 restart crash stack：`plugin/net.cc -> ncclNetPluginInit -> ncclCommInitRank`
  - 本輪同 venv controlled repro：`default=-11`, `IB_DISABLE=1=PASS`, `Socket=PASS`
- 現在最可信的分類已收斂為：
  - `likely-ib-net-plugin-path`
- 因此這一輪的決策也更新為：
  - 不先修 harness-only timeout/retry
  - 不先繼續追 ingress ndjson
  - 更應先修 host2 backend 啟動路徑上的 NCCL IB/net plugin 使用策略

## Backend Startup Path Patch
- 已對 host2 backend restart path 做最小 patch：
  - `restart_host2_backend_30000.py`
  - `probe_host2_backend_scoped_restart_cleanup.py`
- 當前預設會用：
  - `env NCCL_IB_DISABLE=1`
  啟動 `30000` backend
- 這一步的驗證結果已經很關鍵：
  - `STARTUP_ENV: env NCCL_IB_DISABLE=1` 已確實帶進 reload path
  - follow-up probe 顯示：
    - `127.0.0.1:30000` 已 `LISTEN`
    - `127.0.0.1:29923` 也由新 `scheduler_TP0_EP0` 正常監聽
    - `/v1/models`：`200`
    - `/health_generate`：`200`
- 也就是：
  - 只加 `NCCL_IB_DISABLE=1` 就已足以把 host2 backend startup path 打通
  - 這一步暫時不需要再把 `NCCL_NET=Socket` 一起固化

## Post-Patch Verification
- 已新增：
  - `probe_host2_backend_after_ib_disable.py`
  - `probe_host2_backend_trace_after_ib_disable.py`
- 受控驗證結果如下：
  - `/v1/models`：`200`
  - `/health_generate`：`200`
  - 最小 `/v1/chat/completions`：`200`
  - chat completion 已返回一個 token：`Under`
- backend ingress ndjson 也已正式打通：
  - `health_ingress` 已落出對應 `trace_id = host2-ib-health-1782890557`
  - `chat_ingress` 已落出對應 `trace_id = host2-ib-chat-1782890557`
- 但同一輪 grep `worker-*.err` / `worker-*.out` / `python-core-worker*.log` 時：
  - `trace_hits = ""`
  - 仍未看到：
    - `TokenizerTrace.generate_start`
    - `SchedulerTrace.handle_generate_request_received`
    - `request_enqueued_waiting_queue`

## Updated Decision After Patch
- 這一步已足夠做出新的決策：
  - 當前不需要先加 `NCCL_NET=Socket`
  - 因為 `NCCL_IB_DISABLE=1` 已經讓 backend 穩定 `listen` 並可服務 health/chat
- 下一個真正的 blocker 不再是 backend startup path，而是：
  - fixed backend 之後，`TokenizerTrace/SchedulerTrace` 的 log surface / trace sink 仍未在當前 worker/core-worker log 面出現
- 所以下一步更應回到：
  - 在 backend 已穩定的前提下，補更窄的 trace sink/location probe
  - 找出 `generate_start / handle_generate_request_received` 實際被寫到哪一組 log 或 runtime sink

## Fixed-Backend Trace Sink Probe
- 已新增：
  - `probe_host2_fixed_backend_trace_sink.py`
- 這一輪改用同一個新鮮 `chat_ingress trace_id` 做完整對齊：
  - `chat_trace_id = host2-fixed-sink-chat-1782898923`
  - `/v1/chat/completions`：`200`
  - completion 仍返回單 token：`Under`
- `backend_ingress_probe.ndjson` 已再次收實同一筆 ingress：
  - `chat_ingress`
  - `trace_id = host2-fixed-sink-chat-1782898923`
- 最關鍵的新證據是：
  - 同一輪 request 之後，真正出現對應 backend trace 的，不是先前看的 Ray `worker/core-worker` logs
  - 而是：
    - `/root/flashkv0516/logs/host2_backend_30000_scoped_restart.log`
- 同一個 `chat_trace_id` 已在這個 sink 上形成完整閉環：
  - `OpenAITrace stage='handler_received'`
  - `TokenizerTrace stage='generate_start'`
  - `TokenizerTrace stage='dispatch_single_to_scheduler'`
  - `SchedulerTrace stage='handle_generate_request_received'`
  - `SchedulerTrace stage='request_enqueued_waiting_queue'`
  - `TokenizerTrace stage='generate_finished'`
- 同一個 request id 也已被對齊出來：
  - `rid = 3d82a15a03324027b5a33a3866c4b62a`
- 關鍵 trace 形狀已收實：
  - `TokenizerTrace.generate_start`：
    - `trace_id='host2-fixed-sink-chat-1782898923'`
    - `rid='3d82a15a03324027b5a33a3866c4b62a'`
  - `SchedulerTrace.handle_generate_request_received`：
    - TP0/TP1/TP2/TP3 都有命中
    - 同樣帶著相同 `trace_id` 與 `rid`
  - `request_enqueued_waiting_queue` 也在同一個 sink 內可見
- `changed_files` 也支持這個判斷：
  - 這次 chat request 後，真正被觸碰的核心檔只有：
    - `backend_ingress_probe.ndjson`
    - `host2_backend_30000_scoped_restart.log`
- probe 的最終分類已可收斂為：
  - `chat-trace-correlates-to-backend-sink`

## Updated Decision After Trace Sink
- 現在已不需要再補 backend log surface / flush 才能看見：
  - `TokenizerTrace.generate_start`
  - `SchedulerTrace.handle_generate_request_received`
- 更精確地說，先前缺的不是 trace 沒寫，而是看錯了 surface：
  - 這批 backend trace 目前實際落在 backend process 的 stdout/stderr sink
  - 即 `host2_backend_30000_scoped_restart.log`
- 因此下一步可以回到原本主線：
  - 重新驗證 agent multi-turn timeout
  - 把同一套已打通的 backend trace sink 用在真正的 `8001 -> 30000` 長 request 上
  - 確認第 4 次 LM query 卡住時，backend 是否仍有對應 `TokenizerTrace/SchedulerTrace`，以及卡點是在 queue/prefill/batch result 哪一階段

## Multi-Turn Backend Sink Correlation
- 已新增：
  - `probe_host2_agent_multiturn_backend_sink_correlation.py`
- 這一輪仍以第 4 次 LM query 對應的長 request 為主鍵：
  - `rid = 089db19b-3700-454b-9bd4-e5e4c767e801`
  - `start_ts_est = 2026-07-01 05:31:01,322`
  - `end_ts = 2026-07-01 05:41:01,423`
- `8001` 與 gateway worker 面仍然閉環成立：
  - proxy：
    - `Client for request 089db19b-... disconnected, cancelling request`
    - `POST /v1/chat/completions 499 600100.6ms`
  - `cgc-gateway-port-8001` worker：
    - `POST /v1/chat/completions CANCELLED 600098.3ms`
- 但把這個 historical request 去對齊目前的 backend sink
  - `host2_backend_30000_scoped_restart.log`
  時，沒有同窗：
  - `TokenizerTrace`
  - `SchedulerTrace`
  - `request_enqueued_waiting_queue`
  - `prefill_break_add_req`
  - `process_batch_result`
- 這個「沒命中」不能直接解讀成：
  - request 沒有進 `30000`
  - 或 phase gap 一定回退到 `8001 -> 30000`
- 原因是我又做了 direct sink coverage 檢查，收實目前這個 backend sink 的時間邊界是：
  - first timestamp = `2026-07-01 15:19:08`
  - last timestamp = `2026-07-01 17:42:04`
- 也就是說：
  - `host2_backend_30000_scoped_restart.log` 是在後續 `NCCL_IB_DISABLE=1` restart 之後才開始記
  - 它並不覆蓋 historical timeout request 的 `05:31 -> 05:41` 視窗
- 因此這一格更精確的結論應收斂為：
  - `current-backend-sink-does-not-cover-historical-timeout-window`
  - `absence-in-current-backend-sink-is-non-informative`

## Updated Decision After Multi-Turn Sink Correlation
- 目前還不能因為 `host2_backend_30000_scoped_restart.log` 對 `089db19b-...` 沒命中，就把 phase gap 重新宣稱成：
  - `8001 -> 30000` 之間
- 更精確地說：
  - 當前已打通的 backend sink 只證明「現在的 fixed backend 可以把 chat request 的 `TokenizerTrace/SchedulerTrace` 寫出來」
  - 但它無法反證「早先那筆 05:31 歷史 request 沒進 backend generate path」
- 因此下一步若要真正回答 multi-turn timeout 主線，應優先二選一：
  - 找 historical request 當時真正對應的 backend stdout/stderr sink
  - 或在目前已打通 sink 的條件下，重新跑一次 agent multi-turn，直接收第 4 次長 request 在 backend sink 內的 phase
- 就目前證據鏈而言，最穩的決策是：
  - 回到 agent multi-turn 主線
  - 但要用「當前已打通的 backend sink」重跑/重驗，而不是再用它去反推 `05:31` 那筆歷史 request

## Live Backend Sink Rerun
- 已新增：
  - `probe_host2_agent_multiturn_rerun_with_live_backend_sink.py`
- 這一輪改成直接重跑新的 sample：
  - `suffix = host2_live_backend_sink_20260710T220625`
  - sample log:
    - `/root/flashkv0516/swebench_host2_live_backend_sink_20260710T220625.log`
- 原本目標是：
  - 等到新的 `MODEL INPUT 4`
  - 再把同一輪第 4 次 LM query 對齊到 `host2_backend_30000_scoped_restart.log`
  - 看 `TokenizerTrace/SchedulerTrace` 是否命中，以及 phase 落在哪一格
- 但這次 live rerun 沒有跑到可分析的 multi-turn 階段。
- probe 已收實：
  - `stop_reason = timeout_seen_before_model_input_4`
  - `model_input_count = 1`
  - `backend_request_count = 0`
  - `derived_assessment.classification = did-not-reach-model-input-4`
- sample log 的關鍵訊號是：
  - 從 `STEP 1` 開始就持續：
    - `InternalServerError: OpenAIException - Connection error`
  - 目前已至少看到：
    - `attempt 1`
    - `attempt 2`
    - `attempt 3`
    - `attempt 4`
    - `attempt 5`
    - `attempt 6`
- 我又補做了最窄的當前 liveness 檢查，現場已收實：
  - `127.0.0.1:30000` 仍在 `LISTEN`
  - `curl http://127.0.0.1:30000/v1/models`：
    - `200`
  - `127.0.0.1:8001` 沒有 listener
  - `curl http://127.0.0.1:8001/v1/models`：
    - `CURL_CODE:000`
    - `Couldn't connect to server`
  - 同時：
    - proxy tail 空
    - worker `POST /v1/chat/completions` 新增命中空
    - backend sink tail 也沒有對應到新的 agent request
- 因此這一輪更精確的結論不是：
  - 「新的第 4 次 LM query 沒進 backend sink」
- 而是：
  - 根本沒有形成可分析的第 4 次 LM query
  - 因為 rerun 在第一輪 LM query 就已經被 `localhost:8001` 的 connect failure 截斷

## Updated Decision After Live Rerun
- 現在的 active blocker 已從：
  - `historical request sink coverage`
  - 或 `backend sink phase correlation`
  往前收斂到：
  - `gateway liveness`
- 更精確地說，對這次 live rerun：
  - `30000 backend` 是活的
  - 但 `8001 gateway` 當前不可達
  - 所以還不能拿這輪 rerun 去回答「第 4 次 LM query 在 backend sink 落到哪個 phase」
- 這也表示目前不能把缺口正式收斂成：
  - `8001 -> 30000`
- 因為更早一層的事實是：
  - request 連 `8001` 都沒有真正打進去
- 因此下一步若要繼續主線，最小且必要的前置條件是：
  - 先恢復 `localhost:8001` 的可達性
  - 再用同一支 live backend sink probe 重跑
  - 屆時才能真正回答第 4 次 LM query 對應的 `TokenizerTrace/SchedulerTrace` phase

## 8001 Gateway Liveness / Recovery Probe
- 已新增：
  - `probe_host2_8001_gateway_liveness_recovery.py`
- 這一輪一開始的目標是確認：
  - 為什麼 live rerun 當時只看到 `30000` 活著、`8001` 不在
  - 應該先做 gateway recovery，還是先修 probe 對當前 runtime 的觀測面
- probe 先收實了兩個關鍵事實：
  - `127.0.0.1:30000` 仍 `LISTEN` 且 `/v1/models = 200`
  - `127.0.0.1:8001` 沒有 listener，`curl = CURL_CODE:000`
- 但進一步往下看後，新的關鍵發現是：
  - host2 並不是「整個 Ray / Serve 都掛掉」
  - 而是當前 runtime 已不在舊的：
    - `/data/ray/host2-cgc/session_latest`
  - 現場活著的是一套新的本地 Ray cluster / Serve runtime
- 當前存活訊號包括：
  - `gcs_server` 監聽：
    - `*:6379`
  - `raylet` 存活：
    - session dir 落在 `/data2/ray/inst1/session_2026-07-10_22-24-31_896792_699410`
  - `ray::ServeController.listen_for_change`
  - `ray::ProxyActor`
  - `ray::ServeReplica:default:cgc-sglang-openai-gateway`
  - replica 仍持有：
    - `127.0.0.1:30000`
- 最重要的是：
  - `ProxyActor` 當前不是綁在 `8001`
  - 而是實際監聽：
    - `127.0.0.1:50053`
    - 以及 actor internal port `10020`
- 我已直接驗證：
  - `curl http://127.0.0.1:50053/v1/models`
    - `200`
- 所以這一格應更精確地收斂成：
  - `gateway-port-drifted-from-8001`
  - 而不是：
    - `gateway-dead`
    - `8001 -> 30000` gap
- 這也解釋了上一輪 live rerun 為什麼會在 `STEP 1` 就變成：
  - `OpenAIException - Connection error`
  - 因為 sample / harness 仍打 `localhost:8001`
  - 但當前實際可用 gateway port 已經漂到 `50053`

## Updated Decision After 8001 Probe
- 目前最小且必要的 recovery 不再是模糊的：
  - 「把 gateway 救活」
- 而是二選一中的一個明確動作：
  - 恢復預期的 `8001 -> ProxyActor` binding
  - 或暫時把 client / harness 對齊到目前活著的 `50053`
- 就目前主線目標而言，若只想最快回到：
  - `agent multi-turn + live backend sink`
  - 那麼更小的動作其實是：
    - 先把 rerun 用的 API base 從 `8001` 對齊到 `50053`
- 如果目標是恢復原本正式 contract，則下一步要做的才是：
  - 找出這次新的 Ray / Serve runtime 為什麼以 `50053` 起來，而不是 `8001`
  - 再決定是否重建一套回到 `8001` 的 gateway 啟動路徑

## Live Rerun Via 50053
- 已將 live rerun probe 的 API base 最小對齊到：
  - `http://127.0.0.1:50053/v1`
- 這一步的正向收穫是：
  - transport blocker 已被打掉
  - sample 不再停在 `STEP 1 / Connection error`
  - 新的一輪 sample 已實際跑進多輪 LM query
- 這次 rerun 的關鍵路徑：
  - sample log:
    - `/root/flashkv0516/swebench_host2_live_backend_sink_20260710T223909.log`
  - trace log:
    - `/root/flashkv0516/SWE-agent/trajectories/root/run_full_swebench.runtime__openai--deepseek-v4-flash__t-0.00__p-1.00__c-0.00___swe_bench_verified_test__host2_live_backend_sink_20260710T223909/astropy__astropy-13453/astropy__astropy-13453.trace.log`
- probe 已收實：
  - `gateway_api_base = http://127.0.0.1:50053/v1`
  - `model_input_count = 2`
  - sample 沒再出現 `Connection error`
  - backend transport 本身不是這輪 blocker
- 但這次 rerun 最終沒有進到第 4 次 LM query 的 phase 分析，因為 sample 在更前面被新的 blocker 截斷：
  - `Exit due to repeated format/blocklist/bash syntax errors`
  - `exit_format`
- 具體來說，這次第 1~3 次 LM 回覆分別表現為：
  - 第 1 次：
    - 直接輸出自然語言 + `<tool_call>` / `DSML`
    - 沒有進入 `DISCUSSION + fenced code`
  - 第 2 次：
    - 開頭已有 `DISCUSSION`
    - 但後半仍輸出 `<|Assistant|> <tool_call> ... </｜DSML｜>`
  - 第 3 次：
    - 開頭仍是 `DISCUSSION`
    - 但命令區仍變成：
      - `<|command|> ... </command>`
    - 不是正式要求的 fenced code block
- 因此這一輪 agent 雖然已經跨過 transport，但仍在：
  - repeated `FormatError` after requery #3
  - autosubmission
  - `exit_format`
  收尾，沒有形成可用的第 4 次長請求。

## Active Frontier After 50053 Rerun
- 目前離 `tp4ep4 fusionroute 四實例 deepseek-v4-flash swe verified 500` 的一題 demo，比上一輪更近了一步：
  - transport/gateway 已打通到可用 port `50053`
- 但新的 active blocker 已從：
  - `gateway liveness / port drift`
  往後移到：
  - `thought_action contract enforcement`
- 這一格最關鍵的新證據是：
  - active `50053` runtime 的 replica env 中，沒有看到：
    - `CGC_SWEBENCH_ALIGN_THOUGHT_ACTION=1`
    - 或其他直接表明 thought-action 對齊已啟用的 env
  - 當前活著的是：
    - `ray::ServeReplica:default:cgc-sglang-openai-gateway`
  - 而不是先前用來對齊 `8001` contract 的那套既有 host2 runtime surface
- 因此目前不能把這輪失敗再歸因成：
  - backend sink 沒打通
  - `8001 -> 30000` phase gap
- 更精確的結論是：
  - `50053` 已足夠讓 sample 開始跑
  - 但當前活著的 gateway runtime 沒有把模型輸出穩定收斂到 host2 所需的 `DISCUSSION + fenced code` 契約

## Updated Decision After 50053 Rerun
- 若目標是「盡快完成一題 demo」，下一步最小且必要的工作已經不再是：
  - API base 對齊
  - 或 backend sink phase 追蹤
- 而是要二選一：
  - 把目前活著的 `50053` gateway runtime 補上 / 恢復 thought-action 對齊能力
  - 或把原本帶有 host2 contract 對齊的正式 gateway surface 恢復回來（例如回到 `8001`）
- 只要這一格不解，agent 會在進入真正 coding loop 前就被：
  - repeated `FormatError`
  截斷，因此還到不了 `verified 500` 的 demo 完整閉環
