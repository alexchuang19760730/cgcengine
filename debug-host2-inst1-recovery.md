# Debug Session: host2-inst1-recovery
- **Status**: [OPEN]
- **Issue**: host2 inst1 曾因 root 滿盤與損壞 `__pycache__` 導致 `6379` / `50053` / `30000` 無法穩定恢復，需驗證 recovery 後 `50053` 的 thought-action alignment 與 demo rerun 是否真正跨過原本的 `FormatError` 前沿。
- **Debug Server**: not-started
- **Log File**: .dbg/trae-debug-log-host2-inst1-recovery.ndjson

## Reproduction Steps
1. 在本地執行 `python3 temp/misc/start_host2_inst1_direct.py`
2. 於 host2 檢查 `6379`、`50053`、`30000` listener 與 `/health`、`/v1/models`
3. 讀取 `/data2/ray/inst1/session_latest/logs/serve/controller_*.log`
4. 讀取 `/root/flashkv0516/cloud-deepseek-phase-a-inst1.log`

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | Ray head 已起，但 Serve replica 在 backend readiness gate 失敗 | High | Low | Confirmed |
| B | 子進程仍繼承到舊的 temp path，導致初始化中途崩掉 | High | Low | Rejected |
| C | token budget 改動未真正帶進 remote runtime，仍走舊的 256 限制 | Medium | Medium | Rejected |
| D | ServeController/Proxy 綁到錯的 session 或 namespace | Medium | Medium | Inconclusive |
| E | `4096` context 已生效，但 DSV4 `swa_full_tokens_ratio=0.1` 讓 SWA pool 只有 `512`，首筆長 prompt 在 hybrid SWA admission 被 `NO_TOKEN` 擋住 | High | Low | Confirmed |
| F | 最新 `OpenAIException - Connection error` 的直接前因是 `inst1` raylet 非預期終止，而 `/data2` 98% 滿盤與舊 session 的巨量 `worker-*.err` 導致的持續磁碟壓力是最小可驗證 root cause | High | Low | Confirmed |

## Log Evidence
- `start_host2_inst1_direct.py` 最新回報：`start_ray_head.exit_code=0`、`ray_head_listener_probe.listener=true`、`start_instance.exit_code=0`，但 `health_probe` 回 `HTTP 503`
- `cloud-deepseek-phase-a-inst1.log` 顯示 `server_args.context_length=1024`、`max_total_tokens=1024`、`chunked_prefill_size=1024`
- 同一份 cloud log 顯示 scheduler actor 初始化失敗：
  - `EOFError: marshal data too short`
  - `RuntimeError: Scheduler actor failed to initialize`
  - 後續 `OSError: [Errno 28] No space left on device`
- host2 磁碟狀態：
  - `/` = `99G used / 0 avail / 100%`
  - `/data2` 仍有 `71G` 可用
- `ray_serve_sglang_gateway.py` 目前 backend log 固定寫到 `REPO_ROOT/logs/ray_serve_sglang_backend.log`，仍落在 `/root/...`
- 現場修復後：
  - 明確刪除了 stale `/data/ray/inst1`，root 使用率從 `100%` 降到 `49%`
  - 再次執行 `start_host2_inst1_direct.py` 後，`50053 /health = 200`、`30000 /v1/models = 200`
- recovery 後 smoke：
  - `50053 /v1/chat/completions` 回 `DISCUSSION + ```bash`
  - 不再出現 `tool_call / DSML / <|command|> / <execute>`
- recovery 後 demo rerun：
  - 最新 sample log `swebench_host2_live_backend_sink_20260711T010943.log`
  - 已進入 agent 主流程並產生 `MODEL INPUT`
  - 結尾不是 `exit_format`，而是 `exit_context`
- 進一步將 `inst1` context 拉到 `4096` 後：
  - `/server_info` 與 `internal_states[0].memory_usage.token_capacity` 均顯示 `4096`
  - 最新 active request `rid=771cffc328e74ac4a74be4ba3aa3bd60` 在 TP3 worker trace 顯示：
    - `prefill_add_one_req_result ... result='NO_TOKEN' extend_input_len=3035 max_new_tokens=767`
    - `rem_total_tokens=4096`，但 `rem_swa_tokens=512`
  - `pool_configurator.py` 的 DSV4 邏輯顯示 `swa_tokens = int(full_token * swa_ratio)`，而 DeepSeek V4 hook 預設把 `swa_full_tokens_ratio` 壓成 `0.1`
  - 由於 SWA admission 需要約 `extend_input_len + page_size ~= 3291` 的 SWA budget，`512` 明顯不足，因此 request 在 scheduler 入口反覆落入 `prefill_add_one_req_no_token -> prefill_return_none_no_runnable_requests`
- 提高 `--swa-full-tokens-ratio 0.9` 並再次 recovery 後：
  - `cloud-deepseek-phase-a-inst1.log` 顯示：
    - `DSV4 pool sizes: full=4096, swa=3584`
    - `Memory pool resolve post-constraint: constrained=4096 ... full=4096 swa=3584`
  - 最新 active request 已能從 `prefill_add_one_req_result ... result='CONTINUE'` 走到 `process_batch_result` 與 `send_output_to_detokenizer`
- 重新對齊最新 rerun 失敗現場：
  - `swebench_host2_live_backend_sink_20260711T074101.log` 在兩次 `FormatError` requery 後，最終報 `InternalServerError: OpenAIException - Connection error.`
  - `/data2/ray/inst1/session_2026-07-11_07-38-41_785147_905803/logs/dashboard_agent.log` 顯示：
    - `2026-07-11 07:47:41 ... Raylet is terminated. Termination is unexpected.`
  - 同一個 latest session 的 `raylet.err` 在終止前持續重複：
    - `file_system_monitor.cc:116: ... is over 95% full`
  - host2 `df -h /data2` 顯示：
    - `/dev/nvme1n1 984G 913G 21G 98% /data2`
  - `/data2/ray/inst1/session_2026-07-11_01-47-04_508630_800549` 為已非 active 的舊 session，總量 `50G`
    - 幾乎全部來自 `logs/`
    - 其中四個 `worker-*.err` 各約 `13G`
  - 當前 `ss -ltnp` 已無 `6379` / `50053` / `30000` listener，表示 `inst1` crash 後未留存 active service
- 針對已非 active 的舊 session 做最小清理後：
  - 僅刪除 `/data2/ray/inst1/session_2026-07-11_01-47-04_508630_800549/logs/` 下四個各約 `13G` 的 `worker-*.err`
  - `/data2` 使用率從 `98%` 降到 `93%`
  - 可用空間從 `21G` 回升到 `70G`
  - 該舊 session 從 `50G` 降到約 `50M`
- 清理後新的 recovery/session 狀態：
  - `inst1` 自動進入新 session：`/data2/ray/inst1/session_2026-07-11_07-56-09_205873_916736`
  - `6379` listener 恢復，`50053` ProxyActor 與 `30000` `ray::ServeRepli` 皆存在
  - `30000/model_info` 已可穩定返回 model metadata
  - Proxy log 顯示在 `07:58:10` 之後 replicas 已更新為 `1 total (+1 added, -0 removed)`，`GET /` 可回 `200`
- cleanup 後的 live request 行為：
  - 最新 gateway/proxy 仍可見大量 client-side timeout/cancel (`499`)，但不再是整個 `inst1` crash
  - `worker-e7b0...918690.err` 顯示：
    - `POST /v1/chat/completions` 已能完成並返回 `200`
    - 一次 request `rid='8b437d3aa7e346e0ab7c4cc75663640e'` 的 `e2e_latency=73.44s`
  - 最新 TP worker traces 顯示 request `rid='0a57f3c7e47f4f35b40fc0ecbef97e02'` 已進入 generation/decode，`forward_pass_id=112/113` 持續推進
- 與背景中的同題 rerun 對齊：
  - `/root/flashkv0516/swebench_host2_live_backend_sink_20260711T074101.log` 在本輪清理/recovery 後繼續前進
  - 已從先前停在 `STEP 1` / repeated connection errors，推進到：
    - `STEP 4`：先拿到一次正常 `DISCUSSION + fenced command`，但下一次 response 仍出現 `FormatError`
    - `STEP 6`：requery 後再次拿到乾淨 `DISCUSSION + fenced command`
    - `STEP 7`：繼續產生新的 `MODEL INPUT`
  - 這表示 cleanup 後的 active frontier 已不再是 `raylet unexpected termination`，而是 gateway 仍存在高延遲/偶發格式污染，但 end-to-end run 已能繼續往後走
- 進一步追蹤同一條 rerun 到結束：
  - `swebench_host2_live_backend_sink_20260711T074101.log` 已完整跑完單題 batch，不再中途掉回 transport / connection error
  - 最新終點是：
    - `STEP 8`
    - `Exit due to context window`
    - 隨後進入 autosubmission / environment shutdown
  - 這代表「能不能把同一題跑完一次」這個門檻已經跨過
- 對齊 active backend 與 harness budget：
  - `50053/health` 現在穩定回 `200`，`backend_ready=true`
  - `30000/server_info` 與 `50053/v1/models` 顯示當前 active backend 為：
    - `context_length=16384`
    - `max_total_tokens=32768`
    - `max_req_input_len=16378`
    - `max_running_requests=2`
    - `swa_full_tokens_ratio=0.1`
  - 但最新 `run_full_swebench.runtime.yaml` 仍被寫成：
    - `agent.model.max_input_tokens: 3584`
    - `agent.model.max_output_tokens: 512`
  - 因此最新 `STEP 8 exit_context` 更像是 harness 端 token budget 與 active backend capacity 不對齊，而不是 transport / raylet / scheduler 再次退化
- 已做的最小本地修正：
  - `temp/misc/probe_host2_agent_multiturn_rerun_with_live_backend_sink.py`
    - 預設改為 `CGC_SWEBENCH_MAX_INPUT_TOKENS=14336`
    - 預設改為 `CGC_SWEBENCH_MAX_OUTPUT_TOKENS=1024`
  - `temp/misc/patch_remote_host2_swebench_token_limits.py`
    - 同步把 default token limits 改到 `14336/1024`
- 用新的 `14336/1024` 再跑同題後：
  - `run_full_swebench.runtime.yaml` 已確認寫入：
    - `agent.model.max_input_tokens: 14336`
    - `agent.model.max_output_tokens: 1024`
  - 新 log：`/root/flashkv0516/swebench_host2_live_backend_sink_20260711T090313.log`
  - 最新 rerun **不再出現 `Exit due to context window`**
  - 但 run 沒能穩定跨過 `STEP 1`，而是停在 repeated `FormatError`
    - `STEP 1`
    - `Requerying model after FormatError (1th requery)`
    - `Requerying model after FormatError (2th requery)`
  - 真實壞樣本顯示新的格式污染已不只是 `<tool_control>`，而是：
    - `</think>`
    - 重複 `DISCUSSION`
    - `<details><summary>View reasoning</summary> ... </details>`
  - 這表示 `14336/1024` 已經把 context frontier 往後推掉；最新 blocker 改成 reasoning leakage 導致的 response-format instability
- 針對新樣本的最小本地補丁：
  - `ComputeGraphCompiler-main/Backend/CGC/ray_serve_sglang_gateway.py`
  - `remote_cgc_api_server.py`
  - 在 thought sanitizer 內新增清理：
    - `</?think>`
    - `<details> ... </details>`
    - `</?summary ...>`
    - 行首重複 `DISCUSSION`
  - 兩個檔案均已通過 `python3 -m py_compile`
- 把 sanitizer patch 推到 active gateway 的過程中，發現舊的 `patch_remote_host2_gateway_thought_action_alignment.py` 會用錯 replica env 重新起 Serve：
  - remote source 本身已成功更新，`ray_serve_sglang_gateway.py` 上已可直接驗到新的 cleanup pattern
  - 但 redeploy helper 重新拉起的 replica 一度落到錯誤配置：
    - `tp_size=16`
    - `context_length=8192`
    - `max_total_tokens=512`
    - `swa_full_tokens_ratio=0.1`
  - 結果把 active `50053/30000` 再次拖壞
  - 後續不是再用這支 helper 硬 deploy，而是改走既有穩定路徑 `start_host2_inst1_direct.py`，把 `inst1` 用正確 `tp4/4096/swa=0.9` 配置拉回
- recovery 後的直接驗證：
  - `50053/health = 200`
  - `50053/v1/models.max_model_len = 4096`
  - `30000/server_info.context_length = 4096`
  - 新 session: `/data2/ray/inst1/session_2026-07-11_09-30-14_141900_956243`
- 用已上線 sanitizer 再跑同題 `host2_live_backend_sink_20260711T093136`：
  - 不再立即出現 repeated `FormatError`
  - 也沒有 `Exit due to context window`
  - 但 rerun 在 probe 的 `480s` poll budget 內仍只停在：
    - `MODEL INPUT`
    - `STEP 1`
  - 最新 worker trace 顯示 request `rid=0e17565273fb4b58a4b89ddfb1d037d8` 並沒有卡在 admission，而是一直在 decode/generation：
    - `process_batch_result`
    - `send_output_to_detokenizer`
    - `forward_batch_generation_*`
  - 這表示 sanitizer 已把「立刻格式炸掉」的 frontier 往後推開；新的前沿更接近 **首個 non-streaming response 極慢 / 未在 poll budget 內完成**
- probe 觀測面的小修正：
  - `temp/misc/probe_host2_agent_multiturn_rerun_with_live_backend_sink.py`
    - 優先跟 `inst1 session_latest` 的 `proxy_*.log` / 最新 `worker-*.err`
    - 當 `/root/flashkv0516/logs/host2_backend_30000_scoped_restart.log` 不存在時，自動 fallback 到最新 worker log
  - 避免再被舊的 `inst3` 路徑與不存在的 backend sink 誤導
- 針對「首個 non-streaming response 很慢」做更窄的 direct replay 後，最新定位改變：
  - 用 `host2_live_backend_sink_20260711T093136` 的同一題第一輪 prompt 直打 `50053/v1/chat/completions`
  - 直接重放證據：
    - `max_tokens=128`（舊 patch 前）約 `165.7s` 才回，內容是 `<details><summary>...`
    - 補上 `_request_expects_thought_action()` 對 SWE-agent runtime template 的判定後，`max_tokens=128` / `256` 仍會回不同 tool-call 方言：
      - `<tool_calls><invoke name="Bash">...`
      - `<ToolCall>`
      - `<AgentInput>`
      - `<|tool_call|>/bin/bash -c "..."<|tool_call|>`
      - `<tool_call><file_search>...`
  - 結論：
    - 問題不只是 latency，而是 **gateway request side 沒像 `remote_cgc_api_server.py` 一樣先補強 thought-action instruction**
    - 模型在首輪會自由切換多種工具標記方言，後處理 parser 只能被動追著補
- 本輪新增的 gateway patch（本地與 remote source 已同步）：
  - `ComputeGraphCompiler-main/Backend/CGC/ray_serve_sglang_gateway.py`
    1. `_request_expects_thought_action()`：
       - 新增辨識 SWE-agent runtime 的 `RESPONSE FORMAT + DISCUSSION + fenced command` 模板
    2. `sanitize_thought_text()`：
       - 新增清理 `</?ToolCall>`、`</?AgentInput>`
    3. `_normalize_thought_action_text()`：
       - 新增解析：
         - `<invoke name="Bash"> ... string_template="...">`
         - `<|tool_call|> ... <|tool_call|>`
       - 若命令包在 `/bin/bash -c "..."`，會抽出內層真正 command
    4. `prepare_request()`：
       - 對 thought-action 請求補上與 `remote_cgc_api_server.py` 一致的 `CRITICAL INSTRUCTION`
       - 明確禁止 `<tool_calls>`, `<tool_call>`, `<ToolCall>`, `<AgentInput>`, `<file_search>` 等工具標記，要求直接輸出 `DISCUSSION + ```bash`
- request-side injection 生效後的 direct replay：
  - `max_tokens=256` 已能穩定回：
    - `DISCUSSION`
    - 乾淨 ` ```bash ` command block
    - 不再帶 tool markup
  - 代表最新修正不只是把殘留字元洗掉，而是把模型首輪輸出重新拉回 thought-action 軌道
- 在此基礎上重跑同題 probe：
  - 新 log：`/root/flashkv0516/swebench_host2_live_backend_sink_20260711T102813.log`
  - 這次：
    - `STEP 1` 無 `FormatError`
    - `STEP 2` 也成功前進
    - 到 `STEP 3` 才出現 `Exit due to context window`
  - 這說明：
    - `STEP 1 repeated FormatError` 這個 frontier 已被實質打掉
    - 最新主 blocker 再次收斂成較晚階段的 `exit_context`
- 對 `STEP 3 exit_context` 做 message growth / prompt growth 對齊後：
  - latest probe nested artifact 顯示：
    - `model_input_count=4`
    - `backend_request_count=2`
    - `stop_reason=reached_model_input_4`
  - 也就是說，SWE-agent trace 已經組到第 4 次 `MODEL INPUT`，但 backend 真正只收到了前 2 次 request；`STEP 3` 的 `exit_context` 更像是 harness 本地 prompt/context gate 先觸發，而不是 backend 第 3 次請求失敗
- 最新 trace / worker 證據：
  - trace:
    - `MODEL INPUT #1`（對應 `STEP 1`）prompt section 約 `8869 chars`
    - `MODEL INPUT #2`（對應 `STEP 2`）prompt section只有：
      - `OBSERVATION:\n./astropy/io/ascii/html.py`
      - 約 `100 chars`
    - `MODEL INPUT #3`（對應 `STEP 3`）prompt section直接變成整份 `astropy/io/ascii/html.py` 內容
      - 約 `17741 chars`
      - `480` 行
  - worker:
    - 第 1 次 backend request `prompt_tokens=3184`
    - 第 2 次 backend request `prompt_tokens=3296`
    - 只增加了 `112` tokens
  - 結論：
    - 真正把 `STEP 3` 推爆的不是前兩輪 message 累積太多，而是 **`STEP 2` 用 `cat astropy/io/ascii/html.py` 讀了整份檔案，讓下一輪 observation/history 突然暴增**
    - 與其繼續抬 swebench token budget，更合理的下一步是 **壓 observation/history 增長**，尤其避免大檔全文輸出
- 依據上述證據，本輪做的最小 prompt-side 修正：
  - 在 gateway / local request-side `CRITICAL INSTRUCTION` 補充：
    - `Keep command output compact. Avoid dumping whole files when a narrow read is enough.`
    - `Prefer rg, grep, sed -n, head, tail, or targeted file paths over cat on large files.`
  - 目標不是改 parser，而是直接把模型第一時間導向「窄讀」命令，減少後續 `OBSERVATION` 膨脹
- 修正後的 direct replay（同題第一輪、`max_tokens=256`）已驗證新指令開始生效：
  - 回覆仍是乾淨 `DISCUSSION + ```bash`
  - 第一條 command 已不再是 `cat astropy/io/ascii/html.py`
  - 改成：
    - `cd /testbed`
    - `find . -path "*/io/ascii/html.py" -o -path "*/io/ascii/html*.py" | head -20`
  - 這表示 request-side prompt 約束已經開始把模型往「窄讀」而不是「整份大檔全文輸出」的方向推
- 後續再把 request-side 約束往前收緊（禁止 discussion-only、要求探索時也必須給 read-only command）後：
  - active `inst1` 進入新 session：`/data2/ray/inst1/session_2026-07-11_11-26-25_069941_1032959`
  - 用 direct replay 再測第一輪時，request `rid=c307781e85274fb7b3fd3968127eaea6` 在 worker 顯示：
    - `prompt_tokens=3416`
    - `max_new_tokens=255`
    - `swa_needed=3672`
    - `rem_swa_tokens=3584`
    - `prefill_add_one_req_result -> NO_TOKEN`
  - 這表示 direct replay 這條驗證路徑因為 prompt 本身被加厚，已經再次碰到 `4096` 下的 SWA admission 邊界
  - 但它與真實 SWE-agent full run 仍有差別：
    - full run 先前 `STEP 1` 實測 prompt 約為 `3226` tokens
    - 因此後續驗證應回到 full rerun 本身，而不是再依賴已變胖的 direct replay 作為代理
- 用最新 active gateway 再跑 full probe（suffix `host2_live_backend_sink_20260711T113859`）後：
  - 本地 probe 仍是 Paramiko `stdout.read()` timeout 結束，不可直接視為 backend 結論
  - remote swebench log 顯示：
    - `STEP 1` 後連續三次 `InternalServerError: OpenAIException - Connection error`
    - 沒有走到 `FormatError`
    - 也沒有走到較晚階段的 `exit_context`
  - 同時間 active worker 最新關鍵 request 仍是前面 direct replay 的 `rid=c307781e85274fb7b3fd3968127eaea6`
    - `prompt_tokens=3416`
    - `prefill_add_one_req_result -> NO_TOKEN`
    - `prefill_return_none_no_runnable_requests`
  - 因此這次 full probe **不能當成乾淨的 prompt-side A/B 驗證**：
    - 它落在 `STEP 1 connection error`
    - 而且時間上與已知 direct replay `NO_TOKEN` 污染重疊
  - 下一個最小正確動作應是：
    - 先清掉這個受污染的 active session / stuck request
    - 再在 **不併發 direct replay** 的條件下，單獨重跑 full probe
    - 才能判斷「禁止 discussion-only」這輪 prompt 約束是否真的把主線往前推
- 按上述方式清場後：
  - 舊的 `run-batch` 進程已被清掉
  - active session 切到：`/data2/ray/inst1/session_2026-07-11_11-56-28_518055_1051535`
  - `50053/health = 200`
  - `50053/v1/models.max_model_len = 4096`
  - `30000/server_info` 正常
- 在 **不併發 direct replay** 的條件下重跑乾淨 full probe（suffix `host2_live_backend_sink_20260711T115747`）後：
  - remote swebench log 只到：
    - `MODEL INPUT`
    - `STEP 1`
  - 沒有立刻再掉：
    - `InternalServerError: OpenAIException - Connection error`
    - `FormatError`
    - `Exit due to context window`
  - active worker 對到的真實 request 為：`rid=9004effe2b80438992b14f963a99d0e8`
    - `input_tokens=3268`
    - `max_new_tokens=511`
    - `swa_needed=3524 <= rem_swa_tokens=3584`
    - `prefill_add_one_req_result -> CONTINUE`
    - `run_batch_begin -> forward_batch_generation_enter`
    - 首個 prefill forward 約耗時 `73.124s`
    - 之後 decode 已連續進到 `forward_iter=90+`
    - 持續可見 `process_batch_result` / `send_output_to_detokenizer`
  - 這說明：
    - 這次乾淨 full probe 已排除掉前一輪的 direct replay `NO_TOKEN` 污染
    - 最新前沿不再是 `STEP 1 connection error`
    - 也不再是首輪 request admission 失敗
    - 最新主戰場重新收斂成：**首輪 prefill 很慢，但 request 已經進 generation；接下來要看 SWE-agent 何時收到第一個可解析回覆，以及它之後落到 `FormatError`、`exit_context`，還是能往更後面 step 繼續跑**

## Verification Conclusion
- `A` confirmed：初始失敗點確實是 `ServeReplica` backend readiness gate；其根因是 root 滿盤導致 `.pyc` 損壞與例外寫 log 失敗。
- `B` rejected：這輪不是 `TMPDIR=/data/tmp` 造成的 `No usable temporary directory`；新的失敗點已移到 `.pyc` 損壞與 root 滿盤。
- `C` rejected：runtime 已真正帶入 `1024` token budget，不再是 `256 -> max_new_tokens=0`。
- 當前結論：
  1. `inst1` recovery 已成功
  2. `50053` thought-action alignment 已通過最小 smoke 驗證
  3. demo 主線已跨過原本的 `repeated FormatError` 與 `exit_context` frontier
  4. DSV4 hybrid SWA pool 太小的 blocker 已被 `--swa-full-tokens-ratio 0.9` 打掉
  5. `raylet` 非預期終止的直接高風險條件已透過最小清理顯著緩解，且新的 `07:56` recovery 已成功把 `inst1` 再次拉起
  6. 同題 rerun 已能完整跑完一次；transport / connection error 已不再是主 blocker
  7. 最新 frontier 已轉成：active backend capacity 已放大到 `16384`，但 harness 仍沿用 `3584/512`，導致 run 在 `STEP 8` 回到 `exit_context`
  8. `14336/1024` rerun 已證實最新 `exit_context` 被打掉；context budget 不再是主 blocker
  9. `STEP 1` repeated `FormatError` 的根因已定位為 request-side instruction 缺失 + 多種 tool-call 方言輸出，而不只是單純 latency
  10. active gateway 已補上 detection、request-side instruction injection 與多種 tool-call / markup normalization；direct replay 已驗證首輪可回乾淨 `DISCUSSION + ```bash`
  11. 同題 rerun 已從原本卡在 `STEP 1 FormatError`，推進到 `STEP 3 exit_context`
  12. `STEP 3 exit_context` 的主因已更精確地定位為：`STEP 2` 讀整份 `html.py` 導致下一輪 observation/history 暴增，而不是 backend 前兩輪 prompt token 緩慢累積
  13. 因此下一步優先順序應是「壓 observation/history 膨脹」而不是先繼續抬 swebench token budget；budget 仍可作為次要微調面，但不是主解

## 2026-07-11 Binding Upgrade
- 使用者明確要求：`swe-agent`、`sglang`、`CGC engine` 不應再只靠 ad-hoc probe / prompt 熱補，而應遵守 `state ABI / united pipeline kernel / bootstrap / system profile / profile binding`，且由 `CGC 4D perception matrix` 發起。
- 這一輪已把 shared helper 從「只注入 system prompt」升級成「同時生成 formal request contract metadata」：
  - 檔案：`app/shared/swe_agent_profile.py`
  - 新增：
    - `swe_agent_profile_binding_ref(...)`
    - `swe_agent_system_profile_ref(...)`
    - `swe_agent_system_profile_summary(...)`
    - `apply_swe_agent_request_contract(...)`
  - contract metadata 內容包含：
    - `task_type=repo_debug`
    - `initiator=cgc_4d_perception_matrix`
    - `state_abi=united_pipeline_kernel_v1`
    - `output_contract=discussion_fenced_bash_v1`
    - `profile_binding_ref`
    - `system_profile_ref`
    - `system_profile_summary`
    - `task_type_contract_ref`
- 入口同步：
  - `ComputeGraphCompiler-main/Backend/CGC/ray_serve_sglang_gateway.py`
    - 在 `prepare_request(...)` 內，對 SWE-agent request 除了注入 `system profile`，也會套用 `apply_swe_agent_request_contract(...)`
    - 若來自上游的 `x-cgc-task-type` header 缺失，會從 payload metadata 補寫 `repo_debug` 往下游轉發
    - 會自動把 gateway profile bundle 的 `profile_settings_path / bootstrap_contract_path / system_manifest_path` 帶入 binding metadata
  - `remote_cgc_api_server.py`
    - 在本地 edge server 組 payload 給 `50053` 前，對 SWE-agent request 一併寫入 formal request contract metadata
  - `app/servers/cgc_api_server.py`
    - 同步跟進同一套 shared request contract helper，避免不同 server path 漂移
- 本地驗證：
  - `python3 -m py_compile` 已通過：
    - `app/shared/swe_agent_profile.py`
    - `remote_cgc_api_server.py`
    - `app/servers/cgc_api_server.py`
    - `ComputeGraphCompiler-main/Backend/CGC/ray_serve_sglang_gateway.py`
- remote source 已同步到 `host2:/root/flashkv0516/...`，且遠端 `py_compile` 通過。
- 這一輪尚未做新的 scoped restart / clean full probe；因此 active `inst1` 是否已真正以這版 formal request contract 跨過 `STEP 1 repeated FormatError`，還需要下一個乾淨驗證結果。
- 為了讓下一輪 clean probe 有乾淨證據面，已補充最小 instrumentation：
  - `remote_cgc_api_server.py`
    - `debug-point F:request_received / upstream_sent` 會額外記錄：
      - `is_swe_agent_request`
      - `payload_task_type`
      - `payload_metadata_task_type`
      - `payload_has_profile_binding_ref`
      - `payload_has_system_profile_ref`
  - `ray_serve_sglang_gateway.py`
    - gateway 在 `chat_completions` 收到 request 與 upstream forward 前，會額外記錄：
      - `payload_task_type`
      - `payload_metadata_task_type`
      - `payload_has_profile_binding_ref`
      - `payload_has_system_profile_ref`
      - `forwarded_task_type`
  - 目的不是擴 probe，而是讓下一輪只靠單次 clean probe 就能回答：
    - binding 是否已從 edge path 寫入 payload metadata
    - active gateway 是否真的收到並帶著它往下游轉發
