[OPEN] Debug Session: cloud-bridge-swe500

## Symptoms
- `python3 app/cli/cgc.py model swe-verified --yes --json` 已成功生成 session artifact 並啟動遠端 `SWE-agent`
- `remote_launch_manifest.json` 顯示 `PASS`
- `remote_swebench_score_summary.json` 仍為 `state=running`、`trajectory_count=0`、`submitted_count=0`
- head host `localhost:8000/v1/models` 可正常回應
- gateway log 出現 `BadGatewayError`、`cloud_call_failed`、`0 bytes read on a total of 4 expected bytes`
- 雲側目標為 `172.30.132.117:50052`

## Expected
- `swe verified 500` 可持續產生 trajectory / submitted artifact，最終可收斂到可判定的 score / pass boundary

## Hypotheses
1. `172.30.132.117:50052` 上的 cloud service 未啟動、崩潰，或只短暫接受連線後立即關閉。
2. head host 到 `172.30.132.117:50052` 的 TCP 連線可建立，但 protocol framing 與 gateway 預期不一致，導致讀 header 時回 `0 bytes read on a total of 4 expected bytes`。
3. head host 上 `cgc.py serve --port 8000` 的 cloud target 配置錯誤，實際連到錯的 IP/port 或 stale endpoint。
4. `SWE-agent` 請求 payload 觸發了 cloud side 特定路徑錯誤，導致健康檢查能過但 inference call 失敗。
5. 遠端 benchmark 雖然啟動成功，但 session 使用的 env / launch context 與手工 probe 不一致，造成只有 swebench 流程會命中壞配置。

## Evidence Plan
- 先比對本地 session artifact、head host log、cloud target 連線狀態
- 只加最小 instrumentation，定位是 connect failure、protocol mismatch、還是 server-side crash
- 確認根因後再做最小修補

## 2026-06-28 Canonical Path Checkpoint
- `model_swe_verified_session.json` 已確認 `launch_plan.env.CGC_CLUSTER_NFS_MINICPM5_GGUF=/nfs/embodied/minicpm5/MiniCPM5-1B-Q4_K_M.gguf`
- 舊的 `run_cgc_cloud.py` demo listener 已替換為正式 `cloud_socket_server.py -> ray_serve_sglang` gateway
- 目前最新阻塞邊界不再是 canonical path 缺失，而是 head host gateway 下游 `127.0.0.1:30000` backend readiness 未確認
- 下一步直接重取 host1 runtime evidence：`50053/health`、`30000/health`、gateway/backend log、以及既有 `swe-verified` session refresh 結果

## 2026-06-28 HTTP Fallback Checkpoint
- 既有 `swe verified 500` session 已從 `trajectory_count=0` 前進到 `trajectory_count=1`，但最終 `500/500` 幾乎全部為 `exit_error`，尚未形成 `submitted` / `score`
- 新 evidence 顯示 `app/servers/cgc_api_server.py` 仍以 raw socket 直連 `50052`，而正式可用 cloud path 實際是 `50053` 的 Ray Serve HTTP gateway
- 已做最小修補：保留原 `50052` raw socket 路徑，同時加入 `CGC_EDGE_USE_HTTP_GATEWAY=1` 或 socket 失敗後的 HTTP gateway fallback
- host1 `cgc.py serve --port 8000` 已同步到新版本並能回 `/v1/models`
- 目前最新阻塞邊界已收斂為：`50053/health` 持續 `status=starting`，`backend_url=http://127.0.0.1:30000` 未 ready，且 `backend_state.pid`/`launch_started_at` 反覆變化，疑似 backend restart loop

## 2026-06-28 Backend Restart Loop Root Cause
- 透過 `host1_backend_pid_observation.json` 確認 `backend_state.pid` 每約 30 秒輪換一次，`30000` 在 loop 期間始終未 listen
- `backend_exception_tails.json` 已定位同一個重複根因：scheduler 初始化 `ModelRunner` 時在 `torch.cuda._lazy_init()` 報 `RuntimeError: No CUDA GPUs are available`
- 根因不是模型權重路徑缺失，而是 backend 由 `ray::ServeReplica` 啟動時，子進程繼承到空的 `CUDA_VISIBLE_DEVICES`
- 已在 `ray_serve_sglang_gateway.py` 做最小修補：保留現有啟動鏈，只在 launch backend 前用 runtime env snapshot 恢復 `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES` / `CUDA_DEVICE_ORDER`

## 2026-06-28 Remote Divergence Recovery
- 同步修補後發現 host1 runtime code tree 落後於本地 repo；remote 依序缺 `app.shared.*` 與 `cgc_engine.pd.dopd_schema`
- 已最小補齊 gateway 直接依賴鏈，使 host1 `50053` 再次成功啟動，`30000/v1/models` 可列出 `/data/models/DeepSeek-V4-Flash-UD-IQ2`
- 目前 `50053/health` 已回 `status=ok`，表示 gateway 與 backend 基本啟動鏈恢復
- 但 `v1/chat/completions` smoke 仍回 `502` 或長時間掛住，代表當前剩餘阻塞已從「backend 起不來」轉移到「completion path 還未成功」
