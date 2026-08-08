# CGC Psi0 Cloud To Edge RealtimeVLA Runbook

## 1. 目的

這份 runbook 固化目前已可重跑的 `psi0 cloud -> edge realtimevla` 驗證流程：

1. 在雲側 `hostb` 重新執行 `psi0` 最小 training/runtime gate。
2. 拉回 `execution_context`、`state_abi`、`strategy_decision`、`compatibility_report`、`contract_manifest`、`system_execution_manifest`、`distributed_runtime_bootstrap`。
3. 將 cloud artifact 打成 edge push bundle。
4. 在端側用本機 `EdgeLocalInferenceRuntime` 走一次 `realtimevla` 推理，留下 `run_report`、`m4_inference_report`、`edge_inference_bridge` 等證據。

這條流程目前的重點不是完成 `psi0` 多卡訓練收斂，而是驗證：

- cloud 端 `psi0` runtime contract 與 manifest 是否正確落地。
- `blocked_before_ddp` 是否能作為正式 cloud artifact 被 edge 端消費。
- edge 端 `realtimevla` 推理鏈是否能接住 cloud bundle 並留下正式證據。

## 2. 目前入口

### 雲側 `psi0`

- launcher：`temp/misc/launch_hostb_psi0_runtime_gate_blocked_ddp.py`
- remote worker：`temp/misc/run_hostb_psi0_runtime_gate_blocked_ddp_remote.py`
- result fetch：`temp/misc/fetch_hostb_psi0_runtime_gate_blocked_ddp_results.py`

### 端側 `realtimevla`

- runtime：`app/edge_engine/local_infer.py`
- CLI 入口：`app/cli/cgc.py run --use-omlx`

### 一次跑完整條鏈路

- orchestration：`temp/misc/run_hostb_psi0_to_local_realtimevla.py`

## 3. 前置條件

### 雲側

- `hostb` 可 SSH 連線。
- `hostb` 上 `/nfs/embodied/repos/Psi0/.venv-psi/bin/python` 可用。
- `hostb` 上 `/root/flashkv0516` 已具備目前 CGC repo 內容。

### 端側

- 本機為 `Darwin + arm64`。
- 本機可執行 `python3 app/cli/cgc.py run --use-omlx`。
- Hugging Face cache 中至少有一個可用 MLX 模型。

目前實測可用的 edge model：

- `mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit`

## 4. 直接執行

在 workspace 根目錄執行：

```bash
python3 temp/misc/run_hostb_psi0_to_local_realtimevla.py
```

這支腳本會依序做四件事：

1. 執行 `launch_hostb_psi0_runtime_gate_blocked_ddp.py`
2. 執行 `fetch_hostb_psi0_runtime_gate_blocked_ddp_results.py`
3. 在本地建立 `cloud_bundle/`
4. 執行：

```bash
python3 app/cli/cgc.py run \
  mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit \
  --use-omlx \
  --prompt "<由 cloud contract 自動生成>" \
  --max-tokens 96 \
  --report-dir <session>/edge_infer \
  --json
```

## 5. 主要輸出

每次執行都會落在：

```text
temp/test/psi0_cloud_to_edge_realtimevla/<timestamp>/
```

其中重要檔案如下：

### Cloud Bundle

- `cloud_bundle/summary.json`
- `cloud_bundle/rank0_runtime_gate_report.json`
- `cloud_bundle/rank0_contract_manifest.json`
- `cloud_bundle/rank0_system_execution_manifest.json`
- `cloud_bundle/rank0_strategy_decision.json`
- `cloud_bundle/rank0_compatibility_report.json`
- `cloud_bundle/rank0_distributed_runtime_bootstrap.json`

### Edge Bundle

- `edge_push_bundle.json`
- `edge_prompt.txt`

### Edge Inference Evidence

- `edge_infer/run_report.json`
- `edge_infer/m4_inference_report.json`
- `edge_infer/edge_inference_bridge.json`
- `edge_infer/route_decision.json`
- `edge_stdout.txt`
- `edge_stderr.txt`

### Top-Level Summary

- `orchestration_report.json`

## 6. 成功判準

### 雲側成功

以下條件需成立：

- `cloud_bundle/rank0_contract_manifest.json` 存在
- `overall_status = BLOCKED`
- `overall_reason = runtime_collective_evidence_required`
- `cloud_bundle/rank0_system_execution_manifest.json` 存在
- `artifact_paths.distributed_runtime_bootstrap` 有值

### 端側成功

以下條件需成立：

- `orchestration_report.json` 中 `edge.status = PASS`
- `edge.local_execution = true`
- `edge.selected_backend = omlx_mlx_lm`
- `edge.evidence_paths.local_infer` 有值
- `edge_infer/edge_inference_bridge.json` 存在

## 7. 目前語義

這條流程目前表達的是：

- cloud 側 `psi0` 已完成 runtime contract materialization。
- 多卡 DDP 入口被 `strategy_decision = blocked_before_ddp` 正式阻斷。
- `distributed_runtime_bootstrap_v1` 已作為正式 artifact 落盤。
- edge 側已能消費這批 cloud artifact，並用 `realtimevla` runtime host 完成一次本地 inference evidence。

這條流程目前不代表：

- `psi0` 已完成正式多卡訓練收斂。
- edge 側已載入 `psi0` 原生動作模型做真實機器人控制推理。

目前 edge 端驗證的是：

- `cloud artifact -> edge bundle -> local inference evidence`

這條 contract/manifest handoff 鏈。

## 8. 常見失敗點

### 8.1 雲側 launcher 卡住

先分段執行：

```bash
python3 temp/misc/launch_hostb_psi0_runtime_gate_blocked_ddp.py
python3 temp/misc/fetch_hostb_psi0_runtime_gate_blocked_ddp_results.py
```

再檢查：

- `temp/misc/hostb_psi0_runtime_gate_blocked_results/summary.json`

### 8.2 本機端側模型不可用

先跑單獨 smoke：

```bash
python3 app/cli/cgc.py run \
  mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit \
  --use-omlx \
  --prompt "請回覆 edge smoke ok" \
  --max-tokens 24 \
  --report-dir temp/test/realtimevla_edge_smoke \
  --json
```

### 8.3 需要更換 edge model

可直接改：

- `temp/misc/run_hostb_psi0_to_local_realtimevla.py` 中的 `EDGE_MODEL`

或先用：

```bash
python3 app/cli/cgc.py config --set-local-omlx-model <your_model>
```

## 9. 後續建議

若要把這條流程進一步產品化，下一步建議：

1. 將 `edge_push_bundle.json` 正式掛進 `UPKG 4.0` artifact 契約。
2. 讓 `hostb psi0` cloud rerun 除了 `blocked_before_ddp` 路徑，也能提供 `single_gpu_trainable` 的可部署 bundle。
3. 補一個 `host1/host2` 版本 orchestration，讓 edge 端不只支援本機 `Darwin/MLX`，也支援遠端 edge runtime host。
