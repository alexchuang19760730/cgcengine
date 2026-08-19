# CGC UPKG4.0 Embodied Psi0 + RealtimeVLA 全鏈條審計回朔可回放手冊

## 1. 目的

本文件將 `psi0 cloud training/runtime gate -> edge realtimevla inference` 固化為一條可重跑、可追蹤、可審計、可回放的 `UPKG 4.0 embodied` 鏈路。

此鏈路不再只輸出單次 run report，而是正式落地以下 artifact 類型：

- cloud contract
- edge inference contract
- training dataset manifest
- trained model manifest
- teaching / train / infer / audit / replay / trace session
- artifact index
- stage trace
- six-element events
- audit replay bundle

## 2. 執行入口

目前正式入口：

```bash
cgc embodied psi0-train --json
cgc embodied psi0-deploy --json
cgc embodied psi0-realtimevla --json
```

三者分工如下：

- `psi0-train`: 落 `full_weight_manifest / publish_manifest / runtime_contract`，回答 train 側有哪些可交付契約。
- `psi0-deploy`: 落 `deploy_contract / consume_contract / bridge_info`，回答 cloud 到 edge 的交付與接入契約。
- `psi0-realtimevla`: 在 deploy/infer contract 之上，真正跑端側 `realtimevla`，回答實際是否執行成功、走哪條 route、回了什麼結果。

本輪 `UPKG 4.0` 收口後，以上三條正式入口都已重新實跑通過，並確認核心 contract artifact 已一致改成：

- `profile_settings_path`
- `execution_profile_binding_key` 或 `execution_profile_binding_keys`
- `delivery_profile_binding_key`
- `compatible_profile_binding_keys`
- `bootstrap_contract_binding_key` 或 `bootstrap_contract_binding_keys`
- `flow_parameter_contract_binding_key` 或 `flow_parameter_contract_binding_keys`

也就是說，artifact 本身不再平鋪整份 profile descriptor；真正的 profile 定義統一收在 `profile_settings.json`，artifact 只保留可審計的 binding key 引用。

相容的舊入口仍可用：

```bash
python3 temp/misc/run_hostb_psi0_to_local_realtimevla.py
```

但 `temp/misc` 版本現在只是 wrapper，會轉呼叫正式的 `cli/cgc` 實作。

此命令會完成以下動作：

1. 重跑 hostb 上的 `psi0` 最小 training/runtime gate 驗證。
2. 抓回 cloud 端 `contract_manifest / system_execution_manifest / strategy_decision / compatibility_report / distributed_runtime_bootstrap`。
3. 把 cloud contract bundle 推到本地端側。
4. 使用 `cgc run ... --use-omlx` 走本地 `realtimevla` 推理。
5. 在本地 `session_dir/upkg40_embodied/` 下落地完整 `UPKG 4.0 embodied` artifact 集。

## 3. 目錄結構

每次執行都會產生一個 session 目錄：

```text
temp/test/psi0_cloud_to_edge_realtimevla/<timestamp>/
```

重要子目錄：

- `cloud_bundle/`
- `edge_infer/`
- `upkg40_embodied/`

## 4. Cloud 端正式輸入

`cloud_bundle/` 會保留這次 cloud 端 `psi0` 驗證的單一真實來源：

- `summary.json`
- `rank0_runtime_gate_report.json`
- `rank0_contract_manifest.json`
- `rank0_system_execution_manifest.json`
- `rank0_strategy_decision.json`
- `rank0_compatibility_report.json`
- `rank0_distributed_runtime_bootstrap.json`

其中：

- `contract_manifest` 代表本次 cloud component contract。
- `system_execution_manifest` 代表系統級裝配與 artifact 關聯。
- `distributed_runtime_bootstrap` 代表分散式 bootstrap 契約。
- `summary` 代表多 rank 是否都在 `_maybe_wrap_ddp()` 前被 `blocked_before_ddp`。

## 5. UPKG4.0 Embodied Artifact 集

`upkg40_embodied/` 目前會正式輸出：

### 5.1 Contract / Manifest

- `psi0_cloud_training_contract.json`
- `realtime_vla_edge_inference_contract.json`
- `embodied_training_dataset_manifest.json`
- `embodied_trained_model_manifest.json`
- `cloud_ingest_manifest.json`
- `cloud_summary.json`
- `canonical_profile_catalog.json`

### 5.2 Session

- `embodied_teaching_session.json`
- `embodied_inference_session.json`
- `embodied_audit_session.json`
- `embodied_replay_session.json`
- `embodied_trace_session.json`

同時也會提供對齊 `UPKG 3.x / cli/cgc` 風格的 session 命名：

- `psi0_embodied_train_session.json`
- `psi0_embodied_infer_session.json`
- `psi0_embodied_audit_session.json`
- `psi0_embodied_replay_session.json`
- `psi0_embodied_trace_session.json`

### 5.3 Audit / Replay / Trace

- `psi0_realtimevla_audit_replay_bundle.json`
- `replay_anchor.json`
- `embodied_six_element_events.jsonl`
- `embodied_six_element_summary.json`
- `stage_trace.jsonl`
- `artifact_index.json`
- `upkg40_embodied_report.json`
- `upkg40_embodied_summary.json`
- `embodied_parity_report.json`

## 6. Canonical Profile 適配

`upkg40_embodied/canonical_profile_catalog.json` 會把目前這條鏈路可共用的 custom 定義正式收斂成一份可審計 catalog，而不是分散在單一 script 或單一 runtime helper 裡。

目前正式口徑是：

- `canonical_profile_catalog.json` 提供可支援 profile 家族與適配矩陣。
- `profile_settings.json` 把 `canonical_profile_catalog + profile_descriptors + scenario_bindings` 合成單一 profile-setting 結構體。
- `train / deploy / realtimevla` artifact 不再直接攜帶完整 descriptor，而是透過 `profile_settings_path + binding_key` 指回同一份 profile-setting 真源。

目前固定覆蓋四種 canonical execution profile：

- `local_infer`
- `local_train`
- `edge_cloud_infer`
- `edge_cloud_train`

目前 `psi0-realtimevla` 的預設 edge model 已收口為：

- `mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit`

原因不是單純偏好 coder 類模型，而是這個 MLX snapshot 在本機 edge 節點已驗證具備可直接載入的 `safetensors`，因此能讓 `cgc run --use-omlx` 在不額外依賴 `CGC_LOCAL_OMLX_MODEL` override 的情況下，預設就走通 `m4_local -> omlx_mlx_lm -> local_execution=true`。

對應原則如下：

- `psi0_cloud_training_contract`、`cloud_summary`、`embodied_training_dataset_manifest` 以 `edge_cloud_train` 為主 execution profile。
- `realtime_vla_edge_inference_contract`、`edge_inference_result`、`psi0_embodied_infer_session` 以 `local_infer` 為主 execution profile。
- `cloud_ingest_manifest`、`replay_anchor`、cloud-to-edge handoff 類 artifact 以 `edge_cloud_infer` delivery profile 表達 publish/deploy/consume 約束。
- train/audit/replay/trace 類 session 同時保留 `canonical_execution_profiles_supported`，確保同一套 schema 可延伸到後續 `psi0-train / psi0-deploy`。

也就是說，這條 `psi0-realtimevla` 現有產品鏈雖然只實跑 `edge_cloud_train -> edge_cloud_infer -> local_infer`，但 artifact schema 已同時保留 `local_train` 的 canonical 定義，後續 train/deploy CLI 可直接復用，不需再發明第三套 custom 命名。

### 6.1 Bootstrap 與流程參數契約

`UPKG 4.0` 這一輪收口後，`bootstrap` 與不同 runtime mode 的定製化流程參數不再只是 helper 或命令列約定，而是正式落在 profile-setting 與 contract artifact 的雙層結構中：

- `profile_settings.json` 內保留完整 `bootstrap_contract_descriptors / flow_parameter_contract_descriptors`，作為 schema 級註冊表。
- 各 contract / manifest / session artifact 只保留對應的 binding key，不再重複展開同一份 descriptor。

目前至少有以下對應：

- `psi0_runtime_contract.json` 以 `profile_settings_path + execution/delivery/compatible/bootstrap/flow binding keys` 表達 `local_train / edge_cloud_train / edge_cloud_infer` 的契約關係。
- `psi0_deploy_contract.json` 以 binding keys 表達 `source / delivery / target` 三段的 bootstrap 與 flow parameter 關係。
- `realtime_vla_consume_contract.json` 與 `realtime_vla_edge_inference_contract.json` 以 binding keys 表達 `edge_cloud_infer + local_infer` 的 bridge handoff 與本地 runtime consume。
- 若需要查看 descriptor 明細，唯一真源是同 session 下的 `profile_settings.json`，而不是個別 artifact 的重複拷貝。

這代表下列資訊都已上升為正式契約欄位或 profile-setting 註冊欄位：

- `distributed_runtime_bootstrap_path`
- `training_stage_scope`
- `distributed_backend`
- `delivery_channel`
- `transport_strategy`
- `selected_route`
- `selected_backend_family`
- `edge_model`

因此目前可以明確說：

- 描述性產出物的主幹已收口完成。
- `local_infer / local_train / edge_cloud_infer / edge_cloud_train` 的 bootstrap 與流程參數約束都已有正式契約落點。
- 但 `full_weight_manifest / deploy_contract / consume_contract` 目前仍可能是 `contract_only` 物化，並不等於真實 full-weight 已完整下發。

## 7. 成功條件

本鏈路的 PASS 條件不是「edge 端有回應就算成功」，而是：

1. cloud 端 `psi0` contract bundle 成功抓回。
2. cloud 端 `blocked_before_ddp` 證據存在。
3. edge 端 `realtimevla` 必須走本地 route。
4. `selected_route = m4_local`
5. `selected_backend` 必須是 `omlx_*`
6. `local_execution = true`
7. `UPKG 4.0 embodied` session / contract / trace artifact 全部落地。

目前預設 PASS 口徑對應的 edge model 為：

- `mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit`

也就是說，若 edge 端 fallback 到 cloud bridge，不會被記成這條鏈路的最終 PASS。

補充說明：

- 這裡的 `PASS` 代表描述性 artifact、契約層與端側本地 infer 路徑已完成收口與驗證。
- 這裡的 `PASS` 不代表 `psi0 Stage1-3` 真實 full training 已完成，也不代表真 full-weight 已完成端側部署。

## 8. 審計入口

若要做審計，優先看：

- `psi0_embodied_audit_session.json`
- `psi0_realtimevla_audit_replay_bundle.json`
- `artifact_index.json`
- `upkg40_embodied_report.json`

這些檔案已把審計所需的主要 path 收斂成單一入口。

## 9. 回放入口

若要做回放，優先看：

- `psi0_embodied_replay_session.json`
- `replay_anchor.json`
- `edge_prompt.txt`
- `edge_infer/route_decision.json`
- `edge_infer/run_report.json`

`replay_anchor.json` 提供了最小可回放集合，包含：

- prompt
- cloud summary
- edge inference result
- route decision
- local infer evidence path

## 10. Trace 入口

若要做 trace，優先看：

- `psi0_embodied_trace_session.json`
- `stage_trace.jsonl`
- `embodied_six_element_events.jsonl`
- `rank0_runtime_gate_report.json`
- `edge_infer/run_report.json`

這幾個檔案能串起：

- cloud distributed bootstrap / contract materialization
- strategy blocking
- cloud-to-edge handoff
- edge route decision
- local runtime evidence

## 11. 與 UPKG 3.x 的關係

這條 `UPKG 4.0 embodied` 鏈路是以 `UPKG 3.x` 與 `cli/cgc` 的 session 習慣為基礎擴充而來：

- 延續 `train / infer / audit / replay / trace` session 形狀
- 延續 `artifact_index + stage_trace + summary + report` 的收斂方式
- 但把語義提升為具身場景：
  - cloud training contract
  - edge inference contract
  - embodied dataset / model manifest
  - cloud-to-edge replay anchor

## 12. 推薦讀取順序

第一次看單次 session 時，建議順序如下：

1. `upkg40_embodied_summary.json`
2. `upkg40_embodied_report.json`
3. `psi0_embodied_train_session.json`
4. `psi0_embodied_infer_session.json`
5. `psi0_embodied_audit_session.json`
6. `psi0_embodied_replay_session.json`
7. `psi0_embodied_trace_session.json`
8. `artifact_index.json`

## 13. 當前已驗證狀態

目前已驗證：

- hostb cloud 端 `psi0` 可以重跑並維持 `blocked_before_ddp` 契約證據。
- 本地端 `realtimevla` 可使用 `omlx` 本地推理成功回應。
- `UPKG 4.0 embodied` 全鏈條 artifact 集可一次性落地。
- artifact 已可供後續審計、回朔、比對與回放。
