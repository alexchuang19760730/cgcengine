# [OPEN] Debug Session: deepep-realchain

## Symptoms
- 目前對外描述不宜宣稱 `DeepEP real-chain` 已正式跑通。
- 最新 `fresh host1 probe` 顯示 `moe_a2a_backend=none`，與「DeepEP real-chain 生效」的敘述衝突。

## Expected
- 若 `DeepEP real-chain` 已正式跑通，fresh runtime/probe 應能提供一致證據，至少不應仍停在 `moe_a2a_backend=none`。

## Hypotheses
1. host1 當前 fresh probe 走的是 fallback/disabled path，根本沒有命中 DeepEP 啟動配置。
2. DeepEP 相關 contract/profile 已存在，但 gateway/backend restart 使用的不是同一份 profile binding，因此 fresh probe 觀測到 `none`。
3. 文檔/報告中的 DeepEP PASS 來自較早的 gate proof 或 wiring evidence，不等於當前 fresh runtime real-chain。
4. host1 probe 顯示的 `moe_a2a_backend` 只反映 gateway 對外暴露配置，未必等於 backend 內部真正 dispatch path。

## Plan
- 先讀現有 fresh probe、restart 證據、gate/report 與 profile/contract 定義，確認目前口徑衝突點。
- 在不改業務邏輯前提下，釐清 `moe_a2a_backend=none` 的來源與是否存在另一條 runtime 證據能證明 DeepEP real-chain。
- 若證據確認只是配置/啟動口徑不一致，再做最小插樁或最小修補。

## Current Status
- Step 1-3: hypotheses established, no business logic modified in this debug session.
- Evidence update (2026-07-02):
- `fresh host1 probe` no longer reports the helper-forced disabled path. Current `/health` exposes `moe_a2a_backend=deepep`, non-empty `profile_settings_path`, expected binding keys, and `task_type_contract_validation.status=PASS`.
- Confirmed root causes that were fixed:
- 1. `restart_host1_swebench_gateway.py` had hard-coded `CGC_MOE_A2A_BACKEND=none` and omitted `CGC_PROFILE_SETTINGS_PATH`.
- 2. The selected host2 benchmark profile bundle was missing `task_type_contract_ref` across profile / system manifest / bootstrap layers, causing fail-fast validation before startup.
- 3. DeepEP import in gateway/replica startup needed `CUDA_HOME` / `CUDA_PATH` available in the host1 launch env and Ray runtime env.
- New first blocker after the fresh-probe fix:
- `50053 /health` is now stable and reflects the intended DeepEP contract, but backend `30000` is still down, so `/v1/models` and `/v1/chat/completions` remain `backend_unavailable`.
