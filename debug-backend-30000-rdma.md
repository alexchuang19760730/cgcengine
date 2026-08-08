# [OPEN] Debug Session: backend-30000-rdma

## Symptoms
- `fresh host1 real-chain` 已顯示 `requested_dispatch_backend=deepep` 與 `freshness_guard=PASS`。
- 但同一條鏈上的 `backend_available=false`，使 `rdma_contract=FAIL` 且 `deepep_real_chain_gate=FAIL`。
- 目前最短主線是回到 `30000 backend` 未起來的第一錯。

## Expected
- `30000 backend` 可穩定 bind，讓 `fresh host1 real-chain` 在同一時窗內觀測到 `backend_available=true`。
- 在此基礎上，`rdma_contract` 才能進入真正 runtime PASS/FAIL 驗證，而不是被 backend availability 前置阻斷。

## Hypotheses
1. `30000 backend` 根本沒成功啟動，第一錯仍在 scheduler/model-load 鏈，`backend_available=false` 只是結果。
2. backend 曾短暫啟動，但被 import/runtime_env/RDMA 依賴缺口打掉，因此 fresh probe 只看到 `connection refused`。
3. `50053` 的 backend health 檢查地址、端口或檢查時窗不對，造成 false negative。
4. `fresh host1 real-chain` 消費到的 backend 現場和 `m76/upkg21` 內部 runtime evidence 不同時窗，release-facing evidence 落後於真實現場。

## Evidence
- 假設 1：成立。`host1` 的 debug log 顯示 gateway 在同一時窗依序探測 `http://127.0.0.1:30000/health`、`/model_info`、`/v1/models`，全部 `Connection refused`，不是 health 口徑偏差。
- 假設 2：部分成立。backend 啟動鏈已走到 `ray_scheduler_get_info_begin`，但 scheduler actor 在 creation task 內死亡，未達可對外提供 `/health` 的階段。
- 假設 3：不成立。probe 直接打 `127.0.0.1:30000`，且三個候選 URL 全部拒連，沒有觀測到錯端口或錯 URL。
- 假設 4：不成立。新的 `B` 插樁已把同一輪重啟中的 scheduler creation failure 收回來，與 `fresh host1 real-chain` 現場一致。

## Current Root Cause
- `backend_available=false` 的根因不是 RDMA，也不是 gateway。
- 第一錯仍在 `30000 backend` 的 scheduler/model-load 鏈：
  - `ActorDiedError` during `ray.get(get_info)`
  - inner exception:
    - `ValueError: No module or parameter named 'model.layers.0.self_attn.attn_sink' in TransformersMoEForCausalLM.`
- 也就是說，`30000` 未 bind 仍然是 `DeepSeek-V4` 權重映射缺口，新的第一錯已經從 `.attn` 前移到 `.self_attn.attn_sink`。
