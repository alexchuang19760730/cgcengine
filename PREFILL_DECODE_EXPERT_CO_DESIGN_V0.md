# Prefill/Decode Expert Co-Design v0

## Scope

- 目標：先定義 `Prefill` / `Decode` 共用 expert data plane 的最小語義，避免後續做 `ExpertPlex` / `P&D` 分離時返工。
- 本文只定 5 件事：
  - `slot ownership`
  - `prefetch 權限`
  - `decode 保護線`
  - `eviction 規則`
  - `telemetry 指標`
- 本文不展開：
  - 進程拆分實作
  - IPC / Mach / mmap 細節
  - Metal kernel 細節
  - 具體 scheduler 演算法優化

## 背景

- 現有 TurboFieldfare 已具備：
  - layer-local routed expert cache
  - `plan / advise / fetch` expert cache plan
  - prefill tile 級別的 planned fetch 與 I/O overlap
- 最新 long steady-state 對照顯示：
  - 單純增加 template bridge cached tokens，不能保證 `prefill_ms / TTFT` 下降
  - 下一刀要從 `expert residency / prefetch / decode protection` 的共同設計切入

參考：
- [turbofieldfare_multiturn_long_steady_compare_20260804_152037_bridge_prime.md](file:///Users/alexchuang/Documents/flashkv0516/var/colibri_metrics/reports/turbofieldfare_multiturn_long_steady_compare_20260804_152037_bridge_prime.md)

## 1. Slot Ownership

### 1.1 目標

讓 `Prefill` 與 `Decode` 共用 expert pool 時，明確知道「哪些 slot 可以動、哪些不能動」。

### 1.2 v0 語義

每個 expert slot 必須帶以下 ownership metadata：

- `owner_phase`: `decode_protected | prefill_transient | shared_resident`
- `owner_session_id`: 可空；`decode_protected` 時必填
- `layer_id`
- `expert_id`
- `last_touch_step`
- `hit_count`
- `sticky_score`

### 1.3 v0 規則

- `decode_protected`
  - 僅供 decode 關鍵路徑使用
  - Prefill 不得主動驅逐
- `prefill_transient`
  - 允許 prefill 預取與覆蓋
  - Decode 可在必要時搶占
- `shared_resident`
  - 代表跨 phase 都有穩定回報的 resident experts
  - 需要比 transient 更高的驅逐門檻

## 2. Prefetch 權限

### 2.1 目標

允許 prefill 提前準備未來高機率 expert，但不能無上限擴張，避免傷到 decode。

### 2.2 v0 原則

- Prefill 只能做 `bounded lookahead prefetch`
- 不允許「全後續 layer 全量預載」
- 預取必須經過 `plan`，不能繞過 slot budget

### 2.3 v0 權限模型

- `Prefill` 可預取：
  - 當前 layer 下一個 tile 的 routed experts
  - 下一層 `L+1` 的高機率 routed experts
- `Prefill` 不可預取：
  - 超出預算的 `L+2+` experts
  - 會覆蓋 `decode_protected` slots 的 expert set

### 2.4 v0 預取來源

Prefill lookahead hint 只允許來自：

- 當前 request 已算出的 router 結果
- 同 session 最近幾輪的 layer-expert 熱分布
- resident expert 命中統計

不允許只靠靜態先驗，直接假設後面 layer 會用同一批 experts。

## 3. Decode 保護線

### 3.1 目標

保證 decode 關鍵路徑的 residency 穩定，不被 prefill lookahead 拖垮。

### 3.2 v0 保護規則

- 每層 expert slots 必須切出保護配額：
  - `decode_reserved_slots`
  - `shared_resident_slots`
  - `prefill_burst_slots`
- Prefill 的預取只能使用：
  - `prefill_burst_slots`
  - 沒被 decode 保護的 `shared_resident_slots`

### 3.3 v0 建議

- 先做固定比例，而不是動態學習：
  - `decode_reserved_slots >= 50%`
  - `prefill_burst_slots <= 25%`
- 若 decode 發生下列任一情況，立即收緊 prefill 預取：
  - `decode_evict_rate` 升高
  - `decode_io_stall_ms` 升高
  - `decode_hit_rate` 下降

## 4. Eviction 規則

### 4.1 目標

不要再用單一 `LFU/LRU` 規則處理所有 phase；驅逐必須知道 phase 與 residency 價值。

### 4.2 v0 優先級

從最不該保留到最該保留：

1. `prefill_transient` 且低命中
2. `prefill_transient` 且已超出 lookahead 視窗
3. `shared_resident` 但 sticky score 下降
4. `decode_protected`

### 4.3 v0 驅逐原則

- Prefill 只能驅逐：
  - `prefill_transient`
  - 非保護的低價值 `shared_resident`
- Decode 可以驅逐：
  - `prefill_transient`
  - 必要時低 sticky 的 `shared_resident`
- 任何 phase 都不能直接驅逐：
  - `decode_protected`

### 4.4 sticky score

v0 不做複雜模型，只用簡單統計：

- `sticky_score = future_hits / adopts`
- 若某 expert 經常被 prefetch 但很少真正命中，sticky score 應下降
- sticky score 下降的 resident expert，下一輪可降級成 transient

## 5. Telemetry 指標

### 5.1 目標

讓每次預取、命中、驅逐，都能回答一個問題：

`這次 prefill 預取，到底是幫了 decode，還是傷了 decode？`

### 5.2 必備指標

- `prefill_prefetch_requested_experts`
- `prefill_prefetch_loaded_experts`
- `prefill_prefetch_wasted_experts`
- `prefill_prefetch_bytes`
- `prefill_prefetch_hit_later_count`
- `decode_hit_rate`
- `decode_evict_rate`
- `decode_io_stall_ms`
- `decode_protected_slot_pressure`
- `shared_resident_slot_pressure`
- `prefill_burst_slot_pressure`
- `slot_churn_by_phase`
- `ssd_bytes_per_request`
- `ssd_bytes_per_decode_token`

### 5.3 最小對照表

每次 A/B 至少要固定輸出：

- `TTFT`
- `prefill_ms`
- `decode_ms`
- `cached_tokens`
- `prefetch_waste_ratio`
- `decode_hit_rate`
- `decode_evict_rate`
- `decode_io_stall_ms`

## v0 Decision

- 先做 `單進程共享 pool` 原型驗證
- 先驗證 `Prefill lookahead` 是否在不傷 decode 的前提下有效
- 等以上語義與 telemetry 穩定後，再 lower 到：
  - `ExpertPlex`
  - `P&D` 分離 contract
  - `shared expert pool` 的 IPC / transport 實作

## v0 Non-Goals

- 不在這一版決定最終 slot 比例
- 不在這一版決定最終 eviction 公式
- 不在這一版綁定特定進程模型
- 不在這一版承諾多機分布式行為

## 下一步

按順序只做三件事：

1. 在現有 TurboFieldfare runtime 增加 slot ownership metadata 與 telemetry
2. 做 `Prefill L+1 bounded lookahead` 小窗原型
3. 用長回答 steady-state workload 做 `s=5` A/B，先看是否傷 decode
