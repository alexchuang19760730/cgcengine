# llama.cpp fork：Expert Bounded Residency 方案 — 2026-08-22 修改結論紀錄

日期：2026-08-22
基準：`temp/llama_roadB/llama.cpp-master`（同 08-14 方案之 fork）
本文檔為 0822 當日所有修改的結論紀錄（延續 `LLAMACPP_EXPERT_BOUNDED_RESIDENCY_FORK_方案.md`）。

## 當日目標
回到 25 t/s + 正確輸出：把 remap 改成 **GPU data dependency**（`ggml_get_rows` from slot table + Metal ordering），徹底消除 CPU timing 依賴，不再靠 fast-hook 賭 timing。

---

## 修改 1：GPU-side remap（submit-ahead race 根治）

### 背景 / 根因（回顧）
- CGC_OA_ASYNC segmented dispatch：submit-ahead 讓 seg[i+1]（FFN）在 CPU hook 寫 remap 之前就被送出。
- GPU 執行 FFN 時讀到的是**上一輪的 remap 值** → 輸出 garbage（Heisenbug，只在高 CPU hook latency 時浮現）。
- CGC_SUBMIT_AFTER=1（把 submit 序列化到 hook 之後）能救正確性但掉到 6.7 t/s。

### 設計
把「CPU 每步寫 per-token remap leaf」改成「GPU 上做查表」：

```
每層新增 slot table tensor: [1, n_expert] I32（expert id -> slot index）
圖中插入: gr = ggml_get_rows(slot_table, selected_experts)   // [1, n_expert_used, n_tokens]
          remap_ids = reshape_2d(gr, n_expert_used, n_tokens) // 命名沿用 ffn_moe_topk_remap
```

- `process_ubatch` 在 **graph submit 之前**用 `ggml_backend_tensor_set_async` 把 CPU slot table 推入 GPU tensor。
- Metal 上 `tensor_set_async` = 排一個 blit command buffer 到同一 MTL queue：
  - 排在**前一步 graph 之後**（不會污染前一步仍在跑的 get_rows）
  - 排在**本步 graph 之前**（本步 FFN 一定讀到本步的 slot 資料）
  - → **queue order 本身就是 data dependency**，無 CPU-GPU sync、無 per-step 序列化、無 mid-graph CPU write。
- hook（expert_cache_on_topk）在 pool 路徑**不再寫 remap leaf**（該 tensor 現在是 GPU 計算結果）。

### 為何 decode 的 slot table 是靜態的
- decode 固定 n_tokens=1、top-k ≤ 8、union ≤ 8 ≤ n_slots(174) → pool 路徑永遠成立。
- slot table 是 per-layer 的全 expert→slot 靜態對映（41 層 × 256 × 4B = 41KB），隨 residency 變動而更新（prewarm / ensure / prefetch 時）。
- prefill（n_tokens>1）不建 slot tensor → 走完整 expert weights。

### 改動檔案
- `src/llama-graph.h`：`llm_graph_params` / `llm_graph_context` 新增 `expert_cache_pool_active`。
- `src/llama-graph.cpp`：
  - `build_moe_ffn`：pool-active decode 改建 slot table + `ggml_get_rows` + reshape（命名 `ffn_moe_slot_table` / `ffn_moe_topk_remap`）；非 pool（L3-B）維持原 host remap leaf。
- `src/llama-context.h`：新增 `mutable std::map<int, ggml_tensor *> cache_slot_tensors`。
- `src/llama-context.cpp`：
  - `graph_params`：傳入 `expert_cache_pool_active`（`llama_expert_cache_pool_active`）。
  - `graph_get_cb`：capture `ffn_moe_slot_table` → `cache_slot_tensors[il]`。
  - `process_ubatch`：fresh build 前 `cache_slot_tensors.clear()`；graph submit 前對每層 `ggml_backend_tensor_set_async` 推 slot table（`ggml_nbytes(st)` = n_expert×4B）。
  - `expert_cache_on_topk`：pool 路徑移除 remap leaf 寫入；L3-B 的 remap 寫入加 `!pool_active` guard（避免誤寫 GPU 計算 tensor）。

### Metal 支援驗證
- `ggml_metal_supports_op`：`GGML_OP_GET_ROWS` 只排除 NVFP4 src0 → I32 可用。
- `kernel_get_rows_i32`（`get_rows_f_t kernel_get_rows_f<int32_t,int32_t>`）存在於 `.metal`。
- CPU 參考 `ggml_compute_forward_get_rows_f32`：`case GGML_TYPE_I32`，`src1[i]` 為 row index、`dst.ne0=src0.ne0` → 語意吻合。
- dispatch：`(ne10, ne11, ne12)` = (n_expert_used, n_tokens, 1)。

### 正確性邊界（已知）
- 冷 miss（當前 union 含尚未 resident 的 expert）：slot=-1 → 該 token 的 FFN 為 garbage。
  - 依賴 prewarm_hot（prefill 熱門專家預載）+ B async prefetch（前步 union）把 miss 壓到 ~1% 以下。
  - 短測 prompt 的 decode union 通常穩定落在 prewarm top-K 內 → 應全正確。

### 測量結果
（待驗證後補：正確性 + t/s vs CGC_SUBMIT_AFTER 6.7 / submit-ahead garbage 23.99）

---

## 修改 2：4GiB 下最低 miss 實證（0822 下半場）

### 動機
用戶要求：pool 只能用 4GiB（8GiB OOM），在此預算下把 miss 壓到最低。

### 實證結果（submit-after 正確路徑，n=40, seed=42, "The capital of France is"）

| 策略 | hit | misses | 輸出 | t/s |
|---|---|---|---|---|
| **prepopulate 0-173（現行，基準）** | **93.3%** | **1418** | 正確 | **7.07** |
| PIN_PROFILE 只存 list（未 fill，no-op） | 93.3% | 1418 | 正確 | 7.46 |
| PIN_PROFILE + pin_fill（hot-set 預載） | 69.8% | 7434 | **錯**（"a city that has been…"） | 4.40 |
| 整檔 13GB 背景 warm thread | — | 1418 | 正確 | 更慢（與 decode 搶 IO） |
| expert 段 5.8GB 同步 warm（LLAMA_EXPERT_CACHE_WARM） | 93.3% | 1418 | 正確 | 7.19（無效） |

### 關鍵發現

1. **路由多樣性極高**：40 步每層路由 **85-157 個 distinct expert**（layer0=157, layer39=128），
   ≈ pool 容量 174 → **working set 幾乎填滿 pool**，miss 有硬性下限。
2. **prepopulate 0-173 已是此預算最佳填充**：路由 ~50% 落在低 ID（0-173，prepopulate 覆蓋）＋
   ~50% 高 ID（174-255，必 miss）。hit 93.3% 已接近理論值。
3. **hot-set pin_fill 反效果**（69.8% + 輸出錯）：
   - pin_fill 用 ensure_slot 灌熱門專家 → **驅逐 prepopulate 的低 ID 專家**（路由也需要它們）。
   - 被驅逐專家再次被路由 → non-resident → GPU get_rows OOB → garbage logits → **路由跟著變** → 級聯錯亂。
   - 已完整 revert（llama.cpp 呼叫 + pin_fill 函式 + header 宣告）。
4. **page-cache warm 無效（硬體限制）**：
   - 機器空閒 RAM 僅 ~8GB，13GB GGUF 無法整檔常駐；5.8GB expert 段同步 warm 也在 decode 前被逐出。
   - warm 只把冷讀變微熱讀，miss 數不變；背景 warm 反與 decode 搶 disk IO。
5. **冷讀是 7 t/s 天花板**：`pread_usec=4.05s` ≈ 104ms/step ≈ step 時間的 ~75%。
   1418 misses × 3 segments × ~1ms 冷讀 = 主瓶頸，**非 GPU（cb 僅 ~1.7ms/split）**。

### 結論（4GiB）
- miss 率下限 ≈ **93.3% hit / 1418 misses（40 步）**，prepopulate 0-173 為最佳填充。
- 唯一能再降 miss 成本的途徑是**把冷讀與 GPU compute 重疊**（async prefetch），
  但路由 drift 無法預測（~50% 重複 = 時間局部性弱），submit-after 的 B-prefetch 因
  hook 已 ensure（全 resident → drop）而 dead（prefetch=0/0）。

---

## 修改 3：double-compute 根因實證（adaptive re-run 的 blocker）

### 問題
adaptive fast path 的 dirty step re-run 一直輸出 garbage。先前假設是 reset/realloc 不當，
本段系統性測試證明是**更根本的 double-compute 缺陷**。

### 實驗（CGC_DOUBLE_RUN：NO_ADAPTIVE 正確路徑內重算同一 graph 兩次）

| 變體 | after1 (compute1) | after2 (compute2) | 結論 |
|---|---|---|---|
| reset + realloc（舊） | **NaN** | NaN | realloc 在已 compute 的 graph 上 orphan logits buffer |
| 不 reset 直接重算（Metal） | sum=203.8（對） | **sum=-709.3（錯）** | 同一 graph 重算 ≠ 相同結果 |
| 不 reset（CPU ngl=0） | sum=1752（對） | sum=1353（接近但不一致） | **非 Metal 專屬**，是 graph 層級缺陷 |
| 不 reset + 不 set_inputs（CPU） | sum=1752 | 直接崩潰（13.7GB 峰值 RAM） | set_inputs 之外還有狀態累積 |

### 根因判定
- **同一 graph 物件二次 graph_compute 在本 codebase 不 idempotent**（Metal 與 CPU 皆然）。
- 高度懷疑 KV cache / 圖內 intermediate 在第一次 compute 後殘留狀態（二次 compute 讀到
  已寫入的 KV / 非零中間值）；`set_inputs` 只重寫 input tensor，未重置這些累積狀態。
- 正確 re-run 需要 **KV rollback 到 step 邊界 + fresh graph build**（昂貴且仍有風險），
  而非「同一 graph 重算」。

### 影響
- adaptive fast path（frozen table + verify-only + dirty re-run）的 re-run 骨幹不可靠。
- 加上 dirty rate 高（路由多樣性 → 每步都有 non-resident）→ **fast path 目前無法同時滿足
  正確 + 快速**。

---

## 修改 4：semaphore wake（sched_yield poll → completion-handler semaphore）

### 問題
submit-after 的正確路徑在 poll 階段浪費大量時間：`while (cgc_done < target) sched_yield()`
的旋轉在 macOS 上每 segment 約 0.5–1ms，41 segment/step → 每 step 白白燒掉 ~20–23ms。

### 改動
- `ggml-metal-context.{h,m}`：struct 新增 `dispatch_semaphore_t cgc_sem`；
  兩個 `addCompletedHandler` 在 `atomic_fetch_add(cgc_done)` 後 `dispatch_semaphore_signal`；
  新增 `ggml_metal_wait_cgc_done(ctx, target)`（while re-check + semaphore wait，100ms 安全 timeout）。
- `ggml-metal.cpp`：註冊 `ggml_metal_wait_cgc_done` proc address。
- `ggml-backend.cpp`：submit-after / submit-ahead 兩處 poll 改優先走 `cgc_wait`（無 spin），
  保留 sched_yield 為 fallback。

### 結果（qwen36, -ngl 99, 4.77GiB pool, n_cb=8）

| 設定 | eval t/s | per-token | CGC-SEG wait/cb |
|---|---|---|---|
| 改前 submit-after + poll | 8.35 | 118ms | wait 2065 / cb 431µs |
| **改後 submit-after + semaphore** | **15.6** | 64ms | wait 1150 / cb 340µs |

- CGC_N_CB=16 反效果（14.78，encode 飽和），cb8 維持最佳。
- 輸出 bit-identical / 語義正確（短、長、coding prompt 皆驗證）。

### 長短 prompt 驗證（使用者要求：長短都要、coding 優先）

| prompt | prefill | decode eval | hit rate | 輸出 |
|---|---|---|---|---|
| "The capital of France is"（6 tok） | 7.0 t/s | 15.6 t/s | 95.6% | 正確 "Paris" |
| fibonacci coding（19 tok） | 9.7 t/s | 17.1 t/s | 96.4% | 正確 Python code |
| matching-engine code review（610 tok） | **15.6 t/s** | **16.5 t/s** | 97.8% | 正確 code review |

---

## 0822 總結與決策點

- **正確路徑（submit-after + semaphore wake）**：**15.6–17.1 t/s decode、正確輸出** ——
  從 7.07 → 8.35 → 15.6 t/s 三階提升；短/長/coding prompt 全部正確。
- **fast path（submit-ahead + GPU-remap + re-run）**：被 double-compute 缺陷 + 高 dirty rate 卡死；
  submit-ahead 單獨跑可達 22.8 t/s 但 remap race → 錯誤輸出，**禁止進生產**（launcher 已強制
  `CGC_SUBMIT_AFTER=1`）。
- **25 t/s 正確在天花板上不可達**（本硬體 + 本模型）：
  - GPU 純計算牆 = 44ms/step（41 segment × ~1.07ms，Metal FFN MoE）→ 22.7 t/s 極限。
  - 冷讀 hook（drift ~17–28 miss/step，首次見到無法預測）≈ 14ms/step。
  - 4.77GiB pool 是安全上限（5.1GiB 即 OOM）；union 持續成長（step 120 仍 +22）→
    長 prompt 不可能全 resident。
- **若要 25+ t/s**，可行方向（需用戶裁決）：
  1. 放寬硬體（更多 RAM → 更大 pool / 整檔常駐）—— 非軟體解；
  2. 啟用 MTP（graft blk.40，accept ~1.5 token/step → 有效 ~25 t/s）—— 非純 decode 速度，但
     是唯一能跨過 GPU 牆的軟體路徑；
  3. 接受 15.6–17.1 t/s 正確（此預算下的工程地板，已是正確路徑的實際上限）。

