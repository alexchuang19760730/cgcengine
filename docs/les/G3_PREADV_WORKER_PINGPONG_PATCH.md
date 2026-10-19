# G3 施工補丁：後台 preadv worker + 非阻塞 ping-pong（READ=8，攻 SSD miss 等待）

> Charter：`scripts/check/charters/e-les-router-prefetch-2026-10-17.yaml`（G3，行 112–118；產物自證欄 `les:{...}`）
> 上游：`docs/les/upstream/llama-moe-slot.cpp`（`struct lookahead_expert_prefetch` 行 118–462；window 落槽 821–910；`process()` 2287–2637；`build_slot_ids_tensor` 2272–2285）
> 外部依據：`docs/les/G3_EXTERNAL_EVIDENCE_EMBER_SPECPREFETCH_2026-10-18.md`（SpecPrefetch 存儲敏感性：慢 UFS/SD 上 +15~20%，快 NVMe 上 0%）
> 前置：**G2 補丁已 apply 且編譯通過**（本檔複用 G2 的圖內 `cgc_les_topk-<il>` 頂點，不重建圖）。
> 狀態：**施工圖紙（逐字可 apply），供 review；不需 GPU 即可讀完。所有 GPU 驗證等乾淨窗口。**

---

## 0. 一句話目標

G2 已經在圖裡算出「下一層真實 router 的 top-8 專家 id」（`cgc_les_topk-<il>`，G1 離線命中率 0.79–0.82），但 G2 **只算不讀**。
G3 把這 8 個 id 在**第 il 層 FFN 開始計算的那一刻**投遞給專家池的後台讀取線程，讓它在 GPU 算第 il 層 FFN 的同時，把第 il+1 層需要的專家從 SSD preadv 進池；等 GPU 走到 il+1 的 `on_topk` 時，這些專家已在池裡，**原本在關鍵路徑上的同步 pread 等待被藏進 il 層的 compute 窗口**。

這直接攻的是成本模型裡那 **43.20 ms / token（83.8%）的非 kernel 開銷**中的 SSD miss 等待（`fill_wait_us`），不是 GEMV kernel 本身。

**硬約束（從 0C 與 ρ 兩條死線繼承，不可違反）**：
1. **非阻塞、代價有界**：後台讀沒趕上就跳過，主路徑按 miss 自己同步讀，絕不等待整批（上游舊版等 K=8 白等 54.6 ms、K=16 白等 259 ms 是歷史教訓）。
2. **不擴大 union、不洗熱 cache**：G3 只預測**下一層 top-8**（專家數不超過真實 top-k），不是 0C 死掉的「N 個 draft union 8→19」（那條 capacity miss ×10.2）。
3. **預設 OFF 全程 byte-identical**，M1/M2 閘門必須過；MTP 不碰（trunk ctx only）。

---

## 1. 為什麼這次窗口可能夠（與死掉的 ρ 線對比）

| 維度 | ρ layer-batch（已判 NO_EFFECT） | **G3 LES（本檔）** |
|---|---|---|
| 預測信號 | 同層近似 gate（跳過 attn），覆蓋 0.849 但信號是近似 | **下一層真實 router**：`attn_post_norm(il) × W_gate[il+1]`，G1 命中率 0.79–0.82 |
| 圖中位置 | 影子節點在 il 層**第一個算子**，只比真 topk 早一個 submodule | `cgc_les_topk` 在 **attn_post_norm 後、`build_layer_ffn` 前**，窗口＝il 層整段 FFN（gate/up/down 專家 GEMV，decode 每層耗時大頭） |
| 實測窗口 | insert 4.76 ms、窗口 1.3–1.6 ms，讀不完 | 窗口起點更早（**待 G3a 實測**，不預設結論） |
| 教訓 | fill 空轉：8 萬次投遽灌爆設備、fill_wait 56 ms→15–20 s、t/s −21% | 繼承 MAXQ 熔斷 + 非阻塞 + 批次 merge，且 free-slot-only 起步 |

> 誠實邊界：上游 iPad 口徑「讀 1 專家 ~1.00 ms、算 1 層 ~1.15 ms、8 專家串行 8 ms ≫ 1.15 ms 窗口」。
> 在 M4 Air 上**一層 FFN 窗口到底多長、merged pread 8 個專家要多久，必須由 G3a 量 `n_wait_ns` 與 `n_used_from_pf` 決定**，本檔不預設「一定藏得住」。讀不完有 G3b（截斷/並行）兜底。

---

## 2. 上游機制 → CGC 現有構件對照（決定「不重寫、只接線」）

讀碼結論：本倉的專家池是**獨立的 `llama_expert_cache`**（不是上游的 `llama_moe_slot_runtime`），而且後台讀取、批次 merge、狀態機、LRU 保護都已具備。G3 是「**換預測信號源 + 加一個圖內 arm 回調**」，不是新建 slot runtime。

| 上游 slot runtime 構件 | CGC 現有對應物 | G3 動作 |
|---|---|---|
| 圖內 `ggml_map_custom2(...map_custom_lookahead)` 回調 `process()`（`build_slot_ids_tensor:2281`） | `ggml_backend_sched_set_eval_callback(...expert_cache_eval_cb)`（`llama-context.cpp:1971`），已有 `cgc_les_h`/`cgc_rho_logits`/`ffn_moe_topk` 三支 in-cb 先例 | **新增一支 `cgc_les_topk` 的 in-cb 回調**，dispatch 時 arm |
| `arm(layer_next, pred)`（非阻塞，忙則 `n_arm_skipped` 丟棄） | `llama_expert_cache_prefetch_batch(cache, layer, exps[], n)`（`llama-expert-cache.cpp:1992`，入 `pool_batch_queue`，MAXQ 熔斷） | 新增 `llama_context::cgc_les_prefetch(il+1, ids, k)` 包裝它 |
| 後台 `worker()` + `moe_ssd_io_threads()` 並行 pread（slot.cpp:396–461） | `bg_loop()`（`:3712`，batch 優先）→ `fill_segments_merged_serial`（`:3801`，**單線程 merged preadv**）；另有 `pool_loop()`（`:3907`，`pread_job` 多 worker 池，`LLAMA_EXPERT_CACHE_WORKERS`） | **G3a 用 serial merge**；讀不完才在 G3b 接並行 worker |
| window 預讀暫存區（每步重填、**不參與跨 token LRU**）＋`kSlotPending` | `prefetch_claim_slot_locked`（`:1847`，**free-slot 優先**，無 free 才 LRU/SpAc/pin 兜底）＋`slot_queued/loading` 狀態機 | G3a 設 **free-slot-only**（無 free 即 skip，零 evict）；G3c 才開受控 evict |
| `take()` 非阻塞取結果（window 模式完全不等；沒發布主路徑自讀） | DBUF2 預設 OFF（`llama-expert-cache.h:754`，生產 C17 是 `CGC_DBUF=1` 不是 DBUF2）⇒ bg fill 直接 publish **active** `slot_table`（`:3875`），**當步 il+1 即可見** | 不新增 take()；il+1 `on_topk` 自然命中，沒趕上走同步 ensure |
| `publish_slot_in_window` 讀盤期間被主路徑搶裝則丟棄 | bg phase3 已有 stale/`bg_reassign_race` 校驗（`:3776-3782`）＋`prefetch_slot` 已駐留即返回（`:1948`） | 複用，不新增去重 |
| 淘汰保護 `keep[]` / `n_evict_saved`（process:2468-2486, 2530-2554） | `slot_pinned` / `slot_pinned_static` / prefill-protect / SpAc 效用 victim | G3c 若開 evict，新增「一步即釋」的 `les_keep` bitmap，複用 pin 跳過路徑 |
| 讀完計數 `n_armed/n_arm_skipped/n_pred_read/n_pred_published/n_pred_cached/n_wait_ns` | `n_prefetch/n_prefetch_dropped/drop_stats.{maxq_limit,no_free_slot,guard_reject}/fill_wait_us/n_map_hits/misses` | 新增 LES 專屬計數（見 §7），複用 fill_wait |

**關鍵簡化**：上游 ping-pong 尾部專用區（`slot_floor`/`bank_base` A/B 兩區）是為「不依賴每層窗口、整塊共享 bank」設計的。我們已是 **per-layer 固定槽池**（`slot_table[layer*n_expert]`），層間天然不互相覆蓋，且 DBUF2 關閉時當步可見——**不需要移植 A/B 雙區**，只需要 G3c 的「預取不 evict 熱槽」等價保護。標題保留「ping-pong」是指其**非阻塞 handoff 語義**（arm 後不等、用時只取已完成的），不是物理雙緩衝。

---

## 3. 目標時序（每個 decode token，逐層）

```
 il 層:
  attn 計算 ──► attn_post_norm(il)
                    │
                    ├─(G2 圖內頂點) cgc_les_topk-<il> = argsort( W_gate[il+1] · attn_post_norm(il) )  [I32 top-8]
                    │        │  (GPU/Metal 算出)
                    │        ▼
                    │  [G3 in-cb] expert_cache_eval_cb 看到 cgc_les_topk-<il> dispatch(ask=false)
                    │        ├─ 取 8 個 I32 id（target = il+1）
                    │        ├─ 過濾 il+1 已駐留（n_pred_cached）
                    │        ├─ free-slot 認領 + read_n 截斷 + MAXQ 熔斷
                    │        └─ prefetch_batch(il+1, top8) ──► pool_batch_queue（非阻塞，立即返回）
                    │
                    └─► il 層 FFN（gate/up/down 專家 GEMV） ◄──── BG 線程與這段 GPU compute 並行：
                            │                                        bg_loop merge-preadv 讀 il+1 的專家
                            │                                        讀完直接 publish active slot_table
 il+1 層:                   │
  ffn_moe_topk-(il+1) ──► on_topk(t)：ensure_batch 查 slot_table
                            ├─ 預測中的、BG 已填好 → 命中（n_used_from_pf，零同步讀）
                            └─ 沒趕上/預測錯 → 原同步 ensure pread（代價有界，與今天完全相同）
```

- 信號方向是**同一個 token、il→il+1**（不是跨步 token）。層 0 永遠沒有預測（上一層不存在），覆蓋率結構上界 39/40，與 G1 一致。
- 預測錯的專家（~18–21%）若被讀了，是**浪費的讀 IO**，但不影響正確性（on_topk 只認真實 top-8）；G3a 用 `read_n` 與 free-slot-only 把這部分浪費關在小範圍，G4 卡 `file_reads/token +≤10%`。

---

## 4. 分階段施工（小步、可回滾、每步可單獨 A/B）

> 原則：**先用最小改動回答「串行 merged pread 能不能在 il 層 FFN 窗口內藏住、在線命中率是否復現 G1 的 0.79–0.82」**。只有 G3a 證據顯示讀不完，才動 G3b；只有 G6 縮池需要，才動 G3c。不一次到位。

### G3a（最小、零新增線程、零 evict）—— 本檔主體

只加「圖內 arm + 現成 serial batch + free-slot-only + 在線命中率/計數」。

#### 4A.1 `src/llama.cpp/src/llama-context.cpp`：in-cb arm 回調

**定位**：`expert_cache_eval_cb`（`:6392`），緊鄰現有 `cgc_les_h` 探針分支（`:6414-6422`）之後、`cgc_rho_probe` 分支之前。

**新增分支（逐字）**：
```cpp
    // [G3 LES 2026-10-18] 圖內 router 預測一落地就 arm 後台 preadv（charter e-les G3）。
    // 頂點 cgc_les_topk-<il> 由 G2 建在 attn_post_norm(il) 後、il 層 FFN 前，所以它在
    // GPU 執行流到 il 層 FFN 之前 dispatch —— 後台讀 il+1 專家與 il 層 FFN compute 重疊。
    // 只在主幹 decode、ask=false（真 dispatch）時動；MTP draft ctx 不排（見 cgc_rho_prefetch 同款理由）。
    static const bool cgc_les_pf_on =
        getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH") != nullptr &&
        getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH")[0] != '0' &&
        getenv("CGC_LES_PREFETCH") != nullptr &&
        getenv("CGC_LES_PREFETCH")[0] != '0';
    if (cgc_les_pf_on && !ask &&
        ctx->cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
        strncmp(t->name, "cgc_les_topk", 12) == 0) {   // 前綴長度對齊 G2 whitelist（ggml-backend.cpp:117）
        const char * dash = strrchr(t->name, '-');
        const int src_il = dash ? atoi(dash + 1) : -1;
        ctx->cgc_les_arm_from_topk(src_il, t);
    }
```

> 取數（Metal，G2 已鋪路）：`cgc_les_topk` 由 G2 建在 Metal 上（`ggml_argsort_top_k`，見 G2 §2.1）。in-cb 在 `!ask`（節點已 dispatch 算出）時，直接用 G2 已寫好的 `ggml_backend_tensor_get(t, buf, 0, depth*4)`（`cgc_les_topk_capture:177` 同款）把 8×int32（32 B）拉到棧上 staging。
> - 這是一個強同步點，G3a 必須在計數裡量它的耗時（預期極小，但不得假設它免費）。
> - 可選優化（**僅當 G3a 量到該同步抵消重疊**才做）：把 topk 結果落 host（argsort 後接一個 CPU 側拷貝 custom op，或直接用 host backend 算 argsort——top-8 from 256 是極小運算）；下游無 GPU 算子消費它，放 host 零 GPU 成本。
> - 該取數只在 `cgc_les_pf_on` 時存在，預設 OFF 不進圖、byte-identical。

#### 4A.2 `src/llama.cpp/src/llama-context.{h,cpp}`：新增 `cgc_les_arm_from_topk`

宣告放 `llama-context.h` 與其他 `cgc_*` 私有方法同區；實作緊鄰 `cgc_rho_prefetch`（`:6299`，作為結構模板）。

```cpp
void llama_context::cgc_les_arm_from_topk(int src_il, const ggml_tensor * t);
```

```cpp
void llama_context::cgc_les_arm_from_topk(int src_il, const ggml_tensor * t) {
    llama_expert_cache * cache = model.expert_cache;
    if (cache == nullptr || !llama_expert_cache_pool_active(cache)) return;
    const int target_il = src_il + 1;                       // 預測的是下一層
    if (src_il < 0 || target_il >= (int) model.hparams.n_layer_all) return;
    if (target_il < 4) return;   // G1 裁定口徑：豁免前 3 層（與 G2 cgc_les_topk_capture:182 一致）

    // ---- 取 top-k I32 ----
    // G2 頂點語義（G2 補丁 §2.1）：ggml_argsort_top_k 出 I32 排序索引，DESC 降序 + view 前 k，
    // shape [depth,1]，故 stk[0..depth-1] 就是 logits 最高的前 depth 個專家（stk[0] 最強），
    // 截斷取前 read_n 個＝最高置信的前 N 個（不是升序）。
    int32_t stk[64];
    int32_t depth = 8;
    if (const char * d = getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH")) depth = atoi(d);
    if (depth < 1) depth = 1;
    if (depth > 64) depth = 64;
    int32_t read_n = depth;
    if (const char * r = getenv("CGC_LES_PREFETCH_READ")) read_n = atoi(r);   // 截斷
    if (read_n < 0) read_n = 0;
    if (read_n > depth) read_n = depth;

    // 取數：與 G2 已寫好的 cgc_les_topk_capture（G2 補丁 §2.2，:171-177）**逐字同款**
    //   ggml_backend_tensor_get(t, pred, 0, depth*4)（depth*4 字節；Metal 頂點需此同步，
    //   極小，G3a 量它的耗時）。建議把 G2 的取數段抽成共用 helper 供 capture 與本 arm 複用。
    ggml_backend_tensor_get(t, stk, 0, (size_t) depth * sizeof(int32_t));

    // 去重、截到 read_n（降序前面是最強的，截斷自然保留高置信）：
    std::vector<uint32_t> want;
    want.reserve(read_n);
    bool seen[256] = {false};
    for (int j = 0; j < read_n && j < (int)t->ne[0]; ++j) {
        const int32_t e = stk[j];
        if (e < 0 || e >= (int) model.hparams.n_expert || seen[e]) continue;
        seen[e] = 1;
        want.push_back((uint32_t) e);
    }
    if (want.empty()) return;

    // ---- 層容量閘（與 cgc_rho_prefetch 同款：裝不下就是整層讀，不是預取）----
    const uint32_t cap = llama_expert_cache_slots_per_layer_l(cache, (uint32_t) target_il);
    if (cap > 0 && want.size() > (size_t) cap) { g_les.n_arm_skip_cap++; return; }

    // ---- 投遞（G3a：free-slot-only 由 cache 端閘控制，見 4A.3）----
    const int32_t q = llama_expert_cache_prefetch_batch(
        cache, (uint32_t) target_il, want.data(), want.size());
    g_les.n_armed++;
    if (q < 0) g_les.n_arm_skipped++;            // MAXQ / pool inactive / batch 被丟
    else        g_les.n_queued += (uint64_t) q;
    // 另記：被 target 層「已駐留」過濾的數（命中率核算所需，見 §7 線上對賬）
}
```

> 與 `cgc_rho_prefetch` 的唯一本質差異：① 目標層是 `src_il+1`（跨層），ρ 是同層；② 信號是 G2 真實 next-layer router，不是近似 gate；③ 沒有逐 token union（decode n_tokens==1，直接 top-8，最多 8 個）。

#### 4A.3 `src/llama.cpp/src/llama-expert-cache.{h,cpp}`：G3a 強制 free-slot-only（零 churn）

`prefetch_batch`（`:1992`）目前在無 free slot 時走 `prefetch_claim_slot_locked` 的 LRU 兜底 evict（`:1857` 起）。G3a 必須**禁止 LES 批次 evict 熱槽**，新增一個「只認 free slot」的認領模式，避免在 il+1 還沒 ensure 時（union 尚未 bump）誤殺下一步熱集（0C churn 的同款風險）。

- 新增 `llama_expert_cache_prefetch_batch_reserved(cache, layer, exps, n, bool free_only)`，或在現有函式加一個 thread-local/參數 `free_only`。
- `free_only=true` 時，認領只走 `prefetch_claim_slot_locked` 的**第一個循環**（`:1851-1856`，`slot_owner<0 && !queued && !loading`），找不到直接 `drop_stats.no_free_slot++` 返回，**不進 LRU/SpAc/pin evict**。
- G3a 呼叫端固定 `free_only=true`。池滿（8G 下 143 槽通常有 free；6G/4G 下會更早 no_free_slot，這正是 G6 要量的曲線）。

> 語義說明：零 evict 表示「預取只填本來就空的槽」，因此**不可能**把任何已駐留專家洗出——churn 在數學上被歸零，作為 G3a 的安全基線。G3c 才在有 keep 保護的前提下放開受控 evict。

#### 4A.4 不新增任何執行緒、不碰圖的其餘部分

G3a 完全複用 `bg_loop` 單線程 + `fill_segments_merged_serial`（一次 preadv 提交該層多個專家的 file-contiguous 段）。`LLAMA_EXPERT_CACHE_WORKERS` 的 `pool_loop` 並行池**不在 G3a 啟用給 LES**，先量串行 merge 的牆鐘。

---

### G3b（條件性：G3a 證明「讀不完」才做）

**觸發判據（G3a 實測，任一）**：`n_arm_skipped` 佔比高（bg 忙）、`n_used_from_pf / n_pred_published` 顯著低於 G1 的 0.79、il+1 `on_topk` 的 `fill_wait_us` 中位數沒有下降。

按成本從低到高，**先做截斷（最便宜）再做並行**：

1. **read_n 截斷（先做，零並發）**：把 `CGC_LES_PREFETCH_READ` 從 8 往下調（如 6/4），只 preadv 最有把握的前 N 個。G1 recall 曲線是 depth 函數（d8=0.79/d10=.86/d12=.90/d16=.94），但那是「多預測多命中」；截斷方向相反，要在 G3a 埋點裡按**預測名次**分桶統計「第 k 名真實命中率」，選「命中率/讀取量」拐點。讀不完就不硬讀，其餘交給同步路徑（符合 SpecPrefetch 的 window-aware truncation）。
2. **並行 batch（後做）**：新增一條 LES 並行通道，把一批 8 個專家分給 `pool_loop` 的多個 worker（`LLAMA_EXPERT_CACHE_WORKERS`，現成 `pread_job` 池），每個 worker 對一個專家的 packed 連續段 pread（上游 `worker()` 用 `moe_ssd_io_threads()` 個線程同款），牆鐘從「8 段串行 merge」變「ceil(8/n_th) 輪」。
   - 必須保持與 `fill_segments_merged_serial` 相同的 publish 語義（phase3 持鎖、stale/race 校驗、DBUF2 分支），只是把「一個 merged preadv」換成「N 個並行 pread + join」。
   - env `CGC_LES_PREFETCH_PARALLEL=1` 開，預設 0（serial）。並行會增加 SSD 隨機讀壓力與 GPU 搶 DRAM 帶寬，必須 A/B 量 `fill_wait` 與前台 cb/union gap（bg QoS 可用現成 `CGC_BG_QOS_BACKGROUND`）。

---

### G3c（條件性：G6 縮池需要才做）—— 受控 evict + 預測保護

6G/4G 池 free slot 不足時，G3a 會大量 `no_free_slot` skip（預取形同關閉）。要在縮池下仍有預取，需要上游 `keep[]` 的等價物：

1. 在 `llama_expert_cache` 新增 per-layer「LES 一步即釋保護位」`les_keep[layer][expert]`（或複用 `slot_pinned` 但於 il+1 `on_topk` 結束時**立即解除**，絕不跨 token 常駐）。
2. arm 時把 want 集合寫入 `les_keep[target_il]`；`pick_slot`/`prefetch_claim_slot_locked` 在淘汰 il+1 槽位時，pass0 跳過 `les_keep` 標記的 owner（計 `n_evict_saved`），pass1 才放寬（計 `n_evict_forced`）。
3. il+1 `on_topk` 完成後清 `les_keep[target_il]`。
4. env `CGC_LES_PREFETCH_EVICT=1` 才開（預設 0＝G3a 零 evict）。
5. **G6 驗收硬門檻**：開 evict 後 `miss_capacity` 不得上升、`hit_rate_pct` 不得低於同池 off 臂、churn 計數不惡化（否則回退 free-only）。

---

## 5. 環境開關（全部預設關閉）

| env | 預設 | 含義 |
|---|---|---|
| `LLAMA_MOE_LOOKAHEAD_PREFETCH=1` | 關 | G2/G3 總開關：建圖內 next-router 預測頂點（關＝頂點不存在，byte-identical） |
| `LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH` | 8 | 預測專家數（argsort top-depth，1..16 對應上游 kMaxDepth，本檔暫封頂 64） |
| `CGC_LES_PREFETCH=1` | 關 | **G3 讀盤開關**：in-cb arm 後台 preadv。只設 `LLAMA_MOE_LOOKAHEAD_PREFETCH` 不設它＝G2 只預測不讀（等價上游 `LLAMA_MOE_PREFETCH_READ=0`） |
| `CGC_LES_PREFETCH_READ` | =depth | 實際讀幾個（截斷，0..depth），G3b 用 |
| `CGC_LES_PREFETCH_PARALLEL` | 0 | G3b：1＝多 worker 並行 pread；0＝serial merged preadv |
| `CGC_LES_PREFETCH_EVICT` | 0 | G3c：1＝允許 LES 批次 evict（配合 les_keep），0＝free-slot-only |
| 現成 `CGC_RHO_PREFETCH_MAXQ` | 不限 | LES 批次複用同一 MAXQ 熔斷（`pool_batch_queue` 深度），G3a 建議顯式設 8–16 防灌爆 |

> 開關命名說明（與 G2 / 上游的對齊）：上游 slot runtime 用 `LLAMA_MOE_PREFETCH_READ=0/1` 表「只預測/真讀」；G2 階段沿用該約定保持 `LLAMA_MOE_PREFETCH_READ=0`（只預測）。G3 不直接複用此變數，因為它綁定的是**未移植的上游 slot runtime**；本引擎改用獨立的 `CGC_` 前綴開關（與 `CGC_RHO`/`CGC_SPAC` 慣例一致），由 `CGC_LES_PREFETCH=1` 接管「真讀」。對應關係：`LLAMA_MOE_PREFETCH_READ=0` ≡ 只設 `LLAMA_MOE_LOOKAHEAD_PREFETCH`（G2）；`LLAMA_MOE_PREFETCH_READ=1` ≡ 再加 `CGC_LES_PREFETCH=1`（G3）。

> 與 ρ 互斥說明：LES 與 ρ 是兩個不同預測源，**不得同時開 fill**（會爭搶 bg 隊列與槽位、混疊命中率歸因）。G3 臂設 `CGC_LES_PREFETCH=1` 且不設 `CGC_RHO`/`CGC_RHO_FILL`。

---

## 6. 編譯 / 驗證 / 回滾

**前置閘門（同 G2）**：
- [ ] commit2 六檔已 commit（乾淨窗口 replay PASS）。
- [ ] G2 補丁已 apply（`cgc_les_topk` 頂點 + whitelist + capture 已在），純探針編譯 `BUILD_EXIT=0`。
- [ ] 7 個 affinity hunk 已撤，工作區為「G2 探針」狀態。
- [ ] 乾淨 GPU 窗口：重開機 swap used 歸零、thermal nominal、退 WorkBuddy/Chrome/FreeBuff。

**編譯**：
```sh
export PATH=/opt/homebrew/bin:$PATH
cmake --build src/llama.cpp/build --parallel 8
# 產物 build/bin/llama-server（含 G3，預設 OFF）
```

**階段0 自證（OFF 必過）**：
- 不設任何 LES env 跑現有 prod-new cell，`CGC_MM_BITIDENT=1` 的 M1/M2 bit-identical 閘門必須照舊 PASS（驗證頂點/回調在 OFF 時完全不進圖、不進 cb）。

**G3a 觀察臂（單臂先看機制，再做 G4 配對）**：
```sh
# 只開 G3（G2 頂點 + G3 fill），free-slot-only、serial merge、MAXQ 熔斷
CGC_METAL_RESERVE_MB=768 BUDGET_GATE=warn \
LLAMA_MOE_LOOKAHEAD_PREFETCH=1 LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH=8 \
CGC_LES_PREFETCH=1 CGC_LES_PREFETCH_READ=8 CGC_LES_PREFETCH_EVICT=0 \
CGC_RHO_PREFETCH_MAXQ=16 \
/opt/homebrew/bin/python3 scripts/check/harness.py bench \
  --arm "prod-new:LLAMA_MOE_LOOKAHEAD_PREFETCH=1;CGC_LES_PREFETCH=1" \
  --charter scripts/check/charters/e-les-router-prefetch-2026-10-17.yaml \
  --warm-skip 64 --prompt 512 --gen 128 --depths 0 --reps 5 --fixed-fill-seed 1 \
  --arm-timeout 420 --stall-watch 240 --stall-kill \
  --workdir <W> --json <W/g3a_on.json>
```
對照臂同 cell 但不設 `CGC_LES_PREFETCH`（G2 只預測）。export `PYTHONPATH=/opt/homebrew/lib/python3.14/site-packages`。

**回滾**：所有 env 預設關；關閉後路徑完全等價 G2（關 `LLAMA_MOE_LOOKAHEAD_PREFETCH` 則等價 commit2 純探針）。程式碼保留、不刪除既有跨步 prefetch/SpAc/ρ 路徑（互斥由 env 保證）。

---

## 7. 埋點與在線對賬（charter 產物自證欄 `les:*`）

在 server 收尾的 cache 統計區（`llama-expert-cache.cpp` 計數列印處，與 `CGC-RHO-FILL` 同款）新增 `CGC-LES-G3` 一行，欄位對齊 charter 自證欄：

| 欄位 | 來源 | 用途 |
|---|---|---|
| `n_armed` | G3 新增（每層 arm 一次） | 機制是否在跑 |
| `n_arm_skipped` | prefetch_batch 返回 <0（MAXQ/pool） | bg 忙到被丟的比例（讀不完指標） |
| `n_arm_skip_cap` | 層容量閘擋下 | 異常 union 膨脹（decode 單 token 應恆 0） |
| `n_pred_cached` | arm 時 target 層已駐留而過濾 | 熱池已飽和度（8G 應高） |
| `n_pred_read` / `n_pred_published` | bg 實際投遞 / 填好發布 | 真正完成的預取 |
| `n_no_free` | free-slot-only 無空槽 skip | G6 縮池是否需要 G3c |
| `n_used_from_pf` | **il+1 `on_topk` 真實 top-8 命中、且該 slot 是 LES bg 剛填的** | **線上真命中率（分子）**，對 `n_pred_published` 得精度、對真實 top-8 得 recall |
| `n_evict_saved` / `n_evict_forced` | G3c keep 保護 | 縮池 evict 行為 |
| `n_wait_ns` | 複用 `fill_wait_us`（按層中位） | **t/s 直接來源**：G3 on vs G2 的同步等待下降 |

**在線命中率對賬（複用 G2 已鋪的 ring）**：G2 已在 capture 真實路由 `g_cgc_les_real[il]`。G3 arm 時把投遞的 want 集合也存一份 `g_les_armed[target_il]`；il+1 `on_topk` 拿到真實 top-8 時算交集，累加 `n_used_from_pf` 與線上 recall。這在引擎內復現 G1 離線的 0.79–0.82，且進一步區分「**預測對了**」（G1 recall）與「**對了且趕在 il+1 前讀好了**」（`n_used_from_pf`，G3 真正的有效命中）——兩者差值就是「窗口讀不完」的量化。

---

## 8. 驗收標準（對齊 charter G3，行 112–118）

一次 GPU（G3a）需同時看到：
1. **bit-exact**：兩臂（off/on）M1/M2 逐字節一致（`CGC_MM_BITIDENT`），數值不變。
2. **機制活著**：`n_armed>0`、`n_pred_read>0`、`n_pred_published>0`；`n_arm_skip_cap=0`（decode 單 token）。
3. **在線命中率復現**：`n_used_from_pf / 真實 top-8 總數` 落在 G1 離線 0.79–0.82 帶（明顯偏低 ⇒ 窗口讀不完，進 G3b）。
4. **等待下降**：il+1 同步 `fill_wait_us` 中位數較 G2-only 臂下降（這是 t/s 的因果中繼，必須量到，不能只看 t/s）。
5. **不拖垮前台**：後台 IO 不使前台 cb/union gap 上升；`file_reads/token` 增幅在 G4 容許 ≤10% 以內（G3a 先觀察記錄）。
6. **不 churn**：`miss_capacity` 不上升、`hit_rate_pct` 不低於 G2-only 臂、swap/thermal 不惡化。

若 3 不通（讀不完）：先 G3b 截斷（read_n 6/4）量召回-讀取拐點，再 G3b 並行；不許跳過「量 n_wait_ns」直接宣稱收益。

---

## 9. 誠實邊界（不許諾 25）

- G3 攻的是 **decode 非 kernel 側的 SSD miss 等待**，與 SpecPrefetch 同機制；其增益的物理前提是「本機屬慢存儲、I/O 在關鍵路徑上」。本機讀 1 專家、一層 FFN 窗口、merged pread 牆鐘**三個數 G3a 才量得到**，在此之前不給 t/s 承諾。
- 上游 2.97→9.81（3.3×）是**冷啟動**基線；我們已在 12–20 t/s 熱態，邊際完全不同，不能外推。
- G3 是 decode 線，對 **prefill 250 無直接作用**。
- 一層窗口結構上可能讀不完 8 個專家；截斷會放棄部分命中率、並行會加帶寬壓力，二者都有代價，以 G3a 數據選。
- G3 只負責「把 SSD miss 藏進當前層 compute」；decode 25 的其餘缺口（expert kernel 利用率 38–45%、cb 重疊）是獨立課題，不在本檔。

---

## 10. 與 G4 / G6 的銜接

- **G4（生產 ABBA）**：G3a 單臂觀察確認機制有效後，進 G4 配對 off/on ≥3 對，卡配對中位增益 ≥+5%、prefill 不退化、`file_reads/token +≤10%`。
- **G6（縮池轉正）**：G3a/free-only 在 8G 驗證後，G6 跑 off/on × {8,6,4}G；6G/4G 若 free-slot 不足（`n_no_free` 高），再做 G3c（keep 保護 + 受控 evict），用權威一手數據回答「6G+LES 能否追平 8G-off」。
- **MTP（G5）**：本檔不碰 MTP ctx；MTP 維持只驗共存/降級/bit-exact，不把 LES 預測接進 MTP verify union（那是 0C 已判死的擴大 union 路徑）。

---

## 附：本檔讀碼依據（行號以 demo/sweet-spot-windows-fix 工作區當前源碼為準）

- `llama-context.cpp`：eval_cb `:6392`；`cgc_les_h` 分支 `:6414-6422`；ρ in-cb fill `:6462-6472`；`ffn_moe_topk→on_topk` `:6525-6528`；`cgc_rho_prefetch` `:6299-6390`；跨步 B prefetch `:2173-2304`；sched callback 註冊 `:1971`。
- `llama-expert-cache.cpp`：`prefetch_slot` `:1932`；`prefetch_batch` `:1992`；`prefetch_claim_slot_locked` `:1847-1930`；`bg_loop` `:3712`（batch 優先 `:3741`、merged serial `:3801`、publish `:3809-3876`）；`pool_loop` `:3907`；`dbuf2_swap` `:2209`。
- `llama-expert-cache.h`：`cgc_dbuf2_on` `:754`（預設 OFF）；slot 結構與狀態機 `:76-253`。
- 上游鏡像：`docs/les/upstream/llama-moe-slot.cpp`（struct `:118-462`、window 落槽 `:821-910`、`process :2287-2637`、`map_custom_lookahead :2165`、`build_slot_ids_tensor :2272`）。
