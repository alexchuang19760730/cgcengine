# G3a 程式碼補丁：圖內 LES arm + free-slot-only 後台 preadv（最小步，逐字可 apply）

> 2026-10-18 · charter `e-les-router-prefetch-2026-10-17` G3 · 分支 `demo/sweet-spot-windows-fix`
>
> **本檔只含 G3a（最小步）**：圖內 arm、複用現成 serial merged preadv、**free-slot-only（零 evict）**、在線命中率/埋點。
> 不含 G3b（多 worker 並行 pread）、G3c（受控 evict）。
>
> **強制順序（缺一不可）**：commit2 落地 → **apply G2 補丁**（`docs/les/G2_INLINE_ARGSORT_PATCH.md`，它建立 `cgc_les_topk-<il>` 圖頂點與 whitelist）→ **apply 本 G3a 補丁** → 編譯。
> G3a 依賴 G2 的 `cgc_les_topk-<il>` I32 頂點；未 apply G2 時本補丁編譯會過（G3a 只是在 cb 裡按名稱匹配一個不會出現的頂點），但打開開關也無頂點可 arm（no-op）。

---

## 0. 本補丁動了什麼（四個檔）

| 檔 | 改動 | 預設關閉時 |
|---|---|---|
| `llama-expert-cache.h` | 新增 per-layer 狀態 bitmap `les_arm`、8 個 LES 計數器、`llama_expert_cache_prefetch_les()` 宣告 | 欄位存在但恆 0，無路徑寫入 |
| `llama-expert-cache.cpp` | `prefetch_claim_slot_locked` 加 `free_only`；新增 `prefetch_les`；bg 發布標記 2→3；`ensure_slot` 命中/miss 歸因；初始化；析構統計行 | 無開關時不進任何新分支 |
| `llama-context.h` | 宣告 `cgc_les_arm_from_topk()` | 純宣告 |
| `llama-context.cpp` | eval_cb 加 `cgc_les_topk` in-cb arm 分支；實作 `cgc_les_arm_from_topk()` | 頂點不存在（G2 也關）→ 永不觸發 |

**歸因設計（內聚在 cache 側，不依賴 G2 的 ring）**：`les_arm[layer][expert]` 是單 byte 狀態：
`0`=本步未預測；`1`=預測但 arm 時已駐留（免讀）；`2`=預測且已投遞 free-slot 背景讀（讀取中）；`3`=背景讀已在該層 ensure 前發布（**窗口命中**）。每層每步 arm 一次（來自上一層 topk），ensure 觸碰該專家時 consume 清零。

---

## 1. `src/llama.cpp/src/llama-expert-cache.h`

### H1 — 新增 per-layer 狀態 bitmap

**找**（約 418 行）：
```cpp
    std::vector<std::vector<uint8_t>> ever_loaded;   // [layer][expert] 1 = demanded at least once
    std::vector<uint32_t>             n_distinct_demanded; // [layer] distinct experts ever demanded
```
**在其後新增**：
```cpp
    // [G3a LES 2026-10-18] in-graph next-router lookahead prefetch state (charter e-les G3).
    // Per (layer, expert) single byte, armed once per decode step (from the previous layer's
    // G2 cgc_les_topk) and consumed (cleared) the first time this layer's ensure_slot touches the
    // expert:
    //   0 = not predicted this step
    //   1 = predicted, but already resident at arm time (no read needed)
    //   2 = predicted and queued to a free-slot bg fill (read in flight)
    //   3 = bg fill published the LES-predicted expert before the layer's ensure (window hit)
    std::vector<std::vector<uint8_t>> les_arm;       // [layer][expert]
```

### H2 — 新增計數器

**找**（約 453-454 行）：
```cpp
    size_t n_prefetch = 0;          // pool prefetches queued to the bg thread
    size_t n_prefetch_dropped = 0;  // skipped (queue full / no free slot / already resident) — TOTAL of all drop reasons below
```
**在其後新增**：
```cpp
    // [G3a LES] next-router lookahead prefetch effectiveness (default off, gated CGC_LES_PREFETCH).
    // All four writers (prefetch_les, bg publish, ensure hit, ensure miss) hold mutex m, so plain
    // size_t is sufficient — no atomics.
    size_t n_les_armed      = 0;  // distinct experts in LES predictions (incl. already-cached)
    size_t n_les_cached     = 0;  // predicted experts already resident at arm time (state 1)
    size_t n_les_queued     = 0;  // predicted experts queued to a free-slot bg fill (state 2)
    size_t n_les_nofree     = 0;  // predicted experts skipped: no free slot (free-only => no evict)
    size_t n_les_published  = 0;  // bg fills that published an LES-queued expert (state 2 -> 3)
    size_t n_les_used_pf    = 0;  // ensure HIT, no wait, on a LES-published expert (read fully hidden)
    size_t n_les_used_pf_w  = 0;  // ensure HIT but had to wait for the LES fill (read not fully hidden)
    size_t n_les_was_cached = 0;  // ensure HIT on a predicted expert that was already hot at arm time
    size_t n_les_arm_miss   = 0;  // ensure MISS on a predicted expert (read did not fit in window)
```

### H3 — 新增 `prefetch_les` 宣告

**找**（約 1049-1050 行）：
```cpp
int32_t llama_expert_cache_prefetch_batch(llama_expert_cache * cache, uint32_t layer,
                                          const uint32_t * experts, size_t n);
```
**在其後新增**：
```cpp
// [G3a LES 2026-10-18] next-router lookahead prefetch (env CGC_LES_PREFETCH=1). Free-slot-ONLY:
// never evicts for a prediction (zero churn by construction). `experts` are the G2 in-graph
// argsort DESC top-k for TARGET `layer` (il+1). Marks each expert in les_arm, skips already
// resident (state 1), claims only FREE slots for the rest (state 2) and enqueues one merged
// preadv batch. Non-blocking: a full queue or no free slot drops (counted), never blocks.
// Returns the number of experts queued (cached ones not counted); 0 when nothing queued.
int32_t llama_expert_cache_prefetch_les(llama_expert_cache * cache, uint32_t layer,
                                        const uint32_t * experts, size_t n);
```

---

## 2. `src/llama.cpp/src/llama-expert-cache.cpp`

### C1 — `prefetch_claim_slot_locked` 加 `free_only`

**找（函數簽名，約 1847 行）**：
```cpp
static int32_t prefetch_claim_slot_locked(llama_expert_cache * cache, uint32_t layer, uint32_t expert) {
    int32_t slot = -1;
    // [CGC MTP fast path] skip the reserved ZERO slot (never assigned to a real expert).
    const uint32_t ns = llama_expert_cache_usable_slots(cache, layer);
```
**替換為**：
```cpp
static int32_t prefetch_claim_slot_locked(llama_expert_cache * cache, uint32_t layer, uint32_t expert,
                                          bool free_only = false) {
    int32_t slot = -1;
    // [CGC MTP fast path] skip the reserved ZERO slot (never assigned to a real expert).
    const uint32_t ns = llama_expert_cache_usable_slots(cache, layer);
```

**再找（緊接其後的 free 迴圈結束 + evict 入口，約 1856-1857 行）**：
```cpp
            break;
        }
    }
    if (slot < 0) {
        // No free slot: evict the layer's global LRU slot (the union ensure just bumped every
```
**替換為**（在 `if (slot < 0) {` 之後插入 G3a 早返回；其餘 evict 塊原樣不動）：
```cpp
            break;
        }
    }
    if (slot < 0 && free_only) {
        // [G3a LES] a prediction must NEVER evict (zero churn). No free slot means drop this
        // expert; the layer's own ensure will still fill it synchronously if it is really used.
        // Counted under the same #1 reason the evict path uses when it finds nothing.
        cache->n_prefetch_dropped++;
        cache->drop_stats.no_free_slot++;
        return -1;
    }
    if (slot < 0) {
        // No free slot: evict the layer's global LRU slot (the union ensure just bumped every
```

> 現有 `prefetch_slot` / `prefetch_batch` 調用此函數時用預設 `free_only=false`，行為逐字不變。

### C2 — 新增 `llama_expert_cache_prefetch_les`

**找（`prefetch_batch` 結尾，約 2033-2035 行）**：
```cpp
    cache->pool_batch_queue.emplace_back(layer, std::move(q_experts), std::move(q_slots));
    cache->n_prefetch += q_experts.size();
    lk.unlock();
    cache->bg_cv.notify_one();
    return (int32_t) q_experts.size();
}
```
> 該段在 `llama_expert_cache_prefetch_batch` 裡。緊接其**函數閉合 `}` 之後、`llama_expert_cache_prerouter_predict` 註釋之前**，新增整個函數：
```cpp

// [G3a LES 2026-10-18] next-router lookahead prefetch. Free-slot-only (zero eviction => zero
// churn). Called from the graph eval callback when the G2 in-graph cgc_les_topk-<il> vertex
// dispatches: `layer` is the TARGET layer (il+1), `experts` are the DESC top-k predictions.
// Non-blocking throughout; a full queue or no free slot only drops (counted).
int32_t llama_expert_cache_prefetch_les(llama_expert_cache * cache, uint32_t layer,
                                        const uint32_t * experts, size_t n) {
    if (cache == nullptr || !cache->pool_active || experts == nullptr || n == 0) {
        return 0;
    }
    if (layer >= cache->slot_owner.size() || layer >= cache->les_arm.size()) {
        return 0;
    }
    std::unique_lock<std::mutex> lk(cache->m);
    // Reuse the same in-flight depth cap as the rho batch path.
    const int32_t rho_maxq = cgc_rho_maxq();
    if (rho_maxq > 0 && (long) cache->pool_batch_queue.size() >= rho_maxq) {
        // Backpressure: drop the whole arm this step (no mark left), so ensure counts it honestly
        // as a miss rather than as a prediction that never existed. Non-blocking.
        return 0;
    }
    // Each target layer is armed exactly once per decode step (from the previous layer's topk).
    // Clear the prior step's marks before recording this step's prediction. By the time we arm
    // layer L in token t, every mark from token t-1 for L has already been consumed by L's ensure.
    std::fill(cache->les_arm[layer].begin(), cache->les_arm[layer].end(), (uint8_t) 0);

    const int32_t * table = cache->slot_table.data() + (size_t) layer * cache->n_expert;
    std::vector<uint32_t> q_experts;
    std::vector<int32_t>  q_slots;
    q_experts.reserve(n);
    q_slots.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t expert = experts[i];
        if (expert >= cache->n_expert) {
            continue;
        }
        if (cache->les_arm[layer][expert] != 0) {
            continue; // dedup within the prediction
        }
        cache->n_les_armed++;
        if (table[expert] >= 0) {
            // Already resident: no read needed; ensure classifies this as n_les_was_cached.
            cache->les_arm[layer][expert] = 1;
            cache->n_les_cached++;
            continue;
        }
        const int32_t slot = prefetch_claim_slot_locked(cache, layer, expert, /*free_only=*/true);
        if (slot < 0) {
            // No free slot and free-only => drop reason counted in the helper; do not mark, so the
            // later ensure never credits a prediction we could not actually arm.
            cache->n_les_nofree++;
            continue;
        }
        cache->slot_queued[layer][slot] = 1;
        cache->slot_owner[layer][slot]  = (int32_t) expert;
        cache->les_arm[layer][expert]  = 2;
        cache->n_les_queued++;
        q_experts.push_back(expert);
        q_slots.push_back(slot);
    }
    if (q_experts.empty()) {
        return 0;
    }
    cache->pool_batch_queue.emplace_back(layer, std::move(q_experts), std::move(q_slots));
    cache->n_prefetch += q_experts.size();
    lk.unlock();
    cache->bg_cv.notify_one();
    return (int32_t) q_experts.size();
}
```

### C3 — bg 發布時把 LES 狀態 2→3

**找（bg_loop `has_batch` 三段式 phase 3，batch 發布迴圈，約 3822-3830 行）**。注意這是 **batch 分支**（變量 `blayer`/`exp`/`live`），不是下面的 single-fill 分支（`layer`/`expert`/`slot`、`n_published++`）：
```cpp
                    if (exp < n_expert) {
                        if (cgc_dbuf2_on()) {
                            slot_table_scratch[(size_t) blayer * n_expert + exp] = slot;
                        } else {
                            slot_table[(size_t) blayer * n_expert + exp] = slot;
                        }
                    }
                }
                bg_cv.notify_all();
```
**替換為**（在 `if (exp < n_expert) { ... }` 內、發布表寫入後加狀態轉換；bg_loop 是成員函數，用裸欄位名）：
```cpp
                    if (exp < n_expert) {
                        if (cgc_dbuf2_on()) {
                            slot_table_scratch[(size_t) blayer * n_expert + exp] = slot;
                        } else {
                            slot_table[(size_t) blayer * n_expert + exp] = slot;
                        }
                        // [G3a LES] this merged batch just published an expert a lookahead arm
                        // queued: it is now resident before the target layer's ensure (window hit).
                        if (blayer < les_arm.size() && les_arm[blayer][exp] == 2) {
                            les_arm[blayer][exp] = 3;
                            ++n_les_published;
                        }
                    }
                }
                bg_cv.notify_all();
```

> 錨點在 `has_batch` 的 phase 3（解包 `blayer/bexps/bslots` 約 3756-3758，merged pread 於 3801，發布迴圈 3813-3829）。DBUF2 預設 OFF 走 active 分支，但狀態轉換放在 `if (exp < n_expert)` 內、兩個發布表分支之後，故 active/scratch 都覆蓋。

### C4 — `ensure_slot` 命中分支歸因

**找（約 1214-1227 行）**：
```cpp
    if (table[expert] >= 0) {
        const int32_t slot = table[expert];
        if (cache->slot_loading[layer][slot] || cache->slot_queued[layer][slot]) {
            // prefetch queued/in flight: wait for the bg thread to finish the fill (it clears
            // the flags and notifies). The pread is hidden behind the current layer's FFN.
            const int64_t fw0 = ggml_time_us();
            cache->bg_cv.wait(lk, [&]{ return !cache->slot_loading[layer][slot] && !cache->slot_queued[layer][slot]; });
            const uint64_t cgc_fw = (uint64_t) (ggml_time_us() - fw0);
            cache->fill_wait_us.fetch_add(cgc_fw, std::memory_order_relaxed);
            cache->fill_wait_by_phase[cgc_phase_idx(cache)].fetch_add(cgc_fw, std::memory_order_relaxed);
        }
        (count ? cache->n_hits : cache->n_prewarm_hits)++;
        cache->slot_last_use[layer][slot] = ++cache->tick;
        return slot;
    }
```
**替換為**：
```cpp
    if (table[expert] >= 0) {
        const int32_t slot = table[expert];
        bool cgc_les_waited = false;
        if (cache->slot_loading[layer][slot] || cache->slot_queued[layer][slot]) {
            // prefetch queued/in flight: wait for the bg thread to finish the fill (it clears
            // the flags and notifies). The pread is hidden behind the current layer's FFN.
            cgc_les_waited = true;
            const int64_t fw0 = ggml_time_us();
            cache->bg_cv.wait(lk, [&]{ return !cache->slot_loading[layer][slot] && !cache->slot_queued[layer][slot]; });
            const uint64_t cgc_fw = (uint64_t) (ggml_time_us() - fw0);
            cache->fill_wait_us.fetch_add(cgc_fw, std::memory_order_relaxed);
            cache->fill_wait_by_phase[cgc_phase_idx(cache)].fetch_add(cgc_fw, std::memory_order_relaxed);
        }
        (count ? cache->n_hits : cache->n_prewarm_hits)++;
        cache->slot_last_use[layer][slot] = ++cache->tick;
        // [G3a LES] classify this demand hit against the current-step lookahead prediction and
        // consume the mark (decode demand path only; prewarm count=false must not touch it).
        if (count && layer < cache->les_arm.size() && expert < cache->n_expert) {
            uint8_t & la = cache->les_arm[layer][expert];
            if (la == 3) {
                if (cgc_les_waited) { ++cache->n_les_used_pf_w; } else { ++cache->n_les_used_pf; }
            } else if (la == 1) {
                ++cache->n_les_was_cached;
            }
            la = 0;
        }
        return slot;
    }
```

### C5 — `ensure_slot` miss 分支歸因

**找（約 1229 行，緊接命中分支之後）**：
```cpp
    (count ? cache->n_misses : cache->n_prewarm_misses)++;
    if (count && expert < cache->n_expert) {
```
**替換為**（在 miss 計數後、compulsory/capacity 歸因之前插入）：
```cpp
    (count ? cache->n_misses : cache->n_prewarm_misses)++;
    // [G3a LES] a predicted expert (state 2 = queued, or 1) still absent at ensure means the read
    // did NOT fit inside the previous layer's FFN window (the core G3a question). Consume mark.
    if (count && layer < cache->les_arm.size() && expert < cache->n_expert) {
        uint8_t & la = cache->les_arm[layer][expert];
        if (la == 2 || la == 1) { ++cache->n_les_arm_miss; }
        la = 0;
    }
    if (count && expert < cache->n_expert) {
```

### C6 — 初始化 `les_arm`

**找（約 4393 行）**：
```cpp
        // [CGC miss attribution 2026-09-13] session-long demand bitmaps (41 x 256 bytes here).
        cache->ever_loaded.assign(max_layer, std::vector<uint8_t>(cache->n_expert, 0));
        cache->n_distinct_demanded.assign(max_layer, 0);
```
**在其後新增**：
```cpp
        // [G3a LES] per-(layer,expert) lookahead state, zeroed each session/arm.
        cache->les_arm.assign(max_layer, std::vector<uint8_t>(cache->n_expert, 0));
```

### C7 — 析構統計行 `CGC-LES-G3`

**找（析構函數 `~llama_expert_cache` 的 final stats fprintf 結尾，約 3089 行）**：
```cpp
                (unsigned long long) fill_wait_us.load(std::memory_order_relaxed),
                n_prefetch, n_prefetch_dropped);
```
**在其後新增**（同一個 `std::lock_guard` 區塊內）：
```cpp
        // [G3a LES 2026-10-18] only printed when a lookahead arm actually ran. used_pf = the reads
        // fully hidden in the window (no wait); used_pf_w = caught but ensure still had to wait;
        // arm_miss = predicted but not ready in time (the read-overrun G3b targets).
        if (n_les_armed > 0) {
            const size_t n_les_used = n_les_used_pf + n_les_used_pf_w + n_les_was_cached;
            const double eff = n_les_armed > 0
                ? 100.0 * (double) n_les_used / (double) n_les_armed : 0.0;
            fprintf(stderr,
                    "CGC-LES-G3: armed=%zu cached=%zu queued=%zu nofree=%zu published=%zu "
                    "used_pf=%zu used_pf_waited=%zu was_cached=%zu arm_miss=%zu effective_hit=%.1f%%\n",
                    n_les_armed, n_les_cached, n_les_queued, n_les_nofree, n_les_published,
                    n_les_used_pf, n_les_used_pf_w, n_les_was_cached, n_les_arm_miss, eff);
        }
```

---

## 3. `src/llama.cpp/src/llama-context.h`

### H4 — 宣告 `cgc_les_arm_from_topk`

**找（約 296 行）**：
```cpp
    void cgc_rho_prefetch(int il);

    // Wrapper installed as cparams.cb_eval (static so it can be passed to
```
**替換為**：
```cpp
    void cgc_rho_prefetch(int il);

    // [G3a LES 2026-10-18] arm the next-layer background preadv from the G2 in-graph
    // cgc_les_topk-<src_il> vertex (I32 DESC top-k). Called from the sched eval callback at
    // !ask (after the vertex dispatches, before src_il's FFN). Decode / trunk DEFAULT only.
    void cgc_les_arm_from_topk(int src_il, const ggml_tensor * t);

    // Wrapper installed as cparams.cb_eval (static so it can be passed to
```

---

## 4. `src/llama.cpp/src/llama-context.cpp`

### X1 — eval_cb 新增 in-cb arm 分支

**找（`expert_cache_eval_cb` 內，G1 的 `cgc_les_h` 塊閉合之後、緊接 ρ probe 聲明，約 6422-6423 行）**。注意是頂層 4 空格縮進：
```cpp
    }
    static const bool cgc_rho_probe = getenv("CGC_RHO_PROBE") != nullptr;
```
**替換為**（在兩行之間插入 G3a 分支，保持 4 空格縮進）：
```cpp
    }

    // [G3a LES 2026-10-18] in-graph arm of the next-layer background preadv. The G2 vertex
    // cgc_les_topk-<il> is built from attn_post_norm(il) through W_gate[il+1] and sits BEFORE
    // layer il's FFN, so at !ask (the vertex has dispatched) the reads for layer il+1 can be
    // launched to overlap layer il's FFN compute. Decode-only vertex (G2 gates it on
    // n_tokens==1), trunk DEFAULT ctx only. Default off: vertex absent, byte-identical.
    static const bool cgc_les_pf_on =
        getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH") != nullptr &&
        getenv("LLAMA_MOE_LOOKAHEAD_PREFETCH")[0] != '0' &&
        getenv("CGC_LES_PREFETCH") != nullptr &&
        getenv("CGC_LES_PREFETCH")[0] != '0';
    if (cgc_les_pf_on && !ask &&
        ctx->cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
        strncmp(t->name, "cgc_les_topk", 12) == 0) {  // prefix len = strlen("cgc_les_topk")
        const char * les_dash = strrchr(t->name, '-');
        const int les_src_il = les_dash ? atoi(les_dash + 1) : -1;
        ctx->cgc_les_arm_from_topk(les_src_il, t);
    }

    static const bool cgc_rho_probe = getenv("CGC_RHO_PROBE") != nullptr;
```

### X2 — 實作 `cgc_les_arm_from_topk`

**找（`cgc_rho_prefetch` 定義行，約 6299 行）**：
```cpp
void llama_context::cgc_rho_prefetch(int il) {
```
**在其「前」插入整個新函數**（即「替換前」保留，在前面貼上下面這段）：
```cpp
// [G3a LES 2026-10-18] Pull the G2 in-graph predicted top-k (I32 DESC, [depth,1]) and enqueue a
// free-slot-only background fill for the TARGET layer src_il+1 so its reads overlap src_il's FFN.
// The tensor_get is a Metal sync point (32 B / 8 experts); G3a measures its cost via fill_wait
// and t/s (expected negligible, but not assumed free). Non-blocking; decode/trunk only.
void llama_context::cgc_les_arm_from_topk(int src_il, const ggml_tensor * t) {
    llama_expert_cache * cache = model.expert_cache;
    if (cache == nullptr || t == nullptr || !llama_expert_cache_pool_active(cache)) {
        return;
    }
    const int target_il = src_il + 1;
    if (src_il < 0 || target_il >= (int) model.hparams.n_layer_all) {
        return;
    }
    if (target_il < 4) {
        return; // G1 verdict: exempt the first 3 layers (aligned with G2 cgc_les_topk_capture)
    }
    if (t->type != GGML_TYPE_I32 || t->ne[0] <= 0 || t->ne[0] > 64) {
        return;
    }
    if (cache->n_expert > 256) {
        return; // the static seen[]/want[] below are capped at 256 experts; this model E=256
    }
    const int depth = (int) t->ne[0];
    int read_n = depth;
    if (const char * r = getenv("CGC_LES_PREFETCH_READ")) {
        read_n = atoi(r); // truncate to the top-N highest-confidence predictions (G3b sweep)
    }
    if (read_n < 0)     read_n = 0;
    if (read_n > depth) read_n = depth;

    int32_t stk[64];
    // ggml_argsort_top_k returns DESC indices (G2 §2.1): stk[0..read_n-1] are the strongest.
    ggml_backend_tensor_get(t, stk, 0, (size_t) depth * sizeof(int32_t));

    uint32_t want[64];
    size_t   n = 0;
    bool     seen[256] = {false};
    for (int j = 0; j < read_n; ++j) {
        const int32_t e = stk[j];
        if (e < 0 || e >= (int) cache->n_expert || seen[e]) {
            continue;
        }
        seen[e] = true;
        want[n++] = (uint32_t) e;
    }
    if (n == 0) {
        return;
    }
    (void) llama_expert_cache_prefetch_les(cache, (uint32_t) target_il, want, n);
}

```

---

## 5. 編譯

```bash
cd /Users/alexchuang/Documents/flashkv-devserver
export PATH="/opt/homebrew/bin:$PATH"
cmake --build src/llama.cpp/build --parallel 8
```

**階段 0 — 預設全關自證（必須先過）**：不設任何 LES env 啟動，stderr 不得出現 `CGC-LES-G3`、`cgc_les_topk`；跑 cell contract M1/M2（`CGC_MM_BITIDENT=1`）須與 G2 基線**逐位元相同**（coverage 100%、comparable=true）。

---

## 6. G3a 觀察臂（乾淨窗口，harness 唯一入口，warm-skip 64）

在 C17 全套 env（`prod-new:CGC_OVERLAP_FENCE=41`、MTP off）之上疊加：

| 臂 | 額外 env | 目的 |
|---|---|---|
| G2-only 對照 | `LLAMA_MOE_LOOKAHEAD_PREFETCH=1;LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH=8`（**不設** `CGC_LES_PREFETCH`） | 只預測、不讀盤：構圖開銷基線 |
| **G3a** | `LLAMA_MOE_LOOKAHEAD_PREFETCH=1;LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH=8;CGC_LES_PREFETCH=1;CGC_LES_PREFETCH_READ=8;CGC_RHO_PREFETCH_MAXQ=16`（不設 `CGC_LES_PREFETCH_EVICT`） | 圖內 arm + free-slot-only serial preadv |

> 不設 `CGC_LES_PREFETCH_PARALLEL`（那是 G3b）、不設 `CGC_LES_PREFETCH_EVICT`（那是 G3c）。LES 與 ρ 互斥：G3a 臂不設任何 `CGC_RHO`/`CGC_RHO_FILL`。

harness 參數同 G6：`--warm-skip 64 --prompt 512 --gen 128 --depths 0 --reps 5 --fixed-fill-seed 1`，三套 prompt（code/zh_chat/math）各跑 G2-only / G3a，外加一臂全關 C17 基線。

---

## 7. 驗收（對齊 charter G3，全過才進 G3b/G6）

進程結束時抓 stderr 的 `CGC-LES-G3:` 行（三套 prompt 各一），對照 cache JSON：

1. **機制確實 arm 且讀完**：`armed>0`、`queued>0`、`published>0`；`nofree` 在 8G 池應接近 0（若大量 nofree，代表池已滿，先看 G6 容量，不進 G3b）。
2. **窗口命中率有效**：`used_pf`（零等待命中）佔 `queued` 的比例高，且 `used_pf + used_pf_waited + was_cached` 對 `armed` 的 effective_hit 落在 **G1 離線 0.79–0.82 帶**（明顯偏低 → arm 時機/tensor_get 同步有 bug，不是命中率問題）。
3. **讀不完的量（決定 G3b）**：`arm_miss`（預測對卻沒趕上）佔 `queued` 比例。接近 0 ⇒ serial merged preadv 已在窗口內讀完，**G3b 不用做**；偏高 ⇒ 才上 G3b（先 `CGC_LES_PREFETCH_READ` 截斷 6/4，再多 worker）。
4. **零 churn（free-only 硬門檻）**：G3a 臂 vs G2-only 臂，cache JSON 的 `miss_capacity` 不升、`hit_rate_pct` 不降；`evictions` 不因 LES 增加（free-only 不應 evict 任何槽）。
5. **不增讀取**：`file_reads` 每 token 相比 G2-only 增幅 ≤10%（預取錯位的代價；LES 命中率 0.8 下應遠低於此）。
6. **速度**：G3a vs G2-only 同場 warm-skip64 decode t/s 中位數應**不降**（構圖+同步成本被隱藏）；`fill_wait_us` 中位數下降才代表真正把 SSD miss 藏進窗口。
7. **bit-exact**：兩臂輸出 token 與 C17 一致（LES 只改預取時機、不回饋路由數學）。

---

## 8. 回滾

- 執行期回滾：`unset LLAMA_MOE_LOOKAHEAD_PREFETCH CGC_LES_PREFETCH CGC_LES_PREFETCH_READ CGC_LES_PREFETCH_PARALLEL CGC_LES_PREFETCH_EVICT`（預設即全關）。
- 代碼回滾：四檔皆在獨立 env 閘/新欄位內，`git checkout -- src/llama.cpp/src/llama-expert-cache.{h,cpp} src/llama.cpp/src/llama-context.{h,cpp}` 移除本補丁（注意先確認沒有混入他人未提交 hunk，見下）。

## 9. 與 commit 邊界 / 工作區的配合（重要）

- apply 前確認 `src/` 沒有別的未提交改動會與本補丁同檔衝突（特別是 context.cpp 既有的 7 個 affinity hunk）。建議在 commit2 落地、G2 apply 後，`git status` 乾淨再逐檔 apply 本補丁。
- 本補丁**不新增線程、不新增雙緩衝、不 evict、不碰 MTP ctx/verify union**；所有新計數器在 mutex `m` 內自增，無 atomic 需求。
- 預設 OFF 下：`les_arm` 全 0、`prefetch_les` 永不被調用（cb 分支閘關）、`prefetch_claim_slot_locked` 走 `free_only=false` 原路徑、ensure 歸因塊讀到 `la==0` 不計數——對基線零行為差異。

## 10. 已知限制（誠實）

- in-cb 的 `ggml_backend_tensor_get` 是強同步點，G3a 必須用 t/s 與 fill_wait 量它；若它使 decode 退化 >2%，再考慮把 topk 結果落 host（見 G3 施工方案 §4A.1 可選優化）。
- `arm_miss` 觸發時，ensure 的同步 fill 與尚未完成的 bg fill 可能短暫競態；現有 bg phase1 validate（reassign-race）會讓 bg 放棄重複讀，不致重複讀取，但該專家的預取收益作廢（如實計為 arm_miss）。
- G3a 不回答「25 t/s」，只回答「一層 FFN 窗口裡 serial merged preadv 能藏住幾個專家」；其餘 G3b/G3c/G6 按 §7 數據閘門決定。
