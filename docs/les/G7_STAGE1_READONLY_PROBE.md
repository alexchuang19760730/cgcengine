# G7（C 線階段 1）：L4 冷頁歸還的只讀插樁 — fence 安全窗口 + 頁/slot 幾何

> 狀態：**只讀、env-gated、預設 OFF，不碰任何決策路徑**。補丁已可乾淨 apply（`git apply --check` rc=0），
> 已用 `compile_commands.json` 真實編譯旗標對臨時副本做 `-fsyntax-only`：**rc=0、零新增 warning**
> （基線 4 個舊 warning，補丁後仍 4 個，新增行 0 warning）。
> 前置：**commit2 落地 → 撤 affinity 之後**才 apply（見末尾「apply 順序」）。
> 補丁檔：`docs/les/G7_STAGE1_READONLY_PROBE.patch`（只改 `llama-expert-cache.cpp` 一個檔）。
> 離線分析：`docs/les/tools/l4_slab_geometry.py`。

---

## 1. 階段 1 要回答的 4 個問題，與現有結論

階段 2 的冷頁歸還（在固定大 Metal shared buffer 上對冷頁 `MADV_FREE_REUSABLE`，已由獨立 Metal 探針
證實 phys_footprint 立即下降、GPU 不 hang、重填 bit-exact）要落到引擎，先得證明 4 件事。下表標註
哪些已由**靜態讀碼釘死**、哪些必須**跑一次乾淨視窗**確認。

| # | 問題 | 結論 | 依據 |
|---|---|---|---|
| ② | per-kind stride 是否 16KiB 頁整數倍 | **否（已釘死）**：`stride = expert_bytes = ggml_row_size(type,ne0)*ne1`，**無任何 round up**；註釋明寫 1.0703 MiB = 68.5 個 16KiB 頁 | `llama-model-loader.cpp:1504/1507`；`cgc_discard_pages` 註釋 |
| ③ | layer 切片 base 是否頁對齊 | **池 base 對齊、layer 內部分片可能半頁（需運行時 per-kind 實值）**：pool buffer 經 Metal host `vm_allocate`（整頁對齊）；layer base = base + Σ(slots×stride)，stride 非頁整數倍 → 後續層 base 可能偏移半頁 | `llama.cpp:524-545`；`pool_split_adopt` |
| ④ | fill 是否全覆寫 slot（FREEREUS 後重填不殘留） | **是（已釘死）**：fill 按 `key_segs` 覆蓋該專家整個權重張量分片，註釋「Every dst ... overwritten in full by pread」；短讀一律 `memset(0)`（可偵測、不靜默 stale） | `fill_pool_direct_collect`；merged fill 4042-4052；bg fill 3799-3802 |
| ① | evict 點是否存在 GPU 仍讀 victim 的安全窗口 | **靜態強烈支持安全，需運行時確認執行緒身份**：L4 `pool_ext` 的回收（`cgc_discard_pool_slot`）只經 `pick_slot`（`ensure_slot`/`ensure_batch`，build 線程、圖提交前）；bg 線程的 `evict_lru` 只碰舊 malloc `map`，**不碰 pool_ext** | 見下方 §2 時序論證；運行時以 `CGC_L4_EVICT_TRACE` 驗 distinct 線程數 |

**階段 1 真正要跑的只有兩件事**：
1. `CGC_POOL_SPLIT_DBG=1`（補丁在其 GEOM 行補了頁對齊字段）→ 離線腳本算每個 kind 的回收粒度；
2. `CGC_L4_EVICT_TRACE=1`（補丁新增）→ 確認一整輪 decode 中 evict 只在單一 build 線程。

---

## 2. ① fence 安全窗口的靜態時序論證（先讀，再用運行時背書）

- **L4 pool_ext 的實際驅逐只在 build 線程**：
  - `cgc_discard_pool_slot` 唯一調用點在 `pick_slot` 內（`llama-expert-cache.cpp:884`）；
  - `pick_slot` 只被 `ensure_slot`（:1248）與 `ensure_batch`（:1602）調用，都是同步 build 路徑；
  - bg 線程 `bg_loop` 的三個分支（has_batch / has_pool / key）**只拿 build 線程預先分配好的
    `(layer, slot, expert)` 去 pread**，自己不 `pick_slot`；唯一的 bg 側驅逐 `evict_lru`（:3902）
    只遍歷舊 `cache->map` 的 `blob`（malloc），**不調 `cgc_discard_pool_slot`、不碰 pool_ext**。
- **victim 與當前 dispatch 同層**：`ensure_batch` 在 `graph_compute` 內逐層處理；為 layer L 補洞時
  驅逐的 victim 是 **L 層**的槽，而 L 層 FFN 要等 `drain_layer`（outstanding==0）後才 dispatch。
  即 victim 被回收時，當前步 L 層 GPU 尚未讀它。
- **上一步 GPU 已完成（decode 串行）**：decode 單 token 需上一步 logits 才能選路由，取 logits 強制
  command buffer 完成；故 build 新一步時，上一步讀過 victim 的命令已結束。
- **唯一例外是異步流水**（OA_ASYNC / PREFILL_STREAM 主要作用於 prefill；MTP 已按 G5 口徑 **off**）。
  這就是為什麼運行時還要確認 distinct 線程數，且階段 2 必須用 bit-identical + swap 壓力測試最終背書。

判定：**一整輪 decode 中 `CGC-L4-EVICT-TRACE` 只有 1 個 distinct `tid`（build 線程）即通過**；
若出現第 2 個帶大量 evict 的 `tid`（worker/bg/metal），階段 2 不得在 evict 點直接 FREEREUS，必須先
加 GPU fence。

---

## 3. 補丁內容（兩個 hunk，均只讀、預設 OFF）

### Hunk A — 幾何 dump（增強現有 `CGC_POOL_SPLIT_DBG`，僅在該 env 設置時多打一行）

在 `llama_expert_cache_pool_split_adopt` 的 DBG 分支末尾，原 ADOPT 行之後加：

```c
fprintf(stderr, "CGC-POOL-SPLIT-GEOM: layer=%u kind=%d page=%zu base_mod_pg=%zu "
        "stride_mod_pg=%zu slice_mod_pg=%zu stride_pg=%.4f\n",
        layer, kind, cgc_pg,
        (size_t)((uintptr_t) base % cgc_pg),
        stride % cgc_pg,
        cgc_slice % cgc_pg,
        cgc_pg ? (double) stride / (double) cgc_pg : 0.0);
```

### Hunk B — evict 線程 trace（`CGC_L4_EVICT_TRACE`，預設 OFF）

在 `cgc_discard_pool_slot` 函數開頭插入一段限頻統計：用 `pthread_self()` 記錄調用線程 id，
固定 16 槽表 + mutex 計數，每線程前 3 次及每 2048 次打一行 `CGC-L4-EVICT-TRACE`（含 `[NEW THREAD]`）。
不改變其後任何 discard 決策。

> 完整可 apply 內容見 `docs/les/G7_STAGE1_READONLY_PROBE.patch`（63 行，1 個檔）。
> 僅用到已 include 的 `pthread.h`、`std::mutex`（檔內已大量使用）、`sysconf(_SC_PAGESIZE)`，無新增 include。

### 合成幾何預覽（離線腳本自測，stride=68.5 頁、143 槽/層、40 層）

```
覆蓋 1 個 slot 的頁 : 388960 頁 (99.3%)
覆蓋 2 個 slot 的頁 :  2860 頁 ( 0.7%)   # 一頁最多橫跨 2 個 slot
跨 layer 邊界頁 20 個
同層內：每 2 個連續 slot 寬度恰好頁對齊
```

工程含義：**99.3% 的頁只屬於單一 slot（該 slot 冷即可回收）**，僅 0.7% 的共享頁需等相鄰 slot
（偶爾跨層）也冷。階段 2 採「按絕對 16KiB 頁、頁 owner 集合全空才 FREEREUS」即可，開銷很小。
真實 per-kind 數值（gate/up/down 各異）以乾淨視窗 GEOM 輸出為準。

---

## 4. 乾淨視窗：apply → 編譯 → 跑一格 → 離線分析

### 4.1 前置與 apply（順序不可顛倒）

```bash
cd /Users/alexchuang/Documents/flashkv-devserver
# 1) commit2 已落地（含 D 的 budget v3；提交前 git status / git diff --staged 核對）
# 2) 8 檔撤 affinity 已完成（見 ~/LES_APPLY_RUNBOOK.md §2，精確只刪、保留 context.cpp bhead/les s2c）
# 3) 才 apply 本只讀補丁
git apply --check docs/les/G7_STAGE1_READONLY_PROBE.patch   # 期望 rc=0
git apply        docs/les/G7_STAGE1_READONLY_PROBE.patch
```

> 與 G2/G3a 不沖突：本補丁只改 `cgc_discard_pool_slot` 開頭與 `pool_split_adopt` 的 DBG 分支，
> 不在 G2 argsort / G3a prefetch 的行上。若 G2/G3a 先 apply，apply 前重跑一次 `git apply --check`，
> 行號漂移時以 hunk 錨點字串為準（補丁是語義上下文匹配）。

### 4.2 編譯

```bash
export PATH=/opt/homebrew/bin:$PATH
cmake --build src/llama.cpp/build --parallel 8 2>&1 | tee /tmp/g7_build.log
```

### 4.3 跑一格（加在 prod-new；可直接借用 G6 的 8g-off 那一格，一次啟動同時收幾何+evict）

啟動 server 時**追加兩個 env**（其餘 prod-new 不變；`CGC_POOL_MADVISE=2` 是 evict trace 觸發前提，
prod-new 已含），stderr 存檔：

```bash
CGC_POOL_SPLIT_DBG=1 CGC_L4_EVICT_TRACE=1  \
# …其餘 prod-new env（CGC_POOL_MADVISE=2 等）… 2> /tmp/g7_stderr.log
# 接著跑 warm-skip64 同一套 decode（無需測速，只要觸發足量 evict；順便可作 G6 一臂）
```

### 4.4 離線分析

```bash
# 幾何：per-kind 頁對齊 / 共享頁比例 / 回收 slab 粒度
python3 docs/les/tools/l4_slab_geometry.py /tmp/g7_stderr.log | tee /tmp/g7_geom.txt

# evict 線程：distinct tid 數（期望 1）與每線程 evict 數
grep CGC-L4-EVICT-TRACE /tmp/g7_stderr.log | grep -oE 'tid=[0-9]+' | sort | uniq -c
grep -c 'CGC-L4-EVICT-TRACE.*\[NEW THREAD\]' /tmp/g7_stderr.log
```

---

## 5. 進階段 2 的門檻（全部滿足才做 FREEREUS）

1. `l4_slab_geometry.py` 已給出每個 kind 的真實 stride/slots/base 對齊與共享頁比例、slab 粒度；
2. evict trace **distinct tid == 1**（全 build 線程）；若 >1，階段 2 先補 GPU fence；
3. 階段 2 落地後仍須：**全關 bit-identical 自證**（CGC_MM_BITIDENT=1）+ swap 壓力/長時並發測試，
   確認 FREEREUS 冷頁被 GPU 重讀時不 hang、重填 bit-exact（獨立探針已驗證單次，引擎高頻 evict 需複測）。

## 6. 本階段明確不做

- 不寫任何 FREEREUS / REUSE / 改 madvise 的程式碼（那是階段 2，方案見
  `docs/les/L4_COLD_PAGE_RECLAIM_PROBE_2026-10-18.md` §8）；
- 不改任何 fill / evict / 路由決策，補丁關閉時與主線 byte-identical；
- 不替 G2/G3a/G6 的測速結論背書，不承諾 <6GB 或 25 t/s。
