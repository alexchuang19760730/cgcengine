# Swap 結構解：機制定量與 L1–L4 決策

日期：2026-09-24（凌晨，重開機後乾淨窗口）· 方法：footprint/vm_stat 實測 + 三趟 prod-new run 重現
口徑：16GB MacBook Air M4（Mac16,12，10 核）；模型 13.65GB；prod-new 完整 env；8G pool；-p 2048 -n 128 -d 512 -r 2

---

## 0. 一句話根因

**模型檔案 13.6GB 全量 mmap（觸碰即駐留，file-backed）＋ pool 固定 8GB 保留 ＋ Metal wired 1.87GB ＝ 21.6GB 潛在駐留需求，塞進 16GB；macOS swap 只累積、不回收 ⇒ 每跑一次量測 swap 就漲一點，永遠回不去。**

架構解不是「清 swap」（sudo purge / 殺進程都是治標），是把潛在駐留需求壓回物理記憶體內，讓 swap 從結構上不存在。swap 不可追蹤時所有 A/B 都無效——本項目兩天已被反覆證明。

---

## 1. 機制鏈（已定量，三趟 run 全可重現）

```
prefill 2048 token
  → 路由觸碰幾乎所有 256 專家（2048 × 8 expert × 40 層 ≈ 65 萬次訪問）
  → 模型 13.6GB 檔案 mmap 幾乎全量駐留
     （file-backed clean 頁：footprint 隱形、壓力下被回收又重讀）
  + pool 8GB 保留（footprint 最大項：untagged 8621 MB）
  + Metal wired 1.87GB（不可回收）
  = 16GB 超載
  → swap 當場啟動（run 進行中 footprint 已含 609 MB swapped）
```

### 1.1 footprint 分解（llama-bench，8G pool，8.7GB process footprint）

| 類別 | Dirty | Swapped | 是什麼 |
|---|---:|---:|---|
| untagged (VM_ALLOCATE) | 8621 MB | 383 MB | pool 緩衝 + Metal + KV（最大項） |
| MALLOC_LARGE | 161 MB | 158 MB | 大 buffer |
| IOAccelerator | 40 MB + 4 MB | — | Metal/GPU 提交 |
| mapped file / __TEXT | **0 B / 2048 KB** | — | 模型 mmap **不算進 footprint**（file-backed clean） |

**footprint 盲點**：`--load-mode none` 的模型 mmap 頁是 file-backed clean 頁——不計 footprint（clean/reclaimable）卻**真實佔用物理記憶體**。footprint 8.7GB 只是 process 的一部分；加上隱形的 dense 駐留與系統其他，才是 16GB 超載的真實來源。

### 1.2 系統狀態（三趟 run 後，vm_stat）

```
swap used       3892 MB（重開機後基線，仍在漲）
file-backed     244009 頁 ≈ 4.0 GB（含模型 mmap 殘留 + 其他 app）
wired down      114887 頁 ≈ 1.87 GB（不可回收）
free            303956 頁 ≈ 4.97 GB
```

---

## 2. 自動 swap 製造機：slab fills（兩趟完全一致）

| 指標 | run1 | run2 | 含義 |
|---|---:|---:|---|
| 累計填池 | 24.7 GiB | 24.7 GiB | 8G pool 在一次 run 裡被反覆填 |
| 從盤讀 | **19.6 GiB** | **19.6 GiB** | 每趟都重讀 ~20GB 專家權重 |
| 非駐留 share | 44.1% | 44.1% | 填充中 44% 沒留在記憶體 |
| evictions | 3872 | 4037 | 踢出次數 |
| miss 歸因 | compulsory 91.4% / capacity 8.6% | 90.8% / 9.2% | 池子反覆 fill-refill |
| decode hit | 95.6% / 95.4% | 95.4% | 高 hit 但照樣重讀 |

**機制**：SpAc EMA 踢錯專家 ⇒ 每踢錯一次就付一次 1.0703 MiB pread ⇒ 讀到的 file-backed 頁在壓力下變 swap。**高 hit（95%）與高 disk 讀（19.6GiB）並存**——hit 是 request-level 統計，disk 讀是 evict-refill 的代價，兩者不矛盾。

### 2.1 MASSCOV 錯配（池質量的量化）

```
CURRENT membership（run slots=143）: mass coverage mean = 58.6%（min 40.3 / max 77.5）
COUNTERFACTUAL top-K by mass: K=143 → 98.6%（min 93.4 / max 99.9）
錯配 ≈ 40pp：池有 143 槽，今天只服務 58.6% 的 routing mass
```

理想置換（按 mass 靜態釘住 top-143）能讓 8G 池吃到 98.6%——這是 L2/L3 的理論上界。

---

## 3. 四根架構槓桿（決策表）

| 槓桿 | 內容 | 數據支撐 | 收益 | 成本/風險 | 狀態 |
|---|---|---|---|---|---|
| **L1** | pool 上限 = f(物理記憶體預算)：**甜點 = 4G**（見 §L1 實測） | 4/6/8G footprint 實測 5335/6899/8460 MB（線性 ~780MB/GB，無相變）；hit 96 vs ~96.3（差 0.3pp）| 砍 3.1GB footprint（8G→4G）；4GB 平行雙實驗效率翻倍 | 4G 速度代價需乾淨 ABBA 再定（舊 a4_1 8.42 是熱污染，撤回）| **✅ 6G 已補（2026-09-24）** |
| **L2** | expert 權重單一駐留：pool 內 expert 的 mmap 頁 `madvise(DONTNEED)`，或 expert 段不 mmap——權重只在 pool 存在一份 | file-backed 4.0GB 系統殘留 + 19.6GiB 重讀：同一份權重有 mmap 副本 + pool 副本兩份「可能駐留」 | **13.6GB 潛在駐留 → dense + pool；swap 從結構上消失（唯一手段）** | 改 llama.cpp 載入/緩存路徑（中量）；evict 後重讀成本不變但不再雙重佔用 | **未動手（P0）** |
| **L3** | 自適應閉環：swap 進 autotuner 目標函數；LAYER_CAPS（已在樹上）按路由直方圖做逐層容量分配；監測 (hit, capacity miss, swap 水位)，swap 觸發就收 pool | LAYER_CAPS 樹上、swap 現已可追蹤（dswap 記錄） | 量測可復現性（所有 A/B 的前提） | 需要 harness 支援（已有 dswap 記錄雛形） | 未動手 |
| **L4** | Metal wired 壓力轉移：union 的 GPU 緩衝區是 wired（不受回收）→ 高壓時 anonymous 被換出、wired 暴漲；pool 駐留納入 wired 級別管理 | wired 1.87GB（系統級，不可回收） | 防「為省記憶體反而製造 swap」 | 需理解 Metal heap 配置 | 未量透（footprint wired 列顯示 0，需其他工具） |

---

## 4. 已測數據附表（重開機後窗口）

| 實驗 | decode t/s | prefill t/s | hit | swap0→swap1 | 備註 |
|---|---:|---:|---:|---:|---|
| 完整 env 8G 熱池（r1） | **13.25** | 280.5 | — | — | 昨天，非深冷卻 |
| 完整 env 8G 熱池（dense_fp1） | 12.24 | 293.3 | 95.6% | — | 無深冷卻 |
| 完整 env 8G 熱池（dense_fp2） | 12.54 | — | 95.4% | — | 無深冷卻 |
| pin_abba base（配對中位） | 11.88 | — | 95.5% | 兩趟 −394/+847 | 深冷卻 ABBA |
| pin_abba p0-pin（配對中位） | 12.585 | — | 97.5% | +706/+489 | ⚠ 同 prompt 泄題 |
| p1-pin（真實文本 profile × 合成 cell） | **8.0** | 283.9 | **85.8%** | — | 泛化失敗：−33% |
| a4_1 4G pool | 8.42 | 281.5 | 87.2% | −16 | 4G 可跑 |
| 甜點首輪 8/6/5/4G | 9.98/6.56/7.99/8.34 | — | 96/93.3/91.4/87.6 | 熱污染，僅相對可信 | — |

**pin 判決**：p0-pin 的 +6% 是「profile 與驗證同 prompt」的泄題產物；真實文本 profile 在生產 cell 上 hit 掉 12pp、decode 掉 33% ⇒ 靜態 pin 放棄 SpAc 的動態適應，unseen 內容時釘錯熱集。**不設 prod-new 預設**。可泛化路徑：p0 vs p1 逐層重合率 → 只 pin 跨 prompt 穩定層、其餘留 SpAc（待做）。

---

## 5. 執行優先級

| 順序 | 動作 | 預期收益 |
|---|---|---|
| **P0** | **L2 實作**：讀 llama.cpp expert 載入路徑 → 確認 madvise(DONTNEED) 落點 / expert 段不 mmap | swap 結構消失（本項目第一要務） |
| P1 | L1：補 6G 一臂 → 4/6/8G 乾淨 ABBA 定甜點 | 砍 2–4GB footprint；4GB 平行雙實驗 | **✅ 6G 已補（footprint 6899MB）；hit 一臂待定** |
| P2 | L3：swap 進 autotuner 目標 + LAYER_CAPS 逐層容量 | 量測可復現性閉環 |
| P3 | L4：pool 駐留納入 wired 級別管理 | 防 swap 反向製造 |

---

## 6. 誠實邊界

- **dense 實際觸碰駐留是推導**（prefill 2048 × 8×40 訪問 ⇒ 覆蓋幾乎全部 256 專家 ⇒ 13.6GB 幾乎全量），vmmap 抓取失敗（進程監控匹配問題），未直接實測到「模型 mmap 段 dirty 大小」。
- **6G 一臂未跑**（重開機後只補了 a4_1 4G 與 8G 三趟）。
- footprint 的 wired 列顯示 0（macOS 版本行為），Metal wired 用的是 vm_stat 系統級 1.87GB。
- 三趟 run decode 12.2–12.5（熱池）vs 錨點 12.57、vs 13.25 單趟——13.25 的條件（快取/熱池狀態）尚未完全復現，是後續要量的第一格。

---

## 7. P0（L2）落點確認 —— 代碼證據（2026-09-24，未 commit）

> 用戶指令「先執行再說」→ P0 第一步 = 讀 llama.cpp expert 載入路徑確認 madvise 落點。**結論：落點不是 madvise（load-mode none 不 mmap），是載入迴圈的 read_raw 分支。**

### 7.1 事實鏈（全部有代碼行號）

| # | 事實 | 證據 |
|---|---|---|
| 1 | **pool fill 直接 `pread/preadv` 讀檔案**，不走 mmap、不走 CPU 副本 | llama-expert-cache.cpp:35（`pread(fileno(f), dst, seg->bytes, seg->file_offset)`）、:149（`preadv`） |
| 2 | **`LLAMA_LOAD_MODE_NONE = 0` = 不 mmap**（`use_mmap` 只含 MMAP/MMAP_MLOCK/AUTO） | llama.h:207、llama-model-loader.cpp:547 |
| 3 | **L4 adopt 設計**：load-mode=mmap 時 expert tensor 用 Metal `_Pool` buft（可寫），fill pread 直接寫它，GPU FFN zero-copy 讀 → **expert 只在 pool 一份** | llama-model-loader.cpp:1347-1356、llama-expert-cache.cpp:1881（adopt_pool_region） |
| 4 | **load-mode=none 分支不走 _Pool buft**（`select_pool_buft` 只在 `use_mmap` 時查）→ expert 落入 skip-load（CPU buffer） | llama-model-loader.cpp:1347-1355 |
| 5 | **skip-load 的 expert tensor 載入時仍被 `file->read_raw(cur->data, n_size)` 全量讀進 CPU RAM**（anonymous heap，非 file-backed） | llama-model-loader.cpp:1844-1850（`else` 分支：`ggml_backend_buffer_is_host(cur->buffer)` → read_raw） |
| 6 | **expert 的 CPU 副本是多餘的**：compute 時 GPU FFN 讀 pool（pread 供給），CPU buffer 只是佔位 | 同 #3/#5 機制：pread 直接寫 pool buffer |

### 7.2 修正 SWAP §1.1 的機制鏈

原主張「--load-mode none 的模型 mmap 頁是 file-backed clean 頁」**有誤**：load-mode none 不建立 mmap。13.6GB 潛在駐留的來源是 **skip-load expert 的 read_raw（anonymous heap）＋ dense 的 Metal upload ＋ pool 8GiB**，三者都是 anonymous/wired，不是 file-backed。

**L2 的正確落點 = llama-model-loader.cpp:1844 的 host-buffer read_raw 分支**：skip-load 的 expert tensor **不 read_raw**（資料由 pool fill pread 供給，CPU buffer 保持佔位/零填充），使 expert 權重只在 pool 存在一份。

### 7.3 待驗證前提（動 L2 前必須先量）

1. **expert 實際駐留形態**：跑一次 load + 觀察 RSS / swap 增長，確認 skip-load expert 是否真的被 read_raw（footprint 的 MALLOC_LARGE 只有 161MB 與「全量讀 RAM」矛盾——**需實測**，可能 prod-new 的 ALLOW_NGL=1 使 expert 走 adopt 而非 skip-load）。
2. **MoE kernel 100% 從 pool 讀**：確認 MUL_MAT_ID / GLU_FUSED_DOWN / CGC_MM_BITIDENT 路徑無任何 fallback 從 `tensor->data`（CPU 副本）讀 expert——有 fallback 則不能安全跳過 read_raw。
3. `LLAMA_EXPERT_CACHE_ALLOW_NGL=1` 的實際語意（決定 expert 走 adopt 還是 skip-load）——grep 定義。

### 7.4 驗證後的三選一

| 選項 | 內容 | 條件 |
|---|---|---|
| A（最簡） | skip-load expert 不 read_raw（佔位 buffer） | #1 確認全量讀 RAM、#2 確認零 fallback |
| B（架構） | expert 段不載入（只建 expert_index + file_offset，資料全走 pread） | A 的佔位 buffer 仍觸發 ggml 檢查失敗時 |
| C（無需改） | 若 #1 實測 expert 根本沒進 RAM（adopt 已接管）→ 修改 §1.1 機制鏈即可，L2 降級為「文檔修正 + 驗證」 | #1 否定全量讀 RAM |

### 7.5 ★ 實測結果（2026-09-24 01:42 prod-new server load probe，footprint 直抓）—— **選項 C：L2 無需實作，§7.1 假設被推翻**

| 量測 | 值 | 含義 |
|---|---:|---|
| llama-server footprint | **8460 MB** | 總駐留 |
| untagged (VM_ALLOCATE) | **7970 MB** | pool（8GiB 上限的實際分配）+ dense Metal |
| MALLOC 全部 | **~474 MB** | （LARGE_REUSABLE 387 + MEDIUM 39 + NANO 36 + LARGE 12）——**沒有 ~10GB CPU 副本** |
| mapped file | 880 KB | **幾乎零 file-backed** |
| server log: skip-load 行數 | **0** | `keeping … out of GPU` 一行都沒有 → **skip_load 未生效** |
| server log: L4 pool | `143 slots/layer, regions adopted from expert tensors` | **L4 adopt 生效**（expert tensor 自身 storage = pool 區域，zero-copy） |
| server log: `GPU pool buffer` 行數 | 0 | `_Pool` buft 未選（load-mode=none 分支）→ 走 L4 shrunk + adopt（143/256） |
| swap | 5140 → 5012 MB | **降了 128 MB**（load 本身不製造 swap） |

**推翻 §7.1 的兩條假設**：
- #4「load-mode=none 落入 skip-load」**不成立**：實際走 **L4 shrunk（expert ne[2] 縮到 143/256）+ adopt**，skip_load 條件未觸發（llama.cpp:384 的語意需再確認，但 log 0 行是事實）。
- #5「skip-load expert 被 read_raw 全量讀 CPU RAM」**不成立**：footprint MALLOC 僅 474MB，**expert 無 CPU 副本**；56% expert 進 Metal（pool，zero-copy），44% 不載入（pread 按需供給）。

**修正後機制鏈**（取代 §1.1 的「13.6GB mmap 全量駐留」主張）：
```
模型 13.65GB
  → dense（-ngl 99）→ Metal（untagged，併入 7970MB）
  → expert 只載入 143/256（L4 shrunk，≈7.6GB）→ Metal pool（zero-copy，fill pread 直接寫）
  → 其餘 113/256 expert 不載入（pread 按需，不駐留）
= 實際駐留 ≈ pool 分配（7970MB untagged 主體）+ MALLOC 474MB + 系統
⇒ **「expert 權重單一駐留」已由 L4 實現**（不是要做的 L2，是既有的 L4 adopt + shrunk）
```

**L2 降級結論**：P0 不需要改載入路徑（選項 C）。swap 結構解的真槓桿是 **L1（pool 甜點 4–6GB）**——footprint 最大項就是 pool 的 7970MB untagged。L4 已把「模型檔案駐留」這一項砍到 880KB file-backed；剩餘 swap 壓力 = pool 8GiB 保留 + dense + 系統。（**✅ 已拆（2026-09-24，見 `docs/H_MEASURED_2026-09-24.md` §二）**：4G vs 8G footprint 對比 → untagged 增量 3120MB = pool；dense+固定 ≈ 555MB —— untagged 的 ~93% 是 pool。）

### 7.6 三前提驗證結果（2026-09-25，0 GPU：純讀碼 + §7.5 實測復用）— 選項 C 定案

| 前提 | 驗證方式 | 結果 |
|---|---|---|
| #1 expert 駐留形態 | §7.5 實測（load probe）復用 | **否定全量讀 RAM**：MALLOC 474MB、無 ~10GB CPU 副本、skip-load 0 行、L4 adopt 生效 |
| #2 MoE kernel 無 CPU fallback | 本次讀碼（loader 判定樹 + ggml-metal supports_buft） | **成立**：見下方證據鏈 |
| #3 ALLOW_NGL 語意 | 本次讀碼（compute_l4_pool_capacity gate + llama.cpp:384/402） | **L4 優先**：expert 走 adopt 不走 skip-load |

**#2 證據鏈（純讀碼，llama-model-loader.cpp + ggml-metal.cpp）**：
1. loader 判定樹（llama-model-loader.cpp:1350-1385）：`l4_kind >= 0` 分支**優先於** skip-load 分支。prod-new（ALLOW_NGL=1）下所有 `_exps`/`blk.` tensor 命中 l4_kind → 走 `select_weight_buft`（L4 zero-copy，Metal weight buft）→ storage 在 Metal，**根本到不了 CPU buft 分支**。
2. ggml-metal.cpp:927-937：pool buft 已註冊 `supports_buft` → FFN `mul_mat_id` **不離開 GPU**（註釋明說：不註冊會「silently leave the GPU」）。
3. 唯一 CPU 副本路徑 = skip-load 分支（`ggml_backend_cpu_buffer_type`）——prod-new 不觸發（§7.5 實測 0 行）。
4. `cgc_exact_cache_verify_post_fill`（bit-exact 驗證）是**診斷路徑**：`CGC_EXACT_CACHE_VERIFY` env 未設即 return（llama-expert-cache.cpp:895-897）——prod-new 默認關。

**#3 語意**：
- `compute_l4_pool_capacity` gate（llama-model-loader.cpp:1106-1113）= `NGL>0 && ALLOW_NGL 存在 && 無 NOGATHER` → 容量 143/256@8GiB。
- llama.cpp:384/402：`expert_cache_skip_load` 與 `l4_path` 條件都含 ALLOW_NGL——但 loader 的 `l4_kind>=0` 先命中 → **expert 走 L4 adopt（zero-copy pool），skip-load 語意僅對無 l4_kind 的 tensor 生效**（與 §7.5 一致）。

**結論（選項 C 定案）**：**L2 無需實作**。「expert 權重單一駐留」已由既有 **L4 adopt + shrunk（143/256）** 實現——expert 數據流 = `檔案 → pread → Metal pool（zero-copy），CPU 零副本`。P0 不需要改載入路徑；`llama-model-loader.cpp:1844` 的 read_raw 分支在 prod-new 下**不被 expert 觸及**（dense 權重的正常載入另計）。

**swap 結構解剩餘槓桿**（L1 已被 `docs/POOL_SWEET_SPOT_2026-09-25.md` §7 實測判死：4G/6G 掉速、8G 為穩定態必需）：只剩 **L3（自適應閉環：swap 進 autotuner + LAYER_CAPS 逐層容量）** 與 **L4（pool 駐留納入 wired 級管理）**——見本文 §2-§4。

---

## 8. L0–L4 敘事重組（2026-09-25）

按「根因 → 已閉環 → 實作中 → 已兌現」重組全部工作的敘事層。

### L0 —— 根因與驗證閉環（已閉環；含原 P0 工作）

- **一句話根因（§0）**：13.0 GiB 模型 + 8 GiB pool = 21.2 GiB 潛在駐留塞進 16 GB；macOS swap 只累積、不回收。
- **機制鏈 + footprint 分解（§1）**；**swap 製造機定量（§2）**：SpAc 踢錯 → pread（一趟 19.6 GiB 重讀）→ 高壓下頁變 swap。
- **原 P0（L2 單一駐留實作）以驗證閉環收尾（§7.3–7.6）**：三前提驗證 → expert 數據流 = `檔案 → pread → Metal pool（L4 adopt, zero-copy），CPU 零副本`（MALLOC 全部 474 MB、skip-load 0 行）→ L2 已由 L4 路徑實現，**P0 不需改載入路徑，以「驗證 + 降級」而非實作收尾**。

### L1 —— 縮 pool（已死）

`POOL_SWEET_SPOT_2026-09-25.md` §7 同窗口真曲線：4G −36%（7.67 vs 12.06 t/s）、6G −13%（10.48）；8G 是穩定態必需（cap miss：4G 32.1% / 6G ~15% / 8G 3.8%）。

### L2 —— 單一駐留（已實現，結論歸 L0）

### L3 —— 實作敘事（當前主線；兩條正交路線）

**路線 3a：capacity 口徑修正（判別式 #1 已跑完，2026-09-25 ~01:10）—— 143 可升到 191**

一次 prod-new init（8 GiB budget）+ footprint，三重交叉驗證閉合：

| 量 | 數值 | 口徑 |
|---|---:|---|
| untagged (VM_ALLOCATE) | 7970 MB | footprint pid（= pool + dense Metal） |
| dense Metal（推算） | 1848 MB | 7970 − 6122 |
| **pool 實際分配** | **6122 MiB** | 143 slots × 1.0703 MiB × 40 decoder 層 |
| **budget 未花滿** | **2070 MiB（25.3%）** | 8192 − 6122 |
| 4G/8G untagged 差 | 理論 3083 / 實測 3120 MiB | H_MEASURED 交叉驗證（差 1.2%） |

根因（代碼定位）：capacity 用**雙重保守口徑**——`per_slot = max over decoder layers`（1.397 MiB，來自最大層；平均僅 1.0703）× `denom = max_layer=41`（**含 MTP 層，而 MTP 不佔 pool region**）（llama-model-loader.cpp:1186-1188；llama-expert-cache.cpp:3782）。

**修法（確定可升 191）**：`capacity = budget / Σ_decoder per_slot_layer`（總 stride 口徑）：
- Σ_decoder per_slot_layer = 6122/143 = 42.81 MiB（一個「40 層全開 slot」的實際 stride）
- capacity = 8192/42.81 = **191.3 → 191 slots/layer，同 8 GiB RSS**
- 升後總 Metal 分配 = 191 × 42.81 = 8177 MiB ≈ budget 8192（**不超**：按各層實際 stride 線性放大，footprint 已含層間差異）
- 191 對應 counterfactual K=192 的 **99.8% mass coverage**（MASSCOV 實測）→ **capacity miss 結構性接近消除**

> **誠實邊界**：穩態 cap miss 實測 3.8%（真曲線；commit_bench hit 96.2%）→ 消除後 disk 讀/swap 製造相應減少；**first-touch（compulsory）與 policy miss（SpAc 踢錯）不在此列**；t/s 收益需 ABBA 實測（143 vs 191），不預先宣稱。

**路線 3b：可泛化 pin（SWAP §4 待做）**：只 pin 跨 prompt 穩定層（p0 vs p1 逐層重合率篩選）、其餘留 SpAc——靜態全 pin 已判死（真實文本 profile × 合成 cell 泛化 −33%）。

**判別式 #2（未跑）**：層間路由不均——CGC_MASSCOV 按層輸出一次即可，決定 LAYER_CAPS 逐層重分配的收益上限。

### L4 —— 務實內核（已兌現）

- **P1/P2 的 `cgc_discard_pages` 已落地**：`madvise(MADV_DONTNEED)` 丟棄**將被覆蓋 / 已垃圾**的 pool 頁（頁對齊防護：expert stride 1.0703 MiB = 68.5 个 16 KiB 頁，start round UP / end round DOWN，只丟完全在範圍內的頁；Metal pool_ext 指針禁入）→ **避免 kernel 把將被覆蓋的頁寫盤——這就是 L4 的務實兌現**。
- 原「pool 納入 wired 級管理」方向**不建議**：pool 是 anonymous malloc（§7.5 確認），納入 wired 需改分配路徑（→ Metal buft，大改 + 高回歸），收益模糊。


---

### L3 路線 3a/3b 終判（2026-09-25，用戶拍板 B）

**路線 3a（capacity 口徑修正，CGC_L4_CAP_TOTALSTRIDE）—— 判死：高水位 OOM**
- 代碼生效確認：cap 143 → **189 slots/layer**（理論 191，口徑差 −1%）；loader/allowlist/重建/env 驗證全過。
- 但 server 成功 listening 後被 **SIGKILL（Killed: 9，Jetsam 高水位）**：189 × 實際 stride → pool ~8091 MiB + dense 1848 = untagged ~9939 MiB，在 swap 已佔 3.8 GB 的環境下裝不下。
- **判決修正**：「budget 未花滿 25.3%」不是可免費提容量的冗餘，而是 capacity 保守口徑預留的**高水位安全邊際**。未 commit（改動留在工作樹：llama-model-loader.cpp、run_server.sh allowlist）。

**路線 3b（可泛化 pin：只 pin 跨 prompt 穩定層）—— 判死：泛化覆蓋跑不贏 SpAc**
- 產物：`scripts/check/pin_profiles/pin_3b_stable21.txt`（21 穩定層 pin 三 prompt 全交集共識 core avg 102/層、留 41 槽 SpAc；19 層空行全 SpAc）。
- LOO 三折（2 prompt 共識 → 第 3 個未見 prompt）：穩定層 top-96/top-143 recall = **0.802 / 0.706**，不穩定層 = 0.729 / 0.643（僅差 7/6 pp）。
- **判決**：共識 core 佔 71% 槽、泛化 ~80% < **SpAc 穩態實測 92.8%**；p1-pin 真實 GPU 已驗同類缺陷（佔槽 → decode −33%）。路由本質 prompt-dependent。
- **唯一未閉合**：prompt 切換的冷啟動 miss 曲線（pin 可能只在冷啟動有價值），需逐 token hit 曲線，未量。

**L3 殘留（未被證偽）**：判別式 #2 —— LAYER_CAPS **運行時**逐層容量（按實際路由頻率直方圖動態分配，非靜態 pin）。現有 route dump 為純 top-143 排名、不帶頻次，精確定價需一次按層頻次 route record；先做零 GPU 層間不均預分析決定是否值得。


**判別式 #2（LAYER_CAPS 運行時逐層容量）—— 判死：收益 +0.5%，低於 3% 門檻（2026-09-25 精算）**
- masscov_decode_shape_code 已帶按層頻次與 counterfactual K 曲線（k96/128/143/192），零新 GPU 直接定價。
- 現狀：SpAc 實際 **cur=0.7874**；143 slot 理論 **k143=0.9610**；**policy gap=0.1736**（17.4% mass 因 SpAc 踢錯/動態窗口落到冷 expert）。
- 最優逐層分配（總 slot 5720 不變、貪心邊際、各層 96–192）：覆蓋 0.9662 vs 均勻 0.9610 ⇒ **僅 +0.52 pp（+0.54%）**；對 SpAc 實際 +0.66%。
- 層間熱點確實不同（層間 Jaccard 0.388、40 層並集 256/256），但均勻 143 已接近總預算下的覆蓋上界，逐層重分配買不到量。
- **真正槓桿是 policy gap（17.4%），不是 capacity** ⇒ 正路＝**B 真臂（熱門優先替換）**，代碼已在樹上，待重開機乾淨環境 ABBA 驗證（冷啟動逐 token hit 曲線同一次收集）。

**L0–L4 至此全部判決/兌現，唯一在途實作＝B 真臂驗證。**


---

## §9 收尾：乾淨 SpAc 現狀基線（2026-09-25 02:23–02:25，重開機 swap=0 後權威數字）

命令：`harness.py bench --arm prod-new --reps 3`（p2048/n128/d512、warm-skip64、base_check PASS）

| 指標 | 成績 | 各 rep | 狀態 |
|---|---:|---:|---|
| **prefill pp2048** | **301.47 t/s** | 307.5 / 300.0 / 296.9（std 5.5） | ✅ 超 250 目標 |
| **decode tg** | **11.49 t/s** | 10.81 / 11.73 / 11.93（std 0.60） | 現狀基線 |

- thermal：launch/worst 全程 **NOMINAL**；swap：launch **0** → after **625 MiB**（prefill 在 16GB 上的固有製造、decode 段平穩；attribution 標 swap 但 growth 主要來自 pp）。
- 對歷史：11.49 與 commit_bench 12.17、峰值 12.57 同量級（launch-to-launch ±5–8%）；r3 中位數口徑下乾淨現狀即 11.5 檔。

## §10 終局判決：現狀最優，無更多可落地槓桿

L0–L4 所有分支已用數據/讀碼逐一判決或兌現：
- **capacity / 靜態側無錢**：3a cap189 高水位 OOM；3b 泛化 pin 跑不贏 SpAc；#2 逐層容量僅 +0.5%；NSG sweep 0.6pp；縮 pool L1 −36%。
- **預測/重疊側無路**：prebind h=0.032；ρ 按層批次化 −47.6%；L3 async fill 可遮窗口=0；verify 宿主批次化僅 ~2%。
- **已兌現**：SpAc EMA 熱門優先替換/預取（prod-new 默認）、P0 skip-readraw、P1/P2 cgc_discard_pages、L4 adopt zero-copy 單一駐留、prefix reuse checkpoint、prefill stream。
- 模型/kernel 側的更大 shape（更大 expert、連續佈局、更高量化）需改模型、不在工程範圍。

**⇒ 16GB M4 上此引擎的可交付現狀＝prefill ~300 t/s、decode ~11.5 t/s；L0–L4 收案。**
