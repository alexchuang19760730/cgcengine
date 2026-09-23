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
