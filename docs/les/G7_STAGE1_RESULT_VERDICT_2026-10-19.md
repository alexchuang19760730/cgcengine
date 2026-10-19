# G7（C 線階段 1）只讀探針結果判詞：fence 安全窗口 PASS + 生產路徑真實幾何

> **日期**：2026-10-19　**機器**：M4 Air 16GB（統一記憶體）
> **模型**：`Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`（3.06 bpw、L=41、路由層 40、E=256 top-8）
> **分支**：`demo/sweet-spot-windows-fix`
> **配置**：`prod-new`、`-expert-cache 8589934592`（8 GiB）、`--ngl 99 --load-mode none`、`CGC_OVERLAP_FENCE=41`、MTP off
> **探針 env（只讀、預設 OFF）**：`CGC_L4_EVICT_TRACE=1`、`CGC_POOL_SPLIT_DBG=1`
> **探針源碼歸檔**：探針是診斷工具、不併入主幹；完整可 apply 源碼以 **`docs/les/G7_STAGE1_READONLY_PROBE.patch`**（5 hunk）歸檔，乾淨 checkout `git apply` 即可重現（本提交的二進制主幹保持原樣、無探針）。
> **結論等級**：Q1 運行時實證（真實 llama-bench stderr）；Q2 運行時實證。**不許諾階段 2 收益**。

---

## 0. 一句話

**階段 1 兩個安全前提裡，①「evict 是否需要額外 GPU fence」已實證 PASS：8192 次 pool_ext evict 全部發生在單一 build 執行緒（distinct tid=1），冷頁歸還不需要跨執行緒 fence；②「生產路徑幾何」已從真實日誌取得：120 個專家 tensor 在一個連續 Metal 分配裡，絕對 16KiB 頁中 84.7% 只覆蓋單一 slot、3.8% 覆蓋相鄰 2 slot、11.5% 是 gap/dense，stride 是半頁整數倍（每 2 個連續 slot 湊整頁）。** 剩下唯一的階段 2 硬問題是 **`madvise(MADV_FREE_REUSABLE)` 對 Metal shared 連續匿名頁是否真的釋放、GPU 是否安全、重填是否 bit-exact**（L4 判詞 §6.3），必須由階段 2 探針回答，本判詞不預判。

---

## 1. Q1：evict fence 安全窗口 —— PASS（實證）

**證據**：真實 llama-bench stderr 含 7 行 `CGC-L4-EVICT-TRACE`（限頻抽樣：前 3 次 + evict_no=2048/4096/6144/8192），**全部 tid=8287076928，只有 1 個 `[NEW THREAD]`**：

```
CGC-L4-EVICT-TRACE: tid=8287076928 layer=0 slot=14 evict_no=1 [NEW THREAD]
CGC-L4-EVICT-TRACE: tid=8287076928 layer=0 slot=23 evict_no=2
CGC-L4-EVICT-TRACE: tid=8287076928 layer=0 slot=24 evict_no=3
CGC-L4-EVICT-TRACE: tid=8287076928 layer=31 slot=33 evict_no=2048
CGC-L4-EVICT-TRACE: tid=8287076928 layer=30 slot=135 evict_no=4096
CGC-L4-EVICT-TRACE: tid=8287076928 layer=2 slot=73 evict_no=6144
CGC-L4-EVICT-TRACE: tid=8287076928 layer=33 slot=57 evict_no=8192
```

**含義**：L4 pool_ext 的 evict（`cgc_discard_pool_slot`）只在單一 build 執行緒發生；階段 2 對冷頁下 `FREEREUS` 時，不會與另一個同時 evict 的執行緒競爭，**不需額外 GPU fence**。靜態也支持：`cgc_discard_pool_slot` 唯一調用點 `pick_slot`（只被 ensure_slot/ensure_batch 調用），背景 evict_lru 只碰舊 malloc map、不碰 pool_ext。

---

## 2. Q2：生產路徑真實幾何（實證）

### 2.1 為何第一次沒拿到、本次如何修正

G7 原始幾何探針插在 `pool_split_adopt`（僅 `CGC_POOL_SPLIT=1` 路徑）。但 **`CGC_POOL_SPLIT=1` 已被 M1 判定「做了、量了、結論是不要做」（預設 OFF，Metal 上 wide 路徑段錯 + slot 歸屬錯）**，不是生產模式。生產模式 `CGC_POOL_SPLIT=0` 走 `adopt_pool_region`，故第一次 GEOM 行零命中。

本次把幾何 dump（ADOPT+GEOM）補在 `adopt_pool_region`（兩模式都會呼叫），重編後重跑，從生產路徑取得真實數據。

### 2.2 連續佈局

- **120 個 tensor（40 layer × 3 kind）在一個連續 Metal 分配**（vm_allocate 匿名，非檔案 mmap）。
- pool 連續區域 = **7001 MiB**，base0 = `0x332f5c300`（base0 mod page = 768，**pool 起始不頁對齊**）。
- 17160 個 (slot×kind)（120 tensor × 143）；邏輯槽 = 40 × 143 = 5720（gate/up/down 共享 slot index）。
- 同層 3 kind 緊密相鄰（down→gate→up），layer 間有 17–21 MiB gap（39 個、共 ~843 MiB，層間對齊/其他內容）。

### 2.3 絕對頁覆蓋（全區域，正確口徑）

| 一頁覆蓋 slot 數 | 頁數 | 占比 |
|---|---:|---:|
| 0（gap/dense，不屬 pool） | 51,428 | 11.5% |
| **1（單 slot）** | **379,744** | **84.7%** |
| 2（相鄰 slot） | 16,906 | 3.8% |

- 跨 tensor(layer/kind) 共享頁僅 79（層/kind 邊界）。
- **stride 為半頁整數倍**：gate/up `335872` = 20.5 頁、down `450560` = 27.5 頁（均 mod 8192）；3 個 IQ4_XS 層（layer 34/38/39）down 為 `557056` = 34 頁整。
- **每 2 個連續 slot 的位元組寬恰好頁對齊**。

### 2.4 對階段 2 的含義

- **84.7% 的頁是單 slot 頁**：slot 冷時理論上可按絕對頁獨立回收，不傷鄰居。
- 3.8% 跨 2 相鄰 slot 的頁：需兩個 slot 同時無 owner 才能回收。
- 11.5% 覆蓋 0 slot 的頁：是 gap/dense，**不可動**。
- pool 首/尾頁因 base0 不對齊（768）與前後內容共享，回收時以絕對頁、owner 集合為準。

---

## 3. 仍未解（階段 2 探針，待測，不預判）

1. **`madvise(MADV_FREE_REUSABLE)` 對 Metal shared 連續匿名頁是否真的釋放實體頁**（vmmap resident 實際下降），還是如 L4 判詞所述在 Metal 頁上近乎 no-op。
2. **GPU 安全性**：`newBufferWithBytesNoCopy` 是否仍持有該物理頁引用，歸還後 Metal 存取是否正常。
3. **重填 bit-exact**：歸還後下一次 pread 全覆蓋，與不歸還路徑的 digest 是否一致（`CGC_MM_BITIDENT=1`）。
4. 上述需在乾淨 GPU 窗口（重開機 swap=0、thermal nominal）對單一 slot 的內部頁做最小實驗，先只讀、單頁、可回退。

---

## 4. 證據索引

- 真實 llama-bench stderr：`Backup/les_g7_probe_20261019_114545/g7/llama_bench_prod-new_..._r5.stderr.log`
- 幾何結果（全區域 + 分 kind）：`Backup/les_g7_probe_20261019_114545/g7_geometry.txt`
- Q1：`evict_trace.txt` / `evict_tid_counts.txt` / `evict_verdict.txt`（同目錄）
- 離線工具（已增全區域分析）：`docs/les/tools/l4_slab_geometry.py`
- 關聯：`docs/L4_POOL_ANON_NONRECLAIMABLE_VERDICT_2026-10-18.md`、`docs/les/G7_STAGE1_READONLY_PROBE.md`、`docs/M1_POOL_SPLIT_COST_2026-09-14.md`
