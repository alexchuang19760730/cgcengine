# Shape 1 的等待預算：每層都有 miss，而 wait 不是 miss 的函數

**日期** 2026-09-25 22:00 · 載體 `prod-new`（**分段臂**，MTP off、8 GiB 池、`-p 2048 -n 128 -d 512 -r 3 -b 5632 --warm-skip 64 --fixed-fill-seed 1`）· binary `Backup/eseries/E2b/summary.json` 的 `engine` 區塊 · 產物 `Backup/eseries/E2b/`

## 0. 一句話

**形狀 1（S1 ＋ 同步等 fill ＝ 48.2＋20.8 ms ⇒ +28%）不成立**，理由是三條互相獨立的讀數：**① 40/40 層都有 miss**（逐層退回會在「每一層」觸發，不是先前推估的一半）；**② wait 與 miss 幾乎無關且近乎均勻**（1.32–1.92 ms/層、rho = **−0.32**）⇒ 它是 round-trip 結構，減 miss 動不了它；**③ fill 本身在這兩個欄位裡只看到 1.86 ms/step，與 EBTIMER 的 20.8 ms/step 差 ~10×** ⇒ 那筆帳還沒對上，不能拿它當乘法的輸入。

## 1. 為什麼量這個（問題形狀）

形狀 1 的便宜論證是「S1 單段 48.2 ms ＋ fill 20.8 ms = 69 ms ⇒ 14.5 t/s（+28%），而且不需要異步、不需要逐位元補算」。**這個論證只有在「等待的成本 ∝ 它等的那幾個 miss」時才成立。** 若等待其實由「每段 round trip（等整段完成 → 發 hook → 提交下一段）」主導，則減 miss 買不到任何東西，只有重疊（形狀 2）能付。

## 2. 判準（**跑之前**就寫死，見工具 docstring）

`W_miss` = 每層中位 wait 在「平均 miss/步 > 0」的層之和，`W_clean` = 其餘，rho = 每層平均 miss/步 與該層中位 wait 的 Spearman：

| | 條件 | 意義 |
|---|---|---|
| **R1** | `W_miss/(W_miss+W_clean) ≥ 0.60` 且 `rho ≥ 0.50` | wait 是 miss-bound ⇒ 降 miss（形狀 3）直接付錢 |
| **R2** | 否則 | wait 是 round-trip-bound ⇒ 只有重疊（形狀 2）能付 |
| **INVALID** | 可用 miss 步 < 3，或無 DECPROF 區塊，或無共同層 | 沒有 miss 軸就答不了（**不是**「通過」） |

依變數預先定為 **`wait`**；`cb`（hook，也就是 fill 實際花掉的地方）以同一條規則附報，**不替換**已凍結的那一欄。

## 3. 載體與儀器（為何不能用 S1 臂）

逐層儀器 `CGC-DECPROF all:` **位於 41 段迴圈之後**（`ggml-backend.cpp:2703` 區塊），而 `CGC_SEG_BATCH=1` 在 `ggml-backend.cpp:1781` **迴圈之前就 return** ⇒ 在 S1 臂上這支儀器**結構上不存在**（2026-09-25 21:30 量到 DECPROF=0／HOOK=0／miss dump 檔未生成）。所以本量測用**分段臂**當儀器載體，miss 軸用：

- `CGC_DECODE_PROFILE=1 CGC_DECODE_PROFILE_ALL=1 CGC_GPU_TIMING=1`（每步逐層 wait/cb/submit ＋ gpu/union/gap）
- `LLAMA_EXPERT_CACHE_BATCH_DBG=1`（每層每步的 miss 數，**只印有 miss 的層**）
- `LLAMA_EXPERT_CACHE_MISS_DUMP=<path>`（全 run 逐筆，**只當退路**：它含 prefill 的專家讀取）

對齊：兩支生產者不共用步號（DECPROF 只在 `dp_step%8`／step 1／任一 batched graph 印，BATCHDBG 只印有 miss 的層，且 run 會在步中結束）⇒ **不取「最後一塊」也不取「最後一步」**，兩側一律**跨所有完整步聚合**（每層平均 miss/步、每層中位 wait）。

## 4. 讀數

臂：pp 255.50 ± 11.43、tg **11.05 ± 0.18**（hit 95.7%、misses 5567／128920 requests = 4.3%）；53 個 DECPROF 區塊、2120 列逐層、209 個完整 miss 步（BATCHDBG 共 3001 筆 miss，**佔全 run 的 54%** —— 另外 46% 走的是不印 BATCHDBG 的 `ensure_slot` 路徑）。

| 量 | 值 |
|---|---|
| **有 miss 的層數** | **40 / 40**（`n_clean_layers = 0`） |
| 每步 miss 數（BATCHDBG 軸之和） | 11.66 |
| 每層中位 `wait` | 最小 1.32、中位 1.90、最大 1.92 ms（**幾乎均勻**） |
| Σ 每層中位 `wait` | **70.27 ms/步** |
| rho（wait vs miss） | **−0.32** |
| Σ 每層中位 `cb` | **1.86 ms/步**（最大層 0.86 ms） |
| rho（cb vs miss） | +0.52 |

**判定（照凍結規則）**

```
column = wait （預先凍結）  ->  R2_FAILS_roundtrip_bound
   wait 在 miss 層的佔比 = 1.0（閾值 0.6）… 但這格是退化的：沒有任何「乾淨層」
   rho = -0.32（閾值 0.50）⇒ 不過
column = cb  （附報）        ->  R1_HOLDS_miss_bound（rho = +0.52，剛好壓線）
```

## 5. 三條結論與一條被推翻的推估

1. **每層都 miss ⇒ 逐層 fail-closed 會在每層觸發。** 我在上一輪寫的「約一半的層會踩到 miss」是**我自己的 Poisson 近似**（取 4% 的 demand 缺失率、每層約 19 個 demand ⇒ λ≈0.75 miss/層/步 ⇒ `1−P(0)`≈0.53），**不是量測**。本輪量到的是 **40/40** ⇒ 那個推估**作廢**，而它原本是形狀 1 可成立的主要理由之一。
2. **wait 不是 miss 的函數。** 逐層中位 wait 落在 1.32–1.92 ms 這個 0.6 ms 寬的帶裡（40 層幾乎同值），而各自的 miss 數從 0.1 到 0.6/步不等；rho = −0.32（略負）。**這是 round-trip 的形狀，不是等待缺頁的形狀** ⇒ 形狀 1 的「等到 fill 好」不會隨 miss 數縮放。
3. **fill 的成本沒有出現在這兩個欄位裡。** cb（hook，fill 真正花掉的地方）全 40 層只有 **1.86 ms/步**，而 E1 裁決的 `CGC-EBTIMER` 值是 **20.8 ms/步** ⇒ 差 ~10×。兩者不可能同時描述同一個 decode 步；可能的原因有三（本輪**不足以**裁決）：BATCHDBG 只涵蓋 54% 的 miss、cb 只圍住 hook 而不含 pool worker 的等待、或 EBTIMER 量的是 server 載體而非 bench 載體。**這是要修的下一格。**

## 6. 限制（不可外推的部分）

- 這一臂的盒況不合格：起跑 swap **5586 MiB**、收尾 8206（**growth +2619**，門檻 500）、min_free **14 MiB**、worst thermal MODERATE ⇒ 依 `MEASUREMENT_CONTRACT §5.1` 這是**診斷臂**，**不對外宣稱任何 t/s**（tg 11.05 只當「與 11.2–11.6 同族」的旁證）。
- 絕對值（70.27 ms/步的 wait）被 swap 灌大；**可比的是形狀**：均勻性、rho、40/40、「cb ≪ wait」。
- miss 軸只覆蓋 BATCHDBG 那條路徑（54%）；`n_clean_layers = 0` 可能部分是這個原因，但**即使只算這 54%，也沒有任何一層是乾淨的**。
- 工具自測 11 項（含「無 miss 軸必須 INVALID」與「尾端碎片不得算成一步」）。

## 7. 下一步（由本輪的三條結論直接指定）

不是再算一次 `48.2 + 20.8`，而是**把 fill 的時間定位到哪一欄**：在同一 arm 上把 `wait / cb / submit` 三欄與 `CGC-EBTIMER` 的 `step_usec`、`fill_wait_us` **同框對帳**，回答「20.8 ms 在哪一欄」。這一格若落在 `wait`（即當前 arm 的 round-trip 裡），形狀 1 就真的死了、只剩形狀 2；若落在 `cb`（只是被 BATCHDBG 的 54% 遮住），形狀 1 的成本才有機會隨 miss 縮放。

## 8. 產物

| 檔案 | 內容 |
|---|---|
| `Backup/eseries/E2b/summary.json`＋`llama_bench_*.stderr.log` | 臂的讀數、逐層 DECPROF、逐層 BATCHDBG、engine/cell/box_gate |
| `Backup/eseries/E2b/wait_budget_wait.json`／`wait_budget_cb.json` | 兩欄的逐層表與判定（含 `miss_source` 與退化警語） |
| `Backup/eseries/E2b_misses.txt` | 3001 筆逐筆 miss（layer expert） |
| `scripts/check/s1_wait_budget.py` | 分析器＋凍結規則＋11 項自測 |
