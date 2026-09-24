# fill 空轉：已定位（含 ρ 的 A/B）

**Date:** 2026-09-23 · **Status:** 定位完成，修復尚未做完 · **ABBA 2 輪 × reps 3，生產 cell `prod25-stream`**

## 0. 先回答「不是之前已經定位了嗎」

是，而且結案結論是「**別再打**」：

| 來源 | 結論 |
|---|---|
| `docs/CB_IS_THE_DEVICE_RESULT_2026-09-20.md`（F1/F3/F5） | `cb` 的兩個項**都是裝置**：per-miss 0.695 ms = 1.11 MiB ÷ 裝置 1.6 GiB/s；「每層 barrier 0.46 ms」是**二區間擬合的截距偽影**（直接殘差 m≤4 只有 +0.16~0.28 ms、m≥6 為負）。引擎側只剩 **6~13 ms/step（3~6%）** |
| `docs/STEP_SERIALIZATION_2026-09-23.md` §4 方案 E | 「繼續優化 fill 的 IO」**是浪費** |

⇒ 所以「fill 空轉」這個詞的真身不是 fill IO，是 **GPU 空轉（gap 44.4 ms/step）**：
ids 太晚給 CPU ⇒ CPU 站關鍵路徑（hook+submit 34.6 ms）⇒ 41 段 ping-pong。
處方是方案 A「把 ids 提前給 CPU」＝手上的 **ρ**。

**本文要報的是 ρ 上線後才出現的一個新的空轉**，它不是上面任何一份文件裡的東西。

## 1. A/B（tag=192500，ABBA，`SEG=0`）

| arm | r1 | r2 | mean | hit% | 讀的位元組 | jobs | MiB/job | `fill_wait_us` |
|---|---|---|---|---|---|---|---|---|
| base | 11.09 | 9.45 | **10.27** | 61.4 | 8.80 GiB | 39 219 | 0.23 | **56~75 ms** |
| rho | 7.69 | 8.55 | **8.12** | 87.4 | 3.18 GiB | 79 837 | 0.04 | **18.2~20.0 s** |
| rholeg | 8.23 | 7.91 | **8.07** | 87.3 | 3.12 GiB | 80 204 | 0.04 | **14.6~15.3 s** |

`rholeg` = ρ + `CGC_PREFETCH_LEGACY_FILL=1`（舊的 thread-per-segment 填法）。
兩臂的 ids **完全相同**，只有「背景 prefetch 怎麼讀」不同。

## 2. 三個結論

1. **機制是對的**：hit% 61.4 → **87.4**（+26 pp），讀的位元組 8.80 → **3.18 GiB（−64%）**。
   ρ 真的把池填對了。
2. **但 t/s −21%**：贏了指標、沒贏瓶頸。`fill_wait_us` **56~75 ms → 15~20 s（約 200×）**
   ⇒ 關鍵路徑在乾等。**這就是「fill 空轉」的實體。**
3. **讀的形狀不是成因**（我賭錯一次，有 A/B 為證）：把 bg prefetch 從
   `fill_segments_concurrent`（每段一個 `std::thread` + 一次未合併 pread）換成
   `fill_segments_merged_serial`（專家內合併成 preadv + bg 執行緒內聯）後，
   **rho 8.12 vs rholeg 8.07 ⇒ 中性**，且 `fill_wait_us` 反而**變差**（20.0 vs 15.3 s：
   bg 串行把 `slot_loading` 拉長，關鍵路徑更容易撞上「正在填」的 slot）。

   ⚠ **專家內部合併不了**：一個 expert 的 gate/up/down 是三個**不相鄰**的 GGUF tensor
   ⇒ 排序後沒有任何相鄰對 ⇒ 3 段永遠是 3 次讀。0.04 MiB/job 就是這麼來的。

## 3. 唯一對症的下一步

要讓 80k 個 41 KB 小讀降下來，**只能跨 expert 合併**——同一 kind 內相鄰 expert id 在檔案裡
相鄰，這正是同步批次路徑 `fill_segments_pool` 已在做的事（「72 個 90KB pread 塌成少數幾個
大讀」）。所以 bg 側要**按層批次化**：

- `pool_queue` 一次抓同層的 K 個 `(slot, expert)` → 收集全部 segments → 同一套
  merge + preadv → 再一次發布。
- 代價：動 `bg_loop` 的發布語意（`slot_last_use` / `slot_table_scratch` 逐 expert 發布、notify）。
- **便宜的保險絲**：bg 佇列深度或 `pool_outstanding` 超過上限就跳過 prefetch
  （`CGC_RHO_PREFETCH_MAXQ`）——先用它證明「干涉」是成因，再決定值不值得做批次化。

## 4. 本節的儀器改動

- `scripts/check/rho_fill_ab.sh`
  - swap 閘門 `> 0` → `> MAX_SWAP_MIB`（預設 2048，可覆寫 8192 成「只記錄」）。
    理由：重開機 11 分鐘 swap 就 1962 MiB（macOS 常態換頁）⇒「>0」讓 12 格全 REFUSE；
    真正的崩壞點是 5301 MiB。**swap 是累積水位不是當前壓力**（實測 used 3427 MiB 而 free 83%）
    ⇒ 它該被記錄，不該當硬閘門。
  - 修 hit% 的 grep（`^cache: hit` 抓不到縮排行 ⇒ 全印 NA）。
  - 新增 `rholeg` / `probe` 兩臂。
- `src/llama.cpp/src/llama-expert-cache.cpp`
  - 新增 `fill_segments_merged_serial`，`fill_pool_direct`（bg 側）預設改走它；
    `CGC_PREFETCH_LEGACY_FILL=1` 退回舊路徑。

## 5. 量測衛生（本節買來的）

- 重開機後 base 一路漂：**11.97 → 9.92 → 10.41 → 8.87 → 9.26 → 11.09 → 9.45**。
  單臂不能直接比，**一定要 ABBA**。
- `probe` 臂 8.94 vs base 9.26 ⇒ **影子 matmul + capture 只值 −3.5%**
  ⇒ 先前「rho 單跑 5.39」不是機制代價，是那一輪帶了 `SEG=1`（細分段）。
