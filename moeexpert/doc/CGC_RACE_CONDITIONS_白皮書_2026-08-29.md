# CGC Race Condition 分析與修復 白皮書

**日期**: 2026-08-29
**範圍**: llama.cpp fork（bounded-residency expert cache / MTP / Metal CGC-SEG pipeline）中所有已定案與調查中的 race condition
**性質**: 技術存檔——每個 race 的症狀、機制、修復、驗證方法、代碼位置，供後續開發避免重蹈與快速辨識同族問題

---

## 0. 總綱：本 fork 的三個 race 家族

所有已觀察的 race 都落在三個相交的平面：

| 家族 | 平面 | 根因句 |
|---|---|---|
| **A. CPU 寫 × in-flight GPU 讀** | Metal 共享記憶體 | 對已被 in-flight command buffer 引用的資源做 CPU 寫入是 UB |
| **B. 生命週期 × 排序** | pipeline 排程 | hook/消費者在 producer 未完成時觸發（過早讀）或 slot 資源被重複發放（重複寫） |
| **C. kernel 級 driver race** | IOGPU/Metal driver | CPU 對共享 Metal 頁的 fault/寫入 × driver 排程路徑的 kernel 死鎖（**未結案**）|

家族 A/B 已全部修復並驗證 bit-identical；家族 C 是現行調查（§R8）。

---

## R1. submit-ahead remap race（CPU 寫 remap leaf × GPU 讀）

- **症狀**: 短 prompt / 冷 cache 時輸出 garbage（`0000`、`!`、echo prompt）；暖 prompt 偶爾「僥倖」bit-identical。
- **機制**: CGC_OA_ASYNC 分段流水線原先把 segment[i+1] **先提交**，等 segment[i] 完成後才跑 top-k hook 寫 remap leaf。segment[i+1] 的 command buffer 已引用該 remap buffer —— GPU 可能趕在 CPU 寫入落地前讀到 stale remap。**修改 in-flight command buffer 引用的 buffer 內容 = Metal UB**。
- **修復**: 嚴格順序 `wait(seg i 完全完成) → hook(寫 remap) → submit(seg i+1)`（ggml-backend.cpp:1830-1904）。`CGC_SUBMIT_AHEAD=1` 保留舊 racy 順序供 A/B 對照。
- **驗證**: 短 prompt 全輸出 diff bit-identical（原本短 prompt 必 divergence）。
- **教訓**: 「submit-ahead」的效能收益建立在 race 之上；管線化必須以「寫入發生在被引用 buffer commit 之前」為邊界。

## R2. topk-VIEW segment 邊界 race（hook 過早觸發）

- **症狀**: garbage routing（remap 用到未算完的 expert ids）。
- **機制**: 以 `ffn_moe_topk-` VIEW 作為 segment 邊界時，VIEW 是 dependency-free alias，ggml 可能把它排在 producer（argsort）之前 → hook 在 ids 尚未算出時觸發。
- **修復**: 邊界改為 `ffn_moe_argsort-`（真正產生 expert ids 的 op）（ggml-backend.cpp:1735-1760）。`CGC_TOPK_BOUNDARY=1` 保留舊邊界供 A/B。
- **教訓**: hook 邊界必須錨定在「計算 op」而非「view/alias」。

## R3. 只等 main buffer 的部分完成 race

- **症狀**: stale ids → garbage remap → whole-graph corruption。
- **機制**: 舊版 hook 只等 `cmd_buf[n_cb]`（main buffer）完成即觸發；但 argsort 可能落在某個 secondary buffer（cb[0..n_cb-1]）仍在執行。
- **修復**: poll target = `done0 + (i+1) * (n_cb+1)`，等**全部** command buffer（ggml-backend.cpp:1856-1864；計數器語義見 ggml-metal-context.m 的 `addCompletedHandler` + `cgc_done`）。
- **教訓**: 多 command buffer 的「圖完成」= 所有 buffer 完成，缺一不可。

## R4. prefetch 背景 thread race（fill/evict × GPU verify 執行中）

- **症狀**: MTP 多 token verify 輸出 garbage、accept rate 崩塌。
- **機制**: prefetch 背景 thread 在 **GPU verify 仍在執行** 時 fill/evict slot，覆寫 GPU 還在讀的 pool slot。
- **修復**: MTP 路徑停用 async prefetch（`CGC_PREFETCH_OFF`）；恢復的 `CGC_PREFETCH_SRC=hist` 僅限非 MTP decode（rolling window union，確保下個 decode 的 ensure_batch 命中；生產 profile 由 run_n30cache.sh 控制）。
- **教訓**: 任何「背景寫 pool」的設計都必須先證明與 in-flight GPU 讀無重疊窗口。

## R5. ZERO-slot OOB 讀（whole-graph corruption）

- **症狀**: 全輸出 `!`。
- **機制**: 冷 expert 在 slot table 為 `-1`；`ggml_get_rows` 對 pool 做 OOB row 讀取 → NaN 級聯。
- **修復**: **ZERO_SLOT fallback** —— 每層保留最後一格（`slots_l-1`）為 reserved zero slot，`-1` 一律映射到 ZERO_SLOT（`llama_expert_cache_slot_table_safe`，llama-expert-cache.cpp）。
- **教訓**: 稀疏表（-1 = miss）餵給稠密 kernel（get_rows）前必須做 safe 映射。

## R6. batch fill 的 slot 雙重分配 race

- **症狀**: nondeterministic FFN garbage（同 slot 兩 expert 併發 fill 互相覆寫）。
- **機制**: batch 在一把鎖下分配 slot，若 `last_use` 在 fill 完成後才 bump，下一次 `pick_slot` 的 LRU 會把「剛分配、尚未填入」的 slot 視為最舊而**再度發放**。
- **修復**: (a) 分配當下即 bump `last_use`；(b) `batch_tick` 在 per-expert loop 之前取一次，整個 batch 的 bump 同屬一個 tick（WIN_PIN 修正的副產品，llama-expert-cache.cpp:450-511）。
- **教訓**: LRU 發放與 bump 必須在同一臨界區內完成，否則「分配即被逐出」。

## R7. bulk pool write × async pipeline 的 GPU 死鎖（prewarm 家族）

- **症狀**: GPU 死鎖 —— `sched_graph_compute` 永不返回、CPU 97% 自旋。
- **機制**: prewarm_hot 一次把 2911 個 expert 連續 fill 進 Metal pool（大量 CPU 寫入共享 Metal 頁）→ GPU wedge（與 R8 同族）。
- **修復**: REVERT prewarm（修復其 dead-code bug 後重試仍死鎖）；**per-step `ensure_batch`（約 24 fills/step）安全**。bulk pool writes 需要安靜窗口（quiescent window）設計才可能安全。
- **教訓**: 「每步小量」安全、「一次大量」死鎖 —— fill 速率與 in-flight GPU 工作密度是調變變數。

## R8. 間歇 prefill GPU wedge（**phase 2 probe 定案：driver 排程 lost-wakeup；probe kick = 生產自救**）

- **症狀**: 約 1-in-4 ~ 1-in-26 的 steady run，prefill 期 CPU ~93% 自旋於 `sched_graph_compute`（CGC-SEG hook 等待迴圈），停滯不前。
- **證據鏈**（watchdog dump + ioreg + sample）：
  - `expected − done = 卡死的 Committed buffer 數`（P1: 5、P2: 3），`computes × 9 = expected`（每 graph_compute = n_cb+1 = 9 個 buffer，計數自洽）
  - 卡死形態：**連續帶狀 worker buffer 永卡 `Committed`（P1: cb[3..7]、P2: cb[5..7]，從未被 driver 排程）**，其餘停 `Executing` → 「隊列位置型」wedge：某位置之後的命令全部不再排程
  - wedge_at = **submit 後 2-5ms** 內成形（高頻段提交期：~110-180us/submit）
  - kernel 側：GPU scheduler busy=0、BusyWorkQueues 空、recoveryCount=0 → **GPU 實際上 idle，driver 的 completion/排程管線遺失了這批 buffer**
  - encode workers 全部完成（enc_done 齊）、pool fill workers 閒置 → 非 CPU 側鎖死；main thread 停在 `cthread_yield/swtch_pri` 健康輪詢（sample 取證）
- **已排除**:
  - residency heartbeat（A/B：B 0/7 vs T(NO_RESIDENCY) 2/9，2 次全落 T 不顯著 p≈0.3 → H1 推翻）
  - merge-read preadv（ON/OFF 皆死鎖，同簽名）
  - 記憶體壓力「相關性」實為 abort 後 Metal 記憶體回收中的暫態量測假象（T2 abort 後 38%、下一 run 回 65%）
- **phase 2 P arm 判定（2026-08-29，16 runs）**: stall 2/16（均落冷 page-cache 前兩跑 = prefill preadv I/O 最重時），probe 結果 **兩次均 ALIVE/ALIVE** → 判定矩陣第三行成立：**queue 未堵、device 健康，僅本批 buffer 的排程狀態被 driver 遺失（lost-wakeup）**。
  - P2 same-queue probe 耗 **146ms** = 16B fill 排在 5 個 wedge buffer 之後，隨 driver 被「踢」醒後一併排空 → **新提交即恢復觸發器**，無須重發遺失 buffer
  - fresh-queue probe 11-12ms = device/kernel 排程器本身健康
  - 兩 run 均 probe 後自動復原（rc=0、全輸出、accept 98.919%）——**watchdog probe = 偵測 + 診斷 + 自救三合一**，pkill workaround 退役
- **Apple 文檔佐證**: `MTLCommandBufferStatusCommitted` = "command queue is preparing to schedule the command buffer by **resolving its dependencies**"——卡死階段正是 driver 的 kernel 側依賴解析步驟。
- **root-cause 模型（H3 精煉）**: IOGPU 依賴解析（含共享 pool 頁 residency）與 CPU 端對同一批頁的寫入（pool fill 的 preadv 頁 fault / compressor 頁擠壓）競爭 → 依賴解析完成事件遺失 → 該位置後的 buffer 永不轉 `Scheduled`；新 commit 重新觸發排程即恢復（同 R7 家族，但觸發點在 driver 依賴解析而非 GPU 執行）。
- **觸發模型（D arm 反證強化）**: wedge 成形於「**quiet window 後的提交爆發**」——證據：(a) P arm stall 集中於冷 page-cache 前兩跑（prefill preadv I/O 暫停 CPU 提交 = quiet window）；(b) DRAIN 每 64 computes 全排空 queue 再爆發重啟，wedge 率 3×；(c) 卡死帶 = 提交批次（9 buffer/段）的**尾部** buffer（quiet 後最先進入依賴解析的批次尾巴遺失 wakeup）。
- **緩解**:
  - **已驗證**: watchdog probe kick（`CGC_WATCHDOG=1`，run script 內建）——stall 時自動踢醒 driver
  - **已否決（D arm 16 runs）**: `CGC_DRAIN_EVERY=64` 週期全 queue 排水——**非但不能預防，反而使 wedge 率 3×化（6/16 vs P 2/16）並引入致命雙事件（2/16 abort）**：drain 每 64 computes 製造「queue 全排空 → 爆發重啟」的窗口，正是 wedge 的成形條件（見觸發模型）。速度零收益（clean run 同速）。**結論：DRAIN 維持預設 OFF**
  - **repeat-kick 修復（2026-08-29，已驗證）**: probe pair 由一次性改為**每次 dump cadence（15s）重複觸發 + recovery 後 re-arm**——D3/D13 的第二事件因一次性 probe 從未被踢而掛到 60s abort；修復後每 15s 一踢直到復原。修復後穩態驗證：rc=0、26.28 t/s、accept 98.9%（無回歸；本跑未觸發 stall，屬機率性事件）
- **調查工具**（見工具白皮書）: watchdog dump（stale/submit_age/wedge_at 取證）、liveness probe 雙探針、kernel 狀態捕捉（ioreg/memory_pressure/sample）、`dlk_phase2.sh` P/D 臂 harness。
- **判定矩陣**（probe 結果 → 結論；2026-08-29 實測 = 第三行）:
  | probe[same] | probe[fresh] | 結論 | 恢復策略 |
  |---|---|---|---|
  | DEAD | ALIVE | queue 級 wedge | 換新 command queue 可恢復 |
  | DEAD | DEAD | device/kernel 級 wedge | 只能 avoid-or-restart |
  | **ALIVE** | **ALIVE** | **queue 未堵，driver 遺失本批 buffer 的排程（lost-wakeup）** | **同一 queue 提交任一新 buffer 即踢醒排程（probe kick 實證）** |

## R9. MTP 選項隔離約束（防 race 的工程紀律）

- 所有 MTP 專用改動**必須**用 MTP 編譯選項分離，不得影響 base（非 MTP）解碼行為（bit-identical 硬性要求）。
- draft sampler 允許 repeat penalty（1.1-1.2，提升輸出品質且 accept 67-79%）；**target sampler 禁用** repeat penalty（accept 崩至 25.6%，target 拒絕 draft tokens）。
- bit-identity 驗證一律固定 seed（`N30CACHE_SEED=1` 或 `--seed 42`）。

---

## 附錄：共通設計原則

1. **寫前等完成**：任何 CPU 對 GPU 將讀/正讀的 buffer 的寫入，必須在該 buffer 的 command buffer 完成之後（R1/R4）。
2. **邊界錨定計算 op**：hook/segment 邊界錨定在計算節點，而非 view/alias（R2）。
3. **完成 = 全部 buffer**：圖完成的判定是 n_cb+1 個 buffer 全完成（R3）。
4. **稀疏表安全映射**：-1/miss 進稠密 kernel 前映射到 reserved zero slot（R5）。
5. **發放與 bump 同臨界區**：LRU slot 發放即 bump last_use（R6）。
6. **pool 寫入速率有上限**：每步小量安全、bulk 死鎖（R7/R8）——「async pipeline × pool fill」是本 fork 的核心風險平面。

## 修復驗證方法總表

| Race | 驗證 |
|---|---|
| R1/R2/R3 | 短 prompt 全輸出 diff bit-identical + steady stats bit-identical |
| R4 | MTP accept rate 恢復（67-79%）+ 輸出可讀 |
| R5 | 全 `!` 消失、輸出可讀 |
| R6 | FFN 輸出跨 run 確定性（同 seed 同輸出） |
| R7 | prewarm on → 死鎖復現；revert → 消失；per-step ensure_batch 24/step 長跑無死鎖 |
| R8 | 已定案（driver lost-wakeup）：P arm 2/2 stall run probe kick 自動復原（rc=0、全輸出、accept 98.9%）；repeat-kick 修復（dev `ff87ffe97`）後每事件每 15s 重複踢直到復原 |
