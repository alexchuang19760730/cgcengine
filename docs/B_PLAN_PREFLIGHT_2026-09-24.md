# B 方案前置驗證（2026-09-24）— miss 分佈 + LAYER_CAPS 可行性

> 承接白皮書 §10（B 方案復活判定）與 CGC_SEG_BATCH 診斷（+44% 方向確認）。
> 本檔完成 B 方案落地前的兩格前置驗證（0 重建，現有 binary + 儀器 env）。

---

## 1. miss 分佈（實測，480 tokens）

載體：prod-new（MTP off）、`LLAMA_EXPERT_CACHE_MISS_DUMP=/tmp/miss_dump.txt`、
decode_bench 2 rounds + 1 warmup（n=160/token each）。**0 重建**。

| 指標 | 值 |
|---|---:|
| 總 miss（expert 級） | 4010 |
| 平均 miss/step | **~8.4** |
| miss 層數 | 40/40（全層） |
| miss 專家數 | 256/256（session 累計） |

**層級分佈（極度重尾）**：

| 層 | miss | 備註 |
|---|---:|---|
| 0–3 | 527/494/540/322（合計 **1883，47%**） | 前層熱路由 + 首批 compulsory |
| 4–39 | 45–93/層（合計 ~2130，53%） | 平均 ~54/層 → ~0.11 miss/層/step |

**解讀**：
1. **miss 是替換錯配，不是容量不足**：143 slots/層 對 480 tokens 的觸碰集合綽綽有餘
   （每個 token 只觸碰 8 專家，480 steps × 8 = 3840 觸碰；143 槽 × 40 層 = 5720 ≥ 觸碰集合）。
   SpAc EMA 踢錯專家（容量 98.9% 可覆蓋 vs 實拿 65.9%，§EN-482）才是 miss 來源。
2. **LAYER_CAPS（容量分配）不是解**：總容量夠，錯在替換策略。LAYER_CAPS 只改分配不修踢換。
   ⇒ 對 B 方案無直接幫助；B 方案用「錯層重算」兜 miss，不依賴 miss 消失。
3. **旁路成本**：每 step ~5–6 層有新 expert（87% 層 prev_token 全中，§EN-4xx 重合率）
   → 錯層重算 ≈ 5–6 層 × 1.6ms（moe union/40）≈ **8–10ms/step**。

## 2. LAYER_CAPS 容量修正可行性

| 前提 | 驗證 |
|---|---|
| 8GiB pool 裝得下 143 slots/層全常駐 | ✅ 143 × 40 × 1.07 MiB = **6.1 GiB ≤ 8 GiB** |
| 全常駐 ⇒ miss→0 | ⚠ 反證：480 tokens 仍 4010 miss ⇒ 常駐集合≠路由集合，替換錯配是主因 |
| LAYER_CAPS 可改每層容量 | ✅ `LLAMA_EXPERT_CACHE_LAYER_CAPS="start-end:cap;..."`，loader+cache 共用 |

**結論**：LAYER_CAPS 機制可用（若未來要按路由直方圖重分配），但**對 B 方案的 miss 旁路無直接價值**
——B 方案靠「錯層重算」兜底，不賭 miss 消失。

## 3. B 方案實作設計（無 kernel 改動版）

**核心洞察（本輪新結論）**：kernel 對映表（原難點 3）**不需要**——
ids 已是 slot 空間（hook remap 寫 `st[e]`），只要把「每層 argsort 後寫 ids」改成「step 前一次
用 prev_token 預測寫全部層 ids」，一次提交時 kernel 直接用預測 ids（現狀 kernel 零改動）。

**架構**（每 step）：
1. **step 前**：CPU 用 prev_token routing（上一步實際）remap → 寫 40 層 ids（prev_token 的
   experts 上一步已 fill ⇒ 全 resident ⇒ 查表即中）→ **一次提交整圖**（Metal 全速排層間依賴）
2. **sync 後**：CPU 讀每層 argsort 實際輸出 → 比對預測 → **錯層**（~5–6 層，13% 新 expert）
3. **錯層重算**：fill 錯層的新 expert（~8.4 miss/step × 1.07MiB ≈ 9MB，3.3GB/s ⇒ ~2.7ms）+
   該層 moe 子圖重算（1.6ms/層）+ 正確 ids 覆蓋
4. **fill 後台化**：步驟 2 之後立即發起下一步預測的 fill（後台 pread，與 GPU 完全重疊）

**收益模型（診斷基線 13.59 t/s / step 79.2ms）**：

| 組成 | 現狀（41 段） | B 版 |
|---|---:|---:|
| 一次提交（Metal 內部排層鏈） | —（分段串行） | ~51ms（診斷 19.61 的 8-token 退化讀數，待足量 token 復測） |
| 錯層重算 | — | 5–6 層 × 1.6ms ≈ 8–10ms |
| miss fill（同步小額） | 已含在 cb | ~2.7ms |
| **step 總計** | **79.2ms（13.59）** | **~62ms（≈16.1 t/s，+18%）** |

**風險**：
1. 一次提交真速（51ms）來自 8-token 退化讀數 ⇒ **第一步必須量足量 token 的一次提交真速**
   （「預測 ids + 一次提交」即可持續生成 ⇒ 可量）
2. 錯層數（~5–6）是數學期望，非直接實測 ⇒ 重算成本待實測
3. bit-exact：錯層重算後輸出 = 正確 ids 的完整計算 ⇒ M1/M2/M3 重基範圍 = 對映表雙射時數值相同
4. 補跑層的 moe 子圖構造（輸入殘差流 + gate/up + ids）需在 dispatch 層做 view

**實作順序（依賴鏈）**：
1. **件 2（dispatch 去串行）**：「step 前預測寫 ids + 一次提交」→ **先量足量 token 一次提交真速**
2. **件 1（miss 旁路）**：sync 後讀實際 + 錯層重算 + fill 後台化
3. **件 3（kernel 對映表）**：**暫不需要**（ids 提前寫足矣）；僅當「step 前預測」無法覆蓋時備用

---

## 資產
- `/tmp/miss_dump.txt`（4010 行 `layer expert`）、`/tmp/missdump_bench.json`（decode 成績）
- `/tmp/missdump.ctrl.log`（server log，含 SpAc/池統計）

---

## 4. 本輪實測追加（2026-09-24 晚，0 新量測外）

### 4.1 一次提交真速（足量 token，關鍵讀數）
| 臂 | 開關 | tokens | decode t/s | step ms |
|---|---|---|---:|---:|
| 分段版（對照） | — | 98 | 13.59 | 79.2 |
| B_SCHEME v1（佔位 remap） | SEG_BATCH + B_SCHEME | **200** | 16.06 | 62.3 |
| **S1 組合** | **SLOT_TABLE_GPU + SEG_BATCH + B_SCHEME v2** | **120** | **16.66** | **60.0** |

- 8-token 停因找到：**CGC-LOOP-GUARD 短語循環保護**（routing 錯→輸出重複「文摘」）強制停，
  非 EOS。`CGC_LOOP_GUARD=0` 後 200/120 token 持續 ✅
- S1 組合的 ids = GPU 端 `get_rows(slot_table, argsort)`（真實 routing 查表），**只有 miss 層錯**
  （佔位），非全部層錯 → **補跑只需修 miss 層（~4-5 層/step）**。

### 4.2 prev-token 層級準確率（判死無 kernel 預測版）
CGC_PREV_TOKEN_PREFETCH=1 + CGC_PREV_PF_DBG=1（改印全部層）→ 58 對 decode 步：
| 指標 | 值 |
|---|---:|
| 層級全中率（8/8） | **0.1%**（3/2320） |
| 每層平均 expert 重合 | 38.8% |
| 每步錯層 | **40/40** |
| 每層平均錯 experts | **4.90/8** |

⇒ **相鄰 decode token 的 routing 差異巨大**（此前「70-90% 重合」在本載體不成立）。
無 kernel B 方案（prev-token 預測 + 錯層重算）**判死**——補跑 40 層 × 1.6ms = 64ms，吃光
一次提交收益。

### 4.3 正解收斂：S1（GPU slot lookup）+ 一次提交 + miss 補跑
| 組成 | step ms | 說明 |
|---|---:|---|
| S1 + 一次提交（實測） | **60.0** | 16.66 t/s（診斷，miss 層未補跑） |
| miss 層重算（~4-5 層 × 1.6ms） | +7~10 | fill 後台化（與下步 GPU 重疊） |
| **總（保守）** | **~67-70** | **14.3-15.0 t/s（+5~10%）** |
| + miss→0（替換策略修正，天花板） | 60.0 | **16.66 t/s（+23%）** |

**下一步選項**：
- **A. miss 補跑實作**（難：子圖構造 + 錯層 moe 重算）→ +5~10%
- **B. miss→0**（中：SpAc EMA 踢錯→熱門優先替換，143 槽可覆蓋 98.9% 路由）→ 0 補跑 → +23% 天花板
- **建議**：先 B（不碰 kernel/子圖，改替換策略即有數據），後 A（補跑兜底）

---

## 5. SPAC_HOT（熱門優先替換）實作+實測：判死撤回（2026-09-24 深夜）

### 5.1 實作（默認 off，未 commit）
- h：`spac_count`（[layer][expert] 累計路由次數，不衰減）
- spac_update：同步 bump count
- pick_slot：`CGC_SPAC_HOT=1` 時 victim = count 最低者（tie: EMA util → LRU）

### 5.2 實測（prod-new 分段版 + CGC_SPAC_HOT=1，同 build）
| 臂 | decode t/s | miss |
|---|---:|---:|
| prod-new（EMA，對照） | 13.59 | 8.4/token |
| **SPAC_HOT** | **7.13** | **10.6/token（3388 個唯一 (layer,expert)）** |

### 5.3 判死原因（實測證據，不是推測）
- **3388 個 miss 幾乎全唯一**（重複率 0.4%）→ **非 thrash**，是「觸碰面 > 容量」的 compulsory miss
- **prev-token 層級重合僅 38.8%**（§4.2）→ **decode 的 routing 窗口持續漂移** → 任何「歷史統計」策略
  （EMA / LFU / 熱門 count）都追不上窗口 → **miss→0 在本載體 decode 下不成立**
- 全體路由重尾（top-143 = 98.8%，MASSCOV）與「當下窗口漂移」**不矛盾**：重尾集合是統計的，
  窗口是時間的——SPAC_HOT 的累計 count 對新進者不公平（count 低→優先被踢），且窗口漂移
  使「歷史熱門」≠「當下需要」→ **比 EMA 更慢適應** → 實測更糟

### 5.4 正解修正
- **B（替換策略 miss→0）撤回**——除非 pool 容量 ≥ 全部觸碰面（16GB 機器不可行）
- **唯一活路 = A：miss 層補跑**（S1 + 一次提交 + 錯層重算）——60ms + 錯層重算成本
- **A 的成本未知項 = 「單層小圖重算的 submit 成本」**——一次提交省 submit（40→1），
  補跑錯層要重新提交（錯層數 × submit）——**若 submit ~10ms/層 → +50ms 不划算；若 ~1-2ms/層 → +8ms 划算（14.3-15.0 t/s）**
- **下一步：量單層小圖 submit 成本**（CGC_GPU_TIMING 現有 log 分析，0 新 code）→ 決定 A 值不值得

---

## 6. A 方案收益判定：submit 便宜，但錯層數吃光收益（2026-09-24 深夜）

### 6.1 submit 成本（實測，CGC-DECPROF 穩態 19 step 中位）
| 通道 | 穩態中位 |
|---|---:|
| submit（41 段總） | **4.06 ms**（total 137.8 的 3%） |
| **單層 submit 估** | **0.10 ms** |

⇒ **submit 不是 A 的瓶頸**——錯層重算的 submit 成本可忽略。

### 6.2 錯層數（實測，§1 miss 分佈重算）
- miss 層數 **40/40**（480 tokens 每層至少 miss）
- 層 0-3：1.1/0.98/1.13/0.67 miss/層/step → **幾乎每 step 全錯（4 層）**
- 層 4-39：0.11 miss/層/step → P(層錯) = 10.5% → **~3.8 層/step**
- **總錯層 ≈ 8 層/step**

### 6.3 A 的帳（S1 組合 60ms 基線）
| 項 | ms |
|---|---:|
| S1 + 一次提交（實測） | 60.0 |
| 補跑 8 錯層 × 1.6-2.5ms（GPU moe 重算） | +12.8~20 |
| **總** | **72.8~80 → 12.5-13.7 t/s** |
| vs 分段 prod-new 13.59 | **+1% ~ -8%** |

⇒ **A（整層重算）的收益幾乎被錯層數吃光**——錯層 ~8 層時，補跑 12.8ms 對沖一次提交的 13.6ms。

### 6.4 誠實結論與轉向
- **A 值得實作的前提 = 錯層 < 4 層**——實測 40/40 層 miss、前 4 層每 step 全錯 → **前提不成立**
- **真正的剩餘選項**：
  1. **同環境 ABBA 量「S1 組合 vs 分段」的真實差**——13.59（A 輪）與 16.66（b3 輪）跨環境，
     差 13.6ms 不純是一次提交的收益 → **先確認一次提交真實值**（0 新 code，3 對交錯）
  2. **kernel 級 miss 處理**（錯層只重算 miss experts / GPU 查表直連）——唯一可能拿到 +23%
     （60ms → 16.66）——但改 kernel，工程大
- **建議**：先做 1（確認一次提交真實收益）→ 決定 2 值不值得

---

## 7. 同環境 ABBA（修正版）：一次提交 + 不 fill 是真實 2 倍速（2026-09-24 深夜）

### 7.1 第一次 ABBA 作廢
B 臂 env 沒傳（run_arm 忘帶開關）→ A/B 同配置 → 0.6% 假象 → **作廢**

### 7.2 修正版（3 對交錯，B env = S1+SEG_BATCH+B_SCHEME+LOOP_GUARD=0）
| 對 | A（分段 prod-new）t/s | B（S1 組合）t/s |
|---|---:|---:|
| 1 | 8.24 | **18.08** |
| 2 | 7.81 | **17.72** |
| 3 | 7.93 | **17.78** |

**同環境配對：B = 2.2×A（~55ms vs ~125ms/step）**

### 7.3 機制（數帳）
- A 125ms = union ~55 + cb(fill miss，髒環境讀盤) ~31 + gap(41 段串行) ~42
- B 55ms = **union 真計算**（cb 與 gap 都消了——SEG_BATCH 無 hook 不 fill、一次提交無串行）
- **環境越髒，B 優勢越大**（fill 的 swap 成本被完全避開）——b3 那輪較乾淨時 B=16.66 vs 13.59（+23%）

### 7.4 對 2（kernel 級 miss 處理）的意義
- B 的 17.7-18 t/s 是「miss 不處理」的上界（輸出錯，佔位）
- **真實版 = B + miss 處理**：sync 後讀 argsort → 錯層（~8 層）fill + 重算
  → 55 + fill(8-30ms 髒) + 重算(8-13ms) ≈ **71-98ms → 10-14 t/s（髒環境，vs A 8 → +27-75%）**
- **乾淨環境**：55 + fill(~5) + 重算(~13) ≈ 73ms → 13.7 t/s（vs 13.59 持平）——**但**
  **fill 可後台化（與下步 GPU 重疊）→ 62-70ms → 14.3-16.1 t/s（+5-18%）**
- ⇒ **2 值得投入**：kernel 級 miss 處理（現場 fill + 錯層重算）是唯一正路，
  收益 = 一次提交的 2 倍速 + 可後台 fill 的 miss 成本
