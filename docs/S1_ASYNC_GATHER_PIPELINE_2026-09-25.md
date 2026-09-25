# S1 異步 gather 流水線 — 單段提交下保住正確性的方案（2026-09-25）

> **一句話**：單段提交能消掉 ~36 ms 序列化，但它關掉 hook／demand fill ⇒ expert 不進池 ⇒ garbage。
> 本方案**不預測 ids**（異於 prebind／ρ）、用**真實 ids**，把「按需取 expert」從同步路徑搬到 GPU 後台、
> 與下一步計算**重疊**，再只補算 miss expert 的線性貢獻。

---

## 1. 要解的矛盾（皆為已驗證事實）

| 事實 | 來源 |
|---|---|
| 41 段 88.5 ms → 單段 48.2 ms（×1.83），但 hook=0、`file_reads=0`、正確性未證 | s1-segbatch |
| 被消的 40.3 ms 裡：序列化 ≈ 36.3 ms、fill 同步僅 3.96 ms | S1 TPOT 分解 |
| GPU slot table 與 host 版**逐位相同** | s1-probe |
| MISS_MASK（未填充 expert 貢獻歸零）**逐位正確、零成本** | miss-3a |

**缺口**：讓按需 fill 在「單段／去同步」結構下仍發生，但**不進關鍵路徑**。

---

## 2. 機制（三步流水線 ＋ double-buffer）

### 2.1 常駐於 GPU 的數據結構
- **presence bitmap**：(layer, expert) → 在／不在 pool。
- **slot table**：(layer, expert) → pool slot（s1-probe 已驗可在 GPU）。
- **miss list ring**：GPU 寫、CPU 批量 drain。

### 2.2 每個 decode step

**A. routing ＋ 命中即算（不回讀、不等）**
- gate matmul 在圖內算出 ids。
- 改造 `MUL_MAT_ID` gather kernel：
  - **hit** → 查 slot table、直接從 pool 取權重參與 matmul；
  - **miss** → 把 (layer, expert) 寫進 miss list，該 expert 輸出以 MISS_MASK 置零。
- 當前 step 用已駐留 expert 立即得到**部分結果**，GPU 不停。

**B. CPU 異步 fill（不擋 GPU）**
- 獨立 fill worker，與 GPU 命令隊列**並行**：
  - 非阻塞／批量 drain miss list → 去重、合併、`preadv` SSD → 填空 slot → 更新 presence／slot table。
- 此步不在當前 step 關鍵路徑，與下一步 GPU 計算**重疊**（即所求的 CPU/GPU overlap）。
- **雙緩衝** current/next pool view ＋ Metal event，防「fill 在寫、kernel 在讀」；MAXQ 限在途量、防干涉。

**C. 局部補算（只補 miss、線性疊加）**
- miss expert 填好後，只跑「新 expert 權重 × 該層輸入」小 GEMM，把結果**加回**部分和（非整層重算）。
- 理想：在 sampler 消費輸出前完成。
- 若來不及：須**顯式**處理 stale、驗證不移動 greedy 決策，**不可靜默**。

---

## 3. 與先前判死路線的關鍵差異

| 對象 | 判死理由 | 本方案為何不同 |
|---|---|---|
| **miss-3b** fill batch 化（~2.9%） | 是讓「同步 fill」變便宜 | 本方案讓 fill **離開同步路徑、與 GPU 重疊**，不是同一問 |
| **miss-3c** per-expert 重算 | 依賴 3b；舊估 +13~18% 來自作廢 cell | 用異步 drain＋雙緩衝**補上依賴**；收益**重新乾淨實測**、不沿用舊估 |
| **prebind／ρ／layer-ahead** | 提前預測 ids、受覆蓋率／窗口限制 | **不預測**、用真實 ids，只把取數延後重疊 ⇒ 避開預測覆蓋率問題 |

---

## 4. 風險與未驗證（誠實邊界、成敗點）

1. **時序窗口（最關鍵）**：補算須在 sampler 消費前完成。一步 miss 數 × fill 時間 vs 下一步 GPU 窗口；
   miss 偏高時可能來不及 ⇒ 必須 Stage B 先量。
2. **Metal 單隊列＋雙緩衝複雜度**：event／fence 同步做不好，要麼 stall（白做）、要麼 race（錯）。
3. **異步 bit-exact**：MASK 歸零再補、浮點求和順序，要 M1 驗，不能假設線性疊加必逐位同。
4. **bg fill 與當前 gather 的頻寬／總線爭用**（MAXQ、干涉，已證是真問題）。

---

## 5. 收益（【推算】、非實測、不預先宣稱）

- fill **完全重疊**：趨近單段 48.2 ms ≈ **20.7 t/s**。
- fill **部分重疊**：落 ~52.2 ms ≈ **19.2 t/s（×1.70）**。
- **不宣稱 25**（both／C 已判不可達）。
- 權威數字一律：乾淨 cell、r3、ABBA、帶 thermal／swap 前後快照。

---

## 6. 分階段（每段可證偽、先便宜後貴）

| Stage | 內容 | 成本 |
|---|---|---|
| **A** | 讀碼＋離線：MUL_MAT_ID 改造邊界、miss list／雙緩衝／Metal event 可行性 | 0 GPU |
| **B** | 只測量不改行為：量「一步 miss 的 fill 時間 vs 下一步 GPU 窗口」⇒ **go/no-go** | 1 GPU |
| **C** | 實作（默認 off、可 A/B）：gather miss 分支＋異步 fill worker＋雙緩衝 | 中 |
| **D** | 局部補算＋M1 bit-exact 驗證 | 中 |
| **E** | 乾淨 cell ABBA r3、出 HTML、定級、commit | 完整驗收 |

---

## 7. 現況

- 方案設計完成（本檔），**尚未實施、無實測**。
- 下一步建議 **Stage B**（一次 GPU 給時序窗口定價），不直接寫全套 —— 避免在前提被自己推翻前先實作。
