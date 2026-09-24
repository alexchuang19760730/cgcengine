# 怎麼消除 gap 44.4 ms

> 日期：2026-09-24 · 0 benchmark（純結帳 + 既有實測溯源）
> 口徑：**交付 cell**（12.57 t/s / step 247.98 ms）；穩態賬用 `STEP_SERIALIZATION_2026-09-23.md`
> §1（中位數 n=221，`segs=41`）。本篇是 §EN-478 的續篇——那篇解釋 gap 是什麼，這篇講怎麼拿掉。

---

## 1. 先確定一件事：gap 不是「GPU 在等資料」

這決定了所有處方。三條獨立事證：

| 事證 | 數值 | 出處 |
|---|---|---|
| `gap` vs `cb+submit` 回歸 | **r = 0.957，slope = 1.05**，截距 +10.1 ms | `STEP_SERIALIZATION` §1 |
| MoE 在 segment 中的位置 | 永遠在第一個 buffer（**5811/5811**）⇒ 可遮窗口 = 0 | `L3_WINDOW_ZERO` |
| 提前 fill 的實測 | ρ：hit +26pp、讀位元組 −64%，但 **t/s −21%**（`fill_wait_us` 44 ms → 15.4 s） | §EN-4xx / `FILL_IDLE_LOCATED` |

⇒ **GPU 空轉 = CPU 在 top-k hook 與 submit 上花的時間**，幾乎 1:1 傳導。
⇒ 任何「把 IO 藏進 GPU 空檔」的做法都註定失敗：**空檔本身就是 CPU 造成的，而 CPU 正是因為要做 hook 才卡在那裡**。這是循環，不是洞。

---

## 2. gap 的成分（按回歸拆）

```
回歸預測：1.05 × (cb 23.8 + submit 10.8) + 10.1 = 46.4 ms   實測 44.4 ms（差 2.0）
```

| 成分 | ms | 佔 gap | 是誰的時間 |
|---|---:|---:|---|
| ① `cb`：CPU 在 top-k hook 阻塞填池 | 25.0 | 56% | CPU |
| ② `submit`：CPU 提交下一段 | 11.3 | 26% | CPU |
| ③ 啟動偏斜截距（41 段？） | 10.1 | 23% | 段間同步 |

**`cb` 23.8 內部再拆**（`DEVICE_BUSY_ATTRIBUTED`）：固定項每層 barrier ≈ **12.9 ms/step**、
真搬資料 ≈ 11.0 ms/step。**超過一半的 `cb` 根本不是 IO** ⇒ 打「IO 太慢」這個靶是錯的。

**截距 10.1 ms 是什麼（還沒查）**：41 段 ⇒ 0.246 ms/段；而 `K0_RESULT` 實測單 command buffer
固定成本僅 **13 µs**，差 **19 倍** ⇒ 它不是 per-buffer 啟動成本，更像段間同步／排空。
⇒ 若屬實，「減少段數」是一條獨立槓桿，但 41 段源自 40 層的資料依賴，不能隨便併。

---

## 3. 三條處方

| 處方 | 做法 | step | t/s | 工程 | 狀態 |
|---|---|---:|---:|---|---|
| **A. 預指派 slot** | 提交前用預測 ids 把圖建好，真 ids 出來只校驗 | 126.6 | **15.99**（+27%） | 中 | 未動 |
| **B. slot 間接化** | ids 傳 slot index 而非 expert index，全鏈路無 CPU，41 段收成 1 段 | 116.6 | **17.36**（+38%） | 大 | 未動 |
| **M-S2** | slot_table 裝置化，讓 submit-ahead 合法 | — | 上界 **19.0** | 工程 | 未動 |
| 地板 | union 113.4 ms 不會因重疊變小 | 113.4 | **17.85** | — | 物理 |

⚠ **M-S2 的 22.9 那端已被否證**（`MILESTONE_MAP` §3：`total → union` 假設 cb 可被 submit-ahead
吸收，實測不能）⇒ **真上界是 19.0**。

### 方案 A 的敏感性（重要：别把上界当承诺）

原文「命中時 `gap → ~10`」的「命中時」三個字是關鍵。真實值要乘命中率 h：

```
gap_eff = 44.4 × (1−h) + 10.0 × h
```

| h | gap_eff | step | t/s | vs 12.57 |
|---:|---:|---:|---:|---:|
| 33% | 33.0 | 149.6 | 13.52 | +8% |
| 50% | 27.2 | 143.8 | 14.07 | +12% |
| 65% | 22.0 | 138.6 | 14.60 | +16% |
| 80% | 16.9 | 133.5 | 15.16 | +21% |
| 100% | 10.0 | 126.6 | 15.99 | +27% |

**h = 33% 是天真預指派的落點**（`0.87⁸`，方案 A 原文警示的 per-expert 複用率）
⇒ 只到 **13.52 t/s**。要往上必須用**比 top-8 寬的候選集**（原文建議上一 token 的 top-16）。

### ★ 今晚拿到的一塊拼圖：PIN_PROFILE 讓 A 變便宜

方案 A 原文列了三個已知風險，其中兩個今晚有解：

- **「slot 地址會因踢換而失效」** ⇒ `PIN_PROFILE` 實測 **5680 專家 static pin**
  （`llama-expert-cache.cpp:2440`，load 時 `ensure_slot` + `slot_pinned_static=1`；
  `pick_slot` 在 `pass < 2` 跳過 static pin `:628`）⇒ **slot 地址 load 時定死，可提前算**。
- **「候選 expert 不在池裡」** ⇒ 路由重尾實測 **143 個專家吃 98.8% 訪問**
  （`CGC_MASSCOV` counterfactual 與 `ROUTE_DUMP` 兩個獨立口徑一致）
  ⇒ 候選集裡的 expert 有 ~98.8% 機率屬於 pin 集合。

⇒ **PIN_PROFILE 不是 t/s 優化，是方案 A 的基礎設施。** 這也是為什麼
§EN-476 第 1 步（base vs pin 同場配對）要先做——它的價值不止那 12.57~14.24。

⚠ 未解：static pin 在 `pass == 2` 仍會被踢（`:628` 是 `pass < 2`），這是 pin 後剩餘
**9.5 pp** 落差的成因。直接改成「pass 2 也跳過」會讓 `pick_slot` 回 −1 ⇒ 觸發
`table[e]==-1 → OOB pool row → NaN cascade`，**要連 caller 契約一起改**。

---

## 4. 已判死，別碰

| 路線 | 否決事證 |
|---|---|
| 提前 fill / 背景預取（含 ρ） | `L3_WINDOW_ZERO`：MoE 永遠在 segment 第一個 buffer ⇒ 可遮窗口 = 0。ρ 實測 hit +26pp 但 t/s −21% |
| 繼續優化 fill 的 IO | F3 CLOSED（1760 MiB/s，已在 RAM 拷貝速度之上）；且 fill 只佔 cb 的 11 ms，另一半是 barrier |
| 攤薄（方案 D） | 24 ms/step 只攤 2~3.1 token；且 ntok↑ ⇒ union↑ ⇒ fill↑，不免費 |
| 25 t/s | GPU busy 142.6~181.6 ms > 預算 124.7 ms；P < 3% |
| M-K5 / L2 | 判 (C) 成立 ⇒ 上界作廢 ⇒ L2 = 0 |

---

## 5. 建議順序（**不要同時做** —— 所有比較都建立在「同一 build、同一 cell、swap=0」）

```
0  重開機（環境已污染：swap 10018/10240 時 base2 OOM）                    ~5 min
1  base vs PIN_PROFILE 同場配對 ABBA×2                    ~15 min，0 重建
     → 定 12.57~14.24，同時驗證 slot 靜態化
2  量 top-K 候選集命中率（用既有 ROUTE_DUMP + CGC_MASSCOV）  0 重建
     → 定方案 A 的 h，這是方案 A 值的唯一未知數
3  依 h 決定要不要做方案 A（h ≥ 0.65 才值得，對應 +16%）
4  方案 B / M-S2（大工程，且 B 的前提是 miss = 0）
```

## 6. 歸屬

方案 A／B 要動 `llama-expert-cache.cpp` 的**命中／替換／IO 路徑**與圖構建 ⇒ 按 09-20 分工屬
**線 I**（`cb`／快取命中儀器）。本線（WorkBuddy/freebuff）＝靜態分析／長報告 ⇒ 本篇只出結論。
本線可協助寫 `scripts/check/*` 的量測與重放工具。

## 7. 待閉合

- 截距 10.1 ms 的實體（段間同步？排空？）⇒ 決定「減少段數」是否為獨立槓桿
- 方案 A 的 h 沒有實測值（33% 是天真估計，0.854／0.726 是別的口徑，不能直接當 h）
- M-S2 19.0 與方案 B 17.36 的口徑差未釐清
