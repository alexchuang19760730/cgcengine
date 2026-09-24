# oracle all-hit replay：加了它能提升什麼？

日期：2026-09-23 · 方法：零量測推導 · 口徑：交付 cell（12.57 t/s / step 247.98 ms / cb 42.04 ms / hit 61.4%）

來源：FlashMLX（`Anemll/anemll-flash-mlx`）的 `tools/diagnostics/bench_slot_bank_oracle_hits.py`。

---

## 0. 先講清楚：它**提升 0 t/s**

它不是優化，是**診斷**。任何「加上它會變快」的期待都是誤解。

它提升的是**決策的確定性** —— 把「要不要投數小時做池加大」變成一個 30 分鐘的判定。

---

## 1. 三項產出

### ① 把「cb 全藏」從外推變成實測

`docs/REMAINING_LEVERS` §1 那張表的**最右一格（全常駐 15.14）目前是外推的**。

做法：miss 時**不讀盤**，直接把 slot 的 owner 改成該 expert（內容是垃圾）。
⇒ 得到一個「**數學錯誤、時間正確**」的上界。

這能直接反推真實可藏的 `cb`：

| oracle 實測 X | 全藏後 step | 可藏 cb | 判讀 |
|---:|---:|---:|---|
| 13.00 | 239.8 ms | **8.2 ms** | 只能藏 1/5 ⇒ miss 之外有不可消除的開銷 |
| 14.00 | 222.7 ms | 25.3 ms | 邊際 |
| 15.00 | 207.8 ms | 40.2 ms | 接近理論值 |
| 15.14（外推值） | 205.9 ms | 42.1 ms | 完全可藏 |

### ② 命中率曲線 ＋ 自校驗判據

錄下 ids 序列後，**在純 CPU 上用不同 slots/layer 重放**替換策略 ⇒ 得到每個池大小的**真實 hit%**。

- 不需要真的分配記憶體、不需要 rebuild（錄一次，重放是 Python 秒級）
- **★ 自校驗判據**：`N=143`（今天的實測值）重放出來的 hit% **必須落在 61.4% ± 2pp**。
  對不上就說明替換策略沒複刻對 ⇒ **整張表不可外推**，直接停手。

### ③ 替換策略的上界（Belady）⇒ **可能挖出一條不用記憶體的槓桿**

同一份 trace 上可以對比三種策略，**全部免費**：

| 策略 | 說明 |
|---|---|
| LRU | 下界（純存取時間） |
| SpAc EMA | 今天生產用的（`CGC_SPAC=1`） |
| **Belady（最優）** | 預知未來 ⇒ **替換策略的理論上界** |

⇒ Belady − SpAc 的差距 = **「換策略」這條路還剩多少空間**。

---

## 2. ★ 最有力的發現：換策略 ≈ 加 1.5 GiB，而且免費

用 `cb ∝ miss 數` 換算（一階），`hit 61.4% → 75%`：

```
miss 38.6% → 25%  ⇒  cb 42.04 → 27.2 ms  ⇒  step 233.2 ms  ⇒  13.37 t/s  (+6.4%)
```

對照 `REMAINING_LEVERS` 表：

| 手段 | hit | t/s | 代價 |
|---|---:|---:|---|
| 換替換策略到 75% | 75.0% | **13.37** | **0 記憶體** |
| 池 +1.5 GiB（coverage 70.2%） | 75.7% | **13.39** | 要跟 KV cache 搶 |

⇒ **兩者幾乎完全等價（差 0.04 t/s），但一個免費、一個要動記憶體。**

而 §EN-473 已標明「加記憶體」的第 0 步（KV cache 到底佔多少）**還沒做**，且 swap 絕對不能碰（5301 MiB 時 base 12.57 → 4.62）。
⇒ **換策略是繞過第 0 步的一條路。**

---

## 3. 決策分支（這就是它的全部價值）

oracle 實跑一次（~2 min）＋ 重放（秒級），得到 X：

| X | 判決 | 下一步 |
|---:|---|---|
| **< 13.5** | 池加大**整條路作廢** | 轉 `REMAINING_LEVERS` 第二名：attention/GDN kernel 地板 |
| 13.5 ~ 14.5 | 池加大**邊際** | 只做便宜變體（per-expert bytes↓），不動記憶體 |
| **> 14.5** | 池加大**值得做** | 才去執行第 0 步（印 KV cache 佔用） |

⇒ **不做這個，就得先花數小時查記憶體從哪來，才知道這條路值不值。**

---

## 4. 設計（兩個 env gate ＋ 一個重放器）

### gate 1：`CGC_EXPERT_TRACE=<path>`（錄 ids）

錄製點：`llama_expert_cache_ensure_batch()`（`llama-expert-cache.cpp:1002`）—— **host 側 ids 本來就有**，不需要從 GPU 撈。

格式：每筆 `(layer, n, expert_id[n])`。
資料量估算：40 層 × 19.4 ids × ~164 步（gen 512）≈ 127k ids × 4 B ≈ **0.5 MB**。可忽略。

### gate 2：`CGC_ORACLE_HIT=1`（miss 時跳過 IO）

miss 時**只跳過 pread**，其餘全部保持原樣：
- `slot_loading` / `slot_queued` 狀態機
- `slot_pinned` / `slot_pinned_static` / `batch_mask` 的 pin 邏輯
- `spac_update` 的 EMA 更新
- LRU tick

⇒ 只改變「有沒有真的讀 bytes」，不改任何控制流。

### 重放器：`scripts/check/oracle_hit_replay.py`

輸入 trace，給定 `slots/layer = N`，模擬替換策略 ⇒ 輸出 hit%。
純 Python、秒級、可掃 N = 64…300 全曲線。

---

## 5. 陷阱（每一條都會讓結果不可引用）

| # | 陷阱 | 處理 |
|---|---|---|
| 1 | **oracle 跑的內容是垃圾** ⇒ 後續 token 路由會變 ⇒ trace 不再代表真實序列 | **trace 與 oracle 必須分兩次跑**（trace 用正常模式） |
| 2 | 生產用的是 **SpAc EMA-utility victim**（`CGC_SPAC=1`），**不是純 LRU** | 重放要複刻 `spac_util` 的 EMA（`u[e] += bump` 每次出現，再衰減）；複刻不了就用 LRU／Belady **包成區間** |
| 3 | `cb ∝ miss 數` 是一階近似（IO 隊列非線性） | oracle 實測 X 本身就是這個假設的檢驗：X=13 就代表假設破了 |
| 4 | oracle 模式下 hit% 恆為 100% ⇒ 不能再拿它算任何 hit 相關的量 | oracle 只用來定「時間上界」；命中率曲線靠重放器 |
| 5 | 41 段 ping-pong、bg thread 的時序在 oracle 下可能改變 | oracle 的 step 數要與 base 一致，且比對 `fill_wait_us` 是否真的歸零 |

---

## 6. 歸屬：這個不是本線的地盤

按 09-20 的分工裁定：

- **本線（線 A / ace）**：S1／段邊界（`wait`／`gap`／S2）＋ 逐層 KIND×OP 儀器
- **線 I**：**`cb`（expert cache 填池 IO）／快取命中儀器**

oracle all-hit replay 要改的是 `llama-expert-cache.cpp` 的**命中／替換／IO 路徑** ⇒ **歸線 I**。

⚠ 本線（`WorkBuddy/freebuff`）的區是**靜態分析／長報告** ⇒ **本文只出設計，不實作**。
請交給線 I 執行；本線可協助寫重放器（`scripts/check/` 屬本線）。

---

## 7. 一句話

**它不提速，它讓你 30 分鐘就知道「池加大」這條路值不值得投數小時 —— 而且順便可能挖出一條不用記憶體的等值槓桿（換替換策略）。**
