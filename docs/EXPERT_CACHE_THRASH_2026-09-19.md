# Expert-cache thrash: 算出來的分佈，以及它指向的解法（2026-09-19）

問題（使用者原文）：*「token 不斷在不同專家之間跳來跳去，剛載入專家就被踢掉、馬上又要重新載入……
結構性 thrash。我們現在有這問題嗎？」*

累積計數器（compulsory / capacity）只能說「有多少 miss 是重載」，**不能**說那次重載是發生在
被淘汰後 2 步還是 200 步。這個差別就是全部的決策：

- 距離很小（1–3 步）⇒ 是真 thrash，替換策略 / pinning 有救。
- 距離很大（幾十步）⇒ 池子就是比工作集小，只有容量 / 更小的 expert / 不同的路由有救。

## 方法：從已記錄的路由軌跡離線重播，不需要跑 server、也不需要改引擎

`CGC-IDS: ctx=… il=<layer> ntok=<N> <ids…>` 是**每一步、每一層的 top-k 專家 id**（llama-context.cpp）。
池子是**逐層分割**的（`slots_l(cache, layer)`，一層的填充不會碰到別層的 slot），所以
「每一層各自當成獨立的 LRU 快取」不是近似 —— 那就是引擎的實際結構。

軌跡：`Backup/cgc_logs/llama_server_20260904_123016.log`（91 步、40 層、129,728 個 demand）
＋ `…_131433.log`（81 步）。模型同為 `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X`。

工具：`/tmp/thrash_sim.py`（`lru` / `lru2` / `min`=Belady / `filt`=兩次觸及才 admitted / `pin`=頻率前 F 名釘住）

```bash
python3 /tmp/thrash_sim.py Backup/cgc_logs/llama_server_20260904_123016.log \
        Backup/cgc_logs/llama_server_20260904_131433.log --sweep 71,107,143,179,215,251
```

### 先驗證軌跡是真的（不是 09-17 那顆 ids bug 的殘留）

| 檢查 | 結果 | 意義 |
|---|---|---|
| top-8 專家佔 demand 比例 | **16.6% / 26.6%**（均勻分佈會是 3.1%） | 路由是**高度偏斜**的真路由，不是亂數 |
| 模擬 @143 slot 的 capacity 佔比 | **34%** | 引擎實測同一設定是 **26%（08:33）/ 53%（13:10）** ⇒ 落在實測區間內 |
| 模型檔 | 同名同 size（13,663,116,512 B） | 舊 checkout 的檔案已不在，**無法再驗 byte identity** |

## 結果一：reuse distance——**這不是「馬上又要重新載入」**

143 slot/layer（8 GiB@iq3）下的 capacity miss 重載距離（單位：步）：

| 軌跡 | n | p10 | p50 | p90 | max | ≤1 | ≤2 | ≤4 | ≤8 | ≤32 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0916 | 7253 | 5 | **32** | 62 | 62 | **0%** | **0%** | 7% | 15% | 51% |
| 131433 | 890 | 4 | **12** | 39 | 78 | **0%** | **0%** | 27% | 41% | 85% |

**0% 的 capacity miss 發生在被淘汰後 ≤2 步**。中位數 12–32 步。所以：
**不是**「剛載入就被踢掉、馬上又要重新載入」的病態 churn；是工作集超過容量後的
一般 LRU 汰換。單步工作集只有 8（top_k）到 32（MTP verify 4 token ×8）個，池有 143 —— 單步永遠不會自我汰換。

足跡成長率 ≈ **0.56 個新 expert / 層 / token**（364 token ⇒ distinct 204）。
⇒ **143 slot 約等於每層 255 token 的足跡**。跑得比這長就開始 capacity 汰換。

## 結果二：容量曲線

| slots/layer | 池（iq3） | capacity 佔 miss |
|---|---|---|
| 71 | 4 GiB | 64%
| 107 | 6 GiB | 50%
| **143** | **8 GiB** | **34%**（實測 26 / 53）
| 179 | 10 GiB | 19%（實測：短 run 為 0%）
| 215 | 12 GiB | 0.7%
| 251 | 14 GiB | **0%**

## 結果三：**替換策略不是槓桿**（三個可實作的策略都實測比較差）

@143 slot/layer，兩條軌跡合計：

| 策略 | misses | capacity | 相對 LRU |
|---|---|---|---|
| `lru`（現況） | 23992 | 8143 | — |
| `lru2`（LRU-K=2） | 23486 | 7637 | **−6%** |
| `pin`（頻率前 48 名釘住，上限） | 24261 | 8412 | **+3%（更差）** |
| `filt`（兩次觸及才 admitted） | 32325 | 16476 | **+35%（更差）** |
| `min`（Belady，需要未來知識） | 18463 | 2614 | −68%（**不可實作**） |

`pin` 用**整條軌跡**的頻率選前 48 名 —— 那是任何線上釘選的**上界**，而它還是比較差（因為它把
slot 綁給已經降溫的專家）。`filt` 之所以大幅變差：這個工作集的重用距離多在 1–8 步內，
拒絕 admitted 會把本可命中的重用變成兩次 miss。

**Belady 可達 2614，但沒有任何可實作的策略接近它**（LRU-2 只拿 6%）。所以
「不改路由 / 快取替換策略就根治不了」——**替換策略那一半是死的**，證據是三個策略的實測。

## 結果四：逐層重分配也不行（每一層都缺，不是只有幾層）

`S_for_0`（讓該層 capacity miss 歸零所需的最小 slots）：

| | |
|---|---|
| 每層所需 | layer 0:216, 1:192, 2:208, … 38:176, 39:168（**全部 40 層都在 168–224**） |
| 合計需求 | **8112 slots** |
| 8 GiB 的預算 | **5890 slots** |

引擎那行 `layers_distinct_over_slots=4~5` 是**列印當下**的讀數（該輪只有 4–5 層累積超過），
而這條 364-token 的軌跡是**每一層**都在 168–224 ⇒ 沒有「有閒置 slot 的層」可以借。
所以 `LAYER_CAPS` 重分配（上一輪我提的便宜變體）**在這條工作負載上不成立**。

## 結果五：MTP 的 verify batch 不是原因（我差點做成錯誤結論）

把每一步的 32 個 demand 拆成 4 個「單 token」步（**demand 總數不變**，只改批次/順序）：

| | distinct/層 | misses | capacity |
|---|---|---|---|
| 批次（91 步 × 32 demand） | 204 | 15420 | 47.0% |
| 拆開（364 步 × 8 demand） | **204** | 16249 | **49.7%** |

足跡完全相同，批次甚至**略好**（同一批的 union 去重 + 後面 token 重用前面剛載入的）。
教訓：如果只取每步前 8 個 demand（我第一版就是這樣），demand 數會少 4 倍 ⇒ distinct 68、
capacity 0% —— 那是**假的「MTP 是元凶」**。必須固定 demand 總數才可比。

## 解法：唯一在預算內的槓桿是 **per-slot bytes**（= 量化），不是策略、不是重分配

`slots = pool_bytes / (41 × per_slot)`，iq3 的 `per_slot = 1,458,176 B`（binding blk.39，`pool_feasibility.py`）。

要達到「capacity ≈ 0」（215–251 slot/layer）所需的池：

| per_slot | 相當於 | S=143 | S=215（miss 0.7%） | S=251（miss 0%） |
|---|---|---|---|---|
| 1.39 MiB | iq3 現況 | 8.0G | **12.0G** | **14.0G** |
| 1.11 MiB | 0.8× | 6.4G | 9.6G | 11.2G |
| 0.97 MiB | 0.7× | 5.6G | 8.4G | 9.8G |
| **0.70 MiB** | **0.5×（2-bit 級）** | 4.0G | **6.0G** | **7.0G** |

反過來，固定預算下同一個池買到幾個 slot：

| per_slot | 8 GiB | 10 GiB |
|---|---|---|
| 1.39 MiB（現況） | 143 | 179 |
| 1.11 MiB（0.8×） | 179 | 224 |
| 0.97 MiB（0.7×） | 205 | 256 |
| **0.70 MiB（0.5×）** | **287** | 359 |

**⇒ 只要每 slot 縮到 ~0.7 MiB，8 GiB 的池就有 287 slot/layer，capacity miss 歸零 —— 而且
RSS 不變。** 相反地，維持 iq3 的 per_slot 要 12–14 GiB 的池，而這台 16 GB 機器同時要放
12.7 GB 的模型，做不到（20.7 GB over-commit 就是一直在換頁的根因）。

排序（同一個 8 GiB 預算）：

1. **縮 per-slot bytes**（≈0.7 MiB）：唯一能把 capacity miss 壓到 0 又不加 RSS 的路。
   代價是品質 —— 必須用 48 題實測，不能假設。候選：expert 走更低 bits、attention 保持稠密
   （Ornith MTPv2 APEX-I-Compact / IQ4_XS 已在 `models/gguf/`）。
2. **加大池到 10 GiB + 0.8× per-slot**：224 slot/layer，capacity ≈ 0；RSS 只多 2 GiB。
3. 策略、逐層重分配：**已排除**（結果三、結果四）。
4. 把 miss 藏起來（預取 / GPU slot-table fallback）：不減 miss，只改歸屬。以本分佈
   （中位 12–32 步）理論上可預取，但需要預測「哪一顆」= prerouter 家族，前面已量過覆蓋率 19%。

## 誠實邊界

- 軌跡來自 **09-04** 的 run，早於 09-17 的 ids 修正；模型同名同 size，**舊檔已不在，無法再做
  byte-exact 比對**。兩個獨立軌跡的偏斜度與 capacity 佔比都和引擎今天的實測對得上，所以把它當
  「代表這顆模型的路由形狀」是合理的，但不是同一顆 binary 產出的逐位元證據。
- 模擬器我修掉兩個自己的 bug，兩個都會**製造看起來合理的錯答案**：
  (1) `last_access` 在算距離前就被更新 ⇒ 所有距離讀成 0；
  (2) Belady 的 next-use 表只涵蓋當步的 demand ⇒ MIN 竟然比 LRU 差（真 Belady 不可能）= bug 的訊號。
- 「ntok=1 ⇒ capacity 0%」是 **demand 數量的 artifact**，不是 MTP 的因果（見結果五）。
- 池子是逐層分割的假設已用引擎的 `slots_l(cache, layer)` 與 `L3/L4` 佈局核對；若未來改成
  **跨層共享**池，這份模擬就要重寫。
- 只涵蓋 decode（ntok ≤ 8，無 prefill chunk）。prefill 走 slab、不寫池，所以這份分析不適用於 prefill。

## 檔案

- 模擬器：`/tmp/thrash_sim.py`（尚未搬進 `scripts/check/`）
- 啟動器：`scripts/run_server.sh` 新增 `LLAMA_EXPERT_CACHE_MISS_DUMP` 與
  `LLAMA_EXPERT_CACHE_MISS_ATTR_LAYERS` 的 allowlist 區塊（引擎自 2026-09-13 就讀這兩個，
  但**從來沒有 producer** —— 這個專案的老模式：機制在、開關沒接）。本輪的結論**不依賴**它。
- 引擎、其他檔案：**未改動**。機器：**未啟動 server**。
