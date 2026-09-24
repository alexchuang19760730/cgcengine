# FlashMLX 評估：expert cache 換過去值不值？

日期：2026-09-23 · 方法：**零量測**（純靜態推導＋對方公開實測數字）· 口徑：交付 cell（12.57 t/s / step 247.98 ms）

---

## 0. 先分清兩個同名東西

搜索「FlashMLX」會撞到**兩個完全不同的專案**，判決也不同：

| 專案 | 是什麼 | 跟我們的關係 |
|---|---|---|
| **anemll-flash-mlx**（Flash-MoE） | MLX runtime，slot-bank + ids-as-data，SSD 串流 MoE。**targets Qwen3.5-35B-A3B** | **架構幾乎跟我們一樣**，最可比。本文主體。 |
| **szibis/mlx-flash** | Python + Rust sidecar，15 種技術（entropy coding、LCP 驅逐、投機執行） | 招牌技術是「entropy coding 65% smaller」，本文 §2② 單獨判。 |

---

## 1. 架構前提：**不能「只換 expert cache」**

FlashMLX 自己的 README 開宗明義：

> The dense path should stay inside the backend that is already good at it. The sparse path should be reshaped around a stable bank or slot-bank plus ids-as-data.

它的設計假設**整個模型都在 MLX 裡**。所謂「只換 expert cache」在這裡不成立，因為：

1. **pool 就是 Metal buffer**。我們實測 `L4 metal pool: 143 slots/layer`，slot 位址直接餵給 `mul_mat_id` 的 `src0`。換 MLX ⇒ 權重要變成 `mlx.core.array`，而 dense 路徑還在 llama.cpp ⇒ **兩套 runtime、兩條 Metal command queue、兩份記憶體池**，跨框架同步會把一切吃回去。
2. **量化格式不相容**。我們是 `IQ2_S`(2.5625 bpw) ×2 + `IQ3_S`(3.4375 bpw) —— llama.cpp 的 i-quant，MLX 沒有。必須用 `export_mixed_sidecar.py` 從 MLX checkpoint 重新匯出。
3. **模型也要換**。它 targets `Qwen3.5-35B-A3B`；我們是 Qwen3.6-35B-A3B（Nail/Ornith 自定義，含 MTP head）。⇒ 連模型帶量化一起換。

⇒ 「換 expert cache」的實際含義是 **換掉整個推論 runtime**。

---

## 2. 三條硬帳（這三條決定判決）

### ① 量化倒退：MLX 4-bit 的專家比我們大 51%

從 GGUF 直讀（`blk.0.ffn_*_exps.weight`，每專家 1,048,576 elem = 4096 block），與對方 README 的 expert size 對齊：

| 來源 | per-expert | vs 我們 |
|---|---:|---:|
| **我們** UD-IQ3_XXS（IQ2_S ×2 + IQ3_S） | **1.0703 MiB** | — |
| FlashMLX 4-bit | 1.6117 MiB | **+51%** |
| FlashMLX UD-Q2_K_XL | 0.8965 MiB | −16% |

**換成 resident 6153.7 MiB 不變，池覆蓋率會怎樣：**

| 格式 | slots | /層 | 覆蓋率 |
|---|---:|---:|---:|
| 我們 IQ2_S/IQ3_S | 5750 | 143.7 | **56.1%** |
| MLX 4-bit | 3818 | 95.5 | **37.3%** ← 崩 |
| MLX UD-Q2 | 6864 | 171.6 | 67.0% |

⇒ 走 4-bit 是**災難**（覆蓋率 −19pp，hit 會跟著崩，見 §EN-471「hit% 由記憶體能放幾個專家決定」）。
⇒ 唯一不倒退的是 UD-Q2（−16%）。但那只是「換更低 bpw」——**這件事在 llama.cpp 裡直接做就行，不必換框架**。

### ② entropy coding 在本機是**負收益**（mlx-flash 的招牌技術）

mlx-flash 宣稱 "Entropy-coded — 65% smaller"。算一下這台機器吃不吃得下：

```
讀一個專家 1.0703 MiB @ 1638 MiB/s(實測設備) = 0.653 ms
壓到 65% ⇒ 少讀 35% ⇒ 紅利 = 0.229 ms/專家   ← 全部紅利就這些
⇒ 解壓必須在 0.229 ms 內做完 1 個專家 ⇒ 需要 4.6 GiB/s 的 CPU 解壓頻寬
```

| 解壓實作 | 解壓耗時 | 淨 |
|---|---:|---:|
| 單核 Huffman ~150 MB/s | 7.48 ms | **−7.25 ms** |
| 8 執行緒 ~600 MB/s | 1.87 ms | **−1.64 ms** |
| 樂觀 1.2 GB/s | 0.94 ms | **−0.71 ms** |

⇒ **三種情況全虧**。這台機器 SSD 有 1.6 GiB/s，而 CPU 解壓頻寬比它低一個數量級 ⇒ 拿 CPU 換 IO 是反的。
（Apple 的 *LLM in a Flash* 論文與 mlx-flash 的前提是「IO 遠慢於 CPU」；在 M4 + 高速 NVMe 上這個前提不成立。）

### ③ miss 代價斜率：我們的 SSD path 其實**更有效率**

用雙方各自「100% hit → 實際 hit」的兩個點做斜率：

| | miss 率 | 相對全命中損失 | 斜率 |
|---|---:|---:|---:|
| FlashMLX（94.5 → 47.1 @ 82.7%） | 17.3% | 50.2% | **2.90%/pp** |
| 我們（15.14 → 12.57 @ 61.4%） | 38.6% | 17.0% | **0.44%/pp** |

⇒ **同樣 1pp 的 miss，我們只付 0.44%，對方付 2.90%（6.6×）。**

原因很清楚：我們的 miss 走 bg thread + 41 段填充，IO 有部分被藏在 GPU 空檔裡（`cb` 只佔 step 的 17%）；對方 slot-bank 的 miss path 看起來是同步阻塞的（17.3% miss 就吃掉一半吞吐）。

⇒ **換過去在 IO 隱藏這塊不會更好，很可能更差。**

---

## 3. 對方的實測數字不可比（別拿來當目標）

FlashMLX 的 benchmark 全部在 **M5 Max 128 GB**：

| 模式 | tok/s | hit | expert |
|---|---:|---:|---:|
| `--resident-pread-mlx`（packed-bank ceiling） | 101.6 | 100% | 1.69 MB |
| `--resident` | 94.5 | 100% | 1.69 MB |
| `--slot-bank 128` | 47.1 | 82.7% | 1.69 MB |
| `--slot-bank 64` | 42.8 | 80.8% | 1.69 MB |
| `--slot-bank 16` | 17.1 | 59.5% | 1.69 MB |
| UD-Q2 `--slot-bank 128` | 42.2 | 82.5% | 0.94 MB |

**硬體：Apple M4 / 10 核(4P+6E) / GPU 8 核 / 16 GB 統一內存 / MacBook Air（無風扇）。**
對方是 M5 Max 128 GB。帶寬與 GPU 核數都差 4× 以上 ⇒ **t/s 絕對值完全不能對照**。

（線上流傳的「MLX 130 tok/s vs llama.cpp 43 tok/s」是 M4 Pro 64 GB 上**全常駐**跑出來的，我們 16 GB 根本裝不下全模型 ⇒ 那條 3× 宣稱對本機不適用。）

---

## 4. 換過去的成本清單

| 項目 | 代價 |
|---|---|
| 模型 | 換成 mlx-community 的 Qwen3.5-35B-A3B（含重新下載 ~20 GB） |
| 量化 | 重新匯出 expert sidecar；UD-IQ3_XXS 的 i-quant 無法沿用 |
| **MTP / speculative** | **FlashMLX 沒有 MTP**。我們每 step 產 3.117 token（接受率 77.9%），MTP off 實測 9.82 vs on 12.62 = **+28.5%**。換過去這 28.5% 直接歸零。 |
| 生產配置 | prod25 全套（`OA_ASYNC/SPAC/PREFILL_STREAM/MM_BITIDENT/MTP/DENSE_IQ4X`）作廢 |
| 量測口徑 | llama-bench 錨點 12.57、所有歷史 A/B、`.workbuddy/memory` 的結案全部失效 |
| 本機環境 | `import mlx` 目前**失敗**（未安裝） |

---

## 5. 值得抄的三件事（不用換框架就能拿）

| FlashMLX 的東西 | 我們已有？ | 動作 |
|---|---|---|
| slot-bank + ids-as-data（stable bank，換 ids 不換形狀） | ✅ 已有（143 slots/layer pool + ids 索引） | 無 |
| `--prefetch-temporal`（= 我們的 prebind） | ✅ 實測過，已判死 | 無 |
| **`bench_slot_bank_oracle_hits.py`（oracle all-hit replay）** | ❌ **沒有** | **★ 最值得補** |
| `bench_slot_commit.py`（miss 服務時間單獨量） | 部分（有 `fill_wait_us`） | 可補 |

### ★ 推薦動作：補一次 oracle all-hit replay

`docs/REMAINING_LEVERS` §1 那張收益表（+1.5 GiB ⇒ 13.39 / +3.0 GiB ⇒ 14.33 / 全常駐 ⇒ 15.14）**是外推出來的**，假設 `hit ≈ 覆蓋率 + 5.5pp`。

FlashMLX 的做法可以更便宜地把它變成**實測**：錄下 ids 序列，強制 hit（miss 時不讀盤，直接複用現有 slot）⇒ 得到一個**數學錯誤、但時間正確**的上界。

- 不需要真的擠出記憶體（繞過 §EN-473「第 0 步：記憶體從哪來」這個未解前提）
- 不需要重建模型格式
- 直接回答「cb 全藏到底值多少 t/s」，也就是那張表的可信度

---

## 6. 判決

**不換。**

| 理由 | 證據 |
|---|---|
| 量化倒退 | MLX 4-bit per-expert **+51%** ⇒ 覆蓋率 56% → **37%** |
| 唯一不倒退的選項（UD-Q2 −16%）在 llama.cpp 裡直接做即可 | 就是 §EN-473 的「per-expert bytes↓」變體 |
| 招牌技術 entropy coding 在本機負收益 | 需 **4.6 GiB/s** 解壓頻寬才打平，實測三種實作全虧 |
| IO 隱藏不會更好 | miss 代價斜率我們 **0.44%/pp** vs 對方 **2.90%/pp** |
| 會失去 MTP | **+28.5%** 直接歸零 |
| 實測數字不可比 | 對方 M5 Max 128 GB；本機 M4 16 GB 無風扇 |
| 換的不是模組是 runtime | pool 是 Metal buffer、i-quant、MTP head 全綁在一起 |

**真正該做的仍是 `docs/REMAINING_LEVERS_2026-09-23.md` 的順序**，其中第 0 步（印出 KV cache 佔用）決定池能加多大。

---

## 待閉合

- FlashMLX 的 `k=4` vs 我們 `top-8` —— 若屬實，對方每 token 讀的專家量只有我們的一半，那 §2③ 的斜率差會被放大。需確認 Qwen3.5-35B-A3B 的 top-k。
- 對方的 `--slot-bank 128` 47.1 是否含 Python 開銷（可能高估其 miss 代價）。
- 本機 SSD 實測帶寬 1638 MiB/s 是既有結論（`CB_IS_THE_DEVICE_RESULT`），本次未重測。
