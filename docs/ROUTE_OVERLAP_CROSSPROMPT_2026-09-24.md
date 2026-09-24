# 跨 prompt 路由重疊實測（2026-09-24）

> 這一步在 `docs/ACCEPTMOE_ADAPTATION_2026-09-24.md` §6 被排在最前面，
> 因為它同時決定兩件事：**33pp 有多少可修**，以及**「修駐留 +13.8%」是不是真的**。
> 判定標準（用戶給的）：重疊高 ⇒ 靜態／混合放置可行；重疊低 ⇒ 上界虛高，只剩 AcceptMoE 的 +12~13%。
>
> **結論先講：答案是「中間」，而且偏向靜態放置不可行。但過程中翻出兩個更大的發現 ——
> 33pp 被 prefill 放大了一倍多，以及一個 10 倍的口徑矛盾還沒解。**

---

## 1. 方法

| 項目 | 設定 |
|---|---|
| profile | **prod-new**（`pool_sweet_spot.sh:14` 的完整 env，20 個變數） |
| pool | **8 GiB**（`CGC_EXPERT_CACHE_BYTES=8589934592`，= prod-new 預設） |
| MTP | **off**（`CGC_SERVER_MTP=0`，與 prod-new 交付口徑一致） |
| 形狀 | `-p 256 -n 512 -d 0 -r 2` ⇒ **decode 佔 67%**（刻意讓 decode 主導，見 §5） |
| prompt | `--prompt-file`，3 個**領域差異極大**的真實文本 |
| 儀器 | `LLAMA_EXPERT_CACHE_ROUTE_RECORD=1` + `_ROUTE_DUMP`（0 重建，env 門控） |

三個 prompt：

| tag | 領域 | 語言 |
|---|---|---|
| `code` | 執行緒安全 LRU cache 的 C++ 實作 | 英文 |
| `history` | 唐代安史之亂的制度背景與影響 | 繁體中文 |
| `bio` | 真核生物 mRNA 剪接體的分子機制 | 英文 |

### 1.1 為什麼一定要 `--prompt-file`（這是個口徑坑）

`llama-bench --help` 原文：

> `--prompt-file <path>`: fill the prompt/depth with REAL text read from `<path>` (cycled)
> instead of `std::rand()%n_vocab`. **The random fill is NOT what the server sees**, and it
> moves both cache requests/round and draft acceptance.

⇒ **歷來沒給 `--prompt-file` 的 bench，其路由統計都不是真實語言分佈。**
這直接影響既有資產的解讀：`scripts/check/pin_profiles/route_top142_p0_2026-09-23.txt`
是 `--prompt 0`（= `-p 0`）**且沒有 `--prompt-file`** 的產物 ⇒ **它是隨機 token 的路由**，
不是任何真實 prompt 的駐留需求。用它做 PIN_PROFILE 本來就不該期望泛化 ——
這比「同 prompt 泄題」更根本：**連語言都不是**。

已寫入 `.workbuddy/memory/MEMORY_HYGIENE.md`「儀器口徑坑」。

---

## 2. 結果：兩兩重疊（top-142）

隨機基線 = 兩個獨立的 142/256 子集：jaccard `0.3838`、recall `0.5547`。

| pair | jaccard | recall | jaccard min |
|---|---:|---:|---:|
| code vs history | **0.6182** | 0.7625 | 0.4869 |
| code vs bio | **0.6576** | 0.7921 | 0.5778 |
| history vs bio | **0.6121** | 0.7579 | 0.4792 |
| （隨機基線） | 0.3838 | 0.5547 | — |

**三對高度一致（0.612~0.658）** ⇒ 不是「某個 prompt 特殊」，是普遍現象。
相對於隨機基線有顯著的共享結構（0.62 vs 0.38），但**離 1 很遠**。

### 2.1 交集與並集 —— 靜態放置的容量判據

| | 專家數（per layer） | vs 池容量 143 |
|---|---:|---|
| 三 prompt **交集**（共享核心） | **95.8** | 67.5% |
| 兩 prompt 交集 / 並集 | 108.3 / 175.7 | 並集 **超 23%** |
| 三 prompt **並集** | **193.4** | 並集 **超 35%** |

⇒ **只看 3 個 prompt，需求就已經是 193，池只有 143。真實使用有無數 prompt。**
⇒ **純靜態放置（PIN_PROFILE 類）判死**：不是準不準的問題，是**裝不下**。

---

## 3. 結果：hit 對 prompt 完全不敏感

| arm | decode/pool hit |
|---|---:|
| code | 96.2% |
| history | 96.3% |
| bio | 96.3% |
| masscov 驗證臂（同形狀） | 96.1% |

三個領域完全不同的 prompt，**hit 全在 96.1~96.3%**。
⇒ 差異部分（並集 193 裡多出的 50 個）是**低頻專家**，LRU 自適應把它們的 miss 代價壓到很小。

---

## 4. 判定：靜態／混合放置

回到用戶的判定標準：

| 標準 | 實測 | 判定 |
|---|---|---|
| 重疊高 ⇒ 靜態放置可行 | jaccard 0.62，並集 193 > 143 | ❌ **不可行**（裝不下，超 35%） |
| 重疊低 ⇒ 上界虛高，只剩 AcceptMoE | 顯著 > 隨機基線（0.62 vs 0.38） | ⚠ **部分虛高**（見 §5，33pp → 17.4pp） |

⇒ **中間結論**：
- 有真實的共享核心（95.8 個 = 67.5%），所以「路由是模型固有 heavy-tail」這句**對**；
- 但**並集裝不進池子**，所以「靜態放置可行」這句**錯**；
- 而 **LRU 已經拿到 96.1~96.3%**，所以「修駐留」的空間比想像小得多。

### 4.1 PIN_PROFILE 的 −23% 現在有機制解釋了

之前只能說「不能泛化」。現在三個數把它講清楚：

1. profile 是 **`-p 0` + 隨機 token**（不是真實語言）—— §1.1；
2. 換 prompt 時 top-142 失配 **~24%**（recall 0.76）；
3. pin 是**靜態釘住**，slot 不讓 LRU 換掉 ⇒ 失配無法被修正（而 LRU 可以，所以 base 96.1%）。

⇒ **不是「放置無效」，是「靜態 + 隨機 token profile + 禁止替換」三者疊加。**

---

## 5. 33pp 被 prefill 放大了一倍多（本輪最大的更正）

`masscov_base_2026-09-23.txt` 的 34.15% 是在 **`pin_abba.sh` 的形狀**下測的：
`-p 2048 -n 128 -d 512` ⇒ prefill 2560 token 佔 **91%**，decode 只有 256 token。
**池在 prefill 階段從空開始填** ⇒ 冷訪問被 prefill 大量貢獻。

本輪在 **decode 主導形狀**（`-p 256 -n 512`，decode 佔 67%）重測（`masscov_decode_shape.sh`）：

| 形狀 | `cur`（MASS 駐留覆蓋） | 冷訪問 **mass** | 冷訪問 **count**（`selcold`） | `k143` |
|---|---:|---:|---:|---:|
| prefill 主導（`-p 2048 -n 128`） | 0.6585 | **34.15%** | **34.16%** | 0.9888 |
| **decode 主導（`-p 256 -n 512`）** | **0.7874** | **21.26%** | **21.22%** | **0.9610** |

兩個結論：

1. **mass 與 count 始終重合**（34.15/34.16、21.26/21.22，差 <0.05pp）
   ⇒ 在 base（LRU）上這兩個口徑等價，`ACCEPTMOE_ADAPTATION` §2.3 的擔憂**對 base 不成立**。
2. **33pp 有一半多是 prefill 偽影**：真實的 decode 缺口是 **96.10% − 78.74% = 17.4pp**，不是 33pp。
   ⇒ 之前寫的「修駐留到 95% 得 +13.8%」要下修。

---

## 6. ⚠ 未解：hit 96.1% 與 selcold 21.2% 差 10 倍

同一個形狀、同一支 run：

```
decode/pool (ensure_slot+batch) hits=320661/333720 (96.1%)
masscov: cur=0.7874 (冷 21.26%)   selcold=0.2122 (冷 21.22%)
```

一個說冷訪問 3.9%，一個說 21.2%。**這不是暖機或快照時機能糊過去的差距。**

分母也對不上：每層 `sel` = 14344 次，而 `n_requests` = 333720/40 = 8343 次（**1.72×**）。

候選機制（**都未證實**）：
- `masscov_record` 記的是「選中時刻的 `slot_table` 快照」（`:2064` `st[e] < 0`），
  而 `ensure_batch` 記的是「請求是否被滿足」；若 slot 已 claim 但未填滿，兩者會分歧
  （ρ 那輪的 `no_free_slot=3570` 正是這個狀態）；
- 兩者的去重口徑不同（同一專家在同一步被多 token 選中時計 1 次還是 n 次）。

**這對 AcceptMoE 是定價的分母，不能猜：**

| 以哪個為準 | 代價（丟掉的 mass） | 收益（少搬的次數） |
|---|---:|---|
| `selcold` = 21.2% | 21.26% | cb 23.8 ms 可望全消失 ⇒ **+17.5%** |
| `hit` = 96.1%（冷 3.9%） | 21.26% | 只能省 3.9% 的搬運 ⇒ **很小** |

⇒ **下一步不是跑更多臂，是把這兩個計數器對齊**（見 §8）。

---

## 7. decode 速度（本輪副產品，⚠ 非交付口徑）

| arm | t/s | stddev | hit |
|---|---:|---:|---:|
| code | 9.59 | 1.96 | 96.2% |
| history | 9.02 | 1.03 | 96.3% |
| bio | **9.37** | 0.99 | 96.3% |
| **mean ± sd** | **9.33 ± 0.29** | — | **96.27%** |

三個領域完全不同的 prompt，t/s 只差 ±0.29（3.1%）、**hit 只差 0.1pp**
⇒ 本輪形狀下「哪個 prompt」幾乎不影響這兩個數。

**不能直接跟交付數字比**，因為同時變了三個變數：

| | 本輪 | prod-new 交付 |
|---|---|---|
| 形狀 | `-p 256 -n 512 -d 0 -r 2` | `-p 2048 -n 128 -d 512 -r 3` |
| prompt 填充 | `--prompt-file` 真實文本 | `std::rand()%n_vocab` |
| 儀器 | `ROUTE_RECORD=1` 開（mutex + 逐層累加） | 關 |
| t/s | 9.02~9.59 | **11.72**（最新 commit `a1ca0c23a` 標題 11.89） |

差 ~20~24%，但**沒有任何一個變數被單獨隔離過**。要歸因得補「同形狀、關儀器」的對照臂。

---

## 8. 下一步（順序）

```
0  對齊 §6 的兩個計數器（0 GPU，讀碼）
   —— 這是 AcceptMoE 定價的分母，也決定 cb 23.8 ms 到底是誰造成的。

1  同形狀、關 ROUTE_RECORD 的對照臂（量化儀器開銷 + 隨機 fill vs 真實文本的 t/s 差）

2  才回頭定 AcceptMoE 的價格（現在兩個候選值差 4 倍）

3  「修駐留」降級：空間是 17.4pp 不是 33pp，且並集 193 > 143 ⇒ 只能靠自適應替換，
   不能靠靜態放置
```

---

## 9. 資產

| 檔案 | 用途 |
|---|---|
| `scripts/check/route_overlap_3prompt.sh` | 3 prompt 路由 dump driver（含共用閘門） |
| `scripts/check/masscov_decode_shape.sh` | decode 主導形狀的 masscov 驗證（含判據） |
| `scripts/check/accept_moe_sizing.py --overlap A B --n 142` | 重疊計算（selftest 22/22） |
| `scripts/prompts/xprompt_zh_history.txt` | 中文歷史 prompt |
| `scripts/prompts/xprompt_biology.txt` | 分子生物學 prompt |
| `scripts/check/pin_profiles/route_ovl3_{code,history,bio}.txt` | 本輪 3 份 dump |
| `scripts/check/pin_profiles/masscov_decode_shape_code.txt` | decode 主導 masscov |

## 10. 待辦／未閉合

1. **§6 的 10 倍口徑矛盾未解**（最優先，它是定價的分母）。
2. 本輪 t/s 是**非交付口徑**，不能引用進任何 commit 標題或白皮書。
3. 三個 prompt 仍然只是 3 個樣本；並集 193 是**下界**（更多 prompt 只會更大）。
4. 本輪 `-p 256` 的 prefill 只有 256 token，而交付是 `-p 2048`；
   「並集 193」是在短上下文下量的，長上下文可能更大。
