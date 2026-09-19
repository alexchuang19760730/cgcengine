# 每 draft token 的 46–59 ms 住在哪裡 ＋ `plain_match` 的真假
`docs/MTP_VERIFY_SPLIT_2026-09-19.md` · 2026-09-19 02:4x–03:0x · HEAD `98b44c8c6`

## §0 兩句話

1. **`m` 住在 DRAFT FORWARD，不在 verify 路徑。** 把 verify 批次大小釘住、只拿掉 draft 前向
   （n-gram 自投機），每一步少掉 **28–51 ms／draft token（中位 +27.7）**；而 verify 批次多一個 token
   只值 **約 +10 ms**。所以 `m` 的分解是 **draft forward ≈ 80–90%、verify ≈ 10–20%**。
2. **`plain_match` = TRUE**（同一份位元組、greedy、2 次獨立開機 × 2 reps、262 字元全等）。
   ⇒ `accept 0.465` **不是 verify 路徑的 bug**，`MTP_2X_BOUNDARY` §3 第 1 條那條「免費 15pp」的路
   **在本 carrier 上被實測關閉**。

兩者合起來改變了 25 t/s 的優先序：不要再去修 accept，去修**那顆 nextn 層前向**。

---

## §1 Experiment A：n-gram 控制臂（同一個 verify 批次大小，沒有 draft 前向）

### 1.1 為什麼只有這個形狀能回答

`POOL_BUDGET_COST_DECOMP §6.1b` 的三個候選裡，pool-budget 軸已經殺掉「expert 位元組流量」
（彈性 0.10–0.17）。剩下的兩個 —— **draft 前向本身** 與 **verify 批次多帶的那個 token** ——
都隨 k 線性成長，所以**任何掃 k 或掃 pool 的實驗都分不開它們**。n-gram 自投機從**token 歷史**取 draft，
完全沒有 draft 前向，verify 批次大小不變：

| arm | draft 前向 | verify 批次 |
|---|---|---|
| plain | 無 | T = 1 |
| draft-mtp | **有（每輪 k 步）** | T = 1 + k |
| ngram-map-k | **無** | T = 1 + k |

`m(ngram) ≪ m(mtp)` ⇒ 住在 draft 前向；`m(ngram) ≈ m(mtp)` ⇒ 住在 verify 的每 token 路徑。

### 1.2 這個臂有三個「沉默的零」，三個都被拒絕而不是被回報

| 陷阱 | 機制 | 為什麼會說反話 | 處理 |
|---|---|---|---|
| 歷史為空 | `common_ngram_map_draft()` 在 `inp.size() < 2*size_key + size_m` 時**直接 return**（ngram-map.cpp:229-234） | draft=0 ⇒ 每輪只 verify 1 個 token ⇒ 看起來「verify 很便宜」 | llama-bench 端**拒跑**並要求 `-d ≥ 128`（本輪真實歷史 512 tokens） |
| 歷史為常數 | draft 會生出來，但 verify 批次是 k+1 個**同一個** token | expert union 塌掉 ⇒ 控制臂的 verify 比 MTP 臂便宜 ⇒ 用另一個門把同一個假答案做出來 | 歷史改成**週期 48 的確定性 ramp**：48 < 2×12+48−1，鍵落在搜尋窗內，且提出的 m-gram 是**不同**的 token |
| `n_max` 無界 | llama-bench 過去傳 `.n_max = -1` | n-gram 預設 `size_m = 48` ⇒ 49-token 的 verify 批次，**不是**同一個 k | 控制臂綁 `spec_draft_n_max`（= k） |

### 1.3 實作（`llama-bench.cpp`，未 commit）

- `is_ngram_spec_type()` / `is_ngram_spec_arm()` 兩個**單一來源**的判定，三個消費者共用
  （型別驗證、spec setup、被量測的迴圈）——與 phase 判定同一種紀律。
- 型別驗證從「只准 draft-mtp」放寬到「draft-mtp ＋ 一個 ngram-\* 控制臂」，其他仍然**拒跑**。
- `bench_spec_setup()`：ngram 臂**跳過 MTP draft context** 的建立與 `release_context()`。
- 量測迴圈：ngram 臂的 `.prompt` 指向歷史、`.n_max = k`，且每輪把 `sampled` 重新錨回 ramp。

> **我踩到的坑（記在程式碼註解裡）**：第一版只把**建構**包進 `if (!ngram_arm)`，卻把
> `s.init->release_context()` 留在外面 ⇒ `s.init` 為 null ⇒ **SIGSEGV**。崩潰報告只有 3 個 frame，
> 唯一的理由 frame 就是 `common_speculative_init_result::release_context()` 從 `llama_bench()` 呼叫。
> 兩個半邊都要在同一個 guard 裡。

### 1.4 網格（15 次啟動，5 臂 × 3 reps，臂序每 rep 輪替）

profile `prefill250`、8 GiB pool、`-p 0 -n 128 -d 512 -b 512`、`--load-mode none`。
`ms/step = E / t/s × 1000`；`draft` 是 SPECDBG 每輪的**實測** draft 長度（不是配置的 k）。

| rep（熱態） | plain | mtp1 (d=1.0) | ngram1 (d=1.0) | mtp3 (d≈2.5) | ngram3 (d=1.016) |
|---|---|---|---|---|---|
| 1 · NOMINAL | 113.85 | 139.61 | 111.95 | 234.29 (d=2.678) | 111.75 |
| 2 · MOD/HEAVY | 110.00 | 184.43 | **190.67** ← 離群 | 206.11 (d=2.500) | 123.18 |
| 3 · HEAVY | 128.61 | 175.12 | 123.94 | 244.76 (d=2.473) | 130.37 |
| **中位（每臂）** | **113.85** | **175.12** | **123.94** | **234.29** | **123.18** |

同一格裡的計數器（證明「沒有 draft 前向」是計數器、不是推論）：

```
mtp1   v_union/call = 12.39 / 13.01 / 12.60      draft calls > 0
ngram1 v_union/call = 14.90 / 15.09 / 15.05      draft: calls=0  union=0      <- 零 draft 前向
ngram3 v_union/call = 15.09 / 15.29 / 15.14      draft: calls=0  union=0
mtp3   v_union/call = 18.65 / 18.18 / 18.21
```

### 1.5 兩個讀數與它們互相印證

**(a) 釘住 verify 批次（兩臂 d = 1，都是 2-token 批次），只差一個 draft 前向：**

| rep | mtp1 − ngram1 |
|---|---|
| 1 | **+27.7 ms** |
| 2 | −6.2 ms（ngram1 那格離群） |
| 3 | **+51.2 ms** |
| 中位 | **+27.7 ms** |

**(b) 用 k≈2.5 的那一對（含約 1.5 個額外 verify token，用 (a) 的單價扣掉）：**

| rep | mtp3 − ngram3 | 扣掉 1.5×10 ms 後 ÷ 2.5 draft token |
|---|---|---|
| 1 | +122.5 | 43 ms |
| 2 | +82.9 | 27 ms |
| 3 | +114.4 | 40 ms |

⇒ **draft forward ≈ 28–51 ms／draft token（中央約 40）**，三個獨立估計落在同一帶。

**(c) verify 路徑的單價（同一條 n-gram 臂內，批次大小只差 1）：**
`ngram1` 中位 123.94 − `plain` 中位 113.85 = **+10.1 ms／額外 verify token**。
而 `ngram3`(d=1.016) 對 `ngram1`(d=1.0) 是 **−0.8 ms**（批次幾乎一樣 ⇒ 應該不動）✓ 自我一致。

**(d) 交叉驗證 `m`：** `mtp3` 每輪比 plain 多 120.4 ms、換 2.678 個 draft token ⇒
`m = (120.4/113.85)/2.678 = **0.395**` —— 與 `spec_cost_curve` 獨立量到的 **0.322（server）/ 0.474（bench）**
同一帶 ✓。而 120.4 ms 的分解是 `2.678×40 (draft forward) + 2.678×10 (verify) = 134 ms`，與量到的 120 ms
閉合到 ±12%。

### 1.6 ★ 順手抓到的機制（比結論更可用）

|  | 每 token 的 expert union | 每 token 的牆鐘成本 |
|---|---|---|
| trunk verify（批 T≈2.4–3.5） | 20.31 / 2.4 ≈ **7.8** | ~10 ms |
| **draft forward（T=1）** | **8.00**（= top_k，`calls=300 union=2400`） | **~40 ms** |

**每 token 抓的 expert 數幾乎一樣（7.8 vs 8.0），成本卻差 4 倍。** 所以 draft 前向貴的不是它抓了什麼，
是**單 token 路徑的固定成本**（圖建構 + 同步 + 沒有任何批次可攤）—— 與本專案已量到的
「輪時 ≈ 請求數 × 1.59 ms、截距 ≈ 0」是同一條線。

而 **one nextn 層 = 40 ms**，同一顆模型的主幹一層是 `113.85 / 41 = **2.78 ms**` ⇒ **draft 前向每層貴 14×**。
這不是「nextn 天生貴」，這是可以修的形狀。

---

## §2 Experiment B：`plain_match`

### 2.1 方法

同一支 binary、**同一份權重位元組**、同一個 prompt、greedy、MTP on/off 交錯各兩次開機（`1,0,1,0`）。

- **權重同一性逐 arm 驗證**（不是假設）：`Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`（MTP=0 走的那顆）
  與 `Nail-…-denseIQ4X.gguf`（MTP=1）**realpath 同一路徑、size 13,663,116,512、head-1MiB md5 相同**。
- **greedy 由伺服器自己釘**：`CGC_FORCE_TEMP0=1`（run_server.sh:1379 只在此值為 1 時 export，
  C++ 端 server-common.cpp:1356 同時釘 temperature 與 seed）。
- profile `prod25` + `CGC_PREFILL_STREAM=1`、8 GiB pool、`n_predict=128`、4 個 `memory_pressure`
  視窗全 NOMINAL。

### 2.2 結果

| arm | 4 次請求的輸出 | 逐字元比較 | MTP 計數器 |
|---|---|---|---|
| MTP=1 ×2 開機 | 每個開機內 2 次**完全相同** | 262 字元 | `draft_n_accepted / draft_n = 76/149` ⇒ **accept 51.0%**，mean len 2.53 |
| MTP=0 ×2 開機 | 每個開機內 2 次**完全相同** | 262 字元 | n/a |
| **跨臂** | — | **0 個字元不同（0.0%）** | — |

⇒ **`plain_match = TRUE`**：batch verify 產生的目標分佈與逐 token 解碼**在同一 argmax 上**。

### 2.3 這條線的兩個儀器缺陷（已修，都是同一類）

| 缺陷 | 症狀 | 修法 |
|---|---|---|
| `tokens` 欄位這個 build 不回傳 | 第一版驅動比較**兩個空陣列**、發現相等、印出 **`PLAIN_MATCH TRUE`** —— 零資料的綠燈 | 加 `variants_nonempty()`：只有每個 rep 都 ≥ 8 個單位才發判定；否則 `NO DATA`。比較改用 render 後的文字並在 JSON 裡記 `compared_by` |
| carrier 用正則從 server log 撈 | 4 個 arm 全部 `carrier: None` ⇒「同一份位元組」在**那次執行裡沒有被驗證** | 改從 launcher 自己的 log（每次啟動印一行）撈，並記 realpath + size + head/tail hash；本報告的權重同一性另外用 shell 直接驗（§2.1） |

### 2.4 它**沒有**說什麼（重要）

1. 這是 **argmax 層級**的相等。logits 有微小差異但 argmax 不變，這裡看不到。
   在 temp 0 的 exact-match 接受規則下，這已經足夠回答「accept 是不是被 verify 路徑壓低」。
2. **它不覆蓋 temp > 0 的拒絕取樣**。`min(1, p_t/p_d)` 只在 sampling 下存在
   （`run_server.sh:1807` 自己寫了 ngram draft 的 `dist` 為 null 時走 exact-match）。
   「accept 0.29 → 0.98」（令一個 agent 記在案的那對數字）屬於那個 regime，本輪沒有量。
3. 本輪是 **Nail carrier**。`MTP_HEAD_PROVENANCE_GATE §5` 那條 `plain_match=False` 記於 2026-09-14，
   而 **09-17 那顆「token ≥ 1 用別的 token 的專家」的 ids 修正**正好在它之後 ⇒
   最可能的解釋是**它已經被那次修正關掉了**；本輪是修正後第一次在同 carrier、同形狀上重測。

---

## §3 對 25 t/s 的計畫意味著什麼

| 原本的優先序 | 本輪之後 |
|---|---|
| ① `plain_match` divergence（判 accept 真假） | **已判：TRUE ⇒ 這條路關閉**，accept 0.465–0.53 是真值 |
| ② 量化天花板探針（Ornith） | 降級：accept 若沒有 verify bug，才輪到「head 是按 fp16 base 訓的」這條 |
| ③ ngram 消融（判 m 住哪） | **已判：住 draft forward ⇒ 升為第 1** |
| ④ 加大 k | 維持封死（`a/m = 1.445`） |

**新第 1 名：把那顆 nextn 前向從 14× 主幹層拉回接近 1×。**
算術：若 draft forward 從 ~40 ms 降到 ~3 ms（主幹層的水準），`m` 從 0.39 掉到
`(2.678×(3+10))/113.85/2.678 ≈ **0.114**` ⇒ 在今天的 accept 0.53、k=3 下
`S = (1+3×0.53)/(1+3×0.114) = **2.02×**` ⇒ **12.99 → 26 t/s**。
這正是 M4 的離開條件，而且是**乘法**、不需要動品質。

**下一步（有界、可量）**：給 draft context 掛上同一套 `CGC_DECODE_PROFILE`／`CGC_GPU_TIMING`，
把 40 ms 拆成「圖建構 / 同步 / MoE / 前向」四項，再決定是批次化、快取圖，還是讓 draft 走 pool fast path。

---

## §4 出處與限制

- **程式**：`tools/llama-bench/llama-bench.cpp`（未 commit，`git status` 唯一 dirty 的來源檔）、
  `libllama-bench-impl.dylib` md5 `c0a1207b4984e0123e324003a06c75ec`、
  `llama-bench` md5 `28ae5feed73660da9fa45ec5db626e54`、
  `libllama-common.0.dylib` md5 `bda161fafff1e5e19f6d884dc96ff1c6`、
  `libggml-metal.dylib` md5 `1be366306c604669eb6d061bd05ad47a`（重建前後相同 ⇒ 沒有把別條線的 metal 改動烘進來）。
- **驅動**：`scripts/check/spec_ngram_ablation.py`、`scripts/check/plain_match_ab.py`（新）。
- **原始產物**：`/tmp/ngram_ab/grid.log`（15 行，逐 run）、`/tmp/plain_match/result.json`（4 arm 全文）。
- **熱漂移是這輪最大的敵人**：rep1 全 NOMINAL、rep2 MOD/HEAVY、rep3 全 HEAVY，plain 本身
  110.0 → 128.61（1.17×）。所以**所有結論都用回合內比值或配對差**，絕對 ms 只在同 rep 內引用。
  (§1.5 給了配對與非配對兩個版本，兩者同號。)
- **樣本數 3 reps／臂**，低於本專案慣用的 MIN_KEPT=3；ngram1 在 rep2 有 1 個離群（190.67），
  已在表中標出，中位數不受它主導。
- 這輪**沒有**跑 §7 的 cooldown gate；15 次啟動是連續的，所以熱漂移是本輪的限制而不是被控制的變因。

## §5 可證偽性

1. 若有人把 draft 前向的成本壓到 3 ms 而 `m` 不動 ⇒ §1.5(c) 的分解錯，成本在別的地方。
2. 若在 `CGC_GPU_TIMING` 下 draft context 的 40 ms 大部分落在「MoE」而不是「圖/同步」⇒
   修法要換（不是批次化，是快取/常駐）。
3. 若在 Edge0 carrier（或任何第二顆 base）上 `plain_match` 變 False ⇒ §2 是 carrier-specific，
   第 ① 條要重開。
4. 若 temp > 0 的拒絕取樣下 accept 明顯高於 temp 0 的 exact-match ⇒ §2.4(2) 那條界限會變成主線。
