# AcceptMoE 移植評估：先修駐留，再談資格限制

日期：2026-09-24　作者：線 A（WorkBuddy）　狀態：**結論 —— 現在不要抄；先修駐留率**
原文：AcceptMoE, *Commitment-Weighted Self-Sizing Verifier Expert Sets for Efficient MoE
Speculative Decoding*, arXiv 2608.02989（Shuang Liang 等，2026-08-04）
工具：`scripts/check/accept_moe_sizing.py`（selftest 22/22）

---

## 0. 一句話

AcceptMoE 的機制是對的，但它**預設「駐留集合已經覆蓋絕大部分路由」**。
我們這個前提不成立：容量足以覆蓋 **98.9%**，實拿 **65.9%**。
⇒ 現在照抄＝丟掉 **34.1%** 的訪問，去換最多 **+17.5%**。這是壞交易。
⇒ **真正的 bug 是那 33pp 的落差**，它本身值 +13~17%，而且**零精度代價**。

---

## 1. 原文已核（上一輪是轉引，本輪覆核）

摘要原文關鍵句（逐字）：

> "Under offloading, AcceptMoE conditions expert eligibility on **cache residency** instead of
> predicting natural routes and prefetching the corresponding expert weights. Although
> constraining target-expert eligibility **changes the model distribution**, across 12
> model-task pairs spanning three MoE targets and four benchmarks, AcceptMoE's mean accuracy is
> **0.27 percentage points** lower than that of EAGLE-3 speculative decoding with natural routing.
> Served with SGLang at batch size one, it reaches **1.290×** the throughput of this baseline with
> all expert weights in GPU memory, and **2.06×** under physical expert offloading, while reducing
> host-to-device traffic by **73.6% to 77.1%**."

三個機制要拆開看：

| 機制 | 內容 | 我們能不能抄 |
|---|---|---|
| **① residency 條件化資格** | 不用預測路由＋預取，改成「只有駐留的專家有資格」 | 能掛（有 `slot_table`），但**價格取決於駐留率** ← 本文重點 |
| **② commitment-weighted** | 用離線估的「樹節點會不會被接受」的機率加權 | **匹配度差**：它是為 **tree draft**（EAGLE-3）設計的，MTP 是 **chain**，commitment 衰減慢（見 §4） |
| **③ self-sizing** | 自動決定每塊驗證的專家數，免使用者給 budget | 框架可抄，但需要「每步可容忍的流量」這個輸入，我們沒有 |

⚠ ② 是本文最重要的一條負面結論，下面 §4 單獨講。

---

## 2. 決定性數字：駐留落差 33pp　⚠ **2026-09-24 03:40 更正為 17.4pp**

> ### ⚠ 本節的 33pp 是 **prefill 主導形狀**下的數，decode 真值是 **17.4pp**
>
> `masscov_base_2026-09-23.txt` 產自 `pin_abba.sh` 的 `-p 2048 -n 128 -d 512`
> ⇒ prefill 2560 token 佔 **91%**，池在 prefill 階段**從空開始填** ⇒ 冷訪問被 prefill 大量貢獻。
>
> 本輪在 **decode 主導形狀**（`-p 256 -n 512`，decode 佔 67%）重測
> （`scripts/check/masscov_decode_shape.sh`，產物 `masscov_decode_shape_code.txt`）：
>
> | 形狀 | `cur` | 冷訪問 mass | 冷訪問 count | `k143` | **落差** |
> |---|---:|---:|---:|---:|---:|
> | prefill 主導 | 0.6585 | 34.15% | 34.16% | 0.9888 | **33.0pp** |
> | **decode 主導** | **0.7874** | **21.26%** | **21.22%** | **0.9610** | **17.4pp** |
>
> ⇒ 前面所有「代價 34.1%」「修駐留 +13.8%」的數字都要**下修約一半**。
> 詳見 `docs/ROUTE_OVERLAP_CROSSPROMPT_2026-09-24.md` §5。
> （兩個口徑仍然重合：34.15/34.16、21.26/21.22 ⇒ §2.3 的 mass-vs-count 擔憂**對 base 不成立**。）

來源：`scripts/check/pin_profiles/masscov_base_2026-09-23.txt`（40 層，`ROUTE_DUMP`＋`CGC_MASSCOV` 實跑產物）。
用 `accept_moe_sizing.py --masscov` 可重算。

```
cur      mean=0.6585  min=0.5238  max=0.7847     ← 目前駐留集合的覆蓋率
selcold  mean=0.3416  min=0.2155  max=0.4762     ← 落在非駐留專家上的訪問比例
k96      mean=0.9565                              ← 靜態駐留前 96 高頻專家 → 覆蓋 95.65%
k128     mean=0.9820
k143     mean=0.9888                              ← 我們池子就是 143 slots/層 → 覆蓋 98.88%
k192     mean=0.9984
```

### 2.1 容量早就夠了，問題不是容量

**靜態放 96 個專家就覆蓋 95.65%**，而我們有 143 個 slot 卻只拿到 65.85%。
多給了 49% 的容量，換來 **−30pp** 的覆蓋率 ⇒ 這是**替換策略把容量浪費掉**，不是容量不足。

（`k143 = 0.9888` 與記憶裡「143 專家吃 98.8% 訪問」對上，互證。）

### 2.2 池子是滿的 ⇒ 不是「填不滿」造成的

上一輪 ρ 回歸實測：base（`on`）臂 **resident = 6430 MiB**（滿池），
而 ρ 臂掉到 3310 MiB（slot 被「已 claim 未填滿」占住）。
⇒ base 的 65.85% **不是**因為池子沒填滿，是**填滿了但裝錯人**。

### 2.3 口徑已閉合（2026-09-24 03:0x，讀碼定案，0 GPU）

**同一行 masscov 裡混了兩個口徑**，這是坑的源頭。依 `llama-expert-cache.cpp`
`masscov_record()`（:2041-2070）與 dump（:3084-3105）逐行對碼：

| 欄位 | 算式 | 口徑 |
|---|---|---|
| `total` | `massc_total[l] += w` | **MASS**（路由權重 w 的累計） |
| `cur` | `massc_cur_cov[l] / massc_total[l]`，僅 `st[e] >= 0` 才加 | **MASS** 駐留比例 |
| `k96/k128/k143/k192` | 依 `massc_mass[e]` 降冪取前 N 除以 `total` | **MASS** |
| `selcold` | `massc_sel_cold[l] / massc_sel_total[l]`，`st[e] < 0` 才加 | **COUNT**（逐次選擇，含重複） |
| `sel` | `massc_sel_total[l]++` | **COUNT** |

⇒ **`cur` 是 mass、`selcold` 是 count**，兩者在同一行相鄰印出，極易混用。
（檔頭註釋自己就矛盾：`:2062` 寫 "count ratio, not mass"、`:3025` 寫 "MASS, not counts"。）

**第三個口徑**：llama-bench 的 `hit%` = `n_hits / n_requests`，
而 `n_requests` 在 `ensure_batch` 是 `+= n`（:1064，該層該步的專家請求數）、
在 `ensure_slot` 是 `++`（:877）⇒ **逐「專家請求」計數**，與 masscov 的逐次選擇
**分母不同**（後者含 4 個 token 選到同一專家的重複）。

**實測對照**：

- **base**：mass-resident 65.85% / count-resident 65.84% / llama-bench hit 67.9~68.0%
  ⇒ **三者差 <2.2pp，base 上口徑不是問題**。
- **pin**：masscov 89.3~89.4%，llama-bench hit 73~75% ⇒ **差 ~15pp，是真實分歧**。
  方向符合「冷專家多半只被 1 個 token 選中」（冷專家若只被 1 token 選，
  `cold_by_sel / cold_by_req ≈ 19.36/32 = 0.605`；實測 10.64/25 = 0.43，同階）。

⇒ **AcceptMoE 定價請用這兩個數，且標明口徑**：
精度代價（丟掉的質量）= `1 − cur` = **34.15%（mass）**；
IO 收益（少搬的次數）應取 **llama-bench 的逐請求口徑**，不是 `selcold`。

---

## 3. 價格：抄了要付多少、能賺多少

`accept_moe_sizing.py` 用錨點（per-step 161.0 ms、cb 23.8 ms、steps/round 1.55、
tok/round 3.117、錨點 12.57 t/s）算：

| 情境 | t/s | vs 錨點 |
|---|---:|---:|
| 現在（12.57） | 12.57 | — |
| 流量削減 73.6%（AcceptMoE 下界） | 14.11 | **+12.3%** |
| 流量削減 77.1%（AcceptMoE 上界） | 14.20 | **+13.0%** |
| **cb 全部消失（結構上限）** | **14.77** | **+17.5%** |

**代價**：丟掉 **34.1%** 的 (token, expert) 訪問（精度代理），對照 AcceptMoE 宣稱 **0.27pp**。

### 3.1 為什麼 AcceptMoE 能拿 2.06× 而我們只有 +13%

不是它比較厲害，是**分母不同**：

- 它的 baseline 是 **transfer-dominated**（offloading 下搬運佔大頭）⇒ 砍 73.6% 流量 ≈ 翻倍。
- 我們的 **cb 只佔 per-step 的 14.8%**（23.8 / 161.0）⇒ 就算砍到 0，也只有 +17.5%。

⇒ **任何「減少搬運」的處方在我們這裡的上限就是 +17.5%**。
這不是 AcceptMoE 的特色，是所有這類處方的共同天花板（`L3_WINDOW_ZERO`、ρ、prebind 都一樣）。

### 3.2 二階收益（未量測，別拿來決策）

若丟掉的專家也減少 GPU 側 MoE 權重頻寬：MoE core 佔 segment 60.1%，
GPU busy ~142.6 ms/round ⇒ 悲觀估約 **+13%**。樂觀合計 ~+25%。
但 ntok=4 的 batch 會**共用**同一專家的權重載入，實際折扣遠低於訪問數比例
⇒ **這個數必須實測，目前不可引用。**

---

## 4. 為什麼 commitment-weighted 在我們這裡打折（機制 ②）

AcceptMoE 的 ② 是為 **tree draft** 設計的：樹上大部分節點 commitment 很低，
限制它們的專家幾乎不影響最終輸出。**MTP 是 chain**：第 p 個草稿位置只有在
前 p−1 個全被接受時才有用，而我們的接受率是高的那側。

由「每回合 3.117 token / 4 個位置」反解每位置接受率 q：
`q + q² + q³ + q⁴ = 3.117` ⇒ **q ≈ 0.90**。

| 位置 | commitment |
|---|---|
| 1 | ~1.00 |
| 2 | ~0.90 |
| 3 | ~0.81 |
| 4 | ~0.73 |

⇒ 連最尾的位置都有 **73%** 的機率真的會被用上。
「只限制低 commitment 位置」在 chain 上**省不到多少、風險不小**，
不像 tree 上有一大把 commitment ≈ 0.1 的節點可以隨便砍。

**結論**：② 在我們這裡最多只能動位置 3~4，且要付 ~0.7~0.8 的 commitment 風險。
它是「把 AcceptMoE 從 −34% 代價降到 −10% 左右」的調節器，**不是主要收益來源**。

---

## 5. 駐留落差：零精度代價的 +7%（⚠ 不是 +13.8%，也不是 33pp）

> **2026-09-24 03:40 重算**：decode 主導形狀實測 `cur`=0.7874、`k143`=0.9610
> ⇒ 落差 **17.4pp**（不是 33pp），冷訪問 **21.26%**（不是 34.1%）。
> 下面這段用 decode 真值重算：
>
> - 冷訪問 21.26% → ~5% ⇒ cb 23.8 ms → ~5.6 ms
> - per-step 161.0 → 142.8 ⇒ round 221.4 ms ⇒ **14.08 t/s（+12.0%）**
> - **零精度代價**（不改模型分佈）
>
> ⚠ 但 §5.1 已實測：「靜態放置」這條路**裝不下**（三 prompt 並集 193 > 143 slots），
> 所以這 17.4pp 只能靠**自適應替換**去拿，不能靠靜態 profile。

如果駐留率從 65.85% 拉到 ~95%（`k96` 證明這個覆蓋率只用 96 slot 就夠）：

- 冷訪問 34.1% → ~5% ⇒ cb 23.8 ms → ~3.5 ms
- per-step 161.0 → 140.7 ⇒ round 218.1 ms ⇒ **14.30 t/s（+13.8%）**　⚠ 此路已被 §5.1 判死
- **而且完全不改模型分佈**（沒有精度代價）

對比：AcceptMoE 要付 34% 訪問才買到 +12.3~13.0%。

> ### ⚠ 5.0 上面這段的「95%」不能當目標（2026-09-24 03:0x 自我更正）
>
> `k96`/`k143` 的算法是**在同一批資料裡取累計質量的前 N 名**（`:3092-3105`），
> 也就是「用答案對答案」的 **in-sample oracle 上界**。
> 而這批資料的來源更致命：
> `scripts/check/rho_fill_ab.sh:34` 的交付 cell 本身就帶 **`--prompt 0`**，
> `pin_abba.sh:9` 的 profile 也來自 `--prompt 0` ⇒ **profile 與評測是同一個 prompt**
> （`.workbuddy/memory/2026-09-23.md:1859` 已記「oracle 泄題」）。
>
> ⇒ **`k96 = 95.65%`／`k143 = 98.88%` 是「已知未來」才拿得到的數**，
> 線上可達的駐留率**沒有任何實測值**。§5 的 +13.8% 因此是**上界的上界**。
>
> 這也把 §5.1 那個矛盾解開了一半：pin 之所以在 masscov 上有 89.3%，
> 是因為它在**同一個 prompt** 上被評測；它不是「方法有效」的證據。
> ⇒ 33pp 裡有多少是「策略可修」、多少是「上界虛高」，**目前分不開**。

### 5.0b 跨 prompt 實測：靜態放置「裝不下」（2026-09-24 03:40，3 個真實文本）

§5.0 說「分不開」，現在分開了 —— `docs/ROUTE_OVERLAP_CROSSPROMPT_2026-09-24.md` 實測：

| | 專家數（per layer） | vs 池 143 slots |
|---|---:|---|
| 三 prompt **交集**（共享核心） | **95.8**（67.5%） | 裝得下 |
| 三 prompt **並集** | **193.4** | **超出 35%** |
| 兩兩 jaccard | 0.612~0.658（隨機基線 0.384） | — |

⇒ **只看 3 個 prompt，需求就是 193 > 143。真實使用有無數 prompt ⇒ 靜態放置判死，
不是準不準的問題，是容量裝不下。**（同場實測：三 prompt 的 hit 全在 96.1~96.3%，
差異部分是低頻專家，LRU 已經把它們的代價壓到很小。）

⇒ **17.4pp 只能靠自適應替換去拿，不能靠靜態 profile。** 這也把 PIN_PROFILE 的 −23%
從「不能泛化」升級成有機制解釋：profile 是 `-p 0` + **隨機 token**（`llama-bench` 預設
`std::rand()%n_vocab`，`--help` 明說 "NOT what the server sees"）＋ 靜態釘住禁止替換。

### 5.1 但是：兩次嘗試都失敗了，而且失敗原因不同

| 嘗試 | 駐留率 | t/s | 失敗原因 |
|---|---:|---:|---|
| PIN_PROFILE（靜態釘住） | mass 0.893 / count 僅 +5~7pp | **−23%** | profile 由 `--prompt 0` + 隨機 token 生成；靜態釘住禁止替換 ⇒ 跨 prompt 失配 ~24% 無法修正 |
| ρ 預取 | count hit 67.9 → **90.9%** | **−50%** | `fill_wait` 炸到 15~20 s（42 KB 小讀、slot 飢餓） |

⇒ **我們從來沒有在「不弄壞別的東西」的前提下拿到高計數駐留率。**
33pp 看得到，但兩條路都沒走到。這是本文件最該被記住的不確定性。

### 5.2 成因未定：三個候選，各有不同修法

1. **替換策略留了一堆一次性專家**（LRU 只認近期）
   ⇒ 支持證據：池子是滿的（6430 MiB）卻只有 66% ⇒ **裝錯人**，不是裝不滿。目前領先。
2. **slot 被 in-flight/部分填充占住**
   ⇒ base 的 `resident` 是滿的 ⇒ 對 base 排除；但對 ρ 臂成立（3310 MiB）。
3. **masscov 口徑 ≠ llama-bench hit**（mass vs count、暖機、快照時機）
   ⇒ **已閉合（§2.3）**：base 三者差 <2.2pp ⇒ 對 base **排除**；
   pin 的 15pp 分歧是真的，但已由「同 prompt 泄題」解釋掉大半，不再是獨立成因。

---

## 6. 建議順序（取代「直接抄 AcceptMoE」）

```
0  ✅ 閉合口徑（2026-09-24 完成，0 GPU）：`cur`/`total`/`kN` = MASS，`selcold`/`sel` = COUNT，
   llama-bench `hit%` = 逐專家請求。base 三者差 <2.2pp。⇒ AcceptMoE 的價格現在是已知數：
   **代價 34.15%（mass）／收益上限 +17.5%（cb 全消失）**。

1  ⚠ 跨 prompt 路由穩定性 —— 提前到第一順位（原為第 2 步）
   因為 §5.0 發現 `k96`/`k143` 是 **in-sample oracle 上界**（profile 與評測同一 prompt），
   「修駐留能到 95%」這句話目前**沒有任何線上證據**。所以它從「確認成因」升級為**前提**。
   0 重建（ROUTE_DUMP 已 env 門控），跑 2 個**不同** prompt，用
   `accept_moe_sizing.py --overlap dumpA dumpB --n 143` 量 top-143 重疊。
   → 重疊高 ⇒ in-sample 上界接近可達，33pp 大半可修 ⇒ 修駐留值 +13~17%。
   → 重疊低 ⇒ 上界虛高，33pp 大半不可修 ⇒ §5 的 +13.8% 也是虛的，只剩 AcceptMoE 的 +12~13%。

2  確認成因（0 GPU，可與 1 並行）：湊齊 base 的 resident MiB / 每步 miss 數 / cur，
   判定 5.2 的 ① 還是 ②。（③ 已排除）

3  修駐留率（零精度代價，+13~17%）—— 歸 線 I（動 llama-expert-cache.cpp）

4  只有 cur ≥ ~0.9 之後才回頭做 AcceptMoE：
   那時它的代價從 34% 降到 ~1~5%，才輪得到它當第二槓桿。
```

---

## 7. 不要做的事

- **不要現在就限制專家資格**。34% 的訪問代價，換 ≤+17.5%，且精度損失沒有基準可對照
  （我們沒有 AcceptMoE 那種 12 model-task pairs 的評測台）。
- **不要把 2.06× 當成我們的目標**。那是以「搬運佔大頭」為前提的加速比；
  我們搬運只佔 14.8%，同機制的天花板是 +17.5%。
- **不要照抄 ② 的 tree 假設**。MTP 是 chain，commitment 衰減太慢。
- **不要用 `k96` 當理由縮池子**。`k96` 是「靜態頻次放置」的上限，線上做不到；
  縮到 96 slot 只會讓 LRU 更慘。（不過它提示池子相對路由需求是**過大**的，
  在 swap 壓力下這件事本身值得另開一格評估。）

---

## 8. 與其他文件的關係

- `docs/S2MOE_VERIFY_COST_ASSESSMENT_2026-09-24.md`：S²-MoE 三機制對表，其中
  「reuse-aware gating 上界 +17.7%」與本文 §3 的 +17.5% 是同一個天花板，互證。
  該文對 AcceptMoE 的轉引（2.06× / 0.27pp）本輪已由原文覆核，**數字無誤**。
- `docs/RHO_BATCH_REGRESSION_2026-09-24.md`：ρ 的「高 hit 但 fill_wait 炸掉」，
  是本文 §5.1 的一臂實測。
- `docs/H_MEASURED_2026-09-24.md`：h=0.373 判死 A（預指派 slot）。
  ⚠ h 是**跨 token 全中率**，與本文的 `cur`（單次訪問駐留率）**不是同一個數**，別混用。
