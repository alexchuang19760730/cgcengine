# RSL-MTP：增益的修正公式、實作方案與可證偽點（2026-09-18）

對象：`docs/MOE_MTP_FEASIBILITY_2026-09-18.md` §4 的 **RSL-MTP（Resident-Set-Locked MTP）**，
以及 `docs/SPEC_COST_CURVE_2026-09-18.md` §6 對它的「1.31–1.88×」投影。

---

## 0. RSL-MTP 是什麼（正本，不重述）

對第 i 個 draft token 先算它的 top-8；若 `top8_i ⊄ 本步已付費的專家並集` ⇒ **在 i 處截斷**。
效果：`union` 恆等於 anchor 的 8/層，不隨 k 增長 ⇒ verify 步的邊際成本只剩 compute + gather 寬度。
代價：accept 從 `a` 降到 `a · p_route`，其中 `p_route = P(top8_{t+i} ⊆ 已駐留集)`。

---

## 1. ★ 先修正投影：**不能用「實測 E」當 RSL 的答案**

`SPEC_COST_CURVE` §6 的投影表寫「若 draft 的 top-8 被鎖在本步已付費的並集內（⇒ `cost → 1`），
則**同一批實測的 `E`** 會直接變成加速比」，並列出 1.31 / 1.36 / 1.66 / 1.88 / 1.66。

**但那與 RSL 自己的定義衝突**：`MOE_MTP_FEASIBILITY` §4.4 明說代價是 `accept → a · p_route`。
兩者不能同時成立：

- 若 `E` 真的不變 ⇒ RSL 沒有代價 ⇒ 那 `p_route` 就不該出現在判別式裡；
- 若 `E` 會降（定義如此）⇒ 投影表**高估**，必須用降完的 `E`。

**⇒ 正確形式（用 MOE_MTP_FEASIBILITY 自己的判別式，只補上 cost 不會剛好等於 1）：**

```
E_RSL(k)  = 1 + p + p² + … + p^k          p = a · p_route
S_RSL(k)  = E_RSL(k) / cost_RSL(k)
```

## 2. `a` 是什麼（我核準過——它不是「接受數」，是幾何級數的底）

從 cost curve 的 5 個點反推 `E(k) = 1 + a + a² + … + a^k`：

| k | a（表上給的） | `1+a+…+a^k` 算出 | 表上實測 E | 差 |
|---|---|---|---|---|
| 1 | 0.306 | 1.306 | 1.306 | 0% |
| 2 | 0.291 | 1.376 | 1.362 | 1.0% |
| 3 | 0.431 | 1.697 | 1.662 | 2.1% |
| 5 | 0.483 | 1.909 | 1.882 | 1.4% |
| 7 | 0.402 | 1.672 | 1.662 | 0.6% |

⇒ **五個點全部吻合（≤2%）⇒ `a` 就是有效 accept rate**，`E` 是它的幾何級數。
（實測 `a` = 0.291–0.483，也解釋了 `E` 為何在 k=7 反而下降：那裡的 `a` 較低。）

## 3. `cost_RSL` 不會剛好是 1（今天的飽和發現）

`union` 鎖回 8 ⇒ union 那一半的成本消失，但**成本不是只有 union**：

- `m = 0.4730` vs `Δunion/8 = 0.4389` ⇒ **約 7% 的成本不在 union**（見 `SPEC_COST_CURVE_REVIEW`）
- `k_eff 3.99→4.68`：union 只 **+0.5%** 而 cost 仍 **+11.3%** ⇒ 飽和區有確定的非-union 成本

⇒ 保守取 **`cost_RSL ≈ 1.07`（k=3 的量級）**。這比「cost → 1」悲觀 7%，但比現狀（2.608）樂觀得多。

## 4. 增益表（k=3、a=0.431、cost_RSL=1.07）

| `p_route` | p = a·p_route | `E_RSL` = 1+p+p²+p³ | `S_RSL` = E/1.07 | 對照：**現狀** S=0.637 |
|---|---|---|---|---|
| 1.00 | 0.431 | 1.697 | **1.59** | ×2.5 |
| 0.75 | 0.323 | 1.450 | **1.36** | ×2.1 |
| **0.50** | 0.216 | 1.273 | **1.19** | ×1.9 |
| 0.25 | 0.108 | 1.121 | **1.05** | ×1.6 |
| 0.10 | 0.043 | 1.045 | 0.98 | ×1.5 |

**⇒ 關鍵：RSL 的增益**幾乎全部來自「`cost` 從 2.608 降到 ~1」**，不是「E 變高」。**
（`E_RSL` 在 `p_route ≤ 0.5` 時反而比實測 `E`=1.662 低——但 `cost` 降得更多。）

## 4.5 ★ 實測結果：`p_route` 幾乎是 0 ⇒ **RSL-MTP 被證偽**

本輪加了儀表（`CGC_P_ROUTE=1`，落點 `llama-context.cpp` 的 union 構建旁；`routes` 是
`[n_tokens][n_expert_used]` 的逐 token top-k 攤平，而 verify 的 batch 就是 `t..t+k`
⇒ anchor 與每個 draft 的集合都在同一份資料裡，**不需要額外前向**）。

跑法：`prod25` + `CGC_PREFILL_STREAM=1` + `CGC_GATHER_SLAB_CAP=256` + `CGC_P_ROUTE=1`、
`-p 0 -n 128 -d 512 -r 3 -b 512 -ub 512 --spec-type draft-mtp --spec-draft-n-max 3`。
（`CGC Soft Pool init: L0=0 L1=0 … partition disabled` ⇒ 與 arm 1 同配置；`anchor_shrunk=0`
⇒ anchor 的 8 個不重複，所以下面的分母是乾淨的。）

```
llama_expert_cache: RSL p_route: steps=9430 anchor_shrunk=0 (0.0%)
   i=1  0.212%  (20/9430)      i=2  1.538%  (111/7216)      i=3  0.198%  (12/6068)
```

| i | 命中/比較 | `p_route(i)` |
|---|---|---|
| 1 | 20 / 9430 | **0.212%** |
| 2 | 111 / 7216 | 1.538% |
| 3 | 12 / 6068 | 0.198% |

**對照**：隨機基準（256 專家、`|S| = 8`）是 `1/C(256,8) ≈ 2.4e-15`，
所以實測比隨機高 11 個數量級 —— **但門檻是 0.5（50%）⇒ 實測差 236 倍。**

**⇒ `MOE_MTP_FEASIBILITY` §5(iii) 的證偽條件成立：`p_route < 0.5` ⇒ RSL-MTP 不值得實作。**
（不是「不值得」，是**不可行**：`accept → a · 0.002` ⇒ `E_RSL ≈ 1.002` ⇒ `S_RSL ≈ 1.0`，
與不做 spec 一樣。）

### 為什麼這個負結果與「相鄰 token 路由相關性很強」不矛盾

`llama-context.cpp:5464` 的 prev-token prefetch 註解寫 *"Adjacent tokens have strong routing
correlation (~70-90% overlap)"* —— **那是「重疊」（`|S_t ∩ S_{t+1}| / 8`），不是「包含」
（`S_{t+1} ⊆ S_t`）。**

| 量 | 意義 | 實測 |
|---|---|---|
| overlap | 兩個 top-8 交集多大 | ~70–90%（既有註解） |
| **subset** | draft 的 8 個**全在** anchor 的 8 個裡 | **0.21%** |

**prefetch 只需要 overlap**（猜中一部分就有用）；**RSL 需要 subset**（要把 union 鎖回 8，
就不能讓 draft 引進任何新專家）⇒ **同一個資料，兩個結論。** 這是 RSL 的設計假設失準的地方：
它假設「既然相鄰 token 的專家高度重疊，那把並集鎖在 anchor 的 8 個損失不大」，
但**損失不是按重疊算的，是按包含算的**。

## 5. 上界（與 m 無關）

`cost ≥ 1` ⇒ `S_RSL ≤ E_RSL ≤ E_max`。實測 `E_max = 1.882`（k=5）⇒
**即使 `p_route = 1` 且 `cost_RSL = 1`，上限也只有 1.88×。**
⇒ **2× 在這條路線上不可達**，與「先別練 head」是同一條判據的兩個切面。

**可證偽門檻**：`p_route < 0.5` ⇒ `S_RSL ≤ 1.19` 且逼近 1.0 ⇒ 不值得實作（原計畫的 0.5 門檻**成立**）。

## 6. 實作方案（**保留供查，但本輪不實作** —— §4.5 已證偽）

> **狀態：不實作。** `p_route = 0.21%` ⇒ 這個方案的代價（accept → `a·0.002`）遠大於它省下的
> `cost`（2.6 → 1）。以下方案留在這裡，是因為**若哪天有人想改「鎖定」的定義**（例如只鎖
> 部分層、或允許並集逐步擴張），這是最小的落點與必須注意的順序。



**落點**：`src/llama.cpp/src/llama-context.cpp` 的 union 構建（`:5320-5329`）——
RSL 開啟且 `n_tokens > 1` 時**只遍歷 `j = 0`（anchor）**，union 因此恆為 anchor 的 top-k。

**為什麼這樣就等價**（不需要「提前算第 i 個 draft 的 top-8」）：

- remap 用的是 **union 索引**，而 `:6831-6835` 的註解明說 *"Experts outside the union map to 0"*
  ⇒ draft token 的額外專家會落到 union 索引 0（**有限值、不會 NaN**）；
- ⇒ 那個 token 的 FFN 用的是**錯的專家權重** ⇒ 它的 logits 是錯的 ⇒ **自然不會被 accept**；
- ⇒ 效果 == 「`top8_i ⊄ anchor top8` 的 token 拒絕」，**正是 RSL 的語義，且零額外判斷**。
- 安全性：MoE 是 per-token 的，draft token 的錯誤**不污染 anchor 的輸出**
  ⇒ anchor 的 logits 仍正確 ⇒ accept 判定對 anchor 有效。

**★ 一個必須處理的細節**：`routes` 有雙重用途。

| 用途 | 需要 |
|---|---|
| `prerouter_score` / `spac_update` / `record_routes` | 截斷**後**（成本才是我們要量的） |
| **`record_proute`（本輪新加的儀表）** | **截斷前**的完整 `routes`（否則它自己失效，`p_route` 永遠量不到） |

⇒ 實作時要先在完整 `routes` 上算 `p_route`，再走截斷後的 union。**順序錯了儀表會說謊。**

## 7. 建議的開關與命名

`CGC_RSL_MTP=1`（opt-in，預設關）——與 `CGC_VERIFY_DECODE` 那類「presence 即真」的既有風格一致。
**不要**做成 profile 的一部分：它改變 accept 語義，必須與 `p_route` 的讀數一起報。

## 8. 未解決

- `cost_RSL` 的 1.07 是**估計**（由 §3 的兩處推論），不是量測。
- RSL 之後 `verify` 的 union 恆為 8 ⇒ 今天 `verify union/call ≈ 18.7` 這個讀數**會變成 8**，
  那將是「RSL 真的生效」的機檢判準（與 `p_route` 一起看）。
