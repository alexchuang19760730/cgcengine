# 前提 B 的重新量測：儀器存在、gate 錯了、而「47.5% → 16.5%」不是一個效應（2026-09-20，零 GPU）

> 本文件更正 `docs/G1_ACHIEVABLE_CEILING_2026-09-20.md` §3 的一半，並推翻樹上沿用了五天的
> 一條歸因。它的三個產物：**(1) 儀器不必寫，(2) gate 必改（一行），(3) 「前提 B 的答案」
> 目前不是一個可以引用的數字。**

---

## 0. 一句話

`docs/G1_ACHIEVABLE_CEILING` §3 說「前提 B 的儀器還沒寫 ⇒ 這是一個要排窗口的工作」。
**前半是錯的**（儀器三週前就寫好了，在 binary 裡、在 allowlist 裡、已經產出 45 份讀數），
**後半是對的但理由不同**（它在**交付配置上觸發 0 次**，所以那些讀數答不了交付配置的問題）。
而查證的過程順手推翻了一條沿用了五天的歸因：**「churn 從 47.5% 修後降到 16.5%」——
那兩個數字來自兩個不同的 run，而在同一個 run 家族內 churn 自己就散 37.2%–84.4%。**

---

## 1. 儀器早就存在（更正 §3 的前半）

| 事實 | 證據 |
|---|---|
| 計數器實作 | `llama-context.cpp:4663-4838`，註解自標 `[CGC 2026-09-15 premise B, take 2]` |
| 它量的正是前提 B 的問題 | 「**an id the consumer actually READS** differs from the same layer's previous publish」——`take 2` 的註解自己寫明「`chg` above 是**整表**count 而且**是錯的問題**」 |
| 已在現行 binary | `strings libllama.0.dylib` ⇒ `CGC-S1: TABLE-CHURN` ×2、`SEL-DRIFT` ×1 |
| 已在 allowlist | `run_server.sh:1384`（`CGC_S1_TABLE_CHURN`）⇒ **不需要改程式就能用** |
| teardown 會印 | `llama-expert-cache.cpp:2656`，並在未開儀器時明印 `not instrumented`（**預設 0 與量到的 0 長得不一樣**——`llama-expert-cache.h:346-354` 的原話） |

⇒ **「需要一個新儀器」是不成立的。** 這一條的教訓與 09-20 稍早那次同型：**先搜自己已有的產物**。

---

## 2. 但它在**交付配置**上觸發 0 次（支持 §3 的後半）

gate 是 `n_tokens == 1`（`llama-context.cpp:4668`），而 `n_tokens = t->ne[1]`（`:4846`）＝
該步 top-k 張量的 token 數。拿 09-20 02:30 那輪 G6 的兩臂對照（`CGC-DECPROF` 的 `ntok` 欄）：

| 臂 | ntok 分布 | `n_tokens==1` 觸發 | `<=2` 觸發 |
|---|---|---:|---:|
| **arm1 MTP on（＝交付態，`mean_len` 3.375）** | `{2:1, 3:2, 4:107, 8:18}` | **0 / 128** | 1 / 128 |
| arm2 MTP off | `{1:31, 2:2, 3:1, 4:18, 8:8}` | 31 / 60 | 33 / 60 |

⇒ **G1 的目標（`union <= 38*mean_len`）是在 MTP-on 的形狀上講的，而這個儀器在 MTP-on 上
一次都不觸發。** 所以樹上那 45 份讀數**全部來自 MTP-off 的 run**——那是**先驗**，不是答案。

⚠️ **不要用 `<= 2` 修它**（雖然 `llama-context.cpp:4932` 的另一個儀器就是這樣寫的）：在 MTP-on 上
`<= 2` 只覆蓋 **1/128**。§EN-16 選 2 是為了**可讀性**（原話：「skips the 8-token middle」）——
那對一個 divergence 儀器是合理的取捨，對「答案會決定 S2 還是 S3」的這個儀器不是。

---

## 3. ★ 45 份讀數重讀：「47.5% → 16.5%」是一個未受控的比較

### 3.1 分組（`publishes = 39 × graphs`，39 ＝ 被服務的層數）

| publishes | graphs | instrument 步 | 覆蓋率 | churn | 整表 entries | runs | 例子 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1599 | 41 | 10 | 24.4% | **49.2%** | 572 | **20** | `20260917_003440` |
| 1599 | 41 | 10 | 24.4% | **84.4%** | 1472 | **12** | `20260917_112922` |
| 1599 | 41 | 10 | 24.4% | 53.3% | 750 | 4 | `20260917_010402` |
| 1599 | 41 | 10 | 24.4% | 83.8% | 1466 | 2 | `20260917_115617` |
| 1599 | 41 | 10 | 24.4% | 50.8% | 676 | 2 | `20260917_130443` |
| 1599 | 41 | 10 | 24.4% | 37.2% | 448 | 1 | `20260917_131247` |
| 1599 | 41 | 10 | 24.4% | 67.2% | 990 | 1 | `20260917_132734` |
| **1014** | **26** | **21** | **80.8%** | **47.5%** | 1423 | **1** | `20260915_233832` ← **被引用的那個** |
| **2379** | **61** | **29** | 47.5% | **16.5%** | 508 | **1** | `20260918_113550` ← 「修後」 |
| 2635 | 67.6 | 1 | 1.3% | 0.0% | 0 | 1 | `20260918_133556` |

### 3.2 兩個發現

**(a) `47.5%` 與 `16.5%` 各來自一份 run，而它們的步組成不同。**
`47.5%`（`20260915_233832`）那個 run 有 **21 個 instrument 步 / 26 個 graph（80.8%）**——
它幾乎全是 decode 步。`16.5%`（`20260918_113550`）有 **29 / 61（47.5%）**，其餘是
`ntok` 分布 `{1:4, 2:3, 4:3, 6:1, 8:24}` 裡的 24 個 8-token prompt 塊。
⇒ **兩個數字的「分母同類（都是 `n_tokens==1` 的步）但 run 不同」**，中間隔三天、不同 build、
不同池狀態，**沒有任何控制**。所以「**修後大幅下降**」（`.workbuddy/memory/2026-09-18.md:1920`）
**不能成立** —— 它讀起來像一個效應，實際上是一個未受控的兩點比較。

**(b) ★ 更根本：在同一個 run 家族內，churn 自己就散 2.3 倍（37.2% – 84.4%）。**
那 43 份 `publishes=1599` 的 run 有**完全相同的步組成**（41 graphs、10 個 instrument 步、
覆蓋率 24.4%），而 churn 落在兩個叢集：**{37.2, 49.2, 50.8, 53.3, 67.2}** 與 **{83.8, 84.4}**。
同一簽名**逐位元重複 20 次**（`192/198/572`）⇒ 這**不是噪音，是配置**：某個沒被記錄下來的旋鈕
讓 churn 在 2.3 倍範圍內移動。
⇒ **所以 `16.5%` 與 `47.5%` 的差，落在一個已知會移動 2.3 倍的量裡面，而它們的配置差異沒有被記錄。**

### 3.3 但**方向的結論不動**

**44 / 45 份 run 的 `consumed_changed > 0`。** 唯一為 0 的那份（`20260918_133556`）
只有 **1 個 instrument 步**（覆蓋率 1.3%）⇒ 它什麼都沒證明（樣本 35 個 publish）。
⇒ **「被消費的映射會動 ⇒ 表內容不是常數 ⇒ 發布不是冗餘 ⇒ S3 的前提不成立」這個方向仍然成立**，
而它與 `REMAP_ROUNDTRIP_REMOVAL_PLAN` §9.17(c) 的建議（S1 → S3）**相反**、
與 G1 的 `new_question_verdict`（指向 S2/overlap）**一致**。
**但「多少」目前不可引用**，而交付配置上**一個讀數都沒有**。

---

## 4. 修法：一行 gate（補丁已備、已驗、**未套用**）

```
- if (cgc_churn_on && n_tokens == 1) {
+ if (cgc_churn_on && cgc_is_decode_graph(n_tokens, cgc_pool_max_tokens())) {
```

另一個改動只在輸出：逐 graph 行加 `ntok=%lld`，讓放寬後的混合**看得見而不是靜默**。

**為什麼是 `cgc_is_decode_graph(n_tokens, cgc_pool_max_tokens())` 而不是別的字面值**：

1. `cgc_pool_max_tokens()`（`llama-graph.h:23`）自我描述為「**max n_tokens that the expert-cache
   pool path handles** (multi-token decode, e.g. speculative/MTP verify batches)」。
   「這一步走不走池路徑」**就是**前提 B 的問題——池路徑是唯一會發布表的東西。
2. 兩個 helper 都是 `static inline`（`cgc_is_decode_graph`: `llama-cgc-phase.h:141`；
   `cgc_pool_max_tokens`: `llama-graph.h:23`）⇒ **可以從那個 static 自由函式呼叫，
   不需要改簽名**——這正是它是一行而不是三處改動的原因。
3. 同一對 predicate 已經是 `llama-graph.cpp:2022` 與 `:2192` 的判準 ⇒ 讓 churn 的計數
   與 remap 的決策**對「什麼算一步」的定義一致**，而且是 by construction。

**代價（明說）**：這會放進走池路徑的多 token prompt 塊（那個 run 的塊序列是 `2,2,8×21,6,8,2,4,4`）。
它們**確實也發布表**，所以在問題域裡面、不是要濾掉的噪音；但對它們來說「上一步」的語意略有不同
——所以逐 graph 行才要印 `ntok`，讓讀者能分開看，而不是被迫相信一個混合值。

**驗證（在排隊進 build 窗口之前做的）**：

- 乾跑：兩處錨點各命中 1 次；`+26` 行；未加 `--apply` 時真檔**位元不變**（`git status` 乾淨）。
- scratch 副本上套用：`diff` 只有兩個 hunk（gate/註解 4663-4668、print 4772-4775）。
- **語法檢查**：拿 `compile_commands.json` 裡 `llama-context.cpp` 的那條命令做
  `-fsyntax-only` ⇒ **rc=0**；補丁前後 warning 清單**逐項相同（7 = 7，無新增、無消失）**。
  （警告內容是既有的 `%zu` vs `unsigned long long` 與 `drift_argmax` unused，與本補丁無關。）
- 兩個 header（`llama-graph.h:5`、`llama-cgc-phase.h:18`）本來就已 include。

⇒ 補丁：`Backup/patch_churn_gate_20260920.py`（`--apply` 才寫；重跑會說 already patched）。

---

## 5. 量測卡（**已於 2026-09-20 03:0x 執行完畢 —— 見 §7；本節保留為當時的判斷**）

```sh
# 前提：一次 build（會蓋掉別人正在 map 的 dylib ⇒ 見 docs/G4_MEASUREMENT_CARD 的窗口規則）
python3 Backup/patch_churn_gate_20260920.py --apply
cmake --build src/llama.cpp/build --target llama-server -j 8

# 量：交付配置（MTP on）＋ 開 churn 儀器（已在 allowlist，不需改 run_server.sh）
CGC_S1_TABLE_CHURN=1 <交付臂> run_server.sh
# 讀兩行：
#   CGC-S1: TABLE-CHURN graph=N ntok=4 publishes=39 ...        ← 逐 graph，確認 ntok=4 也進來了
#   llama_expert_cache: S1 slot-table: ... consumed_changed=X consumed_unchanged_publishes=Y
#   ⇒ churn = X/(X+Y)。要它的中位與散布，不是單一值（見 §3.2(b)：同一配置可差 2.3 倍）
```

**執行結果見 §7**（補丁已套用並 build；兩輪已量；**但這張卡漏了一件後來才發現的事**：
`consumed_total = n_tokens x n_expert_used` ⇒ 混合族群的讀數會被 `ntok=8` 的 prefill 塊主導，
所以卡的判準要在**加上 rate 權重**之後才讀得對。§7.3 是那個權重的量化。）

**判準**：
- `X/(X+Y)` 的中位若 ≫ 0 ⇒ 前提 B 不成立 ⇒ **S2**（§9.17(c) 的 S1→S3 建議要撤回）
- 若 ≈ 0 ⇒ 前提 B 成立 ⇒ **S3**（而且它的價值不只是省時間：§9.17(e) 記著分段本身引入
  不確定性，3 輪 3 值 ⇒ bit-identity 那道閘門在 S3 之前是擲骰子）
- **必須報散布**：既然同一配置下 churn 在 37%–84% 之間移動，**單點的中位數不足以支撐**這個判決。

---

## 6. 這份文件**不能**支持的

1. ❌「`47.5%` 是錯的」——它是**那個 run 的**正確讀數（`389/819`，來源見
   `.workbuddy/memory/2026-09-15.md:1630`、白皮書 `PREFILL250_DECODE25_..._2355.html:443`）。
   被推翻的是**它的歸因**（「修後降到 16.5%」），不是它的算術。
2. ❌「前提 B 已被量掉、S3 可以劃掉」——交付配置上 **0 讀數**；既有讀數全是 MTP-off。
3. ❌「churn 是 47.5%（或 16.5%）」——**同配置內散 2.3 倍**，不要把任何單點當常數引用。
4. ❌「`n_tokens == 1` 要改成 `<= 2`」——在 MTP-on 上只覆蓋 1/128（§2）。
5. ❌「儀器需要新寫」——它存在、在 binary、在 allowlist（§1）。
6. ⚠️ **未驗**：那個驅動 churn 在 37%–84% 之間移動的旋鈕是什麼。本輪只確認它**存在且未被記錄**。

---

## 7. 量測結果：補丁已套用並 build，交付配置已量到（2026-09-20 03:0x）

### 7.1 補丁與 build（都驗過了）

`Backup/patch_churn_gate_20260920.py --apply` → `cmake --build ... --target llama-server -j 8`
→ 6 秒、rc=0、7 個 warning（與補丁前逐項相同）。驗證：

| 檢查 | 結果 |
|---|---|
| 新格式字串在 binary 裡 | `strings libllama.0.dylib` ⇒ `CGC-S1: TABLE-CHURN graph=%lld ntok=%lld ...` ✓ |
| 舊字串（不含 ntok） | 命中 **0** ⇒ 確實換掉了 |
| build 新鮮度（判準是「輸出有沒有編譯行」） | `.cpp` 02:55:21 → `libllama.0.dylib` 02:55:30 → `llama-server` 02:55:31 ✓ |
| 閘門 | build 前檢查「無 llama 行程、無 listener」，串在同一個分支裡 |

### 7.2 兩輪讀數（交付配置 `prod25` ＋ `p25-s1-churn` 臂；無 SIGTERM，品質閘門全綠）

臂 = `{CGC_GPU_TIMING=1, CGC_DECODE_PROFILE=1, CGC_SLOT_TABLE_GPU=1, CGC_S1_TABLE_CHURN=1}`
（`decode_sweep.py:203` 已定義，不是我新造的）。

| run | 形狀 | publishes | `consumed_changed` | churn | **decode 的 rate 權重** |
|---|---|---:|---:|---:|---:|
| 1 | `--rounds 3 --n-predict 32` | 6595 | 4810 / 6555 | **73.4%** | 19% |
| 2 | `--rounds 3 --n-predict 512` | 12971 | 7150 / 12931 | **55.3%** | 50% |

兩輪都是：`clamped_selected=0`、`clamped_table=0`、`verify-strict refused=0`、
`zero_mapped_selected=0`、**`Received SIGTERM` = 0**（有效樣本）。
⚠️ 兩輪的 thermal 全程 **HEAVY** ⇒ **t/s 不可引用**；但 churn 是計數器，不受散熱影響。

### 7.3 ★ 那 73.4% 只有 19% 是交付態 decode 決定的

`consumed_total = n_tokens x n_expert_used`，所以一個 `ntok=8` 的塊**每步帶 2 倍的 id**
⇒ 它對「率」的貢獻是 `ntok=4` 的兩倍。第一輪的族群與權重：

| ntok | graphs | rate 權重 | 是什麼 |
|---:|---:|---:|---|
| 8 | 104 | **80%** | chunked prefill 的塊（prompt ≈ 208 token / 8） |
| 4 | 48 | **19%** | **交付態的 verify 步** |
| 2 | 4 | 1% | |
| 1 | 4 | 0% | |

⇒ **一個「50:50 的步數」是「33:67 的率權重」**，而第一輪是 19:80。
⇒ 所以 `73.4%` **不能**讀成「交付態 decode 的 churn」。

### 7.4 兩輪的差（73.4% → 55.3%）**不能**直接當成「decode 佔比的效應」

逐 graph 的整表 churn 顯示它在**衰減**：run1 的前 12 個 graph 是
`13% 6% 5% 4% 7% 5% 4% 4% 2% 2% 2%`，後段落到 1–2%。而 **`ntok=4` 類的「後 3 中位」
在 run1 是 1.1%、run2 是 0.5%** ⇒ run2 走得更深，還沒到平台。

⚠️ **而且兩輪的 prefill 前綴逐位元相同**：`ntok=8` 在兩輪都是 **104 個 graph、同一序列**
（`graph=0→0%、1→13%、2→6%…` 完全一致）⇒ 那是**同一段 prompt 處理**。

⇒ 所以「decode 佔比」與「暫態佔比」**在這兩輪裡同時變化**，兩個解釋都活著。
**⇒ 需要固定 decode 步數、只變 prefill 塊數，才能分離。**（見 §7.5）

### 7.5 ★ A/B 這個設計**分離不了** —— 而這是一個要記下來的錯

我原本設計的 A/B 是「**固定 decode 步數、只變 prefill 塊數**」：

| | prompt | prefill 塊/輪 | decode 步/輪 | decode 的 rate 權重 | 總步數 |
|---|---:|---:|---:|---:|---:|
| A `--prefill-target-tokens 512` | 512 | 64 | 32 | 14% | 大 |
| B `--prefill-target-tokens 64` | 64 | 8 | 32 | 57% | 小 |

**問題**：改 prompt 長度必然同時改變**總步數**（A 比 B 長得多）。
而兩個競爭解釋都預測同一個方向 ——

* 若 churn 的主因是 **prefill 的 rate 權重** ⇒ A（權重低）churn 低；
* 若主因是 **暫態**（池在填，§7.4 的衰減）⇒ A（走得深）churn 也低。

⇒ **A < B 在兩種解釋下都成立，所以 A/B 看到任何結果都不能歸因。**
**這個配對是我設計錯的，在跑之前就該看出來** —— 判準是「兩個解釋是否預測同方向」，
而它們預測同方向。同型的教訓本專案已經有很多條（見技能陷阱 38）。

### 7.6 唯一乾淨的路：把 `ntok` 納入計數（**已做 —— 見 §7.8**）

`consumed_changed` 是一個**累計純量**，事後拆不開。要拿到「交付態 decode 的 churn」，
唯一乾淨的做法是讓儀器**按 `n_tokens` 分桶**：

```cpp
// llama-expert-cache.h（新成員）
std::map<int64_t, size_t> n_slot_table_consumed_changed_by_ntok;
std::map<int64_t, size_t> n_slot_table_consumed_same_by_ntok;
// llama-context.cpp（:4815-4821 那兩行旁邊各加一行分桶累加）
// llama-expert-cache.cpp（teardown 按 ntok 逐行印率）
```

成本：3 個檔 ＋ 一次 build（`.h` 被 6 個檔引用 ⇒ 實測 36 秒）。而它讓「交付態 decode 的
churn」變成**直接量到的**而不是外推的。
實作：`Backup/patch_churn_by_ntok_20260920.py`（`.h` 的兩個 `std::map`、`context.cpp` 的兩行
分桶累加、`expert-cache.cpp` 的 teardown 一行；並補上缺的 `#include <map>` / `<set>` ——
**補丁自己的守衛抓到那兩個 include 不存在**）。build 後 `strings` 確認新行在 binary 裡。

### 7.7 替代方案：讓 prefill 的權重降到 ~3%

不改變儀器也能逼近答案：把 **prompt 縮到最小**、把 **decode 拉長**。

| prefill 塊 | decode 步 | decode 的 rate 權重 |
|---:|---:|---:|
| 8 | 128 | 89% |
| 8 | 512 | **97%** |
| 8 | 1024 | 98% |

⇒ 在 97% 的權重下，混合讀數**就是**交付態 decode 的讀數（誤差 ≤3%），
而且它同時把暫態攤薄（512 個 decode 步 vs 32 個）。

⇒ 這是本輪採用的路，結果見 §7.8。

### 7.8 ★★ 結果：交付態 decode 步的 churn = **42.3%**

分桶儀器 build 後，用**與 run2 完全相同的形狀**跑第三次
（`--profile prod25 --arms p25-s1-churn --rounds 3 --warmup 1 --n-predict 512`）：

```
llama_expert_cache: S1 churn by ntok (consumed subset):
    ntok=1     0/612    =  0.0%      MTP draft 步
    ntok=4  3449/8160   = 42.3%      ★ 交付態 decode（MTP verify）步
    ntok=8  3701/4159   = 89.0%      chunked prefill 塊（cgc_pool_max_tokens()）
```

**這一行就是前提 B 的答案，而且它可以直接引用**：

| 讀數 | 值 | 能引用嗎 |
|---|---:|---|
| 混合（run2 / run3 的純量） | 55.3% | ❌ 是 42.3% 與 89.0% 的加權平均，單獨引用會誤導 |
| **交付態 decode（ntok=4）** | **42.3%**（3449/8160） | ✅ **這是前提 B 要的那個數** |
| prefill 塊（ntok=8） | 89.0%（3701/4159） | ⚠️ 事實，但它不是交付態的步 |
| draft 步（ntok=1） | 0.0%（0/612） | ⚠️ 有趣：draft 的表**從不動** |

**★ 分桶沒有改變累加（強驗證）**：run3 的 `consumed_changed=7150`、`consumed_unchanged=5781`
與 run2 **逐位元相同**，而 3449+3701+0 = 7150、8160+4159+612 = 12931 ⇒ 分桶只是把同一個
純量拆開。兩輪的品質閘門也都全綠、都沒有 SIGTERM。

**⇒ 判決：前提 B 不成立。** 42.3% ≠ 0 —— 交付態 decode 步的**被消費映射有 42% 的
（步,層）對會變動** ⇒ 表內容不是常數 ⇒ **發布不是冗餘** ⇒
**S3（收段 40→1）的前提不成立，S2 是對的那條**（與 `REMAP_ROUNDTRIP_REMOVAL_PLAN` §9.17(c)
的 S1→S3 建議相反、與 G1 的 `new_question_verdict` 一致）。

⚠️ **仍然要報散布而不是單點**：§3.2(b) 量到同一個簽名的 run 家族裡 churn 散 37.2%–84.4%
（那是舊儀器、混合族群）；而本節的 42.3% 是**單一 run 的三個桶之一**。
**本輪只有一個 run3**，所以 42.3% 是「一次直接量到的交付態讀數」，
**不是**「交付態 churn 的分布」。要得到分布需要多跑幾次。

### 7.9 這一節之後，G1 的狀態

| 問題 | 答案 |
|---|---|
| 前提 B 的儀器存在嗎 | 是（§1），且在交付配置上**現在**觸發（§7.1） |
| 交付態 decode 的 churn 量到了嗎 | **是：42.3%**（§7.8，直接量到，非外推） |
| S2 還是 S3 | **S2**（前提 B 不成立） |
| 這個數字的分布 | **未知** —— 只有一個 run3（§7.8 的警告） |
