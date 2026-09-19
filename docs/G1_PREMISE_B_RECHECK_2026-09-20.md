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

## 5. 量測卡（窗口一開照抄）

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
