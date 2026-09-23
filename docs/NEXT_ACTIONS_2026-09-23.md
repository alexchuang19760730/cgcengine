# 下一步行動清單 — 2026-09-23

> 這份文件是優先級排序後的待辦事項。請把手邊的事情做完，然後照這個順序做。

> **⚠️ 強制前置（所有速度 A/B）**：2026-09-23 實測證明本機 launch-to-launch 波動 ±20%
> （同一 binary：深冷卻 12.37 vs 淺冷卻 9.9）。**任何 decode/prefill 速度 A/B 必須先讀
> `docs/ABBA_MEASUREMENT_PROTOCOL_2026-09-23.md` 並照做**：交錯（ABBA/bracketed）、
> launch 前深度冷卻 ≥300s（--min-idle-s 300）、交付用中位數（配對比率中位）、log-space
> 校正、窗口守門、build 指紋。缺協議的數字不進報告。

> **🔴 commit 口徑統一（2026-09-23 拍板）**：**研發/實驗階段可以用任何 profile**（prod25、
> prefill250、p25-* 等）快速探索；**commit 統一用 `prod-new`**（單一介面，MTP 開關 = 
> `CGC_SERVER_MTP`：0=decode 支柱，1=prod25-stream 血統，prefill 支柱共用）。除了 `commit_bench.py`
> 的默認就是 prod-new 之外，任何要寫進報告/白皮書/commit 標題的數字，一律以 prod-new 口徑產出。

> **🔴 commit gate（2026-09-23）**：每次 commit 前必須跑
> `python3 scripts/check/commit_bench.py`（prod-new profile，prefill 2048 + decode 128 + d512，
> llama-bench 口徑）。未達閾值（prefill <120 / decode <10）commit 被擋；
> 加 --record-only 只記錄。測量在 ABBA 協議下進行（冷卻 ≥300s、窗口守門）。 **commit 標題必須帶成績**：跑完把 `[commit-gate]` 輸出的那行（prefill=…t/s decode=…t/s）貼進 commit message。



***

## 目前狀態摘要



```
已完成：

&#x20; ✅ P0 實驗跑完（CGC\_DRAFT\_PREFETCH 劑量=0，M 寬度分不出來）

&#x20; ✅ KNOB\_FULL\_MAP\_15（15 個 knobs 全部核對完）

&#x20; ✅ JOINT\_SHAPE\_KERNEL\_AXIS（dense 已判死，MoE 是空間）

&#x20; ✅ INSTRUMENT\_RESOLUTION（搞清楚測量噪音來源 = 熱/功率狀態）

&#x20; ✅ FOOTPRINT\_HALFSTEP（footprint 不是主因，機器自己擺盪才是）

&#x20; ✅ 工具改進（mmid\_shapes.py 會檢查窗口、footprint\_ab.py 新增）

&#x20; ✅ 任務 1：MoE NSG sweep — 沒效果，排除這個方向

&#x20; ✅ 任務 2：k=3 飄動分析 — expert cache IO 抖動 + kernel 換手

&#x20; ✅ 任務 3：Expert cache 大小測試 — 4GB 慢 21%

&#x20; ✅ 任務 4：Knob 優先級列表 — 4 個判死，剩 2-3 個活著

&#x20; ✅ 任務 5：平行跑測試 + 4GB baseline — 平行跑不行，用 8GB

重要決策：

&#x20; ❌ 平行跑兩個 server 不行（舊閘門）

&#x20;    → 引擎閘門擋下，代碼裡寫死不允許

&#x20;    → 一個 server 就要 17GB，兩個 36GB

&#x20;    → 16GB 機器跑不了兩個

&#x20; ✅ 用 8GB cache，單個 server（現況）

現在的問題：

&#x20; ❌ 掃了半天 knobs，大部分都沒效果

&#x20; ❌ 最大的瓶頸還是黑盒（round remainder）

&#x20; ❌ 測量那麼久，還有黑盒，很可笑

新方向（top-down，不是 bottom-up）：

&#x20; → 先定義理想形狀

&#x20; → 再回頭看需要改什麼

&#x20; → 不是從 knobs 往前碰
```



***

## 分工說明



```
兩個 Agent 都做實驗，不是一個設計一個實驗：

&#x20; → 兩個人都跑實驗

&#x20; → 但時間錯開，不要同時跑

&#x20; → 一個跑上午，一個跑下午

&#x20; → 這樣不會互相干擾

另外：

&#x20; → 4GB parallel 是可行的

&#x20; → 但需要修改閘門

&#x20; → 要做一個 4GB parallel bypass mode

&#x20; → 這樣兩個實驗可以同時跑
```



***

## 優先級排序

### 已完成任務



***

#### ✅ 任務 1：MoE NSG sweep

**結果：** 沒效果，排除這個方向



* gate 通道：6 個值總跨度只有 0.6 個百分點

* down 通道：nsg≥16 就開始變差

* 結論：NSG 這個旋鈕沒有正面效果，不用繼續掃了

詳見：docs/MOE\_NSG\_ADMITTED\_WINDOW\_2026-09-23.md



***

#### ✅ 任務 2：分析 k=3 為什麼會飄

**結果：** expert cache IO 抖動 + kernel 換手



* 不是 thermal 的問題

* 不是 pool event 數量的問題

* 是每個事件的時間本身就在飄

詳見：docs/K3\_SWING\_ANALYSIS\_2026-09-23.md



***

#### ✅ 任務 3：Expert cache 大小同場配對測試

**結果：** 4GB 慢 21%，但可以平行跑兩個



* 8GB：10.74 t/s

* 4GB：8.43 t/s（-21.4%）

* 但 0.79 × 2 = 1.58 > 1，平行跑兩個總效率更高

**重要決策：**

→ 可以用 4GB cache 平行跑兩個實驗

→ 雖然單個慢 21%，但總效率提升 58%

→ 划算！

詳見：docs/CACHE\_SIZE\_AB\_2026-09-23.md



***

#### ✅ 任務 4：列出所有還沒掃的 knobs，排優先級

**結果：** 15 個 knobs 全部核對完



* 4 個已判死

* 2 個重複

* 1 個不存在

* 真正新的只有 2-3 個

詳見：docs/KNOB\_FULL\_MAP\_15\_2026-09-22.md



***

### P0：先做這個（最重要）



***

#### ✅ 任務 0：4GB parallel bypass mode 做完了

**結果：** 做完了



* 找到閘門位置：run\_server.sh 的 other\_llama\_servers 檢查

* 做了 bypass mode：CGC\_PARALLEL\_BYPASS=1

* 只有 4GB cache 才生效

* 8GB cache 自動關閉 bypass

**⚠️ 兩個 Agent 注意：你們可以同時跑實驗了！**

**怎麼用（兩邊都要設）：**



```
export CGC\_PARALLEL\_BYPASS=1

export CGC\_SERVER\_EXPERT\_CACHE\_BYTES=4294967296

./scripts/run\_server.sh
```

**重要：**



* 兩個 Agent 都要設 CGC\_PARALLEL\_BYPASS=1

* 兩個都要用 4GB cache（不能用 8GB）

* 這樣就可以同時跑，不會被閘門擋下

* 雖然單個慢 21%，但兩個同時跑，總效率提升 58%

* 本來要跑兩天的實驗，一天就跑完



***

#### 任務 1：【兩個 Agent 分工進行中】

**WorkBuddy（freebuff）在做：**

> ✅ **Prompt cache 已完成並 commit（2026-09-23 核實，此標記過時）**：
> - `ea8703a2c`（11:48）：prefix 重用與投機回滾分離（M1 9/9、dump md5 72d82a33… 逐位元相同、記憶體減半）
> - `292baacee`（13:28）：prod25 profile 預設開 prompt cache（外部 =0 可覆寫）
> - 均未 push。剩餘：restore 路徑多請求 A/B（§6g）、llama-bench env 關閉確認




1. 盡快讓 prompt cache /prefix cache 可用

   → 最快能見效

   → 不用改 kernel，只是打開功能

   → 重複 prompt 的時候不用重算

2. MTP verify 還是逐個 token verify，沒有 batch verify

   → 現在第 2 個 verify token 要付第 1 個的 98% 成本

   → 如果改成 batch verify，第 2 個 token 只花 20-30% 成本

   → 這樣 MTP 才真的划算

**另一個 Agent（workbuffy）在做：**

3. ✅ Round remainder 已經查清楚了！

   → 結論：根本沒有什麼神秘的 round remainder
   → 它是減法算錯了：一個 round 有 1.5-2.3 個 step，不是 1 個
   → 之前減掉的是「一個 step 的時間」，所以剩下的「remainder」其實是「(n_step - 1) 個 step」
   → 真正的瓶頸：GPU 空窗 44.4ms，完全是 CPU 造成的（CPU 在 hook + submit 上花的時間）
   → 解法：預指派 slot（預測 expert IDs，提前做 hook）
   → 預期收益：step 161ms → ~127ms（-21%），速度 +38%
   
   詳見：docs/ROUND_REMAINDER_IDENTIFIED_2026-09-23.md
   詳見：docs/STEP_SERIALIZATION_2026-09-23.md
   
   **下一步：實作預指派 slot（speculative slot binding）**


**為什麼重要：**



* 掃了半天 knobs，大部分都沒效果

* 這三件事是最快能見效的

* 不用重寫 kernel，只是打開功能或小改架構

**已知：**



* 邊際 verify token 的 70-102%（+29.35 ms/token）

* 拿掉它 step 352→245 ms（-31%）

* 它隨 verify token 數成長

* 現有的儀器（CGC\_VERIFY\_OP\_TIMING）本身會擾動那一輪

**要做什麼：**



1. 建一個不擾動的 round ledger

   → 在既有 CGC-MTP-PERF 的 round 前後掛 pair

   → 不新增 per-op timing（那就不會擾動）

2. 量出來：round remainder 到底是什麼？

   → 是哪個 kernel？

   → 是哪個操作？

   → 占多少時間？

3. 把結果寫成報告

**兩個 Agent 怎麼分工：**

→ Agent A：上午跑，用 8GB cache

→ Agent B：下午跑，用 8GB cache

→ 錯開時間，不要同時跑

→ 或者：兩個都用 4GB，parallel bypass mode 同時跑

**產出：**



* docs/ROUND\_REMAINDER\_IDENTIFIED\_2026-09-23.md

* 明確的結論：round remainder 是什麼，占多少時間



***

#### 任務 2：【兩個都做】研究 MoE gather kernel 重寫方案

**為什麼重要：**



* MoE gather 只有 40% 峰值頻寬

* 是最大的槓桿之一

* 不是讀得散，不是 prefetch，不是 dispatch

* 是 kernel 本身的問題

**已知：**



* 現況：40% 峰值，kernel 本身的問題

* 最好的 kernel（router）：99.7% 峰值

* 差的 kernel（MoE gather）：40% 峰值

* 不是讀得散（shared vs cycled 只差 2-3.7%）

* 要改的是 kernel 結構，不是調參數

**要做什麼：**



1. 研究現有的 MoE gather kernel

   → 它的 tile /simdgroup 配置是什麼？

   → 為什麼效率這麼低？

2. 研究跑得好的 kernel（比如 router）

   → 它的設計思路是什麼？

   → 為什麼它能跑到 99.7%？

3. 設計 MoE gather kernel 的重寫方案

   → 把 router 的設計思路套到 MoE gather 上

   → 參考 dense 裡跑得好的形狀

   → 不是調 NSG，是改 kernel 結構

4. 寫出重寫方案

**兩個 Agent 怎麼分工：**

→ Agent A：上午研究，下午跑實驗

→ Agent B：下午研究，晚上跑實驗

→ 錯開時間，不要同時跑 GPU

**產出：**



* docs/MOE\_GATHER\_REWRITE\_PLAN\_2026-09-23.md

* 完整的 kernel 重寫方案



***

#### 任務 3：【兩個都做】研究 batch verify 架構方案

**為什麼重要：**



* 現在 k=1 最好，k=2/3 反而慢

* 因為沒有批次攤薄

* 第 2 個 verify token 要付第 1 個的 98% 成本

* MTP 根本不划算

**已知：**



* 現況：逐個 verify，第 2 個 token 花 98% 成本

* 理想：batch verify，第 2 個 token 只花 20-30% 成本

* MoE gather 的成本可以批次攤薄

* 這樣 MTP 才真的划算

**要做什麼：**



1. 研究現有的 verify 架構

   → 它是怎麼逐個 verify 的？

   → 為什麼沒有批次攤薄？

2. 設計 batch verify 架構

   → 一次把多個 verify token 的 expert 一起 gather

   → MoE gather 的成本可以批次攤薄

   → 第 2 個 token 的邊際成本從 98% → 20-30%

3. 寫出 batch verify 架構方案

**兩個 Agent 怎麼分工：**

→ Agent A：上午研究，下午跑實驗

→ Agent B：下午研究，晚上跑實驗

→ 錯開時間，不要同時跑 GPU

**產出：**



* docs/BATCH\_VERIFY\_ARCHITECTURE\_2026-09-23.md

* 完整的 batch verify 架構方案



***

#### 任務 4：Dense GEMV — 不用做

**為什麼不用做：**



* Dense 已經跑到 76-91% 峰值

* 就算全部打滿 100%，也只省 2.42%

* 低於 3% 門檻

* 浪費時間

**結論：**



* Dense GEMV 家族作為優化目標結案

* 不用做，不用管，浪費時間



***

## 絕對不要做的事



```
❌ 不要用 --allow-busy 硬跑

&#x20;  → 五次實驗都證明了：2 分鐘一定白花

❌ 不要在熱機上跑（thermal level ≠ 0）

&#x20;  → HEAVY 比 NOMINAL 慢 18%

&#x20;  → 量到的數字不可信

❌ 不要跨 session 比較

&#x20;  → 漂移太大，同 config 都能差 2.1 倍

&#x20;  → 所有 A/B 都要在同一個 session 內

❌ 不要動 NEED\_MB=8000

&#x20;  → 量過了，降 footprint 換不到可讀性

&#x20;  → 這是共用常數，不要偷偷放寬

✅ 用 8GB cache，單個 server（現況）

&#x20;  → 4GB 慢 23%，沒有平行跑補償

&#x20;  → 還是用 8GB 比較快

&#x20;  → 所有實驗都跟 8GB baseline 比

✅ 但要做 4GB parallel bypass mode

&#x20;  → 做完之後，兩個實驗可以同時跑

&#x20;  → 效率翻倍
```



***

## 工具使用提醒



```
✅ 每次跑實驗前，先檢查窗口：

&#x20;  python3 scripts/check/shape\_probe/box\_probe\_compare.py

✅ 跑完實驗後，把結果存到 Backup/

&#x20;  → 不要存在 /tmp（重開就沒了）

✅ 寫報告時，誠實標註：

&#x20;  → 什麼時候跑的

&#x20;  → 當時的窗口狀態

&#x20;  → 如果是 NO READING，就寫 NO READING

&#x20;  → 不要硬湊出一個結論
```



***

## 完成後回報

做完任一項任務後：



1. 把結果寫成 doc，放在 docs/

2. 檔名格式：任務名稱\_日期.md

3. 告訴我（在對話裡）：做完了什麼，結果如何