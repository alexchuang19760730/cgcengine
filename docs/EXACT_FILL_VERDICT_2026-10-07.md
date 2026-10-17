# 判詞：「MTP 下每步多一趟 exact fill」——一半成立，一半未證

日期 2026-10-07 15:2x–15:40
產物 `Backup/spec_tree/pool10g_20261007_145633`（九臂，`--hot`，`warm_skip=64`，tg64 視窗）
對照 `Backup/spec_tree/pool10g_20261007_135534`（同九臂，`warm_skip=0`，tg128 視窗）
工具 `scripts/check/hot_compare.py`（本輪新增 `POOL-COLD VERDICT` 欄）

## 0. 命題原文

2026-10-07 15:2x 我在回報裡寫：

> 這解釋了 MTP 臂慢且抖——**不是 accept 率的問題**（fwd/tok 與 t/s 反而**反向**：a8 rep1 是
> 18.45 t/s（轉引 15:2x 回報的反例 rep、非量測主張），但那個 rep 的 fwd/tok 只有 1.092，最省），而是 **fast path 被打斷，每步多一趟
> exact fill**。

前半（不是 accept 率）**成立且可證**。後半（每步多一趟 exact fill 造成慢與抖）**是推論，本輪
查證後只能判為未證**，本文件把兩者分開落帳。

## 1. 已證：fast path 在 MTP 下**沒有被走**（這部分原文成立）

九臂 stderr 的三支池冷儀器（`llama-context.cpp:8780` 附近的 il≤1 分支，無條件列印）：

| 臂類 | `CGC-SYNCFILL` | `CGC-COLD-GUARD` | `CGC-VERIFY-STRICT` | il=0 cold% |
|---|---|---|---|---|
| MTP-off（3 臂：a1/a4/a9） | **0 / 0 / 0** | **0 / 0 / 0** | **0 / 0 / 0** | n/a |
| MTP-on（6 臂） | 337–444 | 54–150 | 10–13 | 38–45% |

零／非零的切分是**沒有門檻參與**的乾淨對比，所以這條可以引用：MTP 臂需要回填槽位
（`CGC-SYNCFILL cold_filled` = 每次 1.87–3.35 槽），off 臂一次都不需要。`CGC-COLD-GUARD`
和 `CGC-VERIFY-STRICT ... ZERO-slot refused` 一起說明後果：fast path 的 ZERO-slot 被拒，
走 exact path。⇒ **原文「fast path 被打斷」成立。**

### 1a. 這也駁斥了 accept 率作為解釋
逐 rep 對齊（a8_t04_n2 為例）：rep1 `t/s=18.45 / fwd/tok=1.092`、rep2 `10.88 / 1.600`、
rep4 `8.55 / 1.545`。**最慢的 rep 不是 accept 最差的 rep**（rep1 的 accept 最好卻也最快，
方向隨 rep 變），而 accept 中位數只有 1.545。⇒ accept 率不解釋 82% 的 rep 內 spread。

## 2. 未證：回填是不是 41 層 `cb` 的成分

原文說「多一趟 exact fill」暗示時間花在回填上。查證結果：**方向對，但這份資料無法定量，甚至
無法排除組間混淆。**

### 2a. `cb` 確實 MTP 臂高 2.5–3.1 倍
`CGC-SEG`（`ggml-backend.cpp:2819`，每 160 層印一次 running mean，故取尾端中位）：

| 類 | wait/layer | cb/layer | submit/layer | ×41 層 |
|---|---|---|---|---|
| MTP-off | 2151–2249 µs | **481–512 µs** | 86–109 µs | cb ≈ 20 ms/step |
| MTP-on | 3107–3473 µs | **1172–1532 µs** | 90–93 µs | cb ≈ 48–63 ms/step |

### 2b. 但 cold↔cb 的相關是**組間效應**
pooled `corr(cold_filled/次, cb/layer)` over 6 MTP arms = **+0.816**，看似漂亮。分層後：

| n_max | cold_filled/次 | cb/layer | 組內方向 |
|---|---|---|---|
| 1 | 1.87 | 1240.4 | ↑ |
| 1 | 2.13 | 1172.4 | ↓ **反向** |
| 2 | 2.82 | 1453.0 | ↑ |
| 2 | 2.71 | 1446.5 | ↓ **反向** |
| 3 | 3.35 | 1412.1 | ↑ |
| 3 | 3.05 | 1531.8 | ↓ **反向** |

組間 spread 19.3%、組內 spread 0.4–8.1%。**pooled r 完全由「n_max 高的臂既 cold 又慢」這個
組間差撐起來**；每個 n_max 組內只有 2 點，方向一致為負。⇒ 九臂設計（每 n_max 兩臂）在統計上
**無法分離「cold 驅動 cb」與「n_max 同時驅動兩者」**。這是實驗設計的缺陷，不是資料的缺陷。

### 2c. 而且 cold 儀器只描述兩層，不是 41 層
`CGC-COLD-GUARD` / `CGC-SYNCFILL` 只在 `il<=1` 的 fast-path 分支列印（實測：a8 的 61 行中
il=0 佔 38、il=1 佔 23）。`cb/layer` 是 41 層平均。⇒ 把 41 層的 cb 歸因到 2 層的冷事件，
在本資料上**不成立**，即使相關性看起來很高。

### 2d. 定價需要的儀器沒有開
`CGC_FASTPATH_SPLIT`（`llama-context.cpp:8812`）是唯一誠實的窗口：它就地計時
`zero/touch/wait/pub/remap/dbuf` 六個 bucket 且列恆等式。**本輪兩趟都沒設這個 env**
（env-gated，預設關）。⚠ 該儀器自己的註解記著它在 2026-10-03 **推翻過一個同型前提**
（「fast path 不是那 4 s/step 的去處」，真正的載體是 prefill slab 分支）。
⇒ ★ **不要因為這裡的 r=+0.816 就相信回填是主因**：這棵樹已經在同一個問題上錯過一次。

### 2e. 順帶排除：不是 prefill slab
`CGC-PHASE-SPLIT` 九臂全部讀 `cap=8 routable=142-143 slots -> decode graph width=8 tokens
(bound=8, non-binding)` ⇒ verify 批（≤3 token）走 **decode graph**，不是 10-03 那個
`LAYER_CAPS=40-40:16`（bound=1 ⇒ PREFILL）的 slab 懸崖。**那個 4 秒/step 的機制在這一輪
不成立**，不可與 10-03 的數字互比。

## 3. 判詞（15:2x 版，已被 15:4x 取代）

| 命題 | 判 |
|---|---|
| 不是 accept 率造成的 | ✅ **可引用**（§1a） |
| MTP 下 fast path 沒被走（需回填） | ✅ **可引用**（§1，零／非零乾淨切分） |
| 每步多一趟 exact fill **就是**慢與抖的原因 | ⚠ **未證**（§2b 組間效應、§2c 層數不對應、§2d 儀器未開） |

**下一個能判它的實驗**（唯一路徑）：在 MTP-on 臂上開 `CGC_FASTPATH_SPLIT=1`，用
`cold_filled` 高的臂 vs 低的臂比對 `dbuf_us`／`total_us`。若 `total_us` 隨回填量上升 ⇒ 命題
成立；若不動 ⇒ 主因在別處（屆時方向是 `n_max` 帶來的 verify 批次形狀，而非池冷）。

⚠ **不要**用 `CGC-EB_TIMER` 的讀數去減速這個窗口：`llama-context.cpp:8815-8825` 記著
`EB_TIMER` 只包 `ensure_batch`，而這條路徑從不呼叫它；2026-10-03 曾做過這個減法並已收回。
跨路徑的儀器讀數不可相減。

---

# 【15:4x 補充】上面的「唯一路徑」是錯的 —— 正確儀器是 `CGC_EB_TIMER`，而且它已經量到了

## 4. 我 15:3x 建議的儀器**結構上量不到這件事**

`CGC_FASTPATH_SPLIT` 的六個 bucket 全部位於 `verify_fast && cgc_fast_eligible` 區塊內，並在
`llama-context.cpp:9071` `return`。**回填發生在那個 return 之後。** ⇒ 在 MTP 臂上它會印出
標籤而每個 bucket 都是 0。

實測確認（`pool10g_20261007_145633` 的 `a8_t04_n2`）：**330 行 `CGC-FASTSEG`，全部為 0**
（`calls=41 zero_us=0 touch_us=0 wait_us=0 pub_us=0 remap_us=0 dbuf_us=0 total_us=0`）。
★ 這不是「儀器沒開」而是「儀器結構上看不到」—— 就算開了也不會有數字。

## 5. 正確的儀器是 `CGC_EB_TIMER`，而且**既有產物裡就有**

`CGC-EBTIMER` 包住**整個 `ensure_batch`**（assignment 迴圈 + 同步 demand fill + bg_cv 等待，
`llama-expert-cache.cpp:1187`），每步印一行，env gate = `CGC_EB_TIMER=1`：

```
CGC-EBTIMER: step_usec=<us> calls=<n_layers> miss=<n> n_sum=<n> seg=<decode|prefill|mixed>
```

`step_usec` 就是**一步的 fill 時間** —— 正是本命題需要而當時沒有的那個數。

★ **不必等新實驗**：`Backup/spec_tree/rs0_bitident/` 已有 off/on 兩臂的 EBTIMER 產物。
新工具 `scripts/check/fill_price.py`（selftest 12/12）直接讀它們：

```
  calls         n   arms     us p50   miss p50      ms/miss      R^2
  40          321    off       8.46        7.0       +0.902    0.519
  41           99 on_rs0      54.83       55.0       +0.566    0.104
  POOLED      420                                     0.969    0.843

VERDICT: FILL_PRICED   (within-stratum bar R^2 >= 0.18)
```

## 6. 修正後的判詞

| 命題 | 判 |
|---|---|
| 不是 accept 率造成的 | ✅ **可引用**（§1a） |
| MTP 下 fast path 沒被走（需回填） | ✅ **可引用**（§1） |
| **回填本身有真實價格** | ✅ **可引用**：**組內**（calls=40 內）**+0.902 ms/miss，R²=0.519**（n=321） |
| 回填**就是** MTP 慢的原因（全部） | ⚠ **部分**：MTP-on 臂 `step_usec` p50 **54.83 ms** vs off **8.46 ms**，差 ~46 ms/step；但 MTP-on 組內 R² 僅 **0.104**（低於噪聲地板 0.178）⇒ **「每個 miss 值多少 ms」在 MTP 臂內不成立**，主因另有其人 |

★ **這修正了機制理解**：off 臂的 `miss` p50 只有 7（偶爾補 1–2 槽），MTP 臂是 55（每步補 ~55 槽）。
⇒ **價格的證據來自 off 臂的組內變異**（它的 miss 從 0 到 134 全距），而不是 MTP 臂的組間差。
★ 這也解釋了為什麼 §2b 的 pooled r=+0.816 不可信 —— 真關係存在，但強度取決於哪一段變異在說話。

⚠ **工具層的教訓**：`MIN_R2` 我先寫 0.05 並註解「derived」，selftest 第一次跑就把它打回
（2000 次零斜率模擬的 worst spurious R² = **0.1271**）。⇒ 門檻改成**在 import 時由模擬算出**
（`MIN_R2 = worst × 1.4 = 0.178`），selftest 釘住「它等於推導值」而不是「它大於自己」。

## 7. 下一步（唯一未判的一格）

**MTP-on 臂內 `step_usec` 的 46 ms 從哪來？** 三個候選，全部可用 `CGC_EB_TIMER` 已有的欄位區分：
`miss` p50=55（已定價，但 R²=0.104）、`n_sum` p50=63 vs off 的 320（**數量差 5 倍**）、
`calls` 41 vs 40。★ **`n_sum` 是新的最佳嫌疑**：MTP 臂每層只請求 63 個 expert 卻要補 55 個
⇒ **命中率極低（≈0）**，這與 §1 的「ZERO-slot refused / ensure_batch」完全一致。

⚠ 仍**不可**做的事：把 `step_usec` 與 `CGC-SEG` 的 `cb` 相減。兩個窗口不同，2026-10-03 已做過並收回。

---

# 【15:6x 補充】★ 這整段探索重新發現了 30 天前已成立的結論 —— 先讀這裡再往下

## 8. `docs/FILL_COST_MEASURED_2026-09-25.md` 已經判過這件事，而且已經是 NOMINAL 配對

我 15:4x 才找到的 `CGC_EB_TIMER`，那份文件在 **2026-09-25** 就已經把它立起來、量過、並判了：

| 09-25 的量 | 值 | 我 15:4x 用 `fill_price.py` 重算 |
|---|---|---|
| 單 miss 成本（`--prompt 0`，prod-new 形狀） | **0.71 ms** | **0.902 ms**（calls=40 組內，R²=0.519） |
| 穩態 fill | **20.8 ms/step**（遞減曲線 35.1→27.9→20.2→17.2） | off 臂 p50 **8.46 ms**（較早的 step，不是穩態尾段） |
| 每步 miss | 30（率 4.0%） | off 臂 p50 **7** |
| **fill 是否在關鍵路徑上** | ✅ **已判：是**（§8，兩臂 launch 與 worst 皆 NOMINAL） | 未判（我當時只做到「回填有價格」） |

★ 兩者同量級、同方向。**09-25 的數字更強**（有 §8 的 NOMINAL 配對、有 swap 對照、
有 `CGC_EB_NOFILL` 差分把 fill 從 20 ms 壓到 0.2 ms 證明儀器自檢通過）。

## 9. ⇒ 我今天做對了什麼、做錯了什麼

**做對的**：
- 找到了正確的儀器（`CGC_EB_TIMER`），並證明 `CGC_FASTPATH_SPLIT` **結構上量不到**（330 行全 0）。
- `fill_price.py` 的**分層閘**（pooled 強但組內平 ⇒ 不許報 `FILL_PRICED`）是 09-25 沒有的一層保護：
  09-25 的 0.71 ms 是 `step_usec/miss` 的**比值**，沒有迴歸、沒有組內檢查。
- 修了 `pool10g_probe.sh` 裡第二份 `n_gen == 128` 硬編碼（與 `hot_compare.py` 同族）。

**做錯的**：
- ★★★ **我沒有先查有沒有既有判詞**。`EXACT_FILL_VERDICT` 的 §2d 寫「定價儀器沒開」，
  而**同一個資料夾**裡的 `FILL_COST_MEASURED_2026-09-25.md` 就在回答這件事，
  連 stderr 路徑（`<workdir>/llama_bench_*.stderr.log`）與 env 轉發的坑都記了。
  ⇒ **這是本 session 最貴的一個教訓：同一主題的判詞必須先按檔名搜過再推論。**
- 我把 `CGC_FASTPATH_SPLIT` 當成「唯一路徑」建議給 operator，而它在 MTP 上永遠是 0。

## 10. `--fillprice` 模式還要不要？

**不要跑，因為它要回答的問題 09-25 已answered。** 但程式留著，理由不是「再量一次」：

1. `rs0_bitident/` 的 off/on EBTIMER 產物**從未被 `fill_price.py` 的分層閘檢查過**——
   09-25 的 0.71 ms 是比值，09-25 的 §8 是 A/B 端到端，**兩者都沒有回答**
   「MTP-on 臂內 `miss`→`step_usec` 的斜率為何只有 R²=0.104」。那一格至今 UNRESOLVED，
   而它是「MTP 慢的主因是否 = 池冷」的最後一格。
2. 若要跑，`n_sum` 是必須一併看的第三軸（MTP-on 的 `n_sum` p50=63 vs off 的 320，差 5 倍）。

⚠ **若要跑，必守 09-25 §8 的兩條**：強制冷卻 **420 s**（§7.6 的坑：間隔 50 s 時後輪繼承前輪熱），
以及 attribution 必須 0/NOMINAL（否則 t/s 不可引用，§7.4）。


