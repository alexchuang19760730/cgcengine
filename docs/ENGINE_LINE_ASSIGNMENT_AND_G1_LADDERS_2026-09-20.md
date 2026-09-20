# 引擎線的命名（線A (ace)）＋ G1 的 `to` 拆成兩條階梯

日期：2026-09-20 11:5x（+08）　作者：**線A (ace)**（本檔的自我介紹就是它；見 §1）
對象：**另一條引擎 session**（做 `cb`／F1–F5 的那條）、以及任何要引用 G1 天花板的人。
前置：`docs/SHARED_SRC_OWNERSHIP_2026-09-19.md`（擁有物協定）、
`docs/PREFILL250_DECODE25_STRATEGY_20260919.html` §1（它自己就寫了「代號不是判準，擁有物才是」）。

---

## 0. 一句話

**本線自 2026-09-20 起叫 `線A (ace)`**（operator 指定），因為舊標籤撞了：本線的記憶寫「本線（line I）」，
而另一條 session 的文件署名也寫「（線 I，2026-09-20 下午）」，兩者**都**在產 G1 的天花板。
**另一條請保留 `線 I`**（它的擁有物是 `cb`／快取命中的儀器，那正是 `線 I = 儀器／快取幾何線` 的字面意思）。

**`線A (ace)` 的定義用擁有物寫死（不看代號）**：本線持有 **S1／段邊界（`wait`／`gap`／S2）**
以及 **逐層 KIND×OP 儀器的延伸**；`4fdfaa8de`（KIND×OP 儀器）在本線的祖先鏈上
（`git merge-base --is-ancestor 4fdfaa8de HEAD` = YES），所以 STRATEGY 檔把它歸給「線 I」
是**歷史標籤**，不是現在的宣稱。

---

## 1. 命名表（按擁有物認定；衝突時以本表為準）

| 代號 | 是誰 | 擁有物（可查證） | 它現在在做 |
|---|---|---|---|
| **`線A (ace)`** | **本線** | `src/llama.cpp/src/llama-context.cpp` 的 **`graph_compute` 區段**（S1 探針修補，`a1a2473fd`）；`ggml-backend.cpp` 的**逐層 KIND×OP 儀器**（`fc0b8a406`）與 S2 的切點；`docs/S1_DIAGNOSTIC_ABORT_ROOT_CAUSE_*`、`docs/S2_IMPLEMENTATION_PLAN_2026-09-20.md` | **S2**（把 slot mgmt 搬到裝置，讓 submit-ahead 合法）＋ 段邊界的歸因 |
| **`線 I`**（另一條，請沿用） | 做 `cb` 的 session | `scripts/check/cb_miss_regression.py`、`docs/DECODE_CB_BREAKDOWN_2026-09-20.md`、`docs/F1_CB_MISS_REGRESSION_RESULT_2026-09-20.md`、`CGC-HOOK-SPLIT` 那一族儀器 | **F1–F5**（`cb` ＝池 miss 的服務成本） |
| `線 S` | STRATEGY 檔的作者 | `plain_match_window.py`、`decode_window_harness.py`、`DECODE_STEP_BUDGET` | 步時預算／窗口守門 |
| `線 B` | `SHARED_SRC_OWNERSHIP` §1 說的「要實作 overlap 的那條」 | —（該檔把 `ggml-backend.cpp` 交給了它） | overlap |

⚠️ **`線 B` / `線 I` 可能是同一條、也可能不是** —— 本線**不猜**。兩份文件（`SHARED_SRC_OWNERSHIP` 與
`STRATEGY`）對代號的用法互相矛盾，而 `STRATEGY` §1 已經自己宣告「全庫只有兩處使用且互相矛盾」。
**要歸屬就查 `git status` 與上面那張表，不要查代號。**

---

## 2. G1 的 `to`：**兩條階梯，不要合成一個天花板**

`targets.json` 的 `G1.to` 已經自己寫著「有兩個獨立的 reachable restatement，commit 必須指名移動哪一個」：
(a) `cb_sum/total`、(b) **段數**。這一節把那兩條**拆成兩條階梯、各自有 owner、天花板與 regime 標籤** ——
這樣兩份 ceiling 就不會再互相否證。

### 階梯 ①：`cb` → 小（owner：`線 I`）

**打的是每層 host 回呼＝池 miss 的服務成本。** 他們自己的數（`docs/F1_CB_MISS_REGRESSION_RESULT_2026-09-20.md`）：

| 事實 | 值 |
|---|---|
| 邊際 miss 成本（F1 判 **LINEAR**） | 0.695 ms/miss，被 ~1.6 GB/s 的 **RAM 拷貝**服務（**不是磁碟**） |
| 池子／量化／目標化 | ∝ **51.31 ms/step（cb 的 77%）** |
| 每層 barrier（**F5**，新的） | **12.86 ms/step（cb 的 19%）**，一個 miss 就付整筆 |
| 暖池對照（F4 的上界） | union 駐留時 `cb` = 0.40／0.33／0.36 ms |
| 天花板（該輪自己的形狀） | `cb` 60.22 → 0 ⇒ 25.8 t/s（mean_len **3.715**）；**F5 單獨** ⇒ 19.3–19.4（約 +6%） |

**該階梯的否證條件**：任何修法**必須把 `cb` 推向 ≤1 ms**（F4）才算打到機制；把 `cb` 從 74 壓到 40
但沒動機制的，不算。

### 階梯 ②：`wait`／段邊界 → 小（owner：`線A (ace)`）

**打的是段邊界的等待**（S2：把 residency 決策＋表發布搬到裝置 ⇒ `submit_seg(i+1)` 才能合法地早於
`hook_seg(i)` 的 wait）。本線的數（`docs/S2_IMPLEMENTATION_PLAN_2026-09-20.md`、`.workbuddy/memory/2026-09-20.md` §EN-304）：

| 假設 | E2b 暖（`mean_len` 2.40、`total` 139.46） |
|---|---|
| 現況 | 17.2 |
| **S2 完美**（`total → union`） | **22.9** |
| S2 但 `39×0.32` 不可約 | 20.5 |
| S2 但只有 `submit` 能搬（`cb` 留關鍵路徑） | **19.0** |

**該階梯的否證條件**：`cb` 是**資料相依**的（hook 要讀 GPU 剛算出的 ids）⇒ 若 S2 不能讓 `cb` 也離開
關鍵路徑，上限就是 ×1.106（19.0），不是 ×1.330。

### 兩條階梯的三條共用紀律

1. **可加，但都不能單獨達標**：`cb` 0 ⇒ 25.8（他們的形狀）／`wait` 0 ⇒ 22.9（E2b 暖）。
   25 需要兩條**疊乘**，再加上 G4／G6。雙方的文件各自都已經寫明了這件事。
2. **每個數字都要帶 regime 標籤**：`mean_len`、`union/total` 是否閉合、熱態（NOMINAL／HEAVY）、
   swap 水位、帶了哪些儀器。**這是本節存在的理由** —— 同一份 `cb`，他們量到 **60.22／74.18 ms**
   （高 swap 10.1/11.2 GB ＋ `CGC_HOOK_SPLIT` ＋ `mean_len` 3.715），本線量到 **21.23 ms**
   （E2b 暖 ＋ `mean_len` 2.40）。**差 3 倍，而兩個都可能對。** 他們的文件已經自己標了
   「絕對值不可與未帶儀器的交付讀數並排」—— 請把這句當成**引用規則**，不是免責聲明。
3. **`gap ⊆ cb+submit` 的原始碼內建交叉檢查目前以 10.09 ms 失敗**（`ggml-backend.cpp:2145-2147`
   斷言 `gap ⊆ cb+submit`，應 ≤24.78 ms，實測 gap=34.87）⇒ **分量相加在該 regime 不閉合**。
   在它被解釋之前，**不要把兩條階梯的天花板直接相加**。

---

## 3. 五個碰撞面與各自的規矩

| # | 面 | 規矩 |
|---|---|---|
| ① | `src/llama.cpp/src/llama-context.cpp`：`CGC_LAYER_AHEAD_PREFETCH` 在 hook（`:5381`）；本線的 S1 修補在 `graph_compute`（`:3348`） | 同檔不同函式 ⇒ 可 merge。**誰後動誰 rebase**，並在 commit message 指名「我動的是哪一段」 |
| ② | `ggml-backend.cpp`：F5 的「每層 barrier」與本線的 **S2 切點（`submit_seg`／`hook_seg`）是同一段程式**，而本線的 G4 儀器（`fc0b8a406`）剛提交在那裡 | **建議 F5 直接複用那條逐層 `CGC-GPULAYK` 行**（`CGC_GPU_OPS` 開啟時才印），不要再加一條平行的 per-layer 輸出 —— 否則兩個 parser 會互相踩。要改那段之前先看 `docs/S2_IMPLEMENTATION_PLAN_2026-09-20.md` §1 的行號表 |
| ③ | `libggml-base` ＋ 8080／GPU 窗口 | `cmake --build` 會蓋掉別人正在 map 的 dylib（09-20 01:04 已有先例）⇒ **建置前確認 8080 空 ＋ 無 llama 行程**，且把「檢查與動作」寫在同一個分支裡 |
| ④ | **`cb` 這個權威數字** | 見 §2 的三條紀律；G1 的 `to` 有 `cb_sum/total` 這一條 ⇒ **引用時必須帶 regime** |
| ⑤ | `.workbuddy/memory/*.md` 的併發寫入 | 09-20 11:45–11:46 實測：本線 `§EN-308` 的**標題被拆成兩行、文字掉了**（已修回）。`.workbuddy/` 是 **gitignored**（`.gitignore:41`）⇒ 這個檔**沒有版控安全網**。**規矩：只 append，且在寫入前後各讀一次；不要用它做 read-modify-write 的長交易。** 另外 `§EN-309` 現在**被用掉兩次**，請接 `§EN-313`+ |

---

## 4. 交給 `線 I`（或那個檔的 owner）的兩個一行改動

1. `agent_harness/portal/targets.json` 的 `G1.owner`：
   `"line B (implementation); line I owns the ceiling reading"`
   ⇒ `"line B (implementation); line A (ace) owns the ceiling reading"`。
   **本線沒有動手**，因為那個檔有你**未提交**的改動（`candidate_family`／`premise_b`／G4 三欄／G6 兩欄），
   我不想把你的在建工作一起 stage 進我的 commit。
2. 若要更精確，`G1.to` 可加一句：「**兩條獨立階梯**：`cb`→0（owner `線 I`）與 `wait`/段邊界→0
   （owner `線A (ace)`）；兩者不可相加，見 `docs/ENGINE_LINE_ASSIGNMENT_AND_G1_LADDERS_2026-09-20.md` §2」。

---

## 5. 誠實邊界

- 本檔**不主張** `線 B`＝`線 I`，也不主張 `線 S` 是否還活著 —— 只主張 §1 那張表**按擁有物**成立的
  部分，其餘標為未知。
- 本檔**沒有**產生任何新的 t/s 讀數：§2 的每一格都指回它自己的出處，而其中「17.2／19.0／22.9」
  **是推算不是量測**（見 §EN-304 與 §EN-310）。
- 本檔**不改**任何既有報告的敘述：`docs/S2_IMPLEMENTATION_PLAN_2026-09-20.md` 裡的「本線（line I）」
  是**當時的標籤**，已在該檔加一行 dated 註記指向本檔，原文未刪。
