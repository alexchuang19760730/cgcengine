# 速度提升的驗收鏈：什麼叫「驗收」、要過哪幾關、最快什麼時候

日期：2026-09-26 01:3x　作者：本線（S1／段邊界）
狀態：**判準在跑之前寫死。本文不改任何已量到的數字，只規定「什麼時候可以按哪個鍵」。**
關聯：`docs/S1_LINE_VERDICT_2026-09-25.md`（×1.83）、`docs/STEP23_BLOCKER_AND_RECOVERY_2026-09-26.md`（§8 第 2 步重建）

---

## 0. 一句話答案

**在「穩態 miss 率」被量出來之前，沒有可靠的驗收日期可以給。** 那個數字決定三條路線裡走哪一條
（直接做 kernel ／ 先做 async fill ／ 這條線收益被吃掉），而它**今晚就能量**（但**不是** G1 同一支臂 —— 見 §4）。第 2 步已在 `553424ec1` 進版控，`strings` 證明兩側（`MISSMASK il=` 與
`BATCHDBG layer=`）都在 `libllama.dylib` 裡 ⇒ 儀器不是瓶頸，窗口才是。

---

## 1. 先定義：「驗收速度提升」到底是驗什麼

不是「跑一次看 t/s 有沒有變快」。**速度數字早就量到了**：`S1_LINE_VERDICT` §1 的同 build 同 cell
配對是 A 分段 **11.30** → B 單段 **20.73**（**×1.83**，兩臂皆無 spec）。

問題在於那 20.73 是**用錯的權重算出來的**（§5 會給證據鏈）。所以：

> **驗收判據 = 「20.73（或它退化後的值）＋ `answer_md5` 與分段基準逐 rep 相同」。**
> 只有速度沒有正確性，那個數字不是「變快」，是「算別的東西」。

這也解釋了為什麼不能再跑一次「看看多快」當驗收：跑得越快，只代表錯得越快。

---

## 2. 驗收鏈五道閘門（判準凍結於此，跑完不得改）

| 閘 | 名稱 | 前置 | 要跑什麼 | 凍結判準（PASS 才算過） | 成本 |
|---|---|---|---|---|---|
| **G1** | 第 2 步正確性複驗 | `553424ec1` 已入檔（✅） | 一臂帶 4 個 env（§4） | `SET_DIFF=0` **且** `LEN_DIFF=0` **且** 可比層 ≥30 **且** 對齊元素 ≥200 **且** 每元素 miss 數平均絕對差 = 0.0（`miss_mask_check.py` 寫死值） | ~5 min |
| **G2** | **單段下**的 miss 率 | G1 過 | ⚠ **另一支臂**（§4 的理由） | 見 §3 的三分支判決 | ~5 min |
| **G1b** | mask 的成本 | G1 過 | 兩臂：有／無 `CGC_MISS_MASK` | 差值 **≤ 0.2 ms/step**（`COST_BUDGET_MS`） | ~10 min |
| **G3** | zero slot（第 3 步第一個 commit） | G1 過 | 改 `CGC_B_SCHEME` 的 `e % ns` → zero slot；建置；一臂 | 逐 rep greedy 與分段基準相同；`answer_md5` 相同 | 改碼 ~0.5 h ＋ 一臂 ~5 min |
| **G4** | per-expert 重算 kernel | **G2 落在低區間才做**（§3） | 新 kernel ＋ 建置 | 同 G3 的正確性判據 | 未估（設計未完成、owner 未定） |
| **G5** | 交付 cell 配對 | G3（＋G4 若需要） | 交付 cell ABBA，基準 **12.57 t/s** | 落點 **13.6~14.2**；**<13.0 退回**；配對門檻 3% | ~20–30 min |

**G5 才是「驗收速度提升」那一鍵。** G1–G4 全是它的前置。

---

## 3. G2 是分歧點：三條路線，判準先寫死

`STEP23` §4 的定價：低 miss 率補算 **3.6~4.5 ms**，高 miss 率（當年診斷臂量到 k=8 裡 5 個是佔位
≈62%）補算 **24.6~33 ms** —— 後者直接吃掉全部收益。

| G2 量到的穩態 miss 率 | 判決 | 下一關 |
|---|---|---|
| **< 10%** | kernel 划算 | G3 → **G4** → G5 |
| **10%~20%** | 邊界，補算吃一半 | G3 → G4，但 G5 門檻要重訂（不得沿用 13.6~14.2） |
| **> 20%** | **補算不划算，路線要換** | 先做 async／後台 fill。⚠ prebind／預測路線已判死（h=0.03），「先猜再補」這條路沒有已成立的機制 ⇒ 這一格是**本線目前最大的未決風險**，不是「多花點時間」 |

---

## 4. G1 與 G2 是**兩支臂**（本輪更正：先前寫成同一支是錯的）

mask 建在 `if (cgc_slot_table_gpu && il >= cgc_s1_min_il)`（`llama-graph.cpp:2261`）—— **不是**
`CGC_SEG_BATCH` 的分支。所以 `CGC_SLOT_TABLE_GPU=1` 單獨就建得出 mask；但這也意味著
「單段」與「有 hook」是兩個獨立開關，而 `BATCHDBG` 是 **`ensure_batch` 印的**
（`llama-expert-cache.cpp:1433`）—— 單段下 hook 不跑 ⇒ `BATCHDBG` 一行都沒有 ⇒ `SET_DIFF`
無從比起。⇒：

| 臂 | env | 產什麼 | 為什麼必須這樣 |
|---|---|---|---|
| **G1（正確性）** | `CGC_SLOT_TABLE_GPU=1 CGC_MISS_MASK=1 CGC_MISS_MASK_DBG=1 LLAMA_EXPERT_CACHE_BATCH_DBG=1`（**不開** `CGC_SEG_BATCH`） | `MISSMASK` ＋ `BATCHDBG` ⇒ `SET_DIFF`／`LEN_DIFF` | 兩側都要有：hook 必須跑（它是 `BATCHDBG` 的 producer），mask 也要建。分段提交，速度無意義也沒關係 |
| **G2（miss 率）** | `CGC_SEG_BATCH=1 CGC_B_SCHEME=1 CGC_SLOT_TABLE_GPU=1 CGC_MISS_MASK=1 CGC_MISS_MASK_DBG=1` | 只有 `MISSMASK` ⇒ `CGC-MISSMASK-STEP` 的 `misses/layers` | 要的是**單段下**的 miss 率，而那正是 hook 不跑的狀態。**mask 存在的全部意義就在這裡**：它是裝置端答案，不需要 host |

### 4.1 前置：`run_server.sh` 的 env allowlist（本輪已補）

`scripts/run_server.sh:1495` 明文警告：launch line 走 `env "${SERVER_ENV[@]}"`，是 **allowlist**，
沒列的 `CGC_*` **靜默丟棄**，與「儀器沒作用」無法區分。本輪查：`LLAMA_EXPERT_CACHE_BATCH_DBG`
**已**在 allowlist（:1666），但 **`CGC_MISS_MASK`／`CGC_MISS_MASK_DBG` 都不在** ⇒ 就算有窗口，
直接設這兩個 env 會是一個靜默 no-op（不是本輪獨有的陷阱：:1495 的註解說 NOHOOK/NOGATHER
曾因此從 `run_server.sh` 完全不可達）。

本輪已加（:1675／:1689 兩塊，`bash -n` 通過）。加嚴不加寬：`-n` 判斷，未設時完全惰性。

### 4.2 兩個還沒解的風險

1. **`MIN_PAIRS=200` 可能湊不滿。** `Backup/run_req2_retest.sh:125` 的請求是 `max_tokens=24`
   ⇒ 24 個 decode 步 × 38 層，而且**只有「該步該層有 miss」才印一行**。若 miss 稀疏，對齊元素
   可能 < 200 ⇒ G1 判 FAIL，但那是**樣本不足**，不是 mask 錯。
   **凍結在這裡：對齊元素 < 200 時，先加長 `max_tokens` 重跑，不判 mask 錯。**
   （G1 本來就不產速度，所以改請求形狀不違反任何東西。）
2. 這兩支臂的 **t/s 都不可引用**（`_DBG` 每步多一次 `synchronize`）。

### 4.3 窗口

01:29 查為空；**01:36 重查已被佔**（`llama-bench` PID 80831，`-p 512 -n 128 --spec-type draft-mtp`，
不是本線起的）⇒ **今晚的臂要等那個窗口**，或明天。

## 5. 一個會改變讀法的發現（本輪新增，需 G2 裁定）

三條獨立事實串起來：

1. 單段提交下 hook 完全不跑（`CGC-HOOK=0`、`file_reads=0`）⇒ **權重永遠不進池**。
2. 載入時**沒有預填池**（`grep preload|warm_pool|prefetch_all|CGC_PRELOAD src/llama.cpp/src/` **0 命中**）。
3. 缺席 expert 現在被指到 `e % ns` —— **別人的權重**，不是 0（`llama-context.cpp:3607-3634`）。

⇒ **20.73 t/s 是「用別人的權重算，而且不用等 SSD」的速度。** 它快，正是因為它不必做任何
I/O、也不必等任何 fill —— 而那 88.7% 的 `wait` 本來就是 fill 的等待。

由此推出一個**會讓 G2 很難看**的推論：在單段提交下沒有 fill，池子只會變冷不會變熱 ⇒
**miss 率不是圍繞某個穩態抖動，而是隨步數單調上升**，直到工作集規模。若 G2 證實如此，
則第 3 步的 kernel **單獨不夠** —— 重算一個從沒被填過的池，等於把 MoE 算兩遍。

⚠ 這是**推論**，不是量測。它必須由 G2 裁定，而且**它若成立，§3 的高區間分支就是預設路線**，
「明天就能驗收」不成立。正因如此，§0 的答案是「先量 G2 才有日期」。

---

## 6. 最快時間（把未知數標出來，不用樂觀值填）

| 項 | 估計 | 依據 |
|---|---|---|
| G1＋G2 | **~10–15 min**（兩側儀器已入檔、allowlist 已補；**窗口 01:36 被佔，要等**） | 兩臂各 ~5 min |
| G1b | ~10 min | 兩臂 |
| G3 | ~0.5 h 改碼＋建置＋一臂 | 與 `CGC_B_SCHEME` 同區塊，已有 zero slot 設施（`llama-expert-cache.cpp:268-270` 的 gate 要放寬） |
| G4 | **未知** | 設計未完成、**owner 未明文**（本線＝S1／段邊界，freebuff＝GPU 實驗＋量測工具，`src/llama.cpp/src/` 的 kernel 無歸屬） |
| G5 | ~20–30 min | 交付 cell ABBA |

**若 G2 < 10% 且 G4 順利** ⇒ 最快「G3 當晚＋G4 一個工作段＋G5 一窗口」。
**若 G2 > 20%** ⇒ 日期取決於 async fill 這個**還沒有已成立機制**的工程，本文件不給日期。

---

## 7. 現在就該決定的兩件事

1. **要不要跑 G1/G2 這兩支臂**（01:29 閘門為空、**01:36 已被 `llama-bench` PID 80831 佔用**
   ⇒ 現在不能跑，要等窗口）。判準已凍結於 §2，allowlist 已於本輪補上 ⇒ 剩下的只有窗口。
2. **本輪對 `scripts/run_server.sh` 的 allowlist 修改要不要 commit**（它不在 `src/` ⇒ D5 數值閘門
   不是必需；但它是共用檔，另一條線也在用）。
3. **G4（kernel）的 owner**。它在 `src/llama.cpp/src/`，撞 09-20 那五個碰撞面
   （`llama-context.cpp`、`ggml-backend.cpp`、`libggml-base`）。G2 出來之前不定也行，
   但 G3 落地之後它立刻變成關鍵路徑。
