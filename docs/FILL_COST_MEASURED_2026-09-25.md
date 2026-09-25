> ⚠ **非生产口径（`--prompt 0` 冷池 ＋ `MTP on` ＋ `--batch 512`）的数据已于 2026-09-25 全部作废并删除。**
> 权威结论只看 `docs/IO_AXIS_VERDICT_2026-09-25.md`（生产口径）。

# fill 成本已量到：**穩態 21~24 ms/step**（3b 第一個真數字）

> **性質**：實測（GPU、`prod-new` 標準形狀、A 臂分段正確路徑、reps=3）　**2 次 build**
> **一句話**：`ensure_batch` 全程（含 demand fill）在 decode 穩態佔 **≈21~24 ms/step**，
> 每步 miss ≈ 28~30 個 expert（miss 率 ≈4%），單個 miss ≈ **0.71 ms**。
> swap 對照（`CGC_POOL_MADVISE=1`）**無差異** ⇒ 這個數字沒有被 swap 污染，可引用。

---

## 1. 為什麼要這個數字

3b 的判據寫的是「decode 段 `fill_ms/step` 可測，目標 ≤ 5 ms/步」，但在此之前**沒有任何數字**：
既有的 `fill_wait_us` 兩個累加点都不在 batch 路徑上（skill 第 65 條），
`pread_usec` 跨 worker 累加、可超出 wall time，兩者都不能定價 fill。

## 2. 儀器：`CGC_EB_TIMER`（已落地並 build）

`src/llama.cpp/src/llama-expert-cache.{h,cpp}`：

- `struct cgc_eb_timer`（RAII，dtor 累加）⇒ **涵蓋 `ensure_batch` 所有 return 路徑**，
  包含 assignment loop、`bg_cv` 等待、以及 `fill_segments_pool` 的 demand fill。
- env gate **`CGC_EB_TIMER=1`**；關閉時傳 `nullptr`，dtor 直接 return ⇒ 熱路徑零成本。
- 每累計滿 `slot_owner.size()`（＝層數）次呼叫印一行並清零 ⇒ **decode 時一行 = 一步**：

```
CGC-EBTIMER: step_usec=<us> calls=<n> miss=<n> n_sum=<n>
```

- `n_sum` 用來分離 decode（n≈19/層）與 prefill chunk（n 大）。

### 出口在哪（踩過的坑）

stderr **不在** `--log-dir`，而在 `llama_bench_matrix.py:407-408` 寫的
`<workdir>/llama_bench_<tag>_<shape>.stderr.log`，**`--workdir` 預設 `/tmp`**。
第一次我用 `--log-dir` 拿到 0 位元組的空檔，誤以為儀器沒輸出。

## 3. 數字

命令（`prod-new` 標準形狀，不改任何形狀參數）：

218 行 `CGC-EBTIMER`，分段中位數（每 40 步一段）：

| 區段 | step_usec p50 | ms/step |
|---|---:|---:|
| 前 40 步（冷啟動） | 48662 | 48.7 |
| 40+ | — | 29.1 |
| 80+ | — | 38.8 |
| 120+ | — | 28.9 |
| **160+（穩態）** | **20800** | **20.8** |

### 交叉驗證（所以這個數字可信）

| 量 | 實測 | 既有獨立數字 | 結論 |
|---|---:|---:|---|
| `n_sum`/步 | 752（＝40 層 × 18.8） | hook union ≈ 19 | 吻合 |
| miss/步 | 30 | — | — |
| **miss 率** | **30/752 = 4.0%** | A0 server 實測 **4.3%** | 吻合 |
| 單 miss 成本 | 21.4 ms / 30 = **0.71 ms** | 舊定價 0.126~0.169 ms | **貴 4~5 倍** |

⇒ 支持 `STEP3_PER_EXPERT_RECOMPUTE` §2.1 的結論：**瓶頸是 fill 的固定開銷，不是 bytes**。

## 4. swap 對照 ⇒ 不污染（§6.3 結案）

`CGC_POOL_MADVISE=1` 對照輪（同形狀同 reps）：

| 段 | madv0（基線） | madv1（對照） |
|---|---:|---:|
| 0+ | 48.7 | 56.3 |
| 40+ | 29.1 | 27.7 |
| 80+ | 38.8 | 33.1 |
| 120+ | 28.9 | 25.1 |
| **160+（穩態）** | **20.8** | **24.1** |

兩組在噪音內一致 ⇒ `CGC_POOL_MADVISE` 對 fill 無影響 ⇒ **21~24 ms/step 可引用**。

⚠ **差點誤判**：對照組「最後 40 行」p50 = 88.93 ms，看似基線的 4 倍。真因是
**reps=3 每輪重新載入，最後一段跨到下一個 rep 的冷啟動**（243 行 vs 218 行）。
⇒ 分段看，且**每段不要跨 rep 邊界**。

## 5. 判據對照與量級

- 3b 目標 ≤ 5 ms/step ⇒ 現況 **超標 4.3 倍**。
- 若 decode step ≈ 79 ms（既有 wait 66.3 + cb 6.1 + submit 4.2），fill 21.4 ms 佔 **27%**；
  壓到 5 ms ⇒ 省 ~16 ms ⇒ 79.5 → 63.5 ms ⇒ 12.57 → **~15.7 t/s（+25%）**。
- ⚠ 上面這筆帳**只有在 fill 位於關鍵路徑時才成立**。hook 在 wait 之後串行執行，看起來是，
  但**尚未實測** ⇒ 這是下一個要補的判據。

## 6. 下一步

---

## 7. 附錄：關鍵路徑實驗（2026-09-25 02:0x）—— **UNRESOLVED，thermal 不合格**

### 7.1 方法：no-op 診斷臂差分

不用時間對齊（兩個儀器口徑／regime 不同，對不齊），而是**把 fill 關掉看端到端差多少**。

新增 env gate `CGC_EB_NOFILL=1`，加在 `fill_segments_pool` **函式開頭**（不是呼叫點，
一次覆蓋所有路徑）：slot 照常分配發布，但**不讀任何 bytes**。
`ok.assign(n, 1)` 假裝成功——否則呼叫端的 memset 補零會把想排除的成本加回來。
輸出是 garbage ⇒ **只能看時間，不能看正確性**。

### 7.2 結果（ABBA 四輪，同 build、同交付 cell）

| 輪 | 臂 | tg t/s (sd) | launch thermal | fill ms/step（分段剖面） |
|---|---|---|---|---|
| r1 | fill | 10.44 (1.16) | MODERATE | **[35.1, 27.9, 20.2, 17.2]** |
| r2 | nofill | 10.32 (1.31) | **HEAVY** | **[0.1, 0.3, 0.2, 0.2]** |
| r3 | nofill | 15.17 (5.16) | **HEAVY** | [0.2, 0.3, 0.3, 0.3] |
| r4 | fill | 7.46 (0.18) | **HEAVY** | — |

### 7.3 ★ 儀器自檢**通過**（這條不受 thermal 影響）

**fill 20 ms/step → 0.2 ms/step（100×）** ⇒ 開關有效，且**證明 `CGC-EBTIMER` 量到的就是
fill 本身**，沒有把別的東西算進來。這是本輪最硬的收穫。

另外 fill 剖面**單調遞減** 35.1 → 27.9 → 20.2 → 17.2 ⇒ **池在逐步填滿，fill 成本隨時間
下降（不是常數）**。所以 §2 的「21 ms」應理解成一個遞減曲線上的取值，不是穩態常數。

### 7.4 ✗ t/s **不可引用**

- 四輪 launch thermal：1×MODERATE + 3×**HEAVY**，`worst` 全為 level 2。
  `cgc-prefill-thermal-delivery` 的判據：**只有 0/NOMINAL 可引用**。
- 兩臂**內部散度**就有 40~47%（fill 臂 10.44 vs 7.46；nofill 臂 10.32 vs 15.17）
  ⇒ 大於任何組間差異 ⇒ 無法判讀。
- 根因：四輪發射間隔僅 ~50 s（02:01:15 → 02:03:44），每輪還重載 13 GB
  ⇒ 後一輪必然繼承前一輪的熱。skill §154 記過同一件事，`--cooldown-timeout` 應是 **420 s** 級。

### 7.5 ⇒ 結論

**本輪：fill 是否在關鍵路徑上，仍然未定。** 不是「不在」，是「還不知道」。

⚠ 特別警告：不要因為 r2（10.32）≈ r1（10.44）就判定「fill 不在關鍵路徑」——
那一輪是 HEAVY，且同臂的 r3 是 15.17。要判就要 NOMINAL 的配對。

### 7.6 本輪兩個流程坑

1. **zsh 不做 word splitting**：`SHAPE="--arms prod-new ..."` 展開後是**一個參數**
   ⇒ `unrecognized arguments`。要用陣列 `SHAPE=(...)` + `"${SHAPE[@]}"`。
2. `llama_bench_matrix.py --workdir` **不會自建目錄**，且它在收尾時才去讀
   `<workdir>/llama_bench_*.stderr.log` ⇒ 目錄不存在時會先跑完 80 s 才 `FileNotFoundError`
   ⇒ 每輪先 `mkdir -p`。

---

## 8. ★ 附錄補完：NOMINAL 配對（2026-09-25 02:31 / 02:47）—— **已判決：fill 在關鍵路徑上**

### 8.1 結果（同 build、同交付 cell，兩臂 launch 與 worst 皆 NOMINAL）

兩臂間隔 15 分 20 秒（r5 發射 02:31:52，r6 發射 02:47:13）⇒ 無熱繼承。
**強制冷卻 420 s** 的坑（§7.4）已排除。

### 8.2 交叉支撐：這個方向不是單點

四輪 HEAVY 雖然不可引用絕對值，但**方向一致**：

即便取最保守口徑（fill 用歷史最高的 12.57，nofill 用最低的 15.17）也有 **+21%**，
仍在 15% 門檻之上。**fill 佔 end-to-end step 的 21%~50%，中樞約 35%。**

### 8.3 ⚠ 三個必須帶著的 caveat

### 8.4 ⇒ 對 3b 的意義

**3b 從「不確定值不值得」變成「已知值得」。** fill 吃掉 step 的 41%，
而 §5 已量到單 miss 0.71 ms 對上舊定價 0.126~0.169 ms（**貴 4~5 倍**）
⇒ 固定開銷主導 ⇒ 合併 pread／加大 span 是對的方向。

下一步**不是直接改 C++**：先過 `cgc-io-request-shape` 的三條獨立判準估一次值，
再決定動手。見 §6 第 2 項。

---

## 9. 3b 的地形圖（2026-09-25 03:0x，零 GPU 靜態勘察）

§8 判了「fill 值得打」，但**打哪裡**會被既有程式碼決定。三條結論，全部有出處：

### 9.1 「fill 觸發點在 hook 裡」要校準

`llama_expert_cache_ensure_batch` 的呼叫點在 **`llama-context.cpp:7220`**（per-layer 循環）
與 **`llama-context.cpp:6977`**。**不在 `ggml-backend.cpp`**。
（memory 裡「hook 在 `ggml-backend.cpp:1782`」指的是 **S1 單段提交的 return 點**，是另一件事。）

### 9.2 層內 batch 化**早已存在**（3b-3 的一半是已做完的）

`llama-expert-cache.cpp:1448-1469`：一層的所有 miss 會被 `fill_pool_direct_collect`
flatten 成**一個** job list（`all_segs` / `all_dsts`），再一次 `fill_segments_pool` 交給
8 worker 的持久 pool。`LLAMA_EXPERT_CACHE_BATCH_SPAWN=1` 才是舊的 spawn-per-expert + join
（留著作 A/B）。

⇒ **「把同一層的 pread 合併」這件事 2026-08 就做過了。** 再喊「batch 化」如果指的是這個，
那就沒有新東西。

### 9.3 跨層 batch **被資料依賴鎖死**（3b-2 的正統做法是做不到的）

`llama-context.cpp:7225-7234` 的註釋寫死了：

> layer il's expert set is produced by the argsort at the END of segment il and consumed by
> the mul_mat_id at the START of segment il+1 — so **the fills for il cannot start before
> il's top-k exists**.

第 L 層要用哪些 expert，要等第 L 段的 argsort 在 GPU 上跑完才知道 ⇒
**沒有「在 step 邊界一次拿到 40 層的 ids 再統一發 IO」這種事**，除非先把 routing 全部算完
（等於重跑）。所以 3b-2「搬出 hook」的標準動機不存在。

既有的替代方案是 **layer-ahead prefetch**（同一段註釋，`CGC_LAYER_AHEAD_PREFETCH=1`，
非預設的 A/B flag，且判的是 `getenv(...)[0] == '1'`）：用**上一個 token** 的 routing
（相鄰 token 復用 ~87%）提前一層把 prefetch 排進去。它已知的天花板：

> drain_layer(next) **DROPS** prefetches for the next layer that have not started yet, so
> only fills that actually land inside the window pay off. … it is why this is an A/B flag
> rather than a default.

### 9.4 ⇒ 3b 剩下真正可變的東西

層內已 batch、跨層被鎖死 ⇒ **唯一槓桿是 IO 請求的形狀**：
每層 ~30 miss × 3 segment ≈ 90 個 pread，單 miss 0.71 ms 但舊定價只有 0.126~0.169 ms
⇒ **貴出來的 4~5 倍是固定開銷**，合併相鄰 segment／加大 span 才是變數。

這正是 `cgc-io-request-shape` 的管轄範圍。**先估值，再決定要不要動 C++。**
