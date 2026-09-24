# step 為什麼慢：不是 GPU 被餓著，是 CPU 站在每一段的關鍵路徑上

**Date** 2026-09-23 10:5x **線** 線A (ace，引擎層) **零 GPU**：全部數字由既有 log 重算
**資料** `/tmp/verify_layers/20260922_165215_prod25_decode-spec/*.stderr.log`
（`prod25` decode-spec，298 個 verify step）
**工具** 本報告的數字由一次性腳本重算；段級口徑見 `scripts/check/round_ledger.py`（selftest 9/9）

---

## 0. 先更正我上一輪講錯的一句（這句會導致解法打錯靶）

上一輪我說「**wait 佔 71-77% ⇒ GPU 空轉**」。**錯。** 讀 `ggml-backend.cpp:2109-2122`：

```c
auto hook_seg = [&](int i) -> bool {
    const int64_t st0 = ggml_time_us();
    if (cgc_done) {
        const int target = done0 + (i + 1) * bufs;
        while (cgc_done(split_backend) < target) { sched_yield(); }   // ← wait
    }
    const int64_t st1 = ggml_time_us();
    ... callback_eval(ttopk, ...)                                      // ← cb（top-k hook）
    const int64_t st2 = ggml_time_us();
```

- **`wait` = CPU 阻塞等 GPU 跑完 segment i**（而且是 `sched_yield()` 自旋忙等，佔著一顆核）。
- **GPU 空轉是另一個欄位：`gap_sum`**（Metal 時間戳：上一段 end 到下一段 start 的空窗）。

所以「wait 78%」的正確讀法是「**CPU 有 78% 的時間在空等**」，而 GPU 端是
**忙 71% / 空 28%**。兩個數字都對，但指向完全相反的解法：

> 餓的是 **CPU 沒事做**（它有 123 ms/step 的閒置，而它真正需要的只有 35 ms），
> 不是 GPU 沒事做。

---

## 1. 穩態 verify step 的時間帳（中位數，n=221）

篩選條件：`segs=41`（target model 的 verify step，排除 draft model 的廉價 step）
且 `total > 50 ms` 且 `step >= 100`（跳過冷啟動）。

| 量 | 中位數 | 佔 total | 是什麼 |
|---|---|---|---|
| **total** | **161.0 ms** | 100% | CPU 側串接總和 = wait+cb+submit |
| wait | 123.2 ms | **78.1%** | CPU 自旋等 GPU 跑完這一段 |
| cb | 23.8 ms | 15.0% | top-k hook：slot 分配 ＋ 阻塞式填池 |
| submit | 10.8 ms | 7.0% | 提交下一段的 CPU 時間 |
| gpu_sum | 168.3 ms | 105.8% | Metal busy 之和（含重疊 ⇒ >100%） |
| **union** | **113.4 ms** | **71.3%** | **GPU 真正忙的跨度** |
| **gap** | **44.4 ms** | **27.9%** | **GPU 空窗 ← 這才是被餓掉的部分** |

**閉合**：`union + gap = 113.4 + 44.4 = 157.8` vs `total = 161.0`（差 3.2 ms，2%）
⇒ 兩邊的時鐘對得上，這張表可以互相引用。

---

## 2. gap 的成因：幾乎完全是 CPU 的 hook＋submit

對 221 個 step 做回歸（`gap` vs `cb + submit`）：

```
r      = 0.957
slope  = 1.05
截距   = +10.1 ms      ← 每段提交後的命令緩衝啟動偏斜
```

**斜率 1.05、r=0.957** ⇒ 幾乎 1:1 的因果：**GPU 空轉 = CPU 在 hook 與 submit 上花的時間**
（外加約 10 ms 的固定啟動偏斜）。

拆解那 34.6 ms（cb 23.8 ＋ submit 10.8）：

- `CGC-HOOKSPLIT`：gather 的 **99% 是 `ensure_batch`**（ensure 2267.6 / total 2291.2 µs）。
- 用既有量測代入（F1／F5，見 `docs/DECODE_CB_BREAKDOWN_2026-09-20.md`）：
  - **每層 barrier ≈ 12.9 ms/step**（0.46 ms/層 × 約 28 層有 miss；**只要該層有 ≥1 次 miss 就付**）
  - **剩餘 ≈ 11 ms ≈ 16 次 fill** × 0.695 ms/次（邊際 miss 成本，由 ~1.6 GB/s 的 **RAM 拷貝**服務，不是磁碟）
- ⇒ **cb 裡「固定項（barrier）」比「真正的搬資料」還大**。

---

## 3. 為什麼 CPU 非得站在關鍵路徑上

`llama-context.cpp:6740-6746` 的註解講的是真的：

```
argsort 在 segment il 的段末產生 il 的專家集合
    → mul_mat_id 在 segment il+1 的段首要用
    → 填池不能提前
```

而 `ggml-backend.cpp:2114-2118` 還有一條更硬的約束（註解原文）：

> Waiting only on the main buffer fired the top-k hook while the argsort was still running
> → stale ids → garbage remap → whole-graph corruption.

⇒ 不能只等主緩衝區，必須等**整段**跑完。結果就是 41 段嚴格 ping-pong：

```
submit(il) ── GPU 跑 il ── CPU 空等 123ms/step ── hook(cb) ── submit(il+1) ── ...
                              ↑                        ↑
                        CPU 有空但不知道 ids      GPU 在這裡空轉
```

**關鍵的不對稱**：CPU 在一步裡閒置 123 ms，而它真正需要的只有 35 ms。
**擋住它的唯一東西是「它不知道 ids」。** ⇒ 整個問題可以濃縮成一句：

> **把 ids 提前給 CPU。**

---

## 4. 解法（按對症程度 × 可行性排序）

### ★ 方案 A：預指派 slot（speculative slot binding）—— 最對症，不碰 kernel

**做法**：在 segment il 提交**之前**，用預測的 ids 把 il 的 slot 指好、把 `mul_mat_id` 的指標寫進去、
把整段圖建好。等真 ids 出來只做一次**校驗**：全中 ⇒ 什麼都不做（GPU 已經跑過去了）；
有缺 ⇒ fallback 走現在這條路。

**為什麼現在有本錢做**：CPU 有 123 ms/step 的空窗，而 hook+submit 只需要 35 ms。
預測/預建的成本完全放得下。

**已知風險（別當成沒事）**：
- 87% 是 **per-expert 複用率**。8 個 id 全中的機率若獨立是 `0.87⁸ ≈ 33%` ⇒
  **必須預指派比 top-8 寬的候選集**（例如上一 token 的 top-16），代價是多佔 slot。
- F2 的實測說「預測目標 100% 已常駐」（`resident 320/320`）⇒
  **prefetch 本身沒東西可做**。這表示 A 的價值不在「省 fill」，而在
  **「讓 CPU 不必在關鍵路徑上」**——跟 F2 打的不是同一個東西，別用 F2 的 null 去否決 A。
- 真正會 miss 的那 ~0.4 次/層-step，按定義**不在**上一 token 的 top-8 裡 ⇒
  A 需要一個比「上一 token top-8」更強的預測器，否則命中率上不來。

**預期**：命中時 `gap: 44.4 → ~10`（只剩啟動偏斜）⇒ step `161 → ~127 ms`（**−21%**）。

### 方案 B：slot 間接化 —— 徹底拆掉段邊界，最貴

`mul_mat_id` 的 `src0` 直接吃 pool 視圖 `[ne00, ne01, n_slots]`，ids 傳 **slot index**
而不是 expert index；expert→slot 的映射用 GPU gather（每層 256 個 int32，每步上傳一次）。
ids 來自 GPU 的 argsort ⇒ **全鏈路無 CPU** ⇒ 41 段可收回 1 段，`gap → 0` 且 submit 也省掉。

- 前提：**miss 必須 = 0**（否則 slot 裡沒有資料）。
- 代價：動圖構建 ＋ kernel（`ne12` 從 n_expert 變 n_slots），且 143 格的 batch 維要 Metal 能接受。
- **建議先做 A**：用 A 實測到的命中率判斷 B 值不值得。

### 方案 C：把 miss 打到 0 —— 提高有效容量（收益不確定）

- 現況 **143 格/層 vs 256 專家 = 56% 覆蓋**（`LAYER_CAPS total 5976, avg 145.8, min 143`）
  ⇒ 穩態必然 churn，這就是為什麼「暖機之後還在填」。
- 全常駐需要 256 格/層 = **14.3 GiB**（不可行）。
- 便宜的替代：`LAYER_CAPS` 已在樹上 ⇒ 依**路由直方圖按層重分配**（集中的層少給、分散的層多給），
  總格數不變。先量直方圖再決定，別先動手。

### 方案 D：攤薄 —— 提高每 step 的 token 數

`emit_tok_per_round = 2.000`（`CGC-MTP-PERF`）⇒ 約 24 ms/step 的固定成本只攤在 2 個 token 上。
但 **ntok 變大 ⇒ 每層 union 變大 ⇒ fill 變多**，不是免費的；而且這塊屬 MTP/spec，
既有結論是「別練 draft head」（m 攤薄係數不利）。

### 方案 E：別再做 —— 繼續優化 fill 的 IO

F3 已 CLOSED（1760 MiB/s，已在 RAM 拷貝速度之上）；且 fill 只佔 cb 的約 11 ms，
另一半是 barrier。**打這個靶是浪費。**

---

## 5. 誠實的邊界

- **上界**：gap 全部消掉 ⇒ step `161 → 116.6 ms`（−27.6%）。
  若 tokens/step 不變，t/s 12.57 → **約 17.3（+38%）**。這是**儀器給的上界，不是承諾**，
  且它假設命中率 100%。
- **地板**：GPU busy `union = 113.4 ms` 不會因為重疊而變小 ⇒
  「只消滅空轉」這條路的盡頭就是它；要再快必須減少 GPU 工作量或提高 E。
- **前車之鑑（F2）**：flag 在 env 裡 ≠ 生效。任何新機制**必須先有「真的執行了」的計數器**
  才能判 A/B，否則又是一次 null，而且會被誤讀成「overlap 沒用」。
- **還沒查的**：① `gap` 的 10.1 ms 截距到底是幾段的啟動偏斜（41 段 × 0.25 ms？）
  ⇒ 若它跟段數成正比，**減少段數本身就是一條獨立槓桿**；
  ② 每層 barrier 0.46 ms 的實體是什麼（鎖？cv？）⇒ 它是 cb 裡最大的單項，卻沒被歸因過。
