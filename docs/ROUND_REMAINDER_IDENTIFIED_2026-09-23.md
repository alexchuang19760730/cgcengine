# round remainder 不是一項工作，是一次減法錯誤

**Date** 2026-09-23 10:2x · 零 GPU（重算既有 log）· 資料 `/tmp/verify_layers/arms.jsonl`（6 arm，
k1/k3 × 3 rep，2026-09-22 16:52-16:59）· 工具 `scripts/check/round_ledger.py`（selftest **9/9**）

## 0. One line

`docs/VERIFY_MARGINAL_2026-09-22.md` §5 的「round remainder 108-138 ms（邊際 verify token 的
70-102%）」**不是一個存在的工作項**。它是 `round_ms −（一個 step 的層時間）− draft_ms` 的殘差，
而**一個 round 有 1.5-2.3 個 verify step** ⇒ 減掉的比實際花的少 ⇒ 剩下的部分是
「(n_step − 1) 個 step」，它隨 T 成長是免費的（因為 n_step 隨 T 成長）。

單一 verify step 的中位時間**幾乎不隨 T 變**（179.8 vs 178.0、160.1 vs 187.1 ms）。
⇒ 「邊際 verify token 的代價」是**多跑一個 step**，不是某個 post-layer 工作變貴。

## 1. 定義溯源：remainder 是怎麼算出來的

`scripts/check/verify_marginal.py:245`（`round_budget`）：

```python
s3 = sum(r["wait_3"] + r["cb_3"] + r["submit_3"] for r in lt)   # lt = 40 列，每列一層
remainder = run3["round_ms"] - (s3 + draft_ms + begin_ms)
```

`layer_table` 的每一列是「該層在該 arm 所有取樣 step 上的 **mean**」⇒ `s3` 是
**一個 step** 的層時間（mean 版），不是一個 round 的。

而 `round_ms = 1000 × emit_tok_per_round / tps` 是宏觀讀數。兩者之間差著 **step 數**。

## 2. 實證：round = n_step × step + draft

`CGC-DECPROF` 的 step 行有兩類：`segs=41`（target verify forward，40 層 + 1）與
`segs=2`（draft head 的 forward，median total 只有 1.3-1.8 ms）。用 `calls_draft` 當分母：

| arm | round_ms | rounds | segs=41 steps | **steps/round** | step median | step mean | 閉合誤差 |
|---|---|---|---|---|---|---|---|
| rep0 k1 (T=2) | 273.5 | 192 | 298 | **1.55** | 179.8 | 220.4 | **+7.4%** |
| rep0 k3 (T=4) | 352.4 | 129 | 290 | **2.25** | 178.0 | 224.2 | +21.9% |
| rep1 k1 | 255.0 | 192 | 312 | **1.62** | 160.1 | 252.8 | **+7.2%** |
| rep1 k3 | 383.5 | 127 | 289 | **2.28** | 187.1 | 232.3 | +18.8% |
| rep2 k1 | 178.5 | 192 | 288 | **1.50** | 125.4 | 170.9 | +11.8% |
| rep2 k3 | 369.1 | 127 | 292 | **2.30** | 180.5 | 211.1 | +20.3% |

（`閉合誤差 = [steps/round × step_median + draft/round + begin/round] / round_ms − 1`）

⇒ **round 被「step 數 × 單 step 時間 + draft」解釋到 7-22%，沒有留下 100+ ms 的黑盒。**
k3 側的 +19~22% 是已知的未閉合（§5），但它比「108-138 ms 的未知項」小一個量級。

同一份資料也證實 **`ntok=1`**（`CGC-DECPROF` 的 ntok 分佈：k1 只有 `1/2/8`，k3 才有 `4`）
⇒ **verify 確實是逐 token 的**，與 `docs/NEXT_ACTIONS_2026-09-23.md` 任務 1 的猜測一致：
「第 2 個 verify token 要付第 1 個的 98% 成本」是對的，因為它真的重跑一次完整的 41 段 step。

## 3. 那 step 的時間花在哪裡（中位數，不是單個 warmup step）

⚠ 先更正一個會誤導的數字：DECPROF **第一行**（step=1，warmup）是 `wait 37% / cb 51% /
submit 12%`，容易被當成典型。**中位數**完全不同：

| arm | wait（eval-sync，等 GPU） | cb（gather） | submit（dispatch） |
|---|---|---|---|
| rep0 k1 | 139.3 ms **77%** | 30.1 ms 17% | 10.6 ms 6% |
| rep0 k3 | 131.1 ms **75%** | 33.0 ms 19% | 11.6 ms 7% |
| rep1 k1 | 112.6 ms **71%** | 37.0 ms 23% | 9.0 ms 6% |
| rep1 k3 | 132.0 ms **72%** | 39.2 ms 21% | 11.4 ms 6% |
| rep2 k1 | 95.1 ms **76%** | 23.9 ms 19% | 5.8 ms 5% |
| rep2 k3 | 127.2 ms **72%** | 41.3 ms 23% | 8.0 ms 5% |

⇒ **71-77% 是等 GPU**（eval-sync）。這三項在 DECPROF 口徑下加起來就是 step total 的 100%
（`wait+cb+submit` 的百分比欄位恆等於 100）⇒ **step 內部沒有殘差**。

而 gather 那 17-23% 裡，**99% 是 `llama_expert_cache_ensure_batch()`**：

```
CGC-HOOKSPLIT: n=2880  pre=20.8  ensure=2267.6  drain=0.6  tail=2.1 us/call (total 2291.1)
```
`ensure` 是 `llama-context.cpp:6735` 的 `llama_expert_cache_ensure_batch(cache, il, uni...)`，
即「確保這一層 union 的專家權常駐池裡」。

## 4. 為什麼 step 這麼慢：一個資料依賴把 GPU 餓著

`llama-context.cpp:6740-6746` 的註解把機制講明了：

> the measured shape of a decode step at HEAD is `GPU busy ~45 ms + GPU idle ~19-43 ms`, and the
> idle tracks the CPU hook (`gap ~= 1.3 x (cb + submit)`)... **layer il's expert set is produced by
> the argsort at the END of segment il and consumed by the mul_mat_id at the START of segment
> il+1 -- so the fills for il cannot start before il's top-k exists.**

⇒ **每層的專家集合要到該層段末才產生**，所以填池不能提前 ⇒ CPU 的 `ensure_batch` 串在 GPU 前面，
GPU 只好空轉等它。這就是 wait 佔 71-77% 的原因，也是「一個 41 段的 verify step 要 125-187 ms」
的直接機制。

## 5. 未解（寫在這，不要當成已解決）

1. **steps/round 的比例（1.55 → 2.25，+45%）與 T 的比例（2 → 4，+100%）不相等。**
   可能原因：DECPROF 是取樣的（552-588 個 step 行裡只有 288-312 個帶 40 層明細），
   而兩個 arm 的 `rounds` 不同（192 vs 129）⇒ 取樣率不同會直接污染 steps/round。
   **要定死 step 數，需要一個不取樣的 round-level step 計數器**（見 §6）。
2. **k3 的閉合誤差 +19~22%（k1 只有 +7~12%）**。同一個偏差方向出現在三個 rep ⇒ 不是漂移，
   是口徑：k3 的 round 裡可能還有沒被 `segs=41` 覆蓋的 step，或 median 高估了典型 step。
3. `ensure_batch` 每次 1.9-2.7 ms、`n=2880` 次；一個 verify step 只觸發約 **9.7 次 hook**
   （2880 / 298 steps），不是 40 次 ⇒ **只有約 10 層是 routable（MoE）**，其餘是 dense
   （`denseIQ4X`）。這個「哪些層走 hook」的分佈沒有現成儀器，橫向比較時要小心。

## 6. 下一步：一個不取樣的 round ledger

`docs/NEXT_ACTIONS_2026-09-23.md` 任務 1 要求「不擾動的 round ledger」。現在知道該量什麼了：

- 在既有 `CGC-MTP-PERF` 的 round 邊界上掛**計數器**（不是 per-op timing）：
  每 round 的 `segs=41` step 數、`segs=2` step 數、hook 呼叫數。這三個數現在只能靠 DECPROF
  取樣反推，而取樣率本身就隨 arm 變（§5-1）。
- 有了真正的 step 數，閉合式 `round ≈ n_step × step + draft` 才變成可判定的：
  **若閉合到 ±5%，remainder 這個詞可以退休；若仍然差 >20%，那 20% 才是真正該追的黑盒。**
- 在那之前，**不要**再引用「round remainder 70-102%」。

## 7. 誠實邊界

- 全部數字來自 2026-09-22 那 6 個 arm 的既有 log，**零 GPU 重算，沒有跑任何新實驗**。
- step 時間的中位數/平均數差 1.16-1.58×（mean 被少數慢 step 撐大）⇒ 上表用中位數；
  `verify_marginal.py` 的 `s3` 用平均數，這是兩個口徑，不能混著引用。
- `round_ms` 本身是 `1000 × E / tps` 的宏觀推導，與 DECPROF 的微觀取樣是兩個母體；
  閉合誤差的一部分就是這個母體差，無法從現有資料分離。
- 沒有改任何引擎程式碼；`round_ledger.py` 是純讀 log 的解析器。
