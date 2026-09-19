# MTP verify 還能優化什麼： residency thrash 是主要成本項

日期：2026-09-18 · 作者：本 session（引擎层）
資料來源：`Backup/phase_decomp/spec_cost_curve_20260918_2030.json` ＋
`Backup/phase_decomp/spec_cost_logs_20260918_2030/spec_cost_k*_r*_p512_n128_d0.stderr.log`
（同一批樣本，即 `docs/SPEC_COST_CURVE_2026-09-18.md` 那一輪；`prefill250`，llama-bench，
`-p 512 -n 128 -d 0`，3 輪 ABBA）
上游推論見 `docs/RSL_MTP_GAIN_ESTIMATE_2026-09-18.md`。

> **本文的定位**：回答「除了 batch／llama-bench-vs-HTTP 那個差異之外，MTP verify 還能優化什麼」。
> 所有數字都從上述 log 直接讀出（不是再跑一次），算式寫出來以便复算。
> 但是：**下面的因果鏈是一個假設，本文沒有任何新的量測去驗證它** —— §6 列出能證偽它的實驗。

---

## 1. 一句话结论

**verify 的成本不是 accept rate，也不是 draft 模型的算力，而是 verify step 把 expert working set
撐破 per-layer slot 配額之後的反覆重讀。**

我們自己的 teardown 計數器把這件事印在每一支 log 的尾巴 —— 只是一直沒有人把它跟 `cost(k)` 對起來看。

---

## 2. 證據：thrash 計數器 vs 成本（同一批樣本，逐 k）

每行取自對應 stderr 的 teardown 段：

```text
llama_expert_cache: miss attribution: ... layers_distinct_over_slots=N  worst=layer L distinct=D slots=S
llama_expert_cache: read shape: jobs=... bytes=...
llama_expert_cache: decode/pool (ensure_slot+batch) hits=.../... (H%)
```

| k | `layers_distinct_over_slots`（各層 working set > 該層 slots 的層數） | worst layer distinct / slots | pool hit% | 每步讀取 MiB | 每步 ms |
|---|---|---|---|---|---|
| 0（無 MTP） | **0** | 143 / **143** | 92.6 | 7.9 | 106.4 |
| 1 | **4** | 185 / 143 | **49.2** | 35.4 | 193.8 |
| 2 | （同一趨勢） | — | 49.9 | 37.6 | 188.2 |
| 3 | **15** | 211 / 143 | 51.6 | 53.3 | 227.2 |
| 5 | — | — | 51.5 | 38.5 | 299.9 |
| 7 | **17** | 219 / 143 | **48.3** | 67.0 | 322.6 |

两位数的关键点：

1. **`k=0` 時最壞層剛好是 `distinct=143 / slots=143` —— 零餘量。**
   沒有 MTP 的時候.working set 就已經頂到配額（所以先前才有那次 143→179 slots 的實驗）。
2. **k 只要從 0 加到 1，working set 立刻越界（4 層），命中率從 92.6% 崩到 49.2%，
   每步讀取量從 7.9 MiB 漲到 35.4 MiB（4.5×）。** 這個跳變遠大於 union 計數的增長
   （8.0 → 12.2 experts/call，1.5×）—— 它不是「多讀了一點」，是**進了 thrash 區間**：
   同一批專家每步被踢掉再讀回來，miss attribution 會把它們記成 compulsory 而非 capacity
   （實測 k=7 時 compulsory 佔 79.5%，capacity 只有 20.5%）。

---

## 3. 線性化之後：每步時間 vs 每步讀取量

```
ms/step = 89.0 + 3.35 × MiB/step          r² = 0.70   (n=6, k=0..7)
         └ 固定 ~89 ms     └ 3.35 ms/MiB ⇒ ~298 MiB/s effective
```

分解後的佔比（k=0 → k=7）：

| k | 讀取佔 step | 固定佔 step |
|---|---|---|
| 0 | 25% | 84%（兩項之和 >100%：佔比各自除以實測值，而回歸有截距誤差） |
| 1 | 61% | 46% |
| 3 | 79% | 39% |
| 7 | 70% | 28% |

**caveat（重要）**：`MiB/step` 與 `k` 在這組樣本裡幾乎完全共線 ⇒ 這條回歸**不能單獨證明因果**。
它只說明「每步多花的時間，在數量級上剛好等於多讀的位元組除以 ~300 MB/s」。
真正提供機制的是 §2 的 thrash 計數器 —— 它是**獨立於** k 的一條證據（層數是整數計數，不是比率）。

---

## 4. 由此推出：can 有多大 Today

假設 residency 修好後，每步讀取量回到 `k=0` 的水準（~8 MiB/step），而每個 step 的時間不再隨 k 增長：

```
step ≈ 116 ms（= 今天的 baseline），對所有 k
S(k) = E(k) / 1 = E(k)
```

用實測的 E：

| k | 實測 E | 若 cost→1 的 S（上界） |
|---|---|---|
| 1 | 1.306 | 1.31× |
| 3 | 1.662 | 1.66× |
| 5 | 1.882 | **1.88×** |

⚠ **這是上界**，因為它假設 draft 那 k 次前向的成本為 0（見 §3 的共線 caveat）。
但它給出了天花板的量級：**1.3–1.9×，而不是 2–3×。**

### 兩條軸要一起動才有用

`S = E(a,k) / (1 + m·k)`，today `m = 0.474`、`a = 0.29–0.48`：

| | 今天 `a`（E=1.88 @ k=5） | `a = 0.85`（E=4.15 @ k=5） |
|---|---|---|
| `m = 0.474`（今天） | 0.56× | 1.23× |
| `m = 0.20` | 0.94× | 2.07× |
| `m = 0.10` | **1.25×** | **2.77×** |

⇒ **只修 residency（m→0.2）在今天这个 accept 下仍然不到 1.0×**；
residency 的价值是 **把 accept 变得值得买** —— 今天即使把 `a` 拉到 0.85，m 不动也只有 1.23×。

---

## 5. 排行榜（我先做的顺序）

### ① 重跑「per-layer slots × MTP verify」矩阵 —— 最重要，也最便宜
09-18 那次「pool 143→179 slots ⇒ I/O 少 3.7×，但 decode 只 +9%」是在 **MTP off、非 verify 路徑**
上量的；而那條路徑在 `k=0` 時 `layers_distinct_over_slots=0` —— **它在根本不會 thrash 的區間裡，
所以那個結論不能推到 verify**。同一條 `spec_cost_curve.py` 曲線跑兩個 pool 大小就能回答。
它同時是我先前欠著的那個「判定 m 的 union 成分」的实验，而且**只有它能把「代價來自位元組」與
「代價來自 k 次 draft 前向」分開**（今天兩者共線）。

### ② 非均勻配額（per-layer slot budget）
`k=7` 時 17 層越界（其餘層沒越界）⇒ working set 明顯不均勻。
「總 pool 不變、按實測 per-layer distinct 重新分配」預期比「整體加大 25%」更省記憶體。

### ③ prefetch 現在是**關**的
`run_server.sh:2029`：`SERVER_NO_PREFETCH="${CGC_SERVER_NO_PREFETCH:-1}"` ⇒ 我們那條曲線
（含 §2 那份 thrash 數據）全程 `CGC_NO_PREFETCH=1`。程序註解給的理由（`llama-context.cpp:2000`）：
「MTP verify safety：bg thread 在 GPU 做 verify 時填空／逐出會蓋掉正在讀的 slot」。
但同一段註解也主張 `prefetch_slot` 的 LRU victim 構造上不在當前 union 裡 ⇒ 有可能是過度保守。
**只有在 thrash 修好之後才輪到它**（thrash 沒修好時 prefetch 只是在搬更多一樣會被踢掉的東西）。
風險：歷史上有一整個 deadlock class（`llama-expert-cache.cpp:1082`）。

### ④ 把已找到的 batch 修復延伸到 verify
若 N 條併發 request 的 verify 能**共用一次 union materialize/ensure**，T 變大的同時每步讀取不增加
⇒ 這是唯一結構性的乘數。**先決條件**：union hook 是 per-ubatch 還是 per-sequence
（若是 per-sequence，batch 只幫到 compute，對 thrash 反而更糟）。這個讀程式碼 30 分鐘就有答案。

### ⑤ n-gram draft（零訓練，主要價值是診斷）
llama.cpp 有 `--spec-type ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod|ngram-cache`
（`common.h:177-181`）。實測 draft 的 expert 流量只有 verify 的 ~1%
（`draft: calls=64 union=512` vs `verify: calls=3823 union=47122` @ k=1）
⇒ 它的价值不是省 expert traffic，而是**去掉 k 次 draft 前向**，看 `m` 掉多少：
掉得多 ⇒ 該優化 draft 本身；掉得少 ⇒ 全部壓在 ①②。

### ⑥ 別做：dynamic-k（自適應深度）
用現有三輪資料算 oracle（每輪挑最好的 k）：**mean 1.035× / median 1.068×**，
而這個上界還被噪音膨脹過（同一個 `k=3` 配置 r1=7.32 vs r2=11.49 t/s，差 1.57×）
⇒ 真正的自適應上界只會更低。**同一個結論適用於任何「看單輪數字挑 k」的做法。**

### ⑦ 別做：更大的 k / tree verification
兩者都直接擴大 working set ⇒ 先把 ①② 修好再說。

---

## 6. 這條因果鏈怎麼被證偽

| 觀察 | 若 thrash 假說成立 | 若不成立，同樣結果還能怎麼解釋 |
|---|---|---|
| `layers_distinct_over_slots` 隨 k 0→4→15→17 | ✓（與 read 量同向） | 只是一个 aftermath，非因 |
| hit% 92.6→~50% 發生在 distinct 越界那一刻 | ✓ | verify 走的是另一條 path（`MTP fast path` 計數器只出現在 k≥1），跳變可能來自 path 差異而非 thrash |
| `ms/step` 對 `MiB/step` 斜率 ≈ 298 MiB/s | ✓ | `MiB/step` 與 k 共線 ⇒ 也可能是「k 次 draft 前向」的線性項 |
| k=5 偏離趨勢（讀取下降但時間上升） | ✗不一致 | 提醒這條回歸只有 r²=0.70，是弱證據 |

**證偽實驗（按訊息量排序）**：①（pool-size × verify 矩陣）單獨就能定案 —— 若 working set 回到配額內
而 `m` 不動 ⇒ thrash 假說死亡，成本在別處；② 非均勻配額；⑤ n-gram 隔離 draft 前向。

---

## 7. 复算

`Backup/phase_decomp/rsl_gain_estimate.py`（上行推導）。本文的新數字全部可由下列 shell 行重現：

```sh
python3 scripts/check/spec_cost_curve.py --self-test        # 15/15
# bytes/step 取自 teardown 行：
grep -hE "read shape|decode/pool|layers_distinct_over_slots" \
  Backup/phase_decomp/spec_cost_logs_20260918_2030/spec_cost_k*_r1_p512_n128_d0.stderr.log
python3 -c "import json;print(json.load(open('Backup/phase_decomp/spec_cost_curve_20260918_2030.json'))['analysis']['per_k'])"
```
