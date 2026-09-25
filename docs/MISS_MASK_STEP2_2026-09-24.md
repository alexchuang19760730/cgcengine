# 第 2 步：miss mask 出口 —— 2026-09-24

## 0. 一句話

**「GPU 端知道哪些 (層, expert) 是佔位」這件事做出來了，而且跟 host 的 `BATCHDBG` 逐位相同
（38 層 / 421 個對齊元素，0 處不一致）。成本量不出來（−0.95 ± 2.01 ms/step），但結構上它沒有
新增 command buffer、也沒有新增同步。判決：正確性 PASS，成本 UNRESOLVED（不是 FAIL）。**

---

## 1. 做了什麼（三個檔案，一個 build）

| 檔案 | 改動 |
|---|---|
| `src/llama.cpp/src/llama-graph.cpp` | 在 `cgc_slot_table_gpu` 分支裡，用**同一個** index vector `ids_flat` 再掛一次 gather：`vmask = get_rows(valid_table, ids_flat)`。另外給 `ids_cont` 命名（`ffn_moe_ids_cont`），讓 host 能把它抓下來。 |
| `src/llama.cpp/src/llama-context.h` | 三個新的 per-layer 捕獲表：`cache_valid_tensors` / `cache_missmask_tensors` / `cache_ids_cont_tensors`。 |
| `src/llama.cpp/src/llama-context.cpp` | (a) 按名字捕獲上面三個節點；(b) `graph_compute` 裡、dispatch **之前**、每步一次把 `llama_expert_cache_slot_table()` 的快照寫進 `valid_table`（`valid[e] = st[e] >= 0 ? 1 : 0`）；(c) `CGC_MISS_MASK_DBG` 下做一次讀回並印 `MISSMASK` 行。 |

新 env：

| env | 作用 | 代價 |
|---|---|---|
| `CGC_MISS_MASK=1` | 建節點 ＋ 每步發布 valid_table。**這是要定價的那一個。** | 2 節點/層 × 39 層 |
| `CGC_MISS_MASK_DBG=1` | 額外做一次 `synchronize` ＋ 讀回 ＋ 印。**僅診斷，禁止拿來報吞吐量。** | 1 次同步/步 |

兩者分開是刻意的：成本門檻（≤0.2 ms/step）量的是 `CGC_MISS_MASK=1` **單獨**一個；
`_DBG` 付的同步是「不許多一次同步」那條判據裡唯一不該付的東西，所以它獨立成一個開關。

### 為什麼是 gather，而不是別的

*  residency 在 host（`slot_table[e] < 0`），但**選了哪些 expert 只在 device 上** —— 這正是
   `CGC_SLOT_TABLE_GPU` 的全部意義，也是 prebind／預測路線被判死的原因（h=0.032/0.022）。
    所以交集只能在 device 上取。
*  沒有把 `slot_table` 加寬成 `[2, n_expert]`：`mul_mat_id` 必須吃 `[k, n_tokens]` 的 id 張量，
    而把 `[2, k*T]` 的 gather 切回去需要 strided view —— 那個結構已經戳過兩次
    `ggml-alloc.c:623 GGML_ASSERT(buffer_id >= 0)`。兩個 root 是唯一證明排得進去的形狀。

### snapshot 語意（重要的設計決定）

`valid_table` 是**每步一次、dispatch 之前**的快照，不是逐層即時的。理由：對某一層 L，resident
狀態只會在 **L 自己的 `ensure_batch`** 跑的時候變動，所以一步之內快照和逐層 ensure 看到的是同一
個狀態；唯一能在中途改動它的是背景 prefetch 發布 slot。**這正是跟 BATCHDBG 做逐位比對要量的東西**
—— 實測差異為 0，所以這個語意在本次臂上是對的。

---

## 2. 正確性：逐位對得上（PASS）

工具 `scripts/check/miss_mask_check.py`（`--self-test` 22/22）。

```
臂：prod-new ＋ CGC_SLOT_TABLE_GPU=1 ＋ CGC_MISS_MASK=1 ＋ CGC_MISS_MASK_DBG=1
    ＋ LLAMA_EXPERT_CACHE_BATCH_DBG=1 ＋ CGC_DECODE_PROFILE=1
cell: p2048 / n128 / d512 / b=ub 5632 / ctx 0 / warm-skip 64 / r1（gen 128，64 步計時）
log : Backup/mm_dbg/llama_bench_prod-new_p2048_n128_d512_r1.stderr.log
```

| | 值 |
|---|---:|
| 可比層數 | **38**（layer 0 不在 GPU 表路徑上，`CGC_S1_MIN_IL=1`；另有 1 層兩邊都沒 miss） |
| 對齊元素數 | **421** |
| `SET_DIFF`（同一層同一個 step 的 expert 集合不同） | **0** |
| `LEN_DIFF`（同一層兩邊序列長度不同） | **0** |
| 每元素 miss 數的平均絕對差 | **0.0** |
| **判決** | **PASS** |

抽樣對照（完全同一批 expert）：

```
BATCHDBG layer=1 misses=1 slots: e190->s36
BATCHDBG layer=2 misses=3 slots: e18->s43 e202->s9 e234->s138
MISSMASK il=1 step=4 nsel=8 misses=1 exps: 190
MISSMASK il=2 step=4 nsel=8 misses=3 exps: 18 202 234
```

### 一個必須記下的方法論坑

第一版校驗器用「layer id 不再遞增 = 新 step」切 step，**實測是錯的**：miss 數為 0 的 step 不會
產生任何行，於是兩個相鄰 step 會被併成一個（若後者首個 layer id 大於前者最後一個）。它一度報出
57/103 step 的 `LAYER_DIFF`，但逐層抽樣每一層其實都對得上 —— 假的紅燈。

改成**逐層序列比對**就沒有這個問題：同一層在所有 step 的 miss 集合排成序列，兩邊各自省略「該 step
該層沒 miss」的情形，只要判斷一致序列就等長且逐項相同；長度不同時（BATCHDBG 含 prefill step）
從尾端對齊。

⇒ **通則：任何「某個 step 可能一行都不印」的 log，都不能用單調性切 step。**

---

## 3. 成本：量不出來（UNRESOLVED，不是 FAIL）

配對臂（`gen 256` / `warm-skip 64` / `r3` / 全程 NOMINAL，唯一變數 = `CGC_MISS_MASK`）：

| | mask off | mask on |
|---|---:|---:|
| llama-bench tg（3 rep） | 5.24 / 11.79 / 11.60 t/s | 11.63 / 11.87 / 12.10 t/s |
| `CGC-DECPROF` step total（pooled 96 步）mean | 75.59 ms | 74.64 ms |
| 同上 median | 74.33 ms | 74.92 ms |
| 同上 sd | 6.74 ms | 7.47 ms |

```
Δ(mean) = −0.948 ms/step    SE = 1.027    95% CI = [−2.961, +1.064]
Δ(median) = +0.585 ms/step
解析度（95% CI 半寬） = ±2.01 ms/step     門檻 = 0.2 ms/step
```

⇒ **點估計是負的（開 mask 反而快 0.95 ms），但 CI 半寬 2.01 ms 是門檻的 10 倍**。
要在這個載體上把 ±0.2 ms 量出來，需要 ~200 個 rep（約 2 小時 GPU），不划算。

**所以成本的證據是結構性的，不是數字：**

* `CGC-DECPROF` 的 `segs=41` 兩臂**完全相同** ⇒ 沒有新增 command buffer、沒有新增段。
* 生產路徑（`CGC_MISS_MASK=1` 單獨）**沒有任何額外的 `synchronize`**；唯一那一次同步在
  `CGC_MISS_MASK_DBG` 裡，是診斷臂專屬。
* 加的是 39 個 `get_rows`，index 長度 8、I32、跟既有的 slot gather 同形狀同後端。

⇒ 結論寫成「**成本 ≤ 解析度；結構上無新增 CB／無新增同步；點估計 ≤ 0**」，
而不是「成本 ≤ 0.2 ms」。第 3 步落地時整個 delta 會是 ~10 ms 級，那時候再一次量就有意義了。

---

## 4. 順手量到、而且比成本重要的一個數

同一支 mask 放在**單段提交的診斷臂**上（`CGC_SEG_BATCH=1 CGC_B_SCHEME=1 CGC_SLOT_TABLE_GPU=1`
＋ `CGC_MISS_MASK=1 CGC_MISS_MASK_DBG=1`，identity 模式 gen 8）：

```
MISSMASK=958 行，BATCHDBG=0 行（B 臂沒有 hook，本來就沒有 BATCHDBG）
MISSMASK il=1 step=6 nsel=8 misses=5 exps: 214 182 143 175 241
MISSMASK il=3 step=6 nsel=8 misses=5 exps: 163 208 205 201 149
```

⇒ **在「完全不 fill」的診斷臂上，k=8 裡有 5 個是佔位 ≈ 62%**，39 層 ⇒ ~195 次/步。

⚠ 這個數**不能直接拿去定價**。它是不 fill 的代價；成品有背景 fill，穩態 miss 率應該靠近 A 臂
的 6.9%（實測 hit 93.1%，miss_compulsory 3167 vs capacity 93）。但它給出一個明確的警告：

> **per-expert 重算的價格對「成品穩態 miss 率」極度敏感。** 用 A 臂穩態的 21.9 次/步算是
> 3.6~4.5 ms（淨 +9~10 ms）；若成品穩態退化到 62%，就是 24.6~33 ms —— 直接吃掉全部收益。

而這正好是第 2 步的價值：**現在這個數可以被直接量了**，不用再推。第 3 步開工前，先用
`CGC_MISS_MASK_DBG` 把「單段提交 ＋ 背景 fill」臂的穩態 miss 率量出來，再決定重算 kernel 的
batch 形狀。

---

## 5. 已知限制

1. **layer 0 不在 mask 裡**（`CGC_S1_MIN_IL=1`，它的 MoE 走 CPU）。校驗器用 `--min-il 1` 排除，
   `layers_only_in_batchdbg: [0]` 是預期輸出，不是 bug。
2. **snapshot vs prefetch**：本次臂 `CGC_EVICTED_RING=0`／prefetch 策略固定，實測 0 差異 ⇒
   「背景 prefetch 中途 publish」這個風險**在這個設定下沒被觸發**。換 profile（含 prefetch）時
   要重跑校驗。
3. **只在 ntok=1 的 decode 圖上建**（`cgc_is_decode_graph` 條件），MTP verify（ntok=4）的路徑
   尚未驗證。
4. 成本數字是在 **帶 `CGC_DECODE_PROFILE` 儀器**的臂上量的（兩臂都帶，所以配對仍成立）。

---

## 6. 下一步（第 3 步）的入口條件

1. 先量「單段提交 ＋ 背景 fill」臂的**穩態 miss 率**（`CGC_MISS_MASK_DBG`）。
   * < 10% ⇒ per-expert 重算的價格維持 3.6~4.5 ms，淨 +9~10 ms，開工。
   * 20%~60% ⇒ 重算價格 8~24 ms，必須先把 fill 後台化做出來再說。
2. 重算 kernel 的判據不變：greedy 逐 rep 相同 ＋ 落在 13.6~14.2 t/s，低於 13.0 退回。
