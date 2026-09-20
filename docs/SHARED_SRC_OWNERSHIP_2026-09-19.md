# 共享 `src/` 的擁有權與歸因協定（2026-09-19）

寫這份的原因：多條 session 同時改同一個 repo，而這個 repo 的**所有比較都建立在「同一顆 build」**
這個前提上。當一個檔案有未提交的改動時，任何人 rebuild 出來的 binary **不對應任何 commit**，
於是兩邊的數字都失去歸因。這份文件把擁有權與協定一次寫定。

## 1. 現況（HEAD = `4fdfaa8de`）

| 檔案 | 未提交的改動屬於誰 | 狀態 |
|---|---|---|
| `src/llama.cpp/ggml/src/ggml-backend.cpp` | — | **已提交於 `4fdfaa8de`**，工作樹乾淨 ⇒ **可轉移** |
| `scripts/check/attn_moe_split.py` | — | 同上（新增檔，一起提交） |
| `src/llama.cpp/build/bin/libggml-base.0.19.0.dylib` | — | 同上（`+50` 行的 KIND×OP 儀器在裡面） |
| `src/llama.cpp/src/llama-context.cpp` | 線 B/C（未提交） | 他們持有 |
| `src/llama.cpp/src/llama-expert-cache.cpp` / `.h` | 線 B/C（未提交） | 他們持有 |
| `src/llama.cpp/tools/llama-bench/llama-bench.cpp` | **兩條線都有** | ⚠ 見 §4 |
| `scripts/check/{decode_sweep,http_duo,m123_oracle_gate,mtp_accept_ab,prod_matrix}.py`、`scripts/run_server.sh` | 線 B/C（P0-2 的 pattern-kill 修復） | 他們持有 |

**`ggml-backend.cpp` 自此歸線 B（要實作 overlap 的那條）。** 本線（引擎歸因）已把它的改動提交，
不再編輯它；若日後需要再改，用 `Backup/patch_kind_op_xref.py`（5 個錨點，全部 assert 命中數＝1，
對未改前的檔已驗過每個錨點唯一）重放，並先敲窗口。

## 2. 為什麼是「提交」而不是「還原」

還原（`git checkout -- ggml-backend.cpp`）看起來更乾淨，實際更糟：**被追蹤的
`libggml-base.0.19.0.dylib` 仍然含那個儀器**，於是原始碼說 A、binary 說 B ——
那正是閘門檢查 8（原始碼↔產物同步）存在的理由。提交則讓兩者一致，且把改動寫進歷史（不會像
`/tmp` 產物一樣消失）。閘門結果：`0 FAIL`；檢查 8 的 (a) 產物一起 staged、(b) binary 比 staged
原始碼新 **兩項皆 PASS**；4 個 SKIP 都是既有可選項（非 dev 分支的 @rpath 驗收、兩個重型驗收
未啟用、`RUN_REPLAY_BENCH=0`）。

## 3. 歸因協定（這才是真正解決「無法歸因」的部分）

**git 乾淨不是判準** —— 這個 repo 把 build 產物納入版控，所以 `git status` 在每次 rebuild 之後
本來就會髒。判準是**指紋**：

```sh
python3 scripts/check/engine_freeze.py freeze --tag <你的名字>
python3 scripts/check/engine_freeze.py verify --tag <你的名字>   # 每一臂前後各一次，貼 flags/artifacts/source
```

現行 tag：**`post_kindop_0919`**（`Backup/engine_freeze/post_kindop_0919.json`，**本檔被 gitignore，
是本機記錄**）：

```
source.head = 4fdfaa8de
llama-server                 b70ff34a8bc69441aa6d158b
libllama.0.0.279.dylib       1b7c6d1bae17ba9f4dd9d366
libllama-common.0.dylib      b5ce0bb98397227152f5f9f7
libggml-metal.0.19.0.dylib   2603e676eeadff44f090b3f8
libggml-base.0.19.0.dylib    8cb4a6b74b9cc2f5433b7738   ← KIND×OP 儀器在這一顆
```

### ⚠ 兩套 digest 函數並存 —— 混用會看到幻影漂移

- `engine_freeze.py` 用 **sha256 前 24 hex**（`libggml-base 8cb4a6b74b9cc2f5433b7738`）。
- 另一條線的 harness 報告頁首、以及 `m123_oracle_gate` 的 `build` 行用**另一種**（例：
  `libggml-base 054fb22f04a01c5c`、`libllama aab787412e572550`）。

兩者**不是同一個函數**，數字不可能相等。引用時要指名是哪一套，否則會把「格式不同」誤判成「build 變了」。
（本線曾把兩者當同一組比對並宣稱相符，那是錯的。）

### 對 `pre_plainmatch_0919`（08:31）的實測漂移

`verify --tag pre_plainmatch_0919` ⇒ 唯一漂移的產物是 **`libggml-base`（本線的改動）**；
`llama-server`／`libllama`／`libllama-common`／`libggml-metal` **全部 MATCH**。
⇒ 本線只動了 ggml-base 這一顆，其餘受試對象沒有被碰到。

## 4. ⚠ `llama-bench.cpp` 是混合擁有權

它同時含**兩條線**的未提交改動：
- 本線：`--warm-skip N`、`-c/--ctx-size N`、讀 `CGC_SERVER_TEMP`/`TOP_P`/`TOP_K`/`MIN_P` 與
  `CGC_MTP_REJECTION`（`dist`）—— 全部預設 0／不設＝逐位保持舊行為；
- 另一線：`--prompt-file`、`--fixed-fill-seed`。

⇒ **任何一方都不能整檔提交並宣稱是自己的**。而且它只影響 `llama-bench` 這顆 binary，
**不影響 `llama-server`**（服務路徑不受影響）。

## 5. overlap 專案要用的兩個數字（先過這一關再動手）

overlap（把專家填充藏到 GPU 工作底下）能動的是 `cb`（DECPROF 的 fill/槽管理桶）。它的量級
**隨池的冷熱差很多**：

### ⚠ 本節已於 19:3x 更正 —— 先前那個「warm 只有 7%」是**中位數乘法**的產物

先前的表是：

| 狀態 | 步總時 | `cb` | 佔比 | 出處 |
|---|---:|---:|---:|---|
| cold | 152.60 ms | 35.23 | 23.1% | `DECODE_STEP_BUDGET` §3（step 級，真加總）|
| ~~warm~~ | ~137 ms | ~~9.7~~ | ~~~7%~~ | ~~`sntok_curve.json` `buckets.sums`~~ |

**「warm 7%」那一格是錯的**：它把 `per_layer` 的中位數**乘上層數**當成總和
（`0.2377 × 41`）。逐層中位數不是總和 —— 分佈是右偏的，乘起來會系統性低估。
實測對帳：`step=1` 的步級行印 `cb=476.95`，而同一支 log 的 40 層逐層 `cb` 相加 = **476.9** ⇒
**真值要用逐層相加，不是中位數×層數。**

### 更正後（實測，引擎自己的步級加總，`191400` log，121 個 decode 步）

| 量 | 值 | 佔步時 |
|---|---:|---:|
| 步總時（= `wait + cb + submit`，三者相加等於 total 到 0.1 ms） | **198.2 ms** | 100% |
| **hook 窗（`cb` + `submit`）** | **55.7 ms** | **28.1%** |
| **`gap`（直接量到的 GPU 空檔）** | **63.1 ms** | **31.9%** |
| `gap / hook` | **1.13** | — |
| `corr(gap, cb+submit)` | **1.000** | — |

而且 `gap/hook` 在**整個 run 內是穩定的**（分段 1.03 → 1.16 → 1.21 → 1.20 → 1.20），
同期 `Σcb` 從 169 掉到 33 ms、步時從 419 掉到 153 ms ⇒ **不是冷池產物，是結構性的**。

`ggml-backend.cpp:2130` 自己就寫了為什麼：「這個視窗坐在**前一段的 hook+submit** 裡面，所以
`gap` vs `cb+submit` 是兩個儀器的內建交叉校驗」。**corr = 1.000 說明它們是同一個區間的兩個視角：
CPU 在做 top-k 槽管理＋阻塞填充的整段時間，GPU 沒有工作。**

⇒ **overlap 的天花板是 ~28–32%，不是 ~7%。** 25 t/s（mean_len 2.4）要把步時
172 → 94 ms ＝ 拿掉 ~45% ⇒ **overlap 是唯一一個量級到得了那個方向的候選**，
這比我先前給的估法樂觀 4 倍。

### 但它**不是** `cache_r` 的功勞（同時證偽我上一輪的說法）

同一支工具看逐層（`attn_moe_split.py layers`）：

| 家族 | 層數 | wait | cb | submit | gpu | union | gap | gpu/union |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **GDN（擁有 `cache_r_l*`）** | 30 | 3.74 | 0.54 | 0.19 | 4.03 | 3.53 | **1.09** | **1.14** |
| full-attn（`cache_k/v`） | 10 | 2.59 | 0.40 | 0.18 | 3.85 | 2.34 | **1.00** | 1.65 |

⇒ `gpu/union ≥ 1` ⇒ **span 內 GPU 是塞滿的**（buffer 有重疊）⇒ 逐層空檔不是故事。
⇒ GDN 的每層 `gap` 與 full-attn **幾乎相同**（1.09 vs 1.00）⇒ **`cache_r` 的 cpy 沒有呈現為空檔**；
它花的是 span 內的 GPU 時間（union 3.53 vs 2.34 ＝ +1.19 ms/層 × 30 層 ≈ +36 ms/步）。
⇒ 真正的大空檔在**步層級**，而且它屬於 **hook 窗**，不屬於任何一層的計算。

### 「剩下的 68%」是什麼 —— 兩種切法都閉合，答案一致

`wait + cb + submit` 與 `union + gap` **不是兩個模型，是同一條步時長的兩個視角**（中位數，129 個 decode 步）：

| 切法 | 分解 | 合計 | vs `total` |
|---|---|---:|---:|
| A（CPU 側） | `wait` 145.3 ＋ `cb` 51.1 ＋ `submit` 8.4 | 204.8 | 216.5（**94.6%**） |
| B（GPU 側） | **`union` 135.9（62.8%）** ＋ **`gap` 69.1（31.9%）** | 204.9 | 216.5（**94.7%**） |

⇒ **未閉合的 ~5% 是 MTP 頭（`L40`）與沒有歸屬到 trunk 層的 segment**（`segs=41` vs `layers=40`）。

**⇒ 答案：剩下的是 `union`（GPU 跨距）≈ 63%。** 而它是「GPU 在做事」的區間 ——
下面的分布支持這句話：

| 逐層 `gpu/union`（4600 個層-樣本） | 數量 | 佔比 | 意思 |
|---|---:|---:|---|
| ≈ 1.00（≤1.02） | 2453 | **53%** | **span 內完全塞滿，沒有隱藏空檔** |
| > 1.02（buffer 重疊） | 2147 | 47% | 重疊量會蓋住未知的 span 內空檔 |

⇒ **超過一半的層可以斷言「span 內沒有空檔」**（`gpu` 加總剛好等於 span ⇒ 每個 buffer 首尾相接）。
⇒ 那些層的 3.55 ms 就是**真的在執行**。剩下的 47% 分不開，是儀器邊界（`gpu` 重複計數）。

### 那 63% 的 span 裡在執行什麼（op 級，`190212` log，同一顆引擎）

`MUL_MAT` 15.1% ＋ `MUL_MAT_ID` 3.7% ＝ **兩個 matmul 只有 ~19%**；
其餘是 **elementwise（`ADD` 14.2 ＋ `MUL` 11.1 ＋ `UNARY` 7.5 ＋ `SCALE` 1.9 ≈ 34.7%）**、
norm（`RMS_NORM` 7.5 ＋ `L2_NORM` 3.3 ≈ 10.8%）、copy/gather（`CPY` 6.9 ＋ `GET_ROWS` 5.1 ＝ 12%）。
⇒ **span 是記憶體型的管線工作，不是 matmul FLOPs。**

### 對 25 t/s 的路線圖（用 warm 步時 ~175 ms、mean_len 2.4）

| 步驟 | 步時 | t/s |
|---|---:|---:|
| 現況 | 175 ms | 13.7 |
| **把 gap 全藏掉（overlap 的上界）** | **122 ms** | **~19.7** |
| 再從 span 砍 23%（108 → 83 ms） | 94 ms | **~25.5** |

⇒ **overlap 單獨到 ~20 t/s，到不了 25；但它是唯一能把 32% 那個大塊搬走的候選**，
剩下的缺口只能從「63% 的 span」裡砍（而那 63% 已經被證明有一半是塞滿的 ⇒ 要真的減少工作，
不是填洞）。

⚠ 另一個必須先講清楚的區別：**先前被判死的是 split-MMID（把 H/C 切開重排）**，死因是
`llama-graph.cpp` combine 的 left-to-right 加法結合序 ⇒ 不逐位元相同。**若 overlap 是
「把填充排到另一條 stream、不重排任何計算」，它不屬於那個死因**。這兩件事不要混。

## 6. 本線現在的改動清單（供對帳）

| 檔案 | 內容 |
|---|---|
| `ggml-backend.cpp` | KIND×OP 儀器（已提交 `4fdfaa8de`） |
| `scripts/check/attn_moe_split.py` | 讀 `CGC_GPU_NODES`／`CGC_GPU_OPS`／`CGC_GPU_NODES_TRACE`（已提交） |
| `scripts/check/llama_bench_matrix.py` | `--warm-skip`／`--ctx-size` 透傳（未提交） |
| `scripts/check/decode_step_profile.py` | `mtp` arm、移除 pattern-kill、`ps`→`pgrep` 回退（未提交） |
| `llama-bench.cpp` | 見 §4（未提交，混合） |

---

## 7. 2026-09-20 11:5x：本線改名 **`線A (ace)`**（operator 指定），並記下新的碰撞面

舊標籤撞了：本線的記憶寫「本線（line I）」，而另一條 session 的文件也署名「（線 I，2026-09-20 下午）」。
**⇒ 本線 ＝ `線A (ace)`；`線 I` 留給做 `cb`／F1–F5 的那條**（它的擁有物＝快取命中儀器，
正是 `線 I = 儀器／快取幾何線` 的字面意思）。

- 本檔 §1 的「`ggml-backend.cpp` 自此歸線 B」**仍然有效**；但 §EN-308 起本線在該檔的
  **`hook_seg`／submit 迴圈**放了逐層 KIND×OP 儀器（`fc0b8a406`），所以那一區要動之前請先讀
  `docs/S2_IMPLEMENTATION_PLAN_2026-09-20.md` §1 的行號表。
- `src/llama.cpp/src/llama-context.cpp`：本線現在只持有 **`graph_compute` 區段**
  （`a1a2473fd`，S1 探針的兩個守衛）；hook（`:5381` 的 `CGC_LAYER_AHEAD_PREFETCH`）不是本線的。
- 完整的命名表、五個碰撞面與 G1 的**兩條階梯**：
  `docs/ENGINE_LINE_ASSIGNMENT_AND_G1_LADDERS_2026-09-20.md`。
