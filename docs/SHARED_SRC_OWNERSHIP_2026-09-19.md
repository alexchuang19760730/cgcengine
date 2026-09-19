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

| 狀態 | 步總時 | `cb` | 佔比 | 出處 |
|---|---:|---:|---:|---|
| **cold**（新 server 的第一個請求） | 152.60 ms | **35.23** | **23.1%** | `DECODE_STEP_BUDGET` §3（`verify, ntok=4, n=39`） |
| **warm**（單臂逐層中位數） | ~137 ms | **9.7** | **~7%** | `/tmp/sntok_curve.json` `buckets.sums`（41 層） |

⇒ **生產是從 warm 提供的**（13.1–13.4 t/s），所以 overlap 的天花板是 **~7%**，
而 25 t/s（mean_len 2.4）要把步時從 ~137 ms 降到 ~96 ms ＝ **要拿掉 ~30%**。
**overlap 單獨到不了 25**，但它是少數**不動數值**的候選，值得做；只是不要用它來解釋 25。

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
