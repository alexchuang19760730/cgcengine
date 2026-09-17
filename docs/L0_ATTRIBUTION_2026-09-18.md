# L0 的 GPU 異常：把「四件計畫」第 2 件重新定錨（2026-09-18）

> 起因：使用者 00:57 指示「繼續優化」。依 `MEMORY_PERF` §「下一步的四件事（依投報率）」，
> 第 1 件（served 基準）已完成，第 2 件是「**L0 的 9.46 ms**」，且原本被寫成
> 「把它拉回中位（1.76）⇒ step 85.65→77.95 = **+10%**，而且**不改數值**」。
> 本文查證兩件事：**那個機制說明是錯的**，以及**那個數字只在特定形狀下成立**。

## 0. 一句話

| 問題 | 答案 |
|---|---|
| 第 2 件的**槓桿**（`CGC_S1_MIN_IL=0`）能不能用？ | **不能。** 兩個獨立的理由，都不是效能問題而是**可達性**問題（§1）。 |
| L0 的異常**是不是真的**？ | **在 `ntok=1` 的形狀下是真的**，且這次用 47 個 steady step 的逐層中位數把它釘住了（§2）。 |
| 它在**要優化的形狀**（MTP 真的投機，`ntok=2/4`）下也存在嗎？ | **這次沒有重現**（3 個 steady step，證據弱）⇒ **形狀相依，機制未歸因**（§3）。 |
| 「9.46 ms ＝ span 的 12.5%」可以直接引用嗎？ | **不行**，要先指名形狀與聚合方式（§4）。 |

---

## 1. ★ 第 2 件的機制說明是錯的 —— 兩個獨立理由

原本的述句（`MEMORY_PERF` 的「第 2 件」，出自 09-17 §EN-97）是：

> `src/llama-graph.cpp:2172` 的註解逐字：「the GPU table is gated to `il >= CGC_S1_MIN_IL`
> (default 1); **layer 0 keeps the host leaf.**」⇒ L0 是唯一還在走 host leaf 的層 —— 它 9.46 ms 的
> union（其他層 1.76）不是「大層」，是另一條路徑。⇒ 把 `CGC_S1_MIN_IL` 設 0 與預設 1 對比。

### 理由一：`union` 不是 host 工作，是 **GPU 時鐘的跨度**

`ggml-metal-context.m:497` 逐字：

> `out[1] = GPU busy union (ns) -- max(end) - min(start); << out[0] means the buffers overlap`

計算在 `:526-537`，讀 `[cb GPUStartTime]`／`[cb GPUEndTime]`。累加點是
`ggml-backend.cpp:2042` `dp_lay_uni[dp_il] += sg_union;`（`sg_union = g[1]`，`:1971-1972`）。
⇒ **它是 GPU 側的量**。host 側的括號是 `wait`（`:1949-1962`）與 `cb`（`:2020-2052`）。
「L0 走 host leaf，所以它的 host 工作變多」**與被量到的欄位無關**。
（旁證：`union/wait = 109%` 早就寫進記憶了 —— union 不是 wait 的子集。）

### 理由二：`CGC_S1_MIN_IL` 在**生產路徑上根本不會生效**，而且 L0 是**結構上的永久排除項**

- `CGC_S1_MIN_IL` 只在 `cgc_slot_table_gpu` 為真時被讀（`llama-graph.cpp:2146`：
  `getenv("CGC_SLOT_TABLE_GPU") != nullptr`），而 **`run_server.sh:1518` 只在顯式給值時才傳遞它**
  ⇒ **生產 profile 沒有開 S1** ⇒ 該旗標在生產上是 no-op。
- 就算開了，L0 也被明文排除。`scripts/check/decode_sweep.py:410-412` 逐字：
  > Layer 0 is never a candidate: CGC_S1_MIN_IL's default of 1 exists precisely to keep layer 0
  > on the host leaf (**its MoE FFN runs on CPU/BLAS, so a GPU-computed id vector would need a
  > cross-backend copy -- the shape of the 20:18 SIGSEGV**).
- 而 §EN-97 引用的那段註解，描述的其實是 **`prod25`（舊 profile）**：
  `llama-graph.cpp:2156` 寫「in the prod25 profile its expert tensors keep their full size
  (blk.0.ffn_gate_exps = 82M vs 45M for blk.1)」。**現行生產配置的 L0 是在池裡的** ——
  23:28 那筆產物的 env 區塊逐字是 `LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0: "0"`
  （`llama-expert-cache.h:52-54` 的 `cgc_env_on` 把 `"0"` 讀成 off），
  而 `llama-context.cpp:5883` 的 `if (cgc_l4_skip_layer0_on() && il == 0) return;` 因此**不觸發**。

⇒ **第 2 件需要重寫**：它的標題（L0 有異常）是對的，它的機制與槓桿是錯的。

---

## 2. L0 的異常是真的 —— 但要用「逐層 × steady steps 中位數」才看得到

### 聚合方式的陷阱（這件事本身就是一個缺陷）

`decode_step_profile.py:279` 是 `last = {l["layer"]: l for l in layers}`（**last-wins**）
⇒ 既有的逐層表描述的是**最後一個 step**，不是穩態。而多數表格還把第 1 個 step（暖池、
`cb=286.81`）一起平均。兩者都會製造假異常。

### 修好之後的數字（`Backup/phase_decomp/m3_layergpu2_20260917.json` 的 stderr，8 GiB 池、MTP env on、`ntok=1`）

47 個 steady step（丟掉前 3 個），逐層中位數：

| 層 | gpu (ms) | union (ms) | wait | cb | submit | gap |
|---|---|---|---|---|---|---|
| **L0** | **17.88** | **9.25** | 1.49 | 1.55 | 0.14 | 0.00 |
| L1..L39（40 層） | 1.37 – 2.87 | 1.19 – 1.82 | 1.37 – 2.08 | 0.01 – 1.59 | 0.11 – 0.26 | 0.29 – 2.34 |
| 中位 | 2.47 | 1.77 | 1.91 | 0.94 | 0.13 | 1.28 |

- **L0 = 中位的 7.2×（gpu）／5.2×（union）**，其餘 40 層極均勻。
- **L0 的 host 側完全正常**（wait 1.49、cb 1.55、submit 0.14 都在中位內）
  ⇒ 超額的部分**在 GPU 側**，這一條同時否掉了 §1 的機制說明。
- 量級：40×2.47 + 17.88 ≈ **116.7 ms** 的 gpu busy 總和裡 L0 佔 **15.3%**。

## 3. ★ 已裁決：**L0 在真實 decode 形狀下沒有異常 —— 7.2× 屬於 llama-bench 的那個 regime**

同一支分析器跑 `Backup/cgc_logs/llama_server_20260918_010607.log`
（`en-sched` 臂：`prefill250 + CGC_GPU_TIMING + CGC_DECODE_PROFILE_ALL`，**MTP 真的投機，`ntok=2/4`**）：

| 量 | 值 |
|---|---|
| L0 gpu / union | 10.62 / 5.74 —— **不在 top6**，gpu 只是中位的 1.8× |
| gpu top6 | L18 16.51(2.8×)、L19 12.93、L7 10.79、**L0 10.62**、L17、L10 |
| steady step 數 | **只有 3**（`--n-predict 8`），且 launch=worst=**HEAVY** |

⇒ 熱點散開、L0 不突出。於是**跑了一次 ntok 掃描**（判準事先寫在腳本 docstring 裡）：
`decode_sweep --profile prefill250 --arms en-l0-mtpoff,en-l0-mtpon --rounds 2 --warmup 1
--n-predict 128` ⇒ `Backup/phase_decomp/en_l0_shape_20260918.json`。統計量是**每次 run 內部**的
`L0 gpu ÷ 其餘 39 層 gpu 的中位數`（用 run 內比值，因為臂的位置本身就值 +12.5%，`eng-mh-0061`）。

| run | 形狀 | L0 gpu | 該臂中位 | **比值** | L0 union ÷ 中位 |
|---|---|---|---|---|---|
| 23:28（llama-bench `-p0 -n128 -d512`，MTP env **開但無投機**） | `ntok=1` | 17.88 | 2.47 | **7.2×** | 5.2× |
| 01:11 `en-l0-mtpoff`（HTTP，MTP env **關**） | `ntok=1` | **1.29** | 2.83 | **0.46×** | 0.69× |
| 01:13 `en-l0-mtpon`（HTTP，MTP env 開 **＋真的投機**） | `ntok=2/4` | 8.01 | 5.06 | **1.6×** | **0.62×** |

**判準（事先寫定）第三條成立：兩臂都 < 2×。** ⇒ **L0 沒有可重現的異常。**

- 在**純 decode（無 MTP env）**裡 L0 **比中位還快**（0.46×）—— 它是最先被排程的那一層，
  池預熱之後它沒有任何額外工作。
- 在**投機**臂裡 L0 的 `gpu` 排第一但只有 1.6×，而它的 **`union` 反而低於中位**（0.62×）
  ⇒ 抬升來自 `busy`（重疊），不是跨度。而兩臂的 `cb` 前四名都是 **L2/L1/L0/L3**
  ⇒ 那是**池填充由最前面幾層付錢**的既有模式，**不是 L0 這一層的性質**。
- ⇒ **第 2 件結案：沒有槓桿可拿。** 7.2× 是 **llama-bench `-d 512` ＋ MTP env 開但沒有投機**
  那個 regime 的產物（`llama_bench_matrix.forward_argv()` 不轉發 `--spec-type`），**那不是出貨形狀**。
  副作用：§2 的 9.46 ms 從「可引用的 L0 成本」降級為「某個非生產形狀的讀數」。

### 兩個候選（只在 llama-bench 那個 regime 裡還需要解釋）

- **(a) segment 0 真的扛了額外工作。** `GGML_SCHED_DEBUG=2` 顯示一個 segment 圖只有 **10 個節點**
  （2 個 split：空的 CPU ＋ 一個 MTL0），而 segment 0 把 `model.input_embed`＋2 views 與 layer 0
  的 leaf／`attn_inp_*` 群併在一起。但 1-token 的 get_rows 只有 8 KB ⇒ **這條仍缺一個環節**。
- **(b) segment 0 付每步開頭的啟動坡。** 分段 dispatcher 在每個 segment 邊界等 `cgc_done`
  ⇒ 每一步都從「GPU 已排空」開始。**但這條被上表第三列否掉**：投機臂（`ntok=2/4`）同樣每步開頭
  排空，L0 的 `union` 卻是**低於中位**的。

### 剩下兩個候選（尚未分離）

- **(a) segment 0 真的扛了額外工作。** 已量到的結構事實：`GGML_SCHED_DEBUG=2` 的 dump 顯示
  一個 segment 圖**只有 10 個節點**（2 個 split：空的 CPU ＋ 一個 MTL0），而
  **segment 0 把 `model.input_embed`、2 個 view 與 layer 0 的 leaf／`attn_inp_*` 群併在一起**。
  但一個 `ntok=1` 的 `get_rows` 只有 8 KB，**撐不起 15 ms** ⇒ 這條還缺一個環節。
- **(b) segment 0 是正常的一層，付的是每步開頭的啟動坡。** 分段的 dispatcher 在**每個 segment
  邊界**都要等 `cgc_done`，所以**每一步都是從「GPU 已排空」開始的**；第一步的 9 個 command
  buffer 因此落在空管線上。（旁證：`CGC-GPUTIME` 的 `gap` 每步 27–98 ms，與 `union` 同量級。）

⚠️ 這兩個候選**不能用既有的 per-command-buffer 時間戳分開**，理由見 §4。

---

## 4. 這一輪建立的可引用儀器事實（新）

1. **一個 segment = `n_cb + 1` 個 command buffer，每條 worker 執行緒一個，不是每個節點一個。**
   `ggml-metal-context.m:1075-1083` 是 `ctx->cmd_bufs[n_cb].obj = cmd_buf;`，而 `n_cb` 是
   **執行緒索引**（`:114-115`「n_cb command buffers + 1 used by the main thread」、
   `:1020-1021`「each thread creates its own command buffer」、
   `:1031` `n_nodes_per_cb = (n_nodes_1 + n_cb - 1) / n_cb`）。
   實測指紋：`CGC-GPUTIME` 的 `bufs=360` ÷ `segs=40` = **9 = `n_cb + 1`**（env `CGC_N_CB: "8"`）。
   ⇒ `ggml_metal_cgc_gpu_take`（`:503-541`）拿到的是**一層的 9 個並行切片**，
   **不是** 9 個節點的名字。
2. ⇒ **節點級 GPU 時間不需要 `MTLCounterSampleBuffer`，而且已經做好 —— 見 §4.5。**
   （`M3_VERDICT` 的「唯一活路是逐節點 GPU 時間」對了一半：**活路對，成本判斷錯**。）
3. **逐層索引的來源不是層號。** `ggml-backend.cpp:2013-2036`：有 topk 就用 topk 名字，
   沒有就用 **split 的最後一個節點**的名字，再退回 **split 序號 `i`**；
   而 `:2017` 的 `callback_eval(ttopk, …)` 若回 false，**整個 segment 不進任何桶**
   （`dp_lay_n[dp_il]++` 在 `:2047`，在那之後）。
   ⇒ 讀逐層表時要先問「哪個 segment 落在這個桶」。
4. **既有的逐層表會給出「假異常」**：`last`-wins ＋ 含第 1 步 ⇒ 必須改成
   「逐層 × steady steps 的中位數」（`/tmp/en_l0_layer_med.py`，一次性工具）。

---

## 4.5 ★ 第 3 件已落地：`CGC_GPU_NODES=1` 的**節點範圍級** GPU 時間

**不需要 `MTLCounterSampleBuffer`（因此不需要抽樣點、不需要 barrier、不擾動被量的東西）。**
`ggml-metal-context.m:1304-1315` 的 `idx_start/idx_end` 顯示每個 command buffer 編的是**連續的節點範圍**
（主執行緒的 slot `n_cb` 編 `[0, n_nodes_0)`；worker `cb_idx < n_cb` 編
`[n_nodes_0 + cb_idx*p, n_nodes_0 + min((cb_idx==n_cb-1) ? n_nodes_1 : (cb_idx+1)*p, n_nodes_1))`），
而 `n_cb / n_nodes_0 / n_nodes_per_cb / gf` 在 hook 讀取時都還在 ⇒ **範圍可以直接導出，零新狀態**。

實作：`ggml_metal_cgc_gpu_take_cb()`（逐 cb 回 `{start_ns, end_ns, first_node, last_node, ok}`）
＋ `ggml_metal_cgc_node_name()`；兩個 proc-address 註冊（`ggml-metal.cpp`）；
`ggml-backend.cpp` 在 hook 裡把每個 cb 的時長**按節點種類攤分**（固定前綴字彙表、最長匹配），
每 8 步印一張 `CGC-GPUNODE:` 表；`run_server.sh` 的 allowlist 加 `CGC_GPU_NODES`。

**自我檢查（設計成會失敗的那種）**：`seg_busy` 必須等於同一行的 `layer gpu_sum`，因為兩者加總的是
同一批 per-segment Metal busy time，只是走兩條不同的路徑。實測 **5 個 step 全部 `delta=0.00%`、
`(other)=0.00 ms`** ⇒ 範圍與 buffer↔節點的對應**逐位吻合**。

```sh
RUN_REPLAY_BENCH=0 python3 scripts/check/decode_sweep.py --profile prefill250 \
  --arms en-nodes --rounds 2 --warmup 1 --n-predict 48 \
  --json Backup/phase_decomp/en_nodes_20260918.json --force
```

實際輸出（`step=67 seg_busy=248.07 ms`）：

| kind | ms | % |
|---|---|---|
| `(other)` | 39.93 | 16.1% |
| `node` | 33.77 | 13.6% |
| `cache` | 31.76 | 12.8% |
| **`ffn_moe_`** | 25.24 | 10.2% |
| `norm` | 9.41 | 3.8% |
| `conv` | 7.17 | 2.9% |
| `attn_output` | 6.09 | 2.5% |
| `attn_post_norm` | 6.06 | 2.4% |
| `ffn_` | 5.94 | 2.4% |
| `linear_attn` | 5.20 | 2.1% |
| `ffn_moe_probs` | 4.87 | 2.0% |
| `ffn_moe_topk` | 1.48 | 0.6% |

### ⚠️ 兩個限制 —— **先不要拿這張表去套 M3 的「`ffn_moe_*` ≥ 40%」判準**

1. **`ffn_moe_gate`／`ffn_moe_up`／`ffn_moe_down` 沒有出現在表上**，而前三名是 `(other)`／`node`／`cache`。
   要嘛那些熱節點的名字不是 `ffn_moe_*`（`node_NN` 是 ggml 的自動名、`cache_*` 是池的張量名，池 repoint
   之後 MoE 的運算元可能就叫這些），要嘛攤分**按節點數**加權會把大 kernel 的時間攤給同一 buffer 裡的小節點。
2. **攤分是按節點數加權的**（一個 cb 只給出一個 group 時長）⇒ 一個含 1 個大 GEMV ＋ 數個小節點的
   buffer 會**高估小節點**。要真正的「每節點時鐘」需要每節點一個 sample point（barrier，會擾動）
   或每個節點一個 command buffer（會改派送）。

⇒ **第 3 件交付的是「可用的」儀器（有精確自我檢查），不是「已調校的」詞彙表。**

## 5. 這對「四件計畫」的修正

| # | 原述句 | 修正後 |
|---|---|---|
| 1 | served 基準（已完成） | 不變 |
| 2 | **L0 的 9.46 ms，可用 `CGC_S1_MIN_IL=0` 省，+10%，不改數值** | **結案：沒有槓桿。** 機制說明錯（§1）、槓桿不可達（§1）、異常只在 llama-bench 的 `-d512`＋MTP-env-無投機 regime（§3）。真實 decode 裡 L0 在或低於中位。 |
| 3 | 節點級 GPU 時間（唯一儀器） | **已落地**（§4.5）：範圍可導出、零新狀態、有精確自我檢查、**不需要 `MTLCounterSampleBuffer`**。詞彙表待調校。 |
| 4 | MoE gather 融合／batched-union（唯一 1.8× 項） | **不變**：它的閘門現在**有了**（第 3 件），所以「沒有判準就不要實作」這條不再是阻礙；剩下的是先調校 §4.5 的兩個限制。 |

**仍然不要投**：overlap 的單向 A/B（生產 pool 已 1.13）、`n_max`（幾何上限 +9%）、
非生產 pool 的任何 A/B。

---

## 6. 產物與指令

| 檔案 | 內容 |
|---|---|
| 本文 | 第 2 件的重定錨 ＋ 逐層中位數表 ＋ 儀器事實 |
| `scripts/check/decode_sweep.py` | 新臂 **`en-sched`**（`GGML_SCHED_DEBUG=2` ＋ `GPU_TIMING` ＋ `ALL`，**MTP 保持 on**）—— 一次跑就同時給 split 組成與逐層表 |
| `Backup/phase_decomp/en_l0_sched_20260918.json` | `en-sched` 的產物 |
| `Backup/cgc_logs/llama_server_20260918_010607.log` | 它的 stderr（含 `## SPLIT` dump 與逐層表） |
| `/tmp/en_l0_layer_med.py` | 逐層 × steady steps 中位數分析器（一次性） |

```sh
# 產出本文 §2 的表（既有產物）與 §3 的表（新臂）
python3 /tmp/en_l0_layer_med.py <server log> <drop_first_n>

RUN_REPLAY_BENCH=0 python3 scripts/check/decode_sweep.py --profile prefill250 \
  --arms en-sched --rounds 1 --warmup 0 --n-predict 8 \
  --json Backup/phase_decomp/en_l0_sched_20260918.json --force
```
