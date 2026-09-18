# 「會編碼」的權重接到名字表 ⇒ 一個**排名**（`ffn_moe_*` vs `cache` vs `attn_*`）

日期：2026-09-18 03:1x–03:2x（+08）
儀器：`CGC_GPU_NODES=1`（名字表，新增 `wcntw` 欄與 `*bywork` 排序）＋ `CGC_GPU_OPS=1`
狀態：**未 commit**（動了 `src/`，未跑 D5）
相關：`docs/LAYER_NODE_CENSUS_2026-09-18.md`（節點普查）、`docs/OP_WORK_WEIGHTED_BUDGET_2026-09-18.md`（op 表）、
`docs/M3_MOE_SHARE_2026-09-18.md`（已被本檔取代其 §4 的讀數）

---

> ## ⚠️ 2026-09-18 11:0x 補註（由引擎層那條線加上，**原文一字未刪**）
>
> **§6.1／§6.2 把 `cache` 標成「池自己的暫存／池的搬運」是錯的。** 用**當前詞彙表**把一份新
> `CGC-GRPH` dump（log `llama_server_20260918_050331`，4116 節點）的 `cache` 桶成員名字逐字印出來：
>
> ```
> 150 cache_r_l# (view)                        120 cache_r_l# (view) (copy of conv_input-# (view))
>  60 cache_s_l# (view)                         90 cache_r_l#/cache_s_l# (reshaped)(…)
>  20 cache_v_l# (view)  20 cache_k_l# (view)   20 cache_k_l#/cache_v_l# (view) (permuted)
> op: VIEW 290 | CPY 210 | RESHAPE 60 | SCALE 60 | SET_ROWS 20 | PERMUTE 20   （與本檔 §6.2 逐格相同）
> ```
>
> `cache_r_l*`／`cache_s_l*` 是**遞歸狀態**（`src/llama.cpp/src/llama-model.cpp:378-379` 的
> `pattern_r_cache`／`pattern_s_cache`），`cache_k_l*`／`cache_v_l*` 是 **10 個全注意力層的 KV**。
> ⇒ **這個桶是「每 token 讀寫遞歸／KV 狀態」的管線，與專家池無關。**
> 池的填充在 **CPU 側的 hook**（計在 `CGC-SEG` 的 `cb`），**在 GPU 節點表裡沒有成本** ——
> 這與 `fill_wait = 0.000` 一致。
>
> **因此下列三句作廢**（本檔 §6.1 與 §6.2 各一段）：
> 1. 「**池自己的暫存（`cache`，7.7–9.8%）比整個 MoE 專家 GEMV 家族（3.2–6.3%）大**」——
>    份額對，**歸屬錯**：那是**狀態快取**，不是池。
> 2. 「`cache`（池自己的暫存：CPY／SCALE／SET_ROWS／views）… **全部是池自己的搬運**」——
>    同上，**全部是狀態管線**。
> 3. 「**最大的一塊是「真運算但沒有子系統名」**（`node` 20–29%）… 任何進一步的排名都先卡在**命名**上」
>    —— 前半對，後半**已由命名解決**：`node` 的 350 個逐名核對後**幾乎全是 delta-net**
>    （`MUL_MAT ne=[8192,2]`×29、`MUL_MAT ne=[32,2]`×30、`GET_ROWS`×90、`ADD`/`UNARY`×30、
>    `FLASH_ATTN_EXT`×10）⇒ 它是**線性注意力**，不是雜項。
>
> **更正後語意**：`node` 17.7%（delta-net 運算）＋ `cache` 10.6%（遞歸／KV 狀態管線）＋
> `z-`／`gdn_out`／`conv`／`linear_attn`／`q/k/v_conv`／`alpha`／`beta` ≈15–20%
> ⇒ **線性注意力家族 ≈35–45%，MoE 家族 ≈12–15%。支配區塊是 delta-net，不是 MoE。**
>
> **仍然有效**：§3 的三個自我檢查、§4 的兩個識別更正（`ffn_moe_gate/up/down`＝MUL_MAT_ID、
> `ffn_moe_topk`＝VIEW）、§5 的粒度限制、§6.3 的「M3 的 40% 判準三法皆否」、§7 的用法禁令。
> 出處：`.workbuddy/memory/2026-09-18.md` §EN-132。

---

## 0. 一句話

把 op 表的「**只有會編碼的節點才進分母**」這個權重接到名字表之後，得到的第一個可引用排名是：

> **最大的一塊是「沒有名字的真運算」*（`node`，20–29% of `wait`）；
> 而在有名字的區塊裡，池自己的暫存（`cache`，7.7–9.8%）大於整個 MoE 專家 GEMV 家族（3.2–6.3%），
> 也大於整個注意力家族（2.8–4.7%）。**

⚠️ 但這個排名有一個**必須同時交付的條件**：權重**不是粒度不變的**（§5）。可引用的是
**家族總和**（在同粒度下 run-to-run 穩定，見 §5.1），不是逐項的百分比。

---

## 1. 為什麼需要它（③ 的原述句）

`docs/OP_WORK_WEIGHTED_BUDGET_2026-09-18.md` 的結尾：

> 把「會編碼」的權重**接到名字表**（`ffn_moe_*` vs `cache` vs `attn_*`）—— 這是唯一能讓 ③
> 從「方向」變成「排名」的一步，且是純消費端改動。

原本的名字表有一個**結構性偏差**：它把每個 command buffer 的時長按**節點數**分給各名字桶，
而 `ggml-metal-context.m:1099` 是 `const int n_main = MAX(64, 0.1*gf->n_nodes);`
⇒ **每個 segment 的主執行緒 buffer 永遠吃 ≥64 個節點**，任何住在裡面的桶都被除以 ≥64。
而一層裡 **約 40%** 的節點是編碼器明確跳過的 no-op（`ggml-metal-ops.cpp:242-252`：
`case GGML_OP_NONE/RESHAPE/VIEW/TRANSPOSE/PERMUTE: // noop -> next node`）
⇒ 那些節點**稀釋了分母而不貢獻任何 GPU 命令**（本輪 dump 直接數，**單一** 4116 節點的圖：
VIEW 972 ＋ RESHAPE 600 ＋ TRANSPOSE 30 ＋ PERMUTE 30 ＝ **1632/4116 ＝ 39.7%**）。
實測後果：`ffn_moe_gate` 的 `cntw` 只有 **0.7%**，而它的上界是 **44.7%**。

權重的定義（`ggml-backend.cpp` 的 `cgc_op_emits_work`，由上面的編碼器原始碼背書，不是模型）：

```
wcntw[k] = Σ_buffers  dur(buffer) * (#named nodes of kind k in buffer that ENCODE)
                                  / (#named nodes in buffer that ENCODE)
```

`cntw`（舊）的分子分母則用**全部有名節點**。剩下的差額不再被靜默重分配，而是印成
`residual`（§3.3）。

---

## 2. 改動（純消費端；沒有動 Metal 側）

| # | 位置 | 內容 |
|---|---|---|
| 1 | `ggml-backend.cpp` | `cgc_node_op` 的解析條件由 `ns_ops` 放寬到 `ns_on`（`CGC_GPU_NODES`）。**理由**：名字表需要每個節點的 op，而若要求第二個環境變數，這個欄就會在**唯一會讀名字表的臂**裡靜默缺席 —— 同 `CGC_VERIFY_OP_TIMING`／`CGC_GRPH_DBG` 已經付過兩次的陷阱。op 表本身仍由 `CGC_GPU_OPS` 開。 |
| 2 | 同上 | 新增 `ns_kind_wns[48]`、`ns_kind_work_ns`、`ns_kind_work_nd` 三個累加器。 |
| 3 | 同上 | 名字迴圈加 `wcnt[49]`／`wtot`，與 `cnt`／`tot` **同一趟**算（兩個分母必須描述同一組節點，否則名字表就不再是那個 buffer 的分割）。 |
| 4 | 同上 | 累加：`wtot > 0` 才分配，並記 `ns_kind_work_ns`。 |
| 5 | 同上 | 印表機：每列加 `wcntw`；新增 `work-attributed … / residual …` 標頭行；新增 `*bywork` 前 10 名（按 `wcntw` 排序）。 |
| 6 | 同上 | 重置區補上三個新累加器。 |
| 7 | 同上 | 詞表加 `top_k`（見 §4.2 —— **對 trunk 是惰性的**）。 |
| 8 | `decode_sweep.py` | 兩個臂：`en-work`（**預設粒度**，這才是 `cntw` 失效的那個臂）、`en-work-fine`（`CGC_CB_N_MAIN=1` + `CGC_SERVER_N_CB=16`）。 |

---

## 3. 儀器的自我檢查（三個，都會失敗）

### 3.1 內部一致：`delta = 0.00%`
名字表仍然必須與逐層 `layer gpu_sum` 逐格相同。三個臂全部 `delta=0.00%`、`other=0.00 ms`。

### 3.2 **跨表**：名字表 ≡ op 表（絕對值）
名字表的 `ffn_moe_gate+up+down` 與 op 表的 `MUL_MAT_ID` 是**同一批節點**（§4.1），

| 臂 | 名字表 | op 表 | ms-ratio |
|---|---|---|---|
| `en-work`（粗） | 7.83 ms | 7.82 ms | **1.001** |
| `en-work-fine`（細） | 12.46 ms | 12.47 ms | **1.000** |

**這證明實作正確，但不證明權重良置** —— 兩件事要分開講（lesson `eng-mh-0064`）。
⚠️ 兩表的**百分比**卻不同（3.61% vs 3.99%）：名字表除以 `ns_total`（**每一個** buffer），
op 表除以 `nsop_total`（**有 ≥1 個可解析 op** 的 buffer）。跨表比對一律回到 ms
（lesson `eng-diag-0036`）。

### 3.3 殘差（新印的那一欄）
`work-attributed 241.44 of 290.95 ms (83.0%) | named_work_nodes=2455 | residual=49.51 ms (17.0%)`
殘差＝「整格沒有任何有名工作節點」的 buffer 的時長，也就是**名字表無法主張的質量**。
它不是誤差，是儀器偽影的量（與 op 表的 `total - sum(wcntw)` 同源）。

| 臂 | 熱態 | 粒度 | 殘差中位（n=4 trunk 步） |
|---|---|---|---|
| `en-work` v1 | NOMINAL | 粗 | **19.9%**（17.0–23.9） |
| `en-work-fine` v1 | HEAVY | 細 | **21.9%**（13.1–33.9） |
| `en-work` v2 | HEAVY | 粗 | **23.9%**（20.3–26.4） |
| `en-work-fine` v2 | HEAVY | 細 | **20.0%**（13.0–27.2） |

⚠️ **我一開始把它歸因給熱態，那是錯的**：HEAVY 的**細**臂是 20.0%，與 NOMINAL 的粗臂相同、
低於 HEAVY 的粗臂 ⇒ **殘差不是熱態的量，是切片組成的量**（哪個 buffer 剛好整格都是 no-op）。
四臂都落在約 **20–24%**，而它的存在就是「不要把絕對 `seg_busy` 當時長引用」的證據。

---

## 4. 兩個識別更正（都是**用 dump 證的**，不是推論）

### 4.1 `ffn_moe_gate` / `up` / `down` **就是** MUL_MAT_ID；`ffn_gate` / `ffn_up` 是稠密的 MUL_MAT
```
CGC-GRPH[..] name=ffn_moe_gate-40 op=30(MUL_MAT_ID)      ← 專家 GEMV
CGC-GRPH[..] name=ffn_gate-0     op=.. (MUL_MAT)          ← 稠密（共享專家）
```
逐桶 op 普查（4116 節點的 prefill dump）：`ffn_moe_gate` 80 節點、`ffn_moe_up` 80、`ffn_moe_down` 80，
**全部都只有一個 op：MUL_MAT_ID**；`ffn_gate`／`ffn_up` 各 80、全部 `MUL_MAT`。
⇒ 這三個桶的**和**必須等於 op 表的 `MUL_MAT_ID` —— 而它確實等於（§3.2 的 1.001／1.000）。

### 4.2 `ffn_moe_topk` 是 **VIEW**，不是 top-k
```
CGC-GRPH[76] name=ffn_moe_topk-0 op=38(VIEW)     ne=[8,2]
CGC-GRPH[75] name=ffn_moe_argsort-0 op=70(ARGSORT) ne=[256,2]
```
⇒ 那個**看起來像**「MoE 選擇步」的桶，成員全是 view，所以它的 `wcntw` **按建構為 0.00**
（表中逐格可見，而 `cntw` 是 0.7%）——**這是儀器正確，不是權重壞了**。
真正的選擇成本在 `ffn_moe_argsort`（ARGSORT，0.4%）與一個裸名 `top_k`（TOP_K，`ne=[256,2]`）。

同時更正上一輪留下的兩處：先前把「`ffn_moe_topk` solo 精確 11.5%／620 µs 每節點」當成陽性對照 ——
那是**對一個 view 的 buffer 計時**，與 `docs/M3_MOE_SHARE_2026-09-18.md` 的偽影更正同源，
現在有了**第二個獨立理由**（不只是粒度，還有身分）。

**`top_k` 詞表修正對 trunk 是惰性的**（誠實記下）：`top_k` 節點只出現在 **MTP draft** 圖，
trunk decode 圖裡沒有；而「log 裡有 4 次 `top_k`」是**假陽性** —— 那 4 行是
`CGC-PHASE-SPLIT: cap=8 routable=… / top_k=8 -> …`（prefill 的**參數**，不是節點名）。

---

## 5. ★ 限制：權重**不是粒度不變的**

### 5.1 先把兩個軸分開

| 臂 | 熱態 | 粒度 | 用途 |
|---|---|---|---|
| `en-work` v1（`031141`） | **NOMINAL** | n_main=64, n_cb=8 | 熱態軸的一端 |
| `en-work` v2（`031423`） | HEAVY | n_main=64, n_cb=8 | 粒度軸的粗端 |
| `en-work-fine` v2（`031528`） | HEAVY | n_main=1, n_cb=16 | 粒度軸的細端 |
| `en-work-fine` v1（`031240`） | HEAVY | n_main=1, n_cb=16 | （與 v2 細端同粒度、不同 build） |

⚠️ **v1 的「粒度對」橫跨 NOMINAL→HEAVY，是被混淆的** —— 一發現就丟棄，改用 v2 這一對
（兩臂都 HEAVY）。這是本輪最容易誤導自己的地方。

### 5.2 熱態軸：**~15% 的均勻位移，排名不變**
`v1`粗（NOMINAL）→ `v2`粗（HEAVY），以 `% of wait` 計：`node` 28.9→24.1（0.83）、`(other)` 14.0→11.4（0.81）、
`cache` 9.4→7.7（0.82）、`norm` 9.4→7.7（0.82）、`ffn_moe_` 8.8→7.6（0.86）、`ffn_` 5.0→4.3（0.86）、
`attn` 4.7→3.9（0.83）、`moe_gemv` 3.8→3.2（0.84）、`dense_gemm` 2.5→2.2（0.88）、`conv` 2.9→2.5（0.86）、
`moe_select` 1.5→1.1（0.73）。**11 個家族裡 10 個落在 0.81–0.88、中位 0.84** ⇒ 熱態造成的是
**均勻位移**，不是重新排序（殘差不會跟著熱態動 —— 它是切片組成的量，見 §3.3）。
**⇒ 排名對熱態穩健。**

### 5.3 粒度軸（v2 粗 → v2 細，同熱態）：**GEMM 家族 ~1.9×，逐項最差 3.6×**

| 家族 | 粗 | 細 | 比值 | |
|---|---|---|---|---|
| `moe_gemv`（MUL_MAT_ID） | 3.2% | 6.3% | **1.97** | ✗ |
| `dense_gemm`（ffn_gate/up） | 2.2% | 4.2% | **1.91** | ✗ |
| `conv/ssm` | 2.5% | 0.7% | **0.28** | ✗ |
| `pool staging`（cache） | 7.7% | 9.8% | 1.27 | ~ |
| `attn` + `linear_attn` | 3.9% | 2.8% | 0.72 | ~ |
| `(other)` | 11.4% | 12.1% | 1.06 | ✓ |
| `norm` 家族 | 7.7% | 7.5% | 0.97 | ✓ |
| `dense_elem`（ffn_） | 4.3% | 3.8% | 0.88 | ✓ |
| `ffn_moe_` combine | 7.6% | 6.7% | 0.88 | ✓ |
| **`node`（無名真運算）** | 24.1% | 20.3% | 0.84 | ✓ |

**機制是結構性的**：權重是 `dur * cnt_k / Σ(同 buffer 的會編碼節點)`，所以**同一個 buffer 裡有誰**
決定了分配，而 buffer 的組成隨寬度改變 ⇒ 換個切片寬度就會重新排序。
（與 `eng-bound-0008` 的離線最小平方不同：那裡是**不可辨識**，這裡是**權重本身是代理量**。）

**⇒ 交付規則**：家族總和可引用；逐項只在兩端差 <1.3× 時才講成排名；否則報 `[粗, 細]` 區間。

---

## 6. 排名（步配對：家族 ms ÷ **同一步**的 `wait`）

分母用**同一步**的 `wait`（不是兩個中位數相除）—— 否則會把 prefill／冷池步與穩態步混在一起
（同一臂的 `wait` 中位數在四次列印之間可以是 191 vs 283 ms）。`wait` 是 CPU 側自旋，
**不受** `seg_busy` 的膨脹影響。

| 家族 | NOMINAL 粗 | HEAVY 粗 | HEAVY 細 | 排名 |
|---|---|---|---|---|
| **`node`（無名真運算：FLASH_ATTN_EXT / GATED_DELTA_NET / ROPE / ADD / MUL_MAT / MUL / GET_ROWS）** | **28.9%** | **24.1%** | **20.3%** | **1** |
| **`(other)`（未映射：空基底名 + `Kcur/Vcur` + 雜項）** | 14.0% | 11.4% | 12.1% | **2** |
| **`cache`（池自己的暫存：CPY / SCALE / SET_ROWS / views）** | 9.4% | 7.7% | 9.8% | **3** |
| `norm` 家族（RMS_NORM / attn_norm / attn_post_norm） | 9.4% | 7.7% | 7.5% | 4 |
| `ffn_moe_`（**MoE 逐元素合併鏈，不含任何矩陣乘**） | 8.8% | 7.6% | 6.7% | 5 |
| `ffn_`（`ffn_swiglu` / `ffn_shexp` / `ffn_out`，稠密逐元素） | 5.0% | 4.3% | 3.8% | 6 |
| `attn_*` ＋ `linear_attn` | 4.7% | 3.9% | 2.8% | 7 |
| **`moe_gemv` ＝ MUL_MAT_ID（`ffn_moe_gate/up/down`）** | 3.8% | 3.2% | **6.3%** | **8** |
| `dense_gemm`（`ffn_gate`/`ffn_up`，MUL_MAT） | 2.5% | 2.2% | 4.2% | 9 |
| `conv/ssm`（SSM_CONV / GET_ROWS / CONCAT） | 2.9% | 2.5% | 0.7% | 10 |
| `moe_select`（ARGSORT / SOFT_MAX / logits / probs） | 1.5% | 1.1% | 0.8% | 11 |
| `moe_topk`（**VIEW，恆 0**） | 0.0% | 0.0% | 0.0% | — |

各臂列印原文（`% of wait`，逐 step）可在 §8 的指令下重生。

### 6.1 ③ 的答案
- **池自己的暫存（`cache`，7.7–9.8%）比整個 MoE 專家 GEMV 家族（3.2–6.3%）大，也大於整個注意力家族（2.8–4.7%）。**
  這與 op 表的 `CPY 7.0% + SCALE 2.0% + SET_ROWS 0.4%` 吻合（兩個不同鍵、不同 run）。
- **注意力不是大戶。** `attn_*` ＋ `linear_attn` 排第 7，**低於** `norm` 家族與 `ffn_moe_`。
- **最大的一塊是「真運算但沒有子系統名」**（`node` 20–29%）。這**不是**一個子系統，
  所以任何進一步的排名都先卡在**命名**上 —— 這是本輪最有行動價值的結論。
- 逐節點普查的形狀（39.9% staging／13.4% matmul）與這裡**不衝突**：
  節點數 ≠ GPU 時間，而 `wcntw` 正是把 no-op 節點從分母拿掉之後的樣子。

### 6.2 這些桶的**已證成員**（一個 4116 節點的圖，逐 op 數；這是「桶名不是操作」的解毒劑）

| 桶 | 節點 | 會編碼 | op 組成（每層約） |
|---|---|---|---|
| `ffn_moe_` | 680 | **280** | VIEW 320、RESHAPE 80，**會編碼的只有** GET_ROWS 40、GLU 40、SUM_ROWS 40、CLAMP 40、DIV 40、MUL 40（＝每層各 1） ⇒ **一個矩陣乘都沒有**；`ffn_moe_weighted-<層>` ＝ **1 個 MUL `ne=[2048,8]` ＋ 它自己的 8 個 VIEW** |
| `ffn_` | 160 | **160** | GLU 40、MUL_MAT 40、MUL 40、ADD 40 ⇒ 每層 1 組稠密 FFN（共享專家）鏈 |
| `cache` | 660 | **290** | VIEW 290、RESHAPE 60、PERMUTE 20（staging）＋ **CPY 210、SCALE 60、SET_ROWS 20**（真工作）⇒ **每層約 5 個 CPY ＋ 1.5 個 SCALE**，全部是池自己的搬運 |
| `node` | 620 | **620（零 no-op）** | ADD 270、MUL_MAT 130、GET_ROWS 90、MUL 60、GATED_DELTA_NET 30、UNARY 30、FLASH_ATTN_EXT 10 |
| `(other)` | 883 | 391 | RESHAPE 280、VIEW 172、UNARY 110、**MUL_MAT 100**、L2_NORM 60、MUL 51、ADD 40、TRANSPOSE 30 |
| `norm` | 131 | **131** | RMS_NORM 131（每層約 3.2 個） |

兩件事因此變成事實而不是推測：
- **`ffn_moe_` 那 7.6% 裡沒有任何專家 GEMV** —— 它是逐元素合併鏈。所以「MoE 的**逐元素合併**
  （7.6%）比 MoE 的**專家矩陣乘**（3.2–6.3%）還大」。
- **`node` 桶一個 no-op 都沒有**（`ffn_moe_` 是 680 裡只有 280 會編碼、`cache` 是 660 裡只有 290）
  ⇒ 這正是為什麼 work-weighting 會把 `node` **抬高**（`cntw` 14.9% → `wcntw` 23.9%），
  而不是像 `ffn_moe_gate` 那樣把它壓低。**它原本就被低估。**

### 6.3 M3 判準：**第三個獨立方法，仍然不成立**
判準（`docs/M3_VERDICT_2026-09-17.md`）：「**MoE 專家矩陣乘 ≥ `wait` 的 40%** ⇒ Cell 2 有量」。
本輪讀數：**MUL_MAT_ID ＝ 3.2–6.3% of `wait`**（粗／細；v1 粗 3.8%）⇒ **差約 6 倍**。

| 方法 | 讀數 | 依賴什麼 |
|---|---|---|
| 聯集上界（`M3_MOE_SHARE`） | ≤ 28.3% of bus | 只依賴 buffer 邊界 |
| op 表的 work-weighted 份額 | 8.0 / 8.4% | 兩個細粒度臂 |
| **名字表的 work-weighted 份額（本輪）** | **3.2–6.3%** | 跨表驗證過（§3.2） |

三個方法都不足以支持 40% ⇒ **Cell 2（MoE gather 融合／batched-union）沒有量**。
⚠️ 三個方法的讀數不同（3.2 vs 8.4%），差異的來源已定位為**粒度**（§5.3），
不是矛盾 —— 這正是本輪必須交付的限制。

---

## 7. 怎麼用這張表，以及不要怎麼用

**可以：**
- 讀**家族總和**的排名（§6 的表），並附上粒度區間。
- 把 `wcntw` 與 `ub` 一起看：`cntw` 小而 `ub` 大的桶就是被稀釋的桶（`ffn_moe_gate` 舊況）。
- 用 `residual` 判斷這一步的數字能不能引用（HEAVY 下 24%）。

**不要：**
- **不要只看欄位名不看單位**：列印格式是 `wcntw=` 接**毫秒**再接**百分比**（例：`wcntw=  24.60   8.5%`），
  兩張表都是。本輪寫核對腳本時把「%」那一欄當成 ms 又除了一次 `wait` ⇒ 整張表規模錯誤，
  而**它看起來完全合理**（家族的排序都對）。用 `/tmp/en_step_paired.py`（它只讀 ms 欄）而不是自己重寫。
- **不要把 `sum(wcntw)` 當步時長**。它是 `seg_busy` 的一部分（76–83%），而 `seg_busy`
  本身隨切片數膨脹（同一 workload：143 / 296 / 766 ms @ n_cb = 8 / 16 / 63）。
- **不要逐項引用百分比**，除非它在 §5.3 的兩端差 <1.3×。
- **不要把 `node` 或 `(other)` 當子系統**：它們是「名字沒有子系統資訊」的證明，
  不是兩個模組。
- **不要用 t/s**：本輪兩臂首發熱態是 **HEAVY(2)**（`en-work` 4.73、`en-work-fine` 3.39 t/s 不可引用）。

---

## 8. 重現

```sh
# 兩個臂（粗＝預設粒度、細＝2–7 節點切片）
RUN_REPLAY_BENCH=0 python3 scripts/check/decode_sweep.py --profile prefill250 \
  --arms en-work,en-work-fine --rounds 2 --warmup 1 --n-predict 16 \
  --json Backup/phase_decomp/en_work2_20260918.json --force

# 名字表 + op 表（同一次跑兩者都有）
L=$(ls -t Backup/cgc_logs/llama_server_20260918_*.log | head -1)
grep -h "CGC-GPUNODE\|CGC-GPUOPS" "$L"

# 步配對的家族表（家族 ms ÷ 同一步的 wait）
python3 /tmp/en_step_paired.py "$L"
python3 /tmp/en_namework_family.py <粗 log> <細 log>    # 含跨表 ms-ratio 驗證
python3 /tmp/en_namework_parse.py "$L"                  # 逐項排名 + *bywork 次序
```

產物：`Backup/phase_decomp/en_work_20260918.json`（v1，NOMINAL）、`en_work2_20260918.json`（v2，HEAVY）。
build 指紋：`libggml-base 1940ab03 → 473ebf60 → e1ea9595`；`libggml-metal` **未動**（本輪只改消費端）。

---

## 9. 誠實邊界

1. **樣本**：每個臂只有 **4 個 trunk 步**（`seg_busy ≥ 50 ms`、`ntok ∈ {2,4}`、`--n-predict 16`）。
   表格印在「每 8 個合格 graph_compute」⇒ 一個短 run 只有 3–4 次列印。數字是**中位數**，
   但 n=4 的區間很寬（§6 的逐 step 值可見）。
2. **只驗到一個 profile**：`prefill250` + 生產池。沒有驗其他池尺寸。
3. **`(other)` 的成員只列到 dump 層級**（空基底名 204 個、`Kcur-*`/`Vcur-*`），
   沒有逐一歸因到子系統；`node` 的 620 個自動名同理。
4. **詞表仍是手寫前綴**（40 條），所以「桶」的邊界是**詞表的產物**，不是語意的產物
   （lesson `eng-lf-0008`）。本輪再次撞到：`ffn_moe_topk`＝VIEW。
5. 本檔的排名**沒有**回答「哪一塊可壓縮」—— 它只回答「誰佔了多少」。`node` 那 20–29%
   要能動，先要能命名它。
6. **★ 權重把「每一個會編碼的節點」當成等成本。** 它修掉的是 **no-op 稀釋**（由編碼器原始碼證明），
   **不是**「貴的節點 vs 便宜的節點」。所以它是**共同出現（co-location）的排名，不是成本排名**：
   一層裡那個 `ffn_moe_weighted` 的 MUL（只有 `ne=[2048,8]`）與同一格裡的 `MUL_MAT_ID`
   拿到**一樣的份額**。這正好解釋 §5.3 為什麼**細粒度會把兩個 GEMM 家族抬高約 1.9×**
   （切片內的同儕更同質）⇒ **真值在粗與細之間，而且偏向細的那一端**。
   要真的拿到成本，需要的是節點級的時間，而那條路（每節點一個 command buffer）已被
   Metal 的在途 command buffer 配額擋死（`n_cb ≥ 64` 在載入期死鎖，
   見 `docs/M3_MOE_SHARE_2026-09-18.md` 與 lesson `eng-diag-0035`）。
