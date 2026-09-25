# 第 3 步開工的阻塞：第 2 步的程式碼不在樹裡（2026-09-26）

**日期** 2026-09-26 01:0x · 全程**零 GPU、零建置**（只做 grep／`strings`／讀檔）· 上游
`docs/NEXT_STEP_MISS_HANDLER_2026-09-24.md` §4、`docs/MISS_MASK_STEP2_2026-09-24.md`、
`docs/MISS_GATE_STEP3_2026-09-24.md`

## 0. 一句話

`docs/MISS_MASK_STEP2_2026-09-24.md` 與 `MISS_GATE_STEP3` 都寫著「第 2 步已完成、第 3 步閘門
OPEN」，但 **`CGC_MISS_MASK` 這段程式碼從來沒有進入版控，而且現在也不在工作樹裡** ⇒
**第 3 步今天不能開工**。要先照 §1 的規格把第 2 步重建回來並重新驗一次正確性，才能碰第 3 步。

⚠ 這同時修正我 2026-09-26 00:5x 給 operator 的那句「第 2 步 miss mask 已完成」——
**那是文件上的結論，不是樹裡的事實**。

## 1. 三條獨立的查證（任一條單獨都足以定案）

| # | 查證 | 指令 | 結果 |
|---|---|---|---|
| 1 | 原始碼樹 | `grep -rn "CGC_MISS_MASK\|MISSMASK\|valid_table\|vmask\|cache_missmask_tensors\|ffn_moe_ids_cont" src/` | **0 命中** |
| 2 | 現行 binary | `strings -a <bin> \| grep -c "CGC_MISS_MASK\|MISSMASK"`，掃 `src/llama.cpp/build/bin/` 全部 `*.dylib` ＋ `llama-server` ＋ `llama-bench`（**53 個檔**） | **全部 0** |
| 3 | 版控 | `git grep -l "CGC_MISS_MASK" HEAD --` | 只有 5 個 **docs**（`DIAGNOSTIC_ARMS_LEDGER`、`MISS_GATE_STEP3`、`MISS_MASK_STEP2`、`mindmap/TAXONOMY`、`mindmap/briefs/index.html`）⇒ **src 從未進版控** |

補充：`git log --all -S "CGC_MISS_MASK"` 只回一支 `65c76b8c7`（mindmap 那支，且它本身
**D5 未通過、owner 重新基線前勿引用**），而 `git grep` 在該 commit 上同樣只有 docs。

⇒ **推測的成因（未證實，但與既有記錄一致）**：第 2 步是在工作樹上建起來量測的，沒有 commit；
之後被別條 session 的重建／checkout 蓋掉。這正是 `.workbuddy/MEMORY.md` 已記的兩個共用面：
「多個 session 同時寫同一個 repo」＋「建置產物是共用資源」。
⇒ **本輪的教訓要落到流程**：第 2 步重建後**當天就要 commit**，不要留成工作樹狀態。

## 2. 第 2 步的重建規格（照 `MISS_MASK_STEP2` §1 原文，掛點已對到今天的樹）

| 檔案 | 要加什麼 | 今天的掛點（file:line） |
|---|---|---|
| `src/llama.cpp/src/llama-graph.cpp` | 在 `cgc_slot_table_gpu` 分支裡，用**同一個** index vector `ids_flat` 再掛一次 gather：`vmask = ggml_get_rows(valid_table, ids_flat)`；並把 cont 張量命名為 `ffn_moe_ids_cont`，讓 host 能抓下來 | 分支在 **:2240**（`if (cgc_slot_table_gpu && il >= cgc_s1_min_il)`）；既有的 slot gather 在 **:2353** `ggml_tensor * slots = ggml_get_rows(ctx0, slot_table, ids_flat);` ⇒ **緊貼它加一行** |
| `src/llama.cpp/src/llama-context.h` | 三個新的 per-layer 捕獲表：`cache_valid_tensors` / `cache_missmask_tensors` / `cache_ids_cont_tensors` | 既有表區在 **:445–458**（`cache_down_out_tensors`／`cache_rn_mask_tensors`／`cache_probs_tensors`／`cache_ffn_tensors`）⇒ 加在這一區 |
| `src/llama.cpp/src/llama-context.cpp` | (a) 按名字捕獲上面三個節點；(b) `graph_compute` 裡、**dispatch 之前**、每步一次把 `llama_expert_cache_slot_table()` 快照寫進 `valid_table`（`valid[e] = st[e] >= 0 ? 1 : 0`）；(c) `CGC_MISS_MASK_DBG` 下一次讀回並印 `MISSMASK` 行 | (a) 捕獲區 **:7673–7763**（`strcmp(name,"ffn_moe_rn_mask")` 等既有樣式）；(b) **:3597–3659** 的 `CGC_B_SCHEME` 區塊就是「dispatch 前寫 leaf」的現成範本（RN mask 版在 **:3492–3544**）；(c) `CGC-RN-MASK:` 印出點在 **:3533/3540** |

兩個 env 要**分開**（原設計，不要合併）：

| env | 作用 | 代價 |
|---|---|---|
| `CGC_MISS_MASK=1` | 建節點 ＋ 每步發布 valid_table（**要定價的那一個**） | 2 節點/層 × 39 層 |
| `CGC_MISS_MASK_DBG=1` | 再一次 `synchronize` ＋ 讀回 ＋ 印。**僅診斷，禁止拿來報吞吐量** | 1 次同步/步 |

為何是 gather 不是別的（原文件的理由，照抄以免重蹈）：residency 在 host，但「選了哪些 expert」
只在 device（`CGC_SLOT_TABLE_GPU` 的全部意義）⇒ 交集只能在 device 上取。且**不可**把
`slot_table` 加寬成 `[2, n_expert]`:`mul_mat_id` 必須吃 `[k, n_tokens]` 的 id 張量，切回去需要
strided view，那個結構已經戳過兩次 `ggml-alloc.c:623 GGML_ASSERT(buffer_id >= 0)`。

### 重建後的驗收（跑前凍結，同 `MISS_MASK_STEP2` §2）

- **正確性**：與 host `LLAMA_EXPERT_CACHE_BATCH_DBG` 逐位相同 ⇒ `SET_DIFF=0` 且 `LEN_DIFF=0`
  （上次是 38 層／421 元素／0 差異，工具 `scripts/check/miss_mask_check.py --self-test` 22/22）。
- **成本**：門檻 ≤0.2 ms/step；上次量到的是 **UNRESOLVED（CI 半寬 ±2.01 ms，是門檻的 10 倍）**，
  所以結論要寫成「**≤ 解析度；結構上無新增 CB（`CGC-DECPROF` `segs=41` 兩臂相同）／無新增同步**」，
  不是「≤0.2 ms」。
- ⚠ 方法論坑（上次踩過，別再踩）：**「某個 step 可能一行都不印」的 log 不能用單調性切 step**，
  要逐層序列比對。

## 3. 第 3 步（per-expert 重算）——設計與插入點

### 3.1 數學前提（這是它比整層重算便宜 8 倍的原因）

```
y = Σ_i  g_i · down_i( up_i(x) )        i ∈ top-k
```
缺席 expert 被指到佔位 slot ⇒ 那一項 ≈ 0，**其餘 k−1 項是對的** ⇒ 只需補算缺席那一項再加回。

### 3.2 ⛔ 但這個前提「現在」不成立 —— B 臂的佔位是別人的權重，不是 0

| 位置 | 事實 |
|---|---|
| `src/llama.cpp/src/llama-context.cpp:3607–3634`（`CGC_B_SCHEME` 區塊） | 缺席 expert 被指到 `v = (uint32_t) e % ns` ⇒ **指向另一個 expert 的權重**，不是 0 ⇒ 那一項不為 0，輸出是 garbage 而不是「少一項」 |
| `src/llama.cpp/src/llama-expert-cache.cpp:371–408`（`llama_expert_cache_publish_slot_table`） | 正確的佔位：`else if (zs >= 0) dst[e] = zs + cgc_s1_tag;`（`zs` = 零 slot） |
| `llama-expert-cache.cpp:277–282` / `:292–316` | 零 slot = 該層**最後一個** slot（`slots_l-1`），並把 4 種權重 `memset` 成 0，每層一次 |
| ⚠ `llama-expert-cache.cpp:268–270`（`zero_slot_enabled()`） | 零 slot **只有在 `CGC_VERIFY_DECODE` 或 `CGC_DRAFT_DECODE` 有設時才啟用** ⇒ 第 3 步必須先把它在 B 臂上打開（或放寬這個 gate） |

⇒ **第 3 步的第一個 commit 不是 kernel，是把 `CGC_B_SCHEME` 的 `e % ns` 換成 zero slot。**
這一步做完，「缺席那一項 ≈ 0」才成立，補算才有意義。

### 3.3 補算要用的東西（都在 `llm_graph_context::build_moe_ffn`，`llama-graph.cpp:1964`／重載 :1920）

| 東西 | 位置 |
|---|---|
| 層輸入 `x` | 函式參數 `cur`（上層傳進來的是 `ffn_norm` 輸出，`models/qwen3moe.cpp:131–137`）；**:2538** `cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);` ⇒ 補算的輸入要是這個 3D 形狀 |
| `g_i` | **:2499** `weights = ggml_get_rows(ctx0, probs, selected_experts);` |
| expert → slot | **:2353** `slots = ggml_get_rows(ctx0, slot_table, ids_flat);` ⇒ **:2376** `remap_ids = slots;` ⇒ **:2505** `mm_id_ids = remap_ids`（權重本身不用 `get_rows`，`mul_mat_id` 在 kernel 內按 row 索引池） |
| up / gate / down | **:2572** `up = build_lora_mm_id(up_exps, cur, mm_id_ids, up_exps_s)`；**:2585** gate；**:2767** down（gate_up 合併在 :2552） |
| 活化 | **:2631** `ggml_swiglu_split` |
| 乘權重 | **:2781** `experts = ggml_mul(ctx0, experts, weights);` |
| 層輸出 | **:2880** `cb(moe_out, "ffn_moe_out", il);` ／ caller `qwen3moe.cpp:152–155` 做殘差 ⇒ **`y += 補算項` 最自然的掛點是 :2880 之前，或 caller 的 `ffn_inp` add** |

可複用的最小單位：**`llm_graph_context::build_lora_mm_id(w, cur, ids, w_s)`（`llama-graph.cpp:1567`）**
= 一行 `ggml_mul_mat_id` ＋ 可選 scale ⇒ 補算就是對缺席 expert j 造一個常數 id 張量（填它的
**slot index**，不是 expert id），依序 `build_lora_mm_id(up_exps, x, ids_j)` → gate → swiglu →
`build_lora_mm_id(down_exps, ..., ids_j)` → `ggml_mul(..., g_j)` → 加回 `moe_out`。

⚠ 兩個會咬人的細節：
- `ids` 必須是 **slot index**（`ggml_mul_mat_id` 按 row 索引池），不是 expert id。
- `cur` 必須是 `[n_embd, 1, n_tokens]` 的 3D（照 :2538 reshape），不能直接餵 2D。
- 有 `CGC_DOWN_COMBINE` 融合快路徑（**:2678–2765**），命中時 :2764 直接 return，不會走到 :2767
  ⇒ 補算要考慮這條路徑，**否則只在沒開融合時正確**。
- 「把某個 expert 的 FFN 單獨建一次」的現成函式：**找不到**（已查 `recomp`/`repair`/`fixup`/
  `deferred`/`lazy` 全 repo，無關）⇒ 要自己寫，且沒有重複的輪子可抄。

### 3.4 為什麼單段時 hook 不跑（這是「為什麼要補算」的根因，不是 bug）

- `sched->callback_eval` 只在 `hook_seg`（`ggml-backend.cpp:2568–2574`）被呼叫；
- `CGC_SEG_BATCH` 在 `ggml-backend.cpp:1780–1788` 一次送整張圖後 `synchronize` 就 **return**
  ⇒ `n_segs`/`as_topk` 都還沒算出來，41 段主迴圈（**:2682–2700**）、hook、`CGC-DECPROF`
  印出（**:2702–2758**）全部跳過 ⇒ 這就是 S1 臂上 `DECPROF=0 / HOOK=0` 的機制解釋。
- hook 本身在 `llama-context.cpp:4223–4226` 分派 → `:5220` `expert_cache_on_topk` → `:7144`
  `llama_expert_cache_ensure_batch`。

## 4. 跑前凍結的判準（四條，不過就退回）

| # | 判準 | 門檻 |
|---|---|---|
| 1 | **端到端正確性** | 同 prompt、greedy、**逐 rep 相同**，且 `answer_md5` 等於分段臂（既有 acceptance 判準）。**不過就退回，沒有第二條路** |
| 2 | **落點** | **13.6 ~ 14.2 t/s**；**低於 13.0 退回**（`NEXT_STEP_MISS_HANDLER` §4 原判準，不改） |
| 3 | **穩態 miss 率閘門**（`MISS_MASK_STEP2` §6，第 2 步重建後**先量這個**） | < 10% ⇒ 補算 3.6~4.5 ms，淨 +9~10 ms，開工；**20%~60% ⇒ 補算 8~24 ms，必須先把 fill 後台化做出來再說** |
| 4 | **mask 自身成本** | 結構判據（`segs=41` 不變、生產路徑無新增 synchronize），不是 ±0.2 ms 的數字 |

⚠ **第 3 條是真正的關卡**：`MISS_MASK_STEP2` §4 在「完全不 fill」的診斷臂上量到
**k=8 裡有 5 個是佔位 ≈ 62%**（39 層 ⇒ ~195 次/步）。若成品穩態退化到那個數字，補算是
**24.6~33 ms**，直接吃掉全部收益。原文件的警告要照抄：
> **per-expert 重算的價格對「成品穩態 miss 率」極度敏感。**

## 5. 不做／風險

- **整層重算**：已判死（`NEXT_STEP_MISS_HANDLER` §0：−6.5~+1.9 ms；用 18.1 層算是 −31.9~−13.0）。
- **錯層數**：定死 **18.1**（不是 8；「8」在資料裡一步都沒出現，min = 9）。它決定「要保留幾層的
  `x`」⇒ **保守做法是全 40 層都留**（hidden × ntok=1 ≈ 320 KB，可忽略）。
- **fill 後台化**：不是錦上添花（`NEXT_STEP_MISS_HANDLER` §6.4 已作廢那句）。下界
  （fill 仍同步）**14.3 t/s**、上界（完全後台化）**19.1 t/s** ⇒ **補算與後台化要一起做，
  先做哪個都只能拿一半**。這兩個數是**推算**，`34.5 ms` 的內部分解（gap 13.4／cb 3.6／
  41 段 submit ~8.3／fill 等待 ~9.2）**各項沒有獨立量測支撐**，不要當交付值。
- **`CGC_MISS_MASK_DBG` 的同步**：只准出現在診斷臂，禁止拿它報吞吐量。
- **MTP verify（ntok=4）路徑**：miss mask 只在 ntok=1 的 decode 圖上建過（`cgc_is_decode_graph`）
  ⇒ **spec 開著時尚未驗證**，不要外推。

## 6. 開工前還要決定的兩件事（不是技術問題）

1. **owner**：本線＝`線A (ace)`（S1／段邊界／逐層 KIND×OP 儀器）；freebuff＝GPU 實驗＋量測工具。
   **per-expert 重算 kernel 是引擎碼（`src/llama.cpp/src/`），歸屬未明文** ⇒ 開工前先定，
   否則會撞 09-20 那五個碰撞面（`llama-context.cpp`、`ggml-backend.cpp`、`libggml-base`…）。
2. **建置窗口**：`cmake --build` 是對**所有正在跑的實驗**的一次寫入。本輪 01:0x 查閘門
   （`lsof -nP -iTCP:8080 -sTCP:LISTEN` ＋ `pgrep -fl 'run_ids_dst_capture\|decode_sweep\|llama-server\|restart_rerun\|req2retest\|llama-bench\|run_mmap_ab'`）
   **皆為空 ⇒ 可建置**；但 `Backup/rerun` 在 00:48 才被動過 ⇒ 別條線隨時可能回來
   ⇒ **動手前要重跑同一個閘門，且「檢查與動作必須在同一個分支裡」**。

## 7. 建議順序（下一步）

1. **先 commit 一段流程修正**：本文件入版控（它記錄了「第 2 步的程式碼是怎麼不見的」）。
2. **第 2 步重建**（§2，三個檔案、兩個 env），**當天 commit**。
3. 重跑 `miss_mask_check.py` 驗 `SET_DIFF=0/LEN_DIFF=0`。
4. **量穩態 miss 率**（判準 3）⇒ 決定要不要先做 fill 後台化。
5. **第 3 步第一個 commit**：`CGC_B_SCHEME` 的 `e % ns` → zero slot（§3.2）。
6. 才做補算 kernel（§3.3）。

## 8. 2026-09-26 01:2x：第 2 步已重建並進入版控

§1 的阻塞已解除。三個檔案、六個掛點（本次實測的行號）：

| 檔案 | 掛點 | 內容 |
|---|---|---|
| `src/llama.cpp/src/llama-graph.cpp` | :2174 | `static const bool cgc_miss_mask = getenv("CGC_MISS_MASK")` |
| | :2378 | `cb(ids_cont, "ffn_moe_ids_cont", il)`（既有張量，只加名字） |
| | :2414–2428 | `if (cgc_miss_mask) { valid_table [1,n_expert] I32 + vmask = get_rows(valid_table, ids_flat) }` |
| `src/llama.cpp/src/llama-context.h` | :467–469 | `cache_valid_tensors` / `cache_missmask_tensors` / `cache_ids_cont_tensors` |
| `src/llama.cpp/src/llama-context.cpp` | :7802–7813 | 按名字捕獲上面三個節點 |
| | :3672–3708 | publish：`valid[e] = st[e] >= 0 ? 1 : 0`，在 `alloc_graph`(:1956) **之後**、`graph_compute_async` **之前** |
| | :3718–3790 | `CGC_MISS_MASK_DBG`：一次 `synchronize` ＋ 讀回 ＋ 印 `MISSMASK`／`CGC-MISSMASK-STEP` |

**binary 證據（「有沒有編進去」的唯一判準）**：`strings -a src/llama.cpp/build/bin/libllama.dylib`
命中 `CGC_MISS_MASK` **3**、`CGC_MISS_MASK_DBG` **2**、`ffn_moe_valid`／`ffn_moe_missmask`／
`ffn_moe_ids_cont` 各 **1**。重建前（§1）這五個字串在 53 個 binary 裡**全部為 0**。

### 與 09-24 原設計的兩點差異（都刻意，都加嚴不加寬）

1. **publish 與 DBG 都加了 `cgc_node_in_graph(gf, t)` ＋ `cgc_is_i32_n(t, n)` 兩道守衛。**
   原設計沒有。理由是 mask 節點**只在 decode 圖上建**（`cgc_is_decode_graph`），而捕獲表是
   **跨圖存活**的 ⇒ prefill 圖上那些 stale 捕獲會被寫進一個已被 alloc 回收的指標。這正是 S1
   readback 在 :3737 學到的同一個坑（arena 每次 build 都重用，只看位址會接受全部 39 筆舊值）。
2. **`CGC_MISS_MASK_DBG` 少了 `CGC_MISS_MASK` 時會印一行警告就什麼都不做**，不是靜默讀出全 0。
   全 0 的 mask 會被 `miss_mask_check.py` 讀成「每一步每個 expert 都 miss」，那是一個看起來
   完全合理的假結果。

### 還沒做（下一棒）

- **正確性複驗**：`scripts/check/miss_mask_check.py` 的 `SET_DIFF=0`／`LEN_DIFF=0`。需要一個 GPU
  空窗跑 `CGC_SLOT_TABLE_GPU=1 CGC_MISS_MASK=1 CGC_MISS_MASK_DBG=1 LLAMA_EXPERT_CACHE_BATCH_DBG=1`
  的臂。本次沒跑（01:2x 的 D5 用的是生產 profile，那個臂不帶這些 env）。
- **穩態 miss 率**（§4 判準 3）：第 2 步重建後這才是第一個該量的數。
- **第 3 步仍然沒有 owner 明文**（§6-1）。
