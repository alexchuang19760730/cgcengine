
### §EN-8 §9.18.6 第二輪：擷取儀器擴到 5 個 dispatcher，並把分歧定位到 **layer 2 的 gated delta-net 核心**（2026-09-16 20:12–20:30）

#### 儀器（`src/.../ggml-metal-ops.cpp`，`src/` 有改 ⇒ 重建 + check 8 + D5）

1. **`CGC_TENSOR_CAPTURE` 改成逗號分隔清單**（或 `*`），每個 token **精確比對**。`dst_filter` 64→256、`CGC_DST_SLOTS` 1024→**4096**，且**槽位耗盡會印一次 WARN**（原本是靜默截斷＝安靜的假陰性）。
2. **新增 4 個 dispatcher 尾端擷取點**：`mul_mat`、`flash_attn_ext`、`bin`（殘差 add）、`norm`（rms_norm）。加上既有的 `mul_mat_id` 共 **5 個**。
   **命名規則（這一輪真正的技術發現）**：dispatcher 尾端**不能**用 `node(idx)` 命名——可融合的 dispatcher 會把 `bid_dst` 指向**融合群的最後一個節點**（`ops.cpp:5169-5170` 的 `bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1))`）。所以規則固定為 **`node(idx + n_fuse - 1)`**；不可融合的（`mul_mat`、`flash_attn_ext` 都是 `return 1;`）等價於 `node(idx)`。`fuse` 也寫進列尾（在 `ids=[...]` 之後，讀者的 regex 不受影響）。
   **觀測者成本**：match 測試在 barrier **之前**，所以只有在名單內且出現的節點各付一次 barrier ⇒ 名單刻意短（11 個）。
3. **先列舉再測**：`CGC_TENSOR_CAPTURE='*'` 跑一次就把 1021 個節點名與 fuse 值列出來。**我先前猜的 `ffn_out-*` 與 `post_moe-*` 根本不存在**——真名是 `ffn_moe_out-*` 與 `l_out-*`（`build_cvec` 把殘差重新命名）。**猜名字的代價是一整輪跑**，列舉只要一次。
   **`node_NNN` 不可跨臂比對**：匿名節點的名字來自 ggml 的計數器，而兩臂的圖不同（S1 每層多 4 個節點）⇒ 同名不同物。只能用具名節點。

#### 三個儀器事實（都會誤導讀者，必須記）

- **發射順序不是計算順序，而且不穩定**：同一個臂跑兩次，**同樣的值**卻以**不同順序**送出（`linear_attn_out-2` 有時在 `attn_residual-2` 之前、有時在之後）。⇒ 任何「哪一列先出現＝哪個節點先算」的推論都是讀雜訊。分析器改成用**原始碼的層鏈順序**（`CHAIN`）排序。
- **`ABSENT` ≠ `SAME`**：節點只有在剛好是融合群的最後一個節點時才被擷取，而**融合狀態取決於節點順序**（本身不穩定）。實際後果：`norm-2` 在 **S1 臂完全不見**（41 次全滅）而在 baseline 臂出現 41 次 ⇒ **該節點跨臂不可比**。分析器現在把 `ABSENT`（配合 `present=N`）與 `never` 分開印，永不把缺席報成相同。
- **尾段 graph 邊界不可信**：ids 目的地上限 4096 而一個 forward pass 約吃 114 列 ⇒ **尾端幾個 chunk 沒有 graph 標記、把好幾個 pass 併在一起**，按名字配對就變成「拿不同 pass 的列相比」。⇒ 一律用 **`--upto 24`（只採 graph 1..23）**。這也解釋了為什麼 unrestricted 跑會在 graph 27/30/31 冒出假訊號（`gate-2` 在同臂對照裡於 graph 27 出差異）。

#### 結果（同臂對照先過：兩個臂各跑兩次、共 23 graph × 11 節點 × 2 對 = **零差異**）

兩個獨立跨臂對（`p25-gputime` vs `p25-slotgpu`）**逐項相同**，只採 graph 1..23：

| 順序 | 節點 | 結果 |
|---|---|---|
| 1 | `l_out-0`（layer 0 輸出） | **SAME**（24/24） |
| 2 | `l_out-1`（layer 1 輸出） | **SAME**（23/23） |
| 3 | `attn_norm-2`（layer 2 attention 的**輸入**） | **SAME** |
| 4 | `z-2`（layer 2 的 z 投影，mul_mat） | **SAME** |
| 5 | `gate-2`（layer 2 的 alpha/gate 投影，mul_mat+mul） | **SAME** |
| 6 | `norm-2`（gated norm，核心之後） | **ABSENT**（跨臂不可比） |
| 7 | **`linear_attn_out-2`（layer 2 attention 的輸出投影）** | **DIFF，自 graph 4 起** |
| 8–11 | `attn_residual-2` / `attn_post_norm-2` / `ffn_moe_logits_raw-2` / `l_out-2` | DIFF@4（**下游，已被 7 解釋**） |

**⇒ 定位：分歧發生在 layer 2 的 `z-2`/`gate-2` 之後、`linear_attn_out-2` 之前，也就是 layer 2 的 gated delta-net 核心（conv1d／delta-rule scan／state 路徑）。**
**layer 2 的密集矩陣全部逐位元相同**（輸入、z、gate 三個都同）⇒ 這一層的**沒有任何 matmul 是分歧點**。

#### 這一輪推翻了什麼

- 上一輪的候選清單寫「layer 2 的 attention 路徑（q/k/v、rope、KV）**或 layer 1→2 之間的 norm／殘差**」——**後者被排除**：`attn_norm-2`（＝ layer 1 輸出經過 attn_norm 後的值）逐位元相同，而 layer 1 的輸出本身也相同。
- §9.18.4「載體是池／slot 的權重內容」其唯一證據早已被上一輪推翻；這一輪**再排除它在 layer 2 的 attention 上的變體**：layer 2 的所有 dense matmul 都相同。
- **但要小心不過度外推**：這只說「**第一個**分歧在 layer 2 的 delta-net 核心」，沒有說池沒問題。MoE 在那之後才分歧，已被解釋。

#### 下一步（明確、且已被本輪界定）

path 上還缺的 dispatcher：**`ggml_metal_op_ssm_conv`** 與 **`ggml_metal_op_gated_delta_net`**（`gate`/`z` 之後、`norm-2` 之前的那一段）。同時要**修融合造成的 `ABSENT`**（把「融合群最後節點匿名」的情況改成用群首名字 ＋ `+fuse<N>` 的合成名），否則 `norm-2` 永遠只能在一臂看到。

#### 交接

`agent_harness/` 那條線仍在動（我 commit 前一分鐘它還在寫 `lessons.jsonl`）。本輪**只 stage 自己的檔案**。
