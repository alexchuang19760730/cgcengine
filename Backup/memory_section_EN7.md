
### §EN-7 §9.18.6 做完了：**儀器先用一個對照證明自己，然後推翻了 §9.18.4 的結論**（2026-09-16 19:43–19:55）

**(A) 儀器**：`CGC_TENSOR_CAPTURE=<節點名>`（＋`_WORDS`，預設 32）把一個節點的**輸出張量**快照進
第二個 destination。實作是**重用**既有的 `kernel_cgc_ids_capture`（它本來就是通用的「把 stride 個
int32 從某個 buffer 抄進 slot」內核，與 ids 無關）⇒ 只加了 `cgc_submit()` 這個共用提交體、
第二組 buffer/cursor、以及一個 `seq` 合併；`.metal`、`impl.h`、`device.*`、`context.m` **一行都沒改**。
完整比對器 `scripts/check/ids_capture_diff.py` **原封不動可用**。
三個設計要點（都寫在原始碼註解裡）：精確名稱比對（substring 會命中 `-10..-19`）；
獨立的 slot 空間（ids 那條已經正好在 4096 上限）；**依提交序合併兩條流**（否則 dst 列會被歸進最後一個
偽 graph，比較器會**靜默報 IDENTICAL**——這正是最該避免的假陰性）。

**(B) 一個用對照抓到的假陽性（本次最重要的方法學結果）**：
第一版沒有屏障。**同臂**對照（同一臂跑兩次、互比）顯示：**ids 120 SAME / 0 DIFF，而 `ffn_moe_down-1.dst`
每個 graph 都 DIFF**。跨臂跑則顯示「ids 相同、輸出不同」——**看起來正是 §9.18.4 假說的證實**。
機制：ids 運算元的生產者在很多節點之前（早已沉降），而 dst 的生產者是**緊鄰的前一個 kernel**，
這個 fork 支援內核併發（`ggml_metal_op_concurrency_reset` → `ggml_metal_encoder_memory_barrier`），
所以 copy 可能讀到該 buffer 的**前一位佔用者**。

修法＝在診斷路徑上插一個 `ggml_metal_encoder_memory_barrier(ctx->enc)`（屏障不改變任何數值，
只把診斷 copy 排在生產者之後）。**修完後同臂對照每個 graph 都 SAME**（兩個獨立的同臂對照都如此）。

⇒ **規則：任何新儀器在跨臂 diff 之前，必須先做「同臂兩跑、互比」的對照。** 沒有它，
不可重現的讀數會產生一個**看起來很確定的假陽性**，而且方向剛好會迎合你正在測的假說。

**(C) 有了屏障之後的跨臂結果（兩次獨立重跑給出相同結論）**：

| graph | ids SAME | ids DIFF | `ffn_moe_down-1.dst` |
|---|---|---|---|
| 1–3 | 120 | 0 | SAME |
| **4–30** | **6** | **114** | **SAME** |
| 31–35 | 0 | 120 | DIFF |

而 graph 4/5 的 **6 個 SAME 節點正好是 layer 0 與 layer 1 的 gate/up/down**：

```
SAME = [ffn_moe_down-0, ffn_moe_down-1, ffn_moe_gate-0, ffn_moe_gate-1, ffn_moe_up-0, ffn_moe_up-1]
DIFF = layers 2..39（114 個）
```

**(D) 結論與更正**：
- **layer 0 的 router ids 相同 ⇒ 兩臂處理的是同一個 token**（排除「token 流分歧」的解釋）。
- **layer 1 的 ids 與它的 MoE 輸出（`ffn_moe_down-1.dst`）都逐位元相同。**
- ⇒ **§9.18.4 的「缺陷在 ids 指向的權重內容（池／slot 內容）」被直接量測推翻**：
  它是由排除法推出的，而那個推理的前提（layer 1 的 MoE 輸出不同）**是錯的**。
  連帶 **§9.18.7 的「要修的是 residency」失去唯一的直接證據**，應降級為「未受支持」。
- **新的定位**：分歧出現在 **layer 1 的 MoE 輸出之後、layer 2 的 router 之前**。
  由於 layer 1 的 MoE 輸出相同即意味 layer 1 的 attention 輸出相同（否則 MoE 不會逐位元相同），
  ⇒ 候選是 **layer 2 的 attention 路徑（q/k/v、rope、KV 讀取）或 layer 1→2 之間的 norm／殘差**，
  **不是專家池的權重內容**。

**(E) 儀器的已知限制（不假裝沒有）**：`cgc_dst_capture()` 目前只掛在
`ggml_metal_op_mul_mat_id` 裡 ⇒ **只能擷取 mul_mat_id 節點的輸出**（gate/up/down）。
要擷取 attention 的輸出需要掛在別的 dispatcher 上——那是下一個 probé 的前提，
也正好是 (D) 指出的方向。
