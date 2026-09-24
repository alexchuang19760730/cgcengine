# L3 M1 —— 停在哪裡：**交付路徑裡沒有那個 await 點**（三個程式事實 ＋ 一個量測）

日期：2026-09-20　binary：`libllama` md5 `316393918ed651e4`（與 M0 同一顆）
形狀：Nail IQ3_XXS-denseIQ4X、MTP on k=3、pool 8 GiB、`prefill250`、temp 0.4

---

## 0. 一句話

M1 的設計（「await 掛在該層 segment 第一個 consumer 的 eval callback」）**在交付路徑上不存在**：
派送器在分段模式（`CGC_SERVER_OA_ASYNC` 預設 **1**）下**只以 `ask=false` 呼叫 callback，而那是在節點算完之後**。
而「把等待往後挪一格」已經被這個程式庫自己判定為 **racy** —— 我用既有的開關把它量了，結果是 **logits 全 NaN**。
⇒ 這不是工時問題，是**歸屬問題**：L3 的那 44 ms 只能在**派送器的分段提交**裡拿（`ggml-backend.cpp`，線 A 持有）。

---

## 1. 三個程式事實（都可以逐行核對）

**① 分段派送器只在算完之後叫 callback。**
`ggml-backend.cpp:2495 / 2502 / 3015` 全在 `...sched->callback_eval(t, false, ...)` ——
`ask=false` 的三個呼叫點都位於該節點/segment **已計算完**之後。`ask=true`（「這個節點要不要 callback」）
只出現在 `:3061` 的**通用路徑**，而交付 profile 不走那條（`run_server.sh:118 SERVER_OA_ASYNC="${CGC_SERVER_OA_ASYNC:-1}"`，
且 `cgc_oa_async_enabled()` 未設時回 `true`）。
⇒ 計畫 §3.2 的 (b)「await 掛在第一個 consumer 的 eval callback」**沒有可用的掛點**。

**② hook 跑之前，GPU 已經被等乾淨了。**
`hook_seg(i)`（`ggml-backend.cpp:2110`）第一件事就是
`while (cgc_done(split_backend) < done0 + (i+1)*bufs) sched_yield();` —— 等 segment i 的**全部** command buffer 完成
（註解：只等 main buffer 會在 argsort 還在跑時就拿到 stale ids）。所以 fill 開始的那一刻，**GPU 上沒有任何工作在飛**，
`gap` 就是這樣產生的。等待之所以必要，是因為**ids 是 GPU 算出來的**（同一段註解）。

**③ 把提交提前 = 這個程式庫自己標記為 racy 的那條路。**
`ggml-backend.cpp:2090`：`CGC_SUBMIT_AHEAD=1 restores the old **racy** submit-ahead order`；
`:1788` 同樣寫著 racy。也就是說：**「讓 fill 與 GPU 工作重疊」在這個派送器裡等價於「在 bytes 落地前就提交讀它的 segment」**，
而這正是 M0 那條 L3 線想要的形狀。

---

## 2. 把那個「racy 版」拿來量（零程式改動，既有的旋鈕）

兩臂，同 binary、同形狀、`CGC_HOOK_SPLIT=1 CGC_DECODE_PROFILE=1 CGC_GPU_TIMING=1`、96 tok：

| | 臂 1（出貨順序） | 臂 2（`CGC_SUBMIT_AHEAD=1`） |
|---|---|---|
| step 中位（ntok=4） | **167.24 ms** | **50.11 ms** |
| `wait / cb / submit` | 118.25 / **42.04** / 3.26 | 14.72 / **2.06** / 1.48 |
| `union / gap` | 105.46 / 54.35 | 0.00 / 0.00（觀察點失效） |
| `pool_wait`（M0 新計數器） | **36.38 ms/步** | 21.42 ms/步 |
| accept | 58.06% | **33.97%** |
| decode（HTTP） | 12.66 t/s | 14.14 t/s |

**同一顆 binary 的閘門讀數（把 `CGC_SUBMIT_AHEAD=1` 當作被測臂）**：

```
answer  : '文摘文摘文摘文摘文摘文摘文摘'
INVALID DUMP: {"reason":"non-finite values: 993280 of 993280", "n_tokens":4, "n_vocab":248320}
```

⇒ racy 不是「有點不準」，是**全部 logits 非有限值**。arm 2 的 accept 崩到 33.97% 是同一件事的另一面。

---

## 3. 這改變了什麼

1. **L3 自己那 44 ms 不是「獎品」，是 cb 的子集**：`pool_wait 36.38 ⊂ cb 42.04`。
   而 racy 探針顯示：**整個 cb（42 ms/步）可以離開臨界路徑** —— 代價是 numerics 崩掉。
   ⇒ 真正要拿的不是「把等待挪走」，而是「**讓 segment i+1 的非 MoE 部分先跑**」，等 fill 落地後再提交 MoE 部分。
2. **50.11 ms 不是合法上界**：那個數字是在 GPU 讀到未落地 bytes 的管線上量的（NaN）。它只能當「**最多有 ~117 ms/步 是可動的**」的上界，不能當預期值。
3. **L3 與「分段提交」是同一筆錢**：M0 已把等待逐項拆開（`pool_wait`／`pool_submit`／殘餘 12.2 ms），
   而搬得動它們的唯一位置在派送器。所以 M1 不該在 cache 裡再寫一套 async。

---

## 4. 建議（含歸屬與第一步）

- **不寫** cache 側的 `fill_begin/fill_await`：沒有合法的 await 點，寫出來只能預設關著，是純負債。
- **要動的是派送器**：在 `hook_seg(i)` 之後把 **segment i+1 拆兩段提交** ——
  A 段 = 不讀 union slots 的節點（norm／attention／GDN／shared-expert 等），先提交；
  B 段 = MoE 節點（`ffn_moe_gate/up/down`，讀 pool slots），fill 落地後提交。
  A 段的 GPU 執行時間就是 fill 可遮的上界。
- **第一步（~5 min，一個 arm）**：用既有的 `CGC_GPU_NODES_MATRIX=1` 最小平方路徑，
  量出「segment 內非 MoE 種類的 GPU 時間」佔比 —— 那就是這次改動的定價。
  沒有這個數字就開始改派送器，等於再一次用感覺排優先順序。
- **合併條件**：M1/M2/M3 **9/9/9** ＋ 同 prompt 輸出 md5 相同（M0 已建立這個閘門流程）；
  racy 版已經用 NaN 證明這條線不能靠「反正很慢所以先上」。
- **歸屬**：`ggml-backend.cpp` 是線 A（S2／hook_seg）的地盤。本文件就是交接單：把三個事實、一個 NaN 讀數、一個待定價的 A/B 段切法交給他們。

---

## 5. 沒有宣稱的事

- 沒有宣稱「L3 死了」：`pool_wait` 36 ms/步仍是真的、可動的；只是它的鑰匙不在 cache 裡。
- 沒有宣稱 14.14 t/s 是可用的速度：那是壞管線的數字（NaN、accept −24pp）。
- 沒有給「分段提交」的預期值：A 段有多大還沒量（§4 的第一步）。

---

## 6. 產物

- 讀數：`Backup/phase_decomp/l3prize_off.json`、`l3prize_on.json`
- 閘門日誌：`/tmp/l3_prize_gate.log`（`INVALID DUMP` 全文）、`Backup/m123_oracle_gate/summary_l3_prize_submit_ahead.json`
- 驅動：`Backup/phase_decomp/l3_prize_probe.sh`
- 日誌：`Backup/cgc_logs/llama_server_20260920_203100.log`（臂 1）、`…_203158.log`（臂 2）
- **未 commit**；**L3 的 C++ 一行都沒寫**（這是本文件的結論，不是進度落後）
