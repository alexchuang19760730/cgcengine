# M1 工作項 4：canonical gather order —— 設計稿（2026-09-17）

> 對應 `docs/ROADMAP_PREFILL250_DECODE25_2026-09-13.md` §M1 工作項 4：
> 「**canonical gather order**：compacted 之後的 expert **按 expert id 排序**再進 batched matmul。
> **這是這個新架構唯一的 M1 陷阱**——slot index 是 pool 相關的，若累加順序跟著 slot 走，
> 換 pool 大小就會改變舍入，M1 直接垮。」
>
> 這份文件把那一句變成可實作的規格，並且補上今天才拿到的**必要性證據**。

---

## 0. 為什麼現在寫：它從「審慎」變成「必要」

工作項 4 原本的理由是**推理**（slot 順序是 pool 相關的 ⇒ 換 pool 會換舍入）。
2026-09-17 有了一條**實測**：`CGC_ADD_ORDER=rev`（把 8 個專家加權和的**結合順序**反轉，
其餘不動、同一個 build、同一個臂）：

| 同一個錨臂 | md5 |
|---|---|
| left-to-right `(((c0+c1)+c2)+…)` | `2d4e5099` |
| **reversed** `(((c7+c6)+c5)+…)` | **`f6718d44`** |

⇒ **這個聚合對結合順序敏感，而且敏感到會改變生成的文字。**

而探針自己的註解把「為什麼反轉是最強的單一測試」寫得很清楚：
> fp32 `add` 不是結合的，所以**若連反轉都逐位元相同，那麼這個和的任何 canonicalisation
> 都不可能改變結果**；反之，移動的幅度就是這個化簡的順序敏感度 —— 而那正是 canonical order
> 要控制的數字。

**現在已知它會移動** ⇒ **canonical order 是必要的，不是可選的**；沒有它，任何「換 pool 大小」
的比較都建立在一個會動的數字上。

---

## 1. 今天的順序是誰定的（原始碼）

`src/llama.cpp/src/llama-graph.cpp`（**工作區版本，該檔目前由另一條線持有未提交改動**）：

```cpp
// 2591：第 i 個專家的「列」是 batched 張量裡的第 i 列
cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

// 2650：左到右的鏈，累加順序 = 位置 i 的順序
ggml_tensor * moe_out = cur_experts[0];
for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
    moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
    ggml_build_forward_expand(gf, moe_out);
}
```

而位置 `i` 的**內容**來自 ids 運算元：在池路徑下那是 **slot 索引**（`ffn_moe_gate-N` 的 ids），
不是專家 id。⇒ **今天的累加順序＝slot 順序＝pool 佈局的函數。** 這就是工作項 4 指的東西。

### ★ 唯一的那個陷阱（必須寫進規格）

探針註解裡那句話是規格的另一半：

> the routing weights are gathered **positionally** (`weights = get_rows(probs, selected_experts)`)
> and consumed **positionally** by `ggml_mul`, so permuting the ids alone would pair expert A's
> output with expert B's weight —— **a silent wrong answer, not a reordering**.

⇒ **canonical order 不是「把 ids 排序」**，而是「**把 (專家貢獻, 權重) 這一對一起置換**」。
只排 ids＝靜默錯答。任何實作、任何 review、任何測試都要以這一句為前提。

---

## 2. 設計

### 2.1 排序鍵

**按 expert id 遞增**（roadmap 的字面要求），鍵由 host 取得：

- 池路徑：`llama_expert_cache` 的 **`slot_owner[il][slot]`** —— 這正是
  `docs/S1_OWNER_EXPECT_CHANNELS_20260917_1145.html` 那個 `own=` 通道讀的同一個陣列。
- 非池路徑（host leaf）：ids 本身就是專家 id（或其可直接對應的索引），鍵可直接取。

**為什麼排序可以在 host 做**：排序只需要「每個位置是哪個專家」，而那在 host 是可得的
（`slot_owner`）。⇒ **不需要裝置側排序，也不需要新的 kernel。**

### 2.2 置換的施加方式（joint permutation）

對 `n_expert_used` 個位置求一個排列 `π`，使 `key(π(0)) ≤ key(π(1)) ≤ …`，然後**同時**：

1. 把 **ids** 依 `π` 重排（⇒ `cur_experts[i]` 的第 i 列跟著變成 `π(i)` 那個專家）；
2. 把 **weights** 依 `π` 重排（因為它是按位置 gather、按位置消費）。

兩者用**同一個 `π`**。這是「同一個 (貢獻, 權重) 對」被搬到新位置，所以是**精確**的置換，
不是重新配對。

### 2.3 平局與無主（必須先決定，否則順序不唯一＝沒 canonical）

| 情況 | 建議 | 理由 |
|---|---|---|
| 兩個位置同一 expert id | 以**原位置索引**為次鍵（穩定排序） | 否則順序不唯一，canonical 就沒有定義 |
| `slot_owner == -1`（無主／非常駐） | 視為 **+∞**（排最後），次鍵仍是原位置 | 與 `CGC-SLOT-TABLE-CLAMP` 的那些位置一致地擺在尾端 |
| `n_expert_used == 1` | 不做事（只有一項，順序無意義） | 也是一個可以省掉的 fast path |
| MTP 層（blk.40） | **同一個規則**，不要例外 | 例外會讓「同一份程式碼兩套順序」變成新的不可比來源 |

### 2.4 開關與預設

做成 `CGC_CANON_ORDER=1`（**預設關**），理由與 `CGC_ADD_ORDER` 相同：
**它會改變模型輸出** ⇒ 打開的那些 run 不得被引用為品質或 D5 閘門的證據。
`run_server.sh` 的 env allowlist 要同步加（否則「沒效果」與「沒設到」同形）。

---

## 3. 驗證（三段，從便宜到貴）

| # | 測試 | 判準 | 需要什麼 |
|---|---|---|---|
| **A** | **置換不變性**（單元） | 同一組 8 個 (expert, weight) 對，用兩種不同的**輸入位置順序**餵進去，開 canonical 後 `ffn_moe_out` 的整張量摘要**逐位元相同** | 合成／現有 capture 儀器，**不需要 GPU 對照** |
| **B** | **cross-pool 不變性**（這是工作項 4 存在的理由） | 同一 prompt、**兩個不同的 `CGC_SERVER_EXPERT_CACHE_BYTES`** ⇒ 生成 token 流逐位元相同（今天會不同，因為順序跟著 pool 走） | 兩次 decode，需要空窗 |
| **C** | **S1 對照（順便裁決那條懸案）** | 開 canonical 後重跑 S1 臂 vs 錨臂的 A/B：若順序是載體，`ffn_moe_out` 的第一分歧應該消失；若仍存在，**順序被排除**，剩下的就是 mapping（`own=`/寬記錄已指到的 token ≥1 讀 slot 0） | 需要空窗；與 `docs/ROUTING_TRACE_2026-09-17.md` 同一組臂 |

**A 是唯一不需要機器的**，而且它可以直接用既有的 `CGC_TENSOR_CAPTURE`（整張量摘要）做，
不必新增儀器。**在動手改引擎之前，A 應該先能被寫出來**。

---

## 4. 它**不**修什麼（誠實邊界）

1. **不修 mapping 缺陷**：一個 canonical order 只是把**同一組（可能是錯的）專家**用固定順序相加。
   token ≥1 讀到 slot 0 的那個問題（`own=` 的 `ROUTING=477`／`CONTENT=0`、寬記錄的
   「token 0 相同、token 1 大多讀 slot 0」）**不會因此消失**。
2. **不修數值**：它改變舍入 ⇒ **所有 oracle 參考必須重建**（`m123_oracle_gate --write-ref`），
   而且 D5 的 `comparable` 要先看。
3. **不動段數**：與 D3 的 S2／S3 無關。
4. **不是 slot 的直接重排**：如上，那是靜默錯答。

---

## 5. 成本與排序

- **成本**：每層每 token 一次「≤8 項的穩定排序」＋一次成對置換（host、O(k log k)、k=8）。
  與工作項 2（動 loader 與建圖）相比幾乎是零。
- **相依**：工作項 4 與工作項 2（phase split）**互相獨立**；但 4 **更便宜且已被證明必要**
  ⇒ 若只能先做一件，**4 應該先做**（roadmap 說「只能做一件事就做 M1」指的是里程碑，
  而在 M1 內部，4 是那個可以單獨落地並單獨驗證的部分）。
- **風險**：低（純 host 的索引運算、預設關、有 A 段測試可以離線證）。唯一真風險是
  **與 MTP / zero-slot 的互動**（`-1` 與 clamp 的那兩個寫入點），所以 §2.3 的平局規則要先定。

---

## 6. 出處

- `docs/ROADMAP_PREFILL250_DECODE25_2026-09-13.md` §M1 工作項 4（原文引用見本檔開頭）
- `src/llama.cpp/src/llama-graph.cpp`（工作區版本）2591、2601-2626（探針註解）、2649-2662（鏈）
- `docs/S1_OWNER_EXPECT_CHANNELS_20260917_1145.html`（`own=` / `exp=` 兩個通道；
  `slot_owner` 的來源與「同 id 不同專家」的判讀）
- `docs/ROUTING_TRACE_2026-09-17.md` §10（寬記錄後的真正第一分歧：pass 0、117/120）
- 本輪實測：`CGC-ADD-ORDER: expert aggregation = REVERSED` ⇒ 錨臂 md5 `2d4e5099` → `f6718d44`
