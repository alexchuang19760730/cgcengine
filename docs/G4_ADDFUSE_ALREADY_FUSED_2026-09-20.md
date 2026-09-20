# G4：「9 連 ADD 融合」這條槓桿已經被上游做掉了 —— 2026-09-20 16:1x

**結論先講**：`§EN-334` 指名的那條「每層 9 連 ADD 鏈」（`ffn_moe_add`×6 → `ffn_moe_out` → `ffn_out` →
`l_out`），在 Metal 層**早已被 ggml-metal 自己的多操作數融合吃掉**。
**它不是「還沒動工」，它是「已經在做」。** 我上一輪算的「移除 328 個 dispatch/步、上界 11.5%」
**作廢** —— 那個數字數的是**圖上的節點**，而這些節點在編碼時已經被併成 2 個 dispatch。

---

## 1. 實測（本輪唯一的新證據，用了 GPU）

新探針 `CGC_ADDFUSE_DBG=1`（add-only、env-gated、預設靜默），印在 `ggml_metal_op_bin` 裡的
`n_fuse`。載體：`llama-bench -p 0 -n 16 -d 512 -r 1 --spec-type draft-mtp`（ntok=1）。

```
  23 ×  CGC-ADDFUSE: op=ADD n_fuse=7 ne0=2048 ne1=1
  23 ×  CGC-ADDFUSE: op=ADD n_fuse=2 ne0=2048 ne1=1
```

**7 + 2 = 9**，正好是那條鏈：**9 個 ADD 節點 → 2 個 dispatch**。
（`ne1=1` 是因為這一輪 ntok=1；MTP verify 下是 2／4，拓樸相同。）

⚠️ 這一輪是**診斷跑，不是量測**：直接跑 `llama-bench`（沒走 `run_server.sh` 的完整 env）在第一個
command buffer 就 `Insufficient Memory`（`CGC-METAL-FAIL`）。結構結論不受影響 —— 融合決策發生在
編碼期，與那個 OOM 無關；但**這一輪沒有產生任何 t/s**。

## 2. 三條各自獨立的佐證（零 GPU）

1. `ggml-metal-ops.cpp:5660-5698`：`ggml_metal_op_bin` 對連續的 `ADD/SUB/MUL/DIV` 做融合，
   條件是 `f0 == f1->src[0]`（鏈）、`ggml_are_same_layout(f0->src[1], f1->src[1])`、
   所有 `src[1]` 在同一個 Metal buffer；`args.o1[0..7]` ⇒ 最多 **8 個 b 操作數 + src0 = 9 個操作數**。
2. `ggml-metal.metal:1323`：`res += ((device const T1 *)(src1 + args.o1[j]))[i1];`
   —— **同一個 kernel 內依 j 順序串序累加**，f32 按原順序 ⇒ **逐位元等價是構造性的**。
   這正是「修法 A」要寫的東西，而它已經在樹上。
3. `ggml-metal-context.m:324`：`use_fusion = getenv("GGML_METAL_FUSION_DISABLE") == nil`
   ⇒ **預設開**；而 `GGML_METAL_FUSION_DISABLE` 在 `scripts/`、記憶、任何 resolved env 裡
   **一次都沒被設過**。

## 3. ★ 方法論更正（影響之後所有 KIND×OP 的讀法）

`CGC-GPUOPS` 的 `wcntw` 是「buffer 時長 × 該 op 在該 buffer 裡的工作節點佔比」。
**融合開著的時候，節點數 ≠ dispatch 數** ⇒ 它**不能**被當成 dispatch 成本讀。

實測的落差：ADD 有 **421 個節點/步**，但這條鏈只貢獻 **~50 個 dispatch/步**（41 層 × 2）。
所以「ADD 佔 14.8%」**不是** 328 次發派的開銷 —— 它是被節點佔比放大出來的份額。
`§EN-334` 的 11.5% 上界建立在「節點數 ≈ dispatch 數」之上 ⇒ **前提是錯的，上界不成立。**

## 4. 剩下的碎屑（不值得做）

把融合窗從 8 個操作數放到 9（kargs 的 `o1` 從 8 格加到 9 格、迴圈上界 +1）⇒ 每步再省
**41 個 dispatch**，約 `nodes_work 2455` 的 **1.7%**。代價是動 kargs 結構與 kernel。
⇒ **不做。** 記下來只是為了說明「這條路已經走到盡頭」。

## 5. 這輪實際改了什麼

- `src/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp`（+18）：`CGC_ADDFUSE_DBG` 探針。
  **ADD-ONLY**：讀 `n_fuse`、印一行、不改編碼、不改任何值。未設 env 時完全靜默。
- **沒有**寫新的融合 op、**沒有**改 `llama-graph.cpp`、**沒有**動 `ggml-metal.metal`。

## 6. 誠實邊界

- 沒有新的 t/s；本輪的 GPU 使用是**結構診斷**，不是量測（且它 OOM 了）。
- `n_fuse` 的取樣只到前 200 行（`cgc_addfuse_n < 200`），且只是一個 ntok=1 的圖。
- 「融合是開的」有程式碼 ＋ 實測兩條證據；「每一層的鏈都融合」是從 23 次的規律推得的（41 層
  中取得前 200 行樣本），**沒有逐層驗證**。
