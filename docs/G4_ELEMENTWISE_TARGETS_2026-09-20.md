# G4：elementwise 那 376 次 dispatch 的**逐張量清單**

2026-09-20 16:5x–17:1x。新儀器 `CGC_ELEMW_CENSUS=1`（`ggml-metal-ops.cpp`，add-only）。

## 0. 這一份回答什麼

`§EN-338` 的 dispatch 普查說：一步 1098 次 dispatch，其中 **376 次是 elementwise、比率全 1.00
（一次都沒被融合）**。但 `UNARY` 在 ggml 裡是**一個 op 帶一個 sub-op**（`op_params[0]`），
所以那張表答不出「是哪一個 unary、哪一顆張量」。這一份把 376 次**指名**。

## 1. 儀器

在 `ggml_metal_op_encode_impl()` 裡，對 `UNARY/MUL/GLU/SCALE/L2_NORM/DIV/CLAMP/SUM_ROWS`
的每一次 dispatch 印一行：op、unary sub-op（`ggml_unary_op_name`）、**張量名**、形狀。

兩個踩到的坑（都已修，寫在程式碼註解裡）：
- `ggml_get_unary_op()` **只斷言 `GGML_OP_UNARY`**（`ggml.c:1934`），對 `GGML_OP_GLU` 呼叫會
  **abort 整個行程** —— 第一版就是這樣只印了 11 行就死（`rc=-6`）。
- stderr 被重導到檔案後是**區塊緩衝** ⇒ 每次印完要 `fflush(stderr)`，否則 abort 時緩衝區裡的
  大半個普查會一起不見。

## 2. 清單（3000 行上限 ≈ 8 步；「每步」＝ raw ÷ 8）

| op | sub-op | 張量名（去掉層號後綴） | 形狀 | raw | ≈每步 |
|---|---|---|---|---:|---:|
| GLU | — | `ffn_moe_swiglu` | [512,8] | 192 | 24 |
| GLU | — | `ffn_swiglu` | [512,1] | 192 | 24 |
| **SUM_ROWS** | — | `ffn_moe_weights_sum` | **[1]** | 192 | **24** |
| **CLAMP** | — | `ffn_moe_weights_sum_clamped` | **[1]** | 192 | **24** |
| **DIV** | — | `ffn_moe_weights_norm` | **[8]** | 192 | **24** |
| MUL | — | `ffn_moe_weighted` | [2048,8] | 192 | 24 |
| **UNARY** | **SIGMOID** | `shared_expert_gate_sigmoid` | **[1]** | 184 | **23** |
| MUL | — | `ffn_shexp_gated` | [2048,1] | 184 | 23 |
| UNARY | SIGMOID | `beta_sigmoid` | [1,32] | 139 | 17 |
| UNARY | SILU | `conv_output_silu` | [8192] | 139 | 17 |
| UNARY | SOFTPLUS | `a_softplus` | [32] | 139 | 17 |
| L2_NORM | — | `q_conv_predelta` | [128,16] | 139 | 17 |
| MUL | — | `gate` | [32] | 139 | 17 |
| L2_NORM | — | `k_conv_predelta` | [128,16] | 138 | 17 |
| MUL | — | `dn_normg_mul` | [128,32] | 138 | 17 |
| UNARY | SIGMOID | `gate_sigmoid` | [4096] | 46 | 6 |
| MUL | — | `attn_gated` | [4096] | 46 | 6 |

（`node_35`／`node_149` 等未命名張量零星出現，各 5–8 次，屬於 MTP head 與 full-attn 層。）

## 3. ★ 讀出來的一件事：**最浪費的不是「大的」，是「小的」**

三個已經指名的集群，加起來約 220 次/步（佔 elementwise 的 376）：

1. **MoE 權重正規化鏈**（`ffn_moe_weights_sum` → `_clamped` → `_norm`）：
   **SUM_ROWS〔1 個元素〕→ CLAMP〔1 個元素〕→ DIV〔8 個元素〕**，各 24 次/步
   ⇒ **72 次 dispatch/步，當中 48 次是為了一顆純量在發 kernel。**
   下游的 `ffn_moe_weighted`（MUL [2048,8]）是它唯一的消費者 ⇒ 這是**最乾淨的融合目標**：
   把 sum/clamp/div 折進那顆 MUL 的 prologue。
2. **shared expert 的 gate**：`shared_expert_gate_sigmoid`（**SIGMOID，[1] 一顆純量**）23 次/步，
   緊接著 `ffn_shexp_gated`（MUL [2048]）23 次/步 ⇒ 同一個形狀的融合目標。
3. **GDN／delta-net 的支撐 op**：`beta_sigmoid`／`conv_output_silu`／`a_softplus`／
   `q_conv_predelta`／`k_conv_predelta`／`gate`／`dn_normg_mul`，各 17 次/步，
   6–7 種 ⇒ **約 102 次 dispatch/步**，但張量形狀各異（[1,32]／[8192]／[32]／[128,16]），
   不像 1、2 那樣是線性鏈 ⇒ 需要逐個設計。

## 4. 為什麼它值得做（不是「感覺」）

- 這些 dispatch 的**工作量大到可以忽略**（1～8 個元素），成本幾乎全是發派；
- 而 `§EN-338` 已量到 elementwise 佔 **34.2% 的 dispatch**，比率 1.00；
- 集群 1＋2 是**線性鏈**（生產者→消費者一對一），不像 ADD 那樣要改加法順序
  ⇒ 這兩個是「**不動數值就能減 dispatch**」的候選（但仍要過 G2 的逐位元自證）。

## 5. 誠實邊界

- **沒有 t/s**：這一輪是編碼期普查（`CGC_ELEMW_CENSUS`），不是量測。
- 3000 行上限 ⇒ 只涵蓋前 ~8 步的 elementwise dispatch，**第 8 步之後沒有**；「每步」是 raw ÷ 8
  的估計（8 這個除數來自 GLU：raw 384 ÷ 每步 46）。
- **未做任何融合**：這一份只是把靶指名；`ggml-metal.metal` 一行未改。
- ⚠ **更正上一則的錯誤**：我上一則說「`ggml-metal.metal` 不在本線擁有範圍內（碰撞面 ②）」——
  **錯**。碰撞面 ② 是 **`ggml-backend.cpp`**；`ggml-metal.metal` 最近一次被改是
  `18e5411ec`（本線 09-18 的 down-combine），**本線可以動**。
