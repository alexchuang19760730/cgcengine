# 一層的節點／運算普查：MoE 專家 GEMV 只佔 **3% 的節點**，而層的 66% 是資料搬運

日期：2026-09-18 02:5x（+08）
儀器：`CGC_GRPH_DBG=1`（新增到 allowlist）＋ `CGC_GPU_NODES`／`CGC-NSM` 的既有 dump
狀態：**未 commit**

---

## 0. 為什麼做這個

`docs/M3_MOE_SHARE_2026-09-18.md` 的結論是「MoE 專家矩陣乘 ≤ 30% of `wait` ⇒ 40% 判準不成立」。
判準死了之後，下一個問題是**「那質量在哪」**。而回答它需要先知道兩件事，兩件都還沒有：
① 那些歸屬桶（`(other)` 24.4%、`cache` 17.3%、`node` 15.7% of cntw）**裝的是什麼**；
② 「節點數」與「GPU 時間」是不是同一件事。

---

## 1. 桶的真實內容（由 `CGC-GRPH` 的原始名歸類，複製儀器的最長前綴規則）

`CGC_GRPH_DBG=1` 會印前 6 次 `graph_compute` 的完整節點清單（`name`＋`op`＋`ne`）。
實得：**2 × 4116 節點（prefill，41 層 ⇒ 100.4 節點／層）＋ 4 × 91 節點（MTP draft 層）**。

| 桶 | 真實成員（實測字串） | 這是什麼 |
|---|---|---|
| `(other)` | `" (view)"`、`" (reshaped)"`、`" (reshaped) (view)"`、`Kcur-19`／`Vcur-19`、`conv_states-0`、`linear_attn_qkv_mixed-0`、`qkv_mixed_transposed-0` | **無名／半無名的 view・reshape** ＋ 線性注意力的管線張量 |
| `node` | `node_13`／`node_20` = ROPE、`node_31` = **FLASH_ATTN_EXT**、`node_14`／`node_16`／`node_77` = MUL_MAT、`node_66..node_71` = ADD、`node_86` = PAD、`node_9` = GET_ROWS | **有真運算但名字是自動產生的** ⇒ 這個桶是「注意力＋逐元素」的混合，**不是一個子系統** |
| `cache` | `cache_r_l0 (view)`、`cache_v_l40 (view)`、`cache_k_l40 (view)`、CPY、SET_ROWS | **CGC 專家池自己的暫存**（讀取快取、KV 寫入） |
| `ffn_moe_` | `ffn_moe_swiglu-40`（GLU）、`ffn_moe_weights*`（GET_ROWS／RESHAPE／SUM_ROWS／DIV／CLAMP）、`ffn_moe_weighted-40`（MUL）＋**它的 8 個 VIEW**、`ffn_moe_out` | MoE 的**逐元素部分**（激活、權重、合併），**不是 GEMV** |
| `ffn_moe_gate/up/down` | `MUL_MAT_ID` ne=[512,8]／[512,8]／[2048,8] | **專家 GEMV（Cell 2 唯一的靶）** |

`node` 這個桶名會誤導人：它裝的不是「雜項」，而是 **FLASH_ATTN_EXT / ROPE / MUL_MAT / ADD / PAD**。

---

## 2. 每層的運算種類普查（精確節點數，不是歸屬估計）

4116 節點 ÷ 41 層：

| 類別 | 節點／層 | 佔比 |
|---|---|---|
| **staging**（VIEW・RESHAPE・PERMUTE・PAD） | 40.0 | **39.9%** |
| **elementwise**（MUL・ADD・DIV・UNARY・SCALE・SUM_ROWS・CONCAT） | 26.6 | **26.5%** |
| **matmul**（MUL_MAT・MUL_MAT_ID） | 13.4 | **13.4%** |
| copy／gather／scatter（CPY・GET_ROWS・SET_ROWS） | 9.5 | 9.5% |
| norm／softmax | 5.6 | 5.6% |
| 其他 | 2.7 | 2.7% |
| attention／rope／ssm／argmax | 2.4 | 2.4% |

前幾名：`VIEW 972`、`RESHAPE 600`、`MUL_MAT 431`、`ADD 430`、`MUL 281`、`CPY 210`、`UNARY 170`、
`GET_ROWS 161`、`RMS_NORM 131`、**`MUL_MAT_ID 120`**。

### ⇒ **專家 GEMV 是 3 個節點／100 個（3.0%）；整層 66% 是資料搬運與逐元素。**

---

## 3. 交叉驗證：decode 段（獨立來源）給同樣的桶佔比

同一組桶，從 355 個 decode segment 的 `CGC-NSM` dump 直接數節點（96.1 節點／層）：

| 桶 | decode 節點佔比 | prefill dump 的桶佔比 |
|---|---|---|
| `(other)` | 22.6% | 21.5% |
| `cache` | 16.3% | 16.0% |
| `node` | 15.5% | 15.1% |
| `ffn_moe_` | 15.1% | 16.5% |
| `conv` | 6.5% | 6.6% |
| `ffn_moe_gate`／`up` | **各 0.9%** | **各 1.0%** |

兩個不同形狀（prefill vs decode）、兩個不同 run 的 dump 給出同一組佔比 ⇒ **普查是形狀無關的。**

---

## 4. 與 GPU 時間預算並排（步層級中位，步 busy 151.8 ms、`wait` 138.78 ms）

| 桶／家族 | 節點佔比 | **GPU cntw** | **GPU ub（硬上界）** | lb |
|---|---|---|---|---|
| `(other)` | 22.6% | 24.4% | 77.6% | 0.5% |
| `cache` | 16.3% | 17.3% | 66.2% | 0.0% |
| `node` | 15.5% | 15.7% | 51.0% | 0.0% |
| `ffn_moe_`（逐元素 MoE） | 15.1% | 14.8% | 32.0% | 3.0% |
| 全部 `ffn_moe_*` | — | 22.2% | 41.8% | **10.0%** |
| **專家 GEMV gate+up+down** | **2.7%** | **2.7%** | **28.3%** | 0.0% |
| `ffn_moe_topk` | 1.0% | 0.3% | 6.6% | **6.6%（solo 精確）** |

**節點佔比與 GPU cntw 在桶層級一致到 ±1pp**（`(other)` 22.6/24.4、`cache` 16.3/17.3、`node` 15.5/15.7、
MoE GEMV 2.7/2.7）⇒ 兩台獨立儀器同意。

---

## 5. 對 M3 判準的影響：**判準仍然不成立，但理由要說精確**

* 判準指名的是專家矩陣乘（Cell 2 唯一的靶）。它們是 **3 個節點／層**、**cntw 2.7%**、
  **硬上界 28.3% of 步 busy（≈31% of `wait`）**。⇒ **< 40%，不成立。**
* 但 **「節點數 ≠ GPU 時間」**，所以不能只用 3% 這個數字下結論 —— 上界 28.3% 才是約束。
  兩個方法（節點普查、GPU 上界）都指向「不足 40%」，這一點是穩的。
* 若把判準寬鬆讀成**整個 MoE 區塊**（含逐元素的合併鏈：8 VIEW ＋ 6 ADD ＋ MUL ＋ DIV ＋ SUM_ROWS），
  則 [lb 10.0%, ub 41.8%] **跨越 40%**，**仍未裁決** —— 而這一條的節點數其實很大
  （MTP draft 層 91 節點中有 **39 個**屬 MoE FFN 區塊）。

---

## 6. 誠實邊界

1. **op 普查是 prefill 圖**（`CGC-GRPH` 只 dump 前 6 次，被載入期的圖吃掉）。桶層級的佔比已由
   decode 段的 NSM dump 獨立驗證（§3），但 **op 層級沒有 decode 版本**。
2. **節點數不是 GPU 時間**：VIEW/RESHAPE 很可能是零成本的 metadata 操作（Metal 編碼器不一定
   真的發出 kernel），所以「66% 是搬運」不等於「66% 的 GPU 時間」。§4 的 ub 欄才是 GPU 側的界。
3. **`wait` 是 CPU 側自旋**，`seg_busy` 是 GPU 忙碌；兩者在本 run 差 ~9%（151.8 vs 138.78）。
   引用佔比時要指名分母。
4. `(other)` 內含**空名節點**（`cgc_node_name` 回 `""` 者會被儀器整格跳過，不計入分母），
   所以 `(other)` 的 22.6% 是「有名字但不在詞彙表」，不是「沒有名字」。

---

## 7. 下一步（依證據排序）

1. **把 op 普查搬到 decode 形狀**：`CGC-GRPH` 目前是「前 6 次」⇒ 被載入期吃掉。改成
   「前 6 次 **ntok ≤ 8** 的圖」就能 dump 真正的 decode 段，op 層級的普查才對得上本題。
2. **分辨「staging 是零成本還是真的成本」**：這是唯一能決定「合併 shape 鏈」值不值得做的量測。
   便宜的判準：`VIEW`／`RESHAPE` 那 ~1600 個節點在 GPU 側有沒有 `GPUEndTime-GPUStartTime`。
3. **池的暫存是比 MoE 專家路徑更大的節點消費者**（`cache` 桶 16.3% vs 專家 GEMV 2.7%，
   其中 VIEW 290 ＋ CPY 210）⇒ 若要減節點數，**先看 CGC 池的 staging，而不是專家 gather**。
4. **不要**再把 `ffn_moe_*` 的 ub 相加當成家族界（要用聯集，見 `docs/M3_MOE_SHARE_2026-09-18.md` §4）。
