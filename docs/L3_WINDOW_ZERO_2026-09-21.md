# L3 —— 可遮窗口实测：**零**。以及唯一還活著的那條路（不是 A/B 段拆分）

日期：2026-09-21 19:5x　方法：**0 重建、0 起 server**，全部讀今天既有的 `CGC_GPU_NODES_MATRIX=1`
raw（`Backup/phase_decomp/K5/raw/nsm.stderr`，47360 行）＋ 既有派送器原始碼。
分工備註：本輪起 operator 裁定「沒有分工問題，全部我做」，故本線同時動 `ggml-backend.cpp` 與
expert cache 的歸屬判斷（但仍然**沒有寫 C++、沒有重建**）。

---

## 0. 一句話

**L3 的「把 cb 藏到 GPU work 底下」在單流 decode 上沒有窗口可藏**：MoE 節點永遠落在 segment 的
**第一個 command buffer** 裡（5811/5811 segment，MoE 之前的 duration 佔比中位 **0.000**），
而 fill 只能在前一個 segment 跑完之後才拿得到 ids ⇒ **issue 與 await 之間沒有任何 GPU 工作**。
⇒ `18.59 t/s` 是**算術上界**（`step → gpu_sum`），不是可達值；它與 `CGC_SUBMIT_AHEAD=1` 的 NaN
是同一件事的兩面。

**唯一還活著的路不是 M1 交接單建議的 A/B 段拆分（那條也是零），而是「GPU 陰影下的背景預取」
——它不需要 ids，因此不受上面的依賴鏈限制。** 定價：**上界 +7%**，且餘量只有 9%。

---

## 1. 量測：MoE 節點在 segment 裡的位置

`CGC-NSM` 每一列是一個 command buffer：`[a,b)` 節點區間、實測 `dur_ns`、該區間內各 kind 的節點數。
按 `a` 排序即可重建 segment 內的節點順序（`a==0` 是每個 segment 的第一個 buffer）。

腳本：`Backup/phase_decomp/L3/segment_moe_position.py`
（kind 分類陷阱：`ffn_moe_` 同時是 `ffn_moe_logits/probs/argsort` 的前綴，那些是路由、**不讀 pool slot**，
必須排除；第一版沒排除時得到假陽性）

```
rows=47360  segments=6362  (7.4 buffers/segment)
segments with MoE nodes: 5811 / 6362
  work BEFORE MoE  (A-window)   p10=0.000  med=0.000  p90=0.000
  MoE nodes (core)              p10=0.439  med=0.666  p90=0.734
  routing (argsort/logits/...)  p10=0.652  med=0.706  p90=0.865
  segments whose FIRST buffer already contains MoE core: 5811 / 6362
```

**分層穩健性**（按 segment 節點數分桶，排除「這是 prefill 圖造成的假象」）：

| bucket | n | pre-MoE duration 佔比（med） |
|---|---:|---:|
| small（≤70 節點） | 149 | **0.000** |
| mid（71–110 節點） | 5662 | **0.000** |

一個完整 segment 長這樣（`cum` = 該 buffer 之前的累積 duration 佔比）：

```
a=0   b=64  dur=1959.2 us  cum= 0.0%  ffn_moe_:17 cache:7 conv:7 ffn_moe_add:6 node:5 ffn_:4  [MoE-core][route]
a=64  b=69  dur=  15.4 us  cum=60.1%  cache:1 conv:1 q_conv:1 a_softplus:1 state_predelta:1
a=69  b=74  dur=  15.8 us  cum=60.6%  k_conv:2 gate:2 q_conv:1
a=74  b=79  dur= 330.1 us  cum=61.1%  z-:2 cache:1 v_conv:1 gdn_state:1
a=79  b=84  dur= 566.0 us  cum=71.2%  norm:1 cache:1 (other):1 node:1 attn_output:1
a=84  b=89  dur= 254.1 us  cum=88.6%  linear_attn:2 node:1 dn_normg_mul:1 final_output:1
a=89  b=94  dur=  69.2 us  cum=96.3%  norm:1 attn_residual:1 attn_post_norm:1 ffn_moe_logits:1 ffn_moe_probs:1 [route]
a=94  b=97  dur=  49.8 us  cum=98.5%  ffn_moe_argsort:1 attn_post_norm:1 ffn_moe_probs:1 [route]
```

⇒ **segment 形狀**：`[MoE core(層 L)] + [attn/GDN(層 L+1)] + [route/argsort(層 L+1)]`。

---

## 2. 為什麼這讓「async fill」沒有窗口

派送器主迴圈（`ggml-backend.cpp:2653-2673`，非 submit-ahead 順序）：

```
hook_seg(i)  →  while (cgc_done < done0+(i+1)*bufs) sched_yield();   // 等 segment i 全跑完
             →  top-k callback：讀 ids（GPU 算出來的）+ union + ensure_batch()   // fill 阻塞
             →  submit_seg(i+1)
```

配合 §1 的 segment 形狀：

| 時刻 | 發生什麼 |
|---|---|
| segment i 執行 | MoE core(L) 用**上一個** hook 填好的 slots；然後 attn/GDN(L+1)；然後 argsort(L+1) |
| hook_seg(i) | GPU **已空**（segment i 全等完）⇒ 讀 ids(L+1) ⇒ 發起 fill(L+1) ⇒ **阻塞等它** |
| segment i+1 提交 | 第一個 buffer **立刻就是 MoE core(L+1)**，讀剛填好的 slots |

⇒ **issue 與 await 之間沒有任何 GPU 工作可以插入**。把 MoE core 挪後？segment i+1 裡排在它後面的
是 attn(L+2)，而 attn(L+2) 依賴 MoE core(L+1) 的輸出 ⇒ **不能挪**。
這就是 `CGC_SUBMIT_AHEAD=1` 為何只有兩條路可走：要嘛等（今天的行為），要嘛讓 GPU 讀未落地的 bytes
（全部 logits NaN，`L3_M1_BLOCKED_STRUCTURAL_2026-09-20.md` §2）。

**⇒ M1 交接單 §4 建議的「segment i+1 拆 A/B 段，A 段先提交」在單流 decode 上拿不到任何時間**：
A 段（層 L+2 的 attention）依賴 B 段（層 L+1 的 MoE）的輸出，A 段不可能跑在 B 段前面。
**那份建議裡「A 段的 GPU 執行時間就是 fill 可遮的上界」——那個上界實測為 0。**

---

## 3. 交叉驗證：GPU idle ≥ fill 時間（同一支 run，兩個獨立儀器）

`L3_ADOPT_DECISION` / `L3_M1_BLOCKED_STRUCTURAL` 的同一顆 log（MTP on、`ntok=4`）：

```
total 247.98 = wait 159.77 + cb 74.18 + submit 10.09     （CPU 側計時，逐步）
union 105.46 / gap 54.35                                  （GPU 時鐘，逐步）
```

`gap` 定義就是「上一個 segment 結束到下一個 segment 開始之間 GPU 時鐘的空檔」，也就是
**hook + fill 期間 GPU 乾等到什麼程度**。若 fill 被任何 GPU 工作遮到，`gap` 必須**小於** `cb`。
實測 **54.35 > 42.04**（該 run 的 cb）⇒ **遮蔽量為零，而且 GPU 空轉得比 fill 本身還久。**
兩個儀器（CPU 計時 vs GPU 時鐘）在彼此不知道對方的情況下指向同一個結論。

---

## 4. 唯一還活著的路：GPU 陰影下的背景預取（不需要 ids）

上面的依賴鏈只鎖死**「需要 ids 的 demand fill」**。有一種填充**不需要 ids**：

> file-neighbour prefetch —— miss 時順手把檔案相鄰的 expert id（e±1..±R）也讀進 free slot。

它不預測 routing（是搭 file adjacency 的便車），因此**可以在 GPU busy 的任何時刻發起**，
視窗 = 整個 GPU busy（159.77 ms/步），而不是「issue 與 await 之間」。

而 `NEIGHBOUR_PREFETCH_VERDICT_2026-09-21.md` §3.3 判它「淨虧」的那一步，
**前提是多讀的位元組全部落在臨界路徑上（同步 IO）**：

- R=1：miss −22.1%，但讀入專家數 **1.98×**；「時間 ∝ 位元組」⇒ 1.98× 的 IO 換 22.1% 的 miss ⇒ 虧。
- 改成**背景**（在 GPU busy 的陰影下讀）⇒ 判據從「位元組數」換成「**IO 時間 vs GPU busy 時間**」：

| 量 | 值 | 來源 |
|---|---:|---|
| 臨界路徑 IO（cb） | 74.18 ms/步 | 同 run 的 `CGC-DECPROF` row |
| R=1 的 IO 總量 | **≈146.9 ms/步**（1.98×） | replay 的讀入倍數 × 上列 |
| GPU busy 可用陰影 | 159.77 ms/步（`wait`）／181.56（`gpu_sum`） | 同 run |
| 餘量 | **1 − 146.9/159.77 ≈ 8%** | — |

⇒ **R=1 剛好落在可行區（餘量 8%），R=2（2.80× ⇒ 207 ms）就出局。**
若成立，cb 隨 miss 降 22.1%：`74.18 → 57.8 ms` ⇒ `total 247.98 → 231.6`（-6.6%）
⇒ **13.61 → 14.57 t/s（+7.1%）**，交付口徑同倍約 **12.57 → 13.5 t/s**。

⚠ **這三個數字跨了 run**（miss 降幅來自 capture run，cb/wait 來自
`llama_server_20260920_023021.log`）⇒ **只能當量級，不可相乘引用**；要引用必須同 run 重測。

### 為什麼它不是「L3 原本那個形狀」

| | 原 L3（async fill / issue-and-return） | 本輪的路（GPU 陰影背景預取） |
|---|---|---|
| 需要 ids？ | **是**（demand）⇒ 被依賴鏈鎖死 | **否**（file 鄰居）⇒ 隨時可發 |
| await 點 | 不存在合法掛點（M1 §1） | 不需要新掛點：demand fill 仍在 `hook_seg` 同步等 |
| 視窗 | 0（§1–§2 實測） | GPU busy 159.77 ms/步 |
| 數值風險 | 讀未落地 bytes（NaN，已實測） | 只寫 **free slot**；但要讓路給 demand，否則反而拖慢它 |

---

## 5. 建議

1. **不要寫 A/B 段拆分。** 它的上界實測為 0（§1–§2），而改派送器分段是這輪最貴的 C++。
2. **也不要把 `18.59 t/s` 當成 L3 的目標**：那是 `step → gpu_sum` 的算術地板，不是可達值。
   本輪之後 L3 的誠實上界要改寫成 **~+7%（13.5 t/s）**。
3. 若要動手，只做一件事：**背景 neighbour prefetch R=1，只在 GPU busy 時跑、只寫 free slot、
   demand 來臨時讓路**。預註冊否證（同一支 run 三個數一起讀）：
   - ✅ 成立：`fill_wait_us`↓ **且** `cb`↓ **且** `gap` 沒有等量上升 **且** 同 prompt 輸出 md5 相同；
   - ❌ 否證：`cb` 持平或上升（⇒ SSD 已飽和，或背景讀把 demand 擠掉）／`gap` 等量上升／md5 不同。
   - 第二個否證條件（更便宜，先跑）：**量「GPU busy 期間 SSD 還剩下多少吞吐」** ——
     今天沒有這個數字，而 §4 的餘量只有 8%，它是整條路的成敗所在。
4. 成本：寫 C++（背景緒 ＋ free-slot-only ＋ 讓路）⇒ **要重建、要乾淨窗口**；本輪不做（另有線在量測）。

---

## 6. 弱點

- §1 的 raw 來自 **llama-bench 的 prefill250 cell**，不是 server 的 decode 路徑；
  segment 形狀應由同一份建圖程式決定，但**未在 server decode row 上直接印證**（要重建才有）。
- 「MoE core 在第一個 buffer」是 **buffer 粒度**的結論：同一個 buffer 內若 MoE core 之前還有別的節點，
  理論上仍可拆；但 command buffer 是 encode 完成才提交的一次性物件，**encode 前就必須知道 slots**
  ⇒ 拆了也救不了（這正是 §2 的第二段）。
- §4 的定價跨 run，且「SSD 吞吐在 GPU busy 期間不受影響」**沒有實測**（unified memory 上 SSD DMA
  與 GPU 讀權重會爭記憶體頻寬，雖然量級差 ~40×，但 8% 的餘量吃不起任何競爭）。
- replay 的 victim 規則與今天引擎相同 ⇒ pool thrash 已計入 22.1%；但**背景讀與 demand 讀並存時**
  的 victim 行為沒有 replay 過。

---

## 7. 產物

- 腳本：`Backup/phase_decomp/L3/segment_moe_position.py`
- 輸入：`Backup/phase_decomp/K5/raw/nsm.stderr`（47360 行）
- 引用：`ggml-backend.cpp:2107`（SUBMIT_AHEAD 註解）、`:2127`（hook_seg）、`:2653-2673`（主迴圈）
