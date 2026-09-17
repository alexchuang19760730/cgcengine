# down-combine 融合：可達性查證（2026-09-18）

日期：2026-09-18 03:4x–04:0x（+08）
儀器：`CGC_DOWN_COMBINE_AUDIT`（新，閘門六條件的逐層審計）＋ `CGC_DCFUSED`（新，融合節點本身）
狀態：**未 commit**（動了 `src/`，未跑 D5）

---

## 0. 被查的三句話

使用者的指令是三件事：**① 確認 down-combine 的 Metal kernel 支援哪些型別；② 把閘門開到
IQ3_S／IQ2_S／IQ4_XS；③ 交錯 A/B ×3 並確認融合節點真的出現在 dump 裡。**

查證的結果是：**① 只有 Q3_K；② 那不是「開閘門」，是寫新 kernel，而且還不夠；③ 節點確實
出現（已證），但 A/B 的第一版因熱態而無效，正確的受試是另一個模型。**

---

## 1. ① 的答案：Metal kernel **只有 Q3_K**，而且是硬寫死的

三個獨立的證據，任一即可定案：

| 位置 | 逐字 | 意義 |
|---|---|---|
| `ggml-metal.metal:11744` | `void kernel_mul_mv_id_down_combine_q3_K_impl(...)` | **只有一個 impl**，名字帶型別 |
| `ggml-metal.metal:11886-11887` | `[[host_name("kernel_mul_mv_id_down_combine_q3_K_f32")]]` | **只有一個 host_name** ⇒ library 裡只有這一個變體 |
| `ggml-metal-device.cpp:1309` | `snprintf(base, 256, "kernel_mul_mv_id_down_combine_q3_K_f32");` | pipeline 名字是**硬寫死的**，getter 沒有型別分支 |

而且 getter 的其餘部分也是 Q3_K 專屬的：

- `:1291-1292` `int nsg = N_SG_Q3_K; int nr0 = N_R0_Q3_K;`（用 Q3_K 的 tile 常數）
- `:1307` `size_t smem = 0;`，註解逐字：**「Q3_K does not need lookup table in threadgroup memory」**
  ⇒ **IQ 系列需要 lookup table**（`smem != 0`），dequant 邏輯也不同。

消費端還有第四個鎖：`ggml-metal-ops.cpp:4821` `GGML_ASSERT(op->src[0]->type == GGML_TYPE_Q3_K);`
—— 那是**硬斷言**，不是 fallback。

⇒ **「把閘門開到 IQ3_S／IQ2_S／IQ4_XS」= 實作一個新的 Metal kernel**（dequant ＋ 8 專家的
GEMV ＋ weighted combine），不是改一行型別檢查。

---

## 2. 生產上到底是什麼型別：**41 層裡只有 1 層是 Q3_K**

新儀器 `CGC_DOWN_COMBINE_AUDIT=1` 在**建圖時**逐層印出閘門六條件的實際取值
（`CGC-DCAUDIT il=.. flag=.. n_tokens=.. down_type=.. has_scale=.. has_bias=..
weight_before_ffn=.. ids=.. => fuse=..`）。生產模型 `Nail-…-IQ3_XXS-denseIQ4X.gguf`：
1189 行審計、86 個圖：

| 圖 | 層數 | 型別 | `fuse=1` |
|---|---|---|---|
| **trunk** | 40（il 0–39） | **`iq3_s` ×37、`iq4_xs` ×3** | **0** |
| MTP nextn head | 1（il=40） | **`q3_K`** | 31 |

**⇒ trunk 的 1120 行審計裡，`down_type == q3_K` 的有 0 行。**

（這也修正了上一輪用 GGUF header 推的「IQ3_S ×37 ＋ IQ4_XS ×3 ＋ Q3_K ×1」——
數字對，但**誰是 Q3_K** 那個 1 層，現在有直接證據：**是 MTP nextn head（il=40）**，
不是 trunk 的某一層。）

---

## 3. ③ 的一半：融合節點**確實出現**（而且只在 il=40）

`CGC-GRPH` 與 `CGC-GPUOPS` 看不到它 —— 兩個都在**分段路徑**內
（`ggml-backend.cpp:2055` 在同一個 `hook_seg` scope），而 MTP nextn head 的圖不走那條路。
所以在**建圖處**直接印節點本身（`CGC-DCFUSED`）：

```
CGC-DCFUSED il=40 op=31(MUL_MAT_ID_DOWN_COMBINE) name=ffn_moe_out_down_combine-40
            out_ne=[2048,1,1] n_tokens=1 down_type=q3_K swiglu_src=101(GLU)
```

7 行，與同一 log 的 `CGC-DCAUDIT … => fuse=1` 的 7 行**完全一致**。

⇒ **「閘門為真」與「節點存在」是兩個不同的宣稱**，這一輪把它們分開印了：
audit 說前者，DCFUSED 說後者。而 `ggml_build_forward_expand(gf, moe_out)` 就在那一行之後
（`llama-graph.cpp`），所以節點必然進圖。

---

## 4. trunk 為什麼永遠不走（**兩個獨立的障礙**）

| 障礙 | 條件 | 生產上的實際取值 |
|---|---|---|
| **型別** | `down_exps->type == GGML_TYPE_Q3_K` | trunk 是 `iq3_s`／`iq4_xs` ⇒ **0/1120 行符合** |
| **形狀** | `n_tokens == 1` | MTP-on 的 trunk 是 **verify 圖（ntok=2/4）**；audit 的 trunk 行裡只有 560/1120 是 `ntok==1` |

⇒ **即使寫了 IQ3_S kernel，MTP-on 下 trunk 的 verify 形狀仍然走不了。**
兩個都要解，而第二個（讓 kernel 支援多 token）不是「開閘門」。

---

## 5. ③ 的另一半：第一版 A/B **因熱態而無效**

`en-dc-on` vs `en-dc-ctl`（Nail，`--rounds 3`）：

| 臂 | median t/s | min/max | `thermal_hist` | hit% | io MB |
|---|---|---|---|---|---|
| `en-dc-on` | 2.780 | 2.52/4.23 | **`{HEAVY: 8}`** | 82.70 | 1967.3 |
| `en-dc-ctl` | **4.690** | 4.21/5.62 | **`{HEAVY: 1, MODERATE: 7}`** | 82.70 | 1966.2 |

看起來是「融合慢 41%」，但**兩臂的熱歷程不同**（`ctl` 只有 1 次 HEAVY），而池統計完全相同
（hit 82.70 對 82.70、io 1967.3 對 1966.2 MB）⇒ **那 41% 是熱態**。
同熱態比（`ctl` 的 round 0 = 2.33 @HEAVY vs `on` 的 2.52–4.23 @HEAVY）兩臂接近。

**★ 而這裡有一個判準錯誤要記下**：`thermal_launch` 兩臂**都是 HEAVY**，看不出差別；
要看 **`thermal_hist`（全程分佈）**。已經寫進 skill。

**而且這個 A/B 本來就測不出東西**：Nail 的覆蓋只有 **1/41 層**（而且那一層是 MTP head，
不在 trunk 的熱路徑上）⇒ 訊號 ≈ 0。

---

## 6. 正確的受試：**第二個模型**

樹上第二個模型的 `ffn_down_exps` 是 **Q3_K ×30**：

```
models/gguf/Ornith-1.5-35B-A3B-Abliterated-MTPv2-APEX-I-Compact-v2D-lite.gguf
  down_exps types: {'Q3_K': 30, 'Q4_K': 10, 'Q8_0': 1}   (41 MoE layers)
```

審計（`CGC_SERVER_MTP=0` ⇒ 純 decode，`ntok==1`）：

- **30 層（il 5–34）全部 `q3_K`**，10 層（il 0–4、35–39）是 `q4_K`
- 若 `flag=1`：`ntok==1 且 type==q3_K` 的行數 = **390**（30 層 × 13 個 decode 圖）
- ⇒ **覆蓋 trunk 的 75%**

⇒ 這是唯一能真正量出融合收益的受試，**不需要寫任何 kernel**。

### 6.1 ABBA 的結果

`--arms en-dc-orn-on,en-dc-orn-ctl`（正序）與 `--arms en-dc-orn-ctl,en-dc-orn-on`（反序），
各 `--rounds 2`（每臂 2 個穩態 round 計入中位數）：

| run | 臂 | median t/s | min/max | `thermal_hist` | hit% | io MB | file_reads |
|---|---|---|---|---|---|---|---|
| 正序 | `on` | **17.880** | 17.40/18.36 | **`{NOMINAL: 6}`** | 93.3 | 2981.1 | 22671 |
| 正序 | `ctl` | 17.830 | 15.24/20.42 | **`{NOMINAL: 3, MODERATE: 3}`** | 93.3 | 2981.5 | 22668 |
| 反序 | `ctl` | 15.720 | 12.43/19.01 | **`{MODERATE: 6}`** | 93.3 | 2981.5 | 22668 |
| 反序 | `on` | **19.650** | **19.56/19.75** | **`{MODERATE: 6}`** | 93.3 | 2981.1 | 22671 |

⚠️ **正序那一組不可引用**：`on` 全程 NOMINAL 而 `ctl` 有一半是 MODERATE ⇒
`AB = 1.0028` 是**熱態補償**的產物（`ctl` 的 MODERATE round 把它的中位數壓低了），
不是融合的效果。`ABBA` 的 `M = √(AB×BA) = 1.1196` 也因此**不可引用**。

★ **反序那一組可用**：兩臂都**全程 MODERATE**（同熱態）⇒ **BA = 1.2500**。

## ⇒ **融合的淨效果：+25%**（在 30/40 層的覆蓋下）

**為什麼這個 +25% 不是熱態**：`on` 跑在 `ctl` **之後**（熱態只會更差、不會更好），
而它的 `min/max` 是 **19.56/19.75（±0.5%）** —— 比 `ctl` 的 12.43/19.01（±42%）窄得多。

**池統計在四臂中完全相同**（`hit` 93.3、`io` 2981.1–2981.5 MB、`file_reads` 22668–22671）
⇒ **差異不在池**。

**對照組（同一次跑內）**：`ctl` 在 MODERATE 下是 15.720，`on` 在 MODERATE 下是 19.650
⇒ 同一熱態、同一池、同一模型、只差一個 flag。

---

## 7. 結論

1. **收益量級已量到**：在覆蓋 trunk 的 75%（30/40 層）的受試上，融合的淨效果是 **+25%**
   （反序組，兩臂同熱態）。這是本輪最重要的數字 —— 它把「值不值得為 IQ3_S 寫 kernel」
   從猜測變成有一把尺。
2. **機制**：融合路徑是活的、可達的、節點確實被建立（`CGC-DCFUSED`），但**在旗艦模型上只覆蓋
   1/41 層**（MTP nextn head），而那不是熱路徑。
3. **要真的在旗艦模型上吃到它，要解兩件事**：寫 IQ3_S／IQ2_S／IQ4_XS 的 kernel
   （IQ 系列還要 lookup table，`smem != 0`），**以及**讓它支援 `n_tokens > 1`
   （MTP-on 的 verify）。**只做前者不足。**
4. **旗艦模型的 A/B 測不出**（覆蓋 1/41，且那一層不在 trunk 的熱路徑上），
   而**第二個模型可以**（覆蓋 30/41）⇒ **以後這類判據要用對的受試**。
5. 這一輪**沒有動 kernel**：依專案已付兩次學費的「沒有判準就不要實作」
   （`CGC_MMV_FUSE`、離線最小平方），先量出收益量級再決定 —— 現在量到了。

---

## 8. 這一輪的儀器與旋鈕

| 旋鈕 | 輸出 | 用途 |
|---|---|---|
| `CGC_DOWN_COMBINE_AUDIT=1` | `CGC-DCAUDIT`（逐層六條件 ＋ 決策） | 回答「閘門在生產上取值為何」，不必推論 |
| 同上 ＋ 融合分支 | `CGC-DCFUSED`（節點本身：op／name／shape／src op） | 回答「節點是否真的進圖」 |

兩者都是**唯讀**（不改決策、不改數值），且都印在**建圖時** ⇒ 不受
`CGC_GRPH_DBG` 的「只捕捉前 6 個圖」或分段路徑的限制。

臂：`en-dc-on`／`en-dc-ctl`（Nail）、`en-dc-orn-on`／`en-dc-orn-ctl`（Ornith，強制 MTP off）。

## 9. 誠實邊界

- **本輪沒有實作任何 kernel。** +25% 是在**另一個模型**（Ornith，Q3_K ×30）上量的，
  用**既有的** Q3_K kernel。**它不能直接外推到 Nail**：Nail 要的是 IQ3_S／IQ4_XS 的 kernel，
  那是**未寫的**，而 IQ 系列的 dequant 成本與 Q3_K 不同（還要 lookup table）。
  ⇒ 可引用的述句是「**同一個融合在覆蓋 3/4 的 trunk 時值 +25% 量級**」，
  **不是**「Nail 會快 25%」。
- **反序那一組只有一個配對**（BA），而正序那一組因熱態不可用 ⇒ `M = √(AB×BA)` **不可引用**。
  要更強的結論需要更多組同熱態的配對。
- 兩臂是**先後**跑的（不是逐 round 交錯）⇒ 順序效應**沒有被完全分離**。
  緩解：`on` 跑在 `ctl` **之後**（熱態只會更差），而它反而快了 25% ⇒ 順序效應的方向與結論相反。
- `CGC-DCFUSED` 證明**節點進圖**，不證明 **kernel 真的執行**。它的旁證是
  `ggml-metal-ops.cpp:4821` 的斷言（型別不符會 abort，而 log 沒有 abort）—— 那是**推論**。
  ⚠️ 而 `+25%` **同時是「kernel 真的在做事」的證據**：若它只是讓圖變小而不執行，
  不會有正向的 t/s。
- `--rounds 0` 在 `decode_sweep.py` 下 `n_rounds=0` ⇒ **t/s 欄位全為 0.0**（探測時踩到）。
