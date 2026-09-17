# IQ3_S／IQ4_XS 的 down-combine：實作完成、**正確**，但**更慢 15–18%**

日期：2026-09-18 04:1x–04:4x（+08）
狀態：**未 commit**（動了 `src/`，D5 已跑 `--tag en-dc-iq`）
臂：`en-dc-on`／`en-dc-ctl`（單 token）、`en-dc-mt`／`en-dc-mt-ctl`（多 token）
前一份（可達性調查）：`docs/DOWN_COMBINE_REACHABILITY_2026-09-18.md`

---

## 0. 一句話

**實作是對的**（輸出與未融合逐字相同），**但它比不融合慢**：單 token **−18%**、
多 token（verify 形狀）**−15%**。
⇒ **這一輪沒有提速**，但它把「Ornith 的 +25% 能不能搬過來」這個問題回答掉了：
**不行，至少不是這個實作。**

---

## 1. 實作了什麼

| 檔案 | 改動 |
|---|---|
| `ggml-metal.metal` | `kernel_mul_mv_id_down_combine_iq3_s_impl` ＋ `_iq4_xs_impl` ＋ 兩個 `host_name` |
| `ggml-metal-device.cpp` | pipeline getter 加型別分派；名字改用 `ggml_type_name` |
| `ggml-metal-ops.cpp` | `:4821` 的 `GGML_ASSERT` → 型別 switch（其餘 `return 0` fallback）；`ne12 != 1` 的早退 → env-gated ＋ 上界 8 |
| `llama-graph.cpp` | 型別放寬到三種；`n_tokens` 上界 8；**條件抽成單一 `cgc_dc_fuse`** |
| `run_server.sh` | `CGC_DC_MULTITOK` 進 allowlist |

**三個設計上的關鍵點**：

1. **IQ 系列需要 threadgroup table**：`smem` = `512*4`（IQ3_S 的 `iq3s_grid`）／
   `32*sizeof(float)`（IQ4_XS 的 `kvalues`），而 Q3_K 是 **0**（它不需要 lookup table）。
   ⇒ `smem` 不是裝飾：給錯會讓 kernel 讀未初始化的 threadgroup 記憶體，**而且不會報錯**。
2. **lane 映射不能照抄**：IQ3_S 是 `ix = tiisg`、`ib32 += 32`；IQ4_XS 是 `ix = tiisg/16`、
   `it = tiisg%16`、`ibl += 2`。骨架（expert 迴圈 ＋ weighted 累加 ＋ 輸出）可以照 Q3_K 版，
   內層不行。
3. **`n_tokens` 的上界必須存在**：見 §2.1。

---

## 2. 驗證：**正確性**

| 實驗 | 判準 | 結果 |
|---|---|---|
| **覆蓋率** | `CGC-DCFUSED` 逐層 vs audit 的 `fuse=1` | il 0–39 **每層 24**、il=40 **37** ⇒ **完全吻合**，覆蓋 **40/40**（IQ3_S ×37 ＋ IQ4_XS ×3） |
| **單 token** | `answer_md5_set` | `72ca6608` == `72ca6608` ⇒ **逐字相同** |
| **多 token** | `answer_md5_set` | `72ca6608` == `72ca6608` ⇒ **逐字相同** |

⚠️ **M1（逐位元）必然不同** —— 融合改了結合順序（`CGC_ADD_ORDER=rev` 早已證聚合對順序敏感）
—— 所以**不能用 D5 的 M1 判這個路徑**。正確的判準是**輸出是否逐字相同**（`answer_md5_set`）。

### 2.1 走錯的一步：初版 `CGC_DC_MULTITOK` 放行了 **prefill**

初版的閘門是「`n_tokens == 1 || CGC_DC_MULTITOK`」⇒ audit 顯示 `fuse=1` 出現在
**37 行 `n_tokens=16`** 與 **74 行 `n_tokens=5632`** ⇒ **prefill 也走了融合** ⇒ 兩臂的
`answer_md5_set` **不同**（`87647aec` vs `72ca6608`），而 `hit%` 也從 82.3 跳到 93.7
（那是**算錯**造成的，不是池變好）。

**修法**：上界 **`n_tokens <= 8`**（＝池的 `CGC_POOL_MAX_TOKENS` 預設）⇒ 涵蓋 verify 的 2/4、
排除 prefill。修完之後兩臂輸出**逐字相同**。

⚠️ **`llama-graph.cpp` 與 `ggml-metal-ops.cpp` 兩處必須同步**：只改一邊會讓「圖上有節點而
Metal 不執行它」。

---

## 3. 結果：**更慢**（這是本輪的結論）

| 實驗 | 融合 | 不融合 | 比值 | `thermal_hist` |
|---|---|---|---|---|
| **單 token** | **8.09** | **9.85** | **0.821** | 兩臂皆 `{HEAVY: 6}` |
| **多 token（verify）** | **5.32** | **6.23** | **0.854** | 兩臂皆 `{HEAVY: 6}` |

（同 profile、同模型、同池、只差一個 flag；兩臂熱歷程相同 ⇒ 可比。
絕對 t/s 不可與生產的 12.0–12.9 比較 —— 這一批全程 HEAVY。）

---

## 4. 為什麼慢：三個候選，**都沒驗**

1. **★ 平行度掉了（我認為最可能）**：原本的 `kernel_mul_mv_id_*` 把 **expert 放在 grid 的某一維**
   ⇒ 8 個 expert **並行**（不同 threadgroup）；而融合是 `for e in 0..nei0` ⇒ **8 個 expert 序列**
   （同一個 threadgroup 依序做）。
   **融合省下 8 個 kernel launch，但把 8× 的平行度換成 1×。**
   這也解釋了為什麼 Ornith 的 +25% 沒有搬過來：那是 **Q3_K 的 kernel**，而它的 `nsg`/`nr0`
   （2/2）與 IQ3_S（2/4）不同，dequant 的成本結構也不同。
2. **tile 參數**：`N_SG_IQ3_S`/`N_R0_IQ3_S` 是為 `mul_mv` 調的，未必適合這個融合。
   既有旋鈕 `CGC_DC_NSG` 可以直接掃。
3. **池的互動**：融合直接讀 `expert_id * nb02 + first_row * nb01` ⇒ 存取模式與 `MUL_MAT_ID`
   不同。（⚠️ §2.1 那個 `hit%` 的差異**不能**當這條的證據 —— 那是算錯造成的。）

---

## 5. 結論與建議

1. **這一輪沒有提速。** 實作**正確**（輸出逐字相同）但**慢 15–18%**，兩個形狀一致。
2. **它留在樹上**，但 **env-gated 且預設 off** ⇒ **預設路徑與 D5 不受影響**（見 §6）。
3. **下一個該做的不是調這個 kernel的 tile，而是先回答「平行度」那條**：
   把 `for e` 換成「expert 放 grid 維 ＋ 跨 threadgroup 累加」的成本與可行性要先估。
   **若那條也不通，正確的結論是「融合對 IQ 型別不划算」**，而不是繼續調參數。
4. **Ornith 的 +25% 不可外推** —— 那是 Q3_K kernel 的成績，而 Q3_K 在本模型上只覆蓋 1/41 層。
5. **兩個可重用的方法論**：
   * 驗證「改了數值」的優化時，**判準是輸出是否逐字相同**（`answer_md5_set`），不是 M1 逐位元。
   * **閘門放寬時要問上界**：「放行多 token」不等於「放行 prefill」——
     而 audit 一行就把這件事說清楚了（`n_tokens=5632` 出現在 `fuse=1`）。

---

## 6. 閘門與產物

| 項目 | 值 |
|---|---|
| D5 | `--tag en-dc-iq`（見 commit message 的原文） |
| 臂 | `en-dc-on`／`en-dc-ctl`／`en-dc-mt`／`en-dc-mt-ctl` |
| 產物 | `Backup/phase_decomp/en_dc_st_20260918.json`（單 token）、`en_dc_mt_ab_fwd2_20260918.json`（多 token，修正後）、`en_dc_mt_ab_fwd_20260918.json`（多 token，**修正前**，保留作為反例） |

```sh
# 單 token（只測型別放寬）
RUN_REPLAY_BENCH=0 python3 scripts/check/decode_sweep.py --profile prefill250 \
  --arms en-dc-on,en-dc-ctl --rounds 2 --warmup 1 --n-predict 24 \
  --json Backup/phase_decomp/en_dc_st_20260918.json --force

# 多 token（verify 形狀；上界 8）
RUN_REPLAY_BENCH=0 python3 scripts/check/decode_sweep.py --profile prefill250 \
  --arms en-dc-mt,en-dc-mt-ctl --rounds 2 --warmup 1 --n-predict 24 \
  --json Backup/phase_decomp/en_dc_mt_ab_fwd2_20260918.json --force
```
