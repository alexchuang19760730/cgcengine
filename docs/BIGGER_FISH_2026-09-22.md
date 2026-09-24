# 兩條「更大的魚」查證：`node` 與 `cache` 的 CPY

日期：2026-09-22　線：線A (ace)　HEAD：`demo/sweet-spot-windows-fix`　**0 重建，純查證**

## 結論摘要

| 魚 | 判定 | 一句話 |
|---|---|---|
| `node` 20–25% | **身份已查明（09-18 就查明了）**，但那個百分比**不可引用** | 不是「未查明」，是「查明後發現數字是壞口徑的產物」 |
| `cache` CPY 72% | **真魚，但先驗條件否決** | 120 個 CPY 來自 MTP；而 MTP 值 **+28.5%**，CPY 只值 2–4.6% ⇒ 淨虧 |
| **（新）非 MoE 權重 70%** | **比上面兩條都大** | 每步位元組 70% 是 dense，op 級第一名是 dense `MUL_MAT` 32.7% |

⚠️ **更正我上一輪的一句話**：我在 `FP_ORDER_TARGETS` §8 寫「`node` 24.1/20.3% … **身分未查明**」——
**錯**。它的成員在 09-18 就列出來了（`MEMORY_PERF.md:443`）。而且那 24.1% 來自壞口徑（§1）。

---

## 1. `node` 的真實身份（已查明）+ 那個數字不可引用

**成員**（`MEMORY_PERF.md:443`，350 個節點，更正後 17.7%）：

```
`MUL_MAT ne=[8192,2]` ×29　`MUL_MAT ne=[32,2]` ×30　`GET_ROWS` ×90
`ADD`/`UNARY` ×30　10 個 `FLASH_ATTN_EXT`
```

⇒ **它是 delta-net 內部運算**，不是「某個未知的大東西」。

**但 24.1/20.3% 這個數字不能用。** 它來自家族表的 `wcntw`
（`ggml-backend.cpp:2306`：`ns_kind_wns[q] += dur * wcnt[q] / wtot`）——
**把一個 command buffer 的 `dur` 平均分給該 buffer 內有工作的節點**，而 decode 一步有
`bufs=633 / nodes_all=3979`（≈6.3 節點/buffer）⇒ **落在小 buffer 裡的節點會獨吞整段時間**。

實證：`attn_norm` 被報成 **36.76 ms（8.5%）**，但 dump 直讀它是 `MUL ne=[2048,2]`
—— 40 個 4096 元素的逐元素乘法，不可能花 36.76 ms。

⇒ **細粒度改讀 op 級表**（`CGC-GPUOPS`，`nd=` 是確定值）：

```
MUL_MAT 426 個 32.7% ＞ MUL 278 16.6% ＞ MUL_MAT_ID 117 個 10.0%（MoE 專家 GEMV）
＞ CPY 210 8.1% ＞ UNARY 6.6% ＞ GATED_DELTA_NET 4.8% ＞ GET_ROWS 4.3%
```

⇒ **op 口徑下第一名是 dense `MUL_MAT`（32.7%），MoE 專家 GEMV 只有 10.0%。**

---

## 2. `cache` 的 CPY：來源是 MTP，而 MTP 值 +28.5%

### 2.1 因果鏈（本次查證，全部有源碼）

```
--spec-type draft-mtp --spec-draft-n-max 3
  ⇒ common.h:395  need_n_rs_seq() → types 含 DRAFT_MTP ? draft.n_max : 0   = 3
  ⇒ common.cpp:1645  cparams.n_rs_seq = params.speculative.need_n_rs_seq()  = 3
  ⇒ delta-net-base.cpp:503  if (cparams.n_rs_seq == 0) { 1 個 CPY }
                            else { K = n_rs_seq + 1 = 4; for t=1..K 各建 1 個 CPY }
  ⇒ 30 層 × 4 = 120 個 CPY/步   （與 dump 逐格吻合）
```

`n_rs_seq` 是 **recurrent 狀態的部分回滾快照數**（`llama-memory-recurrent.cpp:184`
`rollback >= 1 && rollback <= n_rs_seq`；`:424` `split_equal(..., n_rs_seq+1)`）——
**它是 MTP 被拒時回退狀態所必需的**，不是冗餘。

### 2.2 三條路都走不通

| 做法 | 判定 | 依據 |
|---|---|---|
| 關 MTP 省 90 個 CPY | ❌ **淨虧** | **MTP off 9.82 t/s vs MTP on 12.62 t/s ＝ +28.5%**（§EN 09-17）。CPY 只值 2–4.6% |
| 折疊 K 個 CPY 成 1 個 | ❌ **非數值等價** | `s_slot = K - t`，t=1..K ⇒ 寫進 **K 個不同 slot**（0..K-1），是 K 份不同的回滾快照 |
| 把 `n_max` 3→1 省 60 個 CPY | ❌ 大概率淨虧 | `n_max 3→5` 已作廢（k 加到多大都不到 2×，dynamic-k oracle 1.035×）；反方向減小會損失 accept，而 MTP 值 28.5% ≫ CPY 2–4.6% |

CPY 的時間價值估算（兩個獨立路徑，量級一致）：
- 固定開銷法：90 個 × 18 µs/dispatch ≈ 1.6 ms/步 ÷ 79.5 ms ≈ **2%**
- 佔比法：120/210 × CPY 8.1% ≈ **4.6%**

⇒ **2–4.6%，在 3% 門檻附近或略超，但遠小於 MTP 的 28.5%。**

---

## 3. 比兩條魚都大的：非 MoE 佔每步位元組 70%

`MEMORY_PERF.md:887`：

```
單 token 每層 29.34 MiB（非 MoE）vs 專家 1.083 MiB/個 ⇒ **非 MoE 佔 70%**
```

配合 §1 的 op 級表（dense `MUL_MAT` 32.7% 是第一名）與第②步屋頂線
（有效頻寬只佔實測峰值 **7–14%**，峰值 108.8 GB/s）：

- 每步 ~1.277 GB，其中 **~0.89 GB 是 dense 權重的重複讀取**（batch=1 GEMV 的本質代價）
- 有效頻寬 7–14% ⇒ **不是頻寬受限，有空間**，但這空間的性質還沒定（啟動開銷／occupancy／dequant）

⚠️ 記憶裡 `dnqkv_proj` 那個「29.34 → 30.55」**是家族表 `wcntw`，同樣不可引用**
（同一個壞口徑；且 29.34 在另一處是 **MiB** 不是 ms，兩處數值需重新釐清）。

---

## 4. 建議的下一步（按性價比）

1. **不做** CPY／MTP 方向 —— 先驗已被 MTP 的 +28.5% 否決
2. **不做** `node` 命名 —— 09-18 已做完
3. **要動 dense GEMV，先補一個可信的分量**：
   用 **`CGC_GPUOPS`（op 級，`nd=` 確定值）** 而非 `wcntw`，量 dense `MUL_MAT` 那 426 個的時間
   與幾何（`ne`），判定它是**啟動開銷受限**還是**頻寬受限**：
   - 啟動開銷受限 ⇒ 走合併（但 G4／K5 的結論要重新對照：**K5 的 18 µs 是在 5-node 小 op 群上量
     的，dense GEMV 是大 op，沒被判過**）
   - 頻寬受限 ⇒ 走 batching／權重佈局，但那改的是吞吐不是單流延遲

⚠️ **分工**：`M3_M4_STATUS` 與節點歸因是**別條線**在做 ⇒ 動手前先確認，別同時碰
`ggml-backend.cpp`。
