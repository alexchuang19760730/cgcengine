# G4 的工单：union 的 −12.8% 要从哪里来（2026-09-20）

作者：引擎歸因線。對象：要動 kernel／graph 的人，以及任何打算引用「node 61%」或
「KIND × OP 表」的人。

**這份文件的結論有三條，前兩條是負面但可證的，第三條是唯一還站著的路。**

1. 昨天那張 G4 地圖的**最大項是一個標籤缺口，不是一個 kernel**（§1）。
2. 那張表的權重方式讓**每一種 kind 都得到相同的成本**，它排的是 node 數，不是時間（§2）。
   而「少開 command buffer」這條路已經被 `cb_sweep.json` 關掉了（§3）。
3. ~~剩下唯一站著的是**減少 GPU kernel 的數量**~~ —— **這第 3 條在寫完一小時後被一份
   已在樹上的量測推翻**，見下面的「勘誤」。§4–§6 保留原文，但**結論已作廢**。

---

## 勘誤（2026-09-20 01:5x，寫完本文件後才讀到）

**本文件 §4–§5 的核心模型（「union ≈ work node 數 × 每 node 成本 ⇒ 少 320 個 dispatch
= −13.0%」）是錯的，而且推翻它的量測早就跑完並留在樹上：**
`docs/DOWN_COMBINE_IQ3S_IMPL_2026-09-18.md` ＋ `Backup/phase_decomp/en_dc_st_20260918.json`
＋ `en_dc_mt_ab_fwd2_20260918.json`。

| 形狀 | 融合 on | off（ctl） | 比值 | `answer_md5_set` |
|---|---:|---:|---:|---|
| 單 token | 8.09 | 9.85 | **×0.821** | 兩臂皆 `['72ca6608']` |
| 多 token（verify） | 5.32 | 6.23 | **×0.854** | 兩臂皆 `['72ca6608']` |

**也就是說：那 8 個 dispatch/層一個都沒省到，反而慢 15–18%，而輸出逐字相同。**
⇒ 我的「−320 dispatch = −13.0%」算術**被一個直接量測否證**（不是被論證否證）。
⇒ 連帶地，§2 那個 ~150 µs/node 的正確讀法是「**該步自己的平均在重新分配**」，
**不是「每個 node 的成本」**——我當時寫對了讀法，卻在 §4 用它當成本模型。

**兩個必須同時掛上的警告（否則這張表會被過度引用）：**

1. **效應量 1.76 t/s 落在 ±1.9 t/s 的單臂噪音裡**（本專案自己的噪音底）。所以
   **t/s 單獨不能判這個問題**；能分辨的儀器是 **`union_sum`**。§7 已據此改寫。
2. **同批的 Q3_K（Ornith）「+25%」不可引用**：`en_dc_orn_ab_fwd` 是 17.88 vs 17.83（打平），
   `en_dc_orn_ab_rev` 是 19.65 vs 15.72（+25%）——而**對照臂自己在兩次之間從 17.83 掉到 15.72**。
   兩次互相矛盾 ⇒ 那不是一個結果，是一次未受控的漂移。

**這一輪真正學到的（比工單本身值錢）**：geist「少 kernel 就會快」是一個**關於機制的假設**，
而它與「8 個專家從並行 threadgroup 變成單一 threadgroup 的 `for e` 序列」是**同時發生**的兩個變化。
**唯一被這份量測動到的量是「平行度」。** ⇒ G4 剩下的問題是機制優先的：
**在不是位元組、不是 buffer 分組、也不是 dispatch 數的前提下，union 到底由什麼決定？**
（線索：`docs/DOWN_COMBINE_IQ3S_IMPL_2026-09-18.md` §4 的三個候選全**未驗**。）

### 第二輪（同一天稍晚）：**那個「慢 15–18%」本身也不成立** —— 這對 A/B 沒有對照成功

「它慢 15–18%」是從 **兩個先後跑的 server**（`llama_server_20260918_041527.log` = on、
`..._041727.log` = ctl，相隔 2 分鐘）的 t/s 讀出來的。**用 union_sum 逐步配對重讀那兩個 log，
出現了一個讓這個比較作廢的事實：**

`en-dc-on` 那一臂**沒有** `CGC_DC_MULTITOK` ⇒ 閘門 `cgc_dc_shape_ok` 只在 `n_tokens == 1` 成立。
所以**在 `ntok=4` 的 verify 步上，兩臂走的是同一條圖路徑**（都不融合），那些步上兩臂**應該逐位元相同**。

而逐步配對（同 step 號，n=27）讀到的是：

| 量 | 中位 Δ(on−ctl) | on>ctl 的步 | 比值 | 這是什麼 |
|---|---:|---:|---:|---|
| `total` | +38.31 ms | 17/27 | ×1.262 | 整步 |
| `wait` | +15.27 ms | 17/27 | ×1.236 | **GPU 側**（CPU 等 GPU 完成） |
| **`union`** | **+13.55 ms** | **19/27** | **×1.215** | **GPU 側（Metal 時間戳）** |
| `gap` | +6.12 ms | 17/27 | ×1.269 | GPU 側 |
| `cb` | +1.53 ms | 19/27 | ×1.031 | **host 側 top-k hook**（與圖路徑無關） |
| `submit` | +0.12 ms | 16/27 | ×1.022 | host 側 |

**讀法**：兩臂在**閘門不可能觸發的步**上，**host 側只差 2–3%，GPU 側差 21–24%**。
⇒ 這不是「融合比較慢」，這是**兩次 server 之間有一個 ~22% 的 GPU 狀態差**。
⇒ **而那個差正好和題目要量的效應一樣大** ⇒ **「慢 15–18%」與「t/s 差 1.76」都被同一個混淆吃掉。**
（佐證：本專案自己的配對設計規則說 arm 間散佈可達 1.54×、
單臂噪音 ±1.9 t/s；而這對臂是**先後跑、沒有交錯**的。）

**⇒ 對 G4 的淨效果：這條路「有沒有變慢」是 UNRESOLVED，不是「已知變慢」。**
我上一輪把「沒被證實的慢」當成「已證實的慢」寫進了 records，**這一節是那件事的更正**。

**兩個都要留著的結論**：
1. **「−320 dispatch = −13.0%」仍然是不成立的推論** —— 它把 node 數當成本，從來沒被量過。
2. **但它也不是被否證的** —— 唯一能測它的那次嘗試（09-18）解析度不足。**問題是開的。**

**可重用的控制（這一節真正的產物）**：
> **任何有「閘門／旗標」的 A/B，都要先檢查一個「旗標不該影響的子集」上兩臂是否相同。**
> 這裡的子集是 `ntok=4` 的步、對照量是 `cb`/`submit`（host 側）與 `union`（GPU 側）。
> 若一個本該不動的量動了，**這個 A/B 就沒有對照成功，不必再讀它的效應量**。
> 成本：一個 grep 已存在的 log。**它能救回一整輪實驗，也能阻止一條錯的否證。**

**所以 G4 的下一步不是「修 kernel」，也不是「放棄融合」，而是：先證明儀器量得出 13%。**

**仍然成立的部分**：§0 的全部口徑警告、§1（`node` 是未命名桶）、§2（KIND×OP 排的是 node 數、
逐 kind 不可識別、容器級簽名分組可用）、§3（command buffer 粒度已關）、以及 **§5 的 M1 判定**
——後者樹上早已有證明（`llama-graph.cpp:2803-2822` 寫明 `CGC_ADD_ORDER=rev`
「早已證聚合對順序敏感」），我的程式碼閱讀只是重新推導了一次。

---

## 0. 口徑（先讀這節，否則後面每個數字都會被誤用）

- **兩個不同的總量**。`union` = 一層的 GPU 區間（span），`gpu_sum` = 該層各 command buffer
  時長之和（buffer 並行 ⇒ 會重複計數）。G4 的目標量是 **`union_sum`**。
- **`CGC-GPUOPS` / `CGC-GPUOPK` / `CGC-GPUNODE` 分解的是 `gpu_sum`／`ns_total`，不是 `union`。**
- 本文件用的熱態步：`Backup/cgc_logs/llama_server_20260919_210013.log`（E2b，`slot_alloc_head2b`，
  capB）**step 916–951**：`total` 105–135 ms、`union_sum` 95–98 ms、`gap_sum` 21–31 ms、
  `ntok=4`、`wait` 82–91%。這是與 E2b 的 139.16 / 104.60 同一制度。
- **NSM（§3、§5）那批數據來自冷池段**，平均值 `gpu_sum ≈ 249 ms/step`（≠ 熱態 94）。凡引用
  NSM 的**絕對值**都要標明；它的**結構**（誰和誰同一個 command buffer）不受影響。

---

## 1. 「node」不是一個 kernel，是一個沒被命名的桶

`CGC-GPUNODE` 的 kind 表最大一項是 `node`。它的來源是：

```c
// ggml.c:7190-7193  (ggml_build_forward_expand)
if (strlen(node->name) == 0) {
    ggml_format_name(node, "node_%d", cgraph->n_nodes);
}
```

⇒ **凡是在加進圖時還沒有名字的節點，都會拿到 `node_<索引>`**；`llama-graph.cpp` 的 kind
詞彙表裡那條 `"node"` 前綴就把它們整桶收走。而 `(other)`（名字存在但前綴不在詞彙表裡）在這份
log 上是 **0.00 ms**（`CGC-GPUNODE: ... | other=0.00 ms | nkind=44`），因為自動名先一步把
它們收乾淨了。

**所以「node 61% 是未 fuse 的 elementwise」這句話不成立。** `node` 桶的 op 組成（step 1，
`CGC-GPUOPK`）是：

| op | 佔該桶 | 佔該步 |
|---|---:|---:|
| **GET_ROWS** | **25.3%** | 3.1% |
| MUL | 22.5% | 2.7% |
| UNARY | 22.5% | 2.7% |
| MUL_MAT | 18.3% | 2.2% |
| ADD | 8.4% | 1.0% |
| FLASH_ATTN_EXT | 3.1% | 0.4% |

最大的一項是 **GET_ROWS**（專家權重的 gather），不是 elementwise。而這六項裡**每一項都在別的
具名桶裡更貴**（`norm` 的 RMS_NORM、`ffn_moe_gate/up/down` 的 MUL_MAT_ID、`cache` 的 CPY）——
換句話說，`node` 桶裡的東西是**未被命名的副本**，把它們命名不會讓任何東西變快，只會讓下一張
地圖可用。

**G4 的動作（便宜、零風險）：** 在 `llama-graph.cpp` 給這些節點補名（`cb()` 就是
`ggml_format_name`，不進數值路徑；`CGC_ADD_ORDER` 那段的註解已經論證過這件事）。補完之後
`node` 桶會解散成可排序的項。**這是工具工作，不是加速**，但沒有它，剩下 30% 的 union 無法歸因。

---

## 2. KIND × OP 那張「工作加權」表排的是 node 數，不是成本

`ns_kop_wns[q][o]` 的定義（`ggml-backend.cpp:1962-1971` 的註解自己寫得很清楚）是：把**該 buffer
的時長**分給「裡面會發出工作的節點」。檢查六個 kind：

| op | `nd` | `nd/2455` | `wcntw` 佔比 |
|---|---:|---:|---:|
| MUL | 278 | 11.3% | 10.6% |
| ADD | 421 | 17.1% | 16.1% |
| MUL_MAT | 426 | 17.4% | 15.8% |
| RMS_NORM | 130 | 5.3% | 6.6% |
| GET_ROWS | 159 | 6.5% | 6.0% |
| CLAMP | 39 | 1.6% | 1.5% |

⇒ `wcntw[op]/ns_total ≈ nd[op]/2455`。同理，`CGC-GPUOPS` 的每個 op 得到 **105–170 µs/node**，
而該步的平均就是 **150 µs/node**。**這張表按 node 數排序，不是按時間。**（我用最小二乘直接
去識別每個 kind 的邊際成本，結果 R² = 0.13 且出現負係數——因為同一個 command buffer 裡的 kind
向量是共線的，**逐 kind 成本從這份數據不可識別**。工具：`Backup/g4_kind_cost_lsq_20260920.py`。）

**能用的是「容器級」那張表**（`Backup/g4_buffer_signature_20260920.py`）：command buffer 的
kind **集合**就是它的身份，分組是精確的，不是模型。整段 NSM 窗口（69494 個 buffer）：

| ms/總 | n/總 | 中位 us | 簽名（截斷） |
|---:|---:|---:|---|
| 42% | 5887 | 1480.8 | `alpha attn_norm beta cache conv dnbeta_proj dnqkv_proj ffn_ ffn_gate ffn_moe_ …`（每層的主 buffer，64 節點，n_main = MAX(64,0.1N)） |
| 15% | 2030 | 1439.8 | `(other) Kcur Qcur Vcur attn_norm cache ffn_ …`（full_attn 層的主 buffer） |
| 10% | 5887 | 329.5 | `cache gdn_out k_conv v_conv z-` |
| 8% | 5887 | 622.1 | `attn_output cache gdn_out node norm` |
| 6% | 3857 | 1378.3 | `attn_post_norm cache ffn_moe_probs` ← 2–3 節點卻 1.38 ms |
| 6% | 5887 | 233.0 | `attn_residual dn_normg_mul final_output linear_attn norm` |

每步約 **351 個 command buffer**，其中 **40 個**是主 buffer（`1=9 2=26 3-7=264 8+=40` 的直方圖，
step 951）。**主 buffer 一共吃掉約 42% 的 buffer 時間。**

⚠️ **不要**從這張表推「主 buffer 貴 ⇒ n_cb 要調小」——§3 已經把這條路量過了。

---

## 3. 已關閉：command buffer 的粒度

`Backup/phase_decomp/cb_sweep.json`（2026-09-15）已經跑過 `p25-cb{1,2,16}-phase`：

| 臂 | decode t/s | answer_md5 |
|---|---:|---|
| n_cb=1 | **10.34** | 28097996 |
| n_cb=2 | **9.98** | 28097996 |
| n_cb=16 | **10.28** | 28097996 |

**平的**（散布 3.5% < 單臂噪音 ±1.9 t/s），而且三個臂的答案 md5 相同。
⇒ 預設的 n_cb=8「甜點」（`run_server.sh:106`）其實是一段**高原**，不是一個峰。
⇒ **G4 的 −12.8% 不能從「少開 MTLCommandBuffer」拿到。**

⚠️ 這條測的是 **MTLCommandBuffer 的分組**，**不是 GPU kernel 的數量**。kernel 一個都沒少。
所以「kernel 數量是否是約束」**仍然是開的**——而那正是 §4 要打的東西。

---

## 4. 唯一站著的路：每層那條 7 深的串行 ADD 鏈

`llama-graph.cpp:2849-2873`（**這是我今天打開的那段**）：

```cpp
ggml_tensor * moe_out = cur_experts[0];
for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {   // n_expert_used = 8
    moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
    ggml_build_forward_expand(gf, moe_out);
    cb(moe_out, "ffn_moe_add", il);
}
cb(moe_out, "ffn_moe_out", il);
```

其中 `cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])`
——**每個專家的切片都不連續**（元素 (e,j,t) 在 `e*4 + j*n_embd*4 + t*n_embd*k*4`）。

**規模**（`CGC-GRPH` 圖轉儲，`llama_server_20260918_050331.log`，2604 節點）：

| op | 總數 | 構成 |
|---|---:|---|
| ADD | 430 | **`ffn_moe_add` 240**、`ffn_moe_out` 40、`attn_residual` 40、`ffn_out` 40、`l_out` 40、`node` 30 |
| MUL | 281 | `node` 60、`attn_norm` 40、`attn_post_norm` 40、`ffn_shexp_gated` 40、**`ffn_moe_weighted` 40**、`gate` 30 |
| UNARY | 170 | `shared_expert_gate_sigmoid` 40、`beta_sigmoid` 30、`a_softplus` 30、`conv_output_silu` 30、`node` 30、`gate_sigmoid` 10 |
| MUL_MAT | 431 | `node` 130、`ffn_moe_logits_raw` 40、`ffn_gate` 40、`ffn_up` 40、`shared_expert_gate` 40、`ffn_shexp` 40、`z` 30、`linear_attn_out` 30 … |
| GET_ROWS | 161 | `node` 90、`ffn_moe_weights` 40、`conv_states` 30 |
| RMS_NORM | 131 | `norm` 131 |

**每層 7 個互相依賴的 ADD，每步 280 個**——這是**單一最大的具名群集**，佔 work node 的
240/2455 = **9.8%**，且 7 個都在同一條關鍵路徑上，資料量卻只有 `n_embd × n_tokens × 4` ≈ 32 KB。

**ROI（以 work node 數為代理）**：`(9−1)/9 × 9 = 8 個 node/層` → **−320 個 dispatch/步 = −13.0%**。

> ⚠️ **這條 ROI 已被否證，不要再引用。** 同一個改動（§5）已量測為 **慢 15–18%**。
> 保留原句是為了讓「錯的推論長什麼樣」留在紙上：它把 **node 數**當成**成本**，
> 而 §2 自己說的只是「它排的是 node 數」。

---

## 5. 已經實作、但算術上過不了 M1 的那條路（**本節是今天最重要的負面結果**）

樹裡**已經有**一個融合 op：`GGML_OP_MUL_MAT_ID_DOWN_COMBINE`，把
`down MUL_MAT_ID + weight MUL + 7×ADD`（9 個節點）合成 **1 個**。

- gate：`llama-graph.cpp:2714`，收 `Q3_K | IQ3_S | IQ4_XS`，要求 `!weight_before_ffn`、
  無 down scale/bias；`n_tokens==1 || (CGC_DC_MULTITOK && n_tokens<=8)`。
- Metal kernel：`kernel_mul_mv_id_down_combine_{q3_K,iq3_s,iq4_xs}_f32` **三個都在**
  （`ggml-metal.metal:11886 / 12012 / 12124`）。`llama-graph.cpp:2729` 那句「the kernel is
  **Q3_K-only**」是**過期註解**。
- **本模型的 `ffn_down_exps` 逐層查表**（讀 GGUF）：**IQ3_S ×37、IQ4_XS ×3（blk 34/38/39）、
  Q3_K ×1（blk 40 = MTP head）** ⇒ **gate 剛好覆蓋全部 40 個主幹層**。
- env：`CGC_DOWN_COMBINE` 與 `CGC_DC_MULTITOK` **都已在 `run_server.sh` 的 allowlist**
  （`:1406`、`:1419`），不會被靜默丟掉。**預設關。**

**但 kernel 的算術不保序。** `kernel_mul_mv_id_down_combine_q3_K_impl`
（`ggml-metal.metal:11788-11864`）的結構是：**專家外層、K 區塊內層，而且權重乘在每一個 K 項上**：

```metal
for (int e = 0; e < args.nei0; ++e) {          // 專家，外層
    const float weight = weights[e + token*args.nei0];
    for (int i = ix; i < nb; i += 4) {          // K 區塊，內層
        ...
        sumf1[row] += weight * d1 * (scales[0] - 32);   // ← 權重進了 K 累加
        sumf2[row] += weight * d2 * (scales[2] - 32);
    }
}
float sumf = (sumf1[row] + 0.25f*sumf2[row]) / (1 << shift);
float sum_all = simd_sum(sumf);                 // ← 整個專家迴圈只做一次
```

它算的是 **`Σ_e Σ_k (w_e · q_e,k)`**，而且 8 個專家共用同一個累加器、只做一次 `simd_sum`。
圖上那條路算的是 **`Σ_e ( w_e · (Σ_k q_e,k) )`**，每個專家各自 `simd_sum` 一次、權重只乘一次、
然後才相加。

**代數相等，浮點不等。** 而 **M1 是 `same row_fnv1a64` ⇒ 逐位元 logits**
（`m123_oracle_gate.py:16`），G2/G5 的閘就是 `M1/M2 = 884/884`
⇒ **以現在這個寫法，它會被 G2 依構造否決。**

⚠️ 一個會誤導人的舊觀察：`llama-graph.cpp:2706-2710` 記著「單 token 那對的 `answer_md5` 相同
（72ca6608 == 72ca6608），寬形狀不同」。**那是生成文字的 md5，不是 logits 的逐位元**——argmax
對最後幾個 ulp 是穩健的，所以它**不能**當成 M1 會過的證據。

---

## 6. 兩個保序的修法（**前提已被否證 —— 見前言與 §6 開頭的警告；兩者都不要做**）

判準只有一條：**M1 = 884/884**（`Backup/knifeedge_matrix/ref_..._v6_nbaware.jsonl`，
md5 `72d82a33ad79e0e69bc935acd24228f2`）。任何重新結合（reassociation）都會失敗。

**先排除掉一個看起來很聰明的做法**：把 7 個 ADD 換成 `ggml_sum_rows`。
`kernel_sum_rows_impl`（`ggml-metal.metal:1721`）是**樹狀歸約**——
`for (i0 = tpitg.x; i0 < ne00; i0 += ntg.x) sumf += src_row[i0]; sumf = simd_sum(sumf);`
⇒ 得到的是 `(a0+a2+a4+…) ⊕ (a1+a3+a5+…)` 的配對和，**不是** `((((a0+a1)+a2)+a3)+…)`。
**不可用。**

### 修法 A（小、穩、必過）——只融合那條 ADD 鏈

新增一個 op（或一個 Metal 端的 pattern 特例），**每個執行緒負責一個輸出元素**，對
`e = 0 … 7` 做**字面的序列 f32 加法**：

```metal
float acc = src_e0[i];                 // 與 ggml_add 完全相同的順序
for (int e = 1; e < nei0; ++e) {
    acc = acc + src_e(i, e, row, token);   // 步幅 nb11 / nb12，見 cur_experts 的 view 定義
}
dst[row + token*nb1] = acc;
```

沒有跨執行緒歸約 ⇒ **逐位元等價是構造性的，不是測出來的**。
**收益：6 個 dispatch/層 = 240/步 = −9.8% of work nodes。** 另外省掉每層 6 次 f32 中間值的往返。

### 修法 B（大、是真正的目標）——修好 down-combine

把權重**提出 K 迴圈**，並讓每個專家各自歸約（複刻 `kernel_mul_mv_id_<type>_f32` 的
`nr0/NSG/simd_sum` 形狀），再以**專家索引順序**序列相加：

```
total[row] 累加            // 對應 7 個 ADD 的鏈
for e in 0..nei0:
    dot1[nr0] = 0; dot2[nr0] = 0            // ← 每個專家從零開始（現在是共用）
    for i = ix; i < nb; i += 4:  dot += d*(scales-32)      // 無 weight
    de = simd_sum((dot1[row] + 0.25f*dot2[row]) / (1 << shift))
    total[row] = total[row] + weight * de   // 權重只乘一次，然後才加
```

~~**收益：8 個 node/層 → −320 dispatch/步 = −13.0%，剛好跨過 G4 的 −12.8%。**
風險：`simd_sum` 的樹狀歸約必須與 `mul_mv_id` 逐位元一致…**這是本 repo 唯一一個「已實作、
已 allowlist、只差算術保序」的 13%。**~~

> ⚠️ **本節的前提已被否證**（見勘誤）：這個融合**已經存在、已經量過**，而且是 **慢 15–18%**
> （輸出逐字相同）。所以：
> - 修法 B（保序重寫）**不會有任何收益**——保序之後它仍然比較慢。**不要做。**
> - 修法 A（只 fuse 那條 7-ADD 鏈）**也不該做**：它同樣是「用節點數當成本」的推論，
>   而唯一相關的量測顯示「拿掉並行度會變慢、拿掉 dispatch 不一定變快」。
> - **正確的下一步不是修 kernel，而是先量「平行度換 union 的交換率」**（§7）。

---

## 7. 下一步（**已依勘誤改寫**）

**不要**先跑 audit、**不要**跑 M1 oracle —— 兩者的答案樹上都有：覆蓋率 40/40 已量
（`CGC-DCFUSED` vs audit），順序敏感性已證（`CGC_ADD_ORDER=rev`），而且那條路徑已經知道**比較慢**。

唯一值得做的是**機制優先**的那一量：**在唯一一個「已知發生過平行度變化」的配對上，把讀數從
t/s 換成 `union_sum`。**

```sh
# 配對就是 en-dc-on/en-dc-ctl（單 token）與 en-dc-mt/en-dc-mt-ctl（verify 形狀）。
# decode_sweep.py 不報 union ⇒ 兩個臂都要開 CGC_DECODE_PROFILE，再從 server log 讀。
RUN_REPLAY_BENCH=0 python3 scripts/check/decode_sweep.py --profile prefill250 \
  --arms en-dc-on,en-dc-ctl --rounds 3 --warmup 1 --n-predict 24 \
  --json Backup/phase_decomp/g4_dc_union_st.json --force
# log 路徑從 banner `[log] <path>` 取（不要讀 launcher log），然後：
python3 Backup/union_by_layer_20260919.py <server log>     # header 的 union_sum / gap_sum / cb
# 判準：只看 ntok=4 完整步的 union_sum 中位數，不看 decode_tps。
```

**三種結局各自的意思（事前寫下來，免得事後挑一個）**：

| 讀到什麼 | 意思 | 下一步 |
|---|---|---|
| 融合臂 `union_sum` **上升 ~2.6 ms/層** | **平行度 → union 的交換率是真的**；union 由「每顆 kernel 填得出多少並行」決定 | 在**未融合**的臂上掃 `CGC_MMV_NSG`（`ggml-metal-device.cpp:818`）的 occupancy 旋鈕 |
| `union_sum` **平** 而 t/s 掉 | fusion 傷的不是 union ⇒ **G4 的目標量與瓶頸不是同一個東西** | 回頭質疑 G4 的 `to`（`union <= 38*mean_len`）是否寫對了 |
| 兩臂都在噪音內 | 這一對**量不出東西**（±10% 的儀器底） | 換差異更大的配對，或承認這條路不可測 |

**修法 A / B 都不要做**（§6 的警告已改）。**任何 `cmake --build` 都要等窗空**——
它是對機器上所有在跑的量測的一次寫入（2026-09-17 11:29 那次代價是別人一整輪 A/B 橫跨兩個 build）。

---

## 8. 這份數據**不能**支持的（免得被引用出去）

1. ❌「union 是頻寬受限」——**反過來**。每步強制流量 ≈ 1.45 GB（專家 32 個選擇 × 1.136 MB × 40）
   加稠密權重 ≈ 2.2 GB，在 `union_sum` 96 ms 內 ⇒ **約 22.5 GB/s**，而 M4 約 120 GB/s。
   **離地板 5 倍**。（⚠️ 這是**上界估算，不是量到的位元數**；樹裡沒有 GPU 流量計數器。
   它的用途是排除「頻寬已滿」這個解釋，不是給出可優化的 GB/s。）
2. ❌「主 buffer 貴 ⇒ 要調 n_cb」——§3 已關。
3. ❌「`node` 桶貴所以 elementwise 值得 fuse」——`node` 是未命名桶，裡面最大的是 GET_ROWS。
4. ❌「KIND × OP 表可以按 kind 排序成本」——它排的是 node 數（§2），且逐 kind 邊際成本**不可識別**。
5. ❌「down-combine 會加速所以應該開」——它的**算術不保序**，G2 依構造否決（§5）。
6. ❌❌ **「少 320 個 dispatch 就少 13% 的 union」** —— **這條是本文自己犯的錯，已被直接量測否證**
   （同一個改動慢 15–18%）。**任何「以 node 數估算收益」的句子都不要再寫。**
7. ❌「Ornith 的 down-combine +25%」——**兩次互相矛盾**（`fwd` 打平、`rev` +25%，
   而對照臂自己漂了 13%）。**不可作為正面結果引用。**
8. ❌「`answer_md5` 相同 ⇒ M1 會過」——`answer_md5` 是**生成文字**，不是 logits；
   而 `CGC_ADD_ORDER=rev` 早已證聚合**對順序敏感**。要判值變更的優化，判準是**輸出逐字相同**。

---

## 9. G4 的 −12.8% 現在**量不出來**（2026-09-20，離線，零 GPU）

前面 §7 說「先證明儀器量得出 13%」。**這一節把那個問題答掉了，而答案是：以現有的設計，答不出來。**

語料裡只有一個臂是**完整的 AB/BA**（四份 server log、一個旗標、兩個順序）：Ornith 的
down-combine —— `llama_server_20260918_043439=on_fwd`／`043255=ctl_fwd`／`043746=on_rev`／`043625=ctl_rev`。
按**步號**在四份共有的 **45 個 ntok=1 的步**上配對（工具 `Backup/paired_union_ab.py`，
設計不可比時它**拒絕出數**而不是報一個數）：

| 半 | Δunion (on−ctl) | on>ctl |
|---|---:|---:|
| fwd | **+39.59 ms（+44.3%）** | 40/45 |
| rev | **−26.77 ms（−29.9%）** | 12/45 |

**兩半的符號相反 ⇒ 沒有任何一個旗標能同時造成兩者**：那是漂移。
而「兩次先後跑」的設計會把 **|+44.3 − (−29.9)|/2 = 37.1%** 當成效應報出來 ——
**那正是 09-18 的 IQ3_S「−15~18%」與已發表的 Ornith「+25%」所用的設計**。

用 AB/BA 估計量之後，殘留漂移降到 **+9.95%**，效應估計變成 **+37.6% ± 6.9 ms**
（SE(中位) = 1.253·σ/√n），即 **3SE = 23.3% 的 union —— 仍然比 G4 的 12.8% 寬**。
在 σ(逐步 Δ)=37.2 ms 下，要讓 12.8% 達到 3SE 需要 **n ≈ 103–150 個配對步**（現有最大是 45）。

**⇒ 結論**：G4 的 −12.8% **用任何現有紀錄上的設計都量不出來**，而且它**原理上**不可能靠
「比兩次先後跑」量到。所以**該做的不是槓桿，是把量測本身做出來**：
`paired_ab.py --null`（兩槽同 binary、開 `CGC_DECODE_PROFILE`、**同一個 server 內交錯**）
先確定底。上面那個 37.2 ms 是**上界**（四台 server ＋ 有偏的共享步子集：fwd 自己的中位是
113.3 ms，而共享 45 步上只有 89.4 ms）。null 若回到 4% 以上，G4 的正確狀態是
**NOT MEASURABLE**，不是「沒有變化」，而 `to` 要重切。

### 9.1 這件事**不**阻塞 G6（要講清楚，免得別人在等）

`union <= 38 * mean_len` 是**一條式子**，但兩邊**各自可量、互不相等**：
G6 提升 `mean_len` 完全不需要 union 讀數；union 的門檻再從 G4 最終的 union 用算術推出來。
本節只記錄「G4 那一半目前沒有量到」，**不動 G6 的門檻** ——
配對裡目前唯一有支持的分支仍然是 G1 落地後的那支（step 99.4 ms、`mean_len >= 2.485`）。
