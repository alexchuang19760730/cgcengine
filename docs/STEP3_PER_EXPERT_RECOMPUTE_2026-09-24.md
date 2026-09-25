# 第 3 步：per-expert 重算 —— 設計定稿與兩個結構發現

日期：2026-09-24 22:5x
狀態：**設計定稿，未落地**（共用閘門 BLOCKED：22:46 有非本線的 `llama-bench` 在跑，swap 6.47 GB）
上游：`docs/MISS_GATE_STEP3_2026-09-24.md`（閘門 OPEN）、`docs/MISS_MASK_STEP2_2026-09-24.md`（vmask 出口）

---

## 0. 一句話

閘門雖然 OPEN，但**開工前先改判據**：原判據「greedy 逐 rep 相同」在單段提交臂上**不可能達成**
（那一臂零 fill，缺席 expert 的權重永遠不會出現）。而真正阻塞的不是帶寬也不是記憶體，
是 **fill 的同步等待固定開銷** —— 所以第 3 步要拆成三件，順序不能換。

---

## 1. 結構發現 A：`vmask` 的語義是「**fill 前**缺失」，不是「計算時缺失」

這件事決定了「把 vmask 乘到 gate 權重上（讓佔位路歸零）」在哪些臂上是**正確**、哪些臂上是**破壞**。

### 1.1 時序（靜態定案，讀碼即可確認）

```
llama_context::graph_compute()
  ├─ [step 2 我加的] 發布 valid_table（讀 slot_table[e]）      <-- vmask 的來源
  └─ ggml_backend_sched_graph_compute_async()  →  GPU 開始跑
        └─ 執行到 top-k 節點 → eval callback → expert_cache_on_topk()
              └─ ensure_batch()  →  fill_segments_pool()  →  bg_cv.wait(...)   <-- fill 在這裡，同步
                    （llama-expert-cache.cpp:1414，等待見 :997 / :1245）
```

⇒ **vmask 反映的是 dispatch 那一刻的池狀態**，也就是 `ensure_batch` 跑**之前**。

### 1.2 兩個推論（都很要緊）

| 臂 | fill | vmask=0 的那一路在計算時是什麼 | 乘 vmask 歸零 |
|---|---|---|---|
| A（41 段） | **同步**（hook 裡等到 `outstanding==0`） | 已經被 `ensure_batch` 填好，**是真權重** | ⛔ **會抹掉正確項**（miss 率 4.7% ⇒ 輸出變差） |
| B（單段） | **無**（hook 整個消失） | 真的是佔位（B-scheme 寫 `e % slots`） | ✅ **正確**，且是巨大改進（garbage → 結構化缺失） |

⇒ 這也解釋了 step 2 那個一度看似矛盾的觀測：**vmask 與 `BATCHDBG` 逐位相同**
（38 層 / 421 元素 / 0 差異）。兩者數的都是「本次 ensure 要填哪些」，而不是「計算時缺哪些」。
當初寫 step 2 時我把它當成後者，是錯的；結論（位元對得上）不受影響，但**用途**要改。

⚠ **因此：`CGC_ZERO_MISS` 只能在單段提交臂上開。** 在 A 臂開會讓輸出變差，
而這種「變差」不會崩潰、只會讓 perplexity 悄悄上升 —— 是最難發現的一類 bug。

---

## 2. 結構發現 B：阻塞項是 fill 的**固定開銷**，不是帶寬，也不是記憶體

### 2.1 供給側（全部用本日實測，不引入新假設）

輸入：`CGC-SHAPE phase=final` 的 `read_mib=1674.6 / fills=4123` ⇒ **0.406 MiB 每次 fill**。
（`fill_wait_us` / `pread_usec` 不能用，見 skill 第 65 條。）

| 情境 | miss 次數/步 | 搬運量 | 步長 | 所需持續帶寬 |
|---|---:|---:|---:|---:|
| 零 fill 天花板（B 臂實測 43.0%） | 132.75 | 53.9 MiB | 82.9 ms | **0.65 GB/s** |
| 同上，但步長縮到 B 臂的 42.2 ms | 132.75 | 53.9 MiB | 42.2 ms | **1.28 GB/s** |
| 同步 fill 穩態（A 臂實測 4.7%） | 15.0 | 6.1 MiB | 82.9 ms | **0.073 GB/s** |

⇒ **帶寬供得上**（NVMe 隨機讀量級 1~3 GB/s，最壞那一行 1.28 GB/s 也只是吃緊不是不可能）。
⇒ **瓶頸是每次 fill 的固定開銷**（`pread` syscall + `bg_cv` 喚醒 + 一次同步點），不是 bytes。
⇒ 所以正確的用力方向是 **batch 化 fill（一次 pread 搬多個 expert）**，而不是加大池、也不是換介質。

### 2.2 保留 `x` 的成本 —— **免費，不用挑 18.1 層**

per-expert 重算要 `y += g_e · down_e(up_e(x_L))`，前提是留著每層 MoE 的輸入 `x_L`。

```
40 層 × n_embd × 2 B(f16) = 160 KiB/步      （n_embd=2048）
40 層 × n_embd × 4 B(f32) = 320 KiB/步
```

⇒ 就算 **40 層全保留**也才 160~320 KiB/步，相對 8 GiB 池是 0。
⇒ **原計畫「x 按 18.1 層保留」不必要**：挑層要額外的 mask/索引邏輯，省下的記憶體是 0。
   **全保留 40 層**，邏輯最簡單。（n_embd 未實測，取 Qwen3-30B-A3B 的 2048；即使 4096 也只是 320 KiB。）

---

## 3. 修訂後的第 3 步：拆成 3a / 3b / 3c，順序不能換

| | 內容 | 依賴 fill？ | 判據 |
|---|---|---|---|
| **3a** | `vmask` 乘到 gate 權重（佔位路歸零） | **否** | B 臂輸出從 garbage 變成「k−1 項正確和」；**A 臂不許開**（見 §1.2） |
| **3b** | fill 觸發點搬出 hook ＋ batch 化 | 是（它就是 fill） | decode 段 `fill_ms/step` 可測；目標 ≤ 5 ms/步 |
| **3c** | per-expert 重算 kernel | 依賴 3b | greedy 逐 rep 相同 ＋ 落在 13.6~14.2，低於 13.0 退回 |

⇒ **3c 的判據「greedy 相同」只有在 3b 做完後才可能達成。** 直接做 3c = 白工。

---

## 4. 落點（已定位到行，等空窗落地）

### 4.1 3a — `src/llama.cpp/src/llama-graph.cpp`

- `vmask` 建於 **:2416**（`if (cgc_miss_mask)` 塊內，形狀 I32 `[1, k*n_tokens]`，由 `ids_flat` gather 而來）。
  要掛到 gate 上須**提到外層作用域**：`remap_ids` 宣告於 **:2114**，在其旁加一個
  `ggml_tensor * cgc_vmask = nullptr;`，並在 :2416 之後賦值。
- 乘法插入點：**:2700-2705**（`gating_op == SOFTMAX_WEIGHT` 的 `ggml_soft_max` 之後），
  此時 `weights` 是 `[1, n_expert_used, n_tokens]`，與 `reshape_3d(vmask, 1, k, n_tokens)`
  **元素序一致**（兩者都是 expert-major，因 `ids_flat = reshape_1d(cont(selected_experts))`
  對 `[k, ntok]` 展平）。
- 形狀不確定的風險：`weights` 在非 softmax 分支的最終形狀我**沒有讀完**，所以落地時
  **必須先加 `GGML_ASSERT` 或只在 `n_tokens == 1` 時啟用**。不要憑猜測放寬。

```
static const bool cgc_zero_miss = getenv("CGC_ZERO_MISS") != nullptr;
if (cgc_zero_miss && cgc_vmask != nullptr && n_tokens == 1) {
    ggml_tensor * vf = ggml_cast(ctx0, cgc_vmask, GGML_TYPE_F32);
    vf = ggml_reshape_3d(ctx0, vf, 1, n_expert_used, n_tokens);
    cb(vf, "ffn_moe_vmask_f32", il);
    weights = ggml_mul(ctx0, weights, vf);
    cb(weights, "ffn_moe_weights_zeromiss", il);
}
```

### 4.2 3b — `src/llama.cpp/src/llama-expert-cache.cpp`

- `ensure_batch` 主體 **:1180-1500**，`fill_segments_pool` 呼叫於 **:1414**；
  同步等待在 **:997** 與 **:1245**（`bg_cv.wait`）。
- 需要的計時器：**只包住 `ensure_batch`、按 step 累加**（現有 `fill_wait_us` 的兩個累加点
  都不在 batch 路徑上，見 skill 第 65 條）。這是「單段＋fill 落在 14.1 還是 19.9」的唯一判據來源。

### 4.3 3c — 重算 kernel

- 40 層各自一個「只算缺席 expert」的 `mul_mat_id`；ids 長度動態（0~8）。
  ggml 圖是靜態的 ⇒ **只能固定為 8 並用 vmask 把已算過的路歸零**，不能省。
- 保留 `x_L`：40 層全留（§2.2 證明了記憶體免費）。

---

## 5. 沒閉的口子

1. `weights` 在非 softmax 分支的最終形狀未讀 ⇒ 3a 落地前必須讀完或用 `n_tokens==1` guard。
2. `n_embd` 未從 log 直接讀到（只查到 `n_layer=40 / n_expert=256`），2048 是模型先驗值；
   不影響結論（4096 也只是 320 KiB/步）。
3. 供給側的 0.65~1.28 GB/s 是**需求**，不是實測供給。實測供給需要 3b 的計時器。
4. 本輪**沒有改任何 `src/` 檔案**（閘門 BLOCKED 時不應留下未經 build 驗證的 C++）。

---

## 6. swap 對 3a / 3b / 3c 的影響（2026-09-24 23:2x 追加）

### 6.1 先分清水位與速率 —— 兩件事，只有後者有害

```
sysctl vm.swapusage : used = 4.69 GB（跑的時候一度 6.47 GB）
vm_stat 5 秒增量    : pageouts = 0    pageins = 0
```

⇒ **水位高、速率為 0 = 換出去的都是冷頁，當前無害。**
累計的 `Swapouts 10.03M / Swapins 5.80M` pages 是**歷史總量**，不是當前速率，不要拿它當壓力指標。
判定壓力的唯一指標是**單位時間的 pageout/pagein 增量**。

### 6.2 但 swap 是結構性的，不是偶發的

`llama-expert-cache.cpp:491` 自己寫了原因：

> "12.7 GB model (**--no-mmap**) + 8 GiB pool over-commits RAM, swap"

⇒ 模型 `--no-mmap` ⇒ 12.7 GB **常駐、不可換出**；池 8 GiB；合計 ~20.7 GB > 16 GB 實體記憶體。
⇒ **池頁必然有一部分在 swap 裡**，這不是「今天機器比較忙」，是每次跑都會發生。

而 `CGC_POOL_MADVISE`（P1/P2，丟頁策略）在今天所有臂裡都是 **0（關）** ⇒ 沒做任何干預。

### 6.3 分項判定

| | 受 swap 影響？ | 理由 |
|---|---|---|
| **3a** vmask 乘 gate | ❌ **不受影響** | 純 GPU 圖，每層多 2 個節點（CAST + MUL），**不新增記憶體、不碰池**。判據是**正確性**不是速度 ⇒ 短跑（gen 8）即可驗，噪音無關緊要。 |
| **3b** fill 觸發點 ＋ 計時器 | ⛔ **決定性影響，必須先處理** | fill = `pread` 到**池地址**。池頁若在 swap ⇒ 寫入前觸發 **swap-in** ⇒ 這段延遲會被算進「fill 的固定開銷」。見 §6.4。 |
| **3c** per-expert 重算 | ⚠️ 間接 | GPU kernel 本身不受影響，但依賴 3b 的結論 ⇒ 3b 若被污染，3c 的判據（13.6~14.2）跟著錯。 |

### 6.4 ★★ swap 會讓 §2.1 的結論不可信（我自己的發現 B 要打折）

§2.1 說「瓶頸是每次 fill 的固定開銷（syscall／喚醒／同步點），不是 bytes，因為 NVMe 供得上」。
**這個推論有一個沒說出口的前提：fill 的目標頁在記憶體裡。**

如果目標頁在 swap 裡，那「固定開銷」裡混進了 **swap-in 的 page fault + 從 swapfile 讀回**，
它跟 bytes 成正比 ⇒ **「不是 bytes 問題」這個結論就不成立**，該優化的方向會完全不同
（變成「把池固定住／縮小常駐」，而不是「batch 化 pread」）。

⇒ **§2.1 的結論目前是「未定案」，不是「已定案」。**

### 6.5 結論與動作

1. **3a 現在就能做、也不受 swap 干擾** ⇒ 本輪已落地（見 §7）。
2. **3b 的計時器實驗必須帶一組 `CGC_POOL_MADVISE=1`（或 2）對照**，否則 `fill_ms/step`
   的讀數無法解釋（不知道裡面有多少是 swap-in）。
3. 附帶風險：**swap 水位會漂移 ⇒ 跨時間的 A/B 配對不可靠**。今天的 12.05 / 11.54 / 23.68
   都是在某個水位下量的，換一天重跑數字會動。每次配對都應該記 `sysctl vm.swapusage`
   並要求**兩臂的水位相近**，而不只是「都在 NOMINAL」。

---

## 7. 3a 已落地並驗證（2026-09-24 23:2x–23:3x）

### 7.1 改動（`src/llama.cpp/src/llama-graph.cpp`，+96 行，3 處）

1. **`:2120`** 外層宣告 `ggml_tensor * cgc_vmask = nullptr;`（在 `remap_ids` 旁）
   —— vmask 建在 `if (cgc_slot_table_gpu)` 塊內，必須提到外層才能被 ~300 行後的權重處拿到。
2. **`:2426`** `cgc_vmask = vmask;`（step 2 的 `get_rows` 之後）
3. **`:2718` 後** `CGC_ZERO_MISS=1` 塊：`cast(I32→F32) → reshape_3d(1,k,n) → mul(weights, vf)`

⚠ **插入點必須在 softmax（`:2700`）與 norm_w（`:2559`）之後** —— 兩者都會**重新歸一化**，
乘在前面會被洗掉，佔位路仍會分到權重，整個設計失效。

### 7.2 驗證結果（`seg_batch_abba.py --mode identity --gen 8`，計數是確定性的）

```
B 臂（單段提交）: skipped il=0 (vmask=0 ...) → applied=39 skipped=1 -- gating active
A 臂（41 段）   : skipped il=0/1 (vmask=0 ...) → applied=0 skipped=40 -- NOT MEASURING ANYTHING
```

- **B 臂 39/40 層生效**，唯一跳過的是 `il=0`（走 CPU，不在 GPU slot-table 路徑上）⇒ **3a 成立**。
- **A 臂 applied=0**：`B_VARS = ("CGC_SEG_BATCH","CGC_B_SCHEME","CGC_SLOT_TABLE_GPU")`
  （`seg_batch_abba.py:74`）只有 B 臂帶 ⇒ vmask 在 A 臂**根本不存在**。
  ⇒ 比「A 臂不許開」更強：**A 臂結構上不可能誤用**。

### 7.3 ★ 一個差點讓我誤判的 bug（已修，見 skill 第 67 條）

第一版告警寫 `if (il == 0 && !announced)`。但 `il=0` 正是那個**永遠走 else 分支**的層
⇒ 它在 il=0 印出 NOT APPLIED 並把一次性旗標設成 true ⇒ **39 層真的生效了卻被靜默**。

⇒ 教訓：**正確性儀器不要用單一層號當一次性 gate，用計數**。已改為
`n_applied / n_skipped` 累加 ＋ 累計達 40 時印一次摘要 ＋ 前 2 筆 skip 印明細。

### 7.4 ⚠️ 本次的 t/s 數字全部不可用 —— 但這本身就是一個量測結果

| | 本次（ZERO_MISS=1） | 上次（無） | 差 |
|---|---:|---:|---:|
| A 臂 tg | 7.56 | 11.97 | **−37%** |
| B 臂 tg | 21.65 | 22.22 | −2.6% |

**A 臂 `applied=0`，即 ZERO_MISS 對它零效應** ⇒ 那 −37% **不可能**來自本次改動。
它來自「另一條線同時在跑」（23:31 觀察到 pid 65013，`-expert-cache 3221225472`，非本線）。

⇒ **A 臂在本次充當了「零效應對照組」，量化出噪音底 = 37%**（單臂 sd 約 0.56 是乾淨窗口的值）。
⇒ 判決：**本次的計時數字一律不引用**；**計數（`applied/skipped`）可用**，因為它是確定性邏輯。
⇒ 這也直接坐實 §6.5 第 3 點：配對前不但要看 swap 水位，還要確認**沒有別的行程在搶 GPU**。

### 7.5 尚未驗證

- 3a 的**正確性收益**（B 臂輸出是否真的從 garbage 變成「k−1 項正確和」）還沒量。
  困難：B 臂無 hook ⇒ `CGC_IDSEQ_DUMP` 不存在，無法直接比 token。
  可行做法：用 llama-bench 的 answer digest 比「B 開 vs B 不開 vs A 基線」三者，
  看開了之後是否**更接近** A。需要在**乾淨窗口**做。
