# ② 的判決：**合併 shape 鏈不值得做**（那 39% 的節點不含任何 GPU 命令，由編碼器原始碼證明）；並發現逐 cb 的「GPU busy」**不是工作量**

日期：2026-09-18 02:5x–03:1x（+08）
儀器：`CGC_GPU_OPS=1`（新）＋ `ggml_metal_cgc_node_op`（新）＋ 既有的逐 command buffer 時戳
狀態：**未 commit**（動了 `src/`，未跑 D5）

---

## 0. 一句話

**① ② 都做完了，而 ② 的答案是否定的，並且是原始碼級的證明：**
`ggml-metal-ops.cpp:242-252` 把 `NONE／RESHAPE／VIEW／TRANSPOSE／PERMUTE` 當成顯式 **no-op**，
這五個佔一層 **39%** 的節點 ⇒ **它們的 GPU 成本按建構為零**，合併它們在 GPU 側一無所獲；
CPU 側的上限也只有步的 **0.8–4.2%**（量到的 `cb` 欄）⇒ **不值得做**。

**副產品（比 ② 更重要）**：為了做 ② 而把權重從「節點數」改成「會編碼的節點數」之後，
發現**逐 command buffer 的「GPU busy」不是工作量**：
* 同一個 workload，切片由 17 個 cb／segment 改成 ~50 個，**報出來的 busy 由 296 ms 變成 766 ms**（2.6×）；
* 其中 **10.6%（細）/ 51.7%（更細）坐在「整格不含任何會編碼的 op」的 buffer 裡** —— 可證為偽影；
* ⇒ 這個量大致正比於 **command buffer 的個數**，不是工作的量。
* **唯一在兩個粒度下都穩定的是 `MUL_MAT_ID`（MoE 專家 GEMV）＝ 8.0% / 8.4%** ⇒ 這是唯一能引用的 per-op 數字。

---

## 1. ② 的判決：編碼器說那 39% 是 no-op

```c
// ggml-metal-ops.cpp:242
switch (node->op) {
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_PERMUTE:
        // noop -> next node            ← 這五個不發任何 encoder
```

* 節點普查：4027 個節點裡，這一組佔 **1572 個 = 39.0%**。
* ⇒ **合併 shape 鏈在 GPU 側的收益 = 0**（它們本來就不執行）。
* CPU 側上限：`CGC-DECPROF` 的 `cb` 欄（編碼）在穩態 decode 步是 **0.8%／1.0%／1.2%／4.2%**；
  09-15 的獨立量測是 **4.54 of 80.07 ms（6%）**。⇒ 即使整條鏈消失也換不到 5%。
* **否證了上一輪普查可能引出的行動**（「66% 是資料搬運 ⇒ 去合併它」）：那是**節點數**的事實，**沒有 GPU 後果**。

---

## 2. 逐 cb 的時長**不能**歸屬到節點（先證明，再看它有多嚴重）

一個**整格只有 VIEW** 的 buffer（依 §1，它不含任何 GPU 命令）被報成 **592 µs／節點**；
而 2048 寬的 `ADD` 只有 **6.7 µs／節點**。metadata 比真的逐元素運算「貴」88 倍 ⇒ 不可能。

**而且它不是一個小修正，而是量級問題。** 同一個 workload、同一個 profile，只改切片數：

| 臂 | cb／segment | buffers／步（估） | 報出來的 busy／步 | Σwcntw ÷ busy |
|---|---|---|---|---|
| `en-ops`（`n_cb=16`） | 17 | ~700 | **296.07 ms** | 89.4% ⇒ 偽影 **10.6%** |
| `en-ops-2node`（`n_cb=63`） | ~50 | ~2050 | **766.06 ms** | 48.3% ⇒ 偽影 **51.7%** |

⇒ **「segment busy」正比於 command buffer 的個數，而不是工作量。**
（每 buffer 約 0.22–0.55 ms，跨臂大致同量級。）

**這推翻了幾條既有的推論線**（原文保留，只加標示）：
* `docs/M3_MOE_SHARE_2026-09-18.md` §4 的 `lb`（solo）與「`ffn_moe_topk` 11.5% 是精確值」——
  那些 solo 樣本絕大多數是**主槽（`a=0 b=1`）的單節點 buffer**，而它們的時長正是這裡證明為偽的東西。
* 舊的 `delta=0.00%` 自我檢查（`seg_busy == layer gpu_sum`）只證明**內部一致**（我的節點範圍推導吻合編碼器的切法），
  **不證明**一個 buffer 的時長等於它範圍內的工作。
* **仍然成立的是「聯集上界」那條**：它只用「含某家族成員的 buffer 時長之和 ÷ 全部 buffer 時長之和」——
  同一個 run 內的同一個量，偽影在比值裡大致同向，所以 `ffn_moe_* ≤ 28.3% of 步 busy` 的結論不受影響。

---

## 3. 修正：權重改成「會編碼的節點數」（`wcntw`）

`CGC_GPU_OPS=1` 新增**以 op 為鍵**的表，印兩欄：
* `cntw` = 舊的節點數加權（**會被 no-op 節點分走一大半**）
* `wcntw` = 只有「會編碼」的節點才分權重（no-op 的權重是 0，由 §1 的編碼器開關決定，不是模型）

`VIEW`＋`RESHAPE`＋`TRANSPOSE`＋`PERMUTE` 在 `wcntw` 下合計 **0.00%**，
在 `cntw` 下卻是 **38.5%（7 節點切片）/ 62.6%（2 節點切片）**。
⇒ **舊表把一半以上的歸屬浪費在不執行的節點上。**

---

## 4. per-op 預算（兩個粒度並排 —— 只有一列穩定）

步層級中位。`wcntw%`：

| op | 編碼? | **`en-ops`（7 節點切片）** | `en-ops-2node`（2 節點切片） | 穩定? |
|---|---|---|---|---|
| **`MUL_MAT_ID`（MoE 專家 GEMV）** | work | **8.0%** | **8.4%** | ✅ |
| `MUL_MAT`（稠密矩陣乘） | work | 27.3% | 11.1% | ❌ |
| `MUL`（逐元素乘） | work | 12.3% | 3.5% | ❌ |
| `CPY`（複製） | work | 9.1% | 7.1% | ~ |
| `GET_ROWS`（gather） | work | 4.3% | 9.0% | ❌ |
| `UNARY` | work | 7.5% | 3.3% | ❌ |
| `GATED_DELTA_NET`／`SCALE`／`SUM_ROWS`／`ADD`／… | work | ≤5.6% | ≤2.6% | ❌ |
| `VIEW`／`RESHAPE`／`TRANSPOSE`／`PERMUTE` | **NOOP** | **0.00%** | **0.00%** | ✅ （可證） |

**⇒ 只有 `MUL_MAT_ID` 跨兩個粒度穩定。** 其餘排序不可引用（`MUL_MAT` 27.3→11.1、
`GET_ROWS` 4.3→9.0 都翻了一倍以上）—— 因為每一個 cb 的時長都混進了與它範圍無關的固定成分。

---

## 5. 對 M3 的影響：判準**仍然不成立**，而且現在有兩個獨立方法

| 方法 | 家族 | 讀數 |
|---|---|---|
| work-weighted per-op 份額（**跨兩個粒度穩定**） | `MUL_MAT_ID`（專家 GEMV，gate/up/down） | **8.0% / 8.4%** |
| 聯集上界（不依賴 per-node 歸屬） | `ffn_moe_gate`＋`up`＋`down` | **≤ 28.3% of 步 busy（≈31% of `wait`）** |

判準要 ≥40% ⇒ **兩個方法都遠低於門檻**。**Cell 2（MoE gather 融合）沒有量，走「找不到路」分支。**

---

## 6. ③ 的答案，以及它為什麼只答了一半

* **可以確定的**：`CPY` 與 `GET_ROWS`（池的 staging／KB 搬運）**會編碼**，所以它們是真工作；
  shape 鏈則**不會**。⇒ 「若要動刀，先看池 staging 而不是 shape 鏈」這個**方向**成立。
* **不能確定的**：池 staging（`CPY`＋`GET_ROWS` = 13.4%（7 節點）／16.1%（2 節點））會不會比專家 GEMV（8%）大
  —— 兩個粒度給的排名不一致（§4），所以**不該引用「誰比較大」**。
* 同理，`MUL_MAT` 那個「最大區塊」（27.3%）也**不可引用**。

---

## 7. 誠實邊界

1. **「步 busy」不是物理量**（§2）：它隨切片數變 2.6×，且一半可能是偽影。任何**絕對毫秒**都不可引用；
   同一個 run 內的**比值**可用，且要指名粒度。
2. 只有 **2 個 trunk 步**（`--n-predict 12`）、單發、未交錯。
3. `no-op` 集合是**為 Metal 後端**證的（`ggml-metal-ops.cpp`）；若 CPU 後端把 `VIEW` 實作成複製，那個結論不適用
   —— 但本題問的是 GPU 側。
4. 本表是 **op 層級**，不是名字層級。「`ffn_moe_*` 家族」的 work-weighted 版本**尚未做**（名字表還沒接上 op）。
5. `MUL_MAT_ID` 的 8% 也只到「op 類」為止；要拆成 gate／up／down 需要再一層解析度（目前拿不到）。

---

## 8. 附帶量到：Metal 的 command buffer 上限被框住了

`n_cb=63`（**每 segment 64 個 buffer**）可以正常載入並產出；`n_cb=127`（129 個）
在載入期死鎖於 `commandBufferWithUnretainedReferences`（`_dispatch_semaphore_wait_slow`）。
⇒ **上限落在 (64, 129] 之間**。這也是為什麼「每節點一個 cb」不可達的直接原因。

---

## 9. 產物與指令

```sh
# op-keyed、work-weighted 預算（7 節點切片）
RUN_REPLAY_BENCH=0 python3 scripts/check/decode_sweep.py --profile prefill250 \
  --arms en-ops --rounds 1 --warmup 1 --n-predict 12 --json Backup/phase_decomp/en_ops3_20260918.json --force

# 2 節點切片（交叉檢查，也是目前最細的可載入配置）
RUN_REPLAY_BENCH=0 python3 scripts/check/decode_sweep.py --profile prefill250 \
  --arms en-ops-2node --rounds 1 --warmup 1 --n-predict 12 --json Backup/phase_decomp/en_ops3_20260918.json --force
```

| 產物 | 路徑 |
|---|---|
| op 表日誌（7 節點切片） | `Backup/cgc_logs/llama_server_20260918_030551.log` |
| op 表日誌（2 節點切片） | `Backup/cgc_logs/llama_server_20260918_030635.log` |
| 離線解析器 | `/tmp/en_ops_parse.py`（op 表）、`/tmp/en_step_budget.py`（名字層級預算） |

新增開關：`CGC_GPU_OPS`（需 `CGC_GPU_NODES`）已加進 `run_server.sh` 的 allowlist；
新增存取器 `ggml_metal_cgc_node_op`。

---

## 10. 下一步（依證據）

1. **把「會編碼」的權重接到名字表**（`ffn_moe_*` vs `cache` vs `attn_*`）—— 這是唯一能讓 ③ 從「方向」變成「排名」的一步，
   而且是純消費端改動（`ggml-backend.cpp` 的名字那圈已經有 op 可用）。
2. **不要**再引用任何絕對的 `seg_busy`／`dp_lay_gpu`／`gpu_union`（§2）。既有的 profile 數字（例如「`wait` 的 91%」）
   與它們的關係需要重新檢視：`wait` 是 CPU 側自旋，不受這個偽影影響，這一點反而是好消息。
3. 若真要攻 GPU 側，**唯一站得住的目標是 `MUL_MAT_ID` 的 8%**（跨粒度穩定）；其餘排序需要先解決 §2 的偽影。
