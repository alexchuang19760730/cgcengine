# S2 的實作切點：**不是把提交順序對調**（那已經有了）

2026-09-20 07:0x，零 GPU、純讀碼。作者：引擎線。對象：G1 的 `owner`（line B）、以及任何要動手的人。

---

## 0. 一句話

`CGC_SUBMIT_AHEAD=1`（＝「把順序對調」）**已經在樹上、隨時可開**，而它是**故意錯**的探針。
把那個順序變**合法**，需要的不是改順序，而是**把 `slot_table` 的發布從 host 搬到裝置上** ——
因為那張表是**由 segment i 算出來的 ids 推導**出來的，而 host 除非等到 segment i 完成，否則拿不到那些 ids。

**⇒ S2 ＝ 一個 cache 策略的裝置化工程，不是一行 reorder。** 本文把切點釘到行號。

---

## 1. 現行程式碼的實際順序（`ggml-backend.cpp:2599-2617`）

```cpp
for (int i = 0; i < n_segs; i++) {
    if (submit_ahead && i + 1 < n_segs) { ec = submit_seg(i + 1); ... }   // ← 探針走這條
    if (i < n_as_found) { if (!hook_seg(i)) break; }                      // ← WAIT + hook
    if (!submit_ahead && i + 1 < n_segs) { ec = submit_seg(i + 1); ... }  // ← 正確順序
}
```

而 `hook_seg(i)`（`:2110-2545`）內部是：**wait → GPU 讀時 → 儀器 → hook → 記帳**，其中

| 行 | 動作 |
|---|---|
| `:2119` | `while (cgc_done(split_backend) < target) sched_yield();` ← **那個 wait** |
| `:2495` | `sched->callback_eval(tn, false, ud)` ← bit-bisect 用的 tensor dump（只在 `CGC_TD_CB` 時） |
| `:2502` | **`sched->callback_eval(ttopk, false, ud)` ← 真正的 hook**（`expert_cache_on_topk`）|

`:2086-2090` 的註解自己寫明為什麼要等：

> **DEFAULT (correct)**: per segment, first WAIT for segment[i] to fully complete, then fire the
> top-k hook which writes the remap leaf, and only THEN submit segment[i+1] … Submitting segment[i+1]
> before the hook writes its remap is a CPU/GPU race … (modifying a buffer of an in-flight command
> buffer is UB) → stale remap → garbage / non-deterministic output.

⇒ **對調順序已經實作**（`submit_ahead` 那兩行），**缺的是讓它不必等**。

---

## 2. S1 已經做到了哪一步 —— 這是 S2 全部的基礎

`llama-graph.cpp:2116-2145` 的註解把 S1 的定義寫得很完整：

> The remap leaf below is an **INPUT tensor that the eval hook hand-writes every step**, and it is
> **the reason the dispatcher has to drain the pipeline at every layer boundary** … This switch moves
> the mapping into the graph as a gather: `slots = get_rows(slot_table, selected_experts)`
> … which is exactly what the hook computes today (`expert_cache_on_topk`), just executed on the GPU
> instead of the host.

**而實測狀態（`llama-graph.cpp:2240-2320`）**：

| 元件 | 現況 |
|---|---|
| **index vector（ids）** | ✅ **已經是裝置產生的** —— `:2300-2313` 的 r31 實測把「host 寫的 1-D root 給裝置 GET_ROWS 讀」**證否**（裝置讀到 0），所以索引向量**回到 `CONT(selected_experts)`**，即裝置側 |
| **`slot_table`（I32 [1, n_expert]）** | ❌ **仍由 host hook 發布** —— `:2243-2246`：`ggml_set_output(slot_table); cb(slot_table, "ffn_moe_slot_table", il);`，且 `:2287` 的 contract 寫「**the hook MUST fill this tensor for every step** in which the S1 nodes are built」 |

⇒ **S1 留下來的唯一 host 步驟就是那張表。**

---

## 3. 為什麼「把那張表也搬到裝置」不能靠一行改動

因為那張表是**由 ids 推導**的。`dp_lay_cb` 的自我描述（`:2014`）是
**「top-k hook (slot mgmt + **blocking fill**)」** ⇒ 每個 layer 的 hook 做兩件 host 工作：

1. **slot management**：依這一層**當下選中的 ids** 決定誰進哪個槽（＝寫 `slot_table`）
2. **blocking fill**：若被選中的專家不在池裡，**在這裡等它到**（這就是
   `cross_examination` 量到的 **6.16% of selected experts are not resident at the start of their step**）

**兩件都相依於「segment i 算出來的 ids」**，而 host 拿到那些 ids 的唯一方式就是**等 segment i 完成**。
⇒ **等的是正確性，不是習慣**（`:1782` 的原話）。

⇒ **而且那張表真的會變**：§EN-294 在交付配置上量到 **ntok=4 的 verify 步有 42.3% 的
被消費映射改值**（3449/8160）⇒ 「發布一次就好」不成立。

### 三條候選路，兩條已被判死

| | 做法 | 判決 |
|---|---|---|
| **(a)** | 把 **residency 決策 ＋ 表發布** 搬到裝置（MSL kernel／裝置側表） | **唯一倖存** —— 這就是 S2 |
| (b) | 表**延後一步**發布（用上一步的 ids） | ❌ 就是 `one_step_lag_cost`：會 **zero-map 6.16%** 的選中專家 ⇒ 掉貢獻 ⇒ `G2.zero_mapping_invariant` 直接否決 |
| (c) | 表**不變**／只發布一次 | ❌ 42.3% churn（§EN-294） |
| (d) | 對調順序、不管合法 | ❌ 探針 `CGC_SUBMIT_AHEAD=1`，**故意錯**，`G2` by construction 否決 |

---

## 4. 所以「實作 S2」真正的工作量

1. **決定 residency 的介面**：誰在裝置上決定「expert e 放哪個槽」。現行的 `expert_cache_on_topk`
   （`llama-context.cpp`，host）要在裝置上有一個等價物，或改成「表由裝置增量維護」。
2. **那 6.16% 的 first-touch（3.79%）怎麼處理**：它今天是靠 host 的 blocking fill 保證
   `zero_mapped_selected == 0`。裝置側若不能等（那正是要拿掉的），就得有一條
   「缺席時查什麼」的規則，**而這會動到數值**（除非那條規則與今天逐位元等價）。
3. **然後才**：`submit_seg(i+1)` 移到 `hook_seg(i)` 之前（＝把 `submit_ahead` 那兩行變成預設，
   並拿掉 `hook_seg` 裡的 wait）。

**⇒ 第 1、2 項是工程主體；第 3 項是一行。** 這也解釋了記錄說的
「**S2 was NEVER implemented**」—— 不是沒人想到，是它比名字看起來大。

---

## 5. 動手前的兩個阻塞（需要操作者決定，不是技術問題）

1. **owner**：`agent_harness/portal/targets.json` 的 `G1.owner` ＝
   **`line B (implementation); line I owns the ceiling reading`**。
   本線（line I）動手就是越界，**而今天早上本線自己記過「我不越界去寫 S2 的程式」**。
2. **檔案佔用**：`src/llama.cpp/ggml/src/ggml-backend.cpp` **現在是 dirty 的**
   （**本線自己**未提交的 G4 `CGC-GPULAYK` 逐層儀器，見 `.workbuddy/memory/2026-09-20.md`
   的產物清單；這裡 07:0x 原先寫成「另一條線」，是誤記）。S2 的三個改動點裡有兩個就在這個檔
   （`:2599-2617` 的迴圈、`:2086` 的註解），**而 `:2110-2545` 的 `hook_seg` 就是他正在改的區域**。
   ⇒ 在這個狀態下編輯它，會把兩條線的改動混成一團。

---

## 6. 我建議的第一步（可立即做、零風險、不碰 src/）

**為裝置側表寫一個「等價性探針」**：不改 dispatch，只在**現行正確順序**下，
額外算一次「若由裝置增量維護表，會得到什麼」，並與 host 的表逐位元比對。
它回答的是第 4 節第 2 項那個會決定成敗的問題：
**裝置側規則能不能與今天的 host 規則逐位元等價？** —— 若不能，S2 在 G2 上就沒救，
而這個答案可以在**不碰 dispatch**的前提下先拿到。

---

## 7. ★★ 執行結果（2026-09-20 07:00，交付配置，實跑）：**S1 現在會 abort**

> ## ⚠️ 2026-09-20 07:2x 取代標註（新增，**原文一字未刪**，但本節的*結論*已被推翻）
>
> **本節標題是錯的。S1 不會 abort；abort 的是這條診斷指令自己。** 判準是同一個 build、同一棵樹上
> 的一次對照（`CGC_SLOT_TABLE_GPU=1` 開著、**只是不開 `CGC_S1_DBG`**）：
>
> | 臂 | 旋鈕 | 結果 |
> |---|---|---|
> | `s1-nodbg` | `slot_table_gpu=1 s1_dbg=off` | **不 abort**；gate `PASS`，M1 9/9、M2 9/9、M3 9/9，`zero_mapped_selected=0` |
> | `s1-baseline` | `slot_table_gpu=1 s1_dbg=on` | abort（本節 §7 的那份 log） |
>
> 兩臂的 banner 逐字只差那一格（`launch_s1-nodbg.log:29` vs `launch_s1-baseline.log:29`）。
> ⇒ **`CGC_S1_DBG` 這條探針本身會殺掉它要量測的那一輪。**
>
> 根因與修法見 **`docs/S1_DIAGNOSTIC_ABORT_ROOT_CAUSE_2026-09-20.md`**。三句話：
> (1) 那張捕獲表只在**建 decode 圖**時被填（`llama-graph.cpp:2191`），而它**從不清空**；
> (2) 探針在**別的圖**（prefill 等）上照樣遍歷它，而 ggml 每個 build 都 reset 再重用 arena
>     ⇒ 舊指標會落在**當前圖的別的張量**上（實測：同一次啟動的兩次呼叫，第一次 39 條全對、
>     第二次 39 條全錯，而「是不是本圖節點」的檢查**一條都沒攔下**）；
> (3) 那些張量不保證是 4 bytes/元素，而讀取長度是按 4 bytes 算的 ⇒ `ggml-backend.cpp:349` 的越界斷言。
>
> **這一節仍然成立的部分**：崩潰的現場（§7 的 log 行、堆疊、`id_oob=16`）與「`CGC-S1-IDS-SHAPE`
> 這次沒印出來」都還是那次的原始讀數，一個字沒改。作廢的只有「S1 會崩」這個**推論**。
>
> **對 S2 的後果（與本節第 1 條相反）**：S2 **不是**「卡在一個要先修的 defect 上」。`llama-graph.cpp:2144-2145`
> 要求的前提「S1 先過 bit-identical 閘門」**已經滿足**（`s1-nodbg` 的 9/9/9）。
> ⇒ 第 §4 節的裝置化工作沒有前置阻塞；阻塞改成 §5 的兩條（owner、以及本線自己那個檔的未提交 G4 儀器）。

**指令**（閘門檢查與啟動同一個分支）：
```sh
python3 scripts/check/m123_oracle_gate.py --tag s1-baseline \
  --env CGC_SLOT_TABLE_GPU=1 --env CGC_S1_DBG=1
```

**旋鈕確實生效**（`run_server.sh` 的 banner 自證，所以不是「沒設到」）：
```
[perf]  diag: submit_ahead=off slot_table_gpu=1 canon_order=off pool_split_db=off s1_dbg=1
[guard] memory_mode=dev class=full-mtp phys=16GB free=84% other_llama_servers=0
```

**結果：請求失敗，server 已 abort。**
```
WARN: probe request failed: Remote end closed connection without response
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  INVALID RUN: the probe returned an empty answer.
  M1/M2/M3 below would happily report 1.0 on such a dump ...
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

**崩潰點（`llama_server_20260920_070035.log:368`）**：
```
src/llama.cpp/ggml/src/ggml-backend.cpp:349:
  GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds")

1  libggml-base   ggml_abort + 156
2  libggml-base   ggml_backend_tensor_set_2d_async + 0
3  libllama       llama_context::graph_compute + 3964
4  libllama       llama_context::process_ubatch
5  libllama       llama_context::decode
```

**而在 abort 之前，`mul_mat_id` 已經在吃垃圾 id**（`CGC-MMID-ASSERT`，`ffn_moe_gate/up/down-1,2,3`）：
```
name=ffn_moe_gate-1 ne02=143 n_ids=16 | id_oob=16 (first=1003731148, total=16) | zero_row=0 ...
```
`id_oob=16` ＝ **16 個 id 全部越界**，而 `first=1003731148` 不是合法專家號。

### 這意味著什麼

- **S1 的前置不是「還沒過閘」，是「目前會崩」** —— 而 `llama-graph.cpp:2144-2145` 說
  「it depends on this stage passing the bit-identical gate first」。所以 S2 卡在一個**要先修的 defect** 上。
- **兩條線索指向同一個地方**：`tensor_set_2d_async` 是 **meta backend** 的寫入路徑
  （唯一呼叫點：`ggml-backend-meta.cpp:1712`），而它從 `graph_compute` 被呼叫
  ⇒ 是 **圖的 leaf / input 在 dispatch 時被寫入**的那條路，而 S1 正好新增了 leaf
  （`slot_table` 與 `ffn_moe_ids_leaf` 都被 `ggml_set_output`）。
- `llama-context.cpp:6482` 的 `memcpy` 有 `ggml_nelements(t_ids) == n_ids_step` 守衛（不是那裡），
  而 `:6489` 的 `CGC-S1-IDS-SHAPE` 分支就是為「形狀不符」寫的（**這次沒有印出來**）。

### 下一步（按裁決力）

1. **先修這個 abort**，再談 S2 —— 它現在擋住了一切 S1/S2 的量測。
   最低成本的定位：`CGC_S1_KEEP_LEAF=1`（保留 host leaf）與 `CGC_S1_BUILD_LEAF=1`（建 leaf 但不消費）
   這兩個**現成的控制臂**，能把「哪一個新 leaf 導致越界」分開。
2. **然後才**做第 4 節的裝置化。

### 附帶：一個工具 bug

gate 的清理用了 `pkill -INT`，而 BSD 的 `pkill` 不吃 `-INT`：
```
-INT: illegal option -- -
usage: pkill [-signal] [-ILfilnovx] [-F pidfile] ...
```
⇒ 那段清理**靜默無效**（server 是被 abort 帶走的，所以這次沒暴露後果）。
應改成 `pkill -INT -f ...` 或 `kill -INT`。
