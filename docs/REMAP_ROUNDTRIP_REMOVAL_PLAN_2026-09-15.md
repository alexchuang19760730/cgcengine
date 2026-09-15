# remap leaf 往返移除設計（decode 序列化槓桿）

日期：2026-09-15　狀態：**設計，未實作**
前置讀物：`docs/PREFILL250_DECODE25_WHITEPAPER_20260915_1900.html`（CGC-DECPROF 首測）、
`docs/ROADMAP_PREFILL250_DECODE25_2026-09-13.md` M3 工作項 3

---

## 0. TL;DR

1. 今天新增的 **GPU 端時間戳儀器**（`CGC_GPU_TIMING=1`）給出第一個判準數字，並配一個已存在的
   零改碼天花板上界探針（`CGC_SUBMIT_AHEAD=1`）。
2. 探針結果：**decode 9.45 → 16.82 t/s（×1.78），每步 82.5 → 31–44 ms。**
   第二個在飛的 segment **不可能讓真實 GPU 計算變快**，所以「約 44% 的步時間是流程可移除的
   序列化延遲」是被證明的事實，不是推論。→ **病因是 B（序列化），不是 A（GPU 執行效率）**。
3. 因此 **`remap` 雙緩衝本身買不到東西**：CPU 往返仍在關鍵路徑上（§3 論證）。
   唯一能吃到那 1.78× 的設計是 **把 expert→slot 查表搬到 GPU，消除段邊界**（§5 D3）。
4. 本文交付 D0–D3 的階梯、D3 的實作切點、閘門與風險。
   **尚未動任何一行生產程式碼。**

---

## 1. 今天量到什麼（同一支儀器、同一 session）

`decode_sweep.py --profile prod25 --arms p25-gputime,p25-submit-ahead --n-predict 120`

| | baseline（MTP off） | `CGC_SUBMIT_AHEAD=1` |
|---|---|---|
| llama-bench decode | **9.16 / 9.45 t/s** | **16.82 t/s** |
| 輸出 md5 | `28097996`（正常） | `87647aec`（**退化**：`文摘文摘文摘…`） |
| CGC-DECPROF per step | total **71.8–90.8 ms**<br>wait 57.7–75.3 (83–90%)<br>cb 5.5–11.9 (7–15%)<br>submit 2.9–3.6 (4%) | total **31.4–44.3 ms**<br>wait 27.8–40.2<br>cb **0.28–0.98**<br>submit 3.2–3.3 |
| 池 hit% / miss | 88.3 / 15828 | 84.0 / 12755 |

`CGC-GPUTIME`（同一 run，與 DECPROF 逐步對齊，`wait` 兩者到小數第二位相同）：

```
step=216 segs=39 bufs=350 skipped=1  wait=75.27 gpu_busy_sum=95.73 (127%) gpu_union=68.00 (90%) gap=22.36 (30%) ms
step=220 segs=39 bufs=350 skipped=1  wait=70.44 gpu_busy_sum=83.75 (119%) gpu_union=64.11 (91%) gap=16.82 (24%) ms
step=224 segs=39 bufs=351 skipped=0  wait=57.72 gpu_busy_sum=59.85 (104%) gpu_union=50.10 (87%) gap=20.02 (35%) ms
step=228 segs=39 bufs=351 skipped=0  wait=70.27 gpu_busy_sum=87.26 (124%) gpu_union=63.82 (91%) gap=17.25 (25%) ms
```

- `skipped` 是 0–1：**M4 有回報 `GPUStartTime`/`GPUEndTime`**，儀器可用。
- `gpu_busy_sum / gpu_union ≈ 1.2–1.4`：一個 segment 的 `n_cb+1 ≈ 9` 個 command buffer
  **幾乎是序列的，不是並行的**。這獨立解釋了為什麼 `n_cb` 由 1 掃到 16 對 decode 完全無效。
- `gap` 是**在 GPU 自己的時鐘域**量到的段間空隙（相鄰段 `start_{i+1} − end_i`），
  與 CPU 時鐘無關 → **每步 12.3–22.4 ms（wait 的 17–35%）是可證的 GPU 閒置**。
  它比 CPU 側窗口（`cb+submit` = 8.5–15 ms）還大，差額即 commit→啟動／完成回報延遲。

### 誠實邊界（儀器的已知限制）

- `gpu_union / wait ≈ 87–95%` **不能**讀成「GPU 有 90% 的時間在執行」。同一條 queue 上，
  `[min start, max end]` 本來就會橫跨整個「queue 被佔用」的窗口（第一顆 buffer 一被 commit 就
  算開始，最後一顆完成才算結束），除非 GPU 真的整段閒置。所以 `union ≈ wait` 是**結構性的**，
  不是證據。**能當證據的是 `gap`**（它需要 queue 真的空掉才會出現）。
- `union` 與 `wait − gap` 每段差約 0.3 ms（union 略大），即相鄰段的 buffer 邊界有輕微重疊
  （9 顆 buffer 由 9 條執行緒分別 commit，順序不保證）。量級不影響結論，但別把 union 當精確值。

---

## 2. 為什麼會有段邊界（現況，附程式位置）

每一層的 MoE 依賴鏈：

```
GPU: … front_i (attn / router / argsort) ──► topk_i        ← 段 i 的最後一個節點
                                    │
CPU:                         讀回 topk_i  ──► 選 slot / ensure 填充 ──► 寫 remap leaf
                                    │
GPU:                                mmid_i 讀 remap，抓 pool 裡的專家權重  ← 段 i+1
```

- `remap` 是一個 **host 寫入的 input leaf**：
  `llama-graph.cpp:2122-2126`（`ggml_new_tensor_2d(I32, n_expert_used, n_tokens)` +
  `ggml_set_output` + `cb(remap, "ffn_moe_topk_remap")`），不是 graph 算出來的。
- 消費者：`llama-graph.cpp:2144`
  `mm_id_ids = mmid_raw ? selected_experts : (remap_ids ? remap_ids : selected_experts)`。
- 寫入端：hook（`llama-context.cpp:5142-5169`）逐 id 查 `llama_expert_cache_slot_table_safe`
  後填 `rd[i + j*n_expert_used]`。**這是一個純查表，不是計算**——這一點是 D3 的前提。
- 段邊界由 `ggml-backend.cpp` 的分段派發器提供：`as_idx[]` 收集每個 `ffn_moe_topk-*` 的位置，
  `n_segs = n_as_found + 1`（≈ 每層一段，實測 40）。每個段邊界呼叫 `hook_seg(i)`。
- **排空是刻意的**：`ggml-backend.cpp` 的註解明寫「submit segment[i+1] only AFTER the top-k hook
  of segment[i] has written its remap leaf」；`llama-context.cpp:5177-5187` 也明寫 hook 必須在
  兩個 GPU 段之間跑（同一段註解裡的 dbuf 預取是**靠這個時序**才安全的）。
- `CGC_SUBMIT_AHEAD=1` 就是還原「先提交再寫」的舊順序——**它是錯的**（GPU 可能讀到上一步的
  remap），所以今天的探針輸出才會退化成 `文摘文摘…`。它的價值純粹是**上界量測**。

---

## 3. 為什麼「雙緩衝 remap」本身買不到東西

直覺是「ping-pong 兩份 leaf，寫 A 的時候 GPU 讀 B」。但這解不了真正的依賴：

1. hook 要先**讀回 topk_i** 才能寫 leaf。`topk_i` 是段 i 的**最後**一個節點，
   所以 CPU 一定得等段 i 整段完成（`cgc_done` 輪詢），無法提前。
2. 段 i+1 的頭部是層 i 的 FFN（`mmid_i`）——它**就是** leaf 的消費者。
   所以「提交段 i+1」與「寫 leaf_i」的先後無法交換（交換就回到 submit-ahead 的錯誤）。
3. 那麼能不能讓 GPU 在 CPU 忙的時候做點別的事？不能：
   層 i+1 的 attention 依賴層 i 的殘差輸出（= `mmid_i` + shared expert），
   所以在 `mmid_i` 之前，**層 i+1 沒有任何節點可以提交**。
   → CPU 窗口內 GPU 必然沒有可執行的工作。
4. 而且雙緩衝**不改變段數**（`n_segs` 仍 = 層數），所以 39 次排空一次都不會少。

**結論：只要 CPU 還在迴路裡（讀 ids → 寫 leaf），往返就在關鍵路徑上，
雙緩衝只能省下 leaf 的「寫入」本身（µs 級），不能省下往返。**
量級也對得上：`cb+submit` 只佔 10–18%，而探針證明可移除的是 ~44% — 差額不在 CPU 窗口，
在**每個段邊界固定的啟動／回報延遲**（`gap` 大於 CPU 窗口那一段）。

---

## 4. 設計階梯

| | 做法 | 上限 | 是否值得 |
|---|---|---|---|
| **D0** | `CGC_SUBMIT_AHEAD=1`（已存在，已加進 `run_server.sh` allowlist） | **1.78×**（但輸出損壞） | 純量測，已做 |
| **D1** | 不變結構，把 CPU 窗口縮短（去掉 `slot_table_safe` 的間接、向量化、readback 改 staging） | ≤ `cb+submit` ≈ 10–18%，且**段數不變** | ✗ 性價比極差 |
| **D2** | 一個段放多層，段邊界從 40 降到 ~4 | **不可行**：host 寫 leaf 的段數上限就是層數（§3） | ✗ |
| **D2'** | 用預測路由把 leaf 提前寫好，一段多層 | 需要路由預測正確率接近 1 且要有回滾 | ✗ 投機複雜度過高 |
| **D3** | **把 expert→slot 查表搬到 GPU，消除段邊界** | 可吃到 D0 的 1.78×，且**輸出正確** | ✓ **本文的主設計** |

---

## 5. D3：GPU 端 slot 查表

### 5.1 機制

現況 `remap` 是 host 寫的 input leaf。改成 **graph 算出來的節點**：

```
slot_table_il : I32[n_expert]         ← host 維護的常駐裝置緩衝（每層一份）
slots         = slot_lookup(selected_experts, slot_table_il)   ← 在 GPU 上跑
mm_id_ids     = slots
```

- 新算子（`GGML_OP_CGC_SLOT_LOOKUP` 或 `ggml_map_custom` 起步）：`out[k, n_tok] = table[ids[k, n_tok]]`。
  這正是 hook 現在做的事（`llama-context.cpp:5142-5169`），只是搬到 GPU。
- `slot_table_il` 由 host 擁有並在**每次 residency 變動後**發布（雙緩衝 + generation 計數，
  或單調 generation + release store，讓 reader 只在 gen 變更時重讀）。
- `mm_id_ids` 隨之改為 `slots`（`llama-graph.cpp:2144` 的那一行換掉）。
- **Cold 語意不變**：不在 pool 的 expert，table 指向既有的 ZERO slot，
  與今天 fast path 的行為一致（`n_zero_mapped_selected` 這個計數器照舊會抓到）。

### 5.2 為什麼這樣才能吃到 1.78×

段邊界存在的**唯一**理由是「host 必須在 mmid 之前寫 leaf」。leaf 由 GPU 算之後：

- 圖中不再有 host 寫入 → **層與層之間沒有同步點** → 分段派發器可以不再為 remap 斷開，
  `n_segs` 由 40 降到 ~1（或保留少量段做別的事）。
- 39 次「等完成 → CPU → 提交」被消掉，而那正是 D0 量到的 44%。

### 5.3 Hook 的去向

residency 管理（eviction / fill / LRU）**必須**留在 CPU，但它不再是**依賴**：

- 今天：hook 必須在 mmid_i 之前完成（阻塞關鍵路徑）。
- D3：hook 只需要在某個時點之前把 table 更新好，落後就退化成 ZERO-mapping
  （貢獻被丟棄 = 品質代價，不是正確性代價，但要被計數與閘門約束）。
- 現成的形狀就是 `CGC_DBUF`（`llama-context.cpp:5188`）：用**上一步**的路由推測下一步的
  residency（步間路由穩定度約 87%）。D3 等於把這個思路變成主路徑。

### 5.4 實作切點

| 檔案 | 動作 |
|---|---|
| `llama-graph.cpp:2120-2127` | 建 op 取代 input leaf（保留 `ggml_set_output` 防 alloc 提前覆寫） |
| `llama-graph.cpp:2144` | `mm_id_ids = slots` |
| `ggml/src/…` | 新 op + Metal kernel（一個 1D gather kernel，`ne` 極小） |
| `llama-context.cpp:5142-5169` | 移除 leaf 寫入；保留 table 發布 + ZERO-mapping 計數 |
| `ggml-backend.cpp` | 段邊界不再需要為 remap 斷開（可整張圖一次提交，或保留段但去掉等待） |

### 5.5 分階段落地（每階段都要過閘門）

1. **S1 — table 化，段數不變。** leaf 改由 GPU 算，但派發器照舊分段。
   目的：證明 M1/M2/M3 + M2 oracle **bit-identical**（同 id → 同 slot → 同求和順序）。
2. **S2 — 去掉等待。** 段邊界不再 `cgc_done` 輪詢，改回 `CGC_SUBMIT_AHEAD` 式流水線
   （此時它是**正確的**：GPU 讀的是自己算出來的 ids）。
3. **S3 — 收段。** `n_segs` 由 40 收到 ~1–4，量 `gap` 是否塌到 0。

### 5.6 驗證與風險

**閘門（缺一不可）**
- M1 / M2 / M3 三支柱 + 最新 M2 oracle：**bit-identical**。這是硬閘門——D3 是排程改變，
  合法的實作必須產生同一組 slot。
- `CGC-GPUTIME`：`gap` 必須從 12–22 ms/步 塌向 0（成功的直接量測）。
- `CGC-DECPROF`：`wait` 下降量應 ≈ `cb+submit` + 被移掉的延遲。
- 品質：`zero_mapped_selected` / `n_fast_cold` 不得上升（否則就是拿品質換速度）。
- llama-bench 生產矩陣（pp/tg × depth）複測。

**風險**
- **galloc 生命週期**：leaf 從 input 變成 computed node，ggml-alloc 可能提前重用其緩衝
  （08-30 的 fused 病灶同型，見 `docs/…WHITEPAPER…` 的「根因 B」）。既有 `ggml_set_output` 是必要條件。
- **`graph_get_cb` 重指向**：L4 pool region 的重指向發生在 leaf 建立時；改為 computed 後必須重新推導。
- **dbuf 時序**：`llama-context.cpp:5177-5187` 的安全性論證建立在「hook 在段間跑」之上，D3 會推翻它。
- **MTP draft context**：draft ctx 目前比 verify 早一步複製同一套 hook，必須一起遷移。
- **ZERO-mapping 品質**：段邊界消失後，residency 的「及時性」下降是唯一的真實代價。

---

## 6. 尚未解決

- 探針只跑了 **1 round**（`--rounds 1`），且因為輸出退化、樣本數短。
  1.78× 必須用交錯 A/B ×3 + md5 複測才可寫進任何對外數字。
- `gap` 大於 `cb+submit` 的那 ~9 ms/步，目前只知是「commit→啟動 + 完成回報」，
  未逐項分解。若它在 D3 後仍存在，需要再切一次。
- `n_cb` 的 9 顆 buffer 幾乎序列（`sum/union ≈ 1.2–1.4`）與 `n_cb` 掃描無效互相印證，
  但「為什麼不並行」還沒定因。

---

## 7. S1 實作診斷記錄（2026-09-15 20:0x–20:4x）

S1（`CGC_SLOT_TABLE_GPU=1`）第一次實測**不是 bit-identical**，而且出現一天中唯一一次的
`libggml-cpu` SIGSEGV。以下是定位過程與三個新儀器，全部 env-gated、留下來可重用。

### 7.1 三個新儀器

| 儀器 | 位置 | 作用 |
|---|---|---|
| `GGML_SCHED_DEBUG=1\|2` | `run_server.sh` allowlist → `ggml-backend.cpp:2271` 讀、`:986` 印 | ggml 自己的排程器傾印：`## SPLIT #N: <backend> # M inputs`（=1）；逐節點 backend + `GET_CAUSE`（=2） |
| `CGC-MMID-ASSERT` 的 `ids_name/ids_op/ids_view_op/ids_data` | `ggml-metal-ops.cpp`（ids = `op->src[2]`） | 回答「mul_mat_id 到底在讀哪個張量、那個張量在哪個位址」 |
| `CGC_S1_MIN_IL` | `llama-graph.cpp` `build_moe_ffn`（預設 1） | 限制 GPU 查表適用的最低層號；layer 0 保留 host leaf |

> ⚠️ **`GGML_SCHED_DEBUG` 的沉默陷阱（已修）**：第一次跑 `=1` 得到 **0 行 `## SPLIT`**——
> 兩個輸出點都是 `GGML_LOG_DEBUG`，被 llama 預設 verbosity（INFO）濾掉。
> **0 行輸出與「只有 1 個 split」不可區分。** 已提升為 `GGML_LOG_WARN`（仍受 `sched->debug` 閘門）。

### 7.2 被否證的假設

**「S1 讓 MoE 節點跑到別的 backend」在 graph 粒度上不成立。** base 與 S1 都是 18 個 graph
build、每個 build 切成 `[CPU, MTL0, CPU(attn_post_norm 殘差), MTL0]`，`## SPLIT` 各 72 行。

### 7.3 唯一結構差異：layer 0

`GGML_SCHED_DEBUG=2` 逐節點，兩臂各一行就講完：

```
baseline : node #73 (MUL_MAT_ID) ffn_moe_gate-0 [CPU] … src[2] = ffn_moe_topk_remap-0  [CPU]
S1       : node #77 (MUL_MAT_ID) ffn_moe_gate-0 [CPU] … src[2] = CPU#ffn_moe_slots-0#  [NULL]
layer≥1  : node #180 ffn_moe_gate-1 [MTL0] … src[2] = ffn_moe_slots-1 [MTL0]（兩臂一致）
```

- layer 0 的 MoE FFN **在 CPU/BLAS**：它的專家權重仍是全尺寸（`blk.0.ffn_gate_exps` 82 M，
  其餘層 45 M → 未被 pool 收編），且 `LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=0` **並沒有**讓它上 GPU。
- baseline 的 ids 是 **CPU backend 的 graph input**——host 直接寫進 CPU kernel 要讀的 buffer，
  這個「可見性」是隱含契約。
- S1 的 ids 是 **Metal 算出來的**，於是必須跨 backend 拷貝，scheduler 把拷貝目的地記成 backend
  **NULL**。這正是 20:18 那次 SIGSEGV 的形狀（`ggml_compute_forward_mul_mat_id` on
  `libggml-cpu`，載入期 warmup）。
- 修法：`CGC_S1_MIN_IL` 預設 1，layer 0 保留 leaf。**仍然移除 40 次往返中的 39 次。**

### 7.4 真正的根因：**gather 輸出被 ggml-alloc 疊到同一個 buffer**

加上 provenance 欄位後第一次讀就命中（`Backup/cgc_logs/llama_server_20260915_203053.log`）：

```
ids_name='ffn_moe_slots-1'  ids_op=VIEW  ids_view_op=GET_ROWS  ids_data=0x128179d60
ids_name='ffn_moe_slots-2'  …                                 ids_data=0x128179d60
ids_name='ffn_moe_slots-4'  …                                 ids_data=0x128179d60
… 觀察到的 11 層裡有 10 層同址（只有 layer 3 是 0x1281f12a0）
```

→ **每一層的 gather 結果都落在同一個 16-int buffer 上**，最多只有一層的內容是對的。
這直接解釋觀測到的兩個怪象：

1. `id_oob=16/16` 幾乎每層都中，而 `first=` 的值是**上一個佔用該記憶體的 F32 張量**
   （`0x3F4EC2F8 ≈ 0.808` = router 機率、`0x7FC00000` = +NaN）——不是表內容錯，是**讀到別的東西**。
2. `CGC_S1_IDENT`（常數 0）「改了輸出但斷言還在」——所有層共用一個 buffer 時，寫 0 只影響
   最後寫入的那一層，其餘層讀到的仍是殘留。

**leaf 路徑從來沒有這個問題**，因為 `ggml_set_output(remap)` 把每層釘在各自的 buffer。
S1 漏了對等的保護。修法：對 `ggml_get_rows` 的**輸出**呼叫 `ggml_set_output`
（不是 reshape view——view 不單獨配置，旗標必須落在 ggml-alloc 真正 sizing 的那個張量上）。

### 7.5 教訓（可移植）

1. **同一個 graph 混用多個 backend 時，「誰算的、誰讀的」是契約**，不是實作細節。
   診斷順序應該是：先印 `## SPLIT` + per-node backend，再談內容。
2. **一個可以被靜默過濾掉的儀器等於沒有儀器**（`GGML_SCHED_DEBUG` 的 0 行）。
3. **一張 GPU 端張量的 `data` 指標是廉價而決定性的證據**——`ids_data` 一行就結束了兩天的猜測。
4. **凡是靠 `ggml_set_output` 取得「不被 ggml-alloc 疊掉」保證的張量，重寫時必須逐字照抄這個保證。**


---

## 8. S1 續做（2026-09-15 20:49–21:0x）：`id_oob` 被證明是探針假警報

§7 把根因歸給「gather 輸出被 ggml-alloc 疊用」，修完（`ggml_set_output(slots)`）斷言從 41 行降到 9 行，
但**剩下的 9 行沒有被解釋**，而且答案 digest 仍與基準不同。§8 把它解決掉，並推翻 §7 遺留的推論。

### 8.1 兩個新探針

**(a) `CGC-S1: POST`（post-sync 讀回）**——`llama-context.cpp` 的 `graph_compute` 尾端，在
`ggml_backend_sched_synchronize()` **之後**回讀同一批張量。新增捕捉 `ffn_moe_slots` 節點
（`cache_slots_out_tensors`）。輸出包含：gather 輸出、索引向量位址、表位址、leaf 值，
以及**完整的差分** `table[idx[j]] == gather[j]` 對所有 j 的比較。

**(b) `CGC-S1: EQUIV-<site>`（等價性差分）**——四個 remap 寫入點各呼叫
`cgc_s1_equiv_dbg()`，對**這一步選到的每一個專家**比較
`publish_slot_table()` 放進表的値 與 `slot_table_safe()`（＝ leaf 路徑的權威映射）的返回值，
並印出 `zero_slot`。

### 8.2 結論 1：`id_oob` 是**編碼期競態**，不是缺陷

`CGC-MMID-ASSERT` 在 Metal **encode 期**於 host 讀 `op->src[2]->data`。baseline 的 ids 是
host leaf（CPU 寫的），讀它沒有問題；S1 的 ids 是 **Metal kernel 算出來的**，encode 期那顆 kernel
還沒執行 → 讀到的是配置器留在那塊記憶體裡的殘值。

post-sync 讀回的證據（`Backup/cgc_logs/llama_server_20260915_204938.log`）：

```
EXPECT-pool il=1  ids=[193->2 105->105 229->7 249->9 220->5 106->106 181->1 84->84]
POST      il=1  gather=[2 105 7 9 5 106 1 84  0 103 0 74 121 110 3 99]
                gather_data=0x127d79d60  ids_data(assert)=0x127d79d60   ← 同一個位址
```

前 8 項與查表結果**逐項相同**，而且斷言讀的位址就是 gather 輸出的位址。
⇒ 那些 F32 機率位模式（`0x3F64E6C4 ≈ 0.893`）、`+NaN`、以及同一個 node 內合法/非法混雜，
**全部是「讀得太早」**。§7.2 的推論（「那些層的 gather 輸出根本沒被寫入」）因此被推翻。

**可移植的判準**：要在 CPU 上驗證 GPU 的產物，只有兩個合法時機——同步之後，或圖外。
dispatch-encode 期是第三個，而它不合法。

### 8.3 結論 2：`publish_slot_table()` 與 `slot_table_safe()` **只在有 ZERO slot 時才等價**

`CGC-S1: EQUIV-pool` 在 1443 個樣本上 `mismatch=0`——**但每一行同時印出 `zero_slot=-1`**，
即這些層**沒有保留 ZERO slot**。而兩個函式的差異正好落在這裡：

| 情況 | `slots[e]` | `slot_table_safe(e)` | `publish_slot_table()[e]` |
|---|---|---|---|
| e resident | ≥0 | slot | slot（一致） |
| e 非 resident，`zero_slot >= 0` | −1 | `zero_slot` | `zero_slot`（一致） |
| e 非 resident，`zero_slot < 0` | −1 | **−1**（非法索引，**吵**） | **0**（合法索引→別人的權重，**靜默**） |

prod25 profile 是第三列。`mismatch=0` 之所以成立，只是因為 verify-strict 讓**每一個被選到的專家
在每一次 hook 都是 resident**——那是當前 profile 的**偶然**，不是程式的不變量。

**⇒ 這是一條尚未踩到的地雷**：一旦某個被選到的專家變成非 resident，
leaf 路徑會吵（`id_oob`）、GPU 表會**靜默讀到 slot 0 的另一個專家**。
修法（S1 閘門之前必須處理其一）：

1. 對所有 GPU 表服務的層**強制保留 ZERO slot**（`zero_reserved_slot`），使兩條路徑重新等價；或
2. 把 `publish_slot_table()` 的 clamp 計數從 debug print 升級為**硬前置條件**（非 0 即 abort／記錄為閘門量）。

**監控量**：`clamped` 必須進 `CONVENTIONS.md` A 組的「閘門量」清單，與 `zero_mapped_selected`、
`n_fast_cold` 並列。

### 8.4 尚未解決（交給下一個 session）

- **S1 臂的答案 digest 跨輪不穩定，baseline 穩定。** 見 §9：層梯二分已把命題壓到
  「服務層數 ≥22 就分歧（20 層仍逐位元相同）」，但**層別**與**層數門檻**兩種讀法都還活著。
- bit-identical 閘門因此**仍未通過**。

---

## 9. 層梯二分與 KEEP_LEAF 對照（2026-09-15 21:0x）

### 9.1 兩梯的完整數據

`CGC_S1_MIN_IL=m` 讓 layer `m..39` 用 GPU 表（`m=1` 是預設）。**每個臂都是後綴**，
這是後面 §9.2 的關鍵。

rung 1（`s1_bisect_20260915.json`，`--rounds 2 --n-predict 24`）：

| 臂 | min_il | 服務層 | md5 | 形態 |
|---|---|---|---|---|
| `p25-gputime` | — | 0 | `dc055e63` | 基準 |
| `p25-slotgpu-l39` | 39 | 1 | `dc055e63` | 相同、穩定 |
| `p25-slotgpu-l38` | 38 | 2 | `dc055e63` | 相同、穩定 |
| `p25-slotgpu-l20` | 20 | 20 | `dc055e63` | 相同、穩定 |
| `p25-slotgpu` | 1 | 39 | `{29ca694a, b8c705cc}` | **不穩定** |

rung 2（`s1_bisect2_20260915.json`，`--rounds 3 --n-predict 24`）：

| 臂 | min_il | 服務層 | md5 | 形態 | 實際長度 | sample 前綴 |
|---|---|---|---|---|---|---|
| `p25-gputime` | — | 0 | `dc055e63` | 基準 | 24 | `<think>` 推理前綴 |
| `p25-slotgpu-l18` | 18 | 22 | `4991e6d2` | 不同、穩定 | 6 | `巴黎是法国首都。`（簡體） |
| `p25-slotgpu-l14` | 14 | 26 | `65bd254b` | 不同、穩定 | 7 | `巴黎是法國的首都。` |
| `p25-slotgpu-l10` | 10 | 30 | `1c5a05df` | 不同、穩定 | 22 | `巴黎是法國首都，因歷史演變…` |
| `p25-slotgpu-l6` | 6 | 34 | `7042e5e4` | 不同、穩定 | 12 | `巴黎並非法國的首都，法國的首都是巴黎。`（自相矛盾） |
| `p25-slotgpu-l2` | 2 | 38 | `{0d472bf5,51548f50}` | **不穩定** | 24 | `巴黎並非「法國的首都」，而是法國的首都。`（自相矛盾） |

### 9.2 方法論：後綴層梯**結構上**無法區分「某一層壞」與「層數超過門檻」

`CGC_S1_MIN_IL` 是**前綴閘**，所以每個臂服務的都是**連續後綴**：左界與「服務層數」
永遠同步移動，無法獨立變動。數據因此同時吻合兩種模型：

- **層別模型**：layer 18 或 19 的映射是錯的（`l20` 不含它們 → 相同；`l18` 含 → 不同）。
- **計數／門檻模型**：服務層數 ≥21 或 22 就分歧（20 層相同、22 層不同）。

再加幾梯同樣形狀的臂**不能**分開它們——它只有一個自由度，只會把邊界再挪一格。
**不要**把 `{18,19}` 當成答案引用（`decisions.jsonl` 的
`dec-20260915-2120-suffix-ladder-cannot-separate-layer-from-count`）。

### 9.3 這批數據**額外**提供的（與上述歧義無關）

所有分歧臂**從第 1 個字元**就與基準不同（`sample` 是 `texts[0][:160]`，與長度無關），
而且 `l6`/`l2` 產生**自相矛盾**的句子；三個與基準相同的臂則都生成**正好 24** 個 token
（與基準同長）。這是**生成模式的差異**，形態上更像 FFN 讀到**別的專家的權重**，
而不是最後位元的 logit 位移。但這是旁證、不是證明：嚴重度**不隨**服務層數單調
（`l18` 服務 22 層最糟，`l2` 服務 38 層反而流暢），所以不能拿它去否證門檻讀法。

### 9.4 儀器的第二個缺陷：掃描視窗比假設窄

今天的兩個 host 側探針**都寫死在 layer 0..3**：

- `llama-context.cpp` 的 POST 讀回：`for (int il = 0; il < 4; ++il)`。
- `cgc_s1_expect_dbg()`：`if (!on || il > 3) return;`，且總共只印 8 行。

而分歧在 **18..19**。所以「探針說映射正確、閘門卻不過」不是矛盾：**探針從未看過缺陷所在的層**。
一個掃描範圍比假設窄的儀器無法否證該假設，它的沉默會被讀成確認——這是 B2 的上一層
（B2 是「什麼都沒印」，這裡是「印了真話，但關於錯的地方」）。已改為走訪**全部**被捕獲的層。

### 9.5 還有一個被靜默關掉的儀器

`cache->n_zero_mapped_selected`（QUALITY LEAK 計數）**唯一**的加總點在 leaf 寫入區塊內；
S1 不建 leaf，於是 40 層只剩 layer 0 在計數，teardown 印出的 `0` 與「真的沒有 leak」不可區分
（見 `lessons.jsonl` 的 `eng-gate-0003`）。

### 9.6 KEEP_LEAF 對照：切開「映射」與「節點」

`CGC_S1_KEEP_LEAF=1` 建**所有** S1 節點（table / CONT / GET_ROWS / VIEW）**並且**建 host leaf，
然後讓 `mul_mat_id` 消費 **leaf**。於是圖裡帶著完整的 S1 額外節點，ids 卻仍來自 host。
配合已修好的 POST 視窗（現在走訪全部層並印 `same=`，因為 leaf 與 gather 都是 `set_output`
所以都還在），三種結果互斥且可分辨：

| keepleaf md5 vs 基準 | 逐層 `same=` | 讀法 |
|---|---|---|
| 相同 | 全部 1 | 映射**正確**、節點無害 → 分歧來自「leaf 的存在與否」造成的佈局差異 |
| 相同 | 某層 0 | **映射錯**，且 `same=0` 的那層就是缺陷位置 |
| 不同 | 全部 1 | 節點本身會動答案 → 映射從來不是問題 |

### 9.7 量測衛生：digest 綁在**實際生成長度**

同一批數據裡各臂的實際長度是 **24 / 24 / 12 / 22 / 7 / 6**（同一個 `--n-predict 24`）。
`decode_bench.py` 是 `md5(整段 text)`，所以「digest 不同」在那些列上**同時**被長度混淆，
單看 digest 分不清「軌跡分歧」與「同樣前綴、只是提早停」。與長度無關的證據是 `sample` 前綴。
（另一層：`--n-predict 24` 與 `--n-predict 40` 之間 digest 本來就不可比，即使兩邊每個 token 都一樣。
見 `CONVENTIONS.md` A5 與 `eng-mh-0004`。）

### 9.8 KEEP_LEAF 結果：**節點是惰性的**，而 ids **也不是**原因

`CGC_S1_KEEP_LEAF=1`（建所有 S1 節點 **且** 建 host leaf，`mul_mat_id` 消費 leaf）
在 `s1_keepleaf2_20260915.json` 的結果：

| 臂 | md5 | 穩定 | n | 判定 |
|---|---|---|---|---|
| `p25-gputime` | `dc055e63` | ✓ | 24 | anchor |
| `p25-keepleaf` | `dc055e63` | ✓ | 24 | **逐位元相同** |
| `p25-keepleaf-dbg` | `dc055e63` | ✓ | 24 | **逐位元相同** |

pool 計數也與 anchor 完全相同（`misses=13040`、`file_reads=36360`）。
⇒ **39 層 × 4 個額外節點（共 156 個）對數值零影響**，且**不存在服務層數門檻**
（keepleaf 服務全部 39 層）。這回頭也**解除了 rung 2 的歧義**：門檻效果被排除後，
`min_il=20` 相同 + `min_il=18` 不同 ⇒ 缺陷在 layer 18 或 19。

接著用修好視窗的 POST readback 逐層比 gather vs leaf（39 層 × 每層 6 個樣本）：

- **token 0：每一層、每個樣本都完全相同**（0 個差異）。
- **row ≥1：每一層都是 6/6 不同**。

例（il=1）：
```
gather = [2 105 7 9 5 106 1 84 | 0 103 0 74 121 110 3 99]
leaf   = [2 105 7 9 5 106 1 84 | 8   0  50 4  20   6 55  3]
```
前 8 項完全一致（`[k, n_tokens]` token-major 佈局 ⇒ 那是 token 0），後 8 項才分歧（token 1）。

**row ≥1 的差異是 padding，且已被獨立證明無害**：它在**每一層**都出現（不是層別缺陷的形態），
而 `min_il=20` 攜帶 20 層的它卻仍逐位元相同。⇒ **真正有意義的 row 上，GPU 算的 ids 與 host 寫的一致。**

**注意陷阱**：POST 的 `same=` 是**整段向量** memcmp，因為 row ≥1 而**每一行都印 0**——
只看它會得到「每一層、每一處都不一致」這個完全相反的結論。必須按 row 拆開
（`eng-mh-0005`）。

### 9.9 結論：兩個原有解釋都死了，剩下的唯一差別是「leaf 節點不存在」

- **節點不是原因**（keepleaf 全建仍相同）——⚠️ **這個結論只在分段 dispatcher 下成立**，見 §9.15。`§9.9` 其餘推理的前提（節點惰性）因此要重新加註。
- **ids 不是原因**（有意義的 row 上逐層一致；差異的 row 是 padding 且證明無害）。

所以真臂與逐位元相同的控制臂之間，**唯一剩下的差別是那個 leaf 節點不存在**。
而 `llama-context.h` 對 `cache_slot_table_tensors` 的註解**一直聲稱 S1 就是這樣做的**：

> The leaf itself is still built when this is on: it is simply not consumed. That keeps every
> other path bit-identical and makes the S1 A/B a pure scheduler/dispatch change.

**但實作從來沒這樣做**——`llama-graph.cpp` 的 S1 分支是 `if/else`，開了表就不建 leaf。
⇒ 下一步不是發明新設計，而是**量「照文件做」的版本**：`CGC_S1_BUILD_LEAF=1`（建 leaf、不消費它）。
若它逐位元相同，修法就是「讓實作符合自己的文件」，順帶也修好 §9.5 那個被靜默關掉的計數器。

---

### 9.10 「照文件做」的版本**跑不起來**——而這本身就是結論

`p25-buildleaf` 連一個 token 都沒生成就 abort 了（`Backup/cgc_logs/llama_server_20260915_212420.log`）：

```
ggml-alloc.c:623: GGML_ASSERT(buffer_id >= 0) failed
  ggml_gallocr_allocate_node <- ggml_gallocr_reserve_n_impl
  <- ggml_backend_sched_reserve <- llama_context::graph_reserve
  <- llama_context::sched_reserve <- llama_context::llama_context <- llama_init_from_model
```

原因不是寫錯，而是**那個設計在本 allocator 裡不可構造**：

- `slots` 既不被消費（`remap_ids` 沒動）也沒被 expand ⇒ CONT / GET_ROWS / VIEW 全部 orphan，不在
  `cgraph->nodes` 裡；
- `slot_table` 只因為上面那行**無條件**的 `ggml_build_forward_expand(gf, slot_table)` 才留在圖裡；
- 新建的 `remap` leaf 也被 expand 進去了。

⇒ 圖裡出現**兩個沒有任何 consumer 的 root**。排程器不會給「沒人讀的 root」指派 backend，
ggml-alloc 於是在 `buffer_id = -1` 上斷言。

**所以那份註解錯了兩次**：分支是 `if/else` 所以 leaf 不建（`eng-gate-0003`），而「建了不消費」
也不可行（`eng-diag-0018`）。這條路要劃掉，不是「之後再試」。

還有一個推論很重要：**唯一可實現的「圖裡有 leaf」的配置就是 keep-leaf 控制臂**
（leaf 被 `mul_mat_id` 消費），而那正是已經逐位元相同的配置。因此真臂與每一個逐位元參考之間的差別，
**不可避免地**就是「那個節點不存在」及其對圖／allocator 的後果。

---

### 9.11 真臂的 ids 也是對的（這次是在**真臂**上量的）

上一輪的逐層讀回是在 `p25-keepleaf-dbg` 上做的——而那個臂的定義性質就是 `mul_mat_id` 消費 **host leaf**，
所以它「驗證了 gather」是在一個**不消費 gather** 的圖裡（記為 `eng-mh-0006`）。
這次補上真臂：`p25-slotgpu-dbg`（**沒有 leaf**，POST 每行都印 `leaf=[n/a]`）。

| 儀器 | 樣本數 | 結果 |
| --- | --- | --- |
| `CGC-S1: EQUIV-*`（bulk `publish_slot_table` vs 逐查詢 `slot_table_safe`，對每個被選專家） | **2067** | **全部 `mismatch=0`**，且每行 `zero_slot=-1` |
| `CGC-S1: POST`（post-sync 的 gather）vs `CGC-S1: EXPECT`（hook 當時的表值） | 39 層 × 6 個 graph build | **每行都一致** |
| `CGC-SLOT-TABLE`（clamp 計數） | 當日全部 S1 日誌 | **0 行** |

例（il=1，第 0 組）：

```
EXPECT ids=[193->2 105->105 229->7 249->9 220->5 106->106 181->1 84->84]  ⇒ [2 105 7 9 5 106 1 84]
POST   gather=[2 105 7 9 5 106 1 84 | 0 103 0 74 121 110 3 99]            leaf=[n/a]
```

⚠️ POST 那些 `MISMATCH` 行**不是**證據：它們的 index 是從 `ids_cont` 讀的，而 `ids_cont` 不是 output，
buffer 在 gather 消費後就被 ggml-alloc 回收，所以讀到的是垃圾（`k=0 idx=4`，而該層 ids 是
`[193 105 229 249 …]`）。探針自己就為此印 `ids_src_valid`。

⇒ **值層面的解釋全部死透了**：節點惰性、ids 正確、bulk 與 safe 不漂移。命題只剩「那個 tensor 的
身分／時序」，不是它的內容。

---

### 9.12 OA_ASYNC 配對是**空操作**：開關的值從來沒被讀

原本以為「關掉分段 dispatcher」是剩下的唯一乾淨實驗（分段 dispatch 的等待點綁在 leaf 寫入上）。
結果兩個臂**逐位元複製**了它們的分段對應臂：

| 臂 | md5 集合 | misses | file_reads |
| --- | --- | --- | --- |
| `p25-gputime`（分段） | `{dc055e63}` | 13040 | 36360 |
| `p25-gputime-noasync` | `{dc055e63}` | 13040 | 36360 |
| `p25-slotgpu`（分段） | `{29ca694a,672585db,b8c705cc}` | 14296 | 39948 |
| `p25-slotgpu-noasync` | `{29ca694a,672585db,b8c705cc}` | 14296 | 39948 |

池計數是**路徑相依**的量，兩個不同的 dispatcher 要巧合落在同一組 `14296 / 39948` 上不現實。原因在
`ggml/src/ggml-backend.cpp:1745`：

```cpp
} else if (getenv("CGC_OA_ASYNC") != nullptr &&      // ← 選的是「有沒有設」，不是值
           getenv("CGC_VERIFY_OP_TIMING") == nullptr && ...) {
```

`!= nullptr` 只判**存在**；`CGC_OA_ASYNC=0` 與空字串（`getenv` 回 `""` 而非 `NULL`）都選到分段分支。
而 `run_server.sh` 無條件把 `CGC_OA_ASYNC` 放進 `SERVER_ENV`，所以**連「不設它」都做不到**。

這是 `run_server.sh` allowlist 陷阱的鏡像：那裡是「設了被靜默丟棄」，這裡是「設了但**值被忽略**」——
兩者都讓「沒效果」與「沒接上」同形（`eng-gate-0005`）。

**已修**：閘門改為 value-aware（未設／空／非零首字＝分段；`"0"`＝關）。prod25 設 `"1"`，
所以**既有 digest 不會被移動**（錨 `dc055e63` 不變）。

**第二次跑才是有意義的那一次**（`s1_noasync2_20260915.json`，21:38–21:44，閘門已修）：

| 臂 | dispatcher | md5 集合 | 穩定 | misses | file_reads | t/s |
|---|---|---|---|---|---|---|
| `p25-gputime` | 分段 | `{dc055e63}` | ✓ | 13040 | 36360 | 6.95 |
| `p25-gputime-noasync` | 非分段 | `{dc055e63}` | ✓ | 13040 | 36360 | **0.72** |
| `p25-slotgpu-noasync` | 非分段 | `{ff68c5a2}` | ✓ | 16717 | 47574 | 0.61 |

四件事同時成立：

1. **對基準而言 dispatcher 是數值中性的**（md5 與兩個池計數全同）。這既是「閘門真的被讀了」的證據，
   也是分段／非分段等價性的確認。
2. **時序假說被否決**：真臂在非分段下**仍然錯**，而且錯成一個**全新的穩定值** `ff68c5a2`
   （不在分段 S1 集合 `{29ca694a, 672585db, b8c705cc}` 內，也不在錨裡）。「移除段邊界 hazard
   就會過閘門」這個預測沒有實現。
3. **問題的形狀變清楚了**：分段下真臂**不穩定**（3 輪 3 值），非分段下**確定但錯**（1 值）。
   ⇒ 不確定性**來自分段 dispatcher**，而它底下還有一個**確定性缺陷**。
4. 關閉分段的代價是 **~10×**（6.95 → 0.72 t/s）。這不是候選吞吐數字，而是 S2/S3 要追回的獎金大小。

⚠️ **這兩次跑（修閘門前後）帶完全相同的建置指紋**，因為閘門住在 `ggml-backend.cpp`，而它編進
`libggml-base`——**不在指紋的三個檔案裡**。見 §9.14.2。

---

### 9.13 現況與下一步

**已排除（有數據）**：壞層假說（keepleaf 全建仍相同）、層數門檻、ids 映射值、bulk/safe 漂移、
節點存在本身、`mul_mat_id` 的 backend 指派（§9.14.2 的 =2 dump：layer 0 以外兩臂逐節點相同）、
**段邊界的提交時序**（§9.12 第二次跑）。

**剩下的唯一候選是「消費的那一刻，那個 buffer 裡是什麼」**——不是它的最終值，而是 consumer
讀它時的値。三個現有儀器都答不了這個問題，而且都會吐出「看起來像答案」的數字：

| 儀器 | 它實際量到什麼 | 為什麼不能回答本問題 |
| --- | --- | --- |
| `CGC-S1: EXPECT` vs `POST` | sync **之後** gather 的輸出 | 之後正確不代表之前正確（§9.11） |
| `CGC-MMID-ASSERT` 的 `id_oob` | encode 期**主機端**讀一個**裝置寫入**的 buffer | 量到的是寫入前的殘留（§9.14.1） |
| `CGC-S1: EQUIV` | 主機側兩個映射實作的一致性 | 兩個主機實作一致，與裝置讀到什麼無關 |

⇒ 需要一個**內核側**的儀器：讓 `mul_mat_id` 的 Metal kernel 把它**實際解碼出來的 ids** 寫進一個
pinned debug buffer，sync 之後讀回。這是唯一能同時切開「時序」與「allocator 別名」兩個子假說的
讀數，而且它不依賴任何主機端推論。

**不要**再加同形狀的層梯（§9.2）、**不要**重加 `p25-buildleaf`（§9.10）、**不要**再引用 `id_oob`
當證據（§9.14.1）。

---

### 9.14 三個「有數字、沒意義」的量測假象

#### 9.14.1 `id_oob` 是 encode 期主機讀一個裝置寫入的 buffer

`CGC-MMID-ASSERT` 在 `ggml-metal-ops.cpp` 的 `ggml_metal_mul_mat_id` **encode 路徑**裡直接
`((int32_t *) op->src[2]->data)[i]`。當 `src[2]` 是主機寫入的 leaf 時，這讀數是有效的；一旦
`src[2]` 是**裝置算出來**的 gather 輸出，它量到的就只是「**這一步的 kernel 跑之前**，那塊 buffer
裡躺著什麼」。

| 臂 | dispatcher | ids 誰寫 | `id_oob` 讀數 | 該臂的輸出 |
|---|---|---|---|---|
| `p25-gputime-noasync` | 非分段 | 主機（leaf） | **0 / 240** | 正常 |
| `p25-slotgpu-noasync`（第一次，實為分段） | 分段 | 裝置（gather） | **202000+** | 可讀法語（非亂碼） |
| `p25-slotgpu-noasync`（第二次，真非分段） | 非分段 | 裝置（gather） | **0 / 288** | 可讀法語（非亂碼） |

同一個臂在兩個 dispatcher 下，一次是 0、一次是 20 萬；兩次的輸出都是**可讀的法語**，不是亂碼。`first=1063585220`
（`0x3F65…`）與 `1058604036`（`0x3F19…`）都是 **F32 的路由器機率**（≈0.90 / ≈0.60）——即「別人的
殘留」。⇒ 這個讀數只區分「主機寫的 ids（永遠 0）」與「裝置寫的 ids（任意）」，而這件事
`ids_name` / `ids_op` / `ids_data` 三個欄位已經**可靠地**回答了。把它當成「ids 壞掉了」的證據，
是把**身分判準誤讀成數值判準**；基準的 0 格外具有誤導性，因為它看起來像「基準是乾淨的」。

#### 9.14.2 建置指紋漏了 `libggml-base`

`build_fingerprint()`（`decode_sweep.py:33`；`ab_interleave.py:42` 是同一份邏輯）只雜湊三個檔案：

```
("server", "src/llama.cpp/build/bin/llama-server")
("metal",  "src/llama.cpp/build/bin/libggml-metal.0.19.0.dylib")
("llama",  "src/llama.cpp/build/bin/libllama.0.0.239.dylib")
```

而今天的閘門修正在 `ggml-backend.cpp`，它編進 **`libggml-base.0.19.0.dylib`**
（`build/ggml/src/CMakeFiles/ggml-base.dir/ggml-backend.cpp.o`），**沒有被雜湊**。本專案的規則正是
「只在同一指紋內比較」，於是 21:34 那次（閘門未修、兩臂逐位元複製分段臂）與 21:41 那次
（閘門已修、`{ff68c5a2}`）被標成**可比較**——而它們的排程器程式碼不同。

次生缺陷：檔名**寫死版本號**（`libllama.0.0.239.dylib`），版本一動，指紋就靜默變成 `"missing"`
（`"missing"` 是可見的，但「看不見變化」與「看見變化」的界線就此位移）。

修法沿用 repo 內已有的做法（`knifeedge_matrix.py:483` 用 `libllama*.dylib` / `libggml*.dylib` glob）。
**這會改變指紋的形狀**：v1（三鍵）的歷史列在「libggml-base 這一維」上不可比——這正是要揭露的事實。

#### 9.14.3 POST 的 `MISMATCH` 行

同一家族的第三個：`ids_cont` 不是 output，buffer 已被回收，讀到垃圾，於是每一行都印 `MISMATCH`。
已記於 §9.11，此處只做索引。三個假象的共同形狀是：**儀器量的東西與它宣稱量的東西不是同一個**，
而它**沉默或吵鬧都與真相無關**。

---

### 9.15 決定性配對：**「節點是惰性的」只成立於分段 dispatcher**

`p25-keepleaf-noasync`（S1 節點全建、`mul_mat_id` 仍消費 **host leaf**、dispatcher 關掉）
在 `s1_keepleaf_noasync_20260915.json` 的結果，與同一個 sweep 的基準併列：

| 臂 | dispatcher | md5 集合 | 穩定 | misses | file_reads |
|---|---|---|---|---|---|
| `p25-gputime-noasync` | 非分段 | `{dc055e63}` | ✓ | 13040 | 36360 |
| `p25-keepleaf-noasync` | 非分段 | **`{8c870183}`** | ✓ | 15117 | 42801 |
| `p25-keepleaf`（21:19，分段） | 分段 | `{dc055e63}` | ✓ | 13040 | 36360 |

⇒ 同一個控制臂，**只換 dispatcher**：分段下逐位元相同，非分段下**穩定地不同**。
`decode_sweep.py` 裡對這個臂**事先寫下**的判讀正好命中第二支：

> `keepleaf-noasync different -> the S1 nodes are only inert under the segmented dispatcher [...]`
> `and would move the suspect back onto the nodes themselves.`

所以 §9.9 的「節點不是原因」必須降級為**有條件的**：節點惰性不是節點的性質，是
（節點 × dispatcher）的性質。§9.13 的候選 2（「少一個 pinned 節點造成 allocator 佈局差異」）
也失去它的結構前提——**leaf 在場且被消費**時，非分段下仍然分歧。

兩個附帶事實同樣重要：

1. **差異的形態不是最後位元**。基準的輸出是 `<think> Here's a thinking process:` 的英文推理前綴；
   keepleaf-noasync 的輸出是「這是一個基於錯誤前提的問題。1. **巴黎（Paris）** 是 **法國（France…」
   ——**從第一個 token 就是另一種回答模式**。任何「最後幾位浮點數」的解釋都不能產生這個。
2. **池計數跟著動**：`13040 → 15117`（+16%）、`36360 → 42801`（+18%），而兩個臂的**節點集合**
   只差那些「算了但沒人消費」的節點。⇒「同一張圖的節點集合相同」**不蘊含**「專家快取的存取模式相同」。

**可比較性的誠實邊界**：`keepleaf-noasync`（21:49）與 `keepleaf`（21:19）的對照跨越了 21:36 的重建。
這個比較在此**可接受**，理由必須寫明：閘門修正對這兩個臂都不可能改變分支——prod25 會設
`CGC_SERVER_OA_ASYNC=1`（由基準配對證明：6.95 vs 0.72 t/s、同 digest），`p25-keepleaf` 沿用該預設
（修正前後都是分段），而 `p25-keepleaf-noasync` 顯式設 0（修正前後都是非分段）。三鍵指紋看不出這件事，
那是另一個缺陷（§9.14.2）。

---

### 9.16 「讓閘門讀值」是一次行為變更，不只是修一個 bug

把 `CGC_OA_ASYNC` 從「讀存在」改成「讀值」之後，**只有 `prod25` 不受影響**——因為它是唯一顯式設 `1`
的 profile。其餘全部繼承 `run_server.sh:109` 的預設 `0`，而在修正前那個 `0` 從來沒有生效過
（閘門讀存在，腳本又無條件塞入），所以它們一直是分段。修正後它們第一次真的變成非分段：

| profile | 修正前 `CGC_OA_ASYNC` | 修正前實際 | 修正後實際 |
|---|---|---|---|
| `prod25` | 1（第 265 行顯式） | 分段 | 分段 |
| `off`（預設）/ `prefill250` / `qa-zh` / `longform-zh` | 0 | **分段**（值被忽略） | **非分段（~10× 慢）** |

`prefill250` 正是 oracle 閘門與 bench matrix 跑的口徑，所以影響落在**驗證鏈**上。
已修：`run_server.sh:109` 的預設回到 `1`（= 那些配置**實際**在做的事），顯式 `CGC_SERVER_OA_ASYNC=0`
仍然可用（S1 診斷需要的那一格）。乾跑驗證五個 profile 加顯式 0，`bash -n` 通過；prod25 錨 `dc055e63`
不受影響。教訓：`eng-gate-0006`（**錨 digest 不變 ≠ 行為不變**——顯式設了那個現在才有意義的值的那個
profile，正是修正不可能移動的那一個，所以它是最差的證人）。

---

### 9.17 D3（把 expert→slot 查表搬到 GPU）框架下，submit-ahead 的定位與 S2 的必要性

D3 的形態：`remap` 現在是 **host 寫的 input leaf**（`llama-graph.cpp:2122-2126`），改成 graph 算出來的
`slot_lookup(selected_experts, slot_table)`（消費點 `:2144`），slot table 由 host 維護並發布。
若圖內不再有 per-step 的 host 寫入，則**段邊界失去存在理由**，`n_segs` 由 40 收到 ~1。

**（a）submit-ahead 不是被廢棄的方案，它一天都不曾是方案。** `CGC_SUBMIT_AHEAD=1` 是**故意錯**的
ceiling probe（在 segment i 的 top-k hook 寫 remap leaf **之前**就提交 segment i+1，那正是現行順序
存在的理由），輸出**預期 corrupt**。它的作用是量具，不是 candidate。

**（b）但 D3 改變了它的量測對象。** 舊框架問「能不能**安全地跨段重疊**」，D3 問「能不能**根本不分段**」。
⇒ submit-ahead 的數字由「S2 的天花板」變成「**段邊界的總價值**（= S3 的天花板）」。它不廢棄，
其數字成為 D3 的**驗收參考**。而 D3 明確優於它：D3 正確（無 race），submit-ahead 的 race 修不好。

**（c）S2 在 D3 下很可能是繞路。** S2（去掉等待）是「**保留分段**、只是不等」的產物；一旦接受
「段邊界失去理由」，正確動作是「不要分段」而非「去掉等待」——`n_segs → ~1` 之後等待自動消失。
⇒ 建議 **S1 → S3**，S2 只作 fallback。

**（d）但「圖裡不再有 host 寫入」不完全成立——這是 D3 唯一的前提風險。** `slot_lookup` 的 table
**本身仍是 host 寫的 tensor**（`publish_slot_table`）。所以段邊界失去理由的真正條件是：

> **`slot_table` 的發布不在熱路徑上。**

若每個 decode step 都要更新它（expert 被換入換出），host 寫入**還在**，只是從 per-step 的 remap
換成 per-step 的 slot_table ⇒ 段邊界仍然需要，D3 的推論塌。**待測**：`publish_slot_table` 每 step
被呼叫幾次。注意現有的 `CGC-SLOT-TABLE` clamp 計數（全天 0 行）**不能**回答這個問題——它只在
clamp 發生時觸發，0 行只證明「沒有 clamp」，不證明「沒有更新」。必須新增一個發布次數的計數。

**（e）一個新論據支持 D3（來自 §9.13 的補實驗）。** 分段 dispatcher **本身引入不確定性**：S1 真臂在
分段下同 build、同 N、**3 輪 3 值**；同一臂在非分段下**穩定單值**。⇒ **收段不只省時間，還消除一個
不確定性來源**，讓 S1 的 bit-identical 閘門從「擲骰子」變成「可判定」（穩定失敗 vs 隨機假通過）。
這是 D3 原本沒被算進去的收益，且它直接決定 S1 的閘門**能不能有意義地跑**。

**（f）風險清單重排（按「是否為前提」）。**

| # | 風險 | 性質 | 處置 |
|---|---|---|---|
| 1 | `slot_table` 發布頻率（= dbuf 時序假設） | **前提** | 先量，最便宜，決定 (c) 是否成立 |
| 2 | MTP draft context 需同步遷移 | **範圍** | 第二個 context 也有 host 寫入 ⇒ 問題複製 |
| 3 | galloc 生命週期 | 實作 | 已有可行模式：`ggml_metal_buffer_init(size, shared)` 建的 buffer **不受 allocator 管理**，正是 slot_table 需要的（本輪為 ids capture 已驗證該模式可編譯可跑） |
| 4 | `graph_get_cb` 重指向 | 實作 | 純接線 |

**（g）一個必須避開的數字陷阱。** 本輪測到「非分段 `CGC_OA_ASYNC=0` = 0.72 t/s vs 分段 16.17 t/s」，
**不能**用來推論「D3 收段會慢 ~20×」。那是 **40 個 split + 序列化 dispatcher**；D3 的收段是
**~1 個 split**。兩者層次不同：`CGC_OA_ASYNC` 是**調度策略**（如何在既有 split 間調度），D3 動的是
**split 數**。收段後 split=1，dispatcher 無事可調度。所以那 44% 的歸因要小心，別把「調度策略的價值」
誤讀成「分段本身的價值」。

---

### 9.18 內核側 ids 讀數：ids 是**逐位元相同**的，缺陷在它們**指向的東西**

#### 9.18.1 儀器

`kernel_cgc_ids_capture`（`ggml-metal.metal`）：一個 1 threadgroup × 1 thread 的 kernel，
**緊接在消費 kernel 之後、提交進同一個 command buffer**，把 ids operand 的前 8 個 int32 寫進
一個**不受 allocator 管理**的 shared buffer（`ggml_metal_buffer_init(size, shared=true)`），
dump 掛在 `ggml_metal_synchronize` 尾端（只印未印過的 slot，游標式）。

七個檔案：`ggml-metal-impl.h`（kargs）、`.metal`（kernel）、`device.h/.cpp`（pipeline getter）、
`device.m`（`ggml_metal_buffer_get_id_whole`；**`.m` 是 C 模式編譯，不能用 `nullptr`**）、
`ops.cpp`（state + MV/MM 兩處提交）、`context.m`（dump 掛點）、`run_server.sh`（**allowlist**）。

**兩個設計要點**：不改主 kernel、不改 graph ⇒ 不擾動 allocator 佈局（`eng-diag-0018`）；
`map0` 的 `hids` 是 `src2` 的確定性函數（存的是索引 `(i21+t)*ne20 + sel - 1`）⇒ MV 與 MM
**共用同一個讀數點**。

#### 9.18.2 同一筆執行，兩個探針給出相反答案

| 探針 | 時機 | 讀數 | 判定 |
|---|---|---|---|
| `CGC-MMID-ASSERT`（主機） | **encode 期** | `id_oob=16/16 first=1063585220` | 垃圾（`0x3F65…` = F32 0.895，router 機率 = 前一位佔用者） |
| `CGC-IDS-CAP`（內核） | **消費後** | `[2,105,7,9,5,106,1,84]`，全在 `[0,143)` | 合法 |

`ids ne=[8,2]`、`ne21=2 < ne21_mm_id_min=32` ⇒ 全程走 **MV**（36 graphs × 117 = 4096 slots，
`path=MM` **0 次**；本批 batch 最大 8 token）。

#### 9.18.3 與基準臂逐 node 比對（`scripts/check/ids_capture_diff.py`）

| graph | SAME | DIFF | 分佈 |
|---|---|---|---|
| 0 / 1 / 2 | **117 / 117** | 0 | 全部 39 層逐位元相同 |
| 3 / 4 | **3** | **114** | 只有 layer 1 相同；第一個 DIFF = `ffn_moe_gate-2` |

⇒ **S1 的 host leaf 與 GPU `get_rows(table, selected_experts)` 在共享輸入的 graph 上逐位元相同**
（39 層、117 node、連續三個 graph）——這正是 S1 宣稱的 bit-identity，**成立**。

#### 9.18.4 缺陷定位

graph 3 的 **layer 1 gate/up/down ids 相同**，而 **layer 2 的 router 不同**。layer 1 的 router ids
相同 ⇒ layer 1 的 **輸入（attention 輸出）與 router logits 相同**；layer 1 的 MoE ids 也相同。

⇒ 在**輸入相同 + ids 相同**的前提下，**layer 1 的 MoE 輸出不同**。
⇒ 缺陷**不是 ids**、**不是 expert→slot 映射**，而是 **ids 指向的權重內容**。

嫌疑集合收斂到**池／slot 的內容本身**（`publish_slot_table` vs `slot_table_safe` 的 clamp 分歧，
以及表與它所描述的池狀態之間的時序）。

#### 9.18.5 儀器的已知限制

capture 讀固定 **stride=8**，所以 `n_ids=64` 時**只有 token 0 的 ids 可見**。若分歧只出現在
batch 的 token 1..7，這裡**看不到**——「第一個 DIFF 是 layer 2」這個結論是關於 token 0 的。
要觸及尾部需用非零 `n_skip` 再跑一次。

#### 9.18.6 下一步（唯一的決定性測量）

**capture `ffn_moe_down-1` 的輸出**（兩臂、graph 3）。若在輸入與 ids 都相同的情況下輸出不同
⇒ **池內容被證明是載體**，調查從「映射層」移到「residency／publish 層」。

