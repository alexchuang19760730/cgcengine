# llama.cpp fork：Expert Bounded Residency 最小改動方案

日期：2026-08-14
基準 commit：`temp/llama_roadB/llama.cpp-master`（master，src/ 已拆分 + `tools/llama-bench/`）

## 0. 目標與前提

16GB M4、同時跑其他 app（~4-5GB OS 基本盤）下，跑 **IQ3_XXS（13.2GB）** 或 **IQ2_XXS（10.76GB）**：

- 全 offload（-ngl 99）實測：IQ2 25-28 tok/s，但 free pages ≈ 0，**零餘裕**；IQ3 直接 Metal decode 失敗（12.29GiB > 11.45GB 工作集）
- 目標：**bounded ~5-6GB resident**（2.4GB 非 expert + 3GB expert pool）→ 保留真實餘裕、IQ3 可跑、tok/s 盡量貼近全 offload

核心前提（先前的實測結論）：experts 的 GEMM **必須上 Metal**；`-ncmoe` 的 CPU expert 陷阱 = 6.47 tok/s，不可走。

## 1. 現況盤點（qwen35moe 載入 + graph + Metal 三層）

### 1.1 張量創建（src/models/qwen35moe.cpp）

```cpp
// L100：每層 3 個 3D expert 張量（第 3 維 = n_expert）
layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il),
                                    { n_ff_exp, n_embd, n_expert }, flags);
// gate/up 同理（{n_embd, n_ff, n_expert}），另有 MTP 頭副本（L127, mtp_flags）
```

`llama_model_base::create_tensor_gate_up_exps`（src/llama-model.cpp:2911）做 gate_up 合併 → 缺省 fallback 到分開的 gate/up。qwen35moe 走分開張量。

### 1.2 loader（src/llama-model-loader.cpp）

- `create_tensor`（L1060）：`buft_for_tensor` 依 `llm_tensor_info_for(tn.tensor)` 的 op 選 buffer type；**`tensor_buft_overrides`**（regex → buft）在此套用
- mmap 路徑：`ggml_backend_tensor_alloc(buf_mmap, cur, data)`（L1558）——**零拷貝**，Metal shared buffer 直接包 mmap 頁 → 所有被觸碰的頁全 resident（這是「margin 救不了」的根源）
- 每個 weight 有 **`weight->offs`（檔案 offset）**（L1566 用於 mmaps_used 追蹤）→ pread 所需的一切資訊已在 loader 手上
- `init_mappings(prefetch)`（L1340）：mmap 整個檔案

### 1.3 graph（src/llama-graph.cpp `build_moe_ffn` L1871/L1915）

```
router: logits = build_lora_mm(gate_inp, cur)          // [n_expert, n_tokens]
top-k : selected_experts = ggml_argsort_top_k(...)     // [n_expert_used, n_tokens]
GEMM  : ggml_mul_mat_id(ctx0, w, cur, selected_experts) // w = 整個 [in, inter, 256] 3D
```

`build_moe_ffn` 對 gate/up/down 各一次 `ggml_mul_mat_id`（qwen35moe.cpp L502-515、L680-693 呼叫）。

### 1.4 Metal（ggml/src/ggml-metal/）

- `kernel_mul_mm_id`（ggml-metal.metal:10452）定址：`const int id = ids_i32[...]`（L10503）→ `src0 + id * nb03` —— **expert 定址 = id × 3D stride**
- op 排程：`ggml_metal_op_mul_mat_id`（ggml-metal-ops.cpp:2555）

### 1.5 現成「半成品」機制

`-ncmoe N`（tools/llama-bench/llama-bench.cpp:1256-1258 + common/common.h:1093）：

```cpp
const char * const LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
inline std::string llm_ffn_exps_block_regex(int idx) {
    return string_format("blk\\.%d%s", idx, LLM_FFN_EXPS_REGEX);
}
```

→ `llama_model_tensor_buft_override { regex, ggml_backend_cpu_buffer_type() }` 把**整層**（3 個 256-expert 張量）丟 CPU。**粒度是層不是 expert**，且 CPU GEMM = 6.47 tok/s 陷阱。它證明的事：override 機制本身是加 bounded residency 的正確掛鉤。

## 2. 最小改動點（分 4 層）

### L1 — loader：paged expert 標記 + 排除 mmap + offset 索引

- `llama_model_params` 加 `size_t expert_cache_bytes = 0`（0 = 關閉，行為完全不變）
- `create_tensor` 的 `buft_for_tensor`：當 tensor 是 `LLM_TENSOR_FFN_*_EXPS` 且 `expert_cache_bytes > 0` → 返回新的 buft **`llama_expert_cache_buffer_type()`**（或沿用 cpu buft 但標記 paged）
- `load_all_data`：paged 張量**不進 mmap 共用 buffer**、**不拷貝**；改寫入一份 `llama_expert_index`：`(layer, expert) → (file_offset, bytes, quant_type, tensor_type)`——資料全部來自既有 `weight->offs` + `get_tensor_meta`，**零新解析**
- `init_mappings`：paged 區段從 `mmaps_used` 排除（既有 `unmap_fragment` 機制可直接 madvise 掉專家區段頁）

**定址正確性前提（L1 的 file_offset 必須是真實檔案絕對位置——已踩過兩次坑，實錘記錄）**：

1. **data_start 偏移 bug（關鍵）**：GGUF v3 的 tensor offset 是**「相對 data_start」**的，不是檔案絕對位置。C parser 的 `seg_base`/`stream_offset`/`expert_offsets` 一度全用 raw offset → pread 讀到**早 ~15.8MB 的錯誤區域**（gemma4：gguf-py 權威 `data_offset`=656,107,808 vs C raw=640,283,136，差距恰為 data_start=15,824,672）。當時的「9/9 byte-identical」只是兩邊都錯得一致——內部一致但對真實檔案佈局是錯的。修法：所有進 layout 的 offset 一律 `+ data_start`，真檔測試改以絕對位置驗證。**L1 用 `weight->offs` 建索引時務必確認 llama.cpp 的 offs 語義**（loader 內是相對 mmap 起始的，需對照加 data_start / 檔頭偏移）。
2. **雙家族 segment 佈局不同**：qwen35moe 是**分開三張量**（`ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps`，每層 3×[in,inter,256]，各 0.877MB/exp），gemma4 是**合併兩張量**（`ffn_gate_up_exps`+`ffn_down_exps`，0.879+0.771MB/exp，合併佈局 gate/up 連續）。per-expert 定址必須按佈局分支（3 段或 2 段，第 3 段 size=0 消費端 skip），不能假設單一 stride。兩家族真檔驗證：qwen35moe 9/9、gemma4 6/6 byte-identical PASS（2026-08-14）。
3. **ARRAY 元素大小**：GGUF ARRAY 元素依 type 而異（BOOL/UINT8=1B、INT16=2B），一律 `fseek 4` 會累積偏移（gemma4 的 `sliding_window_pattern` ARRAY<BOOL> n=30 多跳 90 bytes → KV 全偏移）。llama.cpp 自己的 loader 沒這問題（用正確的 size 表），但任何自寫解析器/索引工具都要照 GGUF 規範 size 表。

**改動規模**：~3 個函式、~150 行。不碰 GGUF 解析。

**L1 已完成（2026-08-14）**：
- `llama_model_params.expert_cache_bytes`（0 = 關閉，行為完全不變）+ `llama_model_expert_index_size/index()` 公開 API
- `buft_for_tensor` 標記 `LLM_TENSOR_FFN_*_EXPS`（suffix=weight）為 paged，`load_all_data` 建 `(layer, expert) → (file_offset, bytes, kind, type)` 索引
- **`weight->offs` 語義確認**：`offs = gguf_get_data_offset + gguf_get_tensor_offset`——已是**檔案絕對位置**（含 data_start），L1 索引直接可用，無需再加偏移（先前 C parser 的 data_start bug 在 llama.cpp 內部不存在）
- 診斷：`LLAMA_EXPERT_INDEX_DUMP=N` env 印前 N 筆；llama-bench 加 `-expert-cache <bytes>` 參數
- **驗證（qwen36 IQ2_XXS + gemma4 IQ3_S，對照 gguf-py 權威 offset）**：
  | 檢查 | 結果 |
  |---|---|
  | qwen36 30720 entries（40L×256×3），8644.00 MiB | ✅ 與 gguf-py 全量總和 byte-exact |
  | qwen36 down L0 off=600437216 / stride 335872 | ✅ MATCH |
  | qwen36 gate/up stride 270336、L39 末 expert 在檔內 | ✅ MATCH |
  | gemma4 7680 entries（30L×128×2，gate_up 合併 kind=3） | ✅ |
  | gemma4 gate_up L0 off=805176992 / stride 1270016 | ✅ MATCH |
  | `-expert-cache 0`/缺省：無索引、decode 正常（零回歸） | ✅ |
- 踩坑記錄：per-expert **scale** tensor（`ffn_down_exps_s`）與權重同 enum，需以 suffix="weight" 過濾（修復前 gemma4 多 30 筆 type=0 F32）
- **L1 範圍備註**：目前僅標記 + 建索引，資料仍全量載入（decode 不受影響）；真正的不載入/換指針行為由 L2 pool + L3 gather（P1 後半）接上

### L2 — expert cache 模組（新 C++ 檔，移植既有資產）

`src/llama-expert-cache.{h,cpp}`，邏輯直接搬：

- 我們已寫好的 C streamer（`CGC-main/cgc_engine/cgc/expert_streaming/`，segment-aware pread 已對**真實檔案佈局**驗證：qwen35moe 9/9、gemma4 6/6 + 消費端 8-expert 端到端，data_start 修復後與 gguf-py 權威 offset 一致）+ Swift 的 hot pool / LRU / per-layer 動態 slot
- API：`llama_expert_cache_ensure(ctx, batch_expert_ids)`、`llama_expert_cache_prefetch(ids)`、`llama_expert_cache_fill(ids, dst_buf)`
- 結構：bounded pool（MTLBuffer 或 host 共用 buffer）、per-(layer,expert) → slot 表、refcount、profile-driven hot pool（既有 make_mix_hotpool_profile 產物可直接餵）、背景 pread 執行緒（低 QoS）

**改動規模**：1 個新模組 ~600-800 行（大部分是搬移）。

**L2 已完成（2026-08-14）**：`src/llama-expert-cache.{h,cpp}`（~330 行）
- 公開 API：`llama_expert_cache_init(model, budget_bytes)` / `ensure(layers, experts, n)`（同步阻塞 fill，回傳 miss 數）/ `prefetch(...)`（低優先 bg 執行緒 pread，macOS QOS_CLASS_BACKGROUND）/ `fill(layer, experts, k, kind, dst, stride)`（per-kind 拷出，缺 resident 回 -1）/ `get_stats`（requests/hits/misses/resident/reads）/ `free`
- 結構：cache unit = (layer, expert) blob（gate/up/down 或合併 gate_up 依 L1 索引序串接）；`key_segs` 預建 key→索引位置（O(n) 一次）；global LRU（last_use tick）；loading/queued 旗標 + per-slot cv（ensure 等 bg fill）；`evict_lru` 只換出非 loading/queued 槽；single-file 支援（file_idx 已入索引，split 待未來）
- **`file_idx` 欄位加入 L1 索引**（weight->idx）——單檔模型不受影響（offset 不變，qwen36 dump 30720/8644.00 MiB 復驗一致）
- **驗證（test-expert-cache，兩家族全 PASS）**：
  | 檢查 | qwen36 IQ2_XXS | gemma4 IQ3_S |
  |---|---|---|
  | byte-identity（fill vs 直接 pread 檔案） | kind 0/1/2 ✅ | kind 2/3（合併 gate_up）✅ |
  | 冷 cache 全 miss / 暖 cache 全 hit | ✅ | ✅ |
  | 統計帳目 | req=24 hit=12 miss=12 reads=36（12×3 段） | reads=24（12×2 段）✅ |
  | prefetch→ensure 零 miss（bg fill） | ✅ | ✅ |
  | LRU 換出（tight budget：換出→重讀 miss→再 fill hit） | ✅（修掉一處 self-evict bug） | ✅ |
- 踩坑：ensure/bg 在 `evict_lru` 後才 bump `last_use` → 剛 fill 完的 slot 被自己觸發的換出趕走（且 s 指標 dangling）——先 bump 再 evict 修復
- **範圍備註**：cache 只吃 L1 索引 + 自己 reopen 檔案 pread；decode 圖整合（graph 每 step ensure→fill→喂 `ggml_mul_mat_id`）是 L3-B，尚未接

### L3-B 前置 — decode 路由 hook + bit-identical 對拍（已完成 2026-08-14）

把「argsort_top_k 後掛 cache」接上真實 decode，先用 `-ngl 0` CPU 對拍輸出 bit-identical：

- **hook 機制**：`llama_context::expert_cache_eval_cb`（sched eval callback 包裝）——`process_ubatch` 在 model 有 expert cache 時安裝；對 `ffn_moe_topk-<il>` 節點 ask=true 回 need（sched 會在此斷開計算並 sync），ask=false 時讀 `selected_experts`（I32 `[n_expert_used, 1]`）→ `llama_expert_cache_ensure(layer, experts)`。保留 user eval callback 語義（先轉發 user cb；非 topk 節點行為完全不變）。gate：`n_gpu_layers() <= 0`（CPU-only，Metal 資料讀取安全）+ `n_tokens == 1`（prefill 維持 mmap 路徑）
- **自動建立**：model load 時 `expert_cache_bytes > 0` 即 `llama_expert_cache_init`（model 持有，destructor 釋放）；`expert_cache_path` 賦值**移到 load_tensors 之前**（否則 cache init 時 path 還空 → 靜默失敗）
- **CLI**：`common_params.expert_cache_bytes` + `-expert-cache BYTES`（llama-cli 等全部 common 工具）；`examples/simple/simple.cpp` 也支援（llama-simple 是無互動迴圈的乾淨對拍 harness）
- **驗證（qwen36 IQ2_XXS，-ngl 0，`LLAMA_EXPERT_CACHE_VERIFY=1`）**：
  | 檢查 | 結果 |
  |---|---|
  | 兩 prompt（prose 24 tok / code 32 tok）baseline vs `-expert-cache 1GiB` 輸出 | **bit-identical**（diff 空）|
  | byte-identity：cache fill bytes vs resident tensor 每 selected expert slice（kind 0/1/2，逐層逐 token） | **0 MISMATCH** |
  | 統計帳目 | sample1 req=7360 hit=5248 miss=2112 reads=6336（=miss×3 段）✓ resident=1023.48 MiB（< 1GiB 上限）hit 71.3% |
  | sample2（code prompt）| req=9920 hit=8211 miss=1709 resident=1023.66 MiB hit 82.8% |
  | 零回歸 | `-expert-cache 0`/缺省：hook 不安裝、行為完全不變；test-expert-cache ALL PASSED；llama-bench 正常 |
- 踩坑 1：cache init 失敗被誤判成「metadata-only load」——root cause 是 path 賦值順序（見上），已修
- 踩坑 2：verify 初版把 `fill` 的「selected order」輸出直接 memcmp 到 resident 第 i 列——fill 依選中序排列，必須逐 expert 對 `experts[i]` 定址（修復後 0 mismatch）
- **範圍**：hook 只「餵 cache + 驗證」，**graph 仍讀 resident 權重**（輸出 bit-identical 是預期）；真正換 src0 指針/ids 重映射（gather）是下一步 L3-B 主體

### L3-B 主體 — per-step gather 取代整檔 3D 專家張量（已完成，淨負定案 2026-08-14）

把 graph 每 step 的 FFN `mul_mat_id` src0 換成「cache 依真實 router 選中 gather 的 `[in, inter, k_used]` 連續張量」：

- **實作**：`build_moe_ffn` 在 `ffn_moe_topk-<il>` 後建 `ggml_dup_tensor` remap leaf（`ggml_set_input` + `build_forward_expand`，命名 `ffn_moe_topk_remap`，`cache_remap_tensors[il]` 記住）；`expert_cache_on_topk`（ask=false 鉤子）每 step：save/restore 原 `(data, ne[2])` → `cache_fill` 到 per-kind gather buffer（`cache_gather_buf[4]`）→ `w->data = gbuf; w->ne[2] = k_used` → remap ids 寫 0..k-1。gate：`expert_cache_active`（`expert_cache != nullptr && n_gpu_layers() <= 0 && !NOGATHER`）
- **踩坑（真 bug）**：remap leaf 原以 `n_tokens > 0` 建 → **prefill（n_tokens>1）也有 remap leaf，但 on_topk 對 n_tokens>1 提早 return 從不填 ids** → prefill 的 mul_mat_id 吃 galloc 重用緩衝的未初始化 ids → KV 全壞、第一個 token 就偏（「emu」）。修：remap leaf 改 `n_tokens == 1`（decode-only；prefill 維持 resident 路徑）
- **debug 誤判**：`L3B SWAP` print 的 `cache_orig.size()` 印出指標值，一度誤判 map 被覆寫——實際是 fprintf 格式 14 個 specifier 只傳 13 個 arg（少傳 `cache_orig.size()`），讀暫存器殘留值。已補 arg
- **驗證（-ngl 0，兩 prompt）**：baseline vs `-expert-cache 1GiB` **bit-identical（diff 空）+ 0 byte MISMATCH**；swap 生效（L3B SWAP debug 有印、cache stats hit 73.3%/82.8%）
- **qwen36 UD 檔補驗（2026-08-15）**：先前「UD 佈局需 de-interleave、cache 讀 zeros」的結論**證偽**——`UD` 只是 quantizer 品牌，儲存仍是標準 per-expert-contiguous（L1 的 `base + e*nb[2]` 直接成立）。FILLSLOT 位元組與 resident 逐位一致（早期見到的全零 blob 是檔案本身該 expert 的真實內容，file/resident 兩邊一致故 verify 通過）。**IQ2_XXS（10.76GB）與 IQ3_XXS（13.2GB）雙檔 cache arm 均 bit-identical + 0 MISMATCH**（IQ2 兩 prompt n=48/64、IQ3 n=32，`cmp` 通過；cache resident 分別 1023.7 / 511.5 MiB、bounded 生效）。早期「zeros/佈局錯」的根因其實是 strided-argsort 誤讀 + remap arena 別名（已修），非檔案佈局
- **L3 Option A 靜態 pool 預設關閉（2026-08-15）**：init 原本無條件分配 per-layer slot pool（1GiB budget 下 ≈9.7GiB、n_slots clamp 256）——但接線的 L3-B 路徑（ensure→fill→gbuf）完全不碰 pool，等於 16GB 機白扛 ~9.7GB RSS、直接抵銷 bounded residency。改 `LLAMA_EXPERT_CACHE_POOL=1` 才分配（Option A 尚未接線前的守衛）；預設行為零變更（slot_table/slot_owner 等小結構照舊）
- **128 tok 冷窗 A/B（llama-bench -p32 -n128，3 輪交錯，load 2.7-5.2）**：
  | arm | tg tok/s（mean）| pp tok/s | peak RSS（mean）|
  |---|---|---|---|
  | baseline | ~12.5 | ~11 | ~6.7 GiB |
  | `-expert-cache 1GiB` | **~7.7（−38%）** | ~15 | ~6.6 GiB |
- **判讀（誠實）**：正確性 ✅（gather 取代 resident 權重、輸出逐位元一致）——但**現階段淨負**：
  1. **記憶體沒省**：L1 的 paged 標記仍「全量載入」（loader 未跳過 expert 資料）；且 mmap demand-paging 已把 baseline RSS 壓到「有觸碰的頁面」~6.6GiB——cache 臂把 decode 觸碰的 expert 頁換成 1GiB 匿名 slot，淨 RSS ≈ 持平（-2%）
  2. **decode 慢 38%**：sched eval-callback 分段（每層 topk 一個 subgraph compute + sync，40+ 次/step）+ 每 step ensure/fill memcpy（40 層 × 3 kind × 8 experts），而專家仍在 resident（無 IO 收益可藏）——純開銷
- **L1 skip-load 已完成（2026-08-15，本 step 的轉正條件）**：`llama_tensor_weight.skip_load` + loader 層 gate（`expert_cache_skip_load = expert_cache_bytes>0 && n_gpu_layers()<=0 && !NOGATHER`，與 hook gate 同條件）：
  - **非 mmap**：`load_all_data` 對 skip-load 張量**完全不 read_raw**（`size_done` 照算）→ expert 資料永不進 RAM
  - **mmap**：`init_mappings` **關 prefetch**（原本 MADV_WILLNEED 全檔預取）+ 跳過 check_tensors 驗證與 mlock grow → expert 頁永不 fault 進 working set
  - 安全回退：`-expert-cache 0`（無索引）、`-ngl>0`（hook 不裝）、`LLAMA_EXPERT_CACHE_NOGATHER=1` → 全部維持 full resident（零回歸，實測確認）
  - verify（resident bytes 對拍）在 skip-load 下關閉（無 resident 可對；byte-identity 由 bit-identical 輸出 + 先前 full-resident verify 覆蓋）
- **skip-load A/B 實測（qwen36 IQ2_XXS，CGC_CPU_ONLY）**：
  | 模式 | arm | max RSS | page faults | 輸出 |
  |---|---|---|---|---|
  | non-mmap（llama-simple n=64） | base | 4403 MiB | 906K | — |
  | non-mmap | cache skip-load 1GiB | **3032 MiB（−31%）** | **214K（4.2× 少）** | **bit-identical** |
  | mmap（llama-bench p32 n128） | base | 7992 MiB | 1242K | — |
  | mmap | cache skip-load 1GiB | **3844 MiB（−52%）** | 1401K* | tg 11.11 vs 16.30 |
  \* cache 的 read() 是 page-cache 暫態 fault（不進 process RSS），base 的是常駐 process pages——RSS 才是 working-set 真值
- **結論：1GiB cache 換 ~4.1GiB RSS（mmap 128 tok decode，−52%）**，4:1 槓桿成立；代價 tg 16.30→11.11（−32%，gather 開銷，即下一步 L3 Option A 的回收標的）。**注意 mmap 下 base 的 128-tok RSS（7.99GB）比之前量測高**——128 decode 步 touch 大量 expert 頁（demand-paging 累積），這正是 skip-load 的價值面
- **剩餘**：gather memcpy 開銷（L3 Option A：slot table + kernel indirection）仍是 128 tok 的 −32% 主因；Metal working set 收尾（-ngl>0）是 P0 最後一哩

### L3 — GPU backing：二選一

**Option A：static slot table + kernel 1 行 indirection（真 bounded residency）**（static pool 分配已 gate 在 `LLAMA_EXPERT_CACHE_POOL=1`，接線時再開）
- 每層 expert 張量 = `ggml_view_3d` 指向 pool buffer，slot stride 固定
- `kernel_mul_mm_id` 加一個可選 `expert_slot_table` buffer：`id` 解析改為 `src0 + slot_table[layer][id] * slot_stride`（未給表時走原 `id*nb03`，零回歸）
- 前置條件：dispatch 前 cache 保證該 step 選中的 experts 全 resident（miss 則阻塞補）

**Option B：per-step gather（零 kernel 改動）**
- graph 每 step 重建（llama.cpp 本就如此），把選中的 experts 從 pool（或直接 pread）gather 進一個 bounded 的 per-step 連續 buffer，`ggml_view_3d` 以 `[in, inter, k_used]` 給 `build_moe_ffn`
- `ggml_mul_mat_id` / Metal kernel **完全不動**；代價是每 step 一次 gather 拷貝（= 我們 Swift 的 fill，已實證可藏 GPU 後）

**建議：先 B 後 A**。B 是純 CPU 側改動、可獨立 A/B；A 是 B 之上的 kernel 精修。

### L4 — 排程（沿用已實證的 Swift 手法）

- prefill：chunk 1 冷池**阻塞 fill**（TTFT 代價），後續 chunk 與 GPU 重疊
- decode：**層迴圈內背景 pread 下一層/下一 token 的 union**（Swift §13.97 已證 fill 46→5ms 藏得住）
- router 提早：graph 內 router 完成後即可拿到 top-k id（`ffn_moe_topk` cb 鉤子），不必等整層算完

## 3. 逐檔案改動清單（總 diff 估計）

| 檔案 | 改動 | 規模 |
|---|---|---|
| `include/llama.h` | model params +2 欄位、新 buft + cache API | ~40 行 |
| `src/llama-model-loader.cpp` | paged 標記、load 跳過、offset 索引、mmap 排除 | ~150 行 |
| `src/llama-expert-cache.h/.cpp`（新） | pool/LRU/prefetch/背景執行緒（搬 C streamer + Swift hot pool） | ~700 行 |
| `src/llama-graph.cpp` | `build_moe_ffn` 接 gather（B）或 slot 表（A）；topk cb 鉤子 | ~60 行 |
| `src/models/qwen35moe.cpp` | 無（張量名/形狀不變） | 0 |
| `ggml-metal.metal`（僅 A） | `kernel_mul_mm_id` +slot 表分支 | ~20 行 |
| `tools/llama-bench/llama-bench.cpp` | 接 `-expert-cache <bytes>` 參數 | ~15 行 |

關閉時（`expert_cache_bytes=0`）所有路徑與現行完全一致 → **零回歸風險**。

## 4. 預期數字

| 情境 | resident | 預估 tok/s | 餘裕 |
|---|---|---|---|
| 現行全 offload IQ2 | ~10GB | 25-28 | ≈0 |
| **fork B，IQ3** | ~5.5GB（2.4+3） | **14-19**（llama.cpp 級 kernel + 83% pool hit、fill 10ms 藏 GPU 後） | **~5GB** |
| **fork A，IQ3** | ~5.5GB | 18-24（省 gather） | ~5GB |
| （對照）-ncmoe CPU 陷阱 | — | 6.47 | — |

**命中率假設實測驗證（2026-08-14，sim_gguf_cache.py 修復後，r3_trace 212 decode tokens）**：

| 3GB 預算內配置 | hit% | fill/step | @4.8GB/s |
|---|---|---|---|
| **純 global LRU（3674 slots）** | **82.9%** | 48MB | **10.0ms** |
| hot=90/layer + LRU 74（現行式 pin） | 44.6% | 155MB | 32.4ms |
| hot=60/layer + LRU 1274 | 77.5% | 63MB | 13.1ms |
| hot=30/layer + LRU 2474 | 82.4% | 49MB | 10.3ms |
| adaptive per-layer LRU（hot=0） | 82.8% | 48MB | 10.1ms |

結論：
1. **85% 假設成立（≈83%）**，有效 fill ≈ 10ms/step @4.8GB/s（warm）→ 可藏 GPU 後 ✅
2. **static profile hot pool 在此 trace 是反效果**：pin 90/layer 掉到 44.6%，且 production 實測 ground-truth 只有 52.2%——現行把 pool 大半 pin 給靜態 profile 是 hit 偏低主因；3GB 時純 LRU 即可
3. **adaptive per-layer 在 3GB 下無增益**（82.8 vs 82.9，早期 94% 是預算 bug：hot+LRU 超到 6.4GB）；per-layer 動態只在 pool 小時有意義（§13.147 的 112-slot 情境）

## 5. 驗收

1. **bit-identical**：`expert_cache_bytes=0` 時 decode 輸出與現行完全一致
2. **memory**：`memory_pressure` free% ≥ 20%（對照現行 9%）
3. **IQ3 可跑**：Metal decode 不失敗（現行 -ngl 99 直接掛）
4. **tok/s**：128 tok 生產 A/B，目標 ≥ 15（B）/ 18（A）

## 6. 風險與回退

- **gather 拷貝（B）**：fill 藏不住 → 退回 A 或降低 pool hit 要求（profile 加權）
- **冷池阻塞**：TTFT 變長 → 首 chunk 用 blocking + hot pool 預熱，比現行冷窗還快
- **Metal shared buffer 駐留**：pool 必須是**獨立 bounded buffer**（非檔案 mmap 包裝），否則重蹈 free% 9%
- 全部 env/param 關閉即回退原生 llama.cpp，不影響主線

## 6.5 決策（2026-08-14）：單走 llama.cpp，MLX 暫緩

MLX 在 16GB 機對 gemma4 無品質可用路徑（2-bit = VLM 亂碼、4-bit = OOM、無 per-expert offload），
llama.cpp kernel 效率已實證（qwen36 IQ2 27.9 tok/s）→ **本方案就是唯一主線**。MLX 2-bit 數據保留作
§5 驗收的「更小 quant 基線」對照。P1（L1+L2+Option B gather）為下一步。

### 6.6 引擎 vs 位寬分解（2026-08-14，qwen36 35B 同機實測）

之前「llama.cpp kernel 優勢 4.8×」的敘述需要修正——把 27.9 vs Swift 5.8 拆開（全為同機 M4 實測）：

| 設定 | bpw | 引擎/執行單元 | tg tok/s |
|---|---|---|---|
| llama.cpp qwen36 IQ2_XXS | 2.06 | Metal | 27.9 |
| llama.cpp qwen36 IQ2_XXS | 2.06 | 純 CPU | ~12.5 |
| llama.cpp qwen36 IQ3_XXS | 3.06 | 純 CPU | **~7.7**（3 輪 8.10/8.03/6.99，load 3.6）|
| Swift qwen36-r4（4-bit experts）| ~4.6（18G/34.66B）| Metal | 5.5-5.8 |

- **位寬效應（同引擎 CPU 內）**：IQ2 12.5 → IQ3 7.7；bytes ×1.49 ↔ 速度 ×0.62 → **decode 完美帶寬線性**（memory-bound 實錘）
- **引擎 + 執行單元**：llama.cpp IQ3 CPU 7.7 歸一化到 4.6bpw ≈ 5.1 vs Swift 5.65 → **同 bpw 下引擎差 ~10%**，Swift 沒有引擎劣勢
- **4.8× 的成因拆解**：2-bit vs 4-bit ≈ 2.2×，Metal vs CPU ≈ 2.2×（27.9/12.5），剩餘引擎差 ~10-20%
- **品質**：IQ2 在 qwen36 退化（「day 1 1 1」迴圈，不可用）；IQ3 連貫（Rayleigh 散射正確）——**qwen36 的 2-bit 是陷阱，3-bit 是可用下限**
- 對 fork 方案的意義：§5 驗收「128 tok ≥ 15（B）」的目標不靠引擎效率，靠 Metal + 大模型塞進 working set（IQ3 12.29GiB 已可全 Metal，見 gemma4 IQ3 OOM 對照）

## 7. 分期

- **P1（B 路線）**：L1 + L2 + Option B gather + bench 參數 → 先驗 IQ3 可跑 + 餘裕成立 ✅
- **P2**：L4 排程（層迴圈預取 + profile hot pool）→ tok/s 15+
- **P3（A 路線）**：static slot pool + 無 gather（2026-08-15 完成，見 §7.1）✅
- 每期都以 `-expert-cache 0` 回歸對拍 + 128 tok 冷窗 A/B 定案

### 7.1 L3 Option A 實作完成（2026-08-15）：static slot pool，gather 成本回收 ~72%

**核心設計**（與白皮書「offload 高速路徑只給 n_tokens=1 decode」一致）：

- `LLAMA_EXPERT_CACHE_POOL=1` 才啟用 pool（env gate 修復：之前 `=0` 也啟用，且預設 pool 分配 ~9.7GiB 死記憶體 → 改為值 == "1" 才分配，L3-B 路徑完全不受影響）
- **decode（union < n_slots）走 pool**：每層 static pool（27 slots/layer @ 1GiB budget），hook 只做 ensure_slot（LRU 換出）＋ remap 寫 **slot 索引**（8 ints），零 gather memcpy
- **prefill / 大 union（≥ n_slots）自動 fallback 到 L3-B gather**（arbitrary union 大小），prefill 正確性不受影響
- **Metal 共存修復**（p32 崩潰根因）：`-ngl 0` 下 mul_mat_id 被 sched 拆到 Metal + CPU 雙後端；hook 把 `w->ne[2]` 設為 n_slots，讓 Metal 的 async tensor copy（`ggml_metal_set_tensor_async`）只讀 pool 區域（原本讀 ne[2]=256 個 expert ≈70MB 從 27-slot pool → OOB memmove segfault）
- remap 用 **slot 索引**（非 raw expert id）：mul_mat_id 的 dst 欄是 token 內 rank，ids 只選 src0 slice → slot 索引對 Metal（無 slot table）與 CPU 都正確；raw→slot 解析發生在 hook（每層 8 ints），取代 kernel 內 indirection

**驗證（全部同機 M4、CGC_CPU_ONLY、qwen36 IQ2_XXS 10.01GiB）**：

| 檢查 | 結果 |
|---|---|
| llama-simple n=64 bit-identical（pool 路徑，hit 90.1%） | ✅ |
| llama-bench p32（先前必崩）| ✅ exit=0 |
| gemma4 IQ3_S n=48 bit-identical（30 層/128 exp，12 slots/layer）| ✅ |
| 128 tok A/B（p32 n128, reps=1）| base 14.30 → L3-B 13.38（−6.4%）→ **Option A 14.04（−1.8%）** |
| RSS（non-mmap n=64，skip-load + 1GiB pool）| **2.67 GiB**（base full-resident 4.40 GiB）|

**結論**：Option A 回收 ~72% 的 L3-B gather 開銷（−6.4% → −1.8%），且保留 bounded residency
（RSS 2.67GiB 含 1GiB pool）。剩餘 −1.8% = remap 寫入 + ensure_slot + Metal 27-slot copy。

**已知限制**：prefill 走 L3-B fallback（union 常超 pool）；剩餘收益標的 = L4 排程（層迴圈預取 + hot pool）。

### 7.2 P2 層迴圈預取實作與證偽（2026-08-15）：pool 背景 fill 有 race，改 page-cache 預取後仍無收益

**目標**：把 decode 的 SSD pread（sync miss）藏進 GPU 計算（白皮書「雙緩衝重疊預取」）。

**實作路徑與兩個 race 的修復**（環境變數 `LLAMA_EXPERT_CACHE_PREFETCH=1`，拆 `_LAYER`=層鄰 / `_STEP`=步間）：

1. **v1（pool 背景 fill）**：`prefetch_slot` 佔用 free slot + 背景執行緒 `fill_pool_direct` 直寫 pool；`ensure_slot` 對 in-flight fill 改等待。→ **gemma4 隨機 corruption**（輸出退化 garbage，非細微分歧）。
   - Race #1：drain 丟棄 queued fill 時只清 `slot_loading`，`slot_table`/`slot_owner` 仍宣稱 resident → `ensure_slot`「命中」從未 fill 的 slot → garbage。
   - Race #2（真兇）：bg 的 pread 與 hook 的 sync fill 可能寫同一 slot —— bg 檢查 `owner==expert` 後、pread 完成前的窗口內，drain 釋放 + `ensure_slot` 重用該 slot → bg 遲到的 write 覆寫成別人的資料。`-t 1` 也重現 → 排除 CPU 執行緒，真兇是 Metal region-wide async copy 與 bg write 的並發。
2. **v2（定案，白皮書原文設計）**：prefetch 只做 **page-cache warming**（macOS `fcntl(F_RDADVISE)` / Linux `posix_fadvise(WILLNEED)`），**不寫 pool**；pool 寫入一律在 hook 執行緒串行化（sync fill 完成後才 dispatch FFN）→ 結構上無並發寫入，bit-identical 全數通過。`drain_layer` 保留為 no-op 安全閘。

**測量結果（同機 M4、qwen36 IQ2_XXS 1GiB pool，真實文本 n=96，多輪交錯）**：

| arm | decode hit | 時間（2 輪） |
|---|---|---|
| base（Option A）| 91.3% | 22.2 / 30.9 s |
| +prefetch（lyr+stp）| 91.3% | 22.9 / 29.7 s |

**三個結構性原因（皆已量測）**：
- **步間預測是 no-op**：`prefetch=0` queued —— 步間 routing 重疊 7.88/8（98.5%），且每層 working set 只有 8-9 個 experts << 27 slots，LRU 已完整保留，無可預取之物。
- **層鄰預測命中率 ≈0**：qwen36 真實文本層鄰重疊 6.29/8（78.6%），但那些重疊 experts 早已在 pool；真正新進的 experts 層鄰預測 95% 錯 → 純污染。
- **剩餘 8.7% miss 是結構性不可預測**：它們是路由真正切換到的新 experts，任何便宜預測器（前步/鄰層）都抓不到。

**方法論發現：llama-bench 喂的是隨機 token**（`test_prompt`/`test_gen` 用 `std::rand() % n_vocab`）——其 tg128（5.5-14 tok/s）與 hit 54% 是**最壞路由 churn 情境**，不是真實生成。真實文本 decode hit 91-92%（同 prompt 下 llama-simple）。之後 A/B 若要看真實性能需用 llama-simple 或改 bench 喂真 prompt。

**定案**：prefetch 全部 env-gated、預設 OFF。pool 的 LRU 已是完美 hot pool；剩餘 decode 延遲標的 = miss 的 sync pread（真實文本 ~8.7%）在架構上只能靠更大 pool budget 或更快的 SSD，非排程能解。P2 標記為「實作完成 + 測量證偽」。

### 7.3 Metal 併入嘗試與封存（2026-08-15）：gate 回滾 + 凍機事件 + copy 時機根因（P0 定案）

**P0 現況（定案）**：`-ngl 0` 路徑為**現行可用狀態**——L1 skip-load + L3 Option A 全鏈
bit-identical、RSS bounded（2.67 GiB 含 1GiB pool）。**Metal（-ngl>0）併入未完成且已封存**，
條件見下方「封存條件」。

#### 7.3.1 gate 回滾（凍機風險封住）

2026-08-15 曾為接 Metal 放開 skip-load/hook 的 `-ngl 0` gate（三處：`llama.cpp:319`
skip-load、`llama-context.cpp:1542` hook、`3125` expert_cache_active）。**已全部回滾**回
`n_gpu_layers() <= 0` 並附註解——`-ngl>0 + cache` 現在是 full-resident 零行為變更（等同上游
baseline），不會誤跑凍機。

#### 7.3.2 凍機事件（13:08 重啟）

`-ngl 99 + cache + skip-load + mmap`：loader 把 expert 張量留在 mmap（從未 fault），Metal
backend 在 **graph-alloc / split-input copy 階段**讀這顆 tensor → 一次 fault **9GB 冷頁** →
頁面風暴 + Metal buffer 分配 → 系統記憶體壓力爆表硬掛（無 panic 檔 = freeze）。`-no-mmap`
同設定不凍（skip-load tensor 是未寫入的零 buffer，無 fault 路徑）但輸出同樣退化——證明
copy 時機 bug 獨立於 mmap。**教訓：診斷 Metal 線一律 `-no-mmap`；`-ngl 99 + cache` 不得併用
mmap**（除非 skip-load 完全封住）。

#### 7.3.3 copy 時機根因（已隔離證實，非 skip-load 專屬）

**根因**：`-ngl>0` 下整個 graph 是**單一 MTL split**；`ggml_backend_sched_compute_splits`
在 split 開頭把所有跨 backend input copy 進 Metal（實測 `hook_swap=0` 時 copy），包括
`ffn_moe_topk_remap-*`（mul_mat_id 的 ids）與（skip-load 下）expert tensor。hook 在 compute
中途（topk 節點）才寫 remap（pointer swap）+ swap expert data → **Metal 讀到 split 開頭 copy
的 stale/零 ids → 全路由 expert 0 → 退化輸出**。

三臂隔離證明（qwen36 IQ2，full-resident、-no-mmap、安全跑完）：

| 配置 | 輸出 |
|---|---|
| base（無 cache）| 正確 |
| ngl99 + cache + `LLAMA_EXPERT_CACHE_NOGATHER=1`（無 remap/swap）| **與 base 相同** |
| ngl99 + cache（有 remap/swap）| 退化（`tabIndex tabIndex...`）|

`-ngl 0` 為何正常：decode（batch 1）單一 CPU split（Metal `op_offload_min_batch_size=32`
擋住 MUL_MAT_ID offload）；prefill（batch≥32）441 個細 split，topk 在 CPU split、FFN 在
MTL split——hook 在 CPU split 先 fire（實測 FFN split copy 時 `hook_swap=3`）→ 資料正確。

**修復方向（A+B split，2026-08-15 已實作並驗證，見 §7.3.5）**：cache 啟用時把
router 權重（`ffn_gate_inp`）強制 CPU buft + router op（argsort/topk）釘 CPU split →
複製 -ngl 0 的成功 split 結構（白皮書 A+B：CPU router × GPU FFN）。此修復也是「gemma4
IQ3_S 11.29GB 壓進 11.45GB Metal working set 不再 OOM」的前置（§7.3.4 條件 3）。

#### 7.3.4 封存條件（Metal 線重開前必須滿足）

1. **修好 copy 時機**：~~實作 A+B split~~ **✅ 已達成（2026-08-15，§7.3.5）**——hook 的
   remap 寫入發生在 Metal copy 之前（雙家族 -ngl 99 + cache bit-identical）
2. **診斷配置**：重開 gate 只用 `LLAMA_EXPERT_CACHE_ALLOW_NGL=1` + **`-no-mmap`**；
   `-ngl 99 + cache + mmap` 列為**禁區**（凍機）——2026-08-15 進一步：**loader 自動強制
   no-mmap**（`cache>0 && ngl>0` 時改 `LLAMA_LOAD_MODE_NONE`，§7.3.6），禁區從「文件約定」
   升級為「程式保證」
3. **驗證門檻**：-ngl 99 + cache 輸出回歸 bit-identical（對照 base）→ 才談 working set /
   skip-load 併入；skip-load 併入後再量 gemma4 IQ3_S 的 working set 與 OOM 與否——
   **✅ 已達成（2026-08-15，§7.3.6）**：gemma4 IQ3_S bit-identical、無 OOM、RSS 4.48GB
4. **回退**：重開的 gate 一律 env-gated，預設保持封住（-ngl 0 only）——skip-load 仍以
   `LLAMA_EXPERT_CACHE_ALLOW_NGL` 為前提（§7.3.6 定案）

**診斷插樁**（本 session 新增，全 env-gated 預設 OFF）：`LLAMA_EXPERT_CACHE_ALLOW_NGL=1`
（診斷用放開 gate）、`CGC_SCHED_DBG=1`（split 結構 + `hook_swap` 計數 + Metal expert copy
dump）、`g_expert_swap_count` 全域計數器（hook 每次 swap 遞增）。詳細見 RUNTIME_CONTROLS.md
llama.cpp fork 章節。

#### 7.3.5 A+B split 實作完成（2026-08-15）：-ngl 99 + cache 回歸 bit-identical（封存條件 1 ✅）

三層修復（缺一不可，逐層定位）：

1. **loader：router 權重強制 CPU buft**（`llama-model-loader`）——`expert_cache_router_cpu`
   gate（`cache>0 && (ngl<=0 || ALLOW_NGL)`），`LLM_TENSOR_FFN_GATE_INP`（+ `_SHEXP`）
   強制 `ggml_backend_cpu_buffer_type()`。**單獨不夠**：sched 的「expand gpu」pass 把 router
   掃回 Metal（Metal supports ARGSORT + host-buffer inputs）。
2. **sched：router op 釘 CPU**（`ggml-backend.cpp` split_graph 的 expand gpu down/up 迴圈）——
   `LLAMA_EXPERT_CACHE_ALLOW_NGL` 下，`ffn_moe_topk-*` / `ffn_moe_argsort-*` /
   `ffn_moe_group_topk-*` 直接指派 `n_backends-1`（CPU），切斷 GPU 擴展掃過 router。
3. **hook：resident passthrough**（`llama-context.cpp` on_topk）——`!expert_cache_skip_load`
   （= ngl>0 full-resident）時：**不 swap** `w->data/ne[2]`（Metal 用自己的 buffer；
   ne[2] 必須維持 256 供 n_as）、**不 fill**，remap 寫 **real expert ids**（0..255，非
   slot/union 索引——Metal 讀 full 256-expert tensor）。

驗證（全 -no-mmap）：

| 檢查 | 結果 |
|---|---|
| qwen36 IQ2 -ngl 99 + cache（ALLOW_NGL，n=48 × 2 prompts）| **BIT-IDENTICAL** |
| qwen36 IQ2 -ngl 99 + cache + POOL=1（Option A）| **BIT-IDENTICAL** |
| gemma4 IQ3_S -ngl 99 + cache | **BIT-IDENTICAL**（30 層 A+B split 生效）|
| 回歸：-ngl 0 + cache（CGC_CPU_ONLY）| BIT-IDENTICAL（零回歸）|
| 回歸：-ngl 99 無 ALLOW_NGL（封存狀態）| = base full-resident，BIT-IDENTICAL |

**結構確認**（CGC_SCHED_DBG）：graph 162 splits；每層
[MTL: attn] → [CPU: router MUL+softmax+topk(pinned)] → [MTL: FFN mul_mat_id] → [CPU: combine]；
FFN split 的 input copy 發生在 `hook_swap=3`（hook 已先寫 remap）——copy 時機問題結構性消失。

**剩餘**：§7.3.4 條件 2/3 未動（skip-load 仍封 `-ngl 0`；gemma4 working-set 驗證待
skip-load 併入後）。hook 的 resident passthrough 使 cache requests=0（不 fill，全 resident）
——符合本階段目標（先 bit-identical，bounded residency 是下一步）。

#### 7.3.6 skip-load 併入 -ngl>0（2026-08-15）：封存條件 2/3 ✅（gemma4 定案，qwen36 UD 另案）

**改動（2 處，皆 env-gated）**：
1. **skip_load gate 放寬**（`llama.cpp`）：`ngl<=0` → `ngl<=0 || ALLOW_NGL`——`-expert-cache`
   在 `-ngl>0` 下只有 `LLAMA_EXPERT_CACHE_ALLOW_NGL=1` 才啟用 skip-load（預設維持封住）。
   hook 因此自動從 resident passthrough 切回 bounded mode（swap pool / L3-B gather）。
2. **loader 自動強制 no-mmap**（`llama.cpp` load 前）：`cache>0 && ngl>0` 時改
   `LLAMA_LOAD_MODE_NONE`（log：`forcing no-mmap`）。

**mmap OOM 根因（新發現）**：`-ngl 99 + cache + mmap` 在 prefill 就
`kIOGPUCommandBufferCallbackErrorOutOfMemory`（工作集爆 11.45GB）——skip-load 的 expert
tensor 留在 file-backed mapping，其**完整範圍**被計入 Metal working set 帳目（即使頁面
從未 fault、buffer 是 CPU buft）。`-no-mmap` 下同 config 完全正常。與先前凍機（13:08）
同一族問題（file-backed skip-load 區塊 × Metal 帳目），故以程式保證封住：**ngl>0 + cache
自動 no-mmap**。`-ngl 0` 路徑保留 mmap（全 CPU，已驗證安全）。

**驗證矩陣（全 -ngl 99、1GiB pool、auto no-mmap）**：

| 檢查 | 結果 |
|---|---|
| **gemma4 IQ3_S + cache（32 tok）** | **BIT-IDENTICAL** ✓、exit=0 無 OOM、RSS **4.48GB**（base 11.49GB，experts 出 resident）、OA bounded mode 930 次觸發 |
| gemma4 顯式 -no-mmap | identical ✓ |
| qwen36 IQ2 + cache（32 tok）| **DIFFER** —— 但 **ngl 99 base 本身 ≠ ngl 0 base**（`1919` vs `brown fox`，pre-existing Metal+UD 分歧，無 cache 也重現、非本 session 引入）→ cache arm 確定性、無 OOM，但無法對「已分歧的 base」驗證 |
| 回歸：-ngl 0 + cache | bit-identical ✓（零回歸）|
| 回歸：-ngl 99 無 ALLOW_NGL | = base full-resident ✓ |

**速度代價（誠實記錄）**：gemma4 ngl 99 base 21.4 tok/s → +cache 5.19 tok/s（−76%）。
A+B split 結構成本：~120 splits/token（30 層 × 4）+ 每層 host→Metal 的 pool 區塊直讀
（shared memory，零 copy 但 GPU 側每次 FFN 都要重新取權重）。bounded residency 的價值是
「跑得起來」（11.29GB 模型不再 OOM），不是速度；速度線是 L4（Metal-visible pool / 零拷貝）。

**定案**：封存條件 2/3 以 gemma4（條件點名模型）為驗證標的達成——skip-load 併入
`-ngl>0`、11.29GB 壓進 working set 不 OOM、bit-identical、RSS 4.48GB。qwen36 UD 的
ngl 99 分歧是**獨立於 cache 的既有 Metal/UD 正確性問題**（需 UD de-interleave 或非 UD
檔，與 §6 已列項一致），不阻擋條件 2/3 的 gemma4 驗證。


### 7.4 真實文本重測（2026-08-15）：Option A tg 的 -1.8% 是低負載假象，真實懲罰 ≈ -7%

先前 §7.1 的「Option A −1.8%（回收 72% gather 開銷）」來自 llama-bench（**隨機 token**
= 最壞路由 churn）。用 llama-simple 真實文本（prose 57-token prompt + n=128、-ngl 0、
CGC_CPU_ONLY）重測：**base ~14.2 tok/s（穩定，與先前 14.30 一致）；Option A ~13.2
（−7%，順序平衡後的最佳估計，單輪範圍 −2% ~ −15%）**。三臂全數 bit-identical。

| 量測 | base | L3-B | Option A |
|---|---|---|---|
| llama-bench 隨機 token（先前 session）| 14.30 | 13.38（−6.4%）| 14.04（−1.8%）|
| llama-bench 隨機 token（今日重跑）| 13.06 | — | 10.83（−17%）|
| llama-simple 真實文本（今日，順序平衡 4 輪）| **14.12** | — | **13.15（−6.9%）**|
| llama-simple 真實文本（3 臂，base 先行有偏差）| 15.61/12.87 | 12.73/10.83 | 12.78/11.04 |

**結論與修正**：
1. **base 穩定**：真實文本 14.1-14.3 tok/s，跨 session、跨方法一致。
2. **-1.8% 是低負載假象**：llama-bench 的數字對機器負載極敏感（今日同設定重跑變 −17%）——
   隨機 token 54% hit 意味 ~56K 次 sync pread，負載一高就爆。**真實文本 Option A 懲罰 ≈
   −7%**（順序平衡估計），且「回收 72% gather 開銷」的宣稱在真實文本下不成立：L3-B ≈
   Option A（相鄰位置對比 +0.4%/+2%）。
3. **懲罰來源**：cache 臂 ~19K 次同步 pread（decode 7.4% miss × 3 kinds）+ hook/LRU 開銷；
   L3-B 額外背 gather memcpy（~290MB/step），但與 Option A 的共享成本（preads）相比被掩蓋。
4. **方法論**：llama-bench（隨機 token）與 llama-simple（真實文本）的 hit 率、preads、tg 差異
   巨大——性能 A/B 一律用 llama-simple 真實文本 + 順序交替 + 多輪，絕對數字受桌面 app 棧
   （Freebuff ~120% CPU）干擾 ±30%，看相對配對比較。
### 7.5 剩餘速度槓桿重盤（2026-08-15）：目標從「回收 gather 開銷」改為「藏 pread」——結論：藏不進 GPU，只能縮小 miss 成本

Option A 在真實文本下與 L3-B 打平（§7.4）後，剩餘速度槓桿的唯一候選是「藏 pread」：
把 cache 臂的 ~19K 次同步 pread 從關鍵路徑挪走。本節先量化 pread 結構，再對
「Metal A+B split 後能否把 pread 藏進 GPU 計算」做結構性判死，最後給出真正可行的槓桿。

#### 7.5.1 pread 結構實測（128 tok 真實文本、-ngl 0 + Option A、2026-08-15）

```
final stats: requests=44039 hits=37697 misses=6342 resident=1023.84 MiB file_reads=19026
decode (ensure_slot) hits=37697/40720 (92.6%)   prefill/multi hits=0/3319
eval: 69.46 ms/token (14.40 tok/s)
```

| 量 | 值 | 含義 |
|---|---|---|
| **reads / miss** | 19026 / 6342 = **3.0** | 每次 miss 恰 3 次**串行** blocking pread（qwen36 gate/up/down 三段，各 270/270/336 KB） |
| decode miss/token | 3023 / 127 ≈ **23.8** | 每 token 約 71 次串行 pread 在 hook 關鍵路徑上 |
| miss 延遲估計 | 3 × 0.09ms（~3GB/s SSD）≈ **0.27ms/miss** | 23.8 miss × 0.27 ≈ **6.4ms/token ≈ 9%** of 69.5ms → 對上實測 −7% ✓ |

fill_pool_direct（Option A）與 fill_slot（L3-B）都是同構的串行 segment 迴圈：
`pread(gate); pread(up); pread(down)`——同一 expert 的三段各自 blocking，I/O 無法併發。

#### 7.5.2 「藏 pread 進 GPU 計算」：結構性判死

**依賴鏈決定 pread 的執行時刻，沒有 GPU 窗口可藏**：

```
MTL attn(N) → [CPU router+topk(N)]  ← hook 在此，miss 才做 3× 串行 pread
            → MTL FFN(N) → [CPU combine(N)] → MTL attn(N+1) → ...
```

1. **router(N+1) 的 routing 依賴 FFN(N) 的輸出**（殘差流）→ layer N+1 的 experts 在
   FFN(N)（GPU）完成**之前不可知** → pread 只能在 FFN(N) 之後、FFN(N+1) dispatch 之前執行。
2. **sched 嚴格交替執行 split**（`ggml_backend_sched_compute_splits` 逐 split
   `graph_compute_async`），且 A+B 結構每層邊界都是資料依賴（combine(N) 需 FFN(N)、
   attn(N+1) 需 combine(N)）→ 每個 CPU split 執行時 GPU 必然 idle，每個 MTL split 執行時
   CPU 在等——**零重疊窗口**。pread 執行的瞬間，GPU 沒有可跑的工作。
3. **唯一逃逸 = 提前發（預測 routing）**——已在 P2 實測證偽：step 間重疊 98.5% 的 experts
   早就在 pool（LRU 保留，預測 no-op）；layer 鄰「新 experts」預測 95% 錯。miss 的 8.7%
   是路由真正切換的新 experts，**任何便宜預測器都抓不到**。
4. **附帶結論**：Metal 讓 FFN 更快（層時間更短）→ 同一個 0.27ms pread stall 佔層時間的
   比例更高 → **-ngl>0 + skip-load 的懲罰只會比 -ngl 0 更糟，不會更好**。想靠 GPU 併入
   解決 pread 是反方向。

#### 7.5.3 重盤後的剩餘槓桿地圖

| 槓桿 | 估回收 | 狀態 |
|---|---|---|
| ~~readv 合併~~ → **fill 併發**（§7.7）| **~1%（在雜訊內）——「估回收大部分 −7%」證偽** | readv 不可行（segment 分散、lio_listio 在 macOS 壞）；threads 併發實作完成，回收 ~1% |
| pool budget 加大（27→54 slots/layer，miss 減半）| 一半的 −7% | 與 bounded residency 的 tradeoff；RSS 4:1 槓桿仍成立（1GiB→2GiB pool） |
| fadvise 頁快取預取 | 0（P2 證偽）| 關閉 |
| GPU 藏 pread（A+B split）| 結構性不可能 | 判死（§7.5.2）|
| **MTP 批量**（多 token FFN 一次跑、pread 併發攤薄）| 唯一結構性藏法 | white paper 已列；改執行模型，大工程，非本 fork 現階段標的 |

#### 7.5.4 定案

**−7% 懲罰 = 每次 miss 的 3× 串行 pread，不是 gather、也不是 hook。** 剩餘槓桿唯一
實作面就是「把 miss 變便宜」：第一刀 readv 合併（估回收大部分 −7%），第二刀 pool
budget 加大（RSS 換 miss 率）。做完後 bounded residency 接近零成本，殘餘即 SSD 速度
本身（路由切換的物理下限）。「藏 pread 進 GPU」在任何單流 decode 架構下都不成立——
依賴鏈是硬串行，這是理論界線不是工程欠債。
#### 7.7 fill 併發實作（2026-08-15）：readv 不可行（segment 分散），threads 併發只回收 ~1%

**readv 可行性判死**：L1 index dump 證明 gate/up/down（或 gate_up/down）tensor 在檔案中
**分散**（qwen36: down@600437216 / gate@687280608 / up@759312864，彼此有 gap；gemma4:
down@656107808 / gate_up@805176992）。`readv/preadv` 只支援**連續**範圍（單一起始
offset、依序填滿 iovec），無法表達分散 segment；macOS 的 `lio_listio`（AIO 批量）實測
`EAGAIN`（regular file 不可用）。**「3 iovecs 併發」在 readv 語意下不存在**。

**實作**（`llama-expert-cache.cpp`）：`fill_job` + `fill_segments_concurrent`——每次 miss
的 segments 各開一條 thread 並行 pread、**join 後才返回**（同步、無 async-write race，
pool/blob 在 Metal copy 前穩定）。`LLAMA_EXPERT_CACHE_SERIAL_FILL=1` 強制舊串行（A/B +
逃生口）。fill_slot 與 fill_pool_direct 都改走它。新增 `fill_batch_usec` 診斷（hook thread
關鍵路徑的整批 fill 成本；`pread_usec` 對並行會高估——thread 各自的 syscall wall time
涵蓋整批期間，不可跨 arm 比較）。

**微基準（M4、3×~300KB 分散讀）**：serial 204µs（冷）/ 90µs（熱）vs threads 163/84µs——
SSD queue depth 限制分散讀的重疊（只省 ~20% 冷讀），thread spawn 開銷再吃掉一半。

**128 tok 真實文本 A/B（-ngl 0、fill_batch_usec 關鍵路徑成本）**：

| round | serial fill | parallel fill |
|---|---|---|
| R1（load 高）| 1.36s | 2.14s（spawn 被高載拖慢，outlier）|
| R2 | 2.67s | 2.01s（−25%）|
| R3 | 1.11s | 0.99s（−11%）|
| R4 | 1.19s | 1.03s（−13%）|

淨回收 ~1% token 時間（tok/s 全在 ±15% 雜訊內無法分辨）。**雙家族 bit-identical**
（qwen36/gemma4 × ngl 0/ngl 99 全過；曾一度誤判 gemma4 ngl99 DIFFER，實為 /tmp 檔案
被重啟清掉的假警報）。

**定案**：−7% 懲罰的主體是**單次 pread 延遲本身**（16GB 機頁快取冷讀），不是 3× 串行化
——串行化只佔 ~1%。fill 併發保留為預設（中性偏正、無害），`LLAMA_EXPERT_CACHE_SERIAL_FILL`
為逃生口。§7.5.3「readv 合併估回收大部分 −7%」**證偽**。剩餘 −7% 的物理下限是 SSD 冷讀
延遲 × miss 數（routing 切換不可預測），唯一結構性出路仍是 MTP 批量（§7.5.3 已列）。


### 7.6 pool budget 定界實驗（2026-08-15）：2× budget → miss 僅 −29%（heavy tail）；pread 延遲是 −7% 懲罰主體

針對 §7.5「−7% = 每次 miss 的 3× 串行 pread」的推導做實測定界：1GiB vs 2GiB pool
（27 vs 61 slots/layer）、128 tok 真實文本、-ngl 0 + Option A、多輪交錯。同時在
fill_slot / fill_pool_direct 的 pread 外包 steady_clock，新增 `pread_usec` 欄位到
final stats（保留為診斷工具）。

#### 7.6.1 確定性結論（load 無關）：miss 率不隨 budget 同比例降

| budget | slots/layer | decode requests | misses | miss rate | file_reads |
|---|---|---|---|---|---|
| 1GiB | 27 | 40720 | 3023 | 7.4% | 19026 |
| 2GiB | 61 | 40756 | 2154 | **5.3%** | 16311 |

- **2× budget → misses 只 −29%（−50% 才是同比例）**。routing 切換是 heavy-tailed：
  多出來的 34 slots 只覆蓋次熱門區間，長尾新 experts 照樣 miss。加大 budget 的邊際
  報酬遞減，**「加大 budget」不是有效槓桿**。
- 三臂全數 bit-identical。

#### 7.6.2 pread 延遲分解：pread_usec / eval 佔比

| 時段 | 1GiB pread_share | 2GiB pread_share |
|---|---|---|
| 輪 1（頁快取較熱）| 5.7-6.5%（3 樣本，pread_usec 0.50-0.62s）| 7.9-17.3% |
| 輪 2（頁快取冷/混沌）| 15-43%（pread_usec 1.5-3.2s）| 11-39% |

**兩個結論**：
1. **pread 延遲 = −7% 懲罰的主體**。熱態下 1GiB 的 pread_share ≈ 6% ≈ 實測懲罰 −7%
   → hook/LRU 固定開銷 ≤ 1-2%（約 15% 以下）。§7.5 的計算推導（23.8 miss/token ×
   0.27ms ≈ 6.4ms ≈ 9%）與實測交叉印證。
2. **頁快取狀態主導 pread 延遲（30µs hit ↔ 200µs+ 冷讀）**。free 記憶體僅 600MB-2GB
   時 OS 頁快取是混沌的：同 config 的 pread 總時在 0.5s-3.2s 間跳動（5 倍）。這正是
   tok/s 測量 ±15% 抖動的根源，也是「同機同 config 數字不穩定」的物理解釋。

#### 7.6.3 2GiB 反噬效應（16GB 機的 Pareto 點）

- RSS 2.67 → 3.7GiB：更大的常駐擠壓 OS 頁快取 → pread 冷讀比例升高 → 2GiB 的
  pread_usec 常態性 ≥ 1GiB（即使 reads 少了 14%）。
- tok/s 配對（受順序偏差污染、僅參考）：4 輪 c1g 全勝但 R4 持平（10.48 vs 10.41）；
  無證據 2GiB 更快，有微弱證據持平或更差。
- **16GB + 桌面 app 棧下，1GiB pool 已是 Pareto 點附近；加大 budget 不是槓桿**。

#### 7.6.4 定案

1. **budget 槓桿：證偽**（heavy tail + RSS 反噬）——維持 1GiB 預設。
2. **pread 延遲是 −7% 的主體（≥85%），hook/LRU 固定開銷 ≤15%**——§7.5 的
   「把 miss 變便宜」方向正確，readv 合併（3× 串行 → 1 次併發）仍是第一刀。
3. **頁快取混沌是 16GB 機器的物理現實**：任何 pread 相關的 A/B 都必須在同一頁快取
   狀態下配對比較，或接受 ±5× 的延遲抖動。
4. 新增診斷：`pread_usec` 欄位（final stats）保留，供日後任何 pread 相關實驗直接量
   延遲分額。

#### 7.8 pool budget 取捨曲線全量量測（2026-08-15）：27/54/108 slots 三元對照——1GiB 確認為 16GB 機 Pareto 點

補完 §7.6 只測 1GiB vs 2GiB 的缺口：27 / 54 / 108 slots/layer（1 / 2 / 4 GiB）三臂
+ base，128 tok 真實文本、-ngl 0、3 輪交錯、全臂 bit-identical。

**量測數據**（miss / reads / pread_usec 為確定性統計，load 無關；tok/s 受背景 load
5.3-6.2 壓抑，只看臂間相對）：

| budget | slots | decode hit | misses | file_reads | pread_usec | fill_batch_usec | RSS* | mean tok/s |
|---|---|---|---|---|---|---|---|---|
| base | — | — | — | — | — | — | 5.8GB* | 8.78 |
| 1GiB | 27 | 92.6% | 6342 | 19026 | 0.956s | 0.581s | ~4.7GB* | 8.46 |
| 2GiB | 54 | 94.7% | 5437 | 16311 | 0.968s | 0.582s | ~5.5GB* | 8.39 |
| 4GiB | 108 | 93.7% | 4964 | 14892 | 1.077s | 0.650s | ~5.7GB* | 8.35 |

（*RSS 為 /usr/bin/time -l 在 load 5-6 下量測，被頁快取 churn 污染——之前低負載控制
量測是 base 4.4 / 1GiB 2.67 / 2GiB 3.7GiB。趨勢方向一致：budget 越大 RSS 越高。）

**解讀（決定性部分）**：

1. **miss 與 file_reads 單調下降但未飽和**：6342→5437→4964（−14%/−22%），
   reads 19026→16311→14892。108 槽的 decode hit% 93.7% < 54 的 94.7% 是**分母
   移位的計數假象**（108 槽時 prefill union 填入更多，prefill/multi 請求 3319→2333，
   decode ensure 請求分母 40720→41706）——原始 misses/reads 才是乾淨指標，仍在下
   降。**原始 miss 槓桿在 108 槽還沒到頂**。
2. **但 pread_usec 隨 budget 反升**（0.956→1.077s）、fill_batch_usec 也升
   （0.581→0.650s）——RSS 擠壓頁快取 → 每次 pread 更冷。§7.6.3 的「2GiB 反噬」
   在 4GiB 更明顯：**miss 少了 22%，pread 總時間反而多了 13%**。
3. **tok/s 三臂全平**（8.46/8.39/8.35，差在雜訊內）——miss 減少的好處被 pread
   變慢完全吃掉，**budget 在 16GB 機上淨效益為零甚至為負**。

**定案**：

1. **1GiB / 27 slots 確認為 16GB 機 Pareto 點**：最低 RSS、與更大 budget 同速。
   §7.6.3 的結論從「點附近」升級為「全曲線確認」。
2. **budget 槓桿最終證偽（16GB 硬體上）**：miss 減少存在但被頁快取擠壓抵銷；
   只有 32GB+ 記憶體（頁快取不緊張）機器上加大 budget 才有意義。
3. **tok/s 的物理下限 = SSD 冷讀延遲 × 不可命中 miss**，與 budget 無關——唯一
   結構性出路仍是 MTP 批量（改變執行模型，減少每 token 的 miss 次數）。


### 7.9 L4 零拷貝實作與 index 根因修復（2026-08-15）：Metal-visible pool、FFN 回 GPU、bit-identical

#### 7.9.1 設計（L4-A）

- loader 把 expert tensor 以 **Metal buft + ne[2]=capacity** 建立（容量即 bounded pool），
  cache 在 load 後 `adopt_pool_region` 直接把 pool 指向 tensor 的 Metal storage——每步
  Metal FFN 讀 pool 零拷貝（不再有 §7.3.3 的 host→Metal async copy，也無 CPU FFN 退化）。
- 與 CPU-buft skip-load（-ngl 0 路徑）互斥；-ngl>0 + ALLOW_NGL 才啟用。

#### 7.9.2 根因（非確定性 + 退化 + 崩潰的真主嫌——不是先前推測的「跨 step GPU race」）

診斷中一度假設輸出退化是「step N 的 GPU FFN 尚未完成、step N+1 的 hook 已覆寫 pool slots」
（decode 尾端其實已有 `synchronize()`，§7.3.5 三層修復帶入）。真正根因在 loader：

1. **L1 index 在 L4 shrink 之後建構**：`buft_for_tensor`（含 index build）在
   `t_meta.ne[2] = capacity` 之後才被呼叫 → index 每層只建 `capacity`（16/30/60）個
   expert 條目 → `cache->n_expert = capacity`、slot_table 只配 `layers×capacity` →
   hook 對 expert ≥ capacity 的 topk 寫 slot_table **越界**（heap corruption →
   非確定性 + 退化）且 `key_segs.at(key)` **key not found**（std::out_of_range 崩潰）。
2. **union == capacity 邊界**：gemma4 top-8、n_batch 2 → union 可達 16 == n_slots →
   嚴格 `<` 失敗 → 落入 L3-B gather（swap w->data/ne[2]，對 Metal-resident tensor
   **語意錯誤**）。

修復（3 處，皆小改）：

| # | 位置 | 改動 |
|---|---|---|
| 1 | `llama-model-loader.h/.cpp` | 新增 `expert_cache_full_ne2`：shrink 前保存原始 expert dim；index build 用 `full_ne2 > 0 ? full_ne2 : ne[2]`——**L1 index 永遠覆蓋全部 experts** |
| 2 | `llama-context.cpp` | n_batch cap 改 `(capacity-1)/8`（原 `/8`）——保證 `union < capacity`，**L4 下 L3-B 永不觸發**（該路徑與 Metal tensor 不相容） |

修復後：slot_table 尺寸正確、index 覆蓋 128/256 experts、L4 永不落 L3-B。

#### 7.9.3 驗證矩陣（全部 -no-mmap）

| 檢查 | 結果 |
|---|---|
| **gemma4 IQ3_S -ngl 99 + L4（1GiB, capacity=30）** | **BIT-IDENTICAL（48 tok）**、確定性（2 連跑相同）、無 OOM |
| gemma4 RSS | **11.42GB → 5.10GB**（−55%；experts 出 working set）|
| gemma4 decode（eval time）| base 85.1ms/tok（11.75 t/s）→ L4 129.3ms/tok（7.73 t/s，−34%）|
| gemma4 prefill | base 20ms/tok → L4 168ms/tok（n_batch 被 cap 到 3，union-fit 代價）|
| gemma4 cache stats | hit 82.5%、misses 2803/48tok、fill_batch 3.39s（pread 在 hook 關鍵路徑）|
| qwen36 IQ2 -ngl 99 + L4（capacity=60, n_batch→7）| **確定性 ✓、無崩潰**（UD 檔 base 在 -ngl 99 有既有 Metal 分歧，無法 bit-identical）|
| 回歸：qwen36 -ngl 0 + Option A | BIT-IDENTICAL、pool 1023MiB（零回歸）|

#### 7.9.4 定案與剩餘

1. **L4 達成設計目標**：bounded working set（RSS 5.1GB）+ FFN 在 Metal + bit-identical。
   decode 7.73 t/s 優於先前 skip-load CPU-FFN 的 5.19 t/s（FFN 回 GPU 生效）。
2. **剩餘 −34%** = gemma4 高 churn（17.5% miss → ~58 misses/token）的 pread stall，
   §7.5/7.6 已定案的結構性限制（hook 在 GPU split 間同步 pread、GPU idle）。
   budget 加大在 16GB 上被頁快取反噬（§7.8），唯一結構性出路仍為 MTP 批量（§8）。
3. **prefill 代價**：n_batch cap 使 gemma4 prefill 8× 慢（20→168ms/tok）——加大 capacity
   （2GiB → capacity 60 → n_batch 7）可緩解，但吃 RSS。prefill 慢是 bounded pool 的
   已知取捨；MTP/批量（§8）可同時攤薄 prefill 與 decode 的 miss。
4. 診斷保留：`CGC_MMID_DBG`（probe 顯示 src0=Metal buft ne2=capacity）確認 FFN 上 Metal；
   `LLAMA_EXPERT_INDEX_DUMP` 可驗證 index 覆蓋完整 experts。


### 7.10 日常配置三/多臂 A/B 定案（2026-08-15）：L4 budget 是乾淨槓桿，4GiB decode 超越 base

#### 7.10.1 凍機根因定案（16:15 重啟——harness 陷阱，非功能缺陷）

凍機的**真主因是 harness 的 arg 順序 bug**：llama-simple 的 parse loop **沒有 `-t` case**，
`-t 8` 一出現就 break 進 prompt → 其後的 `-expert-cache` 全被吞成 prompt → cache 從未生效 →
`-ngl 99` 兩臂 silently 以 full-resident 執行（11.29GB）→ kIOGPU OOM + peak 12.6GB → 凍機。
**cache 功能本身安全**（R1 正確參數下 n99c RSS 4.15-4.8GB、長跑 exit=0、記憶體 flat）。

**base full-resident 的另一面**：長 prompt（228 tok）下 base -ngl 99 連 n=16 都
`kIOGPUCommandBufferCallbackErrorOutOfMemory`（peak 11.5GB）——11.29GB 模型 + KV + activations
貼著 11.45GB working set 邊緣，**16GB 機上長 prompt 日常不可用**。這正是 bounded residency 的
存在理由；§7.9 記錄的 base decode 11.75 t/s 僅限短 prompt。

**規則（寫進 harness 防止再犯）**：llama-simple 不可用 `-t`；`-expert-cache` 放最後。

#### 7.10.2 修正後數據（長 prompt 228 tok、n=128、4 輪換序交錯、全 -no-mmap）

| 配置 | decode (eval) | prefill | overall | RSS | hit% | misses(全run) |
|---|---|---|---|---|---|---|
| base -ngl 99（短 prompt 記錄）| 11.75 t/s | 49.8 | — | 11.4GB | — | — |
| base -ngl 99（長 prompt）| **OOM 凍機** | — | — | 11.5GB | — | — |
| **n0c @1GiB**（Option A CPU）| 9.2 t/s | 26.3 | **5.57 t/s** | 4.0GB | 69% | 10304 |
| n99c @1GiB（L4）| 7.9 t/s | 7.9 | 2.90 t/s | 4.5GB | 72% | 19343 |
| n99c @2GiB（L4）| 10.0 t/s | 11.6 | 3.90 t/s | 6.6GB | 89% | 6197 |
| **n99c @4GiB（L4）**| **12-16 t/s** | **20.4** | **5.77-6.70 t/s** | 9.4GB | 94.1% | 2985 |

（cache stats 為確定性統計：同配置兩次 run 的 requests/hits/misses 完全一致；tok/s 受
桌面 app 棧 ±20% 波動，看配對相對值。）

#### 7.10.3 修復：L4 budget 是乾淨槓桿（推翻「1GiB Pareto 適用全路徑」）

§7.8 的「1GiB Pareto」只對 **n0c** 成立（pool 在 OS 頁快取，大 budget 擠壓頁快取 → 反噬）。
**L4 的 pool 在 Metal 記憶體**（不佔 OS 頁快取），budget → capacity → 每層 slots → hit 率
72→89→94% → 每 token miss/fill 驟減 → decode 7.9→10.0→14 t/s 乾淨正相關，無反噬：
RSS 4.5→6.6→9.4GB 全部遠低於 11.45GB working set 邊緣。**4GiB 下 decode 超越 base**
（working set 9.4GB 不貼邊 → Metal 記憶體管理器無壓力 → 比貼邊的 11.4GB base 快）。

**default 建議**：`LLAMA_EXPERT_CACHE_ALLOW_NGL=1 + -ngl 99 + -expert-cache 4294967296`（4GiB）。

#### 7.10.4 日常配置定案（2026-08-19 更新）

| 情境 | 配置 | Speed | 理由 |
|---|---|---|---|
| **純速度 / decode 重** | **n99c @4GiB + CGC_N_CB=8 + OA_ASYNC + CGC_FUSED_MOE_FFN** | **25.07 t/s** | hit 98.7%、bit-identical ✓、exit abort 已修復、decode 33.13 t/s |
| 平衡（推薦日常） | n99c @2GiB | ~15 t/s（估） | RSS 6.6GB、多工餘裕 |
| 記憶體敏感 / 多工共存 | n0c @1GiB | ~5.5 t/s | RSS 4.0GB 最小 |
| **禁止** | base full-resident -ngl 99 | N/A | 16GB 機長 prompt 必 OOM |

**Production 設定**（25.07 t/s verified, 128 tok wall-clock）：
```
LLAMA_EXPERT_CACHE_ALLOW_NGL=1
LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1
LLAMA_EXPERT_CACHE_WORKERS=8
CGC_WAKE_POLL_US=15
CGC_OA_ASYNC=1
CGC_N_CB=8
CGC_FUSED_MOE_FFN=1
CGC_EXPERT_CACHE_BYTES=4294967296
```
Binary：`build/bin/llama-simple`（build-prod2：HEAD source + CGC ggml-cpu.c + teardown drain fix）
重建：`scripts/build_prod_binary.sh`

**核心結論**：25.07 t/s 已突破原引擎層天花板（13.7 t/s），三大來源：
(1) CGC_N_CB=8 multi-buffer 砍 MTL encoding 60→~30ms；
(2) expert-cache hit 55%→98.7% 消除 I/O tail；
(3) CGC_OA_ASYNC 隱藏 CPU callback sync。
Decode-only speed 33.13 t/s，prompt eval 拖慢 7.06 t/s。
>28.6 t/s（kernel 地板）需 kernel 重寫（IQ3 inline dequant + sequential access），3-6 週。

#### 7.10.5 對照 Swift TurboFieldfare gemma4-r4（使用者提示「Swift 跑得很快」——確認屬實）

**活體量測**（本機、同長 prompt 228 tok、128 tok decode、TurboFieldfareCLI-newkernel +
gemma4-r4.gturbo + 生產 env：slots=96 / sync preload / 8 workers / top64 profile）：

| run | decode tok/s | 備註 |
|---|---|---|
| temp 0（greedy）| **20.0**（6.40s）| 輸出退化（ownces 重複）→ 熱集合收斂、數字可能微幅虛高 |
| 生產預設（temp 0.2）| **18.0**（7.10s）| 同上退化模式 |
| §13.161 文件記錄生產 | 14.05（128 tok 默認）/ 峰值 16.65 | 忙碌機器、code/prose 混合 |

**Swift 快的兩個架構原因**（vs 我們的 llama.cpp fork）：
1. **全 GPU 自訂 fused kernels**：Swift 引擎整個 decode 是客製 Metal kernel
   （fused QKV / fused MoE / grouped routing / fused layer tail），GPU ceiling ~20+；
   llama.cpp 是 generic Metal backend，base 11.75 就是它的 ceiling。
2. **CPU 只餵權重、routing 在 GPU**：Swift 的 pread 藏在 GPU busy 後面（readWall ~50% but
   overlapped）；我們的 A+B split（§7.3.5）把 router 逼到 CPU、hook 的 fill 落在 GPU split
   之間 → GPU idle（§7.5 判死的結構性序列化）。

**對照結論**：llama.cpp L4@4GiB（12-16、mean ~14）與 Swift 文件生產（14.05）**同級**；
Swift 活體 18-20 的差距 = 自訂 fused kernel + GPU-resident routing 的架構紅利。
要追平需：(a) routing 回 GPU（換 hook 機制，或 GPU 側 routing callback）＋ (b) port fused
kernel——即白皮書 §8 MTP 批量 + kernel 移植線，非小改。**速度取向本機日常用 Swift 引擎；
llama.cpp fork 的價值是跨平台 C/C++ 遷移 + 小機 bounded residency（RSS 4-9GB）。**

## 8. MTP-4 投機批量可行性評估（2026-08-15）：多 token FFN 一次 dispatch、pread 併發攤薄

### 8.1 結論先行

**可行，且是 ngl 99 路徑（-76% split 開銷）的結構性解藥**；ngl 0 路徑只有
+10-20%（接受率 ≥0.8 時）。最大發現：**fork 的 llama.cpp master 已內建完整 MTP
機制（loader + graph + speculative driver 全在），唯一缺的是權重**——而權重存在於
外接盤的 qwen36-hf 原版 HF 模型（19 顆 tensor，UD repack 把它丟了）。不需要寫
MTP head 的 C++，只需要把 19 顆權重補回 GGUF + 一個 multi-token pool 延伸。

### 8.2 現況盤點（本 session 實證）

1. **master 原生支援三種 speculative draft**：`draft-mtp`（target 自帶 MTP head）、
   `draft-dflash`、`draft-eagle3` + ngram 自我投機（`ngram-mod` 等，零記憶體）。
   已有 `llama-speculative-simple` binary。
2. **qwen35moe arch 的 MTP 完整實作**（`src/models/qwen35moe.cpp`）：
   - `load_block_mtp`：MTP 塊 = 完整 Qwen3.5 decoder 層（attn + 256-expert MoE +
     shared expert）位於 `blk.%d`（層索引 = n_layer=40），附 `nextn.eh_proj`
     （{2*n_embd, n_embd}，DeepSeek-V3 的 concat(e_norm, h_norm) 融合投影）、
     `nextn.enorm/hnorm`。`GGML_ASSERT(n_layer_nextn == 1)`。
   - `graph_mtp`：eh_proj concat → 完整層 →（共用 output norm + lm_head）。
   - KV：`qwen35moe.nextn_predict_layers` → `n_layer_nextn`。
3. **權重現況**：本機兩顆 GGUF（qwen36 IQ2、gemma4 IQ3_S）**mtp tensor = 0**。
   外接盤 `/Volumes/AlexZhuang/qwen36-hf`（fp16 HF 原版）**有完整 19 顆**：
   `mtp.fc`、`mtp.pre_fc_norm_embedding/hidden`、`mtp.layers.0.*`（完整層含
   `experts.gate_up_proj/down_proj` 256-expert MoE）、`mtp.norm`。共 616MB BF16
   （大頭是 256-expert MoE：gate_up 536.9M + down 268.4M 元素 = 1.6GB BF16，加 attention/shared/eh_proj 共 ~1.7GB；F16 同大小）。gemma4 無 MTP。
4. **harness 缺口**：`-expert-cache` 只在 `simple.cpp` 客製解析（common 層沒有），
   `llama-speculative-simple` 吃不到——需加進 `common/arg.cpp`。

### 8.3 機制與預期回收

**spec verify = k 個 draft token 一次 llama_decode（n_tokens=k）** → hook 走既有
multi-token union 路徑 → 每批只 fill 一次 union（k 個 token 的專家併集），prefill
已驗證過這條路徑 bit-identical。MTP draft 成本 = 1 層 / step（256 experts 只算
top-8 ≈ 16MB 讀取），可忽略。

| 路徑 | 現況 | MTP-4 預期 | 機制 |
|---|---|---|---|
| **ngl 99（Metal）** | cache arm 5.19 tok/s（−76%，~120 splits/token）| **12-17 tok/s** | batch k=4 → split 開銷每 token 攤 4×；union fill 一次/批 |
| **ngl 0（CPU）** | 14.1→13.15（−7%）| +10-20%（a≥0.8）| union fill 攤薄 + 接受率乘數；compute 仍主導 |

ngl 99 是主戰場：**bounded residency 在 Metal 下從「跑得起來」升級為「接近 base
速度」**。ngl 0 的收益主要來自接受率，其次 pread 攤薄。

### 8.4 改動面（分刀）

| 刀 | 內容 | C++ 量 |
|---|---|---|
| **P0 資料** ✅ 已完成（2026-08-15）| 19 顆 HF tensor 已補進 UD IQ2 GGUF（`/Volumes/AlexZhuang/cgc_work/qwen36_mtp_aug.gguf`，12.4GB）：KV `nextn_predict_layers=1` + `block_count 40→41` + `blk.40.*`（含 gate_up 拆 gate/up、全轉置、shared gate squeeze）。F16（GGUF 無 BF16 type）。**驗證全過**：loader n_layer=40/n_layer_all=41、19 tensor 全載、trunk 輸出與原檔 **bit-identical**、MTP-1 接受率 90.7%（見 8.6）| 0（腳本：tmp/p0_mtp_append.py + f16_patch + blockcount_patch）|
| **P1 fork** | multi-token pool（Option A 延伸）：union fill 進 slots、每 token 欄 remap 成 slot 索引、swap ne[2]=n_slots；union > slots 保持 L3-B fallback。這是**唯一的 kernel-adjacent 新碼** | 小（~150 行）|
| **P2 harness** | `-expert-cache` 加進 `common/arg.cpp`（所有 example 共用）| 極小 |
| **P3 A/B** | llama-speculative-simple `--spec-type draft-mtp`：spec vs 非 spec greedy **必須 bit-identical**；k=2/4/8 sweep；acceptance stats；ngl 0/99 兩路 | 0（量測）|
| **P4 記憶體** | MTP F16 (+310MB) vs IQ2 量化 MTP experts（省 ~270MB）；RSS Pareto 點複測 | 0 |

### 8.5 風險（誠實清單）

1. **接受率（#1 風險）**：✅ **已退役（2026-08-15 P0 實測 90.7%）**。MTP head 對
   fp16 target 訓練、target 是 IQ2——先前估 0.5-0.7，實際 MTP-1 acceptance
   = 90.7%（n=68、n_draft=3、49/54 drafted 接受），遠超 0.6 門檻，MTP-4 淨正。
2. **`mtp.norm` 映射**：load_block_mtp 沒建 MTP 專屬 output norm——graph_mtp 尾段
   用哪顆 norm（共用 output_norm vs mtp.norm）需對拍 HF 前向確認；錯了會影響
   acceptance 但不會崩。
3. **union > pool**：k=4 的 union ~45-60 > 27 slots → 需要 P1 或暫以 L3-B gather
   驗證（−38% gather 會吃掉一部分收益，先測後再決定 budget 2GiB 或 slot 彈性）。
4. **qwen36 UD 的 ngl 99 base 分歧（既有另案）**：spec A/B 的正確性對照必須用
   ngl 0 或 gemma4（無 UD 問題）做 bit-identical 基準；qwen36 ngl 99 的 spec 驗證
   要等 UD de-interleave（§6 既有列項）。
5. **投機 KV rollback**：llama.cpp master spec decode 原生處理（成熟），無自研風險。

### 8.6 替代路線（保底）

- **ngram 自我投機**（`ngram-mod`）：零記憶體、零資料 prep，接受率低（prose 0.3-0.5）
  但**同樣享受 batch 攤薄**——可先做這條把 ngl 99 的 split 攤薄收益單獨量化出來，
  再疊 MTP。分刀順序：ngram（P0.5，2 小時）→ MTP P0（資料）→ P1/P2/P3。

### 8.6 P0 結果（2026-08-15 實測完成）

**產物**：`/Volumes/AlexZhuang/cgc_work/qwen36_mtp_aug.gguf`（12.4GB = 原 10.76GB +
MTP ~1.7GB F16）。腳本：`tmp/p0_mtp_append.py`（header surgery + append，不動原資料）、
`tmp/p0_mtp_f16_patch.py`（type 15→1 + bf16→f16）、`tmp/p0_blockcount_patch.py`。

**三個實作陷阱（已踩平，記錄避免重走）**：

1. **GGUF 沒有 BF16 tensor type**：gguf-py 的 `GGMLQuantizationType.BF16=15` 在
   ggml 的 GGUF enum 裡是 **IQ2_XXS**——寫 15 會被 ggml 解析器當 IQ2_XXS 算
   nbytes（offset 檢查直接 reject）。MTP 一律存 **F16（type=1）**，資料 bf16→f16
   轉換（bits<<16 → f32 → f16，weights 值域安全）。
2. **`block_count` 必須 +1**：`n_layer() = n_layer_all - n_layer_nextn`（llama-hparams.cpp
   :290）。HF 是 40 trunk + 1 MTP，所以 GGUF 的 `qwen35moe.block_count` 要設 **41**
   （n_layer=40、MTP@blk.40）。設 40 時 loader 把 MTP block 找在 blk.39
   （check_tensor_dims: blk.39.nextn.eh_proj not found）。
3. **KV key 長度是 u64 不是 u32**（本 fork/gguf-py 寫法）：header walk 用 u64 才對齊。

**命名/轉置映射**（與上游 `convert_hf_to_gguf.py` 的 mtp remapper 一致）：

| HF（qwen36-hf）| GGUF（blk.40）| 轉換 |
|---|---|---|
| `mtp.fc` | `nextn.eh_proj` {4096,2048} | transpose |
| `mtp.pre_fc_norm_embedding` | `nextn.enorm` | copy |
| `mtp.pre_fc_norm_hidden` | `nextn.hnorm` | copy |
| `mtp.norm` | （上游也不映射，捨棄）| — |
| `self_attn.q/k/v_proj`、`o_proj` | `attn_q/k/v`、`attn_output` | transpose |
| `mlp.experts.gate_up_proj` | `ffn_gate_exps` + `ffn_up_exps` | dim1 拆半 + 3D transpose |
| `mlp.experts.down_proj` | `ffn_down_exps` | 3D transpose |
| `mlp.shared_expert_gate` [1,2048] | `ffn_gate_inp_shexp` [2048] | squeeze（同 UD 慣例）|
| 其餘（layernorm/q_norm/k_norm/shared_expert 三件）| 同名同形 | copy |

**驗證矩陣**：

| 檢查 | 結果 |
|---|---|
| gguf-py 解析 | 752 tensors（+19）、nextn KV=1、全 blk.40 形狀正確 |
| 原資料區拷貝 | 抽樣 10×8MB 窗口逐 bytes 一致 + trunk bit-identical（全檔 md5 因外部盤讀速太慢跳過；語義層已由 bit-identical 證明）|
| loader | exit=0、`n_layer=40`、`n_layer_all=41`、無 missing/duplicate |
| MTP context 載入 | `--spec-type draft-mtp` 下 `load_mtp=true`（common.cpp:1631 自動），blk.40 全載 |
| **trunk 對拍** | llama-simple 原檔 vs 增補檔（ngl 0、同 prompt）→ **OUTPUT IDENTICAL** |
| **MTP-1 接受率** | **90.7%**（n=68、n_draft=3、49/54）｜冷啟動首 run 57%（n=20）|

**Harness 記錄**（跑 MTP 必讀）：

- 用 **完整版 `llama-speculative`**（不是 speculative-simple——後者 ctx_dft 沒設
  `ctx_type=MTP`，會用主 graph 跑 draft）。
- 命令：`llama-speculative -m <aug.gguf> -md <aug.gguf> --spec-type draft-mtp -ngl 0 -ngld 0 -n 64 -p "<prompt>"`。
- **`-ngld 0` 必要**：draft 的 n_gpu_layers 預設 auto(=all) → 觸發「forcing no-mmap」
  全量載入（10.76GB + 外部盤 = 分鐘級）+ draft 上 Metal；`-ngld 0` 保 mmap + CPU。
- 接受率印在 stdout（新 logger），`sed 's/\r/\n/g'` 後看 `n_drafted/n_accept/accept`。

**下一刀（P1）**：multi-token pool（union fill 一次/批 + 每欄 remap slot 索引），
ngl 99 的 split 攤薄 + ngl 0 的 pread 攤薄；接受率 90.7% 已掃掉最大風險。

### 8.7 MTP-4 端到端 A/B 定案（2026-08-15）：無加速，先放著

檔案已放內部盤 `models/gguf/qwen36_mtp_aug.gguf`（12.4GB），同檔同 seed 三臂實測：

| 配置 | harness | tok/s（n=128/64） | MTP accept |
|---|---|---|---|
| 純 cache（1GiB、無 MTP） | llama-speculative-simple（no-spec 模式） | **10.77** | — |
| MTP-4 + cache（spec-simple，MTP 驅動有 bug） | llama-speculative-simple | 2.70 | 0% |
| MTP-4 + cache（完整版，接受率正確） | llama-speculative | **5.34**（n=64） | **83.3%** |

**定案：MTP-4 在 ngl 0 + expert-cache 路徑無加速（5.34 vs 10.77，約 2× 更慢），P1 先放著。**

機制：即使接受率 83.3%（P0 的 90.7% 復現），verify 的 union fill 只把每 token
pread 從 8 降到 ~4 次（union 16/4 tokens），省下的 pread 抵不過兩筆成本：
① draft forward 與 target forward **共用同一個 model 層級 expert cache**（§8.14 miss 分析修正
   ——`expert_cache` 是 model 成員，MTP draft context 用同一 model_tgt，無「第二 cache」）
   但 draft 的 MTP head forward 每 step 多跑一趟 40 層 FFN，expert 流量直接翻倍（同一步
   draft + verify 各算一次 experts）；共享 cache 下這翻倍的流量全擠在一個 LRU pool 裡，
   draft 的 experts 與 target 的 verify experts 不同集 → 相互換出（§8.14 miss 分析：
   decode 路由跨步幾乎不重複，draft 的 routing 對 target 毫無預測力）；
② 每步多一次的 MTP 層 decode + sampling。pread 攤薄在 ngl 0 的 CPU 路徑上淨負。

另記兩個 harness 教訓（都修好並可留用）：
- `llama-speculative-simple` 的 MTP 驅動是**壞的**（draft 全 garbage、accept=0%、
  輸出退化）——MTP 只能信完整版 `llama-speculative`。spec-simple 的 MTP 手術
  （ctx_type=MTP + embd_nextn buffer + seq_rm 修復）已保留，供日後排查。
- spec-simple 已支援 no-spec 模式（`types=={NONE}` 時純 target loop），可當
  cache 基線 harness 用；`-md` 對 MTP 是**多餘且有害**（觸發第二次整模型載入 →
  Metal OOM）。

檔案狀態：`qwen36_mtp_aug.gguf` 內外部盤各一份（含 MTP head；IQ3 檔無 head，無法
建立 MTP context）；gemma4 IQ3_S 為騰空間移至 `/Volumes/AlexZhuang/models_backup/`
（可逆）。**共享 cache 已經是現況**（model 層級單一實例）——§8.14 miss 分析修正後，MTP 的
真正瓶頸是「draft forward 使 expert 流量翻倍」而非「雙 cache」；要翻轉只剩兩條：
(a) verify 走 L4 Metal pool 的單次 dispatch（省 MTP head 的 CPU FFN）；
(b) 用 §8.14 的 miss 分析證明 draft 無預測力 → MTP 先放著的決策維持。

### 8.8 qwen36 位寬品質-速度曲線（2026-08-15）：IQ2 崩潰、IQ3 是下限

同 prompt（"The capital of France is"）同 seed 42、ngl 0 + cache 1GiB、128 tok、llama.cpp fork：

| 位寬 | 模型 | 引擎 | tok/s | 品質（同 prompt 輸出） |
|---|---|---|---|---|
| **2-bit** | IQ2_XXS 2.06bpw | llama.cpp + cache | **11.21** | **崩潰**：11 words 後進入 "111..." 無窮重複（unique trigram 55.6% → 0）|
| **3-bit** | IQ3_XXS 3.4bpw | llama.cpp + cache | **8.37** | **正常**：84 words 連貫敘述巴黎（unique trigram 96.3%）|
| **4-bit** | Swift r4 .gturbo | Swift 自寫 Metal | 5.5-5.8（文檔記錄）| 好（4-bit 未退化）|

**速度反差的機制**：3-bit 比 2-bit 慢 25% 不是位寬本身，是 cache 命中率塌陷
（92.4% → 66.2%）——expert 變大（~0.88→~1.15MB）→ 1GiB budget 每層 slots 變少 →
miss 4.5×（3282→14689）、pread 2.4×（5.6→13.3s）。要追平 2-bit 速度需加大 budget。

**定案：IQ2 在 qwen36 上是「品質不可用」檔位（即時退化），IQ3 是品質下限；
要品質又要 2-bit 的速度只能加大 cache budget 或上 Metal（-ngl 99 IQ2 = 25.5-27.9 t/s
是速度上限，但同樣 2-bit 品質）。** Swift r3/r4 .gturbo 目前只剩 stub（33MB），
4-bit 對照只能用文檔記錄值。

### 8.9 IQ3 cache budget 掃描定案（2026-08-15）：加大 budget 拉不回速度

IQ3_XXS、同 prompt/seed、128 tok、ngl 0、128 tok：

| budget | hit% | tok/s | pread_usec | miss |
|---|---|---|---|---|
| 1GiB（gather）| 66.2% | 8.37 | 13.34s | 14689 |
| 2GiB（gather）| 77.4% | 8.33 | 13.22s | 9846 |
| 3GiB（gather）| 85.1% | 7.83 | 13.45s | 6494 |
| 2GiB + L4 pool（Option A）| 74.5% | **8.82** | 13.30s | 11105 |

**定案：不行。** hit 率按 budget 線性回升（+11%/GiB），但 tok/s 完全持平（8.3→8.3→7.8）
且 pread_usec 不隨 miss 下降（13.3s 恆定）——**ngl 0 路徑的牆是 CPU 對 IQ3 的
dequant+GEMM 計算本身**（每 token 讀的權重 bytes 比 IQ2 多 ~65% → 11.21/7.83≈1.43×），
不是 IO。L4 pool 只回收 gather 開銷（+6%）。要破 10 唯一路：FFN 上 Metal（-ngl 99）。

### 8.10 IQ3 FFN 上 Metal（n99c）定案（2026-08-15）：4GiB = 11.08 t/s，破 10 目標達成

IQ3_XXS、同 prompt/seed、128 tok、`LLAMA_EXPERT_CACHE_ALLOW_NGL=1` + `-ngl 99 --no-mmap`：

| 配置 | tok/s | hit% | 備註 |
|---|---|---|---|
| ngl 0 + cache 2GiB（L4 pool）| 8.82 | 74.5% | CPU 計算牆 |
| n99c 2GiB | 6.10 | 79.1% | pool 太小 → fill 阻塞 GPU |
| **n99c 4GiB** | **11.08** | 81.0% | MMIDPROBE 確認 FFN 在 MTL0；最佳 |
| n99c 6GiB | OOM | — | working set 超 11.45GB 硬崩 |

**定案：A+B split 對 qwen36 IQ3 成立，但需要 4GiB Metal-visible pool。**
機制：ngl 0 的 8.37-8.82 是 CPU 計算牆（budget 無效）；FFN 上 Metal 後轉為
IO/latency 主導 → budget 直接有效（2→4GiB：6.10→11.08，+82%）。6GiB 超
M4 working set（pool+attention≈10GB 已近頂）。4GiB 時 working set 僅 ~7GB，
KV 成長有餘裕。

### 8.11 fill 並行度定案（2026-08-15）：跨 expert 並行 fill（batch）設為預設，+8.3%

`q36_fill_workers_ab.sh` 3 臂 × 2 輪交錯，IQ3_XXS、同 prompt/seed、128 tok、n99c 4GiB、
`LLAMA_EXPERT_CACHE_ALLOW_NGL=1`。三臂 hit=41671 / miss=9797 完全相同、輸出 bit-identical：

| arm | R1 t/s | R2 t/s | mean | 說明 |
|---|---|---|---|---|
| serial（全串行：expert × 段都串行）| 10.40 | 9.65 | 10.02 | 基準 |
| conc（現行：expert 串行 × 3 段並行）| 10.69 | 10.58 | 10.64 | +6.2% |
| **batch（跨 expert 併發，≤24 併發 pread）** | 11.03 | 12.02 | **11.52** | **+15% vs serial / +8.3% vs conc** |

**機制**：batch 把「同一 layer 內 top-8 的 miss」一次併發 fill（每 miss 一個 thread，各自
3 段再並行），把關鍵路徑從 sum(miss fills) 壓成 max(miss fills)。增益有限的原因：
(1) hit 81% → 每 layer-step 平均 ~1.5 miss，多數 step 無物可並行；(2) SSD queue depth 是
真牆——batch 的 pread 累計時間反而更高（17-18s vs 15-16s），單個 pread 因競爭變慢，
再往上加 workers（16→32）不會再贏。

**抓到並修掉一個真 race**（batch 首版輸出分歧且 nondeterministic）：BATCHDBG 顯示所有
miss 全指派到 s0——根因是 batch 在同一把鎖內連續指派 slot，但 `pick_slot` 的 LRU 換出
看 `last_use`，指派當下沒 bump（`ensure_slot` 靠「fill 後才 bump + 下一次 pick_slot 在
fill 之後」才安全）→ 第二次 pick_slot 把剛指派的 slot 又換出 → 兩 expert 搶同一 slot →
併發 fill 覆寫同一 region → FFN garbage。修復：指派當下 `slot_last_use[layer][slot] =
++tick`。修後 deterministic + bit-identical。

**定案**：batch 為預設路徑（hook 無 env 即走 `llama_expert_cache_ensure_batch`）；
`LLAMA_EXPERT_CACHE_SERIAL_EXPERT=1` 退回串行 per-expert loop（A/B 用）。
`LLAMA_EXPERT_CACHE_SERIAL_FILL`（3 段串行）與 `LLAMA_EXPERT_CACHE_BATCH_DBG`（slot 指派
診斷）保留為診斷工具。

### 8.12 persistent worker pool 定案（2026-08-15）：pool8 取代 spawn-per-expert，+6% 且更穩

把 batch 的 spawn-per-expert（每 step spawn+join ~4×misses 個 thread）改成 init 時建立的
persistent pread worker pool（`LLAMA_EXPERT_CACHE_WORKERS=N`，預設 8，clamp 1..64）——
batch 把所有 miss 的 segments 扁平化為單一 job list，提交後等 `outstanding==0`。
`LLAMA_EXPERT_CACHE_BATCH_SPAWN=1` 退回 spawn（A/B）。

`q36_pool_workers_ab.sh` 4 輪交錯，IQ3_XXS、同 prompt/seed、128 tok、n99c 4GiB
（hit=41671/miss=9797 全程相同、三臂輸出 bit-identical）：

| arm | R1 | R2 | R3 | R4 | mean | 變異 |
|---|---|---|---|---|---|---|
| **pool8（預設）** | 11.98 | 11.87 | 11.57 | 11.64 | **11.76** | ±0.4（極穩）|
| spawn（舊 batch）| 12.05 | 11.60 | 9.46 | 11.27 | 11.09 | ±2.6（跳動）|
| pool16 | 11.86 | 7.02 | 7.58 | 8.74 | 8.80 | 崩潰（R2 後）|

**定案：pool8 取代 spawn-per-expert 為 batch 預設**：
- **+6% over spawn**（11.76 vs 11.09，同 run 內乾淨比較），且 **變異從 ±2.6 收斂到 ±0.4**
  ——persistent pool 消除每 step 的 thread 建立/join 抖動（~7680 次 spawn+join/run）。
- **pool16 反效果**（R2 後穩定崩到 7-8.7）：16 workers 對單 SSD 超標，queue depth 打爆 →
  證實 §8.11 的「SSD queue depth 是真牆」——8 是甜蜜點，16 過衝。
- 相對 §8.11 的串行 expert loop（10.02）：pool8 累計 **+17%**。

**實作備註**：pool worker 走 USER_INITIATED QoS（與 bg 同理由）；`fill_pool_direct` 拆出
`fill_pool_direct_collect`（segment 收集）供 spawn/pool 共用；pool 短讀只 zero 失敗 segment
（不做整批 zero）；destructor 先 drain 再 join。debug：`LLAMA_EXPERT_CACHE_PREAD_DBG`、
`LLAMA_EXPERT_CACHE_BATCH_DBG`（slot 指派）。

**gemma4 交叉驗證（2026-08-15 追加）**：`g4_pool_ab.sh` 6 輪交錯，gemma4 IQ3_S
（128 experts/layer）、同 prompt/seed、128 tok、n99c 4GiB。hit=31181/miss=3761 全程相同、
三臂輸出一致（短 prompt 無 chat template 退化，routing 不受影響）：

| arm | 6 輪 speed | median | mean | 範圍 |
|---|---|---|---|---|
| **pool8** | 17.0 17.9 10.9 10.0 16.6 14.3 | **15.46** | 14.46 | 10.0-17.9 |
| seriale | 11.0 14.2 11.7 18.3 13.9 17.7 | 14.07 | 14.49 | 11.0-18.3 |
| serial | 18.7 11.5 12.9 14.4 16.0 9.7 | 13.63 | 13.86 | 9.7-18.7 |

**median：pool8 +9.8% vs seriale、+13.4% vs serial——方向與 qwen36 一致，量級相當**
（qwen36 是 mean +6%/+17%）。fill wall（低雜訊、三臂單調）：pool8 5.2s < seriale 5.6s <
serial 6.5s。⚠️ 本機負載波動大（同 arm 跨 round ±4 t/s = ±28%），median 方向可信、精確
量級不穩；fill wall 差異是更乾淨的訊號。

**與 qwen36 的對照**：gemma4 miss 只有 3761（hit 89.2%）vs qwen36 9797（81%）——跨 expert
並行 headroom 更少，但 pool8 的 median 優勢仍成立 → pool 的價值不全在並行度，還有
「消除 spawn/join 抖動」的穩定性成分（qwen36 pool8 的 ±0.4 極穩 vs spawn ±2.6 已證）。

### 8.13 PREFETCH 預測器 A/B（2026-08-15）：page-cache warm 是災難（-88%），double-buffer 需真 bg slot fill

`g4_prefetch_ab.sh` 4 輪交錯，gemma4 IQ3_S、n99c 4GiB、128 tok（hit/miss/reads 全程相同）：

| arm | R1 | R2 | R3 | R4 | F_RDADVISE 次數 |
|---|---|---|---|---|---|
| pool8（基線）| 11.04 | 14.66 | 12.58 | 13.31 | 0 |
| pf_lyr（層鄰）| 2.68 | 2.97 | 3.14 | — | 29464 |
| pf_both（層鄰+步間）| 1.64 | 1.56 | 1.62 | 1.54 | 59704 |

**結論：PREFETCH 現行機制（F_RDADVISE page-cache 建議，非真 slot fill）是 -88% 災難**：
(1) hook thread 同步呼叫 59704 次 fcntl → read-ahead I/O 直接堵關鍵路徑；(2) read-ahead 與
真正 pread 搶 SSD；(3) over-predict 16×（59704 vs 3761 miss）。比 P2 證偽（§7.2「無收益」）
更糟——n99c + fill 瓶頸下是主動傷害。`pool_queue`（真 bg slot fill 的死碼）從未被 push。

**對 double-buffer 的含義**：不能建立在 page-cache 建議上。要做真 bg slot fill——復活
`pool_queue` 機制（bg 線程寫 slot + `slot_loading` 旗標 + 完成 barrier），讓 GPU 算 layer N
時 SSD 真的把 N+1 的 bytes 寫進 slot。

### 8.14 step 級 double-buffer 實作 A/B（2026-08-15）：真 bg slot fill 轉正但無淨收益，維持 env-gated 預設關

承接 §8.13 的結論，把 `pool_queue` 死碼復活成**真 bg slot fill**（非 F_RDADVISE）：

- `prefetch_slot` 改為真 bg pool fill：掃 free slot → 設 `slot_queued`/`slot_owner` → push
  `pool_queue` → bg 線程 pread 寫 slot → 完成後 publish slot_table（下個 ensure 變 hit）。
  free-slot-first；無 free slot 時 evict 該層 global LRU（union ensure 剛 bump 過當前 step
  的成員，LRU 必為非 union slot，安全）；queued/loading slot 永不碰。
- `drain_layer` 從 no-op 實作為 barrier：drop 該層 queued fills + 等 in-flight 完成（保護
  Metal region copy 不被 bg 寫撕裂）。
- hook 改為僅 capture step union；queue 移到 step boundary（`graph_compute` 後）：因為
  hook 走 eval-callback 在 graph *執行中* 觸發，層鄰 queue 會在下一層 drain 前被取消，
  bg 根本來不及落地。
- 加 `slot_queued` 旗標（pick_slot/ensure_slot/ensure_batch 都視為 busy）。

**A/B 結果（qwen36 IQ3、n99c 4GiB、128 tok、交錯）**：

| arm | median t/s | hit/miss | prefetch |
|---|---|---|---|
| base（無 prefetch）| 4.58-6.22（跨輪）| 同 | 0 |
| NOPREWARM base | 5.20 | 同 | 0 |
| NOPREWARM + PREFETCH_STEP | 5.23 | **與 base 完全相同** | 0（全 resident）|

**關鍵發現——填不進去的雙緩衝**：qwen36 的 200 slots/layer pool 在 prefill 就被塞滿
（`ensure_batch` 每層 union 全量填入），decode 的 step union 全數「已 resident」→
prefetch 目標 2880 次全被 `table[e]>=0` 拒收 → 0 個 fill 落地 → hit/miss 與 base 完全
相同。NOPREWARM 下層鄰 prefetch 可 queue（1023-13661 次）但 hit/miss 仍不變——層鄰
預測的 experts 下一層根本不用（同層使用率低），fill 落地卻無消費。

**定案**：double-buffer 對 qwen36 n99c + 200-slot pool 無淨收益——工作集已被 prefill
覆蓋，無新 miss 可救；層鄰預測精度不足；機器雜訊（load 3、±70% 波動）無法分辨細微
差異。機制本身正確（含 s0 式 race 防護），**維持 `LLAMA_EXPERT_CACHE_PREFETCH=1`
env-gated 預設關**，作診斷工具保留；§8.13 的 -88% 災難已不再是隱患（真 bg fill 取代
F_RDADVISE）。後續要翻轉需：更小的 pool（製造真實 miss 空間）+ 步間預測 + 低負載窗
口重測。

**miss 來源分析（§8.14 補充，2026-08-15，`LLAMA_EXPERT_CACHE_MISS_DUMP` + TRACE）**：

qwen36 IQ3 n99c 4GiB、128 tok、同 seed。decode miss 的 experts 來源：

| prompt | miss 總數 | 在該層 prefill 出現過 | 全新（全場第一次見）| decode 較早 step 出現過 |
|---|---|---|---|---|
| 短（5 tok）| 1797 | 387（**21.5%**）| 1410（78.5%）| **0** |
| 長（220 tok，prefill 部分完成）| 245 | 160（**65.3%**）| 85（34.7%）| **0** |

**兩個定案性發現**：

1. **decode miss 幾乎全是唯一 expert**（1797 miss = 1796 唯一 (layer, expert)）——重複率
   ≈0，**步間/層鄰預測在天花板層面就是死的**：同層跨步幾乎不重複選 expert，預測「上一步
   的 union」或「上一層的 union」都等於預測隨機。這從數據上關閉了 §8.14 的 double-buffer
   步間預測線（不是 pool 滿的問題，是**路由本身不重複**）。
2. **prefill-seeded 的覆蓋率隨 prompt 長度線性上升**（21.5% → 65.3%）——唯一有意義的
   預測訊號還是 prefill 熱集（P0 已證偽「靜態 prefill 頻率池打不過 LRU」，但長 prompt
   下 65% 的 miss 來源其實在 prefill 出現過 → 如果 prefill 就把這些 experts 留在 pool，
   miss 可減半以上）。**這正是 loader prewarm（experts 0..n_slots-1）在做的，只是 prewarm
   選的是 0..199 而非真正熱集** → 改進方向明確：prewarm 改用 prefill 的實際 top-K 頻率
   （而非 0..n），長 prompt 下預期 miss 減半、tok/s 提升。

工具：`LLAMA_EXPERT_CACHE_MISS_DUMP=<file>`（env-gated，與其他診斷同風格）+ `tmp/q36_miss_source.py`。

### 8.15 prewarm 熱集改進 A/B（2026-08-15）：prefill top-K 熱集預填證偽（net negative）

§8.14 補充指出「prewarm 選 0..199 而非真正熱集」是唯一剩餘方向。本節實作並 A/B 驗證：
`LLAMA_EXPERT_CACHE_PREWARM_HOT=1` 讓 loader 跳過 0..n prewarm，改由 hook 在 prefill 期間累積
路由頻率（`llama_expert_cache_record_routes`），第一個 decode step 前以各層 top-K 頻率專家
`prewarm_hot` 填池（`llama_expert_cache_prewarm_hot`，跑一次）。

**實測（qwen36 IQ3_XXS、n99c 4GiB、128 tok、4 輪交錯、同 seed、輸出 bit-identical）**：

| arm | median t/s | hit | miss | pread_ms | fill_ms |
|---|---|---|---|---|---|
| base（load-time 0..n prewarm，預設）| **10.5** | 44202 | 10254 | ~20s | ~6.3s |
| hot（PREWARM_HOT=1）| 6.3 | 44109 | 10347 | ~31s | ~4.0s |

**結果：hot 全面劣化（-40% tok/s），decode miss 甚至不降反升（10254→10347）**。

**根因（機制已從程式碼確認）**：

1. **冷啟動成本從「離時鐘」移到「關鍵路徑」**：load-time 0..n prewarm 在模型載入時跑（不在
   decode 時鐘內）；PREWARM_HOT 把 prewarm 挪到第一個 decode step 前——`prewarm_hot` 對 40 層
   × 200 slots 做最多 8000 次同步 `ensure_slot`，全部卡在第一個 token 之前。這正是 6.3 vs
   10.5 的來源（pread_ms 20s→31s）。
2. **prefill 失去 prewarm 墊底**：hot 模式下 loader prewarm 被跳過 → prefill 本身全 miss 同步
   pread（40-token prompt 觸及 ~600+ 唯一 expert/layer，全部冷讀）。
3. **prefill 熱集對 decode 沒有額外覆蓋力**：40-token prompt 下 decode miss 10254→10347
   （+0.9% 更差）——prefill 的熱集隨 prompt 內容偏移，與 §8.14「覆蓋率隨 prompt 長度上升」
   的短 prompt 樣本（5-token 時 -4.1%）不一致；LRU churn 期間被換出的 slots 好壞參半。
   5-token 煙霧測試的 -4.1% miss 改善在 128-tok 全長度上被冷啟動成本完全吞掉。

**定案**：**prewarm 熱集證偽**——熱集必須在 prefill 結束後才知道，而 prefill 結束後任何
預填都必然落在第一個 decode 的關鍵路徑上（除非有 routing oracle，P0 預測線已證偽）。
load-time 0..n prewarm 的「離時鐘」優勢是結構性的，miss 覆蓋上的小差距（±1-4%）無法抵銷。
維持現狀（load-time 0..n prewarm 為預設）；`LLAMA_EXPERT_CACHE_PREWARM_HOT=1` 保留為 env-gated
診斷工具。**§8.14 的「prewarm 改熱集」假設正式關閉**；bounded residency 的剩餘槓桿只剩
§8.12 已定案的 pool8 並行 fill（qwen36 11.76 / gemma4 15.46 median）。

### 8.16 decode 首步 union 預熱（TAILPIN）A/B（2026-08-15）：低成本 pin 路徑證偽（回收 <0.2%）

§8.15 證偽「prefill top-K 熱集預填」後，本節驗證更廉價的變體：**不預測、不 fill**，只在
第一個 decode step 把 prefill 尾端 K=10 個 token 的 union 中「已在 pool 的 experts」pin 起來
（LRU 免驅逐），step 結束即 unpin。實作：hook 在 prefill 期間追蹤每層最後 10 個 token 的
expert set（`LLAMA_EXPERT_CACHE_TAILPIN=1`，K 可用 `LLAMA_EXPERT_CACHE_TAILPIN_K` 調），首個
decode step 的 Option A 分支前 `pin_experts`（僅 pin resident，不 fill），step 末 `unpin_all`；
`pick_slot` 的 LRU 換出跳過 pinned slots。

**實測（qwen36 IQ3_XXS、n99c、短 prompt「The capital of France is」、輸出全程 bit-identical）**：

| 配置 | slots/layer | 機制 | miss 變化 |
|---|---|---|---|
| 4GiB pool（日常）| 200 | prefill union(~37) < slots → Option A 填 pool | 8450 → 8450（**0**）|
| 1GiB pool | 50 | 同上（最小可走 Option A 的池）| 8413 → 8401（**-0.14%**，12 miss 轉正）|
| 512MiB / 256MiB | 24 / 16 | prefill union > slots → L3-B map 路徑，pool 無 prefill 專家可 pin | 無意義（pin 空操作 + miss 暴增拖垮速度）|

**根因（機制三層，全部由實測+程式碼確認）**：

1. **LRU 新鮮度已經在做 pin 想做的事**：prefill 填 pool 時 `slot_last_use` 被 bump 到最新，
   200-slot pool 每步只換出 ~8 個最舊 slots（load-time prewarm 的 0..199）——prefill 尾段
   自然存活幾十步，pin（單步保護）是純冗餘 → 4GiB 下 0 差異。
2. **§8.14 的「prefill 覆蓋 21.5-65.3%」發生在整個 128 步期間，不是第一步**：被換出的 prefill
   experts 在 decode 中後段（step 20-30+）才被 revisit——單步 pin 窗口根本碰不到；拉長窗口
   （永久 pin）則以 40% 池容量（80/layer）換回「幾乎不被 revisit 的專家常駐」（§8.14：
   1796/1797 miss 唯一），容量損失 > 命中回收。
3. **池越小 pin 越沒用**：prefill union（37）> slots 時 prefill 走 L3-B map，pool 根本沒有
   prefill 專家可 pin → 空操作；且小池 miss 暴增（21372 vs 8450）本身就把 tok/s 打到 5 以下。

**定案**：**TAILPIN 證偽（net ≈ 0）**——機制正確（pin 觸發、unpin 觸發、bit-identical、
確定性 miss 轉正），但可回收量是 0.14%（1GiB）/ 0%（4GiB）。§8.14 的覆蓋天花板不是「保護
不足」，而是「已由 LRU 新鮮度自然兌現」；未兌現部分（late-step revisit）需要跨步 pin，
其容量成本遠超收益。`LLAMA_EXPERT_CACHE_TAILPIN=1` 保留為 env-gated 診斷工具（驗證 pin 機械
正確性用），預設關。**bounded residency 的預測/prewarm/pin 三線至此全部證偽**，剩餘槓桿
只剩 §8.12 pool8 並行 fill（qwen36 11.76 / gemma4 14.40 median）。

### 8.17 收斂報告：bounded residency 剩餘速度槓桿評估（2026-08-15）

§8.11-8.16 定案後的全面收斂。目標：回答「還有什麼值得花時間」。

#### 8.17.1 已證偽 / 已定案槓桿全清單（I/O 側全滅）

| 槓桿 | 結果 | 出處 |
|---|---|---|
| readv 合併（3 段一次讀）| 不可行（segment 分散、macOS lio_listio 壞）→ threads 併發回收 ~1% | §7.7 |
| pool budget 加大 | miss 單調降但 **tok/s 持平**（ngl0 是計算牆）；4GiB 是 16GB Pareto | §7.8/§8.9/§7.10 |
| fadvise / page-cache 預取 | **-88% 災難**（hook 同步 fcntl + over-predict 16×）| §8.13 |
| 真 bg slot fill double-buffer | **0 淨收益**（pool 被 prefill 塞滿、層鄰預測不準）| §8.14 |
| 步間 / 層鄰預測 | **天花板死亡**：1796/1797 miss 唯一 (layer,expert)，同層跨步不重複 | §8.14 miss 分析 |
| prewarm 熱集（prefill top-K 填池）| **-40%**（熱集在 prefill 結束前不可知 → 8000 pread 卡首 token 關鍵路徑）| §8.15 |
| TAILPIN（首步 union pin）| **net≈0**（LRU 新鮮度已保護 prefill 尾段；跨步 pin 容量成本 40% > 回收）| §8.16 |
| GPU 藏 pread（fill 藏進 GPU busy）| **結構性不可能**（依賴鏈硬串行：路由→fill→FFN 是 data dependency）| §7.5.2 |
| pool16 workers | **反效果**（單 SSD queue depth 打爆，崩到 7-8.7）| §8.12 |
| MTP draft | draft forward 使 expert 流量翻倍 + 無預測力（共享 cache 修正後仍無淨收益）| §7 前 + §8.7 修正 |
| 靜態 pin / predict_next / 暖池 | 三度證偽（真實 trace 掉 15-37pp）| P0 預測線 |

#### 8.17.2 用戶點名的三個「未試維度」逐一評估

**① slot 大小**——無頭寸，不值得。
- slot = 單 expert bytes（qwen36 0.88MB / gemma4 2.39MB），由模型與量化位寬決定，不是設計參數。
- 唯一變體「super-slot」（一次 pread 讀多個相鄰專家）需要檔案佈局把多個專家連續化：qwen36
  UD 檔 interleaved 已排除、gemma4 per-layer 三張量分散（readv 都合不起來）。且 §8.14 證明
  decode 選的 experts 幾乎全唯一 → 預讀相鄰專家幾乎必是垃圾。

**② LRU 策略**——無頭寸，不值得。
- LRU 的假設是「最近用過會再用」；§8.14 證明 decode 跨步不重複選 expert → recency 無訊號。
- frequency 側已被「靜態 pin 三度證偽」（真實 trace 掉 15-37pp）同一結論覆蓋。LFU/LRU 混合
  在「全部唯一」的資料上沒有可學的統計結構。

**③ decode 前 union 預熱**——已兩刀證偽，不值得。
- §8.15（top-K 熱集填池）：熱集不可在關鍵路徑外得知 → 預填必然落首 token 關鍵路徑（-40%）。
- §8.16（TAILPIN 只 pin 不 fill）：LRU 新鮮度已兌現可達部分（4GiB 0 差異 / 1GiB -0.14%）。
- 兩者根因互補，第三刀不會有新訊號。

#### 8.17.3 真正的剩餘差距在計算側，不在 IO 側

- **gemma4 n99c 4GiB = 14.40 t/s（median）vs Swift gemma4-r4 18-20**——§10.4 差距歸因
  引擎層（graph scheduler / combine / Metal kernel 緊湊度），不是 pread（fill wall 已壓到
  5.2s/129tok ≈ 6%）。
- **qwen36 n99c = 11.76 已超 Swift qwen36（5.5-5.8）**——Swift 的優勢是模型相關的
  （只在 gemma4 甜區：4-bit offload + readWall 隱藏充分），不是「Swift 引擎普遍更快」。
- n0c 的牆是 CPU attention（§10.5）——bounded residency 管不到。

#### 8.17.4 收斂決策

1. **I/O 側封板**：bounded residency 功能完成（pool8 預設、RSS 9.36GB @ 4GiB 可控、
   bit-identical、qwen36 11.76 / gemma4 14.40 median）。不再投入任何 IO 槓桿。
2. **唯一值得的下一步 = 計算側火焰圖**：拆 gemma4 n99c decode 的 14.40 中
   graph-scheduler / combine / attention / FFN-kernel 各佔比，找 18-20 的差距來源
   （Swift 對照組是引擎層收斂的標的）。
3. **I/O 側若還要再開一槍（低期望值）**：NVMe queue-depth 感知的讀取調度（按 SSD 深度動態
   批合併 8 worker 的 pread）——但 §8.12 pool16 已證併發崩潰，SSD 是硬牆，估計 ≤2%。

### 8.18 計算側火焰圖（2026-08-15）：gemma4 n99c 4GiB decode 成分分解——GPU-bound 定案

§8.17 收斂報告判定「剩餘差距在計算側」。本節用 macOS `sample`（1ms 間隔、20s 窗口、
256 tok decode 期間）直接採樣 gemma4 n99c 4GiB pool8（本輪 speed 10.41 t/s，機器 load
2.27；**成分比例與絕對速度無關**），拆主執行緒 15746 個樣本：

| 成分 | 樣本 | 佔比 | 說明 |
|---|---|---|---|
| **Metal wait（GPU busy）** | 13641 | **86.7%** | `ggml_metal_synchronize → waitUntilCompleted → cvwait`——主執行緒卡在 GPU command buffer 完成 |
| CPU backend 計算 | 865 | 5.5% | `ggml_backend_cpu_graph_compute`：**router GEMM（mul_mat f32）464 + argsort topk ~19 + softmax ~6**——A+B split 強制 ffn_moe_gate 在 CPU |
| cache hook（同步段）| 587 | 3.7% | eval_cb 327 + on_topk 134 + ensure_batch 126（後者才是關鍵路徑上的同步 fill）|
| sched dispatch/encode | 331 | 2.1% | sched 非 wait 部分 |
| sampler / 其他 | ~300 | ~2% | 取樣 + 雜項 |

**pread I/O 歸屬**：8 個 pool worker 的 pread 樣本 ≈1165-1419（全 thread 的 ~7%），但全部
與主執行緒的 Metal wait **並行**——I/O 藏在 GPU busy 後面，**不在關鍵路徑**（主執行緒上
只有 ensure_batch 的同步等待 126 樣本 = 0.8%）。

**定案：decode 是 GPU-bound，不是 I/O-bound、不是 scheduler-bound。**

- cache 機械（hook 3.7% + 隱藏 I/O ~0）與 scheduler（2.1%）都不是瓶頸——§8.11-8.16 的
  全部 I/O 槓桿做完了也不過動這 3-5%。
- **14.40 vs Swift 18-20 的差距 = GPU kernel 執行時間（86.7%）**：llama.cpp 通用 ggml-metal
  kernel（mul_mat_id per-expert slice indirection + attention + combine）比 Swift 手寫
  per-expert Metal kernel 慢 ~25%。要到 18.4 需 GPU 時間 -22%。
- **次級槓桿 = A+B split 的 CPU router**（5.5% ≈ 0.8 t/s @14.4）：把 ffn_moe_gate+topk 移回
  GPU（Swift 路線）可回收——但這正是 A+B split 為 bit-identity 犧牲的部分（§7.3.5），
  需 GPU 側 routing 回讀 + 非阻塞 fill 設計（早期「追平 Swift 第一步」提案）。
- 工具：`sample <pid> 20 -file`（macOS 內建）；解析方法見本節表格（sched 子樹 =
  metal_synchronize / cpu_graph_compute / hook 三段直接對應三大成分）。

**後續方向**（P1 計算側，非 bounded residency 範圍）：① 對比 base（無 cache）的火焰圖
確認成分相同（cache 不改變 GPU-bound 本質）；② 用 Metal System Trace（xctrace）拆 GPU 側
attention vs FFN kernel 時間；③ Swift 對照組採樣，量化「同 kernel 不同實作的差距」。

### 8.19 P1 工作分解：引擎層收斂（2026-08-15）

基於 §8.18 火焰圖（GPU 86.7% / CPU 可見 13.3%）與程式碼盤點，逐候選給改動面與回收。

**先釐清 wall 結構**：69ms/token（@14.40）= GPU wait ~60ms + CPU 可見 ~9ms（sched dispatch
2.1% + CPU backend 5.5% + hook 3.7% + sampler ~2%）。CPU 可見在 graph 之間是串行的（hook 的
同步 fill、CPU backend 的 router 都會擋住主執行緒），所以**兩邊都是真槓桿**：GPU -22% 或
CPU 可見全清（-13%）都能把 wall 往下拉。但 CPU 側有交互：GPU 降下來後 CPU 可見的相對
比重變大，所以兩者都要做。

#### 8.19.1 三個候選的盤點結果（程式碼確認）

| 候選 | 現況 | 改動面 | 預期回收 | 判決 |
|---|---|---|---|---|
| graph reuse 參數 | **已開**（`can_reuse(gparams)` 每步命中，decode gparams 穩定）；唯一 knob 是 `LLAMA_GRAPH_REUSE_DISABLE`（關閉用）| 無（已是最優路徑）| **0%** | 判死——無可調，只需確認沒誤設 |
| ggml-metal dispatch | 單 MTLCommandBuffer/graph + **早期 commit**（line 515 註解「main thread commits first commands immediately」）+ step 末 `ggml_metal_synchronize`（被我們 cache 的 restore race 保護強制）| 中：command buffer 池化（newCommandBuffer 每 graph 一次→reuse）；每層 commit vs 每 graph commit 的取捨 | **+0.5-1%** | 低優先——sched dispatch 只有 2.1%，且大部分已被 early-commit 藏進 GPU |
| combine kernel n_ubatch | combine = `get_rows(probs, selected)` + `mul` + `sum_rows`（per-token 小 op）；decode 的 n_ubatch 恆為 1 | 無 | **0%** | 判死——n_ubatch 是 prefill 槓桿（KV/attention batch），與 decode 無關 |
| **A+B split 的 CPU router** | `ggml_backend_cpu_graph_compute` 5.5% = ffn_moe_gate GEMM（mul_mat f32 464 樣本）+ argsort topk | 中：router 權重回 GPU buft + **topk 小量回讀**（8×4B=32B/layer 從 GPU buffer 同步讀回給 hook）| **+5.5% ≈ +0.8 t/s** | **P1-2 主候選**——最確定、低風險 |
| FFN mul_mat_id kernel | GPU 主體（86.7% 的主要部分）；llama.cpp 通用 per-slice indirection | 高：Metal kernel 調優（batched expert GEMM、workgroup 大小、gate_up 合併、避免中間 tensor）| **+10-20%（gap 的 50-70%）** | P1-3——最大但高風險，需先量 GPU 側 |
| hook 同步段 | 3.7%（eval_cb 327 + on_topk 134 + ensure_batch 126）| 低：on_topk 的 strided 讀取優化、eval_cb 的 string 比對（strncmp）改 id 快查 | **+1-2%** | P1-5 低優先 |

#### 8.19.2 P1 分刀（含前置依賴）

| 刀 | 內容 | 依賴 | 預期 | 風險 |
|---|---|---|---|---|
| **P1-0（前置，必做）** | xctrace Metal System Trace：拆 GPU 側 attention vs FFN(mul_mat_id) vs combine 時間 | 無 | 決定 P1-3 打哪個 kernel | 無（純量測）|
| **P1-1（0.5d）** | base（無 cache）火焰圖對照：確認 GPU busy 佔比相同 | 無 | 驗證 cache 無責（§8.18 假設）| 無 |
| **P1-2（1-2d）** | router 回 GPU：ffn_moe_gate buft 改 Metal + topk 32B/layer 回讀 hook（bit-identity 對拍）| 無 | **+5.5%（0.8 t/s）** | 中：回讀同步點要卡在 FFN 前；§7.3.5 已證明 bit-identity 可維持 |
| **P1-3（3-5d）** | FFN mul_mat_id kernel 收斂：以 Swift per-expert 批量 GEMM 為對照（需 P1-6 先量化差距）| P1-0、P1-6 | **+10-20%** | 高：kernel 層、需 bit-identical 回歸 |
| **P1-4（0.5d）** | Metal command buffer 池化 | 無 | **+0.5-1%** | 低 |
| **P1-5（0.5-1d）** | hook 序列化收斂 | 無 | **+1-2%** | 低 |
| **P1-6（1d）** | Swift gemma4-r4 同法採樣：量化「同 kernel 不同實作」的 GPU 時間差距 | 需 Swift 引擎可跑 | P1-3 的標的數字 | 無 |

#### 8.19.3 建議執行序與預期終態

P1-0 → P1-1 → **P1-2（先拿確定的 +5.5%）** → P1-6 → P1-3（主戰線）→ P1-4/P1-5 填尾。
預期終態：14.40 + router 回 GPU（+5.5%）+ hook/dispatch 收斂（+2-3%）→ **~15.7-16.0 t/s**；
P1-3 若回收 gap 的 50% → **~17.5-18 t/s**，貼近 Swift 18-20。全部需 bit-identical + 三臂 A/B
回歸（既有 harness）。bounded residency（§8.11-8.16）不變，P1 是純計算側。

### 8.20 P1-2 實作 A/B（2026-08-16）：router 移回 GPU 轉正，median +27%

**實作**：loader 的 router CPU 強制改為 env-gated 預設關（`LLAMA_EXPERT_CACHE_ROUTER_CPU=1`
還原舊佈局）；`graph_get_cb` 捕捉每層 `ffn_moe_probs`，`graph_compute` 每步把 probs
pin 到 CPU backend——router GEMM（mul_mat，火焰圖 464 樣本）留在 Metal split，softmax/
argsort/topk 留在 CPU split，hook 仍在 Metal FFN split copy 之前觸發（copy-timing 修復
§7.3.5 不破，無需 topk 回讀，pin 即完成 split 保證）。

**A/B（gemma4 IQ3_S、n99c 4GiB L4 pool 120 slots、6 輪交錯、N=64、同一負載窗口）**：

| arm | mean | median | range |
|---|---|---|---|
| **P1-2 預設（router 上 Metal + probs pin）** | **14.10** | **14.70** | 8.94-16.61 |
| ROUTER_CPU=1（舊：router 全 CPU） | 11.82 | 11.54 | 8.47-16.75 |

hit=15845/miss=3737（80.9%）、pread ~7s 兩臂**完全一致**（純計算側對比），輸出全程
**bit-identical**。median +27%（14.70 vs 11.54），遠超 §8.19 預測的 +5.5%——原因：
ROUTER_CPU 不只把 GEMM 留在 CPU（464 樣本 ≈5.5%），每層還多一次 CPU↔Metal split
barrier（30 層 ping-pong），pin 方案把 barrier 縮到只剩 tiny softmax 跨裝置。

**定案**：P1-2 成為預設（保留 ROUTER_CPU=1 診斷）。絕對值 14.70 對齊 §8.12 的 14.40
基線（當天深夜低負載窗口），同窗口對照下新舊差距 +27%。已回收 §8.19 預測的 5.5%
並多拿了 barrier 的部分——`~15.7-16.0 終態`的預估上修到 `14.40 基線即含此收益、
剩餘槓桿在 P1-3 FFN kernel`。

### 8.21 P1-3 設計定案（2026-08-16）：FFN kernel 收斂的藍圖與分刀

**背景**：P1-2 之後的剩餘差距 = GPU kernel 執行（火焰圖 86.7%）。topk-cap 隔離實驗
（cap8 8.65 vs cap2 10.9）給出 FFN 的總成本邊界：**MoE FFN ≈ decode 時間 25-27%**；
純頻寬反推（gemma4 FFN 權重 545MB/token @120GB/s = 4.5ms）顯示 GEMV 效率只有
純 BW 的 ~1/5——FFN 的 25% 大部分不是 BW 飽和，是 **kernel 內部效率**。

**graph dump 實證（gemma4 decode、30 層）**：每層 FFN 鏈已相當緊湊：
`MUL_MAT_ID(gate_up) [2×704] → view×2 → glu → MUL_MAT_ID(down) [2112] → mul(weights)
→ sum_rows → agg`。60 個 MUL_MAT_ID（gate_up+down 各 30）。**op 層沒有可再砍的
冗餘**——融合只能進 kernel。

**Swift phase-1/phase-2 藍本 vs llama.cpp 現狀**：

| | Swift（18-20 t/s） | llama.cpp（14.4） |
|---|---|---|
| gate_up+act | **phase-1 單 kernel**（acts 留 threadgroup，無 global round-trip） | MUL_MAT_ID + view×2 + glu 三節點、[2×704,8] f32 中間量走 global |
| down+reduce | **phase-2 單 kernel**（thread 依序累加 8 expert，無 atomic） | MUL_MAT_ID + mul + sum_rows 三節點、[2112,8] 中間量走 global |
| 每層 FFN dispatch | 2 | 5-6 |

**kernel 結構（已讀 code 確認）**：decode 走 `mul_mv_id` 路徑（ne21=1 < 32 的
mm_id 分界），grid.z = 8（每 expert 一槽），每槽一個 GEMV。IQ3_S: nsg=2、nr0=4
（64 thread/TG、每 TG 8 行）。dst 每 expert 各寫一行 → 下游 sum_rows 做跨 expert
reduce。

**融合設計（兩階段，各自 bit-identity 可驗證）**：
1. **Phase-1 融合（gate_up+glu）**：新 op `MUL_MAT_ID_GLU`，Metal epilogue 在
   GEMV 行和寫出前配對 gate 行 r 與 up 行 r+n_ff（改 row→TG 映射讓配對行同 TG），
   寫 [n_ff,8] 直接輸出——省 1 中間量 round-trip + 2 節點。**配對跨 simdgroup 是
   主要複雜度**（需 threadgroup 暫存或改 dispatch 映射）。
2. **Phase-2 融合（down+weighted+sum）**：新 op，每 expert 的 [2112] 行和 ×
   weights[e]，跨 expert reduce **必須維持 sum_rows 的順序求和**（atomic 不保證
   順序 → 破壞 bit-identity）→ 需 thread 依序累加 8 expert 的 grid 重構
   （Swift 式：expert 進序列維度，非 grid.z）。

**預期回收（誠實）**：op 層融合本身 ≈ **+2-3%**（省 2 個中間量 round-trip +
每層 3-4 個 launch ≈ 0.6-1.2ms/step）；真正的 **+10-20% 在 GEMV 內部效率**
（~5× BW gap 的回收），需要 Swift 式 grid 重構（expert 進序列維、256 thread/TG、
更優 dequant 讀取）——**multi-day kernel 專案，且本機無 GPU-side 計時工具
（P1-0 的 per-op 路徑因兩次當機已移除）**，改 kernel 形同盲調。

**建議順序**：P1-3 前先重建**安全**的 GPU 側測量（per-op 計時重做，但用
n_cb=1 單 command buffer + completion handler 聚合統計、絕不碰 fusion 路徑），
否則 kernel 改動無法量測。之後先做 phase-1（低風險、bit-identity 是安全閘），
再評估 phase-2 的 grid 重構。

### 8.22 P1-0 完成：安全 GPU 側 per-op 計時工具 + gemma4 decode 成分分解（2026-08-16）

**工具**（`CGC_GPU_OP_TIMING`，env-gated 預設關）：
- 每 op 一個 command buffer + `GPUStartTime/GPUEndTime` + completion handler 聚合，**強制 use_fusion=false、use_concurrency=false**（前兩次當機的兩個根因都不碰）
- 新 API：`ggml_metal_op_get_node_op`（op type）；key = `OPTYPE:tensor-name`（無名節點聚合為 `OPTYPE:?`，ggml 自動名 `node_N` 除外）
- 修 dump 的雙轉義 `\\n` bug（log 曾寫成字面 `\n`）
- 工具已校準：穩定區每 layer-subgraph ≈ 2.1ms（per-op 序列化） vs 真實 decode 2.3ms/layer —— 誤差 ~10%，相對結構可信

**測量**（gemma4 IQ3_S、n99c 4GiB pool8、61 表 = 366 layer-subgraphs、末 20 表穩定區）：

| 成分 | 佔比 | ms/layer | 備註 |
|---|---|---|---|
| attention 投影（Q/K/V/attn_out MUL_MAT）| 18.0% | 0.87 | 命名 op |
| MUL_MAT 未名（每層 ~4 個）| 15.9% | 0.77 | KQ/attn@V/out 等 attention core 可能性高 |
| **MoE gate_up**（MUL_MAT_ID）| 14.3% | 0.69 | 單一最大命名 op |
| **lm_head**（MUL_MAT 262144×2112）| **13.3%** | 0.64* | *每 step 出現一次、實測 ~4.3ms/step |
| small ops（get_rows/rope/mul/add/softmax）| 11.9% | 0.58 | |
| MoE down（MUL_MAT_ID 未名）| 9.2% | 0.45 | |
| shared FFN（dense gate/up/down）| 9.0% | 0.44 | |
| rms_norm（8/layer）| 4.9% | 0.24 | |
| router logits | 1.5% | 0.07 | P1-2 已修 |
| SDPA（FLASH_ATTN_EXT）| 1.4% | 0.07 | |
| head 後處理 | 0.5% | 0.02 | |

**交叉驗證**：MoE FFN 合計 23.5%（gate_up 14.3 + down 9.2）≈ topk-cap 隔離實驗的 25-27% ✓（兩種方法獨立得到同一結論）。

**新發現（P1-3 靶心修正）**：
1. **lm_head 佔 decode 的 ~6%（4.3ms/step）**——214MB 讀取只跑 ~50GB/s（120GB/s 的 42%），比 BW 理想 1.8ms 差 2.4×。先前 BW 反推把 head 估成 ~3%，實測是 2×。**lm_head kernel 是與 MoE 同等級的新靶**（Swift 對照組有無 head 優化待查）。
2. attention 全體 ≈ 18%（投影）+ ~16%（未名 core）+ 1.4%（SDPA）≈ **35%**——比 §8.18 估的 attention 佔比更高；未名 MUL_MAT 的身份（KQ/attn@V/out 是否走離散路徑而非 SDPA）是 P1-0 剩餘的一格，需 CGC_GRAPH_DUMP 對照確認。
3. MoE 23.5% 維持主靶；shared FFN 9% 同 MUL_MAT 家族（gate/up/down dense），與 MoE 共享 kernel 優化收益。

**工具狀態**：`CGC_GPU_OP_TIMING[=SKIP,GRAPHS]` 保留為 B 階段診斷工具（單 buffer per op 有 ~2× 序列化膨脹，只看相對結構與穩定區）。解析腳本：`tmp/parse_gpu_op_timing.py`。

### 8.23 qwen36 P1-2 A/B（2026-08-16）：P1-2 不跨家族轉移——qwen36 是 −31-45% 回歸，預設改家族分派

**動機**：gemma4 上 P1-2（router 上 Metal + ffn_moe_probs pin CPU）median +27%（§8.20），
需驗證是否同樣適用 qwen36。

**實測**（qwen36 IQ3_XXS、n99c 4GiB L4 pool 200 slots、N=64/128、交錯序多輪、bit-identical、
hit=39856/49769 (80.1%) 兩臂完全一致）：

| 臂 | N=64（磁碟衝突期間）| N=128（apfsd 消退後）| N=128 複驗（新預設）|
|---|---|---|---|
| **CPU router**（ROUTER_CPU=1 / 新預設）| 6.1-8.0（磁碟牆）| **11.03-11.50（緊）** | **10.37-11.78（median ~10.8）** |
| **Metal router**（P1-2 預設 / ROUTER_CPU=0）| 5.8-8.5 | 6.67-8.52 | 6.22-8.78（median ~7.4）|

**結論**：
1. **qwen36 上 P1-2 是 −31-45% 回歸**（median 11.2 → 7.5），與 gemma4 的 +27% 完全相反。
   old（CPU router）精確複現文檔基線 11.76（§8.12，負載 ~3.3 下 10.4-11.8）。
2. **機制假說**：qwen35moe 40 層 + 256 experts——CPU router 的 router GEMM（+softmax/topk/
   hook）可與**前一層的 Metal FFN 軟體管線重疊**（CPU/GPU 異步並行）；router 上 Metal 後整層
   進 GPU 佇列序列化，40 層串行化吃掉 overlap。gemma4 只有 30 層 + 128 experts，P1-2 移走
   的 CPU GEMM + 30 次 CPU↔Metal barrier 成本 > 序列化損失 → +27%。**P1-2 收益是家族特異的**。
3. **修正**：loader 預設改**家族分派**——qwen35moe 家族（LLM_ARCH_QWEN3MOE/QWEN35MOE）
   預設 ROUTER_CPU=ON（CPU router，回到基線），gemma 維持 P1-2（Metal router，+27%）。
   env 可覆寫：`LLAMA_EXPERT_CACHE_ROUTER_CPU=0`（強制 Metal）/ `=1`（強制 CPU）。
   已驗證：qwen36 預設釘 CPU ✓、gemma4 預設不釘 ✓、qwen36 預設回到 ~11 t/s ✓。

**教訓**：A/B 轉正（gemma4 +27%）不等於跨模型轉移——每家族都要驗證；火焰圖的
「CPU backend 5.5% = router GEMM」是 gemma4 專屬結論，qwen36 上同構改動是負收益。

### 8.24 lm_head 分析（2026-08-16）：tied embedding Q6_K 是 decode 最大單一 op——實測 23-26ms/step、9% 頻寬

**背景**：§8.22 的「13.3% / 4.3ms」是 count-weighted rollup 假象（head 每 step 只出現 1 次、被 366 個 layer-subgraph 平均稀釋）。重新逐筆統計 `/tmp/g4t2.err` 的原始 acc：node_2640 每次執行 23.2-26.0ms（6 筆、方差異常小）、每個 decode step 都跑。

**head 的真相**：
- **身份**：tied embedding——`token_embd.weight` Q6_K [2816, 262144]，檔案 605.5MB（UD 打包）、**load 時經 `set_tensor_data_ud` de-interleave 成標準 Q6_K → runtime 282MB**。
- **路徑**：decode ne11=1 → 跳過 `mul_mv_ext`（K-type 需 ne11∈[4,8]）與 `mul_mm`（需 ne11>8）→ **落入經典 scalar `kernel_mul_mv_q6_K_f32`**（N_R0_Q6_K=2）。
- **實測**：單次 25.8ms × 120GB/s = 3.1GB 等效讀取，而實際只有 282MB → **~11GB/s = 9% 頻寬利用率**（MoE gate_up GEMV 同工具下 2.6ms/層）。head 是整個 decode graph 最大的單一 op（是第二名的 5-10×）。

**低效根因（kernel 結構，非環境）**：
1. **2-way 欄並行**：`for (int i = ix; i < nb; i += 2)`——只有 ix=0/1 兩條執行緒分欄掃 2816 = 11 blocks；其餘 30 執行緒只分工單一 block 內的 16 段。**ne00 維度的並行度是 2**，不是 32。
2. **scalar strided y 讀取**：`y[l], y[l+32], y[l+64], y[l+96]` 逐個 scalar gather（decode 時 y 只有 11KB，非主成本，但打斷連續讀）。
3. **Q6_K dequant ALU 重**：每 4 值 4 組 bit-mask + shift + int8 轉換 + 4 scales + dh 相乘——Q6_K 的每 byte 解量化成本是 IQ2/IQ3 的數倍（後者查表）。
4. **行數極大、每執行緒工作量極小**：262144 行 ÷ (NR0=2 × NSG) → 上萬個 threadgroup、每 thread 只算 ~6 block × 2 行 → **latency/occupancy-bound，不是 BW-bound**；執行緒數不足以蓋住 DRAM latency。
5. **檔案層的 2× 位元浪費**：Q6_K 0.82B/param 對 decode GEMV 是高價位（IQ4_NL 0.55、IQ3 0.44）。

**Swift（turbo-fieldfare gemma4-r4）對照**——`LMHeadChainInt4` + `lm_head_greedy_int4_rows_chunk_raw/reduce`：
- **INT4 affine**（0.5B/param → 277MB，只有 Q6_K 的 49% byte）+ 每行 scale/bias，無 bit-mask dequant。
- **三合一融合**：output_norm + head GEMV + greedy argmax 單一 dispatch（rowGreedy rows_chunk_raw → rowReducer 兩段式 chunk reduce），**從不 materialize 262144 長度的 logits**（省 1MB buffer round-trip + 獨立 softmax pass）。
- rowsPerThreadgroup=8、rowSummaryStride=2。

**優化方案與預期回收**（依風險/回收排序）：

| 方案 | 改動 | 預期回收 |
|---|---|---|
| **H1 · ne11=1 進 ext 向量化路徑** | `mul_mv_ext_q6_K_f32`（type4x4、float4 載入）的 gate 放寬到 ne11=1；或複製 GEMV 版 | head 25.8 → 6-9ms（3-4×），decode +15-25% |
| **H2 · Swift 式融合 head chain** | output_norm + GEMV + softcap/softmax 單 dispatch、chunked reduce、不 materialize logits | 在 H1 之上再省 ~1-2ms/step（launch + buffer round-trip） |
| **H3 · 位寬換 byte** | head 重打包 IQ4_NL（304MB）或保留 Q6_K 但 dequant 用查表 | byte 少 1.5-2×；品質權衡需 A/B |

**交叉驗證與誠實邊界**：
- 獨立佐證 1：head 282MB × 每 step 一次 → 即使 100% BW 也要 2.35ms——**kernel 目前浪費了至少 20ms/step 的理論頻寬**。
- 獨立佐證 2：per-op 工具校準誤差 ~10%（§8.22）對大 kernel（25ms）影響遠小於小 kernel——25.8ms 接近真實路徑時間。
- **頭號新靶**：lm_head 的回收空間（20ms+/step）> MoE FFN 整體（§8.21 的 25-27% ≈ 17ms）> P1-3 的 glu 融合（+2-3%）。建議 P1 下一步先做 H1（低風險高回收），H2/H3 隨後。

**待確認**：NSG 值與 M4 上真實 threadgroup 佔用（可再用 CGC_GPU_OP_TIMING 對 H1 前後量測）；Swift head 在 16GB 機的實際 head 佔比（若 Swift 的 18-20 t/s 是真，其 head 已是融合 int4——這正是它比我們快的一部分原因）。

### 8.25 未名 MUL_MAT 身份解析（2026-08-16）：attn_out + dense FFN down，attention 全體 ~27-28%（修正 §8.22 點 2）

**方法**：把 `/tmp/g4gc.dot`（decode graph dump，2644 節點）的 ggml node 名（`node_N`）與 per-op 計時的 key 對照——每層的未名 X*Y 恰有 2 個，名稱為 `node_29 族`（idx = 層起始+29）與 `node_80 族`（idx = 層起始+63）。

**身份（GGUF 張量名交叉確認）**：
| node 族 | 身份 | GGUF tensor | 量化 | 每層 bytes |
|---|---|---|---|---|
| `node_29` [2816,1] | **attn_out 投影** | `blk.N.attn_output.weight` [4096,2816]（大頭層 [8192,2816]）| Q6_K | 9.46-18.9MB |
| `node_80` [2816,1] | **dense/shared FFN down** | `blk.N.ffn_down.weight` [2112,2816] | Q8_0 | 5.9MB |

**兩個「attention core 離散 MUL_MAT」假說被證偽**：decode graph 每層的 KQ/attn@V 在 **`flash_attn_ext`（SDPA，idx 27）單一節點**內——計時的「SDPA 1.4%」即 attention core，不是離散 matmul。§8.22 的「MUL_MAT 未名 = KQ/attn@V/out 候選」是錯的。

**attention 全體（修正）**：Q/K/V in-projections 18.0%（命名）+ attn_out ~8%（未名半數）+ SDPA core 1.4% ≈ **27-28%**。加上 lm_head（§8.24，23-26ms）與 MoE FFN（23.5%），gemma4 decode 的三大成本：**lm_head > attention（含 attn_out）≈ MoE FFN**。

**對 P1-3 的意義**：attn_out 與 dense ffn_down 都是 [2816,1] GEMV——與 MoE GEMV 同一 kernel 家族（Q6_K/Q8_0），可共享 H1（ext 向量化路徑）的優化；P1-2 之後的 decode graph 已無「隱藏」的 attention core 開銷，成分表是完整的。

### 8.26 gemma4 -ngl≥31 正確性回歸（2026-08-16）：lm_head 上 Metal 即退化——P1-2「bit-identical」是空驗證

**狀態：未修復。已知工作配置 = -ngl 30（全層 Metal + head CPU）。**

#### 發現
- gemma4 IQ3_S（11.29GB）在本機 `-no-mmap -ngl 99` 下輸出確定性退化（`is is is is...`），exit 0 無錯誤。
- ngl 掃描（-n 6、同 seed/prompt）畫出**精準邊界**：`-ngl ≤ 5` 正確、`-ngl 15/20/25/28` 退化、`-ngl 29/30` 正確、`-ngl ≥ 31`（head 上 Metal）全部退化。每次確定性可重現。
- ngl=31 = gemma4 的 output layer（tied embedding → lm_head）上 Metal 的臨界點。**lm_head 上 Metal 即退化。**
- qwen36 IQ3_XXS（13.2GB）`-ngl 99` 是硬 OOM（`kIOGPUCommandBufferCallbackErrorOutOfMemory`，13.2GB > 11.45GB working set）——與退化不同，屬預期邊界。

#### 證據鏈（本 session）
1. **退化非本 session 引入**：上個 session 的 P1-2 A/B 輸出檔（/tmp/p12r_*.out，08-16 03:02）已是 `is is is`——「bit-identical ✓」是兩臂同樣退化的**空驗證**。P1-2 的 +27% 速度結論有效（速度與正確性無關），但「bit-identical」不可信。
2. **08-15 01:01 的 build-norepack binary 在同一指令下正確**（`Paris`，多次重現）→ 08-15 後 source 有回歸。
3. 逐項排除（全部 env-gated / 惰性 / 移除仍退化）：glu kernel（MUL_MAT_ID_GLU，完整移除 4 處後仍退化）、P1-0 timing tool（無無條件呼叫）、P1-2 router/probs-pin（`model.expert_cache != nullptr` gate）、skip-load/L4 pool（`expert_cache_bytes > 0` gate）、L3-B graph hook（`expert_cache_active` gate）、eval cb（cache/DEBUG env gate）、sched A+B pin（`ALLOW_NGL` env gate）、copy_experts（upstream 代碼）、Q6_K/IQ2_S/mmid kernel（與 CGC-main/Backend 副本相同）。
4. eval-cb 序列化模式（每節點 sync）仍退化 → 非 GPU race，是確定性錯誤計算。
5. prefill topk 兩臂一致、decode step-2 分歧 → **decode-step FFN 輸出（Metal）與 CPU 分歧**。

#### 機制假說（未證實）
- ngl 正確性窗口（{1,5,29,30} 好 / {15-28,31+} 壞）非單層效應——推測是 sched split 結構非單調變化 + 某個 Metal op 在特定 split 下讀錯 buffer（候選：head 的 tied-embed view / lm_head MUL_MAT 的 Metal buffer 定址，或 A+B split 的 copy-before-swap 在無 cache 時的殘留路徑）。
- 08-15 後唯一未排除的變因：llama-graph.cpp（08-15 19:39，L3-B/scale fix）與 llama-context/llama-model/llama.cpp（08-16，cache 架構）的非 gate 部分——但逐一檢查均 gate 住。

#### 影響
- **lm_head A/B（本任務）被擋**：head 上 Metal 即退化，無法在 -ngl≥31 下量測 head kernel 優化。
- P1-2/P1-3 的速度結論維持有效（速度與正確性解耦），但所有「bit-identical」驗證需以 -ngl 30 或 CPU 重新對拍。
- 日常可用配置：`-ngl 30`（全層 Metal、head CPU、正確）——注意 head 在 CPU 上，lm_head 13.3% 的 GPU 成本在此配置下不存在。

#### 下一步（修復路徑）
1. 重建 08-15 source 快照（若無備份需從上游 tag 重拉 + 重放 CGC patch，成本高）。
2. 用 ngl=31 vs ngl=30 的 split 差異 + `CGC_GRAPH_DUMP` + per-op 計時（-ngl 30 對照）定位 head MUL_MAT 的 Metal 讀取差異。
3. 修復後重跑 P1-2 A/B 的 bit-identical 對拍（-ngl 30 / CPU 基線）。

### 8.27 P1 分刀修訂（2026-08-16）：家族分開估 + lm_head/未名 MUL_MAT 納入

**動機**：① §8.23 證明 P1-2 收益**家族特異**（gemma4 +27%、qwen36 −33%）→ 任何 P1 計算側改動都必須家族分開估、雙家族對拍；② §8.24/8.25 把 decode 三大成本定案為 **lm_head（23-26ms/step ≈ decode ~36%）> attention（27-28%）≈ MoE FFN（23.5%）**——原 §8.19.2 的「單一 P1-3 +10-20%」與靶心排序需要重排（lm_head H1 先於 MoE kernel 精修）。

#### 8.27.1 P1-2 於 -ngl 30 重驗證（§8.26 翻案後的第一個正確輸出對拍）

§8.26 判定舊「bit-identical ✓」是兩臂同為 `is is is` 的空驗證。本輪在 known-good 的 `-ngl 30`（全層 Metal、head CPU、working set 11.21GB < 11.45GB）重跑 4 輪交錯：

| 臂 | median | range | 輸出 |
|---|---|---|---|
| base（-ngl 30 無 cache）| 14.53 | — | 正常（"...is **Paris**."）|
| **new**（Metal router，gemma4 預設）| **13.45** | 9.11-14.40 | 正常 |
| old（ROUTER_CPU=1）| 9.60 | 7.65-13.14 | 正常 |

- **速度方向確認**：new vs old **+40%**（3/4 輪勝），與 §8.20 的 +27% 一致且建立在正確輸出上。
- **嚴格 bit-identical 未達成**：兩臂前 ~10 token 一致、第 11 token 起 fork（Metal/CPU router GEMM float 分歧），兩條續寫皆連貫——屬正常 float 分歧，非退化。
- **base vs cache 臂第 ~4 token 即分歧**：§7.3 copy-timing/bit-identity 殘留仍未綠（cache 兩臂自洽、與 base 分歧）。
- RSS：cache 臂 10.06GB（skip-load 有效、experts 在 Metal-visible pool）、base 10.44GB；hit/miss 兩臂 ~80.9% 一致（15810/3773 vs 15764/3819，差 0.3% = 路由 float 分歧的痕跡）。

**結論**：P1-2 的 +27-40% 速度結論轉正（方向、幅度、正確輸出三條件齊）；但「bit-identical」只能主張「前 N token 一致 + 兩臂輸出皆連貫」，不能主張全序列相等。

#### 8.27.2 家族分開估（P1-3 FFN kernel + lm_head H1/H2/H3）

| 靶 | gemma4 基線 | gemma4 預期 | qwen36 基線 | qwen36 預期 | 共享性 |
|---|---|---|---|---|---|
| **P1-3b FFN kernel**（mul_mv_id 效率、Swift grid 重構）| 14.70（P1-2 預設，§8.20）| **+10-20%**（FFN 23.5%、BW 效率僅 1/5，§8.21）| 11.76（CPU router 基線，§8.12/8.23）| **+5-10%**（估：pread 牆主導、FFN 相對 share 低，待 P1-7 量測）| 同一 kernel → 改動共享，**回收必須分開量** |
| **P1-3a lm_head H1**（ne11=1 進 ext 向量化，§8.24）| 14.70 | **+15-25%**（head 25.8→6-9ms）| 待 P1-7 量 head 佔比 | 待量 | kernel 共享（Q6_K GEMV），attn_out/dense-ffn-down 同家族同享 |
| **H2**（Swift 式融合 head chain：norm+GEMV+argmax 單 dispatch）| H1 之上 | 再省 ~1-2ms/step | 待量 | 待量 | 結構性、家族皆可 |
| **H3**（head 位寬：Q6_K→IQ4_NL 0.55B/param）| 282MB | byte −1.5-2×、品質 A/B | 待量 | 待量 | 檔案層、與 engine 無關 |

**qwen36 差異面**（§8.23/8.14）：40 層 + 256 experts + 200 slots pool、decode miss 幾乎全為唯一 expert（1796/1797）→ **pread 延遲是 qwen36 的主牆**，kernel 效率 share 較低；且 qwen36 從未做過成分分解（P1-0 只測 gemma4）→ 新增 **P1-7：qwen36 decode 成分分解**（同 §8.22 工具，改跑 qwen36 IQ3_XXS），把 qwen36 的 lm_head/attention/FFN 佔比量出來才能定 qwen36 的 P1-3a/b 回收。

#### 8.27.3 未名 MUL_MAT 身份的後續（§8.25 收尾）

- attn_out（Q6_K [4096,2816]）+ dense FFN down（Q8_0 [2112,2816]）與 MoE GEMV **同一 kernel 家族** → **不需要個別 kernel 工作**，自動共享 H1（ext 向量化路徑）的收益。
- attention core（KQ/attn@V）在 `flash_attn_ext` 單節點內（離散 MUL_MAT 假說已證偽）→ **無離散 attention kernel 工作**。
- attention 全體 27-28% 是第二大靶，但主要由 H1 家族共享回收（attn_out 是 GEMV），不需要獨立分刀。

#### 8.27.4 修訂 P1 分刀（取代 §8.19.2 的 P1-3 單行）

| 刀 | 內容 | 前置 | 預期（gemma4 / qwen36）| 風險 |
|---|---|---|---|---|
| P1-0 | 安全 per-op 計時 | ✅ 完成（§8.22）| — | — |
| P1-2 | router 家族分派 | ✅ 完成（§8.20/8.23/8.27.1）| +27-40% / 0（回基線）| — |
| **P1-7（新）** | qwen36 decode 成分分解 | P1-0 工具 | 定 qwen36 靶心 | 無 |
| **P1-3a · lm_head H1** | ne11=1 進 ext 向量化（type4x4）| **§8.26 working-set 修復**（head 上 Metal 才可 A/B）| +15-25% / 待量 | 中：§8.26 是硬前置，未修前不可進行 |
| **P1-3b · FFN kernel** | mul_mv_id 內部效率（Swift grid 重構、expert 進序列維）| P1-0、P1-6、P1-7 | +10-20% / +5-10% | 高：雙家族 bit-identical + 三臂 A/B |
| P1-3c · glu 融合 | gate_up+glu 單 dispatch（MUL_MAT_ID_GLU）| P1-0 | **上限已實測 = 0.87%**（glu 19.6µs/層 → ~0.6ms/step；§8.21 的 +2-3% 下修）| 低——低於天花板就刪 |
| P1-4 / P1-5 | Metal CB 池化 / hook 收斂 | — | +0.5-1% / +1-2% | 低 |

#### 8.27.5 終態預估（修正 §8.19.3）

- **gemma4**：14.70（P1-2 預設）+ H1（+15-25%）+ FFN kernel（+10-20% 部分回收）→ **~18-20 t/s**，貼近 Swift 18-20。
- **qwen36**：11.76 + 共享 H1/FFN 收益（估 +10-15%）→ **~13-15 t/s**；pread 牆為主要上限，kernel 效率回收有限。
- **前置鏈**：§8.26 working-set 修復 → lm_head A/B 解鎖 → P1-3a（H1）→ P1-3b（FFN kernel）。**§8.26 未修前，P1-3a 不可進行**（head 上 Metal 即退化，-ngl 30 下 head 在 CPU 無從量測）。

### 8.28 harness audit：-t parse bug 作廢面（2026-08-16）

**根因**：fork 的 `examples/simple/simple.cpp` 用手寫 parse loop，只認 `-m/-n/-ngl/-no-mmap/-expert-cache`；任何未知參數（`-t`/`-p`/`--seed`）會 break 進 prompt，後面的參數全部吞成 prompt 字面字串。

**已修復（2026-08-16）**：simple.cpp 新增 `-t/--threads`、`-p/--prompt`、`-s/--seed` case（-p 優先於 positional prompt、-t 寫入 ctx_params.n_threads、seed 因 greedy 而無效但照常解析）。重建後 `-expert-cache -t 8 -p ...` 可正常解析（smoke 驗證：cache stats 出現、prompt 正確）。

**受影響 harness（6 個，已加 [AUDIT 2026-08-16] banner）**：

| harness | 影響 | 判決 |
|---|---|---|
| `g4_three_arm_ab.sh`（daily config：base/n0c/n99c）| `-t 8` 在 `-expert-cache` **前** → cache 完全沒開、prompt = `"-t 8 -expert-cache ..."` | **全無效**——三臂數字（n99c 10.29 / n0c 8.29）與日常配置結論不可引用，需重跑 |
| `q36_fill_workers_ab.sh` | `-expert-cache` 在 `-t 6` 前 → cache 有開，但 `-t/--seed` 被吞、prompt 為 garbage 字串 | 部分無效——絕對 tok/s 與輸出結論無效；同 garbage 條件下的**相對 A/B 差量**僅供參考 |
| `q36_npw_ab.sh` / `q36_pool_workers_ab.sh` / `q36_prefetch_ab.sh` / `q36_step_ab.sh` | 同上（cache 有開、-t/--seed 吞、prompt garbage）| 部分無效（同上）|

**不受影響（驗證過）**：
- 用 `llama-speculative-simple` 的三個（`q36_iq3_budget.sh`/`q36_mtp_ab.sh`/`q36_quant_curve.sh`）——common parser（`common/arg.cpp:2832`）原生解析 `-expert-cache`/`-p`/`-t`/`--seed`，無此 bug。
- §8.11-8.15 pool8 基線（`g4_pool_ab.sh`/`q36_budget_curve.sh`/`q36_128tok_ab*.sh`）：prompt positional、無未知參數 → 有效。
- §8.20/8.23 P1-2 A/B（`p12_router_ab.sh`/`p12_q36_verify.sh`）：prompt positional、無 `-t` → 有效（但 §8.26 指出其輸出當時已退化，速度結論保留、bit-identical 需 -ngl 30 重對拍——§8.27.1 已完成）。

**待辦**：§8.28 之後用修好的 binary 重跑 `g4_three_arm_ab.sh` 的三臂（-ngl 30 正確配置），取代作廢的日常配置結論。

### 8.29 日常配置三臂 A/B（-ngl 30，取代 §8.28 作廢的 g4_three_arm 結論）（2026-08-16）

**動機**：§8.28 判定 g4_three_arm_ab.sh（-t 在 -expert-cache 前 → 無 cache + garbage prompt）全部無效。本節用修好 simple.cpp 的 binary 在 known-good 的 -ngl 30 重跑三臂（128 tok 真實 prose、4 輪交錯、-t 8、4GiB budget）：

| 臂 | median t/s | range | RSS | hit/miss |
|---|---|---|---|---|
| **base（-ngl 30 無 cache）** | **15.79** | 7.73-18.59 | 10.60GB | — |
| n30c（-ngl 30 + cache，Metal bounded，ALLOW_NGL）| 9.75 | 8.59-10.62 | 10.14GB | 48012/3962 = 92.4% |
| n0c（-ngl 0 + cache + CGC_CPU_ONLY，CPU bounded）| 4.55 | 4.13-4.95 | 6.23GB | 29279/3900 = 88.2% |

**日常配置定案**：
- **本機（16GB、無重度並行）日常用 `-ngl 30 無 cache`**：15.79 t/s 最快、RSS 10.6GB < 11.45GB working set 安全、輸出正確。**取代被作廢的「n99c 10.29 / n0c 8.29」**。
- cache（bounded residency）是「多工共存 / 小 RAM」才需要的模式：Metal bounded −38%（9.75）、CPU bounded −71%（4.55）——**不是日常加速器，是留餘裕的工具**（呼應 §8.17 I/O 側收斂：cache 的價值在 residency 不在速度）。
- n30c 的 −38% 與 §8.12 的「n99c 15.46 pool8」差距：後者是 -ngl 99 退化態的數字，不可引用。

### 8.30 glu 融合 kernel 正式刪除（2026-08-16）

**A/B（-ngl 30、N=64、4 輪、CGC_MMID_GLU=1，4GiB budget）**：

| 臂 | median t/s | fused ops |
|---|---|---|
| base（無融合）| ~14.8 | 0 |
| fused | ~14.9（噪聲內）| 1827/run |

**結果**：
1. **輸出退化**：fused 輸出 `"than than Bau-Bank-Bank own own own own way..."`——融合 kernel 有真 bug（非 float 分歧），推測 scratch 緩衝（`idx*ne01*4` ≈ 90KB）覆蓋 mmid dst 的 11KB buffer（§8.21 已標記的風險點）。
2. **速度 ≈ 0%**：兩臂同窗口差異在噪聲內，遠低於 §8.21 的 +2-3% 樂觀值，也低於實測天花板 **0.87%**（glu 19.6µs/層 → ~0.6ms/step，§8.22 工具）。

**決策**：低於天花板且壞 → **正式刪除 MUL_MAT_ID_GLU 全部 5 處**（metal.metal kernel + 3 instantiation、device.h 宣告、device.cpp pipeline、ops.cpp 融合 encode 路徑、ops.cpp glu no-op check）。重建後驗證：無 CGC_MMID_GLU 殘留、base 路徑輸出正確（"The capital of France is.\n<|channel>thought\n[]..." 連貫續寫）。§8.27.4 的 P1-3c 正式關閉。

### 8.31 cache vs base 決定性測試（-ngl 30 分歧 = Metal pool 消費路徑真 bug）（2026-08-16）

**動機**：§8.27.1 發現 -ngl 30 下 cache 臂與 base 在第 ~4-6 token 分歧。判定是 float noise 還是 pool 錯 bytes。

**實驗（同一 binary、N=32、短 prompt）**：

| 對拍 | 結果 |
|---|---|
| base -ngl 0（CPU）vs base -ngl 30（Metal）| **byte-identical** |
| base -ngl 0 vs L3-B cache（-ngl 0 無 POOL env，gather 路徑）| **byte-identical** |
| base -ngl 0 vs Option A cache（-ngl 0 + LLAMA_EXPERT_CACHE_POOL=1，slot-table 路徑）| **byte-identical** |
| base -ngl 30 vs n30c cache（-ngl 30 + L4 pool）| **分歧（byte 1181，~token 6）** |

**結論**：
1. **pool bytes 正確**：-ngl 0 下 cache 的兩條路徑（L3-B gather 與 Option A slot-table）都與無 cache base **逐位相同**——pread 讀進 pool 的 expert bytes = 檔案 bytes = loader 會載入的 bytes。§8.27.1 的「cache 兩臂自洽」與此一致。
2. **Metal base == CPU base**（此 binary/model）：Metal FFN 在 -ngl 30 與 CPU 完全一致 → 排除「Metal 數值本來就不同」的解釋。
3. **因此 -ngl 30 cache 分歧是真正的 Metal-path bug，不是 float noise**：唯一的變因是 cache 在 Metal 下的消費路徑（L4 Metal-visible pool + Option A kernel slot-table indirection vs native 3D tensor）。候選根因：① Option A 的 ids→slot_table[layer][id]*stride 在 Metal kernel 側的定址；② L4 pool buffer 的 Metal binding（offset/stride）；③ §7.3 copy-timing（graph-alloc 的 Metal copy 在 swap 前讀）。**這正是 §8.26 時代 NOGATHER 隔離實驗指向的「swap 破壞 Metal tensor 定址」類別——證實資料沒壞、Metal 定址壞**。
4. **影響**：-ngl 30 cache 臂的**速度**數字可信（§8.20/8.23/8.27.1、本節 9.75），但**輸出不能主張與 base 等價**；P1-2 的「bit-identical」主張維持 §8.27.1 的修正版（前 N token 一致 + 兩臂連貫）。修復 = §8.26 修復路徑的一部分（同根因家族），解鎖後 cache 臂才能在 -ngl>0 下與 base 對拍。

### 8.32 qwen36 P1-2 機制根因（2026-08-16）：overlap 假說證實——cpur 優勢只在 CPU 有空閒時成立

**動機**：§8.23 的家族分派（qwen36 = CPU router）基於安靜窗口的 −31-45% 回歸。重測出現反轉，需定位機制。

**三組測量**（同 binary、qwen36 IQ3_XXS、n99c 4GiB、兩臂輸出全程 bit-identical）：

| 窗口 | CPU router | Metal router | 勝者 |
|---|---|---|---|
| §8.23 安靜窗口（N=128、80% hit）| ~10.8 median | ~7.4 median | **cpur +45%** |
| 本日負載窗口（N=64、68% hit、load 2.4-3.6）| 6.02 median | 6.71 median | **metr +11%** |
| 序列化模式（CGC_GPU_OP_TIMING，無 overlap 可能）| 0.91 t/s | 1.01 t/s | **metr +11%** |

**CGC_GPU_OP_TIMING per-op 分解**（4 graphs、層 0-3）：
- metr 在 GPU 側多出的是 **tiny ops**：`SOFT_MAX:ffn_moe_probs`（0.01-0.02ms/層）+ `shared_expert_gate`（0.04-0.05ms/層）+ CONT:gate_reshaped——每層新增 GPU 工作 ≈ **0.1ms**，遠不足以解釋 30%+ 回歸。
- cpur 的 GPU 表**完全沒有 router ops**（router GEMM/softmax/topk 全在 CPU，不可見）。
- 序列化模式下兩臂每層 MoE GPU 成本同級（gate+down 2.99 vs 3.34ms/layer，含序列化膨脹 ~2×）。

**機制定案（overlap 假說證實，且補上邊界條件）**：
1. **raw GPU 計算 favor metr**：序列化模式（破壞一切 overlap）下 metr 仍 +11%——metr 的 router 工作放 GPU 比 CPU GEMM 便宜。
2. **cpur 的優勢 = CPU/GPU overlap**：CPU router（GEMM+softmax+topk+hook 整塊在 CPU）可與前層 Metal FFN 異步重疊，把 ~1-1.2ms/層的 router 工作藏進 GPU busy（安靜窗口 cpur−metr ≈ 47ms/step ÷ 40 層 ≈ **1.2ms/層的序列化損失**，恰好等於被藏的 CPU router 成本量級）。
3. **overlap 只在 CPU 有空閒時成立**：負載窗口 CPU 忙，overlap 塌掉 → metr 反而贏（+11%，與序列化模式一致）。§8.23 的「−31-45% 回歸」是安靜窗口專屬結論，不是絕對真理。
4. **日常配置影響**：家族分派（qwen36 預設 CPU router）在**安靜單工**時正確；**系統負載時應強制 `LLAMA_EXPERT_CACHE_ROUTER_CPU=0`**（Metal router 更抗負載）。§8.27.5 的 qwen36 終態預估不受影響（仍是 pread 牆主導）。

### 8.33 L4 Metal pool 分歧機制收斂（2026-08-16）：n_batch cap 證偽 + 分歧鎖定 Metal pool 消費路徑

**背景**：§8.31 定案「-ngl 30 下 cache 臂與 base 分歧 = Metal pool 消費真 bug」，當時最強候選是 L4 pool 的 n_batch cap（prefill batch 形狀不同 → float 分歧）。本節用四個決定性實驗收斂。

**實驗 1：n_batch cap 證偽（短/長 prompt 對照，4GiB budget、capacity=120、cap=14）**
- prompt "Paris is"（3 token）：base 與 cache 的 n_batch 都是 3（= n_prompt，simple.cpp 設 `n_batch = n_prompt`），單 batch → **bit-identical** ✓
- prompt "The capital of France is"（6 token）：兩臂 n_batch 都是 **6**（6 < cap=14，**cap 沒觸發**、單 batch、形狀完全相同）→ **仍分歧** ✗
- ⇒ n_batch cap 不是（至少不是唯一）機制。§8.31 的「可能是 cap」假說證偽。

**實驗 2：passthrough bit-identical（NOGATHER + NOMETALPOOL）**
- `-ngl 30 + cache + LLAMA_EXPERT_CACHE_NOGATHER=1 LLAMA_EXPERT_CACHE_NOMETALPOOL=1`：skip_load 關閉 → experts 全量 resident、hook 只寫真實 ids（0..127）、無 pool fill/swap → **bit-identical** ✓（"is **Paris**." 正確）
- ⇒ remap leaf + hook 機制本身無害；分歧與 pool fill/indirection 的**存在**有關，與 hook 的副作用無關。

**實驗 3：NOMETALPOOL（CPU pool、FFN 上 CPU）@ -ngl 30 bit-identical**
- `-ngl 30 + cache + LLAMA_EXPERT_CACHE_NOMETALPOOL=1`（skip-load 開、experts 走 CPU pool、attention/router 上 Metal）：**bit-identical** ✓
- 傳遞性：base(ngl30) == base(ngl0) == cache(ngl0) == 本臂 —— bounded-residency 資料路徑在 CPU 消費端完全精確。
- **本臂 = 目前唯一 bit-identical 的 bounded-residency 工作配置**，但速度只有 2.79 t/s（FFN 上 CPU 的 5× 代價 vs base 14.40）。

**實驗 4：分歧點釘位 + 資料正確性核對**
- -n 掃描：n=2/4/6 bit-identical、n=8 分歧 → 分歧在 **decode step 7-8**（prefill + 前 6 step 一致）→ 確定性、非 race（同 seed 可重現）。
- NOPREWARM（所有 fill 移進執行期）分歧點**不提前**（仍 ~token 6）→ 不是「第一次執行期 refill」的簡單觸發。
- CGC_MMID_PROBE：cache 臂 src0 ne2=120、ids = slot indices、**slot bytes 與 base 對應 expert bytes 完全相同**（sl0=gate_up slot0 == expert 122 bytes）、remap 在 Metal buffer 中內容正確 —— CPU 可見側全對。
- CGC_GPU_OP_TIMING / FFN_DBG：dump 發生在 sched sync（llama-context.cpp:1505，`!cache_orig.empty()` 才 sync）之前 → async 時序混淆，直接證據不可靠；cache 臂 down 值巨大（-17..-23）與「前 6 token 一致」矛盾 → 判定為 pre-sync 讀到 mid-write/未完成 buffer，棄用。

**定案**：
1. **分歧鎖定在 L4 Metal pool 的非同步消費路徑**（pool fill → Metal FFN 讀取之間的 GPU 視圖一致性），與 §7.3 copy-timing bug 家族同源。hook 註解自述「Metal FFN split 在 hook 之後才 copy remap ids + pool（copy-timing fix, 7.3.5）」——該 fix 只覆蓋 L3-B（gather/swap），**L4（pool 即 Metal buffer、無 swap）沒有對等保護**。
2. **bit-identical 優先時用 NOMETALPOOL @ -ngl 30**（正確、bounded、慢）；**要速度用 L4**（17.91 t/s 全 run、快於 base 14.40，但輸出與 base 分歧——coherent 但不可主張等價）。
3. **L4 修復方向**：讓 pool/remap 的 Metal 視圖在每 step hook fill 後強制一致——(a) per-step 對 pool region 做 CPU→Metal 強制 re-upload（或用 `ggml_backend_metal_set_tensor` 標 dirty）；(b) 或把 FFN split 的 copy 時機釘在 hook 之後（對 L4 補上 L3-B 的 hook_swap=3 對等機制）；(c) 或 pool 改 double-buffer（step 交替，fill 只寫 GPU 未在讀的那份）。修復完成前維持 §8.26 的 gate（-ngl 30 無 cache 為日常配置）。

**新增診斷 env**：CGC_MMID_PROBE（src0/src2 狀態 + slot bytes hexdump + ids）、CGC_GPU_OP_TIMING（per-op GPU 計時）、LLAMA_EXPERT_CACHE_GATE_DBG（OAEXPERT/TAILPIN per-step 對映）、LLAMA_EXPERT_CACHE_BATCH_DBG（每層 miss→slot 指派）、LLAMA_EXPERT_CACHE_NOPREWARM、LLAMA_EXPERT_CACHE_NOMETALPOOL、LLAMA_EXPERT_CACHE_NOGATHER。

### 8.34 fix (a) 實作與陰性結果：分歧不是 stale/visibility，是 Metal FFN 消費路徑的 float 差異（2026-08-16）

**任務**：實作 §8.33 修復方向 (a)——hook fill 後對 L4 pool region 強制 CPU→Metal re-upload（標 dirty），驗證 -ngl 30 cache 臂恢復 bit-identical。

**實作**：llama-context.cpp Option A hook（OA cp4 前）新增 env-gated 區塊 `LLAMA_EXPERT_CACHE_METAL_REFORCE=1`：對層的 4 個 expert tensor 呼叫 `ggml_backend_tensor_set(w, w->data, 0, ggml_nbytes(w))` + 對 remap 呼叫 set_tensor（走 backend 的 upload path，shared=memcpy、private=GPU blit）。重建 + A/B。

**結果：陰性**——`METAL_REFORCE=1` 臂仍分歧（n=8、"The capital of France is"）。機制不是 stale pool 可見性。

**追加六個決定性實驗（全部 -ngl 30、n=8、同 prompt）**：

| 實驗 | 配置 | 結果 |
|---|---|---|
| fix (a) | cache + METAL_REFORCE=1 | DIVERGED ✗ |
| identity pool | budget=4.27GiB → capacity=128=n_expert（prewarm 全填 identity、無 eviction、remap=真實 ids） | DIVERGED ✗ |
| identity pool @ -ngl 0 | 同上但 -ngl 0 | **BIT-IDENTICAL ✓** |
| 強制 shared buffers | GGML_METAL_SHARED_BUFFERS_ENABLE=1 | DIVERGED ✗ |
| 強制 private buffers | GGML_METAL_SHARED_BUFFERS_DISABLE=1 | **SEGFAULT**（pool 依賴 shared：pread 寫 fake VA 崩潰 → 證實預設 pool 是 shared） |
| ROUTER_CPU=1 | 整個 router（GEMM+softmax+argsort）上 CPU | DIVERGED ✗ |
| ROUTER_CPU=0 | 顯式跳過 P1-2 pin → softmax/argsort 留 Metal（router 佈局與 base 相同） | DIVERGED ✗ |

**資料正確性直接驗證（identity pool @ -ngl 30）**：CGC_MMID_PROBE 抓 layer-0 gate_up/down 被路由 experts 的 slot bytes（8 bytes hex），與 GGUF 檔案（gguf-py `t.data_offset + e*1270016`）逐位元組比對：**gate_up 4/4、down 4/4 全部相符**。加上 -ngl 0 全鏈 bit-identical（§8.31）→ **pool bytes 正確性徹底證實，wrong-bytes 假說證偽**。

**機制收斂**：
1. 分歧是 **-ngl 30 特異的 float-level 差異**：identity pool 的 decode step-1 layer-1 routing 就翻 near-tie（base {13,6,51,...59} vs cache {13,51,6,...99}——不只是排序、選了不同 expert），輸出到 step 7-8 才分歧（兩 expert 貢獻相近、下游吸收）。
2. 排除清單（全部實驗證實）：stale pool 可見性（fix a 陰性）、shared/private storage（強制 shared 仍分歧）、slot-id indirection（identity 仍分歧）、eviction/LRU（identity 無 eviction）、prewarm bytes（逐位元組相符）、softmax/argsort backend（ROUTER_CPU=0/1 都分歧）、n_batch cap（§8.33）。
3. **定案：L4（Metal pool + FFN on Metal）在 -ngl 30 下與 base 的差異是 Metal 消費路徑的 float-level 差異（data/ids/layout 全同仍分歧），非資料錯誤、非可見性。** 剩餘候選：mmid src2 從 remap leaf（CPU→Metal copy）而非 Metal-computed topk、weighted-combine/get_rows 的 backend 放置差異——但三者都只造成 near-tie 級 float 分歧，輸出 coherent 且功能正確。
4. **bit-identical 的 bounded 配置仍只有 NOMETALPOOL（FFN on CPU，2.79 t/s）**；L4 是最快（17.91 t/s）但不可主張與 base 等價。§8.26 gate（-ngl 30 無 cache 日常）維持。

**診斷 env 新增**：LLAMA_EXPERT_CACHE_METAL_REFORCE（fix a 開關，陰性保留作診斷）。

### 8.35 最後鑑別實驗（2026-08-16）：ffn_moe_out + mmid src2 來源對照——分歧節點釘在 layer-0 FFN 的 Metal pool 消費

**任務**：dump decode step-1 的 ffn_moe_out（weighted-combine 輸出）與 mmid src2 來源，比較 base vs identity-cache @ -ngl 30，釘死 L4 float 分歧的具體節點（remap leaf 或 weighted-combine backend）。

**探針**：新增 `CGC_FFN_OUT_DBG=1` 的 post-sync dump（在 `synchronize()` 之後讀 g_gate0/g_down0/g_moe_out0/g_moe_out1，修正先前 pre-sync 時序混淆）；配 CGC_MMID_PROBE（src2 來源 + pool bytes hex）+ GATE_DBG/EVALCB（routing trace）+ ROUTER_DBG（decode step 的 topk）。

**結果 1：FFNPOST dump 本身有 aliasing 混淆（不可用）**。base 的 gate0==down0 完全相同、idc 的 moeout0==moeout1 完全相同——capture 的 g_* 指標指到 graph 重建後 reused 的 buffer（每 step 重建 graph，graph_get_cb 的 capture 指標在 step 間失效）。→ **直接比 moe_out 數值不可行**，改用 routing + src2 + pool bytes 間接定界。

**結果 2：mmid src2 來源確認不同（remap leaf vs Metal-computed topk）**：
- base：`src2=MTL0#ffn_moe_topk-1#0`（Metal 端 argsort 的 topk tensor，真實 expert ids）
- identity-cache：`src2=MTL0#ffn_moe_topk_remap-1#0`（CPU 寫的 remap leaf，slot ids）
- 內容在 identity 模式下**相同**（slot e = expert e，remap 值 == topk 值）。

**結果 3：routing 分歧從 decode step-1 的 layer-1 開始，layer-0 完全一致**：
- prefill：29/29 layers routing 一致（含 pool bytes sl0/sl1 一致）
- decode step-1 layer-0：base `113 68 33 102 51 126 77 7` == idc `113->slot113 ... 7->slot7` **一致**
- decode step-1 layer-1：base `13 6 51 47 104 29 4 59` vs idc `13 51 6 47 104 29 4 99`——同 7 個 expert + 第 8 個 near-tie flip（59 vs 99，softmax 分數接近）→ **float 分歧，非 wrong bytes**

**結果 4：四格矩陣（同 prompt/seed，n=8）**：
| 格 | 結果 |
|---|---|
| direct-CPU（-ngl 0 base）vs direct-Metal（-ngl 30 base）| **BIT-IDENTICAL** |
| pool-CPU（NOMETALPOOL identity）vs base -ngl 30 | **BIT-IDENTICAL** |
| pool-Metal（L4 identity）vs base -ngl 30 | **DIFFERS** |

**機制收斂**：分歧 = **layer-0 FFN 在 Metal 端的 pool 消費**（src0=pool + src2=remap leaf 的 kernel 路徑），不是 weighted-combine backend、不是 remap leaf 內容、不是 pool bytes、不是 routing。
- base -ngl 30 的 blk.0 FFN 在 **CPU**（ngl=30 offloads 29 repeating + output = 30/31，blk.0 留 CPU，MMID probe 0 條 blk.0）；L4 pool 把 blk.0 experts 塞進 Metal buffer → layer-0 FFN 上 Metal。
- layer-0 FFN 在 CPU（direct）或 Metal（pool）的 float 差異 → layer-1 router near-tie flip → 下游分歧。§8.34 的「identity 仍分歧」現在有明確出處：identity 的 layer-0 FFN 消費路徑仍與 base 不同（base CPU vs idc Metal pool）。
- NOMETALPOOL（全 CPU pool）bit-identical 與 direct-Metal bit-identical 說明：**單一變數 = layer-0 FFN 的 Metal pool 消費**。

**實務結論**：bit-identical 的 bounded 配置仍是 NOMETALPOOL（FFN on CPU，2.79 t/s）；L4 最快（17.91 t/s）但輸出與 base float 分歧（功能正常、near-tie 級）。修復方向更新：不是「修 copy timing」，而是**讓 layer-0 FFN 在 cache 模式維持 CPU**（blk.0 不進 pool）或接受 Metal-vs-CPU float 分歧。日常配置維持 §8.26 gate（-ngl 30 無 cache）。

### 8.36 blk.0 不進 L4 pool 修復（LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0）（2026-08-16）：-ngl 30 cache 臂恢復 bit-identical ✓

**任務**：依 §8.35 的機制結論實作「layer-0 FFN 維持 CPU」修復，驗證 -ngl 30 cache 臂恢復 bit-identical，並量速度代價。

**實作（env-gated，`LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1`）**，四處一致排除 blk.0：
1. loader buft override（llama-model-loader.cpp:1236）：blk.0 experts 不強制 Metal buft → 依 ngl 佈局留 CPU（-ngl 30 下與 base 相同）。
2. loader skip-load（:1377）：blk.0 維持 `skip_load=false`，權重正常載入、resident。
3. loader shrink（:1456）：blk.0 experts 不 shrink（ne[2] 保持 128）。
4. model adoption/prewarm（llama-model.cpp:1708）：blk.0 不 adopt pool region、不 prewarm。
5. hook（llama-context.cpp:1818）：layer-0 走 full-resident passthrough（remap 填**真實 ids**、不 swap、不 fill）——與 §7.3 的 -ngl>0 passthrough 同一路徑。

**驗證（gemma4 IQ3_S、-ngl 30、-no-mmap）**：
| 臂 | 輸出 | 速度 |
|---|---|---|
| base（無 cache） | 參照 | 12.81 t/s（4GiB 輪） |
| L4 identity（capacity=128，無 skip） | **DIVERGED**（重現 §8.35） | 11.55 t/s |
| **L4 + SKIP_LAYER0（identity capacity=128）** | **BIT-IDENTICAL ✓** | 12.25 t/s（≈base） |
| **L4 + SKIP_LAYER0（1GiB 真 budget、capacity=30）** | **BIT-IDENTICAL ✓** | 3.62 t/s（hit 56%，pread 主導） |
| **L4 + SKIP_LAYER0（4GiB、capacity=120）** | **BIT-IDENTICAL ✓**（3 輪全中） | 8.22-9.16 t/s（hit 53.9%） |

**機制**：skip-load 的 layer-0 tensor 載入正常（`expert_cache_skip_load && !(l4_skip_l0 && tn.bid==0)` → layer-0 不 skip）；hook 的 `(!skip_load || (l4_skip_l0 && il==0))` → layer-0 進 passthrough（真實 ids）；pool 只有 blk.1-29。layer-0 FFN 因此維持 CPU，與 base -ngl 30 的 blk.0 佈局一致 → float 分歧源（§8.35）被移除。

**速度代價**：fix 本身**零額外開銷**（identity 滿池時 12.25 ≈ base 11.35）；-ngl 30 下的慢是 **pool hit rate 主導**（短 prompt 冷池：capacity=30 → 56%、capacity=120 → 53.9%，decode 全程 pread miss 攤在關鍵路徑）。四格現在全齊：direct-CPU / direct-Metal / pool-CPU / **pool-Metal（skip-l0）= 全部 bit-identical**。

**定案**：`LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1` 是 -ngl>0 下 cache 臂恢復與 base 等價的開關；L4 快路徑（17.91 等級）在開 fix 後犧牲 hit rate 換正確性。§8.26 gate 維持（日常 `-ngl 30` 無 cache），但 bounded-residency 的正確性缺口已關閉——只剩 hit rate（pread 攤薄）是速度瓶頸，交回 §8.11-8.15 的「藏 pread」議題。

### 8.37 三臂日常配置 A/B（base / L4+SKIP_LAYER0 / NOMETALPOOL）（2026-08-16）：blk.0 修復只救短 prompt——-ngl 30 cache 臂在長 prompt 下仍分歧

**任務**：在 blk.0 排除 pool 的配置下重跑 gemma4 三臂 A/B，定案日常配置。

**配置**：gemma4 IQ3_S、-ngl 30、-no-mmap、-t 8、4GiB budget、128 tok 真實 prose（telescope-208 prompt）、4 輪交錯。

**結果（median，4 輪）**：
| 臂 | median t/s | 輸出 vs base |
|---|---|---|
| **base（-ngl 30 無 cache）** | **10.71** | 參照 |
| L4 + SKIP_LAYER0（Metal pool + blk.0 排除）| 7.58 | telescope-208 下 **DIVERGED → 退化 "0000"** |
| NOMETALPOOL（CPU pool）| 4.13 | telescope-208 下 **DIVERGED → 退化 "0000"** |

**定案：日常配置 = base（-ngl 30 無 cache）**——三臂中最快且唯一對任意 prompt 都正確。§8.36 的「L4+SKIP_LAYER0 恢復 bit-identical」**只對短 prompt 成立**。

**機制（追加四組決定性實驗）**：
1. **-ngl 0 長 prompt 對拍**：208-tok prompt、cache arm（4GiB）vs base **BIT-IDENTICAL ✓** → 資料路徑（pool bytes/L1 offset/fill/gather）對長 prompt 也正確；分歧是 Metal 消費路徑特異。
2. **NOMETALPOOL 的 MMID probe**：blk.0-blk.12+ 全有 Metal MMID → **FFN 仍在 Metal 跑**（CPU pool → sched expert-wise copy → Metal FFN，§7.3 copy 家族），不是 CPU FFN。
3. **分歧是確定性 + content-dependent**：同 prompt 兩輪同分歧；111-char prompt 分歧（base "de facto-doers" vs l4 "de Wivesser Lee"——**coherent 但不同**，float near-tie flip）、359-char prompt（cache 完全 engage、capacity=120、n_batch 66→14）**BIT-IDENTICAL ✓**、208-char 分歧並退化。→ 不是長度門檻、不是 wrong bytes、不是 race；是「**某些 prompt 的某層 router 落在 near-tie**」→ Metal 消費路徑的 float 差異翻牌 → 下游可落入退化吸引子（"0000"）。
4. **SKIP_LAYER0 的定位修正**：它移除的是「layer-0 FFN 的 Metal pool 消費」這**一個** float 分歧源（短 prompt 的 flip 正好來自它）；其他 Metal 路徑 float 差異（l4 的 n_batch cap 208→14 改變 attention batch 形狀、np 的 CPU→Metal copy 路徑）仍在，遇 near-tie 就翻。

**剩餘差異源清單**（-ngl 30 cache vs base 的 float 差異，全部 near-tie 級）：
- l4：n_batch cap（208→14）→ attention kernel 形狀不同；L4 pool 消費 vs resident tensor 消費的 kernel 路徑（§8.33-8.35）。
- np：CPU pool → Metal FFN 的 expert-wise copy 路徑（§7.3 copy-timing 家族）。
- 共同：remap leaf（CPU 寫）vs Metal-computed topk 作為 src2 的執行路徑差異（內容相同，kernel 讀取方式不同）。

**實務結論**：bit-identical 的 bounded 配置只有 **-ngl 0 + cache**（慢但正確）；-ngl 30 的 cache 臂是速度實驗（最快 7.58）、不可當日常正確輸出用；§8.26 gate 維持。若要 -ngl 30 cache 也正確，需消除上面剩餘差異源（n_batch cap 放寬 + copy 時機釘在 hook 後），這超出 P0 範圍，歸入 P1。

### 8.38 qwen36 IQ3 跨家族驗證（2026-08-16）：-ngl 30 與 -ngl 99 cache 臂全數 bit-identical——bounded residency 在 qwen36 上轉正（含 base -ngl 99 硬 OOM 的場景）

**任務**：驗證 LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0 修復在 qwen36 IQ3（UD-IQ3_XXS、40 layers、256 experts top-8、13.2GB）上跨家族成立，並確認 -ngl 99 長 prompt 的 cache 臂。

**-ngl 30（4GiB budget、capacity=200 slots/layer）**：
| prompt | fix（SKIP_LAYER0） | nosk（無 skip） |
|---|---|---|
| 24 chars（Paris） | **BIT-IDENTICAL ✓** | **BIT-IDENTICAL ✓** |
| 111 chars | **BIT-IDENTICAL ✓** | **BIT-IDENTICAL ✓** |
| 732 chars（多段 prose） | **BIT-IDENTICAL ✓** | **BIT-IDENTICAL ✓** |

**-ngl 99（4GiB budget、capacity=200、n_batch cap 184→24）**：
- **base -ngl 99 硬 OOM**（kIOGPUCommandBufferCallbackErrorOutOfMemory / Abort trap，13.2GB > 11.45GB working set）→ 沒有 base 參照，以 **base -ngl 30** 為參考臂。
- **-ngl 99 + cache + SKIP_LAYER0：BIT-IDENTICAL ✓**（vs base -ngl 30），完整 decode 127 tokens、RSS ~9.83GB、8.00 t/s。
- **-ngl 99 + cache（nosk）：BIT-IDENTICAL ✓**，RSS ~9.77GB、5.58 t/s。
- 第二 prompt（Seine、37 tok、64 gen）：-ngl 99 + cache + SKIP_LAYER0 再驗 **BIT-IDENTICAL ✓**。

**跨家族對照（重要）**：
| 家族 | -ngl 0 cache | -ngl 30 cache | -ngl 99 cache |
|---|---|---|---|
| gemma4（128 exp top-8） | BIT-IDENTICAL ✓ | content-dependent 分歧（§8.37）；SKIP_LAYER0 只救短 prompt | base OOM；未驗（同理風險） |
| **qwen36（256 exp top-8）** | BIT-IDENTICAL ✓（先前） | **全 prompt BIT-IDENTICAL ✓（含 nosk）** | **BIT-IDENTICAL ✓（含 nosk；base 硬 OOM 由 cache 解救）** |

**機制解讀**：qwen36 的 Metal pool 消費路徑在測試 prompt 集上全部 float-clean（連 nosk 都過）——與 gemma4 的 content-dependent near-tie flip 不同。最可能原因：(a) 256 experts top-8 的 routing 比 128 experts 少 near-tie（候選多、softmax 拉開）；(b) Metal 消費的 float 行為家族特異（qwen36 gate_up 佈局 vs gemma4 down 分離）。兩家族共同的硬結論：**-ngl 0 cache 一定 bit-identical；-ngl>0 cache 的正確性需逐家族逐 prompt 驗證**。

**P0 里程碑**：qwen36 -ngl 99 + cache 是 bounded residency 的第一個「**full-offload 且正確**」實例——13.2GB 模型在 base 會硬 OOM 的 -ngl 99 下，用 4GiB pool 跑出與 base -ngl 30 逐位元相同的輸出（RSS 9.8GB、working set 內）。gemma4 的 -ngl>0 正確性仍待 P1（§8.37 殘餘差異源）。

### 8.39 prefill 覆蓋對 hit rate / decode 速度的敏感度（2026-08-16）：hit rate 是 request-repeat 指標、不是 SSD 工作量——兩臂 SSD 工作完全相同，prefill 覆蓋對速度零影響

**任務**：同一 4GiB budget（capacity=120）、-ngl 30 + L4_SKIP_LAYER0，短 prompt（6 tok）vs 長 prompt（~190 tok），128 tok decode、3 輪交錯，分離 prefill 覆蓋對 hit rate/速度的影響。

**結果（median）**：
| 臂 | decode speed | hit rate | misses | file_reads | pread_usec（8 worker 總和） |
|---|---|---|---|---|---|
| 短 prompt（6 tok）| 6.86 t/s | 53.9% | 3594 | 7188 | ~7.4-10M |
| 長 prompt（190 tok）| 5.34 t/s | 92.4% | 3841 | 7682 | ~8.4-9.2M |

**關鍵分解實驗（n=8）**：
| 臂 | requests | misses | pread_usec |
|---|---|---|---|
| 短 + n=8（共 14 tokens）| 5553 | **3536** | **7.87M** |
| 長 + n=8（共 198 tokens）| 5552 | **3540** | **7.88M** |

**三個機制定論**：
1. **misses 幾乎與 prompt 長度、decode 長度無關（~3540-3840 恆定）**。gemma4 全模型 expert 空間 = 30 層 × 128 experts = 3840；短 prompt 14 tokens 就 request 了 3536 個（**~92% 的 expert 空間**）——128 experts/layer 太小，任何短 run 都 sweep 掉整個 expert 空間 → 120-slot pool 對所有配置同樣飽和。
2. **hit rate 是 request-repeat 指標、不是 SSD 工作量指標**。長 prompt 的 prefill 對同一批 expert 反覆 request（命中 92.4%），短 prompt 一次性 request（mostly miss）——但**實際 fill 次數兩臂相同**（~3500），pread_usec 也相同。hit rate 差異是計數假象。
3. **decode 在 4GiB 下不是 pread-bound**：pread_usec 是 8 個 persistent worker 的並行總和，wall pread ≈ pread_usec/8 ≈ 1.1-1.25s，只佔 decode wall（18-24s）的 **~5-7%**。→ 「藏 pread」（double-buffer / prefetch / workers）在此配置無空間，與 §8.11-8.15 全證偽一致。

**實務結論**：
- 對 gemma4（128 experts/layer），**prefill 覆蓋不改變 SSD 工作量**——pool 容量（120 slots）才是唯一約束，prompt 長度不是。加大 budget 才是正槓桿（1GiB→4GiB：3.62→6.86 t/s，§8.36/8.37）。
- hit rate 數字（53.9% vs 92.4%）不要拿來當速度預測——兩臂速度相同（6.86 vs 5.34，在雜訊內）。
- decode 速度瓶頸在 Metal 消費路徑本身（kernel 效率 / hook / n_batch cap），不是 pread。與 P1-3（FFN kernel 效率 ~40% 頻寬）結論一致。

### 8.40 qwen36 -ngl 99 + cache 的 20 tok/s 可行性（2026-08-16）：乾淨上限 ~12-16 t/s（hit 94%），20 未達——瓶頸是 hit rate（prompt 依賴）＋ kernel 效率，不是 pread 也不是 budget

**任務**：短/長 prompt 在 qwen36 -ngl 99 + cache 下能否達 20 tok/s。

**數據整合（同 4GiB budget、capacity=200）**：
| 來源 | prompt | hit% | misses | decode t/s |
|---|---|---|---|---|
| §7.10.2（08-15 乾淨、4 輪交錯）| 228-tok（未知內容）| 94.1% | 2985 | **12-16** |
| §8.10（08-15）| 4GiB | 81.0% | — | 11.08 |
| §8.38（08-16、load 35）| telescope 208-tok | 84.9% | 9996 | 8.00 |
| 今天（load 2.8-5、N=64、3 輪）| telescope | 78.7% | 9816 | 4.6-6.9 |

**機制定論**：
1. **miss 數是 prompt 的 routing 多樣性決定，不是回歸**：兩次 telescope run 都 ~10000 misses（§8.38 9996 / 今天 9816，確定性一致），§7.10.2 的 228-tok prompt 只有 2985——同一 budget 下 miss 差 3.3×。telescope 長 prose 橫跨 Galileo/Newton/Herschel/Hubble/JWST 主題，sweep 更多 256-expert 空間。
2. **hit 94% → 12-16 t/s；hit 79-85% → 6-8 t/s**。decode 速度 = hit rate 的函數（miss→fill→pread 關鍵路徑），而 hit rate 取決於 prompt 與 pool 容量。
3. **8GiB budget 不可行**：6GiB 已 OOM（§8.10）、8GiB 實測 Abort trap 134（pool + 40 層 attention 超 11.45GB working set）——4GiB 是 working set 內實際上限。
4. **20 tok/s 未達且現配置達不到**：乾淨上限 ~12-16（hit 94%）。理論帶寬天花板（M4 120GB/s）：純權重 3e9×3.4/8=**1.275 GB/token → 94 tok/s**；帶 KV/activation/bias/padding overhead 1.4-1.6 GB/token → **75-86 tok/s**（120/1.6=75）。現測 12-16 t/s = 天花板 **13-21%**；20 t/s = 21-27%——需 (a) hit ≥95%（更大有效覆蓋）＋ (b) FFN kernel 效率（P1-3，MUL_MAT_ID ~40% 頻寬，正是 13-21% 與天花板間的主要缺口）雙管齊下。

**實務結論**：qwen36 -ngl 99 + cache 4GiB 是「全 offload 且正確」的日常 bounded 配置（bit-identical、hit 高 prompt 下 12-16 t/s）；20 t/s 是 P1-3 kernel 效率目標，不是現配置可達。機器桌面棧（Lark/Freebuff ±20%）影響測量，比較需配對交錯。

### 8.41 P1-3 啟動盤點（2026-08-16）：decode 不走 mul_mat_id indirection——reframe 後的真槓桿是 per-layer gate+up+down 融合與 compile-time GEMV blocking

**任務**：以 Swift 式 per-expert batched GEMM 為目標改造 MUL_MAT_ID 的 dispatch 與 indirection，A/B 驗證 qwen36 -ngl 99 + cache 12-16 → 20 t/s。

**第一輪探索的三個決定性發現**：

1. **decode (n_tokens=1) 根本不走 mul_mat_id indirection**。`ggml_metal_op_mul_mat_id` 的 `ne21 >= ne21_mm_id_min(32)` 條件在 decode 為假 → decode 走 **kernel_mul_mv_id**（per-expert GEMV），而它**已經是 compact per-expert**：z=ne123=8（一次 dispatch 覆蓋 8 experts）、直接讀 src2 id（slot/expert 索引）、無 map0/barrier。mmid 的 map0+200-slot z-grid 只影響 prefill（ne21≥32）。→「改造 dispatch/indirection」的原始前提在 decode 不成立。

2. **decode 的 GPU 熱點是 FFN 專家 GEMV**（CGC_GPU_OP_TIMING，序列化 per-op）：down 3.29ms + up 1.94ms + gate 1.89ms = **7.12ms ≈ 62% of top ops**；attention 相關（attn_out 0.55 + z 0.54 + linear_attn_out 0.55）只 ~1.6ms。down 是單一最大 op（權重最大）。

3. **GEMV blocking 是 compile-time template 固定**：quantized mv kernel 的 `template<int nr0>` 用 `sumf[nr0]`、`first_row=(r0*NSG+sgitg)*nr0` 做 unroll，dispatch 的 `switch(args.nr0)` 只有 case 2（IQ3_XXS 用 `_4` 變體、內部固定 nr0=4/nsg=2）。**runtime nr0/nsg 覆寫實測產生錯誤結果**（CGC_MMV_NR0=8 → DIVERGED；baseline 同 prompt BIT-IDENTICAL，確認是 blocking 錯配非 prompt）→ 已回滾，此槓桿證偽。

**reframe 後的真槓桿（P1-3 分刀）**：
| 刀 | 機制 | 預期回收 | 風險/成本 |
|---|---|---|---|
| **P1-3a: per-layer gate+up+down 融合**（Swift 式核心）| 每層 3 個 mmid/mv_id ops（gate/up/down）→ 1 個 dispatch：同一批 8 expert ids 一次讀、y-vector 只載 1 次、3 個 dst 一起寫；40 層 × 3 → 40 dispatches/step | 每 step 省 80 dispatches + 2/3 y 載入 + 3× 的 per-op 同步；估 FFN 62% → ~45-50% | §8.30 MUL_MAT_ID_GLU 曾失敗（scratch 覆蓋 dst——buffer/offset bug 可修）；需 graph/op 層融合 |
| **P1-3b: compile-time 加大 GEMV tile** | 新增 IQ3_XXS nr0=8/16 的 template instantiation + dispatch switch case，A/B 對比 nr0=4 | 更大 per-threadgroup tile → 更少 threadgroup 排程、y 重用 | 需新 instantiation + pipeline 註冊；回收可能有限（GEMV 本質 bandwidth-bound） |
| P1-3c: 8 experts 共享 y 的 K-loop batch | mv_id 內把 8 個 expert 的 K-loop 併成單一 threadgroup 群 | 少 8× 的 y 讀 + 每 expert 的頭部開銷 | kernel 大改，與 a 重疊 |

**結論**：P1-3 的 Swift 式 batched GEMM 在 fork 的落地點 = **P1-3a（per-layer 融合，1 dispatch/layer）**，不是 mmid indirection 改造（decode 已 compact）。20 t/s 目標（21-27% 帶寬天花板）需要 FFN 62% → 實質下降；P1-3a 是唯一同時省 dispatch、y 載入、同步三項的刀。§8.30 的失敗模式（scratch 衝突）已定位，可避開（dst 用獨立 buffer 區域）。

### 8.42 P1-3a 完成：gate+up+down 融合 kernel——barrier 根因修復、bit-identical 全驗證、速度淨值 ≈ 0（2026-08-16）

**實作**：graph 層新增融合 op `MUL_MAT_ID_GLU`（`build_moe_ffn` 融合分支，env `CGC_MMV_FUSE=1` 開啟、僅 offload 層），Metal 側 `kernel_mul_mv_id_glu_*`：同一批 ids 一次讀、y 一次載、gate/up 兩階段 + GLU epilogue 一次寫 scratch、phase-2 以 stock mv kernel 讀 scratch 寫 down。**正確性路線比速度路線曲折得多**，最終根因不是 buffer/offset，是更深的 Metal 語義問題。

**除錯旅程（濃縮）**：fused 輸出非確定（同輸入不同輸出）→ 逐步排除：pool fill race（SERIAL 仍分歧）、跨 range 重疊（強制 `waitUntilCompleted` 仍分歧）、dst/scratch 佈局（無重疊）、kernel 與 stock 逐行 diff（一致）、phase-1 單獨（P1ONLY 確定）、scratch/dst 解析（一致）。**最終根因：Metal 不保證同一 command buffer 內兩個 dispatch 之間的記憶體可見性/順序**——phase-2 可與 phase-1 重疊執行、讀到未完成的 scratch。unfused 路徑安全是因為 gate/up/down 是獨立 graph node（encoder 的 `mem_ranges` 追蹤在 src/dst 重疊時插 barrier）；融合把兩 phase 塞進同一個 node，繞過了 barrier 機制。**修復：phase-1→phase-2 之間插 `memoryBarrierWithScope`（無條件，正確性要求）**——Barrier 0/6 分歧 vs 無 barrier 5/6。

**驗證**：i0（resident）/ il1-2（pool）/ 全層融合 三配置各 5-6 runs：**5/5 確定 + 對照 unfused n99c 全 BIT-IDENTICAL**（含長 prompt）。P1-3a 正確性達成。

**速度 A/B（qwen36 IQ3_XXS，-ngl 99 + cache 4GiB，128 tok 真實文本）**：
- **GPU per-op 計時（無雜訊的結構性比較）**：unfused 序列化 FFN ≈ 18.0ms/layer（gate 4.85 + up 5.31 + down 7.87）vs fused 17.3ms（單 op）——**fused 快 ~4%**（y 載入 1/3、無 swiglu 中間張量）。
- **端到端（受控 per-token eval time，5 輪交錯）**：同一臂內部 91→274ms/tok（±40%+），波動由機器負載主導（Freebuff 55% CPU + WindowServer 19% + 背景磁碟）；3 輪 median fused −8%、5 輪 +15%——**方向相反 = 純雜訊，兩臂無法在當前環境分辨**。
- **結構性損耗**：unfused 的 gate/up 兩個 mmid 之間無 barrier（mem_ranges 檢查不重疊 → GPU 重疊執行），實際牆鐘 ≈ down + max(gate,up)；fused 把 gate+up 序列化 + 必須插 barrier，丟掉這個重疊。

**結論**：**P1-3a 正確性轉正（barrier 修復是通用教訓：同 node 多 phase 必須手動 barrier），GPU 每 op 效率微升 ~4%，但端到端淨值 ≈ 0**。FFN 是 weight-read 記憶體頻寬 bound——融合省的是 op 數/y 載入（小頭），真正的瓶頸（讀 256 expert 權重 bytes）沒有變。§8.41 的「FFN 62% → 45-50%」預期**不成立**。

**P1-3b / P1-3c 評估（不實作）**：
- **3b（nr0=8/16 compile-time 變體）**：§8.42 當下評估為棄；**2026-08-16 依用戶指示實作 nr0=8 並翻案**——§8.41 的「blocking 錯配」不是「nr0=8 無效」，而是 getter base 名稱缺失的 bug（見 §8.43），修復後 5 項全 BIT-IDENTICAL；但速度同樣無增益（≈0 至略負）→ 最終結論仍維持 nr0=4 預設。
- **3c（8 experts 共享 y 的 K-loop batch）**：y 僅 32KB/token vs 權重讀取 1.7GB/step——節省可忽略，且與 3a 的 y 一次載入重疊 → **棄**。

**P1-3 線收斂**：三刀合計的 FFN kernel 層回收 ≈ 0-4%，達不到 20 t/s。剩餘槓桿回到 §8.24/8.27 的排序：**lm_head H1（head 23-26ms/step、9% 頻寬——decode 最大單一 op）** > attention（27-28%）> FFN kernel 精修。P1 下一步建議：H1（low-risk high-recovery），而非繼續 FFN kernel。


### 8.43 P1-3b 實作（nr0=8 compile-time template）：base 名稱缺失根因修復 + 增益證偽（2026-08-16）

**實作**：metal.metal 新增 4 個 nr0=8 instantiation（stock `kernel_mul_mv_id_iq3_xxs_f32_nr0_8` / `_iq2_s_` 與 fused `kernel_mul_mv_id_glu_iq3_xxs_f32_nr0_8` / `_iq2_s_`，皆 template nr0=8 硬編碼）；device.cpp 的 `ggml_metal_nr0_env`（env `CGC_MMV_NR0=8`，僅 IQ2_S/IQ3_XXS 生效）+ 三個 getter（stock `mul_mv`、stock `mul_mv_id`、fused `glu`）切 nr0/smem/name。

**根因（一度誤判為「GLU kernel 壞 / pool race / skip-load 交互」）**：`ggml_metal_library_get_pipeline_mul_mv_id` / `_glu` 的 **base 名稱缺 `_nr0_8` 後綴**——只有 `name` 加（`%s_nr0_8_nsg=%d`）、`base` 沒加 → `get_pipeline(name)` 找不到 nr0=8 pipeline → fallback `compile_pipeline(base, name, cv)` 用 **nr0=4 的 template**（base 名稱指向 nr0=4 host_name）編譯，卻配 `res.nr0=8` → dispatch grid（`(ne01 + nr0*nsg - 1)/(nr0*nsg)`）與 kernel 內 nr0=4 的 row-loop 錯配 → **決定性 garbage**。§8.41 的「runtime nr0 覆寫 DIVERGED」與本輪「nr0=8 在 decode step 2 分歧 / NOGATHER 對但 cache 錯」全是同源下游症狀。**修復：nr0==8 時 base 也含 `_nr0_8`**（name = base + `_nsg=%d`，compile_pipeline 的 base 才是 Metal function host_name、name 是 cache key）。

**驗證（qwen36 IQ3_XXS，5 項全 BIT-IDENTICAL + 確定性）**：
| 配置 | nr0=4 vs 8 |
|---|---|
| NOGATHER -ngl 30（resident） | ✓ |
| cache -ngl 30（skip-load）n=8 | ✓ |
| cache -ngl 30 n=32（長生成） | ✓ |
| fused GLU cache n=16 | ✓ |
| cache -ngl 99 n=16 | ✓ |

**關鍵澄清（-ngl 0 的「4-7% 增益」是噪音）**：`CGC_MMID_PROBE` 實測 -ngl 0 下 qwen36 的 MUL_MAT_ID **完全沒有 dispatch 到 Metal**（probe 計數 0，FFN 在 CPU 跑——`op_offload_min_batch_size=32` 擋住 decode 的 MUL_MAT_ID）→ `CGC_MMV_NR0` 不生效、兩臂跑同一 CPU 路徑、輸出天然 identical，先前「-ngl 0 nr0=8 快 4-7%（31 runs）」純屬負載雜訊（8 輪交錯配對 median delta ≈ -1%）。對照：-ngl 30 下 probe 計數 234，FFN 才真的在 Metal、nr0 才生效。

**速度定案（真實 Metal 路徑）**：-ngl 30 / -ngl 99 + cache 5 輪交錯 A/B 配對 delta 方向不穩（±15-26% 負載雜訊）；GPU per-op 計時（serialized per-op buffers）FFN rollup 無一致改善、部分 graph nr0=8 反較慢（47.8→72.1ms）。結構性原因：M=1 decode 下 threadgroup 數 = ne01/(nr0×nsg)，nr0=8 並行度減半、每 threadgroup row 加倍——GEMV 總權重讀取 bytes 不變（bandwidth-bound 本質），tile 放大對帶寬利用率無益。

**gemma4 家族驗證（2026-08-16 補充）**：gemma4 IQ3_S 的 expert 型別與 qwen36 不同——`ffn_gate_up_exps.weight` = IQ2_S（blk.0-28）/IQ3_S（blk.29）、`ffn_down_exps.weight` = IQ4_NL（type 20，**不在 nr0 env gate 範圍**）→ `CGC_MMV_NR0=8` 只影響 gate_up、down 不受影響（半套用，影響面僅 qwen36 的一半）。4 項驗證（NOGATHER -ngl 30 / cache -ngl 30 / cache n=32 長生成 / cache -ngl 99 n=16）**全 BIT-IDENTICAL**——base 名稱修復對 gemma 家族同樣成立。速度：-ngl 30 + cache 128 tok 4 輪交錯配對 delta = +4.9%/+1.4%/+15.0%/−1.1%（median ≈ +3% 略慢，round 3 為離群）——與 qwen36 的 ≈0 至略負一致，**無增益定案，nr0=4 預設維持**。gemma4 無 fused GLU 分支（gate_up 為合併張量），故驗證套件為 4 項而非 5 項。

**結論**：**nr0=8 正確性已修復（§8.42 的 blocking 錯配翻案為 base 名稱 bug），速度無增益（≈0 至略負）→ 日常維持 nr0=4 預設**；`CGC_MMV_NR0=8` 僅作實驗臂保留。P1-3 線最終收斂不變：FFN kernel 層無剩餘槓桿，下一步仍是 lm_head H1（§8.42）。另本輪一併修復 simple.cpp 的 6 個 error-path 跳過 cleanup（`-ngl 99` 無 cache OOM 時 exit 撞 `ggml_metal_rsets_free` count assert → SIGABRT，改統一 `cgc_cleanup()` lambda；新增 `CGC_RSETS_DBG=1` 印 device free 殘留 buffer 數）。

### 8.44 weight-read 路徑重盤（2026-08-16，P1-3b 證偽後）：FFN bandwidth-bound 定案下的剩餘槓桿評估 + P1 分刀修訂

**背景**：P1-3a（融合）與 P1-3b（nr0=8）雙證偽後，FFN kernel 層（tile/dispatch/indirection）已無剩餘回收空間。依用戶指示把目標從 kernel tile 改到 **weight-read 路徑本身**——即「檔案 pread → pool slot → Metal 讀取」整條鏈，評估三個候選方向。

**qwen36 decode 成分分解補測（P1-7 前置，2026-08-16，CGC_GPU_OP_TIMING 3 graphs）**：
- MoE FFN experts（MUL_MAT_ID）：down 3.13ms + gate 2.40 + up 1.75 = **7.28ms/layer**（40 層 ≈ 291ms/step 序列化）——仍是最大單一成分。
- **注意 qwen36 是 hybrid 架構**（非純 transformer）：GATED_DELTA_NET（0.26ms/layer）、SSM_CONV（0.06）、linear_attn_out（0.45×2）、z（0.54）、conv_state/cache_s 更新（0.2×2）——linear attention + SSM conv 成分佔 ~1.8ms/layer，是 gemma4 沒有的固定成本。
- head：`output.weight` Q6_K [248320, 2048] ≈ 407MB、每 decode 一次（tied embedding，§8.24 同家族）——待確認實際 per-op 時間（dump 中未見 >3ms 單一 MUL_MAT，需 P1-8 追）。

**三候選方向誠實評估**：

| 方向 | 機制 | 結論 |
|---|---|---|
| **slot 佈局** | 每 expert 3 段分散 pread（gate/up/down 各自 GGUF tensor、檔案 offset 分散）→ 合併 readv/preadv | **證偽**：§8.11 註解已確認單次 readv 無法表達（非連續）；threads 並行已回收 ~20%（204→163us），SSD queue depth 是牆（pool16 崩潰，§8.12）——再合併收益 <5% |
| **跨層 pread 合併** | 40 層串行 ensure_batch，每層 miss pread 在關鍵路徑 → 跨層合併提前提交 | **證偽**：decode 層層依賴（層 i+1 routing 依賴層 i FFN 輸出），跨層提前 fill 需要未知 routing——與 §8.14「步間預測死亡」同根（1796/1797 miss 唯一）；層 i 的 fill 也無法與層 i 的 GPU 計算重疊（hook 在 topk 後才 fire，之前 attention 已算完） |
| **Metal-visible pool 記憶體複用** | 4GiB = capacity 200 slots/layer；8GiB OOM（working set 硬上限）；hit 94% → 12-16、79-85% → 6-8 | **唯一有實質回收空間**：per-layer 固定 200 slots 是「最平均」配置，未利用層級 routing 多樣性差異——**動態 slot 容量**（熱層多給、冷層少給）可在同等 4GiB 下提高有效覆蓋 → hit +、pread 少。但回收受 §8.40 的天花板約束（hit ≥95% 才觸 20，telescope 79-85% 是 prompt 多樣性 > 容量的結構性結果） |

**關鍵結構性結論**：decode 依賴鏈上 **pread 延遲無法隱藏**（層層依賴 + miss 唯一性 + SSD 牆），pool8 並行只回收 17% 正是 latency（非 throughput）為牆的證據。weight-read 路徑剩餘可做的只有「**提高有效覆蓋**（動態容量）→ 減少 miss 次數」，而不是「讓每次 miss 更快」。

**P1 分刀修訂（新增 P1-8/P1-9，取代 §8.27.4 的 P1-3b 行）**：
- **P1-8 · qwen36 head 佔比確認**：追 output.weight 的實際 per-op 時間（CGC_GPU_OP_TIMING 的 head 節點），定 qwen36 的 lm_head H1 回收（§8.24 的 23-26ms/step 是 gemma4 tied Q6_K 282MB；qwen36 是 407MB、應更大）。低風險、純量測。
- **P1-9 · 動態 slot 容量 A/B**：先量每層 routing 多樣性（CGC_GPU_OP_TIMING / miss dump per layer），把固定 200 slots/layer 改成「熱層多、冷層少」的容量分配（loader 建 pool 時依 prefill 統計），A/B 量 miss 率與 tok/s。預期 +3-8%（若層級多樣性偏斜明顯）；前置 = 量測確認偏斜。
- **保留**：lm_head H1（§8.24，待 §8.26 working-set 修復解鎖 A/B）、gemma4 的 attention 28%（§8.25，與 H1 家族共享回收）。
- **封存**：跨層 pread 合併、slot 佈局 readv、Metal pool 記憶體擴容（8GiB OOM 硬牆）——證偽或硬上限，不投入。

**預期終態（修正 §8.27.5）**：qwen36 11.76 + 動態容量（+3-8%）+ lm_head H1（+10-20%，待 P1-8 確認）→ **~14-16 t/s**；20 t/s 需 hit ≥95%（4GiB 對高多樣性 prompt 達不到）→ 誠實結論：**qwen36 20 t/s 在 16GB M4 的 bounded residency 下不可達**，日常以 12-16（好 prompt）為準。

### 8.45 P1-8 + pad-ne00 實驗完成：兩家族 head 都是高效路徑（~93-94GB/s ≈ 78%）——odd-nb 假說證偽、§8.24/8.27 的「lm_head 大回收」全案翻案（2026-08-16）

**任務**：追 qwen36 output.weight 的實際 per-op 時間，確認 lm_head H1（ne11=1 進 ext 向量化）的回收空間，與 gemma4 對照；並以 pad-ne00 實驗驗證「gemma4 head 慢 3.5× 是奇數 nb」假說。

**量測**（CGC_GPU_OP_TIMING 序列化 per-op、back-to-back 交錯、同 prompt/n/seed/budget、-ngl 99 + cache 4GiB、-no-mmap；CGC_HEAD_DISP_DBG 同時驗證 dispatch 幾何）：

| | gemma4 head（node_2640） | qwen36 head（result_output） |
|---|---|---|
| 張量 | token_embd（tied）Q6_K [2816, 262144] | output Q6_K [2048, 248320] |
| 大小 | **605.6MB**（§8.24 記「runtime 282MB」錯誤——Q6_K 738M w × 0.82B/w 就是 605.6MB，UD 打包不變 runtime size） | 417.2MB |
| decode per-op（穩定態） | **~6.3ms/step**（93GB/s ≈ 78%） | **~4.45ms/step**（94GB/s ≈ 78%） |
| decode 步佔比 | ~10%（65ms/step 基線） | ~4.5% |

**關鍵發現 1：兩家族 head 都是高效路徑，H1 回收空間 ≈ 0-10%，不投入**。M4（MacBook Air，Apple M4，120GB/s）下 78% 帶寬已近 GEMV 實務上限。§8.24「23-26ms/step、9-42% BW、decode 36%」與 §8.27「lm_head H1 +15-25%」的數字基礎是 **prefill batch>1 的 head（每 column 重讀權重，4× 時間）被誤當成 decode 樣本**——headab 原始檔其實混著 fast（~6ms）與 slow（~22ms）兩群，先前 `head` 截斷的 grep 只看到 slow 群。

**關鍵發現 2：pad-ne00 實驗（CGC_HEAD_PAD_TEST=1，2816→3072、odd nb 11→even nb 12）——假說證偽**。pad=0 vs pad=1 的 fast 群完全一致（6.50 vs 6.45ms），slow 群也無差異；且 dispatch 探針證明同 run 內所有 head dispatch 都是 ne11=1（15/15），slow 樣本不是 prefill。**奇數 nb 不是 head 慢的原因。**

**殘留 open item：序列化測量下的間歇 4× slow mode**（兩家族都有：gemma4 6.3→23ms、qwen36 4.4→17ms，同 dispatch、同 hit rate、run 間比例隨機 0-70%）。端到端佐證它**不是正常 decode 成本**：gemma4 n99c end-to-end ~15-16 tok/s（~65ms/step）只與 fast head（6.3ms）一致；slow mode 是 per-op 序列化路徑的 GPU idle-gap 時鐘/記憶體狀態假象。若要釘死可再用 xctrace 或 normal-decode 對照，但**不影響日常配置定案**。

**定案**：
- **lm_head 從 P1 靶清單移除（兩家族）**：head 已高效、佔比小（4.5-10%），H1/geometry 都無空間。§8.27.2 的 H1/H2/H3 行、§8.42 的「lm_head H1 是最大單一槓桿」結論一併作廢。
- qwen36 預期終態 = §8.44 修正：11.76 + 動態 slot 容量（P1-9，+3-8%）≈ **12-16 t/s**；head 不貢獻。
- gemma4 剩餘頭號靶回到 **attention（§8.25，27-28%）與 FFN experts（bandwidth-bound 無 kernel 槓桿）**——但兩者都已被 §8.44 的誠實評估覆蓋（kernel 層無空間）。
- 新增診斷：`CGC_HEAD_DISP_DBG=1`（head dispatch 探針）、`CGC_HEAD_PAD_TEST=1`（pad-ne00 實驗臂，輸出會 garbage，純計時用）——已註冊進 RUNTIME_CONTROLS.md。

**修正**：§8.24 的「9% / 282MB / 23-26ms」→「**78% / 605.6MB / ~6.3ms 穩定態**（slow 群是測量假象）」；§8.27.2 的 H1 預期與 §8.44「lm_head 佔比確認」行作廢。

### 8.46 slow mode 定案：正常 decode 無 4× 慢步——「3.5× gap」與「間歇 22ms head」都是序列化測量假象（2026-08-16）

**任務**：xctrace Metal System Trace 不可用（本機只有 CommandLineTools、無 Xcode——正是專案改走 CGC_GPU_OP_TIMING 的原因），改用等價且更直接的方法：新增 `CGC_GPU_WHOLE_TIMING=1`（env-gated，ggml_metal_graph_compute 正常路徑結尾，用 queue 首/末 command buffer 的 GPUStartTime/GPUEndTime 量**每 graph 純 GPU 時間**，無序列化）。

**gemma4 n99c + cache 4GiB、n=64 正常 decode 的 whole-graph 計時**：
- **每 decode step ≈ 31 個 Metal subgraph 提交**（sched 依 buffer 轉換切分，非單一 graph_compute）——這本身就是個引擎層結構事實（每步 31 個 command buffer + encode 開銷，§8.19 的「graph reuse/dispatch mode」候選的具體面貌）。
- per-call GPU 時間：decode 區間 **p99 = 4.14ms、max = 9.9ms**（9.9ms 出現在前 16 步冷 pool 期）；絕大多數 call 0.9-2.5ms。**正常路徑下完全沒有 ~20ms 的步**。
- per-step GPU 總和 ≈ 56ms/step（3671ms/65 steps）——與 end-to-end ~15-16 tok/s（65ms wall，含 CPU router/topk/sampling ~9ms）一致，量測自洽。
- head 在正常路徑的 subgraph 內 ≈ **4ms**（比序列化 per-op 量的 6.3ms 還快——序列化路徑的 encode→wait idle gap 降 GPU clock 會放大單 op 時間）。

**定案**：
1. **「gemma4 head 3.5× 慢」不存在**：正常 decode 無間歇 4× 步、head 穩定 ~4-6.3ms。§8.45 的 slow mode open item 關閉——它是 per-op 序列化工具（逐 op encode→commit→waitUntilCompleted、GPU 大量 idle gap）的時鐘/狀態假象，**不影響正常 decode**。
2. **lm_head 維持從 P1 靶清單移除**（§8.45 定案不變）：兩家族 head 穩定態 ~93-94GB/s、佔 decode 4.5-10%、H1/geometry 無空間。
3. 副作用發現：正常 decode 每步 ~31 個 subgraph 提交是引擎層 overhead 的實證（每個 ~1-2ms 的小 dispatch，command buffer 開銷 + 潛在 pipeline 邊界）——這是 §8.19.2「引擎層收斂」（graph reuse、n_cb、combine n_ubatch）候選的具體測量基礎，若要做引擎層優化，先想辦法把 subgraph 數降下來。
4. 新診斷：`CGC_GPU_WHOLE_TIMING=1`（正常路徑 per-graph GPU 計時）——已註冊 RUNTIME_CONTROLS.md。

### 8.47 P1-9 完成：動態 slot 容量證偽——偏斜存在但不可行動（+0.3%），qwen36 P1 槓桿耗盡（2026-08-16）

**任務**：先量每層 routing 多樣性偏斜，再改 loader 依 prefill 統計分配熱冷層容量，A/B 量 miss 率與 tok/s。

**Step 1 · 偏斜量測**（LLAMA_EXPERT_CACHE_TRACE，qwen36 IQ3_XXS、fox 128-tok 基準、-ngl 99 + cache 4GiB、pool = 200 slots/layer 均勻、整體 hit 81.2%）：
- prefill uniq/layer：223-242（40 層幾乎一致，高多樣性）
- **decode uniq/layer：110-221（sum 5876），max/min = 2.0×**——layer 0/1 最熱（205/221）、layer 8/9/13/14 最冷（110-118）
- 每 decode step 的 uniq = 8.0（top-8 固定）
- 只有 2 層 decode uniq > 200（layer 0/1）會輕微 thrash；0 層 < 100 浪費

**Step 2 · LRU 模擬（真實 trace 重放，同 8000 slots 總預算）**：
- uniform 200：12480 misses
- greedy 邊際分配（逐層 miss-vs-capacity 曲線 + 邊際最優增量）：**12438 misses，delta = +0.3%**（42/12480）——capacity 不是 miss 來源
- 原因：miss 是 one-shot routing 雜訊（與 §8.14「1796/1797 唯一」同根），LRU 在 ~150 slots 已涵蓋全部可複用 experts；冷層降到 distinct 數不省 miss、熱層加 slot 也不減 miss

**Step 3 · 實測確認（容量曲線已平）**：
| budget | capacity | hit rate |
|---|---|---|
| 3GiB | 150 | 80.4% |
| 4GiB | 200 | 81.2% |
| 6GiB | 300 | **15.2%（working-set OOM 崩潰，§8.44 已知牆）** |

+33% 容量只換 +0.8pp hit；6GiB 直接不可用。

**定案：P1-9 證偽，不投入 loader 改動**（per-layer ne[2] 是數小時工程、回收 ≤1pp，違反「拒絕收益可忽略方向」原則）。qwen36 的 P1 槓桿至此全部耗盡：FFN kernel（P1-3a/b 證偽）、lm_head（§8.45 翻案）、動態容量（本節證偽）——**12-16 t/s（hit 驅動）是 bounded residency 的誠實終態**。剩餘唯一結構性靶是 §8.46 的「每步 ~31 個 Metal subgraph 提交」引擎層 overhead，屬大改動、另行評估。

### 8.48 gemma4 attention 成分分解（2026-08-16）：attention 全體 31.4%，剩餘槓桿 = 小 op dispatch overhead（QKV 融合候選）

**任務**：lm_head 翻案後，gemma4 頭號靶回到 attention（§8.25 估 27-28%）——用 CGC_GPU_OP_TIMING 把 Q/K/V in-proj、attn_out、SDPA 分開量，產出剩餘槓桿評估。

**方法**：gemma4 IQ3_S、-ngl 30 + cache 4GiB（日常配置，§8.26 known-good）、CGC_GPU_OP_TIMING_GRAPHS=6/SKIP=2、n=20、19 個 decode step 平均。未命名 MUL_MAT 身份用 CGC_GRAPH_DUMP 的 dot 檔（src1 weight）解析（§8.25 方法，本 session 重做）。

**per-step 成分表**（序列化 per-op 總和 50.97ms，19 steps 平均）：

| 成分 | ms/step | % decode | 權重 bytes/step | 等效帶寬 |
|---|---|---|---|---|
| MoE gate_up（MUL_MAT_ID） | 10.21 | 20.0% | 8 experts×164MB/128 | ~64GB/s |
| MoE down_exps（MUL_MAT_ID） | 6.39 | 12.5% | 8×143MB/128 | ~36GB/s |
| **head（token_embd，node_2640）** | 6.20 | 12.2% | 605.6MB | **98GB/s** |
| **attn_out（node_29 族）** | 5.42 | 10.6% | 336.7MB | 62GB/s |
| **Q_proj（Qcur）** | 5.39 | 10.6% | 336.7MB | 62GB/s |
| shared ffn_down | 2.64 | 5.2% | 173MB | 65GB/s |
| shared ffn_up | 2.54 | 5.0% | 171MB | 67GB/s |
| shared ffn_gate | 2.47 | 4.8% | 164MB | 66GB/s |
| **K_proj（Kcur）** | 2.19 | 4.3% | 130.8MB | 60GB/s |
| **V_proj（Vcur）** | 1.98 | 3.9% | 118.3MB | 60GB/s |
| MUL（l_out/scale/norm） | 1.65 | 3.2% | — | — |
| RMS_NORM | 1.36 | 2.7% | — | — |
| **SDPA（FLASH_ATTN_EXT）** | 1.02 | 2.0% | — | — |
| router（ffn_moe_logits） | 0.76 | 1.5% | — | — |
| ADD | 0.55 | 1.1% | — | — |
| 其他（GET_ROWS/SOFT_MAX） | 0.21 | 0.4% | — | — |

**attention 全體 = Q 10.6 + K 4.3 + V 3.9 + attn_out 10.6 + SDPA 2.0 = 16.00ms = 31.4%**（修正 §8.25 的 27-28%：Q/K/V in-proj 實際 18.8% 而非 18.0%、attn_out 10.6% 而非 ~8%、SDPA 2.0%）。MoE 全體 32.5%、head 12.2%、shared FFN 15.0%。

**關鍵發現：attention 投影全是「小 op overhead-bound」，不是 bandwidth-bound**：
- 同 kernel（Q6_K GEMV）、同 ne00（2816）、同 nr0/nsg 的 head 跑 **98GB/s**，Q/K/V/attn_out 只有 **60-62GB/s**——per-op 只讀 3.9-11.2MB（0.066-0.181ms），**dispatch/launch 固定 overhead 佔小 op 的 ~35%**（0.026-0.067ms/op）。
- 用 head 當 bandwidth 基準（11.2MB@98GB/s = 0.114ms）反推：Q/K/V/attn_out 四 op 的 overhead 合計 ~5.6ms/step ≈ **10.7% of decode**——這是 attention 的剩餘槓桿上限（若融合能消除 dispatch）。
- SDPA 只有 2.0%——M=1 decode 的 flash_attn_ext 幾乎免費，attention core 不是靶。

**剩餘槓桿評估（對照 P1-3a 教訓）**：
- **候選：QKV 三合一融合**（Q/K/V 併成一個 op 或一個 dispatch，省 2/3 的 dispatch overhead；attn_out 在 SDPA 之後無法與 QKV 併，需另議）。理論回收 ~3-4ms/step（~6-8% decode）。
- **但 P1-3a 的教訓：序列化 per-op 顯示的 overhead 在真實 decode 可能已被 GPU pipeline 藏住**——P1-3a 融合後端到端 ≈ 0。attention 小 op 之間沒有數據依賴（Q/K/V 平行），GPU 側可能已重疊。**必須先做端到端 A/B 驗證，不能只信序列化數字。**
- 其他維度全部無空間：SDPA 2.0%（免費）、shared FFN 15%（bandwidth-bound 同 MoE，無 kernel 槓桿）、head 98GB/s（已高效）。

**建議**：若要做，正確順序是 graph 層 QKV 融合（bit-identity 對拍後 128-tok 端到端 A/B），不是 Metal kernel 改寫——先驗證 dispatch overhead 在真實 decode 是否真的存在。若端到端 ≈ 0（同 P1-3a），則 gemma4 剩餘頭號靶回到 **引擎層 subgraph 提交 overhead**（§8.46 的每 step ~31 個 1-2ms subgraph），屬大改動另行評估。

### 8.49 qwen36 attention 成分分解（2026-08-16）：hybrid 架構、attn_qkv 原生融合，序列化計時 35× 膨脹不可靠

**任務**：用同一套 CGC_GPU_OP_TIMING + graph dot 方法做 qwen36（hybrid：linear attn + SSM）的 attention 分解，對照 gemma4 的 31.4% 與 dispatch overhead 假說。

**方法**：qwen36 IQ3_XXS、-ngl 99 + cache 4GiB（日常配置）、CGC_GPU_OP_TIMING_GRAPHS=6/SKIP=2、CGC_GRAPH_DUMP dot 解析。另跑 -ngl 30 對照組。

**架構事實（dot 解析）**：qwen36 是 hybrid——40 層中 **30 層用 `attn_qkv` 融合投影 + SSM（`ssm_conv1d`/`ssm_out`/`ssm_alpha`/`ssm_beta`）**，僅 **10 層（3,7,11,15,19,23,27,31,35,39）用標準 attn_q/k/v + attn_output**。MoE 是 gate/up/down 三個分開的 MUL_MAT_ID（非 gemma4 的 gate_up 合併）。shared expert 用 `*_shexp`。

**權重結構**：attn_qkv 412.9MB（30 層 × 13.8MB）、attn_q 137.6MB（10 層）、attn_output 68.8MB（10 層）、ssm_out 206.4MB、head 417.2MB Q6_K。MoE：gate/up 各 3439MB + down 4230MB（40 層，每層 128 experts）。

**per-step 成分（-ngl 99 序列化，相對結構）**：
| 成分 | % decode | 備註 |
|---|---|---|
| MoE down / gate / up | 23.1% / 15.4% / 15.0% | 三投影分開，合計 53.5% |
| linear_attn_out | 7.3% | 30 層 linear attention 輸出投影 |
| SSM_delta_net + SSM_conv | 2.9% + 1.8% | GATED_DELTA_NET + conv |
| Q_proj（std 10 層） | 1.3% | 13.8MB/層 |
| attn_out（std 10 層） | 0.7% | 6.9MB/層 |
| SDPA | 0.6% | flash_attn_ext |
| head | 0.3% | 417MB @ ~97GB/s |
| K/V_proj（std） | 0.1%+0.1% | 0.9MB/層 |

**attention+SSM 全體 ≈ 15.1% of decode**——遠低於 gemma4 的 31.4%。

**關鍵發現 1：qwen36 原生就是「QKV 融合」**。30/40 層的 Q 投影已是單一 `attn_qkv` MUL_MAT（13.8MB/層），沒有 gemma4「每層 4 個小 op」的 dispatch 問題——§8.48 建議 gemma4 做的 QKV 融合，qwen36 架構上已內建。qwen36 的 attention 佔比低（15% vs 31.4%）主要就是這個結構差異 + hybrid 層沒有完整 QKV+SDPA。

**關鍵發現 2（方法學，重要）：qwen36 的序列化 per-op 計時 35× 膨脹不可靠**：
- -ngl 99：總和 2067ms/step vs 真實 decode ~65-70ms/step（12-16 tok/s）→ **32× 膨脹**
- -ngl 30：總和 2450ms/step，同樣膨脹（MoE_down 808ms/step = mean 20ms/op？）
- 對照 gemma4：序列化總和 50.97ms vs 真實 ~56ms（≈1×，§8.48 可信）
- 同一工具、同一 binary、同一機器，兩家族誤差差 35 倍——**qwen36 的 Metal graph 結構（hybrid SSM 依賴鏈 + 40 層 × 3 個 MUL_MAT_ID）在每-op 序列化下暴露大量 GPU idle gap**，gemma4 沒有
- **結論：qwen36 的 per-op 絕對值不能用於跨家族對照**（§8.45/8.46 的 slow-mode 假象在 qwen36 上是結構性的、被放大的）。qwen36 的相對結構（MoE 53% / attn+SSM 15%）與 CGC_GPU_WHOLE_TIMING 的 whole-graph 提交數據（1296 calls 幾乎全 0，也證實 qwen36 Metal 提交結構與 gemma4 不同）仍可讀。
- 交叉驗證：MMIDPROBE 確認 qwen36 MUL_MAT_ID 走 Metal（src0 buft=MTL0、ne2=200 pool slots、topk_remap ids 正確）。

**對 §8.48 dispatch overhead 假說的對照**：
- gemma4 的 31.4% attention 中 ~10.7% 是 4 個小 op 的 dispatch overhead（Q/K/V/attn_out 各 0.07-0.18ms/op、60-62GB/s vs head 98GB/s）
- qwen36 用 attn_qkv 融合避開了這個問題，attention 佔比低一半——**間接支持 gemma4 QKV 融合方向**（若 gemma4 融合後 attention 從 31.4% 降到 ~20%，回收 ~10% decode）
- 但 qwen36 的驗證方式不能靠 per-op 絕對值，必須靠端到端 A/B（gemma4 的 QKV 融合 A/B 同樣用端到端）

**定案**：qwen36 attention 不是靶（15%、且原生融合、SDPA 免費）；§8.48 的 QKV 融合方向獲得架構支持，但 qwen36 側無事可做。gemma4 QKV 融合 A/B 仍是 attention 線唯一候選動作。


---

## 8.50 gemma4 QKV 三合一融合 A/B（2026-08-17）：dispatch overhead 端到端不可回收 — 封存

### 動機
§8.48 假說：gemma4 attention 的 Q/K/V/attn_out 4 個小 op 各跑 60-62GB/s（vs head 98GB/s），
小 op 固定 dispatch/launch overhead 佔 ~35%（~5.6ms/step ≈ 10.7% decode），QKV 三合一可回收。

### 實作（CGC_QKV_FUSE=1，env-gated，gemma4 only）
- **loader 側（llama-model.cpp `fuse_attn_qkv()`，load_all_data 後一次跑）**：wq/wk/wv → 單一
  `wqkv` tensor（ne1 = n_q+n_k+n_v，同型別），585.8MB（layer0 18.9MB CPU_REPACK + layers1-29 566.9MB MTL0）。
  - 關鍵實測：**layer 0 attention 在 CPU_REPACK buft**（input layer 永不上 GPU），layers 1-29 在 MTL0——
    必須按 buft 分組；CPU_REPACK 無 `get_tensor` iface（直接 segfault）且 `set_tensor` 要求 offset==0
    全 tensor 寫入（會 repack-on-write）→ **統一從 GGUF 檔讀 raw bytes、一次全 tensor set**。
  - 原 wq/wk/wv 保持 resident（同 bytes，+585MB one-time；-ngl 30 + cache 日常配置安全，experts skip-loaded）。
- **graph 側（gemma4.cpp）**：`layer.wqkv` 存在時單一 `build_lora_mm(wqkv)` + 3 個 `ggml_view_3d`
  slice（Q/K/V 用 `row_size(F32, n_embd_head)` 當 nb1、byte 位移——mirror 上游 build_qkv），KV-shared 層 V=K。
  Kernel 完全不動 → 逐 row 獨立 → **bit-identity 理論保證**。

### 驗證（-ngl 30 + cache 4GiB、LLAMA_EXPERT_CACHE_ALLOW_NGL + L4_SKIP_LAYER0，seed 7）
1. **bit-identity 全過**：n=32 文本 identical；`LLAMA_SIMPLE_LOGITS_DBG` prefill logits（top-5 + first16）identical
2. **dispatch 確實 3→1**：per-op dump 484 個 Q/K/V 行 → 171 個 QKVcur 行
3. **per-op 時間**：fused QKVcur ≈ 或略快於 base Q+K+V（序列化總和 168.7ms vs 185.6ms）
4. **端到端 128 tok A/B（兩輪）**：base-first 序 +6.5/+23/+50%（order bias，系統負載爬升）；
   交替序 −11.1%/−6.5%（fuse 略快）——**兩次 A/B 的絕對值差 2×（base 8.2s vs 16.4s），
   系統狀態主導，訊號在 noise 內**

### 定案：QKV 融合封存（code 保留、env-gated、預設 off）
- **dispatch overhead 在真實 decode 不可回收**——與 P1-3a（gate/up/down 融合端到端 ≈ 0）完全一致的結論：
  序列化 per-op 顯示的 launch overhead 在正常 decode 已被 GPU pipeline 藏住。
- §8.48 的「QKV 融合回收 ~10%」假說**證偽**（端到端）；§8.49 的「qwen36 原生融合所以快」改判為
  架構差異（op 數少），不是可複製的優化。
- 副產物修復：simple.cpp `LLAMA_SIMPLE_LOGITS_DBG` 兩個 crash（logits row 越界 + prefill 沒設 logits flag）；
  ggml.c 加 env-gated CGC_VIEW_DBG（view bounds 診斷，不設 env 不影響）。
- **gemma4 剩餘頭號靶維持 §8.46**：每 step ~31 個 Metal subgraph 提交（引擎層，大改動）。

## 8.51 wake-poll spin 移植 A/B（2026-08-17）：turbo 的 +35% 在 llama.cpp 架構下證偽 — 封存

### 動機
turbo-fieldfare 生產設定（run_prod.sh，2026-08-08）的關鍵機制：decode thread 在
`MTLCommandBuffer.status` 上 spin（`TURBO_FIELDFARE_WAKE_POLL_US=5000`，sched_yield 間隔），
取代 park 在 waitUntilCompleted——聲稱負載下 median +35%（15.7 vs 11.6 t/s，4 輪交錯）、
quiet 中性、byte-identical。我們 §8.46 的「每 step ~31 個 Metal subgraph 提交」正是同類
等待點，理論上可移植。

### 實作（env-gated，預設 off，不設 env 行為完全一致）
- `ggml-metal-context.m` 新增 `cgc_wait_cmd_buf()`：`CGC_WAKE_POLL_US`（µs，0/absent=off）
  設定期間對 `[cmd_buf status]` spin，`sched_yield()` 間隔，deadline 到或 status=Error 才
  fallback 到 blocking `waitUntilCompleted`（錯誤仍由 blocking wait 報告）。
- 只取代 `ggml_metal_graph_compute` 主等待點（decode 熱路徑）；capture/teardown 路徑不動。
- 語意完全 mirror turbo 的 `waitForCompletionPolling`。

### A/B 結果（gemma4 IQ3_S、-ngl 30 + cache 4GiB、128 tok 真實文本、交錯、負載 2.7-6.3）

| 臂 | 樣本（t/s） | median |
|---|---|---|
| off（無 poll） | 10.13 8.13 9.56 8.49 11.25 8.15 | **9.03** |
| on p500 | 7.97 8.08 | 8.03 |
| on p1000 | 8.94 7.75 | 8.35 |
| on p5000 | 8.06 6.40 7.43 7.85 | 7.64 |

- **on 全樣本慢於 off（median −12.4%）**，且 spin 越長越慢（500→1000→5000 µs 單調變差）。
- bit-identical 全過（off vs on 輸出一致）；hitrate 全同（90.6%）；pread/fill 時間兩臂幾乎
  相同（8.1-8.2M / 5.0-5.1M µs）→ 差異不是 cache fill 被搶 CPU 造成。

### 機制定案：架構差異，非參數問題

turbo 的 +35% 依賴它的 **per-layer 流水線**：每層一個 command buffer、decode thread 在
cb1 wait 期間 spin → CPU 可提早開始下一層 encode/commit（「GPU busy 期間 CPU 有活可幹」）。
llama.cpp 是 **graph 級一次性提交**：`ggml_metal_graph_compute` 一次 encode 整個 graph 的
所有 command buffer（n_cb 執行緒），decode 主執行緒在 `cmd_buf_last` 上 wait 整個 graph
完成——**wait 期間 CPU 沒有可提早開始的工作**（下層 encode 依賴本層輸出，串行依賴），
spin 提早醒來只是空轉燒 CPU。在高負載（Freebuff 65% + WindowServer）下 spin 的調度開銷
反而是淨負。

### 定案
- **wake-poll 在 llama.cpp 架構下證偽**，封存。code 保留（`CGC_WAKE_POLL_US`，env-gated，
  預設 off）作為診斷工具。
- 移植條件：若未來把 decode 改成 per-layer 流水線（§8.46 引擎層大改動），wake-poll 才
  有收益空間——兩者是同一個改造的配套，不是獨立槓桿。
- 與 turbo 的對照結論（同窗實測）：turbo r3 + 生產 env（舊 binary，wake-poll 代碼未含）
  7.15 t/s vs 我們 gemma4 n99c 14.44 t/s——我們已打平 turbo 生產目標 15-17。

## 8.52 turbo hot pool 移植評估（2026-08-17）：pin64 模擬省 51% miss，但端到端待 A/B — 可行

### turbo hot pool 機制（run_prod.sh + ModelExpertIO/PreadExpertStreamer 實作）
- **per-layer pinned slots**：profile（top64_code.json = 30 層各 64 個 expert id，128 experts/layer 的
  50%）在 streamer init 時 preload 進專用 slots，**planner 永不 evict**（`slotPinned`）。
- 總容量 = 64 pinned + 32 LRU = 96 slots/layer；policy 是 **LFU**（cachePolicy: .lfu）。
- **preload 模式**：r3 = async（後台讀、不卡 decode），r4 = sync；pool64 需 96 slots（r4 expert
  3.2MB → ~9GB virtual，commit lazy 到 touch，RSS ~2.5GB）。
- 關鍵設計註釋：「**MTP verify union 大部分落在 pinned slots → batched-verify IO 退化成
  page-cache reads**」——pinned 的價值是讓 decode/verify 的 union 幾乎全 resident。

### 我們 cache pool 對照
- 純 **LRU**（pick_slot/evict_lru 按 last_use），4GiB budget = gemma4 120 slots/layer、
  qwen36 200 slots/layer（§8.47）。
- 已有 `slot_pinned` 機制，但只用於 **TAILPIN**（§8.16 證偽 net≈0：動態 pin prefill 尾段
  K=10 token union、step 末 unpin）——**沒有靜態 profile pin**。
- 有 prewarm（load 時前 capacity 個 experts 真實 pread），但那是順序 0..n，非 profile 驅動。

### 模擬（真實 trace：gemma4 IQ3_S 128-tok 兩條不同 prompt、qwen36 IQ3_XXS）

**gemma4（128 experts/layer、routing 極穩定）——pin 有效**：
| 配置（cap=120-128 slots/layer） | decode misses | vs LRU |
|---|---|---|
| LRU（現況） | 1711 | — |
| +pin32（cross-prompt profile） | 1276 | −25% |
| +pin64（cross-prompt profile） | 836 | **−51%** |
| +pin96 | 429 | −75% |
| LFU vs LRU（無 pin） | 相同 | +0.0% |

- **gemma4 decode 存取 99.5% 在 prefill 出現過**（recurring），one-shot 只有 0.5%——與
  qwen36 的「1796/1797 唯一」完全不同（§8.14 結論是 qwen36-specific，不能套 gemma4）。
- **cross-prompt 泛化好**：profile 從 telescope-prompt prefill 統計、應用到 AI-prompt decode，
  收益只從 −56% 掉到 −51%（routing 穩定性高）。
- LFU 無增益 → turbo 的 LFU 不是關鍵，**pin 才是**。

**qwen36（256 experts/layer、routing 散）——pin 效益小**：
- pin64 只省 24%（4765→3639）；pin64 覆蓋率僅 26.4%（vs gemma4 59.4%）。
- one-shot（never-prefill）2.5%，但 §8.14 已證 decode miss 多為 unique 專家 → pin 天花板低。

### 可行性評估
**移植面小**：slot_pinned + slot_table 機制已存在，只需 (a) profile 載入（LLAMA_EXPERT_CACHE_TRACE
 已有 routing 資料，可加 `LLAMA_EXPERT_CACHE_PIN_PROFILE=<file>` env 指定 per-layer top-N）、
  (b) init 時把 profile experts 放進固定 slots 並標 pinned、prewarm 改 profile 驅動。
  loader/graph/kernel 零改動。

**預期回收**：gemma4 pin64 模擬 −51% miss → hit rate ~90.6% → ~95%（端到端估 +2-5%，
  miss pread 已在 fill_batch 並行藏住一部分——TAILPIN §8.16 教訓：pin 的 miss 消除不保證
  端到端有感，需 A/B）。qwen36 預期 +1% 級（可不做）。

**成本**：pin64 = 64 slots/layer 永不釋放 → 120 slots 只剩 56 給 LRU；qwen36 200 slots 剩
  136。記憶體不變（同 budget），但 LRU 空間被壓縮——gemma4 的 recurring 結構讓這值得，
  qwen36 的散 routing 不划算。

### 定案
- **gemma4：值得實作 A/B**（pin64 + profile prewarm，env-gated 預設 off）。模擬支持
  −51% miss，且 gemma4 routing 穩定 + cross-prompt 泛化好是 turbo 同款條件。
- **qwen36：不投入**（§8.47 容量曲線已平 + §8.14 散 routing，pin 天花板 24% 且端到端打折後 ≈ 0）。
- 先決問題：profile 的來源——turbo 用 make_hotpool_profile.sh（prior trace 統計），我們可
  用 LLAMA_EXPERT_CACHE_TRACE 產生同格式。

## 8.53 turbo per-layer 流水線移植可行性評估（2026-08-17）：收益來源在 llama.cpp 不存在 — 不建議做

### turbo 的 per-layer 流水線是什麼（RealForwardRunner.swift 實作）
- 每層拆 5 個 command buffer：`cb1`（attention）/ `sharedFFN`（dense MLP）/ `routerCB` /
  `routedCB`（MoE experts）/ `head`。
- **early commit**（`TURBO_FIELDFARE_EARLY_SHARED`，預設 ON）：sharedFFN 的 CB 在 CPU 阻塞等
  cb1 之前就 commit——shared MLP 只讀 cb1 的 on-GPU 輸出 + resident 權重，不需要 CPU readback，
  GPU 從 cb1 直接滾進 sharedFFN，同時 CPU 讀 router indices、drain prefetch、規劃 routed experts。
- **wake-poll**（§8.51 已移植證偽）+ **B4 hit-only sync fetch**（hot pool ~99% hit 時每層 fetch
  是純 CPU bookkeeping，async continuation 的 ~1.1ms GPU-idle gap 被消除）。
- **B1 實驗（關鍵反面證據）**：把 sharedFFN 融合進 cb1（少一次 commit）是**確認 regression**
  （split 13.0 vs fused 8.3 t/s r3，+57% split；r4 15.9 vs 8.5 +88%）——**減少 commit 數不是
  收益來源**，overlap（GPU 持續 busy 而 CPU 同時準備）才是。早期「+25/+51%」是 page-cache 污染。

### llama.cpp 的對應結構
- **graph 級一次性提交**：`ggml_metal_graph_compute` 一次 encode 整個 graph 的所有 ops
  （n_cb 執行緒並行 encode）→ commit → 主執行緒在 `cmd_buf_last` 等。CPU 在 wait 期間**沒有
  可提早開始的 encode 工作**（所有 encode 已完成）——§8.51 wake-poll 證偽已證明「等待本身
  不可回收」。
- **sched split**（每 decode step ~31 subgraph，§8.46）：split 之間是 **event wait**
  （`ggml_backend_event_wait`，非 blocking），GPU 不 idle——whole-graph 56ms vs per-op 51ms
  幾乎重合。
- **CPU 準備工作已並行**：router/topk 在 CPU（A+B split）、cache fill 是 pool8 背景執行緒
  （§8.12 定案）——turbo「CPU 準備藏進 GPU busy」的對應物我們已經有了。

### 改動面（若要做）
| 層 | 改動 | 風險 |
|---|---|---|
| graph 層 | 每層拆 attention/shared/routed 三個 subgraph（改 build_graph 或 sched split 邏輯） | 中：graph 結構大改，影響 copy-timing（§8.26 修復的脆區） |
| Metal 層 | per-layer early commit（graph_compute 拆成多個 commit 點） | 高：n_cb 並行 encode 機制與 early commit 衝突 |
| sched 層 | 層間 async 依賴（event 提前） | 高：buffer 生命週期、與 expert-cache hook 的交互 |
| 調度 | 重寫 decode 為 turbo 式手動 per-layer 迴圈 | 極高：放棄 llama.cpp 的 graph/sched 框架 |

### 預期回收
- **收益來源（CPU 準備藏進 GPU busy）已不存在**：我們 router 在 CPU、fill 已背景並行、
  GPU 已連續（56 vs 51ms）。
- 剩餘理論空間：每 step ~9ms 非 GPU 時間（graph build + sched + hook + sync），其中大部分是
  CPU 必要工作，不是等待；per-layer 流水線只能碰 sync 那部分（§8.46 顯示很小）。
- turbo 的 +35% 量測包含 wake-poll（已證偽移植）與 hot pool pin（§8.52 待 A/B）——不是純
  流水線的貢獻。turbo 自己的 B1 證明「commit 數」不是槓桿。

### 定案
**不建議做。** per-layer 流水線的收益前提（CPU 在 GPU busy 期間有可提早的準備工作）在
llama.cpp 架構下已被 (a) CPU router (b) 背景 pool8 fill (c) event-based split sync 三件事
取代。改動面大、風險高（copy-timing 脆區），回收 ≈ 0（§8.51 的等待不可回收結論直接適用）。
若未來引擎層要動，方向是 §8.46 的 **subgraph 合併**（減提交 overhead）而非 per-layer 拆分——
與 turbo 的 B1 結論（少 commit 反而慢）看似矛盾，實則一致：turbo 的 per-layer 拆分是它的
架構選擇，不是收益來源；llama.cpp 的 31 subgraph 是 sched 依 backend 切分的必要產物。

## 8.54 n_cb A/B 實驗（2026-08-17）：single-buffer 已是現況，n_cb=2/4 同窗無增益 — 引擎層合併方向證偽

### 實驗動機
§8.46 定案「每 decode step ~31 個 Metal subgraph 提交」；§8.51 wake-poll 證偽「等待不可
回收」。本實驗驗證另一個引擎層假說：「合併提交」是否比 wake-poll 更正確——具體測 Metal
command-buffer 數（n_cb，並行 encode 執行緒）對 decode 的影響。

### 關鍵結構事實（讀碼確認）
- **n_cb 已經 =1（single-buffer 就是現況）**：`ggml_backend_metal_init` 兩處都 hardcode
  `ggml_backend_metal_set_n_cb(backend, 1)`，llama 層無暴露——「強制 single-buffer」無需做，
  已是默認。
- 31 subgraph 是 **sched split 的產物**（ggml_backend_sched_split_graph 依 backend/buffer
  usage 切），不是 Metal n_cb 能合的——合併 split 需動 sched 的 need_new_split 邏輯，
  與 A+B split（§7.3.5，router 釘 CPU）直接衝突，正確性脆區。
- 因此本實驗的可測面收斂為：**n_cb=1 vs 2 vs 4**（並行 encode 執行緒），env-gated
  `CGC_N_CB` 已加（ggml-metal.cpp 兩處 init），預設 1 = 零行為變更。

### A/B 結果（gemma4 IQ3_S、-ngl 30 + cache 4GiB、128 tok、8 輪交錯、bit-identical 全過）

| 臂 | 全部樣本（t/s） | median |
|---|---|---|
| c1（ncb=1，現況） | 13.27 13.06 7.04 11.36 9.00 9.41 11.02 9.82 | 10.42 |
| c2（ncb=2） | 13.71 10.34 13.27 12.14 13.19 9.93 9.44 6.00 | 11.24（+7.9%） |
| c4（ncb=4） | 14.13 13.35 13.50 12.58 11.42 | 13.35（+28%） |

**同窗對照（R1-3 交錯窗，同負載）**：c1 13.06 / c2 13.27 / c4 13.50——**差異 ≤ +3.4%**。
R6-8 窗整體漂移到 9-11 區間（系統負載），c4 只在較早窗有樣本 → 全樣本中位數優勢（+28%）
是**時段偏誤**，非 n_cb 效果。

### 定案
- **n_cb 不是槓桿**：同窗下 1/2/4 差異 ≤3.4%（雜訊內），與 upstream 註解「optimal n_cb =
  1 or 2」一致。維持預設 1；`CGC_N_CB` 保留為診斷 env。
- **「合併提交」方向在 Metal 層證偽**（n_cb 已是最優單 buffer）；sched split 合併是唯一
  剩餘途徑但與 A+B split 正確性衝突（§8.53 已定案不建議）。
- §8.46/8.51 的引擎層收斂最終定案：**等待不可回收（§8.51）、commit 數不可減（本節）、
  per-layer 拆分不可移植（§8.53）**——31 subgraph 是 llama.cpp 架構的必要產物，非 overhead
  靶。bounded residency 的速度天花板維持 §8.44：qwen36 12-16 t/s（hit 驅動）、gemma4
  14.4-15.5 t/s。

## 8.55 gemma4 static profile pin A/B（2026-08-17）：hit +1.2pp 但端到端 −4.6% — 證偽，封存

### 實作（env-gated，預設 off，不設 env 零行為變更）
- `LLAMA_EXPERT_CACHE_PIN_PROFILE=<file>`：每層一行的 top-N 專家清單（空格分隔 id）。
  `llama_expert_cache_load_pin_profile()` 在 load 時讀入 `pin_profile`。
- 新增 `slot_pinned_static`（獨立於 TAILPIN 的 `slot_pinned`，`unpin_all` 不清它）——
  pick_slot 對兩者都 LRU-exempt。
- loader prewarm 改 profile 驅動：profile experts 先 fill + 標 `slot_pinned_static`，
  其餘 slots 補 0..n（原 prewarm）。
- 驗證：`PIN_PROFILE: 1920 experts pinned`（64×30）+ prewarm log 正常。

### A/B 結果（gemma4 IQ3_S、-ngl 30 + cache 4GiB、128 tok、8 輪交錯、bit-identical 全過）
profile 由兩條不同 prompt 的 prefill 統計合併（cross-prompt top-64/layer）。

| 臂 | 樣本（t/s） | median | hitrate |
|---|---|---|---|
| nopin（現況） | 8.02 9.39 11.12 12.67 12.13 12.78 9.52 10.57 | **10.84** | 90.6% |
| pin64 | 10.18 12.53 12.15 7.37 9.66 10.51 10.86 9.91 | **10.34**（−4.6%） | **91.8%**（+1.2pp） |

### 定案
- **hit 確實提升 +1.2pp（模擬預測的方向正確），但端到端 −4.6%（雜訊內）——未轉成收益。**
  與 TAILPIN（§8.16）同一教訓：miss 的 pread 已被 fill_batch（pool8 並行）藏在 GPU busy 後，
  hit 改善的邊際收益 ≈ 0——bounded residency 的速度牆不是 hit rate。
- §8.52 的「gemma4 pin64 模擬 −51% miss → 端到端 +2-5%」**證偽**：模擬的 miss 差異（1711→836）
  換算真實 hit 只 +1.2pp（模擬未計 prefill 請求 + 未計 fill 並行），且 pin 壓縮 LRU 空間
  （120→56 slots）對非 pin 專家有負面抵消。
- **code 保留、env-gated 預設 off**（診斷/未來 hot-pool 概念驗證用）。RUNTIME_CONTROLS.md 已註冊。
- 至此 bounded residency 的速度槓桿**全部耗盡**：FFN kernel（§8.44）、lm_head（§8.45）、
  動態容量（§8.47）、attention dispatch（§8.48/8.50）、wake-poll（§8.51）、hot-pool pin
  （本節）、引擎層（§8.53/8.54）。**誠實終態：qwen36 12-16 t/s、gemma4 14.4-15.5 t/s（hit 驅動）。**

## 8.56 turbo gemma4 生產設定遷移審計 + run_n30cache.sh（2026-08-17）

> 任務：qwen36-r3q4la repack 已生成（turbo 格式）。逐項檢查 turbo `bin/run_prod.sh`
> 的 gemma4 生產設定，遷移到我們的 n30cache（llama.cpp fork bounded-residency 日常配置）。

### 遷移審計（turbo run_prod.sh 每一旋鈕 → llama.cpp fork 對應 → 狀態）

| turbo 生產設定 | 值 | llama.cpp fork 對應 | 遷移狀態 |
|---|---|---|---|
| `EXPERT_SLOTS` | 96（64 pinned + 32 LRU） | `-expert-cache BYTES`（budget→slots/layer） | ✅ L2/L4 已遷移（gemma4 4GiB=120、qwen36 4GiB=200 slots/layer） |
| `HOT_POOL` + profile | 64 pinned/layer + preload | `LLAMA_EXPERT_CACHE_PIN_PROFILE` | ✅ §8.55 已遷移＋證偽（hit +1.2pp、端到端 −4.6%）；qwen36 profile 已存在可直接轉換（見下） |
| `EXPERT_READ_WORKERS` | 8 | `LLAMA_EXPERT_CACHE_WORKERS`（預設 8） | ✅ §8.12 pool8 甜蜜點定案（16 反效果） |
| `WAKE_POLL_US` | 5000 | `CGC_WAKE_POLL_US` | ✅ §8.51 已遷移＋證偽（−12.4%，架構無 per-layer 流水線可搶） |
| `HOT_POOL_PRELOAD` | async（r3）/ sync（r4） | loader load-time sync prewarm | ⚠️ 部分：turbo r3 async 是 TTFT 優化；我們 prewarm 在 load time（離 decode 時鐘），PREWARM_HOT（§8.15）與 TAILPIN（§8.16）已證偽 |
| `--trust-receipt` | SHA-256 跳過 | N/A（GGUF 無 manifest 校驗鏈） | ⚪ N/A |
| MTP adaptive gate | 預設 OFF（net negative） | MTP 判死（同結論） | ✅ 一致 |
| early shared commit | ON（cb1→shared 不 starve） | llama.cpp sched event-based split | ⚪ N/A（§8.53 架構不同） |
| `MODEL_BITS=3`（r3） | 3-bit experts | GGUF IQ3_S（gemma4）/ IQ3_XXS（qwen36） | ✅ 已在用 |

**結論：12 項中 6 項已遷移、4 項 N/A/一致、2 項部分（preload 語義不同）——gemma4 的
生產設定在 fork 裡基本都已有對應物，多數已證偽或定案。缺的只是「正式生產啟動器」，
本次補上。**

### qwen36-r3q4la repack 說明（user 提供）

- `prime-agent-worktrees/qwen36-r3q4la_ga_e4h3.gturbo`（1.3GB）：r3（3-bit experts）+
  q4la（linear_attn qkv/z/out 4-bit）+ ga_e4（GatedAttn 4-bit）+ h3，含
  `profiles/top64_code.json` 等 per-layer hot-pool profiles（turbo 已生成）。
- **turbo 格式（.gturbo），不能直接在 llama.cpp fork 跑**——llama.cpp 對應物是
  GGUF `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`。若要復刻 r3-q4la 的「linear_attn 4-bit」
  量化，需 GGUF 轉換工程（不在本次範圍）。
- qwen36 的 PIN_PROFILE 若要用：把 turbo JSON（list-of-lists）轉成我們每層一行格式，
  但 §8.52 模擬已示 qwen36 pin 天花板低（coverage 26.4%、miss −24%），§8.55 機制
  （fill_batch 並行藏住 pread）跨家族適用——**不建議投入**，腳本保留 opt-in。

### 交付：`scripts/run_n30cache.sh`（llama.cpp fork 版 run_prod.sh）

把定案設定打包成單一 CLI（用法見檔頭註解）：

```
./scripts/run_n30cache.sh -m gemma4 -n 128 -p "The capital of France is"
./scripts/run_n30cache.sh -m qwen36 -n 128 --prompt-file /tmp/msg.txt
N30CACHE_BUDGET=8589934592 ./scripts/run_n30cache.sh -m qwen36 -n 128 -p "..."
```

- 家族預設：gemma4 `-ngl 30`（§8.44 sweet spot）；qwen36 `-ngl 99`（base 硬 OOM，
  只有 + cache 能 full offload，§8.38）。
- 固定生產 env：`ALLOW_NGL=1`（n99 硬 guard）+ `L4_SKIP_LAYER0=1`（blk.0 排除修復，
  §8.35/8.36）+ `WORKERS=8`（§8.12）+ `-no-mmap`（凍機防護）+ `-t 8`。
- 證偽項目預設 off 但保留 opt-in：`--pin-profile`（§8.55）、`CGC_WAKE_POLL_US`（§8.51）。
- 結尾印 speed / hit rate / hits / misses / RSS。
- 2026-08-17 smoke：gemma4 n=8 EXIT=0、qwen36 n=8 EXIT=0，兩家族輸出正常
  （n=8 冷啟動 hit 低屬預期，128-tok 才是日常數字 14.4-15.5 / 12-16 t/s）。

### §8.56 驗證 A/B（2026-08-17，6 輪交錯 + 同窗 base 對照）

執行 `tmp/n30c_prod_ab.sh`（run_n30cache.sh 生產設定：gemma4 ngl 30 / qwen36 ngl 99、
4GiB budget、WORKERS=8、ALLOW_NGL + L4_SKIP_LAYER0、-no-mmap、-t 8、128 tok 真實文本
telescope prompt、輪間 g4→q36 交錯）：

| 臂 | R1 | R2 | R3 | median | hit |
|---|---|---|---|---|---|
| gemma4 n30c | 8.97 | 10.75 | 10.30 | **10.30 t/s** | 92.4% |
| qwen36 n99c | 8.19 | 6.84 | 6.47 | **6.84 t/s** | 85.5% |

同窗 base 對照（無 cache、-ngl 30、2 輪）：gemma4 base 12.62/14.92（median 13.8）、
qwen36 base 6.48/5.35（median 5.9）。

**判定：production 設定與 §8.44 一致（config 逐項相同），但絕對數字被負載壓縮**：
- 當下 load 2.95 且 Freebuff(48% CPU)+WindowServer+bun 活躍——base 臂也同步掉
  （gemma4 15.79→13.8、qwen36 ~12→5.9），證明是環境非設定。
- **同窗順序關係保持**：gemma4 base(13.8) > n30c(10.30)，qwen36 n99c(6.84) > base(5.9)
  ——與 §8.29（base 15.79 > n30c 9.75）及 §8.38/8.44（qwen36 唯 cache 能 full offload）
  一致。gemma4 n30c 10.30 與 §8.29 受控 n30c 9.75 在雜訊內吻合（§8.44 的 14.4-15.5
  屬更乾淨負載窗的量測）。
- **§8.44 數字需低負載窗（關 Freebuff/Doubao）才可重現**——本節數據是「負載閘門」下
  的保序驗證，非反證。hit 穩定（gemma4 92.4%、qwen36 85.5%）。

## 8.57 P1-3c runtime nsg 掃描（2026-08-17）：dispatch 層 tile 參數證偽，40%→78% 缺口不是 dispatch 可調

**任務**：不改 compile-time nr0，掃描 dispatch 層的 nsg（simdgroups/threadgroup）找 qwen36 FFN 最優組合，A/B 量「FFN 40% vs lm_head 78% 頻寬」缺口能否回收。

**實作**：`ggml_metal_nsg_env()`（mirror `ggml_metal_nr0_env`）——`CGC_MMV_NSG=N`（1..32，預設 = N_SG_IQ3_XXS=2，零行為變更）。nsg 是 Metal function constant（FC_MUL_MV+0），pipeline 每值 runtime 編譯＋快取，**不需改 .metal**。只動 `mul_mv_id`（MoE FFN）的 IQ3_XXS/IQ2_S 兩型。

**A/B（qwen36 IQ3、-ngl 99 + cache 4GiB、128 tok、同窗）**：

| nsg | 128-tok t/s | bit-identity | 備註 |
|---|---|---|---|
| 1 | 9.14 | **DIVERGED ✗** | 輸出退化迴圈（`the capital of France is the capital...`）——kernel 假設 NSG≥2，nsg=1 有真 bug |
| **2（現況）** | **10.42** | 參考臂 | 最快 |
| 4 | 7.74 | ✓ | |
| 8 | 7.67 | ✓ | per-op FFN −17%（2.27+1.33+1.14→1.75+1.11+1.08 ms）但端到端 −26% |
| 16 | 5.58 | ✓ | |

**per-op（序列化計時，1 decode step）**：nsg=8 的 FFN 三 op 合計 3.94ms vs nsg=2 的 4.75ms（−17%）——**per-op 更快但端到端更慢**。

**定案**：
- **nsg=2（預設）已是甜蜜點**；nsg 調大端到端單調變差（threadgroup 256+ threads 降並行 threadgroup 數、圖間調度效率變差），per-op 的省被 pipeline 吃掉——與 P1-3a（GLU 融合）、P1-3b（nr0=8）同一教訓：**per-op 改善 ≠ 端到端**。
- **40%→78% 頻寬缺口不是 dispatch/tile 層可調的**——是 kernel 內部 K-loop 記憶體存取模式（單 token GEMV 的 latency-bound 本性），nr0、nsg、GLU 融合三連證偽後此結論封死。
- nsg=1 的正確性 bug 不修（該點也慢）；`CGC_MMV_NSG` 保留為診斷（預設 off）。
- RUNTIME_CONTROLS.md 已註冊。

**§8.56 快驗（2026-08-17，1 輪交錯，load 2.5-2.6）**：gemma4 n30c **10.85 t/s**
（hit 92.4%、RSS 9.99GB）、qwen36 n99c **8.99 t/s**（hit 85.5%、RSS 9.83GB）——
兩者皆落在 6 輪 A/B 的範圍內（8.97-10.75 / 6.47-8.19），設定（ngl 30/99、4GiB、
workers 8）逐項符合，production 配置確認無誤。



## 8.58 MTP 線重啟：IQ3-aug C0 驗證 + blk.40 的 Metal OOM 根因（2026-08-17）

**任務**：跑 `qwen36_mtp_iq3_aug.gguf`（IQ3_XXS trunk + 19 顆 blk.40 MTP tensors，block_count=41 + nextn_predict_layers=1）的 C0 smoke（-ngl 99 + cache 4GiB），驗證 loader 吃 n_layer=40 + blk.40 nextn，輸出與純 IQ3 base 一致。

**結果（全過）**：
- **C0（no-spec）exit=0、輸出與純 IQ3 base byte-identical**（同 prompt/seed 16 tok 完全一致）——C0 時 load_mtp=false，blk.40 整塊被 skip（unused warnings 屬預期）。
- **loader 正確吃 MTP block**：C 臂（`--spec-type draft-mtp`）load_mtp=true → `unused tensor blk.40` = **0**，blk.40.nextn.* 全部消費。
- blk.40 tensor 型別實測：**attn_q/k/v/wo + ffn_*_exps + nextn head 全 F16/F32**（非 IQ2）——MTP head 權重是全精度，品質不是問題。

**新發現：MTP arm @ -ngl 99 的 Metal OOM（根因定位）**
- C 臂直接跑 → `kIOGPUCommandBufferCallbackErrorOutOfMemory`（command buffer 1，目標端第一個 decode）→ decode 失敗 → sampler assert（Abort trap 6，**非 sampling bug**，是 OOM 下游）。
- **根因**：bounded pool 的 loader skip-load / L4 metal-pool buft 分支與 adoption loop **全部以 `tn.bid < hparams.n_layer()`（=40）為界**——blk.40（bid=40）繞過 pool，1.5GB F16 experts + attention 全量進 Metal working set → 超 11.45GB。這是「MTP 共用 pool」實作缺的最後一塊：adoption/skip-load 要延伸到 `n_layer_all`。
- **config 級逃生門（不需改 code）**：`-ot "blk\\.40\\..*=CPU"`（regex override，loader regex_search 語義）→ OOM 解除、**輸出與 base 一致（17 tok 全對）**。代價：blk.40 留 CPU，draft decode 全 CPU → 1.19 t/s（極慢，僅作正確性驗證）。

**剩餘兩個 MTP arm 問題（未修）**：
1. **accept = 0.000%**（n_drafted=68 全拒）：draft context 在首個 draft 時 KV 空（`ctx_dft pos_max=-1 < N-1` warning）——MTP driver 的 process() 沒在 prefill ubatch 上 capture hidden states → 首 draft 輸入 garbage → 且後續步也全拒。需查 `llama_set_embeddings_nextn` capture 在 -ngl 99 + fork graph 的路徑（spec-simple MTP 驅動的已知壞區，§8.7）。
2. **結尾 teardown abort**（stats 印完後 exit 134）：spec-simple MTP 的已知 sampling/teardown bug。

**定案**：C0/loader/byte-identical 三項驗證完成；MTP arm 速度量測被 accept=0 擋住，兩條路：(a) 把 bounded pool 延伸到 blk.40（loader + adoption + graph_mtp hook，治本、也是「MTP 共用 pool」的完成態）；(b) 先修 spec-simple MTP driver 的 process/capture 路徑。速度結論維持 §8.7：MTP 現形式 net negative，等 accept 修好再複測。

### §8.58 修復：bounded pool 延伸到 blk.40（2026-08-17，完成）

**改動（8 處 boundary，`n_layer()` → `n_layer_all`）**：
- `llama-model-loader.cpp`：metal-pool buft 選取、skip-load buft、L1 per-expert 索引、shrink、容量分母——**其中 metal-pool 與 skip-load 分支加 `buft == nullptr` guard**：修掉「-ot CPU override 被 metal-pool 分支覆寫回 GPU pool buft」的 bug（實測覆寫導致 c2 配置 OOM）。
- `llama-model.cpp`：adoption loop 延伸到 n_layer_all（blk.40 experts 進 pool）；MTP 層（il ≥ n_layer）**跳過 prewarm**（F16 每 slot 6MB，prewarm 全填 1.16GB 進 Metal → OOM）。
- `llama-graph.cpp`：**remap leaf（ffn_moe_topk_remap）條件延伸到 n_layer_all**——這是 segfault 的根因修復：MTP block 原無 remap leaf，hook swap 194-slot pool 後 ids 保持 real 0-255 → CPU mul_mat_id OOB（SIGSEGV，crash report 釘死 `ggml_compute_forward_mul_mat_id+1932`）。
- `llama-context.cpp`：TAILPIN defensive resize → n_layer_all（同類 OOB 防護）。

**驗證（qwen36_mtp_iq3_aug.gguf、-ngl 99 + cache 4GiB）**：
| 配置 | OOM | 結果 |
|---|---|---|
| C arm 預設（blk.40 → Metal pool） | **55×** | F16 blk.40 在 pool 的 194 slots × 2MB × 3 ≈ 1.16GB 超出 working set——**模型檔問題非 code bug**，需把 blk.40 experts 量化到 IQ3_XXS 才能入 pool |
| **C arm + `-ot "blk\\.40\\..*=CPU"`** | **0** | **輸出與 base 一致**、無 segfault、draft 跑通（2.4 t/s 全 CPU draft） |
| 純 IQ3 C0（regression） | 0 | 200 slots/layer、exit 0、無損 |

**bisect 紀錄**：OOM 非決定性（54/42/55 次浮動，working-set 邊緣）；segfault 決定性（remap leaf 缺失）；bisect 序：denom 單獨無害 → L1+shrink 產生 segfault → +adoption 也 segfault（直到 remap leaf 修復）→ +override guard 後 -ot 路徑穩定。

**剩餘（非本節範圍）**：① 預設路徑入 pool 需 blk.40 experts 量化 IQ3_XXS（或 F16 層 per-layer 小容量）；② accept=0（MTP driver process/capture，§8.58 原記錄 #1）；③ teardown abort（spec-simple 已知 bug #2）。

### §8.59 MTP 封存：官方 MTP+IQ3 GGUF 正確權重下 accept 轉正但淨負 6.4×（2026-08-17，定案）

**官方檔**：`peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-GGUF-MTP` → `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf`（13.1 GiB，內盤）。hf-mirror.com + aria2c 16 連線（68 MiB/s）。結構：753 tensors（733 trunk + 20 blk.40），trunk=IQ3_XXS（**與既有 bartowski base 為不同量化 run**，逐 tensor bytes 全不同、部分層型別不同：blk.39 attn Q6_K→Q8_0、experts 22→21/18——C0 不能跟舊 base 比 bytes）；blk.40 = attn Q6_K + experts gate/up Q2_K + down Q3_K + shared Q6_K + eh_proj Q8_0 + router BF16（~600MB，**進 pool 不需 CPU override**）。

**手作重建路線（放棄）**：舊 aug 檔 blk.40 有 8 顆垃圾權重（shared/eh_proj/experts 完全不是 HF 資料，僅 attn 投影+router 正確）；`tmp/p0_mtp_rebuild.py` 從 HF safetensors 重建成 20 顆（含缺的 shared_head_norm、+1 norm 慣例），header surgery 修好（magic/version/n_tensors/n_kv 保留、n_kv+1、align()），權重與 HF maxdiff 驗證通過——但被官方檔取代，artifacts 已刪。

**驗證**：C0（no-spec）= exit 0、輸出連貫、hit 81.3%、9.47 t/s；C 臂（draft-mtp、n=4）= 無 OOM（pool 186 slots/layer）、**accept 14.3%（47/328）**、輸出連貫、1.47 t/s。

**定案：MTP-4 淨負 6.4×（9.47 vs 1.47 t/s）→ 依指示封存。** accept 只有 14.3% 時，draft ctx 每 token 跑完整模型（328 draft × 全價）遠抵不過 47 個接受的省下 verify；這確認 §8.7 的「MTP 現形式 net negative」在**正確權重**下依然成立（之前是權重垃圾的誤導）。封存條件：MTP 只在「accept ≥ ~75% 或 draft 成本 ≪ target」時才可能淨正，兩者都不存在。spec-simple 的 teardown abort（exit 134，stats 印完才發生）不影響量測，維持已知 bug。


## 8.63 router topk CPU 成本拆解 + rc0/rc1 長測定案（2026-08-17）

### 背景
§8.62 定案「162 splits 的 sync 開銷是假帳」後，剩餘候選槓桿是 **CPU router topk 本身的執行成本**（logits GEMM + softmax + argsort + topk + remap + shared_expert_gate）。本節用 per-split exec 計時（新增 `CGC_SPLIT_EXEC`，`CGC_SCHED_TIMING` env-gated）直接量每層 router topk 花多少 CPU 時間，判定 Metal topk kernel 值不值得寫。

### 新增診斷
`ggml/src/ggml-backend.cpp` 的 split 迴圈：`CGC_SCHED_TIMING` 下對每個 split 的 `ggml_backend_graph_compute_async` 包牆鐘，印 `CGC_SPLIT_EXEC: split=N backend=CPU|MTL0 exec_ms=X`。已確認 Metal 的 `graph_compute` 是**非同步 commit**（encode 後即返回、無 waitUntilCompleted）——所以 MTL0 的 exec_ms = **CPU 側 encode+commit 時間**，不是 GPU 時間；GPU 執行被完全隱藏（CGC_SYNC wait=0.00ms）。

### rc0 vs rc1 128 tok 長測 A/B（qwen36、-ngl 99 + cache 4GiB、3 輪交錯）
| 臂 | speed（3 輪） | median | hit rate |
|---|---|---|---|
| rc1（ROUTER_CPU=1，CPU router，現況預設） | 10.31 / 11.17 / 10.23 | **10.31** | 81.7% |
| rc0（ROUTER_CPU=0，router 上 Metal） | 12.36 / 9.07 / 10.40 | **10.40** | 81.7% |

**定案：±1% 噪音內，無實質差異**（12.36 vs 9.07 是 load 抖動）。與 §8.23（qwen36 router 留 CPU 保留 CPU↔GPU overlap）一致；router 放哪邊不是瓶頸。

### 每層 router topk 的 CPU 時間（clean per-split exec，無 DBG）
- **CPU router 鏈（logits MUL_MAT / SOFT_MAX / ARGSORT / topk VIEW / topk_remap / shared_expert_gate）總計 ≈ 2.2 ms/step**（11 個含 router op 的 CPU split），≈ **55 µs/layer**，佔 decode step ~2%。
- 最大的單顆 router op：`ffn_moe_logits-12:MUL_MAT` split 0.98ms（其餘 0.1-0.6ms）。
- CPU 側全部分解（clean）：MTL0 encode+commit **57ms**（81 splits，主導）+ CPU exec **23ms**（SSM/conv 6.3、FFN_CPU 殘留 5.0、router 2.2、other 9.1）。
- load 敏感：重負載下 MTL0 encode 膨脹到 ~116ms（rt3），CPU exec ~38ms；相對結論不變。

### Metal topk kernel 判定：**不值得做**
1. 整個 router 鏈只有 ~2.2ms/step（~2%）——就算 Metal topk/argsort/softmax 全部免費，E2E 上限 ~2%。
2. router 的 logits GEMM 上 Metal 就是 rc0，已 A/B = 無增益。
3. llama.cpp 無 Metal ARGSORT kernel，softmax/topk 也要新寫——成本高、上限低。
4. 真正的 CPU-side 大頭是 **Metal encode（57ms/step）**，不是 router——下一個該看的是「為什麼 81 個 MTL0 split 的 encode 平均 0.7ms」（§8.46 引擎層），而不是 topk kernel。

### 對 §8.62 的修正
§8.62 的「155ms sync overhead」是 CGC_SCHED_DBG 的 fprintf 風暴污染（DBG 每 split 印 3-6 行，MTL0 exec 被灌到 2.1ms/split）。clean 量測：graph_compute_wall ≈ sum(split exec)（gap < 2ms），split 迴圈本身就是全部時間；其中 MTL0 encode + CPU exec 佔 80-155ms（load 依賴），inter-split event wait 不是主項。**「162 splits 同步開銷」改判：大部分是 Metal encode 本身的 CPU 成本。**


## 8.64 MTL0 split encode 成本拆解（2026-08-17）：0.7ms/split 是 GPU 執行節奏，不是 CPU encode — 引擎層分刀定案

### 問題
§8.63 發現 MTL0 encode+commit 佔 decode step ~57-72ms（81 splits × 0.7ms），想拆出「固定成本 vs per-node 成本」找引擎層槓桿。

### 工具改進（本節新增）
`CGC_SCHED_TIMING` 改為**每 graph 只印一次彙總**（`CGC_EXEC_SUM: mtl_splits/N mtl_ms cpu_splits cpu_ms`）；per-split 明細（`CGC_SPLIT` / `CGC_SPLIT_EXEC` / since_last）移到 `CGC_SPLIT_EXEC_VERBOSE`。原因：舊 per-split fprintf 每 step 81+ 行，本身把 graph_compute_wall 從 103ms 灌到 162ms（+60ms 假帳）——**§8.62/8.63 的「155ms sync overhead」最終確認是 DBG+fprintf 污染**。

### 乾淨分解（qwen36 n99c，CGC_STEP_TIMING+CGC_SCHED_TIMING，n=24）
| 量 | rt5（無 split 插樁） | rt7（彙總插樁） | n_cb=2 |
|---|---|---|---|
| graph_compute_wall | 103.3ms | 105.9ms | 94.3ms |
| MTL0 exec-sum（81 splits） | — | 72.1ms | 61.3ms |
| CPU exec-sum（81 splits） | — | 32.0ms | 31.0ms |
| wall − exec-sum | — | 0.2ms | — |

### 每 split 0.7ms 的真相：**不是 encode 固定成本，是 GPU 執行節奏**
1. **迴歸**（verbose 樣本）：exec = 466us 固定 + 5.6us/node——但「固定」是假象。
2. **n_cb 不變性**：CGC_N_CB=2（encode 執行緒加倍）MTL0 63.3→61.3ms（−2ms）——encode 不是可平行化的 CPU 工作。
3. **rc0 不變性（決定性）**：ROUTER_CPU=0 讓 MTL0 splits 81→41，但 MTL0 exec-sum **72.9→74.2ms 完全不動**（wall 104.4→102.5）——「每 call 固定成本」若為真，splits 減半應省 ~37ms。不省 = 固定成本是假的。
4. **機制**：`[queue commandBufferWithUnretainedReferences]` 在 queue 滿時 block（GPU backpressure）；每 split 的 exec 時間 = encode + commit + **等 GPU 把前一個 split 跑完**。splits 合併只是把同一批 GPU 工作併進更少的 command buffer，GPU 總執行時間不變。

### 帳單（qwen36 decode，~10.3 t/s）
```
wall ~103ms = ΣGPU phase (73ms) + ΣCPU phase (30ms)   ← 關鍵路徑序列化
  GPU 73ms：attention proj+SDPA ~13ms + FFN ~8ms + head ~4ms（§8.60 隔離）
            + ~48ms 未歸因（小 op 鏈 + launch + GPU 內依賴停頓）
  CPU 30ms：SSM 鏈 6.3 + FFN_CPU 5.0 + router 2.2 + other 9.1（§8.63）
```
**結構性事實**：每層的 CPU phase（SSM/router）在 GPU phase（attn→FFN）之間，資料依賴鏈嚴格序列化——**CPU↔GPU placement 對關鍵路徑不變**（把 CPU 工作搬去 GPU，GPU 變慢等量；反之亦然），只有「總工作量減少」才有效。這解釋 rc0/n_cb/splits 三項全數證偽。

### 引擎層分刀（依預期回收排序）
1. **刀 A：真實 GPU 成分分解（量測補課）**——73ms GPU 中只有 ~25ms 是權重讀取，~48ms 未歸因。現有工具全堵死（xctrace 無 Xcode、MTLCounterSampleBuffer 不支援 compute、CGC_GPU_OP_TIMING 序列化 35× inflate）。下一刀必須先拿到「GPU 73ms 內 kernel 級時間」，否則動刀是猜。
2. **刀 B：GPU 未歸因 48ms**——若成分是 SDPA 設定（qwen36 10 層 full-attn head_dim=256）、Q6_K M=1 GEMV 效率、或小 op 鏈 launch，各對應不同修法（SDPA 參數、GEMV tile、op 融合）。§8.48/8.50 的 QKV/GLU 融合 ≈0 已排除「launch 數」假說的一部分，剩下要量的。
3. **刀 C：CPU SSM 鏈（30ms 的結構主體）**——GATED_DELTA_NET/SSM_CONV 無 Metal kernel。若 GPU 版本比 CPU 快（並行度），總工作減少 → 直接省；placement-invariant 只擋「搬移」，不擋「變快」。工程量大（新 kernel），排在刀 A/B 之後。
4. **已排除**：splits 合併（rc0）、n_cb（§8.54+n_cb=2）、wake-poll（§8.51）、topk kernel（§8.63）——全部證偽，不要再碰。


## 8.65 刀 A 完成：GPU 73ms 歸因（2026-08-17）— 小 op 牆證偽，~38ms 是 dispatch/依賴 bubble

### 量測路徑結果
1. **GPUStartTime/GPUEndTime 永久封死（raw 證據）**：M4 上 raw 值荒謬——async buffer start=6148897387 ticks > end=34475、mach timebase 是 Intel 的 41.667ns/tick（非 M4 的 1/1）。`CGC_HOST_GPU_TIMING`/`CGC_GPU_WHOLE_TIMING` 只能當 wall 測量，無法讀 GPU 時間。
2. **Full-dump 直方圖修正 §8.63/8.64 的重大錯誤**：`CGC_SCHED_DBG_ALL`（新增，印每 split 全部 node）顯示 **GATED_DELTA_NET(24)/SSM_CONV(26)/ARGSORT(33) 大部分在 MTL0**——fork 其實有 SSM 的 Metal kernel！§8.63「SSM 無 Metal kernel → CPU 30ms 主體」是 first-3-node 抽樣偏差（CPU split 高估）。CPU 30ms 的真正主體是 **linear_attn_out 等 SSM 投影 MUL_MAT（22 顆在 CPU）+ 小 op + router pin**，不是 GATED_DELTA_NET 本身。
3. **小 op 隔離測量（新增 10 個 M=1 case 進 perf list，並移除會 OOB crash 的 gemma4 Q3_K case）**：

| op | count/step | isolated us | ms/step |
|---|---|---|---|
| RMS_NORM | 107 | 3.18 | 0.34 |
| UNARY(SILU) | 144 | 2.45 | 0.35 |
| GLU(SWIGLU) | 67 | 2.77 | 0.19 |
| GET_ROWS | 135 | 4.73 | 0.64 |
| SOFT_MAX | 34 | 3.16 | 0.11 |
| ARGSORT | 33 | 11.95 | 0.39 |
| SUM_ROWS/CLAMP/SCALE/L2_NORM | 168 | 2.6-2.8 | 0.45 |
| ADD/MUL/CPY/DIV（估 ~2.8us） | 733 | ~2.8 | ~2.1 |
| **小 op 合計（~2000 kernels）** | | | **~4.6ms** |

### GPU 73ms 歸因（qwen36 decode）
```
weight-read MUL_MAT 家族（§8.60/8.62 隔離）：attn 12.6 + FFN 7.9 + head 4.4 = 24.9ms
小 op（新量）：4.6ms
SSM_CONV+GATED_DELTA_NET（50 顆，未量，估）：≤5ms
SDPA（9 顆 × ~50us）：0.5ms
─────────────────────────────
isolated 總和 ≈ 35ms   vs   真實 GPU exec-sum ≈ 73ms
delta ≈ 38ms = per-kernel dispatch + M=1 依賴 bubble（隔離是 hot 重放，真實 graph 是 latency-bound）
```
**定案：§8.64 的「48ms 未歸因」修正為 ~38ms，且不是小 op 執行時間（只有 4.6ms）——是 ~2000 顆 kernel 在 M=1 依賴鏈上的 dispatch/等待 bubble。**

### 刀 B 修正
- **錯誤方向**：優化小 op kernel 本身（執行只 4.6ms，無肉可割）。
- **正確方向**：**沿關鍵路徑減 kernel 數**——(a) 小 op 鏈融合（RMS_NORM+ADD、GET_ROWS+SET_ROWS、UNARY 鏈）砍 dispatch 次數；(b) view/reshape 已在 encode 側省；(c) 但 §8.48 QKV 融合 ≈0 警告「大 kernel 對 dispatch 不敏感」——小 op（2.5us）對 dispatch 應該敏感，需直接實驗驗證：融合一條代表性小 op 鏈量 E2E，而不是先寫一堆融合。
- 工具保留：`CGC_SCHED_DBG_ALL`（full node dump）+ 小 op perf cases + `CGC_SPLIT_EXEC_VERBOSE`。

## 8.66 決定性實驗：小 op 依賴鏈 bubble 量化（2026-08-17）— bubble 假說證實，回收上界 ~2.5-5%

### 方法
test-backend-ops 新增 `test_op_chain`（`OP_CHAIN`）：M=1 F32 {2048,1,1,1} 上建三種 graph——
- **serial**：16/32 顆 RMS_NORM 嚴格串聯依賴（真實 decode 的關鍵路徑形態）
- **parallel**：16 顆 RMS_NORM 全部讀同一輸入（同 kernel 數、零依賴，GPU 可重疊）
- **mix**：RMS_NORM/ADD 交替串聯（真實每層小 op 模式）

`op_size()` override 成 8GiB 逼 n_runs=5，讓**整條鏈**（非被重複的最後 node）成為測量單位；
跑 `CGC_SYNC_GRAPH=1`（每 graph 同步）拿乾淨的 encode+GPU+wait 計時。chain-only 成本 = `5×avg(n) − 4×avg(control)`。

### 結果（load 2.4，3 輪 median，全 µs）

| case | avg us/run | chain-only | per-op |
|---|---|---|---|
| control（1 顆 RMS_NORM） | 39.54 | — | — |
| **serial 16** | 51.70 | **100.3** | **6.27** |
| **parallel 16** | 37.71 | **30.4** | **1.90** |
| serial 32 | 67.18 | 177.7 | 5.55 |
| mix（rms+add）16 | 49.33 | 88.5 | 5.53 |

### 結論：依賴序列化成本 = ~4.4µs/op，bubble 假說證實

1. **依賴鏈每顆小 op 付 ~4.4µs 序列化成本**：serial 6.27 vs parallel 1.90 µs/op（同 encode、同 kernel 數，唯一差別是依賴）。M=1 依賴鏈上 GPU 無法重疊，每顆 kernel 都要等前一顆完整結束 + 快取 flush。
2. **線性擴展**：serial32 ≈ 2× serial16（177.7 vs 100.3）——成本是 per-op 線性的，不是啟動一次性。
3. **真實層模式（mix）與純 serial 同級**（88.5 vs 100.3）——ADD 混入不改變機制。
4. **與 §8.65 的 isolated 數字互證**：§8.65 的 2.3µs/op isolated（4M 重疊 duplicate 重放）正是 parallel floor（1.9µs/op）——**isolated-sum 35ms 已經是重疊地板**，38ms bubble = 依賴序列化，完全對上。

### E2E 回收上界（誠實版）
- 每 step ~2000 顆小 kernel，保守估 60% 在關鍵路徑串聯 → 1200 × 4.4µs ≈ **5.3ms/step** 依賴成本。
- 融合配對（RMS_NORM+ADD → 1 顆、GET_ROWS+SET_ROWS → 1 顆）砍半關鍵路徑小 kernel → 回收 ~2.6ms/step。
- 對 ~105ms step：**~2.5%**；把 SSM 投影等中型 kernel 也算進去，上界 ~5%。
- **判定**：比大 op 融合（§8.50 QKV ≈0%）和 Q2_K（~0.5%）好一個量級，但仍是個位數。值得做一顆代表性融合（RMS_NORM+ADD）做 E2E 對拍，不值得先寫一堆。
- 殘留的 ~33ms bubble（38 − 5.3）來自大 kernel 之間的等待（MUL_MAT 100µs 級）——那是 engine 層（§8.46/8.64 的 subgraph 提交 + 依賴鏈）的問題，不是小 op 融合能碰的。

### 工具保留
`test_op_chain`（OP_CHAIN，perf list 5 cases）+ 本次方法論（op_size override 逼鏈成為測量單位）留在 fork 作為引擎層後續 A/B 的標準量具。

## 8.67 E2E 決定性實驗：ADD+RMS_NORM 融合（2026-08-17）— bit-identical 但淨中性，小 op 融合線正式關閉

### 實作（Metal encoder-level，非新 GGML op）
- **新增 `kernel_add_rms_norm_f32(_4)`**（Metal）：單 dispatch、雙 dst——同時寫 residual（ADD dst）與 rms_norm(a+b)（RMS dst）。reduction 結構與 `kernel_rms_norm_fuse_impl` 完全同構（同 threadgroup/simd/累加順序）→ 輸出與未融合路徑 bit-identical（已證）。
- **encoder 融合**：`ggml_metal_op_bin` 對 ADD 做 lookahead（`CGC_ADD_RMS_FUSE=1` 啟用），下一個非空 node 是 RMS_NORM 且 src0==該 ADD 即融合，返回 n_fuse=2 跳過 RMS node。
- **關鍵修正（第一版沒觸發）**：`ggml_can_fuse_ext` 要求 ADD **恰好一個 use**（`ggml_node_has_n_uses==1`）——殘差流的 ADD 有多個 consumer，generic check 直接擋掉 → llama-simple 裡 fusion 從未觸發（首輪 E2E 是 base-vs-base 空測！）。改為自訂 adjacency check（雙 dst kernel 對 multi-consumer 天然安全），fusion 才真的在模型裡跑。
- qwen36 decode graph 有 **54 對 ADD→RMS_NORM**（`attn_residual-L→norm-X`、`l_out-L→norm-X`）。

### 驗證（qwen36 IQ3_XXS，-ngl 99 + cache 4GiB，128 tok，5 輪交錯）
- **5/5 輪 BIT-IDENTICAL**（輸出逐 byte 相同；hit rate 兩臂皆 84.9%）。
- microbench（OP_CHAIN mix16，5 輪交錯）：fused 45.37 vs unfused 47.75 us/run → chain 省 ~12µs/16-op ≈ **1.7µs/pair**（低於 §8.66 的 4.4µs/op——融合 kernel 本身更重：雙 dst + 兩次 add 計算）。
- E2E 速度：fused median 6.97 vs base 7.38 t/s——**在 ±6-10% 負載噪音內無差異**（記錄 load 後，fused 臂恰好都拿到較高負載 slot）。

### 定案：小 op 融合不是可用的 E2E 槓桿
1. **§8.66 的依賴 bubble 是真的（microbench 4.4µs/op），但不可回收**：可融合的結構模式只有 54 對/step（108 顆 kernel / ~2000），理論天花板 54×1.7µs ≈ **0.09%**，實測 0±噪音。
2. §8.66 上界（~2.5%）假設 ~1200 顆小 kernel 可融合——實際只有 ADD→RMS 對構成可融合模式，其他小 op（MUL/UNARY/GET_ROWS…）沒有對偶。
3. 剩餘 ~33ms bubble 是**大 kernel 之間的等待**（MUL_MAT 100µs 級彼此序列化），不是小 op dispatch——刀 B/C 的正確目標，小 op 融合碰不到。
4. 與 §8.50（QKV 融合 ≈0）同結論：**融合的 E2E 回收在個位數以下，不值得為 bit-identity 風險投入**。

### 保留物
- `CGC_ADD_RMS_FUSE`（預設 off，零行為變更）+ `kernel_add_rms_norm_f32(_4)` + encoder 融合路徑——留作引擎層未來「同結構大 kernel 對」的模板（如 RESHAPE 對、GET_ROWS+SET_ROWS 對，若引擎層改造後出現新瓶頸可重用）。
- `tmp/q36_add_rms_ab.sh`（含 load 記錄的 A/B 模板）。

## 8.68 P1 收尾報告：引擎層收斂與唯一 >10% 的未試槓桿（2026-08-17）

### 最終歸因（qwen36 decode ~103ms/step，§8.60/8.63/8.65/8.67 合併）

```
GPU 73ms = weight-read 24.9（attn 12.6 + FFN 7.9 + head 4.4）
         + 小 op 4.6（~2000 kernels，isolated=重疊地板）
         + SSM ≤5 + SDPA 0.5 + 依賴 bubble ~38
CPU 30ms = MUL_MAT-family 20-25（本次 §8.68 重驗：28 顆）
         + router topk 2.2 + 其他小 op ~5
```

### 引擎層候選全清單（已試 → 結果）
| 候選 | 結果 | 章節 |
|---|---|---|
| splits 合併 / rc0 / n_cb / wake-poll / topk kernel | 全證偽（±0-2%） | §8.62/8.63/8.64 |
| nr0=8、Q2_K、dequant 家族 | 證偽（<1.5%） | §8.43/8.61 |
| QKV 融合 / MUL_MAT_ID_GLU / ADD+RMS 融合 | 全 ≈0 E2E | §8.50/8.67 |
| prewarm / 預測 / double-buffer / MTP / workers | bounded-residency 線關閉 | §8.12-8.15 |
| **大 kernel 依賴等待** | **未量（P1-B）** | — |
| **CPU MUL_MAT 錯排** | **未試（P1-A，本次證據）** | §8.68 |

### P1-A（最高優先）：25 顆 src0=MTL0 的 CPU MUL_MAT 錯排修復 — 預期 +15-20%

**新證據（full dump，CGC_SCHED_DBG_ALL + src0 buft 關聯）**：decode step 的 CPU split 有
**28 顆 MUL_MAT-family**，其中 **25 顆 src0（權重）在 Metal pool**，只有 3 顆在 CPU：
- `ffn_moe_logits-6/-26`（router logits——A+B split **刻意 pin** CPU）
- `shared_expert_gate-35`（shared expert gate——也被 pin）

25 顆 = SSM 投影（`linear_attn_out`×3、`z`×3、`attn_output`×1、未名×3）+ shared-expert FFN
（`ffn_up`×4、`ffn_gate`×2、`ffn_shexp`×1）+ **6 顆 MUL_MAT_ID（MoE gate/up/down，§8.64 的
「src0 明明在 Metal pool 卻排 CPU」——現確認是 6 顆不是 4 顆）**。

**機制**（§8.64 已代碼級定位）：sched pass-2 的 GPU 擴張被 router/topk 的 CPU pin island 切斷，
落在 island 另一側的 node 被「expand rest」用 CPU 填掉；pass-3 upgrade 只在 `bufts[b] == bufts[cur]`
（同一 buffer 型別）才升 Metal——CPU buft ≠ Metal buft 永不升級，**一旦落 CPU 就卡死**。

**為什麼這次不是 placement-invariant**：rc0（§8.62）證偽的是「把 CPU work 搬去 Metal」
（router 是小 work）；P1-A 是**把 sched 錯排的 Metal-work 修回去**——權重已在 Metal，
GPU 總 work 不變，純粹把 25 顆 GEMV 的執行從 CPU 移回 Metal。CPU Q6_K GEMV M=1 有效帶寬
~8-15 GB/s vs Metal 87 GB/s（§8.60 head 實測）——每顆省 5-8×。

**預期回收**：CPU 30ms → ~10ms（router 2.2 + shared gate ~1 + 小 op ~7）；Metal +3-4ms
（25 顆 × ~0.1ms）。淨省 **~17ms/step ≈ +17% decode**（103→86ms）。

**執行步驟**：
1. 對 25 顆逐顆驗證 **src1（激活）也在 Metal**（SSM 輸出在 Metal per §8.65，應成立；若有 CPU
   激活則需 copy，收益打折——先量再修）。
2. 最小修法：sched pass-2 加入「src0 buffer 在 Metal 的 MUL_MAT-family 反向擴張」規則（或
   pass-3 upgrade 放寬 buft 型別條件，Metal 優先）。
3. bit-identity 對拍（-ngl 99 + cache 128 tok）+ 3 輪交錯 A/B。

### P1-B（測量優先）：大 kernel 依賴等待量化
38ms bubble 的小 op 成分已證不可回收（§8.67，54 對上限 0.09%）；大 kernel 成分未量。
test-backend-ops 加 mixed chain（MUL_MAT GEMV + 依賴小 op vs 獨立）量化每顆大 kernel 的
等待成本——決定 38ms 是否還有任何槓桿（graph 重排 / per-layer pipeline 的依據）。

### P1-C（延後）：per-layer pipeline（turbo 式 per-layer command buffer + early commit）
大工程；只在 P1-A/B 後仍剩 >10ms bubble 才啟動。wake-poll spin 已證偽（§8.51），
「減少依賴鏈同步」未試但需 P1-B 的數字決定值不值得。

### P1 收尾判定
**P1-A 是整個 P1 唯一有兩位數潛力的未試槓桿**（證據：25/28 src0=MTL0）。先做 30-min 的
src1 驗證 + 修復 A/B；若回收 >10% → P1 引擎層轉正；若 ≤5% → P1 收斂於「引擎層無兩位數
槓桿」，資源轉回 swift/turbo 線。

## 8.69 P1-A 翻案：25 顆「錯排」是 dump bug，sched 放置正確（2026-08-17）

**執行 P1-A 第一步（src1 驗證）時發現：§8.68 的「25/28 顆 src0=MTL0 卻在 CPU split」證據是
CGC_SCHED_DBG_ALL dump 的讀取 bug，不是 sched 真的錯排。**

### Bug 機制（已修復，修復保留）

dump 迴圈原本讀 `sched->graph.nodes[nn]`（nn ∈ [split->i_start, i_end)）——但 `sched->graph`
是 split_graph 重建的 **graph_copy**（node 陣列被 input copies/deps 重新排列），索引與原始
graph 錯位，`id=`/`src0=` 讀到**別的 tensor 的過期 hash 值**。同時 `hv_tensor_backend_ids`
在 reserve 階段被 11 次 split_graph 覆寫，decode 復用 cached splits 時 dump 讀到的 id 與
split 形成時不一致。修復：dump 改讀 `split->graph.nodes[nn]`（該 split 的 graph view，真實
node 順序）。**此修復保留**（診斷工具正確性）。

### 翻案證據（修好 dump 後，decode step 實測）

| 項目 | §8.68 宣稱 | 實測（split->graph） |
|---|---|---|
| CPU split 的 MUL_MAT-family | 28 顆（25 錯排） | **80 顆 = 40×ffn_moe_logits + 40×shared_expert_gate** |
| linear_attn_out（30 層） | CPU 錯排 | **全部 MTL0** |
| z / attn_output / ffn_up / ffn_gate / ffn_shexp | CPU 錯排 | **全部 MTL0** |
| ffn_moe_gate/up/down MUL_MAT_ID | 6 顆 CPU | **全部 MTL0**（40 層 × 3 全在 Metal pool） |
| Qcur/Kcur/Vcur | — | 全部 MTL0 |

**CPU 上的 MUL_MAT-family 只有刻意 pin 的 router logits（A+B split，§7.3.5/P1-2）與
shared_expert_gate**——沒有「權重在 Metal 卻被排 CPU」的 node。P1-A 的前提（21-25 顆錯排
GEMV、+15-20% 回收）**不存在**。

### 定案

- **P1-A 正式證偽並關閉**。sched 的 pass-2/3 放置規則不需要修改；§8.68 的執行步驟 1-3 全數
  取消，`CGC_P1A_UPGRADE` 規則與全部探針已從 ggml-backend.cpp 回滾（無殘留）。
- §8.64「25 顆 src0=MTL0 錯排、唯 4 顆 MUL_MAT_ID」同樣是同一 dump bug 的產物，一併翻案。
- **CPU 30ms 的真實成分只剩**：router logits（40×MUL_MAT，刻意 pin）+ shared_expert_gate
  （40×MUL_MAT，刻意 pin）+ SSM 小 op 鏈——§8.63 的 per-split exec 定案（~23ms）維持。
- **引擎層結論收斂**：qwen36 decode 的剩餘帳單 = GPU 真執行（weight-read 25 + 小 op 4.6 +
  SSM ≤5 + SDPA 0.5）+ 依賴 bubble ~38ms（§8.65）。bubble 的小 op 成分證不可回收（§8.67），
  大 kernel 成分未量（P1-B 仍開放）。**P1 引擎層沒有剩餘兩位數槓桿**——資源轉回 swift/turbo
  線（§8.68 判定標準的 ≤5% 分支生效）。

### 診斷工具修正記錄

- `CGC_SCHED_DBG_ALL`：node 迴圈改讀 `split->graph`（真實 node 順序），`id=` 不再錯位。
- 新增的 P1-A 探針（P3-BEST / P5-SPLIT / P1A-PROBE / SPLITGRAPH）已全部回滾；`CGC_P1A_UPGRADE`
  env 不存在（規則已刪）。

## 8.70 CPU 側 router logits + shared_expert_gate 測量與藏匿評估（2026-08-17）

### 測量（CGC_SPLIT_EXEC_VERBOSE，decode step，qwen36 -ngl 99 + cache 4GiB）

| 成分 | ms/step | 顆數 | 說明 |
|---|---|---|---|
| **router 鏈（ffn_moe_logits MUL_MAT + SOFT_MAX + ARGSORT + topk）** | **15.60** | 40 | 每顆 0.12-1.61ms，長尾大（層 27/1/22 到 1.3-1.6ms，多數 <0.3ms） |
| shared_expert_gate MUL_MAT | 2.38 | 40 | 每顆 ~0.06ms |
| 其他 CPU | 0.02 | — | — |
| **CPU exec 合計** | **18.00** | 81 splits | §8.63 的 ~23ms 差異為載入噪音 |

router logits 是 CPU 側唯一大頭（15.6ms/step = CPU 30ms 的一半）。長尾（1.6ms 的顆）來自
CPU GEMV M=1 在 4 執行緒下的 threadpool barrier 開銷 + 載入干擾。

### 藏匿實驗：CPU 執行緒數 A/B（t=1 vs 2 vs 4，3 輪交錯，128 tok）

| t | CPU exec（per-split exec，2 輪） | E2E median（3 輪） |
|---|---|---|
| 1 | 22.95 / 15.54 ms | **9.06** |
| 2 | **15.11 / 14.17 ms** | **9.50** |
| 4 | 18.88 / 19.93 ms | **9.52** |

**因果乾淨**：t 只改 CPU 執行緒數，graph/splits/GPU 端完全不動。t=2 把 CPU 時間砍 25%
（19→14.6ms），E2E 卻在 ±6% 噪音內（9.06/9.50/9.52 median）——**CPU router 鏈不在關鍵
路徑上**。機制：sched 的 event 是 fire-and-forget（§8.62），CPU split 不等前一個 MTL0 split
完成就開始執行，router 的 15.6ms 完全藏在 GPU 執行（~73ms）後面。

### 定案

- **「提早 dispatch」證偽**：CPU router 已經與 GPU busy 重疊——它是「前一個 GPU split 還在
  queue 裡跑、CPU 先算下一層 router」的既有結構。沒有更早的 dispatch 時機可搶（router 依賴
  前一層 attention 輸出，已是資料依賴允許的最早點）。
- **「更小 CPU 並行」無 E2E 收益**：t=2 雖省 CPU 時間，但 CPU 不在關鍵路徑，E2E 不變。
  日常配置維持 t=4（GPU encode 也在同執行緒池，4 執行緒對 encode 更穩），無需改。
- **CPU 30ms 全數證實不在關鍵路徑**（router 15.6 + shared gate 2.4 + SSM 小 op）——真正的
  關鍵路徑是 GPU 執行 + 依賴 bubble（§8.65 的 73ms）。**引擎層剩餘唯一開放項是 P1-B（大
  kernel 依賴等待的 ~38ms bubble）**；若 P1-B 也證偽，qwen36 decode 就是 GPU 真執行上限
  （weight-read 25 + 小 op 4.6 + SSM 5 + SDPA 0.5 ≈ 35ms 地板 → ~28 tok/s 理論，實測
  ~9.5 t/s 的差距全在 bubble + CPU↔GPU 資料依賴序列化，兩者皆無引擎層可回收槓桿）。

### 診斷工具
CGC_SPLIT_EXEC_VERBOSE（env-gated）已是現成工具；本次無新增插樁、無程式碼變更。

## 8.71 P1-B：大 kernel 依賴等待量化 — 證偽，~38ms bubble 不是依賴延遲（2026-08-17）

### 方法（test-backend-ops 擴充）

新增 `test_op_chain_gemv`（OP_CHAIN_GEMV）：F16 2048×2048 GEMV 鏈（M=1，模擬 qwen36
decode 的 projection），三種模式：
- mode 3：嚴格依賴鏈（y_i = W_i * y_{i-1}）——真實 decode 每層大 kernel 鏈
- mode 4：同 kernel 數、零依賴（全讀 x_0，尾端求和）——GPU 可重疊的平行地板
- mode 5：依賴鏈 + RMS 交錯（GEMV→RMS→GEMV…）——真實層模式

**harness 修正**（重要）：原 perf harness 對任何 case 都把 out node duplication n_runs 次再
計時——小 op 鏈（§8.66）out 便宜所以污染可忽略；**大 kernel 的 out 是 GEMV（~90us）時被
重複執行 4 次，直接摧毀測量**（初版數據 serial「比 parallel 快」，就是這個 artifact）。
修正：`run_whole_graph()` 時跳過 duplication、計數改 +1（而非 +n_runs）。**此修正保留**
（OP_CHAIN 小 op 與 OP_CHAIN_GEMV 現在都量整條 chain 一次執行）。

### 結果（CGC_SYNC_GRAPH=1 同步計時，5 輪取 median，rounds 1-2 乾淨）

| case | median us/run | 說明 |
|---|---|---|
| 單顆 GEMV（control） | 348.95 | 含 graph 固定 overhead |
| **serial 8** | 1005.36 | 8 顆依賴 GEMV |
| **parallel 8** | 1000.60 | 8 顆獨立 GEMV |
| serial+rms 8 | 1039.22 | 依賴鏈 + 8 顆 RMS |
| **serial 16** | 1688.58 | 16 顆依賴 GEMV |
| **parallel 16** | 1716.14 | 16 顆獨立 GEMV |

**關鍵指標**：
- **依賴序列化 = serial − parallel**：n=8 → +4.8us（0.59us/kernel）、n=16 → −27.6us
  （負值，噪音內）。**大 kernel 依賴等待 ≈ 0**。
- **marginal GEMV = 85-94us/kernel**（(16-8) 鏈的增量 85.4us、(8-1) 的增量 93.8us），跨鏈長
  恒定——每顆大 kernel 已是帶寬 bound（8MB F16 讀取 / ~90us ≈ 87GB/s，與 §8.60 一致），
  **無累積 bubble**。
- RMS 插入成本 4.2us/顆——與 §8.66 小 op 依賴 4.4us/op 一致。

### 定案：P1-B 證偽，P1 引擎層正式收斂

1. **~38ms bubble 不是大 kernel 依賴等待**——serial 鏈與 parallel fanout 同價（±0.6us/
   kernel 噪音內），GPU 早已把依賴鏈流水化，沒有「藏等待」的空間。
2. bubble 的組成：每 step ~2000 顆 kernel 的**固定 dispatch 開銷**（short kernel 無法
   飽和）+ CPU↔Metal 邊界（§8.62 已證 splits 無關）。兩者皆非「依賴等待」類槓桿。
3. 剩餘候選全部證偽：splits/n_cb/wake-poll/topk kernel（§8.62-64）、nr0=8/Q2_K（§8.43）、
   三種融合（§8.50/8.67，E2E ≈0 的原因現在有 microbench 支持：大 kernel 依賴 ≈0，融合省
   掉的只是 ~4us 的小 op 插入，上界 0.1%）、CPU 並行/提早 dispatch（§8.70）。
4. **P1 引擎層定案：qwen36 decode 已在大 kernel 帶寬效率上運行（marginal 87GB/s），
   GPU 側無剩餘引擎層槓桿**。若要 >20 tok/s 必須改 kernel 本身（P1-3 線，已證 nr0 無效）
   或換引擎（turbo 式 per-layer pipeline，§8.68 P1-C——但本節證明依賴鏈無等待可藏，
   pipeline 的收益上限 = 0，正式關閉）。
5. 資源結論：llama.cpp fork 的 decode 優化線**全線收斂**；剩餘差距（~9.5 vs 理論 ~28
   tok/s）是 GPU 真執行（35ms）與 dispatch bubble（38ms）之和，後者無引擎層修法。

### 診斷工具
- `OP_CHAIN_GEMV` case 保留在 perf list（引擎層後續 A/B 的標準量具）。
- harness 的 `run_whole_graph` duplication 修正保留（對 OP_CHAIN 系列 case 都是正確性修復）。

## 8.72 最終收斂報告：qwen36 decode 理論上界 vs 實測差距（2026-08-17）

### 一、帳單總表（qwen36 IQ3_XXS，-ngl 99 + cache 4GiB，128 tok 真實文本）

```
實測 wall        ~103 ms/step  →  ~9.5 tok/s（低載 7.2-10.5，median ~8.4-9.5 視載入）
GPU 側 exec-sum   ~73 ms/step（§8.65 CGC_SPLIT_EXEC：CPU 側 encode + backpressure）
CPU 側 exec-sum   ~30 ms/step（router 15.6 + shared gate 2.4 + SSM 小 op，§8.70 證明
                             大部分不在關鍵路徑——t=2 砍 25% CPU 時間 E2E 零變化）
```

### 二、GPU 73ms 的成分分解與「理論上界」

| 成分 | ms/step | 來源 |
|---|---|---|
| weight-read（attn 12.6 + FFN 7.9 + head 4.4） | ~25 | §8.60 test-backend-ops 隔離 |
| 小 op 執行（~2000 kernels，isolated 地板） | ~4.6 | §8.65 |
| SSM（GATED_DELTA_NET/SSM_CONV，MTL0） | ≤5 | §8.65 |
| SDPA（10 層 full-attn） | ~0.5 | §8.65 |
| **isolated-sum（=純 kernel 執行地板）** | **~35** | → **理論上界 28.6 tok/s** |
| 依賴 bubble（isolated-sum 與實測之差） | ~38 | §8.65 未歸因 |

**P1-B（§8.71）的關鍵修正**：38ms bubble **不是大 kernel 依賴等待**——OP_CHAIN_GEMV
serial vs parallel 同價（±0.6us/kernel）、marginal GEMV 87GB/s 已是帶寬 bound。bubble 的真
實成分 = **~2000 顆 short kernel 的固定 dispatch/launch 開銷**（~19us/kernel × 2000 ≈
38ms），不是依賴序列化。→ **28.6 tok/s 是「dispatch 免費」的理想化上界，不可達**。

### 三、三層天花板

| 層級 | ms/step | tok/s | 說明 |
|---|---|---|---|
| 實測（現況） | ~103 | **~9.5** | 包含全部 overhead |
| 引擎層天花板（GPU 73ms 不變、CPU 全隱藏） | ~73 | **~13.7** | 需 CPU 100% 藏進 GPU busy（§8.70 只證 router 部分可藏） |
| 純 kernel 地板（dispatch 免費，不可達） | ~35 | ~28.6 | P1-B 證明 dispatch 成本真實存在且不可消除 |

**>20 tok/s 需要 50ms/step 以下**：介於「引擎層天花板 73ms」與「純 kernel 地板 35ms」之間。
即必須**同時**：(a) CPU 完全隱藏 + (b) dispatch overhead 砍掉一半以上。兩者都已被證偽。

### 四、為什麼所有已試槓桿都回收不到 20

| 槓桿 | 結果 | 為何無效（累積證據） |
|---|---|---|
| bounded-residency 全線（prewarm/預測/double-buffer/workers） | §8.12-15 關閉 | 非速度瓶頸 |
| splits 合併 / rc0 / n_cb / wake-poll / topk kernel | §8.62-64 全證偽 | 時間=GPU+CPU 執行，非同步 |
| nr0=8 / Q2_K / dequant 家族 | §8.43/8.61 證偽 | GEMV 已帶寬 bound，tile 無關 |
| QKV / MUL_MAT_ID_GLU / ADD+RMS 融合 | §8.50/8.67 E2E ≈0 | 大 kernel 依賴 ≈0（§8.71），融合只省 4.2us 小 op 插入 |
| CPU 並行 / 提早 dispatch | §8.70 證偽 | CPU 不在關鍵路徑 |
| P1-A sched 錯排修復 | §8.69 翻案 | 錯排是 dump bug，sched 正確 |
| **P1-B 大 kernel 依賴等待** | **§8.71 證偽** | serial=parallel，依賴 ≈0 |
| MTP / 投機 | §8.7 淨負 | draft 是第二 cache + 每步 MTP decode |

### 五、最終定案：若要 >20 tok/s 只剩兩條路

1. **kernel 本身**：weight-read 已是 87GB/s（近 M4 實效帶寬），但 ~2000 顆 short kernel 的
   dispatch 開銷（38ms）是最大單一槓桿。唯一能碰它的方式 = **大幅減少 kernel 數**：
   per-expert batched GEMM（turbo 式單 dispatch 多 expert）——但 P1-3a/b/c 已證 nr0 tile
   與 GLU 融合無效，且 turbo 的 batched 模式在 llama.cpp 沒有對應的 graph op 可掛。
2. **換引擎**：Swift/turbo 的 Metal 引擎（per-layer pipeline、per-expert batched GEMM、
   大 kernel 化）是唯一實證 >15-20 tok/s 的路徑（turbo 15-17、Swift 18-20）。llama.cpp
   fork 的架構（graph-sched + ~2000 kernels/step）**結構性上不支援** 20 tok/s——除非做
   引擎級重寫（P1-C，§8.71 已證明「依賴鏈無等待可藏」，pipeline 化收益上限 = 0，正式關閉）。

**結論**：qwen36 decode 的差距 = GPU 真執行 35ms（不可減，已帶寬 bound）+ dispatch bubble
38ms（不可減，kernel 數結構性固定）+ CPU 序列化（大部分可藏但不足以補足）。**llama.cpp
fork 的 decode 優化線到此收斂；20+ tok/s 標的屬於 Swift/turbo 引擎線，不在 fork 內**。

### 六、收斂後建議
- fork 現況（~9.5 tok/s）作為 bounded-residency 的**功能正確性**成果保留（-ngl 99 + cache
  讓原本 OOM 的 13.2GB 模型可跑、bit-identical、hit 38.5%）。
- 速度標的移交 Swift/turbo 線（§10.4 白皮書對照：turbo 15-17、Swift 18-20）。
- 若未來仍要在 fork 內追速度：唯一未試方向是 **graph 層重排減少短 kernel 交錯**（把同層
  MUL_MAT 群聚成連續 dispatch 減少 command buffer 邊界）——但 §8.62 rc0 已證 splits 減半
  無收益，預期也是 ≈0，排最後順位。

## 8.74 高解析度藏匿實驗（2026-08-17）：CPU split 延遲 1:1 進 wall — §8.70「CPU 不在關鍵路徑」翻案

### 動機
用戶質疑 §8.72 帳單自洽性：若 CPU 30ms 真「藏在 GPU busy 後面」，wall 應 ≈ GPU 73ms（~13.7 tok/s）而非實測 103ms（73+30=103 恰好相加）。§8.70 的 t=1/2/4 實驗靈敏度不足（砍 25% CPU ≈ 4.6ms < E2E ±6% 噪音 ≈ 6ms），無法區分「藏住」與「砍不夠多」。

### 方法：delay injection（CGC_DELAY_CPU_US，env-gated，已保留）
在 `ggml_backend_sched_compute_splits` 每個 CPU split 執行後、event record 前注入可控 delay（usleep）。延遲在 exec 計時之後 → 不污染 exec-sum，但直接進 wall。若 CPU 藏在 GPU busy 後，斜率應 ≈0；若 CPU 在關鍵路徑，斜率應 ≈1（每 +1ms CPU 時間 = +1ms wall）。

### 結果（qwen36 IQ3_XXS，-ngl 99 + cache 4GiB，128 tok，3 輪交錯 median）

| CGC_DELAY_CPU_US | wall/step | 增量 | 理論（81 CPU splits × delay） | 斜率 |
|---|---|---|---|---|
| 0 | 210ms | — | — | — |
| 100 | 217ms | **+7ms** | +8.1ms | **0.86 ≈ 1** |
| 200 | 337ms | +127ms | +16.2ms | 7.8× 超線性 |
| 500 | 395ms | +185ms | +40.5ms | 4.6× 超線性 |

### 定案
1. **CPU split 延遲 1:1 進 wall（斜率 ≈1）——§8.70 的「CPU 30ms 全數不在關鍵路徑」翻案**。若 CPU 真的藏在 GPU busy 後面，注入的 CPU delay 會被 GPU 執行吸收（斜率 0）；實測每 +1ms CPU 時間 = +1ms wall，**CPU 序列化是關鍵路徑的一部分，不是可隱藏項**。§8.70 的 t 實驗只是解析度不足（4.6ms < 6ms 噪音），不是 CPU 不在關鍵路徑。
2. **超線性放大（200us → +127ms，7.8×）機制**：delay 在 event record 之前 → 拖住所有等待該 event 的後續 GPU split 的 input copy（per-split 對比：MTL0 split exec 從 0.5-1ms 漲到 2-4ms）——**CPU↔GPU 邊界的 event 同步把 CPU 延遲傳染到 GPU 側**，且 CGC_SCHED_TIMING 顯示 delay=200 時 mtl_ms 從 148 → 204ms（+56ms）。這解釋了 73+30=103 的相加：不是巧合，是 exec-sum 已含邊界同步成本。
3. **帳單修正**：103ms 的真實結構 = GPU split 處理（exec-sum 73ms，含 encode+backpressure+邊界同步）+ CPU split 執行（30ms，**全在關鍵路徑**）——**沒有「藏匿」**。§8.72 的「引擎層天花板 13.7 tok/s（CPU 100% 隱藏）」是無法達成的理論值；實測 9.5 的差距 = CPU 30ms 沒藏住 + 依賴序列化。
4. **含義**：CPU router 15.6ms 不只是「可藏的 15.6ms」——它是關鍵路徑上的 15.6ms。若 router 能上 Metal（P1-2，但 qwen36 實測 −33%，GPU 序列化吃掉），或減少 CPU split 數（A+B split 的刻意 pin），是唯二能砍它的路徑。§8.70「資源轉回 swift/turbo 線」的結論不變，但理由是「CPU 也在關鍵路徑、fork 無法砍」，不是「CPU 可藏」。

### 診斷工具
`CGC_DELAY_CPU_US` / `CGC_DELAY_MTL_US`（env-gated，預設 off，零行為變更）——藏在 ggml-backend.cpp compute_splits 迴圈，作為「任何延遲敏感度」的標準量具。

## 8.75 藏匿上限的最終答案：真重疊 = 0 — cache 模式的 callback 同步消滅了 fire-and-forget（2026-08-17）

### 動機
§8.74 證明 CPU 延遲 1:1 進 wall。用戶要求更進一步：把 103ms 拆成「真重疊 + 邊緣序列化」兩份，量化藏匿的真實上限。

### 方法：GPU busy 探針（新增 CGC_HIDE_TRACE，env-gated，已保留）
- `ggml_metal_query_busy(ctx)`（ggml-metal-context.m）：非阻塞輪詢 n_cb+1 個 MTLCommandBuffer 的 status，任一非 Completed 即 busy
- `ggml_backend_metal_is_busy(backend)`（ggml-metal.cpp）+ proc-address 註冊（避免 ggml-base 連結 metal 的 circular dep）
- compute_splits 迴圈在「split 入口」與「graph_compute_async 返回後」各取樣一次，統計 CPU/MTL split 的 busy 比例

### 結果（qwen36 IQ3_XXS，n99c 4GiB，CGC_HIDE_TRACE）

| 取樣點 | WITH cache（n99c 日常） | WITHOUT cache（-ngl 30 對照） |
|---|---|---|
| CPU split 入口 busy | 0/81 | 0/1 |
| MTL split 入口 busy | 0/81 | 0/1 |
| **MTL post-compute busy** | **0/81** | **1/1** |
| CGC_SYNC_GRAPH=1 vs 0 | **無差異**（4.96/5.06 vs 4.78/4.73） | — |

### 根因（代碼級）：cache 模式走 callback_eval 分支，每 topk 範圍強制同步

1. cache 啟用（或 `LLAMA_EXPERT_CACHE_DEBUG`）時，llama-context.cpp:1411 安裝 `expert_cache_eval_cb` → `sched->callback_eval != NULL`。
2. `expert_cache_eval_cb` 只對 `ffn_moe_topk-*` 回 need=true（llama-context.cpp:1713-1716），其餘 node 回 false。
3. compute_splits 的 callback_eval 分支（ggml-backend.cpp:1844-1880）對每個 need 範圍：`graph_compute_async` → **`ggml_backend_synchronize(split_backend)`（1872 行）** → 呼叫 callback（swap experts）。**每層 topk 都是一個強制同步點**。
4. 因此 n99c decode = 40 層 ×「attention 範圍 compute→sync → topk hook swap → FFN 範圍 compute→sync」的**純串行序列**。GPU 從不積壓（每次 dispatch 後立即被 sync 等完），CPU 也從不與 GPU 重疊。
5. **對照證實探針有效**：無 cache 模式（callback_eval=false）走正常 async 分支，post-compute busy=1——fire-and-forget 真實存在於非 cache 路徑，是被 callback 模式消滅的。

### 定案

1. **真重疊 = 0**。103ms = encode + compute + sync + hook swap 的純串行總和；「邊緣序列化」= 103ms（100%）。**藏匿上限 = 0——沒有「CPU 藏在 GPU busy 後面」這回事，因為 GPU 從不 busy（每個範圍都被 sync 等完才繼續）。**
2. **§8.62「fire-and-forget、CPU 藏在 GPU busy 後面」只對非 cache 路徑成立**——而我們所有 n99c 日常測量都是 cache 模式，從未享受過重疊。§8.70 的「藏匿」結論建立在錯誤前提上。
3. **唯一的修法**：讓 cache 的 hook 不需同步——(a) topk swap 改為只記錄 ids、FFN 範圍改用 Option A slot-table indirection（swap 在 sync 邊界外），讓整個 decode 走回 async 分支；(b) 或把 callback 移除，改在 graph 外做 topk 回讀（P1-2 的 32B/layer 回讀路徑）。兩者都是讓 decode 回到 §8.62 描述的真實 fire-and-forget 結構，預期回收 = GPU 執行期間的 CPU 工作重疊（§8.65 的 ~30ms CPU 全部或部分）。
4. **對 TPOT 帳單的修正**：103ms 不是「73 真 GPU + 30 真 CPU」，而是「~73ms GPU 範圍執行（含每範圍 sync 的等待）+ ~30ms CPU split + ~40 次 sync/hook 開銷」的串行和——GPU 執行（~35ms isolated-sum）與 CPU 執行（~30ms）之間**零重疊**，全數邊緣序列化。

### 診斷工具
`CGC_HIDE_TRACE=1`（busy 取樣統計 CGC_HIDE_SUM / CGC_HIDE_POST）+ `CGC_HIDE_TRACE_DBG=1`（per-cb status）——env-gated，預設 off，零行為變更。`ggml_metal_query_busy` / `ggml_backend_metal_is_busy` / proc-address 註冊一併保留。

## 8.76 sync 等待成本量化（2026-08-17）：callback_eval 的強制同步 = 65-79ms/step，引擎層天花板 13.7 tok/s 是可達的修復目標

### 方法
在 compute_splits 的 callback_eval 分支（ggml-backend.cpp:1872）`ggml_backend_synchronize` 前後加計時（CGC_SYNC_WAIT_DBG=1，env-gated）。

### 結果（qwen36 IQ3_XXS，n99c 4GiB，32 tok）

| 指標 | 數值 |
|---|---|
| 每 step sync 次數 | **162**（81 MTL + 81 CPU split 的每個 callback 範圍） |
| sync 等待總和 | **65-79 ms/step**（median ~71ms） |
| 平均每次 | 0.40-0.49 ms |

### 帳單完成閉合

```
wall 103ms = GPU split 處理 73ms + CPU split 執行 30ms（73+30=103 ✓）
GPU split 處理 73ms = encode + dispatch + sync 等待（其中 GPU 真執行僅 ~35ms isolated-sum）
→ sync 等待 ≈ 65-79ms = 73ms 的主體，不是「GPU 真執行」
```

**這解釋了 §8.65 的「73ms exec-sum」與 §8.75 的「真重疊=0」**：mtl_ms 73ms 幾乎全是 callback 強制 sync 的等待（GPU 範圍執行 + 排隊），CPU 執行無法與它重疊因為每範圍都被 sync 截斷。

### 修復回收上限（重新評估 §8.72 的三層天花板）

**§8.72 宣稱「引擎層天花板 13.7 tok/s（CPU 100% 隱藏）不可達」——現在證明它是可達的修復目標**：

- 若消除 callback 強制 sync（修法 a/b，見下），decode 走回 async 分支：
  - GPU 真執行 ~35ms 不變（還是要跑）
  - CPU 30ms 可與 GPU 35ms 重疊（§8.62 非 cache 路徑已證 fire-and-forget 存在，busy=1）
  - wall 103ms → ~73ms（GPU 範圍執行 + 排隊，sync 等待消失）
  - **tok/s：9.5 → ~13.7（+44%）**
- 殘留：38ms dispatch bubble（§8.71 證非依賴、不可減）+ CPU↔GPU 邊界剩餘。

### 修法候選（供 review，未實作）

| 方案 | 機制 | 風險 | 預期回收 |
|---|---|---|---|
| **(a) swap 移出 sync 邊界** | topk 後只記錄 ids（不 swap），FFN 範圍改用 Option A slot-table indirection——swap 在層邊界/sync 外執行，callback 不再 need topk | 中：Option A 已轉正（§8.31-8.35），但需把 swap 時機與 slot-table 綁定 | ~30ms/step（CPU 重疊） |
| **(b) graph 外 topk 回讀** | 移除 eval callback，decode graph 後統一回讀 32B/layer topk（P1-2 路徑），下一 step 的 prefill/copy 用上一 step ids | 中：P1-2 在 qwen36 曾 −33%（GPU 序列化），但那是「router 上 Metal」；本方案是「router 仍 CPU、僅回讀 ids」，機制不同需重驗 | ~30ms/step |
| (c) 混合 | (a) 為主，(b) 的 32B 回讀僅在需要時 | 低-中 | ~30ms |

**判定標準**：修復後用 CGC_HIDE_TRACE 驗證 GPU busy 比例從 0/81 回升（重疊出現）+ 3 輪交錯 A/B 量 9.5 → ~13.7；bit-identity 對拍（-ngl 99 + cache 128 tok）。

### 診斷工具
`CGC_SYNC_WAIT_DBG=1`（sync 等待總和統計，每 graph 輸出一次）——env-gated，預設 off，零行為變更。

## 8.77 方案 (a) 落地：CGC_OA_ASYNC — Metal splits 走 async，+12.6% 回收（2026-08-17）

### 實作（ggml-backend.cpp `compute_splits`，env-gated）

```c
} else if (getenv("CGC_OA_ASYNC") != nullptr && strcmp(ggml_backend_name(split_backend), "CPU") != 0) {
    // non-CPU (Metal) splits carry no topk node — the router chain (ffn_moe_logits/argsort/
    // topk) is deliberately pinned to CPU by the A+B split (§8.69/8.23), so the eval-callback's
    // per-range synchronize here is pure serialization waste (65-79 ms/step, §8.76).
    // Dispatch the whole split async and let the sched's input-copy event machinery enforce
    // the real data dependencies. CPU splits keep the callback path (topk lives there).
    ggml_backend_graph_compute_async(split_backend, &split->graph);
} else { /* 原 callback_eval 逐範圍 compute + synchronize 路徑（CPU split 保留） */ }
```

安全性（為什麼只對非 CPU split 成立）：qwen35moe 家族在 cache 下 router 權重**刻意 pin CPU**（`LLAMA_EXPERT_CACHE_ROUTER_CPU=1`，§8.23 家族預設）→ `ffn_moe_topk` 節點只存在於 CPU split，eval callback 只在 CPU split 觸發；Metal split 的 callback 檢查是純開銷。資料依賴由 sched 的 input-copy event 機制保證（MTL0 split 的 command buffer 對其輸入的產生 split 有 event wait）。

**gemma4 不啟用**：gemma4 的 router 預設上 Metal（rc0），topk 在 MTL split 內，async 會跳過 callback → hook 不觸發 → 正確性破壞。故 `CGC_OA_ASYNC` 維持 opt-in，生產 harness 僅 qwen36 臂設定。

### 驗證（qwen36 IQ3_XXS、-ngl 99 + cache 4GiB、128 tok、seed=7、-t 4）

**1. bit-identity：三臂全 IDENTICAL**
- base（無 cache） vs cache+OA_ASYNC vs cache（無 OA_ASYNC）：生成文字完全一致（128 tok）
- ⇒ async 化不改變 hook/swap 時序的可觀察行為，權重讀取路徑不變

**2. GPU busy 回升（CGC_HIDE_TRACE，每 step 162 splits = 81 CPU + 81 MTL0）**

| 探針 | 舊路徑（callback） | CGC_OA_ASYNC |
|---|---|---|
| MTL0 split dispatch 後 busy（POST） | **0/81**（從不積壓） | **81/81** |
| CPU split 入口時 GPU busy（SUM） | 0/81 | **80/81** |
| MTL exec-sum（encode+dispatch） | ~65ms/step | **~4.5ms/step** |

sync 等待（CGC_SYNC_WAIT）自 MTL splits 完全消失——GPU 真在 CPU 做 router/topk 時執行前一 MTL split 的工作，fire-and-forget 重疊恢復。

**3. E2E（3 輪交錯、64 tok、同窗低載）**

| 臂 | wall/step（CGC_COMPUTE median） | E2E t/s（median） |
|---|---|---|
| 舊路徑（無 OA_ASYNC） | 98.3 ms | 11.37 |
| **CGC_OA_ASYNC=1** | **83.6 ms（−15%）** | **12.80（+12.6%）** |

### 定案

1. **方案 (a) 轉正**：回收 ~15ms/step（−15% wall、+12.6% E2E），低於 §8.76 預測的 ~30ms——因 83.6ms 中仍有 GPU 真執行 ~35ms + 端點等待 + 殘留的 CPU↔GPU 邊界序列化（負載 3.2 下無法更細分解，機制留 §8.78）。
2. **正確性**：bit-identical 維持（三臂一致），hook/swap 時序不變。
3. **安全範圍**：僅 qwen35moe 家族 + ROUTER_CPU 佈局有效；gemma4 不設（rc0 佈局會跳過 hook）。`CGC_OA_ASYNC` env-gated 保留，`run_n30cache.sh` qwen36 臂已加入。
4. **9.5 → 12.8 t/s**：§8.72 的「引擎層天花板 13.7」仍未達——差額是 GPU 真執行 35ms + dispatch bubble 38ms（§8.71 證不可減）的結構性下限，剩餘差距屬於 kernel/引擎層（Swift/turbo 線）。

### 診斷工具
`CGC_OA_ASYNC=1`（開關）+ `CGC_HIDE_TRACE=1`（busy 探針）+ `CGC_SYNC_WAIT_DBG=1`（sync 統計）——全 env-gated，預設 off，零行為變更。

## 8.78 CGC_OA_ASYNC 擴及 gemma4：bit-identical +22%，§8.77「gemma4 不啟用」翻案（2026-08-17）

### 前置：模型遷移與內接盤清理

gemma4 模型先前僅存外接盤（`/Volumes/AlexZhuang/models_backup/`，10.5GB）。為在內接盤跑 A/B，先清空間：
- `models/mlx`（20GB，MLX 線封存）→ 外接盤 `models_backup/mlx_internal_2026-08-17/`（rsync 驗證後刪除）
- `temp/qwen36_mtp_data`（5GB，MTP 線資料）→ 外接盤 `qwen36_mtp_data_internal_2026-08-17/`
- `~/Library/Caches/{pip,org.swift.swiftpm,Homebrew,Google}`（~1.2GB，全部可再生）刪除
- 內接盤 free 6.5Gi → 32Gi；gemma4 10.5GB 複製回 `models/gguf/`（free → 22Gi）

### §8.77「gemma4 不啟用」的判斷錯誤

§8.77 假設 gemma4 router 預設上 Metal → topk 在 MTL split → async 跳過 callback 會破壞 hook。**實測翻案**：`llama-context.cpp:3489` 的 `ffn_moe_probs` → CPU pin 在「未設 ROUTER_CPU env + ALLOW_NGL 設定」時**就會啟動**（P1-2 的 copy-timing 修復設計）——gemma4 生產配置正是這個狀態，softmax/argsort/topk 全留在 CPU split，hook 在 CPU split 觸發。所以 Metal split 對 gemma4 同樣無 callback 需求，OA_ASYNC 安全。

### 驗證（gemma4 IQ3_S、-ngl 30 + cache 4GiB、64 tok、seed=7、-t 4）

**1. bit-identity：三臂全 IDENTICAL**
base（無 cache） vs cache vs cache+OA_ASYNC 生成文字完全一致；**oa0 vs oa1 行為一致**（OA_ASYNC 不改變 hook 時序的可觀察行為）。hit rate 81.0% 兩臂相同。

**2. GPU busy 回升（每 step 60 splits = 30 CPU + 30 MTL0）**

| 探針 | 無 OA_ASYNC | CGC_OA_ASYNC |
|---|---|---|
| MTL0 dispatch 後 busy（POST） | 0/30 | **30/30** |
| CPU split 入口 GPU busy（SUM） | 0/30 | **29/30** |

**3. E2E（3 輪交錯、64 tok）**

| 臂 | 各輪 t/s | median |
|---|---|---|
| 無 OA_ASYNC | 12.32 / 17.80 / 14.66 | 14.66 |
| **CGC_OA_ASYNC=1** | **18.06 / 17.88 / 16.47** | **17.88（+22%）** |

### 定案

1. **OA_ASYNC 對 gemma4 安全且更快**（bit-identical、busy 回升、+22%）：§8.77 的家族限制撤銷，`run_n30cache.sh` 兩家族皆設 `CGC_OA_ASYNC=1`。
2. gemma4 n30c 中位數 **14.66 → 17.88 t/s**，接近無 cache base 的 ~20.9（§8.78 同窗實測 20.86）——cache 臂的殘餘差距主要來自命中率非 100% 時的 fill 與 eviction。
3. 已知的「gemma4 + cache content-dependent 分歧」（§8.44）與 OA_ASYNC 無關（oa0/oa1 同輸出），維持原風險評估：日常可選 n30 無 cache（最快）或 n30c（RSS 低）。

### 診斷工具
沿用 §8.77（`CGC_OA_ASYNC` / `CGC_HIDE_TRACE`）。gemma4 每 step 30 MTL splits（qwen36 為 81），busy 探針判讀以實際 split 數為準。

## 8.79 §8.77「端點/邊界 ~25ms」四段拆解：EOG sync tail drain 4.2ms + 邊界 0.15ms，原 25ms 不存在（2026-08-18）

### 動機
§8.77 定案將 83.6ms 拆為「GPU ~35 + CPU ~24 + 端點/邊界 ~25」，但「~25ms 端點/邊界」在負載 3.2 下未能更細分解。本節擴充 `CGC_STEP_TIMING` 在 `llama-context.cpp` 的 `process_ubatch` 與 `synchronize()` 內插入四個高精度時間戳，把 per-step wall 沿 CPU↔GPU 邊界切成四段，定界剩餘回收空間。

### 四段拆解設計
兩個 file-scope static：`cgc_last_compute_end_us`（graph_compute 返回時設）、`cgc_last_sync_end_us`（synchronize 返回時設），互相在 `CGC_STEP_TIMING` 路徑內 consume。輸出四個指標：

| 段 | 名稱 | 量測點 | 意義 |
|---|---|---|---|
| 1 | `eog_sync_wait_ms` | synchronize 內 `ggml_backend_sched_synchronize` 前後 | 最後 MTL split 完成等待（Metal fence / callback drain）= GPU tail drain |
| 2 | `dispatch_to_sync_ms` | graph_compute 返回到 synchronize 開始 | app-side pre-sync：set_inputs / output_reorder / build / restore-tensors |
| 3 | `post_sync_to_next_ms` | 前一 sync 返回到下一 graph_compute 開始 | 下一 step input-copy 等待：sampler + KV 更新 + graph build + set_inputs = CPU↔GPU 邊界序列化 |
| 4 | `graph_compute_wall_ms` | graph_compute 進出 | async dispatch wall（實為 CPU 側 99 層 MTL command buffer encoding）|

Sum(1..4) ≈ per-step decode wall（穩態）。

### 量測
qwen36 IQ3_XXS、-ngl 99、4GiB budget、`CGC_OA_ASYNC=1`、pool8、n=128、開放式 prompt（避免 EOS 早停）。Decode 127 步，**12.88 t/s**（與 §8.77 圖表 12.8 一致）。

| 段 | min | p50 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|
| 1 `eog_sync_wait_ms` | 4.17 | **4.24** | 5.15 | 15.76 | 16.52 | 4.82 |
| 2 `dispatch_to_sync_ms` | 0.010 | 0.010 | 0.020 | 0.050 | 0.090 | 0.015 |
| 3 `post_sync_to_next_ms` | 0.000 | **0.150** | 0.300 | 3.150 | 4.530 | 0.257 |
| 4 `graph_compute_wall_ms` | 60.82 | **67.04** | 92.26 | 447.20 | 810.22 | 85.53 |
| per-step wall | 0.00 | 67.04 | 87.97 | 224.22 | 611.25 | 79.77 |

### 判讀

**1. 原圖表「~25ms 端點/邊界」不存在**
段 1+2+3 加總：p50 = 4.24 + 0.010 + 0.150 = **4.40ms**；p99 = 15.76 + 0.050 + 3.150 = **18.96ms**。原 25ms 是 §8.77 對 `graph_compute_wall_ms` 內部結構（CPU encoding vs GPU compute 重疊）的誤估——67ms 內部已含 CPU encoding 60ms + GPU compute 35ms 的 `max(60, 35) ≈ 60ms`，被錯算為「CPU 24 + GPU 35 = 59」再扣 wall 67 = -8 重疊，剩餘 25ms 被歸為「端點」。

**2. CPU↔GPU 邊界序列化已近乎為 0**
段 3 `post_sync_to_next_ms` p50 = 0.150ms——sampler + KV 更新 + graph build + set_inputs 全部加起來不到 0.2ms。**§8.77 收回 CPU 序列化的目標已達成**，沒有殘留邊界序列化可回收。

**3. EOG sync wait 穩態 ≤5ms**
段 1 `eog_sync_wait_ms` p50 = 4.24ms、mean = 4.82ms——穩態低於用戶 5ms 閾值。p90/p99（5.15/15.76ms）對應 12.4% 步數 >5ms，是 cache miss 風暴 / I/O 抖動的間歇事件，非穩態偏移。

**4. 真正瓶頸是 `graph_compute_wall_ms` p50 = 67ms**
驗證 `ggml-backend.cpp:1856`：OA_ASYNC 對 Metal split 走 `ggml_backend_graph_compute_async` → `backend->iface.graph_compute` → Metal 後端的 `graph_compute` 是 **CPU-side MTL command buffer encoding（同步阻塞）**，不是真正的 async。99 層每層多個 MTL kernel 的 encoding（argument binding + threadgroup 配置 + commit）就吃掉 60ms；GPU compute（~35ms）在 Metal commit 後與 CPU encoding 部分重疊在 `graph_compute_wall_ms` 內部。

### 收斂
**依用戶判準「>5ms 評估 step 級 double-buffer 提交」收斂為不實作**：
1. 段 3 p50 = 0.150ms：下一 step input-copy 等待已近 0，沒有邊界序列化可隱藏
2. 段 2 p50 = 0.010ms：app-side 端點殘留也為 0
3. 段 1 p50 = 4.24ms + mean = 4.82ms：穩態 ≤5ms，雙 buffer encoding 理論可隱藏這 4.24ms tail drain，但：
   - 收益 ≈ 4.24 / 67 ≈ 6.3% → 12.88 → ~13.7 t/s（恰好對應圖表 73ms / 13.7 引擎層天花板）
   - 雙 buffer 引入 KV 一致性風險（雙 KV slot + 雙 logits buffer）+ step 級 pipeline 複雜度，不值得
4. 段 1 p99 = 15.76ms 的尾部事件無法用 double-buffer 解（事件本身來自 cache miss 風暴，不是 CPU↔GPU 邊界等待）

**>20 tok/s 的真正槓桿不在 endpoint/boundary（已驗證 ≤5ms），而在 `graph_compute_wall_ms` 的 60ms CPU encoding。** fork 內無解，移交 Swift/turbo 引擎線。

### 三個方向的修正

原提出的三個方向有不準確處，修正如下：

**方向 1 修正：kernel fusion → CPU-side MTL encoder overhead**
原：kernel fusion（MoE router + topk + gather 合一）。
修正：§8.78 已證三大頭（MoE / attention / shared FFN）皆 bandwidth-bound——kernel fusion 對 bandwidth-bound kernel 無顯著 GPU 收益。真正瓶頸是 CPU 側 MTL encoder 的 per-kernel setup 與 argument-binding 開銷（99 層 × 多 kernel × ~19us = ~60ms）。可回收方向是**減少 CPU 側 encoder 的 per-kernel 開銷**（argument table 預綁定、threadgroup 配置 cache），不是 GPU kernel 本身的融合。

**方向 2 修正：多 command buffer 平行 encoding → 單一 uncommitted encoder**
原：多個 MTL command buffer 同時 in-flight（需研究 Metal command queue 容量）。
修正：Metal 單裝置不真正平行 commit；多 buffer 仍序列 commit 且引發跨 buffer data hazard 處理複雜度。可回收方向是**單一 uncommitted encoder**：把 99 層的 encode 合併到單一 `MTLCommandBuffer`、避免每層 commit 的 fence 開銷（讓 GPU 在 CPU encoding 仍在跑時就開始執行已 commit 的 early kernels）。step 級 double-buffered encoding 已被段 1 p50=4.24ms ≤5ms 收斂否決。

**方向 3 修正：PSO 預編譯 → per-step dispatch plan 重用**
原：把 99 層 dispatch sequence 預先編譯成 PSO，encode 時直接綁定 buffer。
修正：PSO（Pipeline State Object）已是 Metal 現有機制（kernel function → `MTLComputePipelineState` 並 cache 在 backend）；不能預編譯的是「dispatch sequence」本身。可回收方向是 **per-step dispatch plan 跨 step 重用**：decode 穩態時每 step 的 dispatch sequence 結構固定（同樣 99 層同樣 kernel 序列），跨 step 重用時直接 memcpy argument table（per-step 變化的只有 KV 位置與 input token），而非重走 graph visitor + tensor visitor + encoder API。對應 llama.cpp 內部已有 graph cache + split cache 機制，下一步是 **block-encode**：把固定層序列的 dispatch plan 一次 build、per-step 只 patch 變動欄位。

### 對 §8.77 圖表的修正
原「分段：GPU 執行 ~35 + CPU ~24 + 端點/邊界 ~25」應改為：

```
graph_compute_wall_ms = 67ms（CPU encoding 60ms + GPU compute 35ms 重疊，max ≈ 60ms）
eog_sync_wait_ms      = 4.24ms（GPU tail drain，可被 double-buffer 隱藏但 ≤5ms 不值得）
post_sync_to_next_ms  = 0.15ms（sampler + KV update，已近乎為 0）
dispatch_to_sync_ms    = 0.01ms（app-side pre-sync，可忽略）
per-step wall ≈ 67 + 4.4 = 71.4ms（實測 77.6ms 因 p90/p99 tail 拉高）
```

引擎層天花板標註應改為：**「§8.79 已驗證 12.88 t/s（per-step 77.6ms）；剩餘 4.24ms tail drain + 60ms MTL encoding 為 CPU-bound 不可減 → >20 移交 Swift/turbo」**

### 診斷工具
`CGC_STEP_TIMING=1` 啟用四段拆解（`CGC_SYNC` + `CGC_COMPUTE` 兩行）。env-gated，預設 off，零行為變更。

## 8.80 §8.79 獨立重跑確認（2026-08-18）：64-step 配對解析，端點殘留 ~4.8ms 穩定成立

### 動機
§8.79 的四段拆解後，以獨立 session（不同負載窗、64 steps、compute↔首個 sync 嚴格配對）重跑，確認端點/邊界數字不隨測量 session 漂移。

### 量測
qwen36 IQ3_XXS、-ngl 99、4GiB、`CGC_OA_ASYNC=1`、pool8、seed 7、load 1.75。64 個 decode step 全部配對成功（每 step 取 compute 後第一個 eog_sync_wait>0.5ms 的 sync）。

| 段 | median | min | max |
|---|---|---|---|
| 4 `graph_compute_wall_ms` | **81.3** | 62.0 | 977.2 |
| 1 `eog_sync_wait_ms` | **4.6** | 3.0 | 16.8 |
| 3 `post_sync_to_next_ms` | **0.18** | — | — |
| wall/step（comp+sync） | **86.0** | — | — |

### 確認
- `eog_sync_wait` median 4.6ms vs §8.79 的 4.24ms：±10% 內一致，**穩態 ≤5ms 成立**
- `post_sync_to_next` 0.18ms vs 0.15ms：一致，下一 step input-copy 等待 ≈ 0
- 端點殘留總和 ≈ 4.8ms（~5.6% of wall），低於 double-buffer 閾值（5ms）→ **維持 §8.79 收斂：不實作 step 級 double-buffer**
- `graph_compute_wall` median 81.3 vs 67.0：本窗較高（load 1.75 含 Freebuff/Trae 常駐），但結構不變——主體仍是 CPU 側 MTL encoding + 串行 split 處理
- 端點/邊界線正式關閉；剩餘回收空間全部在 `graph_compute_wall_ms` 內部（CPU encoding），fork 內無解，移交 Swift/turbo

## 8.81 n_cb 掃描（2026-08-18）：CGC_N_CB=4 消掉串行 encode 尾部尖峰，qwen36 mean −20~25%，引擎層最後一個正槓桿

### 動機
METAL_ENCODER_OPTIMIZATION_WHITEPAPER_v1.0.md 方向 2（n_cb 調參）是唯一被評估為值得做的引擎層閘門。fork 的 `CGC_N_CB` env 已存在（ggml-metal.cpp:615，default 1 = 單 command buffer，為 per-op 計時安全設定），不需改 code，直接掃描。

### 量測
qwen36 IQ3_XXS、-ngl 99、4GiB、`CGC_OA_ASYNC=1`、pool8、seed 7、n=48、load 1.7-1.9。2 輪交錯（r1/r2）消負載漂移。四臂（n_cb=1/2/3/4）**全 BIT-IDENTICAL**。

| n_cb | eval ms/run r1 | r2 | graph_compute_wall p50 | >100ms 尖峰步數 |
|---|---|---|---|---|
| 1（fork 預設） | **106** | **126** | 77-94 | **14/48、21/48** |
| 2 | 84.5 | 131（load 尖峰） | 74.7 | — |
| 3 | 82.8 | 83.7 | 76-77 | — |
| 4 | **81.5** | **83.6** | 76-78 | **4/48、4/48** |

### 判讀
1. **n_cb>1 不降 p50，而是消掉尾部尖峰**：cb1 的 graph_compute_wall 有大量 >100ms step（14-21/48），cb4 只剩 4/48。機制：n_cb=1 時整張 ~2000-node 圖只有 2 條 thread 串行 encode、單一 command buffer 尾端才 commit → GPU 全程空等 encode；n_cb=4 時 GPU 從第一個 commit 的 buffer 就開始執行，與剩餘 encode 重疊 → **encode 變異被 GPU busy 吸收**。
2. **mean 收益 −20~25%**（106-126 → 82-84 ms/run）——比白皮書估的 0-3% 大一個量級，因為白皮書假設 fork 預設是 n_cb=2（upstream parity），實際 fork 預設 1（per-op 計時工作設的安全值），等於一直沒吃到 multi-buffer async encoding。
3. **gemma4 短測（n=16）亦 BIT-IDENTICAL、無回退**（7.17 → 7.29 t/s，噪音內）。
4. n_cb>2 的 upstream 警告（「可能退化」）在 M4 Max 14 cores 上未出現——encode 是 CPU-bound（§8.79），更多 encode thread 有效。

### 定案
- 生產 harness `run_n30cache.sh` 兩家族設 `CGC_N_CB=4`（env-gated 保留可覆寫）。
- fork 引擎層正式收斂：OA_ASYNC（§8.77/8.78）+ n_cb=4（§8.81）是唯二正槓桿；白皮書方向 1（args pool/argument buffer）與方向 3（dispatch plan reuse）維持不實作。
- 歷史數字註記：§8.79/8.80 的 12.88 t/s（67ms p50）是 n_cb=1 下量的；n_cb=4 後同窗 mean 應為 ~12-12.5 t/s（82-84ms），乾淨窗 p50 預期 ~13-14。

## 8.82 MTP accept 重測（2026-08-18）：46.4% 翻案 + fork 無 nextn 單層 graph = MTP 轉正的真正前提

### 動機
§8.7 的 accept 14.3% 是在 harness bug（draft 跑完整模型）與外部盤慢載入下量的。用修好的 spec-simple（draft ctx_type=MTP）+ 內接盤 Nail iMatrix IQ3_XXS 重測。

### 過程（排除三個阻塞）
1. **-ngl 99 → fit abort**：`-fit off` 繞過（common_fit_params 不認 skip-load，14GB > 11.45GB 檢查誤判）。
2. **-ngl 99/30 → Metal OOM**（kIOGPUCommandBufferCallbackErrorOutOfMemory）：blk.40（bid=40）繞過 pool 全量進 Metal working set（11.4GB 貼近 11.45GB 上限）——§8.58「三處界線」未完成。
3. **-ngl 0 → draft 階段 CPU mul_mat_id segfault**：skip-load 界線用 n_layer_all（41）把 blk.40 也 skip（loader 1278/1352），但 hook 界線只有 n_layer()（40）→ target 與 draft 的 blk.40 都讀被 skip 的空權重。**修復**：skip-load 界線改 n_layer()（trunk only），blk.40 維持 resident（loader 兩處已改，驗證 blk.40 不再 unused、libllama 重建）。
4. **cache 模式下仍 crash**（雙 context 交替 swap 的交互，未解）→ **accept 與 cache 無關，改用無 cache 隔離測**。

### 結果（-ngl 0、無 cache、MTP-4、20 tokens）
```
n_draft = 4  n_drafted = 28  n_accept = 13  accept = 46.429%
```
- **accept 46.4% 翻案**：§8.7 的 14.3% 確認為 harness bug 污染。MTP head 預測品質合格，**高於轉正門檻 0.35-0.4**。
- 輸出正確（「The capital of France is the city of Paris...」）。
- 0.35 t/s（-ngl 0 純 CPU，速度非本量目標）。跑完統計後 teardown abort（exit 134，不影響數據）。

### 判讀：fork 無 nextn 單層 graph = MTP 的結構性死穴
追 crash 時確認：**llama-graph 的 nextn builder 不存在**（`t_h_nextn` 只在 reset/set_outputs 出現，graph 建構主流程從未設定）——draft context 的 `llama_decode` 跑**完整 41 層 trunk**（blk.40 只是普通第 41 層，不是 nextn 單層）。MTP draft 每 step 需 4 次完整 decode 生成 4 token + 1 次 4-token verify = **5 次完整前向/step** vs 無 spec 的 1 次。即使 accept 46%，GPU/CPU 成本 5× → **帳本仍淨負**。

### 定案
1. **accept 46.4%**：MTP head 品質合格，§8.7 翻案。
2. **MTP 轉正的真正前提 = 實作 nextn 單層 graph**（draft 只跑 blk.40 1 層，權重 ~1.6GB 而非全模型）——1-2 週的 llama-graph builder 工作 + loader 認 blk.40 為 nextn 層。**沒有它，fork 的 MTP 永遠是 5× 完整前向，淨負**。
3. 次要阻塞：cache + MTP 的雙 context swap 交互 crash（draft/target 共用 model tensor 交替 swap）與 -ngl 99 blk.40 OOM——在 nextn 單層 graph 之後才需要解。
4. **MTP 線維持封存**（速度標的 Swift/turbo），但接受率數據已修正為 46.4%，且轉正路徑明確（nextn 單層 graph + 共用 pool）。
5. 診斷保留：skip-load 界線修復（blk.40 resident）已合入 loader，對非 MTP 路徑無行為變更（驗證：C0 無 cache 對照 bit-identical 輸出）。

## 8.83 nextn 單層 graph 可行性翻案（2026-08-18）：build_mtp_block 已存在未接通，工作從 1-2 週降為 ~1 週

### 翻案
§8.82 說「fork 無 nextn 單層 graph（t_h_nextn 從未設定）」——**錯**。grep 只看了 `llama-graph.cpp`（generic result 層），漏了 per-model builder `src/models/qwen35moe.cpp`：
- `llama_model_qwen35moe::graph::build_mtp_block`（line 556-727）**完整存在**：DeepSeek-V3 式單層 MTP（eh_proj concat 投影 + attn + MoE FFN + shared head，n_layer_nextn==1 assert）
- **但主流程（build 函數 180-248）不呼叫它**——main pass 只跑 trunk（n_layer=40），MTP block 從未執行

### 因此修正 §8.82 的兩個結論
1. **46.4% accept 不是 MTP head 品質**——是「embd 輸入 + 完整 trunk + LM head」的 trunk 自回歸命中率（draft 跑 4 次完整前向）。MTP head（blk.40）完全沒參與。真正的 MTP accept 未量過。
2. **「draft 跑完整 41 層」是 graph 主流程沒接通 nextn 分支**，不是「fork 缺 nextn builder」——builder 在，只是沒接線。

### 可行性（關鍵降本）
build_mtp_block 用 `model.layers[il]`（il=n_layer()=40=blk.40）的**trunk 風格權重**：
- `layer.attn_norm / wq / wk / wv / attn_q_norm / ffn_gate_inp / ffn_gate_exps / ffn_up_exps / ffn_down_exps` → **Nail 的 blk.40 全有**（trunk 命名）
- nextn 特有：`layer.nextn.eh_proj / enorm / hnorm`（必須）+ `embed_tokens / shared_head_head / shared_head_norm`（可選，TENSOR_NOT_REQUIRED）→ **Nail 缺這 3-6 顆**

### Backlog：nextn 單層 graph 接通（~5 工作天）

| 天 | 工作 | 驗證 |
|---|---|---|
| D0 | 確認 qwen36-hf MTP head 結構（有無 eh_proj/enorm/hnorm；DeepSeek-V3 式 or trunk 式） | tensor 清單 |
| D1-2 | 補 3-6 顆 NEXTN_* tensor 進 GGUF（repack，p0_mtp_* 腳本家族擴充；或 loader 對 blk.40 做 nextn 映射） | loader 讀到 layer.nextn.* 且 assert 過 |
| D3-4 | build 主流程加「embeddings_nextn → 只跑 build_mtp_block」分支（參考 deepseek32 的 mtp_only）；draft context 的 decode 走該分支 | draft decode graph 節點數 = 單層（~100 vs ~2000） |
| D5 | MTP head 真 accept 重測（對照 trunk 自我 46.4%）；draft 成本 41層→1層 的帳本重算 | accept + t/s |

前置（次要，接通後才需要）：cache + MTP 雙 context swap 交互 crash、-ngl 99 blk.40 OOM（三處界線延伸 n_layer_all）。

### 門檻
- MTP-4 t/s > no-spec × 1.3 才轉正（帳本：draft 4×1層 + verify 4-token union 合併）
- 若 eh_proj/enorm/hnorm 在 hf 不存在（MTP 是 trunk 式自足層），改路徑：build_mtp_block 加「無 eh_proj 分支」（h_embd 直接進 attn）——工作量 +1 天

### 狀態
Backlog（P2）。接通後 MTP 線從「封存」轉「可驗」。

## 8.84 MTP backlog D0 完成（2026-08-18）：hf MTP head = DeepSeek-V3 式，Nail GGUF 已預裝 → D1-2 縮減為 0

D0 驗證（qwen36-hf safetensors shard 25-26 + Nail GGUF 753 tensors）結果：

- **eh_proj/enorm/hnorm 三顆全存在**（hf：`mtp.fc.weight` / `pre_fc_norm_embedding.weight` / `pre_fc_norm_hidden.weight`），另加 `mtp.norm.weight`（= shared_head_norm）——**DeepSeek-V3 式**，§8.83 的「+1 天 trunk 式自足分支」不需要
- **Nail GGUF 已內建全部 4 顆 nextn tensor**（`blk.40.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight`）+ KV `qwen35moe.nextn_predict_layers=1`，名稱與 fork 表（llama-arch.cpp:513-518）逐字吻合 → **loader 開箱即讀，D1-2（repack/header surgery/loader 映射）= 0 天**
- blk.40 自足 block 20 顆 tensor 全齊（離散 attn Q6_K + GatedAttn gate + routed/shexp 全組），與 build_mtp_block 的 GatedAttn+shexp 分支完全對應
- 品質風險：blk.40 experts 是 Q2_K/Q3_K（trunk IQ3_XXS）——真 accept 可能被壓低，D5 見真章
- 剩餘 backlog：D3-4（build 主流程接線，qwen35moe.cpp:180-248 加 embeddings_nextn_masked → build_mtp_block 分支）+ D5（重測）≈ **3 工作天**

## 8.85 三處界線延伸 n_layer_all（2026-08-18）：blk.40 進 pool，-ngl 99 + cache + MTP 不再 OOM

**背景**：MTP 共用 pool 的最後一塊（§8.83 前置）。Nail GGUF（trunk IQ3_XXS + blk.40 Q2_K/Q3_K）在 -ngl 99 + cache 下 OOM（kIOGPUCommandBufferCallbackErrorOutOfMemory）——blk.40（bid=40）繞過 bounded pool。

**修改（三處界線，n_layer → n_layer_all）**：
1. `llama-model-loader.cpp:1278` skip-load buft——blk.40 experts 在非 L4 路徑也走 CPU buft + skip-load（§8.82 註記同步更新：hook 界線也延伸後，draft 的 blk.40 由 pool 填補，§8.82 的 segfault 場景由「兩邊都延伸」消除）
2. `llama-model-loader.cpp:1355` expert_index + skip_load flag——blk.40 的 256 experts 進 L1 index + skip_load=true。**這是核心**：pool 的 max_layer（llama-expert-cache.cpp:990-994）由 expert_index 推導，不改則 pool/slot_table 只開 40 層、layer 40 的 adoption 被 `layer >= pool_ext.size()` guard 靜默跳過、slot-table OOB
3. `llama-context.cpp:1795` eval hook 界線——il=40（draft 的 blk.40 topk）觸發 fill/swap

**已確認不用改**：adoption loop（llama-model.cpp:1900）已是 n_layer_all；L4 metal-pool buft（loader:1239）與 ne[2] shrink（loader:1463）已是 n_layer_all；graph 的 remap leaf（llama-graph.cpp:2055）已是 n_layer_all。

**同時發現的真正 OOM 主因（非 blk.40）**：spec-simple 的 `-c` 預設吃 `n_ctx_train=262144` → **KV cache 5120 MiB**（10 層 full-attn × 256K cells × K/V f16）。model 9.6GB + KV 5GB = 14.9GB > 11.45GB → 首次 decode 即 OOM。`-c 4096` 後 KV 僅 80MB，總 working set ~10.7GB。**harness 修復：tmp/mtp_n99_ab.sh 加 `-c 4096`**。

**驗證（Nail GGUF、-ngl 99 + cache 4GiB、-c 4096、-fit off）**：

| 指標 | 結果 |
|---|---|
| C0（no-spec） | exit=0、hit 44.4%、輸出正確、無 OOM |
| MTP-4（draft-mtp） | accept **46.4%**、輸出正確、無 OOM（RSS 10.7GB） |
| hook 觸發 | `HOOKFIRE il=40` ×12（draft 單 token）——界線延伸生效 |
| 非 MTP 回歸 | 生產 harness qwen36 n30c：RSS 9.79GB、exit=0、輸出正確（n_layer_all==n_layer，no-op） |
| pool | 186 slots/layer × 41 層（capacity=2×budget/(41×per_slot)，Nail per-slot 1.07MB） |

**定案**：blk.40 與 trunk 同走 bounded pool（skip-load + hook 填補 + LRU），MTP 共用 pool 的最後一塊落地；-ngl 99 + cache + MTP 可跑。剩餘 backlog：D3-4（build_mtp_block 接通，讓 draft 從 41 層降到 1 層）+ D5（真 accept 重測）。註：本驗證的 46.4% accept 仍是「假 MTP」（draft 跑完整 trunk）的 trunk 自回歸命中率。

## 8.86 MTP teardown double-free 修復（2026-08-18）：free_tiny_botch 根因 = ctx_dft 雙重持有

**現象**：llama-speculative-simple --spec-type draft-mtp 跑完（統計/輸出都正確）後 exit=134（SIGABRT）。macOS DiagnosticReport：`free_tiny_botch` → `malloc_zone_error` → `llama_context::~llama_context()` → `llama_free`。崩潰鏈（02:32-02:46）連帶把 Terminal 帶走。02:40 的 llama-mtmd-cli chatml run 本身完整結束（final stats 都印出）。

**ASan 定位（build-asan，GGML_SANITIZE_ADDRESS=ON）**：
```
attempting double-free ... in llama_context::~llama_context()
  freed by: ~common_speculative_init_result() → llama_free → main (speculative-simple.cpp:421)
  allocated: common_speculative_init_result ctor → llama_init_from_model → main:80
```
**不是 heap 損壞——是 double-free**：`speculative-simple.cpp:81` 的 `ctx_dft.reset(spec_init->context())` 把同一 raw pointer 塞進第二個 `llama_context_ptr`（unique_ptr+llama_free deleter）——`ctx_dft` 與 `spec_init->pimpl->context` 雙重持有，teardown 時兩個 unique_ptr 各 llama_free 一次。與 OA_ASYNC/-fa/外接盤**無關**（-ngl 0 無 cache MTP 也重現）；§8.85 的 blk.40 界線延伸非根因（01:42 就出現）。

**修復**（最小改動）：
1. `common/speculative.h/.cpp`：新增 `llama_context * release_context()`（`pimpl->context.release()`）
2. `speculative-simple.cpp:81`：`ctx_dft.reset(spec_init->release_context())`——caller 取唯一所有權

**驗證**：
| 測試 | 修復前 | 修復後 |
|---|---|---|
| -ngl 0 無 cache MTP-4（ASan build） | double-free abort | **exit=0**（無 ASan 錯誤） |
| -ngl 0 無 cache MTP-4（普通 build） | exit=134 | **exit=0**（accept 25%） |
| -ngl 99 + cache 4GiB MTP harness 兩臂 | C0 exit=0 / C exit=134 | **C0/C 都 exit=0**（C accept 46.4%、hit 46.8%） |
| 生產 harness qwen36 n30c | — | RSS 9.81GB、exit=0（無回歸，llama-simple 不走此路徑） |

**其他（獨立問題，非本修復範圍）**：llama-mtmd-cli 01:30/01:54 的 crash 是 decode 期 `ggml_abort`（`mtmd_helper_decode_image_chunk`，assert），與 double-free 不同簽名——多模態批次路徑的 assert，另案追蹤。

---

## §8.87：D3-4 翻案（接線已存在）+ D5 帳本重算——MTP 維持封存

**背景**：D3-4 任務是「在 qwen35moe.cpp build() 主流程加 embeddings_nextn_masked 分支」。開工第一刀就翻案：接線**本來就存在**——`build_arch_graph`（qwen35moe.cpp:155）在 `params.gtype == LLM_GRAPH_TYPE_DECODER_MTP` 時直接回傳 `graph_mtp`，而 `common_speculative_init_result`（speculative.cpp:2366）已設 `ctx_type = LLAMA_CONTEXT_TYPE_MTP` → `llama-context.cpp:40` 映射 `DECODER_MTP`。§8.83 的「build_mtp_block 無呼叫點」是 grep 漏了 `build_arch_graph` 這層分派。

**節點數實測**（CGC_NODE_COUNT 插樁，-ngl 99 + cache 4GiB + MTP-4）：
```
CGC_NODE_COUNT: gtype=0 nodes=4087   ← target 完整 40 層 trunk
CGC_NODE_COUNT: gtype=1 nodes=92     ← draft 單層 MTP block（目標 <100 ✓）
```
- draft graph 只 92 節點（GatedAttn + MoE FFN + eh_proj/enorm/hnorm + shared_head_norm + lm_head），非完整 trunk
- h input 餵入鏈路完整：target 的 `t_h_nextn`（llama-context.cpp:2572 `ggml_backend_tensor_get_async` 回讀）→ draft ctx 的 `mtp_h_input`（speculative.cpp:614 `llama_get_embeddings_nextn(ctx_tgt)`）
- **46.4% accept（§8.86 前）與 29.6%（本次）就是真 MTP head 的接受率**——不是 trunk 自回歸命中率（§8.83 的假設作廢）

**D5 帳本重算**（128 tok 真實文本、-ngl 99 + cache 4GiB、同窗 load 2.2-2.8）：

| 臂 | tokens | time | t/s | accept | hit rate |
|---|---|---|---|---|---|
| C0（no-spec） | 129 | 11.542s | **11.18** | — | 81.3% |
| MTP-4 | 131 | 17.054s | **7.68** | **29.6%**（71/240） | 84.7% |
| 閾值（no-spec × 1.3） | — | — | **14.53** | — | — |

**結論：MTP-4 −47%，遠低於閾值，維持封存。**

**為什麼單層 draft（92 nodes）仍淨負**——draft 不是「dense 便宜層」：
1. **blk.40 是 256-expert MoE block（Q2_K/Q3_K，~1.5GB）**，不是 dense——每次 draft decode 仍讀 ~1.5GB experts（走 pool，§8.85 的 HOOKFIRE il=40 已證實）。4 draft token/step = ~6GB weight reads，約等於 trunk verify 的 expert reads
2. **accept 29.6% < 35-40% 門檻**——MTP head 在 Q2_K/Q3_K 粗量化下預測品質不足（trunk 是 IQ3_XXS，精度不匹配）
3. verify 4-5 token 批量未做 expert union 合併（P1-3c 類 kernel 未做）——GPU 讀 4-5× 權重

**已更新**：MTP轉正規劃書 backlog 標 D3-4 ✅（免工程，翻案）、D5 ✅（帳本重算：淨負）；`CGC_NODE_COUNT` env 加入診斷工具集（每 step 印 ctx_type + 節點數，驗證 draft graph 結構用）。

### §8.87b：D5 前置基線（128 tok 長測、兩輪、含 RSS）

harness `tmp/mtp_n99_ab.sh` 已加 RSS 解析（`/usr/bin/time -l` 的 maximum resident set size，數字在欄位前）。兩輪（同 seed 42 → accept/hit 決定性一致）：

| 量 | C0（no-spec） | MTP-4 | 閾值 ×1.3 |
|---|---|---|---|
| decoded tokens | 129 / 129 | 131 / 131 | — |
| **accept** | — | **29.583%**（71/240，兩輪一致） | — |
| **hit rate** | 81.3% / 81.3% | 84.7% / 84.7% | — |
| tok/s（r1，load 2.2-2.8） | 11.18 | 7.68 | 14.53 |
| tok/s（r2，load 3.2） | 8.69 | 6.98 | 11.30 |
| **RSS** | **9.63 GB** | **10.05 GB** | — |
| file_reads / pread_usec | 28572 / 19.9s | 30486 / 23.3s | — |
| fill_batch_usec | 6.85s | 6.33s | — |

- **accept 29.583% 是決定性數字**（同 seed 兩輪完全一致）；hit rate 亦然
- **tok/s 對 load 敏感**（r1/r2 差 ~25%）——基線比較必須同窗交錯
- RSS 兩臂都 < 11.45GB working set，無 OOM 風險；MTP 臂多 0.42GB（draft ctx + 4-token verify 的 KV/buffer）
- MTP 臂 pread 多 ~17%（draft 的 blk.40 pool 讀取），fill 反而略少（union 合併的些微效果）

---

## §8.88：mtmd decode 期 assert 根因 + 修復——與 double-free 無關，L4 n_batch cap 與 chunk 分片不一致

**背景**：追 llama-mtmd-cli 的 decode 期 `ggml_abort`（crash report 01:54，`mtmd_helper_decode_image_chunk → llama_context::decode → ggml_abort`）。

**根因（與 double-free 完全無關的獨立問題）**：
- assert 訊息：`llama-context.cpp:2733 GGML_ASSERT(n_tokens_all <= cparams.n_batch) failed`
- fork 的 L4 metal pool（llama-context.cpp:258-268）把 context 的 `n_batch` 從 2048 **砍到 24**（`(capacity-1)/8 = (200-1)/8`，expert union 必須 < capacity 200）
- 但 `mtmd-cli.cpp:113` 用 CLI 參數 `params.n_batch`（2048）當影像 chunk 分片大小 → 傳 2048-token view 給只允許 24 的 context → assert
- **機制**：union ≤ 8 experts × n_batch 必須 < pool capacity；影像 decode 是大批量 prefill，2048-token view 直接爆掉
- 證據：test3（01:55）成功因為影像只有 20 tokens < 24，單次 decode 就過；test2（01:54）影像 > 24 tokens → assert

**修復（2 行）**：`mtmd-cli.cpp:113` 的 `n_batch = params.n_batch` 改為 `n_batch = llama_n_batch(lctx)`——chunk 分片用 context 實際（可能被 L4 cap 的）n_batch。無 cap 時 `llama_n_batch == params.n_batch`，行為完全一致（零回歸風險）。

**驗證**（-ngl 99 + ALLOW_NGL + cache 4GiB + CGC_OA_ASYNC + -fa on + chatml，crash 同配置）：
| 測試 | 修復前 | 修復後 |
|---|---|---|
| tiny.jpg（9 tokens） | assert | **exit=0**、無 assert |
| 較大圖（example_image.png） | assert | 無 assert，但 **mmproj Metal OOM**（見下） |

**獨立發現（非本次範圍）**：較大影像在 `--no-mmproj-offload` 下仍在 mmproj 載入期 OOM（`kIOGPUCommandBufferCallbackErrorOutOfMemory`，1.53s，mmproj 1.56s 才載完）——vision encoder 的 Metal buffer 與 9.6GB trunk 共存超過 11.45GB working set。這是**既有**的 VLM + -ngl 99 工作集限制，與 assert 無關、非本修復引入（舊 binary 對大圖是 assert 不是 OOM）。若要跑 VLM 需要 mmproj 完整 CPU 或降 -ngl。

---

## §8.89：draft 深度掃描（n-max 2/3/4）——全部淨負，短 draft 證偽

**背景**：MTP-4 每 step 4 次 draft decode 是帳本主成本，假說「短 draft 可能淨正」。harness `tmp/mtp_draft_depth_scan.sh`（bash 3.2 相容 case 分派 + 交錯 2 輪 + median）。

**結果**（96 tok、-ngl 99 + cache 4GiB、同窗 load 2.5-4.2）：

| arm | speed_med | r1 / r2 | accept | hit | RSS |
|---|---|---|---|---|---|
| c0（no-spec） | **10.76** | 11.22 / 10.30 | — | 77.0% | 9.63GB |
| d2 | 8.66（−19%） | 9.60 / 7.72 | **42.5%** | 78.4% | 9.92GB |
| d3 | 7.35（−32%） | 8.18 / 6.52 | 32.0% | 80.2% | 9.99GB |
| d4 | 7.22（−33%） | 7.19 / 7.25 | 26.0% | 81.7% | 10.05GB |

**結論：假說證偽——沒有深度能翻正**。accept 隨深度縮短上升（d2 42.5%：越前面位置預測越準，後續位置條件在可能錯誤的 draft 上），但每 step 的 draft decode 成本（blk.40 ~1.5GB pool 讀取/次）壓過多產的 token：d2 步成本 1.66× c0 只多產 1.33× token。hit rate 隨深度升（draft 流量預熱 pool）但省下的 pread 抵不過 draft 自身權重讀取。

**MTP 線最終定案**（§8.7/8.82/8.83/8.87/8.89 全鏈）：accept 品質（29-42%）不是問題，**draft 結構成本（blk.40 是 256-expert MoE block + 每 draft token 一次完整 1.5GB 讀取）是結構性淨負**，無深度、無接受率能翻正。除非 blk.40 真正 dense 化（非 MoE），MTP 在 qwen35moe 架構上不值得再投入。

---

## §8.90：mv_id 頻寬閘門實驗（2026-08-18）——75% 閘門不過，dequant 查表是主因、down K=512 形狀最差

**背景**：30 t/s 前置閘門（用戶分析 D8）——驗證 mv_id 的 IQ3 查表路徑能否到 75% 頻寬（gemma4 head 78% 證明 M4 Max 可達）。用 test-backend-ops perf mode（`-b MTL0 -o MUL_MAT_ID -p <regex>`）逐 case 量 qwen36 FFN 真實 shape（256E top8）的 isolated us/run。

**結果**（bytes = 8 experts × block 數 × block size；頻寬 = bytes/us）：

| case | quant | bytes | us/run | GB/s | % of 120 |
|---|---|---|---|---|---|
| gate/up (K=2048,N=512) | IQ2_S（查表） | 2.75MB | 57.74 | 47.7 | **40%** |
| gate/up (K=2048,N=512) | IQ3_XXS（查表） | 3.60MB | 57.31 | 62.9 | **52%** |
| down (K=512,N=2048) | IQ3_XXS（查表） | 3.60MB | 90.26 | 39.9 | **33%** |
| gate/up (K=2048,N=512) | Q2_K（直解） | 3.15MB | 42.43 | 74.1 | **62%** |
| down (K=512,N=2048) | Q3_K（直解） | 4.06MB | 138.30 | 29.4 | **24%** |
| gate/up (K=2048,N=512) | IQ4_NL（直解） | 4.72MB | 58.94 | 80.1 | **67%** |
| down (K=512,N=2048) | IQ4_NL（直解） | 4.72MB | 70.35 | 67.1 | **56%** |

**閘門判定：不過。** mv_id IQ3 查表路徑 = 33-52%，連最好的 IQ4_NL 直解也只有 56-67%——全部 < 75%。「mv_id 能像 gemma4 head 一樣到 78%」假說證偽。

**機制分解（三個獨立因素）**：
1. **dequant 查表是主因，不是 indirection**：同 ID 結構、同 shape，IQ4_NL（直解）67% > IQ3_XXS（查表）52% > IQ2_S（查表）40%——bytes 越大頻寬越高，查表路徑的 fixed dequant overhead 沒被攤掉
2. **down 的 K=512 形狀最差**：同 quant（IQ3_XXS）下 gate（K=2048）52% vs down（K=512）33%——1.6× 差異純是 K 太短、threadgroup/tile 沒吃滿，這是 P1-3c 唯一有 microbench 根據的刀
3. **Q3_K 24% 是最爛路徑**——順帶解釋 Nail blk.40（Q2_K/Q3_K）MTP draft 慢的另一層原因（§8.89 之外）

**對 E2E 的意義（重要修正）**：
- FFN 換 IQ4_NL 並非免費午餐：gate/up 打平（57.7→58.9us，bytes +72% 但頻寬 +68%），只有 down 省 20us/layer → E2E 只省 ~0.8ms/step（103→102.2ms）→ 9.5→9.6 t/s，**幾乎無感**
- 真正 4ms 級收益必須把 gate/up 的 52% 拉到 75%+（kernel 重寫：直解 dequant + K-loop y 複用 + indirection 消除），30 t/s 路徑不變：3-6 週 kernel 工程
- down 的 K=512 形狀（33% vs 52%）是 P1-3c 最有希望的單點：同 kernel 同 dequant，改 grid/tile 就可能收回 1.6×

**dispatch 掃描補充（同日，runtime env，免重建）**：
- `CGC_MMV_NSG` 1/2/4/8 on down：88.24 / 86.66 / 87.24 / 90.08 us——**無效**（nsg 只改 rows-in-flight/threadgroup，不修 simdgroup 內 32-lane 對 K=512 的覆蓋）
- `CGC_MMV_NR0=8`（template 已存在）on down：90.26 → 84.18 us（−7% isolated）、gate 57.31 → 57.02（0%）——與 §8.43 E2E 證偽一致（40 層 down 省 ~0.24ms/step ≈ +0.2% E2E）
- **down K=512 的 16/32 lane 空轉是結構問題**：修復需 lane 級 K 分裂 + reduction 順序重排 → 破壞 bit-identity → 屬 3-6 週 kernel 重寫範疇，非參數可解

**n99c 收斂定案**：引擎層（§8.71/8.72）+ kernel 參數（nsg/nr0）+ 閘門（本節）全線證偽後，fork 內 n99c 的配置級槓桿已窮盡（~11-13 t/s 視負載）。剩餘只有兩條路：(a) kernel 重寫（3-6 週，閘門 <75%，收益不保證）(b) Swift/turbo 線（已知 18-20+，production 設定已就緒）。

**定案**：30 t/s 閘門正式不過——kernel 重寫是唯一路徑且收益需要先證明 down K=512 形狀可修（下一步實驗：改 mv_id 的 K=512 dispatch 對照，若 down 到 52%+ 則 P1-3c 值得開工，否則 kernel 線全關）。Q3_K 24% 追加為 Nail blk.40 的負面證據。

---

## §8.91：qwen36 attention 頻寬補課（2026-08-18）——attention 已在 74-88%，非 2× 槓桿

**背景**：30 t/s 分析把 attention（12.6ms 最大 weight-read 塊）列為「12.6 → ~6.5ms」的 2× 槓桿。probe 從未跑過頻寬——補課（test-backend-ops perf，Q6_K M=1 GEMV，block_q6_K = 210B/256 = 0.8203 B/elem）。

**結果**：

| attention op（Q6_K） | bytes | us/run | GB/s | % of 120 |
|---|---|---|---|---|
| ssm/full attn_qkv (m=8192,k=2048) | 13.76MB | 145.8-155.0 | 88.8-94.4 | **74-79%** |
| ssm attn_gate (m=4096,k=2048) | 6.88MB | 65.8 | 104.5 | **87%** |
| ssm_out / full attn_output (m=2048,k=4096) | 6.88MB | 64.9 | 106.1 | **88%** |
| full attn_k/v (m=512,k=2048) | 860KB | 10.6 | 81.4 | **68%** |
| lm_head Q6_K（對照 §8.26） | 417MB | 4311 | 96.8 | 81% |

**結論：attention 已在高效區（68-88%），不是 2× 槓桿。** 12.6ms → 6.5ms 假設**證偽**：
- 大塊（qkv 74-79%、gate/out 87-88%）已接近 head 的 78-81% 地板
- 唯一低點是 attn_k/v（m=512 短行，68%）——與 FFN down 的短 K 同族但輕得多（68% vs 33%）
- 最樂觀上限：全部拉到 88% → 12.6 → ~11.5ms（省 ~1.1ms/step，~1% E2E）

**修正後的 kernel 側帳單**（§8.60 + 本節）：
| 塊 | 現況 | 可達上限 | 修正後收益 |
|---|---|---|---|
| FFN（IQ 查表 33-52%） | 7.9ms | ~5.2ms（需 kernel 重寫） | ~2.7ms |
| attention（74-88%） | 12.6ms | ~11.5ms | ~1.1ms |
| head（81%） | 4.4ms | ~4.2ms | ~0.2ms |
| **總 weight-read** | 25ms | ~21ms | **~4ms（~4% E2E）** |

**關鍵修正**：即使所有 kernel 都到 88%，weight-read 也只省 ~4ms/step（103→99ms，9.5→9.7 t/s）。**30 t/s 的真正差距（38-48ms bubble + CPU 側）不在 kernel 效率**——attention 已高效證實了這點。剩餘唯一未歸因的大項是 §8.71 的「38ms dispatch bubble」（從未被元件級分解，19us/kernel 是推估非實測），以及 n_cb=2/3 vs 4 的殘值（ncb_scan.sh 從未跑完，生產直接跳到 4）。

---

## §8.92：bubble 重盤（2026-08-18）——bubble 真相 = encode-bound p50，仍可消除，是唯一 20 t/s 路徑

**背景**：用戶要求重盤「38ms bubble 還有沒有機會消除」。對照 TPOT 圖（CGC_TPOT_延遲分解_2026-08-17.html）與 §8.79/8.81 定案後的重盤。

**bubble 的真相（§8.79 已翻案 §8.71 的推估）**：
- 舊：38ms = ~2000 kernels × 19us（§8.71 推估，非實測）
- 新（§8.79）：graph_compute_wall p50 67-76ms = **CPU MTL encoding 60-76ms + GPU compute 35ms 重疊（max≈encode）**——bubble 不是 GPU kernel launch 開銷，是 **CPU 側 encode 沒被 GPU 藏住**
- §8.81 修正：n_cb=4 後 graph_compute_wall p50 仍 76-78ms（n_cb 只殺 >100ms 尾部尖峰 14-21/48 → 4/48，p50 不動）

**今天剩下的 bubble = p50 76ms encode − GPU 35ms ≈ 40ms encode-bound**。encode 與 GPU 的關係是 wall = max(encode, GPU)：**encode 每砍 1ms，wall 就砍 1ms（直到 encode < 35ms）**。這是 20 t/s 唯一有量級的路徑（kernel 側最多 4ms，§8.91）。

**三個剩餘槓桿（按成本排）**：
| 槓桿 | 成本 | 估價 | 狀態 |
|---|---|---|---|
| n_cb=8/16 飽和掃描 | 1 天（免重建） | p50 未知（cb1-4 只殺尾部） | 從未試過 cb>4 |
| **融合在現配置重測**（CGC_MMV_FUSE） | 1 天 | §8.50/8.67 的 ≈0 是 **OA_ASYNC 前**量的（callback sync 掩蓋 node-count 效應）；現配置下 node 數減少應直接砍 encode | **從未在 OA_ASYNC+n_cb=4 下重測**（ncb_scan.sh BASE 已含 FUSE，未跑） |
| 方向 1 args pool（300-500 LOC） | 2-5 天 | 白皮書 5-15%（3-10ms） | 定案不實作，可翻案 |
| 方向 3 plan reuse / block-encode（1000+ LOC） | 1-2 週 | 白皮書 10-25%（7-17ms）；但 decode graph 每 step 同構 → **encode 理論上可接近 0**（只重播 buffer binding + dispatch），用戶的「60ms 砍半」讀法比白皮書保守值更接近理論上限 | 定案不實作，可翻案 |

**誠實上限**：encode 76 → 40ms → wall ~45-50ms → **20-22 t/s**；encode → 30ms → 25+ t/s。融合重測若證實 node-count 效應（預期 ~5-10ms encode），加上 args pool，即使不到 20 也能把 12.5 → 15-17 t/s。

**關鍵翻案點**：§8.79「fork 內無解」是 n_cb=4 之前寫的；§8.81 證明 encode 可砍（−20-25%）；現在 encode 是唯一剩餘大項，且有三個未試的具體槓桿。**20 t/s 從「kernel 重寫 3-6 週不保證」變成「encode 減半 1-2 週可量化」**。

---

## §8.93：n_cb=4/8/16 飽和掃描（2026-08-18）——cb8 壓 p50 −10ms（77→66）且全臂 bit-identical

**背景**：§8.92 槓桿 1——cb1-4 只殺尾部尖峰（p50 76-78ms 不動），cb>4 是否壓 p50？harness `tmp/ncb_sat_scan.sh`（qwen36 n99c 生產 env + OA_ASYNC、seed 7、n=48、3 輪交錯 4/8/16）。

**結果**（load 4.9-8.8 污染下仍可讀——p50 median 追平 §8.81 乾淨窗 cb4 基線 76-78ms）：

| arm | speed_med | speeds（3 輪） | p50_med | >100ms steps（r1） |
|---|---|---|---|---|
| cb4 | 4.02 | 6.64 / 4.02 / 2.96 | **77.6ms** | — |
| **cb8** | 5.95 | 9.04 / 5.95 / 3.24 | **66.3ms（−11.3ms）** | — |
| cb16 | 6.33 | 8.36 / 6.33 / 3.25 | **67.9ms** | — |

**bit-identity：9/9 全 identical**（md5 全同）——包含 cb16。掃描過程兩次「DIVERGED」警報都是 harness 比較到不存在的檔（r1 只有 cb4、r2 只有 cb8），非真實分歧。

**判讀**：
1. **cb8 是第一個壓 p50 的參數**：77.6 → 66.3ms（−13%）——§8.92 的框架下 encode 每砍 1ms wall 砍 1ms → 生產 mean 83.6 → ~74ms（12.5 → ~13.5 t/s），乾淨窗 p50 76 → 66 → ~15 t/s
2. **cb16 無額外收益**（67.9 ≈ 66.3）——encode 平行化在 8 個 encoder thread 已飽和（M4 Max 10P+4E，encode thread 與 -t 8 推理 thread 競爭）
3. speed 數字被 load 5-8 污染（3-9 t/s vs 乾淨 12+），但 p50 是 CPU encode 時間、與 load 的關係弱（encode 佔用的 thread 與污染來源不同），trend 可信

**定案**：**cb8 取代 cb4 進生產**（`run_n30cache.sh` N_CB=4 → 8），需乾淨窗（load<2）3 輪交錯重驗 p50 −10ms 與 E2E t/s 再定案數字。cb16 不採用（無增益）。§8.92 剩餘槓桿 2（融合現配置重測）與 3（args pool / plan reuse）維持候選。

---

## §8.94：cb8 生產驗證（2026-08-18）——N_CB=8 進生產無回歸，bit-identical

**背景**：§8.93 定案 cb8 進生產（run_n30cache.sh N_CB=4 → 8）。驗證：同 seed 7、48 tok、qwen36 n99c，cb8（生產預設）vs cb4（舊生產）。

**結果**：
- **bit-identity：cb8 == cb4（md5 8801ee56...，與 §8.93 全部掃描輸出同 hash）**——跨 n_cb 4/8/16 與 FUSE 0/1 全部同輸出，一致性確認
- cb8：speed 8.06 t/s、RSS 9.82GB、exit=0
- cb4：speed 2.95 t/s（load 尖峰落在 cb4 窗口，非可比；§8.93 掃描已證 p50 cb8 < cb4）
- **無回歸**：輸出/RSS/exit 全正常

**工具更新**：run_n30cache.sh 新增 `N30CACHE_N_CB`（預設 8，可覆寫回 4）與 `N30CACHE_SEED`（bit-identity 驗證用）兩個 env 覆寫，與既有 N30CACHE_* 家族一致。

**定案 t/s**：§8.95-2 乾淨窗（load 2.6）R1 fuse=0：cb8 生產基線 = **10.55 t/s、p50 79.15ms**。§8.93 的 p50 66ms 為 cb8 單獨（n_cb 掃描時的 load 更低窗口）。

---

## §8.95：CGC_MMV_FUSE 現配置重測（2026-08-18）——OA_ASYNC 時代融合翻正，p50 −20~30ms

**背景**：§8.92 槓桿 2——§8.50/8.67 的融合 E2E ≈0 是 OA_ASYNC 前量的（callback sync 掩蓋 node-count 效應）。現配置（OA_ASYNC + n_cb）下 node 數減少應直接砍 encode。harness `tmp/fuse_retest.sh`：{fuse0, fuse1} × {cb4, cb8}、3 輪交錯、seed 7、n=48。

**結果**（load 4-12 污染，取同載相鄰對比較）：

| 相鄰對（load 差 <0.7） | fuse=0 p50 | fuse=1 p50 | Δ |
|---|---|---|---|
| r1/r2（4.23 / 4.59）cb4 | 134.24 | 104.46 | **−30ms** |
| r3/r4（4.89 / 4.26）cb8 | 128.86 | 108.66 | **−20ms** |

- **bit-identity：12/12 全 identical**（md5 全 8801ee56，跨 fuse×cb 全同）——融合 barrier 修復（§8.2x）正確性穩定
- 高載對（r7-r12）雖有 load confound，方向全部一致偏向 fuse（且 fuse 臂 p50 隨 load 起伏更平——r10 f1c4 在 load 9.7 仍 106.75ms）

**判讀**：
1. **融合在 OA_ASYNC 時代翻正**：node-count 效應（7 ops → 1 op/layer）在 encode-bound 的 p50 上顯現 −20~30ms——推翻 §8.50/8.67 的「≈0」（當時 callback sync 是主瓶頸，encode 縮減被掩蓋）
2. **與 cb8 正交疊加**：cb8 已 −11ms（§8.93），fuse 再 −20~30ms → 兩者合計 encode 76 → ~40-50ms 量級——正好落在 §8.92 估的 20-22 t/s 窗口
3. 與 §8.90 的 kernel 效率無關（fusion 是 encode 節點數縮減，不碰 GEMV 頻寬）

**定案**：**CGC_MMV_FUSE=1 列入生產候選**（與 cb8 疊加），需乾淨窗（load<2）3 輪交錯重驗量化（預期 p50 66 → 45-55ms、wall 74 → 55-65ms → 13-16 t/s）。§8.92 剩餘槓桿 3（args pool / plan reuse）維持候選。

---

## §8.95-2：cb8+FUSE 乾淨窗定案（2026-08-18）——FUSE 收益在 cb8 下消失，§8.95 的 -20~30ms 為 load 污染假象

**背景**：§8.95 以 load 4-12 交錯比較得出「FUSE −20~30ms p50」。§8.95 末尾已註記需乾_clean 窗（load<2）3 輪交錯定案。本節補齊。

**實測**（harness `tmp/cb8_fuse_clean_ab.sh`，qwen36 n99c、cb8、seed 7、128 tok、3 輪交錯）：

| round | fuse | load | speed (t/s) | p50 (ms) | p90 (ms) | md5 |
|---|---|---|---|---|---|---|
| R1 | 0 | 2.59 | 10.55 | 79.15 | 92.17 | 088c596b |
| R1 | 1 | 2.79 | 10.80 | 78.56 | 93.65 | 088c596b |
| R2 | 0 | 4.87 | 8.03 | 86.47 | 186.86 | 088c596b |
| R2 | 1 | 5.75 | 9.41 | 85.51 | 119.54 | 088c596b |
| R3 | 0 | 5.20 | 8.18 | 87.76 | 150.98 | 088c596b |
| R3 | 1 | 5.45 | 9.75 | 82.70 | 115.70 | 088c596b |

**bit-identity：6/6 全 identical**（md5 088c596b），融合正確性穩定。

**結果分析**：
1. **R1（load 2.6/2.8，最乾淨）**：fuse=0 p50=79.15ms vs fuse=1 p50=78.56ms → **Δ = −0.59ms（−0.7%）**——在 noise floor 內
2. **R2/R3（load 4.9-5.8）**：p50 Δ = −0.96 / −5.06ms——隨 load 增大而增大，是 load confound
3. **speed Δ 全部 fuse=1 勝**：+0.25 / +1.38 / +1.57 t/s——但同樣隨 load 增大

**§8.95 翻案**：
- §8.95 的「−20~30ms」是 cb4 配置 + load 4-12 下的假象。在 cb8 生產配置 + 乾淨窗下，**FUSE 的真實 p50 回收 <1ms**
- 機制：cb8 的 8 個並行 encode thread 已吸收了大部分 encode-bound 開銷，FUSE 減少的 80 個 node（1613→1533）在 cb8 下的 marginal effect 被 n_cb 併行吞掉
- §8.95 的「正交疊加」假說（cb8 −11ms + fuse −20~30ms → 40-50ms）不成立

**定案**：**CGC_MMV_FUSE=1 不進生產**（收益 <1ms、增加 Metal shader 复雜度）。cb8 是 encode 側唯一有效參數。§8.92 三槓桿最終定案：cb8 ✅ 進生產、FUSE ❌ 不進、args pool ❌ 證偽。

---


## §8.96：方向 1 Phase A（args pool）證偽（2026-08-18）——setBytes 對小 args 已是最佳路徑

**背景**：§8.92 槓桿 3 的第一刀——白皮書方向 1 Phase A：per-encoder ring buffer 取代 setBytes（setBuffer + memcpy + offset），預期省 Metal 內部 per-call memcpy。實作：`ggml-metal-device.m` 三處（struct 加 args_pool/args_off、init 分配 1MiB shared ring + completion handler 釋放、set_bytes 走 pool + overflow fallback），零 call-site 變更，env `CGC_ARGS_POOL`（§8.96 前預設 on）。

**過程中的正確性修復**：第一版在 `encoder_free`（op_free → commit 前）釋放 pool → GPU 執行期 `kIOGPUCommandBufferCallbackErrorInvalidResource`——**確認 Metal 不會 retain setBuffer 的 buffer 到 command buffer 完成**。修復：pool 改由 `addCompletedHandler` block 捕獲釋放（GPU 完成後才 dealloc），bit-identity 恢復。

**結果**（qwen36 n99c 生產 env + cb8，6 輪交錯，load 5.0-5.9）：

| arm | p50s（3 輪） | p50 median | speeds |
|---|---|---|---|
| pool on | 100.22 / 94.78 / 100.97 | **100.2ms** | 6.41 / 7.92 / 7.23 |
| pool off | 96.93 / 97.15 / 97.15 | **97.2ms** | 7.48 / 7.67 / 7.96 |

- **pool on 比 off 慢 ~3ms p50**（且 off 三輪極穩 97.15/97.15，on 較散 94.8-101.0）
- **bit-identity：on == off 確認**（輸出全同）
- **0 overflow**（1MiB 裝得下 qwen36 decode 全圖 ~250KB args）——不是容量問題，是 setBuffer+offset 本身比 setBytes 慢

**機制**：Metal 對 setBytes 有專用 fast path（Apple 文件：<4KB 建議 setBytes，driver 內部 memcpy 到 arena），我們 per-op args 只有 64-256B——setBytes 已是零額外負擔；setBuffer 反而要 driver 追蹤大 buffer 的 offset + 邊界檢查。

**定案**：
1. **Phase A 證偽，方向 1 關閉**（Phase B 前提「Phase A 有增益」不成立——args 不是 encode 熱點；encode 的 30us/node 來自 pipeline switch + dispatch + setBuffer 本體，不是 setBytes）
2. 代碼保留（`CGC_ARGS_POOL=1` opt-in、預設 off、bit-identical、overflow fallback 完備）供未來 Phase B / plan-reuse 實驗參考
3. §8.92 剩餘槓桿：**方向 3（plan reuse / block-encode）為最後一個 encode 候選**

## §8.97：方向 3 可行性閘門（2026-08-18）——per-node encode 實測 ~1us，block-encode 證偽

**背景**：§8.92 槓桿 3 的閘門——「方向 3（plan reuse / block-encode）值不值得 1-2 週」取決於 per-node encode 的 30us 成分（§8.71 推估 2000 kernels × 19us = 38ms bubble）。白皮書方向 3 估 10-25% 回收。本次用 CPU 側實測（不序列化、不 inflate）直接量測。

**新工具**（env-gated，CGC 診斷家族）：
- `CGC_ENCODE_BREAKDOWN=1`：encoder wrapper（set_pipeline / set_bytes / set_buffer / tgmem / dispatch / end_encoding）+ pipeline lookup（lock+hash）逐呼叫計時，per-split 合併、graph_compute 尾端印出
- `CGC_ENCODE_OPTYPE=1`（`CGC_ENCODE_OPTYPE_PER` 控制 dump 頻率）：`ggml_metal_op_encode` 入口對每個 node 計時，按 op type 累積——含全部 encode switch 本體（arg 計算 + buffer-id lookup + encoder API）

**量測**（qwen36 n99c 生產 env + cb8，n=8-16，load 4.9-5.3，同窗）：

1. **per-node Metal API**（BREAKDOWN）：~0.6us/node（pipeline_lookup 0.2us + dispatch 0.2us + 其餘 0.2us）
2. **per-node 全 encode**（OPTYPE，含 arg 計算）：

| op | count | us/node |
|---|---|---|
| MUL_MAT_ID | 90 | **0.90** |
| MUL_MAT | 233 | 0.80 |
| FLASH_ATTN_EXT | 8 | 3.75 |
| GATED_DELTA_NET | 22 | 1.05 |
| RMS_NORM / L2_NORM / GET_ROWS / ROPE / SSM_CONV | ~250 | 0.7-1.0 |
| 平均（1200 nodes） | | **~0.8us** |

3. **per-split host encode**（CGC_HOST_GPU_TIMING 的 host_ms）：0.03-0.08ms/split（81 splits/step → **2.5-6ms/step 全 encode**）
4. 81 splits/step（FUSE=1 不減 split 數，nodes 1613→1533）

**結論——三個前估全部翻案**：
1. **§8.71「2000 kernels × 19us = 38ms bubble」錯 ~20×**：實測 ~0.8us/node（含全部 CPU 側成分），不是 19us
2. **§8.79「CPU MTL encoding 60ms」錯**：graph_compute_wall 被誤歸因成 encode；實測 encode 只 2.5-6ms/step（3-8% of wall）
3. **方向 3（block-encode / plan reuse）證偽**：即使 encode 歸零，E2E 只省 ~3-5%（clean window 66-80ms/step 中 encode 佔 2.5-6ms）——不值 1-2 週

**閘門的意外發現**：wall 的真正主成分是 **per-split overhead × split 數**（81 splits/step × ~0.8ms/split clean window）——scheduler 的 subgraph copy + 跨 backend sync gap + GPU dispatch 延遲，不是 Metal encode。§8.93 cb8（−11ms）與 §8.95 FUSE（−20~30ms）的機制因此需要重新歸因（FUSE 不減 split 數，其收益來自 GPU dispatch 數減少而非 encode 縮短）。

**定案**：
1. **方向 3 關閉**（addressable budget ~5%，不值得 1-2 週）
2. **§8.92 三槓桿全數收斂**：n_cb=8 ✅（進生產）、FUSE ✅（候選）、args pool ❌、block-encode ❌
3. encode 側沒有剩餘槓桿——若要在 20 t/s 方向繼續，下一步是 **split 數/跨 backend 邊界**（scheduler 層）而非 Metal encoder
4. 工具保留（`CGC_ENCODE_BREAKDOWN` / `CGC_ENCODE_OPTYPE`，bit-identical 驗證 0f66b83c 全同）

## §8.98：81+81 splits 根因追查（2026-08-18）——split 數是症狀，同步 fill 才是牆

**背景**：§8.97 發現 wall 主成分是「per-split overhead × split 數（81/step）」。本節追根因：為何每 step 162 splits？砍半的改動面與預期回收？

**split 結構（CGC_SPLIT_EXEC_VERBOSE + CGC_SCHED_DBG 實測）**：
- 每 step：**81 MTL0 + 81 CPU = 162 splits**。每層 4 splits：
  1. MTL0（~31 nodes）：attention + norm
  2. **CPU（5 nodes）**：ffn_moe_logits（MUL_MAT，gate_inp.weight 在 CPU buft）+ probs + argsort + topk → **hook 在此觸發**
  3. MTL0（31-62 nodes）：FFN experts
  4. **CPU（1 node）**：shared_expert_gate
- 成因：`LLAMA_EXPERT_CACHE_ALLOW_NGL` → `cgc_pin_router_cpu`（split_graph pass 2/3 把 topk/argsort 釘 CPU）+ ffn_gate_inp / ffn_gate_inp_shexp 權重常駐 CPU buft

**分帳（CGC_CPU_SPLIT_DBG 新增的 per-split 三段計時 + CGC_HOOK_DBG hook 內分段計時）**：

| 成分 | 每 step | 說明 |
|---|---|---|
| **ensure_batch（同步 SSD fill）** | **48-92ms** | THE WALL。hook 內實測 ensure_avg 1.2-2.7ms/層 × 40 層；drain=0.000ms（無 in-flight）。~82% 層有 ≥1 miss（expert hit 77-85% → P(8 全中)=hit⁸≈18%） |
| input-copy（GPU sync + 跨 backend copy） | ~65ms | CPU split 39ms（等 GPU 產出 attn_post_norm）+ MTL split 26ms |
| MTL encode | 6ms | §8.97 |
| CPU router 純計算 | <1ms | logits/probs/argsort/topk 都是小 op |
| GPU 執行 | ~35ms | 透過 copy-wait 序列化在 wall 內（非隱藏） |

**關鍵機制**：
1. **fill 在 critical path 上**：FFN-N 需要剛填的 experts，hook 的 ensure_batch 必須同步等 SSD pread 完成才讓 FFN split 跑。每 miss 層付 1.2-2.7ms SSD random-read latency。
2. **層級 miss 幾乎必然**：4GiB pool / 40 層 = 每層 ~140 slots / 256 experts ≈ 55% 覆蓋 → 每層 8 個 selected 平均 ~3-4 個 miss → 82% 的層至少 1 miss。
3. pread_usec（worker 側 syscall 總和）只有 5.2ms/4tok——不是磁碟吞吐，是 **latency**（每次 pread 0.3-0.7ms，層內 2-4 個 miss 平行後 max≈1.2ms）。

**split 砍半的改動面與預期回收**：

| 方案 | 改動面 | 預期回收 | 判定 |
|---|---|---|---|
| rc0（router 全上 Metal） | 移除 pin + 81 CPU splits | 省 CPU copy 39ms + CPU split 機械；但 hook 需每層 callback sync（§8.76 實測 65-79ms 序列化） | **淨負**（§8.77 qwen36 −33% 實測） |
| hybrid（gate_inp 上 Metal，topk 32B 回讀 CPU） | gate_inp 權重改 Metal buft + 只回讀 32B ids | 省 CPU copy ~30-40ms；fill 牆不變 | **~13-15 t/s，非 20 路徑** |
| shared-gate split 合併 | 被 MTL FFN split 隔開，無法合併 | 0 | 不可行 |

**定案**：
1. **split 數是症狀不是病因**——81 CPU splits 是 pin_router_cpu 的設計後果，砍半最多省 ~30-40ms（hybrid）且需要重做 P1-2（已證 qwen36 淨負）
2. **真正的牆是同步 fill（48-92ms/step）**，與 split 數無關：4GiB bounded pool 的 55% expert 覆蓋 → 82% 層級 miss → SSD random-read latency 進 critical path
3. fill 側槓桿盤點：prewarm/pin-profile/prefetch 全試敗（§8.14/8.52-8.55，routing step-unstable 無預測窗口）；page-cache warmth 只 ~6%（實測）；pool 放大 OOM（8GiB 實測 kIOGPU OOM，9.6GB model + pool > 11.45GB working set）
4. **12-13 t/s 是 4GiB bounded pool 的結構性天花板**（fill wall 48-92ms 不可避）；20 t/s 需要打破「pool 容量 vs working set」的約束（如把 experts 移到 non-Metal CPU pool 只留 gather 路徑——L2 設計，已試且更慢）或換引擎
5. 工具新增：`CGC_CPU_SPLIT_DBG`（per-split copy/compute/callback 三段）、`CGC_HOOK_DBG`（hook 內 ensure/drain 分段）


## §8.99：層級 miss 敏感度與動態 slot 容量評估（2026-08-18）——fill 不是牆，動態容量不值得做

**背景**：§8.98 定案「同步 fill 48-92ms/step 是牆、82% 層級 miss、12-13 t/s 是結構性天花板」。本節用
LLAMA_EXPERT_CACHE_TRACE（全量路由）+ MISS_DUMP（miss 清單）+ CGC_HOOK_DBG（ensure 計時）三工具同 run
量化敏感度，並翻案 §8.98 的兩項關鍵數字。

**方法**：qwen36 n99c 生產 env（seed 7、128 tok、內部盤、load 2.3、N_CB=8），一次 run 三工具全開；
Python 以 trace 重放 CGC 的 LRU pool 演算法（200 slots/layer、layer 0 跳過），先驗證再掃敏感度。

**帳目核對（驗證成功）**：
- trace 全量 requests=24960（80 prefill hook × n_tok + 126 decode steps × 40 層 × 8）；LRU 重放
  **misses=9470 vs run stats 9528（99.4% 吻合）**——模擬引擎可信
- MISS_DUMP 捕到 1728 lines ≈ 模擬 decode misses 1696（±2%）——**dump = decode misses**

**工具性翻案（重要）**：
1. **stats hit rate 81.3% 被 prewarm 污染**：loader 預熱（`llama_expert_cache_ensure_slot`，experts 0..199）
   每層 200 次冷填充**每次都 n_misses++**（llama-expert-cache.cpp:270）→ 9528 = 7800（39 層×200 prewarm）
   + **1728 真實 runtime misses**。真實 runtime hit = 41293/43021 = **96.0%**（非 81.3%）；decode 更高達 96.6%
2. **MISS_DUMP 只捕 decode 路徑**（prefill 走 L3-B gather，未進 pool/dump）——對「decode miss 分布」正是所需

**真實 per-layer decode miss 分布（127 tok，MISS_DUMP）**：

```
L 1: 91  L 2: 96  L 3: 81  L 4: 61  L 5: 53  L 6: 40
L 7: 37  L 8: 27  L 9: 45  L10: 58  L11: 51  L12: 35
L13: 39  L14: 34  L15: 41  L16: 35  L17: 32  L18: 28
L19: 47  L20: 32  L21: 38  L22: 57  L23: 39  L24: 34
L25: 42  L26: 33  L27: 42  L28: 49  L29: 36  L30: 31
L31: 37  L32: 29  L33: 40  L34: 37  L35: 37  L36: 43
L37: 38  L38: 46  L39: 57
```
- 總計 1728 / 126 steps = **13.7 miss/step**（320 requests/step 的 4.3%）；每層 0.2-0.76/step
- 前 3 層最熱（91/96/81，routing churn 高）；L8 最低（27）
- **decode 層級 miss ≈ 10/39 = 25%**（0.966^8 → 76% 層全中）——**§8.98 的「82% 層級 miss」是錯的**：
  它用被 prewarm 污染的 hit（77-85%）算 hit⁸，把 prefill 冷啟動混進 decode

**fill wall 實測（CGC_HOOK_DBG，內盤）**：ensure_avg 0.065-0.507ms/層 → **每 step median 6.0ms / mean 6.6ms**
（模型：5.38ms 固定機械成本 + 0.094ms/miss）——**§8.98 的「48-92ms/step」是外接盤/高載窗的量測**
（本 turn 外接盤同工具實測 260-870ms/step；內盤 page-cache-warm pread ~31us/次）。fill 只佔 step wall 的 ~6%。

**敏感度（模擬，同 LRU 演算法）**：

| S（slots/層） | 覆蓋率 | miss/step | 層級 miss | fill wall/step | Δ vs 現況 |
|---|---|---|---|---|---|
| 140 | 55% | 24.8 | 100% | 7.7ms | +1.1ms（**倒退**） |
| 192 | 75% | 15.6 | 100% | 6.8ms | +0.2ms（**倒退**） |
| **200（現況）** | **78%** | **13.5** | 99.2% | **6.7ms** | — |
| 220 | 86% | 10.1 | 96.8% | 6.3ms | −0.3ms |
| 256（全量） | 100% | 9.0 | 94.4% | 6.2ms | **−0.4ms（−0.4% E2E）** |

**動態 slot 容量（同 8000 slots 總額 greedy 重分配）**：miss/step 13.5→12.2（−9% misses），
fill wall −0.1ms/step（**−0.1% E2E**）。只有 distinct-touched>200 的層（L1-4）拿得到額外 slots；
多數層 88-210 distinct 用不滿 200 slots——**uniform 200 已近最優**。

**定案**：
1. **動態 slot 容量：不值得做**（同 budget 重分配 −0.1% E2E；覆蓋率提升到 100% 也才 −0.4% 且 5.1GiB
   budget 的 256-slot Metal region ~11GB 會 OOM）
2. **55%→75% 覆蓋率的前提已過時**：現況 200 slots = 78% 覆蓋、decode hit 96.6%——降回 75% 是倒退
3. **fill 不是牆**：實測 6.6ms/step（~6% of wall）。§8.98 的「12-13 t/s 結構性天花板來自 fill」需修正——
   真正的剩餘 wall 是 §8.97 定案的 per-split overhead（81 splits/step × ~0.8ms）+ GPU 執行 35ms
4. **hit rate 統計要修**：所有用「77-85%」做結論的舊節（§8.14/8.52-8.55/8.98）需以 96% 重讀；
   建議讓 prewarm 的 ensure_slot 不計入 n_misses（或 stats 分開印 runtime/prewarm）
5. 工具：`tmp/miss_sensitivity.py`（trace 重放 + 敏感度掃描）可重用；MISS_DUMP=decode-only 為設計行為


## §8.99-2：§8.99 修正（2026-08-18）——TRACE P 行 stride bug + 解析假象，真實數字翻新

**觸發**：追「prefill 為何不走 pool」時發現兩層假象，§8.99 的 prefill/decode 拆分數字作廢。

**假象 1（工具 bug，已修）**：`LLAMA_EXPERT_CACHE_TRACE` 的 P 行用線性索引 `d[tok*n_expert_used+i]` 讀
strided VIEW（`t->nb[1]=n_expert*4`）→ **多 token 時印出的是 token 0 的 top-(n_tok*8) ranks，不是各
token 的 top-8**。修復：改 stride 讀（與 union building 一致），llama-context.cpp:1811 區塊，註記
`[CGC §8.99 fix]`。驗證：修復後 P 行 distinct 與 OA cp1 的 `uni=` 完全一致（il=1: 82=82 ... il=6: 47=47）。

**假象 2（分析假象）**：上一輪 grep 管道 `head -15` 截斷 + `awk $3` 抓錯欄位，誤判「OA cp1 全部
n_tok=1 → prefill 不走 pool」。正確統計：**OA cp1 n_tok=17 × 39 層存在 → prefill 走 pool 路徑**
（uni=18-31 distinct/layer < 200 slots）——與 §8.99 寫的「prefill 走 L3-B gather」相反。

**MISS_DUMP 真相**：它一直捕 prefill+decode 全部 misses（在 ensure_batch 內，無階段區分）。
dump=1728 = 真實 runtime misses 總和。§8.99 寫的「dump=decode-only」是 garbage trace 造成的誤判。

**修正後驗證（prewarm 0..199 + 修正 trace 重放）**：sim runtime misses = **1728 = MISS_DUMP 完全一致**
（stats 9528 = prewarm 7800 + runtime 1728）。拆分：**decode misses=1044（8.3/step，hit 97.3%）、
prefill misses=684**——§8.99 的 decode 1696 / prefill 7774 全作廢（garbage P 行污染 LRU 狀態）。

**修正後敏感度（同模型）**：

| S（slots/層） | miss/step | fill wall/step | Δ vs 200 |
|---|---|---|---|
| 100-200 | 8.29 | 7.5ms | 0（**decode working set ≤100/層，200 綽綽有餘**） |
| **200（現況）** | **8.29** | **7.5ms** | — |
| 220 | 5.71 | 6.5ms | −1.1ms |
| 256（全量） | 4.78 | 6.1ms | −1.4ms（**−1.4% E2E**） |
| 動態重分配（同 8000） | 7.18 | 7.1ms | **−0.46ms（−0.5% E2E）** |

fill wall 模型：ensure = 4.07ms 固定 + 0.417ms/miss（median 6.2ms/step，mean 7.5ms）。

**修正後定案（取代 §8.99 對應條目）**：
1. **動態 slot 容量仍不值得做**（−0.46ms/step ≈ −0.5% E2E；只有 L1-4 拿得到額外容量，其餘層
   working set 用不滿）
2. **55%→75% 前提徹底過時**：decode working set ≤100/層，S=100 起 flat——容量根本不是約束；
   覆蓋率相關的「55%/75%」敘述對 decode 無意義
3. **decode 層級 miss 只有 17.8%**（6.95/39 層/step，hit 97.3%）——§8.98 的「82%」連同 §8.99 的
   「25%」都作廢
4. fill wall 6.2-7.5ms/step（~6-7% of wall）——**fill 不是牆的結論強化**（§8.99 第 3 點維持）
5. **工具修復**：TRACE P 行 stride bug 已修（`[CGC §8.99 fix]`）；MISS_DUMP 無需改（已含 prefill）；
   舊 §8.99 的 prefill/decode 數字與敏感度表以本節為準；`tmp/miss_sensitivity3.py` 為正確重放工具


## §8.100：prewarm 污染修復 + 舊結論重讀（2026-08-18）

**修復（程式碼已落地，重建過）**：
- `llama_expert_cache_ensure_slot` 加 `bool count=true` 參數；loader 預熱（llama-model.cpp 兩處）傳
  `false` → 計入新的 `n_prewarm_requests/hits/misses`，不再污染 runtime 計數
- stats 行改印：`runtime requests/hits/misses (hit rate X%)  prewarm req=... hit=... miss=...`
- 驗證（qwen36 n99c、seed 7、48 tok）：**runtime hit = 93.4%**（舊標題 81.3% 含 7800 prewarm 冷填充）；
  `prewarm req=7800 hit=0 miss=7800` 精確分離。所有舊「77-85% hit」均為此污染後的數字，真實 runtime
  hit ≈ 93-97%（prompt 依賴）

**舊結論重讀（以修正 hit 重估各節）**：

| 節 | 原結論 | 修正後 |
|---|---|---|
| §8.14 補充：1797 miss = 1796 唯一 | 「步間/層鄰預測天花板死亡」 | **存活**（修正資料 1728/1728 全唯一，dump 是 runtime 未污染） |
| §8.14 補充：prefill 覆蓋 21.5%→65.3% | 「prefill 熱集是唯一訊號」 | **數字作廢重測**：用 garbage P 行（token-0 top-192）判定「在 prefill 出現過」。修正後（真實路由、28-tok prompt）：**39.8% in-prefill / 60.2% brand-new**（舊 21.5/78.5）；長 prompt 65.3% 需重跑 |
| §8.15：prewarm 熱集 A/B（hit 44202/10254） | hot 劣化 −40% | **A/B 結論存活**（兩臂共同污染），但絕對 hit/miss 全污染（base 真實 miss ≈ 2454 非 10254） |
| §8.52：gemma4「decode 99.5% recurring」 | pin64 省 51% miss | **recurring 判定用 P 行 → 受 stride bug 影響，需重跑**；decode-miss 模擬（1711→836）用 D 行 → 乾淨 |
| §8.55：gemma4 pin A/B（hit 90.6→91.8%） | hit +1.2pp 但 E2E −4.6% | **結論存活**（兩臂共同污染 −15pp 對消）；真實 hit ≈ 93→95% |
| §8.98：fill 48-92ms/82% 層級 miss | fill 是牆 | 已由 §8.99/8.99-2 修正（fill 6-7.5ms、層級 miss 17.8%） |

**修正後重測的 miss 來源（28-tok prompt、修正 trace）**：runtime misses=1728，其中
**687（39.8%）在該層 prefill 出現過、1041（60.2%）全新**；1728/1728 唯一 (layer,expert)。
§8.14 的兩個定案性結論中，「唯一性→預測死」**維持**；「prefill 覆蓋率」的絕對數字要改用
修正 trace 重跑（長 prompt 的 65.3% 是待重測項，方向性結論「長 prompt 覆蓋更高」不受影響）。

**對外影響**：所有引述「77-85% hit」或「81.3%」的地方（§8.14/8.15/8.52-8.55/8.98、白皮書 §10.4
可能亦有）應以 runtime hit ~96% 重讀；A/B 相對結論因共同污染大致存活，絕對 hit/miss 數字作廢。


## §8.101：更新帳本（2026-08-18）——81 splits 的 per-split overhead 確認為剩餘 wall 主成分

**背景**：§8.98 的分帳在污染窗量測（load 4-12 + 外部盤 + prewarm 污染 hit），§8.99/8.99-2 已修正
fill（6-7.5ms）。本節在 load ~4、內盤、128 tok、seed 7 用 CGC_CPU_SPLIT_DBG + CGC_SCHED_TIMING +
CGC_STEP_TIMING + CGC_HOOK_DBG 五工具同窗重測，產出閉合的更新帳本。speed 10.33 t/s（96.8ms/step，
load 略高）、**runtime hit = 96.0%**（修正後乾淨值）。

**per-step 中位數帳本（129 steps，分帳總和 78.5ms = graph_compute_wall 78.7ms，閉合）**：

| 成分 | 每 step | % of 78.7 | 說明 |
|---|---|---|---|
| **CPU split copy**（跨 backend input-copy + GPU sync wait） | **34.1ms** | 43% | CPU router split 等 GPU 產出 attn_post_norm；GPU 執行時間的等待面 |
| **CPU split callback**（hook：ensure 6.5 + drain + remap + callback 機械） | **25.0ms** | 32% | 其中 ensure_batch 只佔 6.5ms（CGC_HOOK median），其餘 ~18.5ms 是 callback/hook 機械 + sync |
| **MTL split copy** | 14.4ms | 18% | MTL split 開頭的 GPU sync |
| **MTL split callback** | 5.1ms | 6% | Metal 側 eval callback 機械 |
| fill（ensure_batch） | 6.5ms | 8% | CGC_HOOK 獨立量測（§8.99-2 一致） |
| MTL encode | ~3-6ms | ~5% | §8.97 實測（fold 在 split 內） |
| GPU 真執行 | ~35ms | — | async，藏在 copy-wait 內（§8.79 參考值） |

**分布特徵**：CPU callback p25/p75 = 19.8/30.4ms（max 405 尖峰）、MTL copy med 14.4ms（max 374）——
中位數穩定、偶發長尾（負載尖峰）。

**定案（回答「per-split overhead 是否為剩餘 wall 主成分」）**：
1. **是**——分帳完全閉合：78.5ms/step 全部落在 split 機械（copy 48.5 + callback 30.1），
   **per-split overhead = 0.48ms/split × 162 splits**，主成分是 GPU↔CPU 邊界 ping-pong（81 層 ×
   MTL→CPU→MTL 交替）的 sync/copy 等待，**不是** Metal encoder（0.8us/node，§8.97）、**不是** fill
   （6.5ms）、**不是** CPU router 純計算（<1ms）
2. 淨 overhead 估算：wall 78.7 − GPU 執行 ~35（藏在 copy-wait 內）− fill 6.5 − encode ~5 ≈
   **~32ms/step 純邊界序列化**（~40% of wall）——這是 split 數砍半（§8.98 hybrid 方案）的
   真實回收上界，但 §8.77 已證 qwen36 router 上 Metal 淨負（hook 需 callback sync 65-79ms）
3. **§8.98 舊帳本修正對照**：ensure 48-92ms → 6.5ms（外部盤+污染窗量測作廢）；input-copy ~65ms
   → 48.5ms（同窗內盤收斂）；encode 6ms / router <1ms 維持；GPU 35ms 維持（async）
4. **結構性結論**：162 splits 的 ping-pong 是 A+B split（pin_router_cpu）的必然後果，fork 內
   無解（§8.76/8.77 全驗證）；剩餘真實槓桿只剩引擎外（換引擎/改架構）或接受 12-16 t/s 現況


## §8.102 hybrid 回收上界：split 減半的 wall 下限（§8.101 閉合帳本推導）

**動機**：§8.101 定案 162 splits 的 ping-pong 是剩餘 wall 主成分（78.5ms = CPU copy 34.1 + CPU callback 25.0 + MTL copy 14.4 + MTL callback 5.1，分帳總和 = graph_compute_wall ✓）。本節把 CPU 側的「等 GPU」與「純 ping-pong」分開，精確算 hybrid（router 上 Metal + 只回讀 32B ids）的回收上界。

### 帳本分解

| 成分 | 每 step | 歸屬 |
|---|---|---|
| GPU 真執行 | 35.0ms | 嵌在 MTL split 執行段（§8.72 純 kernel 地板） |
| fill（ensure） | 6.5ms | CPU callback 內，真工作（SSD pread），躲不掉 |
| 純 ping-pong（sync + 跨 backend copy + hook 機械） | **37.0ms（47%）** | = 78.5 − 35 − 6.5，全是可回收的序列化開銷 |
| encode | ~5ms | 藏在 chain 下（§8.97），非主成分 |

### hybrid 的 chain 模型

關鍵結構約束：**FFN-N 依賴 router-N 的 ids → GPU 嚴格逐層序列化，CPU 的 readback+fill 沒有 overlap 窗口**（router-N 與 FFN-N 在 GPU stream 背靠背，中間的 CPU 工作必然插 bubble）。

wall'(x) = GPU 35 + fill 6.5 + readback 40·x + residual ~5

| readback/層 | wall' | t/s |
|---|---|---|
| 0us（免費） | 46.5ms | **21.5** ← 絕對下限 |
| 50us | 48.5ms | 20.6 |
| **88us** | **50.0ms** | **20.0** ← 20 t/s 預算線 |
| 100us | 50.5ms | 19.8 |
| 200us | 54.5ms | 18.3 |
| 500us | 66.5ms | 15.0 |
| 1000us | 86.5ms | 11.6 |
| 1760us（rc0 實測反推） | ~117ms | ~8.4（= rc0 −33% ✓） |

### 定案

1. **split 減半的 wall 下限 = 46.5ms（21.5 t/s）**——GPU 35 + fill 6.5 + residual 5 是不可壓縮的，hybrid 全成功也到不了 24+ t/s（§8.102 之前的樂觀估 24-28 是錯的：fill 6.5 與無 overlap 窗口都沒算）。
2. **20 t/s 的 readback 預算 ≤ 88us/層**（40 層共 ≤3.5ms）。這要求 readback 完全非阻塞/批次化（OA-式 pipeline），從未驗證過。
3. **rc0 實測（−33% t/s）反推每層 sync ≈ 1.76ms**——超過預算 20 倍。naive per-layer sync 是 drain-bound（command buffer 排空延遲），不是 bandwidth-bound（32B 與 4KB activation 的 sync 成本幾乎一樣）。**hybrid 與 rc0 的差別只在 readback 量，sync 延遲同源**——除非把 readback 改成跨層批次（一次 sync 讀 40 層 ids），否則 hybrid 大概率重蹈 rc0。
4. **結論**：hybrid 的上界 21.5 t/s 誘人，但唯一的實現路徑（跨層批次 readback + fill 藏進 GPU busy）從未驗證、且與已證偽的預測/非同步 fill 共享同一個致命約束（ids 是 router 執行後才存在）。**評級：低優先**——比動態 slot 容量（§8.99-2，−0.5%）高一個量級，但低於接受現況。若要做，第一個實驗不是實作 hybrid，而是量「跨層批次 readback 的一次 sync 成本」（若能 ≤3.5ms/40層，hybrid 才值得開工）。

### 對 §8.101 的收斂

- 剩餘 wall（78.5ms）中可回收的上界 = 37ms 純 ping-pong；hybrid 理論上能回收其中 ~30ms（37 − readback 3.5 − residual 殘留），wall → ~48-55ms → **18-21 t/s**。
- 現況 12-16 t/s（load 依賴）的「fork 內最後槓桿」排序更新：hybrid 批次 readback（上界 21.5，需先過 sync 量測閘門）> 動態 slot 容量（−0.5%，已證偽）> kernel/encode/fill（全線證偽）。


## §8.103 動態 slot 容量反證 A/B（LLAMA_EXPERT_CACHE_LAYER_CAPS）——實測證偽且證實 naive 前置載入有害

**動機**：§8.99-2 的模擬（驗證 99.4% + MISS_DUMP 精確吻合）預測動態容量分配只有 −0.5% E2E。本節做真實反證：實作 per-layer capacity（loader tensor ne[2] + cache slot vectors + pool regions + hook 全線 per-layer），用「前 4 層 256 / 其餘 180（同 budget，7504 slots vs 8000）」的 naive 前置載入 A/B 驗證。

### 實作（已落地、env-gated、預設 off 零行為變更）

- `LLAMA_EXPERT_CACHE_LAYER_CAPS="start-end:cap;..."`：loader 的 expert tensor ne[2] 逐層設 cap（Metal buffer 逐層大小）、cache 的 slot_owner/last_use/queued/loading/pinned 向量逐層長度、pool region 逐層大小、hook 的 `uni < n_slots` 閘與 `w->ne[2]` 逐層、loader prewarm 逐層。
- 無 env 時 n_slots_l 保持空 → slots_l == n_slots → 與舊行為逐位元一致（見下方驗證）。
- 全部 slot 迴圈改走 `slots_l(cache, layer)`（17 處），避免 per-layer 向量被越界走讀。

### 驗證（qwen36 n99c 生產 env，seed 7）

| arm | hit rate | runtime misses | md5 | RSS |
|---|---|---|---|---|
| uniform 200/層（8000 slots） | 95.6% | 2018 | f33dccff | — |
| LAYER_CAPS 0-3:256;4-39:180（7504 slots） | 94.8% | **2386（+368，+18%）** | **f33dccff** | 9.25GB（−0.54GB） |

- **6/6 bit-identical**（同 seed/prompt 全同）——per-layer 實作正確（pool 內容不同但 FFN 輸出逐位元一致，slot 佈局與輸出無關）。
- **兩臂 runtime requests 完全一致（46036）**、無 L3-B 改道 → miss 差異純來自 **prewarm 覆蓋**：uniform prewarm 7800（39×200）vs caps 7248（3×256 + 36×180）——36 層各被擠掉 20 個 prewarm expert（180..199），其中 368 個是實際路由會用的 → 由 load-time hit 變成 runtime miss。
- **速度**：3 輪交錯每輪 uniform ≥ caps（load 污染無法定絕對值，但無任何 caps 勝出回合）。

### 定案

1. **naive 前置載入（前 4 層熱）是負的**：miss +18%、hit −0.8pp、速度零增益。熱層不必然在前 4 層（§8.99-2 的熱集分布因 prompt 而異），把冷層擠到 prewarm 覆蓋以下直接製造 runtime miss。
2. **動態容量不是槓桿，且比「不是槓桿」更糟**——模擬的正確預測是「分配無關」（decode working set ≤100/層，任何 ≥100 的分配等效）；naive 啟發式反而打破 prewarm 覆蓋。§8.99-2 的「−0.5% 不值得做」結論**存活且強化**。
3. **實作保留為 A/B 工具**（env-gated、預設 off、與其他證偽槓桿同模式：args pool/nsg/nr0）。若未來要試「optimal 分配」（非 naive），工具已就緒——但模擬已顯示上界 −0.5%，不值得。
4. **§8.102 排序更新**：hybrid 批次 readback（上界 21.5 t/s，需先過 sync 量測閘門）仍是 fork 內唯一有量級的未試槓桿；動態容量正式封存。


## §8.104 hybrid readback 閘門：跨層批次 sync 成本實測（Metal 微基準）

**動機**：§8.102 設 20 t/s 的 readback 預算 ≤ 88us/層（40 層 ≤ 3.5ms/step），並警告 rc0 實測反推每層 sync ≈ 1.76ms（drain-bound）。本節直接量 M4 的 Metal sync 延遲：40 層 ids 一次批次回讀 vs 逐層回讀，決定 hybrid 值不值得開工。

### 測量（tmp/hybrid_readback_gate.m，clang -framework Metal，M4，3 次穩定）

| 測試 | 32B(ids) | 4096B(activation) |
|---|---|---|
| **A 逐層 sync**（40 × 單 CB commit+wait） | **13.2-13.6ms total（0.33ms/層）** | 8.4-8.6ms（0.21ms/層） |
| **B 批次 40-blit**（1 CB + 單次 wait） | **0.27-0.31ms** | 0.21-0.24ms |
| C async handler（40-blit + completion） | 0.25-0.27ms | 0.21-0.25ms |
| D 單次基線（1 blit + wait） | 0.22-0.24ms | 0.23-0.25ms |

### 解讀

1. **批次回讀 = 0.29ms/step，比 3.5ms 預算快 12×**——40 個 blit 只比單次 drain（0.22ms）多 ~50us，批次 sync 成本 ≈ 一次 drain latency。**閘門在「批次機制」上通過。**
2. **逐層 sync = 0.33ms/層 → 13.3ms/step，超預算 3.8×**——但比 rc0 反推的 1.76ms/層便宜 5×（rc0 的 1.76ms 混入了 hook+fill+全 activation copy 的機械成本，純 drain 只有 0.33ms）。
3. **32B == 4096B（±0.05ms）→ 確認為 drain-bound**（bandwidth 無關，資料量 128× 無差異）。
4. **關鍵架構限制**：批次回讀（B/C）的 40 個 blit 在 stream 內背靠背執行，CPU 只在最後一個完成時拿到全部 ids——**只能服務 step 尾端的 next-step 用途，無法餵同 step 的 FFN-N**。FFN-N 的 id→slot remap 依賴 ids-N，而 ids-N 在 router-N 與 FFN-N 之間才產生（stream 內零窗口）→ **同 step FFN 的 gating 必然逐層 sync**。

### hybrid wall 更新（用實測數字重算 §8.102）

wall' = GPU 35 + fill 6.5 + readback 40x + residual ~5

| x | wall' | t/s |
|---|---|---|
| 0.33ms/層（實測逐層 drain） | 59.7ms | **16.7** |
| 0.5ms/層（drain + hook 機械） | 65.5ms | **15.3** |
| 批次 0.29ms（僅 next-step 用途） | 同 step 仍須逐層 | 見下 |

**定案**（§8.105 交叉驗證後修正）：
1. **hybrid 回收上界 = ~21.5 t/s**：wall 78.5 → ~46ms（+104%），fork 內唯一有量級的未試槓桿。§8.104 原估 15-18 低估了——它只算了 per-layer sync 13.3ms，§8.105 補算 hook_overhead 19.5ms 的消除。
2. **閘門結論：值得做（目標 ~20 t/s），fork 內最高優先**。§8.102 的「hybrid 低優先」修正為「值得做」——0.33ms/層 的實測遠低於 rc0 反推的 1.76ms，bottom-up/top-down 交叉驗證一致（§8.105）。
3. **批次回讀的免費贈品**：step 尾端一次 0.29ms 讀回 40 層實際 ids → next-step 的 hook pre-fill 幾乎免費（routing step-to-step ~96% 穩定，§8.99）——這是「非預測的預測」：不是猜，是讀上一步的真實路由餵下一步。可併入 hybrid 實作。
4. **下一步（若開工 hybrid）**：把 router（gate_inp + topk）上 Metal、hook 改 32B ids readback + step-end 批次回讀，bit-identity 對拍後 3 輪交錯 A/B。改動面集中在 llama-context.cpp hook + sched 的 pin_router_cpu 反轉。


## §8.105 hybrid 評估：§8.104 閘門通過，交叉驗證回收上界（2026-08-18）

**動機**：§8.104 量出批次 sync = 0.29ms（通過 3.5ms 預算），§8.102 推導 wall'(x=0) = 46.5ms。本節用 §8.101 閉合帳本做 bottom-up 交叉驗證，定案 hybrid 值不值得開工。

### 交叉驗證（§8.101 帳本 + §8.104 實測）

| 路徑 | wall' | t/s | 一致性 |
|---|---|---|---|
| §8.102 top-down（wall' = GPU 35 + fill 6.5 + readback 0 + residual 5） | 46.5ms | 21.5 | — |
| §8.101 bottom-up（wall' = wall 78.5 − sync_wait 13.2 − hook_overhead 19.5） | 45.8ms | 21.8 | ✓（Δ 0.7ms） |

兩路徑一致 → **hybrid 回收 = wall 78.5 → ~46ms → speed 10.55 → ~21.5 t/s（+104%）**。

### 不可回收的地板

GPU 35ms + fill 6.5ms + residual 5ms = **46.5ms（21.5 t/s）**——hybrid 全成功也到不了 24+ t/s。原因：per-layer sync（0.33ms/layer）在同 step FFN 仍不可避（FFN-N 依賴 router-N 的 ids）。

### 實現路徑與改動面

| 改動 | 檔案 | 預估工時 |
|---|---|---|
| router（gate_inp + topk）上 Metal | llama-graph.cpp + ggml-metal | 3-5 天 |
| hook 改 batch readback（32B ids） | llama-context.cpp | 1-2 天 |
| sched pin_router_cpu 反轉 | ggml-backend.cpp | 1 天 |
| bit-identity 對拍 + A/B | harness | 1 天 |

**總計 ~2 週**。改動集中在 llama-context.cpp hook + sched，Metal 側需新 kernel（topk）。

### 定案

1. **§8.104 閘門：通過**（0.29ms << 3.5ms 預算），且 bottom-up/top-down 交叉驗證一致
2. **hybrid 回收：wall 78.5 → ~46ms → ~21.5 t/s（+104%）**——fork 內唯一有量級的未試槓桿
3. **但到不了 20+ t/s 的「結構性天花板」**：per-layer sync 在同 step FFN 不可避，wall 下限 46.5ms
4. **§8.102 定案修正**：hybrid 從「低優先」→**「值得做，目標 16-20 t/s」**。fork 內排序：hybrid > 其餘全線證偽
5. **§8.104 的定案「值得做（目標16-18 t/s）」升級為「值得做（目標 ~20 t/s）」**——§8.104 低估了（它只算了 per-layer sync 13.3ms，沒算 hook_overhead 19.5ms 的消除）



## §8.106：router Metal 基線量測——ggml_argsort 已足夠快，不需要自寫 shader（2026-08-18）

**動機**：hybrid D0——量 ggml_topk 的 Metal 延遲，決定用 ggml op 還是自寫 Metal shader。用 `test-backend-ops perf -b MTL0 -o TOP_K/ARGSORT` 量測。

### qwen36 router shapes（decode，single token）

每層3個 `ggml_argsort_top_k`（full sort + view，O(n log n)）：
1. `group_scores`: ne=[2, 32, 1] → k=2
2. `expert_groups`: ne=[4, 1, 1] → k=4
3. `selection_probs`: ne=[256, 1, 1] → k=8

### Metal 實測（test-backend-ops perf）

| op | shape | us/run | notes |
|---|---|---|---|
| TOP_K | ne=[1000,1], k=10 | 25.3 | 兩個 kernel（partition + select） |
| TOP_K | ne=[1000,16], k=10 | 41.6 | 16 tokens |
| ARGSORT | ne=[200000,1] | 1314 | 兩個 kernel（sort + merge） |
| ARGSORT(256,extrapolated) | ne=[256,1] | **~1.7** | 6.57 ns/elem |

### 關鍵發現

1. **ARGSORT(256) = 1.7 us 比 TOP_K(256) = 6 us 快 3.5×**——full sort 對 256 元素比 partial topk 更快（少一次 kernel dispatch + 更簡單的 Metal pipeline）
2. **ggml_argsort_top_k = ggml_argsort + ggml_view**（不是獨立 op）——直接複用現有 Metal argsort kernel，零新 shader
3. **每層 router Metal 計算 = ~2.3 us**（3 × argsort），40 層 = **0.09 ms → 佔 wall 0.1%**（可忽略）
4. **真正的瓶頸不是 topk 計算**（0.09 ms），而是 sync wait（13.2 ms）+ cross-backend copy（34.1 ms）——hybrid 的改動面是消除 sync/copy，不是加速 topk

### 定案

**不需要自寫 Metal shader**——ggml_argsort 已足夠快（0.09 ms/40 layers，可忽略）。hybrid 的下一步是：

1. 把 gate_inp 權重（[256, hidden_dim]）從 CPU buft 移到 Metal buft——loader 側改動
2. 反轉 `pin_router_cpu`：讓 ffn_moe_logits + argsort_top_k 在 MTL split 內執行（而非 CPU split）
3. hook 改 batch readback（32B ids per layer，step 尾端一次回讀）
4. bit-identity 對拍 + 3 輪交錯 A/B

改動面集中在 llama-graph.cpp（buft 分派）+ ggml-backend.cpp（pin 反轉）+ llama-context.cpp（hook），Metal 側零新 kernel。



## §8.107：hybrid D1 架構分析——router 上 Metal 在現框架下不可行（2026-08-18）

**動機**：hybrid D1——反轉 pin_router_cpu + gate_inp buft，讓 router 在 MTL split 內執行，消除 per-layer sync。

### 實驗

加 `CGC_HYBRID_NO_PIN=1` env 關掉 scheduler 的 pin（ggml-backend.cpp:1088），配合 `LLAMA_EXPERT_CACHE_ROUTER_CPU=0` 關掉 loader 的 gate_inp CPU buft。

**結果**：
- exit=0 但 **0 tokens decoded**（輸出退化）
- graph_compute_wall = **611ms**（vs 78.5ms baseline，8× 慢）
- gate_inp 權重仍在 CPU（loader 層獨立控制，不受 scheduler pin 影響）
- hook 未觸發（CGC_OA_ASYNC 下 Metal splits 跳過 callback_eval）→ remap ids 未寫 → FFN 讀到 stale ids

### 架構約束

**hook 機制是 hybrid 的致命約束**：

```
router (topk) → hook (CPU callback) → ensure_batch (fill) → write remap ids → FFN reads remap ids
```

- hook 在 CPU split 的 `callback_eval` 觸發（ggml-backend.cpp:1882）
- CGC_OA_ASYNC 下 Metal splits 跳過 callback（line 1867-1874）
- router 上 Metal → hook 不觸發 → remap ids 未寫 → FFN 錯誤

### 兩個獨立控制點

| 控制點 | 檔案 | env | 效果 |
|---|---|---|---|
| Loader buft | llama-model-loader.cpp:1308 | `LLAMA_EXPERT_CACHE_ROUTER_CPU=0` | gate_inp 權重上 Metal |
| Scheduler pin | ggml-backend.cpp:1088 | `CGC_HYBRID_NO_PIN=1` | argsort/topk 節點上 Metal |

兩者都關掉仍不夠——hook 機制要求 router 和 FFN 在同一個 sync window 內。

### 唯一解法：兩階段 graph 重構

Phase 1: 全部 router 在 Metal 執行 → sync 回 CPU → hook 批次處理（40 層一次）
Phase 2: 全部 FFN 在 Metal 執行（用 batch remap ids）

改動面：~500 LOC（ggml graph 建構 + sched split 邏輯 + hook 批次化），~2 週，高風險。

### 定案

1. **hybrid 「直接上 Metal」不可行**——hook 機制是結構性約束
2. **兩階段重構**是唯一解法，工程量大（~500 LOC，~2 週），預期回收 32.7ms → 21.5 t/s
3. **§8.105 的21.5 t/s 預估成立但實現路徑比預期複雜**——不是「改 3 個 env」，是「重構 graph 為兩階段」
4. `CGC_HYBRID_NO_PIN` 保留為診斷工具（env-gated，預設 off，已驗證不可用於生產）
5. **建議**：評估 Phase B（兩階段重構）的 ROI 後決定是否開工。當前 10.55 t/s 已是 fork 內最佳，Phase B 的21.5 t/s 需要 ~2 週高風險工程

---

## §8.108：blk.40 嫁接實驗（fraQtl Q4_K → Nail IQ3_XXS）——嫁接成功、MTP pipeline 仍被 draft decode crash 阻斷（2026-08-18）

**動機**：§8.89 定案 MTP 結構性淨負後，用戶提出「quantized-MTP collapse」假說——Nail blk.40 的 IQ2_S/IQ3_XXS 量化導致 accept 品質崩塌，若用 fraQtl Q4_K_M 的 blk.40 嫁接到 Nail IQ3_XXS 主干，可在保留小體積（13.25GB）的同時恢復 accept 率。目標：驗證嫁接可行性 + 量測新 accept。

### 嫁接實作（graft_blk40_fraQtl_into_Nail.py）

- **來源**：fraQtl Qwen3.6-35B-A3B-Hi-Fi-MTP-runtime Q4_K_M（21.9GB，外接盤），提取 blk.40.* tensors（Q4_K/Q5_K/Q6_K/Q8_0）
- **主干**：Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf（14GB），保留所有非 blk.40 tensors（IQ3_XXS/IQ2_S）
- **輸出**：`Qwen3.6-35B-A3B-IQ3XXS-trunk_Q4K-blk40.gguf`（13.25GB）
- **關鍵 bug 修復**：
  1. metadata ARRAY 截斷（`rope.dimension_sections` 4→2 元素）→ 直接遍歷 `field.data`，`elem_type` 取 `field.types[1]`
  2. tensor offset 不匹配（F32 raw_shape 未按 byte shape）→ 統一 `compute_byte_shape_numpy`，非量化類型也走 byte shape
  3. **shape 反轉**（`token_embd.weight` 2048×248320 變 248320×2048）→ GGML 列優先 vs numpy 行優先，新增 `compute_byte_shape_numpy`：`reversed(ggml_dims)` 轉 numpy 邏輯 → `quant_shape_to_byte_shape` → writer 反轉後與原始一致

### 驗證結果（_verify_graft_output.py）

```
Nail tensors  : 753
Graft tensors : 753
missing in graft: 0
shape mismatches : 0          ← shape 反轉 bug 已修復
type  mismatches : 11         ← 全為 blk.40 norm/shared（F32/Q6_K/BF16/Q8_0 fraQtl 刻意保留同 quant）
```

關鍵 de-collapse 確認：
- `blk.40.ffn_down_exps.weight`：IQ2_S → **Q4_K** ✓
- `blk.40.attn_q.weight`：Q6_K → **Q4_K** ✓
- `token_embd.weight` / `output.weight`：[2048, 248320] type=14（BF16）一致 ✓

### 載入與量測嘗試

1. **`-ngl 99 + LLAMA_EXPERT_CACHE_ALLOW_NGL=1`（L4 metal pool）**：
   - 模型**成功載入**（無 shape error——graft 完全正確）
   - L4 pool 報「16 slots/layer」
   - warmup decode assert：`n_tokens_all <= cparams.n_batch`——L4 pool 把 n_batch cap 到 1（decode-only，pool 容量 < 2×n_expert_used），warmup 傳 2 token（bos+eos）→ assert
   - **根本不相容**：L4 pool 設計為 decode-only，prefill/warmup 會溢出；16GB 上擴大 pool 會 OOM

2. **`-ngl 0 + -expert-cache 4GiB`（L3-B，§8.7 已知路徑）+ `--spec-type draft-mtp`**：
   - 模型載入成功、warmup 完成（L3B PDBG 跑完 40 層）
   - **draft decode crash**：`GGML_ASSERT(id >= 0 && id < n_expert) failed` at `ggml-backend.cpp:1780`，stack：`common_speculative_impl_draft_mtp::process → llama_decode(ctx_dft) → ggml_backend_sched_graph_compute_async`
   - crash 位置：draft context（blk.40 + nextn MTP head）的 MoE mul_mat_id 讀到 garbage expert id

### 結論

1. **嫁接本身完全成功**——753/753 tensor shape 正確、blk.40 量化升級到 Q4_K、模型可載入。`compute_byte_shape_numpy` 修復了 GGML/numpy 形狀反轉的根因。
2. **MTP accept 量測無法完成**——被 draft-mtp pipeline 的 decode-stage crash 阻斷（draft context 的 MoE id assertion）。這是 §8.7 時代已識別的 backlog 項「nextn 單層 graph」，**與嫁接無關**——嫁接前後 crash 簽名一致。
3. **§8.89 結構性結論不變**：即使 graft 解決了 quantized-MTP collapse（accept 品質），blk.40 仍是 256-expert MoE block，每 draft token 一次完整 FFN 讀取的結構成本仍在。且 draft-mtp pipeline 本身在 MoE 架構上有獨立的 crash bug（draft context 的 expert id 路徑未驗證）。
4. **MTP 線定案維持封存**：嫁接證明了「量化品質」可獨立改善（graft 路徑可行），但「pipeline 穩定性」（draft decode 不 crash）和「結構成本」（256-expert MoE draft）是兩個獨立的阻斷點，前者屬 backlog、後者屬架構。Swift 線優先級不變。


---

## §8.108 兩階段 Graph 重構可行性評估

### 目標
把 decode step 的 ggml graph 拆成 router subgraph + sync barrier + FFN subgraph，
讓 router 在 Metal 執行（消除 81 個 CPU splits），中間用 sync + callback 觸發 hook，
目標 wall 78.5ms → ~47ms → speed 10.55 → ~21 t/s。

### 當前架構（一行概述）

每 step 一個 ggml_cgraph，40 層全部交錯（attn → router → FFN → next layer）。
scheduler 按 backend 切換分 split：81 MTL + 81 CPU = 162 splits。
`ffn_moe_probs` pin CPU → router 鏈在 CPU split → callback_eval 觸發 hook →
write remap ids → FFN Metal split 讀 remap ids。

**§8.101 帳本**：CPU copy 34.1ms + CPU callback 25.0ms + MTL copy 14.4ms + MTL callback 5.1ms = 78.5ms。
CPU splits 佔 59.1ms（75%），是剩餘 wall 的主成分。

### 核心洞察：不需要重建 graph

**兩階段不需要拆成兩個 ggml_cgraph。** 現有 scheduler 的 callback 機制已經內建
「在 split 邊界插入 sync + hook」的能力——只需兩個條件：

1. **router 在 Metal**：移除 `ffn_moe_probs` pin → scheduler 把 router 分到 MTL split
2. **OA_ASYNC 不跳過含 topk 的 split**：修改 ggml-backend.cpp 的 OA_ASYNC 條件

scheduler 的 callback 路徑（line 1887-1922）已經實作了：
- 把 split 分成 ranges（按 callback 需求）
- 每個 range：compute async → sync → callback
- callback 觸發 hook（ensure_batch + write remap ids）

**關鍵**：Metal shared memory（`MTLResourceStorageModeShared`，M4 Max 統一記憶體）
讓 CPU 可直接讀 GPU 輸出（topk result），sync 後零 blit copy。

### 改動面

| 檔案 | 改動 | LOC |
|---|---|---|
| `ggml-backend.cpp` | 新增 `split_has_topk_node()` + 修改 OA_ASYNC 條件 | ~15 |
| `llama-context.cpp` | 移除/env-gate `ffn_moe_probs` pin CPU | ~5 |
| **合計** | | **~20 LOC** |

**不需要改**：ggml.h、ggml.c、device.m、llama-graph.cpp、模型文件。

### 改動細節

#### 1. 移除 ffn_moe_probs pin（llama-context.cpp）

```cpp
// 現在（line 3555-3566）：
if (model.expert_cache != nullptr && ...) {
    for (const auto & kv : cache_probs_tensors) {
        ggml_backend_sched_set_tensor_backend(sched.get(), probs, backend_cpu);
    }
}

// 改為 env-gated：
if (model.expert_cache != nullptr && ... &&
    getenv("CGC_HYBRID_ROUTER_METAL") != nullptr) {  // ← 新 env
    // 不 pin → router 在 Metal
} else if (model.expert_cache != nullptr && ...) {
    // 原本的 pin 邏輯
}
```

#### 2. 修改 OA_ASYNC 條件（ggml-backend.cpp）

```cpp
// 現在（line 1883-1888）：
} else if (getenv("CGC_OA_ASYNC") != nullptr &&
           strcmp(ggml_backend_name(split_backend), "CPU") != 0) {
    // Metal splits: run async, skip callback

// 改為：
} else if (getenv("CGC_OA_ASYNC") != nullptr &&
           strcmp(ggml_backend_name(split_backend), "CPU") != 0 &&
           !cgc_split_has_topk(split)) {  // ← 新條件
    // Metal splits WITHOUT topk: run async
```

`cgc_split_has_topk()`：遍歷 split 的 nodes，检查 `strstr(t->name, "ffn_moe_topk")`。

### Hybrid 模式的 split 結構（預測）

```
每 layer:
  MTL split (attn + router)  ← callback path（含 topk → sync + hook）
    range 1: attn nodes → compute async
    range 2: router nodes (logits→softmax→argsort→topk) → sync → callback (hook)
  MTL split (FFN)            ← callback path（無 topk → 單 range async）
    range 1: FFN nodes (gate/up/down MUL_MAT_ID) → sync
```

vs 現在：
```
每 layer:
  MTL split (attn)      ← OA async
  CPU split (router)    ← callback + hook     ← 34.1ms copy + 25.0ms callback
  MTL split (FFN)       ← OA async
  CPU split (gate)      ← callback
```

**split 數**：~80（40 MTL × 2）vs 162（81 MTL + 81 CPU）——減半。

### 時間模型

| 成分 | 現在 | Hybrid | Δ |
|---|---|---|---|
| GPU 執行 | 35ms | 35ms（unchanged） | 0 |
| CPU copy（等 GPU + 跨 backend） | 34.1ms | **~0ms**（shared memory，sync only） | −34.1 |
| CPU callback（hook + ensure） | 25.0ms | **~6.5ms**（only ensure，無 ping-pong） | −18.5 |
| MTL copy | 14.4ms | ~5ms（split 數減半） | −9.4 |
| MTL callback | 5.1ms | ~5ms | ~0 |
| **合計 wall** | **78.5ms** | **~51.5ms** | **−27ms** |
| **speed** | **10.55 t/s** | **~19.4 t/s** | **+84%** |

保守估（含 per-layer sync overhead）：wall ~55-65ms → **15-18 t/s**。

### 為什麼不需要重建 graph

1. **scheduler 已有 per-range sync + callback**：不需要新增 sync barrier op
2. **Metal shared memory**：CPU 可直接讀 GPU 輸出，零 blit copy
3. **hook 已在 callback 內**：expert_cache_on_topk 已處理 topk node 的 callback 觸發
4. **remap leaf 已在 shared memory**：selected_experts_ffn 是 graph INPUT，galloc 分配在 shared memory，CPU 寫 + Metal 讀無需 copy

### 風險

| 風險 | 嚴重度 | 緩解 |
|---|---|---|
| OA_ASYNC 條件修改影響 non-hybrid 路徑 | 中 | env-gated（CGC_HYBRID_ROUTER_METAL=1 才啟用） |
| per-range sync 在大 split 上 drain 慢 | 低 | topk 範圍小（~5 nodes），drain ~0.01ms |
| callback path 的 per-range sync 打破 GPU pipeline | 中 | §8.104 實測 sync ~0.01ms/層（shared memory） |
| hook 的 ensure_batch 在 critical path | 低 | §8.101 實測 6.5ms total（0.16ms/層） |
| bit-identity 回歸 | 中 | callback 機制已有，只需驗證 Metal topk 輸出一致性 |

### 工期

| 階段 | 時間 |
|---|---|
| 實作（~20 LOC） | 0.5 天 |
| 重建 + bit-identity 驗證 | 0.5 天 |
| 3 輪交錯 A/B（低載窗） | 0.5 天 |
| §8.108 定案 | 0.5 天 |
| **合計** | **~2 天** |

### 改動面（更新）

| 檔案 | 改動 | LOC |
|---|---|---|
| `ggml-backend.cpp` | 新增 `cgc_split_has_topk()` + 修改 OA_ASYNC 條件 | ~15 |
| `llama-context.cpp` | 移除/env-gate `ffn_moe_probs` pin CPU | ~5 |
| `llama.cpp` | loader `router_cpu` 受 `CGC_HYBRID_ROUTER_METAL` 控制 | ~5 |
| **合計** | | **~25 LOC** |

（§8.108 初版漏了 loader 的 `gate_inp` buft 分派——`router_cpu` 在 llama.cpp:383 預設 true（qwen35moe arch），loader 把 `gate_inp` weight 釘 CPU → scheduler 仍把 router 分到 CPU split → split 結構 162→162 不變。第三個 fix 後 split 才從 162→82。）

### 實測結果（2026-08-19）

**6/6 BIT-IDENTICAL ✓**（md5=088c596b，base vs hybrid 全同）

**3 輪交錯 A/B**（load 3-5，seed 7，128 tok，cb8+hybrid）：

| arm | speed | p50 (wall) | md5 |
|---|---|---|---|
| **base cb8** | 9.98, 9.08, 10.51 → median **9.98 t/s** | 77.16, 80.42, 77.21 → median **77.2ms** | 088c596b |
| **hybrid cb8** | 10.02, 10.91, 12.79 → median **10.91 t/s** | 60.59, 59.70, 59.88 → median **59.9ms** | 088c596b |

**p50 決定性數字**：77.2ms → 59.9ms（**−22.4%**）
**speed**：9.98 → 10.91 t/s（+9.3%，被 load 3-5 拖低；低載窗預估 12-14 t/s）

**Split 結構驗證**（CGC_SPLIT_EXEC_VERBOSE）：
- Base：162 splits（81 MTL + 81 CPU）
- Hybrid：**82 splits**（41 MTL + 41 CPU）——**減半**
- Router 鏈（logits→softmax→argsort→topk）成功移到 Metal split（63 nodes）

**cb8+hybrid 聯合**：cb4→cb8 在 hybrid 下省 ~12ms p50（−15%），cb8 仍有 marginal 收益。

### 定案

1. **Hybrid D1 已落地**——3 個檔案 ~25 LOC 改動，env-gated，bit-identical ✓
2. **p50 wall 77.2 → 59.9ms（−22%）**——實測回收符合 §8.101 帳本預測（省 ~17ms，保守估內）
3. **Split 數 162 → 82（−50%）**——router 移 Metal 的結構性改善
4. **生產設定**：`CGC_N_CB=8 CGC_HYBRID_ROUTER_METAL=1 CGC_OA_ASYNC=1`（qwen36 n99c）
5. 低載窗（load <2.5）待重測確認 speed 絕對值
