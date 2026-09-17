---
name: cgc-decode-attribution
description: 在 flashkv-devserver（TurboFieldfare / llama.cpp CGC fork）上把一個 decode 步歸因到 GPU 執行、CPU 序列化、池/IO 或編碼，並用既有儀器取得可證的判準數字。當使用者問「decode 為什麼慢」「wait 是什麼」「瓶頸在哪」「要不要做 fusion / kernel 優化」或要求 decode 相位分解、GPU 佔用率、速度槓桿排序時使用。
agent_created: true
---

> **這是快照，不是權威副本。**
> 權威位置：`~/.workbuddy/skills/cgc-decode-attribution/SKILL.md`（由 host 持續寫入）。
> 本檔於 2026-09-18 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
> 底下的內容能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 要改 skill 請改原檔，再重跑 `python3 agent_harness/scripts/import_harness_snapshot.py`。

# CGC decode 步歸因（儀器與陷阱）

專案：`/Users/alexchuang/Documents/flashkv-devserver`
目標配置：prod25 / prefill250，Gemma 4 26B-A4B，M4 Air 16GB。

## 鐵律

1. **每次跑之前 `pkill -9 -f llama-server`**，並用交錯 A/B ×3 + md5 對比。

   ★★ **2026-09-18 修正：交錯 A/B 不夠，要 ABBA（含反序對）。** 同一個形狀上量 MTP，
   用 `off,on,off` 的 A/B/A ⇒ 兩個控制臂 **10.049 vs 6.791**（差 **48%**），而它們的池統計一致到
   0.2%（hit 96.6/96.7%、file_reads 26925/26685）⇒ 漂移是**外部的慢變數**。
   改成**配對**（交替、短臂、對內取比值）之後仍然有偏：**ABAB 的每一對裡 `off` 總是先跑**，
   於是 10 個臂裡「先跑的 5 個」中位 **8.387**、「後跑的 5 個」中位 **9.434**
   ⇒ **臂的位置本身就值 12.5%**。加上反序對（`on` 先跑）之後，正序 ratio 1.071/1.171/1.013、
   反序 0.938/0.701 ⇒ 順序校正後 `M = sqrt(1.071 × 0.820) = 0.937`。
   ⇒ **量兩臂一律 ABBA，並在報告裡寫明「誰先跑」。**
   （附帶定則：臂要短、丟掉 sample 0、用中位數、**計數器優於 t/s**。完整資料見
   `docs/MTP_NET_EFFECT_PAIRED_2026-09-18.md`。）
2. **啟動器有 env allowlist**：`scripts/run_server.sh` 只透傳列出的 `CGC_*`，
   未列出的**靜默丟棄** → 「沒效果」與「沒設到」長得一模一樣。要用任何新開關前，
   先確認它在 `run_server.sh` 裡有 `if [ -n "${VAR:-}" ]; then SERVER_ENV+=(VAR="$VAR"); fi` 區塊。
   歷史上 `CGC_MMV_FUSE`、`CGC_GPU_TIMING`、`CGC_SUBMIT_AHEAD` 都因此踩過。

   ★ **2026-09-17 新增第三種變體：旋鈕的 push 被關在「另一個 profile 分支」裡。**
   `CGC_SERVER_LAYER_CAPS` 在 `run_server.sh` 有讀者（`:421`）、有回顯（`:1033`），看起來完全支援；
   但它的 `SERVER_ENV+=(LLAMA_EXPERT_CACHE_LAYER_CAPS=…)` **原本放在 `if [ "$SERVER_MTP" = "1" ]` 區塊內**
   ⇒ 在 **MTP=0** 的臂（`p25-gputime`、`decode_sweep.py:273`）上**從未被 export**。
   實測：`CGC_SERVER_MTP=1` → 變數有值；`CGC_SERVER_MTP=0` → **空**。
   症狀與上一條**同形**（「沒設到」＝「沒效果」），而且它剛好落在「最想量它的那一臂」上。
   **判準（唯一可靠的）**：**讀引擎自己印的「已解析值」那一行**，不是讀你傳進去的值。
   這裡是 `llama_expert_cache: LAYER_CAPS per-layer caps: total N slots (avg A/layer, min M/layer)`
   —— **它缺席就是旋鈕沒生效**；只要它出現，`N` 就能與算好的計畫逐格對帳（實測 `37×232+2×212+179=9187` 全中）。
   ⇒ 通則：**任何旋鈕都要有一個「已解析值」的輸出，否則你無法區分「沒設到」與「沒效果」。**
   零成本查法：`CGC_DUMP_ENV=1 CGC_SERVER_PROFILE=<p> CGC_SERVER_MTP=<0|1> bash scripts/run_server.sh`
   （印完即 exit，不啟動任何東西）。
   （此條已修：push 移出 MTP 區塊、保留既有預設 ⇒ 對既有配置逐位元等價。）
3. **不要相信 wall-clock 推論**。`compute` / `wait` 都是 CPU 側牆鐘，無法區分「GPU 真的在算」
   與「GPU 早就算完，剩下是啟動／回報延遲」。要區分就用 GPU 端時間戳（見下）。
4. **md5 只在「同 build 指紋 + 同實際生成長度」下可比**。指紋自 2026-09-15 22:xx 起是
   **glob 所有 `libggml*` / `libllama*` shared library + server binary（8 鍵）**，
   由 `decode_sweep.py` 逐列寫入 `build`（`ab_interleave.py` 已改為委派它，不再自己留一份規則）。
   **舊版只雜湊 `server`/`metal`/`llama` 三檔，而閘門修正所在的 `ggml-backend.cpp` 編進
   `libggml-base`（不在那三檔裡）⇒ 修前修後兩跑被判定「可比」**，也就是閘門修正本身在指紋層面
   **不可見**。這不是「不準」，是**反向**的失效：漏檔會讓不同的東西看起來相同。
   長度是 **`predicted_n`**（實際吐出幾個 token），**不是** `--n-predict` 預算：
   `answer_md5_set` 覆蓋整段 completion，所以同一個 `--n-predict 24` 下各臂長度可以是 24/12/7/6，
   那些列的「md5 不同」**同時被長度混淆**，分不清「軌跡分歧」與「同樣前綴、只是提早停」。
   與長度無關的證據是 `sample`（`texts[0][:160]`）前綴。

## 儀器清單（由粗到細）

| env | 輸出 | 作用 |
|---|---|---|
| `CGC_M2_PROFILE=1` | `CGC-M2-FILL` / `CGC-M2-PROF` / teardown `read shape` | 每個 (layer,kind) 的 pool vs disk 位元組、`us/job`、`effective_rate` |
| `CGC_PHASE_TIMING=1` | `CGC-PHASE` | build/alloc/inputs/compute/fill_wait/gpu（每 32 步一行） |
| `CGC_DECODE_PROFILE=1`（`+ALL`） | `CGC-DECPROF` | 每步 wait/cb/submit 三分 + 逐層歸因（top-8 或全部） |
| `CGC_GPU_TIMING=1` | `CGC-GPUTIME` | **GPU 自己的時鐘**：wait / gpu_busy_sum / gpu_union / gap |
| `CGC_GPU_NODES=1`（＋`CGC_GPU_TIMING=1`＋`CGC_DECODE_PROFILE=1`） | `CGC-GPUNODE` | **節點範圍級的 GPU 時間**（2026-09-18 新增）。把每個 command buffer 的 `GPUStartTime/GPUEndTime` **按它編的節點範圍**攤到節點種類上 ⇒ 比逐層細一級。**不需要 `MTLCounterSampleBuffer`**（見下方註）。自我檢查：該行的 `seg_busy` 必須等於同一行的 `layer gpu_sum`（兩條路徑加總同一批時間），`delta` 不為 0 就代表範圍或 buffer↔節點的對應錯了。**`CGC_GPU_TIMING=1` 是必要的**：`dp_lay_gpu[]` 只在它開著時才填，否則分母是 0 而 `delta=0.00%` 是空轉。**2026-09-18 新增三欄**：`wcntw`（只按**會編碼**的節點分攤，見下方「兩張表」）、`*bywork`（按 `wcntw` 排序的前 10 名）、`work-attributed … / residual …`（可主張質量的標頭行） |
| `CGC_GPU_OPS=1`（＋`CGC_GPU_NODES=1`） | `CGC-GPUOPS` | **以 ggml op 為鍵**的第二張表（2026-09-18）。回答名字表答不了的「**這個 op 值不值得動**」：`wcntw` 是 work-weighted 份額、`cntw` 是節點數份額、`ub` 上界、`uni` 只在「整格同一個 op」時才是精確的每節點成本。**只有五個 op 是 no-op**（`NONE/RESHAPE/VIEW/TRANSPOSE/PERMUTE`，`ggml-metal-ops.cpp:242-252` 逐字 `// noop -> next node`）⇒ 它們的 `wcntw` **按建構為 0**，這是證明不是量測 |
| `CGC_CB_N_MAIN=<n>` / `CGC_SERVER_N_CB=<n>` | `n_cb = N` 那行 | **切細 command buffer 的兩個旋鈕**（2026-09-18）。預設 `n_main = MAX(64, 0.1·n_nodes)`（`ggml-metal-context.m:1099`）⇒ **每個 segment 的主執行緒 buffer 永遠吃 ≥64 個節點**，任何住在裡面的桶都被除以 ≥64（這是 `ffn_moe_gate` 的 `cntw` 只有 0.7% 的原因）。`CGC_CB_N_MAIN=1` 免費（cb 數不變）；`CGC_SERVER_N_CB=16` 要付 17 顆 buffer/segment。⚠️ **`n_cb ≥ 64` 在載入期死鎖**（Metal 的在途 command buffer 配額；症狀只有 `/health` 回 `Loading model` ＋ `server never became ready`，與「載入慢」同形，lesson `eng-diag-0035`）。實測上限在 **(64, 129]** 之間 |
| `CGC_SUBMIT_AHEAD=1` | 無（改變順序） | **天花板上界探針**，輸出必然損壞，只用來量上限 |
| `CGC_SLOT_TABLE_GPU=1` | 見 `CGC_S1_DBG` | S1：把 expert→slot 查表搬進圖（`slots = get_rows(table, selected_experts)`），移除每層 host 寫 leaf 的往返 |
| `CGC_S1_MIN_IL=<n>` | — | 只有 layer ≥ n 用 GPU 表（預設 1）。**layer 0 留在 host**：它的 FFN 讀全寬張量 + **原始 expert id**（程式為它寫 **IDENTITY 表**），而 GPU 算出的 ids 需要跨 backend 拷貝（20:18 那次 `libggml-cpu` SIGSEGV 的形狀）。**★ 2026-09-16 更正**：舊理由「它不被池化」已失效——layer 0 自 2026-09-16 起**是池化層**（見陷阱 14/17）。所以這條限制現在只靠跨 backend 拷貝那一半支撐，**能不能下調 `MIN_IL` 未測**；要動就自己跑閘門。**前綴閘的限制**：每個臂都是連續後綴 ⇒ 左界與服務層數同步移動 ⇒ 無法區分「某一層壞」與「服務層數 ≥ N 就壞」；再加同形狀的臂沒有用 |
| `CGC_S1_KEEP_LEAF=1` | 見 `CGC_S1_DBG` 的 `POST ... same=` | **對照臂**：建所有 S1 節點**且**建 host leaf，但讓 `mul_mat_id` 消費 **leaf**。用來切開「GPU 算出的 ids 是錯的」與「多出這些節點本身就會動答案」 |
| `CGC_S1_DBG=1` | `CGC-S1: graph/CAPTURE/POST/EXPECT-*/EQUIV-*` | 打開 S1 的三個 host 側探針（`POST` = 同步後讀回 GPU 產物；`EXPECT` = hook 當下印 `table[ids]`；`EQUIV` = 發表的表 vs `slot_table_safe` 逐專家比對） |
| ~~`CGC_S1_IDENT=1` / `CGC_S1_TAG=1`~~ | — | **已於 2026-09-15 從程式碼移除**（故意寫錯表的分支）。同輪也被移除的臨時診斷：`CGC-S1: CAPTURE`、`CGC-FAST:`、兩處 identity map 的 tag 分支。**`CGC_S1_DBG` 保留為 opt-in。** |
| `CGC_S1_OUT_CAP=1` / `=pre` | `CGC-S1: OUT`（整張 `fnv1a64`）、`CGC-S1: OUTSET`（逐專家列多重集雜湊） | **`ffn_moe_down` 的輸出捕捉**。`=1` 抓 decode（含 graph 序號），`=pre` 抓 prefill。用 `ggml_set_output` 釘住 ⇒ **診斷臂專用、閘門臂永不可開**；成本是「每 graph 釘幾個張量」（見陷阱 15） |
| `CGC_S1_OUT_LAYERS=<spec>` | — | `CGC_S1_OUT_CAP=pre` 時要釘的層（預設 `"0"`；支援 `0-3` 與 `0,4,8,...`）。**層窗是硬需求**：prefill 全 40 層會 fail-stop |
| `CGC_S1_TABLE_CHURN=1` | teardown `S1 slot-table: publishes=… consumed_changed=…`、逐 graph `TABLE-CHURN` | `publish_slot_table` 的發布量與**被消費子集**的變動。**不開時 teardown 會明印 `(consumed-subset churn not instrumented: ...)`**——預設 0 與量到的 0 必須長得不一樣 |
| `CGC_S1_CLAMP_ABORT=1` | `GGML_ABORT` | 對「被消費 id 被 clamp」升為**硬前置條件**（§8.3 的 tripwire，不是修法） |
| `CGC_IDS_CAPTURE=1` | `CGC-IDS-CAP`（dump 於 `ggml_metal_synchronize` 尾端） | **內核側的 ids 讀數**：`mul_mat_id` 的 Metal kernel **實際解碼出的 ids**。不做任何 host 推論，是唯一能同時切開「時序」與「allocator 別名」的儀器。4096 slots、stride 8；dump 用**游標**只印未印過的 slot（分段 dispatcher 會多次 sync）。**注意 stride=8 ⇒ `n_ids=64` 時只看得到 token 0**，看尾部要非零 `n_skip` |
| `CGC_TENSOR_CAPTURE=<精確節點名>`（＋`CGC_TENSOR_CAPTURE_WORDS`，預設 32 上限 32） | `CGC-IDS-CAP path=DST name=<節點>.dst` | **輸出張量**的快照（2026-09-16 新增）。重用 `CGC_IDS_CAPTURE` 的同一個內核（那個內核與 ids 無關，就是「把 stride 個 int32 抄進 slot」）⇒ `.metal`／`impl.h`／`device.*`／`context.m` 一行未改，比較器也原封不動。**必須與 `CGC_IDS_CAPTURE=1` 一起開**：ids 列是比較器切 graph 的依據，少了它們 dst 列會落進一個偽 graph 而被報成 IDENTICAL（假陰性）。**只掛在 `ggml_metal_op_mul_mat_id`** ⇒ 只能擷取 gate／up／down 的輸出。名稱**精確比對**（`ffn_moe_down-1` 是 `-10..-19` 的子字串）。★ **需要一個 `ggml_metal_encoder_memory_barrier`**：ids 運算元的生產者在很多節點之前（已沉降），dst 的生產者是**緊鄰的前一個 kernel**，而這個 fork 支援內核併發。沒有屏障時讀數**不可重現** |
| ~~`CGC_S1_OUT_CAP=1`／`=pre`~~（**既有，較舊**） | `CGC-S1: OUT`（`fnv1a64` ＋ `vals=[...]`）、`OUTSET` | **同一件事的舊版**：釘住 `ffn_moe_down` 的輸出。差別有兩點——(1) 它用 `ggml_set_output` **釘住張量**，而那會動 allocator／graph（原始碼自己寫「診斷臂專用、閘門臂永不可開」）；(2) 它印**雜湊**加少量值。⇒ **要「不擾動圖」的讀數用 `CGC_TENSOR_CAPTURE`，要層窗與雜湊用這一支。** ⚠️ 2026-09-16 交叉確認**未成立**：`CGC_S1_OUT_LAYERS=1` 跑出來的列是 `il=27` 而不是 layer 1，而 480/480 全不同 ⇒ **它的參數語意尚未弄清，不要拿它當獨立確認** |
| `GGML_SCHED_DEBUG=2` | per-node 後端 + `GET_CAUSE` | **唯一**能回答「這個 node 落在哪個 backend、它的 src 從哪來」的儀器（=1 只有 `## SPLIT`，且走 `GGML_LOG_DEBUG` 會被預設 verbosity 濾掉） |

**★ 節點級 GPU 時間是怎麼來的（2026-09-18；別再重推一次，也不要以為它需要 `MTLCounterSampleBuffer`）。**
`docs/M3_VERDICT_2026-09-17.md` 說「逐節點 GPU 時間是唯一活路、要用 Metal counter sample buffer、成本最高」
—— **活路對，成本判斷錯**。`ggml-metal-context.m:1304-1315` 的編碼回呼**本來就把圖切成固定範圍**：
主執行緒的 slot `n_cb` 編 `[0, n_nodes_0)`、worker `cb_idx < n_cb` 編
`[n_nodes_0 + cb_idx·p, …)`（`p = n_nodes_per_cb`），而 `n_cb / n_nodes_0 / n_nodes_per_cb`
在 sched 側讀取時**都還在** ⇒ **範圍可以直接導出，零新狀態、無抽樣點、無 barrier**。
既有的 `CGC_GPU_TIMING` **早就**逐 cb 讀 `[cb GPUStartTime]`／`[cb GPUEndTime` 的配對
（`ggml-metal-context.m:503-541`），只是把它們聚合成一個 segment 跨度（`union` 的定義在 `:497`）
⇒ **節點身分是在「聚合」那一步丟掉的，不是「取樣」那一步。** 實測指紋：
`bufs ÷ segs = 360 ÷ 40 = 9 = n_cb + 1`（env `CGC_N_CB=8`）⇒ 一層 9 個切片。
**★ 兩張表的關係、以及「份額不是粒度不變的」（2026-09-18；引用任何份額前先讀這裡）。**
- **名字表**（`CGC-GPUNODE`，按節點名前綴分桶）與 **op 表**（`CGC-GPUOPS`，按 ggml op 分桶）
  是**同一批 buffer 的兩種鍵**。它們的 `wcntw` 在**絕對值**上一致到 0.1%（實測 ms-ratio
  1.001／1.000：名字表 `ffn_moe_gate+up+down` vs op 表 `MUL_MAT_ID`）——**這是實作正確的證據，
  不是權重良置的證據**。⚠️ 兩表的**百分比不可比**：名字表除以 `ns_total`（每一個 buffer），
  op 表除以 `nsop_total`（有 ≥1 個可解析 op 的 buffer）。
- **權重是什麼**：`dur(buffer) × (#該桶且會編碼的節點) / (#會編碼的節點)`。
  它修掉的是「**no-op 稀釋**」（`n_main ≥ 64` + 約 40% 的節點是 no-op ⇒ 一個桶可能被除以 64；
  實測 `ffn_moe_gate` 的節點數份額 0.7% 而上界 44.7%）。**它假設每個會編碼的節點等成本**，
  所以它是**共同出現（co-location）的排名、不是成本排名**。
- **⚠️ 份額不是粒度不變的**：同一 profile、同熱態，只改切片寬度（`en-work` 預設 vs
  `en-work-fine`＝`CB_N_MAIN=1`+`SERVER_N_CB=16`）⇒ `moe_gemv` 3.2%→6.3%（1.97×）、
  `dense_gemm` 2.2%→4.2%（1.91×）、`conv/ssm` 2.5%→0.7%（0.28×）；穩定的只有
  `node`／`ffn_moe_`／`norm`／`(other)`／`ffn_`（0.84–1.06×）。
  ⇒ **規則：只引用「家族總和」；逐項只在兩端差 <1.3× 時才講成排名，否則報 `[粗, 細]` 區間。**
  （lesson `eng-mh-0064`。熱態是**另一個軸**：NOMINAL→HEAVY 對 11 個家族是**均勻**的 0.81–0.88。）
- **`residual`（標頭行）**＝整格沒有任何有名工作節點的 buffer 質量 ⇒ 四臂實測 19.9–23.9%。
  **它不是熱態的量**（HEAVY 細臂 20.0% ＝ NOMINAL 粗臂），是**切片組成**的量。
- **兩個已證的識別陷阱**：(1) `ffn_moe_gate/up/down` **就是** MUL_MAT_ID（每桶 80 節點、只有這個 op），
  但 **`ffn_moe_topk` 是 VIEW**（`op=38`, `ne=[8,2]`）⇒ 它的 `wcntw` 恆為 0.00 **是正確的**；
  真正的選擇成本在 `ffn_moe_argsort`（ARGSORT）與裸名 `top_k`（TOP_K）。
  (2) `node`／`(other)` 的差距不是詞表缺陷：`node` 620 節點**全部會編碼、零 no-op**
  （ADD 270／MUL_MAT 130／GET_ROWS 90／MUL 60／GATED_DELTA_NET 30／FLASH_ATTN_EXT 10）、
  `ffn_moe_` 680 節點只有 280 會編碼、`cache` 660 只有 290。**所以前三名是排名，不是偽影。**
- **千萬不要**：把 `sum(wcntw)`（＝`seg_busy` 的 76–83%）當步時長。`seg_busy` 本身隨切片數
  膨脹（同一 workload：143／296／766 ms @ n_cb = 8／16／63）⇒ **絕對 `seg_busy` 不可引用**。
  分母請用**同一步**的 `wait`（步配對），不要拿兩個中位數相除。

**兩個已知限制（先讀再引用）**：(1) ~~詞彙表尚未調校~~ **2026-09-18 已解決**：詞表是 40 條固定前綴，
`ffn_moe_gate/up/down` 在表上；但**桶名仍是匹配規則的產物**（`ffn_moe_topk`＝VIEW 就是反例）
⇒ 要用 `CGC-GRPH` dump 或 op 表交叉確認成員（lesson `eng-lf-0008`）。
(2) ~~攤分是按節點數加權~~ **2026-09-18 已由 `wcntw` 取代**（只按會編碼的節點分攤）；
但 `wcntw` **不是粒度不變的**（見上）⇒ 只引用家族總和。
實作上的坑：`ggml_metal_cgc_node_name()` 讀的是**快照**（在 `graph_compute` 內複製的節點名指標）——
`ctx->gf` 是**呼叫者**的 cgraph，而分段 dispatcher 把它建在區域變數上（`submit_seg` 的
`struct ggml_cgraph gv = seg_view(s);`）⇒ 呼叫一返回就懸空（lesson `eng-diag-0034`；症狀是載入期
SIGSEGV 11，而既有路徑完全正常）。**這支是全檔唯一在「呼叫之後」讀 `ctx->gf` 的讀者。**

**★ 不要為了 L0 去動 `CGC_S1_MIN_IL`（2026-09-18 實測；這是第二個理由）。**
上一列已經說了「layer 0 自 2026-09-16 起是池化層」，另外兩點：`CGC_S1_MIN_IL` **只在
`CGC_SLOT_TABLE_GPU` 存在時才被讀**，而 `run_server.sh:1518` 只在顯式給值時才傳它
⇒ **生產沒開 S1，它在生產上是 no-op**；而且 `decode_sweep.py:410-412` 把 L0 明文列為永久排除項
（它的 MoE 跑 CPU/BLAS，跨 backend 拷貝就是 20:18 SIGSEGV 的形狀）。**更重要的**：
「L0 的 GPU 時間是中位的 6~7 倍」**只在 llama-bench `-d 512` ＋ MTP env 開但沒有真的投機**那個
regime 成立（`llama_bench_matrix.forward_argv()` 不轉發 `--spec-type` ⇒ 它的 decode step `ntok` 恆為 1）。
在真的 decode 裡（`MTP off` 與 `MTP on+投機` 兩臂、各 42／137 個 steady step）L0 的 gpu 是
**0.46×／1.6×**，而它的 `union` **低於中位** ⇒ **L0 沒有可重現的異常，不要去追它。**
（`cb` 的前四名兩臂都是 L2/L1/L0/L3 —— 那是**池填充由最前面幾層付錢**的既有模式，不是 L0 的性質。）

**★ `read shape: … effective_rate=` 不是裝置速率（2026-09-17 22:4x 實測）。** 那一行是 `bytes/jobs/us_job`
的**導出值**，而 `us/job` 含 **fill worker 的排隊／鎖／CPU 調度**，不是 SSD 延遲：同一形狀
（散佈 0.17 MiB）**裝置實測每個 job 只要 0.24 ms（p90 0.34）**，而 log 上同一輪寫的是
**`us/job=19406`（19.4 ms）＝ 80×**。它的算術也自相矛盾：`jobs × us/job`（26883 × 19.4 ms = **521.7 s**）
**大於**該 cell 的 wall（**89.4 s**），因為 8 個 worker 並行（`pread_usec` 也正好是 521.7 s）。
⇒ **不要用 `effective_rate` 或「非居民 share」推論 IO 瓶頸。** 要判 IO，折算**每 token 的磁碟位元組**
（該輪 4.54 GiB / 512 token = **9.1 MiB/token**）× **裝置實測速率**（散佈 **639 MiB/s**、循序 **2.26 GiB/s**）
＝ **14 ms/token**，對比 209 ms/token 的預算 ⇒ **約 7%**；而同一輪 `prefill-house` 同樣 44.5% 非居民卻跑
**254 t/s**，是這個假設的直接反例。量法：`Backup/cgc_logs/en_io_probe_20260917.py`（唯讀，選沒被載入過的
大檔以免 page cache 假象；重讀若快一個數量級就代表前兩項不是裝置速率）。**`fill_wait=0` 是對的**：
消費者沒有在等 —— 這不等於 IO 不在成本裡，但這個案例裡它真的不在。

**★ `CGC-SEG` 怎麼讀（2026-09-17 用它把一個 209 ms/token 的帳拆平）。** 那裡印的是**每段平均**
（`ggml-backend.cpp:1848-1855` 註解原文）：分段迴圈把 **GPU 層 i → CPU top-k hook → submit 層 i+1**
序列化 ⇒ **step 的 wall = Σ(wait+cb+submit) over layers**；每行平均 **160 段 ≈ 4 steps** ⇒ **1 段 = 1 層**，
括號裡的數字是**累計段數**（÷40 = 步數，可與 reps×n_gen 對帳 ⇒ 能分辨「有沒有把某個 pass 算進去」）。
`cb` 是「**下一層的 top-k hook：slot 管理 ＋ 阻塞式 pool 填充**」——**不是 GPU**；`submit` 是 CPU 編碼。
判讀用**中位數**（第一段含冷池，`cb` 可貴 3 倍）。實例：96 行 × median(3021/1003/248 µs) × 40 層
= **171 ms/token**，對上同輪實測中位 rep **178 ms**（差 4%）⇒ 帳是平的，**可以直接這樣引用**。
**但它拆不開 `wait`**：wait 只說「CPU 在等 GPU」，要知道 GPU 是忙是閒必須**同時**開 `CGC_GPU_TIMING`
（`gpu_busy_sum` / `gpu_union` / `gap`）。⚠️ 這個迴圈只在 `CGC_OA_ASYNC=1` 下可達；沒有它整個 graph 是一次
async submit，三個成分都不可分。

跑法：`/opt/homebrew/bin/python3 scripts/check/decode_sweep.py --profile prod25 --arms <臂>,<臂> --rounds 1 --warmup 1 --n-predict 120 --json Backup/phase_decomp/x.json`
（`--profile` 吃的是 `CGC_SERVER_PROFILE`（如 `prod25`），**不是**矩陣臂名 `prod25-stream`。）
閘門：`scripts/check/m123_oracle_gate.py`（M1/M2/M3 bit-identical + 最新 M2 oracle）。

## 判準（已校準，直接照用）

- `gpu_union / wait ≥ 70%` → 傾向「執行受限」；`≤ 40%` → 傾向「序列化受限」。
  **但先讀陷阱 1**，`union` 的結構性偏誤會把兩種情況都推向 ~90%。
- **`gap`（相鄰段 `start_{i+1} − end_i`，GPU 時鐘）是唯一乾淨的證據**：
  `gap ≈ 0` ⇒ GPU 被餵飽；`gap` 佔 `wait` 的 20–35% ⇒ 有可證的閒置。
- `gpu_busy_sum / gpu_union ≈ 1` ⇒ 一個段的 `n_cb+1` 顆 buffer **幾乎序列**
  （此時 `n_cb` 掃描必然無效，兩者互相印證）。
- **最強的一招是上界探針**：把可疑的序列化用 `CGC_SUBMIT_AHEAD=1`（或任何等價的
  「移除該窗口但語意錯誤」開關）整段刪掉量一次。
  「第二個在飛的東西不可能讓真實計算變快」⇒ 若步時間下降 X%，就有 X% 是序列化。
  這比任何相位分解都決定性。輸出損壞是**預期**的，md5 必須變，否則代表旗標沒生效。
- **★ 一個新的歸屬份額（任何 weight／攤分模型）在能被引用之前，先過「換粒度」測試**：
  同 profile、**同熱態**（讀 `thermal_launch`！）跑粗／細兩臂，逐項比。
  不通過的項目就只報家族＋區間，不要報點估計（lesson `eng-mh-0064`）。
  同粒度下 run-to-run 穩定（實測 11 個家族 0.89–1.02）**不代表**跨粒度穩定。
- **M3 的 40% 判準已經有三個獨立方法都是否定的**（2026-09-18）：聯集上界 ≤28.3% of busy、
  op 表 work-weighted 8.0／8.4%、名字表 work-weighted **3.2–6.3% of `wait`**。
  ⇒ **Cell 2（MoE gather 融合／batched-union）沒有量，不要再投。**
  而「MoE 的**逐元素合併**（`ffn_moe_`，7.6%）比 MoE 的**專家矩陣乘**（3.2–6.3%）還大」。

## 唯一的 decode 儀器：`llama-bench`（2026-09-17 **使用者裁定**；舊標題「兩個 decode 儀器不能並排」）

**`decode_bench` 已退休 —— 不要再量、不要再引用它的數字。** 理由是 2026-09-16 的實測：同 env／
同模型／同 n／同場交錯，`llama-bench` 與 `decode_bench` 差 **9.4 vs 12.4**（n≈128）與
**7.2 vs 18.7**（n=24），而且 `decode_bench` 的離散大得多（12.36／12.95／10.64 vs llama-bench
三次 9.42／9.32／9.38 ＝ **1.1%**）。與其維護兩套互不並排的口徑，**現在只認 `llama-bench`**。
（細節 `docs/INSTRUMENT_COMPARE_20260916_1821.html`）

**標準形狀 —— 報任何 decode 數字都要附這三樣：reps 數、warmup 規則、模型家族。**

- **`-d 512`**：`--depths 0` 是最冷的格子，**不要拿它當標準**。depth 的預填充在 `t_start`
  之前完成、不計時 ⇒ 只有 `-d ≥ 512` 才是暖平台。
- **warmup 保持 ON、並丟掉 rep1**（llama-bench 的 tg warmup 只有 1 個 token，
  `llama-bench.cpp:2392`）⇒ **報平台值，不報 `avg_ts` 的原始平均**。
- **模型家族必須指名，而且要看 `-m` 那一行、不是看臂名。**（2026-09-17 更正）
  `MTP=0` 確實會換檔案，**但觸發條件是「顯式設 `CGC_SERVER_MTP=0`」而不是 profile 名**：
  16:1x 用 `prefill_certifiability.py --dry-run` 逐字讀它印出的 `-m` 行，
  **`--arm prefill250` 與 `--arm prod25` 都載 `Nail-…-denseIQ4X.gguf`**；
  `ARMS` 表裡**只有 `prod25-stream-mtpoff` 設了 `CGC_SERVER_MTP`** ⇒ 只有它換成
  `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`。
  **⚠ `tag` 不含模型檔**（json 只存 tag）⇒ **每份產出都要記 `-m` 那一行**（matrix 已經會印），
  否則事後無法分辨。**後果（好消息）**：`prefill250` 與 `prod25-stream` 血統**同一個檔** ⇒
  decode 的 10.8–10.9 與 prefill 的 `pp2048` cell **是自洽的一對**。
- **暖平台 ＝ `10.78 / 10.91 t/s`**（`prefill250+SPAC=1`、`-b 512`、d512、NOMINAL）。
  歷史交叉驗證：09-15 `prod25-stream` **10.79**、今天 `prod25-stream` **10.89**、
  `prefill250+SPAC` **10.78／10.91** ⇒ **拿 10.8–10.9 當 25 t/s 的分母。**
- **★ 要對外可比，就得跑上游形狀的那一列（2026-09-17 追加）。** 上游 `llama-bench` 的預設是
  **`-p 512 -n 128 -d 0 -b 2048`**（`llama-bench.cpp:367-377`），標準輸出列是
  **`pp512` / `tg128` / `pp512 @ d512` / `tg128 @ d512`**（`README.md:180-187`）。
  我們的 `--depths 512` **逐字就是 `tg128 @ d512` 那一列**（`-p 0` 只是把 pp 列關掉）——
  但**我們的 decode cell 跑在 `-b 512`，上游是 `-b 2048`**，而我們的 prefill cell 跑 `-p 2048`。
  ⇒ **「對外說得出口」的 cell ＝ `--prompt 512 --gen 128 --depths 0 --batch 2048`**，
  而且要**另開一次獨立 run**（同一行程內第二格繼承暖池 ⇒ 不獨立）。

**⇒ 一句話的後果：可引用的 decode 值是 `10.8–10.9`，不是 `decode_bench` 的 `12.36`；
距 25 t/s 約 `2.3×`，不是 `2.0×`。**

**⚠ 推論（未定，等裁定）：MTP 不在這個口徑裡。** 歷史上的 MTP A/B（HTTP 路的 `12.62 vs 9.82`、
以及「accept 是被抬高的指標」那條）都是**伺服器／`llama-speculative-simple`** 量的 ⇒
在「只認 llama-bench」之下，那類結論**目前沒有 instrument of record**。兩條路：
**(1)** 把 `--spec-*` 加進 `llama-bench`（下面那節說只有三處要改，但那是 `src/`，歸引擎層）；
**(2)** 保留 `llama-speculative-simple` 當 MTP 臂，並**永遠不與 llama-bench 並排**。

要配對，env 必須是解析出來的而不是手抄的——`llama_bench_matrix.py` 支援 `PROFILE:ENV=VAL`：

```sh
python3 scripts/check/llama_bench_matrix.py \
  --arms 'prod25:CGC_SERVER_MTP=0;CGC_GPU_TIMING=1;CGC_DECODE_PROFILE=1' \
  --prompt 0 --gen 128 --depths 0,512 --reps 3 --json Backup/phase_decomp/lb_x.json
```

（這一支帶 `CGC_GPU_TIMING/CGC_DECODE_PROFILE` 是為了**逐步分解／歸因**，不是 headline；
headline 用上面那條 `--depths 512` 的標準形狀。`-d 0` 留著只為了看冷格。）

逐字就是 `decode_sweep --arms p25-gputime` 的 env（兩邊都經 **`CGC_DUMP_ENV=1` 環境變數**下的
`run_server.sh` —— 注意那是**環境變數**，不是 argv；弄錯會真的啟一個 13 GB 的 server，見下文）。

**★ 分隔符是冒號，不是分號 —— 弄錯會偽裝成「靜默空跑」**（2026-09-17 實例）：
`PROFILE` 與第一個 env 之間是 **`:`**，env 彼此才是 `;`。整串因此**必須含一個冒號**，否則
`llama_bench_matrix.py:492` 的 `if ":" in spec` 為假 ⇒ `:498` 拋 `unknown arm '...'`。

它偽裝得極好，因為 `raise SystemExit` 發生在 `run_arm()` 的無條件落檔（`:367-368`）**之前** ⇒
**cell 目錄全空**、wall **0.0s**、`rows=[]`、`build_commit=None`，看起來像「跑了但沒輸出」。
實例：`prod_matrix.py:234` 曾用 `";".join(...)` 拼出沒有冒號的臂，於是**所有** `--extra-env`
呼叫都在 0.05 s 內死掉（`/tmp/decode_ab` 臂 B、`/tmp/gpu_split` 臂 A 都是），而我一度把它
歸因成「MTP=0 特有的失敗」。

⇒ **遇到「空目錄 ＋ 0.0 s」時，下一個要看的地方是 `prod_matrix` 那份 summary JSON 的 `error`
欄位**（`prod_matrix.py:280-281` 會把 child 的 stdout+stderr 尾 1500 字存進去），
**不是 child 的 log —— 那份根本沒被建立**。

**★ `CGC_DUMP_ENV=1` 是「環境變數」，不是「參數」** —— 弄錯的代價是一次完整的 13 GB 載入：
`bash scripts/run_server.sh CGC_DUMP_ENV=1`（argv 形狀）**不會**進 dump 模式，而是**照常啟動 server**
（2026-09-17 23:26 實例：pid 16194、RSS **7.89 GB**、LISTEN 8080、存活 2:08，得手動 TERM）。
正確形狀是 `CGC_DUMP_ENV=1 bash scripts/run_server.sh`；`llama_bench_matrix.py:205` 用的就是
`env["CGC_DUMP_ENV"] = "1"`。⇒ **`run_server.sh` 把「查環境」與「啟動」放在同一支腳本裡，
形狀錯了沒有中間狀態。**

**⚠ 而且 `resolve()` / `prod_matrix.py --dry-run` 都「不是唯讀」—— 它們會殺別人的行程。**
`run_server.sh` 有一個 preflight，會 **SIGTERM 任何它找到的殘留 llama 行程**
（`[preflight] 發現 1 支殘留 llama 行程，先清乾淨再起 server`）。2026-09-17 23:31，一次
「零 GPU 讀 env」的 `resolve()` 呼叫**殺掉了鄰居正在跑的 llama-bench**（他們 30 秒後自動重試）。
`prod_matrix.py` 的 docstring 寫「Zero GPU: this is `run_server.sh CGC_DUMP_ENV=1` plus arithmetic」
—— **那句話是錯的**：它會摧毀鄰居的視窗。它的閘門（`--no-gate` 之外那個）保護的是 **cell**，
**不保護 `--dry-run` 或裸 `resolve()`**。
⇒ **別人可能正在量測時，不要呼叫 `resolve()`，也不要跑 `--dry-run`。** 要讀 env 就讀本 skill
或 `Backup/` 的既有 dump，或先確認沒有任何 llama 行程且對方明確 idle。

### `full-mtp` 的記憶體守衛，以及「不要用 prod fallback 繞過它」

`run_server.sh` 的守衛（讀碼，不是推論）：

| 位置 | 內容 |
|---|---|
| `:552 cgc_memory_guard_class` | `MTP=1 && NGL>=90 && CTX>=3072` ⇒ **`full-mtp`**（`SERVER_PROFILE=legacy-25plus` 才是 `…-known-profile`）；MTP=1 其他 ⇒ `fallback-mtp`；否則 `baseline` |
| `:566 cgc_memory_guard_req` | `full-mtp` ⇒ 只要求 **`free_pct ≥ 40`**；`…-known-profile` 35；`fallback-mtp` 20；`baseline` 15 |
| `:801` | `FREE_PCT` 來自 **`memory_pressure -Q`**，**不是 `vm_stat`** |
| `:816` → `:599 cgc_apply_prod_memory_fallback` | `CGC_SERVER_MEMORY_MODE=prod` ＋ `full-mtp` ⇒ **`CTX=1024`、`NGL=8`、`BATCH=64`、`UBATCH=32`、`BUDGET=0`** |

⇒ **`CGC_SERVER_MEMORY_MODE=prod` 不是「讓 MTP-on 跑得起來」的辦法**：它能把行程拉起來，但會把
cell 換成另一套配置（**8 層上 GPU、沒有 expert pool**）⇒ 那不是生產 cell，量到的數字不可用。
**正確的前提是等 `free_pct ≥ 40`。**

**而 `memory_pressure -Q` 的 free_pct 極不穩定**：同一台機器 2026-09-17 23:25 讀 **34%**（守衛擋掉）、
23:28 讀 **72%**（放行）—— 三分鐘內翻面。所以「現在能不能跑 MTP-on」**要用 `memory_pressure -Q` 問**，
不要用 `vm_stat` 的 `Pages free`（它當時只有 ~80 MB，會得到相反的答案）。

**`--depths 0` 是最冷的格子，不要拿它當標準。** 歷史上的「llama-bench tg 10–13 t/s」是
**depth 512–2048**，而 depth 0 一直是 **8.7–9.8**（2026-09-16 重跑：d0 **9.52**、
d512 **10.89**，重現 09-15 的 9.65/9.79 與 12.83/10.79，誤差 1.5–3%）。
`-d` 的前置填充在 `t_start`（`llama-bench.cpp:2444`）**之前**完成、不計時，所以 depth ≥512 的
實例自帶一份前置填充，d=0 沒有。**機制未確立**——同協定下 `tg@d0` 前面也有一個 `pp 512`，
所以不是單純的前置填充，也不能只歸給池。**報 decode 一律附 depth**，而且 `0,512,1024` 起跳。

**也要附「哪個臂」——本專案有三個 decode 相關的臂，只差 6 項 env，但其中一項是機制級的**
（見 `Backup/cgc_logs/instr_compare/arms_matrix_20260916.tsv`，由 `CGC_DUMP_ENV=1` 現場解析）：

| arm | CTX | -b/-ub | PREFILL_STREAM | **SPAC** |
|---|---|---|---|---|
| `prod25`（decode sweep 基準 `p25-gputime`） | 4096 | 模型預設（llama-bench 推 8/8） | - | **1** |
| `prod25-stream`（歷史 depth 矩陣；「10–13」出自此） | 4096 | 512/512 | 1 | **1** |
| `prefill250`（prefill 250 交付用） | **8192** | **5632/5632** | 1 | **-（全域預設關）** |

`CGC_SPAC` 全域預設關（`run_server.sh:1635-1637`），只有 prod25 血統自己開；而 prod25 的註解寫
「SPAC 同時是 membership 驅動，**關掉後 decode 工作集不駐留**」⇒ **prefill250 缺的是 decode 的
駐留機制，不只是 batch**。**prefill 的 profile 不能當 decode 的標準，也不該用 prefill 的 t/s 講 decode。**
`prefill250` 至今**沒有任何 decode 數字**：唯一嘗試（`llama_bench_prefill250.json`，`-b 6144`）是零列殘檔，
死於 `test_prompt: failed to decode prompt batch, res = -3`（GPU OOM）——**它的 batch 與 depth 矩陣不相容**。
要一個 profile 同時服務兩者，那是**新配置**（至少 `CGC_SPAC=1`），需要自己的基準。

**要選「哪個臂最有潛力」，判準是「剩下來的槓桿有沒有地方跑」，不是現在的 t/s。**
`CGC_POOL_MAX_TOKENS`（`src/llama.cpp/src/llama-graph.h:18-37`）預設 **8**、可調 **[2,64]**，
而它的註解自己把用途寫成「**Default 8 covers MTP n_max up to 7 (verify = n_max+1)**」；
`llama-context.cpp:286-291` 則寫「**The clamp is NOT lifted for an owned pool**」。
⇒ `prod25`（無 PREFILL_STREAM）把池路徑的批寬鎖在 8；放寬它就要在 **143 slots/layer** 裡裝下更寬的
expert union（worst layer 248 distinct vs 143 slots）。而 **M1（batched-union gather）與 M4（verify 真批次）
住在「寬 batch」的世界，那需要 whole-layer slab（`ne[2]=n_expert`，裝得下 256 個）——只有
PREFILL_STREAM 的臂有**。⇒ 三個臂裡潛力最低的是 `prod25`，最高的是 `prefill250`＋`CGC_SPAC=1`。

**已定案（2026-09-16）：統一在 `prefill250` ＋ `CGC_SPAC=1`（alpha 0.75）。**
否證實驗是 `Backup/run_unified_ab.sh`（交錯 A/B/A/B、每臂等 NOMINAL、
`-b 512 -p 0 -n 128 -d 0,512 -r 3`）；結果在 **d512**：

    SPAC 關  9.25 ±2.97  逐 rep 6.28–12.21（1.94×）
    SPAC 開 10.78 ±0.04 / 10.91 ±0.47  逐 rep 10.74–10.82（0.7%）   配對中位 +1.60（+17%）

**SPAC 主要不是把均值推上去，是把「decode 工作集不駐留」造成的不穩定拿掉。在 d0 上兩臂重疊**
（A 的 reps 自己就跨 5.95–10.58）⇒ **只有 d512 是決定性的。**
暖值交叉驗證：09-15 `prod25-stream` 10.79、今天 `prod25-stream` 10.89、今天 `prefill250+SPAC` 10.78／10.91
⇒ **本機 llama-bench 暖 decode ≈ 10.8–10.9 t/s**（拿它當 25 t/s 的分母，不是 d0 的 9.4）。
量測形狀：prefill 用 profile 的 `-b/-ub 5632`；decode/depth 矩陣用 `-b 512`（5632 會 OOM）。

**llama-bench 的 tg 沒有 `srand`**（`llama-bench.cpp` 只有 `std::rand()`）⇒ **每次跑餵的 token 流逐位相同**
（指紋：同 config 兩跑的 avg 與 stddev 可以逐位相同）。好處是 A/B 的工作負載完全相同、只剩時序；
壞處是它不是「一個隨機流」而是**一個固定的偽隨機流**，所以它與被服務請求的差異是**系統性**的。

**四個讓差距看起來像引擎快慢、其實不是的東西（按重要性）**

1. **第一個 rep 是冷的。** llama-bench 的 tg warmup 是 `test_gen(ctx, 1, ...)`（**1 個 token**，
   `llama-bench.cpp:2392`），而 context／池每 instance 建一次 ⇒ 池從空開始；
   `avg_ts` 把冷 rep 平均進去，`decode_bench --warmup 1` 恰好丟掉對應輪。
   n=128 的 rep1/平台 ＝ **1.35×**（120 vs 89 ms）。**丟掉後 11.3 vs 12.4 ＝ 1.10×。**
   ⇒ 報 llama-bench 一律附 rep 數與 warmup 規則；報 decode_bench 一律附 round 數與 `--warmup`。
2. **它對 MTP 是瞎的。** `sampler|speculat|draft|MTP` 在 `llama-bench.cpp` **零命中**
   （2026-09-16 二次核實，2507 行仍為 0），tg 迴圈是 `test_gen()`（`:2167-2186`）＝
   `llama_decode(llama_batch_get_one(&token, 1))` ＋ `llama_synchronize()`
   ＋ `token = std::rand() % n_vocab`（首 token 若 `add_bos` 則是 BOS）⇒ 沒有取樣器就沒有投機迴圈。
   **但 MTP 不是量不到**，只是要用另一支工具 —— 見下面「要量 MTP 時」那一節。
3. **兩個 token 流給池的壓力結構不同**：隨機 id → 超訂 **7/40** 層、miss **85.4% compulsory**；
   連貫文本 → **40/40** 層、**51% compulsory / 49% capacity**。所以
   **hit%／miss／reads 不可跨行程比**（生命週期累加器，累加的工作不同）；
   可比的只有行程內部定義的比例。命中率高的那一邊反而慢（96.0% → 9.4；92.6% → 12.4）。
4. **`MTP=0` 會換模型。** `run_server.sh:154` 的 `if [ "$SERVER_MTP" = "1" ]` 才選 MTP 家族
   ⇒ MTP=0 的臂載入 `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`，不是 `Nail-…denseIQ4X.gguf`。
   引用「prod25 的 decode」時必須指名是哪一個。

### 要量 MTP 時：`llama-speculative-simple`，不要動 llama-bench，也不必開伺服器

投機迴圈在 `common/speculative.{h,cpp}`（`_init/_draft/_process/_accept/_print_stats`，另有
`common_speculative_need_embd_nextn`），全 repo **只有兩個呼叫者**：`server-context.cpp` 與
`examples/speculative-simple/speculative-simple.cpp`。後者用 `common_params_parse` ⇒ 吃
**逐字相同**的 `--spec-type draft-mtp --spec-draft-n-max 3`（`run_server.sh:1137` 就是這樣餵
llama-server 的），**內建無投機的基線臂**（原始碼註解 `C0 baseline arm`），本專案已把兩筆 MTP
修復提交給它（`b7364f886` bit-identical spec vs non-spec／`eb16bd129` chunked prefill），而
`check_build_tracked.sh` 早把它列為關鍵 exe。llama-bench 用的是**自己的**解析器
（`llama-bench.cpp:509` 的 `parse_cmd_params()`，內部 `arg_prefix = "--"`），所以 `--spec-*` 對它無效。
**「不要動 llama-bench」的判準不是「做不到」**：`libllama-bench-impl` 已經 link `llama-common`
（`tools/llama-bench/CMakeLists.txt`），`common/speculative.h` 就在手邊，要加也只是三處——參數解析、
`test_gen()` 的 token 來源、以及 `avg_ts` 的 token 定義（投機下應該是含接受的 `n_predict`，不是 `n_gen`）。
真正的理由是**做完會多出第三套不可並排的口徑**（本節開頭那條）。除非目標改成「llama-bench 當唯一的
instrument of record」，否則不要動它。

```sh
bash Backup/run_spec_simple.sh                 # MTP 臂
SPEC_TYPE=off bash Backup/run_spec_simple.sh   # 基線臂（同一支工具、同 env）
```

三個坑（2026-09-16 實測，都已寫進 runner 的註解）：
- **`-c 0` 會 OOM**（模型預設 32768 × 13.66 GiB 模型 ＋ 8 GiB 池）⇒ 用 profile 的 4096。
- **profile 的 CGC 旋鈕是必需的**（`CGC_N_CB=8`／`DBUF`／`NO_PREFETCH`／`OA_ASYNC` 是記憶體形狀）；
  一個都不給 → 連 `-c 4096` 都在 t=16.7 s OOM。**env 一律從 `run_server.sh CGC_DUMP_ENV=1` 取。**
- **`--spec-type none` 不是基線**：它會去載空路徑的 draft model（`failed to load draft model, ''`，
  exit 1）。基線是**完全省略** `--spec-type`。

### 第四條路：`llama-bench` 內的 MTP（`--spec-type draft-mtp`，2026-09-18 **可用**）

**★★ 最重要的一條（2026-09-18 真因，1 行）：batch 的 `n_past` 必須從
`llama_memory_seq_pos_max(...) + 1` 開始，不是 0。**
`test_gen_spec` 明確指定 batch 的 pos（`common_batch_add(batch, id_last, n_past++, …)`），
而非 spec 的 `test_gen` 用 `llama_batch_get_one()` 讓 llama 自己配位置 —— 所以只有這條路徑要自己知道位置。
llama-bench 每 instance 的順序是 **target ctx → warmup → depth prefill(`-d`) → prompt → gen**，
所以 `-d 512` 已經寫過 pos 0..511；`n_past` 從 0 重來時**第一個 verify batch 的 `llama_decode` 靜默回 -1**
（它不印任何訊息）。

**★★ 症狀會誤導**：stderr 只有 `verify decode failed: ret=-1 n_tokens=4(pos 0..3) n_ctx=768 n_past=1 draft=3`
與 teardown 的 `CGC-M2-UNREPOINT: … a second context built from this model would otherwise read a
freed buffer` ⇒ 看起來像 Metal／記憶體／「第二個 context 踩到 freed buffer」。
**判別式**：`-n 16 -d 0` 過、`-n 128 -d 512` 掛、`-n 128 -d 512 --no-warmup` **也**掛 ⇒ 觸發條件是
**depth prefill**，不是 warmup。**把 draft context 的建立提前到 warmup 之前沒有用**（2026-09-18 用一次
真的建置與一次真的執行否證了那個假設；位移本身保留在樹上，因為它與 server 同序，但**它不是原因**）。

`tools/llama-bench/llama-bench.cpp` 的 `test_gen_spec()`；設計與四個坑見
`docs/MTP_INSTRUMENT_PLAN_2026-09-17.md` 的「實作結果」節。摘要（每一條都是跑出來的）：

- **★ argv 不能自己組。** 只帶 `-m` ⇒ 少了 profile 的 `--load-mode none` ⇒ **每一次**都在第一個
  decode 的 `CGC-METAL-FAIL: command buffer 8 failed (status 5, Insufficient Memory)` abort，而
  8 GiB vs 2 GiB pool、`-b 256` vs `-b 512`、swap 8.7–12.4 GB **全都一樣**（很容易誤判成環境）。
  正解 `forward_argv(resolve(profile, {})['server_argv'])` —— **argv 與 env 一樣只有一條解析路徑**。
- **`llama_model_params.load_mtp` 預設 false** ⇒ qwen35moe 的 loader 用 `TENSOR_SKIP` 建 MTP 區塊
  （`qwen35moe.cpp:45`）⇒ `graph_mtp` assert `layer.nextn.eh_proj`（`:566`）。server 是靠
  `--spec-type draft-mtp` 經 `common.cpp:1635` 打開的，llama-bench 沒有那條路 ⇒ 要在
  `to_llama_mparams()` 自己設。
- **`n_ctx = n_prompt + n_gen + n_depth`**（llama-bench 自己算的）⇒ `-n 16 -d 0` 的 ctx 只有 16；
  verify 批次要 `1 + draft.size()` 個空位 ⇒ 短形狀會撞牆（生產 cell `-n 128 -d 512` ⇒ 640，有餘裕）。
- **`bin/llama-bench` 的 md5 不變不代表沒重建**：它只是 33 KB 的 stub，程式在
  `libllama-bench-impl.dylib` ⇒ 新鮮度要看後者。
- **部分接受時不要直接 `llama_memory_seq_rm`**：實測第三輪 `llama_decode` 靜默回 **-1**
  （`common_context_can_seq_rm` 卻回報 FULL）。參考 `examples/speculative-simple` 的
  `common_prompt_checkpoint` 路徑（`:570-593`）；`LLAMA_BENCH_SPEC_NOTRIM=1` 可把這一項單獨拿掉來定位。

**★ 但上一輪那個 −23% 的讀數已經被推翻（2026-09-18 00:48）。** 那個 A/B/A 的兩個控制臂
（10.049 / 6.791）自己差 48% ⇒ 分母不可用。用**配對 + 反序對**（10 臂）重做之後：
正序 ratio 1.071/1.171/1.013、反序 0.938/0.701 ⇒ **順序校正後的 MTP 倍數 = 0.937（−6%），
而 5 對的範圍是 0.701–1.171 ⇒ 與 0 不可區分**。
⇒ **`llama-bench` 的 `decode` cell 在生產 pool 上是 9.7–10.6 / 8.0–10.2（隨窗口），
而 MTP 在這個形狀上沒有可測的淨增益。** 現在可引用的只剩計數器（見下）。

**輸出（2026-09-18 起）**：`llama-bench` 的 JSON（`avg_ts` ＋ `samples_ts`）＋ stderr 的
`CGC-MTP-PERF type=draft-mtp calls_draft=… acc_rate=… gen_tok_per_round=… emit_tok_per_round=… ms_per_round=…`。
選法是 **`--spec-type draft-mtp` ＋ `--spec-draft-n-max <n>`**（22:20 由另一條線做成正式 CLI；
`LLAMA_BENCH_SPEC=1` 降級為 deprecated alias）。`--spec-type` **不是**註冊的 cell：
`--cells decode` 仍然 by construction 是 MTP-off（`cgc_spec_on == false` ⇒ 走原 `test_gen`），
而這正是它必須保持的（與上游 `tg128 @ d512` 逐位元可比）。

**可引用的計數器（10 個 on 臂）**：`gen_tok_per_round = 3.000`（每臂都吃滿 `n_max=3`）、
`acc_rate` **0.626–0.795**（變異 27%！）、`emit_tok_per_round` 2.878–3.385、
`ms_per_round`（**只是 draft head**，不含 verify）25.0–31.1、池 `hit rate` **93.7–94.8%**
（off 臂 96.0–96.1% ⇒ **MTP 讓命中率降 2.2pp**）。

⚠️ **`accept` 依取樣、prompt 與臂而變**：llama-bench 預設取樣（temp 0.8、top_k 40、top_p 0.95）下
量到 0.626–0.795；served 生產 prompt 下是 38.5% ⇒ **兩者不可比**，而且**單一 accept 讀數的變異
（27%）比「≥60%」這個門檻還寬** ⇒ 它不該當閘門。

★ **機制（代數吻合，非直接量測）**：`acc_tok_per_round = n × p`（實測 `2.023 = 3 × 0.6744`，逐位吻合）
⇒ MTP 的草稿是**平行**產生、各自與 target 比對的 ⇒ **加大 `n_max` 的邊際收益是線性的**
（不像 chain 投機那樣衰減）⇒ `n_max` 掃描是值得做的下一件事。

**歷史（A/B/A，2026-09-18 00:26）—— 已被上面的配對實驗取代，三個數字都不可引用**：
`off1` 10.049 / `on` 7.715 / `off2` 6.791。留著只是為了讓「為什麼舊紀錄寫 −23%」有出處。
附帶一個仍然成立的事實：**那兩個控制臂的池統計幾乎相同**（hit 96.6 vs 96.7%、file_reads 26925 vs 26685）
⇒ 那次漂移**不是池**造成的。

**MTP 的池副作用**（兩個 off 臂一致 ⇒ 非漂移；10 臂的配對實驗也重現：on 的 hit 率 93.7–94.8% vs
off 的 96.0–96.1%）：`file_reads` **+62%**、`bytes` **+129%**、`hit rate` **−2.2 ~ −3.2pp**、
`resident` **+277 MiB**。

**設計這種矩陣時**：`ARMS["prod25"]` 裸臂**不帶** `CGC_DECODE_PROFILE`，所以它只有 t/s、
沒有每步分解；要分解得用 `prod25:CGC_GPU_TIMING=1;CGC_DECODE_PROFILE=1`（不要加 `MTP=0`）。
`llama-bench` 側的散熱讀數用 `thermal_pressure.Sampler`（2 Hz 背景執行緒）——
它是一個獨佔 GPU 數分鐘的子行程，沒有「每個請求」可以掛讀數。

**跑多臂時，每臂之間要等讀數回 0。** 本輪的教訓：四個臂連續跑，第 2 臂起就是 HEAVY
（llama-bench −7%、decode_bench −14%），於是「加長會變快」那條預測**無法判讀**。
滿載後約 **35–47 s** 回 NOMINAL。**不要**用 `--no-warmup` 去對照：它在 `-p 0` 上是 no-op
（9.40 vs 9.38），因為 warmup 只有一個 token。

## 輸出擷取（kernel-side tensor capture）：四條規則，缺一條就會讀到假結果

`CGC_TENSOR_CAPTURE=<逗號分隔的節點名清單，或 `*`>`（＋`CGC_TENSOR_CAPTURE_WORDS`，預設 32）
把指定節點的**輸出張量**快照進一個不受 allocator 管理的 shared buffer，**在 synchronize 點**才印出。
`CGC_IDS_CAPTURE=1` 會被自動帶上，因為 ids 列是比較器切 graph 的標記。
實作在 `src/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp`；分析器 `Backup/analyze_capture_nodes.py`；
驅動 `Backup/run_ids_dst_capture.sh`。

1. **同臂對照是引用任何跨臂結果的前提。** 同一個臂跑兩次、互相比對自己。
   第一版沒有記憶體屏障時，同臂**每個 graph 都 DIFF**——而跨臂跑出來的结果**看起來正好證實**
   正在測的假說（一個自信的假陽性）。屏障在儀器裡（`ggml_metal_encoder_memory_barrier`），
   但每次擴大節點清單都要重跑對照。成本：兩趟約四分鐘。
2. **命名規則是 `node(idx + n_fuse - 1)`，不是 `node(idx)`。** 可融合的 dispatcher 結尾會把
   `bid_dst` 重新指向**融合群的最後一個節點**；不可融合的以常數 `return 1;` 收尾。同一個運算式兩種都對。
   沒有這條就得追每條內部路徑（`mul_mat` 一個有八處 `set_buffer(..., bid_dst, ...)`）。
3. **`ABSENT` 不是 `SAME`。** 節點只有在融合群結束於它時才被擷取，而融合狀態取決於節點順序
   （不穩定）。實測：`norm-2` 在一個臂 41/41、在另一個臂 **0/41**。分析器把 `ABSENT`（附 `present=N`）
   與 `never` 分開印——**一個名字在其中一份日誌裡不存在，永遠不可讀成「相同」**。
4. **★ 32 詞的窗口從 element 0 起算 ⇒ 不同節點讀到不同 token，不能互相對照。**
   ggml 張量是 **ne0 最快**，所以「前 32 個 element」的意思取決於張量形狀。實測幾何（T=8）：

   | 節點 | ne | 形狀 | head 窗口覆蓋 |
   |---|---|---|---|
   | `conv_input-N` | 90112 | `(K-1+T, 8192)` = `(11, 8192)` | **全部 11 行**（state 3 行 + 8 個 token 的 qkv） |
   | `conv_output_raw-N` | 65536 | `(8192, T)` | **只有 token 0** 的 channel 0–31 |
   | `attn_norm-N`／`z-N`／`gate-N`／`linear_attn_out-N` | 16384 | `(2048, T)` | **只有 token 0** |

   ⇒ **所有 “SAME” 其實是「token 0 相同」**，而 token 0 是那次 pass 裡**最舊**、資訊量最低的 token。
   2026-09-16 為此白走了 r1–r4 七輪：整條定位鏈讀的是 prompt chunk 的第一個 token，
   而「dense 投影全同、只有 `conv_input` 不同」**不是矛盾，是兩個不同的窗口**。
   修法：內核本來就有 `n_skip`（恆為 0）⇒ **`CGC_TENSOR_CAPTURE_TAIL=1`** 令 `n_skip = ne - n`，
   把窗口錨到**張量尾端 ＝ 最新的 token**（遞歸鏈真正往下傳的那一個）；dump 加 `off=` 記錄錨點，
   分析器 `--windows` 先印 ne/off。**在讀任何 SAME/DIFF 之前先跑 `--windows`**：
   兩個節點的 `ne` 不同就代表量的不是同一個東西。**每個節點都補了 `off`，是因為一個沒印出錨點的
   窗口，與一次全張量讀取無法區分。**

**兩個比對時的陷阱**（都量到過）：

- **發射順序不是計算順序，而且不穩定**：同一個臂跑兩次，同樣的值以不同順序送出
  （`linear_attn_out-2` 有時在 `attn_residual-2` 之前、有時在之後）。localisation 要用**原始碼層鏈**排序，
  不是流的順序。
- **尾段 graph 邊界不可信**：ids 目的地上限 4096、一個 pass 約 114 列 ⇒ 尾端 chunk 把數個 pass 併在一起，
  按名字配對＝拿不同 pass 的列相比。**但 `--upto 24` 這個「解法」自己成了第二個盲點**：
  2026-09-16 實測一次 36 圖的捕獲，階段序列是
  `g0:T128, g1-2:T2, g3-27:T8, g28:T2, g29-30:T4, g31-35:T1`
  ⇒ graph 1..23 **100% 是 prefill**，而 `--upto 24` 恰好採了它。**r1–r5 的所有定位結論因此
  都是 prefill 的結論**。正法是用 **`--decode-only`**（由 `ne` 推導 T，選 T=1），不要手填數字。
  **通則：一個用來避開偽影的旗標，會變成一個盲點 —— 凡是「只看前 N 個」的預設，都必須先回答
  「這 N 個裡面有幾個是 decode」。** 階段要**量**出來，不是假設。

**先列舉再量測**：`CGC_TENSOR_CAPTURE='*'` 跑一次列出**全部節點名與 fuse 值**（本機 1021 個）。
猜名字的代價是一整輪跑。**匿名節點是 `node_NNN`，不可跨臂比對**（名字來自 ggml 計數器，
而兩臂的圖不同 ⇒ 同名不同物）。只有 builder 明確命名的節點能用。

**★ 找出「一個新鉤子貢獻了哪些名字」的正法：兩次列舉的差集。** 掛鉤子**之前**先存一份 `'*'` 列舉
（名字集合 ＋ 那份 log 的路徑），掛完再跑一次，取差集 ⇒ **新增的名字就正好是那個 dispatcher 的 dst**。
2026-09-16 實測：掛上 `ssm_conv` / `ssm_scan` / `gated_delta_net` 後差集是 **60 個**——
30 個 `conv_output_raw-N` ＋ 30 個匿名 `node_NNN`，而層號是 `0,1,2,4,5,6,8,…`，**正好跳過 3,7,11**
（`full_attention_interval=4`）⇒ 順帶證明了鉤子落在對的位置；**`ssm_scan` 零命中**（模型沒有 Mamba 路徑）。
這比猜名字強，也比逐個鉤子加 debug 打印便宜——而且它同時回答了「鉤子有沒有被觸發」這個獨立問題。

**★ dispatcher 的歸屬要查 `switch (node->op)`，不要猜。** 我猜 `ggml_concat` 走 `bin` ⇒ **錯**：
`GGML_OP_CONCAT` 有自己的 `ggml_metal_op_concat`；`GGML_OP_CPY`/`DUP`/`CONT` 共用
`ggml_metal_op_cpy`。猜錯的形狀是「名單裡加了一個永遠不會出現的名字」——**安靜無效**，
與「鉤子根本沒生效」同形。查表 5 秒。

**★ 鉤子自己的位置也是儀器的一部分：`cgc_dst_capture_at` 的宣告要放檔案最前**（`#include` 之後）。
宣告若跟著「當前第一個呼叫者」跑，每次把鉤子往前移就會 `use of undeclared identifier` 丟一個 build
（實測三次：`mul_mat` → `ssm_conv` → `concat`）。**一個可以被加在任何位置的鉤子，必須宣告在最前面。**

**這一層的模型事實會決定「attention」是什麼**：Qwen3.6-35B-A3B 是混合堆疊，
`qwen35moe.full_attention_interval = 4` ⇒ **layers 0,1,2 是 gated delta-net，layer 3 才是第一個
full attention**。所以「layer 2 的 attention」的輸出投影是普通 `mul_mat`，不是 `flash_attn_ext`。
**已掛 10 個 dispatcher**（`src/.../ggml-metal-ops.cpp`）：`mul_mat`、`mul_mat_id`、`flash_attn_ext`、
`bin`、`norm`、`ssm_conv`、`ssm_scan`、`gated_delta_net`、`concat`、`cpy`。
**CPY/DUP/CONT 共用一個 dispatcher** ⇒ 擋住捕獲量的是**名字過濾器**，不是鉤子；不要配寬泛的名字。

**gated delta-net 的鏈條**（`src/models/qwen35moe.cpp:415-425` ＋ `src/models/delta-net-base.cpp:449-496`）
——這是 2026-09-16 用來把分歧往裡推的骨架：

```
cur = attn_norm(inp)
 ├ z / beta / alpha→softplus→gate              （投影，與 conv 並行）
 └ conv_input = ggml_concat(conv_states, qkv_mixed)      ← 前段是 recurrence state
     conv_output_raw = ggml_ssm_conv(conv_input, kernel)
     … → conv_state_update = ggml_cpy(conv_state_last)   ← state 寫回 KV cache，下一步再讀
     → q/k/v_conv → l2_norm → ggml_gated_delta_net → norm → linear_attn_out
```

**`conv_states` / `conv_state_last` 是 view（無 kernel）⇒ 擷取不到**；那個循環裡唯一能讀的一格是
`conv_state_update`（`ggml_cpy` 的目的地）。

**新判準：比對前先看「逐 graph 的 DST 列數」，不是總數。** 總數不等**未必**致命：實測
`gputime 533` vs `slotgpu 492`，但被測節點在**每個 graph 都是 1/1** ⇒ 那個差是**別的**節點被融合吸收
（局部、無害——配對是按 `(graph, name)`，不是列序號）。**致命的是圖標記（`GRAPH_START`）缺席**，
那會把兩個 pass 併成一個、之後每個 graph 整體錯位。⇒ 分析器現在印**逐 graph 的行數**並列出不一致的
graph；**最強的保證是「你要的節點在兩份日誌的每個 graph 都恰好出現一次」**（比總數相等更強）。

## 陷阱（都踩過）

1. **`union > gpu_busy_sum` 是數學上不可能的指紋** ⇒ 取樣有 ABA 競爭。
   在 completion handler 裡累加 atomics、再由 reader `atomic_exchange` 歸零，
   handler 的 min/max CAS 會在歸零後把舊值寫回。
   **正解：不要累加，直接讀已完成的 `ctx->cmd_bufs[i].obj` 的
   `GPUStartTime`/`GPUEndTime`**（每個 segment 各自是一次 `graph_compute_async`，
   段邊界剛好就是那批 buffer 還活著的時刻）。零共享狀態。
2. `gpu_union ≈ wait` 是**結構性**的，不是 GPU 在忙：同一條 queue 上
   `[min start, max end]` 本來就橫跨整個 queue 佔用窗口。
3. `ggml-metal-context.m` 是 **Objective-C**：函式內 `static` 需要編譯期常數初始子，
   `static const bool x = getenv(...)` 編不過。快取 env 要放 struct 欄位或惰性初始化。
4. 跨 dylib 呼叫 Metal 只能走 `ggml_backend_reg_get_proc_address(reg, "ggml_metal_get_*")`
   （`libggml-metal` 是獨立 dylib，`libggml` 不 link 它）。
5. `decode_sweep.py` 的 `CGC_SERVER_EXPERT_CACHE_BYTES=0` **不會**關掉快取
   （run_server.sh 仍會加 `-expert-cache 0` → 幾何不一致 → 每請求 HTTP 500）。
   真正 cache-free 要用 `CGC_SERVER_EXPERT_CACHE_OFF=1`；但 16GB 機器上那會 Metal OOM。
6. macOS BSD `grep` 的 BRE 不支援 `\|`：多模式一律加 `-E`。zsh 遇 `*.log` 無匹配會直接
   報 `no matches found` 而**不執行**整條指令，用 `ls -t ... | head` 取檔名再迭代。
7. **儀器的掃描視窗常常比假設窄，而沉默會被讀成確認。** 兩個 S1 host 側探針都寫死在
   **layer 0..3**（`for (int il = 0; il < 4; ++il)`；`if (!on || il > 3) return;` 且只印 8 行），
   而層梯二分把缺陷壓在 **18..19**。於是「探針說映射正確、bit-identical 閘門卻不過」看起來像矛盾，
   其實是**探針從未看過缺陷所在的層**。動任何診斷前先問：它的掃描範圍覆蓋我的假設嗎？
   （同一家族：rate-limited 斷言只能當樣本當普查；「0 行輸出」與「沒有東西可印」不可分。）
8. **`ggml_set_output` 是隱含契約，不是保險。** 沒有它，ggml-alloc 會把同形狀的節點疊到同一塊
   buffer：S1 的 11 層裡 10 層的 `ids_data` 是同一個位址（`0x128179d60`），每層讀到同一份 16 個 int。
   重寫資料流時要**清點原路徑所有看起來像保險的呼叫**。
9. **編碼期讀 GPU 產物是競態。** `CGC-MMID-ASSERT` 在 Metal **encode** 時於 host 讀 `op->src[2]->data`；
   對 host 寫的 leaf 無害，對 GPU 算出的 id 向量會讀到配置器殘留（F32 router 機率 `0x3F64E6C4` /
   `0x3F65BD44`、`+NaN`、同節點內合法/非法索引混雜）。S1 的 `id_oob` 全部是這個假警報。
   **現在有三個時刻，必須分清「你想量哪一個」**：

   | 時刻 | 量到什麼 | 儀器 |
   |---|---|---|
   | encode 期，command buffer **還沒跑** | **前一位佔用者**的殘留 | `CGC-MMID-ASSERT` |
   | 消費 kernel **之後**、**同一** command buffer 內 | **kernel 真正消費掉的值** | `CGC_IDS_CAPTURE` |
   | `ggml_backend_sched_synchronize()` 之後 | GPU **產物**（已經寫回的結果） | `CGC_S1_DBG` 的 `POST` |

   實測同一層、同一步：主機說 `id_oob=16/16 first=1063585220`（= `0x3F65BD44` = F32 0.895 = 路由器機率），
   內核說 `[2,105,7,9,5,106,1,84]` 全在 `[0,143)`。**兩個都「對」，因為它們量的是不同時刻的同一個 buffer。**
   要主張「kernel 用了什麼 id」，只有第二列算證據。
10. **反過來也一樣：圖跑完後讀「瞬態」張量是讀死緩衝。** 非 output 的張量在最後一個 consumer
   跑完就被回收。POST 探針第一版去讀 gather 的**索引向量**（CONT 的輸出，不是 output），
   layer 2/3 讀到 F32 位模式（`idx=1043923934`），報出 16/16 假 mismatch。
   兩個方向是同一個教訓的鏡像：**一個讀太早、一個讀太晚**。
11. **改道會靜默關掉儀器。** `cache->n_zero_mapped_selected`（QUALITY LEAK）唯一的加總點在
   leaf 寫入區塊內；S1 不建 leaf ⇒ 40 層只剩 layer 0 在計數，teardown 印的 `0` 與「真的沒 leak」
   不可區分。**任何「純排程／dispatch 改動」都要檢查它順手關掉了哪些計數器。**
   注意 `llama-context.h:364` 的註解聲稱「leaf 仍會建」——**程式是 if/else，leaf 不建**，註解是錯的。
12. **可比性 stamp 只跟「寫它的那個閘門」一樣有行為意義。** `m123_oracle_gate.py` 的 comparability
   檢查比對兩側解析出的 config 值。若某個閘門曾是 **presence-based**（`getenv(...) != nullptr`）
   而啟動器**無條件**把變數塞進 `SERVER_ENV`，那時**值不生效** ⇒ 那個年代寫下的 `.cap`
   記錄的是**意圖**，不是**行為**。實例：v2 oracle 的 `.cap` 記 `CGC_OA_ASYNC='0'`，
   但它是在 presence 閘門下 dump 的，那個 `0` 選的是**分段**分支，與今天的 `1` 同一條路。
   ⇒ **改閘門（presence→value）之後，所有舊 cap 都可能變成 stale metadata，會讓 comparability
   對每一個舊 ref 永久判 INVALID**，反過來把閘門自己的判別力關掉。
   **判別器是比 stamp 更強的那個證人**：兩條 dispatcher 是實測 10× 的分支、digest 不同
   （非分段 `ff68c5a2` vs 錨 `dc055e63`），所以若今天跑的是非分段就**不可能**對分段的舊 ref
   量到 9/9 bit-identical。**處置是修 stamp（重錄 ref + 讓 profile 顯式 pin），
   不是 `--allow-incomparable`**——後者是對邊界已知的一次性問題給永久特赦，
   會讓真正的跨配置回歸以 PASS 的形式到來。
   重錄之所以不算「移動球門」，唯一授權是**新 dump 與舊 dump 逐位元相同**（`md5` 兩檔一致）。
13. **profile 沒有顯式寫的那一格，就是會漂移的那一格。** 改一個**全域預設**會靜默重解析所有
   「只靠繼承」的 profile。實測：全域 `SERVER_OA_ASYNC` 由 0 改成 1 之後，
   `off / prefill250 / qa-zh / longform-zh` 四個 profile 的有效行為全變了，而
   **`prod25` 是唯一不受影響的**——因為它是唯一**顯式**把該旋鈕寫死的 profile。
   ⇒ 若用 `prod25` 的 digest 不變來「證明」修正無害，那是**用最差的證人**：顯式設定值的那一格
   正是 value-aware 修正**不可能移動**的那一格。**改 profile/預設時要拿「靠繼承的那幾個」當證人。**
   `prefill250` 已於 2026-09-15 顯式 pin `CGC_SERVER_OA_ASYNC=1`。
14. **profile 寫 `=0` 期望「關掉」一個測「存在」的旋鈕 ⇒ 等於打開它。** 陷阱 12/13 是
   **閘門**測存在；這一條是**同一個病在 profile 上**，而且後果更重：它讓一條文件化的修復
   **六年天數的空轉**。實測：`LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0` 三個讀者全部測存在
   （`llama.cpp:396`、`llama-expert-cache.cpp:1708`、`llama-context.cpp:5222`），
   而 `run_server.sh:939-945` 把它設成 `0`、由 `:1493` 的 `env "${SERVER_ENV[@]}"` 真的傳進子行程
   ⇒ `getenv` 回傳**非空** `"0"` ⇒ **skip0 一直是開的**，blk.0 一直在 pool 之外，
   而註解白紙黑字寫的意圖正好相反（「blk.0 **回到 pool 內**」，因為 skip0 是 4/10 品質殺手）。
   - **唯一吐出真相的觀測是推導量**：`CGC-DECPROF` 的 **`layers=39`**（19/19 步；`layers=40` **0 次**）。
     旋鈕自己不會說謊也不會說話——**記錄下來的值只代表意圖**。
   - ⇒ 所有 `cap_*.json` 與 llama-bench 的 `env` 區塊對這個 knob 記的都是**意圖**。
     **任何宣稱 profile 行為的句子都不得引用它。**
   - 動任何旋鈕前先 grep 它的**讀者**，確認測的是**存在**還是**值**。
     `0` 與 unset 對 presence 閘門**同義** ⇒ 寫 `=0` 是寫了一個 no-op。
   - 這也是**改進 profile 前必須先查**的一步：你以為關掉的東西可能一直開著，
     於是你對「現況」的每一個解釋都建立在一份錯的配置上。
15. **擾動型儀器的成本是「每個 graph 釘幾個張量」，不是位元組（A8）。** `CGC_S1_OUT_CAP` 用
   `ggml_set_output` 釘住 `ffn_moe_down` 以便 sync 後讀活緩衝。實測邊界：
   **全 graph × 40 層 → 載入期就死**（`kIOGPUCommandBufferCallbackErrorOutOfMemory`
   → `CGC_METAL_FAIL_STOP` abort）；**只有 prefill × 11 層 → 同樣死**；
   **只有 prefill × 4 層 → 過**；**只有 decode × 40 層 → 過**。
   失敗的那個 graph 是 `ntok=2`，11 層共 **1.4 MB**——**1.4 MB 撐不爆任何 Metal 預算**。
   控制變數是「被標為 graph output 的張量個數」，改變了排程器的配置與切分；**機制未確立**。
   - **不要用算式訂上限**（我寫過兩個錯的位元組預算，都被同一組配對反駁兩次）；
     **也不要逐步加寬逼近**——失敗是 fail-stop abort，不是變慢。
   - 讀回**必須綁相位**：`if (cgc_s1_outcap && !batched)`。
     只綁旋鈕的話，`graph_compute` 的**前幾個 graph 是 warmup/prefill**，
     而 prefill 走的是**另一個分支**（不建 remap leaf）⇒ 兩臂會在「機制根本沒啟動」的圖上被比較，
     然後報「相同」，理由與機制無關。
16. **llama-bench 的 `build_commit` / `build_number` 不是本專案的指紋。** 三跑（其中一跑在重建之後）
    **全部**印 `build_commit=e8b85393d build_number=239`。它不追蹤本地編輯 ⇒
    **不可用來判斷「兩跑是否同一支 binary」**。要指紋就自己 md5 `libllama*.dylib` + `llama-server`。
    另一個同族：**純註解編輯也會改變 `libllama` 的 md5**（debug 資訊內嵌），而 `llama-server` 不變
    ⇒ **每一列都要記自己的指紋**，不能靠「上次閘門 PASS 過」推論。
17. **值語意（value-semantics）變更，閘門<u>看不見</u> —— 必須自己宣告重新基線。**
    `m123_oracle_gate.py` 的可比性前置條件是**比對 resolved env 字串**。若一個修復只改變
    「某個值代表什麼」而**字串不變**（例：讀者由測「存在」改成測「值」，而 profile 前後都寫 `"0"`），
    閘門會看到 **0 個差異** ⇒ 直接走正常判決 ⇒ 報一個**普通的 M1 FAIL，而那是 category error**。
    - 對照組：`CGC_OA_ASYNC` 那次**字串真的變了**（`0 → 1`），閘門自己就報 `INVALID COMPARISON`
      叫醒人。值語意類變更**沒有這個警報**。
    - **唯一的分類器是控制臂**：用**修復前的值**跑一次（`CGC_SERVER_SKIP0=1`），
      它必須**逐位元重現舊參照**。這同時證明 (a) 舊參照是在舊行為下 dump 的，
      (b) predicate 改動在其餘維度數值中性。**沒有控制臂，「新的值」與「改壞了」無法區分。**
    - 然後才用 `--write-ref` 寫新參照，並把 `DEFAULT_REF` 指過去（**否則之後每一跑都對著退役參照
      FAIL**）。`--write-ref` 在比對**之前**執行，所以同一跑就能同時拿到新參照與對舊參照的差異形狀。
    - 判讀新舊差異時看 **cross-tab**：`diff/diff = 0` + `M2 9/9` ⇒ **漂移**，不是分歧
      （2026-09-16 實測：M1 4/9、M2 9/9、M3 6/9、`{same/same 4, diff/same 5, diff/diff 0}`）。
18. **兩臂都印的同一個數字，不能用來區分兩臂。** 用它證明「狀態變了」之前，先確認它**真的隨該狀態變動**。
    - 反例：`LAYER_CAPS per-layer caps: total 5976 slots` 在 **39 層池化臂與 40 層池化臂印出一樣的 5976**
      （它是 LAYER_CAPS 的計畫總和 ＝ 40×143＋MTP 256，從不查詢實際池化層集合）。
      2026-09-09 的甜蜜點 log 把這個**常數**當成「layer 0 在 pool 內」的證據。
    - 能區分狀態的是 teardown 的 `owner-set slots`（**5793 → 5935，＋142 ＝ 恰好一層的
      resident slot 數**，不是湊整數 ⇒ 不是雜訊）。
    - **★ 2026-09-16 更正：`zero regions` 不能用來區分這個狀態。** 它自己就過不了這一條的檢驗——
      同為 **39 層池化**的兩臂印出不同的值（`5577 → zero-regions=2`、`5793 → zero-regions=0`）。
      它是**內容讀數**（池中當下有哪些 expert resident），不是狀態讀數：
      已查清是 checkpoint 的**零前綴**（10/31488 條列的開頭有 4592–13120 B 全零、整列 17.8–47% 非零），
      池中那 4 個 = 2 個 resident 零前綴 expert × {gate,up}。**不是填充缺陷**：
      同層 138/142 resident slot 非零；真正會靜默丟專家的保留 ZERO slot 讀數
      `verify-strict: refused=1 zero_mapped_selected=0`。
      工具：`scripts/check/gguf_dead_expert_census.py`（檔案側，不需 log）、
      `scripts/check/mmid_zero_row_triage.py`（log 側，讀整條 stride，`exit 3` ＝ 修探針）。
      判準寫在 CONVENTIONS **A13**：判定用的儀器不得沿用被判定對象的視窗寬度。
    - 乾淨的寫法是量**推出量**：`owner-set` ＋142、`resident` ＋152 MiB、`file_reads` ＋396、
      `requests` ＋191。**旋鈕的值永遠不得當成它自己生效的證據**（CONVENTIONS A9）。
19. **跑控制臂 / 開關旋鈕之前，先確認它<u>能不能被外部覆寫</u>。**
    `run_server.sh` 的 `SERVER_ENV` 是陣列，`env "${SERVER_ENV[@]}"` 會**蓋掉**呼叫端傳入的同名變數
    ⇒ 寫死字面值的那一格**無法用 `--env` 覆寫**，控制臂根本跑不起來。
    2026-09-16 為此把它改成 `${CGC_SERVER_SKIP0:-0}`。**要掃旋鈕前先確認它是 `${VAR:-default}` 形式。**
20. **吞吐數字先問「它重複得出來嗎」，再問它是多少。而記憶體壓力計數器不是資源的量度。**
    250 t/s 在 `prefill250` 形狀（13.66 GiB 模型 + 8 GiB pool + 6144-wide ubatch）上
    **11 次獨立啟動只有 1 次越過**（286.67，其餘十次全在 155.33–184.09）。
    兩個候選機制都被自己的實驗砍掉：`free`（rho −0.70，方向是反的）與
    **page cache**（`File-backed pages`）—— 後者被 `--warm-runs` **強制拉高 4.3× 到 9.96 GiB**，
    吞吐**沒動**（rho +0.09）。最乾淨的判準是一對同狀態樣本：free 3.90 GiB → 286.67、
    free 3.74 GiB → 179.09。**幾乎相同的啟動狀態，60% 的差距。**
    - 儀器：`scripts/check/prefill_certifiability.py`（`--warm-runs` 是打破
      「執行順序 ≡ 變數順序」的唯一方法；純加樣本永遠分不開因果與累積）。
    - 規則：**輸出「可重複的帶」＋「越過目標的次數 k/N」**，不是範圍也不是目標（CONVENTIONS **A15**）。
    - `Pages free` 低可以是 cache 大（快）也可以是別的行程佔住（慢），**方向相反而不可加總**
      （CONVENTIONS **A14**）。要宣稱記憶體機制，先指名**哪一個**資源並強制它到兩個極端。

## 統一 profile（2026-09-16 定案，已落進 `run_server.sh`，不只是建議）

`prefill250` 現在 pin `CGC_SPAC=1` ＋ `CGC_SPAC_ALPHA=0.75`（`run_server.sh:335`），所以
**一個 profile 同時是 prefill 與 decode 的臂**。`-b/-ub` 是**量測形狀、不是 profile 差異**：
prefill 用 profile 的 5632；decode／depth 矩陣用 **512**（5632 在 16 GB 上於 `-d≥512` 會 OOM，
`Backup/llama_bench/llama_bench_prefill250.json` 就是那個零列殘檔）。

**判準不是現在的 t/s，是「剩下的槓桿有沒有地方跑」**：池夾制把 `prod25` 的池路徑批寬鎖在 8
（`llama-graph.h:18-37` 的註解自己把它綁在 MTP 上），而 M1（batched-union gather）與 M4（verify
真批次）需要 **whole-layer slab（ne[2]=n_expert，裝得下 256 個）**——只有本 profile 有。

**pin 一個 env 鍵的代價（每次都要盤點，兩個都真的發生了）**：
- D5 會對該鍵報 INVALID COMPARISON ⇒ **重新基線**，而**證據是 jsonl md5 相同**（v4/v5 都是
  `a0a0ca742ca94e843c54b39981742738`），不是「M1 9/9」。流程見 skill `cgc-commit-gate` §2.5。
- `decode_sweep.py` 的 `--profile` **預設就是 prefill250** ⇒ 不帶 `--profile` 的 sweep 會靜默取得
  SPAC=1，而它的 `spac-on` 臂從此與 `baseline` **同義**（退化）。要真正的 SPAC A/B 用
  `--profile prod25 --arms baseline,p25-nospac`，或 `prefill250:CGC_SPAC=0`。

## 現況結論（2026-09-15 收盤，供後續對照）

> **記憶位置（2026-09-17 拆分後）**：專案長期記憶已拆成 **1 索引 ＋ 3 主題檔**
> （`.workbuddy/memory/MEMORY.md` ＋ `_PERF`／`_S1`／`_FACTS`；原 ~19 KB 的單檔會被 session 注入截斷）。
> **S1／分歧定位的權威檔是 `.workbuddy/memory/MEMORY_S1.md`** —— 本節與下面的 S1 段是**摘要**，
> 兩者不一致時**以記憶檔為準**（舊報告寫的「`MEMORY.md` 的 S1 節」也是指它）。

- decode 步 ≈ 80 ms；`wait` 佔 83–90%，`cb` 7–15%，`submit` 4%；`fill_wait = 0`（fast path 不 fill）。
- `n_cb` 1→16 對 decode **完全無影響**；`CGC_MMV_FUSE` 現配置下**輸出損壞**且更慢。
- `gap` = 12–22 ms/步（wait 的 17–35%），**大於** CPU 側窗口（cb+submit = 8.5–15 ms）。
- **序列化天花板的認證數字是 ×1.711（配對中位，區間 1.327–1.924，交錯 3 輪 + build 指紋）。**
  由 `CGC_SUBMIT_AHEAD` 上界探針取得，且**它的輸出必然損壞** ⇒ 只能當天花板，不得進對外表格。
  單輪曾量到 ×1.78，那個數字**不可引用**（不是不夠準，是不成立——每一輪 md5 都不同）。
  誠實的讀法是：區間下界 1.327 = 已證明可拿到的量；上界 1.924 = 最樂觀的天花板。
  設計見 `docs/REMAP_ROUNDTRIP_REMOVAL_PLAN_2026-09-15.md`。
- **`gpu_union/wait ≈ 90%` 不代表 GPU 飽和**（見陷阱 2，那是結構性偏誤）。因此
  「病因是序列化」**不是**由佔用率推出的，而是由上界探針推出的。**不要用佔用率反推病因。**
- node fusion / kernel 微優化 / encode 側調整優先度仍低——除非 S1 落地後 `n_segs` 從 40 降下來。
- **W3（`cap × top_k ≤ usable slots`）這個「總開關」仍未解 —— 而且不是「未實作」，是「已否決」。**
  `llama-context.cpp:286-293` 的 `cparams.n_batch = cgc_pool_max_tokens()`（預設 8）仍在樹上；
  只有 `CGC_PREFILL_STREAM` 會解除它，而且**只對 prefill**（「Decode / MTP verify still run
  n_tokens <= pmax through the pool path」）。M1 work item 1（`CGC_POOL_SPLIT` 把 pool 與
  graph 幾何解耦）**存在**，但 `docs/M1_POOL_SPLIT_COST_2026-09-14.md` 的狀態欄就是
  `implemented, measured, and rejected`，`llama.cpp:424` 印
  `EXPERIMENTAL, KNOWN-BROKEN (degenerate routing + SIGSEGV)`，而
  `llama-context.cpp:268-278` 記載**即使 pool 自有配置，夾制也刻意不解除**。
  兩個阻斷各有簽名：A＝wide 路徑（`mode=wide ne2=256`）是 Metal 已知讀不對的配置；
  B＝`CGC-POST: il=1 st[0]=1 st[1]=1`（兩 expert 共用一 slot）⇒ NaN ⇒ il=2 SIGSEGV。
  該文件自己指出真正的形狀：wide tensors 作為 **host 側讀取來源**餵給 pool，
  **不是**作為 graph 能 dispatch 的張量。**動這條路之前先讀那份文件**，起點是 ownership model
  （`slot_owner`/`slot_table`/`batch_owned`），不是 loader。
- **S1（GPU 側 slot 查表）仍未通過 bit-identical 閘門（M1 5/9、M2 7/9、M3 5/9），
  但分歧的「形狀」已定案（2026-09-15 23:5x，`dec-20260915-2350`）。**
  **★★★★ 2026-09-17 02:55 再往下一格（§EN-17）：`src[2]` 那條路線的儀器在 S1 臂上是瞎的。**

  想驗「同一組 ids 讀到不同位元組」時，**先別寫新儀器** —— `CGC_MMID_MV_DBG`（`ggml_metal_op_mul_mat_id`，
  已在 `run_server.sh` allowlist）逐 MoE 節點印 `src0 ne`、ids 運算元與 **ids 所選前 4 列各前 ≤4096 bytes
  的 FNV-1a 指紋**，它自己的註解就是判準：「ids 同而 hash 不同 ⇒ 池的內容錯了」。
  `=1` → 150 節點（一個完整 pass ＝ 41 層 × 3）；**`=0` 會被讀成 4096 節點**（解析器只看第一個字元），
  所以 `run_ids_dst_capture.sh` 的 `MMID=1|2|3` 穿透會拒收 `0`。比較器 `Backup/compare_mmid_fp.py`。

  **但它對 S1 臂無效**：同一節點 `ffn_moe_gate-1`，錨臂 `ids=[2,105,7,9,5,106,…]`、`id_oob_vs_ne02=0`、
  指紋完整；S1 臂 `ids=[1063628079,-1100333772,…]`（**float 位元模式**）、**`id_oob_vs_ne02=16/16`**、
  **一列指紋都印不出來** ⇒ **S1 臂在 encode 時刻 `op->src[2]->data` 還沒被填**，讀到的是被重用的工作緩衝
  殘留。**「主機側、encode 期」讀中間運算元不是「消費者讀到什麼」的證據**（repo 對 `CGC-MMID-ASSERT`
  的 `id_oob` 已有同樣註記）。有效的只有**內核側**讀數（在 command buffer 內、裝置上讀）。
  順帶量到：同一個 ids 運算元，兩臂的**填充時序不同**（錨臂 encode 時已正確、S1 臂仍殘留）。

  **附帶的通用教訓（比較器）**：指紋要**按 row id 獨立配對**，不能因為 ids 清單不同就短路 ——
  第一版短路後 `BYTES_DIFFER=0` 是**空洞的**（117 列全被判 IDS_ONLY，一列的位元組都沒比過）。
  修好後它印「0 shared rows」，把「讀不到」變成看得見。**「沒有差異」與「沒有比較」必須印得出來。**

  **★★★ 2026-09-17 02:40 取代下面全部（r23–r25）。分歧跟著 `CGC_S1_MIN_IL` 走，不跟著層號走。**

  **先講「哪一個 pass」怎麼定**（上一輪我是用推的，這輪直接量）：
  - 分析器的 **stage ＝ 模型 pass**，標記是 **ids 列 `name=ffn_moe_gate-1`（沒有 `.dst`）**。
    用 `.dst` 去找標記會讓整份 log 被當成一個 stage（我第一版就是這樣）。
  - T 要**用節點自己的 `ne` 直接讀**（`l_out-0` 的 `ne / 2048`），不要用「graph 內 min ne 的比值」——
    後者取決於該 graph 哪些名字被捕捉到，node set 一換就會給出不同的 T。
  - 本 run 形狀（`n_predict=12`、208-token prompt、`n_batch=8` 的池路徑）：chunk 序列是
    **`2,2,8×21,6,8,2,4,4`，之後 11 個 T=1 的 decode**，與引擎側 `n_past` 的 +8 節奏一致。
  - 新選項：**`--select-t N`** = 第一個**完整**的 T==N stage（「完整」＝列數等於該 T 的眾數；capture
    起始的 stage 0 只有 1 列，是 fragment，選它會把 `present=1` 讀成「另一臂缺這個節點」）；
    **`--list-stages`** 先列出每個 stage 的 T 與列數再選。**永遠先 list 再 select。**

  **r23／r24（17 個 layer-0 名字、兩臂 × 2；同臂對照全乾淨）——在第一次完整 T=2 pass 上**

  | 觀測 | 結果 |
  |---|---|
  | layer 0 全部 15 節點（含 `ffn_moe_logits_raw-0` 與 `gate/up/down/out-0`） | **SAME** |
  | `mul_mat_id` 消費的 **ids** | **120 SAME / 0 DIFF** |
  | 池佈局（SLOT-OWNER）與被消費專家集合（SLOT-SEL） | **40/40 SAME**（g0、g1 兩個 2-token pass） |
  | `wrong` / `unowned` / `owner==ids` | 0 / 0 / 560–560（degeneracy 同上一段） |
  | 唯一不同的節點 | **`l_out-1`** |

  ⇒ 引擎側的 SLOT-OWNER／SLOT-SEL 閘要放寬到 **`n_tokens <= 2`**，否則那個 pass 根本不被量到
  （decode-only 的閘會讓最關鍵的一步空白）。

  **★ r25 因果測試（零改碼，本輪最關鍵）**：把閘從 layer 1 移到 layer 2（`CGC_S1_MIN_IL=2`，臂
  `p25-slotgpu-l2` 已存在）：

  | 配置 | 第一個分歧節點 | 它之前 |
  |---|---|---|
  | `MIN_IL=1` | **`l_out-1`** | `l_out-0` SAME |
  | `MIN_IL=2` | **`ffn_moe_out-2`** | `l_out-0`／**`ffn_moe_out-1`／`l_out-1` 全部 SAME** |

  ⇒ **第一個分歧的層跟著閘走。同一個層 1：host leaf 服務時逐位元相同，GPU table 服務時就不同。**
  「某一層壞了」與「從 layer 0 數值傳染」**兩種讀法全部作廢**——之前所有「第一個分歧＝層 N」的述句都要
  改寫成「第一個被 GPU table 服務的層」。

  **⇒ 盒子裡剩什麼**：那個 pass 上 layer 0 整條鏈、ids、池佈局、被消費集合、反查一致性**全部相同**，
  而第一個被服務的層的 gather 輸出不同 ⇒ ids 相同 ⇒ 讀到的 slot 相同 ⇒ **唯一剩下的是 gather 讀到的
  「位元組」不同**，即 expert 權重運算元（`src[2]`）的指向／repoint 在兩種對映下不同。
  **這是「位址類」缺陷**，與先前排除的數值／對映／時序**不同類別**。
  **下一步**：在 `ggml_metal_op_mul_mat_id` 對 `il <= 2` 限次印 `op->src[2]->data` 與其 buffer id。

  **★★ 2026-09-17 01:25 取代下面的 r10/r11 段（r21/r22）。載體不在池；而且我自己的讀數退化了。**

  **問題**：前向對映（expert→slot）已被 `EQUIV-pool`（1599/1599 `mismatch=0`，同瞬）與 `SEL-DRIFT`
  （每圖 `entries=0`）清乾淨。剩下的是**反查**：兩臂都同意「slot *s* 屬於專家 *e*」時，*s* 裡裝的是不是
  *e* 的權重。`slot_owner[layer][slot]` 由**另一條路徑**寫入（`llama-expert-cache.cpp:827` 的
  `pick_slot`／`prefetch_slot`），publish 只讀 `slot_table` —— 同一指派、兩份表示、兩個寫者。

  **兩個新讀數**（gated on `CGC_S1_TABLE_CHURN=1`、只印不進 gate）：
  `CGC-S1: SLOT-OWNER graph=G il=L sum=… xor=… wsum=… n=… owned=…`（整層）；
  `CGC-S1: SLOT-SEL graph=G il=L ntok=T sum=… wrong=… unowned=… idsum=… idsxor=… idwsum=…`（只取被消費的 ids）。
  比較器 **`Backup/compare_slot_owner.py`**；新臂 **`p25-gputime-churn`／`p25-keepleaf-churn`**。

  **★ 兩條新的儀器規則（都踩過，比先前那三條更前置）**
  1. **掛點必須是「兩臂都跑到」的地方**。第一版掛在 publish 路徑
     （`cgc_publish_slot_table_counted`）——而它**只有裝了 GPU table 的臂才會被呼叫** ⇒ 錨臂輸出
     **零行**、S1 臂 390 行，比較的一側是空的。要跨臂比，就得選兩臂都經過的 hook
     （這裡是 `expert_cache_on_topk`；判準是兩臂的 `CGC-HOOK` trace 行數相等）。
  2. **取樣點必須在「填充之後」**。在 hook 開頭取會讀到**本步尚未填充**的表
     （`slot_table[e] = -1`）⇒ `unowned` ≈ 每格 1（≈12%，正是冷專家率），**會被讀成損壞**。

  **★ 第三條（這輪最貴的一條）：反查回身分的摘要，在池自洽時是恆等式。**
  若 `wrong == 0` 且 `unowned == 0`，則 `owner == ids[j]` 逐元素成立 ⇒ **owner 的摘要就等於 ids 的摘要**，
  讀數量的是**路由**不是池。修法是**在同一行印出對 `ids` 本身的摘要**（`idsum/idsxor/idwsum`）當作這個
  讀數自己意思的控制：相等 ⇒ 退化，把結論寫成「路由不同」而不是「池給了不同的專家」。
  **通則：任何「把 A 反查回 B 再摘要」的讀數，都要同時摘要 B 本身，否則測到的是恆等式。**

  **結果（r21/r22，三臂 × 兩輪；同臂對照 440/440 全 SAME、兩個絕對閘皆 0）**

  | 配對 | SLOT-OWNER | SLOT-SEL | 讀法 |
  |---|---|---|---|
  | 錨 vs `p25-keepleaf-churn` | **440/440 SAME** | 440/440 SAME | 光是**建出** S1 節點不改變池、也不改變路由 |
  | 錨 vs `p25-slotgpu-churn` | **438/440 DIFFERENT**（`PERMUTED=0`） | 440/440 DIFFERENT | 差異綁在「GPU table 真的被消費」上 |

  - **★ 否證**：`wrong = 0` 且 `owner == ids` 在 **440/440 格、每個臂**成立 ⇒ **被消費的專家，它落到的
    slot 裝的就是它自己**。「同一個 slot 裝了不同專家 ⇒ 同一組 ids 讀到不同權重」**不成立**；
    「gather 靜默讀到另一個專家的權重」在被消費子集上從未發生。**§9.18.4（池/slot 權重內容）這條走完。**
  - `PERMUTED = 0` 要單獨讀：若是同一組專家換 slot，判準會給 PERMUTED；現在 `sum`／`xor` 都不同
    ⇒ **常駐的專家集合本身不同**，不是重排 —— 但那是**後果**（見下）。
  - **★ 真正的讀數**：兩臂的 ids 摘要**各自 440 個互異、交集只有 1**；A 的 graph 0（40 層整組）不曾在
    B 的任何 graph 出現（⇒ 不是圖索引位移）；graph 0 逐層 il=0..5 全 DIFF；`sample` 欄
    錨「巴黎是法国首都…」vs S1「巴黎并非法国的首都…」⇒ 分歧在**第 1～2 個生成 token**；池統計也分岔
    （`miss_capacity` 4648 vs 5349）。而 `CGC-HOOK` 前 80 行**全是 `ntok=2`** ⇒ 那兩趟是
    **2-token 的池路徑 prefill chunk**，之後的 1-token 步才被記到。
    **⇒ 池不是載體，路由才是，且差異在 prompt 處理階段已存在**（與內核側 r5b/r6/r7 的
    「prefill `ffn_moe_*` DIFF@1」方向一致 —— 兩個獨立儀器互相支持）。
  - **下一步**：觀察點放到**第一次 2-token 池路徑 pass** 的 layer 0；**注意不能再用 `--decode-only`**
    （T 由 `ne` 自校准推出），那一步是 T=2。

  **★ 2026-09-17 00:35 取代下面的 r7 段（r10/r11）。又兩個儀器缺陷，然後是一個未裁決的矛盾。**
  **缺陷 #5**：`mul_mat_id` 的擷取點寫死 `cgc_dst_capture(..., (int32_t)(ne0*ne1))`，而該輸出是 3 維
  `[n_embd, n_expert_used, n_tokens]` ⇒ **只讀 token 0 切片**。簽名：同一 `n_embd=2048, n_expert_used=8, T=2`
  下 `ffn_moe_down-1` 報 `ne=16384` 而 `ffn_moe_weighted-1`（`mul`，全量）報 `ne=32768`。修法：拿掉 `n_words`
  參數，一律用 `ggml_nelements(op)`。**這讓 §9.18.6 第一輪對 §9.18.4 的否證失效。**
  **缺陷 #6（自己造的）**：`TAIL=1` 對任何 >32 元素的張量只寫哨兵——host 把 kargs 的 `n_ids` 傳成**窗口長度**，
  而內核的界是 `j < n_ids`、`j = n_skip + i`。實測 head 0/533 全哨兵、tail **188/205** 全哨兵。
  **⇒ r5／r5b 作廢**（它們的「幾乎全部 never」是哨兵，而全哨兵列互比是「相等」）。修法：`arg_n = n_words`。
  **讀法更正**：ids 在池路徑是 **`ffn_moe_slots-N`＝池 slot 索引**，不是專家 id；`ffn_moe_gate/up/down/weighted-*`
  **一直可讀**（`'*'` 列舉各 40 個），先前的 ABSENT 是 NODES 沒請求。
  **r10/r11 結果（兩臂各自的同臂對照皆乾淨）**：pass 0 的層 1 —— `ffn_moe_logits_raw-1`（router logits）
  **逐位元相同**（窗口 32/32、摘要四字全同），`ffn_moe_weights_norm-1` **逐位元相同**，而
  `ffn_moe_gate-1`／`up-1`／`down-1`／`weighted-1`／`out-1` **全部 DIFF@1**，且 **`sum` 與 `xor` 都不同**
  ⇒ **不是重排**（`sum`/`xor` 對置換不變），窗口真值顯示兩臂**相關係數 0.16–0.18、平均相對差 0.82–0.92
  ＝互相無關**。**⇒ 分歧在 gather 本身，不在之後的合成。§9.18.4 復活。**
  **★ 但這是未裁決的矛盾**：若層 1 的 MoE 在 pass 0 就讀到無關權重，token 流不可能一致 30 個 pass
  ⇒ 要嘛是引擎真缺陷，要嘛是**擷取位置**讀到「同形狀但不同位置」的資料（§3.2 的附帶觀察支持後者）。
  **裁決（零改碼）**：先跑 `m123_oracle_gate.py` 確認兩臂是否真 bit-identical；再用 `CGC_S1_KEEP_LEAF=1`
  看是否變 SAME。**陷阱**：`max相對差 ≈ 2.0` 看起來像「取負」，但整條分布否掉它（`b == -a` 精確 0/32）
  ——**一個極值可以偽造出一個不存在的形狀**。

  **★ 2026-09-17 00:05 更新（r7，全張量摘要）：分歧落在 layer 1 的 MoE block。**（**已被上面取代**）
  **先前 09-16 那幾輪的結論（「第一個分歧 = `conv_input-2` 的 recurrence state」等）全部作廢**，
  因為它們有兩個盲點，都在當晚被證實：**（1）32 詞窗口恆從 element 0 起算** ⇒ 對 `(2048, T)` 的輸出，
  窗口**就是 token 0**（該 pass 最舊的 token），只有 `conv_input`（ne0 = K-1+T < 32）跨全部 token
  ⇒ 那些 “SAME” 只是「token 0 相同」；**（2）`--upto 24` 只採 graph 1..23，而那 100% 是 prefill**。
  另外 `dst_filter[256]` 曾**靜默截斷**清單（23 名 = 312 字元 ⇒ 只進 19 名、第 20 名切在名字中間），
  丟掉的包含 `ffn_moe_logits_raw-2`，而分析器只列「出現過的名字」⇒ 截斷是隱形的。

  現行儀器是 **`CGC_TENSOR_CAPTURE_HASH=1`**：內核走**完整個張量**，寫 4 字
  `[sum, xor, 加權和(index+1), 元素數]`（加權和看得見**置換**；元素數讓形狀不能冒充相同）。
  **DIFF 是結論性的，SAME 是強證據而非證明。**
  r7（兩臂 × 2、同臂對照全 `never`、逐圖元素數兩臂相同 ⇒ 非形狀假象）在**同一個 pass** 內：

  | 節點 | first_diff |
  |---|---|
  | 層 1：`attn_norm-1`／`z-1`／`gate-1`／`conv_input-1`／`conv_output_raw-1`／`linear_attn_out-1`／`attn_residual-1`／`attn_post_norm-1`／**`ffn_moe_logits_raw-1`（router logits）** | 30（即 pass 0 全 SAME） |
  | **`ffn_moe_out-1`（層 1 的 MoE 輸出）** | **1** |
  | **`l_out-1`** | **1** |
  | 層 2 全部 11 個（含 `ffn_moe_logits_raw-2`／`l_out-2`） | 1 |
  | 層 0 全部 12 個（含 `l_out-0`） | 30 |

  ⇒ **層 1 的 MoE block：輸入相同（含 router logits）、專家 ids 相同（`120S/0D`）、輸出不同。**
  （ids 要到 graph 4 才開始不同：`6S/114D`，6 = 層 1、2 的 top-k 仍相同；graph 31 起 `0S/120D`，
  即 token 流本身分叉。）
  **分段陷阱**：段界是 `ffn_moe_gate-1`，所以段 i ＝ {pass P：層 1 的 MoE 出口起 → 層 2..40}
  ＋ {pass P+1：層 0 ＋ 層 1 的 attention 到 router}。`ffn_moe_out-1`／`l_out-1` 屬 **pass P**，
  `attn_norm-1`…`ffn_moe_logits_raw-1` 屬 **pass P+1**；「同一 pass」的判讀要靠段 0（21 列、全 SAME
  ＝ pass 0 的層 0 ＋ 層 1 attention），別把同段當同一 pass。
  **下一步**：摘要看不到**幅度** ⇒ 在 `ffn_moe_out-1` 上用窗口模式取實際數值比對。
  細節：`docs/S1_CAPTURE_ROUND2_20260916_2035.html`（已過時）／`docs/INSTRUMENT_COMPARE_20260916_1821.html`。

  **判別式：分歧由「服務路徑」決定，與層號無關。**
  - prefill 階梯做兩階（層 0-3、層 4-7；8 層 × 6 graph × 2 臂 = **96 個配對**），
    兩階**同構**：**slab 服務**的 graph（256/256 experts resident）**逐位元相同**；
    **partial pool 服務**的 graph（143/256）**從第一個池化層起分歧**。
  - 預測子是一個**獨立的 runtime log 行**：`CGC-PREFILL-STREAM: il=0 kind=0 ntok=182`
    恰好只出現在那兩個全 EQ 的 graph，96 個配對上完全一致。
  - **layer 0 從不分歧是「構造」不是相關性**（★ **本條已在 2026-09-16 失效，見下**）：
    當時它**根本不被池化**（`CGC-DECPROF` 報 `layers=39`，19/19 步；`layers=40` 0 次）
    ⇒「第一個分歧層 = 第一個池化層」是精確的。
    **★ 2026-09-16 起 layer 0 進池（40 層池化）**：`LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0` 的三個讀者
    原本測「存在」⇒ profile 寫的 `=0` 其實是**開啟**，六天來 layer 0 一直被排除在池外。
    修成值語意後 `=0` 才是關閉 ⇒ **現在 layer 0 是池化層**。
    - **舊 log 的 `layers=39` 是修復前的狀態，不可再當「現況」引用**；
      判別式的句子要改成「第一個分歧層 = 第一個池化層」，而 layer 0 **現在也可能是那一個**。
    - 每個 profile 的池化層數改由 `CGC_SERVER_SKIP0`（`0`/unset ⇒ 40 層；`1` ⇒ 39 層）決定。
    - `-ngl 30`（`run_n30cache.sh`）**仍應設 1**：base 的 blk.0 在那裡留在 CPU，
      要與 base 同位元就得讓它留在 CPU（§8.36）。這是**佈局對齊**旋鈕，不是品質旋鈕（CONVENTIONS A10）。
    - 新 M2 oracle 參照是 `ref_iq3_pool8gb_M2_6144_bitident_v4_skip0off.jsonl`；
      v3 已退役（兩者的 resolved env 字串**相同**，都是 `"0"`，只有語意不同 —— 見下方陷阱 17）。
  - **decode 階梯不可用於定位**（`dec-20260915-2345`）：decode 第 0 步的 layer 0 輸入由 prefill 的 KV
    導出，而兩臂 prefill 本來就不同（M1 5/9）⇒ 每個 decode 層都是上游分歧的下游，
    「輸入相同 + ids 相同 ⇒ 輸出不同」**在那裡跑不起來**。decode 對的 480/480 全層不同，
    **逐專家列多重集比對也是 480/480 不同**（0 層同多重集）⇒ **不是排列，是內容**。
  - **探針本身是決定性的**（先立前提）：同一臂跑兩個獨立 process，**480/480 逐位元相同**
    ⇒ 釘住沒有引入不確定性，跨臂的 100% 差異是臂的性質。
  - **下一步唯一還站著的候選**：量**被消費 id 背後的 slot 內容（owner-set）**。
    映射（`dec-2148`）與排列（本輪）都已被排除。**開更深的層窗價值低於表面**——
    變數是路徑不是索引。
- **前提 B 已倒（`dec-20260915-2346`）。** `publish_slot_table` **每 decode step 呼 39 次**
  （每服務層一次 region；`1014 = 39 × 26 graphs`），而**被消費的映射在 389/819 = 47.5% 的
  decode 發布上會動** ⇒ **發布不是冗餘、逐步的 host→GPU 排序要求真實存在**
  ⇒ **D3 的 `n_segs` 收縮不由此推出**，S2/S3 的驗收條件要重建。
  要量這個必須開 `CGC_S1_TABLE_CHURN=1`（未開時 teardown 會明印
  `(consumed-subset churn not instrumented: ...)`——**預設值 0 與量到的 0 必須長得不一樣**）。
- **§8.3 地雷對「被消費 id」是 inert。** `clamped_selected=0`（兩臂）。
  機制是**時序**而非政策：`ensure_batch`（填槽）→ `publish` → 寫 leaf
  （`llama-context.cpp` `:5795` 在 `:5819` 之前）⇒ publish 看到的是已填好的槽。
  **因此選「clamp 升為硬前置條件」（`CGC_S1_CLAMP_ABORT`）而不是「強制保留 ZERO slot」**——
  後者會動 `usable_slots`/`pick_slot` 的算術 ⇒ 動 pool 布局 ⇒ 移動閘門正在比較的數值。
  `clamped_table` 的 113/256（= 143 vs 256 的非 resident 比例）**結構性、不含資訊**（B1）。
- **submit-ahead 不是「被廢棄的方案」——它一天都不曾是方案**（`CGC_SUBMIT_AHEAD=1` 是故意錯的
  ceiling probe）。D3 之下它的角色由「S2 天花板」變成「**段邊界的總價值 = S3 天花板**」。
  D3 明確優於它：**D3 沒有 race**。
- **數字陷阱**：`CGC_OA_ASYNC=0` 的 0.72 t/s **不能**推論「D3 收段會慢 ~20×」——那是
  **40 splits + 序列化調度**；D3 是 **~1 split**。`CGC_OA_ASYNC` 動的是**調度策略**，
  D3 動的是 **split 數**。
- **D3 有兩個獨立前提，只有一個買到速度**（`dec-20260915-2246`、計畫檔 §9.18.7）。
  前提 A（池／表一致）⇒ 買到 bit-identical **正確**，速度增益 **0**；
  前提 B（發布離開熱路徑）⇒ 才買到 `n_segs 40 → ~1`。**前提 B 已實測失敗**（見上）。
- **16.82 t/s 不可當 D3 的驗收門檻**（同一決策）。它是 **MTP off** 量到的
  （baseline 9.16/9.45 → 16.82，×1.78，step 82.5 → 31–44 ms）；而 production（**MTP on**）
  已經 ~17 t/s。兩個 ≈1.8× 都是「填滿空窗」型手段 ⇒ **很可能吃同一份空窗、不可加乘**。
  任何 D3 驗收數字之前，先在 **MTP on** 下重測 ceiling。
- **池內容仍未「被證明是載體」**（`dec-2148`）。它證明的是 **ids 與查表實作無罪**，比「池壞」窄。
  剩下兩個子嫌疑**修法不同**：(i) **table ≠ leaf** ⇒ 修發布時序／一致性；
  (ii) **slot 內容本身錯** ⇒ 修池的填充。判別式已把範圍縮到 **partial pool 路徑**，
  但 (i)/(ii) 尚未分開。

## commit 前的驗證鏈（慣例，勿省）

```bash
# 1) 重建（incremental；不要用 REPEAT=1 全量，build_fork_llama.sh 的 REBUILD=1 會 rm -rf build/）
cd src/llama.cpp && cmake --build build -j"$(sysctl -n hw.ncpu)"

# 1b) 記自己的指紋（llama-bench 的 build_commit 不可用，見陷阱 16）
md5 -q build/bin/libllama.*.dylib build/bin/llama-server build/bin/llama-bench

# 1c) ★ 若這一輪動到 run_server.sh（啟動邏輯 / profile 預設）⇒ 先做**啟動環境指紋前後對照**。
#     dump 印完即 exit、不啟動任何東西（run_server.sh:1497），可以放心跑。
norm() { grep -E '^(ENV |ARG |CGCENV )' | grep -vE '^CGCENV LOG|\.log$'; }
CGC_DUMP_ENV=1 CGC_SERVER_PROFILE=prefill250 bash scripts/run_server.sh 2>/dev/null | norm | md5 -q
#     ★ 必須濾掉 CGCENV LOG（含時間戳）與 `free=NN%` 兩類行，否則同一配置兩跑就不同 md5
#       （實測：prefill250 連兩跑不同、prod25 恰好相同 ⇒ 不過濾時「穩定/已變」兩個判決都不可信）。
#     指紋不變 ⇒ 生產 profile 的數值不可能移動 ⇒ **不需要重新基線**（比重跑閘門便宜得多）。
#     指紋變了 ⇒ 走 2b 的控制臂流程，不要憑感覺。
#     邊界：指紋只涵蓋啟動環境、不涵蓋程式碼路徑 ⇒ 步驟 2 仍然要跑。

# 2) M1/M2/M3 + 最新 M2 oracle —— 跑兩臂：
#    預設臂是「必須 PASS」的那一臂（提交門檻），被測臂記錄差異形狀。
RUN_REPLAY_BENCH=0 python3 scripts/check/m123_oracle_gate.py --profile prefill250 --tag <tag>
RUN_REPLAY_BENCH=0 python3 scripts/check/m123_oracle_gate.py --profile prefill250 \
  --env CGC_<被測旋鈕>=1 --tag <tag>_arm

# 2b) ★ 若這一輪動到「值的語意」或任何會改數值的東西 ⇒ 先跑**控制臂**（複現舊行為），
#     它必須逐位元重現舊參照；然後才換參照。舊參照要留檔（`v3` 這種），不要刪。
RUN_REPLAY_BENCH=0 python3 scripts/check/m123_oracle_gate.py --profile prefill250 \
  --env CGC_SERVER_SKIP0=1 --allow-incomparable --tag <tag>_control   # 期望 M1/M2/M3 全 9/9
RUN_REPLAY_BENCH=0 python3 scripts/check/m123_oracle_gate.py --profile prefill250 \
  --write-ref Backup/knifeedge_matrix/ref_<...>_v4_<語意>.jsonl --ref-note "<為什麼退役舊的>" \
  --tag <tag>_newdefault
#     換完之後把 m123_oracle_gate.py 的 DEFAULT_REF 指到新參照（漏了 ⇒ 之後每一跑都 FAIL）。

# 3) llama-bench（生產標準）—— 約 7 分鐘；會吃掉機器，不要和閘門並行
RUN_REPLAY_BENCH=0 python3 scripts/check/llama_bench_matrix.py \
  --arms prod25-stream --depths 0,512,1024,2048,4096 \
  --json Backup/llama_bench/matrix_<tag>.json --md Backup/llama_bench/matrix_<tag>.md

# 4) 白皮書 HTML → docs/PREFILL250_DECODE25_WHITEPAPER_YYYYMMDD_HHMM.html
#    （附錄要寫：兩臂閘門判決、matrix 三跑對照、M2 oracle 的 md5 + cap、本檔自身指紋）

# 5) 索引（**最容易漏，漏了要補一個 follow-up commit**；順序重要：先 INDEX 後 MANIFEST）
python3 agent_harness/engine_loop/memory/build_memory_index.py
python3 agent_harness/engine_loop/index_assets.py
python3 agent_harness/engine_loop/memory/build_memory_index.py --check
python3 agent_harness/engine_loop/index_assets.py --check
python3 agent_harness/engine_loop/traces/validate.py
python3 agent_harness/engine_loop/traces/selftest.py    # 必須 8/8 被拒

# 6) commit —— pre-commit hook 會跑 scripts/check_build_tracked.sh
RUN_REPLAY_BENCH=0 git commit -F <msg-file>
```

- **`RUN_REPLAY_BENCH=0` 必須與 `git commit` 寫在「同一條指令」上。** 每次 Bash 呼叫是獨立 shell
  ⇒ 前一條 `export` 不保留 ⇒ 單獨 `git commit` 會被 hook 第 11 項擋下（找不到 `llama-server` PID）。
- **步驟 5 不要跳**：新增 trace / 白皮書但沒重跑索引，會讓 `MANIFEST.jsonl` 與
  `memory/INDEX.jsonl` 描述「上一個瞬間的樹」⇒ 只能用 follow-up commit 補。
- **hook 不需要 `--no-verify`**：其他項（dylib/symlink/exe 追蹤、`@rpath`、死鎖防護、
  原始碼↔binary 同步）都會自己 PASS，只要 build 產物有一起 staged。

### 驗證鏈的三個補充陷阱（2026-09-16 新增）

- **`run_server.sh` 的 `if/else` 巢狀會讓一個旋鈕靜默決定另一個。** 實例：`CGC_LOOP_GUARD` 的 push
  被夾在 `CGC_SOFT_POOL_L1` 的 `else` 分支裡（少一個 `fi`，`:1381-1392`）⇒ 設了 L1
  （正是註解推薦的 `CGC_SOFT_POOL_L0=48 CGC_SOFT_POOL_L1=48`）會**丟掉** phrase-loop guard，
  連 `CGC_LOOP_GUARD=1` 都要不回來。教訓：**「不存在」有兩種——預設關（pass flag 可解）
  與不可達（pass flag 無效）**；從 dump 分不出來，必須讀控制流（`grep -n` 那個旋鈕的 push 點，
  看它被幾個 `fi` 包住）。
- **dump 時環境變數要寫成獨立賦值。** `env "A=1 B=2"` 在 zsh 下不會 word-split，會變成
  「一個變數名含空格」的賦值，再被 `SERVER_ENV+=(VAR="$VAR")` 原樣印成「一個元素含空格」，
  足以偽造出一整列錯誤的對照表（本輪因此誤判過 `L0+L1` 那列，一度以為 guard 還在）。
  寫成 `env A=1 B=2 …`，或每次只給一個變數。
- **hook 在 linked worktree 是「活的」，但 `--install-hook` 會裝錯位置。** git 用
  `git rev-parse --git-path hooks/…` 找 **common 目錄** 的 hook（所以 worktree 共用主 repo 的）；
  但在 worktree 內跑 `check_build_tracked.sh --install-hook` 會寫到第 98 行的
  `--absolute-git-dir`（**worktree 私有**目錄）⇒ **印出「已安裝」卻永遠不會被執行**。
  要裝就在主 worktree 裝。完整使用說明：`docs/PRECOMMIT_HOOK_GUIDE_20260916.html`。
- commit message 用 `-F -`（heredoc）或 `-F <file>`，不要用 `-m`：內容含中文、`§`、反引號與換行。
- **排除清單**：`hf-space-deploy/` 是**嵌套 git repo**（已被 gitignore，勿強加）；
  `.tmp_*.py` 是 scratch；`data/replay_bench/` 是 stale 資料；
  `Backup/` **整個 gitignored** ⇒ 附件不能進版控，白皮書要寫清楚可引用副本的路徑。
- **閘門結果與吞吐要分開講**：閘門（M1/M2/M3）是**相對不變量**，不是絕對正確性；
  一個**確定性**的錯誤在兩側完全相同、依構造不可見。cache-free 絕對 ground truth 在 16GB 上不可得
  （需 256 experts × 40 層常駐 ⇒ Metal OOM）。
- **吞吐一律不可由單次 matrix 引用**：同一 nominal 配置的三跑（其中兩跑同 binary）
  極差達 **33.6%**（`pp@1024` 79.88 → 106.69），**方向不一致**，且連 pool 行為都不同
  （hit 97.1%/reads 519501 vs hit 97.3%/misses 16185）。列原始讀數可以，**下結論不行**。
