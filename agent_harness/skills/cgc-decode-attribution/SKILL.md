---
name: cgc-decode-attribution
description: 在 flashkv-devserver（TurboFieldfare / llama.cpp CGC fork）上把一個 decode 步歸因到 GPU 執行、CPU 序列化、池/IO 或編碼，並用既有儀器取得可證的判準數字。當使用者問「decode 為什麼慢」「wait 是什麼」「瓶頸在哪」「要不要做 fusion / kernel 優化」或要求 decode 相位分解、GPU 佔用率、速度槓桿排序時使用。
agent_created: true
---

> **這是快照，不是權威副本。**
> 權威位置：`~/.workbuddy/skills/cgc-decode-attribution/SKILL.md`（由 host 持續寫入）。
> 本檔於 2026-09-16 手動複製進 repo，唯一目的是讓 `agent_harness/` 底下的內容
> 能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 要改 skill 請改原檔，再重跑 `Backup/import_harness_snapshot.py`。

# CGC decode 步歸因（儀器與陷阱）

專案：`/Users/alexchuang/Documents/flashkv-devserver`
目標配置：prod25 / prefill250，Gemma 4 26B-A4B，M4 Air 16GB。

## 鐵律

1. **每次跑之前 `pkill -9 -f llama-server`**，並用交錯 A/B ×3 + md5 對比。
2. **啟動器有 env allowlist**：`scripts/run_server.sh` 只透傳列出的 `CGC_*`，
   未列出的**靜默丟棄** → 「沒效果」與「沒設到」長得一模一樣。要用任何新開關前，
   先確認它在 `run_server.sh` 裡有 `if [ -n "${VAR:-}" ]; then SERVER_ENV+=(VAR="$VAR"); fi` 區塊。
   歷史上 `CGC_MMV_FUSE`、`CGC_GPU_TIMING`、`CGC_SUBMIT_AHEAD` 都因此踩過。
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
| `GGML_SCHED_DEBUG=2` | per-node 後端 + `GET_CAUSE` | **唯一**能回答「這個 node 落在哪個 backend、它的 src 從哪來」的儀器（=1 只有 `## SPLIT`，且走 `GGML_LOG_DEBUG` 會被預設 verbosity 濾掉） |

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

## 現況結論（2026-09-15 收盤，供後續對照）

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
