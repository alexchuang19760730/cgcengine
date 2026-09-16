---
name: cgc-decode-attribution
description: 在 flashkv-devserver（TurboFieldfare / llama.cpp CGC fork）上把一個 decode 步歸因到 GPU 執行、CPU 序列化、池/IO 或編碼，並用既有儀器取得可證的判準數字。當使用者問「decode 為什麼慢」「wait 是什麼」「瓶頸在哪」「要不要做 fusion / kernel 優化」或要求 decode 相位分解、GPU 佔用率、速度槓桿排序時使用。
agent_created: true
---

> **這是快照，不是權威副本。**
> 權威位置：`~/.workbuddy/skills/cgc-decode-attribution/SKILL.md`（由 host 持續寫入）。
> 本檔於 2026-09-17 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
> 底下的內容能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 要改 skill 請改原檔，再重跑 `python3 agent_harness/scripts/import_harness_snapshot.py`。

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
| `CGC_TENSOR_CAPTURE=<精確節點名>`（＋`CGC_TENSOR_CAPTURE_WORDS`，預設 32 上限 32） | `CGC-IDS-CAP path=DST name=<節點>.dst` | **輸出張量**的快照（2026-09-16 新增）。重用 `CGC_IDS_CAPTURE` 的同一個內核（那個內核與 ids 無關，就是「把 stride 個 int32 抄進 slot」）⇒ `.metal`／`impl.h`／`device.*`／`context.m` 一行未改，比較器也原封不動。**必須與 `CGC_IDS_CAPTURE=1` 一起開**：ids 列是比較器切 graph 的依據，少了它們 dst 列會落進一個偽 graph 而被報成 IDENTICAL（假陰性）。**只掛在 `ggml_metal_op_mul_mat_id`** ⇒ 只能擷取 gate／up／down 的輸出。名稱**精確比對**（`ffn_moe_down-1` 是 `-10..-19` 的子字串）。★ **需要一個 `ggml_metal_encoder_memory_barrier`**：ids 運算元的生產者在很多節點之前（已沉降），dst 的生產者是**緊鄰的前一個 kernel**，而這個 fork 支援內核併發。沒有屏障時讀數**不可重現** |
| ~~`CGC_S1_OUT_CAP=1`／`=pre`~~（**既有，較舊**） | `CGC-S1: OUT`（`fnv1a64` ＋ `vals=[...]`）、`OUTSET` | **同一件事的舊版**：釘住 `ffn_moe_down` 的輸出。差別有兩點——(1) 它用 `ggml_set_output` **釘住張量**，而那會動 allocator／graph（原始碼自己寫「診斷臂專用、閘門臂永不可開」）；(2) 它印**雜湊**加少量值。⇒ **要「不擾動圖」的讀數用 `CGC_TENSOR_CAPTURE`，要層窗與雜湊用這一支。** ⚠️ 2026-09-16 交叉確認**未成立**：`CGC_S1_OUT_LAYERS=1` 跑出來的列是 `il=27` 而不是 layer 1，而 480/480 全不同 ⇒ **它的參數語意尚未弄清，不要拿它當獨立確認** |
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

## 兩個 decode 儀器不能並排（2026-09-16 實測；細節 `docs/INSTRUMENT_COMPARE_20260916_1821.html`）

**`llama-bench` 與 `decode_bench` 的數字不得互相引用，也不得放在同一句話裡比大小。**
同 env、同模型、同 n、同場交錯量到的是 **9.4 vs 12.4**（n≈128）與 **7.2 vs 18.7**（n=24）。

要配對，env 必須是解析出來的而不是手抄的——`llama_bench_matrix.py` 支援
`PROFILE:ENV=VAL`，所以

```sh
python3 scripts/check/llama_bench_matrix.py \
  --arms 'prod25:CGC_SERVER_MTP=0;CGC_GPU_TIMING=1;CGC_DECODE_PROFILE=1' \
  --prompt 0 --gen 128 --depths 0 --reps 3 --json Backup/phase_decomp/lb_x.json
```

逐字就是 `decode_sweep --arms p25-gputime` 的 env（兩邊都經 `run_server.sh CGC_DUMP_ENV=1`）。

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

輸出是 `decoded N tokens in X seconds, speed: Y t/s` ＋ `n_drafted/n_accept/accept%` ＋
`common_perf_print`。**這是第三個儀器**（視窗含 sampling 與 draft/verify）⇒ 不可與 llama-bench
或伺服器的數字並排。首測（n≈48、單樣本、示範非 A/B）：no-spec **7.483** vs MTP **6.517**
（同 NOMINAL；accept 36–53%）⇒ MTP 目前慢 **1.15×**，方向與伺服器結論一致但幅度不可互引。

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
