# MEMORY_PERF — profile／模型／幾何／速度／散熱

> **這是快照，不是權威副本。**
> 權威位置：`.workbuddy/memory/MEMORY_PERF.md`（由 host 持續寫入）。
> 本檔於 2026-09-17 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
> 底下的內容能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 索引與漂移檢查見 `agent_harness/engine_loop/memory/INDEX.jsonl`。

> **這是 `MEMORY.md` 的主題分檔（2026-09-17 拆分），不是歷史存檔。** 動手前的規則、指令入口、
> 索引導覽在 `MEMORY.md`。**要引用任何 t/s、profile、幾何、散熱數字之前讀本檔** —— 本專案的
> 「decode 速度」有四個互不相容的定義，不指名就一定會引用錯。

## profile／模型／幾何

- **統一 profile（09-16 定案）＝ `prefill250` ＋ `CGC_SPAC=1`（alpha 0.75）**，同時服務 prefill 與 decode。
  依據：`CGC_POOL_MAX_TOKENS`（`llama-graph.h:18-37`，預設 8、可調 [2,64]）的註解把「MTP verify batch」
  綁在它上面，夾具只在 `!cgc_prefill_stream` 時生效 ⇒ `prod25` 把池路徑鎖在 8，而 **M1／M4 的槓桿住在
  寬 batch（whole-layer slab）**。
- **SPAC=1 的證據**（`Backup/run_unified_ab.sh`，交錯、**09-16 21:47 第二次獨立確認**）：ctx 8192 的 d512
  上**均值 +16.5～17%**（9.21 → 10.73；與 08:48 的 9.25 → 10.85 互在 2% 內）、**離散 ±2.97 → ±0.04**
  （§EN-4／§EN-5）—— **主效果是移掉不穩定，均值增益是附帶的**。
  - ⚠️ **alpha 的值還沒有交錯證據**：C2 的循序 α 掃描（a050/a075/a090）已被 21:52 的 A/B 判為受順序／
    熱污染（它給出 +6.2%，真值是 +16.5%）⇒ **四個 arm 的 α 排序全部不可引用**；要選 α 得做**輪轉式**掃描。
  - ⚠️ **`run_unified_ab.sh` 的 A 臂必須寫 `prefill250:CGC_SPAC=0`**：profile 自 20:33 起自己會設
    SPAC=1，裸 `prefill250` 會讓 A≡B 而讀成「SPAC 沒效果」。
- **量測形狀**：prefill 用 profile 的 `-b/-ub 5632`；**decode／depth 矩陣用 `-b 512`**（5632 於 `-d≥512`
  在 16 GB 會 OOM）。
- **模型**：`prod25` → `Nail-…-MTP-…-denseIQ4X.gguf`（13.6 GB）；**`CGC_SERVER_MTP=0` 換成非 MTP 的
  `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`**（`run_server.sh:154`）⇒ **引用任何數字都要指名模型**。
  **不是 Gemma 4 26B-A4B**（更早的設定，忽略）。家族 `qwen35moe`、41 blocks、`full_attention_interval=4`
  ⇒ layer 0/1/2 是 gated delta-net，**layer 3 才是第一個 full attention**。
- **幾何**：pool 8 GiB → 143 slots/layer；`LAYER_CAPS 40-40:256`；CTX 4096（prefill250 8192）；
  `SPEC_DRAFT_N_MAX=3`。三支柱 bit-identical：`CGC_MM_BITIDENT=1`／`SERVER_MTP_NO_WARMUP=1`／
  `SERVER_NO_SEQ_RM_PROBE=1`。**池壓力熱點層會變**（layer 0 與 layer 2 都出現過）。

## 里程碑現況（M0–M6；**09-17 查證**，不是憑記憶）

- **M0 量測能力：完成。**
- **M1 解耦 pool 與圖：做了一半以上、卡住**（不是「未開始」——這個錯誤我犯過一次）。
  數值那一半 **09-14 已達成**：2/4/6/8/10 GiB 全部 **M1 = M2 = M3 = 117/117**，2 GiB 是唯一
  `union > slots` 那格（compacted gather），`union-routable` PASS，RSS 達標。**兩條沒過**：
  ① `decode 不得退步` ⇒ 2 GiB（gather＋slab）**6.36** vs 8 GiB（pool）**8.87 t/s** ＝ **0.72×**；
  ② `prefill chunk 2048` ⇒ 缺工作項 2。**工作項 1**（`CGC_POOL_SPLIT=1`，保持 expert tensor 全寬）
  實作了但被判 **EXPERIMENTAL, NOT USABLE**：Blocker A（寬 tensor 讓 gather 把 Metal buffer 的指標
  重指到 Metal 不知道的 host 指標 ⇒ **靜默** `tensor buffer is nil`、**M1 2/42**；且 16 GB 上
  warmup OOM），Blocker B 已於 **09-16** 修好。**工作項 2（phase split）未實作**（src 無
  `T_prefill`／`PREFILL_GRAPH`／`DECODE_GRAPH`）。
- **M2 prefill 整層串流：核心機制已落地**（`CGC_PREFILL_STREAM=1` ＋ `CGC_GATHER_SLAB_CAP=256`，
  `prefill250` 用它跑到 250+）。**它的五條離開條件未逐條核對。**
- **M3 decode 的 compute 削減：未開始**（依賴 M1）。兩個探針試過且**不可引用**：`CGC_MMV_FUSE`
  （MoE gather 融合）輸出損壞且更慢；`CGC_SUBMIT_AHEAD`（序列化）天花板 ×1.711 但輸出損壞。
  離開條件「decode（MTP off）≥ 15 t/s」未達。
- **M4 MTP 拒絕取樣 ＋ verify 真批次：未開始**（accept 19.9%）。
- 文件：`docs/ROADMAP_PREFILL250_DECODE25_2026-09-13.md`（M0–M6 定義）；
  **`docs/roadmap-2026-09-14/ROADMAP_PREFILL250_DECODE25_2026-09-14.html`**（M1 的實作與量測結果、
  「M1 還沒完成的離開條件」、「M2 的狀態」，**09-14 之後未更新**）；
  `docs/M1_POOL_SPLIT_COST_2026-09-14.md`（Blocker A/B 的成本分析）。
- ⚠️ **`M1/M2/M3` 在本 repo 有兩個意思**：roadmap 的里程碑 vs **D5 的三個判決指標**
  （今天跑是 M1/M2/M3 各 9/9、`comparable=true`）。被問「M1/M2 狀態」時先確認是哪一個。

## decode 速度：現在到底多少（09-16 20:36 盤點）

「decode 速度」有四個互不相容的定義，引用必須指名：

| 定義 | 數字 | 條件 |
|---|---|---|
| **llama-bench 暖平台**（丟 rep1） | **10.78／10.91**（d512）；11.3（n=128 平台） | `prefill250+SPAC=1`、`-b512`、NOMINAL、**MTP=0 模型** |
| llama-bench `-d 0`（冷格） | 9.52–9.79 | 同上 |
| **decode_bench（HTTP）** n=128 | **12.36**（NOMINAL）→ 10.64（HEAVY） | 同 env、同模型、交錯 |
| **llama-server 非 MTP**（`p25-gputime`、n=124） | **12.95 可持續** | 全程 NOMINAL；**不要引用 20.3**（n=24 短爆） |
| **「25 t/s」的出處** | 09-05 的 **27.71** | **71 slots／4 GiB pool ＋ L0=32/L1=32 分區**，`draft accept 0.9974`（3.99 token/step、144 ms/step）——**不是**現在的 143 slots／8 GiB |

- **「加大 pool／提高 hit rate 是槓桿」已推翻**：71 slots 的舊幾何（09-05）反而快（該筆 `resident=0.00 MiB`
  ⇒ 另一條填充路徑）。09-15 同幾何 MTP-on 只有 6.48 ⇒ **MTP 當前是淨損失**（`llama-speculative-simple`：
  無投機 **7.483** vs draft-mtp **6.517**）。
- **llama-bench 對 MTP 是瞎的**（`sampler|speculat|draft|MTP` 零命中、token 是 `rand()%n_vocab`）⇒ 當不了
  instrument of record；「量不到」已解決（用 `llama-speculative-simple`），露出的是 accept 太低（19.9%）。
- **兩個 decode 儀器不能並排引用**：同場交錯 LB 9.4 vs DB 12.4（n≈128），大半是「**第一個 rep 是冷的**」
  ⇒ 丟掉後 **11.3 vs 12.4 = 1.10×**；殘差與 n=24 的 2.6× **皆未歸因**。
  `depth` 是 llama-bench 唯一有效的暖機軸（`-d≥512` 才 10–13）；`llama-bench.cpp` **沒有 `srand`**。
- 病因已證是**序列化**（`CGC_SUBMIT_AHEAD=1` 讓每步 82.5→31–44 ms，~44% 可移除；**該探針輸出損壞**⇒
  其 16.82 t/s 不可引用）。
- **16–18 的歸屬（易搞混）**：那是 **M3／D3 的目標區間**，**不是 S1 的產物**。`dec-20260915-2246` 拆成
  **前提 A＝池／表一致 ⇒ 速度增益 0**；**前提 B＝`publish_slot_table` 發布離開熱路徑 ⇒ 才買到
  `n_segs 40 → ~1` 與那 ~44%**。折扣：16.82 是 **MTP off** 量的（production MTP on 已 ~17 ⇒ 兩個 ≈1.8×
  可能吃**同一份空窗、不可加乘 ⇒ `ceiling 必須在 MTP on 下重量`，還沒做**）；輸出損壞。
  **⇒ 16–18 掛在 M1→M3，不是 S1。** 到 25 的算術：9–10 × 1.78 ≈ 16–18，**還缺 ~1.5×**，只能來自 M4
  （依賴 M1/M3）。
- 文件：`docs/ROADMAP_PREFILL250_DECODE25_2026-09-13.md`（M0–M6）、
  `docs/INSTRUMENT_COMPARE_20260916_1821.html`。

## prefill 250 的條件式交付

`CGC_SERVER_PROFILE=prefill250`（`-b/-ub 5632`、`CGC_PREFILL_STREAM=1`、`CGC_GATHER_SLAB_CAP=256`、
pool 8 GiB、ctx 8192）**必要非充分**；還要散熱前提 ＋ 量測紀律。

- **權威儀器（非 root、11 ms）**：`notifyutil -g com.apple.system.thermalpressurelevel`
  （`0=Nominal 1=Moderate 2=Heavy 3=Trapping 4=Sleeping`）。**判準：發射前讀到 `0`。** 分離度（request
  級、零重疊）：發射 0 → **6/6 ≥250**；發射 1 或 2 → **0/21**。反面教材：`NSProcessInfo.thermalState`
  367/367 讀 `fair`、零區辨力——**找到介面 ≠ 找到儀器**。
- **「距上次持續 prefill ≥150 s」作為充分條件已被推翻**（安靜 18 s → 289.86；34 s → 166.95）⇒ 自變數是
  **累積負載（同序列第幾次啟動）**；機制是 DVFS 階（1470→928→618 MHz）與熱壓同秒。
- **可交付述句只有「發射時讀到 `0` 的那一臂，req1–req3 全部 ≥250」**。**不能**說「250 隨時可重現」。
  熱態平台 167–201。`t(token) = 0.5575 ms + 4226/f_eff(MHz)`；純階反讀 1470→291、928→227、618→135。
  **250 不是上限**；`swap` **不是因是果**。
- **閘門**：`COLD-STATE` 需安靜 ≥1800 s，否則 `HOT-STATE`；**HOT 不得當交付數字**。**機器地板是 HEAVY**
  （零 llama 行程時 thermal=2、GPU util 20%、Electron ~103%）⇒ agent UI 造成。
  `ARMS=2 bash Backup/run_thermal_gate.sh`（不成立 exit 3，fail closed）；
  `docs/PREFILL250_CONDITIONAL_DELIVERY_20260916.html`。
- **未結清**（09-17 03:0x）：`prefill250 + CGC_SPAC=1` 的 COLD 單臂量到 req1 ＝ **244.96**（未達 250），
  且**無法與更早的 254.29/282.38/265.27 比**（不同 build、機器狀態不同，更早那次連 COLD/HOT 標籤都沒有）
  ⇒ 唯一能裁決的是**同 build 的 COLD 交錯 A/B**（`Backup/run_spac_cold_ab.sh`，off/on/off/on，每臂前
  1800 s 靜置）。

## 指紋／戳記

可比性 key ＝ **pool／engine／weights／launch 四組**（`decode_sweep.build_fingerprint()` 用 glob 8 鍵；
`knifeedge_matrix.{source,pool_geometry,binary,model}_stamp`；`mtp_head_identity.fingerprint()`；
A11 啟動環境指紋），`suite` 是第五軸，`harness.script_digest` **刻意不進 key**。

- `build_fingerprint()` **用 glob**（`libggml*.dylib`／`libllama*.dylib` ＋ server，8 鍵）；舊版只手動雜湊
  3 檔、**漏了 `libggml-base`** ⇒ 兩跑指紋相同被誤判可比；**v1（3 鍵）的歷史列不可比**。
- `binary_stamp()` 只記 `{size, sha256(前 64 KiB)}`，而 Metal kernel 在 offset ~169 KB ⇒ 判別力「安全但靠
  意外」（靠 ld64 把內容衍生的 LC_UUID 寫在前 1,881 位元組）。**不是設計出來的**，沒有測試釘住。
- `source_stamp`／`pool_geometry_stamp` 的來源清單是**人工列舉**；不在清單但會改數值的至少還有
  `ggml-metal.metal`、`ggml-backend.cpp`、`llama-model-loader.cpp`、`ggml-metal-context.m`（靠
  `binary_stamp` 兜住）。
