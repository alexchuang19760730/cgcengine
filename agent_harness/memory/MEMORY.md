# flashkv-devserver — 專案長期筆記

> **這是快照，不是權威副本。**
> 權威位置：`.workbuddy/memory/MEMORY.md`（由 host 持續寫入）。
> 本檔於 2026-09-16 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
> 底下的內容能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 索引與漂移檢查見 `agent_harness/engine_loop/memory/INDEX.jsonl`。

llama.cpp 的 CGC fork：Metal + **expert cache pool**（專家權重常駐 SSD→池，decode 為 IO-bound）。
專案根 `/Users/alexchuang/Documents/flashkv-devserver`（**是 git worktree**，`.git` 是一個檔案）。
細節與逐日經過在 `.workbuddy/memory/2026-09-1{5,6}.md`（append-only）；本檔只留長期可引用的部分。

## 分工（2026-09-16 起）

**`agent_harness/` 歸另一個 WorkBuddy session；引擎層（`src/`、`scripts/check/`、decode／prefill 量測）
歸本 session。** 動手前後各跑一次 `git status --porcelain -uall`；看到**不是自己的**
modified／staged 檔就停手（對方可能正在做 E1 的 `git mv`，600+ rename）。
**不要在對方 working 時重生索引或 commit**，也不要改 `agent_harness/` 底下的東西。

## 模型與 profile

- **統一目標（2026-09-16 定案）**：一個 profile ＝ **`prefill250` ＋ `CGC_SPAC=1`（alpha 0.75）**，
  同時服務 prefill 與 decode。依據：三個 decode 臂只差 6 項 env，真正有語意的只有 `CGC_SPAC`
  與 CTX；而 `CGC_POOL_MAX_TOKENS`（`llama-graph.h:18-37`，預設 8、可調 [2,64]）的註解自己把
  「MTP verify batch（n_max+1 ≤ cap）」綁在它上面 ⇒ `prod25`（無 PREFILL_STREAM）把池路徑鎖在 8，
  而 **M1／M4 的槓桿住在寬 batch（whole-layer slab）的世界**。
  否證實驗（`Backup/run_unified_ab.sh`，交錯 A/B/A/B、每臂等 NOMINAL）：SPAC=1 在 d512
  **均值 +17%（9.25 → 10.79）且離散從 ±2.97 崩到 ±0.04**（逐 rep 6.28–12.21 → 10.74–10.82）。
  **量測形狀**：prefill 用 profile 的 `-b/-ub 5632`；decode/depth 矩陣用 `-b 512`
  （5632 在 16 GB 上於 `-d≥512` 會 OOM）。細節見日誌 §EN-4／§EN-5。
- `prod25` 用 `models/gguf/Nail-…-MTP-…-denseIQ4X.gguf`（13.6 GB，含 MTP）；**`CGC_SERVER_MTP=0`
  會換成非 MTP 的 `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`**（`run_server.sh:154`）⇒ 引用「prod25 的數字」
  必須指名是哪一個模型。**不是 Gemma 4 26B-A4B**（那是更早的設定；舊筆記寫 Gemma 要忽略）。
- 幾何：pool 8 GiB → 143 slots/layer；`LAYER_CAPS 40-40:256`；CTX 4096（prefill250 是 8192）；
  `SPEC_DRAFT_N_MAX=3`；sweep 一律 `CGC_SERVER_MTP=0`。三支柱 bit-identical：`CGC_MM_BITIDENT=1` /
  `SERVER_MTP_NO_WARMUP=1` / `SERVER_NO_SEQ_RM_PROBE=1`。
  池壓力熱點的層**會變**（觀察到 layer 0（250 distinct）與 layer 2（218–248）），不是固定的 layer 2。

## 入口（照抄）

```sh
cmake --build src/llama.cpp/build --target llama-server -j 8   # 產物在 src/llama.cpp/build/bin
RUN_REPLAY_BENCH=0 python3 scripts/check/decode_sweep.py --profile prod25 \
  --arms p25-gputime,<臂> --rounds 3 --warmup 0 --n-predict 24 \
  --json Backup/phase_decomp/<名>.json --force
python3 scripts/check/m123_oracle_gate.py --tag <標籤>          # D5，自己起 server（先確認 8080 空）
```

- **`run_server.sh` 有 env allowlist**：未列出的 `CGC_*` **靜默丟棄** ⇒「沒效果」與「沒設到」同形。
  加新開關一定要同步加 `if [ -n "${VAR:-}" ]; then SERVER_ENV+=(VAR="$VAR"); fi`。
- 索引重生**順序固定**：先 `agent_harness/engine_loop/memory/build_memory_index.py`（寫 INDEX），
  再 `agent_harness/engine_loop/index_assets.py`（MANIFEST 記 INDEX 的 bytes/mtime）。顛倒 → `--check`
  報漂移，且只重生 manifest 不會修。**memory 寫完要放在索引重生之前。** `index_assets.py` **不要加
  `--out`**（相對路徑會寫出第二份 manifest）。掃描範圍＝`scripts/check/*` + `docs/*` +
  `agent_harness/engine_loop/*` + `.workbuddy/memory/*`；**不含 `agent_harness/memory/` 與 `skills/`**。

## agent_harness

- 兩個同構迴圈：`tb_loop/`（Terminal-Bench）與 `engine_loop/`（引擎調校）。學習發生在 harness 狀態（文本）。
- `CONVENTIONS.md` = 判準憲章（A 可引用／B 診斷／C 讀原始碼／D 流程／E 自我約束），每條要指到證據。
  **它同時是 `engine_loop/sft_pi/` 的 system prompt 與 `harness_engine/` 的注入來源** ⇒ 改它要走
  PLAN §6.3 閉環對照。
- `engine_loop/traces/`：`episodes.jsonl` / `decisions.jsonl`（含 `ruled_out`）/ `lessons.jsonl`。
  `validate.py`、`selftest.py`（10/10）。`emit_episodes.py` 是**純讀** T0 emitter。
  硬規則：`build == null ⇒ usable_as_evidence == false`；有生成 ⇒ `answer_md5` 非空；
  `judgement == refuted ⇒ superseded_by` 非空。
- `engine_loop/memory/` **只放索引**（`INDEX.jsonl`），原檔在 `.workbuddy/memory/`。
- **D6 修訂（2026-09-16）**：原 D6「記憶只由索引進入 loop，不複製」仍管 **loop 讀什麼**；另承認
  `agent_harness/memory/`（3 檔）＋ `agent_harness/skills/`（3 skill）是**非權威 dated 快照**，
  唯一用途是跨機器搬運——`agent_harness/scripts/auto_git_push.ps1` 會 `git add agent_harness` 後 push，
  而**索引運送的是指標、到不了那條線**。兩者要一起重生：先
  `agent_harness/scripts/import_harness_snapshot.py`，後索引。
  **兩個 `--check` 都不驗快照。** `eng-bound-0004`。
  （匯入器 2026-09-16 從 `Backup/import_harness_snapshot.py` 搬進版控——它原本在 `.gitignore:396`
  排除的 `Backup/` 底下，而「一個要跨機器的機制，自己得先能跨機器」；同時兩個硬編碼清單
  `SKILL_NAMES`／`MEM_FILES` 改成 glob，所以新增 skill 或新的一天都不必改清單。）
- **E2 欠帳（已認領，別讓它變無主）**：§E 要求改 CONVENTIONS.md 走一次 §6.3 閉環對照；D6 修訂當天
  `sft_pi/` 與 `harness_engine/` **都還不存在** ⇒ 改不到任何執行時行為，但對照**尚未執行**。
  E2 一建立那兩個目錄，欠帳就變成可執行、可失敗的檢查。記在三處：**PLAN §9 E2 欄** ＋
  `CONVENTIONS.md` D6 修訂段 ＋ 本檔。
- **不要把 `agent_harness/` 拆成獨立 git repo**：它刻意不自足——`index_assets.py` 是 `REPO=HERE/../..`、
  `build_memory_index.py` 是 `HERE/../../..`、`emit_episodes.py` 用 `find_repo()` 往上找 `.git`，
  而它索引的 `scripts/check/*`、`docs/*`、`Backup/cgc_logs`、`.workbuddy/memory/*` 全在它外面。

## 指紋／戳記：7 個 ＋ 3 個檔案層索引

7 個＝`decode_sweep.build_fingerprint()`、
`knifeedge_matrix.{source_stamp, pool_geometry_stamp, binary_stamp, model_stamp}`、
`mtp_head_identity.fingerprint()`、A11 啟動環境指紋；檔案層＝`MANIFEST.jsonl` / `INDEX.jsonl` /
`SNAPSHOT.jsonl`。可比性 key（`record_geometry_key`）＝ **pool / engine / weights / launch 四組**，
`suite` 是**分開的第五軸**，`harness.script_digest` **刻意不進 key**。

- **`build_fingerprint()` 用 glob**（`libggml*.dylib` / `libllama*.dylib` + server，**8 鍵、鍵名不含
  版本號**；`decode_sweep.py:34`，`ab_interleave.py` 委派同一份實作）。原本只手動雜湊 3 檔、
  **漏了 `libggml-base`**（`ggml-backend.cpp` 的排程器與 `CGC_OA_ASYNC` 閘門就在裡面）⇒ 21:34 與
  21:41 兩跑指紋相同、被「只在同一指紋內比較」判成可比。**v1（3 鍵）的歷史列在這一維不可比。**
  `eng-mh-0007`。
- **`binary_stamp()` 的判別力是「安全，但靠意外」**（`eng-mh-0037`，陽性對照）：它只記
  `{size, sha256(前 64 KiB)}`，而 Metal kernel 位元組在 `libggml-metal…dylib` 的 **offset ~169 KB**
  ——可是**同長度**的 kernel 改動（`ggml-metal.metal:3100` 的 `1.0f`→`1.1f`）**仍然被偵測到**：
  ld64 把**內容衍生的 LC_UUID** 寫在前 1,881 個位元組裡（`cmp -l`：80 個差異中 15 個 ≤64 KiB）。
  偵測力**不是設計出來的**，依賴連結器行為、且沒有測試釘住它。產物在 `Backup/positive_control/`。
  同一實驗附帶確立：**改回原始碼重建後 md5 回到 `f2d1c961…`、size 954,920 ＝ byte-identical**。
- `source_stamp` 的 `NUMERIC_SOURCES`（4 檔）與 `pool_geometry_stamp` 的 `POOL_GEOMETRY_SOURCES`
  （3 檔）是**人工列舉清單**——同一個 bug 的形狀。不在清單裡但會改數值的至少還有
  `ggml-metal.metal`、`ggml-backend.cpp`、`llama-model-loader.cpp`、`ggml-metal-context.m`；
  目前靠 `binary_stamp` 兜住。

## 量測衛生

- **md5 只在同 build 指紋 + 同實際生成長度（`predicted_n`）下可比**（與長度無關的證據是 `sample`
  前綴）。`pkill -9 -f llama-server` 每次跑前；交錯 A/B ×3 + 配對中位。
- macOS BSD `grep` 不支援 `\|`（一律 `-E`）；**本 sandbox 的 bash `grep` 會靜默失效 ⇒ 用內建 Grep
  工具**；zsh `*.log` 無匹配會不執行整條指令；**沒有 `timeout`/`gtimeout`**；`ps` 被擋、`pgrep` 可用。
- **`CGC-MMID-ASSERT` 的 `id_oob` 不能當「消費者讀到什麼」的證據**：它在 **encode 期由主機**讀
  `op->src[2]->data`；ids 是裝置 gather 輸出時量到的是**寫入前殘留**（同臂分段/非分段讀到 0 與
  202000，而兩者輸出都是可讀法語）。`POST` 是 **sync 之後**的讀回 ⇒ 這類問題只能用**內核側**讀數。
  `eng-mh-0008`。
- **建置新鮮度的判準是「建置輸出有沒有編譯行」**，不是 exit code、也不是產物的存在或 mtime
  （`cmake --build` 在「已最新」與「剛編好」都回 0）。一旦印出 `Building …`，先前在同一棵樹上取得的
  所有數字全部失效。實例：13:00 改了 `ggml-metal-context.m`（只改註解）沒重建 ⇒ 12:44–13:50 每一筆
  量測（含出廠驗收與一份 D5 PASS）都屬於舊產物。
- **D5 的 oracle 旋鈕已釘住**：`ORACLE_PINNED_ENV`（`BATCH/UBATCH=6144`，放在 `DEFAULT_REF` 旁），
  `--env` 可覆蓋但**有效集合先印**；重新基線＝`--write-ref` ＋ `--no-pin-oracle-env`。
  **先看 `comparable` 再讀 M1/M2/M3**——`comparable=False` 時那三個 9/9 不是裁決。
  殘留：出廠預設 5632 沒有自己的參考檔。
- **「兩次量測之間只有一個產物不同」≠ 單變數實驗**（前提是**交錯**）。兩個候選都無法排除時，
  正確述句是「未歸因」，並指名能裁決它的那個實驗。
- 判讀產物指紋兩個陷阱：重寫後可能與 HEAD **逐位元相同**（install name 沒動就不 relink）；
  `cmp -l` 的差異數會被**內嵌建置戳記變長造成的整體位移**放大（判準是 `strings` 差集，不是 `cmp` 計數）。

## prefill 250 的條件式交付

設定是三層，只有一層在設定檔裡：(1) `CGC_SERVER_PROFILE=prefill250`（`-b/-ub 5632`、
`CGC_PREFILL_STREAM=1`、`CGC_GATHER_SLAB_CAP=256`、pool 8 GiB、ctx 8192）**必要，非充分**；
(2) **散熱前提**；(3) 量測紀律。

- **權威儀器（非 root、11 ms）**：`notifyutil -g com.apple.system.thermalpressurelevel`
  （`0=Nominal 1=Moderate 2=Heavy 3=Trapping 4=Sleeping`；正是 `powermetrics` thermal sampler 讀的
  notify(3) key）。**判準：發射前讀到 `0`。** 分離度（request 級、零重疊）：發射 0 → **6/6 ≥250**
  （253.42–271.64）；發射 1 或 2 → **0/21**（104.88–211.65）。讀的是**發射前那一刻**，不是全程。
  反面教材：`NSProcessInfo.thermalState`（JXA、非 root、四級）**367/367 讀 `fair`**，零區辨力
  ——**找到介面 ≠ 找到儀器**。
- **「距上次持續 prefill ≥150 s」作為<u>充分</u>條件已被推翻**：安靜 18 s → 289.86 t/s；
  安靜 34 s → 166.95 ⇒ 安靜最長的最慢、最短的最快 ⇒ 自變數是**累積負載（同序列第幾次啟動）**。
  機制沒變：DVFS 階由 thermal governor 選定（1470→928→618 MHz），熱壓與時脈下降同秒。
- **可交付述句只有「發射時讀到 `0` 的那一臂，其 req1–req3 全部 ≥250」**（2026-09-16 量到
  254.29 / 282.38 / 265.27）。**不能**說「250 隨時可重現」，也不能保證 req2/req3 ≥250
  （第 2 臂的 req3 是 233.51）。熱態平台 167–201；13:45 soak 平台 118–155（**差 55 t/s 未歸因**）。
- `t(token) = 0.5575 ms + 4226/f_eff(MHz)`（3 點擬合）：地板項不隨時脈縮放、冷態佔 15.4%。
  純階反讀：1470 → 291、928 → 227、618 → 135 t/s。**250 不是上限**（1470 階純階預測 291.3）。
- **`swap` 不是因，是果**（32 分安靜期間自 `5066.06M` 回 `2743.00M`；反例順序相反）。
- 閘門 `ARMS=2 bash Backup/run_thermal_gate.sh`（不成立 exit 3，fail closed）；2 Hz 序列
  `Backup/thermal_pressure_probe.sh`；in-band 讀數已做進 `Backup/run_req2_retest.sh`。
  見 `docs/PREFILL250_CONDITIONAL_DELIVERY_20260916.html`（＋ `.md`）。

## decode 25 t/s 的現況（2026-09-16 盤點；細節在 `.workbuddy/memory/2026-09-16.md` §U）

- **基線的 2.4× 散佈已被歸因**（2026-09-16，§V）：同一天、同一設定（`p25-gputime`、MTP=0、n=24、r=3）
  量到 6.6 / 9.75 / 10.24 / 16.17 / 6.95 t/s。散熱讀數（`scripts/check/thermal_pressure.py`，
  **唯一實作**，`decode_sweep`／`decode_bench`／`llama_bench_matrix` 都匯入它）接進之後，
  20 個等級讀數**全 NOMINAL** ⇒ **不是散熱**；真來源是**生成長度**（n=24 短爆 vs n=124 可持續）
  與**被計分的暖機輪**（第 1 個請求 wall 23.83 s vs 穩態 1.4 s）。
  **可交付述句**：`p25-gputime`、n=124、全程 NOMINAL ⇒ 可持續 **12.95 t/s**；**不要引用 20.3**。
- 病因已證明是**序列化**（`CGC_SUBMIT_AHEAD=1`：每步 82.5 → 31–44 ms；`gap` 12.3–22.4 ms/步
  是可證的 GPU 閒置；~44% 步時間可移除）。**該探針輸出損壞**（`文摘文摘…`），
  所以它的 16.82 t/s 不可引用；正道是 **D3／S1：expert→slot 查表搬到 GPU**。
- **M1/M2/M3 不能說「保持」**：(a) **S1 不過 M1，但原因不是 ids。** §9.18 的**內核側**讀數
  （`kernel_cgc_ids_capture` ＋ `scripts/check/ids_capture_diff.py`）量到 **39 層 × 117 node × 連續 3 個
  graph 逐位元相同**，第一個 DIFF 是 graph 3 的 `ffn_moe_gate-2`；而 **layer 1 的 gate/up/down ids 相同、
  layer 2 的 router 不同 ⇒ 在「輸入相同 ＋ ids 相同」之下 layer 1 的 MoE 輸出不同 ⇒ 缺陷在 ids
  <u>指向的權重內容</u>（池／slot 內容）**。⚠️ **不要**再引用「缺陷在 layer 18/19」或
  「POST row ≥1 每層 6/6 不同」：前者被 §9.8 的 KEEP_LEAF 排除法取代（§9.2 的層梯**結構上**分不開層別
  與計數），後者 §9.8 已證明是 **padding 且無害**（每層都有 ⇒ 不是層別缺陷的形態）。
  ⇒ **S1 是探針，不是缺陷；要修的是 residency**，而**任何**把查表搬到 GPU 的方案（含 D3）都會踩到同一個
  缺陷（§9.18.7）。**§9.18.6 的唯一決定性測量：capture `ffn_moe_down-1` 的<u>輸出</u>（兩臂、graph 3）。**
  (b) **MTP ON 不過 `plain_match`**，
  而 MTP 是 25 t/s 最可能的乘數。守護不變量只有三條：**ids 相同／canonical gather order
  （按 expert id 排序，2026-09-16 確認**只在 M1 的 compaction 適用、S1 不適用**）／`cap` 是常數**。
- **兩個 decode 儀器不能並排引用（2026-09-16 實測，§W）**：同 env、同模型、同 n、同場交錯，
  llama-bench 讀 **9.4** 而 decode_bench 讀 **12.4**（n≈128，1.32×）／**7.2 vs 18.7**（n=24，2.6×）。
  大半是「**第一個 rep 是冷的**」：llama-bench 的 tg warmup 原始碼上是 `test_gen(ctx, 1, …)`
  ＝**1 個 token**，而 context／池每 instance 建一次；`avg_ts` 把冷 rep 平均進去，
  `decode_bench --warmup 1` 恰好丟掉對應輪。**丟掉後 11.3 vs 12.4 ＝ 1.10×**。
  殘差**未歸因**（候選：llama-bench 每 token 的 `llama_synchronize`、`rand()%n_vocab` 的
  不可預取存取、`n_ctx` 128 vs 4096）。
- **llama-bench 沒有取樣器**：`llama-bench.cpp` 裡 `sampler|speculat|draft|MTP` **零命中**，token 是
  `std::rand() % n_vocab` ⇒ **它自己**量不到投機加速。**但這不等於 MTP 量不到**（更正先前的
  「只能走 HTTP 路徑」）：**`llama-speculative-simple`** 才是本專案的 MTP 驅動器——它用
  `common_params_parse` ⇒ 吃**逐字相同**的 `--spec-type draft-mtp --spec-draft-n-max 3`、
  **內建無投機的基線臂**（省略 `--spec-type`；`--spec-type none` 反而會去載空路徑的 draft model
  並 exit 1）、`check_build_tracked.sh` 已列為關鍵 exe、git 歷史有兩筆本專案的 MTP 修復直接提交給它。
  實測（示範、每臂 n=1）：基線 **7.483** vs draft-mtp **6.517**（同 NOMINAL）⇒ **MTP 現在比 no-spec
  慢 1.15×**。`Backup/run_spec_simple.sh`（env 由 `run_server.sh CGC_DUMP_ENV=1` 取，不手抄）。
- **decode 的散熱分離已有第一個數**：decode_bench 同 env／同模型／同 n，**NOMINAL 12.36 →
  HEAVY 10.64（−14%）**——遠小於 prefill（250 vs <212），且每個等級 n=1 ⇒ **只記錄、不閘門**。
- 到 25 的算術：9–10 × 1.78（D3 上界）≈ **16–18**，**還缺 ~1.5×**，只能來自 M4
  （MTP 拒絕取樣 ＋ verify 真批次；accept 現 **19.9%**，MTP-on 是 **1.8× 淨損失**；M4 依賴 M1/M3）。
- 相關文件：`docs/ROADMAP_PREFILL250_DECODE25_2026-09-13.md`（W1/W2/W3、M0–M6）、
  `docs/REMAP_ROUNDTRIP_REMOVAL_PLAN_2026-09-15.md`（D0–D3、S1 診斷；**其 §1 的「llama-bench」標籤
  無法從該文件稽核**）、`docs/M1_POOL_SPLIT_COST_2026-09-14.md`、
  `docs/INSTRUMENT_COMPARE_20260916_1821.html`（兩儀器對比）。

## 長期事實（踩過就不該再踩）

- **`MTP_SUPPORT` 是編譯期開關且必須定義**（macOS 沒預設，靠 `scripts/build_fork_llama.sh`
  `MTP_SUPPORT:-ON` 傳入）。少了它被編掉的是**修正**（`qwen35moe.cpp:737` 與四處 `[CGC MTP fix]`），
  不只是除錯。現況：`libllama-common.0.0.277.dylib` 已 MTP-on ＋ 四個 `GGML_*` 旗標對齊
  （`D5 --tag flagalign2` 四軸 9/9）。
- **SONAME 內嵌 commit 數**（`libllama-common.0.0.<N>.dylib`）⇒ 全量重建換檔名、舊檔變孤兒。
  載入集要從 `otool -L` 反推再 realpath，**不要寫死檔名**；`strings -a` 要一起印**不受 guard 保護的
  對照字串**，否則分不出「開關沒開」與「artifact 壞了」。
- **錯誤路徑不得「先釋放、後記錄」**（`prefill250` req2 SIGSEGV 的根因：`cmd_bufs_ext` 錯誤路徑
  先釋放、後呼叫 `cgc_metal_record_error()` 去讀 `[cmd_buf error]`）。判準：**指令序本身就是證據**
  ⇒ **要保留當時的二進位**（`Backup/pre_mtp_rebuild_20260916/`）。崩潰報告的 `imageOffset` 可直接對
  二進位反組譯；`usedImages` 的 UUID 是驗單變數最便宜的方法。
- **`-ub` 是 req2 OOM 的旋鈕，而且是單變數**：同 profile、同 `load_mode=none`，`-b/-ub 4096` 全過、
  `6144` 的 **req2 以 status 5 OOM 死**（`kIOGPUCommandBufferCallbackErrorOutOfMemory`），代價 −22%。
  **機制沒修，只有繞道。** 2026-09-16 13:4x 起出廠預設是 **5632**。存活表：`4096 5/5`、`5120 3/3`、
  `5632 4/4`、`6144 0/5`、`6144＋池 6 GiB 3/3`、`mmap 全寬 0/3` ⇒ **邊界是 (pool + compute) 的和**，
  不是 `ub` 一個旋鈕。
- **`--load-mode mmap` 與 expert cache 不相容**：L4 pool 是「regions adopted from expert tensors」，
  mmap 下是唯讀 file-backed 映射 ⇒ `fill_pool_direct` 的 zeroing 寫唯讀頁 ⇒ `SIGBUS` 在 `__bzero`。
  缺陷本體已修（L4 pool 落到 `_Pool` buft，使 `llama-model.cpp:1595` 的 `is_default_buft` 不成立），
  **但 mmap 在任何量過的寬度仍是 0/5**，而且會把 Metal 工作集撐大 ⇒ **不是 OOM 的槓桿，不要再試**。
- **`-ngl` 一律被顯式帶上 ⇒ `-fit` 永遠是 no-op**（`common/fit.cpp:377-379` 在
  `n_gpu_layers != -1` 時 throw）。`CGC_SERVER_FIT=1` 才會真的跑（預設關）。**argv 是位置比對
  `ARG[i]`（D5 config stamp）⇒ 不可重排**，只能用兩段式 append。
- **KV 只有 10 層**（`llama_memory_hybrid`，attn filter `il < n_layer() && !is_recr(il)`，
  `full_attention_interval=4` ⇒ `i=3,7,…,39`）⇒ fp16 @8192 = 160 MiB，**不是瓶頸**。
  MTP 那顆走 plain `llama_kv_cache`（1 層）。
- **`-expert-cache 8192` 是上限不是實配**（實配 ≈ 2.5 GiB）；啟動預算 13030 + 8192 = 21222 vs
  16384 MiB ⇒ **OVERSUBSCRIBED 4838 MiB**。`run_server.sh` 的 `[防護 2d]` 會印
  （`CGC_SERVER_STRICT_BUDGET=1` 超額即 exit 1）。
- **`CGC_METAL_LIB` 不是載入覆蓋**（只做 `_Pool` 的閘門偵測）⇒ metal 庫 A/B 只能**互換檔案**。

## Skill（動手前先讀）

- `~/.workbuddy/skills/cgc-commit-gate/SKILL.md` —— commit 閘門鏈、`BIN_DIR`、`RUN_REPLAY_BENCH=0`、
  索引重生順序、多段式收尾。**本檔的 D6 修訂、快照機制、`eng-mh-0037` 都已寫進它的第七輪實測。**
- `~/.workbuddy/skills/cgc-decode-attribution/SKILL.md` —— decode 相位歸因的儀器清單、判準與陷阱。
- `~/.workbuddy/skills/cgc-prefill-thermal-delivery/SKILL.md` —— prefill t/s 的條件式交付。
