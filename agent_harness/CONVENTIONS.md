# CONVENTIONS.md — 判準憲章

適用範圍：`agent_harness/` 底下**兩個 loop**（`tb_loop` 任務迴圈、`engine_loop` 引擎迴圈）。
這份文件同時是 `engine_loop/sft_pi/` 訓練樣本的 **system prompt 來源**：模型推理時看到的判準，
和它被訓練時看到的判準逐位元組一致（沿用 `tb_loop/README.md` 已驗證的原則）。

每一條都必須能指到**今天的具體證據**（檔名／log 行／欄位值）。指不到的條文不是憲章，是感想。
新增條文時，同一標準適用。

---

## A. 數字什麼時候可以引用

**A1｜吞吐差只以「交錯 ≥3 輪的 paired per-rep median」形式引用，且必須同時附上 build fingerprint 與 md5 集合。**
- 為什麼：單輪 `CGC_SUBMIT_AHEAD` 量到 ×1.78；交錯 ×3 量到 ×1.711（區間 1.327–1.924），
  而且每一輪 md5 都不同。單輪那個數字**根本不可引用**——不是「不夠準」，是「不成立」。
- 證據：`Backup/phase_decomp/ab_submit_ahead.json`（6 列，`p25-gputime` median 9.84 t/s md5 恆 `28097996`；
  `p25-submit-ahead` median 16.75 t/s，md5 每輪不同）。
- 檢查：`scripts/check/ab_interleave.py`（headline 數字就是 paired per-rep ratio 的中位數）；
  `traces/validate.py` 會擋掉 `n_rounds < 3` 的 episode 進入對外表格。
- 反例：`eng-20260915-1900-p25-submit-ahead`（單輪）。

**A2｜沒有 build fingerprint 的數字**不能比**——不是「比較不準」，是「不可比」。**
- 為什麼：重建之後，同一個臂、同一台機器、同一份設定，數字可以不一樣。指紋缺席時，
  「兩個數字不同」有三種解釋（程式改了／環境變了／量測漂移），無法收斂。
- 證據：`scripts/check/ab_interleave.py:42` 的 `build_fingerprint()`（server / metal / llama 三個 md5 前 12 位）。
  `scripts/check/decode_sweep.py` 在 2026-09-15 之前**沒有**記這個欄位，所以那批歷史列一律是 `build: null`。
- 檢查：`episode.schema.json` 的 `build` 欄位；`validate.py` 強制 `build == null ⇒ usable_as_evidence == false`。

**A3｜上界探針（故意輸出錯誤）的數字必須全程標註，不得進入任何對外表格。**
- 為什麼：`p25-submit-ahead` 的 ×1.711 量的是「把段邊界等待整個刪掉」的天花板。它可以拿來
  證明「這個家族最多值多少」，但一旦脫離標註就會被讀成「已經拿到 ×1.711」。
- 證據：`scripts/check/decode_sweep.py` 的 `p25-submit-ahead` 臂註解；`emit_episodes.py` 的 `ARM_OVERRIDE`。
- 檢查：episode 的 `caveats` 必含 `UPPER BOUND ONLY`；`usable_for_throughput == false`。

**A4｜抽樣不等於普查。** rate-limited 的斷言輸出不得用來推「只有某幾層受影響」。
- 為什麼：`CGC-MMID-ASSERT` 只逐字印前 8 筆，之後每 1000 筆印 1 行。當天從 10 行輸出推出
  「只有第 15 層有問題」，而同一份 log 裡 `total=202000` 說明受影響的遠不止那些層。
- 證據：`Backup/cgc_logs/llama_server_20260915_203346.log`（8 行逐字 + 1 行 `total=202000`）。
- 檢查：`validate.py` 的 `_caveat`；`parse_asserts()` 明確區分 `mmid_lines`（樣本）與 `mmid_oob_total`（累計）。

**A5｜生成 digest 只在同一個實際長度下可比。** digest 覆蓋的是**整段** completion，
而長度是 `predicted_n`（模型實際吐出幾個 token），不是 `--n-predict`（預算）——
所以兩個臂即使每個 token 都逐位元相同，只要有一個提早停，digest 就不同。
- 為什麼（第一層）：層梯二分用 `--n-predict 24`，`p25-gputime` 報 md5 `dc055e63`；25 分鐘前兩次
  `--n-predict 40` 都報 `3d8fa55f`。中間**剛好**重建過 `llama` 二進位（`5dc407e0f900` → `bcf32c3f0917`），
  於是「新加的診斷探針改了模型輸出」成了最順的解釋——但 `decode_bench.py` 是 `md5(整段 text)`，
  且把 `predicted_n` 並排存成 `n_tokens_sample`，差異純粹來自截斷。
- 為什麼（第二層，更要命）：同一個 `--n-predict 24` 下，二分的各臂實際長度是
  **24 / 24 / 12 / 22 / 7 / 6**。也就是說「md5 與基準不同」在那些列上**同時**被長度混淆，
  單看 digest **無法**區分「軌跡分歧」與「同樣的前綴、只是提早停」。那些列成立的分歧證據是
  `sample` 前綴（`texts[0][:160]`）：基準是 `<think>\nHere's a thinking process:`，其餘臂從
  第 1 個字元就不同——這件事與長度無關。
- 證據：`scripts/check/decode_bench.py:115`（`n_tokens_sample = predicted_n`）與 `:117`（`md5(text)`）、
  `:119`（`sample = texts[0][:160]`）；`Backup/phase_decomp/s1_bisect2_20260915.json`（六列長度不一）。
- 檢查：比對任何兩個 digest 之前先比 `n_tokens_sample`；不相等就**不可比**，改用 `sample` 前綴
  或（更好的）等 `decode_bench.py` 長出逐輪 digest 與定長前綴 digest。
  這是 A2 用同一條邏輯換一個維度（指紋 vs 實際長度）：**先確認兩個數字說的是同一件事，再問它們是否相等。**

**A6｜配對比值要求兩臂量的是同一個窗。** decode t/s 是「已生成 token 數」上的平均，
所以一個臂提早停，它的 t/s 就是**另一段窗口**的平均（早期 decode 通常較快），配對比值
於是**不再是同一件事的比值**。**先比 `n_tokens_sample`，不相等就不要引用 paired ratio。**
- 為什麼：2026-09-15 的交錯 A/B（`p25-gputime` vs `p25-slotgpu`，交錯 3 輪、
  指紋 `131bc5316ebf` 六列一致）paired per-rep ratio 是 1.403 / 1.243 / **0.834**
  （median 1.243，`all>1: False`），看似可以給一個結論；但兩臂的 `n_tokens_sample` 是
  **100 vs 58**——用的是同一個 `--n-predict 120`。那一臂只跑了不到六成的步數，
  連 miss 數（15760 vs 17954）都因此不可比。
  這一條與 A5 是同一個病的兩個欄位：A5 擋 digest，A6 擋吞吐。
- 證據：`Backup/phase_decomp/ab_s1_20260915.json`（`p25-gputime` 8.67/9.55/8.93，
  md5 恆 `28097996`，n=100；`p25-slotgpu` 12.16/11.87/7.45，md5 集合 2 個，n=58）。
- 檢查：`ab_interleave.py` 已把 `n_tokens_sample` 印在每一列；引用前逐列對齊。
- 附帶讀數（同一批、不受上述混淆影響）：**同 build 的重複性帶**——穩定臂 max/min =
  **1.101（10.1%）**、不穩定臂 **1.632（63.2%）**。所以單輪跨 build 的吞吐差若小於 10%，
  在這個配置上**不可判別**。

**A7｜閘門量必須在任何配置下都被列印，而且只有「算了卻沒被觀測」才是它真正的失效模式。**
- 為什麼：S1 的 `publish_slot_table` 有兩個呼叫點，其中**慢路徑那一個把回傳值直接丟棄**——
  而慢路徑正是每一個 MTP-off 的 S1 臂實際走的那一條：`CGC_SERVER_MTP=0` 讓 `verify_fast`
  與 `draft_fast` 皆為 false，`(verify_fast || draft_fast) && cgc_fast_eligible` 永不成立。
  於是 §8.3 認定為「靜默替換」信號的 clamp 計數，在**所有已量測的 S1 臂上都被算出來然後丟掉**。
- 閘門量清單（現行）：`n_zero_mapped_selected`（選中專家被讀成 0）、`n_fast_cold`（佔比）、
  `n_slot_table_clamped`（GPU 表與 host leaf 失去等價：表把非 resident 專家夾到 **0**，
  那是**合法索引**⇒ 讀到**別的專家**的權重，靜默；host leaf 在同樣情形下寫 **−1**，吵）。
- 檢查：`llama_expert_cache::~llama_expert_cache()` 的 teardown 行——`clamped` 非 0 即標註
  `TABLE/LEAF EQUIVALENCE BROKEN`；`CGC_S1_CLAMP_ABORT=1` 把它從報告升為硬前置條件。
- 反例：`llama-context.cpp` 舊的 pool-path 呼叫點（已修；兩條路徑現在都走
  `cgc_publish_slot_table_counted`，讓兩個量**按構造**一致）。
- **2026-09-15 修訂（量到了，且量到的不是那個量）**：上列清單裡的 `n_slot_table_clamped`
  是**全表**計數，在這個配置下等於「143 slot 對 256 expert 的非 resident 比例」——**每次
  執行都是 113/256**（`clamped_table/publishes = 113/256`，實測），依 B1 **不含資訊**。
  真正該看的兩個量是：
  - `clamped_selected`：clamp 只計**被消費的 id**。實測 **0**（兩個臂都是），而它為 0 的
    原因是**時序**而不是運氣：publish 在 `ensure_batch`/`drain_layer`（`llama-context.cpp:5795`）
    **之後**（`:5819`），填槽後每個被消費的 id 都已有真實 slot，clamp 分支對被消費的 id
    **不可達**——它只會碰到這一步沒人選的專家。⇒ §8.3 選擇「把 clamp 升為硬前置條件」而非
    「強制保留 ZERO slot」在證據上成立：後者會動 `pick_slot` 的算術，也就是動閘門要比的數。
  - `consumed_changed` / `consumed_unchanged_publishes`：被消費映射是否在步與步之間移動。
    實測 **389 / 430（47.5%）** ⇒ 發布**不是**冗餘工作，逐步的 host→GPU 排序要求**真實存在**。
- 檢查：`CGC_S1_TABLE_CHURN=1` 才會啟用 consumed 子集計數；未啟用時 teardown 會明印
  `(consumed-subset churn not instrumented: ...)`——**預設值 0 與量到的 0 必須長得不一樣**，
  否則又是一個「算了卻沒被觀測」的反例。

**A8｜擾動型儀器的成本是「每個 graph 釘幾個張量」，不是「釘住幾個位元組」。**
一個為了讀取而必須改動 allocator/排程的探針（`ggml_set_output`、pin、`ggml_set_output`
這類），它的安全上限**不能用張量尺寸推算**，只能量。
- 為什麼：`CGC_S1_OUT_CAP` 用 `ggml_set_output` 釘住 `ffn_moe_down`。實測邊界是：
  - 所有 graph × 40 層 → 載入期 warmup decode 就 **死**（`kIOGPUCommandBufferCallbackErrorOutOfMemory`
    → `CGC_METAL_FAIL_STOP` abort），2/2；
  - 只有 prefill × **11** 層 → **同樣死**，2/2；
  - 只有 prefill × **4** 層 → 過，2/2；
  - 只有 decode × **40** 層 → 過，2/2。
  而失敗的那個 graph 是 `ntok=2`，此時 `ffn_moe_down` 是 `[2048, 8, 2]` F32，11 層共 **1.4 MB**。
  **1.4 MB 撐不爆任何 Metal 預算** ⇒ 這不是尺寸效應，是「被標為 graph output 的張量個數」
  改變了排程器的配置與切分。
- 檢查：探針要有 **per-graph 釘住數量**的上限，且上限用實驗訂（本機在 4 與 11 之間），
  不要用算式訂。失敗是 fail-stop abort，不是變慢，所以**不能靠逐步加寬去逼近邊界**。
- 邊界：機制本身**未確立**（只知道 4 可、11 不可）。任何「這些張量很小所以釘住很便宜」的
  推理在本機已被上面那組配對反駁兩次。

**A9｜測「存在」的旋鈕，它記錄下來的值只代表意圖。** 讀者寫 `getenv(...) != nullptr` 的
環境變數，`"0"` 是**非空指標** ⇒ **設成 0 等於打開它**。引用任何 knob 的值之前，
先 grep 它的讀者，確認測的是**存在**還是**值**。
- 為什麼：`LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0` 的三個讀者
  （`llama.cpp:396`、`llama-expert-cache.cpp:1708`、`llama-context.cpp:5222`）**全部測存在**，
  而 profile 在 `run_server.sh:939-945` 把它設成 `0`（註解白紙黑字寫著意圖是
  「blk.0 **回到 pool 內**」，因為 skip0 是 4/10 品質殺手），並由 `:1493` 的
  `env "${SERVER_ENV[@]}"` 真正傳進子行程 ⇒ **skip0 一直是開的**，
  blk.0 落在 pool 之外（`llama-model-loader.cpp:1464` 對 layer 0 強制 `l4_kind = -1`），
  那條 **2026-09-09 的品質修復六天來從未生效**。
- 症狀判準：**快照記錄的是意圖，行為卻相反**。所有 `cap_*.json` 與 llama-bench 的 `env` 區塊
  都印 `LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0: "0"`。這與舊 oracle cap
  （`dec-20260915-2233`，記錄意圖）**同一個病**，也是本專案 presence-vs-value 家族的第三例
  （前兩例是 `CGC_OA_ASYNC`，`dec-20260915-2142`／`2215`；後者的值感知修正**靜默改道四個 profile**
  並迫使閘門重新基線）。
- 檢查：用**推導量**而不是旋鈕自己來驗行為。本例唯一吐出真相的觀測是
  `CGC-DECPROF` 的 `layers=39`（同 log 內 19 次，`layers=40` **0 次**）——
  池服務 39 層而非 40。**任何宣稱 profile 行為的句子都不得引用這個 knob 的值。**
- **已修（2026-09-16）**：三個站點改成呼叫**單一 predicate**
  `cgc_l4_skip_layer0_on()`（定義在 `llama-expert-cache.h`，三個 TU 都已 include ⇒
  結構上不可能再各自漂移）。`0`／空字串 = OFF，其餘 = ON。
  `run_server.sh` 的註解改為值語意，並把字面值改成 `${CGC_SERVER_SKIP0:-0}`——
  原本的字面值**無法從外部覆寫**（`env "${SERVER_ENV[@]}"` 會蓋掉傳入值），
  所以控制臂跑不起來。
- ★ **值語意的變更，快照看不出來。** 這是本例與 `CGC_OA_ASYNC` 那次最關鍵的差別：
  OA_ASYNC 動的是「值 → 解析方式」，resolved env **字串變了** ⇒ 閘門自動報
  `INVALID COMPARISON` 叫醒人。本例前後字串**都是 `"0"`**，閘門的可比性檢查是字串比對
  ⇒ 它會報一個**普通的 M1 FAIL，而那是 category error**。
  **凡是「env 相同、意義不同」的變更，一律主動換參照檔，不得沿用舊參照。**
- 兩臂驗證法（可複用）：**先跑控制臂**，讓「舊參照 × 舊行為」自己證明診斷。
  控制臂 `CGC_SERVER_SKIP0=1` 對 v3 得 M1/M2/M3/整列 fnv1a64 **全部 9/9**
  ⇒ 同時證明 (a) v3 是 skip0 開啟時 dump 的，(b) predicate 改動在其餘維度數值中性。
  然後新預設臂 `=0` 對 v3 得 M1 4/9（cross-tab：漂移 5、真分歧 0）⇒ 語意真的變了 ⇒
  用 `--write-ref` 換 v4。**沒有控制臂，`=0` 的 FAIL 就無法與「改壞了」區分。**
- **反面判準（本輪新學）：拿恆定的計畫值當狀態的證據，與 presence-vs-value 同病。**
  `LAYER_CAPS per-layer caps: total 5976 slots` 在**兩臂都印 5976**
  （＝40 層 ×143 ＋ MTP 層 256 的計畫總和），它與 layer 0 是否真的進池無關。
  2026-09-09 甜蜜點 log 的「5976 slots = layer 0 在 pool 內」把計畫值讀成了狀態。
  能區分狀態的是 teardown 的 `owner-set slots`（5793 → 5935 ＝**恰好 +142**，
  正是 layer 0 的 resident slot 數）與 `zero regions`（0 → 4，全在 layer 0）。
- 邊界：presence-gating **不是錯的**——本專案大量診斷就是這樣拼「開」。它的代價是
  **`0` 與 unset 同義** ⇒ 任何「寫 `=0` 期望關掉」的 profile 都寫了一個 no-op。

**A10｜`LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0` 是「對齊 base 佈局」的旋鈕，不是品質旋鈕。**
它的正確值由 base 的 `n_gpu_layers` 決定，不由「品質好不好」決定。
- 規則：**blk.0 的 FFN 在 base 裡住在哪個後端，skip0 就要讓它住在同一個後端。**
  - `-ngl 30`（`run_n30cache.sh`）：base 把 blk.0 留在 CPU ⇒ **skip0=1** 才 bit-identical（§8.36）。
  - full-offload L4（`ALLOW_NGL=1`、40 層全在 Metal）：base 的 blk.0 就在 Metal
    ⇒ **skip0=0**；寫 1 會把 layer 0 變成「Metal graph 讀 CPU 全寬張量」，
    15+27 十連發量到 **4/10**（2026-09-09），而無 skip0 的甜蜜點是 90–100%。
- 為什麼要立這條：這兩個結論**互相矛盾卻都對**，於是六天內文件同時存在
  「skip0=1 是正確性修復」與「skip0=1 是品質殺手」兩種敘述，任何引用者都會踩到一半。
  把它寫成規則後，「skip0 該設多少」必須連帶回答「哪個 base」。
- 檢查：**任何宣稱 skip0 行為的句子都必須標出 base 的 ngl**；否則該句無法證偽。
- 副作用（2026-09-16 量的）：39 → 40 層池化 ⇒ owner-set slots +142、pool +152 MiB、
  file_reads +396、requests +191，**且 layer 0 出現 4 個 zero region（A 臂為 0）**
  ⇒ 待量：這 4 個是「填充沒落地」還是既定的 ZERO slot 機制。**淨速度效果未定，不得假設。**

**A11｜啟動環境要在「你要的那個配置」下 dump 出來看，不能只看預設。**
- 儀器：`CGC_DUMP_ENV=1 CGC_SERVER_PROFILE=<p> [旋鈕=值] bash scripts/run_server.sh`
  印出 `ENV …`（逐項＝交給 `env` 的陣列）、`ARG …`（逐項＝argv）與 `CGCENV …`，
  然後**不啟動任何東西就 exit**（`run_server.sh:1497`）。
- 為什麼不能只看預設：旋鈕之間的互動只在你那個配置下顯形。2026-09-16 就是這樣抓到
  `CGC_SOFT_POOL_L1` 的 `else` 分支裡夾著 phrase-loop guard（少一個 `fi`，`:1381-1392`）——
  設了 L1（正是註解推薦的「opt back in with L0=48 L1=48」寫法）會**靜默丟掉 `CGC_LOOP_GUARD`**，
  連 `CGC_LOOP_GUARD=1` 都要不回來。預設 profile 不設 L1 ⇒ 只看預設永遠看不到。
- 指紋（把 dump 當「啟動環境有無改變」的變更偵測器）：
  ```
  CGC_DUMP_ENV=1 ... bash scripts/run_server.sh 2>/dev/null \
    | grep -E '^(ENV |ARG |CGCENV )' | grep -vE '^CGCENV LOG|\.log$' | md5 -q
  ```
  **必須濾掉 `CGCENV LOG`（含時間戳）與 `free=NN%` 兩類行**，否則同一配置兩跑就不同 md5
  （實測：不濾時 `prefill250` 連跑兩次不同，`prod25` 恰好相同 ⇒ 會誤判為穩定）。
- 檢查：任何改動啟動邏輯的 commit，都要附**預設配置**的正常化指紋前後對照。指紋不變
  ⇒ 生產 profile 的數值不可能移動 ⇒ 不需要重新基線。這是可比性論證，比重跑一次閘門便宜。
- 邊界：指紋只涵蓋**啟動環境**，不涵蓋程式碼路徑。指紋不變仍要跑閘門（本輪：指紋不變，
  且 M1/M2/M3/整列 fnv1a64 對 v4 = 9/9）。
- 反向陷阱：**在 linked worktree 內 dump 的時候，環境變數要寫成兩個獨立賦值**。
  `env "A=1 B=2"` 在 zsh 下不會被 word-split，會變成一個變數名含空格的賦值，
  再經 `SERVER_ENV+=(VAR="$VAR")` 印成看起來像「一個元素含空格」的假象。本輪為此誤判過一次。

**A12｜路徑解析工具回傳的相對路徑，基準是「那個工具的 cwd」，不是你的 cwd。**
- 事證：`git -C <main-worktree> rev-parse --git-path hooks` 從**別的** cwd 跑，回
  `.git/hooks`。把它當成絕對路徑用是錯的，而且錯得靜默。
- 為什麼這是獨立的一條（不是 A7「工具回報成功 ≠ 改動在樹裡」的變體）：
  A7 講的是**寫入沒落盤**；這一條是**讀到了一個語意不完整的值**，
  而值本身「沒錯」（在 git 的 cwd 下它完全正確）。錯的是把它接到別人身上。
- 規則：任何跨目錄使用的工具輸出，一律先錨定成絕對路徑——
  `case "$P" in /*) ;; '') ;; *) P="$ANCHOR/$P" ;; esac`，錨用**你**的基準目錄。
  不要相信「它上次回絕對路徑」。
- 反例（本輪真實形狀）：`--install-hook` 的目的地。修 `--absolute-git-dir`（永遠絕對、
  但不是 git 讀的位置）時，若照字面只換成 `--git-path hooks`，會被解析到
  `<linked-worktree>/.git/hooks`——在 linked worktree 裡 `.git` 是**檔案**，
  於是 `mkdir`/`cat`/`chmod` 全失敗而腳本**照樣印成功**。
  比原缺陷更糟：從「寫到真目錄但沒人讀」退化為「完全沒寫還說裝好了」。
- 檢查：改成相對風險路徑後，用**兩個不同 cwd** 各跑一次安裝/寫入，並對
  「git 實際宣告的路徑」做自我核對（`git rev-parse --git-path …`），
  外加對「另一個可能是雙胞胎的位置」做存在性警告。

**A13｜「判定用」的儀器若沿用被判定對象的視窗寬度，它只能確認前提，不能檢驗前提。**
- 事證：兩個警報（`CGC-MMID-ASSERT … zero_row=`、`pool integrity: … zero-regions=`）都以
  「某個 expert 列的前 4096 bytes 全零」判定「填充掉了、貢獻被靜默丟棄」。裁定工具
  `scripts/check/mmid_zero_row_triage.py` 也用**同樣的 4096 bytes** 讀檔來分類 MODEL-ZERO /
  ENGINE-ZERO。於是它對任何「前 4 KiB 為零」的列**只可能**回 MODEL-ZERO——它在 2026-09-15
  寫下的「14 MODEL-ZERO / 0 ENGINE-ZERO」是結構保證的，不是量出來的。
- 真相（`scripts/check/gguf_dead_expert_census.py`，不需 log，直接讀檔）：gate/up/down 共
  31488 列中有 10 列的前綴為零——**零前綴長 4592–13120 B（恰為整數條量化列），整列
  17.8–47.0% 非零**；其餘 31478 列的零前綴長度**恰為 0**。這 10 列不是 dead expert，
  是 zero-PREFIXED expert。所以池中那個區域為零**是檔案要求**，不是缺陷。
- 規則：任何「此區域為零 ⇒ 有缺陷」的儀器，其確認步驟必須用**與警報不同的寬度**：
  先窄探針（便宜），全零時再讀**整條 stride** 才計數。否則警報與裁定共享同一個盲點，
  而共享盲點的兩個儀器永遠互相印證。
- 同一條規則的第二個後果：`CGC_MMID_ASSERT_FATAL=1` 原本對這 10 列 abort（舊註解自己
  承認「on this model that means every run」）⇒ 在健康配置上不可用。修好分類後，
  fatal 只認**已用整列確認**的零列。
- 檢查：改動 probe 後要同時附 (1) 檔案側普查（零前綴 vs 整列零）、(2) 裁定工具在
  **新舊日誌**上的判決（`exit 3` = 修探針，不是修填充）、(3) 啟動環境指紋（A11）與
  M1/M2/M3 閘門；本輪三者：指紋不變 `e68a5dc5…`、閘門 M1/M2/M3/fnv1a64 = 9/9/9/9。

**A14｜記憶體壓力計數器不是「引擎依賴的那個資源」的量度；而且一個變數若與執行順序同步移動，
相關係數再高也不能分離「因果」與「累積」。**
- 事證：`prefill250` 的 250 t/s 已被達到過（286.67、254.74、264.78、峰值 276.59），也被同一個
  命令錯過過（122.68）。候選機制是啟動時記憶體水位。儀器
  `scripts/check/prefill_certifiability.py` 第一版記錄 `free + inactive`，5 個獨立行程得到
  286.67 / 173.54 / 155.62 / 166.06 / 166.63，**rho = −0.70**——記憶體最少的**那一次**是唯一達標的。
- 反相關暴露儀器錯誤：`Pages free` 低可以是「page cache 大」（macOS 把閒置 RAM 填成 cache
  ⇒ 應該快），也可以是「別的行程佔住」（⇒ 應該慢）。**兩個方向相反，而我把它們加總平均掉了。**
  當場實測：free 5.94 GiB，但真正的 cache（`File-backed pages`）只有 3.33 GiB，另有 anon 3.82 GiB。
- 修法與其極限：改記錄 `File-backed pages`，並加 `--warm-runs`——在指定次數之前**預先讀完整個
  13.66 GB 模型檔**，強制 cache 上升而 free 下降。**這是打破「執行順序 ≡ 變數順序」的唯一方法**；
  自然狀態下 cache 與 free 一起動，純加樣本永遠分不開。
- 結果（6 次，第 2/4/6 次強制暖機）：cache 4.51 / 9.61 / 2.98 / 9.89 / 2.32 / 9.96 GiB（4.3×）、
  free 3.74 / 0.07 / 7.81 / 0.06 / 7.26 / 0.06 GiB、吞吐 179.09 / 184.09 / 184.06 / 173.76 /
  155.33 / 175.23。**rho(cache, t/s) = +0.09；0/6 越過 250。把候選變數拉高 4.3 倍，吞吐沒動。**
- 最乾淨的判準不是相關係數，是一對同狀態樣本：第 1 輪 run 1（free 3.90 GiB）→ 286.67；
  第 2 輪 run 1（free 3.74 GiB）→ 179.09。**幾乎相同的啟動狀態，60% 的差距。**
  這同時說明 286.67 這個唯一達標樣本**不能由它旁邊記錄到的狀態重現**。
- 規則：宣稱「吞吐取決於機器記憶體狀態」之前，要先指出那是**哪一個**資源，並**強制**它到兩個
  極端；否則就說「未定位」。**不要用 pool 大小或記憶體水位去追一個數字**——本輪已否證那條路。

**A15｜一次達標要先問它是不是極端值。輸出是「可重複的帶」，不是「範圍」，也不是「目標」。**
- 事證：11 次獨立啟動，只有 1 次越過 250（286.67，1/11）。範圍寫成 `155–287` 會暗示一個寬而居中
  的分布；誠實的讀法是**十次中有十次落在 155.33–184.09（18.5%）**⇒ 引擎的操作點是穩定的
  ~170 t/s 帶，250 是極端值，而不是反過來。差距因此可以定價為 **1.36×**。
- 規則：`prefill_certifiability.py` 的輸出必須同時報 (1) 越過目標的次數（k/N）、(2) 排除單一極端值
  後的可重複帶與其離散、(3) 組內離散（同行程，證明帶間離散不是量測噪聲）。三者缺一，數字就會
  被讀成它不支持的結論。
- 邊界：這條不否定「暫態可以觀測」——286.67 是真實量到的。它否定的是**把暫態當操作點**。
- `superseded_by: A16`（部分）——2026-09-16 追加：這條把極端值歸為「離群」，但沒有問**它是不是
  這個 session 的第一次啟動**。「帶」是真的，但成因不是抽獎而是熱暫態，見 A16。

**A16｜離群值先問「它是不是每次 session 的第一次」。目標要先問「熱態還是冷態」。**
- 事證：四個「有間隔後第一次啟動」的 session = 286.57 / 293.11 / 299.35 / 286.67 t/s，
  而同一批次後續啟動 = 155–184。間隔是**導出**的（前一 session 最後一個 run 的完成時刻 →
  本 session run 1 的啟動時刻，來源為日誌的 UTC 時戳與 `summary.json` 的 mtime）：
  **923 s → 299.35、848 s → 293.11、247 s → 286.57、67 s → 179.09**（單調）。
  也就是說判別量不是「第幾次」，而是**距上次 GPU 負載結束多久**；恢復常數被夾在
  **67 s < τ ≤ 247 s**。★ 2026-09-16 更正：本條初版寫的「5 秒」是**未量測的敘述**
  （當時 `--idle-before` 尚不存在，日誌與 json 也無時間欄位），實際為 67 s，見 lesson `eng-mh-0020`。
  ★★ 2026-09-16 再更正（**直接量測，取代上面的推導區間**）：
  `scripts/check/prefill_idle_sweep.py` 掃 7 個 idle × 2 次啟動（加 preheat 共 15 次）得
  **120 s < τ ≤ 150 s**——≤ 120 s 一律 122–134 t/s（逐層中位數 368–403 ms），
  150 s 跳到 **294.05**、180 s 是 300.09、240 s 是 307.84 t/s（中位數 161–167 ms）。
  與推導端一致（67 < 120、247 > 150）並收緊一個數量級。**轉換是陡的**：兩個相鄰取樣點之間沒有中間值。
  副產品：**熱態沒有下限**——連續熱跑（間隔 0–120 s）低到 **122.39 t/s**，低於本條寫的 155–184 帶，
  因為升溫是在**連續數次 prefill 之間累積**的（run 2 隨前一個 idle 變長而變冷：172.55 / 222.97 /
  273.86 t/s @ 150/180/240）。所以「持續負載規格」不是一個帶，而是一段會繼續下探的值域。
  見 decision `dec-20260916-0500`。
- 機制證據（`CGC_GPU_TIMING=1` 的 `CGC-GPUTIME`，Metal 的 `GPUStartTime/GPUEndTime`）：
  三張 run 的 prefill 大圖都是 `segs=40 bufs=352/353/360`、`gpu_union/wait = 100–101%`、
  `gap = 109–416 ms`（1.5–4%）；**GPU 忙碌時間比 1.44× 對上吞吐比 1.42×**，而 CPU 編碼時間
  反而下降（0.74×）。同一工作量、同一分段、同一位元組計數（`slab_pool/disk`、`resident`、
  `owner_slots`、`nonresident=44.5%` 逐位元相同）⇒ 差別是**GPU 執行速率**，不是任何資料搬移。
- ★ **第二個獨立儀器同向**（2026-09-16 追加）：`ioreg -r -c IOAccelerator` 的 Device Utilization %
  （1 Hz、300 筆，`Backup/cgc_logs/gpu_util_prefill_paired_20260916.log`）在 S4 三個 run 的 prefill
  期間平均 91.1 / 97.3 / 94.2%、**三個都觸到 99%**、區塊內最低 66–72%；只有 run 之間約 10 秒的
  重載模型期掉到 11–18%。與 Metal 時戳同向，且來源完全不同套計數器 ⇒
  「慢的那次不是因為 GPU 在發呆」有兩份互不相干的證據。
- 硬體前提：`Mac16,12` = MacBook Air M4（8 核 GPU、16 GB、**無風扇**）。無風扇機體在持續負載下
  降頻是可預期的，因此這不是缺陷而是**機器的規格**。
- 規則：任何 throughput 目標的判定都要同時報 (1) 本 session 第幾次啟動、(2) 距上次啟動結束的
  **間隔秒數（導出值或儀器值，不是估計值）**、(3) `gpu_union/wait`。只看 (1) 會把熱暫態誤讀成抽獎。
- 邊界：`gpu_union/wait` 高只證明「等待期間 GPU 在忙」，**不**證明原因是頻率（也可能是功耗上限或
  記憶體子系統）。利用率同樣是**佔用比例、不是頻率**。要指認頻率需要 root 的 `powermetrics`
  （沙箱內 `sudo` 被封鎖，需人工跑；本輪已試且失敗——`powermetrics_prefill_paired_20260916.log`
  只有一行 `operation not permitted: sudo`，見 lesson `eng-mh-0021`）。
- 反例（不可用 `gpu_union/wait` 論斷的情形）：`skipped` 非 0 時 Metal 沒回報時間戳，該行無效。
  本輪 `skipped=16–46`／`bufs≈352`（4–13%），故只以 `union ≈ wait` 這個**大尺度**關係立論。

**A17｜宣告一個儀器「不存在」或「可以量某階段」之前，先查兩件事：日誌裡有沒有它的輸出字面，以及它有沒有第二個閘門。**
- 為什麼：`CGC-DECPROF` 被前一輪的報告與 lesson `eng-src-0009` 斷言「這個字面在原始碼樹裡不存在」。
  它存在（`ggml-backend.cpp:2064/2088/2100`），而保留的日誌裡有 **95 份、數萬行**它的輸出
  （單一最大份 `llama_server_20260915_185704.log` 有 2736 行）。假陰性的成因是 grep 只掃了
  `src/llama.cpp/src/`，漏掉 `ggml/src/`——這是本專案**反覆發生**的窄範圍 grep 陷阱。
- 第二件事（更容易漏）：**一個儀器活著，不等於它看得見你要量的事件。** `CGC-DECPROF` 有三個閘門：
  (1) `CGC_DECODE_PROFILE` 必須設；(2) 分段路徑必須開（`cgc_oa_async_enabled()`，`CGC_OA_ASYNC=0` 會關掉它）；
  (3) 列印閘門 `(dp_step % 8) == 0` 且逐層累加器每步重置 ⇒ prefill（`ubatch=6144` 下只有一個
  `graph_compute`，`dp_step=1`）**永遠不列印**。實證指紋：95 份日誌**每一份的最小 step 都是 8**，
  `step<8` 出現 0 次。
- 規則：(a) 要說某個字面不存在，先 `grep -rl` **全樹**並把「日誌裡有沒有它」當第一判準；
  (b) 要說某個儀器能回答某問題，必須逐個列出它的**全部**閘門，並在日誌裡確認該事件真的產生過輸出；
  (c) 閘門沒開時，儀器是**靜默**的——「沒有輸出」與「事件沒發生」在日誌上長得一樣。
- 正面用法：閘門 3 只需一行即可解鎖 prefill 歸因。★ 2026-09-16：**已實作並跑完**。實際落地的閘門是
  `if (dp_tot > 0 && ((dp_step % 8) == 0 || dp_step == 1 || dp_ntok > 1))`（`ggml-backend.cpp:2080`），
  比原案多了 `dp_ntok > 1`，並讓每行輸出帶 `ntok=`。多這一項的理由是本條自己的教訓：
  `dp_step == 1` 仍然**假設**「第一個圖就是 prefill」，而 warmup 圖可以悄悄佔走那個位置，
  於是剖面描述的是 decode 卻自稱 prefill。`dp_ntok` 讀 top-k 張量的 `ne[1]`（`ne = [n_expert_used, n_tokens]`），
  所以任何 batched 圖都會放行，且每行**自證形狀**。改動全部落在 `if (dp_on)` 區塊內 ⇒
  未設 `CGC_DECODE_PROFILE` 時整段不進入，生產配置的數值路徑不變。

**A18｜把兩份日誌當「同一個量」配對之前，先確認兩邊的粒度相同；逐層/逐段剖面要跨圖平均；統計量要連讀法一起寫。**
- 為什麼：這一輪同一個問題（「是某一層變慢，還是全層等比？」）在同一天得到三個都算「層間 CV」的答案——
  **6.5% / 8.7% / 30.1%**，而差別全部來自讀法，不是來自機器；其中 30.1% 那個還會把結論翻成相反的。
  (a) **粒度**：快的那份日誌是在 `CGC_DECODE_PROFILE_ALL` 生效前抓的，每圖只印 `top1..top8`；
  慢的那份印 `all1..all40`。於是「共同層」退化成**快跑自己成本最高的 8 層**（全在 L28–L38）——
  不是隨機抽樣。8 層子集給 `2.32× / CV 6.5%`，換成全覆蓋 40 層給 `1.64× / CV 30.1%`。
  (b) **單圖**：`llama-bench -r 3` 讓每個行程留下多個圖，**第一個圖帶一次性的載入／首次觸碰成本**，
  那個暫態落在「當時正在跑的那幾層」身上，看起來就像局部熱點。同一對日誌只讀第一個圖是
  `1.64× / CV 30.1%`（判 NOT UNIFORM），跨 4 圖平均是 `1.60× / CV 8.7%`（判 UNIFORM）。
  最便宜的檢驗：印出每個圖的 `worst layer`——它在圖之間漂移就是暫態。
- 規則：(a) 配對前先**印出兩邊的層／段覆蓋**再算比值；(b) 逐層剖面一律**跨圖平均**，並印出每圖形狀；
  (c) 一個統計量的值若取決於讀法，結論必須**把讀法寫在數字旁邊**（同一個量可以在 6.5%/8.7%/30.1% 之間移動）；
  (d) 判「均勻還是局部」用**中位數 + IQR/中位數 + 落在 ±25% 內的比例**，不要用 `peak/mean`——
  管線化迴圈（`submit_ahead`）之下最前面幾層天生等得少，`peak/mean` 會把均勻的圖判成局部
  （熱跑 `peak/mean=1.67–1.74` 判 LOCALISED，`IQR/中位數` 只有 0.04–0.07）。見 lessons `eng-mh-0024/0026/0027/0028`。
- 落地：`prefill_gputime_report.py --decprof-pair --graph first|last|mean`（**預設 `mean`**），
  配對前自動印兩邊的層覆蓋與每圖 `worst layer`，並量化局部超額（超過中位數 1.5 倍的層，及其佔缺口的比例）。

**A19｜長時間執行的掃描／彙總工具會一直用它「啟動時載入」的那份程式碼；收尾要用當下的程式碼對保存的原始日誌統一重取。**
- 為什麼：`prefill_idle_sweep.py` 對每個 session 是**開子行程**呼叫 `prefill_certifiability.py`
  （所以每個 session 的 `summary.json` 用的是當時的新程式碼），但它自己的彙總表在啟動時就載入了。
  本輪 sweep 跑到一半時 harvester 被修正，於是**前 2 個 session 報 48 層、後 5 個報 40 層**，
  而最後那張彙總表仍印舊欄位（`peak/mean`）——兩種版本同時存在於同一份 artifact，且都不報錯。
- 規則：(a) 掃描開始時先把**要用的程式碼的 mtime／大小印進日誌**（`provenance()` 已這麼做），
  這樣「中途改過」在日誌裡看得見；(b) 掃描結束後**一律**用當下程式碼對保存的日誌重取一次，
  以重取的結果為準（`--reharvest`）；(c) 原始日誌必須保存——能重取的唯一前提是它還在。
  見 lesson `eng-mh-0029`。

**A20｜門檻若是「整條序列的百分位」，它就會隨序列的<em>工作週期</em>漂移；用「把尾巴接長」檢驗它，不要只看手上這一份。**
- 為什麼：2026-09-16 的 GPU 功率分段門檻 `0.15 × p95`。p95 是**整份擷取**的百分位：
  壓測結束後 `powermetrics` 變成孤兒又寫了 32 分鐘，p95 就滑進閒置分佈、門檻塌到 200 mW 的地板，
  於是 240–300 mW 的**閒置毛刺被當成啟動**（本機閒置 p50 51 / p95 217 / max **398** mW）。
  同一份擷取報出 **25–31 個活動區塊**而不是 3 個，「冷」被配到 09:09:30 一筆 237 mW 的毛刺
  ——整組判準失效，而輸出長得完全正常。
- 規則：(a) 門檻要錨在**活躍模態**上，不是錨在「全序列的某個百分位」上；工作週期可能低到 0.5%，
  百分位要選到在那個週期下仍然落在活躍區（本例 `0.15 × p99.5`）。
  (b) 檢驗方式是**不變性**：把尾巴接長（0 / 10 / 32 / 90 分鐘）再看推導出來的東西有沒有變
  （舊規則 3/6/22/31 塊，新規則 3/3/3/3 塊，且邊界 ±1 樣本）。只看手上這一份不算檢驗。
  (c) **門檻與工作週期一律印出**——塌掉的門檻只有印出來才看得見。
- 附帶：回歸測試的合成資料要模仿真實的**形狀**，不是模仿它的平均。第一版接了一條「平的」尾巴，
  而平尾巴與最後一次啟動**相鄰**、會併進同一區塊 ⇒ 舊規則也得 2 塊，測試因為錯誤的理由通過。
  真實閒置是尖刺狀的，建成「4 樣本爆叢 + 6 樣本真閒置」的交錯之後，舊規則才露出 401 塊 vs 新規則 2 塊。
- 見 lesson `eng-gate-0024`；`scripts/check/powermetrics_gpu_freq_parse.py` 的 `--selftest` 已含此回歸項。

---

## B. 診斷

**B1｜結構性恆零的欄位不是證據。**
- 為什麼：`host=0` 被讀成「表在 private buffer」。實際上 `ggml_backend_metal_buffer_type_shared_is_host()`
  與 `..._private_is_host()` **都** `return false`——Metal 一律印 0。這個欄位無法區分任何兩種情況。
- 證據：`src/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m` 的兩個 `is_host`；
  `ggml_backend_buffer_is_host()` 讀的是 buft 的 `is_host`（`ggml-backend.cpp:174`）。
- 症狀判準：一個欄位如果在**所有**觀測下都是同一個值，「它沒變」就不含資訊。

**B2｜診斷本身要先被證明會印。**
- 為什麼：`GGML_SCHED_DEBUG=1` 的 `## SPLIT` 用 `GGML_LOG_DEBUG` 輸出，被預設 verbosity（INFO）濾掉，
  log 裡 0 行。**「0 行輸出」與「只有 1 個 split」在 log 上不可區分**，那一輪探針白跑。
- 證據：`Backup/cgc_logs/llama_server_20260915_2026*.log`（改為 WARN 級之前為 0 行）。
- 實務：任何新診斷，先用一個**已知會產生輸出**的配置跑一次，確認它真的會印。

**B3｜同一個 graph 混用多個 backend 時，「誰算的、誰讀的」是契約；先查後端指派，再查內容。**
- 為什麼：S1 的 layer 0 崩在 `libggml-cpu` 的 `mul_mat_id`——當天唯一一次 CPU 崩潰。
  `GGML_SCHED_DEBUG=2` 的 per-node 傾印一行就講完：
  ```
  baseline: node #73 (MUL_MAT_ID) ffn_moe_gate-0 [CPU]   src[2] = ffn_moe_topk_remap-0 [CPU]
  S1      : node #77 (MUL_MAT_ID) ffn_moe_gate-0 [CPU]   src[2] = CPU#ffn_moe_slots-0# [NULL]
  layer>=1: node #180                ffn_moe_gate-1 [MTL0] src[2] = ffn_moe_slots-1 [MTL0]   ← 兩臂一致
  ```
  layer 0 的 MoE FFN 在 CPU/BLAS（專家權重仍是全尺寸 82M vs 其餘 45M），host leaf 是 CPU graph input
  所以天然可見；改 GPU 查表就得跨 backend 拷貝。
- 證據：`Backup/cgc_logs/llama_server_20260915_202629.log`（base）/ `..._202725.log`（S1）。
- 先後順序很重要：**在懷疑數值之前，先懷疑這個值是不是同一個 backend 上的同一個 buffer。**

**B4｜重寫一條資料流時，逐項清點原實作依賴的「隱含契約」。**
- 為什麼：S1 用 `get_rows` 取代 host 寫的 leaf，漏抄了 `ggml_set_output(remap)` 這一行。
  少了它，ggml-alloc 把 11 層裡 10 層的 gather 輸出疊到同一個位址（`ids_data=0x128179d60` ×10），
  每層讀同一份 16 個 int。兩個致命 bug 之外，前面的猜測（表內容？host 可見性？）全部無效。
- 證據：`Backup/cgc_logs/llama_server_20260915_203053.log`；修完 `ids_data` 逐層不同、斷言 41 → 9 行。
- 通則：原實作有一行「看起來只是保險」的呼叫，通常就是在補契約。抄之前先問它在防什麼。

**B5｜編碼期（dispatch build 期）的 CPU 讀取，對 GPU 算出來的張量**一律**是競態。**
- 為什麼：`CGC-MMID-ASSERT` 在 Metal **encode 期**讀 `op->src[2]->data`。baseline 的 ids 是 host leaf
  （CPU 寫的），沒問題；S1 的 ids 是 GPU 算的，於是讀到的是配置器留在那塊記憶體裡的殘值——
  F32 router 機率（`0x3F64E6C4≈0.893`）、`0x7FC00000`（+NaN）、以及同一個 node 內合法與非法索引混雜。
- 決定性反證：`ggml_backend_sched_synchronize()` **之後**回讀同一塊 buffer（`CGC-S1: POST`），
  layer 1 是 `[2 105 7 9 5 106 1 84 0 103 0 74 121 110 3 99]`，與 hook 發表的表查表結果
  `ids=[193->2 105->105 229->7 …]` 逐項相同。
  **⇒ 那些 `id_oob` 全部是探針假警報。**
- 證據：`Backup/cgc_logs/llama_server_20260915_204938.log`（EXPECT-pool 行 59 / POST 行 187）。
- 通則：要在 CPU 上驗證 GPU 的產物，只有兩個合法時機——**同步之後**，或**圖外**。

**B6｜把斷言先分類（MODEL-ZERO vs ENGINE-ZERO）再修。**
- 為什麼：`mmid_zero_row_triage.py` 把 14 筆斷言全歸為 MODEL-ZERO（模型本身把專家路由到零權重），
  ENGINE-ZERO 是 0。先分類才不會把模型的性質當成引擎的缺陷去修。
- 證據：`scripts/check/mmid_zero_row_triage.py`；`docs/MMID_GEOMETRY_PROBE_2026-09-15.md`。

**B7｜閘門必須先證明「它會擋錯」。**
- 為什麼：bit-identical 參考檔本身是 cache-ON 產物，自我比較只能證明非決定性與 pool-size 不變性，
  不能證明 cache 的數值等價於無 cache。差點誤信 19/19。
- 證據：`scripts/check/oracle_truth_gate_selftest.py`、`feasibility_gate_selftest.py`；
  `m123_oracle_gate.py --oracle-abs` 會拒絕 cache-ON 的 `.cap`。
- 實務：每個閘門都要有一個「餵它錯的東西，它必須紅」的自測。

**B8｜儀器說「什麼都沒有」時，必須同時交出它看到的原始輸入。**
- 為什麼：`powermetrics_gpu_freq_parse.py` 的無樣本分支原本只印**自己期待的那些標籤**
  （`GPU-looking lines actually present`），於是 2026-09-16 那份 40 bytes 檔——唯一內容是
  `(eval):1: operation not permitted: sudo`——被回報成 `looks empty or was truncated`。
  **整份診斷就在那個檔裡**（它為了找 GPU 行已經把檔開過了），工具只是選擇不印。
  結果「capture 從來沒被執行」被讀成「擷取是空的」，而這兩件事的處置完全相反。
  這與 B7 是同一類缺陷，只是作用在儀器的**輸入**而不是輸出：它分得出「標籤變了」與
  「沒有標籤」，分不出「這個檔根本不是擷取」。
- 規則：(a) 空結果要印**原始輸入的前幾行**，不能只印「符合我預期的那些行」；
  (b) 多個輸入時逐一給 verdict 行，**不可靜默丟掉參數**——glob 是自然用法，而失敗產物
  會混在裡面（本輪的 `.sh` 原本只吃 `$2`，其餘靜默丟棄）；
  (c) 兩者都要讓「跑失敗」與「量到零」在輸出上不同形。
- 驗收：selftest 要有「死檔在前＋好檔在後」與「只有死檔」兩個案例，後者斷言輸出裡
  必須出現那個檔的錯誤行。見 lesson `eng-gate-0018`、`eng-gate-0017`。
- 附帶（同一輪踩到）：改 CLI 契約（`nargs`）時要同時跑一次**不帶位置參數**的入口
  （`--selftest`）——那是契約的另一半。`nargs="+"` 讓 `--selftest` 直接被 argparse 拒絕，
  是 selftest 自己抓到的。

**B9｜在同一個 dylib 兩側以指標傳遞的 struct，新增成員是一次連結器看不見的 ABI 破壞。**
- 為什麼：2026-09-16 為斷言 `ensure_batch` 的兩趟指派，在 `llama_expert_cache` 加了
  `n_hit_adopted_queued`（8 bytes），把 `ever_loaded` 的偏移從 `0x608` 推到 `0x610`。
  A/B 用 `git stash` 換臂建的舊 dylib 留在 `build/bin`，`stash pop` 之後**沒重建**，
  於是「照新標頭編的測試」連上「照舊佈局編的庫」。症狀是
  `SIGSEGV / KERN_INVALID_ADDRESS at 0x0` **崩在函式庫裡**
  （`llama_expert_cache_ensure_batch +1068`），**不是連結失敗**——連結器只看見符號，
  看不見成員偏移。反組譯那條指令（`ldr x9, [x10, x9]`，x10 來自 `[x8, #0x608]`）
  對照 `offsetof` 印出的兩套偏移即可確認：`0x608` 在新佈局是 `n_slot_table_unchanged`
  （`size_t`，值 0，被當成 data pointer 索引 ⇒ 空指標），在舊佈局才是 `ever_loaded._M_start`。
- 規則：(a) 動到跨 dylib 邊界的 struct ⇒ 與該標頭連結的測試／工具要與**它所連的樹同狀態重建**；
  (b) `git stash` 換臂之後 build tree 屬於**另一臂**，pop 後第一件事是重建、不是跑測試；
  (c) 診斷順序 `nm -gU`（符號在不在）→ `offsetof` 探針（兩套偏移）→
  `strings` 找新增字串（本例該 dylib 完全不含 `CGC-BATCH-INVARIANT` ⇒ 它是舊碼）→
  `stat` 兩個 mtime。四步一分鐘內收斂。
- 附帶：錯誤的直覺是去改測試（第一反應是回頭審測試的 `key_segs`/`slot_owner` 初始化）。
  判準是「庫和標頭是不是同一棵樹」。見 lesson `eng-gate-0019`；
  `scripts/check/expert_cache_ensure_batch_order.cpp` 檔頭已寫入 ABI 警告與編譯指令。

**B10｜把一次 A/B 當成驗證之前，先確認修法的前提在該配置下<em>可達</em>。**
- 為什麼：`ensure_batch` 的兩趟指派只在「池填滿、且 LRU 淘汰真的觸發」時才改變行為。
  出貨預設（非 `CGC_POOL_SPLIT`）的池在 warmup 期間**從不填滿** ⇒ miss 落在空槽
  ⇒ 淘汰不觸發 ⇒ 兩趟是 no-op。2026-09-16 把修法 stash 掉、同一 4 GiB 配置重跑，
  新舊二進位吐出的 `CGC-PRE`/`CGC-POST`/`CGC-SLOT` **逐位元相同**。
- 規則：逐位元相同的 A/B 要讀成兩句話——「**沒壞的地方沒有變**」（有價值的安全確認）
  與「**修法被檢驗了**」（沒有）。當前提取不到鑑別性訊號時，去找**能把前提做到的最小實驗**
  （本例：不需模型、不需 IO 的單元測試，直接連 dylib 裡真正的函式），
  不要把 no-op 的 A/B 寫成通過。
- 見 lesson `eng-gate-0022`；`scripts/check/expert_cache_ensure_batch_order.cpp`
  （HEAD 逐字重現 09-14 日誌的映射 ⇒ FAIL；兩趟之後 ⇒ PASS）。

**B11｜用特權啟動的子行程不能用 `kill $!` 停；而且非互動 shell 的背景工作會忽略 `SIGINT`。**
- 為什麼：2026-09-16 的 `powermetrics` 擷取——壓測 09:14 就結束，它卻寫到 09:46 之後（多出約 4000 筆
  純閒置樣本），而腳本最後那行「跑完會自動 parse」**從未執行**（卡在 `wait`）。原碼
  `sudo powermetrics … & PM=$!; … kill -INT "$PM" 2>/dev/null` 有**兩個獨立**原因，各自都足以讓
  `kill` 回 0 卻什麼都沒發生：
  (a) `$!` 是 **root 的 sudo** pid，非特權 `kill` 得 `EPERM`，而錯被 `2>/dev/null` 吞掉；
  (b) POSIX 讓非互動 shell 的*背景*工作把 `SIGINT/SIGQUIT` 設為 `SIG_IGN`——實測
  `sh -c 'sleep 30 & P=$!; sleep 0.2; kill -INT $P; sleep 0.5; kill -0 $P && echo alive'` 印 `alive`。
- 規則：(a) 停止要**委託給同層特權的守護行程**（哨兵檔通訊，父行程不需特權），訊號改 `TERM`；
  (b) 再加一個**自我終止的數量上限**（`powermetrics -n`）與**行緩衝**（`-b 1`，硬停不掉最後一條記錄）
  ——三層，因為進度不該取決於單一機制；(c) 停止後**驗證「檔案是否凍結」，不是「pid 還活著沒」**：
  後者與訊號投遞競態，會在*正常*停止之後誤報，而會誤報的警告等於沒有警告。
- 附帶：只修 (a) 會**看起來像修好了卻仍然失敗**；(b) 是用 stub `sudo`/`powermetrics` 實跑腳本時
  才暴露的。拿不到 root 也要把**不需要 root 的那條路徑**實跑一遍。
- 補記（同日稍晚）：**那次留下的孤兒行程其實收得掉，而本輪一開始把它寫成了「收不掉」。**
  鏈上有兩個 pid 且**分屬不同擁有者**：
  `84847 sudo powermetrics -i 500 …`（包裝層，自己的）與 `84849 powermetrics -i 500 …`（本體，root 的）。
  `kill -TERM 84849` → `operation not permitted`（**這才是殺不動的那一個**）；
  `kill -TERM 84847` → 回 0，兩個一起消失，擷取檔凍結在 5 410 344 bytes。
  規則 (d)：**「殺不掉」是一個關於<em>某個 pid</em> 的述句，不是關於<em>整條行程鏈</em>的述句**——
  先 `pgrep -fl` 把整條鏈讀出來；`sudo` 包裝層永遠是自己的，就算它的子行程不是。
  這與 (c) 是同一個錯誤家族（在被污染的證據上作判斷），也正是 B7／B12 在說的事：
  把錯誤丟進 `/dev/null` 之後，「我試過了」與「我沒試」同形。
- 見 lesson `eng-gate-0023`（與其補記 `eng-gate-0027`）；`scripts/check/powermetrics_gpu_freq.sh`。

**B12｜儀器要<em>在自己的資料上</em>跑過一次才算驗證過；而這包含「它讀了你付費取得的每一個欄位」。**
- 為什麼：`powermetrics_gpu_freq_parse.py` 是為了取代一個「丟掉決定性欄位」的舊 `--parse` 而寫的，
  已對三個獨立來源的真實擷取核對過標籤、`--selftest` 4/4 全綠。第一次對本機自己的擷取跑，仍暴露
  兩件事：(a) 它**從未讀取 `Current pressure level`**，而那份擷取正是用
  `--samplers gpu_power,thermal` 取的，理由就是「時脈限制 **vs** 功耗上限」分不開
  ——**問題的另一半，寫它的那個解析器答不了**；(b) 分段門檻在真實的閒置分佈上塌掉（見 A20）。
- 規則：(a) 驗收清單加一條——**擷取命令用到的每個 sampler，解析器都有讀，而且讀到的值有印出來**；
  (b) 「已對外部樣本驗證」與「這個儀器能回答那個問題」是兩句話，不要當成一句寫進結論；
  (c) 自測量的是「合成樣本能走完規則」，不量「規則覆蓋了資料的所有欄位」。
- 見 lesson `eng-gate-0025`。

**B13｜「讓它可見」的計數器要有人<em>讀</em>；閘門的輸出要能與它的<em>不存在</em>分辨；而閘門不可以被放在排除掉它要守的那個 case 的分支裡。**
- 為什麼：2026-09-16 的 Blocker B 修正加了三個東西，**三個都各自以不同方式無聲**：
  (a) `n_hit_adopted_queued` 的註解寫「makes that path visible instead of silent」，
      而全樹**只有兩處**：`.h` 的宣告與 `.cpp` 的 `++`。**它是唯寫的**——
      收養路徑在出貨配置下若真的觸發，沒有任何輸出看得見。
  (b) 不變量閘門（`LLAMA_EXPERT_CACHE_BATCH_INVARIANT`）的 OK 行**只印 `il<=2`**，
      於是「log 裡沒有 VIOLATIONS」與「閘門根本沒跑」在證據上同形
      （`SERVER_ENV` allow-list 沒帶進去就是後面那一種）。
  (c) 同一段閘門**巢狀在 `if (!miss_exps.empty())` 之內**——所以**零 miss 的批次從來不被檢查**，
      而「一個冷 expert 把 hits-only 批次變成全 miss」正是它要守的那個失敗的**前置形狀**。
- 規則：(a) 新增一個「為了可見性」的計數器時，**同一個 commit 要指出它被誰讀**；
      在一棵樹裡 `grep` 只找到宣告與 `++`，就是還沒接線。
  (b) 閘門要有**正的覆蓋計數**（跑了幾次／幾層／幾個 batch），不要讓「沉默」成為唯一的合格訊號；
      這一條是 B12 的推廣：B12 說儀器要在自己的資料上跑過，這裡說儀器要**宣告它跑過**。
  (c) 放閘門之前先問「這個分支排除掉的是哪一種輸入」，並確認被排除的那一種不是它要抓的。
- 見 lesson `eng-gate-0029`；`scripts/check/expert_cache_ensure_batch_order.cpp`（現在 **2/2**：
  case 1 是順序、case 2 是收養——case 2 的存在本身證明了 (a) 的缺口有多大）。

**B14｜「這個修正在配置 X 下是 no-op」必須在**你要出貨的那個工作負載**上量；warmup 不是那個負載。**
- 為什麼：Blocker B 的兩趟式在**預設（非 split）配置**下被判為 no-op，證據是「warmup 期間池子從不填滿，
  miss 落進空槽、LRU 從不觸發」→ 直接修好，與未修的 binary **逐位元相同**。
  這是對的，但**它量的是 warmup**。served request 的形狀不同：8 GiB 池 + 一個 2873-token 的 prefill chunk
  會碰 256 個 expert，池子**會**滿，pass 2 **會**淘汰。所以「在該配置下是 no-op」是關於 warmup 的述句
  被當成了關於配置的述句。
- 規則：(a) 要宣告 no-op，先指出**該配置下讓那段碼可達的具體條件**，再去那個條件會成立的工作負載上量；
  (b) 把可達性本身做成一個**計數**（本輪：`n_batch_evict_batches` = 這一批的 miss 指派必須淘汰一個
  resident），而不是靠推論——「沒看到效果」與「沒有效果」是兩件事；
  (c) 逐位元相同只證明「沒有東西被改壞」，不證明「修正被測過」（與 B10 同族）。
- 見 lesson `eng-gate-0029`；`docs/M1_POOL_SPLIT_COST_2026-09-14.md` §4.1／§4.2。

---

**B15｜要先問「這個缺陷擋住誰」，再問「它是怎麼壞的」。** blast radius 是**獨立於**機制的一個量，
而且它比機制便宜得多就能量到。
- 為什麼：`prefill250` 第二個請求的 SIGSEGV 有兩個讀法，嚴重度差一個數量級 ——
  「MTP 壞了」（擋住整個出貨預設，因為 `run_server.sh:88` 的 `SERVER_MTP:-1` 預設就是 ON）
  與「這個為 prefill 吞吐而存在的 profile 壞了」。在還不知道任何機制之前，這個問題就**已經可以回答**了：
  拿一個普通配置當對照臂，連發同樣的請求。實測結果是後者（見 lesson `eng-gate-0030`）。
- 怎麼做：(a) 對照臂要選**「如果它壞了，嚴重度就跳到另一級」**的那個配置，不是選最方便的那個；
  (b) 階梯要**單變數**：`prefill250`(2048) 與 `prefill250`(6144) 之間只差 `batch/ubatch` 一格；
  (c) blast radius 要在任何「根因」或「修法」的敘述**之前**寫進文件。
- 檢查：任何寫成「X 壞了」的述句，都要能回答「那 Y 呢，Y 量了沒有」；答不出來就改寫成「X 在配置 Z 下壞了」。
- 見 lesson `eng-gate-0030`；`docs/PREFILL250_THERMAL_TRANSIENT_20260916.html` §11.9。

**B16｜一個實驗臂的標籤必須由**它實際解析出來的設定**產生，不可以由**它的意圖**產生。**
- 為什麼：本輪的 triage 腳本第一版寫 `run_arm prod25`，卻**沒有傳任何環境變數**給它，
  於是那一臂實際跑的是 `CGC_SERVER_PROFILE` 的預設值 `off`。數字是真的、有用的，
  但**標籤是假的** —— 而假標籤的產物會以「正確的證據」的形式被引用。
  這是 `eng-gate-0006`（「profile 不寫的那一格，就是會漂移的那一格」）的同一個形狀，
  只是這次漂移的是**證據的名字**而不是設定。
- 怎麼做：用專案自己的 `CGC_DUMP_ENV=1`（A11）當**前置檢查**：它在啟動任何東西之前印出
  已解析的 `CGCENV`/`ENV`/`ARG` 就退出。把 `CGCENV PROFILE` / `CGCENV BATCH` / `CGCENV UBATCH`
  抄進報告，標籤就變成**證據**而不是**宣稱**。這一招在第一秒就會印出 `CGCENV PROFILE   off`
  並當場否決 `prod25`。
- 檢查：任何多臂實驗的報告裡，每一臂都要有從 dump 抓出來的 `PROFILE`／`BATCH`／`UBATCH` 欄位；
  只有臂名而沒有 dump 的表格，不得引用。

---

**B17｜錯誤處理路徑不得「先釋放、後記錄」。** 錯誤路徑裡任何用來填寫診斷資訊的物件，
都必須在**任何** release 之前讀完。
- 為什麼：`ggml-metal-context.m` 的 `cmd_bufs_ext` 失敗路徑舊碼是
  「印 log → 釋放迴圈 → `removeAllObjects` → **才**呼叫 `cgc_metal_record_error(ctx, i, status, cmd_buf)`」，
  而那個函式在 `status == 5` 時會讀 `[cmd_buf error]`。於是 Metal OOM 的結局不是它該有的
  `CGC-METAL-FAIL` abort，而是 `EXC_BAD_ACCESS SIGSEGV`／`KERN_INVALID_ADDRESS at 0x10`
  —— **唯一的線索（GPU out of memory）被自己的錯誤處理吃掉了**。
- 這件事是**機械可證**的，不要停在「原始碼看起來是這樣」：崩潰報告的 `imageOffset` 就是位址。
  `llama-server-2026-09-16-113207.ips` 的第 1 格是 `ggml_metal_synchronize + 612`、
  `imageOffset = 81412 = 0x13E04`；把**當時那份二進位**（不是現在這份）反組譯，`0x13d7c`／`0x13db8`
  是兩個 release、`0x13e00` 才是 `bl _objc_msgSend$error`、`0x13e04` 是它的返回位址。指令序即證據。
- 檢查：(a) 錯誤路徑裡「讀物件」的每一行都要能指出它排在所有 release 之前；
  (b) 保留**當時的二進位**當證據的前提 —— 重建會換檔名（見 B18），不保留就再也證不了；
  (c) `EXC_BAD_ACCESS` 出現在錯誤處理裡時，先問「它遮蔽了什麼」，不要先問「什麼壞了」。
- 見 lesson `eng-gate-0031`；`docs/PREFILL250_THERMAL_TRANSIENT_20260916.html` §11.10.4。

**B18｜稽核「編譯期開關」要量**載入器實際映射的那個檔案**的字串存在性；不要量 mtime，也不要把檔名寫死。**
- 為什麼：SONAME 內嵌 commit 數（`libllama-common.0.0.<N>.dylib`），**每次全量重建都會換檔名**，
  舊檔留在原地變成孤兒；而載入器走的是 symlink 鏈
  （`libllama-common.dylib → .0.dylib → .0.0.<N>.dylib`，用 `otool -L` 可查）。
  一份把檔名寫死成 `0.0.239` 的證據頭，在重建後量到的是**沒有被載入的那個檔案**：
  兩份報告都印「`MTP_SUPPORT` 字串 ABSENT」—— 與事實相反，而且錯誤方向剛好是「開關沒開」。
- 另外：**mtime 不是一致性證據**。增量 `cmake --build` 只重編 CMake 自己判定為 stale 的 TU，
  所以「A 的 `.o` 比 B 的原始碼舊」可以同時成立於一個完全一致的 build。
  要判斷 stale 就去問 CMake，不要用 mtime 推。
- 怎麼做：從 `otool -L` 反推 `@rpath` 載入集 → 逐一 `realpath` → 對**那個**檔案做 `strings -a`，
  並印出**判準句**（`=> guarded 3/3, control present => MTP_SUPPORT=ON (compiled in)`）。
  **control 字串（不受 guard 保護的那個）必須一起印**：control 也不在時，那是 artifact 壞了，
  不是開關的判準。
- 檢查：任何「開關是開／關」的結論，都要附上被量檔案的**解析後檔名**與 control 的結果；
  缺任一個就不算量過。
- **同族的第二個實例（2026-09-16 實測）：閘門的**檔案比對樣式**漏一種副檔名，等於對那個類別完全無效。**
  `scripts/check_build_tracked.sh` 的檢查 8 用一個 `case` 樣式挑出「會被編進 binary 的 llama 原始碼」，
  原本列了 `*src/*.cpp|*.h|*.c|*.mm|*.metal`——**獨漏 `.m`**（只列了 Objective-C++ 的 `.mm`）。
  後果：改 `ggml/src/ggml-metal/ggml-metal-context.m`（本 repo 的 Metal 工作幾乎都在這個檔型裡）
  會被判成 `8 無 llama 原始碼變更（僅 doc/腳本/產物）`——一個**假 PASS**，
  而且它同時跳掉了「產物有沒有一起 staged」與「binary 是否比原始碼新」兩條。已補上 `*src/*.m`
  （該 repo 只有 2 個 `.m`，都在 `ggml/src/ggml-metal/`，都編進 `libggml-metal`，所以樣式是精確的），
  陽性對照：餵它一份含 `.m` 的假 staged 清單，`staged_src` 由 0 → 1。
- 見 lesson `eng-gate-0032`；`Backup/run_req2_retest.sh` 的證據頭、`scripts/run_server.sh:325-341`、
  `scripts/check_build_tracked.sh:384-397`。

**B19｜交回來的「仍未解」清單，先分類再動手：缺陷／缺失的可見性／政策漂移／已過期的述句。**
- 為什麼：這四類的「修好」是不同的動作，而且只有第一類需要改行為。
  2026-09-16 一次交回來四條，其中
  (a) `req2` 的 OOM 是**真缺陷**（只能繞道或修機制）；
  (b) `--ngl 99` 讓 `common_fit_params` 被跳過**不是 bug** —— `-ngl` 是本腳本顯式帶的，
  `fit.cpp:377-379` 因此 throw，`-fit` 在這條路徑上**從來沒跑過**。要修的是「啟動訊息完全不提它」，
  改 fit 的行為反而會改掉記憶體／效能剖面與 oracle 的可比性；
  (c) build dir 與 `build_fork_llama.sh` 不一致是**政策漂移**（其中的 `GGML_*` 四個是已裁決的，要對齊；
  `LLAMA_BUILD_SERVER` 的差異是刻意的，不要對齊）；
  (d) §11.9.5 的述句是**已過期的述句** —— 只能加時間範圍，不能偷偷改數字。
- 怎麼做：每一條先寫下「它是哪一類」再動手；把 (b) 這種當 bug 去修，會把一個**可見性問題**
  換成一個**配置變更**（更貴、且會污染既有閘門的可比性）。
- 檢查：白皮書那一節要有一張「類別／處置／證據」表；說不出類別的那一條，代表還沒想清楚。

**B20｜相容性缺陷（唯讀映射被寫）要用**硬閘門**擋住，而閘門要四面都量。**
- 為什麼：`CGC_SERVER_LOAD_MODE=mmap` 與 expert cache 不相容 —— 後者的 L4 pool 是
  「regions adopted from expert tensors」，直接寫進模型張量儲存；`mmap` 下那是唯讀 file-backed 映射，
  於是 `fill_pool_direct` 的 zeroing（`llama-expert-cache.cpp:704`）寫進唯讀頁 →
  `SIGBUS` / `KERN_PROTECTION_FAILURE` 在 `__bzero`，**載入階段**就死。
  這種缺陷的「修」要動儲存所有權（另一輪的事），所以本輪的正確產物是**閘門**，不是一個
  「試試看」的建議。
- 怎麼做：閘門四面對照 —— ① 單獨（拒跑，rc=1）、② 合法豁免（`EXPERT_CACHE_OFF=1`，rc=0）、
  ③ 強制放行（`ALLOW_…=1`，rc=0 + warning）、④ 對照（原路徑不受影響）。
  只量①等於沒有量：你不知道豁免路徑是否也被誤擋。
- 另外：**閘門一旦成立，就要回頭把它從「建議槓桿」清單裡拿掉**。同一次改動裡
  `[budget]` 超額段原本把 `mmap` 列為槓桿 —— 一個會 SIGBUS 的選項不是槓桿。
- 見 `scripts/run_server.sh` 的 `[防護 2e]`、`Backup/cgc_logs/req2retest_20260916_120057.txt`、
  `~/Library/Logs/DiagnosticReports/llama-server-2026-09-16-120239.ips`。

**B21｜OOM 要寫成算式，不要寫成形容詞；而且「池的上限」不等於「實配」。**
- 為什麼：`-expert-cache $BUDGET` 的 8192 MiB 是**上限**，實配是 `5976 槽 ÷ 40 層`
  （`LAYER_CAPS per-layer caps: total 5976 slots`），每層 `110 MiB@256 experts`
  （`CGC-PREFILL-STREAM … slab=110.00 MiB`）⇒ 約 2.5 GiB。
  把上限當實配，會把 `21222 MiB vs 16384 MiB`（OVERSUBSCRIBED 4838 MiB）算成別的東西。
  同理 `KV` 不能靠直覺：主 context 是 hybrid memory，filter 是 `!is_recr(il)`
  （`llama-model.cpp:2292-2295`），`full_attention_interval=4` ⇒ 只有 10 層帶 K/V ⇒
  fp16 @8192 = 160 MiB（本模型；全 40 層 dense 才是 0.64 GiB）。
- 怎麼做：啟動時把算式印出來（`[budget]` 五～九行），並且**每一項都標明是算術還是實測**。
  沒量過就標「算術，未實測」——寫成既成事實的估計值，比沒有這個數字更糟。
- 檢查：算式要能指到它的兩個來源（此處：`llama-model.cpp:2292-2295`／`llama-memory-hybrid.cpp:48-50`
  與 `LAYER_CAPS` 的 log 行）。指不到的項不要印。
- 反例（自己踩的）：第一版把 10 層算成 `164 MiB`（`20 KiB/token × 8192` 的 KiB/MiB 換算算錯，
  正確是 `160 MiB`）。**算術也要複核**，尤其是單位換算。

---

**B22｜`set -o pipefail` 下 `… | grep -q` 會因為上游的 `SIGPIPE` 回報失敗 —— 閘門的判定要用 `grep -c`。**
- 為什麼：`grep -q` 一命中就關閉 pipe，上游（`strings`、`otool`、`nm`）在寫入時收到 `SIGPIPE`，
  以 `141` 結束；`pipefail` 把整條 pipeline 的 rc 判成非零。於是**條件成立的那一面回報失敗**。
  這是 fail-closed 的假陰性，也就是最危險的方向：閘門在「已修好」的 binary 上 `exit 1`。
- 實測（同一台機器、同一個含修復的 `libggml-metal`，三面）：
  | 寫法 | rc | 觀察 |
  |---|---|---|
  | `pipefail` + `grep -q '_Pool'` | **1** | `strings: failed to flush output` ⇒ 修好的binary被判成沒修 |
  | `pipefail` + `grep -c '_Pool'` | 0 | `n=1`，正確 |
  | 無 `pipefail` + `grep -q` | 0 | 正確（但別靠這個） |
- 怎麼做：判定式寫成 `n="$(strings -a "$f" \| grep -c 'needle' \|\| true)"`，再 `[ "$n" -gt 0 ]`。
  `grep -c` 會讀完輸入，所以上游不會拿到 `SIGPIPE`。
- 檢查：閘門的**兩面**都要跑（含修復／不含修復）。只跑失敗的那一面永遠不會發現這個 bug。
- 見 `scripts/run_server.sh` 的 `CGC_MMAP_POOL_FIX` 判定；同一條也適用於
  `Backup/run_req2_retest.sh` 的 `strings -a | grep` 診斷。
- 同一族、另一半（自己踩的）：**`grep` 的 rc 不只在 pipeline 裡有意義，它會短路 `&&` 串。**
  `cat summary.tsv | grep -v '^#' && date && pgrep …` 對一個「只有註解行」的檔案回報 rc=1
  （沒有任何行被選中），於是 `date` 與 `pgrep` 從來沒跑，`||` 分支印出「runner: done」——
  而那個實驗其實正在跑。**用 `grep` 的 rc 當「有沒有東西」的判準時，要明說你要的是
  「有不符合的行」還是「有選中的行」**，並且不要讓它去串控制流；要串就加 `|| true`。

**B26｜`${VAR:-default}` 把「空字串」當成「沒給」——用空值去關掉一個有預設的參數，會拿到預設。**
- 為什麼：`ARMS_R1="${ARMS_R1:-none:4096 mmap:4096 mmap:6144 none:6144}"` 對 `ARMS_R1=""`
  會套用預設。所以「我想跑一個不啟動任何臂的測試」實際上啟動了那個 4 臂、兩輪的實驗。
- 怎麼做：要一個「什麼都不做」的值，用一個真的會被解讀成空集合的值（例如 `ARMS_R1=" "`
  再在迴圈裡跳過空字串），或加一個顯式的 `DRY_RUN=1` 分支；**不要用空字串當 sentinel**。
  反過來，要「沒給就用預設」，`:-` 是對的；**但要意識到它同時吃掉了空字串**。
- 實測（2026-09-16 13:04）：`OUTDIR=… ROUNDS=1 ARMS_R1="" ARMS_R2="" bash Backup/run_mmap_ab.sh`
  的本意只是測摘要表的列印，結果啟動了預設 4 臂；而且它在前景被工具切斷後**變成孤兒繼續跑**，
  三個連續的呼叫各自「完成第一臂、開始第二臂、被切斷」，於是在 `summary.tsv` 裡留下三個
  獨立的 run header 與三列 `none:4096`——看起來像三次實驗，實際上是三次被截斷的同一次。
  副產物是好的（三列同指紋的 `none:4096` 給了噪聲估計），但那是運氣，不是方法。
- 推論（同一件事的另一面）：**長時間的實驗一律用 `nohup` 起、不要掛在會被收回的前景呼叫上。**
  同一輪裡 `nohup` 起來的 8 臂階梯就沒有這個問題。

**B23｜不要編輯**正在執行**的 shell 腳本：bash 以位元組位移增量讀取腳本，改檔會讓它從錯位處繼續讀。**
- 為什麼：bash 不會把腳本整個讀進記憶體再執行，而是邊執行邊 `read` 下一段。檔案在執行中被改動後，
  bash 用它手上的舊位移去讀新檔，於是讀到的是別的東西——症狀是「既有的行突然變成 `command not found`
  或 `syntax error`」，而且錯誤行號指向**已經修正過、語法完全正確**的行。
- 實測（2026-09-16 12:53–12:57）：`Backup/run_mmap_ab.sh` 正在跑 `none:5120` 這一臂時，
  我對它連發四個 `Edit`。該臂**本身完整跑完並印出結果**（子行程 python 已把 summary 列寫入），
  但父行程接著從錯位的位移讀到 python 的 `"common_md5": common[:12],` 與 `(`，
  報 `line 156: n[:12],: command not found` / `line 157: syntax error near unexpected token '('`。
- 怎麼做：實驗跑完之前，`Edit` 只碰**執行鏈之外**的檔案。要改執行鏈上的腳本，
  先等它結束（或先 `kill` 再改）。同一條也適用於 `scripts/run_server.sh`：它會被每一臂重新
  `bash` 起來，中途改它等於**在同一輪實驗裡換掉受測配置**。
- 檢查：改動執行鏈上的檔案前，先確認 `pgrep -f <腳本名>` 是空的；實驗進行中只讀不寫。

**B24｜修好一個缺陷之後，要重新量**它原本蓋住的限制**。**
- 為什麼：缺陷會把下游的行為遮住。SIGBUS 讓「pool 真的配置」這件事在載入階段就死，
  所以沒人看得到「pool 真配置之後，`ub` 的上限變成多少」。修好之後那條限制才會顯形，
  而且它與缺陷本身無關。
- 實測（2026-09-16 12:02 vs 12:46，同 profile、同 `ub=4096`）：
  | 面 | 死在哪 | 訊號 |
  |---|---|---|
  | 修復前 | 載入中的 fill pool | `KERN_PROTECTION_FAILURE`（`__bzero ← fill_pool_direct`），SIGBUS |
  | 修復後 | 載入完 40 層、第一個 `synchronize` | `status 5` `kIOGPUCommandBufferCallbackErrorOutOfMemory`，SIGABRT |
  SIGBUS **消失**（修復成立），但換成 GPU OOM 在 warmup —— **位置從「負載下」提前到「warmup」**，
  因果正確：pool 現在要真配置，載入期的壓力不再被可回收的檔案映射吸收。
- 怎麼做：修復的驗收條件寫成「X 的訊號消失」**加上**「新的第一失敗點在哪、它是什麼」。
  只寫前半句，會把一個退步（更早死）記成一個進步。
- 檢查：新失敗點必須指到具體的 `status`/`errno` 與堆疊行，不能寫「還是會當」。

**B25｜macOS 的 BSD `grep` 不支援 `\|` 交替 —— 它會**安靜地**比對字面上的一根直線，rc=1。**
- 為什麼：`grep -n 'a\|b' f` 在 GNU grep 是交替，在 BSD grep 是**字面** `a|b` ⇒ 找不到 ⇒ 空輸出、rc=1。
  「沒有符合的行」與「這個字串不存在」在輸出上同形。
- 實測（2026-09-16 13:0x）：`grep -n 'id="s1111"\|id="s1110"' docs/*.html` 空輸出，
  而同一組 id 用 ripgrep 立刻命中（`:2081`、`:2128`、`:2356`、`:2372`）。
- 怎麼做：需要交替就用 `grep -E 'a|b'`，或直接用 ripgrep（本 repo 的 `Grep` 工具即是）。
  `grep -c`/`grep -q` 這種單一固定字串不受影響（B22 的修法仍然是對的）。

**B27｜`cmake --build` 的 exit code 0 不代表產物是最新的 —— 唯一的判準是它有沒有印出編譯行。**
- 為什麼：cmake 只在目標比來源舊時才編譯，所以 `0` 這個 exit code 同時代表「已經最新」與
  「剛剛編好」兩件事。看 mtime 也一樣看不見：磁碟上有一份產物，它看起來就是權威的。
  於是「改完原始碼、還沒重建、直接開始量測」會產生一整套數字，而它描述的是**上一份**二進位。
- 實測（2026-09-16）：12:44 全量重建之後，13:00:34 又編輯了
  `ggml/src/ggml-metal/ggml-metal-context.m`（把 `cmd_buf_last` 的註解從「窗口很窄」改寫成
  查過的事實）。沒有重建。於是 12:44–14:0x 之間**每一筆**量測都跑在不含該檔的產物上：
  13:45 的生產驗收（268.09/273.55/256.61 t/s）與 13:50 那個 **D5 PASS** 都是。
  判準是 `cmake --build` 印出的那一行
  `Building C object ggml/src/ggml-metal/CMakeFiles/ggml-metal.dir/ggml-metal-context.m.o`；
  重建後 `libggml-metal` md5 `968c36cf45742bb1667d5a02629c67be` → `f2d1c96193939bd15404ba713a6fa85d`。
- 怎麼做：原始碼改過就在跑**任何**測試之前對該 target 跑 `cmake --build`，並把「有沒有編譯行」
  當成驗收條件（印出編譯行 ⇒ 先前在同一棵樹上取得的所有數字全部失效，要重跑）；
  追求更強證據時比對重建前後的產物 md5，把兩個值都貼進 commit body。
- 與 B7 同族但更便宜：**exit code 是 0，錯的是沒有人讀輸出。**
- 這一格閘門本來是有的，錯的是時序：check 8（原始碼 ↔ binary 同步）用的是 mtime
  （`binary 12:44` < `source 13:00` ⇒ 會 FAIL），但它在 **commit 前**才跑，而量測在 commit 前
  更早就發生。**中間那段空窗裡只有建置輸出能擋。**

**B28｜gate 的「參考身分」不能寄生在生產調參上：oracle 的 numerics-determining 旋鈕要釘在參考檔旁邊。**
- 為什麼：`scripts/check/m123_oracle_gate.py` 的 `--profile` 預設 `prefill250`，而它「預設會重現
  oracle 的配置」這個性質，**寄生在那個 profile 的生產預設值上**（batch=ubatch=6144）。
  2026-09-16 把生產預設改成 5632（6144 在 req2 上 0/5 死、5632 4/4 活）之後，gate 立刻對
  `CGCENV.BATCH` / `CGCENV.UBATCH` / `ARG[27]` / `ARG[29]` 四列差異報 **INVALID COMPARISON**，
  而那些差異 100% 是我自己的預設變動。當下 M1/M2/M3 都是 9/9 —— 於是「讀成 PASS」與
  「讀成 FAIL」都能各自找到支持。一個閘門的預設值被無關的生產決定移動，等於這個閘門的
  意義是可變的。
- 怎麼做：把 oracle 的 numerics-determining 旋鈕寫成常數放在 `DEFAULT_REF` 旁邊
  （`ORACLE_PINNED_ENV`），由 gate 在 resolve 之前併進 `--env`；重新基線要**顯式**
  （`--write-ref` ＋ `--no-pin-oracle-env`）。CLI 覆蓋仍然允許，但**有效集合必須在啟動前印出來**：
  「被靜默覆蓋」與「被靜默丟棄」在 transcript 上同形，而後者正是這個旋鈕存在的理由。
- 殘留（不可省略）：這樣一來 gate 證明的是 **oracle 配置（6144）**的數值不變。出廠預設 5632
  沒有自己的參考檔，所以它只量得到「M1/M2/M3 都 9/9」（跨配置，gate 依法不裁決），
  **不能被 gate 認證**。要認證就得為 5632 重新基線一份參考。

**B29｜交付述句可以是條件式的，但條件本身必須可量測；量不到的時候，閘門只能「分類宣稱」，不能「宣稱已滿足前提」。**
- 為什麼：條件式交付（「在 X 下達到 250 t/s」）是合法的，前提是 X 是一個**消費者可以自己驗證**的述句。
  2026-09-16 的 `prefill250`：出廠預設在最終產物上量到
  **254.29 / 282.38 / 265.27 t/s**（req1–req3 全部 ≥ 250，全程存活、0 crash report），
  而同一個小時內另外四臂是 118.30–154.77 —— 於是「已交付」與「未達成」都能各自找到支持。
  把條件寫成「機器涼的時候」是不可驗的；寫成「啟動前靜置 ≥150 s」是**可驗但錯的**（見 B30）。
- 更硬的一條：**條件的可觀測性決定了閘門的形式**。`pmset -g therm` 在本機只回
  `Note: No thermal warning level has been recorded`；要讀 DVFS 駐留與有效時脈需要 root 的
  `powermetrics`。所以**沒有任何非 root 儀器能回答「governor 現在選了哪一階」**。
  在這種情況下，一個會印「已滿足前提」的閘門是**假證據**——它宣稱讀到了一個它讀不到的東西。
  正確的形式是**分類**：`Backup/run_req2_retest.sh` 依狀態檔算出距上一臂結束的安靜秒數，
  把每臂標成 `COLD-STATE`（≥ `COLD_QUIET`，預設 1800 s，它的 t/s 可被引用）／
  `HOT-STATE`（只能當熱態觀察）／`UNKNOWN-STATE`（無狀態檔 ⇒ **兩邊都不宣稱**）。
- 怎麼做：驗收協議要**自己帶證據**——(1) 量測前 ≥30 分鐘沒有持續 prefill 與 CPU 密集建置；
  (2) 只採信該安靜期後的第一次啟動；(3) 樣本是該臂的 `req1` 且必須同時報機器狀態
  （安靜秒數／swap／load）；(4) 有 root 時**同時**跑 `scripts/check/powermetrics_gpu_freq.sh`，
  判準取 parser 的 verdict（DVFS 駐留是否由最高階主導），而不是 t/s 的絕對值；
  (5) 用 §10.6 的 `t = 0.5575 + 4226/f` 反讀機器在哪一階（1470 → 291、928 → 227、618 → 135）。
- 邊界：即使照做，**熱態平台的高度仍未被建模**——兩個 soak 序列的平台是 ≈130 與 ≈185 t/s
  （差 55 t/s），中間夾著兩次 D5 閘門與一次 8 核 CPU 重建，而它們的負載本輪沒有量化。
  所以可交付的是「`req1` ≥ 250 且該臂全程存活」，不是「250 隨時可重現」。

**B30｜用「事件時間戳」算間隔，不要用「報告戳記」相減；而且熱態的預測子是啟動序，不是間隔長度。**
- 為什麼（第一個錯，我自己犯的）：`run_req2_retest.sh` 的報告檔名是**腳本啟動**的戳記，
  而內容是跑完才寫的。用「相鄰兩個報告戳記相減」量「臂與臂之間的休息」，
  等於把**上一臂自己約 90 秒的執行**算成了休息：13:45–13:57 那四臂因此被記成
  「111 / 94 / 144 s 的間隔」，而用伺服器 log（下一臂 log 的創建時間 − 上一臂 log 的 mtime）
  重算是 **21 / 8 / 55 s**。差一個量級，而且兩種數字支持相反的讀法。
  判準：要量的是「上一個事件結束到這一個事件開始」，所以必須用產生那些事件的檔案的
  **創建／mtime**，不是包住它們的驅動腳本的戳記。
- 為什麼（第二個錯，白皮書 §10.10.4 第二層的）：那一層寫「距上次持續 prefill ≥150 s」，
  讀起來是充分條件。實測把它推翻：安靜 **18 s** 的那一臂 req1 是 **289.86 t/s**，
  比安靜 32 分鐘那臂的 254.29 更快；而安靜 **34 s** 的那一臂只有 **166.95**。
  **安靜最長的最慢、最短的最快** ⇒ 一致的解釋不是「休息了多久」，而是「已經被操了多久」：
  一次啟動（約 40 s prefill）不足以耗盡熱預算，三次以上會，之後秒級的安靜買不回多少。
- 怎麼做：要預測一臂會落在哪個平台，看的是**同一序列裡這是第幾次啟動**（累積負載），
  不是它前面空了幾秒。要排除一個候選（例：swap），找**順序相反**的配對而不是找相關：
  289.86 t/s 那臂的 `swap used=4975.06M`、188.35 那臂是 `4549.69M`、13:51 的慢臂是 `5066.06M`
  —— 而且那 32 分鐘安靜期間 swap 自己從 5066M 掉到 2743M，所以它是**同一個潛在變數的果**。

---

**B31｜不要在同一則訊息裡對同一個檔案送出兩個編輯 —— 它們會 race，其中一組會靜默消失。**
- 為什麼（我自己犯的，2026-09-16 15:15）：對 `Backup/run_req2_retest.sh` 同時送出
  「加更正註解 + 加 `_thermal_level()` 函式」與「加機器狀態讀數」兩個 Edit，
  兩個都回報成功，但**只有第二個的內容在磁碟上**。呼叫點（`T_PRELAUNCH="$(_thermal_level)"`、
  `_L0=`、`_L1=`）都落地了，函式定義沒有。
- 後果是**假陰性**，不是崩潰：腳本沒有 `set -e`，未定義函式在 `$( )` 裡只讓 stderr 多一行，
  而 stdout 是空字串 ⇒ 報告印出 `[thermal pressure] before req1 = `（空白），
  看起來像「這個儀器讀不到」，而不是「這個儀器不存在」。
  三臂的**發射時等級**由外層 shell 印出，所以那些臂的判決仍然有效；
  **in-band 讀數**則完全無效。
- 怎麼做：同一個檔案的多處修改一律**逐次送出**（一次一個 Edit，等結果再送下一個）。
  若已經平行送出，收尾前用 `grep -n` 確認**定義與呼叫都在**——
  「呼叫點存在」不等於「定義存在」，而兩者在畫面上長得很像。
- 題外但相關：未定義指令的錯誤在 `{ ... } 2>&1 | tee` 裡會被寫進報告，
  但**替換後的字串是空的**，所以只看 `grep 'thermal pressure'` 會漏掉它。

**B32｜「沒有儀器可以讀」是關於**某個工具**的推論，不是關於**系統**的事實。**
- 為什麼（我自己犯的，寫進了白皮書 §11.14.7）：我寫下「沒有任何非 root 的即時儀器可以回答
  governor 選了哪一階」，理由是 `pmset -g therm` 回 `No thermal warning level has been recorded`、
  而 DVFS 駐留要 root `powermetrics`。兩句話都對，結論是錯的。真正的儀器是
  `notifyutil -g com.apple.system.thermalpressurelevel` —— 就是 `powermetrics` 讀的那條
  notify(3) key，**不需要 root、約 11 ms/次、可 2 Hz 取樣**。
- 搜尋停在「這個工具要 root」，於是「**這個**工具不可用」變成「**沒有**工具可用」。
  一個不存在的儀器與一個沒找到的儀器，在結論的措辭上長得一模一樣。
- 怎麼做：要寫「不可觀測」之前，先把介面列一遍再下結論 ——
  notify(3) key、Foundation/AppKit API（`NSProcessInfo.thermalState` 就是這樣找到的）、
  `sysctl`、`ioreg`（IOReport 的 channel **id** 在，讀數不在）、`pmset`、`powermetrics`。
  **正確的措辭是「這些介面都試過了，以下是各自的結果」，不是「沒有儀器」。**
- 反面教材同樣要記：`NSProcessInfo.thermalState` **看起來**是那個缺失的儀器
  （非 root、Foundation、四級刻度），但 367 個樣本跨越滿載與 4 分鐘閒置**全部讀到 `1/fair`**
  ⇒ 它沒有區辨力。**找到一個介面不等於找到一個儀器；要用它與被解釋的量之間的實際分離度來認證。**

---

## C. 讀原始碼

**C1｜變數名稱不等於語意。**
- 為什麼：`build_moe_ffn` 的 `n_tokens` **不是** `ubatch.n_tokens`（實測 hook 看到 `t->ne[1]==2` 而 ubatch 是 1；
  80 條 hook 行全部 `n_tok=2`）。照字面寫 `n_tokens == 1` 當 decode 條件，整個 decode 步根本沒建表。
- 證據：`Backup/cgc_logs/llama_server_20260915_201*.log`（`CGC-S1: CAPTURE … ntok=`）。

**C2｜`ggml_reshape_*` 之前先確認連續性。**
- 為什麼：`ggml_argsort_top_k` 的輸出是 `ggml_view_4d`，保留 `nb[1] = n_expert*4`、只把 `ne[0]` 切到 k。
  `ggml_reshape_1d` 內含 `GGML_ASSERT(ggml_is_contiguous(a))` → 編圖期直接 abort（trap 6）。
- 證據：`src/llama.cpp/ggml/src/ggml.c` 的 `ggml_argsort_top_k` / `ggml_reshape_1d`；
  `ggml_cont` 不短路（無條件建 `GGML_OP_CONT`）。

**C3｜「同一個值」要問它是不是**結構性**地相同。**
- 為什麼：`series` 看起來「兩臂一樣」，但一臂的值來自 host 寫入、另一臂來自 GPU 計算——
  同一個數字，不同的因果。分不出因果就不能預測它下一次會不會一樣。

---

## D. 流程與入口

**D1｜入口腳本不能只做 `bash -n`。**
- 為什麼：`tb_loop/` 的三個入口（`run_round.sh:39`、`gen_sft.sh:27`、`finetune/eval_round.sh:34`）
  都寫 `--agent-import-path "tb_loop.agents.…"`，所以解譯器必須看得見名為 `tb_loop` 的**套件**——
  而 `PYTHONPATH` 必須是它的**父目錄**。E1 之前三者都指向 `$TB_LOOP_DIR`，而 `TB_LOOP_DIR`
  當時恰好就是 `agent_harness/` 本身（因為 `tb_loop/` 這個目錄還不存在）⇒ 實測
  `ModuleNotFoundError: No module named 'tb_loop'`，**三個都跑不起來**，`bash -n` 抓不到。
- **一般化的那一條**（比這個實例重要）：`PYTHONPATH` 指向的目錄與 import 字串的**第一段**
  必須是同一件事的兩個名字。當「套件名」與「檔案所在地」由同一次搬動決定時，
  兩者會一起對、也一起錯——所以**搬動之後要重驗的是 import，不是路徑字串**。
- 驗收方式：一次真的 `import` ＋ 一次 `--n-tasks 1` 的 smoke，不是語法檢查。
  而且 `import` 測試要有**對照組**：把 `PYTHONPATH` 指回套件自己，必須仍然報
  `No module named 'tb_loop'`；否則「通過」可能只是因為當前工作目錄剛好也在 `sys.path` 上。
- 現況（2026-09-16，E1）：三個入口的 `PYTHONPATH` 已改為 `$TB_HARNESS_ROOT`（= `tb_loop` 的父目錄），
  `tb_loop/` 補了 `__init__.py`。**`import` 層面已通過**（四個模組都解析到新位置、對照組正確失敗）；
  **smoke 未跑**（`tb_loop/.venv` 不存在、`terminal-bench` 未安裝、Docker daemon 未執行），
  所以這一條的驗收**只結清一半**，不得寫成「已通過」。
  見 `docs/AGENT_HARNESS_E1_RESTRUCTURE_20260916.html`。

**D2｜`RUN_REPLAY_BENCH=0` 一律預設。**
- 為什麼：`.replay_bench_baseline.json` 是 stale 的，跑它等於拿舊基線比新程式。

**D3｜訓練數據不是對外數字。**
- 為什麼：`traces/` 裡的 ×1.78、`id_oob first=2143289344` 都是**未定案或已知錯誤**的觀測。
  對外引用一律以 `docs/*.html` 白皮書為準，而且每筆 record 都保留 `usable_as_evidence` 欄位。
- 檢查：`validate.py` 會擋 `build == null` 卻 `usable_as_evidence == true` 的記錄。

**D4｜被推翻的假設不得當正例訓練。**
- 為什麼：「雙緩衝 remap 就能拿到 ×1.78」是**被推翻**的假設（CPU 往返仍在關鍵路徑，`n_segs` 仍是 39–40）。
  當正例訓練等於教模型重犯。
- 檢查：`decision.judgement` 必填，`judgement == refuted ⇒ superseded_by` 非空（`validate.py` 強制）。
- 反例的用法：只能配**偏好對**（DPO/ORPO），或加顯式「這是錯誤示範」前綴。

**D5｜每次 commit 附一份技術白皮書，並在提交前跑完 `llama-bench` + M1/M2/M3 + 最新 M2 oracle。**
- 格式沿用既有 HTML 版式（`docs/PREFILL250_DECODE25_WHITEPAPER_*.html`）。

**D6｜專案記憶只由索引進入 loop，不複製。**
- 為什麼：`.workbuddy/memory/`（`MEMORY.md` ＋ 每日 append-only 日誌）由 host 在 loop 執行時持續
  寫入。把內容複製進 `agent_harness/` 會讓同一批 bytes 有第二個權威來源，而第二個來源的失效方式
  是**安靜**的：原檔改了，副本看起來還一樣權威。`scripts/check/*` 已有同一條規則（放索引、不複製）。
- 怎麼做：`engine_loop/memory/build_memory_index.py` 由 `.workbuddy/memory/` 推導 `INDEX.jsonl`
  （一檔一列，＋每個 `##` 一列，含 `line_start`/`line_end`/`###` 子標題）；`--query TERM` 回報
  **段落位置**而不是全文。理由是可讀性：當日誌到 1000+ 行時，整檔讀取會靜默截斷，而截斷後的
  「沒有查到」與「從沒寫過」同形。
- 引用方式：`decision.action`／`lesson.evidence` 要引 `path:line_start`（例
  `.workbuddy/memory/2026-09-15.md:854`），**不要複述內容**——複述會漂移且無從察覺。
- 檢查：`build_memory_index.py --check` 重新推導並在 drift 時 exit 1；`index_assets.py --check`
  另驗 MANIFEST 的 bytes/mtime（記憶檔在 `VOLATILE_PREFIXES` 內，只驗存在——每日日誌當天必變，
  永遠紅的閘門等於沒有閘門）。
- 邊界：`~/.workbuddy/MEMORY.md`（跨專案個人偏好）與雲端 profile **不在**這條規則內，它們的
  scope 不是這個 repo。
- **修訂 2026-09-16｜本條對 loop 的約束一字未改，只多承認一份非權威的 dated 快照。**
  - 新事實：`agent_harness/scripts/auto_git_push.ps1`（來自 remote 0907 的 `fusionroutemot`，
    commit `1f3b0a78d`）會週期性 `git add agent_harness` 後 push。**索引運送的是指標，不是內容**
    ⇒ 那條推送線拿不到任何記憶本文。這讓「內容只放 `.workbuddy/memory/`」出現一個具體的失效消費者。
  - 因此 `agent_harness/memory/`（3 檔）與 `agent_harness/skills/`（3 skill）多了一份 **dated
    快照**：banner 插在 YAML frontmatter **之後**並標明權威位置；`SNAPSHOT.jsonl` 逐檔記
    `source` / `source_sha256` / `source_bytes` / `source_mtime` / `snapshot_date`，讓「對應原檔
    哪一版」可查。
  - **要引用事實或餵 loop，仍然讀 `.workbuddy/memory/`。** 快照的**唯一**用途是跨機器搬運；
    `build_memory_index.py --check` 與 `index_assets.py --check` **都不驗**快照（只驗原檔↔索引），
    所以兩者要一起重生：先 `agent_harness/scripts/import_harness_snapshot.py`，後 `build_memory_index.py`。
    （該腳本 2026-09-16 從 `Backup/import_harness_snapshot.py` 移進來——它原本不受版控，
    而「一個要跨機器的機制，自己得先能跨機器」。同時它的兩個硬編碼清單改成 glob。）
  - **未完成的義務（不可當成已完成）**：本檔是 `engine_loop/sft_pi/` 的 system prompt 與
    `harness_engine/memories/engine/` 的注入來源，§E 要求「每次改動都要走一次閉環對照」。
    2026-09-16 當天**這兩個目錄都還不存在**（E2 才建），所以這次修訂改變不到任何執行時行為；
    但該閉環對照**尚未執行**，義務記在 E2 頭上。
    ★ **19:4x 更新（E2 已落地，這一段的前半現在是錯的）**：那兩個目錄**現在存在**，
    所以「改變不到任何執行時行為」這句話不再能靠「沒有消費者」成立——它變成一個**待驗證的斷言**。
    對照器與四臂已建（見 §E 的最後一條），**模型半仍未跑**，所以這句話目前**沒有產物支撐**。
    在它跑過之前，本條的措辭應讀成「**預期**改變不到任何執行時行為（D6 動的是搬運管道，
    不是判準），但未經對照」。
  - 反向紀錄：`agent_harness/memory/README.md`、`agent_harness/skills/README.md` 與本條一起
    構成三個防線（banner / `SNAPSHOT.jsonl` / 條文），lesson `eng-bound-0004` 記的是「索引
    可驗證 ≠ 索引充分」這個一般化的規訓。

---

## E. 這份憲章的自我約束

- 以上每一條都指到了具體檔案／行／欄位。指不到的條文不寫進來。
- 條文本身可以被推翻：若某條的證據被後續實驗推翻，該條標 `superseded_by` 而不是刪掉
  （刪掉會讓後人重新踩一次）。
- 這份文件是 `engine_loop/sft_pi/` 的 system prompt 與 `harness_engine/memories/engine/` 的注入來源，
  所以**改動它等於改動兩個 loop 的行為**，每次改動都要走一次閉環對照（見 `PLAN_ENGINE_LOOP_2026-09-15.md` §6.3）。
- **2026-09-16 19:4x：上一條的「注入來源」現在是真的了，因此閉環對照的義務從「未到期的條件」變成「已到期的欠帳」。**
  E2 建立了 `engine_loop/sft_pi/` 與 `engine_loop/harness_engine/memories/engine/`（106 條），
  所以 `sft_common.charter()` 現在真的把這份檔案讀成 system prompt 的位元組。
  對照器是 `engine_loop/distill/closed_loop.py`（四臂：A_preD6／B_postD6／C_head／D_head_mem，
  **只有 A→B 回答 D6 的欠帳**，另兩對是反歸因用的；`--dry-run` 已證實每對只隔離一件事）。
  **狀態：機制已建並離線驗證；模型半未跑。**
  未跑的理由是具體的、不是「做不到」：本機無模型端點，而 `llama-server` 與 13.6 GB 的模型都在
  ⇒ 起 server 會與另一個 session 正在進行的 prefill/decode 量測互相干擾（那些量測對 GPU 時脈
  與記憶體狀態極敏感，見 A16）。**為了結清一條欠帳而弄壞另一條線的資料不算結清。**
  ⇒ **在 `closed_loop.py` 真的跑過、且 `closed_loop_questions.md` 的承重點逐題核對過之前，
  不得聲稱 D6 的修訂「改變不到任何執行時行為」。**
