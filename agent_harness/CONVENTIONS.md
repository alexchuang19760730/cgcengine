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
- 為什麼：`agent_harness/` 的三個入口（`run_round.sh:37`、`gen_sft.sh:27`、`finetune/eval_round.sh:34`）
  都寫 `--agent-import-path "tb_loop.agents.…"`，而 `PYTHONPATH=$TB_LOOP_DIR`（＝ `agent_harness/` 本身）
  → 實測 `ModuleNotFoundError: No module named 'tb_loop'`。**三個都跑不起來**，`bash -n` 抓不到。
- 驗收方式：一次真的 `import` ＋ 一次 `--n-tasks 1` 的 smoke，不是語法檢查。

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

---

## E. 這份憲章的自我約束

- 以上每一條都指到了具體檔案／行／欄位。指不到的條文不寫進來。
- 條文本身可以被推翻：若某條的證據被後續實驗推翻，該條標 `superseded_by` 而不是刪掉
  （刪掉會讓後人重新踩一次）。
- 這份文件是 `engine_loop/sft_pi/` 的 system prompt 與 `harness_engine/memories/engine/` 的注入來源，
  所以**改動它等於改動兩個 loop 的行為**，每次改動都要走一次閉環對照（見 `PLAN_ENGINE_LOOP_2026-09-15.md` §6.3）。
