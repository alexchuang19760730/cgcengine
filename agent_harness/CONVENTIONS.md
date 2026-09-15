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
