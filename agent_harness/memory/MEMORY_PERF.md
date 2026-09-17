# MEMORY_PERF — profile／模型／幾何／速度／散熱

> **這是快照，不是權威副本。**
> 權威位置：`.workbuddy/memory/MEMORY_PERF.md`（由 host 持續寫入）。
> 本檔於 2026-09-18 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
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
- **M1 解耦 pool 與圖：五個工作項實作了四個（第 1 個判 EXPERIMENTAL 不可用），但★ 五條離開條件
  沒有一條被系統性重核 ⇒ 現在的瓶頸是「驗收」不是「實作」。**
  （★ 09-17 18:2x 改寫：先前寫「做了一半以上、卡住」，那句在 `605452177` 之後已經不準 ——
  工作項 2/3/4 都已實作並提交，卡住的是離開條件，細節見下面 §離開條件的更正。）
  **★ 09-17 15:45 狀態更新：別條線已經開始做工作項 4（canonical gather order）** —— 新增
  `src/llama.cpp/src/llama-cgc-canon.h` 與 `scripts/check/canon_order_selftest.cpp`，並在
  `scripts/run_server.sh` 的 allowlist 加了 **`CGC_CANON_ORDER`**（`=1` 依 expert id 排序、
  **`=2` 是 identity 對照**，用來把「重排改了數字」與「多出來的節點改了數字」分開）。
  他們自己在註解裡寫明：**兩個模式都會改變模型輸出，所以都不能當品質或 D5 證據**
  （對某個在別的 mode dump 的參考）。⇒ 這正是交接白皮書 §4 建議的起手式，**已接走**；
  本線不碰 `src/`。
  **★ 09-17 17:0x：工作項 2＋3 也實作並驗證了**（`docs/M1_WORKITEM2_PHASE_SPLIT_STATUS_2026-09-17.md`）。
  可引用的**持久事實**（這句話是判準，不是進度）：
  - 相位判定式在 `src/llama-cgc-phase.h`，**兩側共用**：
    `cgc_decode_width(slots, top_k, cap) = min(cgc_decode_bound(slots, top_k, cap), T_prefill-1)`，
    `cgc_select_graph_phase(n_tokens, w) = n_tokens <= w ? DECODE : PREFILL`。
    真實幾何下 `bound = floor(routable_slots / top_k) = floor(142/8) = 17`。
  - ⇒ **而這有一個直接後果：decode（T=1；MTP 也只 2–4）恆 ≤ 17 ⇒ 恆走 DECODE
    ⇒ whole-layer slab 永不參與 decode。**
    **slab 是 prefill-only**（`run_server.sh:1618-1620` 原文：*Decode stays on the pool path*）。
    ⇒ **任何「armed slab vs pool」的 decode A/B 是 by construction 的零差異**，
    那個零**不可以**被讀成「slab 沒有收益」。要量就量 prefill。
  - **未 arm 時 prefill 同時吃 `n_batch` clamp**（`[arm] OFF` 行自證）⇒
    「armed vs unarmed」在寬 chunk 上是**兩個變數**（slab＋clamp），不是單變數。
  - `cap` 已**降級**成 decode 圖的寬度上限（不再是相位開關）：`cap=8 → width 8`、
    `cap=64 → width 17`（被 bound 夾住，不是 64）。
  - gate：`Backup/m123_oracle_gate/summary_phase_p2.json`（17:00:27）`comparable=True`、9/9/9；
    真實 cross-tab ＝ `{eq_eq:9, eq_ne:0, ne_eq:0, ne_ne:0}`。
  - ⚠ 同批的 `r52_canon1_pool4gb`／`r53_base_pool4gb` 是 **`comparable=False` 而 M1/M2/M3 仍印 9/9**
    ⇒ 又一個「先讀 comparable 再讀判決」的活例（他們沒引用它們，處置正確）。
  數值那一半 **09-14 已達成**：2/4/6/8/10 GiB 全部 **M1 = M2 = M3 = 117/117**，2 GiB 是唯一
  `union > slots` 那格（compacted gather），`union-routable` PASS，RSS 達標。
  **⚠ 證據地位（09-17 更正）**：這個 117/117 是**跨池不變性**，**不蘊含正確性** ——
  別條線的 r33 nb-aware 修正（`docs/ROUTING_TRACE_2026-09-17.md` §12.1／§12.4）證明
  **每一臂都把 T≥2 的 token 路由到 token 0 的專家**，而「每一臂都犯同一個錯」正好讓跨池
  不變性通過。該檔原文：*cross-pool invariance (M1/M2) passed while the ids were wrong for
  all of them*。⇒ M1 的數值半應改述為「**不變性**已達成」；要主張「數對」必須用 r33 之後的
  參考（v6_nbaware）。**兩條沒過**：
  ① `decode 不得退步` ⇒ 2 GiB（gather＋slab）**6.36** vs 8 GiB（pool）**8.87 t/s** ＝ **0.72×**；
  **⚠ 這條的基準要重定（09-17 15:0x）**：8.87 是 09-14 的 build，而今天同類量測（Nail、8 GiB、MTP off）
  是 **9.82**（`ROUTING_TRACE` §14.2），且 r33 修正改動了 routing（hit 57.1→72.0）⇒ **0.72× 這個比值
  跨了 build，不能直接沿用**；要重跑 M1 的 2 GiB 格對**當天**的 8 GiB 格比。
  ② `prefill chunk 2048` ⇒ 當初缺工作項 2。**工作項 1**（`CGC_POOL_SPLIT=1`，保持 expert tensor 全寬）
  實作了但被判 **EXPERIMENTAL, NOT USABLE**：Blocker A（寬 tensor 讓 gather 把 Metal buffer 的指標
  重指到 Metal 不知道的 host 指標 ⇒ **靜默** `tensor buffer is nil`、**M1 2/42**；且 16 GB 上
  warmup OOM），Blocker B 已於 **09-16** 修好。
  **★ 09-17 18:2x 更正（本行先前寫「工作項 2 未實作」，那句已經過時且與上面 17:0x 那段矛盾）**：
  **工作項 2 與 3 已實作並提交** —— `605452177 feat(moe): decide the build graph from the request
  phase, and demote cap to a width ceiling`（`src/llama-cgc-phase.h`：`cgc_decode_width()` ＋
  `cgc_select_graph_phase()`，兩側共用）。⇒ ② 的處置從「缺實作」變成「**要重測**」。
  **★★ 而五條離開條件沒有一條被系統性重核（這是 M1 現在真正的缺口）**：
  - **「M1/M2 @ 4/6/8/10 GiB = 100%（含 `union > slots`）」** —— 今天 **75 筆 D5 裡
    `comparable=True` 的全部是 8 GiB（46 筆）**；4 GiB 只有 `r52_canon1_pool4gb`／`r53_base_pool4gb`
    **兩筆，且都是 `comparable=False`**（池預算 4 GiB ⇒ `.cap` 的 `CGCENV.BUDGET` 與 8 GiB 參考不符
    ⇒ 不可比）。⇒ **phase split 之後，多池尺寸那條從未被重新確立**；要它就得像 09-14 那樣
    **每個預算各建一份參考**（那次是 117/117）。
  - **`prefill chunk 2048`** —— 他們的 §A 只證了 35-token 與 12-token 兩個**單點**（舊述詞會誤路由
    的那格），**不是 chunk 2048**。
  - **`RSS @ 10 GiB` ±0.5 GiB vs 9.08** —— **沒有任何一筆讀數**。
  - **`union-routable` PASS** —— 09-14 的結果，**未在 phase split 之後重跑**。
  - **`decode 不得退步`** —— 見 ①，基準本身要重定。
- **M2 prefill 整層串流：核心機制已落地**（`CGC_PREFILL_STREAM=1` ＋ `CGC_GATHER_SLAB_CAP=256`，
  `prefill250` 用它跑到 250+）。**★ 09-17 17:0x 五條離開條件已逐條審計（本線，`docs/M2_EXIT_CONDITIONS_AUDIT_2026-09-17.md`）
  ⇒ 3 PASS / 0 FAIL / 2 未裁決**：條件 1（bytes/token @ chunk 2048 ≤3.0 MB）PASS（靠引擎自印的
  `slab fills` 非駐留 44.1%，`11.92 GB × 0.441 ÷ 2048 = 2.57 MB/token`）；條件 4（prefill t/s 誠實記錄）
  PASS **但附否證**（15:37 的 `COLD-STATE`、安靜 1940 s 那臂反而只有 242.42／227.27／249.06，
  見 lesson `eng-mh-0054`）；條件 5 PASS（D5 9/9）。**條件 2（裝置持續 ≥1.0 GB/s）與 3
  （wall < I/O + compute）＝ 未裁決，不是不合格** —— `grep slab_*us/ms/time` 在 `src/` 裡**零命中**
  ⇒ 引擎沒有 slab 計時計數器，這兩條**沒有儀器**。儀器設計寫在該文件裡，**且都不需要動 `src/`**。
- **M3 decode 的 compute 削減：判定書已出（尚未實作；依賴 M1）**。
  （★ 09-17 18:2x 改寫：先前寫「未開始」，但當天已產出 `docs/M3_VERDICT_2026-09-17.md` 並**建了它缺的儀器**。）
  兩個探針試過且**不可引用**：`CGC_MMV_FUSE`
  （MoE gather 融合）輸出損壞且更慢；`CGC_SUBMIT_AHEAD`（序列化）天花板 ×1.711 但輸出損壞。
  離開條件「decode（MTP off）≥ 15 t/s」未達。
  **★ 09-17 17:1x 判定（`docs/M3_VERDICT_2026-09-17.md`）：它與 D3 的 S2／S3 是同一件事。**
  今天 10.78 t/s ＝ 92.8 ms/token。兩個目標要分開：**離開條件 15 t/s 只要 1.39×，
  目標 100→40 ms 要 2.32×**。判決：
  ① **host 側正式關閉** —— `CGC-PHASE` 實測 `build 0.023 + alloc 0.037 + inputs 0.026 = 0.086 ms/步`
     ＝ **0.07%**，而 `compute=118.459`（**99.9%**）、`fill_wait=0.000`（IO 不在關鍵路徑）
     ⇒「減少圖重建／配置／IO」三條路**沒有量**（三個數量級差 ⇒ 對熱不敏感）；
  ② **去序列化的空間存在**（`SUBMIT_AHEAD` 實測 82.5→31–44 ms ⇒ 38–51 ms/步，
     足以讓 15 t/s 擦線過），**但步級三分（cb 8–10% + submit 4% + gap 19–36%）只能解釋 11–29 ms**
     ⇒ **差額 9–22 ms 住在 `wait` 的儀器盲區裡**；
  ③ **M3 缺的不是路，是儀器** —— 兩個既有儀器都**拆不到 T=1 的層內**：
     `CGC_PHASE_TIMING` 把整步歸成 `compute`（`compute == gpu`，是副本不是 GPU 鐘）；
     `CGC_VERIFY_OP_TIMING` 的目標判定含 `node->src[2]->ne[1] > 1`（`llama-context.cpp:3830`）
     ⇒ **只在 T>1（verify）生效**，對 MTP off 產生不了輸出 ⇒
     `debug-verify-path-breakdown.md` 那組 gate/up/down（0.8/0.8/1.0 ms）**不能移植到 T=1**。
     唯一活路是**逐節點 GPU 時間**（Metal counter sample buffer），成本最高。
  ⇒ **判準（先寫死）**：若逐節點 GPU 時間顯示 `ffn_moe_*` 佔 `wait` ≥40% ⇒ Cell 2 有量；
  < 15% ⇒ 三條路全不足，走「找不到路」分支。**沒有判準就不要實作**（`CGC_MMV_FUSE` 的學費）。
- **M4 MTP 拒絕取樣 ＋ verify 真批次：核心數字已達成（09-17 14:2x，別條線），但離開條件之一被證明是錯的判準。**
  同 binary A/B（build 14:12，carrier `nail`，pool 8 GiB，`n_predict=96`，每臂 3 請求）：
  **MTP off 9.82 t/s｜MTP on 修前（`CGC_IDS_LINEAR_READ=1`）8.62 t/s（accept 73.81%）｜
  MTP on nb-aware 修正 12.62 t/s（accept 58.25%）** ⇒ **MTP-on ≥ MTP-off 達成（+28.5%）**，
  修前是淨負。★ **accept 下降而吞吐上升 46%**：修前 verify 批次的 token≥1 用別的 token 的專家算
  ⇒「accepted」是拿錯分布比出來的、而且**被抬高**。⇒ **「accept ≥ 60%」這個離開條件要作廢**
  （改成「MTP-on ≥ MTP-off ＋ M1/M2 = 100% with MTP ON」），並且**修前記錄的每個 accept 數字
  （含 09-14 的 70.4%、roadmap 的 19.9%）都屬於被抬高的 regime**。全文 `ROUTING_TRACE` §14。
  **⇒ 今天補上的正好是 09-14 就指名的那唯一缺口**：`docs/MTP_HEAD_PROVENANCE_GATE_2026-09-14.md` §5
  早已寫「Nail 配對已達 70.39%，超過 M4 的 60% 離開條件；**該配對仍未過的是 `MTP-on ≥ MTP-off`
  （6.74 < 8.87）**」⇒ 今天 12.62 ≥ 9.82 ⇒ **M4 的功能缺口關閉**。仍未處理：`plain_match=False`
  （batch verify 與逐 token 解碼不一致；同一份文件 §5 第 2 點）與「M1/M2 = 100% with MTP ON」。
- **M5 prerouter 只當預取提示：未開始**（可選、期望值低）。**`PREFETCH_ONLY` 在本 repo 0 筆。**
- **M6 換量化幾何：頭條目標已達成，剩下的只有「一個綁定層」**（09-17 12:1x 實測；
  全文 `docs/M6_QUANT_GEOMETRY_PLAN_2026-09-17.md`）。
  **★ roadmap 的 1.769→1.122 MB 與「6 GB pool 85→133 slots」指的是 Edge0-int4，不是出貨模型。**
  出貨的 `Nail-…-denseIQ4X.gguf`（12.72 GiB）實測：41 層 × 256 experts，**典型層 274 MiB
  ＝1.0703 MiB/expert（＝roadmap 的目標值 1.122 MB）**，但 **blk.39 是 356 MiB＝1.3906 MiB
  （gate/up 是 IQ3_S、down 是 IQ4_XS，其餘 38 層是 IQ2_S/IQ3_S）**，而
  `capacity = clamp(budget/(41×per_slot), 8, 256)` 取 **MAX over TRUNK layers**（NextN/MTP 被跳過，
  但 `denom` 仍數它 —— `llama-model-loader.cpp:1171-1174`）⇒ **一層厚、全班付錢**。
  現行 `BUDGET_DEFAULT = 10 GiB` ⇒ **179 slots**；`run_server.sh:411-419` 自記 **143 slots 時
  hit 90.8%**（counterfactual K=96 79.7／128 87.7／192 97.1／256 100）⇒ **hit 早已超過 roadmap 的 84%**。
  **★★ 決定（09-17 13:0x，使用者裁定）：只走無損的設定路線，不動模型。**
  **★★ 裁定（09-17 15:0x，閘門 4 之後）：設定路線「不可交付」，不是「機制被否證」。**
  同負載實測（`docs/M6_…md` §4.9）：均勻 143 的 **capacity 佔 miss 的 45.6%**，但 **miss 只佔抓取的
  3.16%** ⇒ capacity ≈ **1.4% 的抓取**；而 `--slack 256MiB`（薄層 149）**消不掉它** —— 引擎自己報
  **最差層 layer 2 的 distinct ＝ 239 > 149**、且 **9 層 distinct > slots**。
  ⇒ **代價不是 +238 MiB，而是 cap ≈ 240–256（+2 GiB 等級）**，而吃緊的形狀連 +238 MiB 都已在
  MTP-on 上**請求即 GPU OOM**（§4.8 ③）。上限反推：這個負載（hit 96.8%）只有 **≤ +0.7%**；
  用生產負載的 hit（89.2%）反推是 **≤ +2.5%**。
  ⇒ **機制存在、上限存在、代價落在買不起的一側**；而且別條線的 r33 修正已把 hit 由 57.1% 抬到
  **72.0%（與池無關）** ⇒ 一部分本來要 M6 買的東西已被「修路由」買走。
  **能復活的三條路**：① 加預算（> +2 GiB）；② **改淘汰／填充策略**（不靠更多 slot 提高複用 —— 唯一
  不需預算的路）；③ 改目標（少讀本來是 M2 的地盤）。另：M6 的**本體**（per-expert 1.769→目標 1.122 MB）
  **早已在出貨檔裡**（實測 1.0703 MiB）⇒ 頭條價值已兌現、未被推翻。
  **設定路線＝一條既有 env 字串，零程式改動、零重建、零 D5**：`LLAMA_EXPERT_CACHE_LAYER_CAPS`
  是**雙邊讀取**的（loader 的 `ne[2]=cgc_layer_cap(il,cap)` @`llama-model-loader.cpp:1492/1502`
  ＋ cache 的 `n_slots_l` @`llama-expert-cache.cpp:3136-3142`），而 **`run_server.sh:1804-1807`
  每次啟動都寫它**（預設 `40-40:256`）⇒ 覆蓋只要 `CGC_SERVER_LAYER_CAPS`。
  10 GiB 下的字串：`0-33:232;34-34:212;35-37:232;38-38:212;39-39:179;40-40:256`
  ⇒ **37 個典型層 179→232（+29.6%）**、池由 **7.85 GiB（78.5%）→ 9.97 GiB（99.7%）**
  （今天有 **2.15 GiB 預算買不到任何 slot**）。6 GiB＝`0-33:136;34-34:125;35-37:136;38-38:125;39-39:107;40-40:256`、
  8 GiB 把 136/125/107 換成 184/168/143。
  產生器：`python3 scripts/check/gguf_pool_geometry.py --layer-caps`（規則＝每層等位元組，
  **夾在今天的 uniform 值之上** —— 太少 slots 會**靜默讀到空**：Edge0 的 33 slots 是 585 次
  `buffer is nil`、48 題 0/48）。
  **★ 逐格驗證（引擎自己就印普查行 ⇒ 驗證是一行 grep）**：`LAYER_CAPS per-layer caps: total N slots
  (avg A/layer, min M/layer)`，`N = 40×base + 256` ⇒ Nail 8 GiB `5976/145.8/143`、10 GiB `7416/180.9/179`、
  **Ornith 2 GiB `1416/34.5/29`、4 GiB `2616/63.8/59`** —— 工具與引擎**逐格相同**；
  `resident` 亦相符（179→7990.71 MiB、143→6430.6 MiB）。
  **★ 仍未量測**：逐層變動的 trunk caps **從未跑過**（用過的只有 `1-39:32` 與 `40-40:256`）
  ⇒ 四道依序：普查行 → 48 題（基準 **9/48**）→ `m123_oracle_gate`（換池幾何**不該**動 logits，
  M1/M2/M3 應 9/9）→ 才談 hit%／t/s。而 **hit% 在這個區間由負載主導**（真實 log：179→87.9%（18.9 萬請求）、
  143→86.7%、71→44.9–82.8%）⇒ **+29% slots 的收益要量不要推**。
  **MTP 的 cap 與預算無關**：`40-40:256` 固定 256 個 slot，Ornith（Q8_0 頭）在 2 GiB 下它吃掉
  0.80 GiB＝40% ⇒ 今天的形狀在 2 GiB **已超支 118.5%**（要調就用 `--mtp-cap`）。
  **★ 模型路線（拉平 blk.39）已否決**：`gen_denseiq4x_tt.py` 的政策是「**每個非 dense 張量釘住它現有的
  型別 ⇒ byte-copy**」⇒ **expert 型別是從上游 `UD-IQ3_XXS`（UD＝Unsloth Dynamic）逐位元組繼承的**，
  厚的那幾層是**逐張量重要性校準的結果**；拉平＝**有方向的降精度**（gate/up −25.5%、down −19.1%），
  而且只換到 +19%（單層）／+30%（4 層）—— **不比設定路線多，卻要付精度代價**。
  （模型工具與記錄仍在：`scripts/gguf_retensor.py` set-type／restore／verify／digest，dry run 是預設；
  `--normalize-binding` 的那三行情境；**`verify`／`set-type` 必須用 `/opt/homebrew/bin/python3`**，
  系統與 managed python 都沒有 numpy ⇒ gguf-py 對帳 SKIPPED ⇒ 直接拒收。
  roadmap 指名的 `scripts/verify_edge0_gguf.py` **在本 repo 不存在**。）
  **幾何 census ＝ `scripts/check/gguf_pool_geometry.py`**（09-17 建；純讀標頭、不需 numpy、
  可在別人量測跑著時執行）；`analyze_pool_geometry.py` 仍留著但那是為 Edge0 寫的、需要 numpy。
- **★ D3（`REMAP_ROUNDTRIP_REMOVAL_PLAN` 的主設計：把 expert→slot 查表搬到 GPU、消除段邊界）
  不是里程碑，是 S1→S2→S3 階梯；現況（09-17 03:3x 查證）：沒跑通。**
  **S1**（`CGC_SLOT_TABLE_GPU=1`，leaf 改由 GPU 算、**段數不變**）**已實作且會跑，但不過
  bit-identical**（錨 `2d4e5099` vs S1 `f0acf7d2`）；分歧＝**第一個被 GPU table 服務的層的 MoE gather**
  （層號由 `CGC_S1_MIN_IL` 決定）。**S2（段邊界去等待）／S3（收段 `n_segs 40→~1`，＝那 ~44%）
  從未開始**，且 **S3 的前提 B 已倒**（47.5% 的步會動被消費的映射 ⇒ 發布不冗餘）。
  同階梯的 D0 只是量具（輸出損壞）、D1／D2／D2' 判死。⇒ **D3 無對外數字，且仍依賴卡住的 M1**。
  S1 的細節全部在 `MEMORY_S1.md`。
- **★ S1 的「池內容是不是載體」＝ 09-17 09:1x **已用裝置側讀數結清：不是**。**
  兩次 A/B（探針 4096 B 與整列 `nb02`）都得到 `SAME=90 DIFF=15 NOT-READ=0`，**分歧處一律是 ids 不同**；
  兩臂選到同一組 slot 的**每一個** graph（含第一個 T=1 步 g30）上，那些列的**整列位元組逐位元組相同**。
  ⇒ 「同 ids、同池佈局、不同位元組」**被否證**，下一步要查的是**層 0 的 delta-net 遞迴路徑**（g30 就 DIFF）。
  儀器與全文：`docs/POOL_ROW_DIGEST_20260917_0355.html`。
  **★ 09-17 10:2x 修正（重要、會改「下一步」）**：那個「ids 相同」只涵蓋**每個 chunk 的 token 0**
  （`rows=8`；而 T=2 的運算元有 16 個、T=8 有 64 個）。`POOLROWS=12` 一跑就顯示**真正的第一分歧在 g1**：
  S1 臂對「同一 chunk 的第 2 個 token」給出**錯誤的 slot**（含重複 slot 0），而 token 0 的 ids、
  router logits、routing weights、MoE 輸入**全部逐位元組相同**   ⇒ **是 mapping 缺陷，不是 residency**。
  **★ 11:2x 定論（`own=` 欄把這一軸結案）**：同臂控制 1200/1200 SAME 先過，A/B 得
  **`CONTENT=0`、`SLOT-REUSED=0`、`RELAYOUT=0`，只有 `ROUTING=477`**
  ⇒ **池的位元組／residency 這一軸結案**；載體是「**拿到的專家不同**」，位置在 **gather 上游**。
  而 `exp=`（host 認為該讀哪個 slot，§9.18.8）**已建置但實跑回 `exp=none`**（來源在 S1 臂不存在）
  ⇒ 來源要換成 host 現算 `st[e_j]`，**尚未量測**；全文 `docs/S1_OWNER_EXPECT_CHANNELS_20260917_1145.html`。
  細節與下一步在 `MEMORY_S1.md`。
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
| **★ 唯一的 instrument of record：llama-bench 暖平台**（丟 rep1） | **10.78／10.91／10.98**（d512，三次獨立量測）；11.3（n=128 平台） | `prefill250+SPAC=1`、`-b512`、NOMINAL、**Nail denseIQ4X**（見下更正）。10.98 是 09-17 18:04 在**當前 build**（`libggml-base 23f533ad`、per-layer 儀器之後）量的，NOMINAL 105/105 ⇒ **儀器 inert 的量測確認（不只 D5）** |
| llama-bench `-d 0`（冷格，**不是標準格**） | 9.52–9.79 | 同上；`--depths 0` 是最冷的格子 |
| ~~decode_bench（HTTP）n=128~~ | ~~12.36（NOMINAL）→ 10.64（HEAVY）~~ | **★ 2026-09-17 使用者裁定：`decode_bench` 退休，不要再量、不要再引用** |
| llama-server（`p25-gputime` 等 HTTP 臂） | 12.95 可持續（**僅歸因用**） | **不再是 headline 口徑**，角色只剩逐步分解／歸因；`20.3` 一律不得引用（n=24 短爆） |
| **「25 t/s」的出處** | 09-05 的 **27.71** | **71 slots／4 GiB pool ＋ L0=32/L1=32 分區**，`draft accept 0.9974`（3.99 token/step、144 ms/step）——**不是**現在的 143 slots／8 GiB |

- **★★ 09-17 裁定：decode 統一用 `llama-bench`。** 標準形狀＝
  `llama_bench_matrix.py --prompt 0 --gen 128 --depths 512 --reps 3`（warmup ON、**丟 rep1**）；
  報數字一律附 **reps 數 ／ warmup 規則 ／ 模型家族**（`MTP=0` 會換檔）。
  ⇒ **可引用的 decode ＝ 10.8–10.9 t/s；距 25 約 2.3×（不是 2.0×）。**
  退休理由：兩器同 env／同模型／同 n／交錯差 9.4 vs 12.4（n≈128）、7.2 vs 18.7（n=24），
  且 `decode_bench` 離散大得多（12.36／12.95／10.64 vs llama-bench 三次 **1.1%**）。
  **⚠ 未決：MTP 不在這個口徑裡** —— 歷史的 `12.62 vs 9.82` 是 HTTP／`llama-speculative-simple`
  量的；兩條路＝把 `--spec-*` 加進 llama-bench（三處、動 `src/`）或保留 spec-simple 但永不並排。
- **★★ 09-17 更正（`MTP=0 模型` 那條標註是錯的）**：現行 harness 下
  **`--arm prefill250` 與 `--arm prod25` 都載 `Nail-…-denseIQ4X.gguf`**；
  **只有顯式 `prod25:<…>;CGC_SERVER_MTP=0`（臂 `prod25-stream-mtpoff`）才換成
  `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`**（16:1x 用 `prefill_certifiability.py --dry-run`
  逐字讀它印出的 `-m` 行驗證）。`llama_bench_matrix.py` 的 `ARMS` 表裡**只有
  `prod25-stream-mtpoff` 設了 `CGC_SERVER_MTP`**。
  ⇒ 歷史的 **10.78／10.91 與 `prod25-stream` 的 10.79／10.89 都在 Nail denseIQ4X 上**，
  與 prefill 的 `pp2048` cell **同一個檔** ⇒ **pp/tg 這一對是自洽的**。
  ⚠ 但 **`tag` 不含模型檔**（json 只存 tag）⇒ **每一份 llama-bench 產出都要記 `-m` 那一行**
  （matrix 已經會印），否則事後無法分辨。
- **★★ 09-17 (A) 裁定：prefill 也統一 llama-bench，且要兩個 cell**（使用者追加「這個也要加上去」）：
  - **cell 1（自家形狀）**：`prefill_certifiability.py --arm prefill250 --prompt 2048 --gen 16
    --reps 3 --runs 5` ⇒ 形狀 `-p 2048 -n 16 -d 0 -b 5632 [profile]`（dry-run 驗證過）。
  - **cell 2（上游可比）**：`… --prompt 512 --gen 128 --depths 0 --reps 3 --runs 5 --batch 2048`
    ⇒ 形狀 `-p 512 -n 128 -d 0 -b 2048 [cli]`（＝上游預設 `-p 512 -n 128 -d 0 -b 2048`，
    `llama-bench.cpp:367-377`；上游標準輸出列 `pp512/tg128/pp512@d512`，`README.md:180-187`）。
  - **兩個 cell 必須是兩次獨立 run**（同一行程內第二格繼承暖池 ⇒ 不獨立）。
  - **前置條件已補**：`prefill_certifiability.py` 原本**沒有** `--batch` 透傳（固定吃 profile 的
    5632 ⇒ `pp512` 會變成「-b 5632 的 pp512」＝同名不同量）⇒ 已加 `--batch/--ubatch/--dry-run`。

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

- **★★ 09-17 盤點：prefill 有兩個入口，而「哪一個是交付口徑」未定**（decode 已於同日統一在
  llama-bench）：
  ① **HTTP 驗收臂** `run_req2_retest.sh`（**2873-token prompt** 的 `prompt eval t/s`，
  每請求邊界帶熱讀數）—— 本節 §1–§3 的交付規程是為它寫的，白皮書的 278.56／261.17／275.01 出自它；
  ② **llama-bench `-p 2048`** `prefill_certifiability.py --arm prefill250 --prompt 2048 --gen 16
  --reps 3 --runs 5`（走**同一支** `llama_bench_matrix.py`；`-b/-ub` 由 dump 取 ＝ 5632）——
  已記錄 276.25／300.43（Nominal ×2）與 227.27／151.28／145.79／182.39（非 Nominal）。
  ⚠️ **兩邊數字很近但不可並排**（`-p 2048` vs 2873-token、歷史 `-b` 6144 vs 5632）。
  ⚠️ **llama-bench 那條是抽籤不是規格**：`pp2048 @ -ub 6144` 四次獨立啟動
  276.59／198.84／176.18／122.68 ＝ **2.25× 離散**（啟動內部只差 2.08–15.41）
  ⇒ 統一用 llama-bench 對 prefill **不是免費的**：把「熱條件不足」換成「跨啟動離散」，後者未解。

- **權威儀器（非 root、11 ms）**：`notifyutil -g com.apple.system.thermalpressurelevel`
  （`0=Nominal 1=Moderate 2=Heavy 3=Trapping 4=Sleeping`）。**判準：發射前讀到 `0`。** 分離度（request
  級、零重疊）：發射 0 → **6/6 ≥250**；發射 1 或 2 → **0/21**。反面教材：`NSProcessInfo.thermalState`
  367/367 讀 `fair`、零區辨力——**找到介面 ≠ 找到儀器**。
- **★★ 09-17 15:37 更正：上面的判準是「必要非充分」—— `COLD-STATE`（安靜 1940 s）＋ 發射前與每個
  請求邊界都讀到 `0/NOMINAL`，仍然只量到 242.42／227.27／249.06（三個都 <250）。**
  同 build／同 profile／同 2873-token prompt；2 Hz 序列顯示 `15:37:15→15:38:24` 連續 Nominal，
  **req1／req2 完全落在那段之內** ⇒ 熱解釋不了。反向對照：同日 15:03 標籤只是 `UNKNOWN-STATE`
  （無 state file）卻得 **278.56／261.17／275.01** ⇒ **安靜秒數與熱等級都不預測這個 t/s**。
  池路徑已排掉（兩臂 `pread_usec` 1501 vs 1548 s、`us/job` 31733 vs 32533，都在 1.26 MiB/s 線上）。
  ⇒ 標籤只能背書「**讀數的來歷**」，**不能**背書「≥250 這個門檻」。（lesson `eng-mh-0054`；
  白皮書 §2.3／§2.4；skill 已同步修正）
- **「距上次持續 prefill ≥150 s」作為充分條件已被推翻**（安靜 18 s → 289.86；34 s → 166.95）⇒ 自變數是
  **累積負載（同序列第幾次啟動）**；機制是 DVFS 階（1470→928→618 MHz）與熱壓同秒。
- **可交付述句（09-17 修正）**：可引用的是「**該臂的 req1–req3 讀數是 X／Y／Z**」＋ 熱標籤；
  ❌ 舊述句「發射時讀到 `0` 的那一臂，req1–req3 全部 ≥250」**已被否證**（見上）。
  **不能**說「250 隨時可重現」。熱態平台 167–201。`t(token) = 0.5575 ms + 4226/f_eff(MHz)`；
  純階反讀 1470→291、928→227、618→135。**250 不是上限**；`swap` **不是因是果**。
  **直接變數是有效時脈，熱等級只是它的粗代理**（本次 4.13 ms/tok 反推 ≈1130 MHz vs 對照臂 3.59 ≈1250）
  ⇒ 裁決「為何 COLD 仍 <250」得用 `powermetrics` 的 GPU 時脈駐留（**需 root**，且沒試過）；
  未排掉的第一候選是**背景 GPU 客戶端**（`Freebuff Helper (GPU)`／`WorkBuddy Helper (GPU)`／
  `WebKit GPU`／`WindowServer`；發射時 load 1 分鐘 2.42–2.93，同日 15:39 量到 6.18）。
- **閘門**：`COLD-STATE` 需安靜 ≥1800 s，否則 `HOT-STATE`；**HOT 不得當交付數字**。**機器地板是 HEAVY**
  （零 llama 行程時 thermal=2、GPU util 20%、Electron ~103%）⇒ agent UI 造成。
  `ARMS=2 bash Backup/run_thermal_gate.sh`（不成立 exit 3，fail closed）；
  `docs/PREFILL250_CONDITIONAL_DELIVERY_20260916.html`。
- **已結清（09-17 07:0x）：`prefill250 + CGC_SPAC=1` 的 prefill 代價不成立。** 同 build（server
  `054fb22f`／`libggml-metal 2b87af2e`）COLD 交錯 off/on/off/on，四臂全 `COLD-STATE`／`survived=yes`：
  req1 off {215.30, 287.80} median **251.55** vs on {283.36, 284.82} median **284.09**；req2 268.51 vs
  285.95；req3 291.03 vs 295.11。**唯一的大落差（req1 配對 +68.06／−2.98 符號相反）是序列位置造成的**：
  同一個 off 配置在位置 1 與 3 差 **72.50**，而兩臂的池讀 bytes **完全相同**（2 784 305 152，`us/job` 只差 3%）
  ⇒ 第一次啟動的代價在池之外（page cache／Metal 暖機）＝機器的狀態，不是 flag。池：off 永遠 `2604/0`
  （100% compulsory）、on 永遠 `2502/5`；SPAC=on 少讀 3.9% bytes，**不買也不付** prefill。
  **20:54 那臂不可搬用**：它載入的 `libllama 233a172`／`libggml-metal ec3ece90` 與今天不同 ⇒ 不同 build，
  且它的 req1 244.96 比今天的 off@pos1 215.30 **還快**。
  產物 `Backup/cgc_logs/spac_cold_ab/RESULT.md`。**⚠️ 這條數字只在同 build 內可比；且 ABAB 的
  第一個臂不可與後面的臂交換**（pos1 溢價 72.50）。下一個能裁決「反向增益」的設計＝丟棄臂開頭 ＋
  鏡射後半（`off,on,off | on,off,on`）。

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
