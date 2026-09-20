---
name: cgc-decode-attribution
description: 在 flashkv-devserver（TurboFieldfare / llama.cpp CGC fork）上把一個 decode 步歸因到 GPU 執行、CPU 序列化、池/IO 或編碼，並用既有儀器取得可證的判準數字。當使用者問「decode 為什麼慢」「wait 是什麼」「瓶頸在哪」「要不要做 fusion / kernel 優化」或要求 decode 相位分解、GPU 佔用率、速度槓桿排序時使用。
agent_created: true
---

> **這是快照，不是權威副本。**
> 權威位置：`~/.workbuddy/skills/cgc-decode-attribution/SKILL.md`（由 host 持續寫入）。
> 本檔於 2026-09-20 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
> 底下的內容能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 要改 skill 請改原檔，再重跑 `python3 agent_harness/scripts/import_harness_snapshot.py`。

# CGC decode 步歸因（儀器與陷阱）

專案：`/Users/alexchuang/Documents/flashkv-devserver`
目標配置：prod25 / prefill250，Gemma 4 26B-A4B，M4 Air 16GB。

## 鐵律

1. **~~每次跑之前 `pkill -9 -f llama-server`~~，並用 ABBA 配對 + md5 對比。**

   ⚠️ **2026-09-18 作廢前半句（`pkill -9 -f llama-server`）**：同一台機器上有多條 session 同時量測，
   那行會**把別人正在跑的量測一起殺掉**（看 port 不看 pid，殺了也不知道殺的是誰）。同一家族的缺陷
   還是當天兩次 server 離奇死亡的**真因**：`run_server.sh` 的 preflight 用 `pgrep -f` ＋ 固定 binary
   名清單、**不看 port／不看 session**，而且**排在 memory guard 之前** ⇒ 別人起一次 `run_server.sh`
   就把你的 llama SIGTERM 掉，就算他自己隨後被 guard 擋下，你這輪也已經死了。
   **取代作法**：
   - 先問「是誰在用」：`lsof -nP -iTCP:${PORT} -sTCP:LISTEN -t`、`pgrep -fl 'llama-server|http_duo|profile_duo|run_server.sh'`
     （⚠ **只 pgrep llama 不夠**：09-18 的競爭者 argv 是 `http_duo.py`，字串裡沒有 llama）；
   - 要清就**按 pid 清自己的**：`kill -TERM <pid>`（讓 Metal buffer 正常釋放，別一開始就 -9）；
   - `run_server.sh` 的 preflight 自 09-18 起**預設不再送任何訊號**（只列 `pid/etime/command`），
     要清場必須明示 `CGC_PREFLIGHT_KILL=all`；
   - 自己的工具最好**自選 port**（`http_duo.py --port auto`）＋ `start_new_session=True`，
     清理只碰自己的 pid/port。
   - 题外定義「server 死了」：**看它自己的 log 有沒有 `[CGC] Received SIGTERM`**
     （watchdog 走 `GGML_ABORT`、OOM 是另一回事）；中了就把那臂判成 **無效樣本**，不要當數字用。

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

   ★★ **2026-09-19：`CGC_SERVER_MTP=0` 不能當「無投機對照臂」——它是 7+ 個 env 的整塊開關。**
   `run_server.sh:2020-2094` 整個 env 區塊被 `if [ "$SERVER_MTP" = "1" ]` 包住，MTP=0 時全消失：
   `CGC_NO_PREFETCH`／`CGC_VERIFY_DECODE`／`CGC_DRAFT_DECODE`／`CGC_WARM_NPAST`／`CGC_MTP_NO_WARMUP`／
   **`CGC_MM_BITIDENT=1`**／`CGC_NO_SEQ_RM_PROBE`；`LLAMA_EXPERT_CACHE_LAYER_CAPS` 也在 `:2099-2104`
   對 MTP=1 給 `40-40:256`、MTP=0 什麼都不給。
   - **`CGC_MM_BITIDENT` 是 bit-identical pillar 1**：把 **M≤8** 的 matmul 釘在 M-invariant `mul_mv`
     路徑（`:2048-2052`），而 **decode 的 GEMV 是 M=1，正在範圍內** ⇒ 兩臂的 decode 走**不同 kernel**。
   - 它**只在那個 MTP=1 區塊裡讀**（`:2053`）⇒ **MTP=0 時設了也被靜默丟棄**，無法從外部補齊。
   ⇒ **任何「MTP on vs off」的逐值／輸出比對都不是單變量**：輸出一旦不同，無法區分
     「verify 路徑有缺陷」與「decode GEMV 換了 kernel、rounding 不同」。
   ⇒ **無投機對照臂＝ `CGC_SERVER_MTP=1` ＋ `CGC_SERVER_MTP_N_MAX=0`**（`:420`→`:1218` 轉發成
     `--spec-draft-n-max`）⇒ 同 env set、同 carrier、同 launcher 分支。
     ⚠ **2026-09-19 18:0x 實測推翻「k=0 合法」這一句**：`--spec-draft-n-max 0` 在**首次 decode**
     就 abort —— `llama-context.cpp:2961: GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max) failed`
     ⇒ **這條對照臂不存在**（`spec_cost_curve.py` 的 k=0 基線也受影響）。
     ⇒ 要無投機對照只能用 **`CGC_SERVER_MTP=0`**，那就回到上面那 7 個 env 的混淆
       ⇒ **兩條路都不乾淨 ⇒ 「MTP on/off 的單變量對照」目前無解，不要假裝有。**
       （補償式替代：F 用 **MTP-off 的 ntok=1** 直接量，再對照曲線外推 ——
        兩者差 **−2.5%** ⇒ 該 env block 不實質移動 F，DESIGN GAP 被實証結清。）
   （同型歷史教訓：`:2048` 原注——「expert cache ON is not bit-identical」的調查一直少開這根支柱。）
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
5. **★★ 2026-09-19：引用任何 decode t/s 之前，先問「生成長度是多少」—— 而且別假設有穩態。**
   ⚠ **本條 12:0x 由 §EN-195 實測改寫**：原寫「穩態 22.2、暫態 8.1–8.9，拉長就回到 22.2」——
   **對當前 build 不成立**。實測（profile250、8 GiB、NOMINAL 到底）：`-n 128` 7.96（σ **45%**）、
   `-n 512` 隨機 **10.22**（σ 10.5%）、`-n 512` **真文本 10.19**（σ 8.7%）。
   ⇒ **真文本 vs 隨機差 0.2%**（§EN-190「bench 慢是因為餵隨機碼」對 t/s 被證偽）；
   ⇒ **沒有收斂到 22.2**，因為 `-n` 變大時 `capacity miss 12.9%→51.6%（變主項）`、
     `layers_over_slots 5→22`、`worst layer distinct 245 vs slots 143`、evictions/misses≈1.00
     ⇒ **143 槽裝不下真 token 工作集，且越長越糟 ⇒ 結構性 thrash，不是「暫態後收斂」。**
   ⇒ 仍然成立的部分：**採樣位置決定數字**（`-n 128` 比 `-n 512` 低 28%、噪音大 4×）
     ⇒ **報 decode 一律用 `-n ≥512`，且必須同時報 σ**（只看平均會把 45% 的散佈藏掉）。
   **`src/llama.cpp/src/llama-expert-cache.cpp:381-385` 的註解是本機實測，不是推測**：
   一次 67-token prefill 會把池翻掉，接下來 **32 個 decode token 必須重讀 ~1850 experts
   （~2 GB、~2.4 s）才收斂** ⇒ **steady 22.2 t/s，但 prefill 之後的第一段生成掉到 8.1–8.9 t/s**。
   自洽驗算：2 GB ÷ 2.4 s ÷ 32 tok ≈ **75 ms/token** 額外重讀 ＋ 穩態 45 = 120 ms/tok ≈ **8.3 t/s** ✓。
   ⇒ **`n_predict=128` 走不完暫態** ⇒ 今日所有 6.38–13.0 的讀數**全是暫態與穩態的混合**。
   兩台儀器都不例外（`llama-bench` 更糟：**它從不呼叫 `srand`**，4 個 rep 的 2025-token depth
   是 4 串不同隨機 token，每 rep 把剛暖起來的池整個沖掉 ⇒ 從來沒有穩態，spread 1.424 > 自定的 1.10）。
   **⇒ 修量測窗口（零代碼）優先於任何引擎改動**：`n_predict ≥512` 且只報尾段，或用
   `CGC_PREFILL_PROTECT_FILE`（⚠ **`CGC_PREFILL_PROTECT=0` 也會開**——`getenv != nullptr` 就成立）。
   ⚠ 但 **`PREFILL_PROTECT` 的 A/B 不要再跑第二遍**（09-19 §EN-191：漂移 −0.619 t/s per rep > 效應）。
   ⇒ 通則：**「差 2.4×」這種直覺先問「被比較的兩個量是不是同一個東西」** —— 本 repo 已第三次命中的
   正是這一型（§EN-190 的取樣位置、§EN-191 的暫態、此處的穩態 vs 暫態）。
6. **roofline 要先算「每 token 位元組的構成」，不要默認瓶頸在最顯眼的那一塊**（2026-09-19）。
   本機實測（`gguf_pool_geometry.py` 直讀標頭）：**專家 347 MiB/token ＋ 稠密 ~1.45 GiB/token
   ≈ 1.8 GiB/token** ⇒ **專家只佔 20%，稠密佔 80%**。
   而 M1–M6 那一整套池優化打的正是那 **20%**（與 09-18 節點剖析同向：GatedDeltaNet＋狀態管線
   35–45%，MoE 12–15%）。三個 roof 的實測佔用：DRAM ~19 GB/s（**15%** of 120）、
   SSD 7.9 MiB/token（`fill_wait=0.000`）、算力 ~65 GFLOPS（**2%**）
   ⇒ **既不是 IO bound 也不是 compute bound，是固定開銷／序列化受限**
   （633 command buffer/步、6.3 節點/buffer、GPU idle 35.3%；`cache` 桶真工作只夠 0.1–0.5 ms 卻量到 27 ms）。
   **「25 t/s 只需 48 GB/s ＝ 峰值 40%」⇒ 它是可達的，不是天花板問題。**
   （SSD 頂：347 MiB ÷ 3 GB/s = **8.6 t/s** ⇒ 沒有池就回個位數，池是命根子。）
   ⚠️ **2026-09-19 修正：這一條的「固定開銷／序列化受限」框架與由它推出的
   「天花板 1000/55.3 = 18.1 t/s」在生產載體上都不成立，引用前先讀這四行。**
   - `55.3 ms idle` / `633 CB` / `GPU idle 35.3%` 是 **`decode_step_profile.py` 的 `prof` arm** 量的，
     而該 arm 是 **MTP-off ＋ `CGC_SERVER_PROFILE=off`**（`run_server.sh:119` 的預設），
     且 **55.3 是每步、18.1 是每 token**（1 token/步）⇒ 對 MTP-on 的 25 t/s 目標三重不適用。
   - 在 **MTP-on ＋ prod25** 載體上實測（`--arms mtp`，兩支，12-tok 與 2250-tok prompt 一致）：
     **verify 步 203–232 ms，其中 GPU 162–176 ms ＝ 76–80%**；`cb` 54–73 ms（25–32%）；`submit` ~4 ms（2%）。
     ⇒ **GPU 在做實事，不是空轉**；`sync = wait − gpu` 在該載體上**為負** ⇒ 該分解不適用。
   - 每步拆帳：`segs=41 layers=40 ntok=4` ＝ **verify**（1＋3 drafted）；`segs=2 layers=1 ntok=4`
     ＝ **draft forward ≈ 1.8 ms（<1.5%）** ⇒ **優化 nextn 層前向給不了 25**；
     要動的是 verify 的 41 層（其主項是 4 個 token 的專家 union ≈ 132 ms）。
   - ★ 25 t/s 的算術：`13.98 ÷ mean_len 2.35 ⇒ 171.7 ms/步`，目標 **94 ms/步**；
     每步 GPU ≈ 0.78 × 172 ≈ **134 ms > 94** ⇒ **把 CPU 側全部砍到 0 也到不了 25**
     ⇒ **必須降低「每步 GPU 工作量」（＝讓同一個 union 攤更多被接受的 token）**。
7. **★★ 2026-09-19：報 decode 必須同時報「每步 token 數」（接受率），而且 server 那條線的
   第一個 request 是 warmup —— 它的 `mean len` 不是生產值。** 兩點各讓一個結論翻車過一次：
   - 兩側同定義的量是 **`mean_len ≡ emitted / verify_steps`**（server `1 + accepted/verif_steps`；
     bench `n_gen / rounds`）。它與 t/s 一起才能分解缺口：**`t/s 比 = (mean_len 比) × (ms/step 比)`**。
   - **陷阱 1（認 task 編號，不要抓第一行）**：`slot print_timing` 是一 request 一組，**第一組屬於
     warmup**（`n_predict=16`）⇒ 抓第一行會拿到 `mean len = 2.50`，而**實測請求是 2.40**
     （2026-09-19 §EN-212 就是這樣把 2.50 寫進分解的）。
   - **陷阱 2（KV 複用）**：server 的 `prompt eval time` 若只有十幾個 token ⇒ **它複用了前一個 request
     的 KV**，rep2+ **沒有重新 prefill**。所以 server 的 `mean_len` 跨 rep 不變（且逐字元相同），
     而 llama-bench **每個 rep 都重新 fill** ⇒ 池被 churn ⇒ `mean_len` 逐 rep 遞減。
     **這才是 bench/HTTP 差距的主要來源，不是引擎快慢**（`llama-bench.cpp:470` 的註解講的就是這個；
     用 `--fixed-fill-seed` 去驗它是**驗錯變數** —— churn 來自「重新 fill」這個動作，不是 fill 的內容）。
   - 取 `mean_len` 用 `LLAMA_BENCH_SPEC_DBG=1` ＋ 正則 `SPECDBG round: n_done=(\d+) n_past=(\d+) draft=(\d+)`。
     ⚠ `n_done` 印在累加**之前**（`:2955` vs `:3017`）⇒ 用 **`n_gen / rounds`**，不要用 `(末-首)/rounds`。
   - ⚠ **形狀本身會燒機**：`-p 2025 -d 0` 每 rep 全速 prefill ⇒ **thermal HEAVY** ⇒
     **`-p` 形狀的 t/s 不可引用**（該形狀要引用得先確認 NOMINAL）；`mean_len` 是 token 判定，不受熱影響。
   - ★ **權威值是「平台」，而只有 HTTP 能直接給出平台。** 2026-09-19 的階梯（同為 prod25、
     2025-token prompt、128 gen、k=3、8 GiB pool；單位 t/s）：
     | 值 | 形狀 | 為什麼 |
     |---|---|---|
     | 7.48 | `-p 0 -n 128 -d 512`（`profile_duo` 交付 cell） | 冷池＋只填 512＋窗口 128 ⇒ 全在暫態 |
     | 10.20 / 10.79 | llama-bench `-d 2025 -n 128 -r 4` 的 avg / platform | 每個 rep 重填 ⇒ 含暫態；4 rep 散 **1.94×** |
     | **13.1–14.0** | **HTTP `http_duo.py --reps ≥6`，rep 2 起** | **收斂平台（5＋3 個 rep 都不再爬升）** |
     | 10.2 / 11.1 | 首個 request | 冷 |
     ⇒ 報 decode **一律報這一格**：`decode_tps_steady`（reps 2..N 均值）＋ `decode_tps_cold`（rep1）。
     ⚠ **單一 rep 的 `samples_ts` 尖峰不要當收斂值** —— 2026-09-19 曾把 bench 的 rep4（**15.33**，n=1）
     當成「bench 收斂後超越 HTTP」，而 HTTP 側 8 個 rep 一致停在 13.0–14.0 ⇒ 那是上界樣本。
   - ⚠ 對帳 bench／server 的 env 時：**它們本來就是對齊的**（`lbm.resolve(profile, extra)` 與 bench 的
     `env` 逐鍵 diff = 0）。`extra_env` 只是 arm 在 profile 之上多加的 ⇒ 別把「arm 只有 2 個 env」
     誤讀成「少了 MTP env 塊」。

8. **★★ 2026-09-19：這台機器上「連著跑兩臂」默認是無效對照 ---- 第二臂會整段跑在熱池裡。**
   實例：`A(-d 512 -n 256)` 跑完後 thermal 已 **HEAVY 59/180**，緊接的
   `B(--warm-skip 128)` 是 **HEAVY 259/259**（**全程**）⇒ B 的讀數（avg 7.02）是降頻產物，
   而 A（avg 9.72）也已部分受污染 ⇒ **兩臂都不能引用**。
   ⇒ 規則：**每一臂起跑前都要 `thermal=0`**（實測約 2-3 分鐘）。
   ★ 2026-09-19 再踩：我把欄位寫成「兩臂之間」，結果把 **build 排在第一臂之前**
   —— 8 執行緒編譯本身就把機器加熱了，第一臂從一開始就是熱的（C 全段無 NOMINAL、
   D 的 HEAVY 占 172/223）⇒ **兩臒讀數全部作廢**。
   ⇒ **「build」、「前一臂」、「別條線的 run」全部算「前一臂」**；機器熱了就不要開始，
   不要把 build 與量測串在同一個 chain 裡。
   ★★ 後續量出來的**實際噪音底**（這是本機器上任何 decode A/B 的前提）：
   **同配置、同 binary、同形狀的兩臂相差 15–18%**
   （`decode_window_harness.py` 自己的 b1 vs b1b：**102.45 vs 89.25 ms/token = 1.148×**；
   它的 docstring 另記一次 **17.5%**；我自己 `--ctx-size` 重複實驗得 **1.12–1.18×**）。
   ⇒ 任何小於 ~15% 的 decode 效應，**單次或雙次啟動都不可判**，而且你會憑運氣生出 20% 的假效懜
   （2026-09-19 实例：我把某一次 13.819 當成突破報出，重跑變 8.845）。
   ⇒ 規則：**寫任何 decode A/B 結論之前，先跑 `paired_ab.py --null`（兩槽同配置）量底**；
   底若 ≥10% 則正解是「不可測」，不是繼續找槓桿。另外報**配對中位**而非 mean（本日出現過 17.64 這種離群 rep）。，並且每一臂都要報
   `thermal.hist`，不只報 launch/worst**。`worst` 是哪一瞬都不知道的值，
   `hist` 才能讓人看出「這一臂有多少時間在降頻」。
   ⇒ 可行的替代：**ABBA 交錯**（漂移會抵消），或把每臂做短、多跑幾輪。
   ⇒ 附帶：不要用 `pgrep -f 'llama-server'` 當閘門 ---- 會命中別人的包裝指令行
   （`bash -c ... pgrep -f 'llama-server' ...`）而誤報；用二進位路徑 `build/bin/llama-server` 或 `pgrep -x`。

9. **★★ 2026-09-19 晚：F 的組成量出來了 —— **70% 是 GPU `union`**，25 t/s 是「GPU 工作」問題，不是啟動稅問題。**
   量法：`scripts/check/sntok_curve.py` 在**單一臂**內（r0 ＝ MTP off、ntok=1）用**逐層中位數**
   拆開 —— 不走跨臂梯子（跨臂被證明不可行，見下）。
   `F = 71.1 ms/step = 1.81 ms/layer`（40 層；77.3 ms/token）
   | 桶 | 合計 ms | 每層 | 佔比 |
   |---|---:|---:|---:|
   | `union`（GPU span，**誠實的那個**） | 50.5 | 1.26 | **70%** |
   | `gap`（segment 之間 GPU idle） | 14.5 | 0.36 | 20% |
   | `cb`（專家填充） | 4.2 | 0.10 | 6% |
   | `submit`（segment 派發） | 3.0 | 0.08 | 4% |
   ⇒ **可當 overhead 的至多 30%。把非 GPU 的毫秒全部歸零，F 仍是 1.26 ms/layer ＝ 目標的 2.5×。**
   ⇒ **25 t/s 是關於 GPU 執行（attention ＋ MoE 數學）的陳述**，不是 dispatch／prefetch／cache fill。
   ⇒ **GDN 層是貴的**：比 10 個 full-attn 層多 **+0.48 ms/layer**（union），30 個 GDN ⇒ 14 ms/step
     ＝ F 的 20%；要砍 52 ms/step ⇒ **需要 3.6 個這種量級的缺口**。單一「GDN 修好」只是零頭。
   ⇒ **暖填充住在 L0–L4**：ntok=4 warm 時 92% 的 `cb`（8.92/9.75 ms）在 L0–L4，L5+ 只 0.83；
     冷的時候是散開的（L0–L4 10.40、L5+ 21.95）。**步總量看不到這件事。**
   ⚠ 儀器限制（要跟數字一起引）：`gpu` **重複計數重疊的 buffer**（可超過 `wait`）
     ⇒ **只能引 `union`**；四個桶**不嚴格分割步驟**（closure 75–121%）⇒ 百分比帶著這個寬度；
     沒有 per-node timing ⇒ **attention vs MoE 無法分開**，GDN/full-attn 的差只是下界。
   ⇒ 附帶：**跨臂比較在這台機器上已被證明不可行**（pass 2 的 4 個同配置 ntok=4 錨點散
     **97.5%**，且全程 **0 次** `window lost` ⇒ 温度之外還有 **ambient** 扰動：Spotlight 大量建索、
     TimeMachine、GUI 負載 ⇒ **「等安靜窗口」不會消除它**）。可行的只剩**單臂內、逐層、看中位數**。

10. **★★ 2026-09-19 晚：per-node 儀器本來就在樹裡 —— 別重造。**
   引擎側既有（全在 `run_server.sh` allowlist）：`CGC_DECODE_PROFILE=1`（**前置**）、
   `CGC_DECODE_PROFILE_ALL=1`、`CGC_GPU_TIMING=1`、**`CGC_GPU_NODES=1`（per-KIND 表）**、
   `CGC_GPU_OPS=1`（按 ggml OP 的第二張表）、`CGC_GPU_NODES_TRACE=1`、
   `CGC_GPU_NODES_MATRIX=1`（每個 command buffer 一行，`CGC-NSM`）。
   讀它的工具：`scripts/check/attn_moe_split.py`（`run` ／ `analyze <log>` ／ `selftest`）。
   實測（prod25、MTP on、`-n 128`；**只取 `kinds=44` 的 40 層表**，中位數佔該步 `seg_busy`）：
   | 桶 | `wcntw`% |
   |---|---:|
   | `attention` | 6.2 |
   | **`moe`（`ffn_moe_*` ＋ `shared_expert_gate`）** | **21.3** |
   | `gdn` | 11.6 |
   | `ffn_dense`（`ffn_*`） | 6.6 |
   | **`other`** | **34.2** |
   單項前二：**`node` 13.1、`cache` 8.1**（`ffn_moe_` 只有 **7.7**）。
   ⇒ ① MoE 家族（27.9%）約是 attention 家族（17.8%）的 **1.6×**。
   ⇒ ② **最大的可攻擊目標是 `node`／`cache` 這類管線葉子，不是 expert GEMV**。
   ⇒ ③ 引擎自檢 **`delta = 0.000%`**（`seg_busy` == `layer gpu_sum`，節點→buffer 映射自洽），
     **但 `lb` 幾乎全 0、`ub` 大量重疊 ⇒ [lb, ub] 分不開任何兩桶**；
     只有 `wcntw` 能排序，而它是「buffer 時長分給會 encode 的 node」這個 **歸因模型**，
     **不是獨立量測**。named-work 佔 80%、残差 20% 未歸屬。
   ★ **陷阱（我踩了）：分組要按「列數」。** 引擎**每個在該步出現過的 kind 印一列**
     ⇒ 列數就是形狀簽名。用「有沒有 `ffn_moe/attn/gdn/conv`」判形狀會**全部誤判**
     （MTP 層自己就是 GDN 層，帶 `conv`／`gdn_out`）⇒ 聚合把 trunk 種類稀釋 **2.2×**，
     讀出 `attention 0.0% / moe 2.5%`。實測形狀：38 張表 = **17×44 列**（40 層）＋ **21×7 列**（單一 MTP 層）。
   ★★ **op 級（`CGC_GPU_OPS`，trunk 形狀 `nodes_all=4076`，中位 total 115.6 ms）—— 兩個 matmul 只有 ~19%：**
   `MUL_MAT` **15.1** ／ `ADD` **14.2** ／ `MUL` **11.1** ／ `RMS_NORM` 7.5 ／ `UNARY` 7.5 ／
   `CPY` 6.9 ／ `GET_ROWS` 5.1 ／ **`MUL_MAT_ID` 3.7** ／ `L2_NORM` 3.3 ／ `GATED_DELTA_NET` 2.8 ／
   `GLU` 2.5 ／ `SCALE` 1.9 ／ `SUM_ROWS`·`CLAMP`·`DIV` 1.2 ／ `CONCAT` 1.0 ／ `ROPE` 0.6 ／ `SSM_CONV` 0.5 ／ `SET_ROWS` 0.4 ／ `SOFT_MAX` 0.4
   ⇒ **`MUL_MAT : MUL_MAT_ID = 4.1×` ，而每 token 位元組比＝稠密 1.45 GiB : 專家 347 MiB ＝ 4.2×**
     ⇒ 兩個 matmul 都是**純頻寬串流**，比例完全由位元組解釋。
   ⇒ **錢在 elementwise（`ADD`＋`MUL`＋`UNARY`＋`SCALE` ≈ **34.7%**）＋ norm ＋ copy/cache，不在 expert GEMV。**
   ⚠ 這與 §16 的「25 t/s 是 attention ＋ MoE 數學問題」**不同調**：§16 引的是 `union`（GPU **span**）佔 70%，
     op 級是**歸因模型**的佔比 —— 兩者不可互換。
   ★ **`VIEW` 962 個節點（23.6%）＋ `RESHAPE` 595（14.6%）＋ `TRANSPOSE`/`PERMUTE` 各 30 ⇒ 0% GPU**；
     **40% 的節點不產生任何工作**（`nodes_all 4076` vs `nodes_work 2455`）。
   ★ **身分：`node` ＝ 沒有名字的節點**（`ggml.c:7192` 自動 `node_%d`；單一最大 kind **13.1%**）；
     **`cache` ＝ KV cache ＋ GDN 遞歸狀態 cache**（`cache_k/v_l*`、`cache_r/s_l*`）—— **不是專家池**
     （`llama-expert-cache.cpp` 對節點 `format_name`／`set_name` **零命中**）。kind 名來自 **67 條前綴表 `ns_fix[]`**
     （`ggml-backend.cpp:2175-2231`，最長前綴命中，否則 `(other)`）⇒ **那份清單就是解碼環**。
   ★ **`lb` 為何永遠不能排序（機制）**：NSM（每個 command buffer 一行）44,135 個 buffer 裡，
     **沒有任何一個是前 40 個 kind 的 solo**（直方圖 `{1:1161, 2:3349, 3:895, 4:6901, 6:26187, 45:482, 64:5160}`；
     有 **26,187 個剛好 6 節點**、5,160 個是 64 節點）⇒ **`lb ≡ 0` 是構造性的**。
     ⇒ **只能引 `wcntw`，且要説它是一個模型**。
   ★ **形狀簽名在 op 表是 `nodes_all`，不是列數**（op 表每種形狀都印滿 27 列）
     —— kind 表的教訓**不轉移**。混算會得到 `CPY` 佔 466% 節點的荒謬值。
     工具：`attn_moe_split.py ops <log>`（自測 27 項）。
   ★★ **實測結果（trunk 形狀＝44 列；prod25、MTP on、`-n 128`）**：
     **`node`（12.8% of step）究竟是什麼**：**MUL 26% ＋ UNARY 26% ＋ GET_ROWS 21% ＋ MUL_MAT 14% ＋ ADD 7% ＋ FLASH_ATTN_EXT 5%**
     ⇒ **未命名節點 ＝ elementwise（MUL＋UNARY＋ADD ≈ **59%** of node）＋ row-gather（GET_ROWS 21%）＋ 未命名的稠密 MUL_MAT 14%**。
     ⇒ **不是 attention、不是 MoE 數學** —— §EN-229 的第二個獨立確認。
     **`ffn_moe_` 8.0%** ＝ ADD/MUL/DIV/SUM_ROWS/GET_ROWS/CLAMP **各 14%**（combine 管線，**不是 gather**）。
     **`cache` 8.0%** ＝ **CPY 74% ＋ SCALE 22% ＋ SET_ROWS 4%** ⇒ **KV／GDN 狀態 cache 的成本是「複製」，不是算術**。
     `ffn_moe_add` 6.8（ADD 100）、`norm` 6.5（RMS_NORM 100）、`conv` 2.6（CONCAT 35/GET_ROWS 35/SSM_CONV 15/UNARY 15）、
     `k_conv` 2.3（L2_NORM 100）、`z-` 2.3（MUL_MAT 100）、`gdn_out` 2.3（GATED_DELTA_NET 100）。
   ★ **強制交叉驗算（我第一版就是這樣被抓到的）**：把 GPUOPK 該 kind 各列相加，
     去對 kind 表**同一 block** 的 `wcntw` 值 —— **比值必須 1.000**。
     ★ 隨之而來的規則：**新增的 per-step 累加器要跟其他累加器一起在每步結尾歸零**（`ggml-backend.cpp`
     約 `:2822-2855`），**不要放在「印」的區塊裡**（它 1-in-8 才觸發）。
     我把歸零放在印的區塊 ⇒ 累了 **8 步** 對 **1 步**的 `ns_total` ⇒ `node` 讀出 **177×** 偏大（修後 22 個 block 全 1.000）。
   ★ 第三度踩形狀陷阱：我新寫的 GPUOPK 行**沒帶形狀** ⇒ 混算讀出 `node 75.3% of step`。
     修法不必改 C++：**讓每列繼承它所屬 `CGC-GPUNODE` block 的「kind 列數」**。
     ⇒ **通則：任何 per-step 儀器行都必須能歸屬到它所屬的步**，否則跨形狀聚合會產生「>100% of step」。
   ★★ **`cache` 的 CPY 指名了（2026-09-19 M8）**：kind `cache`（8.0–8.2% of trunk step）＝
     **CPY 74–75% ＋ SCALE 21–22% ＋ SET_ROWS 4%**。两次獨立 run 重現。
     CPY 的來源：**`build_rwkv_token_shift_store`（`llama-graph.cpp:4077`）逐 GDN 層對 `cache_r_l<il>`
     做 `ggml_cpy`**（TRACE 原始名直接看到 `cache_r_l30` 落在 `ffn_moe_*-29` 的 64-node 主 buffer 裡）。
     大小＝`n_embd_r()`＝`(d_conv−1)·(d_inner ＋ 2·n_group·d_state)`；本模型 **96 KiB／層 ⇒ **2.9 MiB／步**。
     ⇒ **2.9 MiB 卻值 ~6% 的步 ＝ 0.36 GB/s，比 120 GB/s 低 ~330× ⇒ overhead-bound
     （30 次序列化小 copy，~265 µs/copy），不是頻寬。**
     ⚠ **`delta-net-base.cpp:509-526` 的 rollback 迴圈（`K = n_rs_seq+1` 次 cpy）是另一組、而且沒有名字**
       ⇒ 它們落在 kind **`node`**，不是 `cache`。KV cache 也不是它 —— KV 是 `SET_ROWS` 那 4%。
     ⚠ 佔比是**模型**（`wcntw` 把 buffer 時長分給會 encode 的節點）；NSM 分不開
       （`cache` 出現在 **75% 的 buffer**、涵蓋 90.6% 的 buffer 時間、**從不 solo**）。
       ⇒ 報它時要分三層說：**身分確定（TRACE）、大小精確（幾何）、成本佔比是模型**。
   ★ **KIND × OP（新增儀器，2026-09-19）**：`node` 是 **13.1% 的未命名節點**，而三張既有表都
     答不了「它是哪些 op」（kind 表按**名字**、op 表按 **op**、TRACE 印的就是 `node_<i>`）。
     → **名字是標籤，op 才是工作**。實作在 `ggml-backend.cpp`（5 處插入、+48 行，
     `Backup/patch_kind_op_xref.py` 可重放），輸出 **`CGC-GPUOPK: <kind> <op> <ms> <% of kind> \| <% of step>`**，
     **只在 `CGC_GPU_OPS=1` 下印**（它本身已要求 `CGC_GPU_NODES=1`）⇒ 預設路徑不受影響。
     分配用**與 kind 欄相同的分母**（每個工作節點 `dur / wtot`）⇒ **同一 kind 的 op 列相加＝它的 `wcntw`**，
     是**細化**不是另一個模型。讀它：`attn_moe_split.py ops <log>` 的 `KIND x OP` 區塊。
   ★ **在別人正在量測、不能 build 時，怎麼驗證共享樹裡的 C++ 改動（不寫任何產物）**：
     從 `src/llama.cpp/build/compile_commands.json` 取該 TU 的完整編譯命令 → regex 去掉 `-o <obj>` 與 `-c`
     → 接上 **`-fsyntax-only`** → 以該 entry 的 `directory` 為 cwd 執行。**rc=0 且零診斷 ⇒ 編得過。**
     ⚠ **改動活在共享工作樹裡，別條線隨時可能編譯它** —— 不驗就等於把未爆的編譯錯誤
     放進別人的 build。而 `pgrep -x` 閨門會（且應該）擋下自己的 build。

   ★★ **引用指紋之前先指名是哪一套 digest —— 本 repo 同時有兩套，數字不可能相等**
     （2026-09-19 發現；本線當日曾把兩者當同一組比對而誤判「逐位相同」）：

     | 來源 | 範例 |
     |---|---|
     | `scripts/check/engine_freeze.py` | sha256 前 24 hex：`libggml-base 8cb4a6b74b9cc2f5433b7738` |
     | 線 B 的 harness 報告頁首 ╱ `m123_oracle_gate` 的 `build` 行 | 另一套：`libggml-base 054fb22f04a01c5c` |

     ⇒ 兩套各自同源（harness ↔ oracle gate 可互比），**但跨套比對等於比兩個不同的函數**。
     → **歸因的判準是指紋，不是 `git status` 乾淨**（本 repo 把 build 產物納入版控
     ⇒ rebuild 之後 `git status` 本來就會髒）。每一臂前後各跑一次
     `python3 scripts/check/engine_freeze.py verify --tag <tag>`，並貼 `flags`／`artifacts`／`source` 三段。
     ⚠ `Backup/engine_freeze/` **被 gitignore ⇒ 那些 tag 是本機記錄**，交接時要指明檔名。
     ⚠ 實測（2026-09-19 19:2x 對 `pre_plainmatch_0919`）：**唯一漂移的產物是 `libggml-base`**
     （KIND×OP 儀器）；`llama-server`／`libllama`／`libggml-metal` 全部 MATCH。
   ★ **共享 `src/` 的擁有權與歸因協定寫在 `docs/SHARED_SRC_OWNERSHIP_2026-09-19.md`**：
     目前 `ggml-backend.cpp` 已提交（`4fdfaa8de`）並轉移給 overlap 那條線；本線不再編輯它。
     要重放它的儀器：`Backup/patch_kind_op_xref.py`（5 個錨點，各自恰好 1 次）。
     ⚠「還原源碼」比「提交」更糟：**被追蹤的 dylib 仍含該儀器**
     ⇒ 原碼說 A、binary 說 B，正是閘門檢查 8 存在的理由。
   ★★ **逐層表（`CGC-DECPROF all: L<il>`）的欄位語意 —— 它是目前唯一能回答「這一層 GPU 在忙還是在等」的儀器**
     （`attn_moe_split.py layers <log>`；語意讀自 `ggml-backend.cpp` 的 `dp_lay_*`／`sg_*`）：
     - **可加的只有三項**：`wait`（CPU 等前一段 GPU）＋ `cb`（top-k hook：槽管理＋阻塞填充）＋ `submit`（派發）
       ＝ 步級行的 `total`。實測對帳：step=1 印 `cb=476.95`，逐層相加 **476.9** ⇒
       **逐層相加就是步級真值**。
     - `gpu`＝segment 內各 buffer 忙時**加總**，重疊者**重複計數** ⇒ 可以 > `union`，**永不可加**。
     - `union`＝segment 的 GPU **跨距**（span）⇒ **span 內的空檔看不見**。
     - `gap`＝**段間** GPU 空檔（前一段 GPU 結束 → 本段 GPU 開始）⇒ 這是這張表唯一能指名的空檔。
     - ★ **`ggml-backend.cpp:2130` 自己寫了內建交叉校驗**：該視窗坐在**前一段的 hook＋submit** 裡
       ⇒ **`gap` vs `cb+submit`**。實測：`gap/hook`＝**1.03–1.21**（全 run 穩定）、
       **`corr(Σgap, Σ(cb+submit))`＝1.000** ⇒ **它們是同一個區間的兩個視角**
       （CPU 在做槽管理＋阻塞填充的整段時間，GPU 沒有工作），不是兩個獨立量。
     - ★★ **逐層中位數 × 層數 ≠ 總和**（右偏分佈 ⇒ 乘起來系統性低估）。
       2026-09-19 就是這個差造出一個錯的「bottleneck 只有 7%」並且已交給別條線。
       規則：**要總和就相加，不要拿中位數乘次數**。
     - 家族判準：`FULL_ATTN = set(range(3, 40, 4))`（10 層），其餘 **30 層是 GDN** ——
       而 **GDN 就是擁有 `cache_r_l*` / `cache_s_l*` 的家族**（`build_rwkv_token_shift_store`，
       `llama-graph.cpp:4077`）⇒ **「GDN vs full-attn」就是「cache_r 的 cpy 有沒有變成空檔」**。
   ★★ **不要假設 `cb`（hook 窗）是「等填充」—— 用池自己的 `fill_wait_us` 對帳。**
     實測（2026-09-19，同一支 log）：**`fill_wait_us` 全程 只有 78 ms**，而 `cb` 是 **36.5 ms/步**
     （暖半段中位）；若 cb 是阻塞等填充，光暖半段就需 ≥2374 ms
     ⇒ **實際只夠 2.13 個步的 cb** ⇒ **填充早已完全被藏住**（`pread` 5378 s 在 IO 執行緒上跑）。
     ⚠ **預取的 drop 統計不等於「有人在等」**：`prefetch=482/190` 且 **190/190 全部 `drain_cleared`**
     （發起 482、完成 190、完成的全被丟），**但沒有人因此等待** ⇒ **不要拿它當「修預取就有收益」的證據**。
     ★ 那 `cb` 到底是什麽？先看**它集中在哪幾層**：實測是 **L0–L5**（每層 1.6–4.2 ms，
     其餘層平底 ≈0.2–0.4；**top4 = 42%%、top8 = 59%%**），而池自己的 dump 該幾層正是
     `layers_distinct_over_slots=5`、`worst=layer 1 distinct=217 slots=143`（**1.5× 超額**）
     ⇒ **hook 窗＝那幾層的槽位／驅逐簿記的 CPU 成本**（最一致的解釋，未證）。
     反面判據：**步內 `corr(cb, union)` 跨層 = −0.085**（若 cb 是等該層自己的 GPU，
     它應該同步）⇒ **不是 per-layer 同步/readback**。
     ⇒ 實測結語：**逐層空檔不是故事**（兩家族 `gpu/union`≥1，span 內塞滿；
     GDN 的 gap 1.09 vs attn 1.00 —— 沒有因為 cache_r 而多出空檔）；
     **真正的大空檔在步層級的 hook 窗（約 28–32% of step）**。


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
| `CGC_SUBMIT_AHEAD=1` | 無（改變順序） | 天花板上界探針。**★ 2026-09-19 分欄位更正**：**步時上界有效且重現（×1.702，與認證的 ×1.711 差 0.5%）**，但 `gap/union`（`(NO TIMESTAMPS)`，結構性）與 **t/s**（acceptance 崩掉 2.40→1.00）都不可讀 ⇒ 兩臂圖寬必須相同才可比，見陷阱 28 |
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

## 匿名節點（`node_NN`）：先零成本歸屬，需要時才命名（2026-09-18）

`node_NN` 是 **ggml 計數器自動名**（⇒ 兩臂的圖不同 ⇒ **同名不同物、不可跨臂比對**）。
41 層的 trunk 圖裡有 **620 個**，在節點加權表裡佔 **15–29%** —— 是最大的一塊「真運算但沒有子系統名」。
2026-09-18 把它降到 **350**。方法是兩步，**順序重要**：先歸屬（免費），再決定要不要動原始碼。

### 步驟 1（免費，不必重建）：用 dump 的**索引順序** ＋ 左右最近的有名節點

`CGC_GRPH_DBG=1` 的 dump 逐字印
`CGC-GRPH[<idx>] name=<name> op=<n>(<OP>) ne=[<ne0>,<ne1>]`，而 **`idx` 就是圖建構的順序**
⇒ 對每個 `node_NN`，往左／往右各走到第一個**有名**的節點，那兩個名字就是它的上下文。
實測（一個 4116 節點的圖）乾淨得可直接解讀：

| op | 數 | 左鄰 → 右鄰 | 解讀 |
|---|---|---|---|
| `ADD` | **240**/270 | `ffn_moe_weighted` → `ffn_moe_out` | **MoE 專家輸出的歸約鏈** |
| `GET_ROWS` | 90 | `cache_r_l*` → `cache_s_l*` | **遞歸狀態的 gather**（★ 2026-09-18 更正：舊寫「池的 gather」是錯的，見下） |
| `MUL_MAT` | 130 | `alpha`/`beta`、`attn_norm` → `linear_attn_qkv_mixed` | 線性注意力投影 |
| `MUL` | 60 | `norm` → `final_output` | 輸出縮放 |
| `GATED_DELTA_NET`／`UNARY`／`FLASH_ATTN_EXT` | 30／30／10 | — | 線性注意力核心／遞歸狀態更新／10 個全注意力層 |

**★ 2026-09-18 更正：`cache` 桶不是「池自己的暫存」，是「遞歸／KV 狀態管線」。**
把 `cache` 桶成員的名字逐字印出來（新 dump，log `050331`）是
`cache_r_l*`（遞歸狀態，`llama-model.cpp:378-379` 的 `pattern_r_cache`）、`cache_s_l*`、
`cache_k_l*`／`cache_v_l*`（10 個全注意力層的 KV）—— **與專家池無關**。
池的填充在 **CPU 側的 hook**（計在 `CGC-SEG` 的 `cb`），**在 GPU 節點表裡沒有成本**
（與 `fill_wait = 0.000` 一致）。
⇒ 「`cache` ≥ 整個 MoE 專家 GEMV 家族」這句的**份額**成立、**歸屬**不成立；
而更正後結論更強：**線性注意力（GatedDeltaNet）家族 ≈35–45%，MoE 家族 ≈12–15%。**
另外它是**操作數／啟動開銷**受限而非頻寬（`cache` 真工作位元組 ≈ 每 token 十幾 MB ⇒
0.1–0.5 ms，量到 27 ms）：**120 個 CPY** 來自 `delta-net-base.cpp:509-526`，`n_rs_seq != 0`
時每層建 `K = n_rs_seq+1` 個獨立 `ggml_cpy`（30 層 × 4 = 120，與 dump 逐格吻合）。
（出處 `.workbuddy/memory/2026-09-18.md` §EN-132。）

**★ 同一輪的另一個陷阱：`cb` 的份額會被「細粒度臂自己」灌大。**
本 skill 曾記「decode 步 ≈80 ms；`wait` 83–90%、`cb` 7–15%、`submit` 4%」。用
`en-work-fine`（`CB_N_MAIN=1` + `SERVER_N_CB=16` ⇒ **每段 17 顆 command buffer**）量，
`cb` 會變成步的 **42–75%** —— 那是**那個臂自己造成的偽影**，不是生產 shape。
⇒ **要引用相位分解，先指名臂的 `n_main`／`n_cb`；`n_cb` 掃描在生產 shape 上已於 09-15
判死（1→16 無影響、`cb+submit` 僅 9%）。**

**★ 而這個偽影不只影響 `cb` —— 同一個臂會同時灌大 `gap` 與名字表的份額（09-18 13:3x 實測）。**
判別式是 `CGC-GPUTIME` 那行的 **`bufs`**：生產 shape **360**（`skipped=0`），細粒度臂 **約 640**
（`skipped=37–47`）。同一批 log 的對照（都取 `segs=41 layers=40` 的 decode 步）：

| 臂 | `cb` | `gap/union` |
|---|---|---|
| 生產形狀（`p25-mtp-on-diag`） | **17.5–24.0 ms** | **21–25%** |
| 細粒度（`en-work-fine`） | **38.5–112.0 ms** | **29–93%** |

⇒ **可引用的述句是「生產形狀下 `gap` ≈ 21–33%」**；`en-work-fine` 的 57%／93% **不是生產數字**。
⇒ 名字表的 `wcntw` 也是同一個原因：`ns_kind_wns[q] += dur * wcnt[q] / wtot`（`ggml-backend.cpp:2306`）
把每個 buffer 的 `dur` **平均分給該 buffer 內有工作的節點**，所以 **`bufs` 越小／每個 buffer 的節點越少，
「落在小 buffer 裡的節點」就越會獨吞**。實證：`attn_norm` 被報成 **36.76 ms（8.5%）**，而它是
`MUL ne=[2048,2]` 的 4096 元素乘法。**⇒ 細粒度排名要讀 `CGC-GPUOPS`**（`nd=` 是確定值），
不要讀名字表。（lesson `eng-mh-0070`；白皮書 §25／§25.1／§26。）

**這一步只讀既有的 log**（`Backup/cgc_logs/*.log` 裡任一份帶 `CGC_GRPH_DBG` 的），不跑任何東西。
⇒ **通則：在「猜名字的代價是一整輪跑」之前，先問「我能不能從既有的 dump 讀出來」。**

⚠️ **兩個必須先知道的邊界**（`ggml-backend.cpp:2038-2057`）：

- dump 的條件是 **`cgc_grph_dbg_n < 6`**（`:2054-2055`）⇒ **只捕捉前 6 次 `graph_compute`**，
  而它們會被載入期的 warmup／prefill 吃掉 ⇒ **印出來的 `ne` 是 prefill 形狀**（不是 decode）。
  ⇒ **節點結構（名字、op、索引順序）可用；形狀不可用。** 要 decode 的形狀就得改那個條件，
  而那是動 `src/`。
- 它 dump 的是 **`split->graph`，也就是那個 split 的「完整」cgraph**（`:2057-2058`）——
  `gv0 = seg_view(0)` 只是它的 `ggml_graph_view`（`:2038-2043`），**不是** dump 的對象。
  所以**一個 dump ≈ 一步**（trunk 的 `n_nodes=4116` ÷ 41 層 ≈ 100 節點/層），**不是一層**。
  ⚠️ 2026-09-18 我在同一天先寫成「一個 dump ＝ 一個 segment（一層）」⇒ **正好說反**：
  拿一個 dump 的節點數除以層數才是「一層」，而不是把它當成一層。

### 步驟 2（要動原始碼）：在 builder 裡命名

只有當某個叢集**大到值得被單獨引用**時才做（本例：那條 240 核心的歸約鏈 ＝ **6.7% of `wait`**）。
改動是**只呼叫 `cb()`／`ggml_format_name`**，不碰圖的結構、不碰數值。實例（行號會漂）：

- `llama-graph.cpp` 的 MoE 合併迴圈 → `cb(moe_out, "ffn_moe_add", il)`。**寫在迴圈內**；
  最後一項仍會被既有的 `cb(moe_out, "ffn_moe_out")` 改名 ⇒ 實際 **6/7 生效**（`240 = 40 層 × 6`）。
- `models/delta-net-base.cpp` 的兩個 GDN 呼叫點 → `"gdn_state"`（K=1 路徑）／`"gdn_out_raw"`（K>1）。
  （K=1 那條**在 2-token 的圖裡不走** ⇒ 它會是 0 次，不要以為命名失敗。）
- ★ **詞彙表要同步加**（`ggml-backend.cpp` 的 `ns_fix[]`），否則新名字會被 longest-prefix
  兜底桶吃掉（`ffn_moe_add` → `ffn_moe_`、`gdn_*` → `(other)`），**等於沒命名**。

**命名之前必須驗的四條**（這是「我有沒有只是改了名字」的判準；四條都要查，不要用推論）：

1. `cb()` ≡ `ggml_set_name`／`ggml_format_name`（`llama-context.cpp`，搜 `cb = [`）—— 只寫名字。
2. 有沒有程式用「**名字為空**」當判準？搜 `name[0] ==`／`strlen(...->name)`；命中的要是**診斷路徑**。
3. 該節點有沒有進 `add_fused_node`？它**只 push_back、不讀名字** ⇒ 安全（要查，不要假設）。
4. 名字比對是 `strcmp` **精確**還是**前綴**？（前綴比對會被新名字意外命中。）

**證明是 D5 的 M1 逐位元 9/9**（`--tag <你的>`）：M1 不過 ⇒ 你不只是改了名字。

### ⚠️ 三個會讓人算出離譜數字的東西

- **`ne` 只印前兩維**（`ne=[ne0,ne1]`），而 MoE 的 token 數在 **`ne[2]`** ⇒ 用 `ne0*ne1*4`
  當「搬了幾 bytes」在 `n_tokens>1` 時**系統性低估**（decode 的 `ne[1]` 剛好等於 token 數，
  所以只在批次形狀露出來）。**prefill 的 dump 一律不能這樣算。**
- **`SET_ROWS`／`GET_ROWS` 的 `ne` 是整個張量，不是「這次搬的」**：`SET_ROWS ne=[512,8192]`
  是整條 KV cache，而 decode 每步只寫**一行**。實測我一度算出「335 MB/步」，複核時自己推翻
  （lesson `eng-mh-0063` 的形狀）。⇒ 要算搬運量，先問「**這個 op 是切片嗎**」。
- **`name` 是匹配規則的產物，不是語意的產物**：`ffn_moe_` 是 longest-prefix 的**兜底桶**，
  裝的是 `ffn_moe_swiglu`／`weights`／`weighted` ＋ **它自己的 8 個 VIEW**——**逐元素運算**，
  與同名的專家 GEMV（`ffn_moe_gate`／`up`／`down`，op 是 `MUL_MAT_ID`）**是兩回事**。
  照字面把「`ffn_moe_*` 那群」當成 MoE 矩陣乘，結論會**相反**。

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
  ⚠ **★ 2026-09-19 更正：這一招今天「只對一半」—— 步時上界有效，gap/union 與 t/s 無效**（見陷阱 28）。那個臂的 header 印
  `(NO TIMESTAMPS)`（25/28 步）、`mean len = 1.00`、生成截到 21 token ⇒ **既沒有 gap 讀數、
  也沒有可比的 t/s**（acceptance 崩掉）。**但它仍然重現了步時上界：169.21 → 99.39 ms = ×1.702**
  （兩臂 verify 圖寬都是 4）。⇒ 用上界探針時**逐欄位判定哪一個可以讀**：
  `grep -c "NO TIMESTAMPS"` 非 0 ⇒ 禁讀 `gpu/union/gap`；`mean len` 與 base 不同級 ⇒ 禁讀 `t/s`。
  **兩臂圖寬相同時步時比仍然合法**，而那正是這一招要的那個數。
- **★ 一個新的歸屬份額（任何 weight／攤分模型）在能被引用之前，先過「換粒度」測試**：
  同 profile、**同熱態**跑粗／細兩臂，逐項比。
  ⚠️ **判準是 `thermal_hist`（全程分佈），不是 `thermal_launch`（單點）**（2026-09-18 實測）：
  兩臂的 `thermal_launch` 可以**都是 HEAVY**，而 `thermal_hist` 一邊 `{HEAVY: 8}`、
  另一邊 `{HEAVY: 1, MODERATE: 7}` ⇒ 後者的 median 高了 **69%**（4.69 vs 2.78 t/s），
  而池統計完全相同（hit 82.7% 對 82.7%、io 1967.3 對 1966.2 MB）⇒ **那個差異全是熱態**。
  **不是同一種熱歷程的兩臂不可比，即使 `thermal_launch` 相同。** 要嘛等到穩定、要嘛看 `thermal_hist`。
  不通過的項目就只報家族＋區間，不要報點估計（lesson `eng-mh-0064`）。
  同粒度下 run-to-run 穩定（實測 11 個家族 0.89–1.02）**不代表**跨粒度穩定。
- **M3 的 40% 判準已經有三個獨立方法都是否定的**（2026-09-18）：聯集上界 ≤28.3% of busy、
  op 表 work-weighted 8.0／8.4%、名字表 work-weighted **3.2–6.3% of `wait`**。
  ⇒ **Cell 2（MoE gather 融合／batched-union）沒有量，不要再投。**
  而「MoE 的**逐元素合併**（`ffn_moe_`，7.6%）比 MoE 的**專家矩陣乘**（3.2–6.3%）還大」。
- **★ 統計的可採性（admissibility，2026-09-18 定義）：一個 `t/s` 或一個 `r` 在能被引用前要過三道**
  ① **同一個量**：可同池 ⟺ `cell_key = (profile, metric, n_prompt, n_gen, depth, n_batch, n_ubatch,
     spec_state, pool_bytes)` 逐項相等。判準是**量綱**（y 軸標籤必須每列一樣）——
     **prefill(`prompt_per_second`) 與 decode(`predicted_per_second`) 永遠不可同池，與 n 無關**。
     混池會把 |r| **抬 2–3.4×**（實測 `inactive` **+0.182 → +0.628**），而該 n 的臨界是 0.631
     ⇒ **只差 0.003** ⇒ **危險不是「它過了」，而是「它被抬到臨界線邊上」**；
     ⇒ 規則的正當性**不可依賴當下有沒有過線**，否則它會隨資料漂移。
  ② **三態**：`decided`（key 完整 ∧ n≥3 ∧ |r|≥r_crit）／`suggestion`（|r|<r_crit）／
     `cannot decide`（key 不完整 ∨ n<3 ∨ 沒有 arm 同時帶吞吐與環境讀數）——第三態是**要出聲的結果**。
  ③ **裁決要機器可讀，而且要附「餵了哪些檔」**：**一句 caveat 不是閘門**。輸出要含
     `{cell_key, metric, n, r, r_crit, verdict}` 與 `--json-glob` 的實際值
     （實例：一個「n=8」因為沒記 glob，多花一輪才復現）。
  工具：`scripts/check/caliber_env.py --memory --cell-filter <cell> --json-glob …`（2026-09-18 新增）。
  ⚠ 它**尚未**有「預設拒絕」與 `--json` ⇒ 目前是**榮譽制**，引用前自己確認三態。定義全文：
  `docs/HTTP_VS_BENCH_CALIBER_2026-09-18.md` §8（含四條可機檢條件 H1–H4）。
- **★ 說「兩條路只差在量測路徑」之前，先跑 `caliber_env.py --equiv`**（2026-09-18 實測打臉）：
  **prod25 判 NOT CONFIG-EQUIVALENT** —— server 側 `-b/-ub` **完全沒給**、bench cell 硬編碼 `-b 512`
  （`prefill250` 下同一格是 5632 vs 512，差 11×）。根因是 `prod_matrix.py:319` 的
  `b = ub = spec["batch"]` 蓋掉 `resolve()` 從 profile 算出的 BATCH/UBATCH。
  ⇒ 對齊前，跨路徑的 t/s 差只能叫「**路徑 × batch**」的合併效應，不能叫口徑差、更不能拿去換算。
  **另記**：`CGC_SERVER_MTP=0` 換出去的兩份 gguf 是**同一份 bytes**（size ＋ 5 個視窗 shasum 全同，
  不同 inode）⇒ 模型檔不是 confound；但 MTP=0 會讓 **8 個 engine env 整塊消失**
  （`CGC_MM_BITIDENT`／`CGC_DRAFT_DECODE`／`CGC_VERIFY_DECODE`／`CGC_NO_PREFETCH`／layer caps…）
  ⇒ **「MTP off」是一整組旋鈕，不是一個旗標**。

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

**★★ 2026-09-18 補充（MTP 的池成本儀器 — 已存在，不要重寫）：**

要問「verify 的 union 有多少複用」時，**儀器早就印了**，位置是
`src/llama.cpp/src/llama-expert-cache.cpp:2529`（**不是 env-gated**，只在 `n_fast_calls > 0` 時於結尾印一次）：

```
llama_expert_cache: MTP fast path: calls=510 union=9316 cold(ZERO)=0 (0.0%)   verify: calls=474 union=9028 cold=0 (0.0%)   draft: calls=36 union=288 cold=0 (0.0%)
```

- `n = uni.size()`（`llama-context.cpp:6303` 傳的是去重後的 union）⇒ **`union` 已是去重後的個數**。
- **讀法**：`verify 的 union / verify 的 calls` = 每次 verify 的平均 union；**除以 `(1+n_max) × top_k`**（我們是 `4 × 8 = 32`）
  ＝ 複用率。實測 **9028/474/32 = 0.595** ⇒ 4 個 verify token 之間有 **40.5% 的專家複用**。
- **兩個現成的內部核定**（讀之前先跑）：`draft` 的 `union/calls` 必須 **＝ top_k**（我們 288/36 = 8.00 ✓）；
  且 `union` 必須 `< (1+n_max)*top_k`（19.05 < 32 ⇒ batch 真的是 4）。
- ⚠️ **陷阱（會讓兩份量測互相矛盾）**：**llama-bench 路徑下 `verify: calls = 0`**（所有 call 都被算進 `draft`），
  因為那條路的 verify 走 `ensure_batch`（真實填充 + LRU），**而 server 路徑的 verify 走 fast path（`touch`，no-fill）**。
  ⇒ **MTP 的池成本在兩條路上結構不同，不可並排**（`MTP ≈ 0`（llama-bench −4~−6%）與 `+5.7%`（HTTP）的落差有這個成分；
  **不是**跨模型檔造成的 —— 見下一條，那兩份配對其實都是同檔的）。
- **成本的主因不是單次 union**：`cold(ZERO) = 0` ⇒ 單次 verify 不打穿池。真正在動的是
  `layers_distinct_over_slots`（**3 → 11／18**）與 `capacity` miss（**218 → 2229，×10.2**）
  ⇒ **長期工作集超過 143 slots** ⇒ 對症的是「改淘汰／填充策略」，不是「增大池」。
- ⚠️⚠️ **陷阱（一句話就會把「MTP off/on」變成跨模型檔比較）**：`run_server.sh:153-161` 是
  `MODEL_DEFAULT="$Q36"` ＋ `if [ "$SERVER_MTP" = "1" ] → "$Q36_MTP_DENSEIQ4X"` ⇒
  **只設 `CGC_SERVER_MTP=0` 就會把模型換成 `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`**（沒有 nextn 頭的另一個檔）。
  `CGC_DUMP_ENV=1` 逐字驗過（兩次的 `CGCENV MODEL` 不同）。
  ⇒ 影響面：`decode_sweep.py` 的 `p25-mtpoff`／`p25-gputime`／`p25-mtpoff-phase`／`p25-phase-w32`
  與 `llama_bench_matrix.py` 的 `prod25-stream-mtpoff` **全部跨檔**；
  `mtp_accept_ab.py` 的 `nail_nomtp`（`:122` **顯式釘 `MODEL=Nail`**）與 llama-bench（`-m` 顯式）
  **才是同檔** ⇒ 只有它們的配對算數。
  ⇒ **要同檔配對就必須顯式設 `CGC_SERVER_MODEL`**（`decode_sweep.py` 的 `p25-nail-mtpoff` 就是為此存在）。
  ⇒ 任何 MTP off 臂**開跑前**先比兩臂 ctrl log 的 `CGCENV MODEL` 那一行；跑完也要比。

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
2. ~~**它對 MTP 是瞎的。**~~ **2026-09-19 更正：這一條已作廢，別再引用。**
   `8194a2ba4`（09-18 20:24）在 `test_gen_spec` 補上 `llama_context_set_cgc_phase(ctx, CGC_PHASE_VERIFY)`
   （＋decode 後 fail-closed reset）。病因是 fast path 的閘門為 **caller 設的 phase**
   （`llama-context.cpp:6065`），而全樹只有 `server-context.cpp` 與 `speculative.cpp` 兩個 caller
   ⇒ 修前 bench 的 verify 批次一律掉回 `ensure_batch` 精確路徑。
   **已驗證生效**：四支帶 spec 的 run `verify: calls` 全部非零、`union/call` **16.30–18.15**
   （server 側參考值 18.69）；修前症狀是 `verify: calls=0`、`union/calls = 8.00`。
   ⇒ 現在 `--spec-type draft-mtp --spec-draft-n-max 3` 在 llama-bench 上可直接量 MTP。
   ⚠ 只有**修復前**（09-18 上午及更早）量的 bench MTP 數字要打折；
   `m=0.474`（09-18 §EN-187）等在此之後 ⇒ 不受影響。
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

**★★★ 但 `llama-bench` 的 MTP 只接了一半：它的 verify 不走生產的 fast path（2026-09-18 19:0x）**
`llama-context.cpp:6065-6090` 的門是
`getenv("CGC_VERIFY_DECODE"|"CGC_DRAFT_DECODE") && cgc_phase ∈ {VERIFY,DRAFT} && ctx_type/n_tokens 條件`，
而 **`cgc_phase` 由呼叫端在每次 `llama_decode()` 前設定**（該處註解逐字：*UNKNOWN is the safe default:
when caller forgets to set phase, we fall back to exact path*）。**`llama-bench.cpp` 對 phase 的引用是 0**
⇒ bench 的 target 多 token verify 永遠是 `UNKNOWN` ⇒ 走 `ensure_batch` 的精確填充路徑。
日誌直接印出來（`Backup/prod_matrix/20260918_15*_prod25_decode-spec/*.stderr.log`）：

```
MTP fast path: calls=395 union=3160   verify: calls=0 union=0   draft: calls=395 union=3160   ← llama-bench
MTP fast path: calls=9497 union=170202 verify: calls=8819 union=164778 draft: calls=678 union=5424  ← server
```

`union/calls = 8.00` ⇒ bench 那 395 次**全是單 token 的 draft**，多 token verify 一次都沒進來。
⇒ **bench 的 verify 與生產的 verify 走不同分支**（server 的 verify 是 18.69 experts/次、走 fast path）。
⇒ 這才是「bench 量到 MTP ≈1.0 vs server 量到 ×0.695」的真解釋：**不是量到不同的數字，是量到不同的東西**；
也解釋了上面那格 `ms_per_round` 為什麼「只是 draft head」——**verify 那一段在 bench 裡沒被分開量**。

**⚠ 但「走 ensure_batch」≠「一個一個丟」（19:1x 更正一處容易讀錯的措辭）**
兩條路都**用本層 union 整批**處理：`llama-context.cpp:6237-6260` 的 union 在分支**之前**就算好，
非快路徑是 `ensure_batch(cache, il, cold.data(), cold.size(), …)` —— **一批一次**；
而「串行」是**可切換的舊行為**（同處註解：`CGC_SYNCFILL_SERIAL=1 restores the old serial loop for A/B`），
**預設批次，且兩次量測都沒設它**（`run_server.sh:1504-1511` 只在明示時轉發）。
日誌的填充粒度兩邊逐位相同：**`read shape` 的 job 都是 0.37 MiB**（bench jobs=39282／server 8400）。
真正的差別是**策略**：fast path（`touch`）只 LRU-touch **已 resident** 的專家、**不 fill 不等**；
exact path（`ensure_batch`）**先把 cold 補齊再寫 remap**（有 IO／等待）。
**這兩條的貴賤沒有量過** ⇒ 只能說「bench 的數字不可搬去生產」，**不能**說 bench 偏樂觀或偏保守
（那個 14.08 GiB vs 3.01 GiB 是兩個不同 fixture，不能正規化成「每 token」比）。
（要判貴賤，得先有下面那支 ctx 標籤。）

**⚠⚠ 這個洞不是「傳個參數」能補的，而且修法方向容易被想反（19:1x）**
- **`--spec-type draft-mtp` 不是那個開關**（它有效，MTP 真的跑了）。它管的是**引擎/模式**；
  缺的是 **phase**，而 phase **沒有任何 CLI／env 表面**：唯一入口是
  `llama_context_set_cgc_phase(ctx, phase)`（`llama-ext.h:154` 的 `LLAMA_API`），
  **每次 `llama_decode()` 前由呼叫端設**。`CGC_VERIFY_DECODE` 只是 `getenv(...) != nullptr` 的**許可**。
  全樹呼叫者只有兩處：`tools/server/server-context.cpp:3897`（server）與
  `common/speculative.cpp`（**只設 `ctx_dft`**：DRAFT／CATCHUP／UNKNOWN）；**`llama-bench.cpp` 零**。
- **修法方向**：verify 要進的是 **`verify_fast`**（`phase==VERIFY && ctx_type==DEFAULT`）——
  **不是** draft 分支（`draft_fast` 要 `ctx_type==MTP && n_tokens==1`，verify 是 DEFAULT 的多 token）。
  作法＝在 target verify decode 前呼叫同一支 API，**分類邏輯照抄**
  `server-context.cpp:3845-3897` 的 `classify_batch_phase()`（掃 slot 狀態、**`n_phases==1` 才回傳**、
  混了回 `UNKNOWN` fail-closed、**phase 在 batch 層算一次**而不是每個 view 重算）。
  **不要另寫一份判斷**（兩份真相必然漂移）。
- **★ 為什麼這個洞能存在**：`common/speculative.cpp` 這支**共用** spec 庫**本身不設 VERIFY** ⇒
  就算讓 bench 改用 `common/speculative` **也不會**解決；VERIFY 只存在於 **server 私有 decode wrapper**。
  ⇒ **洞在「共用庫」與「server 私有 wrapper」的交界**；這類交界的東西最容易兩邊都以為對方有做。
- **第二個會擋的閘門**：`cgc_fast_eligible` 還要 `!warm_gate`（`CGC_WARM_NPAST` **預設 2048**），
  所以 `-d 512 / -d 0` 這種 bench cell 就算 phase 設對也會被擋。**但實測 bench 的 env 已是對的**
  （日誌 `CGC-WARM verify n_past=0 warm=0`；`run_server.sh:2035-2040` 在 `denseIQ4X=1` 時顯式設 0，
  因為 bench arm 吃的是 `resolve()` 的同一份 env）⇒ **缺的只有那一行呼叫**。

**★ 啟動必跑參數的權威清單（不要背清單，去問腳本）**：
`CGC_DUMP_ENV=1 CGC_SERVER_PROFILE=prod25 ./scripts/run_server.sh`（**只解析、不起 server**，
`run_server.sh:746-804`）逐行印 `ENV …`／`ARG …` ⇒ 那是完全解析後的清單；
**不在 `SERVER_ENV` allowlist 的變數會被靜默丟掉**，所以「文件上寫的」≠「真的傳進去的」。
三層分類（必跑 7 個 env／效能口徑／三個通用陷阱）、以及 **`--temp` 在 argv 裡出現兩次 ⇒ 後面的 0.4
生效、MTP 塊的 `--temp 0` 是死旗標**（server 自己會印 `W DEPRECATED: … only last value will be used`），
全部記在 `docs/MTP_LAUNCH_REQUIRED_PARAMS_2026-09-18.md`。
**不要在本 skill 再抄一份清單** —— 兩份必然漂移（本 repo 已為此付過學費）。
**機檢判準**：清單只證明「應該傳了什麼」；有沒有接上要看 launch 後的
`MTP fast path: … verify: calls=<>0> … draft: calls=< >0>` —— **兩類都要非 0**。

**★ k 的可掃旋鈕 ＝ `CGC_SERVER_MTP_N_MAX`（免改 `src/`，prod25 內部寫死 3、外部可覆寫）**。
2026-09-18 19:04–19:05 **同小時 A/B**（同命令、同請求）：

| 臂 | mean_len | ms/step | verify union/次 | **experts/token** | t/s |
|---|---|---|---|---|---|
| MTP off | 1.00 | 85.1 | —（無 fast path） | **8.00** | 11.75 |
| **k=1** | 1.72 | 167.3 | 12.27 | **7.13** | 10.28 |
| **k=3** | 2.31 | 259.5 | 18.61 | 8.06 | 8.90 |

- ⇒ k=3 步時比 k=1 高 55%、每 token 收益只多 34% ⇒ **k 的最優點在 3 的左邊**（k=2 未量）。
- ★ **k=1 的 experts/token 7.13 ＜ MTP off 8.00** ⇒ 投機在「專家讀取」軸上**本來有賺（−11%）**；
  它仍然更慢（10.28 ＜ 11.75）⇒ 差額只能是**每步多跑的那一次前向（draft pass）**。
  ⇒ **轉正判準可以寫死：`draft pass 的固定成本 ＜ verify 省下的 expert 讀取成本`**。
- **陷阱（我自己踩過）**：拿不同時段的 `ms/step` 相減會得到相反的結論。跨環境比會給出
  「union −34% 而步時不動」的假象；**同小時**一比，步時確實隨 union 漲（+19.1／+14.5 ms per expert）。
  **同 config 跨時段離散可達 ~1.4×** ⇒ 單點 MTP 數字不可引用，只認配對設計。
- **缺的儀器（「verify 還能省多少」的前置）**：`llama-context.cpp:1971` 的 `CGC-PHASE` 已印每次 decode 的
  `compute=`/`n=`，但**沒有 ctx 標籤** ⇒ MTP-on 每步兩次 decode（draft `n=1`、target `n=1+k`）
  **混在同一個平均裡**。加 ctx 標籤（或按 `n_tokens` 分桶）之前，那個問題無法回答。

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
⇒ **但它們不是只能忍的**：上面的「匿名節點：先零成本歸屬，需要時才命名」給出兩步——先用
`CGC-GRPH` 的**索引順序＋左右鄰居**免費歸屬（2026-09-18 把 trunk 的 620 個降到 350），
再對值得引用的叢集做 builder 命名（**也要驗那四條安全性**，並以 **D5 M1 逐位元 9/9** 為證）。

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


### ★ 第五條規則（2026-09-20 新增）：**捕獲表不是「這一輪的」** —— 讀之前先驗讀取的前置條件

任何「在 build 時把 tensor 指標存進一張表、稍後再讀它」的探針（`cache_slots_out_tensors` /
`cache_slot_table_tensors` / `cache_remap_tensors` 就是），都會遇到同一件事：**那張表只在它自己
建的那種圖上被填，而且從不清空**；而 ggml 每個 build 都 reset 並**重用 arena**。

⇒ 舊指標會落在**當前圖的別的張量**上，於是：

- **「這個位址是不是當前圖的節點」不是有效判準** —— 2026-09-20 實測：它對 39 條陳舊條目
  **全部放行**（同一次啟動的兩次呼叫，第一次 39 條形狀全對、第二次同一批 key 全錯）；
- 真正的判準是**讀取本身的前置條件**。要讀 `n` 個 int32 就要求
  `ggml_nbytes(t) == n * sizeof(int32_t)`（連續 ＋ 4-byte ＋ 長度對），因為 `ggml_nbytes()`
  是**由 `nb[]` 算的**：落在 F16／量化張量上時 `4n` 會超過它自己的位元組數 ⇒
  `ggml-backend.cpp:349 GGML_ASSERT(offset + size <= ggml_nbytes(tensor))` **abort**。

價格：一條純診斷指令 abort 之後，**它正在量測的那一輪就沒了**，而預設的讀法是把責任歸給受試物。
實例：`CGC_S1_DBG` 的 POST 探針被記成「S1 會 abort ⇒ S2 卡在一個要先修的 defect」，
而同 build、只把該旋鈕關掉的對照是 **PASS**（`comparable=true`、M1/M2/M3 各 9/9）。
lesson `eng-diag-0037`；根因報告 `docs/S1_DIAGNOSTIC_ABORT_ROOT_CAUSE_2026-09-20.md`。

**殘餘（明寫）**：那個判準讓讀取**安全**，沒有讓它**可歸屬** —— 長度對得上的舊指標仍會產生
錯誤的報告行。要關掉它需要在 capture 端記下**建置世代戳記**（「是哪一次 build 寫的」）。

### ★ 第六條規則（2026-09-20 新增）：**引用 decode t/s 之前先指名 cell —— 交付形狀不是預設形狀**

`t/s` 是**一個 cell** 的讀數，不是引擎的性質。本 repo 有兩個 decode 入口，而**同一個 profile**
在兩者之間差約 **1.4×**：

| 入口 | cell | 同一 profile 讀到 |
|---|---|---|
| `profile_duo.py` / `prod_matrix.py` 的 `decode` | `-p 0 -n 128 -d 512 -b 512`，**無 spec、無 warm-skip、
  ctx 由 llama-bench 自行推導 ~704** | **7.96–9.12** |
| **交付形狀**（`scripts/check/prod_profile.py` 的 `decode-delivery`） | 上列 ＋ `--ctx-size 4096`、
  `--warm-skip 64`、`--spec-type draft-mtp` | **12.57**（09-20，`NOMINAL` 全程） |

⇒ **交付 decode 的四個約定**：`--batch 512`、`--ctx-size 4096`、`--warm-skip 64`、
`--spec-type draft-mtp`。**任缺一個，那一列就不是交付 decode。**
⚠ 所以「記錄 7.7–10.8、今天 9.90」**不是退步** —— 那是**兩個 cell**。
（`--ctx-size` 特別容易忘：llama-bench 的 `n_ctx = p+n+d ≈ 704`，而生產 server 跑 4096；
KV 配置與 batch 夾制都吃它。`--warm-skip N` 讓時鐘在 N 個 token 之後才起算，報告的 `n_gen` 排除它們。）

**★ 同一條命令的單臂噪音 ≈ ±27%。** 2026-09-20 實測：**同一次 session、同一個命令**（只差 `--arms`
的名字），讀 **9.90**（`worst=MODERATE`）與 **12.57**（`worst=NOMINAL`）—— 更低的那次是更熱的那次。
⇒ **小於 ~27% 的效應，單臂前後對比證明不出來**；只能用配對交錯 A/B（AB/BA ＋ `median(A/B)`），
而第一步永遠是 `--null`（兩槽同 binary）。這與 09-18「四臂全 launch=NOMINAL 卻給 10.80/10.34/8.92/8.58」同向。

權威出處：`MEMORY_PERF.md` 的 profile 節（交付 decode＝12.57、四個約定、±27%）與
`docs/PRODUCTION_PROFILE_2026-09-20.md`（旋鈕全文、同一性證明、重現命令）。

### ★ 第七條規則（2026-09-20 新增）：**你自己的命令行會把你的閘門關掉**

`run_server.sh` 的 preflight 用 `pgrep -f` 找「別條 session 的 llama 行程」，而 **`pgrep -f`
比對的是整條命令列** ⇒ **只要你自己那一條 shell 的字串裡出現 `llama-server`（或 preflight 比對的
任何字串），它就會匹配到你自己的包裝 shell**，然後拒絕啟動：

```
[preflight] 發現 1 支 llama 行程（可能是別條 session 正在量測）→ 不送任何訊號
  [preflight]                      <-- 這一行的 pid/etime 列表是空的
error: 仍有 1 支 llama 行程，繼續啟動極可能 GPU OOM (ret=-3)
[detach] 120s timeout
  leader  : None   server log: None
```

**辨識簽名 ＝ 那個列表是空的。**（真的有競爭者時它會印出 pid ＋ etime。）
實測 2026-09-20：`m123_oracle_gate.py --tag s2-caps2` 就是這樣被卡死的，
而我當下「殺掉的殘留 pid」其實是**我自己的 shell**，**窗口一直是空的**。

規則：
1. **診斷命令一律用不會自匹配的寫法** —— `pgrep -f '[l]lama'` 而不是 `pgrep -f llama-server`。
2. **看到空的 pid 列表就當作「沒有競爭者」**，重跑或（確定是自己一個人在用機器時）
   `CGC_PREFLIGHT_SKIP_STALE_CHECK=1`。
3. 這一條與「`ps` 被擋時用 `pgrep`」是**相反的風險**：`pgrep -f` 太好匹配了。
   任何「我用它來判斷窗口」的指令，**先問它會不會匹配到我自己**。

## 陷阱（都踩過）

**★ 陷阱 0（2026-09-18）：`ls -t | head -1` 取到的「最新產物」可能還沒寫完。**
驗證一次跑完的結果時，若 driver 還在跑，最新的 log／json **只寫到一半**，而計數會少。
實例：IQ3_S 融合的首次驗證，我用 `ls -t Backup/cgc_logs/llama_server_*.log | head -1` 讀，
只看到 **10 行** `CGC-DCFUSED`（全是 il=40）⇒ 我寫下「trunk 仍然沒融合」並開始找原因；
等 driver 結束後同一份 log 是 **每層 24 行 × 40 層**，逐層與 audit 的 `fuse=1` **完全吻合**。
**判別式**：讀之前先 `pgrep -f '[d]ecode_sweep'`（或 `[l]lama-server`）。還在跑 ⇒ 要嘛等，
要嘛把「看到的行數」與「你預期的行數」對一眼（差一個數量級就是這條）。
⇒ 通則：**「取到最新產物」與「產物已經寫完」是兩件事**；前者是 `ls -t`，後者要問 driver。
（同一族：`grep -c` 對一個還在成長的檔案，數字沒有意義。）

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

21. **`CGC-DECPROF` 逐層表有三個坑，每個都會製造「所有層一起變差」的假象。**
    用 `scripts/check/decode_layer_cb.py`（2026-09-19，9 項自測），**不要手寫解析**。
    - **欄位**：`CGC-DECPROF all: L<k> wait=<w> cb=<c> submit=<s>` 裡**第 2 個數是 `wait`**，
      `cb` 是第 3 個。取錯得到 2–3.4 ms 的「cb」，而真值在 0.1–1.2 ms。
    - ★ **`ntok` 是「這張圖的 token 維度」（draft 寬），不是「接受了幾個 token」—— 不能拿它算 `mean_len`。**
      定義在 `ggml-backend.cpp:2493-2500`：`dp_t = ttopk->ne[1]`，註釋自己寫「`ntok=2048` ⇒ prefill、
      `ntok=1` ⇒ decode」。它**很好用**當 prefill/decode 的判別器，但 MTP 開啟時它幾乎恆等於
      `1 + n_max`（實測 E2b：**n=4 佔 179/199 步**），所以 `sum(ntok)/步數 ≈ 4`，不是 token 率。
      **`mean_len` 要讀 server 自己的 `mean len =` 行**（per request，由 acceptance 導出：
      `mean_len = 1 + n_max × acceptance`，例如 `acceptance 0.4654`、`n_max 3` ⇒ 2.40）。
      2026-09-19 我曾經用 `http_duo 的 t/s ÷ DECPROF 步/秒` 反推出 `mean_len ≈ 1.80`，
      並據此宣稱「到不了 25」—— **那是窗口／算術產物，已撤回**；正確值是 **2.40**。
    - **局部 profile 不是步**：引擎另外會印 `step=91 segs=2 layers=1 total=2.00 ms … ntok=4`
      這種單段 profile，其 ntok 過得了 decode 篩選 ⇒ 每筆塞進一個假的「2.00 ms decode 步」，
      把 request 的步數從 ~64 灌到 149、step 級中位數從 ~130 ms 壓到 2 ms。
      **只收 `layers == 40` 的 header。**
    - **request 邊界是「prefill→decode 的那一步」**，不是每個 prefill chunk（一個 request 的
      prefill 有 ~8 個 chunk ⇒ 3 個 request 會數成 24 個）。而且 **http_duo 的 log 有 4 個 request：
      第 1 個是啟動 anchor（1 步），第 2/3/4 才是 rep1/rep2/rep3** ⇒ kept 的是 3 與 4。
    - **一個崩潰的 request 能讓「全 run 中位數」對每一層同時說謊**（2026-09-19 E2：第三個 request
      崩潰 ⇒ 跨 request 中位數顯示 40 層全部 0.2–1.2 ms，而同一次 run 的穩定態是「零 loud 層」）。
      **一律先分段再取中位**；而且要檢查「loud 層有沒有搬家」—— 只看原本那幾層會漏掉水床效應
      （E2 的 warmup 有 10 層接手，穩定態才收乾淨）。
    - **`gpu/union` 與 `union` 正交，不要拿它當「還有多少可壓縮」的量度**（2026-09-19 自我否證）。
      逐層欄位是 `wait= cb= submit= ms gpu= union= gap=`（順序固定）。`union` 是該層佔用的 GPU
      窗口，`gpu` 是它的 buffer dur 之和（並行 ⇒ 重複計數 ⇒ 同一 step 常見到 `gpu_sum > union_sum`）。
      實測對照：`L1` union 2.82 / gpu 2.82 / ratio **1.00** 對 `L2` union 2.80 / gpu 4.10 / ratio **1.46**
      —— ratio 變了 1.46 倍，**union 2.82→2.80、gap 0.43→0.43 都沒動**。⇒ `gpu/union` 只是
      「這層的工作分在幾個 buffer」，**不是**重疊空間。而且 linear 層本身就分兩半（19 個 1.00、
      11 個 1.41–1.46，後者每 4 層一個、緊鄰 full_attn）⇒ 看起來像**架構特徵**，先問架構再當優化。
    - **逐層表的每個欄位都要先確認口徑再引用，而 `gap` 的「不閉合」是假警報（2026-09-19 更正）**：
      header 的 `gpu_sum/union_sum/gap_sum` **就是**逐層 `dp_lay_*` 的加總（`ggml-backend.cpp:2579`
      `for li { dp_gg += dp_lay_gap[li]; }`），**同一步內閉合到 0.01%**（184 步，max 0.25%）。
      先前報的「21.43 vs 34.76 ms、不閉合 39%」是**拿「每層中位數之和」去比「每步和的中位數」**：
      **中位數不可加**（各層的高 gap 不在同一步）。⇒ **G1 的 metric 一律讀 header 的 `gap_sum`。**
      判 overlap 值不值得，實驗要挑一對相鄰的 `ratio=1.00` 層（`L4/L5`、`L8/L9`），只改這一對的
      提交方式，量**同一批步的逐層 union**：合起來 < 2×單層 ⇒ 有效；不變 ⇒ `union` 由計算量決定。
22. **`LAYER_CAPS` 不是「只改哪些專家常駐」的免費旋鈕 —— 它會改變 logits 的數值。**
    2026-09-19，prod25、9 步 probe：**同配置跑兩次的空對照 M1 9/9（逐位元相同）**，
    而 **A（均勻 143）vs E2（6816 槽重分配）只有 M1 1/9**、M2 9/9（argmax 沒翻）、
    `d_mean` 0.04–0.25（最大約 12%）。step 0（prefill 的 logits）逐位元相同，只有 decode 步漂移
    ⇒ 與「caps 只作用在 decode 的 gather 路徑，prefill 走 slab」一致。
    ⇒ 要宣稱某個 caps 配置可用，**先跑 A-vs-A 的空對照排除儀器噪音，再跑 A-vs-candidate**；
    **長 probe 定案（同日 21:1x）：`LAYER_CAPS` 會改變模型輸出，它不是等價旋鈕。**
    把 probe 從 9 條拉到 884 條（見下面第 23 條）之後：**空對照 A-vs-A2 仍然 884/884**，
    而 **A-vs-E2 掉到 M1 1/884、M2 25/884、859 條真實分歧**（生成長度都變了：940 vs 884）。
    ⇒ **短樣本的「M2 全同」會把「已經分岔」看成「決策一致」**，那是這條路徑上最貴的誤判。
    機制不是冷 expert 被丟棄（`cold(ZERO)=0.0%`、`zero_mapped_selected=0` 兩支都是 0）。
    ⚠ **「槽位佈局改變歸約順序」這個假說已被否證（同日 22:0x–22:2x），兩條證據都不要引用舊句**：
    (a) **canonical 歸約順序修不了它** —— `CGC_CANON_ORDER=1` 之下 `canonA vs canonE2`（只差 caps）
    仍是 **M1 1/968**，onset 與 canon=0 逐值相同（index 0 逐位元相同 → index 1 出現數值差 →
    index 5 出現 argmax 差）；而 canon 本身確實生效（同 caps 下 off vs on 在 index 0 就分歧）。
    (b) **`ne[2]` 不是載體** —— pool 由 8 GiB 砍到 4 GiB 讓每層 slot 由 145.8 掉到 75.5（`ne[2]` 砍半、
    總數 5976 → 3096），輸出**逐位元相同**（884/884）⇒ 形狀不是載體，M1 的「跨 pool size」在 884 步下成立。
    ⇒ 「權重相同」不等於「浮點結果相同」仍成立，但**修法既不在歸約順序、也不在張量形狀**。
    未排除的只剩「caps 改變每層走哪條計算路徑」⇒ `docs/CANON_CAPS_NEGATIVE_RESULT_2026-09-19.md`。
    ⚠ 引用 logits 差值要用 **per-logit**（首個分歧行 0.0411），別用 `sum` 的 ~20%（那是 248,320 個
    有正負項總和的相對差，被簽號相消放大）。
    入口：`Backup/oracle_caps_longprobe_20260919.sh`。
    **另外 `--port` 不要傳**：它現在預設取 profile 的 `CGCENV PORT`，不符會在 launch 前 abort
    （2026-09-19 修掉的缺陷：原本傳錯埠會產生偽 dump ＋ 乾等滿 300 s ready-timeout ＋ 清錯 listener）。

23. **要判「某個旋鈕會不會改變決策」，樣本數本身是變數 —— 短 probe 會系統性報「沒差」。**
    引擎的 oracle `CGC_LOGITS_ORACLE_FIRST_N` 預設 unlimited，但**預設 probe（`15+27 等於多少？`）
    答案是「42」然後 EOS，永遠只給 9 條記錄** —— 那不是上限在截，是模型自己停的。
    ⇒ `m123_oracle_gate.py` 已加 `--probe-prompt` / `--probe-max-tokens`（預設值維持舊行為）；
    要判決策就用會持續生成的 prompt ＋ 數百 token（實測 400 tokens ⇒ 884 條）。
    兩次同配置的長 probe 連**池統計都逐位元相同**（`requests/hits/misses/resident/owner-set` 全等）
    ⇒ 這台引擎在固定配置下是完全確定性的，**空對照一定會過**；空對照不過就別往下比。

24. **共享機器上，清理程式碼本身就是一個攻擊面 —— 連「不是 server 的工具」也要查。**
    2026-09-19：`m123_oracle_gate.kill_servers(port)` 對 llama-server 已正確地按 port 收斂，
    但**同一函式裡留著無條件的 `pkill -9 -f llama-bench`**，而 llama-bench 沒有 port 可收斂。
    後果：**任何人跑一次 oracle gate，就會殺掉別條線正在跑的 t/s 量測**（`profile_duo.py` 正是用它），
    而受害者看到的是一支死掉的 bench，不是一個死掉的 gate。
    ⇒ 規則：**清理只能指名自己的 pid／port**；不能指名的（依 binary 名字匹配的）預設**只列出 pid+argv、
    不送訊號**，要殺得明示（本 repo 的 `CGC_PREFLIGHT_KILL=all`，`run_server.sh` 同款）。
    落地：`scripts/check/m123_oracle_gate.py` 已改成列表 ＋ opt-in。
    ⇒ 一般化：**改任何「收尾／預檢」程式碼之前，先問「它會不會匹配到別的 session 的行程」**，
    而判準不是「我自己的殘留」而是「這台機器上還有誰」；`--port auto` 的兩個 session 可以並存，
    但**共用同一顆 GPU 的量測不能同時跑**（兩份資料一起毀）。

25. **新增一個會改變排程／圖形的旋鈕時，必須同時給它一行啟動回顯 —— 否則那一臂的實驗不可歸因。**
    2026-09-19：`run_server.sh` 的 `[perf]` banner 印了 `n_cb / glu_fused_down / oa_async /
    load_mode / layer_caps`，但**沒印** `CGC_SUBMIT_AHEAD`、`CGC_SLOT_TABLE_GPU`、`CGC_CANON_ORDER`、
    `CGC_POOL_SPLIT_DBG` —— 這四個都在 allowlist 裡、都會改變圖或提交順序，卻在**任何 log 裡都看不到**。
    後果：若某一臂量出「沒差」，你**分不出「旋鈕沒生效」與「生效了但沒效果」**（allowlist 會靜默丟棄
    未列出的 `CGC_*`，而這兩者留下的產物一模一樣）。
    修法（已落）：`run_server.sh:1098` 之後加一行
    `[perf]  diag: submit_ahead=… slot_table_gpu=… canon_order=… pool_split_dbg=… s1_dbg=…`。
    ⚠ **回顯清單只能放 allowlist 真的轉發的旋鈕** —— 放一個會被丟棄的（如 `CGC_TOPK_BOUNDARY`）
    等於讓 banner 對「你要求了什麼」說謊，比沉默更糟。
    ⚠ 而且**不要編輯正在執行的 shell 腳本**：bash 邊讀邊跑（記 byte offset），改到一半會讓它執行到
    錯位的內容。要改就等那支跑完，或把改動寫進另一支新腳本。

26. **用文字插入改 JSON（本 repo 的既定做法）時，驗收要看「欄位集合」，不是「能不能 parse」。**
    2026-09-19 實例：`targets.json` 的 G6 用 `str.replace` 插入一個新欄位，**改了 `old` 的起點
    （多含一行 `"status": "open",`）卻忘了同步改 `new`** ⇒ 結果是 `"owner"` 出現兩次、`"status"` 消失。
    而 **JSON 對這兩件事都不報錯**（缺鍵合法、重複鍵 last-wins）⇒ `json.load()` 成功、`--check` 也 OK，
    欄位是在**語意上**被吃掉的。
    ⇒ 規則：文字插入之後，**印出目標物件的 `sorted(keys())` 與全表每個物件的鍵數**，
    以及 `len(x) == len(set(x))`（重複鍵偵測）。只驗「parse 得動」等於沒驗。
    ⇒ 同源的一般化：**改錨點時，`old` 與 `new` 必須一起改** —— 只改一邊是把替換變成刪除。
    ⚠ **插欄位要錨在「行首的 key」上**（例如 `"probe": ...` 那一行），**不要錨在句子中間**：
    2026-09-20 實例 —— 我先刪掉某個 value 結尾的半句話，於是插入文字裡的 `"` 變成**字串的收尾**，
    整份 JSON 從那裡開始全錯。同一天還犯過一次「替換文字裡帶未轉義的 `"`」。
    **兩次都是「先 `write_text` 才 `json.load`」讓壞檔落盤**，修法是固定順序：
    `s2 = s.replace(...)` → `json.loads(s2)` → **通過才** `write_text`。
    ⚠ **而驗證要在寫入「之前」做**：替換文字裡若帶了未轉義的 `"`（例如引述舊值），JSON 會當場壞掉。
    2026-09-20 實例：腳本先 `write_text` 才 `json.load` ⇒ **磁碟上留下非法 JSON**，要靠 portal 的檢查
    才發現。**先把「寫入後的內容」在記憶體裡 `json.loads(new_s)` 驗過，再落檔。**
    同類的文字陷阱：**JSON 字串裡用單引號**（不需轉義），不要用雙引號引述。

27. **停掉一支背景量測，不等於停掉它開的 server —— 而且「熱閘門」可能不只一層。**
    2026-09-19 兩個連續的坑：
    - **孤兒 server**：`TaskStop` 殺了 campaign 的 bash，`http_duo` 的子行程鏈斷掉，**它開的
      `llama-server` 留在 8080 上活著**（`http_duo` 自己的 cleanup 只在它正常結束時才跑）。
      ⇒ 停掉量測之後**一定要用 `lsof -nP -iTCP:8080 -sTCP:LISTEN -t` 查出 pid，再 `kill -TERM <pid>`**
      （graceful 才會釋放 Metal buffer；不要用名字批次殺）。判準：`--port auto` 意味著它可能在 8081。
    - **兩層熱閘門**：我在 campaign 裡已經把「等 NOMINAL」放寬成「TRAPPING 才拒絕」，但**內層的
      `http_duo` 自己有另一個** —— 它每一 rep 都 `wait_nominal`，預設 `--cooldown-timeout 420`。
      實測 22:31→22:38 只跑完 2 個 request（≈7 分鐘），thermal 從未到 0，因為**這台 server 自己就是熱源**。
      ⇒ 放寬熱閘門時要**往下問一層**：`http_duo` 用 `--cooldown-timeout <秒>` 收斂（改 CLI，不改原始碼），
      並讓 per-rep 的 `prefill_thermal`/`decode_thermal` 欄位負責記錄，這樣沒有數字是無標籤的。
    - 一般化：**「我已經處理了 X」要問成「X 在這條呼叫鏈上有幾個實例」** —— 今天兩次都是
      「外層處理了、內層沒處理」（熱閘門、清理）。

28. **上界探針的「半死」最危險：它仍量得到步時上界，但另兩樣會給假陽性 —— 先逐欄位判定合法性。**
    2026-09-19 實測 `CGC_SUBMIT_AHEAD=1`（專案指定的「唯一零改碼上界探針」）：
    - 25/28 個 decode 步的 header 印 **`... gpu_sum=0.00 union_sum=0.00 gap_sum=0.00 ms (NO TIMESTAMPS)`**
      —— poll 點移到「下一段已提交」之後，沒有完成的 command buffer 可讀 ⇒ **結構性**，不是 overlap 生效；
    - 同一臂 **`mean len = 1.00`**（base 2.40）、生成截到 **21 token**（base ~141）⇒ **acceptance 崩掉**。
    **但它仍然是有效的上界探針** —— 因為兩臂跑**同一種圖**（實測 base 174/177 步 `ntok=4`、
    ahead 28/28 步 `ntok=4`）⇒ 每步工作相同，差的只有那個 CPU 序列化窗：
    **169.21 → 99.39 ms = ×1.702，與認證的 ×1.711 差 0.5%。**
    ⇒ 那一臂的**合法讀數只有一個：步時比**。另兩樣會給假結論：
      - `gap/union` = 0 ⇒ 是**儀器死**（`(NO TIMESTAMPS)`），**不是「gap 已消除」**；
      - `t/s` ⇒ acceptance 崩掉 ⇒ 探針自己的 t/s 只有 base 的 **0.709×（更慢）**，那是 racy 的
        **損害讀數**，不是下界也不是天花板；合法的 t/s 報酬要分開講：`[×1.0, ×1.70]`。
    ⚠ **我的報告器當時真的把 0/0 算成「G1 MET / union −68.7 pt」** —— 已修成拒絕計分（任何
    `(NO TIMESTAMPS)` 或 median `union_sum==gap_sum==0` ⇒ 印 `UNREADABLE`，不給判定），
    並多印 server 的 `mean_len`，讓「工作量被換掉」當場可見。
    ⇒ **可攜的規則**：任何「把某個東西刪掉量上界」的臂，**先確認哪一個欄位在這個臂合法**
    （`grep -c "NO TIMESTAMPS"` 決定能不能讀 GPU 側；`mean len` 是否同級決定能不能讀 t/s），
    再讀那個欄位。⚠ **不要把「探針只壞一半」誤讀成「全壞」或「全好」—— 這兩種誤讀我同一天各寫過一次。**

29. **層間空檔（gap）已經分解完畢 —— 它只有兩個成分，而且門檻「≤5%」算術上不可達。**
    用逐層欄位對**四支 run**（涵蓋 `sum(cb)` 21→37 ms，所以會縮放、不是固定偏移）擬合：
    ```
    gap_L = 1.00 x cb_{L-1} + 0.29-0.35 ms      r(組內) = +0.94 .. +0.999
    每步：sum(gap) = sum(cb) + 39 x (0.29-0.35)     而 L0 的 gap 永遠是 0.000（唯一無前驅層）
    ```
    - 斜率 **1.00** ⇒ 那一層的 top-k hook **把整段時長加到關鍵路徑上、CPU/GPU 零重疊**；
    - 它**與前一層的 `union`（r +0.01…+0.13）和 `submit`（±0.02…+0.11）都不相關** ⇒ 不是「GPU 交得晚」；
    - hook 只在 **2–11 層**上發火（集合隨 caps 改變）⇒ `cb` 是**雙峰**的，pooled 中位數只有 ~0.02 ms。
    **★ 兩個硬結論：**
    1. 成分 (ii) 單獨是 `39 x 0.32 = 12.5 ms`，不減段數就消不掉 ⇒ `gap_sum ≥ ~11–14 ms`，而
       E2b 的 5% = 6.97 ms、base 的 5% = 8.54 ms **都在地板以下** ⇒ **「gap/total ≤ 5%」不可達**。
       可達的改寫：盯 `cb_sum/total`（15.2–21.6%）、或盯**邊界段數**（12.5 ms 對段數線性）。
    2. 那個窗的**定義**就是「CPU 在跑 hook、GPU 閒著」⇒ **「把 gap 藏到 GPU 工作底下」沒有東西可藏**。
       槓桿只剩：hook 更便宜，或邊界更少。`CGC_SUBMIT_AHEAD` 的 ×1.702 是**刪掉依賴本身**（racy）。
    ⇒ **可攜的規則**：任何「overlap 可以拿回 X%」的主張，先問**那個窗裡 GPU 有沒有別的工作**；
       若窗的定義就是 GPU idle，那它不是排程問題，是 CPU 序列化問題。
    ⚠ 兩個欄位陷阱：`gap` 是 `CGC-DECPROF all` 行的**第 7 組**（第 6 是 `union`）—— 取錯會得到
       「corr(gap, cb) ≈ 0」這種看起來像發現的東西；**修正方法是拿已知值對錨**
       （本 repo 的錨：union 1.85/2.81、gap 0.46/0.49、sum(gap 中位數) 21.4）。
       另一支 log 內 `ntok` 會**混多種圖寬**（2/4/14），不先取眾數過濾，「逐層中位數」就是混血的。

30. **一個名字在同一個函式裡綁兩次 ⇒ 判定档的欄位變成常數，而負向自測抓不到。**
    實例（2026-09-19，`m123_oracle_gate.py`）：`pin` 先綁「reference 的期望 md5」，30 行後又被綁成
    **oracle-env 的顯示字串**（`pin = "" if args.no_pin_oracle_env else ...`）⇒ 判定档的
    `ref_pinned` **永遠是 False**，讀起來正好是事實的反面。負向自測（把 pin 改錯 ⇒ 應拒絕）**抓不到**，
    因為拒絕發生在碰撞**之前**；只有端到端跑一次、去看那一格的值才會發現。
    - 修法：改名（`ref_pin` / `pin_note`）＋ **source-level 守衛**加進自測：
      `re.search(r"^\s*pin\s*=", 模組原始碼, re.M) is None`。名字碰撞用單元測試測不到，
      但「這個名字不准再被綁」測得到。
    - **可攜的規則**：驗收一個寫進判定档的欄位時，問「它在**所有分支**下都能取到不同的值嗎」；
      **恆定的欄位比缺欄位更糟** —— 缺欄位會被發現，恆定欄位會被當成事實。
31. **改變「被比較的內容」的 CLI 參數不在 config stamp 裡 ⇒ `comparable=True` 但比較的是無關序列。**
    實例：`--probe-prompt` 讓 975 筆的 dump 對 9 筆的 reference 報 **`comparable=True` ＋ `M1 1/9`**，
    看起來像退步；實際上 key `(step, token_idx, ctx_type)` 撞在一起卻描述**不同的 token 序列**
    （prompt 是 CLI 參數，不進 CGCENV/ENV/ARG 的 config stamp）。
    - 修法：判定档記 `dump_records` / `coverage_pct` / `probe_prompt_md5`，並在 `coverage_pct < 100`
      時印**警告**（白名「這讀起來像 M1 退步」）。
    - 可攜的規則：**「可比」的判準要涵蓋所有會改變內容的輸入**，不只是會改變環境的東西。

32. **要「借」別人的量測窗口，要讀他的 `require*` 條件，不要猜他「在等什麼」。**
    2026-09-20 實例：另一條線的 driver 用 `server_window.py wait --need-mb 8000` 排隊，我看到它 poll 到
    `reclaimable=6823<8000` **且在下降**，於是推論「他短期等不到 ⇒ 我跑 2 分鐘沒關係」。
    ⇒ **他的兩串實驗（`--rule-ab --reps 5` ＋ 一個 7-profile suite 基線）一次 launch 都沒跑成就 `RC=1`**：
    ```
    server_window.BusyBox: ... other llama process(es): [(40811, 'llama-server')]; reclaimable=1509MB<8000
    === RULE-AB RC=1 ===   === SUITE RC=1 ===
    ```
    - **`reclaimable` 只是他 `wait` 階段的條件**；他的 `measure()` 每次 launch 前還會呼叫
      `SW.require_first(need_mb=8000)`，而**那個條件還包含「沒有別的 llama 行程」**。
    - 規則一：**讀對方的 `require*` 原始碼**（`grep -n "require_first\|BusyBox" scripts/check/*.py`），
      不要從他排隊的訊息反推他在等什麼。
    - 規則二：**「他在等記憶體」與「他不介意別人跑」是兩件事** —— 這次只成立第一件，我把第二件當成了結論。
    - 規則三：對方的 driver **不一定重試**（這裡直接拋出 ⇒ 整串 RC=1）⇒ 借窗口失敗的代價可能是他一整輪。
    - 規則四：**建置也算佔用**（換掉 `binary.head_hash`）⇒ 同一套檢查要放在建置之前，不只是跑之前。

33. **`node` 這個 kind 是「沒有名字的節點」的自動兜底名，不是一種工作。** 2026-09-20：
    `ggml.c:7190-7193`（`ggml_build_forward_expand`）對每個 `strlen(name)==0` 的節點做
    `ggml_format_name(node, "node_%d", ...)`，而 kind 詞彙表（`ggml-backend.cpp:2228`）有一條
    `"node"` 前綴把它整桶收走 ⇒ `(other)` 桶因此在這些 log 上是 **0.00 ms**。
    判準：**任何「`node` 佔 X%」的句子，都要先問 X 是哪些 op 的**（`CGC-GPUOPK` 給）
    ——本模型上是 GET_ROWS 25.3% / MUL 22.5% / UNARY 22.5% / MUL_MAT 18.3% / ADD 8.4%，
    **最大項是專家 gather**。把它當「未 fuse 的 elementwise」會把標籤缺口當成熱點。
    修法是 builder 補名（`cb()` 就是 `ggml_format_name`，不進數值路徑），**不是 kernel 工作**。

34. **KIND × OP 那張「工作加權」（`wcntw`）表排的是 NODE 數，不是時間。** 定義在
    `ggml-backend.cpp:1962-1971`：把 buffer 時長分給裡面「會發出工作的節點」⇒
    **每個 op 都得到該步自己的平均**（實測 ~150 µs/node，六個最大 op 全落在 105–170 之間），
    且 `wcntw[op]/total ≈ nd[op]/nd_work`（MUL 11.3→10.6、ADD 17.1→16.1、MUL_MAT 17.4→15.8、
    RMS_NORM 5.3→6.6、GET_ROWS 6.5→6.0、CLAMP 1.6→1.5）。
    ⇒ **不要**用它排 kernel 工單。**逐 kind 邊際成本不可識別**：同一個 command buffer 裡的
    kind 向量共線，脊回歸得到 R² = 0.13 與負係數（`Backup/g4_kind_cost_lsq_20260920.py`）。
    可用的是**容器級**分組（command buffer 的 kind 集合＝它的身份）：
    `Backup/g4_buffer_signature_20260920.py`（69494 個 buffer／32 種簽名；40 個每層主 buffer
    佔 42% 的 buffer 時間；每步約 351 個 buffer）。

35. **「少開 command buffer」這條路 2026-09-15 就關掉了，別再掃一次；而且它 ≠ kernel 數。**
    `Backup/phase_decomp/cb_sweep.json`：n_cb ∈ {1,2,16} ⇒ decode **10.34 / 9.98 / 10.28 t/s**、
    三者 `answer_md5_set` 全同 ⇒ `run_server.sh:106` 的「cb8 sweet spot」是**高原不是峰**。
    ⚠️ 這條變的是 **MTLCommandBuffer 的分組**，**kernel 一個都沒少** ⇒
    引用它時**不可以**說成「kernel 數不是約束」。要動 kernel 數就得真的減少節點／dispatch。

36. **要 fuse 一條 elementwise 鏈之前，先問「有沒有既有的 fused op」，再問「它保序嗎」，
    最後——**這一條是三問裡最重要的**——先問「它**有沒有已經被量過**」。**
    2026-09-20 的實例（**我當天就犯滿了前兩問、漏掉第三問，代價是一份寫錯的工單**）：
    `GGML_OP_MUL_MAT_ID_DOWN_COMBINE` **早就實作好了**（`llama-graph.cpp:2714` gate；
    `kernel_mul_mv_id_down_combine_{q3_K,iq3_s,iq4_xs}_f32` 在
    `ggml-metal.metal:11886/12012/12124`，**`llama-graph.cpp:2729` 的「kernel is Q3_K-only」是過期註解**），
    gate 的型別集**剛好覆蓋本模型全部 40 個主幹層**（`ffn_down_exps` = IQ3_S ×37 ＋ IQ4_XS ×3
    ＋ Q3_K ×1），兩個 env（`CGC_DOWN_COMBINE:1406`、`CGC_DC_MULTITOK:1419`）**都已在 allowlist**。
    - 它算 `Σ_e Σ_k (w_e·q)`，圖上算 `Σ_e (w_e·(Σ_k q))`（權重乘在每個 K 項上、
      整個專家迴圈只做一次 `simd_sum`，`ggml-metal.metal:11847-11863`）⇒ **代數相等、浮點不等**。
      **M1 是 `row_fnv1a64` ⇒ 逐位元**（`m123_oracle_gate.py:16`），閘是 `M1/M2 = 884/884`
      ⇒ **任何 reassociation 都被依構造否決**。設計 fuse 的第一個問題是「它保序嗎」，不是「它快多少」。
      ⚠️ 而這件事樹上**早有證明**：`llama-graph.cpp:2803-2822` 寫明 `CGC_ADD_ORDER=rev`
      「早已證聚合對順序敏感」。**先讀那條註解，不要重新推導。**
    - ★★ **它早就被量過了，但那次量測沒有對照成功 ⇒ 符號 UNRESOLVED。**
      `docs/DOWN_COMBINE_IQ3S_IMPL_2026-09-18.md` 報「正確、慢 15–18%」
      （單 token 8.09 vs 9.85、verify 5.32 vs 6.23，兩臂 `answer_md5_set` 皆 `['72ca6608']`）。
      **但把 `union_sum` 從那兩個 log 逐步配對重讀之後，那對臂作廢**：`en-dc-on` **沒有**
      `CGC_DC_MULTITOK` ⇒ 閘門只在 `n_tokens == 1` 成立 ⇒ **`ntok=4` 的步上兩臂走同一條路，
      應該逐位元相同**。實測（n=27 配對）：host 側 `cb` ×1.031、`submit` ×1.022，
      而 GPU 側 `union` ×1.215（19/27）、`wait` ×1.236、`gap` ×1.269
      ⇒ **那對臂之間有一個 ~22% 的 GPU 狀態差，與效應同量級**（加上效應量 1.76 t/s 對上
      單臂噪音底 ±1.9 t/s）。⇒ **「慢 15–18%」不可引用，反向也不可引用。**
      仍然成立的只有：**輸出逐字相同、覆蓋 40/40、三個 kernel 都在 shipped dylib 裡**，
      以及未被量過的「平行度」候選（未融合的 `kernel_mul_mv_id_*` 把 expert 放 **grid 維** ⇒
      8 個 expert 並行 threadgroup；融合是 `for e in 0..nei0` ⇒ 同一個 threadgroup **序列**）。
      ⇒ **「9 個節點 → 1、每步 −320 dispatch = −13.0%」仍然是不成立的推論**（把 node 數當成本，
      從沒被量過），**但它也沒有被否證**。**任何「以 node 數估算收益」的句子都不要寫。**
    - ⚠️ **同批的 Q3_K（Ornith）「+25%」不可引用**：`en_dc_orn_ab_fwd` 17.88 vs 17.83（打平）、
      `en_dc_orn_ab_rev` 19.65 vs 15.72（+25%）——而**對照臂自己從 17.83 漂到 15.72（13%）**。
      兩次互相矛盾 ⇒ 那不是結果，是未受控的漂移。
    - ⚠️ **t/s 單獨不能判這題**：效應量 1.76 t/s，而本專案單臂噪音底是 **±1.9 t/s**。
      能分辨的儀器是 **`union_sum`**（`decode_sweep.py` 不報 union ⇒ 要開 `CGC_DECODE_PROFILE`
      再從 server log 用 `Backup/union_by_layer_20260919.py` 讀）。
    - ⚠️ **`answer_md5` 不能當 M1 的證據**：它是**生成文字**的 md5，argmax 對最後幾個 ulp 穩健。
      `llama-graph.cpp:2706-2710` 那句「單 token 那對 md5 相同（72ca6608 == 72ca6608）」
      **不是** M1 會過的證據，不要引用成證據。
      ⇒ 反過來也成立：**判值變更的優化，正確判準是「輸出逐字相同」，不是 M1**。
    - ⚠️ **`ggml_sum_rows` 不是序列和的替代品**：`kernel_sum_rows_impl`（`ggml-metal.metal:1721`）是
      **樹狀歸約**（`for i0 = tpitg.x; i0 < ne00; i0 += ntg.x` ＋ `simd_sum`）＝配對和。
    - **保序融合的唯一構造性安全做法**：**每個執行緒負責一個輸出元素，字面序列 f32 相加**
      （無跨執行緒歸約 ⇒ 逐位元等價是構造出來的，不是測出來的）——但先問它值不值得（見上）。

37. ★★ **在設計任何實驗、或論證任何機制之前，先搜自己已有的產物。10 秒，省一份錯工單。**
    2026-09-20 實錄：我讀完 kernel 原始碼、反推出 down-combine「算術不保序、所以 G2 會否決」，
    寫了一整份工單建議「修好算術以拿回 −13%」。**那份量測 2026-09-18 就做完並留在樹上了**
    （`docs/DOWN_COMBINE_IQ3S_IMPL_2026-09-18.md`；`Backup/phase_decomp/en_dc_*` 共 **8 個 JSON**），
    結論是那個融合**慢 15–18%** ⇒ **修法 A/B 全是白做**。
    我的取證沒有錯（M1 的判定甚至與樹上一致），錯的是**順序**：先推導，後查帳。
    固定第一步（兩條，都零成本、不碰 GPU）：
    ```sh
    # (a) 這個旋鈕/env/op 是否已經被量過？
    grep -rl '<ENV 名或 op 名>' docs/ .workbuddy/memory/ agent_harness/engine_loop/traces/ \
        agent_harness/engine_loop/manifest* 2>/dev/null
    # (b) 這個題目是否已經有產物？（Backup/ 未受版控，所以 grep 版控找不到它）
    ls -t Backup/phase_decomp/ | grep -iE '<關鍵字>' ; ls -t Backup/cgc_logs/ | head
    ```
    附帶一條同樣貴的教訓：**不要用自己剛說不可信的代理去算收益**。
    我前一節才證明「KIND×OP 表排的是 node 數、不是時間」，下一節就用 `node 數 × 平均` 算出
    「−320 dispatch = −13.0%」。**自相矛盾的算式會躲過複查，因為兩半分別看都對。**

    ★★ **同一天在這一條上摔了三次，三次都是「先算一個數，再問它回答的是不是我的問題」**：
    - 把 **node 數**當**成本**去算收益（§EN-281，「−320 dispatch = −13.0%」）。
    - 把一對**未受控**的 A/B 當判決（§EN-282/283，兩半符號相反的漂移被讀成效應）。
    - 把 **`3σ`（每步散佈）** 當**中位數的標準誤**（§EN-284；正確的是 `1.253·σ/√n`，誇大 ~5.4×），
      以及把**跨 run** 的散佈當**可分辨門檻**（§EN-286；正確的是**單一 run 內**的 `SE(中位)`）——
      兩者差 2–3 倍，而且**只有後者回答「這台機器能不能判這個效應」**。
    **判準（寫進流程）**：報任何「能不能分辨」之前，先寫下三件事 ——
    ① 我算的散佈是 **run 內**還是 **run 間**（run 間含機器漂移，Ornith 的 AB/BA 量到殘留 +9.95%）；
    ② 分母是 **每步**還是 **摘要量**（中位數的 SE = 1.253σ/√n，不是 σ）；
    ③ 樣本數是多少（n=45 與 n=180 差 2× 的門檻）。
    實測參考：`union_sum` 在 E2b 暖態下中位 132 ms、σ 37.7（全段）/21.0（暖態後半）
    ⇒ **n=179 給 3SE 8.0%、n=89 給 6.3%**；而在 48 步的短 run 上是 16–17%。
    ⇒ 結論：**12.8% 的效應只能用「一個夠長的暖 run」判，不能用「兩台先後 server」判**。

38. ★★ **任何帶「閘門／旗標」的 A/B，第一個動作是檢查「旗標不該影響的子集」上兩臂是否相同。**
    2026-09-20 實錄（同一天的第二輪更正）：`en-dc-on`/`en-dc-ctl` 那對臂是**先後跑、未交錯**的
    兩次 server。`en-dc-on` 只設 `CGC_DOWN_COMBINE`、**不設 `CGC_DC_MULTITOK`**
    ⇒ 閘門 `cgc_dc_shape_ok` 只在 `n_tokens == 1` 成立 ⇒ **`ntok=4` 的 verify 步上兩臂走同一條
    未融合路徑，理應逐位元相同**。實測（逐步配對 n=27）：`cb` ×1.031、`submit` ×1.022
    （host 側、與路徑無關）**卻** `union` ×1.215(19/27)、`wait` ×1.236、`gap` ×1.269（GPU 側）。
    ⇒ **這個 A/B 沒有對照成功，它的效應量（以及據它下的「已知變慢」結論）全部作廢。**
    - 做法：**子集**＝旗標不可能生效的那些步（這裡 `ntok` 篩選）；**對照量要一側 host
      （`cb`/`submit`）一側 GPU（`union`/`wait`）**——只對一側就看不出是「機器漂」還是「路徑變」。
    - 訊號判準：**本該不動的量動了 ⇒ 不必再讀效應量。** 成本＝對**已存在的 log** 跑一個 grep。
    - 為什麼這條值得單獨記：它**不需要 GPU**，而且今天同時救了兩件事 ——
      阻止把「未受控」當成「已證實」寫進 records，以及避免「放棄一條其實還沒被測過的路」。
    - 附帶：**交錯（AB/BA）不是可選項**。這對臂的 22% 漂移就是「兩次先後跑」的產物；
      本 skill 的配對設計規則（`paired_ab.py`、臂間散佈可達 1.54×）本來就要求交錯。

39. ★★ **「等窗口」的門檻量錯了東西：一個單次抽樣的閘門，落在它授權的動作前 10 秒。**
    2026-09-20 實錄：`server_window.py wait`（當天新寫的 `cmd_wait`）跑 **70 分鐘 → 60 次讀數、
    2 次 `window open`、0 次量測**。全文 `docs/WINDOW_GATE_POSTMORTEM_2026-09-20.md`。
    - **門檻 8000 在這個 box 上算術上不可達**：60 次讀數只有 **1 次** ≥8000，而那次是尖峰。
      降到 6500 之後是 **32/60 = 53%**，仍然是丟硬幣（中位 6515）。
    - **兩次「開啟」都是暫態，10 秒內消失**：`8778 → 6810`、`6753 → 5261`。
      ⇒ `[wait] window open` 是一句**預測**；而 `window stolen between the poll and its launch`
      **這個訊息名是錯的** —— 兩次都不是被搶，是讀數自己在 poll→launch 那個縫（行程啟動 ~10 秒）裡塌掉。
      **訊息指向錯的機制會讓人去查鄰居，而鄰居不存在。**
    - **機制**：`vm_free_mb() = free + purgeable + inactive`，實測 free/purgeable 只有 0.0–0.2 GB
      ⇒ **這個閘門實際上就是 `Pages inactive`**，而它是核心 LRU 的**老化佇列（一個速率，不是容量）**。
      零 llama 行程下 10 秒可動 **1.3 GB**（檔案快取漲 1.5 GB ⇒ 從匿名 inactive 拿頁；
      `page_pageable_internal` 8.63 GB vs `external` 2.05–3.55 GB）。
      ⇒ **「確認沒有別的 llama 行程」擋不住它**，因為成因與 llama 無關。
    - **修法**：宣告前**再確認一次**（`--confirm-s`，預設 0 = 原行為；
      `Backup/patch_cmd_wait_confirm_20260920.py`，已在 scratch 副本驗過 4 項）。
      **只可能更嚴、不可能更鬆**，所以是安全的改法。
    - ⚠️ **先確認你的啟動器在不在閘門名單裡**：走 `server_window` 的是 13 支
      （`phase_split_ab`／`pool_curve`／`mtp_accept_ab`／`window_gate`／`plain_match_ab`…），
      而 **`http_duo.py`／`profile_duo.py`／`decode_sweep.py` 不在其中** ⇒
      照著「等窗口」的卡去跑這三支，就是一次**不設防的啟動**。
    - ⚠️ **這道閘門在沒有 `ps` 權限的環境會拋 `PermissionError`，不是回 BUSY**：
      `foreign_llama` 與 `_fallback_llama` **兩者**都 shell out 到 `ps`，agent session 的 sandbox 拒絕它
      ⇒ 別在這種 session 裡安排「等窗口再跑」的計畫。**守門員缺依賴時該說 BUSY，不該拋例外。**
    - ★ **最有用的一句：先問「這一步真的需要窗口嗎」。** G4 的儀器底原本要自己起一台 server，
      而佇列裡那一輪已經帶著 `CGC_DECODE_PROFILE=1` ＋ `CGC_GPU_TIMING=1` ⇒ **它的 log 就有
      `union_sum`**，收割即可（`scripts/check/union_floor.py --newest 3`）。
      **等待的成本常常高於換一條路。**
    - ★★ **「先問樹上有沒有現成的自然實驗」**（2026-09-20 同日追蹤，★ 這是本條最貴的一課）：
      我當時把「run 之後閘門是否系統性偏高」寫成**「需要一次 run」**——**錯的**。
      `require_first()` 只擋一個行程的第一次啟動、之後**只記錄**，所以多臂的 `mtp_accept_ab`
      留下一個**行程內配對**，而它把抽樣**蓋進產物**（`window` 欄）：
      冷 8026/8116 vs 暖 8783…10234，**完全分離**，p ≈ 1/60。
      **要先搜產物（`Backup/**/*.json*` 裡帶 `window`/`quiet` 的區塊），再決定要不要動機器。**
    - ★★ **但那個效應的機制是反的 ⇒ 別做「顯而易見」的處置**：假說說「page cache 裝著模型頁」，
      實測是**讀檔讓閘門更低**（只讀 8 GB 模型、不起 server：gate 5456 → 3297，同時 `ext` ＋1726、
      `comp` ＋3612 —— 剛讀進來的檔案頁落在這個閘門**不計**的佇列）。回收很慢（180 s 仍 −514 MB，
      還在爬）⇒ **「把 cache 弄熱來開窗口」會讓情況更糟，而且會拖累別人。**
      **要殺掉一個「顯而易見的修法」，最便宜的方式是做一次只變一個因子的最小實驗。**
    - ★ **更正必須落在「作出斷言的那個欄位」，不是隔壁。** 這個 repo 反覆付這個學費：
      `work_order` 說「measured: 15-18% SLOWER」，而收回它的話寫在 `measured_counterexample`；
      `priority` 說「instrument floor is unmeasured」，而量到 floor 的話寫在 `measurement_floor`。
      **改記錄時，grep 那個斷言的原文，確定它在同一個欄位裡被改掉。**（`targets.json` 的逐欄
      整行替換 + 寫入前 `json.loads`：`Backup/patch_g4_record_completion_20260920.py`。）

40. ★★ **引用一個既有儀器的數字之前，先問它的 gate 涵蓋哪些步；而且要報散布，不是單點。**
    2026-09-20 的實例（G1 的前提 B），兩層都踩到：
    - **gate 太窄 ⇒ 讀數為空而不自知。** `CGC-S1: TABLE-CHURN` 的 gate 是 `n_tokens == 1`
      （`llama-context.cpp:4668`；而 `n_tokens = t->ne[1]`，`:4846`）。它在 **MTP off** 的 decode 步
      成立（ntok=1，31/60 步），而在**交付配置 MTP on** 上是 ntok=4 ⇒ **觸發 0/128**。
      ⇒ 樹上那 45 份讀數**全部**來自 MTP-off，**答不了交付配置的問題**。
      **動手前該做的**：拿一個已知配置的 `CGC-DECPROF` `ntok` 分布（它是現成的）去對 gate 的條件，
      **數出它會觸發幾次**。**「有 45 份讀數」不等於「問題被量過」。**
      （同型的另一半：CTX gate 印不出東西、`STEP_DBG` 的 block 在 fast-path 內 ⇒ 0 行。）
    - ⚠️ **不要把某個儀器的常數抄到另一個儀器。** `:4932` 的另一個儀器用 `n_tokens <= 2`，而它
      **留了理由**：**可讀性**（原話「skips the 8-token middle」）。抄過來在 MTP-on 上只覆蓋
      **1/128**。正確的 predicate 是**池路徑自己的**：
      `cgc_is_decode_graph(n_tokens, cgc_pool_max_tokens())` ——「這一步走不走池路徑」
      就是「它要不要發布表」的問題（`cgc_pool_max_tokens` 的自我描述就是
      「max n_tokens that the expert-cache pool path handles (multi-token decode)」）。
      兩個 helper 都是 `static inline` ⇒ static 自由函式可直接呼叫 ⇒ **一行而不是三處改動**。
    - **報散布。** 43 份**步組成完全相同**（`publishes=1599` ＝ 41 graphs × 39 層、10 個 instrument
      步）的 run，churn 散在 **37.2%–84.4%**（兩叢集 {37.2,49.2,50.8,53.3,67.2}／{83.8,84.4}），
      而同一簽名**逐位元重複 20 次** ⇒ **那是配置，不是噪音**（旋鈕未被記錄）。
      ⇒ 在這種量上引用**單一中位數**等於假造精度。
    - ★ **由此可以認出「看似效應的兩點比較」。** 樹上引用了五天的
      「churn `47.5%` → 修後 `16.5%`」：兩個數字**各來自一份 run**，而其中一那份 **80.8% 的步是
      decode**（26 graphs／21 instr 步），另一份只有 47.5%（61 graphs／29 步），隔三天、
      不同 build 與池狀態、**無控制** ⇒ **那個「下降」不是效應**。方向（≠0）成立，幅度不成立。
      **判準：兩個數字若都落在同一個已測的散布範圍內，它們的差就不是一個效應。**
      ⇒ 全文與分組表：`docs/G1_PREMISE_B_RECHECK_2026-09-20.md`。

41. ★ **先問「我要的量受散熱影響嗎」—— `wait_nominal` 是為速度量測設計的，量計數器時不要付它。**
    2026-09-20 實測：`http_duo.py:476` 在**每一軸之前**呼叫
    `wait_nominal(args.cooldown_timeout)`，而 `--cooldown-timeout` 預設 **420 秒**。
    當時 thermal 是 **HEAVY** ⇒ server 已經起來（log 有 `listening on http://0.0.0.0:8080`）
    但 **log 完全停滯**（兩次取樣都是 21072 bytes、`CGC-DECPROF` 只有 step=1）
    ⇒ 看起來像卡死，其實是**在等 NOMINAL**。
    - **繞過：`--cooldown-timeout 0`** ⇒ 立刻放棄等待並繼續（它會把該軸的 thermal 記成非 NOMINAL）。
    - **判準**：我要的是**計數器**還是**速率**？計數器（`consumed_changed`、`clamped_selected`、
      `zero_mapped_selected`、`publishes`、hit/miss）**不受散熱影響** ⇒ 不該為它們等窗口；
      速率（t/s、ms/step）才需要，而那時**必須**等（否則數字不可引用）。
    - ⚠️ **同一個 repo 裡的工具在這點上不一致**：`decode_sweep.py` **沒有**這個等待
      （2026-09-20 同一晚用它跑兩輪都很順），`http_duo.py` 有。**用之前先 grep `wait_nominal`**。
    - ⚠️ 附帶：**`server` 起來 ≠ 請求已送出**。判斷「有沒有在動」要看 **log 的 bytes 變化**，
      不是看行程存在；而 TERM 殺掉的 server **仍會印 teardown**，所以那個 log 是無效樣本
      （`scripts/check/premise_b_read.py` 會拒絕它）。

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
  ✅ **★ 2026-09-19 當天重測：這個 ×1.711 重現了** —— base 169.21 ms vs `CGC_SUBMIT_AHEAD=1`
  99.39 ms = **×1.702**（差 0.5%）。**成立的條件是兩臂的 verify 圖寬相同**（實測 base 174/177 步
  `ntok=4`、ahead 28/28 步 `ntok=4`）⇒ 每步工作一樣，差的就是那個 CPU 窗
  ⇒ **序列化窗 = 步時的 41%。** 但同一支探針**讀不到** `gap/union`（`(NO TIMESTAMPS)`）與 `t/s`
  （acceptance 2.40→1.00 ⇒ 探針自己的 t/s 只有 base 的 **0.709×**，證明不了那一半）
  ⇒ 引用時必須講清楚是**步時比**還是 t/s 比；`t/s` 的報酬是 **[×1.0, ×1.70]**，下界由 G2 保證。
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
