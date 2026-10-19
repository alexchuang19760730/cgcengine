# Strata 三條手法的可搬運化：機制、基線、最小改動、預註冊判準

**2026-10-12 · 分支 `demo/sweet-spot-windows-fix` · 本檔不是交付面文件（檔名不含 DELIVERY/VERDICT/REPORT/WHITEPAPER/CERT），
所有外部數字皆為轉引、不可引用；本機數字為既有已量測錨。**

## §0 為什麼是這三條，而不是換引擎

Strata（`Niko1221/Strata`，MIT，C++，22,358 stars／2,160 forks／812 open issues，建於 2026-09-24）的官方支援面是
Windows/Linux ＋ NVIDIA/AMD **獨顯**（≥12GB VRAM）＋ 32–64GB RAM ＋ 約 80GB 磁碟。
它**沒有 Metal 後端**：Apple Silicon 支援 issue **#836 於 2026-10-07 被 owner 自己關閉**，原話
「Strata has CUDA, HIP and SYCL backends and no Metal backend, so Apple Silicon is not supported. Closing;
llama.cpp and MLX run these models on Macs.」——他指的就是我們已在用的那條路。⇒ 白皮書 §12「不換引擎」不變。

它的加速機制建立在我們換不到的資源上：「RAM 釘住全部 24,576 專家（35–55GB）＋ GPU 熱集 ＋ CPU 算其餘 ＋ SSD 查表」。
本機總共 16GiB 統一記憶體、模型檔 ~13GiB ⇒ **它的第二層（RAM 放全部）在我們這裡不存在**。
所以下列三條要拆成「可搬的規則」而不是「可搬的引擎」。

## §1 投機解碼：可搬的是「成本閘門」，不是「開 MTP」

**Strata 事實（`docs/HOW_IT_WORKS.md`）**：模型自帶的 MTP 層草擬 **≤3** token，一次 pass 檢查全部 48 層，
平均 **2.4–3.2 tokens/pass**；另有 prompt-lookup 從上下文複本草擬 ≤5 token，但
**「only where its measured acceptance and cost say it pays」**（code edits 快 6–11%，其他文本不變 ⇒ 沒賺就不開）。

**我們的基線**：`acc_tok_per_round = 2.023`（n_max=3）——與他們的 2.4–3.2 同量級，所以「草擬品質」不是我們的瓶頸。
轉正失敗的原因是成本側：verify 讓池命中率下降，`file_reads` +62%、`bytes` +129%（該檔轉引 09-17 的原始讀數：`docs/MTP_NET_EFFECT_PAIRED_2026-09-18.md`），
以及單臂／oversubscribed 的判詞（`docs/MTP_TRANSFERRED_2026-10-07.md`）。
機制差異判讀：他們的草擬與 verify 都打在**同一份常駐權重**上（RAM 全量、不落盤）；我們的 verify 會把冷專家推回 `pread`。

**可搬的規則**：投機只在「實測接受率 × 邊際成本」划算時開啟，且閘門必須是**在線量測**，不是靜態開關。

**最小改動**：在 spec 路徑加 per-request 成本閘門（env `CGC_SPEC_COST_GATE=1`，預設 OFF），三個條件全部成立才投機：
(a) 前一窗口接受率 ≥ 門檻；(b) 本窗口預測的冷專家集（沿用 `CGC_PREROUTER` 統計）在池內命中率 ≥ 門檻；(c) decode 長度 ≥ N。
落點：`src/llama.cpp/common/speculative.cpp`（草擬/接受統計）＋ 呼叫端的 spec 決策點；**不動** `mul_mat_id`／池容量／提交併發。

**預註冊判準**（跑之前寫死）：同 session ABBA、on/off 配對、≥3 rep、warm-skip、spread ≤1.10；
成功＝配對中位 **≥ +5%** 且 **verify 期間 file_reads/token 不上升 >10%**（否則就是拿磁碟換速度，不算贏）；
否證＝< +5% 或成本吃掉收益 ⇒ 本軸收斂，與 MTP 既有判詞合流。
**必須一起讀的反證**：§7.6 的 +21%～+41% 是**整條內存/fill 軸**的上界，不是投機單獨可拿的。

## §2 「CPU 同時算其餘」：統一記憶體上不能照搬，但可測

**Strata 事實**：RAM 釘住全部專家；GPU 沒有的專家由 **CPU 就地計算**，與 GPU 同時進行（AVX-512/AVX2、ggml 的 i-quant kernel）；
「每多 1GB VRAM 多放 ~700 專家，GPU 上的每個專家就是 CPU 不必算的那一個」。

**為什麼不能照搬**：那是**離散**架構——CPU 的 RAM 頻寬與 GPU 的 VRAM 頻寬是兩份獨立預算。
我們是 16GiB **統一**記憶體、單一 ~120GB/s 預算 ⇒ CPU 算不會增加總頻寬，只會與 GPU 搶同一份。
但我們的帶寬結論（白皮書 §4：有效帶寬只跑到峰值的 29–50%，即**延遲受限而非飽和**）留下一個可測的假設：
**讓 CPU 對未駐留專家並行計算，是否提高「合計達成的頻寬／記憶體級並行度」，還是只把 GPU 的效率拉低？**

**好消息：旋鈕已經在樹上**——上游 `-ncmoe/--n-cpu-moe`（`src/llama.cpp/common/arg.cpp:2670`）
與 `--spec-draft-n-cpu-moe`（`:3999`），`llama-bench` 也吃（`tools/llama-bench/llama-bench.cpp:357`）。
**未驗**：它在本 fork 的 expert-cache（池／slab）路徑上是否真的生效；`scripts/check/llama_bench_matrix.py`、
`scripts/run_server.sh`、`decode_sweep.py` 目前都沒有轉發它。

**最小改動（只改 harness，不碰引擎）**：
1) 先做**旋鈕生效 smoke**：`nm`／log 探針 ＋ 一次 12 步 decode 的計數器對照（`-ncmoe 0` vs `-ncmoe 8`），
   確認「真的有層被放到 CPU」而不是靜默 no-op（判準：探針必須在**真陽性**下被驗過，見 gate 系列教訓）。
2) 再把 `ncmoe` 接進 `llama_bench_matrix` 參數 → 產生 `-ncmoe {0,4,8,12,16}` 的臂。
   注意既有前置：直起 `llama-bench` 會 OOM，必須走 `llama_bench_matrix.resolve()` ＋ `forward_argv()`（含 `-expert-cache`），
   A/B 一律釘 `--fixed-fill-seed 1`。

**預註冊判準**：同形狀、同 fill seed；成功＝任一 `-ncmoe>0` 的 decode **≥ +5%**，且要**同時報 CPU/GPU 兩側的達成頻寬**
（否則分不清「同時做事」與「把工作搬走」）。
**風險（必須先裁定）**：CPU kernel 與 Metal kernel 的浮點不同 ⇒ 這一臂**極可能不 bit-identical**。
作為**診斷臂**（只回答「統一記憶體上加 CPU 算有沒有用」）不動交付面；若要進交付面，需使用者先裁定「本臂可不 bit-identical」。

## §3 大塊 prefill（8K）vs 我們的熱牆：可搬的是「塊大小 × 佔空比」

**Strata 事實**：長文本以 **≤8,192 token** 為塊讀入（其 32K prompt 讀取 >1,000 t/s）；
且「**下一層的專家在當前層 attention 跑的時候經 PCIe 串流到 GPU**」（prefetch overlap）；第一條訊息之後只讀新增部分。

**我們的基線**：prefill 是**熱限制**——白皮書 §5 的 `t(ms/token) = 0.5575 + 4226 / f_eff(MHz)`，
250 t/s ⇔ `f_eff ≈ 1154–1227 MHz`；冷機 6/6 ≥250、熱機 0/21。
slab 路徑＝`CGC_PREFILL_STREAM=1` ＋ `CGC_GATHER_SLAB_CAP=256`（`run_server.sh:581/618/2383`、`llama-context.cpp:703`）。
我們量過的重疊代價：**每個段邊界多一次提交 ≈ +130µs × 約 41 邊界/步 ≈ +5.5 ms/步**
⇒ 「提前串流下一層專家」在我們的形狀下**不能增加提交數**，否則收益先被吃掉（`eng-perf-0003`；且 `CGC_OVERLAP_FENCE` 讀點在 `ggml-backend.cpp:2902`，fence 41 已量到 +21.8%）。

**可搬的規則**：把「塊大小 × 佔空比（duty cycle）」當自變數——
短塊＋留白讓 `f_eff` 停在拐點以上，**總 wall time** 可能優於長塊（瞬時快但熱墜）。**這一條不需要改引擎**。

**預註冊判準**：同一長 prompt（≥8K）下比較 (a) 單一 8K 塊、 (b) 2K×4、 (c) 4K×2、 (d) 塊間刻意 idle；
成功＝任一排程的**總時間 ≥ +5%** 且 M 閘不掉；否證＝全部 ≤0% ⇒ 熱牆是硬牆，
那就把它與 §5 既有結論併成一堵「已量測的牆」，不再從塊大小找 250。

## §4 順手記下、列為後續（不進本輪）

1. **prompt-lookup 草擬**：Strata 用 SSD 上 28.8GB 的 n-gram 表、每 token 讀幾行，且**只在實測 acceptance 與成本划算時**用。
   對我們的 agentic／重複上下文工作負載是候選槓桿，但本機 IO 已是瓶頸（內接 SSD ~3GB/s）⇒ 先算 IO 預算再談。
2. **`data/experimental-speed-projection`（由模型 activation 導出的向量）**：Strata 稱其為速度投影。
   在我們的語言裡，這正是 S 線判詞（96.3% 冷專家是跨步全新、重用式 prefetch 救不了）所要求的**預測器**。
   值得讀他們的 paper 與 `docs/DETAILS.md`（179KB）確認它是否為 next-expert predictor；若是，這是 fill/prefetch 軸目前唯一未判死的形式。

## §5 執行前置與建議順序

**現在不能跑**：另一條線正在跑 affinity oracle（21:31 起，port 53778/53779，`affinity_oracle_dump.py`）；
`run_server.sh` 的 preflight 會因殘留 llama 行程直接拒跑 ⇒ 本檔三個實驗都要等它結束或另開乾淨窗。

| 方向 | 改動面 | 成本 | 風險 |
|---|---|---|---|
| §3 塊大小 × 佔空比 | **零引擎改動**（現有 harness） | 低 | 低（否證也只是確認熱牆） |
| §2 CPU 算其餘 | harness 一行 ＋ 旋鈕 smoke | 低–中 | 不 bit-identical（先當診斷臂） |
| §1 投機成本閘門 | 引擎（spec 路徑）＋ 需重跑 D5/閘門 | 中–高 | 與 MTP 既有否證同源 |

**建議順序**：§3（零改動、可立即裁決）→ §2（harness ＋ smoke）→ §1（引擎 ＋ 預註冊 charter）。
