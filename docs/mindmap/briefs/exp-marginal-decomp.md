# **「邊際 verify token ＝ 34.7 ms」由什麼構成… — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：**「邊際 verify token ＝ 34.7 ms」由什麼構成？** 由 L254 的兩個實測點反解出 `round_ms ≈ 52.0 + 34.7·T̂`（off `T̂=1.00／86.7 ms`、k=2 `T̂=2.80／149.1 ms`）。 那 **34.7 ms／token 的斜率**有三個候選構成： ① **GPU 執行**（每顆 token 的 GEMV／attention） ② **IO 等待**（該 token 的新專家讀取） ③ **派發入場費**（節點數隨 token 數線性成長） 為什麼這一問決定一切：**若把這個邊際壓掉**，在接受率完全不動（E=1.625）的前提下就是 **×2.71**（邊際砍半 ×1.40、砍到 1/4 ×1.85）。⇒ **(i) 能不能單獨拿到 2× 完全取決於它。**

- 主題：實驗　·　子目標：**M MTP on 加速**（活躍攻關軸：speculative 攤薄係數 m 與 accept rate a）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產

---

## 1. 目標

**「邊際 verify token ＝ 34.7 ms」由什麼構成？** 由 L254 的兩個實測點反解出 `round_ms ≈ 52.0 + 34.7·T̂`（off `T̂=1.00／86.7 ms`、k=2 `T̂=2.80／149.1 ms`）。 那 **34.7 ms／token 的斜率**有三個候選構成： ① **GPU 執行**（每顆 token 的 GEMV／attention） ② **IO 等待**（該 token 的新專家讀取） ③ **派發入場費**（節點數隨 token 數線性成長） 為什麼這一問決定一切：**若把這個邊際壓掉**，在接受率完全不動（E=1.625）的前提下就是 **×2.71**（邊際砍半 ×1.40、砍到 1/4 ×1.85）。⇒ **(i) 能不能單獨拿到 2× 完全取決於它。**

## 2. 判準

指名 34.7 ms 的主項（GPU 執行／IO／派發）並給比例

## 3. 結果

decode 9.97 t/s

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 假設：**設計修正（2026-10-03，第二次發車）**：原設計用 k≈4 取「同一趟內 T̂ 有變化 ⇒ 可逐點回歸」， 但 **k=4 違反存活配方**（必帶 `--spec-draft-n-max 1`）⇒ 兩趟 VOID（4.5 s/step 的 thrash，`incomplete=True`）。 ⇒ 改為 **存活配方 k=1 ＋ 一趟 off 對照**，且因 spec 旗標是**整趟級**（不能按臂混）⇒ **兩個 invocation**： - `off`：`--arm "prod-new"`，**無** `--spec-type` ⇒ 每步都是 ntok=1 的 **target** 前向 - `on-k1`：存活配方 ＋ `--spec-draft-n-max 1` ⇒ draft 步 ntok=1（draft 模型）／verify 步 ntok=2（**target** 模型） 判準：**Δ ＝ on 的 verify 步 − off 的步** ＝ 「target verify 批多裝 1 顆 token」的邊際，逐桶拆： - Δ`gpu_busy_sum` ≈ Δtotal ⇒ **GPU 執行是主項** - Δ`gap` 或 Δ(`wait`−`gpu_union`) ≈ Δtotal ⇒ **IO／idle 是主項** - Δ`segs`／Δ`bufs` 隨 token 數成長 ⇒ **派發入場費** ⚠ k=1 的趟內沒有 T̂ 變化（verify 恆 ntok=2）⇒ **不靠迴歸，靠兩趟配對**； 要 2 點以上的斜率得再加 k=2 臂（verify 寬度會變，但記憶體風險較高、非本次）。 ⚠ 記錄裡有三條**互相矛盾**的一手線索，所以必須量：§8「權重搬運只值 7.5 ms/步（11%）」→ 不是搬運； §9「MoE GEMV 全族 ≈5%、in-situ 只佔 GPU 忙碌 3.4%」→ 不是 GEMV； DECODE25 §8「85.8 ms ≈ 64 ms miss 相關（75%）」→ 又指向 IO。

---

## 5. 子目標綁定 · options · 測試 Log 報告

- **子目標**：`L25-4`　MTP（C6）：唯一「乘」的軸
- **來源**：子目標看板（唯一來源）　·　置信度 `high`
- **profile**：`prod-new`
- **被測 option**：`CGC_SERVER_MTP=1`；`CGC_SERVER_LAYER_CAPS=40-40:16`；`CGC_DRAFT_CTX_ALIGN=1`；`CGC_DRAFT_SMALL_BATCH=1`
- **儀器開關**（不是被測 option）：`CGC_MTP_PERF=1`
- **CLI**：`--spec-type draft-mtp`；`--spec-draft-n-max 1`
- **arm 1（可複製）**：`prod-new:CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:16;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1`
- **結案狀態**：結案（排除）（k 軸經濟學判 DEAD，2026-10-01 16:3x；同日 17:0x 的 k 掃描修正了幅度）—— ★ k 掃描（k=1/2/3/5/7 ＋ 同 session MTP-off，六趟全 attribution=none、NOMINAL、swap 0）：k=2 10.90／10.10（同 session off 的 −5%／−12%）、k=1 10.49（−9%）、k=3 9.34（−19%）、k=5 9.37（−19%）、k=7 5.61（−51%）；同 session 錨點 MTP-off 11.52／11.54（歷史 10.923 是別窗口 ⇒ 本對取代它）。⇒ k 是槓桿、最佳點 k=2，而「+52%／emit token」是 k=7（k_eff 5.06）那一點的產物 —— 最佳點的殘餘虧損 −5~12%。成本律（修正口徑後、六臂、dof 4，殘差 ≤20 ms）：round_ms ≈ 63.6 + 35.2·T̂（T̂ = drafted/rounds + 1）⇒ 每輪固定項 ≈ 0.73 個純解碼步（結構死因）＋邊際 verify token 35.2 ms（＝純步的 0.41 ⇒ 攤薄確實發生）。⚠ 2026-10-01 17:2x 量具修復：上一句原本寫 89 + 33·T —— 那個版本吃了兩個量具的分母不一致（ACCEPT 的 mean_len 把 replay 輪當有產出；MTP-PERF 的 emit_tok_per_round 除以原始 draft 呼叫、漏掉 replay 輪）⇒ 已在 llama-bench.cpp 加 replay=／emit_per_round=（行尾追加）並重建驗證（rounds − calls_draft = Σreplay：208 − 142 = 66 ✓）。真 emit/round 是 1.49–1.98（不是 1.74–2.44），而 n_max 的比較不受影響（t/s 是量出來的）。⇒ 可交付的行動：profile 預設 n_max=3 與本線用過的 7 都不是好設定，2 才是（3→2 同 cell +12%）。判決文 docs/L254_K_AXIS_ECON_2026-10-01.md §k 掃描；立項 charters/e-l254-kaxis-sweep-2026-10-01.yaml、產物 Backup/l254_kaxis_20261001/。兩個問題都答了：① 機制：S1 讓 k_eff 1.8 → 7.000（可行）；② 經濟：不划算 —— 涼窗口成對（B 先跑、同一 build、只差 CGC_MTP_NO_CTX_OTHER）：draft 成本/emit token 9.22 → 13.98 ms（+52%）、emit 只多 2.26× 但 draft 相時間多 3.44×（超線性）、acc_rate 兩臂同（0.416–0.444，沒補回來）⇒ E = 1 + a·k_eff 的「每輪成本固定」前提被實測否定。wall t/s 同向（A 8.60 vs B 7.48；第三趟 B2 13.41 但 attribution=swap）但 四趟的窗口 spread 1.19／1.19／1.19／2.49 均 &gt;1.10 ⇒ t/s 全 VOID，只當方向證據；判死靠的是計數器端點。對照：MTP-off 交付錨點 10.923、生產 ABBA ×0.695（配對）⇒ MTP-on 在此口徑整體為負，拉高 k 更負。reopen 條件（已寫死）：乾淨窗口（none ∧ ≤1.10 ∧ reps3）的成對出現 B_tps ≥ A_tps ⇒ 本判詞被推翻。判決文 docs/L254_K_AXIS_ECON_2026-10-01.md。【底下的機制史留檔】2026-10-01 operator 已拍：MTP-on 獨立口徑 ⇒ 前置①解除；同日 16:05 S1 機制端點達標且兩趟重現：k_eff 7.000 逐輪、draft=0 0%、M-RoPE 拒收 180→2、acc_rate 0.52–0.61（對照臂 1.75–1.97／48%／153–180 行；證據 Backup/l254_s1_20261001/k7_paired{,_clean}.{json,stderr.log}）⇒ 卡點改掛 k 軸經濟學：k_eff 追平 k 之後每輪代價也追上去，且四臂 attribution=thermal ⇒ t/s 全 VOID、需涼窗口重判）。09-30 大 k 接受率穩定性：量了，但量到的不是 MTP 的性質。① 回顧（零盒子成本）：09-24 s1_ksweep 兩場 k=1/2/3/4，k_eff 恰等於 k、0 個 position error，而 k=4 的 a 跨場次讀到 0.489 vs 0.945（1.93×）、k=3 讀到 0.654/0.772/0.892 ⇒ 不穩的是場次，不是 k。② 新跑 k=7 與 k=8（本臂、(default) cell、p2048_n128_d512_r3）：兩趟都被 arm-timeout 殺（rc=-15）；CGC-MTP-PERF 各只出 3 行：k=7 a=0.5238/0.4887/0.4631、gen_tok_per_round=2.032/1.900/1.880；k=8 a=0.4219/0.3516/0.3958、gen_tok_per_round=1.730/1.542/1.655。⛔ 這幾個 a 不可用，理由不是「髒窗口」而是結構：兩趟的 CGC-PHASE-SPLIT 都是 routable=15 slots -> decode graph width=1 tokens，而乾淨那批是 routable=53 -> width=6。兩項交叉驗證同一個結論 —— (i) k_eff &lt;&lt; k（要 7 只得 1.88），實作根本沒把 draft 拉到 k；(ii) position error 密度 104/5055、98/3870 行，遠高於 p2 的 32/11075 ⇒ 驗證區塊被截斷 ⇐ 它放不進 width=1 的圖。⇒ 結論：在現行宣告下，大 k 的接受率不是「不穩定」而是「構上量不到」；0.4631 與 0.3958 是 width=1 的產物，不得寫進 k 曲線。證據：Backup/l254_kbracket_2026-09-30/k7.json（＋k7.stderr.log）、k8.json（＋k8.stderr.log，INCOMPLETE／attribution=both）、Backup/s1_ksweep/20260924_{225548,233141}/*.stderr.log（回顧，k_eff==k）。③ 09-30 15:31 解 width 的實測（交付 cell、p0_n128_d512_r3、其餘同臂）：把 CGC_SERVER_LAYER_CAPS 由 40-40:16 改成 40-40:140 ⇒ LAYER_CAPS ... min 140/layer、routable=139 -> decode graph width=8 tokens，而這一趟跑完了（NOMINAL 全程、attribution=none、base_check PASS、6 行 MTP-PERF 對比 width=1 的 3 行）。⛔ 但 k_eff 仍然只有 1.590（區間 1.59–1.72），不是 7，acc_rate 0.343–0.413。那個 acc_rate 不可用：k_eff 1.6 落在乾淨批 k_eff=1（a=1.000）與 k_eff=2（a≈0.88–0.98）之間，讀到 0.35 自相矛盾 ⇒ 它是被中斷輪次壓低的比值。⇒ 結論修正：width 只解「跑不跑得完」，不解「k 有多深」（width 1 的兩趟是被 stall 殺的，width 8 的這趟活到底且 position error 反而更多：194 vs 98）。真正的截斷＝docs/KEFF_CAP_2026-09-26.md 記的 is_mem_shared draft 位置碰撞（draft loop 第二步解到 X = Y ⇒ llama_decode 拒 ⇒ 迴圈 break ⇒ 静默只吐一個 token）；本趟的指紋是 MTP fast path: draft calls=230 對 verify calls=11916（1:52）。證據：Backup/l254_kbracket_2026-09-30/k7_width8.json（＋k7_width8.stderr.log）。④ 09-30 15:37 帶 LLAMA_BENCH_SPEC_DBG=1 的直方圖（決定性）：不是鏈太短，是六成的輪次一根都沒吐 —— per-round draft=：0 ⇒ 180 輪、1/2/3/4/5 ⇒ 11/18/17/6/4、7 ⇒ 60 輪、8 ⇒ 1（約 297 輪）⇒ 加權平均 1.92 ✓。也就是說 要 7 真的拿得到 7（最深的請求達得到），拖垮平均的是 61% 的 draft=0。⇒ 卡點是 輪級掉線，不是深度。失敗形態指向 target 的 verify 批次：194 次 init 失敗中 X &gt; Y 144（74%）、X == Y 50，而歸屬 draft 迴圈的 llama_decode[0] 只有 6 次 ⇒ 配上本 build 自己印的 shared_kv=1：is_mem_shared=1 下 draft 把位置寫進共用 KV，target 從同一起點 verify 就被 M-RoPE 的嚴格 X &lt; Y 拒掉。⇒ 這能對上 09-24 那批乾淨的 k=1..4（k_eff == k、0 pos-err）：它們跑在 65c76b8c7（09-25 17:16）之前 ⇒ 那時非 GEMMA4 arch 的 ctx_other 被丟掉 ⇒ is_mem_shared=false ⇒ draft 有自己的 KV。這是一個設計取捨（收回去 vs 做對共用 KV 的 verify 佈局），不是一行修補；兩條都要走 M1/M2/M3 oracle gate。證據：Backup/l254_kbracket_2026-09-30/k7_specdbg.json（＋k7_specdbg.stderr.log）。⚠ 另註：speculative.cpp 裡那兩個修法（09-26 one_position_drafts、09-28 stale-tail）在原始碼與 binary 裡都已存在（libllama-common.0.0.630.dylib 14:39:41 驗到 stale-tail 字串、且本次有觸發），它們不對應這條。★ 2026-10-03 批次（MTP 重開工程的帳）：① cb 悬崖定罪：MTP-on k=1 的 4–5 s/步 cb＝分類錯配 —— CGC_SERVER_LAYER_CAPS=40-40:16 ⇒ 15 routable ⇒ cgc_decode_bound(15,8,8)=floor(15/8)=1 ⇒ width=1 ⇒ ntok=2 判成 PREFILL ⇒ CGC_PREFILL_STREAM=1 讓 hook 在 llama-context.cpp:8073/:8334 每層重讀整層 slab（≈356 MiB/層 × 40 層 ≈ 14.2 GiB/步；cb(ntok=2)≈cb(ntok=512)）—— 既有 CGC-HOOK_SPLIT（累加點在 slab 分支之後）、CGC-EB_TIMER（slab 路徑不呼叫 ensure_batch）、以及本輪新裝的 CGC-FASTSEG（整趟只中 1 次、draft ctx）對這條路都是瞎的；10-01 的 k 掃描之所以跑得完，是因為它用 caps=40-40:140（width=8、走 decode 路徑）。⇒ 重開前置：caps ≥17（16 routable ⇒ bound=2）或新 cell；立項 scripts/check/charters/e-mtp-cb-classify-2026-10-03.yaml。② spawn/join 假設已否證（LLAMA_EXPERT_CACHE_SERIAL_FILL=1：cb 中位 4155→5010 ms、EBTIMER 237→150 ms）⇒ 94% 不在逐冷專家 thread+join。③ spec-tree 寬 2＝重開工程 #2（Backup/spec_tree/T1C_VERDICT_2026-10-03.md）：T1c 證「寬 1 ≡ 鏈」原理上不可能打平（accept() 是下一輪 seed 寫入 ⇒ replay 多吐一顆：1.455 vs 1.462）；寬 2 撞 GGML_ASSERT(batch_in.n_seq_id[k]==1) ⇒ 修法＝每個分支各掛一根；判準寫死：寬 2 臂 token/decode ≥ ~2.05 才可能打平 off。④ S1 品質判詞（S1_QUALITY_VERDICT_2026-10-03.md）：控制 10.957（乾淨）vs S1 臂 26.099（swap）×2.38，但品質 −17.2 pp（26/29→21/29、5 題對轉錯）⇒ 20+ 門在 operator 條件下關閉。⑤ 校準軸同日收窄：M2_ONLY_CALIBER_LEVERS（放寬口徑在長 probe 下是空的）、CGC_MM_BITIDENT decode 窗配對 −0.1%（K0 的 −4.9% 不復現、那 4–5% 是 prefill 槓桿）、POOL10G 被 cell 契約拒跑（池=0 也超 11453）。
- **逐條處置**：整合進子目標　→ `L25-4`　—　邊際 verify token 的成本拆解（round ≈ 52.0 + 34.7·T̂）；2026-10-03 首個「同一 invocation」off/on 配對：ON 8.70 vs OFF 11.53 t/s、兩臂皆非 attribution=none ⇒ 讀數 VOID（機制面已驗：深度=2、decode width=8、fast path cold=0）
- **測試 log（實跑）**：[pair.json](../../../Backup/spec_tree/mtpon_pair_20261003/pair.json)　[pair.json](../../../Backup/spec_tree/mtpon_pair_20261003/pair.json)　[warm.json](../../../Backup/spec_tree/mtpon_pair_warm_20261003/warm.json)　[pair.json](../../../Backup/spec_tree/mtpon_pair_warm_20261003/pair.json)　[pair.json](../../../Backup/spec_tree/mtpon_pair_warm_20261003/pair.json)
- ⛔ 必帶 --spec-draft-n-max 1：沒帶就是 k_eff≈2.33 必 thrash。且 MTP on/off 是不同輸出函數 ⇒ t/s 不可互比。

## 6. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [MTP k-sweep（verify batch T 成本曲線）](mtp-ksweep.md) | 3a | step ≈ 14.61 + 42.03·T（max resid 11.5）；最佳 k=2 但那格不可移植（合成 accept 0.93~0.99 vs 交付 0.465~0.58 |
| [M：攤薄係數 m 0.474 → ≤0.073](exp-m-draft-cost.md) | 3a | **未跑（有卡、無讀數）**：t/s(k) = 1000·mean_len(k)/(S0·(1+m·k))；k=3 即使 a=1.0 也只 **19.25**、k→∞ 極限 **2 |
| [M2 slab pool-reuse（整層填的 memcpy 省多少）](e-m2-slab-reuse.md) | 3a | decode 8.26 t/s |
| [同一趟（同 build、同格、同 session）裡： ① caps…](e-victim-classify-2026-10-05.md) | 3a | 尚無可引用讀數：1 個 run 全部非乾淨（attribution.verdict=swap（swap_growth=5369.81 MiB, max_swap=9644.69 M |
| [**寬度軸的因果量測（W1 vs W2，同 session 同 se…](exp-w1w2-paired-2026-10-06.md) | 3a | 實驗進行中（尚無讀數） |
| [**儀器對帳：m 的兩台儀器定標** m（每 draft token…](exp-m-instrument-recon-2026-10-06.md) | 3a | 實驗進行中（尚無讀數） |

## 7. 運行設置與 Log（生產級腳本 prod-new ＋ 自己 option）

**Run 1** · 2026-10-03 21:01　·　thermal HEAVY　·　swap 7640 MiB→12226 MiB
- arm：`prod-new:CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:140;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=2`
- 命令：`llama-bench（prod-new）-p 0 -n 64 --warm-skip 64（由產物參數重建）`
- Log／產物：[pair.json](../../../Backup/spec_tree/mtpon_pair_20261003/pair.json)
- 結果：tg=8.7
- 判定：swap：swap_growth=4297.499999999999 MiB, max_swap=12047.25 MiB

**Run 2** · 2026-10-03 21:01　·　thermal HEAVY　·　swap 7640 MiB→12226 MiB
- arm：`prod-new`
- 命令：`llama-bench（prod-new）-p 0 -n 64 --warm-skip 64（由產物參數重建）`
- Log／產物：[pair.json](../../../Backup/spec_tree/mtpon_pair_20261003/pair.json)
- 結果：tg=11.53
- 判定：both：thermal=HEAVY, swap_growth=296.75 MiB

**Run 3** · 2026-10-03 21:01　·　thermal NOMINAL　·　swap 2160 MiB→6614 MiB
- arm：`prod-new:CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:140;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=2`
- 命令：`llama-bench（prod-new）-p 0 -n 64 --warm-skip 64（由產物參數重建）`
- Log／產物：[warm.json](../../../Backup/spec_tree/mtpon_pair_warm_20261003/warm.json)
- 結果：tg=9.4
- 判定：swap：swap_growth=4462.38 MiB, max_swap=7151.06 MiB

**Run 4** · 2026-10-03 21:01　·　thermal MODERATE　·　swap 6574 MiB→6713 MiB
- arm：`prod-new:CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:140;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=2`
- 命令：`llama-bench（prod-new）-p 0 -n 64 --warm-skip 64（由產物參數重建）`
- Log／產物：[pair.json](../../../Backup/spec_tree/mtpon_pair_warm_20261003/pair.json)
- 結果：tg=9.31
- 判定：swap：swap_growth=567.3100000000004 MiB, max_swap=7141.31 MiB

**Run 5** · 2026-10-03 21:01　·　thermal MODERATE　·　swap 6574 MiB→6713 MiB
- arm：`prod-new`
- 命令：`llama-bench（prod-new）-p 0 -n 64 --warm-skip 64（由產物參數重建）`
- Log／產物：[pair.json](../../../Backup/spec_tree/mtpon_pair_warm_20261003/pair.json)
- 結果：tg=9.97
- 判定：none：swap_growth=-420.3700000000008 MiB, thermal=MODERATE

## 8. 子目標分解（持續更新；2/5 完成）

- [x] 跑前立項（現狀/目標/假設/驗收） [證](../../../scripts/check/charters/exp-marginal-decomp.yaml)
- [x] 用生產級腳本 prod-new ＋ 自己 option 跑實驗臂，留存 log
- [ ] 驗收：指名 34.7 ms 的主項（GPU 執行／IO／派發）並給比例
- [ ] 否證條件：若四個桶的斜率都 ≈0 且加總 ≈34.7 ⇒ 歸因失敗，要換儀器（不主張）
- [ ] 把 34.7 ms 拆進 GPU 執行／IO／派發三桶

## 9. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/L254_K_AXIS_ECON_2026-10-01.md` |
| 備註 | 假設：**設計修正（2026-10-03，第二次發車）**：原設計用 k≈4 取「同一趟內 T̂ 有變化 ⇒ 可逐點回歸」， 但 **k=4 違反存活配方**（必帶 `--spec-draft-n-max 1`）⇒ 兩趟 VOID（4.5 s/step 的 thrash，`incomplete=True`）。 ⇒ 改為 **存活配方 k=1 ＋ 一趟 off 對照**，且因 spec 旗標是**整趟級**（不能按臂混）⇒ **兩個 invocation**： - `off`：`--arm "prod-new"`，**無** `--spec-type` ⇒ 每步都是 ntok=1 的 **target** 前向 - `on-k1`：存活配方 ＋ `--spec-draft-n-max 1` ⇒ draft 步 ntok=1（draft 模型）／verify 步 ntok=2（**target** 模型） 判準：**Δ ＝ on 的 verify 步 − off 的步** ＝ 「target verify 批多裝 1 顆 token」的邊際，逐桶拆： - Δ`gpu_busy_sum` ≈ Δtotal ⇒ **GPU 執行是主項** - Δ`gap` 或 Δ(`wait`−`gpu_union`) ≈ Δtotal ⇒ **IO／idle 是主項** - Δ`segs`／Δ`bufs` 隨 token 數成長 ⇒ **派發入場費** ⚠ k=1 的趟內沒有 T̂ 變化（verify 恆 ntok=2）⇒ **不靠迴歸，靠兩趟配對**； 要 2 點以上的斜率得再加 k=2 臂（verify 寬度會變，但記憶體風險較高、非本次）。 ⚠ 記錄裡有三條**互相矛盾**的一手線索，所以必須量：§8「權重搬運只值 7.5 ms/步（11%）」→ 不是搬運； §9「MoE GEMV 全族 ≈5%、in-situ 只佔 GPU 忙碌 3.4%」→ 不是 GEMV； DECODE25 §8「85.8 ms ≈ 64 ms miss 相關（75%）」→ 又指向 IO。 |
| 軸性質 | 活躍攻關軸：speculative 攤薄係數 m 與 accept rate a |
| 對應報告 | 0 份 |

- （無）

---

← [在單次提交臂上，**把每一步的 union 不經 hook 交給預取…](exp-singlesubmit-fillahead.md)　·　[總目錄](index.md)　·　[HTML 版](exp-marginal-decomp.html)　·　[M2 slab pool-reuse（整層填的 memcpy 省多少） →](e-m2-slab-reuse.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
