# **交付口徑量測（生產級）**：spec-tree MTP-on（`… — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：**交付口徑量測（生產級）**：spec-tree MTP-on（`--spec-type draft-mtp --spec-draft-n-max 2`， `alt_branch` 走 seq 1 繞開 M-RoPE X<Y）vs MTP-off，在 **(default) 格**（llama-bench 出廠形狀 `-p 512 -n 128 -b 2048 -ub 512 -d 0 -r 5`、`warm_skip 0`）＋ **delivery 採樣口徑（temp=0.4）**、 **釘 seed=1**，取真實 decode tg 配對。 這是對「另一條線 10-06 修過 T1c（spec-tree alt-branch 在 sampling 下 draft_ratio=0.70、0 次 X<Y） 是否真能推進 delivery」的判準量測——機制 work（§16 已證）≠ 加速證成，本卡要補 tg 差。

- 主題：實驗　·　子目標：**不適用**（非攻關軸：約束／帳本／上界算術／入口索引／跨線產品）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產

---

## 1. 目標

**交付口徑量測（生產級）**：spec-tree MTP-on（`--spec-type draft-mtp --spec-draft-n-max 2`， `alt_branch` 走 seq 1 繞開 M-RoPE X<Y）vs MTP-off，在 **(default) 格**（llama-bench 出廠形狀 `-p 512 -n 128 -b 2048 -ub 512 -d 0 -r 5`、`warm_skip 0`）＋ **delivery 採樣口徑（temp=0.4）**、 **釘 seed=1**，取真實 decode tg 配對。 這是對「另一條線 10-06 修過 T1c（spec-tree alt-branch 在 sampling 下 draft_ratio=0.70、0 次 X<Y） 是否真能推進 delivery」的判準量測——機制 work（§16 已證）≠ 加速證成，本卡要補 tg 差。

## 2. 判準

on 臂 decode tg 嚴格高於 off 臂，且兩臂皆 caliber=UNIFIED + quote=QUOTABLE（無 swap attribution）；本讀數登記為 (default) 形狀 spec-tree MTP-on 的**首筆可引用 + 最高紀錄**（certified 表 ratchet_exempt 基準列）

## 3. 結果

尚無可引用讀數：1 個 run 全部非乾淨（attribution.verdict=both（thermal=HEAVY, swap_growth=957.3799999999992 MiB））

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 假設：spec-tree alt-branch 把 tree 的 branch 1（本位置 runner-up）種在 seq 1，避開與 seq 0 在 draft KV 的同一位置碰撞（llama-batch 拒絕重複位置 ⇒ M-RoPE X<Y）。這道牆在 plain draft-mtp （n_seq=1）撞（§13/§14 G2/G3 實測 0 drafts），但 alt-branch 是另一條 code path。 必要前置：binary 必須含另一條線未提交的 `common/speculative.cpp` 修動（accept_from + alt_branch）； 以及 `LAYER_CAPS=40-40:140`（caps≥17）讓 `cgc_decode_bound≥2` 才能塞下 verify 區塊。

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
- **逐條處置**：已定案（約束／基準）　—　spec-tree MTP-on vs off，(default) 格＋採樣口徑（temp=0.4）＋seed=1，兩臂同 denseIQ4X 模型。ON tg 6.15 vs OFF 10.38 t/s（＝×0.592，−40.8%；s04 重製 ×0.503）⇒ spec-tree MTP 在 16 GB 盒是 decode 淨回歸：draft_ratio 0.708、mean_len 2.42、X&lt;Y=8（alt_branch 機制可用），但 cache hit 96%→35%、swap 抖動使 net decode 反慢。caliber_certify REFUSED 兩臂（attrib=both、thermal HEAVY、spread&gt;1.10）⇒ 超訂不可引用、不入 ratchet、不入 C15。結論＝「機制成功、交付不 advance」，與 L25-4 k 軸經濟學 DEAD 同判。產物 Backup/spec_tree/rs0_sampling/{sp_mtpon_off_s05,sp_mtpon_on_s05}.json
- **測試 log（實跑）**：[sp_mtpon_on_s05.json](../../../Backup/spec_tree/rs0_sampling/sp_mtpon_on_s05.json)　[a_control.json](../../../Backup/spec_tree/f1f2_diag/a_control.json)　[b_bgate16.json](../../../Backup/spec_tree/f1f2_diag/b_bgate16.json)
- ⛔ 必帶 --spec-draft-n-max 1：沒帶就是 k_eff≈2.33 必 thrash。且 MTP on/off 是不同輸出函數 ⇒ t/s 不可互比。

## 6. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [k=3 的 1.43× 飄移定位 ＋ 配對認證](k3-swing.md) | 3a | 飄移已定位；配對認證 n=5 時 t=2.08 未達 df=4 的 2.776 ⇒ 16.4% 仍未認證；上一輪「bench 配對 sd 更小」被自我撤回 |
| [口徑＋k=3 飄移認證（16.4% 噪聲底）](exp-caliber-calibration.md) | 3a | **未跑（有卡、無讀數）**：本節點的產出是**口徑校準**——記錄裡至少有四個不同的「步」（DECPROF work-row **167.81** ms／實付週期 step_ro |
| [2026-10-03 operator 裁定「移除所有具名格、pro…](exp-default-cell-first-2026-10-03.md) | 3a | 尚無可引用讀數：1 個 run 全部非乾淨（attribution.verdict=swap（swap_growth=3122.3099999999995 MiB, max_swa |
| [**跨 checkpoint 探針**：把 CGC 引擎（交付路徑＝…](exp-cgc-edge0gguf-2026-10-05.md) | 3a | 實驗進行中（尚無讀數） |

## 7. 運行設置與 Log（生產級腳本 prod-new ＋ 自己 option）

**Run 1** · 2026-10-06 04:02　·　thermal HEAVY　·　swap 5971 MiB→6928 MiB
- arm：`prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:140;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=2;CGC_SPEC_TREE=1;CGC_SPEC_TREE_B=2;CGC_SPEC_TREE_RS=2;CGC_MTP_PERF=1`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[sp_mtpon_on_s05.json](../../../Backup/spec_tree/rs0_sampling/sp_mtpon_on_s05.json)
- 結果：pp=161.06　tg=6.15
- 判定：both：thermal=HEAVY, swap_growth=957.3799999999992 MiB

**Run 2** · 2026-10-06 09:04　·　thermal HEAVY　·　swap 5901 MiB→8133 MiB
- arm：`prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:140;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=2;CGC_SPEC_TREE=1;CGC_SPEC_TREE_B=2;CGC_SPEC_TREE_RS=2;CGC_MTP_PERF=1`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[a_control.json](../../../Backup/spec_tree/f1f2_diag/a_control.json)
- 結果：pp=133.22　tg=5.72
- 判定：both：thermal=HEAVY, swap_growth=2239.8199999999997 MiB

**Run 3** · 2026-10-06 09:14　·　thermal HEAVY　·　swap 6149 MiB→8621 MiB
- arm：`prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:140;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=2;CGC_SPEC_TREE=1;CGC_SPEC_TREE_B=2;CGC_SPEC_TREE_RS=2;CGC_MTP_PERF=1;CGC_SPEC_TREE_BGATE=16`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[b_bgate16.json](../../../Backup/spec_tree/f1f2_diag/b_bgate16.json)
- 結果：pp=116.57　tg=5.97
- 判定：both：thermal=HEAVY, swap_growth=2488.249999999999 MiB

## 8. 子目標分解（持續更新；2/5 完成）

- [x] 跑前立項（現狀/目標/假設/驗收） [證](../../../scripts/check/charters/exp-spectree-mtpon-default-2026-10-06.yaml)
- [x] 用生產級腳本 prod-new ＋ 自己 option 跑實驗臂，留存 log
- [ ] 驗收：on 臂 decode tg 嚴格高於 off 臂，且兩臂皆 caliber=UNIFIED + quote=QUOTABLE（無 swap attribution）；本讀數登記為 (default) 形狀 spec-tree MTP-on 的**首筆可引用 + 最高紀錄**（certified 表 ratchet_exempt 基準列）
- [ ] 否證條件：on 臂 tg ≤ off 臂（無加速）或 drafted=0（alt_branch 未生效 / binary 沒含修動）或 swap attribution 致不可引用。 ⚠ 即令 falsify（最可能的結局，依 S 線 10-06 判詞「機制成功、指標無效」），**量測本身仍進 certified**： on 臂 tg 作為 (default) 形狀 spec-tree MTP-on 的 ratchet_exempt 基準列，標籤註明「無淨加速 / 未推進 delivery」 （對照 §13 plain-chain MTP-on 亦 −6.7%）。這正是用戶要的「到底有沒有推進」的誠實答案。
- [ ] 量 spec-tree MTP-on vs off 在 (default) 形狀 sampling 口徑的 decode tg 配對，判定是否推進 delivery

## 9. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `Backup/spec_tree/fence_default_20261005/fence.json` |
| 備註 | 假設：spec-tree alt-branch 把 tree 的 branch 1（本位置 runner-up）種在 seq 1，避開與 seq 0 在 draft KV 的同一位置碰撞（llama-batch 拒絕重複位置 ⇒ M-RoPE X<Y）。這道牆在 plain draft-mtp （n_seq=1）撞（§13/§14 G2/G3 實測 0 drafts），但 alt-branch 是另一條 code path。 必要前置：binary 必須含另一條線未提交的 `common/speculative.cpp` 修動（accept_from + alt_branch）； 以及 `LAYER_CAPS=40-40:140`（caps≥17）讓 `cgc_decode_bound≥2` 才能塞下 verify 區塊。 |
| 軸性質 | 非攻關軸：約束／帳本／上界算術／入口索引／跨線產品 |
| 對應報告 | 0 份 |

- （無）

---

← [**跨 checkpoint 探針**：把 CGC 引擎（交付路徑＝…](exp-cgc-edge0gguf-2026-10-05.md)　·　[總目錄](index.md)　·　[HTML 版](exp-spectree-mtpon-default-2026-10-06.html)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
