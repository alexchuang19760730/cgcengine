# 異步 gather 流水線（單段＋miss 後台補＋局部重算） — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：在單段提交下用 GPU presence/slot table：命中即算、miss 只寫清單並 MASK，CPU 後台異步 fill 與下一步 GPU 重疊，再只補算 miss expert（線性疊加）⇒ 拿到單段的 ×1.7~1.8 同時保住正確性。

- 主題：缺失處理　·　子目標：**S 序列化消減**（活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產
- 原始方案書：[S1_ASYNC_GATHER_PIPELINE_2026-09-25.html](../../S1_ASYNC_GATHER_PIPELINE_2026-09-25.html)

---

## 1. 目標

在單段提交下用 GPU presence/slot table：命中即算、miss 只寫清單並 MASK，CPU 後台異步 fill 與下一步 GPU 重疊，再只補算 miss expert（線性疊加）⇒ 拿到單段的 ×1.7~1.8 同時保住正確性。

## 2. 判準

預先寫死：① 補算在 sampler 消費前完成的時序窗口 ② MASK＋補算 M1 bit-exact ③ 乾淨 cell r3 decode

## 3. 結果

E0 已給出決定性答案：前提成立——S1（暖池 8 GiB）輸出 文摘文摘… 且 logits 全非有限（引擎自蓋 INVALID），同一輪的分段臂是 42 ⇒ 異步 fill＋補算（C/D）是把 ×1.42–1.83 變成可交付的唯一路徑，不是可選項。E2 被自身結構擋下（S1 跳過 41 段迴圈 ⇒ 逐層儀器不在路徑上）；E1 裁決 fill 的可引用值是 20.8 ms/step（殘差 3.955 不得當輸入）；E4 的比值（decode ×1.42／prefill ×0.80）只能當診斷價（兩臂 thermal HEAVY＋起跑 swap 8 GiB）⇒ 仍無任何吞吐可宣稱。

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 不預測 ids（異於 prebind/ρ）、用真實 ids＋異步重疊；收益【推算】上界 ~19–20 t/s、不宣稱 25（E1 裁決後按 20.8 ms/step 重算 ⇒ ~14.5 t/s；19–20 那個數押在已降級的 3.955 上）

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [CGC_EB_NOFILL 診斷臂（fill 成本）](io-nofill.md) | 3a | fill 3.955 → 0.206 ms/step（−95%）⇒ 機制證；生產口徑同步 fill 僅占 step 4.7% |
| [單段提交（41 段 → 1 段）](s1-segbatch.md) | 3a | A 11.30 → B 20.73 t/s；−40.3 ms/token（45.5%），其中序列化 ~36 ms、fill 僅 3.96 ms |
| [ρ 路線（按層批次化 prefetch）](cache-rho.md) | 3a | 判活：覆蓋 0.849，單獨做 14.36 t/s（+14.2%）；但視窗只有 1.30~1.59 ms，且要付 4.76 ms/步 GPU 插入 ⇒ 有前置條件 |
| [prebind／方案 A（預指派 slot）](cache-prebind.md) | 3a | 判活：qu2/qu3 全過、qu1 邊緣 ⇒ 14.33 t/s（+14.0%）；提前一整步發起 ⇒ 視窗 ≈ 一步 |

## 6. 與其它路線的關鍵差異

| 對象 | 它的做法／結果 | 本條目差別 |
|---|---|---|
| prebind／方案 A | 跨 token 預測 ids、加寬 24 候選、影子 38.3 ms、q24=0.354 | 不預測、用真實 ids、保持 top-8、不加寬 |
| ρ 按層批次 | 同 token 近似 gate（跳 attn）、ρ=0.86 但窗口 1.3–1.6 ms、insert 4.76 | 不做近似、miss 後台補、不依賴重合率 |
| miss-3b | 讓同步 fill 變便宜（合併 pread）、收益 ~2.9% 判死 | 讓 fill 離開同步路徑（異步）、不是變便宜 |
| miss-3c | per-expert 重算、自寫依賴 3b、舊估 +13–18% 作廢 | 異步 drain＋雙緩衝補依賴、重算只為加回、收益重測 |

## 7. 成敗點（風險 → 驗證）

| # | 風險 | 驗證 |
|---|---|---|
| 1 | 時序窗口（最關鍵）：一步 miss 的 fill＋補算要在 sampler 消費前完成 | Stage B：量 fill 時間 vs 下一步 GPU 窗口 |
| 2 | 雙緩衝同步：Metal 單隊列、fill 與當前 gather 爭用 | MAXQ 保險絲、背景 fill 主動讓步 |
| 3 | 異步 bit-exact：MASK＋補算要 M1 逐位相同 | 對 host BATCHDBG、M1 oracle |
| 4 | bg 爭用：背景 fill 搶 IO／頻寬 | 對照 bg on/off |

## 8. 生產設置（要進生產必須滿足什麼）

| 項目 | 驗收條件（可否證） | 現況 |
|---|---|---|
| 正確性不能妥協 | MISS_MASK ＋ 補算必須 M1 逐位元相同（M1 == 1.0 且 M2 == 1.0） | MASK 已過（3b，38 層/421 元素/0 差異）；補算尚未實作 |
| 無前置條件 | 不得要求「先把 fill 關掉」或「hook=0」才能跑；miss 路徑必須 fail-closed（守住零） | 未滿足，且後果已量到：E0 證明「先關 fill」的代價不是變慢而是 NaN（暖池亦然） |
| 成本可接受 | 每層要藏的量 ≤ 該層 hits-only 的 MoE 時間（需求 ~0.52 ms/層 vs 窗口 ~0.4–0.8 ms） | 已量 ⇒ 形狀 1（同步等 fill）死：同臂同框（22:07）decode 步中位 75.09 ms ＝ wait 88.7% ＋ cb 6.7% ＋ submit 4.5%（三欄相加與 header 差 0.09 ms）；EBTIMER＝4.32 ms/步（每步 miss 中位 5）⇒ ratio 0.86 ⇒ FILL_IN_CB。⇒ 可移除的只有 6.7%，而 88.7% 的 wait 不是 miss 的函數（rho −0.32） |
| fill 成本可引用值 | 不得引用單一數字而不帶 miss 負載（同一支儀器會給出 4.32 與 20.8 兩種值） | 已裁決：每 miss 0.69–0.86 ms（20.8/30 與 4.32/5 同帶）⇒ 「20.8 ms/步」是重負載下的數字，不是固有步成本；純屬先前把它當成固定成本的誤用 |
| 診斷價與可交付價分離 | 任何 S1 臂的 t/s 只能當 S 軸形狀上界，不得當成交付速度 | 已寫死：E0 證明該臂輸出 NaN（引擎註解「output is WRONG, diagnostic only, never a deliverable arm」） |
| 交付載體接得上 | S1 的三個開關必須在 SERVER_ENV 白名單內，否則 server 路徑會靜默丟掉 | 2026-09-25 已補 CGC_SEG_BATCH（presence-based ⇒ 只傳非 0 值）；CGC_B_SCHEME／CGC_SLOT_TABLE_GPU 已在 |
| 產物可指紋 | 每次量測帶 engine digest ＋ cell ＋ thermal／swap（窗口條件） | harness bench 已帶（base_check／cell／box_gate） |

## 9. 驗收測試（完整 test cases：E 系列）

| # | 測什麼 | 通過條件（先寫死） | 成本／指令 | 結果 |
|---|---|---|---|---|
| **E0** | 暖池下 S1 臂是不是對的——它決定本方案的存在必要性：若暖池下 S1 與分段臂輸出相同，修法就是「暖池 ＋ miss fail-closed」，不需要異步流水線 | 同 build／同 cell／同 pool（8 GiB）、兩臂皆 MTP off：answer_md5 與 oracle 指紋相同（M1/M2/M3 逐項 9/9）。⚠ 必須附冷池對照（小池或首請求）顯示不同，否則「過」可能只是那輪沒發生 miss | `python3 scripts/check/m123_oracle_gate.py --profile prod-new --tag E0 [--env CGC_SEG_BATCH=1 --env CGC_B_SCHEME=1 --env CGC_SLOT_TABLE_GPU=1]`（1–2 launch（各約 3–5 min）） | 已跑（21:34，背靠背同輪）：分段臂 probe_answer=42、dump 有效；兩個 S1 臂（8 GiB 暖池／1 GiB 小池）都回 文摘文摘…，且引擎自蓋 INVALID：non-finite values 248320 of 248320 ⇒ 前提成立，且比預期更硬：S1 在交付形狀下不是「值錯」而是 NaN（原始碼註解早寫「output is WRONG」，這是它第一次被量到）。⚠ 起跑 swap 8.2 GiB、worst thermal HEAVY；NaN 未在三個 env 之間歸因。產物 Backup/eseries/E0/（seg_ref.jsonl、s1*.jsonl＋s1*.jsonl.invalid） |
| **E1** | 裁決 fill 成本 3.955 vs 20.8 ms/step（同一支 EBTIMER、兩個數；相差 5×，而整本帳都押在小的那個） | 兩數在同 cell（warm-skip 64、fixed-fill-seed 1）下可重算 ⇒ 指出哪一個可引用；不能裁決就寫「不可裁決」，不取小的那個當輸入 | `讀 Backup/nofill_prod/nf_fill.json ＋ docs/FILL_COST_MEASURED_2026-09-25.md §3；必要時一次 CGC_EB_NOFILL bench`（0 GPU） | 已裁決（0 GPU，讀既有產物）：取 20.8 ms/step 當可引用值——它是 CGC-EBTIMER 的直接量測（≥160 步穩態）且交叉驗證（n_sum/步 752 vs union≈19；miss 率 4.0% vs server 4.3%）並在 CGC_POOL_MADVISE 對照輪存活（21–24 ms）。3.955 是兩臂步時差的殘差歸屬，不是 fill 的直接量測 ⇒ 降級、不得當輸入 ⇒ E2 的窗口算術一律用 20.8 |
| **E2** | 逐層窗口算術：一步 miss 的 fill＋補算能不能藏進「同一層 hits-only 的 MoE」（下一層才是消費者） | ① 逐層表可加總回 span（rows 對帳）② needed = miss/step ÷ 41 × per-miss ≤ 該層 MoE 的 gpu 時間 ⇒ 才有 Stage C；否則判 no-go。載體必須是分段臂——逐層儀器在 41 段迴圈之後，而 S1 在 ggml-backend.cpp:1781 提前 return ⇒ 在 S1 臂上它結構上不存在；且判準要先凍結依變數（此處 = wait） | `python3 scripts/check/llama_bench_matrix.py --arms 'prod-new:CGC_DECODE_PROFILE=1;CGC_DECODE_PROFILE_ALL=1;CGC_GPU_TIMING=1;LLAMA_EXPERT_CACHE_BATCH_DBG=1;LLAMA_EXPERT_CACHE_MISS_DUMP=' --prompt 2048 --gen 128 --depths 512 --reps 3 --warm-skip 64 --fixed-fill-seed 1；再 python3 scripts/check/s1_wait_budget.py --stderr --column wait`（1 launch（＋0 GPU 分析）） | 已完成（22:00 等待預算 ＋ 22:07 column census）⇒ 形狀 1 不成立。① 等待預算（規則預先凍結：wait 在 miss 層佔比 ≥ 0.6 且 rho ≥ 0.5）：40/40 層都有 miss、rho = −0.32、逐層中位 wait 幾乎均勻（1.32–1.92 ms）⇒ R2 round-trip-bound。② column census（同一臂、三支儀器）：decode 步 75.09 ms ＝ wait 88.7% ＋ cb 6.7% ＋ submit 4.5%，EBTIMER 4.32 ms/步（miss 中位 5）⇒ FILL_IN_CB（ratio 0.86）；兩個舊數字的矛盾也解決：每 miss 0.69–0.86 ms（20.8/30 與 4.32/5 同帶）⇒ 20.8 是重負載產物。⇒ 形狀 1 想省的只值 6.7%，不能動的 wait 值 88.7%，而裝置側 gap 佔跨度 17%（重疊還有東西可拿）。⚠ 兩輪皆診斷臂（swap growth +2619／+929、worst MODERATE），不宣稱 t/s。報告：docs/S1_SHAPE1_WAIT_BUDGET_2026-09-25.md、docs/S1_COLUMN_CENSUS_2026-09-25.md |
| **E3** | 兩個 make-or-break 的離線結論：補算的逐位元等價寫法、以及異步 fill 的寫入面 | ① 寫死「重跑該層該 token 的融合 combine」而不是「加回部分和」（依據：combine 熔在 kernel_mul_mv_id_down_combine_*；CGC_ADD_ORDER=rev 已證結合序會移動 anchor md5）② 指認 pool_ext_buf[layer][kind] 與 in-flight command buffer 的關係（否則就是 codebase 自己註記的 UB） | `讀碼（llama-graph.cpp／llama-expert-cache.h／ggml-backend.cpp）＋ grep 指紋`（0 GPU） | 已裁決（0 GPU 讀碼）：① kernel_mul_mv_id_down_combine_* 確實存在於 Metal kernel（ggml-metal.metal:11744）⇒ 補算不能「加回部分和」，唯一逐位元寫法是用同一顆 kernel 重跑該層該 token 的融合 combine；② pool_ext_buf[layer][kind] 是 pool 的非擁有視圖（llama-expert-cache.h:172-174，被 llama-expert-cache.cpp:2099 逐層指派）⇒ 異步 fill 直接寫它就是寫「in-flight command buffer 正在讀的 buffer」；必須經 staging＋copy-in 雙緩衝 |
| **E4** | 交付 cell 配對：分段臂 vs S1 臂（同 cell、同 pool、同 build），prefill 與 decode 同報 | 比值可引用、絕對值不外推；帶 thermal／swap／binary md5；S1 產物按台帳規則報比值而非獨立 t/s | `python3 scripts/check/harness.py bench --arm prod-new --arm 'prod-new:CGC_SEG_BATCH=1;CGC_B_SCHEME=1;CGC_SLOT_TABLE_GPU=1' --json /tmp/E4/summary.json`（同一輪 2 臂 × r3） | 已跑（21:26–21:29，同輪 AB r3）：分段臂 pp 233.29±20.83／tg 11.40±0.04（hit 96.2%）；S1 臂 pp 187.03±10.38／tg 16.14±1.66（hit 100.0%、misses 0、reads 0）⇒ decode ×1.42、prefill ×0.80。⚠ 比值亦不可引用：兩臂 worst thermal 皆 HEAVY、起跑 swap 7.7–8.1 GiB、非交錯；且 S1 臂按 E0 屬「NaN 級輸出」⇒ 只能當診斷價（S 軸形狀上界）。prefill 方向與 09-24 k-sweep 相反 ⇒ 盒況主宰，待乾淨窗口。產物 Backup/eseries/E4/summary.json |

## 10. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/S1_ASYNC_GATHER_PIPELINE_2026-09-25.md、docs/S1_SHAPE1_WAIT_BUDGET_2026-09-25.md、docs/S1_COLUMN_CENSUS_2026-09-25.md；E 系列 driver＋產物：Backup/eseries/driver.py、results.json（_order ＋ 逐項 stdout）、E0/、E4/summary.json、E2b|census.json` |
| 備註 | 不預測 ids（異於 prebind/ρ）、用真實 ids＋異步重疊；收益【推算】上界 ~19–20 t/s、不宣稱 25（E1 裁決後按 20.8 ms/step 重算 ⇒ ~14.5 t/s；19–20 那個數押在已降級的 3.955 上） |
| 軸性質 | 活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉 |
| 對應報告 | 2 份 |

- [S1_ASYNC_GATHER_PIPELINE_2026-09-25.html](../../S1_ASYNC_GATHER_PIPELINE_2026-09-25.html)
- [S1_ASYNC_GATHER_PIPELINE_2026-09-25.md](../../S1_ASYNC_GATHER_PIPELINE_2026-09-25.md)

---

← [3c：per-expert 重算 kernel](miss-3c.md)　·　[總目錄](index.md)　·　[HTML 版](s1-asyncgather.html)　·　[MTP 儀器化（接進 llama-bench） →](mtp-instrument.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
