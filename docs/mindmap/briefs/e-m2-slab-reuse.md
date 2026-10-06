# M2 slab pool-reuse（整層填的 memcpy 省多少） — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：**整層 slab 填（`llama_expert_cache_fill_layer_slab`）在池已有 15 顆 resident 時， 真的省下 memcpy／pread 嗎？省下多少位元組、值多少毫秒？** —— 即：`CGC-M2-PROF` 的 pool/disk 位元組比與逐 fill 時間， 在「整層」與「只讀非 resident」兩條路上各是多少。

- 主題：實驗　·　子目標：**M MTP on 加速**（活躍攻關軸：speculative 攤薄係數 m 與 accept rate a）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產

---

## 1. 目標

**整層 slab 填（`llama_expert_cache_fill_layer_slab`）在池已有 15 顆 resident 時， 真的省下 memcpy／pread 嗎？省下多少位元組、值多少毫秒？** —— 即：`CGC-M2-PROF` 的 pool/disk 位元組比與逐 fill 時間， 在「整層」與「只讀非 resident」兩條路上各是多少。

## 2. 判準

① `CGC-M2-PROF` 逐 fill 行 ≥ 200（覆蓋至少一輪完整 prefill 的所有層）； ② 由逐 fill 行算出 pool share 與 FAST/非 FAST 的 ms/MiB，並**指名**哪一項主導； ③ 端點是**計數器／比值**（不走引用閘門）；不得把本趟任何 t/s 當成績。

## 3. 結果

decode 8.26 t/s

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 假設：`fill_layer_slab`（llama-expert-cache.cpp:4341）逐專家檢查 `slot_table`： resident 且 slot < usable ⇒ **memcpy 自 pool**（pool_bytes＋），否則收成 segment 走 pread （disk_bytes＋）。收尾印 pool/disk 比。既有產物已顯示 pool:disk ≈ 56:44。 ⚠ 但**機制上有一個反向項**：池命中會讓 `segs.size() != ne` ⇒ 合併大 pread 的 fast path（`:4458` `contiguous`）**失效**，240 顆非 resident 變成 240 次 thread-per-segment pread（`fill_segments_concurrent`，每段一 thread）。 ⇒ 本卡要判定的是**淨效果**：省下的 memcpy/bytes 是否大於打碎 pread 的代價。

---

## 5. 子目標綁定 · options · 測試 Log 報告

- **子目標**：`L25-7`　整層 slab 的 pool-reuse 效率
- **來源**：子目標看板（唯一來源）　·　置信度 `high`
- **profile**：`prod-new`
- **被測 option**：`CGC_SERVER_MTP=1`；`CGC_SERVER_LAYER_CAPS=40-40:140`；`CGC_DRAFT_CTX_ALIGN=1`；`CGC_DRAFT_SMALL_BATCH=1`；`LLAMA_BENCH_SPEC=1`
- **儀器開關**（不是被測 option）：`CGC_M2_PROFILE=1`
- **候選（待指名）**：`CGC_M2_DB_DISABLE=1`
- **arm 1（可複製）**：`prod-new:CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:140;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=2;CGC_M2_PROFILE=1`
- **arm 2（可複製）**：`prod-new:CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:140;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=2;CGC_M2_PROFILE=1;CGC_M2_DB_DISABLE=1`
- **arm 3（可複製）**：`prod-new:CGC_M2_PROFILE=1`
- **結案狀態**：結案（排除）· FAST 對照腿不可測（暖池的結構性、非量具問題）—— 10-04 已量。答案：池命中省下 55.5–55.9% 的磁碟位元組（改走 memcpy），但代價是 merged-pread 快路徑永遠失效。實測（(default) 格、pp512、每臂 200 次取樣）：三臂 pool share 0.5547／0.5547／0.5586，收尾行 non-resident 44.5／44.5／44.1%（比值一致；絕對量因逐 fill log 上限 400 筆而不同——falsify ②的解釋：logged&lt;400 就是上限本身）；整層填 0.343–0.379 ms/MiB（中位 ~30 ms/次、每次 ~82 MiB；磁碟側 ~1.2–1.3 GiB/s）。fill_slab_FAST 0/200、fast=1 0/400，且第一次 slab 填（256 顆裡已有 142 顆是 pool memcpy）就破壞 contiguous 條件（llama-expert-cache.cpp:4435，segs.size()==ne）⇒ 假設裡的反向項不是偶發、是結構性的：只要池是暖的，整層填永遠走 thread-per-segment。FAST 對照腿（預登記：≥2× 或同級）在本形狀無法觀測，需冷池臂（如 --no-warmup 使首次填全非 resident）才能取得 FAST 樣本。⚠ 附註：MTP-on 臂的按臂 spec shim 該趟 drafted=0（M-RoPE 位置檢查失敗 ⇒ 退回 plain decode；CGC-BENCH-ACCEPT phase=timed ... draft_ratio=0）—— 本格端點是 prefill 的 slab 計數器，不受影響，但「MTP-on 臂」在別處被引用前必須回讀 ACCEPT。DB 對照：位元組完全相同下 DB-on 0.379 vs DB-off 0.343 ms/MiB（−9.5%；單一配對、不擴張主張）。端點是計數器／比值，不走引用閘門；本趟任何 t/s 不引用。
- **逐條處置**：整合進子目標　→ `L25-7`　—　整層 slab 的 pool-reuse 效率（pool/disk 位元組比 ＋ FAST vs 非 FAST 的 ms/MiB）；MTP 重開工程的診斷前置，端點是計數器／比值
- **測試 log（實跑）**：[M2_SLAB_REUSE_2026-10-03.json](../../../Backup/spec_tree/M2_SLAB_REUSE_2026-10-03.json)　[M2_SLAB_REUSE_2026-10-03.json](../../../Backup/spec_tree/M2_SLAB_REUSE_2026-10-03.json)　[M2_SLAB_REUSE_2026-10-03.json](../../../Backup/spec_tree/M2_SLAB_REUSE_2026-10-03.json)　[M2_SLAB_REUSE_2026-10-03.mtpoff.json](../../../Backup/spec_tree/M2_SLAB_REUSE_2026-10-03.mtpoff.json)
- ★ 本格端點是計數器／比值（不走引用閘門）；任何 t/s 一律不引用。⚠ 前置（operator 級）已在立項卡寫死：CGC_SERVER_LAYER_CAPS 是 cell 欄位 ⇒ 本卡刻意避開（量的是 slab 路徑，兩個 caps 值都走 slab），主臂用 caps 140（已驗跑完）。⚠ 盒閘：跑前過 l255_close.py --check（即時判定，不寫死數字）。

## 6. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [MTP k-sweep（verify batch T 成本曲線）](mtp-ksweep.md) | 3a | step ≈ 14.61 + 42.03·T（max resid 11.5）；最佳 k=2 但那格不可移植（合成 accept 0.93~0.99 vs 交付 0.465~0.58 |
| [M：攤薄係數 m 0.474 → ≤0.073](exp-m-draft-cost.md) | 3a | **未跑（有卡、無讀數）**：t/s(k) = 1000·mean_len(k)/(S0·(1+m·k))；k=3 即使 a=1.0 也只 **19.25**、k→∞ 極限 **2 |
| [**「邊際 verify token ＝ 34.7 ms」由什麼構成…](exp-marginal-decomp.md) | 3a | decode 9.97 t/s |
| [同一趟（同 build、同格、同 session）裡： ① caps…](e-victim-classify-2026-10-05.md) | 3a | 尚無可引用讀數：1 個 run 全部非乾淨（attribution.verdict=swap（swap_growth=5369.81 MiB, max_swap=9644.69 M |
| [**寬度軸的因果量測（W1 vs W2，同 session 同 se…](exp-w1w2-paired-2026-10-06.md) | 3a | 實驗進行中（尚無讀數） |
| [**儀器對帳：m 的兩台儀器定標** m（每 draft token…](exp-m-instrument-recon-2026-10-06.md) | 3a | 實驗進行中（尚無讀數） |

## 7. 運行設置與 Log（生產級腳本 prod-new ＋ 自己 option）

**Run 1** · 2026-10-04 22:32　·　thermal NOMINAL　·　swap 6891 MiB→7580 MiB
- arm：`prod-new:CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:140;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=2;CGC_M2_PROFILE=1`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[M2_SLAB_REUSE_2026-10-03.json](../../../Backup/spec_tree/M2_SLAB_REUSE_2026-10-03.json)
- 結果：pp=118.84　tg=8.91
- 判定：swap：swap_growth=977.8100000000004 MiB, max_swap=8027.56 MiB

**Run 2** · 2026-10-04 22:32　·　thermal NOMINAL　·　swap 6891 MiB→7580 MiB
- arm：`prod-new:CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:140;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=2;CGC_M2_PROFILE=1;CGC_M2_DB_DISABLE=1`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[M2_SLAB_REUSE_2026-10-03.json](../../../Backup/spec_tree/M2_SLAB_REUSE_2026-10-03.json)
- 結果：pp=78.0　tg=8.26
- 判定：none：swap_growth=-288.0 MiB, thermal=NOMINAL

**Run 3** · 2026-10-04 22:32　·　thermal NOMINAL　·　swap 6891 MiB→7580 MiB
- arm：`prod-new:CGC_M2_PROFILE=1`
- 命令：`llama-bench（prod-new）-p ? -n ? --warm-skip None（由產物參數重建）`
- Log／產物：[M2_SLAB_REUSE_2026-10-03.json](../../../Backup/spec_tree/M2_SLAB_REUSE_2026-10-03.json)
- 判定：none：swap_growth=0.0 MiB, thermal=NOMINAL

**Run 4** · 2026-10-04 22:42　·　thermal NOMINAL　·　swap 6525 MiB→6950 MiB
- arm：`prod-new:CGC_M2_PROFILE=1`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[M2_SLAB_REUSE_2026-10-03.mtpoff.json](../../../Backup/spec_tree/M2_SLAB_REUSE_2026-10-03.mtpoff.json)
- 結果：pp=120.91　tg=11.34
- 判定：swap：swap_growth=424.7399999999998 MiB, max_swap=6973.62 MiB

## 8. 子目標分解（持續更新；2/5 完成）

- [x] 跑前立項（現狀/目標/假設/驗收） [證](../../../scripts/check/charters/e-m2-slab-reuse-2026-10-03.yaml)
- [x] 用生產級腳本 prod-new ＋ 自己 option 跑實驗臂，留存 log
- [ ] 驗收：① `CGC-M2-PROF` 逐 fill 行 ≥ 200（覆蓋至少一輪完整 prefill 的所有層）； ② 由逐 fill 行算出 pool share 與 FAST/非 FAST 的 ms/MiB，並**指名**哪一項主導； ③ 端點是**計數器／比值**（不走引用閘門）；不得把本趟任何 t/s 當成績。
- [ ] 否證條件：① 逐 fill 行 < 200（量具沒真的武裝／被 allowlist 吃掉）⇒ 先修量具，不判； ② 或 pool/disk 比與收尾行矛盾（逐 fill 加總 ≠ 收尾計數）⇒ 量具本身不可信，回讀碼。
- [ ] 整層 slab 的 pool-reuse 效率：pool/disk 位元組比 ＋ FAST vs 非 FAST 的 ms/MiB

## 9. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `Backup/spec_tree/s1speed.json.logs/live/prod-new.p0_n128_d512_r3.stderr.log` |
| 備註 | 假設：`fill_layer_slab`（llama-expert-cache.cpp:4341）逐專家檢查 `slot_table`： resident 且 slot < usable ⇒ **memcpy 自 pool**（pool_bytes＋），否則收成 segment 走 pread （disk_bytes＋）。收尾印 pool/disk 比。既有產物已顯示 pool:disk ≈ 56:44。 ⚠ 但**機制上有一個反向項**：池命中會讓 `segs.size() != ne` ⇒ 合併大 pread 的 fast path（`:4458` `contiguous`）**失效**，240 顆非 resident 變成 240 次 thread-per-segment pread（`fill_segments_concurrent`，每段一 thread）。 ⇒ 本卡要判定的是**淨效果**：省下的 memcpy/bytes 是否大於打碎 pread 的代價。 |
| 軸性質 | 活躍攻關軸：speculative 攤薄係數 m 與 accept rate a |
| 對應報告 | 0 份 |

- （無）

---

← [**「邊際 verify token ＝ 34.7 ms」由什麼構成…](exp-marginal-decomp.md)　·　[總目錄](index.md)　·　[HTML 版](e-m2-slab-reuse.html)　·　[2026-10-03 operator 裁定「移除所有具名格、pro… →](exp-default-cell-first-2026-10-03.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
