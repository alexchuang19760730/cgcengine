# 同一趟（同 build、同格、同 session）裡： ① caps… — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：同一趟（同 build、同格、同 session）裡： ① caps 16→17 是否把 ntok=2 的 MTP verify 步從「大-prefill slab 分支」拉回 decode 池路徑 （cb 3.9–6.1 s/step → ms 級、`CGC-PREFILL-STREAM … ntok=2` → 0 行、趟由 incomplete → 完成）？ ② 在 SpAc 已開（prod-new 預設 CGC_SPAC=1／alpha 0.75）的基礎上，把受害者規則換成 「累計路由次數最低者」（`CGC_SPAC_HOT=1`，llama-expert-cache.cpp:718）會不會動 decode 池的 miss/hit —— 兩個 regime：MTP-off（純 decode）與 MTP-on caps17（修好分類後）？

- 主題：實驗　·　子目標：**M MTP on 加速**（活躍攻關軸：speculative 攤薄係數 m 與 accept rate a）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產

---

## 1. 目標

同一趟（同 build、同格、同 session）裡： ① caps 16→17 是否把 ntok=2 的 MTP verify 步從「大-prefill slab 分支」拉回 decode 池路徑 （cb 3.9–6.1 s/step → ms 級、`CGC-PREFILL-STREAM … ntok=2` → 0 行、趟由 incomplete → 完成）？ ② 在 SpAc 已開（prod-new 預設 CGC_SPAC=1／alpha 0.75）的基礎上，把受害者規則換成 「累計路由次數最低者」（`CGC_SPAC_HOT=1`，llama-expert-cache.cpp:718）會不會動 decode 池的 miss/hit —— 兩個 regime：MTP-off（純 decode）與 MTP-on caps17（修好分類後）？

## 2. 判準

① 分類修正：caps17 臂 0 行 ntok=2 slab 指紋、DECPROF ntok=2 步 cb 中位 <100 ms、趟完成（可被 --json 收下）。 ② 受害者規則：拿到兩組同趟配對的 decode/pool hit%（`decode/pool (ensure_slot+batch) hits=a/b`）＋ `final stats … hit rate` 交叉核對；不論方向如實寫出 Δ。 ③ 純 env/白名單（未動 src/）：CGC_SPAC_HOT 的白名單入口由本輪補上，gate 應 PASS（arm_env_dropped 不再有它）。

## 3. 結果

尚無可引用讀數：1 個 run 全部非乾淨（attribution.verdict=swap（swap_growth=5369.81 MiB, max_swap=9644.69 MiB））

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 假設：① `expert_cache_on_topk` 以 `cgc_is_decode_graph(n_tokens, cgc_decode_max_tokens)` 分類 （llama-context.cpp:8073，slab 分支 return 於 :8334）。caps 16 ⇒ 15 routable ⇒ `cgc_decode_bound(15,8,8)=floor(15/8)=1` ⇒ width 1 ⇒ ntok=2 判 PREFILL； caps 17 ⇒ 16 routable ⇒ bound=2 ⇒ 回 decode（pool ＋ remap leaf）。 ② SpAc 的 EMA victim 會把「熱但安靜」的專家衰減掉（engine 註解記的 58.6%→98.6% 覆蓋落差）； SPAC_HOT 用**不衰減**的累計次數選 victim（保住重尾）。若 victim 規則在交付幾何（S=142、support 205） 真有量級，decode/pool hit% 應上移、`final stats` 的 miss 應下降。 上游反證（先驗＝null）：routing oracle 把「所有 placement 工作」的上限量成 +0.2pp 路由質量 （run_server.sh:2544 註解）；09-28 座位曲線（8→143 席）只值 0.3pp。

---

## 5. 子目標綁定 · options · 測試 Log 報告

- **子目標**：—（未歸屬看板 15 格中的任何一格）
- ⚠ 本條不屬於看板 15 個子目標中的任何一格（已認證／已定案／已作廢）⇒ 沒有待跑的臂，也就沒有要綁的 option。
- **逐條處置**：已定案（約束／基準）　—　受害者規則 × MTP 分類修正（同趟 5 臂）：① caps16→17 把 ntok=2 verify 步拉回 decode —— cb(ntok=2) p50 4074.7→38.2 ms、slab fills 553,833→36,922 MiB、file_reads 1.27M→129k ⇒ L25-4 重開前置 #1 可用；② CGC_SPAC_HOT=1 兩 regime 皆無量級：MTP-off hit 96.4→96.6%（+0.15pp、同 requests），MTP-on capacity miss ×2.5、io_bytes ×1.76（負向）⇒ 受害者規則此路關閉（09-20 的 40.6% 為被否證的離線 replay）。⚠ 超訂標記＋swap attribution ⇒ t/s 不引用。判詞與產物 Backup/spec_tree/VICTIM_CLASSIFY_2026-10-05.*
- **測試 log（實跑）**：[VICTIM_CLASSIFY_2026-10-05.json](../../../Backup/spec_tree/VICTIM_CLASSIFY_2026-10-05.json)　[VICTIM_CLASSIFY_2026-10-05.json](../../../Backup/spec_tree/VICTIM_CLASSIFY_2026-10-05.json)　[VICTIM_CLASSIFY_2026-10-05.json](../../../Backup/spec_tree/VICTIM_CLASSIFY_2026-10-05.json)　[VICTIM_CLASSIFY_2026-10-05.json](../../../Backup/spec_tree/VICTIM_CLASSIFY_2026-10-05.json)　[VICTIM_CLASSIFY_2026-10-05.json](../../../Backup/spec_tree/VICTIM_CLASSIFY_2026-10-05.json)
- 受害者規則 × MTP 分類修正（同趟 5 臂）：① caps16→17 把 ntok=2 verify 步拉回 decode —— cb(ntok=2) p50 4074.7→38.2 ms、slab fills 553,833→36,922 MiB、file_reads 1.27M→129k ⇒ L25-4 重開前置 #1 可用；② CGC_SPAC_HOT=1 兩 regime 皆無量級：MTP-off hit 96.4→96.6%（+0.15pp、同 requests），MTP-on capacity miss ×2.5、io_bytes ×1.76（負向）⇒ 受害者規則此路關閉（09-20 的 40.6% 為被否證的離線 replay）。⚠ 超訂標記＋swap attribution ⇒ t/s 不引用。判詞與產物 Backup/spec_tree/VICTIM_CLASSIFY_2026-10-05.*

## 6. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [MTP k-sweep（verify batch T 成本曲線）](mtp-ksweep.md) | 3a | step ≈ 14.61 + 42.03·T（max resid 11.5）；最佳 k=2 但那格不可移植（合成 accept 0.93~0.99 vs 交付 0.465~0.58 |
| [M：攤薄係數 m 0.474 → ≤0.073](exp-m-draft-cost.md) | 3a | **未跑（有卡、無讀數）**：t/s(k) = 1000·mean_len(k)/(S0·(1+m·k))；k=3 即使 a=1.0 也只 **19.25**、k→∞ 極限 **2 |
| [**「邊際 verify token ＝ 34.7 ms」由什麼構成…](exp-marginal-decomp.md) | 3a | decode 9.97 t/s |
| [M2 slab pool-reuse（整層填的 memcpy 省多少）](e-m2-slab-reuse.md) | 3a | decode 8.26 t/s |
| [**寬度軸的因果量測（W1 vs W2，同 session 同 se…](exp-w1w2-paired-2026-10-06.md) | 3a | 實驗進行中（尚無讀數） |
| [**儀器對帳：m 的兩台儀器定標** m（每 draft token…](exp-m-instrument-recon-2026-10-06.md) | 3a | 實驗進行中（尚無讀數） |

## 7. 運行設置與 Log（生產級腳本 prod-new ＋ 自己 option）

**Run 1** · 2026-10-05 13:57　·　thermal NOMINAL　·　swap 4075 MiB→9657 MiB
- arm：`prod-new:CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:17;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1;CGC_DECODE_PROFILE=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=1`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[VICTIM_CLASSIFY_2026-10-05.json](../../../Backup/spec_tree/VICTIM_CLASSIFY_2026-10-05.json)
- 結果：pp=103.87　tg=8.12
- 判定：swap：swap_growth=5369.81 MiB, max_swap=9644.69 MiB

**Run 2** · 2026-10-05 13:57　·　thermal NOMINAL　·　swap 4075 MiB→9657 MiB
- arm：`prod-new:CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:17;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1;CGC_DECODE_PROFILE=1;CGC_SPAC_HOT=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=1`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[VICTIM_CLASSIFY_2026-10-05.json](../../../Backup/spec_tree/VICTIM_CLASSIFY_2026-10-05.json)
- 結果：pp=117.97　tg=7.43
- 判定：swap：swap_growth=157.1899999999987 MiB, max_swap=9753.88 MiB

**Run 3** · 2026-10-05 13:57　·　thermal NOMINAL　·　swap 4075 MiB→9657 MiB
- arm：`prod-new:CGC_SPAC_HOT=1`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[VICTIM_CLASSIFY_2026-10-05.json](../../../Backup/spec_tree/VICTIM_CLASSIFY_2026-10-05.json)
- 結果：pp=124.01　tg=11.69
- 判定：none：swap_growth=-129.1899999999987 MiB, thermal=MODERATE

**Run 4** · 2026-10-05 13:57　·　thermal NOMINAL　·　swap 4075 MiB→9657 MiB
- arm：`prod-new`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[VICTIM_CLASSIFY_2026-10-05.json](../../../Backup/spec_tree/VICTIM_CLASSIFY_2026-10-05.json)
- 結果：pp=116.9　tg=10.08
- 判定：both：thermal=HEAVY, swap_growth=183.9300000000003 MiB

**Run 5** · 2026-10-05 13:57　·　thermal NOMINAL　·　swap 4075 MiB→9657 MiB
- arm：`prod-new:CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:16;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1;CGC_DECODE_PROFILE=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=1`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[VICTIM_CLASSIFY_2026-10-05.json](../../../Backup/spec_tree/VICTIM_CLASSIFY_2026-10-05.json)
- 結果：pp=117.9　tg=6.5
- 判定：both：thermal=HEAVY, swap_growth=8.569999999999709 MiB

## 8. 子目標分解（持續更新；2/4 完成）

- [x] 跑前立項（現狀/目標/假設/驗收） [證](../../../scripts/check/charters/e-victim-classify-2026-10-05.yaml)
- [x] 用生產級腳本 prod-new ＋ 自己 option 跑實驗臂，留存 log
- [ ] 驗收：① 分類修正：caps17 臂 0 行 ntok=2 slab 指紋、DECPROF ntok=2 步 cb 中位 <100 ms、趟完成（可被 --json 收下）。 ② 受害者規則：拿到兩組同趟配對的 decode/pool hit%（`decode/pool (ensure_slot+batch) hits=a/b`）＋ `final stats … hit rate` 交叉核對；不論方向如實寫出 Δ。 ③ 純 env/白名單（未動 src/）：CGC_SPAC_HOT 的白名單入口由本輪補上，gate 應 PASS（arm_env_dropped 不再有它）。
- [ ] 否證條件：① caps17 臂仍印 ntok=2 slab 行或 cb ≥1 s ⇒ 分類器之外還有第二條路把 verify 送去 slab（回讀碼）。 ② SPAC_HOT 配對的 decode hit Δ ≤0.2pp ⇒ 受害者規則在交付幾何上再一次 null（併入 oracle 判詞，不再單獨立卡）。

## 9. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `Backup/spec_tree/MTP_CB_CLIFF_VERDICT_2026-10-03.md` |
| 備註 | 假設：① `expert_cache_on_topk` 以 `cgc_is_decode_graph(n_tokens, cgc_decode_max_tokens)` 分類 （llama-context.cpp:8073，slab 分支 return 於 :8334）。caps 16 ⇒ 15 routable ⇒ `cgc_decode_bound(15,8,8)=floor(15/8)=1` ⇒ width 1 ⇒ ntok=2 判 PREFILL； caps 17 ⇒ 16 routable ⇒ bound=2 ⇒ 回 decode（pool ＋ remap leaf）。 ② SpAc 的 EMA victim 會把「熱但安靜」的專家衰減掉（engine 註解記的 58.6%→98.6% 覆蓋落差）； SPAC_HOT 用**不衰減**的累計次數選 victim（保住重尾）。若 victim 規則在交付幾何（S=142、support 205） 真有量級，decode/pool hit% 應上移、`final stats` 的 miss 應下降。 上游反證（先驗＝null）：routing oracle 把「所有 placement 工作」的上限量成 +0.2pp 路由質量 （run_server.sh:2544 註解）；09-28 座位曲線（8→143 席）只值 0.3pp。 |
| 軸性質 | 活躍攻關軸：speculative 攤薄係數 m 與 accept rate a |
| 對應報告 | 0 份 |

- （無）

---

← [把段邊界重疊柵欄（CGC_OVERLAP_FENCE=41）搬到**…](exp-fence-default-cell-2026-10-05.md)　·　[總目錄](index.md)　·　[HTML 版](e-victim-classify-2026-10-05.html)　·　[**跨 checkpoint 探針**：把 CGC 引擎（交付路徑＝… →](exp-cgc-edge0gguf-2026-10-05.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
