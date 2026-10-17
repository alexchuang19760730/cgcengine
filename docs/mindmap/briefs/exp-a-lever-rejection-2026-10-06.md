# **拉 a：chain 路徑開 rejection sampling… — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：**拉 a：chain 路徑開 rejection sampling 後，有效接受率能到多少？** 具體問三件事： 1. 開 `CGC_MTP_REJECTION=1` ＋ `CGC_MTP_SAMPLER_PARITY=1` 後，逐輪有效接受率 a_eff 是多少？ （理論值 = 1 − TV(p_t, q)；今天精確比對下 a = **0.46541**） 2. 配對後的 S（＝ t/s_on ÷ t/s_off，同 session、同 seed）是否 > 1？ 3. 輸出分佈是否守住了 operator 新口徑（任務結果相同即可）？

- 主題：實驗　·　子目標：**M MTP on 加速**（活躍攻關軸：speculative 攤薄係數 m 與 accept rate a）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產

---

## 1. 目標

**拉 a：chain 路徑開 rejection sampling 後，有效接受率能到多少？** 具體問三件事： 1. 開 `CGC_MTP_REJECTION=1` ＋ `CGC_MTP_SAMPLER_PARITY=1` 後，逐輪有效接受率 a_eff 是多少？ （理論值 = 1 − TV(p_t, q)；今天精確比對下 a = **0.46541**） 2. 配對後的 S（＝ t/s_on ÷ t/s_off，同 session、同 seed）是否 > 1？ 3. 輸出分佈是否守住了 operator 新口徑（任務結果相同即可）？

## 2. 判準

（三分；全中要同時成立） ① **a_eff ≥ 0.70**（逐輪見證，非單點）； ② 配對 **S > 1**（同 session、同 seed、同 shape／warm-skip）； ③ 輸出分佈守住 operator 新口徑（有任務指標者指標相等；無者以預先指定的等價 oracle 判定）。 ⇒ 入 certified，標 **ratchet_exempt** 基準列（跨口徑，不進全局棘輪）。

## 3. 結果

尚無可引用讀數：1 個 run 全部非乾淨（attribution.verdict=swap（swap_growth=3872.5 MiB, max_swap=3969.88 MiB））

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 假設：今日的低 accept **不是 head 的問題**，是口徑造成的： - draft 取 argmax（one-hot q），target 取 sampling ⇒ accept 規則是「逐 token 精確比對 target 抽到的 id」⇒ `P(match) = Σ p_t·p_d`，被 argmax 封頂在 ≈0.465。 - rejection sampling 把規則換成 `min(1, p_t/q)` ⇒ 接受率 = **1 − TV(p_t, q)**， 且**保輸出分佈**（⇒ 與 MTP-off 統計一致，符合 operator 新口徑）。 ⇒ 理論上只要 MTP head 的 q 接近 p_t，a_eff 應明顯高於 0.465。 機制**已完整存在於 codebase**： - 取樣器 `cgc_rejection_accept()` — `common/sampling.cpp:710-744` - 消費端雙閘 — `common/sampling.cpp:819-821`（否則 fallback 精確比對 `:823`） - 開關 `CGC_MTP_REJECTION` — `common/sampling.cpp:794-797`（process 級、只讀一次） - 生產端 `draft()` 填 dist — `common/speculative.cpp:1947-1955` - 容器 caller-owned、預設 nullptr — `common/speculative.h:78`

---

## 5. 子目標綁定 · options · 測試 Log 報告

- **子目標**：`L25-5`　MTP 的產品化上限
- **來源**：子目標看板（唯一來源）　·　置信度 `high`
- **profile**：`prod-new`
- **被測 option**：`CGC_SERVER_MTP=1`
- **儀器開關**（不是被測 option）：`CGC_MTP_PERF=1`
- **CLI**：`--spec-type draft-mtp`；`--spec-draft-n-max 1`
- **arm 1（可複製）**：`prod-new:CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:16;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1;CGC_MTP_PERF=1`
- **結案狀態**：結案（§63，端點＝m 比值門檻）：乾淨窗口（attribution=none、NOMINAL、起跑 free 7159 MB）一趟 k=1 的 m ＝ 0.159（T_draft 13.70 ms/輪 ÷ 86.13；五條閘全過）≤ 目標 0.30。⇒ 卡片自己推的 k 無窮上界 24.59 是由 m=0.474 推出來的。⚠ 09-30 更正：原文寫「實測更小 ⇒ 上界仍成立」是方向反了 —— m 在分母，m 越小上界越高：1000/(85.8×0.159) ＝ 73.30（不是 24.59）。⚠ 兩件事要說清楚：(i) m 在劣化窗只會被吹大（單邊性）⇒ 乾淨窗量到的 0.159 是上界；(ii) prod-new 的 draft 成本比 09-24 舊 build 高 60%（0.159 vs 0.099）⇒ 但「天花板比舊帳面更低」講反了：draft 成本變高 ⇒ m 變大 ⇒ 上限變低；本格量到的 m 比舊帳面小（0.159 vs 0.474）⇒ 天花板反而被抬高到 73.30。兩者不衝突：09-24 舊 build 的 m≈0.099 是「同 cell、pool 3 GiB」的讀數，不能與 prod-new 的 0.159 直接比價；真正的帳應以同一 cell 為準。
- **逐條處置**：整合進子目標　→ `L25-5`　—　拉 a：server 路徑 chain（tree off）開 rejection sampling（min(1, p_t/q)），量逐輪 a_eff 與同 session／同 seed 配對 S。
- **測試 log（實跑）**：[paired.json](../../../Backup/spec_tree/a_lever_paired_20261006/paired.json)　[paired.json](../../../Backup/spec_tree/a_lever_paired_20261006/paired.json)　[paired.json](../../../Backup/spec_tree/a_lever_paired_20261006/paired.json)　[paired_rev.json](../../../Backup/spec_tree/a_lever_paired_20261006/paired_rev.json)　[paired_rev.json](../../../Backup/spec_tree/a_lever_paired_20261006/paired_rev.json)　[paired_rev.json](../../../Backup/spec_tree/a_lever_paired_20261006/paired_rev.json)　[off_hotkv.json](../../../Backup/spec_tree/a_lever_paired_20261006/off_hotkv.json)
- ★ 09-29 起本格的量測口徑改為 prod-new ＋ harness bench（唯一認可入口），不再是 prod25／server ABBA。判準：m ≤ 0.30 且指名它的主項（draft 流量／verify 寬度／pool 預算）。⚠ 單邊性：窗劣化只把 wall 吹大 ⇒ 實測 t_draft ≥ 真值 ⇒ 由它算出的 m 是上界。

## 6. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [MTP k-sweep（verify batch T 成本曲線）](mtp-ksweep.md) | 3a | step ≈ 14.61 + 42.03·T（max resid 11.5）；最佳 k=2 但那格不可移植（合成 accept 0.93~0.99 vs 交付 0.465~0.58 |
| [M：攤薄係數 m 0.474 → ≤0.073](exp-m-draft-cost.md) | 3a | **未跑（有卡、無讀數）**：t/s(k) = 1000·mean_len(k)/(S0·(1+m·k))；k=3 即使 a=1.0 也只 **19.25**、k→∞ 極限 **2 |
| [**「邊際 verify token ＝ 34.7 ms」由什麼構成…](exp-marginal-decomp.md) | 3a | decode 9.97 t/s |
| [M2 slab pool-reuse（整層填的 memcpy 省多少）](e-m2-slab-reuse.md) | 3a | decode 8.26 t/s |
| [同一趟（同 build、同格、同 session）裡： ① caps…](e-victim-classify-2026-10-05.md) | 3a | 尚無可引用讀數：1 個 run 全部非乾淨（attribution.verdict=swap（swap_growth=5369.81 MiB, max_swap=9644.69 M |
| [**寬度軸的因果量測（W1 vs W2，同 session 同 se…](exp-w1w2-paired-2026-10-06.md) | 3a | 實驗進行中（尚無讀數） |

## 7. 運行設置與 Log（生產級腳本 prod-new ＋ 自己 option）

**Run 1** · 2026-10-06 23:46　·　thermal HEAVY　·　swap 0 MiB→4177 MiB
- arm：`prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=3;CGC_MTP_REJECTION=1;CGC_MTP_SAMPLER_PARITY=1`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[paired.json](../../../Backup/spec_tree/a_lever_paired_20261006/paired.json)
- 結果：pp=134.7　tg=10.69
- 判定：swap：swap_growth=3872.5 MiB, max_swap=3969.88 MiB

**Run 2** · 2026-10-06 23:46　·　thermal HEAVY　·　swap 0 MiB→4177 MiB
- arm：`prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=3`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[paired.json](../../../Backup/spec_tree/a_lever_paired_20261006/paired.json)
- 結果：pp=151.03　tg=9.01
- 判定：thermal：thermal worst HEAVY

**Run 3** · 2026-10-06 23:46　·　thermal HEAVY　·　swap 0 MiB→4177 MiB
- arm：`prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=0`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[paired.json](../../../Backup/spec_tree/a_lever_paired_20261006/paired.json)
- 結果：pp=143.08　tg=9.7
- 判定：thermal：thermal worst HEAVY

**Run 4** · 2026-10-06 23:52　·　thermal MODERATE　·　swap 3913 MiB→3953 MiB
- arm：`prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=0`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[paired_rev.json](../../../Backup/spec_tree/a_lever_paired_20261006/paired_rev.json)
- 結果：pp=160.19　tg=11.11
- 判定：thermal：thermal worst HEAVY

**Run 5** · 2026-10-06 23:52　·　thermal MODERATE　·　swap 3913 MiB→3953 MiB
- arm：`prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=3`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[paired_rev.json](../../../Backup/spec_tree/a_lever_paired_20261006/paired_rev.json)
- 結果：pp=134.1　tg=8.41
- 判定：thermal：thermal worst HEAVY

**Run 6** · 2026-10-06 23:52　·　thermal MODERATE　·　swap 3913 MiB→3953 MiB
- arm：`prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=3;CGC_MTP_REJECTION=1;CGC_MTP_SAMPLER_PARITY=1`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[paired_rev.json](../../../Backup/spec_tree/a_lever_paired_20261006/paired_rev.json)
- 結果：pp=112.12　tg=9.58
- 判定：thermal：thermal worst HEAVY

**Run 7** · 2026-10-07 00:23　·　thermal NOMINAL　·　swap 1842 MiB→4653 MiB
- arm：`prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=0`
- 命令：`llama-bench（prod-new）-p 0 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[off_hotkv.json](../../../Backup/spec_tree/a_lever_paired_20261006/off_hotkv.json)
- 結果：tg=9.82
- 判定：swap：swap_growth=2811.1899999999996 MiB, max_swap=4701.06 MiB

## 8. 子目標分解（持續更新；2/5 完成）

- [x] 跑前立項（現狀/目標/假設/驗收） [證](../../../scripts/check/charters/exp-a-lever-rejection-2026-10-06.yaml)
- [x] 用生產級腳本 prod-new ＋ 自己 option 跑實驗臂，留存 log
- [ ] 驗收：（三分；全中要同時成立） ① **a_eff ≥ 0.70**（逐輪見證，非單點）； ② 配對 **S > 1**（同 session、同 seed、同 shape／warm-skip）； ③ 輸出分佈守住 operator 新口徑（有任務指標者指標相等；無者以預先指定的等價 oracle 判定）。 ⇒ 入 certified，標 **ratchet_exempt** 基準列（跨口徑，不進全局棘輪）。
- [ ] 否證條件：① **a_eff < 0.55**，或 ② 配對 **S ≤ 1** 且差值無法歸因於記憶體／暖機。 ⇒ 記錄「rejection 拉不起 a」及其量到的 a_eff，待辦 ② 結案為負。
- [ ] chain 路徑（tree off）開 rejection sampling，量逐輪有效接受率 a_eff 與配對 S

## 9. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/MTP_2X_BOUNDARY_2026-09-19.md §0–§2；scripts/check/charters/e-mtp-m-acc-2026-09-29.yaml；Backup/spec_tree/rs5_*.md；Backup/spec_tree/rs6_dist_consumer_trace_2026-10-06.md` |
| 備註 | 假設：今日的低 accept **不是 head 的問題**，是口徑造成的： - draft 取 argmax（one-hot q），target 取 sampling ⇒ accept 規則是「逐 token 精確比對 target 抽到的 id」⇒ `P(match) = Σ p_t·p_d`，被 argmax 封頂在 ≈0.465。 - rejection sampling 把規則換成 `min(1, p_t/q)` ⇒ 接受率 = **1 − TV(p_t, q)**， 且**保輸出分佈**（⇒ 與 MTP-off 統計一致，符合 operator 新口徑）。 ⇒ 理論上只要 MTP head 的 q 接近 p_t，a_eff 應明顯高於 0.465。 機制**已完整存在於 codebase**： - 取樣器 `cgc_rejection_accept()` — `common/sampling.cpp:710-744` - 消費端雙閘 — `common/sampling.cpp:819-821`（否則 fallback 精確比對 `:823`） - 開關 `CGC_MTP_REJECTION` — `common/sampling.cpp:794-797`（process 級、只讀一次） - 生產端 `draft()` 填 dist — `common/speculative.cpp:1947-1955` - 容器 caller-owned、預設 nullptr — `common/speculative.h:78` |
| 軸性質 | 活躍攻關軸：speculative 攤薄係數 m 與 accept rate a |
| 對應報告 | 0 份 |

- （無）

---

← [**儀器對帳：m 的兩台儀器定標** m（每 draft token…](exp-m-instrument-recon-2026-10-06.md)　·　[總目錄](index.md)　·　[HTML 版](exp-a-lever-rejection-2026-10-06.html)　·　[兩份權威檔對同一個里程碑的「是否已判死」講了相反的話，而沒有閘門會發… →](exp-targets-board-recon-2026-10-07.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
