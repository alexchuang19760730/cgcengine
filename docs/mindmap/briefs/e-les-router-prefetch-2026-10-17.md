# LES 目標-router 預取移植（qwen35moe 同構 · G1–G6） — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：把面壁 MiniCPM-V 4.7 開源的 LES（Lookahead Expert Scheduling）移植到本引擎：當前層用下一層 router 的真實權重 × 當前 token 的 attn_post_norm 打分 argsort top-8，由後台線程把預測專家 preadv 進 ping-pong 尾部、與當前層 GPU 工作重疊；預測只指導預取、不參與選擇（真實路由兜底）⇒ 逐字節一致。分 G1–G6 階段驗證，先釘死命中率再談吞吐；G6 驗證 LES 能否讓 4/6 GiB 小池在吞吐不掉帶下轉正、釋放 2–4 GiB 物理內存（解除靜態超訂／10G OOM）。

- 主題：實驗　·　子目標：**S 序列化消減**（活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產

---

## 1. 目標

把面壁 MiniCPM-V 4.7 開源的 LES（Lookahead Expert Scheduling）移植到本引擎：當前層用下一層 router 的真實權重 × 當前 token 的 attn_post_norm 打分 argsort top-8，由後台線程把預測專家 preadv 進 ping-pong 尾部、與當前層 GPU 工作重疊；預測只指導預取、不參與選擇（真實路由兜底）⇒ 逐字節一致。分 G1–G6 階段驗證，先釘死命中率再談吞吐；G6 驗證 LES 能否讓 4/6 GiB 小池在吞吐不掉帶下轉正、釋放 2–4 GiB 物理內存（解除靜態超訂／10G OOM）。

## 2. 判準

分階 fail-closed（前階不過不進下階）：G1 三套 prompt 離線 per-layer top-8 命中率 ≥80%（≥+5%、prefill 不退化、file_reads/token +≤10%、swap/churn 不惡化；G5 MTP verify 多 token 安全降級、兩檔 bit-exact；G6（縮池轉正，前置 G3/G4） off/on × 8/6/4 GiB 每池一格全矩陣，先重建權威 8/6/4 基線（舊 12.06/10.48/7.67 是轉述不可引用），6G+on ≥ 8G+off×0.95 且物理峰值降 ≥1.5 GiB、4G+on 縮池懲罰收窄 ≥50%，n_wait 不隨池升、swap/churn 不惡化、bit-exact。

## 3. 結果

G1 三套已跑完並於 2026-10-17 裁定 PASS（可預取層 target≥4 口徑，豁免前3層由 prefill warm）：warm-skip64、steady 577 step/22503 cells 每套，跨層 top8 recall mean code 0.7903 / zh 0.7914 / math 0.7911、median 0.875；樸素「跨 token 同層複用」基線僅 0.397（LES 約 2×）；cold 專家覆蓋 0.60–0.62。唯一破線層是 target 層 1（0.095–0.107）：embedding 後未經 FFN 的最淺表示無法跨層預測、且該層 prefill 首 token 即自然進池、無預取價值；層 2–4 爬升 0.60/0.62/0.66，target 10–38 穩態層 0.84–0.90（層27=0.902），末層39=0.56。分口徑均值（三套一致）：全層 target1–39 = 0.790（字面差 0.80 門檻 0.01）；排除首層 target2–39 = 0.809；排除前3層 target4–39 = 0.820；排除前4層 target5–39 = 0.825。recall@d：d8=0.79/d9=0.83/d10=0.86/d12=0.90/d16=0.94（給 G3 depth 決策）。已裁定（2026-10-17，target≥4 可預取層口徑）：三套 target≥4 mean 0.8197/0.8205/0.8207、最低單層 target39 0.54–0.57 均 ≥0.50、steady 577 step 一致過線 → G1 PASS、進 G2；全層 0.790 僅對照、首層豁免。本節點不主張吞吐（跑數期 thermal HEAVY/swap+422~720MiB，t/s 不引用，以 G4 為準）。

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 背景：S 線終判（S_LINE_FINAL_VERDICT_2026-10-07 §3，640 decode step）cold 專家 96.3% 是跨步全新 cold_new、僅 3.7% cold_in_prev ⇒ 跨步歷史預取（prebind）結構上救不了；池大小線已死（8GiB 143 slots/層、每步只補 1.8 個、10GiB Metal OOM）。LES 信號源是當前 token 對下一層 router 的真實打分，與跨步歷史無關、對 cold_new 同樣有效，不屬已判死的啟發式預取。縮池視角（G6）：8GiB 缺 113 槽但每步只補 1.8 個（寬鬆 63 倍）⇒ 池沒被填滿、瓶頸是預取品質而非容量；若 LES 命中率達標，小池只放跨步熱專家＋ping-pong 緩衝、冷專家被提前讀入，常駐需求可降級為及時預取（8→6G＝−25%、8→4G＝−50%）。線索（未落盤、須重測）：4G miss 增 1.91× 而 t/s 10.28→10.60 反略升。上游實測 iPad Pro M4：2.97→9.81 t/s（僅 LES 3.3×）、命中率 >80%、讀 1 專家≈1.00ms/算 1 層≈1.15ms；該 3.3× 不可直接搬（我們已有 8GiB pool+SpAc+fence 到 13.7–20.8，上游基線極冷）。口徑：prod-new 唯一入口、G4 用 warm-skip 64 熱穩態、seed 1、prefill/decode 同測；所有開關預設 OFF。

---

## 5. 子目標綁定 · options · 測試 Log 報告

- **子目標**：`L20-7`　預取前置分支（ρ 已 NO_EFFECT 結案；LES 以更強信號重開探針）
- **來源**：charter　·　置信度 `n/a`
- **profile**：`prod-new`
- **被測 option**：`LLAMA_MOE_LOOKAHEAD_PREFETCH=1`；`LLAMA_MOE_PREFETCH_READ=0|8`；`LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH=8`；`CGC_EXPERT_CACHE_BYTES=8G|6G|4G（G6 每池一格）`
- **儀器開關**（不是被測 option）：`n_hits/n_armed/n_waits/n_wait_ns/n_pred_read/n_pred_published/n_evict_saved`
- **arm 1（可複製）**：`prod-new:LLAMA_MOE_LOOKAHEAD_PREFETCH=1;LLAMA_MOE_PREFETCH_READ=0`
- **arm 2（可複製）**：`prod-new:LLAMA_MOE_LOOKAHEAD_PREFETCH=1;LLAMA_MOE_PREFETCH_READ=8;LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH=8`
- **arm 3（可複製）**：`G6: off/on × CGC_EXPERT_CACHE_BYTES={8589934592(8G),6442450944(6G),4294967296(4G)}（每池一格）`
- **結案狀態**：G1 已 PASS（2026-10-17，純文檔 commit 79e5b24b0）；G2 落點已釘（圖內 ggml_argsort_top_k，Metal/CPU 均已實作，修正舊「無後端」判斷）；commit2（budget+只讀探針源碼 6 檔）已 stage、待乾淨窗口 replay gate。G3–G6 待 G2。
- （同條另掛：`e-union-cache-shrink-2026-10-14（MTP 前瞻信號，正交可疊加）`）
- **逐條處置**：整合進子目標　→ `L20-7`　—　SSD miss 串行等待的消除屬預取/資料搬運軸（S）。ρ 在該格判 NO_EFFECT（近似 gate 信號不夠）；LES 換用目標層 router 真實權重打分，是同一格上信號更強、且有上游同構開源實證（命中率>80%）的新探針，故重開而非復跑 ρ。
- **測試 log**：無實跑 log（本條的證據是下面的「對應報告」，不是量測產物）
- G1 純離線無臂；G2 起才有 on 臂；G6 須在 G3/G4 過後，每池一格各出完整 json（避免 poolsweep 單格配臂被拒跑）。開關預設全 OFF；ρ 判死的是近似 gate，LES 用真實下一層 router，不在原判詞範圍。

## 6. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [CGC_EB_NOFILL 診斷臂（fill 成本）](io-nofill.md) | 3a | fill 3.955 → 0.206 ms/step（−95%）⇒ 機制證；生產口徑同步 fill 僅占 step 4.7% |
| [單段提交（41 段 → 1 段）](s1-segbatch.md) | 3a | 判詞：41 段→1 段的 A/B **作廢**（兩端都不可引用）。本節點不主張吞吐。**補（09-29）**：它的**可交付上界**＝序列化 − 必須還回去的 fill ⇒ **1 |
| [異步 gather 流水線（單段＋miss 後台補＋局部重算）](s1-asyncgather.md) | 3a | E0 已給出決定性答案：前提成立——S1（暖池 8 GiB）輸出 文摘文摘… 且 logits 全非有限（引擎自蓋 INVALID），同一輪的分段臂是 42 ⇒ 異步 fill＋補 |
| [ρ 路線（按層批次化 prefetch）](cache-rho.md) | 3a | 判活，但本節點**不主張吞吐**：覆蓋 0.849、視窗 1.30~1.59 ms、每步付 4.76 ms GPU 插入 ⇒ 有前置條件。 |
| [prebind／方案 A（預指派 slot）](cache-prebind.md) | 3a | 判活，但本節點**不主張吞吐**：qu2/qu3 全過、qu1 邊緣（預指派 slot，提前一整步發起 ⇒ 視窗 ≈ 一步）。 |
| [G0–G7 序列化／調度消減（41 段→1 段、縮 union、藏 …](exp-s-retro.md) | 3a | 判詞：11.97 那一族**作廢**（該臂 9 個 run 全部非乾淨）。本節點不主張吞吐。 |

## 7. 與其它路線的關鍵差異

| 對象 | 它的做法／結果 | 本條目差別 |
|---|---|---|
| cache-prebind（跨 token 歷史 ids） | 用上一步同層真實 ids 提前一步 fill；S 線實測只覆蓋 3.7% cold_in_prev，已判死 | 用當前 token 下一層 router真實打分，對 96.3% cold_new 同樣有效；深度只 1 層但確定性高 |
| cache-rho（按層批次 prefetch） | 同 token 近似 gate（跳 attn），覆蓋 0.849 但視窗 1.3–1.6ms、insert 4.76ms，NO_EFFECT（2026-10-01） | 不跳 attn（用 attn_post_norm 後的真實特徵），構圖復用 build_lora_mm，開銷藏在當前層 GPU 工作內 |
| e-union-cache-shrink（MTP 前瞻 union） | 用 MTP N 個 draft token 的未來 router 聚合預取，依賴 draft accept（MTP 已判交付無淨加速） | 不依賴 MTP，每個 decode token 都生效；信號正交可疊加，本卡先行 |
| 擴池 10GiB | Metal working set OOM（上限 11453 MiB）；縮池曲線證容量不是槓桿 | 不增常駐：專家仍在 SSD，只提前把下一層真正要用的讀進來 |
| 上游 iPad 3.3× 聲稱 | 基線 2.97 t/s（極冷、無任何池/重疊） | 我們基線 13.7–20.8，邊際須按 G4 實測，不許諾 25 |
| 4/6 GiB 縮池（純調容量） | 記憶曲線 8/6/4G＝12.06/10.48/7.67 已被查證為轉述（全語料僅 2 筆 4G、6G 無一手）；poolsweep p4–p7 被 cell contract 拒跑；純縮池無預取補強，容量缺口補不回 | G6 用同一 LES 預取補小池新增 miss：off/on×8/6/4 每池一格，量「縮池懲罰是否收窄」；先補權威一手基線，目標 6G+on 追平 8G+off（−5% 內）且物理峰值降 ≥1.5GB |

## 8. 成敗點（風險 → 驗證）

| # | 風險 | 驗證 |
|---|---|---|
| 1 | 命中率依賴 router 跨層相關性（相鄰層 attn_post_norm 對下層 gate 的預測力） | G1 離線三套 prompt 分層驗證，<60% 直接否證，零引擎改動 |
| 2 | 一層 compute 窗口讀不完 8 專家（本機 SSD 帶寬/專家 1.07–1.12MiB、SSM 混合層 31/41） | G3 先量「讀 1 專家 ms vs 算 1 層 ms」；讀不完降 depth 或退回 READ=0 淘汰保護 |
| 3 | 後台 IO 與當前 gather 爭用、污染 Metal 隊列 | ping-pong 尾部 + 讓步；對照 bg on/off，cb/union gap 不得上升 |
| 4 | bit-exact：預測指針與真實路由競態 | 預測只指導 preadv、不進選擇路徑；M1/M2/M3 對 host oracle，G5 守 n_tokens==1 |
| 5 | MTP verify 多 token 段（ubatch>1）預測構圖失效 | 與上游一致顯式降級不預測；MTP on/off 各跑 G2/G3 |
| 6 | SSD 讀頻寬/預取深度撐不起小池（池越小、即時補的量越大），後台 IO 反成瓶頸 | G6 全矩陣量 n_wait_ns 是否隨池縮小上升、n_hits 是否上升；懲罰不收斂即封閉縮池、保留 8G，僅 6G 過則上限登記 −25% |

## 9. 子目標分解（持續更新；2/10 完成）

- [x] 跑前立項：charter（現狀/目標/假設/分階驗收） [證](../../../scripts/check/charters/e-les-router-prefetch-2026-10-17.yaml)
- [x] G1 離線命中率：三套 prompt 各 ≥128 step，下一層 top-8 vs 真實路由，逐層分組（零引擎讀盤改動） [證](../../../2026-10-17 裁定 PASS（可預取層口徑 target≥4，豁免前3層由 prefill warm）：三套 target≥4 mean 0.8197/0.8205/0.8207、最低單層 target39 0.54–0.57 均過線，全層對照 0.790/0.791/0.791、同層基線 0.397。判詞 docs/LES_G1_HITRATE_VERDICT_2026-10-17.md；證據 Backup/les_g1_hitrate_20261017/；純文檔 commit 79e5b24b0（判詞/腳本/prompt/charter/mindmap）。)
- [ ] G2 圖內預測（READ=0，只預測不讀盤）：照搬上游 qwen35moe.cpp:455-466 build_lora_mm(il+1.ffn_gate_inp, attn_post_norm)→ggml_argsort_top_k(8)→ggml_cont，cb(prefetch_topk) 抓 top8 在線對真實路由 n_hits/n_armed；本引擎無 moe_slots（不接 set_lookahead_pred，prefadv worker 屬 G3）；驗收在線對齊離線±1%、bit-exact、decode 退化<2% [證](../../../2026-10-17 落點核實並修正舊判斷：ggml_argsort_top_k 在本 fork Metal（ggml-metal-ops.cpp:7818 ggml_metal_op_argsort + argsort_merge kernel、device.m:1303）與 CPU（ggml-cpu/ops.cpp:8329 std::sort）均已實作，非「無後端會 UNIMPLEMENTED」⇒ 走圖內 argsort（GPU kernel、無 GPU→host 回讀同步），不必 host std::sort 回調。argsort_top_k=argsort DESC+view 取前 k，輸出 I32 索引（ggml.c:5382）。上游同構 docs/les/upstream/qwen35moe.cpp:455-466。只讀 cgc_les_h 探針（G1 抓下層 router 原始 logits）已隨 commit2 staged；G2 再加圖內 argsort top8 與在線對賬。無 llama-moe-slot.*（set_lookahead_pred 無對應設施，G3 用本引擎 preadv worker 取代）。)
- [ ] commit2：budget 正確口徑（dense+pool，#37）+ 整組只讀診斷探針（A0/A1/B0/IO_SHAPE/BHEAD/LES）入庫，開關預設 OFF、byte-identical；刻意不含 affinity/metal/strata 功能碼與 dylib [證](../../../2026-10-17 已 stage 6 檔（659+/45-，跨檔功能關鍵字審計全 0）：budget_preflight.py/budget_gate.sh（實跑 8G dense555+pool8192=8747、4G=4651 均<16384 不超訂，舊雙重計 expert 口徑作廢）；run_server.sh（budget 段 + 探針 allowlist，已排 ncmoe/exact-cache-verify/mmid-dqhash/affinity 功能透傳，bash -n 通過）；llama-context.cpp（13 個純探針 hunk 501+/1-，7 個 affinity reorder 功能 hunk 留工作區）；qwen35moe.cpp/ggml-backend.cpp（純探針，複用 HEAD 已有的 cb/cgc_td_cb/cgc_rho_probe 路徑）。待乾淨 GPU 窗口（重啟 swap=0、thermal nominal、退 WorkBuddy/Chrome）下 git commit：ALLOW_ENV_GATED_BIN=1 跳檢查8 binary 同步、不帶 dylib，replay gate（3 profile×3 對 baseline）不可豁免、不能 --no-verify。commit1 純文檔=79e5b24b0。)
- [ ] G3 後台 preadv worker + ping-pong，埋點 n_wait_ns/n_pred_read/n_pred_published，量本機讀算時序窗口
- [ ] G4 生產口徑 ABBA（warm-skip64/seed1/prefill+decode 同測），≥+5% 且 swap/churn 不惡化
- [ ] G5 MTP on/off 共存與 n_tokens>1 安全降級，兩檔 bit-exact
- [ ] G6 LES×4/6GiB 縮池轉正：off/on×8/6/4 每池一格，先重建權威基線，量縮池懲罰收窄與物理峰值（wired/swap/churn）
- [ ] 驗收通過：G1≥80% 且 G4 配對中位 ≥+5%、G6 6G+on 追平 8G+off（−5%內）且峰值降 ≥1.5GB、各階 bit-identical，結果回寫本節點與排行榜
- [ ] 否證：G1<60% 整卡收束；G2 開銷≥2%/bit 翻臉；G3 窗口讀不完退回輕量模式；G4<+5%；G6 縮池懲罰不收斂則保留 8G（與 spec-cost-gate/MTP 判詞合流）

## 10. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `判詞 docs/LES_G1_HITRATE_VERDICT_2026-10-17.md；原始證據 Backup/les_g1_hitrate_20261017/（三套 sidecar .bin / a1raw stderr / harness json / hitrate json / prompt / PROVENANCE_NOTE）；docs/les/README.md（上游鏡像 SHA 88bd37eb，MIT）；charter scripts/check/charters/e-les-router-prefetch-2026-10-17.yaml；探針頁 docs/les/les_router_prefetch_probe.html；commit1 純文檔 79e5b24b0（G1 判詞/腳本/prompt/charter/mindmap）；commit2 源碼已 stage 待乾淨窗口（見 st-commit2 log）` |
| 備註 | 背景：S 線終判（S_LINE_FINAL_VERDICT_2026-10-07 §3，640 decode step）cold 專家 96.3% 是跨步全新 cold_new、僅 3.7% cold_in_prev ⇒ 跨步歷史預取（prebind）結構上救不了；池大小線已死（8GiB 143 slots/層、每步只補 1.8 個、10GiB Metal OOM）。LES 信號源是當前 token 對下一層 router 的真實打分，與跨步歷史無關、對 cold_new 同樣有效，不屬已判死的啟發式預取。縮池視角（G6）：8GiB 缺 113 槽但每步只補 1.8 個（寬鬆 63 倍）⇒ 池沒被填滿、瓶頸是預取品質而非容量；若 LES 命中率達標，小池只放跨步熱專家＋ping-pong 緩衝、冷專家被提前讀入，常駐需求可降級為及時預取（8→6G＝−25%、8→4G＝−50%）。線索（未落盤、須重測）：4G miss 增 1.91× 而 t/s 10.28→10.60 反略升。上游實測 iPad Pro M4：2.97→9.81 t/s（僅 LES 3.3×）、命中率 >80%、讀 1 專家≈1.00ms/算 1 層≈1.15ms；該 3.3× 不可直接搬（我們已有 8GiB pool+SpAc+fence 到 13.7–20.8，上游基線極冷）。口徑：prod-new 唯一入口、G4 用 warm-skip 64 熱穩態、seed 1、prefill/decode 同測；所有開關預設 OFF。 |
| 軸性質 | 活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉 |
| 對應報告 | 0 份 |

- （無）

---

← [兩份權威檔對同一個里程碑的「是否已判死」講了相反的話，而沒有閘門會發…](exp-targets-board-recon-2026-10-07.md)　·　[總目錄](index.md)　·　[HTML 版](e-les-router-prefetch-2026-10-17.html)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
