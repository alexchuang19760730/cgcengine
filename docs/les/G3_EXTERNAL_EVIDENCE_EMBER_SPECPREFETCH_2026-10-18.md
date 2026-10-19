# G3 外部依據：高通 Ember / SpecPrefetch / DraftExpert 對標 —— 為何 MTP 出字轉不了正、而 LES 是正解

日期：2026-10-18　狀態：**外部對標完成（一手公開資料＋本機實測交叉），作為 G3 立項的外部依據**
關聯：LES charter `scripts/check/charters/e-les-router-prefetch-2026-10-17.yaml`（G1–G6）、
上游鏡像 `docs/les/README.md`、0C 判詞 `docs/mtp_union/MTP_UNION_ORACLE_VERDICT_2026-10-18.md`、
MTP 白皮書 `docs/MTP_DRAFTER_TRAINING_WHITEPAPER_2026-10-18.md`、`docs/MTP3_25TPS_ARITHMETIC_2026-09-18.md`。

> **口徑聲明**：本文「外部」數字全部來自下方列出的公開發表/新聞（標註來源），硬體、量化、模型、
> 存儲與 CGC 不同，**只作機制與可行性依據，不可與本機 t/s 直接對比**。「本機」數字均引用既有判詞
> （附檔路徑），為 M4 Air 16GB 一手實測。本文不跑 GPU、不引入新實測。

---

## 一句話

**高通＋Ember 不是「解決了 MTP 轉正」，而是一開始就沒把 MTP 當投機解碼出 token 用**——它把多 token
視窗當成「提前知道下一層要搬哪些專家」的調度視窗，預報只指導後台預取，最終輸出永遠走 target 原生
router（lossless）。這在機制上豁免了我們 MTP 被否決的兩個死穴（α_tok 賠率、bit-exact 翻臉）。
而**高通學術界的 SpecPrefetch 與我們 LES 是同一機制**（next-layer 真實 router、lossless、只異步傳輸），
其存儲速度敏感性數據正好為 G3 的增益前提背書；DraftExpert 則證明「未經訓練對齊的 MTP union 在端側
必然為負」——這與我們 0C 判死同一結構，**不是我們做錯，是這條路缺了訓練/截斷/獨立預取區/專用存儲**。

---

## 1. 兩條 α 的分野（全部判斷基石）

| 軸 | α_tok（token 接受率） | ρ_expert（專家集合覆蓋率） |
|---|---|---|
| MTP 用途 | 投機解碼，draft token **出字** | draft 路由只當**預取 hint** |
| 正確性要求 | 要 bit-identical，過 break-even | 不要求 bit-exact，target 永遠重算 |
| 預報錯的代價 | 整段 reject、白付 verify | 只浪費一次讀，不影響輸出 |
| 本機結果 | IQ3_XXS α=0.43 &lt; k=1 賠率 0.553（0A）；量化 bit 翻臉；數學封頂 19.39 t/s | G1 跨層 82%（LES 真實 router）；MTP 跨 token 僅 α≈0.43、N 短 |
| 代表 | 我們曾經的 MTP fast path（G5 共存驗證） | Ember / SpecPrefetch / DraftExpert prefetch / **我們 LES** |

**關鍵**：Ember 走的是右欄，我們 MTP 被釘死在左欄。兩者不互斥——但 0C 已證明即使把右欄信號假設成
完美（ρ_expert=1 oracle），在本機現有快取結構下對 decode 仍為負、縮池 N=3 也救不回（見 §4、§5）。

---

## 2. 外部對標對象（一手出處）

| 對象 | 形態 | 硬體/存儲 | 模型/量化 | 關鍵數字（外部，勿直比） |
|---|---|---|---|---|
| **高通＋Ember**（產品，2026 驍龍峰會 9/22） | 無量火 Ember 引擎＋Ember Kernel，XCompute 跨 CPU/GPU/NPU；江波龍 iSA 智能預取 | 第六代驍龍8至尊版（Oryon CPU、Adreno Matrix Cores、Hexagon NPU Element Accelerator 共享記憶體 +50%）、**LPDDR6＋UFS 4.1** | 階躍 StepEdge-Omni 30B-MoE；不改模型、不損精度 | 記憶體降 &gt;50%；**prefill &gt;330 / decode &gt;28 t/s**（NPU+GPU 雙引擎較 NPU 單核 +30%） |
| **SpecPrefetch**（arXiv 2607.24787，2026/06） | 用 layer l 的 router input 預測 layer l+1，**只做異步傳輸**，native router 決定執行；輕量 gate-calibration adapter（12.95M 參數） | 驍龍 8 Elite 真機；NVMe/Mid UFS/Slow UFS/SD 對照 | Qwen3-VL-30B-A3B、DeepSeek-VL2-Tiny | 慢存儲 decode **+15%/+20%/+17%**；最快 NVMe 時 **0 增益**（IO 已被 compute 藏住） |
| **DraftExpert**（arXiv 2607.24434，2026/08） | 每層一個常駐小 draft expert，自蒸餾（residual/logit/token＋**router-agreement** loss）；擴張截斷＋非阻塞 target-expert prefetch | SNAPDRAGON 8 Elite（Hexagon HTP v81，~8MB 片上 SRAM，Q4_0，專家 Flash-resident stage 進 shared memory）；另 RTX4090 對照 | DeepSeek-V2-Lite、Moonlight-16B-A3B（top-6） | **訓練免費 top-r 自投機 4.80 &lt; AR 10.18（比不用還慢）**；DraftExpert 15.47（1.52×）；平均 1.45× |
| **BigMoMo**（智源，2026/09，旁證） | 用 spec 驗證視窗解耦專家搬運：分支剪枝＋閃存佈局重組＋批量調度 overlap | 驍龍8 Gen3 / 天璣9300 | Llama-MoE、Mixtral-8x7B 等 4 模型，至 30B | 平均 **4.83× vs 按需卸載**、1.82× vs 最強投機 MoE；已開源 |

> MiniCPM-V 4.7 **LES**（面壁/OpenBMB，iPad Pro M4 2.97→9.81 t/s 3.3×）是我們 G1–G3 的直接移植源，
> 已鏡像於 `docs/les/upstream/`，本文不重複。

---

## 3. 為什麼他們轉得了、我們轉不了（六層差異）

| 層 | M4 Air 16GB＋CGC（本機現狀） | 高通＋Ember／學術同構 |
|---|---|---|
| ① **MTP 用途** | 出 token，要 bit-exact（α_tok 軸）→ α=0.43&lt;0.553 陣亡 | 只預報專家、非同步預取（ρ_expert 軸，lossless） |
| ② **預測器** | 現成 MTP head（為出字訓練，未對齊專家）；跨 token 單步專家相關僅 0.40（0C） | 自蒸餾常駐 draft expert（router-agreement 專訓命中率）＋**擴張截斷**；或 SpecPrefetch 用**下一層真實 router**（零訓練） |
| ③ **存儲通道** | 通用 SSD＋preadv（讀 1 專家 ~1.00ms，上游 iPad 口徑，本機待 G3 實測）；順序讀 1760 MiB/s | UFS 4.1＋iSA 主控**專優化 MoE 隨機讀**，主控固件按上下文/激活規律智能預取 |
| ④ **硬體並行** | 單 GPU 共享統一記憶體/SSD 通道，後台 preadv 與前台搶同一條頻寬 | XCompute 跨 CPU/GPU/NPU；預取 DMA 與 NPU compute **真並行**（BigMoMo 批量重疊） |
| ⑤ **快取結構** | verify union 直接寫進同一個 8G 池（已飽和，143 槽 vs 工作集 175）→ capacity miss ×10.2、hit 96→87%（0C） | 預取進**獨立 staging/ping-pong bank**，不污染熱快取；truncation 保護 |
| ⑥ **量化精度** | IQ3_XXS 3.06 bpw，MTP head/target 誤差大、bit 翻臉 | Q4_0 / INT4（~4 bpw）NPU＋蒸餾對齊，宣稱不損精度 |

---

## 4. 兩個對 CGC 最重要的外部結論

### 4.1 「未訓練 MTP union 為負」是業界共性，不是我們做錯

DraftExpert 論文自己量到（MN=驍龍8 Elite，Q4_0）：**訓練免費的 shared+top-r 自投機只有 4.80 t/s，
比不用的 AR 10.18 還慢**（top-3 接受率仍 &lt;50%），原因是「verify 前多搬一堆專家、然後大量被拒」。
這與本機 MTP 完全同構：

- MTP on/off 乾淨窗 **0.933×**（`docs/MTP_TRANSFERRED_2026-10-07.md` B3 釘死）；
- verify 一步 union 19.05 vs 單步 8.00，每 token 專家讀取 **8.90（+11%）**、verify 步 **1.97×**、淨 1.09×；
- 長期 capacity miss **×10.2**、hit 96.0%→87.1%（0C 判詞、`docs/MTP3_25TPS_ARITHMETIC_2026-09-18.md`）。

DraftExpert 做到 1.52× 是**四件齊備**：(a) 固定足跡常駐 draft expert、(b) router-agreement 自蒸餾提高命中率、
(c) 擴張成本太高就**截斷 verify block**、(d) 非阻塞預取進獨立區。**缺任一項都不成立**（其 top-r 消融即證）。

> 對 CGC 的含義：我們「不訓練、現成 MTP head、union 直寫同一池、無截斷」剛好四項全缺，為負是預期結果。
> 補法不是調 MTP 開關，而是先補 (c)(d)——**這正是 G3 的 preadv worker＋ping-pong 獨立 bank＋擴張截斷**；
> (a)(b) 屬訓練側路（白皮書階段1，自蒸餾 draft expert），等 LES 跑完再評估。

### 4.2 SpecPrefetch＝LES，且其存儲敏感性數據為 G3 增益前提背書

SpecPrefetch 與我們 LES 機制同構（next-layer 真實 router、只異步傳輸、lossless、window-aware 排程），
其最有價值的一張數據是**增益隨存儲速度翻轉**：

| 存儲 | hot 專家讀 | shared 計算 | I/O 是否藏得住 | 預取增益 |
|---|---|---|---|---|
| 最快 NVMe | 0.15 ms | 2.0 ms | 已藏住（0.15≪2.0） | ≈0（4.36 vs 4.31） |
| Mid UFS | — | — | 超出 overlap 窗口 | +15% |
| Slow UFS | — | — | 上關鍵路徑 | **+20%** |
| SD Card | — | — | 上關鍵路徑 | +17% |

**映射到本機**：讀 1 專家 ~1.00 ms、算 1 層 ~1.15 ms（LES README 上游口徑），而無預取時 8 個專家
串行 = 8.0 ms/層，**遠大於 1.15 ms 的 compute 窗口** ⇒ 我們明確落在「慢存儲、I/O 上關鍵路徑」一側，
**G3 把預取 I/O 藏進 compute 有外部增益依據**（前提成立）。

但有一個必須誠實標出的工程風險：**一層窗口（1.15ms）讀不完 8 個專家（8ms）**。所以單靠「當前層
預取當前層」不夠，必須：
1. **提前多層**起讀（G1 已給 recall@d：d8=0.79→d10=0.86→d12=0.90→d16=0.94），depth 拉到視野夠；
2. 後台 **preadv worker＋ping-pong bank** 連續讀、直寫 GPU-visible（G3）；
3. 配套**擴張截斷**：窗口/頻寬讀不完的專家不硬預取，交回按需路徑（SpecPrefetch window-aware scheduler、
   DraftExpert truncation 同此理），避免把預取變成純增 fill IO 去搶當前層（這正是 0C 觀測到的 union 副作用）。

這也解釋了為何上游 LES 在 iPad 上要「預取＋**權重連續 pack 佈局**＋少讀＋直寫 GPU 可見」打包才到 3.3×
（`docs/les/README.md` 官方實測表）——單一個預測信號不夠，存儲佈局與調度要一起到位。

---

## 5. 對 G3 驗收的直接映射

| G3 判據（charter `e-les-router-prefetch-2026-10-17`） | 外部對應 |
|---|---|
| 後台 preadv worker＋ping-pong（READ=8、DEPTH=8），`n_wait_ns` 中位下降 | SpecPrefetch/DraftExpert/BigMoMo/LES 共同的 load-compute overlap |
| depth=8 一層窗口讀完（讀不完則靠提前多層 d10–d16） | LES recall@d 曲線；SpecPrefetch 慢存儲增益前提 |
| window-aware：讀不完的不硬預取（擴張截斷） | SpecPrefetch window-aware scheduler；DraftExpert confidence-expansion truncation |
| 預取進獨立 bank、不污染熱快取（修 0C 的 union churn） | DraftExpert prefetch staging、BigMoMo 批量調度、LES ping-pong 尾部 |
| G4 生產 ABBA off/on≥3 對、配對中位增益 ≥+5% | SpecPrefetch 慢存儲 +15~20%（本機目標先取保守 +5% 作下門檻） |
| 全程 bit-identical | 三者皆 lossless（預測錯只影響傳輸效率，不影響輸出） |

**本機不可直接照搬的部分**：UFS 4.1 專用主控（③）、CPU/GPU/NPU 獨立 DMA（④）是硬體，M4 沒有；
我們只能用軟體 preadv worker＋統一記憶體排程逼近，故 G3 必須以本機 `n_wait_ns`/`file_reads` 實測
決定收益，**不能用高通 28 t/s 或論文 1.5–4.8× 反推本機數字**。

---

## 6. 結論與路線

1. **MTP 出字（α_tok）維持 G5 結論，不調**：本機 α=0.43、量化翻臉、數學封頂；外部亦顯示未訓練自投機
   在端側必然為負（DraftExpert 4.80&lt;10.18）。
2. **decode 25 與縮池 6G 的主票在 LES（ρ_expert 的免費、確定性、bit-exact 版本）**：SpecPrefetch 已在
   同類硬體（驍龍8 Elite）與同一機制上背書慢存儲增益，G1 跨層 82% 強於 MTP 跨 token 43%。
3. **G3 落地清單（外部依據充分）**：preadv worker＋ping-pong 獨立 bank、提前多層（d10–d16）、
   擴張截斷、權重連續 pack（後者評估成本）。
4. **訓練側路（條件性）**：若 LES 全開後仍不足，再依白皮書階段1 評估自蒸餾 draft expert
   （router-agreement loss），補 DraftExpert 的 (a)(b)；那是額外工程與雲端多卡，不提前投入。

---

## 7. 出處（一手）

- 高通＋Ember 產品發布（2026 驍龍峰會，第六代驍龍8至尊版、階躍30B、無量火、江波龍）：
  https://look.huanqiu.com/article/4TKGOV8mnBU
- 江波龍 iSA＋UFS 4.1 智能預取/隨機讀細節：
  http://finance.sina.cn/stock/jdts/2026-09-28/detail-initktnz9715029.d.html
- DraftExpert: Expansion-Aware Self-Speculative Decoding for End-Device MoE Inference：
  <https://arxiv.org/html/2607.24434v1> （PDF：<https://arxiv.org/pdf/2607.24434>）
- SpecPrefetch: Parameter-Efficient Expert Prefetching for Sparse MoE Foundation Models：
  <https://arxiv.org/abs/2607.24787v1> （PDF：<https://arxiv.org/pdf/2607.24787>）
- BigMoMo（手機 MoE 投機驗證視窗，4.83×，開源）：
  <https://hub.baai.ac.cn/paper/9319bccb-1378-4bea-af4e-16533228dc9d> （代碼：<https://github.com/bigmomomo>）
- 上游 LES（本機移植源）：https://github.com/tc-mb/llama.cpp-omni （鏡像見 `docs/les/upstream/UPSTREAM_SHA.txt`）

### 本機數據引用（不重複貼，附檔案）
- 0C union oracle／0A α：`docs/mtp_union/MTP_UNION_ORACLE_VERDICT_2026-10-18.md`、`Backup/mtp_union_oracle_20261018/`
- MTP 成本/verify 實測：`docs/MTP3_25TPS_ARITHMETIC_2026-09-18.md`、`docs/MTP_TRANSFERRED_2026-10-07.md`
- G1 命中率：`Backup/les_g1_hitrate_20261017/hitrate_*.json`；上游時序賬：`docs/les/README.md`
