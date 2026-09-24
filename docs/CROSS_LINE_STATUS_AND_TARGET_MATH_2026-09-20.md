# 兩條線的進度對帳 ＋「prefill 250 / decode 25」還差什麼（2026-09-20 17:2x）

**性質：唯讀對帳（不出 GPU、不改引擎）。** 每一格都指回既有產物；本文件不產生新量測。
現場：兩個量測行程同時在跑（見 §0）。

---

## §0 現場：兩條線正在同一台機器上互相搶 GPU

| pid | 誰 | 做什麼 | 已跑 |
|---|---|---|---:|
| 21334 → 21436 | **線 I（cb）** | `l2_ring_delivery.sh` → `mtp_accept_ab.py`，L2 evicted-ring 的 RING=0/64 交錯 A/B（server port 9932，prefill250、MTP on、8 GiB、`--decprof --gpu-timing`） | 2:43 |
| 21798 → 21867 | **線 A（ace）** | `/tmp/ab2.py`，`GGML_METAL_FUSION_DISABLE=1` vs 融合開（llama-bench `-n 32 -r 2 --warm-skip 64 --ctx-size 4096 --spec-type draft-mtp`） | 1:00 |

- **thermal = 2（HEAVY）**、系統 free 9%。
- ⇒ 這兩筆**現在產出的 t/s 都不可引用**。線 A 自己在 §EN-336 的結論是「只有從 thermal 0 且已穩定出發的短輪才拿得到乾淨標籤」；
  線 I 的 L2 預註冊（`Backup/phase_decomp/L2_RING_20260920.conditions.md`）也要求乾淨窗口。
- ★ 這同時是**「線 A 下午 decode 一次都拿不到 ≥12」的最簡解釋**：不是引擎退步，是視窗撞車。
  §EN-336(3) 已經有旁證 —— 今天下午 per-job 時間**更短**（10513 < 17356 µs）而 t/s 更低 ⇒ 慢的不是 kernel。

---

## §1 線 A（ace）— 段邊界／`wait`／`gap` ＋ G4（union）

| 項目 | 狀態 | 關鍵數字 | 出處 |
|---|---|---|---|
| **G0** 儀器確定性 | met | null 884/884 | `targets.json` |
| **G1** 把 gap 藏掉（overlap） | **open，且四面封死** | `gap_L = 1.00×cb_{L-1} + 0.29–0.35 ms`（r=+0.94…+0.999，四支 run 重縮放）；可達下界 **39×0.32 = 12.5 ms = 9.0%**，所以「≤5%」在任何 hook 成本下都不可達；`CGC_SUBMIT_AHEAD=1` 測到 ×1.702 但**故意錯**（GPU 讀 stale remap）⇒ G2 構造性否決 | `targets.json` G1；§EN-261/§EN-267/§EN-275 |
| **S2**（G1 唯一存活路線） | **診斷結案、實作 0 行** | 駐留那半只有 **4 層**在抖動；drain 那半**主機側做不到**。天花板 **18.7**（引擎內部口徑）、可及 **×1.087** ⇒ 建議不做完 | §EN-329；`docs/S2_RESIDENCY_AND_DRAIN_VERDICT_2026-09-20.md` |
| **G3** union 可壓性 | met | 40 個逐層區間**互斥**（coverage = Σunion 到 0.01 ms） | §EN-259/§EN-260 |
| **G4** 縮 union | **open，有靶無槓桿** | 三階儀器已完成：KIND×OP → 逐層 GPU 區間 → **dispatch 普查** → **elementwise 逐張量**。靶從「9 連 ADD 融合」改成「376 次 elementwise dispatch」 | §EN-327/§EN-338/§EN-339 |
| **G5**（union 不改輸出） | open | 884/884 同規則 | `targets.json` |
| 交付基線 | 凍結 | `prod_profile.py`：decode **12.57 ± 2.26**（NOMINAL 全程，12:32）、單臂噪音 **±27%**；prefill **281.51 PASS 250**（15:28）／275.65（熱漂移 refused） | §EN-321/§EN-335/§EN-336 |

### 線 A 今天推翻的東西（三條，都影響「下一步該做什麼」）

1. **「9 連 ADD 融合」已被上游做掉**：探針實測 `n_fuse=7 + 2`（`ggml-metal-ops.cpp:5660-5698` 的 `ADD/SUB/MUL/DIV` 鏈，
   `ggml-metal.metal:1323` 串序累加 ⇒ 逐位元等價是構造性的）。⇒ §EN-334 的「移除 328 dispatch/步、上界 11.5%」**作廢**。
2. **通則：融合開著時，KIND×OP／`CGC-GPUOPS` 的份額不能當 dispatch 成本讀**（ADD 有 421 節點卻只 ~50 dispatch）。
   找槓桿要用 `CGC_DISPATCH_CENSUS`（一步 **1098 dispatch / 1337 節點**，MUL_MAT 254＝23.1% 是真實矩陣乘）。
3. **elementwise 那 376 次 dispatch 的清單**：最浪費的是小的 —— MoE 權重正規化鏈 `sum[1] → clamp[1] → div[8]`
   各 24/步（**其中 48 次是為一顆純量發一個 kernel**），shared-expert gate `sigmoid[1]+MUL[2048]` 各 23/步，
   GDN 支撐 op ≈102/步。⇒ **有靶，但沒有人估過它的毫秒數**（編碼期普查，無 t/s）。

---

## §2 線 I（cb）— 池／快取幾何、預取、G6

| 項目 | 狀態 | 關鍵數字 | 出處 |
|---|---|---|---|
| **F1** cb 的成因 | closed | `cb_layer = 0.46 ms（barrier） + 0.695 ms × m`；邊際 miss 在 **~1.6 GB/s** 被服務（不是磁碟的 823 MB/s）⇒ LINEAR 但**非 disk-bound** | `docs/F1_CB_MISS_REGRESSION_RESULT_2026-09-20.md` |
| **cb 是裝置** | closed | engine 1.698 ms/job vs 裸 `pread` 1.731 ms/job ⇒ **0.98×**。沒有 64 ms 的引擎側低效可撿，只剩 **6–13 ms** | `docs/CB_IS_THE_DEVICE_RESULT_2026-09-20.md` |
| **F2**（預取／重疊） | **null 無效** | cb_B/cb_A = 0.971，但**治療從未執行**：`qa-zh` 沒開 SpAc（`CGC_SPAC` 缺席）＋ per-layer trigger 找不到東西 ⇒ F2-R 設計已重寫 | `docs/F2_F5_OVERLAP_AND_AUDIT_RESULT_2026-09-20.md`、`docs/F2R_OVERLAP_EXPERIMENT_DESIGN_2026-09-20.md` |
| **F3 / F4 / F5** | CLOSED / PASS / **REFUTED** | F5：每層 barrier **不是**可移除的 overhead（與線 A 的 M-F5 同結論） | 同上；`docs/MF5_BARRIER_ATTRIBUTION_2026-09-20.md` |
| reuse distance | closed | capacity miss **74.4%**；**LRU vs Belady 在 S=142 差 31%**（零記憶體）；compulsory 地板 11.9 miss/call ≈ 8.1 ms/call；D2 池曲線 71→142→284 = −63%→−78% 後平 ⇒ **143 槽已在膝點，加大池無用** | `docs/REUSE_DISTANCE_RESULT_2026-09-20.md` + MTP-on 確認 |
| replay 閘門 | **exact** | MTP-off 40/40、MTP-on 41/41，逐項 multiset 相同、hits/misses 各自相等 | `docs/DEMAND_STREAM_EXACT_GATE_2026-09-20.md` |
| **G6**（accept rule） | **未交付** | 5 個配對 rep 同號，mean_len 中位 **+0.375**；但最弱 rep +0.090 < 門檻 +0.118，且 **−0.45 t/s**（步時自己漲了 ~14%） | `docs/MTP_ACCEPT_RULE_G6_2026-09-20.md` |
| **L2**（evicted ring） | **正在跑** | 開跑前發現 **ring 沒有寫入點**（宣告／resize／讀都有，`pick_slot` 從來不寫）⇒ 守衛永假、`prefetch_slot` 一次沒被呼叫。這是本專案反覆付費的那一族（機制存在、開關沒接）。已修 3 處並寫預註冊（含劑量閘門 `reserved>0`） | §EN-340；`Backup/phase_decomp/l2_ring_delivery.sh` |

### 線 I 修掉的一個會污染所有人數字的缺陷

`CGC-DECPROF` + `CGC_GPU_TIMING=1` 每個 verify round 會寫**兩列**（work / shadow），`gap_sum==0` 的是 shadow。
把它們混著取中位會得到 **62.17 ms/round** —— 一個落在兩個眾數之間、兩者都不是的數。
由此**撤回「穩態已經 54 t/s」**：真的 ntok=4 round 是 **247.98 ms**（cb 74.18／wait 159.77／submit 10.09）⇒ **13.61 t/s**。
（`docs/DECODE_STEP_ROW_POPULATIONS_2026-09-20.md`）

---

## §3 目標對帳

### prefill 250：**已達**，剩下的是可重現性工程，不是引擎工作

今天兩次 **281.51**（NOMINAL/NOMINAL，PASS）與 **275.65**（worst=MODERATE ⇒ refused）。
判準本來就是「條件成立才達標」：冷機 + thermal 0 + 已穩定。⇒ 這一題剩下的工作是**排程／視窗紀律**，不是優化。

### decode 25：**在現行交付配方下，算術上不可達**

恆等式：`t/s = mean_len × 1000 / step_ms`，而**交付配方是 k=3**（server argv 逐字 `--spec-draft-n-max 3`）
⇒ **`mean_len ≤ k+1 = 4` 是硬上界**。

| 情境 | step | mean_len | t/s |
|---|---:|---:|---:|
| 今天交付（凍結 profile） | 268（由 12.57 與 3.375 反推） | 3.375 | **12.57** |
| 線 I 的 work-row | **247.98** | 3.375 | **13.61** |
| **接受率做到滿（k=3 的上界）** | 268 | **4.0** | **14.9** ← k=3 的絕對天花板 |
| 同上 ＋ G1 的 ×1.702 天花板（**不可交付**） | 157 | 4.0 | **25.5** |
| 目標 | — | — | **25** |

⇒ **光靠接受率，k=3 的上界是 14.9 t/s**；要碰到 25 必須**同時**拿到
① G1 的序列化窗全移除（×1.702，唯一的實測來自 `CGC_SUBMIT_AHEAD=1` —— 一個**故意寫錯**、被 G2 構造性否決的探針；
其存活路線 S2 的天花板是 18.7 且建議不做完）與 ② acceptance ≈ 0.98（實測 0.4654／0.781，取決於口徑）。
**兩者今天都不存在。**

### 已指名槓桿的量級（分母＝交付步時 ~250 ms）

| 槓桿 | 歸屬 | 估計量級 | 依據 |
|---|---|---:|---|
| **victim rule（動態替換策略）** | 線 I | −10~12 ms（**−4~5%**） | Belady vs LRU 差 31%（S=142） |
| L2 evicted ring 預取 | 線 I | 未知，**先驗低** | F2 家族在出貨 regime 已關門（§EN-322） |
| elementwise／純量 dispatch 融合 | 線 A | **<1 ms（<0.5%）** | 376 dispatch/步，最髒的 72 次是純量鏈 |
| G1 overlap | 線 A | ×1.702 **不可交付** | G2 否決；S2 天花板 18.7 |
| G6 accept rule | 線 I | +11% mean_len 但 +14% step ⇒ **淨 −0.45 t/s** | 實測 |
| 更小的 expert／量化 | 共同 | 線性、無上限，**但要換品質** | reuse distance D2 |

⇒ **已指名槓桿的總和 ≈ +5~10% ⇒ 13.2–13.8 t/s。25 需要 2×，沒有一個槓桿在 3 倍以內。**

---

## §4 建議（按「先做哪一件」）

**① 先停火（今天，立刻）。** 兩個行程同時跑、thermal=2 ⇒ 兩筆數字都會作廢，而它們各自都帶「乾淨窗口」的預註冊。
讓一條線先跑完，另一條等（這已經寫在 `MEMORY.md` 的並行安全條裡，今天只是沒執行）。

**② 把「25」這顆目標拆成兩格並重新定錨（半小時，純文書）。**
`targets.json` 現在的 `decode-25` 混用了兩個口徑：`from` 寫 12.62（HTTP 口徑）、`gates` 的步時 139.16 ms 來自
**E2b（LAYER_CAPS capB）—— 那組已被 G2 判「會改輸出」（M2 25/884），不能當交付步時**。
用交付步時重算後，25 落在「需要改配方變數」而不是「再優化引擎」。建議寫成：
- `decode-25（原）`：標 **`blocked-on-recipe`**，並寫上 k=3 的硬上界 14.9；
- 新增 `decode-16（交付口徑）`：12.57 → 16 = **+27%**，這是 victim rule ＋ G6 修好 step 之後**看得見路徑**的一格。

**③ 唯一「零記憶體、不改數值、有量」的槓桿先做：victim rule（線 I）。**
快取替換策略不碰 reduction order ⇒ **G2/G5 構造性安全**（對比：LAYER_CAPS 改了輸出）。
上界 Belady 31%，實拿未知 —— 但它值得排到兩條線共同的第一位，且與 G1/G4 都不衝突。

**④ 帳要先閉合，不然「還差多少」永遠是猜的。**
線 A 的 `gap ⊆ cb+submit` 交叉檢查以 **10.09 ms** 失敗；線 I 的 cb 74.18 與線 A 的 union 104.6 落在同一個 ~250 ms 步裡，
相加**超過**步時 ⇒ **分解未閉合**。做一次**聯合捕獲**：同一支 run 同時開
`--decprof --gpu-timing` ＋ cb 計數 ＋ `CGC-IDS`／demand dump，在凍結的 prefill250 交付 profile 上逐層對帳。
⚠ 這件必須在**單一視窗**裡做（就是 ① 要解決的事）。

**⑤ 若 operator 堅持 25：唯一有量級的路是 G1 ⇒ 要拍板「做不做完 S2」。**
線 A 目前的建議是**不做完**（天花板 18.7、可及 ×1.087、drain 那半做不到）。這不是技術判斷題而是策略題：
不做完 ⇒ 25 在 k=3 下數學上不可達，目標必須改成「改配方」（k、量化、模型）；做完 ⇒ 也要承認它的天花板 18.7 仍在 25 之下。

---

## §5 誠實邊界

- 本文件的「268 ms」是**反推值**（12.57 t/s 與 mean_len 3.375），不是直讀；直讀的是線 I 的 247.98 ms。兩者差 8%，不影響結論方向。
- `mean_len` 仍有**兩個口徑在流通**（2.40 vs 3.375）⇒ 任何 t/s 換算都要同時寫 mean_len 與它的來源（`targets.json` G6 已這樣要求）。
- 線 A 的 G4 數字全部是**編碼期普查**（當輪 llama-bench 在 command buffer 1 就 OOM）⇒ **dispatch 數可信、毫秒數不可信**。
- 未處理：`llama_bench_matrix.py` 的 arm env **不轉發**到 llama-bench 行程 ⇒ `CGC_DISPATCH_CENSUS` 進不了正式量測（§EN-338 待辦）。
