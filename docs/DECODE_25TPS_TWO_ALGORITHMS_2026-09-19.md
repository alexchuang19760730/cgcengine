# 兩個演算法：把 decode 從 8.9 t/s 推向 20–25 t/s（2026-09-19）

來源：`docs/DECODE_SYNC_BATCH_2026-09-19.md`（step 分解）、`docs/MTP_VERIFY_SPLIT_2026-09-19.md`
（draft/verify 單價）、`docs/EXPERT_CACHE_THRASH_2026-09-19.md`（miss 結構）。
兩者都不是「量化」或「加池」，而是改**結構**。

---

## 為什麼是這兩個：一行算術

decode（MTP off、8 GiB、實測）每步 108.8 ms：

| 成分 | ms | 可動嗎 |
|---|---|---|
| GPU busy | ~46 | 要更多 token 攤（⇒ 演算法 ②） |
| **cb（強制填充，擋在 submit 前）** | **27** | **⇒ 演算法 ①** |
| submit + 其他 CPU | ~4 | |
| GPU gap（空等） | ~35 | `gap ≈ 1.2 × (cb + submit)` ⇒ **它是 cb 的果** |

**關鍵算術：每層的填充 0.66 ms（27/41）< 每層的 GPU 工作 1.12 ms（46/41）。**
⇒ 填充**在原理上完全藏得住**。今天藏不住，是因為每層的 union 只在該層的 argsort 之後才知道，
而那段 CPU 時間裡 GPU **沒有任何已提交的工作可做**（`gap = 0` 的隱藏 = 完全沒有重疊）。

所以 ① 的目標：**讓 GPU 在 CPU 填冷專家時有事做** → step 108.8 → ~50 ms → **~20 t/s**。
（這正是「池已經熱」時量到的 17.9–20.1 t/s —— 那個數就是①的天花板，實測已存在。）

而 25 t/s（40 ms/token）還過不去，因為 **GPU busy 46 ms/token 是硬牆**：
快取無限快也只有 21.7 t/s。要跨過去必須每 46 ms 的 GPU 工作產出 >1 個 token ⇒ ②。

---

# 演算法①：Split-MMID Fill Overlap（`CGC_SPLIT_MMID`）

## 機制

每層的 routed-expert 集合拆成兩組：

- **H = resident**（`slot_table[e] >= 0`，不需要 I/O）—— 約 94%（143 slot 下命中率 93.9%）
- **C = cold**（需要填充）—— 每層每步約 0.66 個 expert

圖上建**兩個** `mul_mat_id` 加一個 add：

```
          argsort(il) ──► ids_H (leaf, host-written) ──► MMID_H(down/up/gate) ─┐
                        └► ids_C (leaf, host-written, 固定寬度) ──► MMID_C ────┴─► add ─► out
```

執行序（**這就是全部的價值**）：

1. hook(il) 讀 ids、算出 H / C 兩組，寫入 **ids_H leaf**；
2. dispatcher 在 ids_H 這個 host-written leaf 上分段 ⇒ **立刻提交 H 段**；
3. **GPU 開始算 MMID_H（每層約 1.05 ms 的工作）**；
4. 同一時間，CPU 在 worker pool 上發 C 的填充（每層約 0.66 ms 的 I/O）；
5. 填充落地 → 寫 **ids_C leaf** → 提交 C 段 → GPU 算 MMID_C 並加到結果上。

⇒ 填充與 GPU 計算**重疊**。今天兩者是串的（`gap ≈ 1.2 × (cb+submit)` 就是證據）。

## 兩個必須先解決的設計問題（都已找到解法）

**(a) 形狀不穩**：C 的大小每層每步都變 ⇒ `ggml-alloc` 每步重新 sizing、graph 不能重用。
**解法**：C 用**固定寬度** `CGC_SPLIT_MMID_C=<n>`（預設 8），不足的部分指向引擎**已經有**的
保留 ZERO slot / ZERO expert（`llama_expert_cache_usable_slots`、`ZERO-slot` 那套），
多算的貢獻為 0。形狀固定 ⇒ 圖可跨步重用，且與現有 ZERO 機制同一條路。

**(b) 加法的位元一致性**：`MMID_H + MMID_C` 的累加順序與單一 `MMID` 不同 ⇒ 直接威脅 M1/M2/M3。
**解法與風險揭露**：先讓它**過 gate**。若 gate 紅，就改用 `CGC_ADD_ORDER` 已經建立的那套
順序控制（把 H 與 C 依 expert id 排序後再合併），或者接受它是一個**只在 M3=ON 路徑**上的
改動並另立參考。**未通過 gate 之前，任何速度數字都不能引用**（這是這個專案一路的規矩）。

## 落點（已核對）

- `src/llama.cpp/src/llama-graph.cpp:2747`（`build_moe_ffn` 的 `cgc_dc_fuse` 分支）與
  `:2769` 之後的一般 `build_lora_mm_id` 路徑 —— 兩個 MMID 都在這裡生。
- `src/llama.cpp/src/llama-context.cpp:4844` `expert_cache_on_topk`（hook）：分割 H/C、
  寫兩個 leaf、決定填充的**發起時機**（必須在 H 段提交之後）。
- 既有的 S1（`CGC_SLOT_TABLE_GPU`，`llama-graph.cpp:2116`）是**同一條線**的前置：
  它把 remap leaf 從 host 搬到 GPU，正好說明「host-written leaf 是分段的原因」。
  Split-MMID 可以在 S1 之上（或之下）運作，但**先把 S1 的 gate 過完**再疊，否則兩個變因混在一起。

## 期望值與否證條件

| 量 | 今天 | ① 成功 | ① 失敗的樣子 |
|---|---|---|---|
| `cb`（`CGC_HOOK_SPLIT`） | 27 ms | ~5–8 ms | 不動（⇒ 分段沒提早提交） |
| `gap` | 35 ms | ~8–12 ms | 不動（⇒ 假設②的依賴分析錯了） |
| step | 108.8 ms | **~55–60 ms** | >95 ms |
| decode | 8.9 t/s | **17–18 t/s** | 沒變 |
| M1/M2/M3 | 9/9/9 | **9/9/9（硬條件）** | 紅 ⇒ 加法順序要重做，不是放棄 |

**否證**：如果 step 停在 ~95 ms（cb 消失但 gap 不動），表示 GPU 在 H 段上沒有真的跑起來
（分段沒生效），那 ① 的機制就是錯的 —— 這要在同一輪用 `CGC-HOOKSPLIT` + `CGC-GPU` 兩邊同時看。

---

# 演算法②：Batched Draft（n-gram 提議 + 單次 T=K 的 head 前向）

## 為什麼：40 ms 是「每次呼叫」的成本，不是「每個 expert」

實測（`MTP_VERIFY_SPLIT`）：

| | 每 token 的 expert union | 每 token 牆鐘 |
|---|---|---|
| trunk verify（T≈2.4–3.5） | 7.8 | **~10.1 ms** |
| **draft forward（T=1）** | **8.00** | **~40 ms** |

**一樣的 expert 數，貴 4 倍**；而一個 nextn 層 40 ms vs 主幹一層 2.78 ms（純 GPU 約 1.1 ms）
⇒ 14–36×。所以 40 ms 裡絕大多數是**每次呼叫的固定成本**（圖建構 + 提交 + sync + 取樣），
而 draft 是 K 個**獨立的 1-token 呼叫**（`llama_decode(ctx_dft, batch)`，n_tokens=1，
`common/speculative.cpp:268/305/372/720/772/837/1129/1176/1599/1734`）。

**可攤的證據就在同一張表上**：verify 那條 T≈2.4–3.5 的路徑把同樣的 MoE 做到 10.1 ms/token
⇒ **批次化把固定成本攤掉了**。

## 機制

把「K 個自迴歸的 40 ms 呼叫」換成「**一次 T=K 的前向**」：

1. **提議**：用 `ngram-map-k` 產生 K 個位置（已存在，實測 draft 成本只 +10 ms/token，
   即 `ngram1 123.94 − plain 113.85`，**沒有 draft 前向**）；
2. **批次評分**：把這 K 個位置**一次**餵進 MTP head（T=K 的一次前向），而不是 K 次 T=1；
3. **verify**：照現行 batch verify（`+10.1 ms/token`）。

成本模型（`d` = 每個 draft token 的 head 成本）：`d_eff ≈ (T=K 的單次前向成本)/K`。
若一次 T=3 的前向 ≈ 3×10.1 = 30 ms，則 `d_eff ≈ 10 ms`，而今天是 K×40 = 120 ms。

## 期望值（用已量到的三個實測點校準過的模型）

| 情境 | B | d | k | accept | tok/step | ms/token | t/s |
|---|---|---|---|---|---|---|---|
| 今天 MTP OFF（實測 8.88） | 108.8 | – | 0 | – | 1.00 | 108.8 | 9.2 |
| 今天 MTP ON k=3（實測 234 ms/step） | 108.8 | 40 | 3 | 0.53 | 1.96 | 132.2 | 7.6 |
| ① 成功、d 仍 40 | 50 | 40 | 3 | 0.53 | 1.96 | 102.2 | 9.8 |
| ② 成功、① 未做 | 108.8 | 10 | 3 | 0.53 | 1.96 | 71.9 | 13.9 |
| **① + ②（accept 0.53）** | 50 | 10 | 3 | 0.53 | 1.96 | **61.0** | **16.4** |
| **① + ② + 更深的 k / 更好的提議** | 50 | 10 | 4 | 0.70 | 2.77 | **43.0** | **23.3** |

⇒ **② 單獨不夠、① 單獨也不夠；兩個都要，而且還要 accept 上去，25 才在射程內。**
這與使用者原始問題的答案一致，這張表就是它的量化版本。

## 否證條件

- **先量、再寫**：在做任何②的改動前，必須先把 draft forward 的 40 ms 拆成
  {graph build / buffer alloc、dispatch+sync、取樣與 readback、MoE 數學}。
  如果 40 ms 其實是**每次呼叫重建圖**，那②的正確形狀是「重用 draft ctx 的圖」，
  而不是批次化 —— **演算法會不一樣**，所以這一步不能跳。
- 如果拆完發現 40 ms 是 MoE 數學（8 experts × 3 matmul 在 M=1），那②要改成
  對 head 走更便宜的稠密近似，或把 head 的 8 experts 換成常駐的單一稠密路徑。
- `plain_match` 必須維持 TRUE（batch verify 與逐 token argmax 一致）；accept 必須 ≥ 今天的 0.51。

## 落點

- 提議端：`common/speculative.cpp` 的 `draft-mtp` 實作（n-gram 已有 `ngram-map-k`）。
- 評分端：draft `llama_decode(ctx_dft, batch)` 的迴圈改成單次 T=K 的 batch；
  `n_mtp_layers = llama_model_n_layer_nextn(...)`（:1323）決定圖的形狀。
- 位元一致性：verify 的目標分佈不能變 ⇒ 沿用 `plain_match_ab.py` 當閘門。

---

## 執行順序（為什麼這樣排）

1. **① 先做**：它不需要新東西（分段、ZERO slot、hook、`CGC_HOOK_SPLIT`、`CGC-GPU` 全都在），
   而且它的期望值（8.9 → ~18 t/s）是**兩者中較大的單獨增益**，且立刻把 ② 的基數 B 從 108.8 降到 ~50。
2. **② 的第二個未知數（40 ms 的組成）先量**：它決定②的演算法形狀，且量它比寫它便宜。
3. 每一個里程碑都跑 `m123_oracle_gate.py --profile prefill250`：**過 gate 才有資格談速度**。
4. 每一輪都記 `engine_digest`（`engine_freeze.py`）：這台機器有平行 session，
   「同一顆 build」必須是事實而不是假設。

---

# 真實測試（2026-09-19 13:33，`CGC_DECODE_PROFILE=1` + `CGC_DECODE_PROFILE_ALL=1` + `CGC_GPU_TIMING=1`）

一輪、生產 profile（prefill250 ⇒ 8 GiB ⇒ 143 slot/layer）、MTP on、greedy、
`n_predict=96` ⇒ 92 token。**這輪直接量到每一步的 wait/cb/submit/gpu/union/gap，
所以它同時檢驗了①與②兩個前提 —— 而其中一個被推翻。**

原始資料：`Backup/cgc_logs/llama_server_20260919_133356.log`（95 條 step 線；
base = `layers=40`、draft = `layers=1`）。這一輪有 profiling 的同步成本 + 機器負載
（swap 3.5–3.9 GB），所以**絕對值比無 profiling 的基線高**；下表只用**比值**與**結構**。

## ① 被證實（而且比預期更乾淨）

| base step（40 層，n=40，跳過 warm-up） | 中位 | 平均 |
|---|---|---|
| total | 297.88 | 462.30 ms |
| **cb** | **36.81** | 171.16 ms |
| gpu_sum | 247.10 | 295.26 ms |
| union_sum | 241.81 | 261.95 ms |
| gap_sum | 77.55 | 220.28 ms |
| submit | 7.10 | 8.38 ms |

**cb 佔 base step 的 37%（by means）。** 逐層換算：

> **每層 gpu_sum/40 = 7.38 ms，每層 cb/40 = 4.28 ms，cb/gpu = 0.58。**

⇒ **每層自己的 GPU 工作比它要填的量還大** ⇒ split-MMID 的 H 組（resident 子集）
**足夠把整層的填充蓋掉**。這是①的設計前提第一次被**直接量到**，不是從總量推的。

同一輪還給了一個內部對照：**layer 40（MTP head，256 slot＝全常駐）的 cb 中位數是 0.02 ms**，
而 40 個 pooled 層是 36.81 ms。也就是說「填充成本」正是容量效應本身，全常駐就歸零。

## ② 的前提被推翻（直接量測 vs 先前的相減）

| draft decode（1 層，n=54） | 中位 | 平均 |
|---|---|---|
| **total** | **11.81** | 12.58 ms |
| gpu | 10.82 | 14.69 ms |
| cb | **0.02** | 1.24 ms |
| gap | 0.00 | 0.00 ms |

- **54 個 draft decode 合計 679 ms ＝全部 decode 時間的 3.54%。**
- 先前「draft forward ≈ 28–51 ms **每 draft token**（中位 40）」是**相減法**（`mtp1 − ngram1`）
  推出來的；**直接量到的是一整個 draft decode 11.8 ms**，差了 3–4 倍。
- 「nextn 一層貴主幹一層 14×」也不成立：這一輪主幹層 gpu 7.38 ms、nextn 層 gpu 10.82 ms，
  是 **~1.5×**，不是 14×。
- ⇒ **演算法②（把 draft forward 批次化）是錯的方向**：MTP 的時間 96.5% 在
  **verify/base step**，不在 drafter。舊的 40 ms 應該住在 `speculative.cpp` 每輪的**host 端
  簿記**（取樣/readback/accept 判定/KV 處理）或 bench 臂的幾何差異，而不是 draft 的
  `llama_decode(ctx_dft, …)`。

## 修正後的第二個槓桿

既然 MTP 的成本在 verify step，第二個演算法應該改成：**讓 verify batch 的每 token 成本回到
plain 的水準**，而不是讓 drafter 更便宜。可從同一輪資料驗的機制：verify 的 per-layer union
比 decode 大（ntok 2–8 ⇒ 最多 8×8），而 union 變大直接推高 cb（capacity miss）—— 這與
模擬得到的「足跡 ∝ 被 demand 的 expert 數」一致。**下一個該量的就是 verify 與 plain 的
cb/token 對比**，那決定第二個演算法是「縮 union」還是「host 簿記」。

## 成對重跑（2026-09-19 13:52，同一支 driver、背對背、同一機器狀態，加了記憶體守衛）

第一次嘗試在 free=61 MB 下量到 167 ms/token（換頁，不是引擎）；加了「free < 2.5 GB 就等」的守衛後：

| 臂 | token | ms/token | t/s | 啟動前 free |
|---|---|---|---|---|
| MTP off | 92 | **117.32** | 8.52 | 7208 MB |
| MTP on | 96 | **118.74** | 8.42 | 9440 MB |

**⇒ 乾淨的成對量測下 MTP 是「打平」（8.42 vs 8.52 t/s），不是損失。**
兩臂的 `content_head` 逐字相同（greedy）——品質側也一致。
上一輪我報的「MTP 7.6 vs 9.2 t/s」與那之後所有「draft forward = 40 ms」的數字，
**都是同一類污染**：這台機器 12.7 GB 模型 + 8 GiB 池 = 20.7 GB / 16 GB，
每一次啟動都會把前一次的量測基準移走。**背對背成對**是這台機器上唯一可比的形狀。

### ② 的真正診斷（13:33 那一輪的 per-step 資料）

| | MTP off（ntok=1） | MTP on（ntok=4） | 比值 |
|---|---|---|---|
| step total | 103.81 ms | 303.87 ms | **2.93×** |
| gpu_sum | 93.97 | 246.50 | 2.62× |
| wait | 70.92 | 253.37 | 3.57× |
| **cb** | **24.50** | **36.82** | 1.50× |
| cb **per token** | 24.50 | **15.60** | **0.64×（變好）** |

- 一個 4-wide verify 要 **2.93×** 一個 1-wide step，而只交 2.36 token ⇒ **打平**。
- **break-even accept ≈ 0.80**（今天 0.45–0.53）⇒ 在這個成本比下，accept 不到 0.8，MTP 就不可能贏。
- **cb/token 反而變好（0.64×）**⇒ 「縮 union」不是②的槓桿；drafter 也只佔 3.54%。
- ⇒ **② 應改成「提高 accept（提案品質）」**，不是「批次化 draft forward」也不要「縮 union」。
  提案端便宜（3.5% 的時間），所以這是**投入產出最好**的一條。

## ❌ ① 過不了位元一致性（實作前就擋下來）

`ggml_mul_mat_id(w, x, ids)` 對每個 expert 位置是**獨立**的（不在 k 上歸約），所以把 id 向量
拆成 H/C 兩段再 `concat`，gate/up 的輸出**可以**與單一次 MMID 逐位元相同。
**但歸約發生在 combine**：今天 `sum_rows(weights * experts)` 對 k 個位置**依序**相加；
拆成 H/C 之後變成「先 H 全部、再 C 全部」，**同樣的項目、不同的加法順序** ⇒ IEEE 下不是同一個數。
把 cold 位置填 0.0 也救不了：0.0 不改變部分和，但那個 expert 的真值被移到後面才加。

> **⇒ split-MMID 無法同時（a）位元一致與（b）把填充藏在 resident 子集的 matmul 後面。**
> 唯一能覆蓋填充又是位元一致的做法，是把**在同一層內本來就獨立**的工作搬到前面
> —— 即 qwen35moe 的 **shared expert**（`ffn_*_shexp`，不需要 pool slot）。
> 它的量級約該層 FFN 的 20–25%，也就是只能蓋掉 ~1/4 的 cb，不是全部。

**結果：在 M1/M2/M3 的約束下，「把 miss 移出臨界路徑」沒有位元一致且量級足夠的實作。**
只剩三條：(i) 更少的 bytes/更少的 distinct（量化，就是使用者原來那條）、(ii) 更大的池、
(iii) 預取 —— 而預取對 compulsory（本池的主導 miss）先天無效。

## 現況（誠實）

- 兩份演算法**已寫成設計**（本文件）；**都尚未**在引擎裡實作 —— ②已被自身的資料改寫形狀，
  ①被位元一致性擋下（上面那節就是擋下的理由，不是「還沒做」）。
- 機器已清乾淨（0 個 llama 行程）；driver 的 `server_log` 用 mtime 抓到的是 `latest` 符號連結，
  但**臂的身分改用 argv 驗證**（`--spec-type draft-mtp` 在不在 cmdline）——那是決定性的，不是推測。
- 本輪的 recon 已經定出落點、證明了①的算術（0.66 < 1.12 ms/層）與②的成本來源（T=1 vs T=2.4–3.5），
  並且排除了兩個**看似存在但不存在**的機制：`async_eval_per_layer` / `before_layer_cb`
  在這棵樹裡**零命中** —— 那是 Edge0/MLX 引擎的東西，不是這個 C++ fork。
- 機器狀態：沒有啟動任何 server；`scripts/run_server.sh` 的改動只有兩個 allowlist 區塊。
