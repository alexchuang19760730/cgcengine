# `-np N` per-job overhead 完整方案

日期：2026-09-21 · 作者：線A (ace) · 權威文件（後續判決以此為準）
上游：`docs/PARALLEL_AND_WORKERS_AB_2026-09-21.md`（§EN-364）、`docs/M_WIDTH_CURVE_2026-09-21.md`（§EN-369）

---

## 0. 一句話

`-np 4` 慢 7 倍**不是頻寬、不是 miss、不是池大小**，是 **fill submit 端的 per-job 固定成本
（130–338 µs/job）乘上一個爆掉 35–83 倍的 job 數**。兩者都可修，而且第一層是 **0 行程式碼、只改 env**。

但先講清楚代價：**這條路最多給你「聚合 25 t/s」，且必須開 ~107–128 條併發串流，
代價是單條延遲 ~4 s/token。** 它不是單流 25 的替代方案。

---

## 1. 已確定的機制（全部有數字，非推測）

來源：np1 = `Backup/cgc_logs/llama_server_20260921_013016.log`，np4 = `..._013118.log`
（同一 build `efeade7e2`、同一 driver、只差 `-np`）。

| 量 | np1 | np4 | 倍率 |
|---|---|---|---|
| `file_reads`（jobs） | 62,829 | **2,218,197** | **35.3×** |
| bytes 讀取 | 8.36 GB | 0.57 GB | 0.068× |
| **bytes/job** | **133 KiB** | **257 B** | 0.0019× |
| `fill_batch_usec`（ensure_batch 總時） | 12.4 s | **748.9 s** | 60× |
| `pool_fill_wait_us`（等 worker） | 4.19 s | **0.18 s** | 0.043× |
| **submit 端** = fill_batch − wait | **8.2 s** | **748.7 s** | 91× |
| **µs/job（submit）** | **130** | **338** | 2.6× |
| jobs / expert-request | 2.43 | **201.6** | **83×** |
| 池命中率 | 70.9% | 94.1% | — |

### 1.1 三條由此確定的結論

1. **時間在 submit，不在 wait。** np4 的 `pool_fill_wait_us` 只有 0.18 s —— worker 一被叫醒就做完。
   748.9 s 全花在**建立 job 的那條執行緒上**（`fill_segments_pool` 的 sort + job build + `F_RDADVISE`）。
   ⇒ **加大 worker 數完全無效**（與 §EN-364 的 W32 −8% / W4 無差一致，機制現在對上了）。
2. **job 數爆掉是主因，不是 per-job 單價。** per-job 從 130 → 338 µs（2.6×），job 數 35×
   ⇒ 乘起來 91×。**先砍 job 數，報酬是砍單價的 13 倍。**
3. **257 B/job 說明 merge-read 完全沒 coalesce。** `llama-expert-cache.cpp:3529` 的合併條件是
   `segs[j].file_offset + segs[j].bytes == segs[j+1].file_offset`（檔案內嚴格相鄰）。
   np1 靠「同 tensor 內相鄰 expert id」合到 ~2 jobs/layer-call；np4 的 union 是 4 條獨立
   routing 的聯集，expert id 分散 ⇒ **每一個 segment 自己一個 job（70.4 jobs/layer-call）**。

### 1.2 per-job 單價的成分（**S0-2 已裁決：首犯被推翻**）

`llama-expert-cache.cpp:3593` 的註解自己寫著：同步 `F_RDADVISE` 是 **780 µs/call，
佔 submit 的 99%**。而 **`ADVISE_ASYNC` 預設 OFF**（`:3928`，`advise_on=false`）
⇒ **每一個 job 都在關鍵路徑上打一次 fcntl**。

> ⚠️ **2026-09-21 11:5x S0-2 實測更正（原文保留，結論反轉）。
> 「同步 rdadvise 是 np4 per-job 首犯」這個推測**不成立**。**
> - `NO_RDADVISE=1`（**完全關掉**）的 `usec/job` = **1.028×** —— 若它真佔 submit 99%，關掉該看到崩塌；
> - 帳也算不攏：np4 的 2,218,197 jobs × 780 µs = **1,730 s**，而實測 `fill_batch_usec` 只有 **749 s**；
> - `ADVISE_ASYNC=1` 的配對效應只有 **−7.5%**（4 個局部估計 0.773–1.144，跨過 1.0）。
> ⇒ 780 µs 是別條線（M-F5）在**別的 regime** 的數字，**不能外推到 `-np 4`**。
> 自洽的解讀：同步 rdadvise 的成本 ≈ 它帶來的 kernel read-ahead 收益（關掉是平的），async 化才淨賺一點。
> ⇒ **Stage 1 不 ship**。詳見 `docs/S02_PERJOB_AB_2026-09-21.md` §5.3。

其餘成分（待量，現在才是候選首犯）：`new struct iovec[cnt]` + `new int*[cnt]` 兩個 heap alloc、
`std::sort`、mutex + `pool_cv.notify_all()` + `pool_done_cv` 喚醒。

---

## 2. 先算天花板：這條路值不值得走

### 2.1 要 25 t/s 需要多少併發（M）

`docs/M_WIDTH_CURVE_2026-09-21.md` 的實測：**M（一步內的位置數）才是成本驅動項。**

| M | 1 | 8 | 16 | 64 | 128 | 512 |
|---|---|---|---|---|---|---|
| ms/位置 | 92 | 75.4 | 268 | 66.5 | **40.7** | 8.51 |
| 聚合 t/s | 10.8 | 13.3 | 3.7 | 15.0 | **24.6** | 117.5 |

⇒ **25 t/s = 40 ms/位置 ⇒ M ≈ 107**（實測 M=128 → 24.57 t/s）。
`-np N` 時 M = N ⇒ **需要 N ≈ 107–128。**

### 2.2 兩個 regime 與「必須切到 slab」

- **pool 路徑**（M ≤ `decode_width`）：`step ≈ 176 ms + 53.4 ms×M`，但 `decode_width` 被
  `floor(min_usable_slots / top_k) = floor(143/8) = 17` 夾住 ⇒ **pool 路徑 M ≤ 17**，
  漸近線只有 **18.7 t/s（到不了 25）**。
- **slab 路徑**（M ≥ 16）：`step ≈ 4.28 s`，**與 M 幾乎無關**，因為每步填全部 256 個專家
  （11.9 GB）。⇒ 聚合 t/s = M / 4.28 s 線性成長 ⇒ **M=128 → ~30 t/s**。

⇒ **必須讓 N≥107 的步走 slab 路徑**，否則卡在 18.7。

### 2.3 旋鈕都已存在（不用改程式碼就能切）

| 變數 | 預設 | 位置 | 作用 |
|---|---|---|---|
| `CGC_POOL_MAX_TOKENS` | 8，clamp [2,64] | `llama-graph.h:26` | pool 路徑寬度上限 |
| `CGC_PREFILL_THRESHOLD` | 512，clamp [2,2²⁰] | `llama-cgc-phase.h:87` | `n_tokens > thr-1` ⇒ 走 PREFILL/slab |
| `LLAMA_EXPERT_CACHE_ADVISE_ASYNC` | **OFF** | `llama-expert-cache.cpp:3928` | 把 rdadvise 搬到 advisor thread。**S0-2 配對實測 −7.5%**，不達 Stage 1 判準（≥20%）⇒ **保留但未 ship** |
| `LLAMA_EXPERT_CACHE_NO_RDADVISE` | OFF | `:3489` | 完全關掉 rdadvise（A/B 用）。**S0-2 實測 1.028×＝無效果** ⇒ 推翻「佔 submit 99%」的外推 |
| `LLAMA_EXPERT_CACHE_NO_MERGE` | OFF | `:3495`（本文件原文誤記 `:3487`） | 退回 per-segment job（**反向對照**）⚠️ **不在 allowlist ⇒ 走 `run_server.sh` 會被靜默丟棄，必須用 `perjob_s02.py` 的 bypass launcher** |
| `LLAMA_EXPERT_CACHE_WORKERS` | 8 | — | 已證實無效，別再動 |

`decode_width = min(floor(143/8)=17, thr-1, cap)`。
⇒ **`CGC_POOL_MAX_TOKENS=17`** 把 pool 路徑拉滿 17；
⇒ **`CGC_PREFILL_THRESHOLD=18`** 讓 N ≥ 18 的步改走 slab。

### 2.4 KV 預算 —— **S0-1 已實測，門檻通過（原估算高估 16×）**

> ⚠️ **本節在 2026-09-21 10:4x 被實測推翻並重寫。** 原文假設 KV = 168 KB/token（無出處），
> 實測 **10.5 KiB/token** ⇒ 高估 **16 倍**。結論方向不變（KV 不是瓶頸），但**餘裕大得多**。

**方法**：`llama-server --load-mode none`（不載權重，RSS 基準乾淨），固定 `-np 1`、
`-expert-cache 8 GiB`、`--cache-type-k/v q8_0`，只改 `-c`，讀行程 RSS。
（這 fork **不印** `KV self size` / `cells` 行，既有 log 全無 ⇒ 只能用 RSS 差法。）

| `-c` | RSS (MiB) | 區間 Δtoken | ΔRSS | 推算 KV/token |
|---|---|---|---|---|
| 4,096 | 7,578 | – | – | – |
| 8,192 | 7,620 | 4,096 | 51 MiB | 12.8 KiB |
| 32,768 | 7,884 | 28,672 | 315 MiB | 11.5 KiB |
| **131,072** | **8,879** | **126,976** | **1,301 MiB** | **10.5 KiB** ← 最大區間，最可信 |

⇒ **權威值 KV/token = 10.5 KiB**（小區間偏高是 mmap 分頁快取的雜訊，取最大區間）。
（`run_server.sh:967` 的註解「KV@8192 約 160 MiB」⇒ 20 KiB/token，比實測**悲觀 2×**，同向。）

**`n_ctx_slot` 實測**：`-np 1 -c 4096` ⇒ 4096；`-np 4 -c 4096` ⇒ **1024**；
`-np 128 -c 32768` ⇒ **256**；`-np 128 -c 262144` ⇒ **2048**。
⇒ 確認 `kv_unified=false` 下 **總 cells = `-c`，KV 總量只跟總 token 數有關、與 N 無關**。

**可行性（用 10.5 KiB/token 重算）**

| N | ctx_slot | 總 token | KV 總量 | 實測 |
|---|---|---|---|---|
| 4（今天） | 2048 | 8,192 | 84 MiB | ✓ |
| 128 | 256 | 32,768 | 336 MiB | **✓ 實測開得起來**（RSS 8,466 MiB） |
| **128** | **2048** | **262,144** | **2.62 GiB** | **✓ 實測開得起來**（RSS 10,150 MiB，free 9%） |
| 256 | 2048 | 524,288 | 5.24 GiB | ✗ 推估（＋pool 8 GiB 會爆） |

⇒ **N=128 不必委屈到 ctx_slot=256；2048/槽也裝得下。**
⇒ **N_max ≈ 128–192（受 KV＋pool 8 GiB 的總量限制，不是受 N 本身限制）。**
⇒ per-slot 非 KV 固定成本 ~4.6 MiB/槽（`np1` vs `np128` 同 `-c 32768`：8,466−7,884=582 MiB / 127）
  ⚠️ 雜訊大（`np32` 同條件讀到 9,779，與單調性矛盾）⇒ **只當上界，別當精確值**。

### 2.5 延遲代價（必須先接受）

slab 路徑 `step ≈ 4.28 s` 且與 M 無關 ⇒ N=128 時**每條串流 4.28 s 才吐 1 個 token**。

> **25 t/s 聚合 = 128 條併發 × 0.195 t/s/條，單條延遲 4.3 s/token。**
> 這是「批次吞吐」產品，不是互動 decode。若你的 25 是指「一個使用者每秒看到 25 個字」，
> 這條路**不適用**，單流 25 仍按 §EN-366/§EN-369 下架、目標 14–16。

### 2.6 同一個病，也解釋了 slab 的 2.8 GB/s

§EN-369 量到 pool 12.4 GB/s vs slab 2.8 GB/s（4.4×）。現在機制對上了：
slab 每步做 **256 專家 × 41 層 = 10,496 次讀**，11.9 GB / 4.28 s ⇒ **408 µs/讀（1.13 MB）**
⇒ **它也是 per-job overhead bound，不是另一個現象。**
⇒ **修 per-job overhead 會同時改善 prefill**（M=128 可能從 24.6 → 60+ t/s）。

---

## 3. 分層方案

> 紀律：每層有**預註冊判準**、有**反向對照**、有**停止條件**。
> 判準先寫死再跑，跑完只看判準，不改判準。

### Stage 0 — 定位與預算（~40 min，不寫產品碼）

**S0-1 量 KV/token 真值。** ✅ **已完成 2026-09-21 10:4x，結果：10.5 KiB/token。**
- **判準**：KV/token > 250 KB ⇒ N_max < 60 ⇒ 25 聚合不可達，全案終止。
- **實測 10.5 KiB ≪ 250 KB ⇒ 門檻通過，全案繼續。**（詳細數據見 §2.4，原估算高估 16×）
- 附帶實測：**`-np 128` 開得起來**（`-c 262144` ⇒ `n_ctx_slot = 2048`，RSS 10.15 GiB）。
  ⇒ 天花板不是「N 開不起來」，是 §2.5 的延遲與 §2.3 的 per-job overhead。
- ⚠️ 方法論：這 fork 不印 KV size 行，只能用 RSS 差法；**區間要拉大**（小區間被 mmap
  分頁快取污染，4,096→8,192 那一段高估 22%）。

**S0-2 三個便宜 A/B。** ✅ **已完成 2026-09-21 11:5x，兩輪共 10 臂。詳見
`docs/S02_PERJOB_AB_2026-09-21.md`；結論摘要在下表。**

工具 `Backup/phase_decomp/perjob_s02.py`。⚠️ **這三臂不能都走 `run_server.sh`**：
`NO_RDADVISE`/`ADVISE_ASYNC` 在 allowlist（`:1552`/`:1563`），但 **`NO_MERGE` 不在**（C++ 有，
`:3495`）⇒ 走 launcher 會被靜默丟棄。改用 **bypass launcher**
（`CGC_DUMP_ENV=1` 拿到 `CGCENV BIN`/`ENV`/`ARG`，照著它自己 Popen），不動共享腳本。
同構性已驗證：bypass 的 `usec/job` 326.7 vs np4 原始 log 的 337.6（差 3.4%）。

| # | 判準（預先寫死） | 實測 | 裁決 |
|---|---|---|---|
| (a) `NO_MERGE` | job 數變化 **< 20%** ⇒ merge 完全失效 CONFIRMED | **+0.018%**（316,083 → 316,140） | ✅ **CONFIRMED**（計數器，三次重跑完全相同） |
| (b) rdadvise | wall/`usec/job` 降 **≥ 30%** ⇒ per-job submit 是主成分 | `ADVISE_ASYNC` 配對 **−7.5%**；`NO_RDADVISE` 1.028× | ❌ **未達 Stage 1 判準（≥20%）** |

- **(a) 是本輪最強的單一結論**：關掉 merge 之後 318 萬個 job 一個都沒少 ⇒
  `:3528` 的嚴格相鄰條件在 np4 的 union 下**一次都沒命中**。**Stage 2-a 對症。**
- **(b) 寫成「不可判」而非「無效」**：時間讀數在這個 cell 的單臂漂移是 ±11～31%
  （Round 1 兩個相同 base：415.4 → 544.0，+31%；Round 2 三個 base CV=11.5% 且**非單調**），
  而效應只有 −7.5% ⇒ **只有配對設計有發言權，配對後仍跨過 1.0**。
  ⇒ 原來的那個 −17%（單臂）**是漂移，不是效應**。

**⇒ Stage 1 不 ship（保留為已知資訊）。進 Stage 2，且 Stage 2-a 現在有 (a) 的直接支持。**
**Stage 2 的阻塞點**：要寫 C++ + 重建，而建置產物含另一條 session 的未提交熱路徑改動，
`cmake --build` 會蓋掉別人正在 map 的 dylib ⇒ **等乾淨窗口**（無 listener、`pgrep -fl llama-server` 空）。

**S0-3 畫出 job 數 vs N 的曲線**：`-np 1,2,4,8`，讀 `file_reads` 與 `fill_batch_usec`。
- **判準**：job/layer-call 隨 N **線性**成長 ⇒ 是 union 分散問題；
  若**超線性** ⇒ 有 per-sequence 重複 ensure，先修那個（報酬更大）。

**產物**：`Backup/phase_decomp/perjob_diag/`（JSON + log）、一份 S0 結論。
**停止條件**：S0-1 不過 ⇒ 全案終止。

---

### Stage 1 — 0 行程式碼：把 per-job 成本搬離關鍵路徑（~20 min）

動作：只加 env，跑配對 A/B。

```sh
# 交付形狀、-np 8、配對 3 對
LLAMA_EXPERT_CACHE_ADVISE_ASYNC=1 CGC_POOL_MAX_TOKENS=17 ./scripts/run_server.sh
```

- **判準**：聚合 t/s 提升 **≥ 20%** ⇒ 進 Stage 2；
  **< 10%** ⇒ per-job 單價不是主成分，**跳到 Stage 2（砍 job 數）**，Stage 1 只當已知資訊。

> ✅ **S0-2 已替 Stage 1 回答了（2026-09-21 11:5x），量到的是 −7.5%（< 10% 那一側）。**
> ⇒ **按判準：Stage 1 不 ship，直接跳 Stage 2。** `ADVISE_ASYNC` 留下來當已知資訊——
> 它是免費 env，將來若在**交付形狀**（`prod_profile.py`、np1、llama-bench、兩軸）上配對重測且有效，
> 那就是白拿；但**本輪的 np4 數字不能替那個 cell 說話**（見 `docs/S02_PERJOB_AB_2026-09-21.md` §6）。
> 量測細節裡有一條必須帶走：`usec/job` 這種**時間比值的單臂漂移是 ±11～31%**，
> 而 `file_reads` 這類**計數器三次重跑完全相同** ⇒ 能數的東西先數，比值要配對。

---

### Stage 2 — 砍 job 數（主要槓桿，~半天～一天）

目標：**70.4 → ~5 jobs/layer-call**（35× → 與 np1 同級 ⇒ fill submit 749 s → ~21 s）。

三種做法，**由便宜到貴**，各自無損：

**S2-a（最便宜，~1h）放寬合併條件：允許跨過小 gap。**
把 `llama-expert-cache.cpp:3529` 的嚴格相鄰改成
`next_off - (cur_off + cur_bytes) <= G`（建議 `G = 8 × per_expert_bytes`，可由 env `CGC_MERGE_GAP_EXPERTS` 調）。
多讀的中間專家**寫進一個 scratch dst**（不進池），位元層面對池內容無影響。
- 判準：`file_reads` 降 **≥ 5×** 且 `t/s` 不降 ⇒ 採用。
- 🔴 **2026-09-21 已裁決（在 decode 上）：不要做。** 見 `docs/STAGE2A_DECODE_VALUE_2026-09-21.md`。
  上面建議的 `G = 8 × per_expert_bytes` 實測下來：IO 物理**支持**（零 junk 快 35%、gap=8 仍快 21%，
  交叉點 gap≈16）⇒ 本條的目的是對的；但**機會不存在** —— np1 decode 上
  `miss/(layer,step)` 只有 **1.82**（HC hit rate 71.2%），G=8 只能砍 **1.05%** 的 job、
  G=16 砍 2.09%（已打平）、G=32 砍 4.18%（物理上慢 2×）。換算 decode 增益 **≤1%**，
  低於單臂噪音 ±27% ⇒ **證明不出來**。
  ⚠️ 本條對 **np4/prefill 仍可能成立**（那裡 union 寬、相鄰率高），但要在那些 cell 上重測，
  不能用 np1 的數字替它說話。

**S2-b（中，~半天）整層一次 span 讀。**
對每一層的每一種 kind（gate/up/down），讀一次 `[min_demanded_off, max_demanded_off]` 的連續區間
到一塊 scratch，再把需要的專家 copy 進各自的池 slot。
⇒ **每層每步 3 個 job**，與 union 寬度無關。
- 代價：多讀 span 內已 resident 的專家；當 union 分散時 span ≈ 全層 ⇒ 退化成 slab。
- 判準：job 數降到 **≤ 6/layer-call** 且額外 bytes **< 2×** ⇒ 採用。

**S2-c（貴，~1–2 天）當 M ≥ 閾值時直接改用 slab 全讀。**
既然 N ≥ 107 時 union 必然逼近 256 個專家，**pool 在此 regime 本來就沒有意義**
（§EN-368 已證：寬 union ⇒ capacity thrash，命中率 96.5% → 57.5%）。
⇒ 直接走 slab，但**修掉 slab 自己的 10,496 次讀**：改成每層每 kind 一次連續讀（**123 次讀**）。
- 這同時就是 §EN-369 那個「4.4× 頻寬缺口」的修法。
- 判準：slab step time 從 4.28 s 降 **≥ 2×** ⇒ M=128 聚合從 24.6 → **50+ t/s**。

**共同要求**：每一版都要跑 `G2`（逐位元相同）與 `G5` 複查；用
`scripts/check/paired_ab.py` 同 build 配對，不跨場比。

---

### Stage 3 — 把 M 拉到 107（~20 min，純 env）

```sh
CGC_PREFILL_THRESHOLD=18 CGC_POOL_MAX_TOKENS=17 CGC_PREFILL_STREAM=1 \
CGC_GATHER_SLAB_CAP=256 CGC_SERVER_CONCURRENCY=128 ./scripts/run_server.sh
```

⚠️ `CGC_PREFILL_STREAM=1` **必須**配 `CGC_GATHER_SLAB_CAP=256`，否則
`GGML_ASSERT(n_tokens_all <= cparams.n_batch)`（S0 已踩過一次）。

- 量 `-np 1,8,32,64,128` 的**聚合 t/s** 與 **per-stream latency**。
- **判準**：聚合 t/s 隨 N 線性成長 ⇒ Stage 2 的修法正在生效；
  若**持平** ⇒ 瓶頸已不在 fill，回到 M 曲線重測（可能是 Metal/GPU 側）。

---

### Stage 4 — 驗證與封版（~1h）

- 用 `m_width_curve.py` 的口徑（llama-bench、MTP off、reps 3）重測一次 M 曲線，確認
  slab 段（M ≥ 16）的 step time 是否如 Stage 2-c 預測下降。
- 用 `parallel_decode_ab.py`（自寫 HTTP driver，`-np` 是 server-only）量 N=128 的聚合。
- **封版判準**：聚合 **≥ 25 t/s** 且 per-stream latency 在可接受範圍 ⇒ 目標改寫為
  「聚合 25 t/s @ N≥107」；否則降級記錄。

---

## 4. 不做清單（已封，別浪費時間）

| 動作 | 狀態 |
|---|---|
| 降 k / 加大 `--spec-draft-n-max` | 封（§EN-368：`best_k=0`，12 臂掃描） |
| 加大 expert pool | 封（§EN-368：需 13.6 GB，16 GB 硬牆） |
| 加 `CGC_SERVER_WORKERS` | 封（§EN-364：W32 −8%、W4 無差；本文件 §1.1 給出機制） |
| dispatch 融合（G4） | 封（上界 11.5% < 門檻 12.8%） |
| 找 MTP verify 的缺陷 | 封（§EN-369：verify 317.4 ms < plain batch 457 ms，**沒有缺陷**） |
| 壓 bpw | 使用者已排除（xxs = 精度下限） |

---

## 5. 風險

1. **Stage 2 動的是 fill 路徑 ⇒ 必須過 G2（逐位元相同）。** S2-a/b 多讀 bytes 只能進 scratch，
   不得污染池 slot。
2. **Stage 3 的 N=128 需要 `-c ≥ 32768` 且 `ctx_slot=256`** ⇒ 若實際 workload 需要長上下文，
   這條路直接不可用。
3. **並行 session 安全**：`-np` 量測只能走 HTTP，會起 server。動手前後各跑
   `lsof -nP -iTCP:8080 -sTCP:LISTEN` 與 `pgrep -fl llama-server`；
   背景量測**不要接管道**（§EN-364 教訓），用 `nohup ... > log`。
4. **熱與噪音**：單臂噪音 ~27%；所有結論**必須配對**（AB/BA 交替），不跨場比。

---

## 6. 決策點（何時該喊停）

| 檢查點 | 喊停條件 |
|---|---|
| S0-1 後 | KV/token > 250 KB ⇒ N_max < 60 ⇒ 25 聚合不可達 — **實測 10.5 KiB，✅ 通過** |
| S0-2 後 | 三個 A/B 全部 < 10% ⇒ per-job overhead 不是主成分，假說錯，回到 M 曲線找別的 |
| Stage 2 後 | job 數降不到 5× ⇒ 改用 S2-c；S2-c 也失敗 ⇒ 此軸結案 |
| Stage 3 後 | 聚合 t/s 不隨 N 成長 ⇒ 真瓶頸在 GPU/Metal 側，不在 fill（開新洞） |
