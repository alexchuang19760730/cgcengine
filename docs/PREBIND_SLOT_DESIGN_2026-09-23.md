# 預指派 slot（speculative slot binding）的落地設計

**日期** 2026-09-23 · **性質** 零 GPU 靜態分析 ＋ 既有 log 重算 · **歸本線（線 A，引擎層）**
**上游**：`docs/STEP_SERIALIZATION_2026-09-23.md`（wait/gap 口徑更正）
**新工具**：`scripts/check/gap_attribution.py`（selftest **33/33**）
**資料**：`/tmp/verify_layers/*prod25_decode-spec`（6 arm、1311 個穩態 verify step）

---

> ## ⚠ 2026-09-23 更正 —— 本文件的定價模型有一個方向性錯誤
>
> 續篇：**`docs/PREBIND_TRADEOFF_2026-09-23.md`** ＋ 工具 `scripts/check/prebind_ev.py`（27/27）。
>
> 1. **§6 的「`0.87^10 ≈ 25%`」低估了門檻，但 §10 的收益被低估得更多。** 真正的門檻事件
>    是「整層 union（實測 U=15.5）零 miss」，指數是 15.5 不是 8 ⇒ 門檻**更嚴**。
> 2. **但 `f_clean` 不是 `q^U`，是 `r^U`，而 `r = p_res0 + (1-p_res0)*q`。**
>    基線已有 85% 的被選專家常駐，預指派只救那 15% 冷的。本文件把 q 直接當 r，
>    ⇒ **把收益低估了一個數量級**（q=0.87：本文件推得 ~+0.8%，實為 **+17.2%**）。
> 3. **§7 階段 0 的 gate（`all_cov_layers/n_routable >= 0.5`）作廢**，
>    改成量 **q 本身**、門檻 **q(m=24) >= 0.73**（對應 step −10%）。
> 4. **§6 的「必須預指派比 top-8 寬」是對的，而且比想的更便宜**：
>    `llama-context.cpp:5390-5391` 只取 draft 的 j=0 一行（8 ids），
>    draft 其實算好 3 行（24 ids）。改成取全部三行是樹上現成的一行改動，
>    q 上限從 51.6% 升到 91.1%，Δt/s 從 +5.4% 升到 **+19.5%**。
>
> §2 的定價（斜率 ≈ 1）、§4 的可行性（3.3× 餘量）、§5 的 F2 區分**仍然成立**。

---

## 0. 一句話

**GPU 空窗可以用預指派消掉，而且在 MTP 下收益最大** —— 因為實測 CPU 的每一毫秒是
**1.02 倍**全額轉成 GPU 空窗（r=0.981）。而可行性不是賭命中率，是**幾何必然**：
GPU 給的窗口（129.8 ms）是 CPU 需要的（39.3 ms）的 **3.3 倍**。

---

## 1. 現況的精確時序

`ggml-backend.cpp:2109-2122` `hook_seg()`，每一段跑一次：

```
submit(seg il)
  ├─ wait   = st1-st0   CPU 自旋忙等 GPU 跑完本段     ← 不是 GPU 空轉
  ├─ cb     = st2-st1   top-k hook（ensure_batch 佔 99%）
  └─ submit = 提交 seg il+1
```

GPU 側是另一組欄位（Metal 時間戳）：`union`＝忙、`gap`＝空。
閉合：`union + gap = 146.6` vs `total 149.3`（**−1.8%**）。

`llama-context.cpp:6740-6746` 的資料依賴是真的：layer il 的 top-k 在**段 il 末端**
的 argsort 才產生，段 il+1 開頭的 `mul_mat_id` 就要用 ⇒ CPU 必須站在兩段之間。

**但註解那句「the idle is unavoidable per layer」下得太重**：
unavoidable 的是**填池**，不是**CPU 必須站在關鍵路徑上**。

---

## 2. ★ 定價：CPU 成本 → GPU 空窗 的比率（本工具的主輸出）

`gap ~ (cb + submit)`，穩態 verify step（`segs=41`、`step>=100`）：

| ntok | n | **slope** | intercept | r | cb | gap |
|---|---|---|---|---|---|---|
| 2 | 799 | **1.02** | +6.9 ms | 0.975 | 20.6 | 38.6 |
| 3 | 77 | **1.17** | +2.4 ms | 0.974 | 10.4 | 26.1 |
| **4** | 435 | **1.01** | +8.0 ms | 0.980 | **39.3** | **59.0** |
| **全部** | 1311 | **1.02** | +7.2 ms | 0.981 | | |

**斜率 ≈ 1.00** ⇒ CPU 在段間多花的每一毫秒，GPU 就多空轉一毫秒。
⇒ 把 CPU 挪出關鍵路徑，**回報是全額的**，不是打折的。

---

## 3. ★★ ntok=4 斷崖：MTP 的代價有 42% 是純空窗

同一個 arm 內部比（排除「不同 arm 本來就有不同 launch 狀態」的混淆）：

| arm | ntok | total | **cb** | **gap** | union | gap/total |
|---|---|---|---|---|---|---|
| 165341 | 2 | 99.0 | 9.8 | 27.7 | 69.6 | 28% |
| 165341 | 3 | 118.3 | 9.1 | 24.1 | 90.4 | 20% |
| **165341** | **4** | **179.8** | **33.7** | **54.6** | 122.8 | **30%** |
| 165512 | 2 | 96.5 | 8.7 | 25.2 | 65.7 | 26% |
| 165512 | 3 | 130.5 | 9.4 | 24.0 | 101.0 | 18% |
| **165512** | **4** | **186.0** | **39.8** | **60.4** | 120.1 | **32%** |
| 165931 | 2 | 89.3 | 10.2 | 24.0 | 64.1 | 27% |
| 165931 | 3 | 133.4 | 16.8 | 31.9 | 97.6 | 24% |
| **165931** | **4** | **182.8** | **44.0** | **64.1** | 117.0 | **35%** |

**兩個事實：**

1. **ntok 2→3，`cb` 幾乎不動**（9.8→9.1、8.7→9.4）；**ntok 4 是斷崖**（×3.4-4.3）。
2. 從 ntok=2 到 ntok=4：`total +48.9 ms`，其中
   `union +28.7 ms`（真工作，該付） ＋ **`gap +20.4 ms`（純浪費）**
   ⇒ **MTP 邊際成本裡 42% 是空窗**。

這直接回答「MTP 多出來的 CPU 工作會不會增加 GPU 空窗」：**現在是全額增加**
（斜率 1.01）；**預指派正是讓它不再增加的那一味藥**。

> 附帶：這也是「k=2 比 k=3 快」的具體機制之一（k=3 ⇒ ntok=4 ⇒ cb 斷崖）。
> 但它**不能取代** `docs/K3_PAIR_CERT_V2_BENCH_2026-09-23.md` 的認證 —— 那個認證的是
> 端到端 t/s，這裡量的是 step 級計數器，兩件事。

---

## 4. 可行性是幾何必然，不是賭命中率

| 量 | 數值 |
|---|---|
| GPU 給的窗口（`wait`，ntok=4） | **129.8 ms** |
| CPU 需要做的（`cb`） | **39.3 ms** |
| **餘量** | **3.3×** |

⇒ **即使預測的 ids 全部不 resident、每一個都要真讀**，123 ms 也夠（每次 fill 0.695 ms，
40 層 × 8 個 = 320 個 × 0.695 = 222 ms 才會超 —— 那是極端上界，實測 union 只有 15-16 個/層）。

**這點很重要，因為它讓 F2 的死因不再構成障礙：**
F2（`cgc_layer_ahead`）的 null 理由是「預測目標 100% 已在池裡 ⇒ 無事可做」。
對「省 fill」來說那是死亡；對「把 CPU 挪出關鍵路徑」來說**無害甚至有利** ——
已 resident ⇒ 影子 ensure 更便宜；不 resident ⇒ 123 ms 夠填。

---

## 5. 為什麼預指派和 F2 不是同一件事（三層差異）

| | F2 / `cgc_layer_ahead` | 預指派（本設計） |
|---|---|---|
| 做了什麼 | `prefetch_slot`：只發 IO | 影子 ensure ＋ **快路徑寫 remap** |
| hook 是否仍在關鍵路徑 | **是** | **否**（挪進 GPU 窗口） |
| 消掉的是什麼 | 只有 fill | **fill ＋ 每層 barrier** |
| 觸發時機 | `il==1` 一次 queue 40 層 | 逐層、在各自的 GPU 窗口內 |

**第三列是關鍵。** F1 的模型 `cb = 0.46·L + 0.695·M`：`0.46 ms` 的每層 barrier
**只要該層有 ≥1 miss 就付**。F2 只發 IO，miss 數 M 降了但「這一層仍被視為有 miss 過」
⇒ barrier 照付。預指派讓該層在真 ids 到達時**已無 miss** ⇒ barrier 與 fill **一起消失**。

⇒ 預指派的理想收益是 **`cb → 校驗成本`**，遠大於 F2 的「省 fill」。

⚠ 也要記得 F2 的第二個死因：`§EN-319` 判定它**從未執行治療**（flag 在 env 裡、
`prefetch=0/0`）。所以 F2 的 null **沒有測到預指派要測的東西** —— 不能用它否決本設計。

---

## 6. ★ 預測源：MTP 自帶一個比「上一 token」更準的，而且已經在樹上

| 來源 | 準確率 | 位置 | 現況 |
|---|---|---|---|
| `prev_token_expert_ids` | ~87%（相鄰 token 復用） | `llama-context.cpp:5429` | 只用於 prefetch |
| **`draft_prefetch_ids`** | **92-98%**（draft accept） | `llama-context.cpp:5386-5396` | **只用於 prefetch** |

註解原文（`llama-context.cpp:5375`）：**「the draft ctx computes the top-8 expert ids for the
NEXT token one step ahead of the trunk verify ctx」** —— draft ctx 在 verify 之前就跑，
它算出來的正好是 verify 下一步要用的 ids。這比「猜上一 token 會重演」好得多。

**現有三個機制的共同缺陷**（註解自己寫了）：都在 **`il==1` 一次 queue 全部 40 層**
⇒ 「320 experts against the free slots that exist at that instant, so most of it is refused
before it can overlap with anything」。

⇒ 本設計的兩個改動：**換預測源**（draft）＋ **改成逐層在自己窗口內提前**。

⚠ 命中率的真實門檻比「87%」嚴格得多：要的是**整層 union 全覆蓋**。
若 union 大小 U=10 且 per-expert 復用 p=0.87，獨立假設下全中機率只有 `0.87^10 ≈ 25%`。
⇒ **必須預指派比 top-8 寬的候選集**（或用 draft 的高 p）。**這是階段 0 要先量的數字。**

---

## 7. 設計

### 階段 0：只加儀器，不改行為（零風險，本線可做，優先）

在 hook 裡用 `draft_prefetch_ids[il]`（fallback `prev_token_expert_ids[il]`）
**離線**計算命中率，不改變任何行為：

```
CGC-PREBIND: layer=%u union=%zu pred=%zu covered=%zu  (cov_pct)
```

三個必須分開記的數：
- `cov_pct`：真 union 被 pred 覆蓋的比例（**逐 expert**，不是逐層全/無）
- `all_cov_layers`：整層 100% 覆蓋的層數（這才是快路徑能走的比例）
- `pred_cold`：pred 裡有多少個當時不 resident

**門檻（事先寫死）**：若 `all_cov_layers / n_routable < 0.5` ⇒ 預指派不值得，
**停在階段 0**，改走 `pread_usec` 機制路線。理由：快路徑覆蓋不到一半的層，
`cb` 只降一半，而複雜度是全部的。

### 階段 1：影子 ensure ＋ 快路徑（核心，`src/`）

在 `hook_seg` 提交 seg il **之後、`wait` 之前**（`ggml-backend.cpp:2109`）：

```c
submit(seg il);
// ---- 新增：與 GPU 跑 seg il 並行 ----
shadow_ensure(il+1, pred_ids[il+1]);     // 填池；只佔 free slot；記住 slot
// ------------------------------------
while (cgc_done(...) < target) sched_yield();   // wait（CPU 現在有活幹，不是空等）
read_true_ids();                                 // pre 只要 20.8 µs
if (true_ids ⊆ pred && all_resident) {
    write_remap_from_shadow();      // 快路徑：純查表
} else {
    ensure_batch(...); write_remap(...);   // 慢路徑：既有行為，正確性不變
}
submit(seg il+1);
```

**安全不變量（必須遵守，這是正確性的全部依據）**：
1. 快路徑的條件是**真 ids 全部 resident** —— **不是**「預測對了就跳過」。
   預測只用來決定**提前填什麼**，不用來決定**相不信任結果**。
2. 註解的硬約束：*Waiting only on the main buffer fired the top-k hook while the argsort
   was still running → stale ids → garbage remap* ⇒ **必須等整段跑完才能讀 ids**。
   並行的只能是 IO／填池，**不能是讀 ids**。
3. 影子 ensure 只佔 free slot（`prefetch_slot` 不 evict，已如此）。
   ⚠ 但要**保留 reserve**：若 pred 把 free slot 佔滿，真 ids 會無槽可用 ⇒ 慢路徑更慢。

### 階段 2：預測源升級（可選）

`prev_token` → `draft_prefetch_ids`（92-98%）。資料已在樹上，只需接線。

### 階段 3（可選，先別做）：S1 讓 CPU 徹底不讀 ids

`CGC_SLOT_TABLE_GPU`（S1）已 bit-identical。若 remap 在 GPU 算，CPU 連 ids 都不用讀。
但會引入一個新的正確性問題：**GPU 可能讀到還沒填好的 slot**（靜默錯）。
⇒ 需要先解決「publish 與消費的同步」，**階段 1 的量測結果出來再決定**。

---

## 8. 必須新增的計數器（前車之鑑：F2 的 flag 在 env 裡但沒被呼叫過）

`§EN-319` 的教訓是「以為治療在跑，其實沒有」。所以：

```
CGC-PREBIND: calls=%zu fast=%zu slow=%zu pred=%zu cov=%zu
             cold_filled=%zu dropped=%zu cap_skip=%zu slow_us=%zu fast_us=%zu
```

- **`calls == 0` ⇒ 這一輪無效**，直接宣告，不要解讀其他數字（F2 的坑）。
- `fast / (fast+slow)` ＝ 快路徑覆蓋率，就是收益的倍數。
- `slow_us` 必須**不比 baseline 的 cb 差** ⇒ 否則「預測錯」的懲罰會吃掉收益。

---

## 9. 判定實驗（跑數屬【執行 Agent】，本線只交規格）

**主指標用 `gap`，不用 t/s。** 這是本設計最大的成本優勢：

- `gap` 是 **step 級計數器**，一支 log 就有 1300+ 個樣本，**不需要跨 launch 配對**；
- `t/s` 是 launch 級，單臂 sd 18%，要 8 對 ABBA（`K3_PAIR_CERT_V2` §5.1）。
⇒ **這個實驗比「k=2 vs k=3 認證」便宜一個數量級**。

| 判準（同時成立才算過） | 門檻 |
|---|---|
| 有效性 | `prebind_calls > 0` |
| **主指標** | `gap_B / gap_A ≤ 0.60` |
| 覆蓋率 | `fast/(fast+slow) ≥ 0.50` |
| 無副作用 | `slow_us_B ≤ cb_A × 1.10`（慢路徑不該變慢） |
| 正確性 | M1/M2/M3 各 9/9、`zero_mapped_selected=0` |

A/B 同一 build、同一 session、交錯；`--reps 3` 與 12.57 那次同形狀。

---

## 10. 預期收益與誠實邊界

若快路徑全覆蓋（`cb → ~2 ms` 查表）：

```
ntok=4:  gap = 1.02×(2 + 11) + 8.0 ≈ 21.3 ms   （現在 59.0）
         total = union 120.1 + gap 21.3 ≈ 141.4  （現在 183.5）  ⇒ −23%
```

若 emit/round 不變，t/s 約 **+30%**。

⚠ **兩個必須說清楚的不確定**：
1. **截距 +7.2 ms 消不掉**（它與 cb 無關，可能是命令緩衝區啟動偏斜）。
   它是否 ∝ 段數，**無法用既有 log 判定** —— 因為這批 log 只有 `segs ∈ {2, 41}` 兩種值，
   沒有中間點可做回歸。要判它得新增按段記錄的儀器，或做「改段數」的實驗。
2. **上界 +30% 與既有天花板分析衝突**（L3 誠實上界 ~+7% ⇒ 13.5）。
   衝突的來源很可能是**天花板的前提假設了 CPU 必須在關鍵路徑上** ——
   而本設計正是在拆掉那個假設。⇒ **這不是「天花板錯了」，是「天花板的適用範圍要重劃」**，
   在階段 1 有實測之前，不要把 +30% 當成承諾，也不要拿它去推翻 L3。

---

## 11. 未解（別當成已解決）

1. **`ntok=4` 斷崖的機制**：是 hook 次數變多（~4 → ~15 次/step），還是每次 hook 變貴？
   推算是前者（`cb / 2.27 ms-per-call`），但**沒有逐層計數器可以證明**。
   需要 `CGC-HOOKSPLIT` 按 ntok 分組輸出。
2. **draft 預測的實際命中率未量**（只有 92-98% 的 accept 率作為間接證據）⇒ 階段 0。
3. **union 大小與 ntok 的關係**只有兩個取樣點（ntok=1 → 8 ids，ntok=8 → ≥16 ids，
   且 `CGC-HOOK` 截斷在 16）⇒ 階段 0 要印真實 `uni_size`。
4. **每次 hook 的 `ensure_batch` 2.267 ms** 是全程平均（含 warmup），穩態值未分離。
