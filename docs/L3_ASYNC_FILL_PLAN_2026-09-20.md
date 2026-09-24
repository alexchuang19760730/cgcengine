# L3 完整方案 —— 讓 decode 的 demand fill 從「同步等」變成「先發出、延後等」

日期：2026-09-20　狀態：**方案（未實作）**　載體：Nail IQ3_XXS-denseIQ4X、MTP on k=3、pool 8 GiB、`prefill250` 形狀、temp 0.4
引擎：`libllama` md5 `e06b60097a2ee21a`（HEAD `656cfdf1b` ＋ 未提交的 ring/launcher 改動）

---

## 0. 先回答「你已經實作了嗎」：**沒有**

| 項目 | 狀態 |
|---|---|
| L3 的**載體**（`prefetch_slot` ＋ `drain_layer` ＋ `CGC_LAYER_AHEAD_PREFETCH`） | **在**（2026-09-19 的「L3 Option A double-buffer」） |
| L3 的**機制**（demand union 的 fill 改成 issue-and-return、等待移到 consumer） | **不在** —— 今天 `ensure_batch` 仍是 `submit → wait(outstanding==0) → 回傳` |
| 「+40~50% → 19 t/s」這個數字 | **已作廢**，見 `docs/L3_ADOPT_DECISION_2026-09-20.md`：上界改為 **+37%、硬天花板 18.59 t/s @mean_len 3.375** |
| 對應的量測 | **零**。`fill_wait_us`（L3 預註冊的輸出）在 decode 步上一次都沒出現過 |

也就是說：磁碟上那個「L3」是**用預測器餵**的版本，而預測那一半已被 L1 關門（K=64 仍 99.19% resident、池子對工作集 17.9× 超額）；
**真正不需要預測器的那個版本（本文件）還沒有寫**。

---

## 1. 為什麼現在不是「非同步」——程式事實

同步鏈（`src/llama.cpp/src/llama-expert-cache.cpp`）：

```
llama_expert_cache_ensure_batch()          :1036   ← decode hook 的 99.7% 都在這裡（F1）
  ├─ pass1/pass2 兩趟放置（鎖內）           :1140+  → assign 3.1 µs
  ├─ slot_table 發布 ＋ WIN_PIN             ~:1186  → publish 0.4 µs
  └─ fill_segments_pool()
       ├─ 建 job / pread 入隊                        → submit 780 µs（其中 ~99% 是 F_RDADVISE）
       └─ pool_done_cv.wait(pool_outstanding==0)  :3572  ← ★ 阻塞點：wait 1521 µs
llama_expert_cache_drain_layer()           :1471   ← 丟棄該層「已排隊未開始」的 prefetch，並等 slot_loading
```

**掛點是 build time，不是 eval time。** hook 由 `graph_get_cb`（`:7153`）在建圖時逐層呼叫，所以今天的效果是：
**在整張圖還沒開始算之前，這一層的 bytes 就已經落地**。等待因此完全沒有東西可以遮。
`ggml_backend_sched_set_eval_callback(sched, expert_cache_eval_cb, this)`（`:1829`）是既有的**逐節點 eval 勾點**
（`:7278` 已用它按節點名 `ffn_moe_ids_leaf` 分支）—— 這就是把等待搬到「consumer 真正要讀之前」的既有載體。

**F2 的教訓必須先講**：上一輪 B 臂 env 有 `CGC_LAYER_AHEAD_PREFETCH=1`，但 `prefetch=0/0`、零 `PFDBG` ⇒ **治療從未執行**，
那是一個 treatment-application null，不是機制結論。本方案的每一步都以「**先證明劑量非零**」為前置。

---

## 2. 目標與預註冊判準（不改）

**目標**：把 `cb`（交付 work-row 74.18 ms）中屬於「等 bytes」的部分，從臨界路徑移出，讓它落在同一層／前一層的 GPU 忙碌底下。

**預註冊（跑之前就寫死，跑完只讀不算）**：

> ⚠️ **2026-09-20 M0 修正**：主輸出由 `fill_wait_us` 改成 **`pool_wait`**。
> M0 實測（`docs/L3_M0_DOSE_RESULT_2026-09-20.md`）：交付 regime 每一步的 `fill_wait_us` **恆為 0.000**，
> 因為 demand 批次的等待從來不在那個計數器裡；真正可動的項是新的 `pool_wait`（verify ntok=4 中位 **32.4 ms/步**）。
> 拿一個恆 0 的東西當判準，結果會被讀成「重疊沒用」。

- **PASS**：**`pool_wait` 下降** **且** step `total` 下降，
  而且 `CGC_GPU_TIMING` 的 **GPU idle（gap）沒有等量上升**（省下來的時間以 GPU idle 重現就不算省）。
- **NULL**：`pool_wait` 降但 `total` 持平（因為 `wait` 上升）—— 這是**最可能的失敗模式**：
  78 支 log 的組內 OLS 斜率中位 **1.455**（97% ≥ 1.0）說明 cb 滿額串行，時間只會搬到 `wait` 裡。
- **FAIL**：數值不再逐位元相同（free-slot 紀律被破壞 ⇒ GPU 讀到未落地的 slot）。

**驗收同時要求（不可事後補）**：M1/M2/M3 oracle **9/9/9** vs v6 參考，同一支 binary、同一輪。

---

## 3. 機制設計

### 3.1 兩個新入口（並存，預設走舊路）

```
llama_expert_cache_fill_begin(cache, layer, experts, n)   // 放置 + 入隊 + 回傳，不等待
llama_expert_cache_fill_await(cache, layer)               // 等到該層 in-flight 的 demand fill 全部落地 + 發布
```

`ensure_batch` 拆成這兩半；`CGC_FILL_ASYNC=0`（預設）時兩者背靠背呼叫，**行為與今天逐位元相同**（等價於原地 inline）。

### 3.2 等待掛在哪裡（關鍵決策）

| 候選 | 遮得到什麼 | 判定 |
|---|---|---|
| (a) 圖建完、`graph_compute` 之前 | 什麼都遮不到（等於今天） | ✗ |
| (b) **該層 segment 的第一個 consumer 節點**（`ggml_backend_sched` 的 eval callback，按節點名前綴 `blk.<il>.` 判定） | 該層 attention／前一層 FFN tail 已經提交給 GPU ⇒ 真實重疊 | ✅ **採用** |
| (c) `llama_decode` 尾端的 synchronize | 太粗，且把 torn read 風險換成整步同步 | ✗ |

**(b) 的可靠性要求**：每一層 `fill_begin` 之後，**必須**在該層 segment 內存在一個已註冊的 await；
若某層發出後沒有 await 命中 ⇒ **fail-closed**（stderr FATAL ＋ 該步拒絕走 async 路徑，回退同步）。缺 await 的代價是 GPU 讀到未落地的 slot，這是數值災難而不是效能問題。

### 3.3 四條 bit-identity 不變式（合併條件，不是事後檢查）

1. **同槽同序**：放置（assign）仍在發出前完成、`slot_table` 的發布順序不變 ⇒ 歸約順序、fp32 累加順序不變。
   （另一條獨立線已把「MoE 歸約按 expert id 排序」做成與槽位無關，兩者互為保險。）
2. **free-slot 紀律**：in-flight 的 demand fill 對 `pick_slot` / `prefetch_slot` 一律**不可淘汰** —— 沿用既有的
   `slot_owner[s]=e` ＋ `slot_queued/loading[s]=1` 語義。`CGC_NO_PREFETCH` 存在的原因（背景 fill 覆寫 GPU 正在 verify 的 slot）在這裡由「等待前不釋放」明確排除。
3. **drain 不改語義**：`drain_layer` 只丟「已排隊未開始」的 *prefetch*，並等 `slot_loading`；demand 的 in-flight 批次不被它取消。
4. **驗證器跟著搬**：`cgc_exact_cache_verify_post_fill`（byte-identity 檢查）與 `LLAMA_EXPERT_CACHE_BATCH_INVARIANT`
   的呼叫點**一起搬到 await 之後**，否則「檢查」會跑在 bytes 落地之前 ⇒ 假紅燈或假綠燈。

### 3.4 缺的儀器（**M0 已補，見 M0 報告**）

- `fill_wait_us` 在 decode row 裡不是「不存在」，而是**恆為 0**（demand 的等待不在它裡面）⇒ M0 補上 `pool_wait` / `pool_submit`。
- **layer-ahead 觸發計數器**（F2R §7 點名）：M0 已給 `la_entered / pred_missing / pred_empty / pred_ok / issued`；
  臂 3 讀到 `pred_ok=150, issued=0` ⇒ 預測有、非空、但一顆都不冷（L1 的活體確認）。
- M1 仍要新增 `fill_await` 的**每層註冊覆蓋率**（`fill_await_missed` 已就位，恆 0 且標示 `armed=0`）；
  注意 M0 已量到：走批次池路徑的層只有 ~2.7 層/步（ntok=4），不是 41 層。
- 新 env 一律 `${VAR:-default}`（`CGC_EVICTED_RING=0` 曾是**字面值**而靜默蓋掉覆寫）——
  M0 因為沿用既有 `CGC_HOOK_SPLIT`，**本輪沒有動 allowlist**。

---

## 4. 里程碑（每項都含「bit-identical」與驗收）

| # | 工作 | 產物 | 驗收（可否證） | 工時 |
|---|---|---|---|---|
| **M0** | 儀器 ＋ 預註冊 | **✅ 完成 2026-09-20** —— `docs/L3_M0_DOSE_RESULT_2026-09-20.md`；新計數器與 `CGC-L3DOSE` 行進 dylib（`strings` 見證）；`scripts/check/l3_dose.py`（11/11 自測）；三臂讀數 ＋ **閘門 M1/M2/M3 9/9/9**（binary `316393918ed651e4`） | ① dose 可讀（`pool_wait` 中位 32.4 ms @ntok=4，`fill_wait` 恆 0） ② `await_missed=0`（且已明示 `armed=0` ＝未檢查） ③ 數值不動＝**閘門讀數**，不是讀碼結論 | 1 h（無窗口）＋ 兩趟短臂 |
| **M1** | `fill_begin`／`fill_await` 拆分，**預設關**，同一顆 binary A/B | **⛔ 停工 —— 交付路徑裡沒有合法的 await 點**，見 `docs/L3_M1_BLOCKED_STRUCTURAL_2026-09-20.md`：分段派送器只以 `ask=false`（算完之後）叫 callback；`hook_seg` 在 fill 前已把 GPU 等乾淨；把提交提前＝程式庫自己標為 racy 的那條路，實測 **logits 全 NaN**（`non-finite 993280/993280`）。⇒ 這 44 ms 的鑰匙在派送器的**分段提交**（`ggml-backend.cpp`，線 A），不在 cache | 不寫（避免預設關著的純負債） |
| **M2** | 開 async（`CGC_FILL_ASYNC=1`）量劑量 | 一對臂（旋轉序、≥3 reps、丟 rep 1） | `fill_wait_us` ↓ 且 `total` ↓ 且 gap 沒等量升 ⇒ PASS；否則 NULL／FAIL 照 §2 判 | 2 launches（~20 min） |
| **M3** | 若 PASS：把 async 疊到 MTP verify 批次（T=2–4） | 同上 ＋ `mtp_accept_ab.py` 讀數 | accept 不變（同臂比）、step 不升、M1/M2/M3 仍 9/9/9 | 1 h ＋ 2 launches |
| **M4** | 若 PASS：把結果寫進 profile 格（`prefill250`／`prod25` 預設值）並跑一輪交付 `prod_profile.py` | 報告 ＋ profile 註解 | prefill **不回退**（281.51 ± 11.46 維持 PASS 250）、decode 直讀 ≥ 現況 | 1 h |
| **M5** | 收尾：報告、docs index、memory §EN、pre-commit 閘門、零殘留行程 | `docs/L3_ASYNC_FILL_RESULT_*.md` | `window_gate` ungated 數不增加；`pgrep` 零殘留 | 0.5 h |

**每一格的共同閘門**（不得跳過）：M1/M2/M3 oracle 9/9/9（`scripts/check/m123_oracle_gate.py`，預設 profile 已是 `prefill250`）＋
窗口 sidecar（每臂自己的 `.window.json`，回答「這筆數字是不是在吵的盒子上量的」）＋ engine digest（binary md5）＋ 零殘留。

---

## 5. A/B 協議（照本專案已驗證的紀律）

1. **窗口**：`scripts/check/server_window.py`／`window_gate.py`，逐臂判定（不是只在起跑前判一次）。
2. **輪替臂序**：每 rep 換序（`off → on`、`on → off`）；啟動順序本身就是自變數（先前已量到 carried swap 主宰）。
3. **≥3 reps、丟 rep 1、取配對中位**；比值可引用，絕對值必須掛形狀標籤（prompt 長度、thermal、swap）。
4. **同一輪同時讀四個數**：`fill_wait_us`、`total`、`gap`（GPU idle）、t/s ＋ accept／`mean_len`；缺任何一個就不是判準。
5. **provenance**：每份產物蓋 binary md5、model head hash、pool/PROFILE、`--extra-env` 全量 diff（`CGC_DUMP_ENV=1`）。
6. **拒斷規則**：跨幾何（pool 不同）、跨 probe（prompt hash 不同）、轉帶儀器的 run ⇒ 一律不比速度。

---

## 6. 期望值（用交付 work-row 的當代數字算，不是反推）

交付直讀：`total 247.98 = wait 159.77 + cb 74.18 + submit 10.09`；層總量：`gpu_busy_sum 181.56`、`union_sum 149.24`、`gap_sum 95.17`。

> ⚠️ **2026-09-20 修正（撤回一個硬地板）**：`gpu_busy_sum` **不是** busy 地板。`ggml-backend.cpp:2049`
> 自己的定義：`busy_sum: Σ(end-start) over the segment's n_cb+1 buffers (**overlap counted twice**)`，
> 而 `union: max(end)-min(start)`。交付行裡 `busy_sum/union = 1.22` ⇒ 它就是被重疊撐大的那一項。
> 誠實的地板是 **`union_sum`**：全部 gap 消失時 `step → 244.41 → 149.24`（`union_sum + gap_sum = 244.41`
> 對 `total 247.98`，差 +1.4% ⇒ 兩個分解各自閉合）。先前寫的「硬天花板＝GPU busy 181.56 ⇒ 18.59」作廢。

| 情境 | step (ms) | t/s @ml 3.375 | t/s @ml 4.0 |
|---|---:|---:|---:|
| 現況（直讀） | 247.98 | **13.61** | 16.13 |
| `cb` +`submit` 全藏（＝gap 全消失） | 163.71 → 159.24 | **20.62 → 21.19** | 24.43 → 25.12 |
| **L3 重疊做滿（硬天花板＝`union_sum` 地板，gap 也全消失）** | **149.24** | **22.61（+66%）** | 26.80 |
| 誠實區間（部分遮蔽、slot 佔用反噬） | 180–230 | **14.6–18.7** | 17.3–22.2 |
| 25 t/s 需要 @ml 3.375 | ≤135 | — | — |
| 25 t/s 需要 @ml（union 地板） | 149.24 | 需要 **ml ≥ 3.73** | — |

**必須一起講的四件事**：

1. 純 async 的算術上界是 **`union_sum 149.24`**（不是 `gpu_busy_sum`）⇒ **22.61**；先前的 18.59 已撤回。
2. 即使重疊完美 ＋ k=3 accept 做滿（ml=4.0），只有 **26.8** —— 這才是「25 是否不可達」的正確算術：
   **在 union 地板上，25 t/s 只需要 ml ≥ 3.73**（今日 3.375），所以 25 不再是「被地板擋住」，而是
   **「重疊能做到多少」與「accept 能做到多少」兩者的乘積問題**。
3. 這個地板的**可達性**沒被證明：gap 的主因可能不是 host 工作量，而是**依賴延遲**（union 的 ids 是 GPU argsort 算出來的，
   host 必須等 segment i 落地才知道填充什麼）—— 那部分 M1 已證明今天的派送器表達不出來。所以 22.61 是**上界**，
   不是預期值；可交付的 in-cache 部分實測只有 submit + commit→launch ≈ **8.5% → ~14.8**。
4. 剩下的缺口只能在 **gap 的成分（依賴延遲 vs 可移出的 host 工作）** 與 **bytes（池子／量化）** 兩側，
   不是 policy（placement 家族已用 ≤2% 的上界關門）。

---

## 7. 風險、範圍與 rollback

| 風險 | 機制 | 對策 |
|---|---|---|
| **torn read（數值災難）** | await 沒命中 ⇒ GPU 讀未落地 slot | fail-closed：`fill_await_missed` 必須恆 0，否則該步拒走 async；M1/M2/M3 為合併條件 |
| **時間搬家（NULL）** | cb 的時間移到 `wait` | 三個數同讀（§2）；斜率 ≥1 的既有證據說明這是預設結局 |
| **slot 佔用反噬** | in-flight 期間該層 slot 不可淘汰 ⇒ 可用槽短期變少 ⇒ miss ↑ | 量 `hits/misses` 與 `no_free_slot` 當副讀數；若 miss 顯著上升即判 NULL 而非 PASS |
| **共享檔衝突** | 另一條線持有 `ggml/src/ggml-backend.cpp`（S2／hook_seg） | ⚠️ **已反轉**：M1 停工後，這個檔不是「可以避開」，而是**唯一的落點**。交接單＝`docs/L3_M1_BLOCKED_STRUCTURAL_2026-09-20.md`（三事實 ＋ 一個 NaN 讀數 ＋ 一個待定價的 A/B 段切法） |
| **launcher 靜默丟 env** | 字面值覆寫（`CGC_EVICTED_RING=0` 前例） | 新 env 一律 `${VAR:-}`；`CGC_DUMP_ENV=1` 驗「設／未設都只輸出一次」 |
| **rollback** | 全部改動都在 `CGC_FILL_ASYNC` 之後，預設 0 | 預設路徑等價於今天（M1 閘門 1 即證明）；若不通過，flag 留作**已記錄的否證器**，不設為預設 |

**檔案所有權**：本方案會碰 `src/llama.cpp/src/llama-expert-cache.{h,cpp}`、`src/llama.cpp/src/llama-context.cpp`、`scripts/run_server.sh`
（三者目前**都有未提交改動**，含 ring 與 M-F5 儀器）⇒ 動工前先確認沒有別的 session 正在同一輪 rebuild；build 產物是共享的。

---

## 8. 明確不做

- 不做「用預測器餵 fill」（L1 已關門：K=64 → 99.19% resident、池子 17.9× 超額）。
- 不改 `ggml-backend.cpp` 的 submit 分段（那是線 A 的 S2，天花板 18.7、建議不做完；本方案與它**不同塊地**）。
- 不動 prefill 路徑（async 只在 `cgc_is_decode_graph()` 為真的圖上生效；prefill 的 whole-layer slab 不受影響 ⇒ **281.51 不回退**）。
- 不在同一輪同時改 slot 幾何（跨幾何不可比，會把 PASS/NULL 讀成噪音）。

---

## 9. 執行清單（複製即可跑）

```bash
# M0 前置：確認既有計數器會出現在 decode row（這一步本身可否證）
CGC_SERVER_PROFILE=prefill250 CGC_HOOK_SPLIT=1 CGC_DECODE_PROFILE=1 CGC_GPU_TIMING=1 \
  ./scripts/run_server.sh            # 之後 grep: fill_wait_us / CGC-FSSPLIT / CGC-HOOKSPLIT

# M1 閘門 1（bit-identity，同一輪）
python3 scripts/check/m123_oracle_gate.py --ref <v6 參考> --profile prefill250

# M2 A/B（旋轉序、≥3 reps、丟 rep 1）
CGC_FILL_ASYNC=0 … ; CGC_FILL_ASYNC=1 …     # 每臂寫 .window.json，逐臂過窗口
python3 scripts/check/prod_profile.py --profiles prefill250 --cells decode
```

**交付物**：`docs/L3_ASYNC_FILL_RESULT_*.md`（含 engine digest、窗口樣本、四個數的同輪讀值、PASS／NULL／FAIL 判定）。

---

## 10. 一句話

**L3 是「把 13.61 往 22.6 的 `union` 地板推」的那一筆，不需要預測器；但它到不到 25 取決於兩件今天都沒量的事：
gap 有多少是可移出的 host 工作（而非依賴延遲），以及 accept 能不能從 3.375 走到 3.73 以上。
（原句寫「推到 15–18.6、+37% 上限」——那是用 `gpu_busy_sum` 當地板算出來的，已撤回，見 §6。）**
