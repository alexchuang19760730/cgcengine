# Decode 的 host 端 top-k hook（`cb`）74 ms/回合：拆解、成因為何、哪些部分可移出臨界路徑

日期：2026-09-20　範圍：`demo/sweet-spot-windows-fix`　載體：Nail IQ3_XXS-denseIQ4X、8 GiB pool、MTP on、prefill250 profile
方法：**本回合沒有跑任何 server**。以下每一個數字都來自磁碟上既有的 server log 或原始碼，並標明它是哪一份產物。

---

## 0. 一頁結論

| 問題 | 答案 |
|---|---|
| `cb` 74 ms 裡有什麼？ | **99.7% 是 `ensure`（該層 union 的 demand fill）**。pre 0.16 ms/step、drain 0.012、tail 0.016 —— 三者合計 0.2 ms，是 74 ms 的 **0.25%**。 |
| 為什麼隨 token 成長？ | **不是 per-token 的 CPU 記帳**（那部分 ≤0.2 ms/step）。是 **union 的冷成員數**：union 隨 token 變寬 ⇒ 更多層「至少有一個冷專家」⇒ 每層一次 fill round-trip。 |
| 成本的單位是什麼？ | **一次 round-trip（≈1.65 ms），不是 bytes。** 實測 `0.37 MiB/job, us/job=1647, 235 MiB/s`；同一台裝置 823 MB/s ⇒ 0.37 MiB 的資料本身只要 ~0.45 ms，**每筆讀有 ~1.2 ms 是固定開銷**。 |
| 哪些部分可移出臨界路徑？ | 只有 `ensure`，而且它不是 CPU 而是**等待**。機制已存在但**關著**：`CGC_LAYER_AHEAD_PREFETCH` / `CGC_PREV_TOKEN_PREFETCH` 兩者都沒設；而 SpAc（唯一開著的預取）每步都在跑卻蓋不住冷集合。 |
| 移除 `cb` 值多少？ | 248.0 → 173.8 ms ⇒ mean_len 3.375 下 **13.6 → 19.4 t/s**。**單獨不足 25**（需 ≤135 ms），但它是目前最大且有名字的單項（步時的 **29.9%**）。 |
| 可否證的單項目標 | F1（latency vs bandwidth，一次量測、零程式改動）、F2（overlap：cb ≥50% 下降）、F3（round-trip：每層 1.85 → ≤0.6 ms）、F4（暖池對照：cb ≤1 ms）。判定規則在 §5。 |

---

## 1. `cb` 的組成

來源：`Backup/cgc_logs/llama_server_20260919_032308.log` —— 全庫**唯一**一份同時帶 `CGC_HOOK_SPLIT` 與 `CGC_DECODE_PROFILE` 的 log。
`cb` 在 dispatcher 的定義（`ggml-backend.cpp:2524`）就是 `st2 - st1`，`st1..st2` 夾住的正是 `sched->callback_eval(ttopk)`，也就是整個 `expert_cache_on_topk` hook。`CGC_HOOK_SPLIT` 把同一個 hook 切成四塊：

| 區塊 | 內容 | µs/call | 每步（40 層） | 佔 `cb` |
|---|---|---|---|---|
| `pre` | 診斷 + top-k unwrap + union 建構 | 4.1 | 0.16 ms | 0.2% |
| **`ensure`** | **`llama_expert_cache_ensure_batch`：批次 pread + 槽位指派 + 等 outstanding==0** | **736.4** | **29.5 ms** | **99.7%** |
| `drain` | `llama_expert_cache_drain_layer` | 0.3 | 0.012 ms | 0.04% |
| `tail` | publish + remap leaf + union 記錄 | 0.4 | 0.016 ms | 0.05% |

最後一行 `CGC-HOOKSPLIT: n=2880 pre=4.1 ensure=736.4 drain=0.3 tail=0.4 us/call (total 741.2)`。

**這三件事因此被排除為 `cb` 的成因**：remap leaf 的逐 token 寫入、union 的記錄、以及所有 hook 內的診斷。它們加起來 4.8 µs/call。

⚠️ 口徑：上表是**整段 run 的移動平均**（含冷啟）。暖態的實測在 §2。

---

## 2. 為什麼隨 token 成長 —— 而且「線性」這個說法要更正

### 2.1 per-layer 普查：`cb` 攤在「有 miss 的那幾層」上，不是集中在少數層

同一份 log 的 `CGC-DECPROF all:` 給出逐層 `cb`。把它按 step 匯總：

| step | ntok | cb 總和 | 層數(cb≥1 ms) | 層數(cb<0.05) | 最大層 | 中位 |
|---|---|---|---|---|---|---|
| 9 / 10 / 11 | 4 | **0.40 / 0.33 / 0.36** | 0 / 0 / 0 | **40 / 40 / 40** | 0.04 | 0.01 |
| 16 | 1 | 32.61 | 19 | 19 | L22 2.73 | 0.77 |
| 48 | 1 | 10.49 | 7 | 31 | L35 1.50 | 0.01 |
| 1–5（冷） | 2/4 | 129.6–191.6 | 37–40 | 0–1 | 8.8–16.1 | 2.89–4.33 |

兩個結論，都可直接讀出：

1. **union 駐留時 `cb` ≈ 0.4 ms/step（40 層全部 <0.05 ms）。** 這是「同一個 ntok=4、同一套 leaf 寫入、只差池子熱不熱」的對照 —— 所以 `cb` 是 **100% miss 驅動**，不是 per-token 工作量。
2. **有 miss 時，成本是「每層一次」的量，不是「每個 miss」的量**：
   - step 16：19 層 ≥1 ms，總 32.61 ⇒ **每層 ≈1.7 ms**
   - step 48：7 層 ≥1 ms，總 10.49 ⇒ **每層 ≈1.5 ms**
   - 冷啟：39–40 層 ≥1 ms，總 129.6–191.6 ⇒ **每層 ≈3.3–4.8 ms**（>1 個 round）

### 2.2 交付 regime 的同一條曲線（MTP-on、work rows、`gap_sum>0`）

來源：`Backup/cgc_logs/llama_server_20260920_023021.log`，修正後的 parser（work/shadow 分割）。

| ntok | n | total | wait | **cb** | submit | wait+cb+submit | cb/total |
|---|---|---|---|---|---|---|---|
| 3 | 2 | 197.47 | 138.06 | 55.04 | 4.38 | 197.47 | 27.9% |
| **4** | **107** | **247.98** | **159.77** | **74.18** | **10.09** | 244.04 | **29.9%** |
| 8 | 18 | 605.36 | 410.37 | 168.68 | 12.05 | 591.09 | 27.9% |

身分式 `wait + cb + submit = total` 成立（殘差 0–2.4%），所以 **`cb` 是可加在臨界路徑上的純損失**，不是被 `wait` 吸收掉的東西。

⇒ `cb/40 層` = **1.85 ms/層**（ntok=4），與 §2.1 的 1.5–1.7 ms/層 一致（不同臂、同結構）。

### 2.3 更正：這不是「隨 token 線性」

先前寫的 `ntok=1 → 19.61 ms；ntok=4 → 74.18 ms（比值 3.78）` 是 **跨臂**比較（19.61 來自 **MTP-off** 臂的 work rows，n=31；74.18 來自 **MTP-on** 臂）。MTP-on 臂自己的 ntok=1 列全部是 shadow 列（cb=0.01）。

臂內真正的形狀是 3 → 4 → 8：

```
ntok   3      4      8
cb    55.0   74.2  168.7      (tokens ×1.33, ×2.00 -> cb ×1.35, ×2.27)
```

⇒ token 數**不是**自變數。自變數是 **union 的冷成員數**，它隨 token 變寬（top_k=8：1 token → 8 個/層，4 tokens → 約 29，8 tokens → 約 57），透過兩個通道推高 `cb`：

1. **「這層到底有沒有 miss」的機率**：ntok=1 時只有 7–19/40 層有冷專家；ntok=4 時幾乎 40/40 層都有。
2. **每層的 round 數**：一旦 `3·m > 8`（8 = worker 數），每層就不止一個 round；ntok=8 時約 2.5 個 round/層。

**工作模型**（下面 §5 的 F1 就是去證偽它）：

```
cb ≈ 1.65 ms × Σ_layers ceil(3 · m_layer / 8)
```

ntok=4：74.18 / 1.65 ≈ **45 rounds** ≈ 40 層各 1 次 + 5 層第 2 次。ntok=8：168.7 / 1.65 ≈ **102 rounds** ≈ 2.5 次/層。兩者都吻合。

---

## 3. 成本的物理量：round-trip，不是 bytes

同一份 09-19 log 自己的讀取稽核行：

```
read shape: jobs=6150  bytes=2380029952 (0.37 MiB/job as one contiguous run)
            us/job=1647  effective_rate=235 MiB/s  total_bytes=2.22 GiB
```

- 每個 job 只有 **0.37 MiB**；6150 jobs / 2086 misses ⇒ **≈3 jobs/miss**（三顆投影）。合併讀（`fill_segments_pool` 的 sort-by-offset merge）**已經是預設開啟**，但相鄰 expert id 碰巧相鄰才合得起來，所以實務上仍是 3 jobs/miss。
- 這台裝置量到 **823 MB/s**（既有紀錄）。0.37 MiB 在 823 MB/s 下只要 **~0.45 ms**，而實測每 job **1647 µs** ⇒ **每筆讀有 ~1.2 ms 是固定開銷（syscall/seek/queue latency），不是傳輸**。
- 8 個 worker、一層提交 `3·m` 個 job：`m=1` 時 3 個 job 一輪做完 ⇒ 一層的牆鐘 = **一個 round-trip ≈1.65 ms**，正是 §2.1 量到的「每層 1.5–1.7 ms」。

**這是 `cb` 的可否證核心：`cb` 的預算是「層數 × round-trip」，不是「bytes ÷ 823 MB/s」。**

順帶：`LLAMA_EXPERT_CACHE_WORKERS` 預設 8（`run_server.sh:1345`，可用 `CGC_SERVER_WORKERS` 覆寫），但**提高 worker 對 `m=1` 的層毫無用處**（3 個 job 本來就一輪做完）—— 所以「加 worker」不是這條線的槓桿。

---

## 4. 哪些部分可以移出臨界路徑

### 4.1 可以直接排除的

`pre` / `drain` / `tail`：合計 0.2 ms/step。**沒有東西可拿。**

### 4.2 真正的東西：`ensure` 是一個「等待」，而且相依是真的

層 `il` 的 union 由該層 argsort 產生，層 `il+1` 的輸入又相依於 `il` 的 FFN 輸出 ⇒ **同一層內無法提前**，這條鏈是硬的。但有三件事是可分離的：

**(a) 預測 —— 機制已存在、而且在交付 profile 是關的。**
相鄰 token 的路由重疊約 87%（既有量測）。`CGC_LAYER_AHEAD_PREFETCH=1` 會在層 `il` 的 hook 尾端，用**上一個 token** 對 `il+1` 的 ids 去 `prefetch_slot`，讓那些 pread 拿到 `il+1` 的整個 GPU 視窗去落地。它刻意放在 demand `ensure_batch/drain` **之後**，因此只可能佔用**空槽**（設計上從不驅逐），並在 bytes 落地後才 publish `slot_table`。

`run_server.sh:1487` 有 allowlist 通道，但**沒有任何 profile 設它**，且實測 dump 顯示交付 profile 裡它是空的：

```
CGC_NO_PREFETCH=1   CGC_SPAC=1   CGC_SPAC_ALPHA=0.75   CGC_MM_BITIDENT=1
（CGC_LAYER_AHEAD_PREFETCH 未設、CGC_PREV_TOKEN_PREFETCH 未設、CGC_PREFETCH_SRC 未設）
```

已知的誠實限制（原始碼自己寫的）：`drain_layer(next)` 會丟掉尚未開始的預取，所以只有真正落在視窗內的 fill 才算數 —— 這要靠 drop counters 量，不能假設。

**(b) 為什麼背景預取在交付 profile 被關掉：安全，不是遺忘。**
`run_server.sh:2099`（`SERVER_MTP=1` 時）設 `CGC_NO_PREFETCH=1`，理由寫在同一段註解與 `llama-context.cpp:2028`：**避免背景填槽／淘汰在 MTP verify 期間覆寫 GPU 正在讀的 slot**。所以「把預取打開」不是免費的開關，而是要**先給它自己的槽位空間或 publish 紀律**。

**(c) SpAc 已經每步在跑，卻蓋不住冷集合。**
`llama-context.cpp:2058` 讓 SpAc 的 EMA refresh **豁免** `CGC_NO_PREFETCH`，而 prefill250 設 `CGC_SPAC=1`（refresh 預設每步）。**所以交付配置裡確實有一個每步執行的預取驅動，而 `cb` 仍是 74 ms。** 這是「現行預取目標不是冷的那些專家」的直接證據 —— 也把 (a) 從「也許有幫助」變成「有具體對照」。

**(d) round-trip 本身。**
每筆讀 ~1.2 ms 的固定開銷 × 每層 3 個 job。要動它得讓「一層的 union」變成 ≤3 筆「每 kind 一大段」的讀；現行的 sort-by-offset merge 無法形成，因為非相鄰 expert id 在檔案裡不連續。

**(e) 冷專家數本身**（池子大小 / 量化）——已經知道，且不是 hook 改動。

> 補充一個對 (d)/(e) 有決定性的事實：`run_server.sh:1340` 的註解指出 **MTP fast path 的冷修正走的是 `llama_expert_cache_ensure_slot`（每個 expert 現場開 3 條 thread），不走 worker pool** —— 也就是那條路徑是**逐 expert 序列**。交付 regime 到底走哪一條，正是 F1 會順便回答的事（`BATCHDBG` 只在 `ensure_batch` 印，所以它的有無同時標出該層走哪條路）。

---

## 5. 可否證的單項目標

實驗前置：一個乾淨窗口（零 rival llama、既有守門）、同一顆 binary、`CGC_SERVER_PROFILE=prefill250`、MTP on、8 GiB pool、k=3、temp 0.4。全部走 `prod_matrix.py` / 既有 harness，不新增啟動器。

```
CGC_HOOK_SPLIT=1 CGC_DECODE_PROFILE=1 CGC_DECODE_PROFILE_ALL=1 \
CGC_GPU_TIMING=1 LLAMA_EXPERT_CACHE_BATCH_DBG=1
```
（最後一項本回合才進 allowlist，見 §7；在此之前它設了等於沒設。）

### F1 —— 每層成本是 latency-bound 還是 bandwidth-bound？（一次量測、零程式改動）

把 `BATCHDBG layer=N misses=M` 與同 step 的 `CGC-DECPROF all: L<N> ... cb=X` 配對，對 `X` 迴歸 `M`。
（對齊方式：`BATCHDBG` 在該 step 的 `CGC-DECPROF: step=` 行之前，用 step 行切段即可。）

- **平台型**（`m=1,2` 平在 ~1.6 ms，`m=3–5` 跳到 ~3.3，`m=6–8` ~4.9）⇒ **round-trip bound** ⇒ 目標是 §4.2(a) 或 (d)。
- **線性型**（`X ≈ m × ~1.6 ms`，沒有平台）⇒ **bandwidth bound** ⇒ 目標是池子/量化，(d) 一文不值。
- **判定門檻**：`slope(m=1→2) < 0.3 × slope(m=4→6)` 記為平台；否則記為線性。

**這一步的價值**：它把「該修 fatter read 還是該修 overlap」從辯論變成讀數。

### F2 —— overlap（layer-ahead）

在交付配置加 `CGC_LAYER_AHEAD_PREFETCH=1`，交錯 A/B、≥3 reps、丟 rep 1、配對中位。
- **通過**：`cb` 中位 **≤37 ms**（從 74.18 降 ≥50%）、step **≤210 ms**、`M1/M2` 不變（oracle gate 全綠）、`verify-strict: refused=0`。
- **否證**：`cb ≥55 ms` ⇒ 預取沒有進到 demand 路徑。此時讀 `drop_stats`：若 `no_free_slot` 為主 ⇒ 「只佔空槽」的紀律把它餓死了，設計要改成給它自己的槽位空間（這正是 §4.2(b) 的形狀），而不是「再加強預取」。
- **天花板要先講清楚**：把 74 ms 全拿掉 ⇒ step 173.8 ms ⇒ **19.4 t/s**（mean_len 3.375）。**單獨不足以到 25**（需 ≤135 ms）。25 需要它與 G4/G6 疊乘。

### F3 —— round-trip（只有在 F1 判為平台型時才有意義）

目標：**每層 1.85 → ≤0.6 ms**，即 40 層合計 ≤24 ms。充分條件是**同時**滿足：每層 job 數 `3m → ≤3`，且有效讀取率 `235 → ≥600 MiB/s`。
- **否證**：修完後再看同一行 `read shape`。若 `MiB/job` 上升而 `us/job` 仍 ≈1647 µs ⇒ 成本不是傳輸，merge 不是槓桿，回頭打 (a)。

### F4 —— 暖池對照（最重要的一條）

**已量到的上界**：union 駐留時 `cb` = 0.40 / 0.33 / 0.36 ms（ntok=4，40 層全部 <0.05 ms）。
- **通過**：任何修法要把 `cb` 推向 **≤1 ms** 才叫打到了機制。
- 一個把 cb 從 74 壓到 40 但沒有動機制（例如只是多一個 round 被隱藏）的結果，不算通過。

---

## 6. 誠實標註（引用時必須一起帶）

1. **§1 的 HOOKSPLIT 是不同一臂**（09-19 log：resident 7757 MiB、hit 93.9%、requests 34381），與 §2.2 的交付臂（8 GiB pool）不是同一次啟動。**可移植的是結構**（ensure 佔 99.7%、成本攤在「有 miss 的層」上），**絕對值要用交付臂自己的 74.18/40 = 1.85 ms**。
2. **`74.18 ms` 是 work-row 中位**（`gap_sum>0`，ntok=4，n=107）。修正前的 pooled 值 62.17 ms 是 work/shadow 混池的 artifact（見 `docs/DECODE_STEP_ROW_POPULATIONS_2026-09-20.md`）。
3. **`19.61 ms`（ntok=1）屬於 MTP-off 臂的冷池**，不可與 MTP-on 的 ntok=4 相減或相除（§2.3）。
4. **miss 的強制/容量組成會隨臂改變**：09-19 log 是 `compulsory=2086 capacity=0`（100%/0%），而長 run 的交付型 log（166342 requests、27032 misses）是 `25.6% / 74.4%`，evictions 26920。**同一句話不適用於兩個臂** —— 這也意味著「加池子」的效益必須在正確的那一臂上量。
5. `union/total` 在本 regime >1（層 span 重疊），不可以和 E2b 的 0.75 並排比。
6. 本回合**沒有跑任何 server**，全部數字來自既有 log + 原始碼；沒有殘留行程。

---

## 7. 本回合唯一的程式改動

`scripts/run_server.sh`：把 `LLAMA_EXPERT_CACHE_BATCH_DBG` 加進既有的 env allowlist 迴圈（`:1382`）。

理由與該迴圈上方對 `LLAMA_EXPERT_CACHE_STEP_DBG` 的註解**逐字同型**：這個儀器存在於 `llama-expert-cache.cpp:1159`（`BATCHDBG layer=N misses=M`，逐層 miss 數），但沒有進 allowlist ⇒ 透過啟動器**根本打不開**，於是「沒有任何讀數」會被誤讀成「沒有這個現象」。它正是 F1 需要的唯一儀器。

驗證（實跑）：
- `bash -n scripts/run_server.sh` → syntax OK
- `CGC_DUMP_ENV=1 CGC_SERVER_PROFILE=prefill250 LLAMA_EXPERT_CACHE_BATCH_DBG=1 CGC_HOOK_SPLIT=1 ./scripts/run_server.sh` → 啟動 env 同時出現 `LLAMA_EXPERT_CACHE_BATCH_DBG=1`、`CGC_HOOK_SPLIT=1`、`LLAMA_EXPERT_CACHE_WORKERS=8`、`CGC_NO_PREFETCH=1`
- `python3 scripts/check/window_gate.py check` → `ok: 4 ungated launcher(s), unchanged (gated: 9, total: 14)`

未 commit。沒有動 `ggml-backend.cpp` / `libggml-base`（overlap 線的）。
