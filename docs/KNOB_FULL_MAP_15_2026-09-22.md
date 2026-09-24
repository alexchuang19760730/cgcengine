# 15 個 knobs 的完整地圖：哪些是新的、哪些其實已經判過／不是自由變數

承接 `docs/KNOB_INVENTORY_VERIFIED_2026-09-22.md`（§EN-441）與
`docs/JOINT_SHAPE_KERNEL_AXIS_2026-09-22.md`（§EN-442）。
回答的命題：**「我們只掃了調度層 4 個 knobs，所以找不到 2×；要把 15 個全丟進 autotuner。」**

**半對。** 下面每行都掛了既有證據（全部零 GPU：讀源碼／既有的 json／既有的 doc）。
查完的結果是：**15 條裡有 4 條已經有判決或被自然實驗否定、2 條跟已有的 knobs 是同一個自由度、
1 條在這個 fork 不存在**，真正新的只有 2–3 條。但要討論的是另一件事：**瓶頸不在 knobs 的數量**，見 §2。

---

## 1. 逐條對表

| # | 你列的 | 真實把手 | 狀態（既有證據，零 GPU） | 篩選該用哪個儀器 |
|---|---|---|---|---|
| 1 | M（spec 寬度） | `CGC_POOL_MAX_TOKENS` / `CGC_SHAPE_M` | **已在掃**（M≤17 確認、routable 142 非 143） | e2e（`shape_knob_search.py --grid width`） |
| 2 | Pool size | `CGC_SERVER_EXPERT_CACHE_BYTES`（**不是** `LLAMA_ARG_EXPERT_CACHE`） | key 已修、ladder 3/4/6 GiB，**還沒跑**（§EN-441 那一輪是全同配置，已撤回） | e2e（`--grid budget`） |
| 3 | N_CB | `CGC_SERVER_N_CB`（直接設 `CGC_N_CB` 會被覆寫回 8） | 可達，未掃 | e2e（`--grid ncb`） |
| 4 | GDN 實作 | 無（`fused_gdn_ar/ch`） | **結案**：`build_delta_net` 在本 workload 一次都沒進去（`gdn_saw_fused/manual` 皆 0）⇒ 一直在量 0 | — |
| 5 | NSG | `CGC_MMV_NSG`（clamp 1..32） | **dense 那半已否證**（§EN-442：的那三條就是 IQ4_XS；上界 2.42%）；**MoE 那半未定** | surrogate：`mmid_shapes.py --tokens 1,2,3,4 --nsg-sweep` |
| 6–8 | tile / unroll / vector width | **改原始碼**（不存在 env 把手） | 代價 ≠ 94 s，是 **rebuild + 重跑 anchor gate**（會蓋掉別條線的 dylib）；且 dense 那半被同一個 2.42% bound 住 | 只能 e2e，且先寫 patch 到 `/tmp` 獨立 build（照 `DENSE_GEMV_INSTRUMENT` §4 的做法） |
| 9 | Prefetch 策略 | `CGC_PREROUTER`、`CGC_LAYER_AHEAD_PREFETCH`、`CGC_DRAFT_PREFETCH`… | **大部分已判**：prerouter 09-17 實測 `queued=0 precision=12.8%`（被 `prewarm_hot` 全擋，roadmap 上限 0.2pp）；R=1 prefetch 09-21 **撤回**（淨虧）；且 prefetch 家 7 個裡 **4 個 INERT**（`run_server.sh:1361` 硬釘） | surrogate / 先看執行計數 |
| 10 | Cache 置換策略 | **已實作**：`LLAMA_EXPERT_CACHE_WIN_PIN=K`（replacement policy change，default off = pure LRU）、`LLAMA_EXPERT_CACHE_PIN_PROFILE`、soft pool L0 hot / L1 warm | **未掃，但已被一次自然實驗否定過**（見 §3） | surrogate（`mmid_shapes.py`＋expert-cache 計數） |
| 11 | MTP：tree vs chain | **本 fork 不存在**（全樹 grep 沒有 tree/trie speculation） | 這一條不是 knob，是從零寫 engine；n_max／draft depth 實際上＝#1 | — |
| 12 | Op fusion | `CGC_MMV_FUSE`（與 `CGC_GLU_FUSED_DOWN` 綁） | G4 判決：**別寫融合 kernel**；且 prod 的 GLU_FUSED_DOWN 今天是惰性的（`can_batch_mmv_glu_down` 只在 fused 路徑被叫）。**但有相反記錄**：`autotune_mmv.py:108-115` 實測 777/800 rows divergent、端到端 **15.94 vs 22.07 t/s 更慢** | e2e + M1 gate（衝突未解，見 `JOINT_SHAPE_KERNEL_AXIS` §5） |
| 13 | Graph 分段 | S2 軸（`hook_seg`／submit 迴圈） | **S2 建議不做完**（`docs/G4_FUSION_ANSWER_2026-09-20.md`）；且實測 **5811/5811 的 MoE 落在 segment 的第一個 command buffer** ⇒ 可遮窗口 = 0 | — |
| 14 | Dense vs pool 記憶體分配 | **與 #2 同一個自由度** | dense 權重走 OS page cache（mmap），沒有獨立把手；唯一能做的是「縮池」把 RAM 還回去 ⇒ 它就是 #2 | e2e（`--grid budget`） |
| 15 | Batch / ubatch | `-b` / `-ub` | **不是自由變數**：pool 開著時 `llama-context.cpp` 把 `n_batch` **夾到 `cgc_pool_max_tokens()`(=8)**，寬 batch 會 `GGML_ASSERT(n_tokens_all <= cparams.n_batch)`；llama-bench 又不能外部設 n_batch ⇒ harness 固定 `-b 8 -ub 8`。**它被 #1 綁死**；對 decode 分支無 effect | — |

小計：**已判決／被否定的 4 條（4、9、13、12 的一半）、跟已有 knobs 重複的 2 條（14、15）、不存在的 2 條（11、#6–8 的 env 把手）**；
真的新的是 **5（MoE NSG 的 joint 軸）** 與 **10（WIN_PIN，但先看 §3）**，加上 #3 未掃。

---

## 2. 但真正的問題不是 knob 的數量，是 objective 的價格

同一件事用兩個儀器問，價格差兩個數量級：

| | surrogate（op 級，`shape_probe/mmid_shapes.py`） | 端到端（llama-bench, prod25 cell） |
|---|---|---|
| 跑完**一整條軸** | **~3 分鐘**（實測：`mmid_nsg_sweep.json` 19:57 → `_rotated.json` 20:00，6 個 NSG × 3 reps × 2 形狀，含輪轉） | 94 s **一次 launch** |
| 噪音底 | **~2.4%**（unset 與 NSG=2 同值卻差 2.42%） | **CV 11.2%**（10 個恆等 arm，mean 9.976/sd 1.119）／log σ≈0.28 |
| 認證 3% 效應 | 分鐘級 | **~1350 launch ≈ 35 h** |
| 認證 10% 效應 | 分鐘級 | ~121 launch ≈ **3.2 h** |

⇒ **把 15 個 knob 全丟進 autotuner，不會讓 3% 的效應變得可見** —— 它只是把同一份（不夠的）資訊預算
拆成 15 份，而且交叉項還會把辨識度打低。這正是 §EN-440 的工作邊界表說的那件事：16 launch、σ=0.16 時
要真實差距 **~68%** 才認證得到。

**所以正確的順序不是「加 knobs」，而是「換 objective」：**
① 用 surrogate 在**分鐘級**把 15 條全部篩過（第一次最便宜）；
② 只有出現 **≥10%** 的 arm 才值得送上 94 s/launch 的端到端；
③ 端到端上還是分不出勝負的，回到 surrogate 把它問得更準。

---

## 3. §10 為什麼已經被否定過一次（這條值得單獨講）

在花時間寫 LFU／機率式置換之前，先看 09-21 那場**自然實驗**（`docs/IO_PATH_AB_2026-09-21.md` §③，摘要見
`docs/IOCACHE_CONSTRAINT_ADMISSION_2026-09-22.md:140-142`）：

| 池 | miss 數 | capacity miss | `cb` | t/s |
|---|---|---|---|---|
| 8 GiB | 1× | 1× | 41.8 ms | 10.28 |
| 4 GiB | **1.91×** | **3.15×** | **90.0 ms** | **10.60** |

**miss 翻倍、capacity miss 三倍、`cb` 從 41.8 → 90.0 ms，t/s 不動。**
任何置換策略最多只能減少 capacity miss，而這個實驗一次給了它 3.15 倍的反向壓力卻沒反應
⇒ 那一族的效應今天多半落在 **~48 ms/步的 idle slack** 裡。

⚠ 這筆不能完全信：它是**跨 run** 比較（`IO_PATH_AB` §③ 自己也寫了 single-run ±27% ⇒ 不能跨場比）。
這正是為什麼我們一直在排 `pool budget 4 vs 8 GiB` 的**同場配對** —— 也就是 #2/#14 那一條。
**⇒ #10 的正確位置是排在 #2 之後，不是現在。**

---

## 4. 建議的排程（這是務實版本的回答）

1. **先做 #2/#14（pool ↔ dense RAM 分配）的同場配對** —— 它同時是 #10 的前置，
   且 grid key 已經修好（`CGC_SERVER_EXPERT_CACHE_BYTES`，ladder 3/4/6 GiB；8 GiB 是 ref 不進 ladder）。
2. **而 #5**： 與此同時用 surrogate 把 MoE 那條 joint 軸（tokens × NSG）在**乾淨窗口**（usable ≥ 7.8 GiB）
   重跑并把 null cell 壓到 ≤5%（上一輪 verdict 是 INVALID，null cell 寬 44%）。**分鐘級**。
3. **#3 N_CB / #9 剩下的 / #12**： 排後面，因為都沒有 ≥10% 的先驗。
4. **#6–8、#11**: 要寫 code ⇒ 先用 surrogate 證明有 ≥10%，再談 rebuild + 重 anchor。

---

## 5. 本文會變動的地方

- 若 #2 的同場配對顯示 miss 成本其實不在 slack 裡（t/s 有反應），**#10 立刻升到第一位**。
- 若 surrogate 在 MoE 上出現 ≥10% 的 arm，本文第 1 表會重排。
- 若 `#6–8` 有人寫出來的 patch，`libggml-metal` md5 就變了 ⇒ **anchor gate 必須重跑**，別用 `PROBE_ANCHOR=none` 繞。
