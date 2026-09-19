# Decode 每層同步與 union gather 批次化 — 量測與結論 (2026-09-19)

問題（使用者）：把 decode 的每層 eval 同步與 union gather 批次化（一 step 少數幾次同步、單槽單讀），
量 step 是否從 135 ms 降到 ≤96 ms，並確認 M1/M2 仍 bit-identical。

一句話結論：**不能靠批次化達到 96 ms，因為每層的那一次同步是圖的真資料依賴；而它貴的原因不是
「同步次數」也不是「gather 沒批次」，是那次同步裡等的是 6% 的強制性（compulsory）首次觸及——每次
1.65 ms 的 pread 延遲。批次化已經做完了，剩下的是讓那 1.65 ms 不要擋在 GPU 前面。**

---

## 1. 我在 HEAD 量到的（不是推論）

配方：`CGC_SERVER_MTP=0`、模型 pin 在
`models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`、
**`CGC_SERVER_PROFILE` 沒設 ⇒ profile=off、pool=10 GiB**、port 8123、短 prompt、`-n 64`。
（135 ms 那個數字來自 4 GiB pool 的另一輪；10 GiB 上本來就不是 135。）

穩態 decode（`ntok=1`，step ≥ 16，n=8 步，driver 同一輪）：

| 臂 | decode t/s | step | wait | cb | submit | GPU union | GPU gap | GPU idle |
|---|---|---|---|---|---|---|---|---|
| off | 8.88 | 108.80 | 83.91 | 27.12 | 3.31 | 86.76 | 35.38 | 42.0% |
| on（`CGC_LAYER_AHEAD_PREFETCH=1`） | 9.19 | 103.75 | 75.81 | 26.60 | 3.19 | 76.27 | 35.69 | 42.2% |
| off2（對照，同一顆 binary） | 8.93 | 98.08 | 63.99 | 26.72 | 3.15 | 68.12 | 36.28 | 51.1% |

- 三個臂的 **cb 只差 0.5 ms（26.60 / 26.72 / 27.12）**，gap 只差 0.9 ms；
  兩個對照臂自己就差了 10.7 ms（98.08 vs 108.80）⇒ **on/off 的差落在漂移裡，這個預取沒有可量測的效果。**
- step 的最小值 72.9–79.6 ms，**≤96 ms 在最好的步上已經達成，但中位數還沒有。**
- 41 segments/step（`segs=41`），一次都沒少 —— 這是結構，不是旋鈕（見 §3）。

### 1.1 `cb` 是什麼：加了 `CGC_HOOK_SPLIT=1` 把它拆開

新的 env-gated 計時（`llama-context.cpp` expert_cache_on_topk，預設關）：

```
CGC-HOOKSPLIT: n=2880  pre=4.1  ensure=736.4  drain=0.3  tail=0.4 us/call (total 741.2)
```

| 區塊 | 內容 | µs/call | 佔比 |
|---|---|---|---|
| pre | 進入 hook → demand ensure 之前（診斷、top-k 取值） | 4.1 | 0.6% |
| **ensure** | **`ensure_batch`：這一層 union 的批次填充** | **736.4** | **99.4%** |
| drain | `drain_layer` | 0.3 | 0.04% |
| tail | publish slot table + remap leaf + union 記錄 | 0.4 | 0.05% |

（n=1120 時是 1108.8 µs，隨著 pool 暖起來收斂到 736；其餘三項在整個 run 內都是 µs 級。）

### 1.2 填充在等什麼：100% 是強制性首次觸及

同一輪 teardown：

```
miss attribution: compulsory=2086 capacity=0 (100.0% / 0.0% of 2086)  evictions=2086
                  layers_distinct_over_slots=0  worst=layer 0 distinct=106 slots=179
hits=32295/34381 (93.9%)
read shape: jobs=6150 bytes=2380029952 (0.37 MiB/job as one contiguous run)  us/job=1647
            effective_rate=235 MiB/s  total_bytes=2.22 GiB
```

- **hit rate 93.9%**、**capacity miss = 0**（一次容量淘汰都沒有；最壞那層要 106 個 expert、有 179 個 slot）。
- 每次 miss = **3 個 job（gate/up/down）× 0.37 MiB = 1.11 MiB**，每個 job **1.65 ms**、per-job 235 MiB/s。
- 29 個 miss/step × ~1 ms（一個 miss 在 hook 裡被擋住的量級）≈ **27 ms/step —— 就是量到的 cb。**

## 1.3 閘門：M1/M2/M3 仍 bit-identical

`m123_oracle_gate.py --profile prefill250`，參考 `ref_iq3_pool8gb_M2_6144_bitident_v6_nbaware.jsonl`
（v6，nb-aware），在**參考自己的 port 8080** 上跑（見下）：

| 臂 | comparable | config diffs | M1 | M2 | M3 | n |
|---|---|---|---|---|---|---|
| flag off（預設路徑） | **True** | `[]` | **9/9** | **9/9** | 9/9 | 9 |
| `CGC_LAYER_AHEAD_PREFETCH=1` | False | `ENV.CGC_LAYER_AHEAD_PREFETCH: ref='<absent>' now='1'` | **9/9** | **9/9** | 9/9 | 9 |

- **預設路徑是乾淨的 PASS**（`GATE la8080_off: PASS M1=9/9 M2=9/9 M3=9/9`，`comparable=True`）。
- 開啟旗標那一臂的 9 個 probe 也**逐位元相同**；它之所以只被標記為不可比，是因為參考檔比這個旋鈕更早存在
  （唯一的 `config_diffs` 就是那個 env 本身）。所以「這個預取不改變數值」是量到的，不是推論。
- 第一次跑在 port 8123 時兩臂都報 `INVALID COMPARISON`，而唯一差異是 `ARG[18]: ref='8080' now='8123'`
  —— **閘門的可比性判定包含 argv，所以換 port 就會讓一個 M1/M2/M3 全 9/9 的 run 失去裁決權**。
  要裁決這條線，一律用 8080。

## 2. 所以「批次化」在這條路徑上的三個答案

1. **「一 step 少數幾次同步」做不到**（不是沒做，是不能做）。
   切點在每層的 `ffn_moe_argsort-il`；`segs = 41`。第 il 層的 expert 集合是由 segment il **結尾**的
   argsort 產生的，而它的 MoE `mul_mat_id` 在 segment il+1 **開頭**就要讀那些 slot。CPU 必須在兩者
   之間把 union 填好，否則 GPU 讀到未就緒的 slot。要少於 41 次同步，就得在建圖時就知道「之後每一層
   的 routing」——那需要跨層預測，模型自己沒有這個資訊（第 il+1 層的 routing 依賴第 il 層 MoE 的輸出）。
   量到的證據正是這個依賴：`gap ≈ 1.2 × (cb + submit)`（三個臂 35.38/30.43、35.69/29.79、36.28/29.87
   ⇒ 1.16/1.20/1.21；逐 step 大多在 1.16–1.26）。換句話說：**GPU 空等的時間就是 CPU 在 hook 裡的時間。**

2. **union gather 已經是批次的了**，而且不是「每 expert 一次」：
   `llama_expert_cache_ensure_batch` 把該層**所有** miss 攤平成單一 job list，交給常駐 worker pool
   （`llama-expert-cache.cpp` 內 "flatten ALL of the layer's misses into one job list"），每個
   (layer, expert) 在**每個 tensor 上只有一次 pread**（read shape 的 "0.37 MiB/job as one contiguous
   run" 就是證據）。舊的「每個 miss 自己 spawn thread 再 join」路徑還在，但只作為 A/B
   （`LLAMA_EXPERT_CACHE_BATCH_SPAWN=1`）。

3. **剩下的槓桿是把 736 µs 從關鍵路徑上拿掉，不是把它批次化。** 而它之所以還在關鍵路徑上，
   是因為這 6% 的 miss 是**compulsory**——它們就是 routing churn 本身。用「上一個 token 的
   union」去預測（本輪 `CGC_LAYER_AHEAD_PREFETCH`）在定義上抓不到 churn 的那 13%：
   被預測中的 87% 本來就是 resident（`st_next[e] >= 0` ⇒ 一次 prefetch 都不會發），
   真正會 miss 的正是預測錯的那些。**這條量測把「預取 union」這個家族在 HEAD 上關掉了**，
   而它與 `docs/POOL_BUDGET_COST_DECOMP_2026-09-18.md` 的 pool 掃描一致（capacity=0 ⇒ 加 pool 沒用）。

## 3. 對 25 t/s 的算術（用今天這組數字）

| 項 | 現在 | 25 t/s 需要 |
|---|---|---|
| step | 98–138 ms（中位 ~108） | 40 ms |
| GPU busy | 約 46 ms | 46 ms 可接受（但要被更多 token 攤） |
| cb（強制填充） | 27 ms | ~0 |
| submit + 其他 CPU | 3–5 ms | ~0 |
| GPU gap | 35 ms | ~0 |

即使把 cb **全部**藏起來，step 也只到 ~73 ms ⇒ **13.7 t/s**。要到 40 ms，缺的是 GPU busy 本身
（46 ms）—— 那是 M3 工作項 2/3（gather/int8 融合、batched-union gather）的範圍，跟同步批次化無關。

## 4. 兩個「先修 instrument」的發現（本輪順手抓到）

1. **`fill_batch_usec` 對 pool 路徑是盲的。** 它只在 blob 路徑
   （`fill_segments_concurrent`）累加；pool 路徑走 `fill_segments_pool`。所以在 2086 次 miss 的
   run 上它讀 0，而 `pread_usec` 讀 10.1 s。任何用 `fill_batch_usec` 判斷「填充不是瓶頸」的結論
   都不成立 —— **這正是本輪要拆 cb 的原因**，而 `CGC_HOOK_SPLIT` 現在把它拆出來了。
2. **`CGC-HOOK` 的前 80 次列印是無條件開的**（`static int cgc_hook_dbg_n < 80`，不是 env-gated）。
   它會污染任何「前 2 步」的量測（本輪 step 1/2 的 cb 到 166 ms，有一部分是這 80 行 stderr）。

## 5. Provenance

- HEAD `98b44c8c6`（工作區：`src/llama.cpp/src/llama-context.cpp`、`scripts/run_server.sh` 為本輪改動；
  `llama-bench.cpp` 為上一輪未提交）
- binaries：`libllama.0.0.279.dylib 6b0c3a80…`、`libggml-metal.dylib 1be36630…`、
  `libggml-base.dylib 229f03a3…`（A/B 三臂與 gate 皆為同一次 build）
- 幾何：MTP off、profile=off、pool=10 GiB、`CGC_OA_ASYNC=1`（`CGC_DUMP_ENV=1` 已核對）
- **必須揭露的臂差（P0-1，另一個 session 指出，已核實）**：`run_server.sh` 把整個 MTP env
  區塊（`CGC_NO_PREFETCH`、`CGC_VERIFY_DECODE`、`CGC_DRAFT_DECODE`、`CGC_WARM_NPAST`、
  `CGC_MTP_NO_WARMUP`、**`CGC_MM_BITIDENT=1`**、`CGC_NO_SEQ_RM_PROBE`、`LAYER_CAPS=40-40:256`）
  包在 `if [ "$SERVER_MTP" = "1" ]` 裡。所以 `CGC_SERVER_MTP=0` **不等於**「同一個引擎、只是沒有投機」
  —— `CGC_MM_BITIDENT` 是把 M≤8 的 matmul 釘在 M-invariant `mul_mv` 路徑上，而 decode 的 GEMV 正是
  M=1，在範圍內。**本報告的 step/wait/cb/gap 絕對值因此量在非生產 kernel 路徑上**（結構性結論
  cb=ensure=填充不受影響，gpu/gap 的絕對值可能受影響）。正確的「無投機」對照是
  `CGC_SERVER_MTP=1 + CGC_SERVER_MTP_N_MAX=0`（同一個 launcher 分支、同一個 kernel 選擇、
  只差 draft 寬度）；`plain_match_ab.py` 已改用這個對照，見 `SLAB_POOL_HANDOFF` 與該腳本的表頭註解。
- **driver 缺陷（已記）**：`decode_step_profile.py` 不設 `CGC_SERVER_PORT`，所以 `--port` 只決定它
  自己輪詢哪個港；兩者不一致時它會等到 ready-timeout（本輪第一次就是這樣白等 5 分鐘）。
  另外它的 preflight 用 ps 掃 `llama-server` 字串，所以**呼叫它的那條 shell 命令本身**會被當成
  「別的 llama 行程」而拒跑（本輪遇到兩次）。

## 6. 下一步（依「每次量測能回答什麼」排序）

1. **把那 1.65 ms 的 miss 從關鍵路徑拿掉**（唯一有量級的一步）：不改 union 的批次化，而是讓
   miss 不再需要「先填好才能 submit」。這條路徑存在（GPU 端 slot table + kernel 端查表／fallback
   讀 raw 指標），但它在 S1 線上，且需要新的一輪 M1/M2 閘門。
2. **把 prefill 的 routing 用來預熱 decode 的 compulsory 集合**：compulsory=2086 且
   `record_routes` 已經在記 prefill 的 routing。若 prefill 結束時讓 decode 前 N 步的熱集先落地，
   那 2086 次首次觸及可以大部分消失（這是「pool 太慢」的實質解，不是「pool 太小」）。
3. **縮短每個 job 的延遲**：per-job 235 MiB/s（0.37 MiB/1.65 ms）與 SSD 規格差一個量級，
   值得確認是 3 個 job 的並行度、`O_DIRECT`/page cache、還是檔案在內接/外接碟上的差異。
   （注意：這一項**不會**讓 25 t/s 出現，它只把 cb 從 27 降到可能 10 ms。）
