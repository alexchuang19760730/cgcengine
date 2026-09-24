# Stage 2-a 在 decode 上有沒有 value？

日期 2026-09-21 · cell `-np 1` / HTTP / 128 tok × 2 reps / 同一 build（`mtime 09-20 23:05`）
工具 `Backup/phase_decomp/perjob_s02.py`（既有，未改）、`coalesce_race.py`（本輪新寫）、
`parse_expert_geometry.py`（本輪新寫）

## 0. 結論先行

**沒有。上界約 1%，價值在 ±30% 的時間漂移底噪之下。建議不實作 Stage 2-a。**

而且原因跟直覺相反：**實體物理支持合併**（零 junk 時快 35%，吃到 4.5 倍額外位元組
仍然快 21%），**壞的是「機會」** —— 每一個 `(layer, step)` 只有 **1.82** 個 expert 是 miss，
散在 256 個 id 裡，彼此距離很少落在划算的範圍內。

> 一句話：**不是這帖藥沒效，是病人身上沒幾個傷口可以貼。**

## 1. Stage 2-a 在跟什麼換什麼

放寬 `llama-expert-cache.cpp:3528` 的合併條件（今天只接受**檔案內嚴格相鄰**），
讓 id 相距 `d` 的兩個同 kind expert 併成一發 `preadv`：

```
today      : 2 次 pread   -> 讀 2 × stride 位元組
coalesced  : 1 次 preadv  -> 讀 (d+1) × stride 位元組   ← 必須連中間的 expert 一起吞
```

所以要付「`d-1` 個 expert 的垃圾位元組」，換「少一次 syscall ＋ 少一個 queue slot」。
兩個貨幣：**stride（幾何）** 對 **per-pread 固定成本（IO 物理）**。下面分別量。

## 2. 證據 A：檔案幾何（直接讀 GGUF header，免費、不靠任何量測）

`parse_expert_geometry.py` 輸出（layer 0）：

| tensor | dims | quant | bytes | expert stride in file |
|---|---|---|---|---|
| `ffn_gate_exps.weight` | [2048, 512, 256] | IQ4_NL | 35,651,584 | **139,264 B** |
| `ffn_up_exps.weight` | [2048, 512, 256] | IQ4_NL | 35,651,584 | **139,264 B** |
| `ffn_down_exps.weight` | [512, 2048, 256] | IQ1_S | 53,477,376 | **208,896 B** |

⇒ 每個 `(layer, expert)` = **487,424 B（476 KiB）**；3 個 kind；**相鄰 expert id 在檔內嚴格銜接**
（這正是今天 merge 唯一能命中的情形）。`block_count=41`、`expert_count` 不在 header。

## 3. 證據 B：引擎計數器 —— merge 今天實際命中多少

同一 build、`-np 1`、4 臂，`file_reads` 三次完全相同（44505/44505/44505）⇒ 計數器零漂移：

| arm | `file_reads` | `usec/job` | wall | 說明 |
|---|---|---|---|---|
| base | 44,505 | 184.8 | 13.8 s | — |
| `NO_MERGE` | **44,868** | 179.9 | 13.5 s | job 數幾乎不變 |
| `NO_RDADVISE` | 44,505 | 180.7 | 13.5 s | 無效果（≤ base 漂移） |
| base2 | 44,505 | 181.9 | 13.6 s | 漂移監測 |

另有 `--reps 0`（只起來就停）測初始填池 = **912 reads**，所以 runtime 佔 98%。

```
runtime reads/rep   = (44505-912)/2   = 21,796     <- merge 開啟時的 job 數
segments/rep        = (44868-912)/2   = 21,978     <- 完全不合併時的 job 數
=> merge 實際命中   =      182 次/rep = 0.83% of segments
```

**np1 decode 上 merge 幾乎是死的**（跟 np4 的 0.018% 同性質）。配合 `mean_len=3.21`：

| 量 | 值 |
|---|---|
| steps/rep | 39.9 |
| **jobs/step** | **546** |
| miss/step | 74.5 |
| **miss / (layer, step)** | **1.82** ← 只有不到 2 個 expert 是 miss |
| bytes/job | 66,485 B |
| bytes/step | 36.3 MB |

（`request/hits/misses = 20679/14733/5946`，四臂**逐位元相同**，hit rate 71.2%）

### 順手拿到的一條新 fact
`usec/job` 是**平均攤銷**（`fill_batch_usec / file_reads`），**不是邊際成本**。
`NO_MERGE` 改變的 job 數只有 0.8% ⇒ **這個 cell 裡沒有任何旋鈕能測出 per-job 邊際成本**，
所以證據 B 只能給「機會」，不能給「單價」。單價要靠證據 C。

## 4. 證據 C：IO 物理 —— 直接對抗，不擬合任何模型

`coalesce_race.py`：8 thread（對應 `LLAMA_EXPERT_CACHE_WORKERS=8`）掃過檔案、**每筆偏移只用一次**
保證冷讀；偶數 sample 走 `today`（2 次 pread）、奇數走 `coalesced`（1 次 pread 覆蓋整段），
奇偶交替 ⇒ 漂移自動抵消。

| gap | 多吃 junk | today | coalesced | ratio | 裁決 |
|---|---|---|---|---|---|
| 1 | 0.00 MB | 1447 µs | **943 µs** | 0.651 | **合併快 35%**（零 junk = 天花板） |
| 2 | 0.13 MB | 1827 µs | **1115 µs** | 0.610 | 合併快 39% |
| 4 | 0.40 MB | 2512 µs | **1602 µs** | 0.638 | 合併快 36% |
| 8 | 0.93 MB | 3316 µs | **2602 µs** | 0.785 | **合併快 21%** |
| 16 | 1.99 MB | 4010 µs | 4160 µs | 1.037 | 打平 |
| 32 | 4.12 MB | **4079 µs** | 8052 µs | 1.974 | 合併慢 2× |

**讀出來的物理：這顆 SSD 在這個 regime 是 latency-bound，不是 bandwidth-bound** ——
gap=8 時合併要讀 4.5 倍位元組，卻還是快 21%。交叉點約 **gap ≈ 16**。

（冷讀 139 KB p50 **235 µs** / 0.59 GB/s 單流 vs 熱讀 **7.5 µs** / 18.6 GB/s，差 31×）

## 5. 兩條證據碰不到面 —— 這才是裁決

把「物理上划算的 gap」和「隨機能排到的 gap」放在一起。
每個 `(layer, step)` 只有 1.82 個 miss expert 散在 256 個 id 上，任兩兩距離 ≤ G 的機率 ≈ 2G/256：

| 轉換門檻 G | 每 step 省幾個 job | 佔 546 個 job | 物理上划算嗎（§4） |
|---|---|---|---|
| 4 | 2.9 | 0.52% | ✅ 快 36% |
| 8 | 5.7 | 1.05% | ✅ 快 21% |
| **16** | **11.4** | **2.09%** | ⚠ 打平（1.037） |
| 32 | 22.8 | 4.18% | ❌ 慢 2×（1.974） |
| 64 | 45.7 | 8.36% | ❌ 更慢 |

**「划算」和「有得砍」完全不重疊**：唯一有機會量的 G=32/64，物理上是賠的；
物理上賺的 G≤8，只能碰到 1% 的 job。成長是線性的，但物理成本長得更快。

## 6. 換算成 decode 上界

- `fill_batch_usec`（主執行緒在 `fill_segments_pool` 內的時間）/ step ≈ 103 ms，
  step ≈ 345 ms（假設 `mean_len=3.21`）⇒ **fill 最多佔 step 的 30%**（這是上界口徑）。
- 即使樂觀地假設：job 數砍 2%、被合併的 pair 各自省 20% IO、且 fill 全在關鍵路徑上：

```
decode 增益 ≤ 2% × 20% × 30% ≈ 0.1% .... （樂觀口徑 ~1%）
```

對照本專案已知的**單臂噪音 ±1.9 t/s / ±27%**（`MEMORY_PERF.md`）：這個效應證明不出來。

## 7. 裁決與下一步

| | |
|---|---|
| **Stage 2-a（放寬 merge 條件）** | ❌ **不做**。上界 ≤1%，低於噪音底；代價是寫 C++ ＋ 重建（會蓋掉別條線的 dylib）＋ 過 G2 bit-exactness。 |
| 它的真正價值在哪 | prefill/slab 路徑（那裡一次讀滿 256 expert，相鄰率 100%，merge 本來就吃滿）⇒ 那裡 Stage 2-a 已經是 no-op，不是未來的藥。 |

### 順手學到的、更有價值的兩件事

1. **瓶頸不在 request 形狀，在 pool slot 容量。** miss/(layer,step) 只有 1.82，而
   每層 **143 槽 vs 256 expert**（既有 M 曲線 datapoint：`pool 路徑 M ≤ floor(143/8)=17`）
   ⇒ hit rate 卡在 71%。**拉高命中率遠比便宜 request 值錢。**
2. **這顆 SSD 大到 2 MB 都還是 latency-bound**（§4：1.25 MB 一次讀 vs 278 KB 兩次讀，前者贏）。
   ⇒ 一個尚未驗證的槓桿：**在 2 MB 視窗內做 neighbor prefetch**，一次 miss 順手把同視窗的
   ~14 個 expert 帶進池 —— 幾乎白拿。但這是**假說**，且要跟 143 槽的容量限制一起看
   （帶進來就會 evict 別人），先用 `cb` 儀器確認 hit rate 能不能真的上去，不要再寫 C++。

## 8. 方法論教訓（本輪踩到，值得留著）

**量化「命中知道怎麼辦」之前會先被一樣東西騙：**

| 踩到的坑 | 症狀 | 修正 |
|---|---|---|
| 固定隨機種子 + 多輪 trial | 第 1 輪把那些頁讀熱了，後面每輪都 hit ⇒ `nocache` 跟 `cached` 一樣快（都 18–21 GB/s） | 每輪用**不重疊**的偏移，「reuse vs fresh」當作冷熱偵測器（本輪：冷 235 µs vs 熱 7.5 µs） |
| `final stats` 混了初始填池 | `file_reads/misses` 會被初始那 912 reads 污染，`job/miss` 從 8.36 飄到 7.33 | 補一次 `--reps 0` 差分 |
| 把 `usec/job` 當邊際成本 | 它是**攤銷平均**；要邊際得真的改變 job 數 | merge 在今天只命中 0.8% ⇒ **這個 cell 裡沒有旋鈕能給邊際**，改走 IO 微基準的直接對抗 |

## 附：重現命令

```sh
# 幾何（免費，不開 server）
/opt/homebrew/bin/python3 Backup/phase_decomp/parse_expert_geometry.py

# 機會：np1 四臂含 NO_MERGE 反向對照（~7 min）
/opt/homebrew/bin/python3 Backup/phase_decomp/perjob_s02.py --np 1 --tokens 128 --reps 2 \
  --arms base,no_merge,no_rdadvise,base2 --json Backup/phase_decomp/perjob_diag/s03_np1.json
# 初始填池差分
/opt/homebrew/bin/python3 Backup/phase_decomp/perjob_s02.py --np 1 --reps 0 --tokens 128 \
  --arms base --json Backup/phase_decomp/perjob_diag/s04_init0.json

# 物理：冷讀下合併 vs 分離的直接對抗（~2 min）
/opt/homebrew/bin/python3 Backup/phase_decomp/coalesce_race.py --iters 24 --threads 8 \
  --gaps 1,2,4,8,16,32
```
