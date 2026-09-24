# 兩個「並行」旋鈕的配對 A/B — 2026-09-21

> 回答兩個問題：① `CGC_SERVER_WORKERS`（池 fill worker，IO 並行）② `-np N`（序列並行）。
> 兩者在 2026-09-21 之前都是**零實測／無結論**。本文兩個都結案，兩個都**不是槓桿**。
> 逐日經過：`.workbuddy/memory/2026-09-21.md` §EN-364 / §EN-365；perf 權威在 `MEMORY_PERF.md` 末節。
> 機器：Mac16,12 / M4 / 16 GB；模型 `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`（12.72 GiB）。

---

## 0. 先看結論

| 旋鈕 | 對照 | 結果 | 判決 |
|---|---|---|---|
| `CGC_SERVER_WORKERS` | W32 vs W8 | **1.0805**（3/3 對全部偏向 W8） | **往上加會慢 ~8%** |
| `CGC_SERVER_WORKERS` | W4 vs W8 | **0.989**（0.741 / 0.989 / 1.047，跨過 1.0） | **往下減沒差別（未決）** |
| `CGC_SERVER_CONCURRENCY` (`-np`) | N=4 vs N=1 | 聚合 **2.0 t/s vs 13.78 t/s**（HTTP），比值 **0.14** | **損失 7 倍，不是槓桿** |

⇒ **兩個旋鈕都維持預設（`WORKERS=8`、`-np 1`），不進計畫。**

---

## 1. 儀器與合規性（有差別，且差別重要）

| | T2 `CGC_SERVER_WORKERS` | T1 `-np N` |
|---|---|---|
| 交付儀器 llama-bench | ✅ **可用** | ❌ **不可用** |
| 原因 | 它只是個 env（`run_server.sh:1353` → `LLAMA_EXPERT_CACHE_WORKERS`），`prod_matrix.resolve()` 經 `run_server.sh CGC_DUMP_ENV=1` 翻譯後進 bench | `llama-bench --help` **全表沒有 `-np` / `--parallel`**，它只驅動單一序列；`-np` 是 `run_server.sh:1190` 的 server 參數 |
| 替代 | — | 自寫 HTTP driver：`Backup/phase_decomp/parallel_decode_ab.py` |

⚠️ **T1 的絕對值不可當引擎 decode 速度引用。** 已知 HTTP 與 llama-bench 不同口徑
（12.36/12.95 vs 7.74/10.80）。緩解方式：同儀器、AB/BA 交錯、交付的是**對內比值**與
**同輪的 N=1 控制**。

### 13.88 是 HTTP，不是 llama-bench（逐 burst 實錄）

同一次 N=1 臂、server 自己的 `print_timing` `eval time` 行（128 token / burst，共 3 次）：

| burst | t/s |
|---|---|
| warmup | 9.51 |
| rep 1 | 13.68 |
| rep 2 | **13.88** |

⇒ 報告裡的 13.88 是**最後一個 rep**，不是中位；兩個有效 rep 的中位是 **13.78**。warmup 低 31%
（冷池，與 `--warm-skip` 不跨 rep 的既有結論同向）。

**同 build（`efeade7e2`）的 llama-bench 錨點**：T2 的 W8 控制臂（prod25 / decode /
`median(reps[1:])` n=2）

| | pair1 | pair2 | pair3 |
|---|---|---|---|
| W8（vs W32 那輪） | 11.191 | 11.028 | 11.280 |
| W8（vs W4 那輪） | 11.149 | 11.303 | 8.360 |

⇒ **llama-bench ≈ 11.0–11.3**，HTTP 13.78 高出約 **+22%**。這是儀器偏差，不是引擎變快了；
**13.88 不得寫進任何交付表格**。

**但 T1 的結論不受影響**：它交付的是**同儀器內的配對比值 0.14**（兩臂都走 HTTP）。儀器偏差
量級 ~20%，實測損失 ~86%（7 倍），差一個數量級 ⇒ 不可能由口徑解釋。

T2 的三個臂都額外釘死 `CGC_SERVER_EXPERT_CACHE_BYTES=8589934592`：
**BUDGET 會隨 free memory 自動縮放**（空載時 dump 出現過 10 GiB）⇒ 不釘就是一個隱藏變數。

---

## 2. T2 — `CGC_SERVER_WORKERS`（pool fill worker 數）

配對設計：`scripts/check/paired_ab.py`，profile=prod25／cell=decode／pairs 3／reps 3，
兩臂共用 `CGC_PREFILL_STREAM=1;CGC_GATHER_SLAB_CAP=256;CGC_SERVER_EXPERT_CACHE_BYTES=8589934592`。

### W8 vs W32

```
pair          A(W8)     B(W32)     A/B
1            11.19      10.92     1.025
2            11.03      10.21     1.081
3            11.28       8.49     1.328     <- B 這對 worst=MODERATE、headroom 3636 MB，不乾淨
median A/B = 1.0805    spread 1.025..1.328    MAD 5.17%
SEPARATED: 3/3 對全部偏向 W8
```

### W8 vs W4

```
pair          A(W8)     B(W4)      A/B
1            11.15      11.27     0.989
2            11.30      10.80     1.047
3             8.36      11.29     0.741     <- A 這對出現 5.76 的離群樣本，worst=MODERATE
median A/B = 0.989     spread 0.741..1.047   MAD 5.83%
NOT DECIDED: 區間跨過 1.0
```

### 判讀

- **往上加會傷**（32 比 8 慢 ~8%），**往下減沒用**（4 ≈ 8）。⇒ 這個旋鈕**兩邊都沒有油水**。
- 與 `run_server.sh:1341-1352` 的既有註解一致：加 worker 在 **MTP on** 時會 under-deliver，
  因為 MTP fast path 的 cold fix 走 `llama_expert_cache_ensure_slot`（自己開 3 threads/expert），
  **不進池**；只有 `ensure_batch` 用這個池。本次正是 MTP on ⇒ 符合預期。
- 儀器雜訊底 MAD 5.2–5.8%，而效應 ~8% ⇒ **只比雜訊高一點**。所以寫「不起槓桿」，
  不寫「32 有害 8%」這種精度。
- 唯一還沒關的門：MTP **off** 時 workers 有沒有用（那時 fill 主走 `ensure_batch`）。
  但 MTP off 不是交付形狀，且 MTP-on 都沒效 ⇒ **不排入計畫**。

---

## 3. T1 — `-np N`（序列並行）

### 3.1 讀數（用 server 自己的 `slot print_timing`，比我的 driver 更權威）

| | N=1 | N=4 |
|---|---|---|
| eval time / 128 tok | 9 221 ms | 251 498 – 293 389 ms |
| ms / token | **72.04** | **1 965 – 2 292** |
| **單流 t/s** | **13.78**（2 rep 中位；tg 13.59／13.68／13.88） | **0.44 – 0.51** |
| **聚合 t/s** | 13.78 | **≈ 2.0** |
| prompt eval | — | 4.5 – 5.9 s / 12 tok |
| draft acceptance / mean len | 0.444 / 2.31 | 0.459 / 2.35 |

- **4 個 slot 的數字完全相同**（0.44/0.44/0.44/0.44 …）⇒ **server 確實把 4 條序列批進同一個
  decode step**，機制是活的 —— 不是「沒並行起來」，是「並行起來了但代價遠大於收益」。
- 3 個 burst（warmup + 2 reps）全部完成，沒有崩潰、沒有 OOM、沒有 deadlock watchdog 觸發。
- 預註冊判準 `r = agg(4)/agg(1)`：`r < 1.10 ⇒ NOT A LEVER`。實測 **r ≈ 0.14**。

### 3.2 機制：慢在哪（teardown 計數器）

| | N=1 | N=4 | 倍率 |
|---|---|---|---|
| read jobs | 62 829 | **2 218 197** | **35×** |
| read bytes | 8.36 GB | **0.57 GB** | **0.066×（反而更少）** |
| 平均 job 大小 | **133 KiB** | **≈ 257 B** | — |
| `us/job`（worker 加總） | 18.9 ms | 32.5 ms | 1.7× |
| pool hit rate | 18 358/25 875 = **70.9%** | 10 347/11 001 = **94.1%** | 命中率反而**升** |
| 每 step 的 job 數 | ≈ 1 142 | **≈ 13 692** | 12× |
| 每 step 讀的 bytes | ≈ 145 MB | ≈ 3.3 MB | **0.023×** |

⇒ **慢不在 miss（命中率還升了）、也不在頻寬（讀的 bytes 還少了 44×），
在 per-job overhead：4 條序列把每 step 的專家 union 撐寬，池的 fill 路徑碎成上萬個 257 B 的小讀。**

這是「**miss 不是驅動項**」的第二次獨立證據（第一次是 §EN-358 的 IO-path A/B：
4 GiB 臂 miss 高 1.91×／capacity 高 3.15× 而 t/s 持平）。

### 3.3 ★ 更正 §EN-363 的 bytes 模型

`step = (D + B·E) / BW_eff`（D=790 MB dense、E=381 MB/token、BW_eff=13.65 GB/s）
只算了「790 MB 被 N 條序列攤薄」，**完全沒有 per-job overhead 這一項** ⇒

- 模型預測 **N=4（MTP on）聚合 16.05 t/s**
- 實測 **2.0 t/s** ⇒ **差 8 倍**

⇒ 該模型從「假說產生器」降級為「**已被證偽的外推**」。
**不要再拿它排並行的優先序。** 它對 B=1（85.8 ms）與 B=4（167.81 ms）的兩個標定點仍然吻合
（那是同一條序列內的 batch，不會撐寬專家 union），所以它只在單序列範圍內還有一點用。

---

## 4. 「看起來當機」的真正原因（不是崩潰）

1. **np4 每個 128-token burst 要 250–293 秒**（np1 只要 9 秒）。warmup + 2 reps ≈ **830 秒／臂**
   ⇒ 一支臂就跑了 14 分鐘，表現得像 hang。
2. 我又把輸出接了 `| tail -40` —— **管道緩衝，進度行一個都沒吐出來**，於是「14 分鐘零輸出」。
   ⚠️ **教訓：長跑不要接 `tail`；要看進度就讓它直接寫 stdout，或用 `--json` + 直接看 log。**
3. 第二對的 server 在 01:47 隨著背景 shell 一起消失（外部因素，非程式錯誤）。
   沒有 OOM、沒有 `Received SIGTERM`（我方自己發的除外）、沒有 watchdog abort。

---

## 5. 對「25 tok/s」的影響

- 使用者已裁定 25 = **單流 decode**。並行在單流上只會更慢（N=4 單流 0.44–0.51 t/s），
  在聚合上也是 7 倍損失 ⇒ **並行對 25 沒有貢獻，兩條路都不成立。**
- 本次 N=1 控制讀到 **13.78 t/s（兩個 rep 中位，HTTP 口徑）**；既有單流基線中位 12.77 也是
  **HTTP 口徑**（M4 那輪走 server）⇒ 兩者可互比（+8%，在噪音內）。
  ⚠️ **同 build 的 llama-bench 是 11.0–11.3** ⇒ 要寫交付數字就用 llama-bench，別用 13.78。
- **結論不變：單流 25 下架，可達區間仍是 12.5–14。** 兩個並行旋鈕現在都是**已排除**而非未知。

---

## 6. 產物

- `Backup/phase_decomp/parallel_decode_ab.py` — 序列並行 driver（自帶 preflight、只 kill 自己起的 pid、
  預註冊判準寫在檔頭）。
- `Backup/phase_decomp/t2_workers/t2_w8_vs_w32.json`、`t2_w8_vs_w4.json` — T2 兩輪配對原始資料。
- server log（含 `print_timing` 與 teardown 計數器）：
  `Backup/cgc_logs/llama_server_20260921_013016.log`（N=1）、
  `Backup/cgc_logs/llama_server_20260921_013118.log`（N=4）、
  `Backup/cgc_logs/llama_server_20260921_014511.log`（N=4 第二對，被外部砍掉）。
- ⚠️ T1 **沒有產出 json**（背景 shell 在第二對中途消失）⇒ 本節數字取自 server log，不是 driver 的彙整。
