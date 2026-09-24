# decode 25 — 2026-09-21 13:2x 判決（含「記憶體超訂」這個最後嫌疑的實測結案）

答：**沒有。** 而且是今天所有已封的路裡，第一次把「25 需要的倍數」和「剩下可動的比例」
放在同一把尺上量完的結論。

本輪只做了一件事：把今天早上留在 `§EN-382` 裡唯一「0 行碼可驗、優先級最高」的未排除項
——**記憶體超訂**——做成配對 A/B。它不是推論，下面兩個輸入都是算式／直讀。

---

## 1. 超訂的存在是算術，不是擬合

`run_server.sh` 每次啟動自己印（block `[budget]`，`CGC_DUMP_ENV=1 ./scripts/run_server.sh`）：

```
[budget] model       13030 MiB（load_mode=none => 匿名頁、不可回收）
[budget] expert pool 10240 MiB
[budget] 靜態需求    23270 MiB vs 實體 16384 MiB
[budget] OVERSUBSCRIBED by 6886 MiB：這個啟動只能在 macOS 記憶體壓縮 + swap 之上執行。
```

16 GB 機器要裝 **23.27 GB**。差額 6.89 GB 必須由 compressor + swap 供養。
⇒ 「有效頻寬只有峰值 15%」一直有兩個候選解釋：**真的是 DRAM 慢**，或 **那 15% 有一部分是解壓縮**。
這兩者用頻寬儀器分不出來（解壓縮就表現在慢上），只能用**移除壓力**來分離。

## 2. 移除壓力的辦法（這裡有一個陷阱）

launcher 建議的三個槓桿裡，`mmap` **早已判死**（同一支 script 自帶實測：`mmap` 在
ub=4096/5120/6144 **全部不存活 0/3**，死法是載完 40 層後 `ggml_metal_synchronize` status 5 OOM；
而 `recommended max working set` 警告在 119 次 `load_mode=none` 啟動裡出現 **0 次**、
在可判定的警告日誌裡是 **5/5 mmap** ⇒ mmap 撐大的是 **GPU** 工作集，可回收的是 host 頁，
這台機器撞的是後者）。

所以用第三個槓桿 **`CGC_SERVER_EXPERT_CACHE_BYTES`（池上限）**。

> ⚠️ **陷阱（本輪實際踩到）**：池大小**不在 env，在 ARGV**。
> `CGC_DUMP_ENV` 印出的是 `ARG -expert-cache` / `ARG 10737418240`。
> 而 `perjob_s02.py` 的 `BASE_ENV` 長期寫著 `CGC_SERVER_EXPERT_CACHE_BYTES=8589934592`（8 GiB），
> **`Server.start()` 裡 `resolved["env"]` 排在 `BASE_ENV` 之後會把它蓋掉，且真正的 size 來自 argv**
> ⇒ 那個 8 GiB **從來沒有生效過**，所有既有 np1 數據都是 **10 GiB**。
> 這正是本線第 N 次「設了但被靜默覆蓋」。改池必須改 argv（見 `poolsize_ab.py:set_pool()`）。

## 3. 二元判決實驗

| arm | 池 | 靜態需求 | 是否塞得下 | 觀測 RSS | 每 rep 解壓縮 | hit rate | rep1 t/s |
|---|---|---|---|---|---|---|---|
| `big` | 10 GiB | 23.27 GB | ❌ 超訂 6.89 GB | **10.66 / 10.68 GB** | 554k / 409k | **80.6%** | 5.72 / 5.66 |
| `small` | 2 GiB | 15.31 GB | ✅ **< 16 GB** | **4.54 / 5.13 GB** | **83k / 144k** | **57.0%** | 5.95 / 6.12 |

- 工具：`Backup/phase_decomp/poolsize_ab.py`；np1、reps=2（rep0 為暖機不計）、tokens=96、
  **同一 binary、未重建**、bypass 起 server、逐秒採樣 `vm.swapusage` + `vm_stat` compressor +
  該 pid 的 RSS。四臂 ABAB。
- **壓力確實被移除了**（這點要驗，不然整輪只是「改了一個數字」）：
  RSS 掉 **10.7 GB → ~4.8 GB**，每 rep 解壓縮掉 **∼4×**（554k → 83k）。

**配對結果（3 個局部估計）**：`small/big = 1.040 / 1.051 / 1.082` ⇒ **幾何平均 1.058（+5.8%）**，
3/3 同號、帶寬窄。

### 判讀

**把超訂從 6.89 GB 挪到 0（並讓解壓縮掉 4×）， decode 只快 5.8%。**

最有利的那一格也只有 **+8.2%**。⇒ 超訂是真成本，但它量級是 **6%**，不是那缺失的 9×。
**「那 15% 的有效頻寬主要是解壓縮」這個假說被否證。**

### 3.1 「那 miss 到底多了多少」—— 上面那張表漏了主詞（13:5x 補）

上面只寫了 hit rate，把 miss 的倍數藏起來了。完整數字（`final stats`，兩臂各跑兩次、計數器完全一樣）：

| | big 10 GiB | small 2 GiB | **倍數** |
|---|---|---|---|
| `requests` | 15,532 | 57,698 | **3.72×** |
| `hits` | 12,523 | 32,879 | 2.63× |
| `misses` | 3,009 | **24,819** | **8.25×** |
| hit rate | 80.6% | 57.0% | −23.6 pp |
| `file_reads` | 27,345 | 124,317 | **4.55×** |
| `fill_batch_usec` | 7.10 s | 14.55 s | 2.05× |
| `pool_wait_us` | 1.92 s | 5.16 s | 2.69× |
| **`usec_per_job`** | **259.5** | **116.3** | **0.45×（反而降）** |
| `resident` | 7990.71 MiB | 1750.34 MiB | 0.22× |

每台 server 吐 **192 token**（2 request × 96），`draft acceptance 0.43902（54/123）` 四台**逐位元相同**
⇒ 做的事完全一樣，所以 **`requests` 那 3.72× 不是工作量不同，是同一份需求被重複索取**
（步內 thrash：expert 讀進來 → 被同一步後面的 expert 擠掉 → 再要一次）。

**換算成每 token：**

- 多讀 **505 次 pread/token**、多等 `pool_wait` **+16.9 ms/token**
- 但總 `eval time` 反而 **−7.3 ms/token**（173.71 → 166.38 ms/token）
  ⇒ **別處省了 ∼24.2 ms/token**

**最硬的一條證據是 `usec_per_job` 的方向相反**：small 臂做 **4.55 倍**的 job，而
**單個 job 反而便宜 55%**（259.5 → 116.3 µs）。這不是噪音，它說的是：

> **在 16 GB 機器上放 10 GiB 池，池裡的「命中」會落在被壓縮／swap 出去的匿名頁上；
> 讀一次這種「命中」，比從 SSD 讀一次真正的 miss 還貴。**

所以正確的讀法不是「超訂只值 6%」，而是 —— **這個 80.6% 的 hit rate 本身是假的**；
它是用 memory stall 換來的。這跟 §EN-381 的結論不衝突（那裡說 prefetch 換 bytes 不划算），
因為兩者真正的共同前提是：**專家那一腿本來就便宜**（`cb 11.26 / total 122.39 ms = 9.2%）。

### 3.2 ⚠️ 但 small 池對 prefill 是明確的退步（未定案）

| arm | prompt eval（12 token） | ms/token |
|---|---|---|
| big | 4757.04 / 3549.16 ms | 396.42 / 295.76 |
| small | 7231.76 / 7363.62 ms | **602.65 / 613.63** |

**small 的 prefill 慢 1.52×。** 但這是 **12 token 的退化樣本**（固定開銷主導），不能當 prefill 指標引用。
要定性必須用 `profile_duo --profile prefill250`（250 token、交付口徑）另跑一次 —— **本輪沒跑，所以不結論。**

### 3.3 這輪絕對值的可信度

同一 config 早上 13.4 t/s、這輪 5.7–7.5 t/s —— 機器基線就已經是髒的
（`swap_used` 起點 4468 MiB、全程 5.6–7.9 GB 來自別的行程）。
⇒ **只有配對比值有效，絕對值不可引用**，而且「small 贏」有可能是「機器已經超訂」這個條件的產物。

---

## 4. 那 25 到底差在哪裡 —— 把它做成一張帳

交付形狀 anchor：`joint_capture_20260920_2144.json`（n=65 穩態，中位 `total 122.39 ms/step`）。
**25 t/s 需要 step 砍一半**（−50%）；下面是可以指出主詞的部分：

| 成分 | 實測佔 step | 來源 | 砍光的收益 |
|---|---|---|---|
| 專家 SSD 搬運 `cb` | **9.2%**（11.26 ms） | §EN-369 joint capture | +10.1% |
| **記憶體超訂／解壓縮** | **~6%** | **本輪 ABAB** | **+5.8%** |
| delta-net 的 120 個冗餘 `ggml_cpy` | ≤ **22%**（27 ms） | §EN-382，已定位 `models/delta-net-base.cpp:509-526` | +28% |
| 其餘 GPU 序列化／metadata 節點 | 剩餘 ~60–65% | §EN-382（633 CB/步、4076 節點、**40% 是 VIEW/RESHAPE/TRANSPOSE/PERMUTE，0 GPU**） | 未知，**無手段** |

前兩項已實測結案：**合計 ~15%** ⇒ 12.57 → **~14.5 t/s**。
第三項是**唯一還活著的具體槓桿**，但它是要寫 C++ 的（不是某個 env 旋鈕），
且「27 ms」目前還是 `cache` 桶的帳面差值，未經 per-op 儀器確認。
**就算三項全拿**也到不了 25（14.5 × 1.28 ≈ 18.6，且第三項不與前兩項獨立）。

其餘軸已是封版狀態（不必重跑）：
- **accept／k**：`a/m` 的 k→∞ 極限，今天 server 測到的 a 給 **18.2 t/s** ⇒ 任何 k 都到不了（§EN-379）。
- **序列並行**：np4 聚合 2.0 vs 13.78；且使用者已裁定 **25＝單流**（§EN-365/373）。
- **IO request 形狀**（合併讀）：decode 上界 **≤1%**（§EN-376）。
- **neighbour prefetch**：trace replay 驗證 model 上有效，但輸在 bytes（§EN-381）。
- **prefill 側**：根本不進每專家 LRU（3 個 prompt 長度實測 `S` 事件恆為 5976），與本題無關。

## 5. 附：一個儀器缺口（比結論本身更容易咬人）

`perjob_s02.py::free_pct()` 報的是 `vm_stat` 的 free pages ⇒ **完全看不見 swap 與 compressor**。
本輪起點 `[env] free=82.0%` 看起來乾淨，同刻 `vm.swapusage` 其實已經是 **used 4468 MiB**；
完整第一輪結束時系統 swap 變成 **total 7168 / used 5620 MiB**。

⇒ **本輪絕對值不可引用**（同一 config 早上是 13.4 t/s、本輪 5.7 t/s，差 2.3×），
只有配對比值有意義。**往後每台 server 類 driver 都該把 swap 與 compressor 一起記**
（`MemSampler` 已可複用）。這也意味本 repo 今日所有 t/s 絕對值都帶同一個未知偏差。

## 6. 建議

1. **目標維持 14–16 封版**，不可用本輪任何數字往上修。
2. 若要再往前，唯一有主詞的一步：**先跑 per-op 真實時間**
   （`CGC_CB_N_MAIN=1` ＋ 加大 `CGC_SERVER_N_CB`，見 §EN-382）確認那 120 個 CPY 到底值多少，
   再決定要不要動 `delta-net-base.cpp`。**不要先寫碼。**
3. 超訂這條路結案，不要再回來；但它有一個產品側用途：
   **池可以縮到 2 GiB 而不掉速（甚至 +6%）**，省 6 GB 常駐 —— 如果日後要做多實例／並行
   產品線，這 6 GB 是可以投資的地方。
```
