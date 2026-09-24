# CGC Engine 16GB 設備優化 — 開發過程與成果白皮書

> 日期：2026-09-24 ｜ 目標平台：16GB MacBook Air M4 ｜ 模型：Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X（13.6GB / 40 trunk + 1 MTP head / 256 experts / 8 active）
> 長期目標：
>
> **prefill 250+ / decode 25 t/s**
>
>  ｜ 真實錨點：12.57–13.99 t/s（環境變動）｜ 本週末乾淨基線：10.73 t/s



***

## 0. 摘要

在 16GB 統一記憶體設備上運行 35B-A3B MoE 模型，核心矛盾是**結構性超訂**：模型權重 13.6GB（全量觸碰即駐留）+ expert pool 8GB = 21.6GB 潛在駐留需求，塞進 16GB 物理記憶體，macOS swap 只累積不回收 → 每次量測環境越跑越髒、結果不可比。

本週的開發把這個問題從「kernel 不夠快」的誤判，一步步推進到「記憶體架構」的正解，產出三項可複用成果：



| 成果                   | 機制                                             | 實測                                                  |
| -------------------- | ---------------------------------------------- | --------------------------------------------------- |
| **P0（skip-readraw）** | expert 權重不再全量讀進 CPU RAM，只從 pool pread          | **swap ↓92%**（可引用）；同環境 decode **+3.7%**；prefill **250 已達標**（見 §3.1 修訂）     |
| **B 真臂（SPAC\_HOT）**  | 踢「累計路由次數」最低者（永不衰減），熱門重尾常駐                      | hit 96.3%；HTTP ABBA **+4.3%**、llama-bench **+3.7%** |
| **量測基建**             | ABBA 交錯協議、窗口守門、budget 預檢、build 指紋、commit gates | 所有 A/B 從「猜測」變成「可判決」                                 |



***

## 1. 問題診斷演進（過程）

### 1.1 第一階段誤判：kernel 不夠快

最初把 decode 慢歸咎於 MoE kernel 效率（頻寬利用率 25%）。量測後發現：**不是 kernel 單點問題**，而是整條 decode 路徑的串行化。

### 1.2 第二階段誤判：capacity miss

改進 expert cache 容量（LAYER\_CAPS / PIN\_PROFILE），實測 **PIN\_PROFILE −23%**（判死）—— 踢錯專家的代價遠大於省下的容量。

### 1.3 根因鎖定：結構性超訂 → swap 不可追蹤



```
模型檔案 13.6GB 全量 mmap/read（觸碰即駐留） + pool 固定 8GB = 21.6GB 潛在駐留

塞進 16GB → macOS swap 只累積、不回收 → 每次量測 swap 漲一點（4668→6170 MiB 證據）

→ 所有 A/B 在「越來越髒」的環境下跑 → 結果永遠不可比
```

**這才是「量測不到效果」的真正原因**—— 不是儀器不夠準，是環境在漂。



***

## 2. 方法論（第一性原理）

### 2.1 先推導、後量測、再實作

本週最大教訓（用戶反覆強調並被證實）：**能從原理推出的，不要靠量測猜**。



* prebind 失敗的根因（draft\_prefetch\_ids 在 MTP 架構下從未被填）——**讀代碼就知道，不需要跑實驗**

* "Dense 屋頂 47.6 GiB/s"——**從 DRAM 硬體特性推（row activate /turnaround/refresh 15-20% overhead）**

### 2.2 量測基建（可複用資產）



| 組件                     | 作用                                                                |
| ---------------------- | ----------------------------------------------------------------- |
| **ABBA 交錯協議**          | A B A B A B + 深冷卻 + 配對 per-rep 比率中位 → 消除 launch-to-launch ±20% 漂移 |
| **窗口守門（window\_gate）** | 別的 llama-server 在跑就拒跑（乾淨窗口是等出來的，不是排出來的）                           |
| **budget 預檢**          | 超訂 4838 MiB 直接拒跑（`CGC_SERVER_STRICT_BUDGET=1`），避免髒量測              |
| **build 指紋**           | server/libllama-server-impl dylib 的 md5 隨臂記錄，抓「換了 binary 不知道」     |
| **provenance gate**    | 量測產物必須帶 engine digest + 窗口 block（出處契約）                            |
| **commit gates**       | 源碼↔binary 同步、死鎖防護、量測出處、replay 不退化                                 |

### 2.3 已判死方向（禁止重試，節省時間）



| 方向                  | 判死原因                                                                            |
| ------------------- | ------------------------------------------------------------------------------- |
| A/prebind（預指派 slot） | h=0.003（<0.65 門檻），draft ids 在 MTP 下是空 buffer                                    |
| PIN\_PROFILE（固定釘選）  | −23%（踢錯專家代價 > 省容量）                                                              |
| ρ 按層批次化 prefetch    | −47.6%（bg 干涉）                                                                   |
| verify 批次化（宿主路徑）    | dispatch 邊際僅～1ms/token（2-3%）—— 不是槓桿                                             |
| L1 pool 8G→4G（生產配置） | 錯藥：超訂本質是 21.6GB 潛在駐留，砍 pool 省 4GB、付 21% 速度代價（0.79 倍速；並行實驗 0.79×2>1 的效率論點不適用於生產） |



***

## 3. 關鍵成果（數據）

### 3.1 swap 結構性修復（L1-L4 + P0）



```
L1: pool = f(物理記憶體預算)   → 8GB 是「miss 上界」不是需求（resident 6.2GB 就 hit 96.3%）

L2: expert 權重單一駐留        → skip-readraw：權重不預讀 RAM，只從 pool pread（P0）

L3: swap 進 autotuner 目標     → 自適應閉環

L4: pool 駐留納入 wired 管理   → 防「省記憶體反製造 swap」
```

**P0 實測（ABBA + llama-bench）**：



* swap：**5679 → 449 MiB（−92%）**（同 run 對照）

* decode：ctrl 10.73 → p0+B **11.13（+3.7%）**

* prefill：ctrl 162.5 → p0+B **209.5（+29%）**
  * ⚠ **修訂 2026-09-24 夜（作廢此行的結論）**：這個 162.5 是降級視窗的產物，不是 ctrl 的性質。
    同一支臂在同一視窗十分鐘內四次 launch = **275.39 / 196.72 / 207.33 / 204.98**
    （同臂散佈 78.7 t/s = 中位 38%），而那 ±81.7 就是這個。
    **prefill 250+ 另已在 9 次 launch 上達成**（最高 296.24；完整表與來源：
    `docs/PREFILL250_MET_AND_S1_REJUDGE_2026-09-24.md` §1）。
    ⇒ 本行**不可用作 ctrl 臂、也不可作為交付值**；可引用的 prefill 必須連視窗區塊與 build digest 一起報。

* hit：96.3%（misses 4777–4944，pool cap 僅 16%）

### 3.2 B 真臂（CGC\_SPAC\_HOT）



```
機制：踢「累計路由次數」最低者（永不衰減），tie-break EMA→LRU

動機：SpAc EMA 把「熱而靜」的專家衰減到零 → 58.6% 覆蓋 vs 路由重尾 98.6% 的錯配

默認 off，位元相容
```

**實測**：



* HTTP ABBA（swap 極髒，n=2）：**+4.3%（配對中位，兩對皆正）**

* llama-bench（今天同環境）：**+3.7%**（11.13 vs 10.73）

* hit 96.2-96.3%—— 與靜態直方圖上界 98.8% 接近

### 3.3 verify 通道分解（第二線成果）



```
decode step 161ms 分解：

&#x20; union（GPU 忙）     113.4 ms  71.3%  ← 真正在算

&#x20; gap（GPU 空）        44.4 ms  27.9%  ← GPU 等 CPU（=cb+submit，r=0.957，斜率 1.05）

&#x20; wait（CPU 等 GPU）  123.2 ms  78.1%

&#x20; cb（top-k hook）     23.8 ms  15.0%

&#x20; submit（派送）       10.8 ms   7.0%

驗證 token 邊際分解：union 56-61% / cb 35-40% / submit 2-3%
```

**結論**：GPU 空轉 = CPU 的 hook＋submit（1:1 因果）；「batch verify」不是槓桿（dispatch 邊際～1ms）；真正的槓桿在 union 的形狀與 cb 的 miss regime。

### 3.4 prompt cache /prefix reuse（freebuff 線）



* M1（logits 逐位元）**9/9 PASS**，dump md5 與參考逐位元相同

* 重用：208 → 4 tokens（96% 稅拿回），forcing full re-process 0 行

* 記憶體減半：3 checkpoint 256MiB → 1 checkpoint 130MiB

* 關鍵修法：`do_checkpoint_split = do_checkpoint && !cgc_prefix_reuse_ckpt`（斷批點中性化）



***

## 4. 決策記錄（含失敗，供復盤）



| 時間 | 決策                     | 結果                                                  |
| -- | ---------------------- | --------------------------------------------------- |
| D1 | 先寫 batch verify（宿主批次化） | **判死**：dispatch 邊際 1ms（2-3%），先量測後實作的原則              |
| D2 | A/prebind（預指派 slot）    | **判死**：h=0.003——draft ids buffer 在 MTP 下是空的（讀代碼可預知） |
| D3 | PIN\_PROFILE           | **判死**：−23%                                         |
| D4 | P0 skip-readraw        | **成功**：swap 92%↓，零代價（同環境 +3.7%）                     |
| D5 | B 真臂 SPAC\_HOT         | **成功**：+4.3%/+3.7%，hit 96.3%                        |
| D6 | 重開機後完整三臂               | 環境漂移 −19%/-42%（昨天 13.25→今天 10.73）——ABBA 設計正是為此      |

**失敗復盤（白皮書準則）**：



1. **能量測前先讀代碼**（prebind 的 draft ids 空 buffer—— 第一性原理可預知）

2. **先量測、後實作**（verify 批次化 —— 先量出 dispatch 邊際 1ms 就不用寫）

3. **環境漂移是最大的敵人**（swap 不可追蹤 → 所有 A/B 無效）—— 修環境先於修 kernel



***

## 5. 現狀與下一步

### 5.1 現狀（2026-09-24 基線，今天環境）



| 配置             | decode           | prefill          |
| -------------- | ---------------- | ---------------- |
| ctrl（prod-new） | 10.73 ± 0.53     | 162.5 ± 6.4      |
| **p0+B（生產候選）** | **11.13 ± 0.49** | **209.5 ± 81.7** |

> ⚠ **修訂（2026-09-24 夜）**：**prefill 欄的兩列都不可引用 —— 目標已達標，這張表把視窗讀成了缺口。**
> 同 cell（`-p 2048`，prod-new 口徑）**≥ 250 t/s 的 launch 共 9 次**（最高 296.24；
> 283.01 那次是乾淨視窗、150/150 NOMINAL，見 `docs/PROD_NEW_MTP_OFF_2026-09-23.md`）。
> `209.5 ± 81.7` 的散佈是**同臂十分鐘內 275.39 → 196.72** 造成的，不是程式碼。
> **decode 欄不變**（11.13 仍是同環境配對值）。
> 來源：`docs/PREFILL250_MET_AND_S1_REJUDGE_2026-09-24.md` §1。

### 5.2 剩餘槓桿（按優先級）



1. **union 形狀**（verify 圖 GPU 時間 56-61%）——NSG / 形狀軸（P0 那條）

2. **cb 的 miss regime**（35-40%）—— 覆蓋率 / 路由局部性（B 已吃掉一部分）

3. **round remainder / 段間 gap**（GPU 空轉 44ms）—— 調度重排（S1 組合）

### 5.3 通往 25 t/s 的路徑（估算）



```
今天基線 10.73（ctrl）

\+ P0/B（已實測）        → 11.13（+3.7%）

\+ union 形狀優化         → \~14-15（+25-30%，NSG/shape 軸）

\+ 段間 gap 重疊          → \~16-17（+15%，CPU/GPU 重疊）

\+ 頻寬打滿（60-70GB/s）  → \~20+（硬體上界附近）
```

**25 t/s 需要上述全部 + 更積極的量化 / 形狀策略**—— 是架構級目標，不是單點優化。

> ⚠ **修訂（2026-09-24 夜）**：上面這條階梯的兩個問題。
> (1) **「union 形狀」與「段間 gap 重疊」是同一個改動（單段提交 S1）**，把它們列成兩個可加的
> 步驟會重複計數；(2) 每一步的百分比取自不同視窗（單臂散佈 ±27%、啟動漂移 ±20%），相乘等於乘噪音；
> 終點「20+」已超過本線自己發表的兩個上界（20.49 / 18.59）。
> 而那個被引用的 **2.2× 已在產物裡重判為 1.72×，且兩臂 I/O 結構不同**
> （A `file_reads` 82,293 vs B 0）⇒ 不可歸因為 shape 收益。
> ⇒ 25 的下一步不是接更多機制，而是量**單段提交那條路上 verify 第 2–4 個 token 的邊際成本**
> （以 49.2 ms/token 結算，25 t/s 只需 `mean_len` 1.23）。
> 來源：`docs/PREFILL250_MET_AND_S1_REJUDGE_2026-09-24.md` §1–§2、§5。



***

## 6. 附錄

### 6.1 commit 鏈（本週成果入庫）



| commit      | 內容                                            |
| ----------- | --------------------------------------------- |
| `41536c9e2` | B 真臂 SPAC\_HOT + ABBA doc（+4.3% 標題帶成績）        |
| `f2effe569` | P0 skip-readraw + 各線實驗收尾（今天 llama-bench 成績標題） |

### 6.2 關鍵 docs 索引



| doc                                           | 內容                                 |
| --------------------------------------------- | ---------------------------------- |
| `docs/B_ARM_ABBA_2026-09-24.md`               | B 真臂全量 ABBA 判決表                    |
| `docs/SWAP_STRUCTURAL_FIX_2026-09-24.md`      | L1-L4 swap 結構性修復決策表                |
| `docs/REPLACEMENT_POLICY_75PCT_2026-09-23.md` | 路由重尾 98.8% / PIN\_PROFILE 89.4% 實測 |
| `docs/H_MEASURED_2026-09-24.md`               | h=0.373 判死 A/prebind               |
| `docs/VERIFY_TOKEN_CHANNELS_2026-09-23.md`    | verify 通道分解（union/cb/submit）       |
| `docs/GAP_FIX_WHITEPAPER_2026-09-24.md`       | gap 歸因與修復方案                        |
| `docs/T5_INTRA_NP_CONCURRENCY_2026-09-23.md`  | prefix reuse M1 9/9 全記錄            |
| `docs/NEXT_ACTIONS_2026-09-24.md`             | 執行優先級（swap 軸）                      |

### 6.3 復現



```
\# p0+B llama-bench 完整側

python3 scripts/check/llama\_bench\_matrix.py \\

&#x20; \--arms "prod-new:CGC\_EXPERT\_SKIP\_READRAW=1;CGC\_SPAC\_HOT=1" \\

&#x20; \--prompt 2048 --gen 128 --depths 512 --reps 3 --ctx-size 0 \\

&#x20; \--workdir /tmp/commit\_bench\_p0hot --json /tmp/commit\_bench\_p0hot/result.json

\# ABBA 三臂（重開機後）

RUN\_GROUP=all PAIRS=3 COOL\_S=300 bash scripts/check/abba\_p0\_3arm.sh
```