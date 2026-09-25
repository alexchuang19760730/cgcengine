# Column census：fill 住在 `cb`（6.7%），而 `wait` 佔 88.7%

**日期** 2026-09-25 22:07 · 載體 `prod-new`（**分段臂**、MTP off、8 GiB 池、`-p 2048 -n 128 -d 512 -r 3 -b 5632 --warm-skip 64 --fixed-fill-seed 1`）· 產物 `Backup/eseries/E2c/`（`summary.json`、`census.json`、`*.stderr.log`、`E2c_misses.txt`）

## 0. 一句話

在**同一個 arm**上把三支儀器同框：decode 步的中位總時 **75.09 ms** ＝ `wait` **66.58（88.7%）** ＋ `cb` **5.03（6.7%）** ＋ `submit` **3.40（4.5%）**（三欄相加與 header 自報值差 **−0.09 ms**，內部一致）；而同臂的 `CGC-EBTIMER` decode 步中位 **4.32 ms**、每步 miss 中位 **5** ⇒ **ratio = 0.86 ⇒ FILL_IN_CB**。⇒ **兩個儀器不再矛盾**：`EBTIMER` 就是 `cb` 裡的那一段（**86% 的 hook 是 fill**），而 **20.8 ms/step 只是「每步 30 miss」那個較重的負載**，不是穩定步成本。⇒ **形狀 1 死了**：可移除的那塊是 **6.7%**（每步 4.3 ms），而佔 88.7% 的 `wait` 不是 miss 的函數（E2b：rho = −0.32）。

## 1. 為什麼量這個

E2b 之後桌上剩下的唯一分裂：`CGC-EBTIMER` 裁決的 fill ＝ **20.8 ms/步**，而同一家族的 `CGC-DECPROF` 逐層 `cb` 只看到 **1.86–6.3 ms/步**，差 ~3–10×。兩者不可能同時描述同一個 decode 步，而在裁決前**兩個方向的結論都不能引用**：

- 若 fill 真的 20.8 ms（43% of 48.2 ms）⇒ 形狀 1 的「48.2＋20.8」有意義，值得為它做工程；
- 若 fill 只有 ~4 ms（6%）⇒ 形狀 1 的乘法沒有東西可乘，`+28%` 不成立。

## 2. 判準（**跑之前**凍結）

`ratio = EBTIMER step_usec 中位（decode 層） / 同一批 decode 步的 Σcb 中位`

| 判定 | 條件 | 意義 |
|---|---|---|
| **FILL_IN_CB** | `0.5 ≤ ratio ≤ 2.0` | fill 就在 `cb` 裡 ⇒ 可移除份額 = `cb` 的份額 |
| **FILL_UNACCOUNTED** | `ratio > 2.0` | EBTIMER 量到沒有任何欄位承接的工作 ⇒ 儀器缺口 |
| **FILL_UNDER_CB** | `ratio < 0.5` | hook 裡有不屬於 ensure_batch 的工 ⇒ `cb` 也不是 fill 的載體 |
| **INVALID** | 無 EBTIMER 行／無 decode 層區塊／無共同步 | 答不了（**不是**通過） |

層別（stratum）：用 EBTIMER 自帶的 `n_sum`（每步 demand 數；decode ≈ 40 層 × 19 ≈ 752）分 decode／prefill chunk。**只看 decode**，因為上面每一句主張都是 decode 的。

## 3. 先修掉的一個 instrument 缺口（0 GPU）

`CGC_EB_TIMER` **不在 `run_server.sh` 的 `SERVER_ENV` 白名單裡**（`grep` 0 命中）⇒ 經 bench／server 路徑設它會被**靜默丟掉**（引擎自 2026-09-25 就讀它）。這正是該檔自己記錄過的同一類陷阱，也是為什麼 `FILL_COST_MEASURED` 那份只能靠繞過 launcher 的路徑產生。已補一個 block（同 `LLAMA_EXPERT_CACHE_BATCH_DBG` 的樣式），並用 `CGC_DUMP_ENV=1` 乾跑驗證 `ENV CGC_EB_TIMER=1` 確實出現在解析結果裡（1 命中）。

## 4. 讀數（同一 arm）

臂：pp 294.99 ± 5.70、tg **11.62 ± 0.20**（hit 96.3%）；385 個 EBTIMER 步、53 個 DECPROF 區塊（其中 **48 個 decode**）；thermal launch NOMINAL、worst MODERATE（5/149 = 3%）、swap 7709 → 8638（**growth +929**，門檻 500）、min_free 53 MiB。

| 欄 | 每步中位 | 佔步時 |
|---|---:|---:|
| `wait`（輪詢該段 GPU 完成） | **66.58 ms** | **88.7%** |
| `cb`（top-k hook，`ensure_batch` 在其內） | **5.03 ms** | **6.7%** |
| `submit`（排下一段） | 3.40 ms | 4.5% |
| **header 自報 total** | **75.09 ms** | 100% |
| 三欄相加 | 74.995 ms | **Δ = −0.09 ms（一致）** |

| EBTIMER（decode 層） | 值 |
|---|---:|
| `step_usec` 中位 | **4.32 ms** |
| 每步 `miss` 中位 | **5** |
| ratio vs Σcb | **0.86** ⇒ **FILL_IN_CB** |
| 前綴／prefill 步 | **0**（EBTIMER 只印到 decode ⇒ 前綴走 slab 路徑、不呼叫 `ensure_batch`） |

## 5. 裁決：那個 10× 是負載差，不是矛盾

| 來源 | 每步 miss | fill/步 | **每 miss** |
|---|---:|---:|---:|
| `FILL_COST_MEASURED §3`（EBTIMER，server 載體） | 30 | 20.8 ms | **0.69 ms** |
| 本輪（EBTIMER，bench 載體） | 5 | 4.32 ms | **0.86 ms** |

⇒ 兩者的**每 miss 成本在同一個帶（0.69–0.86 ms）**，差的 5× 全部來自 miss 數。**20.8 ms/step 是重負載下的數字，不是 fill 的固有成本**；把它當成「單段化之後會多付的固定成本」是錯的。

## 6. 對形狀 1 的結論

1. **可移除的份額是 `cb` 的 6.7%（4.3 ms/步），不是 43%（20.8 ms）。** 拿掉它：75.09 → 70.8 ms ⇒ tg 11.62 → ~12.3（+6%）。這就是形狀 1 在**當前 miss 負載**下的全部空間。
2. **重負載下有上限**：若 miss 回到 30/步，fill ≈ 21 ms、`cb` ≈ 6.7%→~25% ⇒ 即使全部移除也只是 +25%，而那個 regime 本身是池太冷造成的（不是 fill 效率問題）。
3. **佔 88.7% 的 `wait` 沒有被任何一欄解釋掉，且它不是 miss 的函數**（E2b 的 rho = −0.32、逐層中位 1.32–1.92 ms 近乎均勻）⇒ **它是 round-trip 結構**（41 段：等整段完成 → 發 hook → 提交下一段），**減 miss 動不了它**。
4. ⇒ **「S1 ＋ 同步等 fill」這條便宜路徑死了**：它想省的（fill）只值 6.7%，而它動不了的（wait）值 88.7%。要拿那 88.7%，只有**重疊**（形狀 2）或**減少段數**——而段數由「每層的 ids 必須在該層 gather 之前知道」決定（`CGC_SEG_BATCH=1` 在 41 段迴圈前就 return ⇒ 那段路徑給出 NaN，E0）。
5. 附帶收緊一格：**hook 幾乎就是 fill**（EBTIMER/Σcb = 86%）⇒ `cb` 裡沒有第二塊可回收的非 fill 開銷。

## 7. 限制

- 盒況不合格（swap growth +929、min_free 53 MiB、worst MODERATE）⇒ 依 `MEASUREMENT_CONTRACT §5.1` 這是**診斷臂**，不對外宣稱 t/s（tg 11.62 只當「與 11.2–11.6 同族」的旁證）。但本判決用的是**同 arm 內的比例**（ratio、份額），對 swap 灌大不敏感。
- EBTIMER 只涵蓋 `ensure_batch` 那條路徑（E2b 已量到另有約 46% 的 miss 走 `ensure_slot`）⇒ 「fill 就在 `cb` 裡」是就這條路徑而言；`ensure_slot` 那半的成本同樣落在 `cb`（同一 hook），只是不在 EBTIMER 的計時器內 ⇒ 真實 `cb` 裡的 fill 佔比比 86% 更低。**這使結論方向不變**（fill 份額的**上界**仍是 `cb` = 6.7%）。
- 兩個儀器的步不共用索引 ⇒ 本判決是**分佈對分佈**（同層別中位），不是逐步配對。

## 8. 同一批 log 裡的裝置側（順手記下，不當判決）

header 自報（同 48 個 decode 步的中位）：**`gpu_sum` 97.49、`union_sum` 69.95、`gap_sum` 14.6 ms**。⚠ 這三個是**該步 41 段的總和**，不是 `wait` 的分解——`gpu_sum` 大於 `wait` 正是分段重疊存在的證據。可配對的只有裝置跨度內部：**gap ＝ 17% of (union+gap)**，即裝置時間有 17% 是命令之間的空隙。作為對比，本線 09-23 在另一批 log 量到 gap 佔裝置跨度的 31.9%（`union 135.9 / gap 69.1`）——兩者不可直接比（不同臂、不同盒況），但方向一致：**空隙存在、而且不是 0**，所以形狀 2（重疊）至少有 17% 量級的東西可以拿。這一格要當成結論仍需一次有控制的量測（同臂 × 開關 gap 的可消除部分），本輪不把它寫成判決。

## 9. 產物

| 檔案 | 內容 |
|---|---|
| `Backup/eseries/E2c/{summary.json,*.stderr.log}` | 臂讀數、53 個 DECPROF 區塊、385 個 EBTIMER 步、engine／cell／box_gate |
| `Backup/eseries/E2c/census.json` | 份額、ratio、逐項判定、層別計數 |
| `Backup/eseries/E2c_misses.txt` | 逐筆 miss（layer expert） |
| `scripts/check/column_census.py` | 分析器＋凍結門檻＋10 項自測（含「無 EBTIMER 必須 INVALID」） |
| `scripts/run_server.sh` | 新增 `CGC_EB_TIMER` 白名單 block |
