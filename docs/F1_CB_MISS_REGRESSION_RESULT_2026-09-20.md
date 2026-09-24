# F1 結果：`cb` 是 per-miss 還是 per-round-trip？—— 判定、以及它對目標的修正

日期：2026-09-20　載體：Nail IQ3_XXS-denseIQ4X、MTP on k=3、pool 8 GiB、prefill250、temp 0.4、chat door、qa-zh、`--n-predict 96`
產物：`Backup/cgc_logs/llama_server_20260920_114450.log`、`Backup/cgc_logs/f1_cb_miss_regression_report.json`
工具：`scripts/check/cb_miss_regression.py`（`--selftest` 10/10 PASS）
前置條件（書面、跑之前寫的）：`Backup/phase_decomp/F1_CB_MISS_REGRESSION_20260920.conditions.md`

---

## 0. 一句話

**預先註冊的規則判 LINEAR（bandwidth bound），而且全集合與剔除 null-check 樣本兩個讀法都是 LINEAR。**
**但它的物理內涵與我先前的框架相反**：邊際 miss 成本 **0.695 ms**，而同一顆 miss 的 bytes（3 × 0.37 MiB）**用這台磁碟自己的峰值 823 MB/s 去讀，需要 1.42 ms** —— 觀測值只有它的一半。
⇒ 邊際 miss **不是在付磁碟的傳輸時間**，而是在 ~1.6 GB/s（≈ 記憶體／page-cache 拷貝速度）上被服務。所以「bandwidth bound」對（成本 ∝ bytes），但「**disk bound 是錯的**」，而 §4 的「fatter read」在這一輪**沒有空間**。

同時拆出一條先前沒有名字的項：**每層 barrier 12.86 ms/step（cb 的 19%）**，只要該層有**一個** miss 就要付。實測對照：m=0 → 0.01 ms（n=1251）；m=1 → **1.15 ms**（n=865）。

---

## 1. 讀數

窗口：reclaimable **8911 MB**（≥ 文件化地板 8000，**未動任何地板**）、零 rival llama。
配對樣本：**3921 個 (step, layer)**，跨 **216 個 step**（皆有 per-layer 列）。

### 1.1 per-m（`m` = 該層該步的 cache miss 數）

| m | n | cb 中位 | cb 平均 |
|---|---|---|---|
| **0** | **1251** | **0.01** | 0.01 |
| 1 | 865 | 1.15 | 1.23 |
| 2 | 625 | 1.97 | 2.04 |
| 3 | 366 | 2.77 | 2.94 |
| 4 | 226 | 3.33 | 3.52 |
| 5 | 160 | 3.96 | 4.31 |
| 6 | 123 | 4.56 | 4.95 |
| 8 | 63 | 5.17 | 5.75 |
| 10 | 28 | 6.25 | 6.93 |
| 13 | 13 | 9.27 | 10.08 |

**m=0 的層幾乎不花時間（0.01 ms）—— 沒有任何 per-layer 固定成本。**
而 **m=1 → m=2 是 +0.82 ms，不是 0**：所以「6 個 job 塞進 8 個 worker 應該免費」這個 round-trip 圖像**在小 m 上不成立**。

### 1.2 判定（預先註冊的規則）

```
VERDICT (full set, n=3921)         : LINEAR
  med(m=2)/med(m=1)=1.71 >= 1.6  and  med(m=4)/med(m=2)=1.69 >= 1.6  => bandwidth bound
VERDICT (排除 13 個 null-check 樣本): LINEAR   (同上)
```

### 1.3 post-hoc 模型比較（**未**預先註冊，不得當成判定讀）

對群體中位數做最小平方：

| 模型 | 形式 | R² |
|---|---|---|
| per-miss（線性） | cb = 0.46 + 0.695·m | 0.8617 |
| per-round-trip | cb = −0.70 + 1.934·ceil(3m/8) | **0.9313** |

round-trip 形式配得比較好，但**這是 post-hoc**，而且兩者都高（m≥14 的尾端只有 1–6 個樣本、抖動大）。誠實的述句是：**規則判 LINEAR；形狀更接近「每個 job 都要付一次」而不是「每層一個 round-trip」。**

---

## 2. 可動手的那一半：step 級分解

用 per-layer 擬合 `cb_layer = 0.46 + 0.695·m` 對整個 step 求和，與實測 `cb_total` 對帳（ntok=4、有 miss 的 72 個 step）：

| 量 | 值 |
|---|---|
| 實測 cb_total/step（平均） | **67.00 ms**（中位 61.16） |
| 模型預測 | **64.17 ms**（= 96%） |
| 有 miss 的層數（平均） | **27.96 / 40** |
| miss 總數（平均） | **73.83** |
| 每個受影響層的 miss 數 | 2.64 |

分解：

| 項 | 算式 | ms/step | 佔 cb |
|---|---|---|---|
| **每層 barrier** | 0.46 × 27.96 | **12.86** | **19%** |
| **每個 miss** | 0.695 × 73.83 | **51.31** | **77%** |

⇒ **兩條不同的槓桿，而且它們的回應不同**：
- barrier 項 ∝ **層數**，與池子大小、與量化**無關**；
- per-miss 項 ∝ **M**，這才是池子/目標化/量化在動的量。

### 2.1 每 miss 0.695 ms 是什麼

一顆 miss = 3 個 segment（三顆投影）。09-19 的稽核行給 segment 大小 `0.37 MiB/job`（該臂 2.22 GiB / 6150 jobs）⇒ 一顆 miss ≈ **1.11 MiB**。

| 假設 | 一顆 miss 的 bytes 該付多少 |
|---|---|
| 磁碟峰值 823 MB/s | **1.42 ms** |
| 09-19 實測有效率 235 MiB/s | 4.7 ms |
| **本輪邊際實測** | **0.695 ms** |

⇒ 邊際 miss 的 bytes 以 **≈1.6 GB/s** 流動 = **page cache / RAM 拷貝速度**。**這一輪的邊際 miss 不是從磁碟讀的。**

**⚠️ 這一步是推論，不是本輪量測**：本輪 server 被 SIGKILL 收掉，**沒有** `read shape` / `final stats` 行，所以沒有本輪自己的有效讀取率。`0.37 MiB/job` 與 235 MiB/s 來自 09-19 那一臂。見 §5 的缺口。

---

## 3. 二級檢查

| 檢查 | 結果 | 判讀 |
|---|---|---|
| null check（cb<0.05 卻有 m>0） | **13 / 3921 = 0.33%**，集中在 3 / 216 個 step（310, 367, 383） | 我預先註冊寫「有違反就丟棄」。**我沒有丟棄**，理由寫在下面 —— 但兩個讀法的判定都是 LINEAR，結論不依賴這個決定。 |
| 排序不變式 | **216 / 216** 個 step 行的下一行都是同一發射區塊（`top1`），`all:` 列緊跟其後 | 配對規則本身經獨立驗證，違反不是 parser 錯接 |
| path check（cb≥1 ms 卻無 BATCHDBG） | **0** | **沒有證據支持 `ensure_slot` 那條序列路徑在分擔 cb**；見下方份額稽核 |
| 本輪 HOOKSPLIT | `ensure` 2679.6 / 總 3016.1 µs/call = **88.8%**；`pre` 333.9 µs/call | ensure 仍主導；但 `pre` 比 09-19 那臂（4.1 µs）大 80×，與本輪的高 swap（10.1/11.2 GB）一致 —— **本輪的絕對值帶儀器與壓力標籤** |

**為什麼我沒有照字面丟棄**（明示的偏離）：註冊條款寫的**理由**是「A violation means the pairing is wrong」。排序不變式已在全部 216 個 step 上獨立驗證成立，所以配對規則不是成因；13 個樣本是引擎裡一個真實（罕見）的狀態：**該層確實有 miss 被指名，但 demand 幾乎沒等**（最像 prefetch 已在飛、demand 直接接手）。我把那 13 個樣本排除後重跑判定（結果同為 LINEAR，且 med(m=1) 由 1.15 → 1.15 不變），而不是丟掉 3921 個樣本換取一個資訊量更低的答案。**偏離與理由都在此，讀者可自行套用原條款重判。**

### 3.1 cb 的來源份額稽核（把 path check 從「0 個嫌疑犯」推到「份額是多少」）

把**所有** step 的所有 per-layer cb 加總，按「該層在該 step 有沒有 BATCHDBG 列」分兩堆：

| 來源 | cb 合計（ms） | 佔比 |
|---|---|---|
| 有 miss 列的層 | **7871.9** | **99.8%** |
| 沒有 miss 列的層 | 17.7 | 0.2% |

而「沒有 miss 列卻 cb ≥ 0.05 ms」只有 **25 筆、合計 1.7 ms、最大 0.100 ms**。
⇒ **cb 是「有 miss 的層」的性質，而且只走 `ensure_batch`**。conditions doc 裡我提的「MTP fast path 的 `ensure_slot`（每 expert 序列 3 thread）可能是主因」**被否證**。

---

## 4. 對目標的修正

| 先前的目標 | 現在的狀態 |
|---|---|
| **F2（overlap）** | **不變，而且仍然成立**：74 → 0 ms 的算術不動（step 173.8 ms ⇒ 19.4 t/s at mean_len 3.375）。而且 §2.1 讓它更有利：邊際成本是 **RAM 拷貝 + 每次 job 的固定成本**，而 `cb` 是**等待**，本來就可被遮。 |
| **F3（fatter read）** | **沒有空間，作廢**。per-job 0.23 ms 已經在拷貝速度上；把 3 jobs/miss 合併成更胖的讀不會更快。原本的否證條件（`MiB/job` 升而 `us/job` 不變）現在可以直接宣告成立。 |
| **新增 F5：每層 barrier** | **12.86 ms/step（cb 的 19%、步時的 ~6%）**，且**一個 miss 就要付整筆**。實測對照：m=0 → 0.01 ms（n=1251）；m=1 → 1.15 ms（n=865）。 |
| 池子 / 量化 / 目標化 | **∝ 51.31 ms/step（cb 的 77%）**，回應的是 M（平均 73.8 顆/step），不是 bytes 的速率。 |

### F5 —— 可否證的單項目標

**命題**：`cb` 有 0.46 ms/層 的固定項，只由「該層有沒有 ≥1 個 miss」決定，與 miss 數無關。

- **通過**：m=1 的層，cb 中位 **≤0.5 ms**。
- **否證**：m=1 的層維持 ≈1.15 ms（本輪 n=865，中位與平均 1.15/1.23 幾乎相同 ⇒ 不是尾端造成的）。此時 barrier 不是「有 miss 的層」的性質，就該改問別的東西（例如 miss 命名的 slot 是否剛好在 eviction 路徑上）。
- **量級**：barrier 佔 cb 的 **19%** ⇒ 拿掉它 = step −11.4 ～ −12.9 ms ⇒ **19.3–19.4 t/s**（step 隱含 18.18 t/s）⇒ **約 +6%**。單獨不足 25，但它與池子/目標化是**可加**的兩項（19% + 77%）。

### F2 的天花板（更新，用本輪自己的數）

本輪（有儀器、高 swap）。**只用同一族群的配對中位**（driver 自己的 ntok=4 穩態 74 個 step）：

| 量 | 值 |
|---|---|
| step 中位 | 204.30 ms（= wait 139.09 + cb 60.22 + submit 5.27 = 204.58） |
| cb 中位 | **60.22 ms** |
| mean_len | 3.715（accept 81.86%） |
| 交付 | 15.64 t/s |

把 cb 全拿掉 ⇒ 步時 ≈ **204.30 − 60.22 = 144.08 ms** ⇒ mean_len 3.715 / 0.14408 = **25.8 t/s**。
改用 §2 的逐層族群（72 個有 miss 的 ntok=4 step，cb 中位 61.16）⇒ 143.1 ms ⇒ 26.0 t/s。

**所以在本輪的形狀下，`cb` 單獨就是 25 t/s 的門檻**（25 需要 ≤148.6 ms）。先前「74 ms 拿掉只到 19.4 t/s」的差別全部來自 mean_len（3.375 → 3.715）——再一次說明 `25 = mean_len / step` 必須同輪、同形狀引用。
**但這不能讀成「做到了」**：本輪帶 `CGC_HOOK_SPLIT` 儀器、且處於高 swap（10.1/11.2 GB）與 `pre` 異常膨脹（333.9 µs/call，§3）之下，絕對值不可與未帶儀器的交付讀數並排。

---

## 5. 這一輪**沒有**答到的缺口（下一個量測）

1. **本輪沒有自己的 byte 稽核**。server 未優雅關閉 ⇒ 沒有 `read shape` / `final stats` / `miss attribution`。所以 §2.1 的「page cache 速度」是**推論**。下一個同配置、**graceful stop** 的 run 要拿到 `read shape`（`us/job`, `MiB/job`, `effective_rate`）與 `compulsory / capacity` 的組成 —— 後者直接決定 M 是不是池子能動的。
2. **M 的來源未知**：73.8 顆/step 分散在 ~28 層。要 `LLAMA_EXPERT_CACHE_MISS_DUMP=<path>` 做（layer, expert）直方圖，才能分辨「反覆被淘汰的熱專家」（→ 池子/替換策略）與「一次性首觸」（→ 只有預取/目標化有用）。
3. `pre` 在本輪漲到 333.9 µs/call。與 09-19 的 4.1 µs 差 80×。需要一輪低 swap 的對照來判定這是壓力造成的、還是儀器/配置造成的。

---

## 6. 本回合的可交付物

- `scripts/check/cb_miss_regression.py`：`--selftest` **10/10**，含**反向控制**（linear fixture 不得被判 PLATEAU、3 個 step 必須報 INCONCLUSIVE、null check 必須在該紅的時候紅、path check 必須計數）。預先註冊的判定與 post-hoc 擬合在輸出上**分開列示**，避免被混讀。
- `Backup/phase_decomp/F1_CB_MISS_REGRESSION_20260920.conditions.md`（跑之前寫的註冊）
- `Backup/cgc_logs/f1_cb_miss_regression_report.json`（本報告的機器可讀版）
- `Backup/cgc_logs/llama_server_20260920_114450.log`（原始 log，3921 筆配對）
- 未 commit；零殘留行程。
