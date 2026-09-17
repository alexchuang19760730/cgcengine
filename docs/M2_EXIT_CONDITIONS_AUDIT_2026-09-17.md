# M2 離開條件結清（2026-09-17）

**里程碑**：M2 — prefill 走整層串流（W1 + W3）
**權威定義**：`docs/ROADMAP_PREFILL250_DECODE25_2026-09-13.md` §M2
**這份文件做什麼**：把 M2 的五條離開條件逐條判決 —— 每一條給「量到什麼／標準是什麼／判決」。
**這份文件不做什麼**：它不新增任何量測。**唯一的新讀數是從既有 log 榨出來的**（來源逐條標明）。
未裁決的兩條**不是失敗**，是**缺儀器** —— 兩者處置不同，所以分開寫。

---

## 0. 判決總表

| # | 離開條件（原文） | 判決 | 依據 |
|---|---|---|---|
| 1 | bytes/token @ chunk 2048 **≤ 3.0 MB**（理論 2.79：只讀非常駐的 48%） | **PASS** | 引擎的 `slab fills` 計數器，實測 44.1%／44.3% 非常駐 |
| 2 | 裝置持續速率 **≥ 1.0 GB/s** | **⚠ 未能裁決（缺儀器）** | `src/` 內**沒有**任何 slab 計時計數器（零命中） |
| 3 | I/O 與 compute 重疊（**請求 wall < I/O + compute**） | **⚠ 未能裁決（缺儀器）** | 同上；`db=1` 是結構證據，不是那條算術 |
| 4 | prefill t/s：記錄實測值；**250 可達/不可達要誠實寫明** | **PASS（附否證）** | 今天 6 條臂的讀數 ＋ 跨啟動 2.25× 離散 |
| 5 | M1 / M2 **100%**（讀作 M1/M2/M3，見 roadmap §1 的更正） | **PASS** | D5 `--tag prodmatrix_std_20260917`：9/9、`comparable=True` |

**⇒ M2 是 3/5 結清、2/5 未裁決。** 未裁決的兩條都指向同一件事：**這個里程碑的兩個條件沒有對應的讀數**。

---

## 1. 條件 1：bytes/token @ chunk 2048 ≤ 3.0 MB — **PASS**

**儀器**：引擎自己的計數器，**它的原始碼註解就是為這條條件而寫的**
（`src/llama.cpp/src/llama-expert-cache.cpp:2338`）：

> `[CGC M2 pool reuse 2026-09-14] What the whole-layer slab path actually read. The M2 exit
> condition is "bytes/token @ chunk 2048 <= 3.0 MB (i.e. only the non-resident share)";
> report it as the ratio so a claim about it can be checked instead of inferred.`

輸出：`llama_expert_cache: slab fills: pool=… MiB disk=… MiB (non-resident share …%)`

**實測（今天兩條 prefill250 臂，2873-token prompt × 3 請求）**

| 臂 | pool | **disk** | 非常駐佔比 | disk/token @ chunk 2873 |
|---|---|---|---|---|
| 15:03（`UNKNOWN-STATE`） | 19063.8 MiB | **15052.2 MiB** | **44.1%** | **1.75 MiB = 1.83 MB** |
| 15:37（`COLD-STATE`） | 19005.6 MiB | **15110.4 MiB** | **44.3%** | **1.75 MiB = 1.84 MB** |

**標準要求的是 chunk 2048 的那一格，而這兩條臂跑的是 chunk 2873 ⇒ 必須換算，不能直接比。**
整層 slab 的機制是「每個 chunk 把該層的非駐留專家讀一次」（chunk 內 union 飽和到 256），
所以 `bytes/token ∝ 1/chunk`：

```
在 chunk 2048 上： 11.92 GB（expert set） × 0.441 ÷ 2048 = 2.57 MB/token   ≤ 3.0  ✓
以理論的 48% 算：  11.92 GB × 0.48 ÷ 2048 = 2.79 MB/token                  ≤ 3.0  ✓（＝roadmap 的理論值）
```

⇒ **判決 PASS，而且兩個輸入（11.92 GB expert set、44% 非常駐）都是量到的，不是假設的。**
附帶：**量到的非常駐佔比（44.1%）低於理論（48%）**，所以這條比 roadmap 預期的更寬鬆。

⚠ **一個必須寫下來的邊界**：這條條件對 chunk 大小敏感，而 `prefill250` 的 chunk 是 **5632**
（`run_server.sh` 的 profile 預設），今天的臂用 2873（單一 prompt 裝進一個 chunk）。
**引用時要指名 chunk。** 「bytes/token ≤ 3.0 MB」若在 chunk 2048 上成立，在更大的 chunk 上只會更低。

---

## 2. 條件 2：裝置持續速率 ≥ 1.0 GB/s — **未能裁決（缺儀器）**

**否證「有儀器」的方法（做過了）**：

```sh
grep -rcoE "slab_read_us|slab_read_ms|slab_io_us|n_slab_us" src/llama.cpp/src/*.cpp
# → 零命中
```

引擎印的 slab 相關輸出只有三種：`slab=N MiB`（容量）、`filled=N bytes`（單層單次）、
`slab fills: pool=… disk=…`（位元組，無時間）。**沒有任何一個帶時間。**

**為什麼不能從 wall 反推**：`prompt eval time` 是「wall − 重疊後的結果」，
把它除以 disk bytes 得到的是**下界**（若真有重疊，真實裝置速率只會更高）。
拿一個下界去判「≥ 1.0 GB/s」在方向上是安全的，但它同時會把**條件 3 的答案偷偷用掉**
（條件 3 要問的正是「有沒有重疊」）⇒ **同一份資料不能同時回答這兩條**，那會是循環論證。

**兩個能在不改 `src/` 的前提下裁決它的設計**（都還沒跑）：

| 設計 | 做法 | 它會給什麼 | 代價 |
|---|---|---|---|
| **(a) 外掛裝置計數器** | 跑一條 prefill 臂的同時，以 1 Hz 跑 `iostat -d -w 1`（或等價的 disk 統計），取臂窗的裝置 MB/s | **裝置層**的持續速率，與行程無關 | 一個背景行程；`iostat` 的輸出格式要先驗（本 repo 已有 2 Hz `notifyutil` 前例） |
| **(b) 兩點幾何外推** | 同一 prompt、同 batch，**兩個不同的池幾何**（改 `CGC_SERVER_LAYER_CAPS` ⇒ 不同非常駐佔比 ⇒ 不同 disk bytes），各量 wall | wall 對 disk bytes 的斜率的倒數 ＝ **有效速率**；截距 ＝ 純 compute | 兩次獨立啟動；幾何差異要只動非常駐佔比 |

**(a) 較強**，因為它量的是**裝置**而不是行程推論；而且它同時給條件 3 需要的分子。

---

## 3. 條件 3：I/O 與 compute 重疊（wall < I/O + compute）— **未能裁決（缺儀器）**

這條要的是**三個量的不等式**：`wall` 一個（有）、`I/O` 一個（沒有）、`compute` 一個（沒有）。

**目前唯一的證據是結構性的，不是算術的**：每次 slab 填充的 log 行都帶 `db=1`
（`CGC-PREFILL-STREAM: il=0 kind=0 ntok=2873 experts=256 slab=110.00 MiB filled=115343360 bytes db=1`）
⇒ 雙緩衝在這條路徑上是**開的**。但 roadmap 要的是「請求 wall < I/O + compute」這條**可稽核的算式**，
而一個旗標的值不得當成它自己生效的證據（`CONVENTIONS` **A9**，本 repo 已為此付過學費）。

**裁決設計**：條件 2 的設計 (b) 直接給它 ——
兩點外推的**截距**就是 compute-only 的時間，**斜率 × disk bytes** 就是 I/O；
`wall < 截距 + I/O` 成立 ⇔ 有重疊，而重疊的比例可以順便算出來。

⚠ **不要用「`wall` 比 `I/O` 的單獨估計小」當證明**：那需要一個獨立的 I/O 估計，
而沒有條件 2 的儀器就沒有那個估計。

---

## 4. 條件 4：prefill t/s 要誠實記錄 — **PASS（而誠實的內容是否證）**

今天同一個 profile、同一個 build、同一份 2873-token prompt 的**六個請求**：

| 臂 | 熱標籤 | prefill t/s |
|---|---|---|
| 15:03（`IDLE_BEFORE=0`，無 state file） | `UNKNOWN-STATE` | **278.56 / 261.17 / 275.01** |
| 15:37（安靜 1940 s） | **`COLD-STATE`** | **242.42 / 227.27 / 249.06** |

還有跨啟動的離散（`pp2048 @ -ub 6144`，四次獨立啟動）：
**276.59 / 198.84 / 176.18 / 122.68 ＝ 2.25×**。

**誠實的寫法是三句，缺一句就不誠實**：

1. **250 是可達的，但不是常態。** 15:03 那臂三個請求全部 ≥250（`UNKNOWN-STATE`）。
2. **拿到最強標籤的那一臂反而沒有 250。** 15:37 的 `COLD-STATE`（安靜 1940 s）三個請求全部 <250。
   ⇒ **「已發佈的條件集（安靜 ≥1800 s ＋ 每個請求邊界讀到 `0/NOMINAL`）是必要條件，不是充分條件」**
   （lesson `eng-mh-0054`，白皮書 `docs/PREFILL250_DECODE_MILESTONE_20260917_1510.html` §2.3）。
3. **跨啟動的那 2.25× 與熱無關**（每次啟動內部三次樣本只差 2.08–15.41）
   ⇒ 它來自**行程之間的狀態**；`scripts/check/prefill_certifiability.py` 就是為它寫的，
   而它的結論是「**輸出可重複的帶 ＋ 越過目標的次數 k/N**，不是範圍也不是目標」（`CONVENTIONS` **A15**）。

⇒ **這條離開條件是「要誠實」，不是「要達標」** ⇒ 以「寫明了」為判準，**PASS**。

---

## 5. 條件 5：M1 / M2 = 100%（讀作 M1/M2/M3） — **PASS**

`scripts/check/m123_oracle_gate.py --tag prodmatrix_std_20260917`（2026-09-17 16:26，8080 空）：

```
GATE prodmatrix_std_20260917: PASS   M1(bit-identical)=9/9  M2(argmax)=9/9  M3(topk)=9/9  n=9
cross-tab: {'num_eq_dec_eq': 9, 'num_eq_dec_ne': 0, 'num_ne_dec_eq': 0, 'num_ne_dec_ne': 0}

{'ref': 'Backup/knifeedge_matrix/ref_iq3_pool8gb_M2_6144_bitident_v6_nbaware.jsonl',
 'comparable': True, 'ok': True, 'n_compared': 9}
config_diffs: []
```

**先讀 `comparable` 再讀那三個 9/9**（`comparable=True` 且 `config_diffs=[]` ⇒ 它是判決，不是裝飾）。

**roadmap 附註的「建構保證」也成立**：prefill 走整層 slab ⇒ **永遠 materialize 256 個專家**，
union 不可能超過 slab ⇒ 不變性是**建構**出來的，不是碰巧量到的。這與條件 1 的
`filled=115343360 bytes`（110 MiB／層，`experts=256`）是同一件事的兩個讀數。

---

## 6. 這份結清的誠實邊界

- **本文件沒有跑任何新的量測。**條件 1／4／5 全部來自既有 log 與既有的閘門產物；
  條件 2／3 的兩個設計**都還沒執行**。
- **條件 2／3 的判決是「未能裁決」，不是「不合格」。** 這兩者處置完全不同：
  不合格要改設計，未能裁決要**先做儀器**。而儀器可以在**不碰 `src/`** 的前提下做完
  （設計 (a) 是外掛的）。
- **沒有一條條件是用推論通過的。** 條件 1 有兩個量到的輸入；條件 4 的數字逐條附標籤；
  條件 5 附 `comparable`。
- **`bytes/token` 那一條對 chunk 敏感**（見 §1 的 ⚠）：本文件的 2.57 MB 是**換算到 chunk 2048** 的值，
  不是今天的臂直接印出來的。臂直接印的是 1.83 MB @ chunk 2873。

---

## 7. 要把 2/5 補完，下一步只做一件事

跑**設計 (a)**：在下一條 prefill 臂的同時掛 1 Hz 的磁碟統計。
它一次給兩個條件的分子（裝置速率、I/O 時間），而條件 3 的 compute 由**同一個臂的
pool-only 對照**（或設計 (b) 的第二點）補上。**不需要 `src/` 改動，也不需要重建。**
