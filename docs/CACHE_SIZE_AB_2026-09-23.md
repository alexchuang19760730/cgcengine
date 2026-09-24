# 任務 3（執行側）：expert cache 大小同場配對 —— 幾何是決定性的，速度是「4 GiB 更慢」

**日期** 2026-09-23 03:0x–03:4x · 依 `docs/NEXT_ACTIONS_2026-09-23.md` 任務 3（【執行 Agent】Expert cache
大小同場配對測試）· 載體 Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X · profile `prod25`
（MTP on、temp 0.4、`-exper-cache <bytes>`）· 兩種儀器：`prod_matrix --cells decode-spec`（llama-bench，
MTP verify 形狀）與 `decode_sweep.py --arms baseline,pool-4g`（server／HTTP 交付 regime，會讀回池的
teardown 統計）· 產物 `Backup/cache_size_ab/`

---

## 0. 一句話

**4 GiB 不是免費的，而且「免費」這個前提本身已經被池的計數器否證**：同一個 workload 下，pool 從
8 GiB 降到 4 GiB，**槽數 143 → 71（恰半）**、hit rate **86–87.5% → 66.9–70.5%**、**capacity 淘汰
4.9–6.3k → 22.3–31.7k（×4–5）**、每輪讀回的 expert 位元組 **13–15 GiB → 34–44 GiB**，而
`accept`／`mean_len` **兩臂逐位元相同**（0.37908 / 2.14）。交付 regime 的 decode 從 **10.69 → 8.50 t/s
（−20.5%）**。⇒ 任務書的判準（「4GB 跟 8GB 一樣快（<3%）就平行跑」）**不成立**，而且我必須先修掉一個
讓**所有** `prod_matrix` 臂都在量完之後被判失敗的工具缺陷，否則是這次連「有沒有到達引擎」都看不到。

---

## 1. 漂移無關的證據：池的幾何是決定性的

五支乾淨完成的啟動，逐支讀引擎自己的 teardown 區塊（`CGC-SHAPE … phase=final`）：

| 啟動（臂／順序） | `pool_cap_slots` | `resident` | hit% | misses | **capacity** | evict | 額外位元組 | us/miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p8 / 1 | **143** | 6430.6 MiB | 87.5 | 12465 | **4869** | 12353 | 13.0 GiB | 4590 |
| p4 / 2 | **71** | 3310.4 MiB | 67.2 | 40585 | **31675** | 40191 | 42.7 GiB | 3694 |
| p8 / 3 | **143** | 6430.6 MiB | 86.1 | 14209 | **6280** | 14097 | 14.9 GiB | 4782 |
| p4 / 4 | **71** | 3310.4 MiB | 70.5 | 31467 | **22345** | 31283 | 33.0 GiB | 3624 |
| p4 / 5 | **71** | — | 66.9 | 40375 | **31332** | 40191 | — | — |

三件事因此在沒有時序的情況下就成立：

1. **旋鈕確實到達引擎，且幾何逐位元可重現**：8 GiB 的兩支都是 `143` 槽 / 6430.6 MiB resident，
   4 GiB 的三支都是 `71` 槽 / 3310.4 MiB。`M=8 M_stat=DEFAULTED width=8 union=64 fits=1` 五支完全相同，
   `zero_mapped=0` 五支都是 0（**ZERO-slot 那條已知缺陷沒有混進來**）。
2. **成本結構是容量淘汰，不是首觸**：compulsory 只從 7.6–7.9k 升到 8.9–9.1k（+15%，那是更大池的
   載入差異造成的），而 **capacity 差 4–5 倍** —— 也就是這筆錢付在「同一個 token 重複要的 expert 被
   自己擠掉」上，正是預算的定義。
3. **同樣的 service rate**：us/miss 兩臂同級（3.6–4.8 ms/筆），所以差異不是裝置變慢，是**筆數變多**
   （misses 12–14k → 31–40k，每輪位元組 13–15 GiB → 33–43 GiB）。

⇒ 這一節的結論與 thermal、swap、啟動順序**無關**，因為它們全部是計數器。

---

## 2. 時序：兩支儀器都指向「4 GiB 更慢」，但都還不到可引用的精度

### 2.1 交付 regime（`decode_sweep.py`，server／HTTP，temp 0.4）

| 臂 | decode t/s | hit% | misses | us/miss | io MiB/s | accept | mean_len | 啟動熱章 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline（8 GiB） | **10.69** | 83.8 | 39948 | 5164 | 219 | **0.37908** | **2.14** | NOMINAL（最壞 MODERATE，12 輪全 NOMINAL）|
| pool-4g | **8.50** | 64.1 | 102354 | 3763 | 301 | **0.37908** | **2.14** | MODERATE（最壞 HEAVY：7 MODERATE／4 NOMINAL／1 HEAVY）|

`accept` 與 `mean_len` **兩臂逐位元相同** ⇒ 這個 −20.5% 是**成本差**，不是 token 產出差（不是「4 GiB 順便
改變了投機接受率」）。但這一對**不是乾淨配對**：4 GiB 那支啟動在 MODERATE、跑進 HEAVY。

### 2.1b 反序第二輪：熱擾動被移除，而效應留下來

| 輪 | 8 GiB（baseline） | 4 GiB（pool-4g） | 配對差 |
|---|---:|---:|---:|
| A（baseline → pool-4g） | 10.69（啟動 NOMINAL） | 8.50（啟動 **MODERATE**） | −2.19 |
| B（pool-4g → baseline） | 10.78（起 NOMINAL、收 HEAVY） | **8.35（12/12 輪 NOMINAL）** | −2.43 |
| median | **10.74** | **8.43** | **−2.31 ＝ −21.4%** |

四個論點支持這是一個**可引用的效應**，而不是熱漂移：

1. **兩輪差同號，幅度只差 10%**（−2.19 / −2.43），而臂內啟動間散開是 **0.8%（8 GiB）與 1.8%（4 GiB）**
   ⇒ 效應是儀器噪音的十幾倍（遠大於 6-block 的 14.7% 地板）。
2. **反序那支 4 GiB 全程 NOMINAL（連 12 輪都 NOMINAL）仍讀 8.35** ⇒ 熱不是它慢的原因。
3. **計數器逐位元相同**：兩支 4 GiB 都是 `hit 64.1% / miss 102354 / accept 0.37908 / mean_len 2.14`，
   而 t/s 只差 1.8% ⇒ 機制量（§1）與速度量在同一支儀器上都穩。
4. **臂序在兩輪之間相反**（A：8→4；B：4→8），所以「序列裡第幾次啟動」這個協變數**不與臂對齊**，
   而差在兩輪都向外同一個方向。

所以本節的述句從「方向確定、幅度不可引用」升級為：**在交付 regime、這支 server 儀器上，
4 GiB 比 8 GiB 慢 −21.4%（median 10.74 → 8.43 t/s，兩個相反臂序各一次）**。

#### 2.1c 第三、四輪（purged + 逐臂熱閘，四臂各一輪、臂序相反）：效應的**大小**要改，而真正該拿來當基準的是 4 GiB

`Backup/cache_size_ab/ds_346A.json`（04:06–04:22，baseline→6g→4g→3g）與 `ds_346B.json`
（04:24–04:39，3g→4g→6g→baseline），`Backup/cache_size_ab/run_346.sh` 逐臂等 OS 回報 Nominal 才啟動。
把**今天全部五支 8 GiB** 與**四支 4 GiB** 放在一起：

| 池 | 逐支 t/s（p50，時序） | median | 極差 |
|---|---|---:|---:|
| **8 GiB** | 10.43 (03:51) · **7.35 (04:06)** · 10.69 (03:23) · 10.78 (03:40) · 10.84 (04:36) | 10.69 | **1.47×** |
| **4 GiB** | 8.19 (04:14) · 8.23 (04:29) · 8.35 (03:36) · 8.50 (03:26) | **8.27** | **1.04×** |
| 6 GiB | 8.87 (04:10) · 9.55 (04:33) | 9.21 | 1.08× |
| 3 GiB | 6.90 (04:19) · 7.26 (04:24) | 7.08 | 1.05× |

（07:1x 又跑了五支 4 GiB，`docs/T5_PARALLEL_AND_4G_BASELINE_2026-09-23.md`：8.29 · 8.31 · 8.17 · 8.14 · 7.60，
median 8.17。把那五支併進來，4 GiB 的 n=9 是 **7.60–8.50、median 8.23（1.118×）**，跨池比 **8.23 / 10.69 = −23.0%**，
與下面的 −22.6% 同級。那九支的 `misses / evictions / slots / resident` 仍然全部逐位元相同。）

三件事跟著改變：

1. **中位數判決不變**：8.27 / 10.69 = **−22.6%**，與 §2.1b 的 −21.4% 同級。所以「4 GiB 更慢」是真的。
2. **但單一臂序的幅度可能差 ±20 個百分點**：`ds_346A` 的第一臂（8 GiB，purge 後第一支）讀 **7.35**，
   **比同輪的 4 GiB（8.19）還慢 11%**；`ds_346B` 的 8 GiB（最後一支）讀 10.84，比 4 GiB 快 32%。
   同一個池旋鈕，兩輪臂序相反就從 +11% 翻到 −32% ⇒ §2.1b 那個「兩輪同號」是**該兩輪**的性質，不是通則。
3. **8 GiB 的 1.47× 由**单一**離群值驅動，不是「這個臂永遠不穩」**：那五支只有 04:06 的 **7.35** 偏低，
   其餘四支在 **1.4%** 內（10.43–10.84）。兩件事因此並存：**8 GiB 平常快且穩（−23% 是對它量的），
   但偶爾掉一次 30%**（它是被超訂的那一支：`resident 6430 MiB`，swap 當時滿 4.97–6.14 GiB）；
   而 4 GiB（`resident 3310 MiB`）從不特別快也從不特別慢。我上一輪把 8 GiB 寫成「站不住」是**過頭**，
   正確的述句是「它有離群值」。對照組見 §2.2：同一時段用 llama-bench 量同一件事，單支啟動自己的
   三個 rep 就差 2.7×。

⇒ 這一節真正的產出不是「−22%」，而是**兩支的引擎側量都逐位元可重現，差全在每次事件的時間**：
所以 baseline 要引用的不是「最快的讀數」，而是**在同一個 session、同一種熱章下的中位數**（4 GiB：8.30
於 NOMINAL 的兩支；8 GiB：10.7x）。
一個基準要的正是這件事，所以任務 5 選 4 GiB 當工作區間是對的 —— 理由是可重現性，不是速度。
（0.774 × 2 = 1.55 > 1，所以「平行跑兩個」在算術上仍然划算，前提是平行本身不會互相拖。）

閘門只擋了**啟動時**的熱狀態：這四支全部 NOMINAL 起跑，但每一支都跑到 HEAVY（`thermal_worst`）。
也就是說「7.35 是熱造成的」不成立（它是該輪**第一支**、最冷的一支），時序那格才是未解的協變數。
（07:1x 那五支 4 GiB 反而把這件事補上了：它們的熱章是 NOMINAL/NOMINAL/MODERATE/MODERATE/HEAVY，
而 t/s 正好是 8.29/8.31/8.17/8.14/7.60 —— **單調**。所以「熱章解釋了多少」在 4 GiB 這一邊是可讀的，
在 8 GiB 那一邊不是。）

---

## 2.2 MTP verify 形狀（`prod_matrix --cells decode-spec`，llama-bench）—— **這一支今天不能引用**

四支完成的啟動，`platform` 值（丟 rep 1）：

| 臂 | platform t/s | 同支啟動的三個 rep | 同支內散開 |
|---|---:|---|---:|
| p8 | 12.96 | 9.14 / 13.62 / 12.31 | 1.49× |
| p4 | 7.99 | 7.85 / 7.59 / 8.38 | 1.10× |
| p8 | 10.82 | 5.36 / 11.12 / 10.53 | **2.07×** |
| p4 | 17.12 | 7.58 / 13.57 / **20.66** | **2.73×** |

⇒ 單支啟動**自己**的三個 rep 就能差 2.7 倍，而 20.66 t/s 超出這台機器歷史上任何同形狀讀數的上緣。
**這個 cell 在今天的盒況下無法分辨兩臂**，它也解釋了為什麼「取 median 就好」在這台機器上不成立
（median 要對同分佈的樣本才有意義，而這三個樣本不是）。

⚠️ 這一節與 §2.1 **不矛盾，因為它們是兩支不同的儀器**：server／HTTP（decode_bench，交付 regime）在
同一時段給出 0.8–1.8% 的臂內散開，而 llama-bench 的 `decode-spec` cell 給出 1.1–2.7× 的同支散開。
同一台機器、同一天、同一個池旋鈕 —— 一支出可引用的數字，一支不行。**要問池的大小就用前者。**
本節的臂仍然貢獻了 §1 的計數器證據。

---

## 3. 判定

- **任務 3 的決策前提不成立**：小池不是「一樣快、只是省記憶體」。4 GiB 讓 hit 掉 ~20 個百分點、
  capacity 淘汰 ×4–5、每輪 expert 位元組 ×2.5–3.3；交付 regime 的 decode **10.69 → 8.27 t/s
  ＝ −22.6%**（8 GiB 五支／4 GiB 四支的中位數），而 `accept`／`mean_len` 兩臂逐位元相同。
- **答覆任務書的判準**：問的是「差距 < 3% 就平行跑」。實測是 **−22%**（§2.1c），不是 3% 以內。
  但**這個 −22% 的精度只有一階**：8 GiB 臂本身的極差是 **1.47×**，所以單一臂序可能讀成 +11%（4 GiB 更快）。
  要用這條數字請用中位數並附極差。
- **但「能不能證明 <3%」是另一個問題，那個今天答不了**（3% 需要 ≈110 次交替啟動，
  `docs/PAIRED_PROTOCOL_3PCT_2026-09-23.md` §4）。本報告只宣稱 §2.1b 那個 21% 的效應。
- **建議**：池不要降到 4 GiB。真要省記憶體就得付這 21%；若要量 6 GiB 這一格（曲線是斜的還是平的），
  同一支儀器再跑一輪即可（`--arms baseline,pool-6g`），成本 ~10 分鐘。

---

## 4. 過程中修掉的三個工具缺陷（兩個擋住了這次任務本身）

### 4.1 `decode_sweep.py` 的 3.10-only 標註 ⇒ **每一個 `prod_matrix` 臂都在量完之後被判失敗**

```
File ".../scripts/check/llama_bench_matrix.py", line 318, in harvest_bench_stats
  spec.loader.exec_module(mod)
File ".../scripts/check/decode_sweep.py", line 967, in <module>
  def port_listener(port: int | None) -> list[str]:
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```

`llama_bench_matrix.harvest_bench_stats()` **import 這支 driver** 來讀形狀統計，而 `int | None` 在 3.10
之前在 **def 求值時**就會炸。系統 python3 是 3.9 ⇒ 前兩支啟動 rc=1、`rows: []`、`platform: {}`，
而**完整的量測其實已經躺在同一支 run 的 `.stderr.log` 裡**。兩件事讓它更值得記下來：

* **它是直譯器相依的**：09-18 的 `prod_matrix` 產物 `matrix_cmd` 用的是
  `/Users/alexchuang/.workbuddy/binaries/python/versions/3.13.12/bin/python3`，今天的是
  `/Library/Developer/CommandLineTools/usr/bin/python3`（3.9）——同一道命令，換個 python 就從「可用」變成
  「每個臂都失敗」。這條路徑上**任何**近期產物都要按它的 python 重新判讀。
* **它是在 HEAD 裡的**（`git log -S "def port_listener"` → `612005d94`「引擎 src＋binary 與三條線的工具／
  文件一次進版控」），所以不是某個人工作區的暫時狀態。

修法：`from __future__ import annotations`（PEP 563）加在該檔開頭，一個 import、零行為變更；已驗
`decode_sweep.py` 在 **3.9.6** 下可 import，並用它重新 harvest 失敗臂的 stderr，拿到完整統計
（`resident_mib=6430.62`、`worst_slots=143`、`hit_rate_pct=87.5`…）。另外我的 driver 也正是靠
**上一輪**替 `llama_bench_matrix` 補上的 `rc`／`error` 欄位才看到這件事——沒有那兩欄，一次 rc=1 會
長得跟「沒讀數」一模一樣。

### 4.2 `mmid_shapes.py` 的 `--help` 直接崩潰（任務 1 的前置）

`--pool-mib` 的說明字串先用 `% POOL_MIB_DEFAULT` 格式化過，剩下的 `26%%` 變成單一 `%`；argparse 之後
**再**用 action 的 params 對每個 help 做一次 `%`-格式化 ⇒ `TypeError: must be real number, not dict`。
改成字串串接、完全不進 `%`-格式化。

### 4.3 `mmid_shapes.py` 的產物沒有 thermal 欄位（任務 1 的四條條件之一無法對帳）

新增 `thermal_stamp()`（**匯入既有 `thermal_pressure`**，在 begin/end 各記一次），讀不到時保持 `None`，
不偽造成 NOMINAL。

### 4.4 我自己寫的那支 driver 已刪除

我先寫了 `scripts/check/cache_size_ab.py`（R T R T 交錯 + bracketed 分析），隨後查到
`decode_sweep.py` **本來就有** `baseline`／`pool-4g`／`pool-6g`／`pool-10g` 這些臂、會讀回池的 teardown
統計、也有熱章 —— 也就是同一件事的第二個介面。依照「同一件事只能有一個出處」把它**刪掉**，改用既有的
那支；`bracketed_ab.py` 仍然只負責統計（本報告沒有用它，因為這一對的效應遠大於它的地板）。
它同時示範了這一輪最容易犯的錯：**它把「沒有讀數」寫成空字串並繼續跑**（第一版就是這樣把兩個 rc=1 的臂
寫成兩列 tps 空白的資料）。現在的規則寫在報告裡，而不是留在被我刪掉的檔案裡。

---

## 5. 誠實邊界

- **量測閘與啟動器這次不一致**：整個時段 `server_window.decision()['admits']` 都是 `False`
  （`harness:memory`，reclaimable 6.9–7.7 GiB < 8000），而 `launcher_admits` 一直是 `True`。任務書
  禁止降 `NEED_MB`，我沒有降；但這代表上面每一支啟動都**不是**在任務 1 那種 admit 窗口上跑的——`baseline`
  那支的熱章是 NOMINAL、`pool-4g` 那支是 MODERATE。**計數器（§1）不受影響，速度（§2）因此不可並排。**
- swap 5.6–5.8 GiB 是帶進來的（別條線的痕跡），每列都記了。
- `--rounds 5` 是**啟動內**的 5 輪 HTTP 請求，不是 5 次啟動 ⇒ 本質上仍是「每臂一次啟動」；反序第二輪
  （`ds_order_b.json`）補的是**順序**那一維，不是次數。
- `decode-spec` 是 `prod25` 上唯一可跑的 decode cell（`decode`／`decode-up`／`prefill-house` 都被
  prod_matrix 以「池路徑的 clamp 讓 -b 512 不是有效 batch」為由 SKIP）——這點在任務書裡沒寫，是我在
  dry-run 才發現的。
- 未 commit。所有產物在 `Backup/cache_size_ab/`（`Backup/` 依 .gitignore 不入庫）；零殘留行程。
