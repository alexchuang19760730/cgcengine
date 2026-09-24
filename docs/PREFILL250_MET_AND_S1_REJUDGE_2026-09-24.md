# prefill 250 已達標；單段提交的「2.2×」重判為 1.72×，且那個比值不可歸因為 shape

**2026-09-24** ｜ 性質：**靜態核對 ＋ 離線稽核**（0 GPU、0 rebuild、0 起 server）。全部數字由既有產物機器讀出，
不是手打。這一篇的作用是**改寫兩份現行文件的結論**，不是新增一組量測。

---

## 0. 兩句話

1. **prefill 250+ 已經達標**，而且不是一次僥倖：同一個 cell（`-p 2048`，prod-new 口徑）在可指名來源裡
   有 **9 次 launch ≥ 250 t/s**（最高 296.24）。白皮書 §5.1 報的 `ctrl 162.5 → p0+B 209.5（+29%）`
   是一個**降級視窗**的產物 —— 同一支臂在同一個視窗內另外三次是 275.39 / 196.72 / 207.33 / 204.98
   （同臂、同 build、同 cell，十分鐘內差 **78.7 t/s**）。
2. **單段提交（S1）那對「2.2×」在產物本身裡是 1.72×**，而且兩臂的 **I/O 結構不同**：
   A（41 段、hook 會響）做了 **82,293 次檔案讀、2,440 MiB**；B（單段提交、hook 不響）做了 **0 次、
   0 MiB**。⇒ 這個比值**不是 shape 收益**的一部分是 I/O 本身，不可引用為「單段提交值 2.2×」。

---

## 1. prefill 250：已達標

### 1.1 紀錄（同 cell，可指名來源）

| prefill t/s | 來源 | 備註 |
|---:|---|---|
| **283.01**（293.79 / 280.17 / 275.08） | `docs/PROD_NEW_MTP_OFF_2026-09-23.md`（HEAD `155ee2474`，乾淨視窗 150/150 NOMINAL） | 我這條線的認證格 |
| **296.24** | `Backup/seg_batch_s1_pairs/res_B2.json`（09-24 21:19） | S1 臂 B 第 1 次 |
| **275.98** / **269.23** / **262.01** | 同上 `res_B6` / `res_B3` / `res_B7` | S1 臂 B 第 2–4 次 |
| **275.39** | `res_A1.json`（21:18） | S1 臂 A 第 1 次 |
| **289.28** | commit `a1ca0c23a` 標題（commit-gate） | prod-new p2048 n128 d512 r3 |
| **280.5** | commit `38a59e39f` 標題（commit-gate） | 同上 |
| **289.81** | `docs/RHO_MAXQ_SETTLED_2026-09-23.md` §5（commit `079f2fe46`） | commit_bench 標題 |

⇒ **達到 250+ 的 launch 共 9 次，跨 3 天、跨 4 顆 binary、跨兩條不同的 profile 主張**。這不是邊緣命中。

### 1.2 為什麼文件上會出現「不達標」的 162.5 / 209.5

同一支臂（A，分段 prod-new）在**同一個視窗、十分鐘內**的四次 launch：

| A 臂 launch | prefill t/s |
|---|---:|
| A1（21:18） | **275.39** |
| A4（21:22） | 196.72 |
| A5（21:24） | 207.33 |
| A8（21:28） | 204.98 |

**同臂散佈 78.7 t/s（中位 206.2 的 38%）。** 白皮書那格 `209.5 ± 81.7` 的 ±81.7 就是這個東西。

⇒ 162.5 / 209.5 **不是程式碼的性質，是視窗的性質**。另外兩個低值同樣有出身：
`177.36`（commit `d10a6406d`）自己的標題就寫 `FAIL(swap8.6GB脏,环境不可引用)`。

### 1.3 因此正確的述句

- **目標已達**：prefill 250+ 在 16GB M4 上可重現地達成（9 次 launch）。
- **未達的是「可重現的**地板**」，不是峰值**。要報一個可引用的 prefill 數字，必須像
  `docs/PROD_NEW_MTP_OFF_2026-09-23.md` 那樣**連視窗區塊與 build digest 一起報**；
  單獨一個 `162.5` 既不能當交付值，也不能當 ctrl 臂 —— 它與同一視窗的 275.39 是同一支臂。
- **白皮書的 `+29%`（prefill）不成立**：它的分母取自降級視窗，且效果（+47 t/s）小於該臂自身的散佈
  （±81.7）。這與 P0 的真實效果（swap −92%）是兩件事，後者可以引用。

---

## 2. 單段提交（S1）配對：1.72×，不可歸因為 shape

**來源**：`Backup/seg_batch_s1_pairs/abba_212809.json` ＋ 該目錄的 `stdout_*.log`。
8 次 launch、交錯 `A B B A A B B A`、同 build、同 cell、**兩臂均未傳 `--spec-type`**。

| 臂 | 四次讀數（tg, n_gen=64） | 中位 | ms/token |
|---|---|---:|---:|
| A 分段 prod-new | 12.20 / 11.96 / 9.32 / 11.71 | **11.84** | 84.5 |
| B S1（單段提交） | 22.22 / 20.30 / 20.06 / 20.33 | **20.32** | 49.2 |

**中位比值 1.72×**（`io_symmetry.py` 機器算）。

### 2.1 兩臂的 I/O 結構不同（這一節是重點）

| 計數器 | A（有 hook） | B（無 hook） | 來源 |
|---|---:|---:|---|
| `misses` | **4,835** | **0** | 各臂 combined stderr `final stats` |
| `file_reads` | **82,293** | **0** | 同上 |
| `read_mib` | **2,440.3** | **0.0** | 同一輪 `CGC-SHAPE phase=final` |
| `fill_wait_us` | 3,375,682（3.38 s） | 0 | 同上 |
| hit rate | 96.2 % | 100.0 % | 同上 |

B 的快取**整場沒有讀過一次檔案**：沒有 hook ⇒ 沒有 demand fill ⇒ 填充路徑根本沒跑。
所以 B 的 49.2 ms/token 裡**沒有快取 I/O 這一項**，而 A 的 84.5 裡有。
**「把 IO 藏到 GPU 底下」和「根本沒產生 IO」在 t/s 上長得一樣** —— 這對產物分不開這兩者。

⇒ 正確述句：**這對量的是「S1 ＋ 填充路徑不執行」的合併效果，不是 S1 的 shape 收益。**
（B 的 100 % hit 也不是池子變好：無 hook 即無替換，slot 一旦被佔就不再變動。
至於那 6,197 MiB 的 bytes 實際來自哪裡 —— mmap 頁面或其他路徑 —— 需要一次 page-fault / RSS 佐證才說得準，尚未做。）

### 2.2 這對**不能**支持的三件事

1. **不能說「單段提交值 2.2×」**（B_PLAN §7.2 / MEMORY.md 現行述句）。產物本身是 1.72×，
   而被引用的 2.2× 那一輪分母是 A 的 8.24/7.81/7.93 —— **取自已失敗的子視窗，分子也取自同一段**。
2. **不能說它是交付 regime 的數字**。兩臂都無投機 ⇒ `k_eff = 1`；交付 regime 是 `mean_len 3.117`。
   S1 這個 shape **從來沒有在背著 4-token verify 批次的狀態下量過**。
3. **不能把它的 prefill 一起引用**：同一個目錄 A 臂 prefill 中位 206.2、B 臂 272.6，
   而乾淨視窗是 283.01 ⇒ 這個目錄的視窗本身是降級的（見 §1.2）。

### 2.3 出處缺口（必須記下來）

那對產物**沒有任何 engine digest**：record 只有
`arm / cell / idx / json / pp / pp_batch / pp_n / pp_sd / rc / rows / tag / wall_s`，
頂層只有 `cell / plan / records`。`grep md5|digest|binary` 零命中 ⇒
**1.72× 掛不到任何一顆 binary 上**。唯一的間接指針是 `MEMORY.md` 記的
「`CGC_SEG_BATCH` 在 `libggml-base`（09-24 11:14）、其餘在 `libllama`（14:58）」。
（這正是 gate 12 想防的那一類：一份有數字的產物答不出「是哪顆引擎跑的」。）

---

## 3. 新規則（硬規則，已實作）

**`scripts/check/io_symmetry.py`** —— 在引用任何配對比值之前，先檢查兩臂的 I/O 結構是否一致。

| 判定 | 條件 | 行為 |
|---|---|---|
| `asymmetric` | 一臂有檔案讀、另一臂為 0；或 hit 差 > 2pp；或只有一臂有 fill 等待 | **blocking**：拒絕把比值簽成效果 |
| `unknown` | 任一邊的計數器讀不到 | **blocking**（讀不到是 UNKNOWN，不是「相等」） |
| `comparable` | 在容差內一致 | 放行 |

已接進 **`scripts/check/decode_carrier_ab.py`**（這條線自己的配對判讀工具）：命中時 verdict 變成
`shape-confounded` 並以 **exit 2** 結束，報告裡明確寫出是哪個計數器不同。

```sh
python3 scripts/check/io_symmetry.py --selftest          # 11/11 PASS
python3 scripts/check/io_symmetry.py --dir Backup/seg_batch_s1_pairs      # exit 2（本檔 §2 那一對）
python3 scripts/check/io_symmetry.py --arm A=<A臂log> --arm B=<B臂log>     # 泛用
```

實跑本檔 §2 那一對的輸出：

```
!! I/O STRUCTURE ASYMMETRIC -- DO NOT QUOTE THIS RATIO AS A SHAPE GAIN
!!   * A performed cache I/O (82293 reads) while B performed none -- the fill path never ran in B
!! ratio 1.72x spans arms that did different I/O work.
```

**覆蓋範圍（誠實標註）**：它現在掛在**離線稽核**（`--dir`／`--arm`）與**本線的配對判讀工具**上。
產生那一對產物的驅動 `scripts/check/seg_batch_abba.py` 是**另一條線的未追蹤在途檔**，
我沒有改它（改了也只會進不了這個 commit）；它要接的是同樣一行：

```python
print(io_symmetry.banner(io_symmetry.audit_dir(workdir)))   # 在 _summary 最後
```

---

## 4. 這一篇改了哪些文件的結論

| 文件 | 改了什麼 |
|---|---|
| `docs/NEXT_ACTIONS_2026-09-24.md` | 開頭新增「兩個已定案的前提」＋ 把 prefill 的待辦與 S1 的述句改正 |
| `docs/CGC_ENGINE_WHITEPAPER_2026-09-24.md` | §0 成果表、§3.1 prefill 行、§5.1 現狀表、§5.3 路徑三處加上修訂註記 |
| `.workbuddy/memory/MEMORY.md` | 一句話狀態改寫（**gitignored，不會進 commit**） |

---

## 5. 下一步（不在本檔範圍）

以 §2 的 49.2 ms/token 結算：`25 t/s = mean_len / step` ⇒ **只需要 mean_len 1.23**（accept ~8%），
所以剩下的問題不是 accept，而是**單段提交那條路上 verify 第 2–4 個 token 的邊際成本**：
樂觀（7–8 ms/token，分批攤薄）⇒ step 73 ms ⇒ **42 t/s**；悲觀（完全不攤薄）⇒ 196 ms ⇒ **15.9 t/s**。
⇒ 值得做的是**一支臂**：S1 ＋ MTP on（k=1/2/3/4），量 ms/step 與 mean_len。

---

## 6. 出處

| 數字 | 來源 | 讀法 |
|---|---|---|
| A/B 八次 tg、pp、比值 | `Backup/seg_batch_s1_pairs/abba_212809.json` | 機器讀（`io_symmetry.py --dir`） |
| `misses` / `file_reads` / `hit` | 該目錄各臂 `llama_bench_*.stderr.log` 的 `final stats` 行 | 逐字 |
| `read_mib` / `fill_wait_us` | 同一輪的 `CGC-SHAPE v=1 phase=final` 行 | 逐字 |
| 283.01 / 11.72 | `docs/PROD_NEW_MTP_OFF_2026-09-23.md`（HEAD `155ee2474`） | 既有產物 |
| 289.28 / 280.5 / 177.36 | commit 標題 `a1ca0c23a` / `38a59e39f` / `d10a6406d` | git log |
| 289.81 | `docs/RHO_MAXQ_SETTLED_2026-09-23.md` §5 | 既有產物 |
| `h`、`mean_len 3.117` | `docs/DECODE_25TPS_VERDICT_2026-09-23.md` §1 | 既有產物 |

**本檔 0 次 benchmark、0 次 rebuild、未起任何 server；未留下任何行程。**
