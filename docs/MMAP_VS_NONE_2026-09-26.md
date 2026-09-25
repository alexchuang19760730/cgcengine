# `--load-mode mmap` vs `none`：設計裁決（2026-09-26 00:01–00:16）

**載體** prefill250 profile、ub=5632、engine digest `2ac1cbb9cfcb`、HEAD `28cbf2dca`、
driver `Backup/run_mmap_ab.sh`（本輪新增 `ws_warn` / `ws_info` / `srv_log` 三欄與跑前凍結抬頭）。

## 0. 一句話

在**當前 build（含 2026-09-24 的 P0/P1/P2）**上重測：**mmap 仍 0/2 死亡、none 2/2 存活**，
而且**凍結的主判準（Metal working-set 警告）2/2 預測了死亡、0/2 預測了存活**。
⇒ 有界 pool ＋ `load-mode none` 是這台 16 GB 盒子上的正確設計；`mmap` 不是——
理由不是 host 記憶體，是 **GPU 工作集**。

## 1. 協定（跑前凍結，寫在 driver 抬頭）

- 臂：`none:5632` 與 `mmap:5632`，**交錯兩輪**（R1 = none→mmap，R2 = mmap→none），
  所以死亡不可能由啟動序解釋。選 5632 是因為它是**當前預設寬度**（none 歷史上 4/4 存活）
  ⇒ mmap 若死而 none 活，成因是 load mode，不是寬度、也不是盒子。
- **主判準**：該臂自己的 server log 中 `greater than the recommended max working set size` 的次數
  （`ggml-metal-device.m:1539`；已用 `strings` 確認在出貨 dylib 裡、`GGML_METAL_NDEBUG` 預設 OFF）。
  歷史值：`load_mode=none` 119 次啟動 0 次、mmap 可判定日誌 5/5。
- 次判準：`ready` / `survived` over req1..req3。
- **作廢條件**：none 對照臂若不存活 ⇒ 是盒子決定的，不是設計。**本輪未觸發**（none 兩臂都活）。
- **盒況偏離（宣告）**：起跑 swap **3234 MiB > 契約的 2048 MiB**（未重開機）。
  對照臂同一盒況，故比較仍有效；但絕對吞吐不可與乾淨槽的讀數並排。

## 2. 結果

| round | arm | load_mode | ready | last | survived | req1 | req2 | req3 | **ws_warn** |
|---|---|---|---|---:|---:|---|---|---|---:|
| 1 | `r1_none_ub5632` | none | **yes** | req3 | **yes** | 243.99 | 258.25 | 253.20 | **0** |
| 1 | `r1_mmap_ub5632` | mmap | **no** | 0 | n/a | — | — | — | **1** |
| 2 | `r2_mmap_ub5632` | mmap | **no** | 0 | n/a | — | — | — | **1** |
| 2 | `r2_none_ub5632` | none | **yes** | req3 | **yes** | 238.92 | 243.19 | 228.94 | **0** |

判準與存活**完全對齊**（有警告 ⇒ 死；無警告 ⇒ 活）。這比 09-16 的關聯更強：它在這一輪是**預測性的**。

## 3. 死因（逐字，來自 mmap 臂自己的 log）

```
0.00.891.120 W ggml_metal_log_allocated_size: warning: current allocated size is greater than the recommended max working set size
0.50.719.695 E error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

警告先出現、錯誤後出現同一份 log ⇒ 警告不是裝飾，是死亡的前一行。與 09-16 的 `status 5` 同一死法。

## 4. 引擎**自己**的頁在哪（本輪新儀器 `Backup/rerun/swap_watch.py` + `swap_owner_probe.py`）

| arm | 引擎自身 SWAPPED 峰值 | 可寫區 resident | box swap（該時刻） |
|---|---:|---:|---:|
| `none`（R1／R2） | **1331 MiB** | 8.60 GiB | 6891／7153 MiB |
| `mmap`（R1／R2） | 90／92 MiB | 0.01／6.20 GiB | 5335／4535 MiB |

**這修正了我先前的一個假設**：我說過「引擎自己的頁全程駐留、5 GB 都是別人的」——那只對 **wired** 的部分成立。
實測是：在 `none` 臂裡，引擎**自己的可寫匿名區有 1.33 GiB 在 swap**；`mmap` 臂幾乎是 0（file-backed 的頁
不是「進 swap」而是直接被丟掉）。

⇒ 所以「swap 不降速」這句話對我們**不是免費的**：1.33 GiB 是我們自己的頁，只要其中任何一頁在 step 內被碰，
就是一次 page-in。**逐區分解目前不可引用**：第一次的 parser 讀錯了表（把 MALLOC ZONE 表的數字當成 region），
已修正並補上會抓到這件事的自測（8/8），但要下一輪才能取得可信的逐區表。

## 5. 機制：mmap 為什麼會死（與 09-16 同一結論，但現在有同輪證據）

mmap 讓權重變成 file-backed，**host 側可以免費回收**——本輪直接看到它的效果：mmap 臂期間 box swap
反而**下降**（6526 → 5287 → 4820 MiB）。但同一批頁同時是 **GPU 可見的 zero-copy 來源** ⇒ GPU 工作集被撐大
⇒ 超過 `recommendedMaxWorkingSetSize` ⇒ `status 5` OOM。

**能回收的是 host 頁，撞牆的是 GPU 工作集**，而這台機器撞的是後者。這正是 09-16 的判決，P0/P1/P2 沒有改變它
（P0 動的是 host 側的匿名副本，而那從來不是死因）。

## 6. 對「哪個設計才正確」的結論

1. **維持 `none` ＋ 有界 pool**——不是偏好，是這台盒子的必要形狀：它是唯一能把 GPU 工作集壓住的組態。
2. **代價要誠實列出來**：引擎自己 1.33 GiB 匿名頁在 swap（本輪首次量到），而 P0/P1/P2 正是減付這個代價的方向。
3. **什麼會讓答案翻轉**：任何「冷權重不進 GPU、但 host 頁保持可回收」的機制。目前 llama.cpp 做不到
   （權重只有『整顆在 GPU buffer』或『由我們的 fill 串流』兩條路），所以在它出現之前，
   正確的下一步是**繼續降低 `none` 側的不可回收量**，而不是換回 mmap。

## 7. 這份裁決的界線

- 只測 ub=**5632**、prefill250 profile、MTP 狀態依 profile 預設；未測 `mmap:6144`（既有紀錄是 `none:6144` 會死）。
- 未測「mmap ＋ 其他 knob」的組合；也未測乾淨槽（起跑 swap 3234 MiB）。
- req 吞吐（228.94–258.25 t/s）在此盒況下**不可與乾淨槽的 268/273/256 並排**。
- 引擎自身 SWAPPED 的**逐區分解未取得**（跑這輪時 parser 是壞的）；1.33 GiB 這個**總量**來自
  `vmmap` 的 `Writable regions: … swapped_out=` summary 行，與 parser 修正無關，故可用。

## 8. 補記（09-26 00:20）：parser 修好了，而且它比宣稱的更壞

上一輪把逐區分解列為「下一輪再做」。實際去追時發現它**不會自己好**：修正後的自測（18 項）
在真實 `vmmap -summary` 上跑，才看到三個各自獨立的缺陷。

1. **一行都沒解到。** 區表是**三行**表頭（名稱行 → `REGION TYPE … COUNT` → `=====` 尺規）。舊版讀到名稱行
   （含 `SWAPPED` 與 `REGION`）就進表，**下一行**的開頭正是 `REGION TYPE` ⇒ 立刻退表。它在每個真實行程上
   回傳 **0 列**卻報成功；那些「2347 / see MALLOC ZONE table below」的垃圾資料是從**下面那張 MALLOC ZONE 表**讀來的。
2. **自測的 fixture 是從 parser 的假設寫的**（單行表頭＋資料列），不是從真位元組寫的 —— 所以它全線通過。
   現在 fixture 用真實幾何，並且**斷言自己的欄位幾何等於現場 `vmmap` 的**（不符就紅）。
3. **單位與欄位邊界。** `1.3T` 與裸位元組（`__CTF 824`）舊版都讀不出來 ⇒ 整列消失、殘差看起來像捨入；
   而 `DIRTY` 的尺規只有 5 字元寬，真值 `992.0M` 是 6 個字元 ⇒ 拿尺規當欄位會把數字截成 `98.0M`、把 `M` 推進 SWAPPED。
   欄位邊界現在取自**表頭名稱的右緣**，並用 vmmap 的 **`TOTAL` 列做加總對帳**（實測 35 列：
   swapped 0.0015%、dirty 0.0015%）。VIRTUAL／RESIDENT 的 TOTAL 只印 2 位有效數字（`1.4T`、`1.2G`），
   本身就是 ±4% 的窗，故不對帳。

**這改變了什麼**：逐區表現在**可以**取得了，但**這一輪還是沒有引擎自己的那份**——引擎那時已經退出。
下一台引擎起來時附上 `python3 Backup/rerun/swap_owner_probe.py --pid <pid> --seconds 120`，就會第一次看到
那 1.33 GiB 是落在 `MALLOC_LARGE`（CPU 端 pool）、`Memory Tag 255`（Metal 的統一記憶體）還是別處——
而那一格正是「wired 峰值 ~10.4 GB 是池還是 Metal」那道還沒解的題的同一半。**預測（跑前寫死）**：
若是 `Memory Tag 255` 為主 ⇒ Metal 佔大頭，縮 pool 不會降 wired；若是 `MALLOC_LARGE` 為主 ⇒ 池是主因。
