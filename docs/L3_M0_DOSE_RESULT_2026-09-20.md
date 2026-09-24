# L3 M0 —— 劑量可讀了，而且**預註冊的那個計數器是錯的那一個**

日期：2026-09-20　範圍：**只加儀器**，async 拆分未寫（那是 M1，且已因結構原因停工，見 `L3_M1_BLOCKED_STRUCTURAL_2026-09-20.md`）
binary：`libllama` md5 `1a18e95b7f985846`（arm 1/2）、`316393918ed651e4`（arm 3，多 per-trigger split）、**`af8d895fd4fe1d68`（精簡後：拿掉 `pool_submit` 重複計數器，見 §7）**
形狀：Nail IQ3_XXS-denseIQ4X、MTP on k=3、pool 8 GiB、`prefill250`、temp 0.4、completion door

---

## 0. 一句話

**`fill_wait_us`（L3 的預註冊輸出）在交付 regime 的每一步都是 0.000，一個位都不會動** ——
因為那段成本不在它裡面。真正的項是**新的 `pool_wait`**：verify step（ntok=4）中位 **32.4 ms/步（臂 1）**。
所以 L3 的預註冊要改寫成讀 `pool_wait`（＋ GPU gap），否則它會量到一個恆為 0 的東西，然後把 0 讀成「重疊沒有用」。
這正是 M0 存在的理由：**先證明劑量讀得出來，再決定要不要花 2–3 h 寫 C++**。

---

## 1. 改了什麼（全部可關、預設路徑不動）

| 檔案 | 改動 | 閘門 |
|---|---|---|
| `llama-expert-cache.h/.cpp` | 新原子計數 `pool_fill_wait_us`（`fill_segments_pool` 裡 `pool_done_cv.wait(pool_outstanding==0)` 的時間，**無條件累加**）、`n_fill_await_missed`（M1 的 fail-closed 佔位）；final stats 追加兩欄 | 無（只讀時鐘，不動數值） |
| `llama-context.cpp` | 新行 `CGC-L3DOSE`，每 decode 步一行；layer-ahead 觸發普查（`la_entered/pred_missing/pred_empty/pred_ok/issued`） | 沿用既有 `CGC_HOOK_SPLIT`（已在 allowlist，**沒有動 allowlist**） |
| `scripts/check/l3_dose.py` | 讀該行、逐 (ctx,ntok) 取中位、**缺席＝INVALID 不是 PASS**；`armed=1` 且 `await_missed>0` ＝ FAIL | 自測 11/11 |

見證在**出貨的 dylib**（不是計畫）：`strings` 命中 `CGC-L3DOSE`、`pool_wait_us`、`await_missed`、`armed=0`、`la_pred_empty`。

**語義**：三個數都是**每步總和**（跨該步各層加總），不是每次呼叫 —— 避免重犯 F3 那個 per-job / per-step 的混淆。

---

## 2. 讀數（同一顆 binary，兩臂只差 `CGC_LAYER_AHEAD_PREFETCH`）

| | 臂 1 `layer-ahead=OFF`（交付預設） | 臂 2 `ON` |
|---|---|---|
| decode（HTTP，96 tok） | 10.15 t/s | 11.64 t/s |
| accept / mean_len | 51.96% / 2.610 | 57.88% / 2.880 |
| step 中位（ntok=4） | 190.12 ms | 189.42 ms |
| `wait/pick-cb/submit` | 132.54 / 44.55 / 6.65 | 127.41 / 53.43 / 3.91 |
| **`pool_wait`（新）** | **32.36 ms/步** | 40.94 ms/步 |
| `fill_wait`（舊計數器） | **0.000** | **0.000** |
| `await_missed` / `armed` | 0 / **0**（＝未檢查） | 0 / **0** |

draft context（ntok=1/4/7/8）**全 0**：draft 不走這條 pool 批次路徑 —— 一個結構事實，先前沒有人分開量過。

---

## 3. 三件這次才成立的事

**① 舊計數器與新計數器的關係是「子集之外」，不是「子集」。**
`pool_wait 32.36` 落在 DECPROF 的 `cb 44.55` 之內（73%），差 **12.2 ms** 與 F1 的
`0.46 ms × ~29 有 miss 的層 ≈ 13 ms` **獨立收斂** —— 兩個儀器、兩條線（F1 的迴歸 / 本輪的直接量測）給出同一個殘差。
⇒ `cb` 的組成第一次有了逐項對帳：`cb ≈ pool_wait(32.4) + 放置/發布/殘餘(12.2)`，而 `submit` 由 DECPROF 另行量到。

**② L1 的結論第一次有了活體證據，而不是池子幾何的推論。**
臂 2 把層前預取打開，觸發**確實進去了**（verify ntok=4：`entered ≈ 40/步`），但 **`issued = 0` 兩臂都成立** ——
候選集裡沒有「冷且有空槽」的 expert。這同時解釋了 F2 的 `prefetch=0/0`：
**那不是儀器壞掉，是沒有候選**。F2R §7 要的「進了但 predicate false」與「根本沒進」現在分得開
（`pred_missing` / `pred_empty` / `pred_ok` 三格；`issued` 是候選數）。

**③ 因此 L3 判準要改寫（本輪唯一的實質修正）：**

| | 原判準（`L3_ADOPT_DECISION` §2） | 改寫後 |
|---|---|---|
| 主輸出 | `fill_wait_us` ↓ | **`pool_wait` ↓**（fill_wait 在交付 regime 恆 0，動不了） |
| 次要 | `total` ↓、GPU idle 沒等量升 | 不變（保留） |
| 否證 | 「fill_wait 降但 total 平」 | **「pool_wait 降但 total 平（時間搬進 `wait`）」** |

`L3_ADOPT_DECISION` §3 說「`fill_wait_us` 在 decode row 上一次都沒出現過」——本輪給了機制解釋：
它不是沒被印，是**恆為 0**，因為 demand 批次的等待從來不在那個計數器裡。

---

## 4. 臂 3（新 build，加上 per-trigger split）與閘門

臂 3 是**同一顆新 binary `316393918ed651e4`**、layer-ahead ON、48 tok：

```
verify/ntok=4  n=57   pool_wait=43.926  fill_wait=0.000 ms
              la: entered=156  pred_missing=6  pred_empty=0  pred_ok=150  issued=0
verify/ntok=1  n=1    la: entered=40   pred_missing=1  pred_empty=0  pred_ok=39   issued=0
```

**0.000 的 `fill_wait` 與 43.9 ms 的 `pool_wait` 並存**，這是本輪的核心讀數：它們不是同一筆錢的兩個名字。
`pred_empty=0 / pred_ok=150 / issued=0` 把 L1 的結論從「池子幾何的推論」變成**活體讀數**：預測有、非空、但**一顆都不冷**。

⚠️ **`entered` 的分母要讀對**：`entered=156 / 57 步 ≈ 2.7 層/步`（ntok=4），而 ntok=1 是 40/1。
原因是 `CGC_LAYER_AHEAD_PREFETCH` 那個區塊只長在**批次池路徑**上，而 ntok=4 的 verify 大多數層走的是 fast path
（其 SYNCFILL 自己呼叫 `ensure_batch`，所以 `pool_wait` 仍照樣累加）。⇒
**`la_*` 是「走那條分支的層」的普查，不是全 41 層的普查**；M1 要為每層註冊 await 時，這一格正是必須先知道的覆蓋率。

### 閘門（M0 收尾的關鍵一格）

```
GATE l3_m0_b: PASS   M1(bit-identical)=9/9  M2(argmax)=9/9  M3(topk)=9/9  n=9
cross-tab: {'num_eq_dec_eq': 9, 'num_eq_dec_ne': 0, 'num_ne_dec_eq': 0, 'num_ne_dec_ne': 0}
build: libllama=316393918ed651e4  libggml-metal=544b6c94bc7a372d  llama-server=054fb22f04a01c5c
quality: zero_mapped_selected=0 (must be 0) -> OK ; fast_cold=0 verify_cold=0
```

「M0 不動數值」因此從**讀碼的結論**升級為**閘門的讀數**（v6 參考、prefill250 profile）。

### 順手抓到的一個 teardown 缺陷（真跑中重現並驗證修好）

`m123_oracle_gate.py:1155` 的 teardown 用 `pkill -INT -f f"--port {port}"` —— pattern 以 `--` 開頭，
BSD getopt 會讓**整通呼叫變成 no-op**：

```
pkill -INT -f "--port 9932"   ->  "-INT: illegal option -- -", rc=2, 什麼都沒送
pkill -INT -f  "port 9932"    ->  rc=1（解析正常，只是沒有符合的行程）
```

因為是 `check=False`，它會**安靜地留下一個活著的 server，然後印「stopped」**。第一支 gate 跑到 20:24 就是這樣（exit 0、verdict 正確，
但 teardown 沒生效）。已改成 `port {port}`（同一條 command line 的子字串，匹配的行程完全相同），並在第二支 gate 真跑中確認
`illegal option` 消失、殘留 0。全庫同型掃描：只有這一行用 `--` 開頭的 pattern。

---

## 5. 未做 / 待驗（不在本輪宣稱之內）
- 臂 1 vs 臂 2 的 `pool_wait` 差（32.4 → 40.9 ms）**不可歸因給 layer-ahead**：兩臂 accept 不同（51.96% vs 57.88%），
  miss/工作量也不同；要歸因得等下一輪同形狀配對。
- 絕對值帶著 `--need-mb 5400`（低於文件地板 8000）的窗口警告，已由 launcher 寫進 provenance。

---

## 6. 精簡（2026-09-20 晚，`af8d895fd4fe1d68`）

精簡後重驗（QUIET 窗口、layer-ahead ON、48 tok）：`verify/ntok=4` **pool_wait=47.706 ms/步**（50/50 步 > 0）、
`fill_wait=0.000`、census `entered=170 pred_missing=6 pred_empty=0 pred_ok=164 issued=0`、`await_missed=0 armed=0`；
閘門 **PASS M1 9/9 / M2 9/9 / M3 9/9**（`summary_l3_m0_slim.json`，`libllama=af8d895fd4fe1d68`）。
日誌：`Backup/cgc_logs/llama_server_20260920_204147.log`。

第一次交出去的 M0 比需要的大，已砍：

- **`pool_fill_submit_us` 整個拿掉**：它量的是 calling-thread 的 job 建立＋`F_RDADVISE`，而那個成本已經有兩個儀器在量
  （`CGC-FSSPLIT` 的 `submit=`、`CGC-EBSPLIT` 的 `fill=`），且 L3 搬走的是**等待**不是 submit。留著只是第三個名字。
- **`l3_dose.py` 從 270 行降到 ~170**：拿掉 `--json`、只給一個格式（不再容忍舊 build）、去掉未使用的 `first/last`、
  以及 `hook_side`／`pool_wait_max` 這些由其他欄位算得出卻各自印一次的衍生值。自測保留 8 項（四個判定＋四個數字），
  其中「缺席＝INVALID」、「armed+missed＝FAIL」、「dose 全 0＝READABLE-BUT-ZERO」仍是必須紅/必須不綠的三個。
- 註解縮短：欄位與檔案層註解各砍一半行數，重複的「為什麼」只留一處。

留下的（各自的唯一讀者）：`pool_wait`（L3 的靶）、`fill_wait`（證明舊計數器抓不到它）、`la_*` 五格（F2R §7 要求的分辨率）、
`await_missed`＋`armed`（M1 的 fail-closed 佔位）。不確定的那一項：`await_missed` 現在是**誰都不會動的佔位**——
因為 M1 已停工；留著它是因為它是被明示要求的接口，且回到 M1 時第一件事就是它。要拿掉只是一行。

## 7. 產物

- 引擎：`llama-expert-cache.{h,cpp}`、`llama-context.cpp`（＋ `run_server.sh` 未動，因為沿用既有閘門）
- 工具：`scripts/check/l3_dose.py`（8/8 自測；`llama_server_latest.log` 是懸空 symlink 時只跳過，不 traceback）
- 修 tool：`scripts/check/m123_oracle_gate.py:1155` teardown 的 `--` 開頭 pattern
- 驅動：`Backup/phase_decomp/l3dose_capture.sh`（臂 1／臂 2）
- 讀數：`Backup/phase_decomp/l3dose_off.json`、`l3dose_la.json`、`l3dose_arm3.json`
- 閘門：`Backup/m123_oracle_gate/summary_l3_m0.json`、`summary_l3_m0_b.json`、`summary_l3_m0_slim.json`
- 日誌：`Backup/cgc_logs/llama_server_20260920_201501.log`（臂 1）、`…_201619.log`（臂 2）、`…_202042.log`（臂 3）
- **未 commit**
