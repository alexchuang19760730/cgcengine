# llama-bench 能不能重建 HTTP 的 11.64 / 13.68？—— 非規格臂能，規格臂不能（2026-09-18 21:20–21:50）

> 主題：**兩個儀器在同一顆 checkpoint、同一形狀下，對 MTP 的增益給出相反的符號。**
> 產物：`Backup/bench_parity2_20260918/`（run.log、A*/B*/D1 的 json+txt、`run.sh`、`diag_accept.sh`、`run_a3a4.sh`）
> 未動 `src/`：本輪沒有程式碼改動，因此不需要 D5、也沒有 commit。

## §1 一句話

**HTTP 的 11.64 / 13.68 重建成功（11.73 / 14.13）。llama-bench 沒有重建成功 —— 而且問題不在「差幾個百分點」：
不開 spec 的那一側兩邊只差 6.6%，一開 spec 就差 23–27%，增益的符號是反的（HTTP +16~+23%、bench −3.7%）。**

## §2 為什麼要把 bench 的形狀從 `-d 512` 改成 `-d 2025`

之前所有 bench 對照都用 `--depths 512`。那**不是 server 的形狀**：
`http_duo` 的 prefill_plan 在 prod25 上算出的實測位置是「prefill **2025** token → 在 depth ~2025 解 128 個 token」。
而 llama-bench 的 tg cell 一律是 `n_prompt=0, n_depth=nd`（`llama-bench.cpp:1519` `/* .n_prompt = */ 0`），
填 `nd` 個 token 再解 `n_gen` 個 ⇒ **要對上 server，`nd` 必須是 2025**。深度差 4 倍時專家路由與快取狀態都不是同一個。

## §3 結果（全部 rep1 丟棄；HTTP 取上中位數，bench 取保留 rep 平均）

### HTTP（`http_duo.py --profile prod25 --reps 3`）

| arm | checkpoint | MTP | decode t/s | samples | 對照原記錄 |
|---|---|---|---|---|---|
| A1 | Q36（無 nextn） | off | **11.73** | [11.73, **6.04**] | 11.64 ⇒ +0.8% ✓ |
| A2 | Nail | on | **14.13** | [14.13, 14.03] | 13.68 ⇒ +3.3% ✓ |
| A3 | Nail | **off** | **11.52** | [11.52, 11.22] | —（新，拆混淆用） |
| A4 | Nail | on | **13.40** | [10.85, 13.40] | 同窗口重跑 A2 |

### llama-bench（`llama_bench_matrix.py --prompt 0 --depths 2025 --gen 128 --reps 3`）

| arm | checkpoint | spec | platform t/s | samples | hit% |
|---|---|---|---|---|---|
| B1 | Q36 | — | 8.06 | [8.45, 10.30, **5.82**] | 96.6 |
| B0 | Nail | — | **10.76** | [8.60, 10.83, 10.70] | 96.4 |
| B2 | Nail | draft-mtp k=3 | **10.36** | [8.06, 10.47, 10.26] | 58.6 |
| D1 | Nail | draft-mtp k=3（重跑 + `CGC_MTP_PERF`） | — | — | 62.5 |

### 並排

| | HTTP（server） | bench | bench / HTTP |
|---|---|---|---|
| Nail，MTP **off** | **11.52**（A3） | **10.76**（B0） | **−6.6%** |
| Nail，MTP **on** | **13.40–14.13**（A4/A2） | **10.36**（B2） | **−23% … −27%** |
| **MTP 增益** | **+16.3% ~ +22.7%** | **−3.7%** | **符號相反** |

## §4 被證偽的四個假設（每一個都花了量測才排除）

1. **checkpoint 混淆** —— `run_server.sh:153-161` 讓 `CGC_SERVER_MTP=0` 順帶把 `MODEL_DEFAULT` 換成 `$Q36`（另一顆沒有 nextn head 的 checkpoint）
   ⇒ 「11.64 → 13.68 的 +17.5%」可能是換模型換來的。**證偽**：A3（Nail + MTP off）= 11.52 ≈ A1（Q36 + MTP off）= 11.73，
   換 checkpoint 只值 **+1.8%**；而 A3→A4（同一顆 Nail，只差 MTP）= **+16.3%**。⇒ MTP 增益是真的。
2. **n_batch 沒對齊** —— 上一輪 B512 9.63 / B2048 5.51 看似指向 batch。**證偽**：2048 反而慢 43%，且 B2048 排在 B512 之後、
   HEAVY 樣本 45/152 vs 16/112 ⇒ 那一格是自我加熱，不是 batch。
3. **溫度／接受率** —— `common_params_sampling::temp` 預設 **0.80**，而 server 端 `http_duo` 送 `temperature: 0.0`（greedy）。
   這是文件 §8 具名的缺口，本來是最強的嫌疑。**證偽**（本輪新量）：
   | | emit tok/round | acc tok/round | acc rate |
   |---|---|---|---|
   | server A2 | 2.40 | 1.40 | 0.465 |
   | bench D1 | **3.404** | **2.404** | **0.801** |
   bench 的接受率**比 server 高 72%**，不是低。⇒ 接受率不是病因，`--temp` 這個修法方向可以關掉。
4. **verify 掉回 exact** —— bench stderr `CGC-SYNCFILL: verify` 166 次 / verify calls 6695 = **2.5%**；
   server 286 / 6930 = 4.1%（server 反而更多）⇒ 不是這條。

## §5 真正落點：規格臂的「每輪成本」

用兩邊自己印的計數器換算（同一量：每輪 emit 幾個 token、每輪多久）：

| | emit/round | rounds | t/s | **ms/round** | ms/token |
|---|---|---|---|---|---|
| server（A2） | 2.40 | 53（=128/2.40） | 14.13 | **170** | 70.8 |
| bench（D1/B2） | 3.404 | 38（=128/3.404） | 10.36 | **325** | 95.5 |

- bench **少做 28% 的輪數**，卻多花 36% 的時間 ⇒ 每輪貴 **1.9×**。
- 而 bench 每輪的專家聯集**更小**（`union/call` 16.74 vs server 18.80）⇒ **不是「搬得多」，是「每輪的固定成本高」**。
- 這解釋了為什麼 bench 的 MTP 不賺：輪成本隨 emit 數幾乎線性成長（95.5 ms/token ≈ 不開 spec 的 92.9 ms/token），
  一次驗證 4 個 token 的價錢約等於 3.4 次單 token 步 ⇒ **攤薄發生不了**。

**最一致的機制（尚未直接證實，已具名）**：IO／compulsory miss 主導。bench 的 missing 結構與 server 不同：

| | compulsory | capacity | compulsory 佔比 | 每 request 的 compulsory |
|---|---|---|---|---|
| server A2 | 3736 | 4211 | 47.0% | 0.180 |
| bench D1 | 5307 | 1391 | **79.2%** | **0.297（+65%）** |

原因候選：**bench 的 depth 填充用 `std::rand() % n_vocab` 造 token（`llama-bench.cpp:2321-2323`），
而 server 每一輪 prefill 的是**同一段真實 prompt**。**⇒ server 的 rep2/rep3 prefill 幾乎全命中，bench 每一輪都在首次觸摸新專家。
配合 `prefetch` 計數（server 502 發出 / bench 374，且 bench 有 `drain_cleared=68` 被丟棄），
「驗證批次多做工卻不省時間」與「spec 不賺」是同一個原因的兩面。

## §6 兩個順手發現的正確性問題（與本題無關但別再被騙）

1. **`med()` 在只有 2 個保留 rep 時回傳「較大者」**（`http_duo.py:512` `sorted(vs)[len(vs)//2]`）。
   A1 的 [11.73, **6.04**] 與 B1 的 [8.45, 10.30, **5.82**] 都是雙峰，而 `med()` 靜默挑走沒崩的那個
   ⇒ **記錄中的 11.64 與今天的 11.73 都可能遮蓋了一個會崩到 6 的分佈。** 這個口徑要修。
2. 兩支使用 **Q36 checkpoint** 的臂（A1、B1）都出現「最後一個 rep 崩掉」，而兩支 Nail 臂都穩定
   ⇒ 崩潰跟著 checkpoint 走。樣本太小，僅記錄。

## §7 若要真的讓 bench 重建 HTTP，下一步只有一條

不要從 bench 的參數下手（本輪已排除 batch、溫度/接受率、phase、exact 回退）。
要做的是**讓 bench 的 token 流連貫**：depth 填充與 decode 用同一段 token 序列、且跨 rep 重複
（現在是每輪全新隨機）。最小實驗：
在 `test_prompt` 的 depth 填充前 `srand(0)`（每個深度填充重播同一序列），看
① `miss_compulsory` 是否落到 server 的量級（0.18/request）、② spec 臂是否開始比 no-spec 快。
這一格會直接裁決 §5 的機制假設；若它不成立，就得動用 phase 級儀器（`CGC-DECPROF`）把 170 ms vs 325 ms 拆到各階段。

> **★ 更正（同日 22:00，讀碼後）**：本節原本寫「depth 填充**每輪全新**」——**那句是錯的**。
> `llama-bench.cpp:2971-2981` 有 `is_cached`：`t.n_depth == cstate.depth` 時走
> `llama_state_seq_set_data()` **還原 KV state**，depth 填充只在 rep0 跑一次。
> 所以「跨 rep 重複」這件事在 **KV 層**上本來就是重複的。
> 真正被 §9 證明的是**另一件事**：那份被還原的 state 只涵蓋 **target**，draft context 從來沒拿到過前綴
> （`ctx_dft pos_max = -1`）。§7 提議的 `srand(0)` 因此**不是**正確的修法方向，已被 §9 取代。


## §8 沒做的（具名）

- 沒做**同窗口**的 bench vs HTTP 交錯（A 組 21:22–21:46、B 組 21:27–21:38 有重疊但不交錯）
  ⇒ 表中的百分比仍含時段效應；今天同 config 跨時段可差到 1.4×。
- 沒動 `src/`，所以 `--temp` 這個旋鈕**沒有實作**（§4.3 已說明不需要）。
- D1 是單次重跑，沒有重複；B 組每臂只有 3 個 rep。
- bench 的 `-b 512` 與 llama-server 的預設 batch **沒有對齊**（`run_server.sh` 對 prod25 不傳 `-b`）；
  本輪沒有再驗一次 batch，因為上一輪已量過且方向相反。

---

# 第二部分：兩條修復（同日 21:54–22:35）

## §9 ⑤ 修復：`http_duo.py` 的聚合口徑

**缺陷**：`med(axis)` 是 `sorted(vs)[len(vs)//2]`，即**上中位數**。n=2 時它是 `vs[1]` ＝**兩個保留 rep 中較大者**。
`--reps 4`（3 保留）時索引剛好落在中位數，所以缺陷只在 `--reps 3` 這條捷徑上發作 ——
而**記錄中的 11.64 / 13.68 本身就是 `--reps 3` 跑的**。

**修法三件**（缺一不可）：
1. **真中位數**：n 為偶數時取中間兩數的**中點**，n=2 不再被特例成 max。
2. **`MIN_KEPT = 3`**：兩個樣本定不出中位數。低於 3 個保留 rep ⇒ 直接標記不可引用（工具的預設 `--reps 4` 不受影響，只有捷徑被標）。
3. **`SPREAD_LIMIT = 1.10`**（max/min），**門檻是量出來的、不是選的**：
   當天所有乾淨的 3-rep 運行落在 **1.008–1.034**（V3 1.020、A3 1.027、A2 1.008、歷史 off 臂 1.034）；
   所有**不乾淨**的落在 **1.207 及以上**（V4 1.207＝熱降級窗，prefill 掉到 90.85 t/s；A1 1.941）。
   1.05–1.15 之間是空帶，取中間。**初稿用 1.25，結果 V4 穿過去了** ——
   那正是這個守門要防的失效，所以把數字寫進註解而不是留成品味。

**驗證**（用已記錄的臂重放 + 端到端重跑）：

| arm | kept samples | spread | 可引用 | 理由 |
|---|---|---|---|---|
| A1 | [11.73, 6.04] | 1.941 | ✗ | rep 數 2 |
| A2 | [14.13, 14.03] | 1.007 | ✗ | rep 數 2 |
| A3 | [11.52, 11.22] | 1.027 | ✗ | rep 數 2 |
| A4 | [10.85, 13.40] | 1.235 | ✗ | rep 數 2 |
| **V3** | **[11.23, 11.17, 11.01]** | **1.020** | **✓** | OK |
| V4 | [10.69, 12.90, 10.86] | 1.207 | ✗ | spread ≥ 1.10 |

⇒ **今天唯一可引用的 HTTP 數是 V3 = 11.17 t/s（Nail checkpoint、MTP off、3 保留 rep）。**
MTP-on 那一側（V4）落在熱降級窗 ⇒ **待重跑**。`--self-test` PASS、`py_compile` PASS。

## §10 ④ 修復：找到了根因，也量到了一次真實的 3.3× 回歸

### 10.1 先修的是「儀器不會說話」

`common_speculative_impl_draft_mtp::begin()`（`common/speculative.cpp:1455-1471`）開頭是

```cpp
const int32_t N = (int32_t) prompt.size();
if (N <= 0) { return; }
```

而它的**整個函式體只有一條警告**：`pos_max(ctx_dft) < N-1` ——
即全樹**唯一**在問「target 的 prefill 有沒有在每個 ubatch 送到 draft context」的檢查。
server 餵真 prompt（`server-context.cpp:4076`），bench 餵**空 vector** ⇒ 那條檢查在**每一次** bench 運行裡
都在 `:1457` 直接返回，**永遠不可能響**。這是 `verify: calls=0` 的孿生兄弟：**儀器沉默，因為從來沒人問它。**

修法：`llama-bench.cpp` 加 `prompt_probe`，大小 = `n_past`（＝context 已消耗的 token 數），**只餵 `begin()`**；
值不被讀（`begin` 只用 `.size()`），draft params 的那個遺留 `prompt` 欄位保持原樣
（`common/speculative.h:61` 自己標了「remove in the future」）。+37 / −1 行。

**它立刻說了話**：

```
spec begin: ctx_dft pos_max=-1 < N-1=2024 - process() hook may not have run on every
            prefill ubatch (need_embd / logits=1 on every prompt position?). Drafts may degrade.
```

`pos_max = -1` ⇒ **draft context 一個位置都沒有**。MTP head 一直是對著空前綴在起草。

### 10.2 根因（可指名）：prefill 從來沒餵給 speculator

| 誰 | `common_speculative_process` 的呼叫點 |
|---|---|
| server | `server-context.cpp:766`（mtmd）／`:3962`（文字） |
| 參考實作 | `speculative-simple.cpp:378`（`batch_enc`）／`:520` |
| **llama-bench** | **只有 `:2697`（verify）** —— prefill 一次都沒有 |

而 qwen35moe 上 `is_mem_shared` **按設計是 false**（`speculative.cpp:1413` 由
`llama_get_ctx_other(ctx_dft) == ctx_tgt` 決定，而 `llama-context.cpp:155/164` 只把
`params.ctx_other` 傳播給 gemma4 那兩個 arch，其餘在 `:155` 被清成 `nullptr`）
⇒ 沒有共享 KV ⇒ `process()` 的 catch-up decode 是**唯一**讓 MTP head 看到前綴的途徑。

### 10.3 照 server 那樣補上，結果是 3.3× 回歸（**已回退**）

修法：`test_prompt()` 加可選的 `common_speculative *`，每個 prefill chunk 都餵一次；
餵的 batch 必須另建（`llama_batch_get_one` 的 `n_seq_id`/`seq_id` 是 `nullptr`，
而 `draft_mtp::process()` 在 `:1505` 有 `GGML_ASSERT(batch_in.n_seq_id[k] == 1)`，直接餵會 abort）。
兩個 depth/prompt 呼叫點傳入、warmup 不傳。

| 量 | 補之前（V1） | 補之後（W1） |
|---|---|---|
| `ctx_dft pos_max` 警告 | 響（-1 < 2024） | **消失 ✓** |
| tg samples | [7.62, 10.03, 12.54, 8.81] | **[2.56, 3.90, 2.68, 2.98]** |
| platform | 10.46 | **3.19（−70%）** |
| `CGC-CPU-SPLIT` avg | 52.3 us/split | **1212.3（23×）** |
| `CGC-COLD-GUARD` | 57 | 85 |

判準①（draft 拿到前綴）**通過**，但代價是 decode 崩掉 —— **所以回退**（`git checkout` 後只重貼 `prompt_probe`，+37/−1）。
回退確認（相鄰窗口，W1b）：samples [6.91, 6.24, 7.20, 8.63]、platform **7.36**、警告仍在。
⇒ 補上那一次呼叫讓 bench 把「該付的 catch-up 代價」一次付清（2025 token × 全鏈），
而它在 server 上之所以不是問題，是因為 server 的 prefill 本來就在同一條路徑上（110–116 t/s 已含此成本）。

### 10.4 這一節的結論

- **④ 的前半（讓儀器開口）已修好並保留**：bench 現在會自己報告 draft context 的狀態。
- **④ 的後半（讓路徑真的對齊）沒有任何一行留在樹上**：一次呼叫不夠，draft 的 KV 生命週期與
  per-rep 的 `llama_state_seq_set_data`（只還原 target）必須一起處理，否則就是把代價從「隱形」變成「可見但更貴」。

## §11 狀態與待辦

- **已改動**：`scripts/check/http_duo.py`（⑤，全部驗證通過）、
  `src/llama.cpp/tools/llama-bench/llama-bench.cpp`（④ 的診斷，+37/−1，已重建）。
- **D5**：`src/` 有改動 ⇒ 必須跑；`--tag en-mtp-prompt-probe` 的輸出見 `Backup/bench_parity2_20260918/d5.log`。
- **未 commit**（使用者未要求；樹上另有別條線的 `scripts/check/*.py` 未提交修改，不予觸碰）。
- **待辦（具名）**：
  1. **V4 重跑**：HTTP MTP-on 在同一個乾淨窗口與 V3 配對 —— 現在 MTP-on 側沒有任何可引用的數。
  2. **④ 後半**：把 `common_speculative_process` 的 prefill feed 與 draft 的 KV 生命週期
     （每 rep 清 draft memory 並重餵，或讓 depth-state 快取同時涵蓋 draft）一起設計，
     再量一次「spec 是否終於比 no-spec 快」。**不要在沒有 draft 生命週期配套的情況下單獨補那一行。**
  3. `draft_mtp::begin()` 現在會響了，但**只有 bench 會被它叫醒**；server 那側的 `N` 來自真實 prompt，
     語意不同（`n_past` vs prompt 長度）——若日後要把它當成通用閘門，兩個數的定義要先统一。


---

## 9. 定案：HTTP 目標已釘死，④ 後半被量測否決（2026-09-18 23:20–23:53）

### 9.1 HTTP 一對終於可引用（同窗口、`--reps 4`）

| 臂 | checkpoint | MTP | decode t/s | samples（丟 rep1） | spread | 可引用 |
|---|---|---|---|---|---|---|
| V3b | Nail | off | **10.93** | [11.00, 10.93, 10.75] | 1.023 | ✓ |
| V4b | Nail | on  | **12.99** | [12.96, 13.07, 12.99] | 1.009 | ✓ |

⇒ **HTTP 淨增益 = 12.99 / 10.93 = +18.9%**。這一對終於同時滿足 `MIN_KEPT=3` 與
`SPREAD_LIMIT=1.10`，是本文第一次能直接引用的 MTP 增益。它坐在此前軟數字的中間
（A3→A4 +16.3%、A2 13.68 vs A1 11.73 +20.4%），所以**「MTP 在 HTTP 上確實有 ~19% 增益」可以定案**。

### 9.2 ④ 後半（prefill feed + draft KV 生命週期）＝ 回歸，且比第一次更糟

`run_all_when_free.sh` 在同一個熱窗內跑完：V3b → V4b → 建置 → W2/W3。

| 臂 | 說明 | samples | platform | CPU-SPLIT | COLD-GUARD |
|---|---|---|---|---|---|
| W2 | spec（④ 後半全上） | [4.44, 7.23, 3.58, **1.36**] | **4.06** | **2686.2** us/split | 57 |
| W3 | nospec（同 binary） | [7.70, 8.39, 8.32, 8.62] | 8.44 | 214.2 us/split | 0 |

- 判準①通過：`ctx_dft` 警告消失（draft 真的拿到前綴）、`process` 失敗 0。
- 判準②**慘敗**：bench 淨增益 **4.06 / 8.44 = −52.0%**；spec samples 離散到 **5.3×**
  （1.36 … 7.23），連「單一讀數」都不成立。
- 與第一次嘗試（只補 feed，不補生命週期：3.19）相比**更慢** ⇒ 生命週期那一半沒有救回來。

**已回退**；這份嘗試留檔為 `Backup/bench_parity2_20260918/llama-bench.cpp.VARIANT_B_prefillfeed_lifecycle`。
樹上只剩診斷那一半（`prompt_probe`，+36/−1）。

### 9.3 兩次嘗試合起來說明什麼

| 嘗試 | spec platform | vs no-spec | 結論 |
|---|---|---|---|
| 不補（原狀，V1） | 10.46 | −1.5% | 基線 |
| 只補 feed（W1） | 3.19 | — | −70% |
| feed + 生命週期（W2） | 4.06 | −52% | 更糟 |

**「把 draft context 餵飽」在 bench 上是純成本，不是收益。** 可指名的機制：
qwen35moe 上 `is_mem_shared` 為 false（`llama-context.cpp:155/164` 只對 gemma4 傳遞 `ctx_other`），
所以每次 `process()` 都要做一次 catch-up decode；而 bench 的「前綴」是
`std::rand() % n_vocab` 造出來的亂 token（`llama-bench.cpp:2321-2323`），
**MTP head 對著亂 token 前綴起草本來就不會有額外接受率**，於是多付的 catch-up 一個子兒也賺不回來。
server 餵的是真文本，所以那筆成本在那裡是划算的。

⇒ **路徑不對稱是真的，但它不是 bench 缺的那 25% 的解法。** 剩下的差異在「餵進去的是什麼」
（真文本 vs 亂 token），不在「走哪條路」。要驗證只能讓 bench 餵真 token——那是另一個形狀的改動，
不是路徑對齊。

### 9.4 未結事項（具名）

1. **W3 (nospec) 8.44 vs V2 (nospec) 10.62 差 −21%，未解釋。** 回退後的程式碼在 nospec 路徑上
   與 HEAD 逐位元相同（`feed_spec` 為 false、`is_cached` 只在 `cgc_spec_on` 時變），所以理論上
   只能是環境（W3 緊接在 CPU-SPLIT 2686 的 W2 之後只等了 45 s）。**未證實。**
2. **回退後的重跑（X1b/X2b）沒有拿到數。** 第一次 X1/X2 以 `CGC-METAL-FAIL: Insufficient Memory`
   abort；第二次以 `ggml_metal_library_init: XPC_ERROR_CONNECTION_INVALID (is the OS shutting down?)`
   abort。同時 `notifyutil` 回 `Failed with code 8`、`pgrep` 回 `sysmond service not found`
   ⇒ GPU/系統服務層被佔（另一條線又起了一輪 `spec_cost_curve`，2 users、load 4–6.7）。
   **nospec 錨點沒有在回退後的 binary 上重新量到。**
3. 因此 9.3 的結論建立在「W2 vs W3 同窗口」這個內部對照上（4.06 vs 8.44），
   **它不依賴跨窗口比較**，所以不受第 1、2 條影響。
