# MTP is now measurable inside the instrument of record

2026-09-18 00:20–00:35 · 引擎層線（本線）· `src/llama.cpp/tools/llama-bench/llama-bench.cpp`

## TL;DR

`llama-bench --spec-type draft-mtp` 在**生產 cell 形狀**（`-p 0 -n 128 -d 512 -b 512`、8 GiB pool、
`prefill250` profile）上**現在能跑完**：`rc=0`、印出完整的 `CGC-MTP-PERF` 讀數、沒有任何 verify 失敗。

修的是**一個初始化**：

```cpp
int n_past = (int) llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id) + 1;
```

在此之後，**所有 `mtp=1` 的 profile 都有一個 instrument of record**（先前 22:20 的 `--spec-type` CLI
只修好了介面，生產形狀仍然起不來）。

## 一、它以前為什麼跑不起來

症狀連續三輪逐字相同（`--spec-type` 那版與我 09-17 的 env 那版都一樣）：

```
test_gen_spec: verify decode failed: ret=-1 n_tokens=4(pos 0..3) n_ctx=768 n_past=1 draft=3 n_batch=512
llama_bench: error: failed to run gen
CGC-M2-UNREPOINT: teardown restored 120 expert tensor(s) to their model storage before freeing 6 slab(s)
                  (a second context built from this model would otherwise read a freed buffer)
```

四個觀察把它逼到**一個變數**上：

| 形狀 / 條件 | 結果 |
|---|---|
| `-n 16 -d 0` | **成功** |
| `-n 128 -d 512` | 失敗 |
| `-n 128 -d 512 --no-warmup` | 失敗（逐字相同） |
| `-n 128 -d 512` ＋ 把 draft context 的建立**提前**到 warmup 之前 | **仍然失敗（逐字相同）** |

⇒ 觸發條件是 **`-d 512` 的 depth prefill**；**不是** warmup，**也不是** draft context 的建立時機
（最後一列是本輪用一個真的建置與一次真的執行推翻的假設）。

**真因**：`test_gen_spec` 的 batch 位置是**明確指定的**（`common_batch_add(batch, id_last, n_past++, …)`），
而 `n_past` 被初始化為 **0**。但同一個 rep 裡 **depth prefill 已經寫過 pos 0..511**
（llama-bench 的順序是 depth → prompt → gen），所以第一個 verify batch 要求的位置**已被佔用**，
`llama_decode` 回 **-1** —— 而它**不印任何訊息**（這就是為什麼症狀看起來像「Metal／記憶體／兩個 context」）。

**對照組解釋了為什麼只有這條路徑踩到**：非 spec 路徑的 `test_gen` 用 `llama_batch_get_one()`，
由 llama 自己分配位置 ⇒ 自動接在 depth 之後。只有新寫的 spec 路徑自己管位置。

## 二、兩個修正

### 1. `n_past` 從 context 的實際位置開始 ✅ **這是真因**

```cpp
int n_past = (int) llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id) + 1;
```

- 空 memory 回 **-1** ⇒ `n_past = 0` ⇒ **沒有 depth run 的形狀行為逐位元不變**（短形狀回歸驗證通過：rc=0）。
- 有 depth run ⇒ `n_past = 512` ⇒ 正確接續。
- 順帶修好了 `-p > 0` 的形狀（prompt run 之後 gen 也會接在正確位置）。

### 2. draft context 的建立提前（`bench_spec_setup`）⚠️ **不是真因；保留並已標註**

原本 draft context 在 `test_gen_spec` 內部建立（＝在 depth prefill 之後）。現在由呼叫點在
**target context 建好之後、warmup 之前**建立，與 llama-server 的順序一致。

本輪量到：**把它提前並沒有修好任何東西**（上面表格最後一列）。保留的理由是它**與 server 同序**、
且在「池乾淨時才建第二個 context」是防禦性的。**未經隔離的效能驗證** —— 若日後發現 MTP 臂的效能異常，
第一個要隔離的就是這個順序。程式碼裡的註解已逐字記下這個負結果。

## 三、第一個 instrument of record 上的 MTP 讀數

`prefill250` decode cell 形狀、8 GiB pool、`forward_argv(resolve(...))` 組的 argv、A/B/A 交錯：

| 臂 | rc | `tg128 @ d512` | 樣本 |
|---|---|---|---|
| `off1` | 0 | **10.049** ± 1.181 | 8.699 / 10.555 / 10.892 |
| `on`（`--spec-type draft-mtp --spec-draft-n-max 3`） | **0** | **7.715** ± 0.884 | 7.526 / 8.679 / 6.942 |
| `off2` | 0 | **6.791** ± 0.077 | 6.716 / 6.871 / 6.787 |

`CGC-MTP-PERF`（第 3 輪累計 ＝ 3 個 rep 全部）：

```
calls_begin=3 calls_draft=118 calls_accept=118  gen_tokens=354  acc_tokens=273
t_begin_ms=0.6  t_draft_ms=3482.9  t_accept_ms=5.5
acc_rate=0.7712  gen_tok_per_round=3.000  acc_tok_per_round=2.314
emit_tok_per_round=3.314  ms_per_round=29.516
```

**計數器（穩健，不受時間漂移影響）**：

| 量 | 值 | 說明 |
|---|---|---|
| **accept** | **77.12%** | 273 / 354 |
| **emit** | **3.314 tokens/round** | 1 + 2.314 |
| 草稿長度 | 3.000（用滿 `n_max=3`） | |
| draft head | **29.5 ms/round** | 3482.9 ms / 118 rounds |
| 接受判定成本 | 5.5 ms 總（可忽略） | |

⚠️ **這組 accept 用的是 llama-bench 的預設取樣**（`common_params_sampling`：temp **0.80**、top_k 40、
top_p 0.95、min_p 0.05），**與 served 路徑量到的 38.5% 不可比**（不同 prompt／不同取樣）。
任何「accept ≥ 60%」的判定都必須註明是在哪一組取樣下量的。

## 四、限制：**這次不能判 MTP 的淨 t/s**

控制臂 **10.049 vs 6.791 ＝ 差 48%**（而判準是幾個百分點）。用中位 8.420 當分母誠實嗎？不誠實 ——
兩個控制臂的差距已經大於任何可能的 treatment 效果。

**漂移的來源被池統計排除**：

| 臂 | hit rate | resident | file_reads | bytes | us/job | eff rate |
|---|---|---|---|---|---|---|
| `off1` | 96.6% | 6153.70 MiB | 26925 | 4.88 GB | 20088 | 9 MiB/s |
| `on` | 93.4% | **6430.62 MiB** | **43488** | **11.05 GB** | 14122 | 18 MiB/s |
| `off2` | 96.7% | 6153.70 MiB | 26685 | 4.79 GB | 15843 | 11 MiB/s |

`off1` 與 `off2` 的池行為**幾乎相同**（hit 96.6/96.7、file_reads 26925/26685、bytes 4.88/4.79 GB）
卻差 48% ⇒ **漂移是機器外部狀態**（熱／記憶體壓力／鄰居），不是池、也不是 MTP。
（收尾時 `swap used` 12.5 GB / 13.3 GB ＝ 94%，熱狀態 0 ⇒ 是記憶體壓力那一類的歷史效應。）

**算術可以給一個有支撐的界**：每輪 emit 3.314 tokens、verify 批次 4 tokens、draft head 29.5 ms。
若 verify 的成本像單 token step 一樣**線性**（沒有批量紅利），淨比 ≈ 3.314/4 ＝ 0.83，再扣 draft 開銷
⇒ **約 −20%**。用 `off1` 當分母量到 −23% ⇒ 與這個算術一致。但**這仍然是算術，不是配對實驗**。

⇒ MTP 淨效果的定案需要一個**更乾淨的窗口**（短的臂、更多交錯、或把每次執行縮短）。
**本節的數字是「修好了、並拿到讀數」，不是「MTP 值不值得開」。**

## 五、MTP 的真實副作用（這一節反而比 t/s 可靠）

`on` 臂與兩個 `off` 臂的池統計差異是**系統性**的（不是漂移，因為兩個 off 臂彼此一致）：

| 量 | off（兩臂一致） | on | 變化 |
|---|---|---|---|
| `file_reads` | ~26800 | 43488 | **+62%** |
| `bytes` | ~4.83 GB | 11.05 GB | **+129%** |
| `hit rate` | 96.6–96.7% | 93.4% | **−3.2pp** |
| `resident` | 6153.70 MiB | 6430.62 MiB | **+277 MiB** |

⇒ 投機在**同一個 8 GiB 池**上多要了 **6.2 GB 的磁碟讀取**（＝每 token 多約 48 MB）。
這是 M4 的「verify 批量紅利」之外的第二個成本項，而且它與 **M3 的池路徑**直接相關。

## 六、可引用性與未做的事

- **`--cells decode` 的既有數字不受影響**：`cgc_spec_on == false` ⇒ 走原本的 `test_gen`
  ⇒ 這兩個修正**不會**改變任何已登記的 cell 的數值（靜態可證，不需要重跑）。
- **未跑 D5**。理由：本輪 `libllama` / `llama-server` / `libggml-metal` 三個產物**逐位元不變**
  （只重建 `libllama-bench-impl.dylib`），而 D5 測的是引擎層的 logits oracle，**不經過 llama-bench**。
  ⚠️ 但依閘門鏈，**若要把這兩個修正 commit，D5 必須跑**。
- **新 cell 未登記**：`--spec-type draft-mtp` 目前是**模式**而非 `prod_matrix.py` 的 cell
  （刻意如此：`decode` cell 必須與上游 `tg128 @ d512` 逐位元可比，不能預設長出 spec 分支）。
  正式登記（含「它不能與誰並排」）屬於 `prod_matrix.py` ＋ `docs/PROD_MATRIX_STANDARD_*.html`。
- **未 commit**。`llama-bench.cpp` 現在是 **+507/−1**（別條線 22:20 的 +403/−1 ＋ 本輪的位移與 pos 修正）。
  同一個檔案上還有他們的未定稿。

## 七、產物

`Backup/cgc_logs/en_lb_mtp_20260918/`：

| 檔 | 內容 |
|---|---|
| `off1.stdout.json` / `on.stdout.json` / `off2.stdout.json` | 三臂的 llama-bench JSON |
| `off1.stderr.log` / `on.stderr.log` / `off2.stderr.log` | 完整 stderr（含 `CGC-MTP-PERF`、池統計） |
| `llama-bench.cpp.before-order-fix` | 修正前的檔案（別條線 +403/−1 的版本） |
| `en_build_gate2.sh` | 本輪用的建置閘門（`pgrep -x` ＋ 字元類，避免 self-match） |

分析用：`/tmp/en_lb_mtp_aba.py`（A/B/A 驅動）、`/tmp/en_fix_order.py`（帶斷言的 patch 腳本）。
