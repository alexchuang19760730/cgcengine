# 環境定價：配置等價性 ＋ 記憶體作為參數 —— 2026-09-18

回答 `docs/HTTP_VS_BENCH_CALIBER_2026-09-18.md` §7 自己留下、且明說未查的兩件事。
工具：`scripts/check/caliber_env.py`（`--equiv` / `--model-identity` / `--memory`）。

**一句話：兩個答案都不是那份文件假設的那個。**

1. **「我們的工具跟 `run_server.sh` 一樣」不成立** —— `batch` 差 11 倍（512 vs 5632），
   所以 +30% **現在還不能**歸因於環境。
2. **把記憶體當參數分析後，記憶體解釋不了那個 ±1.9 t/s** —— 所有指標在 decode 單 cell 內
   `|r| = 0.18–0.32`，全部低於臨界值 0.705。

---

## 1. 同臂驗證：工具 vs run_server.sh（零 GPU，可重跑）

```sh
python3 scripts/check/caliber_env.py --equiv --profiles prefill250 --cells decode
```

| knob | run_server.sh | 我們的工具 | verdict |
|---|---|---|---|
| **`-b` / batch** | **5632** | **512** | **MISMATCH** |
| **`-ub` / ubatch** | **5632** | **512** | **MISMATCH** |
| `-c` / ctx | 8192 | n/a（llama-bench 沒有這個概念） | forwarded |
| `-m` / model | `Nail-…-denseIQ4X.gguf` | 同上 | MATCH |
| `-expert-cache` | 8589934592 | 同源 `resolve()` 轉發 | forwarded |
| `--cache-type-k/-v` | q8_0/q8_0 | 同源轉發 | forwarded |
| `-ngl` / `-t` | 99 / 8 | 同源轉發 | forwarded |
| `--spec-type` | `draft-mtp n_max=3` | 只有 `decode-spec` cell 轉發 | BY-DESIGN |

### 根因：不是 `resolve()` 的錯，是 cell 定義蓋掉 profile

`llama_bench_matrix.default_batch()` **有**正確地優先採用 profile 自己的 `BATCH/UBATCH`
（prefill250 = 5632/5632）。但 `prod_matrix.cell_command()` 先執行

```python
b = ub = spec["batch"]          # prod_matrix.py ~line 319
```

只要有宣告 batch 的 cell，profile 就被蓋掉。`decode` cell 宣告了 512 ⇒ 工具跑 512。

⇒ 那份文件 §7 的「`-b 512` 與 server 自身 batch 設定是否等價：未查」現在的答案：
**不等價，差 11 倍。** 這是比較兩側唯一已知會動的旋鈕，在它被對齊之前，
「差異在環境」是**未被證明**的命題。

### 順手排掉一個嫌疑：`CGC_SERVER_MTP=0` 換的那份模型其實是同一個 bytes

```sh
python3 scripts/check/caliber_env.py --model-identity
```

```
      size+windows = 13663116512/123e09606616/836d5d3ee021/b75774c4c065/a15dd6c03603/a609243f0881
Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
      size+windows = 13663116512/123e09606616/836d5d3ee021/b75774c4c065/a15dd6c03603/a609243f0881
```

**同一份內容**（size 與 5 個視窗全等）⇒ 「MTP=0 換模型檔」在本機**不成立**，
模型檔不是混淆變數。（這也回溯修正我今天早先那句「MTP=0 不是同一受試對象」。）

### 但它也不是沒差：MTP off 會讓 8 個 engine env 整塊消失

```
CGC_DRAFT_DECODE=1   CGC_MM_BITIDENT=1   CGC_MTP_NO_WARMUP=1   CGC_NO_PREFETCH=1
CGC_NO_SEQ_RM_PROBE=1   CGC_VERIFY_DECODE=1   CGC_WARM_NPAST=0
LLAMA_EXPERT_CACHE_LAYER_CAPS=40-40:256
```

⇒ **「MTP off」是一整組引擎旋鈕，不是一個旗標。**任何拿 MTP on/off 做的 A/B，
都要假設這 8 個也可能動了，不能只歸給 spec。

---

## 2. 記憶體環境作為參數來分析

```sh
python3 scripts/check/caliber_env.py --memory --cell-filter decode \
    --json-glob 'Backup/prod_matrix/bisect_*.json' --json-glob 'Backup/prod_matrix/duo_*.json'
```

| arm | t/s | worst | swap_used MiB | usable GiB | inactive GiB | anon GiB |
|---|---|---|---|---|---|---|
| A1 old | 10.80 | NOMINAL | 14098 | 6.9 | 4.0 | 7.4 |
| B2 new | 10.34 | MODERATE | 13650 | 7.7 | 4.1 | 7.8 |
| duo | 9.12 | HEAVY | 16396 | 9.5 | 1.8 | 2.4 |
| A2 old | 8.92 | HEAVY | 17263 | 9.9 | 1.0 | 1.3 |
| B1 new | 8.58 | HEAVY | 11722 | 6.4 | 6.2 | 10.7 |
| duo mtp0 | 8.56 | HEAVY | 15404 | 7.7 | 3.6 | 6.5 |
| B3 new | 7.74 | NOMINAL | 13702 | 6.8 | 4.9 | 6.9 |
| B4 new | 7.03 | HEAVY | 16802 | 9.2 | 1.3 | 2.0 |

decode 單 cell、n=8，雙尾臨界 `|r| ≈ 0.705`：

| 指標 | r | |
|---|---|---|
| `inactive_gib` | 0.182 | 低於臨界 |
| `anon_gib` | 0.315 | 低於臨界 |
| `cached_gib` | 0.217 | 低於臨界 |
| `swap_used_mib` | **−0.256** | 低於臨界 |
| `free_gib` | −0.258 | 低於臨界 |
| `usable_gib` | −0.197 | 低於臨界 |

### ★★ 要挑剔自己的地方：那個「強相關」是假的

**不**加 `--cell-filter` 時，同一批資料會給出 `inactive_gib r = +0.628`、`anon_gib r = +0.596`、
`swap_used r = −0.505`，看起來很漂亮。那是假的 —— 它把 **prefill 行（292 / 227 t/s）**
和 decode 行（7–11 t/s）放在同一條軸上，兩個不同量級的量混起來自己就產生了相關。
限定單 cell 後全部塌到 0.18–0.32。

⇒ **含校的教訓：後設相關必須限定在同一個量。** 混 cell 會自動製造 r。

### 熱態也排掉了

decode 八臂依臂內 `worst` 分組：

| worst | n | mean | 組內 spread |
|---|---|---|---|
| HEAVY | 5 | 8.44 | **1.30×** |
| MODERATE | 1 | 10.34 | — |
| NOMINAL | 2 | 9.27 | **1.39×** |

全體 spread 1.54×，而**組內就有 1.30–1.39×** ⇒ 組間差小於組內差，熱態解釋不了。

### 結論：目前沒有任何已記錄的變數能解釋那個散佈

記憶體（swap/free/inactive/anon/cached）、熱態（臂內 worst）、引擎版本（binary bisect 六臂）
—— **三個都排不出順序**。這正是必須走**配對設計**而不是後設分析的原因：
配對比值抵銷的是那個未知的慢漂，不需要事先知道它是誰。

---

## 3. 下一步（照優先順序）

1. **對齊 batch**（一個改動，就在 `prod_matrix.py` ~line 319）—— 讓 cell 能繼承 profile 的
   `BATCH/UBATCH`，或直接加一列在 `--equiv` 表上顯示的「profile-batch」cell。
   做完再比一次；若 +30% 縮小 ⇒ 部分是配置問題；若不變 ⇒ 才有資格說「差異在環境」。
2. **`paired_ab.py --null`** —— 先量儀器噪音底。同 binary 若自身散 ±10%，
   則「疑似 −15%」的正確答案是**不可測**。
3. 只有這兩步都過了，「+30% 是環境造成的」才成為可引用的述句。

## 4. 不可引用

- **「差異在環境」** —— 未證（batch 未對齊）。
- **「記憶體造成 bench 掉 17%」**（caliber 報告 §7 的措辭）—— 未證，且本資料不支持線性關係。
- caliber 那四臂**完全沒有記憶體讀數**（工具沒記）⇒ 它自己的產物**不可能**裁決它自己的懷疑。
  要把記憶體當參數，就必須每臂都量（`caliber_env.py --memory` 現在會把這種臂列出來）。

## 5. 改動檔案

- 新增 `scripts/check/caliber_env.py`（三種模式皆可重跑，`--cell-filter` 必用）
- 新增 `scripts/check/paired_ab.py`（本輪稍早；preflight 會 abort + 列出 pid，
  `--min-headroom-mb` 把記憶體狀態設成 enforced 條件，`--null` 量噪音底）
