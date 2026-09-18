# prefill 的 250 t/s 與 prod25 的 8-token clamp（2026-09-18，生產級實測）

**一句話**：**250+ 沒有消失。** 上一輪在 `decode_sweep` 表上看到的 `prefill ~10 t/s` 不是引擎退步，
是 **`prod25` 這個 profile 的 prompt 處理 regime**；它與 `prefill250` 的 250+ 不是同一件事。

## 1. 根因：`CGC_PREFILL_STREAM` 只存在於 prefill250

`CGC_PREFILL_STREAM=1`（`scripts/run_server.sh:501`）**只在 prefill250 的 BUDGET 分支**被設定，
與它同批的還有 `CGC_SERVER_BATCH=UBATCH=5632`、`CTX=8192`、`CGC_GATHER_SLAB_CAP=256`。

**`prod25` 從不設 `CGC_PREFILL_STREAM`** ⇒ `llama-context.cpp:285` 的 clamp 生效
（`cgc_pool_max_tokens()`，MTP 開啟時為 **8**）⇒ prompt 被切成 **8-token chunk 走專家池**，
每一塊都要碰 SSD ⇒ 該 regime 的 prompt 吞吐就是 **~10 t/s**。

`python3 scripts/check/prod_matrix.py --list`（零 GPU，全部由 `run_server.sh CGC_DUMP_ENV=1` 解析）：

| profile | ctx | batch | ubatch | stream | slab | prompt 處理走哪 |
|---|---|---|---|---|---|---|
| `prod25` | 4096 | **-** | **-** | **0** | - | 專家池（chunk 夾在 8） |
| `prefill250` | 8192 | **5632** | **5632** | **1** | **256** | whole-layer slab |

> ⚠️ 讀法：`prod_matrix.py:222` 是 `env.get("CGC_PREFILL_STREAM", "0")` —— **讀不到就顯示 0**。
> 「沒設」與「設成 0」在那張表上同形（本 repo 的老陷阱的又一個實例）。

旁證：`scripts/check/llama_bench_matrix.py:88` 早就有一個 `prod25-stream` 臂
（`prod25` ＋ `CGC_PREFILL_STREAM=1` ＋ `CGC_GATHER_SLAB_CAP=256`）。

## 2. 生產級實測（`scripts/check/prod_matrix.py`，14:30）

啟動前狀態：`thermalpressurelevel = 0`（NOMINAL）、free 78%、無 8080 listener、無其他量測行程
⇒ 閒置機器閘門 PASS。

| profile | cell | launch | platform t/s | avg_ts | samples |
|---|---|---|---|---|---|
| `prefill250` | `prefill-house`（`-p 2048 -n 16 -b 5632`） | **NOMINAL** | **234.24** | 241.44 | 255.82 / 247.22 / 221.27 |
| `prefill250` | `prefill-up`（`-p 512 -n 128 -b 2048`） | HEAVY | 94.65 | 97.44 | 103.01 / 85.72 / 103.57 |

- **`prefill-house` 的 launch 是 NOMINAL ⇒ 這一格可以當生產數字**（234.24；最佳單輪 255.82）。
- **`prefill-up` 是 HOT 樣本 ⇒ 不是生產數字**（`lesson eng-mh-0054`）。
- `run_server.sh:490-496` 自己記著：250+ 在 **今天 12:51 量到 284.58**、13:45 驗收為
  **268.09 / 273.55 / 256.61**；並且「**同指紋可差到 2×**，報吞吐必須附機器狀態」。

## 3. 生產工具**自己拒絕**量 `prod25` 的 prefill cell（逐字）

```
SKIP prod25 prefill-house  -- -p 2048 does not fit -b 8 (GGML_ASSERT n_tokens_all <= n_batch;
                              the engine clamps n_batch to cgc_pool_max_tokens() on this profile)
SKIP prod25 prefill-up     -- profile pins no BATCH and prod25 has no CGC_PREFILL_STREAM, so the
                              engine's pool-path clamp (cgc_pool_max_tokens, default 8) applies --
                              the cell's -b 2048 would NOT be the effective batch
```

⇒ **`prod25` 的 prompt 處理用現行四個標準 cell 一個都表達不出來。** 這些是 **refusal，不是 failure**，
而且是「標準 cell 還沒對每個 profile 統一」的直接證據。

## 4. 單變數隔離：clamp 一解除，兩個 profile 就沒有差別

| 臂 | cell | launch | platform |
|---|---|---|---|
| `prefill250` | `p512n0d0` | HEAVY | **94.65** |
| `prod25` ＋ `--extra-env 'CGC_PREFILL_STREAM=1;CGC_GATHER_SLAB_CAP=256'` | `p512n0d0` | HEAVY | **93.49** |

差 **1.2%**。（此臂的 `prefill-house` 仍被拒，理由換成 `-p 2048 does not fit -b 512`
—— clamp 解除後工具自己選 `-b 512`。）

`prefill-house`（234）比 `prefill-up`（94）快 2.5× 的原因：**chunk 越長，slab 的整層專家載入被攤得越薄**
（2048 token/chunk vs 512 token/chunk）。

## 5. ★ 方法論後果：現行標準 cell 看不見 `CGC_MM_BITIDENT`

`llama-bench` 是本 repo 唯一的登記儀器，但它**結構上**看不見 (a)／(b) 那一類旋鈕：

- tg 走 `llama_decode(ctx, llama_batch_get_one(&token, 1))`（`tools/llama-bench/llama-bench.cpp:2311`）
  ⇒ **M = 1**，而 M=1 不在任何 small-batch 群組內（group A 要 `ne11 ∈ [2,8]`，group B 要 `[4,8]`）；
- pp 是大 M（`pp512`/`pp2048` ⇒ `ne11 > ne11_mm_min = 8` ⇒ 走 `mul_mm`）。

⇒ **要替 `CGC_MM_BITIDENT=0`（(a)）或「IQ4_XS 進 group A」（(b)）定價，必須另立一個
`-b 8 -ub 8` 的 pp cell**（M=8，正好落在 group A 的 `[2,8]` 內）。這也是為什麼 09-18 那兩臂只能用
已被退役的 `decode_bench` 路徑做配對，而那條路的同配置漂移是 **1.49×**。

## 6. 產物

- `Backup/phase_decomp/prod_matrix_prefill_20260918_1430.json`（本節第 2 節）
- `Backup/phase_decomp/prod25_stream_mechanism_20260918_1446.json`（第 4 節）
- 逐 cell 的 llama-bench 原始輸出：`Backup/prod_matrix/20260918_1430/`、`Backup/prod_matrix/20260918_1446/`
- 生成表：`docs/PREFILL_RECHECK_2026-09-18.md`（`prod_matrix --md` 的產物；
  **刻意不進 MANIFEST** —— `docs/*` 是索引的顯式註冊表，而生成檔每次重跑 bytes 就變，
  入表等於製造註定的漂移）
- build `efeade7e2`，模型 `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`
