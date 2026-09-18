# 生產級量測：prefill250 × 四格（2026-09-18 13:37–13:45）

**一句話：四格都跑完了、資料完整，但沒有一格是可引用的生產數字 —— 因為機器本身在搶記憶體，不是引擎的問題。**

- 入口：`python3 scripts/check/prod_matrix.py --profiles prefill250 --cells decode,decode-up,prefill-house,prefill-up --reps 3`
- 產物：`Backup/phase_decomp/prod_matrix_20260918.json`（完整）、逐格 log 在 `Backup/prod_matrix/20260918_13*`
- build：`efeade7e2` ／ model：`Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`（13.65 GB，35.5B 參數）

## 為什麼只有 `prefill250` 一個 profile

工具**主動 SKIP 了 `prod25` 的四格**，逐字理由：

> `prod25` decode — profile pins no BATCH and prod25 has no `CGC_PREFILL_STREAM`, so the engine's
> pool-path clamp (`cgc_pool_max_tokens`, default 8) applies — the cell's `-b 512` would NOT be the
> effective batch

也就是說：在 `prod25` 上量出來的 decode batch 不是你要的那個 batch，整個 cell 沒有意義。
⇒ **能被 `prod_matrix` 當成生產 cell 的 profile 只有 `prefill250`**（它有 `CGC_PREFILL_STREAM`）。

## 四格的結果

| cell | 形狀 | launch | worst | platform t/s | avg_ts | n_kept |
|---|---|---|---|---|---|---|
| `decode` | `-p 0 -n 128 -d 512 -b 512` | **HEAVY** | HEAVY | **5.61** | 5.43 | 2 |
| `decode-up` | `-p 0 -n 128 -d 512 -b 2048` | **HEAVY** | HEAVY | **5.32** | 4.79 | 2 |
| `prefill-house` | `-p 2048 -n 16 -d 0 -b 5632` | **HEAVY** | HEAVY | **116.94** | 120.59 | 2 |
| `prefill-up` | `-p 512 -n 128 -d 0 -b 2048` | MODERATE | MODERATE | **72.42** | 75.70 | 2 |

`platform t/s` ＝ 丟掉 rep 1 之後的兩次中位（工具自算，`rep1_penalty_pct` 一併附在 JSON 裡）。

## 判決：不可引用（而判準是工具自己給的）

`launch` 是子行程**啟動瞬間**的熱壓（`thermal_pressure.Sampler` 的第一個樣本在子行程存在之前取）。
三格啟動在 **HEAVY**、一格在 **MODERATE**，`thermal_hist` 整段都是 HEAVY/MODERATE。
依工具自己的 `--min-usable-pct` 判準與 2026-09-17 已記的規程 ⇒ **這是 HOT 樣本。**

對照 2026-09-17 的首次四格（同樣全 HOT）：decode **8.89／9.33**、prefill **102.17／102.73**。
今天 decode 5.61 比它低 37% —— 但引擎沒有退步，見下一節。

## ★ 真正的原因：本機 IDE/宿主佔了 ~10.5 GB

```
$ sysctl -n vm.swapusage
total = 14336.00M  used = 12836.56M  free = 1499.44M   (encrypted)      ← swap 已用 90%

$ vm_stat | head
Pages free: 318629 × 16 KiB ≈ 4.9 GiB

$ top -l 1 -o mem -n 8 | tail -12
COMMAND          MEM
WorkBuddy Helper 4820M
Electron         1924M
Electron         1010M
Electron         906M
Electron         704M
Electron         650M
Electron         457M
WindowServer     437M                    ← 前 8 名合計 ≈ 10.5 GB，全部是宿主/IDE
```

而 `prod_matrix` 每一格的 `pre` 區塊記著量測**之前**的記憶體狀態，四格之間還在惡化：
`swap_used_mib` **12182 → 15933 MiB**。

⇒ 模型在這台 16 GB 機器上實際只拿到約 **1.5 GB 可用**，其餘靠 12.8 GB 的 swap 翻頁。
**5.61 t/s 是記憶體爭用的產物，不是引擎的產物。** 記憶裡那個可引用的暖平台 decode
（**10.78–10.91**）是同一顆 M4 在**沒有 IDE 搶記憶體**時量到的。

## 要拿到可引用的生產數字，需要的前置條件

1. **關掉宿主/IDE 的 Electron 群**（或至少確認 `vm.swapusage` 的 `used` 回到低檔），
   讓模型能拿到真實的 unified memory；
2. 靜置 **≥1800 s** 後才發射（skill `cgc-prefill-thermal-delivery` 的 `COLD_QUIET=1800`），
   且每一格的 `launch` 必須是 `NOMINAL` 才可引用；
3. 在那之前**不要報 5.61／116.94 這組數字**（含「我們 decode 只有 5.6」這種結論）。

## 這份報告不主張什麼

- 不主張引擎退步了（今天與 09-17 的差可完全由 swap 解釋，但**我沒有做單變數驗證**；
  要驗就得在記憶體乾淨的窗口重跑同一組 cell）。
- 不主張 prefill 退步了 —— 同樣的理由。
- `decode` 與 `decode-up` 的差（5.61 vs 5.32）在 HOT 樣本下不可區分。
