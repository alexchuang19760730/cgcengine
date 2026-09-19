# decode 量測的噪音底（2026-09-19）—— 為什麼「bench 追上 HTTP 13.68」量不出來

## 一句話

**這台機器的跨啟動噪音是 15–18%（同配置、同 binary、同形狀），而 bench↔HTTP 的待解釋差距約 15%。**
⇒ 兩者同量級 ⇒ **單次 A/B 不可能判定它**。今天所有「一次 A vs 一次 B」的結論全部作廢。

## 證據（兩個獨立來源）

### 來源 1（獨立）：`scripts/check/decode_window_harness.py` 的既有數據
`/tmp/decode_window_harness.json`（`finished_at 17:12:35`），單位 ms/token：

| 臂 | BITIDENT | rep1_cold | rep2_warm |
|---|---|---|---|
| b1 | 1 | 222.18 | **102.45** |
| b0 | 0 | 458.13 | 148.70 |
| b1b | 1 | 158.23 | **89.25** |

- **噪音底 = b1(b1b) 兩個同配置臂：102.45 vs 89.25 = 1.148×（+14.8%）**
- 該 harness 的 docstring 另記一次 **17.5%**（166.4 vs 139.6 ms/step）
- 它的 lane_drift block 0 內、相鄰時間的 b1 vs b1b：194.7 vs 225.9 = **1.16×**
- **冷→暖（同一臂 rep1→rep2）= 222.18 → 102.45 = 2.17×** ← 最大的一個效應

### 來源 2（我這邊）：`--ctx-size` 的重複實驗
同引擎（`libllama 08:32`）、同形狀（`-d 512 -n 192 --warm-skip 64`）、各 2 次啟動：

| 格 | #1 median | #2 median | 格內散布 |
|---|---|---|---|
| `--ctx-size 4096` | 11.66 | 9.87 | **1.18×** |
| ctx 推導（704） | 10.90 | 12.20 | **1.12×** |

⇒ 第一次跑出 `platform 13.819`（跨過 13.68），第二次 `8.845`。**是同一個雜訊造的假效應。**

## 可主張 vs 不可主張

**可主張**
- bench 平台（median，4 次啟動）≈ **10.8–11.6**；HTTP 平台 = **13.1–14.0**。
- `-d 512 -n 128` 的 **7.96 是短窗口的量，不是平台**（`-n 192` 10.34、`-n 512` 10.22）。
- **冷→暖跨 request 是 2.17×** ⇒ 「引用哪一個 request」比任何 15% 的旋鈕重要得多。

**不可主張（已作廢）**
- ~~`--warm-skip` +10.5%~~（4 次 `skip>0` 的 median 匯總 ≈11.4，落在噪音內）
- ~~`--ctx-size 4096` → 13.819~~（第 2 次 8.845）
- ~~greedy / ctx 的 C、D 兩臂（7.997 / 8.475）~~（該次熱污染，且無重複）

## 因此的正確做法

1. **先量底**：`python3 scripts/check/paired_ab.py --null --cell decode --pairs 4 --reps 3`
   ⇒ 預期 ≈15%。**若如此，正解是「此效應在現有設計下不可測」。**
2. **要用配對設計**：相鄰時間成對、block 間**交替順序**、看**配對差**而不是看**水平值**、
   低於底就判 **INCONCLUSIVE**（這是 `decode_window_harness.py` 與 `paired_ab.py` 都已實作的形狀）。
3. **降噪的選項**：更多 rep ＋ 報 **median**（不要 mean——今天出現過 17.64 這種離群 rep）、
   交錯 A/B、以及**不要在同一條 chain 裡 build**（8 執行緒編譯自己就會把機器加熱）。

## 本日新增的兩個可 A/B 旋鈕（`llama-bench`）

| 旋鈕 | env（給 `paired_ab.py --a-env/--b-env` 用） | 預設 |
|---|---|---|
| `--warm-skip N`：每 rep 先跑 N 個 token 不計時，且 `n_gen` 扣掉 | `CGC_BENCH_WARM_SKIP` | 0（逐位相容舊行為） |
| `-c, --ctx-size N`：覆寫推導的 `n_ctx = n_prompt+n_gen+n_depth` | `CGC_BENCH_CTX` | 0（維持推導值） |

CLI 優先於 env。JSON 多出 `warm_skip` / `ctx_override` 兩欄。
背景：`llama-bench.cpp` 的 `to_llama_cparams()`（`cparams.n_ctx = n_prompt + n_gen + n_depth`）讓 house decode 形狀跑在 ctx **704**，而 prod25 server 跑在 **4096**
—— 這是每次 bench-vs-HTTP 比較都帶著的 confound（但**已證實不是**那 15% 的來源）。
