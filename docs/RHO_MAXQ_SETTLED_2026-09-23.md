# ρ 干涉定讞：MAXQ 掃描（2026-09-23）

**Status:** 定讞。干涉是成因、ρ 機制本身淨正、MAXQ=16 是甜點。保險絲已 commit（079f2fe46）。

## 0. 三句話答案

1. ρ（shadow router 提前預取）機制是對的：hit 61.4→87.4（+26pp）、讀的位元組 8.80→3.18 GiB（−64%）。
2. 但它的背景預取把裝置灌滿：80k×41KB 小讀把 `fill_wait` 從 56~75ms 頂到 **15–20s**，t/s −21%。
3. 保險絲 `CGC_RHO_PREFETCH_MAXQ`（佇列深度上限）實測證明干涉是成因：**rho-q16 = 9.64 t/s（> on 基準 +4.7%）、hit 67.9→90.9%**，且 `#13 maxq_limit=3249/3348`（97% drop 由 MAXQ 執行）。

## 1. 方法（rhoq3，prod-new 統一介面，同輪同環境）

- profile：`prod-new:CGC_SERVER_MTP=1`（MTP on 基準）；base = `prod-new`（MTP off 交付口徑）
- cell：`--reps 1 --prompt 0 --gen 128 --depths 512 --batch 512 --ctx-size 4096 --warm-skip 64`
- 臂：base / on / rho（無保險絲）/ rho-q16 / rho-q8 / rho-q4
- 儀器：drop breakdown 輸出行 `#13 maxq_limit`（每臂 stderr.log）

## 2. 結果

| 臂 | t/s | hit% | vs on |
|---|---|---|---|
| base（MTP off） | 9.85 | 94.9% | +6.9% |
| on（MTP on，無 ρ） | 9.21 | 67.9% | — |
| rho（無保險絲） | **4.62** | 88.9% | **−50%** |
| **rho-q16（MAXQ=16）** | **9.64** | **90.9%** | **+4.7%** |
| rho-q8（MAXQ=8） | 8.37 | 90.3% | −9% |
| rho-q4（MAXQ=4） | 7.00 | 87.2% | −24% |

決定性證據：rho-q16 臂 `#13 maxq_limit=3249/3348`（97% drop 是 MAXQ 做的）⇒
保險絲真的觸發、干涉就是 −50% 的元兇、ρ 機制本身淨正（hit 67.9→90.9 且比 on 快）。

MAXQ 掃描單調：q16 > q8 > q4。**MAXQ=16 是甜點。**

⚠ 該輪 swap 5768–6036 MiB 高水位 ⇒ 絕對值偏低；同輪相對比較有效，絕對 t/s 不可直接引用。

## 3. 注意：97% drop 後 hit 反而更高（90.9 > 88.9）

rho（完整預取、塞車）hit 88.9% < rho-q16（3% 預取）90.9%。
⇒ 佇列塞住時，97% 的預測是**過時/有害**的——不只干涉，可能還污染池。
⇒ **完整預取的價值可疑**：按層批次化若「保留完整預取」可能白做甚至比 q16 差。

## 4. 判別實驗（進行中）：rho-q16 vs rhoq16-leg

`rhoq16-leg` = MAXQ=16 + `CGC_PREFETCH_LEGACY_FILL=1`（thread-per-segment 未合併小讀）。

| 結果 | 判定 |
|---|---|
| rhoq16-leg ≈ rho-q16（~9.6） | 干涉是唯一問題、MAXQ 就是答案 → 按層批次化不值得寫 |
| rhoq16-leg 明顯掉（<9） | fill 形狀（0.04 MiB/job）在無塞車時仍有代價 → 批次化值得寫 |

## 5. 落地狀態

- ✅ 保險絲 `CGC_RHO_PREFETCH_MAXQ` 已 commit（079f2fe46：llama-expert-cache.h/.cpp、run_server.sh allowlist、rho_fill_ab.sh、commit_bench.py；build 產物 libllama 隨 commit）
- ⏸ 正解「bg 按層批次化」（跨 expert 合併大讀）：視 §4 判別結果決定是否寫
- 📊 成績：commit_bench prod-new prefill=289.81 / decode=12.19(mtp=off)（079f2fe46 標題）
