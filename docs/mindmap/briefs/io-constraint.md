# IOCACHE 約束承認 ＋ 段邊界 S2 — 技術白皮書　·　4 ④ 廢棄

> **一句話**：把 IOCACHE 約束與段邊界（S2）承認下來並寫成約束——目的是界定什麼不可能，而不是拿增益。

- 主題：上界／kernel　·　子目標：**S 序列化消減**（活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉）
- 階段：已結案（廢棄／不適用）　·　④ 廢棄 ＋ 不適用

---

## 1. 目標

把 IOCACHE 約束與段邊界（S2）承認下來並寫成約束——目的是界定什麼不可能，而不是拿增益。

## 2. 判準

前提是否成立

## 3. 結果

IOCACHE：#6 沒做之前 #3 的 async 版本不可達（且不可並列相加）；S2：建議不做完（段邊界／drain）

## 4. 判定

**4 · ④ 廢棄** — 判死／撤回／不做

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [IO 請求形狀（合併 pread）](io-shape.md) | 4 | 一個 expert 三段相隔 88~115 MB ⇒ 3→1 要付 182.6× 位元組；合併邏輯早已存在（preadv）；固定開銷占 90% |
| [M-W：WORKERS 8→2](io-mw.md) | 4 | us/job −11.1%、us_per_miss −14.7%，但 t/s −0.6% ⇒ 未分離（by-product：IO 只占 step ≈9%） |
| [CGC_LAYER_AHEAD_PREFETCH（提前一層）](io-layer-ahead.md) | 4 | −7.1%（未分離、方向偏負） |
| [M-PF：背景 neighbour prefetch](io-mpf.md) | 4 | 撤回（讀入位元組 1.98×，實測最好只到 35%） |
| [S1 早期診斷系列（09-16/17）](s1-refuted.md) | 4 | 多輪被自己否證：slot owner 推論被推翻、時序推論被推翻、09-16 那七輪「第一個分歧」全部不可引用（同時踩三個盲點） |
| [3b：fill 觸發點搬出 hook ＋ batch 化](miss-3b.md) | 4 | 判死：合併邏輯早已存在、幾何 182.6×、生產口徑上界 ~9% ⇒ 收益 ~2.9% |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/IOCACHE_CONSTRAINT_ADMISSION_2026-09-22.md／docs/S2_RESIDENCY_AND_DRAIN_VERDICT_2026-09-20.md` |
| 備註 |  |
| 軸性質 | 活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉 |
| 對應報告 | 5 份 |

- [IOCACHE_CONSTRAINT_ADMISSION_2026-09-22.md](../../IOCACHE_CONSTRAINT_ADMISSION_2026-09-22.md)
- [L3_M1_BLOCKED_STRUCTURAL_2026-09-20.md](../../L3_M1_BLOCKED_STRUCTURAL_2026-09-20.md)
- [L3_M1_GAP_LEDGER_2026-09-20.md](../../L3_M1_GAP_LEDGER_2026-09-20.md)
- [S2_PROBE2_AND_CEILING_2026-09-20.md](../../S2_PROBE2_AND_CEILING_2026-09-20.md)
- [S2_RESIDENCY_AND_DRAIN_VERDICT_2026-09-20.md](../../S2_RESIDENCY_AND_DRAIN_VERDICT_2026-09-20.md)

---

← [shape／knob 世界模型（14–15 個旋鈕的邊界）](shape-knobs.md)　·　[總目錄](index.md)　·　[HTML 版](io-constraint.html)　·　[swap 結構修復（L0–L4 + P0/P1/P2） →](sys-swap.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
