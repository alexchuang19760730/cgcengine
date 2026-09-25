# RSL-MTP（draft top-8 ⊆ 已付費並集）＋ 練 draft head — 技術白皮書　·　4 ④ 廢棄

> **一句話**：用 RSL-MTP（讓 draft 的 top-8 ⊆ 已付費並集）降低 draft 成本，並評估練 draft head 划不划算。

- 主題：MTP／spec　·　子目標：**M MTP on 加速**（活躍攻關軸：speculative 攤薄係數 m 與 accept rate a）
- 階段：已結案（廢棄／不適用）　·　④ 廢棄 ＋ 不適用

---

## 1. 目標

用 RSL-MTP（讓 draft 的 top-8 ⊆ 已付費並集）降低 draft 成本，並評估練 draft head 划不划算。

## 2. 判準

每格 ≥1.0 才算活

## 3. 結果

被自己的數據否決：0.70–0.97×，無一格 ≥1.0；m=0.474 下即使 a→1 也不可能 2×

## 4. 判定

**4 · ④ 廢棄** — 判死／撤回／不做

> ⇒ 別練 draft head

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [verify residency thrash ／ 池配額 ／ prefetch 開關](mtp-verify-opt.md) | 4 | 降到次要：彈性只有 0.10–0.17（k=1/3 甚至 −0.38）；k_eff 單變數解釋 84%；修 residency 上界只剩 ~15%；m 的機械解釋是「每 draft |
| [accept rule／dynamic-k／n_max 調整](mtp-accept.md) | 4 | 不做：dynamic-k oracle 只 1.035×；n_max 3→5 作廢；greedy 下 accept 不可由 accept rule 移動（是 (base,head) |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/RSL_MTP_GAIN_ESTIMATE_2026-09-18.md` |
| 備註 | ⇒ 別練 draft head |
| 軸性質 | 活躍攻關軸：speculative 攤薄係數 m 與 accept rate a |
| 對應報告 | 4 份 |

- [MOE_MTP_FEASIBILITY_2026-09-18.md](../../MOE_MTP_FEASIBILITY_2026-09-18.md)
- [REUSE_DISTANCE_RESULT_2026-09-20.md](../../REUSE_DISTANCE_RESULT_2026-09-20.md)
- [RSL_MTP_GAIN_2026-09-18.md](../../RSL_MTP_GAIN_2026-09-18.md)
- [RSL_MTP_GAIN_ESTIMATE_2026-09-18.md](../../RSL_MTP_GAIN_ESTIMATE_2026-09-18.md)

---

← [MTP 到 2× 的邊界（攤薄算術）](mtp-2x.md)　·　[總目錄](index.md)　·　[HTML 版](mtp-rsl.html)　·　[verify residency thrash ／ 池配額 ／ prefetch 開關 →](mtp-verify-opt.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
