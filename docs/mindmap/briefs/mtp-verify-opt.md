# verify residency thrash ／ 池配額 ／ prefetch 開關 — 技術白皮書　·　4 ④ 廢棄

> **一句話**：降低 verify 階段的 residency thrash（池配額、prefetch 開關），讓 verify 不因權重換入換出而變慢。

- 主題：MTP／spec　·　子目標：**M MTP on 加速**（活躍攻關軸：speculative 攤薄係數 m 與 accept rate a）
- 階段：已結案（廢棄／不適用）　·　④ 廢棄 ＋ 不適用

---

## 1. 目標

降低 verify 階段的 residency thrash（池配額、prefetch 開關），讓 verify 不因權重換入換出而變慢。

## 2. 判準

同 k 跨池的 bytes→時間彈性

## 3. 結果

降到次要：彈性只有 0.10–0.17（k=1/3 甚至 −0.38）；k_eff 單變數解釋 84%；修 residency 上界只剩 ~15%；m 的機械解釋是「每 draft token 的固定代價」

## 4. 判定

**4 · ④ 廢棄** — 判死／撤回／不做

> 另判：調高 CGC_FAST_COLD_MAX 有損 ⇒ 不做；更大的 k／tree verification 別做

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [RSL-MTP（draft top-8 ⊆ 已付費並集）＋ 練 draft head](mtp-rsl.md) | 4 | 被自己的數據否決：0.70–0.97×，無一格 ≥1.0；m=0.474 下即使 a→1 也不可能 2× |
| [accept rule／dynamic-k／n_max 調整](mtp-accept.md) | 4 | 不做：dynamic-k oracle 只 1.035×；n_max 3→5 作廢；greedy 下 accept 不可由 accept rule 移動（是 (base,head) |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/MTP_VERIFY_OPT_2026-09-18.md／MTP_VERIFY_COST_2026-09-21.md` |
| 備註 | 另判：調高 CGC_FAST_COLD_MAX 有損 ⇒ 不做；更大的 k／tree verification 別做 |
| 軸性質 | 活躍攻關軸：speculative 攤薄係數 m 與 accept rate a |
| 對應報告 | 3 份 |

- [EXPERT_CACHE_THRASH_2026-09-19.md](../../EXPERT_CACHE_THRASH_2026-09-19.md)
- [MTP_VERIFY_COST_2026-09-21.md](../../MTP_VERIFY_COST_2026-09-21.md)
- [MTP_VERIFY_OPT_2026-09-18.md](../../MTP_VERIFY_OPT_2026-09-18.md)

---

← [RSL-MTP（draft top-8 ⊆ 已付費並集）＋ 練 draft head](mtp-rsl.md)　·　[總目錄](index.md)　·　[HTML 版](mtp-verify-opt.html)　·　[accept rule／dynamic-k／n_max 調整 →](mtp-accept.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
