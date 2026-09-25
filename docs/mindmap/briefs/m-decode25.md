# decode ≥ 25（M-25） — 技術白皮書　·　4 ④ 廢棄

> **一句話**：把 decode 端到端推到 ≥25 t/s（「250 / 25」裡的 25 那一半）。

- 主題：交付里程碑　·　子目標：**兩者**（耦合類：需 S 與 M 同時成立（例：M-25）；收益不可假設可相加）
- 階段：已結案（廢棄／不適用）　·　④ 廢棄 ＋ 不適用

---

## 1. 目標

把 decode 端到端推到 ≥25 t/s（「250 / 25」裡的 25 那一半）。

## 2. 判準

25 ⇔ 40.0 ms/token

## 3. 結果

交付 12.57＝79.6 ms ⇒ 要砍 49.6%；IO 全關也只 ~+5%（保守 +9%）⇒ 判死

## 4. 判定

**4 · ④ 廢棄** — 判死／撤回／不做

> P(25) < 3%；物理可達但已知槓桿只到 13.2~13.8

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| （本軸／本階段沒有其它條目） | — | — |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `DECODE25_FINAL_VERDICT_2026-09-21／DECODE_25TPS_VERDICT_2026-09-23／docs/IO_AXIS_VERDICT_2026-09-25.md` |
| 備註 | P(25) < 3%；物理可達但已知槓桿只到 13.2~13.8 |
| 軸性質 | 耦合類：需 S 與 M 同時成立（例：M-25）；收益不可假設可相加 |
| 對應報告 | 15 份 |

- [CEILING_STACK_2026-09-21.md](../../CEILING_STACK_2026-09-21.md)
- [CROSS_LINE_STATUS_AND_TARGET_MATH_2026-09-20.md](../../CROSS_LINE_STATUS_AND_TARGET_MATH_2026-09-20.md)
- [DECODE25_CEILING_2026-09-20.md](../../DECODE25_CEILING_2026-09-20.md)
- [DECODE25_FINAL_VERDICT_2026-09-21.md](../../DECODE25_FINAL_VERDICT_2026-09-21.md)
- [DECODE_25TPS_TWO_ALGORITHMS_2026-09-19.md](../../DECODE_25TPS_TWO_ALGORITHMS_2026-09-19.md)
- [DECODE_25TPS_VERDICT_2026-09-23.md](../../DECODE_25TPS_VERDICT_2026-09-23.md)
- [G7_DELIVERY_ARITHMETIC_2026-09-20.md](../../G7_DELIVERY_ARITHMETIC_2026-09-20.md)
- [PROGRESS_25TPS_2026-09-21.md](../../PROGRESS_25TPS_2026-09-21.md)
- [TARGET25_NECESSARY_CONDITIONS_2026-09-22.html](../../TARGET25_NECESSARY_CONDITIONS_2026-09-22.html)
- [TARGET25_NECESSARY_CONDITIONS_2026-09-22.md](../../TARGET25_NECESSARY_CONDITIONS_2026-09-22.md)
- [TARGET25_REPORT_20260922_184209.html](../../TARGET25_REPORT_20260922_184209.html)
- [TARGET25_REPORT_20260922_184209.json](../../TARGET25_REPORT_20260922_184209.json)
- [TARGET25_TWO_AXIS_2026-09-20.md](../../TARGET25_TWO_AXIS_2026-09-20.md)
- [TOOLCHAIN_25TPS_2026-09-22.html](../../TOOLCHAIN_25TPS_2026-09-22.html)
- [WHY_MODEL_MISSED_AND_IS_25_REACHABLE_2026-09-21.md](../../WHY_MODEL_MISSED_AND_IS_25_REACHABLE_2026-09-21.md)

---

← [① 攻關成功（pp≥250 ∧ tg>12.57）](m-total.md)　·　[總目錄](index.md)　·　[HTML 版](m-decode25.html)　·　[池大小掃描（8→4→2 GiB） →](io-poolsize.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
