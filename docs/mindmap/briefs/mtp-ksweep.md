# MTP k-sweep（verify batch T 成本曲線） — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：掃 verify batch 長度 T（k-sweep），量出每多驗一個 token 的邊際成本曲線，找出該 regime 的最佳 k。

- 主題：MTP／spec　·　子目標：**M MTP on 加速**（活躍攻關軸：speculative 攤薄係數 m 與 accept rate a）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產

---

## 1. 目標

掃 verify batch 長度 T（k-sweep），量出每多驗一個 token 的邊際成本曲線，找出該 regime 的最佳 k。

## 2. 判準

step(T) 線性擬合 ＋ 各 k 的 mean_len

## 3. 結果

step ≈ 14.61 + 42.03·T（max resid 11.5）；最佳 k=2 但那格不可移植（合成 accept 0.93~0.99 vs 交付 0.465~0.58）

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 模型可移植、最佳 k 不可移植

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| （本軸／本階段沒有其它條目） | — | — |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `Backup/s1_ksweep/20260924_225548/s1_ksweep.json（quotable=false）` |
| 備註 | 模型可移植、最佳 k 不可移植 |
| 軸性質 | 活躍攻關軸：speculative 攤薄係數 m 與 accept rate a |
| 對應報告 | 5 份 |

- [BIGMOMO_TRANSFER_AND_MTP_PARAM_PARITY_2026-09-19.md](../../BIGMOMO_TRANSFER_AND_MTP_PARAM_PARITY_2026-09-19.md)
- [MTP_KEY_SHAPE_2026-09-23.html](../../MTP_KEY_SHAPE_2026-09-23.html)
- [MTP_K_SWEEP_2026-09-22.md](../../MTP_K_SWEEP_2026-09-22.md)
- [MTP_VERIFY_HEADROOM_2026-09-24.md](../../MTP_VERIFY_HEADROOM_2026-09-24.md)
- [MTP_VERIFY_SPLIT_2026-09-19.md](../../MTP_VERIFY_SPLIT_2026-09-19.md)

---

← [MTP 口徑與混淆定位（pool／跨啟動／順序）](mtp-caliper.md)　·　[總目錄](index.md)　·　[HTML 版](mtp-ksweep.html)　·　[MTP 到 2× 的邊界（攤薄算術） →](mtp-2x.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
