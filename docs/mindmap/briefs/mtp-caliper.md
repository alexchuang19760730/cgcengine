# MTP 口徑與混淆定位（pool／跨啟動／順序） — 技術白皮書　·　3b ③b 實驗目標達成（可放生產）

> **一句話**：定出 MTP 的量測口徑並定位混淆因子（pool 大小、跨啟動污染、執行順序），讓 MTP 數字可比。

- 主題：MTP／spec　·　子目標：**M MTP on 加速**（活躍攻關軸：speculative 攤薄係數 m 與 accept rate a）
- 階段：生產（可放生產／已交付）　·　① ② ③b —— 已進生產、或已是生產決策依據

---

## 1. 目標

定出 MTP 的量測口徑並定位混淆因子（pool 大小、跨啟動污染、執行順序），讓 MTP 數字可比。

## 2. 判準

配對設計；單點不可引用

## 3. 結果

定案：任何單點 MTP 數字不可引用；pool 4 vs 8 GiB 是一階混淆；同 launch 配對 ×1.06、生產 server 路徑 ×0.695（方向相反）

## 4. 判定

**3b · ③b 實驗目標達成（可放生產）** — 產物已可放進生產級設置：不破壞正確性 ∧ 成本可接受 ∧ 無前置條件

> ⇒ 交付 cell 的 MTP 增益 UNRESOLVED（作廢 9.82/12.62/+28.5% 後）

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [MTP 儀器化（接進 llama-bench）](mtp-instrument.md) | 3b | 接通（真因＝test_gen_spec 的 n_past 初始化）；此後 MTP 不再跨儀器比較 |
| [MTP 到 2× 的邊界（攤薄算術）](mtp-2x.md) | 3b | k 加到多大都不到 2×（server a/m=1.445、bench 0.982）；現況每產出 token 成本 45.6~56.4 vs baseline 48.2 ⇒ 攤薄  |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/MTP_K_AB_2026-09-18.md／MTP_POOL_CONFOUND_2026-09-17.md／MEMORY_PERF.md:402-405` |
| 備註 | ⇒ 交付 cell 的 MTP 增益 UNRESOLVED（作廢 9.82/12.62/+28.5% 後） |
| 軸性質 | 活躍攻關軸：speculative 攤薄係數 m 與 accept rate a |
| 對應報告 | 8 份 |

- [MTP3_25TPS_ARITHMETIC_2026-09-18.md](../../MTP3_25TPS_ARITHMETIC_2026-09-18.md)
- [MTP_ABBA_RECHECK_2026-09-18.md](../../MTP_ABBA_RECHECK_2026-09-18.md)
- [MTP_BENCH_PARITY_20260918_1945.html](../../MTP_BENCH_PARITY_20260918_1945.html)
- [MTP_K_AB_2026-09-18.md](../../MTP_K_AB_2026-09-18.md)
- [MTP_NET_EFFECT_PAIRED_2026-09-18.md](../../MTP_NET_EFFECT_PAIRED_2026-09-18.md)
- [MTP_POOL_CONFOUND_2026-09-17.md](../../MTP_POOL_CONFOUND_2026-09-17.md)
- [ORNITH_ALIGNED_AB_AND_25_ARITHMETIC_20260918_0445.html](../../ORNITH_ALIGNED_AB_AND_25_ARITHMETIC_20260918_0445.html)
- [S0_MTP_ONOFF_AB_2026-09-21.md](../../S0_MTP_ONOFF_AB_2026-09-21.md)

---

← [MTP 儀器化（接進 llama-bench）](mtp-instrument.md)　·　[總目錄](index.md)　·　[HTML 版](mtp-caliper.html)　·　[MTP k-sweep（verify batch T 成本曲線） →](mtp-ksweep.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
