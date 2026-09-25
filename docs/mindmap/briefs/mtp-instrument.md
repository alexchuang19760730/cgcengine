# MTP 儀器化（接進 llama-bench） — 技術白皮書　·　3b ③b 實驗目標達成（可放生產）

> **一句話**：把 MTP 接進 llama-bench，讓 MTP on／off 能在交付口徑下做配對量測（而不是靠 HTTP server 的舊口徑）。

- 主題：MTP／spec　·　子目標：**M MTP on 加速**（活躍攻關軸：speculative 攤薄係數 m 與 accept rate a）
- 階段：生產（可放生產／已交付）　·　① ② ③b —— 已進生產、或已是生產決策依據

---

## 1. 目標

把 MTP 接進 llama-bench，讓 MTP on／off 能在交付口徑下做配對量測（而不是靠 HTTP server 的舊口徑）。

## 2. 判準

同一口徑下能做 MTP on/off

## 3. 結果

接通（真因＝test_gen_spec 的 n_past 初始化）；此後 MTP 不再跨儀器比較

## 4. 判定

**3b · ③b 實驗目標達成（可放生產）** — 產物已可放進生產級設置：不破壞正確性 ∧ 成本可接受 ∧ 無前置條件

> ★ 前身：MTP off 9.82／on 12.62 是 HTTP 口徑 ⇒ 已作廢（契約 §7）

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [MTP 口徑與混淆定位（pool／跨啟動／順序）](mtp-caliper.md) | 3b | 定案：任何單點 MTP 數字不可引用；pool 4 vs 8 GiB 是一階混淆；同 launch 配對 ×1.06、生產 server 路徑 ×0.695（方向相反） |
| [MTP 到 2× 的邊界（攤薄算術）](mtp-2x.md) | 3b | k 加到多大都不到 2×（server a/m=1.445、bench 0.982）；現況每產出 token 成本 45.6~56.4 vs baseline 48.2 ⇒ 攤薄  |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/MTP_IN_LLAMA_BENCH_2026-09-18.md／MTP_LAUNCH_REQUIRED_PARAMS_2026-09-18.md` |
| 備註 | ★ 前身：MTP off 9.82／on 12.62 是 HTTP 口徑 ⇒ 已作廢（契約 §7） |
| 軸性質 | 活躍攻關軸：speculative 攤薄係數 m 與 accept rate a |
| 對應報告 | 6 份 |

- [MTP_HEAD_PROVENANCE_GATE_2026-09-14.md](../../MTP_HEAD_PROVENANCE_GATE_2026-09-14.md)
- [MTP_INSTRUMENT_PLAN_2026-09-17.md](../../MTP_INSTRUMENT_PLAN_2026-09-17.md)
- [MTP_IN_LLAMA_BENCH_2026-09-18.md](../../MTP_IN_LLAMA_BENCH_2026-09-18.md)
- [MTP_LAUNCH_REQUIRED_PARAMS_2026-09-18.md](../../MTP_LAUNCH_REQUIRED_PARAMS_2026-09-18.md)
- [PROD_NEW_MTP_OFF_2026-09-23.json](../../PROD_NEW_MTP_OFF_2026-09-23.json)
- [PROD_NEW_MTP_OFF_2026-09-23.md](../../PROD_NEW_MTP_OFF_2026-09-23.md)

---

← [異步 gather 流水線（單段＋miss 後台補＋局部重算）](s1-asyncgather.md)　·　[總目錄](index.md)　·　[HTML 版](mtp-instrument.html)　·　[MTP 口徑與混淆定位（pool／跨啟動／順序） →](mtp-caliper.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
