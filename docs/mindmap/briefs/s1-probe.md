# S1 探針臂：slot table 放 GPU（數值身分） — 技術白皮書　·　3b ③b 實驗目標達成（可放生產）

> **一句話**：把 slot table 放上 GPU（CGC_SLOT_TABLE_GPU），證明單段提交的查表與 host 版逐位相同（數值身分）。

- 主題：S1／段邊界　·　子目標：**S 序列化消減**（活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉）
- 階段：生產（可放生產／已交付）　·　① ② ③b —— 已進生產、或已是生產決策依據

---

## 1. 目標

把 slot table 放上 GPU（CGC_SLOT_TABLE_GPU），證明單段提交的查表與 host 版逐位相同（數值身分）。

## 2. 判準

預先寫死：逐層 fnv1a64 指紋 ＋ answer_md5

## 3. 結果

576/576 全同；answer_md5 相同

## 4. 判定

**3b · ③b 實驗目標達成（可放生產）** — 產物已可放進生產級設置：不破壞正確性 ∧ 成本可接受 ∧ 無前置條件

> ★ 本臂不可報吞吐（同節寫明「速度：無主張」）

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [池大小掃描（8→4→2 GiB）](io-poolsize.md) | 3b | 4 GiB 10.60 vs 8 GiB 10.28＝+3.1%（噪音內）；4 GiB miss 1.91×、capacity miss 3.15× 而 t/s 不動 |
| [3a：未填充 expert 貢獻歸零（MISS_MASK / ZERO_MISS）](miss-3a.md) | 3b | 兩條都過（38 層/421 元素/0 差異；Δ = −0.95 ± 2.01 ms） |
| [cb 口徑定讞（42 vs 74）](cache-cb.md) | 3b | 定讞 42~51 ms；74.18 撤回 |
| [expert cache 血統設計（09-05~09-09 期）](na-design.md) | 3b | HYBRID_DESIGN／INVARIANTS／INTEGRATION_DIFF／COMMIT_DIGEST：血統進了生產的 expert cache（現行 pool 即其後代） |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/S1_LINE_VERDICT_2026-09-25.md §2.2b` |
| 備註 | ★ 本臂不可報吞吐（同節寫明「速度：無主張」） |
| 軸性質 | 活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉 |
| 對應報告 | 6 份 |

- [S1_DISCRIMINATION_RESULT_20260917.html](../../S1_DISCRIMINATION_RESULT_20260917.html)
- [S1_LAYER1_MOE_GATHER_20260917.html](../../S1_LAYER1_MOE_GATHER_20260917.html)
- [S1_MINIL_TRACKING_20260917.html](../../S1_MINIL_TRACKING_20260917.html)
- [S1_OUTPUT_CAPTURE_20260916_1955.html](../../S1_OUTPUT_CAPTURE_20260916_1955.html)
- [S1_OWNER_EXPECT_CHANNELS_20260917_1145.html](../../S1_OWNER_EXPECT_CHANNELS_20260917_1145.html)
- [S1_SRC2_INSTRUMENT_BLIND_20260917.html](../../S1_SRC2_INSTRUMENT_BLIND_20260917.html)

---

← [CGC_EB_NOFILL 診斷臂（fill 成本）](io-nofill.md)　·　[總目錄](index.md)　·　[HTML 版](s1-probe.html)　·　[單段提交（41 段 → 1 段） →](s1-segbatch.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
