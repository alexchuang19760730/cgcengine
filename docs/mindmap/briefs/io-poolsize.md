# 池大小掃描（8→4→2 GiB） — 技術白皮書　·　3b ③b 實驗目標達成（可放生產）

> **一句話**：掃描 expert cache 池大小（8 → 4 → 2 GiB），量出池容量對 decode 的敏感度，判斷「加大池」能不能換到吞吐。

- 主題：fill／IO　·　子目標：**S 序列化消減**（活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉）
- 階段：生產（可放生產／已交付）　·　① ② ③b —— 已進生產、或已是生產決策依據

---

## 1. 目標

掃描 expert cache 池大小（8 → 4 → 2 GiB），量出池容量對 decode 的敏感度，判斷「加大池」能不能換到吞吐。

## 2. 判準

同 cell 配對，看 t/s 與 miss

## 3. 結果

4 GiB 10.60 vs 8 GiB 10.28＝+3.1%（噪音內）；4 GiB miss 1.91×、capacity miss 3.15× 而 t/s 不動

## 4. 判定

**3b · ③b 實驗目標達成（可放生產）** — 產物已可放進生產級設置：不破壞正確性 ∧ 成本可接受 ∧ 無前置條件

> ★ 與白皮書「8G→4G −21%」矛盾且從未仲裁；另 3 GiB 那格從未乾淨量過

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [S1 探針臂：slot table 放 GPU（數值身分）](s1-probe.md) | 3b | 576/576 全同；answer_md5 相同 |
| [3a：未填充 expert 貢獻歸零（MISS_MASK / ZERO_MISS）](miss-3a.md) | 3b | 兩條都過（38 層/421 元素/0 差異；Δ = −0.95 ± 2.01 ms） |
| [cb 口徑定讞（42 vs 74）](cache-cb.md) | 3b | 定讞 42~51 ms；74.18 撤回 |
| [expert cache 血統設計（09-05~09-09 期）](na-design.md) | 3b | HYBRID_DESIGN／INVARIANTS／INTEGRATION_DIFF／COMMIT_DIGEST：血統進了生產的 expert cache（現行 pool 即其後代） |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/IO_PATH_AB_2026-09-21.md（MEMORY_PERF §EN-359 自記「別再花時間在 pool 大小上」）` |
| 備註 | ★ 與白皮書「8G→4G −21%」矛盾且從未仲裁；另 3 GiB 那格從未乾淨量過 |
| 軸性質 | 活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉 |
| 對應報告 | 1 份 |

- [IO_PATH_AB_2026-09-21.md](../../IO_PATH_AB_2026-09-21.md)

---

← [decode ≥ 25（M-25）](m-decode25.md)　·　[總目錄](index.md)　·　[HTML 版](io-poolsize.html)　·　[IO 請求形狀（合併 pread） →](io-shape.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
