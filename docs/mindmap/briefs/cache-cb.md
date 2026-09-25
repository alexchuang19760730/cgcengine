# cb 口徑定讞（42 vs 74） — 技術白皮書　·　3b ③b 實驗目標達成（可放生產）

> **一句話**：定讞「cb」是什麼、成本多少（42 vs 74 ms 之爭）——它是 expert cache 填池 IO，不是 command buffer。

- 主題：池／快取　·　子目標：**S 序列化消減**（活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉）
- 階段：生產（可放生產／已交付）　·　① ② ③b —— 已進生產、或已是生產決策依據

---

## 1. 目標

定讞「cb」是什麼、成本多少（42 vs 74 ms 之爭）——它是 expert cache 填池 IO，不是 command buffer。

## 2. 判準

交付 mean 口徑重算

## 3. 結果

定讞 42~51 ms；74.18 撤回

## 4. 判定

**3b · ③b 實驗目標達成（可放生產）** — 產物已可放進生產級設置：不破壞正確性 ∧ 成本可接受 ∧ 無前置條件

> 已作為後續所有上界計算的單價

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [池大小掃描（8→4→2 GiB）](io-poolsize.md) | 3b | 4 GiB 10.60 vs 8 GiB 10.28＝+3.1%（噪音內）；4 GiB miss 1.91×、capacity miss 3.15× 而 t/s 不動 |
| [S1 探針臂：slot table 放 GPU（數值身分）](s1-probe.md) | 3b | 576/576 全同；answer_md5 相同 |
| [3a：未填充 expert 貢獻歸零（MISS_MASK / ZERO_MISS）](miss-3a.md) | 3b | 兩條都過（38 層/421 元素/0 差異；Δ = −0.95 ± 2.01 ms） |
| [expert cache 血統設計（09-05~09-09 期）](na-design.md) | 3b | HYBRID_DESIGN／INVARIANTS／INTEGRATION_DIFF／COMMIT_DIGEST：血統進了生產的 expert cache（現行 pool 即其後代） |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/CB_DELIVERY_SETTLED_2026-09-23.md` |
| 備註 | 已作為後續所有上界計算的單價 |
| 軸性質 | 活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉 |
| 對應報告 | 0 份 |

- （無）

---

← [prebind／方案 A（預指派 slot）](cache-prebind.md)　·　[總目錄](index.md)　·　[HTML 版](cache-cb.html)　·　[K3 邊際單價（dispatch 成本） →](k3-price.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
