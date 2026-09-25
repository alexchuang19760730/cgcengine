# MTP 到 2× 的邊界（攤薄算術） — 技術白皮書　·　3b ③b 實驗目標達成（可放生產）

> **一句話**：算清「MTP 加速比 2」需要什麼條件（攤薄算術），判斷這個目標值不值得追、缺的是 k 還是別的。

- 主題：MTP／spec　·　子目標：**M MTP on 加速**（活躍攻關軸：speculative 攤薄係數 m 與 accept rate a）
- 階段：生產（可放生產／已交付）　·　① ② ③b —— 已進生產、或已是生產決策依據

---

## 1. 目標

算清「MTP 加速比 2」需要什麼條件（攤薄算術），判斷這個目標值不值得追、缺的是 k 還是別的。

## 2. 判準

S = (1+a·k)/(1+m·k)，k→∞ 上界 = a/m

## 3. 結果

k 加到多大都不到 2×（server a/m=1.445、bench 0.982）；現況每產出 token 成本 45.6~56.4 vs baseline 48.2 ⇒ 攤薄 ≈ 0；2× 需 verify 邊際 42.03 → ≤17.9 ms（−57%）

## 4. 判定

**3b · ③b 實驗目標達成（可放生產）** — 產物已可放進生產級設置：不破壞正確性 ∧ 成本可接受 ∧ 無前置條件

> ★ 產品化結論：『MTP on 加速比 2』不成立 ⇒ 改寫成 m/acc 兩條可攻目標

---

## 5. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [MTP 儀器化（接進 llama-bench）](mtp-instrument.md) | 3b | 接通（真因＝test_gen_spec 的 n_past 初始化）；此後 MTP 不再跨儀器比較 |
| [MTP 口徑與混淆定位（pool／跨啟動／順序）](mtp-caliper.md) | 3b | 定案：任何單點 MTP 數字不可引用；pool 4 vs 8 GiB 是一階混淆；同 launch 配對 ×1.06、生產 server 路徑 ×0.695（方向相反） |

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/MTP_2X_BOUNDARY_2026-09-19.md／docs/MTP_AMORTIZATION_2026-09-25.html` |
| 備註 | ★ 產品化結論：『MTP on 加速比 2』不成立 ⇒ 改寫成 m/acc 兩條可攻目標 |
| 軸性質 | 活躍攻關軸：speculative 攤薄係數 m 與 accept rate a |
| 對應報告 | 2 份 |

- [MTP_2X_BOUNDARY_2026-09-19.md](../../MTP_2X_BOUNDARY_2026-09-19.md)
- [REUSE_DISTANCE_MTPON_CONFIRM_2026-09-20.md](../../REUSE_DISTANCE_MTPON_CONFIRM_2026-09-20.md)

---

← [MTP k-sweep（verify batch T 成本曲線）](mtp-ksweep.md)　·　[總目錄](index.md)　·　[HTML 版](mtp-2x.html)　·　[RSL-MTP（draft top-8 ⊆ 已付費並集）＋ 練 draft head →](mtp-rsl.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
