# 把段邊界重疊柵欄（CGC_OVERLAP_FENCE=41）搬到**… — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：把段邊界重疊柵欄（CGC_OVERLAP_FENCE=41）搬到**現行的交付格**上值多少？ 柵欄在舊 (default) 形狀（-b 5632 -p 2048 -d 512 --warm-skip 64）上量到 C12 13.867389（乾淨窗） 與 C13 15.210976（降級窗），但**從未在交付格（llama-bench 出廠形狀 -p 512 -b 2048 -ub 512 -d 0 -r 5 --warm-skip 0）跑過** —— 這是「20+ 判死」之後唯一有名字、且沒有被否證的一步。

- 主題：實驗　·　子目標：**S 序列化消減**（活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產

---

## 1. 目標

把段邊界重疊柵欄（CGC_OVERLAP_FENCE=41）搬到**現行的交付格**上值多少？ 柵欄在舊 (default) 形狀（-b 5632 -p 2048 -d 512 --warm-skip 64）上量到 C12 13.867389（乾淨窗） 與 C13 15.210976（降級窗），但**從未在交付格（llama-bench 出廠形狀 -p 512 -b 2048 -ub 512 -d 0 -r 5 --warm-skip 0）跑過** —— 這是「20+ 判死」之後唯一有名字、且沒有被否證的一步。

## 2. 判準

同一 invocation 拿到 off/on 兩臂的 pp512＋tg128；tg 比值 ≥1.10 ⇒ 柵欄在交付格上成立（即使窗不可引用，同場比值仍可讀）

## 3. 結果

尚無可引用讀數：1 個 run 全部非乾淨（attribution.verdict=swap（swap_growth=3431.74 MiB, max_swap=7984.94 MiB））

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 假設：柵欄把段邊界的 drain 與 hook 重疊（早提交 ＋ MTLSharedEvent：host 寫完 remap leaf 後 signalEvent，GPU 等到 leaf 就續跑 ⇒ host 不再 drain），紅利全在 wait。 交付格與舊格的差別是 -d 0（池在 decode 起點是冷的）與 warm_skip 0（冷的 rep 也計入） ⇒ 段邊界往返的**次數與 miss 分布**不同，但柵欄收割的是**近乎固定的段邊界成本** ⇒ 方向應不變、幅度可能縮小。

---

## 5. 子目標綁定 · options · 測試 Log 報告

- **子目標**：—（未歸屬看板 15 格中的任何一格）
- ⚠ 本條不屬於看板 15 個子目標中的任何一格（已認證／已定案／已作廢）⇒ 沒有待跑的臂，也就沒有要綁的 option。
- **逐條處置**：已定案（約束／基準）　—　柵欄 × 現行交付格的兩方向配對：OFF 10.840 → ON 13.722 tg t/s ＝ ×1.266（① ×1.278／② ×1.254，順序無關，四臂 tg spread ≤1.082）。⇒ 定案兩條：(a) 柵欄在交付格成立、幅度大於舊形狀（×1.120）；(b) 13.72 ≈ 舊形狀乾淨窗最高 13.8674 ⇒ 形狀換成 llama-bench 出廠形狀沒有損失速度。⚠ 20 仍差 +45.8% ⇒ 20+ 仍判死（「唯一未否證的一步」已花掉）。產物 Backup/spec_tree/fence_default_20261005/fence.json ＋ fence_rev.json
- **測試 log（實跑）**：[fence.json](../../../Backup/spec_tree/fence_default_20261005/fence.json)　[fence.json](../../../Backup/spec_tree/fence_default_20261005/fence.json)　[fence_rev.json](../../../Backup/spec_tree/fence_default_20261005/fence_rev.json)　[fence_rev.json](../../../Backup/spec_tree/fence_default_20261005/fence_rev.json)　[sp_fence_on_clean_s01.json](../../../Backup/spec_tree/rs0_sampling/sp_fence_on_clean_s01.json)　[sp_fence_on_clean_s02.json](../../../Backup/spec_tree/rs0_sampling/sp_fence_on_clean_s02.json)　[sp_fence_on_clean_s03.json](../../../Backup/spec_tree/rs0_sampling/sp_fence_on_clean_s03.json)　[sp_fence_on_clean_s04.json](../../../Backup/spec_tree/rs0_sampling/sp_fence_on_clean_s04.json)　[sp_fence_on_clean_s05.json](../../../Backup/spec_tree/rs0_sampling/sp_fence_on_clean_s05.json)　[sp_fence_on_clean_s06.json](../../../Backup/spec_tree/rs0_sampling/sp_fence_on_clean_s06.json)
- 柵欄 × 現行交付格的兩方向配對：OFF 10.840 → ON 13.722 tg t/s ＝ ×1.266（① ×1.278／② ×1.254，順序無關，四臂 tg spread ≤1.082）。⇒ 定案兩條：(a) 柵欄在交付格成立、幅度大於舊形狀（×1.120）；(b) 13.72 ≈ 舊形狀乾淨窗最高 13.8674 ⇒ 形狀換成 llama-bench 出廠形狀沒有損失速度。⚠ 20 仍差 +45.8% ⇒ 20+ 仍判死（「唯一未否證的一步」已花掉）。產物 Backup/spec_tree/fence_default_20261005/fence.json ＋ fence_rev.json

## 6. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [CGC_EB_NOFILL 診斷臂（fill 成本）](io-nofill.md) | 3a | fill 3.955 → 0.206 ms/step（−95%）⇒ 機制證；生產口徑同步 fill 僅占 step 4.7% |
| [單段提交（41 段 → 1 段）](s1-segbatch.md) | 3a | 判詞：41 段→1 段的 A/B **作廢**（兩端都不可引用）。本節點不主張吞吐。**補（09-29）**：它的**可交付上界**＝序列化 − 必須還回去的 fill ⇒ **1 |
| [異步 gather 流水線（單段＋miss 後台補＋局部重算）](s1-asyncgather.md) | 3a | E0 已給出決定性答案：前提成立——S1（暖池 8 GiB）輸出 文摘文摘… 且 logits 全非有限（引擎自蓋 INVALID），同一輪的分段臂是 42 ⇒ 異步 fill＋補 |
| [ρ 路線（按層批次化 prefetch）](cache-rho.md) | 3a | 判活，但本節點**不主張吞吐**：覆蓋 0.849、視窗 1.30~1.59 ms、每步付 4.76 ms GPU 插入 ⇒ 有前置條件。 |
| [prebind／方案 A（預指派 slot）](cache-prebind.md) | 3a | 判活，但本節點**不主張吞吐**：qu2/qu3 全過、qu1 邊緣（預指派 slot，提前一整步發起 ⇒ 視窗 ≈ 一步）。 |
| [G0–G7 序列化／調度消減（41 段→1 段、縮 union、藏 …](exp-s-retro.md) | 3a | 判詞：11.97 那一族**作廢**（該臂 9 個 run 全部非乾淨）。本節點不主張吞吐。 |

## 7. 運行設置與 Log（生產級腳本 prod-new ＋ 自己 option）

**Run 1** · 2026-10-05 12:14　·　thermal MODERATE　·　swap 4164 MiB→7917 MiB
- arm：`prod-new`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[fence.json](../../../Backup/spec_tree/fence_default_20261005/fence.json)
- 結果：pp=105.24　tg=10.69
- 判定：swap：swap_growth=3431.74 MiB, max_swap=7984.94 MiB

**Run 2** · 2026-10-05 12:14　·　thermal MODERATE　·　swap 4164 MiB→7917 MiB
- arm：`prod-new:CGC_OVERLAP_FENCE=41`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[fence.json](../../../Backup/spec_tree/fence_default_20261005/fence.json)
- 結果：pp=119.79　tg=13.66
- 判定：swap：swap_growth=329.0 MiB, max_swap=8006.25 MiB

**Run 3** · 2026-10-05 12:14　·　thermal MODERATE　·　swap 5530 MiB→7921 MiB
- arm：`prod-new:CGC_OVERLAP_FENCE=41`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[fence_rev.json](../../../Backup/spec_tree/fence_default_20261005/fence_rev.json)
- 結果：pp=118.07　tg=13.78
- 判定：swap：swap_growth=2132.0 MiB, max_swap=8029.81 MiB

**Run 4** · 2026-10-05 12:14　·　thermal MODERATE　·　swap 5530 MiB→7921 MiB
- arm：`prod-new`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[fence_rev.json](../../../Backup/spec_tree/fence_default_20261005/fence_rev.json)
- 結果：pp=107.43　tg=10.99
- 判定：swap：swap_growth=258.8799999999992 MiB, max_swap=8144.69 MiB

**Run 5** · 2026-10-06 08:09　·　thermal NOMINAL　·　swap 4019 MiB→7383 MiB
- arm：`prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=0;CGC_OVERLAP_FENCE=41`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[sp_fence_on_clean_s01.json](../../../Backup/spec_tree/rs0_sampling/sp_fence_on_clean_s01.json)
- 結果：pp=132.99　tg=14.37
- 判定：swap：swap_growth=3380.0600000000004 MiB, max_swap=7783.06 MiB

**Run 6** · 2026-10-06 08:14　·　thermal HEAVY　·　swap 5270 MiB→7280 MiB
- arm：`prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=0;CGC_OVERLAP_FENCE=41`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[sp_fence_on_clean_s02.json](../../../Backup/spec_tree/rs0_sampling/sp_fence_on_clean_s02.json)
- 結果：pp=151.7　tg=14.09
- 判定：both：thermal=HEAVY, swap_growth=2010.62 MiB

**Run 7** · 2026-10-06 08:37　·　thermal NOMINAL　·　swap 5257 MiB→7428 MiB
- arm：`prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=0;CGC_OVERLAP_FENCE=41`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[sp_fence_on_clean_s03.json](../../../Backup/spec_tree/rs0_sampling/sp_fence_on_clean_s03.json)
- 結果：pp=151.83　tg=14.26
- 判定：swap：swap_growth=2171.129999999999 MiB, max_swap=7732.5 MiB

**Run 8** · 2026-10-06 08:40　·　thermal HEAVY　·　swap 7116 MiB→7622 MiB
- arm：`prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=0;CGC_OVERLAP_FENCE=41`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[sp_fence_on_clean_s04.json](../../../Backup/spec_tree/rs0_sampling/sp_fence_on_clean_s04.json)
- 結果：pp=159.02　tg=14.79
- 判定：both：thermal=HEAVY, swap_growth=505.8100000000004 MiB

**Run 9** · 2026-10-06 08:45　·　thermal MODERATE　·　swap 5945 MiB→8321 MiB
- arm：`prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=0;CGC_OVERLAP_FENCE=41`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[sp_fence_on_clean_s05.json](../../../Backup/spec_tree/rs0_sampling/sp_fence_on_clean_s05.json)
- 結果：pp=145.73　tg=14.1
- 判定：swap：swap_growth=2375.6900000000005 MiB, max_swap=8400.69 MiB

**Run 10** · 2026-10-06 08:55　·　thermal HEAVY　·　swap 5708 MiB→8085 MiB
- arm：`prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=0;CGC_OVERLAP_FENCE=41`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[sp_fence_on_clean_s06.json](../../../Backup/spec_tree/rs0_sampling/sp_fence_on_clean_s06.json)
- 結果：pp=139.04　tg=13.83
- 判定：both：thermal=HEAVY, swap_growth=2377.13 MiB

## 8. 子目標分解（持續更新；2/4 完成）

- [x] 跑前立項（現狀/目標/假設/驗收） [證](../../../scripts/check/charters/exp-fence-default-cell-2026-10-05.yaml)
- [x] 用生產級腳本 prod-new ＋ 自己 option 跑實驗臂，留存 log
- [ ] 驗收：同一 invocation 拿到 off/on 兩臂的 pp512＋tg128；tg 比值 ≥1.10 ⇒ 柵欄在交付格上成立（即使窗不可引用，同場比值仍可讀）
- [ ] 否證條件：tg 比值 ≤1.00（on 沒更快）或 pp 也動 >5%（那就不是 decode 段的效果）⇒ 柵欄在交付格不成立，要把它從「唯一未否證的一步」劃掉

## 9. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `Backup/l201_accel/ovl_cert_b.json` |
| 備註 | 假設：柵欄把段邊界的 drain 與 hook 重疊（早提交 ＋ MTLSharedEvent：host 寫完 remap leaf 後 signalEvent，GPU 等到 leaf 就續跑 ⇒ host 不再 drain），紅利全在 wait。 交付格與舊格的差別是 -d 0（池在 decode 起點是冷的）與 warm_skip 0（冷的 rep 也計入） ⇒ 段邊界往返的**次數與 miss 分布**不同，但柵欄收割的是**近乎固定的段邊界成本** ⇒ 方向應不變、幅度可能縮小。 |
| 軸性質 | 活躍攻關軸：41 段提交的同步／資料搬運，主項可被工程手段消掉 |
| 對應報告 | 0 份 |

- （無）

---

← [2026-10-03 operator 裁定「移除所有具名格、pro…](exp-default-cell-first-2026-10-03.md)　·　[總目錄](index.md)　·　[HTML 版](exp-fence-default-cell-2026-10-05.html)　·　[同一趟（同 build、同格、同 session）裡： ① caps… →](e-victim-classify-2026-10-05.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
