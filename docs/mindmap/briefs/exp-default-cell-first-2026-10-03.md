# 2026-10-03 operator 裁定「移除所有具名格、pro… — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：2026-10-03 operator 裁定「移除所有具名格、prod-new ＋ harness bench 完全貼齊 llama-bench 出廠形狀」之後，**新的預設 cell（唯一可選格）在交付面上值多少**？ 即：`-p 512 -n 128 -b 2048 -ub 512 -d 0 -r 5`、warmup 開、`warm_skip 0`、`ctx_size 0`， 記憶體側維持 prod-new（ngl 99／load_mode none／池 8 GiB／KV q8_0）之下， `pp512` 與 `tg128` 兩列的第一筆讀數是多少、`attribution` 是哪一級？

- 主題：實驗　·　子目標：**不適用**（非攻關軸：約束／帳本／上界算術／入口索引／跨線產品）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產

---

## 1. 目標

2026-10-03 operator 裁定「移除所有具名格、prod-new ＋ harness bench 完全貼齊 llama-bench 出廠形狀」之後，**新的預設 cell（唯一可選格）在交付面上值多少**？ 即：`-p 512 -n 128 -b 2048 -ub 512 -d 0 -r 5`、warmup 開、`warm_skip 0`、`ctx_size 0`， 記憶體側維持 prod-new（ngl 99／load_mode none／池 8 GiB／KV q8_0）之下， `pp512` 與 `tg128` 兩列的第一筆讀數是多少、`attribution` 是哪一級？

## 2. 判準

得到一筆 cell 口徑齊備的 pp512／tg128 讀數（contract ok=True、cell 名 (default)、逐 rep 向量），並如實標注 attribution；pp 落在 100–136 且 tg 落在 8.0–11.0 之內 ⇒ 預期成立

## 3. 結果

尚無可引用讀數：1 個 run 全部非乾淨（attribution.verdict=swap（swap_growth=3122.3099999999995 MiB, max_swap=7770.69 MiB））

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 假設：與 delivery-std 首筆（`-b 512 -d 512 --warm-skip 64`）相比有兩個實質變化： ① `--warm-skip 64 → 0` ⇒ tg 那一列不再丟掉前 64 顆，量的就是 llama-bench 的 tg128； ② `-d 512 → 0` ⇒ **沒有 depth 預填**，池在 decode 起點是冷的，前幾顆會走缺頁路徑。 兩者都指向「同一個引擎、同一個窗口下 tg 會比 11.885 低」，但那正是 llama-bench 的數字。 `-b 2048 -ub 512` 對 `-p 512` 沒有實質影響（ubatch 仍是 512 ⇒ 一個 ubatch），pp 應與 117.6 同級。

---

## 5. 子目標綁定 · options · 測試 Log 報告

- **子目標**：—（未歸屬看板 15 格中的任何一格）
- ⚠ 本條不屬於看板 15 個子目標中的任何一格（已認證／已定案／已作廢）⇒ 沒有待跑的臂，也就沒有要綁的 option。
- **逐條處置**：已定案（約束／基準）　—　新預設格（llama-bench 出廠形狀）的第一次讀數：pp512 94.20／tg128 10.60，0/2 可引用（兩列全 rep max/min 各 1.512／1.114 皆 &gt;1.10，且 attribution=swap）⇒ 定案的是約束不是成績：出廠形狀（-d 0 -r 5 --warm-skip 0）在本機 5 reps 內到不了穩態。產物 Backup/spec_tree/default_cell_20261004/default_cell.json
- **測試 log（實跑）**：[default_cell.json](../../../Backup/spec_tree/default_cell_20261004/default_cell.json)
- 新預設格（llama-bench 出廠形狀）的第一次讀數：pp512 94.20／tg128 10.60，0/2 可引用（兩列全 rep max/min 各 1.512／1.114 皆 &gt;1.10，且 attribution=swap）⇒ 定案的是約束不是成績：出廠形狀（-d 0 -r 5 --warm-skip 0）在本機 5 reps 內到不了穩態。產物 Backup/spec_tree/default_cell_20261004/default_cell.json

## 6. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [k=3 的 1.43× 飄移定位 ＋ 配對認證](k3-swing.md) | 3a | 飄移已定位；配對認證 n=5 時 t=2.08 未達 df=4 的 2.776 ⇒ 16.4% 仍未認證；上一輪「bench 配對 sd 更小」被自我撤回 |
| [口徑＋k=3 飄移認證（16.4% 噪聲底）](exp-caliber-calibration.md) | 3a | **未跑（有卡、無讀數）**：本節點的產出是**口徑校準**——記錄裡至少有四個不同的「步」（DECPROF work-row **167.81** ms／實付週期 step_ro |
| [**跨 checkpoint 探針**：把 CGC 引擎（交付路徑＝…](exp-cgc-edge0gguf-2026-10-05.md) | 3a | 實驗進行中（尚無讀數） |
| [**交付口徑量測（生產級）**：spec-tree MTP-on（`…](exp-spectree-mtpon-default-2026-10-06.md) | 3a | decode 11.36 t/s |
| [兩份權威檔對同一個里程碑的「是否已判死」講了相反的話，而沒有閘門會發…](exp-targets-board-recon-2026-10-07.md) | 3a | 實驗進行中（尚無讀數） |

## 7. 運行設置與 Log（生產級腳本 prod-new ＋ 自己 option）

**Run 1** · 2026-10-04 22:26　·　thermal NOMINAL　·　swap 4160 MiB→7275 MiB
- arm：`prod-new`
- 命令：`llama-bench（prod-new）-p 512 -n 128 --warm-skip None（由產物參數重建）`
- Log／產物：[default_cell.json](../../../Backup/spec_tree/default_cell_20261004/default_cell.json)
- 結果：pp=94.2　tg=10.6
- 判定：swap：swap_growth=3122.3099999999995 MiB, max_swap=7770.69 MiB

## 8. 子目標分解（持續更新；2/4 完成）

- [x] 跑前立項（現狀/目標/假設/驗收） [證](../../../scripts/check/charters/exp-default-cell-first-2026-10-03.yaml)
- [x] 用生產級腳本 prod-new ＋ 自己 option 跑實驗臂，留存 log
- [ ] 驗收：得到一筆 cell 口徑齊備的 pp512／tg128 讀數（contract ok=True、cell 名 (default)、逐 rep 向量），並如實標注 attribution；pp 落在 100–136 且 tg 落在 8.0–11.0 之內 ⇒ 預期成立
- [ ] 否證條件：若 tg > 11.9（高於帶 depth 預填的舊形狀）或 < 8.0（低於冷池可解釋的範圍）⇒ 機制解釋錯，要改寫「-d 0 冷池」這條歸因，不得把它當基準

## 9. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/PROD_NEW_TEST_CARD_2026-09-24.md` |
| 備註 | 假設：與 delivery-std 首筆（`-b 512 -d 512 --warm-skip 64`）相比有兩個實質變化： ① `--warm-skip 64 → 0` ⇒ tg 那一列不再丟掉前 64 顆，量的就是 llama-bench 的 tg128； ② `-d 512 → 0` ⇒ **沒有 depth 預填**，池在 decode 起點是冷的，前幾顆會走缺頁路徑。 兩者都指向「同一個引擎、同一個窗口下 tg 會比 11.885 低」，但那正是 llama-bench 的數字。 `-b 2048 -ub 512` 對 `-p 512` 沒有實質影響（ubatch 仍是 512 ⇒ 一個 ubatch），pp 應與 117.6 同級。 |
| 軸性質 | 非攻關軸：約束／帳本／上界算術／入口索引／跨線產品 |
| 對應報告 | 0 份 |

- （無）

---

← [M2 slab pool-reuse（整層填的 memcpy 省多少）](e-m2-slab-reuse.md)　·　[總目錄](index.md)　·　[HTML 版](exp-default-cell-first-2026-10-03.html)　·　[把段邊界重疊柵欄（CGC_OVERLAP_FENCE=41）搬到**… →](exp-fence-default-cell-2026-10-05.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
