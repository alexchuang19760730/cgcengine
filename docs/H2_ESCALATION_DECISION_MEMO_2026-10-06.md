# H2 升級裁決備忘（一頁）

2026-10-06 · 分支 `demo/sweet-spot-windows-fix` · 判詞：`Backup/spec_tree/F1F2F3_IMPLEMENT_2026-10-06.md` §13（§13.6 ＝ BRESET 否證）·
交付面：`docs/MTPON_SPECTREE_FIXPLAN_WHITEPAPER_2026-10-06.html` §2 V8

## 要裁決什麼

H2（spec-tree 兩條鏈共批）在 bench 側的縫已經縫完（C 的 rep-loop clear、BRESET），剩下的解釋只有兩個，而它們在
唯一可用的自然實驗裡**與 treatment 完全共線**：

| 剩下的解釋 | 為什麼拆不開 |
|---|---|
| ① 驗證形狀共線（`alt_on=1 ⇒ W_eff=2 ⇒ 6 列` vs `alt_on=0 ⇒ 3 列鏈`） | rgate 臂裡「共批」與「6 列」是同一個開關 |
| ② 共用 seq 0 的 hidden row（B 的第一顆 `memcpy` 自 seq 0 的 `h_row`） | 同上；且它就在 alt-seed 那三行裡 |

⇒ 問題是：**升到 `common/speculative.cpp` 把它們拆開，還是明說本線到此？**

## 事實底（都可回查，無 t/s）

- **機制已確認且可量**：`h_orcrgate40` 逐輪 `alt_on` 自然實驗，A 的草稿品質 B-on 0.486（18/37）vs B-off 0.845（197/233），
  Δ=−0.359、z=−5.04；逐 rep 同方向；B-off 段內前 3 → 後 3 ＝ 0.619 → 0.905（7 段、中位 40）⇒ **帶記憶，不是常數懲罰**。
- **bench 側兩條衛生槓桿都用完了**：C 的 rep-loop clear、BRESET。BRESET 否證了殘留假說 —— treatment 與對照
  **逐位元相同**（oracle 逐輪 281 行／ACCEPT 5／SPEC-TREE 10 三條頻道全同），可否證性控制 `e3_k2_br` 亦逐位元相同（如設計）。
- **共批不是假說、是實作**：`speculative.cpp:1980-1993` 把 B 的第一顆用 `common_batch_add(..., {1}, true)` 加進 **seq 0 同一個 batch**
  並 `memcpy` 共用 seq 0 的 hidden row；bench 只呼叫一次 `common_speculative_draft()`（`llama-bench.cpp:3328`）⇒ **bench 沒有「單獨 draft B」的入口**。
- **t/s 不是決策儀器**：同機制重複噪音底 15–43% ≫ 門檻 9% ⇒ 任何選項的驗收都只能用機制頻道。

## 選項 A — 升級到 `speculative.cpp`（拆共批／拆共用 hidden row）

**入場條件（缺一不可）**
1. **先拆選擇效應**（bench 側、零 `speculative.cpp`）：重跑週期閘臂 `CGC_SPEC_TREE_BGATE=16` 帶 `LLAMA_BENCH_SPEC_ORACLE=1`。
   若「不依賴品質反應」的閘下 B-on 罰消失 ⇒ H2 是選擇假象，**A 當場作廢**。
2. **檔的擁有權**：`speculative.cpp`／`speculative.h` 目前 dirty（他線在編輯）⇒ 需協調（等他們的 commit 落地，或先約定 hunk 邊界）。
3. **契約要保住**：`n_seq == 1` 位元不變（`speculative.h:101-102` 的 `alt_branch` 預設 false 承諾、`speculative.cpp:1379-1381` 的 batch 註解）＋
   重建後的 `build/bin` 產物必須與原始碼**同一個 commit**（pre-commit 硬規則）。
4. **正控**：`llama_memory_seq_rm` 的回傳值要記（否則「拆掉了」又落入「逐位元相同有兩種讀法」）。

**風險**
- 動到他線正在編輯的檔 —— 本清單風險最高的一項，且衝突會污染他們的軌跡。
- 改的是 **draft 路徑本身**（B 的 seed 數值會變）⇒ 不是衛生改動，舊臂全部不可比。
- 產出是**機制歸因，不是速度證明**（t/s 不可引用）⇒ 對 25 t/s 目標的邊際貢獻未定。
- 可能還要動 API（per-seq 的 draft 入口）；成本＝一次重建 ＋ GPU session。

## 選項 B — 本線到此（明說停）

- 把 H2 記成「**機制已確認、因果未歸因**」，MTP-on 仍**未轉正**；所有未 commit 的實驗碼的去留一併標明。
- 留下的唯一問號是**設計層**的：若代價來自 6 列 verify 形狀，那 width-2 樹本身就是錯的形狀。
- 風險：把最大的設計問號交接出去，但不必碰他線、不必再燒一趟 GPU。

## 選項 C — 本線可試的第三條（成本最低，且唯一能在本線產生因果）

bench 側**自己造第二條鏈**，讓 draft step 沒有共批：seq 0 的 `.alt` 綁到自己的向量（impl 會把逐位 runner-up 寫進去，
`speculative.cpp:1962-1963`），seq 1 **不綁**、`alt_branch=false` ⇒ alt-seed 不觸發；bench 再用 `alt0` 當 branch 1 的 root
造出 6 列 verify。⇒ **把「6 列」與「共批」解耦，完全不碰 `speculative.cpp`。**

- 讀數的正確寫法：branch 1 的尾巴是 seq 0 逐位的 runner-up ⇒ 這是「**6 列 ＋ runner-up 尾巴**」，不是「6 列 ＋ 獨立鏈」。
- ~~未驗證：先前**同時**綁兩個臂的 `.alt` 造成過 interleave 錯誤；單綁 seq 0 是否安全未試。~~
  **2026-10-06 已實作並驗證**（見下節）：單綁 seq 0 安全（同時只有一個寫者，因為 `alt_branch` 被強制關 ⇒ seq 1 永不 draft）。

## 建議

1. 先做 **C 的第一格**（純讀碼 ＋ 跑得起來驗證，零 GPU），同時做 **A 的入場條件 1**（週期閘臂，零 `speculative.cpp`）。
2. 若週期閘下 B-on 罰仍在、且 C 的「6 列 vs 3 列」對照給出方向 ⇒ **才**值得付 A 的協調成本。
3. 否則選 **B**，並把「6 列是否為代價來源」這個問題明確交接給 `speculative.cpp` 的擁有者。

## 自我反對（兩種都會犯的錯）

- **直接升級**：在選擇效應還沒拆掉的情況下改 `speculative.cpp`，等於在一個混雜的實驗上量因果。
- **直接宣布到此**：等於放棄唯一一條能回答「6 列是否為代價來源」的低成本路徑。

## 更新（2026-10-06）：選項 C 已落地 —— 形狀成為可掃的臂參數

兩個新 env 旋鈕（都在 `scripts/run_server.sh` 的 allowlist 裡；預設關 ⇒ 既有 cell 位元不動）：

| 旋鈕 | 作用 |
|---|---|
| `CGC_SPEC_TREE_ALTBIND=1` | 第二分支改由 seq 0 的 runner-up 頻道供給 ⇒ **draft step 沒有 alt-seed**（＝沒有共批、沒有共用 hidden row），列數不變 |
| `CGC_SPEC_TREE_ROWS=<n>` | 直接指定 verify 列數（`d = floor((n - W_eff)/W_eff)`）；不可達時見證留 `rows_unmet` |

見證（`CGC-SPEC-TREE` 行尾新增四個欄位）：`rows_req= / rows_d= / rows_unmet= / altbind=`。

**已驗證的判準**（煙霧 cell `-p16 -n16 -d0 -r1`，機制頻道，零 t/s 引用）：

| cell | 列數/輪 | `rows_req` | `rows_unmet` | `altbind` | alt-seed 行數 |
|---|---|---|---|---|---|
| 預設（alt-seed） | **6.0** | 0 | 0 | 0 | 11 |
| `ROWS=4` | **4.0** | 4 | 0 | 0 | 11 |
| `ALTBIND=1` | **6.0** | 0 | 0 | 1 | **0** |
| `ALTBIND=1 ROWS=4` | **4.0** | 4 | 0 | 1 | **0** |
| `ALTBIND=1 ROWS=8`（深度 3） | **8.0** | 8 | 0 | 1 | **0** |
| `ROWS=7`（不可達） | 6.0 | 7 | **13** | 0 | 13 |

- 預設路徑**位元不變**：同一 cell 在改動前後各跑兩次（`--fixed-fill-seed 1`），五條機制頻道逐字元相同，只有新增的四個欄位是 0。
- 端到端一走（`harness bench`，`rows_b4`）：`rc=0`、provenance 未拒、env 記錄含兩個新鍵、`tree_rows/rounds_tree = 292/73 = 4.0`、`alt-seed 行數=0`、`X<Y=0`、每步 `ntok` 直方圖只有 {1, 4, 512}（**沒有 6 列的步**）。

**對選項 A 的後果（重要）**：V8 說「拆驗證形狀共線必須改 `speculative.cpp`」—— **這一句現在被本線推翻了**：
「6 列 ＋ 共批」（`rows_a6`）與「6 列 ＋ 無共批」（`rows_b6`）已在同一個 bench 上並存，
所以形狀與共批**可以在不改引擎的情況下分離**。選項 A 的入場條件因此從「必須改引擎」降級成
「先跑 C 的配對，看共批還剩多少解釋力」。掃描器：`Backup/spec_tree/f1f2_diag/run_diag_rows.sh`。

---

本備忘**不新增任何 t/s 讀數**（§12.7）。所有機制數字皆出自 `Backup/spec_tree/F1F2F3_IMPLEMENT_2026-10-06.md` §13 與
`docs/MTPON_SPECTREE_FIXPLAN_WHITEPAPER_2026-10-06.html` §2 V8；工具 `Backup/spec_tree/h2_strata.py`（只讀日誌）。
`speculative.cpp`／`speculative.h` 在他線手上、**正在被編輯**，本線不動；上面的行號是 2026-10-06 本工作樹的快照，該檔一動就會漂移（引用以符號為準）。
