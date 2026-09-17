# M1 工作項 4（canonical gather order）：實作、量測與離開條件重定（2026-09-17）

> 本檔是 `docs/M1_M4_M5_DEV_BRIEF_20260917_1525.html` §4.1／§4.3 的交付：
> 工作項 4 落地 ＋ **測試 A（置換不變性）** 離線通過 ＋ 真機三臂量測 ＋
> M1／M4 兩條離開條件改寫成可量測形式。
>
> 權威仍是 `docs/ROADMAP_PREFILL250_DECODE25_2026-09-13.md`；本文只修正它的離開條件
> 欄位（§4）並記錄實作（§1–§3）。規格來源見
> `docs/M1_WORKITEM4_CANONICAL_GATHER_ORDER_2026-09-17.md`。

---

## 1. 實作了什麼

| 檔案 | 內容 |
|---|---|
| `src/llama.cpp/src/llama-cgc-canon.h` | **新增**：`cgc_canon_build_perm()`／`cgc_canon_apply()`／`cgc_canon_order_mode()`。header-only，好讓離線測試直接 include，不需要編整個專案 |
| `src/llama.cpp/src/llama-graph.cpp` | 在 `build_moe_ffn` 的 pool-path 區塊**頂端**（slot-table／remap 分支之前）建 `ffn_moe_canon_perm`（1-D I32、`ggml_set_output`）並把 `selected_experts` 換成置換後的版本 |
| `src/llama.cpp/src/llama-context.cpp` | hook 端：在 nb-aware 快照之後算 perm、寫進 leaf、**把 `ids` 這個指標換成置換後的 id 向量**（⇒ 四個 remap 寫入點、S1 表、SLOT-SEL、prefetch 收集全部自動跟隨同一個 perm）；`graph_get_cb` 新增 `ffn_moe_canon_perm` 的捕獲 |
| `src/llama.cpp/src/llama-context.h` | `cache_canon_tensors`（含契約註解） |
| `scripts/run_server.sh` | `CGC_CANON_ORDER` 進 env allowlist（不進 allowlist 的 CGC_* 會被靜默丟掉 ⇒「沒效果」與「沒設到」同形） |
| `scripts/check/canon_order_selftest.cpp` | **新增**：測試 A（離線），含三條會失敗的對照 |

**語意**（`CGC_CANON_ORDER`）：

- `0`（預設）：完全不建置換節點，圖與工作項 4 之前相同。
- `1`：canonical —— 每個 token 的 k 個位置按 **expert id 遞增**排序，同 id 以**原位置**為次鍵（穩定）。
- `2`：**identity 對照** —— 建一樣的節點、一樣的 buffer 佈局，但 `perm[p] = p`。
  它存在的唯一理由是：把「重排序改變了數值」與「多了這些節點改變了數值」分開（§3-③ 證明這個區分是必要的，不是修辭）。

置換**同時**施加於 ids 與 weights：weights 是按位置 gather、按位置被 `ggml_mul` 消費，只排 ids 會把 A 專家的輸出配到 B 專家的權重（**安靜錯答，不是重排**）。實作上是「把 `selected_experts` 換成置換後的版本」一個動作，因為下游（weights gather、S1 gather、mmid）全部由它導出。

`perm` 的索引是 **ABSOLUTE**（0..k·T−1），不是 row-local：消費端是對**展平** id 向量的 `GET_ROWS`，row-local 索引會去讀別人的 token row —— 這與本專案已付過一次代價的「線性讀取 strided view」是同一類缺陷。測試 A 專門釘住這一條（§3-②）。

---

## 2. 量測（同一顆 binary，2026-09-17 16:00 建置；`prefill250` profile；oracle pin `CGC_SERVER_BATCH/UBATCH=6144`；MTP **ON**）

| run | 臂 | 池 | 參考 | M1 bit-identical | M2 argmax | M3 top-k | probe answer |
|---|---|---|---|---|---|---|---|
| r46 | canon **off** | 8 GiB | v6（8 GiB, canon off） | **9/9** | 9/9 | 9/9 | `42` |
| r53 | canon **off** | **4 GiB** | v6 | **9/9** | 9/9 | 9/9 | `42` |
| r48 | canon=2（identity） | 8 GiB | v6 | 1/9 | 9/9 | 2/9 | `42` |
| r49 | canon=1 | 8 GiB | r48 的 dump（**同圖**、只有 perm 值不同） | 1/9 | 9/9 | 8/9 | — |
| r50 | canon=1 | 8 GiB | v6 | 0/9 | 9/9 | 1/9 | `42` |
| r51 | canon=1 | 8 GiB | r50（同 config 重跑） | **9/9** | 9/9 | 9/9 | `42` |
| r52 | canon=1 | **4 GiB** | r50（canon=1 @ 8 GiB） | **9/9** | 9/9 | 9/9 | `42` |

讀法（三條，缺一條就會讀錯）：

1. **預設路徑未被改動**：r46 = 9/9 對 v6。canon off 時圖與改動前逐位元相同。
2. **canonical 跨池逐位元相同**：r52（4 GiB）對 r50（8 GiB）= M1 9/9 / M2 9/9 / M3 9/9。
3. **r51 = 9/9**：canon=1 @ 8 GiB 重跑對 r50 逐位元相同 ⇒ 引擎跨啟動決定性（所以 r52 的 9/9 不是「漂移碰巧」）。

跨配置的判定：r51/r52/r53 的 config_diffs 都使 gate 判 `INVALID COMPARISON`（池 bytes、`CGC_CANON_ORDER` 都是 numerics-determining env），所以 M1/M2/M3 是**用 `--allow-incomparable` 讀出來的 observed 值**，不是 gate 蓋章的 PASS。池欄位已由 launch log 的 `[budget] expert pool 4096 MiB` 驗證真的生效（見 §3-④）。

**這份 probe 不能當品質證據**：9 列、全部答 `42`、四臂都相同 ⇒ M2 9/9 只說明「這個樣本上決策沒變」，greedy 是混沌的，gate 自己的註解也這樣寫。品質要有意義得跑 48 題（本輪沒跑）。

---

## 3. 這輪學到的四條（都是量出來的，不是推理）

### ① `ggml_get_rows` 的 src0 是「表」，不是「向量」

第一次嘗試把展平的 id 向量（1-D、ne0 = k·T）當 src0 餵進去，模型載入時直接 abort：

```
ggml.c:3703: GGML_ASSERT(ggml_nelements(a) == ne0*ne1) failed
  ← ggml_reshape_2d ← build_moe_ffn          (Abort trap: 6，在任何 token 之前)
```

機制（讀 ggml.c:3931-3941）：`result = [a->ne[0], b->ne[0], b->ne[1], b->ne[2]]`，且 `a` 的行寬是 `a->ne[0]` —— 所以 1-D src0 既是「一行寬 n_sel」又是「輸出 [n_sel, n_sel]」。正解是把 id 向量 cast 成 **[1, n_sel]**（每個 row 一個 id），也就是 `slot_table` 的形狀。**這條是形狀規則，不是旋鈕**。

### ② row-local vs absolute perm

測試 A 第一次跑就抓到（`multi-token: canonical id block is identical across two input orders` FAIL）：實作原本吐 row-local 索引，T ≥ 2 時 `GET_ROWS` 會去讀**別的 token 的 row**。每一種值都還在範圍內 ⇒ 不會 assert。測試現在逐 token 斷言 `perm[p] ∈ [t·k, (t+1)·k)` 且是該 row 的雙射。

### ③ 「多了節點」本身就會動最後幾個位元 ⇒ identity 對照是必需品

r48（identity、**路由完全相同**）對 v6 是 **M1 1/9 / M2 9/9**：8 列 logits 在最後幾個位元不同，而 argmax 一個都沒變（cross-tab `num_ne_dec_eq = 8`）。這與 S1 時期「min_il=6 穩定不同」是同一個現象。

推論：**canonical 不能用 v6 當乾淨基準**（任何包含新節點的圖都不能）。但 mode 1 與 mode 2 的**圖完全相同、只有 perm 值不同**，所以 mode 2 才是 canonical 的乾淨對照臂 —— 這就是 `CGC_CANON_ORDER=2` 存在的理由，也解釋為什麼 §2 要把「canon off vs v6」與「canon=1 vs canon=2 同圖」分開列。

### ④ 閘門 `--env` 到不了 `CGC_SERVER_EXPERT_CACHE_BYTES`

r51 原本是要測 4 GiB，卻跑了 **8 GiB**：launch log 的 `budget=8589934592B` / `expert pool 8192 MiB` 自己講了。原因是那個變數由 `run_server.sh` 的 **shell** 消費（`BUDGET="${CGC_SERVER_EXPERT_CACHE_BYTES:-...}"`，:452），而 `--env` 只填 `SERVER_ENV`（傳給 server 二進位的那串）。**要掃池子必須把變數 export 到 gate 行程本身**：

```sh
CGC_SERVER_EXPERT_CACHE_BYTES=4294967296 python3 scripts/check/m123_oracle_gate.py ... --allow-incomparable
```

判準是 launch log 的 `[budget] expert pool N MiB`，不是「我傳了參數」。r52/r53 就是這樣跑的（4096 MiB 已驗證）。

---

## 5. 一個必須說清楚的更正：現在的架構本來就已經跨池不變了

設計稿 §1 的論證是「累加順序＝slot 順序＝pool 佈局的函數」，並引 `CGC_ADD_ORDER=rev`（md5 `2d4e5099` → `f6718d44`）作為「必要而非審慎」的證據。**這兩個主張要分開**：

- `rev` 探針證明的是「這個 fp32 鏈**對順序敏感**」——成立，而且它會改變生成文字。
- 它**沒有**證明「順序**隨池子改變**」。而本輪 r53 給了反證：canon **off**、池 4 GiB、對 8 GiB 的 v6 仍是 **M1 9/9**（與之前的 §13.4 一致）。

機制上也對得上：pool path 的位置 `i` 對應的是 router top-k 的**排名** `i`（`ggml_argsort_top_k` 的輸出順序），remap leaf 只是把排名 i 的專家換成它的 slot。**排名順序是 router 的性質，與池子無關** ⇒ 鏈的結合順序本來就與池子無關。

所以工作項 4 的正確定位是：

> 它是**為工作項 2（phase split／整層 gather）準備的前提**，不是修一個當前存在的跨池漂移。
> 一旦 ids 運算元改成由 pool 衍生的「compacted／slot 順序」建構（設計稿設想的形狀），
> 累加順序就會變成 pool 的函數 —— 那時 canonical order 就是必要的。
> 在此之前它的價值是：便宜（host 端 k≤8 的穩定排序 ＋ 一個 GET_ROWS）、預設關、
> 有離線閘門、而且讓「順序」這個自由度在架構改變時不會變成新的不可比來源。

這也修正了設計稿 §0 的「必要」標籤該附的條件。**沒有**推翻 `rev` 探針的數據。

---

## 6. 離開條件改寫（白皮書 §4.3）

### M1「decode 不得退步」

原文只在 roadmap 的表格裡寫「不得退步（roadmap 當時寫 9.3–10.5 t/s）」，而 §1.1 已指出 0.72× 那個比值跨了 build（2 GiB `6.36` vs 8 GiB `8.87` 是當時的 build；今天同類量測是 9.82）。改成可量測形式：

1. **同 build、同池、同 profile、交錯輪替**，每個幾何至少 **3 次**，報**中位數與 IQR**（不是單次值）。
2. 基準是**同一天同一顆 binary 的 8 GiB 臂**（不是 roadmap 的舊數字、也不是三天前的 build）。
3. 判準：候選幾何的中位數 **≥ 0.90 × 8 GiB 中位數**；同時報 **ms/token 絕對值**（t/s 會把長 prompt 的固定成本吸進來）。
4. 只有在 **COLD-STATE**（距上一臂結束 ≥ 1800 s、`notifyutil -g com.apple.system.thermalpressurelevel` = 0）才可引用為**交付數字**；HOT-STATE 的 prefill/decode 只寫進報告，不當交付值。
5. 記憶體形狀的驗收要跑**真實請求**，不能只看「載入完成」（兩個已量到的反例：7.97 GiB 死在載入、6.56 GiB 活過載入但死在第一個請求的 MTP draft 圖配置）。

### M4「accept ≥ 60%」

**作廢**（白皮書 §1.2 已論證：修前 verify 批次的 token ≥ 1 是用別的 token 的專家算的，「accepted」是跟錯的分布比出來的，而且它**抬高**了這個數字：73.81% → 修後 58.25%，同時 decode 8.62 → 12.62 t/s）。改成三條：

1. **MTP-on decode t/s ≥ MTP-off decode t/s**（同 build、同池、同 profile、交錯、≥3 次、中位數）。
   現況（**本輪在 16:00 這顆 binary 上重測**，`mtp_accept_ab.py --arms nail,nail_nomtp --pool-gb 8 --n-predict 96`，每臂 3 請求）：

   | 臂 | accept | acc/gen | prefill t/s | decode t/s | head fingerprint |
   |---|---|---|---|---|---|
   | `nail`（MTP on） | **58.25%** | 180/309 | 22.49 | **12.57** | `219b27c4f9ee` |
   | `nail_nomtp`（MTP off） | n/a | 0/0 | 19.11 | **9.59** | `219b27c4f9ee` |

   ⇒ **12.57 ≥ 9.59（+31%）✓**。與 §14.2 在 14:12 binary 上量到的 58.25% / 12.62 相同到 accept 的第 4 位有效數字，
   所以那個結論不是某一顆 build 的產物。（誠實標註：每臂 3 請求、單 carrier、單池、熱未控制；
   prefill 欄依 §6-M1-4 不可當交付值。）
2. **M1/M2 = 100% with MTP ON**，對**同一個 regime 的參考**比。
   現況：**本輪已核** —— r46（canon off）與 r52（canon=1）都是 MTP ON，都 9/9（§2）。
3. accept rate **照報但不當閘門**，並必須附「這是修後語意」的標註；任何修前記錄的 accept（含 roadmap 的 19.9% 與 09-14 的 70.4%）屬於被抬高的家族，不可作為 before/after 的一側。

### canonical 這個 regime 本身的紀律（新增）

- 任何含 `CGC_CANON_ORDER≠0` 的 run **不得**引用為品質或 D5 證據，去對另一個 mode 產生的參考比。
- canonical 的乾淨對照是 **mode 2**（同圖、只有 perm 值不同），不是 canon off。
- 跨池比較在 canonical 下要用 **canon=1 @ 8 GiB 的參考**（本輪：`ref_iq3_pool8gb_canon1_20260917.jsonl`）；用 v6 比會得到「0/9 不相容」這種無資訊的紅。
- 池掃描的池欄位要以 launch log 的 `[budget] expert pool N MiB` 為準（§3-④）。

---

## 7. 還沒做的（誠實邊界）

- **M1 工作項 2（phase split：`T_prefill`／`PREFILL_GRAPH`／`DECODE_GRAPH`）未實作**（`grep` 0 筆）。這是 M1 的關鍵路徑，也是「prefill chunk 2048 被接受」那條離開條件的唯一路徑。需要長窗口（動 loader 與建圖，且要重跑全部幾何）。
- **M1 工作項 1（`CGC_POOL_SPLIT`）** 仍是 EXPERIMENTAL / NOT USABLE（Blocker A：寬 tensor 讓 gather 把 Metal buffer 指標重指到 Metal 不知道的 host 指標 ⇒ 靜默 `tensor buffer is nil`）。本文沒有動它。
- **M1 工作項 5（計數器：resident bytes vs capacity、per-layer slots、每請求 phase 次數）** 未做；本輪新增的 `CGC-CANON` 行只夠診斷 perm 本身。
- **M4 的 `plain_match`（batch verify vs 逐 token 解碼的逐值比對）未做**。它要回答「oracle 綠了，verify 路徑綠了嗎」。
- **M5 未開始**（`PREFETCH_ONLY` 在此 repo 仍 0 筆；前置結論說期望值低：placement 天花板只 0.2pp，且 start_layer=7 讓它對 churn 最重的 layer 1/2 沒有 head）。
- 本輪**沒有**跑 48 題、沒有測 prefill/decode 速度、沒有測 6 GiB 與 10 GiB 池、沒有測 `edge0head`/`graft` carrier。
- 改動**未 commit**。工作區另有他線的未提交檔（`ggml-metal-ops.cpp`、`libggml-metal`、`arms.json`、`PREFILL250_DECODE_MILESTONE html`），未觸碰。

---

## 8. 產物與重跑指令

```sh
# 離線測試 A（秒級，不需要機器空窗）
clang++ -std=c++17 -O2 -o /tmp/canon_order_selftest scripts/check/canon_order_selftest.cpp && /tmp/canon_order_selftest

# 預設路徑回歸（canon off）
python3 scripts/check/m123_oracle_gate.py --tag rN_canon_off

# identity 對照（同圖、無重排）
python3 scripts/check/m123_oracle_gate.py --tag rN_identity --env CGC_CANON_ORDER=2 --allow-incomparable

# canonical：先立參考，再掃池（池必須 export，見 §3-④）
python3 scripts/check/m123_oracle_gate.py --tag rN_canon1_ref8 --env CGC_CANON_ORDER=1 \
    --write-ref Backup/knifeedge_matrix/ref_iq3_pool8gb_canon1_20260917.jsonl --ref-note "..."
CGC_SERVER_EXPERT_CACHE_BYTES=4294967296 python3 scripts/check/m123_oracle_gate.py --tag rN_canon1_p4 \
    --env CGC_CANON_ORDER=1 --ref Backup/knifeedge_matrix/ref_iq3_pool8gb_canon1_20260917.jsonl --allow-incomparable
```

| 產物 | 路徑 |
|---|---|
| 離線測試 | `scripts/check/canon_order_selftest.cpp`（11 條斷言，含 3 條會失敗的對照） |
| canonical 8 GiB 參考 | `Backup/knifeedge_matrix/ref_iq3_pool8gb_canon1_20260917.jsonl`（＋`.cap`） |
| identity 8 GiB dump | `Backup/knifeedge_matrix/ref_canon_identity_8gb_20260917.jsonl`（＋`.cap`） |
| 本輪 gate 摘要 | `Backup/m123_oracle_gate/summary_r4{6,8,9}_*.json`、`summary_r5{0,1,2,3}_*.json` |
| M4 準則①的當前 binary A/B | `Backup/cgc_logs/mtp_accept_ab_r54_canon_binary.json` |
| 啟動紀錄（含 pool 欄位） | `Backup/m123_oracle_gate/launch_r4{6,8}_*.log`、`launch_r5{0,1,2,3}_*.log` |

⚠ **`Backup/` 是 gitignored**：上列參考與摘要不會隨 commit 走（v5→v6 那次就必須 `git add -f` 才能讓閘門在 clone 後仍可重跑）。
要把 canonical 這個 regime 交給下一條線，`ref_iq3_pool8gb_canon1_20260917.jsonl`（＋`.cap`）必須**一起 force-add**，
否則 `--ref` 會指到一個不存在的檔案，而閘門只會說「找不到參考」，不會說「參考沒進版控」。

**測試 A 的原文輸出**（`/tmp/canon_order_selftest`，2000 次單 token ＋ 1997 個 token row）：

```
ok    perm is a bijection of positions (no id lost, none duplicated)
ok    canonical ids are non-decreasing by expert id
ok    equal expert ids keep their original relative order (order is unique)
ok    mode 2 writes the identity, so the control cannot silently reorder
ok    CONTROL: without canonicalisation the chain sum really does depend on the input order
ok    CONTROL: the identity mode keeps that dependence, so mode 2 is not accidentally canonical
ok    TEST A: with canonical order the chain sum is bit-identical across input orders
ok    multi-token: canonical id block is identical across two input orders
ok    multi-token: each token's perm is its own bijection (no leak across rows)
ok    CONTROL: multi-token raw chain sums do depend on the input order
ok    TEST A: multi-token canonical chain sums are bit-identical
  order-sensitive trials: raw=1185/2000  identity=1185/2000  canonical=0/2000
PASS
```

三條 CONTROL 是這份測試可信度的來源：`raw=1185/2000` 說「在這台機器上、這個 k=8、這些權重分布下，fp32 鏈**真的**會因輸入順序不同而不同」——所以 `canonical=0/2000` 不是因為測試碰巧無感。
