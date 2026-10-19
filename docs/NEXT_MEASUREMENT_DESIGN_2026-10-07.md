# 下一步量測設計：先修量測本身，再談引擎（2026-10-07 16:0x）

前置閱讀：`EXACT_FILL_VERDICT_2026-10-07.md`（今日）、`FILL_COST_MEASURED_2026-09-25.md`（09-25）。

## 0. 結論先講

**我 15:5x 設計的 `--fillprice`（12 臂 / 每 n_max 3 臂）設計錯了，不該照那樣跑。**
理由不是統計，是**量測對象本身錯**：MTP 臂的 `CGC-EBTIMER` 窗口不是「一步」，
所以那個模式只會產出 12 份不可解讀的數字（約 32 分鐘）。

★ **正確的第一步不是加臂，是修 `CGC-EBTIMER` 的 flush 條件。這是一次 rebuild。**

## 1. 診斷：`n_sum/calls` 洩漏了窗口不是一步

`CGC-EBTIMER` 的 flush 條件是 `calls % nl == 0`（`llama-expert-cache.cpp:1229`），
其中 `nl = slot_owner.size()` = **層數**，`calls` 是**累積的 `ensure_batch` 呼叫次數**。
⇒ **它數的是「呼叫」，不是「推理步」**。這在純 decode 下兩者相等，MTP 下不相等。

實測（`Backup/spec_tree/rs0_bitident/`）：

| 臂 | `calls` p50 | `n_sum` p50 | **`n_sum/calls`** | decode 每層應為 |
|---|---:|---:|---:|---:|
| MTP-off | 40 | 320 | **8.00** | 8（top_k）✅ |
| MTP-on | 41 | 63 | **1.54** | 8 ❌ |

★ 每層 1.54 個 expert **在 decode 裡不可能**（`top_k=8`）。
⇒ MTP 臂的窗口裡混進了 draft context 的 `ensure_batch` 呼叫（`llama-context.cpp:8780` 附近
的 draft 路徑會呼叫同一個函式），所以 41 次「呼叫」**遠少於** 41 次「層次齊備的一步」。

### 1a. 這個 bug 怎麼誤導人（這才是要修的理由）
1. **`n_sum` 被誤讀成「這一步請求了幾個 expert」**。實際它是「這 41 次呼叫裡合計請求幾個」。
   兩者差 5 倍（63 vs 320），⇒ ★ **我 15:5x 寫的「每層只請求 63 個 expert 卻要補 55 個，命中率≈0」是錯的**，
   那是視窗邊界的artifact，不是引擎行為。
2. **`miss` 與 `step_usec` 對不上** ⇒ 這正是 MTP-on 組內 R² 只有 0.104 的原因。
   ★ **所以那個 R² 不是「池冷與填補成本無關」的證據，而是「兩個數量屬於不同單位」的證據。**
   ★ 我 15:5x 把 R²=0.104 讀成「MTP 臂內每個 miss 不值錢」—— **這個解讀作廢。**
3. 09-25 的 0.71 ms 是純 decode（窗口正確）下的比值 ⇒ **那個數字不受影響，仍可引用**。

### 1b. 為什麼 09-25 沒踩到
09-25 量的是 `--prompt 0`（冷池、純 decode）⇒ `n_sum/calls = 8` 本來就對。
★ **這個 bug 只在 MTP-on 下暴露，而 09-25 那份文件沒有 MTP-on 的 EBTIMER 對照。**

## 2. 修法（需要 rebuild，一次）

**選項 A（最小改動、推薦）**：讓 flush 條件帶上「這一步」的語意，而不是純計數。
可行且便宜的實作：把 `calls % nl == 0` 改成**同時要求 `n_sum` 已達一個完整步的量**是不靠得住的
（n_sum 會隨 top_k 與 draft 數變）。⇒ **真正的修法是讓引擎知道一個 step 的邊界**：
在 `llama-context.cpp` 的 per-layer 呼叫點，於**每個 segment 的最後一層**打一個 flush 標記，
由 `ensure_batch` 累積器在看到標記時印一行。

**選項 B（零 rebuild、先取得方向）**：不改引擎，改**讀法**——
用 `n_sum` 當作窗口大小的正規化因子，即以 `step_usec / (n_sum/8)` 對 `miss/8` 定價。
⇒ ★ **這是估的，不是量的**：它假設「每次 ensure_batch 呼叫都服務 8 個 expert」，
  而 MTP 臂的 draft 呼叫不是這樣。**只能當上界/方向，不可引用。**

**選項 C（本回合不做）**：既然 09-25 §8 已判 fill 在關鍵路徑上、單 miss 0.71 ms 已可引用，
⇒ ★ **這一格對「引擎該不該改」的決策並不關鍵**。真正關鍵的是別的（見 §4）。

## 3. 我該做什麼（`--fillprice` 現況）

**已寫但建議不要跑**（`scripts/check/pool10g_probe.sh`，12 臂）。若要保留它的價值，
必須先做 §2 的 A 或 B，否則它只會產出 12 份視窗錯誤的數字。

| 項目 | 狀態 |
|---|---|
| `--fillprice` 臂陣列（每 n_max 3 臂） | 已寫，**跑前必須先修 flush**，否則不可解讀 |
| `CGC_EB_TIMER=1` 轉發 | ✅ 已驗證 `run_server.sh:1836-1837` 會轉發 |
| 預先寫死判讀 `P_priced/P_weak/P_none/P_dirty` | 已寫，但門檻語意要配合新 flush 重訂 |
| `fill_price.py` | ✅ 可用，但**必須加一道「窗口合理性」閘**（見 §5） |

## 4. ★ 真正該量的下一步（優先於上面全部）

**MTP-on 臂每層 1.54 vs 8 個 expert 這個落差本身**才是該量的東西，
而且它**不需要 EBTIMER**：15:2x 的 `CGC-SYNCFILL` 與 `CGC-COLD-GUARD` 已經量到同一件事的另一面
（il=0 cold 率 38–45%、`cold_filled` 1.87–3.35/次）。兩者擺在一起會給出一個可判的比值。

**建議的下一趟（不需 rebuild）**：
1. `--hotcache` 或 `(default)` 形狀，MTP off / on 各 1 臂，**只開 `CGC_EB_TIMER=1`**，
   目的是取得 **兩臂同形狀** 的 `n_sum/calls`。⚠ `rs0_bitident` 是 **d512**，與交付 d0 不可互比 ⇒ 這一趟是補形狀對齊。
2. 讀 `n_sum/calls` 在 MTP-on 是否仍遠離 8。
   - 若是 ⇒ **確認「窗口混 draft」假設**，然後才值得做 §2 的 A。
   - 若接近 8 ⇒ 那 1.54 是**引擎真實行為**（draft context 每層只請求 1.5 個 expert），
     ★ **那本身就是一個可引用的發現**，並且直接指向 `LAYER_CAPS`／draft context 的路由問題。

⇒ ★ **這一趟的預期讀數與判別句已寫死**：`n_sum/calls`(MTP-on) = **1.5 或 8**。
  前者 ⇒ 量測視窗問題；後者 ⇒ 引擎行為問題。**兩個結果都有用，這是它比 `--fillprice` 好的地方。**

## 5. `fill_price.py` 要加的一道閘（無論走 A/B/C 都該加）

★ **窗口合理性檢查**：任何 stratum 的 `n_sum/calls` 若不接近整數倍的 `top_k`，
該 stratum **必須**報 `WINDOW_UNSOUND` 並拒絕給斜率。
理由：斜率的 x 軸（miss 數）與 y 軸（step_usec）必須屬於同一個單位；
`n_sum/calls` 是唯一能從**既有產物**檢查這件事的欄位（`calls` 印的是常數 `nl`，不是觀測值）。

⚠ 這道閘會讓 15:5x 那次 `FILL_PRICED` 判定**回溯失效** ⇒ 那個 0.902 ms/miss
（R²=0.519，stratum calls=40）**仍然有效**，因為 off 臂 `n_sum/calls = 8.00` 通過檢查。
⇒ **off 臂的定價不受影響，MTP 臂的 0.566/R²=0.104 作廢。**

## 6. 誠實邊界
- 本輪**未跑任何量測**、**未 rebuild**。§1 的 `n_sum/calls` 是從既有 `rs0_bitident` 產物算出的。
- ⚠ `rs0_bitident` 是 **d512** 形狀 ⇒ 1.54 這個數**不能引用為交付形狀的行為**，只能引用為
  「窗口混合」這個**方法論問題**的存在證據。§4 那趟就是為了補這個。
- ★ 我 15:5x 的「命中率≈0」與「R²=0.104 ⇒ MTP 臂內不成立」兩條**在本輪作廢**（見 §1a）。
- ⚠ `--fillprice` 的臂數（每 n_max 3 臂）**設計是對的**（識別需要），但它解決的是 §1 的問題的**症狀**而非成因。
