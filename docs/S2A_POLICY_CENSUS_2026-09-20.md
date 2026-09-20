# S2-A：`pick_slot` 的策略普查 —— **EMA 是負載相關的；三個 pin 類從未被行使**

2026-09-20 13:05，線A (ace)。載體 `decode_sweep.py --profile prod25 --arms p25-s1-churn`
（與 `§EN-317` 的條目率量測**同一個臂**）。事前判準：`Backup/phase_decomp/S2A_POLICY_CENSUS_20260920.conditions.md`。

---

## 0. 一句話

**S2 的裝置側必須帶走 `spac_util` 這條 FP EMA（57.42% 的受害者由它決定），
但可以把三個 pin 類與 decode-reserved defer 規則整組丟掉（engagement 全為 0）。**

---

## 1. 讀數（原文）

```
llama_expert_cache: S2-PROBE free=112 evict=14004 lru_cand=14004 lru_mismatch=8041 (57.42% of victims)
                   spac_lru_fallback=0 spac_nonfinite=0 pin_yield=0 defer_skip=0 defer_yield=0
```

| 欄位 | 值 | 讀法 |
|---|---:|---|
| `free` | 112 | 拿到**空槽**的次數 ⇒ 免策略的快路徑只佔 **0.8%**（112/14116） |
| `evict` | 14004 | 需要**挑受害者**的次數 ⇒ **策略被行使了 99.2%** |
| `lru_cand` | 14004 | 每次都有一個合法的純 LRU 候選 ⇒ **比較沒有被審查掉**（no censoring） |
| **`lru_mismatch`** | **8041** | **57.42%**：EMA 選出的受害者與純 LRU **不同** |
| `spac_lru_fallback` | 0 | EMA 分支每次都選得出東西 ⇒ **EMA 是權威，不是備援** |
| `spac_nonfinite` | 0 | EMA 健康（無非有限值） |
| `pin_yield` | **0** | **三類 pin 的讓位從未發生** |
| `defer_skip` | **0** | **prefill-defer 規則從未改變任何受害者選擇** |
| `defer_yield` | **0** | 同上（overflow 那一格） |

---

## 2. 判準命中（事前寫死的第三格）

| 事前判準 | 判決 |
|---|---|
| `mismatch == 0` ⇒ EMA 從未決定 ⇒ 裝置只需 `owner[]+last_use[]+batch_mask` | ❌ |
| `0 < mismatch << victims` ⇒ 量化份額 | ❌ |
| **`mismatch ≈ victims` ⇒ EMA 就是 placement ⇒ 裝置必須帶 EMA 與 `spac_update`** | ✅ **命中（57.42%）** |

---

## 3. 所以 S2 的裝置狀態清單（這是本節真正的產出）

`pick_slot` 需要的七項（見 `docs/S2_IMPLEMENTATION_PLAN_2026-09-20.md` §4 的「決定 residency 的介面」）：

| 狀態 | 要不要搬到裝置 | 依據 |
|---|---|---|
| `slot_owner[]` | **要** | 決定「這個槽是誰的」 |
| `slot_last_use[]`（uint64 tick） | **要** | 純 LRU 的依據；且是 EMA tie-break |
| **`spac_util[layer][expert]`（FP EMA）** | **要** | **57.42% 的受害者由它決定** |
| `batch_mask` | **要** | 這一批已擁有的槽不可選 |
| `slot_pinned` / `slot_pinned_static` / `slot_decode_reserved` | **可以丟** | 三者 engagement **全為 0**（`pin_yield=0`、`defer_skip=0`、`defer_yield=0`） |
| `slot_loading` / `slot_queued` | **要**（但在飛的 fill 是 host 的行為 ⇒ 介面問題） | 不可選 |
| `tick` | **要**（一個純量） | LRU 的單調來源 |

⇒ **裝置側的常駐狀態＝ 3 個陣列（owner / last_use / spac_util）＋ 1 張 bitmap（batch_mask）＋ 1 個純量（tick）。**
**難點不是資料量，是 `spac_util` 的算術**：它由 `spac_update` 以 `u[e] += bump` 的 EMA 維護，
要被逐位元重現，否則 placement 會分歧（而 57.42% 的受害者由它決定）。

---

## 4. 誠實邊界

- **單一 regime**：`prod25`、pool 8 GiB、MTP on、HTTP door。**pin 類 engagement 為 0 是這個配置的性質**，
  不是普遍事實 —— 換 pool 大小或開啟 routing-aware placement 會變（`n_pin_marked` 有印，本輪未開）。
- **失配 ≠ 觀測到的分歧**：受害者不同 ⇒ 表不同，但**被消費的條目**是否不同，取決於那位專家後來有沒有被要。
  長 run 的 **79.9% capacity** 強烈暗示會被要，但這需要第二支探針（在**被消費集合**上比對）才能定量。
  **本節只主張「EMA 決定了 victim」，不主張「表在數值上分歧」。**
- `free=112` 幾乎全發生在**預熱期**（池子還沒滿），所以那 0.8% 不是穩態的代表。
- 探針 **add-only（+44 / −0）**、搭 `CGC_S1_TABLE_CHURN` 門控、不改 dispatch／hook 順序／任何數值路徑。
