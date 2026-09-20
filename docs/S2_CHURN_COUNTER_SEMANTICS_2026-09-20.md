# S2 的阻塞量測被誤讀了：那個 42.3% 不是「42% 的映射變了」

日期：2026-09-20 12:0x（+08）　作者：**`線A (ace)`**　零 GPU（純讀碼 ＋ 既有 log）
對象：要引用「被消費映射 churn」的人（含 `targets.json` 的 `G1.premise_b` 與 `G1.candidate_family`）、
以及要依那條結論決定 **S2 / S3** 的人。

---

## 0. 一句話

**42.3% 是「有多少次發布裡至少有一個被消費的 id 換了值」，不是「有多少個被消費的映射換了值」。**
分子是**事件**（publish-level）、分母也是**事件**。而 `llama-context.cpp:4927-4937` **已經把 entry 級的
`consumed_moved` 算出來了**，卻只留下布林值、把數字丟掉 ⇒ **「1/32 動了」與「32/32 動了」在現行儀器上
完全同形。** S2/S3 的裁決需要的正是後者。

---

## 1. 兩個儀器量的是**不同的窗口**，而兩個都對

| 儀器 | 比較 | 窗口 | 09-17 那份 log 的讀數 |
|---|---|---|---|
| **`SEL-DRIFT`**（`TABLE-CHURN` 行） | `cgc_churn_last[layer]`（**已發布**的表）對 `slot_table_safe(cache, layer, e)`（**live pool**，非駐留走 ZERO 槽） | 發布 → 步末 | **`layers=0 entries=0 max_per_layer=0 argmax_il=-1`**（10 個圖全部 0） |
| **`consumed_changed`**（teardown 的 by-ntok） | `prev[e]`（**上一次發布**的表）對 `now[e]`（**這一次發布**的表），只走**這一步被消費的** ids | 上一次發布 → 這一次發布 | 42.3%（ntok=4）／89.0%（ntok=8）／0.0%（ntok=1） |

**⇒ 兩者不矛盾**：一個說「發布之後到步末之間沒動」，另一個說「跨步之間動過」。我第一次看到時
以為是硬矛盾，讀了比較式才發現窗口不同 —— 這一條本身就是「不要憑欄位名推語義」的實例。

同一個 `TABLE-CHURN` 行裡的 `changed_entries=38..104` 是**全表**漂移（`chg`，把 39×256 逐項比），
而程式自己的註解（`:4852-4858`）已經寫明那是 **clamp 造成的**（「約 44.2% ＝ 恰好是非駐留比例」），
所以它**不代表消費者的順序**。

## 2. ★ 為什麼 `consumed_changed` 不能回答 S3/S2 的問題

`llama-context.cpp:4926-4948` 逐字（摘要）：

```cpp
long long consumed_moved = 0, consumed_total = 0;
for (int64_t j = 0; j < n_tokens * n_expert_used; ++j) {
    const int32_t e = ids[j];
    ...
    consumed_total++;
    if (prev[e] != now[e]) { consumed_moved++; }      // ← entry 級，算出來了
}
if (consumed_total > 0) {
    if (consumed_moved > 0) {
        cache->n_slot_table_consumed_changed++;                 // ← 只留布林
        cache->n_slot_table_consumed_changed_by_ntok[n_tokens]++;
    } else { ... }
}
```

- **分子**：`consumed_changed` 每次發布最多 +1 ⇒ 它是「**至少一個**被消費 id 換了值」的**事件計數**。
- **分母**：by-ntok 的分母 612／8160／4159 **就是各 ntok 的發布次數**（合計 12931，
  與同一次運行記的「over 12971 publishes」相符）⇒ `3449/8160 = 42.3%` 讀作
  **「42.3% 的 ntok=4 發布裡，32 個被消費 id 中至少 1 個換了值」**。
- ⇒ **「1/32 動了」與「32/32 動了」都是 +1。** 而程式自己的註解（`:4924-4925`）寫的判準是
  「*the question is whether it is 0/N or N/N*」—— 那個問題**這個計數器答不了**。

**所以 `G1.premise_b` 的那句「the consumed mapping moves on 42% of the (step, layer) pairs」
是把事件率讀成了條目率。** S3 的生死（「發布冗餘嗎」）要的是**條目級的比例**。

## 3. 這對 S3／S2 的結論造成什麼、不造成什麼

**不動的部分（仍是對的）**：`compulsory=791 capacity=0`（100% 首觸、0% 容量）＋ `worst=layer 40
distinct=113 slots=256` ⇒ **池子不是容量受限的**（沒有任何 miss 是「被驅逐後又要」）。而 42.3% > 0
的方向仍然成立 ⇒ **表的內容不是常數** ⇒ 「發布一次就好」不成立。

**動的部分（不能引用的）**：
- **量級**：42.3% 不能讀成「42% 的映射變了」。真正的條目率是 **(0, 42.3%] 裡的某個值**，
  而全表每圖只動 38–104／9984 ≈ **0.4–1%** ⇒ 條目率**很可能遠小於 42.3%**（但這是推論，不是讀數）。
- **推論鏈的強度**：`premise_b` 用 42.3% 去否證 S3 的前提。方向可能仍對，但**它引的數字
  不支持它引的量級**；要恢復力度需要條目級的讀數。
- **對 S2**：S2 的阻塞是「mapping 每一步都得重算嗎」。事件率答不了它 —— **S2 的可行性論證
  目前缺的是同一個條目率**。

## 4. 要補的量測（極小、add-only、不需要改 dispatch）

`llama-context.cpp:4927-4937` 已經有 `consumed_moved`／`consumed_total` 兩個區域變數。
把它們累加成兩個計數器（＋ teardown 一行印 rate）即可，**四行左右**：

- `cache->n_slot_table_consumed_moved_entries += consumed_moved;`
- `cache->n_slot_table_consumed_total_entries += consumed_total;`
- teardown：印 `moved_entries/total_entries` 的比，**並按 ntok 分桶**（沿用既有的 by-ntok 陣列形狀）。

判準（寫在跑之前）：**若條目率 ≈ 0**，S3 的前提回到桌上（S2 也不是唯一生路）；**若條目率與 42.3%
同量級**，則現在的結論不變、而且量級第一次有了出處。這是「0/N 還是 N/N」那個問題唯一的直接答案。

⚠️ **本輪沒有做這個改動**，因為**窗口是別人的**：12:02 的那次啟動被 `run_server.sh` 的 preflight
自己擋下（逐字見 §5），而**建置會蓋掉別人正在 map 的 dylib**（09-20 01:04 的先例）。
⇒ 改動與量測排在窗口空出來之後。

## 5. 現場：被 preflight 擋下的那次啟動（原文）

```
[preflight] 發現 1 支 llama 行程（可能是別條 session 正在量測）→ 不送任何訊號
[preflight] 等 GPU/系統記憶體回穩（free=14% < 25%，最多 20s）
[preflight] free=15%
error: 仍有 1 支 llama 行程，繼續啟動極可能 GPU OOM (ret=-3)
  （preflight 預設已不再幫你清場：2026-09-18 它把別條 session 正在量的 server SIGTERM 掉了）
```

當下 `pgrep -fl llama` 是 `57421 …/build/bin/llama-server -m …/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`
—— **同一顆 binary、不是本線的行程**。`decode_sweep.py` 因此 `[FAIL] server never became ready`，
沒有一筆新讀數產生（`Backup/phase_decomp/s2_churn_two_tools.json` 是空殼）。
**這一輪因此零讀數、零建置、零 src 改動** —— 這是刻意的，不是失敗。

## 6. 誠實邊界

- 本檔的「兩個儀器窗口不同」是**讀比較式**得到的（`:4826-4851` 對 `:4926-4938`），**沒有跑任何東西驗它**。
- 「條目率很可能遠小於 42.3%」是**推論**（由全表 0.4–1%/圖 反推），**不是讀數**，不可引用。
- `compulsory/capacity` 那組數來自 `Backup/cgc_logs/llama_server_20260920_071337.log` 的 teardown
  （本線今天那次 S1 修補後的運行），**不是** 42.3% 那次運行的 —— 兩次不同 build／profile，
  但都是「池子非容量受限」的同向證據；要嚴格同輪，得在補完條目率之後一次跑同時取兩者。
- 本檔**不改**任何既有報告：`G1.premise_b` 等在 `targets.json`（**別條 session 未提交的檔**），
  所以那句要由它的 owner 改成「事件率」的措辭。本線沒有動那個檔。
