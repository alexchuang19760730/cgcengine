# S2 的「駐留」那一半：4 層在抖動，其餘是 3.5 倍的過量配置

**日期**：2026-09-20 · **狀態**：駐留那一半以量測結案；drain 那一半有結構性結論 ·
**引擎改動**：本輪為**零**（配置路線已驗證但**不建議**設成預設，理由見 §7）

> 這一份取代 `docs/S2_IMPLEMENTATION_PLAN_2026-09-20.md` §4 的第 1 項問號：
> 「誰在裝置上決定 expert 放哪個槽」的**前半**（池子實配得對不對）現在有讀數了。
> 後半（那個決定要不要搬上裝置）見 §9。

---

## 1. 計畫書 §4 的三項，逐項現在的狀態

| §4 的項 | 狀態 |
|---|---|
| **① 決定 residency 的介面** | **本文件結案**：現行池子把 4 層餓著、把 36 層餵到 3.5 倍；配置路線已驗證可修（§6），但代價不划算（§7–§8） |
| **② 6.16% 的 first-touch 規則** | **不需要設計了**：first-touch 佔本 regime miss 的 **74.0%**，而它是路由器在探索 ⇒ **任何裝置側規則都不能讓「第一次被選中的專家」變成常駐**（§8.2） |
| **③ 把 `submit_seg(i+1)` 提前、拿掉 wait** | **第 §9 節從原始碼證明它不可能只靠主機側完成**，並給出它真正需要的東西與那個家族的天花板 |

---

## 2. 第一次被打開的儀器

```cpp
// llama-expert-cache.cpp:2699-2701
if (getenv("LLAMA_EXPERT_CACHE_MISS_ATTR_LAYERS") != nullptr) {
    for (size_t li = 0; li < n_distinct_demanded.size(); ++li) {
        fprintf(stderr, "llama_expert_cache:   miss-attr layer=%zu distinct=%u slots=%zu\n", ...);
```

它一直在樹上、也一直在 `run_server.sh` 的轉發白名單裡（`:1562-1563`），但：

```
$ grep -rl 'miss-attr layer=' Backup/cgc_logs/*.log | wc -l
0
```

**⇒ 這台機器上從來沒有人印過逐層的駐留戶籍。** 之前所有關於「池子夠不夠」的討論，用的都是**單一格的
`worst=layer N distinct=D slots=S`**（只報最壞的一層），或是把短 run 的 teardown 拿去對另一個 run 的步數
（= `§EN-314` 那個已作廢的「池子根本不满」）。

---

## 3. 讀數

**命令**（交付 profile。MISS_ATTR_LAYERS 會從熱路徑寫 stderr ⇒ **t/s 不可引用**，本節也不用它）：

```sh
export LLAMA_EXPERT_CACHE_MISS_ATTR_LAYERS=1
RUN_REPLAY_BENCH=0 python3 scripts/check/decode_sweep.py --profile prefill250 \
  --arms p25-mtp-on --rounds 1 --n-predict 512 \
  --json Backup/phase_decomp/s2_geom.json --force
# log: Backup/cgc_logs/llama_server_20260920_140638.log
```

**彙總行**（原文）：

```
miss attribution: compulsory=3014 capacity=1061 (74.0% / 26.0% of 4075)  evictions=3787
                  layers_distinct_over_slots=4  worst=layer 2 distinct=196 slots=143
final stats: runtime requests=13657 hits=9582 misses=4075 (hit rate 70.2%)  resident=6430.62 MiB
```

| il | distinct | slots | | il | distinct | slots | | il | distinct | slots |
|---|---|---|---|---|---|---|---|---|---|---|
| **0** | **193** | 143 | | 14 | 45 | 143 | | 28 | 60 | 143 |
| **1** | **184** | 143 | | 15 | 53 | 143 | | 29 | 49 | 143 |
| **2** | **196** | 143 | | 16 | 50 | 143 | | 30 | 62 | 143 |
| **3** | **154** | 143 | | 17 | 52 | 143 | | 31 | 57 | 143 |
| 4 | 123 | 143 | | 18 | 51 | 143 | | 32 | 60 | 143 |
| 5 | 90 | 143 | | 19 | 58 | 143 | | 33 | 66 | 143 |
| 6 | 71 | 143 | | 20 | **41** | 143 | | 34 | 49 | 143 |
| 7 | 69 | 143 | | 21 | 61 | 143 | | 35 | 70 | 143 |
| 8 | 50 | 143 | | 22 | 66 | 143 | | 36 | 56 | 143 |
| 9 | 67 | 143 | | 23 | 66 | 143 | | 37 | 50 | 143 |
| 10 | 73 | 143 | | 24 | 52 | 143 | | 38 | 60 | 143 |
| 11 | 72 | 143 | | 25 | 61 | 143 | | 39 | 47 | 143 |
| 12 | 48 | 143 | | 26 | 51 | 143 | | 40 | 113 | **256** |
| 13 | 61 | 143 | | 27 | 57 | 143 | | | | |

**兩個極端在同一張表上**：`il=20` 要 **41** 個槽、拿到 **143**（3.5 倍過量）；`il=2` 要 **196** 個、拿到
**143**（缺 53）。而 `layers_distinct_over_slots=4` ⇒ **只有 4 層是餓的**。

---

## 4. 原始碼自己寫的判準（照抄）

```cpp
// llama-expert-cache.cpp:2680-2684
// If a layer's distinct demanded experts fit inside its slot count, no capacity miss is possible
// there once warm -> only locality / routing can help and growing the pool is pure RSS cost. If
// distinct >> slots, the LRU is thrashing and capacity IS the lever.
```

⇒ 套上 §3 的表：**`il 0/1/2/3` 是 `capacity IS the lever` 那一格；`il 4..39` 是
`growing the pool is pure RSS cost` 那一格。** 一句話：**這條槓桿只作用在 4 層上。**

---

## 5. 幾何：143 是怎麼來的（以及為什麼它與需求無關）

```
slots = clamp(pool_bytes / (n_layers * per_slot), 8, 256)          gguf_pool_geometry.py
iq3: 41 層、per_slot = 1,458,176 B（binding blk.39）  → 8 GiB ⇒ 143
```

⇒ **槽數是對 41 層套同一個數字的**，而各層的需求差 **4.8 倍**（il=20 要 41、il=2 要 196）。
`40-40:256` 這個預設（MTP 分支才有，`run_server.sh:2235`）是唯一一個按層給的例外。
**⇒ 「池子夠不夠」這個問題在逐層資料出現之前是提不出來的：143 這個數字與任何一層的需求都沒有關係。**

---

## 6. 配置路線：把 4 層餵飽

`LLAMA_EXPERT_CACHE_LAYER_CAPS` 的語法是 `start-end:cap;...`（`llama-expert-cache.h:578`），
兩側（loader 的 `ne[2]` 縮放與 cache 的槽向量）都讀它。經 `run_server.sh` 的
`CGC_SERVER_LAYER_CAPS`（`:2233`）轉發。

```
CGC_SERVER_LAYER_CAPS="0-3:256;40-40:256"
```

（必須把 `40-40:256` 一起寫 —— `:2233-2235` 是 `if/elif`：一旦明確給了 `SERVER_LAYER_CAPS`，
MTP 分支的那個預設就**不再**被套用。）

實測生效（`Backup/cgc_logs/llama_server_20260920_140934.log`）：

```
llama_expert_cache: LAYER_CAPS per-layer caps: total 6428 slots (avg 156.8/layer, min 143/layer)
```

`6428 − 5976 = 452 = 4 × 113` ✓ 精確等於把四層從 143 提到 256。池子 ≈ 6.80 GiB（+0.48 GiB）。

**數值驗證**（`m123_oracle_gate.py --tag s2-caps` 與 `s2-caps2`，兩次）：

```
REFERENCE IS NOT COMPARABLE (1 numerics-determining difference(s))
  ENV.LLAMA_EXPERT_CACHE_LAYER_CAPS: ref='40-40:256'  now='0-3:256;40-40:256'
```

⇒ `LLAMA_EXPERT_CACHE_LAYER_CAPS` **是被追蹤的鍵**，所以任何 caps 改動都必然是
`INVALID COMPARISON`（這是預期，不是缺陷）。**數值判決要看 M1/M2/M3**：
**兩次閘門（同一天、同一棵樹）：**

- **對照臂（不含 caps）** —— `s2-caps-ctl`：`comparable=True`、`config_diffs=[]`、
  **M1=9/9 M2=9/9 M3=9/9**（n=9），build `libllama.0.0.279.dylib=95030f0177e4574e`、
  tree `a2eda68d6 dirty_tracked=13` ⇒ **PASS**。
  ⚠ **這一步是必要的，不是保險**：樹上的 `libllama` 在 13:57 被**另一條線重建過**
  （他們的 `LLAMA_EXPERT_CACHE_DEMAND_DUMP`，未提交）⇒ 沒有這個對照臂，「與參考不符」
  就無法歸因給 caps 還是歸因給那顆新 libllama。
- **處理臂（含 caps）** —— `s2-caps-run`：**沒有判決**。server 死在 `Insufficient Memory`（§7），
  探針回空 ⇒ gate 判 **`INVALID RUN`**，而它**拒絕**在空 dump 上報 M1/M2/M3 ——
  那正是這個閘門存在的理由（原文：`An empty answer with a healthy-looking dump … M1/M2/M3
  would happily report 1.0 on such a dump`）。

⇒ **caps 是否數值等價＝「未被驗證」**：三次帶 caps 的嘗試有 **2 次**死在同一處（§7）。
**本文件因此不對 caps 的 bit-identity 下任何斷言。**
---

## 7. ⚠ 記憶體代價與一次 OOM（這一節是「不要設成預設」的理由）

**三次帶 caps 的嘗試，2 次死在同一處；同一棵樹上不含 caps 的對照臂通過。**

| log（`Backup/cgc_logs/`） | caps | `Insufficient Memory` | 有沒有真的跑起來 |
|---|---|---|---|
| `llama_server_20260920_140934.log`（`s2-caps`） | 有 | **3** | **0** |
| `llama_server_20260920_141049.log`（`s2-caps2`） | 有 | 0 | 1 |
| `llama_server_20260920_142546.log`（`s2-caps-run`） | 有 | **3** | **0** |
| （`s2-caps-ctl`，**不含 caps**） | 無 | 0 | 1 |

```
Backup/cgc_logs/llama_server_20260920_142546.log:144-147
E ggml_metal_synchronize: error: command buffer 8 failed with status 5
E error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
E CGC-METAL-FAIL: command buffer 8 failed with status 5 (compute #82, fail-stop)
  ← ggml_metal_synchronize ← sched_synchronize ← cgc_logits_oracle_dump ← process_ubatch ← decode
```

**兩次死的位置逐字相同**（`command buffer 8`、`compute #82`），時間戳都在 **0.26–0.32 s**，
也就是**第一個 decode** ⇒ 不是隨機的機器狀態。而**不含 caps 的對照臂在同一棵樹上跑通了**
（`s2-caps-ctl` PASS）⇒ **機器狀態不是解釋變數。**

⇒ **歸因：caps 把 4 層的 `ne[2]` 由 143 提到 256，等於多要約 0.48 GiB 的 GPU-wired 專家緩衝，
而這台 16 GB 的機器在第一個 decode 就交出 `kIOGPUCommandBufferCallbackErrorOutOfMemory`。**

**⇒ 所以駐留重分配在這台機器上是「記憶體上不可行」，不只是「不划算」** ——
這比 `§EN-329` 先寫的「不可歸因」（當時只有 1 次嘗試）強。
`llama_server_20260920_141049.log` 那次跑通，是本表唯一不支持的樣本（1/3）。

---

## 8. 算術：駐留這一半最多值多少

### 8.1 交付（暖態）的 `cb` 拆解

F1 的擬合（`per_miss_linear`，r² = 0.862）：`cb_layer = 0.46 ms + 0.695 ms × m`。
交付暖態 `cb` = **21.23 ms/step**、miss-bearing 層數 L = 29 ⇒

```
barrier  = 29 × 0.46  = 13.34 ms
per-miss = 21.23 − 13.34 = 7.89 ms      ⇒ m ≈ 11.4 misses/step
```

### 8.2 能拿掉多少

- **capacity 佔 miss 的 26.0%**（本 run；長 MTP=0 run 是 26–80%）
  ⇒ 拿掉 = `0.26 × 7.89` = **2.05 ms/step** = **139.46 ms 的 1.5%**
- 就算 capacity 在 decode 段佔到 50% ⇒ 3.9 ms = **2.8%**
- **而 first-touch（74.0%）不可能拿掉**：它是路由器第一次選中某個專家 ⇒ 池子再大也得付一次。
  「裝置側的 first-touch 規則」（計畫書 §4 第 2 項）不會讓那次讀取消失，只會換一個形式。

⇒ **駐留這一半的全部價值 ≈ 1.5–2.8%，低於交付 cell 的單臂噪音底（±27%）。**
**⇒ 它的 t/s 效應在單臂前後對比下測不出來**，只能像本文件這樣從機制（`capacity=`、`evictions=`）
去證明。這是為什麼它**不值得用 +0.48 GiB 去換**。

---

## 9. ★★ drain 那一半：為什麼它不可能只靠主機側完成

### 9.1 現狀（S1 之後）

S1（`CGC_SLOT_TABLE_GPU=1`）把 expert→slot 的映射從**主機寫的 leaf** 換成圖裡的
`get_rows(slot_table, selected_experts)`，而 **index vector 由裝置產生**（`selected_experts` 的 CONT，
`llama-graph.cpp:2290-2345`）。所以主機**不再需要**為了 gather 而寫任何東西。

但 dispatch 迴圈（`ggml-backend.cpp:2612-2630`）仍然是：

```
for i in 0..n_segs-1:
    hook_seg(i)          # 先等 segment i 跑完，再跑 top-k hook（host）
    if i+1 < n_segs: submit_seg(i+1)
```

`hook_seg` 的第一件事就是 `while (cgc_done(...) < target) sched_yield()`（`:2119`），
而 `submit_seg(i+1)` 排在 hook 之後 ⇒ **GPU 在整個 hook 期間閒置**。這就是 `gap ≈ cb + 12.5`。

### 9.2 為什麼不能把 submit 提前

`hook_seg(i)` 做三件事，全部依賴 **segment i 的輸出**（也就是那次 wait）：

1. 讀出該層的 **expert ids**（`expert_cache_on_topk` 的輸入）；
2. 用 ids 跑 `ensure_batch`：把不在池裡的專家填進去（**這就是 74%+26% 的 miss 成本**）；
3. 把結果寫進 `slot_table[layer]` —— 而 **segment i+1 恰好從該層的 `mul_mat_id` 那一段開始**：
   `seg_view(s)` 的邊界是 `a = as_idx[s-1]+1`、`b = as_idx[s]`（`ggml-backend.cpp:2057-2060`）
   ⇒ 邊界節點 `as_idx[i]` 是 **segment i 的最後一個**節點，而它的下一個（`as_idx[i]+1`）
   就是 segment i+1 的起點；該層的 `mul_mat_id` 就在其中，透過 `get_rows(slot_table, …)` 讀這張表。

⇒ **第 3 步的產物是 segment i+1 的直接輸入 ⇒ 它的 commit 必須在 hook 之後。**
`CGC_SUBMIT_AHEAD=1`（把那兩行對調）刪掉的正是這個順序，所以它**必然錯**，
而它量到的 **×1.78** 只是這個家族的**天花板**（`decode_sweep.py:637-652`）。

### 9.3 三條可能的繞法，兩條被堵死

| 繞法 | 為什麼不行 |
|---|---|
| **表延後一步發布**（用上一步的 ids） | 「第一次被選中」的專家在延後的表裡沒有槽 ⇒ zero-map ⇒ `G2.zero_mapping_invariant` 直接否決（`G2` 要求 `zero_mapped_selected == 0`，本 run 實測 0） |
| **表雙緩衝 + 世代選擇器** | 世代必須在 **encode 時**就確定，而世代由 hook 決定 ⇒ 同一個雞生蛋 |
| **✓ 讓「這一步不會改表」在主機側可證** | 唯一可行的方向，但條件是 **cap == n_expert（256）且 256 個專家全部常駐** ⇒ 表成為常數 ⇒ **hook 是 no-op，ids 根本不需要**。代價：41 × 256 × ~1.13 MB ≈ **11.9 GB**，16 GB 的機器放不下（模型本身還要 ~10 GB） |

⇒ **結論：只要「本步的映射依賴本步的 ids」成立，這個 drain 就是必要的。**
要拿掉它只有兩條路：
**(i) 把駐留判定搬上裝置**（裝置側表 + 一個「有沒有被選中的專家不在池裡」的裝置側判定
＋ 在那條時間線上排隊的填寫）—— 這就是 S2 的本體，是 **kernel 工程**，不是配置或幾行程式；
**(ii) 打破「映射依賴本步 ids」**（= 接受錯誤），已被 `G2` 否決。

### 9.4 所以 S2 的天花板要改寫

| 版本 | 值 | 性質 |
|---|---|---|
| `p25-submit-ahead` 量到的 | **×1.78** | **故意錯的臂** ⇒ 家族的**上界**，不是可達值 |
| §EN-326 的「m=0 的 31.9% 邊界」 | ×1.087 | 推算 |
| **本節** | **可達值在 (×1.087, ×1.78) 之間，而它要一個裝置側判定** | 這兩端都不是量測 |

⚠ **`×1.087` 與 `×1.78` 都是「如果 drain 能拿掉」的函數，差別只在拿掉多少。** 而 §9.3 說
拿掉**任何**一格的 drain 都需要那條裝置側建構 ⇒ **在它出現之前，這兩個數字都不能當計畫用。**

---

## 10. 誠實邊界

1. **§3 的 distinct 是 512 token 的讀數。** 更長的 run 會更大（另一批長 run 的
   `worst=layer 0/1/2 distinct=242–255` 就是這個方向）⇒ **「4 層」是下界估計**，
   而那些長 run 是 **MTP=0**（沒有 `40-40:256`、且有 draft 路徑），**兩個 regime 不可直接比**。
2. **§8 的 `cb` 拆解用的是 09-19 的 F1 擬合**，不是本 run 重測的。它與本 run 的總量自洽
   （`0.46×29 + 0.695×11.4 = 21.3` vs 記錄的 21.23），但**沒有在同一 run 內同時量過**。
3. **caps 的 bit-identity「未被驗證」**（§6）：處理臂死在 OOM 上，**沒有** M1/M2/M3。
   **對照臂（不含 caps）在同一棵樹上 PASS** ⇒ 基準仍有效，而且失敗方向是**單向**的
   （不含 caps 從未出過這個錯）。`comparable=False` 只是 caps 被追蹤造成的**預期**結果，不是回歸。
4. **本輪跑了交付 profile 的量測（§11），但沒有跑 caps 的 t/s 對比** —— 而且現在不必跑了：
   那 +0.48 GiB 已經被證明會讓第一個 decode OOM（§7）⇒ **不是「測不測得出來」的問題，
   是「跑不跑得起來」的問題。** 原句「+0.48 GiB 換 1.5% 不划算」的判斷仍然成立，只是理由更硬。
5. **§7 的歸因是「2/3 可重現 ＋ 對照臂通過」**，不是「3/3 必然」。`…_141049` 那次帶 caps 跑通，
   所以嚴格的說法是：caps **使**第一個 decode 的 `Insufficient Memory` **很可能**發生，而不是必然。

---

## 11. 最終量測（交付 profile，operator 指定沿用 250/12）

```sh
python3 scripts/check/prod_profile.py --profile prefill250 --reps 3 \
  --json Backup/prod_profile/prod_profile_20260920_1420.json \
  --md   Backup/prod_profile/prod_profile_20260920_1420.md
```

| 軸 | 臂 | t/s | ± | launch / worst | bar | 判決 |
|---|---|---:|---:|---|---:|---|
| `prefill-house` | `prefill250` | **192.52** | 29.40 | NOMINAL / **HEAVY** | 250 | **refused** |
| `decode-delivery` | `prefill250` | **8.30** | 0.97 | NOMINAL / **HEAVY** | 12 | **refused** |
| `decode-delivery-anchor` | `prod25-stream` | **6.75** | — | NOMINAL / **HEAVY** | 12 | **refused** |

**三條全部被儀器自己的熱閘門拒絕**（逐字：`left NOMINAL (launch=NOMINAL, worst=HEAVY) -> not a bar-meeting number`）⇒ **今天沒有可引用的新數字。** 三條都等過降溫才發射（prefill 等 170 s、decode 等 170 s、錨點等 70 s），**而發射之後臂內又升溫了** —— 這與 09-16 那份記錄的判準一致（`Nominal 6/6 全部 ≥250；非 Nominal 0/21，最高 211.65`）。

**兩件必須一起說的事：**

1. **錨點 6.75 vs profile 8.30 ＝ 同一個命令在同一次 session 裡差 ×1.23。** 兩條臂的 argv 逐字相同（只差 `--arms` 的名字，已於 12:57 那次證明過）⇒ 交付 cell 的單臂噪音**再一次落在 ±27% 的量級上**。
2. ⚠ **本次的 `libllama.0.0.279.dylib` 指紋是 `95030f0177e4574e`**，與記錄中的 `cfd550a18638c9ec`／`0cd5a62833e31549` 都不同 —— **另一條線在 13:57 重建過它**（他們的 `LLAMA_EXPERT_CACHE_DEMAND_DUMP`，未提交）。那個儀器在沒有它的 env 時**不執行**，但**二進位不是同一顆** ⇒ 這次的讀數與 12:57 那批**不構成「同一 build」**。

**⇒ 可引用的基準沒有變：prefill 292.34（09-18，NOMINAL 全程）、decode 12.57（09-20 12:57，NOMINAL 全程）。** 本節的價值有二：(a) **凍結的 profile 逐旋鈕重現**（命令、`-b`／`-c`／`--warm-skip`／`--spec-type` 都對得上）；(b) **「今天拿不到乾淨讀數」現在有熱態標籤當證據**，而不是一句「跑不出來」。
