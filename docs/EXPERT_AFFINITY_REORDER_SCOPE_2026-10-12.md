# 專家親和性重排 —— engineering 範圍評估（2026-10-12）

對應 charter：`scripts/check/charters/e-expert-affinity-reorder-2026-10-12.yaml`
目標：把白皮書 §12.7.3 方案 A（專家親和性重排）從「設計主張」落成「最小 diff 可量測」的一臂。
本文只盤**改哪些點、動哪些行、最小代價**，不做任何實作。

---

## §0 先更正一處 charter 子目標的措辭（重要，會決定 diff 大小）

charter 的 `sg-affinity-implement` 寫的是「載入/度假時重排 256 專家 + remap table」。
逐行讀 `llama-expert-cache.cpp` / `.h` 之後，這個措辭是**不完整的**，最小 diff 不是一次型張量置換：

- 專家權重**常駐在固定 per-(layer,kind) pool 區域**，slot 以 `region + slot_idx*stride` 定址
  （[pool_region](file:///Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/src/llama-expert-cache.cpp#L835)、
  [pool_ext 結構](file:///Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/src/llama-expert-cache.h#L166-L168)）。
- **slot↔expert 對應是動態的**：`slot_table[expert] = slot` 每步被 fill/evict 改寫
  （[table 讀](file:///Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/src/llama-expert-cache.cpp#L1072)、
  [pick_slot 的 LRU/SpAc evict](file:///Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/src/llama-expert-cache.cpp#L654-L712)）。
- ⇒ 一次**靜態置換**（在載入時把「常共選」專家排去相鄰 slot）在**第一次 pick_slot 之後就失效**——
  LRU/SpAc 會把新 expert 放進它「找得到的佛 slots」，不再保證相鄰。
- `prepopulate`（[L2615](file:///Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/src/llama-expert-cache.cpp#L2615)）只決定**初始** slot 指派（identity：slot e ← 放 expert e），
  之後就交給 runtime 的 fill/evict。

**結論**：要拿到「steady-state 的 gather→順讀」，重排必須活在 **slot 指派/放置策略**（`pick_slot` / `prefetch_claim_slot_locked` / `slab`）裡，
**不是** GGUF tensor 置換。下面是三個由易到難的最小 diff 選項。

---

## §1 選項 A：初始 prepopulate 序（最小，~1 個函數＋1 個排序表）

**改哪**：[`llama_expert_cache_prepopulate`](file:///Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/src/llama-expert-cache.cpp#L2615) 的 identity 指派段（L2647-2653）。

**做法**：把 `for (e=0..n) table[e]=e` 換成「按共選統計排出的 perm」——
初始把**常共選的專家**放去相鄰 slot；權重內容逐 byte 不動，**只改寫哪個 slot 指到哪個 expert**（是一張 `<expert→slot>` 映射，不是權重內容）。

**代價**：
- 只動 `prepopulate` 一個函數的指派迴圈；
- 需要一個「共選排序表」輸入（見 §3 統計來源），空則退回 identity（bit-identical 預設）。

**上限**：只影響**冷啟動/換層後的第一次**佈局。之後 pick_slot 開始接管，相鄰性隨 LRU 退化。
**若不證實退化干擾**：這是最便宜的「先把機制架起來、見證行先跑、量一次冷啟動佈局效果」的發車點。

---

## §2 選項 B：pick_slot 放置偏置（真正的 steady-state 最小 diff）

**改哪**：[`pick_slot`](file:///Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/src/llama-expert-cache.cpp#L654) 的第一段「找佛 slot」（free-scan，L662-667）＋
[fault 指派端](file:///Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/src/llama-expert-cache.cpp#L1459)（pass 2 逐 miss 呼叫 pick_slot 處）。

**做法**：當池內有**佛 slots**（free-scan 中選項），不再是「第一個」（L665 `return i`），
而是**挑一個離「本 batch 目前已指派、且與本 expert 共選頻率高」的 partner slot 最近的佛 slot**。
關鍵約束：**「必須仍收斂到一組合法 slot，且不引入新競態」**——B 的偏置只在 free-scan 的選擇順序上做，
fault/evict/batch_owned/cap 等既有不變式**一律不動**。命中 `CGC_EXPERT_AFFINITY=1` 才走，缺省=現行第一個 free slot=bit-identical。

**代價**：
- 動 `pick_slot`（熱路徑、多鎖、既有 batch_owned/pinned/SpAc 交互）——這是**高風險改動點**；
  需嚴守既有「pass 0/1/2 回退 + LRU tie-break」的決定性，否則會重蹈 L201/SUBMT_AHEAD 那類 racy。
- 需要 co-partner 距離表（§3）。
- **不建議預設發車**；建議先選項 A 證機制，確認見證行+量測 harness 跑得動，再上 B。

---

## §3 共選統計（它偏難，需要先想清楚）

「共選排序表」/「co-partner 距離」不能憑空來。來源有三：

1. **既有 `CGC_PREROUTER`**（[L1900](file:///Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/src/llama-expert-cache.cpp#L1900)）
   記錄**每層每個 expert 被路由的次數**（`freq[layer][expert]`，L189）。缺共選（joint-count）——
   但 `slot_owner`/`freq` 只知道「單一 expert 的出現次數」，**不知道「同一 token 誰跟誰一起出現」**。
   ⇒ 需要從「實際 routed union」多拿一維 joint-count（每 token 8 選 → pairwise co-occurrence matrix，256×256/layer）。
2. **prefill 累積**：`freq` 已在 prefill 累積（prewarm_hot L188-189），可在同一掛鉤順手累 joint-count（預設 OFF，非侵入）。
3. **靜態模型先驗**：對 256-expert MoE，「常共選」多數是**相同語意族的專家**（router 學習出語意分群）——
   用 **router 權重做離線近似**當冷啟動排序，**不需 routing trace、不改 decode 熱路徑**（見 §3.3）。

**誠實風險**：joint-count 統計本身需要**改 probe**（`expert_cache_on_topk` 那一路），
它現在只累計加速度地批量單 expert 出現（L203 的 `massc_sel_total` 是 count 不是 joint）。
要把「同 token 的 8 選」pairwise 化，需要在 routed union 迴圈裡多一層巢狀計數——這是**獨立的一塊工程**，不是「擴一行」。
**因此 §5 的先發車建議是 §3.3（離線、零熱路徑），不是 joint-count。**

---

## §3.3 離線 perm 產生方案（純離線、不改 decode；先驗「排不排得出相鄰」）

**輸入張量（2026-10-12 探針實測，非假設）**：`blk.{l}.ffn_gate_inp.weight`，shape `{n_embd_=2048, n_expert=256}`，
**type=F32**（`affinity_probe.py` 實測 `blk.0.ffn_gate_inp.weight -> ne=[2048,256] type=0(f32)`）。它是**路由器權重**：
每一 hidden state（2048）`. ffn_gate_inp` → 256 個專家 logit → softmax → **top-8**。n_expert 本模型 = 256（非 48），topk=8。
router 是稠密、非 pooled（L4 skip），所以它**不是**要重排的專家權重，而是**決定「每 token 哪些專家一起被選」的導覽圖**。

**重要校正**：scope 初期誤寫 `ffn_gate_exps` 且假設 n_expert=48。實測後：
- 路由器張量是 `ffn_gate_inp.weight`（不是 `ffn_gate_exps`，後者是各專家的 up-net gate，shape `{2048,512,256}` IQ4_XXS）；
- n_expert=256（不是 48）；
- router **是 F32** ⇒ 不需 qctl 解量化，可直接讀 GGUF 的 F32 bytes。

模型檔：`models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`（存在，n_layer=40）。

### §3.3.0 機制主張（為何 router 張量能引出共選）
MoE 的「常共選」是由路由器語意分群決定的：**對同一類輸入激活的專家，其 router logit 彼此相似**。
故 pairwise 共選矩陣 ≈ router weight 行向量（256 × 2048）的 pairwise 相似度（餘弦）。用 router 取代 routing trace 的代價，
是「近似、非精確」——但對**選項 A 的冷啟動初始序**已足夠，且可先驗證，不需先動熱路徑。
（誠實邊界：router 相似度 encode 的是「權重空間聚類」，**不是**「同一 token 的實際共選 ground truth」；
真正的共選要等 `CGC_PREROUTER` 的 joint-count。§3.3.2 的離線報告只是**訊號先驗**。）

### §3.3.1 步驟（全離線）
1. **讀 router F32 權重**：用 `gguf_retensor.parse_header` 拿 `stored_off`，mmap 讀 `{2048,256}` F32 塊（`2MB`，一次讀出），
   reshape 成 256 行 × 2048 維。無需解量化（F32）。
2. **（可選）取樣 query**：若想逼近真實 gate score，可對 router 列做 SVD/降維或對列空間取樣得到一組「激活方向」當 query；
   或直接用**行相似度**做主分群（↓ 更省、更穩）。本方案預設用**權重行相似度**（256×256 餘弦）。
3. **累計共選/相似矩陣 C**：router 行對行的餘弦相似度 → 256×256 對稱矩陣。
   （若走 query 版：每 query 取 top-8 logit → pairwise co-occurrence；本方案先不只做行相似度，因不需外部 query。）
4. **排 perm**：對 C 做貪心近鄰鏈（對稱 TSP）或 spectral 降維排序，得 256-order perm =「何時把專家依序放進 slot」。
   tie-break 固定 generator，跨 run 可重現。
5. **匯出**：一張 per-layer `perm_256` 表（JSON/二進位），給 prepopulate（選項 A）吃。

### §3.3.2 先驗證「排不排得出相鄰」（不改碼就能做）
在寫任何 C++ 前，先用 `affinity_probe.py` 讀權重、離線算排序品質：
- **工具**：`scripts/affinity_probe.py`（新增，只讀 GGUF 檔頭 + F32 塊）。**不碰 llama-expert-cache.cpp**。
- **指標（可判）**：給定 perm，量「同一 token 的 top-8 在 perm 後位址的成對距離」——但沒有真實 trace 時，
  改用「權重相似度最高的 8-neighborhood 在 perm 後的平均位址距離」vs「identity 序」。
  若 perm 後該距離**顯著小於** identity ⇒ 排序有意義，值得投選項 A；若無差異 ⇒ router 相似度抓不到集群 ⇒ 收斂或改 joint-count。
- **誠實口徑**：§3.3.2 的量是**權重空間聚類訊號**，**不等於 decode t/s 增益**；真正閘門仍是 charter 的
  `success ≥5% t/s`（配對、同 session、ABBA）。本節目的只是「在花 3–15 天動 C++ 前，先看排序有沒有訊號」。

**2026-10-12 實測 layer 0（affinity_probe.py --verify，探針通過）**：
```
# metric: top-8 neighbourhood mean slot-distance | identity=89.79 perm=50.14
# relative change: +44.2%    (距離縮短 = 訊號)
# signal verdict: YES cluster
# similarity density: pairs>0.5 = 3.7%   (稀疏 / 局部聚類)
# 產物: models/gguf/perm_layer0.json (256-perm)
```
**判讀**：◎ 排序在權重空間**確定能拉近共選專家**（top-8 鄰域平均位址距離 89.79→50.14，−44%）⇒ 值得投選項 A。
⚠ 但 similarity density 僅 3.7% ⇒ 聚類是**稀疏、局部**的（少數語意族緊密，其餘彼此低相似），對選項 A
（冷啟動初始序）是利好，對 steady-state decode 增益是**保守訊號**。誠實邊界不變：此為權重距離，非 decode t/s；
真正閘門是 charter `success ≥5%`（配對、同 session、ABBA）。

### §3.3.3 與選項 A 的介面
perm 表 → `prepopulate` 指派迴圈（[L2647-2653](file:///Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/src/llama-expert-cache.cpp#L2636-L2654)）的替換表：
`for k in 0..n: table[perm[k]] = k`（權重內容零改動，只改 slot↔expert 指標）。
見證行 `CGC Expert affinity reorder: active (n_perm=256)` 在 ON 時印。
缺省＝無 perm，identity（bit-identical）。

### §3.3.4 陷阱
- **本模型 n_expert=256、topk=8、n_layer=40**（多份 REPLACEMENT/decode 文件佐證），不是 48；
  perm 是 256 長度、per-layer（40 層）。最小驗證先只做 layer 0，看訊號再決定全層。
- **router 是 F32** ⇒ 不需解量化，big win；但 F32 讀 2MB/層 × 40 = 80MB 全模型 router 可一次讀完（< 1 秒）。
- **perm 改變 slot 指派 ≠ 改變 bit**：符合 Exact-Cache 原則（M1/M2/M3 透明），見 §6。

---

## §3.5 選項 A 實作：minimal diff 明細（已實作 + 冒煙測試通過，2026-10-12）

實作前把兩個不變式先釘死（LLVM line ref 為現狀）：
- `slot_table[layer*n_expert + expert] = slot`（**expert 索引 → slot**，FND [L2647-2653](file:///Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/src/llama-expert-cache.cpp#L2647-L2653)、[L1068](file:///Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/src/llama-expert-cache.cpp#L1068)）
- `slot_owner[layer][slot] = expert`（**slot 索引 → expert**，[L1181](file:///Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/src/llama-expert-cache.cpp#L1181)）

#### diff 1（核心，prepopulate 指派迴圈）
現行 `for e: table[e]=e; slot_owner[e]=e` 只在 identity（expert==slot）下成立。perm 版改為**以 slot k 為主迭代，放 expert perm[k]**（atomic set）：

```cpp
// 三處都從「用 e 當 slot」改成「用 k 當 slot」——
for (uint32_t k = 0; k < n && k < cache->n_expert; ++k) {
    const uint32_t exp = (affinity_perm.empty() || k >= affinity_perm[layer].size())
                       ? k : affinity_perm[layer][k];
    if (table[exp] < 0 && cache->slot_owner[layer][k] < 0) {
        table[exp] = (int32_t) k;              // expert exp -> slot k
        cache->slot_owner[layer][k] = (int32_t) exp;   // slot k holds expert exp
        cache->slot_last_use[layer][k] = 0;
    }
}
```
`n = min(n_slots, usable_slots, |perm|)` 已含 clamp；`affinity_perm[layer]` 空 ⇒ exp=k ⇒ **identity，bit-identical 預設**。

#### diff 2（CGC_POOL_SPLIT owned-pool 分支，L2630-2634）
owned pool 是**唯一需要 pread 真 bytes** 的路徑，`fill_pool_direct(cache, layer, slot_idx, expert)` 已把 expert bytes 放進 slot_idx。perm 下必須「slot k ← expert perm[k]」：

```cpp
for (uint32_t k = 0; k < n && k < cache->n_expert; ++k) {
    const uint32_t exp = (affinity_perm.empty() || k >= affinity_perm[layer].size())
                       ? k : affinity_perm[layer][k];
    fill_pool_direct(cache, layer, (int32_t) k, exp);
}
```
權重位元來自 `exp` 的原始 GGUF，**逐 byte 不改**，只改「哪個 slot 裝哪個 expert」的指標——符合 Exact-Cache。

#### diff 3（member + 載入函數）
- **header**：`struct` 加一個 member，default 空（非侵入）：
  `std::vector<std::vector<uint32_t>> affinity_perm; // [layer][expert→slot 排列]；空=identity`
- **新外部函數**（`nm` 可驗證，非 static-inline）：
  `size_t llama_expert_cache_load_affinity_perm(llama_expert_cache * cache, const char * path)`
  讀 bare 文字表（`layer expert0 expert1 ...`，strtok_r，**沿用 [load_pin_profile L2567](file:///Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/src/llama-expert-cache.cpp#L2567) 的既有先例**，不引 JSON parser）；
  回傳載入的 (layer,k) 對數；0 = 沒吃到表 ⇒ 退 identity。

#### diff 4（llama.cpp 載入點）
放在 [pin_profile 載入之後、兩圈 prepopulate 之前](file:///Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/src/llama.cpp#L566-L603)——與 PIN_PROFILE 同生命周期（載入即就位，prepopulate/ensure_batch 之後由 runtime 接管）：

```cpp
const char * aff = getenv("CGC_EXPERT_AFFINITY");
if (aff && aff[0]) {
    const char * aff_file = getenv("CGC_EXPERT_AFFINITY_FILE");   // 缺省 = <gguf-dir>/perm_layer{N}.json 退化到 .txt
    const size_t n_aff = llama_expert_cache_load_affinity_perm(model->expert_cache, aff_file);
    llm_... LOG ("CGC Expert affinity reorder: active (n_perm=%zu)\n", n_aff);
}   // 缺省/無 CGC_EXPERT_AFFINITY：見證行不印，走現行 identity（非侵入）
```
見證行只在 env ON 且表載入成功時印——正好命中 charter `window_valid (b)`。

#### 不變式守衛（防 L201/SUBMT_AHEAD 那類 racy）
- perm 只影響 **prepopulate 的初始指派**；`pick_slot`/`ensure_batch`/evict/fault 等**一行不動**。
- 指派在**載入期**（cache->m 鎖內、無 decode 併發）完成，不引入任何 runtime 競態。
- `slot_owner` guard（`< 0` 才寫）讓已 pin（PIN_PROFILE）或已 fill 的槽**維持原樣**，與現行「prepopulate 只補剩餘 slot」語意一致。

#### minimal diff 總量
| 檔 | 改動 |
|---|---|
| `llama-expert-cache.h` | +1 member |
| `llama-expert-cache.cpp` | +1 載入函式（borrow load_pin_profile 骨架）、prepopulate 指派迴圈、owned-pool fill 迴圈 |
| `llama.cpp` | +6~8 行 env gate 呼叫 |

**完全不動**：`pick_slot`、`ensure_batch`、`mul_mat_id`、提交/readback 併發、pool 容量。

---

## §4 見證行 ＋ nm 可驗證（charter `window_valid` 的前置）

charter 要求：見證行 `CGC Expert affinity reorder: active` 且 `nm` 可驗證符號。

- 符號命名對齊現有 `llama_expert_cache_*`：**`llama_expert_cache_load_affinity_perm`**（namespace 外部，非 static-inline，`nm` 已有 `T` 符號）。
- `nm` 可驗證前提：**不要** static-inline 到只用於單一 TU、會被優化掉；要有一個非 inline 的外部函數觸發。
- env 開關：`CGC_EXPERT_AFFINITY=1`，缺省一律走現行路徑（bit-identical，非侵入式）。見證行只印在 ON 時。

**⚠ 見證行輸出通道（2026-10-12 smoke 實測修補）**：
見證行**要用 `fprintf(stderr, ...)`，不要 `LLAMA_LOG_INFO`**。smoke 測試抓到：
在 expert-cache 早窗（llama.cpp L589 這個 point），`LLAMA_LOG_INFO` 走 llama logger，該時窗 logger 尚未綁定，
訊息靜默丟失（實測只有 loader 內部的 `AFFINITY: ...`（`fprintf(stderr)`）出現，見證行 `LLAMA_LOG_INFO` 版完全沒印）。
已改 `fprintf(stderr)`，實測見證行正常輸出：`llama_model_load: CGC Expert affinity reorder: active (n_perm=40)`。

**n_perm 語意**：C++ 印的是**載入的層數（層數 = 40），不是 n_expert（256）**。

---

## §5 建議的落地順序（把風險壓到最低）

1. **先 §3.3 離線 perm**：用 router 權重建 perm 表 → **§3.3.2 離線驗證「排不排得出相鄰」**，
   完全不碰 decode 熱路徑、不改任何 C++，先確認排序是否有訊號。
   ✅ **2026-10-12 已完成**：layer 0 探針通過，`identity=89.79 → perm=50.14`（−44%），verdict **YES cluster**
   （詳情見 §3.3.2。相似度密度 3.7% ⇒ 訊號稀疏/局部，對 decode 為保守訊號）。
   **下一個動作 = 步驟 2（選項 A）**。
2. **再 §1（選項 A）**：prepopulate 吃這張 perm → 最小 diff、最熱路徑以外的改動 → 驗證見證行 + 跑一趟量測 harness + M1/M2/M3。
   ✅ **2026-10-12 C++ 已實作 + 冒煙測試通過**：`CGC_EXPERT_AFFINITY=1` 載入 40 層 perm
   （`models/gguf/perm_all_layers.txt`），見證行 `active (n_perm=40)` 正常輸出（§4 通道修補），server 載入+running 不崩。
   **下一個動作 = 步驟 3：ABBA 量測（identity 13.72 錨 vs affinity-on，同 session 配對）**。
   - 若冷啟動佈局增益 ≥5% ⇒ 證明「佈局軸」有料，charter success 開綠 → 才考慮 §2（B）。
   - 若 <5% 或證實 LRU 立即吃掉相鄰性 ⇒ charter falsify，佈局軸與池軸同例收斂，**不再投 B**（省 5–15 天風險）。
3. 只有在 A 成立且有必要時，才動 §2（B）這個熱路徑點——B 是本專案裡風險最高的一類改動（多鎖、決定性、racy 歷史）。

**一句話**：最小 diff = 選項 A（prepopulate 吃離線 perm），它的作用域最遠離熱路徑、最易先驗證佈局軸是否真有料；B（pick_slot 偏置）是後續才值得碰的高風險點。

---

## §6 與既有判決的對界（避免重蹈）

- **不是**「縮池」（`CGC_POOL` / soft-pool）：本方案**不動池容量**，只動 slot↔expert 指派次序（純指標層）。
- **不是** kernel C 軸（已判死）：本方案**不改 `mul_mat_id`**，只讓它讀到的 slot 在 DRAM 上更相鄰。
- **不是** racy/SUBMT_AHEAD（`#if 0` 封死）：本方案**不改提交/readback 併發**；B 若碰 pick_slot 必須連既有
  `batch_owned`／pinned／SpAc 等不變式一起審，不能採「放寬同步」換增益。
- **權重位元逐 byte 保留**（符合 Exact-Cache 原則 + M1/M2/M3）：重排只改「哪個 slot 裝哪個 web」，不改 GGUF 位元組。

---

## §7 未定項（落地前需定的口徑）

1. 共選統計的維度：pairwise joint-count（精確）vs router 近似（離線、零熱路徑代價）——建議先後者。
2. 選項 A 的增益是否會被 LRU 立即吞掉——**這是本選項 A 的先驗證問題**，跑一趟才知。
3. `n_slots < 256`（本機 n_slots=143 < n_expert=256，故池裝不下全 256 專家）時 perm 範圍要 clamp——A 已含 `n = min(n_slots, usable)`，OK。
4. 「相鄰」的定義：是**同一 kind 的 stride** 相鄰，還是跨 kind（gate|up+down）也疊？——第一步只用 per-kind，最簡單。