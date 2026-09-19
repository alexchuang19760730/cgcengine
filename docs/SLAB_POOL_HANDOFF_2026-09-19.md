# Slab → pool 交棒：把 prefill 的 hot set publish 回 pool (2026-09-19)

問題（使用者）：讓 slab prefill 結束時把 union publish 回 pool（或對 decode 收斂集做一次預取），
量 **prefill 250 與 decode 22+ 是否能同時成立**。

一句話結論：**publish 機制做好了、能跑、成本算得出來，但它對 decode 沒有效果** —— 因為
decode 的 miss 不是「一個固定的熱集」，是產出文字本身的 routing churn。三個臂的
compulsory miss 是 1671 / **2089** / 1669（交出 1280 個 expert 的那臂反而最多），
所以「用集合去 publish」在 slab 這一側同樣被量測關掉。

---

## 1. 為什麼原本的機制不會動

`prewarm_hot`（`llama-expert-cache.cpp`）就是為這件事存在的：把 prefill 記錄的
`freq[layer][expert]`（`record_routes`，**無條件呼叫**）排序取 top-K，填進 pool。
但它的守門是 `hot_prewarm_done` —— **process 級別的一次性旗標**。在 server 裡：

- 第一個 decode step 把它吃掉（`hot_prewarm_done = true`）；
- 之後**每一個請求**的 prefill→decode 轉換都拿到 0 次 publish。

而 slab 路徑（`CGC_PREFILL_STREAM=1`）把每層 FFN 權重重指向 per-layer slab，**完全不寫 pool**，
所以「pool 是滿的、但不是 decode 要的那批」正是 2026-09-17 兩臂的計數器講的事
（slab 臂解碼側 1,210 miss、capacity=0）。

## 2. 實作（預設關，`CGC_SLAB_HANDOFF=<cap>`）

- `llama_expert_cache_prewarm_hot_capped(cache, cap)`：與 `prewarm_hot` 同一套 top-K 選擇，
  但 (a) **沒有一次性守門**，每個 prefill→decode 轉換都能跑；(b) 以 `cap` 為每層上限；
  (c) 走 `ensure_slot` ⇒ `pick_slot` ⇒ `evict_lru`，所以是**publish（可淘汰）**而不是
  free-slot-only 的 prefetch。
- `llama-context.cpp`：slab 分支設 `cache->handoff_pending`；**第一個 decode step 的建圖處**
  執行 publish，並印出成本：`CGC-SLAB-HANDOFF: cap=32 warmed=1280 experts in 568.8 ms`。
- `run_server.sh` 加 allowlist（這個專案的老陷阱：不在 allowlist 的 env 會被靜默丟掉）。

## 3. 量測（同一顆 binary，三臂輪替，2 個請求／臂）

配方：`CGC_SERVER_PROFILE=prefill250`（**slab ON**：`CGC_PREFILL_STREAM=1`、`SLAB_CAP=256`、
ub 6144）、MTP off、pool 8192 MiB、143 slots/層、prompt ≈1294 tokens、`n_predict=24`、port 8123。

| 臂 | req1 prefill | req1 decode | req2 prefill | req2 decode | publish | compulsory / capacity |
|---|---|---|---|---|---|---|
| off | 137.5 t/s | 7.91 t/s | 13.4 t/s | 20.07 t/s | – | 1671 / 34 |
| `CGC_SLAB_HANDOFF=32` | **130.2 t/s** | **7.87 t/s** | 17.2 t/s | **14.87 t/s** | **1280 experts / 568.8 ms** | **2089** / 60 |
| off2（對照） | 138.3 t/s | 7.77 t/s | 17.0 t/s | 17.86 t/s | – | 1669 / 35 |

（req2 的 prefill 只有 4 token：prefix cache 命中 ⇒ req2 基本上是純 decode，正好是「pool 已經熱了之後」的那一格。）

### 3.1 publish 的成本是精確的、而且它是 latency 不是 throughput

1280 = 40 層 × cap 32，568.8 ms。這個時間被算進 **`prompt eval time`**：prefill 9.4 s 上多 0.57 s
⇒ 137.5 → 130.2 t/s（−5.3%，與 0.568/9.4 = 6.0% 對得上）。也就是說 publish 的第一個代價是
**首 token 延遲**，不是 prefill 的吞吐；但它對 decode 沒有回報（見 3.2），所以這 568 ms 是淨損。

### 3.2 它沒有買到任何東西，而且方向是錯的

- req1 decode：7.87（on）vs 7.91 / 7.77（off）—— 完全一樣（publish 發生在 decode 之前）。
- req2 decode：14.87（on）vs 20.07 / 17.86（off）—— 兩個對照臂自己差 12%，而 treatment 比兩個都低。
- **compulsory miss 反而從 1671 升到 2089（+25%）**，hit rate 92.4% → 90.9%。
  「compulsory」的定義是某個 (layer, expert) 第一次被需求 —— 交出 1280 個 expert 之後
  它應該**下降**。它上升，代表那 1280 個與 decode 真正第一次碰到的集合**幾乎不重疊**，
  而那些填充佔用了 slot（capacity 34 → 60）。

### 3.3 為什麼（機制，不是猜測）

`freq` 是 prefill 的 routing 頻率。本輪的 prompt 是同一段文字重複 20 次 ⇒ prefill 的 routing
被那段文字的固定 pattern 主宰，於是「top-32 by frequency」= 那 32 個在重複文字裡最常被選的
expert；而 decode 階段的 churn 落在別的地方。**這個 prompt 形狀把「頻率熱集」放大成了一個
特別不適用的預測器**，這一點必須揭露：換成不重複的長文，頻率熱集的重疊率會比這裡高，
但**方向不會變** —— 因為 `DECODE_SYNC_BATCH_2026-09-19.md` §1.2 已經在 pool 路徑上量過：
hit 93.9%、`capacity=0`、**miss 100% 是 compulsory**。首次觸及無法用「上一個 token」「上一個
prefill 的頻率」或「任何固定的集合」覆蓋，因為它就是新文字帶來的新 routing。

## 4. 所以「prefill 250 + decode 22+ 同時成立」嗎

用這一輪的實測（同一台機器、同一個 binary）：

| | 量到 | 說明 |
|---|---|---|
| prefill（slab armed，1294-token chunk） | **130–138 t/s** | 這不是 250；250 是 2048-token chunk 且 page cache 熱的口徑，且本輪把 publish 的 0.57 s 也算進 prompt eval |
| decode req1（池冷） | **7.8–7.9 t/s** | publish 沒有改變它 |
| decode req2（池熱） | **17.9–20.1 t/s** | 兩個無 publish 的臂；**22 已經在誤差邊緣**，而且它不需要 publish，只需要 pool 是熱的 |

⇒ **兩個數字可以「先後」成立（先 250 的 prefill，之後 decode 逼近 20），但不能靠「把 prefill 的
union publish 回 pool」同時成立**；原因是 decode 的 miss 是 churn，而 churn 的來源是生成內容本身。

**還沒測、但成本已知的變體**（留給下一輪，別把它與已測的混為一談）：
- 「尾巴 union」而不是頻率熱集：只 publish **最後一個 prefill token** 的 union（8 experts/層
  = 320 experts ≈ 0.15–0.5 s）。它比頻率集合更接近「decode 收斂集」，但依 §1.2 的算術，
  320 個 expert 對上 ~1000 個首觸集合，覆蓋上限約 1/3。
- cap 放大（64/128）：成本線性（cap 64 ≈ 1.1 s、128 ≈ 2.3 s），而 §3.2 顯示覆蓋率不隨集合變大
  而改善（2089 vs 1671 是反向的），所以在換掉「集合來源」之前不該加大 cap。

## 5. 閘門

`m123_oracle_gate.py --profile prefill250 --port 8080 --env CGC_SLAB_HANDOFF=32`
（參考 `ref_iq3_pool8gb_M2_6144_bitident_v6_nbaware.jsonl`）：

**M1 9/9、M2 9/9、M3 9/9**；`config_diffs` 只有 `ENV.CGC_SLAB_HANDOFF: ref='<absent>' now='32'`
（參考檔比這個旋鈕更早存在），所以「publish 改變 pool 內容但不改變數值」是量到的。
預設路徑（旗標不設）的乾淨 PASS 見 `DECODE_SYNC_BATCH_2026-09-19.md` §1.3。

## 6. Provenance 與未提交

- HEAD `98b44c8c6`；改動：`src/llama-expert-cache.{h,cpp}`（新函式 + `handoff_pending`）、
  `src/llama-context.cpp`（cap helper、slab 標記、decode 首步 publish + 計時列）、
  `scripts/run_server.sh`（allowlist）、`scripts/check/slab_handoff_ab.py`（本報告的驅動）。
- 原始資料：`/tmp/slab_handoff/{off,on,off2}.launch.log`、`/tmp/slab_handoff/result.json`。
- 機器已清乾淨（0 server、0 driver）。**未 commit。**
