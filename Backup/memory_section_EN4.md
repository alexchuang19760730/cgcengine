
---

## §EN-4. 「哪個臂最有潛力」的判準在原始碼裡：池夾具綁住了 MTP 的深度（2026-09-16 18:5x）

使用者要求「選個最好有潛力的統一一下」。先把**判準**固定下來（與後續量測結果無關的耐久事實）：

### 1. 三個臂其實只差 6 項，其中 4 項只是「寬 prefill 走哪條路」

見 `Backup/cgc_logs/instr_compare/arms_matrix_20260916.tsv`（現場解析）。
真正有語意的只有兩項：**`CGC_SPAC`**（decode 駐留；全域預設**關**，只有 prod25 血統開）與 **CTX**（4096／8192）。

### 2. `CGC_POOL_MAX_TOKENS` 是可調的，但它就是 MTP 深度的天花板

`src/llama.cpp/src/llama-graph.h:18-37`：

```cpp
// Default 8 covers MTP n_max up to 7 (verify = n_max+1).
// Tunable via CGC_POOL_MAX_TOKENS (clamped to [2, 64]).
// Note: larger values may increase memory pressure.
```

`llama-context.cpp:286-291`：`if (pool_capacity > 0 && n_batch > 1 && !cgc_prefill_stream) n_batch = pmax;`
——註解自己說這個夾具的用途是「**so speculative/MTP verify batches (n_max+1 <= cap)** flow through
the pool path」，並且「**The clamp is NOT lifted for an owned pool**」。

⇒ **判準**：`prod25`（無 PREFILL_STREAM）把池路徑的批寬鎖在 8 ⇒ 放寬它就得在 **143 slots/layer**
裡裝更寬的 expert union（這台引擎的容量壓力就在這裡：worst layer 248 distinct vs 143 slots）。
而 **M1（batched-union gather）與 M4（verify 真批次）本質上住在「寬 batch」的世界**，
那需要 **whole-layer slab（`ne[2]=n_expert`，裝得下 256 個）——只有 PREFILL_STREAM 的臂有**。
⇒ **「潛力」不是現在的 t/s，是「剩下來的槓桿有沒有地方跑」。** 這一點讓 `prod25` 成為三個臂裡潛力最低的。

### 3. 我的建議（等 A/B 定案）

**統一在 `prefill250` 形狀 ＋ `CGC_SPAC=1`（alpha 0.75）**，理由：
(a) 只有它同時具備 slab（承載寬 batch 的槓桿）與已交付的 prefill 250+；
(b) 它對 decode 只缺 **一個**旋鈕，而 `SPAC` 是一行、可 A/B；
(c) `prod25-stream` 保住了 SPAC 與 depth 矩陣，但 prefill 只有 111–120 ⇒ 為了 decode 的 ~15%
    放棄 2.2× 的 prefill 是壞交易。
**量測形狀**：prefill 用 profile 的 `-b/-ub 5632`；decode/depth 矩陣用 `-b 512`
（量測參數、不是 profile 差異；5632 在 16 GB 上於 `-d≥512` 會 OOM，已證）。

**未量的保留**：`SPAC_ALPHA=0.75` 是在 ctx 4096 調的，ctx 8192 可能要重調；
ctx 8192 對 decode 駐留的影響未量。

**正在跑的否證實驗**：`Backup/run_unified_ab.sh` —— `prefill250` vs
`prefill250:CGC_SPAC=1;CGC_SPAC_ALPHA=0.75`，交錯 A/B/A/B，每臂等 NOMINAL，
`-b 512 -ub 512 -p 0 -n 128 -d 0,512 -r 3`。**若 SPAC=1 明顯較慢，統一答案就改成 `prod25-stream`。**
（附註：第一版 driver 的等待迴圈不印任何東西 ⇒ log 看起來像掛掉；已改成耐心 10 分鐘 ＋ 每 30 s 印一次。）
