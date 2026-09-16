
## §EN-6 統一 profile 落地：`prefill250` 加 `CGC_SPAC=1`（2026-09-16 19:10–）

使用者裁定「一起做」(a)(b)：(a) 把 SPAC 寫進 `run_server.sh` 的 `prefill250`（讓統一落地成規格，不只是建議）；
(b) 用新 profile 跑完整 depth 矩陣 ＋ 在 ctx 8192 上重掃 `SPAC_ALPHA`。

**(a) 已落地**：`scripts/run_server.sh:335-336`

```sh
[ -z "${CGC_SPAC+x}" ]       && CGC_SPAC=1
[ -z "${CGC_SPAC_ALPHA+x}" ] && CGC_SPAC_ALPHA=0.75
```

其餘一字未動。`CGC_DUMP_ENV=1` 實測 `CGC_SPAC=1` / `CGC_SPAC_ALPHA=0.75`；`CGC_SPAC=0` 覆寫有效
（回到 09-15 GA 形狀）；`qa-zh` / `off` 不受影響（`prod25` 本來就有）。**判準寫進註解與 skill**：
剩下的槓桿住處 —— `llama-graph.h:18-37`（`Default 8 covers MTP n_max up to 7`）＋
`llama-context.cpp:286-291`（`!cgc_prefill_stream` 才夾，owned pool 也不解除）⇒ `prod25` 的池路徑鎖在
`n_batch<=8`；M1／M4 需要 whole-layer slab。

**D5：這是一次名目重基線，不是數值重基線。** 第一次跑（`--tag spac_unified --allow-incomparable`）
報 INVALID COMPARISON，2 個 diff 就是 `ENV.CGC_SPAC` / `ENV.CGC_SPAC_ALPHA`（ref `<absent>`），
而同一份輸出裡 M1／M2／M3／fnv1a64 **全 9/9** —— 與 09-15 `CGC_OA_ASYNC` 那一輪同一形狀
（gate 檔頭有記載）。處置：`--tag spac_v5 --write-ref ..._v5_spac.jsonl --ref-note "…"`，`DEFAULT_REF` 改指 v5，
重跑 `--tag spac_unified_v5` → **`comparable=True`、M1/M2/M3 9/9**。

**證據標準（比結論重要）**：v5 與 v4 的 jsonl **md5 相同**
（`a0a0ca742ca94e843c54b39981742738`）⇒ 檔案沒有被重推導、只是被重新蓋章。
**「M1 = 9/9」是弱述句（「我看的時候是 9/9」）；「md5 相同」才是強述句。** v5 的 `.cap` 有記下
`CGC_SPAC=1`（v4 是 `<absent>`）。

**刻意沒做**：沒把 `CGC_SPAC` 塞進 `DIAGNOSTIC_KEYS`。那個集合會把鍵從 `config_stamp` 濾掉 ⇒ `.cap`
從此不再記錄它，下一個讀者分不出 SPAC-on 與 SPAC-off 的基線。`CGC_SLOT_TABLE_GPU` 的前例**不適用**
（那是「整條主張就是 bit-identity」的實驗旋鈕，不是生產預設）。

**副作用盤點（pin 一個 env 鍵會靜默重解析誰）**：`scripts/check/` 裡帶 `--profile` 的只有 3 支 ——
`decode_sweep.py`（**預設 `prefill250`**）、`m123_oracle_gate.py`（同）、`ab_interleave.py`（`prod25`）。
⇒ **不帶 `--profile` 的 decode sweep 從此取得 SPAC=1**（先前記錄的 6.6–16.17 t/s 用的是
`--profile prod25`，那個本來就是 SPAC=1，不受影響）。而 `decode_sweep` 的 `spac-on` 臂在新 base 上
與 `baseline` **同義**（退化）—— 它的 `ARMS` 註解原本還自相矛盾（第 93 行說「every arm 的 base 是
prefill250」，第 119 行說那一段是「tested on the prod25 base profile」），一併修正。
`run_native_baseline.sh` 直接 `exec` 二進位（不經 profile）⇒ 不受影響。

**成本路徑（為什麼 prefill 必須重量，不是假定）**：`llama-context.cpp:4736-4741` 的 EMA 更新註解自承
「Fires on every path that reaches here — large prefill, MTP draft and trunk verify alike」，且每次取
`cache->m`；`spac_prefetch` 預設 `CGC_SPAC_REFRESH=1` ⇒ **每一步**對每層非駐留專家做 partial sort。
兩者都在 prefill 熱路徑上。

**背景跑的量測**（三個驅動，串接、每臂等讀數回 0）：
`Backup/run_spac_prefill_ab.sh`（交錯 A/B/A/B、`CGC_SPAC=0` vs profile 預設）
→ `Backup/run_unified_depth_matrix.sh`（單行程升冪／降冪 ＋ 每 depth 各自一行程 = 順序效應控制）
→ `Backup/run_spac_alpha_sweep.sh`（SPAC off 對照 ＋ alpha 0.5／0.75／0.9，ctx 8192）。
**每臂之間等 NOMINAL 是硬規則**——今天已經被這個咬過兩次（d1024 那一格作廢、四臂連跑第二臂起就 HEAVY）。
