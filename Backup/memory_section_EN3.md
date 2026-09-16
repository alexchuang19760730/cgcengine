
---

## §EN-3. 三個「decode 臂」不是同一個東西——而 prefill250 缺的是 SPAC（2026-09-16 18:5x）

使用者指正：「`prod25-stream` 跟我們 prefill 用的不同」。**對，而且差在一個機制級的旋鈕。**

現場解析（`run_server.sh CGC_DUMP_ENV=1`，見 `Backup/cgc_logs/instr_compare/arms_matrix_20260916.tsv`）：

| arm | CTX | -b/-ub | PREFILL_STREAM | SLAB_CAP | **SPAC** |
|---|---|---|---|---|---|
| `prod25`（decode sweep 基準 `p25-gputime` 的 profile；§W 配對臂） | 4096 | 模型預設（llama-bench 推 **8/8**） | - | - | **1** |
| `prod25-stream`（歷史 depth 矩陣；「10–13 t/s」出自此） | 4096 | **512/512** | 1 | 256 | **1** |
| `prefill250`（prefill 250 交付用） | **8192** | **5632/5632** | 1 | 256 | **-（全域預設關）** |

**只有 6 項不同，但 `CGC_SPAC` 是機制級的**：全域預設關（`run_server.sh:1635-1637`
`if [ "${CGC_SPAC:-0}" != "0" ]`），只有 prod25 那塊自己開。而 prod25 的註解自己寫
「SPAC 同時是 membership 驅動，關掉後 decode 工作集不駐留」
⇒ **prefill250 缺的是 decode 的駐留機制**，不只是 batch 大小。
**所以 prefill 的 profile 不能拿來當 decode 的標準，也不該拿 prefill 的 t/s 講 decode。**

**prefill250 從來沒有產出過任何 decode 數字**：唯一嘗試
`Backup/llama_bench/llama_bench_prefill250.json`（`-b 6144 -ub 6144`）是**零列**的殘檔，
死在第一個 prompt warmup（`test_prompt: failed to decode prompt batch, res = -3`，GPU OOM），
而 6144 正是 prefill250 要的形狀。⇒ 在 16 GB 機器上 **prefill250 的 batch 與 llama-bench 的
depth 矩陣不相容**（`-d≥512` 會跑 `test_prompt(ctx, n_depth, n_batch)`）。

⇒ 要一個 profile 同時服務 prefill 與 decode，那是**新配置**（至少 `CGC_SPAC=1`），
需要自己的 decode 基準，不能借用 prod25 的。

**本輪重現的歸屬（更正§EN-2 一行）**：我重現「10+」用的是 `prod25-stream`，
它屬於 **prod25 血統**（SPAC=1）⇒ 對 decode 而言血統正確，但它**不是** decode sweep 的基準臂
（那個是 `prod25`，`-b 8`）。兩者的 depth-0 值 8.7–9.8 與 9.4 一致，所以 §EN-2 的結論不變。
