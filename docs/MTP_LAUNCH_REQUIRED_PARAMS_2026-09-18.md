# MTP 啟動必跑參數（清單 ＋ 三個陷阱 ＋ 一個死旗標）—— 2026-09-18 19:19

**取得權威清單的方法（不要抄這份文件，抄這一行）**：

```sh
CGC_DUMP_ENV=1 CGC_SERVER_PROFILE=prod25 ./scripts/run_server.sh
```

它**只解析、不起 server**（`scripts/run_server.sh:746-804`，`DUMP_ONLY=1`），並逐行印出
`ENV …` 與 `ARG …` —— 那是**完全解析後**的清單。任何**不在 `SERVER_ENV` allowlist 裡**的變數
會被靜默丟掉，所以「文件上寫的」不等於「真的傳進去的」。

---

## 1. 必跑（缺了會不對／會崩）

### argv（`run_server.sh:1215-1224`）

| 參數 | 值（prod25） | 為什麼是「必跑」 |
|---|---|---|
| `--spec-type` | `draft-mtp` | 選 spec 引擎；也是 `common.cpp` 打開 `load_mtp` 的那條路（少了它 MTP head 不會載） |
| `--spec-draft-n-max` | `3`（`CGC_SERVER_MTP_N_MAX` 可覆寫） | draft 深度；**也是唯一免改 `src/` 就能掃 k 的旋鈕** |
| `--temp` | 見 §3 —— **這裡寫 0，但實際生效的是後面那個 0.4** | ⚠ 死旗標 |

- **模型要是「有 nextn head 的那一份」**：`Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`
  （`CGC_SERVER_DENSE_IQ4X=1`）。★ **沒有 `-md` / `--model-draft`** —— MTP head 在**同一份 gguf** 裡。

### env（`run_server.sh:2022-2092`，全在 `if [ "$SERVER_MTP" = "1" ]` 之內）

| env | 值 | 缺了會怎樣 |
|---|---|---|
| `CGC_NO_PREFETCH` | `1` | **MTP ＋ OA_ASYNC ＋ 背景 pool fill 會 mid-decode 崩**（0000 防護） |
| `CGC_VERIFY_DECODE` | `1` | `zero_slot_enabled()` 的兩半之一 ⇒ 沒有它 **verify 進不了 fast path**（今天才證明 bench 就是缺這個） |
| `CGC_DRAFT_DECODE` | `1` | 同上，另一半（draft ctx 的 fast path） |
| `CGC_MM_BITIDENT` | `1` | bit-identical 支柱 1／3（M-invariant kernel） |
| `CGC_MTP_NO_WARMUP` | `1` | 支柱 2／3（專家快取污染修復） |
| `CGC_NO_SEQ_RM_PROBE` | `1` | 支柱 3／3 —— 白皮書 §4.3 逐字寫「**缺一不可**」 |
| `CGC_WARM_NPAST` | `0`（`denseIQ4X=1`）／否則 `8` | 暖機門檻。★ **它也擋 fast path**（`cgc_fast_eligible = !warm_gate`，預設 2048）⇒ `-d 512/-d 0` 的 cell 會落在門檻下 |

## 2. 效能口徑（缺了不是崩，是慢或漂）

`CGC_SPAC=1` ＋ `CGC_SPAC_ALPHA=0.75`（membership 驅動，關掉 decode 工作集不駐留）、
`CGC_OA_ASYNC=1`（+12.6%）、`CGC_N_CB=8`（§8.93 sweet spot）、`CGC_GLU_FUSED_DOWN=1`、
`CGC_DBUF=1`、`CGC_LOOP_GUARD=1`、`LLAMA_EXPERT_CACHE_LAYER_CAPS=40-40:256`、
`CGC_EXPERT_CACHE_BYTES=8589934592`（**8 GiB 生產 pool**，4 GiB 會製造假的 treatment 效果）、
`LLAMA_EXPERT_CACHE_WORKERS=8`、`--cache-type-k/-v q8_0`、`-c 4096`（prod25 的 CTX）。

## 3. ★ 一個死旗標 ＋ 一個會誤導的框定（19:19 新發現）

`run_server.sh` 的 argv **同時**有兩個 `--temp`：MTP 塊給 `--temp 0`（`:1217-1220`），
生產採樣塊在其後給 `--temp 0.4`（`:1249-1250`，註解說 0.4 ＋ top_p 0.8 是生產基線）。
**llama-server 自己就警告了**（19:05 那顆的日誌原文）：

```
0.00.083.960 W DEPRECATED: argument '--temp' specified multiple times, use comma-separated
                  values instead (only last value will be used)
```

⇒ **實際生效的是 0.4；MTP 塊那個 `--temp 0` 是死的。**

**這不只是旗標問題**：多處既有述句是以「**greedy**」為前提講的 ——
「accept **在 greedy 下**不可由 accept rule 移動」、「`CGC_MTP_REJECTION` **在 temp 0** 是 by
construction 的 null」。那些量測本身可能是對的，但它們**不適用於這個 launcher 產生的 run**
（那些 run 不是 greedy）。
⇒ **任何關於 accept／`draft_accept` 的 A/B 都要連同溫度一起報**，否則兩份數字沒有同一個前提。
（同一個病本 repo 已記過多次：`CGC_MMV_FUSE`、`CGC_OA_ASYNC` 的 presence-gating、
`CGC_SOFT_POOL_L1` 吃掉 `CGC_LOOP_GUARD` —— **「看起來有做、其實沒做」**。）

## 4. 三個通用陷阱

1. **presence ≠ value（C++ 端）**：`getenv(X) != nullptr` ⇒ `CGC_VERIFY_DECODE=0` **仍然是開**。
   `run_server.sh` 因此把 0 翻成「完全不傳該變數」。第一版 A/B 就是這樣量到「逐位元相同」。
2. **allowlist 會靜默丟東西**：不在 `SERVER_ENV` 清單裡的變數傳不進去。`CGC_MM_BITIDENT` 就曾經
   因此**缺席一整個調查期**；`CGC_MTP_SAMPLER_PARITY`、`CGC_IDS_MAX_LINES` 同病。
3. **別把清單只看 MTP 區塊**：`LAYER_CAPS` 2026-09-17 之前掛在 `if MTP=1` 底下，但它**同時**決定
   loader 的 `ne[2]` 與 cache 的 `n_slots_l`，而 decode 基線臂是 **MTP=0** ⇒ 它在**最需要的那條臂**
   上被靜默丟掉。

## 5. 機檢判準（比清單可靠）

清單是「應該傳了什麼」；**有沒有真的接上**要看 launch 後的 fast-path telemetry：

```
llama_expert_cache: MTP fast path: … verify: calls=<必須 > 0> … draft: calls=<必須 > 0> …
```

- **兩類 calls 都要非 0**，否則 `[mode] MTP ON` 這句話只證明旗標被接受、不證明路徑走對了
  （**llama-bench 就是 `verify: calls=0` 的例子**，今天才發現）。
- `cold(ZERO)=0` 與 `verify-strict: zero_mapped_selected=0` 是正確性的必要條件。

## 6. 沒做（具名）

- **沒有逐一實測「拿掉某個 env」的後果**：§1 的「缺了會怎樣」來自原始碼註解與既有記錄，**未做消融**。
- **`--temp` 那個重複沒有修**（`run_server.sh` 是別條線的在用腳本；改它會動到所有人的生產口徑）。
- **未量**：temp 0 與 0.4 下 accept／t/s 的差（要同 build、同 pool、配對設計）。
- **D5**：本輪改動 0 個 `src/`（只有 `docs/`）⇒ oracle 那一半不適用。
