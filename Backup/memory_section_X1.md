
### §X.1 MTP 量得到——而且**不用改 llama-bench**（2026-09-16 18:40，我先前說「結構上量不到」是過度陳述）

使用者問「它結構上量不到 MTP 的加速 沒法修改嗎」。查完原始碼：**可以，而且不必改。**

**MTP 迴圈不在伺服器私有碼裡**：它在 `common/speculative.{h,cpp}`（`_init/_draft/_process/_accept/
_print_stats`，還有 `common_speculative_need_embd_nextn` —— nextn 就是 MTP）。全 repo 只有**兩個**
呼叫者：`server-context.cpp` 與 **`examples/speculative-simple/speculative-simple.cpp`**。

**`llama-speculative-simple` 本來就是本專案的 MTP 驅動器**：用 `common_params_parse` ⇒ 吃
`--spec-type draft-mtp --spec-draft-n-max 3`（逐字等於 `run_server.sh:1137` 給 llama-server 的旗標）；
原始碼裡明寫「MTP draft: the nextn head lives inside the target model … mirrors llama-server」；
**內建一條無投機的基線臂**（註解 `C0 baseline arm`）；git 歷史有兩筆本專案的 MTP 修復直接提交給它
（`b7364f886` bit-identical spec vs non-spec、`eb16bd129` chunked prefill on L4 pool path）；
`check_build_tracked.sh` **已經把它列為關鍵 exe**。而 llama-bench 用的是**自己的**解析器
（`llama-bench.cpp:514` 的 `--` 迴圈），所以 `--spec-*` 對它是無效參數。

**已跑通（`Backup/run_spec_simple.sh`，env 由 `run_server.sh CGC_DUMP_ENV=1` 取，不手抄）**

| 臂 | 發射溫度 | decode t/s | n_drafted / n_accept | accept |
|---|---|---|---|---|
| `--spec-type` 省略（基線） | NOMINAL | **7.483** | 0 / 0 | — |
| `draft-mtp` n_max=3 | MODERATE | 5.927 | 72 / 26 | 36.1% |
| `draft-mtp` n_max=3 | NOMINAL | 6.517 | 57 / 30 | 52.6% |

⇒ **符號與專案既有結論一致：MTP 現在比 no-spec 慢**（同 NOMINAL 下 7.48 vs 6.52 ⇒ **1.15×**，
在伺服器上量到的是 1.8×，不同儀器不可互引）。**所以「留一條 HTTP 臂專供 MTP」可以升級成
「留一條 `llama-speculative-simple` 臂」** —— 它把伺服器的 sampler／socket／slot 機制移出比較，
且自帶基線臂，是同工具內的 A/B。

**兩個踩到的坑（寫進 runner 的註解）**
1. **`-c 0` 會 OOM**（模型預設 32768；13.66 GiB 模型 ＋ 8 GiB 池撐不住）⇒ 用 profile 的 4096。
2. **脈絡的 CGC 旋鈕是必需的**：一個都不給 → 即使 `-c 4096` 也在 t=16.7 s OOM
   （`kIOGPUCommandBufferCallbackErrorOutOfMemory`）。`CGC_N_CB=8`／`CGC_DBUF`／`CGC_NO_PREFETCH`／
   `CGC_OA_ASYNC` 是記憶體形狀的旋鈕 ⇒ **env 一定要從 `run_server.sh` 取**。
3. `--spec-type none` **不是**基線：它會去載空路徑的 draft model 並 exit 1
   （`failed to load draft model, ''`）。基線是**完全省略** `--spec-type`。

**未做（要才做）**：這三臂是示範不是可引用的 A/B（每個臂一個樣本、n≈48、accept 在兩次 MTP 之間
就從 36.1% 跳到 52.6%）。要可引用得交錯 ×3 ＋ 每臂等讀數回 0（`run_spec_simple.sh` 已印發射前讀數）。
