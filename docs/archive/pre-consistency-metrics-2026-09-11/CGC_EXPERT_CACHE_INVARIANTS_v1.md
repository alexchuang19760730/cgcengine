# CGC Expert Cache §5 不變量契約 (v1, 2026-09-05)

> 本文件定義 CGC Expert Cache 的「不變量 (invariant)」清單 — 也就是系統無論怎麼演化,
> 都**必須**保持的性質。每條不變量對應到:
>   - 守護機制 (V1 hash / V2 logits oracle / V3 sub-buffer alias / always-on FATAL)
>   - 觸發環境變數
>   - 失敗時的 recovery 策略
>   - replay benchmark 的對應指標
>   - 引入守護的 commit hash
>
> 這是 `docs/CGC_EXPERT_CACHE_SOFTPOOL_WHITE_PAPER_2026-09-05.html` §5 的可機器讀版本。
> 任何不變量違反都應該在這裡找到對應條目 + 指針。

---

## 0. 不變量矩陣總覽

| ID | 名稱 | 層 | 守護機制 | 觸發 env | 預設 | 失敗行為 |
|----|------|---|---|---|---|---|
| INV-01 | `file_offset` 算術正確 | storage | V1 SHA-256 pool-fill | `CGC_EXACT_CACHE_VERIFY=1` | OFF | FATAL abort |
| INV-02 | `dst` 指針正確 | storage | V1 SHA-256 pool-fill | `CGC_EXACT_CACHE_VERIFY=1` | OFF | FATAL abort |
| INV-03 | 跨層 `file_offset` aliasing | storage | V1 SHA-256 pool-fill | `CGC_EXACT_CACHE_VERIFY=1` | OFF | FATAL abort |
| INV-04 | logits 與 oracle bit-level 一致 | compute | V2 FNV-1a 64-bit JSONL | `CGC_LOGITS_ORACLE_DUMP=path` | OFF | 工具鏈比對 |
| INV-05 | sub-buffer view 對映正確 | view | V3 SHA-256 reverse-lookup | `CGC_SUBBUFFER_ALIAS_DUMP=path` | OFF (設計中) | FATAL abort |
| INV-06 | Soft Pool L0 容量 | pool | always-on + env tunable | `CGC_SOFT_POOL_L0=N` | 32 | log warning |
| INV-07 | Soft Pool L1 容量 | pool | always-on + env tunable | `CGC_SOFT_POOL_L1=N` | 32 | log warning |
| INV-08 | 掛死看門狗 (確保批 slot 總有解) | scheduling | always-on FATAL | — | ON | FATAL abort |
| INV-09 | 單一可用 slot 都不存在時的 FATAL | scheduling | always-on FATAL | — | ON | FATAL abort |
| INV-10 | `batch_mask` 精確性 (非啟發式) | scheduling | always-on assert | — | ON | assert fail |
| INV-11 | THINK_SEED 條件式 marker 跳過 | prefill | always-on | — | ON | (邏輯分支) |
| INV-12 | coding stop 升級 (```` → ````\n\n) | prefill | always-on | — | ON | (邏輯分支) |
| INV-13 | decode/pool hit rate ≥ 85% (qa-zh) | quality | replay benchmark | (replay) | — | precommit hook fail |

---

## 1. INV-01/02/03: Storage 層 byte-identity (V1)

### 1.1 契約

CGC Expert Cache pool 中的每個 segment byte **必須** 跟對應的 GGUF raw bytes 完全一致。
任何 byte-level 偏差 (無論是 `file_offset` 算錯、`dst` 指錯、還是跨層 aliasing)
都構成 §5 違規, 必須在 production 上線前攔下。

### 1.2 守護機制

**V1 Pool-fill Hash Verify** — `cgc_exact_cache_verify_post_fill`
in `src/llama.cpp/src/llama-expert-cache.cpp:425`

每次 `fill_segments_pool` / `fill_pool_direct` 完成後, 用**獨立 fd** (`::open(..., O_RDONLY)`)
重讀同一個 `(file_idx, file_offset, bytes)` range, byte-compare 對 `dsts[i]`。

獨立 fd 的關鍵: 排除 stdio buffering 在 `cache->files[file_idx]` 上 mask 掉 off-by-N read 的可能
(若用 shared fd, kernel page cache 是對的, 但 stdio buffer 可能還沒 refil)。

### 1.3 三類不變量

| ID | 不變量 | 失敗時印的訊息 |
|----|--------|-----------------|
| INV-01 | `file_offset` 算術正確 | `CGC-EXACT-MISMATCH: file_idx=N file_off=X bytes=Y diff_off=Z` |
| INV-02 | `dst` 指針正確 | 同上 (差異在 segs/dsts 對應) |
| INV-03 | 跨層 `file_offset` aliasing | 同上 (兩個 layer 指向同一段) |

### 1.4 環境變數

| 變數 | 預設 | 說明 |
|------|------|------|
| `CGC_EXACT_CACHE_VERIFY` | (unset = OFF) | 設為 1 開啟 V1 守護 |
| `CGC_EXACT_CACHE_VERIFY_FIRST_N` | 256 | 每個 fill batch 最多驗證的 segment 數 (O(bytes) 會 dominate decode) |

### 1.5 失敗 Recovery

**FATAL abort** — 不可恢復, 必須從錯誤訊息回推:
1. 看 `file_idx` / `file_off` / `bytes` 對應哪個 GGUF tensor
2. 看 `dst` 對應 pool 哪個 slot / layer
3. 比 `pool=0xXX ref=0xXX` 抓具體 byte diff
4. 對 git blame 最近改 `ensure_batch` / `pick_slot` / `fill_segments_pool` 的 commit

### 1.6 Replay 對應指標

| Profile | 指標 | 門檻 |
|---------|------|------|
| 全部 | `decode_tps` 影響 | V1 開啟時 ngl=99 完整 offload 會 −5-15% (sample-bounded 後可接受) |
| 全部 | `peak_rss_mb` 影響 | 無 |

### 1.7 引入 Commit

- **V1 守護**: `1c4e890ee` (2026-09-05) `[CGC FIX v7] Expert Cache bit-correctness gate`
- **FATAL watchdog**: `7a22d8aab` (2026-08-30) `[CGC hang watchdog]`

---

## 2. INV-04: Logits Oracle bit-level (V2)

### 2.1 契約

每個 decode step 的 logits 必須與「**直接從 GGUF 讀, 不走 cache**」產生的 oracle
bit-level 一致 (用 FNV-1a 64-bit 雜湊比對)。argmax token id 必須完全相同, top-N 排名
必須相同。

### 2.2 守護機制

**V2 Logits Oracle Dump** — `cgc_logits_oracle_dump`
in `src/llama.cpp/src/llama-context.cpp:3046`

每個 decode step 把 logits 做:
- 完整 FNV-1a 64-bit 雜湊 (`logits_fnv1a64`)
- per-row 雜湊 (`row_fnv1a64`, 給 partial match 用)
- argmax token id
- top-N token id 排名

寫到 JSONL。

### 2.3 比對工具

`scripts/check/cgc_logits_oracle_compare.py`:
- 對齊 `(step, token_idx)` tuple
- 比 `row_fnv1a64` (per-token) / `logits_fnv1a64` (整體) / `argmax_token` / `top-N`
- 100% 一致才算 bit-correct

### 2.4 環境變數

| 變數 | 預設 | 說明 |
|------|------|------|
| `CGC_LOGITS_ORACLE_DUMP` | (unset = OFF) | 設為 JSONL 路徑才開啟 V2 dump |

### 2.5 失敗 Recovery

**Tool-level diff** — 不 abort, 印出 diff 行, 由人工或 CI 解讀:
- 0 diff → bit-correct
- 只有 1-2 個 token 不一致 → 屬 mild regression, 看 §10.3 契約 + git blame 最近 logits 路徑
- 大量不一致 → 重大 regression, block merge

### 2.6 Replay 對應指標

| Profile | 指標 | 門檻 |
|---------|------|------|
| 全部 | `quality_score` | 0.0 (fail) - 1.0 (pass), 比較 baseline 退化 ≤ 0.1 abs |

### 2.7 引入 Commit

- **V2 dump**: `1c4e890ee` (2026-09-05) `[CGC FIX v7]`
- **V2 compare tool**: `1c4e890ee` (2026-09-05) `cgc_logits_oracle_compare.py`

---

## 3. INV-05: Sub-buffer View Aliasing (V3, 設計階段)

### 3.1 契約

CGC Expert Cache 透過 Metal `MTLBuffer.subBufferWithOffset:length:` 開出的每個
sub-buffer view, 其 `(storage_id, sub_offset, sub_size)` tuple **必須** 對應到該
slot 應有的 expert 權重範圍, 且 `storage[sub_offset : sub_offset + sub_size]`
byte-level 等於 GGUF 該 expert 的 raw bytes。

### 3.2 守護機制 (設計中, 待 commit)

詳見 `docs/CGC_EXPERT_CACHE_SOFTPOOL_WHITE_PAPER_2026-09-05.html` §5.4

### 3.3 環境變數 (設計中)

| 變數 | 預設 | 說明 |
|------|------|------|
| `CGC_SUBBUFFER_ALIAS_DUMP` | (unset = OFF) | 設為 JSONL 路徑才開啟 V3 dump |

---

## 4. INV-06/07: Soft Pool 容量

### 4.1 契約

Soft Pool 容量由 `CGC_SOFT_POOL_L0` (hot tier) + `CGC_SOFT_POOL_L1` (warm LRU tier)
兩個環境變數設定:
- 範圍: 0 - 256 (負值 → 0, 超大值 → clamp 256)
- 預設: 32 + 32 (合計 64 slot)
- 極限配置: 0 + 0 → 走 dev branch 風格 L2 + ZERO-slot

### 4.2 守護機制

**Always-on + env tunable** — `cgc_soft_pool_tier()`
in `src/llama.cpp/src/llama-expert-cache.h:1`

- 讀 env, 缺省/空字串 → fallback
- 解析為 long, 負值 → 0
- 上限 clamp 256

### 4.3 失敗 Recovery

不 abort, 但 `LLAMA_LOG_WARN` 印:
```
[CGC] soft_pool_l0=0 soft_pool_l1=0 → Expert Cache 退化到 L2 + ZERO-slot 路徑, 性能會顯著下降
```

### 4.4 Replay 對應指標

| Profile | 指標 | 預期 |
|---------|------|------|
| longform-zh | `decode_tps` | L0=32 L1=32 vs L0=0 L1=87: 內存省 27%, 速度維持 |
| 全部 | `peak_rss_mb` | 與 soft_pool 容量成正比 |

### 4.5 引入 Commit

- **Soft Pool capacity env**: `b23fb515e` (Hybrid Phase 1) + follow-ups
- **cgc_soft_pool_tier helper**: `2.2.1`

---

## 5. INV-08/09/10: Scheduling 層 (always-on FATAL)

### 5.1 INV-08 契約

`ensure_batch` / `ensure_slot` 必須在 bg fill in flight 時返回 -1 (可恢復),
否則必須 FATAL abort — 不可無聲 hang 死。

### 5.2 INV-09 契約

當 `n_distinct_experts > n_usable_slots` 且 **沒有** fill in flight 時, **必須** FATAL
而非 silently 用 stale slot。

### 5.3 INV-10 契約

`pick_slot` 的 `batch_mask` 必須是精確的「已被 batch 佔用 slot 集合」, 不可用
啟發式近似 (例如 `slot_loading || slot_queued` 不夠, 還要包含已被 assign 但還
沒 dequeue 的)。

### 5.4 失敗 Recovery

**FATAL abort** — 立即進看門狗。log 會印:
- `ensure_batch layer=N: N distinct experts exceed the M usable pool slots and no fill is in flight — cannot assign; aborting`
- `ensure_slot layer=N: no usable slot and no fill in flight — cannot assign; aborting`

### 5.5 Replay 對應指標

| Profile | 指標 | 預期 |
|---------|------|------|
| 全部 | (無 — 這是 hang 防護, 不影響 happy path 指標) |
| 全部 | CI smoke test | 必須在 60s 內產出 token, 否則判定 hang |

### 5.6 引入 Commit

- **Hang watchdog**: `7a22d8aab` (2026-08-30)
- **batch_mask 精確性**: `c13d0f3e5` (2026-08-29) routing-aware placement

---

## 6. INV-11/12: Prefill 語意層 (always-on)

### 6.1 INV-11: THINK_SEED 條件式 marker 跳過 (v6 fix)

**契約**: 當 prefill 結尾是結構化 marker (換行、反引號、JSON bracket 等) 時,
**必須** 跳過 `THINK_SEED` 注入。否則會觸發:
- coding 場景 1-token stop
- longform 場景迴圈

**修法**: `wants_think_scaffold_seed` helper 偵測 prefill_tail_is_structured_marker
→ 跳過注入

**守護機制**: always-on, 無 env 開關 (屬邏輯分支)

**失敗 Recovery**: 重新套用 v6 fix (`ed03b5b2a`)

**引入 Commit**: `ed03b5b2a` (2026-09-05) `[CGC FIX v6]`

### 6.2 INV-12: coding stop 升級 (v7 fix)

**契約**: 偵測到 code fence open (e.g. ```` ```python\n ````) 時, **必須** 把 stop 列表中的
```` ``` ```` 替換為 ```` ```\n\n ````。否則 Qwen3-IQ3-MTP 會在第一個 code block 結尾
立刻觸發 1-token stop。

**修法**: `cgc_patch_coding_stop` helper in `server-common.cpp`, 應用於 completion 與 chat 插入點

**守護機制**: always-on, 無 env 開關

**失敗 Recovery**: 重新套用 v7 fix (`7bc67c31a` / `1c4e890ee`)

**引入 Commit**:
- `7bc67c31a` (2026-09-05) `[CGC FIX v5]` (has_assistant_prefill)
- `1c4e890ee` (2026-09-05) `[CGC FIX v7]` (cgc_patch_coding_stop)

---

## 7. INV-13: Quality 層 (replay benchmark)

### 7.1 契約

3 profile × 3 aspect (quality / decode_tps / peak_rss_mb) = 9 指標, 跟 baseline 比較:
- 至少 1 個指標改善
- 不超過 1 個指標退化 (>10% decode, >15% memory, >0.1 abs quality)
- 否則 precommit hook 拒絕 commit

### 7.2 守護機制

`scripts/check/replay_bench_compare.py`:
- 讀 current vs baseline
- 套用 `replay_bench_reference.json` 的 thresholds
- 出 verdict (PASS / FAIL)

### 7.3 環境變數

| 變數 | 預設 | 說明 |
|------|------|------|
| `ALLOW_REPLAY_BENCH_BASELINE` | 0 | bootstrap 用, 跳過 baseline 比較 |
| `ALLOW_STALE_BIN` | 0 | 跳過 binary sync 檢查 |
| `ALLOW_ENV_GATED_BIN` | 0 | 跳過 env-gated no-op commit 檢查 (dylib byte-identical) |

### 7.4 失敗 Recovery

Hook 印出:
- `n_regression` / `n_improvement` 計數
- 哪個指標 (decode_tps / prefill_tps / peak_rss_mb / quality_score) 退化
- 比對 baseline commit hash

Reviewer 自行判斷:
- 退化是否在 noise floor (dylib swap ±5-15%)
- 是否有正當理由 (例如換 dylib 是必要的)

### 7.5 Replay 對應指標

| Profile | 指標 | 預期 |
|---------|------|------|
| qa-zh | decode_tps ≥ 13, quality ≥ 0.5 | 短 prompt, draft_accept 較低 |
| longform-zh | decode_tps ≥ 25, quality ≥ 0.33 | 長 prompt, MTP fast path |
| coding | decode_tps ≥ 25, quality ≥ 0.67 | coding stop fix 後 |

### 7.6 引入 Commit

- **Replay gate 整合**: `fa2de777b` (2026-09-05)
- **Hook 修正 (chmod + 中文括號)**: `19ade41b2` (2026-09-05)
- **Hook 8a/11 強化 (ALLOW_ENV_GATED_BIN + staged baseline)**: `1c4e890ee` (2026-09-05)
- **新基線 (dylib 0.0.140 noise floor)**: `906e6b90b` (2026-09-05)

---

## 8. 維護指南

### 8.1 新增不變量

新增一條不變量時, 必須:
1. 在本文件 §0 矩陣加一行 (給機器讀)
2. 在對應層加一節 (給人讀)
3. commit 引用 contract section: `[CGC INV-XX] <description>`
4. 配套 replay benchmark 指標 (如果適用)
5. 配套 env var (如果 env-gated)
6. 配套 FATAL / log / tool diff (失敗行為)

### 8.2 變更既有不變量

變更任何一條不變量, 必須:
1. 在 contract doc 更新矩陣 + 章節
2. 在 commit msg 引用 INV-XX
3. 重新跑 replay benchmark 確認沒退化
4. 重新跑 V1/V2 (如果適用) 確認守護本身正確

### 8.3 失效模式 (Failsafe)

如果 contract doc 跟程式碼不一致, 程式碼為準 — 但必須發 issue / 修文件。
如果 replay benchmark 跟手動量測不一致, 確認是 dylib noise floor 還是邏輯退化。

---

## 9. Cross-Reference

- **白皮書**: `docs/CGC_EXPERT_CACHE_SOFTPOOL_WHITE_PAPER_2026-09-05.html`
- **Hybrid 設計**: `docs/CGC_EXPERT_CACHE_HYBRID_DESIGN_2026-09-05.html`
- **CI gate workflow**: `.github/workflows/cgc-replay-bench.yml`
- **CI gate script**: `scripts/ci/cgc_replay_ci.sh`
- **Replay 工具**: `scripts/check/replay_bench_compare.py` + `replay_server_profile.py`
- **V2 比較工具**: `scripts/check/cgc_logits_oracle_compare.py`
- **Pre-commit hook**: `scripts/check_build_tracked.sh` (Hook 8/11)

---

## 10. 版本歷史

- **v1** (2026-09-05): 初版, 涵蓋 V1/V2 守護 + v6/v7 fix + Soft Pool + replay gate
  13 條不變量, 全部對應到目前已進版的 commit
