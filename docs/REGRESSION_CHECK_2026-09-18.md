# 回歸檢查：現在的 prefill/decode vs 上一個 commit（2026-09-18）

**結論：「必須高於上一個 commit」這個要求在本輪不可能由程式成立——引擎自 `79f1b5efc` 起逐位元未變。
四格數字確實全部更高，但同一顆 binary 就已量到 2.30× 的差 ⇒ 那是機器，不是程式。**
可交付的述句是「**沒有退步**，且引用 `launch`／`worst` 皆 `NOMINAL` 的那兩格」。

## 1. 為什麼不可能是程式

| 檢查 | 結果 |
|---|---|
| `git log -1 -- src/llama.cpp/src src/llama.cpp/ggml/src` | `79f1b5efc`（13:30:19，node 命名） |
| 其後三個 commit 的 `src/` 檔數 | `44ab38317` 0／`5078242be` 0／`726fe6823` 0 |
| 本輪 `git status` 的 `src/` 檔數 | **0** |
| 命名 commit 是否已編入 binary | `strings libllama.0.0.279.dylib \| grep -c dnqkv_proj` = **1** |
| 有沒有 `src/*.cpp`／`*.metal` 比 `llama-server` 新 | **沒有** |

⇒ **現在的 binary ≡ HEAD 的引擎**，沒有可归因的加速或退步。

（順帶更正一個過期前提：`scripts/run_server.sh` 的 `CGC_GRPH_DBG` allowlist、`scripts/check/decode_sweep.py`
的 `en-grph` 臂、`docs/LAYER_NODE_CENSUS_2026-09-18.md` **都已經在 HEAD 裡**（`git show HEAD:<f>` 逐檔驗過）。
未 commit 的只剩本輪的 7 個路徑：`decode_sweep.py`（兩臂）、`scripts/check/mul_mv_surface.py`（新）、
`agent_harness/engine_loop/index_assets.py`、`MANIFEST.jsonl`、`INDEX.jsonl`、兩份新 `docs/`，
**0 個 `src/`、0 個引擎效果**。）

## 2. 數字（同一顆 build `efeade7e2`、同一 profile `prefill250`、同一支 `prod_matrix.py`）

| cell | 13:37–13:45（＝上一個 commit 的紀錄） | 14:48–14:52（現在） | 比 |
|---|---|---|---|
| `decode`（`-p 0 -n 128 -d 512 -b 512`） | 5.61（launch HEAVY） | **10.61（launch NOMINAL／worst NOMINAL）** | 1.89× |
| `decode-up`（`-p 0 -n 128 -d 512 -b 2048`） | 5.32（launch HEAVY） | **10.72（launch NOMINAL／worst NOMINAL）** | 2.02× |
| `prefill-house`（`-p 2048 -n 16 -b 5632`） | 116.94（launch HEAVY） | **268.51（launch NOMINAL）** | 2.30× |
| `prefill-up`（`-p 512 -n 128 -b 2048`） | 72.42（launch MODERATE） | 117.22（launch HEAVY） | 1.62× |

**最乾淨的單一證據**：`prefill-house` 同一顆 binary ——
13:37 = **116.94**（HEAVY）→ 14:30 = **234.24**（NOMINAL）→ 14:50 = **268.51**（NOMINAL）。
`run_server.sh:490-496` 自己寫著「同指紋可差到 2×」；這次量到 **2.30×**。

## 3. 「沒有退步」的正確對照：跨日、同機器品質

| 來源 | decode（`tg128 @ d512`） | 標記 |
|---|---|---|
| 09-17 紀錄（d512 暖平台） | 10.78 / 10.91 | 暖平台 |
| **09-18 現在** | **10.61 / 10.72** | **`launch` 與 `worst` 都是 `NOMINAL`** |

⇒ 差 **−1.6% / −1.7%**，在雜訊內；而且**今天這兩格是紀錄上標記最乾淨的 decode 數字**。

## 4. D5（不要用結構論證取代它）

`python3 scripts/check/m123_oracle_gate.py --tag en-mmflip-mulmv-surface-20260918`（60 s）：

```
comparable = True     config_diffs = []
M1 numeric identity 9/9    M2 decision agreement 9/9    M3 top-k set 9/9
cross-tab {'num_eq_dec_eq': 9, 'num_eq_dec_ne': 0, 'num_ne_dec_eq': 0, 'num_ne_dec_ne': 0}
ref = Backup/knifeedge_matrix/ref_iq3_pool8gb_M2_6144_bitident_v6_nbaware.jsonl
⇒ PASS
```

skill `cgc-commit-gate` §2.4 的實務判準：D5 的**數值那一半**看「有沒有動 `src/`」（本輪 0，技術上可省），
但同一節明寫「**跑它只花 23 秒**，而省下 23 秒再寫一段話解釋為何不跑，是這個 repo 已判為錯的做法
（`eng-gate-0016`）」⇒ 跑了就要寫出來。**白皮書那一半沒有豁免**；且 §2.4 ★ 說「同一問題、同一支儀器的
連續推進」時正法是**就地追加節**到既有那份白皮書（`docs/GPU_NODE_ATTRIBUTION_20260918_0126.html`，
§23–§35 之後接 §36–），並在 message 寫明「修訂既有白皮書（追加 §X–§Y）」，**不要出新檔**。

pre-commit hook（`.git/hooks/pre-commit` → `scripts/check_build_tracked.sh`）**本身不含 D5**；
它的 check 8 在無 staged `src/` 時直接 `pass "8 無 llama 原始碼變更（僅 doc/腳本/產物）"`。

## 5. 產物

- `Backup/phase_decomp/prod_matrix_20260918_1448.json`、`docs/PROD_MATRIX_RECHECK_2026-09-18.md`
- `Backup/m123_oracle_gate/summary_en-mmflip-mulmv-surface-20260918.json`
- 逐格 llama-bench 原始輸出：`Backup/prod_matrix/20260918_1448/`
- 對照組（13:37）：`Backup/phase_decomp/prod_matrix_20260918.json`、`docs/PROD_MATRIX_2026-09-18.md`
