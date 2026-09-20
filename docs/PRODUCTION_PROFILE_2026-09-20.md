# 生產 profile（凍結版，2026-09-20）

## 0. 一句話

**profile ＝ `prefill250`。** 它**已經**同時是最高的 prefill 與目前最好的 decode —— 這一點
不是推論，是零 GPU 逐旋鈕比對出來的（§3）。這份文件把它**釘住**，並補上**唯一缺的那件東西**：
一支能在**交付口徑**下重跑兩軸、且**讀數可歸屬**的儀器。

**交付 decode 的現行基線＝ 12.57 t/s**（`NOMINAL` 全程，2026-09-20 12:30，見 §5）。
**這是「另一個 agent 要打敗的數字」。** 而它的單臂噪音底約 **±27%**（§6）—— 小於這個量級的
優化，單臂量不出來。

---

## 1. 這個 profile 是什麼（可機檢，不是抄的）

```sh
# 旋鈕的真相來源只有一個：run_server.sh。這條命令把它的解析結果印出來。
python3 scripts/check/prod_profile.py --emit-spec
```

| | 值 |
|---|---|
| `CGC_SERVER_PROFILE` | **`prefill250`** |
| 模型 | `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`（13.66 GiB） |
| `CTX` / `BATCH` / `UBATCH` | 8192 / **5632** / **5632** |
| `CGC_EXPERT_CACHE_BYTES` | 8589934592（8 GiB） |
| 三支柱（bit-identical） | `CGC_MM_BITIDENT=1`、`CGC_MTP_NO_WARMUP=1`、`CGC_NO_SEQ_RM_PROBE=1` |
| 池幾何 | `LLAMA_EXPERT_CACHE_LAYER_CAPS=40-40:256`、`WORKERS=8`、`ALLOW_NGL=1` |

解析後的引擎 env（**28 個旋鈕，逐字**）：

```
CGC_DBUF                            1        CGC_MM_BITIDENT                     1
CGC_DRAFT_DECODE                    1        CGC_MTP_NO_WARMUP                   1
CGC_EVICTED_RING                    0        CGC_NO_PREFETCH                     1
CGC_EXPERT_CACHE_BYTES              8589934592  CGC_NO_SEQ_RM_PROBE               1
CGC_GATHER_SLAB_CAP                 256      CGC_N_CB                            8
CGC_GLU_FUSED_DOWN                  1        CGC_OA_ASYNC                        1
CGC_LOOP_GUARD                      1        CGC_PREFILL_STREAM                  1
CGC_SERVER_AUTO_ANCHOR              0        CGC_SERVER_DEFAULT_MARKER_STOPS     1
CGC_SOFT_POOL_L0                    0        CGC_SOFT_POOL_L1                    0
CGC_SPAC                            1        CGC_SPAC_ALPHA                      0.75
CGC_VERIFY_DECODE                   1        CGC_WAKE_POLL_US                    15
CGC_WARM_NPAST                      0        CGC_WATCHDOG                        1
LLAMA_EXPERT_CACHE_ALLOW_NGL        1        LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0   0
LLAMA_EXPERT_CACHE_LAYER_CAPS       40-40:256  LLAMA_EXPERT_CACHE_WORKERS         8
```

> **這一節刻意不是一份 repo 內的 JSON。** 旋鈕的第二份拷貝就是第二個權威來源（D6 要防的事）；
> 需要機器可讀的版本時跑 `--emit-spec` 當場重生，它讀的是 `run_server.sh` 本身。
> 原始讀數在 `Backup/prod_profile/`（`Backup/` 依 `.gitignore:396` 不進版控，與既有慣例一致）。

---

## 2. 為什麼不新增一個 profile 名字

候選做法是把「prefill250 ＋ 最好的 decode」做成 `run_server.sh` 裡的新分支。**沒有做**，理由有兩條：

1. **它會是純重複** —— 見 §3，兩者的解析後 env 逐旋鈕相同。
2. **它會靜默改變別的 session 的量測**：`prod_matrix.py` 的 `--profiles` 預設是 **`all`**，而 profile
   清單是從 `run_server.sh` 的拒絕訊息**解析**出來的（`profiles_from_run_server()`，`:179`）⇒
   加一個分支會讓這台機器上**每一支** `prod_matrix --profiles all`（含另一條線正在跑的）變長並多一列。

「profile 不寫的那一格，就是會漂移的那一格」（`run_server.sh:290` 的既有原則）在這裡由**這份記錄
＋ 一支會印出解析結果的儀器**滿足，而不是靠第二個名字。

---

## 3. ★ 「最好的 decode」是什麼，以及它為什麼不是一個 profile

記錄裡最好的 decode 讀數是 **13.10**（09-19 16:58）與 **10.94**（09-19 15:19），兩個都 `launch` 與
`worst` 皆 `NOMINAL`（`Backup/phase_decomp/{carrier2_20260919/C,warmskip2_20260919/B}.json`）。

它們用的不是一個 profile，而是：

| 元件 | 值 | 從哪來 |
|---|---|---|
| profile | **`prod25`** | `run_server.sh` |
| arm override | `CGC_PREFILL_STREAM=1`、`CGC_GATHER_SLAB_CAP=256` | 臂 `prod25-stream`（`llama_bench_matrix.py:84`） |
| bench 側四約定 | `--batch 512`、`--ctx-size 4096`、`--warm-skip 64`、`--spec-type draft-mtp` | 見 §4 |

### 3.1 零 GPU 的同一性證明

```sh
python3 - <<'EOF'
import sys; sys.path.insert(0,'scripts/check')
from llama_bench_matrix import resolve
a = resolve('prefill250', {})
b = resolve('prod25', {"CGC_PREFILL_STREAM":"1","CGC_GATHER_SLAB_CAP":"256"})
print({k:(a['env'].get(k),b['env'].get(k)) for k in set(a['env'])|set(b['env']) if a['env'].get(k)!=b['env'].get(k)})
EOF
# -> {}   （解析後的 env 差異＝空）
```

差異只出現在三處，**而三處都被 decode cell 覆寫**：

| | `prefill250` | `prod25-stream` | bench 的 decode cell |
|---|---|---|---|
| `BATCH/UBATCH` | 5632 / 5632 | 模型預設 | **覆寫成 512** |
| `CTX` | 8192 | 4096 | **覆寫成 4096** |
| `--chat-template-kwargs` | 有（chat-ab prefix） | 無 | 不轉發（`FORWARD_*` 白名單） |

⇒ **`prefill250` 就是那個組合。** 統一 profile 在 09-16 就已經存在；09-20 缺的不是旋鈕，是
**記錄**與**能重現交付口徑的儀器**。

### 3.2 工具缺口（這是本輪真正的發現）

`profile_duo.py` 與 `prod_matrix.py` 的 `decode` cell **表達不了交付形狀**：它不帶 `--spec-type`
（⇒ MTP off）、不帶 `--warm-skip`（⇒ 冷的時鐘）、不帶 `--ctx-size`（⇒ llama-bench 自行推導的 ~704）。
同一個 profile 在那兩支工具裡因此只讀 **7.96–9.12**，比交付低約 **1.4×**。

**「profile 慢」與「cell 不是交付形狀」是兩件事**，而在 09-20 之前它們沒有被分開寫下來。

---

## 4. 交付 decode 的四個約定

| 約定 | 值 | 為什麼不能省 |
|---|---|---|
| `--batch` | 512 | `decode` cell 的既定值；`-b` 大於池上限時 `compat()` 會拒絕 |
| `--ctx-size` | **4096** | llama-bench 會自行推導 `n_ctx = p+n+d ≈ 704`，而生產 server 跑在 4096；KV 配置與 batch 夾制都吃這個 |
| `--warm-skip` | **64** | 時鐘在前 64 個 token 之後才起算，報告的 `n_gen` 是 **64**（排除掉的 64 不平均進去）。沒有它，`-n 128` 讀 7.96、`-n 512` 讀 10.22 —— 同一個引擎兩個數 |
| `--spec-type` | `draft-mtp` | 開啟 MTP verify 回合（M=1..4），即服務路徑真正付的那一段。`--spec-draft-n-max` 由 profile 的 argv 決定（此處 3） |

**四個全在，那一列才是交付 decode；任缺一個，它就不是。**

---

## 5. 2026-09-20 的讀數（同一次 session、同一顆 build）

儀器：`python3 scripts/check/prod_profile.py --profile prefill250 --reps 3`
（三個 launch：prefill 軸、decode 交付軸、decode 錨點臂；**每個 launch 前等 NOMINAL，逾時即拒絕發射**）

| 軸 | 臂 | t/s | ± | `n_batch` | thermal launch / worst | 判決 |
|---|---|---:|---:|---:|---|---|
| `decode-delivery` | **`prod25-stream`**（錨點） | **12.57** | 2.26 | 512 | `NOMINAL` / **`NOMINAL`** | ✅ **PASS**（bar 12） |
| `decode-delivery` | `prefill250` | **9.90** | 1.25 | 512 | `NOMINAL` / `MODERATE` | ❌ 臂內升溫 |
| `prefill-house` | `prefill250` | **212.59** | 4.65 | 5632 | `NOMINAL` / `HEAVY` | ❌ 臂內升溫 |
| `prefill-house`（單獨重跑） | `prefill250` | **222.40** | 22.91 | 5632 | `NOMINAL` / `MODERATE` | ❌ 臂內升溫 |

歷史參照（**不同日期，只能當參照**）：decode **13.10 / 10.94**（09-19，皆 NOMINAL 全程）；
prefill **292.34**（09-18，NOMINAL 全程；`Backup/prod_matrix/duo_prefill250_20260918_1615.json`）。

### 5.1 ★ 兩條 decode 臂是**同一個命令**

把兩條臂的 `cmd` 逐字比對（去掉 `--arms` 的名字）：**完全相同**。

```
... --reps 3 --prompt 0 --gen 128 --depths 512 --batch 512 \
    --ctx-size 4096 --warm-skip 64 --spec-type draft-mtp
```

`prefill250` 的解析後 env 本身就含 `CGC_PREFILL_STREAM=1`／`CGC_GATHER_SLAB_CAP=256`（所以它的
`extra_env` 印出來是空的），錨點臂是把它當 override 傳 —— **淨結果同一個 env**。

⇒ **12.57 就是這個 profile 的交付 decode 讀數**，不是「只屬於錨點臂」的數字。錨點臂在這裡的作用
不是提供一個額外的量，而是**證明交付口徑在 09-20 重現了**（落在 10.94–13.10 的帶內）。

### 5.2 prefill 的 250 bar：**今天無法驗證**

三次讀數 **188.12 / 212.59 / 222.40**，全部 `worst ≥ MODERATE`，全部 < 250。
而 09-16 的記錄（`docs/PREFILL250_CONDITIONAL_DELIVERY_20260916.html`）寫的判準是
**「Nominal 6/6 全部 ≥250（最低 253.42）；非 Nominal 0/21（最高 211.65）」** ——
本日三次都落在**非 Nominal** 那一側。

⇒ **不是「prefill 退步了」，是「這台機器今天撐不住一整條 prefill 臂」**（`-r 3` × 2048 token ×
`-b 5632`，無風扇）。這一格**不可引用**，而「未驗證」必須寫成未驗證。

---

## 6. ★ 交付 decode cell 的單臂噪音 ≈ **±27%**

同一個命令、同一次 session、同一顆 build：**9.90（`worst=MODERATE`）vs 12.57（`worst=NOMINAL`）**。
更低的那一次是更熱的那一次 —— 與 09-18 的既有發現同向（四臂全部 `launch=NOMINAL` 卻給出
10.80 / 10.34 / 8.92 / 8.58，排序跟著「臂內多熱」而不是受試引擎）。

**對下一個 agent 的直接含意**：

- 單臂前後對比**無法**證明小於 ~27% 的效應；
- 要證明小效應，唯一的路是**配對的交錯 A/B（AB/BA ＋ `median(A/B)`）**，而第一步永遠是 `--null`
  （兩槽同 binary）——它量的是儀器自己的底。
- 另一條路是**降低每臂的發熱**（本日三條 prefill 臂的教訓），或把兩軸拆成獨立的 session。

---

## 7. 怎麼重現（一條命令）

```sh
# 兩軸 ＋ 錨點臂；每個 launch 前等 NOMINAL，逾時即拒絕；寫 JSON 與 markdown
python3 scripts/check/prod_profile.py --profile prefill250 --reps 3 \
  --json Backup/prod_profile/prod_profile_<YYYYMMDD_HHMM>.json \
  --md   Backup/prod_profile/prod_profile_<YYYYMMDD_HHMM>.md

# 只要一軸（本日 prefill 的教訓：連跑會把機器加熱）
python3 scripts/check/prod_profile.py --axes prefill --no-ref
python3 scripts/check/prod_profile.py --axes decode  --no-ref

# 零 GPU：旋鈕 / 命令列
python3 scripts/check/prod_profile.py --emit-spec
python3 scripts/check/prod_profile.py --dry-run
```

儀器性質（都寫在 docstring 裡，這裡只列要點）：

- **重用而非複製**：`thermal_pressure`（同一個 11 ms 讀數）、`prefill_certifiability.mem_state`、
  `llama_bench_matrix.resolve`（旋鈕的唯一真相來源）、`profile_duo.wait_nominal`。
- 讀數**從子進程自己的 `--json` 回讀**，不另寫解析器。
- 兩道閘門：發射前 NOMINAL ＋ `usable ≥ 30%`；**逾時＝拒絕發射**（與 `profile_duo` 同語意）。
- `worst` 不是 `launch`：`clean = launch==NOMINAL and worst==NOMINAL`，只有 `clean` 才算達 bar。

⚠ **已知的不足**：`wait_nominal` 可以在上一次重臂之後 **立即回 0（waited 0s）**，而底盤仍然是熱的
—— 本日第三次 prefill 重跑就是「waited 0s 之後 worst=MODERATE」。⇒ **NOMINAL 是必要條件，不是充分條件**；
若要重跑一個大臂，中間的空閒要按**分鐘**計，不能靠這個讀數自己回到 0。

---

## 8. 不要做什麼（省得重走）

- **不要**把 §5 的 212.59／222.40 當 prefill 交付數字 —— 它們是**非 Nominal 側**的讀數。
- **不要**把 decode 的 12.57 與 prefill 的任一個相乘或相除去推 joint 吞吐。
- **不要**在兩台儀器之間換算（引擎內部步時 ↔ llama-bench `avg_ts`）：記錄寫明**校準不是常數**
  （實測差 +13~30%）。
- **不要**為了「讓數字好看」把 `CGC_S1_TABLE_CHURN` 之類的診斷旋鈕塞進交付 cell（`DIAGNOSTIC_KEYS`
  的每一條都是一個主張）。
- **不要**把 `prefill250` 的 `CTX=8192` 當成交付 ctx；交付 server 跑 **4096**，bench 也覆寫成 4096。

---

## 9. 產物與歸屬

| | |
|---|---|
| 新儀器 | `scripts/check/prod_profile.py`（本輪新增；**未動任何共享工具**） |
| 原始讀數 | `Backup/prod_profile/prod_profile_20260920_{1230,1240_prefill}.json/.md`、`prefill_axis_hotrun_20260920_1225.json`（**`Backup/` 不進版控**） |
| build 指紋 | `libllama=cfd550a18638c9ec`、`libggml-base=6fe5b3e0abff3079`、`libggml-metal=1be366306c604669`、`llama-bench=28ae5feed73660da`、`llama-server=054fb22f04a01c5c`、`build_commit=efeade7e2`（#279） |
| 發射時記憶體 | decode 臂 `usable 59.8%`（`cached 1.57 GiB`）／錨點臂 `67.4%`（`cached 0.73`）／prefill 臂 `42.7%`、`41.3%`（`cached 2.22`、`3.71`） |
| 現場 | 8080 空、無 `llama-*` 行程（每一次 launch 前都查過）；**另一條線本日在同一台機器上跑了 F2–F5 的三次啟動**，其熱殘留未量 |

**越界揭露**：`agent_harness/portal/targets.json` 的 `G1.owner` 寫的是
「line B (implementation); **line A (ace) owns the ceiling reading**」。本輪凍結 profile ＋ 量兩軸
**落在 ceiling reading 那一側**，沒有改任何 `src/`、沒有改 `run_server.sh`、沒有改任何共享工具。
