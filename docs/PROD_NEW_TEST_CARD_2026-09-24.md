# prod-new 標準測試卡（2026-09-24）

> **性質**：llama-bench 完整側的統一測試口徑。**所有速度 runner（commit_bench、llama_bench_matrix、A/B）都以此為準**——commit 標題帶的 prefill/decode 成績、以及跨時間比較的數字，都出自這張卡。
> **為什麼需要**：decode 10-14 之間任何跨時間比較都是環境噪音（launch-to-launch ±20%、swap 只累積不回收、thermal 節流），沒有統一形狀 + 歸因行，數字不可比（§EN-473 / ABBA 協議）。

---

## 1. 硬體與模型

| 項 | 值 |
|---|---|
| 機器 | MacBook Air M4 / 16GB（Apple Silicon） |
| 模型 | `models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`（13030 MiB） |
| 結構 | 40 trunk + 1 MTP head / 256 experts / 8 active / MoE IQ3_XXS + dense IQ4X |
| build 指紋 | server=`054fb22f04a0`、libllama-server-impl=`60fb7910a8bb`（每次測試記實際 digest） |

## 2. llama-bench 命令（shape 完整側）

```
llama-bench -m models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf \
  -ngl 99 --load-mode none -t 8 -expert-cache 8589934592 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -b 5632 -ub 5632 -p 2048 -n 128 -d 512 -r 3 -o json \
  --warm-skip 64 --fixed-fill-seed 1
```

| 參數 | 值 | 意義 / 依據 |
|---|---|---|
| `-p 2048` | prefill 2048 token | 完整側 prefill 維 |
| `-n 128` | decode 128 token | 完整側 decode 維 |
| `-d 512` | depth 512 | 已入 context 的 token 數 |
| `-r 3` | 3 reps | 每臂 3 次（中位數口徑） |
| `-b / -ub` | 5632 | 最大存活 chunk（6144 OOM 0/5，見 prefill250 段） |
| `-ngl 99 --load-mode none` | 全層 GPU、不 mmap | |
| `-expert-cache 8589934592` | 8 GiB pool | 縮 pool 確定性降速（8G 12.06 → 6G 10.48 → 4G 7.67），故不縮 |
| `--warm-skip 64` | 時鐘在池預熱後才起算 | 沒它就是「冷啟動含在內」的另一個 cell |
| `--fixed-fill-seed 1` | 每個 rep 重播同一條填充流 | `docs/DECODE_STEADY_BASELINE_2026-09-19.md:45`：σ 3.46→1.09。設 0 ⇒ 每 rep 換一條 ⇒量到 cold/steady 混樣（run3 = 4.13/11.85/11.67），那個混樣會被讀成「噪音」 |

### 2.5 machine-readable CELL（driver 唯一讀取處：人讀 §2 命令、程式讀本塊）

> 下面 JSON 是所有 driver（harness bench / commit_bench / llama_bench_matrix）組命令的**唯一齣處**。
> driver 實際組出的命令與本塊不一致 ⇒ **fail-closed 拒跑**。臂專用開關（§4）在 `switches` 的 default 之上覆寫、並把覆寫記錄在產物；`provenance_required` 欄位必須出現（值可為 null，但不能缺席）。

```json
{
  "schema": "prod-new-cell/v1",
  "model": "models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf",
  "cell": {
    "ngl": 99,
    "load_mode": "none",
    "threads": 8,
    "batch": 5632,
    "ubatch": 5632,
    "prompt": 2048,
    "gen": 128,
    "depths": 512,
    "reps": 3,
    "warm_skip": 64,
    "ctx_size": 0,
    "expert_cache_bytes": 8589934592,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "fixed_fill_seed": 1
  },
  "switches": {
    "LLAMA_EXPERT_CACHE_ALLOW_NGL": {
      "default": 1,
      "values": [0, 1],
      "role": "ngl>0 下 expert skip-load / L4 pool 的硬使能：expert_cache_skip_load = (ngl<=0 || ALLOW_NGL || L3_NGL) && !NOGATHER",
      "prerequisite_for": ["CGC_EXPERT_SKIP_READRAW"]
    },
    "CGC_EXPERT_SKIP_READRAW": {
      "stage": "P0",
      "default": 1,
      "enforced": "run_server.sh cgc_swap_guard：未武裝且未用 CGC_SWAP_GUARD=off 宣告 ⇒ 拒跑（exit 2）",
      "values": [0, 1],
      "requires": "LLAMA_EXPERT_CACHE_ALLOW_NGL=1",
      "effect": "skip-load expert 不 read_raw，省 ~10.9 GiB 匿名駐留、swap 增量 -92%（5679→449 MiB）"
    },
    "CGC_POOL_MADVISE": {
      "default": 2,
      "enforced": "同上（cgc_swap_guard，同一道閘）",
      "values": [0, 1, 2],
      "stage_map": {
        "1": "P1：fill（pread）前 madvise(DONTNEED) 丟將被覆蓋頁",
        "2": "P1+P2：fill 前 + evict 時都丟"
      },
      "effect": "減少駐留與重讀"
    },
    "CGC_B_SCHEME": {
      "default": 1,
      "enforced": "同上（cgc_swap_guard；run_server.sh SERVER_ENV 白名單已接，沒列會被靜默丟）",
      "values": [0, 1],
      "effect": "熱門優先替換：不踢錯熱專家，讓 miss 收斂"
    }
  },
  "runtime_adjustable": ["ctx_size", "fixed_fill_seed"],
  "provenance_required": ["engine_md5", "swap_before", "swap_after", "thermal_launch", "thermal_worst", "fixed_fill_seed"],
  "derived": {
    "warm_skip_applied": "產物 n_gen 應 == cell.gen - cell.warm_skip（=64）；不符即「名義有、實際沒有」，fail-closed"
  }
}
```

## 3. prod-new profile env（默認，顯式 env 永遠贏）

| env | 值 | 角色 |
|---|---|---|
| `CGC_SERVER_MTP` | **0** | decode 支柱（MTP off：省 draft 鏈 + verify batch，同模型實測 13-14 t/s；MTP on 走 prod25-stream 血統） |
| `CGC_SERVER_DENSE_IQ4X` | 1 | dense 走 IQ4X |
| `CGC_SERVER_OA_ASYNC` | 1 | 異步 |
| `CGC_SPAC` / `CGC_SPAC_ALPHA` | 1 / 0.75 | 專家緩存替換策略 |
| `CGC_MM_BITIDENT` | 1 | bit-identical 支柱 |
| `CGC_SERVER_NO_SEQ_RM_PROBE` | 1 | |
| `CGC_SERVER_PREFIX_REUSE_CKPT` | 1 | prefix reuse（MTP off 下 no-op，保留為顯式一致） |
| `LLAMA_EXPERT_CACHE_ALLOW_NGL` | **1** | **通用 SERVER_ENV（run_server.sh:1429，所有 profile 都帶）——L4/skip-load 使能條件**：`expert_cache_skip_load = (ngl<=0 || ALLOW_NGL || cgc_l3_ngl) && !no_gather`。**P0（skip-readraw）要生效，這條必須=1**（值語意，`0` 是關，讀者解析值不測存在） |
| `CGC_EXPERT_SKIP_READRAW` | **1** | P0（**預設開**；砍 ~10.9 GiB 匿名副本＝超訂根因，開了之後 8 GiB pool 才裝得進 16 GB。要跑關掉的 A/B：`CGC_SWAP_GUARD=off CGC_EXPERT_SKIP_READRAW=0`） |
| `CGC_POOL_MADVISE` | **2** | P1+P2（**預設開**） |
| `CGC_B_SCHEME` | **1** | 熱門優先替換（**預設開**；`SERVER_ENV` 白名單 2026-09-25 才接上——沒接之前設了會被靜默丟） |
| `CGC_PREFILL_STREAM` | 1 | prefill250 支柱（大 chunk 走 whole-layer slab） |
| `CGC_GATHER_SLAB_CAP` | 256 | slab 裝得下全部 256 experts |
| `CGC_SERVER_MTP_N_MAX` | 3 | MTP on 時 n_max 顯式釘 3 |
| `SERVER_BATCH` / `SERVER_UBATCH` | 5632 | |
| `CTX` | 8192 | |
| `CGC_EXPERT_CACHE_BYTES`（內部 `BUDGET`） | 8589934592 | 8 GiB pool |
| server 參數 | `-t 8 --temp 0.4` | |

### 3b. 其餘通用 SERVER_ENV 默認（run_server.sh SERVER_ENV 段，prod-new 未覆寫）

| env | 值 | 角色 |
|---|---|---|
| `LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0` | 0 | layer0 在 pool（40 層全 pool；=1 是**數值變更**，需自帶 M1/M2/M3 參考） |
| `LLAMA_EXPERT_CACHE_WORKERS` | 8 | fill 並行度（唯一 concurrency knob；57363 preads×1.73ms 的綁定項） |
| `CGC_WAKE_POLL_US` | 15 | 固定常數 |
| `CGC_EVICTED_RING` | 0 | prefetch 策略固定（step 默認；hist 已移除，opt-in 用 CGC_PREFETCH_SRC=hist） |
| `CGC_N_CB` | 8 | command buffer 深度（§8.93 cb8 sweet spot） |
| `CGC_SERVER_AUTO_ANCHOR` | 0 | 默認關（oracle 實測 anchor ON→echo loop） |
| `CGC_SERVER_DEFAULT_MARKER_STOPS` | 1 | client 沒給 stop 時補 ChatML marker stops |

> 未帶（prod-new 默認不設）：`CGC_FORCE_TEMP0`、`CGC_DOWN_COMBINE`、`CGC_DC_MULTITOK`、`CGC_HOOK_PROFILE` —— 全是 opt-in（if 條件才加）。
> 校準方法：`CGC_DUMP_ENV=1` 的解析輸出是唯一權威——測試卡任何一行與之不符，以 dump 為準。

> 顯式 `0` 不是「不設」：讓 P0/P1/P2 開關在 profile 裡**有名字**。注意下游白名單只傳非 0 值，`CGC_DUMP_ENV=1` 在關閉時看不到這兩行——那是「關」的正常表現。
>
> 2026-09-25：P0/P1/P2 由「顯式 0＝預設關」翻成「**預設開＋沒開就拒跑**」。理由不是 A/B 結果（那一輪量測本身不可比，見 `docs/P0_P1P2_ATTRIBUTION_2026-09-25.md`），
> 而是預設值本身：超訂 4838 MiB 的根因是那 10.9 GiB 匿名副本，而 8 GiB pool 是 decode 的硬需求（8G 12.06 → 6G 10.48 → 4G 7.67，縮 pool 確定性降速）。
> ⇒ 強制閘（`cgc_swap_guard`）與 `test_run_server_swap_guard.py`（三態＋順序）成對存在；閘在 `CGC_DUMP_ENV` **之後**，否則會把量測入口自己的解析路徑擋死。

## 4. 臂差異開關（A/B 用）

| env | 值 | 效果 |
|---|---|---|
| `CGC_SPAC_HOT=1` | B 真臂 | 踢 `spac_count`（累計路由次數，永不衰減）最低者 → 熱門優先替換（命中 143 槽可覆蓋 98.9% 路由） |
| `CGC_SERVER_MTP=1` | MTP on | 對照 decode 支柱（走 prod25-stream） |
| `CGC_EXPERT_SKIP_READRAW=0`（+`CGC_SWAP_GUARD=off`） | P0 對照 | P0 自 2026-09-25 起是**預設**，所以 A/B 的差異臂是「關掉它」；經 `harness bench` 則寫成 `--arm 'prod-new:!CGC_EXPERT_SKIP_READRAW=0'`（`!` = 已宣告，進產物） |

## 5. 量測紀律（強制，所有測試者含兩個 Agent 與 MainAgent）

### 5.0 每一臂必須**同時測並同時報 prefill + decode**

- llama-bench 完整側天然出兩行：`pp p=`（prefill）與 `tg p=`（decode）。
- **禁止只報 tg**：任何報告 / commit 標題 / 結論必須同時給 `prefill=X / decode=Y`。
- 只報 decode 的讀數視為**不完整產物**，不可引用（活例：2026-09-24 早上的單段 ABBA 只抓了 tg，prefill 行被 grep 丟掉）。

### 5.1 每一個報出的數字必須帶 thermal state + swap state 標注

- 格式（每個 prefill/decode 數字旁必須出現，缺標注 = 不可引用）：

```
prefill=X t/s / decode=Y t/s
thermal: launch=MODERATE worst=HEAVY hist={NOMINAL:n, MODERATE:n, HEAVY:n}
swap:    launch=5712 MiB end=6320 MiB worst=8318 MiB growth=+608 MiB
attribution: thermal / swap / both / none        # memory_pressure.py 判定
```

- **HEAVY 或 swap growth > 500 MiB 的讀數標記為污染**，只能當診斷價，不可進錨點 / commit 標題。

### 5.2 機器內嵌（產物契約）

- matrix 產物 json 已內嵌 `thermal` / `memory` / `attribution` / `rows`（含 pp+tg 兩行）——**不得人肉重打**。
- 引用任何產物時，標注必須**取自該產物欄位**，不許手寫或跨產物借用。

### 5.3 既有規則（保留）

- 任何「慢了 X%」的結論，**先過 thermal/swap 歸因門**再下（memory_pressure.py，selftest 14/14）。
- 交錯協議：A/B 用 ABBA 交錯（A1 B1 A2 B2 A3 B3）+ 配對 per-rep 比率中位，**不允許先跑完 A 再跑 B**。
- 啟動前窗口守門：無殘留 llama-server、swap 水位、thermal NOMINAL、budget 預檢。

## 6. 成績錨點（本卡口徑，llama-bench 完整側）

| 臂 | prefill t/s | decode t/s | hit% | attribution | 來源 |
|---|---:|---:|---:|---:|---|
| ctrl（prod-new 冷機） | 259.91 ± 7.12 | 11.90 ± 0.15 | 96.4 | — | /tmp/abba_p0b_ctrl |
| p0+B 單臂 | 218.53 ± 26.79 | **11.37 ± 0.24** | 96.2 | both（swap+1079 MiB、thermal HEAVY 8） | /tmp/commit_bench_p0hot3 |
| ctrl（同場首臂，髒前） | 162.5 | 10.73 | — | — | commit f2effe569 |

**解讀規則**：
- decode 在同帶內（±0.8 內）即可複現；跨環境直接比絕對值沒有意義。
- prefill 跨時間不可比（環境差 60%），必須同場交錯才有結論。
- 目標：prefill 250+（冷機可達）、decode 25+（現 ~48%，缺 CPU/GPU 重疊 + MoE gather 頻寬）。

## 7. 使用入口

> **唯一對外門（2026-09-25 裁定）**：通用測量走 `harness.py bench`、commit 前走 `commit_bench.py`。
> `llama_bench_matrix.py`／`prod_matrix.py` 是 **internal** 驅動：直跑它們、cell 與 §2.5 不符會被合約
> 當場拒跑；即使跑通，未帶完整 §2.5 口徑標注的數字＝**「口徑不明」**，不可寫進 commit 標題或跨時間比較。

2026-09-25 新增：**cell 與三支柱都有強制閘，兩條路各司其職**。

| 閘 | 在哪 | 沒過會怎樣 | 合法繞道 |
|---|---|---|---|
| cell 合約（`cell_contract.py`） | `llama_bench_matrix.run_arm` 拼完命令、執行前 | 拒跑（fail-closed），逐項列出與 §2.5 的差 | 走 `harness.py bench` / `commit_bench.py` 的預設 |
| swap 支柱（P0/P1/P2＋B_SCHEME） | `run_server.sh cgc_swap_guard`（**在 `CGC_DUMP_ENV` 之後**，否則會擋死量測入口自己的解析路徑） | 拒跑（exit 2） | `CGC_SWAP_GUARD=off`（大聲警告），A/B 對照臂就靠它 |
| thermal 冷卻（軟） | `run_server.sh cgc_box_preflight` / `harness.py bench` | 不拒跑：等 NOMINAL 最多 420 s，逾時只警告（那輪不可引用） | `CGC_THERMAL_GATE=off` / `--no-thermal-gate` |
| swap 佔用者提醒 | 同上（`memory_pressure.py --advice`） | 不拒跑：印出超過起跑門檻的 swap 與佔用它的大戶 | —（提醒是重點，不擋啟動） |

```sh
# 單臂（commit_bench 同款）
python3 scripts/check/llama_bench_matrix.py --arms "prod-new:CGC_SPAC_HOT=1" \
  --prompt 2048 --gen 128 --depths 512 --reps 3 --ctx-size 0 \
  --warm-skip 64 --fixed-fill-seed 1 \
  --workdir /tmp/xxx --json /tmp/xxx/result.json

# A/B 對比（同場）
python3 scripts/check/llama_bench_matrix.py --arms "prod-new,prod-new:CGC_SERVER_MTP=1"

# dry-run 看實際命令
python3 scripts/check/llama_bench_matrix.py --arms prod-new --dry-run
```
