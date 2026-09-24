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
  -b 5632 -ub 5632 -p 2048 -n 128 -d 512 -r 3 -o json
```

| 參數 | 值 | 意義 / 依據 |
|---|---|---|
| `-p 2048` | prefill 2048 token | 完整側 prefill 維 |
| `-n 128` | decode 128 token | 完整側 decode 維 |
| `-d 512` | depth 512 | 已入 context 的 token 數 |
| `-r 3` | 3 reps | 每臂 3 次（中位數口徑） |
| `-b / -ub` | 5632 | 最大存活 chunk（6144 OOM 0/5，見 prefill250 段） |
| `-ngl 99 --load-mode none` | 全層 GPU、不 mmap | |
| `-expert-cache 8589934592` | 8 GiB pool | |

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
| `CGC_EXPERT_SKIP_READRAW` | 0 | P0 開關（**預設關**，A/B 時設 1） |
| `CGC_POOL_MADVISE` | 0 | P1/P2 開關（**預設關**） |
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

## 4. 臂差異開關（A/B 用）

| env | 值 | 效果 |
|---|---|---|
| `CGC_EXPERT_SKIP_READRAW=1` | P0 | skip-load expert 不 read_raw → 省 ~10.9 GiB 匿名駐留、swap ↓92%（5679→449 MiB） |
| `CGC_SPAC_HOT=1` | B 真臂 | 踢 `spac_count`（累計路由次數，永不衰減）最低者 → 熱門優先替換（命中 143 槽可覆蓋 98.9% 路由） |
| `CGC_SERVER_MTP=1` | MTP on | 對照 decode 支柱（走 prod25-stream） |

## 5. 每臂自動附帶的歸因行（不可省略）

```
thermal : Sampler 0.5s 間隔，launch / worst / hist（NOMINAL/MODERATE/HEAVY）
memory  : Sampler launch / end / worst —— swap_used / pages_free / pages_wired / llama_procs
attribution: thermal / swap / contention / both / none
```

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

```sh
# 單臂（commit_bench 同款）
python3 scripts/check/llama_bench_matrix.py --arms "prod-new:CGC_EXPERT_SKIP_READRAW=1;CGC_SPAC_HOT=1" \
  --prompt 2048 --gen 128 --depths 512 --reps 3 --ctx-size 0 \
  --workdir /tmp/xxx --json /tmp/xxx/result.json

# A/B 對比（同場）
python3 scripts/check/llama_bench_matrix.py --arms "prod-new,prod-new:CGC_SERVER_MTP=1"

# dry-run 看實際命令
python3 scripts/check/llama_bench_matrix.py --arms prod-new --dry-run
```
